#!/usr/bin/env python3
"""Run one N72R3 current-frame correction with the pinned official SAM3 backend.

The runner deliberately stops at the event frame.  It freezes the official
pre-intervention output first, then lets the isolated simulated oracle read
only the selected current-frame box.  The backend correction, persistent-ID
mutation, and real 512-D human ROI memory write are one atomic transaction.
No future frame or future GT is loaded by this process.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import sys
import tempfile
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sam3_intermot.association.human_intervention import HumanFeatureExtractor  # noqa: E402
from sam3_intermot.association.state_manager import StateManager, StateManagerConfig  # noqa: E402
from sam3_intermot.backend.sam3_backend import Sam3Backend  # noqa: E402
from sam3_intermot.identity.persistent_runtime import (  # noqa: E402
    PersistentIdentityRecord,
    SequencePersistentIdentityRuntime,
)
from sam3_intermot.interaction.runtime_transactions import (  # noqa: E402
    RuntimeCausalGuard,
    RuntimeInteractionError,
    RuntimeInteractionTransaction,
)
from sam3_intermot.simulation.human_oracle import SimulatedHumanOracle  # noqa: E402


DATA_ROOT = Path("/data1/LWR/vranlee/DATASETS/JDE/dancetrack")
CHECKPOINT = Path(
    "/data1/LWR/vranlee/SERVER_ONLY/avis/Interactive/SAM3_InterMOT/"
    "checkpoints/sam3.1_mirror/sam3.1_multiplex.pt"
)
HUMAN_CHECKPOINT = Path(
    "/data1/LWR/vranlee/SERVER_ONLY/avis/Interactive/SAM3_InterMOT/"
    "outputs/n9/checkpoints/osnet_x1_0_market1501.pth"
)
EVENT_MANIFEST = ROOT / "outputs/N72R3/simulation/real_event_manifest.json"
PLAN_PATH = Path(
    "/data1/LWR/vranlee/SERVER_ONLY/avis/Interactive/SAM3_InterMOT/"
    "outputs/N71/candidate_branch/window_plan.json"
)
OUTPUT_ROOT = ROOT / "outputs/N72R3/official_correction"


def atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        directory_fd = os.open(path.parent, os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def digest_json(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")
    ).hexdigest()


def box_iou(left: Any, right: Any) -> float:
    a = np.asarray(left, dtype=float).reshape(-1)
    b = np.asarray(right, dtype=float).reshape(-1)
    if a.size != 4 or b.size != 4 or not np.isfinite(a).all() or not np.isfinite(b).all():
        return 0.0
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    aa = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    ab = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = aa + ab - inter
    return float(inter / union) if union > 0.0 else 0.0


def observation_view(obs: Any) -> dict[str, Any]:
    mask = np.asarray(obs.mask)
    return {
        "frame_idx": int(obs.frame_idx),
        "sam_object_id": int(obs.sam_object_id),
        "raw_sam_object_id": None if obs.raw_sam_object_id is None else int(obs.raw_sam_object_id),
        "box_xyxy": np.asarray(obs.box_xyxy, dtype=float).reshape(-1).tolist(),
        "box_digest": digest_json(np.asarray(obs.box_xyxy, dtype="<f4").reshape(-1).tolist()),
        "mask_sha256": hashlib.sha256(np.ascontiguousarray(mask).tobytes()).hexdigest(),
        "mask_shape": [int(v) for v in mask.shape],
        "confidence": float(obs.confidence),
        "presence_score": None if obs.presence_score is None else float(obs.presence_score),
        "source": str(obs.source),
        "is_human_verified": bool(obs.is_human_verified),
    }


def load_selected_event(event_id: str) -> dict[str, Any]:
    payload = json.loads(EVENT_MANIFEST.read_text(encoding="utf-8"))
    for event in payload.get("events", []):
        if str(event.get("event_id")) == str(event_id):
            return dict(event)
    raise KeyError(f"selected N72R3 event not found: {event_id}")


def load_window(event: dict[str, Any]) -> dict[str, Any]:
    window_id = str(event["current_candidate_v2"]["window_id"])
    plan = json.loads(PLAN_PATH.read_text(encoding="utf-8"))
    for item in plan.get("windows", []):
        if str(item.get("window_id")) == window_id:
            return dict(item)
    raise KeyError(f"frozen N71 window not found: {window_id}")


def make_backend() -> Sam3Backend:
    return Sam3Backend(
        checkpoint_path=str(CHECKPOINT),
        # The frozen sam3.1_multiplex checkpoint has 16 object slots.  Keep
        # the live runner aligned with the verified N72R2/N36 configuration;
        # 32/32 is not a valid capacity for this checkpoint.
        max_num_objects=16,
        multiplex_count=16,
        use_fa3=False,
        use_rope_real=True,
        compile=False,
        warm_up=False,
        session_expiration_sec=1200,
        output_prob_thresh=0.30,
        async_loading_frames=False,
        device="cuda",
    )


def collect_pre_event(backend: Sam3Backend, window: dict[str, Any], event_frame: int) -> list[Any]:
    frame_start = int(window["frame_start"])
    if event_frame < frame_start or event_frame > int(window["frame_end"]):
        raise ValueError(f"event frame {event_frame} is outside frozen window {frame_start}:{window['frame_end']}")
    initial = backend.detect_concept(frame_start, "person")
    if event_frame == frame_start:
        return [obs.copy() for obs in initial]
    backend.propagate(
        frame_start,
        event_frame,
        start_frame_index=frame_start,
        keep_masks=True,
        cache_outputs=True,
    )
    observations = backend.get_frame_outputs(event_frame)
    if not observations:
        raise RuntimeError(f"official backend returned no Y_pre observations at event frame {event_frame}")
    return [obs.copy() for obs in observations]


def create_runtime(
    sequence: str,
    session_id: str,
    frame: int,
    observations: list[Any],
) -> tuple[SequencePersistentIdentityRuntime, StateManager, dict[int, PersistentIdentityRecord], dict[int, str]]:
    runtime = SequencePersistentIdentityRuntime(sequence, public_id_start=1000)
    state_manager = StateManager(
        StateManagerConfig(
            external_identity_authority=True,
            use_appearance_memory=False,
            score_threshold=-100.0,
        )
    )
    records: dict[int, PersistentIdentityRecord] = {}
    candidate_uids: dict[int, str] = {}
    for index, obs in enumerate(observations):
        sam_id = int(obs.sam_object_id)
        if sam_id in records:
            raise RuntimeError(f"duplicate official SAM object in Y_pre: {sam_id}")
        uid = f"n72r3-live:{sequence}:frame:{frame}:candidate:{index}:sam:{sam_id}"
        record = runtime.create_identity(
            frame,
            obs,
            candidate_uid=uid,
            session_id=session_id,
            adapter_external_id=sam_id,
            raw_sam_id=obs.raw_sam_object_id,
        )
        records[sam_id] = record
        candidate_uids[sam_id] = uid
        state_manager.register_from_persistent_identity(
            record,
            {
                "feat": np.zeros(512, dtype=np.float32),
                "box": np.asarray(obs.box_xyxy, dtype=float).tolist(),
                "native_tid": sam_id,
            },
            frame,
        )
    if not records:
        raise RuntimeError("Y_pre has no candidates to establish an outer runtime")
    return runtime, state_manager, records, candidate_uids


def prepare_simulated_assignment(
    event: dict[str, Any],
    frame: int,
    session_id: str,
    runtime: SequencePersistentIdentityRuntime,
    records: dict[int, PersistentIdentityRecord],
    candidate_uids: dict[int, str],
    observations: list[Any],
) -> tuple[
    PersistentIdentityRecord,
    Any,
    dict[str, Any],
    list[dict[str, Any]],
    PersistentIdentityRecord | None,
]:
    human_box = np.asarray(event["current_gt_box"], dtype=float).reshape(-1)
    target_obs = max(observations, key=lambda obs: (box_iou(obs.box_xyxy, human_box), -int(obs.sam_object_id)))
    target_sam = int(target_obs.sam_object_id)
    action = str(event["action_type"])
    target_record = records[target_sam]
    wrong_record: PersistentIdentityRecord | None = None

    # This is the current-frame simulated-human assignment setup.  Public IDs
    # come from the outer runtime only; the dataset GT ID is kept inside the
    # isolated oracle and is never sent to StateManager, SAM3, or memory.
    runtime.clear_current_session_bindings(frame, reason="simulated_oracle_current_frame_assignment_setup")
    if action == "AUTHORITATIVE_REASSIGN":
        alternatives = [record for sam, record in sorted(records.items()) if sam != target_sam]
        if not alternatives:
            raise RuntimeError("AUTHORITATIVE_REASSIGN requires at least two current persistent identities")
        target_record = alternatives[0]
        wrong_record = records[target_sam]
        runtime.mark_lost(target_record, frame - 1, reason="simulated_current_assignment_before_reassign")
        for sam, record in sorted(records.items()):
            if record is target_record or record is wrong_record:
                continue
            obs = next(item for item in observations if int(item.sam_object_id) == sam)
            runtime.bind_candidate(record, candidate_uids[sam], obs, frame, session_id=session_id)
        runtime.bind_candidate(wrong_record, candidate_uids[target_sam], target_obs, frame, session_id=session_id)
    elif action == "RECOVER_IDENTITY":
        runtime.mark_lost(target_record, frame - 1, reason="simulated_current_assignment_before_recover")
        for sam, record in sorted(records.items()):
            if record is target_record:
                continue
            obs = next(item for item in observations if int(item.sam_object_id) == sam)
            runtime.bind_candidate(record, candidate_uids[sam], obs, frame, session_id=session_id)
    else:
        for sam, record in sorted(records.items()):
            obs = next(item for item in observations if int(item.sam_object_id) == sam)
            runtime.bind_candidate(record, candidate_uids[sam], obs, frame, session_id=session_id)

    predictions: list[dict[str, Any]] = []
    for obs in observations:
        sam = int(obs.sam_object_id)
        record = next(
            (item for item in records.values() if item.current_raw_sam_id == (obs.raw_sam_object_id if obs.raw_sam_object_id is not None else sam)),
            None,
        )
        if record is None:
            # Stable adapter IDs can be exposed in place of raw IDs; resolve
            # the current TrackManager binding without treating it as authority.
            track_id = runtime.manager._sam_to_track.get(sam)
            if track_id is not None:
                record = next((item for item in records.values() if item.mot_track_id == int(track_id)), None)
        if record is None or record.current_session_id is None:
            continue
        predictions.append(
            {
                "candidate_uid": candidate_uids[sam],
                "public_id": int(record.public_id),
                "box": np.asarray(obs.box_xyxy, dtype=float).tolist(),
            }
        )
    oracle = SimulatedHumanOracle(
        str(event["sequence"]),
        known_gt_to_public={int(event["dataset_gt_id"]): int(target_record.public_id)},
    )
    decisions = oracle.choose_actions(
        frame,
        {"boxes": [human_box.tolist()], "gt_ids": [int(event["dataset_gt_id"])]},
        predictions,
        localization_iou_threshold=0.5,
    )
    return target_record, target_obs, {
        "target_public_id": int(target_record.public_id),
        "target_sam_object_id": target_sam,
        "wrong_public_id": None if wrong_record is None else int(wrong_record.public_id),
        "oracle": oracle.as_dict(),
        "oracle_decisions": [decision.as_dict() for decision in decisions],
        "prediction_count": len(predictions),
        "target_pre_iou": box_iou(target_obs.box_xyxy, human_box),
        "assignment_setup": action,
    }, predictions, wrong_record


def run_one(event: dict[str, Any], gpu: int) -> dict[str, Any]:
    sequence = str(event["sequence"])
    frame = int(event["event_frame"])
    event_id = str(event["event_id"])
    window = load_window(event)
    sequence_dir = DATA_ROOT / "train" / sequence
    if not sequence_dir.is_dir():
        raise FileNotFoundError(sequence_dir)
    if not CHECKPOINT.is_file():
        raise FileNotFoundError(CHECKPOINT)
    if not HUMAN_CHECKPOINT.is_file():
        raise FileNotFoundError(HUMAN_CHECKPOINT)

    backend: Sam3Backend | None = None
    extractor: HumanFeatureExtractor | None = None
    started = time.time()
    try:
        backend = make_backend()
        session_id = backend.start_video(str(sequence_dir / "img1"))
        pre = collect_pre_event(backend, window, frame)
        pre_frozen = [observation_view(obs) for obs in pre]
        registered_candidates = []
        for obs in pre:
            registered_candidates.append(
                {
                    "sam_object_id": int(obs.sam_object_id),
                    "raw_sam_object_id": (
                        None
                        if obs.raw_sam_object_id is None
                        else int(obs.raw_sam_object_id)
                    ),
                    "frame_idx": int(obs.frame_idx),
                    "adapter_external_id": int(
                        backend.register_detected_observation(obs)
                    ),
                }
            )
        runtime, state_manager, records, candidate_uids = create_runtime(sequence, session_id, frame, pre)
        target_record, target_obs, setup_audit, predictions, wrong_record = prepare_simulated_assignment(
            event, frame, session_id, runtime, records, candidate_uids, pre
        )
        human_box = np.asarray(event["current_gt_box"], dtype=float).reshape(-1)
        extractor = HumanFeatureExtractor(HUMAN_CHECKPOINT)
        causal = RuntimeCausalGuard(event_id, str(event["action_type"]), frame, session_id)
        holder: dict[str, Any] = {}
        action = str(event["action_type"])
        prompt_target_id = int(target_obs.sam_object_id)

        def backend_step():
            if action == "RECOVER_IDENTITY":
                existing = [int(key) for key in getattr(backend, "_objects", {}).keys()]
                prompt_target = (max(existing) + 1000) if existing else 1000
                holder["prompt_target_id"] = prompt_target
                holder["observation"] = backend.add_box(frame, prompt_target, human_box)
                route = "perform_recover_identity/official_backend.add_box"
            else:
                holder["prompt_target_id"] = prompt_target_id
                holder["observation"] = backend.correct_object(frame, prompt_target_id, box_xyxy=human_box)
                route = "perform_correct/official_backend.correct_object"
            obs = holder["observation"]
            if obs is None or not np.all(np.isfinite(np.asarray(obs.box_xyxy, dtype=float))):
                raise RuntimeInteractionError("official backend returned invalid current correction observation")
            fallback_entries = getattr(backend, "_prompt_fallback_log", [])[0:]
            no_output = any(
                int(item.get("object_id", -1)) == int(holder["prompt_target_id"]) and bool(item.get("no_sam_output"))
                for item in fallback_entries
            )
            if no_output:
                raise RuntimeInteractionError("official backend prompt produced no SAM observation after registered fallback attempts")
            holder["route"] = route
            holder["official_prompt_observation_available"] = True
            holder["observation_view"] = observation_view(obs)
            causal.record_spatial_correction(
                frame,
                backend_prompt_route=route,
                correction_id=f"{event_id}:spatial",
                official_backend=True,
                prompt_object_id=int(holder["prompt_target_id"]),
                official_prompt_observation_available=True,
            )
            return obs

        def identity_step():
            obs = holder["observation"]
            raw = obs.raw_sam_object_id
            if raw is None:
                raw = int(obs.sam_object_id)
            if action == "AUTHORITATIVE_REASSIGN" and wrong_record is not None:
                # The simulated pre-state deliberately has the target
                # candidate bound to the wrong persistent identity.  Release
                # that session-local binding inside the same transaction
                # before assigning the corrected observation to the public
                # identity supplied by the isolated oracle.
                runtime.unbind_session_candidate(
                    wrong_record,
                    frame,
                    reason="authoritative_reassign_release_wrong_current_binding",
                )
            record = runtime.bind_candidate(
                target_record,
                f"{event_id}:corrected_candidate",
                obs,
                frame,
                session_id=session_id,
                adapter_external_id=int(obs.sam_object_id),
                raw_sam_id=int(raw),
            )
            record.appearance_state.update(
                {
                    "last_event_id": event_id,
                    "last_write_frame": frame,
                    "write_source": "current_frame_simulated_human_roi",
                }
            )
            return record.as_dict()

        def memory_step():
            feature = extractor.extract(sequence_dir / "img1", frame, human_box)
            feature = np.asarray(feature, dtype=np.float32).reshape(-1)
            if feature.size != 512 or not np.all(np.isfinite(feature)):
                raise RuntimeInteractionError(f"human ROI feature invalid: {feature.shape}")
            norm = float(np.linalg.norm(feature))
            if norm <= 1e-6:
                raise RuntimeInteractionError("human ROI feature has zero norm")
            feature = feature / norm
            feature_sha = hashlib.sha256(feature.astype("<f4", copy=False).tobytes()).hexdigest()
            ok = runtime.appearance_memory.update_from_human(
                int(target_record.public_id),
                frame,
                feature,
                quality=1.0,
                write_event_id=event_id,
            )
            if not ok:
                raise RuntimeInteractionError("persistent appearance memory rejected real ROI feature")
            target_record.appearance_state.update(
                {
                    "feature_sha256": feature_sha,
                    "feature_dim": 512,
                    "feature_norm": norm,
                    "encoder_checkpoint_sha256": sha256(HUMAN_CHECKPOINT),
                    "quality": 1.0,
                    "visible_from_frame": frame + 1,
                    "current_frame_write_hidden": True,
                }
            )
            causal.write_memory(
                frame,
                memory_key=f"public:{target_record.public_id}",
                feature_sha256=feature_sha,
                source="current_frame_simulated_human_box_roi",
                feature_dim=512,
                encoder_checkpoint_sha256=sha256(HUMAN_CHECKPOINT),
                quality=1.0,
            )
            return {
                "public_id": int(target_record.public_id),
                "feature_sha256": feature_sha,
                "feature_dim": 512,
                "feature_norm": norm,
                "source": "current_frame_simulated_human_box_roi",
                "write_frame": frame,
                "first_visible_frame": frame + 1,
                "current_frame_write_hidden": True,
            }

        transaction = RuntimeInteractionTransaction(
            backend=backend,
            persistent_runtime=runtime,
            state_manager=state_manager,
            event_id=event_id,
        )
        transaction_result = transaction.execute(backend_step, identity_step, memory_step)
        # Stage 16/17 intentionally stops before any future inference.  Record
        # the declared event+1 boundary without fabricating a memory read;
        # the actual first read is audited later by the paired replay stage.
        causal.record_future_frame(
            frame + 1,
            runtime_future_gt_used=False,
            boundary_only=True,
            future_frame_loaded=False,
            memory_read_not_executed=True,
        )
        causal_audit = causal.finalize()
        final_runtime = runtime.audit()
        post = holder["observation"]
        post_iou = box_iou(post.box_xyxy, human_box)
        result = {
            "schema_version": "N72R3_STAGE16_17_OFFICIAL_CORRECTION_MEMORY_V1",
            "status": "PASS_STAGE16_OFFICIAL_CORRECTION_AND_STAGE17_MEMORY",
            "event_id": event_id,
            "sequence": sequence,
            "event_frame": frame,
            "action_type": action,
            "interaction_source": "simulated_from_gt",
            "not_real_human_evidence": True,
            "official_backend": True,
            "checkpoint_sha256": sha256(CHECKPOINT),
            "human_encoder_checkpoint_sha256": sha256(HUMAN_CHECKPOINT),
            "frozen_window": {
                "window_id": str(window["window_id"]),
                "frame_start": int(window["frame_start"]),
                "frame_end": int(window["frame_end"]),
            },
            "Y_pre_frozen": pre_frozen,
            "Y_pre_candidate_count": len(pre),
            "official_candidate_registration": {
                "count": len(registered_candidates),
                "records": registered_candidates,
                "source": "explicit_adapter_observation_to_prompt_registry_bridge",
                "public_identity_inference": False,
            },
            "simulated_assignment_setup": setup_audit,
            "simulated_predictions": predictions,
            "official_current_correction": {
                "route": holder["route"],
                "prompt_target_id": int(holder["prompt_target_id"]),
                "official_prompt_observation_available": bool(holder["official_prompt_observation_available"]),
                "post_observation": observation_view(post),
                "human_box": human_box.tolist(),
                "post_box_iou_to_simulated_human_box": post_iou,
                "current_output_semantics": "official_adapter_human_verified_box_at_event_frame",
                "success": bool(post_iou >= 0.98),
            },
            "persistent_identity": {
                "public_id": int(target_record.public_id),
                "mot_track_id": int(target_record.mot_track_id),
                "lineage_id": int(target_record.identity_lineage_id),
                "association_state_id": int(target_record.association_state_id),
                "status": target_record.status,
                "current_session_id": target_record.current_session_id,
                "candidate_uid": target_record.last_candidate_uid,
            },
            "transaction": transaction_result,
            "causal_audit": causal_audit,
            "appearance_memory": runtime.appearance_memory.serialize()["records"].get(str(target_record.public_id), {}),
            "runtime_audit": final_runtime,
            "runtime_future_gt_used": False,
            "future_frames_loaded": 0,
            "future_read_executed": False,
            "scientific_result": "CURRENT_FRAME_AND_MEMORY_ARTIFACT_ONLY_NOT_FUTURE_EFFECT",
            "elapsed_sec": time.time() - started,
        }
        if not result["official_current_correction"]["success"]:
            raise RuntimeInteractionError(f"current correction gate failed: IoU={post_iou}")
        return result
    finally:
        if backend is not None:
            try:
                backend.close()
            except Exception:
                pass
        extractor = None
        backend = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--event-id", required=True)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    output = args.output.resolve() if args.output is not None else OUTPUT_ROOT / "events" / f"{args.event_id}.json"
    started = datetime_now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    try:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable for official Stage16/17 runner")
        visible = os.environ.get("CUDA_VISIBLE_DEVICES")
        torch.cuda.set_device(0 if visible else int(args.gpu))
        event = load_selected_event(args.event_id)
        result = run_one(event, int(args.gpu))
        result["gpu"] = int(args.gpu)
        result["started_at_utc"] = started
        atomic_json(output, result)
        print(json.dumps({"status": result["status"], "event_id": args.event_id, "output": str(output)}, sort_keys=True))
        return 0
    except Exception as exc:
        failure = {
            "schema_version": "N72R3_FAILURE_RECORD_V1",
            "stage": "16_OFFICIAL_CORRECTION_17_REAL_MEMORY",
            "status": "FAIL_STAGE16_17_OFFICIAL_BACKEND",
            "event_id": str(args.event_id),
            "gpu": int(args.gpu),
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
            "runtime_future_gt_used": False,
            "scientific_result": "NO_SCIENTIFIC_RESULT",
            "started_at_utc": started,
            "repair_policy": "preserve failure; targeted official-backend smoke before any batch retry",
        }
        atomic_json(output, failure)
        print(json.dumps({"status": failure["status"], "event_id": args.event_id, "error": failure["error"], "output": str(output)}, sort_keys=True))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
