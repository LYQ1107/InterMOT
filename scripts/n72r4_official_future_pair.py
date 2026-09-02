#!/usr/bin/env python3
"""Paired official-SAM3 future propagation for the frozen N72R3 events.

This runner is deliberately separate from the frozen N72R3 current-frame
artifact and from the N72R4 CPU structural probe.  A worker runs exactly one
event and one official SAM3 branch in one fresh Python process:

* ``B0_NO_INTERVENTION``: the frozen prefix reaches ``Y_pre(t)`` and the
  unmodified official state propagates through ``t+100``;
* ``B1_CURRENT_FRAME_CORRECTION``: the same prefix reaches ``Y_pre(t)``, the
  already-frozen current-frame simulated command is applied through the
  official adapter, and that corrected SAM3 state propagates through
  ``t+100``.

The parent process runs the two workers sequentially on one selected GPU and
checks the pre-event equivalence gate.  The branch artifacts contain only
official observations and machine ROI features; public-ID association is a
later CPU stage and never inferred from a native/adapter number.

This is GT-controlled exploratory evidence.  The current event box is read
only after the branch has frozen its ``Y_pre`` observation.  Future GT is
never loaded by this process.
"""

from __future__ import annotations

import argparse
import base64
import gc
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import time
import traceback
import zlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sam3_intermot.backend.sam3_backend import Sam3Backend  # noqa: E402
from scripts.run_n35_export_tape import FrozenMachineOSNet, image_files  # noqa: E402


FROZEN_N72R3_ROOT = Path("/data2/usr_for_deadline/SAM3_InterMOT_N72R3/worktree/outputs/N72R3")
EVENT_MANIFEST = FROZEN_N72R3_ROOT / "simulation/real_event_manifest.json"
PLAN_PATH = Path(
    "/data1/LWR/vranlee/SERVER_ONLY/avis/Interactive/SAM3_InterMOT/"
    "outputs/N71/candidate_branch/window_plan.json"
)
DATA_ROOT = Path("/data1/LWR/vranlee/DATASETS/JDE/dancetrack")
CHECKPOINT = Path(
    "/data1/LWR/vranlee/SERVER_ONLY/avis/Interactive/SAM3_InterMOT/"
    "checkpoints/sam3.1_mirror/sam3.1_multiplex.pt"
)
DEFAULT_NO_ROOT = ROOT / "outputs/N72R4/official_no_intervention"
DEFAULT_CORRECTED_ROOT = ROOT / "outputs/N72R4/official_corrected"
DEFAULT_ATTEMPT_ROOT = ROOT / "outputs/N72R4/attempts/stage09"
STAGE_ROOT = ROOT / "outputs/N72R4"

BRANCH_NO = "B0_NO_INTERVENTION"
BRANCH_CORRECTED = "B1_CURRENT_FRAME_CORRECTION"
BRANCHES = (BRANCH_NO, BRANCH_CORRECTED)
HORIZON = 100
# The official detector uses an exclusive valid-frame bound while the
# multiplex propagation order is inclusive.  A fixed 200-frame request keeps
# enough detector context around the requested prefix/future range without
# making the runner consume frames beyond its explicit stopping boundary.
OFFICIAL_CONTEXT_WINDOW = 200


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


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


def atomic_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n")
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


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object: {path}")
    return value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def json_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")
    ).hexdigest()


def jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): jsonable(child) for key, child in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(child) for child in value]
    return value


def encode_mask(mask: Any) -> dict[str, Any] | None:
    if mask is None:
        return None
    array = np.asarray(mask, dtype=bool)
    packed = np.packbits(array.reshape(-1), bitorder="little")
    return {
        "encoding": "packbits_zlib_base64",
        "shape": [int(value) for value in array.shape],
        "bitorder": "little",
        "data": base64.b64encode(zlib.compress(packed.tobytes(), level=1)).decode("ascii"),
        "sha256": hashlib.sha256(array.tobytes()).hexdigest(),
    }


def feature_hash(feature: Any) -> str:
    array = np.asarray(feature, dtype="<f4").reshape(-1)
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def observation_view(observation: Any) -> dict[str, Any]:
    mask = np.asarray(observation.mask, dtype=bool)
    return {
        "frame_idx": int(observation.frame_idx),
        "sam_object_id": int(observation.sam_object_id),
        "raw_sam_object_id": None
        if observation.raw_sam_object_id is None
        else int(observation.raw_sam_object_id),
        "box_xyxy": np.asarray(observation.box_xyxy, dtype=float).reshape(-1).tolist(),
        "mask_sha256": hashlib.sha256(np.ascontiguousarray(mask).tobytes()).hexdigest(),
        "mask_shape": [int(value) for value in mask.shape],
        "confidence": float(observation.confidence),
        "presence_score": None
        if observation.presence_score is None
        else float(observation.presence_score),
        "source": str(observation.source),
        "is_human_verified": bool(observation.is_human_verified),
    }


def semantic_pre_view(observations: list[Any]) -> list[dict[str, Any]]:
    return [observation_view(observation) for observation in observations]


def official_cached_observations(backend: Sam3Backend, frame: int) -> list[Any]:
    """Return only official SAM3 observations from an adapter frame cache.

    ``Sam3Backend.correct_object`` returns a human-verified observation for
    the current frame and appends that observation to the adapter cache.  It
    is deliberately useful for the interaction ledger, but it is not an
    official candidate and must not be exported as one.  Filtering on the
    explicit provenance fields avoids counting the same adapter-visible ID
    twice when the official response already contains the corrected object.
    """

    official = backend.get_last_official_prompt_outputs(int(frame))
    if not official:
        raise RuntimeError(f"official SAM3 cache has no non-human observations at frame {frame}")
    return official


def load_events() -> list[dict[str, Any]]:
    payload = read_json(EVENT_MANIFEST)
    events = [dict(item) for item in payload.get("events", [])]
    if len(events) != 6:
        raise RuntimeError(f"frozen N72R3 event count changed: expected 6, found {len(events)}")
    ids = [str(item["event_id"]) for item in events]
    if len(ids) != len(set(ids)):
        raise RuntimeError("frozen N72R3 event IDs are not unique")
    for event in events:
        if event.get("runtime_future_gt_used") is True:
            raise RuntimeError(f"frozen event permits runtime future GT: {event['event_id']}")
    return sorted(events, key=lambda item: str(item["event_id"]))


def load_event(event_id: str) -> dict[str, Any]:
    for event in load_events():
        if str(event["event_id"]) == str(event_id):
            return event
    raise KeyError(f"frozen event not found: {event_id}")


def load_window(event: dict[str, Any]) -> dict[str, Any]:
    window_id = str(event["current_candidate_v2"]["window_id"])
    plan = read_json(PLAN_PATH)
    matches = [dict(item) for item in plan.get("windows", []) if str(item.get("window_id")) == window_id]
    if len(matches) != 1:
        raise RuntimeError(f"frozen window is not unique: {window_id}")
    return matches[0]


def make_backend() -> Sam3Backend:
    return Sam3Backend(
        checkpoint_path=str(CHECKPOINT),
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


def install_official_shape_audit(backend: Sam3Backend) -> None:
    """Attach a non-invasive runtime audit to the pinned official model.

    The split-propagation failure is raised inside the read-only third-party
    implementation, before the adapter receives a frame result.  Wrapping the
    two official boundary methods on this *worker instance* records the exact
    frame and ID/mask axes without changing the official source or its
    algorithm.  The audit is failure provenance only; it is never used to
    alter candidates or metrics.
    """
    model = getattr(getattr(backend, "_predictor", None), "model", None)
    if model is None:
        raise RuntimeError("official model is unavailable for shape audit")
    audit: list[dict[str, Any]] = []

    original_propagate = model._propogate_tracker_one_frame_local_gpu

    def audited_propagate(
        inference_states: list[Any],
        frame_idx: int,
        reverse: bool,
        run_mem_encoder: bool = False,
        filter_obj_ids: list[int] | None = None,
    ):
        before_states = [
            [int(value) for value in state.get("obj_ids", [])]
            for state in inference_states
        ]
        try:
            result = original_propagate(
                inference_states,
                frame_idx=frame_idx,
                reverse=reverse,
                run_mem_encoder=run_mem_encoder,
                filter_obj_ids=filter_obj_ids,
            )
        except Exception as exc:
            audit.append(
                {
                    "phase": "official_local_tracker_propagation",
                    "frame_idx": int(frame_idx),
                    "state_ids_before": before_states,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
            raise
        obj_ids, masks, scores = result
        audit.append(
            {
                "phase": "official_local_tracker_propagation",
                "frame_idx": int(frame_idx),
                "state_ids_before": before_states,
                "returned_ids": [int(value) for value in obj_ids],
                "returned_mask_count": int(masks.shape[0]),
                "returned_score_count": int(scores.shape[0]),
            }
        )
        return result

    original_associate = model._associate_det_trk

    def audited_associate(
        det_masks: Any,
        det_scores: Any,
        det_keep: Any,
        trk_masks: Any,
        trk_obj_ids: Any,
        default_det_thresh: Any = None,
    ):
        try:
            return original_associate(
                det_masks=det_masks,
                det_scores=det_scores,
                det_keep=det_keep,
                trk_masks=trk_masks,
                trk_obj_ids=trk_obj_ids,
                default_det_thresh=default_det_thresh,
            )
        except Exception as exc:
            audit.append(
                {
                    "phase": "official_detection_track_association",
                    "trk_mask_count": int(trk_masks.shape[0]),
                    "trk_id_count": len(trk_obj_ids),
                    "trk_obj_ids": [int(value) for value in trk_obj_ids],
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
            raise

    tracker = getattr(model, "tracker", None)
    original_tracker_propagate = getattr(tracker, "propagate_in_video", None)

    def audited_tracker_propagate(*args: Any, **kwargs: Any):
        if original_tracker_propagate is None:
            return
        for output in original_tracker_propagate(*args, **kwargs):
            try:
                output_ids = output[1]
                output_masks = output[2]
                id_values = [int(value) for value in output_ids]
                mask_count = int(output_masks.shape[0])
                if len(id_values) != mask_count:
                    audit.append(
                        {
                            "phase": "official_tracker_raw_output",
                            "frame_idx": int(output[0]),
                            "raw_output_ids": id_values,
                            "raw_output_mask_count": mask_count,
                            "raw_output_score_count": int(output[4].shape[0]),
                            "raw_output_box_count": int(output[3].shape[0]),
                        }
                    )
            except Exception as exc:
                audit.append(
                    {
                        "phase": "official_tracker_raw_output_audit_error",
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
            yield output

    model._propogate_tracker_one_frame_local_gpu = audited_propagate
    model._associate_det_trk = audited_associate
    if tracker is not None and original_tracker_propagate is not None:
        tracker.propagate_in_video = audited_tracker_propagate
    model._n72r4_shape_audit = audit


def collect_pre_event(backend: Sam3Backend, window: dict[str, Any], event_frame: int) -> list[Any]:
    frame_start = int(window["frame_start"])
    frame_end = int(window["frame_end"])
    if not frame_start <= event_frame <= frame_end:
        raise ValueError(f"event frame is outside frozen window: {event_frame} not in {frame_start}:{frame_end}")
    initial = backend.detect_concept(frame_start, "person")
    if event_frame == frame_start:
        return [observation.copy() for observation in initial]
    backend.propagate(
        frame_start,
        event_frame,
        start_frame_index=frame_start,
        max_frame_num_to_track=OFFICIAL_CONTEXT_WINDOW,
        keep_masks=True,
        cache_outputs=True,
    )
    observations = backend.get_frame_outputs(event_frame)
    if not observations:
        raise RuntimeError(f"official backend returned empty Y_pre: frame={event_frame}")
    return [observation.copy() for observation in observations]


def collect_continuous_baseline(
    backend: Sam3Backend,
    window: dict[str, Any],
    event_frame: int,
    frame_end: int,
) -> tuple[list[Any], dict[int, list[Any]]]:
    """Run the untouched official baseline through H100 in one stream.

    The prefix worker historically stopped and closed the official generator
    at ``event_frame`` and then reopened it at the same frame.  The pinned
    multiplex implementation can diverge at that artificial boundary when a
    detector hot-start adds/removes a tracklet.  The no-intervention baseline
    has no action to insert, so its causal baseline is exactly one continuous
    official propagation stream from the frozen window start through H100.
    The returned event-frame view is taken only after the stream completes;
    no GT is read and no state/output is synthesized.
    """
    frame_start = int(window["frame_start"])
    initial = backend.detect_concept(frame_start, "person")
    # An empty detector hot-start is an admissible official state for this
    # frozen window.  The multiplex predictor may introduce tracklets during
    # its own continuous propagation; rejecting the state here would turn a
    # valid official stream into an adapter-only failure.  Do not synthesize
    # observations: the exact stream below remains the source of truth.
    outputs = backend.propagate(
        frame_start,
        frame_end,
        start_frame_index=frame_start,
        max_frame_num_to_track=OFFICIAL_CONTEXT_WINDOW,
        keep_masks=True,
        cache_outputs=True,
    )
    expected = set(range(frame_start, int(frame_end) + 1))
    observed = {int(frame) for frame in outputs}
    if observed != expected:
        missing = sorted(expected - observed)
        extra = sorted(observed - expected)
        raise RuntimeError(
            f"continuous official baseline coverage mismatch: missing={missing[:8]} extra={extra[:8]}"
        )
    pre = outputs.get(int(event_frame), [])
    if not pre:
        raise RuntimeError(f"continuous official baseline has empty Y_pre: frame={event_frame}")
    return [observation.copy() for observation in pre], {
        int(frame): [observation.copy() for observation in observations]
        for frame, observations in outputs.items()
    }


def correction_target(event: dict[str, Any]) -> tuple[str, int, np.ndarray]:
    event_id = str(event["event_id"])
    official = read_json(FROZEN_N72R3_ROOT / f"official_correction/events/{event_id}.json")
    if official.get("status") != "PASS_STAGE16_OFFICIAL_CORRECTION_AND_STAGE17_MEMORY":
        raise RuntimeError(f"frozen official current correction is not PASS: {event_id}")
    setup = official.get("simulated_assignment_setup") or {}
    if "target_sam_object_id" not in setup:
        raise RuntimeError(f"frozen correction target is missing: {event_id}")
    action = str(event["action_type"])
    human_box = np.asarray(event["current_gt_box"], dtype=float).reshape(-1)
    if human_box.size != 4 or not np.all(np.isfinite(human_box)):
        raise RuntimeError(f"current simulated human box is invalid: {event_id}")
    return action, int(setup["target_sam_object_id"]), human_box


def current_candidate_rows(
    backend: Sam3Backend,
    encoder: FrozenMachineOSNet,
    image_path: Path,
    frame: int,
    observations: list[Any],
) -> list[dict[str, Any]]:
    boxes = [np.asarray(item.box_xyxy, dtype=float).copy() for item in observations]
    features = encoder.encode(image_path, boxes)
    if features.shape != (len(observations), 512) or not np.all(np.isfinite(features)):
        raise RuntimeError(f"official candidate features are invalid at frame {frame}: {features.shape}")
    backend._output_cache[int(frame)] = [item.copy() for item in observations]
    try:
        exported = backend.export_frame_candidates(
            int(frame), embeddings=features, include_masks=True, include_raw_provenance=True
        )
    finally:
        backend._output_cache.pop(int(frame), None)
    if len(exported) != len(observations):
        raise RuntimeError(f"official candidate exporter changed count at frame {frame}")
    rows: list[dict[str, Any]] = []
    seen_native: set[int] = set()
    seen_raw: set[int] = set()
    mapping_debug = {
        "observation_visible_ids": [int(item.sam_object_id) for item in observations],
        "observation_raw_ids": [
            None if item.raw_sam_object_id is None else int(item.raw_sam_object_id)
            for item in observations
        ],
        "backend_ext_to_sam": {
            str(key): int(value) for key, value in getattr(backend, "_ext_to_sam", {}).items()
        },
        "backend_sam_to_ext": {
            str(key): int(value) for key, value in getattr(backend, "_sam_to_ext", {}).items()
        },
    }
    for index, candidate in enumerate(exported):
        native = int(candidate["native_tid"])
        raw = candidate.get("raw_native_id")
        raw_id = None if raw is None else int(raw)
        # The adapter-visible ID is allowed to collide with an unbound raw
        # ID after an official prompt rebind (e.g. visible [6, 1, 5, 4, 1, 0]
        # for raw [0, 1, 2, 3, 4, 5]).  Candidate identity in this official
        # branch must therefore use the immutable raw SAM axis whenever it is
        # present.  The adapter ID remains provenance only; neither axis is a
        # public identity.
        candidate_native = raw_id if raw_id is not None else native
        if candidate_native in seen_native:
            raise RuntimeError(
                f"duplicate official candidate axis at frame {frame}: {candidate_native}; "
                f"mapping_debug={json.dumps(mapping_debug, sort_keys=True)}"
            )
        if raw_id is not None and raw_id in seen_raw:
            raise RuntimeError(
                f"duplicate official raw candidate ID at frame {frame}: {raw_id}; "
                f"mapping_debug={json.dumps(mapping_debug, sort_keys=True)}"
            )
        seen_native.add(candidate_native)
        if raw_id is not None:
            seen_raw.add(raw_id)
        vector = np.asarray(candidate.get("embedding"), dtype=np.float32).reshape(-1)
        if vector.shape != (512,) or not np.all(np.isfinite(vector)) or float(np.linalg.norm(vector)) <= 1.0e-6:
            raise RuntimeError(f"invalid machine ROI feature at frame {frame}, candidate {index}")
        vector = vector / float(np.linalg.norm(vector))
        rows.append(
            {
                "candidate_index": int(index),
                "adapter_external_id": native,
                "adapter_visible_id": native,
                "native_tid": candidate_native,
                "official_raw_sam_id": raw_id,
                "raw_native_id": raw_id,
                "native_id_source": (
                    "official_raw_sam_id_for_branch_candidate_axis"
                    if raw_id is not None
                    else "adapter_visible_stable_id_fallback_no_raw_id"
                ),
                "raw_native_id_source": "official_out_obj_ids",
                "box_xyxy": np.asarray(candidate["box_xyxy"], dtype=float).reshape(4).tolist(),
                "mask": encode_mask(candidate.get("mask")),
                "confidence": float(candidate.get("confidence", 0.0)),
                "presence_score": None
                if candidate.get("presence_score") is None
                else float(candidate["presence_score"]),
                "source": str(candidate.get("source", "automatic_propagation")),
                "is_human_verified": bool(candidate.get("is_human_verified", False)),
                "feature": vector.tolist(),
                "feature_dim": 512,
                "feature_sha256": feature_hash(vector),
                "feature_source": "machine_roi_fallback_osnet_market1501",
                "public_id": None,
                "public_id_status": "NOT_ASSIGNED_IN_OFFICIAL_BRANCH",
            }
        )
    return rows


def frame_row(
    *,
    event: dict[str, Any],
    branch: str,
    frame: int,
    phase: str,
    frame_path: Path,
    candidates: list[dict[str, Any]],
    runtime_memory_policy: dict[str, Any],
    y_pre_semantic_hash: str,
    correction: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "schema_version": "N72R4_OFFICIAL_SAM3_CANDIDATE_FRAME_V1",
        "record_type": "official_candidate_frame",
        "event_id": str(event["event_id"]),
        "sequence": str(event["sequence"]),
        "action_type": str(event["action_type"]),
        "branch": branch,
        "phase": phase,
        "event_frame": int(event["event_frame"]),
        "frame": int(frame),
        "frame_horizon": int(frame) - int(event["event_frame"]),
        "frame_hash_sha256": sha256_file(frame_path),
        "candidates": candidates,
        "candidate_count": len(candidates),
        "candidate_set_complete": True,
        "candidate_order": [int(item["candidate_index"]) for item in candidates],
        "y_pre_semantic_hash": y_pre_semantic_hash,
        "correction": correction,
        "runtime_memory_policy": runtime_memory_policy,
        "runtime_future_gt_used": False,
        "runtime_gt_read": False,
        "posthoc_gt_used": False,
        "interaction_source": "simulated_from_gt",
        "not_real_human_evidence": True,
        "public_id_inference": False,
        "candidate_stream_kind": "OFFICIAL_SAM3_FUTURE_PROPAGATION",
    }


def run_worker(event: dict[str, Any], branch: str, output: Path) -> dict[str, Any]:
    if branch not in BRANCHES:
        raise ValueError(f"unknown official branch: {branch}")
    event_id = str(event["event_id"])
    sequence = str(event["sequence"])
    event_frame = int(event["event_frame"])
    window = load_window(event)
    frame_end = min(int(window["frame_end"]), event_frame + HORIZON)
    if frame_end != event_frame + HORIZON:
        raise RuntimeError(f"frozen window cannot provide H100: {event_id}")
    sequence_dir = DATA_ROOT / "train" / sequence
    image_paths = image_files(sequence_dir)
    if not image_paths or frame_end >= len(image_paths):
        raise RuntimeError(f"official image range is incomplete: {sequence}:{event_frame}:{frame_end}")
    backend: Sam3Backend | None = None
    encoder: FrozenMachineOSNet | None = None
    continuous_baseline_outputs: dict[int, list[Any]] | None = None
    started = time.time()
    try:
        backend = make_backend()
        session_id = backend.start_video(str(sequence_dir / "img1"))
        install_official_shape_audit(backend)
        if branch == BRANCH_NO:
            pre, continuous_baseline_outputs = collect_continuous_baseline(
                backend, window, event_frame, frame_end
            )
        else:
            pre = collect_pre_event(backend, window, event_frame)
        if not pre:
            raise RuntimeError(f"official Y_pre is empty: {event_id}")
        pre_view = semantic_pre_view(pre)
        pre_hash = json_hash(pre_view)
        official_state_reconciliation = (
            {
                "status": "NOT_REQUIRED_CONTINUOUS_BASELINE",
                "reason": "baseline_used_one_uninterrupted_official_prefix_to_H100_stream",
                "frame_idx": event_frame,
                "persistent_public_identity_touched": False,
                "runtime_future_gt_used": False,
            }
            if branch == BRANCH_NO
            else backend.reconcile_official_tracker_to_visible_outputs(event_frame)
        )
        # The current event box is deliberately not accessed until Y_pre has
        # been frozen and hashed.  It is a current-frame simulated command,
        # never a future runtime label.
        correction_audit: dict[str, Any] | None = None
        post_observations: list[Any] | None = None
        if branch == BRANCH_CORRECTED:
            action, target_sam, human_box = correction_target(event)
            for observation in pre:
                backend.register_detected_observation(observation)
            if action == "RECOVER_IDENTITY":
                existing = [int(key) for key in getattr(backend, "_objects", {}).keys()]
                prompt_target = max(existing) + 1000 if existing else 1000
                post = backend.add_box(event_frame, prompt_target, human_box)
                route = "official_backend.add_box"
            else:
                prompt_target = target_sam
                if prompt_target not in getattr(backend, "_objects", {}):
                    raise RuntimeError(f"frozen correction target is absent from official Y_pre: {event_id}/{prompt_target}")
                post = backend.correct_object(event_frame, prompt_target, box_xyxy=human_box)
                route = "official_backend.correct_object"
            if post is None or not np.all(np.isfinite(np.asarray(post.box_xyxy, dtype=float))):
                raise RuntimeError(f"official correction returned invalid observation: {event_id}")
            fallback_entries = getattr(backend, "_prompt_fallback_log", [])[0:]
            no_output = any(
                int(item.get("object_id", -1)) == int(prompt_target) and bool(item.get("no_sam_output"))
                for item in fallback_entries
            )
            if no_output:
                raise RuntimeError(f"official correction emitted no SAM output: {event_id}")
            # The adapter cache also contains the returned human correction
            # observation.  Keep that row in correction_audit, but export only
            # the official SAM3 response as the candidate stream.
            post_observations = official_cached_observations(backend, event_frame)
            correction_audit = {
                "status": "PASS_OFFICIAL_CURRENT_FRAME_CORRECTION",
                "route": route,
                "action_type": action,
                "prompt_target_id": int(prompt_target),
                "frozen_target_sam_id": int(target_sam),
                "human_box": human_box.tolist(),
                "post_observation": observation_view(post),
                "post_observation_count": len(post_observations),
                "event_frame_memory_read": False,
                "first_future_frame": event_frame + 1,
                "runtime_future_gt_used": False,
                "interaction_source": "simulated_from_gt",
                "not_real_human_evidence": True,
                "prompt_fallback_log": fallback_entries,
                "official_state_reconciliation": official_state_reconciliation,
            }
        encoder = FrozenMachineOSNet("cuda:0")
        runtime_memory_policy = backend.runtime_memory_policy()
        output_rows: list[dict[str, Any]] = []
        pre_candidates = current_candidate_rows(backend, encoder, image_paths[event_frame], event_frame, pre)
        post_candidates = None
        if branch == BRANCH_CORRECTED:
            if post_observations is None:
                raise RuntimeError("corrected branch has no post observations")
            post_candidates = current_candidate_rows(
                backend, encoder, image_paths[event_frame], event_frame, post_observations
            )
        output_rows.append(
            frame_row(
                event=event,
                branch=branch,
                frame=event_frame,
                phase="Y_PRE_FROZEN",
                frame_path=image_paths[event_frame],
                candidates=pre_candidates,
                runtime_memory_policy=runtime_memory_policy,
                y_pre_semantic_hash=pre_hash,
                correction=(
                    None
                    if correction_audit is None
                    else {**correction_audit, "post_candidates": post_candidates}
                ),
            )
        )
        # Current-frame human correction is recorded as a sidecar in the Y_pre
        # row.  The pinned multiplex predictor is sensitive to an early-closed
        # propagation stream when the next request starts at t+1.  Existing
        # adapter resume validation uses the event frame as a context resume;
        # consume that frame but discard it from the future artifact so the
        # causal observation boundary remains exactly t+1.
        propagation_start = event_frame if branch == BRANCH_CORRECTED else int(window["frame_start"])
        future_outputs = (
            continuous_baseline_outputs
            if branch == BRANCH_NO
            else backend.propagate(
                event_frame,
                frame_end,
                start_frame_index=event_frame,
                max_frame_num_to_track=OFFICIAL_CONTEXT_WINDOW,
                keep_masks=True,
                cache_outputs=False,
            )
        )
        expected_propagation_frames = set(range(propagation_start, frame_end + 1))
        if set(int(frame) for frame in future_outputs) != expected_propagation_frames:
            missing = sorted(expected_propagation_frames - set(int(frame) for frame in future_outputs))
            extra = sorted(set(int(frame) for frame in future_outputs) - expected_propagation_frames)
            raise RuntimeError(f"official future coverage mismatch: missing={missing[:8]} extra={extra[:8]}")
        for frame in range(event_frame + 1, frame_end + 1):
            observations = [observation.copy() for observation in future_outputs[frame]]
            candidates = current_candidate_rows(backend, encoder, image_paths[frame], frame, observations)
            output_rows.append(
                frame_row(
                    event=event,
                    branch=branch,
                    frame=frame,
                    phase="FUTURE_PROPAGATION",
                    frame_path=image_paths[frame],
                    candidates=candidates,
                    runtime_memory_policy=runtime_memory_policy,
                    y_pre_semantic_hash=pre_hash,
                )
            )
        output_rows.sort(key=lambda row: int(row["frame"]))
        atomic_jsonl(output, output_rows)
        result = {
            "schema_version": "N72R4_OFFICIAL_SAM3_FUTURE_BRANCH_DONE_V1",
            "status": "PASS_OFFICIAL_SAM3_FUTURE_BRANCH",
            "event_id": event_id,
            "sequence": sequence,
            "action_type": str(event["action_type"]),
            "branch": branch,
            "session_id": str(session_id),
            "event_frame": event_frame,
            "frame_start": event_frame,
            "frame_end": frame_end,
            "future_frame_count": HORIZON,
            "propagation_start_frame": int(propagation_start),
            "propagation_context_frame_discarded": None
            if branch == BRANCH_NO
            else int(event_frame),
            "artifact_first_future_frame": int(event_frame + 1),
            "y_pre_candidate_count": len(pre),
            "y_pre_semantic_hash": pre_hash,
            "y_pre_frozen": pre_view,
            "official_state_reconciliation": official_state_reconciliation,
            "correction": correction_audit,
            "post_candidate_count": None if post_candidates is None else len(post_candidates),
            "candidate_artifact": str(output),
            "candidate_artifact_sha256": sha256_file(output),
            "candidate_frame_count": len(output_rows),
            "candidate_complete": True,
            "official_backend": True,
            "official_future_propagation": True,
            "checkpoint_sha256": sha256_file(CHECKPOINT),
            "runtime_memory_policy": runtime_memory_policy,
            "runtime_future_gt_used": False,
            "runtime_gt_read": False,
            "posthoc_gt_used": False,
            "interaction_source": "simulated_from_gt",
            "not_real_human_evidence": True,
            "public_id_inference": False,
            "scientific_result": "OFFICIAL_SAM3_FUTURE_CANDIDATE_STREAM_ONLY_NO_POSTHOC_EFFECT",
            "elapsed_sec": time.time() - started,
        }
        done_path = output.with_suffix(".done.json")
        atomic_json(done_path, result)
        return result
    except Exception as exc:
        if backend is not None and backend._predictor is not None:
            setattr(
                exc,
                "_official_shape_audit",
                list(getattr(backend._predictor.model, "_n72r4_shape_audit", [])),
            )
        raise
    finally:
        if backend is not None:
            try:
                backend.close()
            except Exception:
                pass
        encoder = None
        backend = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def write_worker_failure(output: Path, event: dict[str, Any], branch: str, exc: BaseException) -> Path:
    failure_path = output.with_suffix(".failure.json")
    atomic_json(
        failure_path,
        {
            "schema_version": "N72R4_FAILURE_RECORD_V1",
            "stage": "09_OFFICIAL_SAM3_FUTURE_BRANCH",
            "status": "FAIL_PRESERVED",
            "event_id": str(event["event_id"]),
            "branch": branch,
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
            "official_shape_audit": getattr(exc, "_official_shape_audit", []),
            "runtime_future_gt_used": False,
            "scientific_result": "NO_SCIENTIFIC_RESULT",
        },
    )
    return failure_path


def worker_main(args: argparse.Namespace) -> int:
    event = load_event(args.event_id)
    output = args.output.resolve()
    try:
        result = run_worker(event, args.branch, output)
        print(json.dumps({"status": result["status"], "event_id": args.event_id, "branch": args.branch, "output": str(output)}, sort_keys=True), flush=True)
        return 0
    except Exception as exc:
        failure = write_worker_failure(output, event, args.branch, exc)
        print(json.dumps({"status": "FAIL_OFFICIAL_SAM3_FUTURE_BRANCH", "event_id": args.event_id, "branch": args.branch, "failure": str(failure)}, sort_keys=True), flush=True)
        return 1


def pre_equivalence(no_result: dict[str, Any], corrected_result: dict[str, Any]) -> dict[str, Any]:
    no_view = no_result.get("y_pre_frozen", [])
    corrected_view = corrected_result.get("y_pre_frozen", [])
    fields = ("frame_idx", "sam_object_id", "raw_sam_object_id", "box_xyxy", "mask_sha256", "mask_shape", "confidence", "presence_score")
    mismatches: list[dict[str, Any]] = []
    if len(no_view) != len(corrected_view):
        mismatches.append({"field": "candidate_count", "baseline": len(no_view), "corrected": len(corrected_view)})
    for index, (left, right) in enumerate(zip(no_view, corrected_view)):
        for field in fields:
            if left.get(field) != right.get(field):
                mismatches.append({"index": index, "field": field, "baseline": left.get(field), "corrected": right.get(field)})
    return {
        "status": "PASS_Y_PRE_EQUIVALENCE" if not mismatches else "BLOCKED_NONDETERMINISTIC_PREFIX",
        "baseline_hash": no_result.get("y_pre_semantic_hash"),
        "corrected_hash": corrected_result.get("y_pre_semantic_hash"),
        "hash_equal": no_result.get("y_pre_semantic_hash") == corrected_result.get("y_pre_semantic_hash"),
        "candidate_count_equal": len(no_view) == len(corrected_view),
        "raw_ids_structurally_equal": [item.get("raw_sam_object_id") for item in no_view] == [item.get("raw_sam_object_id") for item in corrected_view],
        "mismatch_count": len(mismatches),
        "mismatches": mismatches[:32],
        "runtime_future_gt_used": False,
    }


def launch_worker(event: dict[str, Any], branch: str, root: Path, gpu: int, log_root: Path) -> dict[str, Any]:
    event_id = str(event["event_id"])
    output = root / f"{event_id}.jsonl"
    log_path = log_root / f"{event_id}.{branch}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_handle = log_path.open("w", encoding="utf-8")
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = str(gpu)
    environment["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    command = [
        str(Path(sys.executable)),
        str(Path(__file__).resolve()),
        "--worker",
        "--event-id",
        event_id,
        "--branch",
        branch,
        "--output",
        str(output),
    ]
    started = time.time()
    try:
        completed = subprocess.run(
            command,
            cwd=str(ROOT),
            env=environment,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    finally:
        log_handle.close()
    done_path = output.with_suffix(".done.json")
    failure_path = output.with_suffix(".failure.json")
    payload = read_json(done_path) if done_path.is_file() else (read_json(failure_path) if failure_path.is_file() else {})
    return {
        "event_id": event_id,
        "branch": branch,
        "gpu": int(gpu),
        "return_code": int(completed.returncode),
        "output": str(output),
        "done": str(done_path) if done_path.is_file() else None,
        "failure": str(failure_path) if failure_path.is_file() else None,
        "log": str(log_path),
        "status": payload.get("status", "MISSING_ARTIFACT"),
        "elapsed_sec": time.time() - started,
        "payload": payload,
    }


def compare_event(event: dict[str, Any], no_record: dict[str, Any], corrected_record: dict[str, Any]) -> dict[str, Any]:
    no_payload = no_record.get("payload", {})
    corrected_payload = corrected_record.get("payload", {})
    if no_record.get("return_code") != 0 or corrected_record.get("return_code") != 0:
        return {
            "event_id": str(event["event_id"]),
            "status": "FAIL_BRANCH_EXECUTION",
            "baseline_status": no_record.get("status"),
            "corrected_status": corrected_record.get("status"),
            "runtime_future_gt_used": False,
        }
    equivalence = pre_equivalence(no_payload, corrected_payload)
    return {
        "event_id": str(event["event_id"]),
        "sequence": str(event["sequence"]),
        "status": "PASS_OFFICIAL_PAIRED_PREFIX" if equivalence["status"] == "PASS_Y_PRE_EQUIVALENCE" else "BLOCKED_NONDETERMINISTIC_PREFIX",
        "baseline_artifact": no_record.get("output"),
        "corrected_artifact": corrected_record.get("output"),
        "baseline_future_frame_count": no_payload.get("future_frame_count"),
        "corrected_future_frame_count": corrected_payload.get("future_frame_count"),
        "pre_event_equivalence": equivalence,
        "runtime_future_gt_used": False,
        "interaction_source": "simulated_from_gt",
        "not_real_human_evidence": True,
    }


def orchestrator_main(args: argparse.Namespace) -> int:
    events = load_events()
    if args.event_id:
        events = [event for event in events if str(event["event_id"]) == str(args.event_id)]
        if not events:
            raise RuntimeError(f"unknown frozen event: {args.event_id}")
    no_root = args.no_root.resolve()
    corrected_root = args.corrected_root.resolve()
    log_root = args.log_root.resolve()
    for root in (no_root, corrected_root):
        if root.exists() and any(root.iterdir()):
            raise RuntimeError(f"official branch output root is not empty: {root}")
        root.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    comparisons: list[dict[str, Any]] = []
    for event in events:
        no_record = launch_worker(event, BRANCH_NO, no_root, int(args.gpu), log_root)
        records.append(no_record)
        corrected_record = launch_worker(event, BRANCH_CORRECTED, corrected_root, int(args.gpu), log_root)
        records.append(corrected_record)
        comparisons.append(compare_event(event, no_record, corrected_record))
        print(json.dumps({"event_id": event["event_id"], "baseline": no_record["status"], "corrected": corrected_record["status"], "paired": comparisons[-1]["status"]}, sort_keys=True), flush=True)
        if comparisons[-1]["status"] != "PASS_OFFICIAL_PAIRED_PREFIX":
            # Preserve the failed pair and stop before CPU effect stages.  A
            # later targeted repair must use this same event and input.
            break
    all_pass = len(comparisons) == len(events) and all(item["status"] == "PASS_OFFICIAL_PAIRED_PREFIX" for item in comparisons)
    scope = "targeted_smoke" if args.event_id else "full_frozen_event_set"
    manifest = {
        "schema_version": "N72R4_OFFICIAL_SAM3_FUTURE_PAIR_MANIFEST_V1",
        "status": "PASS_OFFICIAL_PAIRED_FUTURE_STREAM" if all_pass else "BLOCKED_OFFICIAL_PAIRED_FUTURE_STREAM",
        "execution_scope": scope,
        "event_count_expected": len(events),
        "event_count_completed": len(comparisons),
        "events": comparisons,
        "worker_records": [{key: value for key, value in record.items() if key != "payload"} for record in records],
        "no_intervention_root": str(no_root),
        "corrected_root": str(corrected_root),
        "checkpoint_sha256": sha256_file(CHECKPOINT),
        "horizon": HORIZON,
        "same_checkpoint": True,
        "same_prefix_protocol": True,
        "pre_event_equivalence_required": True,
        "official_future_propagation": True,
        "runtime_future_gt_used": False,
        "interaction_source": "simulated_from_gt",
        "not_real_human_evidence": True,
        "public_id_inference": False,
        "scientific_result": "OFFICIAL_SAM3_BRANCH_EXECUTION_NO_POSTHOC_EFFECT_RESULT",
    }
    manifest_path = args.manifest.resolve()
    atomic_json(manifest_path, manifest)
    status_path = args.stage_status.resolve()
    atomic_json(
        status_path,
        {
            "schema_version": "N72R4_STAGE_STATUS_V1",
            "stage": "09_OFFICIAL_SAM3_BASELINE_TREATMENT_FUTURE_PROPAGATION",
            "status": manifest["status"],
            "execution_scope": scope,
            "event_count_expected": len(events),
            "event_count_completed": len(comparisons),
            "paired_prefix_equivalence": all_pass,
            "future_frame_count_per_branch": HORIZON,
            "official_future_propagation": True,
            "checkpoint_sha256": manifest["checkpoint_sha256"],
            "runtime_future_gt_used": False,
            "interaction_source": "simulated_from_gt",
            "not_real_human_evidence": True,
            "manifest": str(manifest_path),
            "scientific_result": manifest["scientific_result"],
        },
    )
    print(json.dumps({"status": manifest["status"], "manifest": str(manifest_path)}, sort_keys=True))
    return 0 if all_pass else 1


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--event-id")
    parser.add_argument("--branch", choices=BRANCHES)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--gpu", type=int, default=1)
    parser.add_argument("--no-root", type=Path, default=DEFAULT_NO_ROOT)
    parser.add_argument("--corrected-root", type=Path, default=DEFAULT_CORRECTED_ROOT)
    parser.add_argument("--log-root", type=Path, default=DEFAULT_ATTEMPT_ROOT)
    parser.add_argument("--manifest", type=Path, default=STAGE_ROOT / "official_future_pair_manifest.json")
    parser.add_argument("--stage-status", type=Path, default=STAGE_ROOT / "stage_09_status.json")
    args = parser.parse_args()
    if args.worker:
        if args.event_id is None:
            raise SystemExit("--worker requires --event-id")
        if args.branch is None or args.output is None:
            raise SystemExit("--worker requires --branch and --output")
        if not torch.cuda.is_available():
            raise RuntimeError("official SAM3 future worker requires CUDA")
        return worker_main(args)
    if not torch.cuda.is_available():
        raise RuntimeError("official SAM3 future pair requires CUDA")
    return orchestrator_main(args)


if __name__ == "__main__":
    raise SystemExit(main())
