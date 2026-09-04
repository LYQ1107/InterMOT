#!/usr/bin/env python3
"""N72R7-R5 causal official-SAM3 multi-query candidate generator.

This route tests one mechanism only: expose additional candidate evidence by
running a frozen, deterministic set of spatially perturbed box prompts in
independent target-only SAM3 sessions.  Query boxes are derived from the
event-frame human box and never from future frames or GT.  The resulting rows
carry no public-ID authority; the exact global solver remains downstream.
"""

from __future__ import annotations

import argparse
import gc
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import time
import traceback
from typing import Any, Mapping, Sequence

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sam3_intermot.association.branch_public_replay import read_json, sha256_file  # noqa: E402
from sam3_intermot.backend.sam3_backend import Sam3Backend  # noqa: E402
from sam3_intermot.identity.correction_epoch import feature_sha256  # noqa: E402
from sam3_intermot.interaction.target_correction_session import TargetScopedCorrectionSession  # noqa: E402
from scripts.n72r5_stage07_official_full_loop import (  # noqa: E402
    CHECKPOINT,
    DATA_ROOT,
    FrozenMachineOSNetN72R5,
    image_files,
)
from scripts.n72r6_target_correction_stream import (  # noqa: E402
    EVENT_MANIFEST,
    HORIZON,
    STAGE08_MANIFEST,
    _main_y_pre,
    _materialize_event_local_window,
    eligible_event,
)


PROTOCOL_PATH = ROOT / "outputs/N72R7/candidate_generator_protocol.json"
DEFAULT_OUTPUT_ROOT = ROOT / "outputs/N72R7/candidate_generator/r5"
QUERY_SPECS: tuple[dict[str, Any], ...] = (
    {"name": "CENTER_SHRINK", "dx_fraction": 0.0, "dy_fraction": 0.0, "scale": 0.82},
    {"name": "LEFT_OFFSET", "dx_fraction": -0.24, "dy_fraction": 0.0, "scale": 1.0},
    {"name": "RIGHT_OFFSET", "dx_fraction": 0.24, "dy_fraction": 0.0, "scale": 1.0},
    {"name": "UP_OFFSET", "dx_fraction": 0.0, "dy_fraction": -0.20, "scale": 1.0},
    {"name": "DOWN_OFFSET", "dx_fraction": 0.0, "dy_fraction": 0.20, "scale": 1.0},
)


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def atomic_json(path: Path, payload: Any) -> None:
    atomic_text(path, json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n")


def atomic_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    atomic_text(
        path,
        "".join(json.dumps(dict(row), ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n" for row in rows),
    )


def digest_json(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    ).hexdigest()


def resolve(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def write_protocol() -> dict[str, Any]:
    source_protocol = ROOT / "outputs/N72R7/protocol.json"
    body = {
        "schema_version": "N72R7_CANDIDATE_GENERATOR_REQUERY_PROTOCOL_V1",
        "created_at_utc": now_utc(),
        "source_protocol": str(source_protocol),
        "source_protocol_sha256": sha256_file(source_protocol),
        "event_policy": str(EVENT_MANIFEST),
        "event_policy_sha256": sha256_file(EVENT_MANIFEST),
        "stage08_manifest": str(STAGE08_MANIFEST),
        "stage08_manifest_sha256": sha256_file(STAGE08_MANIFEST),
        "checkpoint": str(CHECKPOINT),
        "checkpoint_sha256": sha256_file(CHECKPOINT),
        "mechanism": "CAUSAL_MULTI_QUERY_SAM3_TARGET_REQUERY",
        "query_boxes": [dict(item) for item in QUERY_SPECS],
        "query_box_definition": (
            "center=(x1+x2)/2,(y1+y2)/2; width/height scaled by scale; "
            "center translated by dx_fraction*width,dy_fraction*height; "
            "coordinates are clipped only by the official backend prompt adapter"
        ),
        "future_runtime_inputs_forbidden": [
            "future GT", "future identity labels", "future IoU", "future H20/H50/H100", "posthoc metrics"
        ],
        "candidate_source": "TARGET_SESSION_REQUERY",
        "candidate_kind": "TARGET_CORRECTION_SESSION_REQUERY_CANDIDATE",
        "public_id_authority": "frozen_event_input_and_exact_global_solver_only",
        "same_checkpoint": True,
        "same_candidate_definition": True,
        "same_metric_definition": True,
        "same_hungarian_solver": True,
        "runtime_future_gt_used": False,
        "interaction_source": "simulated_from_gt",
        "not_real_human_evidence": True,
        "independent_process_contract": "one_event_one_python_process; one backend per query, closed before next query",
        "horizon": HORIZON,
    }
    body["protocol_sha256"] = hashlib.sha256(
        json.dumps(body, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    ).hexdigest()
    atomic_json(PROTOCOL_PATH, body)
    return body


def query_box(human_box: Sequence[float], spec: Mapping[str, Any]) -> list[float]:
    box = np.asarray(human_box, dtype=np.float64).reshape(-1)
    if box.size != 4 or not np.all(np.isfinite(box)):
        raise ValueError("human box must contain four finite coordinates")
    x1, y1, x2, y2 = [float(item) for item in box]
    width = x2 - x1
    height = y2 - y1
    if width <= 0.0 or height <= 0.0:
        raise ValueError("human box must have positive area")
    scale = float(spec["scale"])
    dx = float(spec["dx_fraction"]) * width
    dy = float(spec["dy_fraction"]) * height
    cx = (x1 + x2) / 2.0 + dx
    cy = (y1 + y2) / 2.0 + dy
    half_w = width * scale / 2.0
    half_h = height * scale / 2.0
    result = [cx - half_w, cy - half_h, cx + half_w, cy + half_h]
    if not np.all(np.isfinite(result)) or result[2] <= result[0] or result[3] <= result[1]:
        raise ValueError(f"invalid deterministic query box: {spec}")
    return [float(item) for item in result]


def candidate_row(
    event: Mapping[str, Any],
    observation: Any,
    *,
    frame: int,
    query_index: int,
    query_name: str,
    query_box_xyxy: Sequence[float],
    feature: np.ndarray,
    epoch_id: str,
    session: TargetScopedCorrectionSession,
) -> dict[str, Any]:
    raw = observation.raw_sam_object_id
    if raw is None:
        raw = observation.sam_object_id
    adapter = int(observation.sam_object_id)
    vector = np.asarray(feature, dtype=np.float32).reshape(-1)
    if vector.size != 512 or not np.all(np.isfinite(vector)):
        raise ValueError("requery feature must be finite 512-D")
    uid = f"{event['event_id']}:requery:{query_index}:{query_name}:{int(frame)}:{int(raw)}:{adapter}"
    return {
        "candidate_uid": uid,
        "candidate_index": int(query_index),
        "candidate_kind": "TARGET_CORRECTION_SESSION_REQUERY_CANDIDATE",
        "candidate_source": "TARGET_SESSION_REQUERY",
        "sequence": str(event["sequence"]),
        "frame": int(frame),
        "official_raw_sam_id": int(raw),
        "adapter_external_id": adapter,
        "native_tid": adapter,
        "native_scope": str(session.target_session_scope),
        "native_tid_scope": str(session.target_session_scope),
        "box_xyxy": np.asarray(observation.box_xyxy, dtype=float).tolist(),
        "mask_sha256": None if observation.mask is None else hashlib.sha256(
            np.asarray(observation.mask, dtype=bool).tobytes()
        ).hexdigest(),
        "confidence": float(observation.confidence),
        "presence_score": None if observation.presence_score is None else float(observation.presence_score),
        "feature": vector.astype(float).tolist(),
        "feature_dim": int(vector.size),
        "feature_sha256": feature_sha256(vector),
        "feature_source": "target_session_requery_machine_roi_feature",
        "source": str(observation.source),
        "source_session_id": session.session_id,
        "target_session_scope": str(session.target_session_scope),
        "requery_index": int(query_index),
        "requery_name": str(query_name),
        "requery_box_xyxy": [float(value) for value in query_box_xyxy],
        "correction_epoch_id": str(epoch_id),
        "public_id": None,
        "public_id_inference": False,
        "runtime_future_gt_used": False,
        "runtime_gt_read": False,
        "posthoc_gt_used": False,
    }


def _backend(device: str) -> Sam3Backend:
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
        device=device,
    )


def _run_query(
    event: Mapping[str, Any],
    *,
    target_video_dir: Path,
    paths: Sequence[Path],
    event_frame: int,
    end_frame: int,
    target_public_id: int,
    query_index: int,
    spec: Mapping[str, Any],
    human_box: Sequence[float],
    encoder: FrozenMachineOSNetN72R5,
    device: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    query_name = str(spec["name"])
    prompt_box = query_box(human_box, spec)
    backend: Sam3Backend | None = None
    session: TargetScopedCorrectionSession | None = None
    started = time.time()
    try:
        backend = _backend(device)
        session = TargetScopedCorrectionSession(
            backend=backend,
            event_id=f"{event['event_id']}:requery:{query_name}",
            sequence=str(event["sequence"]),
            public_id=int(target_public_id),
            event_frame=int(event_frame),
            frame_offset=int(event_frame),
        )
        session.start(target_video_dir, main_y_pre_frozen=True)
        session.seed_from_human_box(prompt_box)
        session.propagate_to(int(end_frame))
        rows: list[dict[str, Any]] = []
        for frame in range(int(event_frame), int(end_frame) + 1):
            observation = session.candidate_at(frame)
            if observation is None:
                continue
            feature = encoder.encode(paths[frame], [observation.box_xyxy.tolist()])[0]
            rows.append(
                candidate_row(
                    event,
                    observation,
                    frame=frame,
                    query_index=query_index,
                    query_name=query_name,
                    query_box_xyxy=prompt_box,
                    feature=feature,
                    epoch_id=f"{event['event_id']}:correction_epoch:1",
                    session=session,
                )
            )
        audit = session.audit()
        return rows, {
            "status": "PASS_QUERY",
            "query_index": int(query_index),
            "query_name": query_name,
            "query_spec": dict(spec),
            "query_box_xyxy": [float(value) for value in prompt_box],
            "frame_count": int(end_frame - event_frame + 1),
            "candidate_row_count": len(rows),
            "event_frame_candidate_present": any(int(row["frame"]) == int(event_frame) for row in rows),
            "event_plus_one_candidate_present": any(int(row["frame"]) == int(event_frame + 1) for row in rows),
            "target_session_audit": audit,
            "runtime_future_gt_used": False,
            "runtime_gt_read": False,
            "posthoc_gt_used": False,
            "elapsed_sec": time.time() - started,
        }
    finally:
        if session is not None:
            try:
                session.close()
            except Exception:
                pass
        elif backend is not None:
            try:
                backend.close()
            except Exception:
                pass
        del session
        del backend
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def run_event(event_id: str, *, output_root: Path, attempt: int, device: str) -> dict[str, Any]:
    event, _stage_event, branch, main_branch = eligible_event(event_id)
    event_frame = int(event["event_frame"])
    end_frame = event_frame + HORIZON
    sequence = str(event["sequence"])
    paths = image_files(DATA_ROOT / "train" / sequence)
    if not paths or end_frame >= len(paths):
        raise RuntimeError(f"image coverage is incomplete: {sequence}:{event_frame}:{end_frame}")
    target_public_id = branch.get("target_public_id")
    if target_public_id is None:
        raise RuntimeError(f"frozen target public ID is absent: {event_id}")
    main_y_pre_hash, main_y_pre_candidate_hash, _ = _main_y_pre(main_branch, event_frame)
    human_box = event.get("current_gt_box")
    if not isinstance(human_box, list) or len(human_box) != 4:
        raise RuntimeError("simulated event lacks frozen current_gt_box")
    protocol = read_json(PROTOCOL_PATH)
    root = output_root / f"attempt_{int(attempt)}" / str(event_id)
    frames_path = root / "frames.jsonl"
    done_path = root / "done.json"
    mapping_path = root / "target_session_frame_mapping.json"
    if frames_path.exists() or done_path.exists():
        raise RuntimeError(f"refusing to overwrite existing candidate-generator artifact: {root}")
    window_handle: tempfile.TemporaryDirectory[str] | None = None
    encoder: FrozenMachineOSNetN72R5 | None = None
    started = time.time()
    try:
        window_handle, target_video_dir, mapping = _materialize_event_local_window(paths, event_frame, end_frame)
        atomic_json(
            mapping_path,
            {
                "schema_version": "N72R7_REQUERY_FRAME_MAPPING_V1",
                "event_id": str(event_id),
                "sequence": sequence,
                "global_start_frame": event_frame,
                "global_end_frame": end_frame,
                "local_start_frame": 0,
                "local_end_frame": HORIZON,
                "mode": "SYMLINK_EXACT_EVENT_LOCAL_WINDOW",
                "mapping": mapping,
                "runtime_future_gt_used": False,
            },
        )
        encoder = FrozenMachineOSNetN72R5(device)
        by_frame: dict[int, list[dict[str, Any]]] = {frame: [] for frame in range(event_frame, end_frame + 1)}
        query_audits: list[dict[str, Any]] = []
        query_failures: list[dict[str, Any]] = []
        for query_index, spec in enumerate(QUERY_SPECS):
            try:
                query_rows, audit = _run_query(
                    event,
                    target_video_dir=target_video_dir,
                    paths=paths,
                    event_frame=event_frame,
                    end_frame=end_frame,
                    target_public_id=int(target_public_id),
                    query_index=query_index,
                    spec=spec,
                    human_box=human_box,
                    encoder=encoder,
                    device=device,
                )
                query_audits.append(audit)
                for row in query_rows:
                    by_frame[int(row["frame"])].append(row)
            except Exception as exc:
                query_failure = {
                    "status": "FAIL_QUERY",
                    "query_index": int(query_index),
                    "query_name": str(spec["name"]),
                    "query_spec": dict(spec),
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                    "runtime_future_gt_used": False,
                }
                query_failures.append(query_failure)
                query_audits.append(query_failure)
        if query_failures:
            raise RuntimeError(f"candidate-generator query failures: {len(query_failures)}")
        rows: list[dict[str, Any]] = []
        for frame in range(event_frame, end_frame + 1):
            candidates = sorted(by_frame[frame], key=lambda item: (int(item["requery_index"]), str(item["candidate_uid"])))
            rows.append(
                {
                    "schema_version": "N72R7_CANDIDATE_GENERATOR_REQUERY_FRAME_V1",
                    "record_kind": "target_session_requery_frame",
                    "event_id": str(event_id),
                    "sequence": sequence,
                    "action_type": str(event["action_type"]),
                    "event_frame": event_frame,
                    "frame": int(frame),
                    "frame_horizon": int(frame - event_frame),
                    "is_event_frame": bool(frame == event_frame),
                    "is_future_frame": bool(frame > event_frame),
                    "frame_hash_sha256": sha256_file(paths[frame]),
                    "candidate_rows": candidates,
                    "candidate_count": len(candidates),
                    "candidate_set_complete": True,
                    "candidate_stream_kind": "INDEPENDENT_MULTI_QUERY_TARGET_SAM3_SESSIONS",
                    "candidate_source": "TARGET_SESSION_REQUERY",
                    "query_count": len(QUERY_SPECS),
                    "query_names": [str(spec["name"]) for spec in QUERY_SPECS],
                    "main_y_pre_frozen": True,
                    "main_y_pre_semantic_hash": main_y_pre_hash,
                    "main_y_pre_candidate_content_sha256": main_y_pre_candidate_hash,
                    "event_frame_memory_read": False,
                    "memory_read": False,
                    "first_memory_visible_frame": event_frame + 1,
                    "runtime_future_gt_used": False,
                    "runtime_gt_read": False,
                    "posthoc_gt_used": False,
                    "public_id_inference": False,
                    "interaction_source": "simulated_from_gt",
                    "not_real_human_evidence": True,
                }
            )
        if len(rows) != HORIZON + 1:
            raise RuntimeError("requery frame coverage is not exactly event..event+H100")
        atomic_jsonl(frames_path, rows)
        done = {
            "schema_version": "N72R7_CANDIDATE_GENERATOR_REQUERY_DONE_V1",
            "status": "PASS_N72R7_CANDIDATE_GENERATOR_REQUERY",
            "event_id": str(event_id),
            "sequence": sequence,
            "action_type": str(event["action_type"]),
            "event_frame": event_frame,
            "end_frame": end_frame,
            "frame_count": len(rows),
            "candidate_row_count": sum(int(row["candidate_count"]) for row in rows),
            "frames": str(frames_path),
            "frames_sha256": sha256_file(frames_path),
            "target_session_frame_mapping": str(mapping_path),
            "target_session_frame_mapping_sha256": sha256_file(mapping_path),
            "query_count": len(QUERY_SPECS),
            "query_specs": [dict(item) for item in QUERY_SPECS],
            "query_audits": query_audits,
            "target_public_id_input_authority": int(target_public_id),
            "protocol": str(PROTOCOL_PATH),
            "protocol_sha256": protocol["protocol_sha256"],
            "checkpoint": str(CHECKPOINT),
            "checkpoint_sha256": sha256_file(CHECKPOINT),
            "runtime_future_gt_used": False,
            "runtime_gt_read": False,
            "posthoc_gt_used": False,
            "public_id_inference": False,
            "interaction_source": "simulated_from_gt",
            "not_real_human_evidence": True,
            "attempt": int(attempt),
            "elapsed_sec": time.time() - started,
            "created_at_utc": now_utc(),
        }
        atomic_json(done_path, done)
        return done
    finally:
        del encoder
        if window_handle is not None:
            window_handle.cleanup()
        del window_handle
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--event-id")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--attempt", type=int, default=1)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--write-protocol", action="store_true")
    args = parser.parse_args()
    if args.write_protocol:
        payload = write_protocol()
        print(json.dumps({"status": "PASS_N72R7_REQUERY_PROTOCOL_FROZEN", "protocol_sha256": payload["protocol_sha256"]}))
        return 0
    if not args.event_id:
        parser.error("--event-id is required unless --write-protocol is used")
    output_root = args.output_root if args.output_root.is_absolute() else ROOT / args.output_root
    try:
        result = run_event(args.event_id, output_root=output_root, attempt=int(args.attempt), device=str(args.device))
        print(json.dumps({"status": result["status"], "event_id": result["event_id"], "candidate_row_count": result["candidate_row_count"]}))
        return 0
    except Exception as exc:
        failure_dir = output_root / "attempts"
        failure_dir.mkdir(parents=True, exist_ok=True)
        failure = failure_dir / f"{args.event_id}.attempt{int(args.attempt)}.failure.json"
        atomic_json(
            failure,
            {
                "schema_version": "N72R7_CANDIDATE_GENERATOR_REQUERY_FAILURE_V1",
                "status": "FAIL_N72R7_CANDIDATE_GENERATOR_REQUERY",
                "event_id": str(args.event_id),
                "attempt": int(args.attempt),
                "error_type": type(exc).__name__,
                "error": str(exc),
                "traceback": traceback.format_exc(),
                "runtime_future_gt_used": False,
                "historical_outputs_modified": False,
                "created_at_utc": now_utc(),
            },
        )
        print(json.dumps({"status": "FAIL_N72R7_CANDIDATE_GENERATOR_REQUERY", "event_id": args.event_id, "failure": str(failure)}))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
