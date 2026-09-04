#!/usr/bin/env python3
"""N72R6 Stage 01/04: run one independent target-only SAM correction stream.

The script is intentionally one-event-per-process.  It reads the frozen
N72R5R1 event decision only to select an already-APPLIED event and to freeze
the main Y_pre provenance.  The current synthetic event box is allowed by
the N72R6 protocol, but it is explicitly marked ``simulated_from_gt`` and is
never read again as runtime future GT.
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
from sam3_intermot.identity.correction_epoch import feature_sha256  # noqa: E402,F811
from sam3_intermot.interaction.target_correction_session import (  # noqa: E402
    TargetScopedCorrectionSession,
    extract_human_roi_feature,
)
from scripts.n72r5_stage07_official_full_loop import (  # noqa: E402
    CHECKPOINT,
    DATA_ROOT,
    FrozenMachineOSNetN72R5,
    image_files,
)


EVENT_MANIFEST = ROOT / "outputs/N72R5/mechanism_rounds/round_06_event_policy/real_event_manifest.json"
STAGE08_MANIFEST = ROOT / "outputs/N72R5R1/controller/round_05_branch_isolation_v0/full/stage08_runtime_manifest.json"
OUT = ROOT / "outputs/N72R6"
HORIZON = 100


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


def _resolve(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_file() else ROOT / path


def _events() -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    payload = read_json(EVENT_MANIFEST)
    events = [dict(item) for item in payload.get("events", [])]
    stage08 = read_json(STAGE08_MANIFEST)
    stage_by_id = {str(item.get("event_id")): dict(item) for item in stage08.get("events", [])}
    if len(events) != 40 or len(stage_by_id) != 40:
        raise RuntimeError(f"frozen event coverage mismatch: events={len(events)}, stage08={len(stage_by_id)}")
    return events, stage_by_id


def eligible_event(event_id: str) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    events, stage_by_id = _events()
    event = next((item for item in events if str(item.get("event_id")) == str(event_id)), None)
    if event is None:
        raise KeyError(f"unknown frozen event: {event_id}")
    stage_event = stage_by_id.get(str(event_id))
    branches = {str(item.get("branch")): dict(item) for item in stage_event.get("branches", [])}
    branch = branches.get("B1_SPATIAL_CORRECTION_ONLY")
    main_branch = branches.get("B0_NO_INTERVENTION")
    if branch is None or str(branch.get("action_precondition_status")) != "APPLIED":
        raise RuntimeError(f"event is not an eligible N72R5R1 APPLIED event: {event_id}")
    if main_branch is None:
        raise RuntimeError(f"event has no frozen B0 main branch: {event_id}")
    target_public = branch.get("target_public_id")
    if target_public is None:
        raise RuntimeError(f"eligible event has no frozen target public ID: {event_id}")
    event["n72r6_target_public_id"] = int(target_public)
    return event, stage_event, branch, main_branch


def _main_y_pre(branch: Mapping[str, Any], event_frame: int) -> tuple[str, str, dict[str, Any]]:
    path = _resolve(str(branch["output"]))
    if not path.is_file():
        raise FileNotFoundError(path)
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                if int(row.get("frame", -1)) == int(event_frame):
                    rows.append(dict(row))
    if len(rows) != 1:
        raise RuntimeError(f"frozen main Y_pre row coverage is not one: {path}:{event_frame}:{len(rows)}")
    row = rows[0]
    if row.get("candidate_role") != "PRE_INTERVENTION_Y_PRE":
        raise RuntimeError(f"frozen main row is not Y_pre: {row.get('candidate_role')}")
    if any(row.get(flag) is not False for flag in ("runtime_future_gt_used", "runtime_gt_read", "posthoc_gt_used")):
        raise RuntimeError("frozen main Y_pre contains a GT flag")
    candidate_projection = [
        {
            key: item.get(key)
            for key in (
                "candidate_uid",
                "candidate_index",
                "official_raw_sam_id",
                "adapter_external_id",
                "box_xyxy",
                "feature_sha256",
            )
        }
        for item in row.get("candidate_rows", [])
    ]
    y_pre_hash = row.get("y_pre_semantic_hash") or row.get("shared_y_pre_semantic_hash")
    if not y_pre_hash:
        raise RuntimeError("frozen main Y_pre semantic hash is missing")
    return str(y_pre_hash), digest_json(candidate_projection), row


def _mask_sha256(mask: Any) -> str | None:
    if mask is None:
        return None
    value = np.asarray(mask, dtype=bool)
    return hashlib.sha256(value.tobytes()).hexdigest()


def _candidate_row(
    event: Mapping[str, Any],
    observation: Any,
    frame: int,
    feature: np.ndarray,
    epoch_id: str,
    session: TargetScopedCorrectionSession,
) -> dict[str, Any]:
    raw = observation.raw_sam_object_id
    if raw is None:
        raw = observation.sam_object_id
    adapter = int(observation.sam_object_id)
    scope = str(session.target_session_scope)
    uid = f"{event['event_id']}:target:{int(frame)}:{int(raw)}:{adapter}"
    vector = np.asarray(feature, dtype=np.float32).reshape(-1)
    return {
        "candidate_uid": uid,
        "candidate_index": 0,
        "candidate_kind": "TARGET_CORRECTION_SESSION_CANDIDATE",
        "sequence": str(event["sequence"]),
        "frame": int(frame),
        "official_raw_sam_id": int(raw),
        "adapter_external_id": adapter,
        "native_tid": adapter,
        "native_scope": scope,
        "native_tid_scope": scope,
        "box_xyxy": np.asarray(observation.box_xyxy, dtype=float).tolist(),
        "mask_sha256": _mask_sha256(observation.mask),
        "confidence": float(observation.confidence),
        "presence_score": None if observation.presence_score is None else float(observation.presence_score),
        "feature": vector.astype(float).tolist(),
        "feature_dim": int(vector.size),
        "feature_sha256": feature_sha256(vector),
        "feature_source": "target_session_machine_roi_feature",
        "source": str(observation.source),
        "source_session_id": session.session_id,
        "target_session_scope": scope,
        "human_target_scope_public_id": int(event["n72r6_target_public_id"]),
        "correction_epoch_id": epoch_id,
        "public_id": None,
        "public_id_inference": False,
        "runtime_future_gt_used": False,
        "runtime_gt_read": False,
        "posthoc_gt_used": False,
    }


def _frame_row(
    event: Mapping[str, Any],
    frame: int,
    candidate_rows: list[dict[str, Any]],
    *,
    epoch_id: str,
    session: TargetScopedCorrectionSession,
    frame_path: Path,
    main_y_pre_hash: str,
    main_y_pre_candidate_hash: str,
) -> dict[str, Any]:
    event_frame = int(event["event_frame"])
    return {
        "schema_version": "N72R6_TARGET_CORRECTION_FRAME_V1",
        "record_kind": "target_correction_session_frame",
        "event_id": str(event["event_id"]),
        "sequence": str(event["sequence"]),
        "action_type": str(event["action_type"]),
        "event_frame": event_frame,
        "frame": int(frame),
        "frame_horizon": int(frame - event_frame),
        "target_session_local_frame": int(frame - event_frame),
        "phase": "EVENT_FRAME_TARGET_SESSION" if frame == event_frame else "FUTURE_TARGET_SESSION",
        "is_event_frame": bool(frame == event_frame),
        "is_future_frame": bool(frame > event_frame),
        "frame_hash_sha256": sha256_file(frame_path),
        "candidate_rows": candidate_rows,
        "candidate_count": len(candidate_rows),
        "candidate_set_complete": True,
        "candidate_stream_kind": "INDEPENDENT_ONE_TARGET_SAM3_SESSION",
        "target_session_scope": str(session.target_session_scope),
        "source_session_id": session.session_id,
        "correction_epoch_id": epoch_id,
        "human_target_scope_public_id": int(event["n72r6_target_public_id"]),
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


def _failure_path(root: Path, event_id: str) -> Path:
    path = root / f"{event_id}.failure.json"
    if not path.exists():
        return path
    index = 2
    while True:
        path = root / f"{event_id}.failure.attempt{index}.json"
        if not path.exists():
            return path
        index += 1


def _materialize_event_local_window(
    paths: Sequence[Path],
    start_frame: int,
    end_frame: int,
) -> tuple[tempfile.TemporaryDirectory[str], Path, list[dict[str, Any]]]:
    """Expose exact event..H100 pixels as a local official video.

    The pinned multiplex predictor accepts a first box prompt reliably at
    local frame zero, while a fresh full-sequence session can return no
    official object for the same late global frame.  Symlinks change no
    pixels/checkpoint and the returned mapping keeps global coordinates
    auditable; the target session sees no pre-event state.
    """

    if start_frame < 0 or end_frame < start_frame or end_frame >= len(paths):
        raise ValueError(f"invalid event-local window: {start_frame}:{end_frame}:{len(paths)}")
    handle: tempfile.TemporaryDirectory[str] = tempfile.TemporaryDirectory(
        prefix="n72r6-target-window-"
    )
    image_dir = Path(handle.name) / "img1"
    image_dir.mkdir(parents=True, exist_ok=True)
    mapping: list[dict[str, Any]] = []
    for local_frame, global_frame in enumerate(range(int(start_frame), int(end_frame) + 1)):
        source = Path(paths[global_frame]).resolve()
        link = image_dir / f"{local_frame:08d}{source.suffix.lower()}"
        os.symlink(source, link)
        mapping.append(
            {
                "local_frame": int(local_frame),
                "global_frame": int(global_frame),
                "source_frame": str(source),
                "source_frame_sha256": sha256_file(source),
                "local_frame_name": link.name,
            }
        )
    return handle, image_dir, mapping


def run_event(event_id: str, *, attempt: int, device: str, output_root: Path, recovery_mode: bool = False) -> dict[str, Any]:
    event, stage_event, branch, main_branch = eligible_event(event_id)
    event_frame = int(event["event_frame"])
    end_frame = event_frame + HORIZON
    sequence_dir = DATA_ROOT / "train" / str(event["sequence"])
    paths = image_files(sequence_dir)
    if not paths or end_frame >= len(paths):
        raise RuntimeError(f"image coverage is incomplete: {event['sequence']}:{event_frame}:{end_frame}")
    y_pre_hash, y_pre_candidate_hash, y_pre_row = _main_y_pre(main_branch, event_frame)
    event_box = event.get("current_gt_box")
    if not isinstance(event_box, list) or len(event_box) != 4:
        raise RuntimeError("simulated event lacks the frozen current_gt_box")
    epoch_id = f"{event_id}:correction_epoch:1"
    root = output_root / f"attempt_{int(attempt)}" / str(event_id)
    frame_path = root / "frames.jsonl"
    done_path = root / "done.json"
    anchor_path = root / "human_anchor.json"
    mapping_path = root / "target_session_frame_mapping.json"
    if frame_path.exists() or done_path.exists():
        raise RuntimeError(f"refusing to overwrite existing target stream artifact: {root}")
    backend: Sam3Backend | None = None
    session: TargetScopedCorrectionSession | None = None
    encoder: FrozenMachineOSNetN72R5 | None = None
    window_handle: tempfile.TemporaryDirectory[str] | None = None
    started = time.time()
    rows: list[dict[str, Any]] = []
    session_audit: dict[str, Any] = {}
    memory_policy: dict[str, Any] = {}
    try:
        backend = Sam3Backend(
            checkpoint_path=str(CHECKPOINT),
            # The pinned checkpoint has a 16-slot official object axis.  A
            # one-slot builder changes parameter shapes and cannot load this
            # checkpoint; keep the official 16/16 capacity while the session
            # itself creates exactly one target object.
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
        session = TargetScopedCorrectionSession(
            backend=backend,
            event_id=str(event_id),
            sequence=str(event["sequence"]),
            public_id=int(event["n72r6_target_public_id"]),
            event_frame=event_frame,
            frame_offset=event_frame,
        )
        window_handle, target_video_dir, frame_mapping = _materialize_event_local_window(
            paths, event_frame, end_frame
        )
        atomic_json(
            mapping_path,
            {
                "schema_version": "N72R6_TARGET_SESSION_FRAME_MAPPING_V1",
                "event_id": str(event_id),
                "sequence": str(event["sequence"]),
                "global_start_frame": event_frame,
                "global_end_frame": end_frame,
                "local_start_frame": 0,
                "local_end_frame": HORIZON,
                "mode": "SYMLINK_EXACT_EVENT_LOCAL_WINDOW",
                "mapping": frame_mapping,
                "runtime_future_gt_used": False,
            },
        )
        session.start(target_video_dir, main_y_pre_frozen=True)
        encoder = FrozenMachineOSNetN72R5(device)
        human_anchor = extract_human_roi_feature(paths[event_frame], event_box, encoder)
        session.seed_from_human_box(event_box)
        outputs = session.propagate_to(end_frame)
        if recovery_mode:
            # First materialize the ordinary target-only stream.  If a future
            # row is absent, re-prompt this independent target session from
            # the last already observed target-session box and propagate only
            # the suffix.  The main B0 stream is never passed to recovery.
            last_box = np.asarray(event_box, dtype=float)
            last_observed_frame = event_frame
            recovery_failed_in_gap = False
            for global_frame in range(event_frame + 1, end_frame + 1):
                observation = session.candidate_at(global_frame)
                if observation is not None:
                    last_box = np.asarray(observation.box_xyxy, dtype=float)
                    last_observed_frame = global_frame
                    recovery_failed_in_gap = False
                    continue
                if recovery_failed_in_gap:
                    continue
                try:
                    recovered = session.recover_from_last_observation(
                        global_frame,
                        last_box,
                        source_frame=last_observed_frame,
                    )
                except RuntimeError as exc:
                    if not str(exc).startswith("official target recovery returned no observation overlapping the target box"):
                        raise
                    session.record_recovery_failure(
                        global_frame,
                        last_box,
                        source_frame=last_observed_frame,
                        error=exc,
                    )
                    recovery_failed_in_gap = True
                    continue
                if recovered is not None:
                    last_box = np.asarray(recovered.box_xyxy, dtype=float)
                    last_observed_frame = global_frame
                if global_frame < end_frame:
                    session.propagate_from(global_frame + 1, end_frame)
        epoch_scope = str(session.target_session_scope)
        for frame in range(event_frame, end_frame + 1):
            observation = session.candidate_at(frame)
            candidate_rows: list[dict[str, Any]] = []
            if observation is not None:
                target_feature = encoder.encode(paths[frame], [observation.box_xyxy.tolist()])[0]
                candidate_rows.append(
                    _candidate_row(event, observation, frame, target_feature, epoch_id, session)
                )
            rows.append(
                _frame_row(
                    event,
                    frame,
                    candidate_rows,
                    epoch_id=epoch_id,
                    session=session,
                    frame_path=paths[frame],
                    main_y_pre_hash=y_pre_hash,
                    main_y_pre_candidate_hash=y_pre_candidate_hash,
                )
            )
        if len(rows) != HORIZON + 1 or int(rows[0]["frame"]) != event_frame or int(rows[-1]["frame"]) != end_frame:
            raise RuntimeError("target frame coverage is not exactly event_frame..event_frame+H100")
        if not any(int(row["frame"]) == event_frame + 1 for row in rows):
            raise RuntimeError("event+1 row is missing")
        if int(rows[0]["candidate_count"]) != 1:
            raise RuntimeError(
                "official target-session add_box produced no event-frame candidate; "
                "the adapter human ledger row is not an official candidate"
            )
        session_audit = session.audit()
        memory_policy = backend.runtime_memory_policy()
        atomic_jsonl(frame_path, rows)
        atomic_json(
            anchor_path,
            {
                "schema_version": "N72R6_HUMAN_ROI_ANCHOR_V1",
                "event_id": str(event_id),
                "sequence": str(event["sequence"]),
                "event_frame": event_frame,
                "public_id": int(event["n72r6_target_public_id"]),
                "interaction_source": "simulated_from_gt",
                "not_real_human_evidence": True,
                "source": "frozen_current_event_box_only",
                "box_xyxy": [float(value) for value in event_box],
                "feature": np.asarray(human_anchor, dtype=np.float32).astype(float).tolist(),
                "feature_sha256": feature_sha256(human_anchor),
                "feature_source": "raw_current_frame_human_roi_osnet",
                "runtime_future_gt_used": False,
                "runtime_gt_read": False,
                "posthoc_gt_used": False,
            },
        )
        recovery_attempts = session_audit.get("recovery_attempts", [])
        recovery_failure_count = sum(
            int(isinstance(item, dict) and str(item.get("status", "")).startswith("FAIL_TARGET_RECOVERY"))
            for item in recovery_attempts
        )
        done = {
            "schema_version": "N72R6_TARGET_CORRECTION_STREAM_DONE_V1",
            "status": (
                "PASS_TARGET_STREAM_COMPLETE_WITH_RECOVERY_MISS"
                if recovery_failure_count
                else "PASS_TARGET_STREAM_COMPLETE"
            ),
            "event_id": str(event_id),
            "sequence": str(event["sequence"]),
            "action_type": str(event["action_type"]),
            "event_frame": event_frame,
            "end_frame": end_frame,
            "frame_count": len(rows),
            "candidate_row_count": sum(int(row["candidate_count"]) for row in rows),
            "target_candidate_present_event_frame": bool(rows[0]["candidate_count"]),
            "target_candidate_present_event_plus_one": bool(rows[1]["candidate_count"]),
            "frames": str(frame_path),
            "frames_sha256": sha256_file(frame_path),
            "human_anchor": str(anchor_path),
            "human_anchor_sha256": sha256_file(anchor_path),
            "correction_epoch_id": epoch_id,
            "target_session_scope": epoch_scope,
            "target_public_id": int(event["n72r6_target_public_id"]),
            "main_y_pre_semantic_hash": y_pre_hash,
            "main_y_pre_candidate_content_sha256": y_pre_candidate_hash,
            "main_y_pre_row_hash": digest_json(y_pre_row),
            "target_session_audit": session_audit,
            "target_session_recovery_mode": bool(recovery_mode),
            "target_session_recovery_attempt_count": len(recovery_attempts),
            "target_session_recovery_failure_count": recovery_failure_count,
            "target_session_recovery_status": (
                "TARGET_LOST_AFTER_OFFICIAL_RECOVERY_FAILURE"
                if recovery_failure_count
                else "NO_RECOVERY_FAILURE"
            ),
            "target_session_frame_mapping": str(mapping_path),
            "target_session_frame_mapping_sha256": sha256_file(mapping_path),
            "target_session_video_mode": "SYMLINK_EXACT_EVENT_LOCAL_WINDOW",
            "runtime_memory_policy": memory_policy,
            "checkpoint": str(CHECKPOINT),
            "checkpoint_sha256": sha256_file(CHECKPOINT),
            "event_manifest_sha256": sha256_file(EVENT_MANIFEST),
            "stage08_manifest_sha256": sha256_file(STAGE08_MANIFEST),
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
    except Exception as exc:
        failure = {
            "schema_version": "N72R6_TARGET_STREAM_FAILURE_V1",
            "status": "FAIL_TARGET_STREAM",
            "event_id": str(event_id),
            "sequence": str(event.get("sequence")),
            "attempt": int(attempt),
            "failure_type": type(exc).__name__,
            "message": str(exc),
            "traceback": traceback.format_exc(),
            "runtime_future_gt_used": False,
            "historical_outputs_modified": False,
            "created_at_utc": now_utc(),
        }
        atomic_json(_failure_path(output_root / "attempts", str(event_id)), failure)
        raise
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
        del encoder
        del session
        del backend
        if window_handle is not None:
            window_handle.cleanup()
        del window_handle
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--event-id", required=True)
    parser.add_argument("--attempt", type=int, default=1)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-root", type=Path, default=OUT / "target_correction_stream")
    parser.add_argument("--recovery-mode", action="store_true")
    args = parser.parse_args()
    result = run_event(
        args.event_id,
        attempt=int(args.attempt),
        device=str(args.device),
        output_root=args.output_root,
        recovery_mode=bool(args.recovery_mode),
    )
    print(json.dumps({"status": result["status"], "event_id": result["event_id"], "frame_count": result["frame_count"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
