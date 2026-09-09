#!/usr/bin/env python3
"""Generate one frozen N72R11 secondary causal interaction.

This worker is deliberately one secondary event per process.  The schedule is
the only input that contains the simulated current-frame human box.  That box
is consumed as the event prompt, while no dataset GT is opened after the
prompt is frozen.  The target stream is made with the existing
``TargetScopedCorrectionSession`` and the on-demand suffix, when the frozen
causal uncertainty rule fires, is made with ``LiveFutureRequeryController``.

The selector used while building the training corpus is a fixed, auditable
bootstrap selector.  It is not the final V3/bridge model and is never used for
formal E0/E1/E2 effect claims.  It uses only the current B0 score row,
appearance cosine to the frozen human ROI anchor, geometry and presence.  A
later V3/bridge training run consumes the sealed rows and its own causal
rollout.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timezone
import argparse
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import tempfile
import traceback
from typing import Any, Mapping, Sequence

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sam3_intermot.association.effect_assignment import solve_effect_assignment  # noqa: E402
from sam3_intermot.association.target_edge_bridge import SOURCE_NAMES  # noqa: E402
from sam3_intermot.association.secondary_public_score import (  # noqa: E402
    TARGET_PUBLIC_EDGE_CURRENT,
    TARGET_PUBLIC_EDGE_FUTURE,
    build_secondary_public_score_frame,
    supplemental_target_edge,
)
from sam3_intermot.interaction.target_correction_session import TargetScopedCorrectionSession  # noqa: E402
from sam3_intermot.reacquisition.frozen_feature_materializer import (  # noqa: E402
    FrozenOSNetFeatureMaterializer,
)
from sam3_intermot.reacquisition.live_requery_controller import LiveFutureRequeryController  # noqa: E402
from sam3_intermot.reacquisition.target_candidate_pool import (  # noqa: E402
    FUTURE_FRAME_REQUERY,
    build_candidate_pool_with_future_requery,
    serializable_candidate,
)
from scripts.n72r5_stage07_official_full_loop import (  # noqa: E402
    CHECKPOINT,
    DATA_ROOT,
    image_files,
)
from scripts.n72r6_target_correction_stream import (  # noqa: E402
    _materialize_event_local_window,
    atomic_json,
    atomic_jsonl,
    digest_json,
    sha256_file,
)


SCHEDULE_PATH = ROOT / "outputs/N72R11/secondary_event_manifest.json"
DEFAULT_OUTPUT_ROOT = ROOT / "outputs/N72R11/secondary_interactions"
HORIZON = 100
UNCERTAINTY_MARGIN = 0.25
TARGET_PUBLIC_EDGE_BASE = TARGET_PUBLIC_EDGE_FUTURE


@dataclass(frozen=True)
class TargetSessionAuditRef:
    """Small immutable reference retained after the SAM3 session is closed."""

    session_id: str
    target_session_scope: str


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _cuda_memory_snapshot() -> dict[str, int] | None:
    if not torch.cuda.is_available():
        return None
    return {
        "allocated_mib": int(torch.cuda.memory_allocated() / (1024 * 1024)),
        "reserved_mib": int(torch.cuda.memory_reserved() / (1024 * 1024)),
        "max_allocated_mib": int(torch.cuda.max_memory_allocated() / (1024 * 1024)),
        "max_reserved_mib": int(torch.cuda.max_memory_reserved() / (1024 * 1024)),
    }


def _reset_cuda_peak_stats() -> None:
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


def _finite_box(value: Any, label: str) -> np.ndarray:
    box = np.asarray(value, dtype=np.float64).reshape(-1)
    if box.size != 4 or not np.all(np.isfinite(box)) or box[2] <= box[0] or box[3] <= box[1]:
        raise ValueError(f"{label} is not a finite positive XYXY box")
    return box


def _finite_box_values(value: Any, label: str) -> np.ndarray:
    """Validate candidate coordinates without rejecting finite degenerate boxes.

    The frozen candidate pool retains finite zero-area observations and marks
    them as geometry-invalid.  They remain in the candidate stream and must
    contribute zero motion IoU rather than aborting the worker.  Human/prompt
    boxes continue to use ``_finite_box`` and therefore remain strictly
    positive.
    """
    box = np.asarray(value, dtype=np.float64).reshape(-1)
    if box.size != 4 or not np.all(np.isfinite(box)):
        raise ValueError(f"{label} is not a finite XYXY box")
    return box


def _unit(value: Any, label: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32).reshape(-1)
    if array.size != 512 or not np.all(np.isfinite(array)):
        raise ValueError(f"{label} must be finite 512-D")
    norm = float(np.linalg.norm(array))
    if norm <= 1.0e-6:
        raise ValueError(f"{label} has zero norm")
    return array / norm


def _box_iou(left: Any, right: Any) -> float:
    a = _finite_box_values(left, "left box")
    b = _finite_box_values(right, "right box")
    if a[2] <= a[0] or a[3] <= a[1] or b[2] <= b[0] or b[3] <= b[1]:
        return 0.0
    x1, y1 = max(float(a[0]), float(b[0])), max(float(a[1]), float(b[1]))
    x2, y2 = min(float(a[2]), float(b[2])), min(float(a[3]), float(b[3]))
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, float(a[2] - a[0])) * max(0.0, float(a[3] - a[1]))
    area_b = max(0.0, float(b[2] - b[0])) * max(0.0, float(b[3] - b[1]))
    union = area_a + area_b - intersection
    return float(intersection / union) if union > 0.0 else 0.0


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object: {path}")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise TypeError(f"{path}:{line_number} is not an object")
            rows.append(value)
    return rows


def _load_schedule(event_id: str) -> dict[str, Any]:
    payload = _read_json(SCHEDULE_PATH)
    decisions = [
        dict(item)
        for item in payload.get("all_decisions", [])
        if str(item.get("event_id")) == str(event_id)
    ]
    if len(decisions) != 1:
        raise KeyError(f"expected one schedule decision for {event_id}, found {len(decisions)}")
    item = decisions[0]
    if item.get("status") != "ELIGIBLE":
        raise RuntimeError(f"secondary event is not eligible: {event_id}:{item.get('status')}")
    required = (
        "sequence", "secondary_frame", "original_event_frame", "target_public_id",
        "target_dataset_gt_id", "action_type", "current_target_box_posthoc_selection_only",
    )
    for key in required:
        if key not in item:
            raise RuntimeError(f"schedule item lacks {key}: {event_id}")
    if item.get("runtime_future_gt_used") is not False:
        raise RuntimeError(f"schedule permits runtime future GT: {event_id}")
    if item.get("interaction_source") != "simulated_from_gt" or item.get("not_real_human_evidence") is not True:
        raise RuntimeError(f"secondary source provenance is not explicit simulated_from_gt: {event_id}")
    return item


def _load_c0_row(item: Mapping[str, Any], frame: int) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    path = Path(str(item["c0_source"]))
    if not path.is_file():
        raise FileNotFoundError(path)
    expected_hash = str(item.get("c0_source_sha256", ""))
    if expected_hash and sha256_file(path) != expected_hash:
        raise RuntimeError(f"frozen C0 source hash mismatch: {path}")
    rows = [row for row in _read_jsonl(path) if int(row.get("frame", -1)) == int(frame)]
    if len(rows) != 1:
        raise RuntimeError(f"C0 row coverage is not singleton at {item['event_id']}:{frame}:{len(rows)}")
    row = rows[0]
    for flag in ("runtime_future_gt_used", "runtime_gt_read", "posthoc_gt_used"):
        if row.get(flag) is not False:
            raise RuntimeError(f"C0 row has forbidden GT flag {flag}: {item['event_id']}:{frame}")
    candidate_rows = row.get("candidate_rows")
    if not isinstance(candidate_rows, list):
        raise RuntimeError(f"C0 row candidate axis is missing: {item['event_id']}:{frame}")
    return row, [dict(value) for value in candidate_rows]


def _load_c0_rows(item: Mapping[str, Any]) -> dict[int, dict[str, Any]]:
    """Load the frozen C0 window once and expose an exact frame map."""

    path = Path(str(item["c0_source"]))
    if not path.is_file():
        raise FileNotFoundError(path)
    expected_hash = str(item.get("c0_source_sha256", ""))
    if expected_hash and sha256_file(path) != expected_hash:
        raise RuntimeError(f"frozen C0 source hash mismatch: {path}")
    rows = _read_jsonl(path)
    by_frame: dict[int, dict[str, Any]] = {}
    for row in rows:
        frame = int(row.get("frame", -1))
        if frame in by_frame:
            raise RuntimeError(f"duplicate C0 frame: {item['event_id']}:{frame}")
        for flag in ("runtime_future_gt_used", "runtime_gt_read", "posthoc_gt_used"):
            if row.get(flag) is not False:
                raise RuntimeError(f"C0 row has forbidden GT flag {flag}: {item['event_id']}:{frame}")
        if not isinstance(row.get("candidate_rows"), list):
            raise RuntimeError(f"C0 row candidate axis is missing: {item['event_id']}:{frame}")
        by_frame[frame] = dict(row)
    return by_frame


def _candidate_projection(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    keys = (
        "candidate_uid", "candidate_index", "official_raw_sam_id", "adapter_external_id",
        "box_xyxy", "feature_sha256", "solver_public_id", "public_id",
    )
    return [{key: item.get(key) for key in keys} for item in rows]


def _mask_sha256(mask: Any) -> str | None:
    if mask is None:
        return None
    value = np.asarray(mask, dtype=bool)
    return hashlib.sha256(value.tobytes()).hexdigest() if value.size else None


def _candidate_geometry_row(
    event: Mapping[str, Any],
    observation: Any,
    frame: int,
    epoch_id: str,
    *,
    session_id: str,
    target_session_scope: str,
) -> dict[str, Any]:
    """Serialize only official geometry before releasing the SAM3 session."""

    raw = getattr(observation, "raw_sam_object_id", None)
    if raw is None:
        raw = getattr(observation, "sam_object_id", None)
    if raw is None:
        raise RuntimeError("official target observation has no raw/native object ID")
    adapter = getattr(observation, "sam_object_id", raw)
    box = np.asarray(getattr(observation, "box_xyxy"), dtype=float).reshape(-1)
    if box.size != 4 or not np.all(np.isfinite(box)):
        raise ValueError("official target observation box is not finite")
    confidence = float(getattr(observation, "confidence", 0.0))
    presence = getattr(observation, "presence_score", None)
    presence_value = None if presence is None else float(presence)
    if not math.isfinite(confidence) or (
        presence_value is not None and not math.isfinite(presence_value)
    ):
        raise ValueError("official target observation score is non-finite")
    return {
        "candidate_uid": f"{event['event_id']}:target:{int(frame)}:{int(raw)}:{int(adapter)}",
        "candidate_index": 0,
        "candidate_kind": "TARGET_CORRECTION_SESSION_CANDIDATE",
        "sequence": str(event["sequence"]),
        "frame": int(frame),
        "official_raw_sam_id": int(raw),
        "adapter_external_id": int(adapter),
        "native_tid": int(adapter),
        "native_scope": str(target_session_scope),
        "native_tid_scope": str(target_session_scope),
        "box_xyxy": box.tolist(),
        "mask_sha256": _mask_sha256(getattr(observation, "mask", None)),
        "confidence": confidence,
        "presence_score": presence_value,
        "feature": None,
        "feature_dim": None,
        "feature_sha256": None,
        "feature_source": None,
        "source": str(getattr(observation, "source", "official_target_session")),
        "source_session_id": str(session_id),
        "target_session_scope": str(target_session_scope),
        "human_target_scope_public_id": int(event["n72r6_target_public_id"]),
        "correction_epoch_id": str(epoch_id),
        "public_id": None,
        "public_id_inference": False,
        "runtime_future_gt_used": False,
        "runtime_gt_read": False,
        "posthoc_gt_used": False,
    }


def _frame_row_secondary(
    event: Mapping[str, Any],
    frame: int,
    candidate_rows: list[dict[str, Any]],
    *,
    epoch_id: str,
    session: TargetSessionAuditRef,
    frame_path: Path,
    main_y_pre_hash: str,
    main_y_pre_candidate_hash: str,
) -> dict[str, Any]:
    """Build the historical frame schema from a lightweight session ref."""

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
        "source_session_id": str(session.session_id),
        "correction_epoch_id": str(epoch_id),
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


def _make_backend(
    device: str,
    *,
    max_num_objects: int = 16,
    multiplex_count: int = 16,
) -> Any:
    return __import__("sam3_intermot.backend.sam3_backend", fromlist=["Sam3Backend"]).Sam3Backend(
        checkpoint_path=str(CHECKPOINT),
        max_num_objects=int(max_num_objects),
        multiplex_count=int(multiplex_count),
        use_fa3=False,
        use_rope_real=True,
        compile=False,
        warm_up=False,
        session_expiration_sec=1200,
        output_prob_thresh=0.30,
        async_loading_frames=False,
        device=str(device),
        official_batched_grounding_batch_size=1,
        trim_past_non_cond_mem_for_eval=True,
    )


def _source_one_hot(source: str) -> list[float]:
    return [float(str(source) == name) for name in SOURCE_NAMES]


def _bootstrap_edge(
    candidate: Mapping[str, Any],
    *,
    source: str,
    anchor: np.ndarray,
    predicted_box: Sequence[float],
    current: bool,
) -> float:
    expected_source = "TARGET_SESSION_CURRENT_RAW" if current else FUTURE_FRAME_REQUERY
    if str(source) != expected_source:
        raise ValueError(f"bootstrap edge source/current mismatch: {source} current={current}")
    return supplemental_target_edge(
        candidate,
        anchor_feature=anchor,
        predicted_box=predicted_box,
        source=str(source),
    )


def _bootstrap_selection(
    *,
    c0_row: Mapping[str, Any],
    pool: Sequence[Mapping[str, Any]],
    main_rows: Sequence[Mapping[str, Any]],
    anchor: np.ndarray,
    predicted_box: Sequence[float],
    target_public_id: int,
    event_id: str,
    frame: int,
) -> dict[str, Any]:
    score_frame = build_secondary_public_score_frame(
        c0_row=c0_row,
        main_candidates=main_rows,
        pool=pool,
        target_public_id=int(target_public_id),
        anchor_feature=anchor,
        predicted_box=predicted_box,
        event_id=str(event_id),
        frame=int(frame),
    )
    state_axis = score_frame.state_axis
    public_axis = score_frame.public_axis
    target_col = score_frame.target_column
    matrix = score_frame.matrix
    target_edges = score_frame.target_edge_by_uid
    states = [
        type(
            "N72R11SecondaryState",
            (),
            {"association_state_id": state_id, "public_id": public_id},
        )()
        for state_id, public_id in zip(state_axis, public_axis)
    ]
    solver = solve_effect_assignment(
        candidate_rows=pool,
        persistent_states=states,
        fused_state_candidate_scores=matrix.T,
        source_run_id=f"n72r11:secondary:{event_id}:{frame}",
        session_id=f"n72r11:secondary:{event_id}",
        none_score=0.0,
    )
    assignment = next(
        (
            row
            for row in solver.get("assignment_rows", [])
            if row.get("public_id") is not None and int(row["public_id"]) == int(target_public_id)
        ),
        None,
    )
    target_uid = None if assignment is None else str(assignment["candidate_uid"])
    selected = next(
        (
            candidate
            for candidate in pool
            if str(candidate["candidate_uid"]) == str(target_uid)
            and str(candidate["candidate_source"]) == FUTURE_FRAME_REQUERY
        ),
        None,
    )
    target_scores = sorted(
        ((float(matrix[index, target_col]), str(candidate["candidate_uid"])) for index, candidate in enumerate(pool)),
        reverse=True,
    )
    # N72R11's frozen trigger is the causal candidate-pool margin, not the
    # target-public score-column margin.  Keep the latter only as a diagnostic;
    # using it for triggering would make the uncertainty rule depend on the
    # assignment interface being probed.
    causal_scores: list[tuple[float, str]] = []
    for index, candidate in enumerate(pool):
        feature = candidate.get("feature")
        similarity = 0.0 if feature is None else float(np.dot(_unit(feature, "causal candidate feature"), anchor))
        presence = float(candidate.get("presence_score", candidate.get("confidence", 0.0)) or 0.0)
        # The trigger is evaluated before a new probe and uses the causal
        # base score already available for the candidate.  A target-session
        # or live-requery row has no base score from the frozen B0 stream, so
        # its pre-probe causal base is zero; the bootstrap target-edge score
        # below is only for the solver after a probe has been added.
        uid = str(candidate["candidate_uid"])
        main_index = by_uid.get(uid)
        causal_base = 0.0 if main_index is None else float(base[main_index, target_col])
        causal = float(
            np.tanh(causal_base)
            + 0.30 * similarity
            + 0.15 * _box_iou(candidate["box_xyxy"], predicted_box)
            + 0.05 * float(np.clip(presence, 0.0, 1.0))
        )
        causal_scores.append((causal, str(candidate["candidate_uid"])))
    causal_scores.sort(reverse=True)
    causal_margin = float(causal_scores[0][0] - causal_scores[1][0]) if len(causal_scores) > 1 else float("inf")
    target_edge_margin = float(target_scores[0][0] - target_scores[1][0]) if len(target_scores) > 1 else float("inf")
    base_assignment = next(
        (
            row
            for row in c0_row.get("solver", {}).get("assignment_rows", [])
            if row.get("public_id") is not None and int(row["public_id"]) == int(target_public_id)
        ),
        None,
    )
    return {
        "solver": solver,
        "matrix": matrix,
        "target_uid": target_uid,
        "selected_fresh_uid": None if selected is None else str(selected["candidate_uid"]),
        "base_assignment_uid": None if base_assignment is None else str(base_assignment.get("candidate_uid")),
        "target_edge_scores": target_edges,
        "target_edge_margin": target_edge_margin,
        "causal_top1_uid": None if not causal_scores else causal_scores[0][1],
        "causal_top2_uid": None if len(causal_scores) < 2 else causal_scores[1][1],
        "causal_margin": causal_margin,
        "target_margin": causal_margin,
        "uncertain": bool(base_assignment is None or causal_margin < UNCERTAINTY_MARGIN),
        "runtime_future_gt_used": False,
    }


def _failure_path(root: Path, event_id: str) -> Path:
    path = root / "attempts" / f"{event_id}.failure.json"
    if path.exists():
        index = 2
        while True:
            candidate = root / "attempts" / f"{event_id}.failure.attempt{index}.json"
            if not candidate.exists():
                return candidate
            index += 1
    return path


def run_event(
    event_id: str,
    *,
    attempt: int,
    device: str,
    output_root: Path,
    horizon_override: int | None = None,
    enable_live: bool = True,
    max_num_objects: int = 16,
    multiplex_count: int = 16,
) -> dict[str, Any]:
    item = _load_schedule(event_id)
    secondary_frame = int(item["secondary_frame"])
    original_frame = int(item["original_event_frame"])
    target_public_id = int(item["target_public_id"])
    full_end = min(secondary_frame + HORIZON, original_frame + HORIZON)
    end_frame = full_end if horizon_override is None else min(full_end, secondary_frame + int(horizon_override))
    if end_frame <= secondary_frame:
        raise ValueError("secondary window must contain event+1")
    sequence = str(item["sequence"])
    human_box = _finite_box(item["current_target_box_posthoc_selection_only"], "scheduled simulated human box")
    sequence_paths = image_files(DATA_ROOT / "train" / sequence)
    if end_frame >= len(sequence_paths):
        raise RuntimeError(f"image coverage incomplete: {sequence}:{secondary_frame}:{end_frame}")
    c0_rows = _load_c0_rows(item)
    if secondary_frame not in c0_rows or secondary_frame + 1 not in c0_rows:
        raise RuntimeError(f"C0 window lacks secondary event/event+1: {event_id}")
    c0_event_row = c0_rows[secondary_frame]
    c0_event_candidates = [dict(value) for value in c0_event_row["candidate_rows"]]
    event = {
        "event_id": str(event_id),
        "sequence": sequence,
        "action_type": str(item["action_type"]),
        "event_frame": secondary_frame,
        "current_gt_box": human_box.astype(float).tolist(),
        "n72r6_target_public_id": target_public_id,
        "original_event_id": str(item["original_event_id"]),
        "original_event_frame": original_frame,
        "target_dataset_gt_id": int(item["target_dataset_gt_id"]),
    }
    epoch_id = f"{event_id}:correction_epoch:1"
    root = output_root / str(event_id)
    if (root / "done.json").exists():
        raise RuntimeError(f"refusing to overwrite completed secondary artifact: {root}")
    root.mkdir(parents=True, exist_ok=True)
    frames_path = root / "target_stream" / "frames.jsonl"
    mapping_path = root / "target_stream" / "target_session_frame_mapping.json"
    anchor_path = root / "target_stream" / "human_anchor.json"
    live_path = root / "live_requery" / "live_requery.json"
    started = time_now = datetime.now(timezone.utc).timestamp()
    backend: Any | None = None
    session: TargetScopedCorrectionSession | None = None
    window_handle: tempfile.TemporaryDirectory[str] | None = None
    feature_materializer: FrozenOSNetFeatureMaterializer | None = None
    controller: LiveFutureRequeryController | None = None
    live_audit_before_close: dict[str, Any] | None = None
    live_audit_after_close: dict[str, Any] | None = None
    memory_telemetry: dict[str, Any] = {
        "after_target_sam": None,
        "after_target_feature_materialization": None,
        "during_live_sam_peak": None,
    }
    try:
        backend = _make_backend(
            device,
            max_num_objects=int(max_num_objects),
            multiplex_count=int(multiplex_count),
        )
        session = TargetScopedCorrectionSession(
            backend=backend,
            event_id=event_id,
            sequence=sequence,
            public_id=target_public_id,
            event_frame=secondary_frame,
            frame_offset=secondary_frame,
        )
        window_handle, target_video_dir, frame_mapping = _materialize_event_local_window(
            sequence_paths, secondary_frame, end_frame
        )
        atomic_json(
            mapping_path,
            {
                "schema_version": "N72R11_SECONDARY_TARGET_SESSION_FRAME_MAPPING_V1",
                "event_id": event_id,
                "sequence": sequence,
                "global_start_frame": secondary_frame,
                "global_end_frame": end_frame,
                "local_start_frame": 0,
                "local_end_frame": int(end_frame - secondary_frame),
                "mode": "SYMLINK_EXACT_SECONDARY_EVENT_WINDOW",
                "mapping": frame_mapping,
                "runtime_future_gt_used": False,
            },
        )
        session.start(target_video_dir, main_y_pre_frozen=True)
        session.seed_from_human_box(human_box)
        target_session_id = str(session.session_id)
        target_session_scope = str(session.target_session_scope)
        target_geometry_rows_by_frame: dict[int, list[dict[str, Any]]] = {
            frame: [] for frame in range(secondary_frame, end_frame + 1)
        }

        def _target_stream_callback(frame: int, observations: Sequence[Any]) -> None:
            rows = [
                _candidate_geometry_row(
                    event,
                    observation,
                    int(frame),
                    epoch_id,
                    session_id=target_session_id,
                    target_session_scope=target_session_scope,
                )
                for observation in observations
            ]
            target_geometry_rows_by_frame[int(frame)] = rows

        session.propagate_to_streaming(
            end_frame,
            output_callback=_target_stream_callback,
        )
        memory_telemetry["after_target_sam"] = _cuda_memory_snapshot()
        session_audit = session.audit()
        memory_policy = backend.runtime_memory_policy()
        if not target_geometry_rows_by_frame.get(secondary_frame):
            raise RuntimeError("secondary target session produced no event-frame official candidate")
        c0_projection = _candidate_projection(c0_event_candidates)
        main_y_pre_hash = digest_json({"source": str(item["c0_source"]), "frame": secondary_frame, "row": c0_event_row})
        main_y_pre_candidate_hash = digest_json(c0_projection)

        # Phase T1 is now complete.  Keep only plain geometry/audit records;
        # no live official session or temporary SAM3 video remains while the
        # frozen machine feature extractor is constructed.
        target_session_ref = TargetSessionAuditRef(
            session_id=target_session_id,
            target_session_scope=target_session_scope,
        )
        session.close()
        session = None
        backend = None
        if window_handle is not None:
            window_handle.cleanup()
            window_handle = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        feature_materializer = FrozenOSNetFeatureMaterializer(
            device=device,
            frame_paths=sequence_paths,
        )
        anchor = feature_materializer.materialize_human_anchor(
            frame=secondary_frame,
            box_xyxy=human_box,
        )
        geometry_rows = [
            row
            for frame in range(secondary_frame, end_frame + 1)
            for row in target_geometry_rows_by_frame[frame]
        ]
        materialized_rows = feature_materializer.materialize_rows(
            geometry_rows,
            feature_source="target_session_machine_roi_feature",
        )
        memory_telemetry["after_target_feature_materialization"] = _cuda_memory_snapshot()
        target_rows_by_frame: dict[int, list[dict[str, Any]]] = {
            frame: [] for frame in range(secondary_frame, end_frame + 1)
        }
        for row in materialized_rows:
            target_rows_by_frame[int(row["frame"])].append(row)
        target_stream_rows = [
            _frame_row_secondary(
                event,
                frame,
                target_rows_by_frame[frame],
                epoch_id=epoch_id,
                session=target_session_ref,
                frame_path=sequence_paths[frame],
                main_y_pre_hash=main_y_pre_hash,
                main_y_pre_candidate_hash=main_y_pre_candidate_hash,
            )
            for frame in range(secondary_frame, end_frame + 1)
        ]
        atomic_jsonl(frames_path, target_stream_rows)
        atomic_json(
            anchor_path,
            {
                "schema_version": "N72R11_SECONDARY_HUMAN_ROI_ANCHOR_V1",
                "event_id": event_id,
                "sequence": sequence,
                "event_frame": secondary_frame,
                "public_id": target_public_id,
                "interaction_source": "simulated_from_gt",
                "not_real_human_evidence": True,
                "source": "frozen_current_event_box_only",
                "box_xyxy": human_box.astype(float).tolist(),
                "feature": np.asarray(anchor, dtype=np.float32).astype(float).tolist(),
                "feature_sha256": hashlib.sha256(np.asarray(anchor, dtype="<f4").reshape(-1).tobytes()).hexdigest(),
                "feature_source": "raw_current_frame_simulated_human_roi_osnet",
                "runtime_future_gt_used": False,
                "runtime_gt_read": False,
                "posthoc_gt_used": False,
            },
        )
        predicted_box = list(
            target_rows_by_frame[secondary_frame + 1][0]["box_xyxy"]
            if target_rows_by_frame.get(secondary_frame + 1)
            else human_box.astype(float).tolist()
        )

        live_payload: dict[str, Any] = {
            "schema_version": "N72R11_SECONDARY_LIVE_REQUERY_V1",
            "event_id": event_id,
            "sequence": sequence,
            "event_frame": secondary_frame,
            "first_checked_frame": secondary_frame + 1,
            "end_frame": end_frame,
            "enabled": bool(enable_live),
            "trigger_rule": "base_target_assignment_NONE_OR_target_margin_lt_0.25",
            "runtime_future_gt_used": False,
            "runtime_gt_read": False,
            "posthoc_gt_used": False,
            "probe_rows": [],
            "future_rows": [],
            "trigger_records": [],
        }
        if enable_live:
            _reset_cuda_peak_stats()
            def live_feature_materializer(
                rows: Sequence[Mapping[str, Any]],
            ) -> list[dict[str, Any]]:
                if feature_materializer is None:
                    raise RuntimeError("feature materializer was not initialized")
                return feature_materializer.materialize_rows(
                    rows,
                    feature_source="future_frame_requery_machine_roi_feature",
                )

            controller = LiveFutureRequeryController(
                backend_factory=lambda: _make_backend(
                    device,
                    max_num_objects=int(max_num_objects),
                    multiplex_count=int(multiplex_count),
                ),
                sequence=sequence,
                event_id=event_id,
                event_frame=secondary_frame,
                target_public_id=target_public_id,
                frame_paths=sequence_paths,
                feature_fn=None,
                post_session_feature_materializer=live_feature_materializer,
                end_frame=end_frame,
                streaming_propagation=True,
            )
            # Check every available future frame in order.  The pre-probe pool
            # includes only already emitted main/current/active rows; a newly
            # started probe is never allowed to define its own trigger.
            requery_armed = True
            for frame in range(secondary_frame + 1, end_frame + 1):
                c0_row = c0_rows.get(frame)
                if c0_row is None:
                    raise RuntimeError(f"C0 window lacks checked frame: {event_id}:{frame}")
                main_candidates = [dict(value) for value in c0_row["candidate_rows"]]
                current_rows = list(target_rows_by_frame.get(frame, []))
                active_rows = controller.active_candidates(frame)
                pre_pool, pre_audit = build_candidate_pool_with_future_requery(
                    main_candidates,
                    current_rows,
                    active_rows,
                    sequence=sequence,
                    frame=frame,
                )
                pre_selection = _bootstrap_selection(
                    c0_row=c0_row,
                    pool=pre_pool,
                    main_rows=main_candidates,
                    anchor=_unit(anchor, "human anchor"),
                    predicted_box=predicted_box,
                    target_public_id=target_public_id,
                    event_id=event_id,
                    frame=frame,
                )
                uncertain = bool(pre_selection["uncertain"])
                trigger_record: dict[str, Any] = {
                    "frame": int(frame),
                    "uncertain": uncertain,
                    "requery_armed_before": bool(requery_armed),
                    "pre_probe_pool_audit": pre_audit,
                    "pre_probe_selection": {
                        key: value
                        for key, value in pre_selection.items()
                        if key not in {"solver", "matrix"}
                    },
                    "triggered": False,
                    "runtime_future_gt_used": False,
                }
                if uncertain and requery_armed:
                    trigger_record["triggered"] = True
                    session_probe, probe_rows = controller.probe(
                        frame=frame,
                        predicted_box=predicted_box,
                        causal_state={
                            "previous_raw_sam_id": None,
                            "previous_score": 0.0,
                            "previous_uncertainty": 1.0,
                            "trusted_age": 0,
                            "source_candidate_uid": pre_selection["target_uid"],
                            "runtime_future_gt_used": False,
                            "runtime_gt_read": False,
                            "posthoc_gt_used": False,
                        },
                    )
                    probe_pool, probe_audit = build_candidate_pool_with_future_requery(
                        main_candidates,
                        current_rows,
                        probe_rows,
                        sequence=sequence,
                        frame=frame,
                    )
                    probe_selection = _bootstrap_selection(
                        c0_row=c0_row,
                        pool=probe_pool,
                        main_rows=main_candidates,
                        anchor=_unit(anchor, "human anchor"),
                        predicted_box=predicted_box,
                        target_public_id=target_public_id,
                        event_id=event_id,
                        frame=frame,
                    )
                    selected_uid = probe_selection["selected_fresh_uid"]
                    selection_audit = {
                        "selector": "N72R11_FIXED_BOOTSTRAP_CAUSAL_SELECTOR",
                        "base_assignment_uid": pre_selection["base_assignment_uid"],
                        "base_target_uid": pre_selection["target_uid"],
                        "probe_target_uid": probe_selection["target_uid"],
                        "probe_target_source": next(
                            (
                                str(candidate["candidate_source"])
                                for candidate in probe_pool
                                if str(candidate["candidate_uid"]) == str(probe_selection["target_uid"])
                            ),
                            None,
                        ),
                        "uncertainty_threshold": UNCERTAINTY_MARGIN,
                        "runtime_future_gt_used": False,
                        "public_id_inference": False,
                    }
                    future_rows = controller.commit(
                        session=session_probe,
                        selected_candidate_uid=None if selected_uid is None else str(selected_uid),
                        selection_audit=selection_audit,
                        none_score=0.0,
                        margin=float(pre_selection["target_margin"]),
                    )
                    live_payload["probe_rows"].extend(probe_rows)
                    live_payload["future_rows"].extend(future_rows)
                    trigger_record.update(
                        {
                            "probe_pool_audit": probe_audit,
                            "probe_selection": {
                                key: value
                                for key, value in probe_selection.items()
                                if key not in {"solver", "matrix"}
                            },
                            "selection_audit": selection_audit,
                            "selected_fresh_uid": None if selected_uid is None else str(selected_uid),
                            "future_row_count": len(future_rows),
                            "solver_refused_fresh": selected_uid is None,
                        }
                    )
                    requery_armed = False
                elif not uncertain:
                    requery_armed = True
                trigger_record["requery_armed_after"] = bool(requery_armed)
                live_payload["trigger_records"].append(trigger_record)
                # The next causal predicted box is the latest emitted target
                # or selected live observation, never a posthoc box.
                assigned_rows = active_rows if active_rows else current_rows
                if assigned_rows:
                    predicted_box = list(assigned_rows[0]["box_xyxy"])
            memory_telemetry["during_live_sam_peak"] = _cuda_memory_snapshot()
            live_payload["status"] = (
                "PASS_LIVE_PROBE_AND_COMMIT"
                if live_payload["future_rows"]
                else "PASS_LIVE_PROBE_NONE_OR_NO_UNCERTAINTY"
            )
        else:
            live_payload["status"] = "LIVE_DISABLED_FOR_ENGINEERING_SMOKE"
        if controller is not None:
            live_audit_before_close = controller.audit()
            controller.close()
            live_audit_after_close = controller.audit()
        live_payload["controller_audit_before_close"] = live_audit_before_close
        live_payload["controller_audit_after_close"] = live_audit_after_close
        if "status" not in live_payload:
            live_payload["status"] = "PASS_LIVE_PROBE_AND_COMMIT" if live_payload.get("future_rows") else "PASS_LIVE_PROBE_NONE"
        atomic_json(live_path, live_payload)
        done = {
            "schema_version": "N72R11_SECONDARY_INTERACTION_DONE_V1",
            "status": "PASS_N72R11_SECONDARY_INTERACTION",
            "event_id": event_id,
            "original_event_id": str(item["original_event_id"]),
            "sequence": sequence,
            "action_type": str(item["action_type"]),
            "secondary_frame": secondary_frame,
            "original_event_frame": original_frame,
            "end_frame": end_frame,
            "frame_count": len(target_stream_rows),
            "target_candidate_row_count": sum(len(row) for row in target_rows_by_frame.values()),
            "target_stream": str(frames_path),
            "target_stream_sha256": sha256_file(frames_path),
            "human_anchor": str(anchor_path),
            "human_anchor_sha256": sha256_file(anchor_path),
            "mapping": str(mapping_path),
            "mapping_sha256": sha256_file(mapping_path),
            "live_requery": str(live_path),
            "live_requery_sha256": sha256_file(live_path),
            "target_public_id": target_public_id,
            "target_dataset_gt_id": int(item["target_dataset_gt_id"]),
            "interaction_source": "simulated_from_gt",
            "not_real_human_evidence": True,
            "scheduled_box_is_current_input_only": True,
            "runtime_future_gt_used": False,
            "runtime_gt_read": False,
            "posthoc_gt_used": False,
            "candidate_tape_ref": str(item["c0_source"]),
            "candidate_tape_sha256": str(item["c0_source_sha256"]),
            "target_session_audit": session_audit,
            "runtime_memory_policy": memory_policy,
            "memory_telemetry": memory_telemetry,
            "checkpoint": str(CHECKPOINT),
            "checkpoint_sha256": sha256_file(CHECKPOINT),
            "backend_model_config": {
                "max_num_objects": int(max_num_objects),
                "multiplex_count": int(multiplex_count),
            },
            "attempt": int(attempt),
            "horizon_override": None if horizon_override is None else int(horizon_override),
            "engineering_smoke_only": horizon_override is not None,
            "elapsed_sec": datetime.now(timezone.utc).timestamp() - started,
            "created_at_utc": now_utc(),
        }
        atomic_json(root / "done.json", done)
        return done
    except Exception as exc:
        failure = {
            "schema_version": "N72R11_SECONDARY_INTERACTION_FAILURE_V1",
            "status": "FAIL_N72R11_SECONDARY_INTERACTION",
            "event_id": event_id,
            "sequence": sequence,
            "attempt": int(attempt),
            "failure_type": type(exc).__name__,
            "message": str(exc),
            "traceback": traceback.format_exc(),
            "runtime_future_gt_used": False,
            "historical_outputs_modified": False,
            "backend_model_config": {
                "max_num_objects": int(max_num_objects),
                "multiplex_count": int(multiplex_count),
            },
            "memory_telemetry": memory_telemetry,
            "created_at_utc": now_utc(),
        }
        atomic_json(_failure_path(output_root, event_id), failure)
        raise
    finally:
        if controller is not None:
            try:
                controller.close()
            except Exception:
                pass
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
        if window_handle is not None:
            window_handle.cleanup()
        del session
        del backend
        del controller
        del feature_materializer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--event-id", required=True)
    parser.add_argument("--attempt", type=int, default=1)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--horizon", type=int, default=None)
    parser.add_argument("--disable-live", action="store_true")
    parser.add_argument("--max-num-objects", type=int, default=16)
    parser.add_argument("--multiplex-count", type=int, default=16)
    args = parser.parse_args()
    try:
        result = run_event(
            args.event_id,
            attempt=int(args.attempt),
            device=str(args.device),
            output_root=args.output_root,
            horizon_override=None if args.horizon is None else int(args.horizon),
            enable_live=not bool(args.disable_live),
            max_num_objects=int(args.max_num_objects),
            multiplex_count=int(args.multiplex_count),
        )
        print(json.dumps({"status": result["status"], "event_id": result["event_id"], "frame_count": result["frame_count"]}, ensure_ascii=False))
        return 0
    except Exception as exc:
        print(json.dumps({"status": "FAIL_N72R11_SECONDARY_INTERACTION", "error_type": type(exc).__name__, "error": str(exc)}, ensure_ascii=False))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
