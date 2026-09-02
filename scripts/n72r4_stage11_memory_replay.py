#!/usr/bin/env python3
"""CPU-only M0--M4 replay on the official N72R4 corrected candidate stream.

Stage 09 deliberately kept the official SAM3 future stream free of inferred
public IDs.  Stage 10 added a posthoc, explicit-NONE adapter for NO versus
current-frame correction.  This stage keeps that adapter's public-ID
authority and runs the frozen CCAM memory variants on the *same* corrected
stream.  It is a posthoc mechanism replay, not a replacement for the official
SAM3 runtime and not a production authorization path.

The implementation has two important constraints:

* every candidate is assigned either to an explicit persistent public ID or
  to its own explicit NONE column through ``solve_effect_assignment``;
* only the target public-ID score row receives the memory term.  The complete
  state x candidate matrix, assignment audit, mapping, and causal boundary
  are persisted for every frame before any GT file is opened.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import traceback
from typing import Any, Iterable

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sam3_intermot.association.appearance_memory import AppearanceMemory  # noqa: E402
from sam3_intermot.association.effect_assignment import solve_effect_assignment  # noqa: E402
from sam3_intermot.evaluation.interaction_effect_metrics import (  # noqa: E402
    BOOTSTRAP_REPETITIONS,
    BOOTSTRAP_SEED,
    metric_record,
    sequence_cluster_bootstrap,
)
from scripts.n72r4_stage10_cpu_analysis import (  # noqa: E402
    IOU_THRESHOLD,
    M0_ROOT,
    OUT,
    PAIR_MANIFEST,
    STAGE16_EVENT_ROOT,
    State,
    atomic_json,
    atomic_jsonl,
    association_score,
    box_iou,
    center,
    current_post_anchor,
    event_file,
    feature_digest,
    finite_feature,
    initialise_b1,
    load_events,
    load_stage16,
    load_stage18_public_map,
    load_branch_rows,
    now_utc,
    prestate_rows_by_public,
    read_json,
    read_jsonl,
    row_view,
    sha256_file,
    state_copy,
    update_state,
)


VARIANTS = (
    "M0_CURRENT_FRAME_CORRECTION_ONLY",
    "M1_HUMAN_EMA_PROTOTYPE",
    "M2_POSITIVE_HUMAN_ANCHORS",
    "M3_NEGATIVE_COMPETITOR_BANK",
    "M4_RELIABILITY_AGE_ADMISSION",
)
MEMORY_VARIANTS = set(VARIANTS[1:])
HORIZONS = (20, 50, 100)
MEMORY_HUMAN_WEIGHT = 1.0
MEMORY_MACHINE_WEIGHT = 0.35
MEMORY_DECAY_FRAMES = 120.0
M4_MIN_RELIABILITY = 0.75
M4_MAX_AGE = 80
NONE_SCORE = 0.0
STAGE_NAME = "11_CORRECTED_STREAM_M0_M4_MEMORY_REPLAY"


@dataclass(frozen=True)
class SolverState:
    """The two independent identity axes required by the exact solver."""

    association_state_id: int
    public_id: int


def _resolved(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def _json_digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    ).hexdigest()


def _empty_components() -> dict[str, float]:
    return {"prototype": 0.0, "positive": 0.0, "negative": 0.0, "total": 0.0}


def _component_total(raw: dict[str, float], variant: str) -> float:
    if variant == "M1_HUMAN_EMA_PROTOTYPE":
        return float(raw["prototype"])
    if variant == "M2_POSITIVE_HUMAN_ANCHORS":
        return float(raw["prototype"] + raw["positive"])
    if variant in {"M3_NEGATIVE_COMPETITOR_BANK", "M4_RELIABILITY_AGE_ADMISSION"}:
        return float(raw["prototype"] + raw["positive"] + raw["negative"])
    if variant == "M0_CURRENT_FRAME_CORRECTION_ONLY":
        return 0.0
    raise ValueError(f"unknown memory variant: {variant}")


def _load_prestate_axis(event_id: str) -> tuple[dict[int, int], Path]:
    path = OUT / "event_prestate" / "attempt5" / event_id / "persistent_runtime_snapshot.json"
    snapshot = read_json(path)
    if snapshot.get("runtime_future_gt_used") is not False:
        raise RuntimeError(f"persistent prestate used future GT: {event_id}")
    payload = snapshot.get("payload")
    if not isinstance(payload, dict) or not isinstance(payload.get("public_to_state"), dict):
        raise RuntimeError(f"persistent prestate lacks public_to_state authority: {event_id}")
    mapping = {int(public): int(state) for public, state in payload["public_to_state"].items()}
    if not mapping or len(mapping) != len(set(mapping)) or len(mapping) != len(set(mapping.values())):
        raise RuntimeError(f"persistent prestate public/state axes are not one-to-one: {event_id}")
    return mapping, path


def _validate_feature_digest(feature: np.ndarray, expected: str, context: str) -> None:
    actual = feature_digest(feature)
    if str(expected) != actual:
        raise RuntimeError(f"feature digest mismatch at {context}: expected={expected} actual={actual}")


def _build_memory(
    *,
    event: dict[str, Any],
    stage16: dict[str, Any],
    target_post: dict[str, Any],
    post_candidates: list[dict[str, Any]],
    variant: str,
) -> tuple[AppearanceMemory | None, dict[str, Any]]:
    event_id = str(event["event_id"])
    event_frame = int(event["event_frame"])
    target_public = int(stage16["persistent_identity"]["public_id"])
    if variant == "M0_CURRENT_FRAME_CORRECTION_ONLY":
        return None, {
            "variant": variant,
            "memory_enabled": False,
            "memory_write": False,
            "memory_read": False,
            "reason": "M0_CURRENT_FRAME_CORRECTION_ONLY",
        }

    memory = AppearanceMemory(
        human_weight=MEMORY_HUMAN_WEIGHT,
        machine_weight=MEMORY_MACHINE_WEIGHT,
        decay_frames=MEMORY_DECAY_FRAMES,
        reliability_threshold=0.0,
    )
    post_feature = finite_feature(target_post["feature"])
    human_records = (stage16.get("appearance_memory") or {}).get("positive") or []
    if not human_records:
        raise RuntimeError(f"stage16 has no human positive anchor: {event_id}")
    human_record = human_records[0]
    if human_record.get("source") not in {"human", "current_frame_simulated_human_box_roi"}:
        raise RuntimeError(f"stage16 positive anchor is not human-source evidence: {event_id}")
    human_feature = finite_feature(human_record.get("feature"))
    write_audit = (stage16.get("causal_audit") or {}).get("memory_write") or {}
    _validate_feature_digest(human_feature, str(write_audit.get("feature_sha256")), f"{event_id}/human_memory_write")
    if int(write_audit.get("frame", -1)) != event_frame or write_audit.get("current_frame_write_hidden") is not True:
        raise RuntimeError(f"stage16 human memory write boundary is invalid: {event_id}")
    if write_audit.get("write_after_spatial_correction") is not True:
        raise RuntimeError(f"stage16 memory write was not after spatial correction: {event_id}")
    if not memory.update_from_machine(
        target_public,
        event_frame,
        post_feature,
        confidence=float(target_post.get("confidence", 1.0)),
    ):
        raise RuntimeError(f"machine prototype seed failed: {event_id}/{variant}")
    competitor_features = [
        finite_feature(candidate["feature"])
        for candidate in post_candidates
        if int(candidate["official_raw_sam_id"]) != int(target_post["official_raw_sam_id"])
    ]
    if not memory.update_from_human(
        target_public,
        event_frame,
        human_feature,
        quality=float(write_audit.get("quality", 1.0)),
        competing_embeddings=(competitor_features if variant in {"M3_NEGATIVE_COMPETITOR_BANK", "M4_RELIABILITY_AGE_ADMISSION"} else None),
        write_event_id=event_id,
    ):
        raise RuntimeError(f"human memory replay write failed: {event_id}/{variant}")
    serialized = memory.snapshot()
    return memory, {
        "variant": variant,
        "memory_enabled": True,
        "memory_write": True,
        "memory_read": False,
        "write_frame": event_frame,
        "visible_from_frame": event_frame + 1,
        "target_public_id": target_public,
        "machine_seed_feature_sha256": feature_digest(post_feature),
        "human_anchor_feature_sha256": feature_digest(human_feature),
        "human_anchor_source": str(human_record.get("source")),
        "competitor_count": len(competitor_features),
        "memory_state_sha256": _json_digest(serialized),
        "human_weight": MEMORY_HUMAN_WEIGHT,
        "machine_weight": MEMORY_MACHINE_WEIGHT,
        "decay_frames": MEMORY_DECAY_FRAMES,
        "reliability_threshold": 0.0,
        "runtime_future_gt_used": False,
    }


def _appearance_components(
    *,
    memory: AppearanceMemory | None,
    variant: str,
    target_public: int,
    candidate_feature: np.ndarray,
    event_frame: int,
    frame: int,
) -> tuple[dict[str, float], bool, str, dict[str, float] | None]:
    if variant == "M0_CURRENT_FRAME_CORRECTION_ONLY" or memory is None:
        return _empty_components(), False, "M0_MEMORY_DISABLED", None
    if frame <= event_frame:
        return _empty_components(), False, "EVENT_FRAME_READ_FORBIDDEN", None
    record = memory.records.get(int(target_public))
    if record is None:
        return _empty_components(), False, "TARGET_MEMORY_RECORD_MISSING", None
    raw = {
        key: float(value)
        for key, value in memory._score_components(int(target_public), candidate_feature, int(frame)).items()
    }
    if variant == "M4_RELIABILITY_AGE_ADMISSION":
        age = int(frame) - int(event_frame)
        if float(record.reliability) < M4_MIN_RELIABILITY:
            return _empty_components(), False, "RELIABILITY_BELOW_ADMISSION", raw
        if age > M4_MAX_AGE:
            return _empty_components(), False, "AGE_ABOVE_ADMISSION", raw
    total = _component_total(raw, variant)
    applied = {key: float(value) for key, value in raw.items()}
    applied["total"] = float(total)
    return applied, True, "ADMITTED", raw


def _solver_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            **row,
            "candidate_uid": str(row["candidate_key"]),
        }
        for row in rows
    ]


def _mapping_from_solver(artifact: dict[str, Any], rows: list[dict[str, Any]]) -> tuple[list[int | None], list[str]]:
    assignments = artifact.get("assignment_rows")
    if not isinstance(assignments, list) or len(assignments) != len(rows):
        raise RuntimeError("exact solver assignment row count does not match candidate rows")
    by_uid = {str(item.get("candidate_uid")): item for item in assignments}
    if len(by_uid) != len(assignments):
        raise RuntimeError("exact solver returned duplicate candidate UIDs")
    public_ids: list[int | None] = []
    statuses: list[str] = []
    for row in rows:
        item = by_uid.get(str(row["candidate_key"]))
        if item is None:
            raise RuntimeError(f"exact solver omitted candidate {row['candidate_key']}")
        value = item.get("public_id")
        public_ids.append(None if value is None else int(value))
        statuses.append(str(item.get("status")))
    non_none = [value for value in public_ids if value is not None]
    if len(non_none) != len(set(non_none)):
        raise RuntimeError("exact solver returned duplicate public IDs")
    return public_ids, statuses


def _candidate_output_rows(
    rows: list[dict[str, Any]],
    public_ids: list[int | None],
    statuses: list[str],
) -> list[dict[str, Any]]:
    output = []
    for row, public_id, status in zip(rows, public_ids, statuses):
        output.append(
            {
                "candidate_uid": str(row["candidate_key"]),
                "candidate_index": int(row["candidate_index"]),
                "official_raw_sam_id": int(row["official_raw_sam_id"]),
                "raw_native_id": int(row["raw_native_id"]),
                "native_tid": int(row["native_tid"]),
                "adapter_external_id": int(row["adapter_external_id"]),
                "adapter_visible_id": int(row["adapter_visible_id"]),
                "box_xyxy": [float(value) for value in row["box_xyxy"]],
                "feature_sha256": str(row["feature_sha256"]),
                "confidence": float(row["confidence"]),
                "source": str(row["source"]),
                "public_id": None if public_id is None else int(public_id),
                "assignment_status": str(status),
            }
        )
    return output


def _state_axis(public_to_state: dict[int, int], states: dict[int, State]) -> tuple[list[int], list[SolverState]]:
    publics = sorted(int(public) for public in states)
    if set(publics) != set(public_to_state):
        raise RuntimeError(f"persistent public axis changed during replay: states={publics} prestate={sorted(public_to_state)}")
    solver_states = [
        SolverState(association_state_id=int(public_to_state[public]), public_id=public)
        for public in publics
    ]
    if len({item.association_state_id for item in solver_states}) != len(solver_states):
        raise RuntimeError("persistent association state axis duplicated")
    if len({item.public_id for item in solver_states}) != len(solver_states):
        raise RuntimeError("persistent public axis duplicated")
    return publics, solver_states


def _future_frame(
    *,
    event: dict[str, Any],
    branch_row: dict[str, Any],
    states: dict[int, State],
    public_to_state: dict[int, int],
    memory: AppearanceMemory | None,
    memory_summary: dict[str, Any],
    variant: str,
) -> dict[str, Any]:
    event_id = str(event["event_id"])
    event_frame = int(event["event_frame"])
    frame = int(branch_row["frame"])
    rows = row_view("B1_CURRENT_FRAME_CORRECTION", event_id, branch_row)
    publics, solver_states = _state_axis(public_to_state, states)
    base = np.asarray(
        [[association_score(states[public], row, frame) for row in rows] for public in publics],
        dtype=np.float64,
    )
    if base.shape != (len(publics), len(rows)):
        raise RuntimeError(f"base score shape mismatch: {event_id}/{variant}/{frame}/{base.shape}")
    if not np.isfinite(base).all():
        raise RuntimeError(f"base score nonfinite: {event_id}/{variant}/{frame}")
    appearance = np.zeros_like(base, dtype=np.float64)
    target_public = int(event["target_public_id"] if "target_public_id" in event else (event.get("target_public_id") or 0))
    # The frozen event manifest has no separate target_public_id field; Stage
    # 16 is the authority for this value and is attached by the caller.
    target_public = int(event["_target_public_id"])
    target_index = publics.index(target_public) if target_public in publics else None
    components_by_candidate: list[dict[str, Any]] = []
    target_admitted = False
    target_reason = "NO_TARGET_STATE"
    if target_index is not None and variant in MEMORY_VARIANTS:
        for candidate_index, row in enumerate(rows):
            components, admitted, reason, raw = _appearance_components(
                memory=memory,
                variant=variant,
                target_public=target_public,
                candidate_feature=finite_feature(row["feature"]),
                event_frame=event_frame,
                frame=frame,
            )
            appearance[target_index, candidate_index] = float(components["total"])
            components_by_candidate.append(
                {
                    "candidate_uid": str(row["candidate_key"]),
                    "candidate_index": int(row["candidate_index"]),
                    "components": components,
                    "ungated_components": raw,
                    "admitted": bool(admitted),
                    "reason": str(reason),
                }
            )
            target_admitted = target_admitted or admitted
            if admitted:
                target_reason = "ADMITTED"
            elif target_reason == "NO_TARGET_STATE":
                target_reason = str(reason)
    else:
        target_reason = "M0_MEMORY_DISABLED" if variant == "M0_CURRENT_FRAME_CORRECTION_ONLY" else "TARGET_STATE_NOT_PRESENT"
        components_by_candidate = [
            {
                "candidate_uid": str(row["candidate_key"]),
                "candidate_index": int(row["candidate_index"]),
                "components": _empty_components(),
                "ungated_components": None,
                "admitted": False,
                "reason": target_reason,
            }
            for row in rows
        ]
    fused = base.copy()
    if target_index is not None:
        fused[target_index, :] = base[target_index, :] + appearance[target_index, :]
    if not np.isfinite(fused).all():
        raise RuntimeError(f"fused score nonfinite: {event_id}/{variant}/{frame}")
    non_target_indices = [index for index in range(base.shape[0]) if index != target_index]
    non_target_equal = (
        bool(np.array_equal(base[non_target_indices, :], fused[non_target_indices, :]))
        if non_target_indices
        else True
    )
    if not non_target_equal:
        raise RuntimeError(f"target-scoped non-target state row changed: {event_id}/{variant}/{frame}")

    solver = solve_effect_assignment(
        candidate_rows=_solver_rows(rows),
        persistent_states=solver_states,
        fused_state_candidate_scores=fused,
        source_run_id=f"n72r4-stage11:{event_id}:{variant}:{frame}",
        session_id=f"n72r4-stage11:{event_id}:{variant}",
        none_score=NONE_SCORE,
    )
    public_ids, statuses = _mapping_from_solver(solver, rows)
    assigned_publics = {int(value) for value in public_ids if value is not None}
    for public in publics:
        if public in assigned_publics:
            index = public_ids.index(public)
            update_state(states[public], rows[index], frame)
        else:
            states[public].status = "LOST"
    mapping_rows = _candidate_output_rows(rows, public_ids, statuses)
    if len(assigned_publics) != len([value for value in public_ids if value is not None]):
        raise RuntimeError(f"future public mapping is not one-to-one: {event_id}/{variant}/{frame}")
    return {
        "schema_version": "N72R4_STAGE11_CORRECTED_STREAM_FRAME_V1",
        "event_id": event_id,
        "sequence": str(event["sequence"]),
        "action_type": str(event["action_type"]),
        "branch": "B1_CURRENT_FRAME_CORRECTION",
        "candidate_stream_kind": "OFFICIAL_SAM3_FUTURE_PROPAGATION",
        "candidate_stream_sha256": str(branch_row["frame_hash_sha256"]),
        "event_frame": event_frame,
        "frame": frame,
        "frame_horizon": frame - event_frame,
        "phase": "FUTURE_ASSOCIATION",
        "variant": variant,
        "candidate_order": [int(row["candidate_index"]) for row in rows],
        "candidate_rows": mapping_rows,
        "association_state_axis": [int(item.association_state_id) for item in solver_states],
        "public_id_order": publics,
        "target_public_id": target_public,
        "target_state_index": target_index,
        "base_score_matrix": base.astype(float).tolist(),
        "appearance_score_matrix": appearance.astype(float).tolist(),
        "appearance_score_deltas": appearance.astype(float).tolist(),
        "fused_score_matrix": fused.astype(float).tolist(),
        "solver_executed": True,
        "solver": solver,
        "assignment_public_ids": [None if value is None else int(value) for value in public_ids],
        "assignment_status": statuses,
        "assignment_map": {
            str(row["candidate_key"]): None if public_id is None else int(public_id)
            for row, public_id in zip(rows, public_ids)
        },
        "memory_write": False,
        "memory_read": bool(variant in MEMORY_VARIANTS and target_index is not None and bool(rows)),
        "memory_admitted": bool(target_admitted),
        "memory_read_reason": "NO_CANDIDATES" if not rows and variant in MEMORY_VARIANTS else target_reason,
        "memory_components_by_candidate": components_by_candidate,
        "memory_age": frame - event_frame,
        "memory_summary": memory_summary,
        "causal_boundary": {
            "event_frame_memory_read": False,
            "current_frame_write_hidden": True,
            "first_memory_visible_frame": event_frame + 1,
            "memory_read_frame": frame if variant in MEMORY_VARIANTS and rows else None,
            "runtime_future_gt_used": False,
        },
        "public_state_axis_after_frame": [
            {
                "public_id": int(states[public].public_id),
                "association_state_id": int(public_to_state[public]),
                "last_frame": int(states[public].last_frame),
                "last_native": None if states[public].last_native is None else int(states[public].last_native),
                "status": str(states[public].status),
            }
            for public in sorted(states)
        ],
        "target_scoped_non_target_rows_bitwise_equal": bool(non_target_equal),
        "solver_coupled_collateral": False,
        "runtime_future_gt_used": False,
        "runtime_gt_read": False,
        "posthoc_gt_used": False,
    }


def _event_frame_artifact(
    *,
    event: dict[str, Any],
    event_row: dict[str, Any],
    corrected_states: dict[int, State],
    public_to_state: dict[int, int],
    public_by_raw: dict[int, int],
    target_public: int,
    target_post: dict[str, Any],
    public_to_post: dict[int, int],
    memory_summary: dict[str, Any],
    variant: str,
) -> dict[str, Any]:
    event_id = str(event["event_id"])
    event_frame = int(event["event_frame"])
    rows = row_view("B1_CURRENT_FRAME_CORRECTION", event_id, event_row)
    mapping: list[int | None] = []
    statuses: list[str] = []
    for row in rows:
        public = public_by_raw.get(int(row["official_raw_sam_id"]))
        mapping.append(None if public is None else int(public))
        statuses.append("PRESTATE_PERSISTENT_MAPPING" if public is not None else "EXPLICIT_NONE_UNMAPPED_PRESTATE_CANDIDATE")
    non_none = [value for value in mapping if value is not None]
    if len(non_none) != len(set(non_none)):
        raise RuntimeError(f"event-frame prestate mapping duplicated public ID: {event_id}")
    publics = sorted(corrected_states)
    if set(publics) != set(public_to_state):
        raise RuntimeError(f"event-frame public axis mismatch: {event_id}")
    base = np.asarray(
        [[association_score(corrected_states[public], row, event_frame) for row in rows] for public in publics],
        dtype=np.float64,
    )
    if not np.isfinite(base).all():
        raise RuntimeError(f"event-frame base score nonfinite: {event_id}/{variant}")
    zero = np.zeros_like(base, dtype=np.float64)
    post_raw = {str(public): int(raw) for public, raw in sorted(public_to_post.items())}
    return {
        "schema_version": "N72R4_STAGE11_CORRECTED_STREAM_FRAME_V1",
        "event_id": event_id,
        "sequence": str(event["sequence"]),
        "action_type": str(event["action_type"]),
        "branch": "B1_CURRENT_FRAME_CORRECTION",
        "candidate_stream_kind": "OFFICIAL_SAM3_EVENT_FRAME_PRE_CORRECTION_STREAM",
        "candidate_stream_sha256": str(event_row["y_pre_semantic_hash"]),
        "event_frame": event_frame,
        "frame": event_frame,
        "frame_horizon": 0,
        "phase": "CURRENT_FRAME_CORRECTION_AND_MEMORY_WRITE",
        "variant": variant,
        "candidate_order": [int(row["candidate_index"]) for row in rows],
        "candidate_rows": _candidate_output_rows(rows, mapping, statuses),
        "association_state_axis": [int(public_to_state[public]) for public in publics],
        "public_id_order": publics,
        "target_public_id": int(target_public),
        "target_state_index": publics.index(int(target_public)) if int(target_public) in publics else None,
        "base_score_matrix": base.astype(float).tolist(),
        "appearance_score_matrix": zero.tolist(),
        "appearance_score_deltas": zero.tolist(),
        "fused_score_matrix": base.astype(float).tolist(),
        "solver_executed": False,
        "assignment_public_ids": [None if value is None else int(value) for value in mapping],
        "assignment_status": statuses,
        "assignment_map": {
            str(row["candidate_key"]): None if public is None else int(public)
            for row, public in zip(rows, mapping)
        },
        "correction": {
            "official_post_observation": {
                "box_xyxy": [float(value) for value in (event.get("_post_observation") or {}).get("box_xyxy", [])],
                "source": str((event.get("_post_observation") or {}).get("source", "human_correction")),
                "is_human_verified": bool((event.get("_post_observation") or {}).get("is_human_verified", True)),
            },
            "target_post_raw_id": int(target_post["official_raw_sam_id"]),
            "target_public_id": int(target_public),
            "public_to_post_raw": post_raw,
            "spatial_correction_before_memory_write": True,
        },
        "memory_write": bool(variant in MEMORY_VARIANTS),
        "memory_read": False,
        "memory_admitted": False,
        "memory_read_reason": "EVENT_FRAME_READ_FORBIDDEN",
        "memory_components_by_candidate": [],
        "memory_summary": memory_summary,
        "causal_boundary": {
            "event_frame_memory_read": False,
            "current_frame_write_hidden": True,
            "first_memory_visible_frame": event_frame + 1,
            "memory_read_frame": None,
            "write_after_spatial_correction": True,
            "runtime_future_gt_used": False,
        },
        "public_state_axis_after_frame": [
            {
                "public_id": int(corrected_states[public].public_id),
                "association_state_id": int(public_to_state[public]),
                "last_frame": int(corrected_states[public].last_frame),
                "last_native": None if corrected_states[public].last_native is None else int(corrected_states[public].last_native),
                "status": str(corrected_states[public].status),
            }
            for public in publics
        ],
        "target_scoped_non_target_rows_bitwise_equal": True,
        "solver_coupled_collateral": False,
        "runtime_future_gt_used": False,
        "runtime_gt_read": False,
        "posthoc_gt_used": False,
    }


def _run_event(event: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    event_id = str(event["event_id"])
    stage16 = load_stage16(event_id)
    event["_target_public_id"] = int(stage16["persistent_identity"]["public_id"])
    target_public = int(event["_target_public_id"])
    event["_post_observation"] = dict(stage16["official_current_correction"]["post_observation"])
    public_to_state, prestate_path = _load_prestate_axis(event_id)
    public_by_raw = load_stage18_public_map(str(event["sequence"]), int(event["event_frame"]))
    corrected_rows = load_branch_rows(M0_ROOT, event_id)
    event_row = corrected_rows[int(event["event_frame"])]
    target_post, post_candidates = current_post_anchor(stage16, event_row)
    corrected_states, public_to_post = initialise_b1(
        event_row,
        post_candidates,
        target_post,
        public_by_raw,
        target_public,
        event_id,
    )
    if set(corrected_states) != set(public_to_state):
        raise RuntimeError(f"corrected current-frame states changed persistent public axis: {event_id}")
    all_rows: list[dict[str, Any]] = []
    variant_summaries: list[dict[str, Any]] = []
    for variant in VARIANTS:
        states = {public: state_copy(value) for public, value in corrected_states.items()}
        memory, memory_summary = _build_memory(
            event=event,
            stage16=stage16,
            target_post=target_post,
            post_candidates=post_candidates,
            variant=variant,
        )
        event_artifact = _event_frame_artifact(
            event=event,
            event_row=event_row,
            corrected_states=states,
            public_to_state=public_to_state,
            public_by_raw=public_by_raw,
            target_public=target_public,
            target_post=target_post,
            public_to_post=public_to_post,
            memory_summary=memory_summary,
            variant=variant,
        )
        all_rows.append(event_artifact)
        for frame in range(int(event["event_frame"]) + 1, int(event["event_frame"]) + 101):
            all_rows.append(
                _future_frame(
                    event=event,
                    branch_row=corrected_rows[frame],
                    states=states,
                    public_to_state=public_to_state,
                    memory=memory,
                    memory_summary=memory_summary,
                    variant=variant,
                )
            )
        variant_summaries.append(
            {
                "variant": variant,
                "frame_count": 101,
                "memory_summary": memory_summary,
                "final_state_public_ids": sorted(states),
            }
        )
    all_rows.sort(key=lambda row: (int(row["frame"]), VARIANTS.index(str(row["variant"]))))
    return all_rows, {
        "event_id": event_id,
        "sequence": str(event["sequence"]),
        "action_type": str(event["action_type"]),
        "event_frame": int(event["event_frame"]),
        "future_frame_count": 100,
        "variant_count": len(VARIANTS),
        "variant_frame_count": len(all_rows),
        "prestate_snapshot": str(prestate_path),
        "prestate_snapshot_sha256": sha256_file(prestate_path),
        "target_public_id": target_public,
        "target_post_raw_id": int(target_post["official_raw_sam_id"]),
        "public_to_post_raw": {str(public): int(raw) for public, raw in sorted(public_to_post.items())},
        "variants": variant_summaries,
        "runtime_future_gt_used": False,
        "interaction_source": "simulated_from_gt",
        "not_real_human_evidence": True,
    }


def _candidate_signature(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        str(row.get("candidate_uid")),
        int(row.get("candidate_index", -1)),
        int(row.get("official_raw_sam_id", -1)),
        int(row.get("adapter_external_id", -1)),
        tuple(float(value) for value in row.get("box_xyxy", [])),
        str(row.get("feature_sha256")),
    )


def _validate_runtime_artifacts(
    events: list[dict[str, Any]],
    artifact_root: Path,
) -> dict[str, Any]:
    checked_frames = 0
    checked_rows = 0
    non_target_cells = 0
    files: list[dict[str, Any]] = []
    forbidden_runtime_keys = {
        "dataset_gt_id",
        "gt_box",
        "future_gt",
        "public_id_inference_result",
        "reward",
        "future_identity_error",
        "dataset_identity",
    }
    for event in events:
        event_id = str(event["event_id"])
        path = artifact_root / f"{event_id}.jsonl"
        rows = read_jsonl(path)
        if len(rows) != len(VARIANTS) * 101:
            raise RuntimeError(f"event artifact row count mismatch: {event_id}/{len(rows)}")
        by_key: dict[tuple[str, int], dict[str, Any]] = {}
        for row in rows:
            key = (str(row.get("variant")), int(row.get("frame", -1)))
            if key in by_key:
                raise RuntimeError(f"duplicate event/variant/frame artifact key: {event_id}/{key}")
            by_key[key] = row
            if row.get("runtime_future_gt_used") is not False or row.get("runtime_gt_read") is not False or row.get("posthoc_gt_used") is not False:
                raise RuntimeError(f"runtime GT boundary failed: {event_id}/{key}")
            forbidden = forbidden_runtime_keys.intersection(row)
            if forbidden:
                raise RuntimeError(f"forbidden posthoc/GT field entered runtime artifact: {event_id}/{sorted(forbidden)}")
            if str(row.get("event_id")) != event_id:
                raise RuntimeError(f"event ID mismatch: {event_id}/{key}")
            candidate_rows = row.get("candidate_rows")
            if not isinstance(candidate_rows, list):
                raise RuntimeError(f"candidate rows missing: {event_id}/{key}")
            uids = [str(item.get("candidate_uid")) for item in candidate_rows]
            if len(uids) != len(set(uids)):
                raise RuntimeError(f"candidate UID duplicate: {event_id}/{key}")
            public_values = [item.get("public_id") for item in candidate_rows if item.get("public_id") is not None]
            if len(public_values) != len(set(public_values)):
                raise RuntimeError(f"candidate public assignment duplicate: {event_id}/{key}")
            public_axis = [int(value) for value in row.get("public_id_order", [])]
            state_axis = [int(value) for value in row.get("association_state_axis", [])]
            base = np.asarray(row.get("base_score_matrix"), dtype=np.float64)
            appearance = np.asarray(row.get("appearance_score_matrix"), dtype=np.float64)
            fused = np.asarray(row.get("fused_score_matrix"), dtype=np.float64)
            expected_shape = (len(public_axis), len(candidate_rows))
            if base.shape != expected_shape or appearance.shape != expected_shape or fused.shape != expected_shape:
                raise RuntimeError(f"score matrix shape mismatch: {event_id}/{key}/{base.shape}/{expected_shape}")
            if not (np.isfinite(base).all() and np.isfinite(appearance).all() and np.isfinite(fused).all()):
                raise RuntimeError(f"nonfinite runtime score matrix: {event_id}/{key}")
            target = row.get("target_public_id")
            target_index = row.get("target_state_index")
            if target_index is not None and int(target_index) >= len(public_axis):
                raise RuntimeError(f"target state index invalid: {event_id}/{key}")
            non_target = [index for index in range(len(public_axis)) if index != target_index]
            if non_target and not np.array_equal(base[non_target, :], fused[non_target, :]):
                raise RuntimeError(f"non-target score row changed: {event_id}/{key}")
            if str(row.get("variant")) == "M0_CURRENT_FRAME_CORRECTION_ONLY" and not np.array_equal(appearance, np.zeros_like(appearance)):
                raise RuntimeError(f"M0 appearance matrix is nonzero: {event_id}/{key}")
            if int(row.get("frame_horizon", -1)) == 0:
                if row.get("phase") != "CURRENT_FRAME_CORRECTION_AND_MEMORY_WRITE" or row.get("solver_executed") is not False:
                    raise RuntimeError(f"event-frame phase/solver invalid: {event_id}/{key}")
                if row.get("memory_read") is not False or row.get("causal_boundary", {}).get("event_frame_memory_read") is not False:
                    raise RuntimeError(f"event-frame memory read violated: {event_id}/{key}")
                if row.get("causal_boundary", {}).get("first_memory_visible_frame") != int(event["event_frame"]) + 1:
                    raise RuntimeError(f"event-frame first-visible boundary invalid: {event_id}/{key}")
            else:
                if row.get("phase") != "FUTURE_ASSOCIATION" or int(row.get("frame_horizon")) != int(row.get("frame")) - int(event["event_frame"]):
                    raise RuntimeError(f"future phase/horizon invalid: {event_id}/{key}")
                if row.get("solver_executed") is not True:
                    raise RuntimeError(f"future solver was not executed: {event_id}/{key}")
                if row.get("causal_boundary", {}).get("first_memory_visible_frame") != int(event["event_frame"]) + 1:
                    raise RuntimeError(f"future memory boundary invalid: {event_id}/{key}")
                if str(row.get("variant")) in MEMORY_VARIANTS and row.get("memory_age") != int(row["frame"]) - int(event["event_frame"]):
                    raise RuntimeError(f"memory age invalid: {event_id}/{key}")
                assignment_rows = (row.get("solver") or {}).get("assignment_rows")
                if not isinstance(assignment_rows, list) or len(assignment_rows) != len(candidate_rows):
                    raise RuntimeError(f"exact solver audit missing: {event_id}/{key}")
        expected_keys = {(variant, frame) for variant in VARIANTS for frame in range(int(event["event_frame"]), int(event["event_frame"]) + 101)}
        if set(by_key) != expected_keys:
            raise RuntimeError(f"missing event/variant/frame keys: {event_id}")
        for frame in range(int(event["event_frame"]), int(event["event_frame"]) + 101):
            signatures = []
            for variant in VARIANTS:
                row = by_key[(variant, frame)]
                signatures.append(tuple(_candidate_signature(item) for item in row["candidate_rows"]))
                checked_rows += 1
                checked_frames += int(variant == VARIANTS[0])
                base = np.asarray(row["base_score_matrix"], dtype=np.float64)
                non_target_cells += int(base.size - base.shape[1] if base.ndim == 2 and base.shape[0] else 0)
            if len(set(signatures)) != 1:
                raise RuntimeError(f"corrected candidate stream changed across variants: {event_id}/{frame}")
        files.append(
            {
                "event_id": event_id,
                "path": str(path),
                "sha256": sha256_file(path),
                "row_count": len(rows),
            }
        )
    return {
        "status": "PASS_STAGE11_RUNTIME_ARTIFACT_VALIDATION",
        "event_count": len(events),
        "variant_count": len(VARIANTS),
        "frames_per_event": 101,
        "checked_frames": checked_frames,
        "checked_variant_frame_rows": checked_rows,
        "non_target_score_cells_checked": non_target_cells,
        "candidate_stream_shared_across_variants": True,
        "explicit_none_retained": True,
        "persistent_public_axis_retained": True,
        "runtime_future_gt_used": False,
        "gt_loaded_in_worker": False,
        "files": files,
    }


def _load_gt(sequence: str) -> dict[int, dict[int, dict[str, Any]]]:
    path = Path("/data1/LWR/vranlee/DATASETS/JDE/dancetrack") / "train" / sequence / "gt/gt.txt"
    if not path.is_file():
        raise FileNotFoundError(path)
    result: dict[int, dict[int, dict[str, Any]]] = defaultdict(dict)
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            parts = [item.strip() for item in line.split(",")]
            if len(parts) < 6:
                raise ValueError(f"malformed GT row {path}:{line_number}")
            frame_one_based, dataset_id = int(parts[0]), int(parts[1])
            x, y, width, height = [float(item) for item in parts[2:6]]
            result[frame_one_based - 1][dataset_id] = {
                "box": [x, y, x + width, y + height],
                "visibility": None if len(parts) <= 8 else float(parts[8]),
            }
    return result


def _candidate_best(row: dict[str, Any], box: list[float]) -> tuple[float, dict[str, Any] | None]:
    scored = sorted(
        ((box_iou(candidate["box_xyxy"], box), candidate) for candidate in row.get("candidate_rows", [])),
        key=lambda item: (-item[0], str(item[1].get("candidate_uid"))),
    )
    return (float(scored[0][0]), scored[0][1]) if scored else (0.0, None)


def _public_iou(row: dict[str, Any], public_id: int, box: list[float]) -> float:
    return max(
        (
            box_iou(candidate["box_xyxy"], box)
            for candidate in row.get("candidate_rows", [])
            if candidate.get("public_id") is not None and int(candidate["public_id"]) == int(public_id)
        ),
        default=0.0,
    )


def _assignment_map(row: dict[str, Any]) -> dict[str, int | None]:
    result: dict[str, int | None] = {}
    for candidate in row.get("candidate_rows", []):
        value = candidate.get("public_id")
        result[str(candidate["candidate_uid"])] = None if value is None else int(value)
    return result


def _wrong_reassociation(row: dict[str, Any], target_public: int, target_box: list[float], gt_frame: dict[int, dict[str, Any]], target_gid: int) -> bool:
    target_candidate = next(
        (
            candidate
            for candidate in row.get("candidate_rows", [])
            if candidate.get("public_id") is not None and int(candidate["public_id"]) == int(target_public)
        ),
        None,
    )
    if target_candidate is None:
        return False
    for gid, item in gt_frame.items():
        if int(gid) == int(target_gid):
            continue
        if box_iou(target_candidate["box_xyxy"], item["box"]) >= IOU_THRESHOLD:
            return True
    return False


def _metric_template() -> dict[str, float | int]:
    return {
        "evaluated_frames": 0,
        "target_iou_sum": 0.0,
        "baseline_m0_target_iou_sum": 0.0,
        "target_correct_frames": 0,
        "baseline_m0_target_correct_frames": 0,
        "target_missing_frames": 0,
        "baseline_m0_target_missing_frames": 0,
        "target_identity_error_frames": 0,
        "baseline_m0_identity_error_frames": 0,
        "wrong_reassociation_frames": 0,
        "baseline_m0_wrong_reassociation_frames": 0,
        "candidate_present_frames": 0,
        "baseline_m0_candidate_present_frames": 0,
        "id_switch_count": 0,
        "baseline_m0_id_switch_count": 0,
        "recorrection_opportunity_count": 0,
        "baseline_m0_recorrection_opportunity_count": 0,
        "assignment_change_count": 0,
        "assignment_change_true_correct_count": 0,
        "assignment_change_true_incorrect_count": 0,
        "assignment_change_directional_improvement_count": 0,
        "assignment_change_directional_regression_count": 0,
        "assignment_change_neutral_count": 0,
        "solver_coupled_collateral_count": 0,
        "protected_compared": 0,
        "protected_regression_count": 0,
        "protected_improvement_count": 0,
        "identity_error_reduction_sum": 0.0,
        "delta_iou_sum": 0.0,
        "composite_utility_secondary_sum": 0.0,
    }


def _add_metric(destination: dict[str, Any], source: dict[str, Any]) -> None:
    for key in _metric_template():
        destination[key] = destination.get(key, 0) + source.get(key, 0)


def _finalize_metric(metric: dict[str, Any]) -> dict[str, Any]:
    frames = int(metric["evaluated_frames"])
    denom = max(1, frames)
    metric["target_mean_iou"] = float(metric["target_iou_sum"] / denom)
    metric["baseline_m0_target_mean_iou"] = float(metric["baseline_m0_target_iou_sum"] / denom)
    metric["delta_iou_mean_vs_m0"] = float(metric["delta_iou_sum"] / denom)
    metric["identity_error_reduction"] = float(metric["identity_error_reduction_sum"] / denom)
    metric["future_identity_error"] = float(metric["target_identity_error_frames"] / denom)
    metric["baseline_m0_future_identity_error"] = float(metric["baseline_m0_identity_error_frames"] / denom)
    metric["missing_rate"] = float(metric["target_missing_frames"] / denom)
    metric["baseline_m0_missing_rate"] = float(metric["baseline_m0_target_missing_frames"] / denom)
    metric["wrong_reassociation_rate"] = float(metric["wrong_reassociation_frames"] / denom)
    metric["baseline_m0_wrong_reassociation_rate"] = float(metric["baseline_m0_wrong_reassociation_frames"] / denom)
    metric["candidate_recall"] = float(metric["candidate_present_frames"] / denom)
    metric["baseline_m0_candidate_recall"] = float(metric["baseline_m0_candidate_present_frames"] / denom)
    metric["id_switch_rate"] = float(metric["id_switch_count"] / denom)
    metric["baseline_m0_id_switch_rate"] = float(metric["baseline_m0_id_switch_count"] / denom)
    metric["recorrection_rate"] = float(metric["recorrection_opportunity_count"] / denom)
    metric["baseline_m0_recorrection_rate"] = float(metric["baseline_m0_recorrection_opportunity_count"] / denom)
    protected = int(metric["protected_compared"])
    metric["protected_regression_rate"] = None if protected == 0 else float(metric["protected_regression_count"] / protected)
    metric["protected_improvement_rate"] = None if protected == 0 else float(metric["protected_improvement_count"] / protected)
    metric["assignment_change_rate"] = float(metric["assignment_change_count"] / denom)
    return metric


def _protected_public_by_gt(event_row: dict[str, Any], gt_frame: dict[int, dict[str, Any]], target_gid: int) -> dict[int, int]:
    result: dict[int, int] = {}
    for gid, item in gt_frame.items():
        if int(gid) == int(target_gid):
            continue
        best_iou, best = _candidate_best(event_row, item["box"])
        if best is not None and best_iou >= IOU_THRESHOLD and best.get("public_id") is not None:
            public = int(best["public_id"])
            if public != int(event_row["target_public_id"]):
                result[int(gid)] = public
    return result


def _score_event_variant(
    *,
    event: dict[str, Any],
    variant: str,
    rows_by_variant_frame: dict[tuple[str, int], dict[str, Any]],
    gt_frames: dict[int, dict[int, dict[str, Any]]],
) -> dict[str, Any]:
    event_id = str(event["event_id"])
    event_frame = int(event["event_frame"])
    target_public = int(event["_target_public_id"])
    target_gid = int(event["dataset_gt_id"])
    m0_event = rows_by_variant_frame[("M0_CURRENT_FRAME_CORRECTION_ONLY", event_frame)]
    protected = _protected_public_by_gt(m0_event, gt_frames.get(event_frame, {}), target_gid)
    horizons: dict[str, Any] = {}
    for horizon in HORIZONS:
        metric = _metric_template()
        frame_details: list[dict[str, Any]] = []
        previous_treatment_pid: int | None = None
        previous_baseline_pid: int | None = None
        previous_treatment_error = False
        previous_baseline_error = False
        for frame in range(event_frame + 1, event_frame + horizon + 1):
            gt_target = gt_frames.get(frame, {}).get(target_gid)
            if gt_target is None:
                continue
            treatment = rows_by_variant_frame[(variant, frame)]
            baseline = rows_by_variant_frame[("M0_CURRENT_FRAME_CORRECTION_ONLY", frame)]
            target_box = gt_target["box"]
            treatment_iou = _public_iou(treatment, target_public, target_box)
            baseline_iou = _public_iou(baseline, target_public, target_box)
            treatment_best_iou, treatment_best = _candidate_best(treatment, target_box)
            baseline_best_iou, baseline_best = _candidate_best(baseline, target_box)
            treatment_correct = bool(treatment_iou >= IOU_THRESHOLD)
            baseline_correct = bool(baseline_iou >= IOU_THRESHOLD)
            treatment_missing = not any(
                item.get("public_id") is not None and int(item["public_id"]) == target_public
                for item in treatment.get("candidate_rows", [])
            )
            baseline_missing = not any(
                item.get("public_id") is not None and int(item["public_id"]) == target_public
                for item in baseline.get("candidate_rows", [])
            )
            treatment_wrong = _wrong_reassociation(treatment, target_public, target_box, gt_frames.get(frame, {}), target_gid)
            baseline_wrong = _wrong_reassociation(baseline, target_public, target_box, gt_frames.get(frame, {}), target_gid)
            treatment_observed_pid = None
            if treatment_best is not None and treatment_best_iou >= IOU_THRESHOLD and treatment_best.get("public_id") is not None:
                treatment_observed_pid = int(treatment_best["public_id"])
            baseline_observed_pid = None
            if baseline_best is not None and baseline_best_iou >= IOU_THRESHOLD and baseline_best.get("public_id") is not None:
                baseline_observed_pid = int(baseline_best["public_id"])
            treatment_switch = bool(
                previous_treatment_pid is not None
                and treatment_observed_pid is not None
                and treatment_observed_pid != previous_treatment_pid
            )
            baseline_switch = bool(
                previous_baseline_pid is not None
                and baseline_observed_pid is not None
                and baseline_observed_pid != previous_baseline_pid
            )
            if treatment_observed_pid is not None:
                previous_treatment_pid = treatment_observed_pid
            if baseline_observed_pid is not None:
                previous_baseline_pid = baseline_observed_pid
            treatment_recorrect = bool((not treatment_correct) and not previous_treatment_error)
            baseline_recorrect = bool((not baseline_correct) and not previous_baseline_error)
            previous_treatment_error = not treatment_correct
            previous_baseline_error = not baseline_correct
            changed_uids = {
                uid
                for uid in set(_assignment_map(treatment)) | set(_assignment_map(baseline))
                if _assignment_map(treatment).get(uid) != _assignment_map(baseline).get(uid)
            }
            target_related_uids = {
                uid
                for uid in changed_uids
                if _assignment_map(treatment).get(uid) == target_public
                or _assignment_map(baseline).get(uid) == target_public
            }
            assignment_changed = bool(changed_uids)
            record = metric_record(
                baseline_iou=baseline_iou,
                treatment_iou=treatment_iou,
                baseline_correct=baseline_correct,
                treatment_correct=treatment_correct,
                assignment_changed=assignment_changed,
            )
            metric["evaluated_frames"] += 1
            metric["target_iou_sum"] += treatment_iou
            metric["baseline_m0_target_iou_sum"] += baseline_iou
            metric["target_correct_frames"] += int(treatment_correct)
            metric["baseline_m0_target_correct_frames"] += int(baseline_correct)
            metric["target_missing_frames"] += int(treatment_missing)
            metric["baseline_m0_target_missing_frames"] += int(baseline_missing)
            metric["target_identity_error_frames"] += int(not treatment_correct)
            metric["baseline_m0_identity_error_frames"] += int(not baseline_correct)
            metric["wrong_reassociation_frames"] += int(treatment_wrong)
            metric["baseline_m0_wrong_reassociation_frames"] += int(baseline_wrong)
            metric["candidate_present_frames"] += int(treatment_best_iou >= IOU_THRESHOLD)
            metric["baseline_m0_candidate_present_frames"] += int(baseline_best_iou >= IOU_THRESHOLD)
            metric["id_switch_count"] += int(treatment_switch)
            metric["baseline_m0_id_switch_count"] += int(baseline_switch)
            metric["recorrection_opportunity_count"] += int(treatment_recorrect)
            metric["baseline_m0_recorrection_opportunity_count"] += int(baseline_recorrect)
            metric["assignment_change_count"] += int(assignment_changed)
            metric["assignment_change_true_correct_count"] += int(record["true_correct_crossing"])
            metric["assignment_change_true_incorrect_count"] += int(record["true_incorrect_crossing"])
            metric["assignment_change_directional_improvement_count"] += int(record["directional_improvement"])
            metric["assignment_change_directional_regression_count"] += int(record["directional_regression"])
            metric["assignment_change_neutral_count"] += int(record["assignment_change_type"] == "NEUTRAL_CHANGE")
            metric["solver_coupled_collateral_count"] += int(bool(changed_uids - target_related_uids))
            metric["identity_error_reduction_sum"] += float(record["identity_error_reduction"])
            metric["delta_iou_sum"] += float(record["delta_iou"])
            metric["composite_utility_secondary_sum"] += float(record["composite_utility_secondary"])
            for protected_gid, protected_pid in protected.items():
                baseline_other = gt_frames.get(frame, {}).get(int(protected_gid))
                if baseline_other is None:
                    continue
                baseline_other_iou = _public_iou(baseline, protected_pid, baseline_other["box"])
                treatment_other_iou = _public_iou(treatment, protected_pid, baseline_other["box"])
                metric["protected_compared"] += 1
                metric["protected_regression_count"] += int(
                    baseline_other_iou >= IOU_THRESHOLD and treatment_other_iou < IOU_THRESHOLD
                )
                metric["protected_improvement_count"] += int(
                    treatment_other_iou >= IOU_THRESHOLD and baseline_other_iou < IOU_THRESHOLD
                )
            frame_details.append(
                {
                    "frame": frame,
                    "target_iou": float(treatment_iou),
                    "baseline_m0_target_iou": float(baseline_iou),
                    "target_correct": treatment_correct,
                    "baseline_m0_target_correct": baseline_correct,
                    "target_missing": treatment_missing,
                    "baseline_m0_target_missing": baseline_missing,
                    "candidate_best_iou": float(treatment_best_iou),
                    "baseline_m0_candidate_best_iou": float(baseline_best_iou),
                    "wrong_reassociation": treatment_wrong,
                    "baseline_m0_wrong_reassociation": baseline_wrong,
                    "id_switch": treatment_switch,
                    "baseline_m0_id_switch": baseline_switch,
                    "recorrection_opportunity": treatment_recorrect,
                    "baseline_m0_recorrection_opportunity": baseline_recorrect,
                    "assignment_changed": assignment_changed,
                    "assignment_change_type": record["assignment_change_type"],
                    "identity_error_reduction": record["identity_error_reduction"],
                    "delta_iou": record["delta_iou"],
                    "solver_coupled_collateral": bool(changed_uids - target_related_uids),
                    "runtime_future_gt_used": False,
                    "posthoc_gt_used": True,
                }
            )
        horizons[str(horizon)] = {**_finalize_metric(metric), "frame_details": frame_details}
    return {
        "event_id": event_id,
        "sequence": str(event["sequence"]),
        "action_type": str(event["action_type"]),
        "event_frame": event_frame,
        "target_public_id": target_public,
        "target_dataset_gt_id": target_gid,
        "variant": variant,
        "protected_public_by_gt_posthoc": {str(gid): int(public) for gid, public in sorted(protected.items())},
        "horizons": horizons,
        "runtime_future_gt_used": False,
        "gt_usage": "posthoc_only_after_runtime_artifact_validation",
        "interaction_source": "simulated_from_gt",
        "not_real_human_evidence": True,
    }


def _aggregate(
    event_metrics: list[dict[str, Any]],
    *,
    variant: str,
    horizon: int,
    action: str | None = None,
) -> dict[str, Any]:
    selected = [
        item
        for item in event_metrics
        if item["variant"] == variant and (action is None or item["action_type"] == action)
    ]
    values = [item["horizons"][str(horizon)] for item in selected]
    total = _metric_template()
    for value in values:
        _add_metric(total, value)
    result = _finalize_metric(total)
    result["event_count"] = len(values)
    result["independent_sequence_count"] = len({str(item["sequence"]) for item in selected})
    sequence_values: dict[str, list[float]] = defaultdict(list)
    for item in selected:
        sequence_values[str(item["sequence"])].append(float(item["horizons"][str(horizon)]["identity_error_reduction"]))
    result["sequence_cluster_bootstrap_95ci"] = sequence_cluster_bootstrap(
        sequence_values,
        seed=BOOTSTRAP_SEED,
        repetitions=BOOTSTRAP_REPETITIONS,
    )
    result["metric_semantics"] = {
        "identity_error_reduction": "+1 wrong_to_correct, -1 correct_to_wrong, 0 unchanged",
        "delta_iou": "treatment_Mx_minus_M0",
        "primary_effect": "identity_error_reduction",
        "composite_utility": "secondary_only",
    }
    return result


def _posthoc_score(
    events: list[dict[str, Any]],
    artifact_root: Path,
    runtime_validation: dict[str, Any],
) -> dict[str, Any]:
    # This is intentionally the first GT access in this script.
    gt_by_sequence = {str(event["sequence"]): _load_gt(str(event["sequence"])) for event in events}
    event_metrics: list[dict[str, Any]] = []
    runtime_rows: dict[tuple[str, str, int], dict[str, Any]] = {}
    for event in events:
        event_id = str(event["event_id"])
        for row in read_jsonl(artifact_root / f"{event_id}.jsonl"):
            runtime_rows[(event_id, str(row["variant"]), int(row["frame"]))] = row
    for event in events:
        event_id = str(event["event_id"])
        rows_for_event = {
            (variant, frame): runtime_rows[(event_id, variant, frame)]
            for variant in VARIANTS
            for frame in range(int(event["event_frame"]), int(event["event_frame"]) + 101)
        }
        event["_target_public_id"] = int(event.get("_target_public_id") or read_json(STAGE16_EVENT_ROOT / f"{event_id}.json")["persistent_identity"]["public_id"])
        for variant in VARIANTS:
            event_metrics.append(
                _score_event_variant(
                    event=event,
                    variant=variant,
                    rows_by_variant_frame=rows_for_event,
                    gt_frames=gt_by_sequence[str(event["sequence"])],
                )
            )
    aggregate: dict[str, dict[str, Any]] = {variant: {} for variant in VARIANTS}
    for variant in VARIANTS:
        for horizon in HORIZONS:
            aggregate[variant][str(horizon)] = _aggregate(event_metrics, variant=variant, horizon=horizon)
    action_aggregate: dict[str, dict[str, dict[str, Any]]] = {}
    actions = sorted({str(event["action_type"]) for event in events})
    for action in actions:
        action_aggregate[action] = {variant: {} for variant in VARIANTS}
        for variant in VARIANTS:
            for horizon in HORIZONS:
                action_aggregate[action][variant][str(horizon)] = _aggregate(
                    event_metrics,
                    variant=variant,
                    horizon=horizon,
                    action=action,
                )
    gate_by_variant: dict[str, Any] = {}
    for variant in ("M2_POSITIVE_HUMAN_ANCHORS", "M3_NEGATIVE_COMPETITOR_BANK", "M4_RELIABILITY_AGE_ADMISSION"):
        metric = aggregate[variant]["20"]
        ci = metric["sequence_cluster_bootstrap_95ci"]
        gate_by_variant[variant] = {
            "h20_identity_error_reduction": metric["identity_error_reduction"],
            "h20_sequence_cluster_ci": ci,
            "strict_lower_ci_gt_zero": bool(ci.get("lower") is not None and float(ci["lower"]) > 0.0),
            "protected_regression_count": int(metric["protected_regression_count"]),
            "protected_regression_zero": int(metric["protected_regression_count"]) == 0,
        }
    gate_pass = bool(
        all(value["strict_lower_ci_gt_zero"] and value["protected_regression_zero"] for value in gate_by_variant.values())
        and runtime_validation.get("status") == "PASS_STAGE11_RUNTIME_ARTIFACT_VALIDATION"
    )
    no_vs_m0_path = OUT / "candidate_recall" / "no_vs_m0_candidate_recall.json"
    no_vs_m0 = read_json(no_vs_m0_path) if no_vs_m0_path.is_file() else None
    return {
        "schema_version": "N72R4_STAGE11_CORRECTED_STREAM_M0_M4_RESULTS_V1",
        "status": "PASS_STAGE11_POSTHOC_SCORING",
        "event_count": len(events),
        "independent_sequence_count": len({str(event["sequence"]) for event in events}),
        "variants": list(VARIANTS),
        "horizons": list(HORIZONS),
        "event_metrics": event_metrics,
        "aggregate": aggregate,
        "by_action": action_aggregate,
        "gate": {
            "status": "PASS_FUTURE_EFFECT" if gate_pass else "FAIL_FUTURE_EFFECT",
            "by_variant": gate_by_variant,
            "all_required_variants_pass": gate_pass,
            "primary_horizon": 20,
            "strict_sequence_cluster_bootstrap": True,
            "protected_identity_regression_required_zero": True,
            "production_authorized": False,
            "reason_if_not_authorized": "simulated_from_gt evidence and/or strict future-effect gate not satisfied",
        },
        "candidate_recall_decomposition": {
            "no_vs_m0_path": str(no_vs_m0_path),
            "no_vs_m0": no_vs_m0,
            "interpretation": "Stage10 NO versus M0 is candidate availability; Stage11 Mx versus M0 is identity/effect scoring on the corrected official stream.",
        },
        "runtime_validation": runtime_validation,
        "runtime_future_gt_used": False,
        "gt_usage": "posthoc_only_after_runtime_artifact_validation",
        "interaction_source": "simulated_from_gt",
        "not_real_human_evidence": True,
        "scientific_result": "CORRECTED_STREAM_M0_M4_EFFECT_RESULT_NOT_REAL_HUMAN_PRODUCTION_EVIDENCE",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--event-limit", type=int, default=0)
    parser.add_argument("--artifact-root", type=Path, default=OUT / "mechanism_probe" / "corrected_stream_attempt1")
    parser.add_argument("--manifest-path", type=Path, default=OUT / "mechanism_probe" / "corrected_stream_manifest_attempt1.json")
    parser.add_argument("--metrics-path", type=Path, default=OUT / "metrics" / "corrected_stream_m1_m4_results_attempt1.json")
    parser.add_argument("--status-path", type=Path, default=OUT / "stage_status" / "stage_11_status.json")
    parser.add_argument("--attempt", default="attempt1")
    args = parser.parse_args()
    artifact_root = _resolved(args.artifact_root)
    manifest_path = _resolved(args.manifest_path)
    metrics_path = _resolved(args.metrics_path)
    status_path = _resolved(args.status_path)
    started = now_utc()
    try:
        if artifact_root.exists() and any(artifact_root.iterdir()):
            raise RuntimeError(f"artifact root is not empty; choose a new attempt path: {artifact_root}")
        if manifest_path.exists() or metrics_path.exists() or status_path.exists():
            raise RuntimeError("one or more requested Stage11 output paths already exist; refusing overwrite")
        all_events = load_events()
        events = all_events[: int(args.event_limit)] if int(args.event_limit) > 0 else all_events
        if not events:
            raise RuntimeError("no frozen events selected")
        artifact_root.mkdir(parents=True, exist_ok=True)
        event_manifests: list[dict[str, Any]] = []
        for event in events:
            rows, summary = _run_event(event)
            path = artifact_root / f"{event['event_id']}.jsonl"
            atomic_jsonl(path, rows)
            event_manifests.append(summary)
        runtime_validation = _validate_runtime_artifacts(events, artifact_root)
        runtime_manifest = {
            "schema_version": "N72R4_STAGE11_CORRECTED_STREAM_MANIFEST_V1",
            "status": runtime_validation["status"],
            "stage": STAGE_NAME,
            "attempt": str(args.attempt),
            "event_count": len(events),
            "event_count_frozen": len(all_events),
            "full_frozen_event_set": len(events) == len(all_events),
            "variant_count": len(VARIANTS),
            "variants": list(VARIANTS),
            "frames_per_event": 101,
            "event_manifests": event_manifests,
            "runtime_validation": runtime_validation,
            "pair_manifest": str(PAIR_MANIFEST),
            "pair_manifest_sha256": sha256_file(PAIR_MANIFEST),
            "official_corrected_root": str(M0_ROOT),
            "candidate_stream_kind": "OFFICIAL_SAM3_FUTURE_PROPAGATION",
            "persistent_public_id_authority": "N72R3_STAGE18_PERSISTENT_RUNTIME_PRESTATE",
            "candidate_index_to_public_id": False,
            "official_raw_sam_id_to_public_id": False,
            "adapter_id_to_public_id": False,
            "new_public_ids_created": False,
            "explicit_none_retained": True,
            "runtime_future_gt_used": False,
            "gt_loaded_in_worker": False,
            "interaction_source": "simulated_from_gt",
            "not_real_human_evidence": True,
            "scientific_result": "RUNTIME_ARTIFACT_ONLY_UNTIL_POSTHOC_VALIDATION",
        }
        atomic_json(manifest_path, runtime_manifest)
        result = _posthoc_score(events, artifact_root, runtime_validation)
        result["inputs"] = {
            "runtime_manifest": str(manifest_path),
            "runtime_manifest_sha256": sha256_file(manifest_path),
            "event_manifest_sha256": sha256_file(Path("/data2/usr_for_deadline/SAM3_InterMOT_N72R3/worktree/outputs/N72R3/simulation/real_event_manifest.json")),
            "pair_manifest_sha256": sha256_file(PAIR_MANIFEST),
            "official_corrected_root": str(M0_ROOT),
            "official_corrected_root_events": len(events),
            "protocol": str(OUT / "protocol.json"),
            "protocol_sha256": sha256_file(OUT / "protocol.json"),
        }
        atomic_json(metrics_path, result)
        full = bool(len(events) == len(all_events))
        stage_status = {
            "schema_version": "N72R4_STAGE_STATUS_V1",
            "stage": STAGE_NAME,
            "status": "PASS_STAGE11_CORRECTED_STREAM_MEMORY_REPLAY" if full else "PASS_STAGE11_TARGETED_SMOKE",
            "started_at_utc": started,
            "finished_at_utc": now_utc(),
            "attempt": str(args.attempt),
            "event_count": len(events),
            "event_count_frozen": len(all_events),
            "independent_sequence_count": len({str(event["sequence"]) for event in events}),
            "variant_count": len(VARIANTS),
            "runtime_manifest": str(manifest_path),
            "runtime_manifest_sha256": sha256_file(manifest_path),
            "metrics": str(metrics_path),
            "metrics_sha256": sha256_file(metrics_path),
            "runtime_validation": runtime_validation,
            "future_effect_gate": result["gate"],
            "candidate_recall_decomposition": result["candidate_recall_decomposition"],
            "candidate_stream_shared_across_variants": True,
            "persistent_public_id_authority": "N72R3_STAGE18_PERSISTENT_RUNTIME_PRESTATE",
            "explicit_none_retained": True,
            "runtime_future_gt_used": False,
            "gt_loaded_in_worker": False,
            "posthoc_gt_used": True,
            "interaction_source": "simulated_from_gt",
            "not_real_human_evidence": True,
            "production_authorized": False,
        }
        atomic_json(status_path, stage_status)
        print(
            json.dumps(
                {
                    "status": stage_status["status"],
                    "events": len(events),
                    "runtime_rows": runtime_validation["checked_variant_frame_rows"],
                    "metrics": str(metrics_path),
                    "future_effect_gate": result["gate"]["status"],
                },
                sort_keys=True,
            ),
            flush=True,
        )
        return 0
    except Exception as exc:
        failure_root = OUT / "attempts" / "stage11"
        failure_root.mkdir(parents=True, exist_ok=True)
        failure_path = failure_root / f"{str(args.attempt)}_failure_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.json"
        atomic_json(
            failure_path,
            {
                "schema_version": "N72R4_FAILURE_V1",
                "stage": STAGE_NAME,
                "attempt": str(args.attempt),
                "status": "FAIL",
                "started_at_utc": started,
                "finished_at_utc": now_utc(),
                "error_type": type(exc).__name__,
                "error": str(exc),
                "traceback": traceback.format_exc(),
                "artifact_root": str(artifact_root),
                "runtime_future_gt_used": False,
            },
        )
        print(json.dumps({"status": "FAIL", "failure_artifact": str(failure_path), "error": str(exc)}, sort_keys=True), flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
