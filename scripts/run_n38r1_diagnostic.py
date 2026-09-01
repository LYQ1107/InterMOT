#!/usr/bin/env python3
"""Rebuild the N38 mechanism diagnostic from N38R1 lossless sidecars only.

The sidecars contain event-frame and future candidate audits.  This script
loads one event's five variants at a time, immediately compacts the large
feature/mask payloads into scalar diagnostics, and releases the source JSON
before moving to the next event.  No dataset GT is opened here: target
visibility/correctness relative to future labels is explicitly marked as a
posthoc-only field and remains unavailable in this sidecar-only stage.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import statistics
import traceback
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in __import__("sys").path:
    __import__("sys").path.insert(0, str(ROOT))

from scripts.n36_real_eval_common import atomic_json, atomic_jsonl  # noqa: E402


VARIANTS = ("M0", "M1", "M2", "M3", "M4")
BRANCHES = ("memory_write=False", "memory_write=True")
N38_PROTOCOL_PATH = ROOT / "outputs" / "n38" / "diagnostic" / "diagnostic_protocol.json"
DEFAULT_MANIFEST = ROOT / "outputs" / "n38r1" / "sidecar_manifest.json"
DEFAULT_SCHEMA_AUDIT = ROOT / "outputs" / "n38r1" / "sidecar_schema_audit.json"
DEFAULT_OUT = ROOT / "outputs" / "n38r1" / "diagnostic"
REQUIRED_AUDIT_FIELDS = (
    "candidates",
    "candidate_native_ids",
    "candidate_order",
    "public_id_order",
    "scores",
    "base_scores_before_appearance",
    "appearance_memory_scores",
    "appearance_score_deltas",
    "fused_scores",
    "public_id_score_matrix",
    "public_id_base_score_matrix",
    "public_id_appearance_score_matrix",
    "public_id_fused_score_matrix",
    "assignment",
    "assignment_after_scope",
    "assignment_pairs",
    "assignment_pairs_after_scope",
    "public_id_to_native_tid",
    "candidate_records",
    "candidate_rank_by_state",
    "hungarian_cost_audit",
    "target_state_top_two",
)


def frozen_protocol() -> dict[str, Any]:
    payload = json.loads(N38_PROTOCOL_PATH.read_text(encoding="utf-8"))
    if payload.get("protocol_hash") != "02b807b2166061cdf00894c40b4d21e91074696822adfabd50f405ed8b8e27a6":
        raise RuntimeError(f"unexpected frozen N38 protocol hash: {payload.get('protocol_hash')}")
    return payload


def finite_vector(values: Any) -> list[float] | None:
    if not isinstance(values, list):
        return None
    try:
        result = [float(value) for value in values]
    except (TypeError, ValueError):
        return None
    return result if result and all(math.isfinite(value) for value in result) else None


def cosine(left: Any, right: Any) -> float | None:
    a = finite_vector(left)
    b = finite_vector(right)
    if a is None or b is None or len(a) != len(b):
        return None
    an = math.sqrt(sum(value * value for value in a))
    bn = math.sqrt(sum(value * value for value in b))
    if an <= 1.0e-12 or bn <= 1.0e-12:
        return None
    return float(sum(x * y for x, y in zip(a, b)) / (an * bn))


def finite_matrix(values: Any) -> list[list[float]] | None:
    if not isinstance(values, list) or any(not isinstance(row, list) for row in values):
        return None
    result: list[list[float]] = []
    width: int | None = None
    for row in values:
        converted = finite_vector(row)
        if converted is None:
            if row == []:
                converted = []
            else:
                return None
        if width is None:
            width = len(converted)
        elif len(converted) != width:
            return None
        result.append(converted)
    return result


def matrix_shape(values: Any) -> tuple[int, int] | None:
    matrix = finite_matrix(values)
    if matrix is None:
        return None
    return len(matrix), len(matrix[0]) if matrix else 0


def target_state_vector(audit: dict[str, Any], target_public_id: int) -> list[float] | None:
    state_ids = audit.get("public_id_order")
    fused = finite_matrix(audit.get("fused_scores"))
    if not isinstance(state_ids, list) or fused is None or target_public_id not in state_ids:
        return None
    target_index = state_ids.index(target_public_id)
    if any(target_index >= len(row) for row in fused):
        return None
    return [float(row[target_index]) for row in fused]


def top_two_from_audit(audit: dict[str, Any]) -> dict[str, Any]:
    top = audit.get("target_state_top_two")
    if isinstance(top, dict):
        return {
            "target_public_id": top.get("target_public_id"),
            "top1_candidate_public_id": top.get("top1_candidate_public_id"),
            "top2_candidate_public_id": top.get("top2_candidate_public_id"),
            "top1_score": top.get("top1_score"),
            "top2_score": top.get("top2_score"),
            "normalized_margin": top.get("top1_top2_normalized_margin"),
            "distinct_public_id": top.get("top2_distinct_public_id"),
            "near_tie": (
                bool(top.get("top2_distinct_public_id"))
                and isinstance(top.get("top1_top2_normalized_margin"), (int, float))
                and math.isfinite(float(top["top1_top2_normalized_margin"]))
                and float(top["top1_top2_normalized_margin"]) <= 0.05
            ),
        }
    return {
        "target_public_id": None,
        "top1_candidate_public_id": None,
        "top2_candidate_public_id": None,
        "top1_score": None,
        "top2_score": None,
        "normalized_margin": None,
        "distinct_public_id": None,
        "near_tie": None,
    }


def compact_audit(
    audit: Any,
    *,
    source_row_metadata: dict[str, Any],
    event: dict[str, Any],
    frame: int,
    is_event_frame: bool,
    branch: str,
) -> dict[str, Any]:
    if not isinstance(audit, dict):
        raise RuntimeError(f"missing audit at frame={frame} branch={branch}")
    missing = [field for field in REQUIRED_AUDIT_FIELDS if field not in audit]
    if missing:
        raise RuntimeError(f"audit missing fields at frame={frame}: {missing}")
    candidate_ids = audit.get("candidate_public_ids")
    native_ids = audit.get("candidate_native_ids")
    order = audit.get("candidate_order")
    state_ids = audit.get("public_id_order")
    fused = finite_matrix(audit.get("fused_scores"))
    if not all(isinstance(value, list) for value in (candidate_ids, native_ids, order, state_ids)):
        raise RuntimeError(f"audit id vectors invalid at frame={frame}")
    if fused is None or len(fused) != len(candidate_ids) or any(len(row) != len(state_ids) for row in fused):
        raise RuntimeError(f"audit fused matrix invalid at frame={frame}")
    if len(native_ids) != len(candidate_ids) or len(order) != len(candidate_ids):
        raise RuntimeError(f"audit candidate vectors misaligned at frame={frame}")
    assignments = audit.get("assignment_after_scope")
    if not isinstance(assignments, list) or len(assignments) != len(candidate_ids):
        raise RuntimeError(f"audit assignment invalid at frame={frame}")
    target_public_id = int(event.get("public_id", event.get("canonical_public_id")))
    target_scores = target_state_vector(audit, target_public_id)
    human = event.get("human_embedding")
    cosine_values: list[float] = []
    target_source_count = 0
    mask_hash_count = 0
    for candidate in audit.get("candidate_records", []):
        if not isinstance(candidate, dict):
            raise RuntimeError(f"candidate record invalid at frame={frame}")
        feature = finite_vector(candidate.get("feature"))
        if feature is None or len(feature) != 512:
            raise RuntimeError(f"runtime feature invalid at frame={frame}")
        value = cosine(feature, human)
        if value is not None:
            cosine_values.append(value)
        if candidate.get("source_public_id") == target_public_id:
            target_source_count += 1
        source = candidate.get("source_candidate")
        if not isinstance(source, dict) or source.get("machine_feature_finite") is not True:
            raise RuntimeError(f"source feature provenance invalid at frame={frame}")
        if source.get("mask_hash") is not None:
            mask_hash_count += 1
    target_index = state_ids.index(target_public_id) if target_public_id in state_ids else None
    cost_audit = audit.get("hungarian_cost_audit") or {}
    target_row_costs = cost_audit.get("target_row_costs")
    if target_row_costs is not None and finite_vector(target_row_costs) is None:
        raise RuntimeError(f"Hungarian target-row costs invalid at frame={frame}")
    top = top_two_from_audit(audit)
    return {
        "frame": int(frame),
        "is_event_frame": bool(is_event_frame),
        "is_future_frame": not bool(is_event_frame),
        "branch": branch,
        "candidate_public_ids": [int(value) for value in candidate_ids],
        "candidate_native_ids": [int(value) for value in native_ids],
        "candidate_order": [int(value) for value in order],
        "public_id_order": [int(value) for value in state_ids],
        "fused_scores": fused,
        "assignment_after_scope": [int(value) for value in assignments],
        "mapping_complete": audit.get("candidate_public_id_mapping_complete") is True,
        "candidate_count": len(candidate_ids),
        "candidate_coverage": audit.get("candidate_public_id_mapping_complete") is True,
        "target_public_id": target_public_id,
        "target_state_index": target_index,
        "target_state_present": target_index is not None,
        "target_scores": target_scores,
        "top_two": top,
        "assignment_score_margin": cost_audit.get("assignment_score_margin"),
        "assignment_cost_margin": cost_audit.get("assignment_cost_margin"),
        "target_row": cost_audit.get("target_row"),
        "assigned_col": cost_audit.get("assigned_col"),
        "best_alternative_col": cost_audit.get("best_alternative_col"),
        "target_row_costs": target_row_costs,
        "candidate_cosine_mean_to_human": (
            statistics.fmean(cosine_values) if cosine_values else None
        ),
        "candidate_cosine_max_to_human": max(cosine_values) if cosine_values else None,
        "candidate_cosine_count": len(cosine_values),
        "target_source_candidate_count": target_source_count,
        "mask_hash_count": mask_hash_count,
        "memory_read": bool(audit.get("memory_read", False)),
        "memory_write": bool(audit.get("memory_write", False)),
        "current_frame_write_hidden": bool(audit.get("current_frame_write_hidden", False)),
        "runtime_future_gt_used": audit.get("runtime_future_gt_used"),
        "gt_loaded_posthoc": audit.get("gt_loaded_posthoc"),
        "source_row_runtime_future_gt_used": source_row_metadata.get("runtime_future_gt_used"),
        "source_row_runtime_gt_read": source_row_metadata.get("runtime_gt_read"),
    }


def compact_artifact(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("status") != "PASS":
        raise RuntimeError(f"sidecar is not PASS: {path}")
    event = payload.get("event")
    if not isinstance(event, dict):
        raise RuntimeError(f"sidecar event missing: {path}")
    event_frame = int(payload["event_frame"])
    event_part = payload.get("event_frame_audit") or {}
    event_audit = compact_audit(
        event_part.get("candidate_audit"),
        source_row_metadata=(event_part.get("candidate_audit") or {}).get("source_row_metadata", {}),
        event=event,
        frame=event_frame,
        is_event_frame=True,
        branch="event_frame_audit",
    )
    branches = payload.get("branches")
    if not isinstance(branches, dict):
        raise RuntimeError(f"sidecar branches missing: {path}")
    compact_branches: dict[str, dict[int, dict[str, Any]]] = {}
    expected_frames = list(range(event_frame + 1, int(payload["future_frame_end"]) + 1))
    for branch_name in BRANCHES:
        branch = branches.get(branch_name)
        if not isinstance(branch, dict):
            raise RuntimeError(f"sidecar branch missing: {path}:{branch_name}")
        trace = branch.get("future_trace")
        if not isinstance(trace, list) or [int(entry["frame"]) for entry in trace] != expected_frames:
            raise RuntimeError(f"sidecar future trace invalid: {path}:{branch_name}")
        frame_map: dict[int, dict[str, Any]] = {}
        for entry in trace:
            frame = int(entry["frame"])
            audit = entry.get("candidate_audit")
            frame_map[frame] = compact_audit(
                audit,
                source_row_metadata=(audit or {}).get("source_row_metadata", {}),
                event=event,
                frame=frame,
                is_event_frame=False,
                branch=branch_name,
            )
        compact_branches[branch_name] = frame_map
    result = {
        "event_id": str(payload["event_id"]),
        "variant": str(payload["variant"]),
        "sequence": str(payload["sequence"]),
        "action_type": str(event["action_type"]),
        "event_frame": event_frame,
        "future_frame_end": int(payload["future_frame_end"]),
        "event": event,
        "event_frame_audit": event_audit,
        "branches": compact_branches,
        "runtime_future_gt_used": payload.get("runtime_future_gt_used"),
        "gt_loaded_posthoc": payload.get("gt_loaded_posthoc"),
        "frozen_n38_protocol_hash": payload.get("frozen_n38_protocol_hash"),
    }
    del payload
    gc.collect()
    return result


def alignment_detail(variants: dict[str, dict[str, Any]], frame: int, branch: str) -> dict[str, Any]:
    rows = [variants[name]["branches"][branch][frame] for name in VARIANTS]
    first = rows[0]
    candidate_axes_equal = True
    for row in rows[1:]:
        for field in ("candidate_native_ids", "candidate_order", "public_id_order"):
            if field in ("candidate_native_ids", "candidate_order") and row[field] != first[field]:
                candidate_axes_equal = False
    state_sets = [set(row["public_id_order"]) for row in rows]
    common_state_ids = [
        int(value) for value in first["public_id_order"]
        if all(value in state_set for state_set in state_sets)
    ]
    state_axis_changed = any(row["public_id_order"] != first["public_id_order"] for row in rows[1:])
    return {
        "candidate_axes_equal": candidate_axes_equal,
        "state_axis_changed": state_axis_changed,
        "common_state_ids": common_state_ids,
        "state_union": sorted(set().union(*state_sets)),
        "state_intersection_count": len(common_state_ids),
    }


def aligned_delta(
    left: dict[str, Any], right: dict[str, Any]
) -> tuple[list[list[float]] | None, list[int], list[int]]:
    """Align rows by candidate order and columns by public ID, never index."""
    if left["candidate_native_ids"] != right["candidate_native_ids"] or left["candidate_order"] != right["candidate_order"]:
        return None, [], []
    left_states = [int(value) for value in left["public_id_order"]]
    right_states = [int(value) for value in right["public_id_order"]]
    common_states = [value for value in left_states if value in set(right_states)]
    left_matrix = np.asarray(left["fused_scores"], dtype=float)
    right_matrix = np.asarray(right["fused_scores"], dtype=float)
    if left_matrix.ndim != 2 or right_matrix.ndim != 2 or left_matrix.shape[0] != right_matrix.shape[0]:
        return None, [], common_states
    left_indices = [left_states.index(value) for value in common_states]
    right_indices = [right_states.index(value) for value in common_states]
    delta = right_matrix[:, right_indices] - left_matrix[:, left_indices]
    return delta.tolist(), [int(value) for value in left["candidate_native_ids"]], common_states


def diagnostic_rows_for_event(variants: dict[str, dict[str, Any]], protocol: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    event_id = next(iter(variants.values()))["event_id"]
    event = next(iter(variants.values()))["event"]
    event_frame = int(next(iter(variants.values()))["event_frame"])
    frames = [event_frame] + list(range(event_frame + 1, int(next(iter(variants.values()))["future_frame_end"]) + 1))
    rows: list[dict[str, Any]] = []
    alignment_errors: list[str] = []
    for branch in ("event_frame_audit",) + BRANCHES:
        branch_frames = [event_frame] if branch == "event_frame_audit" else frames
        for frame in branch_frames:
            if branch == "event_frame_audit":
                available = {name: variants[name]["event_frame_audit"] for name in VARIANTS}
                frame_phase = "event"
            else:
                if frame == event_frame:
                    continue
                available = {name: variants[name]["branches"][branch][frame] for name in VARIANTS}
                frame_phase = "future"
            alignment = (
                {"candidate_axes_equal": True, "state_axis_changed": False, "common_state_ids": [], "state_union": [], "state_intersection_count": 0}
                if branch == "event_frame_audit"
                else alignment_detail(variants, frame, branch)
            )
            if not alignment["candidate_axes_equal"]:
                alignment_errors.append(f"{event_id}:{branch}:{frame}:unaligned_candidate_axes")
            m0 = available["M0"]
            for variant_index, variant in enumerate(VARIANTS):
                current = available[variant]
                previous = available[VARIANTS[variant_index - 1]] if variant_index > 0 else None
                baseline = m0
                previous_delta, previous_delta_rows, previous_delta_states = (
                    aligned_delta(previous, current) if previous is not None else (None, [], [])
                )
                baseline_delta, baseline_delta_rows, baseline_delta_states = aligned_delta(baseline, current)
                target_delta = None
                target_available_in_previous = bool(
                    previous is not None
                    and current["target_scores"] is not None
                    and previous["target_scores"] is not None
                )
                if target_available_in_previous:
                    if len(previous["target_scores"]) == len(current["target_scores"]):
                        target_delta = [
                            float(right - left)
                            for left, right in zip(previous["target_scores"], current["target_scores"])
                        ]
                max_abs = None
                if previous_delta is not None:
                    flat = np.asarray(previous_delta, dtype=float)
                    max_abs = float(np.max(np.abs(flat))) if flat.size else 0.0
                assignment_changed = (
                    current["candidate_public_ids"] != baseline["candidate_public_ids"]
                    if baseline["candidate_native_ids"] == current["candidate_native_ids"]
                    and baseline["candidate_order"] == current["candidate_order"]
                    else None
                )
                current_near_tie = variants[variant]["event_frame_audit"]["top_two"].get("near_tie")
                future_plus_one = variants[variant]["branches"].get(branch, {}).get(event_frame + 1)
                future_near_tie = None if future_plus_one is None else future_plus_one["top_two"].get("near_tie")
                rows.append({
                    "event_id": event_id,
                    "variant": variant,
                    "branch": branch,
                    "frame_phase": frame_phase,
                    "frame": frame,
                    "sequence": variants[variant]["sequence"],
                    "action_type": variants[variant]["action_type"],
                    "event_frame": event_frame,
                    "target_public_id": current["target_public_id"],
                    "candidate_public_ids": current["candidate_public_ids"],
                    "candidate_native_ids": current["candidate_native_ids"],
                    "candidate_order": current["candidate_order"],
                    "public_id_order": current["public_id_order"],
                    "assignment_after_scope": current["assignment_after_scope"],
                    "assignment_changed_vs_M0": assignment_changed,
                    "score_changed_vs_previous_variant": (
                        bool(max_abs is not None and max_abs > 0.0) if variant_index > 0 else False
                    ),
                    "max_abs_score_delta_vs_previous_variant": max_abs,
                    "score_delta_previous_variant_full_matrix": previous_delta,
                    "score_delta_previous_variant_candidate_native_ids": previous_delta_rows,
                    "score_delta_previous_variant_public_ids": previous_delta_states,
                    "score_delta_previous_variant_target_state": target_delta,
                    "state_axis_changed_across_variants": alignment["state_axis_changed"],
                    "common_public_id_axis_count": alignment["state_intersection_count"],
                    "score_changed_vs_M0": (
                        bool(np.max(np.abs(np.asarray(baseline_delta, dtype=float))) > 0.0)
                        if baseline_delta is not None and np.asarray(baseline_delta).size else False
                    ),
                    "top1_candidate_public_id": current["top_two"].get("top1_candidate_public_id"),
                    "top2_candidate_public_id": current["top_two"].get("top2_candidate_public_id"),
                    "top1_top2_normalized_margin": current["top_two"].get("normalized_margin"),
                    "top2_distinct_public_id": current["top_two"].get("distinct_public_id"),
                    "near_tie_current_event_frame": current_near_tie if frame_phase == "event" else variants[variant]["event_frame_audit"]["top_two"].get("near_tie"),
                    "near_tie_event_plus_one": future_near_tie,
                    "near_tie_event_and_event_plus_one": (
                        bool(current_near_tie and future_near_tie)
                        if current_near_tie is not None and future_near_tie is not None else None
                    ),
                    "target_state_present": current["target_state_present"],
                    "target_visibility_posthoc": "NOT_COMPUTED_R2_SIDECAR_ONLY",
                    "correct_assignment_change": None,
                    "correct_assignment_change_status": "NOT_COMPUTABLE_NO_FUTURE_GT_IN_R1_SIDECAR",
                    "candidate_count": current["candidate_count"],
                    "candidate_coverage": current["candidate_coverage"],
                    "target_source_candidate_count": current["target_source_candidate_count"],
                    "candidate_cosine_mean_to_human": current["candidate_cosine_mean_to_human"],
                    "candidate_cosine_max_to_human": current["candidate_cosine_max_to_human"],
                    "assignment_score_margin": current["assignment_score_margin"],
                    "assignment_cost_margin": current["assignment_cost_margin"],
                    "target_row": current["target_row"],
                    "assigned_col": current["assigned_col"],
                    "best_alternative_col": current["best_alternative_col"],
                    "target_row_costs": current["target_row_costs"],
                    "memory_read": current["memory_read"],
                    "memory_write": current["memory_write"],
                    "current_frame_write_hidden": current["current_frame_write_hidden"],
                    "runtime_future_gt_used": current["runtime_future_gt_used"],
                    "gt_loaded_posthoc": current["gt_loaded_posthoc"],
                    "source_row_runtime_future_gt_used": current["source_row_runtime_future_gt_used"],
                    "source_row_runtime_gt_read": current["source_row_runtime_gt_read"],
                    "frozen_n38_protocol_hash": protocol.get("protocol_hash"),
                })
    return rows, {
        "event_id": event_id,
        "sequence": variants["M0"]["sequence"],
        "action_type": variants["M0"]["action_type"],
        "event_frame": event_frame,
        "current_near_tie_by_variant": {
            name: variants[name]["event_frame_audit"]["top_two"].get("near_tie") for name in VARIANTS
        },
        "event_plus_one_near_tie_by_variant": {
            name: {
                branch: variants[name]["branches"][branch][event_frame + 1]["top_two"].get("near_tie")
                for branch in BRANCHES
            }
            for name in VARIANTS
        },
        "alignment_errors": alignment_errors,
    }


def summarize(rows: list[dict[str, Any]], event_summaries: list[dict[str, Any]], protocol: dict[str, Any]) -> dict[str, Any]:
    def rate(values: list[bool | None]) -> float | None:
        usable = [value for value in values if value is not None]
        return float(sum(bool(value) for value in usable) / len(usable)) if usable else None

    def group_summary(group_rows: list[dict[str, Any]]) -> dict[str, Any]:
        margins = [
            float(value)
            for row in group_rows
            for value in (row.get("assignment_score_margin"),)
            if isinstance(value, (int, float)) and math.isfinite(float(value))
        ]
        coverage = [bool(row.get("candidate_coverage")) for row in group_rows]
        memory_correct = [
            row.get("memory_read") is False if row.get("frame_phase") == "event" else row.get("runtime_future_gt_used") is False
            for row in group_rows
        ]
        return {
            "row_count": len(group_rows),
            "score_change_rate": rate([row.get("score_changed_vs_previous_variant") for row in group_rows]),
            "assignment_change_rate_vs_M0": rate([row.get("assignment_changed_vs_M0") for row in group_rows]),
            "correct_assignment_change_rate": None,
            "correct_assignment_change_status": "NOT_COMPUTABLE_NO_FUTURE_GT_IN_R1_SIDECAR",
            "mean_assignment_margin": statistics.fmean(margins) if margins else None,
            "top_quantile_assignment_margin_p90": float(np.quantile(margins, 0.9)) if margins else None,
            "target_state_present_rate": rate([row.get("target_state_present") for row in group_rows]),
            "target_visible_posthoc_rate": None,
            "candidate_coverage_rate": rate(coverage),
            "memory_read_write_audit_rate": rate(memory_correct),
            "runtime_future_gt_true_count": sum(row.get("runtime_future_gt_used") is True for row in group_rows),
            "forbidden_post_treatment_selection_used": False,
        }

    by_variant: dict[str, Any] = {}
    for variant in VARIANTS:
        by_variant[variant] = group_summary([row for row in rows if row["variant"] == variant])
    by_action: dict[str, Any] = {}
    actions = sorted({row["action_type"] for row in rows})
    for action in actions:
        by_action[action] = {
            variant: group_summary([row for row in rows if row["action_type"] == action and row["variant"] == variant])
            for variant in VARIANTS
        }
    by_sequence: dict[str, Any] = {}
    sequences = sorted({row["sequence"] for row in rows})
    for sequence in sequences:
        by_sequence[sequence] = {
            variant: group_summary([row for row in rows if row["sequence"] == sequence and row["variant"] == variant])
            for variant in VARIANTS
        }
    near_tie_events = {
        variant: {
            "current_event_frame_count": sum(
                summary["current_near_tie_by_variant"].get(variant) is True for summary in event_summaries
            ),
            "event_plus_one_count_memory_write_false": sum(
                summary["event_plus_one_near_tie_by_variant"].get(variant, {}).get("memory_write=False") is True
                for summary in event_summaries
            ),
            "event_plus_one_count_memory_write_true": sum(
                summary["event_plus_one_near_tie_by_variant"].get(variant, {}).get("memory_write=True") is True
                for summary in event_summaries
            ),
            "conjunction_count_memory_write_false": sum(
                summary["current_near_tie_by_variant"].get(variant) is True
                and summary["event_plus_one_near_tie_by_variant"].get(variant, {}).get("memory_write=False") is True
                for summary in event_summaries
            ),
            "conjunction_count_memory_write_true": sum(
                summary["current_near_tie_by_variant"].get(variant) is True
                and summary["event_plus_one_near_tie_by_variant"].get(variant, {}).get("memory_write=True") is True
                for summary in event_summaries
            ),
        }
        for variant in VARIANTS
    }
    errors = [error for summary in event_summaries for error in summary.get("alignment_errors", [])]
    runtime_gt_true = sum(row.get("runtime_future_gt_used") is True for row in rows)
    state_axis_changed_rows = sum(
        row.get("state_axis_changed_across_variants") is True for row in rows
    )
    state_axis_changed_events = len({
        row["event_id"] for row in rows if row.get("state_axis_changed_across_variants") is True
    })
    return {
        "protocol": "N38R1_MECHANISM_DIAGNOSTIC_V1",
        "status": "PASS" if not errors and len(event_summaries) == 24 and len(rows) == 24120 else "BLOCKED_INPUT_ARTIFACT_SCHEMA",
        "event_count": len(event_summaries),
        "independent_sequence_count": len({summary["sequence"] for summary in event_summaries}),
        "variant_count": len(VARIANTS),
        "row_count": len(rows),
        "expected_row_count": 24120,
        "unique_row_keys": len({(row["event_id"], row["variant"], row["branch"], row["frame_phase"], row["frame"]) for row in rows}),
        "duplicate_row_count": len(rows) - len({(row["event_id"], row["variant"], row["branch"], row["frame_phase"], row["frame"]) for row in rows}),
        "state_axis_changed_row_count": state_axis_changed_rows,
        "state_axis_changed_event_count": state_axis_changed_events,
        "state_axis_alignment_semantics": "public-ID columns are aligned by intersection; additions/removals are retained as mechanism evidence and are not schema failure",
        "score_changed_definition": "finite nonzero full fused-score delta between adjacent M variants on aligned candidate/state axes",
        "assignment_changed_definition": "final assignment_after_scope vector differs from M0 on same branch/frame",
        "near_tie_definition": protocol.get("near_tie_definition"),
        "near_tie_threshold": 0.05,
        "by_variant": by_variant,
        "by_action": by_action,
        "by_sequence": by_sequence,
        "near_tie_events": near_tie_events,
        "event_summaries": event_summaries,
        "alignment_errors": errors,
        "runtime_future_gt_true_count": runtime_gt_true,
        "memory_current_frame_write_hidden_count": sum(row.get("current_frame_write_hidden") is True for row in rows if row["frame_phase"] == "event"),
        "correct_assignment_change_status": "NOT_COMPUTABLE_NO_FUTURE_GT_IN_R1_SIDECAR",
        "target_visibility_status": "NOT_COMPUTABLE_NO_FUTURE_GT_IN_R1_SIDECAR",
        "posthoc_gt_loaded": False,
        "downstream_authorized": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--schema-audit", type=Path, default=DEFAULT_SCHEMA_AUDIT)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--stage", type=Path, default=None)
    parser.add_argument("--failure", type=Path, default=None)
    args = parser.parse_args()
    try:
        protocol = frozen_protocol()
        manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
        schema_audit = json.loads(args.schema_audit.read_text(encoding="utf-8"))
        if manifest.get("status") != "PASS" or manifest.get("scope") != "all":
            raise RuntimeError("R1 sidecar manifest must be full-scope PASS")
        if schema_audit.get("status") != "PASS" or schema_audit.get("audited_artifacts") != 120:
            raise RuntimeError("R1 full schema audit must be PASS for 120 artifacts")
        records = manifest.get("policy_rows")
        if not isinstance(records, list) or len(records) != 120:
            raise RuntimeError("R1 manifest does not contain exactly 120 policy rows")
        by_event: dict[str, dict[str, Path]] = defaultdict(dict)
        for record in records:
            key = (str(record["event_id"]), str(record["variant"]))
            if key[1] not in VARIANTS or key[0] in by_event and key[1] in by_event[key[0]]:
                raise RuntimeError(f"duplicate/invalid manifest key {key}")
            by_event[key[0]][key[1]] = ROOT / record["path"]
        if len(by_event) != 24 or any(set(value) != set(VARIANTS) for value in by_event.values()):
            raise RuntimeError("R1 manifest event×variant coverage is not exactly 24×5")
        all_rows: list[dict[str, Any]] = []
        event_summaries: list[dict[str, Any]] = []
        for event_id in [str(item) for item in by_event]:
            variants: dict[str, dict[str, Any]] = {}
            for variant in VARIANTS:
                variants[variant] = compact_artifact(by_event[event_id][variant])
            rows, event_summary = diagnostic_rows_for_event(variants, protocol)
            all_rows.extend(rows)
            event_summaries.append(event_summary)
            del variants, rows
            gc.collect()
        summary = summarize(all_rows, event_summaries, protocol)
        args.out.mkdir(parents=True, exist_ok=True)
        table_path = args.out / "score_assignment_table.jsonl"
        summary_path = args.out / "score_assignment_summary.json"
        protocol_path = args.out / "diagnostic_protocol.json"
        stage_path = args.stage if args.stage is not None else ROOT / "outputs" / "n38r1" / "stage_02_status.json"
        atomic_jsonl(table_path, all_rows)
        atomic_json(summary_path, summary)
        derived_protocol = {
            "protocol": "N38R1_MECHANISM_DIAGNOSTIC_PROTOCOL_V1",
            "base_n38_protocol": str(N38_PROTOCOL_PATH.resolve().relative_to(ROOT)),
            "base_n38_protocol_hash": protocol.get("protocol_hash"),
            "near_tie_definition": protocol.get("near_tie_definition"),
            "near_tie_threshold": 0.05,
            "current_frame_source": "R1 audit-only clone after spatial correction and before event-frame memory read/write",
            "future_branch": "R1 paired_replay memory_write=False/True traces",
            "selection_fields_allowed": ["event_frame", "current-prefix", "event-frame candidate/state scores and margins"],
            "selection_fields_forbidden": ["future GT", "future identity error", "H20", "H50", "H100", "post-treatment metrics"],
            "runtime_future_gt_used": False,
        }
        derived_protocol["protocol_hash"] = hashlib.sha256(
            json.dumps(derived_protocol, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        atomic_json(protocol_path, derived_protocol)
        stage = {
            "stage": "N38R1-02",
            "status": summary["status"],
            "real_data_status": summary["status"],
            "source_manifest": str(args.manifest.resolve().relative_to(ROOT)),
            "source_schema_audit": str(args.schema_audit.resolve().relative_to(ROOT)),
            "artifacts": [
                str(table_path.resolve().relative_to(ROOT)),
                str(summary_path.resolve().relative_to(ROOT)),
                str(protocol_path.resolve().relative_to(ROOT)),
            ],
            "event_count": summary["event_count"],
            "independent_sequence_count": summary["independent_sequence_count"],
            "row_count": summary["row_count"],
            "duplicate_row_count": summary["duplicate_row_count"],
            "alignment_error_count": len(summary["alignment_errors"]),
            "runtime_future_gt_used": False,
            "posthoc_gt_loaded": False,
            "correct_assignment_change": summary["correct_assignment_change_status"],
            "target_visibility": summary["target_visibility_status"],
            "downstream_authorized": False,
            "next_action": "Proceed to R3 only using frozen current/event-frame near-tie fields; no post-treatment selection." if summary["status"] == "PASS" else "Preserve R2 schema/alignment failures and repair only the first actionable root cause.",
        }
        atomic_json(stage_path, stage)
        print(json.dumps({"status": summary["status"], "event_count": summary["event_count"], "row_count": summary["row_count"], "duplicate_row_count": summary["duplicate_row_count"], "alignment_error_count": len(summary["alignment_errors"]), "runtime_future_gt_true_count": summary["runtime_future_gt_true_count"]}, sort_keys=True), flush=True)
        return 0 if summary["status"] == "PASS" else 1
    except Exception as exc:
        failure = {
            "protocol": "N38R1_MECHANISM_DIAGNOSTIC_V1",
            "status": "FAIL",
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
            "runtime_future_gt_used": False,
        }
        failure_path = args.failure if args.failure is not None else ROOT / "outputs" / "n38r1" / "diagnostic_attempt1_failure.json"
        atomic_json(failure_path, failure)
        print(failure["error"], flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
