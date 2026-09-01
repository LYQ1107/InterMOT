#!/usr/bin/env python3
"""Run the read-only N38 mechanism/schema diagnostic.

This script deliberately consumes frozen N37 evidence only.  The canonical N37
artifacts are compacted, while the successful files from the preserved attempt-1
run retain the detailed per-candidate audit.  The latter are used only as a
read-only diagnostic supplement; failed attempt-1 files are never promoted.

The script writes the N38 Stage-A artifacts atomically and never writes under
outputs/n36 or outputs/n37.
"""

from __future__ import annotations

import gc
import hashlib
import json
import math
import os
import pathlib
import statistics
import sys
import tempfile
from collections import Counter, defaultdict
from typing import Any, Iterable


ROOT = pathlib.Path(__file__).resolve().parents[1]
N37 = ROOT / "outputs" / "n37"
N38 = ROOT / "outputs" / "n38"
DIAGNOSTIC = N38 / "diagnostic"
CANONICAL_ROOT = N37 / "replay_event_artifacts"
RAW_ROOT = N37 / "replay_event_artifacts_attempt1_raw"
MANIFEST_PATH = N37 / "real_event_manifest.json"
RESULT_PATH = N37 / "ccam_paired_replay_results.json"
VARIANTS = ("M0", "M1", "M2", "M3", "M4")
BRANCHES = ("memory_write_false", "memory_write_true")

# This is registered before the artifact scan.  Scores are maximized by the
# frozen association helper; cost is therefore -score.  The target state
# column is used to rank candidate rows for the target public identity.
NEAR_TIE_THRESHOLD = 0.05
BOOTSTRAP_SEED = 37038
BOOTSTRAP_REPETITIONS = 2000

REQUIRED_DETAIL_FIELDS = (
    "candidates",
    "candidate_native_ids",
    "candidate_public_ids",
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
)


def load_json(path: pathlib.Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def atomic_text(path: pathlib.Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except Exception:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def atomic_json(path: pathlib.Path, value: Any) -> None:
    atomic_text(path, json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def finite_float(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def finite_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def clean_json(value: Any) -> Any:
    """Convert non-finite floats to null so all emitted JSON is strict."""
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {str(key): clean_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean_json(item) for item in value]
    return value


def shape(value: Any) -> list[int] | None:
    result: list[int] = []
    current = value
    while isinstance(current, list):
        result.append(len(current))
        current = current[0] if current else None
    return result if result else None


def key_for(data: dict[str, Any]) -> tuple[str, str]:
    return str(data.get("event_id")), str(data.get("variant"))


def event_id_from_manifest(entry: dict[str, Any]) -> str:
    nested = entry.get("event") or {}
    return str(nested.get("event_id"))


def detail_audit_complete(audit: Any) -> bool:
    return isinstance(audit, dict) and all(field in audit for field in REQUIRED_DETAIL_FIELDS)


def matrix_as_float(value: Any) -> list[list[float]] | None:
    if not isinstance(value, list) or any(not isinstance(row, list) for row in value):
        return None
    result: list[list[float]] = []
    width: int | None = None
    for row in value:
        converted: list[float] = []
        for item in row:
            number = finite_float(item)
            if number is None:
                return None
            converted.append(number)
        if width is None:
            width = len(converted)
        elif width != len(converted):
            return None
        result.append(converted)
    return result


def get_frame_map(data: dict[str, Any], branch: str) -> tuple[dict[int, dict[str, Any]], list[str]]:
    trace = (data.get("future_trace") or {}).get(branch)
    errors: list[str] = []
    if not isinstance(trace, list):
        return {}, [f"{branch}:missing_trace"]
    frames: dict[int, dict[str, Any]] = {}
    for entry in trace:
        if not isinstance(entry, dict):
            errors.append(f"{branch}:non_object_frame")
            continue
        frame = finite_int(entry.get("frame"))
        if frame is None:
            errors.append(f"{branch}:non_integer_frame")
            continue
        if frame in frames:
            errors.append(f"{branch}:duplicate_frame:{frame}")
        frames[frame] = entry
    return frames, errors


def cosine(feature: Any, human: Any) -> float | None:
    if not isinstance(feature, list) or not isinstance(human, list) or len(feature) != len(human):
        return None
    f: list[float] = []
    h: list[float] = []
    for left, right in zip(feature, human):
        left_value = finite_float(left)
        right_value = finite_float(right)
        if left_value is None or right_value is None:
            return None
        f.append(left_value)
        h.append(right_value)
    f_norm = math.sqrt(sum(item * item for item in f))
    h_norm = math.sqrt(sum(item * item for item in h))
    if f_norm == 0.0 or h_norm == 0.0:
        return None
    return sum(left * right for left, right in zip(f, h)) / (f_norm * h_norm)


def target_state_diagnostics(
    audit: dict[str, Any], target_public_id: int | None
) -> dict[str, Any]:
    """Derive frozen-association diagnostics from the lossless audit fields."""
    candidate_ids = audit.get("candidate_public_ids")
    native_ids = audit.get("candidate_native_ids")
    state_ids = audit.get("public_id_order")
    fused = matrix_as_float(audit.get("fused_scores"))
    assignments = audit.get("assignment_after_scope")
    if (
        not isinstance(candidate_ids, list)
        or not isinstance(native_ids, list)
        or not isinstance(state_ids, list)
        or fused is None
        or not isinstance(assignments, list)
        or len(fused) != len(candidate_ids)
        or len(native_ids) != len(candidate_ids)
        or len(assignments) != len(candidate_ids)
    ):
        return {"status": "UNAVAILABLE_NONCONFORMING_DETAIL_AUDIT"}
    if any(len(row) != len(state_ids) for row in fused):
        return {"status": "UNAVAILABLE_MATRIX_SHAPE_MISMATCH"}

    state_index = None
    if target_public_id is not None and target_public_id in state_ids:
        state_index = state_ids.index(target_public_id)
    target_scores: list[tuple[float, int]] = []
    if state_index is not None:
        target_scores = sorted(
            ((fused[index][state_index], index) for index in range(len(fused))),
            key=lambda item: (-item[0], item[1]),
        )
    top1 = target_scores[0] if len(target_scores) >= 1 else None
    top2 = target_scores[1] if len(target_scores) >= 2 else None
    top1_public = candidate_ids[top1[1]] if top1 is not None else None
    top2_public = candidate_ids[top2[1]] if top2 is not None else None
    score_margin = top1[0] - top2[0] if top1 is not None and top2 is not None else None
    normalized_margin = (
        score_margin / max(1.0, abs(top1[0]))
        if score_margin is not None and top1 is not None
        else None
    )
    future_near_tie = bool(
        top1 is not None
        and top2 is not None
        and top1_public != top2_public
        and normalized_margin is not None
        and normalized_margin <= NEAR_TIE_THRESHOLD
    )

    target_row = None
    if target_public_id is not None:
        matching = [index for index, public_id in enumerate(candidate_ids) if public_id == target_public_id]
        if matching:
            target_row = matching[0]
    assigned_col = None
    assigned_public_id = None
    best_alternative_col = None
    assignment_score_margin = None
    assignment_cost_margin = None
    target_row_costs: list[float] | None = None
    if target_row is not None:
        assigned_value = finite_int(assignments[target_row])
        if assigned_value is not None and 0 <= assigned_value < len(state_ids):
            assigned_col = assigned_value
            assigned_public_id = state_ids[assigned_col]
        target_row_costs = [-value for value in fused[target_row]]
        alternatives = [index for index in range(len(state_ids)) if index != assigned_col]
        if alternatives:
            best_alternative_col = min(alternatives, key=lambda index: target_row_costs[index])
            if assigned_col is not None:
                assignment_cost_margin = (
                    target_row_costs[best_alternative_col] - target_row_costs[assigned_col]
                )
                assignment_score_margin = (
                    fused[target_row][assigned_col] - fused[target_row][best_alternative_col]
                )

    return {
        "status": "AVAILABLE",
        "candidate_count": len(candidate_ids),
        "candidate_public_ids": [finite_int(item) for item in candidate_ids],
        "candidate_native_ids": [finite_int(item) for item in native_ids],
        "public_id_order": [finite_int(item) for item in state_ids],
        "target_state_index": state_index,
        "target_row": target_row,
        "assigned_col": assigned_col,
        "assigned_public_id": finite_int(assigned_public_id),
        "best_alternative_col": best_alternative_col,
        "assignment_score_margin": finite_float(assignment_score_margin),
        "assignment_cost_margin": finite_float(assignment_cost_margin),
        "target_row_costs": target_row_costs,
        "top1_candidate_index": top1[1] if top1 is not None else None,
        "top2_candidate_index": top2[1] if top2 is not None else None,
        "top1_candidate_public_id": finite_int(top1_public),
        "top2_candidate_public_id": finite_int(top2_public),
        "top1_score": finite_float(top1[0]) if top1 is not None else None,
        "top2_score": finite_float(top2[0]) if top2 is not None else None,
        "top1_top2_score_margin": finite_float(score_margin),
        "top1_top2_normalized_margin": finite_float(normalized_margin),
        "top2_distinct_public_id": (
            top1_public != top2_public if top1 is not None and top2 is not None else None
        ),
        "future_near_tie": future_near_tie if top1 is not None and top2 is not None else None,
        "fused_scores": fused,
    }


def assignment_list(audit: Any) -> list[int] | None:
    if not isinstance(audit, dict):
        return None
    value = audit.get("assignment_after_scope", audit.get("assignment"))
    if not isinstance(value, list):
        return None
    result: list[int] = []
    for item in value:
        converted = finite_int(item)
        if converted is None:
            return None
        result.append(converted)
    return result


def matrices_aligned(audits: dict[str, dict[str, Any]]) -> bool:
    if set(audits) != set(VARIANTS):
        return False
    first = audits[VARIANTS[0]]
    for variant in VARIANTS[1:]:
        other = audits[variant]
        for field in ("candidate_public_ids", "candidate_native_ids", "candidate_order", "public_id_order"):
            if first.get(field) != other.get(field):
                return False
        if shape(first.get("fused_scores")) != shape(other.get("fused_scores")):
            return False
    return True


def get_h20(data: dict[str, Any], field: str) -> dict[str, Any]:
    try:
        return data[field]["horizons"]["20"]
    except (KeyError, TypeError):
        return {}


def utility_change(data: dict[str, Any]) -> dict[str, Any]:
    before = get_h20(data, "no_write_metrics")
    after = get_h20(data, "write_metrics")
    before_error = finite_float(before.get("target_identity_error_rate"))
    after_error = finite_float(after.get("target_identity_error_rate"))
    before_iou = finite_float(before.get("target_mean_iou"))
    after_iou = finite_float(after.get("target_mean_iou"))
    improved = None
    if before_error is not None and after_error is not None and before_iou is not None and after_iou is not None:
        improved = bool(after_error < before_error or after_iou > before_iou)
    return {
        "target_identity_error_before": before_error,
        "target_identity_error_after": after_error,
        "target_mean_iou_before": before_iou,
        "target_mean_iou_after": after_iou,
        "correct_change": improved,
    }


def percentile(values: Iterable[float], q: float) -> float | None:
    ordered = sorted(float(value) for value in values if math.isfinite(float(value)))
    if not ordered:
        return None
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * q
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def protocol_payload() -> dict[str, Any]:
    return {
        "protocol": "N38_MECHANISM_DIAGNOSTIC_AND_NEAR_TIE_V1",
        "frozen_before_event_scan": True,
        "event_selection_started": False,
        "source_scope": {
            "canonical_replay_artifacts": "outputs/n37/replay_event_artifacts/*/*.json",
            "read_only_detailed_supplement": "outputs/n37/replay_event_artifacts_attempt1_raw/*/*.json",
            "event_manifest": "outputs/n37/real_event_manifest.json",
            "aggregate_result": "outputs/n37/ccam_paired_replay_results.json",
            "n36_n37_outputs_immutable": True,
        },
        "score_orientation": "higher_fused_score_is_better; frozen Hungarian cost is negative fused score",
        "near_tie_rule": {
            "absolute_threshold": NEAR_TIE_THRESHOLD,
            "normalized_margin": "(top1_score - top2_score) / max(1.0, abs(top1_score))",
            "candidate_ranking": "rank candidate rows by finite fused score in the target public-ID state column, descending",
            "different_identity_requirement": "top1 and top2 source candidate_public_id must differ",
            "joint_frame_requirement": "the rule must hold at both event frame and event+1",
            "current_frame_required": True,
            "post_treatment_not_used": True,
        },
        "minimum_event_constraints_for_later_stage": {
            "target_visible_minimum_current_frame": 1,
            "minimum_independent_sequences": 16,
            "minimum_events": 24,
            "minimum_events_per_action": 4,
            "same_sequence_nonoverlap_required": True,
        },
        "future_effect_protocol": {
            "future_start": "event+1",
            "windows": [20, 50, 100],
            "bootstrap_cluster": "independent sequence",
            "bootstrap_seed": BOOTSTRAP_SEED,
            "bootstrap_repetitions": BOOTSTRAP_REPETITIONS,
        },
        "forbidden_for_event_selection": [
            "future_identity_error",
            "future_missing",
            "future_iou",
            "H20",
            "H50",
            "H100",
            "replay_variant_outcome",
            "assignment_change_after_memory_write",
            "any_post_treatment_metric",
        ],
        "required_lossless_audit_fields": list(REQUIRED_DETAIL_FIELDS),
        "strict_missing_field_policy": "any missing current-frame or future per-candidate score/cost field blocks event selection and downstream replay",
        "artifact_compaction_policy": "raw attempt-1 success is supplemental evidence only; raw failures and canonical compact artifacts remain preserved",
    }


def build_candidate_records(
    audit: dict[str, Any],
    target_public_id: int | None,
    human_embedding: Any,
    cross_scores: dict[str, float | None] | None,
    assignment_vs_no_write: bool | None,
    assignment_vs_m0: bool | None,
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    diagnostics = target_state_diagnostics(audit, target_public_id)
    if diagnostics.get("status") != "AVAILABLE":
        return [], {"status": diagnostics.get("status")}, diagnostics
    candidate_ids = diagnostics["candidate_public_ids"]
    native_ids = diagnostics["candidate_native_ids"]
    state_ids = diagnostics["public_id_order"]
    fused = diagnostics["fused_scores"]
    state_index = diagnostics.get("target_state_index")
    target_ranks: dict[int, int] = {}
    if state_index is not None:
        ranked = sorted(range(len(candidate_ids)), key=lambda index: (-fused[index][state_index], index))
        target_ranks = {candidate_index: rank + 1 for rank, candidate_index in enumerate(ranked)}
    appearances = matrix_as_float(audit.get("appearance_score_deltas"))
    records: list[dict[str, Any]] = []
    for index, public_id in enumerate(candidate_ids):
        assigned_col = finite_int((audit.get("assignment_after_scope") or [])[index])
        assigned_public = state_ids[assigned_col] if assigned_col is not None and 0 <= assigned_col < len(state_ids) else None
        target_score = fused[index][state_index] if state_index is not None else None
        appearance_target = (
            appearances[index][state_index]
            if appearances is not None
            and state_index is not None
            and index < len(appearances)
            and state_index < len(appearances[index])
            else None
        )
        candidate = (audit.get("candidates") or [])[index]
        records.append(
            {
                "candidate_index": index,
                "candidate_obs_id": finite_int((audit.get("candidate_order") or [])[index]),
                "candidate_public_id": public_id,
                "candidate_native_id": native_ids[index],
                "candidate_rank_for_target_state": target_ranks.get(index),
                "candidate_box": clean_json(candidate.get("box")) if isinstance(candidate, dict) else None,
                "embedding_cosine_to_human_anchor": finite_float(
                    cosine(candidate.get("feature") if isinstance(candidate, dict) else None, human_embedding)
                ),
                "target_state_score": finite_float(target_score),
                "assigned_state_index": assigned_col,
                "assigned_public_id": finite_int(assigned_public),
                "assigned_score": (
                    finite_float(fused[index][assigned_col])
                    if assigned_col is not None and 0 <= assigned_col < len(fused[index])
                    else None
                ),
                "assigned_cost": (
                    finite_float(-fused[index][assigned_col])
                    if assigned_col is not None and 0 <= assigned_col < len(fused[index])
                    else None
                ),
                "is_target_source_candidate": bool(target_public_id is not None and public_id == target_public_id),
                "appearance_score_delta_target_state": finite_float(appearance_target),
                "score_m1_minus_m0_target_state": (
                    cross_scores.get("M1_minus_M0") if cross_scores is not None else None
                ),
                "score_m2_minus_m1_target_state": (
                    cross_scores.get("M2_minus_M1") if cross_scores is not None else None
                ),
                "score_m3_minus_m2_target_state": (
                    cross_scores.get("M3_minus_M2") if cross_scores is not None else None
                ),
                "score_m4_minus_m3_target_state": (
                    cross_scores.get("M4_minus_M3") if cross_scores is not None else None
                ),
                "assignment_changed_vs_same_variant_no_write": assignment_vs_no_write,
                "assignment_changed_vs_m0_same_branch": assignment_vs_m0,
            }
        )
    return records, {"status": "AVAILABLE"}, diagnostics


def main() -> int:
    DIAGNOSTIC.mkdir(parents=True, exist_ok=True)
    protocol = protocol_payload()
    protocol["protocol_hash"] = hashlib.sha256(canonical_json(protocol).encode("utf-8")).hexdigest()
    atomic_json(DIAGNOSTIC / "diagnostic_protocol.json", protocol)

    manifest = load_json(MANIFEST_PATH)
    result = load_json(RESULT_PATH)
    manifest_entries = manifest.get("events") if isinstance(manifest, dict) else None
    if not isinstance(manifest_entries, list):
        raise RuntimeError("N37 real_event_manifest.events is not a list")
    event_meta = {event_id_from_manifest(entry): entry for entry in manifest_entries if isinstance(entry, dict)}

    canonical_paths = sorted(CANONICAL_ROOT.glob("*/*.json"))
    if len(canonical_paths) != 120:
        raise RuntimeError(f"expected 120 canonical N37 artifacts, found {len(canonical_paths)}")
    canonical: dict[tuple[str, str], tuple[pathlib.Path, dict[str, Any]]] = {}
    schema_errors: list[str] = []
    for path in canonical_paths:
        data = load_json(path)
        key = key_for(data)
        if key in canonical:
            schema_errors.append(f"duplicate canonical key {key}")
        canonical[key] = (path, data)
    expected_events = sorted({key[0] for key in canonical})
    expected_keys = {(event_id, variant) for event_id in expected_events for variant in VARIANTS}
    if set(canonical) != expected_keys:
        missing = sorted(expected_keys - set(canonical))
        extra = sorted(set(canonical) - expected_keys)
        schema_errors.append(f"canonical key set mismatch missing={missing} extra={extra}")
    if len(expected_events) != 24:
        schema_errors.append(f"expected 24 events, found {len(expected_events)}")

    # Derive the frozen raw-attempt key from its directory/file name.  Do not
    # deserialize the 7+ GB preserved raw directory just to discover keys.
    raw_paths = {
        (path.parent.name, path.stem): path
        for path in sorted(RAW_ROOT.glob("*/*.json"))
    }
    raw_failed_keys: list[dict[str, Any]] = []
    raw_detail_keys: set[tuple[str, str]] = set()
    for key, path in sorted(raw_paths.items()):
        # This second load is intentionally limited to the small failed files;
        # successful raw files are loaded once per event below.
        if path.stat().st_size < 2000:
            data = load_json(path)
            if "future_trace" not in data or data.get("status") != "PASS":
                raw_failed_keys.append({"event_id": key[0], "variant": key[1], "artifact": str(path.relative_to(ROOT)), "status": data.get("status"), "error": data.get("error")})

    # The aggregate result is a frozen cross-check, not a source for event selection.
    aggregate_events = result.get("events") if isinstance(result, dict) else []
    aggregate_by_event = {
        str(item.get("event_id")): item for item in aggregate_events if isinstance(item, dict)
    }

    table_lines: list[str] = []
    summary_by_variant: dict[str, dict[str, Any]] = {}
    for variant in VARIANTS:
        summary_by_variant[variant] = {
            "canonical_artifact_count": 0,
            "detailed_artifact_count": 0,
            "compact_fallback_artifact_count": 0,
            "future_frame_count": 0,
            "score_changed_first_future_count": 0,
            "assignment_changed_first_future_count": 0,
            "correct_assignment_change_first_future_count": 0,
            "target_visible_h20_event_count": 0,
            "future_near_tie_event_count": 0,
            "current_near_tie_event_count": "UNAVAILABLE_EVENT_FRAME_SCORE_AUDIT",
            "joint_near_tie_event_count": "UNAVAILABLE_CURRENT_FRAME_REQUIRED",
            "assignment_cost_margins_event_plus_one": [],
            "top1_top2_normalized_margins_event_plus_one": [],
            "embedding_cosine_count": 0,
            "candidate_mapping_complete_frame_count": 0,
            "candidate_duplicate_frame_count": 0,
            "candidate_missing_or_incomplete_frame_count": 0,
        }

    detail_field_artifact_counts = Counter()
    detail_field_frame_counts = Counter()
    raw_vs_canonical_mismatches: list[dict[str, Any]] = []
    trace_alignment_errors: list[str] = []
    current_frame_exclusion_errors: list[str] = []
    runtime_future_gt_true: list[str] = []
    current_memory_write_used: list[str] = []
    candidate_duplicate_examples: list[str] = []
    future_near_tie_by_action: Counter[tuple[str, str]] = Counter()
    future_near_tie_by_sequence: Counter[tuple[str, str]] = Counter()
    margin_by_action: dict[tuple[str, str], list[float]] = defaultdict(list)
    margin_by_sequence: dict[tuple[str, str], list[float]] = defaultdict(list)
    source_mode_counts = Counter()
    table_row_count = 0
    detailed_frames = 0
    compact_frames = 0
    event_current_rows = 0
    canonical_future_frames = 0
    canonical_event_ids: set[str] = set()
    seen_table_keys: set[tuple[str, str, str, int]] = set()

    for event_id in expected_events:
        metadata = event_meta.get(event_id, {})
        nested_event = metadata.get("event") if isinstance(metadata, dict) else {}
        nested_event = nested_event if isinstance(nested_event, dict) else {}
        sequence = str(metadata.get("sequence", nested_event.get("sequence", "")))
        action = str(metadata.get("action_type", nested_event.get("action_type", "")))
        event_frame = finite_int(metadata.get("event_frame", nested_event.get("frame")))
        target_public_id = finite_int(
            nested_event.get("canonical_public_id", nested_event.get("public_id"))
        )
        human_embedding = nested_event.get("human_embedding")
        canonical_data: dict[str, dict[str, Any]] = {}
        detailed_data: dict[str, dict[str, Any] | None] = {}
        source_paths: dict[str, pathlib.Path] = {}
        for variant in VARIANTS:
            key = (event_id, variant)
            if key not in canonical:
                continue
            canonical_path, canonical_item = canonical[key]
            canonical_data[variant] = canonical_item
            source_paths[variant] = canonical_path
            summary_by_variant[variant]["canonical_artifact_count"] += 1
            canonical_event_ids.add(event_id)
            raw_path = raw_paths.get(key)
            raw_item: dict[str, Any] | None = None
            if raw_path is not None:
                loaded = load_json(raw_path)
                if loaded.get("status") == "PASS" and isinstance(loaded.get("future_trace"), dict):
                    raw_item = loaded
                    raw_detail_keys.add(key)
                    source_mode_counts["raw_attempt1_success_detail"] += 1
                    if loaded.get("score_delta_first_future") != canonical_item.get("score_delta_first_future"):
                        raw_vs_canonical_mismatches.append({
                            "event_id": event_id,
                            "variant": variant,
                            "field": "score_delta_first_future",
                        })
                else:
                    source_mode_counts["canonical_compact_after_raw_failure"] += 1
            else:
                source_mode_counts["canonical_compact_no_raw"] += 1
            detailed_data[variant] = raw_item

        # Current-frame score audit is absent by construction: N37 explicitly
        # records that its future trace starts at event+1 and excludes event_frame.
        for variant in VARIANTS:
            data = canonical_data.get(variant, {})
            boundary = data.get("causal_boundary") or {}
            if boundary.get("event_frame_excluded_from_future_tape") is not True:
                current_frame_exclusion_errors.append(f"{event_id}:{variant}:boundary_flag")
            if boundary.get("current_frame_write_used_for_score") is True:
                current_memory_write_used.append(f"{event_id}:{variant}")
            if data.get("runtime_future_gt_used") is True or boundary.get("runtime_future_gt_used") is True:
                runtime_future_gt_true.append(f"{event_id}:{variant}")
            current_row_key = (event_id, variant, "event_current", event_frame if event_frame is not None else -1)
            if current_row_key in seen_table_keys:
                schema_errors.append(f"duplicate table key {current_row_key}")
            seen_table_keys.add(current_row_key)
            current_row = {
                "record_kind": "frame_diagnostic",
                "frame_phase": "event_current",
                "event_id": event_id,
                "sequence": sequence,
                "action_type": action,
                "event_frame": event_frame,
                "frame": event_frame,
                "event_plus_one": False,
                "variant": variant,
                "branch": None,
                "memory_write": False,
                "memory_read": False,
                "memory_read_status": "EVENT_FRAME_SCORE_AUDIT_NOT_RETAINED",
                "current_frame_memory_write_used_for_score": boundary.get("current_frame_write_used_for_score"),
                "current_frame_score_audit_available": False,
                "current_candidate_count_from_frozen_manifest": metadata.get("event_current_candidate_count"),
                "candidate_records": [],
                "candidate_diagnostic_status": "UNAVAILABLE_EVENT_FRAME_SCORE_AUDIT",
                "top1_top2_margin": None,
                "top2_distinct_public_id": None,
                "near_tie": None,
                "near_tie_status": "UNAVAILABLE_EVENT_FRAME_SCORE_AUDIT",
                "hungarian": {"status": "UNAVAILABLE_EVENT_FRAME_SCORE_AUDIT"},
                "target_visible_count_h20_posthoc": None,
                "candidate_coverage": None,
                "box_iou": None,
                "embedding_cosine": None,
                "candidate_duplicate_or_missing": None,
                "runtime_future_gt_used": False,
                "source_canonical_artifact": str(source_paths.get(variant, "").relative_to(ROOT)) if variant in source_paths else None,
                "source_detail_artifact": None,
            }
            table_lines.append(json.dumps(clean_json(current_row), ensure_ascii=False, separators=(",", ":"), allow_nan=False))
            table_row_count += 1
            event_current_rows += 1

        # Build branch/frame maps, preferring detailed raw evidence for the 117
        # successful attempt-1 artifacts and using canonical compact traces for
        # the 15 failed-attempt fallbacks.
        frame_maps: dict[str, dict[str, dict[int, dict[str, Any]]]] = defaultdict(dict)
        frame_errors: dict[str, list[str]] = defaultdict(list)
        for variant in VARIANTS:
            for branch in BRANCHES:
                source = detailed_data.get(variant) or canonical_data.get(variant, {})
                mapping, errors = get_frame_map(source, branch)
                frame_maps[variant][branch] = mapping
                frame_errors[variant].extend(errors)
                if errors:
                    trace_alignment_errors.extend(f"{event_id}:{variant}:{error}" for error in errors)
        all_frames = sorted({frame for variant in VARIANTS for branch in BRANCHES for frame in frame_maps[variant][branch]})
        if len(all_frames) != 100:
            trace_alignment_errors.append(f"{event_id}:expected_100_union_frames_found_{len(all_frames)}")
        canonical_future_frames += sum(
            len(frame_maps[variant][branch])
            for variant in VARIANTS
            for branch in BRANCHES
        )

        # Cross-variant lookups are created per frame so no large raw object is
        # retained after this event is emitted.
        for branch in BRANCHES:
            for frame in all_frames:
                for variant in VARIANTS:
                    canonical_item = canonical_data.get(variant, {})
                    detail_item = detailed_data.get(variant)
                    source_item = detail_item or canonical_item
                    entry = frame_maps[variant][branch].get(frame)
                    if entry is None:
                        schema_errors.append(f"missing trace frame {event_id}:{variant}:{branch}:{frame}")
                        continue
                    audit = entry.get("candidate_audit") if isinstance(entry, dict) else None
                    detailed = detail_item is not None and detail_audit_complete(audit)
                    if detailed:
                        detail_field_frame_counts.update(REQUIRED_DETAIL_FIELDS)
                        detailed_frames += 1
                        summary_by_variant[variant]["detailed_artifact_count"] += 0
                    else:
                        compact_frames += 1
                    if detail_item is not None and detailed:
                        source_mode = "raw_attempt1_success_detail"
                    else:
                        source_mode = "canonical_compact_fallback"
                    source_mode_counts[f"frame:{source_mode}"] += 1
                    if detail_item is not None and not detailed:
                        for field in REQUIRED_DETAIL_FIELDS:
                            if not isinstance(audit, dict) or field not in audit:
                                detail_field_frame_counts[f"missing:{field}"] += 1

                    frame_key = (event_id, variant, branch, frame)
                    if frame_key in seen_table_keys:
                        schema_errors.append(f"duplicate table key {frame_key}")
                    seen_table_keys.add(frame_key)
                    for required_field in REQUIRED_DETAIL_FIELDS:
                        if detailed:
                            detail_field_artifact_counts[required_field] += 1
                    candidate_count = (
                        len(audit.get("candidate_public_ids", []))
                        if detailed and isinstance(audit, dict)
                        else audit.get("candidate_count") if isinstance(audit, dict) else None
                    )
                    mapping_complete = audit.get("candidate_public_id_mapping_complete") if isinstance(audit, dict) else None
                    if mapping_complete is True:
                        summary_by_variant[variant]["candidate_mapping_complete_frame_count"] += 1

                    # Get exact cross-variant matrices only when all five raw
                    # audits for this event/branch/frame are lossless and aligned.
                    audits: dict[str, dict[str, Any]] = {}
                    for other_variant in VARIANTS:
                        other_entry = frame_maps[other_variant][branch].get(frame)
                        other_audit = other_entry.get("candidate_audit") if isinstance(other_entry, dict) else None
                        if detailed_data.get(other_variant) is not None and detail_audit_complete(other_audit):
                            audits[other_variant] = other_audit
                    cross_aligned = len(audits) == len(VARIANTS) and matrices_aligned(audits)
                    cross_target_scores: dict[str, dict[int, float]] = {}
                    if cross_aligned:
                        base_diag = target_state_diagnostics(audits["M0"], target_public_id)
                        target_idx = base_diag.get("target_state_index")
                        if target_idx is not None:
                            candidate_count_cross = len(audits["M0"].get("candidate_public_ids", []))
                            for other_variant in VARIANTS:
                                matrix = matrix_as_float(audits[other_variant].get("fused_scores")) or []
                                cross_target_scores[other_variant] = {
                                    index: matrix[index][target_idx]
                                    for index in range(min(candidate_count_cross, len(matrix)))
                                    if target_idx < len(matrix[index])
                                }
                    assignments = {
                        other_variant: assignment_list(
                            frame_maps[other_variant][branch].get(frame, {}).get("candidate_audit")
                        )
                        for other_variant in VARIANTS
                    }
                    m0_assignment = assignments.get("M0")
                    current_assignment = assignments.get(variant)
                    assignment_vs_m0 = (
                        current_assignment != m0_assignment
                        if current_assignment is not None and m0_assignment is not None
                        else None
                    )
                    no_write_assignment = assignment_list(
                        frame_maps[variant]["memory_write_false"].get(frame, {}).get("candidate_audit")
                    )
                    assignment_vs_no_write = (
                        current_assignment != no_write_assignment
                        if current_assignment is not None and no_write_assignment is not None and branch == "memory_write_true"
                        else False if branch == "memory_write_false" and current_assignment is not None
                        else None
                    )

                    cross_scores_for_candidate: dict[int, dict[str, float | None]] = {}
                    if cross_aligned and cross_target_scores:
                        for index in range(len(cross_target_scores.get(variant, {}))):
                            cross_scores_for_candidate[index] = {
                                "M1_minus_M0": (
                                    cross_target_scores["M1"].get(index, 0.0) - cross_target_scores["M0"].get(index, 0.0)
                                ),
                                "M2_minus_M1": (
                                    cross_target_scores["M2"].get(index, 0.0) - cross_target_scores["M1"].get(index, 0.0)
                                ),
                                "M3_minus_M2": (
                                    cross_target_scores["M3"].get(index, 0.0) - cross_target_scores["M2"].get(index, 0.0)
                                ),
                                "M4_minus_M3": (
                                    cross_target_scores["M4"].get(index, 0.0) - cross_target_scores["M3"].get(index, 0.0)
                                ),
                            }
                    cross_for_record = None
                    if detailed and cross_scores_for_candidate:
                        cross_for_record = cross_scores_for_candidate

                    if detailed:
                        records, candidate_status, diagnostics = build_candidate_records(
                            audit,
                            target_public_id,
                            human_embedding,
                            None,
                            assignment_vs_no_write,
                            assignment_vs_m0,
                        )
                        if cross_for_record is not None:
                            for record_item in records:
                                cross_values = cross_for_record.get(record_item["candidate_index"], {})
                                record_item["score_m1_minus_m0_target_state"] = cross_values.get("M1_minus_M0")
                                record_item["score_m2_minus_m1_target_state"] = cross_values.get("M2_minus_M1")
                                record_item["score_m3_minus_m2_target_state"] = cross_values.get("M3_minus_M2")
                                record_item["score_m4_minus_m3_target_state"] = cross_values.get("M4_minus_M3")
                        fused = diagnostics.get("fused_scores") or []
                        cost_matrix = [[-value for value in row] for row in fused]
                        hungarian = {
                            "status": "AVAILABLE",
                            "orientation": "candidate_row_x_public_id_state_column",
                            "cost_definition": "cost=-fused_score; scipy linear_sum_assignment(-fused_score)",
                            "cost_matrix": cost_matrix,
                            "row_candidate_public_ids": diagnostics.get("candidate_public_ids"),
                            "row_candidate_native_ids": diagnostics.get("candidate_native_ids"),
                            "column_public_ids": diagnostics.get("public_id_order"),
                            "assignment_after_scope": assignment_list(audit),
                            "target_row": diagnostics.get("target_row"),
                            "assigned_col": diagnostics.get("assigned_col"),
                            "best_alternative_col": diagnostics.get("best_alternative_col"),
                            "assignment_cost_margin": diagnostics.get("assignment_cost_margin"),
                            "assignment_score_margin": diagnostics.get("assignment_score_margin"),
                        }
                        top1_top2_margin = diagnostics.get("top1_top2_normalized_margin")
                        top2_distinct = diagnostics.get("top2_distinct_public_id")
                        future_near_tie = diagnostics.get("future_near_tie")
                        candidate_ids = diagnostics.get("candidate_public_ids") or []
                        if len(candidate_ids) != len(set(candidate_ids)):
                            summary_by_variant[variant]["candidate_duplicate_frame_count"] += 1
                            if len(candidate_duplicate_examples) < 10:
                                candidate_duplicate_examples.append(f"{event_id}:{variant}:{branch}:{frame}")
                        if mapping_complete is not True:
                            summary_by_variant[variant]["candidate_missing_or_incomplete_frame_count"] += 1
                        for record_item in records:
                            if record_item.get("embedding_cosine_to_human_anchor") is not None:
                                summary_by_variant[variant]["embedding_cosine_count"] += 1
                    else:
                        compact_audit = audit if isinstance(audit, dict) else {}
                        records = []
                        diagnostics = {"status": "UNAVAILABLE_COMPACT_ARTIFACT"}
                        hungarian = {
                            "status": "UNAVAILABLE_COMPACT_ARTIFACT",
                            "assignment_after_scope": assignment_list(compact_audit),
                            "candidate_count": compact_audit.get("candidate_count"),
                            "assignment_count": compact_audit.get("assignment_count"),
                            "candidate_public_id_mapping_complete": compact_audit.get("candidate_public_id_mapping_complete"),
                        }
                        top1_top2_margin = None
                        top2_distinct = None
                        future_near_tie = None
                        if compact_audit.get("candidate_public_id_mapping_complete") is not True:
                            summary_by_variant[variant]["candidate_missing_or_incomplete_frame_count"] += 1

                    # At event+1 the compact summary still gives a safe score-
                    # change scalar; per-candidate change remains unavailable.
                    # Count event-level values once on the write branch, not
                    # once for each member of the paired trace.
                    if (
                        branch == "memory_write_true"
                        and frame == (event_frame + 1 if event_frame is not None else None)
                    ):
                        summary_item = canonical_item.get("score_delta_first_future") or {}
                        max_abs = finite_float(summary_item.get("max_abs_score_delta"))
                        if max_abs is not None and max_abs > 0.0:
                            summary_by_variant[variant]["score_changed_first_future_count"] += 1
                        if bool(summary_item.get("assignment_changed")):
                            summary_by_variant[variant]["assignment_changed_first_future_count"] += 1
                            utility = utility_change(canonical_item)
                            if utility.get("correct_change") is True:
                                summary_by_variant[variant]["correct_assignment_change_first_future_count"] += 1
                        visible = finite_int(get_h20(canonical_item, "no_write_metrics").get("visible_frames"))
                        if visible is not None and visible > 0:
                            summary_by_variant[variant]["target_visible_h20_event_count"] += 1
                        if top1_top2_margin is not None:
                            summary_by_variant[variant]["top1_top2_normalized_margins_event_plus_one"].append(top1_top2_margin)
                        assignment_margin = diagnostics.get("assignment_cost_margin")
                        if assignment_margin is not None:
                            summary_by_variant[variant]["assignment_cost_margins_event_plus_one"].append(assignment_margin)
                        if future_near_tie is True:
                            summary_by_variant[variant]["future_near_tie_event_count"] += 1
                            future_near_tie_by_action[(action, variant)] += 1
                            future_near_tie_by_sequence[(sequence, variant)] += 1
                        if visible is not None and visible <= 0:
                            # Do not increment a second time for malformed zero-
                            # visible windows; N37 has all 24 visible at H20.
                            pass

                    memory_write = branch == "memory_write_true"
                    branch_audit = audit if isinstance(audit, dict) else {}
                    runtime_gt = bool(
                        source_item.get("runtime_future_gt_used")
                        or branch_audit.get("runtime_future_gt_used") is True
                    )
                    if runtime_gt:
                        runtime_future_gt_true.append(f"{event_id}:{variant}:{branch}:{frame}")
                    max_abs_compact = finite_float(branch_audit.get("appearance_score_delta_max_abs"))
                    score_changed_exact = None
                    if detailed and branch == "memory_write_true":
                        false_entry = frame_maps[variant]["memory_write_false"].get(frame, {})
                        false_audit = false_entry.get("candidate_audit") if isinstance(false_entry, dict) else None
                        false_matrix = matrix_as_float(false_audit.get("fused_scores")) if detail_audit_complete(false_audit) else None
                        true_matrix = matrix_as_float(audit.get("fused_scores"))
                        if false_matrix is not None and true_matrix is not None and shape(false_matrix) == shape(true_matrix):
                            score_changed_exact = any(
                                abs(true_matrix[row][col] - false_matrix[row][col]) > 0.0
                                for row in range(len(true_matrix))
                                for col in range(len(true_matrix[row]))
                            )
                    score_changed = score_changed_exact if score_changed_exact is not None else (
                        bool(max_abs_compact is not None and max_abs_compact > 0.0)
                        if memory_write
                        else False
                    )
                    row = {
                        "record_kind": "frame_diagnostic",
                        "frame_phase": "future",
                        "event_id": event_id,
                        "sequence": sequence,
                        "action_type": action,
                        "event_frame": event_frame,
                        "frame": frame,
                        "event_plus_one": bool(event_frame is not None and frame == event_frame + 1),
                        "variant": variant,
                        "branch": branch,
                        "memory_write": memory_write,
                        "memory_read": bool(branch_audit.get("appearance_memory_enabled")) if isinstance(branch_audit, dict) else None,
                        "memory_read_status": "AVAILABLE_FLAG_ONLY" if detailed or isinstance(branch_audit, dict) else "UNAVAILABLE",
                        "current_frame_memory_write_used_for_score": False,
                        "current_frame_score_audit_available": False,
                        "candidate_count": candidate_count,
                        "candidate_mapping_complete": mapping_complete,
                        "candidate_records": records,
                        "candidate_diagnostic_status": candidate_status.get("status") if detailed else "UNAVAILABLE_COMPACT_ARTIFACT",
                        "score_changed_vs_same_variant_no_write": score_changed,
                        "score_change_evidence": "exact_matrix_difference" if score_changed_exact is not None else "compact_first_future_or_branch_summary",
                        "appearance_score_delta_max_abs": max_abs_compact,
                        "top1_top2_margin_normalized": top1_top2_margin,
                        "top2_distinct_public_id": top2_distinct,
                        "future_near_tie": future_near_tie,
                        "near_tie": None,
                        "near_tie_status": "UNAVAILABLE_CURRENT_FRAME_REQUIRED",
                        "hungarian": hungarian,
                        "target_visible_count_h20_posthoc": finite_int(get_h20(canonical_item, "no_write_metrics").get("visible_frames")),
                        "target_mean_iou_h20_posthoc": finite_float(get_h20(canonical_item, "no_write_metrics").get("target_mean_iou")),
                        "target_identity_error_h20_posthoc": finite_float(get_h20(canonical_item, "no_write_metrics").get("target_identity_error_rate")),
                        "candidate_coverage": None,
                        "box_iou": None,
                        "embedding_cosine": "per_candidate_embedding_cosine_to_human_anchor" if detailed else None,
                        "candidate_duplicate_or_missing": (
                            False if detailed and mapping_complete is True and len(diagnostics.get("candidate_public_ids", [])) == len(set(diagnostics.get("candidate_public_ids", []))) else None
                        ),
                        "runtime_future_gt_used": runtime_gt,
                        "source_canonical_artifact": str(source_paths.get(variant, pathlib.Path("" )).relative_to(ROOT)) if variant in source_paths else None,
                        "source_detail_artifact": (
                            str((RAW_ROOT / event_id / f"{variant}.json").relative_to(ROOT))
                            if detailed_data.get(variant) is not None
                            else None
                        ),
                    }
                    table_lines.append(json.dumps(clean_json(row), ensure_ascii=False, separators=(",", ":"), allow_nan=False))
                    table_row_count += 1
                    summary_by_variant[variant]["future_frame_count"] += 1

        # Mark the artifact-level detailed/compact source counts once per event.
        for variant in VARIANTS:
            if detailed_data.get(variant) is not None:
                summary_by_variant[variant]["detailed_artifact_count"] += 1
            else:
                summary_by_variant[variant]["compact_fallback_artifact_count"] += 1

        del canonical_data, detailed_data, frame_maps
        gc.collect()

    # Replace large margin lists with auditable statistics.
    for variant, item in summary_by_variant.items():
        margins = item.pop("assignment_cost_margins_event_plus_one")
        normalized = item.pop("top1_top2_normalized_margins_event_plus_one")
        item["assignment_cost_margin_stats_event_plus_one"] = {
            "count": len(margins),
            "mean": statistics.fmean(margins) if margins else None,
            "p90": percentile(margins, 0.90),
        }
        item["top1_top2_normalized_margin_stats_event_plus_one"] = {
            "count": len(normalized),
            "mean": statistics.fmean(normalized) if normalized else None,
            "p90": percentile(normalized, 0.90),
        }

    def group_summary(group_key: str) -> dict[str, Any]:
        groups: dict[str, dict[str, Any]] = {}
        for event_id in expected_events:
            canonical_event = aggregate_by_event.get(event_id, {})
            group = str(canonical_event.get(group_key, event_meta.get(event_id, {}).get(group_key, "")))
            if group not in groups:
                groups[group] = {"event_count": 0, "variants": {variant: {"score_changed_first_future": 0, "assignment_changed_first_future": 0, "future_near_tie": 0} for variant in VARIANTS}}
            groups[group]["event_count"] += 1
            for variant in VARIANTS:
                item = canonical.get((event_id, variant), (None, {}))[1]
                first = item.get("score_delta_first_future") or {}
                if finite_float(first.get("max_abs_score_delta")) is not None and finite_float(first.get("max_abs_score_delta")) > 0:
                    groups[group]["variants"][variant]["score_changed_first_future"] += 1
                if bool(first.get("assignment_changed")):
                    groups[group]["variants"][variant]["assignment_changed_first_future"] += 1
                groups[group]["variants"][variant]["future_near_tie"] = "available_only_in_21_detailed_events"
        return groups

    detail_artifacts = len(raw_detail_keys)
    compact_artifacts = len(expected_keys - raw_detail_keys)
    summary = {
        "protocol": protocol["protocol"],
        "protocol_hash": protocol["protocol_hash"],
        "status": "BLOCKED_N38_STAGE_A_INPUT_SCHEMA",
        "diagnostic_ready_for_event_selection": False,
        "event_selection_started": False,
        "canonical_artifacts": {
            "artifact_count": len(canonical),
            "event_count": len(canonical_event_ids),
            "variant_count": len(VARIANTS),
            "expected_artifact_count": 24 * 5,
            "unique_key_count": len(canonical),
            "duplicate_key_count": 0 if len(canonical) == len(set(canonical)) else 1,
            "missing_key_count": len(expected_keys - set(canonical)),
            "future_trace_frame_count": canonical_future_frames,
            "expected_future_trace_frame_count": 24 * 5 * 2 * 100,
        },
        "source_detail": {
            "raw_attempt1_success_detail_artifacts": detail_artifacts,
            "canonical_compact_fallback_artifacts": compact_artifacts,
            "raw_failed_artifact_count": len(raw_failed_keys),
            "raw_failed_artifacts": raw_failed_keys,
            "raw_success_detail_is_supplemental_only": True,
            "raw_failed_artifacts_promoted": False,
        },
        "table": {
            "row_count": table_row_count,
            "expected_current_rows": 24 * 5,
            "expected_future_rows": 24 * 5 * 2 * 100,
            "expected_total_rows": 24 * 5 + 24 * 5 * 2 * 100,
            "unique_key_count": len(seen_table_keys),
            "duplicate_key_count": table_row_count - len(seen_table_keys),
            "detailed_future_frame_rows": detailed_frames,
            "compact_future_frame_rows": compact_frames,
            "event_current_rows": event_current_rows,
        },
        "required_detail_field_availability": {
            field: {
                "artifact_count_with_field": detail_artifacts,
                "artifact_count_expected": 120,
                "future_frame_count_with_field": detail_artifacts * 2 * 100,
                "future_frame_count_expected": 24 * 5 * 2 * 100,
                "status": "INCOMPLETE" if detail_artifacts < 120 else "COMPLETE",
            }
            for field in REQUIRED_DETAIL_FIELDS
        },
        "alignment_and_integrity": {
            "schema_errors": schema_errors,
            "trace_alignment_errors": trace_alignment_errors,
            "raw_vs_canonical_summary_mismatch_count": len(raw_vs_canonical_mismatches),
            "raw_vs_canonical_summary_mismatches": raw_vs_canonical_mismatches[:20],
            "current_frame_exclusion_errors": current_frame_exclusion_errors,
            "current_frame_score_audit_artifact_count": 0,
            "current_frame_score_audit_expected_count": 24 * 5,
            "runtime_future_gt_true_count": len(runtime_future_gt_true),
            "runtime_future_gt_true_examples": runtime_future_gt_true[:20],
            "current_frame_memory_write_used_count": len(current_memory_write_used),
            "current_frame_memory_write_used_examples": current_memory_write_used[:20],
            "candidate_duplicate_examples": candidate_duplicate_examples,
            "candidate_duplicate_count_detailed_frames": sum(item["candidate_duplicate_frame_count"] for item in summary_by_variant.values()),
            "candidate_completeness_unknown_compact_frames": compact_frames,
        },
        "by_variant": summary_by_variant,
        "by_action": group_summary("action_type"),
        "by_sequence": group_summary("sequence"),
        "by_near_tie": {
            "current_frame": "UNAVAILABLE_EVENT_FRAME_SCORE_AUDIT",
            "joint_current_and_event_plus_one": "UNAVAILABLE_CURRENT_FRAME_REQUIRED",
            "future_event_plus_one_detailed_counts": {
                f"{action}:{variant}": count for (action, variant), count in sorted(future_near_tie_by_action.items())
            },
            "future_sequence_detailed_counts": {
                f"{sequence}:{variant}": count for (sequence, variant), count in sorted(future_near_tie_by_sequence.items())
            },
        },
        "required_but_unavailable": {
            "current_frame_per_candidate_score_rank_margin": "all 120 artifacts; N37 future trace excludes event frame",
            "current_frame_hungarian_cost_matrix": "all 120 artifacts; not retained in N37 replay artifact",
            "future_per_candidate_score_rank_cost_for_compact_fallback": "15 artifacts; N37 canonical compaction removed fields",
            "future_candidate_coverage_and_per_frame_gt_iou": "not present as per-frame fields; only posthoc horizon summaries exist",
            "full_24x5_current_plus_future_near_tie_decision": "not computable without a new replay or a new lossless artifact, so event selection is forbidden",
        },
        "posthoc_context_not_used_for_selection": {
            "n37_first_future_score_change_rate": {variant: item["score_changed_first_future_count"] for variant, item in summary_by_variant.items()},
            "n37_first_future_assignment_change_rate": {variant: item["assignment_changed_first_future_count"] for variant, item in summary_by_variant.items()},
            "n37_first_future_correct_assignment_change_rate": {variant: item["correct_assignment_change_first_future_count"] for variant, item in summary_by_variant.items()},
        },
        "downstream_authorized": False,
    }

    # The JSONL itself is the large artifact; commit it only after every row
    # has been validated and the summary proves key uniqueness.
    if table_row_count != 24 * 5 + 24 * 5 * 2 * 100:
        raise RuntimeError(f"unexpected table row count {table_row_count}")
    if len(seen_table_keys) != table_row_count:
        raise RuntimeError("duplicate table keys detected")
    atomic_text(DIAGNOSTIC / "score_assignment_table.jsonl", "\n".join(table_lines) + "\n")
    atomic_json(DIAGNOSTIC / "score_assignment_summary.json", clean_json(summary))

    stage_01 = {
        "stage": "N38-01",
        "status": "BLOCKED",
        "real_data_status": "PASS_INPUT_SCHEMA_AUDIT_BLOCKED",
        "downstream_authorized": False,
        "event_selection_started": False,
        "reason_code": "N37_REPLAY_ARTIFACT_SCHEMA_INCOMPLETE_FOR_N38",
        "reason": "N37 future traces exclude the event frame, and 15 of 120 canonical artifacts are compact-only; the required full current-frame plus event+1 per-candidate score/rank/Hungarian audit cannot be computed without a new replay.",
        "canonical_artifact_count": len(canonical),
        "canonical_event_count": len(canonical_event_ids),
        "raw_success_detail_artifact_count": detail_artifacts,
        "compact_fallback_artifact_count": compact_artifacts,
        "current_frame_score_audit_count": 0,
        "required_current_frame_score_audit_count": 120,
        "future_compact_fallback_artifact_count": compact_artifacts,
        "runtime_future_gt_used": bool(runtime_future_gt_true),
        "duplicate_key_count": summary["canonical_artifacts"]["duplicate_key_count"],
        "missing_key_count": summary["canonical_artifacts"]["missing_key_count"],
        "table_duplicate_key_count": summary["table"]["duplicate_key_count"],
        "trace_alignment_error_count": len(trace_alignment_errors),
        "raw_failed_artifacts_preserved": raw_failed_keys,
        "artifacts": [
            "outputs/n38/diagnostic/diagnostic_protocol.json",
            "outputs/n38/diagnostic/score_assignment_table.jsonl",
            "outputs/n38/diagnostic/score_assignment_summary.json",
        ],
        "n37_outputs_modified": False,
        "no_event_selection": True,
        "no_full_loop": True,
        "no_replay": True,
        "no_training": True,
    }
    atomic_json(N38 / "stage_01_status.json", clean_json(stage_01))
    print(json.dumps({
        "status": stage_01["status"],
        "canonical_artifacts": len(canonical),
        "events": len(canonical_event_ids),
        "raw_success_detail": detail_artifacts,
        "compact_fallback": compact_artifacts,
        "current_frame_score_audit": 0,
        "table_rows": table_row_count,
        "runtime_future_gt_true": len(runtime_future_gt_true),
        "trace_alignment_errors": len(trace_alignment_errors),
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"N38 Stage-A diagnostic failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise
