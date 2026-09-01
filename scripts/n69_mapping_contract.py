"""Versioned, lossless mapping contract for the N69 sidecar experiments.

The frozen N54 artifacts expose ``native_tid`` and ``public_id`` but do not
expose a separately named local/global identifier or the source metadata that
would be required to invent one.  This module therefore keeps those fields
explicitly nullable, records the provenance gap, and only promotes a target
mapping when the offline intervention supplies both the target native row and
the authoritative public ID.  It never mutates a frozen frame.
"""

from __future__ import annotations

import hashlib
import math
from typing import Any, Iterable

import numpy as np


MAPPING_SCHEMA = "N69_VERSIONED_NATIVE_LOCAL_GLOBAL_PUBLIC_MAPPING_V1"
MAPPING_VERSION = "n69-target-boundary-v1"
REQUIRED_EVIDENCE_KEYS = (
    "sequence",
    "frame",
    "native_id",
    "local_id",
    "global_id",
    "public_id",
    "source",
    "source_version",
    "valid_from_frame",
    "valid_to_frame",
    "confidence",
    "provenance",
)


class MappingContractError(ValueError):
    """Raised when a frame cannot satisfy the structural mapping contract."""


def finite_float(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _as_box(value: Any) -> np.ndarray | None:
    try:
        arr = np.asarray(value, dtype=np.float64).reshape(-1)
    except (TypeError, ValueError):
        return None
    if arr.size != 4 or not np.all(np.isfinite(arr)):
        return None
    return arr


def box_iou(left: Any, right: Any) -> float:
    a, b = _as_box(left), _as_box(right)
    if a is None or b is None:
        return 0.0
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter
    return float(inter / union) if union > 0.0 else 0.0


def _unit(value: Any, dim: int = 512) -> np.ndarray | None:
    try:
        arr = np.asarray(value, dtype=np.float32).reshape(-1)
    except (TypeError, ValueError):
        return None
    if arr.size != dim or not np.all(np.isfinite(arr)):
        return None
    norm = float(np.linalg.norm(arr))
    return arr / norm if norm > 1e-6 else None


def cosine(left: Any, right: Any) -> float | None:
    a, b = _unit(left), _unit(right)
    if a is None or b is None:
        return None
    return float(np.dot(a, b))


def feature_digest(value: Any) -> str | None:
    arr = _unit(value)
    if arr is None:
        return None
    return hashlib.sha256(arr.astype(np.float32).tobytes()).hexdigest()


def make_mapping_evidence(
    *,
    sequence: str,
    frame: int,
    native_id: int | None,
    local_id: int | str | None,
    global_id: int | str | None,
    public_id: int | None,
    valid_from_frame: int,
    valid_to_frame: int,
    confidence: float | None,
    source: str,
    source_version: str = MAPPING_VERSION,
    provenance: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "schema": MAPPING_SCHEMA,
        "version": source_version,
        "sequence": str(sequence),
        "frame": int(frame),
        "native_id": None if native_id is None else int(native_id),
        "local_id": local_id,
        "global_id": global_id,
        "public_id": None if public_id is None else int(public_id),
        "source": source,
        "source_version": source_version,
        "valid_from_frame": int(valid_from_frame),
        "valid_to_frame": int(valid_to_frame),
        "confidence": finite_float(confidence),
        "provenance": dict(provenance or {}),
    }


def validate_mapping_evidence(record: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    missing = [key for key in REQUIRED_EVIDENCE_KEYS if key not in record]
    if missing:
        errors.append(f"missing_evidence_fields:{','.join(missing)}")
    if record.get("schema") != MAPPING_SCHEMA:
        errors.append("schema_mismatch")
    if record.get("source_version") != MAPPING_VERSION:
        errors.append("source_version_mismatch")
    if record.get("sequence") in (None, ""):
        errors.append("sequence_missing")
    if record.get("native_id") is None and record.get("public_id") is None:
        errors.append("native_and_public_missing")
    try:
        start, end, frame = int(record["valid_from_frame"]), int(record["valid_to_frame"]), int(record["frame"])
        if start > end or frame < start or frame > end:
            errors.append("invalid_valid_range")
    except (KeyError, TypeError, ValueError):
        errors.append("invalid_frame_range")
    confidence = finite_float(record.get("confidence"))
    if confidence is None or not 0.0 <= confidence <= 1.0:
        errors.append("confidence_not_finite_or_out_of_range")
    provenance = record.get("provenance")
    if not isinstance(provenance, dict) or not provenance:
        errors.append("provenance_missing")
    return errors


def _public_id(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def validate_candidate_branch(
    branch: dict[str, Any],
    *,
    expected_sequence: str | None = None,
    expected_frame: int | None = None,
) -> dict[str, Any]:
    """Validate row/candidate/public axes without consulting GT.

    The return value is an audit record, not a repaired frame.  Missing
    local/global fields are reported as provenance gaps because the frozen
    artifact never contained those fields; no synthetic identifier is made.
    """

    errors: list[str] = []
    candidate_rows = branch.get("candidate_rows", [])
    rows = branch.get("rows", [])
    public_order = branch.get("public_id_order", [])
    frame = branch.get("frame")
    if not isinstance(candidate_rows, list) or not isinstance(rows, list):
        errors.append("candidate_rows_or_rows_not_list")
        candidate_rows = candidate_rows if isinstance(candidate_rows, list) else []
        rows = rows if isinstance(rows, list) else []
    if len(candidate_rows) != len(rows):
        errors.append("candidate_row_count_mismatch")
    if expected_frame is not None and frame != expected_frame:
        errors.append("frame_mismatch")
    if branch.get("runtime_future_gt_used") is not False:
        errors.append("runtime_future_gt_used_not_false")
    if not isinstance(public_order, list):
        errors.append("public_id_order_not_list")
        public_order = []
    public_ids = [_public_id(value) for value in public_order]
    nonnull_public = [value for value in public_ids if value is not None]
    duplicate_public_axis = sorted({value for value in nonnull_public if nonnull_public.count(value) > 1})
    if duplicate_public_axis:
        errors.append("duplicate_public_id_axis")

    native_ids: list[int] = []
    row_public_ids: list[int | None] = []
    missing_local_global = 0
    duplicate_native: list[int] = []
    for index, (candidate, mapped) in enumerate(zip(candidate_rows, rows)):
        if not isinstance(candidate, dict) or not isinstance(mapped, dict):
            errors.append(f"row_{index}_not_object")
            continue
        native = candidate.get("native_tid", mapped.get("native_tid"))
        try:
            native_int = int(native)
            native_ids.append(native_int)
            if native_ids.count(native_int) > 1:
                duplicate_native.append(native_int)
        except (TypeError, ValueError):
            errors.append(f"row_{index}_native_id_missing")
        row_public = _public_id(mapped.get("public_id"))
        row_public_ids.append(row_public)
        if "local_id" not in candidate and "local_id" not in mapped:
            missing_local_global += 1
        if "global_id" not in candidate and "global_id" not in mapped:
            missing_local_global += 1
        if _as_box(candidate.get("box")) is None:
            errors.append(f"row_{index}_box_invalid")
    if duplicate_native:
        errors.append("duplicate_native_id_rows")
    mapped_public = [value for value in row_public_ids if value is not None]
    duplicate_row_public = sorted({value for value in mapped_public if mapped_public.count(value) > 1})
    if duplicate_row_public:
        errors.append("duplicate_row_public_id")
    if expected_sequence is not None and branch.get("sequence") not in (None, expected_sequence):
        errors.append("sequence_mismatch")
    return {
        "valid": not errors,
        "errors": sorted(set(errors)),
        "frame": frame,
        "candidate_count": len(candidate_rows),
        "native_ids": native_ids,
        "row_public_ids": row_public_ids,
        "public_id_order": public_ids,
        "duplicate_public_axis": duplicate_public_axis,
        "duplicate_row_public_ids": duplicate_row_public,
        "missing_local_global_field_count": missing_local_global,
        "runtime_future_gt_used": branch.get("runtime_future_gt_used"),
    }


def reconcile_target_boundary(
    branch: dict[str, Any],
    *,
    sequence: str,
    event_frame: int,
    future_end_frame: int,
    target_native_id: int,
    target_public_id: int,
    event_provenance: dict[str, Any],
) -> dict[str, Any]:
    """Resolve only the explicitly supplied target at the intervention edge.

    Existing row mappings are retained for audit.  If another row already
    claims the target public ID, the conflict is surfaced and the target
    boundary mapping is marked as a deterministic displacement candidate; the
    frozen frame is never silently overwritten.
    """

    structural = validate_candidate_branch(branch, expected_sequence=sequence)
    candidates = branch.get("candidate_rows", [])
    rows = branch.get("rows", [])
    native_hits = [
        index for index, row in enumerate(candidates)
        if isinstance(row, dict) and _public_id(row.get("native_tid")) == int(target_native_id)
    ]
    public_hits = [
        index for index, row in enumerate(rows)
        if isinstance(row, dict) and _public_id(row.get("public_id")) == int(target_public_id)
    ]
    target_row = native_hits[0] if len(native_hits) == 1 else None
    target_column = None
    for index, value in enumerate(structural["public_id_order"]):
        if value == int(target_public_id):
            target_column = index
            break
    old_public = None
    if target_row is not None and target_row < len(rows) and isinstance(rows[target_row], dict):
        old_public = _public_id(rows[target_row].get("public_id"))
    conflict_rows = [index for index in public_hits if index != target_row]
    provenance_gap = []
    if target_row is not None:
        row = candidates[target_row]
        mapped = rows[target_row] if target_row < len(rows) else {}
        if "local_id" not in row and "local_id" not in mapped:
            provenance_gap.append("local_id_absent_in_frozen_candidate_artifact")
        if "global_id" not in row and "global_id" not in mapped:
            provenance_gap.append("global_id_absent_in_frozen_candidate_artifact")
    mapping = make_mapping_evidence(
        sequence=sequence,
        frame=int(branch.get("frame", event_frame + 1)),
        native_id=int(target_native_id),
        local_id=(candidates[target_row].get("local_id") if target_row is not None else None),
        global_id=(candidates[target_row].get("global_id") if target_row is not None else None),
        public_id=int(target_public_id),
        valid_from_frame=int(event_frame),
        valid_to_frame=int(future_end_frame),
        confidence=1.0 if target_row is not None and target_column is not None and not structural["errors"] else 0.0,
        source="offline_explicit_event_public_id_at_intervention_boundary",
        provenance=event_provenance,
    )
    mapping_errors = validate_mapping_evidence(mapping)
    return {
        "schema": MAPPING_SCHEMA,
        "sequence": sequence,
        "frame": branch.get("frame"),
        "event_frame": int(event_frame),
        "future_end_frame": int(future_end_frame),
        "target_native_id": int(target_native_id),
        "target_public_id": int(target_public_id),
        "target_physical_row": target_row,
        "old_public_id_at_target_row": old_public,
        "old_public_id_rows": public_hits,
        "target_public_column": target_column,
        "conflict_rows_claiming_target_public": conflict_rows,
        "target_row_resolved": target_row is not None and target_column is not None,
        "old_mapping_matches_boundary": old_public == int(target_public_id),
        "resolution": (
            "explicit_target_boundary_mapping_no_old_conflict"
            if target_row is not None and target_column is not None and not conflict_rows and old_public == int(target_public_id)
            else "explicit_target_boundary_mapping_replaces_or_adds_old_scope_with_conflict_audited"
            if target_row is not None and target_column is not None
            else "unresolved_target_native_or_public_column"
        ),
        "provenance_gap": provenance_gap,
        "mapping_evidence": mapping,
        "mapping_evidence_errors": mapping_errors,
        "structural": structural,
        "runtime_future_gt_used": branch.get("runtime_future_gt_used"),
    }


def match_overlap_rows(
    previous_rows: Iterable[dict[str, Any]],
    current_rows: Iterable[dict[str, Any]],
    *,
    previous_frame: int,
    current_frame: int,
    min_score: float = 0.35,
) -> dict[str, Any]:
    """Deterministically match an overlap using native, box, feature, time.

    Native identity is preferred when unique.  Otherwise the score combines
    box overlap, non-negative embedding cosine, and a continuity term.  A
    one-to-one conflict or a score below the fixed floor is returned as a
    failure rather than guessed through.
    """

    prev = [row for row in previous_rows if isinstance(row, dict)]
    curr = [row for row in current_rows if isinstance(row, dict)]
    matches: list[dict[str, Any]] = []
    used: set[int] = set()
    failures: list[str] = []
    for pi, prow in enumerate(prev):
        native = _public_id(prow.get("native_tid"))
        exact = [ci for ci, crow in enumerate(curr) if _public_id(crow.get("native_tid")) == native] if native is not None else []
        candidates = exact if len(exact) == 1 else list(range(len(curr)))
        scored: list[tuple[float, int, float, float, float]] = []
        for ci in candidates:
            if ci in used:
                continue
            crow = curr[ci]
            iou = box_iou(prow.get("box"), crow.get("box"))
            cos = cosine(prow.get("feature", prow.get("feat")), crow.get("feature", crow.get("feat")))
            cos_term = max(0.0, cos if cos is not None else 0.0)
            gap = max(0, int(current_frame) - int(previous_frame))
            time_term = 1.0 / (1.0 + float(gap))
            score = 0.55 * iou + 0.35 * cos_term + 0.10 * time_term
            scored.append((score, ci, iou, cos_term, time_term))
        scored.sort(key=lambda item: (-item[0], item[1]))
        if not scored or scored[0][0] < min_score:
            failures.append(f"no_confident_overlap_match_prev_{pi}")
            continue
        best = scored[0]
        if len(scored) > 1 and abs(best[0] - scored[1][0]) < 1e-9:
            failures.append(f"ambiguous_overlap_match_prev_{pi}")
            continue
        used.add(best[1])
        matches.append({"previous_index": pi, "current_index": best[1], "score": best[0], "box_iou": best[2], "embedding_cosine_nonnegative": best[3], "time_continuity": best[4]})
    return {
        "previous_frame": int(previous_frame),
        "current_frame": int(current_frame),
        "matches": matches,
        "unmatched_previous": [index for index in range(len(prev)) if index not in {item["previous_index"] for item in matches}],
        "unmatched_current": [index for index in range(len(curr)) if index not in used],
        "failures": failures,
        "valid": not failures,
        "min_score": float(min_score),
    }


def validate_causal_boundary(*, event_frame: int, observed_frame: int, memory_read: bool) -> dict[str, Any]:
    errors: list[str] = []
    if int(observed_frame) == int(event_frame) and memory_read:
        errors.append("event_frame_reads_new_memory")
    if int(observed_frame) <= int(event_frame) and not memory_read:
        errors.append("event_frame_observation_not_explicitly_post_write")
    return {
        "event_frame": int(event_frame),
        "observed_frame": int(observed_frame),
        "memory_read": bool(memory_read),
        "first_visible_frame": int(event_frame) + 1,
        "valid": not errors and int(observed_frame) >= int(event_frame) + 1,
        "errors": errors,
    }


def run_fixture_tests() -> dict[str, Any]:
    """Focused non-scientific contract fixtures used by N69 Stage 01."""

    def frame(rows: list[dict[str, Any]], *, number: int = 11, future_gt: bool = False) -> dict[str, Any]:
        return {
            "frame": number,
            "runtime_future_gt_used": future_gt,
            "public_id_order": [101, 102, 103],
            "candidate_rows": [
                {"index": i, "native_tid": row["native_tid"], "box": row.get("box", [0, 0, 10, 10]), "feature": row.get("feature", [1.0] + [0.0] * 511)}
                for i, row in enumerate(rows)
            ],
            "rows": [
                {"candidate_index": i, "native_tid": row["native_tid"], "public_id": row.get("public_id")}
                for i, row in enumerate(rows)
            ],
        }

    feature_a = [1.0] + [0.0] * 511
    feature_b = [0.0, 1.0] + [0.0] * 510
    tests: dict[str, bool] = {}
    correct = frame([{"native_tid": 11, "public_id": 101, "feature": feature_a}, {"native_tid": 12, "public_id": 102, "feature": feature_b}])
    correct_result = reconcile_target_boundary(correct, sequence="toy", event_frame=10, future_end_frame=20, target_native_id=11, target_public_id=101, event_provenance={"interaction_source": "simulated_from_gt", "real_human_tape": False})
    tests["correct_mapping"] = bool(correct_result["target_row_resolved"] and correct_result["old_mapping_matches_boundary"])
    wrong = frame([{"native_tid": 11, "public_id": 102, "feature": feature_a}, {"native_tid": 12, "public_id": 101, "feature": feature_b}])
    wrong_result = reconcile_target_boundary(wrong, sequence="toy", event_frame=10, future_end_frame=20, target_native_id=11, target_public_id=101, event_provenance={"interaction_source": "simulated_from_gt", "real_human_tape": False})
    tests["wrong_mapping_is_audited_not_silently_overwritten"] = bool(wrong_result["target_row_resolved"] and wrong_result["conflict_rows_claiming_target_public"] == [1])
    missing = frame([{"native_tid": 12, "public_id": 102, "feature": feature_b}])
    missing_result = reconcile_target_boundary(missing, sequence="toy", event_frame=10, future_end_frame=20, target_native_id=11, target_public_id=101, event_provenance={"interaction_source": "simulated_from_gt", "real_human_tape": False})
    tests["missing_candidate_rejected"] = missing_result["target_row_resolved"] is False
    duplicate = frame([{"native_tid": 11, "public_id": 101}, {"native_tid": 12, "public_id": 101}])
    tests["duplicate_public_mapping_rejected"] = "duplicate_row_public_id" in validate_candidate_branch(duplicate)["errors"]
    tests["missing_frame_rejected"] = "frame_mismatch" in validate_candidate_branch(correct, expected_frame=12)["errors"]
    tests["future_gt_rejected"] = "runtime_future_gt_used_not_false" in validate_candidate_branch(frame([{"native_tid": 11, "public_id": 101}], future_gt=True))["errors"]
    tests["event_frame_new_memory_rejected"] = validate_causal_boundary(event_frame=10, observed_frame=10, memory_read=True)["valid"] is False
    tests["event_plus_one_is_causal"] = validate_causal_boundary(event_frame=10, observed_frame=11, memory_read=True)["valid"] is True
    overlap_missing = match_overlap_rows([{"native_tid": 11, "box": [0, 0, 10, 10], "feature": feature_a}], [], previous_frame=10, current_frame=11)
    tests["overlap_missing_rejected"] = overlap_missing["valid"] is False
    overlap_duplicate = match_overlap_rows([{"native_tid": 11, "box": [0, 0, 10, 10], "feature": feature_a}], [{"native_tid": 11, "box": [0, 0, 10, 10], "feature": feature_a}, {"native_tid": 11, "box": [0, 0, 10, 10], "feature": feature_a}], previous_frame=10, current_frame=11)
    tests["overlap_duplicate_rejected"] = len(overlap_duplicate["failures"]) > 0
    tests["simulated_not_real_human"] = correct_result["mapping_evidence"]["provenance"]["interaction_source"] == "simulated_from_gt" and correct_result["mapping_evidence"]["provenance"]["real_human_tape"] is False
    return {"schema": "N69_MAPPING_CONTRACT_FIXTURE_RESULTS_V1", "status": "PASS" if all(tests.values()) else "FAIL", "tests": tests, "passed": sum(tests.values()), "total": len(tests)}

