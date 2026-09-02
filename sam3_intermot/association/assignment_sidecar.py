"""Same-run candidate/public assignment sidecar for N72R1.

This module consumes one StateManager audit and the candidate rows emitted by
the same run.  It never joins another run, GT, IoU, appearance, or a MOT text
file.  Public IDs are emitted only when an explicit ``PublicAuthorityResolver``
is supplied; otherwise the sidecar remains useful but records the authority
gap explicitly.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from typing import Any, Iterable, Mapping

import numpy as np

from sam3_intermot.provenance.mapping_v2 import (
    CANDIDATE_ASSIGNMENT_STATUSES,
    INTEGRITY_STATUSES,
    PUBLIC_ASSIGNMENT_STATUSES,
    PublicAuthorityResolver,
)


SCHEMA_VERSION = "N72R1_SAME_RUN_ASSIGNMENT_SIDECAR_V1"


def _sha256_json(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def _matrix(audit: Mapping[str, Any]) -> np.ndarray:
    values = audit.get("fused_scores", audit.get("scores", []))
    try:
        result = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError):
        return np.zeros((0, 0), dtype=np.float64)
    return result if result.ndim == 2 else np.zeros((0, 0), dtype=np.float64)


def _assignment(audit: Mapping[str, Any], n: int) -> np.ndarray:
    values = audit.get("assignment_after_scope", audit.get("assignment", []))
    try:
        result = np.asarray(values, dtype=int).reshape(-1)
    except (TypeError, ValueError):
        return np.full(n, -1, dtype=int)
    if result.size != n:
        return np.full(n, -1, dtype=int)
    return result


def _axis_status(row: Mapping[str, Any], expected_run: str | None, expected_session: str | None) -> tuple[str, list[str]]:
    errors: list[str] = []
    if expected_run is not None and str(row.get("source_run_id")) != str(expected_run):
        errors.append("source_run_mismatch")
    if expected_session is not None and str(row.get("session_id")) != str(expected_session):
        errors.append("session_mismatch")
    for key in ("candidate_uid", "source_run_id", "session_id", "segment_id", "window_id", "chunk_id", "segment_local_id", "sequence_global_id"):
        if row.get(key) in (None, ""):
            errors.append(f"missing_{key}")
    if row.get("official_raw_sam_id") is None:
        errors.append("missing_official_raw_sam_id")
    if row.get("adapter_external_id") is None:
        errors.append("missing_adapter_external_id")
    return ("EXACT" if not errors else "MISSING_PROVENANCE" if any(item.startswith("missing_") for item in errors) else "SOURCE_RUN_MISMATCH" if "source_run_mismatch" in errors else "SESSION_MISMATCH", errors)


def build_assignment_sidecar(
    candidate_rows: Iterable[Mapping[str, Any]],
    association_audit: Mapping[str, Any],
    *,
    resolver: PublicAuthorityResolver | None = None,
    source_run_id: str | None = None,
    session_id: str | None = None,
    threshold: float | None = None,
) -> dict[str, Any]:
    rows = [dict(row) for row in candidate_rows]
    matrix = _matrix(association_audit)
    candidate_axis = [row.get("candidate_uid") for row in rows]
    state_axis = [int(value) for value in association_audit.get("association_state_axis", association_audit.get("public_id_order", []))]
    if source_run_id is None:
        source_run_id = association_audit.get("source_run_id")
    if session_id is None:
        session_id = association_audit.get("session_id")
    if resolver is not None:
        if source_run_id is not None and resolver.source_run_id != str(source_run_id):
            raise ValueError("resolver/source_run_id mismatch")
        if session_id is not None and resolver.session_id != str(session_id):
            raise ValueError("resolver/session_id mismatch")
    public_axis: list[int | None] = []
    supplied_public_axis = association_audit.get("public_id_axis")
    if isinstance(supplied_public_axis, list) and len(supplied_public_axis) == len(state_axis):
        public_axis = [
            (
                None if value is None and resolver is None else
                resolver.resolve(state_axis[index]) if value is None else int(value)
            )
            for index, value in enumerate(supplied_public_axis)
        ]
    else:
        public_axis = [None if resolver is None else resolver.resolve(state_id) for state_id in state_axis]
    if len(public_axis) != len(state_axis):
        public_axis = [None] * len(state_axis)
    public_values = [value for value in public_axis if value is not None]
    if len(public_values) != len(set(public_values)):
        errors = [{"code": "public_id_collision", "public_id_axis": public_axis}]
    else:
        errors = []
    errors: list[dict[str, Any]] = list(errors)
    uid_counts = Counter(str(value) for value in candidate_axis if value is not None)
    for uid, count in sorted(uid_counts.items()):
        if count > 1:
            errors.append({"code": "candidate_uid_collision", "candidate_uid": uid, "count": count})
    row_integrity: list[str] = []
    for index, row in enumerate(rows):
        status, row_errors = _axis_status(row, source_run_id, session_id)
        row_integrity.append(status)
        for error in row_errors:
            errors.append({"code": error, "row": index})
    if len(candidate_axis) != matrix.shape[0] or len(state_axis) != matrix.shape[1]:
        errors.append({"code": "score_matrix_axis_shape_mismatch", "matrix_shape": list(matrix.shape), "candidate_count": len(candidate_axis), "state_count": len(state_axis)})
    assignment = _assignment(association_audit, len(rows))
    score_threshold = float(association_audit.get("score_threshold", 0.0) if threshold is None else threshold)
    explicit_none_indices = {int(value) for value in association_audit.get("explicit_none_state_indices", []) if isinstance(value, int) and not isinstance(value, bool)}
    candidate_assignment_rows: list[dict[str, Any]] = []
    for i, row in enumerate(rows):
        state_index = int(assignment[i]) if i < assignment.size else -1
        assigned = 0 <= state_index < len(state_axis)
        score = None
        if assigned and i < matrix.shape[0] and state_index < matrix.shape[1]:
            score = float(matrix[i, state_index])
        accepted = bool(assigned and score is not None and score >= score_threshold)
        state_id = state_axis[state_index] if assigned else None
        public_id = public_axis[state_index] if assigned and accepted else None
        if not assigned:
            assignment_status = "UNASSIGNED_CANDIDATE"
        elif not accepted:
            assignment_status = "REJECTED_BELOW_THRESHOLD"
        elif public_id is None:
            assignment_status = "CANDIDATE_ONLY_NO_PUBLIC_AUTHORITY"
        else:
            assignment_status = "ASSIGNED_TO_EXISTING_PUBLIC"
        candidate_assignment_rows.append(
            {
                "candidate_uid": row.get("candidate_uid"),
                "candidate_index": int(row.get("candidate_index", i)),
                "association_state_id": state_id,
                "public_id": public_id,
                "score": score,
                "status": assignment_status,
                "birth_new_identity": False,
                "source_run_id": row.get("source_run_id", source_run_id),
                "session_id": row.get("session_id", session_id),
                "integrity_status": row_integrity[i],
            }
        )

    public_assignment_rows: list[dict[str, Any]] = []
    for state_index, state_id in enumerate(state_axis):
        public_id = public_axis[state_index]
        matched = [item for item in candidate_assignment_rows if item.get("association_state_id") == state_id and item.get("status") in {"ASSIGNED_TO_EXISTING_PUBLIC", "ASSIGNED_TO_NEW_PUBLIC"}]
        if matched:
            status = "ASSIGNED_CANDIDATE"
            candidate_uid = matched[0].get("candidate_uid")
            assigned_score = matched[0].get("score")
        elif state_index in explicit_none_indices:
            status = "EXPLICIT_NONE"
            candidate_uid = None
            assigned_score = None
        elif public_id is None:
            status = "PUBLIC_ASSIGNMENT_ARTIFACT_ABSENT"
            candidate_uid = None
            assigned_score = None
        else:
            # The current StateManager solver has no explicit NONE columns;
            # absence of a pair is therefore not silently relabelled NONE.
            status = "PUBLIC_ASSIGNMENT_ARTIFACT_ABSENT"
            candidate_uid = None
            assigned_score = None
        public_assignment_rows.append(
            {
                "public_id": public_id,
                "association_state_id": int(state_id),
                "candidate_uid": candidate_uid,
                "public_assignment_status": status,
                "assigned_score": assigned_score,
                "none_score": None,
                "source_run_id": source_run_id,
                "session_id": session_id,
            }
        )

    integrity_status = "EXACT"
    if any(item["integrity_status"] != "EXACT" for item in candidate_assignment_rows):
        integrity_status = "SOURCE_RUN_MISMATCH" if any(item["code"] == "source_run_mismatch" for item in errors) else "SESSION_MISMATCH" if any(item["code"] == "session_mismatch" for item in errors) else "MISSING_PROVENANCE"
    if any(item["code"] == "candidate_uid_collision" for item in errors):
        integrity_status = "COLLISION"
    return {
        "schema_version": SCHEMA_VERSION,
        "source_run_id": source_run_id,
        "session_id": session_id,
        "integrity_status": integrity_status,
        "candidate_axis": candidate_axis,
        "association_state_axis": state_axis,
        "public_id_axis": public_axis,
        "public_authority_present": any(value is not None for value in public_axis),
        "score_matrix_shape": list(matrix.shape),
        "score_matrix_sha256": _sha256_json(matrix.tolist()),
        "score_threshold": score_threshold,
        "solver_version": "scipy.optimize.linear_sum_assignment_via_StateManager",
        "assignment_before_scope": [int(value) for value in np.asarray(association_audit.get("assignment", []), dtype=int).reshape(-1).tolist()],
        "assignment_after_scope": [int(value) for value in assignment.tolist()],
        "explicit_none_state_indices": sorted(explicit_none_indices),
        "assignment_solver_has_explicit_none": bool(explicit_none_indices),
        "candidate_assignment_rows": candidate_assignment_rows,
        "public_assignment_rows": public_assignment_rows,
        "candidate_assignment_statuses": dict(Counter(item["status"] for item in candidate_assignment_rows)),
        "public_assignment_statuses": dict(Counter(item["public_assignment_status"] for item in public_assignment_rows)),
        "source_run_mismatch_count": sum(1 for item in errors if item["code"] == "source_run_mismatch"),
        "session_mismatch_count": sum(1 for item in errors if item["code"] == "session_mismatch"),
        "candidate_uid_collision_count": sum(count - 1 for count in uid_counts.values() if count > 1),
        "errors": errors,
        "runtime_future_gt_used": False,
    }


def validate_assignment_sidecar(sidecar: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    if sidecar.get("schema_version") != SCHEMA_VERSION:
        errors.append("schema_version_mismatch")
    if sidecar.get("runtime_future_gt_used") is not False:
        errors.append("runtime_future_gt_used_not_false")
    candidate_axis = sidecar.get("candidate_axis")
    state_axis = sidecar.get("association_state_axis")
    public_axis = sidecar.get("public_id_axis")
    if not isinstance(candidate_axis, list) or len(candidate_axis) != len(sidecar.get("candidate_assignment_rows", [])):
        errors.append("candidate_axis_rows_mismatch")
    if not isinstance(state_axis, list) or not isinstance(public_axis, list) or len(state_axis) != len(public_axis):
        errors.append("state_public_axis_mismatch")
    if sidecar.get("candidate_uid_collision_count") != 0:
        errors.append("candidate_uid_collision")
    for item in sidecar.get("candidate_assignment_rows", []):
        if item.get("status") not in CANDIDATE_ASSIGNMENT_STATUSES:
            errors.append("candidate_assignment_status_invalid")
    for item in sidecar.get("public_assignment_rows", []):
        if item.get("public_assignment_status") not in PUBLIC_ASSIGNMENT_STATUSES:
            errors.append("public_assignment_status_invalid")
    return errors


def schema_document() -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "integrity_statuses": list(INTEGRITY_STATUSES),
        "candidate_assignment_statuses": list(CANDIDATE_ASSIGNMENT_STATUSES),
        "public_assignment_statuses": list(PUBLIC_ASSIGNMENT_STATUSES),
        "same_run_key": ["source_run_id", "session_id", "candidate_uid"],
        "public_id_rule": "only explicit PublicAuthorityResolver bindings may populate public_id",
        "explicit_none_rule": "absence of a pair is not EXPLICIT_NONE unless runtime solver records an explicit NONE decision",
        "runtime_future_gt_used": False,
    }


__all__ = [
    "SCHEMA_VERSION",
    "build_assignment_sidecar",
    "schema_document",
    "validate_assignment_sidecar",
]
