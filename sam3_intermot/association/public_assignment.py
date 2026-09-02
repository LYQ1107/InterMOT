"""Exact public-ID assignment for the N72R2 runtime bridge.

The existing StateManager assignment is deliberately kept as a solver-local
candidate-by-state audit.  This module takes the same frozen fused score
matrix and an already-proven state-to-public authority axis, then solves one
global candidate-by-public-ID assignment with explicit per-candidate ``NONE``
columns.  It never infers authority from numeric state IDs and never reads GT.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from scipy.optimize import linear_sum_assignment


SCHEMA_VERSION = "N72R2_EXACT_PUBLIC_ASSIGNMENT_V1"


def _json_digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    ).hexdigest()


def _finite_matrix(values: Any) -> np.ndarray:
    matrix = np.asarray(values, dtype=np.float64)
    if matrix.ndim != 2:
        raise ValueError(f"fused score matrix must be rank 2, got {matrix.shape}")
    if not np.isfinite(matrix).all():
        raise ValueError("fused score matrix contains non-finite values")
    return matrix


def solve_exact_public_assignment(
    candidate_rows: Iterable[Mapping[str, Any]],
    fused_scores: Sequence[Sequence[float]],
    association_state_axis: Sequence[int],
    public_id_axis: Sequence[int],
    *,
    none_score: float = 0.0,
    source_run_id: str | None = None,
    session_id: str | None = None,
    runtime_future_gt_used: bool = False,
) -> dict[str, Any]:
    """Solve a public-ID assignment with one explicit NONE slot per row.

    ``fused_scores`` is candidate-row × association-state-column and must be
    the same matrix that produced the state-side audit.  Public IDs are an
    independently validated axis; equal numeric values are not accepted as a
    substitute for a binding.  One NONE slot per candidate lets multiple
    candidates remain unassigned while keeping the solver globally
    one-to-one over public identities.
    """

    rows = [dict(row) for row in candidate_rows]
    matrix = _finite_matrix(fused_scores)
    state_axis = [int(value) for value in association_state_axis]
    public_axis = [int(value) for value in public_id_axis]
    n, m = matrix.shape
    if n != len(rows) or m != len(state_axis) or len(state_axis) != len(public_axis):
        raise ValueError(
            "public assignment axis mismatch: "
            f"matrix={matrix.shape}, rows={len(rows)}, states={len(state_axis)}, publics={len(public_axis)}"
        )
    if len(set(state_axis)) != len(state_axis):
        raise ValueError("association state axis contains duplicate IDs")
    if len(set(public_axis)) != len(public_axis):
        raise ValueError("public authority axis contains duplicate IDs")
    if not math.isfinite(float(none_score)):
        raise ValueError("none_score must be finite")
    candidate_uids = [row.get("candidate_uid") for row in rows]
    if any(uid in (None, "") for uid in candidate_uids):
        raise ValueError("every candidate row needs a candidate_uid")
    uid_counts = Counter(str(uid) for uid in candidate_uids)
    collisions = {uid: count for uid, count in uid_counts.items() if count > 1}
    if collisions:
        raise ValueError(f"candidate_uid collision: {collisions}")
    if runtime_future_gt_used:
        raise ValueError("runtime_future_gt_used must be false")

    # A candidate can choose any public identity at most once globally, or one
    # of n distinct NONE slots.  The public columns reuse the exact state-side
    # score columns; no scalar reweighting or candidate reordering occurs.
    full_scores = np.concatenate(
        [matrix, np.full((n, n), float(none_score), dtype=np.float64)], axis=1
    )
    assigned_rows, assigned_cols = linear_sum_assignment(-full_scores)
    row_to_col = np.full(n, -1, dtype=np.int64)
    for row_index, column_index in zip(assigned_rows.tolist(), assigned_cols.tolist()):
        row_to_col[int(row_index)] = int(column_index)
    if np.any(row_to_col < 0):
        raise RuntimeError("rectangular public assignment did not assign every candidate row")

    assignments: list[dict[str, Any]] = []
    public_to_candidate: dict[int, str] = {}
    public_status: dict[int, str] = {}
    for row_index, column_index in enumerate(row_to_col.tolist()):
        is_none = column_index >= m
        public_id = None if is_none else int(public_axis[column_index])
        score = float(full_scores[row_index, column_index])
        item = {
            "candidate_index": int(rows[row_index].get("candidate_index", row_index)),
            "candidate_uid": str(candidate_uids[row_index]),
            "association_state_id": None if is_none else int(state_axis[column_index]),
            "public_id": public_id,
            "public_column_index": None if is_none else int(column_index),
            "none_column_index": int(column_index - m) if is_none else None,
            "score": score,
            "status": "EXPLICIT_NONE" if is_none else "ASSIGNED_TO_PUBLIC_ID",
            "source_run_id": source_run_id,
            "session_id": session_id,
        }
        assignments.append(item)
        if public_id is None:
            continue
        if public_id in public_to_candidate:
            raise RuntimeError(f"solver produced duplicate public assignment: {public_id}")
        public_to_candidate[public_id] = str(candidate_uids[row_index])
        public_status[public_id] = "ASSIGNED_CANDIDATE"

    # ``EXPLICIT_NONE`` is a candidate-row decision (the row selected its own
    # NONE column).  A public identity with no candidate assigned is a
    # different audit state and must not be mislabeled as a candidate NONE.
    for public_id in public_axis:
        public_status.setdefault(int(public_id), "NO_CANDIDATE_ASSIGNED")
    return {
        "schema_version": SCHEMA_VERSION,
        "source_run_id": source_run_id,
        "session_id": session_id,
        "candidate_axis": [str(uid) for uid in candidate_uids],
        "association_state_axis": state_axis,
        "public_id_axis": public_axis,
        "public_authority_is_explicit": True,
        "association_state_id_is_public_id": False,
        "fused_score_matrix_shape": [int(n), int(m)],
        "fused_score_matrix_sha256": _json_digest(matrix.tolist()),
        "full_solver_matrix_shape": [int(n), int(m + n)],
        "full_solver_matrix_sha256": _json_digest(full_scores.tolist()),
        "none_score": float(none_score),
        "none_column_count": int(n),
        "solver": "scipy.optimize.linear_sum_assignment_maximize_candidate_x_public_plus_none",
        "assignment_rows": assignments,
        "public_assignments": [
            {
                "public_id": int(public_id),
                "candidate_uid": public_to_candidate.get(int(public_id)),
                "status": public_status[int(public_id)],
                "none_score": float(none_score),
            }
            for public_id in public_axis
        ],
        "assigned_public_count": len(public_to_candidate),
        "explicit_none_count": sum(1 for item in assignments if item["public_id"] is None),
        "runtime_future_gt_used": False,
    }


def validate_exact_public_assignment(artifact: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    if artifact.get("schema_version") != SCHEMA_VERSION:
        errors.append("schema_version_mismatch")
    if artifact.get("runtime_future_gt_used") is not False:
        errors.append("runtime_future_gt_used_not_false")
    if artifact.get("association_state_id_is_public_id") is not False:
        errors.append("association_state_axis_mislabelled_public")
    public_axis = artifact.get("public_id_axis", [])
    if len(public_axis) != len(set(public_axis)):
        errors.append("public_id_collision")
    rows = artifact.get("assignment_rows", [])
    public_values = [row.get("public_id") for row in rows if row.get("public_id") is not None]
    if len(public_values) != len(set(public_values)):
        errors.append("assigned_public_id_collision")
    for row in rows:
        if row.get("status") not in {"EXPLICIT_NONE", "ASSIGNED_TO_PUBLIC_ID"}:
            errors.append("assignment_status_invalid")
    return sorted(set(errors))


__all__ = [
    "SCHEMA_VERSION",
    "solve_exact_public_assignment",
    "validate_exact_public_assignment",
]
