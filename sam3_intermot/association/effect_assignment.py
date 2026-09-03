"""The one explicit-NONE assignment entry point for interaction effects.

The production public-assignment implementation owns the solver.  This module
only adapts the replayer's state-by-candidate score orientation and makes the
two identity axes explicit; it intentionally does not call a second matching
implementation.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
import hashlib
import json
from typing import Any

import numpy as np

from .public_assignment import solve_exact_public_assignment, validate_exact_public_assignment


SCHEMA_VERSION = "N72R3R1_EFFECT_ASSIGNMENT_WRAPPER_V1"


def _matrix_sha256(values: Any) -> str:
    """Hash a matrix in its supplied orientation, without transposing it."""
    return hashlib.sha256(
        json.dumps(
            np.asarray(values, dtype=np.float64).tolist(),
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _field(value: Any, name: str) -> Any:
    if isinstance(value, Mapping):
        return value.get(name)
    return getattr(value, name, None)


def solve_effect_assignment(
    *,
    candidate_rows: Iterable[Mapping[str, Any]],
    persistent_states: Sequence[Any],
    fused_state_candidate_scores: Sequence[Sequence[float]],
    source_run_id: str,
    session_id: str,
    none_score: float = 0.0,
) -> dict[str, Any]:
    """Solve state×candidate scores with explicit per-candidate NONE.

    ``persistent_states`` must expose both ``association_state_id`` and
    ``public_id``.  The wrapper refuses to synthesize either axis from a row
    index or from the other axis.  ``public_assignment.solve_exact...`` then
    receives candidate×state scores and performs the only global solve.
    """

    rows = [dict(row) for row in candidate_rows]
    state_values = list(persistent_states)
    scores = np.asarray(fused_state_candidate_scores, dtype=np.float64)
    if scores.ndim != 2:
        raise ValueError(f"state×candidate scores must be rank-2, got {scores.shape}")
    expected_shape = (len(state_values), len(rows))
    if tuple(scores.shape) != expected_shape:
        raise ValueError(f"state×candidate score shape {scores.shape} != {expected_shape}")
    state_axis: list[int] = []
    public_axis: list[int] = []
    for index, state in enumerate(state_values):
        state_id = _field(state, "association_state_id")
        public_id = _field(state, "public_id")
        if state_id is None or public_id is None:
            raise ValueError(f"persistent state {index} lacks explicit state/public authority axes")
        state_axis.append(int(state_id))
        public_axis.append(int(public_id))
    if len(state_axis) != len(set(state_axis)):
        raise ValueError("association state axis contains duplicate IDs")
    if len(public_axis) != len(set(public_axis)):
        raise ValueError("public authority axis contains duplicate IDs")
    if not np.isfinite(scores).all():
        raise ValueError("state×candidate score matrix contains non-finite values")

    # The frozen solver consumes candidate rows × state columns.
    artifact = solve_exact_public_assignment(
        rows,
        scores.T,
        state_axis,
        public_axis,
        none_score=float(none_score),
        source_run_id=str(source_run_id),
        session_id=str(session_id),
        runtime_future_gt_used=False,
    )
    errors = validate_exact_public_assignment(artifact)
    if errors:
        raise RuntimeError(f"exact public assignment validator failed: {errors}")
    state_candidate_hash = _matrix_sha256(scores)
    candidate_state_hash = _matrix_sha256(scores.T)
    if candidate_state_hash != artifact["fused_score_matrix_sha256"]:
        raise RuntimeError(
            "solver candidate×state provenance hash disagrees with the submitted matrix"
        )
    artifact.update(
        {
            "schema_version": SCHEMA_VERSION,
            "state_candidate_score_matrix_shape": [int(scores.shape[0]), int(scores.shape[1])],
            "state_candidate_score_matrix_sha256": state_candidate_hash,
            "candidate_state_score_matrix_sha256": candidate_state_hash,
            "solver_input_orientation": "state_x_candidate_transposed_to_candidate_x_state",
            "persistent_state_count": len(state_values),
            "runtime_future_gt_used": False,
        }
    )
    return artifact


__all__ = ["SCHEMA_VERSION", "solve_effect_assignment"]
