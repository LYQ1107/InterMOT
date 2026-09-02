"""Toy tests for the N72R3R1 explicit-NONE effect solver."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.optimize import linear_sum_assignment

from sam3_intermot.association.effect_assignment import solve_effect_assignment


def _states(count: int = 2):
    return [
        {"association_state_id": index + 1, "public_id": 1007 + index}
        for index in range(count)
    ]


def _rows(count: int):
    return [{"candidate_index": index, "candidate_uid": f"toy:candidate:{index}"} for index in range(count)]


def test_exact_none_changes_global_solution_vs_posthoc_threshold() -> None:
    scores = np.asarray([[0.9, 0.0], [0.8, -1.0]], dtype=float)
    rows = _rows(2)
    old_rows, old_cols = linear_sum_assignment(-scores)
    old = {
        int(col): (1007 + int(row) if scores[row, col] >= 0 else None)
        for row, col in zip(old_rows, old_cols)
    }
    artifact = solve_effect_assignment(
        candidate_rows=rows,
        persistent_states=_states(),
        fused_state_candidate_scores=scores,
        source_run_id="toy",
        session_id="s",
    )
    new = {item["candidate_index"]: item["public_id"] for item in artifact["assignment_rows"]}
    assert old == {0: 1008, 1: 1007}
    assert new == {0: 1007, 1: None}
    assert artifact["explicit_none_count"] == 1


def test_all_negative_candidates_choose_none() -> None:
    artifact = solve_effect_assignment(
        candidate_rows=_rows(2),
        persistent_states=_states(),
        fused_state_candidate_scores=[[-0.1, -0.2], [-0.3, -0.4]],
        source_run_id="toy",
        session_id="s",
    )
    assert all(item["status"] == "EXPLICIT_NONE" for item in artifact["assignment_rows"])
    assert all(item["status"] == "NO_CANDIDATE_ASSIGNED" for item in artifact["public_assignments"])


def test_one_public_identity_without_candidate_is_distinct_from_candidate_none() -> None:
    artifact = solve_effect_assignment(
        candidate_rows=_rows(1),
        persistent_states=_states(2),
        fused_state_candidate_scores=[[0.8], [-0.2]],
        source_run_id="toy",
        session_id="s",
    )
    assert artifact["assignment_rows"][0]["status"] == "ASSIGNED_TO_PUBLIC_ID"
    statuses = {item["public_id"]: item["status"] for item in artifact["public_assignments"]}
    assert statuses[1008] == "NO_CANDIDATE_ASSIGNED"


def test_duplicate_public_axis_fails() -> None:
    with pytest.raises(ValueError, match="public authority axis"):
        solve_effect_assignment(
            candidate_rows=_rows(2),
            persistent_states=[
                {"association_state_id": 1, "public_id": 1007},
                {"association_state_id": 2, "public_id": 1007},
            ],
            fused_state_candidate_scores=[[1.0, 0.0], [0.0, 1.0]],
            source_run_id="toy",
            session_id="s",
        )


def test_state_and_public_axis_length_mismatch_fails() -> None:
    with pytest.raises(ValueError, match="score shape"):
        solve_effect_assignment(
            candidate_rows=_rows(2),
            persistent_states=_states(1),
            fused_state_candidate_scores=[[1.0]],
            source_run_id="toy",
            session_id="s",
        )


def test_duplicate_candidate_uid_fails() -> None:
    with pytest.raises(ValueError, match="candidate_uid collision"):
        solve_effect_assignment(
            candidate_rows=[{"candidate_uid": "same"}, {"candidate_uid": "same"}],
            persistent_states=_states(),
            fused_state_candidate_scores=[[1.0, 0.0], [0.0, 1.0]],
            source_run_id="toy",
            session_id="s",
        )
