import numpy as np
import pytest
from types import SimpleNamespace
from sam3_intermot.association.global_assignment_adapter import solve_global, validate_global, public_map


def test_complete_global_reallocation_and_none():
    states = [SimpleNamespace(pid=1, public_id=1001), SimpleNamespace(pid=2, public_id=1002)]
    rows = [{"candidate_uid": "a"}, {"candidate_uid": "b"}, {"candidate_uid": "c"}]
    result = solve_global(rows, np.array([[2, 8], [9, 1], [-3, -4]]), states, frame=3)
    assert public_map(result) == {1001: "b", 1002: "a"}
    assert result["assignment_rows"][2]["status"] == "EXPLICIT_NONE"
    assert result["explicit_none_count"] == 1


def test_candidate_collision_hard_fails():
    with pytest.raises(ValueError, match="collision"):
        solve_global([{"candidate_uid": "a"}, {"candidate_uid": "a"}], np.zeros((2, 0)), [], frame=0)


def test_corrupted_result_axis_hard_fails():
    rows = [{"candidate_uid": "a"}, {"candidate_uid": "b"}]
    result = solve_global(rows, np.zeros((2, 0)), [], frame=0)
    result["assignment_rows"][1]["candidate_uid"] = "a"
    with pytest.raises(RuntimeError):
        validate_global(result, rows)
