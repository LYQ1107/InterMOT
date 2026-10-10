from copy import deepcopy
from scripts.n72r21r2_joint_state_curriculum import select_cases, SOURCE_CASES


def seal(frame, changed):
    return {"frame": frame, "artifacts": [{"branch": name, "path": "trace", "sha256": "frozen",
             "current_feasibility": {"feasible": True, "assignment_changed": changed},
             "future_benefit": False, "offline_GT": None} for name in SOURCE_CASES]}


def test_chronological_selection_uses_current_change_not_future_reward():
    first, later = seal(1, False), seal(10, True)
    result = select_cases("click", [later, first], 300)
    assert result["cases"][0]["source_frame"] == 1
    assert next(r for r in result["cases"] if r["source_case"] == "DELAYED_CHALLENGER")["source_frame"] == 1
    assert next(r for r in result["cases"] if r["source_case"] == "RAW_IDENTITY_TOP")["source_frame"] == 10
    altered = deepcopy([later, first])
    for item in altered:
        for arm in item["artifacts"]:
            arm.update(future_benefit=True, offline_GT=999)
    assert result == select_cases("click", altered, 300)


def test_no_future_tail_or_applicable_branch_is_not_replaced():
    result = select_cases("click", [seal(160, True)], 300)
    assert all("source_frame" not in row for row in result["cases"])
    assert select_cases("failed", [], 300)["episode_uid"] == "failed"
