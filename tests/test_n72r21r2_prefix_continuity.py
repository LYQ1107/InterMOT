from copy import deepcopy
import pytest
from sam3_intermot.evaluation.prefix_continuity import (
    structural_late_selection, assemble_prefix, paired_sign_counts, sign)


def events():
    return [{"episode_uid": "click"+str(i), "frame": i*100, "frames": 401,
             "initialization_failure": i == 0} for i in range(3)]


def plans():
    return [{"episode_uid": "click0", "frames": []}, {"episode_uid": "click1", "frames": [101, 200, 300, 301]},
            {"episode_uid": "click2", "frames": [201, 299]}]


def test_structural_late_selection_no_labels_or_best_effect_choice():
    result = structural_late_selection(events(), plans())
    assert result["selection"] == {"episode_uid": "click1", "frame": 300}
    assert result["failed_initialization_slots_retained"] == 1


def test_all_failed_video_not_replaced_or_successful_zero():
    e = events()
    for r in e: r["initialization_failure"] = True
    result = structural_late_selection(e, plans())
    assert result["selection"] is None and result["status"] == "NOT_RUN_ALL_INITIALIZATIONS_FAILED"
    assert result["failed_initialization_slots_retained"] == 3


def test_missing_plan_or_duplicate_click_not_allowed():
    with pytest.raises(ValueError): structural_late_selection(events(), plans()[:2])
    with pytest.raises(ValueError): structural_late_selection(events()+events()[:1], plans())


def test_earliest_valid_without_full_window_does_not_trigger_click_lottery():
    p = plans(); p[1]["frames"] = [301]
    r = structural_late_selection(events(), p)
    assert r["selection"] is None and "NO_COMPLETE" in r["status"]


def source():
    return [{"frame": i, "state_before": str(i), "target_public_id": 5,
             "outputs": [{"public_id": 5, "candidate_uid": str(i)+"a"},
                         {"public_id": 8, "candidate_uid": str(i)+"b"}]} for i in range(5)]


def test_prefix_keeps_full_global_branch_not_target_only_splice_and_no_write():
    src = source(); arm = deepcopy(src[3:]); before = deepcopy(src)
    arm[0]["outputs"] = [{"public_id": 5, "candidate_uid": "3b"}, {"public_id": 8, "candidate_uid": "3a"}]
    r = assemble_prefix(src, arm, frame=3, length=2)
    assert r[:3] == src[:3] and r[3:] == arm
    assert r[3]["outputs"][1]["candidate_uid"] == "3a"
    assert src == before and [x["frame"] for x in r] == [0,1,2,3,4]


@pytest.mark.parametrize("mode", ["missing_prefix", "truncated_arm", "wrong_prestate", "relabel_target"])
def test_noncausal_or_padded_prefix_joins_rejected(mode):
    src = source(); arm = deepcopy(src[3:])
    if mode == "missing_prefix": src = src[1:]
    elif mode == "truncated_arm": arm = arm[:1]
    elif mode == "wrong_prestate": arm[0]["state_before"] = "future_arm_state"
    else: arm[0]["target_public_id"] = 999
    with pytest.raises(ValueError): assemble_prefix(src, arm, frame=3, length=2)


def test_sign_changes_are_descriptive_not_scientific_pass_or_policy_threshold():
    r = paired_sign_counts({"HOTA": 0., "AssA": -.01, "IDF1": 0.}, {"HOTA": .02, "AssA": .02, "IDF1": .01})
    assert r["HOTA_sign_changes"] == r["AssA_sign_changes"] == r["IDF1_sign_changes"] == 1
    assert r["window_nonpositive_but_prefix_HOTA_and_AssA_positive"] == 1
    assert not any("PASS" in k or "independent" in k for k in r)
    assert sign(1e-10) == "ZERO_WITHIN_API_CLI_TOLERANCE"


def test_nonfinite_metric_deltas_rejected():
    with pytest.raises(ValueError): sign(float("nan"))
