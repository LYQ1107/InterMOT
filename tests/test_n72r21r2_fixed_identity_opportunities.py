from copy import deepcopy
import pytest
from sam3_intermot.evaluation import fixed_identity_opportunities as audit
from sam3_intermot.one_click.intervention_features import FEATURE_NAMES, feature_vector


def group(visible=True, available=True):
    return {"episode_uid": "click", "frame": 10, "target_public_id": 1,
            "target_visible_label": visible, "target_candidate_available_label": available,
            "axis": [{"candidate_uid": "target", "current_outcome": "TARGET" if available else "UNKNOWN_UNMATCHED", "current_owner_public": 2},
                     {"candidate_uid": "other", "current_outcome": "VERIFIED_OTHER"},
                     {"candidate_uid": "unknown", "current_outcome": "UNKNOWN_UNMATCHED"},
                     {"candidate_uid": None, "current_outcome": "INCORRECT_NONE" if available else "CORRECT_NONE"}]}


def choice(index, accepted=True):
    return {"axis_index": index, "candidate_uid": ["target", "other", "unknown", None][index], "accepted": accepted}


def count(g, c, direct=None, future=None):
    return audit.count_choice(g, audit.axis_uids(g["axis"]), c, direct or {}, future or {})


def test_unknown_claim_not_verified_other_negative_or_two_percent_safety():
    c = count(group(), choice(2))
    r = audit.rates(c)
    assert c["UNKNOWN_claims"] == 1 and c["VERIFIED_OTHER_claims"] == 0
    assert r["UNKNOWN_per_identity_claim"] == 1
    assert r["verified_OTHER_row_FPR"] == 0 and r["decision_risk_including_UNKNOWN_incorrect_NONE"] == 1


def test_visible_missing_detection_is_not_physical_absence():
    c = count(group(visible=True, available=False), choice(2))
    r = audit.rates(c)
    assert c["visible_without_positive"] == 1 and c["physically_absent"] == 0
    assert r["false_presence_physically_absent"] is None
    assert r["false_presence_no_positive_candidate"] == 1


def test_rejection_misses_safe_direct_and_future_uid_but_not_executed_events():
    opportunities = {("click", 10): {"target"}}
    c = count(group(), choice(0, False), opportunities, opportunities)
    assert c["direct_missed"] == c["future_missed"] == 1
    assert c["identity_claims"] == 0 and audit.rates(c)["identity_precision"] is None
    assert audit.rates(c)["direct_safe_repair_UID_retention_NOT_executed_correction"] == 0


def test_other_public_owner_is_not_verified_other_identity():
    c = count(group(), choice(0), {("click", 10): {"target"}})
    assert c["claim_owned_by_other_public_ID_NOT_GT_wrong_identity"] == 1
    assert c["TARGET_claims"] == c["direct_retained"] == 1 and c["VERIFIED_OTHER_claims"] == 0


@pytest.mark.parametrize("change", [{"axis_index": True}, {"axis_index": 99}, {"candidate_uid": "wrong"}, {"accepted": 1}])
def test_sealed_choice_uid_axis_and_boolean_contract(change):
    with pytest.raises(ValueError): count(group(), {**choice(0), **change})


@pytest.mark.parametrize("uids", [["target", None, "other", None], ["target", "target", None], ["target"]])
def test_duplicate_or_missing_none_axis_never_imputed(uids):
    with pytest.raises(ValueError): audit.axis_uids([{"candidate_uid": u} for u in uids])


def test_noncurrent_oracle_opportunity_uid_is_rejected():
    with pytest.raises(ValueError): count(group(), choice(0), {("click", 10): {"fake_gt_candidate"}})


def test_runtime_axis_forbids_truth_and_uses_exact_feature_vector():
    features = {name: 0. for name in FEATURE_NAMES}
    row = {"candidate_uid": None, "features": features, "feature_vector": feature_vector(features).tolist()}
    audit.check_runtime_axis([row])
    with pytest.raises(ValueError): audit.check_runtime_axis([{**row, "current_outcome": "CORRECT_NONE"}])
    changed = deepcopy(row); changed["feature_vector"][0] = 1
    with pytest.raises(ValueError): audit.check_runtime_axis([changed])


def test_zero_or_missing_cluster_interval_never_population_safety_guarantee():
    assert audit.video_interval([])["CI95"] is None
    r = audit.video_interval([0., 0., None])
    assert r["video_clusters"] == 2 and r["CI95"] == [0., 0.]
    assert r["zero_or_degenerate_interval_NOT_population_safety_proof"]
