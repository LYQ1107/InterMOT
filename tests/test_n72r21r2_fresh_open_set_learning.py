import pytest
from sam3_intermot.one_click.fresh_open_set_learning import flatten_groups, assess_claims, select_point
from sam3_intermot.one_click.intervention_features import FEATURE_NAMES, feature_vector


def group(sequence="v", outcome="TARGET", available=True, visible=True):
    features = {k: 0. for k in FEATURE_NAMES}
    axis = []
    for uid, label in [("p", outcome), (None, "INCORRECT_NONE" if available else "CORRECT_NONE")]:
        axis.append(dict(candidate_uid=uid, current_outcome=label, class_=0,
                         features=features, feature_vector=feature_vector(features).tolist(), verified_identity_negative=label == "VERIFIED_OTHER"))
        axis[-1]["class"] = 0 if label in ("TARGET", "CORRECT_NONE") else 2 if label == "UNKNOWN_UNMATCHED" else 1
    return dict(sequence=sequence, episode_uid=sequence + "click", frame=1, axis=axis,
                target_candidate_available_label=available, target_visible_label=visible)


def test_UNKNOWN_class_not_hard_negative_and_weighting_equalizes_videos():
    groups = [group("a"), group("a"), group("b", "UNKNOWN_UNMATCHED")]
    _, classes, _, weights, hard, spans = flatten_groups(groups)
    assert sum(weights[:4]) == pytest.approx(.5) and sum(weights[4:]) == pytest.approx(.5)
    assert classes[4] == 2 and hard[4] == 1.
    groups[-1]["axis"][0]["verified_identity_negative"] = True
    with pytest.raises(ValueError): flatten_groups(groups)


def test_always_NONE_is_not_a_nonvacuous_safe_calibration():
    groups = [group("a", "VERIFIED_OTHER", False), group("b", "VERIFIED_OTHER", False)]
    predictions = [[dict(correct=.01, incorrect=.99, unknown=0.), dict(correct=.99, incorrect=.01, unknown=0.)]] * 2
    protocol = dict(cutoffs=[.5], margins=[0.], unknown_max=.02, min_identity_claims=1, min_identity_claim_videos=1, risk_max=.02)
    selection, points = select_point(groups, predictions, protocol)
    assert selection["status"] == "CALIBRATION_ABSTAIN" and not points[0]["nonvacuous"]


def test_present_without_candidate_not_physically_absent_and_UNKNOWN_risk_retained():
    groups = [group("a", "UNKNOWN_UNMATCHED", False, True), group("b", "VERIFIED_OTHER", False, False)]
    predictions = [[dict(correct=.99, incorrect=.01, unknown=0.), dict(correct=.01, incorrect=.99, unknown=0.)]] * 2
    result = assess_claims(groups, predictions, dict(status="DIAGNOSTIC", probability_min=.5, margin_min=0., unknown_max=.02))
    assert result["counts"]["false_presence_no_positive_candidate"] == 2
    assert result["counts"]["false_presence_physically_absent"] == 1
    assert result["counts"]["UNKNOWN_claims"] == 1
    assert result["micro"]["decision_risk_including_UNKNOWN_and_incorrect_NONE"] == 1.
