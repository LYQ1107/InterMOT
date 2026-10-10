import numpy as np
import pytest
from sam3_intermot.one_click.intervention_features import FEATURE_NAMES, feature_vector
from sam3_intermot.one_click.pure_candidate_margin import runtime_predictions, choose_control, offline_selector
from sam3_intermot.one_click import fresh_open_set_learning as evaluation


def axis(probabilities):
    result = []
    for i, probability in enumerate(probabilities):
        f = {name: 0. for name in FEATURE_NAMES}
        f["proposed_probability"] = probability
        f["NONE_probability"] = probabilities[-1]
        f["proposal_NONE"] = float(i == len(probabilities) - 1)
        result.append({"candidate_uid": None if i == len(probabilities)-1 else "c" + str(i),
                       "features": f, "feature_vector": feature_vector(f).tolist()})
    return result


POINT = {"status": "UNQUALIFIED_DIAGNOSTIC", "probability_min": .3, "margin_min": .2, "unknown_max": .5}


def test_none_not_in_real_candidate_margin_and_scores_are_identical():
    rows = axis([.6, .1, .7])
    prediction = runtime_predictions(rows, 1.)
    real = choose_control(rows, prediction, POINT, "PURE_REAL_CANDIDATE_MARGIN")
    both = choose_control(rows, prediction, POINT, "PURE_UID_PLUS_NONE_MARGIN")
    assert real["candidate_uid"] == "c0" and real["accepted"]
    assert real["margin"] == pytest.approx(.5/1.4)
    assert both["candidate_uid"] is None and not both["accepted"]
    assert np.allclose([r["correct"] for r in prediction], np.array([.6,.1,.7])/1.4)
    assert not real["NONE_score_bonus_added"] and not both["NONE_score_bonus_added"]


def test_empty_real_axis_and_no_safe_point_cannot_claim_identity():
    rows = axis([1.]);p = runtime_predictions(rows,1.)
    assert not choose_control(rows,p,POINT,"PURE_REAL_CANDIDATE_MARGIN")["accepted"]
    rows = axis([.9,.1]);p=runtime_predictions(rows,1.)
    choice = choose_control(rows,p,{**POINT,"status":"CALIBRATION_ABSTAIN"},"PURE_REAL_CANDIDATE_MARGIN")
    assert not choice["accepted"]


def test_gt_fields_and_broken_runtime_vectors_rejected():
    rows=axis([.8,.2]);rows[0]["current_outcome"]="TARGET"
    with pytest.raises(ValueError,match="GT"):
        runtime_predictions(rows,1.)
    rows=axis([.8,.2]);rows[0]["feature_vector"][0]+=.1
    with pytest.raises(ValueError,match="feature vector"):
        runtime_predictions(rows,1.)
    with pytest.raises(ValueError,match="temperature"):
        runtime_predictions(axis([.8,.2]),0.)


def test_offline_selector_restores_process_local_original_on_failure():
    previous=evaluation.choose
    with pytest.raises(RuntimeError):
        with offline_selector("PURE_REAL_CANDIDATE_MARGIN"):
            assert evaluation.choose is not previous
            raise RuntimeError("deliberate offline fixture failure")
    assert evaluation.choose is previous
