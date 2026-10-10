"""Pure score-margin controls; no relative-NONE bonus or MOT authority.

REAL_CANDIDATE excludes NONE from ranking/margin but still retains its
original mass in the calibrated score denominator. UID_PLUS_NONE includes
NONE in both ranking and margin. Neither changes any candidate's score.
"""
from contextlib import contextmanager
import numpy as np
from .intervention_features import feature_vector
from .open_set_verifier import choose as ordinary_choose, simple_predictions
from . import fresh_open_set_learning as evaluation

CONTROLS = ("PURE_REAL_CANDIDATE_MARGIN", "PURE_UID_PLUS_NONE_MARGIN")
RUNTIME_AXIS_KEYS = {"candidate_uid", "features", "feature_vector", "action", "action_feasible", "action_not_executed", "current_owner_public"}


def runtime_predictions(axis, temperature):
    if not axis or axis[-1]["candidate_uid"] is not None or any(r["candidate_uid"] is None for r in axis[:-1]):
        raise ValueError("Complete unique current candidates plus one last NONE required")
    if len({r["candidate_uid"] for r in axis[:-1]}) != len(axis) - 1:
        raise ValueError("Candidate axis contains duplicate current UIDs")
    if not np.isfinite(temperature) or temperature <= 0:
        raise ValueError("Frozen calibration temperature must be positive")
    for row in axis:
        if set(row) - RUNTIME_AXIS_KEYS:
            raise ValueError("GT/taxonomy/outcome fields cannot enter pure margin runtime")
        if feature_vector(row["features"]).tolist() != row["feature_vector"]:
            raise ValueError("Exact current-runtime feature vector required")
        p = row["features"]["proposed_probability"]
        if not np.isfinite(p) or not 0. <= p <= 1.:
            raise ValueError("Finite current probabilities in [0,1] required")
    return simple_predictions(axis, "E1_CALIBRATED", temperature)


def choose_control(axis, predicted, selection, control):
    if control not in CONTROLS:
        raise ValueError("Only separately registered pure margin controls")
    # Runtime score production enforces GT-free axes. This selector reads
    # just current UID/score; offline evaluators may carry separate labels.
    if not axis or len(axis) != len(predicted) or axis[-1]["candidate_uid"] is not None:
        raise ValueError("Complete current prediction axis required")
    if control == "PURE_UID_PLUS_NONE_MARGIN":
        result = ordinary_choose(axis, predicted, {**selection, "simple_family": None})
        return {**result, "pure_margin_control": control, "NONE_score_bonus_added": False}
    if len(axis) == 1:
        return {"candidate_uid": None, "axis_index": 0, "accepted": False, "score": 0., "margin": None,
                "prediction": predicted[0], "reasons": ["NO_REAL_CANDIDATE_ABSTAIN"],
                "pure_margin_control": control, "NONE_score_bonus_added": False}
    scores = [p["correct"] for p in predicted]
    if not all(np.isfinite(s) and 0. <= s <= 1. for s in scores):
        raise ValueError("Finite calibrated current scores required")
    order = sorted(range(len(axis) - 1), key=lambda i: (-scores[i], str(axis[i]["candidate_uid"])))
    index = order[0]
    gap = scores[index] - (scores[order[1]] if len(order) > 1 else 0.)
    reasons = []
    if selection["status"] == "CALIBRATION_ABSTAIN":
        reasons.append("INNER_NO_NONVACUOUS_SAFE_POINT")
    if scores[index] < selection["probability_min"]:
        reasons.append("LOW_CURRENT_CORRECTNESS")
    if gap < selection["margin_min"]:
        reasons.append("AMBIGUOUS_REAL_CANDIDATES")
    if predicted[index]["unknown"] > selection["unknown_max"]:
        reasons.append("UNKNOWN_ABSTAIN")
    return {"candidate_uid": axis[index]["candidate_uid"], "axis_index": index,
            "accepted": not reasons, "score": scores[index], "margin": gap,
            "prediction": predicted[index], "reasons": reasons,
            "pure_margin_control": control, "NONE_score_bonus_added": False}


@contextmanager
def offline_selector(control):
    """Reuse exact existing offline metric/selection code, process-local only."""
    previous = evaluation.choose
    evaluation.choose = lambda axis, predicted, selection: choose_control(axis, predicted, selection, control)
    try:
        yield
    finally:
        evaluation.choose = previous
