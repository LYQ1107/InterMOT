"""Offline exact-frame identity claims; never executed MOT correction counts."""
from collections import Counter
import math
import numpy as np
from sam3_intermot.one_click.intervention_features import feature_vector
from sam3_intermot.one_click.pure_candidate_margin import RUNTIME_AXIS_KEYS


def axis_uids(axis):
    if not axis or axis[-1].get("candidate_uid") is not None:
        raise ValueError("One last explicit NONE required")
    uids = [r["candidate_uid"] for r in axis]
    if any(u is None for u in uids[:-1]) or len(set(uids)) != len(uids):
        raise ValueError("Complete unique current UID axis required")
    return uids


def check_runtime_axis(axis):
    axis_uids(axis)
    for row in axis:
        if set(row) - RUNTIME_AXIS_KEYS:
            raise ValueError("Truth fields cannot enter runtime score production")
        if feature_vector(row["features"]).tolist() != row["feature_vector"]:
            raise ValueError("Changed current feature vector")
        value = row["features"]["proposed_probability"]
        if not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError("Finite original probability required")


def rates(c):
    def ratio(n, d):
        return c.get(n, 0) / c.get(d, 0) if c.get(d, 0) else None
    return {
        "decision_coverage": ratio("accepted_decisions", "groups"),
        "identity_claim_coverage": ratio("identity_claims", "groups"),
        "identity_precision": ratio("TARGET_claims", "identity_claims"),
        "target_recall_given_available": ratio("TARGET_claims", "positive_available"),
        "verified_OTHER_row_FPR": ratio("VERIFIED_OTHER_claims", "verified_OTHER_candidate_rows"),
        "UNKNOWN_per_identity_claim": ratio("UNKNOWN_claims", "identity_claims"),
        "wrong_identity_claims_per_group_NOT_deployed_takeover": ratio("VERIFIED_OTHER_claims", "groups"),
        "false_presence_physically_absent": ratio("false_presence_absent", "physically_absent"),
        "false_presence_no_positive_candidate": ratio("false_presence_no_positive", "no_positive_candidate"),
        "direct_safe_repair_UID_retention_NOT_executed_correction": ratio("direct_retained", "direct_opportunities"),
        "future_safe_target_UID_retention_NOT_executed_correction": ratio("future_retained", "future_opportunities"),
        "decision_risk_including_UNKNOWN_incorrect_NONE": ratio("incorrect_decisions", "accepted_decisions"),
    }


def count_choice(group, uids, choice, direct, future):
    if axis_uids(group["axis"]) != uids:
        raise ValueError("Prediction and offline truth axes differ")
    index = choice["axis_index"]
    if type(index) is not int or not 0 <= index < len(uids):
        raise ValueError("Chosen index must be a real original axis index")
    if uids[index] != choice["candidate_uid"] or type(choice["accepted"]) is not bool:
        raise ValueError("Chosen UID/acceptance is not the sealed choice")
    visible, available = group["target_visible_label"], group["target_candidate_available_label"]
    if type(visible) is not bool or type(available) is not bool or available and not visible:
        raise ValueError("Physical visibility and positive-candidate availability must stay separate")
    outcomes = [row["current_outcome"] for row in group["axis"]]
    if any(o not in {"TARGET", "VERIFIED_OTHER", "UNKNOWN_UNMATCHED"} for o in outcomes[:-1]):
        raise ValueError("Only actual verified TARGET/OTHER or separate UNKNOWN identity truth")
    if available != ("TARGET" in outcomes[:-1]) or outcomes[-1] != ("INCORRECT_NONE" if available else "CORRECT_NONE"):
        raise ValueError("NONE/positive availability contradict the original full axis")
    c = Counter(groups=1, positive_available=int(available), physically_absent=int(not visible),
                no_positive_candidate=int(not available), visible_without_positive=int(visible and not available),
                verified_OTHER_candidate_rows=sum(r["current_outcome"] == "VERIFIED_OTHER" for r in group["axis"]))
    row = group["axis"][index]
    accepted, uid = choice["accepted"], choice["candidate_uid"]
    if accepted:
        c["accepted_decisions"] += 1
        c["incorrect_decisions"] += row["current_outcome"] not in ("TARGET", "CORRECT_NONE")
        if uid is None:
            c["NONE_claims"] += 1
            c["correct_NONE_claims"] += row["current_outcome"] == "CORRECT_NONE"
        else:
            c["identity_claims"] += 1
            c["TARGET_claims"] += row["current_outcome"] == "TARGET"
            c["VERIFIED_OTHER_claims"] += row["current_outcome"] == "VERIFIED_OTHER"
            c["UNKNOWN_claims"] += row["current_outcome"] == "UNKNOWN_UNMATCHED"
            c["false_presence_absent"] += not visible
            c["false_presence_no_positive"] += not available
            owner = row.get("current_owner_public")
            c["claim_owned_by_other_public_ID_NOT_GT_wrong_identity"] += owner is not None and owner != group["target_public_id"]
    key = group["episode_uid"], group["frame"]
    for label, opportunities in (("direct", direct), ("future", future)):
        if key in opportunities:
            if not opportunities[key] or not opportunities[key] <= set(uids[:-1]):
                raise ValueError("Repair opportunities may only use original real current UIDs")
            c[label + "_opportunities"] += 1
            retained = bool(accepted and uid is not None and uid in opportunities[key])
            c[label + "_retained"] += retained
            c[label + "_missed"] += not retained
    return c


def video_interval(values, seed=730104, repetitions=2000):
    values = np.asarray([v for v in values if v is not None], dtype=np.float64)
    if not len(values):
        return {"mean": None, "CI95": None, "video_clusters": 0}
    if not np.isfinite(values).all():
        raise ValueError("Finite per-video ratios required")
    rng = np.random.default_rng(seed)
    draws = values[rng.integers(0, len(values), size=(repetitions, len(values)))].mean(axis=1)
    return {"mean": float(values.mean()), "CI95": np.quantile(draws, [.025, .975]).tolist(),
            "video_clusters": len(values), "seed": seed, "repetitions": repetitions,
            "TRAIN_side_descriptive_not_independent_generalization": True,
            "zero_or_degenerate_interval_NOT_population_safety_proof": bool(np.all(draws == draws[0]))}
