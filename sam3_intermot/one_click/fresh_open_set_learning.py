"""R2 current-axis identity claims, not global intervention safety.

Truth is consumed by the offline evaluator/loss only. UNKNOWN is separately
predicted; verified-other labels alone define identity hard negatives.
"""
from collections import Counter
import numpy as np
from .intervention_features import FEATURE_NAMES, feature_vector
from .open_set_verifier import choose

SIMPLE_CONTROLS = ("RAW_SCORE", "CALIBRATED_SCORE", "CANDIDATE_MARGIN", "BASE_AGREEMENT", "TEMPORAL_CONFIRMATION")
HEAD_FAMILIES = ("SCALAR", "LOGISTIC", "MLP", "MLP_HARD_NEGATIVE")


def flatten_groups(groups):
    if not groups:
        raise ValueError("Actual complete current-axis groups required")
    video_clicks = {}
    click_frames = Counter()
    for group in groups:
        video_clicks.setdefault(group["sequence"], set()).add(group["episode_uid"])
        click_frames[group["episode_uid"]] += 1
    x, classes, available, weights, hard, spans = [], [], [], [], [], []
    for group in groups:
        start = len(x)
        axis = group["axis"]
        if not axis or axis[-1]["candidate_uid"] is not None or any(r["candidate_uid"] is None for r in axis[:-1]):
            raise ValueError("Every actual current candidate and a unique last explicit NONE required")
        for row in axis:
            vector = feature_vector(row["features"]).tolist()
            if vector != row["feature_vector"] or len(vector) != len(FEATURE_NAMES):
                raise ValueError("Only the registered finite current runtime features may be fitted")
            if row["current_outcome"] == "UNKNOWN_UNMATCHED":
                if row["class"] != 2 or row["verified_identity_negative"]:
                    raise ValueError("UNKNOWN cannot become a verified-other identity negative")
            if row["verified_identity_negative"] != (row["current_outcome"] == "VERIFIED_OTHER"):
                raise ValueError("Only verified OTHER is an identity hard negative")
            x.append(vector)
            classes.append(row["class"])
            available.append(float(group["target_candidate_available_label"]))
            weights.append(1 / len(video_clicks) / len(video_clicks[group["sequence"]]) / click_frames[group["episode_uid"]] / len(axis))
            hard.append(5. if row["verified_identity_negative"] and row["features"]["anchor_cosine"] >= .7 else 1.)
        spans.append((start, len(x)))
    weights = np.asarray(weights, np.float32)
    weights /= weights.sum()
    return (np.asarray(x, np.float32), np.asarray(classes, np.int64), np.asarray(available, np.float32),
            weights, np.asarray(hard, np.float32), spans)


def assess_claims(groups, predicted, selection):
    """Fixed sampled frame groups; never impute uncollected original frames."""
    counts, accepted_videos, per_video = Counter(), set(), {}
    choices = []
    for group, pred in zip(groups, predicted, strict=True):
        choice = choose(group["axis"], pred, selection)
        row = group["axis"][choice["axis_index"]]
        c = per_video.setdefault(group["sequence"], Counter())
        c["groups"] += 1
        c["available_groups"] += bool(group["target_candidate_available_label"])
        c["physically_absent_groups"] += not group["target_visible_label"]
        c["no_valid_positive_candidate_groups"] += not group["target_candidate_available_label"]
        c["verified_OTHER_candidate_rows"] += sum(a["current_outcome"] == "VERIFIED_OTHER" for a in group["axis"])
        if choice["accepted"]:
            c["accepted_decision_groups"] += 1
            c["incorrect_or_UNKNOWN_decisions"] += row["class"] != 0
            if row["candidate_uid"] is None:
                c["accepted_NONE_groups"] += 1
                c["correct_NONE_groups"] += row["current_outcome"] == "CORRECT_NONE"
            else:
                c["accepted_identity_claims"] += 1
                c["correct_TARGET_claims"] += row["current_outcome"] == "TARGET"
                c["wrong_VERIFIED_OTHER_claims"] += row["current_outcome"] == "VERIFIED_OTHER"
                c["UNKNOWN_claims"] += row["current_outcome"] == "UNKNOWN_UNMATCHED"
                c["false_presence_physically_absent"] += not group["target_visible_label"]
                c["false_presence_no_positive_candidate"] += not group["target_candidate_available_label"]
                accepted_videos.add(group["sequence"])
        choices.append({"sequence": group["sequence"], "episode_uid": group["episode_uid"], "frame": group["frame"],
                        "choice": choice, "offline_outcome": row["current_outcome"]})
    counts.update({"groups": 0})
    for c in per_video.values():
        counts.update(c)
    def rates(c):
        ratio = lambda key, den: c[key] / c[den] if c[den] else None
        return {"coverage": ratio("accepted_decision_groups", "groups"),
                "identity_precision": ratio("correct_TARGET_claims", "accepted_identity_claims"),
                "target_recall_given_available": ratio("correct_TARGET_claims", "available_groups"),
                "wrong_takeover_per_group": ratio("wrong_VERIFIED_OTHER_claims", "groups"),
                "verified_OTHER_row_FPR": ratio("wrong_VERIFIED_OTHER_claims", "verified_OTHER_candidate_rows"),
                "UNKNOWN_per_identity_claim": ratio("UNKNOWN_claims", "accepted_identity_claims"),
                "false_presence_rate_physically_absent": ratio("false_presence_physically_absent", "physically_absent_groups"),
                "false_presence_rate_no_positive_candidate": ratio("false_presence_no_positive_candidate", "no_valid_positive_candidate_groups"),
                "decision_risk_including_UNKNOWN_and_incorrect_NONE": ratio("incorrect_or_UNKNOWN_decisions", "accepted_decision_groups")}
    micro = rates(counts)
    macro = {}
    per_sequence = {s: {"counts": dict(c), **rates(c)} for s, c in per_video.items()}
    for key in micro:
        values = [r[key] for r in per_sequence.values() if r[key] is not None]
        macro[key] = float(np.mean(values)) if values else None
    return {"counts": dict(counts), "micro": micro, "sequence_macro": macro, "per_sequence": per_sequence,
            "videos_with_actual_identity_claims": sorted(accepted_videos), "choices": choices,
            "sampled_groups_not_dense_full_video_counts": True, "current_identity_claim_not_future_authority_safety": True}


def select_point(groups, predicted, protocol, simple_family=None):
    points = []
    for cutoff in protocol["cutoffs"]:
        for margin in protocol["margins"]:
            selection = {"status": "INNER_CALIBRATED_CURRENT_IDENTITY_ONLY", "probability_min": cutoff, "margin_min": margin,
                         "unknown_max": protocol["unknown_max"], "global_regret_max": .2, "temperature": 1., "simple_family": simple_family}
            result = assess_claims(groups, predicted, selection)
            risk = result["micro"]["decision_risk_including_UNKNOWN_and_incorrect_NONE"]
            nonvacuous = result["counts"].get("accepted_identity_claims", 0) >= protocol["min_identity_claims"] and len(result["videos_with_actual_identity_claims"]) >= protocol["min_identity_claim_videos"]
            points.append({"selection": selection, "result": {k: v for k, v in result.items() if k != "choices"},
                           "nonvacuous": nonvacuous, "empirical_current_safe": nonvacuous and risk is not None and risk <= protocol["risk_max"]})
    good = [p for p in points if p["empirical_current_safe"]]
    chosen = min(good, key=lambda p: (-p["result"]["counts"].get("correct_TARGET_claims", 0),
                                   p["result"]["micro"]["decision_risk_including_UNKNOWN_and_incorrect_NONE"],
                                   -p["selection"]["probability_min"], -p["selection"]["margin_min"])) if good else None
    selection = chosen["selection"] if chosen else {"status": "CALIBRATION_ABSTAIN", "probability_min": 1.01,
        "margin_min": 1., "unknown_max": protocol["unknown_max"], "temperature": 1., "global_regret_max": .2,
        "simple_family": simple_family, "reason": "NO_INNER_NONVACUOUS_CURRENT_SAFE_POINT_NOT_SCIENTIFIC_PASS"}
    return selection, points
