"""Offline frozen-hand-filter support, not a policy or learned-score result.

Only current causal features enter this helper. An ideal constant model score
isolates the hand-filter ceiling; it cannot authorize an action or prove safety.
"""
import math
from sam3_intermot.one_click.intervention_features import FEATURE_NAMES


def hard_filter_violations(features, point, *, confirmations):
    if set(features) != set(FEATURE_NAMES) or not all(math.isfinite(v) for v in features.values()):
        raise ValueError("Exact finite causal feature axis required, never GT/future fields")
    if set(point) != {"claim_min", "risk_max", "global_regret_max", "anchor_advantage_min", "confirmation_delay"}:
        raise ValueError("Use the unchanged registered operating point")
    if not all(math.isfinite(v) for v in point.values()) or not math.isfinite(confirmations) or confirmations < 0:
        raise ValueError("Invalid operating point/confirmation count")
    reasons = []
    if confirmations < point["confirmation_delay"]:
        reasons.append("CONFIRMATION_DELAY")
    if features["global_regret"] > point["global_regret_max"]:
        reasons.append("GLOBAL_REGRET")
    if features["displaced_count"] > 0:
        reasons.append("ANY_DISPLACED_PUBLIC_ID")
    if features["proposal_NONE"]:
        if features["NONE_probability"] < .95:
            reasons.append("NONE_PROBABILITY")
        if features["NONE_advantage"] < .05:
            reasons.append("NONE_ADVANTAGE")
        if features["KEEP_anchor_cosine"] >= .6:
            reasons.append("KEEP_ANCHOR_TOO_HIGH_FOR_NONE")
    else:
        if features["anchor_advantage_vs_KEEP"] < point["anchor_advantage_min"]:
            reasons.append("ANCHOR_ADVANTAGE")
        if features["quality"] < .5:
            reasons.append("CANDIDATE_QUALITY")
        if features["anchor_cosine"] < .6:
            reasons.append("ANCHOR_COSINE")
    return reasons


def safe_direct_source_row(row):
    """Same frozen exact-CF direct-repair scope, before action-config dedup.

    This OFFLINE label predicate must never be used to select runtime actions.
    Complete H100/component-positive rows are not independent correction onsets.
    """
    labels = row["offline_supervision_labels"]
    future, current = labels["future"]["H100"], labels["current_t"]
    return bool(row["branch"] not in ("KEEP", "DELAYED_CHALLENGER")
                and future["complete"] and future["benefit_label"] is True
                and future["N01"] > future["N10"]
                and not labels["H100_any_harm_including_current_t_label"]
                and row["action"].get("candidate_uid") is not None
                and current["outcome"] == "TARGET" and current["N01"] == 1
                and row["frame"] in labels["effective_direct_action_frames"])
