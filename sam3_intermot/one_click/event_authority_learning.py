"""Causal event inputs and distinct offline objectives for fresh-scene MOT.

No truth field is accepted by the runtime encoder. Offline labels enter only
the separate target builder. Correlated windows never become independent
corrections merely because they occur at different frames.
"""
from collections import Counter, defaultdict
import numpy as np
from .intervention_features import FEATURE_NAMES, feature_vector

BRANCH_NAMES = ("KEEP", "REJECT_TARGET", "RECOVERY", "RAW_IDENTITY_TOP", "LEARNED_IDENTITY_TOP",
                "TOP_ALTERNATIVE", "FEASIBLE_GLOBAL_SWAP", "DELAYED_CHALLENGER")
OBJECTIVES = ("L0_CURRENT_CORRECTNESS", "L1_EVENT_CORRECTION", "L2_LONG_CONTINUITY_H20_H50_H100", "L3_HARM_AWARE",
              "L4_GLOBAL_ASSIGNMENT_VALUE", "L5_ACTUAL_TRAJECTORY_UTILITY", "L6_RISK_CONSTRAINED_MULTI_OBJECTIVE")
CURRENT_DIM = len(FEATURE_NAMES) + len(BRANCH_NAMES)


def encode_runtime_input(runtime_features, branch, *, past_steps=3):
    if branch not in BRANCH_NAMES or past_steps not in (3, 8):
        raise ValueError("Unregistered current action or causal history length")
    permitted = {"features", "feature_vector", "causal_previous_feature_vectors", "current_context_path", "current_context_sha256"}
    if not set(runtime_features).issubset(permitted):
        raise ValueError("Offline labels or unknown fields cannot enter runtime input")
    features = runtime_features["features"]
    if set(features) != set(FEATURE_NAMES):
        raise ValueError("Only frozen causal feature names are permitted; no GT/future extras")
    vector = feature_vector(features)
    supplied = np.asarray(runtime_features["feature_vector"], dtype=np.float32)
    if supplied.shape != (len(FEATURE_NAMES),) or not np.array_equal(vector, supplied):
        raise ValueError("Runtime vector differs from current causal features")
    past = np.asarray(runtime_features["causal_previous_feature_vectors"]["H" + str(past_steps)], dtype=np.float32)
    if past.shape != (past_steps, len(FEATURE_NAMES)) or not np.isfinite(past).all():
        raise ValueError("Exact sealed past-only feature axis required")
    one_hot = np.zeros(len(BRANCH_NAMES), dtype=np.float32)
    one_hot[BRANCH_NAMES.index(branch)] = 1.
    return np.concatenate((vector, one_hot)), past


def objective_target(labels, objective, *, actual_trajectory_utility=None):
    if objective not in OBJECTIVES:
        raise ValueError("Unregistered supervision objective")
    current = labels["current_t"]
    current_harm = bool(current["N10"] or current["non_target_damage"] or current["verified_OTHER_writes"] or current["UNKNOWN_writes"])
    if objective == OBJECTIVES[0]:
        return {"benefit": float(current["target_correct"]), "risk": float(current_harm),
                "value": float(current["N01"] - current["N10"]), "scope": "CURRENT_CORRECTNESS_NOT_FUTURE_CORRECTION"}
    long = labels["future"]["H100"]
    if not long["complete"]:
        return None  # Incomplete futures are NOT negative labels.
    risk = current_harm or bool(long["risk_label"])
    if objective in (OBJECTIVES[1], OBJECTIVES[3]):
        value = long["normalized_target_value_label"]
        benefit = bool(long["benefit_label"]) and not current_harm
        scope = "ONE_SHOT_H100_EVENT_COMPONENTS_NOT_PROVEN_INDEPENDENT_ONSET"
    elif objective == OBJECTIVES[2]:
        windows = [labels["future"]["H" + str(h)] for h in (20, 50, 100)]
        if not all(w["complete"] for w in windows):
            return None
        value = sum(w["normalized_target_value_label"] for w in windows) / 3.
        benefit, scope = value > 0 and not risk, "MEAN_H20_H50_H100_TARGET_CONTINUITY"
    elif objective == OBJECTIVES[4]:
        value = long["normalized_global_proxy_label"]
        benefit, scope = value > 0 and not risk, "ACTUAL_FULL_ASSIGNMENT_COMPONENT_PROXY_NOT_HOTA"
    elif objective == OBJECTIVES[5]:
        if actual_trajectory_utility is None or not actual_trajectory_utility["complete_H100"]:
            return None
        value = actual_trajectory_utility["L5_value_label"]
        if value is None:
            raise ValueError("A complete trajectory utility needs actual pinned metrics")
        benefit, scope = value > 0 and not risk, "ACTUAL_PINNED_FUTURE_HOTA_ASSA_PAIRED_UTILITY"
    else:
        future = labels["raw_frame_components"][1:101]
        if len(future) != 100:
            raise ValueError("Complete H100 cannot contain a truncated raw trajectory")
        extra_unknown = max(0, long["UNKNOWN"] - sum(r["KEEP_outcome"] == "UNKNOWN" for r in future)) / 100.
        extra_takeover = max(0, long["verified_OTHER"] - long["KEEP_verified_OTHER"]) / 100.
        value = long["normalized_global_proxy_label"] - extra_takeover - .1 * extra_unknown
        benefit, scope = value > 0 and not risk, "GLOBAL_COMPONENT_MINUS_PERSISTENT_TAKEOVER_AND_UNCERTAINTY"
    return {"benefit": float(benefit), "risk": float(risk), "value": float(np.clip(value, -3., 3.)), "scope": scope}


def correlated_sequence_weights(rows):
    """Equal video -> episode -> observed interval -> event -> distinct arm.

    Observed intervals are only correlation controls, never claims of proven
    independent causal roots. Caller removes duplicate executed action configs
    BEFORE passing rows, preserving KEEP as the same-event pair reference.
    """
    if not rows:
        raise ValueError("No usable actual event observations")
    episodes, groups, events, arms = defaultdict(set), defaultdict(set), defaultdict(set), Counter()
    for row in rows:
        seq, episode, group, event = (row[k] for k in ("sequence", "episode_uid", "correlation_group_not_proven_causal_origin", "event_uid"))
        episodes[seq].add(episode)
        groups[seq, episode].add(group)
        events[seq, episode, group].add(event)
        arms[seq, episode, group, event] += 1
    weights = []
    for row in rows:
        seq, episode, group, event = (row[k] for k in ("sequence", "episode_uid", "correlation_group_not_proven_causal_origin", "event_uid"))
        weights.append(1. / (len(episodes) * len(episodes[seq]) * len(groups[seq, episode]) * len(events[seq, episode, group]) * arms[seq, episode, group, event]))
    result = np.asarray(weights, dtype=np.float64)
    if not np.isclose(result.sum(), 1.):
        raise ValueError("Hierarchical sequence macro weights must sum to one")
    return result


def fit_normalizer(current, weights):
    """FIT current inputs only; branch one-hots are neither centered nor fit."""
    raw = np.asarray(current, dtype=np.float64)[:, :len(FEATURE_NAMES)]
    weights = np.asarray(weights, dtype=np.float64)
    if raw.shape[0] != len(weights) or not np.isclose(weights.sum(), 1.) or not np.isfinite(raw).all():
        raise ValueError("Invalid FIT-only observations")
    mean = (raw * weights[:, None]).sum(0)
    scale = np.maximum(np.sqrt(((raw - mean) ** 2 * weights[:, None]).sum(0)), .05)
    return mean.astype(np.float32), scale.astype(np.float32)


def normalize_inputs(current, past, mean, scale):
    current, past = np.array(current, dtype=np.float32, copy=True), np.array(past, dtype=np.float32, copy=True)
    current[..., :len(FEATURE_NAMES)] = (current[..., :len(FEATURE_NAMES)] - mean) / scale
    zero_padding = np.all(past == 0, axis=-1)
    past = (past - mean) / scale
    past[zero_padding] = 0.
    return current, past
