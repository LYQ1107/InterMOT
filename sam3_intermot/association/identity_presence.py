"""Small open-set identity-presence layer for N72R20R3.

The module scores an existing candidate set with frozen identity evidence and
returns either ``ABSENT`` or one existing candidate UID.  It never creates a
candidate, changes a public assignment, reads GT, or updates memory.  Causal
callers must perform the score-before-update ordering and decide separately
whether a trusted write is eligible.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np


FEATURE_DIMENSION = 512
R3_SCHEMA_VERSION = "N72R20R3_IDENTITY_PRESENCE_V1"
_POSTHOC_KEYS = {
    "target_gt_id",
    "target_iou",
    "target_candidate_iou",
    "taxonomy",
    "candidate_set_present",
    "best_target_candidate_uid",
    "best_target_candidate_iou",
    "target_gt_present",
}


def _unit(value: Any, label: str) -> np.ndarray | None:
    if value is None:
        return None
    vector = np.asarray(value, dtype=np.float32).reshape(-1)
    if vector.shape != (FEATURE_DIMENSION,) or not np.isfinite(vector).all():
        return None
    norm = float(np.linalg.norm(vector))
    if norm <= 1.0e-8:
        return None
    return vector / norm


def _candidate_feature(row: Mapping[str, Any]) -> np.ndarray | None:
    return _unit(row.get("feature", row.get("embedding")), "candidate feature")


def _finite_or_none(value: Any) -> float | None:
    if value is None:
        return None
    number = float(value)
    return number if np.isfinite(number) else None


def _assignment_uid(base_assignment: Any, public_id: int | None = None) -> str | None:
    if isinstance(base_assignment, Mapping):
        public_assignments = base_assignment.get("public_assignments")
        if isinstance(public_assignments, Sequence) and public_id is not None:
            for item in public_assignments:
                if not isinstance(item, Mapping) or item.get("public_id") is None:
                    continue
                if int(item["public_id"]) == int(public_id):
                    value = item.get("candidate_uid")
                    if value is None or item.get("status") in {"NO_CANDIDATE_ASSIGNED", "ASSIGNED_NONE"}:
                        return None
                    return str(value)
            return None
        value = base_assignment.get("candidate_uid")
        if value is None and base_assignment.get("status") == "NO_CANDIDATE_ASSIGNED":
            return None
    else:
        value = base_assignment
    return None if value in (None, "", "None") else str(value)


def _stats(scores: np.ndarray) -> dict[str, float | None]:
    if scores.size == 0:
        return {
            "mean": None,
            "std": None,
            "median": None,
            "min": None,
            "max": None,
        }
    return {
        "mean": float(np.mean(scores)),
        "std": float(np.std(scores)),
        "median": float(np.median(scores)),
        "min": float(np.min(scores)),
        "max": float(np.max(scores)),
    }


def _rank(scores: np.ndarray, rows: Sequence[Mapping[str, Any]]) -> list[int]:
    return sorted(range(len(rows)), key=lambda index: (-float(scores[index]), str(rows[index]["candidate_uid"])))


def _absent(
    *,
    public_id: int,
    frame: int,
    policy: Mapping[str, Any],
    reason: str,
    features: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "schema_version": R3_SCHEMA_VERSION,
        "public_id": int(public_id),
        "frame": int(frame),
        "method": str(policy.get("method", "UNKNOWN")),
        "policy_version": str(policy.get("policy_version", "UNSPECIFIED")),
        "present": False,
        "selected_candidate_uid": None,
        "presence_reason": str(reason),
        "presence_confidence": None,
        **dict(features or {}),
        "runtime_future_gt_used": False,
        "runtime_gt_read": False,
        "posthoc_gt_used": False,
    }


def _b3_score(features: Mapping[str, Any], name: str) -> float | None:
    if name == "top1_minus_mean":
        return _finite_or_none(features.get("learned_top1_minus_mean"))
    if name == "top1_minus_median":
        return _finite_or_none(features.get("learned_top1_minus_median"))
    if name == "top1_zscore":
        top1 = _finite_or_none(features.get("learned_top1_score"))
        mean = _finite_or_none(features.get("learned_score_mean"))
        std = _finite_or_none(features.get("learned_score_std"))
        if top1 is None or mean is None or std is None:
            return None
        return float((top1 - mean) / max(std, 1.0e-8))
    raise ValueError(f"unknown B3 feature: {name}")


def _b4_probability(features: Mapping[str, Any], policy: Mapping[str, Any]) -> float | None:
    names = [str(name) for name in policy.get("feature_names", [])]
    weights = np.asarray(policy.get("weights", []), dtype=np.float64).reshape(-1)
    means = np.asarray(policy.get("scaler_mean", []), dtype=np.float64).reshape(-1)
    scales = np.asarray(policy.get("scaler_scale", []), dtype=np.float64).reshape(-1)
    values = np.asarray([features.get(name, np.nan) for name in names], dtype=np.float64)
    if not names or weights.shape != values.shape or means.shape != values.shape or scales.shape != values.shape:
        return None
    if not np.isfinite(values).all() or not np.isfinite(weights).all() or not np.isfinite(means).all() or not np.isfinite(scales).all():
        return None
    standardized = (values - means) / np.maximum(scales, 1.0e-8)
    logit = float(np.dot(standardized, weights) + float(policy.get("intercept", 0.0)))
    logit = float(np.clip(logit, -60.0, 60.0))
    return float(1.0 / (1.0 + np.exp(-logit)))


def evaluate_identity_presence(
    *,
    learned_state: Any,
    human_anchor: Any,
    candidate_rows: Sequence[Mapping[str, Any]],
    base_assignment: Any,
    runtime_context: Mapping[str, Any],
    frozen_policy: Mapping[str, Any],
) -> dict[str, Any]:
    """Evaluate one frame without GT, assignment mutation, or memory update."""

    public_id = int(runtime_context.get("public_id", 0))
    frame = int(runtime_context.get("frame", 0))
    rows = [dict(row) for row in candidate_rows]
    uids = [str(row.get("candidate_uid")) for row in rows]
    if any(uid in {"", "None"} for uid in uids) or len(uids) != len(set(uids)):
        raise ValueError("candidate_uid collision or missing candidate UID")
    for row in rows:
        if row.get("runtime_future_gt_used") is True or row.get("runtime_gt_read") is True:
            raise ValueError("runtime GT metadata is forbidden")
        if row.get("posthoc_gt_used") is True or _POSTHOC_KEYS.intersection(row):
            raise ValueError("posthoc labels must not enter runtime presence input")
    expected_encoder = frozen_policy.get("encoder_sha256")
    actual_encoder = runtime_context.get("encoder_sha256")
    if expected_encoder is not None and str(actual_encoder) != str(expected_encoder):
        return _absent(public_id=public_id, frame=frame, policy=frozen_policy, reason="FEATURE_LINEAGE_MISMATCH")
    state = _unit(learned_state, "learned state")
    anchor = _unit(human_anchor, "human anchor")
    if state is None:
        return _absent(public_id=public_id, frame=frame, policy=frozen_policy, reason="STATE_UNAVAILABLE")
    if anchor is None:
        return _absent(public_id=public_id, frame=frame, policy=frozen_policy, reason="STATE_UNAVAILABLE")
    if not rows:
        return _absent(public_id=public_id, frame=frame, policy=frozen_policy, reason="NO_CANDIDATES")
    features_by_index = [_candidate_feature(row) for row in rows]
    valid = [index for index, feature in enumerate(features_by_index) if feature is not None]
    if not valid:
        return _absent(public_id=public_id, frame=frame, policy=frozen_policy, reason="FEATURE_MISSING")
    score_indices = np.asarray(valid, dtype=np.int64)
    matrix = np.asarray([features_by_index[index] for index in valid], dtype=np.float32)
    learned_scores = matrix @ state
    anchor_scores = matrix @ anchor
    learned_order = sorted(range(len(valid)), key=lambda index: (-float(learned_scores[index]), uids[valid[index]]))
    anchor_order = sorted(range(len(valid)), key=lambda index: (-float(anchor_scores[index]), uids[valid[index]]))
    top1_local = learned_order[0]
    top2_local = learned_order[1] if len(learned_order) > 1 else None
    anchor_top1_local = anchor_order[0]
    anchor_top2_local = anchor_order[1] if len(anchor_order) > 1 else None
    top1_score = float(learned_scores[top1_local])
    top2_score = None if top2_local is None else float(learned_scores[top2_local])
    anchor_top1_score = float(anchor_scores[anchor_top1_local])
    anchor_top2_score = None if anchor_top2_local is None else float(anchor_scores[anchor_top2_local])
    learned_stats = _stats(learned_scores)
    anchor_stats = _stats(anchor_scores)
    learned_margin = None if top2_score is None else top1_score - top2_score
    anchor_margin = None if anchor_top2_score is None else anchor_top1_score - anchor_top2_score
    features: dict[str, Any] = {
        "candidate_count": len(rows),
        "valid_feature_count": len(valid),
        "learned_top1_candidate_uid": uids[valid[top1_local]],
        "learned_top1_score": top1_score,
        "learned_top2_candidate_uid": None if top2_local is None else uids[valid[top2_local]],
        "learned_top2_score": top2_score,
        "learned_margin": learned_margin,
        "learned_score_mean": learned_stats["mean"],
        "learned_score_std": learned_stats["std"],
        "learned_score_median": learned_stats["median"],
        "learned_score_min": learned_stats["min"],
        "learned_score_max": learned_stats["max"],
        "learned_top1_minus_mean": None if learned_stats["mean"] is None else top1_score - float(learned_stats["mean"]),
        "learned_top1_minus_median": None if learned_stats["median"] is None else top1_score - float(learned_stats["median"]),
        "human_anchor_top1_candidate_uid": uids[valid[anchor_top1_local]],
        "human_anchor_top1_score": anchor_top1_score,
        "human_anchor_top2_score": anchor_top2_score,
        "human_anchor_margin": anchor_margin,
        "base_target_candidate_uid": _assignment_uid(base_assignment, public_id),
        "base_assignment_margin": _finite_or_none(runtime_context.get("base_assignment_margin")),
        "learned_rank_of_base_candidate": None,
        "learned_score_of_base_candidate": None,
        "predicted_motion_iou_of_base_candidate": _finite_or_none(runtime_context.get("predicted_motion_iou_of_base_candidate")),
        "candidate_presence_of_base_candidate": _assignment_uid(base_assignment) in set(uids),
        "native_continuity_of_base_candidate": runtime_context.get("native_continuity_of_base_candidate"),
        "frames_since_human_initialization": int(runtime_context.get("frames_since_human_initialization", frame)),
        "frames_since_last_memory_write": int(runtime_context.get("frames_since_last_memory_write", frame)),
        "learned_state_human_anchor_cosine": float(np.dot(state, anchor)),
        "memory_state_hash": runtime_context.get("memory_state_hash"),
        "human_anchor_hash": runtime_context.get("human_anchor_hash"),
    }
    base_uid = features["base_target_candidate_uid"]
    if base_uid in uids:
        base_index = uids.index(base_uid)
        base_local = valid.index(base_index) if base_index in valid else None
        if base_local is not None:
            base_rank = next((rank + 1 for rank, local in enumerate(learned_order) if valid[local] == base_index), None)
            features["learned_rank_of_base_candidate"] = base_rank
            features["learned_score_of_base_candidate"] = float(learned_scores[base_local])

    method = str(frozen_policy.get("method", "B0_ALWAYS_PRESENT"))
    present = False
    reason = "DISTRIBUTION_CONFIDENCE_LOW"
    confidence: float | None = None
    if method in {"B0", "B0_ALWAYS_PRESENT"}:
        present = True
        reason = "PRESENT_TOP1"
        confidence = top1_score
    elif method in {"B1", "B1_ABSOLUTE_SCORE"}:
        threshold = float(frozen_policy["tau_abs"])
        present = top1_score >= threshold
        reason = "PRESENT_TOP1" if present else "BELOW_ABSOLUTE_THRESHOLD"
        confidence = top1_score - threshold
    elif method in {"B2", "B2_ABSOLUTE_MARGIN"}:
        threshold = float(frozen_policy["tau_abs"])
        margin_threshold = float(frozen_policy["tau_margin"])
        margin_value = float("inf") if learned_margin is None else learned_margin
        present = top1_score >= threshold and margin_value >= margin_threshold
        reason = "PRESENT_TOP1" if present else ("BELOW_ABSOLUTE_THRESHOLD" if top1_score < threshold else "INSUFFICIENT_MARGIN")
        confidence = min(top1_score - threshold, margin_value - margin_threshold)
    elif method in {"B3", "B3_DISTRIBUTION"}:
        score_name = str(frozen_policy["score_feature"])
        distribution_score = _b3_score(features, score_name)
        threshold = float(frozen_policy["tau_distribution"])
        present = distribution_score is not None and distribution_score >= threshold
        reason = "PRESENT_TOP1" if present else "DISTRIBUTION_CONFIDENCE_LOW"
        confidence = None if distribution_score is None else distribution_score - threshold
        features["distribution_score"] = distribution_score
    elif method in {"B4", "LOGISTIC_PRESENCE"}:
        probability = _b4_probability(features, frozen_policy)
        threshold = float(frozen_policy.get("operating_threshold", 0.5))
        present = probability is not None and probability >= threshold
        reason = "MODEL_PRESENT" if present else "MODEL_ABSTAIN"
        confidence = None if probability is None else probability - threshold
        features["presence_probability"] = probability
    else:
        raise ValueError(f"unknown identity-presence method: {method}")
    return {
        "schema_version": R3_SCHEMA_VERSION,
        "public_id": public_id,
        "frame": frame,
        "method": method,
        "policy_version": str(frozen_policy.get("policy_version", "UNSPECIFIED")),
        "present": bool(present),
        "selected_candidate_uid": features["learned_top1_candidate_uid"] if present else None,
        "presence_reason": reason,
        "presence_confidence": confidence,
        **features,
        "runtime_future_gt_used": False,
        "runtime_gt_read": False,
        "posthoc_gt_used": False,
    }


def commit_eligibility(presence: Mapping[str, Any]) -> dict[str, Any]:
    """Apply R3's conservative eligibility rule without changing assignment."""

    base_uid = presence.get("base_target_candidate_uid")
    learned_uid = presence.get("learned_top1_candidate_uid")
    present = bool(presence.get("present"))
    eligible = present and base_uid not in (None, "", "None") and base_uid == learned_uid
    if not present:
        reason = "NO_WRITE_ABSENT"
    elif base_uid in (None, "", "None"):
        reason = "NO_WRITE_BASE_NONE"
    elif base_uid != learned_uid:
        reason = "NO_WRITE_BASE_LEARNED_DISAGREEMENT"
    else:
        reason = "ELIGIBLE_PRESENT_BASE_LEARNED_AGREE"
    return {
        "commit_eligible": bool(eligible),
        "commit_reason": reason,
        "runtime_future_gt_used": False,
    }


__all__ = ["FEATURE_DIMENSION", "R3_SCHEMA_VERSION", "commit_eligibility", "evaluate_identity_presence"]
