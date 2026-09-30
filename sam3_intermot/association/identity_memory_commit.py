"""Runtime-only trusted commit policy for learned identity memory.

This module evaluates whether an already assigned observation is safe to write
to a human-initialized learned identity state.  It never reads GT, changes a
candidate set, or assigns a public ID.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np


def _unit(value: Any) -> np.ndarray | None:
    if value is None:
        return None
    array = np.asarray(value, dtype=np.float32).reshape(-1)
    if array.size == 0 or not np.isfinite(array).all():
        return None
    norm = float(np.linalg.norm(array))
    return None if norm <= 1.0e-8 else array / norm


def evaluate_commit(
    *,
    public_id: int,
    assigned_candidate_uid: str | None,
    candidate_rows: Sequence[Mapping[str, Any]],
    learned_state: Any,
    frame: int,
    margin_threshold: float = 0.0,
    require_learned_top1: bool = True,
    temporal_confirmed: bool = True,
    require_temporal_confirmation: bool = False,
) -> dict[str, Any]:
    """Return an auditable, GT-blind decision for one machine observation.

    ``temporal_confirmed`` is supplied by the caller from prior runtime
    evidence; it is not inferred from future frames.  A ``NONE`` or missing
    candidate can never be committed.
    """

    if float(margin_threshold) < 0.0 or not np.isfinite(float(margin_threshold)):
        raise ValueError("margin_threshold must be finite and non-negative")
    if any(row.get("runtime_future_gt_used") is True or row.get("runtime_gt_read") is True for row in candidate_rows):
        raise ValueError("runtime_future_gt_used/runtime_gt_read must be false")
    rows = [dict(row) for row in candidate_rows]
    by_uid = {str(row.get("candidate_uid")): row for row in rows}
    state = _unit(learned_state)
    base: dict[str, Any] = {
        "public_id": int(public_id),
        "frame": int(frame),
        "assigned_candidate_uid": None if assigned_candidate_uid in (None, "", "None") else str(assigned_candidate_uid),
        "margin_threshold": float(margin_threshold),
        "require_learned_top1": bool(require_learned_top1),
        "require_temporal_confirmation": bool(require_temporal_confirmation),
        "temporal_confirmed": bool(temporal_confirmed),
        "runtime_future_gt_used": False,
    }
    if base["assigned_candidate_uid"] is None:
        return {**base, "commit": False, "reason": "NONE_ASSIGNMENT", "learned_top1_candidate_uid": None, "learned_rank": None, "learned_score": None, "learned_top2_score": None, "learned_margin": None}
    if base["assigned_candidate_uid"] not in by_uid:
        return {**base, "commit": False, "reason": "ASSIGNED_CANDIDATE_MISSING", "learned_top1_candidate_uid": None, "learned_rank": None, "learned_score": None, "learned_top2_score": None, "learned_margin": None}
    if state is None:
        return {**base, "commit": False, "reason": "LEARNED_STATE_MISSING", "learned_top1_candidate_uid": None, "learned_rank": None, "learned_score": None, "learned_top2_score": None, "learned_margin": None}

    scored: list[tuple[float, str]] = []
    for row in rows:
        feature = _unit(row.get("feature", row.get("embedding")))
        if feature is None:
            continue
        scored.append((float(np.dot(state, feature)), str(row["candidate_uid"])))
    scored.sort(key=lambda item: (-item[0], item[1]))
    if not scored:
        return {**base, "commit": False, "reason": "NO_VALID_FEATURE", "learned_top1_candidate_uid": None, "learned_rank": None, "learned_score": None, "learned_top2_score": None, "learned_margin": None}
    top1_score, top1_uid = scored[0]
    top2_score = scored[1][0] if len(scored) > 1 else None
    margin = None if top2_score is None else float(top1_score - top2_score)
    assigned_scores = {uid: score for score, uid in scored}
    assigned_score = assigned_scores.get(base["assigned_candidate_uid"])
    assigned_rank = next((index + 1 for index, (_, uid) in enumerate(scored) if uid == base["assigned_candidate_uid"]), None)
    if assigned_score is None:
        reason = "ASSIGNED_FEATURE_MISSING"
    elif require_learned_top1 and base["assigned_candidate_uid"] != top1_uid:
        reason = "ASSIGNED_NOT_LEARNED_TOP1"
    elif margin is None or margin < float(margin_threshold):
        reason = "LEARNED_MARGIN_BELOW_THRESHOLD"
    elif require_temporal_confirmation and not temporal_confirmed:
        reason = "TEMPORAL_CONFIRMATION_INCOMPLETE"
    else:
        reason = "COMMIT_ACCEPTED"
    return {
        **base,
        "commit": reason == "COMMIT_ACCEPTED",
        "reason": reason,
        "learned_top1_candidate_uid": top1_uid,
        "learned_rank": assigned_rank,
        "learned_score": assigned_score,
        "learned_top1_score": float(top1_score),
        "learned_top2_score": top2_score,
        "learned_margin": margin,
    }


__all__ = ["evaluate_commit"]
