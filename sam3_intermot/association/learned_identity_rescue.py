"""Target-only, exact-solver-compatible learned identity rescue.

The module creates no candidates and does not assign public IDs.  It only
constructs a residual for the explicitly human-initialized target public ID;
the existing row-max-preserving fusion and exact public assignment remain the
decision authority.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np

from .relative_persistent_state_edge import fuse_row_max_preserving


def _finite_scores(value: Any, count: int) -> np.ndarray:
    scores = np.asarray(value, dtype=np.float64).reshape(-1)
    if scores.shape != (count,) or not np.isfinite(scores).all():
        raise ValueError(f"learned scores must be finite with shape {(count,)}, got {scores.shape}")
    return scores


def _rank(scores: np.ndarray, candidates: Sequence[Mapping[str, Any]]) -> list[int]:
    return sorted(range(len(candidates)), key=lambda index: (-float(scores[index]), str(candidates[index]["candidate_uid"])))


def build_rescue_delta(
    *,
    base_matrix: Sequence[Sequence[float]],
    public_id_axis: Sequence[int],
    target_public_id: int,
    candidate_rows: Sequence[Mapping[str, Any]],
    learned_scores: Sequence[float],
    base_assignment_candidate_uid: str | None,
    runtime_policy: Mapping[str, Any],
    motion_plausibility: float | None = None,
) -> dict[str, Any]:
    """Build an evidence-triggered target-only residual and fused matrix."""

    base = np.asarray(base_matrix, dtype=np.float64)
    if base.ndim != 2 or base.shape[0] != len(candidate_rows) or base.shape[1] != len(public_id_axis):
        raise ValueError("base matrix/candidate/public axes are inconsistent")
    if not np.isfinite(base).all():
        raise ValueError("base matrix contains non-finite values")
    publics = [int(value) for value in public_id_axis]
    if int(target_public_id) not in publics:
        raise ValueError("target public ID is not in the explicit public axis")
    if any(row.get("runtime_future_gt_used") is True or row.get("runtime_gt_read") is True for row in candidate_rows):
        raise ValueError("runtime_future_gt_used/runtime_gt_read must be false")
    scores = _finite_scores(learned_scores, len(candidate_rows))
    order = _rank(scores, candidate_rows)
    top1_index = order[0] if order else None
    top2_index = order[1] if len(order) > 1 else None
    top1_uid = None if top1_index is None else str(candidate_rows[top1_index]["candidate_uid"])
    top2_score = None if top2_index is None else float(scores[top2_index])
    top1_score = None if top1_index is None else float(scores[top1_index])
    margin = None if top1_score is None or top2_score is None else top1_score - top2_score
    lambda_value = float(runtime_policy.get("lambda", 1.0))
    margin_threshold = float(runtime_policy.get("margin_threshold", 0.0))
    min_motion = runtime_policy.get("min_motion_plausibility")
    if not np.isfinite(lambda_value) or lambda_value < 0.0:
        raise ValueError("runtime lambda must be finite and non-negative")
    if not np.isfinite(margin_threshold) or margin_threshold < 0.0:
        raise ValueError("runtime margin_threshold must be finite and non-negative")
    target_column = publics.index(int(target_public_id))
    residual = np.zeros_like(base, dtype=np.float64)
    activation_reason = "INACTIVE_NO_LEARNED_TOP1"
    assigned_uid = None if base_assignment_candidate_uid in (None, "", "None") else str(base_assignment_candidate_uid)
    if top1_uid is None:
        activation_reason = "INACTIVE_NO_VALID_FEATURE"
    elif assigned_uid == top1_uid:
        activation_reason = "INACTIVE_BASE_ALREADY_LEARNED_TOP1"
    elif margin is None or margin < margin_threshold:
        activation_reason = "INACTIVE_LEARNED_MARGIN_BELOW_THRESHOLD"
    elif min_motion is not None and (motion_plausibility is None or float(motion_plausibility) < float(min_motion)):
        activation_reason = "INACTIVE_MOTION_BELOW_THRESHOLD"
    else:
        relative = scores - float(np.mean(scores)) if scores.size else scores.copy()
        residual[:, target_column] = lambda_value * relative
        activation_reason = "ACTIVE_TARGET_ONLY_LEARNED_DISAGREEMENT"
    fusion = fuse_row_max_preserving(base, residual, state_edge_scale=1.0)
    non_target_values = np.delete(residual, target_column, axis=1) if residual.shape[1] > 1 else np.zeros((residual.shape[0], 0), dtype=np.float64)
    return {
        "schema_version": "N72R20R2_TARGET_ONLY_RESCUE_V1",
        "target_public_id": int(target_public_id),
        "public_id_axis": publics,
        "candidate_uids": [str(row["candidate_uid"]) for row in candidate_rows],
        "base_assignment_candidate_uid": assigned_uid,
        "learned_top1_candidate_uid": top1_uid,
        "learned_top1_score": top1_score,
        "learned_top2_score": top2_score,
        "learned_margin": margin,
        "lambda": lambda_value,
        "margin_threshold": margin_threshold,
        "motion_plausibility": None if motion_plausibility is None else float(motion_plausibility),
        "min_motion_plausibility": None if min_motion is None else float(min_motion),
        "rescue_active": activation_reason == "ACTIVE_TARGET_ONLY_LEARNED_DISAGREEMENT",
        "activation_reason": activation_reason,
        "raw_delta_non_target_max_abs": 0.0 if non_target_values.size == 0 else float(np.max(np.abs(non_target_values))),
        "relative_delta": residual.tolist(),
        "fused_candidate_public_scores": np.asarray(fusion["fused"], dtype=np.float64).tolist(),
        "row_max_preserved": bool(fusion["row_max_preserved"]),
        "candidate_axis_unchanged": True,
        "runtime_future_gt_used": False,
    }


__all__ = ["build_rescue_delta"]
