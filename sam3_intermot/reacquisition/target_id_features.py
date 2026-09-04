"""Shared causal feature construction for the N72R7 learned decoder.

The feature builder is intentionally independent of GT and public-ID labels.
Training labels are attached by the offline corpus builder only; runtime
replay calls the same function with the sealed candidate pool and current
causal target state.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np

from .target_candidate_pool import MAIN_B0_CANDIDATE, TARGET_SESSION_CURRENT_RAW, FEATURE_DIM
from .target_candidate_selector import box_iou


CANDIDATE_FEATURE_DIM = 530
CONTEXT_FEATURE_DIM = 522


def _unit_feature(value: Any) -> tuple[np.ndarray, float]:
    if value is None:
        return np.zeros(FEATURE_DIM, dtype=np.float32), 0.0
    array = np.asarray(value, dtype=np.float32).reshape(-1)
    if array.size != FEATURE_DIM or not np.all(np.isfinite(array)):
        raise ValueError("decoder feature must be finite 512-D")
    norm = float(np.linalg.norm(array))
    if norm <= 1.0e-6:
        raise ValueError("decoder feature has zero norm")
    return (array / norm).astype(np.float32), 1.0


def _norm_box(box: Sequence[float] | None, width: float, height: float) -> np.ndarray:
    if box is None:
        return np.zeros(4, dtype=np.float32)
    value = np.asarray(box, dtype=np.float32).reshape(-1)
    if value.size != 4 or not np.all(np.isfinite(value)):
        raise ValueError("decoder box must be four finite values")
    return np.asarray(
        [value[0] / width, value[1] / height, value[2] / width, value[3] / height],
        dtype=np.float32,
    )


def _relative_box(box: Sequence[float], reference: Sequence[float] | None, width: float, height: float) -> np.ndarray:
    candidate = _norm_box(box, width, height)
    if reference is None:
        return candidate
    return candidate - _norm_box(reference, width, height)


def _same_raw(candidate: Mapping[str, Any], previous_raw: int | None, previous_scope: str | None) -> tuple[float, float]:
    raw = candidate.get("official_raw_sam_id")
    scope = candidate.get("native_scope", candidate.get("native_tid_scope"))
    scope_match = float(previous_scope is not None and scope is not None and str(scope) == str(previous_scope))
    raw_match = float(
        scope_match
        and previous_raw is not None
        and raw is not None
        and int(raw) == int(previous_raw)
    )
    return raw_match, scope_match


def candidate_feature_vector(
    candidate: Mapping[str, Any],
    *,
    anchor_feature: Sequence[float] | None,
    anchor_box: Sequence[float],
    predicted_box: Sequence[float] | None,
    previous_raw_sam_id: int | None,
    previous_native_scope: str | None,
    image_width: int,
    image_height: int,
    candidate_count: int,
    base_target_score: float | None,
) -> np.ndarray:
    """Return the fixed 530-D candidate token used by train and replay."""
    feature, available = _unit_feature(candidate.get("feature"))
    box = candidate.get("box_xyxy", candidate.get("box"))
    if box is None:
        raise ValueError("candidate is missing box_xyxy")
    normalized_box = _norm_box(box, float(image_width), float(image_height))
    reference = predicted_box if predicted_box is not None else anchor_box
    relative = _relative_box(box, reference, float(image_width), float(image_height))
    confidence = float(candidate.get("confidence", candidate.get("presence_score", 0.0)))
    presence = float(candidate.get("presence_score", confidence))
    if not np.isfinite(confidence) or not np.isfinite(presence):
        raise ValueError("candidate confidence/presence is non-finite")
    rank = float(candidate.get("candidate_index", 0)) / float(max(candidate_count - 1, 1))
    raw_continuity, scope_match = _same_raw(candidate, previous_raw_sam_id, previous_native_scope)
    source = str(candidate.get("candidate_source", candidate.get("source_kind", MAIN_B0_CANDIDATE)))
    source_one_hot = np.asarray(
        [float(source == MAIN_B0_CANDIDATE), float(source == TARGET_SESSION_CURRENT_RAW)],
        dtype=np.float32,
    )
    base = 0.0 if base_target_score is None else float(base_target_score)
    if not np.isfinite(base):
        raise ValueError("base target score is non-finite")
    motion = float(box_iou(box, reference))
    scalar = np.asarray(
        [
            available,
            confidence,
            presence,
            rank,
            raw_continuity,
            scope_match,
            np.tanh(base),
            motion,
        ],
        dtype=np.float32,
    )
    result = np.concatenate([feature, normalized_box, relative, scalar[:4], np.asarray([raw_continuity, scope_match], dtype=np.float32), source_one_hot, scalar[6:],], axis=0)
    if result.shape != (CANDIDATE_FEATURE_DIM,) or not np.all(np.isfinite(result)):
        raise RuntimeError(f"decoder candidate feature shape/finite failure: {result.shape}")
    return result.astype(np.float32)


def context_feature_vector(
    *,
    anchor_feature: Sequence[float] | None,
    predicted_box: Sequence[float] | None,
    anchor_box: Sequence[float],
    velocity: Sequence[float] | None,
    previous_raw_sam_id: int | None,
    frame: int,
    event_frame: int,
    trusted_count: int,
    image_width: int,
    image_height: int,
) -> np.ndarray:
    """Return the fixed 522-D target-context token."""
    anchor, anchor_available = _unit_feature(anchor_feature)
    reference_box = predicted_box if predicted_box is not None else anchor_box
    box = _norm_box(reference_box, float(image_width), float(image_height))
    velocity_array = np.zeros(2, dtype=np.float32) if velocity is None else np.asarray(velocity, dtype=np.float32).reshape(-1)
    if velocity_array.size != 2 or not np.all(np.isfinite(velocity_array)):
        raise ValueError("decoder velocity must be finite 2-D")
    # Normalize displacement in pixels to a stable image-relative scale.
    velocity_array = velocity_array / np.asarray([max(float(image_width), 1.0), max(float(image_height), 1.0)], dtype=np.float32)
    horizon = float(max(int(frame) - int(event_frame), 0)) / 100.0
    values = np.concatenate(
        [
            anchor,
            np.asarray([anchor_available], dtype=np.float32),
            box,
            velocity_array,
            np.asarray([float(previous_raw_sam_id is not None), np.clip(horizon, 0.0, 1.0), min(int(trusted_count), 8) / 8.0], dtype=np.float32),
        ],
        axis=0,
    )
    if values.shape != (CONTEXT_FEATURE_DIM,) or not np.all(np.isfinite(values)):
        raise RuntimeError(f"decoder context feature shape/finite failure: {values.shape}")
    return values.astype(np.float32)


__all__ = [
    "CANDIDATE_FEATURE_DIM",
    "CONTEXT_FEATURE_DIM",
    "candidate_feature_vector",
    "context_feature_vector",
]
