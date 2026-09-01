"""Causal, action-independent features for the N32 policy selector."""

from __future__ import annotations

import math
from typing import Any, Mapping, Optional, Sequence

import numpy as np


FEATURE_NAMES = (
    "predicted_box_present", "iou_predicted_corrected", "center_dx_normalized", "center_dy_normalized",
    "log_width_ratio", "log_height_ratio", "area_ratio", "corrected_area_normalized", "corrected_aspect_ratio",
    "current_presence", "current_confidence", "official_predicted_iou", "target_state_present", "mapping_valid",
    "frames_since_last_seen", "track_age", "current_mask_area_over_corrected_area", "mask_area_drift_prev1",
    "mask_area_drift_prev3", "mask_area_drift_prev5", "center_velocity_magnitude", "recent_missing_count_prev5",
    "occlusion_recovery_flag", "identity_features_available", "target_positive_similarity",
    "closest_competing_similarity", "identity_margin", "max_neighbor_box_iou", "candidate_count_normalized",
    "association_margin",
)


def _iou(a: Sequence[float], b: Sequence[float]) -> float:
    ax1, ay1, ax2, ay2 = [float(x) for x in a]
    bx1, by1, bx2, by2 = [float(x) for x in b]
    inter = max(0.0, min(ax2, bx2) - max(ax1, bx1)) * max(0.0, min(ay2, by2) - max(ay1, by1))
    aa = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    bb = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = aa + bb - inter
    return inter / union if union > 0 else 0.0


def _area(box: Sequence[float]) -> float:
    return max(1.0, float(box[2]) - float(box[0])) * max(1.0, float(box[3]) - float(box[1]))


def _center(box: Sequence[float]) -> np.ndarray:
    return np.asarray([(float(box[0]) + float(box[2])) / 2.0, (float(box[1]) + float(box[3])) / 2.0], dtype=float)


def _obs_box(obs: Any) -> Optional[np.ndarray]:
    box = getattr(obs, "box_xyxy", None)
    return None if box is None else np.asarray(box, dtype=float).reshape(4)


def _mask_area(obs: Any) -> Optional[float]:
    mask = np.asarray(getattr(obs, "mask", np.zeros((1, 1))), dtype=bool)
    return float(mask.sum()) if mask.ndim == 2 and mask.size > 1 else None


def _safe_log_ratio(numerator: float, denominator: float) -> float:
    return float(np.log(max(numerator, 1.0) / max(denominator, 1.0)))


def build_selector_features(
    *,
    backend: Any,
    correction_frame: int,
    corrected_box: Sequence[float],
    prefix_outputs: Mapping[int, Sequence[Any]],
    public_id: int,
    identity_context: Optional[Mapping[str, float]] = None,
) -> dict[str, Any]:
    """Build one current-only feature vector and a source audit.

    ``prefix_outputs`` ends at the correction frame.  The function never reads
    GT or an episode/sequence identifier; the public ID is only an internal
    lookup key and is not emitted as a feature.
    """

    frame = int(correction_frame)
    corr = np.asarray(corrected_box, dtype=float).reshape(4)
    width = max(1.0, float(getattr(backend, "_frame_w", 1)))
    height = max(1.0, float(getattr(backend, "_frame_h", 1)))
    current = None
    for obs in prefix_outputs.get(frame, []):
        if int(getattr(obs, "sam_object_id", -1)) == int(public_id):
            current = obs
            break
    pred = _obs_box(current)
    pred_present = float(pred is not None)
    if pred is None:
        pred = corr.copy()
    pred_w, pred_h = max(1.0, pred[2] - pred[0]), max(1.0, pred[3] - pred[1])
    corr_w, corr_h = max(1.0, corr[2] - corr[0]), max(1.0, corr[3] - corr[1])
    corr_center = _center(corr)
    pred_center = _center(pred)

    past_items: list[tuple[int, Any]] = []
    for past_frame in sorted(int(x) for x in prefix_outputs if int(x) <= frame):
        for obs in prefix_outputs[past_frame]:
            if int(getattr(obs, "sam_object_id", -1)) == int(public_id):
                past_items.append((past_frame, obs))
    observed_frames = [f for f, _ in past_items]
    last_seen = observed_frames[-1] if observed_frames else frame
    track_age = frame - observed_frames[0] if observed_frames else 0
    frames_since_last = frame - last_seen
    boxes = [_obs_box(obs) for _, obs in past_items]
    boxes = [box for box in boxes if box is not None]
    centers = [_center(box) for box in boxes]
    velocity = 0.0
    if len(centers) >= 2:
        velocity = float(np.linalg.norm(centers[-1] - centers[-2]) / np.sqrt(width * width + height * height))

    area_values = [(_mask_area(obs), _area(_obs_box(obs))) for _, obs in past_items if _obs_box(obs) is not None]
    area_values = [float(mask if mask is not None else box_area) for mask, box_area in area_values]
    corr_area = _area(corr)
    current_area = _mask_area(current) if current is not None else None
    current_area_ratio = 0.0 if current_area is None else float(current_area / max(corr_area, 1.0))

    def drift(n: int) -> float:
        if len(area_values) < 2:
            return 0.0
        tail = area_values[-(n + 1):]
        base = max(tail[0], 1.0)
        return float(np.mean([abs(value / base - 1.0) for value in tail]))

    recent_frames = set(range(max(0, frame - 4), frame + 1))
    recent_seen = {f for f in observed_frames if f in recent_frames}
    recent_missing = float(len(recent_frames - recent_seen))
    target_state_present = float(int(public_id) in _tracker_ids_from_backend(backend))
    mapping = getattr(backend, "_ext_to_sam", {}).get(int(public_id))
    inverse = getattr(backend, "_sam_to_ext", {})
    mapping_valid = float(mapping is not None and inverse.get(int(mapping)) == int(public_id))
    state_context = getattr(backend, "_objects", {}).get(int(public_id), {})
    state_box = np.asarray(state_context.get("box", corr), dtype=float) if isinstance(state_context, Mapping) else corr
    state_track_age = float(max(track_age, frame - int(state_context.get("frame", frame)))) if isinstance(state_context, Mapping) else float(track_age)

    identity = {
        "identity_features_available": 0.0,
        "target_positive_similarity": 0.0,
        "closest_competing_similarity": 0.0,
        "identity_margin": 0.0,
        "max_neighbor_box_iou": 0.0,
        "candidate_count_normalized": 0.0,
        "association_margin": 0.0,
    }
    if identity_context:
        for key in identity:
            if key in identity_context:
                identity[key] = float(identity_context[key])
        identity["identity_features_available"] = float(identity_context.get("identity_features_available", 1.0))

    values = [
        pred_present, _iou(pred, corr), float((pred_center[0] - corr_center[0]) / width), float((pred_center[1] - corr_center[1]) / height),
        _safe_log_ratio(pred_w, corr_w), _safe_log_ratio(pred_h, corr_h), float(_area(pred) / max(corr_area, 1.0)), float(corr_area / (width * height)), float(corr_w / corr_h),
        float(getattr(current, "presence_score", 0.0) or 0.0), float(getattr(current, "confidence", 0.0) or 0.0), float(getattr(current, "presence_score", getattr(current, "confidence", 0.0)) or 0.0),
        target_state_present, mapping_valid, float(frames_since_last), state_track_age, current_area_ratio, drift(1), drift(3), drift(5), velocity, recent_missing,
        float(recent_missing > 0.0),
        identity["identity_features_available"], identity["target_positive_similarity"], identity["closest_competing_similarity"], identity["identity_margin"],
        identity["max_neighbor_box_iou"], identity["candidate_count_normalized"], identity["association_margin"],
    ]
    values = [float(value) if np.isfinite(float(value)) else 0.0 for value in values]
    return {
        "feature_names": list(FEATURE_NAMES),
        "features": values,
        "feature_sources": {
            name: ("past_identity_memory_or_zero_when_unavailable" if name in {
                "identity_features_available", "target_positive_similarity",
                "closest_competing_similarity", "identity_margin",
                "max_neighbor_box_iou", "candidate_count_normalized",
                "association_margin",
            } else "correction_frame_or_past_tracker_state")
            for name in FEATURE_NAMES
        },
        "identity_features_available": bool(identity["identity_features_available"] > 0.5),
        "future_gt_used": False,
        "future_image_used": False,
        "public_id_emitted": False,
        "sequence_id_emitted": False,
    }


def _tracker_ids_from_backend(backend: Any) -> list[int]:
    predictor = getattr(backend, "_predictor", None)
    session_id = getattr(backend, "_session_id", None)
    entry = getattr(predictor, "_all_inference_states", {}).get(session_id) if predictor is not None else None
    state = entry.get("state") if isinstance(entry, Mapping) else None
    ids: list[int] = []
    for tracker_state in (state.get("sam2_inference_states", []) if isinstance(state, Mapping) else []):
        ids.extend(int(value) for value in np.asarray(tracker_state.get("obj_ids", [])).reshape(-1))
    return ids


__all__ = ["FEATURE_NAMES", "build_selector_features"]
