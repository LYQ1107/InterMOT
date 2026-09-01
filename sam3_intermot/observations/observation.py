"""Validation and conversion helpers for observations."""

from typing import Optional

import numpy as np

from sam3_intermot.backend.output_types import PromptObjectObservation


def box_xyxy_to_xywh(box_xyxy: np.ndarray) -> np.ndarray:
    """Convert ``[x1, y1, x2, y2]`` to ``[x, y, w, h]``."""
    box = np.asarray(box_xyxy, dtype=float).reshape(-1)
    if box.size != 4:
        raise ValueError("box must have exactly 4 elements")
    x1, y1, x2, y2 = box
    return np.asarray([x1, y1, x2 - x1, y2 - y1], dtype=float)


def validate_observation(
    obs: PromptObjectObservation,
    *,
    frame_h: Optional[int] = None,
    frame_w: Optional[int] = None,
) -> list:
    """Return a list of violation strings; empty list means valid."""
    violations = []
    box = obs.box_xyxy
    if not np.all(np.isfinite(box)):
        violations.append("non_finite_box")
    if box[0] >= box[2] or box[1] >= box[3]:
        violations.append("non_positive_box")
    if frame_h is not None and frame_w is not None:
        if box[0] < 0 or box[1] < 0 or box[2] > frame_w or box[3] > frame_h:
            violations.append("box_out_of_frame")
    if not np.isfinite(obs.confidence) or not (0.0 <= obs.confidence <= 1.0):
        violations.append("invalid_confidence")
    if obs.mask.ndim != 2:
        violations.append("invalid_mask_ndim")
    if obs.sam_object_id < 0:
        violations.append("negative_sam_object_id")
    return violations
