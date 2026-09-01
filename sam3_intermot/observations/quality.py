"""Mask / box quality helpers."""

import numpy as np


def mask_area_ratio(mask: np.ndarray) -> float:
    mask = np.asarray(mask)
    if mask.size == 0:
        return 0.0
    return float(mask.astype(bool).mean())


def is_valid_box(box_xyxy: np.ndarray) -> bool:
    box = np.asarray(box_xyxy, dtype=float).reshape(-1)
    if box.size != 4 or not np.all(np.isfinite(box)):
        return False
    return bool(box[0] < box[2] and box[1] < box[3])
