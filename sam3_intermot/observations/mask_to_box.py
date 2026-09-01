"""Convert binary masks to axis-aligned bounding boxes."""

from typing import Optional

import numpy as np


def mask_to_box(mask: np.ndarray, min_area: int = 1) -> Optional[np.ndarray]:
    """Return ``[x1, y1, x2, y2]`` for a binary mask, or ``None`` if empty.

    An empty mask returns ``None`` rather than raising, so upper layers can
    treat it as "no usable observation".
    """
    mask = np.asarray(mask)
    if mask.ndim != 2:
        raise ValueError("mask must be a 2D array")
    ys, xs = np.nonzero(mask)
    if xs.size == 0 or xs.size < min_area:
        return None
    return np.asarray(
        [xs.min(), ys.min(), xs.max() + 1, ys.max() + 1], dtype=float
    )
