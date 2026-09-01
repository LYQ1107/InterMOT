import numpy as np
import pytest

from sam3_intermot.observations.mask_to_box import mask_to_box


def test_mask_to_box_returns_xyxy():
    mask = np.zeros((100, 100), dtype=bool)
    mask[10:40, 20:60] = True
    box = mask_to_box(mask)
    np.testing.assert_allclose(box, [20, 10, 60, 40])


def test_mask_to_box_empty_returns_none():
    assert mask_to_box(np.zeros((10, 10), dtype=bool)) is None


def test_mask_to_box_raises_for_3d():
    with pytest.raises(ValueError):
        mask_to_box(np.zeros((2, 2, 2), dtype=bool))
