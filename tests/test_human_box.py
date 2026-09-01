"""Human-verified box semantics (no GPU required)."""

import numpy as np

from sam3_intermot.backend.sam3_backend import Sam3Backend


def _backend():
    b = Sam3Backend(checkpoint_path=None)
    b._frame_w = 1920
    b._frame_h = 1080
    return b


def test_large_human_add_box_not_center_shrunk():
    b = _backend()
    box = np.asarray([50.0, 40.0, 300.0, 360.0])
    clipped = b._clip_box(box)
    assert np.allclose(clipped, box)
    assert clipped[2] - clipped[0] == 250.0
    assert clipped[3] - clipped[1] == 320.0


def test_automatic_oversized_detection_is_safely_scaled():
    b = _backend()
    box = np.asarray([100.0, 100.0, 800.0, 900.0])
    auto = b._sanitize_box(box)
    assert auto[2] - auto[0] < 800.0 - 100.0
    assert auto[3] - auto[1] < 900.0 - 100.0
    assert abs((auto[0] + auto[2]) / 2 - 450.0) < 1e-3


def test_human_prompt_variants_keep_original_first():
    b = _backend()
    box = np.asarray([50.0, 40.0, 300.0, 360.0])
    variants = b._human_prompt_variants(box)
    assert np.allclose(variants[0], box)
    assert len(variants) >= 1


def test_human_current_frame_output_equals_clipped_user_box():
    b = _backend()
    box = np.asarray([-20.0, 30.0, 2000.0, 500.0])
    clipped = b._clip_box(box)
    obs = b._human_observation(7, 42, clipped, "human_add")
    assert obs.frame_idx == 7
    assert obs.sam_object_id == 42
    assert obs.is_human_verified
    assert np.allclose(obs.box_xyxy, clipped)
    assert clipped[0] >= 0 and clipped[2] <= 1920
