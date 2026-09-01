import numpy as np
import pytest

from sam3_intermot.backend.mock_backend import MockBackend
from sam3_intermot.observations.observation import validate_observation


def _fresh_backend():
    b = MockBackend(frame_h=480, frame_w=640, seed=3)
    b.start_video("mock://seq")
    return b


def test_single_object_box_prompt_propagates():
    b = _fresh_backend()
    obs0 = b.add_box(0, 1, [100, 100, 160, 260])
    assert obs0.sam_object_id == 1
    b.propagate(0, 10)
    for f in (1, 5, 10):
        outs = b.get_frame_outputs(f)
        assert any(o.sam_object_id == 1 for o in outs)


def test_two_objects_simultaneously():
    b = _fresh_backend()
    b.add_box(0, 1, [100, 100, 160, 260])
    b.add_box(0, 2, [400, 120, 470, 290])
    b.propagate(0, 5)
    ids = {o.sam_object_id for o in b.get_frame_outputs(5)}
    assert ids == {1, 2}


def test_mid_video_add_third_object():
    b = _fresh_backend()
    b.add_box(0, 1, [100, 100, 160, 260])
    b.add_box(0, 2, [400, 120, 470, 290])
    b.add_box(5, 3, [200, 200, 260, 360])
    b.propagate(0, 10)
    ids5 = {o.sam_object_id for o in b.get_frame_outputs(5)}
    ids10 = {o.sam_object_id for o in b.get_frame_outputs(10)}
    assert 3 in ids5 and 3 in ids10
    assert {1, 2}.issubset(ids10)


def test_mid_video_correct_object():
    b = _fresh_backend()
    b.add_box(0, 1, [100, 100, 160, 260])
    b.propagate(0, 5)
    corrected = b.correct_object(5, 1, box_xyxy=[300, 300, 360, 460])
    assert corrected.source == "human_correction"
    assert corrected.is_human_verified
    b.propagate(5, 7)
    out6 = next(o for o in b.get_frame_outputs(6) if o.sam_object_id == 1)
    assert out6.box_xyxy[0] > 200  # propagation restarted from corrected box


def test_remove_object_does_not_change_others():
    b = _fresh_backend()
    b.add_box(0, 1, [100, 100, 160, 260])
    b.add_box(0, 2, [400, 120, 470, 290])
    b.propagate(0, 10)
    before = {o.sam_object_id: o.box_xyxy.copy() for o in b.get_frame_outputs(10)}
    b.remove_object(2)
    b.propagate(9, 11)
    after = {o.sam_object_id: o.box_xyxy.copy() for o in b.get_frame_outputs(10)}
    assert 2 not in after
    np.testing.assert_allclose(after[1], before[1])


def test_same_inputs_same_seed_repeatable():
    outputs = []
    for _ in range(2):
        b = _fresh_backend()
        b.add_box(0, 1, [100, 100, 160, 260])
        b.propagate(0, 10)
        outputs.append([o.box_xyxy.copy() for o in b.get_frame_outputs(10)])
    np.testing.assert_allclose(outputs[0], outputs[1])


def test_mask_to_box_via_backend():
    b = _fresh_backend()
    mask = np.zeros((480, 640), dtype=bool)
    mask[50:80, 120:200] = True
    obs = b.add_mask(0, 7, mask)
    np.testing.assert_allclose(obs.box_xyxy, [120, 50, 200, 80])


def test_empty_mask_raises_controlled_error():
    b = _fresh_backend()
    with pytest.raises(ValueError):
        b.add_mask(0, 1, np.zeros((480, 640), dtype=bool))


def test_out_of_bounds_box_clipped_and_valid():
    b = _fresh_backend()
    obs = b.add_box(0, 1, [-100, -100, 1000, 1000])
    assert not validate_observation(obs, frame_h=480, frame_w=640)


def test_unique_object_ids_per_frame():
    b = _fresh_backend()
    for oid, box in [(1, [10, 10, 50, 80]), (2, [60, 60, 110, 130]), (3, [200, 200, 260, 300])]:
        b.add_box(0, oid, box)
    b.propagate(0, 3)
    for f in range(4):
        ids = [o.sam_object_id for o in b.get_frame_outputs(f)]
        assert len(ids) == len(set(ids))


def test_close_session_clears_state():
    b = _fresh_backend()
    b.add_box(0, 1, [10, 10, 50, 80])
    b.close()
    assert b._closed
    assert b._objects == {}
    assert b.get_frame_outputs(0) == []


def test_invalid_object_id_correct_raises():
    b = _fresh_backend()
    with pytest.raises(ValueError):
        b.correct_object(0, 999, box_xyxy=[1, 1, 2, 2])
