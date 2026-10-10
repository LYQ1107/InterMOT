import numpy as np
import pytest
from sam3_intermot.evaluation.pinned_event_trajectory import tracker_arrays, numpy_legacy_types


def output():
    return {"public_id": 100004, "candidate_uid": "uid", "box_xyxy": [1.123456, 2.234567, 1.623456, 4.345678], "confidence": .12345678}


def test_export_rounding_and_public_ids_match_original_MOT_geometry():
    result = tracker_arrays([{"outputs": [output()]}])
    assert result["tracker_ids"][0].tolist() == [100004]
    assert result["tracker_dets"][0].tolist() == [[1.1235, 2.2346, 1., 2.1111]]
    assert result["tracker_confidences"][0].tolist() == [.123457]


def test_duplicate_UID_or_public_ownership_not_accepted():
    with pytest.raises(ValueError):
        tracker_arrays([{"outputs": [output(), output()]}])


def test_numpy_compat_aliases_restore_even_after_failure():
    before = {name: np.__dict__.get(name) for name in ("float", "int", "bool")}
    with pytest.raises(RuntimeError):
        with numpy_legacy_types():
            assert np.float is float and np.int is int
            raise RuntimeError("test")
    assert {name: np.__dict__.get(name) for name in before} == before
