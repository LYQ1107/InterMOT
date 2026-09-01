from types import SimpleNamespace

import numpy as np

from scripts.n29_lit_online_replay import _trial_outputs


def test_dataset_and_public_identity_namespaces_are_separate():
    prediction = SimpleNamespace(
        sam_object_id=100000,
        box_xyxy=np.asarray([0.0, 0.0, 10.0, 10.0]),
    )
    result = _trial_outputs(
        {0: [prediction]},
        {0: {0: np.asarray([1.0, 1.0, 9.0, 9.0])}},
        dataset_identity=0,
        public_id=100000,
        start=0,
        end=0,
        require_visible=True,
    )
    assert result["rows"][0]["dataset_identity"] == 0
    assert result["rows"][0]["public_id"] == 100000
    assert result["rows"][0]["box_iou"] > 0.0
    assert result["mean_box_iou_visible"] > 0.0


def test_missing_prediction_is_zero_but_missing_gt_is_null_and_excluded():
    prediction = SimpleNamespace(
        sam_object_id=100000,
        box_xyxy=np.asarray([0.0, 0.0, 10.0, 10.0]),
    )
    result = _trial_outputs(
        {0: [], 1: [prediction]},
        {0: {0: np.asarray([0.0, 0.0, 10.0, 10.0])}, 1: {}},
        dataset_identity=0,
        public_id=100000,
        start=0,
        end=1,
    )
    assert result["rows"][0]["box_iou"] == 0.0
    assert result["rows"][1]["box_iou"] is None
    assert result["visible_frame_count"] == 1
    assert result["missing_prediction_on_visible_count"] == 1
    assert result["absent_gt_frame_count"] == 1
    assert result["mean_box_iou_visible"] == 0.0
