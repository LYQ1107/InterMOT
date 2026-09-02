import numpy as np

from sam3_intermot.backend.output_types import PromptObjectObservation
from sam3_intermot.backend.sam3_backend import Sam3Backend


def _observation(raw_id=17):
    return PromptObjectObservation(
        frame_idx=3,
        sam_object_id=raw_id,
        raw_sam_object_id=raw_id,
        mask=np.ones((2, 2), dtype=bool),
        box_xyxy=np.asarray([1, 2, 8, 12], dtype=float),
        confidence=0.9,
    )


def test_stable_id_binding_does_not_change_raw_axis():
    backend = Sam3Backend(device="cpu")
    backend._sam_to_ext = {17: 9001}
    observation = _observation()

    backend._apply_stable_ids([observation])

    assert observation.sam_object_id == 9001
    assert observation.raw_sam_object_id == 17
    copied = observation.copy()
    assert copied.sam_object_id == 9001
    assert copied.raw_sam_object_id == 17


def test_opt_in_export_exposes_raw_and_adapter_axes():
    backend = Sam3Backend(device="cpu")
    observation = _observation()
    backend._output_cache[3] = [observation]

    legacy = backend.export_frame_candidates(3)
    extended = backend.export_frame_candidates(3, include_raw_provenance=True)

    assert "raw_native_id" not in legacy[0]
    assert legacy[0]["native_tid"] == 17
    assert extended[0]["native_tid"] == 17
    assert extended[0]["raw_native_id"] == 17
    assert extended[0]["adapter_external_id"] == 17
    assert extended[0]["raw_native_id_source"] == "official_out_obj_ids"


def test_extended_export_marks_non_official_human_observation_unavailable():
    backend = Sam3Backend(device="cpu")
    observation = PromptObjectObservation(
        frame_idx=3,
        sam_object_id=9001,
        mask=np.ones((2, 2), dtype=bool),
        box_xyxy=np.asarray([1, 2, 8, 12], dtype=float),
        confidence=1.0,
        source="human_correction",
        is_human_verified=True,
    )
    backend._output_cache[3] = [observation]

    row = backend.export_frame_candidates(3, include_raw_provenance=True)[0]

    assert row["raw_native_id"] is None
    assert row["raw_native_id_source"] == "UNAVAILABLE_NOT_OFFICIAL_OBSERVATION"
    assert row["adapter_external_id"] == 9001


def test_official_shaped_out_obj_ids_survive_stable_binding_and_extended_export():
    backend = Sam3Backend(device="cpu")
    backend._frame_w = 100
    backend._frame_h = 100
    observations = backend._parse_outputs(
        {
            "outputs": {
                "out_obj_ids": np.asarray([17]),
                "out_boxes_xywh": np.asarray([[0.1, 0.2, 0.3, 0.4]]),
                "out_binary_masks": np.ones((1, 10, 10), dtype=bool),
                "out_probs": np.asarray([0.9]),
            }
        },
        frame_idx=3,
        source="automatic_propagation",
    )
    backend._sam_to_ext = {17: 9001}
    backend._apply_stable_ids(observations)
    backend._output_cache[3] = observations
    row = backend.export_frame_candidates(3, include_raw_provenance=True)[0]
    assert row["native_tid"] == 9001
    assert row["raw_native_id"] == 17
    assert row["adapter_external_id"] == 9001
    assert row["raw_native_id_source"] == "official_out_obj_ids"
