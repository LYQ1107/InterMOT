import pytest
from sam3_intermot.backend.sam3_trim_schema_compat import compatible_trim


def native(self, frame, output, current, memory):
    past = output["non_cond_frame_outputs"][frame - 1]
    replacement = {"pred_masks": past["pred_masks"], "object_score_logits": past["object_score_logits"],
                   "multistep_point_inputs": past["multistep_point_inputs"]}
    output["non_cond_frame_outputs"][frame - 1] = replacement
    return current


def test_success_uses_original_without_schema_change():
    counts = {"native_missing_point_debug_retries": 0}
    mask, score, points = object(), object(), [{"point": "existing"}]
    outputs = {"non_cond_frame_outputs": {0: {"pred_masks": mask, "object_score_logits": score, "multistep_point_inputs": points}}}
    current = object()
    assert compatible_trim(native, counts)(None, 1, outputs, current, True) is current
    assert counts["native_missing_point_debug_retries"] == 0
    assert outputs["non_cond_frame_outputs"][0]["multistep_point_inputs"] is points


def test_missing_debug_only_retry_keeps_tensors_and_other_outputs():
    counts = {"native_missing_point_debug_retries": 0}
    mask, score, unrelated = object(), object(), {"not_trimmed": object()}
    outputs = {"non_cond_frame_outputs": {0: {"pred_masks": mask, "object_score_logits": score}, 5: unrelated}}
    current = object()
    assert compatible_trim(native, counts)(None, 1, outputs, current, True) is current
    assert counts["native_missing_point_debug_retries"] == 1
    assert outputs["non_cond_frame_outputs"][0]["pred_masks"] is mask
    assert outputs["non_cond_frame_outputs"][0]["object_score_logits"] is score
    assert outputs["non_cond_frame_outputs"][0]["multistep_point_inputs"] == [None]
    assert outputs["non_cond_frame_outputs"][5] is unrelated


def test_other_key_errors_not_hidden():
    counts = {"native_missing_point_debug_retries": 0}
    with pytest.raises(KeyError, match="pred_masks"):
        compatible_trim(native, counts)(None, 1, {"non_cond_frame_outputs": {0: {}}}, None, True)
    assert counts["native_missing_point_debug_retries"] == 0
