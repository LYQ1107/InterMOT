import pytest

from sam3_intermot.evaluation.mot_detection_audit import audit_unchanged_detections


REFERENCE = (
    "1,1,10.0000,20.0000,30.0000,40.0000,0.900000,-1,-1,-1\n"
    "1,2,50.0000,20.0000,30.0000,40.0000,0.800000,-1,-1,-1\n"
    "2,1,11.0000,20.0000,30.0000,40.0000,0.900000,-1,-1,-1\n"
)


def test_different_identity_and_row_order_preserve_actual_detection_input():
    reordered = (
        "2,200,11,20,30,40,.9,-1,-1,-1\n"
        "1,200,50,20,30,40,.8,-1,-1,-1\n"
        "1,100,10,20,30,40,.9,-1,-1,-1\n"
    )
    actual = audit_unchanged_detections(REFERENCE, reordered)
    unchanged = audit_unchanged_detections(REFERENCE, REFERENCE)
    assert actual["canonical_multiset_sha256"] == unchanged["canonical_multiset_sha256"]
    assert actual["trajectory_rows"] == 3 and actual["frames_with_detections"] == 2
    assert not actual["derived_TrackEval_detection_metrics_required_equal"]


@pytest.mark.parametrize("changed", [
    REFERENCE.replace("10.0000", "10.0001"),
    REFERENCE.replace("0.900000", "0.900001", 1),
    REFERENCE.replace("2,1,11.0000", "3,1,11.0000"),
    "\n".join(REFERENCE.splitlines()[:-1]) + "\n",
    REFERENCE + "1,3,10,20,30,40,.9,-1,-1,-1\n",
])
def test_box_confidence_frame_or_multiplicity_change_is_not_identity_only(changed):
    with pytest.raises(ValueError, match="multiset changed"):
        audit_unchanged_detections(REFERENCE, changed)


@pytest.mark.parametrize("invalid", [
    REFERENCE.replace("1,2,50", "1,1,50"),
    REFERENCE.replace("10.0000", "nan"),
    REFERENCE.replace("30.0000", "0", 1),
    REFERENCE.replace("2,1,11", "2.5,1,11"),
])
def test_invalid_export_is_rejected_before_evaluation(invalid):
    with pytest.raises(ValueError):
        audit_unchanged_detections(REFERENCE, invalid)


def test_identical_geometry_with_distinct_ids_keeps_duplicate_detections():
    duplicated = "1,1,10,20,30,40,.9,-1,-1,-1\n1,2,10,20,30,40,.9,-1,-1,-1\n"
    renamed = duplicated.replace("1,2,10", "1,7,10")
    result = audit_unchanged_detections(duplicated, renamed)
    assert result["trajectory_rows"] == 2
