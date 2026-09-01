import numpy as np

from sam3_intermot.backend.output_types import PromptObjectObservation
from sam3_intermot.evaluation.mot_export import (
    export_mot_file,
    validate_mot_file,
)


def _obs(frame, oid, box):
    return PromptObjectObservation(
        frame_idx=frame,
        sam_object_id=oid,
        mask=np.zeros((10, 10), dtype=bool),
        box_xyxy=box,
        confidence=0.9,
    )


def test_export_and_validate(tmp_path):
    path = tmp_path / "res.txt"
    outputs = {
        0: [(1, _obs(0, 1, [10, 10, 60, 120]))],
        1: [(1, _obs(1, 1, [11, 11, 61, 121])), (2, _obs(1, 2, [200, 200, 260, 320]))],
    }
    export_mot_file(path, outputs)
    assert validate_mot_file(path, num_frames=2, frame_w=640, frame_h=480) == []


def test_validate_flags_duplicate_and_bad_box(tmp_path):
    path = tmp_path / "bad.txt"
    path.write_text(
        "1,1,10,10,0,0,0.9,-1,-1,-1\n"
        "1,1,20,20,30,30,0.9,-1,-1,-1\n",
        encoding="utf-8",
    )
    violations = validate_mot_file(path)
    assert any("non-positive" in v for v in violations)
    assert any("duplicate" in v for v in violations)
