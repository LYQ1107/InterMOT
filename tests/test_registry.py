"""Unified object identity registry tests."""

import numpy as np

from sam3_intermot.backend.output_types import PromptObjectObservation
from sam3_intermot.identity.registry import ObjectIdentityRegistry


def _obs(frame, sam_id, box):
    return PromptObjectObservation(
        frame_idx=frame,
        sam_object_id=sam_id,
        mask=np.zeros((1, 1), dtype=bool),
        box_xyxy=np.asarray(box, dtype=float),
        confidence=0.9,
    )


def test_register_auto_creates_and_updates():
    reg = ObjectIdentityRegistry()
    o1 = _obs(0, 0, [10, 10, 60, 80])
    t1 = reg.register_auto_object(0, o1)
    o2 = _obs(1, 0, [12, 12, 62, 82])
    t2 = reg.register_auto_object(1, o2)
    assert t1.mot_track_id == t2.mot_track_id
    assert len(reg.manager.tracks) == 1
    assert reg.invariant_violations() == []


def test_handover_across_window_rebinds():
    reg = ObjectIdentityRegistry()
    reg.register_auto_object(0, _obs(0, 0, [10, 10, 60, 80]))
    reg.unbind_all_for_window()
    t = reg.handover_across_window(200, _obs(200, 3, [15, 15, 65, 85]))
    assert t is not None
    assert t.sam_object_id == 3
    assert reg.invariant_violations() == []


def test_delete_stops_sam_and_mot():
    reg = ObjectIdentityRegistry()
    t = reg.register_auto_object(0, _obs(0, 0, [10, 10, 60, 80]))
    reg.delete(t.mot_track_id, 5, "false track")
    assert reg.lookup_by_sam_object_id(0) is None
    assert reg.manager.get(t.mot_track_id).state.value == "deleted"


def test_lookup_methods():
    reg = ObjectIdentityRegistry()
    t = reg.register_auto_object(0, _obs(0, 7, [10, 10, 60, 80]))
    assert reg.lookup_by_sam_object_id(7).mot_track_id == t.mot_track_id
    assert reg.lookup_by_mot_track_id(t.mot_track_id).sam_object_id == 7
    assert reg.lookup_by_lineage_id(t.identity_lineage_id).mot_track_id == t.mot_track_id
