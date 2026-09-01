import numpy as np
import pytest

from sam3_intermot.backend.mock_backend import MockBackend
from sam3_intermot.identity.lineage import IdentityLineageRegistry
from sam3_intermot.tracking.lifecycle import LifecycleConfig
from sam3_intermot.tracking.track import TrackState
from sam3_intermot.tracking.track_manager import TrackManager


def _obs(b, frame, oid, box):
    return b.add_box(frame, oid, box)


def _manager(cfg=None):
    return TrackManager(cfg)


def test_create_confirm_lost_terminate():
    b = MockBackend()
    b.start_video("x")
    cfg = LifecycleConfig(confirm_after_frames=2, lost_after_frames=2, max_lost_frames=3)
    m = _manager(cfg)
    lineages = IdentityLineageRegistry()
    lineage = lineages.create(0)
    obs = _obs(b, 0, 1, [10, 10, 50, 80])
    track = m.create_track(0, obs, lineage.lineage_id)
    assert track.state == TrackState.TENTATIVE
    for f in (1, 2):
        m.update_track(track.mot_track_id, f, _obs(b, f, 1, [11, 11, 51, 81]))
    assert track.state == TrackState.CONFIRMED
    m.mark_missed(track.mot_track_id, 3)
    m.mark_missed(track.mot_track_id, 4)
    assert track.state == TrackState.LOST
    m.mark_missed(track.mot_track_id, 5)
    assert track.state == TrackState.TERMINATED


def test_duplicate_sam_binding_rejected():
    b = MockBackend()
    b.start_video("x")
    m = _manager()
    l1 = IdentityLineageRegistry().create(0)
    l2 = IdentityLineageRegistry().create(0)
    m.create_track(0, _obs(b, 0, 1, [10, 10, 50, 80]), l1.lineage_id)
    with pytest.raises(ValueError):
        m.create_track(0, _obs(b, 0, 1, [100, 100, 150, 180]), l2.lineage_id)


def test_same_frame_unique_track_ids():
    b = MockBackend()
    b.start_video("x")
    m = _manager()
    reg = IdentityLineageRegistry()
    l1, l2 = reg.create(0), reg.create(0)
    m.create_track(0, _obs(b, 0, 1, [10, 10, 50, 80]), l1.lineage_id)
    m.create_track(0, _obs(b, 0, 2, [100, 100, 150, 180]), l2.lineage_id)
    ids = [o.sam_object_id for o in m.outputs_for_frame(0)]
    assert len(ids) == len(set(ids))
    assert m.invariant_violations() == []


def test_sam_mot_lineage_ids_are_distinct():
    b = MockBackend()
    b.start_video("x")
    m = _manager()
    reg = IdentityLineageRegistry()
    lineage = reg.create(0)
    track = m.create_track(0, _obs(b, 0, 7, [10, 10, 50, 80]), lineage.lineage_id)
    # Numerical values may collide by coincidence; the requirement is that the
    # three ID namespaces are managed separately.
    assert track.mot_track_id != track.sam_object_id
    assert track.identity_lineage_id != track.sam_object_id


def test_delete_tombstone_cooldown():
    b = MockBackend()
    b.start_video("x")
    m = _manager()
    reg = IdentityLineageRegistry()
    lineage = reg.create(0)
    track = m.create_track(0, _obs(b, 0, 5, [10, 10, 50, 80]), lineage.lineage_id)
    m.delete_track(track.mot_track_id, 0, "fake")
    with pytest.raises(ValueError):
        m.create_track(0, _obs(b, 0, 5, [10, 10, 50, 80]), lineage.lineage_id)
    # after cooldown (31 frames later) the same sam id may be reused
    m.create_track(31, _obs(b, 31, 5, [10, 10, 50, 80]), lineage.lineage_id)
