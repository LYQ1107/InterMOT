"""Unit tests for N5 continuous observer protocol."""

import numpy as np
import pytest

from sam3_intermot.backend.mock_backend import MockBackend
from sam3_intermot.identity.lineage import IdentityLineageRegistry
from sam3_intermot.identity.registry import ObjectIdentityRegistry
from sam3_intermot.interaction.actions import SystemContext
from sam3_intermot.interaction.continuous_observer import (
    CommandType,
    ContinuousObserverDriver,
    GTFrameAccessor,
    N5Config,
    P1OfflineDriver,
    _hungarian_max,
    match_boxes,
    perform_atomic_swap,
    perform_authoritative_reassign,
    perform_recover_identity,
    read_mot_rows,
)
from sam3_intermot.interaction.simulator import GTFrame
from sam3_intermot.tracking.track import TrackState
from sam3_intermot.tracking.track_manager import TrackManager


def _box(x1, y1, x2, y2):
    return np.asarray([x1, y1, x2, y2], dtype=float)


def _gt(*pairs):
    g = GTFrame()
    for gid, box in pairs:
        g.gt_ids.append(gid)
        g.boxes.append(box)
    return g


def test_hungarian_basic():
    scores = np.array([[0.9, 0.1], [0.1, 0.8]])
    assign = _hungarian_max(scores)
    assert list(assign) == [0, 1]
    scores = np.array([[0.0, 0.6], [0.6, 0.0]])
    assign = _hungarian_max(scores)
    assert list(assign) == [1, 0]


def test_match_boxes_threshold():
    gt = [_box(0, 0, 10, 10), _box(30, 0, 40, 10)]
    pre = [_box(0, 0, 10, 10), _box(90, 0, 100, 10)]
    matches = match_boxes(gt, pre, 0.5)
    assert len(matches) == 1
    assert matches[0][0] == 0 and matches[0][1] == 0


def _make_driver(backend, manager, lineages, gt_frames):
    registry = ObjectIdentityRegistry(manager, lineages)
    cfg = N5Config(protocol="p3", budget=0, stateful=True)
    driver = ContinuousObserverDriver(
        backend, manager, lineages, registry, cfg, 5, gt_frames, sequence="test"
    )
    return driver


def _register_track(manager, lineages, tid, sam_id, box, lineage_id=None):
    from sam3_intermot.backend.output_types import PromptObjectObservation

    lineage = lineages.get(lineage_id) if lineage_id else None
    if lineage is None:
        lineage = lineages.create(0)
    obs = PromptObjectObservation(
        frame_idx=0,
        sam_object_id=sam_id,
        mask=np.zeros((1, 1), dtype=bool),
        box_xyxy=box,
        confidence=1.0,
    )
    track = manager.create_track(0, obs, lineage.lineage_id, mot_track_id=tid)
    lineage.bind_track(tid)
    return track


def test_command_generation_swap():
    backend = MockBackend()
    backend.start_video("mock")
    manager = TrackManager()
    lineages = IdentityLineageRegistry()
    _register_track(manager, lineages, 1, 101, _box(0, 0, 10, 10))
    _register_track(manager, lineages, 2, 102, _box(20, 0, 30, 10))
    gt = {0: _gt((1, _box(0, 0, 10, 10)), (2, _box(20, 0, 30, 10)))}
    driver = _make_driver(backend, manager, lineages, gt)
    driver.track_user[1] = 2
    driver.track_user[2] = 1
    driver.user_track[1] = 2
    driver.user_track[2] = 1
    driver._user_gt_ids[1] = 1
    driver._user_gt_ids[2] = 2
    driver.user_seen_frame = {1: 0, 2: 0}
    driver.gt_access = GTFrameAccessor(gt)
    driver.gt_access.begin_prediction(0)
    driver.gt_access.mark_prediction_done()
    pre = manager.outputs_for_frame(0)
    commands = driver._generate_commands(0, pre, gt[0])
    swaps = [c for c in commands if c.command_type == CommandType.ATOMIC_ID_SWAP]
    assert len(swaps) == 1
    assert swaps[0].target_track_id in (1, 2)
    assert swaps[0].other_track_id in (1, 2)


def test_command_generation_miss_new_and_recover():
    backend = MockBackend()
    backend.start_video("mock")
    manager = TrackManager()
    lineages = IdentityLineageRegistry()
    _register_track(manager, lineages, 1, 101, _box(0, 0, 10, 10))
    gt = {
        0: _gt((1, _box(0, 0, 10, 10)), (2, _box(20, 0, 30, 10))),
        1: _gt((2, _box(24, 0, 34, 10))),
    }
    driver = _make_driver(backend, manager, lineages, gt)
    driver.gt_access = GTFrameAccessor(gt)
    # Frame 0: track 1 learns gt1; gt2 is new and missed -> ADD_NEW.
    driver.gt_access.begin_prediction(0)
    driver.gt_access.mark_prediction_done()
    commands = driver._generate_commands(0, manager.outputs_for_frame(0), gt[0])
    adds = [c for c in commands if c.command_type == CommandType.ADD_NEW_IDENTITY]
    assert len(adds) == 1
    assert adds[0].is_first_appearance
    uid2 = adds[0].user_identity_id
    # Frame 1: gt2 reappears (seen before) but track 1 no longer matches -> RECOVER.
    driver.user_track[uid2] = 7
    driver.user_lineage[uid2] = 77
    driver.user_seen_frame[uid2] = 0
    driver._user_gt_ids[uid2] = 2
    driver.gt_access.begin_prediction(1)
    driver.gt_access.mark_prediction_done()
    commands = driver._generate_commands(1, [], gt[1])
    recovers = [c for c in commands if c.command_type == CommandType.RECOVER_IDENTITY]
    assert len(recovers) == 1
    assert recovers[0].is_recovery
    assert recovers[0].user_identity_id == uid2


def test_authoritative_handlers_no_new_ids():
    backend = MockBackend()
    backend.start_video("mock")
    manager = TrackManager()
    lineages = IdentityLineageRegistry()
    _register_track(manager, lineages, 1, 101, _box(0, 0, 10, 10))
    _register_track(manager, lineages, 2, 102, _box(20, 0, 30, 10))
    ctx = SystemContext(backend=backend, manager=manager, lineages=lineages)
    ctx.config.enable_soft_delete = False
    before_ids = {t.mot_track_id for t in manager.active_tracks()}
    before_lineages = {t.identity_lineage_id for t in manager.active_tracks()}

    swap = perform_atomic_swap(ctx, 0, 1, 2)
    assert swap.accepted
    assert {t.mot_track_id for t in manager.active_tracks()} == before_ids
    assert {t.identity_lineage_id for t in manager.active_tracks()} == before_lineages
    assert manager.get(1).sam_object_id == 102
    assert manager.get(2).sam_object_id == 101
    assert manager.invariant_violations() == []

    reassign = perform_authoritative_reassign(ctx, 1, 1, 2)
    assert reassign.accepted
    assert manager.get(2).sam_object_id == 102
    assert manager.get(1).sam_object_id is None
    assert {t.mot_track_id for t in manager.active_tracks()} == before_ids

    track2_lineage = manager.get(2).identity_lineage_id
    manager.mark_lost(2, 2)
    recover = perform_recover_identity(
        ctx, 2, 2, track2_lineage, _box(24, 0, 34, 10)
    )
    assert recover.accepted
    assert manager.get(2).state == TrackState.RECOVERED
    assert {t.mot_track_id for t in manager.active_tracks()} == before_ids
    assert {t.identity_lineage_id for t in manager.active_tracks()} == before_lineages
    assert manager.invariant_violations() == []


def test_p1_offline_six_error_classes():
    pre = {
        0: [(101, _box(0, 0, 10, 10)), (102, _box(20, 0, 30, 10))],
        1: [(101, _box(20, 0, 30, 10)), (102, _box(0, 0, 10, 10))],
        2: [(102, _box(20, 0, 30, 10)), (103, _box(80, 0, 90, 10))],
        3: [(101, _box(0, 0, 10, 10))],
        4: [(101, _box(0, 0, 10, 10))],
    }
    gt = {
        0: _gt((1, _box(0, 0, 10, 10)), (2, _box(20, 0, 30, 10))),
        1: _gt((1, _box(0, 0, 10, 10)), (2, _box(20, 0, 30, 10))),
        2: _gt((1, _box(0, 0, 10, 10))),
        3: _gt((2, _box(40, 0, 50, 10)), (3, _box(60, 0, 70, 10))),
        4: _gt((1, _box(3, 0, 13, 10))),
    }
    driver = P1OfflineDriver(pre, gt, sequence="test", num_frames=5)
    post = driver.run()
    types = {e["action_type"] for e in driver.events}
    # Frame 0 correct: no identity errors.
    assert types == {
        CommandType.RECOVER_IDENTITY,
        CommandType.ADD_NEW_IDENTITY,
        CommandType.ATOMIC_ID_SWAP,
        CommandType.AUTHORITATIVE_DELETE,
        CommandType.AUTHORITATIVE_CORRECT,
    }
    # Frame 1 swap: post rows at correct locations carry correct identity ids.
    frame1 = {tid: box for tid, box, _ in post[1]}
    pid1 = driver.post_id_for_gt[1]
    pid2 = driver.post_id_for_gt[2]
    assert pid1 in frame1 and pid2 in frame1
    assert np.allclose(frame1[pid1], _box(0, 0, 10, 10))
    assert np.allclose(frame1[pid2], _box(20, 0, 30, 10))
    # Frame 2: unmatched 103 deleted; gt1 recovered as MISS_EXISTING? gt1 seen before.
    frame2 = {tid: box for tid, box, _ in post[2]}
    assert 103 not in frame2
    assert pid1 in frame2
    # Frame 3: gt2 seen before -> recover; gt3 new -> add.
    frame3 = {tid: box for tid, box, _ in post[3]}
    assert driver.post_id_for_gt[2] in frame3
    assert driver.post_id_for_gt[3] in frame3
    # Frame 4: localization correction uses GT box.
    frame4 = {tid: box for tid, box, _ in post[4]}
    assert np.allclose(frame4[pid1], _box(3, 0, 13, 10))
    # No duplicate post ids per frame.
    for f in post:
        ids = [tid for tid, _, _ in post[f]]
        assert len(ids) == len(set(ids))


def test_read_mot_rows(tmp_path):
    p = tmp_path / "x.txt"
    p.write_text("1,7,0.00,0.00,10.00,10.00,1.000,-1,-1,-1\n", encoding="utf-8")
    rows = read_mot_rows(p)
    assert rows[0][0][0] == 7
    assert np.allclose(rows[0][0][1], [0, 0, 10, 10])
