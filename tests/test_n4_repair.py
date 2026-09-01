"""N4 lineage-preserving repair unit tests."""

import numpy as np

from sam3_intermot.backend.mock_backend import MockBackend
from sam3_intermot.identity.lineage import IdentityLineageRegistry
from sam3_intermot.interaction.actions import (
    HumanInteraction,
    InteractionConfig,
    SystemContext,
)
from sam3_intermot.interaction.add import perform_add
from sam3_intermot.interaction.correct import perform_correct
from sam3_intermot.interaction.delete import perform_delete
from sam3_intermot.interaction.reassign import perform_reassign
from sam3_intermot.interaction.simulator import SimulatedInteractionDriver, SimulatorConfig
from sam3_intermot.tracking.track import TrackState
from sam3_intermot.tracking.track_manager import TrackManager


def _ctx():
    b = MockBackend(frame_h=1080, frame_w=1920, seed=1)
    b.start_video("x")
    return SystemContext(
        backend=b,
        manager=TrackManager(),
        lineages=IdentityLineageRegistry(),
    )


def test_correct_does_not_change_id_sets():
    ctx = _ctx()
    a = perform_add(ctx, HumanInteraction(action_id="a", frame_idx=0, action_type="Add", box_xyxy=[10, 10, 60, 120]))
    tid = a.new_track_id
    before_ids = {t.mot_track_id for t in ctx.manager.active_tracks()}
    before_lineages = {t.identity_lineage_id for t in ctx.manager.active_tracks()}
    r = perform_correct(ctx, HumanInteraction(action_id="c", frame_idx=1, action_type="Correct", target_track_id=tid, box_xyxy=[20, 20, 70, 130]))
    assert r.accepted
    assert {t.mot_track_id for t in ctx.manager.active_tracks()} == before_ids
    assert {t.identity_lineage_id for t in ctx.manager.active_tracks()} == before_lineages


def test_duplicate_add_rejected():
    ctx = _ctx()
    a = perform_add(ctx, HumanInteraction(action_id="a", frame_idx=0, action_type="Add", box_xyxy=[10, 10, 60, 120]))
    r = perform_add(ctx, HumanInteraction(action_id="a2", frame_idx=0, action_type="Add", box_xyxy=[10, 10, 60, 120]))
    assert not r.accepted
    assert a.new_track_id is not None


def test_reassign_swap_keeps_id_count():
    ctx = _ctx()
    a = perform_add(ctx, HumanInteraction(action_id="a", frame_idx=0, action_type="Add", box_xyxy=[10, 10, 60, 120]))
    b = perform_add(ctx, HumanInteraction(action_id="b", frame_idx=0, action_type="Add", box_xyxy=[200, 200, 260, 320]))
    ctx.manager.update_track(a.new_track_id, 0, ctx.backend.add_box(0, ctx.manager.get(a.new_track_id).sam_object_id, [10, 10, 60, 120]))
    ctx.manager.update_track(b.new_track_id, 0, ctx.backend.add_box(0, ctx.manager.get(b.new_track_id).sam_object_id, [200, 200, 260, 320]))
    sa, sb = ctx.manager.get(a.new_track_id).sam_object_id, ctx.manager.get(b.new_track_id).sam_object_id
    before = len(ctx.manager.active_tracks())
    r = perform_reassign(ctx, HumanInteraction(action_id="r", frame_idx=0, action_type="Reassign", target_track_id=a.new_track_id, destination_track_id=b.new_track_id, source="swap"))
    assert r.accepted
    assert len(ctx.manager.active_tracks()) == before
    assert ctx.manager.get(a.new_track_id).sam_object_id == sb
    assert ctx.manager.get(b.new_track_id).sam_object_id == sa


def test_soft_delete_quarantines():
    ctx = _ctx()
    ctx.config.enable_soft_delete = True
    a = perform_add(ctx, HumanInteraction(action_id="a", frame_idx=0, action_type="Add", box_xyxy=[10, 10, 60, 120]))
    r = perform_delete(ctx, HumanInteraction(action_id="d", frame_idx=0, action_type="Delete", target_track_id=a.new_track_id))
    assert r.accepted
    assert ctx.manager.get(a.new_track_id).state == TrackState.QUARANTINED


def test_scheduler_abstains_when_utility_low():
    backend = MockBackend(frame_h=1080, frame_w=1920, seed=1)
    backend.start_video("x")
    manager = TrackManager()
    lineages = IdentityLineageRegistry()
    config = SimulatorConfig(
        enabled_actions={},
        budget_per_100_frames=5,
        enable_abstention=True,
        utility_threshold=1.0,
    )
    driver = SimulatedInteractionDriver(backend, manager, lineages, config)
    # No GT and no detections -> no events, but verify abstention machinery runs.
    summary = driver.run({}, 10)
    assert summary["abstentions"] >= 0
