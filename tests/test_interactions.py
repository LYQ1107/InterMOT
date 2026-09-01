import numpy as np

from sam3_intermot.backend.mock_backend import MockBackend
from sam3_intermot.identity.lineage import IdentityLineageRegistry
from sam3_intermot.interaction.actions import HumanInteraction, SystemContext
from sam3_intermot.interaction.add import perform_add
from sam3_intermot.interaction.correct import perform_correct
from sam3_intermot.interaction.delete import perform_delete
from sam3_intermot.interaction.reassign import perform_reassign
from sam3_intermot.tracking.track import TrackState
from sam3_intermot.tracking.track_manager import TrackManager


def _context():
    backend = MockBackend(frame_h=480, frame_w=640)
    backend.start_video("mock://inter")
    return SystemContext(
        backend=backend,
        manager=TrackManager(),
        lineages=IdentityLineageRegistry(),
    )


def test_add_creates_new_identity():
    ctx = _context()
    result = perform_add(
        ctx,
        HumanInteraction(action_id="a1", frame_idx=0, action_type="Add", box_xyxy=[10, 10, 60, 120]),
    )
    assert result.accepted
    assert result.new_track_id is not None
    track = ctx.manager.get(result.new_track_id)
    assert track.state == TrackState.TENTATIVE
    assert ctx.manager.outputs_for_frame(0)
    assert len(ctx.transaction_log) == 1


def test_add_duplicate_rejected():
    ctx = _context()
    first = perform_add(
        ctx,
        HumanInteraction(action_id="a1", frame_idx=0, action_type="Add", box_xyxy=[10, 10, 60, 120]),
    )
    assert first.accepted
    second = perform_add(
        ctx,
        HumanInteraction(action_id="a2", frame_idx=0, action_type="Add", box_xyxy=[15, 15, 65, 125]),
    )
    assert not second.accepted
    assert "duplicate" in second.reason


def test_correct_keeps_same_identity():
    ctx = _context()
    add = perform_add(
        ctx,
        HumanInteraction(action_id="a1", frame_idx=0, action_type="Add", box_xyxy=[10, 10, 60, 120]),
    )
    tid = add.new_track_id
    lineage_before = ctx.manager.get(tid).identity_lineage_id
    correct = perform_correct(
        ctx,
        HumanInteraction(
            action_id="c1",
            frame_idx=5,
            action_type="Correct",
            target_track_id=tid,
            box_xyxy=[100, 100, 160, 220],
        ),
    )
    assert correct.accepted
    track = ctx.manager.get(tid)
    assert track.identity_lineage_id == lineage_before
    assert track.last_human_verified_frame == 5
    assert track.last_box[0] == 100


def test_reassign_no_conflict():
    ctx = _context()
    a = perform_add(ctx, HumanInteraction(action_id="a1", frame_idx=0, action_type="Add", box_xyxy=[10, 10, 60, 120]))
    b = perform_add(ctx, HumanInteraction(action_id="a2", frame_idx=0, action_type="Add", box_xyxy=[200, 200, 260, 320]))
    src_id, dst_id = a.new_track_id, b.new_track_id
    src_sam = ctx.manager.get(src_id).sam_object_id
    # At frame 5 neither track has produced an output yet, so the destination
    # identity is not occupied in that frame -> reassign is conflict-free.
    result = perform_reassign(
        ctx,
        HumanInteraction(
            action_id="r1",
            frame_idx=5,
            action_type="Reassign",
            target_track_id=src_id,
            destination_track_id=dst_id,
        ),
    )
    assert result.accepted
    assert ctx.manager.get(dst_id).sam_object_id == src_sam
    assert ctx.manager.get(src_id).sam_object_id is None


def test_reassign_conflict_rejected():
    ctx = _context()
    a = perform_add(ctx, HumanInteraction(action_id="a1", frame_idx=0, action_type="Add", box_xyxy=[10, 10, 60, 120]))
    b = perform_add(ctx, HumanInteraction(action_id="a2", frame_idx=0, action_type="Add", box_xyxy=[200, 200, 260, 320]))
    # force destination to output at frame 0 with its own sam object
    ctx.manager.update_track(
        b.new_track_id,
        0,
        ctx.backend.add_box(0, ctx.manager.get(b.new_track_id).sam_object_id, [200, 200, 260, 320]),
    )
    result = perform_reassign(
        ctx,
        HumanInteraction(
            action_id="r1",
            frame_idx=0,
            action_type="Reassign",
            target_track_id=a.new_track_id,
            destination_track_id=b.new_track_id,
        ),
    )
    assert not result.accepted
    assert "occupied" in result.reason


def test_delete_fake_track():
    ctx = _context()
    a = perform_add(ctx, HumanInteraction(action_id="a1", frame_idx=0, action_type="Add", box_xyxy=[10, 10, 60, 120]))
    tid = a.new_track_id
    sam_id = ctx.manager.get(tid).sam_object_id
    result = perform_delete(
        ctx,
        HumanInteraction(action_id="d1", frame_idx=0, action_type="Delete", target_track_id=tid),
    )
    assert result.accepted
    assert ctx.manager.get(tid).state == TrackState.DELETED
    assert ctx.manager.outputs_for_frame(0) == []
    assert all(o.sam_object_id != sam_id for f in range(3) for o in ctx.backend.get_frame_outputs(f))


def test_delete_does_not_affect_other_objects():
    ctx = _context()
    a = perform_add(ctx, HumanInteraction(action_id="a1", frame_idx=0, action_type="Add", box_xyxy=[10, 10, 60, 120]))
    b = perform_add(ctx, HumanInteraction(action_id="a2", frame_idx=0, action_type="Add", box_xyxy=[200, 200, 260, 320]))
    ctx.backend.propagate(0, 3)
    before_b = next(o for o in ctx.backend.get_frame_outputs(3) if o.sam_object_id == ctx.manager.get(b.new_track_id).sam_object_id)
    perform_delete(ctx, HumanInteraction(action_id="d1", frame_idx=3, action_type="Delete", target_track_id=a.new_track_id))
    ctx.backend.propagate(3, 4)
    after_b = next(o for o in ctx.backend.get_frame_outputs(4) if o.sam_object_id == ctx.manager.get(b.new_track_id).sam_object_id)
    assert after_b is not None
    assert before_b.box_xyxy[0] > 0
