"""Delete interaction: remove a fake/wrong trajectory safely."""

from sam3_intermot.identity.transaction import Transaction
from sam3_intermot.tracking.track import TrackState
from sam3_intermot.interaction.actions import (
    HumanInteraction,
    InteractionResult,
    SystemContext,
    summarize_manager,
)


def perform_delete(
    ctx: SystemContext,
    interaction: HumanInteraction,
) -> InteractionResult:
    frame = interaction.frame_idx
    before = summarize_manager(ctx.manager)
    track = ctx.manager.get(interaction.target_track_id)
    if track is None:
        return InteractionResult(
            action_id=interaction.action_id,
            action_type="Delete",
            frame_idx=frame,
            accepted=False,
            reason="unknown target track",
            before_summary=before,
            after_summary=before,
        )
    txn = Transaction(ctx.manager, ctx.lineages)
    try:
        sam_id = track.sam_object_id
        if ctx.config.enable_soft_delete:
            track.state = TrackState.QUARANTINED
            track.delete_reason = interaction.source or "human delete"
            if track.sam_object_id is not None:
                ctx.manager._sam_to_track.pop(track.sam_object_id, None)
                ctx.manager._tombstones[track.sam_object_id] = frame
        else:
            ctx.manager.delete_track(
                track.mot_track_id,
                frame,
                reason=interaction.source or "human delete",
            )
        ctx.manager.remove_output(frame, track.mot_track_id)
        if sam_id is not None:
            ctx.backend.remove_object(sam_id)
        after = summarize_manager(ctx.manager)
        ctx.log_transaction(
            {
                "action_id": interaction.action_id,
                "action_type": "Delete",
                "frame_idx": frame,
                "accepted": True,
                "mot_track_id": track.mot_track_id,
                "sam_object_id": sam_id,
                "delete_mode": "SOFT_QUARANTINE" if ctx.config.enable_soft_delete else "HARD_DELETE",
            }
        )
        txn.commit()
        return InteractionResult(
            action_id=interaction.action_id,
            action_type="Delete",
            frame_idx=frame,
            accepted=True,
            before_summary=before,
            after_summary=after,
        )
    except Exception as exc:
        txn.rollback()
        return InteractionResult(
            action_id=interaction.action_id,
            action_type="Delete",
            frame_idx=frame,
            accepted=False,
            rolled_back=True,
            reason=f"error: {exc}",
            before_summary=before,
            after_summary=summarize_manager(ctx.manager),
        )
