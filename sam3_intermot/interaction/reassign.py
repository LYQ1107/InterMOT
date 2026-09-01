"""Reassign interaction: local identity transaction between two tracks."""

from sam3_intermot.identity.transaction import Transaction
from sam3_intermot.interaction.actions import (
    HumanInteraction,
    InteractionResult,
    SystemContext,
    summarize_manager,
)


def perform_reassign(
    ctx: SystemContext,
    interaction: HumanInteraction,
) -> InteractionResult:
    frame = interaction.frame_idx
    before = summarize_manager(ctx.manager)
    before_id_count = len({t.mot_track_id for t in ctx.manager.active_tracks()})
    before_lineage_count = len({t.identity_lineage_id for t in ctx.manager.active_tracks()})
    src = ctx.manager.get(interaction.target_track_id)
    dst = ctx.manager.get(interaction.destination_track_id)
    if src is None or dst is None:
        return InteractionResult(
            action_id=interaction.action_id,
            action_type="Reassign",
            frame_idx=frame,
            accepted=False,
            reason="source or destination track is unknown",
            before_summary=before,
            after_summary=before,
        )
    if src.mot_track_id == dst.mot_track_id:
        return InteractionResult(
            action_id=interaction.action_id,
            action_type="Reassign",
            frame_idx=frame,
            accepted=False,
            reason="source and destination are the same track",
            before_summary=before,
            after_summary=before,
        )
    if src.sam_object_id is None:
        return InteractionResult(
            action_id=interaction.action_id,
            action_type="Reassign",
            frame_idx=frame,
            accepted=False,
            reason="source track has no SAM object",
            before_summary=before,
            after_summary=before,
        )
    txn = Transaction(ctx.manager, ctx.lineages)
    try:
        dst_occupied = (
            dst.sam_object_id is not None
            and any(
                obs.sam_object_id == dst.sam_object_id
                for obs in ctx.manager.outputs_for_frame(frame)
            )
        )
        swap = interaction.source == "swap" and ctx.config.enable_atomic_reassign
        if dst_occupied and not swap:
            txn.rollback()
            return InteractionResult(
                action_id=interaction.action_id,
                action_type="Reassign",
                frame_idx=frame,
                accepted=False,
                reason="destination identity occupied this frame",
                before_summary=before,
                after_summary=summarize_manager(ctx.manager),
            )
        src_obs = None
        for obs in ctx.manager.outputs_for_frame(frame):
            if obs.sam_object_id == src.sam_object_id:
                src_obs = obs
                break
        sam_id = src.sam_object_id
        dst_sam_id = dst.sam_object_id
        ctx.manager.unbind_sam_object(src.mot_track_id)
        if swap:
            ctx.manager.unbind_sam_object(dst.mot_track_id)
            ctx.manager.rebind_sam_object(dst.mot_track_id, sam_id, frame)
            ctx.manager.rebind_sam_object(src.mot_track_id, dst_sam_id, frame)
            ctx.manager.remove_output(frame, src.mot_track_id)
            ctx.manager.remove_output(frame, dst.mot_track_id)
        else:
            ctx.manager.rebind_sam_object(dst.mot_track_id, sam_id, frame)
            ctx.manager.remove_output(frame, src.mot_track_id)
            if src_obs is not None:
                ctx.manager.update_track(dst.mot_track_id, frame, src_obs, human_verified=True)
        after_id_count = len({t.mot_track_id for t in ctx.manager.active_tracks()})
        after_lineage_count = len({t.identity_lineage_id for t in ctx.manager.active_tracks()})
        if after_id_count > before_id_count or after_lineage_count > before_lineage_count:
            txn.rollback()
            return InteractionResult(
                action_id=interaction.action_id,
                action_type="Reassign",
                frame_idx=frame,
                accepted=False,
                rolled_back=True,
                reason="reassign increased mot/lineage count",
                before_summary=before,
                after_summary=summarize_manager(ctx.manager),
            )
        after = summarize_manager(ctx.manager)
        ctx.log_transaction(
            {
                "action_id": interaction.action_id,
                "action_type": "Reassign",
                "frame_idx": frame,
                "accepted": True,
                "source_track_id": src.mot_track_id,
                "destination_track_id": dst.mot_track_id,
                "sam_object_id": sam_id,
                "reassign_mode": "SWAP" if swap else "REBIND",
            }
        )
        txn.commit()
        return InteractionResult(
            action_id=interaction.action_id,
            action_type="Reassign",
            frame_idx=frame,
            accepted=True,
            before_summary=before,
            after_summary=after,
        )
    except Exception as exc:
        txn.rollback()
        return InteractionResult(
            action_id=interaction.action_id,
            action_type="Reassign",
            frame_idx=frame,
            accepted=False,
            rolled_back=True,
            reason=f"error: {exc}",
            before_summary=before,
            after_summary=summarize_manager(ctx.manager),
        )
