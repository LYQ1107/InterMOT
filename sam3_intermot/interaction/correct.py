"""Correct interaction: replace current-frame observation, keep identity."""

from sam3_intermot.identity.transaction import Transaction
from sam3_intermot.interaction.actions import (
    HumanInteraction,
    InteractionResult,
    SystemContext,
    summarize_manager,
)


def perform_correct(
    ctx: SystemContext,
    interaction: HumanInteraction,
) -> InteractionResult:
    frame = interaction.frame_idx
    before = summarize_manager(ctx.manager)
    before_ids = {t.mot_track_id for t in ctx.manager.active_tracks()}
    before_lineages = {t.identity_lineage_id for t in ctx.manager.active_tracks()}
    track = ctx.manager.get(interaction.target_track_id)
    if track is None:
        return InteractionResult(
            action_id=interaction.action_id,
            action_type="Correct",
            frame_idx=frame,
            accepted=False,
            reason="unknown target track",
            before_summary=before,
            after_summary=before,
        )
    if track.sam_object_id is None:
        return InteractionResult(
            action_id=interaction.action_id,
            action_type="Correct",
            frame_idx=frame,
            accepted=False,
            reason="track has no SAM object to correct",
            before_summary=before,
            after_summary=before,
        )
    txn = Transaction(ctx.manager, ctx.lineages)
    try:
        obs = ctx.backend.correct_object(
            frame_idx=frame,
            object_id=track.sam_object_id,
            box_xyxy=interaction.box_xyxy,
            points=interaction.points,
            labels=interaction.point_labels,
            mask=interaction.mask,
        )
        ctx.manager.update_track(
            track.mot_track_id,
            frame,
            obs,
            human_verified=True,
        )
        after_ids = {t.mot_track_id for t in ctx.manager.active_tracks()}
        after_lineages = {t.identity_lineage_id for t in ctx.manager.active_tracks()}
        if before_ids != after_ids or before_lineages != after_lineages:
            txn.rollback()
            return InteractionResult(
                action_id=interaction.action_id,
                action_type="Correct",
                frame_idx=frame,
                accepted=False,
                rolled_back=True,
                reason="correct violated identity-count invariants",
                before_summary=before,
                after_summary=summarize_manager(ctx.manager),
            )
        after = summarize_manager(ctx.manager)
        ctx.log_transaction(
            {
                "action_id": interaction.action_id,
                "action_type": "Correct",
                "frame_idx": frame,
                "accepted": True,
                "mot_track_id": track.mot_track_id,
                "sam_object_id": track.sam_object_id,
            }
        )
        txn.commit()
        return InteractionResult(
            action_id=interaction.action_id,
            action_type="Correct",
            frame_idx=frame,
            accepted=True,
            before_summary=before,
            after_summary=after,
        )
    except Exception as exc:
        txn.rollback()
        return InteractionResult(
            action_id=interaction.action_id,
            action_type="Correct",
            frame_idx=frame,
            accepted=False,
            rolled_back=True,
            reason=f"error: {exc}",
            before_summary=before,
            after_summary=summarize_manager(ctx.manager),
        )
