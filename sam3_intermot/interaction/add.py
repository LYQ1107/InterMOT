"""Add interaction: user creates a new trajectory at the current frame."""

from typing import Optional

import numpy as np

from sam3_intermot.identity.handover import find_handover
from sam3_intermot.identity.transaction import Transaction
from sam3_intermot.interaction.actions import (
    HumanInteraction,
    InteractionResult,
    SystemContext,
    summarize_manager,
)
from sam3_intermot.observations.mask_to_box import mask_to_box
from sam3_intermot.tracking.association import box_iou
from sam3_intermot.tracking.track import TrackState


def perform_add(
    ctx: SystemContext,
    interaction: HumanInteraction,
) -> InteractionResult:
    frame = interaction.frame_idx
    before = summarize_manager(ctx.manager)
    box = _resolve_box(interaction)
    if box is None:
        return InteractionResult(
            action_id=interaction.action_id,
            action_type="Add",
            frame_idx=frame,
            accepted=False,
            reason="no usable box/mask/points",
            before_summary=before,
            after_summary=before,
        )
    txn = Transaction(ctx.manager, ctx.lineages)
    try:
        resolution = "new_identity"
        for track in ctx.manager.active_tracks():
            if track.last_box is not None and box_iou(box, track.last_box) >= ctx.config.duplicate_iou_threshold:
                txn.rollback()
                return InteractionResult(
                    action_id=interaction.action_id,
                    action_type="Add",
                    frame_idx=frame,
                    accepted=False,
                    reason=f"duplicate of track {track.mot_track_id}",
                    before_summary=before,
                    after_summary=summarize_manager(ctx.manager),
                )
        # Lineage-aware Add: recover a recently-lost identity instead of
        # creating a new MOT ID.
        recovered_lineage_id = None
        recovered_track_id = None
        if ctx.config.enable_lineage_aware_add:
            dummy = _dummy_observation(frame, box)
            lid = ctx.lineages.find_lost_lineage(
                dummy,
                frame,
                ctx.manager,
                max_gap=ctx.config.max_lost_gap_for_handover,
            )
            if lid is not None:
                lineage = ctx.lineages.get(lid)
                if lineage and lineage.mot_track_ids:
                    recovered_lineage_id = lid
                    recovered_track_id = lineage.mot_track_ids[-1]
                    resolution = "recover_lost_identity"
        lineage_id = interaction.target_lineage_id
        if lineage_id is None and recovered_lineage_id is None:
            lineage_id = find_handover(
                ctx.manager,
                ctx.lineages,
                _dummy_observation(frame, box),
                frame,
                max_gap=ctx.config.max_lost_gap_for_handover,
            )
        if recovered_lineage_id is not None:
            lineage_id = recovered_lineage_id
        lineage = ctx.lineages.get(lineage_id) if lineage_id is not None else None
        if lineage is None:
            lineage = ctx.lineages.create(frame)
            lineage_id = lineage.lineage_id

        sam_id = ctx.allocate_sam_object_id()
        if interaction.mask is not None:
            obs = ctx.backend.add_mask(frame, sam_id, interaction.mask)
        elif interaction.points is not None:
            obs = ctx.backend.add_points(
                frame, sam_id, interaction.points, interaction.point_labels
            )
        else:
            obs = ctx.backend.add_box(frame, sam_id, box)
        if obs.box_xyxy is None or not np.all(np.isfinite(obs.box_xyxy)):
            raise RuntimeError("backend returned invalid observation")
        track = ctx.manager.create_track(
            frame,
            obs,
            lineage_id,
            mot_track_id=recovered_track_id,
        )
        lineage.bind_track(track.mot_track_id)
        if recovered_track_id is not None:
            track.state = TrackState.CONFIRMED
        after = summarize_manager(ctx.manager)
        ctx.log_transaction(
            {
                "action_id": interaction.action_id,
                "action_type": "Add",
                "frame_idx": frame,
                "accepted": True,
                "mot_track_id": track.mot_track_id,
                "sam_object_id": sam_id,
                "lineage_id": lineage_id,
                "add_resolution": resolution,
            }
        )
        txn.commit()
        return InteractionResult(
            action_id=interaction.action_id,
            action_type="Add",
            frame_idx=frame,
            accepted=True,
            new_track_id=track.mot_track_id,
            new_sam_object_id=sam_id,
            reason="add_resolution=" + resolution,
            before_summary=before,
            after_summary=after,
        )
    except Exception as exc:  # roll back on any failure
        txn.rollback()
        return InteractionResult(
            action_id=interaction.action_id,
            action_type="Add",
            frame_idx=frame,
            accepted=False,
            rolled_back=True,
            reason=f"error: {exc}",
            before_summary=before,
            after_summary=summarize_manager(ctx.manager),
        )


def _resolve_box(interaction: HumanInteraction) -> Optional[np.ndarray]:
    if interaction.box_xyxy is not None:
        return np.asarray(interaction.box_xyxy, dtype=float).reshape(-1)
    if interaction.mask is not None:
        return mask_to_box(interaction.mask)
    if interaction.points is not None:
        pts = np.asarray(interaction.points, dtype=float).reshape(-1, 2)
        pad = 5.0
        return np.asarray(
            [pts[:, 0].min() - pad, pts[:, 1].min() - pad,
             pts[:, 0].max() + pad, pts[:, 1].max() + pad],
            dtype=float,
        )
    return None


def _dummy_observation(frame: int, box: np.ndarray):
    from sam3_intermot.backend.output_types import PromptObjectObservation

    return PromptObjectObservation(
        frame_idx=frame,
        sam_object_id=-1,
        mask=np.zeros((1, 1), dtype=bool),
        box_xyxy=box,
        confidence=1.0,
    )
