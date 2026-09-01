"""Rediscovery / handover logic based on past state only."""

from typing import Optional

from sam3_intermot.identity.lineage import IdentityLineageRegistry
from sam3_intermot.tracking.track_manager import TrackManager


def find_handover(
    manager: TrackManager,
    lineages: IdentityLineageRegistry,
    observation,
    frame_idx: int,
    max_gap: int = 45,
) -> Optional[int]:
    """Find a lost lineage to hand a new observation to, or None."""
    return lineages.find_lost_lineage(
        observation=observation,
        frame_idx=frame_idx,
        manager=manager,
        max_gap=max_gap,
    )
