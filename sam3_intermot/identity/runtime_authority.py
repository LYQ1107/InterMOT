"""Active same-run TrackManager authority used by the N72R2 bridge.

The older ``ObjectIdentityRegistry`` intentionally has a geometry fallback
for rediscovery.  That fallback is unsafe for an authority join when two
close candidates coexist in one frame: it can collapse two candidate rows
onto one MOT track.  This adapter uses only the exact adapter-visible SAM ID
within a session; cross-session changes are handled by the explicit handover
transaction module instead.
"""

from __future__ import annotations

from typing import Any

from sam3_intermot.identity.lineage import IdentityLineageRegistry
from sam3_intermot.tracking.track_manager import TrackManager


class ActiveTrackAuthority:
    """Register each session observation without same-frame geometry merging."""

    def __init__(self) -> None:
        self.manager = TrackManager()
        self.lineages = IdentityLineageRegistry()
        self._last_frame: int | None = None
        self._frame_sams: set[int] = set()

    def register(self, frame_idx: int, observation: Any):
        frame = int(frame_idx)
        sam_id = int(observation.sam_object_id)
        if self._last_frame != frame:
            self._last_frame = frame
            self._frame_sams = set()
        if sam_id in self._frame_sams:
            raise ValueError(f"duplicate adapter_external_id in frame: {frame}:{sam_id}")
        self._frame_sams.add(sam_id)
        existing_id = self.manager._sam_to_track.get(sam_id)
        if existing_id is not None:
            return self.manager.update_track(existing_id, frame, observation)
        lineage = self.lineages.create(frame)
        track = self.manager.create_track(frame, observation, lineage.lineage_id)
        lineage.bind_track(track.mot_track_id)
        return track

    def audit(self) -> dict[str, Any]:
        tracks = list(self.manager.tracks.values())
        return {
            "schema_version": "N72R2_ACTIVE_TRACK_AUTHORITY_V1",
            "track_count": len(tracks),
            "lineage_count": len(self.lineages.all()),
            "mot_ids": sorted(int(track.mot_track_id) for track in tracks),
            "active_invariant_violations": self.manager.invariant_violations(),
            "geometry_merge_used_for_same_frame": False,
            "public_authority_source": "TrackManager.final_mot_track_id",
            "runtime_future_gt_used": False,
        }


__all__ = ["ActiveTrackAuthority"]
