"""Track Manager: stable MOT IDs, lifecycle and identity-mapping invariants."""

from copy import deepcopy
from typing import Dict, List, Optional

import numpy as np

from sam3_intermot.backend.output_types import PromptObjectObservation
from sam3_intermot.tracking.lifecycle import LifecycleConfig, update_state
from sam3_intermot.tracking.track import Track, TrackState


class TrackManager:
    """Owns MOT track IDs and enforces sam_object_id <-> mot_track_id rules."""

    def __init__(self, lifecycle: Optional[LifecycleConfig] = None) -> None:
        self.lifecycle = lifecycle or LifecycleConfig()
        self._tracks: Dict[int, Track] = {}
        self._sam_to_track: Dict[int, int] = {}
        self._tombstones: Dict[int, int] = {}  # sam_object_id -> deleted at frame
        self._next_track_id = 1
        self._outputs: Dict[int, Dict[int, PromptObjectObservation]] = {}

    # ------------------------------------------------------------------
    @property
    def tracks(self) -> Dict[int, Track]:
        return self._tracks

    def get(self, track_id: int) -> Optional[Track]:
        return self._tracks.get(track_id)

    def active_tracks(self) -> List[Track]:
        return [
            t
            for t in self._tracks.values()
            if t.state
            in (
                TrackState.TENTATIVE,
                TrackState.CONFIRMED,
                TrackState.LOST,
                TrackState.QUARANTINED,
                TrackState.RECOVERED,
            )
        ]

    def next_track_id(self) -> int:
        tid = self._next_track_id
        self._next_track_id += 1
        return tid

    # ------------------------------------------------------------------
    def create_track(
        self,
        frame_idx: int,
        obs: PromptObjectObservation,
        lineage_id: int,
        mot_track_id: Optional[int] = None,
    ) -> Track:
        """Create a tentative track from an observation.

        Duplicate suppression: the observation's sam object must not already
        be bound, and a tombstoned sam object cannot be immediately re-created.
        """
        if obs.sam_object_id in self._sam_to_track:
            raise ValueError(
                f"sam_object_id {obs.sam_object_id} already bound to a track"
            )
        if obs.sam_object_id in self._tombstones:
            deleted_at = self._tombstones[obs.sam_object_id]
            if frame_idx - deleted_at < self.lifecycle.tombstone_cooldown_frames:
                raise ValueError(
                    f"sam_object_id {obs.sam_object_id} is in delete cooldown"
                )
        tid = mot_track_id or self.next_track_id()
        track = Track(
            mot_track_id=tid,
            identity_lineage_id=lineage_id,
            sam_object_id=obs.sam_object_id,
            state=TrackState.TENTATIVE,
            start_frame=frame_idx,
        )
        track.update_observation(
            frame_idx=frame_idx,
            box=obs.box_xyxy,
            mask=obs.mask,
            confidence=obs.confidence,
            presence=obs.presence_score,
            source=obs.source,
            human_verified=obs.is_human_verified,
        )
        self._tracks[tid] = track
        self._sam_to_track[obs.sam_object_id] = tid
        self._update_output(frame_idx, track, obs)
        return track

    def update_track(
        self,
        track_id: int,
        frame_idx: int,
        obs: PromptObjectObservation,
        *,
        human_verified: bool = False,
    ) -> Track:
        track = self._get_active(track_id)
        if track.sam_object_id != obs.sam_object_id:
            raise ValueError("observation sam_object_id does not match track binding")
        track.update_observation(
            frame_idx=frame_idx,
            box=obs.box_xyxy,
            mask=obs.mask,
            confidence=obs.confidence,
            presence=obs.presence_score,
            source=obs.source,
            human_verified=human_verified,
        )
        update_state(track, frame_idx, matched=True, cfg=self.lifecycle)
        self._update_output(frame_idx, track, obs)
        return track

    def mark_missed(self, track_id: int, frame_idx: int) -> Track:
        track = self.get(track_id)
        if track is None or track.state in (TrackState.TERMINATED, TrackState.DELETED):
            raise ValueError(f"track {track_id} is not active")
        update_state(track, frame_idx, matched=False, cfg=self.lifecycle)
        return track

    def mark_lost(self, track_id: int, frame_idx: int) -> Track:
        track = self.get(track_id)
        if track is None:
            raise ValueError(f"unknown track {track_id}")
        track.state = TrackState.LOST
        return track

    def terminate_track(self, track_id: int, frame_idx: int) -> Track:
        track = self.get(track_id)
        if track is None:
            raise ValueError(f"unknown track {track_id}")
        track.state = TrackState.TERMINATED
        return track

    def delete_track(self, track_id: int, frame_idx: int, reason: str) -> Track:
        """Delete a fake/wrong trajectory and tombstone its SAM object."""
        track = self.get(track_id)
        if track is None:
            raise ValueError(f"unknown track {track_id}")
        track.state = TrackState.DELETED
        track.delete_reason = reason
        if track.sam_object_id is not None:
            self._sam_to_track.pop(track.sam_object_id, None)
            self._tombstones[track.sam_object_id] = frame_idx
        return track

    def rebind_sam_object(self, track_id: int, sam_object_id: int, frame_idx: int) -> Track:
        """Rebind a track to another SAM object (Reassign / rediscovery)."""
        track = self.get(track_id)
        if track is None or track.state in (TrackState.TERMINATED, TrackState.DELETED):
            raise ValueError(f"track {track_id} is not active")
        if sam_object_id in self._sam_to_track:
            other = self._sam_to_track[sam_object_id]
            if other != track_id:
                raise ValueError(
                    f"sam_object_id {sam_object_id} already bound to track {other}"
                )
        if track.sam_object_id is not None:
            self._sam_to_track.pop(track.sam_object_id, None)
        track.sam_object_id = sam_object_id
        self._sam_to_track[sam_object_id] = track_id
        return track

    def unbind_sam_object(self, track_id: int) -> Track:
        track = self.get(track_id)
        if track is None:
            raise ValueError(f"unknown track {track_id}")
        if track.sam_object_id is not None:
            self._sam_to_track.pop(track.sam_object_id, None)
        track.sam_object_id = None
        return track

    def detach_all_session_bindings(self) -> List[int]:
        """Detach every session-local SAM binding without deleting tracks.

        ``sam_object_id`` belongs to one active SAM session.  This explicit
        boundary operation clears only that binding; the Track objects, MOT
        IDs, lifecycle state, observations and output history remain owned by
        the caller's sequence-persistent runtime.
        """

        detached: List[int] = []
        for track in self._tracks.values():
            if track.state in (TrackState.TERMINATED, TrackState.DELETED):
                continue
            if track.sam_object_id is None:
                continue
            detached.append(int(track.mot_track_id))
            self._sam_to_track.pop(track.sam_object_id, None)
            track.sam_object_id = None
        return sorted(detached)

    def remove_output(self, frame_idx: int, track_id: int) -> None:
        if frame_idx in self._outputs:
            self._outputs[frame_idx].pop(track_id, None)

    # ------------------------------------------------------------------
    def outputs_for_frame(self, frame_idx: int) -> List[PromptObjectObservation]:
        mapping = self._outputs.get(frame_idx, {})
        return [mapping[k].copy() for k in sorted(mapping)]

    def _update_output(
        self,
        frame_idx: int,
        track: Track,
        obs: PromptObjectObservation,
    ) -> None:
        self._outputs.setdefault(frame_idx, {})[track.mot_track_id] = obs.copy()

    def _get_active(self, track_id: int) -> Track:
        track = self.get(track_id)
        if track is None or track.state in (TrackState.TERMINATED, TrackState.DELETED):
            raise ValueError(f"track {track_id} is not active")
        return track

    # ------------------------------------------------------------------
    def invariant_violations(self) -> List[str]:
        violations: List[str] = []
        sam_seen: Dict[int, int] = {}
        for track in self._tracks.values():
            if track.state in (TrackState.TERMINATED, TrackState.DELETED):
                continue
            if track.sam_object_id is None:
                continue
            if track.sam_object_id in sam_seen:
                violations.append(
                    f"sam_object_id {track.sam_object_id} bound to multiple active "
                    f"tracks ({sam_seen[track.sam_object_id]}, {track.mot_track_id})"
                )
            sam_seen[track.sam_object_id] = track.mot_track_id
        for frame_idx, mapping in self._outputs.items():
            ids = list(mapping.keys())
            if len(ids) != len(set(ids)):
                violations.append(f"duplicate mot_track_id in frame {frame_idx}")
        return violations

    def snapshot(self) -> dict:
        return deepcopy(
            {
                "tracks": self._tracks,
                "sam_to_track": self._sam_to_track,
                "tombstones": self._tombstones,
                "next_track_id": self._next_track_id,
                "outputs": self._outputs,
            }
        )

    def restore(self, snapshot: dict) -> None:
        self._tracks = deepcopy(snapshot["tracks"])
        self._sam_to_track = deepcopy(snapshot["sam_to_track"])
        self._tombstones = deepcopy(snapshot["tombstones"])
        self._next_track_id = snapshot["next_track_id"]
        self._outputs = deepcopy(snapshot["outputs"])
