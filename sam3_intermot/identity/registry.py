"""Unified SAM/MOT/lineage identity registry.

The registry owns the three-way mapping between:

* ``sam_object_id`` (backend object id)
* ``mot_track_id`` (output track id)
* ``identity_lineage_id`` (semantic identity)

It wraps ``TrackManager`` + ``IdentityLineageRegistry`` and adds explicit
window-handover and uniqueness invariants.
"""

from typing import Dict, List, Optional

import numpy as np

from sam3_intermot.backend.output_types import PromptObjectObservation
from sam3_intermot.identity.lineage import IdentityLineageRegistry
from sam3_intermot.tracking.association import box_iou, center_distance
from sam3_intermot.tracking.track_manager import TrackManager


class ObjectIdentityRegistry:
    def __init__(
        self,
        manager: Optional[TrackManager] = None,
        lineages: Optional[IdentityLineageRegistry] = None,
        iou_threshold: float = 0.2,
        center_threshold: float = 200.0,
    ) -> None:
        self.manager = manager or TrackManager()
        self.lineages = lineages or IdentityLineageRegistry()
        self.iou_threshold = iou_threshold
        self.center_threshold = center_threshold

    # ------------------------------------------------------------------
    def register_auto_object(
        self, frame_idx: int, obs: PromptObjectObservation
    ):
        track = self._find_by_sam(obs.sam_object_id)
        if track is not None:
            return self.manager.update_track(track.mot_track_id, frame_idx, obs)
        if obs.sam_object_id in self.manager._sam_to_track:
            stale = self.manager._sam_to_track.pop(obs.sam_object_id)
            stale_track = self.manager.get(stale)
            if stale_track is not None and stale_track.sam_object_id == obs.sam_object_id:
                stale_track.sam_object_id = None
        track = self._find_by_box(obs)
        if track is not None:
            if track.sam_object_id != obs.sam_object_id:
                self.rebind(track.mot_track_id, obs.sam_object_id, frame_idx)
            return self.manager.update_track(track.mot_track_id, frame_idx, obs)
        lineage = self.lineages.create(frame_idx)
        track = self.manager.create_track(frame_idx, obs, lineage.lineage_id)
        lineage.bind_track(track.mot_track_id)
        return track

    def register_human_object(
        self,
        frame_idx: int,
        obs: PromptObjectObservation,
        lineage_id: Optional[int] = None,
    ):
        track = self._find_by_sam(obs.sam_object_id)
        if track is not None:
            return self.manager.update_track(track.mot_track_id, frame_idx, obs)
        track = self._find_by_box(obs)
        if track is not None:
            if track.sam_object_id != obs.sam_object_id:
                self.rebind(track.mot_track_id, obs.sam_object_id, frame_idx)
            return self.manager.update_track(track.mot_track_id, frame_idx, obs)
        lineage = self.lineages.get(lineage_id) if lineage_id is not None else None
        if lineage is None:
            lineage = self.lineages.create(frame_idx)
        track = self.manager.create_track(frame_idx, obs, lineage.lineage_id)
        lineage.bind_track(track.mot_track_id)
        return track

    # ------------------------------------------------------------------
    def lookup_by_sam_object_id(self, sam_object_id: int):
        tid = self.manager._sam_to_track.get(sam_object_id)
        return self.manager.get(tid) if tid is not None else None

    def lookup_by_mot_track_id(self, mot_track_id: int):
        return self.manager.get(mot_track_id)

    def lookup_by_lineage_id(self, lineage_id: int):
        lineage = self.lineages.get(lineage_id)
        if lineage is None or not lineage.mot_track_ids:
            return None
        return self.manager.get(lineage.mot_track_ids[-1])

    def rebind(self, mot_track_id: int, sam_object_id: int, frame_idx: int):
        return self.manager.rebind_sam_object(mot_track_id, sam_object_id, frame_idx)

    def mark_lost(self, mot_track_id: int, frame_idx: int):
        return self.manager.mark_missed(mot_track_id, frame_idx)

    def terminate(self, mot_track_id: int, frame_idx: int):
        return self.manager.terminate_track(mot_track_id, frame_idx)

    def delete(self, mot_track_id: int, frame_idx: int, reason: str):
        return self.manager.delete_track(mot_track_id, frame_idx, reason)

    def handover_across_window(self, frame_idx: int, obs: PromptObjectObservation):
        """Explicit window handover: match by geometry, never by stale id."""
        track = self._find_by_box(obs)
        if track is None:
            return None
        if track.sam_object_id != obs.sam_object_id:
            self.rebind(track.mot_track_id, obs.sam_object_id, frame_idx)
        return track

    def unbind_all_for_window(self) -> None:
        for track in list(self.manager.active_tracks()):
            self.manager.unbind_sam_object(track.mot_track_id)

    # ------------------------------------------------------------------
    def invariant_violations(self) -> List[str]:
        violations = list(self.manager.invariant_violations())
        sam_to_track: Dict[int, int] = {}
        track_to_sam: Dict[int, int] = {}
        for track in self.manager.active_tracks():
            if track.sam_object_id is None:
                continue
            if track.sam_object_id in sam_to_track:
                violations.append(
                    f"sam_object_id {track.sam_object_id} bound to multiple tracks"
                )
            sam_to_track[track.sam_object_id] = track.mot_track_id
            if track.mot_track_id in track_to_sam:
                violations.append(
                    f"mot_track_id {track.mot_track_id} bound to multiple sam ids"
                )
            track_to_sam[track.mot_track_id] = track.sam_object_id
        lineage_outputs: Dict[int, set] = {}
        for frame_idx, mapping in self.manager._outputs.items():
            for tid, obs in mapping.items():
                track = self.manager.get(tid)
                if track is None:
                    continue
                lineage_outputs.setdefault(track.identity_lineage_id, set()).add(frame_idx)
        for lineage_id, frames in lineage_outputs.items():
            pass  # uniqueness within one frame is checked below
        for frame_idx, mapping in self.manager._outputs.items():
            seen_lineage = set()
            for tid in mapping:
                track = self.manager.get(tid)
                if track is not None and track.identity_lineage_id in seen_lineage:
                    violations.append(
                        f"lineage {track.identity_lineage_id} output twice at frame {frame_idx}"
                    )
                if track is not None:
                    seen_lineage.add(track.identity_lineage_id)
        return sorted(set(violations))

    # ------------------------------------------------------------------
    def _find_by_sam(self, sam_object_id: int):
        tid = self.manager._sam_to_track.get(sam_object_id)
        track = self.manager.get(tid) if tid is not None else None
        if track is not None and track.state.value in ("terminated", "deleted"):
            return None
        return track

    def _find_by_box(self, obs: PromptObjectObservation):
        best = None
        best_score = float("-inf")
        for track in self.manager.active_tracks():
            if track.last_box is None:
                continue
            iou = box_iou(obs.box_xyxy, track.last_box)
            if iou < self.iou_threshold:
                continue
            dist = center_distance(obs.box_xyxy, track.last_box)
            score = iou - 1e-4 * dist
            if score > best_score:
                best_score = score
                best = track
        return best
