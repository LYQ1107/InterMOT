"""Identity lineage registry.

``identity_lineage_id`` is the semantic identity across re-detection gaps and
is deliberately distinct from both ``mot_track_id`` and ``sam_object_id``.
"""

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from sam3_intermot.tracking.association import box_iou, center_distance


@dataclass
class IdentityLineage:
    lineage_id: int
    created_frame: int
    mot_track_ids: List[int] = field(default_factory=list)
    closed_frame: Optional[int] = None

    def bind_track(self, mot_track_id: int) -> None:
        if self.closed_frame is not None:
            raise ValueError(f"lineage {self.lineage_id} is closed")
        self.mot_track_ids.append(mot_track_id)

    def close(self, frame_idx: int) -> None:
        self.closed_frame = frame_idx


class IdentityLineageRegistry:
    def __init__(self) -> None:
        self._lineages: Dict[int, IdentityLineage] = {}
        self._next_lineage_id = 1

    def create(self, frame_idx: int) -> IdentityLineage:
        lineage = IdentityLineage(
            lineage_id=self._next_lineage_id, created_frame=frame_idx
        )
        self._next_lineage_id += 1
        self._lineages[lineage.lineage_id] = lineage
        return lineage

    def get(self, lineage_id: int) -> Optional[IdentityLineage]:
        return self._lineages.get(lineage_id)

    def all(self) -> Dict[int, IdentityLineage]:
        return self._lineages

    def find_lost_lineage(
        self,
        observation,
        frame_idx: int,
        manager,
        max_gap: int = 45,
        iou_threshold: float = 0.1,
        center_threshold: float = 180.0,
    ) -> Optional[int]:
        """Return a recently-lost lineage that plausibly matches the observation.

        Uses only past state: lineage close time and last track box.
        """
        best: Optional[int] = None
        best_score = -1.0
        for lineage_id, lineage in self._lineages.items():
            if lineage.closed_frame is None:
                continue
            if frame_idx - lineage.closed_frame > max_gap:
                continue
            if not lineage.mot_track_ids:
                continue
            last_track = manager.get(lineage.mot_track_ids[-1])
            if last_track is None or last_track.last_box is None:
                continue
            iou = box_iou(observation.box_xyxy, last_track.last_box)
            dist = center_distance(observation.box_xyxy, last_track.last_box)
            if iou >= iou_threshold and dist <= center_threshold:
                score = iou - 1e-4 * dist
                if score > best_score:
                    best_score = score
                    best = lineage_id
        return best

    def snapshot(self) -> dict:
        return deepcopy(
            {
                "lineages": self._lineages,
                "next_lineage_id": self._next_lineage_id,
            }
        )

    def restore(self, snapshot: dict) -> None:
        self._lineages = deepcopy(snapshot["lineages"])
        self._next_lineage_id = snapshot["next_lineage_id"]
