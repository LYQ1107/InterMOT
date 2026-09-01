"""Track state and per-track bookkeeping."""

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional

import numpy as np


class TrackState(str, Enum):
    TENTATIVE = "tentative"
    CONFIRMED = "confirmed"
    LOST = "lost"
    QUARANTINED = "quarantined"
    RECOVERED = "recovered"
    CONFIRMED_DELETED = "confirmed_deleted"
    TERMINATED = "terminated"
    DELETED = "deleted"


@dataclass
class Track:
    mot_track_id: int
    identity_lineage_id: int
    sam_object_id: Optional[int]
    state: TrackState = TrackState.TENTATIVE
    start_frame: int = 0
    last_seen_frame: Optional[int] = None
    last_human_verified_frame: Optional[int] = None
    last_box: Optional[np.ndarray] = None
    last_mask: Optional[np.ndarray] = None
    age: int = 0
    time_since_update: int = 0
    confidence_history: List[float] = field(default_factory=list)
    presence_history: List[float] = field(default_factory=list)
    source_history: List[str] = field(default_factory=list)
    delete_reason: Optional[str] = None

    def update_observation(
        self,
        frame_idx: int,
        box: np.ndarray,
        mask: np.ndarray,
        confidence: float,
        presence: Optional[float],
        source: str,
        human_verified: bool = False,
    ) -> None:
        self.last_seen_frame = frame_idx
        self.last_box = np.asarray(box, dtype=float).copy()
        self.last_mask = np.asarray(mask).copy()
        self.age = frame_idx - self.start_frame
        self.time_since_update = 0
        self.confidence_history.append(float(confidence))
        if presence is not None:
            self.presence_history.append(float(presence))
        self.source_history.append(source)
        if human_verified:
            self.last_human_verified_frame = frame_idx
