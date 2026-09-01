"""Unified interaction data structures."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

import numpy as np

from sam3_intermot.backend.base import PromptVideoTrackerBackend
from sam3_intermot.identity.lineage import IdentityLineageRegistry
from sam3_intermot.tracking.track_manager import TrackManager


class ActionType(str, Enum):
    ADD = "Add"
    CORRECT = "Correct"
    REASSIGN = "Reassign"
    DELETE = "Delete"


@dataclass
class HumanInteraction:
    action_id: str
    frame_idx: int
    action_type: str
    target_track_id: Optional[int] = None
    target_lineage_id: Optional[int] = None
    box_xyxy: Optional[np.ndarray] = None
    mask: Optional[np.ndarray] = None
    points: Optional[np.ndarray] = None
    point_labels: Optional[np.ndarray] = None
    destination_track_id: Optional[int] = None
    source: str = "human"


@dataclass
class InteractionConfig:
    duplicate_iou_threshold: float = 0.5
    max_lost_gap_for_handover: int = 45
    min_box_area: float = 16.0
    enable_lineage_aware_add: bool = True
    enable_soft_delete: bool = False
    enable_atomic_reassign: bool = True
    enable_abstention: bool = True
    enable_guard: bool = True
    utility_threshold: float = 0.0
    abstain_stable_utility_penalty: float = 0.5


@dataclass
class InteractionResult:
    action_id: str
    action_type: str
    frame_idx: int
    accepted: bool
    rolled_back: bool = False
    reason: Optional[str] = None
    new_track_id: Optional[int] = None
    new_sam_object_id: Optional[int] = None
    before_summary: Dict[str, Any] = field(default_factory=dict)
    after_summary: Dict[str, Any] = field(default_factory=dict)
    violations: List[str] = field(default_factory=list)


@dataclass
class SystemContext:
    backend: PromptVideoTrackerBackend
    manager: TrackManager
    lineages: IdentityLineageRegistry
    config: InteractionConfig = field(default_factory=InteractionConfig)
    registry: Any = None
    transaction_log: List[Dict[str, Any]] = field(default_factory=list)
    _next_sam_object_id: int = 1

    def allocate_sam_object_id(self) -> int:
        sam_id = self._next_sam_object_id
        self._next_sam_object_id += 1
        return sam_id

    def log_transaction(self, entry: Dict[str, Any]) -> None:
        self.transaction_log.append(entry)


def summarize_manager(manager: TrackManager) -> Dict[str, Any]:
    return {
        "active_tracks": [
            {
                "mot_track_id": t.mot_track_id,
                "lineage_id": t.identity_lineage_id,
                "sam_object_id": t.sam_object_id,
                "state": t.state.value,
            }
            for t in manager.active_tracks()
        ],
        "violations": manager.invariant_violations(),
    }
