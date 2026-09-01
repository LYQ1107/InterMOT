"""Shared output dataclasses produced by video tracker backends."""

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class PromptObjectObservation:
    """One per-object observation on one frame."""

    frame_idx: int
    sam_object_id: int
    mask: np.ndarray
    box_xyxy: np.ndarray
    confidence: float
    presence_score: Optional[float] = None
    source: str = "automatic_propagation"
    is_human_verified: bool = False

    def __post_init__(self) -> None:
        self.mask = np.asarray(self.mask)
        self.box_xyxy = np.asarray(self.box_xyxy, dtype=float).reshape(-1)
        if self.box_xyxy.size != 4:
            raise ValueError("box_xyxy must have exactly 4 elements")

    def copy(self) -> "PromptObjectObservation":
        return PromptObjectObservation(
            frame_idx=self.frame_idx,
            sam_object_id=self.sam_object_id,
            mask=self.mask.copy(),
            box_xyxy=self.box_xyxy.copy(),
            confidence=self.confidence,
            presence_score=self.presence_score,
            source=self.source,
            is_human_verified=self.is_human_verified,
        )
