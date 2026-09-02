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
    # Raw official ``out_obj_ids`` are retained only when the backend parser
    # has an authoritative value.  ``sam_object_id`` remains the historical
    # adapter-visible/stable ID so existing callers keep their semantics.
    raw_sam_object_id: Optional[int] = None
    presence_score: Optional[float] = None
    source: str = "automatic_propagation"
    is_human_verified: bool = False
    # Optional run/segment provenance is carried with the observation when
    # available.  These fields are metadata only; they never replace the
    # official raw axis or become a public identity by numeric coincidence.
    source_run_id: Optional[str] = None
    session_id: Optional[str] = None
    segment_id: Optional[str] = None
    window_id: Optional[str] = None
    chunk_id: Optional[str] = None
    candidate_index: Optional[int] = None

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
            raw_sam_object_id=self.raw_sam_object_id,
            presence_score=self.presence_score,
            source=self.source,
            is_human_verified=self.is_human_verified,
            source_run_id=self.source_run_id,
            session_id=self.session_id,
            segment_id=self.segment_id,
            window_id=self.window_id,
            chunk_id=self.chunk_id,
            candidate_index=self.candidate_index,
        )
