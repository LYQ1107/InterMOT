"""Stable backend interface that decouples upper MOT logic from SAM internals."""

from abc import ABC, abstractmethod
from typing import List, Optional

import numpy as np

from sam3_intermot.backend.output_types import PromptObjectObservation


class NotSupportedError(NotImplementedError):
    """Raised when the underlying backend does not support a capability.

    This must be raised instead of silently emulating an unsupported
    capability, so upper layers can distinguish real support from fallback
    behavior.
    """


class PromptVideoTrackerBackend(ABC):
    """Minimal stable API required by the Interactive MOT manager.

    All geometric inputs to this interface are absolute pixel coordinates
    (``box_xyxy`` as ``[x1, y1, x2, y2]``).  Backend implementations are
    responsible for any conversion into their native coordinate system.
    """

    @abstractmethod
    def start_video(self, video_source: str) -> str:
        """Start a new video session and return a session id."""

    @abstractmethod
    def detect_concept(
        self, frame_idx: int, text_prompt: str
    ) -> List[PromptObjectObservation]:
        """Detect all instances of ``text_prompt`` on a frame.

        Detections are only candidates: identity decisions belong to the
        Track Manager.
        """

    @abstractmethod
    def add_box(
        self,
        frame_idx: int,
        object_id: int,
        box_xyxy: np.ndarray,
    ) -> PromptObjectObservation:
        """Create or re-prompt an instance-level object with a box."""

    @abstractmethod
    def add_points(
        self,
        frame_idx: int,
        object_id: int,
        points: np.ndarray,
        labels: np.ndarray,
    ) -> PromptObjectObservation:
        """Create or re-prompt an instance-level object with points."""

    @abstractmethod
    def add_mask(
        self,
        frame_idx: int,
        object_id: int,
        mask: np.ndarray,
    ) -> PromptObjectObservation:
        """Create or re-prompt an instance-level object with a mask."""

    @abstractmethod
    def correct_object(
        self,
        frame_idx: int,
        object_id: int,
        box_xyxy: Optional[np.ndarray] = None,
        points: Optional[np.ndarray] = None,
        labels: Optional[np.ndarray] = None,
        mask: Optional[np.ndarray] = None,
    ) -> PromptObjectObservation:
        """Correct an existing object with a human-verified prompt."""

    @abstractmethod
    def propagate(
        self,
        start_frame: int,
        end_frame: int,
        start_frame_index: Optional[int] = None,
    ) -> dict:
        """Propagate all active objects from ``start_frame`` to ``end_frame``.

        Returns a mapping ``{frame_idx: [PromptObjectObservation, ...]}``.
        """

    @abstractmethod
    def remove_object(self, object_id: int) -> None:
        """Remove an object from the session without affecting other objects."""

    @abstractmethod
    def reset_object(self, object_id: int) -> None:
        """Reset an object's memory to its initial prompt state."""

    @abstractmethod
    def get_frame_outputs(self, frame_idx: int) -> List[PromptObjectObservation]:
        """Return observations available for ``frame_idx``."""

    def export_frame_candidates(self, frame_idx: int, **kwargs):
        """Export every candidate available on one frame with provenance.

        This is intentionally a concrete capability rather than a new
        abstract requirement: older test doubles and non-SAM backends can
        remain usable, while candidate-complete protocols can fail explicitly
        when the backend has no exporter.
        """
        raise NotSupportedError("backend does not expose candidate export")

    @abstractmethod
    def close(self) -> None:
        """Close the session and release all resources."""
