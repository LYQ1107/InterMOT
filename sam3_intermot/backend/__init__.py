"""Backend abstraction for promptable video trackers."""

from sam3_intermot.backend.base import (
    NotSupportedError,
    PromptVideoTrackerBackend,
)
from sam3_intermot.backend.output_types import PromptObjectObservation

__all__ = [
    "NotSupportedError",
    "PromptVideoTrackerBackend",
    "PromptObjectObservation",
]
