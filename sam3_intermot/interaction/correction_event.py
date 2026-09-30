"""Schema for one human correction of an identity prediction.

The event is deliberately a data contract, not a memory update.  N72R20
creates these records by comparing a GT-free candidate-stream prediction with
GT offline.  A later stage may consume the same schema for real UI events.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, ClassVar, Mapping, Sequence


CORRECTION_EVENT_TYPES = frozenset(
    {
        "identity_switch",
        "missed_target",
        "wrong_recovery",
    }
)


def _vector_tuple(value: Sequence[float] | None, *, name: str, required: bool) -> tuple[float, ...] | None:
    if value is None:
        if required:
            raise ValueError(f"{name} is required")
        return None
    result = tuple(float(item) for item in value)
    if not result:
        raise ValueError(f"{name} must be non-empty")
    if not all(math.isfinite(item) for item in result):
        raise ValueError(f"{name} must contain finite values")
    return result


@dataclass(frozen=True)
class HumanCorrectionEvent:
    """One correction opportunity discovered from an identity error.

    ``predicted_candidate_id`` and ``corrected_candidate_id`` are stream
    identifiers.  The corrected id is ``None`` for a missed target, where the
    simulated human correction supplies ``corrected_embedding`` from the
    clicked GT box instead.  The event keeps candidate context but never an
    image crop, mask, feature map, or attention tensor.
    """

    frame_id: int
    sequence_id: str
    identity_id: int
    predicted_candidate_id: str | None
    corrected_candidate_id: str | None
    predicted_identity_state: tuple[float, ...]
    corrected_embedding: tuple[float, ...]
    candidate_context: tuple[dict[str, Any], ...]
    timestamp: str
    error_type: str
    error_interval_frames: int | None = None
    hard_negative: bool = False
    interaction_source: str = "simulated_from_gt"

    VECTOR_DIMENSION: ClassVar[int] = 512

    def __post_init__(self) -> None:
        if int(self.frame_id) < 1:
            raise ValueError("frame_id must be a positive video frame id")
        if not str(self.sequence_id).strip():
            raise ValueError("sequence_id must be non-empty")
        if int(self.identity_id) < 0:
            raise ValueError("identity_id must be non-negative")
        if self.error_type not in CORRECTION_EVENT_TYPES:
            raise ValueError(f"unsupported correction event type: {self.error_type!r}")
        if not str(self.timestamp).strip():
            raise ValueError("timestamp must be non-empty")
        if not str(self.interaction_source).strip():
            raise ValueError("interaction_source must be non-empty")
        if self.error_interval_frames is not None and int(self.error_interval_frames) < 1:
            raise ValueError("error_interval_frames must be positive when present")

        predicted_state = _vector_tuple(
            self.predicted_identity_state,
            name="predicted_identity_state",
            required=True,
        )
        corrected_embedding = _vector_tuple(
            self.corrected_embedding,
            name="corrected_embedding",
            required=True,
        )
        assert predicted_state is not None
        assert corrected_embedding is not None
        if len(predicted_state) != self.VECTOR_DIMENSION:
            raise ValueError(
                f"predicted_identity_state must have {self.VECTOR_DIMENSION} values, got {len(predicted_state)}"
            )
        if len(corrected_embedding) != self.VECTOR_DIMENSION:
            raise ValueError(
                f"corrected_embedding must have {self.VECTOR_DIMENSION} values, got {len(corrected_embedding)}"
            )
        if self.error_type == "missed_target" and self.corrected_candidate_id is not None:
            raise ValueError("missed_target must not claim a corrected candidate id")
        if self.error_type != "missed_target" and self.corrected_candidate_id is None:
            raise ValueError(f"{self.error_type} requires corrected_candidate_id")

        context = tuple(dict(item) for item in self.candidate_context)
        object.__setattr__(self, "frame_id", int(self.frame_id))
        object.__setattr__(self, "identity_id", int(self.identity_id))
        object.__setattr__(self, "predicted_identity_state", predicted_state)
        object.__setattr__(self, "corrected_embedding", corrected_embedding)
        object.__setattr__(self, "candidate_context", context)
        if self.error_interval_frames is not None:
            object.__setattr__(self, "error_interval_frames", int(self.error_interval_frames))
        object.__setattr__(self, "hard_negative", bool(self.hard_negative))

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible event payload."""

        return {
            "frame_id": self.frame_id,
            "sequence_id": self.sequence_id,
            "identity_id": self.identity_id,
            "predicted_candidate_id": self.predicted_candidate_id,
            "corrected_candidate_id": self.corrected_candidate_id,
            "predicted_identity_state": list(self.predicted_identity_state),
            "corrected_embedding": list(self.corrected_embedding),
            "candidate_context": [dict(item) for item in self.candidate_context],
            "timestamp": self.timestamp,
            "error_type": self.error_type,
            "error_interval_frames": self.error_interval_frames,
            "hard_negative": self.hard_negative,
            "interaction_source": self.interaction_source,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "HumanCorrectionEvent":
        """Construct an event from a JSON object."""

        return cls(
            frame_id=int(value["frame_id"]),
            sequence_id=str(value["sequence_id"]),
            identity_id=int(value["identity_id"]),
            predicted_candidate_id=value.get("predicted_candidate_id"),
            corrected_candidate_id=value.get("corrected_candidate_id"),
            predicted_identity_state=tuple(value["predicted_identity_state"]),
            corrected_embedding=tuple(value["corrected_embedding"]),
            candidate_context=tuple(value.get("candidate_context", ())),
            timestamp=str(value["timestamp"]),
            error_type=str(value["error_type"]),
            error_interval_frames=(
                None if value.get("error_interval_frames") is None else int(value["error_interval_frames"])
            ),
            hard_negative=bool(value.get("hard_negative", False)),
            interaction_source=str(value.get("interaction_source", "simulated_from_gt")),
        )


__all__ = ["CORRECTION_EVENT_TYPES", "HumanCorrectionEvent"]
