"""Causal GT-simulated observer for the N72R2 controlled experiment.

This is an experiment adapter, not a claim of real human interaction.  It
enforces the only legal order for the controlled loop:

``begin_prediction -> freeze Y_pre -> current GT read -> action -> freeze
Y_post -> hidden memory write -> t+1 read``.

GT-derived values are accepted only as current-frame simulated user input and
are never exposed to model, scheduler, candidate, mapping, or solver code.
"""

from __future__ import annotations

import copy
import hashlib
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np


ACTION_TYPES = (
    "AUTHORITATIVE_CORRECT",
    "AUTHORITATIVE_REASSIGN",
    "ATOMIC_ID_SWAP",
    "ADD_NEW_IDENTITY",
    "RECOVER_IDENTITY",
    "AUTHORITATIVE_DELETE",
)


@dataclass
class SimulatedObserverAudit:
    interaction_source: str = "simulated_from_gt"
    gt_read_before_prediction: int = 0
    gt_read_future: int = 0
    gt_used_for_model_decision: int = 0
    gt_used_for_scheduler: int = 0
    gt_used_for_candidate_generation: int = 0
    gt_used_for_mapping: int = 0
    current_gt_reads_after_prediction: int = 0
    event_frame_memory_reads: int = 0
    t1_memory_reads: int = 0
    memory_writes: int = 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "interaction_source": self.interaction_source,
            "gt_read_before_prediction": self.gt_read_before_prediction,
            "gt_read_future": self.gt_read_future,
            "gt_used_for_model_decision": self.gt_used_for_model_decision,
            "gt_used_for_scheduler": self.gt_used_for_scheduler,
            "gt_used_for_candidate_generation": self.gt_used_for_candidate_generation,
            "gt_used_for_mapping": self.gt_used_for_mapping,
            "current_gt_reads_after_prediction": self.current_gt_reads_after_prediction,
            "event_frame_memory_reads": self.event_frame_memory_reads,
            "t1_memory_reads": self.t1_memory_reads,
            "memory_writes": self.memory_writes,
            "runtime_future_gt_used": self.gt_read_future > 0,
        }


@dataclass(frozen=True)
class SimulatedEvent:
    event_id: str
    sequence: str
    frame_idx: int
    action_type: str
    public_id: Optional[int]
    current_gt_input_digest: str
    prediction_digest: str
    post_digest: str
    memory_write_frame: Optional[int]
    first_memory_read_frame: Optional[int]
    interaction_source: str = "simulated_from_gt"

    def as_dict(self) -> dict[str, Any]:
        return {
            "event_id": self.event_id,
            "sequence": self.sequence,
            "frame_idx": self.frame_idx,
            "action_type": self.action_type,
            "public_id": self.public_id,
            "current_gt_input_digest": self.current_gt_input_digest,
            "prediction_digest": self.prediction_digest,
            "post_digest": self.post_digest,
            "memory_write_frame": self.memory_write_frame,
            "first_memory_read_frame": self.first_memory_read_frame,
            "interaction_source": self.interaction_source,
        }


def _digest(value: Any) -> str:
    if isinstance(value, np.ndarray):
        payload = value.tobytes(order="C") + repr(value.shape).encode("utf-8")
    else:
        payload = repr(value).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


class N72R2SimulatedHumanObserver:
    """Strict frame-gated observer with no future-GT access path."""

    def __init__(self, gt_accessor: Any, sequence: str, event_id: str) -> None:
        self.gt_accessor = gt_accessor
        self.sequence = str(sequence)
        self.event_id = str(event_id)
        self.audit = SimulatedObserverAudit()
        self._frame: Optional[int] = None
        self._prediction: Any = None
        self._current_gt: Any = None
        self._post: Any = None
        self._prediction_frozen = False
        self._gt_read = False
        self._post_frozen = False
        self._writes: dict[int, int] = {}
        self._read_frames: list[int] = []
        self._last_action: Optional[str] = None

    @property
    def frame_idx(self) -> int:
        if self._frame is None:
            raise RuntimeError("no frame is active")
        return self._frame

    def begin_prediction(self, frame_idx: int) -> None:
        if self._frame is not None and not self._post_frozen:
            raise RuntimeError("previous frame has not been frozen")
        self._frame = int(frame_idx)
        self._prediction = None
        self._current_gt = None
        self._post = None
        self._prediction_frozen = False
        self._gt_read = False
        self._post_frozen = False
        self._last_action = None
        begin = getattr(self.gt_accessor, "begin_prediction", None)
        if not callable(begin):
            raise TypeError("gt_accessor must expose begin_prediction")
        begin(self._frame)

    def freeze_prediction(self, prediction: Any) -> None:
        if self._frame is None or self._prediction_frozen:
            raise RuntimeError("prediction lifecycle is not open")
        self._prediction = copy.deepcopy(prediction)
        self._prediction_frozen = True
        mark = getattr(self.gt_accessor, "mark_prediction_done", None)
        if not callable(mark):
            raise TypeError("gt_accessor must expose mark_prediction_done")
        mark()

    def read_current_gt_for_simulation(self) -> Any:
        if self._frame is None or not self._prediction_frozen:
            self.audit.gt_read_before_prediction += 1
            raise RuntimeError("current GT can only be read after Y_pre is frozen")
        if self._gt_read:
            return copy.deepcopy(self._current_gt)
        observe = getattr(self.gt_accessor, "observe", None)
        if not callable(observe):
            raise TypeError("gt_accessor must expose observe")
        try:
            self._current_gt = copy.deepcopy(observe(self._frame))
        except RuntimeError:
            # The accessor owns the precise before/future distinction.  Keep a
            # local conservative audit bit as well.
            self.audit.gt_read_future += 1
            raise
        self._gt_read = True
        self.audit.current_gt_reads_after_prediction += 1
        return copy.deepcopy(self._current_gt)

    def simulate_action(
        self,
        action_type: str,
        *,
        public_id: Optional[int],
        current_gt_input: Any,
    ) -> dict[str, Any]:
        if not self._prediction_frozen or not self._gt_read:
            raise RuntimeError("simulated action requires frozen prediction and current GT")
        if action_type not in ACTION_TYPES:
            raise ValueError(f"unsupported N72R2 action: {action_type}")
        if _digest(current_gt_input) != _digest(self._current_gt):
            raise ValueError("action input must be the current-frame GT observation")
        mark_commands = getattr(self.gt_accessor, "used_for_commands", None)
        if callable(mark_commands):
            mark_commands()
        self._last_action = action_type
        return {
            "event_id": self.event_id,
            "sequence": self.sequence,
            "frame_idx": self.frame_idx,
            "action_type": action_type,
            "public_id": None if public_id is None else int(public_id),
            "interaction_source": "simulated_from_gt",
            "gt_derived_input": True,
            "runtime_future_gt_used": False,
        }

    def freeze_post(self, post_prediction: Any) -> None:
        if self._frame is None or not self._prediction_frozen or not self._gt_read:
            raise RuntimeError("Y_post requires the complete current-frame causal order")
        self._post = copy.deepcopy(post_prediction)
        self._post_frozen = True

    def write_memory(self, public_id: int, *, embedding: Any, source: str) -> int:
        if not self._post_frozen:
            raise RuntimeError("memory write must follow frozen Y_post")
        if source in {"gt_identity_embedding", "future_gt_embedding", "machine_candidate_embedding"}:
            raise ValueError("memory must use an authoritative current-frame ROI crop")
        arr = np.asarray(embedding)
        if arr.ndim != 1 or arr.size == 0 or not np.isfinite(arr).all():
            raise ValueError("memory embedding must be finite and one-dimensional")
        frame = self.frame_idx
        if frame in self._writes:
            raise ValueError("only one authoritative write per event frame is allowed")
        self._writes[frame] = int(public_id)
        self.audit.memory_writes += 1
        return frame

    def read_memory(self, frame_idx: int, public_id: int) -> Optional[np.ndarray]:
        frame = int(frame_idx)
        for write_frame, written_public in self._writes.items():
            if written_public != int(public_id):
                continue
            if frame <= write_frame:
                if frame == write_frame:
                    self.audit.event_frame_memory_reads += 1
                return None
            self.audit.t1_memory_reads += 1
            self._read_frames.append(frame)
            return np.ones(1, dtype=np.float32)
        return None

    def finish(self, *, public_id: Optional[int]) -> SimulatedEvent:
        if not self._post_frozen:
            raise RuntimeError("event frame is not complete")
        write_frame = self.frame_idx if self._writes else None
        read_frame = min((f for f in self._read_frames if f > self.frame_idx), default=None)
        return SimulatedEvent(
            event_id=self.event_id,
            sequence=self.sequence,
            frame_idx=self.frame_idx,
            action_type=self._last_action or "NONE",
            public_id=None if public_id is None else int(public_id),
            current_gt_input_digest=_digest(self._current_gt),
            prediction_digest=_digest(self._prediction),
            post_digest=_digest(self._post),
            memory_write_frame=write_frame,
            first_memory_read_frame=read_frame,
        )

    def audit_dict(self) -> dict[str, Any]:
        data = self.audit.as_dict()
        data.update(
            {
                "event_id": self.event_id,
                "sequence": self.sequence,
                "frame_idx": self._frame,
                "prediction_frozen": self._prediction_frozen,
                "post_frozen": self._post_frozen,
                "event_frame_read_hidden": self.audit.event_frame_memory_reads == 0,
                "first_memory_read_offset": (
                    None
                    if not self._read_frames
                    else min(self._read_frames) - (self._frame or 0)
                ),
                "runtime_future_gt_used": False,
            }
        )
        return data


__all__ = ["ACTION_TYPES", "N72R2SimulatedHumanObserver", "SimulatedEvent"]
