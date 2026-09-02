"""Server-side causal guards for N72R1 event transactions.

The guard is deliberately independent of the tracker implementation.  It
records the ordering boundary that a runtime adapter must obey:

``official spatial correction -> memory write at event frame -> read from
event+1``.

It never accepts GT or posthoc labels and it does not infer a public ID.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from typing import Any, Mapping


class RuntimeCausalError(ValueError):
    """Raised when an event transaction violates the causal contract."""


def _text(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def _digest(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass
class RuntimeCausalGuard:
    """Track one server-generated event's correction/memory boundary."""

    event_id: str
    action_type: str
    event_frame: int
    session_id: str
    runtime_future_gt_used: bool = False
    spatial_correction: dict[str, Any] | None = None
    memory_write: dict[str, Any] | None = None
    memory_reads: list[dict[str, Any]] = field(default_factory=list)
    future_frames: list[int] = field(default_factory=list)
    _finalized: bool = False

    def __post_init__(self) -> None:
        if not _text(self.event_id) or not _text(self.action_type) or not _text(self.session_id):
            raise RuntimeCausalError("event_id, action_type and session_id are required")
        if not isinstance(self.event_frame, int) or isinstance(self.event_frame, bool) or self.event_frame < 0:
            raise RuntimeCausalError("event_frame must be a non-negative integer")
        if self.runtime_future_gt_used is not False:
            raise RuntimeCausalError("runtime_future_gt_used must be exactly false")

    def _ensure_open(self) -> None:
        if self._finalized:
            raise RuntimeCausalError("event transaction is finalized")

    def _reject_gt(self, payload: Mapping[str, Any]) -> None:
        forbidden = {"gt", "gt_id", "gt_box", "future_gt", "future_identity", "posthoc_gt", "reward", "iou"}
        present = [str(key) for key in payload if str(key).lower() in forbidden or str(key).lower().startswith("future_gt")]
        if present:
            raise RuntimeCausalError("runtime event contains forbidden GT/posthoc fields: " + ",".join(sorted(present)))

    def record_spatial_correction(
        self,
        frame: int,
        *,
        backend_prompt_route: str,
        correction_id: str,
        official_backend: bool = True,
        **audit: Any,
    ) -> dict[str, Any]:
        self._ensure_open()
        payload = dict(audit)
        self._reject_gt(payload)
        if int(frame) != self.event_frame:
            raise RuntimeCausalError("spatial correction must occur on event_frame")
        if self.spatial_correction is not None:
            raise RuntimeCausalError("spatial correction may be recorded once")
        if self.memory_write is not None:
            raise RuntimeCausalError("spatial correction must precede memory write")
        if not _text(backend_prompt_route) or not _text(correction_id) or official_backend is not True:
            raise RuntimeCausalError("official backend route and correction_id are required")
        self.spatial_correction = {
            "frame": self.event_frame,
            "backend_prompt_route": backend_prompt_route,
            "correction_id": correction_id,
            "official_backend": True,
            "status": "PASS",
            **payload,
        }
        return dict(self.spatial_correction)

    def write_memory(
        self,
        frame: int,
        *,
        memory_key: str,
        feature_sha256: str,
        source: str,
        **audit: Any,
    ) -> dict[str, Any]:
        self._ensure_open()
        payload = dict(audit)
        self._reject_gt(payload)
        if self.spatial_correction is None:
            raise RuntimeCausalError("memory write requires completed spatial correction")
        if int(frame) != self.event_frame:
            raise RuntimeCausalError("memory write must be committed on event_frame")
        if self.memory_write is not None:
            raise RuntimeCausalError("memory write may be recorded once")
        if not _text(memory_key) or not _text(source) or not isinstance(feature_sha256, str) or len(feature_sha256) != 64:
            raise RuntimeCausalError("memory key, source and feature digest are required")
        self.memory_write = {
            "frame": self.event_frame,
            "memory_key": memory_key,
            "feature_sha256": feature_sha256.lower(),
            "source": source,
            "visible_from_frame": self.event_frame + 1,
            "current_frame_write_hidden": True,
            "write_after_spatial_correction": True,
            **payload,
        }
        return dict(self.memory_write)

    def read_memory(self, frame: int, *, memory_key: str, **audit: Any) -> dict[str, Any]:
        self._ensure_open()
        payload = dict(audit)
        self._reject_gt(payload)
        if self.memory_write is None:
            raise RuntimeCausalError("memory read requested before a memory write")
        frame = int(frame)
        if frame <= self.event_frame:
            raise RuntimeCausalError("new event memory is not visible on event_frame or prefix")
        if frame < int(self.memory_write["visible_from_frame"]):
            raise RuntimeCausalError("memory read precedes visible_from_frame")
        if memory_key != self.memory_write["memory_key"]:
            raise RuntimeCausalError("memory key is not the event's written key")
        item = {"frame": frame, "memory_key": memory_key, "new_memory_visible": True, **payload}
        self.memory_reads.append(item)
        return dict(item)

    def record_future_frame(self, frame: int, *, runtime_future_gt_used: bool = False, **audit: Any) -> dict[str, Any]:
        self._ensure_open()
        if runtime_future_gt_used is not False:
            raise RuntimeCausalError("future frame audit must state runtime_future_gt_used=false")
        payload = dict(audit)
        self._reject_gt(payload)
        frame = int(frame)
        if frame <= self.event_frame:
            raise RuntimeCausalError("future frame must be event_frame+1 or later")
        if frame not in self.future_frames:
            self.future_frames.append(frame)
            self.future_frames.sort()
        return {"frame": frame, "runtime_future_gt_used": False, **payload}

    def finalize(self, *, expected_first_future_frame: int | None = None) -> dict[str, Any]:
        self._ensure_open()
        if self.spatial_correction is None:
            raise RuntimeCausalError("cannot finalize without spatial correction")
        if self.memory_write is None:
            raise RuntimeCausalError("cannot finalize without memory write")
        first_read = self.memory_reads[0]["frame"] if self.memory_reads else None
        if first_read is not None and first_read != self.event_frame + 1:
            raise RuntimeCausalError("first memory read must be event_frame+1")
        if expected_first_future_frame is not None and first_read != int(expected_first_future_frame):
            raise RuntimeCausalError("first memory read does not match expected future boundary")
        self._finalized = True
        result = self.audit()
        result["status"] = "PASS_RUNTIME_CAUSAL_BOUNDARY"
        return result

    def audit(self) -> dict[str, Any]:
        return {
            "schema_version": "N72R1_RUNTIME_CAUSAL_AUDIT_V1",
            "event_id": self.event_id,
            "action_type": self.action_type,
            "event_frame": self.event_frame,
            "session_id": self.session_id,
            "spatial_correction": None if self.spatial_correction is None else dict(self.spatial_correction),
            "memory_write": None if self.memory_write is None else dict(self.memory_write),
            "memory_reads": [dict(item) for item in self.memory_reads],
            "future_frames": list(self.future_frames),
            "event_frame_read": False,
            "current_frame_write_hidden": self.memory_write is not None and bool(self.memory_write.get("current_frame_write_hidden")),
            "first_visible_frame": None if self.memory_write is None else self.event_frame + 1,
            "runtime_future_gt_used": False,
            "finalized": self._finalized,
            "audit_sha256": _digest({
                "event_id": self.event_id,
                "event_frame": self.event_frame,
                "spatial_correction": self.spatial_correction,
                "memory_write": self.memory_write,
                "memory_reads": self.memory_reads,
                "future_frames": self.future_frames,
            }),
        }


__all__ = ["RuntimeCausalError", "RuntimeCausalGuard"]
