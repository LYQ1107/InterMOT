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
import copy
from dataclasses import dataclass, field
from typing import Any, Mapping

from sam3_intermot.backend.sam3_state_snapshot import (
    restore_continuation_state,
    snapshot_continuation_state,
)


class RuntimeCausalError(ValueError):
    """Raised when an event transaction violates the causal contract."""


class RuntimeInteractionError(RuntimeError):
    """Raised when an interaction cannot be committed atomically."""


class RuntimeInteractionRollbackError(RuntimeInteractionError):
    """Raised when a failed interaction also cannot be fully restored."""


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


def _capture_component(obj: Any) -> tuple[str, Any]:
    """Capture one mutable CPU-side component without losing object identity."""

    if obj is None:
        return "none", None
    snapshot = getattr(obj, "snapshot", None)
    restore = getattr(obj, "restore", None)
    if callable(snapshot) and callable(restore):
        return "snapshot_restore", copy.deepcopy(snapshot())
    serialize = getattr(obj, "serialize", None)
    deserialize = getattr(type(obj), "deserialize", None)
    if callable(serialize) and callable(deserialize):
        return "serialize_deserialize", copy.deepcopy(serialize())
    if hasattr(obj, "__dict__"):
        return "dict", copy.deepcopy(vars(obj))
    raise RuntimeInteractionError(
        f"component {type(obj).__name__} has no snapshot/restore contract"
    )


def _restore_component(obj: Any, kind: str, payload: Any) -> None:
    if kind == "none":
        return
    if kind == "snapshot_restore":
        obj.restore(copy.deepcopy(payload))
        return
    if kind == "serialize_deserialize":
        restored = type(obj).deserialize(copy.deepcopy(payload))
        if not hasattr(obj, "__dict__") or not hasattr(restored, "__dict__"):
            raise RuntimeInteractionError(
                f"serialized component {type(obj).__name__} is not in-place restorable"
            )
        vars(obj).clear()
        vars(obj).update(copy.deepcopy(vars(restored)))
        return
    if kind == "dict":
        if not hasattr(obj, "__dict__"):
            raise RuntimeInteractionError(f"component {type(obj).__name__} has no __dict__")
        vars(obj).clear()
        vars(obj).update(copy.deepcopy(payload))
        return
    raise RuntimeInteractionError(f"unknown component snapshot kind: {kind}")


def _capture_backend(backend: Any) -> tuple[str, Any]:
    """Use the pinned SAM3 continuation snapshot whenever a real session exists."""

    predictor = getattr(backend, "_predictor", None)
    session_id = getattr(backend, "_session_id", None)
    if predictor is not None and session_id is not None:
        try:
            return "sam3_continuation", snapshot_continuation_state(backend)
        except Exception as exc:  # pragma: no cover - depends on live SAM3 state
            raise RuntimeInteractionError(
                f"official backend continuation snapshot failed: {exc}"
            ) from exc
    if hasattr(backend, "__dict__"):
        return "python_object", copy.deepcopy(vars(backend))
    raise RuntimeInteractionError(
        f"backend {type(backend).__name__} has no continuation snapshot contract"
    )


def _restore_backend(backend: Any, kind: str, payload: Any) -> None:
    if kind == "sam3_continuation":
        try:
            restore_continuation_state(backend, payload)
        except Exception as exc:  # pragma: no cover - depends on live SAM3 state
            raise RuntimeInteractionError(
                f"official backend continuation restore failed: {exc}"
            ) from exc
        return
    if kind == "python_object":
        if not hasattr(backend, "__dict__"):
            raise RuntimeInteractionError(f"backend {type(backend).__name__} has no __dict__")
        vars(backend).clear()
        vars(backend).update(copy.deepcopy(payload))
        return
    raise RuntimeInteractionError(f"unknown backend snapshot kind: {kind}")


@dataclass
class RuntimeInteractionTransaction:
    """Atomic backend → identity → memory transaction for one event.

    The transaction owns no identity policy and never infers a public ID.  The
    caller supplies already-authoritative identity mutations.  A real SAM3
    backend is captured through ``sam3_state_snapshot``; all other mutable
    components are restored in place so existing references remain valid.
    """

    backend: Any
    track_manager: Any = None
    lineages: Any = None
    persistent_runtime: Any = None
    state_manager: Any = None
    appearance_memory: Any = None
    public_authority: Any = None
    allocator: Any = None
    event_id: str = ""
    _backend_kind: str = field(init=False, default="")
    _backend_snapshot: Any = field(init=False, default=None)
    _component_snapshots: dict[str, tuple[str, Any]] = field(init=False, default_factory=dict)
    _phase_results: dict[str, Any] = field(init=False, default_factory=dict)
    _committed: bool = field(init=False, default=False)
    _rolled_back: bool = field(init=False, default=False)
    _rollback_errors: list[str] = field(init=False, default_factory=list)

    def __post_init__(self) -> None:
        if self.backend is None:
            raise RuntimeInteractionError("backend is required")
        if self.persistent_runtime is not None:
            self.track_manager = self.track_manager or getattr(self.persistent_runtime, "manager", None)
            self.lineages = self.lineages or getattr(self.persistent_runtime, "lineages", None)
            self.public_authority = self.public_authority or getattr(self.persistent_runtime, "authority", None)
            self.allocator = self.allocator or getattr(self.persistent_runtime, "_public_allocator", None)
            self.appearance_memory = self.appearance_memory or getattr(self.persistent_runtime, "appearance_memory", None)
        if self.state_manager is not None and self.appearance_memory is None:
            self.appearance_memory = getattr(self.state_manager, "appearance_memory", None)
        self._backend_kind, self._backend_snapshot = _capture_backend(self.backend)
        components = {
            "track_manager": self.track_manager,
            "lineages": self.lineages,
            "persistent_runtime": self.persistent_runtime,
            "state_manager": self.state_manager,
            "appearance_memory": self.appearance_memory,
            "public_authority": self.public_authority,
            "allocator": self.allocator,
        }
        for name, obj in components.items():
            if obj is None:
                self._component_snapshots[name] = ("none", None)
                continue
            kind, payload = _capture_component(obj)
            self._component_snapshots[name] = (kind, payload)

    def __enter__(self) -> "RuntimeInteractionTransaction":
        return self

    @staticmethod
    def _check_phase_result(phase: str, result: Any) -> Any:
        if result is False:
            raise RuntimeInteractionError(f"{phase} phase returned failure")
        if isinstance(result, Mapping) and result.get("accepted") is False:
            raise RuntimeInteractionError(
                f"{phase} phase returned rejected result: {result.get('reason', 'unknown')}"
            )
        return result

    def execute(
        self,
        backend_operation,
        identity_operation,
        memory_operation,
    ) -> dict[str, Any]:
        """Run all three phases and commit only if every phase succeeds."""

        if self._committed or self._rolled_back:
            raise RuntimeInteractionError("transaction is already finalized")
        try:
            self._phase_results["backend"] = self._check_phase_result(
                "backend", backend_operation()
            )
            self._phase_results["identity"] = self._check_phase_result(
                "identity", identity_operation()
            )
            self._phase_results["memory"] = self._check_phase_result(
                "memory", memory_operation()
            )
            self.commit()
            return self.audit(status="PASS_RUNTIME_INTERACTION_COMMITTED")
        except Exception as exc:
            try:
                self.rollback()
            except RuntimeInteractionRollbackError:
                raise
            raise RuntimeInteractionError(
                f"{self.event_id or 'interaction'} failed in atomic transaction: {exc}"
            ) from exc

    def commit(self) -> None:
        if self._committed or self._rolled_back:
            raise RuntimeInteractionError("transaction is already finalized")
        self._committed = True

    def rollback(self) -> None:
        if self._rolled_back:
            return
        if self._committed:
            raise RuntimeInteractionError("cannot roll back a committed transaction")
        errors: list[str] = []
        try:
            _restore_backend(self.backend, self._backend_kind, self._backend_snapshot)
        except Exception as exc:
            errors.append(f"backend:{exc}")
        restored: set[str] = set()
        for name, (kind, payload) in self._component_snapshots.items():
            if kind == "none" or kind == "alias":
                continue
            obj = getattr(self, name if name != "allocator" else "allocator", None)
            if obj is None:
                continue
            try:
                _restore_component(obj, kind, payload)
                restored.add(name)
            except Exception as exc:
                errors.append(f"{name}:{exc}")
        # Aliased component entries need no second restore; their canonical
        # object has already been restored in place.
        self._rollback_errors = errors
        self._rolled_back = True
        if errors:
            raise RuntimeInteractionRollbackError(
                "atomic rollback incomplete: " + "; ".join(errors)
            )

    def __exit__(self, exc_type, exc, traceback) -> bool:
        if exc_type is not None and not self._committed and not self._rolled_back:
            self.rollback()
            return False
        return False

    def audit(self, *, status: Optional[str] = None) -> dict[str, Any]:
        return {
            "schema_version": "N72R3_RUNTIME_INTERACTION_TRANSACTION_V1",
            "event_id": self.event_id or None,
            "status": status
            or ("PASS_RUNTIME_INTERACTION_COMMITTED" if self._committed else "ROLLED_BACK" if self._rolled_back else "OPEN"),
            "backend_snapshot_kind": self._backend_kind,
            "snapshotted_components": sorted(self._component_snapshots),
            "component_aliases": {},
            "phase_names": ["backend", "identity", "memory"],
            "completed_phases": sorted(self._phase_results),
            "committed": self._committed,
            "rolled_back": self._rolled_back,
            "rollback_errors": list(self._rollback_errors),
            "runtime_future_gt_used": False,
        }


__all__ = [
    "RuntimeCausalError",
    "RuntimeCausalGuard",
    "RuntimeInteractionError",
    "RuntimeInteractionRollbackError",
    "RuntimeInteractionTransaction",
]
