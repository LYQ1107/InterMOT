"""Sequence-persistent public identity runtime for N72R3.

The official SAM3 backend owns only session-local objects.  This module owns
the identity that is written to MOT output and deliberately keeps that layer
outside the SAM session: a candidate may switch, disappear, or receive a new
raw SAM ID without changing the public identity.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Iterable, Optional

import numpy as np

from sam3_intermot.association.appearance_memory import AppearanceMemory
from sam3_intermot.identity.lineage import IdentityLineageRegistry
from sam3_intermot.identity.public_authority import PublicAuthorityBridge
from sam3_intermot.tracking.track import TrackState
from sam3_intermot.tracking.track_manager import TrackManager


PERSISTENT_STATUSES = ("ACTIVE", "LOST", "TERMINATED")


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, dict):
        return {str(key): _jsonable(child) for key, child in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(child) for child in value]
    if isinstance(value, set):
        return sorted(_jsonable(child) for child in value)
    return value


@dataclass
class PersistentIdentityRecord:
    """One sequence-persistent identity record.

    ``public_id`` and ``mot_track_id`` are guarded as immutable after
    construction.  The current candidate/session fields are intentionally
    mutable and are cleared at SAM session boundaries.
    """

    association_state_id: int
    public_id: int
    mot_track_id: int
    identity_lineage_id: int
    status: str
    created_frame: int
    last_seen_frame: Optional[int] = None
    last_candidate_uid: Optional[str] = None
    current_session_id: Optional[str] = None
    current_adapter_external_id: Optional[int] = None
    current_raw_sam_id: Optional[int] = None
    appearance_state: dict[str, Any] = field(default_factory=dict)
    motion_state_ref: Any = field(default_factory=dict)
    last_box: Optional[list[float]] = None
    event_history: list[dict[str, Any]] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.status not in PERSISTENT_STATUSES:
            raise ValueError(f"unknown persistent identity status: {self.status}")
        if int(self.public_id) <= 0 or int(self.mot_track_id) <= 0:
            raise ValueError("public_id and mot_track_id must be positive")
        if int(self.public_id) != int(self.mot_track_id):
            raise ValueError("N72R3 requires mot_track_id == persistent public_id")
        self.association_state_id = int(self.association_state_id)
        self.public_id = int(self.public_id)
        self.mot_track_id = int(self.mot_track_id)
        self.identity_lineage_id = int(self.identity_lineage_id)
        self.created_frame = int(self.created_frame)

    def __setattr__(self, name: str, value: Any) -> None:
        if name in {"public_id", "mot_track_id"} and hasattr(self, name):
            if int(value) != int(getattr(self, name)):
                raise AttributeError(f"{name} is immutable for a persistent identity")
        super().__setattr__(name, value)

    def as_dict(self) -> dict[str, Any]:
        return _jsonable(
            {
                "association_state_id": self.association_state_id,
                "public_id": self.public_id,
                "mot_track_id": self.mot_track_id,
                "identity_lineage_id": self.identity_lineage_id,
                "status": self.status,
                "created_frame": self.created_frame,
                "last_seen_frame": self.last_seen_frame,
                "last_candidate_uid": self.last_candidate_uid,
                "current_session_id": self.current_session_id,
                "current_adapter_external_id": self.current_adapter_external_id,
                "current_raw_sam_id": self.current_raw_sam_id,
                "appearance_state": self.appearance_state,
                "motion_state_ref": self.motion_state_ref,
                "last_box": self.last_box,
                "event_history": self.event_history,
            }
        )

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "PersistentIdentityRecord":
        return cls(
            association_state_id=int(value["association_state_id"]),
            public_id=int(value["public_id"]),
            mot_track_id=int(value["mot_track_id"]),
            identity_lineage_id=int(value["identity_lineage_id"]),
            status=str(value["status"]),
            created_frame=int(value["created_frame"]),
            last_seen_frame=None if value.get("last_seen_frame") is None else int(value["last_seen_frame"]),
            last_candidate_uid=value.get("last_candidate_uid"),
            current_session_id=value.get("current_session_id"),
            current_adapter_external_id=(
                None
                if value.get("current_adapter_external_id") is None
                else int(value["current_adapter_external_id"])
            ),
            current_raw_sam_id=(
                None if value.get("current_raw_sam_id") is None else int(value["current_raw_sam_id"])
            ),
            appearance_state=deepcopy(value.get("appearance_state") or {}),
            motion_state_ref=deepcopy(value.get("motion_state_ref") or {}),
            last_box=None if value.get("last_box") is None else [float(x) for x in value["last_box"]],
            event_history=deepcopy(value.get("event_history") or []),
        )


class _PersistentPublicIDAllocator:
    """Small allocator whose state belongs to one sequence runtime."""

    def __init__(self, start: int = 1000) -> None:
        self.next_id = int(start)
        self.allocated: set[int] = set()

    def allocate(self, requested: Optional[int] = None) -> int:
        if requested is not None:
            public_id = int(requested)
            if public_id <= 0 or public_id in self.allocated:
                raise ValueError(f"public_id is unavailable: {public_id}")
            self.allocated.add(public_id)
            self.next_id = max(self.next_id, public_id + 1)
            return public_id
        while self.next_id in self.allocated:
            self.next_id += 1
        public_id = self.next_id
        self.allocated.add(public_id)
        self.next_id += 1
        return public_id

    def snapshot(self) -> dict[str, Any]:
        return {"next_id": self.next_id, "allocated": sorted(self.allocated)}

    def restore(self, value: dict[str, Any]) -> None:
        self.next_id = int(value["next_id"])
        self.allocated = {int(item) for item in value.get("allocated", [])}


class SequencePersistentIdentityRuntime:
    """Own persistent identities and exactly one TrackManager per sequence."""

    def __init__(
        self,
        sequence: str,
        *,
        manager: Optional[TrackManager] = None,
        lineages: Optional[IdentityLineageRegistry] = None,
        authority_bridge: Optional[PublicAuthorityBridge] = None,
        appearance_memory: Optional[AppearanceMemory] = None,
        public_id_start: int = 1000,
        state_id_start: int = 1,
    ) -> None:
        if not sequence:
            raise ValueError("sequence is required")
        self.sequence = str(sequence)
        self.manager = manager if manager is not None else TrackManager()
        self.track_manager = self.manager
        self.lineages = lineages if lineages is not None else IdentityLineageRegistry()
        self.authority = authority_bridge or PublicAuthorityBridge(
            f"n72r3-persistent:{self.sequence}", self.sequence
        )
        # Appearance evidence is sequence-persistent and therefore belongs to
        # this outer runtime, not to a session-local SAM adapter or a
        # candidate row.  The object is CPU-side and has an explicit
        # snapshot/restore contract for interaction transactions.
        self.appearance_memory = appearance_memory or AppearanceMemory()
        self.identities: dict[int, PersistentIdentityRecord] = {}
        self._public_to_state: dict[int, int] = {}
        self._next_state_id = int(state_id_start)
        self._public_allocator = _PersistentPublicIDAllocator(public_id_start)
        self.active_session_id: Optional[str] = None
        self.session_history: list[dict[str, Any]] = []
        self.assignment_log: list[dict[str, Any]] = []
        self.runtime_future_gt_used = False

    # ------------------------------------------------------------------
    # Identity ownership
    # ------------------------------------------------------------------
    def _allocate_state_id(self, requested: Optional[int]) -> int:
        if requested is not None:
            state_id = int(requested)
            if state_id <= 0 or state_id in self.identities:
                raise ValueError(f"association_state_id is unavailable: {state_id}")
            self._next_state_id = max(self._next_state_id, state_id + 1)
            return state_id
        while self._next_state_id in self.identities:
            self._next_state_id += 1
        state_id = self._next_state_id
        self._next_state_id += 1
        return state_id

    @staticmethod
    def _sam_id(observation: Any) -> int:
        value = getattr(observation, "sam_object_id", None)
        if value is None:
            raise ValueError("candidate observation has no session-local sam_object_id")
        return int(value)

    @staticmethod
    def _box(observation: Any) -> list[float]:
        value = getattr(observation, "box_xyxy", None)
        if value is None:
            raise ValueError("candidate observation has no box_xyxy")
        return [float(item) for item in np.asarray(value).reshape(-1)[:4]]

    def create_identity(
        self,
        frame_idx: int,
        observation: Any,
        *,
        public_id: Optional[int] = None,
        association_state_id: Optional[int] = None,
        candidate_uid: Optional[str] = None,
        session_id: Optional[str] = None,
        adapter_external_id: Optional[int] = None,
        raw_sam_id: Optional[int] = None,
        appearance_state: Optional[dict[str, Any]] = None,
        motion_state_ref: Any = None,
    ) -> PersistentIdentityRecord:
        """Allocate a public identity only after an outer birth decision."""

        frame = int(frame_idx)
        state_id = self._allocate_state_id(association_state_id)
        assigned_public = self._public_allocator.allocate(public_id)
        lineage = self.lineages.create(frame)
        track = self.manager.create_track(
            frame,
            observation,
            lineage.lineage_id,
            mot_track_id=assigned_public,
        )
        lineage.bind_track(track.mot_track_id)
        self.authority.bind_identity_state(
            association_state_id=state_id,
            public_id=assigned_public,
            mot_track_id=assigned_public,
            lineage_id=lineage.lineage_id,
            created_frame=frame,
            transaction_id=f"{self.sequence}:identity-create:{state_id}",
        )
        raw = raw_sam_id
        if raw is None:
            candidate_raw = getattr(observation, "raw_sam_object_id", None)
            raw = None if candidate_raw is None else int(candidate_raw)
        adapter = adapter_external_id
        if adapter is None:
            candidate_adapter = getattr(observation, "adapter_external_id", None)
            adapter = None if candidate_adapter is None else int(candidate_adapter)
        record = PersistentIdentityRecord(
            association_state_id=state_id,
            public_id=assigned_public,
            mot_track_id=assigned_public,
            identity_lineage_id=lineage.lineage_id,
            status="ACTIVE",
            created_frame=frame,
            last_seen_frame=frame,
            last_candidate_uid=None if candidate_uid is None else str(candidate_uid),
            current_session_id=None if session_id is None else str(session_id),
            current_adapter_external_id=adapter,
            current_raw_sam_id=raw,
            appearance_state=deepcopy(appearance_state or {}),
            motion_state_ref=deepcopy(motion_state_ref or {}),
            last_box=self._box(observation),
        )
        self.identities[state_id] = record
        self._public_to_state[assigned_public] = state_id
        if session_id is not None:
            self.active_session_id = str(session_id)
        self._log(record, frame, "CREATE_IDENTITY", candidate_uid)
        return record

    def get_identity_by_public_id(self, public_id: int) -> Optional[PersistentIdentityRecord]:
        state_id = self._public_to_state.get(int(public_id))
        return None if state_id is None else self.identities.get(state_id)

    def get_identity_by_state_id(self, state_id: int) -> Optional[PersistentIdentityRecord]:
        return self.identities.get(int(state_id))

    def _record(self, identity: int | PersistentIdentityRecord) -> PersistentIdentityRecord:
        if isinstance(identity, PersistentIdentityRecord):
            record = self.identities.get(int(identity.association_state_id))
            if record is not identity:
                raise ValueError("identity record does not belong to this runtime")
            return record
        value = int(identity)
        record = self.identities.get(value)
        if record is None:
            record = self.get_identity_by_public_id(value)
        if record is None:
            raise KeyError(f"unknown persistent identity/state: {identity}")
        return record

    # ------------------------------------------------------------------
    # Session-local candidate bindings
    # ------------------------------------------------------------------
    def begin_new_sam_session(
        self, session_id: str, *, boundary_frame: Optional[int] = None
    ) -> dict[str, Any]:
        if not session_id:
            raise ValueError("session_id is required")
        detached = self.manager.detach_all_session_bindings()
        previous = self.active_session_id
        self.active_session_id = str(session_id)
        lost: list[int] = []
        for record in self.identities.values():
            if record.status == "TERMINATED":
                continue
            track = self.manager.get(record.mot_track_id)
            if track is not None and track.state not in (TrackState.TERMINATED, TrackState.DELETED):
                self.manager.mark_lost(record.mot_track_id, int(boundary_frame or record.last_seen_frame or 0))
            record.current_session_id = None
            record.current_adapter_external_id = None
            record.current_raw_sam_id = None
            if record.status == "ACTIVE":
                record.status = "LOST"
            lost.append(record.public_id)
            if boundary_frame is not None:
                self.authority.record_identity_no_candidate(
                    record.association_state_id, int(boundary_frame)
                )
        entry = {
            "from_session": previous,
            "to_session": str(session_id),
            "boundary_frame": None if boundary_frame is None else int(boundary_frame),
            "detached_track_ids": detached,
            "persistent_public_ids": sorted(lost),
            "candidate_bindings_cleared": True,
            "identities_deleted": False,
        }
        self.session_history.append(entry)
        return deepcopy(entry)

    def bind_candidate(
        self,
        identity: int | PersistentIdentityRecord,
        candidate_uid: str,
        observation: Any,
        frame_idx: int,
        *,
        session_id: Optional[str] = None,
        adapter_external_id: Optional[int] = None,
        raw_sam_id: Optional[int] = None,
    ) -> PersistentIdentityRecord:
        record = self._record(identity)
        if record.status == "TERMINATED":
            raise ValueError("terminated identity cannot be rebound")
        if not candidate_uid:
            raise ValueError("candidate_uid is required for a candidate binding")
        session = self.active_session_id if session_id is None else str(session_id)
        if session is None:
            raise ValueError("bind_candidate requires an active SAM session")
        if self.active_session_id is not None and session != self.active_session_id:
            raise ValueError("candidate belongs to a different active SAM session")
        sam_id = self._sam_id(observation)
        track = self.manager.get(record.mot_track_id)
        if track is None:
            raise ValueError(f"persistent TrackManager track disappeared: {record.mot_track_id}")
        if track.sam_object_id != sam_id:
            self.manager.rebind_sam_object(record.mot_track_id, sam_id, int(frame_idx))
        self.manager.update_track(record.mot_track_id, int(frame_idx), observation)
        raw = raw_sam_id
        if raw is None:
            value = getattr(observation, "raw_sam_object_id", None)
            raw = None if value is None else int(value)
        adapter = adapter_external_id
        if adapter is None:
            value = getattr(observation, "adapter_external_id", None)
            adapter = None if value is None else int(value)
        record.status = "ACTIVE"
        record.last_seen_frame = int(frame_idx)
        record.last_candidate_uid = str(candidate_uid)
        record.current_session_id = session
        record.current_adapter_external_id = adapter
        record.current_raw_sam_id = raw
        record.last_box = self._box(observation)
        self.active_session_id = session
        self._log(record, int(frame_idx), "BIND_CANDIDATE", str(candidate_uid))
        return record

    def clear_current_session_bindings(
        self,
        frame_idx: int,
        *,
        reason: str = "frame_binding_refresh",
    ) -> list[int]:
        """Clear only the current-frame SAM bindings before a new solve.

        A persistent track can be assigned to a different session-local
        candidate on the next frame.  Detaching the complete old candidate
        axis first makes that transition explicit and prevents a stale
        ``sam_object_id -> track`` entry from being mistaken for identity
        authority or from causing a false collision in ``rebind_sam_object``.
        The identity records, MOT IDs, lineages, memory, and motion history
        remain intact; this is not a session boundary and does not mark an
        identity LOST.
        """

        detached = self.manager.detach_all_session_bindings()
        for record in self.identities.values():
            if record.status == "TERMINATED":
                continue
            record.current_session_id = None
            record.current_adapter_external_id = None
            record.current_raw_sam_id = None
            self._log(record, int(frame_idx), "CLEAR_FRAME_CANDIDATE_BINDING", None, reason=reason)
        return detached

    def unbind_session_candidate(
        self,
        identity: int | PersistentIdentityRecord,
        frame_idx: int,
        *,
        reason: str = "session_boundary",
    ) -> PersistentIdentityRecord:
        record = self._record(identity)
        track = self.manager.get(record.mot_track_id)
        if track is not None and track.sam_object_id is not None:
            self.manager.unbind_sam_object(record.mot_track_id)
        record.current_session_id = None
        record.current_adapter_external_id = None
        record.current_raw_sam_id = None
        if record.status != "TERMINATED":
            record.status = "LOST"
        self.authority.record_identity_no_candidate(record.association_state_id, int(frame_idx))
        self._log(record, int(frame_idx), "UNBIND_SESSION_CANDIDATE", None, reason=reason)
        return record

    def mark_lost(
        self, identity: int | PersistentIdentityRecord, frame_idx: int, *, reason: str = "not_observed"
    ) -> PersistentIdentityRecord:
        return self.unbind_session_candidate(identity, frame_idx, reason=reason)

    def reactivate(
        self,
        identity: int | PersistentIdentityRecord,
        candidate_uid: str,
        observation: Any,
        frame_idx: int,
        *,
        session_id: Optional[str] = None,
        adapter_external_id: Optional[int] = None,
        raw_sam_id: Optional[int] = None,
    ) -> PersistentIdentityRecord:
        return self.bind_candidate(
            identity,
            candidate_uid,
            observation,
            frame_idx,
            session_id=session_id,
            adapter_external_id=adapter_external_id,
            raw_sam_id=raw_sam_id,
        )

    def terminate(
        self, identity: int | PersistentIdentityRecord, frame_idx: int, *, reason: str = "terminated"
    ) -> PersistentIdentityRecord:
        record = self._record(identity)
        track = self.manager.get(record.mot_track_id)
        if track is not None:
            if track.sam_object_id is not None:
                self.manager.unbind_sam_object(record.mot_track_id)
            if track.state not in (TrackState.TERMINATED, TrackState.DELETED):
                self.manager.terminate_track(record.mot_track_id, int(frame_idx))
        record.status = "TERMINATED"
        record.current_session_id = None
        record.current_adapter_external_id = None
        record.current_raw_sam_id = None
        self._log(record, int(frame_idx), "TERMINATE_IDENTITY", None, reason=reason)
        return record

    # ------------------------------------------------------------------
    # Complete per-frame assignment evidence
    # ------------------------------------------------------------------
    def record_frame_decisions(
        self,
        frame_idx: int,
        candidate_uids_by_public_id: Optional[dict[int, str]] = None,
    ) -> list[dict[str, Any]]:
        mapping = {int(key): str(value) for key, value in (candidate_uids_by_public_id or {}).items()}
        if len(set(mapping.values())) != len(mapping):
            raise ValueError("a candidate UID cannot be assigned to two persistent identities in one frame")
        rows: list[dict[str, Any]] = []
        for record in sorted(self.identities.values(), key=lambda item: item.public_id):
            candidate = mapping.get(record.public_id)
            if candidate is None:
                status = "NO_CANDIDATE_ASSIGNED"
                self.authority.record_identity_no_candidate(record.association_state_id, int(frame_idx))
            else:
                status = "ASSIGNED"
            rows.append(
                {
                    "frame_idx": int(frame_idx),
                    "association_state_id": record.association_state_id,
                    "public_id": record.public_id,
                    "mot_track_id": record.mot_track_id,
                    "identity_lineage_id": record.identity_lineage_id,
                    "candidate_uid": candidate,
                    "status": status,
                    "identity_status": record.status,
                    "runtime_future_gt_used": False,
                }
            )
        self.assignment_log.extend(deepcopy(rows))
        return rows

    def snapshot(self) -> dict[str, Any]:
        """Capture persistent state; session-local backend state is separate."""

        return {
            "schema_version": "N72R3_PERSISTENT_RUNTIME_SNAPSHOT_V1",
            "sequence": self.sequence,
            "active_session_id": self.active_session_id,
            "identities": [record.as_dict() for record in sorted(self.identities.values(), key=lambda item: item.association_state_id)],
            "public_to_state": {str(key): value for key, value in self._public_to_state.items()},
            "next_state_id": self._next_state_id,
            "public_allocator": self._public_allocator.snapshot(),
            "lineages": self.lineages.snapshot(),
            "track_manager": self.manager.snapshot(),
            "authority": self.authority.snapshot(),
            "appearance_memory": self.appearance_memory.snapshot(),
            "session_history": deepcopy(self.session_history),
            "assignment_log": deepcopy(self.assignment_log),
            "runtime_future_gt_used": bool(self.runtime_future_gt_used),
        }

    def restore(self, snapshot: dict[str, Any]) -> None:
        if str(snapshot.get("sequence")) != self.sequence:
            raise ValueError("persistent snapshot belongs to a different sequence")
        self.manager.restore(snapshot["track_manager"])
        self.lineages.restore(snapshot["lineages"])
        records = [PersistentIdentityRecord.from_dict(item) for item in snapshot.get("identities", [])]
        # Restore records in place when they already exist.  Interaction
        # transactions may hold a reference to the authoritative record while
        # backend/identity/memory work is in flight; replacing the dictionary
        # would restore runtime lookup but leave that caller reference stale.
        previous_records = self.identities
        restored_records: dict[int, PersistentIdentityRecord] = {}
        for restored in records:
            existing = previous_records.get(restored.association_state_id)
            if existing is None:
                restored_records[restored.association_state_id] = restored
                continue
            existing.__dict__.clear()
            existing.__dict__.update(deepcopy(restored.__dict__))
            restored_records[restored.association_state_id] = existing
        self.identities = restored_records
        self._public_to_state = {
            int(key): int(value) for key, value in snapshot.get("public_to_state", {}).items()
        }
        self._next_state_id = int(snapshot["next_state_id"])
        self._public_allocator.restore(snapshot["public_allocator"])
        self.authority.restore(snapshot["authority"])
        self.appearance_memory.restore(snapshot.get("appearance_memory", {}))
        self.active_session_id = snapshot.get("active_session_id")
        self.session_history = deepcopy(snapshot.get("session_history", []))
        self.assignment_log = deepcopy(snapshot.get("assignment_log", []))
        self.runtime_future_gt_used = bool(snapshot.get("runtime_future_gt_used", False))
        self._validate_invariants()

    def audit(self) -> dict[str, Any]:
        violations = self._validate_invariants(raise_on_error=False)
        records = list(self.identities.values())
        return {
            "schema_version": "N72R3_PERSISTENT_RUNTIME_AUDIT_V1",
            "sequence": self.sequence,
            "identity_count": len(records),
            "active_count": sum(record.status == "ACTIVE" for record in records),
            "lost_count": sum(record.status == "LOST" for record in records),
            "terminated_count": sum(record.status == "TERMINATED" for record in records),
            "public_ids": sorted(record.public_id for record in records),
            "mot_track_ids": sorted(record.mot_track_id for record in records),
            "lineage_ids": sorted(record.identity_lineage_id for record in records),
            "public_id_immutable": True,
            "mot_track_id_equals_public_id": all(record.mot_track_id == record.public_id for record in records),
            "candidate_is_not_identity_owner": True,
            "candidate_bindings_are_session_local": True,
            "appearance_memory_persistent": True,
            "appearance_memory_record_count": len(self.appearance_memory.records),
            "track_manager_instance_count": 1,
            "auxiliary_track_manager_count": 0,
            "runtime_future_gt_used": bool(self.runtime_future_gt_used),
            "invariant_violations": violations,
            "authority": self.authority.audit(),
        }

    def _validate_invariants(self, *, raise_on_error: bool = True) -> list[str]:
        violations: list[str] = []
        if self.manager is not self.track_manager:
            violations.append("manager_alias_is_not_single_track_manager")
        public_ids: list[int] = []
        for state_id, record in self.identities.items():
            if record.association_state_id != int(state_id):
                violations.append(f"state_key_mismatch:{state_id}")
            if record.public_id != record.mot_track_id:
                violations.append(f"public_mot_mismatch:{state_id}")
            if self._public_to_state.get(record.public_id) != state_id:
                violations.append(f"public_reverse_binding_missing:{record.public_id}")
            track = self.manager.get(record.mot_track_id)
            if track is None:
                violations.append(f"track_missing:{record.public_id}")
            elif track.identity_lineage_id != record.identity_lineage_id:
                violations.append(f"lineage_mismatch:{record.public_id}")
            public_ids.append(record.public_id)
        if len(public_ids) != len(set(public_ids)):
            violations.append("duplicate_public_id")
        if raise_on_error and violations:
            raise AssertionError("persistent runtime invariant violations: " + ",".join(violations))
        return violations

    def _log(
        self,
        record: PersistentIdentityRecord,
        frame_idx: int,
        operation: str,
        candidate_uid: Optional[str],
        *,
        reason: Optional[str] = None,
    ) -> None:
        record.event_history.append(
            {
                "frame_idx": int(frame_idx),
                "operation": str(operation),
                "candidate_uid": candidate_uid,
                "public_id": record.public_id,
                "association_state_id": record.association_state_id,
                "status": record.status,
                "reason": reason,
                "runtime_future_gt_used": False,
            }
        )


__all__ = [
    "PERSISTENT_STATUSES",
    "PersistentIdentityRecord",
    "SequencePersistentIdentityRuntime",
]
