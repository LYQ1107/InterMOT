"""Persistent public-ID state for N72R14 global association.

This module is deliberately independent from the legacy ``IdentityState``
class.  The legacy class uses a solver-local ``pid`` and is therefore not an
authority for a persistent public identity.  N72R14 keeps the two axes
explicit and stores the state bank by immutable ``public_id``.

The bank is a causal runtime object: callers initialize it from observations
available at the event frame, apply the human write after the current-frame
transaction, and call :meth:`update_from_solver` only after each future exact
assignment.  It never reads GT and it never invents a feature for a missing
candidate feature.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass, field
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from .appearance_memory import AppearanceMemory


FEATURE_DIM = 512


def _finite_box(value: Any, label: str) -> list[float]:
    box = np.asarray(value, dtype=np.float64).reshape(-1)
    if box.size != 4 or not np.isfinite(box).all():
        raise ValueError(f"{label} must be four finite xyxy coordinates")
    return [float(item) for item in box]


def _feature(value: Any, label: str) -> np.ndarray | None:
    if value is None:
        return None
    array = np.asarray(value, dtype=np.float32).reshape(-1)
    if array.size == 0:
        return None
    if array.size != FEATURE_DIM or not np.isfinite(array).all():
        raise ValueError(f"{label} must be a finite {FEATURE_DIM}-D feature")
    if float(np.linalg.norm(array)) <= 1.0e-6:
        return None
    return (array / float(np.linalg.norm(array))).astype(np.float32)


def _candidate_raw(candidate: Mapping[str, Any]) -> int | None:
    for key in (
        "official_raw_sam_id",
        "raw_sam_id",
        "raw_native_id",
        "native_tid",
        "adapter_external_id",
    ):
        value = candidate.get(key)
        if value is not None:
            return int(value)
    return None


def _candidate_scope(candidate: Mapping[str, Any]) -> str | None:
    value = candidate.get("native_scope", candidate.get("native_tid_scope"))
    return None if value in (None, "") else str(value)


@dataclass
class PublicAssociationMotionState:
    """Motion/provenance state for one explicit public identity.

    ``association_state_id`` is the solver-local state axis.  It is kept
    separate from ``public_id`` even when a historical artifact happens to
    use equal numeric values.
    """

    association_state_id: int
    public_id: int
    last_box: list[float] | None = None
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(2, dtype=np.float64))
    last_seen_frame: int | None = None
    previous_raw_sam_id: int | None = None
    previous_native_scope: str | None = None
    lost_age: int = 0
    assignment_count: int = 0

    def __post_init__(self) -> None:
        self.association_state_id = int(self.association_state_id)
        self.public_id = int(self.public_id)
        if self.association_state_id <= 0 or self.public_id <= 0:
            raise ValueError("association_state_id and public_id must be positive")
        self.last_box = None if self.last_box is None else _finite_box(self.last_box, "last_box")
        velocity = np.asarray(self.velocity, dtype=np.float64).reshape(-1)
        if velocity.size != 2 or not np.isfinite(velocity).all():
            raise ValueError("velocity must be a finite 2-D vector")
        self.velocity = velocity.copy()
        self.last_seen_frame = None if self.last_seen_frame is None else int(self.last_seen_frame)
        self.previous_raw_sam_id = (
            None if self.previous_raw_sam_id is None else int(self.previous_raw_sam_id)
        )
        self.previous_native_scope = (
            None if self.previous_native_scope in (None, "") else str(self.previous_native_scope)
        )
        self.lost_age = max(0, int(self.lost_age))
        self.assignment_count = max(0, int(self.assignment_count))

    def to_dict(self) -> dict[str, Any]:
        return {
            "association_state_id": int(self.association_state_id),
            "public_id": int(self.public_id),
            "last_box": None if self.last_box is None else list(self.last_box),
            "velocity": self.velocity.astype(np.float64).tolist(),
            "last_seen_frame": self.last_seen_frame,
            "previous_raw_sam_id": self.previous_raw_sam_id,
            "previous_native_scope": self.previous_native_scope,
            "lost_age": int(self.lost_age),
            "assignment_count": int(self.assignment_count),
        }


class PersistentPublicAssociationBank:
    """Causal appearance and motion state bank keyed by ``public_id``."""

    def __init__(
        self,
        *,
        appearance_memory: AppearanceMemory | None = None,
        feat_dim: int = FEATURE_DIM,
        human_weight: float = 1.0,
        machine_weight: float = 0.35,
        decay_frames: float = 120.0,
        min_machine_confidence: float = 0.5,
        reliability_threshold: float = 0.0,
        anchor_cap: int = 8,
        negative_cap: int = 16,
    ) -> None:
        self.appearance_memory = appearance_memory or AppearanceMemory(
            feat_dim=int(feat_dim),
            human_weight=float(human_weight),
            machine_weight=float(machine_weight),
            decay_frames=float(decay_frames),
            min_machine_confidence=float(min_machine_confidence),
            reliability_threshold=float(reliability_threshold),
            anchor_cap=int(anchor_cap),
            negative_cap=int(negative_cap),
        )
        self.motion_states: dict[int, PublicAssociationMotionState] = {}
        self.state_to_public: dict[int, int] = {}

    def ensure_state(
        self,
        association_state_id: int,
        public_id: int,
        *,
        last_box: Sequence[float] | None = None,
        last_seen_frame: int | None = None,
        previous_raw_sam_id: int | None = None,
        previous_native_scope: str | None = None,
    ) -> PublicAssociationMotionState:
        """Create or validate an explicit state/public authority pair."""

        state_id, public = int(association_state_id), int(public_id)
        old_public = self.state_to_public.get(state_id)
        if old_public is not None and old_public != public:
            raise ValueError(
                f"association state {state_id} is already bound to public ID {old_public}, not {public}"
            )
        old_state = self.motion_states.get(public)
        if old_state is not None and old_state.association_state_id != state_id:
            raise ValueError(
                f"public ID {public} is already bound to association state "
                f"{old_state.association_state_id}, not {state_id}"
            )
        if old_state is None:
            old_state = PublicAssociationMotionState(
                association_state_id=state_id,
                public_id=public,
                last_box=None if last_box is None else list(last_box),
                last_seen_frame=last_seen_frame,
                previous_raw_sam_id=previous_raw_sam_id,
                previous_native_scope=previous_native_scope,
            )
            self.motion_states[public] = old_state
        elif last_box is not None and old_state.last_box is None:
            old_state.last_box = _finite_box(last_box, "last_box")
            old_state.last_seen_frame = None if last_seen_frame is None else int(last_seen_frame)
            old_state.previous_raw_sam_id = (
                None if previous_raw_sam_id is None else int(previous_raw_sam_id)
            )
            old_state.previous_native_scope = (
                None if previous_native_scope in (None, "") else str(previous_native_scope)
            )
        self.state_to_public[state_id] = public
        return old_state

    def state_for_pair(self, association_state_id: int, public_id: int) -> PublicAssociationMotionState:
        return self.ensure_state(association_state_id, public_id)

    def states_for_pairs(self, pairs: Sequence[tuple[int, int]]) -> list[PublicAssociationMotionState]:
        """Return states in the exact supplied solver-axis order."""

        seen_state: set[int] = set()
        seen_public: set[int] = set()
        result: list[PublicAssociationMotionState] = []
        for state_id, public_id in pairs:
            state_id, public_id = int(state_id), int(public_id)
            if state_id in seen_state or public_id in seen_public:
                raise ValueError("state/public solver axis contains duplicates")
            seen_state.add(state_id)
            seen_public.add(public_id)
            result.append(self.ensure_state(state_id, public_id))
        return result

    @staticmethod
    def _assignment_rows(solver: Mapping[str, Any]) -> list[Mapping[str, Any]]:
        rows = solver.get("assignment_rows")
        if not isinstance(rows, list):
            raise ValueError("exact solver artifact lacks assignment_rows")
        return rows

    def initialize_from_event_frame(
        self,
        *,
        event_frame: int,
        candidate_rows: Sequence[Mapping[str, Any]],
        solver: Mapping[str, Any] | None = None,
        association_state_axis: Sequence[int] | None = None,
        public_id_axis: Sequence[int] | None = None,
        target_public_id: int | None = None,
        target_anchor: Sequence[float] | None = None,
        target_box: Sequence[float] | None = None,
        target_candidate: Mapping[str, Any] | None = None,
        event_id: str | None = None,
    ) -> dict[str, Any]:
        """Initialize from event-frame observations and then write human evidence.

        The optional solver/axes are copied from the frozen event-frame
        artifact; no GT-derived identity is consulted.  The human target write
        is intentionally last, so a caller can audit that the event frame did
        not read the newly written memory.
        """

        frame = int(event_frame)
        by_uid = {str(row.get("candidate_uid")): row for row in candidate_rows}
        if len(by_uid) != len(candidate_rows) or None in by_uid:
            raise ValueError("event-frame candidate UIDs are not unique/complete")
        assignments = [] if solver is None else self._assignment_rows(solver)
        if association_state_axis is None and solver is not None:
            association_state_axis = solver.get("association_state_axis", [])
        if public_id_axis is None and solver is not None:
            public_id_axis = solver.get("public_id_axis", [])
        if association_state_axis is None or public_id_axis is None:
            raise ValueError("event-frame initialization needs explicit state and public axes")
        states = [int(value) for value in association_state_axis]
        publics = [int(value) for value in public_id_axis]
        if len(states) != len(publics) or len(states) != len(set(states)) or len(publics) != len(set(publics)):
            raise ValueError("event-frame state/public axes are invalid")
        state_by_public = dict(zip(publics, states))
        for state_id, public_id in zip(states, publics):
            self.ensure_state(state_id, public_id)
        assigned_publics: set[int] = set()
        for decision in assignments:
            public = decision.get("public_id")
            uid = decision.get("candidate_uid")
            if public is None or uid is None or str(decision.get("status")) != "ASSIGNED_TO_PUBLIC_ID":
                continue
            public = int(public)
            candidate = by_uid.get(str(uid))
            if candidate is None:
                raise ValueError(f"event-frame solver assignment lacks candidate {uid}")
            state_id = decision.get("association_state_id", state_by_public.get(public))
            if state_id is None or state_by_public.get(public) != int(state_id):
                raise ValueError("event-frame assignment does not preserve explicit authority axes")
            self._set_observation(
                self.ensure_state(int(state_id), public),
                candidate,
                frame,
                update_machine=True,
                assignment_increment=False,
            )
            assigned_publics.add(public)
        # Every active public axis gets a state even when its event-frame exact
        # solver result is NONE; that state starts LOST/unknown rather than
        # being fabricated from a row index.
        for public in publics:
            if public not in assigned_publics:
                self.motion_states[public].lost_age = 1

        human_written = False
        if target_public_id is not None:
            target_public = int(target_public_id)
            if target_public not in state_by_public:
                raise ValueError("human target is absent from explicit event-frame public axis")
            if target_anchor is None or target_box is None:
                raise ValueError("target human write requires explicit anchor feature and box")
            if target_candidate is not None:
                target_state = self.motion_states[target_public]
                target_state.previous_raw_sam_id = _candidate_raw(target_candidate)
                target_state.previous_native_scope = _candidate_scope(target_candidate)
            target_state = self.motion_states[target_public]
            target_state.last_box = _finite_box(target_box, "target human box")
            target_state.last_seen_frame = frame
            target_state.velocity = np.zeros(2, dtype=np.float64)
            target_state.lost_age = 0
            human_written = bool(
                self.appearance_memory.update_from_human(
                    target_public,
                    frame,
                    np.asarray(target_anchor, dtype=np.float32),
                    quality=1.0,
                    write_event_id=None if event_id is None else str(event_id),
                )
            )
        return {
            "event_frame": frame,
            "state_axis": states,
            "public_id_axis": publics,
            "assigned_public_ids": sorted(assigned_publics),
            "target_public_id": None if target_public_id is None else int(target_public_id),
            "human_memory_written": human_written,
            "event_frame_memory_read": False,
            "first_memory_visible_frame": frame + 1,
            "runtime_future_gt_used": False,
        }

    def _set_observation(
        self,
        state: PublicAssociationMotionState,
        candidate: Mapping[str, Any],
        frame: int,
        *,
        update_machine: bool,
        assignment_increment: bool,
    ) -> bool:
        box = _finite_box(candidate.get("box_xyxy", candidate.get("box")), "candidate box")
        old_box = state.last_box
        old_frame = state.last_seen_frame
        if old_box is not None and old_frame is not None and int(frame) > int(old_frame):
            old_center = np.asarray([(old_box[0] + old_box[2]) / 2.0, (old_box[1] + old_box[3]) / 2.0])
            new_center = np.asarray([(box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0])
            dt = max(1, int(frame) - int(old_frame))
            state.velocity = 0.8 * state.velocity + 0.2 * (new_center - old_center) / float(dt)
        state.last_box = box
        state.last_seen_frame = int(frame)
        state.previous_raw_sam_id = _candidate_raw(candidate)
        state.previous_native_scope = _candidate_scope(candidate)
        state.lost_age = 0
        if assignment_increment:
            state.assignment_count += 1
        if not update_machine:
            return False
        feature = _feature(candidate.get("feature", candidate.get("embedding")), "candidate feature")
        if feature is None:
            return False
        confidence = candidate.get("confidence", candidate.get("presence_score", 0.0))
        try:
            confidence = float(confidence)
        except (TypeError, ValueError):
            return False
        return bool(
            self.appearance_memory.update_from_machine(
                state.public_id,
                int(frame),
                feature,
                confidence=confidence,
            )
        )

    def update_from_solver(
        self,
        *,
        frame: int,
        candidate_rows: Sequence[Mapping[str, Any]],
        solver: Mapping[str, Any],
        none_score: float = 0.0,
    ) -> dict[str, Any]:
        """Update every public state after one exact solver frame."""

        by_uid = {str(row.get("candidate_uid")): row for row in candidate_rows}
        if len(by_uid) != len(candidate_rows) or None in by_uid:
            raise ValueError("solver update candidate UIDs are not unique")
        assigned: dict[int, Mapping[str, Any]] = {}
        assigned_scores: dict[int, float] = {}
        for decision in self._assignment_rows(solver):
            if str(decision.get("status")) != "ASSIGNED_TO_PUBLIC_ID":
                continue
            public = decision.get("public_id")
            uid = decision.get("candidate_uid")
            if public is None or uid is None:
                continue
            public = int(public)
            if public in assigned:
                raise ValueError(f"public ID {public} assigned more than once")
            candidate = by_uid.get(str(uid))
            if candidate is None:
                raise ValueError(f"solver update candidate missing {uid}")
            assigned[public] = candidate
            assigned_scores[public] = float(decision.get("score", none_score))
        machine_writes: list[int] = []
        for public, state in self.motion_states.items():
            candidate = assigned.get(int(public))
            if candidate is None:
                state.lost_age += 1
                continue
            wrote = self._set_observation(
                state,
                candidate,
                int(frame),
                update_machine=bool(assigned_scores[public] > float(none_score)),
                assignment_increment=True,
            )
            if wrote:
                machine_writes.append(int(public))
        return {
            "frame": int(frame),
            "assigned_public_ids": sorted(int(value) for value in assigned),
            "machine_memory_write_public_ids": sorted(machine_writes),
            "unassigned_public_ids": sorted(int(value) for value in self.motion_states if value not in assigned),
            "runtime_future_gt_used": False,
        }

    def compact_snapshot(self, frame: int | None = None) -> list[dict[str, Any]]:
        """Return bounded per-public audit state, not full memory vectors."""

        records = self.appearance_memory.records
        output: list[dict[str, Any]] = []
        for public in sorted(self.motion_states):
            state = self.motion_states[public]
            record = records.get(int(public))
            human_count = 0 if record is None else len(record.positive)
            machine_count = 0 if record is None else int(record.write_count)
            output.append(
                {
                    "association_state_id": int(state.association_state_id),
                    "public_id": int(state.public_id),
                    "last_seen_frame": state.last_seen_frame,
                    "lost_age": int(state.lost_age),
                    "previous_raw_sam_id": state.previous_raw_sam_id,
                    "previous_native_scope": state.previous_native_scope,
                    "appearance_present": bool(record is not None and (record.prototype is not None or record.positive)),
                    "human_anchor_count": int(human_count),
                    "machine_write_count": int(machine_count),
                    "snapshot_frame": None if frame is None else int(frame),
                }
            )
        return output

    def snapshot(self) -> dict[str, Any]:
        return {
            "schema_version": "N72R14_PERSISTENT_PUBLIC_ASSOCIATION_BANK_V1",
            "state_to_public": {str(k): int(v) for k, v in sorted(self.state_to_public.items())},
            "motion_states": {str(k): value.to_dict() for k, value in sorted(self.motion_states.items())},
            "appearance_memory": self.appearance_memory.snapshot(),
        }


__all__ = [
    "FEATURE_DIM",
    "PublicAssociationMotionState",
    "PersistentPublicAssociationBank",
]
