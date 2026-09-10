"""Trusted persistent public-ID state for the N72R15 repair.

N72R14 allowed the treatment solver's output to immediately become the next
machine observation.  This module keeps the public/state axes explicit, but
only commits a future machine observation when the frozen B0 solver and the
treatment solver agree on the same public-to-candidate binding.  The module
does not read GT and does not own an assignment solver.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
from typing import Any, Mapping, Sequence, Iterable

import numpy as np

from .appearance_memory import AppearanceMemory


FEATURE_DIM = 512


def _box(value: Any, label: str) -> list[float]:
    array = np.asarray(value, dtype=np.float64).reshape(-1)
    if array.size != 4 or not np.isfinite(array).all():
        raise ValueError(f"{label} must be four finite xyxy coordinates")
    return [float(item) for item in array]


def _feature(value: Any, label: str) -> np.ndarray | None:
    if value is None:
        return None
    array = np.asarray(value, dtype=np.float32).reshape(-1)
    if array.size == 0:
        return None
    if array.size != FEATURE_DIM or not np.isfinite(array).all():
        raise ValueError(f"{label} must be a finite {FEATURE_DIM}-D feature")
    norm = float(np.linalg.norm(array))
    if norm <= 1.0e-6:
        return None
    return (array / norm).astype(np.float32)


def _raw(candidate: Mapping[str, Any]) -> int | None:
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


def _scope(candidate: Mapping[str, Any]) -> str | None:
    value = candidate.get("native_scope", candidate.get("native_tid_scope"))
    return None if value in (None, "") else str(value)


@dataclass
class TrustedPublicState:
    """One persistent identity whose authority is an explicit public ID."""

    association_state_id: int
    public_id: int
    trusted_last_box: list[float] | None = None
    trusted_velocity: np.ndarray = field(default_factory=lambda: np.zeros(2, dtype=np.float64))
    trusted_last_seen_frame: int | None = None
    trusted_raw_sam_id: int | None = None
    trusted_native_scope: str | None = None
    trusted_assignment_count: int = 0
    trusted_machine_write_count: int = 0
    future_machine_write_count: int = 0
    human_write_count: int = 0
    trusted_lost_age: int = 0
    last_committed_candidate_uid: str | None = None
    last_committed_frame: int | None = None

    def __post_init__(self) -> None:
        self.association_state_id = int(self.association_state_id)
        self.public_id = int(self.public_id)
        if self.association_state_id <= 0 or self.public_id <= 0:
            raise ValueError("association_state_id and public_id must be positive")
        self.trusted_last_box = (
            None if self.trusted_last_box is None else _box(self.trusted_last_box, "trusted_last_box")
        )
        velocity = np.asarray(self.trusted_velocity, dtype=np.float64).reshape(-1)
        if velocity.size != 2 or not np.isfinite(velocity).all():
            raise ValueError("trusted_velocity must be a finite 2-D vector")
        self.trusted_velocity = velocity.copy()
        self.trusted_last_seen_frame = (
            None if self.trusted_last_seen_frame is None else int(self.trusted_last_seen_frame)
        )
        self.trusted_raw_sam_id = None if self.trusted_raw_sam_id is None else int(self.trusted_raw_sam_id)
        self.trusted_native_scope = (
            None if self.trusted_native_scope in (None, "") else str(self.trusted_native_scope)
        )
        self.trusted_assignment_count = max(0, int(self.trusted_assignment_count))
        self.trusted_machine_write_count = max(0, int(self.trusted_machine_write_count))
        self.future_machine_write_count = max(0, int(self.future_machine_write_count))
        self.human_write_count = max(0, int(self.human_write_count))
        self.trusted_lost_age = max(0, int(self.trusted_lost_age))
        self.last_committed_frame = None if self.last_committed_frame is None else int(self.last_committed_frame)
        if self.last_committed_candidate_uid is not None:
            self.last_committed_candidate_uid = str(self.last_committed_candidate_uid)

    # Compatibility aliases make the state usable by small existing motion
    # helpers without conflating the fields in serialized audit records.
    @property
    def last_box(self) -> list[float] | None:
        return self.trusted_last_box

    @property
    def velocity(self) -> np.ndarray:
        return self.trusted_velocity

    @property
    def last_seen_frame(self) -> int | None:
        return self.trusted_last_seen_frame

    @property
    def previous_raw_sam_id(self) -> int | None:
        return self.trusted_raw_sam_id

    @property
    def previous_native_scope(self) -> str | None:
        return self.trusted_native_scope

    def to_dict(self) -> dict[str, Any]:
        return {
            "association_state_id": int(self.association_state_id),
            "public_id": int(self.public_id),
            "trusted_last_box": None if self.trusted_last_box is None else list(self.trusted_last_box),
            "trusted_velocity": self.trusted_velocity.astype(np.float64).tolist(),
            "trusted_last_seen_frame": self.trusted_last_seen_frame,
            "trusted_raw_sam_id": self.trusted_raw_sam_id,
            "trusted_native_scope": self.trusted_native_scope,
            "trusted_assignment_count": int(self.trusted_assignment_count),
            "trusted_machine_write_count": int(self.trusted_machine_write_count),
            "future_machine_write_count": int(self.future_machine_write_count),
            "human_write_count": int(self.human_write_count),
            "trusted_lost_age": int(self.trusted_lost_age),
            "last_committed_candidate_uid": self.last_committed_candidate_uid,
            "last_committed_frame": self.last_committed_frame,
        }


class TrustedPersistentPublicAssociationBank:
    """Causal bank with consensus-only machine observation commits."""

    schema_version = "N72R15_TRUSTED_PERSISTENT_PUBLIC_ASSOCIATION_BANK_V1"

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
        self.motion_states: dict[int, TrustedPublicState] = {}
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
    ) -> TrustedPublicState:
        state_id, public = int(association_state_id), int(public_id)
        old_public = self.state_to_public.get(state_id)
        if old_public is not None and old_public != public:
            raise ValueError(f"state {state_id} already maps to public {old_public}, not {public}")
        old_state = self.motion_states.get(public)
        if old_state is not None and old_state.association_state_id != state_id:
            raise ValueError(f"public {public} already maps to state {old_state.association_state_id}, not {state_id}")
        if old_state is None:
            old_state = TrustedPublicState(
                association_state_id=state_id,
                public_id=public,
                trusted_last_box=None if last_box is None else list(last_box),
                trusted_last_seen_frame=last_seen_frame,
                trusted_raw_sam_id=previous_raw_sam_id,
                trusted_native_scope=previous_native_scope,
            )
            self.motion_states[public] = old_state
        self.state_to_public[state_id] = public
        return old_state

    def states_for_pairs(self, pairs: Sequence[tuple[int, int]]) -> list[TrustedPublicState]:
        seen_states: set[int] = set()
        seen_publics: set[int] = set()
        result: list[TrustedPublicState] = []
        for state_id, public_id in pairs:
            state_id, public_id = int(state_id), int(public_id)
            if state_id in seen_states or public_id in seen_publics:
                raise ValueError("state/public axis contains duplicates")
            seen_states.add(state_id)
            seen_publics.add(public_id)
            result.append(self.ensure_state(state_id, public_id))
        return result

    @staticmethod
    def _assignment_rows(solver: Mapping[str, Any]) -> list[Mapping[str, Any]]:
        rows = solver.get("assignment_rows")
        if not isinstance(rows, list):
            raise ValueError("solver artifact lacks assignment_rows")
        return rows

    @staticmethod
    def _candidate_map(solver: Mapping[str, Any]) -> dict[int, str | None]:
        result: dict[int, str | None] = {}
        for item in solver.get("public_assignments", []):
            if not isinstance(item, Mapping) or item.get("public_id") is None:
                continue
            public = int(item["public_id"])
            if public in result:
                raise ValueError(f"duplicate public assignment {public}")
            uid = item.get("candidate_uid")
            result[public] = None if uid in (None, "", "None") else str(uid)
        return result

    def _set_observation(
        self,
        state: TrustedPublicState,
        candidate: Mapping[str, Any],
        frame: int,
        *,
        update_machine: bool,
        count_assignment: bool,
        future: bool,
    ) -> bool:
        box = _box(candidate.get("box_xyxy", candidate.get("box")), "candidate box")
        old_box = state.trusted_last_box
        old_frame = state.trusted_last_seen_frame
        if old_box is not None and old_frame is not None and int(frame) > int(old_frame):
            old_center = np.asarray([(old_box[0] + old_box[2]) / 2.0, (old_box[1] + old_box[3]) / 2.0])
            new_center = np.asarray([(box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0])
            dt = max(1, int(frame) - int(old_frame))
            state.trusted_velocity = 0.8 * state.trusted_velocity + 0.2 * (new_center - old_center) / float(dt)
        state.trusted_last_box = box
        state.trusted_last_seen_frame = int(frame)
        state.trusted_raw_sam_id = _raw(candidate)
        state.trusted_native_scope = _scope(candidate)
        state.trusted_lost_age = 0
        if count_assignment:
            state.trusted_assignment_count += 1
        if not update_machine:
            return False
        feature = _feature(candidate.get("feature", candidate.get("embedding")), "candidate feature")
        confidence = candidate.get("confidence", candidate.get("presence_score", 0.0))
        try:
            confidence = float(confidence)
        except (TypeError, ValueError):
            return False
        wrote = False
        if feature is not None and np.isfinite(confidence):
            wrote = bool(
                self.appearance_memory.update_from_machine(
                    state.public_id, int(frame), feature, confidence=confidence
                )
            )
        if wrote:
            state.trusted_machine_write_count += 1
            if future:
                state.future_machine_write_count += 1
        return wrote

    def initialize_from_event_frame(
        self,
        *,
        event_frame: int,
        candidate_rows: Sequence[Mapping[str, Any]],
        solver: Mapping[str, Any],
        association_state_axis: Sequence[int],
        public_id_axis: Sequence[int],
        target_public_id: int,
        target_anchor: Sequence[float],
        target_box: Sequence[float],
        target_candidate: Mapping[str, Any] | None,
        competing_embeddings: Iterable[np.ndarray] = (),
        event_id: str | None = None,
    ) -> dict[str, Any]:
        frame = int(event_frame)
        states = [int(value) for value in association_state_axis]
        publics = [int(value) for value in public_id_axis]
        if len(states) != len(publics) or len(states) != len(set(states)) or len(publics) != len(set(publics)):
            raise ValueError("event-frame state/public axes are invalid")
        state_by_public = dict(zip(publics, states))
        for state_id, public_id in zip(states, publics):
            self.ensure_state(state_id, public_id)
        by_uid = {str(row.get("candidate_uid")): row for row in candidate_rows}
        if len(by_uid) != len(candidate_rows) or any(uid in ("None", "") for uid in by_uid):
            raise ValueError("event-frame candidate UIDs are not unique")
        assigned_publics: set[int] = set()
        initial_machine_writes: list[int] = []
        for decision in self._assignment_rows(solver):
            public = decision.get("public_id")
            uid = decision.get("candidate_uid")
            if public is None or uid is None or str(decision.get("status")) != "ASSIGNED_TO_PUBLIC_ID":
                continue
            public = int(public)
            if public not in state_by_public or str(uid) not in by_uid:
                raise ValueError("event-frame assignment leaves explicit axes/candidate pool")
            state = self.motion_states[public]
            wrote = self._set_observation(
                state,
                by_uid[str(uid)],
                frame,
                update_machine=True,
                count_assignment=False,
                future=False,
            )
            if wrote:
                initial_machine_writes.append(public)
            assigned_publics.add(public)
        for public in publics:
            if public not in assigned_publics:
                self.motion_states[public].trusted_lost_age = 1

        target_public = int(target_public_id)
        if target_public not in state_by_public:
            raise ValueError("human target is absent from the explicit public axis")
        target_state = self.motion_states[target_public]
        target_state.trusted_last_box = _box(target_box, "target human box")
        target_state.trusted_last_seen_frame = frame
        target_state.trusted_velocity = np.zeros(2, dtype=np.float64)
        target_state.trusted_lost_age = 0
        if target_candidate is not None:
            target_state.trusted_raw_sam_id = _raw(target_candidate)
            target_state.trusted_native_scope = _scope(target_candidate)
        competitors = list(competing_embeddings)
        written = bool(
            self.appearance_memory.update_from_human(
                target_public,
                frame,
                np.asarray(target_anchor, dtype=np.float32),
                quality=1.0,
                competing_embeddings=competitors,
                write_event_id=None if event_id is None else str(event_id),
            )
        )
        if written:
            target_state.human_write_count += 1
        return {
            "frame": frame,
            "assigned_public_ids": sorted(assigned_publics),
            "initial_machine_memory_write_public_ids": sorted(initial_machine_writes),
            "target_public_id": target_public,
            "human_memory_written": written,
            "human_write_count": int(written),
            "explicit_competitor_count": len(competitors),
            "event_frame_memory_read": False,
            "first_memory_visible_frame": frame + 1,
            "runtime_future_gt_used": False,
        }

    def age_without_machine(self, frame: int) -> dict[str, Any]:
        """Age every trusted state without accepting any future observation."""

        frame = int(frame)
        for state in self.motion_states.values():
            if state.trusted_last_seen_frame is None or frame > int(state.trusted_last_seen_frame):
                state.trusted_lost_age += 1
        return {
            "frame": frame,
            "machine_memory_write_public_ids": [],
            "motion_state_write_public_ids": [],
            "trusted_lost_age_only": True,
            "runtime_future_gt_used": False,
        }

    def update_from_consensus(
        self,
        *,
        frame: int,
        candidate_rows: Sequence[Mapping[str, Any]],
        base_solver: Mapping[str, Any],
        treatment_solver: Mapping[str, Any],
        target_public_id: int | None = None,
        none_score: float = 0.0,
    ) -> dict[str, Any]:
        """Commit only public IDs with identical non-NONE base/treatment UIDs."""

        by_uid = {str(row.get("candidate_uid")): row for row in candidate_rows}
        if len(by_uid) != len(candidate_rows) or any(uid in ("None", "") for uid in by_uid):
            raise ValueError("consensus candidate UIDs are not unique")
        base_map = self._candidate_map(base_solver)
        treatment_map = self._candidate_map(treatment_solver)
        publics = sorted(self.motion_states)
        if set(base_map) != set(publics) or set(treatment_map) != set(publics):
            raise ValueError("consensus solver public axes do not match trusted state bank")
        consensus: list[int] = []
        disagreements: list[int] = []
        unassigned: list[int] = []
        machine_writes: list[int] = []
        motion_writes: list[int] = []
        for public in publics:
            base_uid, treatment_uid = base_map[public], treatment_map[public]
            state = self.motion_states[public]
            state.last_committed_candidate_uid = treatment_uid
            state.last_committed_frame = int(frame)
            if base_uid is not None and treatment_uid is not None and base_uid == treatment_uid:
                consensus.append(public)
                candidate = by_uid.get(str(treatment_uid))
                if candidate is None:
                    raise ValueError(f"consensus candidate {treatment_uid} is absent")
                motion_writes.append(public)
                score = next(
                    float(item.get("score", none_score))
                    for item in treatment_solver.get("assignment_rows", [])
                    if item.get("public_id") is not None and int(item["public_id"]) == public
                )
                wrote = self._set_observation(
                    state,
                    candidate,
                    int(frame),
                    update_machine=bool(score > float(none_score)),
                    count_assignment=True,
                    future=True,
                )
                if wrote:
                    machine_writes.append(public)
            elif base_uid != treatment_uid:
                disagreements.append(public)
                state.trusted_lost_age += 1
            else:
                unassigned.append(public)
                state.trusted_lost_age += 1
        target = None if target_public_id is None else int(target_public_id)
        return {
            "frame": int(frame),
            "consensus_public_ids": consensus,
            "consensus_public_count": len(consensus),
            "disagreement_public_ids": disagreements,
            "disagreement_public_count": len(disagreements),
            "unassigned_public_ids": unassigned,
            "machine_memory_write_public_ids": machine_writes,
            "motion_state_write_public_ids": motion_writes,
            "target_consensus": None if target is None else target in consensus,
            "target_disagreement": None if target is None else target in disagreements,
            "runtime_future_gt_used": False,
            "none_score": float(none_score),
        }

    def compact_snapshot(self, frame: int | None = None) -> list[dict[str, Any]]:
        output: list[dict[str, Any]] = []
        memory_records = self.appearance_memory.records
        for public in sorted(self.motion_states):
            state = self.motion_states[public]
            record = memory_records.get(public)
            output.append(
                {
                    "association_state_id": int(state.association_state_id),
                    "public_id": int(state.public_id),
                    "trusted_last_seen_frame": state.trusted_last_seen_frame,
                    "trusted_lost_age": int(state.trusted_lost_age),
                    "trusted_raw_sam_id": state.trusted_raw_sam_id,
                    "trusted_native_scope": state.trusted_native_scope,
                    "trusted_assignment_count": int(state.trusted_assignment_count),
                    "trusted_machine_write_count": int(state.trusted_machine_write_count),
                    "future_machine_write_count": int(state.future_machine_write_count),
                    "human_write_count": int(state.human_write_count),
                    "last_committed_candidate_uid": state.last_committed_candidate_uid,
                    "last_committed_frame": state.last_committed_frame,
                    "appearance_present": bool(record is not None and (record.prototype is not None or record.positive)),
                    "human_anchor_count": 0 if record is None else len(record.positive),
                    "machine_anchor_count": 0 if record is None else int(record.write_count),
                    "snapshot_frame": None if frame is None else int(frame),
                }
            )
        return output

    def snapshot(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "state_to_public": {str(k): int(v) for k, v in sorted(self.state_to_public.items())},
            "motion_states": {str(k): v.to_dict() for k, v in sorted(self.motion_states.items())},
            "appearance_memory": self.appearance_memory.snapshot(),
        }

    def digest(self) -> str:
        payload = json.dumps(self.snapshot(), sort_keys=True, separators=(",", ":"), allow_nan=False)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def appearance_memory_hash(self) -> str:
        payload = json.dumps(self.appearance_memory.snapshot(), sort_keys=True, separators=(",", ":"), allow_nan=False)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def collect_explicit_competitor_embeddings(
    *,
    event: Mapping[str, Any],
    candidate_rows: Sequence[Mapping[str, Any]],
    solver: Mapping[str, Any],
    target_public_id: int,
) -> tuple[list[np.ndarray], dict[str, Any]]:
    """Use only explicitly supplied competitor-public fields.

    Frozen N72R9 events currently provide no such field.  In that case this
    returns an empty list; it never treats all other candidates as negatives.
    """

    explicit_keys = (
        "other_canonical_public_id",
        "other_public_id",
        "competitor_public_id",
        "other_identity_public_id",
    )
    values: list[Any] = []
    for key in explicit_keys:
        if event.get(key) is not None:
            values.append(event[key])
    ids: list[int] = []
    for value in values:
        if isinstance(value, (list, tuple)):
            ids.extend(int(item) for item in value)
        else:
            ids.append(int(value))
    ids = sorted(set(value for value in ids if value != int(target_public_id)))
    assigned = {
        int(item["public_id"]): item.get("candidate_uid")
        for item in solver.get("public_assignments", [])
        if item.get("public_id") is not None
    }
    by_uid = {str(item.get("candidate_uid")): item for item in candidate_rows}
    embeddings: list[np.ndarray] = []
    resolved: list[int] = []
    for public in ids:
        uid = assigned.get(public)
        candidate = None if uid in (None, "", "None") else by_uid.get(str(uid))
        feature = None if candidate is None else _feature(candidate.get("feature", candidate.get("embedding")), "competitor feature")
        if feature is not None:
            embeddings.append(feature)
            resolved.append(public)
    return embeddings, {
        "explicit_competitor_public_ids": ids,
        "resolved_competitor_public_ids": resolved,
        "count": len(embeddings),
        "source": "explicit_event_public_id_only",
        "runtime_future_gt_used": False,
    }


__all__ = [
    "FEATURE_DIM",
    "TrustedPublicState",
    "TrustedPersistentPublicAssociationBank",
    "collect_explicit_competitor_embeddings",
]
