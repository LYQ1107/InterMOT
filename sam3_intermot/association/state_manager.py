"""Online identity state machine: birth / active / lost / reactivation."""

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch.nn as nn

from sam3_intermot.association.identity_state import IdentityState
from sam3_intermot.association.appearance_memory import AppearanceMemory
from sam3_intermot.association.online_associator import (
    box_iou,
    hungarian_max,
    has_valid_feature,
    score_matrix_pairwise,
    score_matrix_set,
)


@dataclass
class StateManagerConfig:
    score_threshold: float = 0.0
    max_lost_gap: int = 90
    ema: float = 0.9
    anchor_blend: float = 0.7
    positive_bonus: float = 5.0
    native_bonus: float = 3.0
    variant: str = "pairwise"  # pairwise | set | reid
    use_local_scope: bool = False
    scope_frames: int = 10
    native_constraint_frames: Optional[int] = None  # None = permanent
    authority_mode: str = "permanent"  # permanent | decay | evidence
    authority_hard_frames: int = 1
    authority_decay_frames: int = 8
    evidence_refresh_threshold: float = 0.5
    freeze_machine_in_scope: bool = False
    reid_weights: dict = field(
        default_factory=lambda: {"sim": 1.5, "iou": 1.0, "native": 0.5, "gap": 0.1}
    )
    # CCAM is opt-in so legacy K1/B10 behaviour is byte-for-byte unaffected.
    use_appearance_memory: bool = False
    appearance_positive_weight: float = 1.0
    appearance_negative_weight: float = 1.0
    appearance_score_weight: float = 1.0
    appearance_decay_frames: float = 120.0
    appearance_anchor_cap: int = 8
    appearance_negative_cap: int = 16
    appearance_min_machine_confidence: float = 0.5
    appearance_reliability_threshold: float = 0.0
    # N72R3 mode: the outer persistent identity runtime owns births and public
    # IDs.  The association state manager may score/solve existing states but
    # must never promote an unmatched candidate into authority by itself.
    external_identity_authority: bool = False


class StateManager:
    def __init__(self, config: StateManagerConfig, public_authority_resolver=None) -> None:
        self.cfg = config
        # This bridge is intentionally opt-in.  A StateManager PID is an
        # association-local state ID and must not be presented as public
        # identity without an explicit same-run resolver.
        self.public_authority_resolver = public_authority_resolver
        self.states: Dict[int, IdentityState] = {}
        self.next_pid = 1
        self.output_log: List[dict] = []
        self.candidate_log: List[dict] = []
        self.scope_expiry: Dict[int, int] = {}
        self.external_public_ids: Dict[int, int] = {}
        self.unmatched_candidates: List[dict] = []
        self.unmatched_states: List[int] = []
        self.appearance_memory = AppearanceMemory(
            anchor_cap=config.appearance_anchor_cap,
            negative_cap=config.appearance_negative_cap,
            decay_frames=config.appearance_decay_frames,
            min_machine_confidence=config.appearance_min_machine_confidence,
            reliability_threshold=config.appearance_reliability_threshold,
        )

    def update_human_appearance(
        self,
        public_id: int,
        frame: int,
        embedding: np.ndarray,
        quality: float = 1.0,
        competing_embeddings=None,
        write_event_id: Optional[str] = None,
    ) -> bool:
        if not self.cfg.use_appearance_memory:
            return False
        return self.appearance_memory.update_from_human(
            public_id, frame, embedding, quality, competing_embeddings, write_event_id
        )

    def update_machine_appearance(
        self, public_id: int, frame: int, embedding: np.ndarray, confidence: float = 1.0
    ) -> bool:
        if not self.cfg.use_appearance_memory:
            return False
        return self.appearance_memory.update_from_machine(public_id, frame, embedding, confidence)

    def native_expiry(self, frame: int) -> Optional[int]:
        if self.cfg.native_constraint_frames is None:
            return None
        return frame + self.cfg.native_constraint_frames

    def mark_scope(self, pids, frame: int) -> None:
        for pid in pids:
            pid = int(pid)
            if pid in self.states:
                self.scope_expiry[pid] = frame + self.cfg.scope_frames

    def _active_scope_pids(self, frame: int) -> set:
        expired = [pid for pid, exp in self.scope_expiry.items() if exp < frame]
        for pid in expired:
            self.scope_expiry.pop(pid, None)
        return {pid for pid, exp in self.scope_expiry.items() if exp >= frame}

    def _new_pid(self) -> int:
        if self.cfg.external_identity_authority:
            raise RuntimeError(
                "external_identity_authority=True forbids StateManager-local births"
            )
        pid = self.next_pid
        self.next_pid += 1
        return pid

    def register_identity_state(
        self,
        state_id: int,
        public_id: int,
        initial_observation: dict,
        frame: int,
    ) -> IdentityState:
        """Register an outer-owned identity state without allocating a PID.

        ``state_id`` is the solver axis and ``public_id`` is supplied by the
        sequence-persistent identity runtime.  They are intentionally stored
        separately even when their numeric values happen to coincide.
        """

        if not self.cfg.external_identity_authority:
            raise RuntimeError(
                "register_identity_state requires external_identity_authority=True"
            )
        state = int(state_id)
        public = int(public_id)
        if state <= 0 or public <= 0:
            raise ValueError("state_id and public_id must be positive")
        existing = self.states.get(state)
        if existing is not None:
            if self.external_public_ids.get(state) != public:
                raise ValueError(f"state {state} already has a different public ID")
            return existing
        feature = np.asarray(initial_observation.get("feat", []), dtype=np.float32)
        box = np.asarray(initial_observation.get("box", [0, 0, 0, 0]), dtype=float)
        native = int(initial_observation.get("native_tid", -1))
        identity_state = IdentityState(state, feature, box, int(frame), native)
        self.states[state] = identity_state
        self.external_public_ids[state] = public
        self.next_pid = max(self.next_pid, state + 1)
        return identity_state

    def register_from_persistent_identity(self, record: object, obs: dict, frame: int) -> IdentityState:
        """Adapter for a ``PersistentIdentityRecord`` owned by the outer runtime."""

        return self.register_identity_state(
            int(getattr(record, "association_state_id")),
            int(getattr(record, "public_id")),
            obs,
            frame,
        )

    @staticmethod
    def _json_feature(obs: dict) -> List[float]:
        try:
            return np.asarray(obs.get("feat", []), dtype=np.float32).reshape(-1).tolist()
        except (TypeError, ValueError):
            return []

    def _candidate_audit(
        self,
        frame: int,
        states: List[IdentityState],
        obs_list: List[dict],
        scores: np.ndarray,
        assignment: np.ndarray,
        score_audit: dict,
        assignment_after_scope: Optional[np.ndarray] = None,
    ) -> None:
        """Persist a future-blind, candidate-complete association transaction.

        The tape intentionally stores only observations available at this
        frame.  In particular, no GT box, future candidate outcome, reward, or
        dataset identity is derived here.
        """
        candidates = []
        for index, obs in enumerate(obs_list):
            valid = has_valid_feature(obs)
            candidates.append(
                {
                    "index": int(index),
                    "obs_id": int(obs.get("obs_id", index)),
                    "candidate_uid": obs.get("candidate_uid"),
                    "source_run_id": obs.get("source_run_id"),
                    "session_id": obs.get("session_id"),
                    "segment_id": obs.get("segment_id"),
                    "window_id": obs.get("window_id"),
                    "chunk_id": obs.get("chunk_id"),
                    "official_raw_sam_id": obs.get("official_raw_sam_id"),
                    "adapter_external_id": obs.get("adapter_external_id"),
                    "segment_local_id": obs.get("segment_local_id"),
                    "sequence_global_id": obs.get("sequence_global_id"),
                    "native_tid": int(obs.get("native_tid", -1)),
                    "native_age": float(obs.get("native_age", 0.0)),
                    "confidence": float(obs.get("conf", 1.0)),
                    "box": np.asarray(obs.get("box", [0, 0, 0, 0]), dtype=float).reshape(-1).tolist(),
                    "feature": self._json_feature(obs),
                    "has_feat": bool(obs.get("has_feat", 1.0)),
                    "feature_available": bool(valid),
                }
            )
        public_id_order = [int(s.pid) for s in states]
        candidate_order = [int(o.get("obs_id", index)) for index, o in enumerate(obs_list)]

        def resolved_public_id(state: IdentityState) -> Optional[int]:
            resolver = self.public_authority_resolver
            if resolver is None:
                return None
            resolve = getattr(resolver, "resolve", None)
            if not callable(resolve):
                return None
            value = resolve(int(state.pid))
            return None if value is None else int(value)

        public_id_axis = [resolved_public_id(state) for state in states]

        def assignment_pairs(values: np.ndarray) -> List[dict]:
            pairs: List[dict] = []
            values = np.asarray(values, dtype=int).reshape(-1)
            for candidate_index, state_index in enumerate(values.tolist()):
                if state_index < 0 or state_index >= len(states):
                    continue
                pairs.append(
                    {
                        "candidate_index": int(candidate_index),
                        "candidate_obs_id": int(obs_list[candidate_index].get("obs_id", candidate_index)),
                        "native_tid": int(obs_list[candidate_index].get("native_tid", -1)),
                        "state_index": int(state_index),
                        # Kept for historical readers only.  N72R1 sidecars
                        # use association_state_id + public_id_axis below.
                        "public_id": int(states[state_index].pid),
                        "association_state_id": int(states[state_index].pid),
                        "resolved_public_id": resolved_public_id(states[state_index]),
                        "public_id_semantics": "legacy_state_pid_not_authoritative",
                        "candidate_uid": obs_list[candidate_index].get("candidate_uid"),
                        "score": float(scores[candidate_index, state_index]),
                    }
                )
            return pairs

        # Scores are stored in candidate-row x state-column form for the
        # legacy path.  The transposed matrices below are the explicit
        # public-ID x complete-candidate view required by N35.
        public_id_to_native_tid: Dict[str, Optional[int]] = {}
        final_assignment = np.asarray(
            assignment if assignment_after_scope is None else assignment_after_scope,
            dtype=int,
        ).reshape(-1)
        for state_index, state in enumerate(states):
            matched = np.where(final_assignment == state_index)[0]
            public_id_to_native_tid[str(int(state.pid))] = (
                None
                if matched.size == 0
                else int(obs_list[int(matched[0])].get("native_tid", -1))
            )
        record = {
            "frame": int(frame),
            "public_ids": public_id_order,
            "public_id_order": public_id_order,
            "association_state_axis": public_id_order,
            "public_id_axis": public_id_axis,
            "public_id_axis_complete": bool(states) and all(value is not None for value in public_id_axis),
            "score_threshold": float(self.cfg.score_threshold),
            "solver_version": "scipy.optimize.linear_sum_assignment",
            "explicit_none_state_indices": [],
            "assignment_solver_has_explicit_none": False,
            "candidate_order": candidate_order,
            "candidate_axis": [o.get("candidate_uid") for o in obs_list],
            "source_run_id": next((o.get("source_run_id") for o in obs_list if o.get("source_run_id") is not None), None),
            "session_id": next((o.get("session_id") for o in obs_list if o.get("session_id") is not None), None),
            "segment_id": next((o.get("segment_id") for o in obs_list if o.get("segment_id") is not None), None),
            "window_id": next((o.get("window_id") for o in obs_list if o.get("window_id") is not None), None),
            "chunk_id": next((o.get("chunk_id") for o in obs_list if o.get("chunk_id") is not None), None),
            "candidates": candidates,
            "candidate_native_ids": [int(o.get("native_tid", -1)) for o in obs_list],
            "candidate_complete": bool(all(c["feature_available"] for c in candidates)),
            "candidate_set_complete": bool(all(c["feature_available"] for c in candidates)),
            "scores": np.asarray(scores, dtype=float).tolist(),
            "base_scores_before_appearance": score_audit.get(
                "base_scores_before_appearance", np.asarray(scores, dtype=float).tolist()
            ),
            "appearance_memory_scores": score_audit.get(
                "appearance_memory_scores", np.zeros_like(scores, dtype=float).tolist()
            ),
            "appearance_score_deltas": score_audit.get(
                "appearance_score_deltas", np.zeros_like(scores, dtype=float).tolist()
            ),
            "fused_scores": score_audit.get("fused_scores", np.asarray(scores, dtype=float).tolist()),
            "public_id_score_matrix": np.asarray(scores, dtype=float).T.tolist(),
            "public_id_base_score_matrix": np.asarray(
                score_audit.get("base_scores_before_appearance", scores), dtype=float
            ).T.tolist(),
            "public_id_appearance_score_matrix": np.asarray(
                score_audit.get(
                    "appearance_memory_scores", np.zeros_like(scores, dtype=float)
                ),
                dtype=float,
            ).T.tolist(),
            "public_id_fused_score_matrix": np.asarray(
                score_audit.get("fused_scores", scores), dtype=float
            ).T.tolist(),
            "assignment": [int(x) for x in np.asarray(assignment, dtype=int).tolist()],
            "assignment_after_scope": [
                int(x) for x in final_assignment.tolist()
            ],
            "assignment_pairs": assignment_pairs(assignment),
            "assignment_pairs_after_scope": assignment_pairs(final_assignment),
            "public_id_to_native_tid": public_id_to_native_tid,
            "appearance_memory_enabled": bool(self.cfg.use_appearance_memory),
            "public_authority_resolver_present": self.public_authority_resolver is not None,
            "external_identity_authority": bool(self.cfg.external_identity_authority),
            "unmatched_candidates": deepcopy(self.unmatched_candidates),
            "unmatched_states": list(self.unmatched_states),
            "human_events": [],
        }
        self.candidate_log.append(record)

    def get_or_create(
        self,
        pid: int,
        obs: dict,
        frame: int,
    ) -> IdentityState:
        st = self.states.get(pid)
        if st is None:
            st = IdentityState(pid, obs["feat"], obs["box"], frame, obs["native_tid"])
            self.states[pid] = st
        return st

    def candidates(self, frame: int) -> List[IdentityState]:
        out = []
        for st in self.states.values():
            st.prune_constraints(frame)
            if st.state == IdentityState.TERMINATED:
                continue
            if st.state == IdentityState.LOST and frame - st.last_seen_frame > self.cfg.max_lost_gap:
                st.terminate()
                continue
            out.append(st)
        return out

    def rollout_frame(
        self,
        frame: int,
        obs_list: List[dict],
        model: Optional[nn.Module] = None,
    ) -> List[Tuple[int, np.ndarray]]:
        self.unmatched_candidates = []
        self.unmatched_states = []
        states = self.candidates(frame)
        n_obs = len(obs_list)
        rows: List[Tuple[int, np.ndarray]] = []
        score_audit: dict = {}
        if n_obs == 0:
            self._candidate_audit(
                frame,
                states,
                obs_list,
                np.zeros((0, len(states)), dtype=np.float32),
                np.zeros(0, dtype=int),
                score_audit,
            )
            # An empty candidate set has an explicit, vacuous public/native
            # mapping.  Keep this field present so downstream tape validators
            # can distinguish "no candidates to map" from a non-empty frame
            # whose mapping was lost.
            if self.candidate_log and int(self.candidate_log[-1].get("frame", -1)) == int(frame):
                self.candidate_log[-1]["candidate_public_ids"] = []
                self.candidate_log[-1]["candidate_public_id_mapping_complete"] = True
            for st in states:
                if st.state == IdentityState.ACTIVE:
                    st.mark_lost(frame)
                else:
                    st.advance_lost()
            self.unmatched_states = [int(st.pid) for st in states]
            return rows
        if self.cfg.variant == "set":
            scores = score_matrix_set(
                states,
                obs_list,
                frame,
                model,
                positive_bonus=self.cfg.positive_bonus,
                native_bonus=self.cfg.native_bonus,
                authority_mode=self.cfg.authority_mode,
                hard_frames=self.cfg.authority_hard_frames,
                decay_frames=self.cfg.authority_decay_frames,
                refresh_threshold=self.cfg.evidence_refresh_threshold,
                appearance_memory=self.appearance_memory if self.cfg.use_appearance_memory else None,
                appearance_score_weight=self.cfg.appearance_score_weight,
                appearance_positive_weight=self.cfg.appearance_positive_weight,
                appearance_negative_weight=self.cfg.appearance_negative_weight,
                score_audit=score_audit,
            )
        else:
            scores = score_matrix_pairwise(
                states,
                obs_list,
                frame,
                None if self.cfg.variant == "reid" else model,
                reid_weights=self.cfg.reid_weights,
                positive_bonus=self.cfg.positive_bonus,
                native_bonus=self.cfg.native_bonus,
                authority_mode=self.cfg.authority_mode,
                hard_frames=self.cfg.authority_hard_frames,
                decay_frames=self.cfg.authority_decay_frames,
                refresh_threshold=self.cfg.evidence_refresh_threshold,
                appearance_memory=self.appearance_memory if self.cfg.use_appearance_memory else None,
                appearance_score_weight=self.cfg.appearance_score_weight,
                appearance_positive_weight=self.cfg.appearance_positive_weight,
                appearance_negative_weight=self.cfg.appearance_negative_weight,
                score_audit=score_audit,
            )
        assign = hungarian_max(scores)
        # Candidate-complete, future-blind audit record.  This is intentionally
        # captured before any current-frame human intervention can write CCAM.
        assignment_before_scope = assign.copy()
        scope_pids = self._active_scope_pids(frame)
        locked: Dict[int, int] = {}
        if self.cfg.use_local_scope and scope_pids:
            locked_state = set()
            matched_obs = set()
            for i in range(len(obs_list)):
                j = assign[i]
                if j < 0 or scores[i, j] < self.cfg.score_threshold:
                    continue
                if states[j].pid not in scope_pids:
                    locked[j] = i
                    locked_state.add(j)
                    matched_obs.add(i)
            free_obs = [i for i in range(len(obs_list)) if i not in matched_obs]
            free_state = [j for j in range(len(states)) if j not in locked_state]
            if free_obs and free_state:
                sub = scores[np.ix_(free_obs, free_state)]
                sub_assign = hungarian_max(sub)
                for i_loc in range(len(free_obs)):
                    j_loc = sub_assign[i_loc]
                    if j_loc >= 0 and sub[i_loc, j_loc] >= self.cfg.score_threshold:
                        assign[free_obs[i_loc]] = free_state[j_loc]
        self._candidate_audit(
            frame,
            states,
            obs_list,
            scores,
            assignment_before_scope,
            score_audit,
            assignment_after_scope=assign,
        )
        matched_state = np.zeros(len(states), dtype=bool)
        # The score matrices above are intentionally a before-intervention
        # audit.  Record the final online assignment separately so a sharded
        # tape can prove the public/native mapping even when a candidate is
        # born on the current frame.
        candidate_public_ids: List[Optional[int]] = [None] * n_obs
        for i, obs in enumerate(obs_list):
            j = assign[i]
            if j >= 0:
                matched_state[j] = True
            if j < 0 or scores[i, j] < self.cfg.score_threshold:
                # In N72R3 the outer runtime decides whether an unmatched
                # candidate is a birth.  Keeping it unmatched here prevents
                # an association-local PID from becoming public authority.
                if self.cfg.external_identity_authority:
                    self.unmatched_candidates.append(
                        {
                            "candidate_index": int(i),
                            "candidate_uid": obs.get("candidate_uid"),
                            "obs_id": int(obs.get("obs_id", i)),
                            "box": np.asarray(obs.get("box", [0, 0, 0, 0]), dtype=float).copy(),
                            "feature": np.asarray(obs.get("feat", []), dtype=np.float32).copy(),
                            "frame": int(frame),
                            "reason": "OUTER_BIRTH_DECISION_REQUIRED",
                        }
                    )
                    candidate_public_ids[i] = None
                    continue
                pid = self._new_pid()
                st = IdentityState(pid, obs["feat"], obs["box"], frame, obs["native_tid"])
                self.states[pid] = st
                if self.cfg.use_appearance_memory:
                    self.update_machine_appearance(
                        pid,
                        frame,
                        obs["feat"],
                        float(obs.get("conf", 1.0)),
                    )
                candidate_public_ids[i] = int(pid)
                rows.append((pid, obs["box"].copy()))
                continue
            st = states[j]
            st.last_match_score = float(scores[i, j])
            freeze = self.cfg.freeze_machine_in_scope and st.pid in scope_pids
            st.update_machine(
                obs["feat"],
                obs["box"],
                frame,
                obs["native_tid"],
                self.cfg.ema,
                update_prototype=not freeze,
            )
            if self.cfg.use_appearance_memory:
                self.update_machine_appearance(st.pid, frame, obs["feat"], float(obs.get("conf", 1.0)))
            candidate_public_ids[i] = int(st.pid)
            rows.append((st.pid, obs["box"].copy()))
        for j, st in enumerate(states):
            if matched_state[j]:
                continue
            if st.state == IdentityState.ACTIVE:
                st.mark_lost(frame)
            else:
                st.advance_lost()
        self.unmatched_states = [
            int(st.pid) for index, st in enumerate(states) if not matched_state[index]
        ]
        if self.candidate_log and int(self.candidate_log[-1].get("frame", -1)) == int(frame):
            self.candidate_log[-1]["candidate_public_ids"] = candidate_public_ids
            self.candidate_log[-1]["candidate_public_id_mapping_complete"] = bool(
                all(pid is not None for pid in candidate_public_ids)
            )
        return sorted(rows, key=lambda kv: kv[0])

    def annotate_human_event(self, frame: int, event: dict, record: Optional[dict] = None) -> bool:
        """Attach a sanitized human transaction to the current-frame audit.

        This method is called after spatial application.  It records the
        supplied human event as provenance, but never synthesizes a future or
        dataset label and never changes the already-written candidate scores.
        """
        target = None
        for item in reversed(self.candidate_log):
            if int(item.get("frame", -1)) == int(frame):
                target = item
                break
        if target is None:
            return False
        event_view = {
            "event_id": event.get("event_id") or event.get("id"),
            "event_type": event.get("event_type"),
            "action_type": event.get("action_type"),
            "frame": int(frame),
            "public_id": event.get("public_id") or event.get("canonical_public_id"),
            "canonical_public_id": event.get("canonical_public_id"),
            "current_public_id": event.get("current_public_id"),
            "other_canonical_public_id": event.get("other_canonical_public_id"),
            "other_auto_tid": event.get("other_auto_tid"),
            "gt_box": None,
            "other_gt_box": None,
            "applied": None if record is None else bool(record.get("applied", False)),
            "appearance_memory": [] if record is None else record.get("appearance_memory", []),
        }
        for key in ("gt_box", "other_gt_box"):
            if event.get(key) is not None:
                event_view[key] = np.asarray(event[key], dtype=float).reshape(-1).tolist()
        target.setdefault("human_events", []).append(event_view)
        return True

    def snapshot(self) -> dict:
        """Capture association state without breaking the memory object alias."""

        attributes = {
            key: value
            for key, value in vars(self).items()
            if key not in {"appearance_memory", "public_authority_resolver"}
        }
        return {
            "schema_version": "N72R3_STATE_MANAGER_SNAPSHOT_V1",
            "attributes": deepcopy(attributes),
            "public_authority_resolver": self.public_authority_resolver,
            "appearance_memory": self.appearance_memory.snapshot(),
        }

    def restore(self, snapshot: dict) -> None:
        """Restore association state in place, including appearance memory."""

        if snapshot.get("schema_version") != "N72R3_STATE_MANAGER_SNAPSHOT_V1":
            raise ValueError("unsupported StateManager snapshot schema")
        memory = self.appearance_memory
        resolver = snapshot.get("public_authority_resolver")
        vars(self).clear()
        vars(self).update(deepcopy(snapshot.get("attributes", {})))
        self.public_authority_resolver = resolver
        self.appearance_memory = memory
        memory.restore(snapshot.get("appearance_memory", {}))

    def state_summary(self) -> dict:
        return {
            "n_states": len(self.states),
            "active": sum(1 for s in self.states.values() if s.state == IdentityState.ACTIVE),
            "lost": sum(1 for s in self.states.values() if s.state == IdentityState.LOST),
            "terminated": sum(1 for s in self.states.values() if s.state == IdentityState.TERMINATED),
            "next_pid": self.next_pid,
            "external_identity_authority": bool(self.cfg.external_identity_authority),
            "external_public_ids": {str(key): int(value) for key, value in self.external_public_ids.items()},
            "appearance_memory_enabled": bool(self.cfg.use_appearance_memory),
            "appearance_memory_records": len(self.appearance_memory.records)
            if self.cfg.use_appearance_memory
            else 0,
        }
