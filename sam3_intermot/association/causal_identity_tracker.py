"""Independent, globally consistent R4 online tracker on real candidate frames.

No score tape or GT is accepted. State is owned by this instance. Candidate
births occur only after explicit NONE; all existing-track assignments are
committed from the complete exact solver output before the next frame.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from typing import Any, Mapping, Sequence

import numpy as np

from .global_assignment_adapter import solve_global, public_map, validate_global
from .identity_authority import AdapterEnsemble, AuthorityConfig, AuthorityController, calibrated_residual, authority_value
from .causal_state_commit import commit_memory
from .identity_state import IdentityState
from .state_manager import StateManager, StateManagerConfig
from .online_associator import score_matrix_pairwise, predicted_iou
from .learned_identity_memory import LearnedIdentityMemoryBank


def state_digest(states: Mapping[int, IdentityState]) -> str:
    return hashlib.sha256(json.dumps({str(k): v.to_dict() for k, v in sorted(states.items())}, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def margin(values: np.ndarray, index: int | None) -> float:
    if index is None or len(values) == 0:
        return -1.0
    return float(values[index] - max([0.0] + [float(x) for i, x in enumerate(values) if i != index]))


class CausalIdentityTracker:
    def __init__(self, *, config: AuthorityConfig, event: Mapping[str, Any], adapter: AdapterEnsemble | None = None, bank: LearnedIdentityMemoryBank | None = None, controller: AuthorityController | None = None, max_lost_gap: int = 90) -> None:
        self.config = config
        # Only the simulated human event crosses the preparation boundary.
        allowed = ("event_frame", "human_anchor", "target_candidate_uid", "target_public_id", "target_association_state_id", "target_box_xyxy", "state_bindings", "interaction_source")
        self.event = {key: deepcopy(event[key]) for key in allowed if key in event}
        self.adapter = adapter
        self.bank = bank
        self.controller = controller
        self.manager = StateManager(StateManagerConfig(external_identity_authority=True, variant="reid", max_lost_gap=max_lost_gap))
        self.max_lost_gap = max_lost_gap
        self.states: dict[int, IdentityState] = {}
        self.next_public = 100001
        self.next_state = 1
        self.frame = -1
        self.target_public: int | None = None
        self.previous_challenger_native: int | None = None
        self.last_trusted_frame = -1
        self.adapter_calls = 0

    def clone(self) -> "CausalIdentityTracker":
        memo = {id(value): value for value in (self.adapter, self.controller, None if self.bank is None else self.bank.updater) if value is not None}
        result = deepcopy(self, memo)
        # Frozen models can be shared; tracker/memory state never is.
        result.adapter = self.adapter
        result.controller = self.controller
        return result

    def _birth(self, row: Mapping[str, Any], frame: int, *, public: int | None = None, state_id: int | None = None, feature: np.ndarray | None = None, box: Sequence[float] | None = None) -> IdentityState:
        public = self.next_public if public is None else int(public)
        state_id = self.next_state if state_id is None else int(state_id)
        if public in self.states or state_id in self.manager.states:
            raise RuntimeError("birth authority collision")
        state = self.manager.register_identity_state(state_id, public, {"feat": row["feature"] if feature is None else feature, "box": row["box_xyxy"] if box is None else box, "native_tid": row["native_tid"], "native_scope": row.get("native_scope")}, frame)
        state.public_id = public
        state.association_state_id = state_id
        self.states[public] = state
        self.next_public = max(self.next_public, public + 1)
        self.next_state = max(self.next_state, state_id + 1)
        return state

    def _initialize(self, rows: Sequence[Mapping[str, Any]], frame: int) -> None:
        if frame != 0 or not rows:
            return
        by_uid = {str(r["candidate_uid"]): r for r in rows}
        if int(self.event["event_frame"]) == 0 and self.event.get("state_bindings"):
            for binding in self.event["state_bindings"]:
                public = int(binding["public_id"])
                row = by_uid[str(binding["initial_candidate_uid"])]
                target = public == int(self.event["target_public_id"])
                self._birth(row, frame, public=public, state_id=int(binding["association_state_id"]), feature=np.asarray(self.event["human_anchor"], dtype=np.float32) if target else None, box=self.event["target_box_xyxy"] if target else None)
        elif self.config.lifecycle != "legacy_fixed":
            for row in rows:
                self._birth(row, frame)

    def _click(self, rows: Sequence[Mapping[str, Any]], solver: Mapping[str, Any], frame: int) -> None:
        if frame != int(self.event["event_frame"]):
            return
        uid = str(self.event["target_candidate_uid"])
        row = next((r for r in rows if str(r["candidate_uid"]) == uid), None)
        if row is None:
            raise ValueError("human-confirmed anchor candidate missing at event frame")
        public = next((p for p, u in public_map(solver).items() if u == uid), None)
        if public is None:
            state = self._birth(row, frame)
            public = state.public_id
        self.target_public = int(public)
        state = self.states[public]
        # An explicit user click may initialize current identity appearance;
        # no future observation is involved. Residual starts at event+1.
        anchor = np.asarray(self.event["human_anchor"], dtype=np.float32)
        anchor = anchor / np.linalg.norm(anchor)
        state.prototype = anchor.copy()
        self.last_trusted_frame = frame
        if self.bank is not None:
            self.bank.initialize_public_identity(public_id=public, association_state_id=state.pid, frame=frame, human_anchor=anchor, interaction_source="simulated_from_gt")

    def step(self, rows: Sequence[Mapping[str, Any]], frame: int, *, encoded_candidates: Sequence[np.ndarray] | None = None, baseline_solver: Mapping[str, Any] | None = None) -> dict[str, Any]:
        if frame != self.frame + 1:
            raise ValueError("frames must be consumed exactly once in increasing order")
        rows = list(rows)
        for row in rows:
            if any(key in row for key in ("gt", "gt_id", "target_gt_id", "oracle", "label", "label_index")) or row.get("runtime_future_gt_used", False) or row.get("runtime_gt_read", False):
                raise ValueError("GT/oracle is not a runtime candidate input")
            feature = np.asarray(row["feature"], dtype=np.float32)
            box = np.asarray(row["box_xyxy"], dtype=float)
            if feature.shape != (512,) or not np.isfinite(feature).all() or np.linalg.norm(feature) <= 1e-6:
                raise ValueError("candidate must have a real finite 512-D feature")
            if box.shape != (4,) or not np.isfinite(box).all() or box[2] <= box[0] or box[3] <= box[1]:
                raise ValueError("invalid runtime box")
        before = state_digest(self.states)
        self._initialize(rows, frame)
        ordered = [s for _, s in sorted(self.states.items()) if self.config.lifecycle == "legacy_fixed" or s.state != IdentityState.TERMINATED]
        if frame > 0 and any(s.last_seen_frame >= frame for s in ordered):
            raise RuntimeError("current/future state read before scoring")
        observations = [{"feat": r["feature"], "box": np.asarray(r["box_xyxy"]), "native_tid": int(r["native_tid"]), "native_scope": r.get("native_scope"), "native_age": int(r.get("native_age", 0)), "has_feat": True} for r in rows]
        base = np.asarray(score_matrix_pairwise(ordered, observations, frame, None, reid_weights={"sim": 1.5, "iou": 1.0, "native": 0.5, "gap": 0.1}, native_bonus=3.0, positive_bonus=5.0), dtype=np.float64)
        base_solver = solve_global(rows, base, ordered, frame=frame)
        self._click(rows, base_solver, frame)
        # If an unmatched click created a new authority, append that real
        # observation to the current solver axis before applying the result.
        if self.target_public is not None and self.target_public not in [s.public_id for s in ordered]:
            ordered.append(self.states[self.target_public])
            base = np.asarray(score_matrix_pairwise(ordered, observations, frame, None, reid_weights={"sim": 1.5, "iou": 1.0, "native": 0.5, "gap": 0.1}, native_bonus=3.0, positive_bonus=5.0), dtype=np.float64)
            base_solver = solve_global(rows, base, ordered, frame=frame)
        identity = np.zeros(len(rows), dtype=np.float64)
        features = np.zeros(10, dtype=np.float32)
        authority = 0.0
        score_hash = None
        adjusted = base.copy()
        challenger_native = None
        if self.target_public is not None and frame > int(self.event["event_frame"]):
            state = self.states[self.target_public]
            record = None if self.bank is None else self.bank.records[self.target_public]
            query = np.asarray(self.event["human_anchor"], dtype=np.float32) if record is None else record.current_state
            score_hash = None if record is None else record.state_hash()
            x = np.stack([r["feature"] for r in rows]) if rows else np.empty((0, 512), dtype=np.float32)
            if self.config.source == "adapter":
                if self.adapter is None and self.config.mode != "off":
                    raise ValueError("real adapter required")
                if self.adapter is not None:
                    identity = self.adapter.scores(query, x, encoded_candidates)
                    self.adapter_calls += 1
            elif self.config.source == "shuffled":
                if self.adapter is None:
                    raise ValueError("shuffled-state control uses the same real Adapter")
                identity = self.adapter.scores(np.roll(query, 137), x, encoded_candidates)
                self.adapter_calls += 1
            elif self.config.source == "raw":
                identity = x @ query
            elif self.config.source == "legacy":
                j = ordered.index(state)
                identity = 1 / (1 + np.exp(-(base[:, j] - 2) / 1.5))
            else:
                raise ValueError("unknown identity score source")
            if rows:
                j = ordered.index(state)
                uids = [str(r["candidate_uid"]) for r in rows]
                base_uid = public_map(base_solver).get(self.target_public)
                base_index = uids.index(base_uid) if base_uid in uids else None
                global_margin = -1.0
                if base_index is not None:
                    alternative = base.copy()
                    alternative[base_index,j] = -1.0e9
                    alternative_solver = solve_global(rows,alternative,ordered,frame=frame)
                    global_margin = sum(r["score"] for r in base_solver["assignment_rows"]) - sum(r["score"] for r in alternative_solver["assignment_rows"])
                top = int(np.argmax(identity))
                challenger_native = int(rows[top]["native_tid"])
                competitors = [k for k in range(len(ordered)) if k != j]
                owner = next((int(r["public_column_index"]) for r in base_solver["assignment_rows"] if r["candidate_uid"] == uids[top] and r["public_id"] not in (None, self.target_public)), None)
                opportunity_cost = max(0.0, float(base[top, owner] - max([0.0] + [float(base[i, owner]) for i in range(len(rows)) if i != top]))) if owner is not None else 0.0
                quality = min(1.0, max(0.0, float(rows[top].get("conf", rows[top].get("confidence", 0.0)))))
                features = np.asarray([identity[top], margin(identity, top), global_margin, predicted_iou(state, np.asarray(rows[top]["box_xyxy"]), frame), quality, float(np.dot(query, np.asarray(self.event["human_anchor"]))), min(1.0, (frame - state.birth_frame) / 1000), min(1.0, (frame - self.last_trusted_frame) / 100), min(1.0, len(rows) / 50), opportunity_cost], dtype=np.float32)
                authority = authority_value(self.config, features, missing=state.state == IdentityState.LOST or base_uid is None, controller=self.controller, previous_challenger_native=self.previous_challenger_native, challenger_native=challenger_native)
                adjusted[:, j] += authority * self.config.strength * calibrated_residual(identity, self.config)
        solver = solve_global(rows, adjusted, ordered, frame=frame)
        validate_global(solver, rows)
        commit = commit_memory(self.bank, self.config, frame=frame, rows=rows, base_solver=baseline_solver or base_solver, solver=solver, public_id=self.target_public, scores=identity, score_state_hash=score_hash)
        if commit["accepted"]:
            self.last_trusted_frame = frame
        by_uid = {str(r["candidate_uid"]): r for r in rows}
        assignments = public_map(solver)
        final = dict(assignments)
        births = []
        deaths = []
        for state in ordered:
            uid = assignments.get(state.public_id)
            if uid is not None:
                r = by_uid[uid]
                state.update_machine(np.asarray(r["feature"]), np.asarray(r["box_xyxy"]), frame, int(r["native_tid"]), 0.9, update_prototype=not (frame == int(self.event["event_frame"]) and state.public_id == self.target_public), native_scope=r.get("native_scope"))
            else:
                if state.state == IdentityState.ACTIVE:
                    state.mark_lost(frame)
                else:
                    state.advance_lost()
                if self.config.lifecycle == "dynamic" and frame - state.last_seen_frame > self.max_lost_gap and state.public_id != self.target_public:
                    state.terminate()
                    deaths.append(state.public_id)
        if self.config.lifecycle == "dynamic":
            for decision in solver["assignment_rows"]:
                if decision["public_id"] is None:
                    r = by_uid[decision["candidate_uid"]]
                    state = self._birth(r, frame)
                    final[state.public_id] = str(r["candidate_uid"])
                    births.append(state.public_id)
        assigned_uids = [uid for uid in final.values() if uid is not None]
        if len(assigned_uids) != len(set(assigned_uids)):
            raise RuntimeError("candidate assigned to multiple public IDs after birth")
        output = [{"public_id": p, "candidate_uid": uid, "box_xyxy": list(by_uid[uid]["box_xyxy"]), "confidence": float(by_uid[uid].get("confidence", 1.0) or 1.0)} for p, uid in sorted(final.items()) if uid is not None]
        after = state_digest(self.states)
        self.frame = frame
        self.previous_challenger_native = challenger_native
        return {"frame": frame, "outputs": output, "assignments": {str(p): u for p, u in final.items()}, "base_assignments": {str(p): u for p, u in assignments.items()} if authority == 0 else {str(p): u for p, u in public_map(base_solver).items()}, "target_public_id": self.target_public, "target_uid": final.get(self.target_public), "authority": authority, "identity_scores": identity.tolist(), "identity_score_source": "LEGACY_BASE_SCORE" if self.config.source == "legacy" else self.config.source, "authority_features": features.tolist(), "births": births, "deaths": deaths, "explicit_none_uids": [r["candidate_uid"] for r in solver["assignment_rows"] if r["public_id"] is None], "state_before": before, "state_after": after, "memory": commit, "runtime_future_gt_used": False, "runtime_gt_read": False, "solver": solver, "base_matrix": base, "fused_matrix": adjusted, "states_before_commit_axis": [s.public_id for s in ordered]}
