"""Fixed low-cost current/past controls; unchanged complete global solver."""
from copy import deepcopy
import numpy as np
from .safe_mot_bridge import SafeMOTIdentityBridge
from .joint_intervention_primitives import prepare_proposal, check_action, commit_action
from .intervention_features import opportunity_features, validate_runtime_rows
from .acib_runtime import overlap
from sam3_intermot.association.opportunity_tracker import OpportunityTracker
from sam3_intermot.association.opportunity_scores import decompose_scores
from sam3_intermot.association.opportunity_solver import AssociationAction


class SelectiveNativeTracker(OpportunityTracker):
    native_discount_active = False

    def _scores(self, ordered, rows, frame):
        matrix = super()._scores(ordered, rows, frame)
        if self.native_discount_active and self.target_public is not None and frame > int(self.event["event_frame"]):
            parts = decompose_scores(ordered, rows, frame)
            j = next(i for i, state in enumerate(ordered) if state.public_id == self.target_public)
            matrix[:, j] -= .5 * parts["native_bonus"][:, j]
            matrix[parts["hard_mask"]] = -1e9
        return matrix


def approve_simple(case, features, *, feasible, changed):
    if not feasible or not changed:
        return False, ["HARD_INFEASIBLE" if not feasible else "SAME_AS_OWN_KEEP"]
    reasons = []
    if case == "P8_SELECTIVE_ABSTENTION":
        tests = {"NOT_NONE_PROPOSAL": features["proposal_NONE"] == 1., "NONE_LOW": features["NONE_probability"] >= .95,
                 "NONE_NOT_DOMINANT": features["NONE_advantage"] >= .05, "KEEP_ANCHOR_PROTECTED": features["KEEP_anchor_cosine"] < .6,
                 "HIGH_GLOBAL_REGRET": features["global_regret"] <= .2, "COMPETITOR_PROTECTED": features["displaced_count"] == 0}
    else:
        tests = {"NONE_NOT_CANDIDATE_REPAIR": not features["proposal_NONE"], "LOW_CONFIDENCE": features["proposed_probability"] >= .9,
                 "LOW_IDENTITY_MARGIN": features["candidate_margin"] >= .05, "LOW_QUALITY": features["quality"] >= .5,
                 "LOW_ANCHOR_AGREEMENT": features["anchor_cosine"] >= .6}
        if case == "P2_RECOVERY_ONLY":
            tests["NOT_STRICT_RECOVERY"] = bool(features["KEEP_NONE"] or features["target_LOST"])
        if case == "P3_BASELINE_UNCERTAINTY":
            tests["KEEP_MARGIN_PROTECTED"] = features["base_KEEP_margin"] <= 1.
        if case in ("P4_IDENTITY_MARGIN_GLOBAL_REGRET", "P5_COMPETITOR_PROTECTION", "P6_PERSISTENT_CAUSAL_CHALLENGER", "P7_SELECTIVE_NATIVE_RELIABILITY", "P9_CONSERVATIVE_GLOBAL_REATTACHMENT"):
            tests.update(LOW_RAW_IDENTITY_ADVANTAGE=features["anchor_advantage_vs_KEEP"] >= .1,
                         IDENTITY_ADVANTAGE_NOT_ABOVE_COST=features["anchor_advantage_vs_KEEP"] > features["global_regret"] / 10.,
                         HIGH_GLOBAL_REGRET=features["global_regret"] <= .2)
        if case in ("P5_COMPETITOR_PROTECTION", "P6_PERSISTENT_CAUSAL_CHALLENGER", "P7_SELECTIVE_NATIVE_RELIABILITY", "P9_CONSERVATIVE_GLOBAL_REATTACHMENT"):
            tests["COMPETITOR_PROTECTED"] = features["displaced_count"] == 0
        if case in ("P6_PERSISTENT_CAUSAL_CHALLENGER", "P7_SELECTIVE_NATIVE_RELIABILITY", "P9_CONSERVATIVE_GLOBAL_REATTACHMENT"):
            tests["AWAIT_CAUSAL_CONFIRMATION"] = features["pending_confirmation_count"] >= 3
        if case in ("P6_PERSISTENT_CAUSAL_CHALLENGER", "P9_CONSERVATIVE_GLOBAL_REATTACHMENT"):
            tests["KEEP_MARGIN_PROTECTED"] = features["base_KEEP_margin"] <= 1.
        if case == "P9_CONSERVATIVE_GLOBAL_REATTACHMENT":
            tests["NO_CAUSAL_IDENTITY_BREAK"] = bool(features["KEEP_NONE"] or features["target_LOST"] or features["track_gap"] >= 3)
        if case == "P7_SELECTIVE_NATIVE_RELIABILITY":
            tests["KEEP_ANCHOR_NOT_CONTRADICTED"] = features["KEEP_anchor_cosine"] < .6
    reasons.extend(key for key, passed in tests.items() if not passed)
    return not reasons, reasons or ["APPROVED_FIXED_CURRENT_PAST_CONTROL"]


class SimpleEventBridge(SafeMOTIdentityBridge):
    def __init__(self, event, identity, *, case, frames):
        super().__init__(event, identity, frames=frames)
        if case not in tuple("P" + str(i) + "_" + name for i, name in enumerate(("KEEP", "SHADOW", "RECOVERY_ONLY", "BASELINE_UNCERTAINTY", "IDENTITY_MARGIN_GLOBAL_REGRET", "COMPETITOR_PROTECTION", "PERSISTENT_CAUSAL_CHALLENGER", "SELECTIVE_NATIVE_RELIABILITY", "SELECTIVE_ABSTENTION", "CONSERVATIVE_GLOBAL_REATTACHMENT"))):
            raise ValueError("Unregistered simple mechanism")
        self.case = case
        if case == "P7_SELECTIVE_NATIVE_RELIABILITY":
            self.tracker = SelectiveNativeTracker(config=self.tracker.config, event=self.tracker.event,
                                                  intervention_policy=self.tracker.intervention_policy, memory_policy=self.tracker.memory_policy)

    def step(self, frame, rows):
        validate_runtime_rows(rows)
        if frame != self.tracker.frame + 1:
            raise ValueError("One original frame one causal global decision")
        if frame <= self.event_frame or self.case in ("P0_KEEP", "P1_SHADOW"):
            return super().step(frame, rows)
        prepared = prepare_proposal(self, frame, rows)
        action = prepared["action"]
        check = check_action(self, frame, rows, prepared["preview"], action)
        candidate = next((r for r in rows if r["candidate_uid"] == prepared["proposal"]["selected_candidate_uid"]), None)
        previous = self.authority_pending
        agreement = 0. if candidate is None or previous is None else float(np.dot(candidate["feature"], previous["feature"]))
        confirmed = candidate is not None and previous is not None and previous["frame"] == frame - 1 and agreement >= .9 and overlap(candidate["box_xyxy"], previous["box"]) >= .3
        count = previous["count"] + 1 if confirmed else 1 if candidate is not None else 0
        self.authority_pending = None if candidate is None else {"frame": frame, "count": count, "feature": np.array(candidate["feature"], copy=True), "box": list(candidate["box_xyxy"])}
        features = opportunity_features(self, frame, rows, prepared, check, confirmations=count, previous_intervention=self.last_intervention, previous_agreement=agreement)
        approved, reasons = approve_simple(self.case, features, feasible=check["feasible"], changed=check.get("assignment_changed", False))
        raw_keep = deepcopy(prepared["preview"])
        native_discount = False
        if self.case == "P7_SELECTIVE_NATIVE_RELIABILITY" and approved:
            state = self.tracker.states[self.tracker.target_public]
            keep_row = next((r for r in rows if r["candidate_uid"] == raw_keep["target_uid"]), None)
            locked = keep_row is not None and state.last_native_tid == int(keep_row["native_tid"]) and state.last_native_scope == keep_row.get("native_scope")
            if not locked:
                approved, reasons = False, ["KEEP_NOT_NATIVE_LOCKED"]
            else:
                native_discount = True
                self.tracker.native_discount_active = True
                prepared = prepare_proposal(self, frame, rows)
                action = AssociationAction("KEEP", self.tracker.target_public)
        actual = action if approved else AssociationAction("KEEP", self.tracker.target_public)
        try:
            result = commit_action(self, frame, rows, prepared, actual)
        finally:
            if self.case == "P7_SELECTIVE_NATIVE_RELIABILITY":
                self.tracker.native_discount_active = False
        effective = {o["public_id"]: o["candidate_uid"] for o in result["outputs"]} != {o["public_id"]: o["candidate_uid"] for o in raw_keep["outputs"]}
        if effective:
            self.last_intervention = frame
        result["authority"] = {"family": self.case, "approved": approved, "effective_assignment_change": effective,
                               "features": features, "reasons": reasons, "proposed_action": prepared["action"].to_dict(),
                               "own_unmodified_KEEP_uid": raw_keep["target_uid"], "own_unmodified_KEEP_outputs": raw_keep["outputs"],
                               "selective_native_discount_active_this_frame_only": native_discount,
                               "native_bonus_discount_fraction": .5 if native_discount else 0., "hard_negatives_unchanged": True,
                               "future_GT_used": False}
        return result
