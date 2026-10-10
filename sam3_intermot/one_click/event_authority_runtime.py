"""Development-only learned event authority, exact current global commits."""
from copy import deepcopy
import numpy as np
import torch
from .safe_mot_bridge import SafeMOTIdentityBridge
from .joint_intervention_primitives import prepare_proposal, check_action, commit_action
from .intervention_features import opportunity_features, feature_vector, validate_runtime_rows
from .event_authority_learning import BRANCH_NAMES, encode_runtime_input, normalize_inputs
from .event_authority_models import EventAuthorityHead, runtime_predictions
from .acib_runtime import overlap
from sam3_intermot.association.opportunity_solver import AssociationAction, action_for_candidate


class EventAuthorityPredictor:
    def __init__(self, saved, *, development_diagnostic=False):
        if saved["schema"] != "N72R21R2_EVENT_AUTHORITY_V1":
            raise ValueError("Wrong checkpoint lineage")
        if saved["authority_status"] == "TRAINED_UNCALIBRATED_NOT_DEPLOYABLE" and not development_diagnostic:
            raise ValueError("Uncalibrated checkpoint has no deployment authority")
        self.model = EventAuthorityHead(saved["family"], hidden=saved["hidden"])
        self.model.load_state_dict(saved["model"], strict=True)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.mean = np.asarray(saved["FIT_mean"], dtype=np.float32)
        self.scale = np.asarray(saved["FIT_scale"], dtype=np.float32)
        if self.mean.shape != (32,) or self.scale.shape != (32,) or not np.isfinite(self.mean).all() or not np.isfinite(self.scale).all() or np.any(self.scale <= 0):
            raise ValueError("Invalid frozen FIT-only normalizer")

    def predict(self, runtime, branch, keep_runtime):
        x, past = encode_runtime_input(runtime, branch)
        keep, _ = encode_runtime_input(keep_runtime, "KEEP")
        x, past = normalize_inputs(x[None], past[None], self.mean, self.scale)
        keep, _ = normalize_inputs(keep[None], np.zeros((1, 3, 32)), self.mean, self.scale)
        with torch.inference_mode():
            return runtime_predictions(self.model(torch.from_numpy(x), torch.from_numpy(past), keep_current=torch.from_numpy(keep)))[0]


def qualified_current_candidate(features, prediction, point, *, feasible, changed, confirmations):
    if not feasible or not changed or confirmations < point["confirmation_delay"]:
        return False
    if prediction["beneficial"] < point["claim_min"] or prediction["harmful"] > point["risk_max"] or prediction["value"] <= 0:
        return False
    if features["global_regret"] > point["global_regret_max"] or features["displaced_count"] > 0:
        return False
    if features["proposal_NONE"]:
        return features["NONE_probability"] >= .95 and features["NONE_advantage"] >= .05 and features["KEEP_anchor_cosine"] < .6
    return features["anchor_advantage_vs_KEEP"] >= point["anchor_advantage_min"] and features["quality"] >= .5 and features["anchor_cosine"] >= .6


class LearnedEventBridge(SafeMOTIdentityBridge):
    def __init__(self, event, identity, *, predictor, point, frames):
        super().__init__(event, identity, frames=frames)
        self.event_predictor, self.point = predictor, deepcopy(point)
        self.event_feature_history = {}
        self.event_branch_pending = {}

    def clone(self):
        result = super().clone()
        result.event_feature_history = deepcopy(self.event_feature_history)
        result.event_branch_pending = deepcopy(self.event_branch_pending)
        return result

    def step(self, frame, rows):
        validate_runtime_rows(rows)
        if frame != self.tracker.frame + 1:
            raise ValueError("One original frame one global learned decision")
        if frame <= self.event_frame:
            return super().step(frame, rows)
        prepared = prepare_proposal(self, frame, rows)
        public = self.tracker.target_public
        preview = prepared["preview"]
        candidate = next((r for r in rows if r["candidate_uid"] == prepared["proposal"]["selected_candidate_uid"]), None)
        previous = self.authority_pending
        agreement = 0. if previous is None or candidate is None else float(np.dot(previous["feature"], candidate["feature"]))
        consistent = previous is not None and candidate is not None and previous["frame"] == frame - 1 and agreement >= .9 and overlap(previous["box"], candidate["box_xyxy"]) >= .3
        count = previous["count"] + 1 if consistent else 1 if candidate is not None else 0
        self.authority_pending = None if candidate is None else {"frame": frame, "count": count, "feature": np.array(candidate["feature"], copy=True), "box": list(candidate["box_xyxy"])}
        history = {"H" + str(h): [self.event_feature_history.get(f, [0.] * 32) for f in range(frame - h, frame)] for h in (3, 8)}
        probabilities = prepared["provisional"].model.last["joint_probabilities"][0, :len(rows)].detach().cpu().numpy().copy()
        anchor_scores = [float(np.dot(self.tracker.event["human_anchor"], r["feature"])) for r in rows]
        uids = [r["candidate_uid"] for r in rows]
        owners = {o["candidate_uid"]: o["public_id"] for o in preview["outputs"]}
        choices, seen = [], set()
        for name in BRANCH_NAMES:
            if name == "DELAYED_CHALLENGER":
                continue  # Its CF label is a wait-policy value, not a current forced edge.
            if name == "KEEP":
                action = AssociationAction("KEEP", public)
            elif name == "REJECT_TARGET":
                action = AssociationAction("REJECT_TARGET", public)
            else:
                if name == "RECOVERY" and preview["target_uid"] is not None and self.tracker.states[public].state != "LOST":
                    continue
                eligible = [i for i in range(len(rows)) if (name != "TOP_ALTERNATIVE" or uids[i] != preview["target_uid"])
                            and (name != "FEASIBLE_GLOBAL_SWAP" or owners.get(uids[i]) not in (None, public))]
                if not eligible:
                    continue
                scores = probabilities if name == "LEARNED_IDENTITY_TOP" else anchor_scores
                i = min(eligible, key=lambda i: (-scores[i], uids[i]))
                action = action_for_candidate(public, uids[i], preview["solver"])
            signature = (action.family, action.candidate_uid)
            if signature in seen:
                continue
            seen.add(signature)
            check = check_action(self, frame, rows, preview, action)
            if not check["feasible"]:
                continue
            uid = preview["target_uid"] if action.family == "KEEP" else action.candidate_uid
            proposed_row = next((r for r in rows if r["candidate_uid"] == uid), None)
            past_branch = self.event_branch_pending.get(name)
            branch_agreement = 0. if past_branch is None or proposed_row is None or past_branch.get("kind") == "NONE" else float(np.dot(past_branch["feature"], proposed_row["feature"]))
            branch_consistent = past_branch is not None and proposed_row is not None and past_branch.get("kind") != "NONE" and past_branch["frame"] == frame - 1 and branch_agreement >= .9 and overlap(past_branch["box"], proposed_row["box_xyxy"]) >= .3
            none_consistent = proposed_row is None and past_branch is not None and past_branch.get("kind") == "NONE" and past_branch["frame"] == frame - 1
            branch_count = past_branch["count"] + 1 if branch_consistent or none_consistent else 1
            if proposed_row is not None:
                self.event_branch_pending[name] = {"kind": "CANDIDATE", "frame": frame, "count": branch_count, "feature": np.array(proposed_row["feature"], copy=True), "box": list(proposed_row["box_xyxy"])}
            else:
                self.event_branch_pending[name] = {"kind": "NONE", "frame": frame, "count": branch_count}
            proposal = {**prepared, "proposal": {**prepared["proposal"], "selected_candidate_uid": uid}}
            features = opportunity_features(self, frame, rows, proposal, check, confirmations=branch_count,
                                            previous_intervention=self.last_intervention, previous_agreement=branch_agreement)
            runtime = {"features": features, "feature_vector": feature_vector(features).tolist(), "causal_previous_feature_vectors": history}
            choices.append({"branch": name, "action": action, "check": check, "runtime": runtime, "confirmations": branch_count})
        keep = next(c for c in choices if c["branch"] == "KEEP")
        for c in choices:
            c["prediction"] = self.event_predictor.predict(c["runtime"], c["branch"], keep["runtime"])
            c["approved"] = c["branch"] != "KEEP" and qualified_current_candidate(c["runtime"]["features"], c["prediction"], self.point,
                feasible=c["check"]["feasible"], changed=c["check"].get("assignment_changed", False), confirmations=c["confirmations"])
        approved = [c for c in choices if c["approved"]]
        chosen = min(approved, key=lambda c: (-c["prediction"]["value"], c["branch"], str(c["action"].candidate_uid))) if approved else keep
        primary_check = check_action(self, frame, rows, preview, prepared["action"])
        history_features = opportunity_features(self, frame, rows, prepared, primary_check, confirmations=count,
                                               previous_intervention=self.last_intervention, previous_agreement=agreement)
        result = commit_action(self, frame, rows, prepared, chosen["action"])
        effective = result["outputs"] != preview["outputs"]
        if effective:
            self.last_intervention = frame
        # History is the own current frozen identity proposal, not a future label
        # or another counterfactual arm's state/feature cache.
        self.event_feature_history[frame] = feature_vector(history_features).tolist()
        result["authority"] = {"family": "N72R21R2_LEARNED_EVENT_DEVELOPMENT_DIAGNOSTIC", "approved": bool(approved),
                               "effective_assignment_change": effective, "chosen_branch": chosen["branch"], "features": chosen["runtime"]["features"],
                               "prediction": chosen["prediction"], "point": self.point, "own_unmodified_KEEP_outputs": preview["outputs"],
                               "all_current_choices": [{"branch": c["branch"], "action": c["action"].to_dict(), "prediction": c["prediction"], "approved": c["approved"], "features": c["runtime"]["features"]} for c in choices],
                               "past_distribution": "OWN_POLICY_PRIMARY_CURRENT_CHALLENGER_FEATURES_NOT_FUTURE_CF_STATE",
                               "current_distribution_shift_vs_C0_training_requires_audit": True,
                               "DELAYED_CF_WAIT_POLICY_not_misused_as_current_edge": True, "future_GT_used": False}
        return result
