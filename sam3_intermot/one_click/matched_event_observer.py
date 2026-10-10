"""Deployment-matched current/past features without association authority.

Reuse the exact learned bridge implementation on an isolated always-KEEP
view. The view's core post-KEEP state is NEVER installed in the real bridge:
only causal confirmation/history metadata is accepted after the real commit.
Frozen models may share their ephemeral ``last`` inference cache, so all
features are copied before any future arm and every commit recomputes logits.
"""
from copy import deepcopy
from dataclasses import asdict, is_dataclass

from .causal_state_fingerprint import fingerprint, full_tracker_fingerprint, serial
from .event_authority_runtime import LearnedEventBridge
from .intervention_features import feature_vector, validate_runtime_rows


class _RejectAll:
    def predict(self, runtime, branch, keep_runtime):
        return {"beneficial": 0., "harmful": 1., "value": 0.}


POINT = dict(claim_min=.8, risk_max=.02, global_regret_max=.2,
             anchor_advantage_min=.1, confirmation_delay=3)


def causal_bridge_fingerprint(bridge):
    # Includes actual actor tensors/pending state, not shared inference caches.
    def plain(value):
        if is_dataclass(value):
            return plain(asdict(value))
        if isinstance(value, dict):
            return {str(k): plain(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [plain(v) for v in value]
        return serial(value)
    actor = None if bridge.identity is None else plain({
        key: value for key, value in vars(bridge.identity).items() if key != "model"
    })
    extra = {k: getattr(bridge.tracker, k) for k in
             ("pending", "native_streak", "last_native_key", "last_observation_frame",
              "observed_native_streak", "contradiction_streak", "trust_snapshot")
             if hasattr(bridge.tracker, k)}
    return fingerprint({"tracker": full_tracker_fingerprint(bridge), "tracker_manager": plain(extra),
                        "actor": serial(actor), "authority_pending": bridge.authority_pending,
                        "last_intervention": bridge.last_intervention})


class MatchedEventObserver:
    def __init__(self):
        self.history = {}
        self.branch_pending = {}
        self.last_observed_frame = None

    def clone(self):
        return deepcopy(self)

    def state_fingerprint(self):
        return fingerprint(vars(self))

    def observe(self, bridge, frame, rows):
        validate_runtime_rows(rows)
        if frame <= bridge.event_frame or frame != bridge.tracker.frame + 1:
            raise ValueError("Only the next actual post-click frame is observable")
        before = causal_bridge_fingerprint(bridge)
        observer_before = self.state_fingerprint()
        # An initialized isolated bridge supplies all core state. No JSON
        # snapshot restoration or future-branch state is substituted.
        core = bridge.clone()
        view = object.__new__(LearnedEventBridge)
        view.__dict__ = core.__dict__
        view.event_predictor = _RejectAll()
        view.point = deepcopy(POINT)
        view.event_feature_history = deepcopy(self.history)
        view.event_branch_pending = deepcopy(self.branch_pending)
        result = LearnedEventBridge.step(view, frame, rows)
        assert result["authority"]["chosen_branch"] == "KEEP"
        assert not result["authority"]["effective_assignment_change"]
        choices = []
        history = {"H" + str(h): [deepcopy(self.history.get(f, [0.] * 32))
                                  for f in range(frame - h, frame)] for h in (3, 8)}
        for item in result["authority"]["all_current_choices"]:
            choices.append({"branch": item["branch"], "action": deepcopy(item["action"]),
                            "runtime": {"features": deepcopy(item["features"]),
                                        "feature_vector": feature_vector(item["features"]).tolist(),
                                        "causal_previous_feature_vectors": deepcopy(history)}})
        assert before == causal_bridge_fingerprint(bridge)
        assert observer_before == self.state_fingerprint()
        return {"frame": frame, "prestate_sha256": before,
                "observer_prestate_sha256": observer_before,
                "full_tracker_prestate_sha256": full_tracker_fingerprint(bridge),
                "own_KEEP_outputs": deepcopy(result["outputs"]), "choices": choices,
                "next_primary_vector": deepcopy(view.event_feature_history[frame]),
                "next_authority_pending": deepcopy(view.authority_pending),
                "next_branch_pending": deepcopy(view.event_branch_pending)}

    def accept_commit(self, bridge, frame, observation, result):
        if (observation["frame"] != frame or bridge.tracker.frame != frame
                or self.last_observed_frame is not None and frame != self.last_observed_frame + 1):
            raise ValueError("Current observation must be accepted once, in original-frame order")
        if self.state_fingerprint() != observation["observer_prestate_sha256"]:
            raise ValueError("Another branch's pending/history cannot replace this observer")
        bridge.authority_pending = deepcopy(observation["next_authority_pending"])
        if result["outputs"] != observation["own_KEEP_outputs"]:
            bridge.last_intervention = frame
        self.history[frame] = deepcopy(observation["next_primary_vector"])
        self.branch_pending = deepcopy(observation["next_branch_pending"])
        self.last_observed_frame = frame
