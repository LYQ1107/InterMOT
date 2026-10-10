from copy import deepcopy
import numpy as np
import pytest
import torch
from sam3_intermot.one_click.acib_memory import ACIBMemoryNetwork
from sam3_intermot.one_click.acib_trusted_runtime import TrustedACIBRecognizer
from scripts.n72r21_t2_replay import DecisionCapture
from sam3_intermot.one_click.safe_mot_bridge import SafeMOTIdentityBridge
from sam3_intermot.one_click.intervention_gate import GatePolicy
from sam3_intermot.one_click.event_authority_runtime import LearnedEventBridge
from sam3_intermot.one_click.matched_event_observer import MatchedEventObserver, _RejectAll, POINT, causal_bridge_fingerprint
from sam3_intermot.one_click.joint_intervention_primitives import prepare_proposal, commit_action
from sam3_intermot.association.opportunity_solver import AssociationAction


def setup():
    vector = np.zeros(512, np.float32)
    vector[0] = 1.
    event = dict(event_frame=0, human_anchor=vector, target_candidate_uid="0",
                 target_box_xyxy=[0, 0, 20, 80])
    torch.set_num_threads(1)
    torch.manual_seed(7)
    actor = TrustedACIBRecognizer(DecisionCapture(ACIBMemoryNetwork()), vector, "unit", policy="P0")
    actor.start_recording("unit", fps=20., width=200, height=100, initial_frame=0, initial_box=[0, 0, 20, 80])
    return event, actor, vector


def rows(frame, vector):
    return [dict(candidate_uid=str(frame), feature=vector, box_xyxy=[0, 0, 20, 80], native_tid=1, conf=.99)]


def test_observation_matches_exact_deployment_features_and_does_not_commit():
    event, actor, vector = setup()
    bridge = SafeMOTIdentityBridge(event, deepcopy(actor), policy=GatePolicy(family="shadow"), frames=8)
    deployment = LearnedEventBridge(event, deepcopy(actor), predictor=_RejectAll(), point=POINT, frames=8)
    observer = MatchedEventObserver()
    for current in (bridge, deployment):
        current.configure_fps(20.)
        current.step(0, rows(0, vector))
    for frame in range(1, 5):
        before = causal_bridge_fingerprint(bridge)
        sample = observer.observe(bridge, frame, rows(frame, vector))
        assert before == causal_bridge_fingerprint(bridge)
        expected = deployment.step(frame, rows(frame, vector))["authority"]["all_current_choices"]
        assert [(c["branch"], c["runtime"]["features"]) for c in sample["choices"]] == [(c["branch"], c["features"]) for c in expected]
        actual = bridge.step(frame, rows(frame, vector))
        observer.accept_commit(bridge, frame, sample, actual)
        assert observer.history == deployment.event_feature_history
        assert sample["choices"][0]["runtime"]["features"]["pending_confirmation_count"] == frame


def test_forced_commit_keeps_its_real_state_not_shadow_keep_future():
    event, actor, vector = setup()
    bridge = SafeMOTIdentityBridge(event, actor, policy=GatePolicy(family="shadow"), frames=8)
    bridge.configure_fps(20.)
    bridge.step(0, rows(0, vector))
    observer = MatchedEventObserver()
    sample = observer.observe(bridge, 1, rows(1, vector))
    actual = commit_action(bridge, 1, rows(1, vector), prepare_proposal(bridge, 1, rows(1, vector)),
                           AssociationAction("REJECT_TARGET", bridge.tracker.target_public))
    observer.accept_commit(bridge, 1, sample, actual)
    assert actual["target_uid"] is None
    assert bridge.last_intervention == 1
    future = observer.observe(bridge, 2, rows(2, vector))
    assert future["choices"][0]["runtime"]["features"]["previous_intervention_age"] == 1
    clone = observer.clone()
    clone.history[1][0] += 1.
    assert clone.history != observer.history
    with pytest.raises(ValueError):
        observer.accept_commit(bridge, 1, sample, actual)


def test_observer_rejects_gt_and_noncurrent_frame():
    event, actor, vector = setup()
    bridge = SafeMOTIdentityBridge(event, actor, policy=GatePolicy(family="shadow"), frames=8)
    bridge.configure_fps(20.)
    bridge.step(0, rows(0, vector))
    observer = MatchedEventObserver()
    with pytest.raises(ValueError):
        observer.observe(bridge, 2, rows(2, vector))
    with pytest.raises(ValueError):
        observer.observe(bridge, 1, [{**rows(1, vector)[0], "gt_id": 1}])
