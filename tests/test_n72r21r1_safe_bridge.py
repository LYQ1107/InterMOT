from dataclasses import replace
import numpy as np
import pytest
from sam3_intermot.one_click.mot_bridge import MOTIdentityBridge
from sam3_intermot.one_click.safe_mot_bridge import SafeMOTIdentityBridge
from sam3_intermot.one_click.intervention_gate import GatePolicy, decide_authority
from sam3_intermot.one_click.intervention_features import FEATURE_NAMES
from sam3_intermot.one_click.runtime import OneClickRecognizer, RuntimeConfig
from tests.test_n72r21_mot_bridge import inputs, actor


@pytest.mark.parametrize('family', ['off', 'shadow'])
def test_full_joint_off_and_shadow_match_C0(family):
    rows, event = inputs()
    baseline = MOTIdentityBridge(event)
    bridge = SafeMOTIdentityBridge(event, actor(event, 'P1'), policy=GatePolicy(family=family))
    bridge.configure_fps(20)
    for f in range(8):
        a, b = bridge.step(f, rows), baseline.step(f, rows)
        assert a['outputs'] == b['outputs'] and a['state_after'] == b['state_after']
        assert not a['authority']['approved']


@pytest.mark.parametrize('forbidden', ['gt_id', 'label', 'future_target_uid', 'runtime_future_gt_used'])
def test_gt_rejected_before_model_forward(forbidden):
    rows, event = inputs()
    bridge = SafeMOTIdentityBridge(event, actor(event)); bridge.configure_fps(20)
    bridge.step(0, rows)
    unsafe = [dict(rows[0], **{forbidden: 1}), rows[1]]
    with pytest.raises(ValueError): bridge.step(1, unsafe)
    assert bridge.tracker.frame == 0 and bridge.identity.last_frame == 0


def test_clones_isolate_joint_state_anchor_banks_and_pending():
    rows, event = inputs()
    bridge = SafeMOTIdentityBridge(event, actor(event, 'P1'))
    bridge.configure_fps(20); bridge.step(0, rows); bridge.step(1, rows)
    a, b = bridge.clone(), bridge.clone()
    public = bridge.tracker.target_public
    a.tracker.states[public].prototype[0] = .25
    a.authority_pending['feature'][0] = .2
    a.identity._bank = []
    assert b.tracker.states[public].prototype[0] == bridge.tracker.states[public].prototype[0] == 1.
    assert b.authority_pending['feature'][0] == bridge.authority_pending['feature'][0] == 1.
    assert len(b.identity.bank) == len(bridge.identity.bank) == 1


def strong_opportunity():
    f = dict.fromkeys(FEATURE_NAMES, 0.)
    f.update(proposed_probability=.99, candidate_margin=.3, anchor_cosine=.9, quality=1.,
             base_KEEP_margin=0., global_regret=.1, pending_confirmation_count=3)
    return f


def test_hard_feasibility_is_not_identity_authority():
    f = strong_opportunity(); f['base_KEEP_margin'] = 8.
    approved, reasons, _ = decide_authority(GatePolicy(family='confidence'), f, feasible=True, proposed_change=True)
    assert not approved and 'KEEP_HAS_STRONG_CONTINUITY' in reasons
    assert decide_authority(GatePolicy(family='original'), f, feasible=True, proposed_change=True)[0]


def test_competitor_and_regret_safety_are_explicit():
    f = strong_opportunity(); f['displaced_count'] = 1
    assert not decide_authority(GatePolicy(family='protected'), f, feasible=True, proposed_change=True)[0]
    f['displaced_count'] = 0; f['global_regret'] = 9.
    assert not decide_authority(GatePolicy(family='global_regret'), f, feasible=True, proposed_change=True)[0]


def test_delayed_and_learned_risk_use_only_current_features():
    f = strong_opportunity(); f['pending_confirmation_count'] = 1
    assert not decide_authority(GatePolicy(family='delayed'), f, feasible=True, proposed_change=True)[0]
    f['pending_confirmation_count'] = 3
    class Predictor:
        def predict(self, x):
            assert x.shape == (len(FEATURE_NAMES),)
            return {'beneficial': .99, 'harmful': .2, 'abstain': .01, 'value': 1.}
    approved, reasons, _ = decide_authority(GatePolicy(family='risk'), f, feasible=True, proposed_change=True, predictor=Predictor())
    assert not approved and 'PREDICTED_HARM' in reasons


def test_rejected_proposal_never_writes_challenger_crop():
    rows, event = inputs()
    challenger = OneClickRecognizer(RuntimeConfig(memory_policy='P1'))
    # Synthetic conflicting actor tests the commit contract, not a second research click.
    challenger.initialize(rows[1]['feature'], rows[1]['box_xyxy'], recording_id='unit', frame=0)
    bridge = SafeMOTIdentityBridge(event, challenger, policy=GatePolicy(family='shadow'))
    bridge.configure_fps(20); bridge.step(0, rows)
    result = bridge.step(1, rows)
    assert result['target_uid'] == 'p' and result['identity_decision']['proposed_candidate_uid'] == 'q'
    assert not result['joint_identity_memory_write'] and not bridge.identity.bank


def test_preclick_identity_proposals_cannot_change_any_public_assignment():
    rows, event = inputs(); event = dict(event, event_frame=2)
    initial = OneClickRecognizer(); initial.initialize(event['human_anchor'], event['target_box_xyxy'], recording_id='unit', frame=2)
    bridge = SafeMOTIdentityBridge(event, initial, policy=GatePolicy(family='original'))
    bridge.configure_fps(20); baseline = MOTIdentityBridge(event)
    for f in range(3):
        a, b = bridge.step(f, rows), baseline.step(f, rows)
        assert a['outputs'] == b['outputs'] and a['state_after'] == b['state_after']
        assert initial.last_frame == 2 and not a['authority']['approved']
