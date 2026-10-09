from dataclasses import replace
import hashlib
import numpy as np
import pytest
import torch
from tests.test_n72r21_mot_bridge import inputs
from scripts.n72r21_t2_replay import DecisionCapture
from sam3_intermot.one_click.acib_memory import ACIBMemoryNetwork
from sam3_intermot.one_click.committed_identity_memory import MemoryCommitPolicy,CommittedIdentityMemory
from sam3_intermot.one_click.committed_memory_bridge import CommittedMemoryMOTBridge
from sam3_intermot.one_click.mot_bridge import MOTIdentityBridge


def runtime(policy,*,wrong_proposal=False):
    rows,event=inputs();torch.manual_seed(72111)
    anchor=rows[1]['feature'] if wrong_proposal else event['human_anchor']
    actor=CommittedIdentityMemory(DecisionCapture(ACIBMemoryNetwork()).eval(),anchor,'unit',write_policy=policy)
    actor.start_recording('unit',fps=20,width=100,height=100,initial_frame=0,initial_box=event['target_box_xyxy'])
    bridge=CommittedMemoryMOTBridge(event,actor);bridge.configure_fps(20)
    return rows,event,bridge


@pytest.mark.parametrize('capacity',[1,4,8])
@pytest.mark.parametrize('aggregation',['mean','confidence','attention'])
def test_capacity_and_aggregation_never_replace_anchor_or_change_shadow_MOT(capacity,aggregation):
    rows,event,bridge=runtime(MemoryCommitPolicy(family='unsafe',capacity=capacity,aggregation=aggregation))
    baseline=MOTIdentityBridge(event);anchor_SHA=bridge.identity.anchor_sha
    for f in range(6):
        actual,base=bridge.step(f,rows),baseline.step(f,rows)
        assert actual['outputs']==base['outputs'] and actual['state_after']==base['state_after']
        if f:assert actual['joint_identity_memory_write'] and actual['joint_memory_write_candidate_uid']=='p'
        assert len(bridge.identity.bank)==min(f,capacity)
        assert hashlib.sha256(bridge.identity.anchor.tobytes()).hexdigest()==anchor_SHA
    assert all(e.candidate_uid=='p' and not e.embedding.flags.writeable for e in bridge.identity.bank)


def test_pending_crop_cannot_affect_identity_bank_until_current_commit_confirmed():
    policy=MemoryCommitPolicy(family='delayed',probability_min=0.,anchor_min=-1.,quality_min=0.,motion_min=0.,delay_frames=2)
    rows,event,bridge=runtime(policy);bridge.step(0,rows);first=bridge.step(1,rows)
    assert bridge.identity.write_pending and not bridge.identity.bank and not first['joint_identity_memory_write']
    second=bridge.step(2,rows)
    assert second['joint_identity_memory_write'] and bridge.identity.bank[0].frame==2
    assert not second['committed_memory_diagnostic']['pending_used_for_identity_score']


def test_rejected_proposal_crop_never_written_even_unsafe_control():
    rows,event,bridge=runtime(MemoryCommitPolicy(family='unsafe'),wrong_proposal=True)
    bridge.step(0,rows);actual=bridge.step(1,rows)
    assert actual['identity_decision']['proposed_candidate_uid']!=actual['target_uid']
    assert actual['target_uid']=='p' and bridge.identity.bank[-1].candidate_uid=='p'
    assert np.array_equal(bridge.identity.bank[-1].embedding,rows[0]['feature'])


def test_diversity_does_not_count_repeated_same_tracklet_as_new_identity_evidence():
    policy=MemoryCommitPolicy(family='diverse',delay_frames=1,probability_min=0.,anchor_min=-1.,quality_min=0.,motion_min=0.)
    rows,event,bridge=runtime(policy);bridge.step(0,rows);bridge.step(1,rows)
    for f in range(2,9):bridge.step(f,rows)
    assert len(bridge.identity.bank)==1


def test_cloned_pending_trusted_and_rollback_snapshots_are_isolated():
    rows,event,bridge=runtime(MemoryCommitPolicy(family='unsafe'));bridge.step(0,rows);bridge.step(1,rows)
    left,right=bridge.clone(),bridge.clone();left.identity.write_pending['feature'][0]=.1;left.identity.write_snapshots=[]
    assert right.identity.write_pending['feature'][0]==bridge.identity.write_pending['feature'][0]==1.
    assert len(right.identity.write_snapshots)==len(bridge.identity.write_snapshots)==1


def test_double_commit_or_GT_input_cannot_write_a_second_crop():
    rows,event,bridge=runtime(MemoryCommitPolicy(family='unsafe'));bridge.step(0,rows);actual=bridge.step(1,rows)
    with pytest.raises(ValueError):bridge.identity.observe_committed(1,rows,'p',actual['committed_observation_features'])
    bad=[dict(rows[0],gt_id=1),rows[1]]
    with pytest.raises(ValueError):bridge.step(2,bad)
    assert bridge.tracker.frame==1


def test_risk_writer_is_rejected_before_any_runtime_without_fitted_head():
    with pytest.raises(ValueError,match='fitted joint-state writer'):
        runtime(MemoryCommitPolicy(family='risk'))


def test_training_write_pair_copies_own_committed_crop_only():
    from scripts.n72r21r1_memory_collect import force_current_write
    rows,event,bridge=runtime(MemoryCommitPolicy(family='frozen'));bridge.step(0,rows);actual=bridge.step(1,rows)
    arm=bridge.clone();result=force_current_write(arm,1,rows,actual,actual['committed_observation_features'])
    assert result['joint_memory_write_candidate_uid']==actual['target_uid']=='p'
    assert result['outputs']==actual['outputs'] and result['state_after']==actual['state_after']
    assert not bridge.identity.bank and len(arm.identity.bank)==1
    assert not arm.identity.bank[-1].embedding.flags.writeable
    invalid=dict(actual,target_uid='q')
    with pytest.raises(ValueError):force_current_write(bridge,1,rows,invalid,actual['committed_observation_features'])


def test_rollback_uses_current_causal_contradiction_and_preserves_anchor():
    from sam3_intermot.one_click.acib_runtime import Evidence
    from sam3_intermot.one_click.intervention_features import FEATURE_NAMES
    rows,event,bridge=runtime(MemoryCommitPolicy(family='rollback',delay_frames=1))
    actor=bridge.identity;bad=rows[1]['feature'].copy();bad.setflags(write=False)
    actor.bank=[Evidence(bad,'unit',0,None,0.,1.,'SYNTHETIC_BAD_PRIOR',1.,0.,'q')]
    actor.write_snapshots=[tuple()]
    current=[dict(rows[0],feature=bad),rows[1]];features=dict.fromkeys(FEATURE_NAMES,0.)
    features.update(proposed_probability=1.,quality=1.,motion_iou=1.,native_same=1.)
    actor.step(1,current);first=actor.observe_committed(1,current,'p',features)
    assert not first['rollback'] and len(actor.bank)==1
    actor.step(2,current);second=actor.observe_committed(2,current,'p',features)
    assert second['rollback'] and not actor.bank and not second['accepted']
    assert np.array_equal(actor.anchor,event['human_anchor']) and actor.rollback_count==1
