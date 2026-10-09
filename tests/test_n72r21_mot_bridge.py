import numpy as np
from sam3_intermot.one_click.mot_bridge import MOTIdentityBridge,copy_identity,committed_feedback
from sam3_intermot.one_click.runtime import OneClickRecognizer,RuntimeConfig
from sam3_intermot.association.opportunity_tracker import OpportunityTracker,InterventionPolicy
from sam3_intermot.association.opportunity_memory import MemoryPolicy
from sam3_intermot.association.identity_authority import AuthorityConfig


def inputs():
    vectors=np.eye(2,512,dtype=np.float32)
    rows=[{'candidate_uid':uid,'feature':v,'box_xyxy':[20*i,0,20*i+10,20],'conf':1.,'native_tid':i+1} for i,(uid,v) in enumerate(zip(('p','q'),vectors))]
    event={'event_frame':0,'human_anchor':vectors[0],'target_candidate_uid':'p','target_box_xyxy':rows[0]['box_xyxy']}
    return rows,event


def actor(event,policy='P0'):
    result=OneClickRecognizer(RuntimeConfig(memory_policy=policy))
    result.initialize(event['human_anchor'],event['target_box_xyxy'],recording_id='unit',frame=0)
    return result


def test_C0_bridge_matches_unmodified_joint_tracker_full_outputs_and_state():
    rows,event=inputs();bridge=MOTIdentityBridge(event)
    original=OpportunityTracker(config=AuthorityConfig(mode='off',source='raw',memory='P0',lifecycle='dynamic'),
        event=event,bank=None,intervention_policy=InterventionPolicy(family='C0',source='raw'),memory_policy=MemoryPolicy('P0'))
    for f in range(3):
        a=bridge.step(f,rows);b=original.step(rows,f)
        assert a['outputs']==b['outputs'] and a['state_after']==b['state_after']
        assert len(a['outputs'])==len(rows) and not a['joint_identity_memory_write']


def test_NO_HUMAN_control_never_creates_a_clicked_target():
    rows,event=inputs();bridge=MOTIdentityBridge(event,no_human=True,frames=3)
    for f in range(3):
        result=bridge.step(f,rows)
        assert result['target_public_id'] is None and len(result['outputs'])==2


def test_rule_proposal_commits_current_crop_only_after_joint_ownership():
    rows,event=inputs();bridge=MOTIdentityBridge(event,actor(event,'P1'));bridge.configure_fps(20)
    bridge.step(0,rows);d=bridge.step(1,rows)
    assert d['target_uid']=='p' and d['joint_memory_write_candidate_uid']=='p'
    assert d['joint_identity_memory_write'] and len(bridge.identity.bank)==1
    assert not d['runtime_gt_read'] and not d['runtime_future_gt_used']


def test_infeasible_proposal_vetoes_write_and_feeds_back_actual_joint_selection():
    rows,event=inputs();initial=actor(event,'P1');before=copy_identity(initial);provisional=copy_identity(initial)
    from sam3_intermot.one_click.runtime import Candidate
    proposal=provisional.step(1,[Candidate(r['candidate_uid'],r['feature'],tuple(r['box_xyxy']),1.) for r in rows],fps=20)
    assert proposal['memory_write']
    state,d=committed_feedback(before,provisional,proposal,rows,1,'q')
    assert not state.bank and state.last_frame==1 and state.last_box==tuple(rows[1]['box_xyxy'])
    assert not d['memory_write'] and d['memory_write_candidate_uid'] is None
    assert d['joint_memory_write_vetoed'] and d['proposed_candidate_uid']=='p' and d['selected_candidate_uid']=='q'
    assert not initial.bank and initial.last_frame==0
