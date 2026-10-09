import numpy as np
import torch
from tests.test_n72r21_mot_bridge import inputs
from scripts.n72r21_t2_replay import DecisionCapture
from scripts.n72r21r1_open_set_collect import current_axis
from sam3_intermot.one_click.acib_memory import ACIBMemoryNetwork
from sam3_intermot.one_click.acib_trusted_runtime import TrustedACIBRecognizer
from sam3_intermot.one_click.safe_mot_bridge import SafeMOTIdentityBridge
from sam3_intermot.one_click.intervention_gate import GatePolicy
from sam3_intermot.one_click.mot_bridge import MOTIdentityBridge


def test_all_real_candidates_and_none_are_scored_without_committing_any_branch():
    torch.manual_seed(72111);rows,event=inputs();model=DecisionCapture(ACIBMemoryNetwork()).eval()
    actor=TrustedACIBRecognizer(model,event['human_anchor'],'unit',policy='P0')
    actor.start_recording('unit',fps=20,width=100,height=100,initial_frame=0,initial_box=event['target_box_xyxy'])
    bridge=SafeMOTIdentityBridge(event,actor,policy=GatePolicy(family='shadow'));bridge.configure_fps(20)
    baseline=MOTIdentityBridge(event);bridge.step(0,rows);baseline.step(0,rows)
    record=current_axis(bridge,1,rows)
    assert [r['candidate_uid'] for r in record['axis']]==['p','q',None]
    assert bridge.tracker.frame==0 and bridge.identity.last_frame==0 and not bridge.identity.bank
    assert all(r['action_not_executed'] and np.isfinite(r['feature_vector']).all() for r in record['axis'])
    a,b=bridge.step(1,rows),baseline.step(1,rows)
    assert a['outputs']==b['outputs'] and a['state_after']==b['state_after']
    assert record['own_current_KEEP_UID']==a['target_uid']


def test_open_set_unknown_and_physical_absence_never_become_verified_negative_UID():
    from scripts.n72r21r1_open_set_label import label_uid
    matched={'p':7,'q':8,'unmatched':None}
    assert label_uid('unmatched',matched,7,True)=='UNKNOWN_UNMATCHED'
    assert label_uid('q',matched,7,True)=='VERIFIED_OTHER'
    assert label_uid('p',matched,7,True)=='TARGET'
    assert label_uid(None,matched,7,True)=='INCORRECT_NONE'
    assert label_uid(None,{},7,False)=='CORRECT_NONE'
