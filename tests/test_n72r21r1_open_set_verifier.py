import numpy as np
import pytest
import torch
from tests.test_n72r21_mot_bridge import inputs
from scripts.n72r21_t2_replay import DecisionCapture
from scripts.n72r21r1_open_set_collect import current_axis
from sam3_intermot.one_click.acib_memory import ACIBMemoryNetwork
from sam3_intermot.one_click.acib_trusted_runtime import TrustedACIBRecognizer
from sam3_intermot.one_click.mot_bridge import MOTIdentityBridge
from sam3_intermot.one_click.joint_intervention_primitives import prepare_proposal
from sam3_intermot.one_click.open_set_verifier import OpenSetVerifierHead,OpenSetMOTIdentityBridge,candidate_axis,choose


def selection(**kw):
    return {'status':'CALIBRATED_CURRENT_IDENTITY_DIAGNOSTIC','probability_min':.9,'margin_min':.05,
            'unknown_max':.02,'temperature':1.,'global_regret_max':.2,**kw}


class Predictor:
    selection=selection()
    def predict_axis(self,axis):
        return [{'correct':.99 if a['candidate_uid']=='q' else .001,'incorrect':.001,
                 'unknown':.001,'available':.99} for a in axis]


def bridge(event,intervene=False):
    torch.manual_seed(72111)
    actor=TrustedACIBRecognizer(DecisionCapture(ACIBMemoryNetwork()).eval(),event['human_anchor'],'unit',policy='P0')
    actor.start_recording('unit',fps=20,width=100,height=100,initial_frame=0,initial_box=event['target_box_xyxy'])
    result=OpenSetMOTIdentityBridge(event,actor,verifier=Predictor(),intervene=intervene)
    result.configure_fps(20);return result


def test_all_axis_features_exactly_match_sealed_training_collector():
    rows,event=inputs();runtime=bridge(event);runtime.step(0,rows)
    expected=current_axis(runtime,1,rows)['axis']
    actual=candidate_axis(runtime,1,rows,prepare_proposal(runtime,1,rows))
    assert expected==actual
    assert runtime.tracker.frame==0 and runtime.identity.last_frame==0


def test_open_set_shadow_exact_full_C0_and_own_committed_feedback():
    rows,event=inputs();runtime=bridge(event);baseline=MOTIdentityBridge(event)
    for frame in range(5):
        a,b=runtime.step(frame,rows),baseline.step(frame,rows)
        assert a['outputs']==b['outputs'] and a['state_after']==b['state_after']
        assert not a['joint_identity_memory_write'] and not a['authority']['approved']
        assert not runtime.identity.bank
        if frame:
            assert list(runtime.identity.last_box)==rows[0]['box_xyxy']
            assert a['authority']['immutable_original_ACIB_feedback']


def test_verifier_does_not_steal_owned_candidate_despite_high_correctness():
    rows,event=inputs();runtime=bridge(event,True);runtime.step(0,rows)
    a=runtime.step(1,rows)
    assert a['target_uid']=='p' and not a['authority']['approved']
    assert 'OTHER_PUBLIC_ID_PROTECTED' in a['authority']['protection_reasons']
    assert a['authority']['selected_identity_claim']['candidate_uid']=='q'


def test_unknown_is_abstain_not_verified_negative_or_forced_NONE():
    axis=[{'candidate_uid':'p','features':{},'action':{'family':'KEEP'}},
          {'candidate_uid':None,'features':{},'action':{'family':'REJECT_TARGET'}}]
    pred=[{'correct':.99,'incorrect':0.,'unknown':.9,'available':.5},
          {'correct':.01,'incorrect':.99,'unknown':0.,'available':.5}]
    r=choose(axis,pred,selection())
    assert not r['accepted'] and r['candidate_uid']=='p' and 'UNKNOWN_ABSTAIN' in r['reasons']
    pred[0]['correct']=.001;pred[1]['correct']=.999
    r=choose(axis,pred,selection());assert r['accepted'] and r['candidate_uid'] is None


@pytest.mark.parametrize('family',['SCALAR','LOGISTIC','MLP'])
def test_each_head_has_actual_nonzero_gradients_and_three_classes_plus_availability(family):
    torch.manual_seed(1);model=OpenSetVerifierHead(family);x=torch.randn(9,32);out=model(x)
    assert out.shape==(9,4)
    torch.nn.functional.cross_entropy(out[:,:3],torch.arange(9)%3).backward()
    assert sum(float(p.grad.abs().sum()) for p in model.parameters() if p.grad is not None)>0


def test_axis_rejects_absent_NONE_and_clone_pending_isolated():
    with pytest.raises(ValueError):choose([],[],selection())
    rows,event=inputs();runtime=bridge(event);runtime.step(0,rows);runtime.step(1,rows)
    left,right=runtime.clone(),runtime.clone()
    if left.authority_pending:
        left.authority_pending['feature'][0]=123
        assert not np.array_equal(left.authority_pending['feature'],right.authority_pending['feature'])
    left.selection['probability_min']=0
    assert right.selection['probability_min']==.9
