import numpy as np
import torch
from tests.test_n72r21_mot_bridge import inputs
from sam3_intermot.one_click.acib_memory import ACIBMemoryNetwork
from sam3_intermot.one_click.acib_trusted_runtime import TrustedACIBRecognizer
from sam3_intermot.one_click.learned_mot_bridge import LearnedMOTIdentityBridge, current_candidate_actions
from sam3_intermot.one_click.joint_intervention_primitives import prepare_proposal
from sam3_intermot.one_click.mot_bridge import MOTIdentityBridge
from scripts.n72r21_t2_replay import DecisionCapture
from scripts.n72r21r1_collect_joint import sample_actions


class Predictor:
    selection = {'status':'CALIBRATED_DIAGNOSTIC','benefit_min':.9,'harm_max':.02,'value_min':0.}
    def predict(self, current, history):
        assert current.shape == (32,) and history.shape == (3,32)
        return {'beneficial':.99,'harmful':.001,'abstain':.009,'value':float(1-current[22])}


def bridge(event, *, intervene=False):
    torch.manual_seed(72111)
    actor = TrustedACIBRecognizer(DecisionCapture(ACIBMemoryNetwork()).eval(),event['human_anchor'],'unit',policy='P0')
    actor.start_recording('unit',fps=20,width=200,height=200,initial_frame=0,initial_box=event['target_box_xyxy'])
    result = LearnedMOTIdentityBridge(event,actor,predictor=Predictor(),intervene=intervene)
    result.configure_fps(20); return result


def test_learned_shadow_is_full_joint_C0_including_prototype_and_native_history():
    rows,event = inputs(); actual=bridge(event); baseline=MOTIdentityBridge(event)
    for frame in range(6):
        a,b=actual.step(frame,rows),baseline.step(frame,rows)
        assert a['outputs']==b['outputs'] and a['state_after']==b['state_after']
        assert not a['joint_identity_memory_write'] and not a['authority']['approved']


def test_runtime_action_axis_matches_real_corpus_action_axis():
    rows,event=inputs(); runtime=bridge(event); runtime.step(0,rows)
    prepared=prepare_proposal(runtime,1,rows)
    assert [a.to_dict() for a in current_candidate_actions(runtime,rows,prepared)] == [a.to_dict() for a in sample_actions(runtime,1,rows,prepared)]


def test_temporal_clone_and_current_past_input_are_isolated():
    rows,event=inputs(); runtime=bridge(event); runtime.step(0,rows); first=runtime.step(1,rows)
    assert not np.array(first['authority']['causal_previous_feature_vectors']).any()
    left,right=runtime.clone(),runtime.clone(); left.causal_proposal_history[-1][1][0]=123.
    assert right.causal_proposal_history[-1][1][0] == runtime.causal_proposal_history[-1][1][0]
    assert len(runtime.causal_proposal_history)==1


def test_approved_alternative_is_exact_global_commit_and_p0_never_writes():
    rows,event=inputs(); runtime=bridge(event,intervene=True); runtime.step(0,rows)
    actual=runtime.step(1,rows)
    assert {r['candidate_uid'] for r in actual['outputs']}=={'p','q'}
    assert len({r['public_id'] for r in actual['outputs']})==2
    assert actual['authority']['approved'] and actual['target_uid']=='q'
    assert not actual['joint_identity_memory_write'] and not runtime.identity.bank
    assert runtime.identity.last_frame==1 and list(runtime.identity.last_box)==rows[1]['box_xyxy']
