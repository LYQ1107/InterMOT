import numpy as np
import pytest
import torch
from tests.test_n72r21_mot_bridge import inputs
from scripts.n72r21_t2_replay import DecisionCapture
from sam3_intermot.one_click.acib_memory import ACIBMemoryNetwork
from sam3_intermot.one_click.acib_trusted_runtime import TrustedACIBRecognizer
from sam3_intermot.one_click.intervention_gate import GatePolicy
from sam3_intermot.one_click.trajectory_gate_bridge import TrajectoryGateMOTBridge
from sam3_intermot.one_click.mot_bridge import MOTIdentityBridge


class Spy:
    def __init__(self):self.calls=[]
    def predict(self,current,history):
        self.calls.append((current.copy(),history.copy()))
        return {'beneficial':0.,'harmful':1.,'abstain':0.,'value':-1.}


def bridge(event,family):
    torch.manual_seed(72111);actor=TrustedACIBRecognizer(DecisionCapture(ACIBMemoryNetwork()).eval(),event['human_anchor'],'unit',policy='P0')
    actor.start_recording('unit',fps=20,width=100,height=100,initial_frame=0,initial_box=event['target_box_xyxy'])
    p=GatePolicy(family=family,probability_min=0.,identity_margin_min=-1.,quality_min=0.,anchor_cosine_min=-1.,base_margin_max=100.,delay_frames=1)
    b=TrajectoryGateMOTBridge(event,actor,policy=p,fitted_predictor=Spy());b.configure_fps(20);return b


@pytest.mark.parametrize('family',['two_branch','risk'])
def test_B7_B8_keep_preserves_full_C0_and_records_real_causal_past(family):
    rows,event=inputs();b=bridge(event,family);baseline=MOTIdentityBridge(event)
    for f in range(5):
        a,c=b.step(f,rows),baseline.step(f,rows)
        assert a['outputs']==c['outputs'] and a['state_after']==c['state_after']
        assert not a['joint_identity_memory_write'] and not a['authority']['approved']
    assert len(b.causal_gate_history)==3
    current=np.ones(32,np.float32);b.predictor.predict(current)
    _,past=b.fitted_predictor.calls[-1]
    assert np.array_equal(past,np.array([x for _,x in b.causal_gate_history],np.float32))


def test_clone_predictor_reads_clone_not_parent_or_other_branch_history():
    rows,event=inputs();b=bridge(event,'two_branch');b.step(0,rows);b.step(1,rows)
    left,right=b.clone(),b.clone();left.causal_gate_history[0][1][0]=123.
    left.predictor.predict(np.zeros(32,np.float32));lh=left.fitted_predictor.calls[-1][1].copy()
    right.predictor.predict(np.zeros(32,np.float32));rh=right.fitted_predictor.calls[-1][1].copy()
    assert lh[-1,0]==123. and rh[-1,0]!=123.
    assert left.predictor.owner is left and right.predictor.owner is right
    assert b.causal_gate_history[0][1][0]!=123.
