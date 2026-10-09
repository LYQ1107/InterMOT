import numpy as np
import pytest
import torch
from sam3_intermot.one_click.intervention_features import FEATURE_NAMES
from sam3_intermot.one_click.joint_write_authority import JointWriteHead,JointWritePredictor
from scripts.n72r21r1_train_joint_write import loss,curve


def test_missing_future_labels_have_zero_auxiliary_gradient():
    output=torch.zeros(2,3,requires_grad=True)
    value=loss(output,torch.tensor([1.,0.]),torch.zeros(2,2),torch.zeros(2,dtype=torch.bool),torch.ones(2),torch.tensor(1.))
    value.backward();assert output.grad[:,0].abs().sum()>0
    assert torch.equal(output.grad[:,1:],torch.zeros(2,2))


def test_real_complete_pair_trains_both_future_risk_components():
    output=torch.zeros(2,3,requires_grad=True)
    value=loss(output,torch.tensor([1.,0.]),torch.tensor([[1.,0.],[0.,1.]]),torch.tensor([True,False]),torch.ones(2),torch.tensor(1.))
    value.backward();assert output.grad[0,1:].abs().sum()>0
    assert torch.equal(output.grad[1,1:],torch.zeros(2))


@pytest.mark.parametrize('status',['CALIBRATION_ABSTAIN','CALIBRATED_DIAGNOSTIC'])
def test_strict_writer_loader_and_explicit_calibration_abstention(tmp_path,status):
    torch.manual_seed(9);model=JointWriteHead().eval();path=tmp_path/'writer.pt'
    saved={'schema':'N72R21R1_JOINT_COMMIT_WRITE_V1','model':model.state_dict(),'feature_names':list(FEATURE_NAMES),
        'FIT_mean':[0.]*32,'FIT_std':[1.]*32,'selection':{'status':status}}
    torch.save(saved,path);predictor=JointWritePredictor(path);current=np.zeros(32,np.float32);result=predictor.predict(current)
    with torch.inference_mode():expected=model(torch.from_numpy(current)[None])[0].sigmoid().tolist()
    assert result['current_correct_probability']==expected[0]
    if status=='CALIBRATION_ABSTAIN':assert result['beneficial']==0. and result['harmful']==1.
    else:assert result['beneficial']==expected[0] and result['harmful']==max(1-expected[0],expected[1],expected[2])
    with pytest.raises(ValueError):predictor.predict(np.full(32,np.nan))


def test_zero_writes_are_undefined_risk_and_never_usefulness():
    rows=[{'correct':True,'pair_complete':False,'future_risk':None}]
    point=curve(rows,np.array([[.5,.5,.5]]),.98,2)
    assert point['accepted']==0 and point['wrong_or_UNKNOWN_rate'] is None
    assert point['correct_committed_retention']==point['candidate_available_retention']==0.
