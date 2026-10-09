import numpy as np
import pytest
import torch
from sam3_intermot.one_click.intervention_features import FEATURE_NAMES
from sam3_intermot.one_click.on_policy_authority import OnPolicyAuthorityHead,OnPolicyAuthorityPredictor,decoded_predictions
from scripts.n72r21r1_train_on_policy import objective


@pytest.mark.parametrize('family',['LOGISTIC','GLOBAL_RISK','CAUSAL_TEMPORAL'])
def test_actual_correction_gradient_and_strict_roundtrip(tmp_path,family):
    torch.manual_seed(11);torch.set_num_threads(1);model=OnPolicyAuthorityHead(family)
    x=torch.randn(3,4,32);initial={k:v.clone() for k,v in model.state_dict().items()}
    optimizer=torch.optim.AdamW(model.parameters(),lr=.01)
    loss=objective(model(x),torch.tensor([0,1,2]),torch.tensor([0.,1.,-1.]),torch.tensor([[0.,0.,0.],[1.,0.,0.],[-1.,1.,1.]]),torch.ones(3),family).mean()
    loss.backward();assert sum(float(p.grad.abs().sum()) for p in model.parameters() if p.grad is not None)>0
    optimizer.step();assert any(not torch.equal(v,initial[k]) for k,v in model.state_dict().items())
    path=tmp_path/'head.pt';model.eval()
    torch.save({'schema':'N72R21R1_ON_POLICY_AUTHORITY_V1','family':family,'model':model.state_dict(),
        'feature_names':list(FEATURE_NAMES),'FIT_mean':[0.]*32,'FIT_std':[1.]*32,'selection':{'status':'CALIBRATION_ABSTAIN'}},path)
    loaded=OnPolicyAuthorityPredictor(path)
    with torch.inference_mode():expected=decoded_predictions(model(x[:1]))[0]
    assert loaded.predict(x[0,-1].numpy(),x[0,:3].numpy())==expected


def test_competitor_risk_not_hidden_by_target_gain():
    output=torch.tensor([[0.,9.,-9.,2.,2.,10.,-10.]])
    p=decoded_predictions(output)[0]
    assert p['beneficial']>.99 and p['target_only_value']==2.
    assert p['class_harm']<.001 and p['other_person_harm']>.99 and p['harmful']>.99


@pytest.mark.parametrize('bad_input',[np.full(32,np.nan),np.zeros(33)])
def test_runtime_rejects_noncausal_schema(tmp_path,bad_input):
    model=OnPolicyAuthorityHead('LOGISTIC');path=tmp_path/'head.pt'
    torch.save({'schema':'N72R21R1_ON_POLICY_AUTHORITY_V1','family':'LOGISTIC','model':model.state_dict(),
        'feature_names':list(FEATURE_NAMES),'FIT_mean':[0.]*32,'FIT_std':[1.]*32,'selection':{}},path)
    with pytest.raises(ValueError):OnPolicyAuthorityPredictor(path).predict(bad_input)


def test_temporal_history_affects_only_temporal_head():
    torch.manual_seed(17);x=torch.randn(2,4,32);x[1,-1]=x[0,-1]
    logistic=OnPolicyAuthorityHead('LOGISTIC');temporal=OnPolicyAuthorityHead('CAUSAL_TEMPORAL')
    assert torch.equal(logistic(x)[0],logistic(x)[1])
    assert not torch.equal(temporal(x)[0],temporal(x)[1])
