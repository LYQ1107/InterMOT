"""Model-schema and actual component-supervision regressions, not experiments."""
import numpy as np
import pytest
import torch
from sam3_intermot.association.opportunity_models import ActionValueModel,action_loss
from sam3_intermot.association.opportunity_tracker import FEATURE_NAMES
from sam3_intermot.association.causal_state_commit import memory_metrics
from scripts.n72r20r4r1_fit import causal_features


@pytest.mark.parametrize("family",["C2","C3","C4","C5","C6"])
def test_predictions_are_finite_bounded_and_small(family):
    model=ActionValueModel(family);result=model.predict(np.zeros(len(FEATURE_NAMES)))
    assert 0<=result["beneficial"]<=1 and 0<=result["harmful"]<=1 and -3<=result["value"]<=3
    assert model.parameter_count<50000
    with pytest.raises(ValueError):model.predict(np.zeros(len(FEATURE_NAMES)+1))


@pytest.mark.parametrize("loss",["L0","L1","L2","L3"])
def test_structured_heads_receive_real_component_loss_gradients(loss):
    model=ActionValueModel("C6");torch.manual_seed(1)
    x=torch.randn(5,len(FEATURE_NAMES));y=torch.tensor([[1,0,1],[0,1,-2],[0,0,0],[1,0,.5],[0,1,-3]],dtype=torch.float32)
    components=torch.tensor([[1,0,0,0],[-1,1,1,1],[0,0,0,0],[.5,0,0,0],[-1,2,1,1]],dtype=torch.float32)
    value=action_loss(model,x,y,components,torch.tensor([0,0,0,1,1]),loss,[torch.tensor(2.),torch.tensor(1.)]);value.backward()
    for head in (model.gain_head,model.damage_head,model.risk_head):
        assert any(p.grad is not None and torch.count_nonzero(p.grad)>0 for p in head.parameters())


def test_capacity_matched_mlp_not_arbitrary_name():
    a=ActionValueModel("C4").parameter_count;b=ActionValueModel("C6").parameter_count
    assert abs(a-b)/b<.05


def test_unidentifiable_p0_state_columns_cannot_gain_random_runtime_authority():
    model=ActionValueModel("C6");model.scale.fill_(.05);model.mean[3]=1.
    a=np.zeros(len(FEATURE_NAMES));a[3]=1.;b=a.copy();b[[3,21,22,23]]=[.2,1.,.8,1.]
    assert model.predict(a)==model.predict(b)


def test_zero_writes_cannot_pass_safety():
    x=memory_metrics([{"frame":1,"eligible":True,"correct":True,"accepted":False}])
    assert not x["safety_pass"] and x["wrong_write_rate"] is None and x["correct_write_retention"]==0


def test_corpus_reader_rejects_outer_before_opening_assets():
    with pytest.raises(ValueError,match="overlaps"):
        causal_features("dancetrack0001",None,heldout="dancetrack0001")
    with pytest.raises(ValueError,match="outer labels"):
        causal_features("dancetrack0001",None,heldout="dancetrack0001",role="outer")
