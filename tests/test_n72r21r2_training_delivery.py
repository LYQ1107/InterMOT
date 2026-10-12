from copy import deepcopy
import pytest
import torch
from scripts import n72r21r2_training_delivery_v1 as report
from sam3_intermot.one_click.event_authority_models import EventAuthorityHead
from sam3_intermot.one_click.intervention_features import FEATURE_NAMES


def weights():
    model=EventAuthorityHead("SMALL_MLP",hidden=32)
    initial={k:v.detach().clone() for k,v in model.state_dict().items()}
    with torch.no_grad():
        for p in model.parameters():p.add_(.2)
    saved={"schema":"N72R21R2_EVENT_AUTHORITY_V1","feature_names":list(FEATURE_NAMES),
           "family":"SMALL_MLP","seed":1,"objective":"L3_HARM_AWARE","training_state_source":"MIXED_STATE",
           "epoch":2,"protocol_sha256":"protocol","actual_model_generated_on_policy":False,
           "hidden":32,"model":model.state_dict()}
    record={"family":"SMALL_MLP","seed":1,"objective":"L3_HARM_AWARE","mode":"MIXED_STATE",
            "selected_epoch":2,"protocol_sha256":"protocol","parameter_count":sum(p.numel() for p in model.parameters()),
            "changed_weight_elements":sum(int(torch.count_nonzero(initial[k]!=v)) for k,v in saved['model'].items())}
    return saved,initial,record


def test_actual_strict_state_model_count_layout_and_weight_changes_not_fictitious_cost():
    saved,initial,record=weights()
    model,actual=report.state_capacity(saved,initial,record)
    assert actual['parameter_count']==1411 and actual['actual_changed_weight_elements']==1411
    assert not model.training and not any(p.requires_grad for p in model.parameters())
    assert all(v['trainable'] for v in actual['tensor_layout'].values())


@pytest.mark.parametrize('changed',[
    {'schema':'other'},{'feature_names':[]},{'family':'SCALAR'},{'seed':2},{'objective':'L0'},
    {'training_state_source':'BASELINE_STATE'},{'epoch':3},{'protocol_sha256':'changed'},
    {'actual_model_generated_on_policy':True}])
def test_wrong_or_uncalibrated_scope_cannot_supply_state_capacity(changed):
    saved,initial,record=weights();saved.update(changed)
    with pytest.raises((ValueError,RuntimeError)):report.state_capacity(saved,initial,record)


@pytest.mark.parametrize('changed',[{'parameter_count':True},{'parameter_count':7},{'changed_weight_elements':0}])
def test_invented_count_or_unchanged_weights_are_rejected(changed):
    saved,initial,record=weights();record.update(changed)
    with pytest.raises(ValueError):report.state_capacity(saved,initial,record)


def test_same_seed_group_keeps_support_differences_not_best_seed():
    rows=[{'study':'MAIN','family':'SMALL_MLP','objective':'L3','seed':s,'parameter_count':1411,
           'tensor_layout':{'x':[32,40]},'FIT_rows':n} for s,n in zip([1,2,3],[10,20,30])]
    assert len(report.all_seed_groups(rows,[1,2,3])['MAIN/SMALL_MLP__L3'])==3
    for bad in (rows[:2],rows+[rows[0]],[rows[0],rows[0],rows[2]]):
        with pytest.raises(ValueError):report.all_seed_groups(bad,[1,2,3])
    changed=deepcopy(rows);changed[1]['tensor_layout']={}
    with pytest.raises(ValueError):report.all_seed_groups(changed,[1,2,3])
