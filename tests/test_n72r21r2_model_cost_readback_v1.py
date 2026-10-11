from copy import deepcopy

import pytest
import torch

from scripts import n72r21r2_model_cost_readback_v1 as audit
from sam3_intermot.one_click.event_authority_models import EventAuthorityHead, FAMILIES
from sam3_intermot.one_click.intervention_features import FEATURE_NAMES


def fixture(family="SCALAR"):
    model = EventAuthorityHead(family, hidden=32)
    initial = {k: v.detach().clone() for k, v in model.state_dict().items()}
    with torch.no_grad():
        for parameter in model.parameters(): parameter.add_(.25)
    state = model.state_dict()
    changed = {k: int(torch.count_nonzero(state[k] != initial[k])) for k in state}
    record = dict(family=family, objective="L3_HARM_AWARE", seed=730101, best_epoch=2,
        parameter_count=sum(p.numel() for p in model.parameters()),
        changed_weight_elements_by_tensor=changed, nonzero_gradient_steps=2,
        optimizer_steps=3, CPU_single_event_latency_ms=.1)
    checkpoint = dict(schema="N72R21R2_EVENT_AUTHORITY_V1", family=family,
        objective=record["objective"], seed=record["seed"], pilot=False, epoch=2,
        hidden=32, feature_names=list(FEATURE_NAMES), model=state)
    return checkpoint, initial, record


@pytest.mark.parametrize("family", FAMILIES)
def test_loaded_actual_layout_count_and_weight_changes_are_recomputed(family):
    checkpoint, initial, record = fixture(family)
    result = audit.verified_capacity(checkpoint, initial, record)
    assert result["parameter_count"] == record["parameter_count"]
    assert result["actual_changed_weight_elements"] == sum(record["changed_weight_elements_by_tensor"].values())
    assert result["original_CPU_single_branch_forward_latency_ms"] == .1
    assert set(result["tensor_layout"]) == set(checkpoint["model"])


@pytest.mark.parametrize("change", [{"family": "LOGISTIC_RISK"}, {"seed": 730102},
    {"objective": "other_loss"}, {"pilot": True}, {"epoch": 3}, {"feature_names": []},
    {"schema": "other_schema"}])
def test_wrong_or_pilot_checkpoint_does_not_supply_table4_costs(change):
    checkpoint, initial, record = fixture()
    checkpoint.update(change)
    with pytest.raises(ValueError): audit.verified_capacity(checkpoint, initial, record)


@pytest.mark.parametrize("change", [{"parameter_count": 7}, {"parameter_count": True},
    {"changed_weight_elements_by_tensor": {}}, {"nonzero_gradient_steps": 0},
    {"optimizer_steps": 1}, {"CPU_single_event_latency_ms": None},
    {"CPU_single_event_latency_ms": float("nan")}, {"CPU_single_event_latency_ms": 0.}])
def test_unproven_changed_model_or_invalid_timing_is_not_accepted(change):
    checkpoint, initial, record = fixture()
    record.update(change)
    with pytest.raises(ValueError): audit.verified_capacity(checkpoint, initial, record)


def test_no_weight_change_or_initial_layout_mismatch_is_rejected():
    checkpoint, initial, record = fixture()
    checkpoint["model"] = initial
    record["changed_weight_elements_by_tensor"] = dict.fromkeys(initial, 0)
    with pytest.raises(ValueError): audit.verified_capacity(checkpoint, initial, record)
    checkpoint, initial, record = fixture()
    initial.pop(next(iter(initial)))
    with pytest.raises(ValueError): audit.verified_capacity(checkpoint, initial, record)


def test_all_seed_group_aggregation_is_not_best_seed_and_is_not_timing_ci():
    rows = [dict(family="SCALAR", objective="L3_HARM_AWARE", seed=s, parameter_count=6,
                 tensor_layout={"bias": [3]}, original_CPU_single_branch_forward_latency_ms=t,
                 FIT_rows=10, INNER_rows=8) for s, t in zip([1, 2, 3], [.1, .2, .9])]
    group = audit.complete_groups(rows, [1, 2, 3])["SCALAR__L3_HARM_AWARE"]
    assert group["original_forward_ms_median"] == .2
    assert group["original_forward_ms_min"] == .1 and group["original_forward_ms_max"] == .9
    assert group["timing_NOT_new_controlled_latency_experiment_or_statistical_family_ranking"]
    assert len(group["actual_models"]) == 3
    for invalid in (rows[:-1], rows + [rows[0]], [rows[0], rows[0], rows[2]]):
        with pytest.raises(ValueError): audit.complete_groups(invalid, [1, 2, 3])
    changed = deepcopy(rows); changed[1]["parameter_count"] = 9
    with pytest.raises(ValueError): audit.complete_groups(changed, [1, 2, 3])
    changed = deepcopy(rows); changed[1]["tensor_layout"] = {}
    with pytest.raises(ValueError): audit.complete_groups(changed, [1, 2, 3])
