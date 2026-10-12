from copy import deepcopy
import pytest
from scripts import n72r21r2_state_inner_readback_v1 as audit


def records():
    fit = {"mode": "TREATMENT_STATE", "family": "SMALL_MLP", "seed": 1,
        "objective": "L3_HARM_AWARE", "actual_model_generated_on_policy": False,
        "association_authority": False, "nonzero_gradient_steps": 2,
        "changed_weight_elements": 3,
        "status": "COMPLETE_ACTUAL_STATE_SOURCE_CONTRAST_UNCALIBRATED_NOT_DEPLOYABLE"}
    replay = {"family": "TREATMENT_STATE", "actual_architecture_family": "SMALL_MLP",
        "seed": 1, "objective": "L3_HARM_AWARE",
        "controlled_training_state_source_contrast": True,
        "actual_model_generated_on_policy_training": False}
    return fit, replay, deepcopy(replay)


def test_actual_source_mode_is_not_the_network_architecture():
    audit.validate_state_binding(*records(), "TREATMENT_STATE", 1, "L3_HARM_AWARE")


@pytest.mark.parametrize("record,key,value", [
    (0, "mode", "MIXED_STATE"), (0, "family", "TREATMENT_STATE"),
    (0, "status", "FAIL_MEASURED_STATE_SOURCE_SUPERVISION_DIVERSITY"),
    (0, "actual_model_generated_on_policy", True), (0, "association_authority", True),
    (0, "nonzero_gradient_steps", 0), (0, "changed_weight_elements", 0),
    (1, "family", "SMALL_MLP"), (1, "actual_architecture_family", "SCALAR"),
    (1, "controlled_training_state_source_contrast", False),
    (2, "actual_model_generated_on_policy_training", True),
    (2, "seed", 2), (2, "objective", "L0_CURRENT_CORRECTNESS")])
def test_mismatched_scope_authority_or_no_fit_cannot_supply_evidence(record, key, value):
    rows = list(records()); rows[record][key] = value
    with pytest.raises(ValueError):
        audit.validate_state_binding(*rows, "TREATMENT_STATE", 1, "L3_HARM_AWARE")


def test_inner_selection_requires_all_seeds_both_points_and_all_videos(monkeypatch):
    protocol = {"points": ["P0", "P1"], "seeds": [1, 2, 3], "split": {"inner": ["a", "b"]}}
    cells = {point: [{"seed": seed, "sequence": seq} for seed in protocol["seeds"]
                    for seq in protocol["split"]["inner"]] for point in protocol["points"]}
    result = {"selected_point": "P0", "status": "unqualified", "next_stage_authorized": False}
    calls = []
    def select(rows, seeds, sequences):
        calls.append((seeds, sequences)); return result
    monkeypatch.setattr(audit, "choose_inner_point", select)
    assert audit.verify_selection(cells, result, protocol) == result
    assert len(calls) == 1
    for bad in (cells["P0"][:-1], cells["P0"] + [cells["P0"][0]], [cells["P0"][0]] * 6):
        with pytest.raises(ValueError):
            audit.verify_selection({**cells, "P0": bad}, result, protocol)
    with pytest.raises(ValueError):
        audit.verify_selection(cells, {**result, "next_stage_authorized": True}, protocol)
