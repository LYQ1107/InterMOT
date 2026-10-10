import copy
import json
import pytest
from sam3_intermot.evaluation.event_causal_labels import label_actual_branch
from sam3_intermot.evaluation.simple_policy_delivery import POLICIES, onset_counts


def entry(length=101, *, unknown=False, safe=False):
    keep, actual, matching, visible = [], [], {}, {}
    for f in range(length):
        matching[f] = {"p": 1, "q": None if unknown else 2}; visible[f] = True
        base = {"frame": f, "target_public_id": 10, "outputs": [
            {"public_id": 10, "candidate_uid": "q" if safe else "p"},
            {"public_id": 20, "candidate_uid": "p" if safe else "q"}],
            "identity_memory_write": False, "identity_memory_write_UID": None,
            "births": [], "deaths": [], "native_after": 1,
            "prototype_anchor_cosine": 1., "state_after": "base"}
        treatment = copy.deepcopy(base)
        treatment["outputs"] = [{"public_id": 10, "candidate_uid": "p" if safe else "q"},
                                {"public_id": 20, "candidate_uid": "q" if safe else "p"}]
        treatment["state_after"] = "actual"
        keep.append(base); actual.append(treatment)
    labels = label_actual_branch(actual, keep, matching, 1, {} if safe else {10: 1, 20: 2}, visible)
    labels = json.loads(json.dumps(labels))
    return {"episode_uid": "click0", "effective_decisions": 1, "nonvacuity": True,
            "one_shot_diagnostics": [{"frame": 0, "labels": labels}]}


def test_named_delayed_confirmation_is_only_existing_P6_not_P8_or_P9():
    assert POLICIES["DELAYED_CONFIRMATION"] == "P6_PERSISTENT_CAUSAL_CHALLENGER"
    assert len(POLICIES) == len(set(POLICIES.values())) == 10


def test_one_harmful_action_with100_propagated_errors_is_one_diagnostic_not100_roots():
    value = entry(); frozen = copy.deepcopy(value)
    result = onset_counts([value])
    assert result["effective_decisions"] == result["complete_H100_decisions"] == 1
    assert result["risky_decisions"] == result["severe_non_target_harm_decisions"] == 1
    assert result["beneficial_complete_H100_decisions"] == 0
    assert value == frozen and not any("independent" in k for k in result)


def test_unknown_harm_is_not_verified_other_but_remains_risky():
    value = entry(unknown=True)
    assert value["one_shot_diagnostics"][0]["labels"]["future"]["H100"]["verified_OTHER"] == 0
    assert onset_counts([value])["risky_decisions"] == 1


def test_incomplete_future_is_never_complete_or_safe():
    result = onset_counts([entry(length=6, safe=True)])
    assert result["incomplete_H100_decisions"] == 1
    assert result["complete_H100_decisions"] == result["beneficial_complete_H100_decisions"] == 0


def test_safe_component_diagnostic_is_not_an_independent_root_or_MOT_PASS():
    result = onset_counts([entry(safe=True)])
    assert result["beneficial_complete_H100_decisions"] == 1
    assert "G1" not in result and "PASS" not in result


def test_no_actions_is_a_zero_denominator_not_safety_success():
    result = onset_counts([{"episode_uid": "click0", "effective_decisions": 0,
                            "nonvacuity": False, "one_shot_diagnostics": []}])
    assert set(result.values()) == {0}


@pytest.mark.parametrize("corruption", ("duplicate_click", "missing_decision", "duplicate_frame", "fake_nonvacuity", "future_count", "unknown_as_other"))
def test_incomplete_or_corrupted_units_are_rejected(corruption):
    value = entry(unknown=corruption == "unknown_as_other"); entries = [value]
    if corruption == "duplicate_click": entries.append(copy.deepcopy(value))
    elif corruption == "missing_decision": value["effective_decisions"] = 2
    elif corruption == "duplicate_frame":
        value["one_shot_diagnostics"].append(copy.deepcopy(value["one_shot_diagnostics"][0])); value["effective_decisions"] = 2
    elif corruption == "fake_nonvacuity": value["nonvacuity"] = False
    elif corruption == "future_count": value["one_shot_diagnostics"][0]["labels"]["future"]["H100"]["N10"] = 99
    else: value["one_shot_diagnostics"][0]["labels"]["raw_frame_components"][1]["verified_OTHER"] = 1
    with pytest.raises(ValueError): onset_counts(entries)
