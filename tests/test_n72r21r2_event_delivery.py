import copy
import json
import pytest
from sam3_intermot.evaluation.event_causal_labels import label_actual_branch
from sam3_intermot.evaluation.event_delivery_evidence import component_audit, original_counts, overlap_components


def labels(*, length=101, harmful=False, unavailable=False):
    actual, keep, matching, visible = [], [], {}, {}
    for f in range(length):
        uid, other = "p" + str(f), "q" + str(f)
        matching[f] = {uid: 1, other: None if unavailable else 2}
        visible[f] = True
        base = {"frame": f, "target_public_id": 10,
                "outputs": [{"public_id": 10, "candidate_uid": uid}, {"public_id": 20, "candidate_uid": other}],
                "identity_memory_write": False, "identity_memory_write_UID": None,
                "births": [], "deaths": [], "native_after": 1,
                "prototype_anchor_cosine": 1., "state_after": "base"}
        now = copy.deepcopy(base)
        if harmful:
            now["outputs"] = [{"public_id": 10, "candidate_uid": other}, {"public_id": 20, "candidate_uid": uid}]
            now["state_after"] = "actual"
        actual.append(now)
        keep.append(base)
    value = label_actual_branch(actual, keep, matching, 1, {10: 1, 20: 2}, visible)
    value.update(effective_direct_action_frames=[0] if harmful else [],
                 direct_action_onsets_in_this_one_shot_branch=int(harmful),
                 distinct_correlated_window_not_independent_causal_origin=True,
                 same_executed_action_configuration_as=None,
                 H100_any_harm_including_current_t_label=harmful if length == 101 else None)
    return json.loads(json.dumps(value))


def test_one_direct_action_is_not100_independent_harms():
    result = component_audit(labels(harmful=True), 0, branch="TOP_ALTERNATIVE")
    assert result["direct_decisions_not_independent_roots"] == 1
    assert result["future_H100_N10_overlapping_arm_frames"] == 100
    assert result["N10_future_interval_count_not_roots"] == 1
    assert result["independent_N10_roots"] is None
    assert result["severe_other_harm_H100_arm"]


def test_unknown_is_not_verified_other_takeover():
    value = labels(harmful=True, unavailable=True)
    assert value["future"]["H100"]["UNKNOWN"] == 100
    assert value["future"]["H100"]["verified_OTHER"] == 0
    assert value["verified_OTHER_takeover_intervals"] == []
    component_audit(value, 0, branch="RAW_IDENTITY_TOP")


def test_incomplete_windows_remain_null_not_negative():
    value = labels(length=6)
    result = component_audit(value, 0, branch="KEEP")
    assert not result["complete_H100"]
    assert value["future"]["H5"]["complete"]
    assert value["future"]["H20"]["risk_label"] is None
    assert value["H100_any_harm_including_current_t_label"] is None


@pytest.mark.parametrize("kind", ("offset", "future_count", "current_inclusive", "incomplete_negative", "fake_outcome", "unknown_other", "second_action", "far_future_action", "propagation_roots", "proven_origin", "takeover", "KEEP_harm"))
def test_component_or_unit_corruption_is_rejected(kind):
    value = labels(harmful=True)
    branch = "RAW_IDENTITY_TOP"
    if kind == "offset": value["raw_frame_components"][3]["offset"] = 2
    elif kind == "future_count": value["future"]["H100"]["N10"] = 99
    elif kind == "current_inclusive": value["future"]["H100"]["future_offsets"] = [0, 99]
    elif kind == "incomplete_negative":
        value = labels(length=6)
        value["future"]["H100"]["risk_label"] = False
    elif kind == "fake_outcome": value["raw_frame_components"][2]["outcome"] = "NEGATIVE"
    elif kind == "unknown_other": value["raw_frame_components"][2]["verified_OTHER"] = 0
    elif kind == "second_action": value["effective_direct_action_frames"] = [0, 1]
    elif kind == "far_future_action": value["effective_direct_action_frames"] = [3]
    elif kind == "propagation_roots": value["N10_propagated_intervals_not_action_onsets"] = []
    elif kind == "proven_origin": value["distinct_correlated_window_not_independent_causal_origin"] = False
    elif kind == "takeover": value["max_observed_verified_OTHER_takeover_duration"] = 100
    elif kind == "KEEP_harm": branch = "KEEP"
    with pytest.raises(ValueError):
        component_audit(value, 0, branch=branch)


def test_original_counter_reconstruction_retains_all_arms_and_duplicates():
    safe = component_audit(labels(), 0, branch="KEEP")
    harm = component_audit(labels(harmful=True), 0, branch="RAW_IDENTITY_TOP")
    duplicate = {**harm, "duplicate_executed_action_configuration": True}
    counts = original_counts([safe, harm, duplicate])
    assert counts["actual_arm_records"] == 3
    assert counts["effective_one_shot_arms"] == 2
    assert counts["duplicate_executed_action_configs"] == 1
    assert counts["observed_harmful_H100_arms"] == 2
    assert counts["observed_safe_positive_H100_arms_NOT_independent_corrections"] == 0
    assert original_counts([]) == {key: 0 for key in counts}


def test_overlapping_clicks_same_anonymous_identity_are_correlated_not_roots():
    rows = [{"sequence": "s", "anonymous_identity_scope": "p", "start": 1, "end": 101},
            {"sequence": "s", "anonymous_identity_scope": "p", "start": 50, "end": 150},
            {"sequence": "s", "anonymous_identity_scope": "q", "start": 1, "end": 101},
            {"sequence": "t", "anonymous_identity_scope": "p", "start": 1, "end": 101}]
    result = overlap_components(rows)
    assert len(result) == 3 and result[0]["registered_window_count"] == 2
    assert result[0]["end"] == 150
    assert all(not r["independent_causal_origin_proven"] for r in result)


def test_nonoverlap_or_adjacency_does_not_establish_independence():
    rows = [{"sequence": "s", "anonymous_identity_scope": "p", "start": 1, "end": 101},
            {"sequence": "s", "anonymous_identity_scope": "p", "start": 102, "end": 202}]
    result = overlap_components(rows)
    assert len(result) == 2 and not any(r["independent_causal_origin_proven"] for r in result)
    with pytest.raises(ValueError):
        overlap_components([{**rows[0], "end": 0}])
