import pytest
from sam3_intermot.evaluation.causal_identity_events import intervals, direct_vs_propagated, candidate_identity_outcome, stratified_onsets


def test_hundred_propagated_frames_are_one_interval_not_hundred_onsets():
    rows = [{"frame": f, "category": "WRONG", "C0_correct": True, "actual_correct": False,
             "effective_action": f == 0, "paired_causal_branch_sealed": True, "independent_causal_boundary": f == 0} for f in range(100)]
    assert len(intervals(rows)) == 1
    value = direct_vs_propagated(rows)
    assert value["N10_frames"] == 100 and value["independent_action_onset_count"] == 1


def test_wrong_KEEP_propagation_without_branch_never_proves_new_onset():
    value = direct_vs_propagated([{"frame": 5, "C0_correct": True, "actual_correct": False, "effective_action": False}])
    assert value["N10_frames"] == 1 and value["independent_action_onset_count"] == 0


def test_unmatched_candidate_is_UNKNOWN_not_verified_other():
    assert candidate_identity_outcome("a", {"a": None}, 7) == "UNKNOWN"
    assert candidate_identity_outcome("a", {"a": 8}, 7) == "VERIFIED_OTHER"
    with pytest.raises(ValueError):
        candidate_identity_outcome("future_UID", {"a": 7}, 7)


def test_probe_sampling_retains_all_classes_not_only_gain():
    rows = [{"frame": f, "categories": ["POSITIVE" if f % 3 == 0 else "HARM" if f % 3 == 1 else "NO_POSITIVE"]} for f in range(60)]
    selected = stratified_onsets(rows)
    assert {0, 1, 2} <= set(selected)
    assert len(selected) <= 27
