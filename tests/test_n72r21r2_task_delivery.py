import pytest
from sam3_intermot.evaluation.task_delivery_evidence import REQUIRED, TABLE_REQUIREMENTS, TEST_REQUIREMENTS, coverage_row, unopened_row, named_requirement


def sources():
    # The actual frozen initializer has sequence in EACH click, no top-level sequence.
    init = {"candidate_index_sha256": "index", "inputs": [
        {"episode_uid": "s0", "sequence": "s", "role": "FIT", "initialization_failure": False, "sole_click": True, "runtime_future_GT_used": False},
        {"episode_uid": "s1", "sequence": "s", "role": "FIT", "initialization_failure": True, "sole_click": True, "runtime_future_GT_used": False}]}
    candidate = {"sequence": "s", "index_sha256": "index", "frames": 10, "candidate_count": 50, "whole_video_density": "MIXED"}
    baseline = {"sequence": "s", "candidate_index_sha256": "index", "coverage": {"s0": {
        "visible_frames": 9, "strict_positive_available_frames": 7, "initialization_failure": False}}}
    scene = {"sequence": "s", "original_complete_video": {"GT_video_local_identities_NOT_cross_scene_people": 5}}
    return init, candidate, baseline, scene


def test_all_requested_names_tables_and_test_semantics_are_retained():
    assert len(REQUIRED) == len(set(REQUIRED)) == 74
    assert len(TABLE_REQUIREMENTS) == 6
    assert len(TEST_REQUIREMENTS) == len(set(TEST_REQUIREMENTS)) == 35
    for name in ("training/RELATIONAL.json", "on_policy/CORRECTION_TRAINING.json", "confirmation/RESULTS.json", "FINAL_REPORT.md", "FINAL_RESULT.json"):
        assert name in REQUIRED


def test_actual_coverage_retains_failed_clicks_and_unproven_units():
    row = coverage_row("s", "fit", *sources())
    assert row["sole_click_attempts"] == 2 and row["valid_sole_clicks"] == row["failed_initializations_retained"] == 1
    assert row["coverage_per_registered_click"][1]["actual_same_input_postclick_coverage"] is None
    assert row["video_local_GT_identity_count_NOT_cross_scene_people"] == 5
    assert row["independent_people_count"] is None
    assert row["independent_beneficial_correction_roots"] is None and row["independent_N10_onset_events"] is None
    assert row["scientific_success"] is None


@pytest.mark.parametrize("kind", ("role", "sequence", "duplicate", "candidate", "missing_coverage", "denominator", "event_sequence", "second_click", "future_GT"))
def test_inconsistent_input_or_successful_subset_is_rejected(kind):
    init, candidate, baseline, scene = sources()
    if kind == "role": init["inputs"][0]["role"] = "INNER"
    elif kind == "sequence": scene["sequence"] = "different"
    elif kind == "duplicate": init["inputs"][1]["episode_uid"] = "s0"
    elif kind == "candidate": baseline["candidate_index_sha256"] = "new-input"
    elif kind == "missing_coverage": baseline["coverage"] = {}
    elif kind == "denominator": baseline["coverage"]["s0"]["strict_positive_available_frames"] = 10
    elif kind == "event_sequence": init["inputs"][0]["sequence"] = "another"
    elif kind == "second_click": init["inputs"][0]["sole_click"] = False
    elif kind == "future_GT": init["inputs"][0]["runtime_future_GT_used"] = True
    with pytest.raises(ValueError):
        coverage_row("s", "fit", init, candidate, baseline, scene)


def test_all_failed_initialization_is_not_successful_zero_action_evaluation():
    init, candidate, baseline, scene = sources()
    init["inputs"][0]["initialization_failure"] = True
    baseline["coverage"] = {}
    row = coverage_row("s", "fit", init, candidate, baseline, scene)
    assert row["valid_sole_clicks"] == 0 and row["failed_initializations_retained"] == 2
    assert all(c["actual_same_input_postclick_coverage"] is None for c in row["coverage_per_registered_click"])


def test_confirmation_and_historical_rows_never_invent_fresh_results():
    confirm = unopened_row("s", "confirmation")
    historical = unopened_row("s", "historical_development")
    assert confirm["status"].startswith("NOT_RUN_UNOPENED")
    assert historical["status"].startswith("HISTORICAL_ALREADY_EXPOSED")
    assert confirm["fresh_candidate_coverage"] is None and not confirm["fresh_candidate_generation_or_truth_access"]
    with pytest.raises(ValueError): unopened_row("s", "fit")


def test_presence_narrow_proof_and_full_goal_completion_are_distinct():
    present = named_requirement("FINAL_RESULT.json", True, "hash")
    assert present["evidence_status"] == "PRESENT_SEMANTICS_NOT_FINAL_VERIFIED"
    assert not present["counts_as_full_goal_completion"]
    absent = named_requirement("mot/INNER_RESULTS.json", False)
    assert absent["evidence_status"] == "REQUIRED_REPORT_MISSING"
    frozen = named_requirement("data/FROZEN_SPLITS.json", True, "hash", "original partitions verified")
    assert frozen["evidence_status"] == "VERIFIED_NARROW_PREPARATION" and not frozen["counts_as_full_goal_completion"]
    with pytest.raises(ValueError): named_requirement("FINAL_RESULT.json", False, narrow_proof="pretend")
    with pytest.raises(ValueError): named_requirement("unrequested-placeholder.json", True)
