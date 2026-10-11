from copy import deepcopy
import pytest
from sam3_intermot.evaluation.complete_mot_aa_evidence import (
    checked_frame, sequence_scope, summarize_scopes,
)

P, I, C, BEFORE, AFTER = (character * 64 for character in "abcde")


def original_sources(failed=False):
    event = {"sequence": "s", "role": "FIT", "episode_uid": "s__click0", "sole_click": True,
        "runtime_future_GT_used": False, "runtime_GT_identity_fields": False,
        "initialization_failure": failed, "candidate_index_sha256": C, "frames": 2}
    init = {"candidate_index_sha256": C, "inputs": [event]}
    receipt = {"sequence": "s", "protocol_sha256": P, "initialization_sha256": I,
        "candidate_index_sha256": C, "full_original_video": True, "actual_GT_file_guard": True,
        "all_complete_tracker_attributes_AA": True, "runtime_actor_controller_equality_NOT_claimed": True,
        "old_scientific_source_or_results_modified": False, "next_stage_authorized": False,
        "scientific_success": None, "failed_initializations_retained": int(failed),
        "episodes": [] if failed else [{"episode_uid": event["episode_uid"], "frames": 2}]}
    return init, receipt


def scope(init, receipt):
    return sequence_scope("s", init, receipt, protocol_sha256=P,
                          initialization_sha256=I, candidate_index_sha256=C)


def frame_sources():
    row = {"frame": 0, "all_tracker_attribute_names": ["manager", "states"],
        "complete_MOT_before_AA_sha256": BEFORE, "complete_MOT_after_AA_sha256": AFTER,
        "legacy_subset_after_sha256": C, "unchanged_current_candidate_inputs_sha256": I,
        "actual_GT_file_guard": True, "complete_output_ownership_unique": True,
        "extra_clicks": 0, "candidate_count": 2, "target_uid": "a"}
    c0 = {"frame": 0, "extra_clicks": 0, "runtime_gt_read": False, "runtime_future_gt_used": False,
        "joint_identity_memory_write": False, "current_candidate_index_sha256": C,
        "outputs": [{"candidate_uid": "a", "public_id": 1}, {"candidate_uid": "b", "public_id": 2}],
        "target_public_id": 1, "target_uid": "a", "state_before": BEFORE, "state_after": AFTER}
    shadow = deepcopy(c0)
    shadow["authority"] = {"approved": False}
    return {"row": row, "frame": 0, "candidates": [{"candidate_uid": "a"}, {"candidate_uid": "b"}],
            "input_sha256": I, "attributes": ["manager", "states"], "previous_after": None,
            "c0": c0, "shadow": shadow}


def test_all_failed_unconditional_raw_booleans_never_prove_AA():
    init, receipt = original_sources(failed=True)
    assert receipt["all_complete_tracker_attributes_AA"] and receipt["full_original_video"]
    result = scope(init, receipt)
    assert result["status"] == "NOT_RUN_NO_VALID_CLICK_INITIALIZATIONS_RETAINED"
    assert result["complete_MOT_AA_proof"] is None
    assert result["paired_frame_exposures_NOT_independent_sample_size"] == 0
    summary = summarize_scopes(["s"], [result])
    assert summary["all_registered_sequence_dispositions_reverified"]
    assert summary["nonempty_full_video_AA_sequences"] == []
    assert not summary["full_G0_or_scientific_PASS"]


def test_nonempty_scope_requires_separate_raw_verification():
    result = scope(*original_sources())
    assert result["valid_original_clicks"] == 1 and result["complete_MOT_AA_proof"] is None
    with pytest.raises(ValueError, match="actually complete raw checks"):
        summarize_scopes(["s"], [result])
    result.update(status="VERIFIED_NONEMPTY_FULL_VIDEO_AA_RAW_EVIDENCE", complete_MOT_AA_proof=True)
    summary = summarize_scopes(["s", "unopened"], [result])
    assert not summary["all_registered_sequence_dispositions_reverified"]
    assert summary["missing_sequences"] == ["unopened"]
    assert summary["actual_paired_frame_exposures_NOT_independent_sample_size"] == 2
    assert summary["independent_correction_roots"] is None
    assert not summary["scientific_goal_complete"] and not summary["next_stage_authorized"]


@pytest.mark.parametrize("kind", ["partial", "duplicate", "missing", "replacement", "failed_count",
    "future_GT", "second_click", "candidate", "protocol", "smoke", "role", "identity_GT"])
def test_original_click_population_and_full_video_scope_fail_closed(kind):
    init, receipt = original_sources()
    if kind == "partial": receipt["episodes"][0]["frames"] = 1
    elif kind == "duplicate": receipt["episodes"] *= 2
    elif kind == "missing": receipt["episodes"] = []
    elif kind == "replacement": receipt["episodes"][0]["episode_uid"] = "replacement"
    elif kind == "failed_count": receipt["failed_initializations_retained"] = 1
    elif kind == "future_GT": init["inputs"][0]["runtime_future_GT_used"] = True
    elif kind == "second_click": init["inputs"][0]["sole_click"] = False
    elif kind == "candidate": init["inputs"][0]["candidate_index_sha256"] = BEFORE
    elif kind == "protocol": receipt["protocol_sha256"] = BEFORE
    elif kind == "smoke": receipt["full_original_video"] = False
    elif kind == "role": init["inputs"][0]["role"] = "CONFIRMATION"
    elif kind == "identity_GT": init["inputs"][0]["runtime_GT_identity_fields"] = True
    with pytest.raises(ValueError): scope(init, receipt)


def test_complete_recorded_boundaries_and_current_global_axis():
    arguments = frame_sources()
    assert checked_frame(**arguments) == AFTER
    arguments["previous_after"] = BEFORE
    assert checked_frame(**arguments) == AFTER
    arguments["previous_after"] = AFTER
    with pytest.raises(ValueError, match="discontinuous"): checked_frame(**arguments)


@pytest.mark.parametrize("kind", ["attribute", "input", "public", "candidate", "drop", "target",
    "frame", "hash", "guard", "write", "GT", "future_GT", "authority", "extra_click", "bool_count"])
def test_changed_state_or_ownership_provenance_is_not_AA_evidence(kind):
    arguments = frame_sources()
    row, shadow = arguments["row"], arguments["shadow"]
    if kind == "attribute": row["all_tracker_attribute_names"] = ["states"]
    elif kind == "input": row["unchanged_current_candidate_inputs_sha256"] = BEFORE
    elif kind == "public": shadow["outputs"][1]["public_id"] = 1
    elif kind == "candidate": arguments["candidates"][1]["candidate_uid"] = "a"
    elif kind == "drop": shadow["outputs"].pop()
    elif kind == "target": row["target_uid"] = "b"
    elif kind == "frame": shadow["frame"] = 1
    elif kind == "hash": row["complete_MOT_after_AA_sha256"] = "not-a-hash"
    elif kind == "guard": row["actual_GT_file_guard"] = False
    elif kind == "write": shadow["joint_identity_memory_write"] = True
    elif kind == "GT": shadow["runtime_gt_read"] = True
    elif kind == "future_GT": shadow["runtime_future_gt_used"] = True
    elif kind == "authority": shadow["authority"]["approved"] = True
    elif kind == "extra_click": row["extra_clicks"] = 1
    elif kind == "bool_count": row["extra_clicks"] = False
    with pytest.raises(ValueError): checked_frame(**arguments)


def test_video_count_and_all_failed_dispositions_cannot_be_forged():
    failed = scope(*original_sources(failed=True))
    with pytest.raises(ValueError): summarize_scopes(["s", "s"], [failed])
    with pytest.raises(ValueError): summarize_scopes(["s"], [failed, failed])
    with pytest.raises(ValueError): summarize_scopes(["other"], [failed])
    failed["complete_MOT_AA_proof"] = True
    with pytest.raises(ValueError, match="Empty exposure"): summarize_scopes(["s"], [failed])
