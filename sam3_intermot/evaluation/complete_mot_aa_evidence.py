"""Scope-checked interpretation of sealed complete-MOT A/A evidence.

Recorded boundary hashes are not an independent reconstruction of internal
states. No-valid-click receipts never prove A/A, despite legacy raw booleans.
"""
import re


def require(condition, message):
    if not condition:
        raise ValueError(message)


def count(value):
    require(type(value) is int and value >= 0, "Exact nonnegative integer required")
    return value


def digest(value):
    require(isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None,
            "Exact lowercase SHA256 required")
    return value


def sequence_scope(sequence, initialization, receipt, *, protocol_sha256,
                   initialization_sha256, candidate_index_sha256):
    require(receipt["sequence"] == sequence, "Wrong sequence receipt")
    require(receipt["protocol_sha256"] == digest(protocol_sha256), "Wrong frozen protocol")
    require(receipt["initialization_sha256"] == digest(initialization_sha256), "Wrong original clicks")
    require(initialization["candidate_index_sha256"] == receipt["candidate_index_sha256"]
            == digest(candidate_index_sha256), "Different candidate lineage")
    for key in ("full_original_video", "actual_GT_file_guard", "all_complete_tracker_attributes_AA",
                "runtime_actor_controller_equality_NOT_claimed"):
        require(receipt[key] is True, "Only original full-video guarded MOT-state scope")
    require(receipt["old_scientific_source_or_results_modified"] is False
            and receipt["next_stage_authorized"] is False and receipt["scientific_success"] is None,
            "A/A is not scientific success or stage authorization")
    events = initialization["inputs"]
    require(bool(events), "Registered click attempts must not disappear")
    uids = [event["episode_uid"] for event in events]
    require(len(set(uids)) == len(uids), "Repeated original click")
    for event in events:
        require(event["sequence"] == sequence and event["role"] in ("FIT", "INNER"),
                "Only registered development clicks")
        require(event["sole_click"] is True and event["runtime_future_GT_used"] is False
                and event["runtime_GT_identity_fields"] is False, "Sole click without future GT")
        require(type(event["initialization_failure"]) is bool, "Explicit failure disposition required")
        require(event["candidate_index_sha256"] == candidate_index_sha256, "Click candidate mismatch")
        require(count(event["frames"]) > 0, "Complete original frame exposure required")
    valid = [event for event in events if not event["initialization_failure"]]
    episodes = receipt["episodes"]
    require([episode["episode_uid"] for episode in episodes] == [event["episode_uid"] for event in valid],
            "All original valid clicks, no duplicate, replacement or successful subset")
    failures = len(events) - len(valid)
    require(count(receipt["failed_initializations_retained"]) == failures, "Failed slots lost")
    for event, episode in zip(valid, episodes, strict=True):
        require(count(episode["frames"]) == event["frames"], "Partial video cannot prove whole-video A/A")
    return {"sequence": sequence,
            "status": "VERIFIED_NONEMPTY_FULL_VIDEO_AA_SCOPE_PENDING_RAW_CHECKS" if valid else
                      "NOT_RUN_NO_VALID_CLICK_INITIALIZATIONS_RETAINED",
            "registered_click_attempts": len(events), "valid_original_clicks": len(valid),
            "failed_initializations_retained": failures,
            "paired_frame_exposures_NOT_independent_sample_size": sum(event["frames"] for event in valid),
            "complete_MOT_AA_proof": None,
            "raw_unconditional_AA_boolean_NEVER_nonempty_exposure_proof": True}


def checked_frame(row, *, frame, candidates, input_sha256, attributes, previous_after, c0, shadow):
    require(count(row["frame"]) == count(frame), "Different, repeated or missing frame")
    require(attributes == sorted(set(attributes)) and {"states", "manager"} <= set(attributes),
            "Explicit nonempty unique complete tracker attribute axis required")
    require(row["all_tracker_attribute_names"] == attributes, "Attribute axis changed")
    before = digest(row["complete_MOT_before_AA_sha256"])
    after = digest(row["complete_MOT_after_AA_sha256"])
    digest(row["legacy_subset_after_sha256"])
    if previous_after is not None:
        require(before == digest(previous_after), "Recorded state boundaries are discontinuous")
    require(row["unchanged_current_candidate_inputs_sha256"] == digest(input_sha256),
            "Actual current candidate bytes changed")
    require(row["actual_GT_file_guard"] is True and row["complete_output_ownership_unique"] is True
            and count(row["extra_clicks"]) == 0, "Guard/ownership/sole-click proof missing")
    uids = [candidate["candidate_uid"] for candidate in candidates]
    require(len(uids) == len(set(uids)) and all(isinstance(uid, str) and uid for uid in uids),
            "Original current candidate UID axis is not unique")
    require(count(row["candidate_count"]) == len(uids), "Full current candidate exposure lost")
    for current in (c0, shadow):
        require(count(current["frame"]) == frame and count(current["extra_clicks"]) == 0,
                "Original trace frame/click mismatch")
        require(current["runtime_gt_read"] is False and current["runtime_future_gt_used"] is False
                and current["joint_identity_memory_write"] is False, "GT or memory authority in A/A")
        require(current["current_candidate_index_sha256"] == c0["current_candidate_index_sha256"],
                "Original trace input lineage differs")
        outputs = current["outputs"]
        output_uids = [output["candidate_uid"] for output in outputs]
        publics = [output["public_id"] for output in outputs]
        require(len(outputs) == len(uids) == len(set(output_uids)) == len(set(publics))
                and set(output_uids) == set(uids) and all(type(public) is int and public > 0 for public in publics),
                "All current candidates must have unique full-global public ownership")
        owners = {output["public_id"]: output["candidate_uid"] for output in outputs}
        require(owners.get(current["target_public_id"]) == current["target_uid"],
                "Target UID does not match full-global public owner")
    require(all(c0[key] == shadow[key] for key in ("outputs", "target_uid", "state_before", "state_after")),
            "Original C0/SHADOW traces differ")
    require(row["target_uid"] == c0["target_uid"], "Recorded target ownership differs")
    require(shadow["authority"]["approved"] is False, "SHADOW must not intervene")
    return after


def summarize_scopes(registered, scopes):
    require(bool(registered) and len(registered) == len(set(registered)), "Frozen unique population required")
    require(len(scopes) == len({scope["sequence"] for scope in scopes}), "Repeated video is not another unit")
    require({scope["sequence"] for scope in scopes} <= set(registered), "Unregistered video")
    valid, absent = [], []
    for scope in scopes:
        if count(scope["valid_original_clicks"]) == 0:
            require(scope["complete_MOT_AA_proof"] is None
                    and count(scope["paired_frame_exposures_NOT_independent_sample_size"]) == 0
                    and scope["status"] == "NOT_RUN_NO_VALID_CLICK_INITIALIZATIONS_RETAINED",
                    "Empty exposure cannot be A/A proof")
            absent.append(scope["sequence"])
        else:
            require(scope["complete_MOT_AA_proof"] is True
                    and count(scope["paired_frame_exposures_NOT_independent_sample_size"]) > 0,
                    "Nonempty scope must actually complete raw checks")
            valid.append(scope["sequence"])
    present = {scope["sequence"] for scope in scopes}
    return {"registered_sequence_count": len(registered), "sealed_sequences_reverified": len(scopes),
            "nonempty_full_video_AA_sequences": valid, "retained_all_failed_NOT_RUN_sequences": absent,
            "missing_sequences": [sequence for sequence in registered if sequence not in present],
            "all_registered_sequence_dispositions_reverified": len(present) == len(registered),
            "actual_paired_frame_exposures_NOT_independent_sample_size": sum(
                scope["paired_frame_exposures_NOT_independent_sample_size"] for scope in scopes),
            "independent_correction_roots": None, "full_G0_or_scientific_PASS": False,
            "scientific_goal_complete": False, "next_stage_authorized": False}
