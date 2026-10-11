"""Offline M-A recomputation boundaries; temporary examples are not research."""
from copy import deepcopy
import pytest
from scripts import n72r21r2_memory_gt_readback_v1 as audit
from sam3_intermot.one_click.datasets import strict_candidate_matching
from sam3_intermot.one_click.runtime_file_guard import runtime_file_guard


def fixture():
    event = {"frame": 0, "frames": 5}
    rows = [{"frame": frame, "runtime_gt_read": False, "runtime_future_gt_used": False,
        "extra_clicks": 0, "target_uid": "a", "identity_claim_UID_before_global_feedback": "a",
        "identity_rank1_candidate_UID": "a", "joint_identity_memory_write": False,
        "joint_memory_write_candidate_uid": None, "bank_drift_cosine_distance_to_anchor": .1,
        "committed_memory_diagnostic": {}} for frame in range(5)]
    mapping = {frame: {"a": 7, "b": 8, "u": None} for frame in range(5)}
    visible = {frame: {7, 8} for frame in range(5)}
    return {"trace": rows, "matching": mapping, "visible": visible, "event": event,
            "target": 7, "kind": "fixed", "runtimes_verified": True}


def test_rejected_correct_observations_are_not_dropped_from_retention():
    args = fixture(); row = args["trace"][1]
    row.update(joint_identity_memory_write=True, joint_memory_write_candidate_uid="a")
    result = audit.recompute_case(**args)
    assert result["raw_counts"]["accepted_writes"] == result["raw_counts"]["write_TARGET"] == 1
    assert result["raw_counts"]["correct_C0_committed_observations"] == 4
    assert result["accepted_write_provenance"] == [{"frame": 1, "candidate_uid": "a", "outcome": "TARGET"}]


def test_unknown_other_and_none_have_distinct_actual_frame_UID_outcomes():
    args = fixture()
    for frame, uid in ((1, "u"), (2, "b")):
        args["trace"][frame].update(target_uid=uid, identity_claim_UID_before_global_feedback=uid,
            identity_rank1_candidate_UID=uid, joint_identity_memory_write=True, joint_memory_write_candidate_uid=uid)
    args["trace"][3].update(target_uid=None, identity_claim_UID_before_global_feedback=None, identity_rank1_candidate_UID=None)
    result = audit.recompute_case(**args); counts = result["raw_counts"]
    assert counts["accepted_writes"] == 2 and counts["write_UNKNOWN"] == counts["write_VERIFIED_OTHER"] == 1
    assert counts["identity_claim_NONE"] == counts["identity_claim_UNKNOWN"] == counts["identity_claim_VERIFIED_OTHER"] == 1
    assert counts["correct_C0_committed_observations"] == 1
    assert [r["outcome"] for r in result["accepted_write_provenance"]] == ["UNKNOWN", "VERIFIED_OTHER"]


def test_visible_but_missing_positive_is_not_physical_absence():
    args = fixture(); args["matching"][1]["a"] = None
    args["matching"][2]["a"] = None; args["visible"][2] = {8}
    result = audit.recompute_case(**args); c = result["raw_counts"]
    assert c["positive_available_frames"] == 2 and c["physically_visible_frames"] == 3
    assert c["false_presence_no_positive_candidate"] == 2
    assert c["physically_unavailable_frames"] == c["false_presence_physically_unavailable"] == 1


def test_fresh_risk_does_not_invent_original_fixed_only_fields():
    args = fixture(); args["kind"] = "fresh_risk"
    c = audit.recompute_case(**args)["raw_counts"]
    assert "rollback_events" not in c and "physically_unavailable_frames" not in c
    assert c["accepted_writes"] == 0 and c["correct_C0_committed_observations"] == 4


@pytest.mark.parametrize("kind", ["unverified", "outside_UID", "wrong_write_UID", "preanchor_write",
    "duplicate_frame", "truncated", "missing_truth", "GT_runtime", "future_GT", "second_click",
    "positive_without_GT", "nonfinite_drift", "rejected_UID", "nonbool_write"])
def test_inconsistent_scope_or_causality_cannot_validate_labels(kind):
    args = fixture(); row = args["trace"][1]
    if kind == "unverified": args["runtimes_verified"] = False
    if kind == "outside_UID": row["target_uid"] = "not_current"
    if kind == "wrong_write_UID": row.update(joint_identity_memory_write=True, joint_memory_write_candidate_uid="b")
    if kind == "preanchor_write": args["trace"][0].update(joint_identity_memory_write=True, joint_memory_write_candidate_uid="a")
    if kind == "duplicate_frame": args["trace"][2]["frame"] = 1
    if kind == "truncated": args["trace"].pop()
    if kind == "missing_truth": del args["matching"][2]
    if kind == "GT_runtime": row["runtime_gt_read"] = True
    if kind == "future_GT": row["runtime_future_gt_used"] = True
    if kind == "second_click": row["extra_clicks"] = 1
    if kind == "positive_without_GT": args["visible"][1] = {8}
    if kind == "nonfinite_drift": row["bank_drift_cosine_distance_to_anchor"] = float("nan")
    if kind == "rejected_UID": row["joint_memory_write_candidate_uid"] = "a"
    if kind == "nonbool_write": row["joint_identity_memory_write"] = 1
    with pytest.raises(ValueError): audit.recompute_case(**args)


@pytest.mark.parametrize("kind", ["unknown_count", "positive_denominator", "write_outcome", "frame_UID", "drift"])
def test_original_sealed_labels_are_compared_not_rewritten(kind):
    args = fixture(); row = args["trace"][1]
    row.update(target_uid="u", identity_claim_UID_before_global_feedback="u", identity_rank1_candidate_UID="u",
        joint_identity_memory_write=True, joint_memory_write_candidate_uid="u")
    actual = audit.recompute_case(**args); original = deepcopy(actual)
    audit.compare_labels(original, actual)
    if kind == "unknown_count": original["raw_counts"]["write_UNKNOWN"] = 0
    if kind == "positive_denominator": original["raw_counts"]["positive_available_frames"] = 2
    if kind == "write_outcome": original["accepted_write_provenance"][0]["outcome"] = "VERIFIED_OTHER"
    if kind == "frame_UID": original["accepted_write_provenance"][0]["candidate_uid"] = "b"
    if kind == "drift": original["max_bank_drift"] = 0
    with pytest.raises(ValueError): audit.compare_labels(original, actual)
    assert actual["raw_counts"]["write_UNKNOWN"] == 1


def test_actual_strict_truth_rule_is_one_to_one_and_keeps_unmatched_unknown():
    candidates = [{"candidate_uid": "boundary", "box_xyxy": (0, 0, 20, 10)},
                  {"candidate_uid": "below", "box_xyxy": (0, 0, 21, 10)}]
    labels = [{"identity": 7, "box": (0, 0, 10, 10)}]
    matching = strict_candidate_matching(candidates, labels)
    assert matching == {"boundary": 7, "below": None}
    duplicated = strict_candidate_matching([candidates[0], {**candidates[0], "candidate_uid": "second"}], labels)
    assert sum(identity == 7 for identity in duplicated.values()) == 1


def test_raw_GT_requires_verified_sources_and_is_rejected_by_runtime_guard(tmp_path, monkeypatch):
    gt = tmp_path / "unit_sequence" / "gt" / "gt.txt"; gt.parent.mkdir(parents=True)
    gt.write_text("1,7,0,0,10,10,1,1,1\n")
    checksum = audit.sha256(gt)
    monkeypatch.setattr(audit, "TRAIN", tmp_path)
    monkeypatch.setattr(audit, "development_sequence", lambda name: name == "unit_sequence")
    with pytest.raises(ValueError, match="before raw GT"):
        audit.offline_truth("unit_sequence", checksum, runtimes_verified=False)
    with runtime_file_guard(), pytest.raises(ValueError, match="forbidden in runtime"):
        audit.offline_truth("unit_sequence", checksum, runtimes_verified=True)
    labels = audit.offline_truth("unit_sequence", checksum, runtimes_verified=True)
    assert labels[0][0]["identity"] == 7
    with pytest.raises(ValueError, match="checksum changed"):
        audit.offline_truth("unit_sequence", "0" * 64, runtimes_verified=True)
