from copy import deepcopy
import pytest
from sam3_intermot.evaluation import controlled_error_units as units


def row(frame=1, uid="other", family="GLOBAL_SWAP", before="same"):
    outputs = [{"public_id": 1 if u == uid else 2, "candidate_uid": u} for u in ("target", "other")]
    if uid is None:
        outputs = [{"public_id": 2, "candidate_uid": "target"}, {"public_id": 3, "candidate_uid": "other"}]
    return {"frame": frame, "target_public_id": 1, "target_uid": uid, "outputs": outputs,
        "full_tracker_state_before_sha256": before,
        "selected_action": {"family": family, "public_id": 1, "candidate_uid": uid}}


def parent():
    return {"source_case": "TOP_ALTERNATIVE", "source_frame": 1,
        "actual_no_write_bank": True, "source_history_not_model_generated_on_policy": True,
        "effective_source_action_frames": [1], "own_actual_source_prefix": [row()]}


TRUTH = {"target": 7, "other": 8}
SCOPE = {"sequence": "video", "anonymous_identity_scope": "anonymous"}


def unit():
    return units.source_error_unit(parent(), [row(uid="target", family="KEEP")], {1: TRUTH}, 7, SCOPE)


def test_exact_controlled_action_proves_current_error_not_natural_independence():
    u = unit()
    assert u["status"] == "VERIFIED_CONTROLLED_SINGLE_ACTION_CURRENT_ERROR_UNIT"
    assert u["current_causal_error_proved"]
    assert not u["population_or_distinct_natural_error_independence_proved"]
    assert not u["independent_natural_or_actual_model_error_root"]


def test_case_names_and_repeated_probes_cannot_create_new_controlled_units():
    a = unit(); p = parent(); p["source_case"] = "LEARNED_IDENTITY_TOP"
    b = units.source_error_unit(p, [row(uid="target", family="KEEP")], {1: TRUTH}, 7, SCOPE)
    assert a["unit_uid"] == b["unit_uid"]
    for frame in range(2, 102):
        assert units.attributable_probe(a, row(frame), row(frame, "target", "KEEP"), TRUTH, 7, frame) == "KNOWN_SOURCE_ERROR_PRESENT_AT_PROBE"
    assert a["unit_uid"] == b["unit_uid"]  # Hundred propagated frames, still one source unit.


@pytest.mark.parametrize("change,status", [
    ("no_action", "NO_EFFECTIVE_SOURCE_ACTION"), ("missing_pair", "UNPROVEN_NO_EXACT_ROOT_FRAME_PAIR"),
    ("missing_tensor", "UNPROVEN_NO_EQUAL_ROOT_TENSOR_PRESTATE"),
    ("different_tensor", "UNPROVEN_NO_EQUAL_ROOT_TENSOR_PRESTATE"),
    ("unknown", "UNKNOWN_SOURCE_OUTCOME_NOT_VERIFIED_ERROR"),
    ("base_wrong", "REFERENCE_NOT_TARGET_CORRECT_NOT_ISOLATED_ERROR"),
    ("not_error", "SOURCE_ACTION_NOT_CURRENT_TARGET_ERROR")])
def test_missing_pair_ambiguity_or_unknown_cannot_create_a_verified_root(change, status):
    p, reference, truth = parent(), [row(uid="target", family="KEEP")], dict(TRUTH)
    if change == "no_action": p["effective_source_action_frames"] = []
    elif change == "missing_pair": reference = []
    elif change == "missing_tensor": reference[0].pop("full_tracker_state_before_sha256")
    elif change == "different_tensor": reference[0]["full_tracker_state_before_sha256"] = "different"
    elif change == "unknown": truth["other"] = None
    elif change == "base_wrong": truth["target"] = 9
    else:
        p["own_actual_source_prefix"][0]["target_uid"] = "target"
        p["own_actual_source_prefix"][0]["outputs"] = [{"public_id": 1, "candidate_uid": "target"}, {"public_id": 3, "candidate_uid": "other"}]
    u = units.source_error_unit(p, reference, {1: truth}, 7, SCOPE)
    assert u["status"] == status and u["unit_uid"] is None


def test_multiple_actions_or_false_onset_is_not_an_independent_error():
    p = parent(); p["effective_source_action_frames"] = [1, 2]
    with pytest.raises(ValueError): units.source_error_unit(p, [row(uid="target")], {1: TRUTH}, 7, SCOPE)
    p = parent(); p["own_actual_source_prefix"][0]["selected_action"]["family"] = "KEEP"
    with pytest.raises(ValueError): units.source_error_unit(p, [row(uid="target")], {1: TRUTH}, 7, SCOPE)


def test_controlled_reject_with_positive_candidate_is_verified_error():
    p = parent(); p["own_actual_source_prefix"] = [row(uid=None, family="REJECT_TARGET")]
    u = units.source_error_unit(p, [row(uid="target", family="KEEP")], {1: TRUTH}, 7, SCOPE)
    assert u["unit_uid"] and u["actual_outcome"] == "NONE"


def test_probe_shared_natural_error_or_recovery_is_not_source_cause():
    u = unit()
    assert units.attributable_probe(u, row(2), row(2, "target"), TRUTH, 7, 2) == "KNOWN_SOURCE_ERROR_PRESENT_AT_PROBE"
    assert units.attributable_probe(u, row(2), row(2), TRUTH, 7, 2) == "C0_NOT_CORRECT_AT_PROBE_CANNOT_ISOLATE_SOURCE_CAUSE"
    assert units.attributable_probe(u, row(2, "target"), row(2, "target"), TRUTH, 7, 2) == "SOURCE_ERROR_ALREADY_RECOVERED"
    assert units.attributable_probe(u, row(1), row(1, "target"), TRUTH, 7, 1) == "NOT_AFTER_SOURCE_ERROR"
    with pytest.raises(ValueError): units.attributable_probe(u, row(3), row(2, "target"), TRUTH, 7, 2)


def labels(outcome="TARGET", n01=1):
    return {"current_t": {"outcome": outcome, "KEEP_outcome": "VERIFIED_OTHER", "N01": n01,
        "N10": 0, "non_target_damage": 0, "verified_OTHER_writes": 0, "UNKNOWN_writes": 0},
        "future": {"H100": {"complete": True, "benefit_label": True, "risk_label": False}}}


def test_future_benefit_alone_is_not_current_event_correction():
    d = units.current_repair_label("KNOWN_SOURCE_ERROR_PRESENT_AT_PROBE", labels("VERIFIED_OTHER", 0), "VERIFIED_OTHER", False)
    assert d["current_error_unit_corrected"] == 0 and d["future_H100_benefit_NOT_current_event_correction"]
    positive = units.current_repair_label("KNOWN_SOURCE_ERROR_PRESENT_AT_PROBE", labels(), "TARGET", True)
    assert positive["current_error_unit_corrected"] == 1 and not positive["independent_natural_error_root_proved"]


def test_current_correction_and_harm_remain_distinct():
    l = labels(); l["current_t"]["non_target_damage"] = 1
    d = units.current_repair_label("KNOWN_SOURCE_ERROR_PRESENT_AT_PROBE", l, "TARGET", True)
    assert d["current_error_unit_corrected"] == 1 and d["current_harm"]


def test_unknown_duplicate_unattributed_or_ineffective_correction_not_promoted():
    assert units.current_repair_label("KNOWN_SOURCE_ERROR_PRESENT_AT_PROBE", labels("UNKNOWN", 0), "UNKNOWN", True) is None
    assert units.current_repair_label("KNOWN_SOURCE_ERROR_PRESENT_AT_PROBE", labels(), "TARGET", True, duplicate=True) is None
    assert units.current_repair_label("SOURCE_ERROR_ALREADY_RECOVERED", labels(), "TARGET", True) is None
    with pytest.raises(ValueError): units.current_repair_label("KNOWN_SOURCE_ERROR_PRESENT_AT_PROBE", labels(), "TARGET", False)


def test_dropped_or_duplicate_global_axis_is_never_root_evidence():
    r = row(); r["outputs"].pop()
    with pytest.raises(ValueError): units.outcome(r, TRUTH, 7, 1)
    r = row(); r["outputs"].append(deepcopy(r["outputs"][0]))
    with pytest.raises(ValueError): units.outcome(r, TRUTH, 7, 1)
