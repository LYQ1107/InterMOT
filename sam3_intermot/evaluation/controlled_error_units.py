"""Offline single-action error units, never natural/G1 independent roots.

A paired controlled action can prove its own current error. A chronological
wrong interval, source-case name or late recovery cannot supply this proof.
"""
import hashlib
import json
from .causal_identity_events import candidate_identity_outcome
from .safe_intervention_events import assignment_map


def stable_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def outcome(row, truth, target, public):
    assigned = assignment_map(row)
    if len(assigned) != len(row["outputs"]) or len(set(assigned.values())) != len(assigned):
        raise ValueError("Unique complete actual global assignment required")
    if set(assigned.values()) != set(truth):
        raise ValueError("No oracle injection, dropped candidate or incomplete current axis")
    if row["target_public_id"] != public or row["target_uid"] != assigned.get(public):
        raise ValueError("Actual target-public output identity mismatch")
    return candidate_identity_outcome(assigned.get(public), truth, target)


def source_error_unit(parent, reference, matched, target, scope):
    """Reference is paired actual KEEP prefix or original exact KEEP tape.

    Only a known single effective non-KEEP action from an equal tensor prestate
    can cause a verified TARGET -> OTHER/NONE error. Unknown is not wrong.
    """
    effects = parent["effective_source_action_frames"]
    base = {"source_case": parent["source_case"], "source_frame": parent["source_frame"],
            "independent_natural_or_actual_model_error_root": False}
    if not effects:
        return {**base, "status": "NO_EFFECTIVE_SOURCE_ACTION", "unit_uid": None}
    if len(effects) != 1 or type(effects[0]) is not int:
        raise ValueError("Multiple/invalid source actions cannot be a single-error unit")
    frame = effects[0]
    if not parent["actual_no_write_bank"] or not parent["source_history_not_model_generated_on_policy"]:
        raise ValueError("Only original controlled no-write source experiments")
    actual = {r["frame"]: r for r in parent["own_actual_source_prefix"]}
    keep = {r["frame"]: r for r in reference}
    if len(actual) != len(parent["own_actual_source_prefix"]) or len(keep) != len(reference):
        raise ValueError("Duplicate source/reference frame")
    if frame not in actual or frame not in keep:
        return {**base, "status": "UNPROVEN_NO_EXACT_ROOT_FRAME_PAIR", "unit_uid": None}
    a, b = actual[frame], keep[frame]
    before = a.get("full_tracker_state_before_sha256")
    if not before or before != b.get("full_tracker_state_before_sha256"):
        return {**base, "status": "UNPROVEN_NO_EQUAL_ROOT_TENSOR_PRESTATE", "unit_uid": None}
    if a["selected_action"]["family"] == "KEEP" or assignment_map(a) == assignment_map(b):
        raise ValueError("A declared source onset must have an actual non-KEEP global change")
    public = a["target_public_id"]
    actual_outcome, keep_outcome = outcome(a, matched[frame], target, public), outcome(b, matched[frame], target, public)
    base.update(root_frame=frame, target_public_id=public, actual_outcome=actual_outcome, KEEP_outcome=keep_outcome,
                source_action=a["selected_action"], paired_root_tensor_prestate_sha256=before)
    if keep_outcome != "TARGET":
        return {**base, "status": "REFERENCE_NOT_TARGET_CORRECT_NOT_ISOLATED_ERROR", "unit_uid": None}
    if actual_outcome == "UNKNOWN":
        return {**base, "status": "UNKNOWN_SOURCE_OUTCOME_NOT_VERIFIED_ERROR", "unit_uid": None}
    if actual_outcome not in ("VERIFIED_OTHER", "NONE"):
        return {**base, "status": "SOURCE_ACTION_NOT_CURRENT_TARGET_ERROR", "unit_uid": None}
    # Same actual frame/prestate/action/root identity aliases are one unit,
    # regardless of source-case name or number of later probes/propagated frames.
    key = {**scope, "root_frame": frame, "target_public_id": public,
           "source_action": a["selected_action"], "prestate_sha256": before}
    return {**base, "status": "VERIFIED_CONTROLLED_SINGLE_ACTION_CURRENT_ERROR_UNIT",
            "unit_uid": "CONTROLLED_ERROR__" + stable_hash(key), "scope": scope,
            "unit_key": key, "current_causal_error_proved": True,
            "population_or_distinct_natural_error_independence_proved": False}


def attributable_probe(unit, own_keep, unmodified_c0, truth, target, frame):
    if unit["unit_uid"] is None:
        return "UNPROVEN_OR_NO_SOURCE_ERROR_UNIT"
    if frame <= unit["root_frame"]:
        return "NOT_AFTER_SOURCE_ERROR"
    if own_keep["frame"] != frame or unmodified_c0["frame"] != frame:
        raise ValueError("Only exact original probe frame, no nearest-frame truth")
    public = unit["target_public_id"]
    baseline = outcome(unmodified_c0, truth, target, public)
    contaminated = outcome(own_keep, truth, target, public)
    if baseline != "TARGET":
        return "C0_NOT_CORRECT_AT_PROBE_CANNOT_ISOLATE_SOURCE_CAUSE"
    if contaminated == "UNKNOWN":
        return "UNKNOWN_OWN_KEEP_NOT_VERIFIED_SOURCE_ERROR"
    if contaminated == "TARGET":
        return "SOURCE_ERROR_ALREADY_RECOVERED"
    return "KNOWN_SOURCE_ERROR_PRESENT_AT_PROBE"


def current_repair_label(probe_status, labels, actual_outcome, effective, *, duplicate=False):
    """Distinct current event-correction label; future benefit is separate."""
    if probe_status != "KNOWN_SOURCE_ERROR_PRESENT_AT_PROBE":
        return None
    if duplicate:
        return None
    current = labels["current_t"]
    if current["outcome"] != actual_outcome or current["KEEP_outcome"] not in ("VERIFIED_OTHER", "NONE"):
        raise ValueError("Current paired label is inconsistent with the known source error")
    if actual_outcome == "UNKNOWN":
        return None  # Cannot prove non-repair merely from a strict unmatched UID.
    corrected = actual_outcome == "TARGET"
    if corrected and (not effective or current["N01"] != 1 or current["N10"]):
        raise ValueError("Correcting an error requires an actual paired current N01 action")
    return {"current_error_unit_corrected": int(corrected),
        "scope": "CONTROLLED_CAUSAL_ERROR_UNIT_CURRENT_CORRECTION_NOT_NATURAL_G1_ROOT",
        "current_harm": bool(current["non_target_damage"] or current["verified_OTHER_writes"] or current["UNKNOWN_writes"] or current["N10"]),
        "future_H100_complete": labels["future"]["H100"]["complete"],
        "future_H100_benefit_NOT_current_event_correction": labels["future"]["H100"]["benefit_label"],
        "future_H100_risk": labels["future"]["H100"]["risk_label"],
        "independent_natural_error_root_proved": False}
