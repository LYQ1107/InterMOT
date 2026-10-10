"""Full-global prefix joins without relabeling, target splicing or padding."""
from collections import Counter
import math


def structural_late_selection(initializations, plans):
    """Earliest valid sole click, latest registered COMPLETE original horizon.

    Neither GT nor branch outcomes are arguments. Failed clicks remain in the
    returned census; an all-failed sequence is not a successful zero effect.
    """
    if len({e["episode_uid"] for e in initializations}) != len(initializations):
        raise ValueError("Unique original sole-click axis required")
    if {e["episode_uid"] for e in initializations} != {e["episode_uid"] for e in plans}:
        raise ValueError("Complete original click plans required")
    if len({e["episode_uid"] for e in plans}) != len(plans):
        raise ValueError("Duplicate original click plan")
    failures = sum(bool(e["initialization_failure"]) for e in initializations)
    valid = sorted((e for e in initializations if not e["initialization_failure"]),
                   key=lambda e: (e["frame"], e["episode_uid"]))
    if not valid:
        return {"selection": None, "status": "NOT_RUN_ALL_INITIALIZATIONS_FAILED",
                "failed_initialization_slots_retained": failures}
    event = valid[0]
    positions = next(e["frames"] for e in plans if e["episode_uid"] == event["episode_uid"])
    if (positions != sorted(set(positions)) or any(type(f) is not int or f <= event["frame"]
                                                 or f >= event["frames"] for f in positions)):
        raise ValueError("Exact original chronological post-click positions required")
    complete = [f for f in positions if f + 100 < event["frames"]]
    selection = {"episode_uid": event["episode_uid"], "frame": complete[-1]} if complete else None
    return {"selection": selection, "status": "FROZEN_STRUCTURAL_LATE_POSITION" if complete else "NOT_RUN_EARLIEST_VALID_CLICK_HAS_NO_COMPLETE_REGISTERED_WINDOW",
            "failed_initialization_slots_retained": failures,
            "registered_complete_positions_for_selected_click": len(complete)}


def assemble_prefix(source, arm, *, frame, length):
    if type(frame) is not int or type(length) is not int or frame < 1 or length < 1:
        raise ValueError("Original post-prefix frame/length required")
    if [r["frame"] for r in source] != list(range(len(source))) or frame + length > len(source):
        raise ValueError("Complete original source axis, no truncated/padded prefix")
    if len(arm) != length or [r["frame"] for r in arm] != list(range(frame, frame + length)):
        raise ValueError("Exact sealed arm axis required")
    if source[frame]["state_before"] != arm[0]["state_before"]:
        raise ValueError("Arm is not the original same source prestate")
    if source[frame]["target_public_id"] != arm[0]["target_public_id"]:
        raise ValueError("No substitution/relabeling of target public identity")
    # Both FULL global outputs, never the target trajectory from one arm and
    # non-target outputs from another. Inputs are neither written nor relabeled.
    return source[:frame] + arm


def sign(value, tolerance=1e-10):
    if not math.isfinite(value):
        raise ValueError("Actual finite metric delta required")
    return "POSITIVE" if value > tolerance else "NEGATIVE" if value < -tolerance else "ZERO_WITHIN_API_CLI_TOLERANCE"


def paired_sign_counts(window, prefix):
    """Descriptive sign changes, not a threshold or best-action selection."""
    result = Counter({"paired_correlated_arms": 1})
    for name in ("HOTA", "AssA", "IDF1"):
        a, b = sign(window[name]), sign(prefix[name])
        result[name + "_window_" + a + "__prefix_" + b] += 1
        result[name + "_sign_changes"] += a != b
    result["window_nonpositive_but_prefix_HOTA_and_AssA_positive"] += (
        (sign(window["HOTA"]) != "POSITIVE" or sign(window["AssA"]) != "POSITIVE")
        and sign(prefix["HOTA"]) == sign(prefix["AssA"]) == "POSITIVE")
    return dict(result)
