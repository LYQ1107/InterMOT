"""Audit event evidence without promoting correlated windows to causal roots.

Pure bookkeeping only. Actual trajectory/GT recomputation belongs to the
separate offline worker; none of these labels may enter a runtime encoder.
"""
from collections import Counter
from .event_causal_labels import SUM_FIELDS
from .safe_intervention_events import contiguous_intervals

HORIZONS = (1, 5, 20, 50, 100)
OUTCOMES = ("TARGET", "VERIFIED_OTHER", "UNKNOWN", "NONE")


def component_audit(labels, frame, *, branch):
    """Independently check stored current/future aggregates and their units."""
    rows = labels["raw_frame_components"]
    if not rows or len(rows) > 101 or labels["current_t"] != rows[0]:
        raise ValueError("Current t must be separate from at most100 future frames")
    for offset, row in enumerate(rows):
        if row["frame"] != frame + offset or row["offset"] != offset:
            raise ValueError("Original contiguous t..t+100 frame axis required")
        outcome, keep = row["outcome"], row["KEEP_outcome"]
        if outcome not in OUTCOMES or keep not in OUTCOMES:
            raise ValueError("UNKNOWN is not VERIFIED_OTHER")
        expected = {"target_correct": outcome == "TARGET", "KEEP_correct": keep == "TARGET",
                    "N01": outcome == "TARGET" and keep != "TARGET",
                    "N10": keep == "TARGET" and outcome != "TARGET",
                    "verified_OTHER": outcome == "VERIFIED_OTHER", "KEEP_verified_OTHER": keep == "VERIFIED_OTHER",
                    "UNKNOWN": outcome == "UNKNOWN", "NONE": outcome == "NONE"}
        if any(row[key] != int(value) for key, value in expected.items()):
            raise ValueError("Raw correctness/outcome/N01/N10 inconsistency")
        if any(type(row[key]) is not int or row[key] < 0 for key in SUM_FIELDS):
            raise ValueError("Raw component counts must be nonnegative integers")
        if row["non_target_damage"] != len(set(row["harmed_public_ids"])) or row["non_target_benefit"] != len(set(row["helped_public_ids"])):
            raise ValueError("Other-public damage/benefit axes are inconsistent")
        if row["non_target_damage"] != sum(row[k] for k in ("non_target_verified_OTHER_damage", "non_target_UNKNOWN_damage", "non_target_NONE_damage")):
            raise ValueError("UNKNOWN/non-target harm partition is incomplete")
        if row["memory_writes"] != sum(row[k] for k in ("correct_writes", "verified_OTHER_writes", "UNKNOWN_writes")):
            raise ValueError("Memory write partition is incomplete")
        if branch == "KEEP" and any(row[k] for k in ("N01", "N10", "non_target_damage", "non_target_benefit", "ownership_changed")):
            raise ValueError("Own KEEP reference cannot differ from itself")
    current_harm = bool(rows[0]["N10"] or rows[0]["non_target_damage"] or rows[0]["verified_OTHER_writes"] or rows[0]["UNKNOWN_writes"])
    for horizon in HORIZONS:
        future = rows[1:horizon + 1]
        complete = len(future) == horizon
        stored = labels["future"]["H" + str(horizon)]
        counts = {key: sum(row[key] for row in future) for key in SUM_FIELDS}
        affected = Counter(public for row in future for public in row["harmed_public_ids"])
        target_value = counts["N01"] - counts["N10"]
        proxy = target_value - counts["non_target_damage"] + counts["non_target_benefit"] - counts["verified_OTHER_writes"] - counts["UNKNOWN_writes"]
        risk = bool(counts["N10"] or counts["non_target_damage"] or counts["verified_OTHER_writes"] or counts["UNKNOWN_writes"])
        expected = {**counts, "complete": complete, "observed_future_frames": len(future), "future_offsets": [1, horizon],
                    "affected_public_frame_counts": {str(k): v for k, v in affected.items()},
                    "severe_non_target_harm_public_ids": sorted(p for p, n in affected.items() if n >= 20),
                    "raw_target_value": target_value, "raw_global_component_proxy": proxy,
                    "risk_label": risk if complete else None, "benefit_label": bool(proxy > 0 and not risk) if complete else None,
                    "normalized_target_value_label": target_value / horizon if complete else None,
                    "normalized_global_proxy_label": proxy / horizon if complete else None}
        if any(stored[key] != value for key, value in expected.items()):
            raise ValueError("Stored future aggregates differ from exact k=1..H raw components")
    for kind in ("N01", "N10"):
        expected = contiguous_intervals(row["frame"] for row in rows[1:] if row[kind])
        if labels[kind + "_propagated_intervals_not_action_onsets"] != expected:
            raise ValueError("Propagated intervals cannot be new direct decisions")
    takeover = contiguous_intervals(row["frame"] for row in rows if row["outcome"] == "VERIFIED_OTHER")
    if labels["verified_OTHER_takeover_intervals"] != takeover or labels["max_observed_verified_OTHER_takeover_duration"] != max((r["frames"] for r in takeover), default=0):
        raise ValueError("Verified OTHER duration differs from original frame evidence")
    actions = labels["effective_direct_action_frames"]
    if len(actions) > 1 or actions != sorted(set(actions)) or any(f < frame or f > min(frame + 2, rows[-1]["frame"]) for f in actions):
        raise ValueError("Only a current one-shot or preregistered at-most-t+2 delayed action is allowed")
    if labels["direct_action_onsets_in_this_one_shot_branch"] != len(actions) or (branch == "KEEP" and actions):
        raise ValueError("Wrong direct-decision count")
    if not labels["distinct_correlated_window_not_independent_causal_origin"] or not labels["overlapping_branch_windows_not_independent_correction_events"]:
        raise ValueError("Overlapping windows must retain their unproven independence status")
    full = labels["future"]["H100"]
    expected_harm = bool(current_harm or full["risk_label"]) if full["complete"] else None
    if labels["H100_any_harm_including_current_t_label"] != expected_harm:
        raise ValueError("Incomplete windows cannot receive a complete-future harm label")
    return {"raw_current_plus_future_rows": len(rows), "direct_decisions_not_independent_roots": len(actions),
            "current_N01": rows[0]["N01"], "current_N10": rows[0]["N10"],
            "future_H100_N01_overlapping_arm_frames": full["N01"], "future_H100_N10_overlapping_arm_frames": full["N10"],
            "complete_H100": full["complete"], "current_harm": current_harm,
            "safe_positive_H100_arm_not_independent_correction": full["complete"] and full["benefit_label"] is True and not current_harm,
            "harmful_observed_arm_including_incomplete_current": full["risk_label"] is True or current_harm,
            "complete_H100_any_harm_arm": full["complete"] and expected_harm,
            "severe_other_harm_H100_arm": bool(full["severe_non_target_harm_public_ids"]),
            "N01_future_interval_count_not_roots": len(labels["N01_propagated_intervals_not_action_onsets"]),
            "N10_future_interval_count_not_roots": len(labels["N10_propagated_intervals_not_action_onsets"]),
            "incomplete_H100_arm": not full["complete"],
            "duplicate_executed_action_configuration": labels["same_executed_action_configuration_as"] is not None,
            "independent_beneficial_correction_roots": None, "independent_N10_roots": None}


def original_counts(audits):
    """Exactly reconstruct original label-worker counters, including zero keys."""
    return {"actual_arm_records": len(audits),
            "complete_H100_arms": sum(r["complete_H100"] for r in audits),
            "effective_one_shot_arms": sum(bool(r["direct_decisions_not_independent_roots"]) for r in audits),
            "duplicate_executed_action_configs": sum(r["duplicate_executed_action_configuration"] for r in audits),
            "observed_safe_positive_H100_arms_NOT_independent_corrections": sum(r["safe_positive_H100_arm_not_independent_correction"] for r in audits),
            "observed_harmful_H100_arms": sum(r["harmful_observed_arm_including_incomplete_current"] for r in audits),
            "severe_other_harm_H100_arms": sum(r["severe_other_harm_H100_arm"] for r in audits)}


def overlap_components(intervals):
    """Union overlapping windows, per (video, anonymous identity), NOT roots.

    Different simulated clicks of the same video-local identity are correlated.
    A union component provides conservative correlation bookkeeping, never
    experimental independence or a cause of a baseline error.
    """
    ordered = sorted(intervals, key=lambda r: (r["sequence"], r["anonymous_identity_scope"], r["start"], r["end"]))
    result = []
    for row in ordered:
        if row["end"] < row["start"]:
            raise ValueError("Invalid original-frame interval")
        key = row["sequence"], row["anonymous_identity_scope"]
        if result and key == (result[-1]["sequence"], result[-1]["anonymous_identity_scope"]) and row["start"] <= result[-1]["end"]:
            result[-1]["end"] = max(result[-1]["end"], row["end"])
            result[-1]["registered_window_count"] += 1
        else:
            result.append({"sequence": key[0], "anonymous_identity_scope": key[1], "start": row["start"], "end": row["end"],
                           "registered_window_count": 1, "independent_causal_origin_proven": False})
    return result
