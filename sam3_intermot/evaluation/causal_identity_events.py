"""Offline causal-event bookkeeping, never a runtime decision feature.

An observed error interval is not a proven causal onset. Only independent
same-prestate branches can establish action benefit or propagated harm.
"""
from collections import defaultdict


def intervals(rows, key="category"):
    result = []
    for row in rows:
        f = int(row["frame"])
        if not result or row[key] != result[-1][key] or f != result[-1]["end_frame"] + 1:
            result.append({key: row[key], "start_frame": f, "end_frame": f, "frames": 1, "causal_origin_proven": False})
        else:
            result[-1]["end_frame"] = f
            result[-1]["frames"] += 1
    return result


def direct_vs_propagated(rows):
    n01 = sum(bool(r["actual_correct"] and not r["C0_correct"]) for r in rows)
    n10 = sum(bool(r["C0_correct"] and not r["actual_correct"]) for r in rows)
    roots = [int(r["frame"]) for r in rows if r.get("effective_action") and r.get("paired_causal_branch_sealed")
             and r.get("independent_causal_boundary")]
    # Wrong KEEP on a contaminated trajectory is never a new intervention.
    return {"N01_frames": n01, "N10_frames": n10, "independent_branch_verified_action_onsets": roots,
            "independent_action_onset_count": len(roots), "frames_not_independent_events": True}


def candidate_identity_outcome(uid, matching, target):
    if uid is None:
        return "NONE"
    if uid not in matching:
        raise ValueError("UID outside complete current candidate axis")
    if matching[uid] is None:
        return "UNKNOWN"
    return "TARGET" if matching[uid] == target else "VERIFIED_OTHER"


def stratified_onsets(events, maximum_additional=8):
    """Chronological first plus evenly spaced per-class, before action effects."""
    groups = defaultdict(list)
    for event in events:
        for category in event["categories"]:
            groups[category].append(event["frame"])
    selected = set()
    for frames in groups.values():
        frames = sorted(set(frames))
        selected.add(frames[0])
        remaining = frames[1:]
        if len(remaining) <= maximum_additional:
            selected.update(remaining)
        elif maximum_additional:
            selected.update(remaining[round(i * (len(remaining) - 1) / max(1, maximum_additional - 1))] for i in range(maximum_additional))
    return sorted(selected)
