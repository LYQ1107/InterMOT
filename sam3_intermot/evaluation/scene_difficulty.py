"""Offline scene descriptors, not new density labels or policy selectors."""
from collections import Counter
import numpy as np


def distribution(values):
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or not np.isfinite(values).all():
        raise ValueError("Finite one-dimensional descriptive observations required")
    if not len(values):
        return {"observations_NOT_independent_events": 0, "mean": None, "quantiles": None}
    return {"observations_NOT_independent_events": len(values), "mean": float(values.mean()),
            "quantiles": dict(zip(("P10", "P25", "P50", "P75", "P90"),
                                   np.quantile(values, (.1, .25, .5, .75, .9)).tolist(), strict=True)),
            "minimum": float(values.min()), "maximum": float(values.max())}


def checked_annotations(rows):
    if len({r["identity"] for r in rows}) != len(rows):
        raise ValueError("GT identities are unique only within a video/frame")
    for r in rows:
        b = np.asarray(r["box"], dtype=float)
        if b.shape != (4,) or not np.isfinite(b).all() or b[2] <= b[0] or b[3] <= b[1]:
            raise ValueError("Actual finite positive-area GT boxes required")
    return rows


def max_box_overlap(rows):
    """Intersection/min(box area) is a geometric proxy, not occlusion truth."""
    rows = checked_annotations(rows)
    result = {r["identity"]: 0. for r in rows}
    for i, left in enumerate(rows):
        a = left["box"]
        area_a = (a[2] - a[0]) * (a[3] - a[1])
        for right in rows[i + 1:]:
            b = right["box"]
            intersection = max(0., min(a[2], b[2]) - max(a[0], b[0])) * max(0., min(a[3], b[3]) - max(a[1], b[1]))
            area_b = (b[2] - b[0]) * (b[3] - b[1])
            overlap = intersection / min(area_a, area_b)
            for r in (left, right):
                result[r["identity"]] = max(result[r["identity"]], overlap)
    return result


def transition_reappearance(presence, *, initially_present):
    """Observed return after absence; never infer an independent causal root."""
    seen, previous, gap = bool(initially_present), bool(initially_present), 0
    missing_lengths = []
    for present in presence:
        if present and not previous and seen:
            missing_lengths.append(gap)
        if present:
            seen, gap = True, 0
        else:
            gap += 1
        previous = bool(present)
    return {"observational_returns_NOT_independent_events": len(missing_lengths),
            "preceding_absence_frames": distribution(missing_lengths)}


def describe_video(annotations_by_frame, frames):
    if frames <= 0 or any(f < 0 or f >= frames for f in annotations_by_frame):
        raise ValueError("GT must retain the sealed original frame axis")
    people, visibility, overlaps = [], [], []
    identities = set()
    for f in range(frames):
        rows = checked_annotations(annotations_by_frame.get(f, []))
        people.append(len(rows)); identities.update(r["identity"] for r in rows)
        visibility.extend(r["visibility"] for r in rows)
        overlaps.extend(max_box_overlap(rows).values())
    v = distribution(visibility)
    return {"GT_people_per_original_frame": distribution(people),
            "GT_video_local_identities_NOT_cross_scene_people": len(identities),
            "GT_visibility_annotation": v,
            "GT_visibility_constant_NOT_informative_occlusion_truth": bool(visibility) and min(visibility) == max(visibility),
            "max_GT_box_overlap_per_person_observation_PROXY_NOT_occlusion_truth": distribution(overlaps),
            "GT_box_overlap_at_least_half_person_observation_fraction": sum(x >= .5 for x in overlaps) / len(overlaps) if overlaps else None,
            "occlusion_ground_truth_available": False,
            "density_reclassification": False}


def describe_episode(frames, trace, matches, annotations, anchor, event, target_identity):
    if len(frames) != len(trace) or not 0 <= event["frame"] < len(frames):
        raise ValueError("Full sealed original video and click axis required")
    anchor = np.asarray(anchor, dtype=np.float64)
    if anchor.ndim != 1 or not np.isfinite(anchor).all() or np.linalg.norm(anchor) <= 1e-12:
        raise ValueError("Actual finite nonzero frozen human anchor required")
    anchor = anchor / np.linalg.norm(anchor)
    target_public = trace[event["frame"]]["target_public_id"]
    if target_public is None:
        raise ValueError("Valid sole-click C0 public identity required")
    counts, values, presence, available = Counter(), {k: [] for k in (
        "GT_people", "valid_candidates", "verified_OTHER_candidates", "UNKNOWN_candidates",
        "target_GT_box_overlap", "positive_anchor_cosine", "hardest_verified_OTHER_anchor_cosine",
        "positive_minus_hardest_verified_OTHER_margin", "hardest_nonpositive_including_UNKNOWN_anchor_cosine")}, [], []
    for f in range(event["frame"] + 1, len(frames)):
        p, rows = frames[f]
        if int(p["frame"]) != f or int(trace[f]["frame"]) != f:
            raise ValueError("No renumbered/subset frames")
        if trace[f].get("runtime_gt_read") or trace[f].get("runtime_future_gt_used"):
            raise ValueError("Offline descriptors cannot legalize GT-aided source runtime")
        uid_rows = {str(r["candidate_uid"]): r for r in rows}
        if len(uid_rows) != len(rows) or set(matches[f]) != set(uid_rows):
            raise ValueError("Unique current UIDs and exact offline matching required")
        if trace[f]["target_uid"] is not None and trace[f]["target_uid"] not in uid_rows:
            raise ValueError("Baseline selected UID must exist in the current frame")
        owner = {str(o["candidate_uid"]): o["public_id"] for o in trace[f]["outputs"]}
        if (len(owner) != len(trace[f]["outputs"]) or not set(owner) <= set(uid_rows)
                or len(set(owner.values())) != len(owner)):
            raise ValueError("Actual unique current candidate ownership required")
        gt_rows = checked_annotations(annotations.get(f, []))
        present = any(r["identity"] == target_identity for r in gt_rows)
        positive = [uid for uid, label in matches[f].items() if label == target_identity]
        if len(positive) > 1:
            raise ValueError("Positive availability is strict one-to-one matching")
        other = [uid for uid, label in matches[f].items() if label is not None and label != target_identity]
        unknown = [uid for uid, label in matches[f].items() if label is None]
        presence.append(present); available.append(bool(positive))
        counts.update(postclick_original_frames=1, physically_visible_frames=int(present),
                      strict_positive_available_frames=int(bool(positive)))
        values["GT_people"].append(len(gt_rows)); values["valid_candidates"].append(len(rows))
        values["verified_OTHER_candidates"].append(len(other)); values["UNKNOWN_candidates"].append(len(unknown))
        if present:
            values["target_GT_box_overlap"].append(max_box_overlap(gt_rows)[target_identity])
        scores = {}
        for uid, row in uid_rows.items():
            vector = np.asarray(row["feature"], dtype=np.float64)
            if vector.shape != anchor.shape or not np.isfinite(vector).all() or np.linalg.norm(vector) <= 1e-12:
                raise ValueError("Actual frozen finite compatible candidate embeddings required")
            scores[uid] = float(anchor @ (vector / np.linalg.norm(vector)))
        if other:
            values["hardest_verified_OTHER_anchor_cosine"].append(max(scores[u] for u in other))
        nonpositive = other + unknown
        if nonpositive:
            values["hardest_nonpositive_including_UNKNOWN_anchor_cosine"].append(max(scores[u] for u in nonpositive))
        if positive:
            uid = positive[0]
            ownership = "OWN_TARGET_PUBLIC_ID" if owner.get(uid) == target_public else "OWN_OTHER_PUBLIC_ID" if uid in owner else "UNASSIGNED"
            counts["positive_ownership_" + ownership] += 1
            values["positive_anchor_cosine"].append(scores[uid])
            if other:
                margin = scores[uid] - max(scores[u] for u in other)
                values["positive_minus_hardest_verified_OTHER_margin"].append(margin)
                counts.update(competitive_verified_OTHER_frames=1, strict_positive_wins=int(margin > 0), exact_score_ties=int(margin == 0))
    return {"episode_uid": event["episode_uid"], "status": "OFFLINE_SEALED_C0_SCENE_DESCRIPTOR_NOT_POLICY_RESULT",
            "counts_NOT_independent_events": dict(counts), "distributions": {k: distribution(v) for k, v in values.items()},
            "physical_reappearance": transition_reappearance(presence, initially_present=True),
            "strict_candidate_reappearance": transition_reappearance(available, initially_present=True),
            "verified_OTHER_competition_win_fraction_DESCRIPTIVE_NOT_MOT_GATE": counts["strict_positive_wins"] / counts["competitive_verified_OTHER_frames"] if counts["competitive_verified_OTHER_frames"] else None,
            "UNKNOWN_candidates_NOT_verified_negative_identities": True,
            "candidate_UNASSIGNED_is_NOT_competitor_owned": True,
            "same_raw_human_anchor_no_new_memory_training_or_representation": True,
            "independent_causal_event_roots_proven": False}
