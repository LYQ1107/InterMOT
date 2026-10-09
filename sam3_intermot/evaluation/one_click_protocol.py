"""Offline one-click target evaluation, never imported as runtime truth.

All frames count once. Missing labels are unknown, not target absence.
Scores at FPR2% describe a heldout diagnostic curve, not threshold selection.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence
import numpy as np


def iou(box_a, box_b):
    if box_a is None or box_b is None:
        return 0.0
    a, b = np.asarray(box_a, float), np.asarray(box_b, float)
    if a.shape != (4,) or b.shape != (4,) or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("invalid evaluation box")
    inter = np.maximum(0, np.minimum(a[2:], b[2:]) - np.maximum(a[:2], b[:2])).prod()
    union = np.maximum(0, a[2:] - a[:2]).prod() + np.maximum(0, b[2:] - b[:2]).prod() - inter
    return float(inter / union) if union > 0 else 0.0


def ratio(numerator, denominator):
    return float(numerator / denominator) if denominator else None


def runs(flags):
    result, start = [], None
    for index, value in enumerate(list(flags) + [False]):
        if value and start is None: start = index
        if not value and start is not None:
            result.append((start, index))
            start = None
    return result


@dataclass(frozen=True)
class FrameTruth:
    frame: int
    target_visible: bool
    target_box_xyxy: tuple | None
    valid_target_candidate_uids: frozenset[str]
    other_person_boxes_xyxy: tuple = ()
    complete_person_labels: bool = True
    annotation_complete: bool = True

    def __post_init__(self):
        if self.target_visible != (self.target_box_xyxy is not None):
            raise ValueError("visibility/box mismatch")
        if self.valid_target_candidate_uids and not self.target_visible:
            raise ValueError("candidate target cannot exist without visible target truth")


def open_set_metrics(probability, available, ranking_correct, *, bins=10):
    p = np.asarray(probability, float)
    y, correct = np.asarray(available, bool), np.asarray(ranking_correct, bool)
    if p.shape != y.shape or y.shape != correct.shape or p.ndim != 1:
        raise ValueError("metric array shape mismatch")
    if not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError("invalid availability probability")
    positive, negative = int(y.sum()), int((~y).sum())
    thresholds = np.r_[np.inf, np.unique(p)[::-1]]
    frontier = []
    for threshold in thresholds:
        accept = p >= threshold
        tp, fp = int((accept & y).sum()), int((accept & ~y).sum())
        frontier.append({"threshold": float(threshold) if np.isfinite(threshold) else None,
                         "fpr": ratio(fp, negative), "presence_recall": ratio(tp, positive),
                         "correct_id_recall": ratio(int((accept & y & correct).sum()), positive),
                         "precision": ratio(tp, tp + fp)})
    legal = [r for r in frontier if r["fpr"] is not None and r["fpr"] <= 0.02]
    recall2 = max((r["correct_id_recall"] for r in legal if r["correct_id_recall"] is not None), default=None)
    # Non-interpolated average precision, grouped at tied scores.
    previous_recall, ap = 0.0, 0.0
    for point in frontier:
        recall = point["presence_recall"]
        if recall is not None and point["precision"] is not None:
            ap += (recall - previous_recall) * point["precision"]
            previous_recall = recall
    ece, bin_records = 0.0, []
    for i in range(bins):
        mask = (p >= i / bins) & ((p < (i + 1) / bins) if i < bins - 1 else (p <= 1))
        if mask.any():
            confidence, rate = float(p[mask].mean()), float(y[mask].mean())
            ece += mask.mean() * abs(confidence - rate)
            bin_records.append({"bin": i, "n": int(mask.sum()), "confidence": confidence, "positive_rate": rate})
    return {"positive_frames": positive, "negative_frames": negative, "recall_at_fpr_2pct": recall2,
            "PR_AUC_average_precision": ap if positive else None, "ECE": float(ece) if len(p) else None,
            "calibration_bins": bin_records, "diagnostic_curve_not_deployment_threshold_tuning": True}


def evaluate_episode(runtime: Sequence[Mapping], truth: Sequence[FrameTruth], *, fps: float, recording_id: str):
    if not np.isfinite(fps) or fps <= 0: raise ValueError("actual positive FPS required")
    if len(runtime) != len(truth) or not runtime: raise ValueError("complete aligned frame axes required")
    frames = [int(r["frame"]) for r in runtime]
    if frames != [r.frame for r in truth] or len(set(frames)) != len(frames): raise ValueError("one frame one decision violated")
    if any(b != a + 1 for a, b in zip(frames, frames[1:])): raise ValueError("unrecorded frame gaps")
    if any(r.get("runtime_future_gt_used") is not False for r in runtime): raise ValueError("unsealed runtime GT assertion")
    if any(not t.annotation_complete for t in truth): raise ValueError("unknown annotation gaps cannot become true absence")
    visible = np.asarray([t.target_visible for t in truth])
    available = np.asarray([bool(t.valid_target_candidate_uids) for t in truth])
    accepted = np.asarray([r["selected_candidate_uid"] is not None for r in runtime])
    correct = np.asarray([bool(r["selected_candidate_uid"] is not None and t.target_visible and iou(r["predicted_box_xyxy"], t.target_box_xyxy) >= .5) for r,t in zip(runtime,truth)])
    ranking = np.asarray([r.get("rank1_candidate_uid", r["selected_candidate_uid"]) in t.valid_target_candidate_uids for r,t in zip(runtime,truth)])
    verified_takeover = np.asarray([bool(accepted[i] and not correct[i] and any(iou(runtime[i]["predicted_box_xyxy"], b) >= .5 for b in t.other_person_boxes_xyxy)) for i,t in enumerate(truth)])
    negative_fp = accepted & ~available
    abs_fp = accepted & ~visible
    write_flags = np.asarray([bool(r["memory_write"]) for r in runtime])
    write_correct = np.asarray([r.get("memory_write_candidate_uid", r["selected_candidate_uid"]) in t.valid_target_candidate_uids for r,t in zip(runtime,truth)])
    takeovers = runs(verified_takeover)
    false_presence = runs(abs_fp)
    eligible_reappearances = [(end, next((i for i in range(end, len(truth)) if not visible[i]), len(truth))) for start,end in runs(~visible) if start > 0 and end < len(truth)]
    reappearances = []
    for start,stop in eligible_reappearances:
        returns = np.flatnonzero(correct[start:stop])
        first = int(returns[0]) if len(returns) else None
        first_accepted = next((i for i in range(start, stop) if accepted[i]), None)
        reappearances.append({"frame": frames[start], "visible_window_frames": stop-start, "reacquired": first is not None,
                              "delay_frames": first, "delay_seconds": first/fps if first is not None else None,
                              "first_accept_correct": bool(correct[first_accepted]) if first_accepted is not None else None,
                              "recall_by_seconds": {str(s): bool(first is not None and first/fps <= s) for s in (1,2,5,10)}})
    accepted_returns = [r for r in reappearances if r["first_accept_correct"] is not None]
    probs = [r.get("candidate_available_probability", r["target_present_probability"]) for r in runtime]
    writes, wrong = int(write_flags.sum()), int((write_flags & ~write_correct).sum())
    retained = int((write_flags & write_correct).sum())
    retention = ratio(retained, int(available.sum()))
    wrong_rate = ratio(wrong, writes)
    target = {"frames": len(runtime), "visible_frames": int(visible.sum()), "candidate_available_frames": int(available.sum()),
              "target_recall_all_visible": ratio(int(correct.sum()), int(visible.sum())),
              "target_recall_given_candidate": ratio(int((correct & available).sum()), int(available.sum())),
              "candidate_coverage_given_visible": ratio(int(available.sum()), int(visible.sum())),
              "verified_wrong_person_takeover_frames": int(verified_takeover.sum()) if all(t.complete_person_labels for t in truth) else None,
              "verified_wrong_person_takeover_episodes": len(takeovers) if all(t.complete_person_labels for t in truth) else None,
              "longest_verified_takeover_seconds": max((b-a for a,b in takeovers),default=0)/fps if all(t.complete_person_labels for t in truth) else None,
              "false_presence_frames": int(abs_fp.sum()), "false_presence_duration_seconds": int(abs_fp.sum())/fps,
              "longest_false_presence_seconds": max((b-a for a,b in false_presence),default=0)/fps,
              "negative_fpr": ratio(int(negative_fp.sum()), int((~available).sum())),
              "P0_true_absence_fpr": ratio(int(abs_fp.sum()), int((~visible).sum())),
              "P1_present_unavailable_fpr": ratio(int((accepted & visible & ~available).sum()), int((visible & ~available).sum())),
              "none_false_reject_rate": ratio(int((~accepted & available).sum()), int(available.sum())),
              "reacquisition_episodes": reappearances,
              "reacquisition_recall": ratio(sum(r["reacquired"] for r in reappearances), len(reappearances)),
              "reacquisition_precision_first_accept": ratio(sum(r["first_accept_correct"] for r in accepted_returns), len(accepted_returns)),
              "reacquisition_median_delay_seconds": float(np.median([r["delay_seconds"] for r in reappearances if r["reacquired"]])) if any(r["reacquired"] for r in reappearances) else None}
    return {"recording_id": recording_id, "fps": fps, "target": target,
            "open_set": open_set_metrics(probs, available, ranking),
            "memory": {"accepted_writes": writes, "wrong_writes": wrong, "wrong_write_rate": wrong_rate,
                       "eligible_correct_observations": int(available.sum()), "correct_observation_retention": retention,
                       "joint_safety_pass": bool(wrong_rate is not None and retention is not None and wrong_rate <= .02 and retention >= .60)}}
