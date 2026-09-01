#!/usr/bin/env python
"""N4-A counterfactual interaction audit.

Causal proxy: branch B = actual b5 MOT output (interaction executed);
branch A = frozen A0 output (same automatic propagation, no interaction).
Each accepted event is scored at t, t+1, t+3, t+5, t+10 and sequence end.
"""

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from sam3_intermot.datasets.dancetrack import DanceTrackDataset
from sam3_intermot.utils.io import atomic_write_json, write_csv


ROOT = Path(__file__).resolve().parents[1]
SEQS = ["dancetrack0004", "dancetrack0005", "dancetrack0007"]
GT_ROOT = Path("/path/to/dancetrack")


def load_mot(path):
    rows = []
    for line in path.read_text().splitlines():
        p = line.split(",")
        if len(p) < 7:
            continue
        rows.append({
            "frame": int(float(p[0])),
            "id": int(float(p[1])),
            "box": np.asarray([float(p[2]), float(p[3]), float(p[2]) + float(p[4]), float(p[3]) + float(p[5])]),
        })
    return rows


def iou(a, b):
    x1 = max(a[0], b[0]); y1 = max(a[1], b[1])
    x2 = min(a[2], b[2]); y2 = min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    ua = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    ub = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    u = ua + ub - inter
    return inter / u if u > 0 else 0.0


def correct_count(rows, gt_rows, frame):
    dets = [r for r in rows if r["frame"] == frame]
    gts = [r for r in gt_rows if r["frame"] == frame]
    matched = 0
    used = set()
    for g in gts:
        for i, d in enumerate(dets):
            if i in used:
                continue
            if iou(g["box"], d["box"]) > 0.25:
                matched += 1
                used.add(i)
                break
    return matched


def false_count(rows, gt_rows, frame):
    dets = [r for r in rows if r["frame"] == frame]
    gts = [r for r in gt_rows if r["frame"] == frame]
    false = 0
    for d in dets:
        if not any(iou(g["box"], d["box"]) > 0.25 for g in gts):
            false += 1
    return false


def main():
    out = ROOT / "outputs" / "n4" / "counterfactual"
    out.mkdir(parents=True, exist_ok=True)
    dataset = DanceTrackDataset(str(GT_ROOT), sequences=SEQS, split="val")
    events = []
    for seq in SEQS:
        ep = ROOT / "outputs" / "n3_smoke" / ("events_b5_%s.jsonl" % seq)
        if not ep.exists():
            continue
        for line in ep.read_text().splitlines():
            e = json.loads(line)
            if e["accepted"]:
                events.append(e)
    events.sort(key=lambda e: (e["sequence"], e["frame_idx"]))

    rows = []
    beneficial = []
    harmful = []
    for e in events:
        seq = e["sequence"]
        t = e["frame_idx"]
        a0 = load_mot(ROOT / "outputs" / "n1_5" / "a0_v2_mot" / (seq + ".txt"))
        b5 = load_mot(ROOT / "outputs" / "n3_smoke" / "mot_results" / "b5" / (seq + ".txt"))
        gt_rows = []
        gt = dataset.load_gt(seq)
        for f, g in gt.items():
            for bid, box in zip(g.gt_ids, g.boxes):
                gt_rows.append({"frame": f + 1, "id": bid, "box": box})

        def ids_at(rows, frame):
            return {r["id"] for r in rows if r["frame"] == frame}

        before_ids = ids_at(b5, max(1, t - 1))
        after_ids = ids_at(b5, t)
        created_new = len(after_ids - before_ids)
        a0_ids_t = ids_at(a0, t)
        unrelated = len((a0_ids_t - after_ids) | (after_ids - a0_ids_t))
        row = {
            "sequence": seq,
            "event_id": e["action_id"],
            "frame": t,
            "action_type": e["action_type"],
            "budget": 5,
            "created_new_mot_id": int(e["new_track_id"] is not None),
            "created_new_sam_object": int(e["new_sam_object_id"] is not None),
            "mot_id_count_before": len(before_ids),
            "mot_id_count_after": len(after_ids),
            "unrelated_track_change_count": unrelated,
        }
        for d in [1, 3, 5, 10]:
            f = t + d
            c_b5 = correct_count(b5, gt_rows, f)
            c_a0 = correct_count(a0, gt_rows, f)
            fb5 = false_count(b5, gt_rows, f)
            fa0 = false_count(a0, gt_rows, f)
            row["correct_person_delta_t%d" % d] = c_b5 - c_a0
            row["false_output_delta_t%d" % d] = fb5 - fa0
            row["id_switch_proxy_t%d" % d] = len(ids_at(b5, f) - ids_at(a0, f))
            row["fragmentation_proxy_t%d" % d] = len(ids_at(b5, f))
        rows.append(row)
        avg_delta = np.mean([
            row["correct_person_delta_t%d" % d] for d in (1, 3, 5, 10)
        ])
        item = {**row, "avg_correct_delta": float(avg_delta)}
        if item["created_new_mot_id"] == 0 and avg_delta >= 0:
            beneficial.append(item)
        else:
            harmful.append(item)

    write_csv(out / "per_event_counterfactual.csv", rows)
    with (out / "harmful_actions.jsonl").open("w", encoding="utf-8") as f:
        for item in harmful:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    with (out / "beneficial_actions.jsonl").open("w", encoding="utf-8") as f:
        for item in beneficial:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    # Per-action summary
    summary_rows = []
    for at in ["Add", "Correct", "Reassign", "Delete"]:
        sub = [r for r in rows if r["action_type"] == at]
        if not sub:
            continue
        summary_rows.append({
            "action_type": at,
            "count": len(sub),
            "created_new_mot_id": sum(r["created_new_mot_id"] for r in sub),
            "avg_correct_delta_t1": np.mean([r["correct_person_delta_t1"] for r in sub]),
            "avg_correct_delta_t3": np.mean([r["correct_person_delta_t3"] for r in sub]),
            "avg_correct_delta_t5": np.mean([r["correct_person_delta_t5"] for r in sub]),
            "avg_correct_delta_t10": np.mean([r["correct_person_delta_t10"] for r in sub]),
            "avg_false_delta_t1": np.mean([r["false_output_delta_t1"] for r in sub]),
            "avg_false_delta_t5": np.mean([r["false_output_delta_t5"] for r in sub]),
            "avg_unrelated_changes": np.mean([r["unrelated_track_change_count"] for r in sub]),
        })
    write_csv(out / "per_action_summary.csv", summary_rows)

    # ID creation attribution: new IDs in b5 vs A0 -> nearest prior accepted action
    new_id_first = {}
    for seq in SEQS:
        a0_ids = set()
        for r in load_mot(ROOT / "outputs" / "n1_5" / "a0_v2_mot" / (seq + ".txt")):
            a0_ids.add(r["id"])
        for r in load_mot(ROOT / "outputs" / "n3_smoke" / "mot_results" / "b5" / (seq + ".txt")):
            if r["id"] not in a0_ids:
                key = (seq, r["id"])
                new_id_first.setdefault(key, r["frame"])
    attribution = []
    for (seq, nid), first_frame in sorted(new_id_first.items(), key=lambda x: (x[0][0], x[1])):
        candidates = [e for e in events if e["sequence"] == seq and e["frame_idx"] <= first_frame]
        if candidates:
            cause = max(candidates, key=lambda e: e["frame_idx"])
            cause_type = cause["action_type"]
        else:
            cause_type = "AUTO_WINDOW_HANDOVER"
        attribution.append({
            "sequence": seq,
            "new_mot_id": nid,
            "first_frame": first_frame,
            "attributed_action": cause_type,
            "attributed_event_frame": cause.get("frame_idx") if candidates else "",
        })
    write_csv(out / "id_creation_attribution.csv", attribution)
    atomic_write_json(
        out / "audit_summary.json",
        {
            "events": len(rows),
            "harmful": len(harmful),
            "beneficial": len(beneficial),
            "new_ids_total": len(new_id_first),
            "id_creation_by_action": dict(Counter(r["attributed_action"] for r in attribution)),
        },
    )
    print(json.dumps({
        "events": len(rows),
        "harmful": len(harmful),
        "beneficial": len(beneficial),
        "new_ids_total": len(new_id_first),
        "id_creation_by_action": dict(Counter(r["attributed_action"] for r in attribution)),
        "per_action": summary_rows,
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
