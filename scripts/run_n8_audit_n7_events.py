#!/usr/bin/env python
"""N8-A: semantic re-audit of N7 budget events (read-only).

Classifies every accepted N7 event as:
  FIRST_APPEARANCE_RENAME / TRUE_MISS_NEW / TRUE_ID_BREAK / TRUE_RECOVER /
  TRUE_SWAP / UNKNOWN
based on the frozen P0 backbone, current-frame GT and whether the GT identity
was already seen and matched in the past.
"""

import csv
import json
from pathlib import Path

import numpy as np

from sam3_intermot.datasets.dancetrack import DanceTrackDataset
from sam3_intermot.interaction.continuous_observer import match_boxes, read_mot_rows


ROOT = Path(".")
OUT = ROOT / "outputs/n8/audit"
REAL = ROOT / "outputs/n7/real_cpu"
P0_DIR = ROOT / "outputs/n5/integrity/canonical_mot_results/b0"
DS = DanceTrackDataset("/path/to/dancetrack", split="val")
SEQS = sorted(
    p.name for p in Path("/path/to/dancetrack/val").iterdir()
    if p.is_dir() and (p / "gt" / "gt.txt").is_file()
)
IOU = 0.5


def classify(seq: str, event: dict, p0_rows, gt, memory) -> dict:
    frame = int(event["frame"]) - 1
    gid = event.get("gt_id")
    action = event["action_type"]
    g = gt.get(frame)
    gb = None
    # N6Event jsonl does not persist gt_id; resolve via authoritative box.
    box = event.get("authoritative_box")
    if g is not None and box is not None and len(box) == 4:
        b = np.asarray(box, dtype=float)
        best_gi, best_iou = None, IOU
        for gi, gbx in enumerate(g.boxes):
            from sam3_intermot.tracking.association import box_iou
            iou = box_iou(b, np.asarray(gbx, dtype=float))
            if iou > best_iou:
                best_iou = iou
                best_gi = gi
        if best_gi is not None:
            gid = g.gt_ids[best_gi]
            gb = np.asarray(g.boxes[best_gi], dtype=float)
    if g is not None and gid is not None and gid in g.gt_ids:
        gb = np.asarray(g.boxes[g.gt_ids.index(gid)], dtype=float)
    matched_tid = None
    if gb is not None:
        rows = p0_rows.get(frame, [])
        pm = match_boxes([gb], [np.asarray(b, float) for _, b in rows], IOU)
        if pm:
            matched_tid = rows[pm[0][1]][0]
    seen_before = gid in memory
    if not seen_before:
        if action in ("ADD_NEW_IDENTITY", "RECOVER_IDENTITY"):
            cat = "FIRST_APPEARANCE_RENAME" if matched_tid is not None else "TRUE_MISS_NEW"
        else:
            cat = "UNKNOWN"
        if gid is not None:
            memory[gid] = {"canonical": matched_tid, "last": matched_tid, "seen": 0}
    else:
        canonical = memory[gid]["canonical"]
        if action in ("AUTHORITATIVE_REASSIGN",):
            cat = "TRUE_ID_BREAK" if matched_tid is not None and matched_tid != canonical else "UNKNOWN"
        elif action == "RECOVER_IDENTITY":
            cat = "TRUE_RECOVER"
        elif action == "ATOMIC_ID_SWAP":
            cat = "TRUE_SWAP"
        else:
            cat = "UNKNOWN"
        if gid is not None:
            memory[gid]["last"] = matched_tid
            memory[gid]["seen"] += 1
    return {
        "sequence": seq,
        "budget": event.get("budget"),
        "frame": event["frame"],
        "action_type": action,
        "gt_id": gid,
        "public_mot_id": event.get("public_mot_id"),
        "matched_p0_tid": matched_tid,
        "seen_before": seen_before,
        "classification": cat,
    }


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    counts = {}
    for b in (1, 2, 4, 8):
        for seq in SEQS:
            ev_path = REAL / f"route_a_b{b}" / seq / "events.jsonl"
            if not ev_path.exists():
                continue
            events = [
                json.loads(l)
                for l in ev_path.read_text().splitlines()
                if l.strip()
            ]
            p0 = read_mot_rows(P0_DIR / f"{seq}.txt")
            gt = DS.load_gt(seq)
            memory = {}
            for e in events:
                r = classify(seq, e, p0, gt, memory)
                rows.append(r)
                counts[r["classification"]] = counts.get(r["classification"], 0) + 1
    with (OUT / "n7_event_semantic_reaudit.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    total = sum(counts.values())
    summary = {
        "total_events": total,
        "by_classification": counts,
        "first_appearance_rename_fraction": round(counts.get("FIRST_APPEARANCE_RENAME", 0) / total, 4) if total else 0.0,
    }
    (OUT / "n7_event_semantic_reaudit_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
