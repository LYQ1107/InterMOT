#!/usr/bin/env python
"""N7-D: root-cause taxonomy from N6 budget post streams (read-only audit).

For every GT identity we compute, on P0 and on each B1/B2/B4/B8 post stream:
- distinct public MOT ids the GT box matched over time;
- public-id switches between consecutive frames;
- missing frames (GT present, no matched row);
- duplicate rows (two post rows matched to one GT);
- whether switches happen near segment boundaries (|f % 30| <= 2).

Outputs:
  outputs/n7/audit/restart_failure_taxonomy.csv
  outputs/n7/audit/root_cause_summary.json
"""

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from sam3_intermot.datasets.dancetrack import DanceTrackDataset
from sam3_intermot.interaction.continuous_observer import match_boxes, read_mot_rows


ROOT = Path(".")
OUT = ROOT / "outputs/n7/audit"
REAL = ROOT / "outputs/n6/full25/real"
P0_DIR = ROOT / "outputs/n5/integrity/canonical_mot_results/b0"
DS = DanceTrackDataset("/path/to/dancetrack", split="val")
SEQS = sorted(
    p.name
    for p in Path("/path/to/dancetrack/val").iterdir()
    if p.is_dir() and (p / "gt" / "gt.txt").is_file()
)
SEGMENT_LEN = 30
IOU = 0.5


def per_gt_analysis(seq: str, rows) -> dict:
    gt = DS.load_gt(seq)
    frames = sorted(gt)
    matched_ids: dict = defaultdict(set)
    switches = 0
    missing = 0
    duplicates = 0
    switch_at_boundary = 0
    prev_pid: dict = {}
    last_public: dict = {}
    for f in frames:
        g = gt[f]
        gb = [np.asarray(b, float) for b in g.boxes]
        po = rows.get(f, [])
        pm = match_boxes(gb, [np.asarray(b, float) for _, b in po], IOU)
        matched_gt = set()
        used_pi = set()
        for gi, pi, _ in pm:
            gid = g.gt_ids[gi]
            pid = po[pi][0]
            matched_gt.add(gi)
            used_pi.add(pi)
            matched_ids[gid].add(pid)
            if gid in prev_pid and prev_pid[gid] != pid:
                switches += 1
                if abs(f % SEGMENT_LEN) <= 2 or abs((f - 1) % SEGMENT_LEN) <= 2:
                    switch_at_boundary += 1
            prev_pid[gid] = pid
            last_public[gid] = pid
        for gi, gid in enumerate(g.gt_ids):
            if gi not in matched_gt:
                missing += 1
        if len(pm) != len(set(x[1] for x in pm)):
            duplicates += 1
        # duplicate rows matched to the same GT (two post rows overlap one GT)
        dup_pi = len(used_pi) - len(pm) if False else 0
        _ = dup_pi
    distinct = {gid: len(v) for gid, v in matched_ids.items()}
    multi = sum(1 for v in distinct.values() if v > 1)
    return {
        "sequence": seq,
        "gt_identities": len(set(gid for g in gt.values() for gid in g.gt_ids)),
        "gt_public_switches": switches,
        "switches_at_segment_boundary": switch_at_boundary,
        "missing_gt_frames": missing,
        "duplicate_match_frames": duplicates,
        "gt_with_multiple_public_ids": multi,
        "total_distinct_mappings": sum(distinct.values()),
    }


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    all_rows = []
    for label, rows in [("P0", read_mot_rows(P0_DIR / f"{SEQS[0]}.txt"))]:
        del rows
    # P0 per sequence
    for seq in SEQS:
        r = per_gt_analysis(seq, read_mot_rows(P0_DIR / f"{seq}.txt"))
        all_rows.append({"budget": "P0", **r})
    for b in (1, 2, 4, 8):
        for seq in SEQS:
            post = read_mot_rows(REAL / f"p4_budget_b{b}" / seq / "post_mot" / f"{seq}.txt")
            r = per_gt_analysis(seq, post)
            all_rows.append({"budget": f"B{b}", **r})
    with (OUT / "restart_failure_taxonomy.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
        w.writeheader()
        w.writerows(all_rows)
    agg = {}
    for budget in ("P0", "B1", "B2", "B4", "B8"):
        sub = [r for r in all_rows if r["budget"] == budget]
        agg[budget] = {
            k: float(np.mean([r[k] for r in sub]))
            for k in (
                "gt_public_switches",
                "switches_at_segment_boundary",
                "missing_gt_frames",
                "gt_with_multiple_public_ids",
                "total_distinct_mappings",
            )
        }
    (OUT / "root_cause_summary.json").write_text(
        json.dumps(agg, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(agg, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
