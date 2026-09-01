#!/usr/bin/env python
"""N7-C: event-level audit of N6 sparse-restart budget runs (read-only)."""

import csv
import json
from pathlib import Path

import numpy as np

from sam3_intermot.datasets.dancetrack import DanceTrackDataset
from sam3_intermot.interaction.continuous_observer import match_boxes, read_mot_rows


ROOT = Path(".")
OUT = ROOT / "outputs/n7/audit"
REAL = ROOT / "outputs/n6/full25/real"
DS = DanceTrackDataset("/path/to/dancetrack", split="val")
SEQS = sorted(
    p.name
    for p in Path("/path/to/dancetrack/val").iterdir()
    if p.is_dir() and (p / "gt" / "gt.txt").is_file()
)


def gt_public_switches(seq, rows):
    gt = DS.load_gt(seq)
    g2p = {}
    for f in range(DS.num_frames(seq)):
        g = gt.get(f)
        if not g:
            continue
        gb = [np.asarray(b, float) for b in g.boxes]
        po = rows.get(f, [])
        pm = match_boxes(gb, [np.asarray(b, float) for _, b in po], 0.5)
        pmap = {pi: t for pi, (t, _) in enumerate(po)}
        for gi, pi, _ in pm:
            g2p.setdefault(g.gt_ids[gi], set()).add(pmap[pi])
    return sum(len(v) - 1 for v in g2p.values() if len(v) > 1)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for b in (1, 2, 4, 8):
        for seq in SEQS:
            base = REAL / f"p4_budget_b{b}" / seq
            summary = json.loads((base / "summary.json").read_text())
            events = [
                json.loads(l)
                for l in (base / "events.jsonl").read_text().splitlines()
                if l.strip()
            ]
            pre = read_mot_rows(base / "pre_mot" / f"{seq}.txt")
            post = read_mot_rows(base / "post_mot" / f"{seq}.txt")
            all_post_ids = {t for f in post for t, _ in post[f]}
            alloc_ids = sorted(t for t in all_post_ids if t >= 1000)
            raw_ids = sorted(t for t in all_post_ids if t < 1000)
            # allocator ids anchored (confirmed users) vs raw (unconfirmed autos)
            n_alloc_rows = sum(1 for f in post for t, _ in post[f] if t >= 1000)
            n_raw_rows = sum(1 for f in post for t, _ in post[f] if t < 1000)
            rows.append(
                {
                    "budget": b,
                    "sequence": seq,
                    "accepted": summary["accepted_commands"],
                    "events": len(events),
                    "event_frame": events[0]["frame"] if events else None,
                    "event_type": events[0]["action_type"] if events else None,
                    "unique_post_ids": len(all_post_ids),
                    "allocator_ids": len(alloc_ids),
                    "raw_auto_ids": len(raw_ids),
                    "post_rows_allocator": n_alloc_rows,
                    "post_rows_raw": n_raw_rows,
                    "gt_public_switch_proxy": gt_public_switches(seq, post),
                    "summary_HOTA": summary.get("num_frames", None),
                }
            )
    with (OUT / "n6_budget_event_audit.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    agg = {}
    for b in (1, 2, 4, 8):
        sub = [r for r in rows if r["budget"] == b]
        agg[b] = {
            "mean_unique_post_ids": float(np.mean([r["unique_post_ids"] for r in sub])),
            "mean_allocator_ids": float(np.mean([r["allocator_ids"] for r in sub])),
            "mean_raw_auto_ids": float(np.mean([r["raw_auto_ids"] for r in sub])),
            "mean_post_rows_allocator": float(np.mean([r["post_rows_allocator"] for r in sub])),
            "mean_post_rows_raw": float(np.mean([r["post_rows_raw"] for r in sub])),
            "mean_gt_public_switch_proxy": float(
                np.mean([r["gt_public_switch_proxy"] for r in sub])
            ),
        }
    (OUT / "n6_budget_event_audit_summary.json").write_text(
        json.dumps(agg, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(agg, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
