#!/usr/bin/env python
"""Offline simulation of causal verified-memory anchors at several gates.

Uses the CPU delivery replay (delivered boxes), the offline GFN health audit
(verifier score and matched IoU per delivered row), and the recorded V0
recovery attempts.  For each gate threshold it walks frames in order and
updates M_i only on gated deliveries, then exports the M_i anchor present at
every recovery attempt for a subsequent GFN top-K replay.
"""

import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(".")
OUT = ROOT / "outputs/n18"


def load_gt(seq):
    from sam3_intermot.datasets.dancetrack import DanceTrackDataset
    return DanceTrackDataset(
        str(Path("/path/to/dancetrack")),
        sequences=[], split="train").load_gt(seq)


def first_box(gt, gid):
    for f in sorted(gt):
        gf = gt[f]
        if gid in gf.gt_ids:
            return f, np.asarray(
                gf.boxes[gf.gt_ids.index(gid)], dtype=float)
    return None


def main():
    trace = []
    for line in (OUT / "tables/delivery_trace_cal10.jsonl").open(
            encoding="utf-8"):
        trace.append(json.loads(line))
    health = {}
    for r in csv.DictReader(open(OUT / "tables/health_features.csv")):
        health[(r["sequence"], int(r["frame"]), int(r["gid"]))] = r
    tx = []
    for p in sorted(OUT.glob("reactivation_transactions_full_s*.jsonl")):
        for line in p.open(encoding="utf-8"):
            line = line.strip()
            if line:
                tx.append(json.loads(line))

    by_seq = defaultdict(lambda: defaultdict(list))
    for r in trace:
        if r["source"] in ("p0", "p0_tid") and r["delivered_box"] is not None:
            by_seq[r["sequence"]][r["gid"]].append(r)
    tx_by_seq = defaultdict(list)
    for t in tx:
        tx_by_seq[t["sequence"]].append(t)

    for th in (0.4, 0.5, 0.6):
        out_rows = []
        for seq, gids in by_seq.items():
            gt = load_gt(seq)
            all_gids = sorted({t["gid"] for t in tx_by_seq.get(seq, [])})
            for gid in all_gids:
                evs = gids.get(gid, [])
                evs.sort(key=lambda e: e["frame"])
                fb = first_box(gt, gid)
                if fb is None:
                    continue
                m_frame, m_box = fb
                ev_i = 0
                for t in sorted(tx_by_seq[seq], key=lambda x: x["frame"]):
                    if t["gid"] != gid:
                        continue
                    f = t["frame"]
                    while ev_i < len(evs) and evs[ev_i]["frame"] <= f:
                        e = evs[ev_i]
                        h = health.get((seq, e["frame"], gid))
                        if h is not None and h["verifier_score"] not in (
                                "", None) and h["matched_iou"] not in ("", None):
                            ver = float(h["verifier_score"])
                            mi = float(h["matched_iou"])
                            pi = float(e["delivery_iou_prev"] or 0.0)
                            if mi >= 0.3 and ver >= th and pi >= 0.5:
                                m_frame = e["frame"]
                                m_box = np.asarray(e["delivered_box"],
                                                    dtype=float)
                        ev_i += 1
                    out_rows.append({
                        "sequence": seq, "frame": f, "gid": gid,
                        "anchor_frame": m_frame,
                        "anchor_box": [round(float(x), 1) for x in m_box],
                    })
        p = OUT / "tables" / f"verified_anchors_{th}.jsonl"
        with p.open("w", encoding="utf-8") as f:
            for r in out_rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(th, "anchors", len(out_rows))


if __name__ == "__main__":
    main()
