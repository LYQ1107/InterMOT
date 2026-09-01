#!/usr/bin/env python
"""CPU diagnostic: how well do causal delivery score/continuity separate
correct from wrong P0 deliveries in the V0 loop?"""

import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(".")
sys.path.insert(0, str(ROOT))

from sam3_intermot.evaluation.full_loop_v0 import LoopConfig, run_full_loop
from scripts.run_n18_full_loop_v0 import load_gt, load_p0


def main():
    cal10 = json.loads(
        (ROOT / "outputs/n15/n15_frozen.json").read_text())["split"][
            "calibration10"]
    cfg = LoopConfig(anchor_policy="first")
    rows = []
    for seq in cal10:
        gt = load_gt(seq)
        p0 = load_p0(seq)
        n = max(gt) + 1
        result = run_full_loop(
            seq, gt, p0, n, cfg,
            lambda s, f, b, af: None, lambda r: 0.0,
            lambda s, f, b: {})
        for e in result["trace"]:
            if e["source"] in ("p0", "p0_tid") and e["gt_present"]:
                rows.append({
                    "sequence": seq, "frame": e["frame"], "gid": e["gid"],
                    "correct": e["correct"],
                    "score": e["delivery_score"],
                    "prev_iou": e["delivery_iou_prev"],
                })
    out = ROOT / "outputs/n18/tables/delivery_gate_features.csv"
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    corr = [r for r in rows if r["correct"] == 1]
    wrong = [r for r in rows if r["correct"] == 0]

    def stats(name, rs, key):
        vals = [r[key] for r in rs if r[key] is not None]
        if not vals:
            return
        print(name, key, "n", len(vals), "mean", round(float(np.mean(vals)), 3),
              "q25", round(float(np.percentile(vals, 25)), 3),
              "q50", round(float(np.percentile(vals, 50)), 3),
              "q75", round(float(np.percentile(vals, 75)), 3))

    print("total", len(rows), "correct", len(corr), "wrong", len(wrong))
    for key in ("score", "prev_iou"):
        stats("correct", corr, key)
        stats("wrong", wrong, key)
    # simple gates
    for key, ths in (("score", (0.3, 0.5, 0.7, 0.9)),
                     ("prev_iou", (0.1, 0.3, 0.5, 0.7))):
        for th in ths:
            sel = [r for r in rows if r[key] is not None and r[key] >= th]
            prec = np.mean([r["correct"] for r in sel]) if sel else None
            print(f"{key}>={th}: n={len(sel)} precision={prec}")


if __name__ == "__main__":
    main()
