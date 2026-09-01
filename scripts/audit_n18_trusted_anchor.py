#!/usr/bin/env python
"""CPU replay of the causal trusted-memory anchor without GPU components.

Evaluates whether the V1 M_i update rule actually produces a fresh and
correct query anchor at each recovery attempt, versus the deployed
first-appearance H_i anchor and the offline GT last-seen upper bound.
"""

import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(".")
sys.path.insert(0, str(ROOT))

from sam3_intermot.evaluation.full_loop_v0 import LoopConfig, P0Row, iou, \
    run_full_loop
from scripts.run_n18_full_loop_v0 import load_gt, load_p0

OUT = ROOT / "outputs/n18"


def main():
    cal10 = json.loads(
        (ROOT / "outputs/n15/n15_frozen.json").read_text())["split"][
            "calibration10"]
    variants = {
        "first": LoopConfig(anchor_policy="first"),
        "trusted_score0.5": LoopConfig(anchor_policy="trusted",
                                       anchor_score=0.5,
                                       anchor_continuity_iou=0.5),
        "trusted_score0.0": LoopConfig(anchor_policy="trusted",
                                       anchor_score=0.0,
                                       anchor_continuity_iou=0.5),
    }
    rows = []
    for seq in cal10:
        gt = load_gt(seq)
        p0 = load_p0(seq)
        n = max(gt) + 1
        for name, cfg in variants.items():
            records = []

            def recover(s, f, box, af):
                records.append((f, af, np.asarray(box, dtype=float).copy()))
                return None

            def verify(rec):
                return 0.0

            def reactivate(s, f, box):
                return {}

            run_full_loop(seq, gt, p0, n, cfg, recover, verify, reactivate)
            for f, af, box in records:
                gf = gt.get(af)
                target = None
                if gf is not None:
                    gid = None
                    for k, b in enumerate(gf.boxes):
                        if iou(b, box) >= 0.5:
                            gid = gf.gt_ids[k]
                            target = np.asarray(b, dtype=float)
                            break
                rows.append({
                    "variant": name, "sequence": seq, "frame": f,
                    "anchor_frame": af, "anchor_gap": f - af,
                    "anchor_box": json.dumps([round(float(x), 1)
                                              for x in box]),
                    "anchor_correct": int(target is not None),
                })
    out = OUT / "tables" / "trusted_anchor_replay.csv"
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=[
            "variant", "sequence", "frame", "anchor_frame", "anchor_gap",
            "anchor_box", "anchor_correct"])
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {out} ({len(rows)} attempts)")
    for name in variants:
        sel = [r for r in rows if r["variant"] == name]
        corr = np.mean([r["anchor_correct"] for r in sel])
        gap = np.mean([r["anchor_gap"] for r in sel])
        print(name, "n", len(sel), "anchor_correct", round(corr, 4),
              "mean_gap", round(gap, 2))


if __name__ == "__main__":
    main()
