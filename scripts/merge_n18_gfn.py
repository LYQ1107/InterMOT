#!/usr/bin/env python
"""Merge GFN HCRED shards and report N18.3 recovery metrics."""

import csv
import argparse
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(".")
OUT = ROOT / "outputs/n18"


def load(paths):
    rows = []
    for p in paths:
        if p.exists():
            with p.open(newline="", encoding="utf-8") as f:
                rows.extend(list(csv.DictReader(f)))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix", default="gfn_hcred")
    ap.add_argument("--method", default="GFN-CUHK-ConvNeXtB")
    args = ap.parse_args()
    shards = load(sorted(OUT.glob(f"{args.prefix}_s*.csv")))
    if not shards:
        print("no shards", file=sys.stderr)
        sys.exit(2)
    # attach the source generic_miss flag
    src = {}
    with (OUT.parent / "n17/cal_episodes.csv").open(newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            src[(r["sequence"], r["t"], r["gid"], r["f"])] = r
    for r in shards:
        key = (r["sequence"], r["t"], r["gid"], r["f"])
        r["generic_miss"] = src.get(key, {}).get("generic_miss", "")

    with (OUT / f"hcred_recovery_{args.prefix}.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(shards[0].keys()))
        w.writeheader()
        w.writerows(shards)

    present = [r for r in shards if r["present"] == "1"]
    miss = [r for r in present if r["generic_miss"] == "1"]
    absent = [r for r in shards if r["present"] == "0"]
    gm_ccr = np.mean([int(r["top3"]) for r in miss]) if miss else 0
    gm_top1 = np.mean([int(r["top1"]) for r in miss]) if miss else 0
    gm_r03 = np.mean([float(r["best_iou"]) >= 0.3 for r in miss]) if miss else 0
    gm_r07 = np.mean([float(r["best_iou"]) >= 0.7 for r in miss]) if miss else 0
    gm_gr = np.mean([int(r["generic_rescue"]) for r in miss]) if miss else 0
    novel = sum(1 for r in miss if int(r["top3"]) and not int(r["generic_rescue"]))
    novel_rate = novel / len(miss) if miss else 0
    ghost = np.mean([float(r["top1_sim"]) >= 0.6 for r in absent
                     if r["top1_sim"] not in ("", "nan")]) if absent else 0
    summary = {
        "method": args.method, "n_episodes": len(shards),
        "n_generic_miss": len(miss), "n_absent": len(absent),
        "ccr_05_top3": round(float(gm_ccr), 4),
        "top1_recall_05": round(float(gm_top1), 4),
        "recall_03_best": round(float(gm_r03), 4),
        "recall_07_best": round(float(gm_r07), 4),
        "generic_rescue_rate": round(float(gm_gr), 4),
        "novel_rescue_rate": round(float(novel_rate), 4),
        "ghost_rate_sim06": round(float(ghost), 4),
        "N17_CCR_reference": 0.0147,
    }
    bench = OUT / "recovery_backbone_benchmark.csv"
    existing = []
    if bench.exists():
        with bench.open(newline="", encoding="utf-8") as f:
            existing = list(csv.DictReader(f))
    rows_out = [r for r in existing if r["method"] != args.method]
    rows_out.append(summary)
    with bench.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(summary.keys()))
        w.writeheader()
        w.writerows(rows_out)
    print(summary)

    # per-delta breakdown on generic miss
    by_delta = defaultdict(list)
    for r in miss:
        by_delta[r["delta"]].append(r)
    print("delta n CCR_top3 top1 generic_rescue")
    for d in sorted(by_delta, key=int):
        sub = by_delta[d]
        print(d, len(sub),
              round(np.mean([int(r["top3"]) for r in sub]), 3),
              round(np.mean([int(r["top1"]) for r in sub]), 3),
              round(np.mean([int(r["generic_rescue"]) for r in sub]), 3))


if __name__ == "__main__":
    main()
