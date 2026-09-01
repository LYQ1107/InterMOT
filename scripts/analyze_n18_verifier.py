#!/usr/bin/env python
"""N18.7 verifier baseline: cosine threshold on GFN recovery candidates.

Candidate = top-1 ranked GFN detection. True accept = top1 IoU>=0.5 on a
present episode; false accept = any accept on an absent episode.
"""

import csv
import argparse
from pathlib import Path

import numpy as np

ROOT = Path(".")
OUT = ROOT / "outputs/n18"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix", default="gfn_hcred")
    args = ap.parse_args()
    with (OUT / f"hcred_recovery_{args.prefix}.csv").open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    present = [r for r in rows if r["present"] == "1"]
    absent = [r for r in rows if r["present"] == "0"]
    for r in present + absent:
        r["sim"] = float(r["top1_sim"]) if r["top1_sim"] not in ("", "nan") else float("nan")
    correct = [r for r in present if int(r["top1"]) == 1]
    n_correct = len(correct)
    n_absent = len([r for r in absent if not np.isnan(r["sim"])])
    print(f"present={len(present)} top1_correct={n_correct} absent={n_absent}")
    out = []
    for t in [0.3, 0.4, 0.5, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85]:
        acc_p = sum(1 for r in present if r["sim"] >= t)
        tp = sum(1 for r in correct if r["sim"] >= t)
        fp_abs = sum(1 for r in absent if not np.isnan(r["sim"]) and r["sim"] >= t)
        precision = tp / acc_p if acc_p else float("nan")
        recall = tp / n_correct if n_correct else float("nan")
        frate = fp_abs / n_absent if n_absent else float("nan")
        out.append({"threshold": t, "accepts_present": acc_p,
                    "true_accept": tp, "false_accept_absent": fp_abs,
                    "precision": round(precision, 4), "recall": round(recall, 4),
                    "false_reactivation_rate": round(frate, 4)})
        print(out[-1])
    with (OUT / f"verifier_metrics_{args.prefix}.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0].keys()))
        w.writeheader()
        w.writerows(out)


if __name__ == "__main__":
    main()
