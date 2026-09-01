#!/usr/bin/env python
"""Aggregate N16 HCC metrics into the required tables."""

import csv
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(".")
OUT = ROOT / "outputs/n16"


def main() -> None:
    cc = list(csv.DictReader((OUT / "candidate_creation.csv").open(encoding="utf-8")))
    if not cc:
        print("no candidate_creation.csv")
        return
    for r in cc:
        r["present"] = int(r["present"])
        r["generic_miss"] = int(r["generic_miss"])
        r["recall_05"] = int(r["recall_05"])
        r["ghost"] = int(r["ghost"])
        r["false_capture"] = int(r["false_capture"])
    pres = [r for r in cc if r["present"]]
    miss = [r for r in cc if r["generic_miss"]]
    abs_ = [r for r in cc if not r["present"]]
    # hard negative / distractor analysis by delta and crowd
    rows = []
    for r in pres:
        rows.append(
            {
                "backbone": "HCRD-v1", "seq": r["sequence"], "delta": r["delta"],
                "crowd": r["crowd"], "recall_05": r["recall_05"],
                "false_capture": r["false_capture"],
            }
        )
    with (OUT / "hard_negative_results.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    with (OUT / "absent_results.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["n_absent", "ghost_rate", "mean_presence"])
        w.writerow(
            [
                len(abs_),
                round(float(np.mean([r["ghost"] for r in abs_])), 4) if abs_ else "",
                round(float(np.mean([float(r["presence"]) for r in abs_])), 4) if abs_ else "",
            ]
        )
    print(
        f"n={len(cc)} present={len(pres)} miss={len(miss)} absent={len(abs_)}\n"
        f"recall05_present={np.mean([r['recall_05'] for r in pres]) if pres else 0:.4f} "
        f"CCR_miss={np.mean([r['recall_05'] for r in miss]) if miss else 0:.4f} "
        f"fc_rate={np.mean([r['false_capture'] for r in pres]) if pres else 0:.4f}"
    )


if __name__ == "__main__":
    main()

