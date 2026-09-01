#!/usr/bin/env python
"""Join future-utility labels into the write dataset and add derived labels.

GOOD_WRITE: safe identity write that improves future recovery (any top1
improvement within 60 frames in the offline replay).
SAFE_BUT_REDUNDANT: safe but no utility gain.
DANGEROUS_WRITE: identity-incorrect observation.
"""

import argparse
import csv
from pathlib import Path


def read(path):
    with Path(path).open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--utility", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--utility-horizon", type=int, default=240)
    args = ap.parse_args()

    ds = read(args.dataset)
    ut = read(args.utility)
    ud = {(r["sequence"], int(r["gid"]), int(r["frame"])): r for r in ut}
    ukeys = [k for k in ut[0].keys()
             if k not in ("sequence", "gid", "frame")]
    fields = list(ds[0].keys()) + [f"u_{k}" for k in ukeys] + [
        "good_write", "safe_but_redundant", "dangerous_write",
        "utility_class"]
    with Path(args.out).open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in ds:
            row = dict(r)
            u = ud.get((r["sequence"], int(r["gid"]), int(r["frame"])))
            if u:
                for k, v in u.items():
                    if k not in ("sequence", "gid", "frame"):
                        row[f"u_{k}"] = v
            for k in ukeys:
                row.setdefault(f"u_{k}", "")
            safe = int(row.get("safe_write", 0))
            useful = 0
            key = f"any_improve_{args.utility_horizon}"
            if u and u.get(key) in ("0", "1"):
                useful = int(u[key])
            if safe and useful:
                row["good_write"] = 1
                row["safe_but_redundant"] = 0
                row["dangerous_write"] = 0
                row["utility_class"] = "GOOD_WRITE"
            elif safe:
                row["good_write"] = 0
                row["safe_but_redundant"] = 1
                row["dangerous_write"] = 0
                row["utility_class"] = "SAFE_BUT_REDUNDANT"
            else:
                row["good_write"] = 0
                row["safe_but_redundant"] = 0
                row["dangerous_write"] = 1
                row["utility_class"] = "DANGEROUS_WRITE"
            w.writerow(row)
    print(f"MERGE_DONE out={args.out}", flush=True)


if __name__ == "__main__":
    main()
