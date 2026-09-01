#!/usr/bin/env python
"""Merge N18.1 Oracle Reactivation shards and aggregate retention.

Inputs : outputs/n18/oracle_reactivation_s{0..3}.csv
         outputs/n18/oracle_reactivation_events_s{0..3}.csv
Outputs: outputs/n18/oracle_reactivation.csv
         outputs/n18/oracle_reactivation_events.csv
         outputs/n18/reactivation_retention.csv   (overall horizons)
         outputs/n18/oracle_reactivation_per_seq.csv
"""

import csv
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(".")
OUT = ROOT / "outputs/n18"
HORIZONS = (1, 3, 5, 10, 30, 60)


def read_rows(paths):
    rows = []
    for p in paths:
        if not p.exists():
            print(f"MISSING {p}", file=sys.stderr)
            continue
        with p.open(newline="", encoding="utf-8") as f:
            rows.extend(list(csv.DictReader(f)))
    return rows


def mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else float("nan")


def main():
    shard_rows = read_rows(sorted(OUT.glob("oracle_reactivation_s*.csv")))
    shard_events = read_rows(sorted(OUT.glob("oracle_reactivation_events_s*.csv")))
    if not shard_rows:
        print("NO SHARD ROWS FOUND", file=sys.stderr)
        sys.exit(2)

    # dedupe by (sequence,t,gid)
    seen = set()
    rows = []
    for r in shard_rows:
        key = (r["sequence"], r["t"], r["gid"])
        if key in seen:
            continue
        seen.add(key)
        rows.append(r)
    seen = set()
    events = []
    for r in shard_events:
        key = (r["sequence"], r["t"], r["gid"])
        if key in seen:
            continue
        seen.add(key)
        events.append(r)

    with (OUT / "oracle_reactivation.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    if events:
        with (OUT / "oracle_reactivation_events.csv").open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(events[0].keys()))
            w.writeheader()
            w.writerows(events)

    # overall horizon retention
    overall = []
    for h in HORIZONS:
        sub = [r for r in rows if r.get(f"gt_present_{h}") == "1"]
        a0 = mean([float(r[f"a0_{h}"]) for r in sub])
        rc = mean([float(r[f"react_{h}"]) for r in sub])
        gain = rc - a0 if sub else float("nan")
        overall.append({
            "horizon": h, "n": len(sub), "A0_retention": round(a0, 4),
            "reactivation_retention": round(rc, 4),
            "gain": round(gain, 4),
        })
    with (OUT / "reactivation_retention.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(overall[0].keys()))
        w.writeheader()
        w.writerows(overall)

    # per-sequence horizon-60 retention + reactivation success at h=1
    per_seq = defaultdict(lambda: {"n": 0, "a0_60": [], "react_60": [],
                                   "success_1": [], "react_30": []})
    for r in rows:
        ps = per_seq[r["sequence"]]
        ps["n"] += 1
        if r.get("gt_present_60") == "1":
            ps["a0_60"].append(float(r["a0_60"]))
            ps["react_60"].append(float(r["react_60"]))
        if r.get("gt_present_1") == "1":
            ps["success_1"].append(float(r["react_1"]))
        if r.get("gt_present_30") == "1":
            ps["react_30"].append(float(r["react_30"]))
    with (OUT / "oracle_reactivation_per_seq.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["sequence", "n", "A0_retention_60", "react_retention_60",
                    "react_success_1", "react_retention_30"])
        for seq in sorted(per_seq):
            ps = per_seq[seq]
            w.writerow([seq, ps["n"], round(mean(ps["a0_60"]), 4),
                        round(mean(ps["react_60"]), 4),
                        round(mean(ps["success_1"]), 4),
                        round(mean(ps["react_30"]), 4)])

    print(f"events={len(rows)}")
    for o in overall:
        print(o)


if __name__ == "__main__":
    main()
