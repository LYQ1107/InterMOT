#!/usr/bin/env python
"""Stratified sampling of attempts for the all-candidate shadow dataset."""

import argparse
import csv
from collections import defaultdict
from pathlib import Path

ROOT = Path(".")
N20 = ROOT / "outputs/n20"


def sample_rows(rows, n, seed=0):
    import random
    rng = random.Random(seed)
    by_seq = defaultdict(list)
    for r in rows:
        by_seq[r["sequence"]].append(r)
    # round-robin across sequences to preserve spread
    out = []
    seqs = list(by_seq)
    rng.shuffle(seqs)
    idx = {s: 0 for s in seqs}
    while len(out) < n:
        progressed = False
        for s in seqs:
            if len(out) >= n:
                break
            if idx[s] < len(by_seq[s]):
                out.append(by_seq[s][idx[s]])
                idx[s] += 1
                progressed = True
        if not progressed:
            break
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cal-present", type=int, default=1200)
    ap.add_argument("--cal-absent", type=int, default=500)
    ap.add_argument("--train-present", type=int, default=1000)
    ap.add_argument("--train-absent", type=int, default=500)
    args = ap.parse_args()
    cal = list(csv.DictReader(
        (N20 / "topk_no_commit.csv").open(newline="", encoding="utf-8")))
    tr = list(csv.DictReader(
        (N20 / "topk_train30.csv").open(newline="", encoding="utf-8")))

    def split(rows):
        present = [r for r in rows if r["target_present"] == "1"]
        absent = [r for r in rows if r["target_present"] == "0"]
        return present, absent

    cal_p, cal_a = split(cal)
    tr_p, tr_a = split(tr)
    cal_p_s = sample_rows(cal_p, min(args.cal_present, len(cal_p)), seed=0)
    cal_a_s = sample_rows(cal_a, min(args.cal_absent, len(cal_a)), seed=1)
    tr_p_s = sample_rows(tr_p, min(args.train_present, len(tr_p)), seed=2)
    tr_a_s = sample_rows(tr_a, min(args.train_absent, len(tr_a)), seed=3)
    for name, rows in (("cal10", cal_p_s + cal_a_s),
                       ("train30", tr_p_s + tr_a_s)):
        out = N20 / f"dataset_attempts_{name}.csv"
        with out.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(name, "rows", len(rows), "present", sum(
            1 for r in rows if r["target_present"] == "1"), "->", out)


if __name__ == "__main__":
    main()
