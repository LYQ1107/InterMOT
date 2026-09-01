#!/usr/bin/env python
"""N18.1 Oracle Reactivation analysis: retention, gains, TTE, bootstrap CI."""

import csv
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(".")
OUT = ROOT / "outputs/n18"
HORIZONS = (1, 3, 5, 10, 30, 60)
RNG = np.random.default_rng(18)
N_BOOT = 2000


def load():
    p = OUT / "oracle_reactivation.csv"
    if not p.exists():
        print("run aggregate_n18_oracle.py first", file=sys.stderr)
        sys.exit(2)
    with p.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def main():
    rows = load()
    print(f"events={len(rows)}")

    # 1) overall retention + cluster-bootstrap CI on the gain
    seqs = sorted({r["sequence"] for r in rows})
    by_seq = defaultdict(list)
    for r in rows:
        by_seq[r["sequence"]].append(r)
    print("\nhorizon n A0 react gain [95% CI cluster-bootstrap]")
    for h in HORIZONS:
        sub = [r for r in rows if r.get(f"gt_present_{h}") == "1"]
        if not sub:
            continue
        a0 = np.mean([float(r[f"a0_{h}"]) for r in sub])
        rc = np.mean([float(r[f"react_{h}"]) for r in sub])
        # cluster bootstrap over sequences
        gains = []
        for _ in range(N_BOOT):
            sample_seqs = RNG.choice(seqs, size=len(seqs), replace=True)
            ss = [r for s in sample_seqs for r in by_seq[s]
                  if r.get(f"gt_present_{h}") == "1"]
            if not ss:
                continue
            ga = np.mean([float(r[f"a0_{h}"]) for r in ss])
            gr = np.mean([float(r[f"react_{h}"]) for r in ss])
            gains.append(gr - ga)
        lo, hi = np.percentile(gains, [2.5, 97.5])
        print(f"h={h:2d} n={len(sub):3d} A0={a0:.3f} react={rc:.3f} "
              f"gain={rc-a0:+.3f} CI=[{lo:+.3f},{hi:+.3f}]")

    # 2) run length / TTE proxy at horizon granularity
    def run_len(r, side):
        run = 0
        for h in HORIZONS:
            if r.get(f"gt_present_{h}") != "1":
                continue
            if int(r[f"{side}_{h}"]) == 1:
                run = h
            else:
                break
        return run

    a0_run = np.array([run_len(r, "a0") for r in rows], dtype=float)
    rc_run = np.array([run_len(r, "react") for r in rows], dtype=float)
    print(f"\nerror-free run (horizon-granularity): A0 mean={a0_run.mean():.1f} "
          f"react mean={rc_run.mean():.1f}")

    # 3) event classes
    n_win = n_lose = n_tie = 0
    for r in rows:
        ga = sum(int(r[f"a0_{h}"]) for h in HORIZONS if r.get(f"gt_present_{h}") == "1")
        gr = sum(int(r[f"react_{h}"]) for h in HORIZONS if r.get(f"gt_present_{h}") == "1")
        n_win += gr > ga
        n_lose += gr < ga
        n_tie += gr == ga
    print(f"events with react>A0={n_win} react<A0={n_lose} tie={n_tie}")

    # short-horizon extension: react@1..10 strictly better than A0@1..10
    ext = 0
    for r in rows:
        short = [h for h in (1, 3, 5, 10) if r.get(f"gt_present_{h}") == "1"]
        if not short:
            continue
        ga = sum(int(r[f"a0_{h}"]) for h in short)
        gr = sum(int(r[f"react_{h}"]) for h in short)
        if gr > ga:
            ext += 1
    print(f"events with react>A0 on h=1..10: {ext}/{len(rows)}")

    with (OUT / "oracle_analysis.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["sequence", "t", "gid", "a0_run", "react_run",
                    "short_ext", "sum_a0", "sum_react"])
        for r in rows:
            short = [h for h in (1, 3, 5, 10) if r.get(f"gt_present_{h}") == "1"]
            ga_s = sum(int(r[f"a0_{h}"]) for h in short)
            gr_s = sum(int(r[f"react_{h}"]) for h in short)
            ga = sum(int(r[f"a0_{h}"]) for h in HORIZONS if r.get(f"gt_present_{h}") == "1")
            gr = sum(int(r[f"react_{h}"]) for h in HORIZONS if r.get(f"gt_present_{h}") == "1")
            w.writerow([r["sequence"], r["t"], r["gid"],
                        run_len(r, "a0"), run_len(r, "react"),
                        int(gr_s > ga_s), ga, gr])


if __name__ == "__main__":
    main()
