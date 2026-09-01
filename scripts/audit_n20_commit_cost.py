#!/usr/bin/env python
"""N20.14: empirical downstream cost of correct vs false shadow commits."""

import csv
import glob as globmod
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(".")
N18 = ROOT / "outputs/n18"
N19 = ROOT / "outputs/n19"
N20 = ROOT / "outputs/n20"
OUT = N20 / "full_loop_oracle_shadow"


def load_events(prefix, root=OUT):
    ev = []
    for p in map(Path, sorted(globmod.glob(str(root / f"{prefix}*.jsonl")))):
        for line in p.open(encoding="utf-8"):
            if line.strip():
                ev.append(json.loads(line))
    return ev


def load_tx(prefix, root=OUT):
    tx = []
    for p in map(Path, sorted(globmod.glob(str(root / f"{prefix}*.jsonl")))):
        for line in p.open(encoding="utf-8"):
            if line.strip():
                tx.append(json.loads(line))
    return tx


def audit(prefix, root, tx_prefix=None, skip_shadow_verdict=False):
    ev = load_events(prefix, root)
    tx = load_tx(tx_prefix if tx_prefix is not None else prefix, root)
    ev_by = defaultdict(lambda: defaultdict(dict))
    for e in ev:
        ev_by[(e["sequence"], e["gid"])][int(e["frame"])] = e
    accepts = [t for t in tx if t.get("reactivated")]
    if skip_shadow_verdict:
        accepts = [t for t in accepts if not t.get("shadow_commit")]
    rows = []
    for t in accepts:
        seq, gid = t["sequence"], t["gid"]
        f0 = int(t.get("commit_frame", t["frame"]))
        base = ev_by[(seq, gid)]
        corr = []
        for h in (1, 3, 5, 10, 30, 60, 120):
            e = base.get(f0 + h)
            if e is None:
                corr.append(None)
            else:
                corr.append(int(e.get("correct", 0)) if e.get(
                    "gt_present") else None)
        # correctness at commit frame
        e0 = base.get(f0)
        commit_ok = int(e0.get("correct", 0)) if e0 is not None else None
        rows.append({
            "prefix": prefix, "sequence": seq, "frame": f0, "gid": gid,
            "commit_ok": commit_ok,
            "h1": corr[0], "h3": corr[1], "h5": corr[2], "h10": corr[3],
            "h30": corr[4], "h60": corr[5], "h120": corr[6],
        })
    return rows


def main():
    all_rows = []
    all_rows += audit("full_loop_v0_events_full_s", N18,
                      tx_prefix="reactivation_transactions_full_s")
    all_rows += audit("full_loop_v0_events_oracle_n19_s",
                      N19 / "oracle_full_loop",
                      tx_prefix="reactivation_transactions_oracle_n19_s")
    for v in ("k5_h0", "k5_h1", "k5_h5", "n20_gru_h5"):
        all_rows += audit(f"events_{v}", OUT, tx_prefix=f"transactions_{v}")
    with (N20 / "downstream_commit_cost.csv").open(
            "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
        w.writeheader()
        w.writerows(all_rows)
    # aggregate: correct vs false commits
    for prefix in ("full_loop_v0_events_full_s",
                   "full_loop_v0_events_oracle_n19_s",
                   "events_k5_h0", "events_k5_h1", "events_k5_h5",
                   "events_n20_gru_h5"):
        rows = [r for r in all_rows if r["prefix"] == prefix]
        ok = [r for r in rows if r["commit_ok"] == 1]
        bad = [r for r in rows if r["commit_ok"] == 0]
        def means(rs):
            out = {}
            for h in ("h1", "h3", "h5", "h10", "h30", "h60", "h120"):
                vals = [r[h] for r in rs if r[h] is not None]
                out[h] = round(sum(vals) / len(vals), 3) if vals else None
            return out
        print(prefix, "n", len(rows), "ok", len(ok), "bad", len(bad))
        print("  ok retention:", means(ok))
        print("  bad retention:", means(bad))
    print("COMMIT_COST_DONE", flush=True)


if __name__ == "__main__":
    main()
