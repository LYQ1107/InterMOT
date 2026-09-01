#!/usr/bin/env python
"""N19.6: Feature separability audit for the memory-write decision.

Safe-write (GT-correct delivery) vs dangerous-write (wrong delivery) on the
causal feature set. Reports missing rate, ROC AUC, mutual information,
distribution means, and cross-sequence drift. Offline labels only.
"""

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

ROOT = Path(".")
N19 = ROOT / "outputs/n19"

FEATURES = [
    "gfn_sim_human_root", "r0_sim_human_root",
    "gfn_sim_oracle_last", "gfn_sim_oracle_max", "r0_sim_oracle_max",
    "gfn_sim_heur_last", "gfn_sim_heur_max", "gfn_margin_h",
    "det_score", "box_area", "temporal_iou", "center_delta",
    "consecutive_delivered", "missing_streak", "crowd",
    "overlap_max", "nearest_det_distance", "oracle_memory_age",
    "heur_memory_age", "candidate_age", "slots_oracle_count",
    "slots_heur_count",
]


def read_csv(path):
    with Path(path).open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def join_utility(ds, ut):
    d = {}
    for r in ut:
        d[(r["sequence"], int(r["gid"]), int(r["frame"]))] = r
    for r in ds:
        u = d.get((r["sequence"], int(r["gid"]), int(r["frame"])))
        if u:
            for k, v in u.items():
                r[f"u_{k}"] = v
    return ds


def mi(x, y, bins=20):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=int)
    if len(x) < 2:
        return 0.0
    lo, hi = np.percentile(x, [1, 99])
    if hi <= lo:
        return 0.0
    xb = np.clip(((x - lo) / (hi - lo) * bins).astype(int), 0, bins - 1)
    n = len(y)
    hx = 0.0
    hy = 0.0
    hxy = 0.0
    px = np.bincount(xb, minlength=bins) / n
    py = np.bincount(y, minlength=2) / n
    for p in px:
        if p > 0:
            hx -= p * np.log(p)
    for p in py:
        if p > 0:
            hy -= p * np.log(p)
    for b in range(bins):
        for c in range(2):
            p = float(np.sum((xb == b) & (y == c))) / n
            if p > 0:
                hxy -= p * np.log(p)
    return hx + hy - hxy


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default=str(N19 / "write_dataset_cal10.csv"))
    ap.add_argument("--utility", default=str(N19 / "future_utility_cal10.csv"))
    ap.add_argument("--out", default=str(N19 / "write_feature_analysis.csv"))
    args = ap.parse_args()

    rows = join_utility(read_csv(args.dataset), read_csv(args.utility))
    y = np.asarray([int(r["safe_write"]) for r in rows])
    by_seq = defaultdict(list)
    for r in rows:
        by_seq[r["sequence"]].append(r)

    out = []
    for f in FEATURES:
        vals = []
        labels = []
        for r in rows:
            v = r.get(f)
            if str(v) in ("", "nan", "None"):
                continue
            vals.append(float(v))
            labels.append(int(r["safe_write"]))
        if len(vals) < 50:
            out.append({"feature": f, "missing_rate": 1.0})
            continue
        vals = np.asarray(vals, dtype=float)
        labels = np.asarray(labels, dtype=int)
        auc = roc_auc_score(labels, vals)
        # cross-sequence drift: std of per-seq means
        seq_means = []
        for s, rs in by_seq.items():
            vs = [float(r[f]) for r in rs
                  if str(r.get(f)) not in ("", "nan", "None")]
            if vs:
                seq_means.append(float(np.mean(vs)))
        drift = float(np.std(seq_means)) if seq_means else None
        out.append({
            "feature": f,
            "missing_rate": round(1.0 - len(vals) / len(rows), 4),
            "auc_safe": round(auc, 4),
            "mi_safe": round(mi(vals, labels), 4),
            "mean_safe": round(float(vals[labels == 1].mean()), 4)
            if (labels == 1).any() else None,
            "mean_dangerous": round(float(vals[labels == 0].mean()), 4)
            if (labels == 0).any() else None,
            "median": round(float(np.median(vals)), 4),
            "p05": round(float(np.percentile(vals, 5)), 4),
            "p95": round(float(np.percentile(vals, 95)), 4),
            "cross_seq_mean_std": round(drift, 4) if drift is not None
            else None,
        })
        print(f"feature {f}: auc={auc:.4f} missing="
              f"{out[-1]['missing_rate']}", flush=True)

    # utility separability among safe candidates
    safe = [r for r in rows if r["safe_write"] == "1"]
    ut_cols = ["u_any_improve_10", "u_any_improve_30", "u_any_improve_60",
               "u_any_improve_120", "u_any_improve_240", "u_any_improve_480"]
    util = {}
    for c in ut_cols:
        ys = [int(r[c]) for r in safe if r.get(c) in ("0", "1")]
        util[c] = {"n": len(ys), "positive": sum(ys)} if ys else None
    (N19 / "future_utility_stats.json").write_text(
        json.dumps(util, indent=2), encoding="utf-8")

    with Path(args.out).open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0].keys()))
        w.writeheader()
        w.writerows(out)
    print(f"FEATURE_ANALYSIS_DONE rows={len(out)}", flush=True)


if __name__ == "__main__":
    main()
