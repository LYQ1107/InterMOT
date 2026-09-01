#!/usr/bin/env python
"""N20.7: feature quality / separability for the shadow dataset."""

import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

ROOT = Path(".")
N20 = ROOT / "outputs/n20"

NUMERIC = [
    "rank_mem", "gfn_sim_human_root", "r0_sim_human_root",
    "gfn_sim_mem_last", "gfn_sim_mem_max", "r0_sim_mem_last",
    "r0_sim_mem_max", "mem_age", "n_mem_slots", "temp_sim_prev",
    "temp_sim_first", "box_area", "area_change", "center_delta",
    "velocity", "temporal_iou", "consecutive_delivered",
    "shadow_delivered", "n_dets", "gfn_margin_h", "candidate_age",
    "memory_fresh",
]


def to_float(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return np.nan


def main():
    path = N20 / "features" / "shadow_tracklets_cal10.csv"
    rows = list(csv.DictReader(path.open(newline="", encoding="utf-8")))
    print(f"rows={len(rows)}", flush=True)
    quality = []
    separability = []
    seq_means = defaultdict(lambda: defaultdict(list))
    for col in NUMERIC:
        vals = np.asarray([to_float(r[col]) for r in rows])
        miss = float(np.isnan(vals).mean())
        quality.append({"feature": col, "missing_rate": round(miss, 4),
                        "mean": round(float(np.nanmean(vals)), 4),
                        "std": round(float(np.nanstd(vals)), 4),
                        "min": round(float(np.nanmin(vals)), 4),
                        "max": round(float(np.nanmax(vals)), 4)})
    for r in rows:
        for col in NUMERIC:
            v = to_float(r[col])
            if not np.isnan(v):
                seq_means[r["sequence"]][col].append(v)
    drift = {}
    for col in NUMERIC:
        ms = [float(np.mean(seq_means[s][col]))
              for s in seq_means if seq_means[s][col]]
        drift[col] = float(np.std(ms)) if len(ms) > 1 else 0.0
    # single-feature AUC vs label_correct at final step
    for step in (1, 3, 5, 8):
        sub = [r for r in rows if int(r["evidence_step"]) == step]
        y = np.asarray([int(r["label_correct"]) for r in sub])
        for col in NUMERIC:
            x = np.asarray([to_float(r[col]) for r in sub])
            mask = ~np.isnan(x)
            if mask.sum() < 30 or len(set(y[mask])) < 2:
                auc = np.nan
            else:
                auc = roc_auc_score(y[mask], x[mask])
            separability.append({"evidence_step": step, "feature": col,
                                "auc": round(float(auc), 4),
                                "cross_seq_drift": round(drift[col], 4)})
    with (N20 / "feature_quality.csv").open("w", newline="",
                                            encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(quality[0].keys()))
        w.writeheader()
        w.writerows(quality)
    with (N20 / "feature_separability.csv").open("w", newline="",
                                                 encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(separability[0].keys()))
        w.writeheader()
        w.writerows(separability)
    # summary json
    top5 = sorted((s for s in separability if s["evidence_step"] == 5),
                  key=lambda s: -s["auc"])[:12]
    top_missing = sorted(quality, key=lambda q: -q["missing_rate"])[:8]
    top_drift = sorted(separability, key=lambda s: -s["cross_seq_drift"])[:8]
    summary = {
        "n_rows": len(rows),
        "n_attempts": len({r["attempt"] for r in rows}),
        "label_correct_at_5": round(sum(
            1 for r in rows if int(r["evidence_step"]) == 5 and
            int(r["label_correct"]) == 1) / max(1, sum(
                1 for r in rows if int(r["evidence_step"]) == 5)), 4),
        "top_auc_features": top5,
        "top_missing": top_missing,
        "top_drift": top_drift,
    }
    (N20 / "shadow_feature_analysis.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    print("FEATURE_ANALYSIS_DONE", flush=True)


if __name__ == "__main__":
    main()
