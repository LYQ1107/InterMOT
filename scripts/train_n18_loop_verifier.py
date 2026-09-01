#!/usr/bin/env python
"""Sequence-disjoint verifier recalibration on the actual FULL_LOOP_V0
recovery-candidate distribution.

Labels come from offline GT evaluation of already-recorded causal attempts;
the model only sees causal candidate features.
"""

import csv
import json
from collections import defaultdict
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold

ROOT = Path(".")
OUT = ROOT / "outputs/n18"


def iou(a, b):
    a = np.asarray(a, dtype=float).reshape(-1)
    b = np.asarray(b, dtype=float).reshape(-1)
    if a.size != 4 or b.size != 4:
        return 0.0
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    ua = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    ub = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    return inter / (ua + ub - inter + 1e-9)


def load_gt(seq):
    from sam3_intermot.datasets.dancetrack import DanceTrackDataset
    return DanceTrackDataset(
        str(Path("/path/to/dancetrack")),
        sequences=[], split="train").load_gt(seq)


def main():
    audit = []
    for p in sorted(OUT.glob("tables/gfn_full_loop_topk_audit_s*.csv")):
        audit += list(csv.DictReader(open(p)))
    tx = []
    for p in sorted(OUT.glob("reactivation_transactions_full_s*.jsonl")):
        for line in p.open(encoding="utf-8"):
            line = line.strip()
            if line:
                tx.append(json.loads(line))
    trace = []
    for line in (OUT / "tables/delivery_trace_cal10.jsonl").open(
            encoding="utf-8"):
        trace.append(json.loads(line))
    last_box = defaultdict(dict)
    for r in trace:
        if r["delivered_box"] is not None:
            last_box[(r["sequence"], r["gid"])][r["frame"]] = \
                np.asarray(r["delivered_box"], dtype=float)
    first_frame = {}
    for seq in {r["sequence"] for r in audit}:
        gt = load_gt(seq)
        for f in sorted(gt):
            gf = gt[f]
            for gid in gf.gt_ids:
                if (seq, gid) not in first_frame:
                    first_frame[(seq, gid)] = f

    rows = []
    audit_by_key = {(r["sequence"], int(r["frame"]), int(r["gid"])): r
                    for r in audit}
    for t in tx:
        key = (t["sequence"], t["frame"], t["gid"])
        r = audit_by_key.get(key)
        if r is None or r["top1_sim"] in (None, ""):
            continue
        top1_box = json.loads(r["top1_box"]) if r["top1_box"] else None
        prev = None
        lb = last_box[(t["sequence"], t["gid"])]
        for f in sorted(lb):
            if f < t["frame"]:
                prev = lb[f]
            else:
                break
        motion = iou(top1_box, prev) if top1_box is not None and prev is not None \
            else 0.0
        anchor_gap = t["frame"] - first_frame[(t["sequence"], t["gid"])]
        rows.append({
            "sequence": t["sequence"], "frame": t["frame"], "gid": t["gid"],
            "label": int(r["gt_present"] == "1" and r["top1_correct"] == "1"),
            "gfn_top1_sim": float(r["top1_sim"]),
            "gfn_margin": float(r["top1_margin"]),
            "gfn_top1_score": float(r["top1_score"]),
            "n_dets": int(r["n_dets"]),
            "anchor_gap": anchor_gap,
            "motion_iou": round(float(motion), 4),
        })
    with (OUT / "tables/verifier_loop_dataset.csv").open(
            "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print("dataset", len(rows), "positive", sum(r["label"] for r in rows))

    feats = ["gfn_top1_sim", "gfn_margin", "gfn_top1_score", "n_dets",
             "anchor_gap", "motion_iou"]
    seqs = sorted({r["sequence"] for r in rows})
    kf = KFold(n_splits=5, shuffle=True, random_state=0)
    models = {
        "lr": LogisticRegression(max_iter=2000, class_weight="balanced"),
        "gbdt": GradientBoostingClassifier(n_estimators=200, max_depth=3,
                                           learning_rate=0.05,
                                           random_state=0),
    }
    out_rows = []
    for name, model in models.items():
        oof = np.zeros(len(rows))
        for tr_idx, te_idx in kf.split(seqs):
            tr_seqs = {seqs[i] for i in tr_idx}
            tr = [r for r in rows if r["sequence"] in tr_seqs]
            te = [r for r in rows if r["sequence"] not in tr_seqs]
            X = np.asarray([[r[k] for k in feats] for r in tr], dtype=float)
            y = np.asarray([r["label"] for r in tr])
            model.fit(X, y)
            Xt = np.asarray([[r[k] for k in feats] for r in te], dtype=float)
            for i, r in enumerate(te):
                oof[rows.index(r)] = model.predict_proba(Xt[i:i + 1])[0, 1]
        y = np.asarray([r["label"] for r in rows])
        auc = roc_auc_score(y, oof)
        for th in (0.3, 0.4, 0.5, 0.6, 0.7):
            pred = oof >= th
            tp = ((pred == 1) & (y == 1)).sum()
            fp = ((pred == 1) & (y == 0)).sum()
            fn = ((pred == 0) & (y == 1)).sum()
            out_rows.append({
                "model": name, "threshold": th, "auc": round(auc, 4),
                "accepts": int(pred.sum()),
                "precision": round(tp / max(1, tp + fp), 4),
                "recall": round(tp / max(1, tp + fn), 4),
                "false_accept_rate": round(fp / max(1, (y == 0).sum()), 4),
            })
        print(name, "AUC", round(auc, 4))
    with (OUT / "tables/verifier_loop_cv.csv").open(
            "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
        w.writeheader()
        w.writerows(out_rows)
    print("wrote verifier_loop_dataset.csv and verifier_loop_cv.csv")


if __name__ == "__main__":
    main()
