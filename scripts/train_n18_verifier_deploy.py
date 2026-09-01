#!/usr/bin/env python
"""Fit the deployable N18 verifier (GFN-only features, single torch env).

PSTR requires the separate legacy env, so the online loop uses GFN features
only. Reports 5-fold sequence-disjoint OOF AUC for reference, then fits on all
calibration rows and saves the model for the FULL_LOOP_V0 runner.
"""

import csv
import glob
import joblib
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

ROOT = Path(".")
OUT = ROOT / "outputs/n18"
FEATS = ["gfn_top1_sim", "gfn_margin", "gfn_top1_score", "n_dets"]


def load():
    rows = []
    for p in sorted(glob.glob(str(OUT / "verifier_features_s*.csv"))):
        with open(p, newline="", encoding="utf-8") as f:
            rows += list(csv.DictReader(f))
    return rows


def fv(r):
    return [float(r.get(k)) if str(r.get(k, "")) not in ("", "nan")
            else 0.0 for k in FEATS]


def main():
    rows = load()
    X = np.stack([fv(r) for r in rows])
    y = np.asarray([int(r["present"]) and int(r["top1_correct"])
                    for r in rows], dtype=int)
    seqs = sorted({r["sequence"] for r in rows})
    folds = {s: i % 5 for i, s in enumerate(seqs)}
    preds, labels = [], []
    for fold in range(5):
        tr = [i for i, r in enumerate(rows) if folds[r["sequence"]] != fold]
        te = [i for i, r in enumerate(rows) if folds[r["sequence"]] == fold]
        clf = LogisticRegression(max_iter=2000).fit(X[tr], y[tr])
        p = clf.predict_proba(X[te])[:, 1]
        preds += list(p)
        labels += list(y[te])
    auc = roc_auc_score(labels, preds)
    print(f"GFN-only OOF AUC = {auc:.4f} (5-fold sequence-disjoint)")
    clf = LogisticRegression(max_iter=2000).fit(X, y)
    (OUT / "models").mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": clf, "features": FEATS,
                 "threshold": 0.6, "oof_auc": float(auc)},
                OUT / "models/verifier_v0.joblib")
    print("saved", OUT / "models/verifier_v0.joblib")
    for k, c in zip(FEATS, clf.coef_[0]):
        print(" ", k, round(float(c), 4))
    print(" intercept", round(float(clf.intercept_[0]), 4))


if __name__ == "__main__":
    main()
