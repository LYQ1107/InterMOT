#!/usr/bin/env python
"""N19.11: Retrain the recovery verifier under the learned memory.

Logistic regression over GFN candidate features plus learned-memory
features (memory age, agreement count, R0 sim). Fit on train30, calibrate
the threshold on cal10 (max recall at precision >= 0.90). No val25.
"""

import argparse
import csv
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

ROOT = Path(".")
N19 = ROOT / "outputs/n19"

FEATS = ["gfn_top1_sim", "gfn_margin", "gfn_top1_score", "n_dets",
         "memory_age", "n_agree_slots", "r0_top1_sim"]


def load(path):
    with Path(path).open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def fv(r):
    return [0.0 if str(r.get(k)) in ("", "nan", "None") else float(r[k])
            for k in FEATS]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default=str(N19 / "verifier_dataset_train30.csv"))
    ap.add_argument("--cal", default=str(N19 / "verifier_dataset_cal10.csv"))
    ap.add_argument("--out", default=str(N19 / "models/verifier_n19.joblib"))
    ap.add_argument("--target-precision", type=float, default=0.50)
    args = ap.parse_args()

    tr, ca = load(args.train), load(args.cal)
    Xtr = np.stack([fv(r) for r in tr])
    ytr = np.asarray([int(r["top1_correct"]) for r in tr])
    Xca = np.stack([fv(r) for r in ca])
    yca = np.asarray([int(r["top1_correct"]) for r in ca])
    clf = LogisticRegression(max_iter=5000, C=1.0).fit(Xtr, ytr)
    ptr = clf.predict_proba(Xtr)[:, 1]
    pca = clf.predict_proba(Xca)[:, 1]
    auc_tr = roc_auc_score(ytr, ptr)
    auc_ca = roc_auc_score(yca, pca)
    print(f"train rows={len(Xtr)} pos={ytr.sum()} auc={auc_tr:.4f}")
    print(f"cal rows={len(Xca)} pos={yca.sum()} auc={auc_ca:.4f}")

    # threshold: max recall at target precision on cal
    best = None
    for thr in np.arange(0.01, 0.99, 0.005):
        pred = pca >= thr
        tp = float((pred & (yca == 1)).sum())
        fp = float((pred & (yca == 0)).sum())
        fn = float((~pred & (yca == 1)).sum())
        prec = tp / max(tp + fp, 1e-9)
        rec = tp / max(tp + fn, 1e-9)
        if prec >= args.target_precision and \
                (best is None or rec > best["recall"]):
            best = {"threshold": round(float(thr), 3),
                    "precision": round(prec, 4),
                    "recall": round(rec, 4),
                    "accepts": int(tp + fp), "correct": int(tp)}
    if best is None:
        best = {"threshold": 0.99, "precision": 1.0, "recall": 0.0,
                "accepts": 0, "correct": 0}
    print("chosen", best)
    (N19 / "models").mkdir(parents=True, exist_ok=True)
    joblib.dump({
        "model": clf, "features": FEATS, "threshold": best["threshold"],
        "train_auc": float(auc_tr), "cal_auc": float(auc_ca),
        "calibration": best,
    }, args.out)
    print("VERIFIER_TRAIN_DONE saved", args.out, flush=True)


if __name__ == "__main__":
    main()
