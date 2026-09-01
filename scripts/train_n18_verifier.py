#!/usr/bin/env python
"""N18.7 small learned verifier: logistic regression, 5-fold
sequence-disjoint CV on the 10 calibration sequences.

Label: accept = 1 iff target present AND GFN top-1 localization IoU>=0.5.
Features: gfn_top1_sim, gfn_margin, gfn_top1_score, n_dets, pstr_top1_sim.
"""

import csv
import glob
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

ROOT = Path(".")
OUT = ROOT / "outputs/n18"
FEATS = ["gfn_top1_sim", "gfn_margin", "gfn_top1_score", "n_dets",
         "pstr_top1_sim"]


def load():
    rows = []
    for p in sorted(glob.glob(str(OUT / "verifier_features_s*.csv"))):
        with open(p, newline="", encoding="utf-8") as f:
            rows += list(csv.DictReader(f))
    pstr = {}
    for p in sorted(glob.glob(str(OUT / "pstr_hcred_s*.csv"))):
        with open(p, newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                pstr[(r["sequence"], r["t"], r["gid"], r["f"])] = r
    for r in rows:
        key = (r["sequence"], r["t"], r["gid"], r["f"])
        r["pstr_top1_sim"] = pstr.get(key, {}).get("top1_sim", "")
    return rows


def feat_vec(r):
    v = []
    for k in FEATS:
        s = r.get(k, "")
        v.append(float(s) if s not in ("", "nan") else 0.0)
    return np.asarray(v, dtype=float)


def main():
    rows = load()
    print("rows", len(rows))
    seqs = sorted({r["sequence"] for r in rows})
    folds = {s: i % 5 for i, s in enumerate(seqs)}
    preds, labels, pred_rows = [], [], []
    fold_rows = []
    for fold in range(5):
        tr = [r for r in rows if folds[r["sequence"]] != fold]
        te = [r for r in rows if folds[r["sequence"]] == fold]
        Xtr = np.stack([feat_vec(r) for r in tr])
        ytr = np.asarray([int(r["present"]) and int(r["top1_correct"])
                          for r in tr], dtype=int)
        Xte = np.stack([feat_vec(r) for r in te])
        yte = np.asarray([int(r["present"]) and int(r["top1_correct"])
                          for r in te], dtype=int)
        clf = LogisticRegression(max_iter=2000)
        clf.fit(Xtr, ytr)
        p = clf.predict_proba(Xte)[:, 1]
        auc = roc_auc_score(yte, p) if len(set(yte)) > 1 else float("nan")
        for r, y, s in zip(te, yte, p):
            preds.append(s)
            labels.append(y)
            pred_rows.append(r)
        fold_rows.append({"fold": fold, "n_train": len(tr), "n_test": len(te),
                          "auc": round(float(auc), 4),
                          "pos_rate": round(float(yte.mean()), 4)})
        print(fold_rows[-1])
    labels = np.asarray(labels)
    preds = np.asarray(preds)
    auc = roc_auc_score(labels, preds) if len(set(labels)) > 1 else float("nan")
    print("OOF AUC", round(float(auc), 4), "pos", round(float(labels.mean()), 4))
    out = []
    for t in (0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8):
        acc = preds >= t
        tp = ((acc == 1) & (labels == 1)).sum()
        fp = ((acc == 1) & (labels == 0)).sum()
        fn = ((acc == 0) & (labels == 1)).sum()
        precision = tp / (tp + fp) if tp + fp else float("nan")
        recall = tp / (tp + fn) if tp + fn else float("nan")
        # false-reactivation on absent rows only
        abs_preds = [preds[i] for i in range(len(labels))
                     if pred_rows[i]["present"] == "0"]
        frate = np.mean([1.0 if s >= t else 0.0 for s in abs_preds]) \
            if abs_preds else float("nan")
        out.append({"threshold": t, "precision": round(precision, 4),
                    "recall": round(recall, 4),
                    "false_reactivation": round(float(frate), 4),
                    "accepts": int(acc.sum())})
        print(out[-1])
    with (OUT / "learned_verifier_cv.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(fold_rows[0].keys()))
        w.writeheader()
        w.writerows(fold_rows)
    with (OUT / "learned_verifier_curve.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0].keys()))
        w.writeheader()
        w.writerows(out)


if __name__ == "__main__":
    main()
