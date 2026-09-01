#!/usr/bin/env python
"""N19.11b: Retrain the verifier on live-loop candidates (evidence-driven).

Train on train30 offline candidates + live cal10 candidate features dumped
from FULL_LOOP_N19 (threshold 0.75, learned memory). Calibrate on 3 held-out
cal10 sequences. Labels are offline GT-correctness of the candidate box.
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

ROOT = Path(".")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from eval_route_c_r0 import iou  # noqa: E402
from run_n18_full_loop_v0 import load_gt  # noqa: E402

N19 = ROOT / "outputs/n19"
FEATS = ["gfn_top1_sim", "gfn_margin", "gfn_top1_score", "n_dets",
         "memory_age", "n_agree_slots", "r0_top1_sim"]


def read(path):
    with Path(path).open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def label_live(rows):
    by_seq = {}
    out = []
    for r in rows:
        seq = r["seq"]
        if seq not in by_seq:
            by_seq[seq] = load_gt(seq)
        gf = by_seq[seq].get(int(float(r["frame"])))
        gid = int(float(r["gid"]))
        box = [float(v) for v in r["box"].strip("[]").split(",")]
        lab = 0
        if gf is not None and gid in gf.gt_ids:
            tgt = [float(v) for v in gf.boxes[gf.gt_ids.index(gid)]]
            lab = int(iou(box, tgt) >= 0.5)
        out.append({**r, "top1_correct": lab, "sequence": seq})
    return out


def fv(r):
    return [0.0 if str(r.get(k)) in ("", "nan", "None") else float(r[k])
            for k in FEATS]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--live-prefix", default=str(N19 / "verify_live_s"))
    ap.add_argument("--train30", default=str(N19 / "verifier_dataset_train30.csv"))
    ap.add_argument("--out", default=str(N19 / "models/verifier_n19_v2.joblib"))
    ap.add_argument("--holdout-seqs",
                    default="dancetrack0075,dancetrack0087,dancetrack0099")
    ap.add_argument("--target-precision", type=float, default=0.50)
    args = ap.parse_args()

    live = []
    for i in range(4):
        p = Path(f"{args.live_prefix}{i}.csv")
        if p.exists():
            live.extend(read(p))
    live = label_live(live)
    tr = read(args.train30)
    hold = {s.strip() for s in args.holdout_seqs.split(",")}
    cal_rows = [r for r in live if r["sequence"] in hold]
    tr_rows = tr + [r for r in live if r["sequence"] not in hold]
    Xtr = np.stack([fv(r) for r in tr_rows])
    ytr = np.asarray([int(r["top1_correct"]) for r in tr_rows])
    Xca = np.stack([fv(r) for r in cal_rows])
    yca = np.asarray([int(r["top1_correct"]) for r in cal_rows])
    clf = LogisticRegression(max_iter=5000).fit(Xtr, ytr)
    pca = clf.predict_proba(Xca)[:, 1]
    print(f"train rows={len(Xtr)} pos={ytr.sum()} "
          f"auc={roc_auc_score(ytr, clf.predict_proba(Xtr)[:, 1]):.4f}")
    print(f"cal rows={len(Xca)} pos={yca.sum()} auc={roc_auc_score(yca, pca):.4f}")
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
    joblib.dump({"model": clf, "features": FEATS,
                 "threshold": best["threshold"],
                 "calibration": best,
                 "n_live": len(live), "n_live_pos": int(sum(
                     int(r["top1_correct"]) for r in live))},
                args.out)
    print("VERIFIER_LIVE_TRAIN_DONE", args.out, flush=True)


if __name__ == "__main__":
    main()
