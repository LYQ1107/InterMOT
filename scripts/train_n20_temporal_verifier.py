#!/usr/bin/env python
"""N20.8-13: sequence-disjoint temporal verifier baselines on cal10 shadows.

Per-attempt sequence: evidence steps 1..H (H=5 default). Label: whether the
shadow is still GT-correct at the confirmation frame (offline label only).
Models: mean-pooled LR, mean-pooled MLP, GRU. 5-fold sequence-disjoint CV.
"""

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, precision_recall_curve,
                             roc_auc_score)

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


class MLP(nn.Module):
    def __init__(self, d, hidden=64):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d, hidden), nn.ReLU(),
                                 nn.Linear(hidden, 1))

    def forward(self, x):
        return self.net(x)


class GRUVerifier(nn.Module):
    def __init__(self, d, hidden=32):
        super().__init__()
        self.gru = nn.GRU(d, hidden, batch_first=True)
        self.head = nn.Linear(hidden, 1)

    def forward(self, x):
        out, _ = self.gru(x)
        return self.head(out[:, -1])


def make_sequences(rows, h, feats):
    by_att = defaultdict(list)
    for r in rows:
        by_att[r["attempt"]].append(r)
    X = []
    y = []
    meta = []
    for att, rs in by_att.items():
        rs = sorted(rs, key=lambda r: int(r["evidence_step"]))
        seq = [r for r in rs if int(r["evidence_step"]) <= h]
        if not seq:
            continue
        vecs = []
        for r in seq:
            vecs.append([to_float(r[c]) for c in feats])
        while len(vecs) < h:
            vecs.append(vecs[-1] if vecs else [0.0] * len(feats))
        arr = np.asarray(vecs, dtype=np.float32)
        arr = np.nan_to_num(arr, nan=0.0)
        X.append(arr)
        y.append(int(seq[-1]["label_correct"]))
        meta.append({"attempt": att, "sequence": seq[0]["sequence"],
                     "frame": seq[0]["frame"], "gid": seq[0]["gid"]})
    return X, np.asarray(y), meta


def fit_torch(model, Xtr, ytr, Xte, epochs=30, lr=1e-3, pool=False):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    lossf = nn.BCEWithLogitsLoss()
    Xt = torch.from_numpy(np.stack(Xtr)).float()
    Xe = torch.from_numpy(np.stack(Xte)).float()
    if pool:
        Xt = Xt.mean(1)
        Xe = Xe.mean(1)
    yt = torch.from_numpy(ytr).float()
    model.train()
    for _ in range(epochs):
        opt.zero_grad()
        loss = lossf(model(Xt).squeeze(-1), yt)
        loss.backward()
        opt.step()
    model.eval()
    with torch.inference_mode():
        return torch.sigmoid(model(Xe).squeeze(-1)).numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--h", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--out", default="temporal_verifier_cv.csv")
    args = ap.parse_args()
    rows = list(csv.DictReader(
        (N20 / "features" / "shadow_tracklets_cal10.csv").open(
            newline="", encoding="utf-8")))
    seqs = sorted({r["sequence"] for r in rows})
    print(f"seqs={len(seqs)} rows={len(rows)}", flush=True)
    # drop features with >40% missing
    miss = {}
    for c in NUMERIC:
        vals = np.asarray([to_float(r[c]) for r in rows])
        miss[c] = float(np.isnan(vals).mean())
    feats = [c for c in NUMERIC if miss[c] <= 0.4]
    print(f"features kept={len(feats)}", flush=True)
    Xall, yall, meta = make_sequences(rows, args.h, feats)
    att_seq = {m["attempt"]: m["sequence"] for m in meta}
    # 5 folds: group sequences
    rng = np.random.RandomState(0)
    seqs_shuf = list(seqs)
    rng.shuffle(seqs_shuf)
    folds = [seqs_shuf[i::5] for i in range(5)]
    out_rows = []
    oof_lr = np.zeros(len(yall))
    oof_mlp = np.zeros(len(yall))
    oof_gru = np.zeros(len(yall))
    fold_models = {}
    for fi, test_seqs in enumerate(folds):
        test_idx = [i for i, m in enumerate(meta)
                    if m["sequence"] in set(test_seqs)]
        train_idx = [i for i in range(len(meta)) if i not in set(test_idx)]
        Xtr = [Xall[i] for i in train_idx]
        Xte = [Xall[i] for i in test_idx]
        ytr, yte = yall[train_idx], yall[test_idx]
        # mean-pooled LR
        Xm_tr = np.stack([x.mean(0) for x in Xtr])
        Xm_te = np.stack([x.mean(0) for x in Xte])
        lr = LogisticRegression(max_iter=2000, C=0.1)
        lr.fit(Xm_tr, ytr)
        plr = lr.predict_proba(Xm_te)[:, 1]
        oof_lr[test_idx] = plr
        # mean-pooled MLP + GRU (torch)
        mlp = MLP(len(feats))
        pmlp = fit_torch(mlp, Xtr, ytr, Xte, epochs=args.epochs, pool=True)
        oof_mlp[test_idx] = pmlp
        gru = GRUVerifier(len(feats))
        pgru = fit_torch(gru, Xtr, ytr, Xte, epochs=args.epochs)
        oof_gru[test_idx] = pgru
        fold_models[fi] = {
            "gru": gru.state_dict(),
            "test_seqs": test_seqs,
        }
        for name, p in (("lr", plr), ("mlp", pmlp), ("gru", pgru)):
            auc = roc_auc_score(yte, p) if len(set(yte)) > 1 else np.nan
            ap = average_precision_score(yte, p)
            out_rows.append({"fold": fi, "model": name,
                             "n_test": len(yte),
                             "pos_rate": round(float(yte.mean()), 4),
                             "auc": round(float(auc), 4),
                             "ap": round(float(ap), 4)})
        print(f"fold {fi} done n_test={len(yte)}", flush=True)
    # overall OOF
    for name, p in (("lr", oof_lr), ("mlp", oof_mlp), ("gru", oof_gru)):
        auc = roc_auc_score(yall, p)
        ap = average_precision_score(yall, p)
        prec, rec, thr = precision_recall_curve(yall, p)
        out_rows.append({"fold": "OOF", "model": name,
                         "n_test": len(yall),
                         "pos_rate": round(float(yall.mean()), 4),
                         "auc": round(float(auc), 4),
                         "ap": round(float(ap), 4)})
    with (N20 / args.out).open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
        w.writeheader()
        w.writerows(out_rows)
    print(json.dumps(out_rows, indent=2), flush=True)
    # save OOF predictions
    with (N20 / "temporal_verifier_oof.csv").open(
            "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["attempt", "sequence", "frame",
                                          "gid", "label", "p_lr", "p_mlp",
                                          "p_gru"])
        w.writeheader()
        for i, m in enumerate(meta):
            w.writerow({"attempt": m["attempt"], "sequence": m["sequence"],
                        "frame": m["frame"], "gid": m["gid"],
                        "label": int(yall[i]), "p_lr": round(oof_lr[i], 4),
                        "p_mlp": round(oof_mlp[i], 4),
                        "p_gru": round(oof_gru[i], 4)})
    # save fold models + mapping for FULL_LOOP_N20 deployment
    out_dir = N20 / "models"
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"fold_models": fold_models,
                "feature_cols": feats, "h": args.h,
                "seq_to_fold": {s: fi for fi, fs in enumerate(folds)
                                for s in fs}},
               out_dir / "temporal_gru_folds.pt")
    print("TEMPORAL_CV_DONE", flush=True)
    print("TEMPORAL_CV_DONE", flush=True)


if __name__ == "__main__":
    main()
