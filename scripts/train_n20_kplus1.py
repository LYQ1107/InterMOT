#!/usr/bin/env python
"""N20.8-13: K+1 set-level temporal identity verifier (train30 -> cal10).

Models:
  B_sf  : single-frame K+1 logistic (features at step 1)
  B_h   : single-frame K+1 logistic (features at step H)
  GRU   : shared GRU per hypothesis + set-level head -> K+1 logits
  GRUAD : GRU + adaptive decision simulation (commit if P(best) and margin
          exceed calibrated thresholds; else continue; reject-all otherwise)

Training split: train30 (all candidates, real recovery distribution).
Calibration/evaluation split: cal10 (natural distribution, no rebalance).
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

ROOT = Path(".")
N20 = ROOT / "outputs/n20"

NUMERIC = [
    "candidate_rank", "gfn_sim_human_root", "r0_sim_human_root",
    "gfn_sim_mem_last", "gfn_sim_mem_max", "r0_sim_mem_last",
    "r0_sim_mem_max", "mem_age", "n_mem_slots", "temp_sim_prev",
    "temp_sim_first", "box_area", "area_change", "center_delta",
    "velocity", "temporal_iou", "consecutive_delivered",
    "shadow_delivered", "n_dets", "gfn_margin_h", "candidate_age",
    "memory_fresh", "rank_mem", "initial_correct", "init_rank_correct",
    "comp_delivered_ratio", "comp_mean_gfn_sim", "comp_max_gfn_sim",
    "comp_overlap_max", "comp_sim_margin",
]


def to_float(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return float("nan")


class SharedGRUSet(nn.Module):
    def __init__(self, d, hidden=32, maxk=5):
        super().__init__()
        self.gru = nn.GRU(d, hidden, batch_first=True)
        self.maxk = maxk
        self.head = nn.Sequential(
            nn.Linear(hidden * 4, 64), nn.ReLU(), nn.Linear(64, maxk + 1))

    def forward(self, x, mask):
        # x: (B, K, H, D), mask: (B, K)
        B, K, H, D = x.shape
        xr = x.reshape(B * K, H, D)
        out, _ = self.gru(xr)
        z = out[:, -1].reshape(B, K, -1)  # (B,K,hidden)
        z = z * mask.unsqueeze(-1)
        present = mask.sum(1).clamp(min=1)
        zmean = z.sum(1) / present.unsqueeze(-1)
        zmax = z.max(1).values
        zk = torch.cat([z, zmean.unsqueeze(1).expand(B, K, -1),
                        zmax.unsqueeze(1).expand(B, K, -1)], dim=-1)
        margin = zmax.unsqueeze(1).expand(B, K, -1) - z
        zk = torch.cat([zk, margin], dim=-1)
        logits = self.head(zk)  # (B,K,maxk+1)
        # mask out invalid candidate rows
        logits = logits * mask.unsqueeze(-1) - 1e9 * (1 - mask).unsqueeze(-1)
        # NONE logit = max over rows of column 0; candidate j = max over
        # rows of column j+1 (set-level voting).
        cand_logits = logits[:, :, 1:1 + K].max(1).values  # (B,K)
        none_logit = logits[:, :, 0].max(1).values.unsqueeze(1)  # (B,1)
        out_logits = torch.cat([none_logit, cand_logits], dim=1)
        return out_logits


def load_groups(csv_path, h, feats, splits=None):
    rows = list(csv.DictReader(Path(csv_path).open(
        newline="", encoding="utf-8")))
    if splits is not None:
        rows = [r for r in rows if r["sequence"] in splits]
    by_att = defaultdict(list)
    for r in rows:
        by_att[r["attempt"]].append(r)
    groups = []
    for att, rs in by_att.items():
        by_k = defaultdict(list)
        for r in rs:
            by_k[int(r["candidate_rank"])].append(r)
        seqs = []
        for k in sorted(by_k):
            rs2 = sorted(by_k[k], key=lambda r: int(r["evidence_step"]))
            rs2 = [r for r in rs2 if int(r["evidence_step"]) <= h]
            if not rs2:
                continue
            vecs = []
            for r in rs2:
                vecs.append([to_float(r[c]) for c in feats])
            while len(vecs) < h:
                vecs.append(vecs[-1])
            arr = np.asarray(vecs, dtype=np.float32)
            arr = np.nan_to_num(arr, nan=0.0)
            seqs.append((k, arr, int(rs2[-1]["label_correct"]),
                         rs2[-1]["sequence"], att))
        if not seqs:
            continue
        groups.append(seqs)
    return groups


def groups_to_tensors(groups, h, d, maxk):
    X = np.zeros((len(groups), maxk, h, d), dtype=np.float32)
    M = np.zeros((len(groups), maxk), dtype=np.float32)
    Y = np.full((len(groups), maxk + 1), 0.0, dtype=np.float32)
    for i, seqs in enumerate(groups):
        for j, (k, arr, lab, _, _) in enumerate(seqs):
            if j >= maxk:
                break
            X[i, j] = arr
            M[i, j] = 1.0
            if lab == 1:
                Y[i, j + 1] = 1.0
        if Y[i, 1:].sum() == 0:
            Y[i, 0] = 1.0
    return (torch.from_numpy(X), torch.from_numpy(M),
            torch.from_numpy(Y))


def train_gru(model, Xtr, Mtr, Ytr, Xcal, Mcal, epochs, lr=1e-3,
              none_weight=0.5):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    counts = Ytr.sum(0)
    weights = torch.ones(Ytr.shape[1])
    weights[0] = none_weight
    for j in range(1, Ytr.shape[1]):
        if counts[j] > 0:
            weights[j] = max(1.0, counts[0] / counts[j])
    lossf = nn.CrossEntropyLoss(weight=weights)
    model.train()
    for _ in range(epochs):
        opt.zero_grad()
        logits = model(Xtr, Mtr)
        loss = lossf(logits, Ytr.argmax(1))
        loss.backward()
        opt.step()
    model.eval()
    with torch.inference_mode():
        p_tr = torch.softmax(model(Xtr, Mtr), 1).numpy()
        p_cal = torch.softmax(model(Xcal, Mcal), 1).numpy()
    return p_tr, p_cal


def sf_features(groups, step_idx, feats):
    # per-attempt: feature of candidate j at the given step (mean over steps
    # up to step_idx for robustness)
    X = []
    y = []
    for seqs in groups:
        row = []
        for k, arr, lab, _, _ in seqs:
            row.append(arr[:step_idx + 1].mean(0))
        # pad to maxk
        while len(row) < 5:
            row.append(np.zeros(len(feats), dtype=np.float32))
        X.append(np.concatenate(row))
        labs = [lab for _, _, lab, _, _ in seqs]
        if any(labs):
            y.append([i for i, l in enumerate(labs) if l][0] + 1)
        else:
            y.append(0)
    return np.asarray(X), np.asarray(y)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--h", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--train-csv", default="shadow_kplus1_train30.csv")
    ap.add_argument("--cal-csv", default="shadow_kplus1_cal10.csv")
    ap.add_argument("--out", default="kplus1_metrics.csv")
    ap.add_argument("--none-weight", type=float, default=0.5)
    ap.add_argument("--normalize", action="store_true")
    args = ap.parse_args()
    tr_csv = N20 / args.train_csv
    cal_csv = N20 / args.cal_csv
    if not tr_csv.exists() or not cal_csv.exists():
        raise SystemExit(f"missing dataset: {tr_csv} / {cal_csv}")
    # feature columns present in both files
    with tr_csv.open(newline="") as f:
        tr_cols = next(csv.reader(f))
    feats = [c for c in NUMERIC if c in tr_cols]
    print(f"features={len(feats)} h={args.h}", flush=True)
    tr_groups = load_groups(tr_csv, args.h, feats)
    cal_groups = load_groups(cal_csv, args.h, feats)
    print(f"train30 groups={len(tr_groups)} cal10 groups={len(cal_groups)}",
          flush=True)
    Xtr, Mtr, Ytr = groups_to_tensors(tr_groups, args.h, len(feats), 5)
    Xcal, Mcal, Ycal = groups_to_tensors(cal_groups, args.h, len(feats), 5)
    if args.normalize:
        # z-score over train examples (causal: statistics from train only)
        flat = Xtr.reshape(-1, len(feats))
        mu = flat.mean(0)
        sd = flat.std(0) + 1e-8
        Xtr = (Xtr - mu) / sd
        Xcal = (Xcal - mu) / sd
    else:
        mu = np.zeros(len(feats), dtype=np.float32)
        sd = np.ones(len(feats), dtype=np.float32)
    # --- baselines
    Xsf_tr, ysf_tr = sf_features(tr_groups, 0, feats)
    Xsf_cal, ysf_cal = sf_features(cal_groups, 0, feats)
    lr1 = LogisticRegression(max_iter=2000, C=1.0)
    lr1.fit(Xsf_tr, ysf_tr)
    p_sf1 = lr1.predict_proba(Xsf_cal)
    Xsh_tr, ysh_tr = sf_features(tr_groups, args.h - 1, feats)
    Xsh_cal, ysh_cal = sf_features(cal_groups, args.h - 1, feats)
    lrh = LogisticRegression(max_iter=2000, C=1.0)
    lrh.fit(Xsh_tr, ysh_tr)
    p_sfh = lrh.predict_proba(Xsh_cal)
    # --- GRU set model
    gru = SharedGRUSet(len(feats))
    p_gru_tr, p_gru = train_gru(gru, Xtr, Mtr, Ytr, Xcal, Mcal,
                                args.epochs, none_weight=args.none_weight)
    # --- metrics
    def acc(p, y):
        return float((p.argmax(1) == y).mean())
    rows = [
        {"model": "sf_step1", "cal_acc": round(acc(p_sf1, ysf_cal), 4)},
        {"model": "sf_stepH", "cal_acc": round(acc(p_sfh, ysh_cal), 4)},
        {"model": "gru_kplus1", "cal_acc": round(
            acc(p_gru, Ycal.argmax(1).numpy()), 4)},
    ]
    # NONE / candidate precision-recall for GRU
    ytrue = Ycal.argmax(1).numpy()
    for cls in range(6):
        tp = float(((p_gru.argmax(1) == cls) & (ytrue == cls)).sum())
        pred = float((p_gru.argmax(1) == cls).sum())
        real = float((ytrue == cls).sum())
        rows.append({"model": f"gru_cls{cls}",
                     "precision": round(tp / pred, 4) if pred else None,
                     "recall": round(tp / real, 4) if real else None,
                     "n": int(real)})
    with (N20 / args.out).open("w", newline="", encoding="utf-8") as f:
        cols = ["model", "cal_acc", "precision", "recall", "n"]
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    torch.save({"model": gru.state_dict(), "feature_cols": feats, "h": args.h,
                "mu": mu, "sd": sd, "normalize": args.normalize},
               N20 / "models" / "kplus1_gru.pt")
    print(json.dumps(rows, indent=2), flush=True)
    print("KPLUS1_DONE", flush=True)


if __name__ == "__main__":
    main()
