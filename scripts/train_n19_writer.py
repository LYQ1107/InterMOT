#!/usr/bin/env python
"""N19.7/19.8: Learned memory-write policy (Writer V0).

Small MLP over causal tracker-state + appearance-similarity features.
Train = train30 deliveries; calibrate threshold on cal10. No val25.
Label: SAFE_IDENTITY_WRITE (GT-correct delivery, IoU >= 0.5) - offline label
only; inference uses only the causal feature vector.

Usage (DDP, 4 GPUs):
  torchrun --nproc_per_node=4 scripts/train_n19_writer.py \
    --train outputs/n19/write_dataset_train30.csv \
    --cal outputs/n19/write_dataset_cal10.csv \
    --out outputs/n19/models/writer_v0
"""

import argparse
import csv
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from sklearn.metrics import roc_auc_score

ROOT = Path(".")
sys.path.insert(0, str(ROOT / "scripts"))

from n19_writer_features import (  # noqa
    FEATURES, feature_names, to_feature_vec)


class WriterMLP(nn.Module):
    def __init__(self, d_in, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def load_rows(path):
    with Path(path).open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def build(path, scaler=None, fit=False):
    rows = load_rows(path)
    X = np.stack([to_feature_vec(r) for r in rows])
    y = np.asarray([int(float(r.get("safe_write", 0)))
                    for r in rows], dtype=np.float32)
    if fit:
        mean = X.mean(axis=0)
        std = X.std(axis=0) + 1e-6
        scaler = (mean, std)
    mean, std = scaler
    X = (X - mean) / std
    return X, y, scaler


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--cal", required=True)
    ap.add_argument("--out", default=str(ROOT / "outputs/n19/models/writer_v0"))
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=4096)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--target-precision", type=float, default=0.95)
    args = ap.parse_args()

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world = int(os.environ.get("WORLD_SIZE", "1"))
    if world > 1:
        dist.init_process_group("nccl")
        torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available()
                          else "cpu")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    Xtr, ytr, scaler = build(args.train, fit=True)
    Xcal, ycal, _ = build(args.cal, scaler=scaler)
    if local_rank == 0:
        print(f"train rows={len(Xtr)} pos={ytr.sum():.0f} "
              f"({ytr.mean():.3f}) cal rows={len(Xcal)} "
              f"pos={ycal.sum():.0f} ({ycal.mean():.3f})", flush=True)

    d_in = Xtr.shape[1]
    model = WriterMLP(d_in, hidden=args.hidden).to(device)
    base_model = model
    if world > 1:
        model = nn.parallel.DistributedDataParallel(model)
        base_model = model.module
    # balanced class weights (no magic safety multiplier; threshold tuning
    # will enforce the contamination-averse operating point)
    pos = float(ytr.sum())
    neg = len(ytr) - pos
    pos_w = torch.tensor(neg / max(pos, 1.0), device=device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_w)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr,
                           weight_decay=1e-4)

    Xt = torch.from_numpy(Xtr)
    yt = torch.from_numpy(ytr)
    n = len(Xt)
    best_auc = -1.0
    best_state = None
    for epoch in range(args.epochs):
        model.train()
        perm = torch.randperm(n)
        total_loss = 0.0
        nb = 0
        for i in range(0, n, args.batch_size):
            idx = perm[i:i + args.batch_size]
            if world > 1 and idx.numel() % world != 0:
                idx = idx[:idx.numel() - idx.numel() % world]
            if idx.numel() == 0:
                continue
            rank_idx = idx[local_rank::world] if world > 1 else idx
            if rank_idx.numel() == 0:
                continue
            xb = Xt[rank_idx].to(device)
            yb = yt[rank_idx].to(device)
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()
            total_loss += float(loss) * rank_idx.numel()
            nb += rank_idx.numel()
        if world > 1:
            dist.barrier()
        if local_rank == 0:
            model.eval()
            with torch.inference_mode():
                p = torch.sigmoid(model(
                    torch.from_numpy(Xcal).to(device))).cpu().numpy()
            auc = roc_auc_score(ycal, p)
            print(f"epoch={epoch} loss={total_loss / max(nb, 1):.4f} "
                  f"cal_auc={auc:.4f}", flush=True)
            if auc > best_auc:
                best_auc = auc
                best_state = {k: v.detach().cpu().clone()
                              for k, v in base_model.state_dict().items()}

    if local_rank == 0:
        out_dir = Path(args.out)
        out_dir.mkdir(parents=True, exist_ok=True)
        torch.save(best_state, out_dir / "writer_v0.pt")
        (out_dir / "writer_config.json").write_text(json.dumps({
            "features": feature_names(),
            "hidden": args.hidden, "epochs": args.epochs,
            "lr": args.lr, "seed": args.seed,
            "scaler_mean": scaler[0].tolist(),
            "scaler_std": scaler[1].tolist(),
            "cal_auc": float(best_auc),
        }, indent=2), encoding="utf-8")
        # threshold sweep on cal
        base_model.load_state_dict(best_state)
        model.eval()
        with torch.inference_mode():
            p = torch.sigmoid(model(
                torch.from_numpy(Xcal).to(device))).cpu().numpy()
        rows = []
        for thr in np.arange(0.1, 0.99, 0.01):
            pred = p >= thr
            tp = float((pred & (ycal == 1)).sum())
            fp = float((pred & (ycal == 0)).sum())
            fn = float((~pred & (ycal == 1)).sum())
            prec = tp / max(tp + fp, 1e-9)
            rec = tp / max(tp + fn, 1e-9)
            rows.append({"threshold": round(float(thr), 2),
                         "precision": round(prec, 4),
                         "recall": round(rec, 4),
                         "safe_writes": int(tp + fp)})
        (out_dir / "calibration.csv").write_text(
            "threshold,precision,recall,safe_writes\n" + "\n".join(
                f"{r['threshold']},{r['precision']},{r['recall']},"
                f"{r['safe_writes']}" for r in rows) + "\n",
            encoding="utf-8")
        chosen = None
        for r in rows:
            if r["precision"] >= args.target_precision:
                chosen = r
        (out_dir / "writer_threshold.json").write_text(json.dumps(
            chosen if chosen else rows[-1], indent=2), encoding="utf-8")
        print(f"WRITER_TRAIN_DONE best_auc={best_auc:.4f} "
              f"chosen={chosen}", flush=True)
    if world > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
