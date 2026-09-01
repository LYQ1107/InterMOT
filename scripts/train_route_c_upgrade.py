#!/usr/bin/env python
"""N18 RouteC.10: one evidence-driven architecture upgrade.

A detection-independent identity canonicalizer: a small MLP over the frozen
pre-head features (feat_res4 512d + feat_res5 1024d -> 2048d L2). Unlike R1's
box_head unfreeze, this branch never touches the detection score path. It
tests whether nonlinear capacity (vs R0's linear projection) can recover
stale human anchors from the same frozen features.
"""

import argparse
import csv
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as Fn

ROOT = Path(".")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from train_route_c_r0 import PairStore, run_validation  # noqa: E402

OUT = ROOT / "outputs/n18/route_c"
CACHE = OUT / "gfn_cache"
MODELS = OUT / "models"


class Canonicalizer(nn.Module):
    def __init__(self, hidden=512, out=2048):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1536, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, out))

    def forward(self, f4, f5):
        x = torch.cat([f4, f5], dim=1)
        return Fn.normalize(self.net(x), dim=1)


def upgrade_loss(head, q4, q5, zp, zneg, pos_mask, gids, pos_gids,
                 tau, margin, w_margin):
    B = q4.shape[0]
    pos_idx = torch.nonzero(pos_mask, as_tuple=False).squeeze(1)
    zq = head(q4, q5)
    zp_emb = head(zp[0], zp[1]) if zp is not None else None
    zneg_emb = head(zneg[0], zneg[1]) if zneg is not None else None
    losses = []
    if len(pos_idx):
        zq_p = zq[pos_idx]
        zp_p = zp_emb
        logits_pos = (zq_p * zp_p).sum(1) / tau
        sim_other = zq_p @ zp_p.T / tau
        same_id = gids[pos_idx].unsqueeze(1).to(q4.device) == \
            pos_gids.unsqueeze(0).to(q4.device)
        sim_other = sim_other.masked_fill(same_id, -1e9)
        sim_other = sim_other - torch.eye(
            sim_other.shape[0], device=sim_other.device) * 1e9
        parts = [logits_pos.unsqueeze(1), sim_other]
        if zneg_emb is not None:
            parts.append(zq_p @ zneg_emb.T / tau)
        logits = torch.cat(parts, dim=1)
        labels = torch.zeros(len(pos_idx), dtype=torch.long,
                             device=logits.device)
        losses.append(Fn.cross_entropy(logits, labels))
        if zneg_emb is not None:
            hard = (zq_p @ zneg_emb.T / tau).max(dim=1).values
            losses.append(w_margin * Fn.relu(
                margin - (logits_pos - hard)).mean())
    neg_idx = torch.nonzero(~pos_mask, as_tuple=False).squeeze(1)
    if len(neg_idx) and zneg_emb is not None:
        hard_n = (zq[neg_idx] @ zneg_emb.T).max(dim=1).values
        losses.append(w_margin * Fn.relu(hard_n).mean())
    if not losses:
        return torch.zeros((), device=q4.device, requires_grad=True)
    return torch.stack(losses).mean()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--steps-per-epoch", type=int, default=2000)
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--tau", type=float, default=0.07)
    ap.add_argument("--margin", type=float, default=0.25)
    ap.add_argument("--w-margin", type=float, default=0.5)
    ap.add_argument("--pos-frac", type=float, default=0.8)
    ap.add_argument("--val-every", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--overfit", action="store_true")
    ap.add_argument("--tag", default="upgrade")
    args = ap.parse_args()

    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    torch.cuda.set_device(rank)
    device = f"cuda:{rank}"
    torch.manual_seed(args.seed + rank)

    head = Canonicalizer().to(device)
    head = nn.parallel.DistributedDataParallel(
        head, device_ids=[rank], find_unused_parameters=False)
    store = PairStore(OUT / "temporal_pairs_train.csv", device,
                      args.seed + rank * 1000, overfit=args.overfit)
    params = [p for p in head.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=1e-4)
    total_steps = args.epochs * args.steps_per_epoch
    if args.overfit:
        total_steps = 300
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=args.lr, total_steps=total_steps,
        pct_start=0.03, div_factor=25.0, final_div_factor=1e4)

    MODELS.mkdir(parents=True, exist_ok=True)
    log_path = OUT / f"{args.tag}_training_log.csv"
    if rank == 0 and not log_path.exists():
        with open(log_path, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(
                ["step", "epoch", "loss", "val_top1", "val_top3"])
    step = 0
    t0 = time.time()
    for epoch in range(args.epochs):
        for _ in range(args.steps_per_epoch):
            ids = store.sample_batch(args.batch_size, args.pos_frac)
            q4, q5, zp, zneg, pos_mask, gids, pos_gids, B = \
                store.make_batch(ids)
            if B == 0:
                continue
            loss = upgrade_loss(
                head, q4, q5, zp, zneg, pos_mask, gids, pos_gids,
                args.tau, args.margin, args.w_margin)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, max_norm=5.0)
            opt.step()
            sched.step()
            step += 1
            if rank == 0 and step % (100 if args.overfit else
                                     args.val_every) == 0:
                val = run_validation(
                    lambda d: (head.module(d["feat_res4"],
                                           d["feat_res5"]), None), device)
                with open(log_path, "a", newline="",
                          encoding="utf-8") as f:
                    csv.writer(f).writerow(
                        [step, epoch, round(float(loss), 4),
                         val["top1"], val["top3"]])
                print(json.dumps(
                    {"step": step, "loss": float(loss),
                     **{k: (None if v is None else round(v, 4))
                        for k, v in val.items()}}, ensure_ascii=False),
                    flush=True)
            if args.overfit and step >= total_steps:
                break
        if args.overfit:
            break
    torch.save(head.module.state_dict(), MODELS / f"{args.tag}_last.pt")
    if rank == 0:
        (MODELS / f"{args.tag}_config.json").write_text(json.dumps({
            "tag": args.tag, "arch": "MLP 1536-512-512-2048, L2",
            "epochs": args.epochs,
            "steps_per_epoch": args.steps_per_epoch,
            "batch_size": args.batch_size, "lr": args.lr,
            "tau": args.tau, "margin": args.margin,
            "w_margin": args.w_margin, "seed": args.seed,
            "runtime_s": round(time.time() - t0, 1),
            "train_split": "n15_frozen train30",
        }, indent=1))
    dist.destroy_process_group()
    if rank == 0:
        print("UPGRADE_TRAIN_DONE", flush=True)


if __name__ == "__main__":
    main()
