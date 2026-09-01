#!/usr/bin/env python
"""N18 RouteC.4: R0 head-only temporal adaptation of the GFN identity head.

Only ``NormAwareEmbedding`` (roi_heads.embedding_head) is trainable; the
backbone, RPN and detection heads are never instantiated here. Inputs are the
pre-head features cached by build_route_c_feature_cache.py, so training is a
pure projection-level InfoNCE + hard-negative margin on temporal pairs.

Run with: torchrun --nproc_per_node=4 scripts/train_route_c_r0.py
"""

import argparse
import csv
import json
import os
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
sys.path.insert(0, str(ROOT / "third_party/GFN/src"))
from models.seqnext import NormAwareEmbedding  # noqa: E402

OUT = ROOT / "outputs/n18/route_c"
CACHE = OUT / "gfn_cache"
MODELS = OUT / "models"
CKPT = ROOT / "outputs/n18/checkpoints/gfn_cuhk_convnext_pytorch.pt"

GAP_BINS = [1, 3, 5, 10, 30, 60, 120, 240, 480, 4800]


def build_head(device):
    head = NormAwareEmbedding(
        featmap_names=["feat_res4", "feat_res5"],
        in_channels=[512, 1024], dim=2048, norm_type="batchnorm")
    head.rescaler = None
    ck = torch.load(str(CKPT), map_location="cpu")["model"]
    prefix = "roi_heads.embedding_head."
    sd = {k[len(prefix):]: v for k, v in ck.items()
          if k.startswith(prefix)}
    head.load_state_dict(sd)
    head.to(device)
    return head


def embed(head, f4, f5):
    emb, _ = head({"feat_res4": f4, "feat_res5": f5})
    return Fn.normalize(emb, dim=1)


class PairStore:
    """In-memory feature store + row indices, balanced across gap bins."""

    def __init__(self, pairs_csv, device, seed, overfit=False):
        rows = list(csv.DictReader(open(pairs_csv, encoding="utf-8")))
        self.caches = {}
        self.qcaches = {}
        self.qfeat = {}
        pos_rows = defaultdict(list)
        neg_rows = defaultdict(list)
        for i, r in enumerate(rows):
            bin_key = int(r["gap_bin"]) if r["gap_bin"] != "480+" else 4800
            if int(r["target_present"]) and int(
                    r["detector_contains_target"]):
                pos_rows[bin_key].append(i)
            else:
                neg_rows[bin_key].append(i)
        self.pos_rows = {k: np.asarray(v, dtype=np.int64)
                         for k, v in pos_rows.items() if v}
        self.neg_rows = {k: np.asarray(v, dtype=np.int64)
                         for k, v in neg_rows.items() if v}
        self.bins = sorted({k for k in list(self.pos_rows) +
                            list(self.neg_rows)})
        self.rows = rows
        self.overfit = overfit
        self.rng = np.random.RandomState(seed)
        self.device = device
        if overfit:
            fixed = sorted({
                int(i) for pool in self.pos_rows.values() for i in pool
            })[:64]
            self.rows = [rows[i] for i in fixed]
            self.fixed_ids = list(range(len(self.rows)))
        # preload query features for all identities
        qneed = {}
        for r in self.rows:
            qneed.setdefault(r["sequence"], set()).add(int(r["gid"]))
        for seq, gids in qneed.items():
            qc = np.load(CACHE / f"{seq}_queries.npz")
            idx = {int(g): i for i, g in enumerate(qc["gids"])}
            for g in gids:
                j = idx[g]
                self.qfeat[(seq, g)] = (
                    qc["qfeat4"][j].astype(np.float32),
                    qc["qfeat5"][j].astype(np.float32))
            qc.close()

    def cache_for(self, seq):
        if seq not in self.caches:
            z = np.load(CACHE / f"{seq}.npz")
            # NpzFile re-decompresses on every attribute access; materialize
            self.caches[seq] = {
                "frames": z["frames"], "offsets": z["offsets"],
                "feat4": z["feat4"], "feat5": z["feat5"],
                "emb": z["emb"], "boxes": z["boxes"], "scores": z["scores"],
            }
            z.close()
        return self.caches[seq]

    def feat_at(self, seq, det_idx):
        if det_idx < 0:
            return None
        z = self.cache_for(seq)
        return (z["feat4"][det_idx].astype(np.float32),
                z["feat5"][det_idx].astype(np.float32))

    def sample_batch(self, batch_size, pos_frac=0.8):
        if self.overfit:
            return self.fixed_ids
        ids = []
        for _ in range(batch_size):
            bin_key = int(self.rng.choice(self.bins))
            pools = []
            if bin_key in self.pos_rows:
                pools.append(("pos", self.pos_rows[bin_key]))
            if bin_key in self.neg_rows:
                pools.append(("neg", self.neg_rows[bin_key]))
            if not pools:
                continue
            if len(pools) == 2 and self.rng.rand() >= pos_frac:
                ids.append(int(self.rng.choice(pools[1][1])))
            else:
                ids.append(int(self.rng.choice(pools[0][1])))
        return ids

    def make_batch(self, ids):
        q4, q5, p4, p5 = [], [], [], []
        has_pos = []
        n4, n5, nvalid = [], [], []
        gids, pos_gids = [], []
        for i in ids:
            r = self.rows[i]
            qf4, qf5 = self.qfeat[(r["sequence"], int(r["gid"]))]
            gids.append(int(r["gid"]))
            q4.append(qf4)
            q5.append(qf5)
            det_idx = int(r["det_idx"])
            if (int(r["target_present"]) and
                    int(r["detector_contains_target"]) and det_idx >= 0):
                a, b = self.feat_at(r["sequence"], det_idx)
                p4.append(a)
                p5.append(b)
                pos_gids.append(int(r["gid"]))
                has_pos.append(True)
            else:
                has_pos.append(False)
            for nidx in json.loads(r["hard_neg_idxs"]):
                a, b = self.feat_at(r["sequence"], int(nidx))
                if a is None:
                    continue
                n4.append(a)
                n5.append(b)
                nvalid.append(1)
        B = len(q4)
        q4 = torch.from_numpy(np.stack(q4)).to(self.device)
        q5 = torch.from_numpy(np.stack(q5)).to(self.device)
        zneg = None
        if n4:
            zn4 = torch.from_numpy(np.stack(n4)).to(self.device)
            zn5 = torch.from_numpy(np.stack(n5)).to(self.device)
            zneg = (zn4, zn5)
        zp = None
        if p4:
            zp4 = torch.from_numpy(np.stack(p4)).to(self.device)
            zp5 = torch.from_numpy(np.stack(p5)).to(self.device)
            zp = (zp4, zp5)
        pos_mask = torch.tensor(has_pos, dtype=torch.bool)
        gid_t = torch.tensor(gids, dtype=torch.long)
        pos_gid_t = torch.tensor(pos_gids, dtype=torch.long)
        return q4, q5, zp, zneg, pos_mask, gid_t, pos_gid_t, B


def info_nce(head, q4, q5, zp, zneg, pos_mask, gids, pos_gids,
             tau, margin, w_margin):
    B = q4.shape[0]
    pos_idx = torch.nonzero(pos_mask, as_tuple=False).squeeze(1)
    # One fused forward keeps BatchNorm's in-place running-stat update from
    # being called twice inside the same autograd step (CUDA version check).
    parts4, parts5 = [q4], [q5]
    if zp is not None:
        parts4.append(zp[0])
        parts5.append(zp[1])
    if zneg is not None:
        parts4.append(zneg[0])
        parts5.append(zneg[1])
    z = embed(head, torch.cat(parts4, dim=0), torch.cat(parts5, dim=0))
    zq = z[:B]
    off = B
    zp_emb = None
    if zp is not None:
        P = zp[0].shape[0]
        zp_emb = z[off:off + P]
        off += P
    zneg_emb = z[off:] if zneg is not None else None
    losses = []
    n_pos = 0
    if len(pos_idx):
        zq_p = zq[pos_idx]
        # zp_emb rows are already exactly the positive rows in batch order
        zp_p = zp_emb
        logits_pos = (zq_p * zp_p).sum(1) / tau  # (P,)
        # negatives: all other positives + hard negatives
        sim_other = zq_p @ zp_p.T / tau
        # same-identity positives must not act as negatives (different gaps)
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
        # margin on hardest negative
        if zneg_emb is not None:
            hard = (zq_p @ zneg_emb.T / tau).max(dim=1).values
            losses.append(w_margin * Fn.relu(
                margin - (logits_pos - hard)).mean())
        n_pos = len(pos_idx)
    # absent rows: query must stay away from all gallery negatives
    neg_idx = torch.nonzero(~pos_mask, as_tuple=False).squeeze(1)
    if len(neg_idx) and zneg_emb is not None:
        zq_n = zq[neg_idx]
        hard_n = (zq_n @ zneg_emb.T).max(dim=1).values
        losses.append(w_margin * Fn.relu(hard_n).mean())
    if not losses:
        return torch.zeros((), device=q4.device, requires_grad=True), \
            0, float("nan")
    return torch.stack(losses).mean(), n_pos, float("nan")


_VAL_ROWS = None
_VAL_PICKED = None
_VAL_CACHES = {}
_VAL_QCACHES = {}


def run_validation(head, device, sample_per_bin=12):
    """cal10 held-out temporal retrieval (sequence-disjoint)."""
    global _VAL_ROWS, _VAL_PICKED
    if hasattr(head, "eval"):
        head.eval()
    if _VAL_ROWS is None:
        rows = list(csv.DictReader(open(
            OUT / "temporal_pairs_cal.csv", encoding="utf-8")))
        by_bin = defaultdict(list)
        for i, r in enumerate(rows):
            by_bin[r["gap_bin"]].append(i)
        picked = []
        rng = np.random.RandomState(7)
        for b in sorted(by_bin, key=lambda x: (x == "480+", int(x)
                                                if x != "480+" else 1e9)):
            idx = by_bin[b]
            picked.extend(rng.choice(
                idx, size=min(len(idx), sample_per_bin),
                replace=False).tolist())
        _VAL_ROWS = rows
        _VAL_PICKED = picked
    rows, picked = _VAL_ROWS, _VAL_PICKED
    caches, qcaches = _VAL_CACHES, _VAL_QCACHES
    hits = defaultdict(int)
    hits_long = defaultdict(int)
    n_present = 0
    n_long = 0
    n_absent = 0
    absent_sim = []
    with torch.inference_mode():
        for i in picked:
            r = rows[i]
            seq = r["sequence"]
            if seq not in qcaches:
                qc = np.load(CACHE / f"{seq}_queries.npz")
                qcaches[seq] = ({int(g): j for j, g in enumerate(qc["gids"])},
                                qc)
            if seq not in caches:
                z0 = np.load(CACHE / f"{seq}.npz")
                caches[seq] = {
                    "frames": z0["frames"], "offsets": z0["offsets"],
                    "feat4": z0["feat4"], "feat5": z0["feat5"],
                }
                z0.close()
            z = caches[seq]
            qi = qcaches[seq][0][int(r["gid"])]
            q4 = torch.from_numpy(
                qcaches[seq][1]["qfeat4"][qi].astype(np.float32)).unsqueeze(0).to(device)
            q5 = torch.from_numpy(
                qcaches[seq][1]["qfeat5"][qi].astype(np.float32)).unsqueeze(0).to(device)
            zq = embed(head, q4, q5)
            f = int(r["gallery_frame"])
            o = np.searchsorted(z["frames"], f)
            lo = int(z["offsets"][o - 1]) if o > 0 else 0
            hi = int(z["offsets"][o])
            if hi == lo:
                continue
            g4 = torch.from_numpy(
                z["feat4"][lo:hi].astype(np.float32)).to(device)
            g5 = torch.from_numpy(
                z["feat5"][lo:hi].astype(np.float32)).to(device)
            zg = embed(head, g4, g5)
            sims = (zq @ zg.T)[0]
            if (int(r["target_present"]) and
                    int(r["detector_contains_target"]) and
                    int(r["det_idx"]) >= 0):
                n_present += 1
                di = int(r["det_idx"]) - lo
                rank = int((sims > sims[di]).sum()) + 1
                for k in (1, 3, 5, 10):
                    hits[k] += int(rank <= k)
                if r["gap_bin"] in ("120", "240", "480", "480+"):
                    n_long += 1
                    hits_long[1] += int(rank <= 1)
                    hits_long[3] += int(rank <= 3)
            elif not int(r["target_present"]):
                n_absent += 1
                absent_sim.append(float(sims.max()))
    if hasattr(head, "train"):
        head.train()
    out = {
        "n_present": n_present, "n_absent": n_absent,
        "top1": hits[1] / n_present if n_present else None,
        "top3": hits[3] / n_present if n_present else None,
        "top5": hits[5] / n_present if n_present else None,
        "top10": hits[10] / n_present if n_present else None,
        "top1_long": hits_long[1] / n_long if n_long else None,
        "top3_long": hits_long[3] / n_long if n_long else None,
        "absent_mean_top1_sim": float(np.mean(absent_sim))
        if absent_sim else None,
    }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--steps-per-epoch", type=int, default=2000)
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--warmup-steps", type=int, default=200)
    ap.add_argument("--tau", type=float, default=0.07)
    ap.add_argument("--margin", type=float, default=0.25)
    ap.add_argument("--w-margin", type=float, default=0.5)
    ap.add_argument("--pos-frac", type=float, default=0.8)
    ap.add_argument("--val-every", type=int, default=500)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--overfit", action="store_true")
    ap.add_argument("--tag", default="r0")
    args = ap.parse_args()

    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world = dist.get_world_size()
    torch.cuda.set_device(rank)
    device = f"cuda:{rank}"
    torch.manual_seed(args.seed + rank)

    head = build_head(device)
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
    best_top1 = -1.0
    log_path = OUT / f"{args.tag}_training_log.csv"
    if rank == 0 and not log_path.exists():
        with open(log_path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["step", "epoch", "loss", "val_top1", "val_top3",
                        "val_top10", "val_top1_long", "val_top3_long",
                        "absent_sim"])
    step = 0
    t0 = time.time()
    for epoch in range(args.epochs):
        for _ in range(args.steps_per_epoch):
            ids = store.sample_batch(args.batch_size, args.pos_frac)
            q4, q5, zp, zneg, pos_mask, gids, pos_gids, B = \
                store.make_batch(ids)
            if B == 0:
                continue
            loss, n_pos, _ = info_nce(
                head, q4, q5, zp, zneg, pos_mask, gids, pos_gids,
                args.tau, args.margin, args.w_margin)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, max_norm=5.0)
            opt.step()
            sched.step()
            step += 1
            val_every = 100 if args.overfit else args.val_every
            if rank == 0 and step % val_every == 0:
                print(f"[rank0] validation step={step} begin", flush=True)
                val = run_validation(head.module, device)
                print(f"[rank0] validation step={step} end", flush=True)
                row = [step, epoch, round(float(loss), 4),
                       val["top1"], val["top3"], val["top10"],
                       val["top1_long"], val["top3_long"],
                       val["absent_mean_top1_sim"]]
                with open(log_path, "a", newline="",
                          encoding="utf-8") as f:
                    csv.writer(f).writerow(row)
                print(json.dumps({"step": step, "loss": float(loss),
                                  **{k: (None if v is None else round(v, 4))
                                     for k, v in val.items()}},
                                 ensure_ascii=False), flush=True)
                if val["top1"] is not None and val["top1"] > best_top1:
                    best_top1 = val["top1"]
                    torch.save(
                        head.module.state_dict(),
                        MODELS / f"{args.tag}_best.pt")
            elif rank == 0 and args.overfit and step % 25 == 0:
                print(json.dumps(
                    {"step": step, "loss": float(loss)}, ensure_ascii=False),
                    flush=True)
            if args.overfit and step >= total_steps:
                break
        if args.overfit:
            break
    torch.save(head.module.state_dict(), MODELS / f"{args.tag}_last.pt")
    if rank == 0:
        (MODELS / f"{args.tag}_config.json").write_text(json.dumps({
            "tag": args.tag, "epochs": args.epochs,
            "steps_per_epoch": args.steps_per_epoch,
            "batch_size": args.batch_size, "lr": args.lr,
            "tau": args.tau, "margin": args.margin,
            "w_margin": args.w_margin, "pos_frac": args.pos_frac,
            "seed": args.seed, "best_val_top1": best_top1,
            "gfn_checkpoint":
            "outputs/n18/checkpoints/gfn_cuhk_convnext_pytorch.pt",
            "train_split": "n15_frozen train30",
            "cal_split": "n15_frozen calibration10",
            "runtime_s": round(time.time() - t0, 1),
        }, indent=1))
    dist.destroy_process_group()
    if rank == 0:
        print("R0_TRAIN_DONE best_top1=", best_top1, flush=True)


if __name__ == "__main__":
    main()
