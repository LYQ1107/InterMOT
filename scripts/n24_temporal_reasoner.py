#!/usr/bin/env python3
"""N24 C0/C1/C2 causal temporal identity reasoners.

The offline source is the real N21 all-candidate shadow dataset (train30 and
sequence-disjoint cal10), padded to a 20-step interface.  Existing rows have
up to eight observed causal shadow frames; a separate N24 horizon-20 shadow
subset can be evaluated without being used for training.

Variants:
  C0: masked temporal mean prototype + frozen cosine identity score.
  C1: projected masked causal self-attention + root projection.
  C2: larger causal reasoner with mean/last/max temporal prototypes and
      candidate-competition hard-negative features.

This script is intentionally offline.  It does not mutate public identity
state, does not use GT for features, and never lets a token attend to a later
token or an invalid padded token.  FULL_LOOP deployment is implemented in a
separate runner after this offline gate.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from n24_temporal_diagnostic import SequenceCache, normalize
except ModuleNotFoundError:  # package-style import from the repository root
    from scripts.n24_temporal_diagnostic import SequenceCache, normalize

torch.set_num_threads(8)

ROOT = Path(".")
N21_DS = ROOT / "outputs/n21/tracklet_identity_dataset"
OUT = ROOT / "outputs/n24"
HMAX = 20
D_IN = 4096
K = 5


def load_n21_groups(path: Path):
    z = np.load(path)
    att_all = z["att"]
    rank_all = z["rank"]
    label_all = z["label"]
    vis_all = z["vis"]
    mask_all = z["vis_mask"]
    root_all = z["root"]
    by_att = defaultdict(list)
    h = min(HMAX, vis_all.shape[1])
    for i in range(len(att_all)):
        vis = np.zeros((HMAX, D_IN), dtype=np.float32)
        mask = np.zeros(HMAX, dtype=np.float32)
        vis[:h] = vis_all[i, :h].astype(np.float32)
        mask[:h] = mask_all[i, :h].astype(np.float32)
        by_att[str(att_all[i])].append({
            "rank": int(rank_all[i]),
            "label": int(label_all[i]),
            "vis": vis,
            "mask": mask,
            "root": root_all[i].astype(np.float32),
        })
    groups = []
    for att, rows in sorted(by_att.items()):
        rows.sort(key=lambda r: r["rank"])
        ranks = {r["rank"] for r in rows}
        if not all(i in ranks for i in range(1, K + 1)):
            continue
        y = next((r["rank"] for r in rows if r["label"]), 0)
        groups.append({"att": att, "rows": rows[:K], "y": int(y)})
    z.close()
    return groups


def load_shadow_groups(input_dir: Path):
    rows = []
    for path in sorted(input_dir.glob("*.jsonl")):
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    rows.append(json.loads(line))
    unique = {}
    for row in rows:
        key = (row["sequence"], int(row["frame"]), int(row["gid"]),
               int(row["candidate_rank"]))
        if key not in unique or len(row.get("frames", [])) > len(unique[key].get("frames", [])):
            unique[key] = row
    rows = list(unique.values())
    by_att = defaultdict(list)
    zcache = {}
    for row in rows:
        seq = str(row["sequence"])
        if seq not in zcache:
            zcache[seq] = SequenceCache(seq)
        z = zcache[seq]
        root = z.query(int(row["gid"]))
        if root is None:
            continue
        vis = np.zeros((HMAX, D_IN), dtype=np.float32)
        mask = np.zeros(HMAX, dtype=np.float32)
        f0 = int(row["frame"])
        for fr in row.get("frames", []):
            ff = int(fr["frame"])
            step = ff - f0
            if step < 0 or step >= HMAX:
                continue
            de = z.detection(ff, fr.get("box"))
            if de is None:
                continue
            vis[step] = np.concatenate([de[0], de[1]])
            mask[step] = 1.0
        by_att[f"{seq}:{f0}:{int(row['gid'])}"].append({
            "rank": int(row["candidate_rank"]),
            "label": int(row.get("is_correct", 0)),
            "vis": vis,
            "mask": mask,
            "root": np.concatenate([root[0], root[1]]).astype(np.float32),
        })
    groups = []
    for att, cand in sorted(by_att.items()):
        cand.sort(key=lambda r: r["rank"])
        ranks = {r["rank"] for r in cand}
        if not all(i in ranks for i in range(1, K + 1)):
            continue
        y = next((r["rank"] for r in cand if r["label"]), 0)
        groups.append({"att": att, "rows": cand[:K], "y": int(y)})
    return groups


def batch(groups, history=None):
    b = len(groups)
    x = np.zeros((b, K, HMAX, D_IN), dtype=np.float32)
    m = np.zeros((b, K, HMAX), dtype=np.float32)
    r = np.zeros((b, D_IN), dtype=np.float32)
    y = np.zeros(b, dtype=np.int64)
    for i, group in enumerate(groups):
        for j, row in enumerate(group["rows"][:K]):
            x[i, j] = row["vis"]
            m[i, j] = row["mask"]
        r[i] = group["rows"][0]["root"]
        y[i] = int(group["y"])
    if history is not None and history < HMAX:
        m[:, :, history:] = 0.0
    return (torch.from_numpy(x), torch.from_numpy(m),
            torch.from_numpy(r), torch.from_numpy(y))


class C0MeanPrototype(nn.Module):
    def __init__(self):
        super().__init__()
        self.logit_scale = nn.Parameter(torch.tensor(2.0))
        self.none_bias = nn.Parameter(torch.tensor(0.0))

    def forward(self, vis, mask):
        cand = (vis * mask.unsqueeze(-1)).sum(2) / mask.sum(2, keepdim=True).clamp(min=1.0)
        cand = F.normalize(cand, dim=-1)
        root = F.normalize(vis.new_zeros((vis.shape[0], D_IN)), dim=-1)
        # root is passed through the first frame slot by the wrapper below.
        raise RuntimeError("C0 requires forward_with_root")

    def forward_with_root(self, vis, mask, root):
        cand = (vis * mask.unsqueeze(-1)).sum(2) / mask.sum(2, keepdim=True).clamp(min=1.0)
        cand = F.normalize(cand, dim=-1)
        root = F.normalize(root, dim=-1)
        sims = (cand * root.unsqueeze(1)).sum(-1)
        scale = F.softplus(self.logit_scale) + 0.1
        return torch.cat([self.none_bias.expand(vis.shape[0], 1), scale * sims], dim=1)


class CausalBlock(nn.Module):
    def __init__(self, dim, heads, ff):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.ff = nn.Sequential(nn.Linear(dim, ff), nn.GELU(), nn.Linear(ff, dim))

    def forward(self, x, mask):
        _, t, _ = x.shape
        causal = torch.triu(torch.ones(t, t, dtype=torch.bool, device=x.device), diagonal=1)
        h = self.norm1(x)
        a, _ = self.attn(h, h, h, attn_mask=causal,
                          key_padding_mask=~mask.bool())
        x = x + a
        x = x + self.ff(self.norm2(x))
        # A padded query before the first valid observation has no legal
        # causal key, so PyTorch's softmax can produce NaN for that query.
        # Such query outputs are masked out of pooling and must not poison
        # the pooled valid observations or later blocks.
        return torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)


def compact_valid_tokens(x, mask):
    """Move observed tokens to a prefix while preserving causal order.

    Shadow caches can contain leading or interior missing observations.  A
    conventional causal attention mask gives a leading invalid query no
    legal key, which makes PyTorch's attention softmax undefined.  Compacting
    only the valid observations preserves their order, removes that numerical
    corner case, and does not introduce any future information.
    """
    b, t, d = x.shape
    time = torch.arange(t, device=x.device).view(1, t)
    order_key = (~mask.bool()).long() * t + time
    order = torch.argsort(order_key, dim=1)
    x = torch.gather(x, 1, order.unsqueeze(-1).expand(b, t, d))
    mask = torch.gather(mask.bool(), 1, order)
    return x, mask


class C1Temporal(nn.Module):
    def __init__(self, dim=128, layers=2):
        super().__init__()
        self.proj = nn.Sequential(nn.Linear(D_IN, dim), nn.LayerNorm(dim))
        self.root_proj = nn.Sequential(nn.Linear(D_IN, dim), nn.LayerNorm(dim))
        self.pos = nn.Parameter(torch.zeros(1, HMAX, dim))
        nn.init.normal_(self.pos, std=0.02)
        self.blocks = nn.ModuleList([CausalBlock(dim, 4, dim * 4) for _ in range(layers)])
        self.logit_scale = nn.Parameter(torch.tensor(2.0))
        self.none_bias = nn.Parameter(torch.tensor(0.0))

    def encode(self, vis, mask):
        b, k, t, _ = vis.shape
        x = self.proj(vis.reshape(b * k, t, D_IN))
        mm = mask.reshape(b * k, t).bool()
        x, mm = compact_valid_tokens(x, mm)
        empty = ~mm.any(dim=1)
        if bool(empty.any()):
            mm[empty, 0] = True
        x = x + self.pos[:, :t]
        for block in self.blocks:
            x = block(x, mm)
        valid = mm.unsqueeze(-1).float()
        mean = (x * valid).sum(1) / valid.sum(1).clamp(min=1.0)
        last_idx = mm.long().sum(1).clamp(min=1) - 1
        last = x[torch.arange(b * k, device=x.device), last_idx]
        return (mean.reshape(b, k, -1), last.reshape(b, k, -1))

    def forward_with_root(self, vis, mask, root):
        mean, _ = self.encode(vis, mask)
        z = F.normalize(mean, dim=-1)
        r = F.normalize(self.root_proj(root), dim=-1)
        sims = (z * r.unsqueeze(1)).sum(-1)
        scale = F.softplus(self.logit_scale) + 0.1
        return torch.cat([self.none_bias.expand(vis.shape[0], 1), scale * sims], dim=1)


class C2MultiPrototype(C1Temporal):
    def __init__(self):
        super().__init__(dim=192, layers=3)
        self.pool_logits = nn.Parameter(torch.zeros(3))
        self.competition = nn.Sequential(
            nn.Linear(2 * 192 + 1, 128), nn.GELU(), nn.Linear(128, 1))
        self.none_head = nn.Sequential(nn.Linear(192 + 192, 64), nn.GELU(), nn.Linear(64, 1))

    def encode(self, vis, mask):
        b, k, t, _ = vis.shape
        x = self.proj(vis.reshape(b * k, t, D_IN))
        mm = mask.reshape(b * k, t).bool()
        x, mm = compact_valid_tokens(x, mm)
        empty = ~mm.any(dim=1)
        if bool(empty.any()):
            mm[empty, 0] = True
        x = x + self.pos[:, :t]
        for block in self.blocks:
            x = block(x, mm)
        valid = mm.unsqueeze(-1).float()
        mean = (x * valid).sum(1) / valid.sum(1).clamp(min=1.0)
        last_idx = mm.long().sum(1).clamp(min=1) - 1
        last = x[torch.arange(b * k, device=x.device), last_idx]
        neg_inf = torch.finfo(x.dtype).min
        maxv = x.masked_fill(~mm.unsqueeze(-1), neg_inf).max(1).values
        alpha = torch.softmax(self.pool_logits, dim=0)
        proto = alpha[0] * mean + alpha[1] * last + alpha[2] * maxv
        return proto.reshape(b, k, -1), last.reshape(b, k, -1)

    def forward_with_root(self, vis, mask, root):
        z, _ = self.encode(vis, mask)
        z = F.normalize(z, dim=-1)
        r = F.normalize(self.root_proj(root), dim=-1)
        sims = (z * r.unsqueeze(1)).sum(-1)
        same = z @ z.transpose(1, 2)
        same = same.masked_fill(torch.eye(K, device=z.device, dtype=torch.bool)[None], -1.0)
        other_max = same.max(-1).values.unsqueeze(-1)
        delta = self.competition(torch.cat([z, r.unsqueeze(1).expand_as(z), other_max], dim=-1)).squeeze(-1)
        scale = F.softplus(self.logit_scale) + 0.1
        cand = scale * sims + 0.2 * delta
        group = z.mean(1)
        none = self.none_head(torch.cat([group, r], dim=-1)).squeeze(-1)
        return torch.cat([none.unsqueeze(1), cand], dim=1)


def forward_model(model, x, m, r):
    return model.forward_with_root(x, m, r)


def train_model(model, groups, epochs, lr, none_weight):
    x, m, r, y = batch(groups)
    counts = torch.bincount(y, minlength=K + 1).float()
    weights = torch.ones(K + 1)
    weights[0] = none_weight
    for j in range(1, K + 1):
        if counts[j] > 0:
            weights[j] = max(1.0, float(counts[0] / counts[j]))
    lossf = nn.CrossEntropyLoss(weight=weights)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    history = []
    for ep in range(epochs):
        model.train()
        opt.zero_grad()
        logits = forward_model(model, x, m, r)
        loss = lossf(logits, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()
        if ep == 0 or (ep + 1) % max(1, epochs // 5) == 0:
            history.append({"epoch": ep + 1, "loss": float(loss.item())})
    model.eval()
    return history


def evaluate(model, groups, split, history, source):
    x, m, r, y = batch(groups, history=history)
    model.eval()
    with torch.inference_mode():
        p = torch.softmax(forward_model(model, x, m, r), dim=1).numpy()
    yy = y.numpy()
    decisions = p.argmax(1)
    correct = int(np.sum((decisions == yy)))
    false = int(np.sum((decisions >= 1) & (decisions != yy)))
    miss = int(np.sum((decisions == 0) & (yy >= 1)))
    pos = yy >= 1
    candidate_top1 = []
    for i in np.where(pos)[0]:
        candidate_top1.append(int(p[i, 1:].argmax() + 1 == yy[i]))
    margins = []
    for i in range(len(p)):
        margins.append(float(p[i, 1:].max() - p[i, 0]))
    return {
        "source": source, "split": split, "history": history,
        "groups": len(yy), "none_targets": int(np.sum(yy == 0)),
        "exact_decision_acc": correct / max(1, len(yy)),
        "false_commits": false, "missed_commits": miss,
        "commit_precision": float(np.sum((decisions >= 1) & (decisions == yy)) /
                                  max(1, np.sum(decisions >= 1))),
        "target_present_candidate_top1": float(np.mean(candidate_top1)) if candidate_top1 else 0.0,
        "mean_candidate_minus_none_prob": float(np.mean(margins)) if margins else 0.0,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--none-weight", type=float, default=1.0)
    ap.add_argument("--shadow20-dir", default="outputs/n20/n24_diag_shadow20")
    ap.add_argument("--out-dir", default="outputs/n24/models")
    ap.add_argument("--metrics-out", default="outputs/n24/n24_temporal_reasoner_metrics.csv")
    args = ap.parse_args()
    torch.manual_seed(24)
    np.random.seed(24)
    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    tr = load_n21_groups(N21_DS / "train30.npz")
    cal = load_n21_groups(N21_DS / "cal10.npz")
    shadow_dir = Path(args.shadow20_dir)
    if not shadow_dir.is_absolute():
        shadow_dir = ROOT / shadow_dir
    shadow = load_shadow_groups(shadow_dir) if shadow_dir.exists() else []
    print(json.dumps({"train_groups": len(tr), "cal_groups": len(cal),
                      "shadow20_groups": len(shadow)}, indent=2), flush=True)
    models = {
        "C0": C0MeanPrototype(),
        "C1": C1Temporal(),
        "C2": C2MultiPrototype(),
    }
    all_metrics = []
    for name, model in models.items():
        hist = train_model(model, tr, args.epochs, args.lr, args.none_weight)
        ckpt = out_dir / f"n24_{name}.pt"
        torch.save({"name": name, "hmax": HMAX, "d_in": D_IN,
                    "state_dict": model.state_dict(), "train_groups": len(tr),
                    "epochs": args.epochs, "history": hist}, ckpt)
        for source, groups in (("cal10_n21_h8", cal), ("shadow20_real", shadow)):
            if not groups:
                continue
            for h in (1, 5, 10, 20):
                row = evaluate(model, groups, source, h, name)
                all_metrics.append(row)
                print(json.dumps({"model": name, **row}), flush=True)
    out_path = Path(args.metrics_out)
    if not out_path.is_absolute():
        out_path = ROOT / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if all_metrics:
        with out_path.open("w", newline="", encoding="utf-8") as handle:
            fields = list(all_metrics[0].keys())
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(all_metrics)
    meta = {"train_groups": len(tr), "cal_groups": len(cal),
            "shadow20_groups": len(shadow), "hmax": HMAX,
            "models": list(models), "metrics": str(out_path),
            "training_label": "N21 final shadow-frame label; GT only offline",
            "calibration_caveat": "N21 cal10 has at most eight observed shadow frames; H=10/20 are padded there",
            "shadow20_caveat": "real horizon-20 subset is sequence-disjoint only by source run, not used in training"}
    (out_path.parent / "n24_temporal_reasoner_meta.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps(meta, indent=2), flush=True)
    print("N24_TEMPORAL_REASONER_DONE", flush=True)


if __name__ == "__main__":
    main()
