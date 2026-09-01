#!/usr/bin/env python
"""N22 offline gate for the Correction-Driven Compatible Identity Adapter.

CDCIA is deliberately small: a shared low-rank residual map is applied to
the frozen R0 coordinate for both the human root and the shadow tracklet.
Training uses the corrected target candidate and the strongest wrong
candidate as a margin pair, while a compatibility penalty limits movement of
the old coordinate system.  The evaluation split is sequence-disjoint.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.set_num_threads(4)


ROOT = Path(".")
OUT = ROOT / "outputs/n22"
D = 2048
K = 5


class CDCIA(nn.Module):
    """Shared compatible residual adapter in the frozen R0 space."""

    def __init__(self, dim=D, rank=16, residual_scale=0.5):
        super().__init__()
        self.dim = dim
        self.rank = rank
        self.residual_scale = residual_scale
        self.down = nn.Linear(dim, rank, bias=False)
        self.up = nn.Linear(rank, dim, bias=False)
        nn.init.kaiming_uniform_(self.down.weight, a=math.sqrt(5))
        nn.init.zeros_(self.up.weight)

    def forward(self, x):
        x = F.normalize(x, dim=-1)
        delta = self.up(self.down(x))
        return F.normalize(x + self.residual_scale * delta, dim=-1)

    def score(self, root, tracklet):
        return (self(root).unsqueeze(1) * self(tracklet)).sum(-1)


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_groups(path, horizon):
    z = np.load(path, allow_pickle=False)
    # NpzFile members are compressed.  Materialize each member once; doing
    # z["r0"][i] inside the loop would decompress the 120--230 MB member for
    # every candidate row.
    att_arr = z["att"]
    seq_arr = z["seq"]
    frame_arr = z["frame"]
    rank_arr = z["rank"]
    r0_arr = z["r0"]
    mask_arr = z["r0_mask"]
    root_arr = z["root_r0"]
    label_h_arr = z["label_by_h"]
    by = defaultdict(list)
    for i, att in enumerate(att_arr):
        by[str(att)].append(i)
    groups = []
    for att, idxs in by.items():
        idxs = sorted(idxs, key=lambda i: int(z["rank"][i]))
        by_rank = {int(rank_arr[i]): i for i in idxs}
        if not all(rank in by_rank for rank in range(1, K + 1)):
            continue
        idxs = [by_rank[rank] for rank in range(1, K + 1)]
        # A candidate with no matched R0 token has no identity evidence.  Do
        # not turn that missingness into a learnable zero identity; the raw
        # audit uses the same all-five-valid protocol.  The full dataset
        # stats still record the omitted/partial groups for live coverage.
        if any(mask_arr[i, :horizon].sum() <= 0 for i in idxs):
            continue
        y = 0
        labels = []
        for i in idxs:
            label = int(label_h_arr[i, horizon - 1])
            labels.append(label)
            if label and y == 0:
                y = int(z["rank"][i])
        groups.append({
            "att": att,
            "seq": str(seq_arr[idxs[0]]),
            "frame": int(frame_arr[idxs[0]]),
            "r0": r0_arr[idxs, :horizon].astype(np.float32),
            "mask": mask_arr[idxs, :horizon].astype(np.float32),
            "root": root_arr[idxs[0]].astype(np.float32),
            "labels": np.asarray(labels, dtype=np.int64),
            "y": y,
        })
    groups.sort(key=lambda g: (g["seq"], g["frame"], g["att"]))
    return groups


def batch(groups, device="cpu"):
    r0 = torch.from_numpy(np.stack([g["r0"] for g in groups])).to(device)
    mask = torch.from_numpy(np.stack([g["mask"] for g in groups])).to(device)
    root = torch.from_numpy(np.stack([g["root"] for g in groups])).to(device)
    return r0, mask, root


def tracklet_mean(r0, mask):
    x = (r0 * mask.unsqueeze(-1)).sum(2)
    x = x / mask.sum(2, keepdim=True).clamp(min=1.0)
    return F.normalize(x, dim=-1)


def scores_for(model, groups, device="cpu", batch_size=128):
    out = []
    model.eval() if model is not None else None
    with torch.inference_mode():
        for start in range(0, len(groups), batch_size):
            gs = groups[start:start + batch_size]
            r0, mask, root = batch(gs, device)
            cand = tracklet_mean(r0, mask)
            if model is None:
                s = (cand * F.normalize(root, dim=-1).unsqueeze(1)).sum(-1)
            else:
                s = model.score(root, cand)
            out.extend(s.cpu().numpy())
    return np.asarray(out, dtype=np.float32)


def auc(scores, labels):
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    pos = labels == 1
    neg = labels == 0
    if not pos.any() or not neg.any():
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    vals = scores[order]
    ranks = np.empty(len(scores), dtype=np.float64)
    left = 0
    while left < len(vals):
        right = left + 1
        while right < len(vals) and vals[right] == vals[left]:
            right += 1
        ranks[order[left:right]] = 0.5 * (left + 1 + right)
        left = right
    npos, nneg = int(pos.sum()), int(neg.sum())
    return float((ranks[pos].sum() - npos * (npos + 1) / 2) /
                 (npos * nneg))


def ranking_metrics(groups, scores):
    labels = np.concatenate([g["labels"] for g in groups])
    flat_scores = scores.reshape(-1)
    top1, avail, margins, errors = [], [], [], []
    for g, s in zip(groups, scores):
        order = np.argsort(-s, kind="stable")
        top1.append(int(g["labels"][order[0]] == 1))
        avail.append(int(g["labels"].any()))
        pos = s[g["labels"] == 1]
        neg = s[g["labels"] == 0]
        if len(pos) and len(neg):
            margins.append(float(pos.max() - neg.max()))
            errors.append(int(neg.max() >= pos.max()))
    return {
        "attempts": len(groups),
        "candidate_rows": int(len(labels)),
        "positive_candidates": int(labels.sum()),
        "auc": auc(flat_scores, labels),
        "top1": float(np.mean(top1)) if top1 else float("nan"),
        "candidate_pool_positive_rate": float(np.mean(avail)) if avail else float("nan"),
        "mean_hard_negative_margin": float(np.mean(margins)) if margins else float("nan"),
        "hard_negative_error_rate": float(np.mean(errors)) if errors else float("nan"),
    }


def decide(s, threshold, margin):
    order = np.argsort(-s, kind="stable")
    best = int(order[0])
    second = float(s[order[1]]) if len(order) > 1 else -1.0
    if float(s[best]) >= threshold and float(s[best] - second) >= margin:
        return best + 1
    return 0


def decision_metrics(groups, scores, threshold, margin):
    rows = []
    for g, s in zip(groups, scores):
        d = decide(s, threshold, margin)
        y = int(g["y"])
        rows.append((d, y))
    false = sum(int(d >= 1 and d != y) for d, y in rows)
    missed = sum(int(d == 0 and y >= 1) for d, y in rows)
    correct = sum(int(d >= 1 and d == y) for d, y in rows)
    return {
        "threshold": threshold,
        "margin": margin,
        "correct_commits": correct,
        "false_commits": false,
        "missed_commits": missed,
        "corrections": false + missed,
        "commit_precision": correct / max(1, correct + false),
        "commit_recall": correct / max(1, correct + missed),
    }


def loss_for(model, groups, compatibility, margin=0.08):
    r0, mask, root = batch(groups)
    cand = tracklet_mean(r0, mask)
    root_base = F.normalize(root, dim=-1)
    adapted_root = model(root_base)
    adapted_cand = model(cand.reshape(-1, D)).reshape(cand.shape)
    scores = (adapted_cand * adapted_root.unsqueeze(1)).sum(-1)
    y = torch.as_tensor([int(g["y"]) for g in groups], dtype=torch.long)
    pos_rows = y >= 1
    pos_idx = y.clamp(min=1) - 1
    pos = scores.gather(1, pos_idx[:, None]).squeeze(1)
    neg_mask = torch.ones_like(scores, dtype=torch.bool)
    neg_mask.scatter_(1, pos_idx[:, None], False)
    hard_neg = scores.masked_fill(~neg_mask, float("-inf")).max(1).values
    pair = F.softplus(margin - pos + hard_neg)
    positive_loss = pair + 0.25 * (1.0 - pos)
    none_loss = 0.5 * F.relu(scores.max(1).values - 0.10)
    rank_loss = torch.where(pos_rows, positive_loss, none_loss).mean()
    raw = torch.cat([root_base, cand.reshape(-1, D)], dim=0)
    adapted = model(raw)
    compat = ((adapted - raw) ** 2).mean()
    return rank_loss + compatibility * compat, rank_loss.detach(), compat.detach()


def train_adapter(groups, compatibility, epochs, lr, seed):
    seed_everything(seed)
    model = CDCIA()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        opt.zero_grad()
        total, rank_loss, compat_loss = loss_for(model, groups, compatibility)
        total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()
        if epoch == 1 or epoch % 10 == 0 or epoch == epochs:
            history.append({
                "epoch": epoch,
                "loss": float(total.item()),
                "rank_loss": float(rank_loss.item()),
                "compat_loss": float(compat_loss.item()),
            })
            print(
                f"CDCIA compat={compatibility} epoch={epoch} "
                f"loss={float(total.item()):.5f} rank={float(rank_loss.item()):.5f} "
                f"compat={float(compat_loss.item()):.5f}", flush=True)
    model.eval()
    return model, history


def online_update(model, entries, compatibility, epochs=2, lr=2e-3):
    if not entries:
        return
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()
    for _ in range(epochs):
        opt.zero_grad()
        total, _, _ = loss_for(model, [x[0] for x in entries], compatibility)
        total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()
    model.eval()


def stream_update_eval(model, groups, compatibility, threshold, margin):
    """Causal offline replay simulation; update only after each decision."""
    replay = []
    rows = []
    for g in groups:
        s = scores_for(model, [g])[0]
        d = decide(s, threshold, margin)
        y = int(g["y"])
        err = int((d >= 1 and d != y) or (d == 0 and y >= 1))
        rows.append({"att": g["att"], "decision": d, "y": y, "error": err})
        if not err:
            continue
        replay.append((g, y))
        if len(replay) > 32:
            replay = replay[-32:]
        online_update(model, replay, compatibility)
    false = sum(int(x["decision"] >= 1 and x["decision"] != x["y"])
                for x in rows)
    missed = sum(int(x["decision"] == 0 and x["y"] >= 1) for x in rows)
    correct = sum(int(x["decision"] >= 1 and x["decision"] == x["y"])
                  for x in rows)
    return {
        "attempts": len(rows), "false_commits": false,
        "missed_commits": missed, "corrections": false + missed,
        "correct_commits": correct,
        "online_updates": sum(x["error"] for x in rows),
    }, rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--horizon", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--margin", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=22)
    args = ap.parse_args()
    if args.horizon < 1 or args.horizon > 8:
        raise SystemExit("--horizon must be in [1, 8]")
    OUT.mkdir(parents=True, exist_ok=True)
    train = load_groups(OUT / "datasets/train30_aligned.npz", args.horizon)
    cal = load_groups(OUT / "datasets/cal10_aligned.npz", args.horizon)
    print(f"CDCIA_DATA train_groups={len(train)} cal_groups={len(cal)} "
          f"h={args.horizon}", flush=True)
    results = []
    raw_tr = scores_for(None, train)
    raw_cal = scores_for(None, cal)
    for name, groups, scores in [
            ("frozen_r0_train", train, raw_tr),
            ("frozen_r0_cal10", cal, raw_cal)]:
        row = {"method": name, "split": name.rsplit("_", 1)[-1],
               **ranking_metrics(groups, scores)}
        row.update(decision_metrics(groups, scores, args.threshold, args.margin))
        results.append(row)
    models = {}
    for name, compat in [("cdcia_no_compat", 0.0), ("cdcia_compat", 0.2)]:
        model, history = train_adapter(train, compat, args.epochs, args.lr,
                                        args.seed)
        models[name] = (model, compat)
        torch.save({"model": model.state_dict(), "dim": D, "rank": 16,
                    "residual_scale": 0.5, "horizon": args.horizon,
                    "compatibility": compat, "threshold": args.threshold,
                    "margin": args.margin, "history": history},
                   OUT / f"{name}.pt")
        for split, groups in [("train30", train), ("cal10", cal)]:
            scores = scores_for(model, groups).reshape(-1, K)
            row = {"method": name, "split": split,
                   **ranking_metrics(groups, scores)}
            row.update(decision_metrics(groups, scores, args.threshold,
                                        args.margin))
            results.append(row)
        sim_model = CDCIA()
        sim_model.load_state_dict(model.state_dict())
        sim_metrics, _ = stream_update_eval(
            sim_model, cal, compat, args.threshold, args.margin)
        results.append({"method": f"{name}_online_replay", "split": "cal10",
                        **sim_metrics})
    csv_path = OUT / "cdcia_offline_results.csv"
    fields = sorted({key for row in results for key in row})
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(results)
    summary = {
        "horizon": args.horizon,
        "train_groups": len(train),
        "cal10_groups": len(cal),
        "results": results,
        "method": "CDCIA shared low-rank R0 residual with correction margin and compatibility ablation",
        "offline_labels": "shadow trajectory correct fields; not used for ranking at inference",
        "output_csv": str(csv_path),
    }
    (OUT / "cdcia_offline_summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=True), encoding="utf-8")
    print(json.dumps(summary, indent=2, allow_nan=True), flush=True)
    print("N22_CDCIA_OFFLINE_DONE", flush=True)


if __name__ == "__main__":
    main()
