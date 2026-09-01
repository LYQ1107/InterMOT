#!/usr/bin/env python
"""N21 Phase-II (CATIL): correction-adaptable tracklet identity learning.

Model: per-candidate visual tracklet embeddings (GFN 2048-d + R0 2048-d per
shadow frame, H=8) -> projection -> causal 2-layer Transformer -> tracklet
embedding z_k; identity agreement = cosine(z_k, root_query); K+1/NONE
decision. 22-d engineered features are kept as the Phase-I baseline and can
be fused later; this script evaluates the genuine visual representation
path.

Capacity ladder (online trainable parameters):
  C0_head  : logit scale + none bias + per-candidate linear adjust (tiny)
  C1_lora  : LoRA(r=16) on QKV of both transformer layers (~50K)
  C2_partial_ft : projection + transformer + root projection (~3.5M)

Offline training uses train30 only (GT = offline labels). The cal10 stream
is used for chronological correction-driven online adaptation with
episodic reset, causal updates, replay, KL-to-base regularization, and the
same legal human-supervision protocol as Phase-I.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.set_num_threads(4)
import csv as _csv

ROOT = Path(".")
sys.path.insert(0, str(ROOT))
from sam3_intermot.n21.human_supervision_ledger import (  # noqa: E402
    CorrectionRecord, HumanSupervisionLedger)
from train_n20_kplus1 import NUMERIC, SharedGRUSet, load_groups as kg_load_groups, groups_to_tensors  # noqa: E402

N21 = ROOT / "outputs/n21"
N20 = ROOT / "outputs/n20"
DS = N21 / "tracklet_identity_dataset"
H = 8
D_IN = 4096
D = 256
AUX_D = len(NUMERIC) + 2
DEVICE = torch.device("cpu")


class LoRALinear(nn.Module):
    def __init__(self, linear, rank=16, alpha=16.0):
        super().__init__()
        self.linear = linear
        self.rank = rank
        self.alpha = alpha
        dev = linear.weight.device
        self.lora_a = nn.Parameter(
            torch.zeros(linear.in_features, rank, device=dev))
        self.lora_b = nn.Parameter(
            torch.zeros(rank, linear.out_features, device=dev))
        nn.init.kaiming_uniform_(self.lora_a, a=math.sqrt(5))
        nn.init.zeros_(self.lora_b)

    def forward(self, x):
        base = self.linear(x)
        lora = (self.alpha / self.rank) * (
            x @ self.lora_a @ self.lora_b)
        return base + lora


class TransformerBlock(nn.Module):
    def __init__(self, d, nhead=4, ff=512, lora_rank=0):
        super().__init__()
        self.d = d
        self.nhead = nhead
        self.norm1 = nn.LayerNorm(d)
        self.norm2 = nn.LayerNorm(d)
        qkv = nn.Linear(d, 3 * d)
        if lora_rank > 0:
            qkv = LoRALinear(qkv, lora_rank)
        self.qkv = qkv
        self.out_proj = nn.Linear(d, d)
        self.ff1 = nn.Linear(d, ff)
        self.ff2 = nn.Linear(ff, d)

    def forward(self, x, mask):
        # x: (B, T, D); mask: (B, T) 1=valid
        B, T, _ = x.shape
        mb = mask.bool()
        residual = x
        h = self.norm1(x)
        qkv = self.qkv(h.clone())
        q, k, v = qkv.chunk(3, dim=-1)
        q = q.reshape(B, T, self.nhead, self.d // self.nhead).transpose(1, 2)
        k = k.reshape(B, T, self.nhead, self.d // self.nhead).transpose(1, 2)
        v = v.reshape(B, T, self.nhead, self.d // self.nhead).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.d // self.nhead)
        causal = torch.triu(torch.ones(T, T, dtype=torch.bool,
                                       device=x.device), diagonal=1)
        att = att.masked_fill(causal[None, None], float("-inf"))
        att = att.masked_fill(
            (~mb)[:, None, :, None].expand(B, self.nhead, T, T),
            float("-inf"))
        att = torch.softmax(att, dim=-1)
        att = torch.nan_to_num(att, nan=0.0)
        out = (att @ v).transpose(1, 2).reshape(B, T, self.d)
        out = self.out_proj(out)
        x = residual + out
        residual = x
        h = self.norm2(x)
        h = self.ff2(F.relu(self.ff1(h)))
        x = residual + h
        return x


class TrackletIdentityModel(nn.Module):
    def __init__(self, lora_rank=0, layers=2):
        super().__init__()
        self.lora_rank = lora_rank
        self.proj = nn.Sequential(nn.Linear(D_IN, D), nn.LayerNorm(D))
        self.root_proj = nn.Sequential(nn.Linear(D_IN, D), nn.LayerNorm(D))
        self.blocks = nn.ModuleList(
            [TransformerBlock(D, lora_rank=lora_rank) for _ in range(layers)])
        self.logit_scale = nn.Parameter(torch.tensor(-4.0))
        self.none_bias = nn.Parameter(torch.zeros(1))
        self.delta_mlp = nn.Sequential(
            nn.Linear(D, 64), nn.ReLU(), nn.Linear(64, 1))

    def tracklet_emb(self, vis, vis_mask):
        # vis: (B, H, D_IN), vis_mask: (B, H)
        B, T, _ = vis.shape
        x = self.proj(vis)
        for blk in self.blocks:
            x = blk(x, vis_mask)
        z = (x * vis_mask.unsqueeze(-1)).sum(1) / (
            vis_mask.sum(1, keepdim=True).clamp(min=1))
        return z

    def forward(self, vis, vis_mask, root, aux):
        # per-attempt with K candidates: vis (B,K,H,D_IN)
        B, K, T, _ = vis.shape
        tok, root_tok, raw_k, raw_root = self.encode(vis, vis_mask, root)
        return self.forward_tokens(tok, vis_mask, root_tok, raw_k, raw_root,
                                   aux)

    def encode(self, vis, vis_mask, root):
        B, K, T, _ = vis.shape
        tok = self.proj(vis.reshape(B * K, T, D_IN)).reshape(B, K, T, D)
        root_tok = self.root_proj(root)
        raw_k = self.raw_mean(vis, vis_mask)
        raw_root = root / (torch.norm(root, dim=1, keepdim=True) + 1e-8)
        return tok, root_tok, raw_k, raw_root

    def raw_mean(self, vis, vis_mask):
        B, K, T, _ = vis.shape
        z = (vis * vis_mask.unsqueeze(-1)).sum(2) / (
            vis_mask.sum(2, keepdim=True).clamp(min=1))
        z = z / (torch.norm(z, dim=-1, keepdim=True) + 1e-8)
        return z

    def forward_tokens(self, tok, vis_mask, root_tok, raw_k, raw_root, aux):
        B, K, T, _ = tok.shape
        z = self.tracklet_emb_tokens(tok, vis_mask)
        r = root_tok
        cos_raw = F.cosine_similarity(raw_k, raw_root.unsqueeze(1), dim=-1)
        delta = torch.tanh(self.delta_mlp(z).squeeze(-1)) * 0.5
        p0 = aux[:, :, -2].clamp(min=1e-6)
        pk = aux[:, :, -1].clamp(min=1e-6)
        base_cand = torch.log(pk)
        base_none = torch.log(p0[:, 0])
        visual_scale = F.softplus(self.logit_scale) + 0.1
        scores = base_cand + visual_scale * (cos_raw + delta)
        none = base_none + self.none_bias
        out = torch.cat([none.unsqueeze(1), scores], dim=1)
        return out, z, cos_raw

    def tracklet_emb_tokens(self, tok, vis_mask):
        B, K, T, _ = tok.shape
        x = tok.reshape(B * K, T, D)
        for blk in self.blocks:
            x = blk(x, vis_mask.reshape(B * K, T))
        z = (x * vis_mask.reshape(B * K, T, 1)).sum(1) / (
            vis_mask.reshape(B * K, T).sum(1, keepdim=True).clamp(min=1))
        return z.reshape(B, K, D)

    def tracklet_emb(self, vis, vis_mask):
        B, T, _ = vis.shape
        x = self.proj(vis)
        for blk in self.blocks:
            x = blk(x, vis_mask)
        z = (x * vis_mask.unsqueeze(-1)).sum(1) / (
            vis_mask.sum(1, keepdim=True).clamp(min=1))
        return z

    def n_trainable(self):
        return sum(int(p.numel()) for p in self.parameters()
                   if p.requires_grad)

    def n_params(self):
        return sum(int(p.numel()) for p in self.parameters())


def set_mode(model, mode):
    for p in model.parameters():
        p.requires_grad_(False)
    if mode == "C0_head":
        for p in model.logit_scale, model.none_bias:
            p.requires_grad_(True)
        for p in model.delta_mlp.parameters():
            p.requires_grad_(True)
    elif mode == "C1_lora":
        for blk in model.blocks:
            if not isinstance(blk.qkv, LoRALinear):
                rank = getattr(model, "lora_rank", 16) or 16
                blk.qkv = LoRALinear(blk.qkv, rank)
                model.lora_rank = rank
            for p in blk.qkv.lora_a, blk.qkv.lora_b:
                p.requires_grad_(True)
        for p in model.logit_scale, model.none_bias:
            p.requires_grad_(True)
        for p in model.delta_mlp.parameters():
            p.requires_grad_(True)
    elif mode == "C2_partial_ft":
        # partial fine-tune of the temporal identity module + root
        # projection + head; the per-frame projection stays frozen as the
        # visual backbone extractor.
        for p in model.blocks.parameters():
            p.requires_grad_(True)
        for p in model.root_proj.parameters():
            p.requires_grad_(True)
        for p in model.logit_scale, model.none_bias:
            p.requires_grad_(True)
        for p in model.delta_mlp.parameters():
            p.requires_grad_(True)
    else:
        raise ValueError(mode)
    return model


def load_groups(npz_path):
    loaded = np.load(npz_path)
    d = {k: loaded[k] for k in loaded.files}
    by_att = defaultdict(list)
    for i in range(len(d["att"])):
        by_att[str(d["att"][i])].append({
            "seq": str(d["seq"][i]),
            "frame": int(d["frame"][i]),
            "gid": int(d["gid"][i]),
            "rank": int(d["rank"][i]),
            "label": int(d["label"][i]),
            "vis": d["vis"][i].astype(np.float32),
            "vis_mask": d["vis_mask"][i].astype(np.float32),
            "root": d["root"][i].astype(np.float32),
        })
    groups = []
    for att, rows in by_att.items():
        rows = sorted(rows, key=lambda r: r["rank"])
        y = 0
        for r in rows:
            if r["label"]:
                y = r["rank"]
                break
        groups.append({"att": att, "rows": rows, "y": y})
    groups.sort(key=lambda g: int(g["att"].split(":")[1]))
    return groups


def compute_gru_probs(csv_path):
    bundle = torch.load(N20 / "models/kplus1_gru.pt",
                        map_location="cpu")
    feats = bundle["feature_cols"]
    mu = bundle["mu"].numpy().astype(np.float32)
    sd = bundle["sd"].numpy().astype(np.float32) + 1e-8
    groups = kg_load_groups(csv_path, 5, feats)
    full = [g for g in groups
            if all(k in {k for k, *_ in g} for k in range(1, 6))]
    X, M, _ = groups_to_tensors(full, 5, len(feats), 5)
    X = (X - torch.as_tensor(mu)) / torch.as_tensor(sd)
    model = SharedGRUSet(len(feats))
    model.load_state_dict(bundle["model"])
    model.eval()
    with torch.inference_mode():
        P = torch.softmax(model(X, M), 1).numpy()
    return {g[0][4]: P[i] for i, g in enumerate(full)}


def load_aux_map(csv_path, gru_probs):
    bundle = torch.load(N20 / "models/kplus1_gru.pt",
                        map_location="cpu")
    mu = bundle["mu"].numpy().astype(np.float32)
    sd = bundle["sd"].numpy().astype(np.float32) + 1e-8
    acc = {}
    n = {}
    with csv_path.open(newline="", encoding="utf-8") as f:
        for r in _csv.DictReader(f):
            key = (r["attempt"], int(r["candidate_rank"]))
            vec = np.asarray(
                [float(r[c]) if r[c] not in ("", None) else 0.0
                 for c in NUMERIC], dtype=np.float32)
            vec = np.nan_to_num(vec, nan=0.0)
            vec = (vec - mu) / sd
            pr = gru_probs.get(r["attempt"])
            if pr is not None:
                vec = np.concatenate([
                    vec, np.asarray([pr[0], pr[int(r["candidate_rank"])]],
                                    dtype=np.float32)])
            else:
                vec = np.concatenate([vec, np.zeros(2, dtype=np.float32)])
            if key not in acc:
                acc[key] = np.zeros(AUX_D, dtype=np.float32)
                n[key] = 0
            acc[key] += vec
            n[key] += 1
    return {k: acc[k] / max(1, n[k]) for k in acc}


def load_unified(csv_path, npz_path):
    gru_probs = compute_gru_probs(csv_path)
    aux = load_aux_map(csv_path, gru_probs)
    vis_groups = load_groups(npz_path)
    vis_by = {g["att"]: g["rows"] for g in vis_groups}
    groups = []
    for att, rows in vis_by.items():
        y = 0
        for r in rows:
            if r["label"]:
                y = r["rank"]
                break
        rows = sorted(rows, key=lambda r: r["rank"])
        if len(rows) < 5:
            continue
        ranks = {r["rank"] for r in rows}
        if not all(k in ranks for k in range(1, 6)):
            continue
        out_rows = []
        ok = True
        for r in rows:
            a = aux.get((att, r["rank"]))
            if a is None:
                ok = False
                break
            out_rows.append({**r, "aux": a})
        if not ok:
            continue
        groups.append({"att": att, "rows": out_rows, "y": y})
    groups.sort(key=lambda g: int(g["att"].split(":")[1]))
    return groups


def to_batch(groups, maxk=5):
    B = len(groups)
    vis = np.zeros((B, maxk, H, D_IN), dtype=np.float32)
    vm = np.zeros((B, maxk, H), dtype=np.float32)
    root = np.zeros((B, D_IN), dtype=np.float32)
    aux = np.zeros((B, maxk, AUX_D), dtype=np.float32)
    y = np.zeros(B, dtype=np.int64)
    for i, g in enumerate(groups):
        for j, r in enumerate(g["rows"][:maxk]):
            vis[i, j] = r["vis"][:H]
            vm[i, j] = r["vis_mask"][:H]
            aux[i, j] = r["aux"]
        root[i] = g["rows"][0]["root"]
        y[i] = g["y"] if g["y"] <= maxk else 0
    return (torch.from_numpy(vis).to(DEVICE),
            torch.from_numpy(vm).to(DEVICE),
            torch.from_numpy(root).to(DEVICE),
            torch.from_numpy(y).to(DEVICE),
            torch.from_numpy(aux).to(DEVICE))


def full_top5(groups):
    out = []
    for g in groups:
        ranks = {r["rank"] for r in g["rows"]}
        if all(k in ranks for k in range(1, 6)):
            out.append(g)
    return out


def decide(probs, threshold, margin):
    best = int(np.argmax(probs))
    if best >= 1 and probs[best] >= threshold:
        others = np.delete(probs, best)
        if probs[best] - others.max() >= margin:
            return best
    return 0


def offline_train(model, tr_groups, cal_groups, epochs, lr=1e-4,
                  none_weight=0.5):
    Xtr, Mtr, Rtr, Ytr, Atr = to_batch(tr_groups)
    Xcal, Mcal, Rcal, Ycal, Acal = to_batch(full_top5(cal_groups))
    counts = torch.bincount(Ytr, minlength=6).float()
    w = torch.ones(6)
    w = w.to(DEVICE)
    w[0] = none_weight
    for j in range(1, 6):
        if counts[j] > 0:
            w[j] = max(1.0, float(counts[0] / counts[j]))
    lossf = nn.CrossEntropyLoss(weight=w)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    rows = []
    for ep in range(epochs):
        model.train()
        opt.zero_grad()
        logits, _, _ = model(Xtr, Mtr, Rtr, Atr)
        loss = lossf(logits, Ytr)
        loss.backward()
        opt.step()
        if (ep + 1) % 10 == 0:
            model.eval()
            with torch.inference_mode():
                p_cal, _, _ = model(Xcal, Mcal, Rcal, Acal)
            acc = float((p_cal.argmax(1) == Ycal).float().mean())
            rows.append({"epoch": ep + 1,
                         "train_loss": round(float(loss.item()), 4),
                         "cal10_set_acc": round(acc, 4)})
            print(f"OFFLINE ep={ep + 1} loss={float(loss.item()):.4f} "
                  f"cal10_acc={acc:.4f}", flush=True)
    model.eval()
    return rows


def evaluate(model, groups, threshold=0.5, margin=0.05):
    X, M, R, Y, A = to_batch(groups)
    with torch.inference_mode():
        logits, z, cos = model(X, M, R, A)
    P = torch.softmax(logits, 1).cpu().numpy()
    y = Y.cpu().numpy()
    pred = np.argmax(P, axis=1)
    rows = []
    for i, g in enumerate(groups):
        d = decide(P[i], threshold, margin)
        rows.append({"att": g["att"], "seq": g["rows"][0]["seq"],
                     "ytrue": int(y[i]), "decision": int(d),
                     "p_none": float(P[i, 0]),
                     "p_best": float(P[i, 1:].max())})
    correct_commit = sum(1 for r in rows if r["decision"] == r["ytrue"]
                         and r["decision"] >= 1)
    false_commit = sum(1 for r in rows if r["decision"] >= 1
                       and r["decision"] != r["ytrue"])
    missed = sum(1 for r in rows if r["decision"] == 0 and r["ytrue"] >= 1)
    return {
        "attempts": len(rows),
        "set_acc": float((pred == y).mean()),
        "top1_correct_candidate_acc": float(
            sum(1 for i, g in enumerate(groups)
                if g["y"] >= 1 and pred[i] == g["y"]) /
            max(1, sum(1 for g in groups if g["y"] >= 1))),
        "correct_commits": correct_commit,
        "false_commits": false_commit,
        "missed_commits": missed,
        "corrections": false_commit + missed,
    }, rows


def run_stream(model, groups_seq, threshold=0.5, margin=0.05,
               online=False, mode="C0_head", lr=1e-4, epochs=5,
               replay=32, kl_lambda=2.0, margin_rank=0.2,
               frozen_ref=None, ledger=None):
    X_all, M_all, R_all, Y_all, A_all = to_batch(groups_seq)
    ytrue = Y_all.cpu().numpy()
    with torch.inference_mode():
        Tok_all, _, RawK_all, RawRoot_all = model.encode(X_all, M_all, R_all)
        Rtok_all = model.root_proj(R_all)
        P_all = []
        chunk = 64
        for s in range(0, len(groups_seq), chunk):
            e = min(s + chunk, len(groups_seq))
            P_all.append(torch.softmax(
                model.forward_tokens(Tok_all[s:e], M_all[s:e],
                                     Rtok_all[s:e], RawK_all[s:e],
                                     RawRoot_all[s:e], A_all[s:e])[0],
                1).cpu().numpy())
        P_all = np.concatenate(P_all, 0)
    rows = []
    diag = []
    replay_idx = []
    replay_negs = []
    corrected_neg_ranks = set()
    corrected_pos_ranks = set()
    opt = None
    if online:
        opt = torch.optim.Adam(
            [p for p in model.parameters() if p.requires_grad], lr=lr)
        lossf = nn.CrossEntropyLoss()
    for i, g in enumerate(groups_seq):
        p = P_all[i]
        d = decide(p, threshold, margin)
        y = int(ytrue[i])
        err_type = None
        if d >= 1 and d != y:
            err_type = "FALSE_COMMIT"
        elif d == 0 and y >= 1:
            err_type = "MISSED_COMMIT"
        same_rank_repeat = d in corrected_neg_ranks
        same_id_new_neg = (err_type == "FALSE_COMMIT"
                           and y in corrected_pos_ranks and d != y)
        rows.append({
            "att": g["att"], "seq": g["rows"][0]["seq"],
            "ytrue": y, "decision": d, "error_type": err_type or "OK",
            "p_none": round(float(p[0]), 4),
            "p_best": round(float(p[1:].max()), 4),
            "same_rank_repeat": int(same_rank_repeat),
            "same_id_new_neg": int(same_id_new_neg),
        })
        if err_type is None:
            continue
        target = y
        negs = []
        if err_type == "FALSE_COMMIT":
            if y >= 1:
                negs = [d]
            else:
                target = 0
                negs = [d]
        rec = CorrectionRecord(
            timestamp=datetime.now(timezone.utc).isoformat(),
            sequence=g["rows"][0]["seq"],
            frame=g["rows"][0]["frame"], public_id=-1,
            correction_type="ID_WRONG" if err_type == "FALSE_COMMIT"
            else "MISS",
            positive={"candidate_rank": y} if y >= 1 else None,
            explicit_negatives=[{"candidate_rank": r} for r in negs],
            source="causal", provenance="causal", gt_used=False,
            extra={"att": g["att"]},
        )
        if ledger is not None:
            ledger.record(rec)
        if online:
            replay_idx.append(i)
            replay_negs.append(negs)
            if d in negs:
                corrected_neg_ranks.add(d)
            if y >= 1:
                corrected_pos_ranks.add(y)
            if len(replay_idx) > replay:
                replay_idx = replay_idx[-replay:]
                replay_negs = replay_negs[-replay:]
            t0 = time.time()
            with torch.inference_mode():
                lpre, zpre, cpre = model.forward_tokens(
                    Tok_all[i:i + 1], M_all[i:i + 1], Rtok_all[i:i + 1],
                    RawK_all[i:i + 1], RawRoot_all[i:i + 1],
                    A_all[i:i + 1])
            model.train()
            for _ in range(epochs):
                opt.zero_grad()
                logits, _, _ = model.forward_tokens(
                    Tok_all[replay_idx], M_all[replay_idx],
                    model.root_proj(R_all[replay_idx]),
                    RawK_all[replay_idx], RawRoot_all[replay_idx],
                    A_all[replay_idx])
                yr = torch.from_numpy(ytrue[replay_idx]).to(DEVICE)
                loss = lossf(logits, yr)
                if margin_rank > 0:
                    margins = []
                    for j, negs_j in enumerate(replay_negs):
                        pos = int(yr[j].item())
                        if pos <= 0:
                            continue
                        for rn in negs_j:
                            margins.append(torch.clamp(
                                margin_rank -
                                (logits[j, pos] - logits[j, rn]), min=0.0))
                    if margins:
                        loss = loss + 0.5 * torch.stack(margins).mean()
                if kl_lambda > 0 and frozen_ref is not None:
                    with torch.inference_mode():
                        base_logits, _, _ = frozen_ref.forward_tokens(
                            Tok_all[replay_idx], M_all[replay_idx],
                            frozen_ref.root_proj(R_all[replay_idx]),
                            RawK_all[replay_idx], RawRoot_all[replay_idx],
                            A_all[replay_idx])
                    kl = F.kl_div(
                        F.log_softmax(logits, 1),
                        F.softmax(base_logits, 1), reduction="batchmean")
                    loss = loss + kl_lambda * kl
                loss.backward()
                opt.step()
            model.eval()
            upd_s = time.time() - t0
            with torch.inference_mode():
                lpost, zpost, cpost = model.forward_tokens(
                    Tok_all[i:i + 1], M_all[i:i + 1], Rtok_all[i:i + 1],
                    RawK_all[i:i + 1], RawRoot_all[i:i + 1],
                    A_all[i:i + 1])
            pos_idx = y if y >= 1 else 0
            neg_idx = d if d >= 1 else None
            diag.append({
                "att": g["att"],
                "pre_pos_logit": round(float(lpre[0, pos_idx]), 4),
                "post_pos_logit": round(float(lpost[0, pos_idx]), 4),
                "pre_neg_logit": round(float(lpre[0, neg_idx]), 4)
                if neg_idx else "",
                "post_neg_logit": round(float(lpost[0, neg_idx]), 4)
                if neg_idx else "",
                "embedding_l2_delta": round(
                    float(torch.norm(zpost - zpre)), 4),
                "cos_pos_delta": round(
                    float(cpost[0, pos_idx - 1] - cpre[0, pos_idx - 1]), 4)
                if pos_idx >= 1 else "",
                "update_s": round(upd_s, 4),
                "n_replay": len(replay_idx),
            })
            if i + 1 < len(groups_seq):
                with torch.inference_mode():
                    for s in range(i + 1, len(groups_seq), chunk):
                        e = min(s + chunk, len(groups_seq))
                        Rtok_all[s:e] = model.root_proj(R_all[s:e])
                        P_all[s:e] = torch.softmax(
                            model.forward_tokens(
                                Tok_all[s:e], M_all[s:e],
                                Rtok_all[s:e], RawK_all[s:e],
                                RawRoot_all[s:e], A_all[s:e])[0],
                            1).cpu().numpy()
    return rows, diag


def summarize(rows):
    errs = [r for r in rows if r["error_type"] != "OK"]
    fc = [r for r in rows if r["error_type"] == "FALSE_COMMIT"]
    mc = [r for r in rows if r["error_type"] == "MISSED_COMMIT"]
    return {
        "attempts": len(rows),
        "corrections": len(errs),
        "false_commits": len(fc),
        "missed_commits": len(mc),
        "error_rate": round(len(errs) / max(1, len(rows)), 4),
        "same_rank_repeats": sum(
            1 for r in rows
            if r["error_type"] == "FALSE_COMMIT" and r["same_rank_repeat"]),
        "same_id_new_neg": sum(
            1 for r in rows
            if r["error_type"] == "FALSE_COMMIT" and r["same_id_new_neg"]),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=-1)
    ap.add_argument("--train-csv", default="shadow_kplus1_train30.csv")
    ap.add_argument("--cal-csv", default="shadow_kplus1_cal10.csv")
    ap.add_argument("--train-npz", default="train30.npz")
    ap.add_argument("--cal-npz", default="cal10.npz")
    ap.add_argument("--model-prefix", default="tracklet_identity")
    ap.add_argument("--offline-epochs", type=int, default=60)
    ap.add_argument("--offline-lr", type=float, default=1e-3)
    ap.add_argument("--lora-rank", type=int, default=16)
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--margin", type=float, default=0.05)
    ap.add_argument("--lr-head", type=float, default=1e-4)
    ap.add_argument("--lr-lora", type=float, default=1e-4)
    ap.add_argument("--lr-ft", type=float, default=3e-5)
    ap.add_argument("--online-epochs", type=int, default=20)
    ap.add_argument("--replay", type=int, default=32)
    ap.add_argument("--kl-lambda", type=float, default=2.0)
    ap.add_argument("--epochs-ablation", default="5,20,40")
    args = ap.parse_args()
    global DEVICE
    DEVICE = torch.device(f"cuda:{args.gpu}") if args.gpu >= 0 \
        else torch.device("cpu")
    torch.manual_seed(0)
    np.random.seed(0)
    (N21 / "models").mkdir(parents=True, exist_ok=True)
    (N21 / "logs").mkdir(parents=True, exist_ok=True)
    for p in ["phase2_human_supervision_ledger_cal10.jsonl"]:
        (N21 / p).unlink(missing_ok=True)

    tr_groups = load_unified(N20 / f"features/{args.train_csv}",
                             DS / args.train_npz)
    cal_groups = load_unified(N20 / f"features/{args.cal_csv}",
                              DS / args.cal_npz)
    cal_full = full_top5(cal_groups)
    print(f"train30 attempts={len(tr_groups)} cal10 attempts={len(cal_groups)} "
          f"cal10 full_top5={len(cal_full)}", flush=True)

    # ---- offline base model (visual tracklet identity) ----
    base = TrackletIdentityModel()
    base = base.to(DEVICE)
    train_rows = offline_train(base, tr_groups, cal_full,
                               args.offline_epochs, lr=args.offline_lr)
    with (N21 / "offline_tracklet_training.csv").open(
            "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["epoch", "train_loss",
                                          "cal10_set_acc"])
        w.writeheader()
        w.writerows(train_rows)
    torch.save({"model": base.state_dict(),
                "args": vars(args)},
               N21 / "models" / f"{args.model_prefix}_base.pt")

    # ---- representation gate on cal10 ----
    base_metrics, base_rows = evaluate(base, cal_full, args.threshold,
                                       args.margin)
    print("BASE_VISUAL_GATE", json.dumps(base_metrics), flush=True)

    # ---- variants ----
    variants = {}
    variants["A0_visual_frozen"] = ("none", None)
    variants["C0_head_online"] = ("C0_head", args.lr_head)
    variants["C1_lora_online"] = ("C1_lora", args.lr_lora)
    variants["C2_partialft_online"] = ("C2_partial_ft", args.lr_ft)

    by_seq = defaultdict(list)
    for g in cal_full:
        by_seq[g["rows"][0]["seq"]].append(g)
    by_seq = {s: sorted(gs, key=lambda g: int(g["att"].split(":")[1]))
              for s, gs in by_seq.items()}

    summary = []
    all_rows = {}
    all_diag = {}
    for name, (mode, lr) in variants.items():
        rows_all, diag_all = [], []
        for seq, gs in by_seq.items():
            m = copy.deepcopy(base)
            if mode != "none":
                set_mode(m, mode)
            m.eval()
            m0 = copy.deepcopy(m)
            m0.eval()
            ledger = HumanSupervisionLedger(
                N21 / "phase2_human_supervision_ledger_cal10.jsonl")
            r, dg = run_stream(
                m, gs, args.threshold, args.margin,
                online=(mode != "none"), mode=mode, lr=lr,
                epochs=args.online_epochs, replay=args.replay,
                kl_lambda=args.kl_lambda,
                frozen_ref=m0 if mode != "none" else None,
                ledger=ledger)
            rows_all += r
            diag_all += dg
        all_rows[name] = rows_all
        all_diag[name] = diag_all
        with (N21 / f"phase2_rows_{name}.csv").open(
                "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows_all[0].keys()))
            w.writeheader()
            w.writerows(rows_all)
        s = summarize(rows_all)
        s["variant"] = name
        s["mode"] = mode
        s["trainable_params"] = (m.n_trainable() if mode != "none" else 0)
        s["total_params"] = m.n_params()
        summary.append(s)
        print(json.dumps(s, indent=2), flush=True)

    with (N21 / "capacity_ladder.csv").open(
            "w", newline="", encoding="utf-8") as f:
        cols = ["variant", "mode", "trainable_params", "total_params",
                "attempts", "corrections", "false_commits",
                "missed_commits", "error_rate", "same_rank_repeats",
                "same_id_new_neg"]
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(summary)

    # save lora/partial-ft checkpoints from a fresh cal10 stream (seq 0074)
    m_lora = copy.deepcopy(base)
    set_mode(m_lora, "C1_lora")
    torch.save({"model": m_lora.state_dict(),
                "trainable_params": m_lora.n_trainable(),
                "lora_rank": args.lora_rank,
                "mode": "C1_lora"},
               N21 / "models" / f"{args.model_prefix}_lora.pt")
    m_ft = copy.deepcopy(base)
    set_mode(m_ft, "C2_partial_ft")
    torch.save({"model": m_ft.state_dict(),
                "trainable_params": m_ft.n_trainable(),
                "mode": "C2_partial_ft"},
               N21 / "models" / f"{args.model_prefix}_partial_ft.pt")

    # ---- online epochs ablation on C1 (cal10 full stream, lr fixed) ----
    ep_rows = []
    for ep in [int(x) for x in args.epochs_ablation.split(",")]:
        rows_all = []
        for seq, gs in by_seq.items():
            m = copy.deepcopy(base)
            set_mode(m, "C1_lora")
            m.eval()
            m0 = copy.deepcopy(m)
            m0.eval()
            r, _ = run_stream(m, gs, args.threshold, args.margin,
                              online=True, mode="C1_lora",
                              lr=args.lr_lora, epochs=ep,
                              replay=args.replay, kl_lambda=args.kl_lambda,
                              frozen_ref=m0, ledger=None)
            rows_all += r
        s = summarize(rows_all)
        s["online_epochs"] = ep
        ep_rows.append(s)
        print("EPOCHS_ABLATION", json.dumps(s), flush=True)
    with (N21 / "online_epochs_ablation.csv").open(
            "w", newline="", encoding="utf-8") as f:
        cols = ["online_epochs", "attempts", "corrections",
                "false_commits", "missed_commits", "error_rate",
                "same_rank_repeats", "same_id_new_neg"]
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(ep_rows)

    # diagnostics
    with (N21 / "representation_shift.csv").open(
            "w", newline="", encoding="utf-8") as f:
        if all_diag["C1_lora_online"]:
            w = csv.DictWriter(
                f, fieldnames=list(all_diag["C1_lora_online"][0].keys()))
            w.writeheader()
            w.writerows(all_diag["C1_lora_online"])
    with (N21 / "correction_margin_shift.csv").open(
            "w", newline="", encoding="utf-8") as f:
        if all_diag["C1_lora_online"]:
            w = csv.DictWriter(
                f, fieldnames=list(all_diag["C1_lora_online"][0].keys()))
            w.writeheader()
            w.writerows(all_diag["C1_lora_online"])
    with (N21 / "lora_param_count.csv").open(
            "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["variant", "trainable_params", "total_params"])
        for s in summary:
            w.writerow([s["variant"], s["trainable_params"],
                        s["total_params"]])

    print("PHASE2_DONE", flush=True)


if __name__ == "__main__":
    main()
