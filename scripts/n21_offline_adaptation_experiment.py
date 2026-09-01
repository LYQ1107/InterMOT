#!/usr/bin/env python
"""N21 offline correction-driven online identity adaptation experiment.

Uses the real all-candidate K+1 shadow datasets built in N20 (real GFN
recovery distribution + real SAM3 shadow tracklets).

Variants on the cal10 attempt stream (sequence-disjoint from train30):
  A0              frozen N20 K+1 GRU, no update
  A1              offline on-policy retrain on train30 (+ model-induced
                  hard-negative sample weighting)
  A3_head         A1 + per-sequence online head update from simulated
                  human corrections (causal, episodic reset)
  A3_adapter      A1 + online adapter+head update
  A3_head_fromA0  frozen base + online head update (isolates the value of
                  correction-driven learning)

Human correction simulation (strict, causal):
  - false commit with a real target -> positive = true candidate,
    explicit negative = wrongly committed candidate;
  - false commit without target   -> NONE target, explicit negative =
    wrongly committed candidate;
  - missed commit                 -> positive = true candidate, no
    negatives (a miss correction certifies only the positive);
  - correct decisions             -> no correction.
Updates happen after the correction frame and only affect later attempts.
GT is used only for offline labels; live decisions never see it.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(".")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from train_n20_kplus1 import SharedGRUSet, load_groups, groups_to_tensors  # noqa: E402
from sam3_intermot.n21.human_supervision_ledger import (  # noqa: E402
    CorrectionRecord, HumanSupervisionLedger)

N20 = ROOT / "outputs/n20"
N21 = ROOT / "outputs/n21"


class OnlineSharedGRUSet(nn.Module):
    """SharedGRUSet with an optional residual identity adapter.

    mode:
      head_only    -> update base.head
      adapter_only -> update adapter (base.head frozen)
      adapter_head -> update adapter + base.head
    """

    def __init__(self, d, hidden=32, maxk=5, mode="head_only"):
        super().__init__()
        self.base = SharedGRUSet(d, hidden, maxk)
        self.mode = mode
        if mode != "head_only":
            self.adapter = nn.Sequential(
                nn.Linear(hidden, 16), nn.ReLU(), nn.Linear(16, hidden))
        else:
            self.adapter = None

    def forward(self, x, mask):
        B, K, H, D = x.shape
        xr = x.reshape(B * K, H, D)
        out, _ = self.base.gru(xr)
        z = out[:, -1].reshape(B, K, -1)
        if self.adapter is not None:
            z = z + self.adapter(z)
        z = z * mask.unsqueeze(-1)
        present = mask.sum(1).clamp(min=1)
        zmean = z.sum(1) / present.unsqueeze(-1)
        zmax = z.max(1).values
        zk = torch.cat([z, zmean.unsqueeze(1).expand(B, K, -1),
                        zmax.unsqueeze(1).expand(B, K, -1)], dim=-1)
        margin = zmax.unsqueeze(1).expand(B, K, -1) - z
        zk = torch.cat([zk, margin], dim=-1)
        logits = self.base.head(zk)
        logits = logits * mask.unsqueeze(-1) - 1e9 * (1 - mask).unsqueeze(-1)
        cand_logits = logits[:, :, 1:1 + K].max(1).values
        none_logit = logits[:, :, 0].max(1).values.unsqueeze(1)
        return torch.cat([none_logit, cand_logits], dim=1)

    def trainable(self):
        names = []
        for n, p in self.named_parameters():
            if p.requires_grad:
                names.append(n)
        return names

    def n_trainable(self):
        return sum(int(p.numel()) for p in self.parameters()
                   if p.requires_grad)


def set_mode(model, mode):
    for p in model.parameters():
        p.requires_grad_(False)
    if mode == "head_only":
        for p in model.base.head.parameters():
            p.requires_grad_(True)
    elif mode == "adapter_only":
        for p in model.adapter.parameters():
            p.requires_grad_(True)
    elif mode == "adapter_head":
        for p in model.adapter.parameters():
            p.requires_grad_(True)
        for p in model.base.head.parameters():
            p.requires_grad_(True)
    else:
        raise ValueError(mode)
    model.mode = mode
    return model


def decide(probs, threshold, margin):
    best = int(np.argmax(probs))
    if best >= 1 and probs[best] >= threshold:
        others = np.delete(probs, best)
        if probs[best] - others.max() >= margin:
            return best
    return 0


def ordered_groups(groups):
    """Sort groups chronologically (by frame inside attempt id)."""
    def frame_of(g):
        att = g[0][4]
        try:
            return int(att.split(":")[1])
        except (IndexError, ValueError):
            return 0
    return sorted(groups, key=frame_of)


def run_stream(model, groups_seq, threshold, margin,
               online=False, lr=1e-3, steps=1, replay=8, margin_rank=0.2,
               ledger=None, label_source="causal", mu=None, sd=None,
               frozen_ref=None, kl_lambda=0.0):
    """Run a chronological attempt stream with optional online updates.

    Returns per-attempt decision rows and per-correction diagnostic rows.
    """
    X_all, M_all, Y_all = groups_to_tensors(groups_seq, H, D, 5)
    if mu is not None:
        X_all = (X_all - mu) / sd
    ytrue = Y_all.argmax(1).numpy()
    with torch.inference_mode():
        P_all = torch.softmax(model(X_all, M_all), 1).numpy()

    rows = []
    diag = []
    replay_X = []
    replay_M = []
    replay_y = []
    replay_negs = []          # list of list of negative ranks
    replay_base_logits = []
    corrected_neg_ranks = set()
    corrected_pos_ranks = set()

    opt = None
    if online:
        opt = torch.optim.Adam(
            [p for p in model.parameters() if p.requires_grad], lr=lr)
        lossf = nn.CrossEntropyLoss()

    for i, g in enumerate(groups_seq):
        seq = g[0][4].split(":")[0]
        frame = int(g[0][4].split(":")[1])
        p = P_all[i]
        d = decide(p, threshold, margin)
        y = int(ytrue[i])
        err_type = None
        if d >= 1 and d != y:
            err_type = "FALSE_COMMIT"
        elif d == 0 and y >= 1:
            err_type = "MISSED_COMMIT"
        same_rank_repeat = False
        same_id_new_neg = False
        if err_type == "FALSE_COMMIT":
            same_rank_repeat = d in corrected_neg_ranks
            same_id_new_neg = y in corrected_pos_ranks and d != y
        rows.append({
            "attempt": g[0][4], "sequence": seq, "frame": frame,
            "ytrue": y, "decision": d, "error_type": err_type or "OK",
            "p_none": round(float(p[0]), 4),
            "p_best": round(float(p[1:].max()), 4),
            "same_rank_repeat": int(same_rank_repeat),
            "same_id_new_neg": int(same_id_new_neg),
        })

        if err_type is None:
            continue

        # ----- human correction event -----
        positive = None
        negs = []
        target = y
        if err_type == "FALSE_COMMIT":
            if y >= 1:
                positive = y
                negs = [d]
            else:
                target = 0
                negs = [d]
        else:  # MISSED_COMMIT
            positive = y

        rec = CorrectionRecord(
            timestamp=datetime.now(timezone.utc).isoformat(),
            sequence=seq, frame=frame,
            public_id=-1,  # filled by the loop when identity is known
            correction_type=("ID_WRONG" if err_type == "FALSE_COMMIT"
                             else "MISS"),
            positive={"candidate_rank": positive} if positive else None,
            explicit_negatives=[{"candidate_rank": r} for r in negs],
            source=label_source,
            provenance="causal",
            gt_used=(label_source == "offline_labeling"),
            extra={"attempt": g[0][4], "ytrue": y, "decision": d},
        )
        if ledger is not None:
            ledger.record(rec)

        if online:
            x = X_all[i:i + 1]
            m = M_all[i:i + 1]
            with torch.inference_mode():
                logits_pre = model(x, m)
            replay_X.append(x)
            replay_M.append(m)
            replay_y.append(torch.tensor([target]))
            replay_negs.append(negs)
            if frozen_ref is not None:
                with torch.inference_mode():
                    replay_base_logits.append(frozen_ref(x, m))
            if d in negs:
                corrected_neg_ranks.add(d)
            if positive is not None:
                corrected_pos_ranks.add(positive)
            # keep the last `replay` corrections
            if len(replay_X) > replay:
                replay_X = replay_X[-replay:]
                replay_M = replay_M[-replay:]
                replay_y = replay_y[-replay:]
                replay_negs = replay_negs[-replay:]
                replay_base_logits = replay_base_logits[-replay:]
            # online update (causal: only affects later attempts)
            t_upd = time.time()
            model.train()
            for _ in range(steps):
                opt.zero_grad()
                Xr = torch.cat(replay_X, 0)
                Mr = torch.cat(replay_M, 0)
                yr = torch.cat(replay_y, 0).flatten()
                logits = model(Xr, Mr)
                loss = lossf(logits, yr)
                if kl_lambda > 0 and replay_base_logits:
                    base_logits = torch.cat(replay_base_logits, 0)
                    kl = F.kl_div(
                        F.log_softmax(logits, 1),
                        F.softmax(base_logits.detach(), 1),
                        reduction="batchmean")
                    loss = loss + kl_lambda * kl
                if margin_rank > 0:
                    margins = []
                    for j, negs_j in enumerate(replay_negs):
                        pos = int(yr[j].item())
                        if pos <= 0:
                            continue
                        for r in negs_j:
                            if r < logits.shape[1]:
                                margins.append(
                                    torch.clamp(
                                        margin_rank -
                                        (logits[j, pos] - logits[j, r]),
                                        min=0.0))
                    if margins:
                        loss = loss + 0.5 * torch.stack(margins).mean()
                loss.backward()
                opt.step()
            model.eval()
            update_s = time.time() - t_upd
            with torch.inference_mode():
                logits_post = model(x, m)
            diag.append({
                "attempt": g[0][4], "sequence": seq, "frame": frame,
                "correction_type": rec.correction_type,
                "target": target, "negatives": negs,
                "pre_pos_logit": round(float(logits_pre[0, target]), 4),
                "post_pos_logit": round(float(logits_post[0, target]), 4),
                "pre_best_logit": round(float(logits_pre[0, 1:].max()), 4),
                "post_best_logit": round(float(logits_post[0, 1:].max()), 4),
                "loss": round(float(loss.item()), 4),
                "n_replay": len(replay_X),
                "update_s": round(update_s, 4),
            })
            # recompute probs for later attempts
            if i + 1 < len(X_all):
                with torch.inference_mode():
                    P_all[i + 1:] = torch.softmax(
                        model(X_all[i + 1:], M_all[i + 1:]), 1).numpy()
    return rows, diag


def summarize(rows, diag):
    n = len(rows)
    errs = [r for r in rows if r["error_type"] != "OK"]
    fc = [r for r in rows if r["error_type"] == "FALSE_COMMIT"]
    mc = [r for r in rows if r["error_type"] == "MISSED_COMMIT"]
    first_err = next((i for i, r in enumerate(rows) if r["error_type"] != "OK"),
                     None)
    after = rows[first_err + 1:] if first_err is not None else []
    err_after = [r for r in after if r["error_type"] != "OK"]
    repeat = [r for r in rows if r["same_rank_repeat"]]
    same_id = [r for r in rows if r["same_id_new_neg"]]
    return {
        "attempts": n,
        "corrections": len(errs),
        "false_commits": len(fc),
        "missed_commits": len(mc),
        "error_rate": round(len(errs) / n, 4) if n else None,
        "false_commit_rate": round(len(fc) / n, 4) if n else None,
        "same_rank_repeats": len(repeat),
        "same_id_new_neg": len(same_id),
        "errors_after_first_correction": len(err_after),
        "post_first_correction_rate": round(
            len(err_after) / max(1, len(after)), 4),
        "online_updates": len(diag),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--h", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--hard-weight", type=float, default=3.0)
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--margin", type=float, default=0.05)
    ap.add_argument("--lr-head", type=float, default=1e-3)
    ap.add_argument("--lr-adapter", type=float, default=1e-3)
    ap.add_argument("--steps", type=int, default=2)
    ap.add_argument("--replay", type=int, default=8)
    ap.add_argument("--margin-rank", type=float, default=0.2)
    ap.add_argument("--kl-lambda", type=float, default=0.0)
    args = ap.parse_args()
    global H, D
    torch.manual_seed(0)
    np.random.seed(0)

    N21.mkdir(parents=True, exist_ok=True)
    (N21 / "models").mkdir(parents=True, exist_ok=True)
    for p in ["human_supervision_ledger_cal10.jsonl",
              "correction_events_train30.jsonl"]:
        (N21 / p).unlink(missing_ok=True)
    tr_csv = N20 / "features/shadow_kplus1_train30.csv"
    cal_csv = N20 / "features/shadow_kplus1_cal10.csv"

    # ---- feature schema from the frozen bundle (authoritative) ----
    base_bundle = torch.load(N20 / "models/kplus1_gru.pt",
                             map_location="cpu")
    feats = base_bundle["feature_cols"]
    mu = np.asarray(base_bundle["mu"], dtype=np.float32)
    sd = np.asarray(base_bundle["sd"], dtype=np.float32)
    H = int(base_bundle["h"])
    D = len(feats)

    tr_groups = load_groups(tr_csv, H, feats)
    cal_groups = load_groups(cal_csv, H, feats)
    cal_full = [g for g in cal_groups
                if all(k in {k for k, *_ in g} for k in range(1, 6))]
    tr_full = [g for g in tr_groups
               if all(k in {k for k, *_ in g} for k in range(1, 6))]
    print(f"train30 groups={len(tr_groups)} full_top5={len(tr_full)} "
          f"cal10 groups={len(cal_groups)} full_top5={len(cal_full)}",
          flush=True)

    # ---- train30 on-policy dataset stats ----
    tr_seqs = Counter(g[0][4].split(":")[0] for g in tr_groups)
    tr_err = Counter(g[0][4].split(":")[0] for g in tr_full)
    (N21 / "train30_onpolicy_dataset_stats.json").write_text(json.dumps({
        "n_attempts_all": len(tr_groups),
        "n_attempts_full_top5": len(tr_full),
        "n_sequences": len(tr_seqs),
        "n_rows": sum(len(g) for g in tr_groups),
        "attempts_per_sequence_all": dict(tr_seqs),
        "attempts_per_sequence_full_top5": dict(tr_err),
        "source": "outputs/n20/features/shadow_kplus1_train30.csv "
                  "(real GFN recovery attempts + real SAM3 shadows)",
        "gt_usage": "offline labelling only",
    }, indent=2), encoding="utf-8")

    # ---- A1: offline on-policy retrain with hard-negative weighting ----
    Xtr, Mtr, Ytr = groups_to_tensors(tr_groups, H, D, 5)
    Xcal, Mcal, Ycal = groups_to_tensors(cal_full, H, D, 5)
    Xtr = (Xtr - mu) / sd
    Xcal = (Xcal - mu) / sd
    ytr = Ytr.argmax(1).numpy()
    ycal = Ycal.argmax(1).numpy()

    base_model = OnlineSharedGRUSet(D)
    base_model.load_state_dict(
        {"base." + k: v for k, v in base_bundle["model"].items()})
    base_model.eval()

    with torch.inference_mode():
        P0_tr = torch.softmax(base_model(Xtr, Mtr), 1).numpy()
    pred0 = np.argmax(P0_tr, axis=1)
    false_commit_mask = (pred0 != ytr) & (pred0 != 0)
    missed_mask = (pred0 == 0) & (ytr != 0)
    hard_mask = false_commit_mask | missed_mask
    sample_w = torch.ones(len(Xtr))
    sample_w[hard_mask] = args.hard_weight
    sample_w[false_commit_mask] = max(args.hard_weight, 3.0)

    a1_model = OnlineSharedGRUSet(D)
    a1_model.load_state_dict(
        {"base." + k: v for k, v in base_bundle["model"].items()})
    counts = Ytr.sum(0)
    cls_w = torch.ones(Ytr.shape[1])
    cls_w[0] = 0.5
    for j in range(1, Ytr.shape[1]):
        if counts[j] > 0:
            cls_w[j] = max(1.0, float(counts[0] / counts[j]))
    ce = nn.CrossEntropyLoss(weight=cls_w, reduction="none")
    a1_model.train()
    opt = torch.optim.Adam(a1_model.parameters(), lr=1e-3)
    train_rows = []
    for ep in range(args.epochs):
        opt.zero_grad()
        logits = a1_model(Xtr, Mtr)
        loss_v = ce(logits, Ytr.argmax(1))
        loss = (loss_v * sample_w).mean()
        loss.backward()
        opt.step()
        if (ep + 1) % 10 == 0:
            with torch.inference_mode():
                p_cal = torch.softmax(a1_model(Xcal, Mcal), 1).numpy()
            acc = float((np.argmax(p_cal, axis=1) == ycal).mean())
            train_rows.append({"epoch": ep + 1,
                               "train_loss": round(float(loss.item()), 4),
                               "cal_acc": round(acc, 4)})
            print(f"A1 epoch={ep + 1} loss={float(loss.item()):.4f} "
                  f"cal_acc={acc:.4f}", flush=True)
    a1_model.eval()
    with (N21 / "offline_onpolicy_training.csv").open(
            "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["epoch", "train_loss", "cal_acc"])
        w.writeheader()
        w.writerows(train_rows)

    # model-induced hard negatives (A0 on train30)
    with (N21 / "model_induced_hard_negatives.csv").open(
            "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["attempt", "sequence", "ytrue", "pred_frozen",
                    "hard_type", "p_none", "p_best"])
        for i, g in enumerate(tr_groups):
            if not hard_mask[i]:
                continue
            ht = ("FALSE_COMMIT" if false_commit_mask[i]
                  else "MISSED_COMMIT")
            w.writerow([g[0][4], g[0][4].split(":")[0], ytr[i], pred0[i],
                        ht, round(float(P0_tr[i, 0]), 4),
                        round(float(P0_tr[i, 1:].max()), 4)])

    torch.save({"model": {k: v
                          for k, v in a1_model.state_dict().items()
                          if k.startswith("base.")},
                "feature_cols": feats, "h": H, "mu": mu, "sd": sd,
                "hard_weight": args.hard_weight,
                "epochs": args.epochs},
               N21 / "models/kplus1_offline_onpolicy_gru.pt")

    # ---- calibration / ablation on cal10 full-top5 stream ----
    by_seq = defaultdict(list)
    for g in cal_full:
        by_seq[g[0][4].split(":")[0]].append(g)
    by_seq = {s: ordered_groups(gs) for s, gs in by_seq.items()}

    variants = {}
    variants["A0_frozen"] = (copy.deepcopy(base_model), "none")
    variants["A1_offline_retrain"] = (copy.deepcopy(a1_model), "none")

    def make_online(base, mode, lr):
        m = copy.deepcopy(base)
        if mode != "head_only" and m.adapter is None:
            hdim = m.base.gru.hidden_size
            m.adapter = nn.Sequential(
                nn.Linear(hdim, 16), nn.ReLU(), nn.Linear(16, hdim))
        set_mode(m, mode)
        return m

    variants["A3_head_fromA0"] = (
        make_online(base_model, "head_only", args.lr_head), "head_only")
    variants["A3_head"] = (
        make_online(a1_model, "head_only", args.lr_head), "head_only")
    variants["A3_adapter_head"] = (
        make_online(a1_model, "adapter_head", args.lr_adapter),
        "adapter_head")
    variants["A3_adapter_only"] = (
        make_online(a1_model, "adapter_only", args.lr_adapter),
        "adapter_only")

    summary_rows = []
    all_rows = {}
    all_diag = {}
    for name, (model, mode) in variants.items():
        rows_all, diag_all = [], []
        for seq, gs in by_seq.items():
            m = copy.deepcopy(model)
            if mode != "none":
                set_mode(m, mode)
            m.eval()
            m0 = copy.deepcopy(m)
            m0.eval()
            ledger = HumanSupervisionLedger(
                N21 / "human_supervision_ledger_cal10.jsonl")
            r, dg = run_stream(m, gs, args.threshold, args.margin,
                               online=(mode != "none"),
                               lr=(args.lr_head if "head" in mode
                                   else args.lr_adapter),
                               steps=args.steps, replay=args.replay,
                               margin_rank=args.margin_rank,
                               ledger=ledger, label_source="causal",
                               mu=mu, sd=sd,
                               frozen_ref=m0 if mode != "none" else None,
                               kl_lambda=args.kl_lambda)
            rows_all += r
            diag_all += dg
        all_rows[name] = rows_all
        all_diag[name] = diag_all
        with (N21 / f"rows_{name}.csv").open(
                "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows_all[0].keys()))
            w.writeheader()
            w.writerows(rows_all)
        s = summarize(rows_all, diag_all)
        s["variant"] = name
        s["mode"] = mode
        s["trainable_params"] = (model.n_trainable() if mode != "none"
                                 else 0)
        s["total_params"] = sum(p.numel() for p in model.parameters())
        summary_rows.append(s)
        print(json.dumps(s, indent=2), flush=True)

    with (N21 / "offline_onpolicy_calibration.csv").open(
            "w", newline="", encoding="utf-8") as f:
        cols = ["variant", "mode", "trainable_params", "total_params",
                "attempts", "corrections", "false_commits", "missed_commits",
                "error_rate", "false_commit_rate", "same_rank_repeats",
                "same_id_new_neg", "errors_after_first_correction",
                "post_first_correction_rate", "online_updates"]
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(summary_rows)

    # ---- per-sequence rows ----
    with (N21 / "online_head_baseline.csv").open(
            "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(all_rows["A3_head"][0].keys()))
        w.writeheader()
        w.writerows(all_rows["A3_head"])
    with (N21 / "online_adapter_baseline.csv").open(
            "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f, fieldnames=list(all_rows["A3_adapter_head"][0].keys()))
        w.writeheader()
        w.writerows(all_rows["A3_adapter_head"])
    with (N21 / "online_update_diagnostic.csv").open(
            "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f, fieldnames=list(all_diag["A3_head"][0].keys())
            if all_diag["A3_head"] else [])
        w.writeheader()
        w.writerows(all_diag["A3_head"])
    with (N21 / "repeated_confusion.csv").open(
            "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=[
            "variant", "same_rank_repeats", "same_id_new_neg",
            "false_commits", "missed_commits"])
        w.writeheader()
        for name, rows in all_rows.items():
            s = summarize(rows, all_diag[name])
            w.writerow({"variant": name,
                        "same_rank_repeats": s["same_rank_repeats"],
                        "same_id_new_neg": s["same_id_new_neg"],
                        "false_commits": s["false_commits"],
                        "missed_commits": s["missed_commits"]})

    # ---- pre/post identity margin diagnostics ----
    with (N21 / "pre_post_identity_margin.csv").open(
            "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(all_diag["A3_head"][0].keys()))
        w.writeheader()
        w.writerows(all_diag["A3_head"])

    # ---- adaptation latency / memory ----
    with (N21 / "adaptation_latency.csv").open(
            "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["variant", "n_updates", "mean_update_s",
                    "p95_update_s", "total_update_s"])
        for name, dg in all_diag.items():
            if not dg:
                continue
            us = np.asarray([d["update_s"] for d in dg])
            w.writerow([name, len(us), round(float(us.mean()), 4),
                        round(float(np.percentile(us, 95)), 4),
                        round(float(us.sum()), 4)])
    with (N21 / "adaptation_memory.csv").open(
            "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=[
            "variant", "mode", "trainable_params", "total_params",
            "trainable_ratio"])
        w.writeheader()
        for s in summary_rows:
            w.writerow({
                "variant": s["variant"], "mode": s["mode"],
                "trainable_params": s["trainable_params"],
                "total_params": s["total_params"],
                "trainable_ratio": round(
                    s["trainable_params"] / max(1, s["total_params"]), 6),
            })

    # ---- correction signal audit (what each correction legally provides) ----
    audit = Counter()
    for dg in all_diag.values():
        for d in dg:
            audit[(d["correction_type"],
                   "positive" if d["target"] >= 1 else "none_target",
                   "neg" if d["negatives"] else "no_neg")] += 1
    with (N21 / "correction_signal_audit.csv").open(
            "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["correction_type", "target", "explicit_negatives",
                    "count"])
        for (ct, tg, ng), c in sorted(audit.items()):
            w.writerow([ct, tg, ng, c])

    # ---- train30 offline correction events (labelled with train30 GT) ----
    ledger30 = HumanSupervisionLedger(
        N21 / "correction_events_train30.jsonl")
    for g in ordered_groups(tr_groups):
        seq = g[0][4].split(":")[0]
        frame = int(g[0][4].split(":")[1])
        Xg, Mg, Yg = groups_to_tensors([g], H, D, 5)
        Xg = (Xg - mu) / sd
        with torch.inference_mode():
            pg = torch.softmax(a1_model(Xg, Mg), 1).numpy()[0]
        d = decide(pg, args.threshold, args.margin)
        y = int(Yg.argmax(1).numpy()[0])
        if d >= 1 and d != y:
            ct = "ID_WRONG"
            pos = y if y >= 1 else None
            negs = [d]
        elif d == 0 and y >= 1:
            ct = "MISS"
            pos = y
            negs = []
        else:
            continue
        ledger30.record(CorrectionRecord(
            timestamp=datetime.now(timezone.utc).isoformat(),
            sequence=seq, frame=frame, public_id=-1,
            correction_type=ct,
            positive={"candidate_rank": pos} if pos else None,
            explicit_negatives=[{"candidate_rank": r} for r in negs],
            source="offline_labeling", provenance="causal",
            gt_used=True, extra={"attempt": g[0][4]}))

    print("N21_OFFLINE_DONE", flush=True)


if __name__ == "__main__":
    main()
