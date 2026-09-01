#!/usr/bin/env python
"""Offline GFN top-K audit of the causal FULL_LOOP_V0 recovery attempts.

The online loop recorded only the GFN top-1 box.  To split failure F3
(no correct detection in the gallery) from F4 (correct detection exists but
top-1 ranking is wrong), we replay the exact recorded query anchor (the GT
human box at the identity's first appearance) against the same frame and
measure top-1/3/5/10 and best-detection IoU with GT.

This is an offline evaluation of already-recorded causal decisions; no result
is fed back into the loop.
"""

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision.transforms import functional as F

ROOT = Path(".")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from gfn_recovery_model import load_model  # noqa: E402
from run_n18_full_loop_v0 import crop_query, load_gt  # noqa: E402

DT = Path("/path/to/dancetrack")
OUT = ROOT / "outputs/n18"


def iou(a, b):
    a = np.asarray(a, dtype=float).reshape(-1)
    b = np.asarray(b, dtype=float).reshape(-1)
    if a.size != 4 or b.size != 4:
        return 0.0
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    ua = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    ub = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    return inter / (ua + ub - inter + 1e-9)


def load_transactions():
    rows = []
    for p in sorted(OUT.glob("reactivation_transactions_full_s*.jsonl")):
        for line in p.open(encoding="utf-8"):
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def first_appearance(gt):
    first = {}
    for f in sorted(gt):
        gf = gt[f]
        for gid, box in zip(gf.gt_ids, gf.boxes):
            if gid not in first:
                first[gid] = (f, np.asarray(box, dtype=float))
    return first


def frame_img(seq, f):
    p = DT / "train" / seq / "img1" / f"{f + 1:08d}.jpg"
    return Image.open(p).convert("RGB") if p.exists() else None


def run_sequence(seq, tx_seq, model, device):
    gt = load_gt(seq)
    first = first_appearance(gt)
    gids = sorted({t["gid"] for t in tx_seq})
    query_emb = {}
    for gid in gids:
        af, abox = first[gid]
        aimg = frame_img(seq, af)
        if aimg is None:
            continue
        qcrop = crop_query(aimg, abox)
        with torch.inference_mode():
            qout = model([F.to_tensor(qcrop).to(device)], None,
                         inference_mode="det")[0]
        if qout["det_emb"].shape[0] == 0:
            continue
        qi = int(torch.argmax(qout["det_scores"]).item())
        qe = qout["det_emb"].float()[qi].reshape(-1).cpu()
        query_emb[gid] = qe / (qe.norm() + 1e-8)

    frames = sorted({t["frame"] for t in tx_seq})
    gallery = {}
    for f in frames:
        img = frame_img(seq, f)
        if img is None:
            continue
        with torch.inference_mode():
            out = model([F.to_tensor(img).to(device)], None,
                        inference_mode="det")[0]
        boxes = out["det_boxes"].float().cpu().numpy()
        scores = out["det_scores"].float().cpu().numpy()
        embs = out["det_emb"].float().cpu()
        if boxes.ndim == 1:
            boxes = boxes.reshape(1, -1)
            scores = np.atleast_1d(scores)
            embs = embs.reshape(1, -1)
        if len(boxes) == 0:
            gallery[f] = (boxes, scores, embs)
            continue
        ge = embs / (embs.norm(dim=1, keepdim=True) + 1e-8)
        gallery[f] = (boxes, scores, ge)

    rows = []
    for f in frames:
        if f not in gallery:
            continue
        boxes, scores, ge = gallery[f]
        target = gt.get(f)
        gid_to_tx_f = [t for t in tx_seq if t["frame"] == f]
        for t in gid_to_tx_f:
            gid = t["gid"]
            gbox = None
            if target is not None and gid in target.gt_ids:
                gbox = np.asarray(target.boxes[target.gt_ids.index(gid)],
                                  dtype=float)
            gt_present = gbox is not None
            best = 0.0
            for b in boxes:
                if gt_present:
                    best = max(best, iou(b, gbox))
            any_correct = int(best >= 0.5) if gt_present else 0
            top_correct = {}
            sims = None
            margin = None
            s1 = None
            if len(boxes) and gid in query_emb:
                sims = (ge @ query_emb[gid]).numpy()
                order = np.argsort(-sims)
                s1 = float(sims[order[0]])
                s2 = float(sims[order[1]]) if len(order) > 1 else 0.0
                margin = s1 - s2
                for k in (1, 3, 5, 10):
                    ok = 0
                    if gt_present:
                        ok = int(any(iou(boxes[order[i]], gbox) >= 0.5
                                     for i in range(min(k, len(order)))))
                    top_correct[f"top{k}_correct"] = ok
            else:
                for k in (1, 3, 5, 10):
                    top_correct[f"top{k}_correct"] = 0
            rows.append({
                "sequence": seq, "frame": f, "gid": gid,
                "gt_present": int(gt_present), "n_dets": len(boxes),
                "best_det_iou": round(float(best), 4) if gt_present else None,
                "any_det_correct": any_correct,
                **top_correct,
                "top1_sim": None if s1 is None else round(s1, 4),
                "top1_margin": None if margin is None else round(margin, 4),
                "top1_score": None if s1 is None else round(
                    float(scores[order[0]]), 4),
                "top1_box": None if s1 is None else
                [round(float(x), 2) for x in boxes[order[0]]],
            })
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--seqs", default="")
    args = ap.parse_args()
    torch.cuda.set_device(args.gpu)
    device = f"cuda:{args.gpu}"
    model, _, _, _, _ = load_model(device)

    tx = load_transactions()
    seqs = args.seqs.split(",") if args.seqs else sorted(
        {t["sequence"] for t in tx})
    seqs = sorted(seqs)[args.shard:: args.nshards]
    out = OUT / "tables"
    out.mkdir(parents=True, exist_ok=True)
    tag = f"_s{args.shard}" if args.nshards > 1 else ""
    out_csv = out / f"gfn_full_loop_topk_audit{tag}.csv"
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=[
            "sequence", "frame", "gid", "gt_present", "n_dets",
            "best_det_iou", "any_det_correct", "top1_correct", "top3_correct",
            "top5_correct", "top10_correct", "top1_sim", "top1_margin",
            "top1_score", "top1_box"])
        w.writeheader()
        for seq in seqs:
            seq_tx = [t for t in tx if t["sequence"] == seq]
            rows = run_sequence(seq, seq_tx, model, device)
            w.writerows(rows)
            print(f"{seq}: {len(rows)} attempts", flush=True)
    print(f"AUDIT_DONE shard={args.shard} file={out_csv}", flush=True)


if __name__ == "__main__":
    main()
