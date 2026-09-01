#!/usr/bin/env python
"""Offline query-anchor ablation for FULL_LOOP_V0 recovery ranking.

Compares the deployed first-appearance human anchor (H_i) with a fresher
"last GT-visible box before t" diagnostic anchor.  The second anchor is an
offline upper-bound diagnosis only: it uses GT to find the most recent
visible frame, and would need a causal verified-memory equivalent for
deployment.  Nothing is fed back into the online loop.
"""

import argparse
import csv
import json
import sys
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


def frame_img(seq, f):
    p = DT / "train" / seq / "img1" / f"{f + 1:08d}.jpg"
    return Image.open(p).convert("RGB") if p.exists() else None


def query_emb(model, device, seq, f, box, cache):
    key = (seq, f, tuple(np.round(box, 1)))
    if key not in cache:
        img = frame_img(seq, f)
        if img is None:
            cache[key] = None
        else:
            qcrop = crop_query(img, box)
            with torch.inference_mode():
                qout = model([F.to_tensor(qcrop).to(device)], None,
                             inference_mode="det")[0]
            if qout["det_emb"].shape[0] == 0:
                cache[key] = None
            else:
                qi = int(torch.argmax(qout["det_scores"]).item())
                qe = qout["det_emb"].float()[qi].reshape(-1).cpu()
                cache[key] = qe / (qe.norm() + 1e-8)
    return cache[key]


def run_sequence(seq, tx_seq, model, device):
    gt = load_gt(seq)
    first = {}
    for f in sorted(gt):
        gf = gt[f]
        for gid, box in zip(gf.gt_ids, gf.boxes):
            if gid not in first:
                first[gid] = (f, np.asarray(box, dtype=float))
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
        embs = out["det_emb"].float().cpu()
        if boxes.ndim == 1:
            boxes = boxes.reshape(1, -1)
            embs = embs.reshape(1, -1)
        if len(boxes) == 0:
            gallery[f] = (boxes, None)
            continue
        ge = embs / (embs.norm(dim=1, keepdim=True) + 1e-8)
        gallery[f] = (boxes, ge)
    qcache = {}
    rows = []
    for t in tx_seq:
        f = t["frame"]
        gid = t["gid"]
        if f not in gallery:
            continue
        boxes, ge = gallery[f]
        gf = gt.get(f)
        gbox = None
        if gf is not None and gid in gf.gt_ids:
            gbox = np.asarray(gf.boxes[gf.gt_ids.index(gid)], dtype=float)
        gt_present = gbox is not None
        if not gt_present or len(boxes) == 0:
            rows.append({
                "sequence": seq, "frame": f, "gid": gid, "gt_present": 0,
                "first_top3": 0, "last_top3": 0, "first_top1": 0,
                "last_top1": 0, "anchor_gap_first": None,
                "anchor_gap_last": None,
            })
            continue
        # last GT-visible frame strictly before f
        last_f = None
        last_box = None
        for f2 in range(f - 1, -1, -1):
            gf2 = gt.get(f2)
            if gf2 is not None and gid in gf2.gt_ids:
                last_f = f2
                last_box = np.asarray(
                    gf2.boxes[gf2.gt_ids.index(gid)], dtype=float)
                break
        af, abox = first[gid]
        q_first = query_emb(model, device, seq, af, abox, qcache)
        q_last = query_emb(model, device, seq, last_f, last_box, qcache) \
            if last_f is not None else None
        top = {}
        for name, qe in (("first", q_first), ("last", q_last)):
            if qe is None:
                top[f"{name}_top1"] = 0
                top[f"{name}_top3"] = 0
                continue
            sims = (ge @ qe).numpy()
            order = np.argsort(-sims)
            top[f"{name}_top1"] = int(iou(boxes[order[0]], gbox) >= 0.5)
            top[f"{name}_top3"] = int(any(
                iou(boxes[order[i]], gbox) >= 0.5
                for i in range(min(3, len(order)))))
        rows.append({
            "sequence": seq, "frame": f, "gid": gid, "gt_present": 1,
            "first_top3": top["first_top3"], "last_top3": top["last_top3"],
            "first_top1": top["first_top1"], "last_top1": top["last_top1"],
            "anchor_gap_first": f - af,
            "anchor_gap_last": None if last_f is None else f - last_f,
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
    p = out / f"gfn_full_loop_query_anchor_ablation{tag}.csv"
    with p.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=[
            "sequence", "frame", "gid", "gt_present", "first_top3",
            "last_top3", "first_top1", "last_top1", "anchor_gap_first",
            "anchor_gap_last"])
        w.writeheader()
        for seq in seqs:
            seq_tx = [t for t in tx if t["sequence"] == seq]
            rows = run_sequence(seq, seq_tx, model, device)
            w.writerows(rows)
            print(f"{seq}: {len(rows)} rows", flush=True)
    print(f"QUERY_ANCHOR_DONE shard={args.shard} file={p}", flush=True)


if __name__ == "__main__":
    main()
