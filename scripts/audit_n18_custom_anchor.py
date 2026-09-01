#!/usr/bin/env python
"""Offline GFN top-K replay for a set of precomputed recovery anchors."""

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


def frame_img(seq, f):
    p = DT / "train" / seq / "img1" / f"{f + 1:08d}.jpg"
    return Image.open(p).convert("RGB") if p.exists() else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--anchors", required=True)
    ap.add_argument("--out-name", required=True)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    args = ap.parse_args()
    torch.cuda.set_device(args.gpu)
    device = f"cuda:{args.gpu}"
    model, _, _, _, _ = load_model(device)
    anchors = []
    for line in open(args.anchors, encoding="utf-8"):
        line = line.strip()
        if line:
            anchors.append(json.loads(line))
    seqs = sorted({a["sequence"] for a in anchors})
    seqs = seqs[args.shard:: args.nshards]
    out = OUT / "tables"
    out.mkdir(parents=True, exist_ok=True)
    tag = f"_s{args.shard}" if args.nshards > 1 else ""
    p = out / f"{args.out_name}{tag}.csv"
    with p.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=[
            "sequence", "frame", "gid", "gt_present", "top1_correct",
            "top3_correct", "top5_correct", "top10_correct",
            "anchor_gap"])
        w.writeheader()
        for seq in seqs:
            gt = load_gt(seq)
            seq_a = [a for a in anchors if a["sequence"] == seq]
            frames = sorted({a["frame"] for a in seq_a})
            gallery = {}
            for f0 in frames:
                img = frame_img(seq, f0)
                if img is None:
                    continue
                with torch.inference_mode():
                    g = model([F.to_tensor(img).to(device)], None,
                              inference_mode="det")[0]
                boxes = g["det_boxes"].float().cpu().numpy()
                embs = g["det_emb"].float().cpu()
                if boxes.ndim == 1:
                    boxes = boxes.reshape(1, -1)
                    embs = embs.reshape(1, -1)
                if len(boxes):
                    ge = embs / (embs.norm(dim=1, keepdim=True) + 1e-8)
                else:
                    ge = None
                gallery[f0] = (boxes, ge)
            qcache = {}
            for a in seq_a:
                f0 = a["frame"]
                gid = a["gid"]
                af = a["anchor_frame"]
                abox = np.asarray(a["anchor_box"], dtype=float)
                qkey = (af, tuple(np.round(abox, 1)))
                if qkey not in qcache:
                    img = frame_img(seq, af)
                    qe = None
                    if img is not None:
                        with torch.inference_mode():
                            q = model(
                                [F.to_tensor(crop_query(img, abox)).to(device)],
                                None, inference_mode="det")[0]
                        if q["det_emb"].shape[0]:
                            qi = int(torch.argmax(q["det_scores"]).item())
                            qe = q["det_emb"].float()[qi].reshape(-1).cpu()
                            qe = qe / (qe.norm() + 1e-8)
                    qcache[qkey] = qe
                qe = qcache[qkey]
                gf = gt.get(f0)
                gbox = None
                if gf is not None and gid in gf.gt_ids:
                    gbox = np.asarray(
                        gf.boxes[gf.gt_ids.index(gid)], dtype=float)
                gt_present = gbox is not None
                row = {
                    "sequence": seq, "frame": f0, "gid": gid,
                    "gt_present": int(gt_present),
                    "top1_correct": 0, "top3_correct": 0,
                    "top5_correct": 0, "top10_correct": 0,
                    "anchor_gap": f0 - af,
                }
                if gt_present and f0 in gallery and qe is not None:
                    boxes, ge = gallery[f0]
                    if ge is not None and len(boxes):
                        sims = (ge @ qe).numpy()
                        order = np.argsort(-sims)
                        for k in (1, 3, 5, 10):
                            row[f"top{k}_correct"] = int(any(
                                iou(boxes[order[i]], gbox) >= 0.5
                                for i in range(min(k, len(order)))))
                w.writerow(row)
            print(seq, len(seq_a), flush=True)
    print(f"CUSTOM_ANCHOR_DONE shard={args.shard} file={p}", flush=True)


if __name__ == "__main__":
    main()
