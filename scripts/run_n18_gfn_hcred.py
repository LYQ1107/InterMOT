#!/usr/bin/env python
"""N18.3 GFN (official CUHK-SYSU ConvNeXt-B) on the HCRED calibration set.

Same episodes as N17 (outputs/n17/cal_episodes.csv): query crop at frame t,
gallery at frame f=t+delta. Reports query-person recall@0.3/0.5/0.7, top1/top3
localization, generic-detector-rescue ceiling, and absent false-recovery.
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
sys.path.insert(0, str(ROOT / "scripts"))
from gfn_recovery_model import load_model  # noqa: E402

DT = Path("/path/to/dancetrack")
OUT = ROOT / "outputs/n18"
EPS = OUT.parent / "n17/cal_episodes.csv"


def iou(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    ua = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    ub = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    return inter / (ua + ub - inter + 1e-9)


def load_frame(seq, fidx):
    p = DT / "train" / seq / "img1" / f"{int(fidx) + 1:08d}.jpg"
    if not p.exists():
        return None
    return Image.open(p).convert("RGB")


def crop_query(img, box, margin=0.2):
    W, H = img.size
    x1, y1, x2, y2 = box
    w, h = max(1.0, x2 - x1), max(1.0, y2 - y1)
    x1 = max(0.0, x1 - margin * w)
    y1 = max(0.0, y1 - margin * h)
    x2 = min(float(W), x2 + margin * w)
    y2 = min(float(H), y2 + margin * h)
    if x2 - x1 < 8 or y2 - y1 < 8:
        return img
    return img.crop((int(x1), int(y1), int(x2), int(y2)))


def norm_emb(x):
    return x / (x.norm(dim=-1, keepdim=True) + 1e-8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--max-episodes", type=int, default=0)
    args = ap.parse_args()
    torch.cuda.set_device(args.gpu)

    with EPS.open(newline="", encoding="utf-8") as f:
        eps = list(csv.DictReader(f))
    eps = eps[args.shard:: args.nshards]
    if args.max_episodes:
        eps = eps[: args.max_episodes]

    model, cfg, epoch, _, _ = load_model(f"cuda:{args.gpu}")
    rows = []
    for i, r in enumerate(eps):
        seq, t, f_ = r["sequence"], int(r["t"]), int(r["f"])
        qbox = json.loads(r["human_box"])
        present = int(r["target_present"]) == 1
        target = json.loads(r["target_box"]) if present else None
        gallery_img = load_frame(seq, f_)
        if gallery_img is None:
            continue
        with torch.inference_mode():
            out = model([F.to_tensor(gallery_img).cuda()], None,
                        inference_mode="det")[0]
        boxes = out["det_boxes"].cpu().numpy()
        scores = out["det_scores"].cpu().numpy()
        embs = out["det_emb"]
        row = {
            "sequence": seq, "t": t, "gid": r["gid"], "f": f_,
            "delta": r["delta"], "present": int(present),
        }
        if len(boxes) == 0:
            row.update(top1=0, top3=0, best_iou=0.0, top1_sim=float("nan"),
                       generic_rescue=0, n_dets=0)
        else:
            ge = norm_emb(embs)
            # query embedding: top detection on the cropped query image
            qimg = load_frame(seq, t)
            if qimg is None:
                qimg = gallery_img
            qcrop = crop_query(qimg, qbox)
            with torch.inference_mode():
                qout = model([F.to_tensor(qcrop).cuda()], None,
                             inference_mode="det")[0]
            if qout["det_emb"].shape[0] == 0:
                qe = torch.zeros(ge.shape[1], device=ge.device)
            else:
                qi = int(torch.argmax(qout["det_scores"]).item())
                qe = norm_emb(qout["det_emb"][qi])
            qe = qe.reshape(-1)
            sims = (ge @ qe).cpu().numpy()
            order = np.argsort(-sims)
            top1 = int(iou(boxes[order[0]], target) >= 0.5) if present else 0
            top3 = 0
            best = 0.0
            for k in order[:3]:
                if present:
                    best = max(best, iou(boxes[k], target))
                    top3 |= int(iou(boxes[k], target) >= 0.5)
            generic = 0
            if present:
                generic = int(any(iou(b, target) >= 0.5 for b in boxes))
            row.update(top1=top1, top3=top3, best_iou=round(float(best), 3),
                       top1_sim=round(float(sims[order[0]]), 4),
                       generic_rescue=generic, n_dets=int(len(boxes)))
        rows.append(row)
        if (i + 1) % 25 == 0:
            print(f"shard{args.shard} {i+1}/{len(eps)}", flush=True)

    tag = f"_s{args.shard}" if args.nshards > 1 else ""
    with (OUT / f"gfn_hcred{tag}.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"shard{args.shard} done n={len(rows)}", flush=True)


if __name__ == "__main__":
    main()
