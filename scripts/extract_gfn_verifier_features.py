#!/usr/bin/env python
"""Export GFN recovery-candidate features for the N18.7 verifier.

Per episode: top1 sim, top1-top2 sim margin, top1 detector score, n_dets.
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


def iou(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    ua = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    ub = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    return inter / (ua + ub - inter + 1e-9)


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    args = ap.parse_args()
    torch.cuda.set_device(args.gpu)

    with (OUT.parent / "n17/cal_episodes.csv").open(newline="", encoding="utf-8") as f:
        eps = list(csv.DictReader(f))
    eps = eps[args.shard:: args.nshards]
    model, cfg, epoch, _, _ = load_model(f"cuda:{args.gpu}")
    rows = []
    for i, r in enumerate(eps):
        seq, t, f_ = r["sequence"], int(r["t"]), int(r["f"])
        qbox = json.loads(r["human_box"])
        present = int(r["target_present"]) == 1
        target = json.loads(r["target_box"]) if present else None
        p = DT / "train" / seq / "img1" / f"{f_ + 1:08d}.jpg"
        if not p.exists():
            continue
        img = Image.open(p).convert("RGB")
        with torch.inference_mode():
            out = model([F.to_tensor(img).cuda()], None,
                        inference_mode="det")[0]
        boxes = out["det_boxes"].cpu().numpy()
        scores = out["det_scores"].cpu().numpy()
        embs = out["det_emb"]
        row = {"sequence": seq, "t": t, "gid": r["gid"], "f": f_,
               "present": int(present), "n_dets": int(len(boxes))}
        if len(boxes) == 0:
            row.update(gfn_top1_sim="", gfn_margin="", gfn_top1_score="",
                       top1_correct=0, best_iou="")
        else:
            ge = embs / (embs.norm(dim=1, keepdim=True) + 1e-8)
            qimg = Image.open(DT / "train" / seq / "img1" / f"{t + 1:08d}.jpg").convert("RGB")
            qcrop = crop_query(qimg, qbox)
            with torch.inference_mode():
                qout = model([F.to_tensor(qcrop).cuda()], None,
                             inference_mode="det")[0]
            if qout["det_emb"].shape[0] == 0:
                qe = torch.zeros(ge.shape[1], device=ge.device)
            else:
                qi = int(torch.argmax(qout["det_scores"]).item())
                qe = qout["det_emb"][qi].reshape(-1)
            qe = qe / (qe.norm() + 1e-8)
            sims = (ge @ qe).cpu().numpy()
            order = np.argsort(-sims)
            s1, s2 = float(sims[order[0]]), float(sims[order[1]]) \
                if len(order) > 1 else 0.0
            correct = int(iou(boxes[order[0]], target) >= 0.5) if present else 0
            best = max(iou(b, target) for b in boxes) if present else 0.0
            row.update(gfn_top1_sim=round(s1, 4), gfn_margin=round(s1 - s2, 4),
                       gfn_top1_score=round(float(scores[order[0]]), 4),
                       top1_correct=correct, best_iou=round(float(best), 4))
        rows.append(row)
        if (i + 1) % 50 == 0:
            print(f"shard{args.shard} {i+1}/{len(eps)}", flush=True)
    tag = f"_s{args.shard}" if args.nshards > 1 else ""
    with (OUT / f"verifier_features{tag}.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"shard{args.shard} done n={len(rows)}", flush=True)


if __name__ == "__main__":
    main()
