#!/usr/bin/env python
"""Re-run GFN gallery detection for generic-rescue episodes and store the
oracle-selected (best-IoU-to-GT) recovery box per episode.
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--max-episodes", type=int, default=0)
    args = ap.parse_args()
    torch.cuda.set_device(args.gpu)

    with (OUT / "hcred_recovery.csv").open(newline="", encoding="utf-8") as f:
        all_rows = list(csv.DictReader(f))
    src = {}
    with (OUT.parent / "n17/cal_episodes.csv").open(newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            src[(r["sequence"], r["t"], r["gid"], r["f"])] = r
    target_rows = [r for r in all_rows
                   if r["present"] == "1" and int(r["generic_rescue"]) == 1]
    target_rows = target_rows[args.shard:: args.nshards]
    if args.max_episodes:
        target_rows = target_rows[: args.max_episodes]

    model, cfg, epoch, _, _ = load_model(f"cuda:{args.gpu}")
    out_rows = []
    for i, r in enumerate(target_rows):
        seq, f_ = r["sequence"], int(r["f"])
        key = (r["sequence"], r["t"], r["gid"], r["f"])
        target = json.loads(src[key]["target_box"])
        p = DT / "train" / seq / "img1" / f"{f_ + 1:08d}.jpg"
        if not p.exists():
            continue
        img = Image.open(p).convert("RGB")
        with torch.inference_mode():
            out = model([F.to_tensor(img).cuda()], None,
                        inference_mode="det")[0]
        boxes = out["det_boxes"].cpu().numpy()
        if len(boxes) == 0:
            print(f"skip empty dets {seq} f={f_}", flush=True)
            continue
        ious = [iou(b, target) for b in boxes]
        best = int(np.argmax(ious))
        out_rows.append({
            "sequence": seq, "t": r["t"], "gid": r["gid"], "f": f_,
            "recovered_box": json.dumps(boxes[best].tolist()),
            "recovered_iou": round(float(ious[best]), 4),
            "n_dets": int(len(boxes)),
        })
        if (i + 1) % 25 == 0:
            print(f"shard{args.shard} {i+1}/{len(target_rows)}", flush=True)
    tag = f"_s{args.shard}" if args.nshards > 1 else ""
    with (OUT / f"gfn_recovery_boxes{tag}.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
        w.writeheader()
        w.writerows(out_rows)
    print(f"shard{args.shard} done n={len(out_rows)}", flush=True)


if __name__ == "__main__":
    main()
