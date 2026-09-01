#!/usr/bin/env python
"""N18.3 PSTR (official CUHK-SYSU ResNet50) on the HCRED calibration set.

Run with the pstr_env interpreter. Query crop at frame t, gallery at f=t+delta.
Same metric definitions as the GFN benchmark.
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(".")
sys.path.insert(0, str(ROOT / "third_party/PSTR"))
sys.path.insert(0, str(ROOT / "scripts"))

import pstr_part_attention  # noqa: E402  (registers PartAttention)
import mmcv  # noqa: E402
from mmcv import Config  # noqa: E402
from mmdet.datasets.pipelines import Compose, LoadImageFromFile  # noqa: E402
from mmdet.models import build_detector  # noqa: E402
from mmcv.runner import load_checkpoint  # noqa: E402

DT = Path("/path/to/dancetrack")
OUT = ROOT / "outputs/n18"
EPS = OUT.parent / "n17/cal_episodes.csv"
CKPT = OUT / "checkpoints/pstr_cuhk_resnet50.pth"
CFG = ROOT / "third_party/PSTR/configs/pstr/pstr_r50_24e_cuhk.py"


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
    H, W = img.shape[:2]
    x1, y1, x2, y2 = box
    w, h = max(1.0, x2 - x1), max(1.0, y2 - y1)
    x1 = int(max(0, x1 - margin * w))
    y1 = int(max(0, y1 - margin * h))
    x2 = int(min(W, x2 + margin * w))
    y2 = int(min(H, y2 + margin * h))
    if x2 - x1 < 8 or y2 - y1 < 8:
        return img
    return img[y1:y2, x1:x2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--max-episodes", type=int, default=0)
    args = ap.parse_args()
    torch.cuda.set_device(args.gpu)

    cfg = Config.fromfile(str(CFG))
    cfg.model.pretrained = None
    model = build_detector(cfg.model, test_cfg=cfg.get("test_cfg"))
    model.cfg = cfg
    load_checkpoint(model, str(CKPT), map_location="cpu")
    model.cuda().eval()
    pipeline = Compose(cfg.data.test.pipeline)
    # For pre-loaded query crops: strip the file loader, keep resize/normalize
    # /format/collect.
    qtrans = [t for t in pipeline.transforms
              if not isinstance(t, LoadImageFromFile)]
    qpipeline = Compose(qtrans)

    with EPS.open(newline="", encoding="utf-8") as f:
        eps = list(csv.DictReader(f))
    eps = eps[args.shard:: args.nshards]
    if args.max_episodes:
        eps = eps[: args.max_episodes]

    def run(path=None, bgr=None):
        if path is not None:
            data = dict(img_info=dict(filename=str(path)), img_prefix=None)
            data = pipeline(data)
        else:
            data = qpipeline(dict(img=bgr, img_shape=bgr.shape,
                                  ori_shape=bgr.shape, flip=False,
                                  filename="query.jpg",
                                  ori_filename="query.jpg"))
        img_t = data["img"][0].unsqueeze(0).cuda()
        metas = [[data["img_metas"][0].data]]
        with torch.no_grad():
            res = model(return_loss=False, img=[img_t], img_metas=metas,
                        rescale=True)
        arr = res[0][0]
        if arr.shape[0] == 0:
            return np.zeros((0, 4)), np.zeros((0, 768)), np.zeros((0,))
        boxes = arr[:, :4]
        scores = arr[:, 4]
        emb = arr[:, 5:]
        return boxes, emb, scores

    rows = []
    for i, r in enumerate(eps):
        seq, t, f_ = r["sequence"], int(r["t"]), int(r["f"])
        qbox = json.loads(r["human_box"])
        present = int(r["target_present"]) == 1
        target = json.loads(r["target_box"]) if present else None
        gpath = DT / "train" / seq / "img1" / f"{f_ + 1:08d}.jpg"
        qpath = DT / "train" / seq / "img1" / f"{t + 1:08d}.jpg"
        if not gpath.exists() or not qpath.exists():
            continue
        boxes, emb, scores = run(path=gpath)
        row = {
            "sequence": seq, "t": t, "gid": r["gid"], "f": f_,
            "delta": r["delta"], "present": int(present),
        }
        if len(boxes) == 0:
            row.update(top1=0, top3=0, best_iou=0.0, top1_sim=float("nan"),
                       generic_rescue=0, n_dets=0)
        else:
            ge = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-8)
            qimg = mmcv.imread(str(qpath))
            qcrop = crop_query(qimg, qbox)
            qboxes, qemb, qscores = run(bgr=np.ascontiguousarray(qcrop))
            qarr = np.concatenate([qboxes, qscores[:, None], qemb], axis=1) \
                if len(qboxes) else np.zeros((0, 5 + 768))
            if qarr.shape[0] == 0:
                qe = np.zeros(ge.shape[1], dtype=np.float32)
            else:
                qi = int(np.argmax(qarr[:, 4]))
                qe = qarr[qi, 5:]
            qe = qe / (np.linalg.norm(qe) + 1e-8)
            sims = ge @ qe
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
    with (OUT / f"pstr_hcred{tag}.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"shard{args.shard} done n={len(rows)}", flush=True)


if __name__ == "__main__":
    main()
