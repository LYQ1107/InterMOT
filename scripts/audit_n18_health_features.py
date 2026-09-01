#!/usr/bin/env python
"""Offline audit of the GFN identity-health features for delivered P0 rows.

For each delivered row (from a CPU replay of the causal loop) we compute the
GFN similarity of the matched detection to the immutable H_i anchor and score
it with the deployed LR verifier.  This calibrates the verified-memory gate
before it is deployed in the online loop.
"""

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import joblib
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
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    args = ap.parse_args()
    torch.cuda.set_device(args.gpu)
    device = f"cuda:{args.gpu}"
    model, _, _, _, _ = load_model(device)
    bundle = joblib.load(OUT / "models/verifier_v0.joblib")
    clf = bundle["model"]
    feats = bundle["features"]

    rows = []
    for line in (OUT / "tables/delivery_trace_cal10.jsonl").open(
            encoding="utf-8"):
        r = json.loads(line)
        if r["source"] in ("p0", "p0_tid") and r["gt_present"] \
                and r["delivered_box"] is not None:
            rows.append(r)
    seqs = sorted({r["sequence"] for r in rows})
    seqs = seqs[args.shard:: args.nshards]
    out = OUT / "tables"
    out.mkdir(parents=True, exist_ok=True)
    tag = f"_s{args.shard}" if args.nshards > 1 else ""
    p = out / f"health_features{tag}.csv"
    with p.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=[
            "sequence", "frame", "gid", "correct", "matched_iou",
            "gfn_top1_sim", "gfn_margin", "gfn_top1_score", "n_dets",
            "verifier_score"])
        w.writeheader()
        for seq in seqs:
            gt = load_gt(seq)
            first = {}
            for f0 in sorted(gt):
                gf = gt[f0]
                for gid, box in zip(gf.gt_ids, gf.boxes):
                    if gid not in first:
                        first[gid] = (f0, np.asarray(box, dtype=float))
            seq_rows = [r for r in rows if r["sequence"] == seq]
            frames = sorted({r["frame"] for r in seq_rows})
            gallery = {}
            for f0 in frames:
                img = frame_img(seq, f0)
                if img is None:
                    continue
                with torch.inference_mode():
                    outg = model([F.to_tensor(img).to(device)], None,
                                 inference_mode="det")[0]
                boxes = outg["det_boxes"].float().cpu().numpy()
                scores = outg["det_scores"].float().cpu().numpy()
                embs = outg["det_emb"].float().cpu()
                if boxes.ndim == 1:
                    boxes = boxes.reshape(1, -1)
                    scores = np.atleast_1d(scores)
                    embs = embs.reshape(1, -1)
                if len(boxes):
                    ge = embs / (embs.norm(dim=1, keepdim=True) + 1e-8)
                else:
                    ge = None
                gallery[f0] = (boxes, scores, ge)
            qcache = {}
            for r in seq_rows:
                f0 = r["frame"]
                gid = r["gid"]
                if f0 not in gallery:
                    continue
                boxes, scores, ge = gallery[f0]
                af, abox = first[gid]
                if af not in qcache:
                    img = frame_img(seq, af)
                    qe = None
                    if img is not None:
                        qcrop = crop_query(img, abox)
                        with torch.inference_mode():
                            qout = model(
                                [F.to_tensor(qcrop).to(device)], None,
                                inference_mode="det")[0]
                        if qout["det_emb"].shape[0]:
                            qi = int(torch.argmax(qout["det_scores"]).item())
                            qe = qout["det_emb"].float()[qi].reshape(-1).cpu()
                            qe = qe / (qe.norm() + 1e-8)
                    qcache[af] = qe
                qe = qcache[af]
                dbox = np.asarray(r["delivered_box"], dtype=float)
                matched_iou = None
                sim = margin = top_score = None
                n_dets = len(boxes)
                if qe is not None and ge is not None and len(boxes):
                    ious = [iou(b, dbox) for b in boxes]
                    best = int(np.argmax(ious))
                    matched_iou = ious[best]
                    sims = (ge @ qe).numpy()
                    order = np.argsort(-sims)
                    sim = float(sims[order[0]])
                    margin = sim - (float(sims[order[1]])
                                    if len(order) > 1 else 0.0)
                    top_score = float(scores[order[0]])
                rec = {"gfn_top1_sim": sim if sim is not None else 0.0,
                       "gfn_margin": margin if margin is not None else 0.0,
                       "gfn_top1_score": top_score if top_score is not None
                       else 0.0,
                       "n_dets": n_dets}
                x = np.asarray([[rec[k] for k in feats]], dtype=float)
                prob = float(clf.predict_proba(x)[0, 1])
                w.writerow({
                    "sequence": seq, "frame": f0, "gid": gid,
                    "correct": r["correct"],
                    "matched_iou": matched_iou,
                    "gfn_top1_sim": sim, "gfn_margin": margin,
                    "gfn_top1_score": top_score, "n_dets": n_dets,
                    "verifier_score": prob,
                })
            print(seq, len(seq_rows), flush=True)
    print(f"HEALTH_AUDIT_DONE shard={args.shard} file={p}", flush=True)


if __name__ == "__main__":
    main()
