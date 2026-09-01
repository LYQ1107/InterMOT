#!/usr/bin/env python
"""N21 Phase-II: build a multi-frame tracklet identity dataset.

For every real SAM3 shadow hypothesis in the N20 all-candidate shadow
cache, per-frame visual identity embeddings are recovered by IoU-matching
the shadow box to the frozen GFN detection cache (GFN 2048-d embedding +
R0 2048-d identity embedding). The Human Root query embedding is attached
per attempt. GT is used only for offline labels.

Outputs:
  outputs/n21/tracklet_identity_dataset/train30.npz
  outputs/n21/tracklet_identity_dataset/cal10.npz
  outputs/n21/tracklet_identity_dataset_stats.json
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(".")
N20 = ROOT / "outputs/n20"
OUT = ROOT / "outputs/n21/tracklet_identity_dataset"
CACHE = N20 / "full_shadow_cache_train30"


def iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    aa = max(1e-6, (ax2 - ax1) * (ay2 - ay1))
    bb = max(1e-6, (bx2 - bx1) * (by2 - by1))
    return inter / (aa + bb - inter)


def load_seq(seq):
    z = np.load(ROOT / "outputs/n18/route_c/gfn_cache" / f"{seq}.npz")
    qz = np.load(ROOT / "outputs/n18/route_c/gfn_cache" /
                 f"{seq}_queries.npz")
    rz = np.load(N20 / "gfn_cache_r0" / f"{seq}.npz")
    emb = z["emb"].astype(np.float32)
    emb = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-8)
    r0 = rz["r0g"].astype(np.float32)
    r0 = r0 / (np.linalg.norm(r0, axis=1, keepdims=True) + 1e-8)
    qemb = qz["qemb"].astype(np.float32)
    qemb = qemb / (np.linalg.norm(qemb, axis=1, keepdims=True) + 1e-8)
    qr0 = rz["r0q"].astype(np.float32)
    qr0 = qr0 / (np.linalg.norm(qr0, axis=1, keepdims=True) + 1e-8)
    return {
        "frames": z["frames"], "offsets": z["offsets"],
        "boxes": z["boxes"], "emb": emb, "r0": r0,
        "qgids": [int(g) for g in qz["gids"]], "qemb": qemb, "qr0": qr0,
    }


def det_at(z, frame, box):
    o = int(np.searchsorted(z["frames"], frame))
    lo = int(z["offsets"][o - 1]) if o > 0 else 0
    hi = int(z["offsets"][o])
    if hi == lo:
        return None
    cand = [(i, b) for i, b in enumerate(z["boxes"][lo:hi])
            if b is not None]
    if not cand:
        return None
    idxs = [lo + i for i, _ in cand]
    ious = np.asarray([iou(b, box) for _, b in cand])
    bi = int(np.argmax(ious))
    if ious[bi] < 0.5:
        return None
    gi = idxs[bi]
    return z["emb"][gi], z["r0"][gi]


def build_split(split, h=8, cache_dir=None, out_name=None):
    if cache_dir:
        cache = (Path(cache_dir) if str(cache_dir).startswith("/")
                 else N20 / cache_dir)
    else:
        cache = N20 / f"full_shadow_cache_{split}"
    files = sorted(cache.glob("*.jsonl"))
    zcache = {}
    atts = []
    seqs = []
    frames = []
    gids = []
    ranks = []
    labels = []
    any_correct = []
    vis = []
    vis_mask = []
    roots = []
    start_boxes = []
    n_rows = 0
    n_matched = 0
    for fp in files:
        with fp.open(encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                seq = r["sequence"]
                if seq not in zcache:
                    zcache[seq] = load_seq(seq)
                z = zcache[seq]
                qidx = {g: i for i, g in enumerate(z["qgids"])}
                qi = qidx.get(r["gid"])
                if qi is None:
                    continue
                root = np.concatenate([z["qemb"][qi], z["qr0"][qi]])
                feats = []
                mask = []
                lab = 0
                for i, fr in enumerate(r["frames"][:h]):
                    de = det_at(z, fr["frame"], fr["box"]) \
                        if fr.get("box") is not None else None
                    if de is not None:
                        feats.append(np.concatenate([de[0], de[1]]))
                        mask.append(1.0)
                        n_matched += 1
                    else:
                        feats.append(np.zeros(4096, dtype=np.float32))
                        mask.append(0.0)
                    lab = int(fr["correct"] or 0)
                while len(feats) < h:
                    feats.append(np.zeros(4096, dtype=np.float32))
                    mask.append(0.0)
                atts.append(f"{seq}:{r['frame']}:{r['gid']}")
                seqs.append(seq)
                frames.append(int(r["frame"]))
                gids.append(int(r["gid"]))
                ranks.append(int(r["candidate_rank"]))
                labels.append(lab)
                any_correct.append(
                    int(any(fr["correct"] for fr in r["frames"][:h])))
                vis.append(np.asarray(feats, dtype=np.float32))
                vis_mask.append(np.asarray(mask, dtype=np.float32))
                roots.append(root)
                start_boxes.append(r["start_box"])
                n_rows += 1
    out = {
        "att": np.asarray(atts), "seq": np.asarray(seqs),
        "frame": np.asarray(frames, dtype=np.int64),
        "gid": np.asarray(gids, dtype=np.int64),
        "rank": np.asarray(ranks, dtype=np.int64),
        "label": np.asarray(labels, dtype=np.int64),
        "any_correct": np.asarray(any_correct, dtype=np.int64),
        "vis": np.asarray(vis, dtype=np.float32),
        "vis_mask": np.asarray(vis_mask, dtype=np.float32),
        "root": np.asarray(roots, dtype=np.float32),
        "start_box": np.asarray(start_boxes, dtype=np.float32),
    }
    np.savez_compressed(OUT / f"{out_name or split}.npz", **out)
    return {
        "split": split,
        "candidate_rows": int(n_rows),
        "matched_frame_embs": int(n_matched),
        "coverage": round(n_matched / max(1, int(n_rows * h)), 4),
        "attempts": len(set(atts)),
        "correct_candidates": int(sum(labels)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--h", type=int, default=8)
    ap.add_argument("--cache-dir", default="")
    ap.add_argument("--out-name", default="")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if args.cache_dir:
        stats = [build_split("live", args.h, args.cache_dir,
                             args.out_name or "live_train30")]
    else:
        stats = [build_split("train30", args.h),
                 build_split("cal10", args.h)]
    (ROOT / "outputs/n21/tracklet_identity_dataset_stats.json").write_text(
        json.dumps(stats, indent=2), encoding="utf-8")
    print(json.dumps(stats, indent=2), flush=True)
    print("BUILD_DONE", flush=True)


if __name__ == "__main__":
    main()
