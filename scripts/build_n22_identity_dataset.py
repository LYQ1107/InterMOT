#!/usr/bin/env python
"""Build the N22 live-aligned R0 identity dataset.

The historical N21 builder used the shadow start frame (f0) as visual token
zero.  The live runner waits one frame before collecting evidence, so this
builder uses exactly f0+1 ... f0+H.  It keeps only the R0 identity coordinate
needed by CDCIA; the N22 source audit already records the GFN comparison.

GT ``correct`` values are retained only as offline labels.  They are not
available to the live adapter or to any score used in a future frame.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


ROOT = Path(".")
N20 = ROOT / "outputs/n20"
OUT = ROOT / "outputs/n22/datasets"
BASE_CACHE = ROOT / "outputs/n18/route_c/gfn_cache"
R0_CACHE = N20 / "gfn_cache_r0"


def iou(a, b):
    ax1, ay1, ax2, ay2 = [float(x) for x in a]
    bx1, by1, bx2, by2 = [float(x) for x in b]
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    aa = max(1e-8, (ax2 - ax1) * (ay2 - ay1))
    bb = max(1e-8, (bx2 - bx1) * (by2 - by1))
    return inter / (aa + bb - inter)


class SequenceCache:
    def __init__(self, seq):
        z = np.load(BASE_CACHE / f"{seq}.npz")
        qz = np.load(BASE_CACHE / f"{seq}_queries.npz")
        rz = np.load(R0_CACHE / f"{seq}.npz")
        self.frames = z["frames"].astype(np.int64)
        self.offsets = z["offsets"].astype(np.int64)
        self.boxes = z["boxes"].astype(np.float32)
        r0 = rz["r0g"].astype(np.float32)
        self.r0 = r0 / (np.linalg.norm(r0, axis=1, keepdims=True) + 1e-8)
        qr0 = rz["r0q"].astype(np.float32)
        self.qr0 = qr0 / (np.linalg.norm(qr0, axis=1, keepdims=True) + 1e-8)
        self.qgids = [int(x) for x in qz["gids"]]
        self.qindex = {gid: i for i, gid in enumerate(self.qgids)}
        z.close()
        qz.close()
        rz.close()

    def root(self, gid):
        idx = self.qindex.get(int(gid))
        return None if idx is None else self.qr0[idx]

    def detection(self, frame, box):
        if box is None:
            return None
        pos = int(np.searchsorted(self.frames, int(frame)))
        if pos >= len(self.frames) or int(self.frames[pos]) != int(frame):
            return None
        lo = int(self.offsets[pos - 1]) if pos else 0
        hi = int(self.offsets[pos])
        if hi <= lo:
            return None
        box = np.asarray(box, dtype=np.float32)
        vals = np.asarray([iou(x, box) for x in self.boxes[lo:hi]])
        best = int(np.argmax(vals))
        if float(vals[best]) < 0.5:
            return None
        return self.r0[lo + best]


def build(split, cache_dir, h, out_name):
    cache = Path(cache_dir) if cache_dir else N20 / f"full_shadow_cache_{split}"
    files = sorted(cache.glob("*.jsonl"))
    if not files:
        raise FileNotFoundError(f"no shadow jsonl files under {cache}")
    zcache = {}
    atts, seqs, frames, gids, ranks = [], [], [], [], []
    labels, labels_h, vis, vis_mask, roots = [], [], [], [], []
    matched = 0
    total_boxes = 0
    for fp in files:
        with fp.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                seq = str(row["sequence"])
                if seq not in zcache:
                    zcache[seq] = SequenceCache(seq)
                z = zcache[seq]
                root = z.root(row["gid"])
                if root is None:
                    continue
                by_frame = {
                    int(fr["frame"]): fr
                    for fr in row.get("frames", [])
                    if "frame" in fr
                }
                vectors, masks, labs = [], [], []
                for step in range(1, h + 1):
                    fr = by_frame.get(int(row["frame"]) + step)
                    box = None if fr is None else fr.get("box")
                    total_boxes += int(box is not None)
                    r0 = z.detection(int(row["frame"]) + step, box)
                    if r0 is None:
                        vectors.append(np.zeros(2048, dtype=np.float32))
                        masks.append(0.0)
                    else:
                        vectors.append(r0)
                        masks.append(1.0)
                        matched += 1
                    labs.append(int(fr.get("correct") or 0) if fr else 0)
                atts.append(f"{seq}:{int(row['frame'])}:{int(row['gid'])}")
                seqs.append(seq)
                frames.append(int(row["frame"]))
                gids.append(int(row["gid"]))
                ranks.append(int(row["candidate_rank"]))
                labels.append(int(labs[-1]))
                labels_h.append(labs)
                vis.append(np.asarray(vectors, dtype=np.float32))
                vis_mask.append(np.asarray(masks, dtype=np.float32))
                roots.append(np.asarray(root, dtype=np.float32))
    OUT.mkdir(parents=True, exist_ok=True)
    out = OUT / (out_name or f"{split}_aligned.npz")
    np.savez_compressed(
        out,
        att=np.asarray(atts),
        seq=np.asarray(seqs),
        frame=np.asarray(frames, dtype=np.int64),
        gid=np.asarray(gids, dtype=np.int64),
        rank=np.asarray(ranks, dtype=np.int64),
        label=np.asarray(labels, dtype=np.int64),
        label_by_h=np.asarray(labels_h, dtype=np.int64),
        r0=np.asarray(vis, dtype=np.float32),
        r0_mask=np.asarray(vis_mask, dtype=np.float32),
        root_r0=np.asarray(roots, dtype=np.float32),
    )
    stats = {
        "split": split,
        "source": str(cache),
        "h": h,
        "candidate_rows": len(atts),
        "attempts": len(set(atts)),
        "sequences": len(set(seqs)),
        "matched_frame_embeddings": matched,
        "box_frames": total_boxes,
        "embedding_coverage": round(matched / max(1, total_boxes), 6),
        "positive_final_rows": int(sum(labels)),
        "positive_any_rows": int(np.asarray(labels_h).any(axis=1).sum()),
        "output": str(out),
    }
    print(json.dumps(stats, indent=2), flush=True)
    return stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["train30", "cal10"], default="")
    ap.add_argument("--cache-dir", default="")
    ap.add_argument("--h", type=int, default=8)
    ap.add_argument("--out-name", default="")
    args = ap.parse_args()
    splits = [args.split] if args.split else ["train30", "cal10"]
    all_stats = []
    for split in splits:
        all_stats.append(build(split, args.cache_dir, args.h, args.out_name))
    (OUT / "dataset_stats.json").write_text(
        json.dumps(all_stats, indent=2), encoding="utf-8")
    print("N22_DATASET_DONE", flush=True)


if __name__ == "__main__":
    main()
