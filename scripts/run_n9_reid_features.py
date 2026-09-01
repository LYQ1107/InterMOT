#!/usr/bin/env python
"""Extract frozen OSNet ReID features for all P0 rows of one sequence."""

import json
import os
import time
from pathlib import Path

import numpy as np
from PIL import Image
from torchreid.reid.utils.feature_extractor import FeatureExtractor


ROOT = Path(".")
SEQ = os.environ["N9_SEQ"]
SPLIT = os.environ.get("N9_SPLIT", "train")
P0_DIR = Path(
    os.environ.get(
        "N9_P0_DIR",
        str(ROOT / "outputs/n9/p0_train")
        if SPLIT == "train"
        else str(ROOT / "outputs/n5/integrity/canonical_mot_results/b0"),
    )
)
OUT = Path(os.environ.get("N9_FEAT_DIR", ROOT / "outputs/n9/features"))
CKPT = ROOT / "outputs/n9/checkpoints/osnet_x1_0_market1501.pth"
IMAGE_SIZE = (256, 128)


def main() -> None:
    video = (
        Path("/path/to/dancetrack")
        / SPLIT
        / SEQ
        / "img1"
    )
    frames = sorted(video.glob("*.jpg"))
    p0_path = P0_DIR / f"{SEQ}.txt"
    rows = {}
    for line in p0_path.read_text(encoding="utf-8").splitlines():
        parts = line.split(",")
        if len(parts) < 6:
            continue
        f0 = int(float(parts[0])) - 1
        tid = int(float(parts[1]))
        x, y, w, h = (float(parts[2]), float(parts[3]), float(parts[4]), float(parts[5]))
        if w <= 0 or h <= 0:
            continue
        rows.setdefault(f0, []).append((tid, np.asarray([x, y, x + w, y + h], float)))
    ext = FeatureExtractor(
        model_name="osnet_x1_0",
        model_path=str(CKPT),
        image_size=IMAGE_SIZE,
        device="cuda",
        verbose=False,
    )
    out_frames, out_tids, out_boxes, out_feats = [], [], [], []
    t0 = time.time()
    for fi, fp in enumerate(frames):
        if fi not in rows:
            continue
        img = Image.open(fp).convert("RGB")
        crops, meta = [], []
        for tid, box in rows[fi]:
            x1, y1, x2, y2 = [int(round(v)) for v in box]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(img.width, x2), min(img.height, y2)
            if x2 <= x1 or y2 <= y1:
                continue
            crops.append(np.asarray(img.crop((x1, y1, x2, y2)), dtype=np.uint8))
            meta.append((tid, box))
        if not crops:
            continue
        feats = ext(crops).cpu().numpy()
        for (tid, box), fv in zip(meta, feats):
            out_frames.append(fi)
            out_tids.append(tid)
            out_boxes.append(box)
            out_feats.append(fv.astype(np.float16))
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        OUT / f"{SEQ}.npz",
        frame=np.asarray(out_frames, dtype=np.int32),
        tid=np.asarray(out_tids, dtype=np.int32),
        box=np.asarray(out_boxes, dtype=np.float32),
        feat=np.asarray(out_feats, dtype=np.float16),
    )
    summary = {
        "sequence": SEQ,
        "split": SPLIT,
        "rows": len(out_frames),
        "frames_covered": len(set(out_frames)),
        "wall_seconds": round(time.time() - t0, 2),
    }
    (OUT / f"{SEQ}.summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
