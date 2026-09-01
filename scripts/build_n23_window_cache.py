#!/usr/bin/env python3
"""Extract frozen CLIP-ReID features for N23 whole-frame proposals.

Each cache item represents one causal correction episode: the query is the
human box at frame t and the proposal bank is generated geometrically for a
future frame f.  No GT is used to generate a window or a feature.  GT labels
are copied into the index only for offline evaluation.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torchvision
import torchvision.transforms as T
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sam3_intermot.recovery.n23_query_discovery import generate_windows  # noqa: E402


DT = Path("/path/to/dancetrack")
if not DT.exists():
    DT = Path("/path/to/dancetrack")


def expand_box(box: np.ndarray, image_w: int, image_h: int, margin: float = 0.20) -> np.ndarray:
    x1, y1, x2, y2 = [float(v) for v in box]
    w, h = x2 - x1, y2 - y1
    x1 -= margin * w
    y1 -= margin * h
    x2 += margin * w
    y2 += margin * h
    return np.asarray(
        [
            max(0.0, x1),
            max(0.0, y1),
            min(float(image_w), x2),
            min(float(image_h), y2),
        ],
        dtype=np.float32,
    )


def image_tensor(path: Path) -> torch.Tensor:
    """Load one RGB frame as CHW float in [0, 1]."""
    image = Image.open(path).convert("RGB")
    arr = np.asarray(image, dtype=np.uint8).copy()
    return torch.from_numpy(arr).permute(2, 0, 1).contiguous().float().div_(255.0)


def roi_batch(image: torch.Tensor, boxes: np.ndarray, device: torch.device) -> torch.Tensor:
    """GPU crop/warp, replacing the slow per-window PIL path."""
    x = image.unsqueeze(0).to(device, non_blocking=True)
    b = torch.from_numpy(np.asarray(boxes, dtype=np.float32)).to(device)
    out = torchvision.ops.roi_align(
        x,
        [b],
        output_size=(256, 128),
        spatial_scale=1.0,
        sampling_ratio=2,
        aligned=True,
    )
    return (out - 0.5) / 0.5


def select_rows(rows, max_rows: int, seed: int):
    if not max_rows or len(rows) <= max_rows:
        return rows
    rng = random.Random(seed)
    groups = {}
    for row in rows:
        key = (row.get("target_present") == "1", row.get("generic_miss") == "1")
        groups.setdefault(key, []).append(row)
    # Preserve the F3/generic-miss and NONE strata rather than selecting the
    # first rows, which are correlated by sequence and frame.
    quotas = {
        (True, True): 0.50,
        (True, False): 0.30,
        (False, False): 0.20,
        (False, True): 0.05,
    }
    selected = []
    for key, frac in quotas.items():
        pool = groups.get(key, [])
        n = min(len(pool), max(1, int(round(max_rows * frac))))
        selected.extend(rng.sample(pool, n))
    if len(selected) > max_rows:
        selected = rng.sample(selected, max_rows)
    elif len(selected) < max_rows:
        remaining = [r for r in rows if r not in selected]
        selected.extend(rng.sample(remaining, min(max_rows - len(selected), len(remaining))))
    selected.sort(key=lambda r: int(r["_row_id"]))
    return selected


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--split", required=True, choices=["train", "calibration"])
    parser.add_argument("--out-dir", default=str(ROOT / "outputs/n23/window_cache"))
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--nshards", type=int, default=1)
    parser.add_argument("--max-rows", type=int, default=0)
    parser.add_argument("--max-windows", type=int, default=600)
    parser.add_argument("--stride", type=float, default=0.85)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--seed", type=int, default=23)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("N23 feature cache requires a CUDA device")
    torch.cuda.set_device(args.gpu)
    rng = random.Random(args.seed)

    with Path(args.manifest).open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    for idx, row in enumerate(rows):
        row["_row_id"] = str(idx)
    rows = select_rows(rows, args.max_rows, args.seed)
    rows = rows[args.shard :: args.nshards]
    print(
        f"split={args.split} shard={args.shard}/{args.nshards} rows={len(rows)}",
        flush=True,
    )

    from scripts.run_n15_extract_features import build_clipreid

    model = build_clipreid(
        str(ROOT / "outputs/n15/checkpoints/clip_reid_hf/person_vit_clip_reid.pth"),
        "cuda",
    )
    cache_dir = Path(args.out_dir) / args.split
    cache_dir.mkdir(parents=True, exist_ok=True)
    index_rows = []
    t0 = time.time()
    for n, row in enumerate(rows, 1):
        row_id = int(row["_row_id"])
        file_name = f"row_{row_id:06d}.npz"
        out_path = cache_dir / file_name
        hb = np.asarray(json.loads(row["human_box"]), dtype=np.float32)
        f = int(row["f"])
        seq = row["sequence"]
        image_path = DT / "train" / seq / "img1" / f"{f + 1:08d}.jpg"
        image = Image.open(image_path).convert("RGB")
        image_array = np.asarray(image, dtype=np.uint8).copy()
        image_tensor_f = (
            torch.from_numpy(image_array)
            .permute(2, 0, 1)
            .contiguous()
            .float()
            .div_(255.0)
        )
        boxes = generate_windows(
            hb,
            image_w=float(image.width),
            image_h=float(image.height),
            stride_fraction=args.stride,
            max_windows=args.max_windows,
        )
        query_image = image_tensor(
            DT / "train" / seq / "img1" / f"{int(row['t']) + 1:08d}.jpg"
        )
        query_box = expand_box(hb, query_image.shape[-1], query_image.shape[-2])
        with torch.inference_mode():
            qx = roi_batch(query_image, query_box.reshape(1, 4), torch.device("cuda"))
            _, q12, qproj = model(qx)
            qfeat = torch.nn.functional.normalize(
                torch.cat([q12[:, 0], qproj[:, 0]], dim=1), dim=-1
            )[0].cpu().numpy().astype(np.float16)
        features = []
        with torch.inference_mode():
            for start in range(0, len(boxes), args.batch_size):
                batch_boxes = boxes[start : start + args.batch_size]
                xs = roi_batch(image_tensor_f, batch_boxes, torch.device("cuda"))
                _, x12, xproj = model(xs)
                feat = torch.nn.functional.normalize(
                    torch.cat([x12[:, 0], xproj[:, 0]], dim=1), dim=-1
                )
                features.append(feat.cpu().numpy().astype(np.float16))
        emb = np.concatenate(features, axis=0)
        target = (
            np.asarray(json.loads(row["target_box"]), dtype=np.float32)
            if row.get("target_box")
            else np.full(4, -1.0, dtype=np.float32)
        )
        np.savez_compressed(
            out_path,
            query=qfeat,
            boxes=boxes,
            embeddings=emb,
            target_box=target,
            target_present=np.asarray(int(row["target_present"]), dtype=np.int8),
            gid=np.asarray(int(row["gid"]), dtype=np.int32),
            delta=np.asarray(float(row["delta"]), dtype=np.float32),
        )
        index_rows.append(
            {
                "cache_file": file_name,
                "row_id": row_id,
                "sequence": seq,
                "t": row["t"],
                "f": row["f"],
                "gid": row["gid"],
                "delta": row["delta"],
                "target_present": row["target_present"],
                "generic_miss": row.get("generic_miss", ""),
                "human_box": row["human_box"],
                "target_box": row.get("target_box", ""),
            }
        )
        if n % 25 == 0 or n == len(rows):
            print(
                f"done={n}/{len(rows)} windows={len(boxes)} "
                f"elapsed={time.time() - t0:.1f}s",
                flush=True,
            )
    index_path = cache_dir / f"index_shard{args.shard}.csv"
    if index_rows:
        with index_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(index_rows[0]))
            writer.writeheader()
            writer.writerows(index_rows)
    print(f"CACHE_DONE {index_path}", flush=True)


if __name__ == "__main__":
    main()
