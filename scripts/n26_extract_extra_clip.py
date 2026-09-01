#!/usr/bin/env python3
"""Extract frozen CLIP-ReID tokens for legal attempts filtered out of N25-R.

N20's `target_present=0` means that the GFN gallery has no GT-matched
detection, not that the scene target is absent.  The static top-5 candidates
are reconstructed exactly from the human-root GFN query, which reproduces all
audited N25-R candidate ranks.  This script never reads val25.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
import torchvision.transforms as T


ROOT = Path(".")
DATA = Path("/path/to/dancetrack/train")
GFN = ROOT / "outputs/n18/route_c/gfn_cache"
R0 = ROOT / "outputs/n20/gfn_cache_r0"
OUT = ROOT / "outputs/n26/extra_clip"
CLIP_CKPT = ROOT / "outputs/n15/checkpoints/clip_reid_hf/person_vit_clip_reid.pth"
sys.path.insert(0, str(ROOT / "scripts"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def crop(image: Image.Image, box: np.ndarray) -> Image.Image:
    x1, y1, x2, y2 = [int(round(float(value))) for value in box]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(image.width, x2), min(image.height, y2)
    if x2 <= x1 or y2 <= y1:
        return Image.new("RGB", (8, 8))
    return image.crop((x1, y1, x2, y2))


def normalized(array: np.ndarray) -> np.ndarray:
    return array / (np.linalg.norm(array, axis=-1, keepdims=True) + 1e-8)


def assigned_sequences(rows: list[dict], shard: int, num_shards: int) -> set[str]:
    weight = defaultdict(int)
    for row in rows:
        weight[row["sequence"]] += 1
    loads = [0] * num_shards
    groups: list[list[str]] = [[] for _ in range(num_shards)]
    for sequence, count in sorted(weight.items(), key=lambda item: (-item[1], item[0])):
        target = min(range(num_shards), key=lambda index: (loads[index], index))
        groups[target].append(sequence)
        loads[target] += count
    return set(groups[shard])


def extract_features(model, transform, device: torch.device, requests: list[tuple[Path, np.ndarray]], batch_size: int) -> np.ndarray:
    output = np.empty((len(requests), 1280), dtype=np.float16)
    for start in range(0, len(requests), batch_size):
        block = requests[start : start + batch_size]
        tensors = []
        for image_path, box in block:
            with Image.open(image_path) as handle:
                tensors.append(transform(crop(handle.convert("RGB"), box)))
        inputs = torch.stack(tensors).to(device, non_blocking=True)
        with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.float16):
            _, x12, xproj = model(inputs)
            feature = F.normalize(torch.cat([x12[:, 0], xproj[:, 0]], dim=1).float(), dim=1)
        output[start : start + len(block)] = feature.cpu().numpy().astype(np.float16)
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--shard", type=int, required=True)
    parser.add_argument("--num-shards", type=int, required=True)
    parser.add_argument("--batch-size", type=int, default=96)
    args = parser.parse_args()
    torch.cuda.set_device(args.gpu)
    device = torch.device(f"cuda:{args.gpu}")
    all_rows = []
    for split in ("train30", "cal10"):
        path = ROOT / "outputs/n20" / f"dataset_attempts_{split}.csv"
        for row in csv.DictReader(path.open(newline="", encoding="utf-8")):
            if row.get("target_present") == "0":
                all_rows.append({"split": split, **row})
    selected = assigned_sequences(all_rows, args.shard, args.num_shards)
    rows = [row for row in all_rows if row["sequence"] in selected]
    print(json.dumps({"gpu": args.gpu, "shard": args.shard, "num_shards": args.num_shards, "sequences": sorted(selected), "events": len(rows)}), flush=True)

    from run_n15_extract_features import build_clipreid

    model = build_clipreid(str(CLIP_CKPT), device)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    transform = T.Compose(
        [
            T.Resize((256, 128), interpolation=T.InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )
    started = time.time()
    by_sequence = defaultdict(list)
    for row in rows:
        by_sequence[(row["split"], row["sequence"])].append(row)

    for (split, sequence), sequence_rows in sorted(by_sequence.items()):
        sequence_rows.sort(key=lambda row: (int(row["frame"]), int(row["gid"])))
        output_dir = OUT / split
        output_dir.mkdir(parents=True, exist_ok=True)
        path = output_dir / f"{sequence}.npz"
        metadata_path = output_dir / f"{sequence}.json"
        done_path = output_dir / f"{sequence}.done"
        if path.is_file() and metadata_path.is_file() and done_path.is_file():
            print(f"SKIP_DONE {split}/{sequence}", flush=True)
            continue
        with np.load(GFN / f"{sequence}.npz") as gallery, np.load(GFN / f"{sequence}_queries.npz") as query, np.load(R0 / f"{sequence}.npz") as r0:
            frames = gallery["frames"].astype(np.int64)
            offsets = gallery["offsets"].astype(np.int64)
            boxes_all = gallery["boxes"].astype(np.float32)
            detection_score_all = gallery["scores"].astype(np.float32)
            gfn_all = normalized(gallery["emb"].astype(np.float32))
            r0_all = normalized(r0["r0g"].astype(np.float32))
            query_gfn = normalized(query["qemb"].astype(np.float32))
            query_r0 = normalized(r0["r0q"].astype(np.float32))
            query_index = {int(gid): index for index, gid in enumerate(query["gids"])}
            query_frame_all = query["qframe"].astype(np.int32)
            query_box_all = query["qbox"].astype(np.float32)
            count = len(sequence_rows)
            frame_out = np.zeros(count, dtype=np.int32)
            gid_out = np.zeros(count, dtype=np.int32)
            query_frame_out = np.zeros(count, dtype=np.int32)
            query_box_out = np.zeros((count, 4), dtype=np.float32)
            candidate_box_out = np.full((count, 5, 4), np.nan, dtype=np.float32)
            candidate_mask = np.zeros((count, 5), dtype=bool)
            gfn_similarity = np.full((count, 5), np.nan, dtype=np.float32)
            r0_similarity = np.full((count, 5), np.nan, dtype=np.float32)
            detection_score = np.full((count, 5), np.nan, dtype=np.float32)
            crop_requests: list[tuple[Path, np.ndarray]] = []
            destinations: list[tuple[str, int, int]] = []
            for event_index, row in enumerate(sequence_rows):
                frame, gid = int(row["frame"]), int(row["gid"])
                frame_out[event_index], gid_out[event_index] = frame, gid
                qi = query_index.get(gid)
                if qi is None:
                    raise RuntimeError(f"missing query {sequence}/{gid}")
                query_frame_out[event_index] = query_frame_all[qi]
                query_box_out[event_index] = query_box_all[qi]
                image_query = DATA / sequence / "img1" / f"{int(query_frame_all[qi]) + 1:08d}.jpg"
                crop_requests.append((image_query, query_box_all[qi]))
                destinations.append(("query", event_index, 0))
                position = int(np.searchsorted(frames, frame))
                if position >= len(frames) or int(frames[position]) != frame:
                    raise RuntimeError(f"gallery frame missing {sequence}/{frame}")
                lo = int(offsets[position - 1]) if position else 0
                hi = int(offsets[position])
                similarity = gfn_all[lo:hi] @ query_gfn[qi]
                order = np.argsort(-similarity)[:5]
                for rank, local in enumerate(order):
                    index = lo + int(local)
                    candidate_box_out[event_index, rank] = boxes_all[index]
                    candidate_mask[event_index, rank] = True
                    gfn_similarity[event_index, rank] = float(gfn_all[index] @ query_gfn[qi])
                    r0_similarity[event_index, rank] = float(r0_all[index] @ query_r0[qi])
                    detection_score[event_index, rank] = detection_score_all[index]
                    image_candidate = DATA / sequence / "img1" / f"{frame + 1:08d}.jpg"
                    crop_requests.append((image_candidate, boxes_all[index]))
                    destinations.append(("candidate", event_index, rank))
        extracted = extract_features(model, transform, device, crop_requests, args.batch_size)
        query_clip = np.full((len(sequence_rows), 1280), np.nan, dtype=np.float16)
        candidate_clip = np.full((len(sequence_rows), 5, 1280), np.nan, dtype=np.float16)
        for feature, (kind, event_index, rank) in zip(extracted, destinations):
            if kind == "query":
                query_clip[event_index] = feature
            else:
                candidate_clip[event_index, rank] = feature
        if not np.isfinite(query_clip).all() or not np.isfinite(candidate_clip[candidate_mask]).all():
            raise RuntimeError(f"nonfinite extraction {split}/{sequence}")
        payload = {
            "frame": frame_out,
            "gid": gid_out,
            "query_frame": query_frame_out,
            "query_box": query_box_out,
            "query_clip": query_clip,
            "candidate_box": candidate_box_out,
            "candidate_mask": candidate_mask,
            "candidate_clip": candidate_clip,
            "gfn_similarity": gfn_similarity,
            "r0_similarity": r0_similarity,
            "detection_score": detection_score,
        }
        temporary = path.with_suffix(".npz.tmp")
        with temporary.open("wb") as handle:
            np.savez_compressed(handle, **payload)
        os.replace(temporary, path)
        metadata = {
            "status": "COMPLETE",
            "split": split,
            "sequence": sequence,
            "events": len(sequence_rows),
            "candidate_rows": int(candidate_mask.sum()),
            "candidate_protocol": "static human-root GFN top-5; exact protocol reproduced all 2,200 N25-R parent ranks",
            "feature": "frozen N15 CLIP-ReID symmetric crops",
            "upstream_filter": "target_present == 0 (GFN-gallery matched target absent; not scene absence)",
            "val25_read": False,
            "runtime_seconds": time.time() - started,
            "gpu": args.gpu,
        }
        metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        done_path.write_text(json.dumps({"npz_sha256": sha256(path), "metadata_sha256": sha256(metadata_path)}) + "\n", encoding="utf-8")
        print(f"DONE {split}/{sequence} events={len(sequence_rows)} candidates={int(candidate_mask.sum())}", flush=True)
    print(f"N26_EXTRA_CLIP_DONE shard={args.shard} runtime_s={time.time()-started:.1f}", flush=True)


if __name__ == "__main__":
    main()
