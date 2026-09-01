#!/usr/bin/env python3
"""Extract one recoverable fp16 CLIP-ReID shard for N27."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from torch.utils.data import DataLoader, Dataset


ROOT = Path(".")
OUT = ROOT / "outputs/n27"
DATA = OUT / "data"
CHECKPOINT = ROOT / "outputs/n15/checkpoints/clip_reid_hf/person_vit_clip_reid.pth"
sys.path.insert(0, str(ROOT / "scripts"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_rows(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    rows.sort(key=lambda row: (row["image_path"], row["crop_id"]))
    return rows


class ImageGroupDataset(Dataset):
    """Decode each source frame once and emit all requested crops from it."""

    def __init__(self, rows: list[dict]):
        groups: list[tuple[str, list[tuple[int, dict]]]] = []
        current_path = None
        current_rows: list[tuple[int, dict]] = []
        for index, row in enumerate(rows):
            if row["image_path"] != current_path:
                if current_rows:
                    groups.append((str(current_path), current_rows))
                current_path = row["image_path"]
                current_rows = []
            current_rows.append((index, row))
        if current_rows:
            groups.append((str(current_path), current_rows))
        self.groups = groups
        self.transform = T.Compose(
            [
                T.Resize((256, 128), interpolation=T.InterpolationMode.BICUBIC),
                T.ToTensor(),
                T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )

    def __len__(self) -> int:
        return len(self.groups)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        image_path, rows = self.groups[index]
        indices = []
        tensors = []
        with Image.open(image_path) as handle:
            image = handle.convert("RGB")
            for local_index, row in rows:
                x1, y1, x2, y2 = [int(round(float(value))) for value in row["box"]]
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(image.width, x2), min(image.height, y2)
                crop = image.crop((x1, y1, x2, y2)) if x2 > x1 and y2 > y1 else Image.new("RGB", (8, 8))
                indices.append(local_index)
                tensors.append(self.transform(crop))
        return torch.tensor(indices, dtype=torch.int64), torch.stack(tensors)


def worker_init(_: int) -> None:
    torch.set_num_threads(1)


def valid_artifact(npz_path: Path, metadata_path: Path, done_path: Path, expected: dict) -> bool:
    if not (npz_path.is_file() and metadata_path.is_file() and done_path.is_file()):
        return False
    try:
        done = json.loads(done_path.read_text(encoding="utf-8"))
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        return (
            done.get("npz_sha256") == sha256(npz_path)
            and done.get("metadata_sha256") == sha256(metadata_path)
            and all(metadata.get(key) == value for key, value in expected.items())
        )
    except (OSError, ValueError, json.JSONDecodeError):
        return False


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard", type=int, required=True)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--part-size", type=int, default=8192)
    args = parser.parse_args()
    request_path = DATA / f"crop_requests_shard{args.shard}.jsonl"
    output_path = DATA / f"clipreid_shard{args.shard}.npz"
    metadata_path = DATA / f"clipreid_shard{args.shard}.json"
    done_path = DATA / f"clipreid_shard{args.shard}.done"
    request_sha = sha256(request_path)
    checkpoint_sha = sha256(CHECKPOINT)
    if valid_artifact(output_path, metadata_path, done_path, {"request_sha256": request_sha, "checkpoint_sha256": checkpoint_sha}):
        print(f"N27_FEATURE_SHARD_SKIP_COMPLETE shard={args.shard}", flush=True)
        return
    rows = load_rows(request_path)
    if not rows:
        raise RuntimeError(f"empty request shard {args.shard}")
    torch.cuda.set_device(args.device)
    device = torch.device(f"cuda:{args.device}")
    torch.backends.cuda.matmul.allow_tf32 = True
    from run_n15_extract_features import build_clipreid

    model = build_clipreid(str(CHECKPOINT), device).half().eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    part_dir = DATA / "feature_parts" / f"shard{args.shard}"
    part_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()
    part_records = []
    reused_parts = 0
    for part_index, start in enumerate(range(0, len(rows), args.part_size)):
        end = min(start + args.part_size, len(rows))
        part_rows = rows[start:end]
        part_npz = part_dir / f"part{part_index:03d}.npz"
        part_meta = part_dir / f"part{part_index:03d}.json"
        part_done = part_dir / f"part{part_index:03d}.done"
        expected = {
            "request_sha256": request_sha,
            "checkpoint_sha256": checkpoint_sha,
            "start": start,
            "end": end,
            "precision": "model_and_input_fp16",
        }
        if valid_artifact(part_npz, part_meta, part_done, expected):
            reused_parts += 1
            part_records.append(json.loads(part_meta.read_text(encoding="utf-8")))
            print(f"SHARD {args.shard} PART {part_index} REUSED {start}:{end}", flush=True)
            continue
        part_started = time.time()
        dataset = ImageGroupDataset(part_rows)
        loader = DataLoader(
            dataset,
            batch_size=None,
            shuffle=False,
            num_workers=args.workers,
            pin_memory=True,
            persistent_workers=args.workers > 0,
            prefetch_factor=2 if args.workers > 0 else None,
            worker_init_fn=worker_init if args.workers > 0 else None,
        )
        part_embeddings = np.empty((len(part_rows), 1280), dtype=np.float16)
        pending_indices = []
        pending_tensors = []
        pending_count = 0

        def flush_pending() -> None:
            nonlocal pending_indices, pending_tensors, pending_count
            if not pending_tensors:
                return
            indices = torch.cat(pending_indices).numpy()
            tensors = torch.cat(pending_tensors)
            for batch_start in range(0, len(indices), args.batch_size):
                batch_end = min(batch_start + args.batch_size, len(indices))
                inputs = tensors[batch_start:batch_end].to(device, dtype=torch.float16, non_blocking=True)
                with torch.inference_mode():
                    _, x12, xproj = model(inputs)
                    features = F.normalize(torch.cat([x12[:, 0], xproj[:, 0]], dim=1).float(), dim=1)
                part_embeddings[indices[batch_start:batch_end]] = features.cpu().numpy().astype(np.float16)
            pending_indices, pending_tensors, pending_count = [], [], 0

        for indices, tensors in loader:
            pending_indices.append(indices)
            pending_tensors.append(tensors)
            pending_count += len(indices)
            if pending_count >= args.batch_size:
                flush_pending()
        flush_pending()
        if not np.isfinite(part_embeddings).all():
            raise RuntimeError(f"non-finite embedding in shard {args.shard} part {part_index}")
        part_norms = np.linalg.norm(part_embeddings.astype(np.float32), axis=1)
        max_norm_error = float(np.max(np.abs(part_norms - 1.0)))
        if max_norm_error > 0.002:
            raise RuntimeError(f"normalization regression in shard {args.shard} part {part_index}: {max_norm_error}")
        part_crop_ids = np.asarray([row["crop_id"].encode("ascii") for row in part_rows], dtype="S32")
        temporary_part = part_npz.with_suffix(".npz.tmp")
        with temporary_part.open("wb") as handle:
            np.savez_compressed(handle, crop_id=part_crop_ids, embedding=part_embeddings)
        os.replace(temporary_part, part_npz)
        part_elapsed = time.time() - part_started
        part_metadata = {
            **expected,
            "phase": "N27",
            "status": "COMPLETE",
            "shard": args.shard,
            "part": part_index,
            "requests": end - start,
            "runtime_seconds": part_elapsed,
            "throughput_crops_per_second": (end - start) / part_elapsed,
            "max_norm_error": max_norm_error,
        }
        temporary_meta = part_meta.with_suffix(".json.tmp")
        temporary_meta.write_text(json.dumps(part_metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        os.replace(temporary_meta, part_meta)
        temporary_done = part_done.with_suffix(".done.tmp")
        temporary_done.write_text(json.dumps({"npz_sha256": sha256(part_npz), "metadata_sha256": sha256(part_meta)}, sort_keys=True) + "\n", encoding="utf-8")
        os.replace(temporary_done, part_done)
        part_records.append(part_metadata)
        print(f"SHARD {args.shard} PART {part_index} COMPLETE {start}:{end} rate={(end-start)/part_elapsed:.1f}/s", flush=True)

    embeddings = np.empty((len(rows), 1280), dtype=np.float16)
    crop_ids = np.empty(len(rows), dtype="S32")
    for part_index, start in enumerate(range(0, len(rows), args.part_size)):
        end = min(start + args.part_size, len(rows))
        with np.load(part_dir / f"part{part_index:03d}.npz") as payload:
            crop_ids[start:end] = payload["crop_id"]
            embeddings[start:end] = payload["embedding"]
    expected_ids = np.asarray([row["crop_id"].encode("ascii") for row in rows], dtype="S32")
    if not np.array_equal(crop_ids, expected_ids):
        raise RuntimeError(f"crop-id merge regression in shard {args.shard}")
    norms = np.linalg.norm(embeddings.astype(np.float32), axis=1)
    temporary = output_path.with_suffix(".npz.tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, crop_id=crop_ids, embedding=embeddings)
    os.replace(temporary, output_path)
    elapsed = time.time() - started
    metadata = {
        "phase": "N27",
        "status": "COMPLETE",
        "shard": args.shard,
        "requests": len(rows),
        "request_path": str(request_path),
        "request_sha256": request_sha,
        "output_path": str(output_path),
        "checkpoint_path": str(CHECKPOINT),
        "checkpoint_sha256": checkpoint_sha,
        "backbone": "frozen CLIP-ReID ViT-B/16, concatenated x12 CLS and projected CLS",
        "embedding_dimension": 1280,
        "dtype": "float16",
        "normalized": True,
        "batch_size": args.batch_size,
        "workers": args.workers,
        "part_size": args.part_size,
        "parts": len(part_records),
        "resume_parts_reused": reused_parts,
        "precision": "model_and_input_fp16",
        "runtime_seconds": elapsed,
        "throughput_crops_per_second": len(rows) / elapsed,
        "sum_part_extraction_seconds": float(sum(record["runtime_seconds"] for record in part_records)),
        "gpu_name": torch.cuda.get_device_name(device),
        "max_norm_error": float(np.max(np.abs(norms - 1.0))),
        "test_labels_used": False,
        "val25_read": False,
    }
    temporary_meta = metadata_path.with_suffix(".json.tmp")
    temporary_meta.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary_meta, metadata_path)
    done = {"npz_sha256": sha256(output_path), "metadata_sha256": sha256(metadata_path)}
    temporary_done = done_path.with_suffix(".done.tmp")
    temporary_done.write_text(json.dumps(done, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary_done, done_path)
    print(f"N27_FEATURE_SHARD_COMPLETE shard={args.shard} crops={len(rows)} runtime={elapsed:.1f}s", flush=True)


if __name__ == "__main__":
    main()
