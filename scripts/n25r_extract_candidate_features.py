#!/usr/bin/env python
"""Extract candidate-aligned N25-R CLIP-ReID or SAM3 F1 features.

The output contract is deliberately sequence-local and resumable.  Each
sequence is first written to a temporary file, atomically renamed, validated,
and only then receives a ``.done`` marker.  Interrupted files therefore cannot
enter the R5 merge.

F1 is the frozen SAM3.1 multiplex *propagation* backbone grid.  It is a shared,
candidate-independent [72, 72, 256] frame representation.  The frame backbone
is evaluated once and all human-query/candidate boxes on that frame reuse it.
No multiplex slot, mask, pointer, or memory tensor is inferred by this script.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
import torchvision.transforms as T


ROOT = Path(".")
DATA = Path("/path/to/dancetrack")
DATASET = ROOT / "outputs/n25r/repaired_dataset"
OUT = ROOT / "outputs/n25r/candidate_aligned_features"
SAM3_CKPT = ROOT / "checkpoints/sam3.1_mirror/sam3.1_multiplex.pt"
CLIP_CKPT = ROOT / "outputs/n15/checkpoints/clip_reid_hf/person_vit_clip_reid.pth"
HORIZON = 10
CLIP_TRANSFORM = T.Compose(
    [
        T.Resize((256, 128), interpolation=T.InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ]
)

sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def split_name(name: str) -> str:
    # N20/N25's train30 and held-out cal10 are both sequence partitions of the
    # DanceTrack *training* images.  "cal10" is an experimental role, not the
    # filesystem's official ``val`` directory.
    if name not in {"train30", "cal10"}:
        raise ValueError(name)
    return "train"


def canonical_row_key(row: dict[str, Any]) -> str:
    return "|".join(
        str(value)
        for value in (
            row["sequence"],
            row["public_identity_id"],
            row["gid"],
            row["decision_frame"],
            row["correction_frame"],
            row["candidate_source"],
            row["candidate_rank"],
        )
    )


def request_key(frame: int, box: Iterable[float]) -> tuple[int, tuple[float, ...]]:
    # Both extractors ultimately rasterize a box.  Six decimals retain the
    # source coordinates while allowing exact duplicates to share work.
    return int(frame), tuple(round(float(value), 6) for value in box)


def build_requests(rows: list[dict[str, Any]]) -> tuple[dict, np.ndarray, np.ndarray, np.ndarray]:
    requests: dict[tuple[int, tuple[float, ...]], list[tuple[str, int, int]]] = defaultdict(list)
    candidate_frames = np.full((len(rows), HORIZON), -1, dtype=np.int32)
    candidate_boxes = np.full((len(rows), HORIZON, 4), np.nan, dtype=np.float32)
    candidate_valid = np.zeros((len(rows), HORIZON), dtype=bool)
    for row_index, row in enumerate(rows):
        query = row["legal_human_positive"]
        query_frame = int(query["frame"])
        query_box = query["box"]
        requests[request_key(query_frame, query_box)].append(("query", row_index, 0))
        for step, item in enumerate((row.get("candidate_shadow_tracklet") or [])[:HORIZON]):
            if not item or not bool(item.get("valid", True)) or item.get("box") is None:
                continue
            frame = int(item["frame"])
            box = np.asarray(item["box"], dtype=np.float32)
            if box.shape != (4,) or not np.isfinite(box).all() or box[2] <= box[0] or box[3] <= box[1]:
                continue
            candidate_frames[row_index, step] = frame
            candidate_boxes[row_index, step] = box
            candidate_valid[row_index, step] = True
            requests[request_key(frame, box)].append(("candidate", row_index, step))
    return requests, candidate_frames, candidate_boxes, candidate_valid


def crop(image: Image.Image, box: Iterable[float]) -> Image.Image:
    x1, y1, x2, y2 = [int(round(float(value))) for value in box]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(image.width, x2), min(image.height, y2)
    if x2 <= x1 or y2 <= y1:
        return Image.new("RGB", (8, 8))
    return image.crop((x1, y1, x2, y2))


def clip_transform(image: Image.Image) -> torch.Tensor:
    # Exact transform object used by scripts/run_n15_extract_features.py.
    return CLIP_TRANSFORM(image)


def extract_clip(
    sequence: str,
    split: str,
    rows: list[dict[str, Any]],
    model: torch.nn.Module,
    device: torch.device,
    batch_size: int,
) -> dict[str, np.ndarray]:
    requests, candidate_frames, candidate_boxes, candidate_valid = build_requests(rows)
    query_features = np.full((len(rows), 1280), np.nan, dtype=np.float16)
    candidate_features = np.full((len(rows), HORIZON, 1280), np.nan, dtype=np.float16)
    by_frame: dict[int, list[tuple[tuple[int, tuple[float, ...]], list]]] = defaultdict(list)
    for key, destinations in requests.items():
        by_frame[key[0]].append((key, destinations))

    queued_tensors: list[torch.Tensor] = []
    queued_destinations: list[list[tuple[str, int, int]]] = []

    def flush() -> None:
        if not queued_tensors:
            return
        inputs = torch.stack(queued_tensors).to(device, non_blocking=True)
        with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.float16):
            _, x12, xproj = model(inputs)
            features = torch.cat([x12[:, 0], xproj[:, 0]], dim=1)
            features = F.normalize(features.float(), dim=1).cpu().numpy().astype(np.float16)
        for feature, destinations in zip(features, queued_destinations):
            for kind, row_index, step in destinations:
                if kind == "query":
                    query_features[row_index] = feature
                else:
                    candidate_features[row_index, step] = feature
        queued_tensors.clear()
        queued_destinations.clear()

    image_dir = DATA / split_name(split) / sequence / "img1"
    for frame in sorted(by_frame):
        image_path = image_dir / f"{frame + 1:08d}.jpg"
        with Image.open(image_path) as handle:
            image = handle.convert("RGB")
            for (_, box), destinations in by_frame[frame]:
                queued_tensors.append(clip_transform(crop(image, box)))
                queued_destinations.append(destinations)
                if len(queued_tensors) >= batch_size:
                    flush()
    flush()
    return {
        "query": query_features,
        "candidate": candidate_features,
        "candidate_frames": candidate_frames,
        "candidate_boxes": candidate_boxes,
        "candidate_valid": candidate_valid,
    }


def pool_grid(grid: torch.Tensor, boxes: list[tuple[float, ...]], width: int, height: int) -> tuple[np.ndarray, np.ndarray]:
    means = []
    maxima = []
    gh, gw = grid.shape[-2:]
    for box in boxes:
        x1, y1, x2, y2 = box
        gx1 = max(0, min(gw - 1, int(math.floor(x1 / width * gw))))
        gy1 = max(0, min(gh - 1, int(math.floor(y1 / height * gh))))
        gx2 = max(gx1 + 1, min(gw, int(math.ceil(x2 / width * gw))))
        gy2 = max(gy1 + 1, min(gh, int(math.ceil(y2 / height * gh))))
        roi = grid[:, gy1:gy2, gx1:gx2].float()
        mean = F.normalize(roi.mean(dim=(-2, -1)), dim=0)
        maximum = F.normalize(roi.amax(dim=(-2, -1)), dim=0)
        means.append(mean.cpu().numpy().astype(np.float16))
        maxima.append(maximum.cpu().numpy().astype(np.float16))
    return np.stack(means), np.stack(maxima)


def extract_sam3_f1(
    sequence: str,
    split: str,
    rows: list[dict[str, Any]],
    backend,
    device: torch.device,
    batch_size: int,
) -> dict[str, np.ndarray]:
    from sam3.model.io_utils import _load_img_as_tensor

    requests, candidate_frames, candidate_boxes, candidate_valid = build_requests(rows)
    query_mean = np.full((len(rows), 256), np.nan, dtype=np.float16)
    query_max = np.full((len(rows), 256), np.nan, dtype=np.float16)
    candidate_mean = np.full((len(rows), HORIZON, 256), np.nan, dtype=np.float16)
    candidate_max = np.full((len(rows), HORIZON, 256), np.nan, dtype=np.float16)

    image_dir = DATA / split_name(split) / sequence / "img1"
    first_path = image_dir / "00000001.jpg"
    with Image.open(first_path) as first_image:
        width, height = first_image.size
    backbone = backend._predictor.model.detector.backbone
    backbone.eval()
    frames = sorted({frame for frame, _ in requests})
    by_frame: dict[int, list[tuple[tuple[float, ...], list]]] = defaultdict(list)
    for (frame, box), destinations in requests.items():
        by_frame[frame].append((box, destinations))
    for start in range(0, len(frames), batch_size):
        batch_frames = frames[start : start + batch_size]
        loaded = [
            _load_img_as_tensor(str(image_dir / f"{frame + 1:08d}.jpg"), 1008)[0]
            for frame in batch_frames
        ]
        # Replicate the official video loader exactly: quantize storage to
        # fp16 first, then apply the configured (0.5, 0.5, 0.5) mean/std.
        images = torch.stack(loaded).to(torch.float16)
        normalizer = torch.tensor(0.5, dtype=torch.float16)
        images = ((images - normalizer) / normalizer).float().to(device)
        with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.float16):
            output = backbone.forward_image(
                images,
                need_sam3_out=False,
                need_interactive_out=False,
                need_propagation_out=True,
            )
            grids = output["sam2_backbone_out"]["vision_features"]
        if grids.shape[1:] != (256, 72, 72):
            raise RuntimeError(f"unexpected F1 grid shape {tuple(grids.shape)}")
        for local_index, frame in enumerate(batch_frames):
            items = by_frame[frame]
            means, maxima = pool_grid(grids[local_index], [item[0] for item in items], width, height)
            for mean, maximum, (_, destinations) in zip(means, maxima, items):
                for kind, row_index, step in destinations:
                    if kind == "query":
                        query_mean[row_index] = mean
                        query_max[row_index] = maximum
                    else:
                        candidate_mean[row_index, step] = mean
                        candidate_max[row_index, step] = maximum
        del images, output, grids
    return {
        "query_mean": query_mean,
        "query_max": query_max,
        "candidate_mean": candidate_mean,
        "candidate_max": candidate_max,
        "candidate_frames": candidate_frames,
        "candidate_boxes": candidate_boxes,
        "candidate_valid": candidate_valid,
    }


def validate_payload(backbone_name: str, payload: dict[str, np.ndarray], rows: list[dict[str, Any]]) -> dict[str, Any]:
    expected = payload["candidate_valid"]
    if backbone_name == "clipreid":
        query_ok = np.isfinite(payload["query"]).all(axis=1)
        candidate_ok = np.isfinite(payload["candidate"]).all(axis=-1)
    else:
        query_ok = np.isfinite(payload["query_mean"]).all(axis=1) & np.isfinite(payload["query_max"]).all(axis=1)
        candidate_ok = np.isfinite(payload["candidate_mean"]).all(axis=-1) & np.isfinite(payload["candidate_max"]).all(axis=-1)
    if not query_ok.all():
        raise RuntimeError(f"{backbone_name}: missing {int((~query_ok).sum())} query features")
    missing_valid = expected & ~candidate_ok
    invalid_filled = ~expected & candidate_ok
    if missing_valid.any() or invalid_filled.any():
        raise RuntimeError(
            f"{backbone_name}: missing_valid={int(missing_valid.sum())} invalid_filled={int(invalid_filled.sum())}"
        )
    return {
        "rows": len(rows),
        "query_coverage": float(query_ok.mean()),
        "valid_candidate_steps": int(expected.sum()),
        "candidate_feature_coverage": float(candidate_ok[expected].mean()) if expected.any() else 1.0,
        "invalid_steps_remain_nan": bool(not candidate_ok[~expected].any()),
        "positive_candidate_feature_coverage": float(
            candidate_ok[np.asarray([bool(row["positive"]) for row in rows])][
                expected[np.asarray([bool(row["positive"]) for row in rows])]
            ].mean()
        ) if any(bool(row["positive"]) for row in rows) else None,
        "negative_candidate_feature_coverage": float(
            candidate_ok[np.asarray([not bool(row["positive"]) for row in rows])][
                expected[np.asarray([not bool(row["positive"]) for row in rows])]
            ].mean()
        ) if any(not bool(row["positive"]) for row in rows) else None,
    }


def atomic_save(path: Path, arrays: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    os.replace(temporary, path)


def balanced_sequences(rows: list[dict[str, Any]], shard: int, num_shards: int) -> list[str]:
    by_sequence: dict[str, int] = defaultdict(int)
    for row in rows:
        by_sequence[str(row["sequence"])] += 1
    loads = [0] * num_shards
    assignments: list[list[str]] = [[] for _ in range(num_shards)]
    for sequence, weight in sorted(by_sequence.items(), key=lambda item: (-item[1], item[0])):
        target = min(range(num_shards), key=lambda index: (loads[index], index))
        assignments[target].append(sequence)
        loads[target] += weight
    return sorted(assignments[shard])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backbone", required=True, choices=("clipreid", "sam3_f1"))
    parser.add_argument("--split", required=True, choices=("train30", "cal10"))
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--shard", type=int, required=True)
    parser.add_argument("--num-shards", type=int, required=True)
    parser.add_argument("--batch-size", type=int, default=0)
    parser.add_argument("--sequences", default="", help="comma-separated smoke override")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if not 0 <= args.shard < args.num_shards:
        raise ValueError("shard must satisfy 0 <= shard < num_shards")
    torch.cuda.set_device(args.gpu)
    device = torch.device(f"cuda:{args.gpu}")
    episode_path = DATASET / f"episodes_{args.split}.jsonl"
    all_rows = load_jsonl(episode_path)
    chosen = (
        sorted(part for part in args.sequences.split(",") if part)
        if args.sequences
        else balanced_sequences(all_rows, args.shard, args.num_shards)
    )
    rows_by_sequence = {
        sequence: [(index, row) for index, row in enumerate(all_rows) if row["sequence"] == sequence]
        for sequence in chosen
    }
    output_dir = OUT / args.backbone / args.split
    output_dir.mkdir(parents=True, exist_ok=True)
    batch_size = args.batch_size or (64 if args.backbone == "clipreid" else 2)
    print(
        json.dumps(
            {
                "backbone": args.backbone,
                "split": args.split,
                "gpu": args.gpu,
                "shard": args.shard,
                "num_shards": args.num_shards,
                "sequences": chosen,
                "batch_size": batch_size,
            },
            sort_keys=True,
        ),
        flush=True,
    )

    model = None
    backend = None
    if args.backbone == "clipreid":
        from scripts.run_n15_extract_features import build_clipreid

        model = build_clipreid(str(CLIP_CKPT), device)
        model.eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)
    else:
        from sam3_intermot.adaptation.cfa_backend_runner import CFABackendRunner

        runner = CFABackendRunner(checkpoint_path=str(SAM3_CKPT), split=split_name(args.split))
        backend = runner._ensure_backend()
        backend._ensure_model()
        backend._predictor.model.eval()
        for parameter in backend._predictor.model.parameters():
            parameter.requires_grad_(False)

    started = time.time()
    for sequence in chosen:
        sequence_started = time.time()
        path = output_dir / f"{sequence}.npz"
        done = output_dir / f"{sequence}.done"
        metadata_path = output_dir / f"{sequence}.json"
        if not args.force and path.exists() and done.exists() and metadata_path.exists():
            print(f"SKIP_DONE {sequence}", flush=True)
            continue
        indexed = rows_by_sequence[sequence]
        row_indices = np.asarray([item[0] for item in indexed], dtype=np.int32)
        rows = [item[1] for item in indexed]
        if args.backbone == "clipreid":
            payload = extract_clip(sequence, args.split, rows, model, device, batch_size)
        else:
            payload = extract_sam3_f1(sequence, args.split, rows, backend, device, batch_size)
        payload.update(
            {
                "row_indices": row_indices,
                "row_keys": np.asarray([canonical_row_key(row) for row in rows]),
                "candidate_rank": np.asarray([int(row["candidate_rank"]) for row in rows], dtype=np.int16),
                "positive": np.asarray([bool(row["positive"]) for row in rows], dtype=bool),
                # The repaired historical cache did not retain the selected
                # obj_id.  -1 is explicit missing metadata, never a guessed ID.
                "selected_obj_id": np.full((len(rows), HORIZON), -1, dtype=np.int32),
                "selected_obj_id_valid": np.zeros((len(rows), HORIZON), dtype=bool),
            }
        )
        validation = validate_payload(args.backbone, payload, rows)
        atomic_save(path, payload)
        with np.load(path, allow_pickle=False) as check:
            if len(check["row_indices"]) != len(rows):
                raise RuntimeError("atomic output validation row mismatch")
        metadata = {
            "status": "COMPLETE",
            "backbone": args.backbone,
            "feature_definition": (
                "N15 CLIP-ReID ViT-B/16 symmetric RGB crop CLS768+projectedCLS512, L2 normalized"
                if args.backbone == "clipreid"
                else "frozen SAM3.1 multiplex propagation backbone 72x72x256 box ROI mean/max, L2 normalized"
            ),
            "sequence": sequence,
            "split": args.split,
            "source_episode_path": str(episode_path.relative_to(ROOT)),
            "source_episode_sha256": sha256(episode_path),
            "row_count": len(rows),
            "row_index_min": int(row_indices.min()),
            "row_index_max": int(row_indices.max()),
            "horizon": HORIZON,
            "selected_obj_id_policy": "explicit_invalid_minus_one; historical repaired cache did not retain obj_id; no guessed mapping",
            "validation": validation,
            "runtime_seconds": time.time() - sequence_started,
            "gpu": args.gpu,
        }
        temporary_metadata = metadata_path.with_suffix(".json.tmp")
        temporary_metadata.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        os.replace(temporary_metadata, metadata_path)
        done_tmp = done.with_suffix(".done.tmp")
        done_tmp.write_text(json.dumps({"npz_sha256": sha256(path), "metadata_sha256": sha256(metadata_path)}) + "\n", encoding="utf-8")
        os.replace(done_tmp, done)
        print(
            f"DONE {sequence} rows={len(rows)} valid_steps={validation['valid_candidate_steps']} "
            f"seconds={time.time()-sequence_started:.1f}",
            flush=True,
        )
    print(f"SHARD_COMPLETE seconds={time.time()-started:.1f}", flush=True)


if __name__ == "__main__":
    main()
