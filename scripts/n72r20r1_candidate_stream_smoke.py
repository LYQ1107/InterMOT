#!/usr/bin/env python3
"""Run one N72R20R1 real-SAM3 candidate-stream smoke sequence.

This worker is intentionally one-sequence/one-process.  It never opens GT,
never assigns identity, and never writes dense masks or crop images.  The
parent smoke runner starts it once per selected train sequence.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def feature_sha256(vector: np.ndarray) -> str:
    """Hash the canonical float32 feature before float16 tape storage."""

    return hashlib.sha256(np.asarray(vector, dtype=np.float32).tobytes(order="C")).hexdigest()


def image_files(sequence_dir: Path) -> list[Path]:
    image_dir = sequence_dir / "img1"
    paths = [
        path
        for path in image_dir.iterdir()
        if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png"}
    ]
    return sorted(paths, key=lambda path: int(path.stem))


def crop_batch(image_path: Path, boxes: Sequence[Sequence[float]]) -> torch.Tensor:
    from torchvision.io import ImageReadMode, read_image
    from torchvision.transforms import InterpolationMode
    from torchvision.transforms.functional import normalize, resize

    image = read_image(str(image_path), mode=ImageReadMode.RGB)
    _, image_height, image_width = image.shape
    tensors: list[torch.Tensor] = []
    for box in boxes:
        x1, y1, x2, y2 = [int(round(float(value))) for value in box]
        x1 = max(0, min(image_width, x1))
        y1 = max(0, min(image_height, y1))
        x2 = max(0, min(image_width, x2))
        y2 = max(0, min(image_height, y2))
        if x2 <= x1 or y2 <= y1:
            crop = torch.zeros((3, 8, 8), dtype=torch.uint8)
        else:
            crop = image[:, y1:y2, x1:x2]
        crop = crop.float().div(255.0)
        crop = resize(crop, [256, 128], interpolation=InterpolationMode.BILINEAR, antialias=True)
        normalize(crop, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225], inplace=True)
        tensors.append(crop)
    if not tensors:
        return torch.empty((0, 3, 256, 128), dtype=torch.float32)
    return torch.stack(tensors)


class MachineOSNet:
    """The frozen N72R16 OSNet encoder with path-portable box crops."""

    def __init__(self, checkpoint: Path, device: str) -> None:
        from sam3_intermot.identity_probe.encoders import OSNetEncoder

        self.encoder = OSNetEncoder(checkpoint, device=device)

    def encode(self, image_path: Path, boxes: Sequence[Sequence[float]]) -> np.ndarray:
        if not boxes:
            return np.empty((0, 512), dtype=np.float32)
        batch = crop_batch(image_path, boxes)
        with torch.inference_mode():
            features = self.encoder.encode(batch).cpu().numpy().astype(np.float32, copy=False)
        if features.shape != (len(boxes), 512) or not np.isfinite(features).all():
            raise RuntimeError(f"invalid OSNet feature shape/values: {features.shape}")
        return features


def load_manifest(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("stage") != "N72R20R1":
        raise ValueError(f"asset manifest is not N72R20R1: {path}")
    if payload.get("dataset", {}).get("test_present_or_used"):
        raise ValueError("asset manifest claims an out-of-scope DanceTrack test asset")
    return payload


def resolve_path(cli_value: str | None, env_name: str, manifest_value: str | None = None) -> Path:
    value = cli_value or os.environ.get(env_name) or manifest_value
    if not value:
        raise ValueError(f"provide --{env_name.lower()} or set {env_name}")
    return Path(value).expanduser().resolve()


def zstd_stream(path: Path) -> tuple[Any, subprocess.Popen[bytes]]:
    path.parent.mkdir(parents=True, exist_ok=True)
    file_handle = path.open("wb")
    try:
        process = subprocess.Popen(
            ["zstd", "-q", "-T0", "-c"],
            stdin=subprocess.PIPE,
            stdout=file_handle,
            stderr=subprocess.PIPE,
        )
    except Exception:
        file_handle.close()
        raise
    if process.stdin is None:
        file_handle.close()
        process.kill()
        raise RuntimeError("zstd stdin was not created")
    return file_handle, process


def run(args: argparse.Namespace) -> dict[str, Any]:
    manifest_path = Path(args.asset_manifest).expanduser().resolve()
    manifest = load_manifest(manifest_path)
    dataset_root = resolve_path(args.dataset_root, "DANCETRACK_ROOT", manifest.get("DANCETRACK_ROOT"))
    checkpoint = resolve_path(
        args.checkpoint,
        "SAM3_CHECKPOINT",
        manifest.get("sam3_checkpoint", {}).get("path"),
    )
    osnet = Path(
        args.osnet_checkpoint
        or manifest["frozen_identity_assets"]["osnet_x1_0_market1501"]["path"]
    ).expanduser().resolve()
    output_root = resolve_path(args.output_root, "N72R20R1_ASSET_ROOT")
    if not checkpoint.is_file():
        raise FileNotFoundError(f"missing authenticated SAM3 checkpoint: {checkpoint}")
    if not osnet.is_file():
        raise FileNotFoundError(f"missing frozen OSNet checkpoint: {osnet}")
    expected_checkpoint_sha = manifest.get("sam3_checkpoint", {}).get("sha256")
    if expected_checkpoint_sha and sha256(checkpoint) != expected_checkpoint_sha:
        raise ValueError("SAM3 checkpoint SHA256 does not match the frozen asset manifest")
    expected_osnet_sha = manifest["frozen_identity_assets"]["osnet_x1_0_market1501"].get("sha256")
    if expected_osnet_sha and sha256(osnet) != expected_osnet_sha:
        raise ValueError("OSNet checkpoint SHA256 does not match the frozen asset manifest")
    if args.split != "train":
        raise ValueError("N72R20R1 smoke is train-only; val is reserved for frozen evaluation")
    if not 100 <= int(args.max_frames) <= 200:
        raise ValueError("smoke frame count must be between 100 and 200")

    sequence_dir = dataset_root / args.split / args.sequence
    paths = image_files(sequence_dir)
    if not paths:
        raise FileNotFoundError(f"no image frames under {sequence_dir / 'img1'}")
    frame_count = min(len(paths), int(args.max_frames))
    sequence_out = output_root / "candidates" / args.sequence
    final_meta = sequence_out / "metadata.jsonl.zst"
    final_embeddings = sequence_out / "embeddings.f16"
    final_index = sequence_out / "index.json"
    final_done = sequence_out / "done.json"
    if any(path.exists() for path in (final_meta, final_embeddings, final_index, final_done)):
        raise FileExistsError(f"refusing to overwrite existing N72R20R1 sequence output: {sequence_out}")
    sequence_out.mkdir(parents=True, exist_ok=True)
    meta_tmp = sequence_out / f".{final_meta.name}.{os.getpid()}.tmp"
    embedding_tmp = sequence_out / f".{final_embeddings.name}.{os.getpid()}.tmp"
    started = time.time()
    metadata_handle = None
    compressor = None
    embedding_handle = None
    backend = None
    encoder = None
    seen_frames: set[int] = set()
    candidate_count = 0
    embedding_count = 0
    frame_records: list[dict[str, Any]] = []
    try:
        metadata_handle, compressor = zstd_stream(meta_tmp)
        embedding_handle = embedding_tmp.open("wb")
        from sam3_intermot.backend.sam3_backend import Sam3Backend

        backend = Sam3Backend(
            checkpoint_path=str(checkpoint),
            max_num_objects=16,
            multiplex_count=16,
            use_fa3=False,
            use_rope_real=True,
            compile=False,
            warm_up=False,
            output_prob_thresh=0.30,
            async_loading_frames=False,
            device=str(args.device),
        )
        backend.start_video(str(sequence_dir / "img1"))
        encoder = MachineOSNet(osnet, str(args.device))

        def write_frame(frame: int, observations: list[Any]) -> None:
            nonlocal candidate_count, embedding_count
            frame = int(frame)
            if frame in seen_frames or frame < 0 or frame >= frame_count:
                return
            boxes = [np.asarray(item.box_xyxy, dtype=float).tolist() for item in observations]
            features = encoder.encode(paths[frame], boxes)
            backend._output_cache[frame] = [item.copy() for item in observations]
            try:
                exported = backend.export_frame_candidates(
                    frame,
                    embeddings=features,
                    include_masks=False,
                    include_raw_provenance=True,
                )
            finally:
                backend._output_cache.pop(frame, None)
            rows: list[dict[str, Any]] = []
            frame_embedding_start = embedding_count
            for index, candidate in enumerate(exported):
                vector = np.asarray(candidate.get("embedding"), dtype=np.float32).reshape(-1)
                if vector.shape != (512,) or not np.isfinite(vector).all():
                    raise RuntimeError(f"invalid candidate embedding at {args.sequence}:{frame}:{index}")
                raw_id = candidate.get("raw_native_id")
                adapter_id = int(candidate["native_tid"])
                embedding_handle.write(vector.astype(np.float16).tobytes(order="C"))
                rows.append(
                    {
                        "sequence": args.sequence,
                        "frame": frame,
                        "candidate_uid": f"{args.sequence}:{frame}:{index}:{raw_id if raw_id is not None else adapter_id}",
                        "box_xyxy": np.asarray(candidate["box_xyxy"], dtype=float).tolist(),
                        "presence": candidate.get("presence_score"),
                        "iou_pred": candidate.get("predicted_iou"),
                        "confidence": float(candidate.get("confidence", 0.0)),
                        "candidate_source": str(candidate.get("source", "automatic_propagation")),
                        "source": str(candidate.get("source", "automatic_propagation")),
                        "source_public_id": None,
                        "raw_native_id": raw_id,
                        "adapter_external_id": adapter_id,
                        "native_tid": adapter_id,
                        "valid": True,
                        "reject_reason": None,
                        "embedding_offset": embedding_count,
                        "embedding_dim": 512,
                        "embedding_dtype": "float16",
                        "feature_source": "frozen_osnet_x1_0_market1501_machine_box_crop",
                        "embedding_source": "frozen_osnet_x1_0_market1501_machine_box_crop",
                        "feature_sha256": feature_sha256(vector),
                        "runtime_future_gt_used": False,
                    }
                )
                embedding_count += 1
            frame_row = {
                "sequence": args.sequence,
                "split": args.split,
                "frame": frame,
                "candidate_count": len(rows),
                "candidate_set_complete": True,
                "candidate_set_source": "official_sam3_full_vg_post_nms_propagation",
                "candidates": rows,
                "runtime_future_gt_used": False,
                "runtime_gt_read": False,
            }
            payload = (json.dumps(frame_row, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n").encode()
            assert compressor is not None and compressor.stdin is not None
            compressor.stdin.write(payload)
            candidate_count += len(rows)
            seen_frames.add(frame)
            frame_records.append(
                {
                    "frame": frame,
                    "candidate_count": len(rows),
                    "embedding_start": frame_embedding_start,
                    "embedding_count": len(rows),
                }
            )

        initial = backend.detect_concept(0, "person")
        if initial:
            write_frame(0, initial)
        backend.propagate(
            0,
            frame_count - 1,
            start_frame_index=0,
            keep_masks=False,
            cache_outputs=False,
            output_callback=write_frame,
        )
        expected_frames = set(range(frame_count))
        if seen_frames != expected_frames:
            raise RuntimeError(
                f"candidate stream frame mismatch: missing={sorted(expected_frames - seen_frames)[:8]} "
                f"extra={sorted(seen_frames - expected_frames)[:8]}"
            )
        assert compressor is not None and compressor.stdin is not None and metadata_handle is not None
        compressor.stdin.close()
        compressor.stdin = None
        stderr = compressor.stderr.read().decode(errors="replace") if compressor.stderr is not None else ""
        return_code = compressor.wait()
        metadata_handle.flush()
        os.fsync(metadata_handle.fileno())
        metadata_handle.close()
        metadata_handle = None
        if return_code != 0:
            raise RuntimeError(f"zstd failed with code {return_code}: {stderr[-2000:]}")
        embedding_handle.flush()
        os.fsync(embedding_handle.fileno())
        embedding_handle.close()
        embedding_handle = None
        os.replace(meta_tmp, final_meta)
        os.replace(embedding_tmp, final_embeddings)
        index = {
            "stage": "N72R20R1",
            "sequence": args.sequence,
            "split": args.split,
            "frame_count": frame_count,
            "candidate_count": candidate_count,
            "embedding_count": embedding_count,
            "embedding_dim": 512,
            "embedding_dtype": "float16",
            "metadata": str(final_meta),
            "metadata_sha256": sha256(final_meta),
            "embeddings": str(final_embeddings),
            "embeddings_sha256": sha256(final_embeddings),
            "frames": frame_records,
            "runtime_future_gt_used": False,
            "runtime_gt_read": False,
        }
        final_index.write_text(json.dumps(index, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        profile = {
            "stage": "N72R20R1",
            "sequence": args.sequence,
            "split": args.split,
            "frames": frame_count,
            "candidates": candidate_count,
            "metadata_bytes": final_meta.stat().st_size,
            "embedding_bytes": final_embeddings.stat().st_size,
            "total_bytes": final_meta.stat().st_size + final_embeddings.stat().st_size,
            "bytes_per_frame": (final_meta.stat().st_size + final_embeddings.stat().st_size) / max(frame_count, 1),
            "bytes_per_candidate": (final_meta.stat().st_size + final_embeddings.stat().st_size) / max(candidate_count, 1),
            "runtime_future_gt_used": False,
            "runtime_gt_read": False,
        }
        final_done.write_text(
            json.dumps(
                {
                    "status": "PASS_N72R20R1_CANDIDATE_STREAM_SMOKE_SEQUENCE",
                    "sequence": args.sequence,
                    "index": str(final_index),
                    "profile": profile,
                    "runtime_memory_policy": backend.runtime_memory_policy(),
                    "process_isolation": "one_python_process_one_sequence_one_sam3_session",
                    "elapsed_sec": time.time() - started,
                },
                indent=2,
                ensure_ascii=False,
            )
            + "\n",
            encoding="utf-8",
        )
        return profile
    finally:
        if compressor is not None:
            if compressor.stdin is not None:
                compressor.stdin.close()
            compressor.wait()
        if metadata_handle is not None:
            metadata_handle.close()
        if embedding_handle is not None:
            embedding_handle.close()
        for path in (meta_tmp, embedding_tmp):
            if path.exists():
                path.unlink()
        if backend is not None:
            try:
                backend.close()
            except Exception:
                pass
        del encoder
        del backend
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sequence", required=True)
    parser.add_argument("--split", default="train")
    parser.add_argument("--max-frames", type=int, default=160)
    parser.add_argument("--dataset-root", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--osnet-checkpoint", type=Path)
    parser.add_argument("--asset-manifest", type=Path, default=ROOT / "outputs/N72R20R1/asset_manifest.json")
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    try:
        profile = run(args)
    except Exception as exc:
        print(f"N72R20R1 smoke failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(profile, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
