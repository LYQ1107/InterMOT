#!/usr/bin/env python3
"""Validate all N27 CLIP-ReID shards and freeze the cache manifest."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from pathlib import Path

import numpy as np


ROOT = Path(".")
OUT = ROOT / "outputs/n27"
DATA = OUT / "data"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def main() -> None:
    shards = []
    global_ids: set[str] = set()
    total = 0
    total_bytes = 0
    for shard in range(4):
        request_path = DATA / f"crop_requests_shard{shard}.jsonl"
        npz_path = DATA / f"clipreid_shard{shard}.npz"
        metadata_path = DATA / f"clipreid_shard{shard}.json"
        done_path = DATA / f"clipreid_shard{shard}.done"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        done = json.loads(done_path.read_text(encoding="utf-8"))
        if done["npz_sha256"] != sha256(npz_path) or done["metadata_sha256"] != sha256(metadata_path):
            raise RuntimeError(f"shard {shard} completion hash mismatch")
        if metadata["request_sha256"] != sha256(request_path):
            raise RuntimeError(f"shard {shard} request hash mismatch")
        expected_ids = []
        with request_path.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    expected_ids.append(json.loads(line)["crop_id"])
        with np.load(npz_path) as payload:
            actual_ids = [value.decode("ascii") for value in payload["crop_id"]]
            embeddings = payload["embedding"].astype(np.float32)
        if actual_ids != expected_ids:
            raise RuntimeError(f"shard {shard} crop order mismatch")
        if len(set(actual_ids)) != len(actual_ids):
            raise RuntimeError(f"shard {shard} duplicate crop IDs")
        overlap = global_ids.intersection(actual_ids)
        if overlap:
            raise RuntimeError(f"cross-shard duplicate crop IDs: {next(iter(overlap))}")
        global_ids.update(actual_ids)
        if embeddings.shape != (len(actual_ids), 1280) or not np.isfinite(embeddings).all():
            raise RuntimeError(f"shard {shard} embedding shape/finite regression")
        norms = np.linalg.norm(embeddings, axis=1)
        max_norm_error = float(np.max(np.abs(norms - 1.0)))
        if max_norm_error > 0.002:
            raise RuntimeError(f"shard {shard} norm regression: {max_norm_error}")
        part_dir = DATA / "feature_parts" / f"shard{shard}"
        part_done_count = len(list(part_dir.glob("part*.done")))
        if part_done_count != metadata["parts"]:
            raise RuntimeError(f"shard {shard} part count mismatch")
        total += len(actual_ids)
        total_bytes += npz_path.stat().st_size
        shards.append({
            **metadata,
            "npz_sha256": done["npz_sha256"],
            "metadata_sha256": done["metadata_sha256"],
            "output_bytes": npz_path.stat().st_size,
            "validated_max_norm_error": max_norm_error,
            "crop_ids_unique_within_and_across_shards": True,
            "request_order_exact": True,
            "part_done_count": part_done_count,
        })
    episode_manifest_path = OUT / "large_episode_manifest.json"
    episode_manifest = json.loads(episode_manifest_path.read_text(encoding="utf-8"))
    if total != episode_manifest["unique_crop_requests"]:
        raise RuntimeError(f"global request coverage mismatch {total} != {episode_manifest['unique_crop_requests']}")
    disk = shutil.disk_usage("/data1")
    manifest = {
        "phase": "N27",
        "status": "COMPLETE_VALIDATED",
        "backbone": "frozen CLIP-ReID ViT-B/16",
        "checkpoint_path": str(ROOT / "outputs/n15/checkpoints/clip_reid_hf/person_vit_clip_reid.pth"),
        "checkpoint_sha256": shards[0]["checkpoint_sha256"],
        "embedding_dimension": 1280,
        "dtype": "float16",
        "preprocessing": "symmetric person crop; PIL BICUBIC 256x128; RGB; mean/std 0.5",
        "precision": "model_and_input_fp16",
        "shards": shards,
        "total_unique_crop_ids": total,
        "expected_unique_crop_ids": episode_manifest["unique_crop_requests"],
        "coverage_exact": True,
        "all_finite": True,
        "all_normalized_within_0.002": True,
        "total_final_npz_bytes": total_bytes,
        "successful_extraction_gpu_hours": float(sum(shard["runtime_seconds"] for shard in shards) / 3600),
        "successful_extraction_wall_seconds": float(max(shard["runtime_seconds"] for shard in shards)),
        "recovery": "8192-crop atomic parts with SHA-256 .done markers",
        "initial_uncheckpointed_attempt": "INTERRUPTED_BY_DESKTOP_USER_MESSAGE after the last logged 50,000 crops per shard; no final artifacts accepted",
        "data1_free_bytes": disk.free,
        "minimum_reserved_bytes": 40 * 1024**3,
        "reserve_satisfied": disk.free >= 40 * 1024**3,
        "test_labels_used": False,
        "val25_read": False,
    }
    atomic_json(OUT / "feature_cache_manifest.json", manifest)
    episode_manifest["status"] = "FEATURES_COMPLETE_EPISODE_ROLLOUT_PENDING"
    episode_manifest["feature_cache_manifest"] = str(OUT / "feature_cache_manifest.json")
    episode_manifest["feature_cache_manifest_sha256"] = sha256(OUT / "feature_cache_manifest.json")
    episode_manifest["data1_free_bytes_after_features"] = disk.free
    episode_manifest["reserve_satisfied_after_features"] = disk.free >= 40 * 1024**3
    atomic_json(episode_manifest_path, episode_manifest)
    print(json.dumps({
        "shards": len(shards),
        "unique_crop_ids": total,
        "final_npz_gib": total_bytes / 1024**3,
        "successful_gpu_hours": manifest["successful_extraction_gpu_hours"],
        "wall_seconds": manifest["successful_extraction_wall_seconds"],
        "free_gib": disk.free / 1024**3,
        "val25_read": False,
    }, indent=2, sort_keys=True))
    print("N27_FEATURE_CACHE_VALIDATED")


if __name__ == "__main__":
    main()
