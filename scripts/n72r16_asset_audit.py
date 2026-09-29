#!/usr/bin/env python3
"""Audit N72R16 assets, lineage, storage, and GPU state."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

from sam3_intermot.identity_probe.dataset import discover_sequences


HF_DATASET_COMMIT = "a0ba42ac690c41e9850a20e76a0b9450a6fb6a47"
OSNET_SOURCE = {
    "repository": "KaiyangZhou/deep-person-reid",
    "repository_commit": "f8cd150fdf77e8d9e1ed143b7f308c2c609ded50",
    "model_zoo_url": "https://github.com/KaiyangZhou/deep-person-reid/blob/master/docs/MODEL_ZOO.md",
    "checkpoint_url": "https://drive.google.com/uc?export=download&id=1vduhq5DpN2q1g4fYEZfPI17MJeh9qyrA&confirm=t",
    "checkpoint_name": "osnet_x1_0_market1501.pth",
    "public_pretrained": True,
    "historical_intermot_equivalence_claim": False,
}
HF_FILES = {
    "train1.zip": {
        "bytes": 3606300312,
        "sha256": "70a66f10d8d71df94d03059fc4f966cb719e61e8e7136d2886fd6345ed1ff6dd",
    },
    "train2.zip": {
        "bytes": 3299320948,
        "sha256": "f676e8c6ac2a2d7566f1702d588311eafd4c90cc5416bb70f36a192d1e365af4",
    },
    "val.zip": {
        "bytes": 4209785614,
        "sha256": "90ba30973761ce0b81a9654c11086d87537392475ac8bc666d842e645641277c",
    },
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def command_output(command: list[str]) -> str | None:
    try:
        return subprocess.run(command, check=False, text=True, capture_output=True).stdout.strip()
    except FileNotFoundError:
        return None


def gpu_inventory() -> dict[str, object]:
    query = command_output(
        [
            "nvidia-smi",
            "--query-gpu=index,name,uuid,memory.total,memory.used,memory.free,utilization.gpu",
            "--format=csv,noheader,nounits",
        ]
    )
    gpus = []
    if query:
        for line in query.splitlines():
            fields = [value.strip() for value in line.split(",")]
            if len(fields) == 7:
                gpus.append(
                    {
                        "index": int(fields[0]),
                        "name": fields[1],
                        "uuid": fields[2],
                        "memory_total_mib": int(fields[3]),
                        "memory_used_mib": int(fields[4]),
                        "memory_free_mib": int(fields[5]),
                        "utilization_gpu_percent": int(fields[6]),
                    }
                )
    return {
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "query": query,
        "gpus": gpus,
        "pmon": command_output(["nvidia-smi", "pmon", "-c", "1"]),
    }


def file_record(path: Path, expected: dict[str, object] | None = None) -> dict[str, object]:
    item: dict[str, object] = {"path": str(path), "exists": path.is_file()}
    if path.is_file():
        stat = path.stat()
        item.update({"bytes": stat.st_size, "sha256": sha256(path), "local_mtime_utc": datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat()})
    if expected:
        item["expected"] = expected
        if path.is_file():
            item["sha256_matches_expected"] = item["sha256"] == expected["sha256"] and item["bytes"] == expected["bytes"]
    return item


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--assets-root", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/N72R16"))
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--verify-images", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    gpu_before_path = args.output_dir / "gpu_inventory_before.json"
    if gpu_before_path.is_file():
        gpu_before = json.loads(gpu_before_path.read_text(encoding="utf-8"))
    else:
        gpu_before = gpu_inventory()
    sequence_audits: dict[str, object] = {}
    dataset_root = args.dataset_root.expanduser().resolve()
    for split in ("train", "val"):
        split_dir = dataset_root / split
        if not split_dir.is_dir():
            sequence_audits[split] = {"present": False, "expected_sequence_count": 40 if split == "train" else 25}
            continue
        sequences = discover_sequences(dataset_root, split)
        sequence_audits[split] = {
            "present": True,
            "sequence_count": len(sequences),
            "expected_sequence_count": 40 if split == "train" else 25,
            "sequences": [sequence.validate(verify_images=args.verify_images) for sequence in sequences],
        }

    archives = args.assets_root / "archives"
    manifest = {
        "asset_lineage": "N72R16_NEW_ASSET_LINEAGE",
        "stage": "N72R16",
        "goal": "Human Identity Representation Probe",
        "central_question": "Can one human-confirmed identity observation reliably recognize the same person against hard competing identities over future frames?",
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_order": ["DanceTrack official Baidu direct (attempted)", "DanceTrack official Hugging Face fallback (used)"],
        "baidu": {
            "url": "https://pan.baidu.com/s/19O3IvYNzzrcLqlODHKYUwA",
            "extraction_code": "awew",
            "status": "not_used_headless_verification_errno_2_or_timeout",
        },
        "huggingface": {
            "repo_id": "noahcao/dancetrack",
            "repo_type": "dataset",
            "commit": HF_DATASET_COMMIT,
            "files": {name: file_record(archives / name, expected) for name, expected in HF_FILES.items()},
        },
        "dataset_root": str(dataset_root),
        "sequence_audits": sequence_audits,
        "checkpoint": {
            **file_record(args.checkpoint.expanduser().resolve()),
            "lineage": OSNET_SOURCE,
        } if args.checkpoint else None,
        "storage": {
            "mount": str(args.assets_root.resolve()),
            "disk_total_bytes": shutil.disk_usage(args.assets_root).total,
            "disk_used_bytes": shutil.disk_usage(args.assets_root).used,
            "disk_free_bytes": shutil.disk_usage(args.assets_root).free,
            "hard_budget_bytes": 100 * 1024**3,
            "target_budget_bytes": 60 * 1024**3,
        },
        "gpu": gpu_before,
        "prohibited_assets": ["DanceTrack test split", "SAM3 checkpoint", "old PCTIS/N72 caches", "other MOT datasets"],
    }
    (args.output_dir / "asset_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if not gpu_before_path.is_file():
        gpu_before_path.write_text(json.dumps(manifest["gpu"], indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"manifest": str(args.output_dir / "asset_manifest.json"), "sequence_audits": sequence_audits}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
