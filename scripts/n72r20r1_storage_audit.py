#!/usr/bin/env python3
"""Read-only storage and asset audit for N72R20R1.

This records the current machine state without deleting, copying, downloading,
or opening DanceTrack test data.  The output is intentionally a manifest, not
an authorization to start a large run.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DATASET = Path("/data3/liuyeqiang/InterMOT_N72R16_assets/dataset")
ASSET_ROOT = Path("/data3/liuyeqiang/InterMOT_N72R20_assets")
R1_ASSET_ROOT = Path("/data3/liuyeqiang/InterMOT_N72R20R1_assets")
REFERENCE_ROOT = Path("/data3/liuyeqiang/research_references/N72R20")
WARNING_BYTES = 120_000_000_000
HARD_STOP_BYTES = 100_000_000_000
NEW_BUDGET_BYTES = 15_000_000_000


def sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def bytes_used(path: Path) -> int | None:
    if not path.exists():
        return None
    if path.is_file():
        return path.stat().st_size
    result = subprocess.run(
        ["du", "-sb", "--", str(path)], capture_output=True, text=True, check=True
    )
    return int(result.stdout.split()[0])


def path_record(path: Path, *, digest: bool = False) -> dict[str, Any]:
    result: dict[str, Any] = {
        "path": str(path),
        "exists": path.exists(),
        "kind": "missing",
    }
    if not path.exists():
        return result
    result["kind"] = "file" if path.is_file() else "directory"
    result["bytes"] = bytes_used(path)
    if path.is_file() and digest:
        result["sha256"] = sha256(path)
    return result


def sequence_record(path: Path) -> dict[str, Any]:
    image_dir = path / "img1"
    images = sorted(
        item for item in image_dir.iterdir()
        if item.is_file() and item.suffix.lower() in {".jpg", ".jpeg", ".png"}
    ) if image_dir.is_dir() else []
    gt = path / "gt" / "gt.txt"
    return {
        "sequence": path.name,
        "path": str(path),
        "readable": os.access(path, os.R_OK),
        "image_count": len(images),
        "first_frame": int(images[0].stem) if images and images[0].stem.isdigit() else None,
        "last_frame": int(images[-1].stem) if images and images[-1].stem.isdigit() else None,
        "img1_nonempty": bool(images),
        "gt_path": str(gt),
        "gt_readable": gt.is_file() and os.access(gt, os.R_OK),
        "gt_sha256": sha256(gt),
    }


def split_record(split: str, expected: int) -> dict[str, Any]:
    root = DATASET / split
    sequences = sorted(item for item in root.iterdir() if item.is_dir()) if root.is_dir() else []
    records = [sequence_record(item) for item in sequences]
    return {
        "root": str(root),
        "expected_sequence_count": expected,
        "sequence_count": len(records),
        "count_matches_expected": len(records) == expected,
        "all_img1_nonempty": all(item["img1_nonempty"] for item in records),
        "all_gt_readable": all(item["gt_readable"] for item in records),
        "image_count": sum(item["image_count"] for item in records),
        "sequences": records,
    }


def git_head(path: Path) -> str | None:
    if not (path / ".git").exists():
        return None
    result = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"], capture_output=True, text=True, check=True
    )
    return result.stdout.strip()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/N72R20R1/storage_audit_before.json")
    args = parser.parse_args()
    usage = shutil.disk_usage(ROOT)
    if usage.free < HARD_STOP_BYTES:
        status = "HARD_STOP"
    elif usage.free < WARNING_BYTES:
        status = "WARNING"
    else:
        status = "OK"
    df = subprocess.run(["df", "-h", str(ROOT)], capture_output=True, text=True, check=True).stdout
    clone_shas = {
        "MOTIP": "ffc0e905ac196a603027eca8d18fb0dff48c8bcc",
        "MeMOTR": "eb7a177b9cbcb89742ec69b2545ab3af2ea31a80",
        "TrackTrack": "ee7f1c5fcbdcac48ed8bfab38d52c0006bf304da",
        "GeneralTrack": "dbb727bfb63eddd28e97b1f64462cbf7df1413c6",
        "DiffMOT": "eada72e74e54c153b30674d277b97882edc568b8",
        "MASA": "c5472b9c7615f35abdf1188cb1a0c5408fe50d66",
        "BoostTrack": "fb5bfc3a8f067476565e753b3a73df4d757c9d03",
        "BoT-SORT": "251985436d6712aaf682aaaf5f71edb4987224bd",
        "OC_SORT": "8462e7e729a93ccd3bd995c0a79a890336cb3a0b",
        "CKP": "183218eb1624027a1991586c931b242dd08d3a35",
    }
    checkpoints = [
        path_record(ASSET_ROOT / "checkpoints/sam3.1_multiplex.pt", digest=True),
        path_record(Path("/data3/liuyeqiang/InterMOT_N72R16_assets/checkpoints/osnet_x1_0_market1501.pth"), digest=True),
        path_record(ROOT / "outputs/N72R18/checkpoints/identity_memory_gru.pt", digest=True),
    ]
    cache_root = Path("/data3/liuyeqiang/.cache")
    codex_root = Path("/data3/liuyeqiang/.codex")
    partials = sorted(
        str(item) for item in ASSET_ROOT.rglob("*.tmp") if item.is_file()
    ) if ASSET_ROOT.is_dir() else []
    report: dict[str, Any] = {
        "stage": "N72R20R1",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "read_only": True,
        "policy": {
            "warning_free_bytes": WARNING_BYTES,
            "hard_stop_free_bytes": HARD_STOP_BYTES,
            "new_asset_budget_bytes": NEW_BUDGET_BYTES,
            "test_download_allowed": False,
            "download_performed": False,
            "deletion_performed": False,
        },
        "filesystem": {
            "path": str(ROOT),
            "total_bytes": usage.total,
            "used_bytes": usage.used,
            "free_bytes": usage.free,
            "free_gb_decimal": usage.free / 1_000_000_000,
            "free_gib": usage.free / (1024**3),
            "storage_status": status,
            "df_h_output": df,
            "quota": "none reported by quota -s",
        },
        "dataset": {
            "root": str(DATASET),
            "train": split_record("train", 40),
            "val": split_record("val", 25),
            "test_directory_present": (DATASET / "test").exists(),
        },
        "checkpoints": checkpoints,
        "known_roots": [
            path_record(ROOT),
            path_record(Path("/data3/liuyeqiang/InterMOT_N72R16_assets")),
            path_record(Path("/data3/liuyeqiang/InterMOT_N72R17_assets")),
            path_record(ASSET_ROOT),
            path_record(R1_ASSET_ROOT),
            path_record(REFERENCE_ROOT),
            path_record(cache_root),
            path_record(codex_root),
        ],
        "reference_clones": [
            {
                "name": name,
                "path": str(REFERENCE_ROOT / name),
                "expected_sha": expected,
                "actual_sha": git_head(REFERENCE_ROOT / name),
                "sha_matches": git_head(REFERENCE_ROOT / name) == expected,
                "bytes": bytes_used(REFERENCE_ROOT / name),
            }
            for name, expected in clone_shas.items()
        ],
        "temporary_files_under_old_asset_root": partials,
        "r1_existing_output_bytes": bytes_used(ROOT / "outputs/N72R20R1"),
        "next_action": "Run only the bounded dancetrack0001/dancetrack0002 train candidate smoke if storage status is not HARD_STOP.",
        "runtime_future_gt_used": False,
    }
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "output": str(output),
        "storage_status": status,
        "free_gib": report["filesystem"]["free_gib"],
        "train_sequences": report["dataset"]["train"]["sequence_count"],
        "val_sequences": report["dataset"]["val"]["sequence_count"],
        "test_directory_present": report["dataset"]["test_directory_present"],
    }, sort_keys=True))
    return 0 if status != "HARD_STOP" else 2


if __name__ == "__main__":
    raise SystemExit(main())
