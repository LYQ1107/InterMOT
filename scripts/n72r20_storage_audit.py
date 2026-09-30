#!/usr/bin/env python3
"""Read-only N72R20 storage and asset audit.

The audit deliberately does not remove partial downloads, caches, or old
experiment outputs.  It records enough provenance to decide whether a later
run is safe under the N72R20 storage policy.
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


STAGE = "N72R20"
WARNING_BYTES = 120_000_000_000
HARD_STOP_BYTES = 100_000_000_000


def sha256_file(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def du_bytes(path: Path) -> int | None:
    if not path.exists():
        return None
    if path.is_file():
        return path.stat().st_size
    completed = subprocess.run(
        ["du", "-sb", "--", str(path)],
        check=True,
        capture_output=True,
        text=True,
    )
    return int(completed.stdout.split()[0])


def path_record(path: Path, *, digest: bool = False) -> dict[str, Any]:
    record: dict[str, Any] = {
        "path": str(path),
        "exists": path.exists(),
        "kind": "missing",
    }
    if not path.exists():
        return record
    record["kind"] = "file" if path.is_file() else "directory"
    record["bytes"] = du_bytes(path)
    if path.is_file():
        stat = path.stat()
        record["mtime_utc"] = datetime.fromtimestamp(
            stat.st_mtime, tz=timezone.utc
        ).isoformat()
        if digest:
            record["sha256"] = sha256_file(path)
    return record


def sequence_record(path: Path) -> dict[str, Any]:
    img_dir = path / "img1"
    gt_path = path / "gt" / "gt.txt"
    image_count = 0
    if img_dir.is_dir():
        image_count = sum(
            1
            for entry in img_dir.iterdir()
            if entry.is_file() and entry.suffix.lower() in {".jpg", ".jpeg", ".png"}
        )
    return {
        "sequence": path.name,
        "path": str(path),
        "readable": os.access(path, os.R_OK),
        "img1_exists": img_dir.is_dir(),
        "image_count": image_count,
        "gt_exists": gt_path.is_file(),
        "gt_path": str(gt_path),
    }


def dataset_record(root: Path) -> dict[str, Any]:
    splits: dict[str, Any] = {}
    for split, expected in (("train", 40), ("val", 25)):
        split_root = root / split
        sequences = (
            sorted(p for p in split_root.iterdir() if p.is_dir())
            if split_root.is_dir()
            else []
        )
        rows = [sequence_record(path) for path in sequences]
        splits[split] = {
            "expected_sequence_count": expected,
            "sequence_count": len(rows),
            "count_matches_expected": len(rows) == expected,
            "all_img1_nonempty": all(row["image_count"] > 0 for row in rows),
            "all_gt_present": all(row["gt_exists"] for row in rows),
            "sequences": rows,
        }
    return {"root": str(root), "exists": root.is_dir(), "splits": splits}


def candidate_record(root: Path, sequence: str) -> dict[str, Any]:
    seq_root = root / "candidates" / sequence
    done_path = seq_root / "done.json"
    index_path = seq_root / "index.json"
    record: dict[str, Any] = {
        "sequence": sequence,
        "root": str(seq_root),
        "complete_marker": path_record(done_path),
        "index": path_record(index_path),
        "metadata": path_record(seq_root / "metadata.jsonl.zst"),
        "embeddings": path_record(seq_root / "embeddings.f16", digest=True),
    }
    if index_path.is_file():
        index = json.loads(index_path.read_text())
        record["index_summary"] = {
            key: index.get(key)
            for key in (
                "candidate_count",
                "embedding_count",
                "embedding_dim",
                "embedding_dtype",
                "frame_count",
                "embeddings_sha256",
            )
        }
    if done_path.is_file():
        done = json.loads(done_path.read_text())
        record["done_summary"] = {
            "status": done.get("status"),
            "process_isolation": done.get("process_isolation"),
            "runtime_future_gt_used": done.get("profile", {}).get(
                "runtime_future_gt_used"
            ),
            "runtime_gt_read": done.get("profile", {}).get("runtime_gt_read"),
        }
    return record


def clone_record(root: Path, name: str, expected_sha: str) -> dict[str, Any]:
    path = root / name
    actual = None
    if (path / ".git").exists():
        actual = subprocess.run(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    return {
        "name": name,
        "path": str(path),
        "expected_sha": expected_sha,
        "actual_sha": actual,
        "sha_matches": actual == expected_sha,
        "bytes": du_bytes(path),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
    )
    args = parser.parse_args()
    repo_root = args.repo_root.resolve()
    output = args.output or repo_root / "outputs" / STAGE / "storage_audit_before.json"

    usage = shutil.disk_usage(repo_root)
    if usage.free < HARD_STOP_BYTES:
        storage_status = "HARD_STOP"
    elif usage.free < WARNING_BYTES:
        storage_status = "WARNING"
    else:
        storage_status = "OK"

    n72r20_assets = Path("/data3/liuyeqiang/InterMOT_N72R20_assets")
    dataset_root = Path("/data3/liuyeqiang/InterMOT_N72R16_assets/dataset")
    reference_root = Path("/data3/liuyeqiang/research_references/N72R20")
    clone_specs = {
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

    checkpoint_root = n72r20_assets / "checkpoints"
    report: dict[str, Any] = {
        "stage": STAGE,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "read_only": True,
        "policy": {
            "warning_free_bytes": WARNING_BYTES,
            "hard_stop_free_bytes": HARD_STOP_BYTES,
            "no_deletion_performed": True,
            "no_download_performed": True,
        },
        "filesystem": {
            "path": str(repo_root),
            "total_bytes": usage.total,
            "used_bytes": usage.used,
            "free_bytes": usage.free,
            "free_gb_decimal": usage.free / 1_000_000_000,
            "free_gib": usage.free / (1024**3),
            "storage_status": storage_status,
        },
        "dataset": dataset_record(dataset_root),
        "checkpoints": [
            path_record(
                checkpoint_root / "sam3.1_multiplex.pt",
                digest=True,
            ),
            path_record(
                Path("/data3/liuyeqiang/InterMOT_N72R16_assets/checkpoints/osnet_x1_0_market1501.pth"),
                digest=True,
            ),
            path_record(
                repo_root / "outputs/N72R18/checkpoints/identity_memory_gru.pt",
                digest=True,
            ),
        ],
        "candidate_streams": [
            candidate_record(n72r20_assets, "dancetrack0001"),
            candidate_record(n72r20_assets, "dancetrack0002"),
        ],
        "historical_partials": [
            path_record(
                n72r20_assets
                / "aborted_streaming_val_20260930/dancetrack0004/.embeddings.f16.2379525.tmp"
            ),
            path_record(
                n72r20_assets
                / "aborted_streaming_val_20260930/dancetrack0004/.metadata.jsonl.zst.2379525.tmp"
            ),
        ],
        "reference_clones": [
            clone_record(reference_root, name, sha)
            for name, sha in clone_specs.items()
        ],
        "known_large_roots": [
            path_record(repo_root),
            path_record(Path("/data3/liuyeqiang/InterMOT_N72R16_assets")),
            path_record(Path("/data3/liuyeqiang/InterMOT_N72R17_assets")),
            path_record(Path("/data3/liuyeqiang/.cache")),
            path_record(Path("/data3/liuyeqiang/.codex")),
        ],
        "next_safe_action": (
            "Continue with code-level audit and unit tests only; do not download, "
            "copy the dataset, or launch a long SAM3/val run while free space is "
            "near the warning threshold."
        ),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "output": str(output),
        "storage_status": storage_status,
        "free_gb_decimal": report["filesystem"]["free_gb_decimal"],
        "dataset_train": report["dataset"]["splits"]["train"]["sequence_count"],
        "dataset_val": report["dataset"]["splits"]["val"]["sequence_count"],
        "reference_clone_bytes": sum(
            row["bytes"] or 0 for row in report["reference_clones"]
        ),
    }, indent=2))


if __name__ == "__main__":
    main()
