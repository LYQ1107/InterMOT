#!/usr/bin/env python3
"""Validate the complete, GT-blind R2 train candidate tapes."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ASSET_ROOT = Path("/data3/liuyeqiang/InterMOT_N72R20R2_assets")
DEFAULT_DATASET_ROOT = Path("/data3/liuyeqiang/InterMOT_N72R16_assets/dataset")
SEQUENCES = (
    "dancetrack0001",
    "dancetrack0002",
    "dancetrack0023",
    "dancetrack0024",
    "dancetrack0039",
    "dancetrack0057",
    "dancetrack0062",
    "dancetrack0072",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_zstd_jsonl(path: Path) -> list[dict[str, Any]]:
    result = subprocess.run(["zstd", "-q", "-d", "-c", str(path)], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return [json.loads(line) for line in result.stdout.splitlines() if line.strip()]


def expected_frames(sequence_dir: Path) -> int:
    return len([path for path in (sequence_dir / "img1").iterdir() if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png"}])


def validate_sequence(asset_root: Path, dataset_root: Path, sequence: str) -> dict[str, Any]:
    sequence_root = asset_root / "candidates" / sequence
    index = json.loads((sequence_root / "index.json").read_text(encoding="utf-8"))
    done = json.loads((sequence_root / "done.json").read_text(encoding="utf-8"))
    metadata_path = sequence_root / "metadata.jsonl.zst"
    embedding_path = sequence_root / "embeddings.f16"
    frames = read_zstd_jsonl(metadata_path)
    embedding_count = int(index["embedding_count"])
    embeddings = np.memmap(embedding_path, mode="r", dtype=np.float16, shape=(embedding_count, 512))
    expected = expected_frames(dataset_root / "train" / sequence)
    errors: list[str] = []
    if index.get("stage") != "N72R20R2" or done.get("status") != "PASS_N72R20R2_CANDIDATE_STREAM_SEQUENCE":
        errors.append("stage_or_done_status")
    if int(index.get("frame_count", -1)) != expected or len(frames) != expected:
        errors.append(f"frame_count:{len(frames)}!={expected}")
    if embedding_path.stat().st_size != embedding_count * 512 * 2:
        errors.append("embedding_file_size")
    if str(index.get("metadata_sha256")) != sha256(metadata_path) or str(index.get("embeddings_sha256")) != sha256(embedding_path):
        errors.append("artifact_hash")
    seen_uids: set[str] = set()
    total_candidates = 0
    embedding_offsets: list[int] = []
    frame_numbers: list[int] = []
    for expected_frame, row in enumerate(frames):
        frame = int(row.get("frame", -1))
        frame_numbers.append(frame)
        if frame != expected_frame:
            errors.append(f"non_contiguous_frame:{expected_frame}:{frame}")
        if row.get("runtime_future_gt_used") is not False or row.get("runtime_gt_read") is not False:
            errors.append(f"runtime_gt_flag:{frame}")
        candidates = row.get("candidates", [])
        total_candidates += len(candidates)
        for candidate in candidates:
            uid = str(candidate.get("candidate_uid"))
            if uid in seen_uids or uid in {"", "None"}:
                errors.append(f"duplicate_uid:{frame}:{uid}")
            seen_uids.add(uid)
            offset = int(candidate.get("embedding_offset", -1))
            if offset < 0 or offset >= embedding_count:
                errors.append(f"bad_embedding_offset:{frame}:{offset}")
            else:
                embedding_offsets.append(offset)
            if candidate.get("source_public_id") is not None:
                errors.append(f"public_id_in_candidate:{frame}")
            if candidate.get("runtime_future_gt_used") is not False:
                errors.append(f"candidate_runtime_gt:{frame}")
            if len(str(candidate.get("feature_sha256", ""))) != 64:
                errors.append(f"feature_hash:{frame}")
    if int(index.get("candidate_count", -1)) != total_candidates:
        errors.append("candidate_count")
    if sorted(embedding_offsets) != list(range(embedding_count)):
        errors.append("embedding_offsets_not_dense")
    if frame_numbers != list(range(expected)):
        errors.append("frame_axis")
    return {
        "sequence": sequence,
        "status": "PASS" if not errors else "FAIL",
        "frames": expected,
        "candidates": total_candidates,
        "embedding_count": embedding_count,
        "bytes": metadata_path.stat().st_size + embedding_path.stat().st_size,
        "metadata_sha256": sha256(metadata_path),
        "embeddings_sha256": sha256(embedding_path),
        "sam3_chunk_size": index.get("sam3_chunk_size"),
        "sam3_session_count": len(index.get("sam3_session_records", [])),
        "runtime_future_gt_used": False,
        "errors": errors,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--asset-root", type=Path, default=DEFAULT_ASSET_ROOT)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--sequences", nargs="+", default=list(SEQUENCES))
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/N72R20R2/candidate_tape_validation.json")
    args = parser.parse_args()
    results = []
    for sequence in args.sequences:
        results.append(validate_sequence(args.asset_root.resolve(), args.dataset_root.resolve(), str(sequence)))
    failed = [item for item in results if item["status"] != "PASS"]
    payload = {
        "stage": "N72R20R2",
        "status": "PASS_N72R20R2_CANDIDATE_TAPE_VALIDATION" if not failed else "FAIL_N72R20R2_CANDIDATE_TAPE_VALIDATION",
        "sequences": results,
        "total_frames": sum(int(item["frames"]) for item in results),
        "total_candidates": sum(int(item["candidates"]) for item in results),
        "total_bytes": sum(int(item["bytes"]) for item in results),
        "runtime_future_gt_used": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": payload["status"], "sequences": len(results), "failed": len(failed), "output": str(args.output)}, sort_keys=True))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
