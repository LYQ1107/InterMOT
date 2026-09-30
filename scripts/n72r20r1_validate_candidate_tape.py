#!/usr/bin/env python3
"""Validate the provenance-only N72R20R1 SAM3 candidate tapes.

This validator reads only the compact candidate metadata and float16 feature
files.  It never opens DanceTrack GT and it never assigns a public identity.
The feature digest in metadata is the digest of the canonical float32 vector
before float16 tape storage; the validator therefore checks its presence and
the stored float16 payload independently.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ASSET_ROOT = Path("/data3/liuyeqiang/InterMOT_N72R20R1_assets")
SEQUENCES = ("dancetrack0001", "dancetrack0002")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def read_zstd_jsonl(path: Path) -> list[dict[str, Any]]:
    completed = subprocess.run(
        ["zstd", "-q", "-d", "-c", str(path)],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(completed.stdout.splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"{path}:{line_number} is not an object")
        rows.append(value)
    return rows


def finite_box(value: Any) -> bool:
    array = np.asarray(value, dtype=np.float64).reshape(-1)
    return bool(array.size == 4 and np.isfinite(array).all() and array[2] > array[0] and array[3] > array[1])


def validate_sequence(asset_root: Path, sequence: str) -> dict[str, Any]:
    sequence_root = asset_root / "candidates" / sequence
    index_path = sequence_root / "index.json"
    done_path = sequence_root / "done.json"
    metadata_path = sequence_root / "metadata.jsonl.zst"
    embeddings_path = sequence_root / "embeddings.f16"
    for path in (index_path, done_path, metadata_path, embeddings_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    index = read_json(index_path)
    done = read_json(done_path)
    if index.get("stage") != "N72R20R1" or done.get("status") != "PASS_N72R20R1_CANDIDATE_STREAM_SMOKE_SEQUENCE":
        raise ValueError(f"{sequence}: candidate tape stage/status mismatch")
    if index.get("sequence") != sequence or index.get("split") != "train":
        raise ValueError(f"{sequence}: candidate tape is not train lineage")
    if index.get("runtime_future_gt_used") is not False or index.get("runtime_gt_read") is not False:
        raise ValueError(f"{sequence}: candidate tape GT flags are not false")
    if index.get("metadata_sha256") != sha256(metadata_path):
        raise ValueError(f"{sequence}: metadata SHA256 mismatch")
    if index.get("embeddings_sha256") != sha256(embeddings_path):
        raise ValueError(f"{sequence}: embedding SHA256 mismatch")

    frame_rows = read_zstd_jsonl(metadata_path)
    frame_count = int(index.get("frame_count", -1))
    candidate_count = int(index.get("candidate_count", -1))
    embedding_count = int(index.get("embedding_count", -1))
    if [int(row.get("frame", -1)) for row in frame_rows] != list(range(frame_count)):
        raise ValueError(f"{sequence}: frame axis is not contiguous")
    if len(frame_rows) != frame_count:
        raise ValueError(f"{sequence}: metadata frame count mismatch")
    if embeddings_path.stat().st_size != embedding_count * 512 * np.dtype(np.float16).itemsize:
        raise ValueError(f"{sequence}: embedding byte count mismatch")
    embedding_array = np.memmap(embeddings_path, mode="r", dtype=np.float16, shape=(embedding_count, 512))

    seen_uids: set[str] = set()
    observed_candidates = 0
    observed_embeddings = 0
    null_iou_pred = 0
    finite_embedding_rows = 0
    frame_summaries: list[dict[str, Any]] = []
    for expected_frame, row in enumerate(frame_rows):
        if row.get("sequence") != sequence or row.get("split") != "train":
            raise ValueError(f"{sequence}:{expected_frame}: sequence/split mismatch")
        if row.get("runtime_future_gt_used") is not False or row.get("runtime_gt_read") is not False:
            raise ValueError(f"{sequence}:{expected_frame}: runtime GT flag violation")
        candidates = row.get("candidates")
        if not isinstance(candidates, list) or int(row.get("candidate_count", -1)) != len(candidates):
            raise ValueError(f"{sequence}:{expected_frame}: candidate count mismatch")
        frame_uids: set[str] = set()
        for candidate in candidates:
            uid = str(candidate.get("candidate_uid"))
            if uid in ("None", "") or uid in frame_uids or uid in seen_uids:
                raise ValueError(f"{sequence}:{expected_frame}: candidate UID collision/missing: {uid}")
            frame_uids.add(uid)
            seen_uids.add(uid)
            if int(candidate.get("frame", -1)) != expected_frame or not finite_box(candidate.get("box_xyxy")):
                raise ValueError(f"{sequence}:{expected_frame}: invalid candidate frame/box")
            if candidate.get("runtime_future_gt_used") is not False:
                raise ValueError(f"{sequence}:{expected_frame}: candidate GT flag violation")
            if candidate.get("raw_native_id") is None or candidate.get("native_tid") is None:
                raise ValueError(f"{sequence}:{expected_frame}: missing native provenance")
            if candidate.get("feature_source") != "frozen_osnet_x1_0_market1501_machine_box_crop":
                raise ValueError(f"{sequence}:{expected_frame}: feature source mismatch")
            feature_sha = str(candidate.get("feature_sha256", ""))
            if len(feature_sha) != 64 or any(char not in "0123456789abcdef" for char in feature_sha):
                raise ValueError(f"{sequence}:{expected_frame}: invalid feature SHA")
            offset = int(candidate.get("embedding_offset", -1))
            dim = int(candidate.get("embedding_dim", -1))
            if dim != 512 or offset < 0 or offset >= embedding_count:
                raise ValueError(f"{sequence}:{expected_frame}: invalid embedding index")
            stored = np.asarray(embedding_array[offset], dtype=np.float32)
            if not np.isfinite(stored).all() or float(np.linalg.norm(stored)) <= 1.0e-6:
                raise ValueError(f"{sequence}:{expected_frame}: invalid stored embedding")
            finite_embedding_rows += 1
            observed_embeddings += 1
            if candidate.get("iou_pred") is None:
                null_iou_pred += 1
        observed_candidates += len(candidates)
        frame_summaries.append({"frame": expected_frame, "candidate_count": len(candidates)})
    if observed_candidates != candidate_count or observed_embeddings != embedding_count:
        raise ValueError(f"{sequence}: candidate/embedding count mismatch")
    unexpected = [
        path.name
        for path in sequence_root.iterdir()
        if path.is_file()
        and path.name not in {"index.json", "done.json", "metadata.jsonl.zst", "embeddings.f16"}
        and not path.name.startswith(".")
    ]
    if unexpected:
        raise ValueError(f"{sequence}: unexpected dense/debug files: {unexpected}")
    profile = dict(done.get("profile", {}))
    if int(profile.get("frames", -1)) != frame_count or int(profile.get("candidates", -1)) != candidate_count:
        raise ValueError(f"{sequence}: done profile mismatch")
    return {
        "sequence": sequence,
        "status": "PASS",
        "frame_count": frame_count,
        "candidate_count": candidate_count,
        "embedding_count": embedding_count,
        "unique_candidate_uid_count": len(seen_uids),
        "finite_embedding_count": finite_embedding_rows,
        "null_iou_pred_count": null_iou_pred,
        "metadata_sha256": sha256(metadata_path),
        "embeddings_sha256": sha256(embeddings_path),
        "index_sha256": sha256(index_path),
        "done_sha256": sha256(done_path),
        "runtime_future_gt_used": False,
        "unexpected_files": unexpected,
        "frame_summaries": frame_summaries,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--asset-root", type=Path, default=DEFAULT_ASSET_ROOT)
    parser.add_argument("--sequences", nargs="+", default=list(SEQUENCES))
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/N72R20R1/candidate_tape_validation.json")
    args = parser.parse_args()
    results = [validate_sequence(args.asset_root.resolve(), str(sequence)) for sequence in args.sequences]
    payload = {
        "stage": "N72R20R1",
        "status": "PASS_N72R20R1_CANDIDATE_TAPE_VALIDATION",
        "asset_root": str(args.asset_root.resolve()),
        "sequences": results,
        "candidate_tape_is_fresh": True,
        "runtime_future_gt_used": False,
        "iou_pred_policy": "official SAM3 export did not expose predicted_iou; null retained, presence/confidence not substituted",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": payload["status"], "sequences": len(results)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
