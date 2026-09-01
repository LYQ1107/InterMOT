#!/usr/bin/env python3
"""Validate the independently repaired N25-R train30 shadow cache."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATA = Path("/path/to/dancetrack/train")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_jsonl(path: Path):
    with path.open(encoding="utf-8") as handle:
        for line_index, line in enumerate(handle):
            if line.strip():
                yield line_index, json.loads(line)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cache-dir",
        default="outputs/n25r/protocol_repair/full_shadow_cache_train30",
    )
    parser.add_argument(
        "--attempts-csv", default="outputs/n20/dataset_attempts_train30.csv"
    )
    parser.add_argument("--old-s0-prefix-lines", type=int, default=1634)
    parser.add_argument(
        "--out", default="outputs/n25r/protocol_repair/repair_validation.json"
    )
    args = parser.parse_args()
    cache_dir = ROOT / args.cache_dir
    attempts_path = ROOT / args.attempts_csv
    out_path = ROOT / args.out

    attempts = [
        row
        for row in csv.DictReader(attempts_path.open(encoding="utf-8"))
        if row.get("target_present") == "1"
    ]
    expected_groups = {
        (row["sequence"], int(row["frame"]), int(row["gid"])) for row in attempts
    }
    rows = []
    new_rows = []
    files = sorted(cache_dir.glob("*.jsonl"))
    for path in files:
        for line_index, row in load_jsonl(path):
            rows.append(row)
            if path.name.endswith("_s0.jsonl") and line_index >= args.old_s0_prefix_lines:
                new_rows.append(row)

    groups = defaultdict(list)
    candidate_keys = []
    for row in rows:
        group_key = (row["sequence"], int(row["frame"]), int(row["gid"]))
        groups[group_key].append(row)
        candidate_keys.append((*group_key, int(row["candidate_rank"])))
    rank_gap_groups = []
    for key, members in groups.items():
        ranks = sorted(int(row["candidate_rank"]) for row in members)
        if ranks != list(range(1, max(ranks) + 1)):
            rank_gap_groups.append({"group": key, "ranks": ranks})

    num_frames = {}
    bad_frame_ranges = []
    for row in new_rows:
        sequence = row["sequence"]
        if sequence not in num_frames:
            num_frames[sequence] = len(list((DATA / sequence / "img1").glob("*.jpg")))
        frames = row.get("frames", [])
        expected_last = min(int(row["frame"]) + 8, num_frames[sequence] - 1)
        if (
            not frames
            or int(frames[0]["frame"]) != int(row["frame"])
            or int(frames[-1]["frame"]) != expected_last
        ):
            bad_frame_ranges.append(
                {
                    "key": [
                        sequence,
                        int(row["frame"]),
                        int(row["gid"]),
                        int(row["candidate_rank"]),
                    ],
                    "observed": [frame.get("frame") for frame in frames],
                    "expected_last": expected_last,
                }
            )

    old_s0 = ROOT / "outputs/n20/full_shadow_cache_train30/train30_dataset_k5_s0.jsonl"
    old_s1 = ROOT / "outputs/n20/full_shadow_cache_train30/train30_dataset_k5_s1.jsonl"
    repaired_s1 = cache_dir / "train30_dataset_k5_s1.jsonl"
    summary = {
        "status": "PASS",
        "expected_groups": len(expected_groups),
        "actual_groups": len(groups),
        "missing_groups": len(expected_groups - set(groups)),
        "extra_groups": len(set(groups) - expected_groups),
        "rows": len(rows),
        "unique_candidate_keys": len(set(candidate_keys)),
        "duplicate_candidate_keys": len(candidate_keys) - len(set(candidate_keys)),
        "rank_gap_groups": len(rank_gap_groups),
        "candidate_count_distribution": dict(sorted(Counter(len(v) for v in groups.values()).items())),
        "new_s0_rows": len(new_rows),
        "new_frame_count_distribution": dict(sorted(Counter(len(row.get("frames", [])) for row in new_rows).items())),
        "new_non_null_traj_distribution": dict(sorted(Counter(int(row.get("traj_len", 0)) for row in new_rows).items())),
        "new_bad_frame_ranges": len(bad_frame_ranges),
        "old_s0_sha256_unchanged": sha256(old_s0),
        "old_s1_sha256_unchanged": sha256(old_s1),
        "repaired_s1_is_exact_copy": sha256(old_s1) == sha256(repaired_s1),
        "files": [{"path": str(path.relative_to(ROOT)), "bytes": path.stat().st_size, "sha256": sha256(path)} for path in files],
        "rank_gap_examples": rank_gap_groups[:10],
        "bad_frame_range_examples": bad_frame_ranges[:10],
    }
    required_zero = (
        summary["missing_groups"],
        summary["extra_groups"],
        summary["duplicate_candidate_keys"],
        summary["rank_gap_groups"],
        summary["new_bad_frame_ranges"],
    )
    if summary["actual_groups"] != 1000 or any(required_zero) or not summary["repaired_s1_is_exact_copy"]:
        summary["status"] = "FAIL"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    if summary["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
