#!/usr/bin/env python3
"""Plan inclusive, overlapping N36 frame-range shards."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.n36_tape_common import (
    DATA_ROOT,
    DEFAULT_SEQUENCE_LIST,
    atomic_json,
    image_files,
    load_sequences,
)


def plan_sequence(sequence: str, frame_count: int, chunk_size: int, overlap: int) -> list[dict]:
    step = chunk_size - overlap
    chunks: list[dict] = []
    starts = list(range(0, frame_count, step))
    for index, start in enumerate(starts):
        end = min(frame_count - 1, start + chunk_size - 1)
        next_start = starts[index + 1] if index + 1 < len(starts) else None
        core_end = (next_start - 1) if next_start is not None else end
        if core_end < start:
            raise ValueError(
                f"invalid core range for {sequence} chunk {index}: {start}..{core_end}"
            )
        previous_overlap = (
            None
            if index == 0
            else {
                "start": start,
                "end": min(end, start + overlap - 1),
                "with_chunk_id": f"{sequence}_chunk_{index - 1:04d}",
            }
        )
        next_overlap = (
            None
            if next_start is None
            else {
                "start": max(start, end - overlap + 1),
                "end": end,
                "with_chunk_id": f"{sequence}_chunk_{index + 1:04d}",
            }
        )
        chunks.append(
            {
                "chunk_id": f"{sequence}_chunk_{index:04d}",
                "sequence": sequence,
                # frame_start/frame_end are the actual inclusive range sent
                # to the official predictor.  core_* is the unique owner
                # range used by the merge to remove overlap duplicates.
                "frame_start": int(start),
                "frame_end": int(end),
                "core_frame_start": int(start),
                "core_frame_end": int(core_end),
                "overlap_frames": int(overlap),
                "previous_overlap": previous_overlap,
                "next_overlap": next_overlap,
                "frame_count_total": int(frame_count),
                "chunk_index": int(index),
                "chunk_count": int(len(starts)),
            }
        )
    return chunks


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sequence-list", type=Path, default=DEFAULT_SEQUENCE_LIST)
    parser.add_argument("--sequences", default="")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/n36/chunk_manifests/chunk_plan.json")
    parser.add_argument("--chunk-size", type=int, default=160)
    parser.add_argument("--overlap", type=int, default=20)
    args = parser.parse_args()
    if not 50 <= args.chunk_size <= 200:
        raise ValueError("N36 chunk-size must be between 50 and 200 inclusive")
    if args.overlap < 1 or args.overlap >= args.chunk_size:
        raise ValueError("overlap must be positive and smaller than chunk-size")
    sequences = load_sequences(args.sequence_list, args.sequences)
    if not sequences:
        raise ValueError("no sequences selected")
    plan_sequences: list[dict] = []
    chunks: list[dict] = []
    for sequence in sequences:
        sequence_dir = DATA_ROOT / "train" / sequence
        paths = image_files(sequence_dir)
        if not paths:
            raise FileNotFoundError(f"no train/train_fold frames for {sequence}: {sequence_dir}")
        seq_chunks = plan_sequence(sequence, len(paths), args.chunk_size, args.overlap)
        plan_sequences.append(
            {
                "sequence": sequence,
                "frame_count": len(paths),
                "chunk_count": len(seq_chunks),
                "chunks": seq_chunks,
            }
        )
        chunks.extend(seq_chunks)
    payload = {
        "protocol": "N36_FRAME_RANGE_CHUNK_PLAN",
        "status": "PASS",
        "dataset_split": "train/train_fold",
        "frame_index_base": 0,
        "range_end_inclusive": True,
        "chunk_size": int(args.chunk_size),
        "overlap": int(args.overlap),
        "sequence_count": len(plan_sequences),
        "chunk_count": len(chunks),
        "sequences": plan_sequences,
        "all_chunks": chunks,
        "runtime_future_gt_used": False,
        "third_party_modified": False,
    }
    atomic_json(args.output.resolve(), payload)
    print(json.dumps({"output": str(args.output.resolve()), **{k: payload[k] for k in ("sequence_count", "chunk_count", "chunk_size", "overlap")}}, sort_keys=True))


if __name__ == "__main__":
    main()
