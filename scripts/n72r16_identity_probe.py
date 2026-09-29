#!/usr/bin/env python3
"""Run the frozen single-anchor identity probe."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from sam3_intermot.identity_probe.metrics import sample_records, summarize_records, write_records
from sam3_intermot.identity_probe.protocol import read_protocol
from sam3_intermot.identity_probe.storage import EmbeddingStore


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--embedding-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-reps", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=7216)
    parser.add_argument("--spatial-mode", choices=("nearest_center", "highest_iou"), default="nearest_center")
    args = parser.parse_args()
    if args.bootstrap_reps < 100:
        raise ValueError("use at least 100 bootstrap repetitions")
    document = read_protocol(args.protocol)
    split = str(document["split"])
    store = EmbeddingStore(args.embedding_dir, split)
    records = sample_records(document, store, spatial_mode=args.spatial_mode)
    summary = summarize_records(records, bootstrap_reps=args.bootstrap_reps, seed=args.seed)
    summary.update(
        {
            "stage": "N72R16",
            "goal": "Human Identity Representation Probe",
            "central_question": "Can one human-confirmed identity observation reliably recognize the same person against hard competing identities over future frames?",
            "split": split,
            "protocol": str(args.protocol),
            "embedding_store": str(args.embedding_dir),
            "spatial_hard_mode": args.spatial_mode,
            "posthoc_oracle_memory": False,
        }
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_records(args.output_dir / f"identity_samples_{split}.jsonl", records)
    (args.output_dir / f"identity_probe_{split}.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"split": split, "samples": len(records), "summary": str(args.output_dir / f"identity_probe_{split}.json")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
