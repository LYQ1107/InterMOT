#!/usr/bin/env python3
"""Run the posthoc oracle-positive-memory diagnostic."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from sam3_intermot.identity_probe.memory import memory_records, summarize_memory
from sam3_intermot.identity_probe.metrics import write_records
from sam3_intermot.identity_probe.protocol import read_protocol
from sam3_intermot.identity_probe.storage import EmbeddingStore


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--embedding-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-reps", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=7216)
    args = parser.parse_args()
    document = read_protocol(args.protocol)
    split = str(document["split"])
    store = EmbeddingStore(args.embedding_dir, split)
    records = memory_records(document, store)
    summary = summarize_memory(records, bootstrap_reps=args.bootstrap_reps, seed=args.seed)
    summary.update(
        {
            "stage": "N72R16",
            "goal": "Human Identity Representation Probe",
            "central_question": "Can one human-confirmed identity observation reliably recognize the same person against hard competing identities over future frames?",
            "split": split,
            "protocol": str(args.protocol),
            "embedding_store": str(args.embedding_dir),
        }
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for variant, rows in records.items():
        write_records(args.output_dir / f"memory_samples_{split}_{variant}.jsonl", rows)
    (args.output_dir / f"memory_probe_{split}.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"split": split, "variants": {key: len(value) for key, value in records.items()}}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
