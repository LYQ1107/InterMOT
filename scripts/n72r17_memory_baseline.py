#!/usr/bin/env python3
"""Evaluate causal diagnostic memories without changing the tracker."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from sam3_intermot.identity_research.memory import memory_records, summarize_memory
from sam3_intermot.identity_research.metrics import write_records
from sam3_intermot.identity_research.protocol import read_frozen_protocol
from sam3_intermot.identity_research.storage import EmbeddingStore


def resolve_store_dir(root: Path, split: str, encoder: str) -> Path:
    candidates = [
        root / f"embeddings_{split}",
        root / f"embeddings_{encoder}_{split}",
        root / f"embeddings_clip_{split}",
    ]
    for candidate in candidates:
        if (candidate / f"store_{split}.json").is_file():
            return candidate
    raise FileNotFoundError(f"no {split} embedding store under {root}; tried {candidates}")


def run_one(
    encoder: str,
    split: str,
    protocol_path: Path,
    embedding_dir: Path,
    output_dir: Path,
    bootstrap_reps: int,
    seed: int,
) -> dict[str, object]:
    document = read_frozen_protocol(protocol_path)
    store = EmbeddingStore(embedding_dir, split)
    records = memory_records(document, store)
    summary = summarize_memory(records, bootstrap_reps=bootstrap_reps, seed=seed)
    summary.update(
        {
            "stage": "N72R17",
            "goal": "Human Identity Representation Probe",
            "central_question": "Can one human-confirmed identity observation reliably recognize the same person against hard competing identities over future frames?",
            "encoder": encoder,
            "split": split,
            "protocol": str(protocol_path),
            "embedding_store": str(embedding_dir),
            "training": False,
        }
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    for variant, rows in records.items():
        write_records(output_dir / f"memory_samples_{encoder}_{split}_{variant}.jsonl", rows)
    result_path = output_dir / f"memory_probe_{encoder}_{split}.json"
    result_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {
        "encoder": encoder,
        "split": split,
        "variants": {variant: len(rows) for variant, rows in records.items()},
        "result": str(result_path),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol-dir", type=Path, required=True)
    parser.add_argument("--osnet-embedding-root", type=Path, required=True)
    parser.add_argument("--clip-embedding-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-reps", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=7217)
    args = parser.parse_args()
    if args.bootstrap_reps < 100:
        raise ValueError("use at least 100 bootstrap repetitions")
    results = []
    for encoder, store_root in (("osnet_x1_0_market1501", args.osnet_embedding_root), ("openai_clip_vit_b32_zero_shot", args.clip_embedding_root)):
        for split in ("train", "val"):
            results.append(
                run_one(
                    encoder,
                    split,
                    args.protocol_dir / f"protocol_{split}.json",
                    resolve_store_dir(store_root, split, encoder),
                    args.output_dir,
                    args.bootstrap_reps,
                    args.seed,
                )
            )
    print(json.dumps(results, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
