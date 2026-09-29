#!/usr/bin/env python3
"""Evaluate OSNet and public CLIP under the exact N72R16 hard-negative probe."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from sam3_intermot.identity_research.metrics import sample_records, summarize_records, write_records
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
    spatial_mode: str,
) -> dict[str, object]:
    document = read_frozen_protocol(protocol_path)
    if document.get("split") != split:
        raise ValueError(f"protocol split mismatch: expected {split}, got {document.get('split')}")
    store = EmbeddingStore(embedding_dir, split)
    records = sample_records(document, store, spatial_mode=spatial_mode)
    summary = summarize_records(records, bootstrap_reps=bootstrap_reps, seed=seed)
    summary.update(
        {
            "stage": "N72R17",
            "goal": "Human Identity Representation Probe",
            "central_question": "Can one human-confirmed identity observation reliably recognize the same person against hard competing identities over future frames?",
            "encoder": encoder,
            "split": split,
            "protocol": str(protocol_path),
            "embedding_store": str(embedding_dir),
            "spatial_hard_mode": spatial_mode,
            "weights_frozen": True,
            "training": False,
        }
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    write_records(output_dir / f"encoder_samples_{encoder}_{split}.jsonl", records)
    result_path = output_dir / f"encoder_probe_{encoder}_{split}.json"
    result_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {"encoder": encoder, "split": split, "samples": len(records), "result": str(result_path)}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol-dir", type=Path, required=True)
    parser.add_argument("--osnet-embedding-root", type=Path, required=True)
    parser.add_argument("--clip-embedding-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-reps", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=7217)
    parser.add_argument("--spatial-mode", choices=("nearest_center", "highest_iou"), default="nearest_center")
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
                    args.spatial_mode,
                )
            )
    print(json.dumps(results, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
