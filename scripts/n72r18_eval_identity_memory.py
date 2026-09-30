#!/usr/bin/env python
"""Evaluate an N72R18 HIIM checkpoint on the frozen validation protocol."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from sam3_intermot.identity_memory.dataset import FrozenIdentityEpisodeDataset
from sam3_intermot.identity_memory.evaluator import evaluate_model, write_jsonl
from sam3_intermot.identity_memory.updater import build_updater


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--protocol",
        default="/data3/liuyeqiang/InterMOT_N72R17_assets/outputs/N72R17/protocol_val.json",
    )
    parser.add_argument(
        "--store",
        default="/data3/liuyeqiang/InterMOT_N72R16_assets/outputs/N72R16/embeddings_val",
    )
    parser.add_argument("--split", default="val")
    parser.add_argument("--output-dir", default="outputs/N72R18")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--bootstrap-reps", type=int, default=2000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    variant = str(checkpoint["variant"])
    config = dict(checkpoint["config"])
    model = build_updater(
        variant,
        feature_dim=int(config.get("feature_dimension", 512)),
        gate_hidden_dim=int(config.get("gate_hidden_dim", 256)),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    dataset = FrozenIdentityEpisodeDataset(args.protocol, args.store, split=args.split)
    records, updates, summary = evaluate_model(
        dataset,
        model,
        device=device,
        batch_size=args.batch_size,
        bootstrap_reps=args.bootstrap_reps,
    )

    output_dir = Path(args.output_dir)
    metrics_dir = output_dir / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    slug = "gru" if variant == "GRU_MEMORY" else "gru_gate"
    summary.update(
        {
            "checkpoint": str(Path(args.checkpoint).resolve()),
            "protocol": str(Path(args.protocol).resolve()),
            "embedding_store": str(Path(args.store).resolve()),
            "split": args.split,
            "training": False,
        }
    )
    (metrics_dir / f"{slug}_{args.split}_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    write_jsonl(metrics_dir / f"{slug}_{args.split}_records.jsonl", records)
    write_jsonl(metrics_dir / f"{slug}_{args.split}_updates.jsonl", updates)
    print(
        json.dumps(
            {
                "variant": variant,
                "split": args.split,
                "samples": len(records),
                "h100_win_rate": summary["horizons"]["H100"]["hard_negative_win_rate"],
                "summary": str((metrics_dir / f"{slug}_{args.split}_summary.json").resolve()),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
