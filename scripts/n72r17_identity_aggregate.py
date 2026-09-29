#!/usr/bin/env python3
"""Aggregate N72R17 encoder and memory results without applying a decision."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


GOAL = "Human Identity Representation Probe"
QUESTION = "Can one human-confirmed identity observation reliably recognize the same person against hard competing identities over future frames?"


def load_required(path: Path) -> dict[str, object]:
    if not path.is_file():
        raise FileNotFoundError(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("goal") != GOAL:
        raise ValueError(f"Goal mismatch in {path}")
    return value


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--asset-manifest", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    args = parser.parse_args()

    encoders = ("osnet_x1_0_market1501", "openai_clip_vit_b32_zero_shot")
    aggregate: dict[str, object] = {
        "stage": "N72R17",
        "goal": GOAL,
        "central_question": QUESTION,
        "primary_endpoint": "VAL_H100_HARD_NEGATIVE_IDENTITY_WIN_RATE",
        "horizons": [20, 50, 100],
        "protocol": str(args.protocol),
        "asset_manifest": str(args.asset_manifest),
        "encoder_benchmark": {},
        "memory_baselines": {},
        "next_association_stage_authorized": False,
        "training_run": False,
        "sam3_run": False,
        "mot_evaluation_run": False,
    }
    for encoder in encoders:
        aggregate["encoder_benchmark"][encoder] = {
            split: load_required(args.results_dir / f"encoder_probe_{encoder}_{split}.json")
            for split in ("train", "val")
        }
        aggregate["memory_baselines"][encoder] = {
            split: load_required(args.results_dir / f"memory_probe_{encoder}_{split}.json")
            for split in ("train", "val")
        }
    aggregate["video_encoder"] = json.loads(args.asset_manifest.read_text(encoding="utf-8")).get("video_encoder")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(aggregate, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
