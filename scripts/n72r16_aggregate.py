#!/usr/bin/env python3
"""Collect compact N72R16 train/val metric summaries."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_if_present(path: Path) -> dict[str, object] | None:
    if not path.is_file():
        return None
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("goal") != "Human Identity Representation Probe":
        raise ValueError(f"Goal mismatch in {path}")
    return value


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = {
        "stage": "N72R16",
        "goal": "Human Identity Representation Probe",
        "central_question": "Can one human-confirmed identity observation reliably recognize the same person against hard competing identities over future frames?",
        "primary_endpoint": "VAL_H100_HARD_NEGATIVE_IDENTITY_WIN_RATE",
        "identity": {
            split: load_if_present(args.results_dir / f"identity_probe_{split}.json")
            for split in ("train", "val")
        },
        "memory": {
            split: load_if_present(args.results_dir / f"memory_probe_{split}.json")
            for split in ("train", "val")
        },
        "next_association_stage_authorized": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
