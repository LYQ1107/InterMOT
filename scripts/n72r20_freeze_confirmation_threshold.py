#!/usr/bin/env python3
"""Freeze the non-learning 2-frame confirmation threshold from train only."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
TRAIN_QUANTILE = 0.10


def load_json(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object: {path}")
    return value


def run(args: argparse.Namespace) -> dict[str, object]:
    protocol = load_json(Path(args.protocol).expanduser().resolve())
    if protocol.get("stage") != "N72R20":
        raise ValueError("protocol is not N72R20")
    records_dir = Path(args.records_dir).expanduser().resolve()
    expected_sequences = set(str(item) for item in protocol["train_dev_sequences"])
    values: list[float] = []
    source_files: list[str] = []
    observed_sequences: set[str] = set()
    for path in sorted(records_dir.glob("*.immediate.jsonl")):
        source_files.append(str(path))
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                sequence = str(row["sequence"])
                observed_sequences.add(sequence)
                if (
                    row.get("method") == "B2_FROZEN_N72R18_GRU"
                    and
                    bool(row.get("target_visible", True))
                    and bool(row.get("target_candidate_available"))
                    and bool(row.get("hard_negative_available"))
                    and row.get("selected_candidate_is_target_posthoc") is True
                    and row.get("selection_margin_top1_top2") is not None
                ):
                    values.append(float(row["selection_margin_top1_top2"]))
    if not source_files:
        raise FileNotFoundError(f"no train immediate records under {records_dir}")
    if not observed_sequences.issubset(expected_sequences):
        raise ValueError(f"records contain non-train sequences: {sorted(observed_sequences - expected_sequences)}")
    if len(values) < int(args.min_samples):
        raise ValueError(f"only {len(values)} eligible train margins; need at least {args.min_samples}")
    threshold = float(np.quantile(np.asarray(values, dtype=np.float64), TRAIN_QUANTILE, method="linear"))
    result: dict[str, object] = {
        "stage": "N72R20",
        "split": "train",
        "frozen": True,
        "confirmation_margin_threshold": threshold,
        "selection_margin_definition": "top1 cosine score minus top2 cosine score on the machine candidate pool",
        "freeze_rule": "10th percentile of eligible correct-top1 train margins; no val records used",
        "train_quantile": TRAIN_QUANTILE,
        "eligible_margin_count": len(values),
        "train_sequences_observed": sorted(observed_sequences),
        "source_records": source_files,
        "val_tuning_used": False,
    }
    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": "PASS_N72R20_CONFIRMATION_THRESHOLD_FROZEN", "threshold": threshold, "output": str(output)}, ensure_ascii=False))
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--records-dir", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, default=ROOT / "outputs/N72R20/protocol.json")
    parser.add_argument("--min-samples", type=int, default=100)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        run(args)
    except Exception as exc:
        print(f"N72R20 threshold freeze failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
