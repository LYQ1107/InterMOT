#!/usr/bin/env python3
"""Create explicit NOT_RUN artifacts when an earlier N32 gate blocks learning."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


FILES = {
    "selector_training.json": "selector_training",
    "overfit_gate.json": "overfit_gate",
    "selection_results.json": "selection_results",
    "calibration_results.json": "calibration_results",
    "learn_gate.json": "learn_gate",
}


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def run(*, output_dir: Path, reason: str, source_status: str) -> None:
    for filename, protocol in FILES.items():
        _write(output_dir / filename, {
            "protocol": f"N32-NOT-RUN-{protocol.upper()}",
            "status": "NOT_RUN_UPSTREAM_GATE",
            "reason": reason,
            "upstream_status": source_status,
            "future_gt_used_for_selector_input": False,
            "future_gt_used_for_training_labels": False,
            "future_gt_used_for_selection": False,
            "val25_read": False,
            "test_labels_used": False,
        })


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--reason", required=True)
    parser.add_argument("--source-status", required=True)
    args = parser.parse_args()
    run(output_dir=args.output_dir, reason=args.reason, source_status=args.source_status)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
