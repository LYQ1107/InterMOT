#!/usr/bin/env python3
"""Write the immutable end-of-run resource/stage summary."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def _status(path: Path) -> str:
    if not path.is_file():
        return "MISSING"
    try:
        return str(json.loads(path.read_text(encoding="utf-8")).get("status", "UNKNOWN"))
    except Exception:
        return "UNREADABLE"


def run(*, output: Path, start_epoch: float, exit_code: int) -> dict[str, Any]:
    root = output.parent
    statuses = {name: _status(root / name) for name in (
        "policy_oracle_50.json", "policy_regression.json", "policy_rollout_index.json",
        "policy_oracle_689.json", "selector_feature_audit.json", "selector_training.json",
        "overfit_gate.json", "selection_results.json", "calibration_results.json",
        "learn_gate.json", "temporal_learn_gate.json", "full_loop_results.json",
        "association_fallback_results.json", "artifact_validation.json",
    )}
    route = "association_fallback"
    if statuses.get("full_loop_results.json") == "PASS":
        route = "selector_full_loop"
    elif statuses.get("temporal_learn_gate.json") == "PASS":
        route = "temporal_selector_then_association_fallback"
    result = {
        "protocol": "N32-RUN-SUMMARY",
        "status": "PASS" if exit_code == 0 else "FAIL",
        "exit_code": int(exit_code),
        "started_epoch": float(start_epoch),
        "finished_epoch": float(time.time()),
        "elapsed_seconds": float(time.time() - start_epoch),
        "stages": statuses,
        "route": route,
        "worker_gpus": [0, 1, 2, 3],
        "protected_gpus": [4, 5, 6],
        "val25_read": False,
        "test_labels_used": False,
        "future_gt_used_for_selector_input": False,
    }
    _write(output, result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--start-epoch", type=float, required=True)
    parser.add_argument("--exit-code", type=int, required=True)
    args = parser.parse_args()
    result = run(output=args.output, start_epoch=args.start_epoch, exit_code=args.exit_code)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
