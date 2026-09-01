#!/usr/bin/env python3
"""Materialize N31 full-loop/fallback status without hiding a failed gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "outputs/n31"


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _load(path: Path, default: Any) -> Any:
    if not path.is_file():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def run(*, output_dir: Path) -> dict[str, Any]:
    learn = _load(output_dir / "learn_gate.json", {"status": "NOT_RUN"})
    fallback = _load(output_dir / "fallback_results.json", {"status": "NOT_RUN"})
    if learn.get("status") == "PASS":
        result = {
            "protocol": "N31-FULL-CLOSED-LOOP",
            "status": "NOT_RUN_IMPLEMENTATION_REQUIRES_DEPLOYMENT_GATE_REVIEW",
            "reason": "selector gate passed but this artifact generator does not silently substitute offline candidate rows for a real interactive loop",
            "learn_gate": learn,
            "future_gt_used_for_selection": False,
            "val25_read": False,
            "test_labels_used": False,
        }
    else:
        result = {
            "protocol": "N31-FULL-CLOSED-LOOP",
            "status": "NOT_RUN_LEARN_GATE_FAIL",
            "reason": "N31 learning gate did not authorize deployment",
            "learn_gate": learn,
            "fallback_route": fallback.get("route"),
            "future_gt_used_for_selection": False,
            "val25_read": False,
            "test_labels_used": False,
        }
    _write(output_dir / "full_loop_results.json", result)
    _write(output_dir / "full_loop_train.json", {"protocol": "N31-FULL-LOOP-TRAIN", "status": result["status"], "source": "full_loop_results.json"})
    _write(output_dir / "full_loop_fallback.json", {"protocol": "N31-FULL-LOOP-FALLBACK", "status": fallback.get("status", "NOT_RUN"), "route": fallback.get("route"), "source": "fallback_results.json"})
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUT_DIR)
    args = parser.parse_args()
    result = run(output_dir=args.output_dir)
    print(json.dumps({key: result.get(key) for key in ("protocol", "status", "fallback_route")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
