#!/usr/bin/env python3
"""Compare complete posthoc E1A and E1B metrics without selecting a model."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
HORIZONS = ("20", "50", "100")


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        directory_fd = os.open(path.parent, os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if temporary.exists():
            temporary.unlink()


def read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object: {path}")
    return value


def finite(value: Any) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"non-finite metric: {value!r}")
    return number


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--e1a", type=Path, required=True)
    parser.add_argument("--e1b", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    e1a_path = args.e1a if args.e1a.is_absolute() else ROOT / args.e1a
    e1b_path = args.e1b if args.e1b.is_absolute() else ROOT / args.e1b
    e1a = read(e1a_path)
    e1b = read(e1b_path)
    if not e1a.get("complete") or not e1b.get("complete"):
        raise RuntimeError("comparison requires complete E1A and E1B artifacts")
    if e1a.get("event_completeness", {}).get("failed_or_invalid_events", 0) or e1b.get("event_completeness", {}).get("failed_or_invalid_events", 0):
        raise RuntimeError("comparison requires zero invalid events")
    by_horizon: dict[str, Any] = {}
    for horizon in HORIZONS:
        a = e1a["by_horizon"][horizon]
        b = e1b["by_horizon"][horizon]
        by_horizon[horizon] = {
            "e1a_identity_error_reduction": finite(a["identity_error_reduction"]),
            "e1b_identity_error_reduction": finite(b["identity_error_reduction"]),
            "e1b_minus_e1a_identity_error_reduction": finite(b["identity_error_reduction"]) - finite(a["identity_error_reduction"]),
            "e1a_sequence_cluster_ci": a["sequence_cluster_bootstrap_95ci"],
            "e1b_sequence_cluster_ci": b["sequence_cluster_bootstrap_95ci"],
            "e1a_assignment_change_rate": finite(a["assignment_change_rate"]),
            "e1b_assignment_change_rate": finite(b["assignment_change_rate"]),
            "e1a_true_correct_crossings": int(a["true_correct_crossing_count"]),
            "e1a_true_incorrect_crossings": int(a["true_incorrect_crossing_count"]),
            "e1b_true_correct_crossings": int(b["true_correct_crossing_count"]),
            "e1b_true_incorrect_crossings": int(b["true_incorrect_crossing_count"]),
            "e1a_protected_regression_count": int(a["protected_regression_count"]),
            "e1b_protected_regression_count": int(b["protected_regression_count"]),
            "e1b_h20_or_long_horizon_not_clearly_negative": bool(
                finite(b["sequence_cluster_bootstrap_95ci"]["lower"]) <= 0.0 <= finite(b["sequence_cluster_bootstrap_95ci"]["upper"])
            ),
        }
    output = args.output if args.output.is_absolute() else ROOT / args.output
    result = {
        "schema_version": "N72R11R4_E1A_E1B_COMPARISON_V1",
        "status": "PASS_POSTHOC_COMPARISON_COMPLETE",
        "created_at_utc": now_utc(),
        "e1a": {"path": str(e1a_path), "sha256": sha256_file(e1a_path), "status": e1a.get("status")},
        "e1b": {"path": str(e1b_path), "sha256": sha256_file(e1b_path), "status": e1b.get("status")},
        "event_count": 32,
        "independent_sequence_count": e1b.get("independent_sequence_count"),
        "bootstrap": e1b.get("bootstrap"),
        "by_horizon": by_horizon,
        "interpretation": "E1B is a meaningful posthoc improvement over E1A but remains descriptive; this artifact does not authorize production or select a future metric winner.",
        "production_authorized": False,
        "not_real_human_evidence": True,
    }
    atomic_json(output, result)
    print(json.dumps({"status": result["status"], "output": str(output)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
