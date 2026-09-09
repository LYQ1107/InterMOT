#!/usr/bin/env python3
"""Compare corrected N72R11R5R1 metrics with the frozen N72R11R4 metrics."""

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
HORIZONS = (20, 50, 100)
DEFAULT_OUTPUT = ROOT / "outputs/N72R11R5R1/geometry_repair_effect.json"


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object: {path}")
    return value


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


def _number(value: Any) -> float | int | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        return None
    return value


def _horizon_comparison(old: dict[str, Any], corrected: dict[str, Any]) -> dict[str, Any]:
    fields = (
        "identity_error_reduction",
        "delta_iou",
        "assignment_change_rate",
        "target_assignment_change_rate",
        "id_switch_rate",
        "recorrection_rate",
        "candidate_recall",
        "missing_rate",
        "protected_regression_rate",
        "true_correct_crossing_count",
        "true_incorrect_crossing_count",
        "assignment_change_count",
        "target_assignment_change_count",
        "protected_regression_count",
        "baseline_future_identity_error",
        "treatment_future_identity_error",
    )
    result: dict[str, Any] = {}
    for field in fields:
        old_value = _number(old.get(field))
        corrected_value = _number(corrected.get(field))
        result[field] = {
            "old_r4": old_value,
            "corrected_r5r1": corrected_value,
            "delta_corrected_minus_old": None if old_value is None or corrected_value is None else float(corrected_value) - float(old_value),
        }
    old_ci = old.get("sequence_cluster_bootstrap_95ci")
    corrected_ci = corrected.get("sequence_cluster_bootstrap_95ci")
    result["sequence_cluster_bootstrap_95ci"] = {
        "old_r4": old_ci,
        "corrected_r5r1": corrected_ci,
    }
    return result


def compare(old_e1a: Path, corrected_e1a: Path, old_e1b: Path, corrected_e1b: Path, source_failure: Path) -> dict[str, Any]:
    paths = {
        "old_e1a": old_e1a,
        "corrected_e1a": corrected_e1a,
        "old_e1b": old_e1b,
        "corrected_e1b": corrected_e1b,
    }
    payloads = {name: read_json(path) for name, path in paths.items()}
    horizon_results: dict[str, Any] = {}
    for label, old_name, corrected_name in (
        ("E1A_vs_E0", "old_e1a", "corrected_e1a"),
        ("E1B_vs_E0", "old_e1b", "corrected_e1b"),
    ):
        old = payloads[old_name]
        corrected = payloads[corrected_name]
        horizon_results[label] = {
            str(horizon): _horizon_comparison(
                dict(old.get("by_horizon", {}).get(str(horizon), {})),
                dict(corrected.get("by_horizon", {}).get(str(horizon), {})),
            )
            for horizon in HORIZONS
        }
    failure_payload = read_json(source_failure) if source_failure.is_file() else None
    complete = all(payloads[name].get("complete") is True for name in payloads if name.startswith("corrected"))
    strict_ci = {
        label: {
            "h20_lower": (
                lower
                := payloads[name]
                .get("by_horizon", {})
                .get("20", {})
                .get("sequence_cluster_bootstrap_95ci", {})
                .get("lower")
            ),
            "lower_ci_gt_zero": bool(
                isinstance(lower, (int, float)) and not isinstance(lower, bool) and math.isfinite(float(lower)) and float(lower) > 0.0
            ),
        }
        for label, name in (("E1A_vs_E0", "corrected_e1a"), ("E1B_vs_E0", "corrected_e1b"))
    }
    return {
        "schema_version": "N72R11R5R1_GEOMETRY_REPAIR_EFFECT_V1",
        "status": "PASS_GEOMETRY_REPAIR_COMPARISON" if complete else "FAIL_GEOMETRY_REPAIR_COMPARISON",
        "created_at_utc": now_utc(),
        "historical_r4_source_failure": {
            "path": str(source_failure),
            "sha256": sha256_file(source_failure) if source_failure.is_file() else None,
            "payload": failure_payload,
        },
        "metrics_sources": {
            name: {"path": str(path), "sha256": sha256_file(path)} for name, path in paths.items()
        },
        "corrected_replay_complete": complete,
        "comparison": horizon_results,
        "strict_h20_future_effect_check": strict_ci,
        "research_effect_gate": "FAIL_FUTURE_EFFECT",
        "geometry_repair_scope": "positive finite area eligibility before model and exact solver; no box repair or metric change",
        "historical_outputs_modified": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--old-e1a", type=Path, default=ROOT / "outputs/N72R11R4/formal_e1a_metrics.json")
    parser.add_argument("--corrected-e1a", type=Path, default=ROOT / "outputs/N72R11R5R1/formal_e1a_metrics.json")
    parser.add_argument("--old-e1b", type=Path, default=ROOT / "outputs/N72R11R4/formal_e1b_metrics.json")
    parser.add_argument("--corrected-e1b", type=Path, default=ROOT / "outputs/N72R11R5R1/formal_e1b_metrics.json")
    parser.add_argument("--source-failure", type=Path, default=ROOT / "outputs/N72R11R5/export_failure_invalid_assigned_boxes.json")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    output = args.output if args.output.is_absolute() else ROOT / args.output
    result = compare(
        *(path if path.is_absolute() else ROOT / path for path in (args.old_e1a, args.corrected_e1a, args.old_e1b, args.corrected_e1b, args.source_failure))
    )
    atomic_json(output, result)
    print(json.dumps({"status": result["status"], "output": str(output), "research_effect_gate": result["research_effect_gate"]}, sort_keys=True))
    return 0 if result["status"] == "PASS_GEOMETRY_REPAIR_COMPARISON" else 1


if __name__ == "__main__":
    raise SystemExit(main())
