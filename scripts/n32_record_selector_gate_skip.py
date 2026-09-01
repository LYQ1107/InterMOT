#!/usr/bin/env python3
"""Record the protocol-authorized selector skip after an Oracle Gate failure.

N32 permits no selector training or temporal fallback when the 689-episode
policy Oracle fails its frozen gate.  This script writes explicit NOT_RUN
artifacts rather than leaving missing files that could be mistaken for an
implementation failure.  It never reads validation/test contents and never
changes the policy index or any policy artifact.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "outputs/n32"
ORACLE = OUT_DIR / "policy_oracle_689.json"
AUDIT = OUT_DIR / "selector_feature_audit.json"


def _write(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def run(*, output_dir: Path = OUT_DIR) -> dict[str, Any]:
    oracle = json.loads(ORACLE.read_text(encoding="utf-8"))
    audit = json.loads(AUDIT.read_text(encoding="utf-8"))
    if oracle.get("status") == "PASS":
        raise RuntimeError("selector skip is only legal when the 689-episode Oracle Gate fails")
    identity_count = audit.get("identity_features_available_episode_count", (audit.get("feature_availability") or {}).get("identity_features_available_episode_count"))
    identity_coverage = audit.get("identity_feature_coverage", 0.0)
    common = {
        "status": "NOT_RUN_ORACLE_GATE_FAIL",
        "reason": "the frozen N32 689-episode policy Oracle Gate failed; selector training and temporal fallback are not authorized",
        "oracle_status": oracle.get("status"),
        "oracle_gate_checks": oracle.get("gate_checks", {}),
        "oracle_thresholds": oracle.get("thresholds", {}),
        "identity_features_available_episode_count": identity_count,
        "identity_feature_coverage": identity_coverage,
        "identity_aware_learning_valid": False if identity_count == 0 else audit.get("identity_aware_learning_valid"),
        "selector_scope": "TEMPORAL_GEOMETRY_ONLY_FALLBACK" if identity_count == 0 else "CAUSAL_FEATURE_SELECTOR_WITH_IDENTITY_MEMORY_WHERE_AVAILABLE",
        "future_gt_used_for_selector_input": False,
        "future_gt_used_for_training_labels": False,
        "val25_read": False,
        "test_labels_used": False,
    }
    artifacts = {
        "selector_training.json": {"protocol": "N32-EF-SELECTOR-TRAIN", "feature_dimension": audit.get("feature_dimension"), **common},
        "overfit_gate.json": {"protocol": "N32-E-OVERFIT-GATE", **common},
        "selection_results.json": {"protocol": "N32-G-SELECTION", "split": "selection", **common},
        "calibration_results.json": {"protocol": "N32-G-CALIBRATION", "split": "calibration", **common},
        "learn_gate.json": {"protocol": "N32-G-LEARN-GATE", "checks": {"oracle_gate_pass": False}, **common},
        "temporal_selector_training.json": {"protocol": "N32-G-TEMPORAL-TRAIN", "feature_dimension": audit.get("feature_dimension"), **common},
        "temporal_overfit_gate.json": {"protocol": "N32-G-TEMPORAL-OVERFIT-GATE", **common},
        "temporal_selection_results.json": {"protocol": "N32-G-TEMPORAL-SELECTION", **common},
        "temporal_calibration_results.json": {"protocol": "N32-G-TEMPORAL-CALIBRATION", **common},
        "temporal_learn_gate.json": {"protocol": "N32-G-TEMPORAL-FALLBACK", "route": "association_fallback", **common},
    }
    for name, value in artifacts.items():
        _write(output_dir / name, value)
    route_gate = {
        "protocol": "N32-SELECTOR-ROUTE-GATE",
        "status": "FALLBACK_AUTHORIZED",
        "route": "association_fallback",
        "oracle_status": oracle.get("status"),
        "selector_training_status": common["status"],
        "temporal_fallback_status": common["status"],
        "identity_features_available_episode_count": identity_count,
        "identity_feature_coverage": identity_coverage,
        "identity_aware_learning_valid": common["identity_aware_learning_valid"],
        "reason": common["reason"],
        "future_gt_used_for_selection": False,
        "val25_read": False,
        "test_labels_used": False,
    }
    _write(output_dir / "selector_route_gate.json", route_gate)
    return route_gate


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=OUT_DIR)
    args = parser.parse_args()
    result = run(output_dir=args.output_dir)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
