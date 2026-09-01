#!/usr/bin/env python3
"""Structural and blind-boundary validation for the N32 artifact bundle."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "outputs/n32"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from sam3_intermot.adaptation.correction_selector_features import FEATURE_NAMES  # noqa: E402


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def _load(name: str) -> dict[str, Any]:
    path = OUT_DIR / name
    if not path.is_file():
        return {"status": "MISSING", "_path": str(path)}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(value, dict):
            return value
        return {"status": "INVALID", "_path": str(path)}
    except Exception as exc:
        return {"status": "UNREADABLE", "error": f"{type(exc).__name__}: {exc}", "_path": str(path)}


def _blind(name: str, artifact: Mapping[str, Any], issues: list[str]) -> None:
    for key in ("val25_read", "test_labels_used", "future_gt_used_for_selection", "future_gt_used_for_selector_input"):
        if key in artifact and artifact[key] is not False:
            issues.append(f"{name}.{key} is not false")


def run(*, output: Path = OUT_DIR / "artifact_validation.json") -> dict[str, Any]:
    issues: list[str] = []
    names = [
        "frozen_protocol.json", "policy_regression.json", "policy_oracle_50.json",
        "policy_rollout_index.json", "policy_oracle_689.json", "selector_feature_audit.json",
        "selector_training.json", "overfit_gate.json", "selection_results.json",
        "calibration_results.json", "learn_gate.json", "temporal_learn_gate.json", "full_loop_results.json",
        "association_fallback_results.json",
    ]
    artifacts = {name: _load(name) for name in names}
    for name, artifact in artifacts.items():
        if artifact.get("status") in {"MISSING", "UNREADABLE", "INVALID"}:
            issues.append(f"missing or invalid artifact: {name}")
        _blind(name, artifact, issues)
    index = artifacts["policy_rollout_index.json"]
    oracle = artifacts["policy_oracle_689.json"]
    audit = artifacts["selector_feature_audit.json"]
    if index.get("status") == "PASS":
        if index.get("episode_count_expected") != 689 or index.get("episode_count_merged") != 689 or index.get("policy_row_count_merged") != 2067:
            issues.append("policy rollout index counts are not 689/689/2067")
        if index.get("duplicate_episode_count") != 0 or index.get("missing_episode_count") != 0 or index.get("issues"):
            issues.append("policy rollout index has duplicates, missing episodes, or issues")
    if oracle.get("status") not in {"PASS", "FAIL"}:
        issues.append("policy_oracle_689 has no scientific status")
    expected_feature_dimension = len(FEATURE_NAMES)
    if audit.get("status") == "PASS" and (audit.get("episode_count") != 689 or audit.get("policy_row_count") != 2067 or audit.get("feature_dimension") != expected_feature_dimension):
        issues.append("selector feature audit counts/dimension are wrong")
    if audit.get("status") == "PASS":
        identity_count = audit.get("identity_features_available_episode_count")
        if identity_count is None:
            identity_count = (audit.get("feature_availability") or {}).get("identity_features_available_episode_count")
        if identity_count == 0 and audit.get("identity_aware_learning_valid") is not False:
            issues.append("zero identity-feature coverage is not explicitly marked as non-identity-aware")
    full_loop = artifacts["full_loop_results.json"]
    valid_full_loop_status = full_loop.get("status", "").startswith("NOT_RUN") or full_loop.get("status") in {"PASS", "FAIL"}
    if not valid_full_loop_status:
        issues.append("full-loop artifact has an unknown status")
    fallback = artifacts["association_fallback_results.json"]
    if fallback.get("status") not in {"PASS", "DIAGNOSTIC_ONLY"}:
        issues.append("association fallback did not execute")
    learn = artifacts["learn_gate.json"]
    route = "selector_full_loop" if learn.get("status") == "PASS" and full_loop.get("status") == "PASS" else "association_fallback"
    result = {
        "protocol": "N32-ARTIFACT-VALIDATION",
        "status": "PASS" if not issues else "FAIL",
        "issues": issues,
        "artifact_statuses": {name: artifact.get("status") for name, artifact in artifacts.items()},
        "oracle_status": oracle.get("status"),
        "learn_gate_status": learn.get("status"),
        "full_loop_status": full_loop.get("status"),
        "route": route,
        "required_counts": {"episodes": 689, "policy_rows": 2067, "features": expected_feature_dimension},
        "identity_features_available_episode_count": audit.get("identity_features_available_episode_count", (audit.get("feature_availability") or {}).get("identity_features_available_episode_count")),
        "identity_aware_learning_valid": audit.get("identity_aware_learning_valid"),
        "selector_scope": audit.get("selector_scope"),
        "val25_read": False,
        "test_labels_used": False,
        "future_gt_used_for_selector_input": False,
    }
    _write(output, result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=OUT_DIR / "artifact_validation.json")
    args = parser.parse_args()
    result = run(output=args.output)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
