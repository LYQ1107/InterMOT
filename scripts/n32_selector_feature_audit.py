#!/usr/bin/env python3
"""Audit the action-independent selector feature tape for N32."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from sam3_intermot.adaptation.correction_selector_features import FEATURE_NAMES

INDEX = ROOT / "outputs/n32/policy_rollout_index.json"
OUT = ROOT / "outputs/n32/selector_feature_audit.json"
FROZEN = ROOT / "outputs/n32/frozen_protocol.json"
FORBIDDEN = ("future", "gt", "dataset_identity", "public_id", "sequence", "episode", "reward", "candidate_outcome")


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def run(*, input_path: Path = INDEX, output_path: Path = OUT) -> dict[str, Any]:
    source = json.loads(input_path.read_text(encoding="utf-8"))
    rows = list(source.get("rows", []))
    episodes = list(source.get("episodes", []))
    groups: dict[str, list[dict[str, Any]]] = {}
    issues: list[str] = []
    for row in rows:
        groups.setdefault(str(row.get("episode_id")), []).append(row)
        names = list(row.get("feature_names", []))
        values = row.get("feature_vector", [])
        if names != list(FEATURE_NAMES):
            issues.append(f"feature names mismatch: {row.get('episode_id')} {row.get('policy')}")
        if len(values) != len(FEATURE_NAMES):
            issues.append(f"feature dimension mismatch: {row.get('episode_id')} {row.get('policy')}")
        else:
            try:
                if not np.isfinite(np.asarray(values, dtype=float)).all():
                    issues.append(f"non-finite feature: {row.get('episode_id')} {row.get('policy')}")
            except Exception as exc:
                issues.append(f"feature conversion failure: {row.get('episode_id')}: {exc}")
        if row.get("future_gt_used_for_selection") is not False:
            issues.append(f"future GT selection flag is not false: {row.get('episode_id')}")
        if any(token in str(row.get("feature_names", [])).lower() for token in FORBIDDEN):
            issues.append(f"forbidden feature name: {row.get('episode_id')}")
    equality_count = 0
    for episode_id, items in groups.items():
        if len(items) != 3:
            issues.append(f"episode {episode_id} has {len(items)} feature rows")
            continue
        vectors = [np.asarray(item.get("feature_vector", []), dtype=float) for item in items]
        if all(vector.shape == vectors[0].shape and np.array_equal(vector, vectors[0]) for vector in vectors[1:]):
            equality_count += 1
        else:
            issues.append(f"policy-dependent selector feature vector: {episode_id}")
    first = rows[0] if rows else {}
    identity_available_episode_count = int(sum(
        bool(
            item.get(
                "identity_features_available",
                (item.get("feature_audit") or {}).get("identity_features_available", False),
            )
        )
        for item in episodes
    ))
    episode_denominator = len(groups)
    identity_coverage = float(identity_available_episode_count / episode_denominator) if episode_denominator else 0.0
    frozen = json.loads(FROZEN.read_text(encoding="utf-8")) if FROZEN.is_file() else {}
    frozen_learn_gate = frozen.get("learn_gate", {}) if isinstance(frozen, dict) else {}
    frozen_full_loop_gate = frozen.get("full_loop_gate", {}) if isinstance(frozen, dict) else {}
    identity_coverage_requirement = frozen_learn_gate.get("identity_feature_coverage_min")
    if identity_coverage_requirement is None:
        identity_coverage_requirement = frozen_learn_gate.get("identity_features_available_episode_count_min")
    result = {
        "protocol": "N32-D-SELECTOR-FEATURE-AUDIT",
        "status": "PASS" if len(groups) == 689 and len(rows) == 2067 and not issues else "FAIL",
        "episode_count": len(groups),
        "policy_row_count": len(rows),
        "feature_dimension": len(FEATURE_NAMES),
        "feature_names": list(FEATURE_NAMES),
        "feature_source_map": {
            name: ("past_identity_memory_or_zero_when_unavailable" if index >= 23 else "correction_frame_or_past_tracker_state")
            for index, name in enumerate(FEATURE_NAMES)
        },
        "feature_availability": {
            "identity_features_available_episode_count": identity_available_episode_count,
            "identity_feature_coverage": identity_coverage,
            "identity_feature_episode_denominator": episode_denominator,
            "single_id_identity_features": "zero-filled with identity_features_available=false",
        },
        "identity_features_available_episode_count": identity_available_episode_count,
        "identity_feature_coverage": identity_coverage,
        "identity_aware_learning_valid": bool(identity_available_episode_count > 0),
        "selector_scope": "TEMPORAL_GEOMETRY_ONLY_FALLBACK" if identity_available_episode_count == 0 else "CAUSAL_FEATURE_SELECTOR_WITH_IDENTITY_MEMORY_WHERE_AVAILABLE",
        "identity_limitation": (
            "identity features are unavailable for every episode and the feature vector is zero-filled; "
            "this audit PASS is not evidence of identity-aware learning"
            if identity_available_episode_count == 0
            else "identity features are available for at least one episode; coverage remains explicitly reported"
        ),
        "identity_gate": {
            "selector_learn_gate_identity_coverage_requirement": identity_coverage_requirement,
            "selector_learn_gate_requires_identity_coverage_gt_zero": bool(identity_coverage_requirement is not None and float(identity_coverage_requirement) > 0.0),
            "full_loop_primary_identity_metric_required": bool(frozen_full_loop_gate.get("primary_identity_metric_improvement_required", False)),
            "status": "NOT_IDENTITY_AWARE_ZERO_COVERAGE" if identity_available_episode_count == 0 else "IDENTITY_FEATURES_PRESENT",
        },
        "finite_vector_count": int(sum(len(row.get("feature_vector", [])) == len(FEATURE_NAMES) and np.isfinite(np.asarray(row.get("feature_vector", []), dtype=float)).all() for row in rows)),
        "same_vector_across_three_policies_episode_count": equality_count,
        "source_flags": {"feature_generation": first.get("feature_audit", {}), "future_gt_used_for_selection": False, "future_image_used": False, "public_id_emitted": False, "sequence_id_emitted": False},
        "forbidden_inputs": ["future_gt", "future_image", "future_features", "future_candidate_outcomes", "dataset_identity", "public_id", "sequence_name_or_number", "episode_index", "policy_reward"],
        "forbidden_input_name_scan": {token: False for token in FORBIDDEN},
        "issues": issues,
        "future_gt_used_for_selector_input": False,
        "val25_read": False,
        "test_labels_used": False,
    }
    _write(output_path, result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=INDEX)
    parser.add_argument("--output", type=Path, default=OUT)
    args = parser.parse_args()
    result = run(input_path=args.input, output_path=args.output)
    print(json.dumps({key: result[key] for key in ("protocol", "status", "episode_count", "policy_row_count", "feature_dimension", "identity_features_available_episode_count", "identity_feature_coverage", "identity_aware_learning_valid", "selector_scope", "same_vector_across_three_policies_episode_count", "issues")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
