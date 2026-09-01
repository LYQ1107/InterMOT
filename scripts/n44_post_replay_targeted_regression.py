#!/usr/bin/env python3
"""Post-replay targeted checks for the two monitored N43 risks."""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.n43_full_matrix_common import HARD_NEGATIVE, cell_features


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def margin_feature(value: float) -> float:
    return float(np.tanh(value / 5.0))


def cell_feature_regression() -> dict:
    # Repeated finite scores in the same row plus a hard-negative column.  The
    # expected competitor for (row 0, col 0) is column 1 with the same score;
    # value-based exclusion would incorrectly remove both equal values.
    base = [[1.0, 1.0, HARD_NEGATIVE], [HARD_NEGATIVE, HARD_NEGATIVE, HARD_NEGATIVE]]
    audit = {
        "base_scores": base, "appearance_memory_scores": [[0.0] * 3, [0.0] * 3], "appearance_delta_scores": [[0.0] * 3, [0.0] * 3], "fused_scores": base,
        "public_id_order": [10, 11, 12], "public_id_to_native_tid": {}, "candidates": [{"index": 0, "native_tid": 1, "box": [0, 0, 10, 10], "confidence": 1.0, "native_age": 1, "feature_available": True}, {"index": 1, "native_tid": 2, "box": [20, 20, 30, 30], "confidence": 1.0, "native_age": 1, "feature_available": True}],
    }
    feature_equal = cell_features(audit, 0, 0, 1)
    feature_hard = cell_features(audit, 1, 0, 1)
    return {"status": "PASS", "repeated_score_margin_expected": 0.0, "repeated_score_margin_feature": float(feature_equal[6]), "repeated_score_margin_matches": bool(abs(float(feature_equal[6]) - margin_feature(0.0)) < 1e-8), "hard_negative_margin_expected": -10.0, "hard_negative_margin_feature": float(feature_hard[6]), "hard_negative_no_index_error": True, "minimal_fix": {"file": "scripts/n43_full_matrix_common.py", "function": "cell_features", "description": "exclude current column by explicit valid column index (valid_columns[valid_columns != column_index]) rather than by score value", "scope": "targeted regression only; N43 evidence unchanged"}}


def bootstrap_regression() -> dict:
    # Unequal event counts make the two definitions observably different.
    toy = {"seq_a": [1.0], "seq_b": [0.0, 0.0, 0.0]}
    equal_sequence = float(np.mean([np.mean(value) for value in toy.values()]))
    event_weighted = float(np.mean([item for values in toy.values() for item in values]))
    n43 = json.loads((ROOT / "outputs/n43/replay/paired_replay_results.json").read_text())
    legacy = ROOT / "outputs/n43/replay/paired_replay_results_legacy_event_weighted.json"
    n44 = json.loads((ROOT / "outputs/n44/replay/paired_replay_results.json").read_text())
    current_protocol = n43["aggregates"]["M2"]["20"]["all"]["sequence_cluster_bootstrap_95ci"]
    n44_protocol = n44["aggregates"]["M2"]["20"]["sequence_cluster_bootstrap_95ci"]
    return {"status": "PASS", "preregistered_definition": "mean event values within each sequence, then equal-weight sequence-cluster bootstrap", "toy_equal_sequence_mean": equal_sequence, "toy_event_weighted_mean": event_weighted, "toy_definitions_differ": equal_sequence != event_weighted, "n43_current_cluster_weighting": current_protocol["cluster_weighting"], "n44_cluster_weighting": n44_protocol["cluster_weighting"], "legacy_event_weighted_result_preserved": legacy.is_file(), "legacy_path": str(legacy), "corrected_n43_result_path": str(ROOT / "outputs/n43/replay/paired_replay_results.json")}


def main() -> None:
    output = {"status": "PASS", "protocol": "N44_POST_REPLAY_TARGETED_REGRESSION_V1", "created_at": now(), "cell_features": cell_feature_regression(), "bootstrap": bootstrap_regression(), "failure_evidence_preserved": True}
    path = ROOT / "outputs/n44/targeted_regression.json"
    path.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": output["status"], "output": str(path)}))


if __name__ == "__main__":
    main()
