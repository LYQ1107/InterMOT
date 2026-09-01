#!/usr/bin/env python3
"""Post-replay targeted regressions for N43 audit risks."""

from __future__ import annotations

import json
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.n43_full_matrix_common import FEATURE_NAMES, HARD_NEGATIVE, cell_features


OUT = ROOT / "outputs/n43/audit/targeted_regression.json"


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def audit(base: list[list[float]]) -> dict[str, Any]:
    matrix = np.asarray(base, dtype=float)
    candidates = [{"index": i, "native_tid": i + 1, "box": [10 + i * 20, 10, 30 + i * 20, 40], "confidence": 0.9, "feature_available": True, "native_age": 4} for i in range(matrix.shape[0])]
    pids = [100 + i for i in range(matrix.shape[1])]
    return {"frame": 4, "candidates": candidates, "public_id_order": pids, "public_id_to_native_tid": {str(pid): (i + 1 if i < len(candidates) else None) for i, pid in enumerate(pids)}, "base_scores_before_appearance": matrix.tolist(), "appearance_memory_scores": np.zeros_like(matrix).tolist(), "appearance_score_deltas": np.zeros_like(matrix).tolist(), "fused_scores": matrix.tolist()}


def main() -> None:
    started = now()
    cases = []
    # Valid columns are [1, 3]; current column 3 must compare to column 1,
    # not to a compressed-array position or to the same numeric value.
    first = audit([[HARD_NEGATIVE, 2.0, HARD_NEGATIVE, 2.0, HARD_NEGATIVE]])
    value = float(cell_features(first, 0, 3, 1)[FEATURE_NAMES.index("cell_margin_tanh")])
    expected = float(np.tanh((2.0 - 2.0) / 5.0))
    cases.append({"name": "duplicate_valid_scores_with_hard_columns", "observed_margin_feature": value, "expected_margin_feature": expected, "pass": abs(value - expected) < 1e-7})
    # Current column 4 is valid while only column 1 is another valid column;
    # this exercises explicit index exclusion with non-adjacent columns.
    second = audit([[HARD_NEGATIVE, 1.0, HARD_NEGATIVE, HARD_NEGATIVE, 3.0]])
    value2 = float(cell_features(second, 0, 4, 1)[FEATURE_NAMES.index("cell_margin_tanh")])
    expected2 = float(np.tanh((3.0 - 1.0) / 5.0))
    cases.append({"name": "nonadjacent_valid_column_exclusion", "observed_margin_feature": value2, "expected_margin_feature": expected2, "pass": abs(value2 - expected2) < 1e-7})
    # A hard-negative current cell must remain the fixed negative margin and
    # must not throw even when all other columns are hard negatives.
    third = audit([[HARD_NEGATIVE, HARD_NEGATIVE, HARD_NEGATIVE, HARD_NEGATIVE, HARD_NEGATIVE]])
    value3 = float(cell_features(third, 0, 2, 1)[FEATURE_NAMES.index("cell_margin_tanh")])
    expected3 = float(np.tanh(-10.0 / 5.0))
    cases.append({"name": "all_hard_negative_current_column", "observed_margin_feature": value3, "expected_margin_feature": expected3, "pass": abs(value3 - expected3) < 1e-7})
    result = {"protocol": "N43_POST_REPLAY_TARGETED_REGRESSION_V1", "status": "PASS" if all(x["pass"] for x in cases) else "FAIL", "started_at": started, "finished_at": now(), "cases": cases, "minimal_fix": "use valid_columns = flatnonzero(row_values > HARD_NEGATIVE), then other_columns = valid_columns[valid_columns != column]", "failure_preserved": True, "runtime_future_gt_used": False}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    if result["status"] != "PASS":
        raise RuntimeError(json.dumps(result))
    print(json.dumps({"status": result["status"], "output": str(OUT)}, sort_keys=True))


if __name__ == "__main__":
    try:
        main()
    except Exception:
        failure = ROOT / "outputs/n43/attempts" / f"targeted_regression_failure_{datetime.now().strftime('%Y%m%dT%H%M%S')}.json"
        failure.parent.mkdir(parents=True, exist_ok=True)
        failure.write_text(json.dumps({"status": "FAIL", "traceback": traceback.format_exc(), "failure_preserved": True}, indent=2) + "\n", encoding="utf-8")
        raise
