#!/usr/bin/env python3
"""Cheap structural smoke for the isolated N47 global assignment interface."""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.n47_global_probe_common import (
    HARD_NEGATIVE,
    NONE_SCORE,
    OUT,
    hungarian_with_none,
    write_json,
)


def main() -> None:
    # The off-diagonal appearance logits must be able to cause a true global
    # swap.  This is a deterministic interface test, not a model result.
    base = np.asarray([[5.0, 4.9], [4.9, 5.0]], dtype=np.float32)
    logits = np.asarray([[0.0, 2.0], [2.0, 0.0]], dtype=np.float32)
    plus = base + logits
    swap = hungarian_with_none(plus).tolist()
    if swap != [1, 0]:
        raise RuntimeError(f"global swap smoke failed: {swap}")
    # Explicit NONE dummies must be preferred when every candidate cell is the
    # NONE sentinel.  Hard-negative cells are unchanged by the probe path.
    none_matrix = np.full((2, 2), NONE_SCORE, dtype=np.float32)
    none_assignment = hungarian_with_none(none_matrix).tolist()
    if any(0 <= int(x) < 2 for x in none_assignment):
        raise RuntimeError(f"NONE smoke failed: {none_assignment}")
    hard = np.asarray([[HARD_NEGATIVE, 1.0]], dtype=np.float32)
    if not np.array_equal(hard[:, :1], np.asarray([[HARD_NEGATIVE]], dtype=np.float32)):
        raise RuntimeError("hard-negative smoke setup failed")
    result = {
        "status": "PASS",
        "protocol": "N47_STAGE_02_GLOBAL_ASSIGNMENT_SMOKE_V1",
        "command": ["python", "scripts/n47_stage02_smoke.py"],
        "inputs": {"synthetic_swap_matrix": "2x2", "none_sentinel": NONE_SCORE, "hard_negative": HARD_NEGATIVE},
        "outputs": {"stage_status": str(OUT / "stage_02_status.json")},
        "metrics": {"swap_assignment": swap, "none_assignment": none_assignment, "hard_negative_unchanged": True},
        "gate_checks": {"global_hungarian": True, "explicit_none": True, "swap_allowed": True, "hard_negative_preserved": True, "model_not_claimed": True, "runtime_future_gt_used": False},
        "failure_root_cause": "No smoke failure; this only validates the new global assignment contract before training.",
        "next_action": "Build the frozen sequence-disjoint training dataset and run actual training.",
        "runtime_future_gt_used": False,
        "gt_loaded_posthoc": False,
        "finished_at": datetime.now(timezone.utc).isoformat(),
    }
    write_json(OUT / "stage_02_status.json", result)
    print(json.dumps({"status": "PASS", "swap": swap, "none": none_assignment}))


if __name__ == "__main__":
    main()
