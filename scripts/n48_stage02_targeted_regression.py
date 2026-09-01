#!/usr/bin/env python3
"""Targeted N48 contract regression after the global-margin gate repair."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.n48_assignment_common import (  # noqa: E402
    HARD_NEGATIVE,
    RiskAware512FusionHead,
    global_assignment_margin,
    runtime_sidecar,
    solve_with_none,
)


def main() -> None:
    base = np.asarray([[10.0, 9.0], [9.0, 10.0]], dtype=np.float32)
    margin = global_assignment_margin(base)
    model = RiskAware512FusionHead()
    candidate = np.zeros((4, 512), dtype=np.float32)
    memories = np.zeros((4, 512), dtype=np.float32)
    scalars = np.zeros((4, 8), dtype=np.float32)
    hard_base = np.asarray([[HARD_NEGATIVE, 1.0]], dtype=np.float32)
    hard_probe = runtime_sidecar(model, candidate[:2], memories[:2], scalars[:2], hard_base, np.asarray([True, True], dtype=bool))
    checks = {
        "global_margin_two_by_two_is_exact": bool(abs(margin - 2.0) <= 1e-6),
        "global_solver_explicit_none": bool(solve_with_none(np.asarray([[HARD_NEGATIVE]], dtype=np.float32))[0] == -1),
        "bounded_residual_leq_0_25": bool(np.max(np.abs(np.asarray(hard_probe["bounded_residual"]))) <= 0.25 + 1e-7),
        "hard_negative_unchanged": bool(np.asarray(hard_probe["adjusted_scores"])[0, 0] == HARD_NEGATIVE),
        "runtime_future_gt_false": hard_probe["runtime_future_gt_used"] is False,
    }
    payload = {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "protocol": "N48_STAGE_02_GLOBAL_MARGIN_TARGETED_REGRESSION_V1",
        "command": ["python", "scripts/n48_stage02_targeted_regression.py"],
        "inputs": {"source_contract_failure": "outputs/n48/attempts/runtime_sidecar_local_margin_contract_failure.json"},
        "outputs": {},
        "metrics": {"two_by_two_global_margin": margin},
        "gate_checks": checks,
        "failure_root_cause": "Fixed runtime gate now uses the exact whole-assignment gap with explicit NONE." if all(checks.values()) else "Global-margin gate repair regression failed.",
        "next_action": "Proceed to N48 runtime only if all targeted checks pass." if all(checks.values()) else "Preserve failure and do not start replay.",
        "runtime_future_gt_used": False,
    }
    path = ROOT / "outputs/n48/stage_02_targeted_regression.json"
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": payload["status"], "global_margin": margin}))
    if payload["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
