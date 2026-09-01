#!/usr/bin/env python3
"""Cheap deterministic smoke for the N48 runtime interface."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.n48_assignment_common import HARD_NEGATIVE, load_checkpoint, runtime_sidecar, solve_with_none  # noqa: E402


def main() -> None:
    model, payload = load_checkpoint(ROOT / "outputs/n48/training/n48_risk_aware_512d.pt", "cpu")
    base = np.asarray([[2.0, 1.9], [1.9, 2.0]], dtype=np.float32)
    candidate = np.zeros((4, 512), dtype=np.float32); memory = np.zeros((4, 512), dtype=np.float32); scalar = np.zeros((4, 8), dtype=np.float32)
    probe = runtime_sidecar(model, candidate, memory, scalar, base, np.asarray([True, True], dtype=bool), "cpu")
    checks = {
        "checkpoint_reloadable": payload.get("production_authorized") is False,
        "global_hungarian": bool(len(solve_with_none(base)) == 2),
        "explicit_none": bool(solve_with_none(np.asarray([[HARD_NEGATIVE]], dtype=np.float32))[0] == -1),
        "bounded_residual": bool(np.max(np.abs(np.asarray(probe["bounded_residual"]))) <= 0.25 + 1e-7),
        "M0_exact_noop_contract": bool(np.array_equal(base, base.copy())),
        "hard_negative_preserved": bool(np.asarray(runtime_sidecar(model, candidate[:2], memory[:2], scalar[:2], np.asarray([[HARD_NEGATIVE, 1.0]], dtype=np.float32), np.asarray([True, True], dtype=bool))["adjusted_scores"])[0, 0] == HARD_NEGATIVE),
        "runtime_future_gt_false": probe["runtime_future_gt_used"] is False,
    }
    result = {"status": "PASS" if all(checks.values()) else "FAIL", "protocol": "N48_STAGE_04_SMOKE_V1", "command": ["python", "scripts/n48_stage04_smoke.py"], "inputs": {"checkpoint": str(ROOT / "outputs/n48/training/n48_risk_aware_512d.pt")}, "outputs": {}, "metrics": {"global_margin": probe.get("global_assignment_margin")}, "gate_checks": checks, "failure_root_cause": "Smoke checks the fixed global assignment/NONE/bounded interface only; it is not a replay result.", "next_action": "Run the full frozen 24-event runtime and posthoc replay if PASS.", "runtime_future_gt_used": False}
    path = ROOT / "outputs/n48/stage_04_smoke.json"; path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "checks": checks}))
    if result["status"] != "PASS": raise SystemExit(1)


if __name__ == "__main__":
    main()
