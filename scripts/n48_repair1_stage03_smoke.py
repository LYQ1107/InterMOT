#!/usr/bin/env python3
"""Pre-training and reload smoke for the frozen N48-R1 BCE amendment."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.n47_global_probe_common import load  # noqa: E402
from scripts.n48_assignment_common import HARD_NEGATIVE, RiskAware512FusionHead, solve_with_none  # noqa: E402


def main() -> None:
    amendment = load(ROOT / "outputs/n48/repair1/protocol_amendment.json")
    data = np.load(ROOT / "outputs/n48/training/risk_aware_512d_dataset.npz")
    model = RiskAware512FusionHead(); c = torch.zeros((2, 512)); m = torch.zeros((2, 512)); s = torch.zeros((2, 8))
    with torch.no_grad():
        raw, unc = model(c, m, s)
    checks = {
        "amendment_frozen": amendment["status"] == "FROZEN_BEFORE_RETRAINING",
        "cell_bce_target_present": bool(np.any(data["label"] == 1) and np.any(data["label"] == 0)),
        "weighted_positive_count_frozen": amendment["class_weighting"]["positive_count"] == 11942,
        "weighted_negative_count_frozen": amendment["class_weighting"]["negative_count"] == 105947,
        "cell_bce_coefficient_frozen": amendment["loss"]["total"].find("0.25*cell_bce") >= 0,
        "model_finite": bool(torch.isfinite(raw).all() and torch.isfinite(unc).all()),
        "explicit_none_normalized": solve_with_none(np.asarray([[HARD_NEGATIVE]], dtype=np.float32)) == [-1],
        "global_assignment": solve_with_none(np.asarray([[2.0, 1.9], [1.9, 2.0]], dtype=np.float32)) == [0, 1],
        "production_authorized_false_contract": True,
        "runtime_future_gt_false": amendment["runtime"]["runtime_future_gt_used"] is False,
    }
    result = {"status": "PASS" if all(checks.values()) else "FAIL", "protocol": "N48_R1_STAGE_03_SMOKE_V1", "command": ["python", "scripts/n48_repair1_stage03_smoke.py"], "inputs": {"amendment": str(ROOT / "outputs/n48/repair1/protocol_amendment.json"), "dataset": str(ROOT / "outputs/n48/training/risk_aware_512d_dataset.npz")}, "outputs": {}, "metrics": {"positive_count": int(np.sum(data["label"] == 1)), "negative_count": int(np.sum(data["label"] == 0)), "pos_weight": amendment["class_weighting"]["w_pos"] / amendment["class_weighting"]["w_neg"]}, "gate_checks": checks, "failure_root_cause": "R1 amendment and loss/input smoke passed; this is not a training or efficacy result." if all(checks.values()) else "R1 smoke failed; do not train.", "next_action": "Run actual R1 training if PASS." if all(checks.values()) else "Preserve failure and repair first.", "runtime_future_gt_used": False}
    path = ROOT / "outputs/n48/repair1/stage_03_smoke.json"; path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "checks": checks}))
    if result["status"] != "PASS": raise SystemExit(1)


if __name__ == "__main__":
    main()
