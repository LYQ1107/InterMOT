#!/usr/bin/env python3
"""Post-training reload and loss-contract smoke for N48-R1."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.n48_assignment_common import RiskAware512FusionHead  # noqa: E402


def main() -> None:
    path = ROOT / "outputs/n48/repair1/training/n48_r1_risk_aware_512d_bce.pt"
    payload = torch.load(path, map_location="cpu", weights_only=False)
    history = payload.get("history", [])
    terms = all(all(key in epoch["train"] and key in epoch["validation"] for key in ("rank_loss", "cell_bce", "uncertainty_bce", "residual_l2", "total_objective")) for epoch in history)
    model = RiskAware512FusionHead(int(payload.get("projection_dim", 64))); model.load_state_dict(payload["state_dict"]); model.eval()
    with torch.no_grad():
        raw, uncertainty = model(torch.zeros((2, 512)), torch.zeros((2, 512)), torch.zeros((2, 8)))
    checks = {"checkpoint_reloadable": True, "production_authorized_false": payload.get("production_authorized") is False, "protocol_r1": payload.get("protocol") == "N48_R1_RISK_AWARE_512D_WITH_CELL_BCE_V1", "actual_full_training": payload.get("actual_full_training") is True, "epoch_count_8": payload.get("epoch_count") == 8 and len(history) == 8, "loss_terms_rank_bce_uncertainty_l2_logged": terms, "cell_bce_coefficient_0_25": abs(float(payload.get("cell_bce_weight")) - 0.25) < 1e-12, "fixed_seed_4848": payload.get("seed") == 4848, "finite_reload_output": bool(torch.isfinite(raw).all() and torch.isfinite(uncertainty).all())}
    result = {"status": "PASS" if all(checks.values()) else "FAIL", "protocol": "N48_R1_STAGE_03_RELOAD_V1", "command": ["python", "scripts/n48_repair1_stage03_reload.py"], "inputs": {"checkpoint": str(path)}, "outputs": {}, "metrics": {"checkpoint_sha256": __import__("hashlib").sha256(path.read_bytes()).hexdigest(), "best_epoch": payload.get("best_epoch"), "epoch_count": len(history)}, "gate_checks": checks, "failure_root_cause": "R1 checkpoint reload and decomposed loss contract passed; this is not an efficacy result." if all(checks.values()) else "R1 checkpoint reload/loss contract failed.", "next_action": "Run full R1 replay if PASS." if all(checks.values()) else "Preserve failure and do not replay.", "runtime_future_gt_used": False}
    out = ROOT / "outputs/n48/repair1/stage_03_reload.json"; out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "checks": checks}))
    if result["status"] != "PASS": raise SystemExit(1)


if __name__ == "__main__":
    main()
