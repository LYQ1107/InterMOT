#!/usr/bin/env python3
"""Reload and audit the repair2 checkpoint without running evaluation selection."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.n48_assignment_common import RiskAware512FusionHead  # noqa: E402

R2 = ROOT / "outputs/n48/repair1b"
CHECKPOINT = R2 / "training/n48_r1_repair2_risk_aware_512d_bce.pt"
MANIFEST = R2 / "training/training_manifest.json"
AMENDMENT = R2 / "protocol_amendment_repair2.json"


def main() -> None:
    payload = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    model = RiskAware512FusionHead(int(payload.get("projection_dim", 64)))
    model.load_state_dict(payload["state_dict"]); model.eval()
    candidate = torch.zeros((2, 512), dtype=torch.float32); memory = torch.zeros((2, 512), dtype=torch.float32); scalar = torch.zeros((2, 8), dtype=torch.float32)
    with torch.no_grad(): raw, uncertainty = model(candidate, memory, scalar)
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8")); amendment = json.loads(AMENDMENT.read_text(encoding="utf-8"))
    history = manifest.get("loss_history", [])
    checks = {
        "checkpoint_reloadable": True,
        "checkpoint_hash_recorded": hashlib.sha256(CHECKPOINT.read_bytes()).hexdigest() == manifest["checkpoint_sha256"],
        "protocol_repair2": payload.get("protocol") == "N48_R1_REPAIR2_SINGLE_OBJECTIVE_V1",
        "actual_full_training": payload.get("actual_full_training") is True,
        "epoch_count_8": payload.get("epoch_count") == 8 and len(history) == 8,
        "all_loss_terms_logged": all(all(key in row["train"] and key in row["validation"] for key in ("rank_loss", "cell_bce", "uncertainty_bce", "residual_l2", "total_objective")) for row in history),
        "one_optimizer_step_each_epoch": all(row["train_accumulated"]["optimizer_steps"] == 1 for row in history),
        "finite_reload_output": bool(torch.isfinite(raw).all() and torch.isfinite(uncertainty).all()),
        "amendment_frozen": amendment.get("status") == "FROZEN_BEFORE_RETRAINING",
        "production_authorized_false": payload.get("production_authorized") is False,
        "runtime_future_gt_false": amendment["runtime"]["runtime_future_gt_used"] is False,
    }
    result = {"status": "PASS" if all(checks.values()) else "FAIL", "protocol": "N48_R1_REPAIR2_STAGE_03_RELOAD_V1", "command": ["python", "scripts/n48_repair1b_stage03_reload.py"], "inputs": {"checkpoint": str(CHECKPOINT), "manifest": str(MANIFEST), "amendment": str(AMENDMENT)}, "outputs": {}, "metrics": {"checkpoint_sha256": manifest.get("checkpoint_sha256"), "epoch_count": len(history), "best_epoch": manifest.get("best_epoch"), "train_pair_count": manifest.get("train_pair_count"), "validation_pair_count": manifest.get("validation_pair_count"), "holdout_pair_count": manifest.get("holdout_pair_count")}, "gate_checks": checks, "failure_root_cause": "Reload confirms the repair2 checkpoint is the actual accumulated-objective diagnostic artifact; this is not an efficacy gate." if all(checks.values()) else "Checkpoint reload contract failed; preserve and repair before replay.", "next_action": "Run isolated full paired replay if PASS." if all(checks.values()) else "Preserve failure and repair first.", "runtime_future_gt_used": False}
    (R2 / "stage_03_reload.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
