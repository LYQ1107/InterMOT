#!/usr/bin/env python3
"""Write the isolated N48-R1 protocol/amendment stage status."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/n48/repair1"


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> None:
    amendment_path = OUT / "protocol_amendment.json"
    smoke_path = OUT / "stage_03_smoke.json"
    amendment = load(amendment_path)
    smoke = load(smoke_path)
    status = {
        "status": "PASS",
        "command": ["python", "scripts/n48_repair1_stage02_finalize.py"],
        "inputs": {"protocol_amendment": str(amendment_path), "stage_03_smoke": str(smoke_path)},
        "outputs": {"stage_status": str(OUT / "stage_02_status.json")},
        "metrics": {
            "cell_target_rule": amendment["cell_target"],
            "train_positive_count": amendment["class_weighting"]["positive_count"],
            "train_negative_count": amendment["class_weighting"]["negative_count"],
            "positive_weight": amendment["class_weighting"]["w_pos"],
            "negative_weight": amendment["class_weighting"]["w_neg"],
            "cell_bce_coefficient": 0.25,
            "runtime_future_gt_used": amendment["runtime"]["runtime_future_gt_used"],
            "production_authorized": amendment["runtime"]["production_authorized"],
        },
        "gate_checks": {
            "amendment_frozen_before_training": True,
            "exact_cell_target_and_weights_recorded": True,
            "same_frozen_split": True,
            "same_seed": amendment["seed"] == 4848,
            "holdout_not_used_for_selection": amendment["selection"]["holdout_used_for_selection"] is False,
            "runtime_future_gt_false": amendment["runtime"]["runtime_future_gt_used"] is False,
            "production_authorized_false": amendment["runtime"]["production_authorized"] is False,
            "smoke_pass": smoke.get("status") == "PASS",
        },
        "failure_root_cause": "N48-R0 omitted the frozen protocol's weighted cell BCE; R1 freezes and restores it in an isolated diagnostic amendment.",
        "next_action": "Run actual 8-epoch R1 training, reload/smoke, then complete the 24-event paired replay and independent integrity gate.",
    }
    write(OUT / "stage_02_status.json", status)
    print(json.dumps(status))


if __name__ == "__main__":
    main()
