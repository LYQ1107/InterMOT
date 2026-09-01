#!/usr/bin/env python3
"""Apply the N34 authorization gate before any calibration or LoRA work."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "n34"


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, path)
    finally:
        if os.path.exists(tmp_name):
            os.unlink(tmp_name)


def _load(name: str) -> dict[str, Any]:
    path = OUT / name
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    return value if isinstance(value, dict) else {}


def run() -> dict[str, Any]:
    tape_manifest = _load("tape_manifest.json")
    full_loop = _load("full_loop_transaction_results.json")
    replay = _load("ccam_paired_replay_results.json")
    audit = _load("audit_before_run.json")
    selected = _load("selected_sequences.json")
    identity_constraint = audit.get("identity_feature_constraint", {})

    checks = {
        "real_candidate_complete_tape": bool(tape_manifest.get("candidate_complete", False)),
        "real_full_loop_pass": full_loop.get("real_data_status") == "PASS",
        "real_ccam_future_effect_computable": replay.get("identity_effect") not in {
            None,
            "NOT_COMPUTABLE_REAL_DATA_TAPE_UNAVAILABLE",
        },
        "m2_m3_m4_sequence_cluster_ci_lower_gt_zero": False,
        "at_least_two_independent_sequences_benefit": False,
        "no_untouched_identity_regression_verified": False,
        "identity_feature_coverage_positive": int(identity_constraint.get("n32_selector_identity_features_available_episode_count", 0) or 0) > 0,
    }
    authorized = bool(all(checks.values()))
    gate = {
        "protocol": "N34_CALIBRATION_AND_DECODER_LORA_AUTHORIZATION_GATE",
        "status": "AUTHORIZED" if authorized else "NOT_AUTHORIZED",
        "checks": checks,
        "real_sequence_count": int(selected.get("sequence_count", 0) or 0),
        "real_candidate_complete": bool(tape_manifest.get("candidate_complete", False)),
        "real_future_effect": replay.get("identity_effect", "NOT_COMPUTABLE"),
        "synthetic_score_delta_is_not_gate_evidence": True,
        "calibration_head": {
            "status": "AUTHORIZED" if authorized else "NOT_AUTHORIZED",
            "architecture": "LayerNorm(d) -> Linear(d,64) -> GELU -> Linear(64,4)",
            "outputs": ["M0_weight", "M1_weight", "M2_weight", "gate"],
            "sequence_split": "sequence-disjoint train/select/cal",
            "seeds": [3401, 3402, 3403],
            "loso": True,
            "training_started": False,
            "reason": (
                "real CCAM future-effect gate is not computable and identity feature coverage is zero"
                if not authorized
                else "all N34 gates passed"
            ),
        },
        "decoder_lora": {
            "status": "AUTHORIZED" if authorized else "NOT_AUTHORIZED",
            "training_started": False,
            "four_card_shard_plan": "not_started",
            "reason": "Calibration authorization is a prerequisite" if not authorized else "calibration authorized",
        },
        "fallback_decision": {
            "temporal_geometry_only_fallback_allowed_by_frozen_n32_protocol": True,
            "identity_aware_selector_authorized": False,
            "deployment_route": "association_fallback",
            "interpretation": "fallback remains explicitly non-identity-aware; zero-filled identity fields are not evidence",
        },
        "inputs": {
            "tape_manifest": "outputs/n34/tape_manifest.json",
            "full_loop": "outputs/n34/full_loop_transaction_results.json",
            "replay": "outputs/n34/ccam_paired_replay_results.json",
            "n32_selector_feature_audit": "outputs/n32/selector_feature_audit_attempt2.json",
        },
        "commands_not_started": [
            "scripts/n34_train_calibration_head.py",
            "scripts/n34_train_decoder_lora.py",
        ],
    }
    atomic_json(OUT / "calibration_gate.json", gate)
    atomic_json(OUT / "selector_fallback_decision.json", gate["fallback_decision"])
    stage = {
        "stage": "N34-5",
        "status": "NOT_AUTHORIZED" if not authorized else "PASS",
        "commands": ["python scripts/run_n34_calibration_gate.py"],
        "artifacts": [
            "outputs/n34/calibration_gate.json",
            "outputs/n34/selector_fallback_decision.json",
        ],
        "errors": [],
        "calibration_head": gate["calibration_head"]["status"],
        "decoder_lora": gate["decoder_lora"]["status"],
        "next_action": "Acquire/materialize a real per-frame candidate-complete public-ID tape before any learned selector or LoRA training.",
    }
    atomic_json(OUT / "stage_05_status.json", stage)
    return gate


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.parse_args()
    result = run()
    print(json.dumps({"status": result["status"], "calibration_head": result["calibration_head"]["status"], "decoder_lora": result["decoder_lora"]["status"], "output": "outputs/n34/calibration_gate.json"}, sort_keys=True))


if __name__ == "__main__":
    main()
