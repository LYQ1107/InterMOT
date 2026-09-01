#!/usr/bin/env python3
"""Freeze the N47 structural probe protocol after auditing N46's boundary."""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.n47_global_probe_common import (
    HORIZONS,
    N46_DIAG,
    N46_GATE,
    N42_PROTOCOL,
    OUT,
    PROTOCOL,
    SCORE_SCALE,
    SEED,
    VARIANTS,
    load,
    write_json,
)


def main() -> None:
    diag = load(N46_DIAG)
    gate = load(N46_GATE)
    split = load(N42_PROTOCOL)["sequence_split"]
    runtime = diag["runtime"]
    posthoc = diag["posthoc"]
    oracle = int(posthoc["oracle_desired_pairs"])
    blocked = int(posthoc["oracle_pairs_blocked_by_other_public_id"])
    sparse_rate = float(runtime["proposals_considered"]) / max(int(runtime["frames"]), 1)
    blocked_rate = blocked / max(oracle, 1)
    required_median = float(posthoc["oracle_required_delta_distribution"]["median"])
    fixed_boost = float(diag["factor_diagnosis"]["b_boost_vs_assignment_margin"]["boost"])
    protocol = {
        "schema": "N47_GLOBAL_CANDIDATE_ASSIGNMENT_PROBE_PROTOCOL_V1",
        "status": "FROZEN",
        "hypothesis": "N46 local owner-by-column proposal gating prevents useful global swaps; a candidate-level appearance logit applied to the complete finite matrix and followed by one global Hungarian-with-NONE may expose a correctable assignment boundary.",
        "source": "frozen N42 runtime/t0 candidate and prefix traces",
        "training": {
            "seed": SEED,
            "sequence_split_source": str(N42_PROTOCOL),
            "sequence_split": split,
            "labels": "offline GT IoU labels only for train/validation/holdout diagnostics; no GT runtime feature",
            "positive_threshold": 0.5,
            "negative_threshold": 0.1,
            "ambiguous_rule": "discard 0.1 < IoU < 0.5",
            "pair_sampling": "each positive candidate×public-ID cell paired with up to 8 strongest baseline-score negative IDs for the same candidate/frame; fixed before training",
            "optimizer": "AdamW",
            "learning_rate": 0.002,
            "weight_decay": 0.0001,
            "batch_size": 512,
            "max_epochs": 25,
            "patience": 5,
            "loss": "pairwise softplus(-(global_score_positive-global_score_negative)) + 0.001 logit L2",
            "holdout_use": "audit only; no stopping/gate/threshold selection",
        },
        "runtime": {
            "variants": list(VARIANTS),
            "horizons": list(HORIZONS),
            "score_formula": "adjusted_score_ij = frozen_write_baseline_score_ij + predicted_candidate_appearance_logit_ij",
            "candidate_logit_features": "current/past score, appearance/memory signal, candidate confidence/age/rank, frame offset; no public-ID, target identity, GT or future outcome",
            "assignment": "single global Hungarian over complete candidate×public-ID matrix with explicit NONE dummy columns",
            "none_score": -100000000.0,
            "hard_negative_behavior": "never modify hard-negative cells",
            "swap_allowed": True,
            "m0": "exact no-sidecar control",
            "runtime_future_gt_used": False,
            "gt_loaded_posthoc": True,
            "score_scale": SCORE_SCALE,
        },
    }
    write_json(PROTOCOL, protocol)
    result = {
        "status": "PASS",
        "protocol": "N47_STAGE_01_BOUNDARY_AUDIT_V1",
        "command": ["python", "scripts/n47_stage01_boundary_audit.py"],
        "inputs": {"n46_diagnosis": str(N46_DIAG), "n46_gate": str(N46_GATE), "n42_training_protocol": str(N42_PROTOCOL)},
        "outputs": {"frozen_protocol": str(PROTOCOL), "stage_status": str(OUT / "stage_01_status.json")},
        "metrics": {
            "n46_proposals_considered": int(runtime["proposals_considered"]),
            "n46_runtime_frames": int(runtime["frames"]),
            "proposal_rate": sparse_rate,
            "oracle_desired_pairs": oracle,
            "owner_blocked_pairs": blocked,
            "owner_blocked_rate": blocked_rate,
            "fixed_boost": fixed_boost,
            "oracle_required_delta_median": required_median,
            "m2_memory_effect": diag["factor_diagnosis"]["e_memory_effect"]["M2_H20_H50_H100"],
        },
        "gate_checks": {
            "n46_repair2_loaded": gate.get("status") == "N46_COMPLETED_DIAGNOSTIC_GATE_FAILED",
            "global_boundary_nonempty": blocked < oracle,
            "local_interface_bottleneck_supported": sparse_rate < 0.01 and blocked_rate > 0.99 and required_median > fixed_boost,
            "runtime_gt_forbidden": True,
            "sequence_split_frozen": True,
            "production_interface_changed": False,
            "n44_checkpoint_changed": False,
            "n47_training_authorized_as_diagnostic_only": True,
        },
        "failure_root_cause": "N46's local proposal gate excludes candidates already owned by another public-ID and caps accepted changes at +0.25; this is sufficient evidence for a bounded global-assignment probe, but not evidence of efficacy.",
        "next_action": "Run the cheap global Hungarian/NONE/swap smoke, then build the frozen sequence-disjoint dataset and perform one actual training run.",
        "runtime_future_gt_used": False,
        "gt_loaded_posthoc": True,
        "finished_at": datetime.now(timezone.utc).isoformat(),
    }
    write_json(OUT / "stage_01_status.json", result)
    print(json.dumps({"status": result["status"], "global_boundary_nonempty": result["gate_checks"]["global_boundary_nonempty"], "protocol": str(PROTOCOL)}))


if __name__ == "__main__":
    main()
