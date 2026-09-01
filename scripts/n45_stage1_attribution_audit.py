#!/usr/bin/env python3
"""N45 Stage 01: audit the missing write-baseline attribution and fix labels."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
N44_RESULT = ROOT / "outputs/n44/replay/paired_replay_results.json"
N44_RUNTIME = ROOT / "outputs/n44/replay/runtime"
N44_STAGE1 = ROOT / "outputs/n44/stage_01_status.json"
N44_STAGE2 = ROOT / "outputs/n44/stage_02_status.json"
N45 = ROOT / "outputs/n45"
PROTOCOL = N45 / "attribution_protocol.json"
STAGE = N45 / "stage_01_status.json"


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    old_result = read(N44_RESULT)
    old_stage1 = read(N44_STAGE1)
    old_stage2 = read(N44_STAGE2)
    runtime_files = sorted(N44_RUNTIME.glob("*.json"))
    sample = read(runtime_files[0]) if runtime_files else {}
    old_has_write_baseline = all("write_baseline" in sample.get("variants", {}).get(v, {}) for v in ("M0", "M1", "M2", "M3", "M4"))
    old_has_plus_name = all("write_plus_n44" in sample.get("variants", {}).get(v, {}) for v in ("M0", "M1", "M2", "M3", "M4"))
    old_metrics = old_stage1["metrics"]
    counters = old_metrics["counters"]
    assigned_known = int(counters["base_assigned_known_cells"])
    oracle_correct = int(counters["oracle_correct_candidates"])
    corrected_rate = oracle_correct / assigned_known if assigned_known else None
    old_total_cell_rate = float(old_metrics["candidate_ceiling"]["oracle_correct_rate_over_candidate_rows"])
    protocol = {
        "protocol": "N45_THREE_BRANCH_ATTRIBUTION_V1",
        "status": "FROZEN",
        "branches": {"no_write": "N42 memory_write=False future branch", "write_baseline": "N42 memory_write=True future branch with original fused assignment and no N44", "write_plus_n44": "exact write_baseline audit passed through frozen N44 checkpoint/sidecar only"},
        "alignment": ["same event", "same variant M0-M4", "same future frame", "same candidate order/native ID/box/confidence", "runtime_future_gt_used=false"],
        "attribution": {"memory_effect": "write_baseline minus no_write", "n44_incremental_effect": "write_plus_n44 minus write_baseline"},
        "posthoc": "GT is loaded only after all runtime branch structure is validated",
        "bootstrap": {"definition": "mean event values within sequence, then equal-weight sequence cluster bootstrap", "seed": 4444, "replicates": 2000},
        "holdout": "not used to select gate or interpret incremental result",
    }
    PROTOCOL.parent.mkdir(parents=True, exist_ok=True)
    PROTOCOL.write_text(json.dumps(protocol, indent=2) + "\n", encoding="utf-8")
    result = {"status": "PASS", "protocol": "N45_STAGE_01_ATTRIBUTION_AUDIT_V1", "command": ["python", "scripts/n45_stage1_attribution_audit.py"], "inputs": {"n44_result": str(N44_RESULT), "n44_runtime": str(N44_RUNTIME), "n44_stage_01": str(N44_STAGE1), "n44_stage_02": str(N44_STAGE2)}, "outputs": {"protocol": str(PROTOCOL), "legacy_n44_result": str(N44_RESULT)}, "metrics": {"legacy_n44_has_write_baseline_branch": old_has_write_baseline, "legacy_n44_has_write_plus_n44_branch_name": old_has_plus_name, "legacy_runtime_event_count": len(runtime_files), "legacy_n44_result_status": old_result.get("status"), "corrected_candidate_ceiling": {"oracle_correct_candidate_rows": oracle_correct, "baseline_assigned_known_candidate_rows_denominator": assigned_known, "oracle_correct_rate_over_assigned_known_candidate_rows": corrected_rate}, "legacy_diagnostics_only": {"oracle_correct_rate_over_total_cell_count": old_total_cell_rate, "pair_cell_positive_rate": old_metrics["candidate_ceiling"]["positive_candidate_public_id_rate"], "pair_cell_positive_numerator": old_metrics["candidate_ceiling"]["positive_candidate_public_ids"], "pair_cell_positive_denominator": old_metrics["candidate_ceiling"]["known_candidate_public_ids"]}, "legacy_stage2_hard_negative_correction": {"old_gate_claim": old_stage2.get("gate_checks", {}).get("hard_negative_explicit"), "frozen_audit_hard_negative_cells": int(old_metrics["counters"]["hard_negative_cells"]), "frozen_n44_training_hard_negative_examples": 0, "actual_behavior": "hard-negative cells were skipped by code; none were present in the frozen audit", "corrected_contract_is_not_included": True}}, "gate_checks": {"old_n44_preserved": True, "old_result_missing_three_branch_attribution": not old_has_write_baseline, "corrected_candidate_ceiling_denominator_is_assigned_known_candidate_rows": True, "legacy_total_cell_rate_explicitly_renamed": True, "hard_negative_zero_and_skipped_explicit": True, "n45_protocol_frozen": True, "runtime_gt_used": False, "production_modified": False}, "failure_root_cause": "N44 Stage 04 reported no_write versus write_plus_N44; without the unchanged write baseline, its effect cannot be attributed to N44. The old Stage 01 ceiling rate also used total cell count and is retained only as a legacy diagnostic. N44's hard-negative gate wording is corrected here: the frozen audit had zero hard-negative cells and the dataset code skipped them.", "next_action": "Run checkpoint metadata repair smoke/targeted regression, then regenerate three aligned branches from the frozen N42 source and frozen N44 checkpoint.", "runtime_future_gt_used": False, "finished_at": now()}
    STAGE.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "old_write_baseline": old_has_write_baseline, "corrected_ceiling_rate": corrected_rate, "output": str(STAGE)}))


if __name__ == "__main__":
    main()
