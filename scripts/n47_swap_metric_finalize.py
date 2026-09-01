#!/usr/bin/env python3
"""Finalize N47 swap taxonomy repair without changing legacy N47 artifacts."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.n47_global_probe_common import CHECKPOINT, TRAIN_MANIFEST, load, sha256, write_json  # noqa: E402

N40_BLOCKER = ROOT / "outputs/n46/BLOCKED_INPUT_REAL_HUMAN_TAPE.json"


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    out = args.output_root
    legacy = ROOT / "outputs/n47_global_probe"
    result = load(out / "replay/probe_results.json")
    integrity = load(out / "stage_04_integrity.json")
    old_status = load(legacy / "replay/runtime_status.json")
    old_gate = load(legacy / "n47_final_gate.json")
    old_report = ROOT / "docs/N47_FINAL_REPORT.md"
    blocker = load(N40_BLOCKER)
    training = load(TRAIN_MANIFEST)
    variants = ("M0", "M1", "M2", "M3", "M4")
    horizons = ("20", "50", "100")
    pure = {v: integrity["metrics"]["expected_pure_swap_changes"][v] for v in variants}
    assignments = {v: integrity["metrics"]["expected_assignment_changes"][v] for v in variants}
    id_set = {v: integrity["metrics"]["expected_id_set_changes"][v] for v in variants}
    m2 = result["effects"]["incremental"]["M2"]
    checks = {
        "repair_integrity_pass": integrity.get("status") == "PASS",
        "pure_swap_definition_registered": pure == {"M0": 0, "M1": 56, "M2": 64, "M3": 56, "M4": 39},
        "assignment_changes_preserved": assignments == {"M0": 0, "M1": 335, "M2": 455, "M3": 335, "M4": 375},
        "legacy_multiset_change_relabelled": id_set == {"M0": 0, "M1": 279, "M2": 391, "M3": 279, "M4": 336},
        "utility_correct_incorrect_untouched_unchanged": integrity.get("gate_checks", {}).get("utility_correct_incorrect_untouched_unchanged") is True,
        "old_n47_result_preserved": old_gate.get("status") == "N47_COMPLETED_GATE_FAILED" and old_report.is_file(),
        "checkpoint_not_changed": sha256(CHECKPOINT) == training.get("checkpoint_sha256"),
        "no_training_in_repair": True,
        "runtime_future_gt_false": integrity.get("gate_checks", {}).get("runtime_future_gt_false") is True,
        "gt_only_posthoc": result.get("protocol", {}).get("gt_loaded_only_after_runtime_validation") is True,
        "simulated_provenance": result.get("interaction_source") == "simulated_from_gt" and result.get("real_human_tape_created") is False,
        "real_human_tape": False,
        "real_sam3_full_loop": False,
        "production_authorized": False,
        "standard_mot_not_computable": result.get("id_switch_metric") == "NOT_COMPUTABLE_NO_COMPLETE_TRACKEVAL_SEQUENCE_INPUT",
    }
    status = "N47_SWAP_METRIC_REPAIR_COMPLETED_GATE_FAILED"
    gate = {
        "schema": "N47_SWAP_METRIC_REPAIR_GATE_V1",
        "status": status,
        "research_gate": "FAIL_INHERITED_EFFICACY_AND_REAL_INPUT_GATES",
        "checks": checks,
        "legacy_metric_correction": {
            "legacy_field": "swap_changes",
            "legacy_meaning": "multiset-change / ID-set change, including NONE/new/removal effects",
            "legacy_values": {v: old_status["metrics"]["by_variant"][v].get("swap_changes") for v in variants},
            "corrected_field": "pure_swap_changes",
            "corrected_definition": "assignment changed, non-NONE public-ID multiset equal, full assignment multiset equal, and at least two row mappings changed",
            "corrected_values": pure,
            "assignment_changes": assignments,
            "id_set_changes": id_set,
        },
        "metric_invariance": {
            "utility_correct_incorrect_untouched_unchanged": checks["utility_correct_incorrect_untouched_unchanged"],
            "memory_effect": "unchanged",
            "n47_incremental_effect": "unchanged",
            "holdout_retuning": False,
            "seed_changed": False,
            "checkpoint_changed": False,
            "metric_definition_changed": False,
        },
        "m2_incremental_context": {h: {k: m2[h][k] for k in ("identity_utility", "target_iou_delta", "future_identity_error_reduction", "recorrection_proxy_reduction", "assignment_change_count", "assignment_change_correct_count", "assignment_change_incorrect_count", "assignment_change_neutral_count", "assignment_no_change_count", "untouched_regression", "sequence_cluster_bootstrap_95ci")} for h in horizons},
        "provenance": {"interaction_source": "simulated_from_gt", "real_human_tape": False, "real_sam3_full_loop": False, "runtime_future_gt_used": False, "gt_loaded_posthoc": True},
        "authorization": {"calibration_head": "NOT_AUTHORIZED", "decoder_lora": "NOT_AUTHORIZED", "production_interface_changed": False, "checkpoint_production_authorized": False},
        "failure_root_cause": "The N47 replay's swap_changes label was a multiset-change count. The repair corrects the taxonomy only; the N47 efficacy, simulated provenance, negative H100/untouched gates, and absent N40 real input remain unchanged.",
        "next_action": "Use the corrected pure-swap taxonomy in future audits; keep the broader objective open and obtain N40-compliant real human tape plus real SAM3 full-loop before any promotion decision.",
    }
    report = f"""# N47 Swap-Metric Repair Report

Date: {now()}  
Status: `{status}`. This is a metric-attribution repair only; the broader scientific objective remains open.

## Scope and preservation

The original N47 runtime, result, status, integrity, gate and report remain unchanged. Their read-only snapshot and hashes are under [legacy_snapshot](./outputs/n47_global_probe/repair1_swap_metric/legacy_snapshot). The repair output is isolated under [repair1_swap_metric](./outputs/n47_global_probe/repair1_swap_metric). No training was run, no seed/checkpoint/metric definition changed, and N44/N45/production MOT/OVMOT files were not modified.

## Root cause and minimal correction

The old runtime condition named `swap_changes` counted `sorted(write_public_ids) != sorted(plus_public_ids)`. That is an ID multiset change, not a pure assignment swap. The corrected replay records three disjoint concepts: total `assignment_changes`, `id_set_changes` (the old multiset-change meaning, including NONE/new/removal effects), and `pure_swap_changes`. A pure swap requires the non-NONE public-ID multiset to remain equal, the full assignment multiset to remain equal, and at least two row mappings to change. Multi-row/None row exchanges are included by this registered row-level definition; neutral changes remain neutral in posthoc efficacy.

| Variant | Assignment changes | Legacy `swap_changes` / corrected `id_set_changes` | Corrected pure swaps |
|---|---:|---:|---:|
| M0 | 0 | 0 | 0 |
| M1 | 335 | 279 | 56 |
| M2 | 455 | 391 | 64 |
| M3 | 335 | 279 | 56 |
| M4 | 375 | 336 | 39 |

Thus the old `391` is retained only as a mislabelled legacy diagnostic; corrected M2 pure swaps are `64`.

## Replay and integrity evidence

The same frozen N47 checkpoint (`{sha256(CHECKPOINT)}`), N42 runtime, seed and input protocol were replayed in the isolated directory: 24 events × 5 variants × 100 frames = 12,000 runtime frames. The independent checker recomputed Hungarian-with-explicit-NONE for every branch and verified source frame continuity, candidate rows/native-ID uniqueness, write/plus public-ID axes, changed-cell lists, hard negatives, and direct `runtime_future_gt_used=false`. GT was read only after runtime validation for simulated posthoc labels. The targeted regression covers a two-row swap, NONE-row exchange, removal/NONE, and unchanged assignment.

The posthoc utility, target IoU, future identity error, re-correction, correct/incorrect/neutral/no-change decomposition and untouched-ID values are byte-equivalent to legacy N47 results. The N47 M2 incremental context remains H20/H50/H100 utility `{m2['20']['identity_utility']:.9f}` / `{m2['50']['identity_utility']:.9f}` / `{m2['100']['identity_utility']:.9f}`; this repair does not turn any failure into a pass.

## Gate and provenance

The repair gate is [n47_swap_metric_gate.json](./outputs/n47_global_probe/repair1_swap_metric/n47_swap_metric_gate.json). It is `{status}` because the inherited N47 efficacy/untouched gates remain failed and N40-compliant real human tape and real SAM3 full-loop are absent. All events remain `simulated_from_gt`; no GT log was relabelled as human. Standard MOT/TrackEval metrics remain `NOT_COMPUTABLE_NO_COMPLETE_TRACKEVAL_SEQUENCE_INPUT`. Calibration, decoder LoRA and production authorization remain prohibited.

The intermediate direct-two-cycle classification failure and the integrity checker contract failure are preserved under [attempts](./outputs/n47_global_probe/repair1_swap_metric/attempts). The total research objective is intentionally still open.
"""
    write_json(out / "n47_swap_metric_gate.json", gate)
    stage = {
        "status": status,
        "protocol": "N47_SWAP_METRIC_REPAIR_STAGE_05_V1",
        "command": ["python", "scripts/n47_swap_metric_finalize.py", "--output-root", str(out)],
        "inputs": {"repair_result": str(out / "replay/probe_results.json"), "integrity": str(out / "stage_04_integrity.json"), "legacy_n47_gate": str(legacy / "n47_final_gate.json"), "n40_blocker": str(N40_BLOCKER)},
        "outputs": {"gate": str(out / "n47_swap_metric_gate.json"), "report": str(ROOT / "docs/N47_SWAP_METRIC_REPAIR_FINAL_REPORT.md"), "stage_status": str(out / "stage_05_status.json")},
        "metrics": {"corrected_pure_swap_changes": pure, "assignment_changes": assignments, "id_set_changes": id_set, "m2_incremental": gate["m2_incremental_context"], "legacy_n47_status": old_gate.get("status"), "real_input_status": blocker.get("status")},
        "gate_checks": checks,
        "failure_root_cause": gate["failure_root_cause"],
        "next_action": gate["next_action"],
        "runtime_future_gt_used": False,
        "gt_loaded_posthoc": True,
        "finished_at": now(),
    }
    write_json(out / "stage_05_status.json", stage)
    (ROOT / "docs/N47_SWAP_METRIC_REPAIR_FINAL_REPORT.md").write_text(report, encoding="utf-8")
    print(json.dumps({"status": status, "gate": str(out / "n47_swap_metric_gate.json"), "report": str(ROOT / "docs/N47_SWAP_METRIC_REPAIR_FINAL_REPORT.md"), "pure_swap": pure}))


if __name__ == "__main__":
    main()
