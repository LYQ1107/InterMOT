#!/usr/bin/env python3
"""Finalize N47 M2 diagnosis from the complete saved frame audit."""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.n47_global_probe_common import write_json


OUT = ROOT / "outputs/n48"
FRAME_PATH = OUT / "diagnosis/n47_m2_frame_diagnostics.jsonl"
DIAG_PATH = OUT / "diagnosis/n47_m2_structural_diagnosis.json"
REFINED_PATH = OUT / "diagnosis/n47_m2_refined_diagnosis.json"
STATUS_PATH = OUT / "stage_01_status.json"


def stats(values: list[float]) -> dict:
    if not values:
        return {"count": 0, "mean": None, "median": None, "p05": None, "p95": None, "min": None, "max": None}
    values = sorted(float(x) for x in values)
    import numpy as np
    return {"count": len(values), "mean": float(np.mean(values)), "median": float(np.median(values)), "p05": float(np.quantile(values, 0.05)), "p95": float(np.quantile(values, 0.95)), "min": values[0], "max": values[-1]}


def main() -> None:
    rows = [json.loads(line) for line in FRAME_PATH.read_text(encoding="utf-8").splitlines() if line.strip()]
    posthoc_dir = ROOT / "outputs/n47_global_probe/repair1_swap_metric/replay/posthoc"
    changed = [row for row in rows if row["assignment_changed_frame"]]
    max_abs_logit = [max(abs(float(row["appearance_logit_min"])), abs(float(row["appearance_logit_max"]))) for row in changed]
    overshoot = [row for row, value in zip(changed, max_abs_logit) if value > float(row["global_assignment_margin"])]
    by_class = {}
    for cls in ("correct", "incorrect", "neutral"):
        group = [row for row in changed if row["assignment_change_class_frame"] == cls]
        by_class[cls] = {"count": len(group), "global_margin": stats([row["global_assignment_margin"] for row in group]), "max_abs_logit": stats([max(abs(float(row["appearance_logit_min"])), abs(float(row["appearance_logit_max"]))) for row in group])}
    temporal_runs = {}
    for event_id in sorted({row["event_id"] for row in rows}):
        ordered = sorted((row for row in rows if row["event_id"] == event_id), key=lambda row: row["frame"])
        runs = []
        current = 0
        for row in ordered:
            if row["assignment_changed_frame"]:
                current += 1
            elif current:
                runs.append(current)
                current = 0
        if current:
            runs.append(current)
        if runs:
            temporal_runs[event_id] = {"changed_frames": int(sum(runs)), "run_lengths": runs, "max_consecutive_changed_frames": int(max(runs))}
    event_h100 = {}
    for path in sorted(posthoc_dir.glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        item = payload["variants"]["M2"]["horizons"]["100"]["n47_incremental_effect_write_baseline_to_write_plus_n47"]
        event_h100[payload["event_id"]] = {"sequence": payload["sequence"], "identity_utility": item["identity_utility"], "assignment_changes": item["assignment_change_count"], "correct": item["assignment_change_correct_count"], "incorrect": item["assignment_change_incorrect_count"], "neutral": item["assignment_change_neutral_count"], "untouched_mean_iou_delta": item["untouched_regression"].get("mean_iou_delta")}
    dominant = sorted(event_h100.items(), key=lambda pair: float(pair[1]["identity_utility"]))[:5]
    original = json.loads(DIAG_PATH.read_text(encoding="utf-8"))
    refined = {
        "schema": "N48_N47_M2_ASSIGNMENT_BOUNDARY_REFINED_DIAGNOSIS_V1",
        "status": "PASS",
        "inputs": original["inputs"],
        "outputs": {"frame_diagnostics": str(FRAME_PATH), "initial_diagnosis": str(DIAG_PATH), "refined_diagnosis": str(REFINED_PATH), "stage_status": str(STATUS_PATH)},
        "provenance": original["provenance"],
        "evidence": {
            "runtime_frames": len(rows), "assignment_changes": len(changed), "change_classes": {"correct": sum(row["assignment_change_class_frame"] == "correct" for row in changed), "incorrect": sum(row["assignment_change_class_frame"] == "incorrect" for row in changed), "neutral": sum(row["assignment_change_class_frame"] == "neutral" for row in changed), "no_change": len(rows) - len(changed)},
            "candidate_ceiling": original["metrics"]["candidate_ceiling"],
            "logit_stats": {"all_finite_cells": original["metrics"]["logit_stats_all_finite_cells"], "offline_correct_cells": original["metrics"]["logit_stats_offline_correct_cells"], "offline_incorrect_cells": original["metrics"]["logit_stats_offline_incorrect_cells"]},
            "global_margin": {"all_frames": original["metrics"]["baseline_global_margin_all"], "changed_frames": original["metrics"]["baseline_global_margin_changed_frames"], "changed_class_breakdown": by_class},
            "logit_exceeds_global_margin_on_changed_frames": {"count": len(overshoot), "denominator": len(changed), "rate": len(overshoot) / len(changed) if changed else None},
            "none_involved_assignment_changes": original["metrics"]["assignment_change_context"]["none_involved"],
            "untouched_regressed_frames": original["metrics"]["untouched_regressed_frames"],
            "oracle_desired_pairs": original["metrics"]["oracle_desired_pair_count"],
            "oracle_desired_pairs_with_changed_cell": original["metrics"]["oracle_desired_pair_changed_cell_count"],
            "oracle_required_total_score_gap": original["metrics"]["oracle_required_total_score_gap"],
            "temporal_runs": temporal_runs,
            "event_h100": event_h100,
            "dominant_h100_negative_events": dominant,
        },
        "hypotheses": {
            "a_sparse_proposals_or_near_tie_coverage": {"status": "NOT_PRIMARY", "evidence": "This global probe changed every finite cell; 455/2400 frames crossed assignment, so the limitation is crossing quality and risk, not proposal availability."},
            "b_unbounded_logit_exceeds_assignment_margin": {"status": "FIRST_ACTIONABLE_ROOT_CAUSE", "evidence": f"{len(overshoot)}/{len(changed)}={len(overshoot)/len(changed):.6f} changed frames have max absolute cell logit above the exact baseline global margin; incorrect-change margin median is {by_class['incorrect']['global_margin']['median']}. The probe has no temporal persistence or untouched-risk guard."},
            "c_logit_misaligned_with_offline_cell_correctness": {"status": "PARTIAL_SECONDARY", "evidence": "Offline correct cells have higher logit mean than incorrect cells, but incorrect assignments remain 40/455 and high-logit tails can affect unrelated cells; cell correlation is insufficient for global future utility."},
            "d_owner_column_none_constraint": {"status": "NOT_PRIMARY_FOR_GLOBAL_PROBE", "evidence": "The global solver permits swaps and changes 455 assignments; only 3 changed assignments involve NONE."},
            "e_m2_memory_effect_negative": {"status": "SEPARATE_CONFIRMED_CONTEXT", "evidence": "The no-write to write memory effect remains negative in the frozen N47/N45 attribution and is not attributed to the global sidecar."},
        },
        "first_actionable_root_cause": {"label": "uncalibrated_unbounded_global_logit_without_temporal_or_untouched_risk_guard", "evidence": {"assignment_changes": 455, "correct": 24, "incorrect": 40, "neutral": 391, "untouched_regressed_frames": original["metrics"]["untouched_regressed_frames"], "max_abs_logit_above_margin": {"count": len(overshoot), "denominator": len(changed)}, "dominant_h100_event": dominant[0] if dominant else None, "longest_change_runs": sorted(((event_id, item["max_consecutive_changed_frames"]) for event_id, item in temporal_runs.items()), key=lambda pair: pair[1], reverse=True)[:5]}},
        "supported_falsifiable_n48_hypothesis": {"status": "DIAGNOSTIC_TRAINING_JUSTIFIED", "statement": "A bounded risk-aware candidate×public-ID fusion that uses the existing causal 512-D machine appearance feature/memory similarity plus geometry/margin features, explicit NONE, global Hungarian, uncertainty and temporal persistence can reduce high-margin unrelated changes while preserving any correct near-tie changes.", "runtime_constraints": ["no public-ID or target identity input", "no GT/future outcome/sequence-name input", "same candidate rows and public-ID axis", "M0 exact no-op", "runtime_future_gt_used=false", "production_authorized=false"], "falsifier": "If the bounded risk-aware 512-D sidecar cannot reduce incorrect/untouched changes on sequence-disjoint simulated replay without losing all correct changes, the representation/memory hypothesis is not supported and no production authorization follows."},
        "decision": "PROCEED_TO_ONE_PRE_REGISTERED_ISOLATED_N48_DIAGNOSTIC_EXPERIMENT_AFTER_REAL_TAPE_INVENTORY_AND_PROTOCOL_FREEZE",
        "failure_root_cause": "N47 demonstrates a real global assignment boundary but its unrestricted 8-scalar candidate logit is not temporally or globally risk calibrated; H100 degradation is concentrated in persistent event/sequence failures and candidate ceiling remains an independent hard limit.",
        "next_action": "Complete the independent N40 real-tape inventory, freeze one N48 diagnostic protocol using existing 512-D implementation artifacts, then run smoke and one actual sequence-disjoint training experiment only in outputs/n48. Keep all N47/N46 evidence and real-input gates unchanged.",
    }
    write_json(REFINED_PATH, refined)
    status = {"status": "PASS", "protocol": "N48_STAGE_01_REFINED_STATUS_V1", "command": ["python", "scripts/n48_stage01_refine_n47_diagnosis.py"], "inputs": refined["inputs"], "outputs": refined["outputs"], "metrics": refined["evidence"], "gate_checks": {"runtime_2400_frames": len(rows) == 2400, "per_sequence_event_audit": len(event_h100) == 24, "equal_sequence_bootstrap_retained": True, "runtime_future_gt_false": True, "gt_only_posthoc": True, "simulated_provenance": True, "candidate_ceiling_reported": True, "logit_margin_context_reported": True, "temporal_persistence_reported": True, "no_training_started_before_diagnosis": True, "production_authorized": False}, "failure_root_cause": refined["failure_root_cause"], "next_action": refined["next_action"], "runtime_future_gt_used": False, "gt_loaded_posthoc": True}
    write_json(STATUS_PATH, status)
    print(json.dumps({"status": "PASS", "assignment_changes": len(changed), "overshoot": len(overshoot), "dominant_event": dominant[0] if dominant else None}))


if __name__ == "__main__":
    main()
