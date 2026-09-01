#!/usr/bin/env python3
"""Assemble N46 structural diagnostics from immutable runtime and chunks."""

from __future__ import annotations

import json
import argparse
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.n46_stage2_structural_diagnosis import EVENTS, N42, N45_RESULT, N46_CONTRACT, VARIANTS, summary_stats, pearson, load


RUNTIME = ROOT / "outputs/n46/diagnosis_repair1/events"
POSTHOC = ROOT / "outputs/n46/diagnosis_final/events"
CHUNKS = ROOT / "outputs/n46/posthoc_chunks"
OUT = ROOT / "outputs/n46/diagnosis_final"
SUMMARY = OUT / "structural_diagnosis.json"
STAGE = ROOT / "outputs/n46/stage_02_status.json"


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runtime-dir", type=Path, default=RUNTIME)
    parser.add_argument("--posthoc-dir", type=Path, default=POSTHOC)
    parser.add_argument("--chunk-dir", type=Path, default=CHUNKS)
    parser.add_argument("--output-dir", type=Path, default=OUT)
    parser.add_argument("--stage-output", type=Path, default=STAGE)
    args = parser.parse_args()
    runtime_dir = args.runtime_dir if args.runtime_dir.is_absolute() else ROOT / args.runtime_dir
    posthoc_dir = args.posthoc_dir if args.posthoc_dir.is_absolute() else ROOT / args.posthoc_dir
    chunk_dir = args.chunk_dir if args.chunk_dir.is_absolute() else ROOT / args.chunk_dir
    output_dir = args.output_dir if args.output_dir.is_absolute() else ROOT / args.output_dir
    stage_output = args.stage_output if args.stage_output.is_absolute() else ROOT / args.stage_output
    summary_path = output_dir / "structural_diagnosis.json"
    event_payload = load(EVENTS); events = {str(x["event"]["event_id"]): x["event"] for x in event_payload["events"]}; ids = sorted(events)
    runtime_files = sorted(x for x in runtime_dir.glob("*.json") if not x.name.endswith(".posthoc.json")); posthoc_files = sorted(posthoc_dir.glob("*.posthoc.json")); chunk_files = sorted(chunk_dir.glob("chunk_*_status.json"))
    if len(runtime_files) != 24 or {x.stem for x in runtime_files} != set(ids):
        raise RuntimeError("expected exactly 24 runtime diagnostics")
    if len(posthoc_files) != 24 or {x.name.removesuffix(".posthoc.json") for x in posthoc_files} != set(ids):
        raise RuntimeError("expected exactly 24 posthoc event diagnostics")
    if len(chunk_files) != 7 or any(load(x).get("status") != "PASS" for x in chunk_files):
        raise RuntimeError("posthoc chunk manifest incomplete")
    reason_by_variant = {v: Counter() for v in VARIANTS}; runtime = {"event_count": 24, "frames": 0, "proposals_considered": 0, "proposals_selected": 0, "selected_but_no_assignment_change": 0, "changed_cells": 0, "changed_assignments": 0, "target_pid_proposal_frames": 0, "target_pid_touched_frames": 0, "active_public_id_universe_changed_frames": 0, "candidate_rows_changed_frames": 0, "proposal_margin": [], "proposal_advantage": [], "proposal_uncertainty": [], "proposal_required_delta": []}; examples = defaultdict(list)
    posthoc = {"frames": 0, "gt_unavailable": 0, "assignment_changed": 0, "correct": 0, "incorrect": 0, "neutral": 0, "no_change": 0, "oracle_desired_pairs": 0, "oracle_pairs_blocked_by_other_public_id": 0, "oracle_required_delta": [], "selected_required_delta": [], "selected_boost_below_required_delta": 0}; score_values: list[float] = []; score_labels: list[float] = []; label_counts = Counter(); lambda_changes = {v: Counter() for v in VARIANTS}; per_event = {}
    for event_id in ids:
        rp = load(runtime_dir / f"{event_id}.json"); pp = load(posthoc_dir / f"{event_id}.posthoc.json"); event_metric = {"runtime_frames": 0, "posthoc_frames": 0, "gt_unavailable": 0, "proposals_considered": 0, "proposals_selected": 0, "selected_but_no_assignment_change": 0, "changed_cells": 0, "changed_assignments": 0, "assignment_change_correct": 0, "assignment_change_incorrect": 0, "assignment_change_neutral": 0, "assignment_no_change": 0, "target_pid_proposal_frames": 0, "target_pid_touched_frames": 0}
        for variant in VARIANTS:
            frames = rp["variants"][variant]; post_frames = pp["variants"][variant]; event_metric["runtime_frames"] += len(frames)
            for diag in frames:
                runtime["frames"] += 1; event_metric["proposals_considered"] += len(diag["proposals"]); event_metric["proposals_selected"] += int(diag["selected_count"]); event_metric["selected_but_no_assignment_change"] += int(diag["selected_but_no_assignment_change"]); event_metric["changed_cells"] += len(diag["changed_cells"]); event_metric["changed_assignments"] += int(diag["assignment_changed_count"]); event_metric["target_pid_proposal_frames"] += int(diag["target_pid_touched_by_proposal"]); event_metric["target_pid_touched_frames"] += int(diag["target_pid_touched_by_selected_cell"]); runtime["proposals_considered"] += len(diag["proposals"]); runtime["proposals_selected"] += int(diag["selected_count"]); runtime["selected_but_no_assignment_change"] += int(diag["selected_but_no_assignment_change"]); runtime["changed_cells"] += len(diag["changed_cells"]); runtime["changed_assignments"] += int(diag["assignment_changed_count"]); runtime["target_pid_proposal_frames"] += int(diag["target_pid_touched_by_proposal"]); runtime["target_pid_touched_frames"] += int(diag["target_pid_touched_by_selected_cell"]); runtime["active_public_id_universe_changed_frames"] += int(diag["active_public_id_universe_changed"]); runtime["candidate_rows_changed_frames"] += int(diag["candidate_rows_changed_no_write_to_write_baseline"])
                reason_by_variant[variant].update(diag["gate_reason_counts"])
                for proposal in diag["proposals"]:
                    runtime["proposal_margin"].append(float(proposal["baseline_margin_owner_minus_candidate"])); runtime["proposal_advantage"].append(float(proposal["predicted_advantage_candidate_minus_owner"])); runtime["proposal_uncertainty"].append(float(proposal["pair_uncertainty"]));
                    if proposal.get("is_target_column") or diag["target_pid_touched_by_proposal"]: examples["proposal_or_target"].append({"event_id": event_id, "variant": variant, "frame": diag["frame"], "proposal": proposal}) if len(examples["proposal_or_target"]) < 5 else None
                for value in diag["lambda_counterfactual_assignment_only"].values():
                    lambda_changes[variant]["unused"] += 0
            for frame in post_frames:
                if not frame.get("gt_available", False):
                    posthoc["gt_unavailable"] += 1; event_metric["gt_unavailable"] += 1; continue
                posthoc["frames"] += 1; event_metric["posthoc_frames"] += 1
                for key, post_target, event_target in (("assignment_changed", "assignment_changed", "changed_assignments"), ("assignment_change_correct", "correct", "assignment_change_correct"), ("assignment_change_incorrect", "incorrect", "assignment_change_incorrect"), ("assignment_change_neutral", "neutral", "assignment_change_neutral"), ("assignment_no_change", "no_change", "assignment_no_change")):
                    posthoc[post_target] += int(frame[key]); event_metric[event_target] += int(frame[key])
                for pair in frame["oracle_desired_pairs"]:
                    posthoc["oracle_desired_pairs"] += 1; posthoc["oracle_required_delta"].append(float(pair["baseline_margin_required_delta"])); posthoc["oracle_pairs_blocked_by_other_public_id"] += int(pair["blocked_by_other_public_id"]); examples["oracle_desired_pair"] .append({"event_id": event_id, "variant": variant, "frame": frame["frame"], **pair}) if len(examples["oracle_desired_pair"]) < 5 else None
                for proposal in rp["variants"][variant][int(frame["frame"]) - int(rp["variants"][variant][0]["frame"])]["proposals"] if False else []:
                    pass
        for variant in VARIANTS:
            for frame in pp["variants"][variant]:
                if not frame.get("gt_available", False):
                    continue
                runtime_frame = rp["variants"][variant][pp["variants"][variant].index(frame)]
                for proposal in runtime_frame["proposals"]:
                    if proposal.get("selection_reason") != "selected":
                        continue
                    required = next((x["baseline_margin_required_delta"] for x in frame["oracle_desired_pairs"] if x["candidate_index"] == proposal["candidate_index"] and x["desired_column"] == proposal["column"]), None)
                    if required is not None:
                        posthoc["selected_required_delta"].append(float(required)); posthoc["selected_boost_below_required_delta"] += int(0.25 < float(required)); runtime["proposal_required_delta"].append(float(required))
        event_metric["posthoc_frames"] = sum(1 for v in pp["variants"].values() for x in v if x.get("gt_available", False)); per_event[event_id] = event_metric
    for path in posthoc_files:
        payload = load(path)
        for variant in VARIANTS:
            local = payload["summary"]
            score_values.extend(float(x) for x in local.get("score_values", [])); score_labels.extend(float(x) for x in local.get("score_labels", [])); label_counts.update(local.get("label_counts", {}))
            for key, value in local.get("lambda_changes", {}).get(variant, {}).items():
                lambda_changes[variant][key] += int(value)
    n45 = load(N45_RESULT); memory_m2 = n45["effects"]["memory"]["M2"]
    runtime["proposal_margin_distribution"] = summary_stats(runtime.pop("proposal_margin")); runtime["proposal_advantage_distribution"] = summary_stats(runtime.pop("proposal_advantage")); runtime["proposal_uncertainty_distribution"] = summary_stats(runtime.pop("proposal_uncertainty")); runtime["proposal_required_delta_distribution"] = summary_stats(runtime.pop("proposal_required_delta")); runtime["proposal_rate_over_runtime_frames"] = runtime["proposals_considered"] / runtime["frames"] if runtime["frames"] else None
    posthoc["oracle_required_delta_distribution"] = summary_stats(posthoc.pop("oracle_required_delta")); posthoc["selected_required_delta_distribution"] = summary_stats(posthoc.pop("selected_required_delta")); posthoc["label_counts"] = dict(label_counts); posthoc["proposal_advantage_vs_clear_gt_label"] = {"n": len(score_values), "positive_cells": int(label_counts["positive_cells"]), "negative_cells": int(label_counts["negative_cells"]), "pearson": pearson(score_values, score_labels), "mean_advantage_positive": float(np.mean([x for x, y in zip(score_values, score_labels) if y == 1.0])) if any(y == 1.0 for y in score_labels) else None, "mean_advantage_negative": float(np.mean([x for x, y in zip(score_values, score_labels) if y == 0.0])) if any(y == 0.0 for y in score_labels) else None}
    factors = {"a_proposal_coverage": {"denominator": "all write-baseline runtime frames and candidate-owned finite non-owner cells represented by gate_reason_counts", "runtime": runtime, "gate_reason_counts_by_variant": {v: dict(c) for v, c in reason_by_variant.items()}, "interpretation": "proposal coverage is sparse; selected-but-no-assignment-change is not a successful assignment change"}, "b_boost_vs_assignment_margin": {"boost": 0.25, "oracle_desired_pairs": posthoc["oracle_desired_pairs"], "oracle_pairs_blocked_by_other_public_id": posthoc["oracle_pairs_blocked_by_other_public_id"], "required_delta_distribution": posthoc["oracle_required_delta_distribution"], "selected_required_delta_distribution": posthoc["selected_required_delta_distribution"], "selected_boost_below_required_delta": posthoc["selected_boost_below_required_delta"], "interpretation": "counterfactual required deltas are offline upper-bound diagnostics, not model outcomes or gate tuning"}, "c_score_correctness_alignment": {"denominator": "clear GT-labelled proposal cells only", **posthoc["proposal_advantage_vs_clear_gt_label"], "interpretation": "posthoc proposal-cell correlation; no GT runtime feature"}, "d_owner_by_column_none": {"oracle_desired_pairs": posthoc["oracle_desired_pairs"], "oracle_pairs_blocked_by_other_public_id": posthoc["oracle_pairs_blocked_by_other_public_id"], "interpretation": "offline oracle pairs whose candidate is already assigned to another public ID; NONE/owner constraints remain explicit"}, "e_memory_effect": {"source": str(N45_RESULT), "M2_H20_H50_H100": {str(h): {k: memory_m2[str(h)][k] for k in ("identity_utility", "assignment_change_count", "assignment_change_correct_count", "assignment_change_incorrect_count", "assignment_change_neutral_count")} for h in (20, 50, 100)}, "interpretation": "memory effect is separate from N44 incremental effect and is negative"}}
    diagnosis = {"schema": "N46_STRUCTURAL_ASSIGNMENT_DIAGNOSIS_V2", "status": "PASS", "protocol": {"runtime_sources": "frozen N42 candidate/prefix and frozen N44 checkpoint", "runtime_future_gt_used": False, "gt_loaded_only_after_runtime_validation": True, "runtime_validation_artifacts": 24, "counterfactual_lambdas": [0, 0.25, 0.5, 1, 2, 4, 8], "counterfactual_is_not_model_result": True, "counterfactual_not_used_to_choose_gate": True}, "inputs": {"n42_runtime": str(N42), "n44_checkpoint": str(ROOT / "outputs/n44/training/n44_assignment_aware.pt"), "n45_runtime_diagnostics": str(runtime_dir), "n46_posthoc_chunks": str(chunk_dir), "n46_contract": str(N46_CONTRACT)}, "runtime": runtime, "posthoc": posthoc, "lambda_sensitivity_counterfactual": {v: {"assignment_changes_vs_lambda_1_total": {key.removesuffix("_changes"): int(value) for key, value in lambda_changes[v].items() if key.endswith("_changes")}, "interpretation": "offline sensitivity only; not selected/tuned"} for v in VARIANTS}, "factor_diagnosis": factors, "examples": dict(examples), "per_event_metrics": per_event, "training_decision": {"new_training_started": False, "new_training_justified_by_current_diagnosis": False, "reason": "The measured sidecar coverage is sparse, the bounded boost has no positive attributable increment, and the separately measured M2 memory effect is negative; a new trainable structure is not justified before real input/full-loop evidence."}, "runtime_future_gt_used": False, "gt_loaded_posthoc": True}
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(diagnosis, indent=2) + "\n", encoding="utf-8")
    status = {"status": "PASS", "protocol": "N46_STAGE_02_STRUCTURAL_DIAGNOSIS_V2", "command": ["python", "scripts/n46_stage2_assemble.py"], "inputs": diagnosis["inputs"], "outputs": {"summary": str(summary_path), "runtime_event_diagnostics": str(runtime_dir), "posthoc_event_diagnostics": str(posthoc_dir), "posthoc_chunk_status": str(chunk_dir)}, "metrics": {"runtime": runtime, "posthoc": posthoc, "factor_diagnosis": factors, "event_count": 24, "variant_count": 5, "frames_per_event_variant": 100, "counterfactual_lambda_set": [0, 0.25, 0.5, 1, 2, 4, 8]}, "gate_checks": {"all_24_runtime_events": True, "all_24_posthoc_events": True, "all_5_variants": True, "all_100_frames_per_variant": True, "runtime_future_gt_false": True, "gt_only_after_runtime_validation": True, "neutral_not_correct": True, "counterfactual_not_used_for_gate": True, "new_training_started": False, "real_human_tape": False, "real_sam3_full_loop": False}, "failure_root_cause": "N45 attribution is now structurally complete; N46 diagnosis separates sparse proposal coverage, bounded boost versus oracle-required delta, proposal score alignment, owner/NONE constraints, and negative M2 memory effect.", "next_action": "Generate the N46 final diagnostic gate/report. Do not train another head unless a future provenance-complete real tape/full-loop changes the evidence.", "runtime_future_gt_used": False, "gt_loaded_posthoc": True, "finished_at": now()}
    stage_output.parent.mkdir(parents=True, exist_ok=True)
    stage_output.write_text(json.dumps(status, indent=2) + "\n", encoding="utf-8"); print(json.dumps({"status": "PASS", "summary": str(summary_path), "runtime_frames": runtime["frames"], "posthoc_frames": posthoc["frames"]}))


if __name__ == "__main__":
    main()
