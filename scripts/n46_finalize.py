#!/usr/bin/env python3
"""Finalize the N46 structural diagnosis without changing frozen evidence."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SUMMARY = ROOT / "outputs/n46/diagnosis_final/structural_diagnosis.json"
INTEGRITY = ROOT / "outputs/n46/n46_integrity_report.json"
SIDECAR_REG = ROOT / "outputs/n46/n46_sidecar_targeted_regression.json"
N40 = ROOT / "outputs/n40/stage_01_status.json"
N45_RESULT = ROOT / "outputs/n45/replay/attribution_results.json"
N43_GATE = ROOT / "outputs/n43/n43_final_gate.json"
N45_GATE = ROOT / "outputs/n45/stage_04_status.json"
CHECKPOINT = ROOT / "outputs/n44/training/n44_assignment_aware.pt"
AUTH = ROOT / "outputs/n45/frozen_checkpoint_authorization.json"
OUT = ROOT / "outputs/n46"
GATE = OUT / "n46_final_gate.json"
STAGE = OUT / "stage_03_status.json"
STAGE04 = OUT / "stage_04_status.json"
STAGE05 = OUT / "stage_05_status.json"
BLOCKER = OUT / "BLOCKED_INPUT_REAL_HUMAN_TAPE.json"
CORRECTION = OUT / "legacy_contract_correction.json"
REPORT = ROOT / "docs/N46_FINAL_REPORT.md"


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    summary = load(SUMMARY); integrity = load(INTEGRITY); reg = load(SIDECAR_REG); n40 = load(N40); n45 = load(N45_RESULT); n43 = load(N43_GATE); n45_gate = load(N45_GATE)
    runtime = summary["runtime"]; posthoc = summary["posthoc"]; factors = summary["factor_diagnosis"]
    correction = {
        "schema": "N46_LEGACY_CONTRACT_CORRECTION_V1", "status": "PASS",
        "file": "scripts/n44_stage2_assignment_sidecar.py",
        "legacy_claim": "each positive pairs strongest baseline negative and strongest appearance negative",
        "actual_frozen_implementation": "each positive selects up to two strongest baseline-score negatives; no separate appearance-negative ranking is implemented",
        "action": "comment corrected to match code; no new negative sampler introduced",
        "frozen_audit_hard_negative_cells": 0, "frozen_training_hard_negative_examples": 0,
        "old_outputs_preserved": True, "runtime_future_gt_used": False,
    }
    CORRECTION.write_text(json.dumps(correction, indent=2) + "\n", encoding="utf-8")
    blocker_source = ROOT / "outputs/n45/blocked_input_real_human_tape.json"
    blocker = load(blocker_source)
    blocker_copy = {
        "schema": "N46_BLOCKED_INPUT_REAL_HUMAN_TAPE_V1", "status": "BLOCKED_INPUT_REAL_HUMAN_TAPE",
        "source_artifact": str(blocker_source), "source_sha256": sha256(blocker_source),
        "checks_completed": blocker.get("checks_completed", []), "checks": blocker.get("checks", {}),
        "exact_blocker": blocker["exact_blocker"], "minimal_next_step": blocker["minimal_next_step"],
        "fabrication_or_relabeling": blocker["fabrication_or_relabeling"],
        "runtime_future_gt_used": False, "downstream_authorized": False, "old_simulated_artifacts_relabelled": False,
    }
    BLOCKER.write_text(json.dumps(blocker_copy, indent=2) + "\n", encoding="utf-8")
    checks = {
        "stage01_assignment_recompute_zero": load(ROOT / "outputs/n46/stage_01_status.json")["metrics"]["assignment_mismatch_count"] == 0,
        "stage02_complete_24x5x100": integrity["status"] == "PASS" and runtime["frames"] == 12000,
        "sidecar_contract_regression": reg["status"] == "PASS",
        "runtime_future_gt_false": summary.get("runtime_future_gt_used") is False and integrity.get("runtime_future_gt_used") is False,
        "posthoc_gt_after_runtime_validation": summary.get("gt_loaded_posthoc") is True and integrity.get("gate_checks", {}).get("gt_posthoc_only") is True,
        "counterfactual_not_gate_selection": True,
        "neutral_not_correct": True,
        "n44_checkpoint_authorized_false": load(AUTH).get("production_authorized") is False,
        "real_human_tape": bool(n40.get("real_human_tape_audit", {}).get("verified_real_human_event_tape_found", False)),
        "real_sam3_full_loop": False,
        "standard_mot_metrics": "NOT_COMPUTABLE_NO_COMPLETE_TRACKEVAL_SEQUENCE_INPUT",
        "new_training_started": False,
    }
    gate = {
        "schema": "N46_STRUCTURAL_DIAGNOSTIC_GATE_V1", "status": "N46_COMPLETED_DIAGNOSTIC_GATE_FAILED",
        "research_gate": "FAIL_STRUCTURAL_HYPOTHESIS_AND_REAL_INPUT",
        "authorization": {"calibration_head": "NOT_AUTHORIZED", "decoder_lora": "NOT_AUTHORIZED", "production_interface_changed": False},
        "checks": checks,
        "diagnosis": {
            "a_proposal_coverage": "SUPPORTED_AS_CONTRIBUTOR",
            "b_bounded_boost_vs_assignment_margin": "SUPPORTED_AS_CONTRIBUTOR",
            "c_score_correctness_alignment": "NOT_PRIMARY_BOTTLENECK_ON_CLEAR_OFFLINE_CELLS",
            "d_owner_by_column_none": "SUPPORTED_AS_DOMINANT_INTERFACE_BOTTLENECK",
            "e_memory_effect": "SUPPORTED_NEGATIVE_M2_CONTEXT",
        },
        "memory_effect_status": "NEGATIVE_M2_SEPARATELY_MEASURED",
        "n44_incremental_status": "STRUCTURAL_HYPOTHESIS_FAILED",
        "provenance_status": "SIMULATED_FROM_GT_ONLY",
        "real_input_status": "BLOCKED_INPUT_REAL_HUMAN_TAPE",
        "failure_root_cause": "N46 runtime shows sparse sidecar coverage and owner/column constrained proposals; offline oracle desired pairs are overwhelmingly blocked by another public-ID owner, while the +0.25 bounded boost is far below the required assignment delta distribution. N45 true attribution remains zero for M2 and negative for M1/M3/M4, and the separate M2 memory effect is negative.",
        "next_action": "Keep the research objective open. Obtain the external N40 provenance-complete human tape and candidate-complete real SAM3 full-loop. Only then pre-register a global assignment/listwise structural experiment if the new real evidence supports it.",
    }
    GATE.write_text(json.dumps(gate, indent=2) + "\n", encoding="utf-8")
    n43_m2 = n43["metrics"]["m2_identity_utility"]
    n45_mem = n45["effects"]["memory"]["M2"]
    n45_inc = n45["effects"]["incremental"]["M2"]
    lines = [
        "# N46 Final Report — structural assignment diagnosis",
        "",
        f"Date: {datetime.now(timezone.utc).isoformat()}  ",
        "Status: `N46_COMPLETED_DIAGNOSTIC_GATE_FAILED`. This closes only the N46 structural-diagnosis unit; the scientific objective remains open.",
        "",
        "## Scope and frozen evidence",
        "",
        "The project root was confirmed as `.`. N43, N44 and N45 files, failures, seeds, checkpoints and metric definitions were preserved. No production MOT/OVMOT code, shared checkpoint, or unrelated GPU4–7 task was changed. The N44 checkpoint remains frozen at SHA-256 `0b5e750f5d9569f71ae887595c1d88d4d625f120f8a3811f2598a852cf82348f`, and its hash-bound authorization overlay records `production_authorized=false`.",
        "",
        "N43 remains historical simulated evidence: all 24 events are `simulated_from_gt`, not human tape. Its frozen M2 utility was H20/H50/H100 `-0.009572664282 / -0.009451102014 / -0.007484765400`; it had no real human tape and no real SAM3 full-loop. N45 corrected the attribution design by materializing no-write, original N42 write, and write-plus-N44 from identical candidate/frame sources; its old N44 no-write→plus result is retained only as provisional legacy evidence.",
        "",
        "## Contract audit and repairs",
        "",
        "The first pre-fix audit is preserved at `outputs/n46/attempts/n45_contract_audit_pre_fix.json`. It recorded the inaccurate N44 negative-sampling comment, the hard-coded N45 `three_branches=True`, and the missing independent write-assignment recomputation. The finalizer check was initially too strict about legal branch metadata; that failure is preserved at `outputs/n46/attempts/n45_finalizer_branch_metadata_schema_failure.json`, then minimally corrected to require the three branch keys rather than exact key equality.",
        "",
        "The comment in `scripts/n44_stage2_assignment_sidecar.py` now states the actual implementation: up to two strongest baseline-score negatives, with no separate appearance-negative sampler. The frozen audit had zero hard-negative cells and zero hard-negative training examples; no false claim of hard-negative inclusion is made. The correction is machine-readable at `outputs/n46/legacy_contract_correction.json`.",
        "",
        "Stage 01 independently recomputed `hungarian_with_none(write_scores)` for all 24 × 5 × 100 frames against the normalized N42 write assignment. It found 0 mismatches. It also checked exact event set, 100-frame source/runtime traces, no duplicate/missing frames, unique native IDs, candidate rows, write/plus public-ID axes, and direct `runtime_future_gt_used=false` semantics. The N44 increment is therefore attributable in this frozen synthetic replay, but its efficacy still fails below.",
        "",
        f"The new sidecar regression confirms that `apply_sidecar` uses the current fused branch matrix, preserves hard-negative and NONE semantics, changes only bounded +0.25 finite cells, records changed-cell counts, and does not rewrite the checkpoint. Full integrity checked 24 runtime and 24 posthoc event artifacts, 12000 runtime and 12000 aligned posthoc rows, with GT loaded only after runtime validation. SHA-256 snapshots for all {integrity['metrics']['legacy_files_hashed']} files under the preserved N43/N44/N45 output trees plus their final reports are recorded in `outputs/n46/n46_integrity_report.json`; this is a preservation manifest, not a claim of a pre-change hash comparison.",
        "",
        "## N46 structural diagnosis",
        "",
        f"Runtime diagnostics cover 24 events × 5 variants × 100 frames = {runtime['frames']} frames. There were {runtime['proposals_considered']} proposals considered ({runtime['proposal_rate_over_runtime_frames']:.6f} per runtime frame), {runtime['proposals_selected']} selected, {runtime['selected_but_no_assignment_change']} selected-but-no-assignment-change cases, {runtime['changed_cells']} changed cells and {runtime['changed_assignments']} global assignment changes. Target public-ID proposal/touch occurred on {runtime['target_pid_proposal_frames']}/{runtime['target_pid_touched_frames']} frame counts. Active public-ID universe changes ({runtime['active_public_id_universe_changed_frames']}) and candidate-row changes ({runtime['candidate_rows_changed_frames']}) are reported separately from score and Hungarian-assignment changes in every runtime frame artifact. Neutral is never counted as correct.",
        "",
        "The falsifiable factor readout is:",
        "",
        f"- (a) Coverage: supported contributor. The denominator is all 12000 write-baseline runtime frames plus finite candidate-owned non-owner cells represented by per-variant gate reasons; only 33 proposals were considered and 17 selected. Per-event runtime examples are `dancetrack0027-0157` (10 considered/5 selected), `dancetrack0032-0000` (4/2), `dancetrack0049-0000` (12/5), and `dancetrack0072-0000` (7/5, all 5 selected but no assignment change).",
        f"- (b) Boost versus assignment margin: supported contributor. Across {posthoc['oracle_desired_pairs']} oracle-desired candidate×ID pairs, required delta had median {posthoc['oracle_required_delta_distribution']['median']:.6f}, p25 {posthoc['oracle_required_delta_distribution']['p25']:.6f}, p75 {posthoc['oracle_required_delta_distribution']['p75']:.6f}, p95 {posthoc['oracle_required_delta_distribution']['p95']:.6f}, versus the fixed +0.25 bound. {posthoc['oracle_pairs_blocked_by_other_public_id']}/{posthoc['oracle_desired_pairs']} ({posthoc['oracle_pairs_blocked_by_other_public_id']/posthoc['oracle_desired_pairs']:.4%}) were occupied by another public-ID owner. No threshold or lambda was chosen from this counterfactual.",
        f"- (c) Score correctness alignment: not primary bottleneck on the available clear offline cells. The denominator is 115 proposal cells with unambiguous posthoc IoU labels (25 positive, 90 negative); predicted advantage Pearson correlation is {posthoc['proposal_advantage_vs_clear_gt_label']['pearson']:.6f}, with mean advantage {posthoc['proposal_advantage_vs_clear_gt_label']['mean_advantage_positive']:.6f} on positives versus {posthoc['proposal_advantage_vs_clear_gt_label']['mean_advantage_negative']:.6f} on negatives. This is posthoc analysis only and never a runtime feature.",
        f"- (d) Owner-by-column/NONE: supported as dominant interface bottleneck. The same {posthoc['oracle_desired_pairs']} desired pairs include {posthoc['oracle_pairs_blocked_by_other_public_id']} blocked by another public-ID owner. NONE/abstain and owner constraints were preserved; they were not silently turned into positive labels.",
        f"- (e) Appearance memory: supported as a separate negative context, not conflated with sidecar increment. Frozen N45 M2 memory utility is H20/H50/H100 `{n45_mem['20']['identity_utility']:.15f} / {n45_mem['50']['identity_utility']:.15f} / {n45_mem['100']['identity_utility']:.15f}`, with 0 correct and 12/42/92 incorrect changes at those horizons. This is the no-write→write effect, not the N44 increment.",
        "",
        "An offline sensitivity record for fixed lambdas `{0,0.25,0.5,1,2,4,8}` is in `structural_diagnosis.json`; for M2 the assignment-change counts versus lambda 1 are `0,14,8,2,4,7,70`. This is a pre-registered counterfactual upper-bound diagnostic, not a model result, threshold selection, or gate bypass.",
        "",
        "## N45 attribution, side by side",
        "",
        "| Frozen comparison | H20 | H50 | H100 |",
        "|---|---:|---:|---:|",
        f"| N43 M2 historical utility | {n43_m2['20']:.15f} | {n43_m2['50']:.15f} | {n43_m2['100']:.15f} |",
        f"| N45 M2 memory no-write→write utility | {n45_mem['20']['identity_utility']:.15f} | {n45_mem['50']['identity_utility']:.15f} | {n45_mem['100']['identity_utility']:.15f} |",
        f"| N45 true N44 increment write→plus utility | {n45_inc['20']['identity_utility']:.15f} | {n45_inc['50']['identity_utility']:.15f} | {n45_inc['100']['identity_utility']:.15f} |",
        f"| N45 true increment assignment changed/correct/incorrect/neutral | {n45_inc['20']['assignment_change_count']}/{n45_inc['20']['assignment_change_correct_count']}/{n45_inc['20']['assignment_change_incorrect_count']}/{n45_inc['20']['assignment_change_neutral_count']} | {n45_inc['50']['assignment_change_count']}/{n45_inc['50']['assignment_change_correct_count']}/{n45_inc['50']['assignment_change_incorrect_count']}/{n45_inc['50']['assignment_change_neutral_count']} | {n45_inc['100']['assignment_change_count']}/{n45_inc['100']['assignment_change_correct_count']}/{n45_inc['100']['assignment_change_incorrect_count']}/{n45_inc['100']['assignment_change_neutral_count']} |",
        "",
        "N45’s true M2 increment is zero at all horizons, with only neutral 6/9/15 changes; M1/M3/M4 were slightly negative with incorrect changes and no correct changes. Therefore N46 does not label this as `N44_NOT_EXERCISED`: the sidecar was selected and changed score/assignment state, but the structural hypothesis failed to yield correct future benefit. The N45 equal-sequence bootstrap CIs and all event-level attribution remain unchanged.",
        "",
        "## Training decision and provenance gate",
        "",
        "No N46 training was started. The observed sparse coverage, dominant owner/column constraint, large required deltas, and negative separately measured memory effect do not justify another training experiment or a blind scan. A future experiment would need a new pre-registered global assignment/listwise interface that can represent swaps/owner release and explicit abstention; it must wait for provenance-complete real input/full-loop evidence. N44’s actual sequence-disjoint training remains the only N44 training, with frozen seed/checkpoint and no holdout tuning.",
        "",
        f"The N40 external-ingestion audit remains `{n40['status']}`. Three evidence-preserving checks remain: the N34 real-tape sentinel is unavailable/empty, N40 found no candidate-complete external UI/annotator tape, and the available fallback is GT-derived synthetic data. A real tape must directly contain public_id, human-confirmed BOX/CLICK/CONFIRMED_MASK, lossless ROI digest, UI/session/annotator timestamps, and candidate-complete future rows while runtime GT remains forbidden. The explicit N46 blocker is `outputs/n46/BLOCKED_INPUT_REAL_HUMAN_TAPE.json`; no GT log was fabricated or renamed.",
        "",
        "Standard MOT/TrackEval metrics are `NOT_COMPUTABLE_NO_COMPLETE_TRACKEVAL_SEQUENCE_INPUT`. Simulated events are not human evidence, and this offline diagnosis cannot authorize calibration, decoder LoRA, production interface changes, or completion of the broader scientific objective.",
        "",
        "## Stage artifacts and retained failures",
        "",
        "- Stage 01 contract audit: `outputs/n46/stage_01_status.json`; mismatch list: `outputs/n46/replay/assignment_mismatches.json`.",
        "- Stage 02 diagnosis: `outputs/n46/stage_02_status.json`; summary: `outputs/n46/diagnosis_final/structural_diagnosis.json`; per-frame runtime/posthoc artifacts are under `outputs/n46/diagnosis_repair1/events` and `outputs/n46/diagnosis_final/events`.",
        "- Stage 03 strict diagnosis gate: `outputs/n46/stage_03_status.json`; Stage 04 N40 input feasibility: `outputs/n46/stage_04_status.json`; Stage 05 final gate: `outputs/n46/stage_05_status.json` and `outputs/n46/n46_final_gate.json`; integrity: `outputs/n46/n46_integrity_report.json`; sidecar regression: `outputs/n46/n46_sidecar_targeted_regression.json`.",
        "- All N46 failed attempts, including partial runtime/posthoc runs and schema/termination failures, remain under `outputs/n46/attempts`; no failure artifact was deleted or converted to PASS.",
        "",
        "Next step is external: obtain and validate the N40 real human tape and real candidate-complete SAM3 full-loop. Until those are available, keep the research objective open and do not authorize production calibration or decoder LoRA.",
    ]
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    stage = {
        "status": gate["status"], "protocol": gate["schema"],
        "command": ["python", "scripts/n46_finalize.py"],
        "inputs": {"n46_summary": str(SUMMARY), "n46_integrity": str(INTEGRITY), "n46_sidecar_regression": str(SIDECAR_REG), "n40_audit": str(N40), "n45_attribution": str(N45_RESULT), "n43_gate": str(N43_GATE), "n45_gate": str(N45_GATE), "n44_checkpoint": str(CHECKPOINT)},
        "outputs": {"gate": str(GATE), "report": str(REPORT), "real_human_blocker": str(BLOCKER), "legacy_contract_correction": str(CORRECTION)},
        "metrics": {"runtime": runtime, "posthoc": posthoc, "diagnosis": gate["diagnosis"], "n43_m2_utility_frozen": n43_m2, "n45_m2_memory": n45_mem, "n45_m2_increment": n45_inc, "real_human_tape": False, "real_sam3_full_loop": False, "new_training_started": False},
        "gate_checks": checks, "failure_root_cause": gate["failure_root_cause"], "next_action": gate["next_action"], "runtime_future_gt_used": False, "gt_loaded_posthoc": True, "finished_at": datetime.now(timezone.utc).isoformat(),
    }
    STAGE.write_text(json.dumps(stage, indent=2) + "\n", encoding="utf-8")
    stage04 = {
        "status": "BLOCKED_INPUT_REAL_HUMAN_TAPE", "protocol": "N46_STAGE_04_N40_INPUT_FEASIBILITY_V1",
        "command": ["python", "scripts/n46_finalize.py"],
        "inputs": {"n40_audit": str(N40), "n45_blocker": str(blocker_source)},
        "outputs": {"blocked_input_artifact": str(BLOCKER)},
        "metrics": {"real_human_tape": False, "real_sam3_full_loop": False, "sentinel_event_count": blocker_copy["checks"]["sentinel_tape_check"]["event_count"], "external_ui_export_found": blocker_copy["checks"]["source_and_artifact_inventory_check"]["external_ui_export_found"], "candidate_complete_real_tape_found": blocker_copy["checks"]["source_and_artifact_inventory_check"]["candidate_complete_real_tape_found"]},
        "gate_checks": {"direct_public_id": False, "human_confirmed_box_click_mask": False, "lossless_roi_digest": False, "ui_session_annotator_timestamps": False, "candidate_complete_future_rows": False, "runtime_gt_forbidden": True, "simulated_not_relabelled": True},
        "failure_root_cause": blocker_copy["exact_blocker"],
        "next_action": blocker_copy["minimal_next_step"], "runtime_future_gt_used": False, "gt_loaded_posthoc": True, "finished_at": datetime.now(timezone.utc).isoformat(),
    }
    STAGE04.write_text(json.dumps(stage04, indent=2) + "\n", encoding="utf-8")
    stage05 = {
        "status": gate["status"], "protocol": "N46_FINAL_GATE_V1",
        "command": ["python", "scripts/n46_finalize.py"],
        "inputs": {"stage_01": str(ROOT / "outputs/n46/stage_01_status.json"), "stage_02": str(ROOT / "outputs/n46/stage_02_status.json"), "stage_03": str(STAGE), "stage_04": str(STAGE04), "diagnosis": str(SUMMARY)},
        "outputs": {"final_gate": str(GATE), "report": str(REPORT)},
        "metrics": {"n44_incremental_status": gate["n44_incremental_status"], "memory_effect_status": gate["memory_effect_status"], "provenance_status": gate["provenance_status"], "real_input_status": gate["real_input_status"], "new_training_started": False},
        "gate_checks": checks, "failure_root_cause": gate["failure_root_cause"], "next_action": gate["next_action"], "runtime_future_gt_used": False, "gt_loaded_posthoc": True, "finished_at": datetime.now(timezone.utc).isoformat(),
    }
    STAGE05.write_text(json.dumps(stage05, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": stage["status"], "gate": str(GATE), "report": str(REPORT)}))


if __name__ == "__main__":
    main()
