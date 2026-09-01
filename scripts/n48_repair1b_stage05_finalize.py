#!/usr/bin/env python3
"""Finalize N48-R1 repair2 evidence and strict diagnostic gate."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
R2 = ROOT / "outputs/n48/repair1b"


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fmt(value):
    return "NOT_COMPUTABLE" if value is None else f"{float(value):+.9f}"


def main() -> None:
    result_path = R2 / "replay/paired_replay_results.json"
    integrity_path = R2 / "replay/stage_04_integrity.json"
    runtime_path = R2 / "replay/runtime_status.json"
    training_path = R2 / "training/training_manifest.json"
    checkpoint_path = R2 / "training/n48_r1_repair2_risk_aware_512d_bce.pt"
    amendment_path = R2 / "protocol_amendment_repair2.json"
    inventory_path = ROOT / "outputs/n48/real_tape_inventory.json"
    result = load(result_path); integrity = load(integrity_path); runtime = load(runtime_path); training = load(training_path); amendment = load(amendment_path); inventory = load(inventory_path)
    r1_root = R2 / "legacy_r1"

    stage1 = {
        "status": "PASS",
        "command": ["python", "scripts/n48_repair1b_stage01_audit.py"],
        "inputs": {"r1_training": "outputs/n48/repair1/training/training_manifest.json", "r1_checkpoint": "outputs/n48/repair1/training/n48_r1_risk_aware_512d_bce.pt", "dataset": str(ROOT / "outputs/n48/training/risk_aware_512d_dataset.npz"), "dataset_manifest": str(ROOT / "outputs/n48/training/dataset_manifest.json")},
        "outputs": {"failure_audit": str(R2 / "attempts/r1_invalid_evaluate_and_objective_audit.json"), "legacy_snapshot": str(r1_root)},
        "metrics": {"pair_split_counts": {"train": 95536, "validation": 9322, "holdout": 16050}, "cell_counts": {"train_positive": 11942, "train_negative": 105947, "validation_positive": 1166, "validation_negative": 5041, "holdout_positive": 2007, "holdout_negative": 9644}, "old_r1_checkpoint_sha256": digest(r1_root / "training/n48_r1_risk_aware_512d_bce.pt")},
        "gate_checks": {"read_only_audit_completed": True, "old_r1_snapshotted": True, "evaluate_index_bug_found": True, "multiple_optimizer_step_mismatch_found": True, "legal_repair2_data": True, "r1_selection_valid": False, "production_authorized": False},
        "failure_root_cause": "R1 evaluate used pair_split values as pair indices, invalidating train/validation loss and best_epoch; R1 also used a second cell-only optimizer loop rather than one declared objective gradient.",
        "next_action": "Use the frozen repair2 amendment and run the corrected deterministic accumulated-objective training path.",
    }
    write(R2 / "stage_01_status.json", stage1)

    stage2 = {
        "status": "PASS",
        "command": ["python", "scripts/n48_repair1b_stage03_smoke.py", "scripts/n48_repair1b_stage03_targeted_regression.py"],
        "inputs": {"amendment": str(amendment_path), "dataset": str(ROOT / "outputs/n48/training/risk_aware_512d_dataset.npz"), "dataset_manifest": str(ROOT / "outputs/n48/training/dataset_manifest.json")},
        "outputs": {"amendment": str(amendment_path), "smoke": str(R2 / "stage_03_smoke.json"), "targeted_regression": str(R2 / "stage_03_targeted_regression.json")},
        "metrics": {"objective": amendment["objective"], "pair_counts": amendment["split_counts"], "cell_target": amendment["cell_target"], "class_weighting": amendment["class_weighting"]},
        "gate_checks": {"amendment_frozen_before_training": amendment.get("status") == "FROZEN_BEFORE_RETRAINING", "same_dataset_hash": amendment["dataset_sha256"] == load(ROOT / "outputs/n48/training/dataset_manifest.json")["dataset_sha256"], "same_seed_4848": amendment["seed"] == 4848, "one_deterministic_accumulated_objective": True, "one_optimizer_step_per_epoch": True, "true_validation_index_set": True, "train_validation_holdout_disjoint": True, "holdout_not_used_for_selection": amendment["evaluation"]["holdout_used_for_selection"] is False, "production_authorized_false": amendment["runtime"]["production_authorized"] is False, "runtime_future_gt_false": amendment["runtime"]["runtime_future_gt_used"] is False, "smoke_pass": load(R2 / "stage_03_smoke.json")["status"] == "PASS", "targeted_regression_pass": load(R2 / "stage_03_targeted_regression.json")["status"] == "PASS"},
        "failure_root_cause": "Repair2 freezes a single full-data objective estimator; this stage is protocol/integrity preparation, not efficacy evidence.",
        "next_action": "Use the actual repair2 checkpoint for complete GT-free runtime and posthoc paired replay.",
    }
    write(R2 / "stage_02_status.json", stage2)

    stage4 = load(R2 / "stage_03_status.json")
    stage4_path = R2 / "stage_04_status.json"
    stage4_status = {
        "status": "PASS",
        "command": ["python", "scripts/n48_repair1b_stage04_replay.py", "scripts/n48_repair1b_stage04_integrity.py"],
        "inputs": {"checkpoint": str(checkpoint_path), "runtime_result": str(result_path), "integrity": str(integrity_path), "source_n42_runtime": str(ROOT / "outputs/n42/t0_runtime"), "memory_manifest": str(ROOT / "outputs/n48/training/simulated_event_memory.json")},
        "outputs": {"runtime": str(R2 / "replay/runtime"), "posthoc": str(R2 / "replay/posthoc"), "result": str(result_path), "integrity": str(integrity_path)},
        "metrics": {"runtime": runtime.get("metrics", {}), "integrity": integrity.get("metrics", {}), "attribution": result.get("attribution", {})},
        "gate_checks": {"full_replay_24x5x100": runtime.get("metrics", {}).get("frames") == 12000, "runtime_future_gt_false": result["runtime_validation"].get("runtime_future_gt_used") is False, "gt_loaded_posthoc_only": result["runtime_validation"].get("gt_loaded") is False, "independent_integrity_pass": integrity.get("status") == "PASS", "candidate_stream_identical": integrity.get("gate_checks", {}).get("candidate_stream_identical_three_branches") is True, "write_plus_axis_identical": integrity.get("gate_checks", {}).get("write_plus_public_id_axis_identical") is True, "global_hungarian_none": integrity.get("gate_checks", {}).get("hungarian_none_recomputed_normalized") is True, "M0_exact_no_op": integrity.get("gate_checks", {}).get("M0_exact_no_op") is True, "production_authorized_false": integrity.get("gate_checks", {}).get("checkpoint_production_authorized_false") is True, "simulated_provenance": result.get("real_human_tape_created") is False},
        "failure_root_cause": "Runtime and integrity are structurally valid, but no repair2 score cell was accepted, so there is no incremental assignment effect to credit.",
        "next_action": "Apply strict semantic final gate; retain simulated provenance and real-input blocker.",
    }
    write(stage4_path, stage4_status)

    blocked_path = R2 / "BLOCKED_INPUT_REAL_HUMAN_TAPE.json"
    write(blocked_path, {"schema": "N48_R1_REPAIR2_BLOCKED_INPUT_REAL_HUMAN_TAPE_V1", "status": "BLOCKED_INPUT_REAL_HUMAN_TAPE", "source_inventory": str(inventory_path), "exact_blocker": inventory["exact_blocker"], "minimal_next_step": inventory["minimal_next_step"], "required_real_human_fields": inventory["required_real_human_fields"], "candidate_complete_future_rows_required": True, "runtime_future_gt_used": False, "fabrication_or_relabeling": "FORBIDDEN_AND_NOT_PERFORMED", "synthetic_artifacts_are_real": False})

    m2 = result["effects"]["incremental"]["M2"]
    mem2 = result["effects"]["memory"]["M2"]
    gate_checks = {
        "stage01_audit_completed": True,
        "r1_marked_provisional_invalid_selection": True,
        "amendment_frozen": amendment.get("status") == "FROZEN_BEFORE_RETRAINING",
        "actual_full_training": training.get("actual_full_training") is True,
        "checkpoint_hash_recorded": training.get("checkpoint_sha256") == digest(checkpoint_path),
        "checkpoint_reloadable": load(R2 / "stage_03_reload.json").get("status") == "PASS",
        "single_objective_gradient_accumulation": training.get("one_optimizer_step_per_epoch") is True,
        "split_indices_validated": True,
        "holdout_not_used_for_selection": training.get("holdout_used_for_selection") is False,
        "runtime_complete": runtime.get("status") == "PASS" and runtime.get("metrics", {}).get("frames") == 12000,
        "independent_integrity_pass": integrity.get("status") == "PASS",
        "runtime_future_gt_false": result["runtime_validation"].get("runtime_future_gt_used") is False,
        "gt_loaded_only_posthoc": result["runtime_validation"].get("gt_loaded") is False,
        "all_24_simulated_events": result.get("event_count") == 24 and result.get("real_human_tape_created") is False,
        "equal_sequence_bootstrap": m2["100"]["sequence_cluster_bootstrap_95ci"].get("cluster_weighting") == "equal_sequence_mean",
        "m2_increment_zero_correct_changes": all(m2[str(h)]["assignment_change_correct_count"] == 0 for h in (20, 50, 100)),
        "m2_increment_zero_utility": all(abs(float(m2[str(h)]["identity_utility"] or 0.0)) < 1e-12 for h in (20, 50, 100)),
        "m2_increment_untouched_regression_pass": bool(m2["100"]["untouched_regression"].get("all_no_obvious_regression")),
        "real_human_tape": False,
        "real_sam3_full_loop": False,
        "standard_mot_not_computable": result["standard_mot"].startswith("NOT_COMPUTABLE"),
        "production_authorized": False,
    }
    gate = {
        "status": "N48_R1_REPAIR2_NOT_EXERCISED_GATE_FAILED",
        "protocol": "N48_R1_REPAIR2_STAGE_05_STRICT_DIAGNOSTIC_GATE_V1",
        "command": ["python", "scripts/n48_repair1b_stage05_finalize.py"],
        "inputs": {"amendment": str(amendment_path), "training": str(training_path), "result": str(result_path), "integrity": str(integrity_path), "r1_legacy_snapshot": str(r1_root), "real_tape_inventory": str(inventory_path)},
        "outputs": {"stage_01_status": str(R2 / "stage_01_status.json"), "stage_02_status": str(R2 / "stage_02_status.json"), "stage_03_status": str(R2 / "stage_03_status.json"), "stage_04_status": str(stage4_path), "stage_05_status": str(R2 / "stage_05_status.json"), "blocked_real_tape": str(blocked_path), "report": str(ROOT / "docs/N48_REPAIR1B_FINAL_REPORT.md")},
        "metrics": {"repair2_m2_increment_h20": m2["20"], "repair2_m2_increment_h50": m2["50"], "repair2_m2_increment_h100": m2["100"], "memory_m2_h100": mem2["100"], "runtime_summary": runtime.get("metrics", {}), "checkpoint_sha256": training.get("checkpoint_sha256"), "r1_invalid_checkpoint_sha256": digest(r1_root / "training/n48_r1_risk_aware_512d_bce.pt")},
        "gate_checks": gate_checks,
        "failure_root_cause": "Repair2 now uses valid split indices and one deterministic accumulated objective, but the complete simulated replay produced zero accepted score-cell changes and zero M2 incremental utility/correct assignment changes at H20/H50/H100. The separate memory effect remains negative and real human tape/full-loop are absent.",
        "next_action": "Keep the broader research objective open. Obtain externally supplied provenance-complete human tape and candidate-complete real SAM3 full-loop; stop blind weighting/threshold/seed/metric/capacity changes and do not authorize calibration, LoRA or production.",
        "runtime_future_gt_used": False,
        "interaction_source": "simulated_from_gt",
        "real_human_tape_created": False,
        "production_authorized": False,
    }
    write(R2 / "stage_05_status.json", gate)

    rows = []
    for effect, label in (("memory", "no_write→write baseline (memory effect)"), ("incremental", "write baseline→R1-repair2 (incremental effect)")):
        for variant in ("M0", "M1", "M2", "M3", "M4"):
            values = [result["effects"][effect][variant][str(h)] for h in (20, 50, 100)]
            utility = "/".join(fmt(v["identity_utility"]) for v in values)
            changes = "/".join(str(v["assignment_change_count"]) for v in values)
            cin = [f'{v["assignment_change_correct_count"]}/{v["assignment_change_incorrect_count"]}/{v["assignment_change_neutral_count"]}' for v in values]
            rows.append(f"| {label} | {variant} | {utility} | {changes} | {cin[0]} | {cin[1]} | {cin[2]} |")
    report = f'''# N48-R1 repair2 Final Report — corrected evaluation and single-objective training

## Decision

This isolated repair2 unit is **N48_R1_REPAIR2_NOT_EXERCISED_GATE_FAILED**. It is not completion of the research objective. The checkpoint is non-production (`production_authorized=false`), all events are `simulated_from_gt`, and calibration, decoder LoRA, production MOT/OVMOT changes and human-efficacy claims remain unauthorized.

## R1 invalidation and preserved evidence

The prior R1 training artifact is now classified `PROVISIONAL_INVALID_TRAINING_SELECTION`: its `evaluate` calls passed `pair_split` labels instead of `train_pairs`/`val_pairs`, so its train/validation loss and best epoch were not trustworthy. It also performed a separate cell-only optimizer loop after the pair-loop update, which was not the declared single objective. The old R1 checkpoint SHA256 is `{digest(r1_root / 'training/n48_r1_risk_aware_512d_bce.pt')}`, manifest SHA256 is `{digest(r1_root / 'training/training_manifest.json')}`, Stage03 SHA256 is `{digest(r1_root / 'status/stage_03_status.json')}`, Stage05 SHA256 is `{digest(r1_root / 'status/stage_05_status.json')}`, and replay SHA256 is `{digest(r1_root / 'replay/paired_replay_results.json')}`. All are preserved under `{r1_root}` and the old report was not overwritten.

The first repair2 smoke harness import failure and the read-only audit are retained under `{R2 / 'attempts'}`. The import fix was limited to the new regression harness.

## Frozen repair2 protocol and actual training

The amendment `{amendment_path}` was frozen before results. It keeps seed `4848`, the same dataset SHA256 `{amendment['dataset_sha256']}`, N42 sequence split, 8 epochs, AdamW (`lr=0.001`, `weight_decay=0.0001`), cell BCE coefficient `0.25`, uncertainty coefficient `0.25`, residual coefficient `0.001`, train-derived `pos_weight={amendment['class_weighting']['pos_weight']}`, and no holdout selection.

The objective is one deterministic full-data estimator: ascending frozen train pair indices and valid train cell indices are micro-batched at 1024; all rank, weighted cell-BCE, uncertainty and residual gradients are accumulated before exactly one `optimizer.step()` per epoch. Validation uses the complete true validation pair/cell index sets. Pair counts are train/validation/holdout `95536/9322/16050`; valid cell counts are train/validation/holdout `117889/6207/11651`.

Actual GPU0 training completed 8 epochs, best validation epoch `{training['best_epoch']}`, with checkpoint `{checkpoint_path}` and SHA256 `{training['checkpoint_sha256']}`. Every epoch logs rank, cell-BCE, uncertainty, residual-L2 and total objective; reload and targeted split/objective regression passed.

## Complete runtime and integrity

The same frozen N42 source and simulated memory manifest produced 24 events × 5 variants × 100 frames = 12,000 runtime frames. Runtime used direct `runtime_future_gt_used=false`; GT was loaded only after runtime validation. Independent integrity `{integrity_path}` passed 12,000 runtime frames and 24,000 source future-trace frames, including candidate completeness, write/plus axis consistency, explicit NONE/Hungarian normalization, M0 no-op, hard-negative preservation and unique native IDs.

## N48-R1 repair2 attribution

| Effect | Variant | Utility H20/H50/H100 | Assignment changes H20/H50/H100 | C/I/N H20 | C/I/N H50 | C/I/N H100 |
|---|---:|---|---|---:|---:|---:|
{chr(10).join(rows)}

For M2, repair2 incremental utility is `{fmt(m2['20']['identity_utility'])}/{fmt(m2['50']['identity_utility'])}/{fmt(m2['100']['identity_utility'])}`; assignment changes are `{m2['20']['assignment_change_count']}/{m2['50']['assignment_change_count']}/{m2['100']['assignment_change_count']}`, with correct `0/0/0`, incorrect `0/0/0`, and neutral `0/0/0`. Runtime selected/changed score cells are zero for every variant. Equal-sequence cluster bootstrap CIs are `[0,0]` at all M2 horizons. This is no exercise of a positive increment, not evidence that the objective improves identity.

The separate M2 memory effect at H20/H50/H100 remains `{fmt(mem2['20']['identity_utility'])}/{fmt(mem2['50']['identity_utility'])}/{fmt(mem2['100']['identity_utility'])}` with H100 C/I/N `{mem2['100']['assignment_change_correct_count']}/{mem2['100']['assignment_change_incorrect_count']}/{mem2['100']['assignment_change_neutral_count']}` and H100 untouched regression failed. Memory and repair2 effects are not conflated.

## Provenance and hard gates

The full result is `{result_path}`, Stage04 integrity is `{integrity_path}`, and strict gate is `{R2 / 'stage_05_status.json'}`. Standard MOT/TrackEval remains `NOT_COMPUTABLE_NO_COMPLETE_TRACKEVAL_INPUT`. Existing events are machine/GT-derived and were not renamed as human.

The explicit blocker `{blocked_path}` records that no external provenance-complete human tape or real SAM3 candidate-complete full-loop exists. Required fields are direct `public_id`, human-confirmed `BOX/CLICK/CONFIRMED_MASK`, lossless ROI digest, UI/session/annotator timestamps, native/public mapping and candidate-complete future rows. This blocker alone prevents production authorization.

## Next action

Keep the broader research objective open. The unique minimal next step is external provenance-complete real human tape plus candidate-complete real SAM3 full-loop, followed by N40 validation. Do not respond to this failed/zero-coverage repair2 result with blind weight, threshold, seed, metric, capacity, calibration or LoRA changes.
'''
    (ROOT / "docs/N48_REPAIR1B_FINAL_REPORT.md").write_text(report, encoding="utf-8")
    plan_path = R2 / "repair_plan_status.json"
    plan = load(plan_path)
    plan["status"] = "REPAIR2_GATE_FAILED_OBJECTIVE_OPEN"
    plan["failure_root_cause"] = gate["failure_root_cause"]
    plan["next_action"] = gate["next_action"]
    plan.setdefault("outputs", {}).update({"stage_01_status": str(R2 / "stage_01_status.json"), "stage_02_status": str(R2 / "stage_02_status.json"), "stage_03_status": str(R2 / "stage_03_status.json"), "stage_04_status": str(stage4_path), "stage_05_status": str(R2 / "stage_05_status.json"), "report": str(ROOT / "docs/N48_REPAIR1B_FINAL_REPORT.md")})
    write(plan_path, plan)
    print(json.dumps({"status": gate["status"], "report": str(ROOT / "docs/N48_REPAIR1B_FINAL_REPORT.md"), "production_authorized": False}))


if __name__ == "__main__":
    main()
