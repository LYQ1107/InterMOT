#!/usr/bin/env python3
"""Finalize the isolated N48-R1 diagnostic without authorizing production."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/n48/repair1"


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fmt(value):
    return "NOT_COMPUTABLE" if value is None else f"{float(value):+.9f}"


def main() -> None:
    result_path = OUT / "replay/paired_replay_results.json"
    integrity_path = OUT / "replay/stage_04_integrity.json"
    training_path = OUT / "training/training_manifest.json"
    stage4_path = OUT / "stage_04_status.json"
    runtime_status_path = OUT / "replay/runtime_status.json"
    inventory_path = ROOT / "outputs/n48/real_tape_inventory.json"
    amendment_path = OUT / "protocol_amendment.json"
    result = load(result_path)
    integrity = load(integrity_path)
    training = load(training_path)
    stage4 = load(stage4_path)
    runtime_status = load(runtime_status_path)
    amendment = load(amendment_path)
    inventory = load(inventory_path)

    stage4.setdefault("outputs", {})["independent_integrity"] = str(integrity_path)
    stage4.setdefault("metrics", {})["integrity"] = integrity.get("metrics", {})
    stage4.setdefault("gate_checks", {}).update({
        "independent_integrity": integrity.get("status") == "PASS",
        "posthoc_after_integrity": result.get("runtime_validation", {}).get("gt_loaded") is False,
    })
    stage4["next_action"] = "Proceed to strict R1 semantic gate; no calibration, LoRA or production authorization is permitted."
    write(stage4_path, stage4)

    blocked_path = OUT / "BLOCKED_INPUT_REAL_HUMAN_TAPE.json"
    blocked = {
        "schema": "N48_R1_BLOCKED_INPUT_REAL_HUMAN_TAPE_V1",
        "status": "BLOCKED_INPUT_REAL_HUMAN_TAPE",
        "source_inventory": str(inventory_path),
        "exact_blocker": inventory["exact_blocker"],
        "minimal_next_step": inventory["minimal_next_step"],
        "required_real_human_fields": inventory["required_real_human_fields"],
        "candidate_complete_future_rows_required": True,
        "runtime_future_gt_used": False,
        "fabrication_or_relabeling": "FORBIDDEN_AND_NOT_PERFORMED",
        "synthetic_artifacts_are_real": False,
    }
    write(blocked_path, blocked)

    m2_inc = result["effects"]["incremental"]["M2"]
    m2_mem = result["effects"]["memory"]["M2"]
    h100 = m2_inc["100"]
    h20 = m2_inc["20"]
    h50 = m2_inc["50"]
    inc_zero_correct = all(m2_inc[str(h)]["assignment_change_correct_count"] == 0 for h in (20, 50, 100))
    inc_zero_utility = all(abs(float(m2_inc[str(h)]["identity_utility"] or 0.0)) < 1e-12 for h in (20, 50, 100))
    sequence_ci = h100["sequence_cluster_bootstrap_95ci"]
    gate_checks = {
        "stage01_diagnosis_completed": (OUT / "stage_01_status.json").exists(),
        "stage02_amendment_frozen": amendment.get("schema") == "N48_R1_PROTOCOL_AMENDMENT_V1" and amendment.get("status") == "FROZEN_BEFORE_RETRAINING",
        "actual_full_training": training.get("actual_full_training") is True,
        "checkpoint_hash_recorded": bool(training.get("checkpoint_sha256")),
        "checkpoint_reloadable": (OUT / "stage_03_reload.json").exists() and load(OUT / "stage_03_reload.json").get("status") == "PASS",
        "checkpoint_production_authorized_false": training.get("production_authorized") is False,
        "runtime_complete": runtime_status.get("status") == "PASS" and runtime_status.get("metrics", {}).get("frames") == 12000,
        "independent_integrity_pass": integrity.get("status") == "PASS",
        "runtime_future_gt_false": result["runtime_validation"].get("runtime_future_gt_used") is False,
        "gt_loaded_only_posthoc": result["runtime_validation"].get("gt_loaded") is False,
        "all_24_simulated_events": result.get("event_count") == 24 and result.get("real_human_tape_created") is False,
        "candidate_complete": result["runtime_validation"].get("candidate_complete") is True,
        "equal_sequence_bootstrap": sequence_ci.get("cluster_weighting") == "equal_sequence_mean",
        "m2_increment_zero_correct_changes": inc_zero_correct,
        "m2_increment_zero_identity_utility": inc_zero_utility,
        "m2_increment_untouched_regression_pass": bool(h100["untouched_regression"].get("all_no_obvious_regression")),
        "real_human_tape": False,
        "real_sam3_full_loop": False,
        "standard_mot_not_computable": result["standard_mot"].startswith("NOT_COMPUTABLE"),
        "production_authorized": False,
    }
    gate = {
        "status": "N48_R1_NOT_EXERCISED_GATE_FAILED",
        "protocol": "N48_R1_STAGE_05_STRICT_DIAGNOSTIC_GATE_V1",
        "command": ["python", "scripts/n48_repair1_stage05_finalize.py"],
        "decision": "Do not authorize calibration, decoder LoRA, production or claim human efficacy.",
        "inputs": {
            "protocol_amendment": str(amendment_path),
            "result": str(result_path),
            "integrity": str(integrity_path),
            "training": str(training_path),
            "real_tape_inventory": str(inventory_path),
            "parent_r0_protocol": str(ROOT / "outputs/n48/protocol.json"),
            "parent_r0_checkpoint": str(ROOT / "outputs/n48/training/n48_risk_aware_512d.pt"),
        },
        "outputs": {
            "stage_02_status": str(OUT / "stage_02_status.json"),
            "stage_04_status": str(stage4_path),
            "stage_05_status": str(OUT / "stage_05_status.json"),
            "blocked_real_tape": str(blocked_path),
            "report": str(ROOT / "docs/N48_REPAIR1_FINAL_REPORT.md"),
        },
        "metrics": {
            "n48_r1_m2_increment_h20": h20,
            "n48_r1_m2_increment_h50": h50,
            "n48_r1_m2_increment_h100": h100,
            "memory_effect_m2_h100": m2_mem["100"],
            "runtime_summary": runtime_status.get("metrics", {}),
            "checkpoint_sha256": training.get("checkpoint_sha256"),
            "r0_checkpoint_sha256": training.get("parent_r0_checkpoint_sha256"),
        },
        "gate_checks": gate_checks,
        "failure_root_cause": "The protocol-compliant R1 checkpoint and complete simulated replay are valid, but M2 write-baseline→R1 has zero correct assignment changes and zero identity-utility at H20/H50/H100; observed changes are neutral and H100 untouched regression fails. The independent M2 memory effect remains negative. Real human tape and real SAM3 full-loop are absent.",
        "next_action": "Keep the broader research objective open. Obtain externally supplied provenance-complete real human tape and candidate-complete real SAM3 full-loop; do not add weighting, tune thresholds, authorize calibration/LoRA, or treat simulated R1 as human efficacy.",
        "runtime_future_gt_used": False,
        "interaction_source": "simulated_from_gt",
        "real_human_tape_created": False,
        "production_authorized": False,
    }
    write(OUT / "stage_05_status.json", gate)

    rows = []
    for effect, label in (("memory", "no_write→write baseline (memory effect)"), ("incremental", "write baseline→N48-R1 (incremental effect)")):
        for variant in ("M0", "M1", "M2", "M3", "M4"):
            values = [result["effects"][effect][variant][str(h)] for h in (20, 50, 100)]
            changes = "/".join(str(v["assignment_change_count"]) for v in values)
            cin = [f'{v["assignment_change_correct_count"]}/{v["assignment_change_incorrect_count"]}/{v["assignment_change_neutral_count"]}' for v in values]
            utilities = "/".join(fmt(v["identity_utility"]) for v in values)
            rows.append(f"| {label} | {variant} | {utilities} | {changes} | {cin[0]} | {cin[1]} | {cin[2]} |")

    report = f'''# N48-R1 Final Report — protocol-compliant cell-BCE repair

## Decision

N48-R1 is an isolated, protocol-compliant diagnostic training/replay unit. It is **N48_R1_NOT_EXERCISED_GATE_FAILED**, not completion of the research objective. The checkpoint records `production_authorized=false`; all 24 events are `simulated_from_gt`. No calibration head, decoder LoRA, production MOT/OVMOT change, or human-efficacy claim is authorized.

## R0 mismatch and R1 amendment

The parent N48-R0 protocol `{ROOT / "outputs/n48/protocol.json"}` required pairwise softplus ranking, weighted valid-cell BCE, uncertainty BCE, and fixed residual L2. R0's training code and checkpoint omitted cell BCE, so R0 is retained as a `protocol-mismatch diagnostic attempt`, not protocol-compliant training. R0 protocol SHA256 is `{sha256(ROOT / "outputs/n48/protocol.json")}` and R0 checkpoint SHA256 is `{sha256(ROOT / "outputs/n48/training/n48_risk_aware_512d.pt")}`; neither was overwritten.

R1 uses the frozen same sequence split, seed `4848`, data and 8-epoch budget. The amendment `{amendment_path}` froze before training:

- cell target: `1` for frozen IoU ≥ 0.5, `0` for frozen IoU ≤ 0.3, and `-1` excluded for 0.3 < IoU < 0.5 or unavailable;
- train counts: positive `{amendment['class_weighting']['positive_count']}`, negative `{amendment['class_weighting']['negative_count']}`;
- inverse-frequency weights: positive `{amendment['class_weighting']['w_pos']}`, negative `{amendment['class_weighting']['w_neg']}`; BCE `pos_weight={training['pos_weight']}`;
- objective: `rank + 0.25*weighted_cell_BCE + 0.25*uncertainty_BCE + 0.001*residual_L2`; holdout was not used for selection.

## Actual execution and integrity

R1 actual training completed for 8 epochs with AdamW. Best epoch was `{training['best_epoch']}`; checkpoint `{training['checkpoint']}` has SHA256 `{training['checkpoint_sha256']}`. The manifest logs rank, cell-BCE, uncertainty-BCE, residual-L2, and total objective for every epoch; reload/smoke passed and the checkpoint remains non-production.

The paired replay used 24 events × 5 variants × 100 future frames = 12,000 runtime frames. Runtime used `future_gt=false`; GT was loaded only after runtime validation. Independent integrity `{integrity_path}` passed: candidate-complete rows, explicit NONE/dummy normalization, global Hungarian recomputation, M0 no-op, no duplicate native IDs, and candidate/axis consistency passed.

## N47/R0/R1 attribution

N47 repair remains frozen with assignment changes M0/M1/M2/M3/M4 = `0/335/455/335/375`, legacy id-set changes `0/279/391/279/336`, and registered pure swaps `0/56/64/56/39`. N47 and N48-R1 events remain simulated.

R1 separates no-write→write memory effect from write-baseline→R1 incremental effect. Utility is H20/H50/H100; C/I/N is correct/incorrect/neutral at each horizon.

| Effect | Variant | Utility H20/H50/H100 | Assignment changes H20/H50/H100 | C/I/N H20 | C/I/N H50 | C/I/N H100 |
|---|---:|---|---|---:|---:|---:|
{chr(10).join(rows)}

For M2, R1 incremental utility is `{fmt(h20['identity_utility'])}/{fmt(h50['identity_utility'])}/{fmt(h100['identity_utility'])}`. Correct changes are `{h20['assignment_change_correct_count']}/{h50['assignment_change_correct_count']}/{h100['assignment_change_correct_count']}`; incorrect changes are `{h20['assignment_change_incorrect_count']}/{h50['assignment_change_incorrect_count']}/{h100['assignment_change_incorrect_count']}`; neutral changes are `{h20['assignment_change_neutral_count']}/{h50['assignment_change_neutral_count']}/{h100['assignment_change_neutral_count']}`. The equal-sequence bootstrap CIs and full per-event/per-frame attribution are in `{result_path}`. This is `N48_NOT_EXERCISED`, not a positive efficacy result. The M2 memory effect at H100 is `{fmt(m2_mem['100']['identity_utility'])}` with `{m2_mem['100']['assignment_change_count']}` changes and C/I/N `{m2_mem['100']['assignment_change_correct_count']}/{m2_mem['100']['assignment_change_incorrect_count']}/{m2_mem['100']['assignment_change_neutral_count']}`.

## Stage01 repair and preserved failures

The repaired Stage01 diagnosis enforces per-sequence closure `assignment_changes == correct_changes + incorrect_changes + neutral_changes`, uses changed-row-only NONE accounting, separates frame and target names, and marks mixed-unit `oracle_required_total_score_gap` invalid/non-comparable. Legacy diagnosis/refined JSON and hashes remain under `{OUT / 'legacy_snapshots'}`. The initial R1 replay protocol-loader failure is retained under `{OUT / 'attempts/replay_checkpoint_protocol_mismatch.json'}`; the smallest loader allow-list repair was smoke/reload-regressed before rerun.

R0's protocol, checkpoint, stage status and paired result remain untouched. Standard MOT/TrackEval is `NOT_COMPUTABLE_NO_COMPLETE_TRACKEVAL_INPUT`; no metric definition was changed. All selected/changed cell counts remain separate from assignment changes.

## Real-input hard gate

`{blocked_path}` preserves the N40 blocker. No external UI/session/annotator export or provenance-complete human tape was found. Existing machine/GT-derived events were not fabricated, relabeled, or promoted. Required external fields are direct `public_id`, human-confirmed `BOX/CLICK/CONFIRMED_MASK`, lossless ROI digest, UI/session/annotator timestamps, native/public mapping, and candidate-complete future rows. Real SAM3 candidate-complete full-loop evidence is also absent.

## Next action

Keep the broader research objective open. The unique minimum next step is to obtain and validate provenance-complete real human tape plus real SAM3 candidate-complete full-loop evidence. Do not respond to this failed simulated increment with blind weighting, threshold changes, seed/metric changes, calibration, LoRA, or production integration.
'''
    (ROOT / "docs/N48_REPAIR1_FINAL_REPORT.md").write_text(report, encoding="utf-8")

    plan_path = OUT / "repair_plan_status.json"
    plan = load(plan_path)
    plan["status"] = "R1_GATE_FAILED_OBJECTIVE_OPEN"
    plan["failure_root_cause"] = gate["failure_root_cause"]
    plan["next_action"] = gate["next_action"]
    plan.setdefault("outputs", {})["stage_02_status"] = str(OUT / "stage_02_status.json")
    plan["outputs"]["stage_05_status"] = str(OUT / "stage_05_status.json")
    plan["outputs"]["report"] = str(ROOT / "docs/N48_REPAIR1_FINAL_REPORT.md")
    write(plan_path, plan)
    print(json.dumps({"status": gate["status"], "report": str(ROOT / "docs/N48_REPAIR1_FINAL_REPORT.md"), "production_authorized": False}))


if __name__ == "__main__":
    main()
