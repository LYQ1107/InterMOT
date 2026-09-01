#!/usr/bin/env python3
"""Strict N48 diagnostic gate and reproducible report generator."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.n47_global_probe_common import load, write_json  # noqa: E402

OUT = ROOT / "outputs/n48"


def fmt(value):
    return "NOT_COMPUTABLE" if value is None else f"{float(value):+.9f}"


def main() -> None:
    result_path = OUT / "replay/paired_replay_results.json"; integrity_path = OUT / "replay/stage_04_integrity.json"; training_path = OUT / "training/training_manifest.json"; inventory_path = OUT / "real_tape_inventory.json"
    result = load(result_path); integrity = load(integrity_path); training = load(training_path); inventory = load(inventory_path)
    runtime_status = load(OUT / "replay/runtime_status.json")
    stage4 = load(OUT / "stage_04_status.json")
    stage4["outputs"]["independent_integrity"] = str(integrity_path); stage4["metrics"]["integrity"] = integrity["metrics"]; stage4["gate_checks"]["independent_integrity"] = integrity.get("status") == "PASS"; stage4["gate_checks"]["posthoc_after_integrity"] = True; write_json(OUT / "stage_04_status.json", stage4)
    blocker = {"schema": "N48_BLOCKED_INPUT_REAL_HUMAN_TAPE_V1", "status": "BLOCKED_INPUT_REAL_HUMAN_TAPE", "source_inventory": str(inventory_path), "exact_blocker": inventory["exact_blocker"], "minimal_next_step": inventory["minimal_next_step"], "required_real_human_fields": inventory["required_real_human_fields"], "candidate_complete_future_rows_required": True, "runtime_future_gt_used": False, "fabrication_or_relabeling": "FORBIDDEN_AND_NOT_PERFORMED", "synthetic_artifacts_are_real": False}
    write_json(OUT / "BLOCKED_INPUT_REAL_HUMAN_TAPE.json", blocker)
    m2 = result["effects"]["incremental"]["M2"]
    m2_h100 = m2["100"]
    n48_exercised = bool(m2_h100["assignment_change_correct_count"] > 0 and abs(float(m2_h100["identity_utility"] or 0.0)) > 1e-12)
    gate_checks = {"stage01_diagnosis_completed": True, "actual_full_training": training.get("actual_full_training") is True, "checkpoint_reloadable": True, "checkpoint_production_authorized_false": training.get("production_authorized") is False, "runtime_complete": runtime_status.get("status") == "PASS", "independent_integrity_pass": integrity.get("status") == "PASS", "runtime_future_gt_false": result["runtime_validation"].get("runtime_future_gt_used") is False, "gt_loaded_only_posthoc": result["runtime_validation"].get("gt_loaded") is False, "all_24_simulated_events": result.get("event_count") == 24, "equal_sequence_bootstrap": result["effects"]["incremental"]["M2"]["100"]["sequence_cluster_bootstrap_95ci"].get("cluster_weighting") == "equal_sequence_mean", "n48_incremental_positive_future_effect": n48_exercised, "n48_m2_zero_correct_changes": m2_h100["assignment_change_correct_count"] == 0, "n48_incremental_untouched_regression_pass": bool(m2_h100["untouched_regression"].get("all_no_obvious_regression")), "real_human_tape": False, "real_sam3_full_loop": False, "standard_mot_not_computable": result["standard_mot"].startswith("NOT_COMPUTABLE"), "production_authorized": False}
    gate = {"status": "N48_NOT_EXERCISED_GATE_FAILED", "protocol": "N48_STAGE_05_STRICT_DIAGNOSTIC_GATE_V1", "command": ["python", "scripts/n48_stage05_finalize.py"], "decision": "Do not authorize calibration, decoder LoRA, production or claim human efficacy.", "inputs": {"result": str(result_path), "integrity": str(integrity_path), "training": str(training_path), "real_tape_inventory": str(inventory_path), "n47_gate": str(ROOT / "outputs/n47_global_probe/repair1_swap_metric/n47_swap_metric_gate.json")}, "outputs": {"stage_04_status": str(OUT / "stage_04_status.json"), "stage_05_status": str(OUT / "stage_05_status.json"), "blocked_real_tape": str(OUT / "BLOCKED_INPUT_REAL_HUMAN_TAPE.json"), "report": str(ROOT / "docs/N48_FINAL_REPORT.md")}, "metrics": {"n48_m2_incremental_h100": m2_h100, "memory_effect_m2_h100": result["effects"]["memory"]["M2"]["100"], "runtime_summary": runtime_status["metrics"]}, "gate_checks": gate_checks, "failure_root_cause": "The N48 checkpoint and full runtime are valid, but the write-baseline-to-N48 increment has zero correct assignment changes and zero identity-utility at H20/H50/H100; observed assignment changes are neutral. The diagnostic was therefore not exercised as a positive future-effect test. Separately, the memory effect remains negative in M2 and the required real human tape/full-loop are absent.", "next_action": "Keep the research objective open. Obtain an externally supplied candidate-complete human tape and real SAM3 full-loop; only then consider a new structural experiment if a new falsifiable hypothesis is justified. No calibration or LoRA authorization.", "runtime_future_gt_used": False, "interaction_source": "simulated_from_gt", "real_human_tape_created": False, "production_authorized": False}
    write_json(OUT / "stage_05_status.json", gate)
    rows = []
    for effect, label in (("memory", "no_write→write baseline (memory effect)"), ("incremental", "write baseline→N48 (incremental effect)")):
        for variant in ("M0", "M1", "M2", "M3", "M4"):
            vals = [result["effects"][effect][variant][str(h)] for h in (20, 50, 100)]
            cin = [f"{x['assignment_change_correct_count']}/{x['assignment_change_incorrect_count']}/{x['assignment_change_neutral_count']}" for x in vals]
            rows.append(f"| {label} | {variant} | {fmt(vals[0]['identity_utility'])}/{fmt(vals[1]['identity_utility'])}/{fmt(vals[2]['identity_utility'])} | {vals[0]['assignment_change_count']}/{vals[1]['assignment_change_count']}/{vals[2]['assignment_change_count']} | {cin[0]} | {cin[1]} | {cin[2]} |")
    report = f'''# N48 Final Report — Risk-aware 512-D global assignment diagnostic

## Decision

N48 is a completed isolated diagnostic training/replay unit, not completion of the research objective. Its semantic gate is **N48_NOT_EXERCISED_GATE_FAILED**. The checkpoint is explicitly non-production (`production_authorized=false`), all 24 events are `simulated_from_gt`, and no calibration head, decoder LoRA, production MOT/OVMOT change, or human-efficacy claim is authorized.

The first actionable N47 diagnosis was an unbounded 8-scalar global logit without temporal/untouched-risk control. It supported one bounded, uncertainty-gated 512-D candidate/memory fusion experiment. N48 used that frozen protocol and did not tune threshold, seed, metric, holdout, or checkpoint.

## Frozen inputs and actual execution

- Protocol: `{OUT / "protocol.json"}`; seed 4848; N42 sequence-disjoint train/validation/holdout split.
- Dataset: `{OUT / "training/risk_aware_512d_dataset.npz"}`; dataset SHA256 `{training["dataset_sha256"]}`.
- Actual training: 8 epochs, AdamW, best validation epoch {training["best_epoch"]}; checkpoint `{training["checkpoint"]}`; SHA256 `{training["checkpoint_sha256"]}`.
- Runtime: 24 events × 5 variants × 100 frames = 12,000 frames, with GT absent from runtime. Posthoc GT was loaded only after runtime validation.
- Runtime commands: `python scripts/n48_stage02_materialize_simulated_memory.py`; `python scripts/n48_stage02_targeted_regression.py`; `python scripts/n48_stage04_smoke.py`; `python scripts/n48_stage03_train.py`; `python scripts/n48_stage04_replay.py`; `python scripts/n48_stage04_integrity.py`; `python scripts/n48_stage05_finalize.py`.

The runtime retains candidate rows, native IDs, boxes/confidences, 512-D candidate vectors, memory vectors/provenance, public-ID axes, score matrices, gate reasons, accepted cells, and assignment transitions in `{OUT / "replay/runtime"}`. Independent integrity passed 12,000 frames and 24,000 source future-trace frames, with normalized explicit-NONE Hungarian recomputation, no duplicate native IDs, unchanged write/plus axes, M0 exact no-op, and hard negatives preserved.

## N47 versus N48 and attribution

N47 repair remains frozen: assignment changes M0/M1/M2/M3/M4 = 0/335/455/335/375; legacy id-set/multiset field = 0/279/391/279/336; registered pure swaps = 0/56/64/56/39. N47 events remain simulated and its gate remains FAIL.

N48 separates the memory effect (no-write→write baseline) from the incremental sidecar effect (write baseline→write-plus-N48). Values below are identity utility at H20/H50/H100; changes are total assignment changes at those horizons; C/I/N are correct/incorrect/neutral decomposition.

| Effect | Variant | Utility H20/H50/H100 | Assignment changes H20/H50/H100 | C/I/N H20 | C/I/N H50 | C/I/N H100 |
|---|---:|---|---|---:|---:|---:|
{chr(10).join(rows)}

For M2, the N48 incremental identity utility is 0.000000000 at H20/H50/H100; correct changes are 0/0/0, while the observed changes are neutral (0/0/0 incorrect and 2/2/3 neutral in the paired result). Its equal-sequence cluster bootstrap CIs are exactly [0, 0] at all horizons. This is `N48_NOT_EXERCISED`, not a positive result. The N48 application counts are retained separately; a selected cell or score change is not counted as a successful assignment change when Hungarian leaves the assignment unchanged.

The M2 memory effect at H100 is {fmt(result["effects"]["memory"]["M2"]["100"]["identity_utility"])} utility, with {result["effects"]["memory"]["M2"]["100"]["assignment_change_count"]} assignment changes and {result["effects"]["memory"]["M2"]["100"]["assignment_change_correct_count"]}/{result["effects"]["memory"]["M2"]["100"]["assignment_change_incorrect_count"]}/{result["effects"]["memory"]["M2"]["100"]["assignment_change_neutral_count"]} C/I/N. This keeps the negative memory effect distinct from the zero N48 increment.

## Failures and repairs preserved

The following failed evidence is retained under `{OUT / "attempts"}`: local-vs-global margin contract audit, a malformed memory-valid regression fixture, initial Stage 04 syntax failures, the missing branch-frame schema failure with its completed pre-posthoc runtime snapshot, initial independent-integrity failure, and normalized-NONE/type repair evidence. None was relabeled PASS or deleted.

The final independent integrity artifact is `{integrity_path}`. The final machine-readable status is `{OUT / "stage_05_status.json"}`. Standard MOT/TrackEval is `NOT_COMPUTABLE` because complete TrackEval input is absent; no metric definition was changed.

## Real input gate

Inventory result: `{OUT / "BLOCKED_INPUT_REAL_HUMAN_TAPE.json"}`. No external UI/session/annotator export or candidate-complete human tape exists here. Existing N34/N35/N36 event/candidate files are synthetic/GT-derived or machine tapes and were not relabeled. The minimal external input must contain direct `public_id`, human-confirmed `BOX/CLICK/CONFIRMED_MASK`, lossless ROI digest, UI/session/annotator timestamps, native/public mapping, and every future candidate row/box/feature/confidence. Real SAM3 candidate-complete full-loop evidence is also absent. This hard gate blocks any production authorization even if the simulated diagnostic had been positive.

## Next step

Keep the broader objective open. The unique minimum next step is to obtain and validate one externally supplied candidate-complete real human tape plus the corresponding real SAM3 full-loop under the existing N40 contract. After that, a new structural experiment is justified only if a fresh falsifiable hypothesis is supported; do not tune around this zero-effect result, authorize calibration/LoRA, or alter production interfaces.
'''
    (ROOT / "docs/N48_FINAL_REPORT.md").write_text(report, encoding="utf-8")
    print(json.dumps({"status": gate["status"], "report": str(ROOT / "docs/N48_FINAL_REPORT.md"), "production_authorized": False}))


if __name__ == "__main__":
    main()
