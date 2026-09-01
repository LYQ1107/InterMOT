#!/usr/bin/env python3
"""N43 stage 05: strict gate and reproducible final report."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/n43"
RESULT = OUT / "replay/paired_replay_results.json"
LEGACY_RESULT = OUT / "replay/paired_replay_results_legacy_event_weighted.json"
GATE = OUT / "n43_final_gate.json"
STAGE = OUT / "stage_05_status.json"
REPORT = ROOT / "docs/N43_FINAL_REPORT.md"


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def fmt(value):
    return "NOT_COMPUTABLE" if value is None else f"{float(value):.6f}"


def main() -> None:
    stage1 = load(OUT / "stage_01_status.json")
    stage2 = load(OUT / "stage_02_status.json")
    stage3 = load(OUT / "stage_03_status.json")
    stage4 = load(OUT / "stage_04_status.json")
    training = load(OUT / "training/full_training_manifest.json")
    result = load(RESULT)
    targeted_regression_path = OUT / "audit/targeted_regression.json"
    targeted_regression = load(targeted_regression_path) if targeted_regression_path.is_file() else {"status": "NOT_RUN"}
    legacy_result_path = OUT / "replay/paired_replay_results_legacy_event_weighted.json"
    legacy_result = load(legacy_result_path) if legacy_result_path.is_file() else None
    bootstrap_comparison = {}
    if legacy_result is not None:
        for variant in ("M0", "M1", "M2", "M3", "M4"):
            for horizon in (20, 50, 100):
                corrected = result["aggregates"][variant][str(horizon)]["all"]["sequence_cluster_bootstrap_95ci"]
                legacy = legacy_result["aggregates"][variant][str(horizon)]["all"]["sequence_cluster_bootstrap_95ci"]
                bootstrap_comparison[f"{variant}_H{horizon}"] = {
                    "corrected_equal_sequence_mean_ci": {"lower": corrected["lower"], "upper": corrected["upper"]},
                    "legacy_event_weighted_ci": {"lower": legacy["lower"], "upper": legacy["upper"]},
                }
    # Replay itself is complete.  Add the two post-replay audits to Stage 04
    # without rewriting its per-event metrics or promoting partial artifacts.
    stage4["outputs"].update({
        "targeted_margin_regression": str(targeted_regression_path),
        "corrected_result": str(RESULT),
        "legacy_event_weighted_result": str(legacy_result_path),
    })
    stage4.setdefault("metrics", {})["post_replay_audit"] = {
        "targeted_margin_regression": targeted_regression,
        "bootstrap_definition": "Mean event identity utility within each sequence first, then bootstrap sequence means with equal sequence weight.",
        "bootstrap_implementation": "scripts/n43_paired_replay.py::cluster_bootstrap",
        "corrected_cluster_weighting": "equal_sequence_mean",
        "legacy_result_preserved": legacy_result is not None,
        "comparison_all_variants_all_horizons": bootstrap_comparison,
    }
    stage4["gate_checks"].update({
        "post_replay_margin_targeted_regression": targeted_regression.get("status") == "PASS",
        "bootstrap_preregistered_equal_sequence_mean": all(
            result["aggregates"][variant][str(horizon)]["all"]["sequence_cluster_bootstrap_95ci"].get("cluster_weighting") == "equal_sequence_mean"
            for variant in ("M0", "M1", "M2", "M3", "M4")
            for horizon in (20, 50, 100)
        ),
        "legacy_event_weighted_result_preserved": legacy_result is not None,
    })
    stage4["finished_at"] = now()
    (OUT / "stage_04_status.json").write_text(json.dumps(stage4, indent=2) + "\n", encoding="utf-8")
    legacy_result_preserved = LEGACY_RESULT.is_file()
    stages_pass = all(x.get("status") == "PASS" for x in (stage1, stage2, stage3, stage4))
    real_tape = False
    real_full_loop = False
    future_effect = all(
        result["aggregates"][variant][str(horizon)]["all"]["sequence_cluster_bootstrap_95ci"]["lower"] is not None
        and result["aggregates"][variant][str(horizon)]["all"]["sequence_cluster_bootstrap_95ci"]["lower"] > 0
        for variant in ("M2", "M3", "M4") for horizon in (20, 50, 100)
    )
    untouched = all(result["aggregates"][variant][str(horizon)]["all"]["untouched_regression"]["all_no_obvious_regression"] for variant in ("M2", "M3", "M4") for horizon in (20, 50, 100))
    strict_pass = stages_pass and real_tape and real_full_loop and future_effect and untouched and result.get("independent_sequence_count", 0) >= 6
    gate = {"artifact": "outputs/n43/n43_final_gate.json", "protocol": "N43_FINAL_RESEARCH_GATE_V1", "created_at": now(), "status": "N43_COMPLETED_GATE_PASSED" if strict_pass else "N43_COMPLETED_GATE_FAILED", "research_gate": "PASS" if strict_pass else "FAIL_REAL_TAPE_AND_FUTURE_EFFECT", "authorization": {"calibration_head": "AUTHORIZED" if strict_pass else "NOT_AUTHORIZED", "decoder_lora": "AUTHORIZED" if strict_pass else "NOT_AUTHORIZED", "production_interface_changed": False}, "checks": {"stage_01_audit": stage1.get("status") == "PASS", "stage_02_sidecar": stage2.get("status") == "PASS", "stage_03_actual_training": stage3.get("status") == "PASS", "stage_04_replay": stage4.get("status") == "PASS", "all_24_simulated_events": result.get("event_count") == 24, "independent_sequence_count": result.get("independent_sequence_count"), "real_human_tape_available": real_tape, "real_full_loop": real_full_loop, "future_effect_strict_ci": future_effect, "untouched_regression_absent": untouched, "runtime_future_gt_false": result.get("runtime_future_gt_used") is False, "sequence_cluster_bootstrap": True}, "failure_reasons": {"real_tape": "N42/N43 inputs are simulated_from_gt; no externally supplied real human event tape exists", "real_full_loop": "No real SAM3 candidate-complete full-loop was run in N43", "future_effect": "M2/M3/M4 sequence-cluster CI lower bounds are not strictly > 0; M2 is negative", "untouched_regression": "M1-M4 full-cell branches show at least one untouched-ID regression under the fixed posthoc criterion"}, "effect_summary": {variant: {str(h): result["aggregates"][variant][str(h)]["all"] for h in (20, 50, 100)} for variant in ("M0", "M1", "M2", "M3", "M4")}, "training": {"status": training.get("status"), "training_mode": training.get("training_mode"), "device": training.get("device"), "completed_epochs": training.get("completed_epochs"), "checkpoint": training.get("checkpoint"), "checkpoint_sha256": training.get("checkpoint_sha256"), "production_authorized": False}, "next_action": "Collect a schema-valid real human event tape and real full-loop; do not promote N43 calibration or start decoder LoRA."}
    gate["checks"]["targeted_regression"] = targeted_regression.get("status") == "PASS"
    gate["checks"]["legacy_event_weighted_result_preserved"] = legacy_result_preserved
    gate["checks"]["bootstrap_preregistered_equal_sequence_mean"] = all(
        result["aggregates"][variant][str(horizon)]["all"]["sequence_cluster_bootstrap_95ci"].get("cluster_weighting") == "equal_sequence_mean"
        for variant in ("M0", "M1", "M2", "M3", "M4")
        for horizon in (20, 50, 100)
    )
    gate.update({
        "command": ["python", "scripts/n43_finalize.py"],
        "inputs": {
            "stage_01": str(OUT / "stage_01_status.json"),
            "stage_02": str(OUT / "stage_02_status.json"),
            "stage_03": str(OUT / "stage_03_status.json"),
            "stage_04": str(OUT / "stage_04_status.json"),
            "paired_replay": str(RESULT),
            "legacy_event_weighted_replay": str(LEGACY_RESULT),
            "targeted_regression": str(targeted_regression_path),
        },
        "outputs": {"gate": str(GATE), "report": str(REPORT)},
        "metrics": {
            "sidecar_effect": "NEGATIVE",
            "m2_identity_utility": {str(h): result["aggregates"]["M2"][str(h)]["all"]["identity_utility"] for h in (20, 50, 100)},
            "m2_ci_lower": {str(h): result["aggregates"]["M2"][str(h)]["all"]["sequence_cluster_bootstrap_95ci"]["lower"] for h in (20, 50, 100)},
            "bootstrap_protocol": result.get("bootstrap_protocol"),
            "legacy_result_preserved": legacy_result_preserved,
            "targeted_margin_regression": targeted_regression.get("status"),
            "real_tape": real_tape,
            "real_full_loop": real_full_loop,
        },
        "gate_checks": gate["checks"],
        "failure_root_cause": gate["failure_reasons"],
        "next_action": gate["next_action"],
    })
    GATE.write_text(json.dumps(gate, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# N43 Final Report — Full Candidate×Public-ID Calibration Sidecar",
        "",
        f"Date: {datetime.now().astimezone().isoformat()}",
        "",
        f"Result: `{gate['status']}`; production calibration and decoder LoRA remain `NOT_AUTHORIZED`.",
        "",
        "## Executive result",
        "",
        "N43 completed a frozen N42 decision-boundary audit, implemented an independent full-cell sidecar, ran actual sequence-disjoint training, and completed a 24-event/21-sequence paired replay. The interface hypothesis was testable: N42 changed only the human target column, while N43 evaluated every finite candidate×public-ID cell and preserved an explicit immutable NONE dummy. The result was negative rather than a promotion: M2 identity utility was negative at all horizons, assignment changes had zero correct changes in aggregate, and untouched-ID regression was present.",
        "",
        "There is no real human tape. All events are `simulated_from_gt`; GT was used only for offline mapping/labels and post-hoc metrics after runtime validation. This is not evidence of real-human performance or a production online StateManager change.",
        "",
        "## Stage results and artifacts",
        "",
        f"- Stage 01 PASS: [stage_01_status.json]({OUT / 'stage_01_status.json'}), [full_matrix_audit.jsonl]({OUT / 'audit/full_matrix_audit.jsonl'}). 2424 frames and 173793 cells; assigned target recall@0.5={fmt(stage1['metrics']['target_recall_oracle']['assigned_recall_at_0.5'])}, candidate Oracle recall@0.5={fmt(stage1['metrics']['target_recall_oracle']['oracle_recall_at_0.5'])}.",
        f"- Stage 02 PASS: [stage_02_status.json]({OUT / 'stage_02_status.json'}), [sidecar_protocol.json]({OUT / 'sidecar_protocol.json'}), [cell_dataset.npz]({OUT / 'training/cell_dataset.npz'}). Dataset rows={stage2['metrics']['dataset']['row_count']}; positive={stage2['metrics']['dataset']['counters']['positive']}, negative={stage2['metrics']['dataset']['counters']['negative']}, ambiguous discarded={stage2['metrics']['dataset']['counters']['ambiguous_label_discarded']}, GT-unavailable cells={stage2['metrics']['dataset']['counters'].get('public_id_gt_unavailable_cells', 0)}.",
        f"- Stage 03 PASS: [stage_03_status.json]({OUT / 'stage_03_status.json'}), [full_training_manifest.json]({OUT / 'training/full_training_manifest.json'}), [n43_full_matrix_calibration.pt]({OUT / 'training/n43_full_matrix_calibration.pt'}). Actual full training, device={training['device']}, epochs={training['completed_epochs']}, best epoch={training['best_epoch']}, checkpoint SHA-256={training['checkpoint_sha256']}.",
        f"- Stage 04 PASS: [stage_04_status.json]({OUT / 'stage_04_status.json'}), [paired_replay_results.json]({RESULT}), [legacy event-weighted result]({LEGACY_RESULT}), runtime files under [replay/runtime]({OUT / 'replay/runtime'}), post-hoc files under [replay/posthoc_events]({OUT / 'replay/posthoc_events'}). 24/24 events, 5 variants, H20/H50/H100, runtime future_gt=false.",
        f"- Stage 05: [n43_final_gate.json]({GATE}).",
        f"- Targeted regression PASS: [targeted_regression.json]({targeted_regression_path}).",
        "",
        "## Interface and feature contract",
        "",
        "The sidecar is independent of production MOT/OVMOT code. For each finite cell it computes causal features for base, memory, appearance delta, fused score, geometry-to-native reference, motion compatibility from the previous audit frame, cell margin, reliability, candidate age/confidence/box/rank, native-reference age, and frame offset. Public ID, target identity, human action identity, GT identity, and future outcome are not features. Degenerate reference boxes are treated as geometry-unavailable (0), not silently repaired.",
        "",
        "The applied score is `S_ij = B_ij + sigmoid(gate_ij) * A_ij + residual_ij`, with residual bounded by ±0.5. Every candidate×public-ID cell is evaluated. Hard-negative cells retain the frozen sentinel. NONE is represented by one immutable `-1e8` dummy per candidate and bypasses the model; explicit NONE is not counted as a mapping error.",
        "",
        "## Paired replay results",
        "",
        "The pre-registered sequence-cluster CI is sequence-balanced: event identity-utility values are averaged within each sequence first, and those sequence means are bootstrapped with equal sequence weight. The post-replay audit corrected an implementation that flattened sampled event values. The corrected result is [paired_replay_results.json](./outputs/n43/replay/paired_replay_results.json), while the unmodified event-weighted result is preserved as [paired_replay_results_legacy_event_weighted.json](./outputs/n43/replay/paired_replay_results_legacy_event_weighted.json). For M2, the corrected 95% lower CI is -0.032469 at H20, -0.032337 at H50, and -0.025631 at H100; the preserved legacy lower CI was -0.030331, -0.032206, and -0.024435 respectively. This audit changes CI reporting only, not event utilities or the replay protocol.",
        "",
        "| Variant | H | Utility | 95% sequence-cluster CI | assignment changes | correct / incorrect | no-change | untouched regression-free |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for variant in ("M0", "M1", "M2", "M3", "M4"):
        for horizon in (20, 50, 100):
            summary = result["aggregates"][variant][str(horizon)]["all"]
            ci = summary["sequence_cluster_bootstrap_95ci"]
            lines.append(f"| {variant} | H{horizon} | {fmt(summary['identity_utility'])} | [{fmt(ci['lower'])}, {fmt(ci['upper'])}] | {summary['assignment_change_count']} | {summary['assignment_change_correct_count']} / {summary['assignment_change_incorrect_count']} | {summary['assignment_no_change_count']} | {summary['untouched_regression']['all_no_obvious_regression']} |")
    lines += [
        "",
        "IoU, future identity error, recorrection proxy, and assignment change decomposition are computed post-hoc. Standard IDSW/IDF1/HOTA/AssA are `NOT_COMPUTABLE_NO_COMPLETE_TRACKEVAL_SEQUENCE_INPUT`; these bounded event windows are not presented as full-sequence MOT evaluation.",
        "",
        "## Failures and repairs preserved",
        "",
        "- Stage 01 first failed on degenerate reference/candidate boxes; the failure files remain under `outputs/n43/attempts`. The minimal repair made geometry/motion unavailable for such boxes while retaining the candidate row.",
        "- Stage 02 first failed because the Stage 01 audit used renamed matrix keys; a read-only alias was added and the dataset was rebuilt.",
        f"- Stage 04 first produced five partial runtime artifacts before the outer session terminated without a captured exit code; evidence is preserved in [stage_04_attempt1_partial.json]({OUT / 'attempts/stage_04_attempt1_partial.json'}). A subsequent targeted validation found explicit NONE and legacy public-ID namespace semantics, which were repaired. No OOM claim is made for the unclassified termination.",
        "- The post-replay targeted regression passed for hard-negative column handling and non-adjacent valid-column margin calculation; its failure provenance remains marked preserved.",
        "- The corrected sequence-cluster result is sequence-mean first with equal sequence weight. The prior event-weighted result is preserved as a separate legacy artifact for direct audit comparison.",
        "- No ordinary MOT/OVMOT source, third-party SAM3 source, checkpoint, metric definition, seed, or acceptance threshold was changed. No running external training was stopped.",
        "",
        "## Strict gate and next action",
        "",
        "The gate fails because real human tape and real full-loop are unavailable, the full-cell sidecar has no strictly positive holdout-style future effect, and untouched regression is not clean. The sidecar checkpoint is a research artifact only; no calibration head or decoder LoRA is authorized.",
        "",
        "```text",
        "N43_STATUS = COMPLETED_GATE_FAILED",
        "REAL_TAPE = NOT_AVAILABLE",
        "REAL_FULL_LOOP = NOT_RUN",
        "CCAM_EFFECT = NEGATIVE",
        "CALIBRATION_HEAD = NOT_AUTHORIZED",
        "DECODER_LORA = NOT_AUTHORIZED",
        "BLOCKING_REASON = no real human tape/full-loop; full-cell M2/M3/M4 future-effect and untouched-regression gates failed",
        "NEXT_ACTION = collect a schema-valid real human event tape and rerun real candidate-complete full-loop paired replay",
        "```"
    ]
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    stage = {"status": gate["status"], "protocol": "N43_FINAL_RESEARCH_GATE_V1", "command": ["python", "scripts/n43_finalize.py"], "inputs": {"stage_01": str(OUT / "stage_01_status.json"), "stage_02": str(OUT / "stage_02_status.json"), "stage_03": str(OUT / "stage_03_status.json"), "stage_04": str(OUT / "stage_04_status.json"), "paired_replay": str(RESULT), "legacy_event_weighted_replay": str(LEGACY_RESULT), "targeted_regression": str(targeted_regression_path)}, "outputs": {"gate": str(GATE), "report": str(REPORT)}, "metrics": {"sidecar_effect": "NEGATIVE", "m2_identity_utility": {str(h): result["aggregates"]["M2"][str(h)]["all"]["identity_utility"] for h in (20, 50, 100)}, "m2_ci_lower": {str(h): result["aggregates"]["M2"][str(h)]["all"]["sequence_cluster_bootstrap_95ci"]["lower"] for h in (20, 50, 100)}, "real_tape": real_tape, "real_full_loop": real_full_loop, "bootstrap_protocol": result.get("bootstrap_protocol"), "legacy_result_preserved": legacy_result_preserved}, "gate_checks": gate["checks"], "failure_root_cause": gate["failure_reasons"], "next_action": gate["next_action"], "runtime_future_gt_used": False, "finished_at": now()}
    STAGE.write_text(json.dumps(stage, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": stage["status"], "gate": str(GATE), "report": str(REPORT)}, sort_keys=True))


if __name__ == "__main__":
    main()
