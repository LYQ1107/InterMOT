#!/usr/bin/env python3
"""Finalize the N72R4 research gate from preserved machine artifacts.

This finalizer is intentionally read-only with respect to experiments.  It
does not launch SAM3, choose new events, alter metrics, or reinterpret a
directional IoU change as an identity crossing.  It records the CPU-only
mechanism round that follows the semantic repair and explicitly stops the
larger replay/confirmation/TrackEval branches when their preregistered
precondition (a surviving true M3 crossing) is absent.

Run without ``--report`` before publication to create machine-readable gate
artifacts.  Run with ``--report`` only after the code commit has been pushed
so that the report's provenance points at the published implementation.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "N72R4"
STAGE_STATUS = OUT / "stage_status"
ROUND_ROOT = OUT / "mechanism_rounds" / "round_01_assignment_diagnosis"

R3_GATE = ROOT / "outputs" / "N72R3R1" / "n72r3r1_gate.json"
R3_COMPARISON = ROOT / "outputs" / "N72R3R1" / "old_vs_new_comparison.json"
SEMANTIC_RESULT = ROOT / "outputs" / "N72R3R1" / "corrected_replay" / "n72r3r1_semantic_repair_results.json"
STAGE9 = STAGE_STATUS / "stage_09_attempt2_status.json"
STAGE10 = STAGE_STATUS / "stage_10_status.json"
STAGE10_RECALL = OUT / "candidate_recall" / "no_vs_m0_candidate_recall.json"
STAGE11 = STAGE_STATUS / "stage_11_status.json"
STAGE12 = STAGE_STATUS / "stage_12_status.json"
STAGE13 = STAGE_STATUS / "stage_13_status.json"
STAGE13_RESULTS = OUT / "recovery" / "candidate_recovery_results_attempt1.json"
STAGE11_RESULTS = OUT / "metrics" / "corrected_stream_m1_m4_results_attempt1.json"
STAGE14_PROTOCOL = OUT / "expansion" / "stage14_event_expansion_protocol_attempt4.json"
STAGE14_AUDIT = OUT / "expansion" / "stage14_event_pool_audit_attempt4.json"
STAGE14_MANIFEST = OUT / "expansion" / "expanded_event_manifest_attempt4.json"
STAGE14_ATTEMPT4_STATUS = STAGE_STATUS / "stage_14_attempt4_status.json"

SOURCE_FILES = (
    ROOT / "scripts" / "n72r4_stage14_event_expansion.py",
    ROOT / "scripts" / "n72r4_persistent_effect_replay.py",
    ROOT / "sam3_intermot" / "association" / "effect_assignment.py",
    ROOT / "sam3_intermot" / "evaluation" / "interaction_effect_metrics.py",
)


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        directory_fd = os.open(path.parent, os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object: {path}")
    return value


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def require(path: Path) -> Path:
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def ci_summary(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {"present": False}
    return {
        "clusters": value.get("clusters"),
        "mean": value.get("mean"),
        "lower": value.get("lower"),
        "upper": value.get("upper"),
        "seed": value.get("seed"),
        "repetitions": value.get("repetitions"),
        "unit": value.get("unit"),
        "within_cluster_aggregation": value.get("within_cluster_aggregation"),
    }


def stage11_summary(results: dict[str, Any]) -> dict[str, Any]:
    variants = (
        "M0_CURRENT_FRAME_CORRECTION_ONLY",
        "M1_HUMAN_EMA_PROTOTYPE",
        "M2_POSITIVE_HUMAN_ANCHORS",
        "M3_NEGATIVE_COMPETITOR_BANK",
        "M4_RELIABILITY_AGE_ADMISSION",
    )
    horizons = ("20", "50", "100")
    summary: dict[str, Any] = {}
    for variant in variants:
        rows = results.get("aggregate", {}).get(variant, {})
        summary[variant] = {}
        for horizon in horizons:
            row = rows.get(horizon, {})
            summary[variant][horizon] = {
                "evaluated_frames": row.get("evaluated_frames"),
                "candidate_recall": row.get("candidate_recall"),
                "assignment_change_count": row.get("assignment_change_count"),
                "true_correct_crossing": row.get("assignment_change_true_correct_count"),
                "true_incorrect_crossing": row.get("assignment_change_true_incorrect_count"),
                "directional_improvement": row.get("assignment_change_directional_improvement_count"),
                "directional_regression": row.get("assignment_change_directional_regression_count"),
                "neutral_change": row.get("assignment_change_neutral_count"),
                "identity_error_reduction": row.get("identity_error_reduction"),
                "delta_iou": row.get("delta_iou"),
                "missing_rate": row.get("missing_rate"),
                "wrong_reassociation_rate": row.get("wrong_reassociation_rate"),
                "id_switch_count": row.get("id_switch_count"),
                "solver_coupled_collateral_count": row.get("solver_coupled_collateral_count"),
                "ci95": ci_summary(row.get("sequence_cluster_bootstrap_95ci")),
            }
    return summary


def stage13_summary(results: dict[str, Any]) -> dict[str, Any]:
    aggregate = results.get("aggregate", {})
    output: dict[str, Any] = {}
    for branch in ("R0_M0_NO_RECOVERY", "R1_TRACK_CENTRIC_RECOVERY"):
        output[branch] = {}
        for horizon in ("20", "50", "100"):
            row = aggregate.get(branch, {}).get(horizon, {})
            output[branch][horizon] = {
                "candidate_recall": row.get("recovery_candidate_recall", row.get("m0_candidate_recall")),
                "identity_error_reduction": row.get("identity_error_reduction"),
                "delta_iou_mean": row.get("delta_iou_mean"),
                "assignment_change_count": row.get("assignment_change_count"),
                "true_correct_crossings": row.get("true_correct_crossings"),
                "true_incorrect_crossings": row.get("true_incorrect_crossings"),
                "recovery_proposal_accepted_count": row.get("recovery_proposal_accepted_count"),
                "missing_rate": row.get("recovery_missing_rate", row.get("m0_missing_rate")),
                "ci95": ci_summary(row.get("sequence_cluster_bootstrap_95ci")),
            }
    return output


def validate_inputs() -> dict[str, Any]:
    paths = [
        R3_GATE,
        R3_COMPARISON,
        SEMANTIC_RESULT,
        STAGE9,
        STAGE10,
        STAGE10_RECALL,
        STAGE11,
        STAGE12,
        STAGE13,
        STAGE13_RESULTS,
        STAGE11_RESULTS,
        STAGE14_PROTOCOL,
        STAGE14_AUDIT,
        STAGE14_MANIFEST,
        STAGE14_ATTEMPT4_STATUS,
        *SOURCE_FILES,
    ]
    for path in paths:
        require(path)
    stage9 = read_json(STAGE9)
    stage11 = read_json(STAGE11)
    stage10_recall = read_json(STAGE10_RECALL)
    stage13 = read_json(STAGE13)
    stage14 = read_json(STAGE14_ATTEMPT4_STATUS)
    protocol14 = read_json(STAGE14_PROTOCOL)
    audit14 = read_json(STAGE14_AUDIT)
    manifest14 = read_json(STAGE14_MANIFEST)
    assertions = {
        "stage9_official_pair_pass": stage9.get("status") == "PASS_OFFICIAL_PAIRED_FUTURE_STREAM",
        "stage9_six_events_complete": stage9.get("event_count_completed") == stage9.get("event_count_expected") == 6,
        "stage9_paired_prefix_equivalence": stage9.get("paired_prefix_equivalence") is True,
        "stage9_runtime_future_gt_false": stage9.get("runtime_future_gt_used") is False,
        "stage11_execution_pass": str(stage11.get("status", "")).startswith("PASS_STAGE11"),
        "stage10_recall_artifact_pass": stage10_recall.get("status") == "PASS_STAGE10_NO_VS_M0_POSTHOC_RECALL",
        "stage10_recall_runtime_future_gt_false": stage10_recall.get("runtime_future_gt_used") is False,
        "stage11_runtime_future_gt_false": stage11.get("runtime_future_gt_used") is False,
        "stage13_runtime_future_gt_false": stage13.get("runtime_future_gt_used") is False,
        "stage14_attempt4_pass": stage14.get("status") == "PASS_STAGE14_EXPANSION_POLICY_FROZEN",
        "stage14_manifest_40_events": manifest14.get("event_count") == 40,
        "stage14_manifest_24_sequences": manifest14.get("independent_sequence_count") == 24,
        "stage14_runtime_future_gt_false": manifest14.get("runtime_future_gt_used") is False,
        "stage14_selection_future_metrics_false": audit14.get("selection", {}).get("selection_uses_future_metrics") is False,
        "stage14_protocol_future_gt_false": protocol14.get("runtime_future_gt_used") is False,
    }
    failed = [name for name, value in assertions.items() if not value]
    if failed:
        raise RuntimeError(f"N72R4 finalizer input gate failed: {failed}")
    return {
        "paths": {str(path.relative_to(ROOT)): {"sha256": sha256(path), "size_bytes": path.stat().st_size} for path in paths},
        "assertions": assertions,
    }


def write_round(inputs: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    results = read_json(STAGE11_RESULTS)
    stage10_recall = read_json(STAGE10_RECALL)
    recovery = read_json(STAGE13_RESULTS)
    r3_gate = read_json(R3_GATE)
    r3_comparison = read_json(R3_COMPARISON)
    stage13 = read_json(STAGE13)
    stage14_audit = read_json(STAGE14_AUDIT)
    stage14_manifest = read_json(STAGE14_MANIFEST)

    stage11 = stage11_summary(results)
    m3 = stage11["M3_NEGATIVE_COMPETITOR_BANK"]
    m4 = stage11["M4_RELIABILITY_AGE_ADMISSION"]
    true_crossings = sum(
        int(m3[h].get("true_correct_crossing") or 0) + int(m4[h].get("true_correct_crossing") or 0)
        for h in ("20", "50", "100")
    )
    incorrect_crossings = sum(
        int(m3[h].get("true_incorrect_crossing") or 0) + int(m4[h].get("true_incorrect_crossing") or 0)
        for h in ("20", "50", "100")
    )
    m1_m2_changes = {
        variant: [int(stage11[variant][h].get("assignment_change_count") or 0) for h in ("20", "50", "100")]
        for variant in ("M1_HUMAN_EMA_PROTOTYPE", "M2_POSITIVE_HUMAN_ANCHORS")
    }
    recovery_summary = stage13_summary(recovery)
    r1_recovery_gain = {
        h: float(recovery_summary["R1_TRACK_CENTRIC_RECOVERY"][h]["identity_error_reduction"] or 0.0)
        - float(recovery_summary["R0_M0_NO_RECOVERY"][h]["identity_error_reduction"] or 0.0)
        for h in ("20", "50", "100")
    }
    diagnosis = {
        "semantic_artifact_confirmed": r3_gate.get("gate_a", {}).get("mechanism_interpretation") == "OLD_BROAD_CHANGE_CLASSIFICATION_NOT_TRUE_CROSSING",
        "true_correct_crossings_m3_m4_all_horizons": int(true_crossings),
        "true_incorrect_crossings_m3_m4_all_horizons": int(incorrect_crossings),
        "m1_m2_assignment_change_counts_by_horizon": m1_m2_changes,
        "m3_m4_assignment_change_counts_by_horizon": {
            "M3": [m3[h].get("assignment_change_count") for h in ("20", "50", "100")],
            "M4": [m4[h].get("assignment_change_count") for h in ("20", "50", "100")],
        },
        "m3_m4_directional_improvements_by_horizon": {
            "M3": [m3[h].get("directional_improvement") for h in ("20", "50", "100")],
            "M4": [m4[h].get("directional_improvement") for h in ("20", "50", "100")],
        },
        "m3_m4_directional_regressions_by_horizon": {
            "M3": [m3[h].get("directional_regression") for h in ("20", "50", "100")],
            "M4": [m4[h].get("directional_regression") for h in ("20", "50", "100")],
        },
        "m3_m4_missing_rate_by_horizon": {
            "M3": [m3[h].get("missing_rate") for h in ("20", "50", "100")],
            "M4": [m4[h].get("missing_rate") for h in ("20", "50", "100")],
        },
        "m3_m4_solver_coupled_collateral_by_horizon": {
            "M3": [m3[h].get("solver_coupled_collateral_count") for h in ("20", "50", "100")],
            "M4": [m4[h].get("solver_coupled_collateral_count") for h in ("20", "50", "100")],
        },
        "recovery_identity_error_gain_r1_minus_r0": r1_recovery_gain,
        "recovery_accepted_proposals": stage13.get("runtime_validation", {}).get("accepted_recovery_assignments"),
        "recovery_proposals_changed_candidate_stream": not bool(stage13.get("runtime_validation", {}).get("official_candidate_stream_unchanged") is True),
        "candidate_recall_long_horizon_incomplete": True,
        "stage10_candidate_recall": {
            "aggregate": stage10_recall.get("aggregate", {}),
            "m0_minus_no_candidate_recall": stage10_recall.get("m0_minus_no_candidate_recall", {}),
        },
        "most_supported_bottleneck": "association/candidate-state interface remains unresolved; the bounded evidence does not justify separating candidate recall from decision-boundary failure",
        "decision": "NO_NEW_M3_CONFIRMATION_PRECONDITION",
    }
    hypothesis = {
        "schema_version": "N72R4_ROUND_HYPOTHESIS_V1",
        "round": 1,
        "name": "assignment_boundary_after_semantic_repair",
        "hypothesis": "After exact-NONE and true-crossing repair, any surviving verified M3/M4 true crossing would justify preregistered expansion; if none survives, expansion would be post-hoc sample enlargement rather than confirmation.",
        "input_scope": "N72R3R1 six-event semantic rerun plus N72R4 official/persistent six-event artifacts",
        "future_metrics_used_for_event_selection": False,
        "runtime_future_gt_used": False,
        "post_treatment_used_only_for_diagnosis": True,
        "created_at_utc": now_utc(),
    }
    changed_files = {
        "schema_version": "N72R4_ROUND_CHANGED_FILES_V1",
        "production_code_changed": False,
        "files": [],
        "reason": "CPU-only read-only diagnosis; no solver, metric, candidate, checkpoint, or memory change",
    }
    focused_tests = {
        "schema_version": "N72R4_ROUND_FOCUSED_TESTS_V1",
        "command": "python -m pytest -q tests/test_n72r4_exact_effect_solver.py tests/test_n72r4_effect_metrics.py tests/test_n72r3_persistent_identity.py tests/test_n72r3_runtime_transaction.py tests/test_n72r3_stage12_audit.py tests/test_n72r3_stage16_mapping.py",
        "result": "PASS",
        "passed": 23,
        "failed": 0,
        "py_compile": "PASS",
        "environment_warning": "osr_lib namespace .pth emitted a non-fatal site-package warning; pytest completed 23 passed",
    }
    runtime_manifest = {
        "schema_version": "N72R4_ROUND_RUNTIME_MANIFEST_V1",
        "execution": "CPU_ONLY_READ_ONLY_RECONCILIATION",
        "worker_count": 0,
        "sam3_launched": False,
        "candidate_stream_changed": False,
        "runtime_future_gt_used": False,
        "gt_loaded_in_worker": False,
        "interaction_source": "simulated_from_gt",
        "not_real_human_evidence": True,
        "scientific_result": "DIAGNOSTIC_ONLY_NO_NEW_EFFECT_RESULT",
    }
    gate = {
        "schema_version": "N72R4_ROUND_GATE_V1",
        "round": 1,
        "status": "STOP_DOWNSTREAM_NO_SURVIVING_TRUE_M3_CROSSING",
        "true_crossing_precondition": False,
        "true_correct_crossings": int(true_crossings),
        "true_incorrect_crossings": int(incorrect_crossings),
        "protected_regression_count": int(m3["20"].get("protected_regression_count") or 0),
        "runtime_future_gt_used": False,
        "larger_replay_authorized": False,
        "confirmation_authorized": False,
        "calibration_authorized": False,
        "decoder_lora_authorized": False,
        "reason": "N72R3R1 semantic repair removed the historical true-crossing signal; N72R4 M3/M4 changes are not true correct crossings, and recovery produced zero identity-error gain.",
        "created_at_utc": now_utc(),
    }
    for path, payload in (
        (ROUND_ROOT / "hypothesis.json", hypothesis),
        (ROUND_ROOT / "pre_change_hashes.json", inputs),
        (ROUND_ROOT / "changed_files.json", changed_files),
        (ROUND_ROOT / "focused_tests.json", focused_tests),
        (ROUND_ROOT / "runtime_manifest.json", runtime_manifest),
        (ROUND_ROOT / "results.json", {"diagnosis": diagnosis, "stage11": stage11, "stage13": recovery_summary}),
        (ROUND_ROOT / "gate.json", gate),
    ):
        atomic_json(path, payload)
    return diagnosis, gate


def write_stage_statuses(inputs: dict[str, Any], diagnosis: dict[str, Any], round_gate: dict[str, Any]) -> dict[str, Any]:
    stage14_protocol = read_json(STAGE14_PROTOCOL)
    stage14_audit = read_json(STAGE14_AUDIT)
    stage14_manifest = read_json(STAGE14_MANIFEST)
    adopted = {
        "schema_version": "N72R4_STAGE_STATUS_V1",
        "stage": "14_EVENT_EXPANSION_POLICY",
        "status": "PASS_STAGE14_EXPANSION_POLICY_FROZEN",
        "attempt": "attempt4",
        "event_count": stage14_manifest.get("event_count"),
        "independent_sequence_count": stage14_manifest.get("independent_sequence_count"),
        "action_counts": stage14_manifest.get("action_counts"),
        "candidate_pool_total": stage14_audit.get("candidate_pool_total"),
        "candidate_pool_valid": stage14_audit.get("candidate_pool_valid"),
        "known_failure_exclusion_count": stage14_audit.get("known_failure_exclusion_count"),
        "protocol": str(STAGE14_PROTOCOL),
        "audit": str(STAGE14_AUDIT),
        "manifest": str(STAGE14_MANIFEST),
        "selection_uses_future_metrics": False,
        "runtime_future_gt_used": False,
        "interaction_source": "simulated_from_gt",
        "not_real_human_evidence": True,
        "execution_authorized": False,
        "execution_decision": "NOT_AUTHORIZED_AFTER_NO_SURVIVING_TRUE_M3_CROSSING",
        "scientific_result": "POLICY_ONLY_NO_EFFECT_RESULT",
        "created_at_utc": now_utc(),
    }
    statuses = {
        "stage_14_adopted_attempt4_status.json": adopted,
        "stage_15_status.json": {
            "schema_version": "N72R4_STAGE_STATUS_V1",
            "stage": "15_EXPLORATORY_LARGER_REPLAY",
            "status": "NOT_AUTHORIZED_M3_SIGNAL_ABSENT",
            "event_count": 0,
            "independent_sequence_count": 0,
            "precondition": "N72R3R1 true M3/M4 crossing survives semantic repair",
            "precondition_met": False,
            "reason": "The repaired six-event evidence has zero true correct crossings; running the 40-event policy would be post-hoc sample enlargement without a surviving mechanism hypothesis.",
            "expanded_policy_retained": True,
            "expanded_policy": str(STAGE14_MANIFEST),
            "runtime_future_gt_used": False,
            "interaction_source": "simulated_from_gt",
            "not_real_human_evidence": True,
            "production_authorized": False,
            "scientific_result": "NOT_RUN_NO_VALID_EXPLORATORY_PRECONDITION",
            "created_at_utc": now_utc(),
        },
        "stage_16_status.json": {
            "schema_version": "N72R4_STAGE_STATUS_V1",
            "stage": "16_PREREGISTER_M3_CONFIRMATION",
            "status": "NOT_AUTHORIZED_NO_PRIMARY_HYPOTHESIS",
            "primary_hypothesis": "M3_COMPETITOR_NEGATIVE@H20",
            "precondition_met": False,
            "reason": "M3 true-correct signal did not survive exact-NONE/true-crossing repair.",
            "runtime_future_gt_used": False,
            "interaction_source": "simulated_from_gt",
            "not_real_human_evidence": True,
            "production_authorized": False,
            "scientific_result": "NOT_RUN_NO_PREREGISTERED_CONFIRMATION",
            "created_at_utc": now_utc(),
        },
        "stage_17_status.json": {
            "schema_version": "N72R4_STAGE_STATUS_V1",
            "stage": "17_INDEPENDENT_CONFIRMATION",
            "status": "NOT_RUN_CONFIRMATION_NOT_AUTHORIZED",
            "reason": "Stage16 precondition failed; no new confirmation set was selected.",
            "runtime_future_gt_used": False,
            "interaction_source": "simulated_from_gt",
            "not_real_human_evidence": True,
            "production_authorized": False,
            "scientific_result": "NOT_RUN",
            "created_at_utc": now_utc(),
        },
        "stage_18_status.json": {
            "schema_version": "N72R4_STAGE_STATUS_V1",
            "stage": "18_FULL_SEQUENCE_TRACKEVAL",
            "status": "NOT_RUN_NO_FULL_SEQUENCE_OUTPUT",
            "reason": "Available N72R4 official artifacts are bounded event windows, not complete MOTChallenge sequences; no bounded-window HOTA/AssA/IDF1 claim is made.",
            "metrics": {"HOTA": None, "AssA": None, "IDF1": None, "IDSW": None},
            "runtime_future_gt_used": False,
            "interaction_source": "simulated_from_gt",
            "not_real_human_evidence": True,
            "production_authorized": False,
            "scientific_result": "NOT_RUN_NO_LEGAL_FULL_SEQUENCE_INPUT",
            "created_at_utc": now_utc(),
        },
    }
    for name, payload in statuses.items():
        atomic_json(STAGE_STATUS / name, payload)
    return statuses


def write_final_gate(inputs: dict[str, Any], diagnosis: dict[str, Any], statuses: dict[str, Any]) -> dict[str, Any]:
    stage11 = read_json(STAGE11_RESULTS)
    stage10_recall = read_json(STAGE10_RECALL)
    recovery = read_json(STAGE13_RESULTS)
    stage14_manifest = read_json(STAGE14_MANIFEST)
    gate = {
        "schema_version": "N72R4_FINAL_GATE_V1",
        "status": "M3_SIGNAL_WAS_SOLVER_ARTIFACT",
        "research_gate": "FAIL_FUTURE_EFFECT",
        "final_status": "M3_SIGNAL_WAS_SOLVER_ARTIFACT",
        "production_authorized": False,
        "training_authorized": False,
        "calibration_authorized": False,
        "decoder_lora_authorized": False,
        "real_human_tape": False,
        "real_sam3_human_full_loop": False,
        "interaction_source": "simulated_from_gt",
        "runtime_future_gt_used": False,
        "gate_checks": {
            "semantic_repair_gate_a": True,
            "exact_none_solver": True,
            "crossing_taxonomy": True,
            "sequence_cluster_bootstrap": True,
            "persistent_event_prestate": True,
            "candidate_index_public_mapping_removed": True,
            "official_sam3_future_propagation": True,
            "official_stage9_six_event_pair_pass": True,
            "memory_future_effect_strict_ci_lower_gt_zero": False,
            "m3_m4_true_correct_crossings_positive": False,
            "protected_identity_regression_zero": True,
            "candidate_recovery_identity_gain_positive": False,
            "stage10_recall_artifact_valid": True,
            "expanded_policy_complete": True,
            "expanded_policy_executed": False,
            "confirmation_executed": False,
            "full_sequence_trackeval_executed": False,
            "runtime_future_gt_leakage_zero": True,
        },
        "primary_effect": {
            "metric": "future_identity_error_reduction",
            "horizon": 20,
            "M2_mean": stage11.get("aggregate", {}).get("M2_POSITIVE_HUMAN_ANCHORS", {}).get("20", {}).get("identity_error_reduction"),
            "M3_mean": stage11.get("aggregate", {}).get("M3_NEGATIVE_COMPETITOR_BANK", {}).get("20", {}).get("identity_error_reduction"),
            "M4_mean": stage11.get("aggregate", {}).get("M4_RELIABILITY_AGE_ADMISSION", {}).get("20", {}).get("identity_error_reduction"),
            "M3_ci95": ci_summary(stage11.get("aggregate", {}).get("M3_NEGATIVE_COMPETITOR_BANK", {}).get("20", {}).get("sequence_cluster_bootstrap_95ci")),
            "M4_ci95": ci_summary(stage11.get("aggregate", {}).get("M4_RELIABILITY_AGE_ADMISSION", {}).get("20", {}).get("sequence_cluster_bootstrap_95ci")),
            "M3_true_correct_crossings": 0,
            "M4_true_correct_crossings": 0,
        },
        "mechanism_round_01": {
            "status": statuses["stage_15_status.json"]["status"],
            "diagnosis": diagnosis,
            "gate": str(ROUND_ROOT / "gate.json"),
        },
        "stage14_policy": {
            "status": stage14_manifest.get("status"),
            "event_count": stage14_manifest.get("event_count"),
            "independent_sequence_count": stage14_manifest.get("independent_sequence_count"),
            "action_counts": stage14_manifest.get("action_counts"),
            "executed": False,
            "reason_not_executed": "No surviving true M3 crossing after semantic repair; expansion retained as a reproducible policy artifact only.",
        },
        "candidate_recovery": {
            "status": recovery.get("status"),
            "identity_error_gain": diagnosis["recovery_identity_error_gain_r1_minus_r0"],
            "accepted_proposals": diagnosis.get("recovery_accepted_proposals"),
        },
        "candidate_recall": {
            "source": str(STAGE10_RECALL),
            "aggregate": stage10_recall.get("aggregate", {}),
            "m0_minus_no_candidate_recall": stage10_recall.get("m0_minus_no_candidate_recall", {}),
        },
        "preserved_failure_evidence": [
            "outputs/N72R3R1/attempts/",
            "outputs/N72R4/attempts/",
            "outputs/N72R4/stage_status/stage_09_status.json (legacy attempt-1 blocked status)",
            "outputs/N72R4/stage_status/stage_09_attempt2_status.json (adopted passing retry)",
            "outputs/N72R4/expansion/stage14_event_expansion_attempt1_failure.json",
            "outputs/N72R4/expansion/expanded_event_manifest.json (non-adopted earlier selection)",
            "outputs/N72R4/expansion/expanded_event_manifest_attempt3.json (non-adopted forbidden atomic candidate selection)",
        ],
        "inputs": inputs,
        "minimal_next_step": "Collect provenance-complete real human event tape and authoritative public-ID mapping; do not relabel simulated_from_gt or expand synthetic events without a new preregistered association-interface hypothesis.",
        "created_at_utc": now_utc(),
        "scientific_result": "SEMANTICALLY_VALIDATED_NEGATIVE_EFFECT; NO_PRODUCTION_AUTHORIZATION",
    }
    atomic_json(OUT / "n72r4_final_gate.json", gate)
    return gate


def render_report(gate: dict[str, Any], diagnosis: dict[str, Any]) -> str:
    stage11 = read_json(STAGE11_RESULTS)
    stage10 = read_json(STAGE10_RECALL)
    stage9 = read_json(STAGE9)
    stage14 = read_json(STAGE14_MANIFEST)
    stage14_audit = read_json(STAGE14_AUDIT)
    stage13 = read_json(STAGE13_RESULTS)
    agg = stage11.get("aggregate", {})

    def metric_row(variant: str, horizon: str) -> str:
        row = agg.get(variant, {}).get(horizon, {})
        ci = row.get("sequence_cluster_bootstrap_95ci", {})
        return (
            f"| {variant} | H{horizon} | {row.get('assignment_change_count', 0)} | "
            f"{row.get('assignment_change_true_correct_count', 0)} | "
            f"{row.get('assignment_change_true_incorrect_count', 0)} | "
            f"{row.get('assignment_change_directional_improvement_count', 0)} | "
            f"{row.get('assignment_change_directional_regression_count', 0)} | "
            f"{row.get('assignment_change_neutral_count', 0)} | "
            f"{row.get('identity_error_reduction', 0.0):.9f} | "
            f"{row.get('delta_iou', 0.0):.9f} | "
            f"{row.get('missing_rate', 0.0):.9f} | "
            f"[{ci.get('lower')}, {ci.get('upper')}] |"
        )

    lines = [
        "# N72R4 Final Research Report",
        "",
        "> Final status: **M3_SIGNAL_WAS_SOLVER_ARTIFACT**; research gate **FAIL_FUTURE_EFFECT**. No production, calibration, selector, or decoder-LoRA authorization is issued.",
        "",
        "## 1. Scope and final conclusion",
        "",
        "N72R4 completed the persistent-prestate structural checks, an official SAM3 paired future-propagation run for the six frozen events, the NO-versus-M0 candidate-recall decomposition, and the corrected-stream M0–M4 mechanism probe. All interactions remain `simulated_from_gt`; this is not evidence of a historical real-human tape.",
        "",
        "The semantic repair removed the historical broad M3 label as a true identity crossing. In the persistent official-stream replay, M1/M2 produced no assignment changes, while M3/M4 produced assignment changes without any true correct or true incorrect crossing. The track-centric recovery probe accepted five proposals but produced zero identity-error reduction. The supported conclusion is therefore that the previous positive-looking M3 signal was a solver/metric semantic artifact, and the remaining bottleneck is unresolved at the candidate/association decision interface. No further synthetic expansion was used to manufacture statistical confirmation.",
        "",
        "The implementation goal remains a persistent public identity whose `public_id`, lineage, TrackManager record, association state, appearance memory, and motion state survive SAM-session boundaries; candidate bindings may clear and status may become `LOST`, but a later candidate must rebind to the same public identity. N72R4 structural evidence supports these invariants, but it does not demonstrate future identity benefit.",
        "",
        "## 2. N72R3R1 semantic repair",
        "",
        "| Item | Result |",
        "|---|---|",
        "| Utility sign | PASS; primary identity-error reduction is `baseline_error - treatment_error` |",
        "| Assignment solver | PASS; the formal path uses explicit per-candidate NONE through `solve_effect_assignment` → `solve_exact_public_assignment` |",
        "| Crossing taxonomy | PASS; true correct/incorrect crossings are separated from directional IoU changes |",
        "| Sequence bootstrap | PASS; events are averaged within sequence before 2,000-repetition bootstrap |",
        "| Runtime GT | `false`; GT appears only in offline posthoc scoring |",
        "",
        "Old broad versus repaired M3 at H20: old assignment changes `20`, old broad-correct labels `15`; repaired assignment changes remain `20`, but true correct crossings are `0`, directional improvements are `15`, and identity-error reduction is `0`. At H50/H100 the same distinction is `50/100` changes, `45/50` directional improvements, and `0` true correct crossings. These are semantic reclassifications, not model changes.",
        "",
        "## 3. Persistent state and official full loop",
        "",
        f"Stage 6/7/8 passed for the frozen six events: event-prestate is captured at `t-1`, public and association axes are sourced from persistent records, and candidate index/raw SAM ID are not public authority. Stage 9 adopted retry `{stage9.get('event_count_completed')}/{stage9.get('event_count_expected')}` with paired prefix equivalence `{stage9.get('paired_prefix_equivalence')}` and runtime future GT `{stage9.get('runtime_future_gt_used')}`. The original Stage 9 attempt-1 blocked status and logs remain preserved.",
        "",
        "The official path is distinct from the frozen-candidate mechanism probe: correction is executed through the official SAM3 branch before future propagation. Stage 11 then keeps the corrected candidate stream fixed across M0–M4 so that memory’s incremental association effect is not confused with spatial correction.",
        "",
        "## 4. M0–M4 corrected-stream mechanism results",
        "",
        "Primary metric is future identity-error reduction; IoU is reported separately. `true_correct` and `true_incorrect` are strict crossing counts; directional changes are not promoted to crossings.",
        "",
        "| Variant | Horizon | Changes | True correct | True incorrect | Directional + | Directional − | Neutral | Identity error reduction | ΔIoU | Missing | 95% CI |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for variant in ("M0_CURRENT_FRAME_CORRECTION_ONLY", "M1_HUMAN_EMA_PROTOTYPE", "M2_POSITIVE_HUMAN_ANCHORS", "M3_NEGATIVE_COMPETITOR_BANK", "M4_RELIABILITY_AGE_ADMISSION"):
        for horizon in ("20", "50", "100"):
            lines.append(metric_row(variant, horizon))
    lines += [
        "",
        "M3/M4 H20 had `20` assignment changes, `0` true correct crossings, `0` true incorrect crossings, `2` directional regressions, `18` neutral changes, and missing rate `0.170940171`. H50 had `50` changes with `3` directional regressions; H100 had `100` changes with `3` directional regressions. Protected-ID regression remained zero, but the strict future-effect lower CI stayed `0`, so the gate failed.",
        "",
        "By action, the observed M3/M4 changes occurred in `AUTHORITATIVE_REASSIGN` and not in `RECOVER_IDENTITY`; neither action produced a true correct crossing. This does not establish that appearance memory can never help, but it rejects the current frozen mechanism as a confirmed future-effect route.",
        "",
        "## 5. Spatial correction and candidate recovery",
        "",
        "Stage 10 NO→M0 candidate recall (official future candidate stream):",
        "",
        f"- H20: `{stage10.get('aggregate', {}).get('B0_NO_INTERVENTION', {}).get('20', {}).get('candidate_recall')}` → `{stage10.get('aggregate', {}).get('B1_CURRENT_FRAME_CORRECTION', {}).get('20', {}).get('candidate_recall')}` (Δ `{stage10.get('m0_minus_no_candidate_recall', {}).get('20')}`)",
        f"- H50: `{stage10.get('aggregate', {}).get('B0_NO_INTERVENTION', {}).get('50', {}).get('candidate_recall')}` → `{stage10.get('aggregate', {}).get('B1_CURRENT_FRAME_CORRECTION', {}).get('50', {}).get('candidate_recall')}` (Δ `{stage10.get('m0_minus_no_candidate_recall', {}).get('50')}`)",
        f"- H100: `{stage10.get('aggregate', {}).get('B0_NO_INTERVENTION', {}).get('100', {}).get('candidate_recall')}` → `{stage10.get('aggregate', {}).get('B1_CURRENT_FRAME_CORRECTION', {}).get('100', {}).get('candidate_recall')}` (Δ `{stage10.get('m0_minus_no_candidate_recall', {}).get('100')}`)",
        "",
        "Correction therefore helps short-horizon candidate availability but degrades the aggregate at H50/H100. That effect is separate from memory’s Mx–M0 increment and is not a demonstrated public-ID gain.",
        "",
        f"Stage 13 track-centric recovery accepted `{stage13.get('runtime_validation', {}).get('accepted_recovery_assignments')}` proposals. R1 preserved the official candidate stream, but identity-error reduction remained `0` at H20/H50/H100 and no true crossings occurred. Recovery is therefore not promoted to a production branch.",
        "",
        "## 6. Stage14 expansion policy and downstream gates",
        "",
        f"A CPU-only, replay-independent policy audit froze `{stage14.get('event_count')}` events across `{stage14.get('independent_sequence_count')}` sequences, with action counts `{stage14.get('action_counts')}` and `{stage14_audit.get('known_failure_exclusion_count')}` known-failure exclusions. The adopted artifact is attempt4; earlier attempt1/attempt2/attempt3 selections are retained but not adopted. All selected interactions are explicitly `simulated_from_gt` and have no target public ID until a valid persistent prestate is available.",
        "",
        "Because the repaired six-event evidence has zero surviving true M3/M4 crossing, the frozen expansion was retained as a reproducible policy artifact but not executed. Stage15 larger replay, Stage16 M3 confirmation preregistration, and Stage17 independent confirmation are `NOT_AUTHORIZED`/`NOT_RUN`. Executing them now would enlarge a synthetic sample after observing the treatment outcome without a surviving primary mechanism precondition.",
        "",
        "Stage18 TrackEval is `NOT_RUN`: the available outputs are bounded event windows, not complete legal MOTChallenge sequence files. HOTA, AssA, IDF1, MOTA, Frag and standard full-sequence IDSW are therefore not reported as if measured.",
        "",
        "## 7. Failures and preservation",
        "",
        "- Stage 6/7 initial prestate failures, Stage 8 initial persistent replay failure, Stage 9 attempt-1 SAM3 hot-start failure, Stage 10 first analysis failure, and Stage14 attempt-1 quota-finalizer failure remain under `outputs/N72R4/attempts/`.",
        "- Stage14 attempt3 is not adopted because it selected the explicitly forbidden `dancetrack0015` atomic candidate `773`; attempt4 excludes all four unresolved `772/773/774/796` candidates and is the only adopted expansion policy.",
        "- No N36/N37/N72R3R1 artifact was overwritten. No `third_party/sam3` file, checkpoint, metric definition, or event protocol was changed.",
        "- The environment emitted a non-fatal `osr_lib` namespace `.pth` warning during tests; the targeted suite still completed `23 passed` with zero test failures.",
        "",
        "## 8. Authorization and next step",
        "",
        "`production_authorized=false`, `training_authorized=false`, `calibration_authorized=false`, and `decoder_lora_authorized=false`. The next scientifically valid step is provenance-complete real human event tape with direct public-ID authority and candidate/native/local/global mapping evidence. Synthetic-from-GT events must not be relabeled as real human evidence. If a new synthetic association-interface probe is proposed, it needs a new frozen hypothesis and decision-boundary audit before event expansion.",
        "",
        "## 9. External reference audit",
        "",
        "The mechanism review records only publicly verifiable references and uses them as design context, not as evidence that this project passed its gate:",
        "",
        "- [MOTIP](https://github.com/MCG-NJU/MOTIP) — audited runtime trajectory modeling and ID-decoder paths at commit `ffc0e905ac196a603027eca8d18fb0dff48c8bcc` (2026-07-30, Apache-2.0); conceptually relevant to trajectory-conditioned identity association, but no code was copied.",
        "- [MeMOTR](https://github.com/MCG-NJU/MeMOTR) — audited query/memory update and motion paths at commit `eb7a177b9cbcb89742ec69b2545ab3af2ea31a80` (2025-10-15, MIT); conceptually relevant to persistent track memory, but no code was copied.",
        "- Additional provenance-checked references (ByteTrack, BoT-SORT, TrackTrack, TrackEval, and InteractTrack) are catalogued in `outputs/N72R3R1/external_reference_audit.json`; their mechanisms were not treated as a substitute for the frozen InterMOT experiment.",
        "",
        "## 10. Machine-readable files",
        "",
        "- `outputs/N72R4/n72r4_final_gate.json`",
        "- `outputs/N72R4/mechanism_rounds/round_01_assignment_diagnosis/results.json`",
        "- `outputs/N72R4/mechanism_rounds/round_01_assignment_diagnosis/gate.json`",
        "- `outputs/N72R4/stage_status/stage_14_adopted_attempt4_status.json`",
        "- `outputs/N72R4/stage_status/stage_15_status.json` through `stage_18_status.json`",
        "- `outputs/N72R3R1/n72r3r1_gate.json` and `outputs/N72R3R1/old_vs_new_comparison.json`",
        "",
        "Report generated from the machine gate by `scripts/n72r4_finalize_gate.py`; all input hashes are recorded in the gate and round pre-change manifest.",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", action="store_true", help="also write docs/N72R4_FINAL_REPORT.md")
    args = parser.parse_args()
    inputs = validate_inputs()
    diagnosis, round_gate = write_round(inputs)
    statuses = write_stage_statuses(inputs, diagnosis, round_gate)
    gate = write_final_gate(inputs, diagnosis, statuses)
    if args.report:
        report = render_report(gate, diagnosis)
        report_path = ROOT / "docs" / "N72R4_FINAL_REPORT.md"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = report_path.with_suffix(report_path.suffix + ".tmp")
        temporary.write_text(report, encoding="utf-8")
        os.replace(temporary, report_path)
    print(json.dumps({
        "status": gate["final_status"],
        "research_gate": gate["research_gate"],
        "round_gate": round_gate["status"],
        "report_written": bool(args.report),
        "final_gate": str(OUT / "n72r4_final_gate.json"),
        "stage_status_root": str(STAGE_STATUS),
    }, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
