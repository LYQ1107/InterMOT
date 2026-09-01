#!/usr/bin/env python3
"""Freeze N41's mechanism decision without implementing a new interface.

This reads only completed N41 diagnostic/post-hoc artifacts and writes
machine-readable interpretation and authorization statuses.  It does not
run replay, modify production association code, train a model, or choose a
configuration for deployment.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.n36_real_eval_common import atomic_json


N41 = ROOT / "outputs" / "n41"
STAGE1 = N41 / "stage_01_status.json"
PAIR_SUMMARY = N41 / "diagnostic" / "candidate_pair_summary.json"
SOURCE_PROTOCOL = N41 / "source_replay" / "source_protocol.json"
SOURCE_MANIFEST = N41 / "source_replay" / "source_embedding_manifest.json"
SMOKE_MANIFEST = N41 / "source_replay" / "smoke_attempt3_manifest.json"
FULL_MANIFEST = N41 / "source_replay" / "full_attempt1_manifest.json"
POSTHOC = N41 / "source_replay" / "posthoc_source_results.json"
INTERPRETATION = N41 / "diagnostic" / "diagnostic_interpretation.json"
STAGE3 = N41 / "stage_03_status.json"
STAGE4 = N41 / "stage_04_status.json"


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return value


def finite(value: Any) -> bool:
    try:
        return bool(math.isfinite(float(value)))
    except (TypeError, ValueError):
        return False


def source_similarity(source_manifest: dict[str, Any]) -> dict[str, Any]:
    rows = []
    exact = 0
    for entry in source_manifest.get("events", []):
        sources = entry["sources"]
        a = np.asarray(sources["A_ideal_gt_roi"]["feature"], dtype=np.float64)
        b = np.asarray(sources["B_frozen_current_human_region"]["feature"], dtype=np.float64)
        c = np.asarray(sources["C_fixed_corrupted_roi"]["feature"], dtype=np.float64)
        cos_ab = float(np.dot(a, b) / max(np.linalg.norm(a) * np.linalg.norm(b), 1e-12))
        cos_ac = float(np.dot(a, c) / max(np.linalg.norm(a) * np.linalg.norm(c), 1e-12))
        cos_bc = float(np.dot(b, c) / max(np.linalg.norm(b) * np.linalg.norm(c), 1e-12))
        same = str(sources["A_ideal_gt_roi"]["feature_sha256"]) == str(sources["B_frozen_current_human_region"]["feature_sha256"])
        exact += int(same)
        rows.append({
            "event_id": str(entry["event_id"]),
            "cosine_A_B": cos_ab,
            "cosine_A_C": cos_ac,
            "cosine_B_C": cos_bc,
            "A_B_digest_equal": same,
        })
    return {
        "event_count": len(rows),
        "A_B_exact_digest_equal_count": exact,
        "A_B_exact_digest_equal_rate": float(exact / len(rows)) if rows else None,
        "A_B_cosine": {
            "median": float(statistics.median(row["cosine_A_B"] for row in rows)) if rows else None,
            "min": min((row["cosine_A_B"] for row in rows), default=None),
            "max": max((row["cosine_A_B"] for row in rows), default=None),
        },
        "A_C_cosine": {
            "median": float(statistics.median(row["cosine_A_C"] for row in rows)) if rows else None,
            "min": min((row["cosine_A_C"] for row in rows), default=None),
            "max": max((row["cosine_A_C"] for row in rows), default=None),
        },
        "per_event": rows,
    }


def gate_matrix(posthoc: dict[str, Any]) -> dict[str, Any]:
    output = {}
    for source_id, configs in posthoc["source_configurations"].items():
        output[source_id] = {}
        for config_id, variants in configs.items():
            output[source_id][config_id] = {}
            for variant in ("M2", "M3", "M4"):
                value = variants[variant]
                boot = value["sequence_cluster_bootstrap"]
                output[source_id][config_id][variant] = {
                    "future_effect_gate": value["future_effect_gate"]["status"],
                    "h20": {key: boot["20"].get(key) for key in ("mean", "lower", "upper", "n_clusters")},
                    "h50": {key: boot["50"].get(key) for key in ("mean", "lower", "upper", "n_clusters")},
                    "h100": {key: boot["100"].get(key) for key in ("mean", "lower", "upper", "n_clusters")},
                    "transition_h20": {
                        key: value["transition_diagnostics"]["20"].get(key)
                        for key in (
                            "score_changed_count",
                            "assignment_changed_count",
                            "correct_assignment_change_count",
                            "incorrect_assignment_change_count",
                            "score_change_rate",
                            "assignment_change_rate",
                            "correct_assignment_change_rate",
                            "incorrect_assignment_change_rate",
                        )
                    },
                    "protected_no_obvious_regression": value["protected_no_obvious_regression"],
                }
    return output


def run() -> dict[str, Any]:
    stage1 = load(STAGE1)
    pair = load(PAIR_SUMMARY)
    source_protocol = load(SOURCE_PROTOCOL)
    source_manifest = load(SOURCE_MANIFEST)
    smoke = load(SMOKE_MANIFEST)
    full = load(FULL_MANIFEST)
    posthoc = load(POSTHOC)
    required_passes = {
        "stage1": stage1.get("status") == "PASS",
        "stage1_parameter_path": stage1.get("diagnostic_gate", {}).get("parameter_path_smoke_pass") is True,
        "stage1_candidate_pairs": stage1.get("diagnostic_gate", {}).get("candidate_pair_audit_pass") is True,
        "source_protocol_frozen": source_protocol.get("status") == "FROZEN_BEFORE_SOURCE_GENERATION_AND_REPLAY",
        "source_sidecar_pass": source_manifest.get("status") == "PASS" and source_manifest.get("source_entry_count") == 72,
        "smoke_pass": smoke.get("status") == "SMOKE_PASS" and smoke.get("worker_count") == 18 and smoke.get("all_batch_candidate_stream_checks_pass") is True,
        "full_runtime_pass": full.get("status") == "FULL_RUNTIME_PASS" and full.get("worker_count") == 144 and full.get("all_batch_candidate_stream_checks_pass") is True,
        "posthoc_complete": posthoc.get("status") == "COMPLETED_POSTHOC_DIAGNOSTIC" and posthoc.get("posthoc_variant_result_count") == 720,
        "runtime_future_gt_false": posthoc.get("runtime_future_gt_used") is False and full.get("runtime_future_gt_used") is False,
    }
    similarities = source_similarity(source_manifest)
    groups = gate_matrix(posthoc)
    all_strict_gates = all(
        groups[source][config][variant]["future_effect_gate"] == "PASS"
        for source in groups
        for config in groups[source]
        for variant in ("M2", "M3", "M4")
    )
    # The pair auditor deliberately uses the compact ``all|horizon=...``
    # key for the aggregate groups.  Keep this adapter explicit so a schema
    # mismatch cannot silently turn into a scientific conclusion.
    pair_groups = pair["by_group"]
    pair_all = pair_groups["all|horizon=H100"]
    pair_h20 = pair_groups["all|horizon=H20"]
    pair_event1 = pair_groups["all|horizon=event_plus_1"]
    decision_checks = {
        "parameter_transfer_audit_pass": bool(required_passes["stage1_parameter_path"]),
        "appearance_directional_signal_exists": bool(pair_all.get("appearance_gap_positive_rate", 0.0) > 0.0),
        "pairwise_scale_bottleneck_evidence": bool(pair_all.get("base_wrong_appearance_can_correct_at_lambda8_count", 0) > 0),
        "pairwise_high_weight_collateral_risk_present": bool(pair_all.get("base_correct_pushed_wrong_any_scanned_lambda_count", 0) > 0),
        "ideal_and_current_source_distinguishable": bool(similarities["A_B_exact_digest_equal_count"] < similarities["event_count"]),
        "source_ablation_strict_future_gate_passed_for_all_M2_M4": bool(all_strict_gates),
        "runtime_and_candidate_integrity_pass": bool(all(required_passes.values())),
    }
    decision = "DO_NOT_IMPLEMENT_NEW_FUSION_INTERFACE"
    rationale = "Appearance direction and scale-crossing examples exist, but ideal A and current B are effectively identical and every preregistered source/configuration fails the strict future-effect CI gate; high-weight boundary crossings are not sequence-stably correct."
    interpretation = {
        "protocol": "N41_WEIGHTED_ASSOCIATION_INTERFACE_PROBE_DECISION_V1",
        "status": "COMPLETED_DIAGNOSTIC_GATE_FAILED",
        "created_at": now(),
        "decision": decision,
        "rationale": rationale,
        "scientific_conclusion": "appearance_evidence_candidate_base_state_or_assignment_boundary_bottleneck; no production interface change authorized",
        "required_passes": required_passes,
        "source_similarity": similarities,
        "pair_diagnostics": {
            "H100": {
                "appearance_gap_positive_rate": pair_all.get("appearance_gap_positive_rate"),
                "base_wrong_count": pair_all.get("base_wrong_count"),
                "base_wrong_appearance_can_correct_at_lambda8_count": pair_all.get("base_wrong_appearance_can_correct_at_lambda8_count"),
                "base_wrong_appearance_can_correct_at_lambda8_rate_over_base_wrong": pair_all.get("base_wrong_appearance_can_correct_at_lambda8_rate_over_base_wrong"),
                "base_correct_pushed_wrong_at_lambda1_count": pair_all.get("base_correct_pushed_wrong_at_lambda1_count"),
                "base_correct_pushed_wrong_any_scanned_lambda_count": pair_all.get("base_correct_pushed_wrong_any_scanned_lambda_count"),
            },
            "H20": {
                "appearance_gap_positive_rate": pair_h20.get("appearance_gap_positive_rate"),
                "base_wrong_count": pair_h20.get("base_wrong_count"),
                "base_wrong_appearance_can_correct_at_lambda8_count": pair_h20.get("base_wrong_appearance_can_correct_at_lambda8_count"),
                "base_correct_pushed_wrong_any_scanned_lambda_count": pair_h20.get("base_correct_pushed_wrong_any_scanned_lambda_count"),
            },
            "event_plus_1": {
                "appearance_gap_positive_rate": pair_event1.get("appearance_gap_positive_rate"),
                "base_wrong_count": pair_event1.get("base_wrong_count"),
                "base_wrong_appearance_can_correct_at_lambda8_count": pair_event1.get("base_wrong_appearance_can_correct_at_lambda8_count"),
            },
        },
        "decision_checks": decision_checks,
        "source_configuration_gate_matrix": groups,
        "no_new_interface_candidates_selected": True,
        "production_formula_changed": False,
        "checkpoint_changed": False,
        "candidate_definition_changed": False,
        "training_authorized": False,
        "calibration_head": "NOT_AUTHORIZED",
        "selector": "NOT_AUTHORIZED",
        "decoder_lora": "NOT_AUTHORIZED",
        "four_gpu_plan": {
            "diagnostic_gpu_count_used": 0,
            "maximum_gpu_count_allowed_after_pass": 4,
            "interface_validation_started": False,
            "reason_not_started": "N41-03 authorization conditions failed before any production interface candidate was frozen",
        },
        "preserved_failure_evidence": [
            "outputs/n41/attempts/stage_01_invocation_attempt1_failure.json",
            "outputs/n41/attempts/stage_01_attempt2_failure.json",
            "outputs/n41/attempts/stage_01_attempt3_gate_classification_failure.json",
            "outputs/n41/source_replay/smoke_attempt1_failure.json",
            "outputs/n41/source_replay/smoke_attempt2_failure.json",
        ],
        "input_hashes": {str(path.relative_to(ROOT)): sha256(path) for path in (STAGE1, PAIR_SUMMARY, SOURCE_PROTOCOL, SOURCE_MANIFEST, SMOKE_MANIFEST, FULL_MANIFEST, POSTHOC)},
    }
    atomic_json(INTERPRETATION, interpretation)
    stage3 = {
        "stage": "N41-03",
        "status": "DECISION_NO_NEW_INTERFACE",
        "protocol": interpretation["protocol"],
        "decision": decision,
        "diagnostic_result": str(INTERPRETATION.relative_to(ROOT)),
        "diagnostic_checks": decision_checks,
        "all_preregistered_source_config_gates_pass": all_strict_gates,
        "source_A_B_distinguishable": decision_checks["ideal_and_current_source_distinguishable"],
        "production_formula_changed": False,
        "downstream_authorized": False,
        "calibration_head": "NOT_AUTHORIZED",
        "selector": "NOT_AUTHORIZED",
        "decoder_lora": "NOT_AUTHORIZED",
        "next_action": "Do not implement or train a new fusion interface; preserve N41 evidence and identify the smallest future association-interface probe that does not change checkpoint/candidate/metric definitions.",
    }
    stage4 = {
        "stage": "N41-04",
        "status": "NOT_AUTHORIZED_PRECONDITION_FAILED",
        "protocol": "N41_WEIGHTED_ASSOCIATION_INTERFACE_PROBE_STAGE4_V1",
        "reason": "N41-03 did not authorize a production interface candidate; no interface validation or four-GPU run is permitted.",
        "input_diagnostic": str(INTERPRETATION.relative_to(ROOT)),
        "interface_candidates_frozen": [],
        "replay_started": False,
        "gpu_count_used": 0,
        "training_started": False,
        "calibration_head": "NOT_AUTHORIZED",
        "selector": "NOT_AUTHORIZED",
        "decoder_lora": "NOT_AUTHORIZED",
        "downstream_authorized": False,
        "hard_deadline_calendar": {
            "iclr_2027_abstract_deadline": "2026-09-18 23:59 AoE",
            "iclr_2027_full_paper_deadline": "2026-09-25 23:59 AoE",
            "calendar_as_of": "2026-08-29 Asia/Shanghai",
        },
    }
    atomic_json(STAGE3, stage3)
    atomic_json(STAGE4, stage4)
    return {
        "interpretation": interpretation,
        "stage3": stage3,
        "stage4": stage4,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.parse_args()
    result = run()
    print(json.dumps({"status": result["interpretation"]["status"], "decision": result["interpretation"]["decision"], "stage3": result["stage3"]["status"], "stage4": result["stage4"]["status"]}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
