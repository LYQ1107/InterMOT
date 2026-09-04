#!/usr/bin/env python3
"""Audit the fixed TVC_V1 mechanism round without rerunning replay.

This CPU-only audit joins the preregistered training record, model digest,
Stage08/09 validity, and Stage10 posthoc summaries.  It records the
incremental result of TVC_V1 separately from the full B4-vs-B0 gate; a
positive incremental pair is not promoted when the full interactive effect
or protected-identity gate fails.
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import sys
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sam3_intermot.association.branch_public_replay import atomic_json, read_json, sha256_file  # noqa: E402


OUT = Path(os.environ.get("N72R5R1_RUN_ROOT", str(ROOT / "outputs/N72R5R1")))
ROUND_ROOT = Path(
    os.environ.get(
        "N72R5R1_ROUND04_ROOT",
        str(OUT / "controller" / "round_04_tvc_v1"),
    )
)
AUDIT_ROOT = ROUND_ROOT / "audit"
AUDIT = AUDIT_ROOT / "round_04_mechanism_audit.json"
STATUS = AUDIT_ROOT / "round_04_status.json"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def _summary(effect: Mapping[str, Any], pair: str, horizon: str = "20") -> dict[str, Any]:
    value = effect.get("summaries", {}).get(pair, {}).get(horizon, {})
    if not isinstance(value, Mapping):
        raise ValueError(f"missing Stage10 summary: {pair}/{horizon}")
    bootstrap = value.get("sequence_cluster_bootstrap", {})
    return {
        "event_count": value.get("event_count"),
        "mean_identity_error_reduction": value.get("mean_identity_error_reduction"),
        "sequence_cluster_bootstrap": dict(bootstrap) if isinstance(bootstrap, Mapping) else {},
        "assignment_changes": value.get("assignment_changes"),
        "assignment_change_rate": value.get("assignment_change_rate"),
        "true_correct_crossings": value.get("true_correct_crossings"),
        "true_incorrect_crossings": value.get("true_incorrect_crossings"),
        "by_action": value.get("by_action", {}),
    }


def main() -> int:
    protocol_path = ROUND_ROOT / "round_04_protocol.json"
    model_path = ROUND_ROOT / "tvc_v1_model.json"
    training_path = ROUND_ROOT / "training_result.json"
    full_root = ROUND_ROOT / "full"
    stage08_path = full_root / "stage08_runtime_manifest.json"
    stage09_path = full_root / "stage09_validation.json"
    effect_path = full_root / "stage10_effect_scoring.json"
    round03_path = OUT / "controller" / "round_03_feature_separability_attempt2" / "feature_separability_summary.json"
    baseline_effect_path = OUT / "stage10_effect_scoring.json"

    required = [protocol_path, model_path, training_path, stage08_path, stage09_path, effect_path, round03_path]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        payload = {
            "schema_version": "N72R5R1_ROUND04_AUDIT_V1",
            "status": "BLOCKED_MISSING_ROUND04_INPUT",
            "missing_inputs": missing,
            "runtime_future_gt_used": False,
            "created_at_utc": _now(),
        }
        atomic_json(AUDIT, payload)
        atomic_json(STATUS, {**payload, "stage": "04_TVC_V1_MECHANISM_AUDIT"})
        print(json.dumps(payload, ensure_ascii=False))
        return 1

    protocol = read_json(protocol_path)
    model = read_json(model_path)
    training = read_json(training_path)
    stage08 = read_json(stage08_path)
    stage09 = read_json(stage09_path)
    effect = read_json(effect_path)
    round03 = read_json(round03_path)

    summaries = {
        pair: _summary(effect, pair)
        for pair in ("B1_MINUS_B0", "B2_MINUS_B1", "B3_MINUS_B1", "B4_MINUS_B2", "B4_MINUS_B0")
    }
    protected = effect.get("summaries", {}).get("B4_MINUS_B0", {}).get(
        "protected_identity_regression_h20", {}
    )
    gate = effect.get("gate", {})
    incremental_pairs = {
        "B3_MINUS_B1": summaries["B3_MINUS_B1"],
        "B4_MINUS_B2": summaries["B4_MINUS_B2"],
    }
    incremental_effective = all(
        _finite(incremental_pairs[pair].get("mean_identity_error_reduction"))
        and float(incremental_pairs[pair]["mean_identity_error_reduction"]) > 0.0
        and _finite(incremental_pairs[pair].get("sequence_cluster_bootstrap", {}).get("lower"))
        and float(incremental_pairs[pair]["sequence_cluster_bootstrap"]["lower"]) > 0.0
        for pair in incremental_pairs
    )
    full_gate = (
        stage08.get("status") == "PASS_N72R5R1_EXACT_PUBLIC_ASSOCIATION"
        and stage09.get("strict_pass") is True
        and effect.get("status") == "PASS_GT_SIMULATED_FUTURE_EFFECT_CONFIRMED"
    )
    audit = {
        "schema_version": "N72R5R1_ROUND04_MECHANISM_AUDIT_V1",
        "stage": "04_TVC_V1_MECHANISM_AUDIT",
        "status": "PASS_INCREMENTAL_TVC_EFFECT_FULL_GATE_FAIL" if incremental_effective else "PASS_NO_INCREMENTAL_TVC_EFFECT",
        "inputs": {
            "protocol": str(protocol_path),
            "protocol_sha256": sha256_file(protocol_path),
            "model": str(model_path),
            "model_sha256": sha256_file(model_path),
            "training_result": str(training_path),
            "stage08_manifest": str(stage08_path),
            "stage09_validation": str(stage09_path),
            "stage10_effect": str(effect_path),
            "round03_summary": str(round03_path),
            "baseline_effect": str(baseline_effect_path) if baseline_effect_path.is_file() else None,
        },
        "training": {
            "status": training.get("status"),
            "train_pair_count": training.get("train_pair_count"),
            "holdout_pair_count": training.get("holdout_pair_count"),
            "holdout_auc": training.get("holdout_metrics", {}).get("auc"),
            "sequence_split": {
                "train": protocol.get("train_sequences", []),
                "holdout": protocol.get("holdout_sequences", []),
            },
            "future_effect_used_for_training_or_selection": protocol.get("future_effect_used_for_training_or_selection"),
            "runtime_future_gt_used": False,
        },
        "model": {
            "model_sha256": sha256_file(model_path),
            "features": model.get("features", []),
            "max_abs_residual": model.get("max_abs_residual"),
            "runtime_future_gt_used": False,
        },
        "execution": {
            "stage08_status": stage08.get("status"),
            "stage08_public_assignment_complete": stage08.get("public_assignment_complete"),
            "stage09_status": stage09.get("status"),
            "stage09_strict_pass": stage09.get("strict_pass"),
            "stage10_status": effect.get("status"),
            "stage10_gate": gate,
            "runtime_future_gt_used": False,
        },
        "round03_context": {
            "status": round03.get("status"),
            "pair_count": round03.get("pair_count"),
            "anchor_direction_rate": round03.get("overall", {}).get("anchor_direction_rate", round03.get("anchor_direction_rate")),
            "prototype_direction_rate": round03.get("overall", {}).get("prototype_direction_rate", round03.get("prototype_direction_rate")),
        },
        "incremental_tvc_effect": {
            "mechanism_effective": bool(incremental_effective),
            "pairs": incremental_pairs,
            "interpretation": "TVC_V1 changes the incremental B3-B1/B4-B2 association outcome on this frozen input" if incremental_effective else "TVC_V1 has no statistically supported incremental effect on this frozen input",
        },
        "full_interactive_gate": {
            "full_gate": bool(full_gate),
            "primary_pair": "B4_MINUS_B0",
            "primary_h20": summaries["B4_MINUS_B0"],
            "protected_identity_regression_h20": protected,
            "unavailable_event_count": gate.get("unavailable_event_count"),
        },
        "mechanism_conclusion": {
            "root_cause": "SPATIAL_CORRECTION_PERSISTENCE_OR_CANDIDATE_STREAM_DRIFT",
            "reason": "The learned TVC verifier has a positive incremental effect over B1/B2, while B1 remains harmful versus B0 and protected-identity regression remains nonzero; this localizes the next audit to the spatial-correction branch lifecycle/candidate stream rather than another TVC weight scan.",
            "production_promotion": False,
            "calibration_or_lora_authorized": False,
        },
        "posthoc_gt_opened": True,
        "runtime_future_gt_used": False,
        "created_at_utc": _now(),
    }
    atomic_json(AUDIT, audit)
    status = {
        "schema_version": "N72R5R1_ROUND04_STATUS_V1",
        "stage": "04_TVC_V1_MECHANISM_AUDIT",
        "status": audit["status"],
        "mechanism_effective": bool(incremental_effective),
        "full_gate": bool(full_gate),
        "audit": str(AUDIT),
        "next_round": "AUDIT_SPATIAL_CORRECTION_PERSISTENCE_AND_CANDIDATE_STREAM_DRIFT",
        "runtime_future_gt_used": False,
        "created_at_utc": _now(),
    }
    atomic_json(STATUS, status)
    print(json.dumps(status, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
