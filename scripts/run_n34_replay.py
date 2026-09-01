#!/usr/bin/env python3
"""Run compact M0-M4 paired replays on the explicit synthetic fallback."""

from __future__ import annotations

import argparse
import gc
import json
import os
import tempfile
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from n34_synthetic import build_tape
from sam3_intermot.association.ccam_replay import paired_replay, validate_candidate_tape
from sam3_intermot.association.state_manager import StateManagerConfig


OUT = ROOT / "outputs" / "n34"
HORIZONS = (20, 50, 100)
VARIANTS = {
    "M0": {
        "description": "K1-only baseline; CCAM disabled",
        "use_appearance_memory": False,
    },
    "M1": {
        "description": "human EMA prototype only",
        "use_appearance_memory": True,
        "appearance_anchor_cap": 0,
        "appearance_negative_cap": 0,
    },
    "M2": {
        "description": "human EMA prototype plus multi-positive anchors",
        "use_appearance_memory": True,
        "appearance_anchor_cap": 8,
        "appearance_negative_cap": 0,
    },
    "M3": {
        "description": "multi-positive anchors plus competing negative bank",
        "use_appearance_memory": True,
        "appearance_anchor_cap": 8,
        "appearance_negative_cap": 16,
    },
    "M4": {
        "description": "M3 plus reliability/age gate",
        "use_appearance_memory": True,
        "appearance_anchor_cap": 8,
        "appearance_negative_cap": 16,
        "appearance_reliability_threshold": 0.5,
        "appearance_decay_frames": 60.0,
    },
}
ACTION_TYPES = (
    "ADD_NEW_IDENTITY",
    "AUTHORITATIVE_REASSIGN",
    "ATOMIC_ID_SWAP",
    "RECOVER_IDENTITY",
)


def jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.astype(float).tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    return value


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, path)
    finally:
        if os.path.exists(tmp_name):
            os.unlink(tmp_name)


def atomic_json(path: Path, payload: Any) -> None:
    atomic_text(path, json.dumps(jsonable(payload), indent=2, sort_keys=True, allow_nan=False) + "\n")


def _variant_summary(replay: dict[str, Any], action: str) -> dict[str, Any]:
    comparison = replay.get("comparison", [])
    finite_deltas = [
        float(item["max_abs_score_delta"])
        for item in comparison
        if item.get("max_abs_score_delta") is not None
        and np.isfinite(float(item["max_abs_score_delta"]))
    ]
    first_delta = None
    if comparison and comparison[0].get("max_abs_score_delta") is not None:
        first_delta = float(comparison[0]["max_abs_score_delta"])
    horizon_runtime = {
        str(horizon): bool(len(comparison) >= horizon) for horizon in HORIZONS
    }
    return {
        "action_type": action,
        "status": replay.get("status", "FAIL"),
        "candidate_complete": bool(replay.get("candidate_complete", False)),
        "event_frame": int(replay.get("event_frame", 0)),
        "future_frame_count": len(comparison),
        "horizon_runtime": horizon_runtime,
        "causal_boundary": {
            "first_write_eligible_frame": int(replay.get("event_frame", 0)) + 1,
            "current_frame_write_used_for_score": False,
            "same_prefix_and_spatial_correction": True,
            "future_gt_used_runtime": False,
        },
        "score_delta": {
            "first_future_frame_max_abs": first_delta,
            "max_abs_over_future": max(finite_deltas) if finite_deltas else 0.0,
            "frames_with_assignment_change": sum(bool(item.get("assignment_changed")) for item in comparison),
        },
        "metrics": {
            "future_h20_iou": "NOT_COMPUTABLE_NO_REAL_GT_EVALUATION",
            "future_h50_iou": "NOT_COMPUTABLE_NO_REAL_GT_EVALUATION",
            "future_h100_iou": "NOT_COMPUTABLE_NO_REAL_GT_EVALUATION",
            "missing_prediction_rate": "NOT_COMPUTABLE_NO_REAL_GT_EVALUATION",
            "id_switch_count": "NOT_COMPUTABLE_NO_REAL_IDENTITY_EVALUATOR",
            "re_correction_count": "NOT_COMPUTABLE_NO_REAL_EVENT_STREAM",
            "recovery_latency": "NOT_COMPUTABLE_NO_REAL_EVENT_STREAM",
            "protected_identity_regression": "NOT_COMPUTABLE_NO_REAL_GT_EVALUATION",
            "idf1": "NOT_COMPUTABLE_NO_TRACKEVAL_INPUT",
            "hota": "NOT_COMPUTABLE_NO_TRACKEVAL_INPUT",
            "assa": "NOT_COMPUTABLE_NO_TRACKEVAL_INPUT",
        },
        "identity_effect": "NOT_COMPUTABLE_SYNTHETIC_ONLY",
    }


def run() -> dict[str, Any]:
    OUT.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, Any]] = []
    validation: dict[str, Any] = {}
    for action in ACTION_TYPES:
        tape = build_tape(action, future_frames=100)
        check = validate_candidate_tape(tape)
        validation[action] = check
        action_result: dict[str, Any] = {
            "action_type": action,
            "interaction_source": "simulated_from_gt",
            "synthetic": True,
            "future_gt_used_runtime": False,
            "validation": check,
            "variants": {},
        }
        if not check["valid"] or not check["candidate_complete"]:
            action_result["status"] = "FAIL"
            action_result["error"] = "synthetic_tape_validation_failed"
            results.append(action_result)
            continue
        action_status = "PASS"
        for name, spec in VARIANTS.items():
            config = replace(
                StateManagerConfig(variant="reid"),
                **{key: value for key, value in spec.items() if key != "description"},
            )
            try:
                replay = paired_replay(
                    tape,
                    config=config,
                    write_branch_uses_appearance_memory=(name != "M0"),
                )
                summary = _variant_summary(replay, action)
                summary["description"] = spec["description"]
                summary["write_branch_uses_appearance_memory"] = bool(name != "M0")
                action_result["variants"][name] = summary
                if summary["status"] != "PASS":
                    action_status = "FAIL"
                del replay
            except Exception as exc:
                action_status = "FAIL"
                action_result["variants"][name] = {
                    "description": spec["description"],
                    "status": "FAIL",
                    "error": f"{type(exc).__name__}: {exc}",
                    "identity_effect": "NOT_COMPUTABLE",
                }
            gc.collect()
        action_result["status"] = action_status
        results.append(action_result)

    real_tape_manifest = {}
    manifest_path = OUT / "tape_manifest.json"
    if manifest_path.exists():
        real_tape_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload = {
        "protocol": "N34_CCAM_M0_M4_PAIRED_REPLAY",
        "status": "PARTIAL" if all(row["status"] == "PASS" for row in results) else "FAIL",
        "real_data_status": "NOT_AVAILABLE",
        "real_candidate_complete": bool(real_tape_manifest.get("candidate_complete", False)),
        "synthetic_fallback_status": "PASS" if all(row["status"] == "PASS" for row in results) else "FAIL",
        "synthetic_not_a_real_data_result": True,
        "identity_effect": "NOT_COMPUTABLE_REAL_DATA_TAPE_UNAVAILABLE",
        "future_gt_used_runtime": False,
        "sequence_bootstrap": {
            "real_sequence_count": 0,
            "synthetic_event_count": len(results),
            "independent_sequence_clusters": 0,
            "ci": "NOT_COMPUTABLE",
            "reason": "synthetic events are not independent real sequence clusters",
        },
        "validation": validation,
        "events": results,
        "metrics": {
            "future_h20_iou": "NOT_COMPUTABLE",
            "future_h50_iou": "NOT_COMPUTABLE",
            "future_h100_iou": "NOT_COMPUTABLE",
            "missing_prediction_rate": "NOT_COMPUTABLE",
            "id_switch_count": "NOT_COMPUTABLE",
            "re_correction_count": "NOT_COMPUTABLE",
            "recovery_latency": "NOT_COMPUTABLE",
            "protected_identity_regression": "NOT_COMPUTABLE",
            "sequence_cluster_bootstrap_ci": "NOT_COMPUTABLE",
            "idf1": "NOT_COMPUTABLE",
            "hota": "NOT_COMPUTABLE",
            "assa": "NOT_COMPUTABLE",
        },
        "interpretation": {
            "synthetic_score_delta": "mechanism_smoke_only",
            "real_ccam_future_effect": "not_claimed",
            "zero_identity_coverage_fallback": "not_used_to_authorize_identity_aware_learning",
        },
    }
    atomic_json(OUT / "ccam_paired_replay_results.json", payload)
    stage = {
        "stage": "N34-4",
        "status": payload["status"],
        "commands": ["python scripts/run_n34_replay.py"],
        "artifacts": ["outputs/n34/ccam_paired_replay_results.json"],
        "errors": [] if payload["status"] != "FAIL" else ["synthetic paired replay failed"],
        "real_data_status": "NOT_AVAILABLE",
        "synthetic_fallback_status": payload["synthetic_fallback_status"],
        "ccam_future_effect": "NOT_COMPUTABLE",
        "next_action": "Apply the N34 authorization gate; do not train a selector/head or decoder LoRA without real sequence-cluster evidence.",
    }
    atomic_json(OUT / "stage_04_status.json", stage)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.parse_args()
    payload = run()
    print(json.dumps({"status": payload["status"], "synthetic": payload["synthetic_fallback_status"], "ccam_future_effect": payload["metrics"]["future_h20_iou"], "output": "outputs/n34/ccam_paired_replay_results.json"}, sort_keys=True))


if __name__ == "__main__":
    main()
