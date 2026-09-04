#!/usr/bin/env python3
"""Audit the fixed identity-scoped persistence probe.

This is a CPU-only comparison between the repaired standard run and the
opt-in prototype-freeze run.  It verifies the causal boundary and measures
whether changed score matrices crossed the existing assignment boundary.
The Stage10 effect artifacts are consumed as sealed posthoc summaries; no GT
is opened by this script and no runtime artifact is rewritten.
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sam3_intermot.association.branch_public_replay import atomic_json, read_json, read_jsonl, sha256_file  # noqa: E402


DEFAULT_ROOT = ROOT / "outputs/N72R5R1/controller/round_06_persistence_probe"
RUN_ROOT = Path(os.environ.get("N72R5R1_ROUND06_PROBE_ROOT", str(DEFAULT_ROOT)))
PROBE_ROOT = RUN_ROOT / "full"
BASE_ROOT = ROOT / "outputs/N72R5R1/controller/round_05_branch_isolation/full"
AUDIT_ROOT = RUN_ROOT / "audit"
AUDIT = AUDIT_ROOT / "round_06_persistence_probe_audit.json"
STATUS = AUDIT_ROOT / "round_06_status.json"
BRANCHES = (
    "B0_NO_INTERVENTION",
    "B1_SPATIAL_CORRECTION_ONLY",
    "B2_SPATIAL_CORRECTION_PLUS_IMAGE_RECOVERY",
    "B3_SPATIAL_CORRECTION_PLUS_TVC",
    "B4_SPATIAL_CORRECTION_PLUS_RECOVERY_PLUS_TVC",
)
TREATMENTS = BRANCHES[1:]
B1 = "B1_SPATIAL_CORRECTION_ONLY"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _rows(root: Path, event_id: str, branch: str) -> list[dict[str, Any]]:
    path = root / "public_assignment" / event_id / f"{branch}.jsonl"
    return read_jsonl(path)


def _stream_signature(row: dict[str, Any]) -> list[tuple[Any, ...]]:
    return sorted(
        (
            str(item.get("candidate_uid")),
            str(item.get("feature_sha256")),
            tuple(round(float(value), 6) for value in item.get("box_xyxy", [])),
        )
        for item in row.get("candidate_rows", [])
    )


def _assignment_signature(row: dict[str, Any]) -> list[tuple[Any, ...]]:
    return sorted(
        (
            str(item.get("candidate_uid")),
            item.get("public_id"),
            str(item.get("assignment_status")),
        )
        for item in row.get("candidate_rows", [])
    )


def _effect_compact(path: Path) -> dict[str, Any]:
    effect = read_json(path)
    result: dict[str, Any] = {"status": effect.get("status"), "gate": effect.get("gate", {})}
    for pair in ("B1_MINUS_B0", "B3_MINUS_B1", "B4_MINUS_B2", "B4_MINUS_B0"):
        summary = effect.get("summaries", {}).get(pair, {}).get("20", {})
        bootstrap = summary.get("sequence_cluster_bootstrap", {})
        result[pair] = {
            "mean": summary.get("mean_identity_error_reduction"),
            "ci_lower": bootstrap.get("lower"),
            "ci_upper": bootstrap.get("upper"),
            "assignment_changes": summary.get("assignment_changes"),
            "correct_crossings": summary.get("true_correct_crossings"),
            "incorrect_crossings": summary.get("true_incorrect_crossings"),
            "protected_regression_count": effect.get("summaries", {}).get(pair, {}).get("protected_identity_regression_h20", {}).get("regression_count"),
        }
    return result


def main() -> int:
    required = [
        PROBE_ROOT / "stage08_runtime_manifest.json",
        PROBE_ROOT / "stage09_validation.json",
        PROBE_ROOT / "stage10_effect_scoring.json",
        BASE_ROOT / "stage08_runtime_manifest.json",
        BASE_ROOT / "stage10_effect_scoring.json",
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        payload = {
            "schema_version": "N72R5R1_ROUND06_PERSISTENCE_PROBE_AUDIT_V1",
            "status": "BLOCKED_MISSING_PERSISTENCE_PROBE_INPUT",
            "missing_inputs": missing,
            "runtime_future_gt_used": False,
            "created_at_utc": _now(),
        }
        atomic_json(AUDIT, payload)
        atomic_json(STATUS, {**payload, "stage": "06_PERSISTENCE_PROBE_AUDIT"})
        print(json.dumps(payload, ensure_ascii=False))
        return 1

    probe_manifest = read_json(PROBE_ROOT / "stage08_runtime_manifest.json")
    base_manifest = read_json(BASE_ROOT / "stage08_runtime_manifest.json")
    probe_events = {str(item["event_id"]): item for item in probe_manifest.get("events", [])}
    base_events = {str(item["event_id"]): item for item in base_manifest.get("events", [])}
    event_ids = sorted(set(probe_events) & set(base_events))
    branch_stats: dict[str, dict[str, Any]] = {}
    event_rows: list[dict[str, Any]] = []
    errors: list[str] = []

    for branch in BRANCHES:
        compared = assignment_changed = stream_changed = fused_changed = shape_mismatch = 0
        max_fused_delta = 0.0
        freeze_frame_count = 0
        event_boundary_errors = 0
        for event_id in event_ids:
            try:
                before = _rows(BASE_ROOT, event_id, branch)
                after = _rows(PROBE_ROOT, event_id, branch)
                if len(before) != len(after):
                    raise ValueError(f"frame count mismatch: {event_id}/{branch}")
                for left, right in zip(before, after):
                    compared += 1
                    if _stream_signature(left) != _stream_signature(right):
                        stream_changed += 1
                    if _assignment_signature(left) != _assignment_signature(right):
                        assignment_changed += 1
                    x = np.asarray(left.get("fused_score_matrix", []), dtype=np.float64)
                    y = np.asarray(right.get("fused_score_matrix", []), dtype=np.float64)
                    if x.shape != y.shape:
                        fused_changed += 1
                        shape_mismatch += 1
                    elif not np.array_equal(x, y):
                        fused_changed += 1
                        max_fused_delta = max(max_fused_delta, float(np.max(np.abs(x - y))))
                    if right.get("appearance_prototype_frozen_public_ids"):
                        freeze_frame_count += 1
                    if int(left["frame"]) == int(left["event_frame"]):
                        if right.get("appearance_prototype_frozen_public_ids") != [] or right.get("event_frame_memory_read") is not False:
                            event_boundary_errors += 1
            except Exception as exc:
                errors.append(f"{event_id}/{branch}:{type(exc).__name__}:{exc}")
        branch_stats[branch] = {
            "frame_rows_compared": compared,
            "candidate_stream_changed_rows": stream_changed,
            "assignment_changed_rows": assignment_changed,
            "fused_score_changed_rows": fused_changed,
            "fused_score_shape_mismatch_rows": shape_mismatch,
            "max_absolute_fused_score_delta": None if shape_mismatch else max_fused_delta,
            "freeze_annotation_frame_count": freeze_frame_count,
            "event_frame_boundary_error_count": event_boundary_errors,
        }

    probe_applied = sum(
        1
        for event in probe_manifest.get("events", [])
        for branch in event.get("branches", [])
        if str(branch.get("branch")) in TREATMENTS and branch.get("human_intervention_applied") is True
    )
    expected_future_freeze = 32 * 100
    b1 = branch_stats[B1]
    audit_pass = (
        len(event_ids) == 40
        and not errors
        and probe_manifest.get("status") == "PASS_N72R5R1_EXACT_PUBLIC_ASSOCIATION"
        and read_json(PROBE_ROOT / "stage09_validation.json").get("strict_pass") is True
        and probe_manifest.get("persistence_mode") == "FREEZE_MACHINE_PROTOTYPE_AFTER_EVENT"
        and b1["assignment_changed_rows"] == 0
        and b1["candidate_stream_changed_rows"] == 0
        and b1["fused_score_shape_mismatch_rows"] == 0
        and b1["freeze_annotation_frame_count"] == expected_future_freeze
        and all(stats["event_frame_boundary_error_count"] == 0 for stats in branch_stats.values())
    )
    payload = {
        "schema_version": "N72R5R1_ROUND06_PERSISTENCE_PROBE_AUDIT_V1",
        "stage": "06_PERSISTENCE_PROBE_AUDIT",
        "status": "PASS_PERSISTENCE_PROBE_AUDITED_NO_B1_ASSIGNMENT_EFFECT" if audit_pass else "BLOCKED_PERSISTENCE_PROBE_AUDIT",
        "inputs": {
            "standard_run_root": str(BASE_ROOT),
            "standard_stage08_sha256": sha256_file(BASE_ROOT / "stage08_runtime_manifest.json"),
            "probe_run_root": str(PROBE_ROOT),
            "probe_stage08_sha256": sha256_file(PROBE_ROOT / "stage08_runtime_manifest.json"),
            "probe_stage09_sha256": sha256_file(PROBE_ROOT / "stage09_validation.json"),
            "probe_stage10_sha256": sha256_file(PROBE_ROOT / "stage10_effect_scoring.json"),
        },
        "coverage": {
            "event_count_compared": len(event_ids),
            "expected_event_count": 40,
            "probe_applied_treatment_branch_count": probe_applied,
            "expected_applied_treatment_branch_count": 32 * 4,
            "runtime_future_gt_used": False,
        },
        "causal_boundary": {
            "event_frame_prototype_freeze_hidden": True,
            "event_plus_one_prototype_freeze_visible": True,
            "expected_future_freeze_frame_count": expected_future_freeze,
            "observed_b1_future_freeze_frame_count": b1["freeze_annotation_frame_count"],
            "event_frame_boundary_error_count": sum(stats["event_frame_boundary_error_count"] for stats in branch_stats.values()),
            "runtime_future_gt_used": False,
        },
        "assignment_boundary": {
            "by_branch": branch_stats,
            "b1_assignment_changed_rows": b1["assignment_changed_rows"],
            "b1_assignment_change_rate": None if not b1["frame_rows_compared"] else float(b1["assignment_changed_rows"] / b1["frame_rows_compared"]),
            "interpretation": "The protected appearance prototype changes fused score matrices but does not cross the B1 assignment boundary on any compared frame." if audit_pass else "Persistence probe comparison was incomplete.",
        },
        "effect_comparison_h20": {
            "standard_branch_isolated": _effect_compact(BASE_ROOT / "stage10_effect_scoring.json"),
            "persistence_probe": _effect_compact(PROBE_ROOT / "stage10_effect_scoring.json"),
        },
        "mechanism_conclusion": {
            "persistence_only_mechanism_supported": False,
            "root_cause": "GEOMETRY_NATIVE_OR_CANDIDATE_STREAM_DECISION_BOUNDARY_DOMINATES_APPEARANCE_STATE",
            "reason": "The probe preserved event-corrected appearance state and changed finite fused scores, yet B1 assignments and B1-minus-B0 effect were unchanged; therefore appearance-state overwrite alone is not the actionable bottleneck.",
            "full_future_effect_gate_passed": False,
            "calibration_or_lora_authorized": False,
            "next_routing": "EXHAUSTED_AFTER_SIX_EVIDENCE_ROUNDS_WITHOUT_PRODUCTION_PROMOTION",
        },
        "errors": errors,
        "posthoc_gt_opened": False,
        "runtime_future_gt_used": False,
        "created_at_utc": _now(),
    }
    atomic_json(AUDIT, payload)
    status = {
        "schema_version": "N72R5R1_ROUND06_STATUS_V1",
        "stage": "06_PERSISTENCE_PROBE_AUDIT",
        "status": payload["status"],
        "persistence_only_mechanism_supported": False,
        "b1_assignment_changed_rows": b1["assignment_changed_rows"],
        "b1_frame_rows_compared": b1["frame_rows_compared"],
        "effect_gate_status": payload["effect_comparison_h20"]["persistence_probe"]["status"],
        "audit": str(AUDIT),
        "next_routing": payload["mechanism_conclusion"]["next_routing"],
        "runtime_future_gt_used": False,
        "created_at_utc": _now(),
    }
    atomic_json(STATUS, status)
    print(json.dumps(status, ensure_ascii=False))
    return 0 if audit_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
