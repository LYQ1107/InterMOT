#!/usr/bin/env python3
"""Audit the counterfactual-branch oracle-isolation repair.

The frozen Stage07 SAM3 streams are not rerun here.  This CPU-only audit
compares the prior sequential-oracle output with the repaired Stage08 run and
checks that B1--B4 observe the same Y_PRE action precondition while retaining
their own current-event map.  It does not reinterpret a failed future-effect
gate as a pass.
"""

from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sam3_intermot.association.branch_public_replay import atomic_json, read_json, read_jsonl, sha256_file  # noqa: E402


DEFAULT_RUN_ROOT = ROOT / "outputs/N72R5R1/controller/round_05_branch_isolation"
RUN_ROOT = Path(os.environ.get("N72R5R1_ROUND05_ROOT", str(DEFAULT_RUN_ROOT)))
NEW_ROOT = RUN_ROOT / "full"
OLD_ROOT = ROOT / "outputs/N72R5R1/controller/round_04_tvc_v1/full"
AUDIT_ROOT = RUN_ROOT / "audit"
AUDIT = AUDIT_ROOT / "round_05_mechanism_audit.json"
STATUS = AUDIT_ROOT / "round_05_status.json"

BRANCHES = (
    "B0_NO_INTERVENTION",
    "B1_SPATIAL_CORRECTION_ONLY",
    "B2_SPATIAL_CORRECTION_PLUS_IMAGE_RECOVERY",
    "B3_SPATIAL_CORRECTION_PLUS_TVC",
    "B4_SPATIAL_CORRECTION_PLUS_RECOVERY_PLUS_TVC",
)
TREATMENTS = BRANCHES[1:]


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _first_row(root: Path, event_id: str, branch: str) -> dict[str, Any]:
    path = root / "public_assignment" / event_id / f"{branch}.jsonl"
    rows = read_jsonl(path)
    if not rows:
        raise ValueError(f"empty branch sidecar: {path}")
    return rows[0]


def _branch_status(event: Mapping[str, Any], branch: str) -> tuple[Any, Any, Any, Any]:
    result = next(item for item in event.get("branches", []) if str(item.get("branch")) == branch)
    diagnostic = result.get("treatment_diagnostic") or {}
    return (
        diagnostic.get("status"),
        bool(diagnostic.get("human_intervention_applied")),
        result.get("target_public_id"),
        result.get("target_candidate_uid"),
    )


def _effect_compact(path: Path) -> dict[str, Any]:
    effect = read_json(path)
    result: dict[str, Any] = {"status": effect.get("status"), "gate": effect.get("gate", {})}
    for pair in ("B1_MINUS_B0", "B2_MINUS_B1", "B3_MINUS_B1", "B4_MINUS_B2", "B4_MINUS_B0"):
        summary = effect.get("summaries", {}).get(pair, {}).get("20", {})
        bootstrap = summary.get("sequence_cluster_bootstrap", {})
        result[pair] = {
            "mean": summary.get("mean_identity_error_reduction"),
            "ci_lower": bootstrap.get("lower"),
            "ci_upper": bootstrap.get("upper"),
            "assignment_changes": summary.get("assignment_changes"),
            "correct_crossings": summary.get("true_correct_crossings"),
            "incorrect_crossings": summary.get("true_incorrect_crossings"),
            "protected_regression_count": effect.get("summaries", {}).get(pair, {}).get(
                "protected_identity_regression_h20", {}
            ).get("regression_count"),
        }
    return result


def main() -> int:
    required = [
        NEW_ROOT / "stage08_runtime_manifest.json",
        NEW_ROOT / "stage09_validation.json",
        NEW_ROOT / "stage10_effect_scoring.json",
        OLD_ROOT / "stage08_runtime_manifest.json",
        OLD_ROOT / "stage10_effect_scoring.json",
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        payload = {
            "schema_version": "N72R5R1_ROUND05_AUDIT_V1",
            "status": "BLOCKED_MISSING_BRANCH_ISOLATION_INPUT",
            "missing_inputs": missing,
            "runtime_future_gt_used": False,
            "created_at_utc": _now(),
        }
        atomic_json(AUDIT, payload)
        atomic_json(STATUS, {**payload, "stage": "05_BRANCH_ORACLE_ISOLATION_AUDIT"})
        print(json.dumps(payload, ensure_ascii=False))
        return 1

    new_manifest = read_json(NEW_ROOT / "stage08_runtime_manifest.json")
    old_manifest = read_json(OLD_ROOT / "stage08_runtime_manifest.json")
    new_events = {str(item["event_id"]): item for item in new_manifest.get("events", [])}
    old_events = {str(item["event_id"]): item for item in old_manifest.get("events", [])}
    event_ids = sorted(set(new_events) & set(old_events))
    rows: list[dict[str, Any]] = []
    old_inconsistent: list[str] = []
    new_inconsistent: list[str] = []
    y_pre_mismatch: list[str] = []
    mapping_conflict_events: list[str] = []
    memory_boundary_errors: list[str] = []

    for event_id in event_ids:
        new_event = new_events[event_id]
        old_event = old_events[event_id]
        old_statuses = {branch: _branch_status(old_event, branch) for branch in TREATMENTS}
        new_statuses = {branch: _branch_status(new_event, branch) for branch in TREATMENTS}
        old_signature = {(value[0], value[1]) for value in old_statuses.values()}
        new_signature = {(value[0], value[1]) for value in new_statuses.values()}
        if len(old_signature) != 1:
            old_inconsistent.append(event_id)
        if len(new_signature) != 1:
            new_inconsistent.append(event_id)
        if int(new_event.get("branch_mapping_conflict_count", -1)) != 0:
            mapping_conflict_events.append(event_id)
        event_hashes: set[str] = set()
        for branch in BRANCHES:
            first = _first_row(NEW_ROOT, event_id, branch)
            event_hashes.add(str(first.get("shared_y_pre_semantic_hash")))
            if first.get("event_frame_memory_read") is not False or first.get("memory_read") is not False:
                memory_boundary_errors.append(f"{event_id}/{branch}")
        if len(event_hashes) != 1 or "None" in event_hashes:
            y_pre_mismatch.append(event_id)
        rows.append(
            {
                "event_id": event_id,
                "sequence": str(new_event.get("sequence")),
                "old_treatment_statuses": {branch: list(value[:2]) for branch, value in old_statuses.items()},
                "new_treatment_statuses": {branch: list(value[:2]) for branch, value in new_statuses.items()},
                "new_target_public_ids": {branch: value[2] for branch, value in new_statuses.items()},
                "branch_oracle_isolated": bool(new_event.get("branch_oracle_isolated") is True),
            }
        )

    new_effect = _effect_compact(NEW_ROOT / "stage10_effect_scoring.json")
    old_effect = _effect_compact(OLD_ROOT / "stage10_effect_scoring.json")
    isolated_pass = (
        len(event_ids) == 40
        and not new_inconsistent
        and not y_pre_mismatch
        and not memory_boundary_errors
        and not mapping_conflict_events
        and all(row["branch_oracle_isolated"] for row in rows)
        and new_manifest.get("runtime_future_gt_used") is False
    )
    payload = {
        "schema_version": "N72R5R1_ROUND05_MECHANISM_AUDIT_V1",
        "stage": "05_BRANCH_ORACLE_ISOLATION_AUDIT",
        "status": "PASS_BRANCH_ORACLE_ISOLATION_REPAIR_AUDITED" if isolated_pass else "BLOCKED_BRANCH_ORACLE_ISOLATION_AUDIT",
        "inputs": {
            "repaired_stage08_manifest": str(NEW_ROOT / "stage08_runtime_manifest.json"),
            "repaired_stage08_manifest_sha256": sha256_file(NEW_ROOT / "stage08_runtime_manifest.json"),
            "repaired_stage09_validation": str(NEW_ROOT / "stage09_validation.json"),
            "repaired_stage09_validation_sha256": sha256_file(NEW_ROOT / "stage09_validation.json"),
            "repaired_stage10_effect": str(NEW_ROOT / "stage10_effect_scoring.json"),
            "repaired_stage10_effect_sha256": sha256_file(NEW_ROOT / "stage10_effect_scoring.json"),
            "prior_stage08_manifest": str(OLD_ROOT / "stage08_runtime_manifest.json"),
            "prior_stage10_effect": str(OLD_ROOT / "stage10_effect_scoring.json"),
        },
        "coverage": {
            "event_count_compared": len(event_ids),
            "expected_event_count": 40,
            "new_branch_count": int(new_manifest.get("branch_count_completed", 0)),
            "new_stage08_status": new_manifest.get("status"),
        },
        "precondition_consistency": {
            "old_sequential_oracle_inconsistent_event_count": len(old_inconsistent),
            "old_inconsistent_events": old_inconsistent,
            "new_isolated_oracle_inconsistent_event_count": len(new_inconsistent),
            "new_inconsistent_events": new_inconsistent,
            "y_pre_mismatch_event_count": len(y_pre_mismatch),
            "mapping_conflict_event_count": len(mapping_conflict_events),
            "event_frame_memory_boundary_error_count": len(memory_boundary_errors),
        },
        "paired_effect_comparison_h20": {
            "prior_sequential_oracle": old_effect,
            "repaired_isolated_oracle": new_effect,
        },
        "event_rows": rows,
        "mechanism_conclusion": {
            "branch_isolation_repair_verified": bool(isolated_pass),
            "next_root_cause": "SPATIAL_CORRECTION_PERSISTENCE_OR_CANDIDATE_STREAM_DRIFT",
            "reason": "The repair removes cross-branch current-event mapping leakage and changes the downstream counterfactual results, but B1 remains the same harmful spatial-correction branch and the repaired full B4-minus-B0 gate is still not confirmed.",
            "production_promotion": False,
            "calibration_or_lora_authorized": False,
        },
        "runtime_future_gt_used": False,
        "posthoc_gt_opened": True,
        "created_at_utc": _now(),
    }
    atomic_json(AUDIT, payload)
    status = {
        "schema_version": "N72R5R1_ROUND05_STATUS_V1",
        "stage": "05_BRANCH_ORACLE_ISOLATION_AUDIT",
        "status": payload["status"],
        "branch_isolation_repair_verified": bool(isolated_pass),
        "old_inconsistent_event_count": len(old_inconsistent),
        "new_inconsistent_event_count": len(new_inconsistent),
        "full_effect_gate_status": new_effect.get("status"),
        "audit": str(AUDIT),
        "next_round": "AUDIT_SPATIAL_CORRECTION_PERSISTENCE_AND_CANDIDATE_STREAM_DRIFT",
        "runtime_future_gt_used": False,
        "created_at_utc": _now(),
    }
    atomic_json(STATUS, status)
    print(json.dumps(status, ensure_ascii=False))
    return 0 if isolated_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
