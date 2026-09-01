#!/usr/bin/env python3
"""CPU-only structural audit for an N38R1 sidecar manifest.

The audit consumes already-written JSON only.  It deliberately avoids a
recursive walk through feature/mask payload lists and releases each large
artifact before reading the next one.  It writes its result atomically so a
long audit cannot lose its final status when the terminal wrapper times out.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import sys
import traceback
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.n36_real_eval_common import atomic_json  # noqa: E402


VARIANTS = ("M0", "M1", "M2", "M3", "M4")
REQUIRED_AUDIT_FIELDS = (
    "candidates",
    "candidate_native_ids",
    "candidate_order",
    "public_id_order",
    "scores",
    "base_scores_before_appearance",
    "appearance_memory_scores",
    "appearance_score_deltas",
    "fused_scores",
    "public_id_score_matrix",
    "public_id_base_score_matrix",
    "public_id_appearance_score_matrix",
    "public_id_fused_score_matrix",
    "assignment",
    "assignment_after_scope",
    "assignment_pairs",
    "assignment_pairs_after_scope",
    "public_id_to_native_tid",
    "candidate_records",
    "candidate_rank_by_state",
    "hungarian_cost_audit",
    "target_state_top_two",
)


def finite_feature(value: Any) -> bool:
    return (
        isinstance(value, list)
        and len(value) == 512
        and all(isinstance(item, (int, float)) and math.isfinite(float(item)) for item in value)
    )


def finite_matrix(value: Any, rows: int, columns: int) -> bool:
    if not isinstance(value, list) or len(value) != rows:
        return False
    return all(
        isinstance(row, list)
        and len(row) == columns
        and all(isinstance(item, (int, float)) and math.isfinite(float(item)) for item in row)
        for row in value
    )


def dict_key_violations(mapping: Any, *, allow_event_gt_box: bool = False) -> list[str]:
    """Inspect dict keys without descending into scalar feature/mask lists."""
    forbidden = {
        "future_gt",
        "future_image",
        "future_features",
        "future_candidate_outcomes",
        "gt",
        "gt_id",
        "dataset_identity",
        "reward",
        "selected_candidate",
        "candidate_outcome",
    }
    found: list[str] = []

    def visit(value: Any, path: str) -> None:
        if isinstance(value, dict):
            for key, child in value.items():
                normalized = str(key).strip().lower()
                if normalized.startswith("future_") or normalized in forbidden:
                    found.append(f"{path}/{key}")
                if normalized == "gt_box" and not allow_event_gt_box:
                    found.append(f"{path}/{key}")
                if isinstance(child, dict):
                    visit(child, f"{path}/{key}")
                elif isinstance(child, list) and child and all(isinstance(item, dict) for item in child):
                    for index, item in enumerate(child):
                        visit(item, f"{path}/{key}/{index}")
        elif isinstance(value, list) and value and all(isinstance(item, dict) for item in value):
            for index, item in enumerate(value):
                visit(item, f"{path}/{index}")

    visit(mapping, "")
    return found


def audit_artifact(path: Path, expected_event_id: str, expected_variant: str) -> tuple[dict[str, Any], dict[str, int]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    counts = {
        "frames": 0,
        "candidate_records": 0,
        "feature_512_finite": 0,
        "source_machine_feature_512_finite": 0,
        "mask_hash_present": 0,
        "mapping_complete_frames": 0,
        "runtime_future_gt_true": 0,
        "current_frame_write_hidden": 0,
    }
    errors: list[str] = []
    if payload.get("status") != "PASS":
        errors.append(f"status:{payload.get('status')}")
    if payload.get("event_id") != expected_event_id:
        errors.append("event_id_mismatch")
    if payload.get("variant") != expected_variant:
        errors.append("variant_mismatch")
    if payload.get("runtime_future_gt_used") is not False:
        errors.append("top_level_runtime_future_gt_not_false")
        counts["runtime_future_gt_true"] += int(payload.get("runtime_future_gt_used") is True)
    event_frame = int(payload.get("event_frame"))
    end_frame = int(payload.get("future_frame_end"))
    expected_future = list(range(event_frame + 1, end_frame + 1))
    event_part = payload.get("event_frame_audit") or {}
    if not (
        event_part.get("is_event_frame") is True
        and event_part.get("is_future_frame") is False
        and event_part.get("memory_write") is False
        and event_part.get("memory_read") is False
        and event_part.get("current_frame_write_hidden") is True
        and event_part.get("runtime_future_gt_used") is False
        and event_part.get("gt_loaded_posthoc") is False
    ):
        errors.append("event_frame_contract")
    else:
        counts["current_frame_write_hidden"] += 1
    event_audit = event_part.get("candidate_audit")
    if not isinstance(event_audit, dict) or event_audit.get("frame") != event_frame:
        errors.append("event_frame_audit_missing_or_wrong_frame")
    branches = payload.get("branches")
    if not isinstance(branches, dict) or set(branches) != {"memory_write=False", "memory_write=True"}:
        errors.append("branch_keys")
        branches = {}
    for branch_name, branch in branches.items():
        trace = branch.get("future_trace") if isinstance(branch, dict) else None
        frames = [int(entry.get("frame")) for entry in trace] if isinstance(trace, list) else []
        if frames != expected_future:
            errors.append(f"{branch_name}:future_window")
        for entry in trace if isinstance(trace, list) else []:
            counts["frames"] += 1
            if not (
                entry.get("is_event_frame") is False
                and entry.get("is_future_frame") is True
                and entry.get("runtime_future_gt_used") is False
                and entry.get("gt_loaded_posthoc") is False
            ):
                errors.append(f"{branch_name}:{entry.get('frame')}:frame_flags")
            audit = entry.get("candidate_audit")
            if not isinstance(audit, dict):
                errors.append(f"{branch_name}:{entry.get('frame')}:audit_missing")
                continue
            if audit.get("runtime_future_gt_used") is not False or audit.get("gt_loaded_posthoc") is not False:
                errors.append(f"{branch_name}:{entry.get('frame')}:audit_gt_flag")
            if audit.get("candidate_public_id_mapping_complete") is not True:
                errors.append(f"{branch_name}:{entry.get('frame')}:mapping_incomplete")
            else:
                counts["mapping_complete_frames"] += 1
            for field in REQUIRED_AUDIT_FIELDS:
                if field not in audit:
                    errors.append(f"{branch_name}:{entry.get('frame')}:missing:{field}")
            raw_candidates = audit.get("candidates", [])
            source_records = audit.get("candidate_records", [])
            state_ids = audit.get("public_id_order", [])
            if not isinstance(raw_candidates, list) or not isinstance(source_records, list):
                continue
            if len(raw_candidates) != len(source_records):
                errors.append(f"{branch_name}:{entry.get('frame')}:candidate_record_count")
            counts["candidate_records"] += len(source_records)
            if not finite_matrix(audit.get("fused_scores"), len(raw_candidates), len(state_ids)):
                errors.append(f"{branch_name}:{entry.get('frame')}:fused_matrix")
            ranks = audit.get("candidate_rank_by_state")
            if not isinstance(ranks, list) or len(ranks) != len(raw_candidates) or any(
                not isinstance(row, list) or len(row) != len(state_ids) for row in ranks
            ):
                errors.append(f"{branch_name}:{entry.get('frame')}:rank_shape")
            costs = (audit.get("hungarian_cost_audit") or {}).get("cost_matrix")
            if not finite_matrix(costs, len(raw_candidates), len(state_ids)):
                errors.append(f"{branch_name}:{entry.get('frame')}:cost_matrix")
            for candidate in source_records:
                feature = candidate.get("feature")
                source = candidate.get("source_candidate") or {}
                if finite_feature(feature):
                    counts["feature_512_finite"] += 1
                else:
                    errors.append(f"{branch_name}:{entry.get('frame')}:runtime_feature")
                if finite_feature(source.get("machine_embedding")):
                    counts["source_machine_feature_512_finite"] += 1
                else:
                    errors.append(f"{branch_name}:{entry.get('frame')}:source_feature")
                if source.get("mask_hash") is not None:
                    counts["mask_hash_present"] += 1
                if dict_key_violations(source):
                    errors.append(f"{branch_name}:{entry.get('frame')}:forbidden_source_key")
            if dict_key_violations(audit):
                errors.append(f"{branch_name}:{entry.get('frame')}:forbidden_audit_key")
            source_meta = audit.get("source_row_metadata") or {}
            if source_meta.get("runtime_future_gt_used") is not False or source_meta.get("runtime_gt_read") is not False:
                errors.append(f"{branch_name}:{entry.get('frame')}:source_gt_flag")
    if event_audit is not None:
        if event_audit.get("candidate_public_id_mapping_complete") is not True:
            errors.append("event_mapping_incomplete")
        for field in REQUIRED_AUDIT_FIELDS:
            if field not in event_audit:
                errors.append(f"event:missing:{field}")
        if dict_key_violations(event_audit):
            errors.append("event:forbidden_audit_key")
    result = {
        "status": "PASS" if not errors else "FAIL",
        "event_id": expected_event_id,
        "variant": expected_variant,
        "path": str(path.resolve().relative_to(ROOT)),
        "error_count": len(errors),
        "errors": errors[:100],
        "counts": counts,
    }
    return result, counts


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-artifacts", type=int, default=None)
    args = parser.parse_args()
    payload: dict[str, Any]
    try:
        manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
        records = manifest.get("policy_rows", [])
        results: list[dict[str, Any]] = []
        total = {
            "artifacts": 0,
            "frames": 0,
            "candidate_records": 0,
            "feature_512_finite": 0,
            "source_machine_feature_512_finite": 0,
            "mask_hash_present": 0,
            "mapping_complete_frames": 0,
            "runtime_future_gt_true": 0,
            "current_frame_write_hidden": 0,
        }
        for record in records:
            result, counts = audit_artifact(
                ROOT / record["path"], str(record["event_id"]), str(record["variant"])
            )
            results.append(result)
            total["artifacts"] += 1
            for key in total:
                if key != "artifacts":
                    total[key] += int(counts.get(key, 0))
            del result, counts
            gc.collect()
        expected = (
            int(args.expected_artifacts)
            if args.expected_artifacts is not None
            else (120 if manifest.get("scope") == "all" else len(records))
        )
        payload = {
            "protocol": "N38R1_CPU_ONLY_SIDECAR_SCHEMA_AUDIT_V1",
            "status": "PASS" if len(records) == expected and all(item["status"] == "PASS" for item in results) else "FAIL",
            "source_manifest": str(args.manifest.resolve().relative_to(ROOT)),
            "expected_artifacts": expected,
            "audited_artifacts": len(results),
            "counts": total,
            "results": results,
            "runtime_future_gt_used": False,
            "audit_is_read_only": True,
        }
        atomic_json(args.output, payload)
        print(json.dumps({"status": payload["status"], "counts": total}, sort_keys=True), flush=True)
        return 0 if payload["status"] == "PASS" else 1
    except Exception as exc:
        payload = {
            "protocol": "N38R1_CPU_ONLY_SIDECAR_SCHEMA_AUDIT_V1",
            "status": "FAIL",
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
            "runtime_future_gt_used": False,
            "audit_is_read_only": True,
        }
        atomic_json(args.output, payload)
        print(payload["error"], flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
