#!/usr/bin/env python3
"""CPU-only integrity gate for N36 chunk and merged candidate tapes."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.n36_tape_common import (
    atomic_json,
    decode_mask,
    iter_jsonl,
    load_sequences,
)


def finite_box(value: Any) -> bool:
    try:
        box = np.asarray(value, dtype=float).reshape(-1)
    except (TypeError, ValueError):
        return False
    return bool(box.size == 4 and np.all(np.isfinite(box)) and box[2] >= box[0] and box[3] >= box[1])


def finite_feature(value: Any, dim: int = 512) -> bool:
    try:
        vector = np.asarray(value, dtype=np.float32).reshape(-1)
    except (TypeError, ValueError):
        return False
    return bool(vector.size == dim and np.all(np.isfinite(vector)) and np.linalg.norm(vector) > 1e-6)


def check_candidates(
    row: dict[str, Any], errors: list[str], warnings: list[str], scope: str
) -> int:
    candidates = row.get("candidates")
    if not isinstance(candidates, list):
        errors.append(f"{scope}:candidates_not_list")
        return 0
    if not row.get("candidate_complete") or not row.get("candidate_set_complete"):
        errors.append(f"{scope}:candidate_complete_false")
    audit = row.get("association_audit")
    if not isinstance(audit, dict):
        errors.append(f"{scope}:association_audit_missing")
    else:
        order = audit.get("candidate_order")
        if order != [int(item.get("candidate_index", -1)) for item in candidates]:
            errors.append(f"{scope}:candidate_order_not_stable")
        public_ids = audit.get("candidate_public_ids")
        # Empty frames have no public/native pairs.  Older valid chunk
        # artifacts were emitted before StateManager started serializing the
        # vacuous [] field, so accept a missing field only for zero candidates
        # and retain a visible compatibility warning.  A non-empty frame must
        # still carry one explicit public ID per candidate.
        if public_ids is None and not candidates:
            warnings.append(f"{scope}:empty_candidate_mapping_legacy_compatible")
        elif not isinstance(public_ids, list) or len(public_ids) != len(candidates) or any(pid is None for pid in public_ids):
            errors.append(f"{scope}:candidate_public_mapping_missing")
    for index, candidate in enumerate(candidates):
        prefix = f"{scope}:candidate_{index}"
        if not isinstance(candidate, dict):
            errors.append(f"{prefix}:not_object")
            continue
        if int(candidate.get("candidate_index", -1)) != index:
            errors.append(f"{prefix}:candidate_index_not_contiguous")
        local = candidate.get("local_native_id", candidate.get("native_tid"))
        try:
            if int(local) != int(candidate.get("native_tid")):
                errors.append(f"{prefix}:local_native_id_mismatch")
        except (TypeError, ValueError):
            errors.append(f"{prefix}:native_id_invalid")
        if not finite_box(candidate.get("box")):
            errors.append(f"{prefix}:box_invalid")
        if not finite_feature(candidate.get("machine_embedding")):
            errors.append(f"{prefix}:machine_embedding_invalid")
        if decode_mask(candidate.get("mask")) is None:
            errors.append(f"{prefix}:mask_not_decodable")
        try:
            confidence = float(candidate.get("confidence"))
            if not math.isfinite(confidence):
                raise ValueError
        except (TypeError, ValueError):
            errors.append(f"{prefix}:confidence_invalid")
        try:
            public_id = int(candidate.get("chunk_local_public_id"))
            if public_id < 1:
                raise ValueError
        except (TypeError, ValueError):
            errors.append(f"{prefix}:chunk_local_public_id_invalid")
        if candidate.get("public_native_mapping_status") != "CHUNK_LOCAL_EXPLICIT":
            errors.append(f"{prefix}:public_native_mapping_not_explicit")
    return len(candidates)


def load_rows(path: Path) -> list[dict[str, Any]]:
    return [row for _, row in iter_jsonl(path)]


def validate_chunk(
    chunk: dict[str, Any],
    output_root: Path,
    errors: list[str],
    warnings: list[str],
) -> dict[str, Any]:
    sequence = str(chunk["sequence"])
    chunk_id = str(chunk["chunk_id"])
    scope = f"chunk:{chunk_id}"
    done_path = output_root / "chunk_done" / sequence / f"{chunk_id}.json"
    rows_path = output_root / "chunks" / sequence / f"{chunk_id}.jsonl"
    if not done_path.is_file():
        errors.append(f"{scope}:done_missing")
        return {"status": "NOT_RUN", "chunk_id": chunk_id, "frame_count": 0, "candidate_count": 0}
    try:
        done = json.loads(done_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        errors.append(f"{scope}:done_corrupt")
        return {"status": "FAIL", "chunk_id": chunk_id, "frame_count": 0, "candidate_count": 0}
    if done.get("status") != "PASS" or not done.get("candidate_set_complete"):
        errors.append(f"{scope}:done_not_complete:{done.get('status')}")
    if not rows_path.is_file():
        errors.append(f"{scope}:rows_missing")
        return {"status": "FAIL", "chunk_id": chunk_id, "frame_count": 0, "candidate_count": 0}
    try:
        rows = load_rows(rows_path)
    except Exception as exc:
        errors.append(f"{scope}:rows_corrupt:{type(exc).__name__}")
        return {"status": "FAIL", "chunk_id": chunk_id, "frame_count": 0, "candidate_count": 0}
    frames = [int(row.get("frame", -1)) for row in rows]
    expected = list(range(int(chunk["frame_start"]), int(chunk["frame_end"]) + 1))
    if frames != expected:
        errors.append(f"{scope}:frame_range_mismatch")
    candidate_count = 0
    local_to_public: dict[int, set[int]] = {}
    for row in rows:
        if row.get("protocol") != "N36_REAL_SHARDED_CANDIDATE_TAPE_CHUNK":
            errors.append(f"{scope}:protocol_invalid")
        if row.get("runtime_future_gt_used") is not False or row.get("runtime_gt_read") is not False:
            errors.append(f"{scope}:runtime_gt_flag_invalid")
        if any(str(key).lower().startswith("future") or str(key).lower() in {"gt", "gt_box", "gt_id"} for key in row):
            errors.append(f"{scope}:future_or_gt_key_at_runtime")
        count = check_candidates(row, errors, warnings, f"{scope}:frame_{row.get('frame')}")
        candidate_count += count
        for candidate in row.get("candidates", []):
            try:
                local = int(candidate.get("local_native_id", candidate.get("native_tid")))
                public = int(candidate.get("chunk_local_public_id"))
                local_to_public.setdefault(local, set()).add(public)
            except (TypeError, ValueError):
                pass
    # A raw SAM3 native ID is an observation identifier, not the public
    # identity.  If the online associator changes the public assignment for a
    # native ID, preserve that evidence as an identity-switch signal rather
    # than failing the tape's explicit-mapping gate.
    mapping_transitions = sum(1 for values in local_to_public.values() if len(values) > 1)
    if mapping_transitions:
        warnings.append(f"{scope}:local_native_public_mapping_changes:{mapping_transitions}")
    records = done.get("boundary_mapping", {}).get("records", [])
    if records:
        record_map: dict[int, set[int]] = {}
        for record in records:
            try:
                record_map.setdefault(int(record["local_native_id"]), set()).add(int(record["sequence_global_native_id"]))
            except (KeyError, TypeError, ValueError):
                errors.append(f"{scope}:boundary_mapping_record_invalid")
        if any(len(values) != 1 for values in record_map.values()):
            errors.append(f"{scope}:boundary_local_global_not_unique")
    return {
        "status": "PASS" if not any(item.startswith(scope + ":") for item in errors) else "FAIL",
        "chunk_id": chunk_id,
        "frame_count": len(rows),
        "candidate_count": candidate_count,
        "local_native_id_count": len(local_to_public),
        "local_native_public_mapping_transition_count": int(mapping_transitions),
        "done": done,
    }


def validate_merged(
    sequence: str,
    plan_seq: dict[str, Any],
    output_root: Path,
    errors: list[str],
    warnings: list[str],
) -> dict[str, Any]:
    scope = f"merged:{sequence}"
    done_path = output_root / "done" / f"{sequence}.json"
    rows_path = output_root / "frames" / f"{sequence}.jsonl"
    if not done_path.is_file():
        errors.append(f"{scope}:done_missing")
        return {"status": "NOT_RUN", "frame_count": 0, "candidate_count": 0}
    done = json.loads(done_path.read_text(encoding="utf-8"))
    if done.get("status") != "PASS" or not done.get("candidate_set_complete"):
        errors.append(f"{scope}:done_not_complete:{done.get('status')}")
    if not rows_path.is_file():
        errors.append(f"{scope}:rows_missing")
        return {"status": "FAIL", "frame_count": 0, "candidate_count": 0}
    rows = load_rows(rows_path)
    frames = [int(row.get("frame", -1)) for row in rows]
    expected = list(range(int(plan_seq["frame_count"])))
    if frames != expected:
        errors.append(f"{scope}:merged_frame_coverage_mismatch")
    candidate_count = 0
    global_ids: set[int] = set()
    for row in rows:
        if row.get("protocol") != "N36_REAL_SHARDED_CANDIDATE_TAPE":
            errors.append(f"{scope}:protocol_invalid")
        if row.get("frame_owner_chunk_id") not in row.get("source_chunk_ids", []):
            errors.append(f"{scope}:owner_not_in_sources:{row.get('frame')}")
        if row.get("runtime_future_gt_used") is not False or row.get("runtime_gt_read") is not False:
            errors.append(f"{scope}:runtime_gt_flag_invalid")
        count = check_candidates(row, errors, warnings, f"{scope}:frame_{row.get('frame')}")
        candidate_count += count
        for index, candidate in enumerate(row.get("candidates", [])):
            try:
                global_id = int(candidate["sequence_global_native_id"])
                if global_id < 1:
                    raise ValueError
                global_ids.add(global_id)
            except (KeyError, TypeError, ValueError):
                errors.append(f"{scope}:frame_{row.get('frame')}:candidate_{index}:global_native_mapping_missing")
            if candidate.get("sequence_global_native_id_status") != "EXPLICIT_OVERLAP_RECONCILED":
                errors.append(f"{scope}:frame_{row.get('frame')}:candidate_{index}:global_native_mapping_status_invalid")
    if done.get("boundary_mapping", {}).get("status") != "PASS":
        errors.append(f"{scope}:boundary_mapping_gate_failed")
    return {
        "status": "PASS" if not any(item.startswith(scope + ":") for item in errors) else "FAIL",
        "frame_count": len(rows),
        "candidate_count": candidate_count,
        "global_native_id_count": len(global_ids),
        "done": done,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=ROOT / "outputs/n36/real_tape")
    parser.add_argument("--sequence-list", type=Path, default=ROOT / "outputs/n34/selected_sequences.json")
    parser.add_argument("--sequences", default="")
    parser.add_argument("--audit-output", type=Path, required=True)
    parser.add_argument("--stage-status", type=Path, required=True)
    parser.add_argument("--label", default="smoke")
    args = parser.parse_args()
    plan = json.loads(args.plan.resolve().read_text(encoding="utf-8"))
    sequences = load_sequences(args.sequence_list, args.sequences)
    errors: list[str] = []
    warnings: list[str] = []
    chunk_results = []
    merged_results = []
    for sequence in sequences:
        plan_seq = next(item for item in plan.get("sequences", []) if item.get("sequence") == sequence)
        for chunk in plan_seq.get("chunks", []):
            chunk_results.append(validate_chunk(chunk, args.output_root.resolve(), errors, warnings))
        merged_results.append(validate_merged(sequence, plan_seq, args.output_root.resolve(), errors, warnings))
    error_count = len(errors)
    status = "PASS" if error_count == 0 and all(item["status"] == "PASS" for item in merged_results) else "PARTIAL"
    audit = {
        "protocol": "N36_CPU_TAPE_INTEGRITY_AUDIT",
        "status": status,
        "label": args.label,
        "dataset_split": "train/train_fold",
        "sequence_count_expected": len(sequences),
        "sequence_count_pass": sum(item["status"] == "PASS" for item in merged_results),
        "chunk_count_expected": sum(len(next(item for item in plan["sequences"] if item["sequence"] == sequence).get("chunks", [])) for sequence in sequences),
        "chunk_count_pass": sum(item["status"] == "PASS" for item in chunk_results),
        "frame_count": sum(item.get("frame_count", 0) for item in merged_results),
        "candidate_count": sum(item.get("candidate_count", 0) for item in merged_results),
        "duplicate_frames": 0 if not any("coverage_mismatch" in item for item in errors) else None,
        "missing_frames": 0 if not any("coverage_mismatch" in item or "frame_range_mismatch" in item for item in errors) else None,
        "unavailable_chunks": sum(item["status"] == "NOT_RUN" for item in chunk_results),
        "runtime_gt_read": False,
        "feature_dim": 512,
        "chunk_results": chunk_results,
        "merged_results": merged_results,
        "error_count": error_count,
        "errors": errors[:200],
        "errors_truncated": max(0, error_count - 200),
        "warning_count": len(warnings),
        "warnings": warnings[:200],
        "warnings_truncated": max(0, len(warnings) - 200),
        "third_party_modified": False,
    }
    atomic_json(args.audit_output.resolve(), audit)
    stage = {
        "stage": "N36-03",
        "status": status,
        "label": args.label,
        "artifacts": [str(args.audit_output.resolve().relative_to(ROOT))],
        "errors": errors[:50],
        "warning_count": len(warnings),
        "warnings": warnings[:50],
        "error_count": error_count,
        "sequence_count_expected": len(sequences),
        "sequence_count_pass": audit["sequence_count_pass"],
        "chunk_count_expected": audit["chunk_count_expected"],
        "chunk_count_pass": audit["chunk_count_pass"],
        "frame_count": audit["frame_count"],
        "candidate_count": audit["candidate_count"],
        "next_action": "Proceed to the next N36 smoke ladder step only when this selected-set integrity audit is PASS; otherwise reduce range/overlap or preserve BLOCKED evidence.",
        "third_party_modified": False,
    }
    atomic_json(args.stage_status.resolve(), stage)
    print(json.dumps({"audit": str(args.audit_output.resolve()), "stage": str(args.stage_status.resolve()), "status": status, "error_count": error_count, "sequence_count_pass": audit["sequence_count_pass"], "chunk_count_pass": audit["chunk_count_pass"]}, sort_keys=True))
    # The artifact is the gate.  Keep the command usable for a failed smoke so
    # the next authorized repair can inspect all failures.


if __name__ == "__main__":
    main()
