#!/usr/bin/env python3
"""Posthoc assignment-boundary audit with a strict N72 mapping filter.

The N71 official branch has no public mapping.  This script therefore writes
an empty N72-eligible JSONL rather than fabricating one.  N70's old four-axis
records are summarized as historical context only and are never promoted to
the N72 five-axis contract.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "N72" / "boundary_diagnostic"
N70_MAPPING = ROOT / "outputs" / "N70" / "diagnosis" / "mapping_audit.jsonl"
N70_BOUNDARY = ROOT / "outputs" / "N70" / "replay" / "assignment_boundary.jsonl"
N71_AUDIT = ROOT / "outputs" / "N71" / "candidate_branch" / "full_audit_attempt1.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


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
        directory_fd = os.open(str(path.parent), os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def atomic_empty_jsonl(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        directory_fd = os.open(str(path.parent), os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def main() -> None:
    mapping = {}
    mapping_status_counts: Counter[str] = Counter()
    mapping_candidate_absent_keys: set[tuple[str, int, str]] = set()
    mapping_target_public_absent_keys: set[tuple[str, int, str]] = set()
    mapping_any_public_absent_keys: set[tuple[str, int, str]] = set()
    mapping_rows = 0
    with N70_MAPPING.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            key = (str(row["event_id"]), int(row["frame"]), str(row["variant"]))
            mapping[key] = row
            mapping_rows += 1
            if row.get("target_candidate_present") is not True:
                mapping_candidate_absent_keys.add(key)
            if row.get("target_public_assignment_absent") is True:
                mapping_target_public_absent_keys.add(key)
            if int(row.get("public_assignment_absent_candidate_rows", 0)) > 0:
                mapping_any_public_absent_keys.add(key)

    boundary_rows = 0
    legacy_context = Counter()
    excluded_boundary_rows = Counter()
    axis_keys: set[tuple[str, int, str]] = set()
    boundary_unique_keys: set[tuple[str, str, int]] = set()
    runtime_gt_rows = 0
    with N70_BOUNDARY.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            boundary_rows += 1
            bkey = (str(row.get("event_id")), str(row.get("method")), int(row.get("frame", -1)))
            boundary_unique_keys.add(bkey)
            if row.get("runtime_future_gt_used") is not False:
                runtime_gt_rows += 1
            legacy_context["score_changed_rows"] += int(row.get("score_changed") is True)
            legacy_context["assignment_changed_rows"] += int(row.get("assignment_changed") is True)
            legacy_context["target_assignment_changed_rows"] += int(row.get("target_assignment_changed") is True)
            legacy_context["correct_assignment_change_rows"] += int(row.get("correct_change") is True)
            legacy_context["incorrect_assignment_change_rows"] += int(row.get("incorrect_change") is True)
            legacy_context["untouched_regression_rows"] += int(row.get("untouched_regression") is True)
            key = (str(row.get("event_id")), int(row.get("frame", -1)), str(row.get("variant")))
            mapping_row = mapping.get(key)
            if row.get("axis_compatible_with_m0") is False or row.get("mapping_uncertain") is True:
                axis_keys.add(key)
                excluded_boundary_rows["axis_mismatch_or_uncertain"] += 1
            elif key in mapping_candidate_absent_keys or row.get("target_candidate_present") is not True:
                excluded_boundary_rows["target_candidate_absent"] += 1
            elif key in mapping_any_public_absent_keys or row.get("target_public_assignment_absent") is True:
                excluded_boundary_rows["candidate_public_assignment_absent"] += 1
            elif mapping_row is None:
                excluded_boundary_rows["mapping_join_missing"] += 1
            else:
                # This is only a diagnostic assertion.  N70's old four-axis
                # records lack raw_native_id/adapter_external_id, so this
                # branch is intentionally never emitted as N72 evidence.
                excluded_boundary_rows["legacy_four_axis_not_n72"] += 1

    n71_payload = json.loads(N71_AUDIT.read_text(encoding="utf-8"))
    official = {
        "candidate_row_count": n71_payload.get("total_candidate_row_count", 0),
        "frame_count": n71_payload.get("total_frame_count", 0),
        "mapping_status_counts": n71_payload.get("mapping_status_counts", {}),
    }
    n72_eligible_path = OUT / "n72_exact_assignment_boundary.jsonl"
    atomic_empty_jsonl(n72_eligible_path)

    summary = {
        "schema": "N72_BOUNDARY_DIAGNOSTIC_SUMMARY_V1",
        "status": "BLOCKED_N72_EXACT_MAPPING_UNAVAILABLE",
        "n72_exact_mapping_contract": ["raw_native_id", "adapter_external_id", "segment_local_id", "sequence_global_id", "public_id"],
        "n72_eligible_rows_written": 0,
        "n72_eligible_events": 0,
        "n72_eligible_sequences": 0,
        "n71_official_branch": {
            "candidate_rows": int(official.get("candidate_row_count", 0)),
            "frames": int(official.get("frame_count", 0)),
            "public_mapping_status": official.get("mapping_status_counts", {}),
            "exact_public_mapping_rows": 0,
            "reason": "All N71 official-branch rows explicitly have public mapping unavailable; no N72 exact rows can be emitted.",
        },
        "n70_legacy_context": {
            "mapping_schema": "N70_MAPPING_AUDIT_ROW_V1",
            "boundary_schema": "N70_ASSIGNMENT_BOUNDARY_ROW_V1",
            "mapping_rows": mapping_rows,
            "boundary_rows": boundary_rows,
            "boundary_unique_event_method_frame_keys": len(boundary_unique_keys),
            "variant_axis_mismatch_unique_keys": len(axis_keys),
            "target_candidate_absent_unique_keys": len(mapping_candidate_absent_keys),
            "target_public_assignment_absent_unique_keys": len(mapping_target_public_absent_keys),
            "any_candidate_public_assignment_absent_unique_keys": len(mapping_any_public_absent_keys),
            "excluded_boundary_rows_by_first_strict_reason": dict(sorted(excluded_boundary_rows.items())),
            "descriptive_counts_not_promoted_to_n72": dict(legacy_context),
            "reason_not_promoted": "N70 has the historical native/local/global/public chain but no distinct raw official→adapter axis; its rows are context only.",
        },
        "runtime_future_gt_used_rows_in_context": runtime_gt_rows,
        "no_iou_gt_appearance_mapping_fallback": True,
        "diagnostic_output": str(n72_eligible_path),
        "input_hashes": {
            "n70_mapping": sha256(N70_MAPPING),
            "n70_boundary": sha256(N70_BOUNDARY),
            "n71_official_audit": sha256(N71_AUDIT),
        },
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    atomic_json(OUT / "assignment_boundary_summary.json", summary)
    stage = dict(summary)
    stage.update({"stage": "N72-07_POSTHOC_ASSIGNMENT_BOUNDARY_DIAGNOSTIC", "status": "COMPLETE_DIAGNOSTIC_ONLY_NO_N72_EXACT_ROWS"})
    atomic_json(ROOT / "outputs" / "N72" / "stage_07_status.json", stage)
    print(json.dumps({"status": stage["status"], "n72_eligible_rows": 0, "n70_boundary_rows": boundary_rows, "path": str(ROOT / "outputs" / "N72" / "stage_07_status.json")}, sort_keys=True))


if __name__ == "__main__":
    main()
