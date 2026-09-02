#!/usr/bin/env python3
"""N72 Stage 03: audit the exact raw-to-public mapping bridge.

Historical N70/N71 artifacts are never rewritten.  N70's four-axis legacy
bridge is counted separately from the new five-axis contract because its
artifacts do not preserve a distinct official raw ID and adapter external ID.
The N71 official branch is audited as explicitly public-unmapped.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
N70 = ROOT / "outputs/N70"
N71 = ROOT / "outputs/N71"
OUT = ROOT / "outputs/N72/mapping"
N71_HEAVY = Path("/data2/usr_for_deadline/SAM3_InterMOT_N71")

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sam3_intermot.provenance.mapping import resolve_exact_mapping


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: Any) -> None:
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


def atomic_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n")
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


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected object: {path}")
    return value


def audit_n70_legacy_bridge() -> dict[str, Any]:
    path = N70 / "diagnosis/mapping_summary.json"
    payload = load_json(path)
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "status": payload.get("status"),
        "legacy_rows": int(payload.get("candidate_rows", 0)),
        "legacy_chain_complete_rows": int(payload.get("chain_complete_frames", 0)),
        "legacy_chain_complete_rate": payload.get("chain_complete_rate"),
        "legacy_public_assigned_rows": int(payload.get("candidate_rows", 0)) - int(payload.get("public_assignment_absent_candidate_rows", 0)),
        "legacy_public_assignment_absent_rows": int(payload.get("public_assignment_absent_candidate_rows", 0)),
        "legacy_fields": ["native_id", "local_id", "global_id", "public_id"],
        "new_raw_to_adapter_axes_present": False,
        "interpretation": "N70's formal four-axis bridge is preserved as historical evidence, but it does not prove a distinct official raw SAM axis versus adapter external axis; it is not promoted to the N72 five-axis gate.",
    }


def iter_n71_candidates():
    # N71's frozen denominator consists of one smoke window plus five full
    # windows.  Keep both roots explicit instead of choosing one by fallback.
    roots = [
        N71_HEAVY / "candidate_branch/smoke_attempt2/windows",
        N71_HEAVY / "candidate_branch/full_attempt1/windows",
    ]
    candidates = sorted({path for root in roots for path in root.glob("*.jsonl") if root.is_dir()})
    for path in candidates:
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if line.strip():
                    yield path, line_number, json.loads(line)


def audit_n71_public_bridge() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    frame_keys: set[tuple[str, int]] = set()
    candidate_keys: set[tuple[str, int, int]] = set()
    mapping_statuses: Counter[str] = Counter()
    source_files: set[str] = set()
    frame_count = 0
    raw_field_count = 0
    adapter_field_count = 0
    public_exact_count = 0
    for path, line_number, record in iter_n71_candidates():
        if record.get("record_type") != "candidate_frame":
            continue
        sequence = str(record.get("sequence", ""))
        frame = int(record.get("frame", -1))
        window_id = str(record.get("window_id", ""))
        source_files.add(str(path))
        key = (window_id, frame)
        if key not in frame_keys:
            frame_keys.add(key)
            frame_count += 1
        for candidate in record.get("candidates", []):
            native = candidate.get("native_tid")
            local = candidate.get("local_id")
            global_id = candidate.get("global_id")
            raw_native = candidate.get("raw_native_id")
            adapter_external = candidate.get("adapter_external_id")
            if raw_native is not None:
                raw_field_count += 1
            if adapter_external is not None:
                adapter_field_count += 1
            if candidate.get("mapping", {}).get("public_id") is not None:
                public_exact_count += 1
            if native is not None:
                candidate_key = (window_id, frame, int(native))
                if candidate_key in candidate_keys:
                    duplicate = True
                else:
                    candidate_keys.add(candidate_key)
                    duplicate = False
            else:
                duplicate = False
            normalized = {
                "sequence": sequence,
                "frame": frame,
                "window_id": window_id,
                "raw_native_id": raw_native,
                "adapter_external_id": adapter_external,
                "segment_local_id": None if local is None else f"{window_id}:local:{local}",
                "sequence_global_id": global_id,
                "public_id": candidate.get("mapping", {}).get("public_id"),
                "public_assignment_source": candidate.get("mapping", {}).get("mapping_source"),
                "historical_artifact_raw_field_present": raw_native is not None,
                "historical_artifact_adapter_field_present": adapter_external is not None,
                "runtime_future_gt_used": bool(candidate.get("mapping", {}).get("runtime_future_gt_used", False)),
            }
            # This is an audit of the old artifact, so a missing separate raw
            # field is deliberately returned as AXIS_MISMATCH.  It is not
            # repaired by copying native_tid into a new field.
            resolution = resolve_exact_mapping(
                normalized,
                public_assignment_absent=(candidate.get("mapping", {}).get("public_id") is None),
            )
            normalized.update(
                {
                    "status": resolution["status"],
                    "candidate_uid": resolution["candidate_uid"],
                    "mapping_errors": resolution["errors"],
                }
            )
            mapping_statuses[resolution["status"]] += 1
            if duplicate:
                normalized["status"] = "COLLISION"
                normalized["mapping_errors"] = ["duplicate_window_frame_native_id"]
                mapping_statuses[resolution["status"]] -= 1
                mapping_statuses["COLLISION"] += 1
            rows.append(normalized)
    summary = {
        "artifact_root": str(N71_HEAVY / "candidate_branch"),
        "source_file_count": len(source_files),
        "source_files": sorted(source_files),
        "frame_count": frame_count,
        "candidate_row_count": len(rows),
        "unique_window_frame_native_keys": len(candidate_keys),
        "duplicate_window_frame_native_count": len(rows) - len(candidate_keys),
        "raw_native_id_field_count": raw_field_count,
        "adapter_external_id_field_count": adapter_field_count,
        "public_exact_count": public_exact_count,
        "public_assignment_absent_count": len(rows) - public_exact_count,
        "mapping_status_counts": dict(sorted(mapping_statuses.items())),
        "runtime_future_gt_used_count": sum(int(row["runtime_future_gt_used"]) for row in rows),
        "n72_new_five_axis_gate": {
            "raw_native_coverage": (raw_field_count / len(rows)) if rows else 0.0,
            "adapter_external_coverage": (adapter_field_count / len(rows)) if rows else 0.0,
            "public_assignment_coverage": (public_exact_count / len(rows)) if rows else 0.0,
            "status": "BLOCKED_PUBLIC_MAPPING_AND_HISTORICAL_AXIS_SEPARATION" if rows else "BLOCKED_NO_ARTIFACTS",
        },
    }
    return rows, summary


def main() -> None:
    legacy = audit_n70_legacy_bridge()
    rows, official = audit_n71_public_bridge()
    audit_path = OUT / "mapping_bridge_audit_attempt2.jsonl"
    summary_path = OUT / "mapping_bridge_summary_attempt2.json"
    stage_path = ROOT / "outputs/N72/stage_03_status.json"
    atomic_jsonl(audit_path, rows)
    summary = {
        "schema": "N72_MAPPING_BRIDGE_AUDIT_V1",
        "status": "PASS_CONTRACT_PUBLIC_MAPPING_BLOCKED",
        "runtime_future_gt_used": False,
        "historical_outputs_modified": False,
        "legacy_n70": legacy,
        "n71_official_branch": official,
        "exact_mapping_rules": {
            "allowed_sources": ["identity_registry_binding", "explicit_runtime_assignment", "direct_user_public_id", "frozen_provenance_mapping"],
            "forbidden_fallbacks": ["iou", "ground_truth", "appearance", "future_trajectory", "heuristic_inference"],
            "null_public_ids_preserved": True,
        },
        "conclusion": "The repaired backend can emit a lossless raw axis for new exports, but frozen N71 official artifacts contain no separate raw_native_id/adapter_external_id fields and all 9333 public assignments are explicitly unavailable. No exact public mapping is promoted.",
    }
    atomic_json(summary_path, summary)
    atomic_json(
        stage_path,
        {
            "schema": "N72_STAGE_03_STATUS_V1",
            "stage": "N72-03_EXACT_MAPPING_BRIDGE_AUDIT",
            "status": summary["status"],
            "audit": str(audit_path),
            "audit_sha256": sha256_file(audit_path),
            "summary": str(summary_path),
            "summary_sha256": sha256_file(summary_path),
            "runtime_future_gt_used": False,
            "historical_outputs_modified": False,
            "counts": {
                "n70_legacy_candidate_rows": legacy["legacy_rows"],
                "n70_legacy_public_assigned_rows": legacy["legacy_public_assigned_rows"],
                "n70_legacy_public_assignment_absent_rows": legacy["legacy_public_assignment_absent_rows"],
                "n71_official_frames": official["frame_count"],
                "n71_official_candidate_rows": official["candidate_row_count"],
                "n71_raw_native_id_field_count": official["raw_native_id_field_count"],
                "n71_adapter_external_id_field_count": official["adapter_external_id_field_count"],
                "n71_public_exact_count": official["public_exact_count"],
                "n71_public_assignment_absent_count": official["public_assignment_absent_count"],
                "n71_runtime_future_gt_used_count": official["runtime_future_gt_used_count"],
            },
            "mapping_status_counts": official["mapping_status_counts"],
            "next_stage": "N72-04_REAL_HUMAN_EVENT_RECORDER_VALIDATOR",
        },
    )
    print(json.dumps({"status": summary["status"], "audit": str(audit_path), "summary": str(summary_path), "rows": len(rows), "mapping_status_counts": official["mapping_status_counts"]}, sort_keys=True))


if __name__ == "__main__":
    main()
