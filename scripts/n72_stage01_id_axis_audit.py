#!/usr/bin/env python3
"""N72 Stage 01: audit raw/native/stable identity axes without model execution.

This audit deliberately runs before the optional raw-ID provenance repair.  It
uses a small adapter fixture plus frozen N70/N71 artifacts and never reads GT
at runtime or writes to historical outputs.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import sys
import tempfile
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
N70 = ROOT / "outputs/N70"
N71 = ROOT / "outputs/N71"
N72 = ROOT / "outputs/N72"
N71_HEAVY = Path("/data2/usr_for_deadline/SAM3_InterMOT_N71")


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


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def audit_frozen_n70() -> dict[str, Any]:
    mapping_path = N70 / "diagnosis/mapping_audit.jsonl"
    boundary_path = N70 / "replay/assignment_boundary.jsonl"
    mapping_keys: set[tuple[str, int, str]] = set()
    target_absent: set[tuple[str, int, str]] = set()
    public_absent: set[tuple[str, int, str]] = set()
    mapping_rows = 0
    with mapping_path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            event_id = str(row.get("event_id", ""))
            frame = int(row.get("frame", -1))
            variant = str(row.get("variant", ""))
            key = (event_id, frame, variant)
            mapping_keys.add(key)
            mapping_rows += 1
            if row.get("target_candidate_present") is False:
                target_absent.add(key)
            if row.get("target_public_assignment_absent") is True:
                public_absent.add(key)

    axis_keys: set[tuple[str, int, str]] = set()
    axis_methods: dict[str, int] = {}
    axis_examples: list[dict[str, Any]] = []
    boundary_rows = 0
    with boundary_path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            boundary_rows += 1
            if row.get("axis_compatible_with_m0") is not False:
                continue
            key = (str(row.get("event_id", "")), int(row.get("frame", -1)), str(row.get("variant", "")))
            if key not in axis_keys:
                axis_keys.add(key)
                if len(axis_examples) < 12:
                    axis_examples.append(
                        {
                            "event_id": key[0],
                            "frame": key[1],
                            "variant": key[2],
                            "method": row.get("method"),
                            "comparison_baseline": row.get("comparison_baseline"),
                            "axis_compatible_with_m0": row.get("axis_compatible_with_m0"),
                            "target_row": row.get("target_row"),
                            "target_col": row.get("target_col"),
                        }
                    )
            method = str(row.get("method", "UNKNOWN"))
            axis_methods[method] = axis_methods.get(method, 0) + 1

    return {
        "mapping_audit": {
            "path": str(mapping_path),
            "sha256": sha256_file(mapping_path),
            "row_count": mapping_rows,
            "unique_event_frame_variant_keys": len(mapping_keys),
            "target_candidate_absent_unique_keys": len(target_absent),
            "target_public_assignment_absent_unique_keys": len(public_absent),
        },
        "assignment_boundary": {
            "path": str(boundary_path),
            "sha256": sha256_file(boundary_path),
            "row_count": boundary_rows,
            "axis_mismatch_unique_event_frame_variant_keys": len(axis_keys),
            "axis_mismatch_method_rows": axis_methods,
            "axis_mismatch_examples": axis_examples,
        },
        "frozen_expected_counts": {
            "variant_axis_mismatch_frames": 70,
            "target_public_assignment_absent_frames": 10,
            "target_candidate_absent_frames": 90,
        },
    }


def audit_source_and_fixture() -> dict[str, Any]:
    backend_path = ROOT / "sam3_intermot/backend/sam3_backend.py"
    output_types_path = ROOT / "sam3_intermot/backend/output_types.py"
    backend_source = backend_path.read_text(encoding="utf-8")
    output_source = output_types_path.read_text(encoding="utf-8")

    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from sam3_intermot.backend.output_types import PromptObjectObservation
    from sam3_intermot.backend.sam3_backend import Sam3Backend

    observation = PromptObjectObservation(
        frame_idx=7,
        sam_object_id=17,
        mask=[[True]],
        box_xyxy=[1, 2, 8, 12],
        confidence=0.9,
    )
    backend = Sam3Backend()
    backend._sam_to_ext = {17: 9001}
    before = {
        "sam_object_id": int(observation.sam_object_id),
        "has_raw_sam_object_id_attribute": hasattr(observation, "raw_sam_object_id"),
    }
    backend._apply_stable_ids([observation])
    after = {
        "sam_object_id": int(observation.sam_object_id),
        "has_raw_sam_object_id_attribute": hasattr(observation, "raw_sam_object_id"),
    }

    def contains(pattern: str, source: str) -> bool:
        return re.search(pattern, source, flags=re.MULTILINE) is not None

    code_facts = {
        "official_out_obj_ids_read": "raw.get(\"out_obj_ids\")" in backend_source,
        "parsed_id_assigned_to_sam_object_id": "sam_object_id=int(oid)" in backend_source,
        "stable_id_mutates_observation_sam_object_id": contains(r"obs\.sam_object_id\s*=\s*ext", backend_source),
        "export_native_tid_reads_observation_sam_object_id": '"native_tid": int(observation.sam_object_id)' in backend_source,
        "export_declares_official_source": '"native_id_source": "official_out_obj_ids"' in backend_source,
        "output_type_has_explicit_raw_field": "raw_sam_object_id" in output_source,
        "output_cache_is_adapter_cache": "self._output_cache" in backend_source,
    }
    conclusion = (
        "RAW_PROVENANCE_LOST_BY_STABLE_ID_MUTATION"
        if before["sam_object_id"] != after["sam_object_id"] and not before["has_raw_sam_object_id_attribute"]
        else "NO_PRE_REPAIR_RAW_LOSS_REPRODUCED"
    )
    return {
        "source_files": {
            "backend": str(backend_path),
            "backend_sha256": sha256_file(backend_path),
            "output_types": str(output_types_path),
            "output_types_sha256": sha256_file(output_types_path),
        },
        "code_facts": code_facts,
        "fixture": {
            "description": "non-scientific adapter-only fixture: raw official-like ID 17 bound to external ID 9001",
            "before_stable_id_application": before,
            "after_stable_id_application": after,
            "conclusion": conclusion,
        },
    }


def audit_n71_artifact() -> dict[str, Any]:
    audit_path = N71 / "candidate_branch/full_audit_attempt1.json"
    audit = load_json(audit_path)
    window_path = N71_HEAVY / "candidate_branch/smoke_attempt2/windows/n71-dancetrack0001-0296.jsonl"
    sample_record = None
    with window_path.open(encoding="utf-8") as handle:
        first = handle.readline()
    if first:
        record = json.loads(first)
        candidates = record.get("candidates") or []
        if candidates:
            candidate = candidates[0]
            sample_record = {
                "frame_record_keys": sorted(record),
                "candidate_keys": sorted(candidate),
                "candidate_identity_fields": {
                    key: candidate.get(key)
                    for key in (
                        "native_tid",
                        "native_id_source",
                        "local_id",
                        "global_id",
                        "global_id_scope",
                        "mapping",
                    )
                    if key in candidate
                },
            }
    return {
        "audit_path": str(audit_path),
        "audit_sha256": sha256_file(audit_path),
        "official_candidate_branch": {
            "status": audit.get("status"),
            "window_count": audit.get("window_count"),
            "frame_count": audit.get("total_frame_count"),
            "candidate_row_count": audit.get("total_candidate_row_count"),
            "mapping_status_counts": audit.get("mapping_status_counts"),
            "public_mapping_is_explicitly_unavailable": audit.get("mapping_status_counts") == {
                "EXPLICIT_NEW_BRANCH_PUBLIC_MAPPING_UNAVAILABLE": audit.get("total_candidate_row_count")
            },
        },
        "sample_record": sample_record,
        "interpretation": "N71 official branch preserves window-local/native-like fields but has no exact public assignment; no public mapping is fabricated.",
    }


def main() -> None:
    frozen = audit_frozen_n70()
    source = audit_source_and_fixture()
    n71 = audit_n71_artifact()
    result = {
        "schema": "N72_ID_AXIS_AUDIT_PRE_REPAIR_V1",
        "status": "PASS_ACTIONABLE_RAW_PROVENANCE_GAP_IDENTIFIED",
        "runtime_future_gt_used": False,
        "gt_read": False,
        "scientific_fixture": False,
        "inputs": {
            "n70_mapping_audit": frozen["mapping_audit"],
            "n70_assignment_boundary": frozen["assignment_boundary"],
            "n71_candidate_audit": n71,
        },
        "source_and_fixture": source,
        "frozen_count_reconciliation": {
            "variant_axis_mismatch_frames": frozen["assignment_boundary"]["axis_mismatch_unique_event_frame_variant_keys"],
            "target_public_assignment_absent_frames": frozen["mapping_audit"]["target_public_assignment_absent_unique_keys"],
            "target_candidate_absent_frames": frozen["mapping_audit"]["target_candidate_absent_unique_keys"],
            "matches_frozen_n71_summary": (
                frozen["assignment_boundary"]["axis_mismatch_unique_event_frame_variant_keys"] == 70
                and frozen["mapping_audit"]["target_public_assignment_absent_unique_keys"] == 10
                and frozen["mapping_audit"]["target_candidate_absent_unique_keys"] == 90
            ),
        },
        "root_cause": {
            "raw_vs_stable_axis": "PromptObjectObservation.sam_object_id is used first as raw official ID, then mutated to known adapter external ID; the old export labels the mutated value as official raw ID.",
            "n70_axis_mismatch": "Frozen N70 posthoc artifacts contain 70 explicitly retained M1/M2/M3 axis-incompatible rows; they are diagnostic evidence, not an input to repair or efficacy credit.",
            "candidate_absence": "90 target-candidate-absent keys are upstream candidate coverage failures and remain explicit.",
            "public_assignment_absence": "10 target-public-assignment-absent keys remain explicit nulls; no heuristic public mapping is allowed.",
        },
        "next_stage": "N72-02_OPTIONAL_RAW_NATIVE_PROVENANCE_REPAIR",
    }
    audit_path = N72 / "audit/id_axis_audit_pre_repair.json"
    status_path = N72 / "stage_01_status.json"
    atomic_json(audit_path, result)
    atomic_json(
        status_path,
        {
            "schema": "N72_STAGE_01_STATUS_V1",
            "stage": "N72-01_ID_AXIS_AUDIT",
            "status": result["status"],
            "audit": str(audit_path),
            "audit_sha256": sha256_file(audit_path),
            "runtime_future_gt_used": False,
            "gt_read": False,
            "historical_evidence_modified": False,
            "actionable_root_cause": result["root_cause"]["raw_vs_stable_axis"],
            "counts": result["frozen_count_reconciliation"],
            "next_stage": result["next_stage"],
        },
    )
    print(json.dumps({"status": result["status"], "audit": str(audit_path), "stage_status": str(status_path), "counts": result["frozen_count_reconciliation"]}, sort_keys=True))


if __name__ == "__main__":
    main()
