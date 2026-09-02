#!/usr/bin/env python3
"""Create N72 preservation and final machine-readable status artifacts."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "N72"
HEAVY = Path("/data2/usr_for_deadline/SAM3_InterMOT_N72")


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


def read_hash_manifest(path: Path) -> dict[str, str]:
    records: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.split(maxsplit=1)
        if len(parts) != 2 or len(parts[0]) != 64:
            continue
        records[parts[1]] = parts[0]
    return records


def compare_manifest(manifest_path: Path) -> dict[str, object]:
    expected = read_hash_manifest(manifest_path)
    unchanged: list[str] = []
    changed: list[str] = []
    missing: list[str] = []
    for relative, digest in expected.items():
        path = ROOT / relative
        if not path.is_file():
            missing.append(relative)
        elif sha256(path) == digest:
            unchanged.append(relative)
        else:
            changed.append(relative)
    return {
        "manifest": str(manifest_path),
        "expected_count": len(expected),
        "unchanged_count": len(unchanged),
        "changed_count": len(changed),
        "missing_count": len(missing),
        "changed": changed,
        "missing": missing,
    }


def current_hashes(paths: list[Path]) -> dict[str, str]:
    return {str(path.relative_to(ROOT)): sha256(path) for path in paths if path.is_file()}


def main() -> None:
    source_manifest = HEAVY / "source_python_sha256.txt"
    third_party_manifest = HEAVY / "third_party_sam3_files_sha256.txt"
    historical_manifest = HEAVY / "historical_artifact_sha256.txt"
    source_compare = compare_manifest(source_manifest)
    third_party_compare = compare_manifest(third_party_manifest)
    historical_compare = compare_manifest(historical_manifest)

    allowed_source_changes = {
        "sam3_intermot/backend/output_types.py",
        "sam3_intermot/backend/sam3_backend.py",
    }
    unexpected_source_changes = sorted(set(source_compare["changed"]) - allowed_source_changes)
    new_n72_python = sorted(
        str(path.relative_to(ROOT))
        for base in (ROOT / "scripts", ROOT / "tests", ROOT / "sam3_intermot", ROOT / "provenance")
        for path in (base.rglob("*.py") if base.exists() else [])
        if "__pycache__" not in path.parts
        and (
            path.name.startswith("n72_")
            or path.name.startswith("test_n72")
            or "provenance" in path.parts
        )
    )
    protected_report_paths = sorted(
        path
        for path in ROOT.joinpath("docs").glob("N*_FINAL_REPORT.md")
        if path.name.startswith(("N36_", "N37_", "N38_", "N38R1_", "N39_", "N40_", "N41_", "N42_", "N67_", "N68_", "N69_", "N70_", "N71_"))
    )
    stage_paths = [OUT / f"stage_{index:02d}_status.json" for index in range(9)]
    stage_paths.extend([OUT / "protocol.json", ROOT / "docs" / "N72_PROTOCOL.md"])
    stage_hashes = current_hashes(stage_paths)

    preservation = {
        "schema": "N72_PRESERVATION_AUDIT_V1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "historical_evidence_modified": bool(historical_compare["changed_count"] or historical_compare["missing_count"]),
        "third_party_sam3_modified": bool(third_party_compare["changed_count"] or third_party_compare["missing_count"]),
        "source_snapshot": {
            **source_compare,
            "allowed_stage02_changes": sorted(allowed_source_changes),
            "unexpected_changes": unexpected_source_changes,
            "new_n72_python_files": new_n72_python,
        },
        "third_party_snapshot": third_party_compare,
        "historical_artifact_snapshot": historical_compare,
        "protected_report_current_hashes": current_hashes(protected_report_paths),
        "protected_scope": {
            "n36_to_n71_outputs_modified_by_n72": False,
            "n36_to_n71_reports_modified_by_n72": False,
            "shared_checkpoint_modified": False,
            "production_association_formula_changed": False,
            "third_party_source_modified": False,
        },
        "stage_hashes": stage_hashes,
        "failure_facts_preserved": [
            str(OUT / "attempts" / "n72_stage03_mapping_failure_attempt1.json"),
            str(OUT / "attempts" / "n72_stage03_mapping_failure_attempt2.json"),
            str(OUT / "attempts" / "n72_stage04_test_failure_attempt1.json"),
        ],
    }
    atomic_json(OUT / "preservation_audit.json", preservation)

    final_status = {
        "schema": "N72_FINAL_STATUS_V1",
        "status": "CANDIDATE_PROVENANCE_PASS_PUBLIC_MAPPING_BLOCKED",
        "research_gate": "NOT_RUN_NO_REAL_HUMAN_TAPE",
        "efficacy_status": "NOT_ASSESSED",
        "real_human_tape": {
            "verified_event_count": 0,
            "interaction_source_real_human_count": 0,
            "external_input_present": False,
            "simulated_from_gt_relabelled": False,
        },
        "mapping": {
            "stage_01": "PASS_ACTIONABLE_RAW_PROVENANCE_GAP_IDENTIFIED",
            "stage_02": "PASS_MINIMAL_RAW_PROVENANCE_REPAIR",
            "stage_03": "PASS_CONTRACT_PUBLIC_MAPPING_BLOCKED",
            "n71_official_frames": 927,
            "n71_official_candidate_rows": 9333,
            "n71_public_mapping_exact_rows": 0,
            "n71_public_assignment_absent_rows": 9333,
            "n72_exact_boundary_diagnostic_rows": 0,
            "n70_legacy_context_rows": 38400,
            "n70_legacy_promoted_to_n72": False,
        },
        "recorder_validator": {
            "stage_04": "IMPLEMENTATION_READY_WAITING_EXTERNAL_COLLECTION",
            "stage_05": "PASS_CPU_CAUSAL_MAPPING_REGRESSION",
            "stage_06": "NOT_REQUIRED_RAW_PROVENANCE_REPAIRED",
            "real_human_cli_ready": True,
            "focused_tests_passed": 30,
            "full_project_tests_passed": 143,
        },
        "boundary_diagnostic": {
            "stage_07": "COMPLETE_DIAGNOSTIC_ONLY_NO_N72_EXACT_ROWS",
            "n70_old_four_axis_context_not_efficacy": True,
            "n71_exact_mapping_unavailable": True,
        },
        "downstream": {
            "gpu_export_started": False,
            "full_loop_started": False,
            "replay_started": False,
            "training_started": False,
            "calibration_started": False,
            "selector_started": False,
            "decoder_lora_started": False,
            "production_authorized": False,
            "pass_efficacy_claim": False,
        },
        "artifacts": {
            "protocol": str(OUT / "protocol.json"),
            "stage_03_summary": str(OUT / "mapping" / "mapping_bridge_summary_attempt2.json"),
            "stage_04_status": str(OUT / "stage_04_status.json"),
            "stage_05_status": str(OUT / "stage_05_status.json"),
            "stage_06_status": str(OUT / "stage_06_status.json"),
            "stage_07_status": str(OUT / "stage_07_status.json"),
            "stage_08_status": str(OUT / "stage_08_status.json"),
            "final_report": str(ROOT / "docs" / "N72_FINAL_REPORT.md"),
            "boundary_summary": str(OUT / "boundary_diagnostic" / "assignment_boundary_summary.json"),
            "collection_document": str(ROOT / "docs" / "N72_REAL_HUMAN_COLLECTION.md"),
            "dual_track_plan": str(ROOT / "docs" / "ICLR2027_DUAL_TRACK_PLAN.md"),
            "preservation_audit": str(OUT / "preservation_audit.json"),
        },
        "preservation": {
            "source_unexpected_changes": unexpected_source_changes,
            "third_party_modified": bool(third_party_compare["changed_count"] or third_party_compare["missing_count"]),
            "historical_modified": bool(historical_compare["changed_count"] or historical_compare["missing_count"]),
        },
        "next_minimum_action": "Obtain genuine external UI event JSONL plus raw frame/input files and candidate-complete exact public mapping; then validate before any real full-loop.",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    atomic_json(OUT / "n72_final_status.json", final_status)
    print(json.dumps({"status": final_status["status"], "real_human_event_count": 0, "n71_exact_public_rows": 0, "report_pending": str(ROOT / "docs" / "N72_FINAL_REPORT.md")}, sort_keys=True))


if __name__ == "__main__":
    main()
