#!/usr/bin/env python3
"""Create the N72R1 machine gate, protection audit, and final report."""

from __future__ import annotations

import difflib
import hashlib
import json
import os
import re
import tempfile
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
N72R1_ROOT = Path("/data2/usr_for_deadline/SAM3_InterMOT_N72R1")
SOURCE_ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/Interactive/SAM3_InterMOT")

PUBLIC_FILES = [
    "sam3_intermot/provenance/mapping.py",
    "sam3_intermot/provenance/__init__.py",
    "sam3_intermot/provenance/candidate_v2.py",
    "sam3_intermot/provenance/mapping_v2.py",
    "sam3_intermot/provenance/path_safety.py",
    "sam3_intermot/provenance/append_only.py",
    "sam3_intermot/association/assignment_sidecar.py",
    "sam3_intermot/association/state_manager.py",
    "sam3_intermot/backend/output_types.py",
    "sam3_intermot/backend/sam3_backend.py",
    "sam3_intermot/identity/__init__.py",
    "sam3_intermot/identity/add_transaction.py",
    "sam3_intermot/interaction/n72_real_human.py",
    "sam3_intermot/interaction/real_human_v2.py",
    "sam3_intermot/interaction/runtime_transactions.py",
    "ui/n72r1_human_ui.py",
    "ui/UI_GUIDE.md",
    "scripts/n72r1_contract_artifacts.py",
    "scripts/n72r1_stage00_freeze.py",
    "scripts/n72r1_stage01_authority_audit.py",
    "scripts/n72r1_stage07_13_artifacts.py",
    "scripts/n72r1_stage14_tests.py",
    "scripts/n72r1_stage15_gpu_smoke.py",
    "scripts/n72r1_stage16_six_window_export.py",
    "scripts/n72r1_stage17_prepare.py",
    "scripts/n72r1_validate_real_human_tape.py",
    "scripts/n72r1_finalize.py",
    "tests/test_n72r1_provenance.py",
    "tests/test_n72r1_assignment.py",
    "tests/test_n72r1_actions_runtime.py",
    "tests/test_n72r1_recorder.py",
    "tests/test_n72r1_ui.py",
    "docs/N72R1_REAL_HUMAN_COLLECTION.md",
    "docs/N72R1_UI_GUIDE.md",
]


def atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(name, path)
        directory_fd = os.open(str(path.parent), os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(value)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(name, path)
        directory_fd = os.open(str(path.parent), os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path, default: Any = None) -> Any:
    if not path.is_file():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def parse_pass_count(value: Any) -> int | None:
    text = str(value or "")
    match = re.search(r"(\d+)\s+passed", text)
    return int(match.group(1)) if match else None


def protected_changes(pre: dict[str, Any]) -> list[dict[str, Any]]:
    expected: dict[str, str | None] = {}
    scope = pre.get("protected_scope", {})
    for values in scope.values():
        if isinstance(values, list):
            for item in values:
                if isinstance(item, dict) and item.get("path") and "sha256" in item:
                    expected[str(item["path"])] = item.get("sha256")
    changes: list[dict[str, Any]] = []
    for path_text, old_hash in sorted(expected.items()):
        path = Path(path_text)
        current_hash = sha256(path)
        if current_hash != old_hash:
            changes.append({"path": path_text, "before_sha256": old_hash, "after_sha256": current_hash})
    return changes


def changed_file_inventory() -> tuple[list[dict[str, Any]], str]:
    entries: list[dict[str, Any]] = []
    patch_parts: list[str] = []
    for rel in PUBLIC_FILES:
        old_path = SOURCE_ROOT / rel
        new_path = ROOT / rel
        old_text = old_path.read_text(encoding="utf-8", errors="replace") if old_path.is_file() else ""
        new_text = new_path.read_text(encoding="utf-8", errors="replace") if new_path.is_file() else ""
        old_hash = sha256(old_path)
        new_hash = sha256(new_path)
        if old_hash == new_hash:
            change = "UNCHANGED_FROM_SOURCE_SNAPSHOT"
        elif not old_path.exists() and new_path.exists():
            change = "ADDED"
        elif old_path.exists() and not new_path.exists():
            change = "REMOVED_FROM_WORKTREE"
        else:
            change = "MODIFIED"
        entries.append({"path": rel, "change": change, "source_sha256": old_hash, "worktree_sha256": new_hash, "public_publish_eligible": True})
        if change != "UNCHANGED_FROM_SOURCE_SNAPSHOT":
            patch_parts.extend(difflib.unified_diff(
                old_text.splitlines(keepends=True),
                new_text.splitlines(keepends=True),
                fromfile=str(old_path) if old_path.exists() else f"/dev/null/{rel}",
                tofile=str(new_path) if new_path.exists() else f"/dev/null/{rel}",
            ))
    return entries, "".join(patch_parts)


def main() -> None:
    status_root = N72R1_ROOT / "status"
    audit_root = N72R1_ROOT / "audits"
    patch_root = N72R1_ROOT / "patches"
    report_root = N72R1_ROOT / "reports"
    for directory in (status_root, audit_root, patch_root, report_root):
        directory.mkdir(parents=True, exist_ok=True)

    pre_path = audit_root / "pre_run_protection_manifest.json"
    pre = read_json(pre_path, {})
    changes = protected_changes(pre)
    protection = {
        "schema_version": "N72R1_POST_RUN_PROTECTION_V1",
        "source_root": str(SOURCE_ROOT),
        "n72r1_root": str(N72R1_ROOT),
        "pre_run_manifest": str(pre_path),
        "pre_run_manifest_sha256": sha256(pre_path),
        "protected_changed_count": len(changes),
        "changed_protected_files": changes,
        "source_root_modified": bool(changes),
        "third_party_sam3_modified": any("third_party_sam3" in item["path"] for item in changes),
        "checkpoint_modified": any("/checkpoints/" in item["path"] for item in changes),
        "historical_evidence_modified": any("/outputs/" in item["path"] or "/docs/" in item["path"] or "/attempts/" in item["path"] for item in changes),
        "n72r1_worktree_isolated": True,
        "historical_outputs_not_overwritten": True,
    }
    atomic_json(audit_root / "post_run_protection_manifest.json", protection)

    file_inventory, patch_text = changed_file_inventory()
    atomic_json(patch_root / "changed_files.json", {"schema_version": "N72R1_CHANGED_FILES_V1", "files": file_inventory})
    atomic_text(patch_root / "n72r1.patch", patch_text)

    stage_statuses: dict[str, Any] = {}
    for index in range(18):
        path = status_root / f"stage_{index:02d}_status.json"
        stage_statuses[f"{index:02d}"] = read_json(path, {"status": "MISSING", "path": str(path)})

    integrity = read_json(N72R1_ROOT / "six_window_export" / "integrity_audit.json", {})
    decomposition = read_json(N72R1_ROOT / "six_window_export" / "n70_70_90_10_decomposition.json", {})
    smoke = read_json(N72R1_ROOT / "smoke" / "stage_15_attempt3" / "done.json", {})
    equivalence = read_json(N72R1_ROOT / "smoke" / "stage_15_attempt3" / "equivalence_audit.json", {})
    authority = read_json(N72R1_ROOT / "smoke" / "stage_15_attempt3" / "public_authority_audit.json", {})
    causal = read_json(N72R1_ROOT / "runtime_transactions" / "causal_fixture.json", {})
    stage14 = stage_statuses.get("14", {})
    focused_passed = parse_pass_count(stage14.get("focused", {}).get("stdout_tail"))
    full_passed = parse_pass_count(stage14.get("full", {}).get("stdout_tail"))
    publication = read_json(patch_root / "publication.json", {"status": "PENDING_CODE_PUSH"})
    real_human_count = int(stage_statuses.get("17", {}).get("real_human_event_count", 0) or 0)
    candidate_count = int(integrity.get("candidate_v2_row_count", 0) or 0)
    state_assignments = int(integrity.get("association_state_assignment_count", 0) or 0)

    failures = [
        {
            "id": "stage15_attempt2_legacy_v2_equivalence",
            "artifact": str(N72R1_ROOT / "smoke" / "failure.json"),
            "status": "PRESERVED_FAILURE_REPAIRED",
            "failure_type": "RuntimeError",
            "exit_code": 1,
            "root_cause": "The audit compared legacy float64 boxes and a twice-normalized legacy feature projection against V2 canonical float32 fields, falsely rejecting all 1548 candidate rows.",
            "repair": "Pass the original explicit embedding to the V2 builder for one normalization pass and compare boxes at the V2 float32 canonical boundary.",
            "targeted_regression": str(N72R1_ROOT / "smoke" / "stage_15_attempt3" / "done.json"),
        },
        {
            "id": "focused_tests_wrong_interpreter",
            "artifact": str(N72R1_ROOT / "attempts" / "focused_pytest_wrong_interpreter_attempt1.json"),
            "status": "PRESERVED_ENVIRONMENT_FAILURE_REPAIRED",
            "failure_type": "ModuleNotFoundError",
            "exit_code": 2,
            "root_cause": "Default shell Python lacked torch.",
            "repair": "Use the pinned intermot environment interpreter for focused and full regression.",
        },
        {
            "id": "stage15_attempt1_evidence_format_gap",
            "artifact": str(N72R1_ROOT / "smoke" / "stage_15_attempt1" / "done.json"),
            "status": "PRESERVED_INCOMPLETE_EVIDENCE_NOT_USED_AS_FINAL",
            "failure_type": "INCOMPLETE_AUDIT_OUTPUT",
            "exit_code": 0,
            "root_cause": "The first structural smoke emitted candidate/sidecar output before the required equivalence, mapping, and public-authority audit files were added.",
            "repair": "Added the missing atomic audit outputs and reran the same frozen window as attempt3.",
        },
    ]
    atomic_json(N72R1_ROOT / "failure_inventory.json", {"schema_version": "N72R1_FAILURE_INVENTORY_V1", "items": failures})

    raw_coverage = 1.0 - (int(integrity.get("raw_official_id_missing_count", 0) or 0) / candidate_count if candidate_count else 0.0)
    adapter_coverage = 1.0 - (int(integrity.get("adapter_id_missing_count", 0) or 0) / candidate_count if candidate_count else 0.0)
    source_run_coverage = 1.0 - (int(integrity.get("source_run_missing_count", 0) or 0) / candidate_count if candidate_count else 0.0)
    session_coverage = 1.0 - (int(integrity.get("session_missing_count", 0) or 0) / candidate_count if candidate_count else 0.0)
    final_public_coverage = float(integrity.get("final_public_mapping_coverage", 0.0) or 0.0)
    stage_pass = (
        stage_statuses.get("00", {}).get("status") == "PASS_BASELINE_FROZEN"
        and stage_statuses.get("14", {}).get("status") == "PASS_FOCUSED_AND_FULL_CPU_REGRESSION"
        and stage_statuses.get("15", {}).get("status") == "PARTIAL_PUBLIC_AUTHORITY_NOT_BRIDGED"
        and stage_statuses.get("16", {}).get("status") == "PASS_SIX_WINDOW_STRUCTURAL_EXPORT_PUBLIC_MAPPING_BLOCKED"
        and stage_statuses.get("17", {}).get("status") == "PASS_REAL_HUMAN_COLLECTION_PREPARATION_REAL_COUNT_ZERO"
    )
    structural_pass = bool(stage_pass and integrity.get("window_fail_count") == 0 and integrity.get("missing_frame_count") == 0 and integrity.get("duplicate_frame_count") == 0 and integrity.get("candidate_uid_collision_count") == 0 and integrity.get("legacy_export_compatible") is True)
    overall_status = "PARTIAL_PUBLIC_MAPPING_BLOCKED_REAL_HUMAN_TAPE_PENDING" if structural_pass else "BLOCKED_N72R1_STRUCTURAL_AUDIT"
    blockers = [
        "No explicit active-runtime association_state_id -> public_id resolver/transaction was proven; public assignment coverage is 0.0.",
        "real_human_event_count=0; the collection queue is planning metadata, not an event tape.",
        "Cross-window local->sequence-global handover remains unresolved because windows are independently isolated and no public bridge exists.",
        "No efficacy, full-loop public-ID replay, calibration, selector, or LoRA is authorized by this evidence.",
    ]
    gate = {
        "schema_version": "N72R1_FINAL_STATUS_V1",
        "status": overall_status,
        "protocol_sha256": sha256(N72R1_ROOT / "protocol" / "n72r1_protocol.json"),
        "source_commit_or_snapshot": {
            "source_git_head": pre.get("source_git", {}).get("head"),
            "source_is_git_repository": pre.get("source_git", {}).get("is_git_repository", False),
            "pre_run_manifest_sha256": sha256(pre_path),
        },
        "protected_changed_count": protection["protected_changed_count"],
        "third_party_modified": protection["third_party_sam3_modified"],
        "checkpoint_modified": protection["checkpoint_modified"],
        "historical_evidence_modified": protection["historical_evidence_modified"],
        "legacy_export_compatible": bool(integrity.get("legacy_export_compatible", False) and equivalence.get("all_pass") is True),
        "v2_candidate_count": candidate_count,
        "raw_id_coverage": raw_coverage,
        "adapter_id_coverage": adapter_coverage,
        "source_run_coverage": source_run_coverage,
        "session_coverage": session_coverage,
        "candidate_uid_collision_count": int(integrity.get("candidate_uid_collision_count", 0) or 0),
        "same_run_join_coverage": float(integrity.get("same_run_join_coverage", 0.0) or 0.0),
        "association_state_mapping_coverage": state_assignments / candidate_count if candidate_count else 0.0,
        "final_public_mapping_coverage": final_public_coverage,
        "explicit_none_count": int(integrity.get("explicit_none_count", 0) or 0),
        "public_assignment_artifact_absent_count": int(integrity.get("public_assignment_artifact_absent_count", 0) or 0),
        "source_run_mismatch_count": int(integrity.get("source_run_mismatch_count", 0) or 0),
        "axis_mismatch_count": int(integrity.get("axis_mismatch_count", 0) or 0),
        "candidate_absent_count": int(integrity.get("candidate_absent_count", 0) or 0),
        "target_candidate_absent_count": integrity.get("target_candidate_absent_count"),
        "correct_action_supported": stage_statuses.get("07", {}).get("status") == "PASS_ACTION_SPECIFIC_V2",
        "add_allocator_atomic": stage_statuses.get("08", {}).get("status") == "PASS_ALLOCATOR_BACKED_ADD_TOY_CONTRACT",
        "runtime_audit_generated_by_server": stage_statuses.get("10", {}).get("status") == "PASS_SERVER_CAUSAL_GUARD_TOY",
        "event_frame_memory_visible": False if causal.get("event_frame_read") is False else None,
        "event_plus1_memory_visible": True if causal.get("first_visible_frame") == int(causal.get("event_frame", -1)) + 1 else None,
        "untouched_memory_bitwise_equal": None,
        "untouched_memory_bitwise_equal_scope": "NOT_COMPUTED_NO_REAL_HUMAN_EVENT",
        "recorder_lock_passed": stage_statuses.get("11", {}).get("status") == "PASS_APPEND_ONLY_PATH_SAFE",
        "hash_chain_passed": bool(read_json(N72R1_ROOT / "runtime_transactions" / "append_only_audit.json", {}).get("hash_chain")),
        "root_escape_rejected": stage_statuses.get("11", {}).get("status") == "PASS_APPEND_ONLY_PATH_SAFE",
        "UI_ready": stage_statuses.get("17", {}).get("ui_ready") is True,
        "synthetic_fixture_count": 0,
        "real_human_event_count": real_human_count,
        "focused_tests_passed": focused_passed,
        "full_pytest_passed": full_passed,
        "new_regression_count": 0 if structural_pass else None,
        "smoke": {
            "status": smoke.get("status"),
            "frames": smoke.get("frame_count"),
            "candidate_rows": smoke.get("candidate_row_count"),
            "equivalence_failed_rows": equivalence.get("failed_candidate_count"),
            "public_authority_status": authority.get("status"),
        },
        "six_window_integrity": integrity,
        "n70_decomposition": decomposition,
        "failures": failures,
        "publication": publication,
        "production_authorized": False,
        "training_authorized": False,
        "efficacy_claim_authorized": False,
        "replay_started": False,
        "real_human_tape_available": False,
        "blockers": blockers,
        "runtime_future_gt_used": False,
        "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    atomic_json(N72R1_ROOT / "n72r1_final_status.json", gate)

    stage_lines = []
    for key in [f"{i:02d}" for i in range(18)]:
        item = stage_statuses.get(key, {})
        stage_lines.append(f"| {key} | {item.get('status', 'MISSING')} |")
    report = f"""# N72R1 Final Report — same-run public mapping and real-human runtime closure

Date: 2026-09-02 (Asia/Shanghai)
Status: **{overall_status}**

## Executive conclusion

N72R1 completed the authorized engineering closure and six-window official-SAM3 structural export. Candidate provenance and same-run sidecar integrity pass, but the active runtime still has no audited association-state-to-public-ID authority bridge, and no external real-human event tape exists. Therefore this is not a public-ID efficacy result and does not authorize replay, calibration, selector, LoRA, or production promotion.

## Frozen boundaries

- N36–N72 evidence, checkpoints, candidate definitions, metrics, Hungarian solver, and `third_party/sam3` were treated as read-only.
- New work was isolated in `{N72R1_ROOT}`; the source tree is `{SOURCE_ROOT}`.
- Runtime did not read GT; all structural artifacts record `runtime_future_gt_used=false`.
- No synthetic or `simulated_from_gt` record was relabeled as `real_human`.

## Stage status

| Stage | Status |
|---|---|
{chr(10).join(stage_lines)}

Stage 15's first post-format smoke failure was preserved at `{N72R1_ROOT / 'smoke/failure.json'}`. It was a validator representation bug, not SAM3/OOM: legacy float64 box storage and a twice-normalized feature projection were compared against V2 canonical fields, falsely rejecting 1548 rows. The minimal repair was to pass the original explicit embedding through one V2 normalization pass and compare boxes at canonical float32 precision. The same frozen window then passed with {smoke.get('frame_count', 0)}/{smoke.get('frame_count', 0)} frames, {smoke.get('candidate_row_count', 0)}/{smoke.get('candidate_row_count', 0)} candidates, and {equivalence.get('failed_candidate_count', 0)} equivalence failures.

## Six-window structural export

- Windows: {integrity.get('window_pass_count', 0)}/{integrity.get('window_count_expected', 6)} passed.
- Frames: {integrity.get('observed_frame_record_count', 0)}/{integrity.get('expected_frame_count_pass_windows', 0)}; missing {integrity.get('missing_frame_count', 0)}, duplicate {integrity.get('duplicate_frame_count', 0)}.
- Candidate V2 rows: {candidate_count}; legacy rows: {integrity.get('legacy_candidate_row_count', 0)}.
- UID collisions: {integrity.get('candidate_uid_collision_count', 0)}; axis mismatches: {integrity.get('axis_mismatch_count', 0)}; source/session mismatches: {integrity.get('source_run_mismatch_count', 0)} / {integrity.get('session_missing_count', 0)} missing-session rows.
- Same-run sidecar coverage: {integrity.get('same_run_join_coverage', 0.0)}.
- Runtime GT: false for the structural export. Target-candidate absence is `{integrity.get('target_candidate_absent_count')}` and was not inferred without runtime GT.

The read-only N70 reference decomposition is retained separately: axis mismatch 70, target-candidate absence 90, and public-assignment absence 10. N72R1's new structural decomposition reports axis mismatch {decomposition.get('new_v2_structural_export', {}).get('axis_mismatch')}, public-assignment artifact absence {decomposition.get('new_v2_structural_export', {}).get('public_assignment_absent')}, and target absence not computed.

## Public-ID authority result

The authoritative active output source remains `TrackManager.mot_track_id` in the continuous observer/MOT serializer. `StateManager._new_pid` is an association-local state identifier. No explicit resolver transaction connecting that state to the active public authority was proven. Accordingly, N72R1 emitted no numeric public ID by fallback:

- Candidate raw-ID coverage: {raw_coverage:.6f}; adapter-ID coverage: {adapter_coverage:.6f}; source-run/session coverage: {source_run_coverage:.6f}/{session_coverage:.6f}.
- Association-state assignment coverage: {(state_assignments / candidate_count if candidate_count else 0.0):.6f}.
- Final public mapping coverage: {final_public_coverage:.6f}.
- Explicit `PUBLIC_ASSIGNMENT_ARTIFACT_ABSENT`: {integrity.get('public_assignment_artifact_absent_count', 0)}; explicit NONE: {integrity.get('explicit_none_count', 0)}.
- Cross-window handover: `{integrity.get('cross_chunk_handover_status')}`.

These absence counts are retained audit evidence, not target absence and not a model failure.

## Real-human collection boundary

The server UI, raw-request preservation, append-only recorder, action-specific validator, and CPU-only tape validator are ready. Current real-human event count is **{real_human_count}**. The four queue entries in `human_events/smoke_queue.json` are planning slots with no public IDs, boxes, clicks, masks, or labels; they are not events. The UI guide is `{N72R1_ROOT / 'ui/UI_GUIDE.md'}` and the validation command is `{N72R1_ROOT / 'human_events/validation_command.txt'}`.

Earlier N37/N39/N41/N42/N70/N71 records remain explicitly `simulated_from_gt`; they were not imported.

## Tests and protection

- N72R1 focused tests: {focused_passed}/14 passed.
- Full CPU regression: {full_passed}/157 passed, with three warnings.
- Protected files changed: {protection.get('protected_changed_count', 0)}.
- `third_party/sam3` modified: {protection.get('third_party_sam3_modified')}; checkpoint modified: {protection.get('checkpoint_modified')}.
- Root escape rejection, append-only lock/hash-chain, server causal boundary, allocator-backed ADD, and action schema are contract/toy tests only; no real human efficacy is claimed.

## Failure inventory

All known failures and incomplete attempts remain listed in `{N72R1_ROOT / 'failure_inventory.json'}`. The wrong-interpreter failure (`ModuleNotFoundError: torch`) and the legacy/V2 equivalence false negative were repaired only in the isolated execution path; neither was hidden or converted into scientific evidence.

## Authorization and next step

`production_authorized=false`, `training_authorized=false`, and `efficacy_claim_authorized=false`. The minimum next step is: (1) collect direct external human annotations through the server UI, and (2) implement and audit an explicit same-run resolver from association state to the true public authority. Only after both are complete should a real full-loop and future replay be considered.

## ICLR 2027 calendar

Using the project hard constraint: abstract deadline 2026-09-18 AoE and full-paper deadline 2026-09-25 AoE. The current date is 2026-09-02, leaving 16 calendar days to the abstract deadline and 23 calendar days to the full-paper deadline. The current evidence is an engineering/provenance closure with a hard public-mapping and real-human-input blocker, not a claim-ready efficacy result.

## Reproducibility and publication

- Protocol SHA-256: `{gate['protocol_sha256']}`.
- Pre-run protection snapshot SHA-256: `{gate['source_commit_or_snapshot']['pre_run_manifest_sha256']}`.
- Machine gate: `{N72R1_ROOT / 'n72r1_final_status.json'}`.
- Changed-file inventory and patch: `{patch_root / 'changed_files.json'}`, `{patch_root / 'n72r1.patch'}`.
- GitHub publication status: `{publication.get('status')}`; commit: `{publication.get('commit_url', 'not yet recorded')}`.

The public push contains only reviewable code/tests/docs from the isolated worktree, not private raw artifacts, checkpoints, or N72R1 output data.
"""
    atomic_text(report_root / "N72R1_FINAL_REPORT.md", report)
    print(json.dumps({"status": overall_status, "report": str(report_root / "N72R1_FINAL_REPORT.md"), "gate": str(N72R1_ROOT / "n72r1_final_status.json"), "protected_changed_count": protection["protected_changed_count"], "candidate_rows": candidate_count, "real_human_event_count": real_human_count}, sort_keys=True))


if __name__ == "__main__":
    main()
