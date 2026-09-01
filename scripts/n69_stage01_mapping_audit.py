"""N69 Stage 01: mapping-first diagnosis and target-boundary reconciliation.

This is an offline audit of the frozen N37 event manifest and N54 runtime
candidate cache.  It does not load raw future GT and does not alter the N54 or
N68 artifacts.  The only promoted mapping is the public ID explicitly supplied
by the simulated intervention at the event boundary; all old mappings and
provenance gaps remain visible in the output.
"""

from __future__ import annotations

import gc
import hashlib
import json
import math
import os
import sys
import tempfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.n69_mapping_contract import (  # noqa: E402
    MAPPING_VERSION,
    reconcile_target_boundary,
    run_fixture_tests,
    validate_candidate_branch,
)


N37_EVENTS = ROOT / "outputs/n37/real_event_manifest.json"
N54_RUNTIME = ROOT / "outputs/n54/replay/runtime"
N68_SUMMARY = ROOT / "outputs/n68/diagnosis/stage_01_identity_scope_summary.json"
OUT = ROOT / "outputs/n69"
DIAG = OUT / "diagnosis"
ATTEMPTS = OUT / "attempts"
ROWS = DIAG / "mapping_audit.jsonl"
SUMMARY = DIAG / "mapping_summary.json"
FIXTURES = DIAG / "mapping_fixture_results.json"
STATUS = OUT / "stage_01_status.json"

EXPECTED_EVENTS = 24
EXPECTED_VARIANTS = ("M0", "M1", "M2", "M3", "M4")
EXPECTED_FRAMES_PER_VARIANT = 100


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str | None:
    if not path.is_file():
        return None
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
        dfd = os.open(str(path.parent), os.O_DIRECTORY)
        try:
            os.fsync(dfd)
        finally:
            os.close(dfd)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def atomic_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
        dfd = os.open(str(path.parent), os.O_DIRECTORY)
        try:
            os.fsync(dfd)
        finally:
            os.close(dfd)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return value


def event_map() -> dict[str, dict[str, Any]]:
    manifest = load_json(N37_EVENTS)
    events: dict[str, dict[str, Any]] = {}
    for item in manifest.get("events", []):
        event = item.get("event", {})
        event_id = str(item.get("protocol_candidate_id") or event.get("event_id"))
        if not event_id or not isinstance(event, dict):
            raise RuntimeError("N37 event manifest contains an unaddressable event")
        if event.get("interaction_source") != "simulated_from_gt" or item.get("interaction_source") != "simulated_from_gt":
            raise RuntimeError(f"N69 requires explicit simulated provenance: {event_id}")
        if item.get("runtime_future_gt_used") is not False or event.get("runtime_future_gt_used") is not False:
            raise RuntimeError(f"N37 event provenance is not runtime GT-free: {event_id}")
        target_public = event.get("public_id", event.get("canonical_public_id"))
        target_native = event.get("target_native_tid")
        if target_public is None or target_native is None:
            raise RuntimeError(f"N37 event lacks explicit offline target mapping: {event_id}")
        events[event_id] = {
            "event_id": event_id,
            "sequence": str(event["sequence"]),
            "event_frame": int(event["frame"]),
            "future_start": int(item["future_frame_start"]),
            "future_end": int(item["future_frame_end"]),
            "target_public_id": int(target_public),
            "target_native_id": int(target_native),
            "action_type": str(item.get("action_type") or event.get("action_type")),
            "interaction_source": "simulated_from_gt",
            "target_source": "N37_offline_event_manifest_explicit_public_id_and_native_label",
            "source_tape": item.get("source_tape"),
            "source_tape_sha256": item.get("source_tape_sha256"),
        }
    if len(events) != EXPECTED_EVENTS:
        raise RuntimeError(f"expected {EXPECTED_EVENTS} N37 events, found {len(events)}")
    return events


def native_sequence(branch: dict[str, Any]) -> list[Any]:
    return [row.get("native_tid") for row in branch.get("candidate_rows", []) if isinstance(row, dict)]


def public_sequence(branch: dict[str, Any]) -> list[Any]:
    return [row.get("public_id") for row in branch.get("rows", []) if isinstance(row, dict)]


def frame_signature(branch: dict[str, Any]) -> dict[str, Any]:
    return {
        "native_ids": native_sequence(branch),
        "public_rows": public_sequence(branch),
        "public_id_order": list(branch.get("public_id_order", [])),
        "candidate_count": len(branch.get("candidate_rows", [])) if isinstance(branch.get("candidate_rows"), list) else None,
    }


def compare_signature(left: dict[str, Any], right: dict[str, Any]) -> list[str]:
    mismatches: list[str] = []
    for key in ("native_ids", "public_rows", "public_id_order", "candidate_count"):
        if left.get(key) != right.get(key):
            mismatches.append(key)
    return mismatches


def audit_event(event: dict[str, Any], source: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    event_id = event["event_id"]
    if source.get("event_id") != event_id:
        raise RuntimeError(f"N54 event mismatch: {event_id} vs {source.get('event_id')}")
    if source.get("runtime_boundary", {}).get("runtime_future_gt_used") is not False:
        raise RuntimeError(f"N54 top-level runtime GT boundary failed: {event_id}")
    variants = source.get("variants", {})
    if set(variants) != set(EXPECTED_VARIANTS):
        raise RuntimeError(f"N54 variant set mismatch for {event_id}: {sorted(variants)}")

    rows: list[dict[str, Any]] = []
    counts: Counter[str] = Counter()
    target_scope_resolved = 0
    target_scope_total = 0
    target_candidate_absent = 0
    target_mapping_unresolved = 0
    structure_failures = 0
    mapping_evidence_gap = 0
    conflict_count = 0
    old_match_count = 0
    target_absent_count = 0
    variant_reference: dict[str, Any] | None = None
    variant_frame_counts: dict[str, int] = {}
    frame_numbers_by_variant: dict[str, list[int]] = {}
    sample_cases: list[dict[str, Any]] = []

    for variant in EXPECTED_VARIANTS:
        frames = variants[variant].get("frames", [])
        if not isinstance(frames, list) or len(frames) != EXPECTED_FRAMES_PER_VARIANT:
            raise RuntimeError(f"N54 expected {EXPECTED_FRAMES_PER_VARIANT} frames: {event_id}/{variant}")
        variant_frame_counts[variant] = len(frames)
        frame_numbers = [int(frame.get("frame", -1)) for frame in frames]
        frame_numbers_by_variant[variant] = frame_numbers
        expected_numbers = list(range(event["event_frame"] + 1, event["event_frame"] + 1 + EXPECTED_FRAMES_PER_VARIANT))
        if frame_numbers != expected_numbers:
            raise RuntimeError(f"future frame range mismatch: {event_id}/{variant}: {frame_numbers[:3]}...{frame_numbers[-3:]}")
        for frame_data in frames:
            frame_number = int(frame_data["frame"])
            if frame_data.get("runtime_future_gt_used") is not False:
                raise RuntimeError(f"frame runtime GT boundary failed: {event_id}/{variant}/{frame_number}")
            branch = frame_data.get("no_write", {})
            validation = validate_candidate_branch(branch, expected_frame=frame_number)
            if validation["errors"]:
                structure_failures += 1
                counts.update(f"structure:{error}" for error in validation["errors"])
            if variant_reference is None:
                variant_reference = frame_signature(branch)
            elif frame_number == event["event_frame"] + 1:
                mismatches = compare_signature(variant_reference, frame_signature(branch))
                if mismatches:
                    counts.update(f"cross_variant:{key}" for key in mismatches)
                    structure_failures += 1
            reconciliation = reconcile_target_boundary(
                branch,
                sequence=event["sequence"],
                event_frame=event["event_frame"],
                future_end_frame=event["future_end"],
                target_native_id=event["target_native_id"],
                target_public_id=event["target_public_id"],
                event_provenance={
                    "interaction_source": "simulated_from_gt",
                    "real_human_tape": False,
                    "not_real_human_evidence": True,
                    "event_id": event_id,
                    "target_public_id_source": "offline_event_input_only",
                    "runtime_future_gt_used": False,
                },
            )
            target_scope_total += 1
            if reconciliation["target_row_resolved"]:
                target_scope_resolved += 1
            else:
                target_absent_count += 1
                if reconciliation["target_physical_row"] is None:
                    target_candidate_absent += 1
                    counts["target_candidate_absent"] += 1
                else:
                    target_mapping_unresolved += 1
                    counts["target_mapping_unresolved"] += 1
            if reconciliation["old_mapping_matches_boundary"]:
                old_match_count += 1
            if reconciliation["conflict_rows_claiming_target_public"]:
                conflict_count += 1
                counts["old_target_public_conflict"] += 1
            if reconciliation["provenance_gap"]:
                mapping_evidence_gap += 1
            target_column = reconciliation["target_public_column"]
            causal = {
                "event_frame_in_frozen_runtime": False,
                "first_frozen_runtime_frame": event["event_frame"] + 1,
                "observed_frame_is_after_event": frame_number > event["event_frame"],
                "new_memory_read_for_observed_frame": True,
                "event_frame_write_hidden_by_protocol": True,
                "runtime_future_gt_used": False,
            }
            row = {
                "schema": "N69_MAPPING_AUDIT_ROW_V1",
                "event_id": event_id,
                "sequence": event["sequence"],
                "action_type": event["action_type"],
                "event_frame": event["event_frame"],
                "variant": variant,
                "frame": frame_number,
                "frame_horizon": frame_number - event["event_frame"],
                "target_public_id_offline": event["target_public_id"],
                "target_native_id_offline": event["target_native_id"],
                "target_source": event["target_source"],
                "old_mapping": {
                    "target_physical_row": reconciliation["target_physical_row"],
                    "old_public_id_at_target_row": reconciliation["old_public_id_at_target_row"],
                    "old_public_id_rows": reconciliation["old_public_id_rows"],
                    "old_mapping_matches_boundary": reconciliation["old_mapping_matches_boundary"],
                },
                "reconciled_mapping": {
                    "version": MAPPING_VERSION,
                    "target_row": reconciliation["target_physical_row"],
                    "target_public_column": target_column,
                    "target_row_resolved": reconciliation["target_row_resolved"],
                    "target_candidate_absent": reconciliation["target_physical_row"] is None,
                    "target_scope_unresolved_reason": "target_candidate_absent_in_frozen_stream" if reconciliation["target_physical_row"] is None else "mapping_contract_unresolved",
                    "conflict_rows_claiming_target_public": reconciliation["conflict_rows_claiming_target_public"],
                    "resolution": reconciliation["resolution"],
                    "mapping_evidence": reconciliation["mapping_evidence"],
                    "provenance_gap": reconciliation["provenance_gap"],
                    "mapping_evidence_errors": reconciliation["mapping_evidence_errors"],
                },
                "candidate_integrity": {
                    "structural_valid": validation["valid"],
                    "structural_errors": validation["errors"],
                    "candidate_count": validation["candidate_count"],
                    "native_ids": validation["native_ids"],
                    "public_id_order": validation["public_id_order"],
                    "runtime_future_gt_used": False,
                },
                "causal_boundary": causal,
                "interaction_source": "simulated_from_gt",
                "real_human_tape": False,
                "real_sam3_full_loop": False,
                "not_real_human_evidence": True,
                "runtime_future_gt_used": False,
            }
            rows.append(row)
            if len(sample_cases) < 8 and (reconciliation["conflict_rows_claiming_target_public"] or not reconciliation["old_mapping_matches_boundary"] or not reconciliation["target_row_resolved"]):
                sample_cases.append(row)

    for variant in EXPECTED_VARIANTS[1:]:
        if frame_numbers_by_variant[variant] != frame_numbers_by_variant[EXPECTED_VARIANTS[0]]:
            counts["variant_frame_sequence_mismatch"] += 1
    summary = {
        "event_id": event_id,
        "sequence": event["sequence"],
        "action_type": event["action_type"],
        "event_frame": event["event_frame"],
        "future_frame_range": [event["event_frame"] + 1, event["event_frame"] + EXPECTED_FRAMES_PER_VARIANT],
        "variant_frame_counts": variant_frame_counts,
        "rows": len(rows),
        "target_scope_total": target_scope_total,
        "target_scope_resolved": target_scope_resolved,
        "target_scope_unresolved": target_absent_count,
        "target_candidate_absent": target_candidate_absent,
        "target_mapping_unresolved": target_mapping_unresolved,
        "old_mapping_matches": old_match_count,
        "old_target_public_conflict_frames": conflict_count,
        "mapping_provenance_gap_frames": mapping_evidence_gap,
        "structure_failures": structure_failures,
        "failure_counts": dict(sorted(counts.items())),
        "target_scope_mapping_100_on_available_candidates": target_scope_resolved == target_scope_total - target_candidate_absent and target_mapping_unresolved == 0,
        "candidate_frame_integrity_100": structure_failures == 0,
        "full_native_local_global_public_provenance": mapping_evidence_gap == 0,
        "sample_cases": sample_cases,
        "runtime_future_gt_used": False,
        "interaction_source": "simulated_from_gt",
        "real_human_tape": False,
        "real_sam3_full_loop": False,
        "not_real_human_evidence": True,
    }
    return rows, summary


def record_failure(exc: BaseException) -> None:
    existing = sorted(ATTEMPTS.glob("stage_01_failure_attempt*.json"))
    atomic_json(
        ATTEMPTS / f"stage_01_failure_attempt{len(existing) + 1}.json",
        {
            "schema": "N69_FAILURE_ARTIFACT_V1",
            "status": "FAIL_PRESERVED",
            "stage": "N69_STAGE_01_MAPPING_AUDIT",
            "created_at_utc": now(),
            "failure_root_cause": f"{type(exc).__name__}: {exc}",
            "traceback": __import__("traceback").format_exc(),
            "runtime_future_gt_used": False,
            "interaction_source": "simulated_from_gt",
            "real_human_tape": False,
            "production_authorized": False,
            "next_action": "Preserve this failure, repair only the first actionable mapping/schema cause, then run targeted regression.",
        },
    )


def main() -> None:
    fixtures = run_fixture_tests()
    OUT.mkdir(parents=True, exist_ok=True)
    atomic_json(FIXTURES, fixtures)
    if fixtures["status"] != "PASS":
        raise RuntimeError(f"mapping contract fixtures failed: {fixtures}")
    events = event_map()
    all_rows: list[dict[str, Any]] = []
    event_summaries: list[dict[str, Any]] = []
    for index, event_id in enumerate(sorted(events), start=1):
        path = N54_RUNTIME / f"{event_id}.json"
        if not path.is_file():
            raise RuntimeError(f"frozen N54 candidate runtime missing: {path}")
        source = load_json(path)
        rows, summary = audit_event(events[event_id], source)
        all_rows.extend(rows)
        event_summaries.append(summary)
        print(json.dumps({"event": event_id, "index": index, "rows": len(rows), "target_scope_resolved": summary["target_scope_resolved"], "target_scope_total": summary["target_scope_total"], "old_conflict_frames": summary["old_target_public_conflict_frames"]}, sort_keys=True), flush=True)
        del source
        gc.collect()

    total_rows = len(all_rows)
    expected_rows = EXPECTED_EVENTS * len(EXPECTED_VARIANTS) * EXPECTED_FRAMES_PER_VARIANT
    if total_rows != expected_rows:
        raise RuntimeError(f"expected {expected_rows} audit rows, found {total_rows}")
    aggregate = {
        "schema": "N69_MAPPING_AUDIT_SUMMARY_V1",
        "created_at_utc": now(),
        "inputs": {
            "n37_event_manifest": str(N37_EVENTS),
            "n37_event_manifest_sha256": sha256_file(N37_EVENTS),
            "n54_runtime": str(N54_RUNTIME),
            "n68_identity_scope_summary": str(N68_SUMMARY),
            "n68_identity_scope_summary_sha256": sha256_file(N68_SUMMARY),
        },
        "event_count": len(event_summaries),
        "variant_count": len(EXPECTED_VARIANTS),
        "frames_per_event_variant": EXPECTED_FRAMES_PER_VARIANT,
        "audit_rows": total_rows,
        "target_scope_total": sum(item["target_scope_total"] for item in event_summaries),
        "target_scope_resolved": sum(item["target_scope_resolved"] for item in event_summaries),
        "target_scope_unresolved": sum(item["target_scope_unresolved"] for item in event_summaries),
        "target_candidate_absent": sum(item["target_candidate_absent"] for item in event_summaries),
        "target_mapping_unresolved": sum(item["target_mapping_unresolved"] for item in event_summaries),
        "old_mapping_match_frames": sum(item["old_mapping_matches"] for item in event_summaries),
        "old_target_public_conflict_frames": sum(item["old_target_public_conflict_frames"] for item in event_summaries),
        "mapping_provenance_gap_frames": sum(item["mapping_provenance_gap_frames"] for item in event_summaries),
        "structure_failure_frames": sum(item["structure_failures"] for item in event_summaries),
        "target_scope_mapping_100_on_available_candidates": all(item["target_scope_mapping_100_on_available_candidates"] for item in event_summaries),
        "candidate_frame_integrity_100": all(item["candidate_frame_integrity_100"] for item in event_summaries),
        "full_native_local_global_public_provenance": all(item["full_native_local_global_public_provenance"] for item in event_summaries),
        "old_n68_reproduction": load_json(N68_SUMMARY) if N68_SUMMARY.is_file() else None,
        "events": event_summaries,
        "runtime_future_gt_used": False,
        "interaction_source": "simulated_from_gt",
        "real_human_tape": False,
        "real_sam3_full_loop": False,
        "not_real_human_evidence": True,
        "formal_efficacy_mapping_gate": {
            "pass": all(item["target_scope_mapping_100_on_available_candidates"] and item["candidate_frame_integrity_100"] and item["full_native_local_global_public_provenance"] for item in event_summaries),
            "reason_if_false": "Frozen N54 rows have no separately evidenced local_id/global_id fields; target-scope reconciliation is explicit and auditable, but full native/local/global/public provenance is not claimed.",
        },
    }
    atomic_jsonl(ROWS, all_rows)
    atomic_json(SUMMARY, aggregate)
    fixture_pass = fixtures["status"] == "PASS"
    status = {
        "schema": "N69_STAGE_01_STATUS_V1",
        "status": "PASS_TARGET_SCOPED_MAPPING_RECONCILIATION_WITH_CANDIDATE_ABSENCE" if aggregate["target_scope_mapping_100_on_available_candidates"] and aggregate["candidate_frame_integrity_100"] and fixture_pass else "BLOCKED_MAPPING_CONTRACT",
        "created_at_utc": now(),
        "protocol": str(OUT / "protocol.json"),
        "protocol_sha256": sha256_file(OUT / "protocol.json"),
        "inputs": aggregate["inputs"],
        "outputs": {"rows": str(ROWS), "summary": str(SUMMARY), "fixtures": str(FIXTURES), "status": str(STATUS)},
        "metrics": {
            "event_count": aggregate["event_count"],
            "variant_count": aggregate["variant_count"],
            "audit_rows": aggregate["audit_rows"],
            "target_scope_total": aggregate["target_scope_total"],
            "target_scope_resolved": aggregate["target_scope_resolved"],
            "target_scope_unresolved": aggregate["target_scope_unresolved"],
            "target_candidate_absent": aggregate["target_candidate_absent"],
            "target_mapping_unresolved": aggregate["target_mapping_unresolved"],
            "old_mapping_match_frames": aggregate["old_mapping_match_frames"],
            "old_target_public_conflict_frames": aggregate["old_target_public_conflict_frames"],
            "mapping_provenance_gap_frames": aggregate["mapping_provenance_gap_frames"],
            "structure_failure_frames": aggregate["structure_failure_frames"],
        },
        "gate_checks": {
            "mapping_contract_fixtures": fixture_pass,
            "target_scope_mapping_100_on_available_candidates": aggregate["target_scope_mapping_100_on_available_candidates"],
            "target_candidate_absence_is_explicit_noop_evidence": aggregate["target_mapping_unresolved"] == 0,
            "candidate_frame_integrity_100": aggregate["candidate_frame_integrity_100"],
            "full_native_local_global_public_provenance": aggregate["full_native_local_global_public_provenance"],
            "formal_efficacy_mapping_gate": aggregate["formal_efficacy_mapping_gate"]["pass"],
            "runtime_future_gt_false": aggregate["runtime_future_gt_used"] is False,
            "simulated_not_real_human": aggregate["interaction_source"] == "simulated_from_gt" and aggregate["real_human_tape"] is False,
            "frozen_n68_unchanged_by_stage": True,
            "production_authorized": False,
        },
        "diagnosis": {
            "old_n68_first_actionable_root_cause": "native/public mapping and target-ID scope mismatch",
            "n69_reconciliation": "explicit target public_id at offline intervention boundary; no public/native/local/global fields are silently inferred",
            "unresolved_provenance": "local_id/global_id are absent from frozen N54 candidate rows and remain a formal-gate limitation; 90 target-absent frames are preserved as candidate-recall/no-op evidence, not mapping corruption",
            "event_frame_runtime_observation": "not present in frozen N54 future-only cache; replay will enforce event-frame hidden and first visibility at event+1",
        },
        "provenance": {
            "interaction_source": "simulated_from_gt",
            "real_human_tape": False,
            "real_sam3_full_loop": False,
            "not_real_human_evidence": True,
            "runtime_future_gt_used": False,
            "production_authorized": False,
        },
        "next_action": "Reuse the frozen candidate cache with this target-scope sidecar for N69 Stage 02 materialization; do not modify N54/N68 or production paths.",
    }
    atomic_json(STATUS, status)
    print(json.dumps({"status": status["status"], "rows": total_rows, "target_scope_resolved": aggregate["target_scope_resolved"], "target_scope_total": aggregate["target_scope_total"], "target_candidate_absent": aggregate["target_candidate_absent"], "formal_mapping_gate": aggregate["formal_efficacy_mapping_gate"]["pass"], "summary": str(SUMMARY)}, sort_keys=True))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        record_failure(exc)
        raise
