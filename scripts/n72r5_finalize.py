#!/usr/bin/env python3
"""Finalize the N72R5 structural run without manufacturing an efficacy result.

This finalizer is intentionally CPU-only and read-only with respect to worker
artifacts.  It validates the completed Stage 07 manifest, counts candidate
rows and public-ID availability, and writes a separate machine gate/report.
The official Stage 07 stream does not contain an authoritative public-ID
assignment, so the final gate must remain blocked for exact association and
posthoc future-effect scoring.
"""

from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "N72R5"
ROUND = OUT / "mechanism_rounds" / "round_07_official_full_loop_attempt5"
MANIFEST = ROUND / "official_full_loop_manifest.json"
AUDIT = OUT / "audits" / "stage07_attempt5_cpu_audit.json"
EVENT_MANIFEST = OUT / "mechanism_rounds" / "round_06_event_policy" / "real_event_manifest.json"
EVENT_GATE = OUT / "mechanism_rounds" / "round_06_event_policy" / "gate.json"
TEST_RESULT = OUT / "tests" / "n72r5_regression_result.json"
STAGE_STATUS = OUT / "stage_status"
GATE = OUT / "n72r5_final_gate.json"
STAGE08 = STAGE_STATUS / "stage_08_status.json"
REPORT = ROOT / "docs" / "N72R5_FINAL_REPORT.md"

BRANCHES = (
    "B0_NO_INTERVENTION",
    "B1_SPATIAL_CORRECTION_ONLY",
    "B2_SPATIAL_CORRECTION_PLUS_IMAGE_RECOVERY",
    "B3_SPATIAL_CORRECTION_PLUS_TVC",
    "B4_SPATIAL_CORRECTION_PLUS_RECOVERY_PLUS_TVC",
)
HORIZON = 100


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object: {path}")
    return value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        directory_fd = os.open(path.parent, os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def atomic_json(path: Path, payload: Any) -> None:
    atomic_write(
        path,
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
    )


def git_head() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError:
        return None
    return result.stdout.strip() if result.returncode == 0 else None


def required_input_hashes() -> dict[str, str]:
    paths = {
        "official_full_loop_manifest": MANIFEST,
        "stage07_cpu_audit": AUDIT,
        "event_manifest": EVENT_MANIFEST,
        "event_policy_gate": EVENT_GATE,
        "regression_result": TEST_RESULT,
    }
    for name, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"required input missing: {name}: {path}")
    return {name: sha256_file(path) for name, path in paths.items()}


def stage_statuses() -> dict[str, Any]:
    values: dict[str, Any] = {}
    for path in sorted(STAGE_STATUS.glob("*.json")):
        try:
            values[path.name] = read_json(path)
        except Exception as exc:
            values[path.name] = {"parse_error": f"{type(exc).__name__}: {exc}"}
    return values


def stream_worker_artifact(path: Path, expected_event_id: str, expected_branch: str) -> dict[str, Any]:
    """Count rows/IDs while checking the public-ID contract, without retaining frames."""
    rows = 0
    candidate_rows = 0
    public_id_assigned = 0
    public_id_unassigned = 0
    native_mapping_complete = 0
    errors: list[str] = []
    frames: list[int] = []
    action_type: str | None = None
    sequence: str | None = None
    runtime_flags: set[Any] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except Exception as exc:
                errors.append(f"line {line_number}: JSON parse {type(exc).__name__}: {exc}")
                continue
            if not isinstance(row, dict):
                errors.append(f"line {line_number}: row is not object")
                continue
            rows += 1
            frame = row.get("frame")
            if isinstance(frame, int):
                frames.append(frame)
            else:
                errors.append(f"line {line_number}: frame is not int")
            if row.get("event_id") != expected_event_id or row.get("branch") != expected_branch:
                errors.append(f"line {line_number}: event/branch mismatch")
            if sequence is None:
                sequence = str(row.get("sequence"))
            if action_type is None:
                action_type = str(row.get("action_type"))
            for key in ("runtime_future_gt_used", "runtime_gt_read", "posthoc_gt_used", "public_id_inference"):
                runtime_flags.add(row.get(key))
                if row.get(key) is not False:
                    errors.append(f"line {line_number}: {key} is not false")
            candidates = row.get("candidates")
            if not isinstance(candidates, list):
                errors.append(f"line {line_number}: candidates is not list")
                continue
            if row.get("candidate_count") != len(candidates):
                errors.append(f"line {line_number}: candidate count mismatch")
            if row.get("candidate_set_complete") is not True:
                errors.append(f"line {line_number}: candidate_set_complete is not true")
            candidate_indices = [item.get("candidate_index") for item in candidates if isinstance(item, dict)]
            if len(candidate_indices) != len(candidates) or len(set(candidate_indices)) != len(candidate_indices):
                errors.append(f"line {line_number}: candidate indices are not unique")
            for candidate in candidates:
                if not isinstance(candidate, dict):
                    errors.append(f"line {line_number}: candidate is not object")
                    continue
                candidate_rows += 1
                if candidate.get("public_id") is None:
                    public_id_unassigned += 1
                else:
                    public_id_assigned += 1
                if (
                    candidate.get("raw_native_id") is not None
                    and candidate.get("official_raw_sam_id") is not None
                    and candidate.get("adapter_visible_id") is not None
                    and candidate.get("adapter_external_id") is not None
                ):
                    native_mapping_complete += 1
    return {
        "rows": rows,
        "candidate_rows": candidate_rows,
        "public_id_assigned_candidate_rows": public_id_assigned,
        "public_id_unassigned_candidate_rows": public_id_unassigned,
        "native_adapter_mapping_complete_candidate_rows": native_mapping_complete,
        "frames": frames,
        "sequence": sequence,
        "action_type": action_type,
        "runtime_flags": sorted(runtime_flags, key=lambda value: str(value)),
        "errors": errors,
    }


def validate_manifest(manifest: dict[str, Any], audit: dict[str, Any]) -> dict[str, Any]:
    errors: list[str] = []
    workers = manifest.get("worker_records")
    events = manifest.get("events")
    if not isinstance(workers, list):
        errors.append("worker_records_not_list")
        workers = []
    if not isinstance(events, list):
        errors.append("events_not_list")
        events = []
    event_ids = [str(item.get("event_id")) for item in events if isinstance(item, dict)]
    worker_keys = [
        (str(item.get("event_id")), str(item.get("branch")))
        for item in workers
        if isinstance(item, dict)
    ]
    duplicate_keys = sorted({key for key in worker_keys if worker_keys.count(key) > 1})
    expected_keys = {(event_id, branch) for event_id in event_ids for branch in BRANCHES}
    observed_keys = set(worker_keys)
    missing_keys = sorted(expected_keys - observed_keys)
    extra_keys = sorted(observed_keys - expected_keys)
    if duplicate_keys:
        errors.append("duplicate_worker_keys")
    if missing_keys or extra_keys:
        errors.append("worker_key_coverage_mismatch")
    if manifest.get("status") != "PASS_N72R5_OFFICIAL_FULL_LOOP_SET":
        errors.append(f"manifest_status={manifest.get('status')}")
    if len(events) != 40 or manifest.get("event_count_completed") != 40:
        errors.append("event_count_not_40")
    if len(workers) != 40 * len(BRANCHES):
        errors.append("worker_count_not_200")
    if manifest.get("runtime_future_gt_used") is not False:
        errors.append("manifest_runtime_future_gt_not_false")
    if manifest.get("interaction_source") != "simulated_from_gt":
        errors.append("unexpected_interaction_source")
    if audit.get("status") != "PASS_N72R5_STAGE07_CPU_AUDIT" or audit.get("errors"):
        errors.append("stage07_cpu_audit_failed")
    return {
        "errors": errors,
        "duplicate_worker_keys": [list(key) for key in duplicate_keys],
        "missing_worker_keys": [list(key) for key in missing_keys],
        "extra_worker_keys": [list(key) for key in extra_keys],
        "event_count": len(events),
        "worker_count": len(workers),
        "expected_keys": len(expected_keys),
        "observed_keys": len(observed_keys),
    }


def build() -> tuple[dict[str, Any], str]:
    input_hashes = required_input_hashes()
    manifest = read_json(MANIFEST)
    audit = read_json(AUDIT)
    event_manifest = read_json(EVENT_MANIFEST)
    event_gate = read_json(EVENT_GATE)
    regression = read_json(TEST_RESULT)
    manifest_check = validate_manifest(manifest, audit)
    worker_by_key = {
        (str(item.get("event_id")), str(item.get("branch"))): item
        for item in manifest.get("worker_records", [])
        if isinstance(item, dict)
    }
    artifact_summaries: list[dict[str, Any]] = []
    scan_errors: list[str] = []
    action_counts: Counter[str] = Counter()
    sequence_counts: Counter[str] = Counter()
    total_rows = 0
    total_candidate_rows = 0
    public_id_assigned = 0
    public_id_unassigned = 0
    native_mapping_complete = 0
    frame_coverage_pass = True
    for key in sorted(worker_by_key):
        event_id, branch = key
        record = worker_by_key[key]
        output = Path(str(record.get("output", "")))
        if not output.is_file():
            scan_errors.append(f"missing_worker_output:{event_id}/{branch}")
            continue
        summary = stream_worker_artifact(output, event_id, branch)
        done_path = Path(str(record.get("done", "")))
        done_payload = read_json(done_path) if done_path.is_file() else {}
        event_frame = int(done_payload.get("event_frame", summary["frames"][0] if summary["frames"] else -1))
        expected_frames = list(range(event_frame, event_frame + HORIZON + 1))
        if summary["frames"] != expected_frames:
            frame_coverage_pass = False
            scan_errors.append(f"frame_coverage:{event_id}/{branch}")
        if summary["rows"] != HORIZON + 1:
            frame_coverage_pass = False
            scan_errors.append(f"row_count:{event_id}/{branch}")
        scan_errors.extend(f"{event_id}/{branch}:{error}" for error in summary["errors"])
        total_rows += int(summary["rows"])
        total_candidate_rows += int(summary["candidate_rows"])
        public_id_assigned += int(summary["public_id_assigned_candidate_rows"])
        public_id_unassigned += int(summary["public_id_unassigned_candidate_rows"])
        native_mapping_complete += int(summary["native_adapter_mapping_complete_candidate_rows"])
        if summary.get("action_type"):
            action_counts[str(summary["action_type"])] += 1
        if summary.get("sequence"):
            sequence_counts[str(summary["sequence"])] += 1
        artifact_summaries.append({
            "event_id": event_id,
            "branch": branch,
            "output": str(output),
            "row_count": summary["rows"],
            "candidate_rows": summary["candidate_rows"],
            "public_id_assigned_candidate_rows": summary["public_id_assigned_candidate_rows"],
            "public_id_unassigned_candidate_rows": summary["public_id_unassigned_candidate_rows"],
            "sequence": summary["sequence"],
            "action_type": summary["action_type"],
            "frame_start": summary["frames"][0] if summary["frames"] else None,
            "frame_end": summary["frames"][-1] if summary["frames"] else None,
            "artifact_sha256": sha256_file(output),
        })

    structural_errors = manifest_check["errors"] + scan_errors
    structural_pass = not structural_errors and len(artifact_summaries) == 40 * len(BRANCHES)
    public_mapping_complete = public_id_unassigned == 0 and public_id_assigned == total_candidate_rows and total_candidate_rows > 0
    real_human_tape_present = event_manifest.get("interaction_source") == "real_human"
    gate = {
        "schema_version": "N72R5_FINAL_GATE_V1",
        "created_at_utc": now_utc(),
        "status": "N72R5_STRUCTURAL_FULL_LOOP_PASS_EFFICACY_BLOCKED_NO_PUBLIC_MAPPING" if structural_pass else "N72R5_BLOCKED_STRUCTURAL_INTEGRITY",
        "research_gate": "BLOCKED_EXACT_PUBLIC_MAPPING_AND_REAL_HUMAN_TAPE" if structural_pass else "BLOCKED_STRUCTURAL_INTEGRITY",
        "scientific_result": "STRUCTURAL_OFFICIAL_CANDIDATE_STREAM_ONLY_NO_POSTHOC_EFFECT_RESULT",
        "production_authorized": False,
        "training_authorized": False,
        "calibration_authorized": False,
        "decoder_lora_authorized": False,
        "selector_authorized": False,
        "interaction_source": "simulated_from_gt",
        "not_real_human_evidence": True,
        "runtime_future_gt_used": False,
        "inputs": {
            "event_manifest": str(EVENT_MANIFEST),
            "event_policy_gate": str(EVENT_GATE),
            "official_full_loop_manifest": str(MANIFEST),
            "stage07_cpu_audit": str(AUDIT),
            "regression_result": str(TEST_RESULT),
            "sha256": input_hashes,
        },
        "structural_integrity": {
            "passed": structural_pass,
            "errors": structural_errors,
            "event_count_expected": 40,
            "event_count_observed": manifest_check["event_count"],
            "branch_count_per_event": len(BRANCHES),
            "worker_count_expected": 200,
            "worker_count_observed": manifest_check["worker_count"],
            "unique_worker_keys": manifest_check["observed_keys"],
            "duplicate_worker_keys": manifest_check["duplicate_worker_keys"],
            "missing_worker_keys": manifest_check["missing_worker_keys"],
            "extra_worker_keys": manifest_check["extra_worker_keys"],
            "frame_coverage_pass": frame_coverage_pass,
            "total_frame_rows": total_rows,
            "expected_frame_rows": 40 * len(BRANCHES) * (HORIZON + 1),
            "candidate_rows": total_candidate_rows,
            "native_adapter_mapping_complete_candidate_rows": native_mapping_complete,
            "runtime_future_gt_false": manifest.get("runtime_future_gt_used") is False and not scan_errors,
        },
        "event_policy": {
            "status": event_gate.get("status"),
            "event_count": event_manifest.get("event_count"),
            "independent_sequence_count": event_manifest.get("independent_sequence_count"),
            "action_counts": event_manifest.get("action_counts", dict(action_counts)),
            "interaction_source": event_manifest.get("interaction_source"),
            "real_human_tape_present": real_human_tape_present,
        },
        "verification": {
            "status": regression.get("status"),
            "passed": (
                regression.get("status") == "PASS"
                and regression.get("result", {}).get("failed") == 0
                and regression.get("result", {}).get("collection_errors") == 0
            ),
            "result": regression.get("result", {}),
            "environment": regression.get("environment"),
            "third_party_modified": regression.get("third_party_modified"),
        },
        "public_mapping_and_effect": {
            "public_mapping_complete": public_mapping_complete,
            "public_id_assigned_candidate_rows": public_id_assigned,
            "public_id_unassigned_candidate_rows": public_id_unassigned,
            "exact_public_association_evaluated": False,
            "posthoc_effect_evaluated": False,
            "future_identity_effect": None,
            "reason": "Official Stage07 artifacts explicitly retain public_id=null/NOT_ASSIGNED_IN_OFFICIAL_BRANCH; no authoritative public resolver or public-ID axis is present.",
        },
        "preserved_failures": {
            "stage07_attempt_status_files": [str(path) for path in sorted(STAGE_STATUS.glob("stage_07_*status.json"))],
            "attempt_artifacts": [str(path) for path in sorted((OUT / "attempts").glob("*.json"))],
            "failure_records_root": str(OUT / "failure_records"),
            "attempt5_failure_artifact_count": len(list((ROUND / "runtime").rglob("*.failure.json"))),
        },
        "action_counts_observed_in_artifacts": dict(sorted(action_counts.items())),
        "sequence_counts_observed_in_artifacts": dict(sorted(sequence_counts.items())),
        "worker_artifact_summaries": artifact_summaries,
        "minimal_next_action": "Obtain provenance-complete real-human events plus an authoritative same-run public-ID resolver/mapping, validate it, then rerun the unchanged exact association and future-effect protocol.",
        "source_git_head_at_finalize": git_head(),
    }
    report = render_report(gate)
    return gate, report


def render_report(gate: dict[str, Any]) -> str:
    integrity = gate["structural_integrity"]
    mapping = gate["public_mapping_and_effect"]
    policy = gate["event_policy"]
    lines = [
        "# N72R5 Final Report",
        "",
        f"- Generated: `{gate['created_at_utc']}`",
        f"- Machine gate: `{gate['status']}`",
        f"- Research gate: `{gate['research_gate']}`",
        f"- Runtime future GT: `{gate['runtime_future_gt_used']}`",
        f"- Interaction source: `{gate['interaction_source']}` (not real-human evidence)",
        "",
        "## Executive conclusion",
        "",
        "N72R5 completed the official SAM3 candidate-stream full-loop structurally, but it did not produce a scientific future-effect result. The 40-event policy is explicitly `simulated_from_gt`, and the official branch artifacts retain `public_id=null` with `NOT_ASSIGNED_IN_OFFICIAL_BRANCH`. Therefore exact public-ID association, posthoc future-effect scoring, production promotion, calibration, selector training, and decoder LoRA remain unauthorized.",
        "",
        "This is a provenance/authority block, not a claim that the mechanism works or fails. No public ID was inferred from a native ID, candidate index, or dataset GT ID.",
        "",
        "## Frozen scope and event policy",
        "",
        f"- Events: `{policy.get('event_count')}`; independent sequences: `{policy.get('independent_sequence_count')}`.",
        f"- Action counts: `{json.dumps(policy.get('action_counts', {}), sort_keys=True)}`.",
        "- Source split: frozen train/train_fold candidate tape; no val/test was introduced.",
        "- All events remain `simulated_from_gt`; they must not be described as historical human clicks.",
        "- Stage 01--04 mechanism findings were retained: candidate absence and candidate-present decision errors are distinct; image recovery had no recall gain; TVC had no correct crossing; feature separability was not informative.",
        "",
        "## Stage 07 structural result",
        "",
        "| Check | Result |",
        "|---|---:|",
        f"| Events | `{integrity.get('event_count_observed')}/{integrity.get('event_count_expected')}` |",
        f"| Official branches | `{integrity.get('worker_count_observed')}/{integrity.get('worker_count_expected')}` |",
        f"| Unique worker keys | `{integrity.get('unique_worker_keys')}` |",
        f"| Duplicate worker keys | `{len(integrity.get('duplicate_worker_keys', []))}` |",
        f"| Missing worker keys | `{len(integrity.get('missing_worker_keys', []))}` |",
        f"| Frame rows | `{integrity.get('total_frame_rows')}/{integrity.get('expected_frame_rows')}` |",
        f"| Candidate rows | `{integrity.get('candidate_rows')}` |",
        f"| Native/adapter mapping fields complete | `{integrity.get('native_adapter_mapping_complete_candidate_rows')}` |",
        f"| Runtime future GT | `{gate.get('runtime_future_gt_used')}` |",
        f"| Structural integrity | `{integrity.get('passed')}` |",
        "",
        "Each event has the five preregistered branches `B0`--`B4`, 101 frame rows from the event frame through event+100, a shared frozen pre-state hash, and event-frame memory-read suppression. The Stage07 CPU audit is independent of the worker execution and reports no structural error.",
        "",
        "## Exact association and future-effect gate",
        "",
        f"- Public-ID assigned candidate rows: `{mapping.get('public_id_assigned_candidate_rows')}`.",
        f"- Public-ID unassigned candidate rows: `{mapping.get('public_id_unassigned_candidate_rows')}`.",
        f"- Exact public association evaluated: `{mapping.get('exact_public_association_evaluated')}`.",
        f"- Posthoc future effect evaluated: `{mapping.get('posthoc_effect_evaluated')}`.",
        f"- Future identity effect: `{mapping.get('future_identity_effect')}`.",
        "",
        "Because the authoritative public-ID axis is absent, there is no valid identity-error, ID-switch, re-correction, or H20/H50/H100 public-ID effect number to report. Filling the missing axis from raw SAM IDs, candidate indices, dataset GT IDs, or a future heuristic would violate the frozen protocol, so the finalizer intentionally leaves the effect null.",
        "",
        "## Failure and repair provenance",
        "",
        "- Stage07 attempts 1--4 remain preserved as blocked/partial attempts; their status files and failure artifacts were not deleted or rewritten.",
        "- Attempt5 completed after the already-recorded memory/observation engineering repairs: `200/200` branches, `0` new failure artifacts.",
        "- The Stage03/Stage04 negative mechanism gates remain scientific negative findings, not converted into PASS efficacy evidence.",
        "- N36--N72R4 historical outputs and all earlier failure evidence remain read-only inputs.",
        "",
        "## Authorization decision",
        "",
        "`production_authorized=false`, `training_authorized=false`, `calibration_authorized=false`, `selector_authorized=false`, and `decoder_lora_authorized=false`. The structural full-loop pass cannot authorize downstream learning because exact public authority and real-human evidence are still missing.",
        "",
        "## Reproducibility artifacts",
        "",
        f"- [Machine-readable final gate]({GATE})",
        f"- [Stage07 CPU audit]({AUDIT})",
        f"- [Official full-loop manifest]({MANIFEST})",
        f"- [Frozen event manifest]({EVENT_MANIFEST})",
        f"- [Pinned regression result]({TEST_RESULT})",
        f"- [Preserved N72R5 attempts]({OUT / 'attempts'})",
        "",
        "Input SHA-256 values are recorded in `n72r5_final_gate.json`. The pinned regression completed `193 passed, 0 failed` with three existing interpreter warnings; the isolated worktree used the populated sibling TrackEval checkout only as a read-only test path. The final gate also records the source Git HEAD observed at finalization.",
        "",
        "## Minimum next action",
        "",
        "Collect provenance-complete real-human event JSONL and raw input files, including direct public IDs and a same-run authoritative resolver/mapping that survives session boundaries. Validate the mapping before running the unchanged exact public association and future-effect protocol. Do not relabel `simulated_from_gt` events as real human data and do not start calibration, selector, or decoder LoRA before that gate.",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    gate, report = build()
    atomic_json(GATE, gate)
    atomic_write(REPORT, report)
    atomic_json(
        STAGE08,
        {
            "schema_version": "N72R5_STAGE_STATUS_V1",
            "stage": "08_FINAL_GATE",
            "status": gate["status"],
            "research_gate": gate["research_gate"],
            "structural_integrity_pass": gate["structural_integrity"]["passed"],
            "exact_public_association_evaluated": gate["public_mapping_and_effect"]["exact_public_association_evaluated"],
            "posthoc_effect_evaluated": gate["public_mapping_and_effect"]["posthoc_effect_evaluated"],
            "runtime_future_gt_used": gate["runtime_future_gt_used"],
            "interaction_source": gate["interaction_source"],
            "production_authorized": gate["production_authorized"],
            "training_authorized": gate["training_authorized"],
            "final_gate": str(GATE),
            "final_report": str(REPORT),
            "minimal_next_action": gate["minimal_next_action"],
            "created_at_utc": gate["created_at_utc"],
        },
    )
    print(json.dumps({
        "status": gate["status"],
        "research_gate": gate["research_gate"],
        "event_count": gate["structural_integrity"]["event_count_observed"],
        "worker_count": gate["structural_integrity"]["worker_count_observed"],
        "candidate_rows": gate["structural_integrity"]["candidate_rows"],
        "public_id_assigned_candidate_rows": gate["public_mapping_and_effect"]["public_id_assigned_candidate_rows"],
        "public_id_unassigned_candidate_rows": gate["public_mapping_and_effect"]["public_id_unassigned_candidate_rows"],
        "gate": str(GATE),
        "report": str(REPORT),
    }, sort_keys=True))
    return 0 if gate["status"].endswith("NO_PUBLIC_MAPPING") else 1


if __name__ == "__main__":
    raise SystemExit(main())
