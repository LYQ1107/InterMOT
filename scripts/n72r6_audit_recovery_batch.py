#!/usr/bin/env python3
"""CPU-only audit and manifest builder for N72R6 recovery streams.

The recovery supervisor's process-exit status is not a stream validity result.
This script audits each completed ``done.json`` with the single-event recovery
validator, reconciles it against the frozen 32-event key set, and writes a new
manifest.  It never reads dataset GT and never modifies the frozen target
stream manifest or any historical artifact.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import tempfile
from typing import Any

from n72r6_audit_target_recovery import audit as audit_one


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object: {path}")
    return value


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    try:
        with open(fd, "w", encoding="utf-8", closefd=True) as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            import os

            os.fsync(handle.fileno())
        Path(temporary).replace(path)
    finally:
        temporary_path = Path(temporary)
        if temporary_path.exists():
            temporary_path.unlink()


def resolve(path_value: str, root: Path) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else root / path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frozen-manifest", type=Path, required=True)
    parser.add_argument("--recovery-root", type=Path, required=True)
    parser.add_argument("--manifest-output", type=Path, required=True)
    parser.add_argument("--status-output", type=Path, required=True)
    parser.add_argument(
        "--replacement-done",
        action="append",
        default=[],
        help="deterministic replacement in EVENT_ID=PATH form; original failures remain recorded",
    )
    args = parser.parse_args()

    cwd = Path.cwd()
    frozen_path = resolve(str(args.frozen_manifest), cwd)
    recovery_root = resolve(str(args.recovery_root), cwd)
    manifest_path = resolve(str(args.manifest_output), cwd)
    status_path = resolve(str(args.status_output), cwd)
    frozen = read_json(frozen_path)
    selected = frozen.get("selected", [])
    if not isinstance(selected, list):
        raise TypeError("frozen manifest selected must be a list")

    expected_entries: dict[str, dict[str, Any]] = {}
    duplicate_expected: list[str] = []
    for entry in selected:
        if not isinstance(entry, dict) or not str(entry.get("event_id", "")):
            raise ValueError("frozen manifest contains an invalid selected entry")
        event_id = str(entry["event_id"])
        if event_id in expected_entries:
            duplicate_expected.append(event_id)
        expected_entries[event_id] = entry

    replacement_paths: dict[str, Path] = {}
    for item in args.replacement_done:
        event_id, separator, path_value = str(item).partition("=")
        if not separator or not event_id or not path_value:
            raise ValueError("--replacement-done must use EVENT_ID=PATH")
        path = resolve(path_value, cwd)
        if not path.is_file():
            raise FileNotFoundError(path)
        if event_id in replacement_paths:
            raise ValueError(f"duplicate replacement event: {event_id}")
        replacement_paths[event_id] = path

    done_paths = sorted(recovery_root.rglob("done.json")) if recovery_root.is_dir() else []
    observed: dict[str, list[Path]] = {}
    malformed_done: list[dict[str, str]] = []
    for done_path in done_paths:
        try:
            done = read_json(done_path)
            event_id = str(done.get("event_id", ""))
            if not event_id:
                raise ValueError("event_id_missing")
        except Exception as exc:  # noqa: BLE001 - preserve every malformed artifact
            malformed_done.append({"path": str(done_path), "error": str(exc)})
            continue
        observed.setdefault(event_id, []).append(done_path)

    for event_id, done_path in replacement_paths.items():
        done = read_json(done_path)
        if str(done.get("event_id", "")) != event_id:
            raise ValueError(f"replacement event_id mismatch: {event_id} != {done.get('event_id')}")
        observed[event_id] = [done_path]

    duplicate_observed = sorted(event_id for event_id, paths in observed.items() if len(paths) != 1)
    expected_ids = set(expected_entries)
    observed_ids = set(observed)
    unexpected_ids = sorted(observed_ids - expected_ids)
    missing_ids = sorted(expected_ids - observed_ids)

    audited: list[dict[str, Any]] = []
    audit_failures: list[dict[str, str]] = []
    candidate_rows = 0
    frame_rows = 0
    recovery_attempts = 0
    for event_id in sorted(expected_ids & observed_ids):
        paths = observed[event_id]
        if len(paths) != 1:
            audit_failures.append({"event_id": event_id, "error": "duplicate_done_artifacts"})
            continue
        done_path = paths[0]
        try:
            result = audit_one(done_path)
            done = read_json(done_path)
            if result.get("event_id") != event_id:
                raise ValueError("audit_event_id_mismatch")
            expected = expected_entries[event_id]
            if str(expected.get("sequence")) != str(result.get("sequence")):
                raise ValueError("sequence_mismatch_with_frozen_manifest")
            if int(expected.get("event_frame", -1)) != int(result.get("event_frame", -2)):
                raise ValueError("event_frame_mismatch_with_frozen_manifest")
            candidate_rows += int(result.get("candidate_row_count", 0))
            frame_rows += int(result.get("frame_count", 0))
            recovery_attempts += int(result.get("recovery_attempt_count", 0))
            audited.append(
                {
                    "event_id": event_id,
                    "sequence": result["sequence"],
                    "action_type": expected.get("action_type"),
                    "event_frame": result["event_frame"],
                    "end_frame": result["end_frame"],
                    "target_public_id": expected.get("target_public_id"),
                    "done": str(done_path),
                    "status": result["status"],
                    "frame_count": result["frame_count"],
                    "candidate_row_count": result["candidate_row_count"],
                    "recovery_attempt_count": result["recovery_attempt_count"],
                    "runtime_future_gt_used": False,
                    "interaction_source": "simulated_from_gt",
                    "not_real_human_evidence": True,
                    "target_session_recovery_mode": bool(done.get("target_session_recovery_mode")),
                }
            )
        except Exception as exc:  # noqa: BLE001 - preserve per-event audit fact
            audit_failures.append({"event_id": event_id, "done": str(done_path), "error": str(exc)})

    failure_candidates: set[Path] = set()
    for failure_root in (recovery_root / "attempts", recovery_root.parent / "attempts"):
        if failure_root.is_dir():
            failure_candidates.update(failure_root.glob("*.failure.json"))
    failed_artifacts = sorted(str(path) for path in failure_candidates)
    failed_by_event: dict[str, list[str]] = {}
    for failure_path in failed_artifacts:
        try:
            failure_event = str(read_json(Path(failure_path)).get("event_id", ""))
        except Exception:  # noqa: BLE001 - retain unparseable failure as unresolved
            failure_event = ""
        failed_by_event.setdefault(failure_event, []).append(failure_path)
    resolved_failure_artifacts = sorted(
        path
        for event_id, paths in failed_by_event.items()
        if event_id in replacement_paths
        for path in paths
    )
    unresolved_failure_artifacts = sorted(set(failed_artifacts) - set(resolved_failure_artifacts))
    pass_statuses = {
        "PASS_TARGET_SESSION_RECOVERY_STREAM_AUDIT",
        "PASS_TARGET_SESSION_RECOVERY_STREAM_AUDIT_WITH_LEGITIMATE_LOSS",
    }
    pass_event_ids = sorted(item["event_id"] for item in audited if item["status"] in pass_statuses)
    valid_complete = (
        len(expected_entries) == 32
        and not duplicate_expected
        and not duplicate_observed
        and not unexpected_ids
        and not missing_ids
        and not malformed_done
        and not audit_failures
        and len(pass_event_ids) == 32
        and not unresolved_failure_artifacts
    )
    status_value = (
        "PASS_N72R6_TARGET_SESSION_RECOVERY_32_OF_32_VALIDATED"
        if valid_complete
        else "PARTIAL_TARGET_SESSION_RECOVERY_BATCH_AUDIT"
    )
    manifest = {
        "schema_version": "N72R6_TARGET_SESSION_RECOVERY_MANIFEST_V1",
        "status": status_value,
        "created_at_utc": now_utc(),
        "frozen_manifest": str(frozen_path),
        "frozen_manifest_sha256": hashlib.sha256(frozen_path.read_bytes()).hexdigest(),
        "recovery_root": str(recovery_root),
        "target_session_recovery_mode": True,
        "interaction_source": "simulated_from_gt",
        "not_real_human_evidence": True,
        "runtime_future_gt_used": False,
        "expected_event_count": len(expected_entries),
        "validated_event_count": len(pass_event_ids),
        "process_completed_count": len(done_paths),
        "process_failed_count": len(failed_artifacts),
        "replacement_count": len(replacement_paths),
        "resolved_failure_count": len(resolved_failure_artifacts),
        "unresolved_failure_count": len(unresolved_failure_artifacts),
        "duplicate_expected_event_ids": duplicate_expected,
        "duplicate_observed_event_ids": duplicate_observed,
        "unexpected_event_ids": unexpected_ids,
        "missing_event_ids": missing_ids,
        "malformed_done": malformed_done,
        "audit_failures": audit_failures,
        "failed_artifacts": failed_artifacts,
        "resolved_failure_artifacts": resolved_failure_artifacts,
        "unresolved_failure_artifacts": unresolved_failure_artifacts,
        "replacement_done": {key: str(value) for key, value in sorted(replacement_paths.items())},
        "candidate_row_count": candidate_rows,
        "frame_row_count": frame_rows,
        "recovery_attempt_count": recovery_attempts,
        "replay_ready": valid_complete,
        "selected": audited,
    }
    status = {
        "schema_version": "N72R6_TARGET_SESSION_RECOVERY_BATCH_STATUS_V1",
        "status": status_value,
        "created_at_utc": now_utc(),
        "manifest": str(manifest_path),
        "expected_event_count": len(expected_entries),
        "validated_event_count": len(pass_event_ids),
        "process_completed_count": len(done_paths),
        "process_failed_count": len(failed_artifacts),
        "replacement_count": len(replacement_paths),
        "resolved_failure_count": len(resolved_failure_artifacts),
        "unresolved_failure_count": len(unresolved_failure_artifacts),
        "duplicate_event_count": len(duplicate_observed),
        "missing_event_count": len(missing_ids),
        "unexpected_event_count": len(unexpected_ids),
        "malformed_done_count": len(malformed_done),
        "audit_failure_count": len(audit_failures),
        "candidate_row_count": candidate_rows,
        "frame_row_count": frame_rows,
        "recovery_attempt_count": recovery_attempts,
        "runtime_future_gt_used": False,
        "replay_ready": valid_complete,
        "failure_artifacts_preserved": True,
        "replacement_done": {key: str(value) for key, value in sorted(replacement_paths.items())},
    }
    atomic_json(manifest_path, manifest)
    atomic_json(status_path, status)
    print(json.dumps(status, ensure_ascii=False, sort_keys=True))
    return 0 if valid_complete else 1


if __name__ == "__main__":
    raise SystemExit(main())
