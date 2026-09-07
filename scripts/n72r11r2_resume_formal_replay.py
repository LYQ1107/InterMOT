#!/usr/bin/env python3
"""Audit a stale formal-replay manifest and resume only unfinished events.

This keeps the interrupted attempt immutable.  A new batch output root is used
for every resumed event, so partial event directories from the interrupted
supervisor cannot be mistaken for a completed artifact or overwritten.
"""

from __future__ import annotations

from datetime import datetime, timezone
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
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


def read_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object: {path}")
    return value


def classify_records(source_manifest: Path) -> tuple[list[dict[str, Any]], dict[str, int]]:
    payload = read_object(source_manifest)
    records = payload.get("records")
    if not isinstance(records, list) or len(records) != int(payload.get("event_count", -1)):
        raise RuntimeError("source manifest has no complete records list")
    resume: list[dict[str, Any]] = []
    counts: dict[str, int] = {}
    for record in records:
        if not isinstance(record, dict) or not record.get("event_id"):
            raise RuntimeError("malformed source record")
        status = str(record.get("status"))
        event_id = str(record["event_id"])
        done = Path(str(record["done"])) if record.get("done") else None
        pass_is_sealed = status == "PASS" and done is not None and done.is_file()
        if pass_is_sealed and record.get("done_sha256"):
            pass_is_sealed = sha256_file(done) == str(record["done_sha256"])
        bucket = "SEALED_PASS" if pass_is_sealed else ("INTERRUPTED_RUNNING" if status == "RUNNING" else status)
        counts[bucket] = counts.get(bucket, 0) + 1
        if not pass_is_sealed:
            resume.append({
                "event_id": event_id,
                "sequence": record.get("sequence"),
                "action_type": record.get("action_type"),
                "source_status": status,
                "source_log": record.get("log"),
                "reason": "source_attempt_not_sealed_PASS",
            })
    return sorted(resume, key=lambda value: value["event_id"]), counts


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--resume-output-root", type=Path, required=True)
    parser.add_argument("--model-checkpoint", type=Path, required=True)
    parser.add_argument("--bridge-checkpoint", type=Path, required=True)
    parser.add_argument("--max-workers", type=int, default=4)
    parser.add_argument("--gpu-ids", default="1,2,3,4")
    parser.add_argument("--attempt", type=int, default=2)
    parser.add_argument("--audit-path", type=Path, required=True)
    args = parser.parse_args()
    source = args.source_manifest.resolve()
    resume_root = args.resume_output_root.resolve()
    audit_path = args.audit_path.resolve()
    if not source.is_file():
        raise SystemExit(f"missing source manifest: {source}")
    if int(args.attempt) <= 1:
        raise SystemExit("resume attempt must be greater than one")
    resume, counts = classify_records(source)
    if not resume:
        raise SystemExit("source manifest has no unfinished events")
    audit = {
        "schema_version": "N72R11R2_FORMAL_REPLAY_INTERRUPTION_AUDIT_V1",
        "status": "PASS_RESUME_SET_FROZEN",
        "created_at_utc": now_utc(),
        "source_manifest": str(source),
        "source_manifest_sha256": sha256_file(source),
        "source_status": read_object(source).get("status"),
        "root_cause": "supervisor_and_workers_absent_before_manifest_harvest; source RUNNING records are not PASS evidence",
        "original_failure_preserved": True,
        "sealed_pass_count": counts.get("SEALED_PASS", 0),
        "unfinished_count": len(resume),
        "unfinished_source_status_counts": counts,
        "resume_event_ids": [value["event_id"] for value in resume],
        "resume_output_root": str(resume_root),
        "no_existing_pass_event_rerun": True,
    }
    atomic_json(audit_path, audit)
    command = [
        sys.executable,
        "-u",
        str(ROOT / "scripts/n72r11_run_formal_replay_batch.py"),
        "--attempt", str(int(args.attempt)),
        "--output-root", str(resume_root),
        "--model-checkpoint", str(args.model_checkpoint.resolve()),
        "--bridge-checkpoint", str(args.bridge_checkpoint.resolve()),
        "--max-workers", str(int(args.max_workers)),
        "--gpu-ids", str(args.gpu_ids),
        "--resource-censored",
    ]
    for item in resume:
        command.extend(["--event-id", item["event_id"]])
    controller_path = resume_root / f"resume_controller_attempt_{int(args.attempt):02d}.json"
    controller = {
        "schema_version": "N72R11R2_FORMAL_REPLAY_RESUME_CONTROLLER_V1",
        "status": "RUNNING",
        "created_at_utc": now_utc(),
        "source_audit": str(audit_path),
        "selected_event_count": len(resume),
        "selected_event_ids": [value["event_id"] for value in resume],
        "command": command,
        "resource_censored_development": True,
    }
    atomic_json(controller_path, controller)
    try:
        completed = subprocess.run(command, cwd=str(ROOT), check=False)
    except BaseException as exc:
        controller.update({"status": "INTERRUPTED_WITH_EVIDENCE", "finished_at_utc": now_utc(), "exception": repr(exc)})
        atomic_json(controller_path, controller)
        raise
    controller.update({"status": "COMPLETED", "finished_at_utc": now_utc(), "returncode": int(completed.returncode)})
    atomic_json(controller_path, controller)
    return int(completed.returncode)


if __name__ == "__main__":
    raise SystemExit(main())
