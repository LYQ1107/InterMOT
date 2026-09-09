#!/usr/bin/env python3
"""Create a lossless, post-run audit of N72R11R4 E2 child failures.

The first E2 manifest is intentionally treated as immutable input.  In
particular, a Python ``site`` warning containing the word ``Traceback`` is
not classified as a child traceback when the child was SIGKILLed.  CUDA OOM
failure artifacts remain the authoritative source for OOM classification.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import tempfile
from typing import Any, Mapping

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
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    temporary = Path(temporary_name)
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
        if temporary.exists():
            temporary.unlink()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object: {path}")
    return value


def resolve(value: Any) -> Path:
    path = Path(str(value))
    return path if path.is_absolute() else ROOT / path


def failure_artifact(manifest_path: Path, event_id: str) -> Path | None:
    candidate = manifest_path.parent / "attempts" / f"{event_id}.failure.json"
    return candidate if candidate.is_file() else None


def terminal_log_text(path: Path) -> str:
    if not path.is_file():
        return ""
    # The useful terminal exception is normally at the end; cap memory while
    # retaining enough context to distinguish the pth warning from an OOM.
    with path.open("rb") as handle:
        handle.seek(max(0, path.stat().st_size - 2_000_000))
        return handle.read().decode("utf-8", errors="replace")


def classify(manifest_path: Path, record: Mapping[str, Any]) -> dict[str, Any]:
    event_id = str(record.get("event_id"))
    log_path = resolve(record.get("log")) if record.get("log") else None
    log_text = terminal_log_text(log_path) if log_path else ""
    artifact_path = failure_artifact(manifest_path, event_id)
    artifact = read_json(artifact_path) if artifact_path else {}
    error = str(artifact.get("error", ""))
    traceback = str(artifact.get("traceback", ""))
    combined = "\n".join((error, traceback, log_text))
    oom = bool(re.search(r"(?:CUDA )?out of memory|OutOfMemoryError|torch\.OutOfMemoryError", combined, re.I))
    real_exception_traceback = bool(traceback.strip()) or bool(re.search(r"torch\.(?:OutOfMemoryError|RuntimeError)|Traceback.*\n.*(?:run_event|propagate_if_selected)", log_text, re.S))
    returncode = record.get("returncode")
    if oom:
        classification = "CUDA_OOM"
        root_cause = "official SAM3 live replay exhausted device memory; retain the exact OOM allocation and traceback"
    elif returncode == -9:
        classification = "SIGKILL_NO_TRACEBACK_RESOURCE_OR_EXTERNAL_KILL"
        root_cause = "child exited with SIGKILL (-9) without a process exception traceback; resource pressure or external kill is unresolved"
    else:
        classification = "CHILD_FAILURE_UNCLASSIFIED"
        root_cause = "child returned nonzero without a sealed done artifact; inspect preserved log and failure artifact"
    warning_only = bool("osr_lib-1.1.0-nspkg.pth" in log_text and not real_exception_traceback and not oom)
    return {
        "event_id": event_id,
        "sequence": record.get("sequence"),
        "action_type": record.get("action_type"),
        "attempt": record.get("attempt"),
        "status": record.get("status"),
        "returncode": returncode,
        "signal": 9 if returncode == -9 else None,
        "classification": classification,
        "actionable_root_cause": root_cause,
        "oom_present": oom,
        "process_traceback_present": real_exception_traceback,
        "environment_warning_only": warning_only,
        "log": str(log_path) if log_path else None,
        "log_exists": bool(log_path and log_path.is_file()),
        "log_bytes": log_path.stat().st_size if log_path and log_path.is_file() else None,
        "log_sha256": sha256_file(log_path) if log_path and log_path.is_file() else None,
        "failure_artifact": str(artifact_path) if artifact_path else None,
        "failure_artifact_sha256": sha256_file(artifact_path) if artifact_path else None,
        "failure_artifact_device": artifact.get("device"),
        "failure_artifact_error": error or None,
        "historical_outputs_modified": artifact.get("historical_outputs_modified"),
        "runtime_future_gt_used": artifact.get("runtime_future_gt_used", False),
    }


def kernel_oom_evidence() -> dict[str, Any]:
    command = ["dmesg", "--ctime"]
    try:
        completed = subprocess.run(command, capture_output=True, text=True, check=False)
    except OSError as exc:
        return {"command": command, "available": False, "error": f"{type(exc).__name__}: {exc}", "relevant_lines": []}
    lines = [
        line.strip()
        for line in completed.stdout.splitlines()
        if re.search(r"oom-kill|Out of memory|Killed process", line, re.I)
    ]
    return {
        "command": command,
        "returncode": completed.returncode,
        "available": completed.returncode == 0,
        "relevant_lines": lines[-20:],
        "interpretation": "host-level OOM evidence is not attributed to a replay child unless PID/time match is established",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifests: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for raw_path in args.manifest:
        path = raw_path if raw_path.is_absolute() else ROOT / raw_path
        payload = read_json(path)
        manifest_record = {
            "path": str(path),
            "sha256": sha256_file(path),
            "status": payload.get("status"),
            "counts": payload.get("counts"),
            "attempt": payload.get("attempt"),
        }
        manifests.append(manifest_record)
        for record in payload.get("records", []):
            if isinstance(record, Mapping) and record.get("status") != "PASS":
                failures.append(classify(path, record))
    result = {
        "schema_version": "N72R11R4_E2_FAILURE_AUDIT_V2",
        "created_at_utc": now_utc(),
        "source_manifests": manifests,
        "failure_count": len(failures),
        "class_counts": {
            key: sum(item["classification"] == key for item in failures)
            for key in sorted({item["classification"] for item in failures})
        },
        "failures": failures,
        "kernel_oom_evidence": kernel_oom_evidence(),
        "preservation": "read-only audit; source manifests, logs, and failure artifacts are not overwritten",
        "runtime_future_gt_used": False,
        "not_real_human_evidence": True,
    }
    output = args.output if args.output.is_absolute() else ROOT / args.output
    atomic_json(output, result)
    print(json.dumps({"output": str(output), "failure_count": len(failures), "class_counts": result["class_counts"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
