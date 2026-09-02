#!/usr/bin/env python3
"""CPU-only validator for server-finalized N72R1 human events.

This command validates provenance and path references only.  It never imports
GT, chooses an event, infers a public ID, or runs SAM3/replay.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any

from sam3_intermot.interaction.real_human_v2 import validate_real_human_event_v2
from sam3_intermot.provenance.path_safety import PathSafetyError, resolve_within_root


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


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate N72R1 server-finalized real-human JSONL")
    parser.add_argument("--event-root", type=Path, required=True)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--candidate-root", type=Path, default=None)
    parser.add_argument("--report", type=Path, default=None)
    args = parser.parse_args()
    event_root = args.event_root.resolve()
    raw_root = args.raw_root.resolve()
    event_path = event_root / "real_human_events.jsonl"
    errors: list[dict[str, Any]] = []
    event_ids: set[str] = set()
    rows = 0
    if event_path.exists():
        with event_path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                rows += 1
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    errors.append({"line": line_number, "code": "INVALID_JSON", "error": str(exc)})
                    continue
                audit = validate_real_human_event_v2(record, require_server_auth=True)
                if not audit["valid"]:
                    errors.append({"line": line_number, "code": "EVENT_SCHEMA_INVALID", "event_id": record.get("event_id"), "details": audit["errors"]})
                event_id = record.get("event_id")
                if event_id in event_ids:
                    errors.append({"line": line_number, "code": "DUPLICATE_EVENT_ID", "event_id": event_id})
                if isinstance(event_id, str):
                    event_ids.add(event_id)
                human_input = record.get("human_input") if isinstance(record, dict) else None
                raw_ref = human_input.get("raw_payload_ref") if isinstance(human_input, dict) else None
                raw_digest = human_input.get("raw_payload_sha256") if isinstance(human_input, dict) else None
                if not isinstance(raw_ref, str):
                    errors.append({"line": line_number, "code": "RAW_REQUEST_REF_MISSING"})
                else:
                    try:
                        raw_path = resolve_within_root(raw_ref, raw_root)
                        if not raw_path.is_file():
                            errors.append({"line": line_number, "code": "RAW_REQUEST_MISSING", "path": str(raw_path)})
                        elif isinstance(raw_digest, str) and sha256(raw_path) != raw_digest:
                            errors.append({"line": line_number, "code": "RAW_REQUEST_HASH_MISMATCH", "path": str(raw_path)})
                    except (PathSafetyError, ValueError) as exc:
                        errors.append({"line": line_number, "code": "RAW_REQUEST_PATH_UNSAFE", "error": str(exc)})
                candidate_ref = record.get("candidate_tape_ref") if isinstance(record, dict) else None
                if not isinstance(candidate_ref, str) or not candidate_ref.strip():
                    errors.append({"line": line_number, "code": "CANDIDATE_TAPE_REF_MISSING"})
                elif args.candidate_root is not None:
                    try:
                        candidate_path = resolve_within_root(candidate_ref, args.candidate_root.resolve())
                        if not candidate_path.exists():
                            errors.append({"line": line_number, "code": "CANDIDATE_TAPE_REF_MISSING", "path": str(candidate_path)})
                    except (PathSafetyError, ValueError) as exc:
                        errors.append({"line": line_number, "code": "CANDIDATE_TAPE_PATH_UNSAFE", "error": str(exc)})
    result = {
        "schema_version": "N72R1_REAL_HUMAN_TAPE_AUDIT_V1",
        "status": "PASS_EMPTY_REAL_TAPE" if rows == 0 and not errors else "PASS_REAL_HUMAN_TAPE" if not errors else "FAIL_REAL_HUMAN_TAPE_VALIDATION",
        "event_file": str(event_path),
        "row_count": rows,
        "unique_event_id_count": len(event_ids),
        "error_count": len(errors),
        "errors": errors,
        "runtime_future_gt_used": False,
        "gt_read": False,
        "synthetic_from_gt_accepted": False,
    }
    if args.report is not None:
        atomic_json(args.report.resolve(), result)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
