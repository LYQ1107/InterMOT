#!/usr/bin/env python3
"""Run the frozen N72R9 event set with one fresh Python process per event."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = ROOT / "outputs/N72R9/protocol.json"
WORKER = ROOT / "scripts/n72r10_true_future_requery_event.py"


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(dict(payload), handle, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
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


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object: {path}")
    return value


def load_events() -> list[str]:
    payload = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    events = sorted(str(item["event_id"]) for item in payload.get("source_event_selection", {}).get("events", []))
    if len(events) != 32 or len(events) != len(set(events)):
        raise RuntimeError(f"frozen N72R9 development event set is not exactly 32 unique events: {len(events)}")
    return events


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--only-event", action="append", default=[])
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--manifest-name", default="batch_manifest.json")
    args = parser.parse_args()
    if int(args.shard_count) < 1 or not 0 <= int(args.shard_index) < int(args.shard_count):
        raise ValueError("shard-index must be in [0, shard-count)")
    events = load_events()
    if args.only_event:
        selected = set(str(item) for item in args.only_event)
        unknown = sorted(selected.difference(events))
        if unknown:
            raise RuntimeError(f"requested event is outside frozen protocol: {unknown}")
        events = [event for event in events if event in selected]
    if int(args.limit) > 0:
        events = events[: int(args.limit)]
    if int(args.shard_count) > 1:
        events = [
            event
            for ordinal, event in enumerate(events)
            if ordinal % int(args.shard_count) == int(args.shard_index)
        ]
    args.output_root.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, Any]] = []
    started = now_utc()
    for event_id in events:
        event_dir = args.output_root / "events" / event_id
        done_path = event_dir / "done.json"
        result_path = event_dir / "result.json"
        candidates_path = event_dir / "candidates.json"
        audit_path = event_dir / "audit.json"
        if done_path.is_file() and result_path.is_file() and candidates_path.is_file() and audit_path.is_file():
            done = read_json(done_path)
            result = read_json(result_path)
            expected_result_hash = done.get("result_sha256")
            expected_candidates_hash = done.get("candidates_sha256")
            expected_audit_hash = done.get("audit_sha256")
            valid_existing = (
                done.get("status") == "PASS_N72R10_TRUE_FUTURE_REQUERY_EVENT"
                and result.get("status") == "PASS_N72R10_TRUE_FUTURE_REQUERY_EVENT"
                and expected_result_hash == sha256_file(result_path)
                and expected_candidates_hash == sha256_file(candidates_path)
                and expected_audit_hash == sha256_file(audit_path)
            )
            if valid_existing:
                results.append({
                    "event_id": event_id,
                    "returncode": 0,
                    "status": "PASS",
                    "resumed_existing": True,
                    "log": None,
                    "done": str(done_path),
                    "done_sha256": sha256_file(done_path),
                    "failure": None,
                })
                continue
            raise RuntimeError(f"existing event artifact failed integrity validation: {event_dir}")
        log_path = args.output_root / "logs" / f"{event_id}.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        command = [
            sys.executable,
            str(WORKER),
            "--event-id", event_id,
            "--output-root", str(args.output_root),
            "--device", str(args.device),
        ]
        with log_path.open("w", encoding="utf-8") as log:
            completed = subprocess.run(
                command,
                cwd=str(ROOT),
                env={**os.environ, "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"},
                stdout=log,
                stderr=subprocess.STDOUT,
                check=False,
            )
        done_path = args.output_root / "events" / event_id / "done.json"
        failure_path = args.output_root / "attempts" / f"{event_id}.failure.json"
        results.append({
            "event_id": event_id,
            "returncode": int(completed.returncode),
            "status": "PASS" if completed.returncode == 0 and done_path.is_file() else "FAIL",
            "resumed_existing": False,
            "log": str(log_path),
            "done": str(done_path) if done_path.is_file() else None,
            "done_sha256": sha256_file(done_path) if done_path.is_file() else None,
            "failure": str(failure_path) if failure_path.is_file() else None,
        })
    passed = sum(item["status"] == "PASS" for item in results)
    batch = {
        "schema_version": "N72R10_TRUE_FUTURE_REQUERY_BATCH_V1",
        "status": "PASS_N72R10_TRUE_FUTURE_REQUERY_BATCH" if passed == len(events) else "PARTIAL_N72R10_TRUE_FUTURE_REQUERY_BATCH",
        "requested_event_count": len(events),
        "completed_event_count": passed,
        "failed_event_count": len(events) - passed,
        "resumed_existing_count": sum(bool(item.get("resumed_existing")) for item in results),
        "independent_process_per_event": True,
        "max_concurrent_processes": 1,
        "device": str(args.device),
        "shard_index": int(args.shard_index),
        "shard_count": int(args.shard_count),
        "protocol": str(PROTOCOL_PATH),
        "protocol_sha256": sha256_file(PROTOCOL_PATH),
        "results": results,
        "runtime_future_gt_used": False,
        "posthoc_gt_used": False,
        "interaction_source": "simulated_from_gt",
        "not_real_human_evidence": True,
        "started_at_utc": started,
        "finished_at_utc": now_utc(),
    }
    atomic_json(args.output_root / str(args.manifest_name), batch)
    print(json.dumps(batch, sort_keys=True))
    return 0 if batch["status"].startswith("PASS") else 1


if __name__ == "__main__":
    raise SystemExit(main())
