#!/usr/bin/env python3
"""Run N72R10 replay workers one at a time with resumable manifests."""

from __future__ import annotations

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


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


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


def atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    atomic_write(path, json.dumps(dict(value), ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n")


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object: {path}")
    return value


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--event-id", action="append", default=[])
    args = parser.parse_args()
    output_root = Path(args.output_root)
    if not output_root.is_absolute():
        output_root = ROOT / output_root
    protocol = read_json(PROTOCOL_PATH)
    all_events = {str(item["event_id"]): item for item in protocol.get("source_event_selection", {}).get("events", [])}
    events = sorted(all_events)
    if args.event_id:
        requested = [str(item) for item in args.event_id]
        unknown = sorted(set(requested) - set(all_events))
        if unknown:
            raise RuntimeError(f"unknown frozen event IDs: {unknown}")
        events = requested
    elif int(args.limit) > 0:
        events = events[: int(args.limit)]
    results: list[dict[str, Any]] = []
    started = now_utc()
    for event_id in events:
        event_dir = output_root / event_id
        done_path = event_dir / "done.json"
        reused = False
        returncode = 0
        if done_path.is_file():
            try:
                done = read_json(done_path)
                reused = done.get("status") == "PASS_N72R10_RUNTIME_AND_POSTHOC_EVENT"
            except Exception:
                reused = False
        if reused:
            completed = 0
            status = "PASS"
            log_path = output_root / "logs" / f"{event_id}.resumed.log"
            log_path.parent.mkdir(parents=True, exist_ok=True)
            log_path.write_text("reused sealed N72R10 event\n", encoding="utf-8")
        else:
            log_path = output_root / "logs" / f"{event_id}.log"
            log_path.parent.mkdir(parents=True, exist_ok=True)
            command = [
                sys.executable,
                str(ROOT / "scripts/n72r10_true_closed_loop_replay.py"),
                "--event-id", event_id,
                "--output-root", str(output_root),
                "--device", str(args.device),
            ]
            with log_path.open("w", encoding="utf-8") as log:
                completed = subprocess.run(command, cwd=str(ROOT), stdout=log, stderr=subprocess.STDOUT, check=False)
            returncode = int(completed.returncode)
            try:
                done = read_json(done_path)
            except Exception:
                done = {}
            status = "PASS" if completed.returncode == 0 and done.get("status") == "PASS_N72R10_RUNTIME_AND_POSTHOC_EVENT" else "FAIL"
        results.append({
            "event_id": event_id,
            "status": status,
            "reused": reused,
            "returncode": returncode,
            "log": str(log_path),
            "done": str(done_path) if done_path.is_file() else None,
            "done_sha256": sha256_file(done_path) if done_path.is_file() else None,
        })
    passed = sum(item["status"] == "PASS" for item in results)
    batch = {
        "schema_version": "N72R10_TRUE_CLOSED_LOOP_REPLAY_BATCH_V1",
        "status": "PASS_N72R10_TRUE_CLOSED_LOOP_REPLAY_BATCH" if passed == len(events) else "PARTIAL_N72R10_TRUE_CLOSED_LOOP_REPLAY_BATCH",
        "requested_event_count": len(events),
        "completed_event_count": passed,
        "failed_event_count": len(events) - passed,
        "independent_process_per_event": True,
        "max_concurrent_processes": 1,
        "device": str(args.device),
        "results": results,
        "protocol": str(PROTOCOL_PATH),
        "protocol_sha256": sha256_file(PROTOCOL_PATH),
        "runtime_future_gt_used": False,
        "interaction_source": "simulated_from_gt",
        "not_real_human_evidence": True,
        "started_at_utc": started,
        "finished_at_utc": now_utc(),
    }
    atomic_json(output_root / "batch_manifest.json", batch)
    print(json.dumps(batch, ensure_ascii=False, sort_keys=True))
    return 0 if batch["status"].startswith("PASS") else 1


if __name__ == "__main__":
    raise SystemExit(main())
