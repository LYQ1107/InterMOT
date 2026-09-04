#!/usr/bin/env python3
"""Run the frozen N72R6 target-session jobs with bounded GPU concurrency.

Each child is still one independent event/process.  This controller only
assigns deterministic event IDs to at most four GPU slots and records every
child exit status; it does not retry or reinterpret a failed stream.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile
import time
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
STAGE08_MANIFEST = (
    ROOT
    / "outputs/N72R5R1/controller/round_05_branch_isolation_v0/full/stage08_runtime_manifest.json"
)
STREAM_SCRIPT = ROOT / "scripts/n72r6_target_correction_stream.py"


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def eligible_events() -> list[str]:
    payload = json.loads(STAGE08_MANIFEST.read_text(encoding="utf-8"))
    result: list[str] = []
    for event in payload.get("events", []):
        branches = {
            str(branch.get("branch")): branch
            for branch in event.get("branches", [])
            if isinstance(branch, dict)
        }
        branch = branches.get("B1_SPATIAL_CORRECTION_ONLY")
        if branch and branch.get("action_precondition_status") == "APPLIED":
            result.append(str(event["event_id"]))
    return result


def run_one(event_id: str, gpu: str, args: argparse.Namespace) -> dict[str, Any]:
    job_root = args.output_root / f"attempt_{args.attempt}"
    log_path = job_root / f"{event_id}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        args.python,
        str(STREAM_SCRIPT),
        "--event-id",
        event_id,
        "--attempt",
        str(args.attempt),
        "--device",
        "cuda",
        "--output-root",
        str(args.output_root),
    ]
    if args.recovery_mode:
        command.append("--recovery-mode")
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = str(gpu)
    environment["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    started = time.time()
    with log_path.open("w", encoding="utf-8") as log:
        completed = subprocess.run(
            command,
            cwd=ROOT,
            env=environment,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
        )
    return {
        "event_id": event_id,
        "gpu": str(gpu),
        "attempt": int(args.attempt),
        "returncode": int(completed.returncode),
        "status": "PASS_PROCESS_EXIT" if completed.returncode == 0 else "FAIL_PROCESS_EXIT",
        "log": str(log_path),
        "elapsed_sec": time.time() - started,
        "command": command,
        "created_at_utc": now_utc(),
    }


def run_group(events: list[str], gpu: str, args: argparse.Namespace) -> list[dict[str, Any]]:
    """Run one sequential queue so a GPU never hosts two child processes."""
    return [run_one(event_id, gpu, args) for event_id in events]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--attempt", type=int, default=1)
    parser.add_argument("--output-root", type=Path, default=ROOT / "outputs/N72R6/target_correction_stream/batch")
    parser.add_argument("--python", default="/home/lwr/anaconda3/envs/intermot/bin/python")
    parser.add_argument("--devices", nargs="+", default=["4", "5", "6", "7"])
    parser.add_argument("--exclude-event-id", action="append", default=[])
    parser.add_argument("--recovery-mode", action="store_true")
    args = parser.parse_args()
    args.output_root = args.output_root if args.output_root.is_absolute() else ROOT / args.output_root
    args.output_root = args.output_root.resolve()
    devices = [str(value) for value in args.devices]
    if not devices or len(devices) > 4:
        raise SystemExit("N72R6 permits one process per GPU and at most four GPUs")
    events = [event for event in eligible_events() if event not in set(args.exclude_event_id)]
    if not events:
        raise SystemExit("no eligible target-session events remain")

    groups = [events[index:: len(devices)] for index in range(len(devices))]
    results: list[dict[str, Any]] = []
    batch_started = time.time()
    with ThreadPoolExecutor(max_workers=len(devices)) as executor:
        futures = [
            executor.submit(run_group, group, devices[index], args)
            for index, group in enumerate(groups)
            if group
        ]
        for future in futures:
            results.extend(future.result())
    results.sort(key=lambda item: events.index(str(item["event_id"])))
    payload = {
        "schema_version": "N72R6_TARGET_STREAM_BATCH_V1",
        "status": "PASS_ALL_PROCESS_EXITS" if all(item["returncode"] == 0 for item in results) else "PARTIAL_PROCESS_EXITS",
        "attempt": int(args.attempt),
        "event_count": len(events),
        "events": events,
        "devices": devices,
        "stage08_manifest": str(STAGE08_MANIFEST),
        "stage08_manifest_sha256": sha256_file(STAGE08_MANIFEST),
        "stream_script": str(STREAM_SCRIPT),
        "runtime_future_gt_used": False,
        "target_session_recovery_mode": bool(args.recovery_mode),
        "results": results,
        "completed_count": sum(item["returncode"] == 0 for item in results),
        "failed_count": sum(item["returncode"] != 0 for item in results),
        "elapsed_sec": time.time() - batch_started,
        "created_at_utc": now_utc(),
    }
    atomic_json(args.output_root / f"batch_attempt_{args.attempt}.json", payload)
    print(json.dumps({"status": payload["status"], "event_count": len(events), "failed_count": payload["failed_count"]}))
    return 0 if payload["failed_count"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
