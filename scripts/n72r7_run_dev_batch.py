#!/usr/bin/env python3
"""Run one N72R7 development event per independent Python process."""

from __future__ import annotations

from datetime import datetime, timezone
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from n72r7_dev_replay import atomic_json, atomic_write, read_json, TARGET_MANIFEST  # noqa: E402


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=("D1", "D2"), required=True)
    parser.add_argument("--output-root", required=True)
    args = parser.parse_args()

    output_root = Path(args.output_root)
    if not output_root.is_absolute():
        output_root = ROOT / output_root
    output_root.mkdir(parents=True, exist_ok=True)
    target_manifest = read_json(TARGET_MANIFEST)
    event_ids = sorted(str(item["event_id"]) for item in target_manifest.get("selected", []))
    if len(event_ids) != 32 or len(event_ids) != len(set(event_ids)):
        raise RuntimeError(f"frozen N72R6 event set must contain 32 unique events, got {len(event_ids)}")

    process_log_root = output_root / "process_logs"
    records: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for index, event_id in enumerate(event_ids):
        command = [
            sys.executable,
            str(ROOT / "scripts/n72r7_dev_replay.py"),
            "--variant",
            args.variant,
            "--event-id",
            event_id,
            "--output-root",
            str(output_root),
        ]
        completed = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, check=False)
        stdout_path = process_log_root / f"{index:03d}.{event_id}.{args.variant}.stdout.log"
        stderr_path = process_log_root / f"{index:03d}.{event_id}.{args.variant}.stderr.log"
        atomic_write(stdout_path, completed.stdout)
        atomic_write(stderr_path, completed.stderr)
        event_manifest = output_root / event_id / "event_manifest.json"
        if completed.returncode == 0 and event_manifest.is_file():
            artifact = read_json(event_manifest)
            record = {
                "event_id": event_id,
                "variant": args.variant,
                "status": artifact.get("status"),
                "event_manifest": str(event_manifest),
                "stdout": str(stdout_path),
                "stderr": str(stderr_path),
                "returncode": int(completed.returncode),
                "independent_process": True,
            }
            records.append(record)
            print(json.dumps({"event_id": event_id, "variant": args.variant, "status": record["status"]}))
        else:
            failure = {
                "event_id": event_id,
                "variant": args.variant,
                "status": "FAIL",
                "returncode": int(completed.returncode),
                "stdout": str(stdout_path),
                "stderr": str(stderr_path),
                "child_output": completed.stdout[-4000:],
                "child_error": completed.stderr[-4000:],
                "independent_process": True,
            }
            failures.append(failure)
            print(json.dumps(failure, ensure_ascii=False))

    record_keys = [(item["event_id"], item["variant"]) for item in records]
    failure_keys = [(item["event_id"], item["variant"]) for item in failures]
    all_keys = record_keys + failure_keys
    duplicates = sorted({key for key in all_keys if all_keys.count(key) > 1})
    observed = {item[0] for item in all_keys}
    batch = {
        "schema_version": "N72R7_INDEPENDENT_DEV_BATCH_V1",
        "status": "PASS_N72R7_INDEPENDENT_DEV_BATCH" if len(records) == 32 and not failures and not duplicates else "PARTIAL_N72R7_INDEPENDENT_DEV_BATCH",
        "variant": args.variant,
        "requested_event_count": len(event_ids),
        "completed_event_count": len(records),
        "failed_event_count": len(failures),
        "missing_event_ids": sorted(set(event_ids) - observed),
        "unexpected_event_ids": sorted(observed - set(event_ids)),
        "duplicate_event_keys": [list(key) for key in duplicates],
        "results": records,
        "failures": failures,
        "event_source_manifest": str(TARGET_MANIFEST),
        "event_source_manifest_sha256": __import__("hashlib").sha256(TARGET_MANIFEST.read_bytes()).hexdigest(),
        "independent_process_per_event": True,
        "max_concurrent_processes": 1,
        "runtime_future_gt_used": False,
        "posthoc_gt_used": False,
        "interaction_source": "simulated_from_gt",
        "not_real_human_evidence": True,
        "created_at_utc": now_utc(),
    }
    atomic_json(output_root / "batch_manifest.json", batch)
    print(json.dumps({"status": batch["status"], "completed": len(records), "failed": len(failures)}))
    return 0 if batch["status"] == "PASS_N72R7_INDEPENDENT_DEV_BATCH" else 1


if __name__ == "__main__":
    raise SystemExit(main())
