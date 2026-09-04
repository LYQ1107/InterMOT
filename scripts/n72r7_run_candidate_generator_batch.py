#!/usr/bin/env python3
"""Run R5 candidate-generator events serially, one independent process each."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
WORKER = ROOT / "scripts/n72r7_candidate_generator_requery.py"


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--event-id", action="append", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--attempt", type=int, default=1)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    output_root = args.output_root if args.output_root.is_absolute() else ROOT / args.output_root
    output_root.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    started = now_utc()
    for event_id in args.event_id:
        command = [
            sys.executable,
            str(WORKER),
            "--event-id",
            str(event_id),
            "--output-root",
            str(output_root),
            "--attempt",
            str(int(args.attempt)),
            "--device",
            str(args.device),
        ]
        completed = subprocess.run(command, cwd=str(ROOT), text=True, capture_output=True, check=False)
        log_path = output_root / "logs" / f"{event_id}.attempt{int(args.attempt)}.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text(
            f"command={json.dumps(command, ensure_ascii=False)}\n"
            f"returncode={completed.returncode}\n"
            f"stdout:\n{completed.stdout}\n"
            f"stderr:\n{completed.stderr}\n",
            encoding="utf-8",
        )
        if completed.returncode == 0:
            results.append({
                "event_id": str(event_id),
                "status": "PASS",
                "log": str(log_path),
                "stdout": completed.stdout[-2000:],
            })
        else:
            failures.append({
                "event_id": str(event_id),
                "status": "FAIL",
                "returncode": int(completed.returncode),
                "log": str(log_path),
                "stdout": completed.stdout[-2000:],
                "stderr": completed.stderr[-4000:],
            })
        print(json.dumps({"event_id": str(event_id), "status": "PASS" if completed.returncode == 0 else "FAIL", "returncode": completed.returncode}, ensure_ascii=False), flush=True)
    status = "PASS_N72R7_REQUERY_BATCH" if not failures else "PARTIAL_N72R7_REQUERY_BATCH"
    atomic_json(output_root / f"batch_attempt{int(args.attempt)}.json", {
        "schema_version": "N72R7_CANDIDATE_GENERATOR_REQUERY_BATCH_V1",
        "status": status,
        "requested_event_count": len(args.event_id),
        "completed_event_count": len(results),
        "failed_event_count": len(failures),
        "results": results,
        "failures": failures,
        "attempt": int(args.attempt),
        "device": str(args.device),
        "one_process_per_event": True,
        "runtime_future_gt_used": False,
        "started_at_utc": started,
        "finished_at_utc": now_utc(),
    })
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
