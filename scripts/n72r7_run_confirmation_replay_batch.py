#!/usr/bin/env python3
"""Run confirmation D1/D2 workers serially, one process per event."""

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
WORKER = ROOT / "scripts/n72r7_confirmation_replay.py"
PROTOCOL = ROOT / "outputs/N72R7/confirmation/confirmation_protocol.json"


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
    parser.add_argument("--target-root", type=Path, required=True)
    parser.add_argument("--target-attempt", type=int, default=2)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--attempt", type=int, default=1)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--python", dest="python_executable", default=sys.executable)
    args = parser.parse_args()
    target_root = args.target_root if args.target_root.is_absolute() else ROOT / args.target_root
    output_root = args.output_root if args.output_root.is_absolute() else ROOT / args.output_root
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    event_ids = [str(item["event_id"]) for item in protocol.get("events", [])]
    results: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for event_id in event_ids:
        command = [
            str(args.python_executable), str(WORKER),
            "--event-id", event_id,
            "--target-root", str(target_root),
            "--target-attempt", str(int(args.target_attempt)),
            "--output-root", str(output_root),
            "--attempt", str(int(args.attempt)),
            "--device", str(args.device),
        ]
        completed = subprocess.run(command, cwd=str(ROOT), text=True, capture_output=True, check=False)
        log_path = output_root / "logs" / f"{event_id}.attempt{int(args.attempt)}.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        fd, temporary = tempfile.mkstemp(prefix=f".{log_path.name}.", suffix=".tmp", dir=str(log_path.parent))
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                handle.write(
                    f"command={json.dumps(command, ensure_ascii=False)}\n"
                    f"returncode={completed.returncode}\n"
                    f"stdout:\n{completed.stdout}\n"
                    f"stderr:\n{completed.stderr}\n"
                )
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, log_path)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
        record = {
            "event_id": event_id,
            "status": "PASS" if completed.returncode == 0 else "FAIL",
            "returncode": int(completed.returncode),
            "log": str(log_path),
            "stdout": completed.stdout[-2000:],
            "stderr": completed.stderr[-4000:],
        }
        (results if completed.returncode == 0 else failures).append(record)
        print(json.dumps({"event_id": event_id, "status": record["status"], "returncode": completed.returncode}, ensure_ascii=False), flush=True)
    status = "PASS_N72R7_CONFIRMATION_REPLAY_BATCH" if not failures else "PARTIAL_N72R7_CONFIRMATION_REPLAY_BATCH"
    atomic_json(output_root / f"batch_attempt{int(args.attempt)}.json", {
        "schema_version": "N72R7_CONFIRMATION_REPLAY_BATCH_V1",
        "status": status,
        "protocol": str(PROTOCOL),
        "protocol_sha256": __import__("hashlib").sha256(PROTOCOL.read_bytes()).hexdigest(),
        "requested_event_count": len(event_ids),
        "completed_event_count": len(results),
        "failed_event_count": len(failures),
        "results": results,
        "failures": failures,
        "attempt": int(args.attempt),
        "device": str(args.device),
        "python": str(args.python_executable),
        "one_process_per_event": True,
        "runtime_future_gt_used": False,
        "posthoc_gt_used": False,
        "interaction_source": "simulated_from_gt",
        "not_real_human_evidence": True,
        "started_at_utc": now_utc(),
        "finished_at_utc": now_utc(),
    })
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
