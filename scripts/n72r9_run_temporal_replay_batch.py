#!/usr/bin/env python3
"""Run N72R9 event workers sequentially with one process per event."""

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


def atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(dict(value), handle, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--label", required=True)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    events = sorted(str(item["event_id"]) for item in protocol.get("source_event_selection", {}).get("events", []))
    if args.limit > 0:
        events = events[: int(args.limit)]
    output_root = ROOT / "outputs/N72R9/replay" / str(args.label)
    results: list[dict[str, Any]] = []
    started = now_utc()
    for event_id in events:
        log_path = output_root / "logs" / f"{event_id}.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        command = [sys.executable, str(ROOT / "scripts/n72r9_temporal_replay.py"), "--event-id", event_id, "--output-root", str(output_root), "--device", str(args.device)]
        with log_path.open("w", encoding="utf-8") as log:
            completed = subprocess.run(command, cwd=str(ROOT), stdout=log, stderr=subprocess.STDOUT, check=False)
        done_path = output_root / event_id / "done.json"
        result = {
            "event_id": event_id,
            "returncode": int(completed.returncode),
            "status": "PASS" if completed.returncode == 0 and done_path.is_file() else "FAIL",
            "log": str(log_path),
            "done": str(done_path) if done_path.is_file() else None,
            "done_sha256": sha256_file(done_path) if done_path.is_file() else None,
        }
        results.append(result)
    passed = sum(item["status"] == "PASS" for item in results)
    batch = {
        "schema_version": "N72R9_TEMPORAL_REPLAY_BATCH_V1",
        "status": "PASS_N72R9_REPLAY_BATCH" if passed == len(events) else "PARTIAL_N72R9_REPLAY_BATCH",
        "label": str(args.label),
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
    print(json.dumps(batch, sort_keys=True))
    return 0 if batch["status"].startswith("PASS") else 1


if __name__ == "__main__":
    raise SystemExit(main())
