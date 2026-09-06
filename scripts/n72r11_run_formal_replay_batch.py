#!/usr/bin/env python3
"""Run the fixed N72R9 32-event N72R11 E0/E1/E2 replay safely.

The worker is one event per process and the supervisor reserves physical GPU
IDs, so a completed child cannot cause a second child to land on the same
card.  The manifest records every event, including failures, and the output
root is never reused for a new attempt.
"""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "outputs/N72R9/protocol.json"
DEFAULT_OUTPUT = ROOT / "outputs/N72R11/formal_replay_attempt_01"
DEFAULT_GPUS = (1, 2, 3, 4)


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


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object: {path}")
    return value


def frozen_events() -> list[dict[str, Any]]:
    payload = read_json(PROTOCOL)
    events = [dict(value) for value in payload.get("source_event_selection", {}).get("events", [])]
    if len(events) != 32 or len({str(value.get("event_id")) for value in events}) != 32:
        raise RuntimeError(f"expected 32 frozen N72R9 events, found {len(events)}")
    for event in events:
        if event.get("runtime_future_gt_used") is not False or event.get("interaction_source") != "simulated_from_gt":
            raise RuntimeError(f"invalid frozen event provenance: {event.get('event_id')}")
    return sorted(events, key=lambda value: str(value["event_id"]))


def child_command(event_id: str, output_root: Path, model: Path, bridge: Path, device: str) -> list[str]:
    return [
        sys.executable,
        "-u",
        str(ROOT / "scripts/n72r11_on_demand_replay.py"),
        "--event-id",
        str(event_id),
        "--output-root",
        str(output_root),
        "--device",
        str(device),
        "--model-checkpoint",
        str(model),
        "--bridge-checkpoint",
        str(bridge),
        "--horizon",
        "100",
        "--enable-live",
    ]


def main() -> int:
    parser = __import__("argparse").ArgumentParser()
    parser.add_argument("--attempt", type=int, default=1)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--model-checkpoint", type=Path, required=True)
    parser.add_argument("--bridge-checkpoint", type=Path, required=True)
    parser.add_argument("--max-workers", type=int, default=4)
    parser.add_argument("--gpu-ids", default=",".join(str(value) for value in DEFAULT_GPUS))
    parser.add_argument("--event-id", action="append", default=None)
    parser.add_argument("--resource-censored", action="store_true")
    args = parser.parse_args()
    output_root = args.output_root if args.output_root.is_absolute() else ROOT / args.output_root
    model_path = args.model_checkpoint if args.model_checkpoint.is_absolute() else ROOT / args.model_checkpoint
    bridge_path = args.bridge_checkpoint if args.bridge_checkpoint.is_absolute() else ROOT / args.bridge_checkpoint
    if not model_path.is_file() or not bridge_path.is_file():
        raise SystemExit(f"missing model/bridge checkpoint: {model_path} {bridge_path}")
    gpu_ids = tuple(int(value.strip()) for value in str(args.gpu_ids).split(",") if value.strip())
    if not gpu_ids or int(args.max_workers) < 1 or int(args.max_workers) > len(gpu_ids):
        raise SystemExit("max-workers must be in 1..number of gpu-ids")
    if int(args.attempt) < 1:
        raise SystemExit("attempt must be positive")
    events = frozen_events()
    if args.event_id:
        wanted = {str(value) for value in args.event_id}
        events = [event for event in events if str(event["event_id"]) in wanted]
        if len(events) != len(wanted):
            raise SystemExit(f"some requested event IDs are not frozen N72R9 events: {sorted(wanted - {str(e['event_id']) for e in events})}")
    if not events:
        raise SystemExit("selected formal event set is empty")
    output_root.mkdir(parents=True, exist_ok=True)
    manifest_path = output_root / f"formal_batch_manifest_attempt_{int(args.attempt):02d}.json"
    logs = output_root / "logs"
    records: dict[str, dict[str, Any]] = {}
    for event in events:
        event_id = str(event["event_id"])
        records[event_id] = {
            "event_id": event_id,
            "sequence": str(event["sequence"]),
            "action_type": str(event["action_type"]),
            "status": "NOT_RUN",
            "attempt": int(args.attempt),
            "gpu_id": None,
            "returncode": None,
            "log": None,
            "command": None,
        }

    def write_manifest(status: str) -> None:
        counts: dict[str, int] = {}
        for record in records.values():
            counts[str(record["status"])] = counts.get(str(record["status"]), 0) + 1
        atomic_json(
            manifest_path,
            {
                "schema_version": "N72R11_FORMAL_REPLAY_BATCH_MANIFEST_V1",
                "status": status,
                "created_at_utc": now_utc(),
                "protocol": str(PROTOCOL),
                "protocol_sha256": sha256_file(PROTOCOL),
                "model_checkpoint": str(model_path),
                "model_checkpoint_sha256": sha256_file(model_path),
                "bridge_checkpoint": str(bridge_path),
                "bridge_checkpoint_sha256": sha256_file(bridge_path),
                "attempt": int(args.attempt),
                "max_workers": int(args.max_workers),
                "gpu_ids": list(gpu_ids),
                "event_count": len(events),
                "runtime_future_gt_used": False,
                "interaction_source": "simulated_from_gt",
                "not_real_human_evidence": True,
                "resource_censored_development": bool(args.resource_censored),
                "records": [records[event_id] for event_id in sorted(records)],
                "counts": counts,
            },
        )

    write_manifest("RUNNING")
    pending = [event for event in events if records[str(event["event_id"])] ["status"] == "NOT_RUN"]
    active: dict[int, tuple[dict[str, Any], subprocess.Popen[bytes], Any, Path, int]] = {}
    next_index = 0
    try:
        while next_index < len(pending) or active:
            while next_index < len(pending) and len(active) < int(args.max_workers):
                event = pending[next_index]
                next_index += 1
                event_id = str(event["event_id"])
                available = [gpu_id for gpu_id in gpu_ids if gpu_id not in {value[4] for value in active.values()}]
                if not available:
                    raise RuntimeError("formal scheduler has no free physical GPU slot")
                gpu_id = available[0]
                log_path = logs / f"{event_id}.attempt{int(args.attempt)}.log"
                log_path.parent.mkdir(parents=True, exist_ok=True)
                log_handle = log_path.open("wb")
                env = os.environ.copy()
                env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
                env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
                env["PYTHONUNBUFFERED"] = "1"
                command = child_command(event_id, output_root, model_path, bridge_path, "cuda:0")
                process = subprocess.Popen(command, cwd=str(ROOT), env=env, stdout=log_handle, stderr=subprocess.STDOUT)
                records[event_id].update({
                    "status": "RUNNING",
                    "gpu_id": gpu_id,
                    "log": str(log_path),
                    "command": command,
                    "started_at_utc": now_utc(),
                    "started_epoch": time.time(),
                })
                active[process.pid] = (event, process, log_handle, log_path, gpu_id)
                write_manifest("RUNNING")
            if not active:
                continue
            time.sleep(1.0)
            finished: list[int] = []
            for pid, (event, process, log_handle, log_path, gpu_id) in active.items():
                returncode = process.poll()
                if returncode is None:
                    continue
                event_id = str(event["event_id"])
                log_handle.flush()
                log_handle.close()
                done = output_root / event_id / "done.json"
                done_exists = done.is_file()
                state = "PASS" if returncode == 0 and done_exists else "FAIL_CHILD"
                records[event_id].update({
                    "status": state,
                    "returncode": int(returncode),
                    "done": str(done) if done_exists else None,
                    "done_sha256": sha256_file(done) if done_exists else None,
                    "finished_at_utc": now_utc(),
                    "finished_epoch": time.time(),
                })
                finished.append(pid)
                write_manifest("RUNNING")
            for pid in finished:
                del active[pid]
    except BaseException:
        write_manifest("INTERRUPTED_WITH_EVIDENCE")
        raise
    final_status = "PASS_ALL_SELECTED" if all(record["status"] == "PASS" for record in records.values()) else "PARTIAL_WITH_FAILURES"
    write_manifest(final_status)
    print(json.dumps({"status": final_status, "manifest": str(manifest_path), "counts": {key: sum(record["status"] == key for record in records.values()) for key in sorted({str(record["status"]) for record in records.values()})}}, sort_keys=True))
    return 0 if final_status == "PASS_ALL_SELECTED" else 1


if __name__ == "__main__":
    raise SystemExit(main())
