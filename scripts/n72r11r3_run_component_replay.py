#!/usr/bin/env python3
"""Run the frozen N72R9 component replay one event at a time.

This supervisor deliberately uses blocking ``subprocess.run`` calls rather
than a polling scheduler.  Each child owns one event and one physical device;
all child logs and failed attempts are retained.  It never changes the frozen
event list or chooses an event using a post-treatment result.
"""

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
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "outputs/N72R9/protocol.json"
DEFAULT_OUTPUT = ROOT / "outputs/N72R11R3/component_replay_attempt_01"
VARIANTS = (
    "E1_V3_LEGACY_INJECTION",
    "E2_V3_CORRECTED_BRIDGE",
    "E3_V3_CORRECTED_BRIDGE_LIVE",
)


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
    events = [dict(item) for item in payload.get("source_event_selection", {}).get("events", [])]
    if len(events) != 32 or len({str(item.get("event_id")) for item in events}) != 32:
        raise RuntimeError(f"expected exactly 32 frozen N72R9 events, found {len(events)}")
    for item in events:
        if item.get("runtime_future_gt_used") is not False or item.get("interaction_source") != "simulated_from_gt":
            raise RuntimeError(f"invalid frozen event provenance: {item.get('event_id')}")
    return sorted(events, key=lambda item: str(item["event_id"]))


def child_command(
    event_id: str,
    output_root: Path,
    model: Path,
    bridge: Path | None,
    device: str,
    variants: tuple[str, ...],
    enable_live: bool,
    force_trigger: bool,
) -> list[str]:
    command = [
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
        "--horizon",
        "100",
        "--variants",
        *variants,
    ]
    if bridge is not None:
        command.extend(("--bridge-checkpoint", str(bridge)))
    if enable_live:
        command.append("--enable-live")
    if force_trigger:
        command.append("--force-trigger")
    return command


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--attempt", type=int, default=1)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--model-checkpoint", type=Path, required=True)
    parser.add_argument("--bridge-checkpoint", type=Path, default=None)
    parser.add_argument("--variants", nargs="+", choices=VARIANTS, default=list(VARIANTS))
    parser.add_argument("--event-id", action="append", default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--enable-live", action="store_true")
    parser.add_argument("--force-trigger", action="store_true")
    parser.add_argument("--resource-censored", action="store_true")
    args = parser.parse_args()
    if int(args.attempt) < 1:
        raise SystemExit("attempt must be positive")
    variants = tuple(str(value) for value in args.variants)
    if not variants or len(set(variants)) != len(variants):
        raise SystemExit("variants must be non-empty and unique")
    if any(value in {"E2_V3_CORRECTED_BRIDGE", "E3_V3_CORRECTED_BRIDGE_LIVE"} for value in variants) and args.bridge_checkpoint is None:
        raise SystemExit("E2/E3 replay requires --bridge-checkpoint")
    output_root = args.output_root if args.output_root.is_absolute() else ROOT / args.output_root
    model = args.model_checkpoint if args.model_checkpoint.is_absolute() else ROOT / args.model_checkpoint
    bridge = None if args.bridge_checkpoint is None else (args.bridge_checkpoint if args.bridge_checkpoint.is_absolute() else ROOT / args.bridge_checkpoint)
    if not model.is_file() or (bridge is not None and not bridge.is_file()):
        raise SystemExit(f"missing model/bridge checkpoint: {model} {bridge}")
    events = frozen_events()
    if args.event_id:
        wanted = {str(value) for value in args.event_id}
        available = {str(item["event_id"]) for item in events}
        missing = sorted(wanted - available)
        if missing:
            raise SystemExit(f"requested IDs are not frozen N72R9 events: {missing}")
        events = [item for item in events if str(item["event_id"]) in wanted]
    output_root.mkdir(parents=True, exist_ok=True)
    logs_root = output_root / "logs"
    manifest_path = output_root / f"component_replay_manifest_attempt_{int(args.attempt):02d}.json"
    records = {
        str(item["event_id"]): {
            "event_id": str(item["event_id"]),
            "sequence": str(item["sequence"]),
            "action_type": str(item["action_type"]),
            "status": "NOT_RUN",
            "attempt": int(args.attempt),
            "returncode": None,
            "log": None,
            "command": None,
        }
        for item in events
    }

    def write_manifest(status: str) -> None:
        counts: dict[str, int] = {}
        for record in records.values():
            counts[str(record["status"])] = counts.get(str(record["status"]), 0) + 1
        atomic_json(
            manifest_path,
            {
                "schema_version": "N72R11R3_COMPONENT_REPLAY_MANIFEST_V1",
                "status": status,
                "created_at_utc": now_utc(),
                "protocol": str(PROTOCOL),
                "protocol_sha256": sha256_file(PROTOCOL),
                "model_checkpoint": str(model),
                "model_checkpoint_sha256": sha256_file(model),
                "bridge_checkpoint": None if bridge is None else str(bridge),
                "bridge_checkpoint_sha256": None if bridge is None else sha256_file(bridge),
                "attempt": int(args.attempt),
                "device": str(args.device),
                "variants": list(variants),
                "enable_live": bool(args.enable_live),
                "force_trigger": bool(args.force_trigger),
                "event_count": len(events),
                "runtime_future_gt_used": False,
                "interaction_source": "simulated_from_gt",
                "not_real_human_evidence": True,
                "resource_censored_development": bool(args.resource_censored),
                "execution": "one blocking child per event; no concurrent GPU ownership",
                "records": [records[event_id] for event_id in sorted(records)],
                "counts": counts,
            },
        )

    write_manifest("RUNNING")
    for event in events:
        event_id = str(event["event_id"])
        log_path = logs_root / f"{event_id}.attempt{int(args.attempt):02d}.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        command = child_command(
            event_id,
            output_root,
            model,
            bridge,
            str(args.device),
            variants,
            bool(args.enable_live or "E3_V3_CORRECTED_BRIDGE_LIVE" in variants),
            bool(args.force_trigger),
        )
        env = os.environ.copy()
        env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
        env["PYTHONUNBUFFERED"] = "1"
        records[event_id].update({"status": "RUNNING", "log": str(log_path), "command": command, "started_at_utc": now_utc()})
        write_manifest("RUNNING")
        with log_path.open("wb") as log_handle:
            completed = subprocess.run(command, cwd=str(ROOT), env=env, stdout=log_handle, stderr=subprocess.STDOUT, check=False)
        done = output_root / event_id / "done.json"
        done_exists = done.is_file()
        records[event_id].update({
            "status": "PASS" if completed.returncode == 0 and done_exists else "FAIL_CHILD",
            "returncode": int(completed.returncode),
            "done": str(done) if done_exists else None,
            "done_sha256": sha256_file(done) if done_exists else None,
            "finished_at_utc": now_utc(),
        })
        write_manifest("RUNNING")
    final_status = "PASS_ALL_SELECTED" if all(record["status"] == "PASS" for record in records.values()) else "PARTIAL_WITH_FAILURES"
    write_manifest(final_status)
    counts = {key: sum(record["status"] == key for record in records.values()) for key in sorted({str(record["status"]) for record in records.values()})}
    print(json.dumps({"status": final_status, "manifest": str(manifest_path), "counts": counts}, sort_keys=True))
    return 0 if final_status == "PASS_ALL_SELECTED" else 1


if __name__ == "__main__":
    raise SystemExit(main())
