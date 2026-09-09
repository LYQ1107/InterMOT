#!/usr/bin/env python3
"""Run the frozen N72R9 E0/E1A paired replay, one event per child process.

The event list, protocol, horizon, and checkpoint are frozen inputs.  This
runner only adds the N72R11R4 exact-on-policy V3 treatment and keeps every
child log and failed attempt.  It deliberately owns one GPU serially so an
event cannot inherit model or SAM3 state from another event.
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
from typing import Any, Sequence

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "outputs/N72R9/protocol.json"
DEFAULT_OUTPUT = ROOT / "outputs/N72R11R4/formal_e1a_attempt_01"
VARIANT_CHOICES = ("E1A_EXACT_ONPOLICY_V3_LEGACY", "E1B_PCTIS_LEGACY", "E2_PCTIS_LIVE")


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
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    temporary = Path(temporary_name)
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
        if temporary.exists():
            temporary.unlink()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object: {path}")
    return value


def frozen_events() -> list[dict[str, Any]]:
    payload = read_json(PROTOCOL)
    events = [dict(item) for item in payload.get("source_event_selection", {}).get("events", [])]
    event_ids = [str(item.get("event_id")) for item in events]
    if len(events) != 32 or len(set(event_ids)) != 32:
        raise RuntimeError(f"expected exactly 32 unique frozen N72R9 events, found {len(events)}")
    for item in events:
        if item.get("runtime_future_gt_used") is not False:
            raise RuntimeError(f"frozen event permits runtime future GT: {item.get('event_id')}")
        if item.get("interaction_source") != "simulated_from_gt":
            raise RuntimeError(f"unexpected interaction provenance: {item.get('event_id')}")
        if int(item.get("event_frame", -1)) < 0 or len(item.get("future_window", [])) != 2:
            raise RuntimeError(f"malformed frozen event window: {item.get('event_id')}")
    return sorted(events, key=lambda item: str(item["event_id"]))


def child_command(event_id: str, output_root: Path, model: Path, device: str, horizon: int, variants: Sequence[str], scorer_kind: str, enable_live: bool) -> list[str]:
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
        "--scorer-kind",
        str(scorer_kind),
        "--horizon",
        str(int(horizon)),
        "--variants",
        *[str(value) for value in variants],
    ]
    if enable_live:
        command.append("--enable-live")
    return command


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--attempt", type=int, default=1)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--model-checkpoint", type=Path, required=True)
    parser.add_argument("--model-kind", choices=("v3", "pctis"), default="v3")
    parser.add_argument("--variant", choices=VARIANT_CHOICES, default=None)
    parser.add_argument("--variants", nargs="+", choices=VARIANT_CHOICES, default=None)
    parser.add_argument("--enable-live", action="store_true")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--horizon", type=int, default=100)
    parser.add_argument("--event-id", action="append", default=None)
    parser.add_argument("--resource-censored", action="store_true")
    args = parser.parse_args()
    if int(args.attempt) < 1:
        raise SystemExit("attempt must be positive")
    if int(args.horizon) != 100:
        raise SystemExit("N72R11R4 formal replay requires the frozen H100 horizon")
    variants = tuple(str(value) for value in (args.variants if args.variants is not None else ([args.variant] if args.variant is not None else [VARIANT_CHOICES[0]])))
    if not variants or len(set(variants)) != len(variants):
        raise SystemExit("variants must be non-empty and unique")
    if str(args.model_kind) == "v3" and set(variants) != {"E1A_EXACT_ONPOLICY_V3_LEGACY"}:
        raise SystemExit("V3 formal replay only permits E1A_EXACT_ONPOLICY_V3_LEGACY")
    if str(args.model_kind) == "pctis" and not set(variants).issubset({"E1B_PCTIS_LEGACY", "E2_PCTIS_LIVE"}):
        raise SystemExit("PCTIS formal replay only permits E1B_PCTIS_LEGACY/E2_PCTIS_LIVE")
    if "E2_PCTIS_LIVE" in variants and not bool(args.enable_live):
        raise SystemExit("E2_PCTIS_LIVE requires --enable-live")
    model = args.model_checkpoint if args.model_checkpoint.is_absolute() else ROOT / args.model_checkpoint
    if not model.is_file():
        raise SystemExit(f"missing V3 checkpoint: {model}")
    events = frozen_events()
    if args.event_id:
        wanted = {str(value) for value in args.event_id}
        available = {str(item["event_id"]) for item in events}
        missing = sorted(wanted - available)
        if missing:
            raise SystemExit(f"requested IDs are not frozen N72R9 events: {missing}")
        events = [item for item in events if str(item["event_id"]) in wanted]
    output_root = args.output_root if args.output_root.is_absolute() else ROOT / args.output_root
    output_root.mkdir(parents=True, exist_ok=True)
    manifest_path = output_root / f"exact_v3_replay_manifest_attempt_{int(args.attempt):02d}.json"
    records = {
        str(item["event_id"]): {
            "event_id": str(item["event_id"]),
            "sequence": str(item["sequence"]),
            "action_type": str(item["action_type"]),
            "status": "NOT_RUN",
            "attempt": int(args.attempt),
            "returncode": None,
            "log": None,
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
                "schema_version": "N72R11R4_EXACT_V3_REPLAY_MANIFEST_V1",
                "status": status,
                "created_at_utc": now_utc(),
                "protocol": str(PROTOCOL),
                "protocol_sha256": sha256_file(PROTOCOL),
                "model_checkpoint": str(model),
                "model_checkpoint_sha256": sha256_file(model),
                "attempt": int(args.attempt),
                "device": str(args.device),
                "horizon": int(args.horizon),
                "variants": ["E0_BASELINE_B0", *variants],
                "model_kind": str(args.model_kind),
                "runtime_future_gt_used": False,
                "interaction_source": "simulated_from_gt",
                "not_real_human_evidence": True,
                "resource_censored_development": bool(args.resource_censored),
                "execution": "one blocking child per event; no concurrent GPU ownership",
                "event_count": len(events),
                "records": [records[event_id] for event_id in sorted(records)],
                "counts": counts,
            },
        )

    write_manifest("RUNNING")
    for event in events:
        event_id = str(event["event_id"])
        log_path = output_root / "logs" / f"{event_id}.attempt{int(args.attempt):02d}.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        command = child_command(event_id, output_root, model, str(args.device), int(args.horizon), variants, str(args.model_kind), bool(args.enable_live or "E2_PCTIS_LIVE" in variants))
        env = os.environ.copy()
        env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
        env["PYTHONUNBUFFERED"] = "1"
        records[event_id].update({"status": "RUNNING", "log": str(log_path), "command": command, "started_at_utc": now_utc()})
        write_manifest("RUNNING")
        with log_path.open("wb") as log_handle:
            completed = subprocess.run(command, cwd=str(ROOT), env=env, stdout=log_handle, stderr=subprocess.STDOUT, check=False)
        done = output_root / event_id / "done.json"
        done_exists = done.is_file()
        records[event_id].update(
            {
                "status": "PASS" if completed.returncode == 0 and done_exists else "FAIL_CHILD",
                "returncode": int(completed.returncode),
                "done": str(done) if done_exists else None,
                "done_sha256": sha256_file(done) if done_exists else None,
                "finished_at_utc": now_utc(),
            }
        )
        write_manifest("RUNNING")
    final_status = "PASS_ALL_SELECTED" if all(record["status"] == "PASS" for record in records.values()) else "PARTIAL_WITH_FAILURES"
    write_manifest(final_status)
    counts = {key: sum(record["status"] == key for record in records.values()) for key in sorted({str(record["status"]) for record in records.values()})}
    print(json.dumps({"status": final_status, "manifest": str(manifest_path), "counts": counts}, sort_keys=True))
    return 0 if final_status == "PASS_ALL_SELECTED" else 1


if __name__ == "__main__":
    raise SystemExit(main())
