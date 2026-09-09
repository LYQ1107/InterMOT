#!/usr/bin/env python3
"""Run the frozen 32-event E1C deterministic CSI replay batch.

E0/E1B are read-only references to the corrected N72R11R5R1 artifacts.  Only
the new E1C runtime is executed here, with a fresh event directory and a
positive-geometry baseline regeneration from the same frozen sources.
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
import traceback
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

HORIZON = 100
EXPECTED_EVENTS = 32
EXPECTED_PROTOCOL_SHA = "e012ecc3bd64fec4409fccd57d920f3690b8b1c4b5a2dd557e03e2bfb43ef0e9"
EXPECTED_CHECKPOINT_SHA = "77b41dbe2fc0f03c48ca4b8cda22e6b219eb58e266566eb96fcf0df3fb8c2c16"


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object: {path}")
    return value


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _protocol_events() -> list[dict[str, Any]]:
    protocol_path = ROOT / "outputs/N72R9/protocol.json"
    if _sha(protocol_path) != EXPECTED_PROTOCOL_SHA:
        raise RuntimeError("frozen N72R9 protocol SHA-256 changed")
    protocol = _read(protocol_path)
    events = protocol.get("source_event_selection", {}).get("events")
    if not isinstance(events, list) or len(events) != EXPECTED_EVENTS:
        raise RuntimeError("frozen protocol does not contain exactly 32 events")
    return sorted([dict(item) for item in events], key=lambda item: str(item["event_id"]))


def _r5r1_records() -> dict[str, dict[str, Any]]:
    manifest_path = ROOT / "outputs/N72R11R5R1/formal_e1b_manifest.json"
    manifest = _read(manifest_path)
    if manifest.get("status") != "PASS_ALL_SELECTED" or manifest.get("event_count") != EXPECTED_EVENTS:
        raise RuntimeError("corrected R5R1 E1B manifest is not a complete PASS")
    if manifest.get("protocol_sha256") != EXPECTED_PROTOCOL_SHA:
        raise RuntimeError("R5R1 manifest protocol hash does not match frozen protocol")
    records = manifest.get("records")
    if not isinstance(records, list) or len(records) != EXPECTED_EVENTS:
        raise RuntimeError("R5R1 manifest records are incomplete")
    result = {str(item["event_id"]): dict(item) for item in records}
    if len(result) != EXPECTED_EVENTS or any(item.get("status") != "PASS" for item in result.values()):
        raise RuntimeError("R5R1 manifest has duplicate or non-PASS records")
    return result


def _record_valid(record: dict[str, Any]) -> bool:
    done = record.get("done")
    if not done or not Path(str(done)).is_file():
        return False
    try:
        payload = _read(Path(str(done)))
    except (OSError, ValueError, TypeError):
        return False
    seal_path = payload.get("runtime_event_sealed")
    seal = None
    if seal_path and Path(str(seal_path)).is_file():
        try:
            seal = _read(Path(str(seal_path)))
        except (OSError, ValueError, TypeError):
            return False
    return bool(
        payload.get("status") == "PASS_N72R11_RUNTIME_AND_POSTHOC_EVENT"
        and "E1C_PCTIS_SAFE" in payload.get("treatment_variants", [])
        and payload.get("runtime_future_gt_used") is False
        # Older sealed N72R11 done files omitted this top-level field.  The
        # lossless seal and every runtime row must still prove it was false;
        # new done files written by the replay now include the field.
        and payload.get("runtime_gt_read") in (None, False)
        and isinstance(seal, dict)
        and seal.get("runtime_future_gt_used") is False
        and seal.get("runtime_gt_read") is False
        and payload.get("posthoc_gt_used") is True
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=ROOT / "outputs/N72R12/formal_safe")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--max-events", type=int, default=EXPECTED_EVENTS)
    args = parser.parse_args()
    output_root = args.output_root.resolve()
    if int(args.max_events) != EXPECTED_EVENTS:
        raise RuntimeError("N72R12 formal batch cannot be shortened")
    events = _protocol_events()
    r5r1 = _r5r1_records()
    prior_manifest_path = output_root / "formal_safe_manifest.json"
    if prior_manifest_path.is_file():
        prior_manifest = _read(prior_manifest_path)
        if prior_manifest.get("status") != "PASS_N72R12_FORMAL_RUNTIME":
            _atomic_json(output_root / "attempt_01_manifest_failure.json", prior_manifest)
    checkpoint = ROOT / "outputs/N72R11R4/pctis_onpolicy_finetune/pctis_onpolicy_finetuned.pt"
    if _sha(checkpoint) != EXPECTED_CHECKPOINT_SHA:
        raise RuntimeError("frozen PCTIS checkpoint SHA-256 changed")
    output_root.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    for index, event in enumerate(events, 1):
        event_id = str(event["event_id"])
        event_dir = output_root / event_id
        done = event_dir / "done.json"
        started = datetime.now(timezone.utc).isoformat()
        if done.is_file():
            existing = {"event_id": event_id, "done": str(done), "status": "PASS" if _record_valid({"done": str(done)}) else "FAIL_EXISTING_ARTIFACT", "reused": True}
            if existing["status"] == "PASS":
                records.append({**existing, "sequence": str(event["sequence"]), "action_type": str(event["action_type"]), "e1b_done": r5r1[event_id].get("done"), "started_at_utc": started, "finished_at_utc": datetime.now(timezone.utc).isoformat()})
                continue
            # Never overwrite a sealed invalid event.  A retry receives a
            # fresh attempt root selected by the caller.
            records.append({**existing, "sequence": str(event["sequence"]), "action_type": str(event["action_type"]), "e1b_done": r5r1[event_id].get("done"), "started_at_utc": started, "finished_at_utc": datetime.now(timezone.utc).isoformat()})
            continue
        log_dir = output_root / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / f"{event_id}.log"
        command = [
            str(args.python), "-u", str(ROOT / "scripts/n72r11_on_demand_replay.py"),
            "--event-id", event_id, "--output-root", str(output_root), "--device", str(args.device),
            "--model-checkpoint", str(checkpoint), "--scorer-kind", "pctis", "--horizon", str(HORIZON),
            "--variants", "E1C_PCTIS_SAFE", "--require-positive-geometry", "--rebuild-baseline-geometry",
        ]
        try:
            process = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, env={**os.environ, "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"})
            log_path.write_text(process.stdout + process.stderr, encoding="utf-8")
            status = "PASS" if process.returncode == 0 and _record_valid({"done": str(done)}) else "FAIL"
            record = {
                "event_id": event_id,
                "sequence": str(event["sequence"]),
                "action_type": str(event["action_type"]),
                "status": status,
                "returncode": int(process.returncode),
                "command": command,
                "log": str(log_path),
                "done": str(done) if done.is_file() else None,
                "e1b_done": r5r1[event_id].get("done"),
                "started_at_utc": started,
                "finished_at_utc": datetime.now(timezone.utc).isoformat(),
            }
            if status != "PASS":
                record["stdout_tail"] = process.stdout[-4000:]
                record["stderr_tail"] = process.stderr[-4000:]
        except Exception as exc:
            record = {
                "event_id": event_id,
                "sequence": str(event["sequence"]),
                "action_type": str(event["action_type"]),
                "status": "FAIL_BATCH_EXCEPTION",
                "returncode": None,
                "command": command,
                "log": str(log_path),
                "error_type": type(exc).__name__,
                "error": str(exc),
                "traceback": traceback.format_exc(),
                "e1b_done": r5r1[event_id].get("done"),
                "started_at_utc": started,
                "finished_at_utc": datetime.now(timezone.utc).isoformat(),
            }
        records.append(record)
        print(json.dumps({"index": index, "event_id": event_id, "status": record["status"]}, sort_keys=True), flush=True)
    seen = [str(item.get("event_id")) for item in records]
    failures = [item for item in records if item.get("status") != "PASS"]
    expected_ids = {str(event["event_id"]) for event in events}
    integrity = {
        "expected_event_count": EXPECTED_EVENTS,
        "record_count": len(records),
        "unique_event_count": len(set(seen)),
        "duplicate_event_ids": sorted({event_id for event_id in seen if seen.count(event_id) > 1}),
        "missing_event_ids": sorted(expected_ids - set(seen)),
        "failed_event_ids": sorted(str(item.get("event_id")) for item in failures),
    }
    payload = {
        "schema_version": "N72R12_FORMAL_SAFE_RUNTIME_MANIFEST_V1",
        "status": "PASS_N72R12_FORMAL_RUNTIME" if not failures and len(records) == EXPECTED_EVENTS and len(set(seen)) == EXPECTED_EVENTS else "FAIL_N72R12_FORMAL_RUNTIME",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "event_count": EXPECTED_EVENTS,
        "records": records,
        "integrity": integrity,
        "variants": ["E0_BASELINE_B0", "E1B_PCTIS_LEGACY", "E1C_PCTIS_SAFE"],
        "horizon": HORIZON,
        "protocol": str(ROOT / "outputs/N72R9/protocol.json"),
        "protocol_sha256": EXPECTED_PROTOCOL_SHA,
        "model_checkpoint": str(checkpoint),
        "model_checkpoint_sha256": EXPECTED_CHECKPOINT_SHA,
        "r5r1_e1b_manifest": str(ROOT / "outputs/N72R11R5R1/formal_e1b_manifest.json"),
        "r5r1_e1b_manifest_sha256": _sha(ROOT / "outputs/N72R11R5R1/formal_e1b_manifest.json"),
        "runtime_future_gt_used": False,
        "runtime_gt_read": False,
        "posthoc_gt_used": True,
        "interaction_source": "simulated_from_gt",
        "not_real_human_evidence": True,
        "require_positive_geometry": True,
        "baseline_regenerated_from_frozen_sources": True,
        "resource_policy": {"one_sequence_per_gpu": True, "device": str(args.device), "allocator": "expandable_segments:True"},
    }
    _atomic_json(output_root / "formal_safe_manifest.json", payload)
    print(json.dumps({"status": payload["status"], "events": len(records), "failures": len(failures), "output": str(output_root / "formal_safe_manifest.json")}, sort_keys=True))
    return 0 if payload["status"] == "PASS_N72R12_FORMAL_RUNTIME" else 1


if __name__ == "__main__":
    raise SystemExit(main())
