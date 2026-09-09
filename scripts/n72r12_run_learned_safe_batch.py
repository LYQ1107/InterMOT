#!/usr/bin/env python3
"""Run the single fixed N72R12 E1D learned-gate replay for all 32 events.

The E0/E1B/E1C sealed artifacts are immutable references.  Each E1D event is
run in its own subprocess and writes to a fresh N72R12 output root.  This
batch never chooses events or checkpoints from future metrics.
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
from typing import Any, Mapping

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


def _events() -> list[dict[str, Any]]:
    protocol_path = ROOT / "outputs/N72R9/protocol.json"
    if _sha(protocol_path) != EXPECTED_PROTOCOL_SHA:
        raise RuntimeError("frozen N72R9 protocol SHA-256 changed")
    protocol = _read(protocol_path)
    values = protocol.get("source_event_selection", {}).get("events")
    if not isinstance(values, list) or len(values) != EXPECTED_EVENTS:
        raise RuntimeError("frozen protocol does not contain exactly 32 events")
    result = [dict(item) for item in values]
    if len({str(item["event_id"]) for item in result}) != EXPECTED_EVENTS:
        raise RuntimeError("frozen protocol event IDs are not unique")
    return sorted(result, key=lambda item: str(item["event_id"]))


def _reference_records() -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    r5_path = ROOT / "outputs/N72R11R5R1/formal_e1b_manifest.json"
    safe_path = ROOT / "outputs/N72R12/formal_safe/formal_safe_manifest.json"
    r5 = _read(r5_path)
    safe = _read(safe_path)
    if r5.get("status") != "PASS_ALL_SELECTED" or r5.get("event_count") != EXPECTED_EVENTS:
        raise RuntimeError("R5R1 E1B reference is not a complete PASS")
    if safe.get("status") != "PASS_N72R12_FORMAL_RUNTIME" or safe.get("event_count") != EXPECTED_EVENTS:
        raise RuntimeError("N72R12 E1C reference is not a complete PASS")
    if r5.get("protocol_sha256") != EXPECTED_PROTOCOL_SHA or safe.get("protocol_sha256") != EXPECTED_PROTOCOL_SHA:
        raise RuntimeError("reference protocol hash mismatch")
    r5_rows = {str(item["event_id"]): dict(item) for item in r5.get("records", [])}
    safe_rows = {str(item["event_id"]): dict(item) for item in safe.get("records", [])}
    if len(r5_rows) != EXPECTED_EVENTS or len(safe_rows) != EXPECTED_EVENTS:
        raise RuntimeError("reference event IDs are incomplete or duplicated")
    return r5_rows, safe_rows


def _valid_e1d(done_path: Path, safe_gate_sha256: str) -> bool:
    if not done_path.is_file():
        return False
    try:
        done = _read(done_path)
        seal_path = Path(str(done["runtime_event_sealed"]))
        seal = _read(seal_path)
    except (KeyError, OSError, TypeError, ValueError):
        return False
    if (
        done.get("status") != "PASS_N72R11_RUNTIME_AND_POSTHOC_EVENT"
        or "E1D_PCTIS_LEARNED_SAFE" not in done.get("treatment_variants", [])
        or done.get("runtime_future_gt_used") is not False
        or done.get("runtime_gt_read") is not False
        or seal.get("runtime_future_gt_used") is not False
        or seal.get("runtime_gt_read") is not False
    ):
        return False
    frames = done_path.parent / "E1D_PCTIS_LEARNED_SAFE/runtime_frames.jsonl"
    if not frames.is_file():
        return False
    try:
        rows = [json.loads(line) for line in frames.read_text(encoding="utf-8").splitlines() if line.strip()]
    except (OSError, ValueError):
        return False
    if len(rows) != HORIZON + 1 or any(row.get("runtime_future_gt_used") is not False for row in rows):
        return False
    for row in rows[1:]:
        gate = row.get("counterfactual_intervention", {}).get("learned_gate")
        if not isinstance(gate, Mapping) or gate.get("checkpoint_sha256") != safe_gate_sha256:
            return False
    return True


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=ROOT / "outputs/N72R12/formal_learned")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--max-events", type=int, default=EXPECTED_EVENTS)
    args = parser.parse_args()
    output_root = args.output_root.resolve()
    if int(args.max_events) != EXPECTED_EVENTS:
        raise RuntimeError("N72R12 learned formal batch cannot be shortened")
    events = _events()
    r5_rows, safe_rows = _reference_records()
    pctis = ROOT / "outputs/N72R11R4/pctis_onpolicy_finetune/pctis_onpolicy_finetuned.pt"
    safe_gate = ROOT / "outputs/N72R12/gate_training/safe_gate.pt"
    if _sha(pctis) != EXPECTED_CHECKPOINT_SHA:
        raise RuntimeError("frozen PCTIS checkpoint SHA-256 changed")
    if not safe_gate.is_file():
        raise FileNotFoundError(safe_gate)
    safe_gate_sha256 = _sha(safe_gate)
    output_root.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    for index, event in enumerate(events, 1):
        event_id = str(event["event_id"])
        event_dir = output_root / event_id
        done = event_dir / "done.json"
        started = datetime.now(timezone.utc).isoformat()
        base_record = {
            "event_id": event_id,
            "sequence": str(event["sequence"]),
            "action_type": str(event["action_type"]),
            "e1b_done": r5_rows[event_id].get("done"),
            "e1c_done": safe_rows[event_id].get("done"),
            "e1d_done": str(done),
            "started_at_utc": started,
        }
        if done.is_file():
            status = "PASS" if _valid_e1d(done, safe_gate_sha256) else "FAIL_EXISTING_ARTIFACT"
            records.append({**base_record, "status": status, "returncode": 0 if status == "PASS" else None, "reused": True, "finished_at_utc": datetime.now(timezone.utc).isoformat()})
            print(json.dumps({"index": index, "event_id": event_id, "status": status}, sort_keys=True), flush=True)
            continue
        log_path = output_root / "logs" / f"{event_id}.log"
        command = [
            str(args.python), "-u", str(ROOT / "scripts/n72r11_on_demand_replay.py"),
            "--event-id", event_id, "--output-root", str(output_root), "--device", str(args.device),
            "--model-checkpoint", str(pctis), "--scorer-kind", "pctis", "--horizon", str(HORIZON),
            "--variants", "E1D_PCTIS_LEARNED_SAFE", "--safe-gate-checkpoint", str(safe_gate),
            "--require-positive-geometry", "--rebuild-baseline-geometry",
        ]
        try:
            process = subprocess.run(
                command,
                cwd=ROOT,
                capture_output=True,
                text=True,
                env={**os.environ, "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"},
            )
            log_path.parent.mkdir(parents=True, exist_ok=True)
            log_path.write_text(process.stdout + process.stderr, encoding="utf-8")
            status = "PASS" if process.returncode == 0 and _valid_e1d(done, safe_gate_sha256) else "FAIL"
            record: dict[str, Any] = {
                **base_record,
                "status": status,
                "returncode": int(process.returncode),
                "command": command,
                "log": str(log_path),
                "reused": False,
                "finished_at_utc": datetime.now(timezone.utc).isoformat(),
            }
            if status != "PASS":
                record["stdout_tail"] = process.stdout[-4000:]
                record["stderr_tail"] = process.stderr[-4000:]
        except Exception as exc:
            log_path.parent.mkdir(parents=True, exist_ok=True)
            record = {
                **base_record,
                "status": "FAIL_BATCH_EXCEPTION",
                "returncode": None,
                "command": command,
                "log": str(log_path),
                "error_type": type(exc).__name__,
                "error": str(exc),
                "traceback": traceback.format_exc(),
                "reused": False,
                "finished_at_utc": datetime.now(timezone.utc).isoformat(),
            }
        records.append(record)
        print(json.dumps({"index": index, "event_id": event_id, "status": record["status"]}, sort_keys=True), flush=True)
    seen = [str(item.get("event_id")) for item in records]
    expected_ids = {str(event["event_id"]) for event in events}
    failures = [item for item in records if item.get("status") != "PASS"]
    integrity = {
        "expected_event_count": EXPECTED_EVENTS,
        "record_count": len(records),
        "unique_event_count": len(set(seen)),
        "duplicate_event_ids": sorted({value for value in seen if seen.count(value) > 1}),
        "missing_event_ids": sorted(expected_ids - set(seen)),
        "failed_event_ids": sorted(str(item.get("event_id")) for item in failures),
    }
    payload = {
        "schema_version": "N72R12_FORMAL_LEARNED_SAFE_RUNTIME_MANIFEST_V1",
        "status": "PASS_N72R12_LEARNED_RUNTIME" if not failures and len(records) == EXPECTED_EVENTS and len(set(seen)) == EXPECTED_EVENTS else "FAIL_N72R12_LEARNED_RUNTIME",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "event_count": EXPECTED_EVENTS,
        "records": records,
        "integrity": integrity,
        "variants": ["E0_BASELINE_B0", "E1B_PCTIS_LEGACY", "E1C_PCTIS_SAFE", "E1D_PCTIS_LEARNED_SAFE"],
        "executed_variant": "E1D_PCTIS_LEARNED_SAFE",
        "horizon": HORIZON,
        "protocol": str(ROOT / "outputs/N72R9/protocol.json"),
        "protocol_sha256": EXPECTED_PROTOCOL_SHA,
        "model_checkpoint": str(pctis),
        "model_checkpoint_sha256": EXPECTED_CHECKPOINT_SHA,
        "safe_gate_checkpoint": str(safe_gate),
        "safe_gate_checkpoint_sha256": safe_gate_sha256,
        "r5r1_e1b_manifest": str(ROOT / "outputs/N72R11R5R1/formal_e1b_manifest.json"),
        "r12_e1c_manifest": str(ROOT / "outputs/N72R12/formal_safe/formal_safe_manifest.json"),
        "runtime_future_gt_used": False,
        "runtime_gt_read": False,
        "posthoc_gt_used": True,
        "interaction_source": "simulated_from_gt",
        "not_real_human_evidence": True,
        "require_positive_geometry": True,
        "baseline_regenerated_from_frozen_sources": True,
        "resource_policy": {"one_sequence_per_gpu": True, "device": str(args.device), "allocator": "expandable_segments:True"},
        "historical_outputs_modified": False,
    }
    _atomic_json(output_root / "formal_learned_manifest.json", payload)
    print(json.dumps({"status": payload["status"], "events": len(records), "failures": len(failures), "output": str(output_root / "formal_learned_manifest.json")}, sort_keys=True))
    return 0 if payload["status"] == "PASS_N72R12_LEARNED_RUNTIME" else 1


if __name__ == "__main__":
    raise SystemExit(main())
