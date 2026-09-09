#!/usr/bin/env python3
"""CPU-only validator for the N72R12 E1D sealed runtime artifacts."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import traceback
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import n72r11_on_demand_replay as replay  # noqa: E402


MANIFEST = ROOT / "outputs/N72R12/formal_learned/formal_learned_manifest.json"
PROTOCOL = ROOT / "outputs/N72R9/protocol.json"
SAFE_GATE = ROOT / "outputs/N72R12/gate_training/safe_gate.pt"
OUTPUT = ROOT / "outputs/N72R12/stage_14_status.json"
EXPECTED_EVENTS = 32
EXPECTED_PROTOCOL_SHA = "e012ecc3bd64fec4409fccd57d920f3690b8b1c4b5a2dd557e03e2bfb43ef0e9"


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object: {path}")
    return value


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    try:
        with open(fd, "w", encoding="utf-8", closefd=True) as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
        Path(temporary).replace(path)
    finally:
        if Path(temporary).exists():
            Path(temporary).unlink()


def main() -> int:
    started = now_utc()
    try:
        manifest = read_json(MANIFEST)
        protocol = read_json(PROTOCOL)
        if sha256_file(PROTOCOL) != EXPECTED_PROTOCOL_SHA or manifest.get("protocol_sha256") != EXPECTED_PROTOCOL_SHA:
            raise RuntimeError("learned runtime protocol hash mismatch")
        if manifest.get("status") != "PASS_N72R12_LEARNED_RUNTIME" or manifest.get("event_count") != EXPECTED_EVENTS:
            raise RuntimeError("learned runtime manifest is not a complete PASS")
        safe_gate_sha256 = sha256_file(SAFE_GATE)
        records = manifest.get("records")
        if not isinstance(records, list) or len(records) != EXPECTED_EVENTS:
            raise RuntimeError("learned runtime records are incomplete")
        events = protocol.get("source_event_selection", {}).get("events", [])
        event_by_id = {str(event["event_id"]): dict(event) for event in events}
        if len(event_by_id) != EXPECTED_EVENTS:
            raise RuntimeError("frozen protocol event IDs are incomplete")
        seen: set[str] = set()
        checked = 0
        frames_checked = 0
        predicted_apply = 0
        committed_apply = 0
        checkpoint_hashes: set[str] = set()
        for record in records:
            event_id = str(record.get("event_id"))
            if event_id in seen or event_id not in event_by_id:
                raise RuntimeError(f"duplicate or unknown learned event: {event_id}")
            seen.add(event_id)
            if record.get("status") != "PASS":
                raise RuntimeError(f"learned event is not PASS: {event_id}")
            done_path = Path(str(record.get("e1d_done")))
            done = read_json(done_path)
            if done.get("status") != "PASS_N72R11_RUNTIME_AND_POSTHOC_EVENT" or "E1D_PCTIS_LEARNED_SAFE" not in done.get("treatment_variants", []):
                raise RuntimeError(f"invalid E1D done payload: {event_id}")
            inputs = replay._load_inputs(event_by_id[event_id], horizon=100)
            inputs = dict(inputs)
            inputs["action_type"] = str(event_by_id[event_id]["action_type"])
            inputs["safe_gate_checkpoint_sha256"] = safe_gate_sha256
            frames_path = done_path.parent / "E1D_PCTIS_LEARNED_SAFE/runtime_frames.jsonl"
            rows = [json.loads(line) for line in frames_path.read_text(encoding="utf-8").splitlines() if line.strip()]
            if len(rows) != 101 or [int(row.get("frame", -1)) for row in rows] != list(range(int(event_by_id[event_id]["event_frame"]), int(event_by_id[event_id]["event_frame"]) + 101)):
                raise RuntimeError(f"E1D frame axis is incomplete: {event_id}")
            replay._validate_runtime(rows, inputs, "E1D_PCTIS_LEARNED_SAFE")
            for row in rows:
                if row.get("runtime_future_gt_used") is not False or row.get("runtime_gt_read") is not False or row.get("posthoc_gt_used") is not False:
                    raise RuntimeError(f"runtime GT flag violation: {event_id}:{row.get('frame')}")
                gate = row.get("counterfactual_intervention", {}).get("learned_gate")
                if isinstance(gate, dict):
                    checkpoint_hashes.add(str(gate.get("checkpoint_sha256")))
                    predicted_apply += int(gate.get("decision") == "APPLY_PCTIS")
                committed_apply += int(row.get("counterfactual_intervention", {}).get("intervention_applied") is True)
            checked += 1
            frames_checked += len(rows)
        if seen != set(event_by_id):
            raise RuntimeError(f"learned event coverage mismatch: missing={sorted(set(event_by_id) - seen)}")
        if checkpoint_hashes != {safe_gate_sha256}:
            raise RuntimeError(f"learned gate checkpoint hash set mismatch: {sorted(checkpoint_hashes)}")
        payload = {
            "schema_version": "N72R12_LEARNED_RUNTIME_VALIDATOR_V1",
            "status": "PASS_N72R12_LEARNED_RUNTIME_VALIDATOR",
            "created_at_utc": now_utc(),
            "started_at_utc": started,
            "manifest": str(MANIFEST),
            "manifest_sha256": sha256_file(MANIFEST),
            "protocol": str(PROTOCOL),
            "protocol_sha256": EXPECTED_PROTOCOL_SHA,
            "checked_event_count": checked,
            "checked_frame_count": frames_checked,
            "duplicate_event_count": 0,
            "missing_event_count": 0,
            "unavailable_event_count": 0,
            "runtime_future_gt_used": False,
            "runtime_gt_read": False,
            "posthoc_gt_used": False,
            "safe_gate_checkpoint_sha256": safe_gate_sha256,
            "predicted_apply_count": predicted_apply,
            "committed_apply_count": committed_apply,
            "production_authorized": False,
            "historical_outputs_modified": False,
            "finished_at_utc": now_utc(),
        }
        atomic_json(OUTPUT, payload)
        print(json.dumps({"status": payload["status"], "checked": checked, "frames": frames_checked, "output": str(OUTPUT)}, sort_keys=True))
        return 0
    except Exception as exc:
        failure = {
            "schema_version": "N72R12_LEARNED_RUNTIME_VALIDATOR_FAILURE_V1",
            "status": "FAIL_N72R12_LEARNED_RUNTIME_VALIDATOR",
            "started_at_utc": started,
            "finished_at_utc": now_utc(),
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
            "historical_outputs_modified": False,
        }
        atomic_json(ROOT / "outputs/N72R12/learned_runtime_validator_failure.json", failure)
        print(json.dumps({"status": failure["status"], "error": str(exc)}, sort_keys=True), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
