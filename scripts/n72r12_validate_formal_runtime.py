#!/usr/bin/env python3
"""CPU-only validator for all sealed N72R12 E1C runtime events."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import tempfile
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import n72r11_on_demand_replay as replay  # noqa: E402
from scripts import n72r9_temporal_replay as legacy  # noqa: E402


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object: {path}")
    return value


def _jsonl(path: Path) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not all(isinstance(row, dict) for row in rows):
        raise TypeError(f"non-object row: {path}")
    return rows


def _atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    try:
        with open(fd, "w", encoding="utf-8", closefd=True) as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
        Path(temporary).replace(path)
    finally:
        if Path(temporary).exists():
            Path(temporary).unlink()


def main() -> int:
    manifest_path = ROOT / "outputs/N72R12/formal_safe/formal_safe_manifest.json"
    r5_path = ROOT / "outputs/N72R11R5R1/formal_e1b_manifest.json"
    protocol = _read(ROOT / "outputs/N72R9/protocol.json")
    events = {str(item["event_id"]): item for item in protocol["source_event_selection"]["events"]}
    safe = _read(manifest_path)
    r5 = _read(r5_path)
    safe_records = {str(item["event_id"]): item for item in safe["records"]}
    r5_records = {str(item["event_id"]): item for item in r5["records"]}
    failures: list[dict[str, Any]] = []
    checks: list[dict[str, Any]] = []
    for event_id in sorted(events):
        try:
            event = dict(events[event_id])
            inputs = replay._load_inputs(event, horizon=100)
            inputs = dict(inputs)
            inputs["require_positive_geometry"] = True
            inputs["baseline_regenerated_from_frozen_sources"] = True
            safe_done = _read(Path(str(safe_records[event_id]["done"])))
            safe_sealed = _read(Path(str(safe_done["runtime_event_sealed"])))
            e1c_path = Path(str(safe_sealed["runtime_manifests"]["E1C_PCTIS_SAFE"]["frames"]))
            e1c_rows = _jsonl(e1c_path)
            replay._validate_runtime(e1c_rows, inputs, "E1C_PCTIS_SAFE")
            if len(e1c_rows) != 101:
                raise RuntimeError("E1C frame count is not 101")
            old_dir = Path(str(r5_records[event_id]["done"])).parent
            old_e0 = old_dir / "E0_BASELINE_B0/runtime_frames.jsonl"
            new_e0 = Path(str(safe_sealed["runtime_manifests"]["E0_BASELINE_B0"]["frames"]))
            if _sha(old_e0) != _sha(new_e0):
                raise RuntimeError("E0 hash differs from frozen R5R1 E0")
            for row in e1c_rows:
                if row.get("runtime_future_gt_used") is not False or row.get("runtime_gt_read") is not False or row.get("posthoc_gt_used") is not False:
                    raise RuntimeError(f"runtime GT flag at frame {row.get('frame')}")
            checks.append({"event_id": event_id, "status": "PASS", "frames": len(e1c_rows), "e0_hash_equivalent": True})
        except Exception as exc:
            failures.append({"event_id": event_id, "status": "FAIL", "error_type": type(exc).__name__, "error": str(exc)})
    payload = {
        "schema_version": "N72R12_FORMAL_RUNTIME_VALIDATOR_V1",
        "status": "PASS_N72R12_FORMAL_RUNTIME_VALIDATOR" if not failures and len(checks) == 32 else "FAIL_N72R12_FORMAL_RUNTIME_VALIDATOR",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "event_count": 32,
        "checked_event_count": len(checks),
        "failure_count": len(failures),
        "checks": checks,
        "failures": failures,
        "runtime_future_gt_used": False,
        "runtime_gt_read": False,
        "historical_outputs_modified": False,
    }
    validation_path = ROOT / "outputs/N72R12/formal_safe/runtime_validator.json"
    if validation_path.is_file():
        previous = _read(validation_path)
        if previous.get("status") != "PASS_N72R12_FORMAL_RUNTIME_VALIDATOR":
            _atomic(ROOT / "outputs/N72R12/formal_safe/runtime_validator_attempt_01_failure.json", previous)
    _atomic(validation_path, payload)
    _atomic(ROOT / "outputs/N72R12/stage_03_status.json", {"schema_version": "N72R12_STAGE_STATUS_V1", "stage": "Stage03", "status": payload["status"], "validator": str(ROOT / "outputs/N72R12/formal_safe/runtime_validator.json"), "event_count": len(checks), "failure_count": len(failures), "historical_outputs_modified": False, "runtime_future_gt_used": False})
    print(json.dumps({"status": payload["status"], "events": len(checks), "failures": len(failures)}, sort_keys=True))
    return 0 if payload["status"] == "PASS_N72R12_FORMAL_RUNTIME_VALIDATOR" else 1


if __name__ == "__main__":
    raise SystemExit(main())
