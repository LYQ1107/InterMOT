#!/usr/bin/env python3
"""Preserve the first interrupted/partial N72R12 Oracle attempt as evidence."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import tempfile


def main() -> int:
    root = Path(__file__).resolve().parents[1] / "outputs/N72R12/oracle"
    events = sorted(path.name for path in (root / "events").glob("*.json")) if (root / "events").is_dir() else []
    manifest_path = root / "oracle_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.is_file() else {}
    complete = manifest.get("status") == "PASS_POSTHOC_ORACLE_HEADROOM" and manifest.get("event_count") == 32
    payload = {
        "schema_version": "N72R12_ORACLE_ATTEMPT_FAILURE_V1",
        "status": "PASS_POSTHOC_ORACLE_LATE_COMPLETION_OBSERVED" if complete else "FAIL_POSTHOC_ORACLE_PARTIAL_ATTEMPT",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "attempt": 1,
        "observed_event_artifact_count": len(events),
        "expected_event_artifact_count": 32,
        "missing_event_artifact_count": max(32 - len(events), 0),
        "observed_event_artifacts": events,
        "reason": "the initial tool observation returned before the complete manifest was visible; the same posthoc command later produced a complete 32-event manifest",
        "runtime_eligible": False,
        "scientific_runtime_result": False,
        "oracle_upper_bound_only": True,
        "historical_outputs_modified": False,
    }
    output = root / "attempt_01_partial_failure.json"
    root.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{output.name}.", suffix=".tmp", dir=str(root))
    try:
        with open(fd, "w", encoding="utf-8", closefd=True) as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
        Path(temporary).replace(output)
    finally:
        if Path(temporary).exists():
            Path(temporary).unlink()
    print(json.dumps({"status": payload["status"], "events": len(events), "output": str(output)}, sort_keys=True))
    return 0 if complete else 1


if __name__ == "__main__":
    raise SystemExit(main())
