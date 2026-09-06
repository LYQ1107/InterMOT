#!/usr/bin/env python3
"""Derive the post-smoke N72R11R1 retry IDs from immutable evidence."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
OOM_MANIFEST = ROOT / "outputs/N72R11R1/oom_retry_manifest.json"
SMOKE_ROOT = ROOT / "outputs/N72R11R1/secondary_oom_smoke_attempt_04"
OUTPUT = ROOT / "outputs/N72R11R1/remaining_oom_event_ids.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
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


def main() -> int:
    oom = json.loads(OOM_MANIFEST.read_text(encoding="utf-8"))
    all_ids = [str(value) for value in oom.get("event_ids", [])]
    if len(all_ids) != 37 or len(all_ids) != len(set(all_ids)):
        raise RuntimeError(f"expected 37 unique frozen OOM IDs, found {len(all_ids)}")
    smoke_done_paths = sorted(SMOKE_ROOT.glob("*/done.json"))
    if len(smoke_done_paths) != 1:
        raise RuntimeError(f"expected exactly one smoke done artifact, found {len(smoke_done_paths)}")
    smoke_done_path = smoke_done_paths[0]
    smoke_done = json.loads(smoke_done_path.read_text(encoding="utf-8"))
    smoke_event_id = str(smoke_done.get("event_id"))
    if smoke_event_id not in all_ids:
        raise RuntimeError("smoke event is not a member of the frozen OOM manifest")
    if smoke_done.get("status") != "PASS_N72R11_SECONDARY_INTERACTION":
        raise RuntimeError("required full-window smoke is not PASS")
    if smoke_done.get("runtime_future_gt_used") is not False:
        raise RuntimeError("smoke artifact violates the runtime GT boundary")
    remaining = [event_id for event_id in all_ids if event_id != smoke_event_id]
    result = {
        "schema_version": "N72R11R1_REMAINING_OOM_EVENT_IDS_V1",
        "status": "RETRY_SET_FROZEN_AFTER_REQUIRED_SMOKE",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_oom_manifest": str(OOM_MANIFEST),
        "source_oom_manifest_sha256": sha256_file(OOM_MANIFEST),
        "smoke_done_artifact": str(smoke_done_path),
        "smoke_done_artifact_sha256": sha256_file(smoke_done_path),
        "smoke_event_id": smoke_event_id,
        "frozen_oom_count": len(all_ids),
        "smoke_pass_count": 1,
        "remaining_retry_count": len(remaining),
        "event_ids": remaining,
        "runtime_future_gt_used": False,
        "preserves_smoke_and_original_failure_evidence": True,
    }
    atomic_json(OUTPUT, result)
    print(json.dumps({"status": result["status"], "output": str(OUTPUT), "remaining_retry_count": len(remaining)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
