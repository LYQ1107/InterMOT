#!/usr/bin/env python3
"""Generate one independent official SAM3 target-session stream for confirmation.

This worker deliberately reuses the N72R6 target-session implementation but
supplies a separately sealed simulated-human event authority.  It does not
make a B1 branch look successful and never reads future GT at runtime.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import traceback
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import n72r6_target_correction_stream as base  # noqa: E402


PROTOCOL = ROOT / "outputs/N72R7/confirmation/confirmation_protocol.json"
DEFAULT_OUTPUT_ROOT = ROOT / "outputs/N72R7/confirmation/target_stream"


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object: {path}")
    return value


def load_spec(event_id: str) -> tuple[dict[str, Any], dict[str, Any]]:
    protocol = read_json(PROTOCOL)
    if protocol.get("runtime_future_gt_used") is not False:
        raise RuntimeError("confirmation protocol causal/public authority flags are invalid")
    if any(item.get("public_id_inference") is not False for item in protocol.get("events", [])):
        raise RuntimeError("confirmation event public authority flags are invalid")
    spec = next((dict(item) for item in protocol.get("events", []) if str(item.get("event_id")) == event_id), None)
    if spec is None:
        raise KeyError(f"event is absent from frozen confirmation protocol: {event_id}")
    event_policy = read_json(base.EVENT_MANIFEST)
    event = next((dict(item) for item in event_policy.get("events", []) if str(item.get("event_id")) == event_id), None)
    if event is None:
        raise KeyError(f"event is absent from frozen event policy: {event_id}")
    event["n72r6_target_public_id"] = int(spec["target_public_id"])
    if str(event.get("interaction_source")) != "simulated_from_gt":
        raise RuntimeError("confirmation input is not explicitly simulated_from_gt")
    return spec, event


def run(event_id: str, *, attempt: int, device: str, output_root: Path, recovery_mode: bool) -> dict[str, Any]:
    spec, event = load_spec(event_id)
    stage08 = read_json(base.STAGE08_MANIFEST)
    stage_event = next((dict(item) for item in stage08.get("events", []) if str(item.get("event_id")) == event_id), None)
    if stage_event is None:
        raise RuntimeError(f"stage08 event is absent: {event_id}")
    branches = {str(item.get("branch")): dict(item) for item in stage_event.get("branches", [])}
    main_branch = branches.get("B0_NO_INTERVENTION")
    if main_branch is None:
        raise RuntimeError(f"frozen B0 branch is absent: {event_id}")
    original_eligible = base.eligible_event

    def explicit_eligible(_event_id: str):
        if str(_event_id) != event_id:
            raise KeyError(f"worker was given an unexpected event: {_event_id}")
        # base.run_event only consumes the event and B0 branch.  The empty
        # branch object intentionally prevents accidental use of old B1
        # precondition/target fields.
        return event, stage_event, {}, main_branch

    base.eligible_event = explicit_eligible
    try:
        done = base.run_event(
            event_id,
            attempt=int(attempt),
            device=str(device),
            output_root=output_root,
            recovery_mode=bool(recovery_mode),
        )
    finally:
        base.eligible_event = original_eligible
    protocol_hash = sha256_file(PROTOCOL)
    done_path = output_root / f"attempt_{int(attempt)}" / event_id / "done.json"
    sealed = read_json(done_path)
    sealed.update({
        "confirmation_protocol": str(PROTOCOL),
        "confirmation_protocol_sha256": protocol_hash,
        "confirmation_event_authority": spec["authority"],
        "confirmation_target_public_id": int(spec["target_public_id"]),
        "confirmation_other_public_id": spec.get("other_public_id"),
        "original_stage08_b1_status": spec["original_stage08_b1_status"],
        "public_id_from_gt_id": False,
        "public_id_from_raw_sam_id": False,
        "public_id_inference": False,
        "runtime_future_gt_used": False,
        "runtime_gt_read": False,
        "posthoc_gt_used": False,
        "interaction_source": "simulated_from_gt",
        "not_real_human_evidence": True,
    })
    atomic_json(done_path, sealed)
    return sealed


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--event-id", required=True)
    parser.add_argument("--attempt", type=int, default=1)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--recovery-mode", action="store_true")
    args = parser.parse_args()
    output_root = args.output_root if args.output_root.is_absolute() else ROOT / args.output_root
    try:
        result = run(
            str(args.event_id),
            attempt=int(args.attempt),
            device=str(args.device),
            output_root=output_root,
            recovery_mode=bool(args.recovery_mode),
        )
        print(json.dumps({"status": result["status"], "event_id": result["event_id"], "frame_count": result["frame_count"]}, sort_keys=True))
        return 0
    except Exception as exc:
        failure = output_root / "attempts" / f"{args.event_id}.attempt{int(args.attempt)}.failure.json"
        atomic_json(failure, {
            "schema_version": "N72R7_CONFIRMATION_TARGET_STREAM_FAILURE_V1",
            "status": "FAIL_N72R7_CONFIRMATION_TARGET_STREAM",
            "event_id": str(args.event_id),
            "attempt": int(args.attempt),
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
            "runtime_future_gt_used": False,
            "historical_outputs_modified": False,
            "created_at_utc": now_utc(),
        })
        print(json.dumps({"status": "FAIL_N72R7_CONFIRMATION_TARGET_STREAM", "event_id": str(args.event_id), "failure": str(failure)}, sort_keys=True))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
