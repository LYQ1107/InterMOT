#!/usr/bin/env python3
"""Run each learned N72R7 development event in an independent process."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.n72r7_dev_replay import atomic_json, atomic_write, read_json, TARGET_MANIFEST  # noqa: E402


def resolve_root_path(value: str | None, default: Path) -> Path:
    if value is None:
        return default
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


TRAINING_ROOT = resolve_root_path(
    os.environ.get("N72R7_TRAINING_ROOT"),
    ROOT / "outputs/N72R7/training",
)
CHECKPOINT = resolve_root_path(
    os.environ.get("N72R7_LEARNED_CHECKPOINT"),
    TRAINING_ROOT / "HumanConditionedTargetIDDecoder_v1.pt",
)
REPLAY_PROTOCOL = resolve_root_path(
    os.environ.get("N72R7_LEARNED_REPLAY_PROTOCOL"),
    TRAINING_ROOT / "learned_replay_protocol.json",
)


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_replay_protocol() -> dict[str, Any]:
    if REPLAY_PROTOCOL.is_file():
        existing = read_json(REPLAY_PROTOCOL)
        expected_checkpoint_sha256 = hashlib.sha256(CHECKPOINT.read_bytes()).hexdigest()
        if existing.get("checkpoint_sha256") != expected_checkpoint_sha256:
            raise RuntimeError("existing replay protocol belongs to a different checkpoint")
        if existing.get("runtime_future_gt_used") is not False or existing.get("posthoc_gt_used") is not False:
            raise RuntimeError("existing replay protocol has invalid GT provenance flags")
        return existing
    payload: dict[str, Any] = {
        "schema_version": "N72R7_LEARNED_DECODER_REPLAY_PROTOCOL_V1",
        "created_at_utc": now_utc(),
        "training_manifest": str(TRAINING_ROOT / "decoder_training_manifest.json"),
        "training_protocol": str(TRAINING_ROOT / "training_protocol.json"),
        "checkpoint": str(CHECKPOINT),
        "checkpoint_sha256": hashlib.sha256(CHECKPOINT.read_bytes()).hexdigest(),
        "model": "HumanConditionedTargetIDDecoder",
        "variants": {"D1_R2": "frozen B0 pool plus learned target/NONE decoder", "D2_R2": "frozen B0 plus target-session pool plus learned target/NONE decoder"},
        "none_rule": "candidate logits must beat NONE logit; no post-treatment threshold tuning",
        "admission_score": 0.5,
        "admission_margin": 0.2,
        "admission_rule": "fixed protocol constants, not validation/future-effect selected",
        "selector_mode": os.environ.get("N72R7_SELECTOR_MODE", "greedy"),
        "solver": "sam3_intermot.association.effect_assignment.solve_effect_assignment",
        "candidate_generator_frozen": True,
        "sam3_checkpoint_modified": False,
        "runtime_future_gt_used": False,
        "posthoc_gt_used": False,
        "interaction_source": "simulated_from_gt",
        "not_real_human_evidence": True,
    }
    payload["protocol_sha256"] = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")).hexdigest()
    atomic_json(REPLAY_PROTOCOL, payload)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=("D1", "D2"), required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--event-id", action="append", default=[])
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--selector-mode", choices=("greedy", "beam", "concept"), default="greedy")
    args = parser.parse_args()
    if not CHECKPOINT.is_file():
        raise FileNotFoundError(CHECKPOINT)
    protocol = write_replay_protocol()
    output_root = Path(args.output_root)
    if not output_root.is_absolute():
        output_root = ROOT / output_root
    output_root.mkdir(parents=True, exist_ok=True)
    target_manifest = read_json(TARGET_MANIFEST)
    all_event_ids = sorted(str(item["event_id"]) for item in target_manifest.get("selected", []))
    if len(all_event_ids) != 32 or len(all_event_ids) != len(set(all_event_ids)):
        raise RuntimeError("frozen N72R6 learned replay event set must contain 32 unique events")
    event_ids = sorted(args.event_id) if args.event_id else list(all_event_ids)
    if args.limit is not None:
        if args.limit <= 0:
            raise ValueError("--limit must be positive")
        event_ids = event_ids[: args.limit]
    if not event_ids or not set(event_ids).issubset(set(all_event_ids)) or len(event_ids) != len(set(event_ids)):
        raise RuntimeError("requested learned replay event IDs are not a unique subset of the frozen 32")
    logs = output_root / "process_logs"
    records: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for index, event_id in enumerate(event_ids):
        command = [
            sys.executable,
            str(ROOT / "scripts/n72r7_learned_replay.py"),
            "--variant", args.variant,
            "--event-id", event_id,
            "--output-root", str(output_root),
            "--checkpoint", str(CHECKPOINT),
            "--replay-protocol", str(REPLAY_PROTOCOL),
            "--selector-mode", args.selector_mode,
        ]
        completed = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, check=False)
        stdout_path = logs / f"{index:03d}.{event_id}.{args.variant}.stdout.log"
        stderr_path = logs / f"{index:03d}.{event_id}.{args.variant}.stderr.log"
        atomic_write(stdout_path, completed.stdout)
        atomic_write(stderr_path, completed.stderr)
        manifest_path = output_root / event_id / "event_manifest.json"
        if completed.returncode == 0 and manifest_path.is_file():
            artifact = read_json(manifest_path)
            record = {"event_id": event_id, "variant": args.variant, "selector_mode": args.selector_mode, "status": artifact.get("status"), "event_manifest": str(manifest_path), "stdout": str(stdout_path), "stderr": str(stderr_path), "returncode": int(completed.returncode), "independent_process": True}
            records.append(record)
            print(json.dumps({"event_id": event_id, "variant": args.variant, "status": record["status"]}, sort_keys=True))
        else:
            failure = {"event_id": event_id, "variant": args.variant, "selector_mode": args.selector_mode, "status": "FAIL", "returncode": int(completed.returncode), "stdout": str(stdout_path), "stderr": str(stderr_path), "child_output": completed.stdout[-4000:], "child_error": completed.stderr[-4000:], "independent_process": True}
            failures.append(failure)
            print(json.dumps(failure, ensure_ascii=False, sort_keys=True))
    keys = [(str(item["event_id"]), str(item["variant"])) for item in records + failures]
    duplicates = sorted({key for key in keys if keys.count(key) > 1})
    observed = {key[0] for key in keys}
    batch = {
        "schema_version": "N72R7_LEARNED_DECODER_BATCH_V1",
        "status": (
            "PASS_N72R7_LEARNED_DECODER_BATCH"
            if len(records) == len(event_ids) and not failures and not duplicates
            else "PARTIAL_N72R7_LEARNED_DECODER_BATCH"
        ),
        "variant": args.variant,
        "selector_mode": args.selector_mode,
        "run_kind": "full" if len(event_ids) == 32 else "smoke",
        "requested_event_count": len(event_ids),
        "completed_event_count": len(records),
        "failed_event_count": len(failures),
        "missing_event_ids": sorted(set(event_ids) - observed),
        "unexpected_event_ids": sorted(observed - set(event_ids)),
        "duplicate_event_keys": [list(key) for key in duplicates],
        "results": records,
        "failures": failures,
        "replay_protocol": str(REPLAY_PROTOCOL),
        "replay_protocol_sha256": protocol["protocol_sha256"],
        "checkpoint_sha256": protocol["checkpoint_sha256"],
        "independent_process_per_event": True,
        "max_concurrent_processes": 1,
        "runtime_future_gt_used": False,
        "posthoc_gt_used": False,
        "interaction_source": "simulated_from_gt",
        "not_real_human_evidence": True,
        "created_at_utc": now_utc(),
    }
    atomic_json(output_root / "batch_manifest.json", batch)
    print(json.dumps({"status": batch["status"], "completed": len(records), "failed": len(failures)}, sort_keys=True))
    return 0 if batch["status"] == "PASS_N72R7_LEARNED_DECODER_BATCH" else 1


if __name__ == "__main__":
    raise SystemExit(main())
