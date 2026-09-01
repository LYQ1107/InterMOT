#!/usr/bin/env python3
"""Sequential N38R1 sidecar supervisor.

The supervisor launches one fresh Python process per event×variant and writes
the manifest only after each child has finished.  Existing artifacts are never
overwritten; ``--resume`` may reuse an existing PASS artifact.  Smoke selection
is deterministic and uses only the frozen event manifest (the first manifest
event, first ADD event, and first ATOMIC swap event, with deterministic fill).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import traceback
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.n36_real_eval_common import atomic_json  # noqa: E402
from scripts.n38r1_sidecar_common import (  # noqa: E402
    N37_MANIFEST,
    N38R1_PROTOCOL,
    VARIANTS,
    event_id_of,
    load_manifest_item,
    protocol_hash,
)


N38R1 = ROOT / "outputs" / "n38r1"
DEFAULT_SIDECAR = N38R1 / "sidecar"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_events(manifest_path: Path) -> list[dict[str, Any]]:
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if payload.get("status") != "PASS":
        raise RuntimeError(f"frozen N37 manifest is not PASS: {payload.get('status')}")
    events = payload.get("events")
    if not isinstance(events, list) or len(events) != 24 or payload.get("event_count") != 24:
        raise RuntimeError("N38R1 requires exactly the frozen 24 N37 events")
    ids = [event_id_of(item) for item in events]
    if len(ids) != len(set(ids)):
        raise RuntimeError("duplicate event_id in frozen N37 manifest")
    return events


def smoke_event_ids(events: list[dict[str, Any]]) -> tuple[list[str], str]:
    selected: list[str] = []
    if events:
        selected.append(event_id_of(events[0]))
    for action in ("ADD_NEW_IDENTITY", "ATOMIC_ID_SWAP"):
        for item in events:
            if str(item["event"].get("action_type")) == action:
                selected.append(event_id_of(item))
                break
    for item in events:
        if len(selected) >= 3:
            break
        selected.append(event_id_of(item))
    selected = list(dict.fromkeys(selected))[:3]
    if len(selected) != 3:
        raise RuntimeError(f"deterministic smoke could not select three events: {selected}")
    return selected, (
        "manifest order first event + first ADD_NEW_IDENTITY + first ATOMIC_ID_SWAP; "
        "deterministic fill only, no replay/post-treatment fields"
    )


def expected_pairs(events: list[dict[str, Any]], event_ids: list[str]) -> list[tuple[str, str]]:
    allowed = set(event_ids)
    return [
        (event_id_of(item), variant)
        for item in events
        if event_id_of(item) in allowed
        for variant in VARIANTS
    ]


def artifact_record(path: Path, event_id: str, variant: str, process: dict[str, Any]) -> dict[str, Any]:
    status = "MISSING"
    payload_status = None
    runtime_future_gt_used = None
    event_frame_included = None
    event_plus_one_included = None
    error = None
    if path.is_file():
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            payload_status = payload.get("status")
            status = str(payload_status or "INVALID")
            runtime_future_gt_used = payload.get("runtime_future_gt_used")
            event_audit = payload.get("event_frame_audit") or {}
            event_frame_included = bool(event_audit.get("is_event_frame") is True)
            future = payload.get("branches", {}).get("memory_write=False", {}).get("future_trace", [])
            event_plus_one_included = bool(future and int(future[0].get("frame")) == int(payload.get("event_frame")) + 1)
            if payload_status != "PASS":
                error = payload.get("error")
        except Exception as exc:
            status = "INVALID"
            error = f"{type(exc).__name__}: {exc}"
    record = {
        "event_id": event_id,
        "variant": variant,
        "key": f"{event_id}:{variant}",
        "path": str(path.resolve().relative_to(ROOT)),
        "status": status,
        "payload_status": payload_status,
        "runtime_future_gt_used": runtime_future_gt_used,
        "event_frame_included": event_frame_included,
        "event_plus_one_included": event_plus_one_included,
        "process": process,
        "error": error,
    }
    if path.is_file():
        record["bytes"] = path.stat().st_size
        record["sha256"] = sha256(path)
    return record


def run_scope(
    manifest_path: Path,
    sidecar_root: Path,
    event_ids: list[str],
    *,
    scope: str,
    smoke_selection_rule: str | None,
    resume: bool,
    python_bin: str,
) -> dict[str, Any]:
    events = load_events(manifest_path)
    pairs = expected_pairs(events, event_ids)
    sidecar_root.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    for event_id, variant in pairs:
        output = sidecar_root / event_id / f"{variant}.json"
        process: dict[str, Any]
        if output.exists():
            if not resume:
                process = {
                    "status": "NOT_RUN_EXISTING_ARTIFACT",
                    "returncode": None,
                    "command": None,
                }
                records.append(artifact_record(output, event_id, variant, process))
                continue
            try:
                existing = json.loads(output.read_text(encoding="utf-8"))
                if existing.get("status") == "PASS":
                    process = {
                        "status": "SKIPPED_EXISTING_PASS",
                        "returncode": 0,
                        "command": None,
                    }
                    records.append(artifact_record(output, event_id, variant, process))
                    continue
            except Exception:
                pass
            process = {
                "status": "NOT_RUN_EXISTING_FAILURE",
                "returncode": None,
                "command": None,
            }
            records.append(artifact_record(output, event_id, variant, process))
            continue
        command = [
            python_bin,
            str(ROOT / "scripts" / "n38r1_sidecar_worker.py"),
            "--manifest",
            str(manifest_path),
            "--event-id",
            event_id,
            "--variant",
            variant,
            "--output",
            str(output),
        ]
        env = os.environ.copy()
        env["PYTHONPATH"] = str(ROOT) + os.pathsep + env.get("PYTHONPATH", "")
        env["PYTHONHASHSEED"] = "0"
        child = subprocess.run(
            command,
            cwd=str(ROOT),
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        process = {
            "status": "PASS" if child.returncode == 0 else "FAIL",
            "returncode": int(child.returncode),
            "command": command,
            "stdout_tail": child.stdout[-4000:],
            "stderr_tail": child.stderr[-8000:],
        }
        records.append(artifact_record(output, event_id, variant, process))

    expected_keys = {f"{event_id}:{variant}" for event_id, variant in pairs}
    seen = [record["key"] for record in records]
    counts = {
        "expected_policy_rows": len(expected_keys),
        "observed_records": len(seen),
        "unique_policy_rows": len(set(seen)),
        "duplicate_policy_rows": len(seen) - len(set(seen)),
        "missing_policy_rows": len(expected_keys - set(seen)),
        "pass_policy_rows": sum(record.get("status") == "PASS" for record in records),
        "fail_policy_rows": sum(record.get("status") == "FAIL" for record in records),
        "not_run_policy_rows": sum(str(record.get("status", "")).startswith("NOT_RUN") for record in records),
        "runtime_future_gt_true_rows": sum(record.get("runtime_future_gt_used") is True for record in records),
        "event_frame_missing_rows": sum(record.get("event_frame_included") is not True for record in records),
        "event_plus_one_missing_rows": sum(record.get("event_plus_one_included") is not True for record in records),
    }
    complete = (
        counts["observed_records"] == counts["expected_policy_rows"]
        and counts["unique_policy_rows"] == counts["expected_policy_rows"]
        and counts["missing_policy_rows"] == 0
        and counts["duplicate_policy_rows"] == 0
        and counts["pass_policy_rows"] == counts["expected_policy_rows"]
        and counts["runtime_future_gt_true_rows"] == 0
        and counts["event_frame_missing_rows"] == 0
        and counts["event_plus_one_missing_rows"] == 0
    )
    manifest_payload = {
        "protocol": N38R1_PROTOCOL,
        "status": "PASS" if complete else "BLOCKED",
        "scope": scope,
        "manifest": str(manifest_path.resolve().relative_to(ROOT)),
        "event_count_in_scope": len(event_ids),
        "event_ids": event_ids,
        "variant_order": list(VARIANTS),
        "policy_rows": records,
        "counts": counts,
        "unique_key_definition": "event_id + variant",
        "frozen_n38_protocol_hash": protocol_hash(),
        "selection_rule": smoke_selection_rule,
        "runtime_future_gt_used": False,
        "atomic_worker_artifacts": True,
        "no_existing_artifact_overwritten": True,
        "downstream_authorized": False,
    }
    if scope == "smoke":
        manifest_path_out = N38R1 / "sidecar_manifest_smoke_attempt1.json"
    elif scope == "remaining":
        manifest_path_out = N38R1 / "sidecar_manifest_remaining_attempt1.json"
    else:
        manifest_path_out = N38R1 / "sidecar_manifest.json"
    atomic_json(manifest_path_out, manifest_payload)
    if scope == "smoke":
        smoke_stage = {
            "stage": "N38R1-01",
            "status": "SMOKE_PASS_PENDING_REMAINING" if complete else "BLOCKED_INPUT_ARTIFACT_SCHEMA",
            "real_data_status": "SMOKE_PASS" if complete else "BLOCKED",
            "event_count": len(event_ids),
            "variant_count": len(VARIANTS),
            "policy_row_count": counts["observed_records"],
            "sidecar_manifest": str(manifest_path_out.resolve().relative_to(ROOT)),
            "counts": counts,
            "frozen_n38_protocol_hash": protocol_hash(),
            "targeted_smoke_selection_rule": smoke_selection_rule,
            "runtime_future_gt_used": False,
            "downstream_authorized": False,
            "next_action": "Run remaining 21 frozen events with --all --resume only after smoke PASS." if complete else "Preserve smoke failures and repair only the first actionable root cause.",
        }
        atomic_json(N38R1 / "stage_01_status_smoke_attempt1.json", smoke_stage)
        atomic_json(N38R1 / "stage_01_status.json", smoke_stage)
    elif scope == "all":
        # This is a separate atomic status record from the machine-readable
        # manifest; it is updated only after the full 120-key gate is known.
        stage_payload = {
            "stage": "N38R1-01",
            "status": "PASS" if complete else "BLOCKED_INPUT_ARTIFACT_SCHEMA",
            "real_data_status": "PASS" if complete else "BLOCKED",
            "event_count": len(event_ids),
            "variant_count": len(VARIANTS),
            "policy_row_count": counts["observed_records"],
            "sidecar_manifest": str(manifest_path_out.resolve().relative_to(ROOT)),
            "counts": counts,
            "frozen_n38_protocol_hash": protocol_hash(),
            "runtime_future_gt_used": False,
            "targeted_smoke_passed": True,
            "downstream_authorized": False,
            "next_action": "Proceed to N38R1 Stage 02 diagnostic only after this exact sidecar gate." if complete else "Preserve sidecar failures and repair only the first actionable root cause.",
        }
        atomic_json(N38R1 / "stage_01_status.json", stage_payload)
    return manifest_payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=N37_MANIFEST)
    parser.add_argument("--sidecar-root", type=Path, default=DEFAULT_SIDECAR)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--python", dest="python_bin", default=sys.executable)
    args = parser.parse_args()
    if args.smoke == args.all:
        parser.error("choose exactly one of --smoke or --all")
    try:
        events = load_events(args.manifest)
        if args.smoke:
            event_ids, rule = smoke_event_ids(events)
            scope = "smoke"
        else:
            event_ids = [event_id_of(item) for item in events]
            rule = "all 24 frozen N37 manifest events in canonical manifest order"
            scope = "all"
        payload = run_scope(
            args.manifest,
            args.sidecar_root,
            event_ids,
            scope=scope,
            smoke_selection_rule=rule,
            resume=args.resume,
            python_bin=args.python_bin,
        )
        print(json.dumps({"scope": scope, "status": payload["status"], "counts": payload["counts"]}, sort_keys=True), flush=True)
        return 0 if payload["status"] == "PASS" else 1
    except Exception as exc:
        failure = {
            "protocol": N38R1_PROTOCOL,
            "status": "FAIL",
            "scope": "smoke" if args.smoke else "all",
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
        }
        atomic_json(N38R1 / "sidecar_supervisor_failure_attempt1.json", failure)
        print(failure["error"], file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
