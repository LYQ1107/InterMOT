#!/usr/bin/env python3
"""N39 Stage 2 supervisor for preregistered association-weight scans.

It launches one independent Python worker per (configuration, event), first for
the fixed three-event smoke and then for all 24 frozen N37 events.  The
supervisor itself never supplies GT to a worker and never selects a value from
results.  A failed worker is kept as an artifact and stops the relevant scan;
the caller can run a new ``--attempt`` after the smallest repair.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.n36_real_eval_common import atomic_json


N37_MANIFEST = ROOT / "outputs/n37/real_event_manifest.json"
N38R1_MANIFEST = ROOT / "outputs/n38r1/sidecar_manifest.json"
N38R1_SUMMARY = ROOT / "outputs/n38r1/diagnostic_attempt3/score_assignment_summary.json"
N36_TAPE_MANIFEST = ROOT / "outputs/n36/real_tape/tape_manifest.json"
WORKER = ROOT / "scripts/n39_weight_worker.py"
OUT = ROOT / "outputs/n39"
PROTOCOL = "N39_WEIGHTED_ASSOCIATION_INTERFACE_PROBE_SCAN_V1"
VALUES = (0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0)
MODES = ("lambda_assoc", "human_weight")


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def value_token(value: float) -> str:
    text = f"{float(value):g}"
    return text.replace("-", "m").replace(".", "p")


def config_id(mode: str, value: float) -> str:
    return f"{mode}_{value_token(value)}"


def load_events() -> list[dict[str, Any]]:
    payload = json.loads(N37_MANIFEST.read_text(encoding="utf-8"))
    if payload.get("status") != "PASS" or payload.get("event_count") != 24:
        raise RuntimeError("N37 frozen event manifest is not the expected PASS/24 input")
    events = payload.get("events")
    if not isinstance(events, list) or len(events) != 24:
        raise RuntimeError("N37 event list is not exactly 24")
    return events


def smoke_events(events: list[dict[str, Any]]) -> list[str]:
    """Select three canonical events with distinct actions and sequences."""
    selected: list[str] = []
    actions: set[str] = set()
    sequences: set[str] = set()
    for item in events:
        event = item["event"]
        action = str(event["action_type"])
        sequence = str(event["sequence"])
        if action in actions or sequence in sequences:
            continue
        selected.append(str(event["event_id"]))
        actions.add(action)
        sequences.add(sequence)
        if len(selected) == 3:
            break
    if len(selected) != 3:
        raise RuntimeError(f"deterministic smoke selection found {len(selected)} events, expected 3")
    return selected


def protocol(events: list[dict[str, Any]], smoke_ids: list[str]) -> dict[str, Any]:
    return {
        "protocol": PROTOCOL,
        "status": "FROZEN_BEFORE_SCAN",
        "frozen_inputs": {
            "n37_manifest": str(N37_MANIFEST.relative_to(ROOT)),
            "n37_manifest_sha256": sha256(N37_MANIFEST),
            "n38r1_sidecar_manifest": str(N38R1_MANIFEST.relative_to(ROOT)),
            "n38r1_sidecar_manifest_sha256": sha256(N38R1_MANIFEST),
            "n38r1_diagnostic_summary": str(N38R1_SUMMARY.relative_to(ROOT)),
            "n38r1_diagnostic_summary_sha256": sha256(N38R1_SUMMARY),
            "n36_tape_manifest": str(N36_TAPE_MANIFEST.relative_to(ROOT)),
            "n36_tape_manifest_sha256": sha256(N36_TAPE_MANIFEST),
            "scale_audit_summary": "outputs/n39/scale_audit_summary.json",
        },
        "checkpoint_and_candidate_definition": "frozen N36/N37/N38R1 inputs; no checkpoint, embedding, candidate stream, prefix, event or future window changes",
        "variants": {
            "M0": "K1 only; CCAM disabled",
            "M1": "human EMA prototype",
            "M2": "human EMA prototype + positive human anchors",
            "M3": "M2 + negative competitor bank",
            "M4": "M3 + reliability/age gate",
        },
        "scan_order": [
            {"mode": "lambda_assoc", "values": list(VALUES), "fixed_internal_human_weight": 1.0},
            {"mode": "human_weight", "values": list(VALUES), "fixed_external_appearance_score_weight": 1.0},
        ],
        "smoke": {
            "event_ids": smoke_ids,
            "require_distinct_action_and_sequence": True,
            "same_input_for_all_values": True,
            "checks": [
                "matrix_dimensions",
                "candidate_order",
                "public_id_mapping",
                "no_duplicate_or_missing_future_frames",
                "runtime_future_gt_false",
                "event_frame_write_hidden_and_t_plus_1_visible",
                "assignment_reproducible",
            ],
        },
        "posthoc_protocol": {
            "gt_loaded_only_after_all_runtime_variants_finish": True,
            "horizons": [20, 50, 100],
            "sequence_cluster_unit": "independent sequence",
            "bootstrap_seed": 36,
            "bootstrap_replicates": 2000,
        },
        "forbidden_adaptations": [
            "future-GT event/config selection",
            "threshold tuning or silent score normalization",
            "checkpoint/candidate/window/metric changes",
            "calibration head, selector, decoder LoRA before strict gate",
        ],
        "events": [
            {
                "event_id": str(item["event"]["event_id"]),
                "sequence": str(item["event"]["sequence"]),
                "action_type": str(item["event"]["action_type"]),
                "event_frame": int(item["event"]["frame"]),
            }
            for item in events
        ],
    }


def output_path(phase: str, attempt: int, mode: str, value: float, event_id: str) -> Path:
    return OUT / "weight_runs" / phase / f"attempt{int(attempt)}" / config_id(mode, value) / f"{event_id}.json"


def launch(mode: str, value: float, event_id: str, path: Path) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    started = now()
    env = os.environ.copy()
    env["PYTHONHASHSEED"] = "0"
    env["OMP_NUM_THREADS"] = "1"
    env["MKL_NUM_THREADS"] = "1"
    env["N39_WEIGHT_SCAN_WORKER"] = "1"
    command = [
        sys.executable,
        str(WORKER),
        "--event-id",
        str(event_id),
        "--mode",
        mode,
        "--value",
        str(float(value)),
        "--output",
        str(path),
    ]
    completed = subprocess.run(
        command,
        cwd=str(ROOT),
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    record = {
        "event_id": str(event_id),
        "mode": mode,
        "value": float(value),
        "config_id": config_id(mode, value),
        "output": str(path.relative_to(ROOT)),
        "command": command,
        "environment": {key: env[key] for key in ("PYTHONHASHSEED", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "N39_WEIGHT_SCAN_WORKER")},
        "started_at": started,
        "finished_at": now(),
        "returncode": int(completed.returncode),
        "stdout": completed.stdout[-20000:],
        "stderr": completed.stderr[-20000:],
    }
    if completed.returncode != 0:
        failure_path = OUT / "attempts" / f"weight_scan_{mode}_{value_token(value)}_{event_id}.json"
        atomic_json(
            failure_path,
            {
                "protocol": PROTOCOL,
                "status": "FAIL_WORKER",
                "record": record,
                "artifact_is_failure_evidence": True,
            },
        )
        record["failure_artifact"] = str(failure_path.relative_to(ROOT))
    return record


def run(phase: str, attempt: int) -> dict[str, Any]:
    events = load_events()
    smoke_ids = smoke_events(events)
    protocol_path = OUT / "weight_protocol.json"
    frozen_protocol = protocol(events, smoke_ids)
    if protocol_path.exists():
        existing = json.loads(protocol_path.read_text(encoding="utf-8"))
        if existing != frozen_protocol:
            raise RuntimeError("existing frozen weight_protocol.json differs; refusing to change protocol")
    else:
        atomic_json(protocol_path, frozen_protocol)
    started = now()
    selected_ids = smoke_ids if phase == "smoke" else [str(item["event"]["event_id"]) for item in events]
    records: list[dict[str, Any]] = []
    expected = len(MODES) * len(VALUES) * len(selected_ids)
    for mode in MODES:
        for value in VALUES:
            for event_id in selected_ids:
                path = output_path(phase, attempt, mode, value, event_id)
                record = launch(mode, value, event_id, path)
                records.append(record)
                print(json.dumps({"mode": mode, "value": value, "event_id": event_id, "returncode": record["returncode"]}, sort_keys=True), flush=True)
                if record["returncode"] != 0:
                    stage_path = OUT / "stage_02_status.json"
                    failure_path = record.get("failure_artifact")
                    atomic_json(
                        stage_path,
                        {
                            "stage": "N39-02",
                            "status": "FAIL",
                            "phase": phase,
                            "attempt": int(attempt),
                            "protocol": PROTOCOL,
                            "completed_worker_count": len(records),
                            "expected_worker_count": expected,
                            "failure": record,
                            "failure_artifact": failure_path,
                            "downstream_authorized": False,
                            "next_action": "Preserve worker failure, apply only the smallest repair, rerun the same smoke unit, then resume unfinished workers.",
                        },
                    )
                    raise RuntimeError(f"N39 weight worker failed: {record}")
    finished = now()
    status = "SMOKE_PASS" if phase == "smoke" else "PASS"
    manifest_path = OUT / "weight_runs" / f"{phase}_attempt{int(attempt)}_manifest.json"
    payload = {
        "protocol": PROTOCOL,
        "status": status,
        "phase": phase,
        "attempt": int(attempt),
        "started_at": started,
        "finished_at": finished,
        "event_count": len(selected_ids),
        "configuration_count": len(MODES) * len(VALUES),
        "worker_count": len(records),
        "expected_worker_count": expected,
        "events": selected_ids,
        "configurations": [
            {"mode": mode, "value": float(value), "config_id": config_id(mode, value)}
            for mode in MODES
            for value in VALUES
        ],
        "workers": records,
        "runtime_future_gt_used": False,
        "output_root": str((OUT / "weight_runs" / phase / f"attempt{int(attempt)}").relative_to(ROOT)),
        "protocol_artifact": str(protocol_path.relative_to(ROOT)),
    }
    atomic_json(manifest_path, payload)
    atomic_json(
        OUT / "stage_02_status.json",
        {
            "stage": "N39-02",
            "status": status,
            "phase": phase,
            "attempt": int(attempt),
            "protocol": PROTOCOL,
            "manifest": str(manifest_path.relative_to(ROOT)),
            "event_count": len(selected_ids),
            "configuration_count": len(MODES) * len(VALUES),
            "worker_count": len(records),
            "expected_worker_count": expected,
            "all_workers_returncode_zero": all(row["returncode"] == 0 for row in records),
            "runtime_future_gt_used": False,
            "downstream_authorized": phase == "full",
            "next_action": "Run fixed 24-event scan after smoke." if phase == "smoke" else "Run posthoc metrics and strict sequence-cluster gate.",
        },
    )
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("smoke", "full"), required=True)
    parser.add_argument("--attempt", type=int, default=1)
    args = parser.parse_args()
    try:
        result = run(args.phase, args.attempt)
        print(json.dumps({"status": result["status"], "phase": result["phase"], "worker_count": result["worker_count"], "manifest": result["output_root"]}, sort_keys=True), flush=True)
    except Exception as exc:
        print(json.dumps({"status": "FAIL", "error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc()}, sort_keys=True), flush=True)
        raise


if __name__ == "__main__":
    main()
