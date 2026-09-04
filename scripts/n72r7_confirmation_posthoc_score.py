#!/usr/bin/env python3
"""Posthoc-score the sealed two-event N72R7 confirmation replay.

This wrapper is intentionally separate from the historical 32-event scorer:
the two deferred events have no N72R6 event manifest.  It reuses the exact
metric implementation and bootstrap settings, but supplies only the audited
N72R5R1 B0 rows and the new D1/D2 confirmation rows.  Dataset GT is opened
only after the CPU runtime audit has passed.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import traceback
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import n72r7_posthoc_score as scorer  # noqa: E402
from scripts.n72r7_confirmation_runtime_audit import audit as runtime_audit  # noqa: E402
from scripts.n72r7_confirmation_runtime_audit import read_json, read_jsonl, resolve, sha256  # noqa: E402


CONFIRMATION_PROTOCOL = ROOT / "outputs/N72R7/confirmation/confirmation_protocol.json"
EVENT_POLICY = ROOT / "outputs/N72R5/mechanism_rounds/round_06_event_policy/real_event_manifest.json"


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path: Path, payload: Any) -> None:
    scorer.atomic_json(path, payload)


def atomic_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    scorer.atomic_jsonl(path, rows)


def _manifest_rows(path: Path) -> list[dict[str, Any]]:
    manifest = read_json(path)
    frames = resolve(str(manifest["frames"]))
    if sha256(frames) != str(manifest["frames_sha256"]):
        raise RuntimeError(f"replay frames hash mismatch: {frames}")
    return read_jsonl(frames)


def _scenarios(replay_root: Path, protocol: dict[str, Any], policy: dict[str, Any]) -> dict[str, dict[str, Any]]:
    policy_by_id = {str(item["event_id"]): item for item in policy.get("events", [])}
    scenarios: dict[str, dict[str, Any]] = {}
    for spec in protocol.get("events", []):
        event_id = str(spec["event_id"])
        if event_id not in policy_by_id:
            raise RuntimeError(f"confirmation event missing from frozen N72R5 policy: {event_id}")
        worker = read_json(replay_root / event_id / "worker_status.json")
        if worker.get("status") != "PASS_N72R7_CONFIRMATION_REPLAY":
            raise RuntimeError(f"confirmation worker is not PASS: {event_id}")
        c0_rows = read_jsonl(resolve(str(worker["frozen_manifest"]["c0"]["path"])))
        d1_rows = _manifest_rows(resolve(str(worker["D1_manifest"])))
        d2_rows = _manifest_rows(resolve(str(worker["D2_manifest"])))
        scenarios[event_id] = {
            "event_id": event_id,
            "sequence": str(spec["sequence"]),
            "event_frame": int(spec["event_frame"]),
            "target_public_id": int(spec["target_public_id"]),
            "action_type": str(policy_by_id[event_id]["action_type"]),
            "target_dataset_gt_id": int(policy_by_id[event_id]["dataset_gt_id"]),
            "c0_rows": c0_rows,
            "c0_manifest": worker["frozen_manifest"],
            "D1_rows": d1_rows,
            "D2_rows": d2_rows,
        }
    return scenarios


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--replay-root", type=Path, required=True)
    parser.add_argument("--attempt", type=int, default=1)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    replay_root = resolve(args.replay_root)
    output_root = resolve(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    failure_path = output_root / "attempts" / f"posthoc_failure_attempt{int(args.attempt)}.json"
    try:
        audit_result = runtime_audit(replay_root, attempt=int(args.attempt))
        if not audit_result["status"].startswith("PASS"):
            raise RuntimeError(f"confirmation runtime audit did not pass: {audit_result['errors'][:3]}")
        protocol = read_json(CONFIRMATION_PROTOCOL)
        policy = read_json(EVENT_POLICY)
        scenarios = _scenarios(replay_root, protocol, policy)
        runtime_validation_path = output_root / "runtime_validation.json"
        result_path = output_root / "n72r7_confirmation_posthoc_results.json"
        event_metrics_path = output_root / "event_metrics.jsonl"
        runtime_validation = {
            "schema_version": "N72R7_CONFIRMATION_POSTHOC_RUNTIME_VALIDATION_V1",
            "status": "PASS_N72R7_CONFIRMATION_RUNTIME_AUDIT",
            "runtime_audit": str(replay_root / f"runtime_audit.json"),
            "runtime_audit_sha256": sha256(replay_root / "runtime_audit.json"),
            "event_count": len(scenarios),
            "variants": ["D0", "D1", "D2"],
            "D0_source": "N72R5R1_B0_public_assignment_rows",
            "D1_source": "N72R7_confirmation_frozen_B0_plus_learned_selector",
            "D2_source": "N72R7_confirmation_frozen_B0_plus_current_target_session",
            "gt_loaded_in_worker": False,
            "posthoc_gt_not_loaded_during_validation": True,
            "runtime_future_gt_used": False,
            "posthoc_gt_used": False,
            "interaction_source": "simulated_from_gt",
            "real_human_evidence": False,
            "created_at_utc": now_utc(),
        }
        atomic_json(runtime_validation_path, runtime_validation)
        original_loader = scorer._load_runtime_scenarios
        original_validation = scorer.RUNTIME_VALIDATION_PATH
        original_result = scorer.RESULT_PATH
        original_event_metrics = scorer.EVENT_METRICS_PATH
        scorer._load_runtime_scenarios = lambda: scenarios
        scorer.RUNTIME_VALIDATION_PATH = runtime_validation_path
        scorer.RESULT_PATH = result_path
        scorer.EVENT_METRICS_PATH = event_metrics_path
        try:
            runtime_rows = {
                "D0": {event_id: item["c0_rows"] for event_id, item in scenarios.items()},
                "D1": {event_id: item["D1_rows"] for event_id, item in scenarios.items()},
                "D2": {event_id: item["D2_rows"] for event_id, item in scenarios.items()},
            }
            # posthoc_score is the first call in this wrapper that loads train GT.
            result = scorer.posthoc_score({"audit": runtime_validation, "rows": runtime_rows})
        finally:
            scorer._load_runtime_scenarios = original_loader
            scorer.RUNTIME_VALIDATION_PATH = original_validation
            scorer.RESULT_PATH = original_result
            scorer.EVENT_METRICS_PATH = original_event_metrics
        d1 = result["aggregate"]["D1_vs_D0"]["20"]
        d2 = result["aggregate"]["D2_vs_D0"]["20"]
        d2_d1 = result["aggregate"]["D2_vs_D1"]["20"]
        stage = {
            "schema_version": "N72R7_CONFIRMATION_POSTHOC_STATUS_V1",
            "status": "PASS_CONFIRMATION_EXECUTION_FAIL_FUTURE_EFFECT" if result["gate"]["research_gate"] != "PASS_GT_SIMULATED_CLOSED_LOOP_REACQUISITION_CONFIRMED" else "PASS_CONFIRMATION_DEVELOPMENT_EFFECT",
            "research_gate": result["gate"]["research_gate"],
            "event_count": result["event_count"],
            "independent_sequence_count": result["independent_sequence_count"],
            "runtime_audit": str(replay_root / "runtime_audit.json"),
            "runtime_validation": str(runtime_validation_path),
            "result_artifact": str(result_path),
            "event_metrics": str(event_metrics_path),
            "D1_vs_D0_H20": d1,
            "D2_vs_D0_H20": d2,
            "D2_vs_D1_H20": d2_d1,
            "bootstrap": result["bootstrap_protocol"],
            "production_authorized": False,
            "calibration_selector_decoder_lora_authorized": False,
            "interaction_source": "simulated_from_gt",
            "real_human_evidence": False,
            "runtime_future_gt_used": False,
            "posthoc_gt_used": True,
            "created_at_utc": now_utc(),
        }
        atomic_json(output_root / "stage_13_confirmation_posthoc_status.json", stage)
        print(json.dumps({"status": stage["status"], "research_gate": stage["research_gate"], "D2_vs_D1_H20": d2_d1}, sort_keys=True))
        return 0
    except Exception as exc:
        atomic_json(failure_path, {
            "schema_version": "N72R7_CONFIRMATION_POSTHOC_FAILURE_V1",
            "status": "FAIL_N72R7_CONFIRMATION_POSTHOC",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
            "runtime_future_gt_used": False,
            "posthoc_gt_used": False,
            "created_at_utc": now_utc(),
        })
        print(json.dumps({"status": "FAIL_N72R7_CONFIRMATION_POSTHOC", "failure": str(failure_path), "error": str(exc)}, sort_keys=True))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
