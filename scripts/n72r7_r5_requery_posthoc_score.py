#!/usr/bin/env python3
"""Posthoc score R5 re-query treatment versus its same-selector baseline."""

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
from scripts.n72r7_dev_replay import validate_inputs  # noqa: E402
from scripts.n72r7_r5_requery_runtime_audit import audit as runtime_audit  # noqa: E402


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object: {path}")
    return value


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def resolved(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def manifest_rows(path: Path) -> list[dict[str, Any]]:
    manifest = read_json(path)
    frames = resolved(str(manifest["frames"]))
    digest = hashlib.sha256(frames.read_bytes()).hexdigest()
    if digest != str(manifest["frames_sha256"]):
        raise RuntimeError(f"frames hash mismatch: {frames}")
    return read_jsonl(frames)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--attempt", type=int, default=1)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    root = resolved(args.root)
    output_root = resolved(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    try:
        runtime = runtime_audit(root, attempt=int(args.attempt))
        if not runtime["status"].startswith("PASS"):
            raise RuntimeError(f"R5 runtime audit failed: {runtime['errors'][:3]}")
        _batch, policy, frozen = validate_inputs()
        batch = read_json(root / f"batch_attempt{int(args.attempt)}.json")
        policy_by_id = {str(item["event_id"]): item for item in policy["events"]}
        scenarios: dict[str, dict[str, Any]] = {}
        current_rows: dict[str, list[dict[str, Any]]] = {}
        treatment_rows: dict[str, list[dict[str, Any]]] = {}
        for item in batch.get("results", []):
            event_id = str(item["event_id"])
            worker = read_json(root / event_id / "worker_status.json")
            current_manifest = resolved(str(worker["current_manifest"]))
            treatment_manifest = resolved(str(worker["treatment_manifest"]))
            current_rows[event_id] = manifest_rows(current_manifest)
            treatment_rows[event_id] = manifest_rows(treatment_manifest)
            frozen_item = frozen[event_id]
            policy_item = policy_by_id[event_id]
            scenarios[event_id] = {
                "event_id": event_id,
                "sequence": str(frozen_item["sequence"]),
                "event_frame": int(frozen_item["event_frame"]),
                "target_public_id": int(frozen_item["target_public_id"]),
                "action_type": str(policy_item["action_type"]),
                "target_dataset_gt_id": int(policy_item["dataset_gt_id"]),
                "c0_rows": read_jsonl(resolved(str(frozen_item["c0"]["path"]))),
                "c0_manifest": frozen_item,
            }
        original_loader = scorer._load_runtime_scenarios
        scorer._load_runtime_scenarios = lambda: scenarios
        try:
            validation_path = output_root / "runtime_validation.json"
            result_path = output_root / "n72r7_r5_requery_posthoc_results.json"
            event_metrics_path = output_root / "event_metrics.jsonl"
            scorer.RUNTIME_VALIDATION_PATH = validation_path
            scorer.RESULT_PATH = result_path
            scorer.EVENT_METRICS_PATH = event_metrics_path
            atomic_json(validation_path, {
                "schema_version": "N72R7_R5_REQUERY_POSTHOC_RUNTIME_VALIDATION_V1",
                "status": "PASS_N72R7_R5_REQUERY_RUNTIME_AUDIT",
                "source_runtime_audit": str(root / f"batch_attempt{int(args.attempt)}.json"),
                "event_count": len(scenarios),
                "baseline_label": "D1_same_R2_decoder_plus_current_target_session",
                "treatment_label": "D2_same_R2_decoder_plus_current_target_session_plus_R5_requery_pool",
                "candidate_generator_protocol": str(ROOT / "outputs/N72R7/candidate_generator_protocol.json"),
                "runtime_future_gt_used": False,
                "gt_loaded_in_worker": False,
                "posthoc_gt_not_loaded_during_validation": True,
                "interaction_source": "simulated_from_gt",
                "real_human_evidence": False,
                "created_at_utc": now_utc(),
            })
            runtime_rows = {
                "D0": {event_id: scenarios[event_id]["c0_rows"] for event_id in scenarios},
                "D1": current_rows,
                "D2": treatment_rows,
            }
            result = scorer.posthoc_score({"audit": read_json(validation_path), "rows": runtime_rows})
        finally:
            scorer._load_runtime_scenarios = original_loader
        gate = result["gate"]
        d2 = result["aggregate"]["D2_vs_D0"]["20"]
        d2_d1 = result["aggregate"]["D2_vs_D1"]["20"]
        stage = {
            "schema_version": "N72R7_STAGE_R5_REQUERY_POSTHOC_STATUS_V1",
            "status": "PASS_R5_DEVELOPMENT_EFFECT" if gate["research_gate"].startswith("PASS") else "PASS_R5_EXECUTION_FAIL_FUTURE_EFFECT",
            "mechanism": "CAUSAL_MULTI_QUERY_SAM3_TARGET_REQUERY",
            "created_at_utc": now_utc(),
            "runtime_audit": str(root / f"batch_attempt{int(args.attempt)}.json"),
            "runtime_validation": str(output_root / "runtime_validation.json"),
            "result_artifact": str(output_root / "n72r7_r5_requery_posthoc_results.json"),
            "event_metrics": str(output_root / "event_metrics.jsonl"),
            "event_count": len(scenarios),
            "independent_sequence_count": len({item["sequence"] for item in scenarios.values()}),
            "baseline": "same R2 decoder v2 greedy + B0/current target session",
            "treatment": "same R2 decoder v2 greedy + B0/current target session/R5 multi-query pool",
            "candidate_generator_protocol": str(ROOT / "outputs/N72R7/candidate_generator_protocol.json"),
            "treatment_vs_d0_h20": result["aggregate"]["D2_vs_D0"]["20"],
            "treatment_vs_baseline_h20": d2_d1,
            "treatment_vs_baseline_h50": result["aggregate"]["D2_vs_D1"]["50"],
            "treatment_vs_baseline_h100": result["aggregate"]["D2_vs_D1"]["100"],
            "research_gate": gate["research_gate"],
            "production_authorized": False,
            "independent_confirmation": False,
            "posthoc_gt_used": True,
            "runtime_future_gt_used": False,
            "interaction_source": "simulated_from_gt",
            "real_human_evidence": False,
        }
        atomic_json(output_root / "stage_r5_posthoc_status.json", stage)
        print(json.dumps({"status": stage["status"], "research_gate": gate["research_gate"], "d2_vs_d1_h20": d2_d1}, sort_keys=True))
        return 0
    except Exception as exc:
        failure = output_root / "attempts" / f"posthoc_failure_attempt{int(args.attempt)}.json"
        atomic_json(failure, {
            "schema_version": "N72R7_R5_REQUERY_POSTHOC_FAILURE_V1",
            "status": "FAIL_PRESERVED",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
            "runtime_future_gt_used": False,
            "posthoc_gt_used": False,
            "created_at_utc": now_utc(),
        })
        print(json.dumps({"status": "FAIL_N72R7_R5_REQUERY_POSTHOC", "failure": str(failure)}, sort_keys=True))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
