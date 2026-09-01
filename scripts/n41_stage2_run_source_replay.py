#!/usr/bin/env python3
"""N41-02 supervisor for preregistered A/B/C source replays.

One worker is launched per (event, source, weight configuration).  Workers
are sequential and CPU-only so there is no resource contention and every
failure gets an independent artifact.  This script never imports GT and does
not select a source/configuration from results.
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


N37_MANIFEST = ROOT / "outputs" / "n37" / "real_event_manifest.json"
SOURCE_MANIFEST = ROOT / "outputs" / "n41" / "source_replay" / "source_embedding_manifest.json"
PROTOCOL_PATH = ROOT / "outputs" / "n41" / "source_replay" / "source_protocol.json"
WORKER = ROOT / "scripts" / "n41_stage2_source_worker.py"
OUT = ROOT / "outputs" / "n41" / "source_replay"
ATTEMPTS = ROOT / "outputs" / "n41" / "attempts"
PROTOCOL = "N41_GT_CONTROLLED_APPEARANCE_SOURCE_ABLATION_SCAN_V1"
SOURCES = ("A_ideal_gt_roi", "B_frozen_current_human_region", "C_fixed_corrupted_roi")


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return value


def load_frozen_inputs() -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    protocol = load_json(PROTOCOL_PATH)
    if protocol.get("status") != "FROZEN_BEFORE_SOURCE_GENERATION_AND_REPLAY":
        raise RuntimeError("N41-02 source protocol is not frozen")
    source_payload = load_json(SOURCE_MANIFEST)
    if source_payload.get("status") != "PASS" or source_payload.get("event_count") != 24:
        raise RuntimeError("N41 source manifest is not PASS/24")
    n37 = load_json(N37_MANIFEST)
    if n37.get("status") != "PASS" or n37.get("event_count") != 24:
        raise RuntimeError("N37 frozen manifest is not PASS/24")
    events = n37.get("events")
    if not isinstance(events, list) or len(events) != 24:
        raise RuntimeError("N37 event list is not exactly 24")
    source_entries = {str(item["event_id"]): item for item in source_payload.get("events", [])}
    if len(source_entries) != 24:
        raise RuntimeError("N41 source sidecar event keys are not unique/complete")
    n37_ids = [str(item["event"]["event_id"]) for item in events]
    if set(n37_ids) != set(source_entries):
        raise RuntimeError("N41 source sidecar and N37 event key sets differ")
    return protocol, events, source_payload


def event_ids_for_phase(protocol: dict[str, Any], events: list[dict[str, Any]], phase: str) -> list[str]:
    smoke = [str(value) for value in protocol["smoke"]["event_ids"]]
    all_ids = [str(item["event"]["event_id"]) for item in events]
    if phase == "smoke":
        if len(smoke) != 3 or not set(smoke).issubset(all_ids):
            raise RuntimeError("frozen smoke IDs are not exactly three valid events")
        return smoke
    return all_ids


def config_token(config_id: str) -> str:
    return str(config_id).replace("/", "_")


def artifact_path(phase: str, attempt: int, event_id: str, source_id: str, config_id: str) -> Path:
    safe_event = str(event_id).replace("/", "_")
    safe_source = str(source_id).replace("/", "_")
    return OUT / phase / f"attempt{int(attempt)}" / safe_event / safe_source / f"{config_token(config_id)}.json"


def candidate_signature(audit: dict[str, Any]) -> tuple[Any, ...]:
    # The candidate stream is the detector/native axis.  ``public_id_order``
    # and ``candidate_public_ids`` are intentionally excluded: appearance is
    # allowed to change the state assignment and can therefore change the
    # active public-ID columns.  Treating those outcome fields as candidate
    # axes would reject the very assignment-boundary effect under study.
    mapping = audit.get("candidate_public_id_mapping", [])
    stable_mapping = []
    if isinstance(mapping, list):
        for row in mapping:
            if not isinstance(row, dict):
                stable_mapping.append(("INVALID",))
                continue
            stable_mapping.append(
                (
                    row.get("candidate_index"),
                    row.get("candidate_native_id"),
                    row.get("candidate_local_native_id"),
                    row.get("sequence_global_id"),
                    row.get("source_public_id"),
                )
            )
    return (
        tuple(audit.get("candidate_order", [])),
        tuple(audit.get("candidate_native_ids", [])),
        tuple(stable_mapping),
    )


def finite_matrices(audit: dict[str, Any]) -> bool:
    shapes = []
    for key in ("base_scores_before_appearance", "appearance_memory_scores", "appearance_score_deltas", "fused_scores"):
        values = np.asarray(audit.get(key, []), dtype=float)
        if values.ndim != 2 or not np.all(np.isfinite(values)):
            return False
        shapes.append(values.shape)
    return len(set(shapes)) == 1 and shapes[0][0] == len(audit.get("candidates", [])) and shapes[0][1] == len(audit.get("public_id_order", []))


def validate_artifact(path: Path, event_id: str, source_id: str, config_id: str, check_determinism: bool) -> dict[str, Any]:
    artifact = load_json(path)
    issues: list[str] = []
    if artifact.get("status") != "PASS":
        issues.append(f"status:{artifact.get('status')}")
    if str(artifact.get("event_id")) != str(event_id):
        issues.append("event_id_mismatch")
    if str(artifact.get("source_id")) != str(source_id):
        issues.append("source_id_mismatch")
    if str(artifact.get("weight_configuration", {}).get("config_id")) != str(config_id):
        issues.append("config_id_mismatch")
    if artifact.get("runtime_boundary", {}).get("runtime_future_gt_used") is not False:
        issues.append("runtime_future_gt_used")
    variants = artifact.get("variants")
    if not isinstance(variants, dict) or set(variants) != {"M0", "M1", "M2", "M3", "M4"}:
        issues.append("variant_key_set_invalid")
        variants = variants if isinstance(variants, dict) else {}
    event_frame = int(artifact.get("event_frame", -1))
    future_count = int(artifact.get("future_frame_count", -1))
    event_reference_signature = None
    future_reference_signatures: dict[int, tuple[Any, ...]] = {}
    for variant in ("M0", "M1", "M2", "M3", "M4"):
        value = variants.get(variant, {})
        if value.get("status") != "PASS":
            issues.append(f"variant_status:{variant}")
        event_payload = value.get("event_frame_audit", {})
        event_audit = event_payload.get("candidate_audit", {})
        if event_payload.get("is_event_frame") is not True or event_payload.get("is_future_frame") is not False:
            issues.append(f"event_flags:{variant}")
        if event_payload.get("memory_read") is not False or event_payload.get("memory_write") is not False or event_payload.get("current_frame_write_hidden") is not True:
            issues.append(f"event_causal_boundary:{variant}")
        if event_audit.get("frame") != event_frame or not finite_matrices(event_audit):
            issues.append(f"event_audit_invalid:{variant}")
        sig = candidate_signature(event_audit)
        if event_reference_signature is None:
            event_reference_signature = sig
        elif sig != event_reference_signature:
            issues.append(f"event_candidate_axis_changed:{variant}")
        for branch_name in ("memory_write=False", "memory_write=True"):
            branch = value.get("branches", {}).get(branch_name, {})
            trace = branch.get("future_trace", [])
            if not isinstance(trace, list):
                issues.append(f"trace_missing:{variant}:{branch_name}")
                continue
            frames = [int(entry.get("frame", -1)) for entry in trace if isinstance(entry, dict)]
            if len(frames) != future_count or frames != list(range(event_frame + 1, event_frame + 1 + future_count)):
                issues.append(f"future_frames_invalid:{variant}:{branch_name}")
            for entry in trace:
                audit = entry.get("candidate_audit", {})
                if not finite_matrices(audit) or audit.get("runtime_future_gt_used") is not False:
                    issues.append(f"future_audit_invalid:{variant}:{branch_name}:{entry.get('frame')}")
                frame = int(entry.get("frame", -1))
                frame_signature = candidate_signature(audit)
                reference = future_reference_signatures.setdefault(frame, frame_signature)
                if frame_signature != reference:
                    issues.append(f"future_candidate_axis_changed:{variant}:{branch_name}:{frame}")
    if check_determinism:
        for variant in ("M0", "M1", "M2", "M3", "M4"):
            if variants.get(variant, {}).get("reproducibility", {}).get("status") != "PASS":
                issues.append(f"reproducibility_not_pass:{variant}")
    return {
        "event_id": str(event_id),
        "source_id": str(source_id),
        "config_id": str(config_id),
        "path": str(path.relative_to(ROOT)),
        "status": "PASS" if not issues else "FAIL",
        "issues": issues,
        "future_frame_count": future_count,
        "variant_count": len(variants),
    }


def validate_batch_candidate_stream(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Verify the frozen detector/native stream across source/config workers.

    Assignment-dependent public IDs are retained in each artifact but are not
    used as a stream-equality key.  This distinction is necessary because an
    assignment change is a permitted treatment outcome, whereas a candidate
    order/native/mapping change would invalidate the paired experiment.
    """
    reference: dict[tuple[str, int], tuple[Any, ...]] = {}
    comparisons = 0
    issues: list[str] = []
    for record in records:
        if record.get("returncode") != 0 or record.get("artifact_audit", {}).get("status") != "PASS":
            continue
        artifact = load_json(ROOT / str(record["output"]))
        event_id = str(record["event_id"])
        for variant, value in artifact.get("variants", {}).items():
            event_audit = value.get("event_frame_audit", {}).get("candidate_audit", {})
            key = (event_id, int(event_audit.get("frame", -1)))
            sig = candidate_signature(event_audit)
            if key in reference and reference[key] != sig:
                issues.append(f"event_stream_changed:{event_id}:{key[1]}:{variant}")
            else:
                reference.setdefault(key, sig)
            comparisons += 1
            for branch_name, branch in value.get("branches", {}).items():
                for entry in branch.get("future_trace", []):
                    audit = entry.get("candidate_audit", {})
                    frame = int(entry.get("frame", -1))
                    key = (event_id, frame)
                    sig = candidate_signature(audit)
                    if key in reference and reference[key] != sig:
                        issues.append(f"future_stream_changed:{event_id}:{frame}:{variant}:{branch_name}")
                    else:
                        reference.setdefault(key, sig)
                    comparisons += 1
    return {
        "status": "PASS" if not issues else "FAIL",
        "event_frame_and_future_key_count": len(reference),
        "comparison_count": comparisons,
        "issue_count": len(issues),
        "issues": issues[:100],
        "assignment_public_id_columns_excluded_from_stream_key": True,
    }


def launch(event_id: str, source_id: str, config_id: str, output: Path, check_determinism: bool) -> dict[str, Any]:
    output.parent.mkdir(parents=True, exist_ok=True)
    started = now()
    env = os.environ.copy()
    env.update({
        "PYTHONHASHSEED": "0",
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "CUDA_VISIBLE_DEVICES": "",
        "N41_SOURCE_REPLAY_WORKER": "1",
    })
    command = [
        sys.executable,
        str(WORKER),
        "--event-id", str(event_id),
        "--source-id", str(source_id),
        "--config-id", str(config_id),
        "--output", str(output),
    ]
    if check_determinism:
        command.append("--check-determinism")
    completed = subprocess.run(command, cwd=str(ROOT), env=env, text=True, capture_output=True, check=False)
    record = {
        "event_id": str(event_id),
        "source_id": str(source_id),
        "config_id": str(config_id),
        "output": str(output.relative_to(ROOT)),
        "command": command,
        "environment": {key: env[key] for key in ("PYTHONHASHSEED", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "CUDA_VISIBLE_DEVICES", "N41_SOURCE_REPLAY_WORKER")},
        "started_at": started,
        "finished_at": now(),
        "returncode": int(completed.returncode),
        "stdout": completed.stdout[-20000:],
        "stderr": completed.stderr[-20000:],
    }
    if completed.returncode == 0:
        try:
            record["artifact_audit"] = validate_artifact(output, event_id, source_id, config_id, check_determinism)
        except Exception as exc:
            record["artifact_audit"] = {"status": "FAIL", "error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc()}
            record["returncode"] = 99
    if record["returncode"] != 0 or record.get("artifact_audit", {}).get("status") != "PASS":
        failure = ATTEMPTS / f"n41_source_worker_{event_id}_{source_id}_{config_token(config_id)}.json"
        atomic_json(failure, {
            "protocol": PROTOCOL,
            "status": "FAIL_WORKER_OR_AUDIT",
            "record": record,
            "artifact_is_failure_evidence": True,
        })
        record["failure_artifact"] = str(failure.relative_to(ROOT))
    return record


def run(phase: str, attempt: int) -> dict[str, Any]:
    protocol, events, source_payload = load_frozen_inputs()
    event_ids = event_ids_for_phase(protocol, events, phase)
    source_ids = [str(value) for value in protocol.get("sources", {})]
    if tuple(source_ids) != SOURCES:
        raise RuntimeError(f"frozen source order changed: {source_ids}")
    configs = protocol.get("weight_grid", [])
    config_ids = [str(config["config_id"]) for config in configs]
    if len(config_ids) != 2 or len(set(config_ids)) != 2:
        raise RuntimeError("frozen source replay weight grid is not exactly two unique configs")
    records: list[dict[str, Any]] = []
    started = now()
    expected = len(event_ids) * len(source_ids) * len(config_ids)
    for event_id in event_ids:
        for source_id in source_ids:
            for config_id in config_ids:
                path = artifact_path(phase, attempt, event_id, source_id, config_id)
                record = launch(event_id, source_id, config_id, path, check_determinism=(phase == "smoke"))
                records.append(record)
                print(json.dumps({"phase": phase, "event_id": event_id, "source_id": source_id, "config_id": config_id, "returncode": record["returncode"]}, sort_keys=True), flush=True)
                if record["returncode"] != 0 or record.get("artifact_audit", {}).get("status") != "PASS":
                    stage = {
                        "stage": "N41-02",
                        "status": "FAIL_SOURCE_REPLAY",
                        "phase": phase,
                        "attempt": int(attempt),
                        "protocol": PROTOCOL,
                        "completed_worker_count": len(records),
                        "expected_worker_count": expected,
                        "failure_artifact": record.get("failure_artifact"),
                        "failure_record": record,
                        "runtime_future_gt_used": False,
                        "real_human_tape_created": False,
                        "downstream_authorized": False,
                    }
                    atomic_json(OUT / f"{phase}_attempt{int(attempt)}_failure.json", stage)
                    raise RuntimeError(f"N41 source replay worker/audit failed: {record}")
    manifest_path = OUT / f"{phase}_attempt{int(attempt)}_manifest.json"
    manifest = {
        "protocol": PROTOCOL,
        "status": "SMOKE_PASS" if phase == "smoke" else "FULL_RUNTIME_PASS",
        "phase": phase,
        "attempt": int(attempt),
        "started_at": started,
        "finished_at": now(),
        "event_count": len(event_ids),
        "independent_sequence_count": len({str(item["event"]["sequence"]) for item in events if str(item["event"]["event_id"]) in set(event_ids)}),
        "source_count": len(source_ids),
        "configuration_count": len(config_ids),
        "worker_count": len(records),
        "expected_worker_count": expected,
        "all_workers_returncode_zero": all(int(item["returncode"]) == 0 for item in records),
        "all_artifact_audits_pass": all(item.get("artifact_audit", {}).get("status") == "PASS" for item in records),
        "events": event_ids,
        "sources": source_ids,
        "configurations": configs,
        "runtime_future_gt_used": False,
        "gt_loaded_in_supervisor": False,
        "worker_manifests": records,
        "output_root": str((OUT / phase / f"attempt{int(attempt)}").relative_to(ROOT)),
        "source_protocol": str(PROTOCOL_PATH.relative_to(ROOT)),
        "source_protocol_sha256": sha256(PROTOCOL_PATH),
        "source_manifest": str(SOURCE_MANIFEST.relative_to(ROOT)),
        "source_manifest_sha256": sha256(SOURCE_MANIFEST),
        "source_sidecar_event_count": source_payload.get("event_count"),
    }
    manifest["batch_candidate_stream_audit"] = validate_batch_candidate_stream(records)
    manifest["all_batch_candidate_stream_checks_pass"] = manifest["batch_candidate_stream_audit"]["status"] == "PASS"
    if not manifest["all_batch_candidate_stream_checks_pass"]:
        atomic_json(OUT / f"{phase}_attempt{int(attempt)}_failure.json", {
            "protocol": PROTOCOL,
            "status": "FAIL_BATCH_CANDIDATE_STREAM_AUDIT",
            "manifest_partial": manifest,
            "artifact_is_failure_evidence": True,
        })
        raise RuntimeError(f"N41 batch candidate stream audit failed: {manifest['batch_candidate_stream_audit']}")
    atomic_json(manifest_path, manifest)
    stage = {
        "stage": "N41-02",
        "status": "SMOKE_PASS" if phase == "smoke" else "FULL_RUNTIME_PASS_READY_FOR_POSTHOC",
        "phase": phase,
        "attempt": int(attempt),
        "protocol": PROTOCOL,
        "manifest": str(manifest_path.relative_to(ROOT)),
        "event_count": len(event_ids),
        "source_count": len(source_ids),
        "configuration_count": len(config_ids),
        "worker_count": len(records),
        "expected_worker_count": expected,
        "all_workers_returncode_zero": manifest["all_workers_returncode_zero"],
        "all_artifact_audits_pass": manifest["all_artifact_audits_pass"],
        "all_batch_candidate_stream_checks_pass": manifest["all_batch_candidate_stream_checks_pass"],
        "runtime_future_gt_used": False,
        "gt_loaded_in_supervisor": False,
        "real_human_tape_created": False,
        "downstream_authorized": phase == "full",
        "next_action": "Run the fixed 24-event A/B/C source replay." if phase == "smoke" else "Run separate posthoc source/action diagnostics; do not train before strict future-effect gate.",
    }
    atomic_json(ROOT / "outputs" / "n41" / "stage_02_status.json", stage)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("smoke", "full"), required=True)
    parser.add_argument("--attempt", type=int, default=1)
    args = parser.parse_args()
    try:
        result = run(args.phase, args.attempt)
        print(json.dumps({"status": result["status"], "phase": result["phase"], "worker_count": result["worker_count"], "manifest": str((OUT / f'{args.phase}_attempt{int(args.attempt)}_manifest.json').relative_to(ROOT))}, sort_keys=True), flush=True)
    except Exception as exc:
        print(json.dumps({"status": "FAIL", "phase": args.phase, "error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc()}, sort_keys=True), flush=True)
        raise


if __name__ == "__main__":
    main()
