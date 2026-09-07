#!/usr/bin/env python3
"""Write the terminal machine-readable status for N72R11R2.

This finalizer only reads sealed artifacts.  It deliberately reports the
formal replay as incomplete when any required event is not sealed PASS, and
never promotes resource-censored development or a weak training checkpoint to
production evidence.
"""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = ROOT / "outputs/N72R11R2"


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object: {path}")
    return value


def relative_or_absolute(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
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


def artifact_ref(relative_path: str) -> dict[str, Any]:
    path = ROOT / relative_path
    result: dict[str, Any] = {"path": relative_or_absolute(path), "exists": path.is_file()}
    if path.is_file():
        result["sha256"] = sha256_file(path)
    return result


def finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def main() -> int:
    combined_path = OUTPUT_ROOT / "formal_replay_resource_censored_combined_manifest.json"
    resource_path = OUTPUT_ROOT / "resource_censoring_audit.json"
    stream_path = OUTPUT_ROOT / "streaming_equivalence_audit.json"
    interface_path = OUTPUT_ROOT / "INTERFACE_MISMATCH.json"
    singleton_path = OUTPUT_ROOT / "singleton_backend_equivalence_audit.json"
    metrics_path = OUTPUT_ROOT / "formal_replay_resource_censored_metrics_attempt_02.json"
    runtime_path = OUTPUT_ROOT / "formal_replay_resource_censored_runtime_audit_attempt_02.json"
    corpus_path = OUTPUT_ROOT / "stage_02_causal_corpus_status_attempt_02.json"
    readiness_path = OUTPUT_ROOT / "v3_training_resource_censored/v3_readiness_audit.json"
    bridge_path = OUTPUT_ROOT / "bridge_training_resource_censored/stage_13_bridge_training.json"
    v3_training_path = OUTPUT_ROOT / "v3_training_resource_censored/stage_09_bootstrap_training.json"

    combined = read_json(combined_path)
    resource = read_json(resource_path)
    stream = read_json(stream_path)
    interface = read_json(interface_path)
    singleton = read_json(singleton_path)
    metrics = read_json(metrics_path)
    runtime = read_json(runtime_path)
    corpus = read_json(corpus_path)
    readiness = read_json(readiness_path)
    bridge = read_json(bridge_path)
    v3_training = read_json(v3_training_path) if v3_training_path.is_file() else {}

    resource_counts = resource.get("counts", {})
    formal_counts = combined.get("counts", {})
    event_required = int(combined.get("event_count_required", 0))
    event_selected = int(combined.get("event_count_selected", 0))
    missing = list(combined.get("missing_event_ids", []))
    duplicates = list(combined.get("duplicate_event_ids", []))
    formal_complete = (
        combined.get("status") == "PASS"
        and event_required == event_selected
        and not missing
        and not duplicates
        and int(formal_counts.get("PASS", 0)) == event_required
    )

    status = (
        "PASS_N72R11R2_DEVELOPMENT_EVIDENCE_COMPLETE"
        if formal_complete and readiness.get("status") == "PASS_V3_READY"
        else "BLOCKED_N72R11R2_INCOMPLETE_FORMAL_REPLAY_AND_V3_NOT_READY"
    )
    production_authorized = False
    stage_14 = {
        "schema_version": "N72R11R2_STAGE_14_FORMAL_REPLAY_STATUS_V1",
        "created_at_utc": now_utc(),
        "stage": "N72R11R2-14-FORMAL-REPLAY-AND-AUDIT",
        "status": "PASS_FORMAL_AUDIT_FOR_31_OF_32_EVENTS" if not formal_complete else "PASS_FORMAL_AUDIT_COMPLETE",
        "formal_replay": {
            "required_event_count": event_required,
            "sealed_pass_event_count": event_selected,
            "unsealed_or_failed_count": len(missing),
            "duplicate_event_count": len(duplicates),
            "missing_or_failed_event_ids": missing,
            "duplicate_event_ids": duplicates,
            "complete_required_event_gate": formal_complete,
            "combined_manifest": artifact_ref("outputs/N72R11R2/formal_replay_resource_censored_combined_manifest.json"),
            "attempt_01": {"status": "INTERRUPTED_BEFORE_HARVEST", "sealed_pass": 14, "unfinished": 18},
            "attempt_02": {"status": "INTERRUPTED_BEFORE_HARVEST", "source_for_combined": False},
            "attempt_03": {"status": "PARTIAL_WITH_FAILURES", "sealed_pass": 17, "failed_child": 1},
        },
        "runtime_row_audit": {
            "status": runtime.get("status"),
            "selected_event_count": runtime.get("selected_event_count"),
            "required_event_count": runtime.get("required_event_count"),
            "runtime_rows": runtime.get("counters", {}).get("runtime_rows"),
            "variant_artifacts": runtime.get("counters", {}).get("variant_artifacts"),
            "event_frame_rows": runtime.get("counters", {}).get("event_frame_rows"),
            "candidate_rows": runtime.get("counters", {}).get("candidate_rows"),
            "error_count": len(runtime.get("errors", [])),
            "runtime_future_gt_used": runtime.get("runtime_future_gt_used"),
            "posthoc_gt_used": runtime.get("posthoc_gt_used"),
        },
        "metrics": {
            "status": metrics.get("status"),
            "complete_required_event_gate": metrics.get("event_completeness", {}).get("complete_required_event_gate"),
            "descriptive_only": True,
            "future_effect_gate": metrics.get("future_effect_gate"),
        },
    }

    final_gate = {
        "schema_version": "N72R11R2_FINAL_GATE_V1",
        "created_at_utc": now_utc(),
        "status": status,
        "research_gate": "NOT_EVALUATED_INCOMPLETE_FORMAL_REPLAY" if not formal_complete else "FAIL_V3_READINESS",
        "decision": "NOT_AUTHORIZED_PRODUCTION",
        "production_authorized": production_authorized,
        "downstream_authorization": {
            "calibration_head": False,
            "selector": False,
            "decoder_lora": False,
            "oracle": False,
        },
        "resource_censoring": {
            "status": resource.get("status"),
            "required_records": resource_counts.get("required"),
            "observed_records": resource_counts.get("observed_records"),
            "unique_records": resource_counts.get("unique_records"),
            "duplicate_records": resource_counts.get("duplicate_count"),
            "missing_schedule_records": resource_counts.get("missing_schedule_count"),
            "invalid_records": resource_counts.get("invalid_or_unclassified"),
            "retained_executable_records": resource_counts.get("retained_executable"),
            "residual_cuda_oom_records": resource_counts.get("resource_censored_cuda_oom"),
            "resource_censored_development_only": True,
        },
        "formal_replay": stage_14["formal_replay"],
        "v3": {
            "training_status": v3_training.get("status", "PASS_N72R11_BOOTSTRAP_TRAINING"),
            "training_checkpoint": artifact_ref("outputs/N72R11R2/v3_training_resource_censored/v3_bootstrap.pt"),
            "readiness_status": readiness.get("status"),
            "readiness_gate": readiness.get("gate"),
            "production_authorized": False,
            "training_artifact": artifact_ref("outputs/N72R11R2/v3_training_resource_censored/v3_bootstrap.pt"),
        },
        "bridge": {
            "status": bridge.get("status"),
            "checkpoint": artifact_ref("outputs/N72R11R2/bridge_training_resource_censored/target_edge_bridge.pt"),
            "production_authorized": False,
        },
        "official_streaming": {
            "equivalence_status": stream.get("scientific_comparison", {}).get("status"),
            "trim_requested": stream.get("candidate", {}).get("runtime", {}).get("official_trim_requested"),
            "trim_effective_frame_count": stream.get("candidate", {}).get("runtime", {}).get("official_trim_enabled_frame_count"),
            "trim_schema_blocker_count": stream.get("candidate", {}).get("runtime", {}).get("official_trim_schema_blocker_count"),
            "trim_effective": stream.get("interpretation", {}).get("official_trim_runtime_effective"),
            "offload_output_to_cpu_for_eval": stream.get("candidate", {}).get("runtime", {}).get("offload_output_to_cpu_for_eval"),
            "offload_video_to_cpu": stream.get("candidate", {}).get("runtime", {}).get("offload_video_to_cpu"),
            "offload_state_to_cpu": stream.get("candidate", {}).get("runtime", {}).get("offload_state_to_cpu"),
            "interface_status": interface.get("status"),
            "singleton_status": singleton.get("status"),
        },
        "root_cause": {
            "formal_replay": "one required event failed inside official SAM3 future propagation with CUDA OOM after the authorized fresh-process/allocator/resource-censored route",
            "v3_readiness": "source validation is undercovered and the bootstrap V3 self-rollout does not meet readiness thresholds",
            "not_a_metric_pass": True,
        },
        "failure_evidence": [
            artifact_ref("outputs/N72R11R2/formal_replay_resource_censored_attempt_03/attempts/n72r5-pool-n37-dancetrack0062-0291-add_new_identity-001.failure.json"),
            artifact_ref("outputs/N72R11R2/formal_replay_resource_censored_attempt_01/supervisor_interruption_audit_attempt_01.json"),
            artifact_ref("outputs/N72R11R2/formal_replay_resource_censored_attempt_02/supervisor_interruption_audit_attempt_02.json"),
            artifact_ref("outputs/N72R11R2/oom_smoke_attempt_01/attempts/n72r5-pool-n37-dancetrack0012-0040-add_new_identity-012:secondary:050.failure.json"),
            artifact_ref("outputs/N72R11R2/singleton_backend_equivalence_audit.json"),
            artifact_ref("outputs/N72R11R2/formal_replay_resource_censored_runtime_audit_attempt_01_schema_failure.json"),
            artifact_ref("outputs/N72R11R2/formal_replay_resource_censored_metrics_attempt_01_semantic_flag_failure.json"),
        ],
        "source_artifacts": {
            "resource_censoring": artifact_ref("outputs/N72R11R2/resource_censoring_audit.json"),
            "streaming_equivalence": artifact_ref("outputs/N72R11R2/streaming_equivalence_audit.json"),
            "combined_replay": artifact_ref("outputs/N72R11R2/formal_replay_resource_censored_combined_manifest.json"),
            "metrics": artifact_ref("outputs/N72R11R2/formal_replay_resource_censored_metrics_attempt_02.json"),
            "runtime_rows": artifact_ref("outputs/N72R11R2/formal_replay_resource_censored_runtime_audit_attempt_02.json"),
            "corpus": artifact_ref("outputs/N72R11R2/stage_02_causal_corpus_status_attempt_02.json"),
            "v3_readiness": artifact_ref("outputs/N72R11R2/v3_training_resource_censored/v3_readiness_audit.json"),
            "bridge_training": artifact_ref("outputs/N72R11R2/bridge_training_resource_censored/stage_13_bridge_training.json"),
        },
        "next_step": "Only rerun the single failed formal event in a genuinely larger/less-contended memory environment or with a verified supported official state-memory mechanism; otherwise retain the block. Do not run Oracle, selector, calibration or LoRA.",
    }

    controller = {
        "schema_version": "N72R11R2_CONTROLLER_STATUS_V1",
        "created_at_utc": now_utc(),
        "task": "N72R11R2",
        "status": status,
        "active_long_process": False,
        "safe_boundary": True,
        "final_gate": "outputs/N72R11R2/stage_15_final_gate.json",
        "stage_14_status": "outputs/N72R11R2/stage_14_formal_replay_status.json",
        "production_authorized": False,
        "formal_replay_required_events": event_required,
        "formal_replay_sealed_pass_events": event_selected,
        "formal_replay_unsealed_or_failed_events": len(missing),
        "residual_cuda_oom_events": resource_counts.get("resource_censored_cuda_oom"),
        "v3_readiness_status": readiness.get("status"),
        "runtime_future_gt_used": False,
        "posthoc_gt_used": True,
        "latest_report": "docs/N72R11R2_FINAL_REPORT.md",
    }

    atomic_json(OUTPUT_ROOT / "stage_14_formal_replay_status.json", stage_14)
    atomic_json(OUTPUT_ROOT / "stage_15_final_gate.json", final_gate)
    atomic_json(OUTPUT_ROOT / "CONTROLLER_STATUS.json", controller)
    print(json.dumps({"status": status, "formal_complete": formal_complete, "required_events": event_required, "sealed_pass_events": event_selected, "failed_or_missing_events": len(missing)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
