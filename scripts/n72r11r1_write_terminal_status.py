#!/usr/bin/env python3
"""Write the terminal N72R11R1 resource-gate artifacts from sealed evidence."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
AUDIT = ROOT / "outputs/N72R11R1/retry_output_audit_attempt_01.json"
OOM = ROOT / "outputs/N72R11R1/oom_retry_manifest.json"
RETRY = ROOT / "outputs/N72R11R1/secondary_retry_manifest_attempt_05.json"
MERGED = ROOT / "outputs/N72R11R1/secondary_batch_merged_attempt_05.json"
OUT_ROOT = ROOT / "outputs/N72R11R1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(text)
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


def atomic_json(path: Path, value: Any) -> None:
    atomic_write(path, json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n")


def main() -> int:
    audit = json.loads(AUDIT.read_text(encoding="utf-8"))
    oom = json.loads(OOM.read_text(encoding="utf-8"))
    retry = json.loads(RETRY.read_text(encoding="utf-8"))
    merged = json.loads(MERGED.read_text(encoding="utf-8"))
    merged_counts = dict(merged.get("counts", {}))
    dynamic = audit["dynamic_retry"]
    smoke = audit["required_smoke"]
    failed_ids = [str(item["event_id"]) for item in dynamic.get("failure_artifacts", [])]
    status = "BLOCKED_INTRINSIC_OFFICIAL_SAM3_MEMORY"
    now = datetime.now(timezone.utc).isoformat()

    controller = {
        "schema_version": "N72R11R1_CONTROLLER_STATUS_V1",
        "created_at_utc": now,
        "stage": "N72R11R1_TERMINAL_RESOURCE_GATE",
        "status": status,
        "research_gate": "BLOCKED_INCOMPLETE_SECONDARY_CORPUS",
        "production_authorized": False,
        "calibration_authorized": False,
        "selector_authorized": False,
        "decoder_lora_authorized": False,
        "interaction_source": "simulated_from_gt",
        "real_human_evidence": False,
        "oom_required": 37,
        "oom_pass": 26,
        "oom_remaining": 11,
        "gpu_snapshot_failures": int(dynamic["gpu_snapshot_failures"]),
        "idle_gpu_launches": int(dynamic["idle_launches"]),
        "shared_gpu_launches": int(dynamic["shared_gpu_launches"]),
        "secondary_complete_count": int(merged_counts.get("PASS", 0)),
        "secondary_required_count": 527,
        "train_examples": None,
        "validation_examples": None,
        "future_train_positive": None,
        "future_val_positive": None,
        "v3_target_accuracy": None,
        "v3_none_accuracy": None,
        "v3_future_accuracy": None,
        "bridge_validation_loss": None,
        "bridge_protected_violation": None,
        "E1_H20": None,
        "E1_H50": None,
        "E1_H100": None,
        "E2_E1_H20": None,
        "E2_E1_H50": None,
        "E2_E1_H100": None,
        "good_fresh_selected": None,
        "good_fresh_solver_refused": None,
        "protected_E1": None,
        "protected_E2": None,
        "complete_correct_reacquisition": None,
        "complete_wrong_reacquisition": None,
        "runtime_future_gt_used": False,
        "gpu_active_count": 0,
        "gpu_max_concurrent_count": int(retry.get("max_workers", 4)),
        "gpu_ids_discovered": retry.get("gpu_ids", []),
        "next_root_cause": "INTRINSIC_OOM_AFTER_SERIALIZED_FEATURE_PIPELINE",
        "notes": "The required full-window smoke passed, but 11 of the remaining 36 OOM keys still failed on high-free idle GPUs. Downstream corpus/training/replay was not authorized.",
        "evidence": {
            "oom_manifest": str(OOM),
            "oom_manifest_sha256": sha256_file(OOM),
            "retry_manifest": str(RETRY),
            "retry_manifest_sha256": sha256_file(RETRY),
            "retry_output_audit": str(AUDIT),
            "retry_output_audit_sha256": sha256_file(AUDIT),
            "merged_manifest": str(MERGED),
            "merged_manifest_sha256": sha256_file(MERGED),
            "failed_event_ids": failed_ids,
        },
    }

    final_gate = {
        "schema_version": "N72R11R1_FINAL_GATE_V1",
        "created_at_utc": now,
        "status": status,
        "terminal": True,
        "research_gate": "BLOCKED_INCOMPLETE_SECONDARY_CORPUS",
        "production_authorized": False,
        "calibration_authorized": False,
        "selector_authorized": False,
        "decoder_lora_authorized": False,
        "interaction_source": "simulated_from_gt",
        "real_human_evidence": False,
        "runtime_future_gt_used": False,
        "historical_outputs_modified": False,
        "third_party_sam3_modified": False,
        "secondary_batch": {
            "required_event_policy_keys": 527,
            "original_n72r11_merged_pass": 490,
            "r1_required_smoke_pass": 1,
            "r1_dynamic_retry_pass": 25,
            "r1_dynamic_retry_failures": 11,
            "r1_dynamic_retry_failure_types": {"OutOfMemoryError": 11},
            "r1_retry_total_with_smoke": 37,
            "merged_record_count": int(merged.get("record_count", 0)),
            "merged_pass": int(merged_counts.get("PASS", 0)),
            "merged_fail_child": int(merged_counts.get("FAIL_CHILD", 0)),
            "duplicate_count": int(merged.get("duplicate_count", -1)),
            "missing_count": int(merged.get("missing_count", -1)),
            "status": merged.get("status"),
            "complete_pass_all_selected": False,
        },
        "retry_engineering": {
            "required_smoke": smoke,
            "serial_sam3_then_osnet": True,
            "official_batched_grounding_batch_size": 1,
            "allocator": "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True",
            "dynamic_gpu_discovery": True,
            "dynamic_gpu_ids": retry.get("gpu_ids", []),
            "max_workers": retry.get("max_workers"),
            "compute_process_probe_supported": dynamic.get("compute_process_probe_supported"),
            "idle_launches": dynamic.get("idle_launches"),
            "shared_gpu_launches": dynamic.get("shared_gpu_launches"),
            "same_physical_gpu_overlap_pairs": dynamic.get("same_physical_gpu_overlap_pairs"),
            "official_video_cpu_offload": True,
            "official_output_cpu_offload": True,
            "official_state_cpu_offload": False,
            "controller_post_session_materializer": "after_sam3_session_release",
            "runtime_future_gt_used": False,
        },
        "integrity_gate": {
            "required_records": 527,
            "actual_records": int(merged.get("record_count", 0)),
            "complete_pass_records": int(merged_counts.get("PASS", 0)),
            "failure_records": int(merged_counts.get("FAIL_CHILD", 0)),
            "duplicate_count": int(merged.get("duplicate_count", -1)),
            "missing_count": int(merged.get("missing_count", -1)),
            "unavailable_count": int(merged_counts.get("FAIL_CHILD", 0)),
            "not_run_count": 0,
            "partial_count": int(merged_counts.get("FAIL_CHILD", 0)),
            "pass": False,
        },
        "downstream": {
            "causal_corpus": "NOT_RUN_UPSTREAM_BLOCKED",
            "temporal_v3_training": "NOT_RUN_UPSTREAM_BLOCKED",
            "target_edge_bridge_training": "NOT_RUN_UPSTREAM_BLOCKED",
            "formal_E0_E1_E2_replay": "NOT_RUN_UPSTREAM_BLOCKED",
            "paired_scoring": "NOT_RUN_UPSTREAM_BLOCKED",
            "oracle_or_selector": "NOT_RUN_UPSTREAM_BLOCKED",
        },
        "blocker": {
            "root_cause": "Official SAM3 future-frame propagation still reaches the 40 GiB device limit after the serialized SAM3→OSNet pipeline, official grounding batch size 1, allocator setting and dynamic high-free-GPU scheduling.",
            "remaining_failure_count": 11,
            "remaining_failure_type": "OutOfMemoryError",
            "remaining_failure_ids": failed_ids,
            "resource_gate_only": True,
            "protocol_changed": False,
            "allowed_next_step": "Use a genuinely larger or less-contended environment, or an already-supported official state-memory mechanism; rerun only the 11 preserved keys with the frozen protocol.",
        },
        "evidence_paths": {
            "retry_output_audit": str(AUDIT),
            "oom_retry_manifest": str(OOM),
            "remaining_oom_event_ids": str(OUT_ROOT / "remaining_oom_event_ids.json"),
            "secondary_retry_manifest": str(RETRY),
            "merged_manifest": str(MERGED),
        },
    }
    atomic_json(OUT_ROOT / "CONTROLLER_STATUS.json", controller)
    atomic_json(OUT_ROOT / "n72r11r1_final_gate.json", final_gate)
    human = f"""# N72R11R1 Terminal Status

- Status: `{status}`
- Frozen OOM keys: `37`; required full-window smoke: `1 PASS`
- Dynamic retry: `25 PASS`, `11 OutOfMemoryError`
- Merged secondary corpus: `516/527 PASS`, `11 FAIL_CHILD`, duplicate `0`, missing `0`
- Dynamic scheduler: physical GPUs `{retry.get('gpu_ids', [])}`, maximum concurrent children `{retry.get('max_workers')}`, idle launches `{dynamic.get('idle_launches')}`, same-GPU overlaps `{len(dynamic.get('same_physical_gpu_overlap_pairs', []))}`
- Runtime future GT: `false`; historical outputs and `third_party/sam3`: unchanged
- Causal corpus, V3, TargetEdgeBridge and formal E0/E1/E2: `NOT_RUN_UPSTREAM_BLOCKED`

The remaining blocker is intrinsic official SAM3 future-propagation memory exhaustion after the authorized serialized feature pipeline, grounding batch size 1, allocator setting and high-free dynamic scheduling. All failure artifacts remain preserved under the R1 output tree.
"""
    atomic_write(OUT_ROOT / "HUMAN_READABLE_STATUS.md", human)
    print(json.dumps({
        "status": status,
        "controller_status": str(OUT_ROOT / "CONTROLLER_STATUS.json"),
        "final_gate": str(OUT_ROOT / "n72r11r1_final_gate.json"),
        "secondary_complete": int(merged_counts.get("PASS", 0)),
        "secondary_required": 527,
        "oom_remaining": 11,
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
