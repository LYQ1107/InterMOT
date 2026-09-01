#!/usr/bin/env python3
"""Finalize machine-readable N35 evidence when real export is blocked.

The script never edits an original done/log/tape artifact.  It summarizes the
attempts and writes separate failure, stage, and report artifacts so an
incomplete real tape cannot be mistaken for a complete experiment.
"""

from __future__ import annotations

import json
import os
import re
import tempfile
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/n35"
TAPE = OUT / "real_tape"
LOGS = OUT / "logs"
SEQUENCE_LIST = OUT / "../n34/selected_sequences.json"


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, path)
    finally:
        if os.path.exists(tmp_name):
            os.unlink(tmp_name)


def atomic_json(path: Path, payload: Any) -> None:
    atomic_text(path, json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")


def load_json(path: Path, default: Any) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return default


def load_sequences() -> list[str]:
    payload = load_json(SEQUENCE_LIST.resolve(), {})
    return sorted(
        str(item["sequence"])
        for item in payload.get("sequences", [])
        if isinstance(item, dict) and item.get("sequence")
    )


def classify_log(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8", errors="replace") if path.is_file() else ""
    reasons: list[str] = []
    if "torch.OutOfMemoryError" in text or "CUDA out of memory" in text:
        reasons.append("CUDA_OOM")
    if "KeyError: 'multistep_point_inputs'" in text:
        reasons.append("OFFLOAD_TRIM_KEYERROR")
    if "Expected all tensors to be on the same device" in text:
        reasons.append("OFFLOAD_STATE_DEVICE_MISMATCH")
    if "Traceback (most recent call last)" in text and not reasons:
        reasons.append("EXCEPTION")
    return {
        "path": str(path.resolve()),
        "exists": path.is_file(),
        "reasons": reasons,
        "has_traceback": "Traceback (most recent call last)" in text,
        "last_status_lines": [
            line.strip()
            for line in text.splitlines()
            if '"status"' in line or "OutOfMemoryError" in line or "KeyError:" in line
        ][-8:],
    }


def done_summary(sequences: list[str]) -> dict[str, Any]:
    rows: dict[str, Any] = {}
    for sequence in sequences:
        path = TAPE / "done" / f"{sequence}.json"
        rows[sequence] = load_json(path, {"status": "NOT_RUN", "artifact": str(path.resolve())})
        rows[sequence]["artifact"] = str(path.resolve())
    status_counts: dict[str, int] = {}
    for item in rows.values():
        status = str(item.get("status", "NOT_RUN"))
        status_counts[status] = status_counts.get(status, 0) + 1
    return {"by_sequence": rows, "status_counts": status_counts}


def stage(path: Path, stage_name: str, status: str, errors: list[Any], next_action: str, artifacts: list[Path] | None = None) -> None:
    atomic_json(
        path,
        {
            "stage": stage_name,
            "status": status,
            "commands": [],
            "artifacts": [str(item.resolve()) for item in (artifacts or [])],
            "errors": errors,
            "next_action": next_action,
        },
    )


def main() -> None:
    sequences = load_sequences()
    manifest = load_json(TAPE / "tape_manifest.json", {})
    audit = load_json(OUT / "real_tape_integrity_audit.json", {})
    done = done_summary(sequences)
    expected = len(sequences)
    pass_count = int(done["status_counts"].get("PASS", 0))
    fail_count = int(done["status_counts"].get("FAIL", 0))
    not_run_count = int(done["status_counts"].get("NOT_RUN", 0))

    original_shard_logs = {
        0: LOGS / "n35_export_shard_0.log",
        1: LOGS / "n35_export_shard_1.log",
        2: LOGS / "n35_export_shard_2.log",
        3: LOGS / "n35_export_shard_3.log",
    }
    original_assignment = {
        "dancetrack0006": 0,
        "dancetrack0001": 1,
        "dancetrack0015": 1,
        "dancetrack0008": 2,
        "dancetrack0002": 3,
    }
    attempts = [
        {
            "attempt": "attempt_1_4gpu_shared_process",
            "description": "initial four-GPU shard run; each shard stopped after its first failing sequence",
            "sequence_logs": {
                sequence: classify_log(original_shard_logs[shard])
                for sequence, shard in original_assignment.items()
            },
            "failure_fact": "four long-sequence CUDA OOMs were observed; dancetrack0001 completed before its shard hit the next sequence failure",
        },
        {
            "attempt": "repair_1_trim_past_non_cond_mem",
            "sequence_logs": {
                "dancetrack0002": classify_log(LOGS / "n35_repair1_dancetrack0002_gpu4.log")
            },
            "failure_fact": "official trim path raised KeyError multistep_point_inputs at the early propagation stage",
        },
        {
            "attempt": "repair_2_nested_offload_state",
            "sequence_logs": {
                "dancetrack0002": classify_log(LOGS / "n35_repair2_dancetrack0002_gpu4.log")
            },
            "failure_fact": "official nested state CPU storage reached reconditioning and raised CPU/CUDA device mismatch",
        },
        {
            "attempt": "repair_3_nested_offload_without_reconditioning",
            "sequence_logs": {
                "dancetrack0002": classify_log(LOGS / "n35_repair3_dancetrack0002_gpu4.log")
            },
            "failure_fact": "official nested state CPU storage reached new-object addition and raised CPU/CUDA device mismatch",
        },
    ]
    failure_evidence = {
        "protocol": "N35_REAL_TAPE_FAILURE_EVIDENCE",
        "sequence_expected": expected,
        "sequence_pass": pass_count,
        "sequence_fail": fail_count,
        "sequence_not_run": not_run_count,
        "duplicate_sequence_artifacts": 0,
        "original_artifacts_preserved": True,
        "done_summary": done,
        "manifest": manifest,
        "integrity_audit": audit,
        "attempts": attempts,
        "repair_budget_exhausted": True,
        "blocking_reason": "official SAM3 multiplex long-sequence state storage has no adapter-validated device-consistent CPU-offload/reconditioning path on this pinned build",
    }
    failure_path = OUT / "real_tape_failure_evidence.json"
    atomic_json(failure_path, failure_evidence)

    backend_final = {
        "protocol": "N35_BACKEND_EXPORT_AUDIT_FINAL",
        "pre_modification_audit": str((OUT / "backend_export_audit.json").resolve()),
        "adapter_fields_added": [
            "export_frame_candidates",
            "embedding",
            "embedding_status",
            "feature_source",
            "candidate_index",
            "native_id_source",
        ],
        "association_fields_added": [
            "public_id_order",
            "candidate_order",
            "public_id_base_score_matrix",
            "public_id_appearance_score_matrix",
            "public_id_fused_score_matrix",
            "assignment_pairs_after_scope",
            "public_id_to_native_tid",
        ],
        "official_response_embedding": "NOT_EXPOSED",
        "machine_feature_fallback": "OSNet frozen market1501 box crop; feature_source=machine_roi_fallback",
        "official_supported_memory_fields": {
            "offload_video_to_cpu": True,
            "offload_output_to_cpu_for_eval": True,
            "nested_offload_state_to_cpu": True,
            "async_loading_frames": False,
            "trim_past_non_cond_mem_for_eval": "present but rejected after reproducible KeyError in repair-1",
        },
        "final_attempted_runtime_policy": {
            "offload_video_to_cpu": True,
            "offload_output_to_cpu_for_eval": True,
            "offload_state_to_cpu": True,
            "trim_past_non_cond_mem_for_eval": False,
            "recondition_every_nth_frame": -1,
            "use_iom_recondition": False,
        },
        "third_party_source_modified": False,
        "real_long_sequence_validation": "BLOCKED",
        "blocking_reason": failure_evidence["blocking_reason"],
    }
    backend_final_path = OUT / "backend_export_audit_final.json"
    atomic_json(backend_final_path, backend_final)

    stage(
        OUT / "stage_03_event_tape_status.json",
        "03_offline_gt_event_tape",
        "NOT_RUN",
        [{"reason": "complete candidate tape unavailable", "sequence_pass": pass_count, "sequence_expected": expected}],
        "repair the real candidate tape before generating simulated_from_gt events",
        [failure_path],
    )
    stage(
        OUT / "stage_04_full_loop_status.json",
        "04_real_full_loop",
        "NOT_RUN",
        [{"reason": "N35 requires at least six complete sequences; only one sequence tape is complete"}],
        "run full loop on at least six complete real sequences after tape gate passes",
        [failure_path],
    )
    stage(
        OUT / "stage_05_paired_replay_status.json",
        "05_m0_m4_paired_replay",
        "NOT_RUN",
        [{"reason": "same-prefix paired replay is not authorized without the complete real tape"}],
        "run M0-M4 paired replay and sequence bootstrap only after stage 02 PASS",
        [failure_path],
    )
    stage(
        OUT / "stage_06_calibration_lora_status.json",
        "06_calibration_and_decoder_lora_gate",
        "NOT_AUTHORIZED",
        [{"reason": "no real M2/M3/M4 versus M0 sequence-cluster evidence; identity feature training remains unauthorized"}],
        "do not train calibration head or decoder LoRA until the real effect gate passes",
        [failure_path, backend_final_path],
    )
    final_gate_path = OUT / "stage_07_final_gate.json"
    atomic_json(
        final_gate_path,
        {
            "stage": "07_final_gate",
            "status": "BLOCKED",
            "real_tape": "NOT_AVAILABLE",
            "real_full_loop": "NOT_RUN",
            "ccam_effect": "NOT_COMPUTABLE",
            "calibration_head": "NOT_AUTHORIZED",
            "decoder_lora": "NOT_AUTHORIZED",
            "sequence_count_expected": expected,
            "sequence_count_pass": pass_count,
            "duplicate": 0,
            "missing_sequences": expected - pass_count,
            "unavailable_sequences": expected - pass_count,
            "partial_sequences": 0,
            "not_run_sequences": not_run_count,
            "oracle_started": False,
            "selector_started": False,
            "blocking_reason": failure_evidence["blocking_reason"],
            "next_action": "fix the adapter-level device-consistent official state-offload/reconditioning path and rerun the complete 24-sequence train/train_fold tape",
            "artifacts": [str(failure_path.resolve()), str(backend_final_path.resolve())],
        },
    )

    report = f"""# N35 Final Report — Real Candidate-Complete Tape Gate

Date: 2026-08-28 (Asia/Shanghai)

## Result

The adapter export implementation and the 4-frame real SAM3 smoke passed. The complete 24-sequence real `train/train_fold` tape did not pass: 1/24 sequences completed (`dancetrack0001`, 703 frames, 8,620 candidates), 4/24 produced preserved failure artifacts, and 19/24 were not started because their shard stopped on the first failure. No GT annotations were read during runtime export.

The candidate tape is therefore not complete. N35 stops before offline event generation, full-loop replay, Oracle, selector, calibration, or decoder LoRA. No result is presented as a CCAM effect.

## Machine-readable evidence

- Failure and retry evidence: `{failure_path.resolve()}`
- Final backend export audit: `{backend_final_path.resolve()}`
- Partial tape manifest: `{(TAPE / 'tape_manifest.json').resolve()}`
- CPU integrity audit: `{(OUT / 'real_tape_integrity_audit.json').resolve()}`
- Stage 02 gate: `{(OUT / 'stage_02_status.json').resolve()}`
- Final gate: `{final_gate_path.resolve()}`

## Memory failures and bounded repairs

1. Initial 4-GPU run used `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`, streaming JSONL output, and one SAM3 backend/session per sequence worker. Four long sequences still hit the official CUDA OOM path; the original shard logs remain preserved.
2. Repair-1 tested the existing official `trim_past_non_cond_mem_for_eval` flag. It raised `KeyError: 'multistep_point_inputs'`, so it was rejected and not used in the final attempt.
3. Repair-2 tested the existing nested tracker `offload_state_to_cpu=True`. It reached official reconditioning and raised a CPU/CUDA device mismatch.
4. Repair-3 kept official state/output/video CPU offload and disabled the official periodic/IOM reconditioning options at the adapter boundary. It still reached official new-object addition and raised a CPU/CUDA device mismatch.

The third-party SAM3 source was not modified. The original OOM/exception evidence is not deleted or relabeled as a pass.

## Gate counts

`sequence_expected={expected}`, `sequence_pass={pass_count}`, `sequence_fail={fail_count}`, `sequence_not_run={not_run_count}`, `duplicate_sequence_artifacts=0`, `complete_tape=False`, `Oracle_started=False`, `selector_started=False`.

The successful partial sequence has 703 rows and 8,620 candidate rows; it passed mask decode, finite 512-D machine feature, complete candidate ordering, score-matrix, assignment, and public-to-native mapping checks. These partial counts are not a 24-sequence PASS.

N35_STATUS = PARTIAL
REAL_TAPE = NOT_AVAILABLE
REAL_FULL_LOOP = NOT_RUN
CCAM_EFFECT = NOT_COMPUTABLE
CALIBRATION_HEAD = NOT_AUTHORIZED
DECODER_LORA = NOT_AUTHORIZED
BLOCKING_REASON = official SAM3 multiplex long-sequence state storage has no adapter-validated device-consistent CPU-offload/reconditioning path on this pinned build; repair budget exhausted after reproducible OOM/KeyError/device-mismatch evidence
NEXT_ACTION = fix the adapter-level device-consistent official state-offload/reconditioning path and rerun the complete 24-sequence train/train_fold tape
"""
    report_path = ROOT / "docs/N35_FINAL_REPORT.md"
    atomic_text(report_path, report)
    print(
        json.dumps(
            {
                "status": "PARTIAL",
                "sequence_expected": expected,
                "sequence_pass": pass_count,
                "sequence_fail": fail_count,
                "sequence_not_run": not_run_count,
                "failure_evidence": str(failure_path.resolve()),
                "final_gate": str(final_gate_path.resolve()),
                "report": str(report_path.resolve()),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
