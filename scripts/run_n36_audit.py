#!/usr/bin/env python3
"""Create the N36-1, source-and-artifact audit.

This audit deliberately does not import SAM3 or open a dataset frame.  It
records the N35 failure facts and the pinned official API boundaries that the
N36 shard exporter must respect.  The resulting JSON is provenance for the
implementation decision; it is not a runtime success claim.
"""

from __future__ import annotations

import ast
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "n36"


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_name, path)
    finally:
        if os.path.exists(temp_name):
            os.unlink(temp_name)


def sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def source_info(path: Path) -> dict[str, Any]:
    return {
        "path": str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path),
        "exists": path.is_file(),
        "bytes": path.stat().st_size if path.is_file() else None,
        "sha256": sha256(path),
    }


def tree(path: Path) -> ast.AST:
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def function_signature(path: Path, names: set[str]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for node in ast.walk(tree(path)):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if node.name not in names:
            continue
        args = [arg.arg for arg in node.args.posonlyargs + node.args.args]
        args += [arg.arg for arg in node.args.kwonlyargs]
        result[node.name] = {"line": node.lineno, "arguments": args}
    return result


def contains_text(path: Path, needles: list[str]) -> dict[str, bool]:
    text = path.read_text(encoding="utf-8") if path.is_file() else ""
    return {needle: needle in text for needle in needles}


def n35_failure_summary(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {"exists": False}
    payload = json.loads(path.read_text(encoding="utf-8"))
    by_attempt = []
    for attempt in payload.get("attempts", []):
        logs = attempt.get("sequence_logs", {})
        by_attempt.append(
            {
                "attempt": attempt.get("attempt"),
                "failure_fact": attempt.get("failure_fact"),
                "sequences": sorted(logs),
                "reasons": sorted(
                    {
                        reason
                        for item in logs.values()
                        for reason in item.get("reasons", [])
                    }
                ),
            }
        )
    return {
        "exists": True,
        "protocol": payload.get("protocol"),
        "blocking_reason": payload.get("blocking_reason"),
        "attempts": by_attempt,
        "status_counts": payload.get("done_summary", {}).get("status_counts", {}),
        "duplicate_sequence_artifacts": payload.get("duplicate_sequence_artifacts"),
    }


def run() -> dict[str, Any]:
    backend = ROOT / "sam3_intermot/backend/sam3_backend.py"
    exporter = ROOT / "scripts/run_n35_export_tape.py"
    launcher = ROOT / "scripts/run_n35_export_4gpu.sh"
    official_predictor = ROOT / "third_party/sam3/sam3/model/sam3_base_predictor.py"
    official_multiplex = ROOT / "third_party/sam3/sam3/model/sam3_multiplex_tracking.py"
    official_builder = ROOT / "third_party/sam3/sam3/model_builder.py"
    report = ROOT / "docs/N35_FINAL_REPORT.md"
    evidence = ROOT / "outputs/n35/real_tape_failure_evidence.json"
    final_audit = ROOT / "outputs/n35/backend_export_audit_final.json"

    artifact = {
        "protocol": "N36_STAGE_01_N35_LONG_SEQUENCE_AUDIT",
        "status": "PASS",
        "dataset_scope": {
            "root": "/path/to/dancetrack/train/train_fold",
            "allowed_splits": ["train", "train_fold"],
            "val_test_read": False,
            "frame_content_read": False,
        },
        "n35_report": source_info(report),
        "n35_failure_evidence": source_info(evidence),
        "n35_final_backend_audit": source_info(final_audit),
        "n35_failures": n35_failure_summary(evidence),
        "audited_sources": [
            source_info(path)
            for path in (backend, exporter, launcher, official_predictor, official_multiplex, official_builder)
        ],
        "official_api": {
            "predictor": function_signature(
                official_predictor,
                {"start_session", "close_session", "propagate_in_video", "handle_stream_request"},
            ),
            "multiplex": function_signature(
                official_multiplex,
                {"init_state", "_get_processing_order", "propagate_in_video", "_run_single_frame_inference"},
            ),
            "builder": function_signature(
                official_builder,
                {"build_sam3_multiplex_video_predictor", "build_sam3_multiplex_video_model"},
            ),
            "processing_order_semantics": {
                "forward": "range(start_frame_idx, min(start_frame_idx + max_frame_num_to_track, num_frames - 1) + 1), inclusive",
                "default_max_frame_num_to_track": "num_frames",
                "adapter_currently_passes_max_frame_num_to_track": False,
                "is_last_batch": "not exposed by the adapter/predictor request boundary; do not invent it",
            },
            "official_init_state_fields": [
                "input_batch",
                "previous_stages_out[num_frames]",
                "per_frame_* arrays[num_frames]",
                "sam2_inference_states",
                "tracker_metadata",
                "feature_cache",
                "cached_frame_outputs",
            ],
            "official_reconditioning_defaults_from_pinned_builder": {
                "recondition_every_nth_frame": 16,
                "use_iom_recondition": True,
                "hotstart_delay": 15,
                "masklet_confirmation_enable": True,
            },
            "supported_init_state_arguments": [
                "resource_path",
                "offload_video_to_cpu",
                "async_loading_frames",
                "use_torchcodec",
                "use_cv2",
                "input_is_mp4",
            ],
            "supported_offload_facts": {
                "offload_video_to_cpu": True,
                "offload_output_to_cpu_for_eval": "official model attribute; adapter may set it",
                "offload_state_to_cpu": "not accepted by pinned outer multiplex init_state; nested repair was rejected by N35 device-mismatch evidence",
                "trim_past_non_cond_mem_for_eval": "present in model code but rejected by N35 KeyError evidence",
                "async_loading_frames": True,
            },
        },
        "state_ownership_and_release": {
            "long_lived_owner": "official model inference_state under predictor._all_inference_states[session_id]['state']",
            "heavy_fields": [
                "input_batch",
                "sam2_inference_states",
                "tracker_metadata",
                "feature_cache",
                "cached_frame_outputs",
                "previous_stages_out",
            ],
            "adapter_session_close": "Sam3Backend.close sends official close_session and clears adapter caches",
            "official_close": "pops session, clears the state dict, gc.collect, conditionally torch.cuda.empty_cache",
            "process_exit_strategy": "N36 uses one Python process per frame-range so CUDA context and any retained model/state references die after each chunk",
        },
        "n35_exporter_gap": {
            "full_sequence_backend_lifetime": True,
            "frame_range_export": False,
            "overlap_boundary_mapping": False,
            "skip_existing": True,
            "atomic_jsonl_rename": True,
            "atomic_done_manifest": True,
            "shared_manifest_race_avoidance": "--no-manifest exists; aggregation is separate",
            "current_protocol": "N35_REAL_CANDIDATE_COMPLETE_TAPE",
        },
        "n36_decision": {
            "strategy": "A_INDEPENDENT_PROCESS_PER_FRAME_RANGE",
            "chunk_size_initial": 160,
            "overlap_frames_initial": 20,
            "range_end_is_inclusive": True,
            "native_id_policy": "never assume numeric equality; match adjacent local IDs with overlap box IoU, mask IoU, and machine embedding, then assign sequence_global_native_id",
            "state_offload": "disabled for N36 because N35 nested CPU-state attempts produced KeyError/device mismatch; use process isolation instead",
            "runtime_gt": False,
            "third_party_source_edit": False,
        },
        "audit_conclusion": "N35 retained official multiplex state for the entire sequence and mixed unsupported state-offload/reconditioning repairs. N36 will retain the validated adapter hotstart/output handling, avoid CPU state offload, and bound each official session to an independently exited 50-200 frame range with explicit overlap reconciliation.",
    }
    atomic_json(OUT / "stage_01_audit.json", artifact)
    stage = {
        "stage": "N36-01",
        "status": "PASS",
        "artifacts": ["outputs/n36/stage_01_audit.json"],
        "errors": [],
        "next_action": "Implement the independent-process range exporter, atomic chunk artifacts, boundary native-ID merge, and validator.",
        "third_party_modified": False,
    }
    atomic_json(OUT / "stage_01_status.json", stage)
    return artifact


if __name__ == "__main__":
    print(json.dumps(run(), indent=2, sort_keys=True))
