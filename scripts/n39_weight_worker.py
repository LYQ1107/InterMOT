#!/usr/bin/env python3
"""Run one N39 weight configuration on one frozen N37 event.

The worker is deliberately independent from the supervisor: one invocation
loads one event tape, runs all M0--M4 paired branches, and atomically writes one
compact full per-frame association audit.  Ground truth is never imported or
read in this process.  ``human_weight`` is applied only to the freshly created
AppearanceMemory instance in this worker, leaving the production default and
all frozen N36--N38R1 artifacts untouched.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sam3_intermot.association.appearance_memory import AppearanceMemory
from sam3_intermot.association.ccam_replay import paired_replay, validate_candidate_tape
from sam3_intermot.association.state_manager import StateManager
from scripts.n36_real_eval_common import FEATURE_DIM, atomic_json, variant_config
from scripts.n38r1_sidecar_common import (
    build_event_frame_audit,
    load_manifest_item,
    protocol_hash,
    read_source_rows,
)
from scripts.run_n37_replay import build_runtime_tape


N37_MANIFEST = ROOT / "outputs/n37/real_event_manifest.json"
VARIANTS = ("M0", "M1", "M2", "M3", "M4")
PROTOCOL = "N39_WEIGHTED_ASSOCIATION_INTERFACE_PROBE_WORKER_V1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def compact_candidate(candidate: dict[str, Any]) -> dict[str, Any]:
    """Retain complete association axes while reusing frozen feature sidecars."""
    return {
        "index": int(candidate.get("index", candidate.get("candidate_index", -1))),
        "obs_id": int(candidate.get("obs_id", candidate.get("candidate_obs_id", -1))),
        "native_tid": int(candidate.get("native_tid", candidate.get("candidate_native_id", -1))),
        "native_age": float(candidate.get("native_age", 0.0)),
        "confidence": float(candidate.get("confidence", 1.0)),
        "box": np.asarray(candidate.get("box", [0, 0, 0, 0]), dtype=float).tolist(),
        "has_feat": bool(candidate.get("has_feat", True)),
        "feature_available": bool(candidate.get("feature_available", True)),
    }


def compact_audit(audit: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(audit, dict):
        raise RuntimeError("candidate_audit_missing")
    output = {}
    for key in (
        "frame",
        "public_ids",
        "public_id_order",
        "candidate_order",
        "candidate_native_ids",
        "candidate_public_ids",
        "candidate_public_id_mapping",
        "candidate_public_id_mapping_complete",
        "public_id_to_native_tid",
        "assignment",
        "assignment_after_scope",
        "assignment_pairs",
        "assignment_pairs_after_scope",
        "public_id_score_matrix",
        "public_id_base_score_matrix",
        "public_id_appearance_score_matrix",
        "public_id_fused_score_matrix",
        "scores",
        "base_scores_before_appearance",
        "appearance_memory_scores",
        "appearance_score_deltas",
        "fused_scores",
        "appearance_memory_enabled",
        "human_events",
        "memory_read",
        "memory_write",
        "current_frame_write_hidden",
        "runtime_future_gt_used",
        "candidate_complete",
        "candidate_set_complete",
    ):
        if key in audit:
            value = audit[key]
            if isinstance(value, np.ndarray):
                value = value.tolist()
            output[key] = copy.deepcopy(value)
    output["candidates"] = [
        compact_candidate(item)
        for item in audit.get("candidates", [])
        if isinstance(item, dict)
    ]
    # The raw StateManager candidate log predates the N38R1 enrichment and
    # does not carry an explicit runtime boundary field.  This default is a
    # local schema completion after the tape itself has been validated; it is
    # not inferred from GT and is kept explicit in the worker artifact.
    output["runtime_future_gt_used"] = bool(audit.get("runtime_future_gt_used", False))
    output["gt_loaded_posthoc"] = bool(audit.get("gt_loaded_posthoc", False))
    fused = np.asarray(output.get("fused_scores", []), dtype=float)
    public_ids = [int(value) for value in output.get("public_id_order", [])]
    assignments = output.get("assignment_after_scope", output.get("assignment", []))
    candidate_public_ids = output.get("candidate_public_ids", [])
    target_public_id = None
    cost_audit = output.get("hungarian_cost_audit") or {}
    if fused.ndim == 2 and fused.shape[1] == len(public_ids):
        cost_audit.update(
            {
                "status": "AVAILABLE_DERIVED_FROM_FULL_SCORE_MATRIX",
                "orientation": "candidate_row_x_public_id_state_column",
                "cost_definition": "cost=-fused_score; scipy linear_sum_assignment(-fused_score)",
                "cost_matrix": (-fused).tolist(),
                "row_candidate_public_ids": copy.deepcopy(candidate_public_ids),
                "column_public_ids": public_ids,
                "assignment_after_scope": copy.deepcopy(assignments),
            }
        )
    output["hungarian_cost_audit"] = cost_audit
    output["hungarian_cost_audit"] = {
        key: copy.deepcopy(audit.get("hungarian_cost_audit", {}).get(key))
        for key in (
            "status",
            "orientation",
            "cost_definition",
            "row_candidate_public_ids",
            "row_candidate_native_ids",
            "column_public_ids",
            "assignment_after_scope",
            "target_public_id",
            "target_state_index",
            "target_row",
            "assigned_col",
            "best_alternative_col",
            "assignment_score_margin",
            "assignment_cost_margin",
            "target_row_costs",
        )
        if key in audit.get("hungarian_cost_audit", {})
    }
    return output


def compact_trace(trace: Any, *, memory_write: bool, memory_read: bool) -> list[dict[str, Any]]:
    if not isinstance(trace, list):
        raise RuntimeError("future_trace_missing")
    output = []
    for entry in trace:
        if not isinstance(entry, dict):
            raise RuntimeError("future_trace_entry_invalid")
        audit = compact_audit(entry.get("candidate_audit", {}))
        audit["memory_write"] = bool(memory_write)
        audit["memory_read"] = bool(memory_read)
        audit["current_frame_write_hidden"] = False
        output.append(
            {
                "frame": int(entry["frame"]),
                "rows": [
                    [int(row[0]), np.asarray(row[1], dtype=float).tolist()]
                    for row in entry.get("rows", [])
                    if isinstance(row, (list, tuple)) and len(row) >= 2
                ],
                "candidate_audit": audit,
            }
        )
    return output


def memory_summary(payload: dict[str, Any]) -> dict[str, Any]:
    records = payload.get("records", {})
    return {
        "feat_dim": payload.get("feat_dim"),
        "human_weight": payload.get("human_weight"),
        "machine_weight": payload.get("machine_weight"),
        "decay_frames": payload.get("decay_frames"),
        "record_count": len(records) if isinstance(records, dict) else 0,
        "record_ids": sorted(str(key) for key in records) if isinstance(records, dict) else [],
        "positive_anchor_count": sum(
            len(item.get("positive", []))
            for item in records.values()
            if isinstance(item, dict)
        ) if isinstance(records, dict) else 0,
        "negative_anchor_count": sum(
            len(item.get("negative", []))
            for item in records.values()
            if isinstance(item, dict)
        ) if isinstance(records, dict) else 0,
    }


def install_human_weight(weight: float):
    """Set internal weight only on N39 worker-created memory instances."""
    original_init = StateManager.__init__

    def patched_init(self, config):
        original_init(self, config)
        self.appearance_memory.human_weight = float(weight)

    StateManager.__init__ = patched_init  # type: ignore[method-assign]
    return original_init


def restore_human_weight(original_init) -> None:
    StateManager.__init__ = original_init  # type: ignore[method-assign]


def validate_audit(audit: dict[str, Any], expected_frame: int, event_frame: int) -> None:
    if int(audit.get("frame", -1)) != int(expected_frame):
        raise RuntimeError(f"audit_frame_mismatch:{audit.get('frame')}!={expected_frame}")
    if audit.get("runtime_future_gt_used") is not False:
        raise RuntimeError(f"runtime_future_gt_used_at_frame:{expected_frame}")
    if audit.get("candidate_complete") is not True or audit.get("candidate_set_complete") is not True:
        raise RuntimeError(f"candidate_incomplete_at_frame:{expected_frame}")
    if audit.get("candidate_public_id_mapping_complete") is not True:
        raise RuntimeError(f"candidate_mapping_incomplete_at_frame:{expected_frame}")
    candidates = audit.get("candidates", [])
    candidate_order = audit.get("candidate_order", [])
    if len(candidates) != len(candidate_order):
        raise RuntimeError(f"candidate_order_count_mismatch_at_frame:{expected_frame}")
    matrix_shapes = []
    for key in ("base_scores_before_appearance", "appearance_memory_scores", "appearance_score_deltas", "fused_scores"):
        values = np.asarray(audit.get(key, []), dtype=float)
        if values.ndim != 2 or not np.all(np.isfinite(values)):
            raise RuntimeError(f"invalid_score_matrix:{key}:{expected_frame}")
        matrix_shapes.append(values.shape)
    if len(set(matrix_shapes)) != 1:
        raise RuntimeError(f"score_matrix_shape_mismatch:{expected_frame}:{matrix_shapes}")
    if matrix_shapes[0][0] != len(candidates) or matrix_shapes[0][1] != len(audit.get("public_id_order", [])):
        raise RuntimeError(f"score_axis_mismatch:{expected_frame}:{matrix_shapes[0]}")
    if expected_frame == event_frame:
        if audit.get("memory_read") is not False or audit.get("memory_write") is not False:
            raise RuntimeError("event_frame_memory_visibility_violation")


def run(event_id: str, mode: str, value: float, output: Path) -> dict[str, Any]:
    started = now()
    output.parent.mkdir(parents=True, exist_ok=True)
    manifest, item = load_manifest_item(N37_MANIFEST, event_id)
    event = item["event"]
    event_frame = int(event["frame"])
    tape = build_runtime_tape(item, horizon=100)
    validation = validate_candidate_tape(tape, feat_dim=FEATURE_DIM)
    if not validation["valid"] or not validation["candidate_complete"]:
        raise RuntimeError(f"candidate_tape_validation_failed:{validation}")
    expected_future = list(range(event_frame + 1, min(int(item["sequence_frame_count"]) - 1, event_frame + 100) + 1))
    tape_frames = [int(row["frame"]) for row in tape.get("frames", [])]
    if tape_frames != expected_future:
        raise RuntimeError(f"future_window_mismatch:{tape_frames[:2]}..{tape_frames[-2:]} != {expected_future[:2]}..{expected_future[-2:]}")

    source_rows = read_source_rows(item, event_frame, event_frame)
    configs: dict[str, dict[str, Any]] = {}
    original_init = None
    if mode == "human_weight":
        original_init = install_human_weight(float(value))
    try:
        for variant in VARIANTS:
            config, description = variant_config(variant)
            if mode == "lambda_assoc":
                config.appearance_score_weight = float(value)
            elif mode == "human_weight":
                config.appearance_score_weight = 1.0
            else:
                raise ValueError(f"unsupported_mode:{mode}")
            event_audit = build_event_frame_audit(tape, source_rows[event_frame], item, config)
            validate_audit(event_audit, event_frame, event_frame)
            replay = paired_replay(
                tape,
                config=config,
                feat_dim=FEATURE_DIM,
                write_branch_uses_appearance_memory=(variant != "M0"),
            )
            if replay.get("status") != "PASS":
                raise RuntimeError(f"paired_replay_failed:{variant}:{replay.get('status')}:{replay.get('validation')}")
            compact_branches = {}
            for branch_name in ("memory_write=False", "memory_write=True"):
                branch = replay["branches"][branch_name]
                trace = branch.get("future_trace")
                frames = [int(entry["frame"]) for entry in trace]
                if frames != expected_future:
                    raise RuntimeError(f"future_trace_incomplete:{variant}:{branch_name}")
                actual_write = bool(branch.get("memory_write", False))
                compacted = compact_trace(
                    trace,
                    memory_write=actual_write,
                    memory_read=actual_write,
                )
                for entry in compacted:
                    validate_audit(entry["candidate_audit"], int(entry["frame"]), event_frame)
                    if entry["candidate_audit"].get("memory_write") and int(entry["frame"]) <= event_frame:
                        raise RuntimeError("memory_write_visible_on_event_frame")
                compact_branches[branch_name] = {
                    "memory_write": actual_write,
                    "memory_read": actual_write,
                    "future_trace": compacted,
                    "state_summary": copy.deepcopy(branch.get("state_summary", {})),
                    "appearance_memory": memory_summary(branch.get("appearance_memory", {})),
                }
            no_trace = compact_branches["memory_write=False"]["future_trace"]
            yes_trace = compact_branches["memory_write=True"]["future_trace"]
            if [int(x["frame"]) for x in no_trace] != [int(x["frame"]) for x in yes_trace]:
                raise RuntimeError(f"paired_frame_alignment_failed:{variant}")
            configs[variant] = {
                "description": description,
                "event_frame_audit": {
                    "frame": event_frame,
                    "candidate_audit": compact_audit(event_audit),
                    "audit_only_current_frame": True,
                    "memory_read": False,
                    "memory_write": False,
                    "current_frame_write_hidden": True,
                    "runtime_future_gt_used": False,
                },
                "branches": compact_branches,
                "comparison": copy.deepcopy(replay.get("comparison", [])),
                "status": "PASS",
            }
            del replay, event_audit, config
    finally:
        if original_init is not None:
            restore_human_weight(original_init)

    payload = {
        "protocol": PROTOCOL,
        "status": "PASS",
        "mode": mode,
        "weight_value": float(value),
        "event_id": str(event_id),
        "sequence": str(event["sequence"]),
        "action_type": str(event["action_type"]),
        "event_frame": event_frame,
        "future_frame_start": event_frame + 1,
        "future_frame_end": expected_future[-1] if expected_future else event_frame,
        "future_frame_count": len(expected_future),
        "interaction_source": "simulated_from_gt",
        "synthetic": False,
        "input": {
            "n37_manifest": str(N37_MANIFEST.relative_to(ROOT)),
            "n37_manifest_sha256": sha256(N37_MANIFEST),
            "source_tape": str(item["source_tape"]),
            "source_tape_sha256": item.get("source_tape_sha256"),
            "frozen_n38r1_scale_audit": "outputs/n39/scale_audit_summary.json",
            "frozen_n38_protocol_hash": protocol_hash(),
        },
        "runtime_boundary": {
            "runtime_future_gt_used": False,
            "future_gt_fields_sent": [],
            "gt_loaded_in_worker": False,
            "event_frame_memory_write_hidden": True,
            "first_memory_read_frame": event_frame + 1,
        },
        "candidate_stream_contract": {
            "same_frozen_tape_for_all_variants": True,
            "candidate_complete_required": True,
            "candidate_order_preserved": True,
            "public_id_mapping_audited_per_frame": True,
            "no_duplicate_or_missing_future_frames": True,
        },
        "weight_configuration": {
            "appearance_score_weight": float(value) if mode == "lambda_assoc" else 1.0,
            "appearance_memory_human_weight": float(value) if mode == "human_weight" else 1.0,
            "machine_weight": 0.35,
            "positive_weight": 1.0,
            "negative_weight": 1.0,
        },
        "variants": configs,
        "started_at": started,
        "finished_at": now(),
    }
    atomic_json(output, payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--event-id", required=True)
    parser.add_argument("--mode", choices=("lambda_assoc", "human_weight"), required=True)
    parser.add_argument("--value", type=float, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        payload = run(args.event_id, args.mode, args.value, args.output)
        print(json.dumps({"status": payload["status"], "event_id": payload["event_id"], "mode": payload["mode"], "value": payload["weight_value"], "output": str(args.output)}, sort_keys=True), flush=True)
    except Exception as exc:
        failure = {
            "protocol": PROTOCOL,
            "status": "FAIL",
            "mode": args.mode,
            "weight_value": float(args.value),
            "event_id": args.event_id,
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
            "runtime_future_gt_used": False,
            "artifact_is_failure_evidence": True,
            "failed_at": now(),
        }
        atomic_json(args.output, failure)
        print(json.dumps({"status": "FAIL", "event_id": args.event_id, "error": failure["error"], "output": str(args.output)}, sort_keys=True), flush=True)
        raise


if __name__ == "__main__":
    main()
