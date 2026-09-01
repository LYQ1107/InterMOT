#!/usr/bin/env python3
"""Post-hoc GT scoring for the completed N42 T0/T1 replay.

The worker artifacts are validated completely before this module constructs a
DanceTrack dataset or reads any GT.  Runtime replay remains GT-free; labels
are used only below the explicit post-hoc boundary.  This is an isolated
association-interface probe over frozen N41 candidate/state audits, not a
claim that the production online StateManager was changed.
"""

from __future__ import annotations

import argparse
import copy
import gc
import hashlib
import json
import math
import sys
import traceback
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sam3_intermot.datasets.dancetrack import DanceTrackDataset
from scripts.n36_real_eval_common import (
    DATA_ROOT,
    HORIZONS,
    atomic_json,
    evaluate_trace,
    finite_iou,
)
from scripts.run_n36_replay import (
    BOOTSTRAP_REPLICATES,
    BOOTSTRAP_SEED,
    cluster_bootstrap,
    event_variant_summary,
    finite,
    protected_regression,
)
from scripts.run_n37_replay import add_identity_error_aliases


N37_MANIFEST = ROOT / "outputs/n37/real_event_manifest.json"
TRAINING_PROTOCOL = ROOT / "outputs/n42/training/training_protocol.json"
REPLAY_DIR = ROOT / "outputs/n42/replay"
STAGE = ROOT / "outputs/n42/stage_03_status.json"
ATTEMPTS = ROOT / "outputs/n42/attempts"
OUT = REPLAY_DIR / "posthoc_events"
RESULT = REPLAY_DIR / "posthoc_results.json"
VARIANTS = ("M0", "M1", "M2", "M3", "M4")
MODES = ("t0", "t1")
PROTOCOL = "N42_T0_T1_POSTHOC_REPLAY_V1"


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
        raise ValueError(f"expected JSON object: {path}")
    return value


def stream_signature(audit: dict[str, Any]) -> tuple[Any, ...]:
    """Signature of detector/native stream, excluding assignment outcomes."""
    candidates = audit.get("candidates", [])
    candidate_values = []
    for index, candidate in enumerate(candidates):
        if not isinstance(candidate, dict):
            candidate_values.append((index, "INVALID"))
            continue
        box = tuple(float(value) for value in candidate.get("box", []))
        candidate_values.append(
            (
                int(candidate.get("index", index)),
                int(candidate.get("obs_id", index)),
                int(candidate.get("native_tid", -1)),
                box,
                float(candidate.get("confidence", 1.0)),
                float(candidate.get("native_age", 0.0)),
            )
        )
    mapping = []
    for row in audit.get("candidate_public_id_mapping", []):
        if not isinstance(row, dict):
            mapping.append(("INVALID",))
            continue
        mapping.append(
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
        tuple(candidate_values),
        tuple(mapping),
    )


def finite_score_matrices(audit: dict[str, Any]) -> tuple[bool, tuple[int, int] | None]:
    shapes = []
    for key in (
        "base_scores_before_appearance",
        "appearance_memory_scores",
        "appearance_score_deltas",
        "fused_scores",
    ):
        values = np.asarray(audit.get(key, []), dtype=float)
        if values.ndim != 2 or not np.all(np.isfinite(values)):
            return False, None
        shapes.append(values.shape)
    if len(set(shapes)) != 1:
        return False, None
    shape = shapes[0]
    if shape[0] != len(audit.get("candidates", [])):
        return False, shape
    if shape[1] != len(audit.get("public_id_order", [])):
        return False, shape
    return True, shape


def validate_audit(
    audit: dict[str, Any],
    *,
    expected_frame: int,
    event_frame: int,
    branch: str,
    variant: str,
    mode: str,
) -> list[str]:
    issues: list[str] = []
    frame = int(audit.get("frame", -1))
    if frame != expected_frame:
        issues.append(f"frame:{frame}!={expected_frame}")
    if bool(audit.get("is_event_frame")) != (frame == event_frame):
        issues.append("is_event_frame_mismatch")
    if bool(audit.get("is_future_frame")) != (frame > event_frame):
        issues.append("is_future_frame_mismatch")
    if audit.get("runtime_future_gt_used") is not False:
        issues.append("runtime_future_gt_used")
    if audit.get("gt_loaded_posthoc") is not False:
        issues.append("gt_loaded_posthoc_not_false_in_runtime_artifact")
    if audit.get("candidate_complete") is not True:
        issues.append("candidate_complete")
    if audit.get("candidate_set_complete") is not True:
        issues.append("candidate_set_complete")
    if audit.get("candidate_public_id_mapping_complete") is not True:
        issues.append("candidate_public_id_mapping_complete")
    finite, shape = finite_score_matrices(audit)
    if not finite:
        issues.append("score_matrices_not_finite_or_shape_mismatch")
    candidates = audit.get("candidates", [])
    if not isinstance(candidates, list):
        issues.append("candidates_not_list")
        candidates = []
    if len(audit.get("candidate_native_ids", [])) != len(candidates):
        issues.append("candidate_native_id_count")
    if len(audit.get("candidate_public_ids", [])) != len(candidates):
        issues.append("candidate_public_id_count")
    if shape is not None and len(audit.get("assignment_after_scope", [])) != shape[0]:
        issues.append("assignment_count")
    if frame == event_frame:
        if audit.get("memory_read") is not False:
            issues.append("event_frame_memory_read")
        if audit.get("current_frame_write_hidden") is not True:
            issues.append("event_frame_write_not_hidden")
        if audit.get("t1_calibration", {}).get("applied") is not False:
            issues.append("event_frame_t1_calibration_applied")
    expected_write = branch == "memory_write=True" and variant != "M0"
    if bool(audit.get("memory_write")) != expected_write:
        issues.append(f"memory_write:{audit.get('memory_write')}!={expected_write}")
    if bool(audit.get("memory_read")) != expected_write:
        issues.append(f"memory_read:{audit.get('memory_read')}!={expected_write}")
    if mode == "t0" and audit.get("t1_calibration", {}).get("applied") is True:
        issues.append("t1_applied_in_t0")
    calibration = audit.get("t1_calibration")
    if isinstance(calibration, dict) and "runtime_future_gt_used" in calibration:
        if calibration.get("runtime_future_gt_used") is not False:
            issues.append("t1_calibration_runtime_gt")
    return issues


def load_event_map() -> dict[str, dict[str, Any]]:
    payload = load_json(N37_MANIFEST)
    events = payload.get("events", [])
    if payload.get("status") != "PASS" or payload.get("event_count") != 24 or len(events) != 24:
        raise RuntimeError("N37 frozen event manifest is not PASS/24")
    output: dict[str, dict[str, Any]] = {}
    for item in events:
        event = item.get("event", {})
        event_id = str(event.get("event_id"))
        if event_id in output:
            raise RuntimeError(f"duplicate frozen event: {event_id}")
        output[event_id] = item
    if len({str(item["event"]["sequence"]) for item in output.values()}) != 21:
        raise RuntimeError("frozen event manifest is not 21 independent sequences")
    return output


def validate_artifact(path: Path, event: dict[str, Any], mode: str) -> dict[str, Any]:
    payload = load_json(path)
    issues: list[str] = []
    event_id = str(event["event_id"])
    event_frame = int(event["frame"])
    if payload.get("status") != "PASS":
        issues.append(f"status:{payload.get('status')}")
    if str(payload.get("event_id")) != event_id:
        issues.append("event_id")
    if str(payload.get("sequence")) != str(event["sequence"]):
        issues.append("sequence")
    if int(payload.get("event_frame", -1)) != event_frame:
        issues.append("event_frame")
    if payload.get("runtime_boundary", {}).get("runtime_future_gt_used") is not False:
        issues.append("artifact_runtime_future_gt_used")
    variants = payload.get("variants")
    if not isinstance(variants, dict) or set(variants) != set(VARIANTS):
        issues.append("variant_key_set")
        variants = variants if isinstance(variants, dict) else {}
    expected_count = int(payload.get("future_frame_count", -1))
    future_end = int(payload.get("future_frame_end", -1))
    signatures: list[tuple[Any, ...]] = []
    branch_counts: dict[str, int] = {}
    for variant in VARIANTS:
        variant_payload = variants.get(variant, {})
        if not isinstance(variant_payload, dict) or variant_payload.get("status") != "PASS":
            issues.append(f"variant_not_pass:{variant}")
            continue
        event_wrapper = variant_payload.get("event_frame_audit", {})
        event_audit = dict(event_wrapper.get("candidate_audit", {}))
        # The event-frame record is a wrapper: causal/t1 fields are stored at
        # the wrapper level while score/candidate fields are nested.  Merge
        # the wrapper metadata into a validation view without changing the
        # frozen artifact.
        for metadata_key in (
            "current_frame_write_hidden",
            "frame",
            "gt_loaded_posthoc",
            "is_event_frame",
            "is_future_frame",
            "memory_read",
            "memory_write",
            "runtime_future_gt_used",
            "t1_calibration",
        ):
            if metadata_key in event_wrapper:
                event_audit[metadata_key] = event_wrapper[metadata_key]
        issues.extend(
            f"{variant}/event:{reason}"
            for reason in validate_audit(
                event_audit,
                expected_frame=event_frame,
                event_frame=event_frame,
                branch="memory_write=False",
                variant="M0",
                mode=mode,
            )
        )
        if event_audit.get("memory_write") is not False:
            issues.append(f"{variant}/event:memory_write")
        for branch in ("memory_write=False", "memory_write=True"):
            branch_payload = variant_payload.get("branches", {}).get(branch)
            if not isinstance(branch_payload, dict):
                issues.append(f"{variant}/{branch}:missing")
                continue
            trace = branch_payload.get("future_trace", [])
            if not isinstance(trace, list) or len(trace) != expected_count:
                issues.append(f"{variant}/{branch}:trace_count")
                continue
            branch_counts[branch] = len(trace)
            if bool(branch_payload.get("memory_write")) != (branch == "memory_write=True" and variant != "M0"):
                issues.append(f"{variant}/{branch}:branch_memory_write")
            if bool(branch_payload.get("memory_read")) != (branch == "memory_write=True" and variant != "M0"):
                issues.append(f"{variant}/{branch}:branch_memory_read")
            previous = None
            branch_signature = None
            for offset, entry in enumerate(trace, start=1):
                frame = int(entry.get("frame", -1))
                expected = event_frame + offset
                if frame != expected:
                    issues.append(f"{variant}/{branch}:frame_{frame}_expected_{expected}")
                if previous is not None and frame != previous + 1:
                    issues.append(f"{variant}/{branch}:non_contiguous")
                previous = frame
                audit = entry.get("candidate_audit", {})
                issues.extend(
                    f"{variant}/{branch}/frame_{frame}:{reason}"
                    for reason in validate_audit(
                        audit,
                        expected_frame=frame,
                        event_frame=event_frame,
                        branch=branch,
                        variant=variant,
                        mode=mode,
                    )
                )
                current_signature = stream_signature(audit)
                if branch_signature is None:
                    branch_signature = current_signature
                elif current_signature != branch_signature:
                    # A real detector stream may change per frame; this check
                    # is intentionally per corresponding branch/variant only.
                    pass
            if trace:
                if int(trace[-1]["frame"]) != future_end:
                    issues.append(f"{variant}/{branch}:future_end")
                signatures.append(stream_signature(trace[0]["candidate_audit"]))
    if expected_count != 100:
        issues.append(f"future_frame_count:{expected_count}")
    if expected_count > 0 and future_end != event_frame + expected_count:
        issues.append("future_end_count_relation")
    if len(signatures) > 1 and any(value != signatures[0] for value in signatures[1:]):
        issues.append("same-event initial candidate stream differs across variants/branches")
    return {
        "status": "PASS" if not issues else "FAIL",
        "path": str(path.relative_to(ROOT)),
        "event_id": event_id,
        "mode": mode,
        "future_frame_count": expected_count,
        "future_frame_end": future_end,
        "issues": issues,
        "sha256": sha256(path),
    }


def load_runtime_paths(event_map: dict[str, dict[str, Any]]) -> tuple[dict[str, dict[str, Path]], dict[str, Any]]:
    paths: dict[str, dict[str, Path]] = {mode: {} for mode in MODES}
    audit_rows = []
    for mode in MODES:
        manifest_path = REPLAY_DIR / f"runtime_{mode}_manifest.json"
        manifest = load_json(manifest_path)
        expected = {
            "status": "PASS",
            "mode": mode,
            "expected_event_count": 24,
            "completed_event_count": 24,
            "runtime_future_gt_used": False,
        }
        for key, value in expected.items():
            if manifest.get(key) != value:
                raise RuntimeError(f"runtime manifest gate failed {mode}/{key}: {manifest.get(key)!r} != {value!r}")
        if manifest.get("failures"):
            raise RuntimeError(f"runtime manifest contains failures: {mode}")
        records = manifest.get("worker_records", [])
        if not isinstance(records, list) or len(records) != 24:
            raise RuntimeError(f"runtime worker count invalid: {mode}")
        for record in records:
            event_id = str(record.get("event_id"))
            if event_id in paths[mode] or event_id not in event_map:
                raise RuntimeError(f"runtime event key invalid/duplicate: {mode}/{event_id}")
            if record.get("returncode") != 0 or record.get("artifact_status") != "PASS":
                raise RuntimeError(f"runtime worker not PASS: {mode}/{event_id}")
            path = ROOT / str(record["output"])
            if not path.is_file():
                raise FileNotFoundError(path)
            check = validate_artifact(path, event_map[event_id]["event"], mode)
            audit_rows.append(check)
            if check["status"] != "PASS":
                raise RuntimeError(f"runtime artifact validation failed: {check}")
            paths[mode][event_id] = path
    if set(paths["t0"]) != set(event_map) or set(paths["t1"]) != set(event_map):
        raise RuntimeError("runtime event key coverage is not exactly the frozen 24 events")

    # This comparison is still pre-GT: it checks that T1 changed only the
    # isolated score/assignment interface and retained the frozen stream.
    cross_mode = []
    for event_id in sorted(event_map):
        t0 = load_json(paths["t0"][event_id])
        t1 = load_json(paths["t1"][event_id])
        for variant in VARIANTS:
            for branch in ("memory_write=False", "memory_write=True"):
                left = t0["variants"][variant]["branches"][branch]["future_trace"]
                right = t1["variants"][variant]["branches"][branch]["future_trace"]
                if len(left) != len(right):
                    raise RuntimeError(f"T0/T1 trace length mismatch: {event_id}/{variant}/{branch}")
                for left_entry, right_entry in zip(left, right):
                    left_audit = left_entry["candidate_audit"]
                    right_audit = right_entry["candidate_audit"]
                    if stream_signature(left_audit) != stream_signature(right_audit):
                        raise RuntimeError(f"T0/T1 candidate stream changed: {event_id}/{variant}/{branch}/{left_entry['frame']}")
                    if left_entry["frame"] != right_entry["frame"]:
                        raise RuntimeError(f"T0/T1 frame changed: {event_id}/{variant}/{branch}")
                cross_mode.append({"event_id": event_id, "variant": variant, "branch": branch, "stream_unchanged": True})
        del t0, t1
        gc.collect()
    return paths, {
        "status": "PASS",
        "manifest_modes": list(MODES),
        "artifact_count": len(audit_rows),
        "event_count": 24,
        "variant_count": 5,
        "cross_mode_stream_checks": len(cross_mode),
        "cross_mode_candidate_stream_unchanged": True,
        "runtime_future_gt_used": False,
        "gt_loaded_before_validation": False,
        "artifact_checks": audit_rows,
    }


def build_comparison(variant_payload: dict[str, Any]) -> list[dict[str, Any]]:
    left_trace = variant_payload["branches"]["memory_write=False"]["future_trace"]
    right_trace = variant_payload["branches"]["memory_write=True"]["future_trace"]
    output = []
    for left, right in zip(left_trace, right_trace):
        left_audit = left["candidate_audit"]
        right_audit = right["candidate_audit"]
        left_scores = np.asarray(left_audit["fused_scores"], dtype=float)
        right_scores = np.asarray(right_audit["fused_scores"], dtype=float)
        left_native = [int(value) for value in left_audit["candidate_native_ids"]]
        right_native = [int(value) for value in right_audit["candidate_native_ids"]]
        left_states = [int(value) for value in left_audit["public_id_order"]]
        right_states = [int(value) for value in right_audit["public_id_order"]]
        left_map = {
            (native, pid): float(left_scores[i, j])
            for i, native in enumerate(left_native)
            for j, pid in enumerate(left_states)
        }
        right_map = {
            (native, pid): float(right_scores[i, j])
            for i, native in enumerate(right_native)
            for j, pid in enumerate(right_states)
        }
        common = sorted(set(left_map) & set(right_map))
        deltas = [right_map[key] - left_map[key] for key in common]
        left_assignment = {
            int(native): (left_states[int(column)] if 0 <= int(column) < len(left_states) else None)
            for native, column in zip(left_native, left_audit.get("assignment_after_scope", []))
        }
        right_assignment = {
            int(native): (right_states[int(column)] if 0 <= int(column) < len(right_states) else None)
            for native, column in zip(right_native, right_audit.get("assignment_after_scope", []))
        }
        output.append(
            {
                "frame": int(left["frame"]),
                "score_shape_equal": bool(left_scores.shape == right_scores.shape),
                "aligned_score_pair_count": len(common),
                "max_abs_score_delta": float(max((abs(value) for value in deltas), default=0.0)),
                "score_changed": bool(any(abs(value) > 1.0e-12 for value in deltas)),
                "assignment_changed": bool(left_assignment != right_assignment),
                "no_write_assignment_by_native": left_assignment,
                "write_assignment_by_native": right_assignment,
                "candidate_stream_signature_equal": stream_signature(left_audit) == stream_signature(right_audit),
            }
        )
    if len(output) != len(left_trace):
        raise RuntimeError("paired comparison trace length mismatch")
    return output


def target_iou(trace_entry: dict[str, Any], event: dict[str, Any], gt_frames: dict[int, Any]) -> float | None:
    gt = gt_frames.get(int(trace_entry["frame"]))
    if gt is None:
        return None
    target_box = None
    for gid, box in zip(gt.gt_ids, gt.boxes):
        if int(gid) == int(event["dataset_gt_id"]):
            target_box = box
            break
    if target_box is None:
        return None
    target_pid = int(event["public_id"])
    return max(
        (finite_iou(row[1], target_box) for row in trace_entry.get("rows", []) if int(row[0]) == target_pid),
        default=0.0,
    )


def transition_diagnostics(
    comparison: list[dict[str, Any]],
    no_trace: list[dict[str, Any]],
    yes_trace: list[dict[str, Any]],
    event: dict[str, Any],
    gt_frames: dict[int, Any],
) -> dict[str, Any]:
    by_frame = {int(row["frame"]): row for row in comparison}
    output: dict[str, Any] = {}
    for horizon in HORIZONS:
        rows = []
        for index in range(min(int(horizon), len(no_trace), len(yes_trace))):
            left, right = no_trace[index], yes_trace[index]
            frame = int(left["frame"])
            audit = by_frame[frame]
            left_iou = target_iou(left, event, gt_frames)
            right_iou = target_iou(right, event, gt_frames)
            changed = bool(audit["assignment_changed"])
            rows.append(
                {
                    "frame": frame,
                    "score_changed": bool(audit["score_changed"]),
                    "assignment_changed": changed,
                    "correct_assignment_change": bool(changed and left_iou is not None and right_iou is not None and right_iou > left_iou + 1.0e-9),
                    "incorrect_assignment_change": bool(changed and left_iou is not None and right_iou is not None and right_iou < left_iou - 1.0e-9),
                    "target_iou_no_write": left_iou,
                    "target_iou_write": right_iou,
                    "max_abs_score_delta": audit["max_abs_score_delta"],
                    "aligned_score_pair_count": audit["aligned_score_pair_count"],
                }
            )
        score_count = sum(int(row["score_changed"]) for row in rows)
        assignment_count = sum(int(row["assignment_changed"]) for row in rows)
        correct_count = sum(int(row["correct_assignment_change"]) for row in rows)
        incorrect_count = sum(int(row["incorrect_assignment_change"]) for row in rows)
        output[str(horizon)] = {
            "evaluated_future_frames": len(rows),
            "score_changed_count": score_count,
            "score_change_rate": float(score_count / len(rows)) if rows else None,
            "assignment_changed_count": assignment_count,
            "assignment_change_rate": float(assignment_count / len(rows)) if rows else None,
            "correct_assignment_change_count": correct_count,
            "incorrect_assignment_change_count": incorrect_count,
            "correct_assignment_change_rate": float(correct_count / len(rows)) if rows else None,
            "incorrect_assignment_change_rate": float(incorrect_count / len(rows)) if rows else None,
            "score_changed_without_assignment_change_count": sum(int(row["score_changed"] and not row["assignment_changed"]) for row in rows),
            "assignment_changed_without_aligned_score_count": sum(int(row["assignment_changed"] and row["aligned_score_pair_count"] == 0) for row in rows),
            "frame_details": rows,
        }
    return output


def split_map(protocol: dict[str, Any]) -> dict[str, str]:
    output = {}
    for name in ("train", "validation", "holdout"):
        for sequence in protocol.get("sequence_split", {}).get(name, []):
            if sequence in output:
                raise RuntimeError(f"sequence appears in multiple splits: {sequence}")
            output[str(sequence)] = name
    return output


def mean_finite(values: list[Any]) -> float | None:
    clean = [float(value) for value in values if finite(value)]
    return float(np.mean(clean)) if clean else None


def transition_aggregate(rows: list[dict[str, Any]], horizon: int) -> dict[str, Any]:
    fields = (
        "score_changed_count",
        "assignment_changed_count",
        "correct_assignment_change_count",
        "incorrect_assignment_change_count",
        "score_changed_without_assignment_change_count",
        "assignment_changed_without_aligned_score_count",
    )
    selected = [row["transition_diagnostics"][str(horizon)] for row in rows]
    counts = {field: sum(int(row[field]) for row in selected) for field in fields}
    total = sum(int(row["evaluated_future_frames"]) for row in selected)
    return {
        **counts,
        "event_count": len(selected),
        "evaluated_future_frames": total,
        "score_change_rate": float(counts["score_changed_count"] / total) if total else None,
        "assignment_change_rate": float(counts["assignment_changed_count"] / total) if total else None,
        "correct_assignment_change_rate": float(counts["correct_assignment_change_count"] / total) if total else None,
        "incorrect_assignment_change_rate": float(counts["incorrect_assignment_change_count"] / total) if total else None,
    }


def metrics_aggregate(rows: list[dict[str, Any]], horizon: int) -> dict[str, Any]:
    delta_rows = [row["horizon_deltas"][str(horizon)] for row in rows]
    no_metrics = [row["no_write_metrics"]["horizons"][str(horizon)] for row in rows]
    yes_metrics = [row["write_metrics"]["horizons"][str(horizon)] for row in rows]
    return {
        "event_count": len(rows),
        "no_write_target_mean_iou": mean_finite([row.get("target_mean_iou") for row in no_metrics]),
        "write_target_mean_iou": mean_finite([row.get("target_mean_iou") for row in yes_metrics]),
        "no_write_target_identity_error_rate": mean_finite([row.get("target_identity_error_rate") for row in no_metrics]),
        "write_target_identity_error_rate": mean_finite([row.get("target_identity_error_rate") for row in yes_metrics]),
        "target_iou_delta_write_minus_no_write": mean_finite([row.get("target_iou_delta_write_minus_no_write") for row in delta_rows]),
        "target_missing_rate_reduction_no_write_minus_write": mean_finite([row.get("target_missing_rate_reduction_no_write_minus_write") for row in delta_rows]),
        "identity_utility_delta": mean_finite([row.get("identity_utility_delta") for row in delta_rows]),
        "id_switch_reduction": mean_finite([row.get("id_switch_reduction") for row in delta_rows]),
        "recorrection_opportunity_reduction": mean_finite([row.get("posthoc_recorrection_opportunity_reduction") for row in delta_rows]),
        "visible_frames_no_write": sum(int(row.get("visible_frames", 0)) for row in no_metrics),
        "visible_frames_write": sum(int(row.get("visible_frames", 0)) for row in yes_metrics),
    }


def protected_summary(rows: list[dict[str, Any]], event_map: dict[str, dict[str, Any]], horizon: int) -> dict[str, Any]:
    checks = []
    for row in rows:
        event = event_map[row["event_id"]]["event"]
        check = protected_regression(
            row["no_write_metrics"], row["write_metrics"], event, horizon=horizon
        )
        checks.append(check)
    return {
        "event_count": len(checks),
        "all_no_obvious_regression": bool(checks) and all(row["no_obvious_regression"] for row in checks),
        "regression_event_count": sum(int(not row["no_obvious_regression"]) for row in checks),
        "total_regression_identity_count": sum(int(row["regression_count"]) for row in checks),
        "compared_untouched_gt_count": sum(int(row["compared_untouched_gt_count"]) for row in checks),
        "mean_untouched_utility_delta": mean_finite([row.get("mean_untouched_utility_delta") for row in checks]),
        "per_event": checks,
    }


def variant_aggregate(
    rows: list[dict[str, Any]],
    event_map: dict[str, dict[str, Any]],
    sequence_split: dict[str, str],
    mode: str,
    variant: str,
) -> dict[str, Any]:
    by_action = {
        action: [row for row in rows if row["action_type"] == action]
        for action in ("ADD_NEW_IDENTITY", "AUTHORITATIVE_REASSIGN", "ATOMIC_ID_SWAP", "RECOVER_IDENTITY")
    }
    by_split = {
        split: [row for row in rows if sequence_split.get(str(row["sequence"])) == split]
        for split in ("train", "validation", "holdout")
    }
    all_bootstrap = {str(h): cluster_bootstrap(rows, h) for h in HORIZONS}
    split_bootstrap = {
        split: {str(h): cluster_bootstrap(split_rows, h) for h in HORIZONS}
        for split, split_rows in by_split.items()
    }
    actions = {}
    for action, action_rows in by_action.items():
        actions[action] = {
            "event_count": len(action_rows),
            "independent_sequence_count": len({str(row["sequence"]) for row in action_rows}),
            "metrics": {str(h): metrics_aggregate(action_rows, h) for h in HORIZONS},
            "transition": {str(h): transition_aggregate(action_rows, h) for h in HORIZONS},
        }
    splits = {}
    for split, split_rows in by_split.items():
        splits[split] = {
            "event_count": len(split_rows),
            "independent_sequence_count": len({str(row["sequence"]) for row in split_rows}),
            "metrics": {str(h): metrics_aggregate(split_rows, h) for h in HORIZONS},
            "transition": {str(h): transition_aggregate(split_rows, h) for h in HORIZONS},
            "protected_regression": {str(h): protected_summary(split_rows, event_map, h) for h in HORIZONS},
            "sequence_cluster_bootstrap": split_bootstrap[split],
        }
    protected = {str(h): protected_summary(rows, event_map, h) for h in HORIZONS}
    holdout = by_split["holdout"]
    holdout_checks = {
        str(h): {
            "lower_ci_gt_zero": finite(split_bootstrap["holdout"][str(h)].get("lower")) and float(split_bootstrap["holdout"][str(h)]["lower"]) > 0.0,
            "lower_ci": split_bootstrap["holdout"][str(h)].get("lower"),
            "protected_no_obvious_regression": protected_summary(holdout, event_map, h)["all_no_obvious_regression"],
        }
        for h in HORIZONS
    }
    gate_checks = {
        "mode_is_t1": mode == "t1",
        "variant_is_gate_variant": variant in ("M2", "M3", "M4"),
        "holdout_event_count": len(holdout),
        "holdout_sequence_count": len({str(row["sequence"]) for row in holdout}),
        "holdout_h20_h50_h100_lower_ci_strictly_gt_zero": all(row["lower_ci_gt_zero"] for row in holdout_checks.values()),
        "holdout_protected_no_obvious_regression_all_horizons": all(row["protected_no_obvious_regression"] for row in holdout_checks.values()),
    }
    gate_pass = bool(all(gate_checks.values()))
    return {
        "mode": mode,
        "variant": variant,
        "status": "PASS" if len(rows) == 24 else "FAIL",
        "event_count": len(rows),
        "independent_sequence_count": len({str(row["sequence"]) for row in rows}),
        "metrics": {str(h): metrics_aggregate(rows, h) for h in HORIZONS},
        "transition": {str(h): transition_aggregate(rows, h) for h in HORIZONS},
        "sequence_cluster_bootstrap": all_bootstrap,
        "protected_regression": protected,
        "actions": actions,
        "splits": splits,
        "holdout_gate": {
            "status": "PASS" if gate_pass else "FAIL_FUTURE_EFFECT",
            "checks": gate_checks,
            "per_horizon": holdout_checks,
            "strict_requirement": "T1 M2/M3/M4 holdout sequence-cluster lower CI > 0 at H20/H50/H100 and no untouched-ID regression",
        },
        "future_effect_gate_not_authorized_by_default": True,
    }


def run() -> dict[str, Any]:
    started = now()
    event_map = load_event_map()
    training_protocol = load_json(TRAINING_PROTOCOL)
    sequence_split = split_map(training_protocol)
    if set(sequence_split) != {str(item["event"]["sequence"]) for item in event_map.values()}:
        raise RuntimeError("training sequence split does not cover exactly the frozen event sequences")
    # This function validates every runtime artifact and both manifests before
    # the first GT object is loaded.
    paths, runtime_validation = load_runtime_paths(event_map)
    if STAGE.is_file():
        runtime_stage_snapshot = ATTEMPTS / "stage_03_runtime_pass_snapshot.json"
        if not runtime_stage_snapshot.exists():
            atomic_json(runtime_stage_snapshot, load_json(STAGE))

    datasets = DanceTrackDataset(
        str(DATA_ROOT),
        sequences=sorted({str(item["event"]["sequence"]) for item in event_map.values()}),
        split="train",
    )
    gt_by_sequence = {
        sequence: datasets.load_gt(sequence)
        for sequence in sorted({str(item["event"]["sequence"]) for item in event_map.values()})
    }
    rows_by_mode_variant: dict[tuple[str, str], list[dict[str, Any]]] = {
        (mode, variant): [] for mode in MODES for variant in VARIANTS
    }
    event_index = []
    for event_id in sorted(event_map):
        item = event_map[event_id]
        event = item["event"]
        per_event_index = {"event_id": event_id, "sequence": str(event["sequence"]), "variants": {}}
        for mode in MODES:
            artifact = load_json(paths[mode][event_id])
            per_event_index["variants"][mode] = {}
            for variant in VARIANTS:
                variant_payload = artifact["variants"][variant]
                no_trace = variant_payload["branches"]["memory_write=False"]["future_trace"]
                yes_trace = variant_payload["branches"]["memory_write=True"]["future_trace"]
                comparison = build_comparison(variant_payload)
                replay = {
                    "status": "PASS",
                    "candidate_complete": True,
                    "branches": variant_payload["branches"],
                    "comparison": comparison,
                }
                summary = event_variant_summary(
                    str(event["action_type"]), event, variant, replay, gt_by_sequence[str(event["sequence"])]
                )
                add_identity_error_aliases(summary)
                summary["mode"] = mode
                summary["replay_kind"] = "frozen_candidate_state_interface_probe"
                summary["training_split"] = sequence_split[str(event["sequence"])]
                summary["checkpoint_used"] = artifact.get("checkpoint")
                summary["checkpoint_sha256"] = artifact.get("checkpoint_sha256")
                summary["t1_calibration_applied_future_frames"] = sum(
                    int(entry["candidate_audit"].get("t1_calibration", {}).get("applied", False))
                    for entry in yes_trace
                )
                summary["transition_diagnostics"] = transition_diagnostics(
                    comparison, no_trace, yes_trace, event, gt_by_sequence[str(event["sequence"])]
                )
                summary["runtime_boundary"] = {
                    "runtime_future_gt_used": False,
                    "gt_loaded_in_worker": False,
                    "gt_loaded_here": "posthoc_only",
                    "event_frame_memory_read": False,
                    "event_frame_calibration_applied": False,
                    "first_future_frame": int(event["frame"]) + 1,
                }
                event_file = OUT / mode / f"{event_id}.json"
                atomic_json(
                    event_file,
                    {
                        "protocol": PROTOCOL,
                        "status": "PASS",
                        "event_id": event_id,
                        "mode": mode,
                        "sequence": str(event["sequence"]),
                        "action_type": str(event["action_type"]),
                        "interaction_source": "simulated_from_gt",
                        "not_real_human_evidence": True,
                        "summary": summary,
                    },
                )
                rows_by_mode_variant[(mode, variant)].append(summary)
                per_event_index["variants"][mode][variant] = {
                    "status": "PASS",
                    "artifact": str(event_file.relative_to(ROOT)),
                    "runtime_artifact": str(paths[mode][event_id].relative_to(ROOT)),
                }
        event_index.append(per_event_index)
        print(json.dumps({"event_id": event_id, "status": "PASS"}, sort_keys=True), flush=True)
        gc.collect()

    aggregate = {}
    for mode in MODES:
        aggregate[mode] = {}
        for variant in VARIANTS:
            aggregate[mode][variant] = variant_aggregate(
                rows_by_mode_variant[(mode, variant)], event_map, sequence_split, mode, variant
            )

    payload = {
        "protocol": PROTOCOL,
        "status": "COMPLETED_POSTHOC",
        "started_at": started,
        "finished_at": now(),
        "frozen_inputs": {
            "n37_event_manifest": str(N37_MANIFEST.relative_to(ROOT)),
            "n37_event_manifest_sha256": sha256(N37_MANIFEST),
            "training_protocol": str(TRAINING_PROTOCOL.relative_to(ROOT)),
            "training_protocol_sha256": sha256(TRAINING_PROTOCOL),
            "runtime_manifests": {mode: str((REPLAY_DIR / f"runtime_{mode}_manifest.json").relative_to(ROOT)) for mode in MODES},
            "runtime_manifest_sha256": {mode: sha256(REPLAY_DIR / f"runtime_{mode}_manifest.json") for mode in MODES},
        },
        "event_count": 24,
        "independent_sequence_count": 21,
        "variant_count": 5,
        "mode_count": 2,
        "posthoc_event_artifact_count": len(event_index) * len(MODES),
        "posthoc_variant_result_count": len(event_index) * len(MODES) * len(VARIANTS),
        "runtime_validation": runtime_validation,
        "runtime_future_gt_used": False,
        "gt_loaded_only_after_all_runtime_validation": True,
        "gt_used_only_posthoc_scoring_and_direction": True,
        "interaction_source": "simulated_from_gt",
        "real_human_tape_created": False,
        "event_index": event_index,
        "aggregates": aggregate,
        "replay_kind": "frozen_candidate_state_interface_probe",
        "replay_limitation": "T1 recomputes the recorded Hungarian score/assignment interface over frozen N41 candidate/state audits; it is not a production online StateManager deployment or a new checkpoint.",
        "metric_semantics": {
            "identity_utility_delta": "mean(target IoU delta write-minus-no-write, target missing-rate reduction no-write-minus-write), per existing N36/N37 posthoc convention",
            "identity_error": "target public-ID absent or below IoU threshold on visible GT frames",
            "recorrection": "posthoc contiguous target identity-error opportunity proxy, not observed human clicks",
            "id_switch": "posthoc GT-to-public matching at the frozen IoU threshold",
            "idf1_hota_assa": "NOT_COMPUTABLE_NO_COMPLETE_TRACKEVAL_SEQUENCE_INPUT",
            "bootstrap": f"sequence-cluster bootstrap, seed={BOOTSTRAP_SEED}, repetitions={BOOTSTRAP_REPLICATES}",
        },
    }
    atomic_json(RESULT, payload)
    stage = {
        "stage": "N42-03_POSTHOC",
        "status": "POSTHOC_PASS",
        "protocol": PROTOCOL,
        "result": str(RESULT.relative_to(ROOT)),
        "event_count": 24,
        "independent_sequence_count": 21,
        "posthoc_variant_result_count": len(event_index) * len(MODES) * len(VARIANTS),
        "runtime_status": "PASS",
        "runtime_future_gt_used": False,
        "gt_loaded_only_after_all_runtime_validation": True,
        "real_human_tape_created": False,
        "future_effect_gate_evaluated": True,
        "downstream_authorized": False,
        "next_action": "Run MOT/OVMOT isolation regression and apply the pre-registered holdout gate; do not promote T1 automatically.",
    }
    atomic_json(STAGE, stage)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=RESULT)
    args = parser.parse_args()
    try:
        payload = run()
        print(
            json.dumps(
                {
                    "status": payload["status"],
                    "event_count": payload["event_count"],
                    "posthoc_variant_result_count": payload["posthoc_variant_result_count"],
                    "output": str(args.output),
                },
                sort_keys=True,
            ),
            flush=True,
        )
    except Exception as exc:
        failure = {
            "protocol": PROTOCOL,
            "status": "FAIL_POSTHOC",
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
            "runtime_future_gt_used": False,
            "failure_preserved": True,
        }
        failure_path = ATTEMPTS / (
            "posthoc_attempt1_failure.json"
            if not (ATTEMPTS / "posthoc_attempt1_failure.json").exists()
            else "posthoc_attempt2_failure.json"
        )
        atomic_json(failure_path, failure)
        raise


if __name__ == "__main__":
    main()
