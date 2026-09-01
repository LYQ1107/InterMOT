#!/usr/bin/env python3
"""N39 Stage 1: audit appearance-score scale against the association margin.

The frozen N38R1 sidecars contain the complete candidate/state score matrices,
but the older sidecar schema stores only the total appearance score.  This
N39-only audit therefore replays the same future-blind tape in a CPU process
with a tracing wrapper around the existing ``AppearanceMemory.score`` method.
The wrapper calls the unchanged private decomposition and checks that its
total agrees with the frozen sidecar before writing a compact per-candidate /
per-public-ID table.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import tempfile
import traceback
from array import array
from collections import defaultdict
from copy import deepcopy
from pathlib import Path
from typing import Any, Iterable

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sam3_intermot.association.appearance_memory import AppearanceMemory
from sam3_intermot.association.ccam_replay import paired_replay
from scripts.n36_real_eval_common import FEATURE_DIM, atomic_json
from scripts.n38r1_sidecar_common import (
    VARIANTS,
    build_sidecar,
    load_manifest_item,
    protocol_hash,
)
from scripts.run_n37_replay import build_runtime_tape
from scripts.n36_real_eval_common import variant_config


OUT = ROOT / "outputs/n39"
SIDECAR_ROOT = ROOT / "outputs/n38r1/sidecar"
N38R1_MANIFEST = ROOT / "outputs/n38r1/sidecar_manifest.json"
N38R1_SUMMARY = ROOT / "outputs/n38r1/diagnostic_attempt3/score_assignment_summary.json"
N37_MANIFEST = ROOT / "outputs/n37/real_event_manifest.json"
PROTOCOL = "N39_WEIGHTED_ASSOCIATION_INTERFACE_PROBE_STAGE1_V1"
METRIC_NAMES = (
    "base_score",
    "memory_total",
    "human_positive",
    "machine_prototype",
    "negative",
    "appearance_delta",
    "fused_score",
    "top1_top2_margin",
    "assignment_margin",
    "abs_appearance_delta_over_margin",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def finite(value: Any) -> bool:
    try:
        return bool(math.isfinite(float(value)))
    except (TypeError, ValueError):
        return False


def atomic_jsonl_stream(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    """Write a potentially large JSONL artifact with fsync + atomic rename."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    count = 0
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
                count += 1
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_name, path)
    finally:
        if os.path.exists(temp_name):
            os.unlink(temp_name)
    return count


class DistributionAccumulator:
    """Compact exact percentile accumulator using C-level double arrays."""

    def __init__(self) -> None:
        self.values: dict[str, dict[str, array]] = defaultdict(dict)
        self.counts: dict[str, int] = defaultdict(int)
        self.nonfinite: dict[str, int] = defaultdict(int)

    def add(self, group: str, metric: str, value: Any) -> None:
        if not finite(value):
            self.nonfinite[group] += 1
            return
        bucket = self.values[group].get(metric)
        if bucket is None:
            bucket = array("d")
            self.values[group][metric] = bucket
        bucket.append(float(value))
        self.counts[group] += 1

    @staticmethod
    def _stats(values: array) -> dict[str, Any]:
        if not values:
            return {
                "count": 0,
                "median": None,
                "p90": None,
                "p95": None,
                "max": None,
            }
        data = np.asarray(values, dtype=np.float64)
        return {
            "count": int(data.size),
            "median": float(np.quantile(data, 0.50)),
            "p90": float(np.quantile(data, 0.90)),
            "p95": float(np.quantile(data, 0.95)),
            "max": float(np.max(data)),
        }

    def summary(self) -> dict[str, Any]:
        output: dict[str, Any] = {}
        for group in sorted(self.values):
            output[group] = {
                metric: self._stats(self.values[group].get(metric, array("d")))
                for metric in METRIC_NAMES
            }
            output[group]["nonfinite_observations"] = int(self.nonfinite.get(group, 0))
        return output


def _json_matrix(audit: dict[str, Any], key: str) -> np.ndarray:
    value = np.asarray(audit.get(key, []), dtype=np.float64)
    if value.ndim != 2:
        raise RuntimeError(f"{key} is not a matrix: shape={value.shape}")
    return value


def _state_margins(fused: np.ndarray) -> list[float | None]:
    margins: list[float | None] = []
    for column in range(fused.shape[1]):
        values = np.asarray(fused[:, column], dtype=np.float64)
        if values.size < 2 or not np.all(np.isfinite(values)):
            margins.append(None)
            continue
        ordered = np.sort(values)[::-1]
        margins.append(float(ordered[0] - ordered[1]))
    return margins


def _horizon_bucket(event_frame: int, frame: int) -> str:
    offset = int(frame) - int(event_frame)
    if offset == 0:
        return "event_frame"
    if offset == 1:
        return "future_t_plus_1"
    if 2 <= offset <= 20:
        return "future_h20_after_t_plus_1"
    if 21 <= offset <= 50:
        return "future_h50_after_t_plus_1"
    return "future_h100_after_t_plus_1"


def _trace_components(item: dict[str, Any], variant: str) -> tuple[dict[tuple[int, int, int], dict[str, float]], dict[str, Any]]:
    """Replay one variant and return (frame,row,state-pid) score components."""
    calls: list[dict[str, Any]] = []
    original_score = AppearanceMemory.score

    def traced_score(self, public_id, candidate_embedding, frame, positive_weight=1.0, negative_weight=1.0, gate_floor=0.0):
        components = self._score_components(
            public_id,
            candidate_embedding,
            frame,
            positive_weight=positive_weight,
            negative_weight=negative_weight,
            gate_floor=gate_floor,
        )
        value = original_score(
            self,
            public_id,
            candidate_embedding,
            frame,
            positive_weight=positive_weight,
            negative_weight=negative_weight,
            gate_floor=gate_floor,
        )
        calls.append(
            {
                "frame": int(frame),
                "public_id": int(public_id),
                "components": {key: float(val) for key, val in components.items()},
                "score": float(value),
            }
        )
        return value

    AppearanceMemory.score = traced_score  # type: ignore[method-assign]
    try:
        tape = build_runtime_tape(item, horizon=100)
        config, _description = variant_config(variant)
        replay = paired_replay(
            tape,
            config=config,
            feat_dim=FEATURE_DIM,
            write_branch_uses_appearance_memory=(variant != "M0"),
        )
    finally:
        AppearanceMemory.score = original_score  # type: ignore[method-assign]

    if replay.get("status") != "PASS":
        raise RuntimeError(f"instrumented replay failed for {variant}: {replay.get('status')}")
    trace = replay["branches"]["memory_write=True"]["future_trace"]
    calls_by_frame: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for call in calls:
        calls_by_frame[int(call["frame"])].append(call)
    component_map: dict[tuple[int, int, int], dict[str, float]] = {}
    count_checks = []
    for entry in trace:
        frame = int(entry["frame"])
        audit = entry["candidate_audit"]
        n = len(audit.get("candidates", []))
        state_ids = [int(value) for value in audit.get("public_id_order", [])]
        expected = n * len(state_ids)
        observed = calls_by_frame.get(frame, [])
        if len(observed) != expected:
            raise RuntimeError(
                f"score trace cardinality mismatch variant={variant} frame={frame}: {len(observed)} != {expected}"
            )
        for call_index, call in enumerate(observed):
            row = call_index // len(state_ids) if state_ids else 0
            column = call_index % len(state_ids) if state_ids else 0
            if not state_ids:
                continue
            if int(call["public_id"]) != state_ids[column]:
                raise RuntimeError(
                    f"score trace state order mismatch variant={variant} frame={frame} call={call_index}"
                )
            component_map[(frame, row, state_ids[column])] = call["components"]
        count_checks.append({"frame": frame, "expected": expected, "observed": len(observed)})
    return component_map, {
        "variant": variant,
        "future_frame_count": len(trace),
        "score_call_count": len(calls),
        "frame_cardinality_checks": count_checks,
    }


def _component_zero() -> dict[str, float]:
    return {"prototype": 0.0, "positive": 0.0, "negative": 0.0, "total": 0.0}


def _sidecar_files() -> list[Path]:
    return sorted(SIDECAR_ROOT.glob("*/*.json"))


def _make_protocol() -> dict[str, Any]:
    return {
        "protocol": PROTOCOL,
        "description": "Frozen N38R1 score scale audit; no event selection and no post-treatment GT.",
        "frozen_inputs": {
            "n37_event_manifest": str(N37_MANIFEST.relative_to(ROOT)),
            "n37_event_manifest_sha256": sha256(N37_MANIFEST),
            "n38r1_sidecar_manifest": str(N38R1_MANIFEST.relative_to(ROOT)),
            "n38r1_sidecar_manifest_sha256": sha256(N38R1_MANIFEST),
            "n38r1_diagnostic_summary": str(N38R1_SUMMARY.relative_to(ROOT)),
            "n38r1_diagnostic_summary_sha256": sha256(N38R1_SUMMARY),
            "frozen_n38_protocol_hash": protocol_hash(),
        },
        "score_semantics": {
            "base_score": "online associator score after native/positive/hard-negative legacy terms and before appearance memory",
            "machine_prototype": "AppearanceMemory._score_components prototype term, machine_weight times decayed prototype cosine",
            "human_positive": "AppearanceMemory._score_components positive term, human_weight times decayed best eligible anchor cosine",
            "negative": "AppearanceMemory._score_components negative penalty from eligible human competitor anchors",
            "memory_total": "prototype + positive + negative",
            "appearance_delta": "appearance_score_weight * memory_total",
            "fused_score": "base_score + appearance_delta, with hard negatives re-applied",
            "assignment": "scipy linear_sum_assignment(-fused_score)",
        },
        "ratio_definition": {
            "numerator": "abs(appearance_delta)",
            "denominator": "positive target-state top1-top2 score margin",
            "zero_margin": "ratio=null and counted separately when margin <= 1e-12",
            "raw_scores_retained": True,
        },
        "causal_contract": {
            "event_frame_memory_read": False,
            "first_memory_eligible_frame": "event_frame + 1",
            "runtime_future_gt_used": False,
            "posthoc_gt_loaded": False,
        },
        "group_dimensions": ["action_type", "variant", "frame_horizon", "target_row"],
        "metrics": list(METRIC_NAMES),
    }


def _record_rows(
    artifact: dict[str, Any],
    item: dict[str, Any],
    variant: str,
    branch_name: str,
    component_map: dict[tuple[int, int, int], dict[str, float]],
    comparison: dict[str, Any],
    accumulator: DistributionAccumulator,
    validation: dict[str, Any],
) -> list[dict[str, Any]]:
    event = artifact["event"]
    event_frame = int(artifact["event_frame"])
    target_pid = int(event.get("public_id", event.get("canonical_public_id")))
    branch = artifact["branches"][branch_name]
    rows: list[dict[str, Any]] = []
    for entry in [
        artifact["event_frame_audit"],
        *branch.get("future_trace", []),
    ]:
        frame = int(entry["frame"])
        audit = entry["candidate_audit"] if "candidate_audit" in entry else entry
        base = _json_matrix(audit, "base_scores_before_appearance")
        memory = _json_matrix(audit, "appearance_memory_scores")
        delta = _json_matrix(audit, "appearance_score_deltas")
        fused = _json_matrix(audit, "fused_scores")
        if not (base.shape == memory.shape == delta.shape == fused.shape):
            raise RuntimeError(f"matrix shape mismatch {artifact['event_id']} {variant} {branch_name} frame={frame}")
        public_ids = [int(value) for value in audit.get("public_id_order", [])]
        if base.shape[1] != len(public_ids):
            raise RuntimeError(f"state axis mismatch {artifact['event_id']} frame={frame}")
        state_margins = _state_margins(fused)
        assignments = audit.get("assignment_after_scope", audit.get("assignment", []))
        if not isinstance(assignments, list):
            assignments = []
        candidates = audit.get("candidate_records", audit.get("candidates", []))
        if base.shape[0] != len(candidates):
            raise RuntimeError(f"candidate axis mismatch {artifact['event_id']} frame={frame}")
        target_row = None
        for index, value in enumerate(assignments):
            try:
                if 0 <= int(value) < len(public_ids) and public_ids[int(value)] == target_pid:
                    target_row = index
                    break
            except (TypeError, ValueError):
                continue
        assignment_margin = audit.get("hungarian_cost_audit", {}).get("assignment_score_margin")
        if not finite(assignment_margin):
            assignment_margin = None
        for candidate_index in range(base.shape[0]):
            candidate = candidates[candidate_index] if isinstance(candidates[candidate_index], dict) else {}
            for state_index, public_id in enumerate(public_ids):
                components = component_map.get(
                    (frame, candidate_index, int(public_id)), _component_zero()
                )
                values = {
                    "base_score": float(base[candidate_index, state_index]),
                    "memory_total": float(memory[candidate_index, state_index]),
                    "human_positive": float(components.get("positive", 0.0)),
                    "machine_prototype": float(components.get("prototype", 0.0)),
                    "negative": float(components.get("negative", 0.0)),
                    "appearance_delta": float(delta[candidate_index, state_index]),
                    "fused_score": float(fused[candidate_index, state_index]),
                    "top1_top2_margin": state_margins[state_index],
                    "assignment_margin": assignment_margin if candidate_index == target_row else None,
                    "abs_appearance_delta_over_margin": (
                        float(abs(delta[candidate_index, state_index]) / max(abs(float(state_margins[state_index])), 1.0e-12))
                        if finite(state_margins[state_index]) and abs(float(state_margins[state_index])) > 1.0e-12
                        else None
                    ),
                }
                component_sum = values["machine_prototype"] + values["human_positive"] + values["negative"]
                if abs(component_sum - values["memory_total"]) > 2.0e-5:
                    raise RuntimeError(
                        f"component total mismatch event={artifact['event_id']} variant={variant} frame={frame} row={candidate_index} state={public_id}: {component_sum} != {values['memory_total']}"
                    )
                if branch_name == "memory_write=False" and any(abs(values[key]) > 2.0e-7 for key in ("memory_total", "appearance_delta")):
                    raise RuntimeError(f"nonzero no-write memory score event={artifact['event_id']} frame={frame}")
                is_target_row = target_row is not None and candidate_index == target_row
                group_values = {
                    "all": True,
                    f"action={event['action_type']}": True,
                    f"variant={variant}": True,
                    f"frame_horizon={_horizon_bucket(event_frame, frame)}": True,
                    f"target_row={'true' if is_target_row else 'false'}": True,
                }
                for group in group_values:
                    for metric in METRIC_NAMES:
                        accumulator.add(group, metric, values[metric])
                rows.append(
                    {
                        "protocol": PROTOCOL,
                        "event_id": str(artifact["event_id"]),
                        "sequence": str(artifact["sequence"]),
                        "action_type": str(event["action_type"]),
                        "variant": str(variant),
                        "branch": str(branch_name),
                        "frame": frame,
                        "frame_offset_from_event": frame - event_frame,
                        "frame_horizon": _horizon_bucket(event_frame, frame),
                        "is_event_frame": frame == event_frame,
                        "is_future_frame": frame > event_frame,
                        "candidate_index": int(candidate_index),
                        "candidate_obs_id": candidate.get("candidate_obs_id", candidate.get("obs_id")),
                        "candidate_native_id": candidate.get("candidate_native_id", candidate.get("native_tid")),
                        "candidate_public_id": candidate.get("candidate_public_id"),
                        "state_public_id": int(public_id),
                        "target_public_id": target_pid,
                        "is_target_state": bool(int(public_id) == target_pid),
                        "is_target_row": bool(is_target_row),
                        "target_row_index": target_row,
                        "candidate_feature_norm": candidate.get("feature_norm"),
                        "candidate_mapping_complete": bool(audit.get("candidate_public_id_mapping_complete", False)),
                        "memory_read": bool(audit.get("memory_read", False)),
                        "memory_write": bool(audit.get("memory_write", False)),
                        "current_frame_write_hidden": bool(audit.get("current_frame_write_hidden", False)),
                        "runtime_future_gt_used": bool(audit.get("runtime_future_gt_used", False)),
                        "component_source": "instrumented_existing_score_components" if component_map else "frozen_no_write_zero",
                        "sidecar_relative_path": str((SIDECAR_ROOT / str(artifact["event_id"]) / f"{variant}.json").relative_to(ROOT)),
                        **values,
                    }
                )
    validation.setdefault("processed_frames", 0)
    validation["processed_frames"] += len({int(row["frame"]) for row in rows})
    return rows


def run(attempt: int = 1) -> dict[str, Any]:
    OUT.mkdir(parents=True, exist_ok=True)
    protocol_path = OUT / "scale_audit_protocol.json"
    table_path = OUT / "scale_audit_table.jsonl"
    summary_path = OUT / "scale_audit_summary.json"
    stage_path = OUT / "stage_01_status.json"
    atomic_json(protocol_path, _make_protocol())
    started = __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat()
    validation: dict[str, Any] = {
        "artifact_count": 0,
        "component_replay_count": 0,
        "max_matrix_diffs": defaultdict(float),
        "zero_margin_count": 0,
        "runtime_future_gt_true_count": 0,
        "causal_boundary_violations": [],
    }
    accumulator = DistributionAccumulator()
    manifest, _ = load_manifest_item(N37_MANIFEST, "n37-dancetrack0001-0296-authoritative_reassign-001")
    event_items = {
        str(item["event"]["event_id"]): item
        for item in json.loads(N37_MANIFEST.read_text(encoding="utf-8"))["events"]
    }
    artifacts = _sidecar_files()
    if len(artifacts) != 120:
        raise RuntimeError(f"expected 120 N38R1 artifacts, found {len(artifacts)}")
    component_cache: dict[tuple[str, str], tuple[dict[tuple[int, int, int], dict[str, float]], dict[str, Any]]] = {}
    rows_by_artifact: list[dict[str, Any]] = []

    def row_iter():
        for path in artifacts:
            artifact = json.loads(path.read_text(encoding="utf-8"))
            if artifact.get("status") != "PASS":
                raise RuntimeError(f"N38R1 artifact not PASS: {path}")
            event_id = str(artifact["event_id"])
            variant = str(artifact["variant"])
            item = event_items.get(event_id)
            if item is None:
                raise RuntimeError(f"event missing from N37 manifest: {event_id}")
            if artifact.get("runtime_future_gt_used") is not False:
                raise RuntimeError(f"runtime future GT in artifact {path}")
            for branch_name, branch in artifact.get("branches", {}).items():
                for entry in branch.get("future_trace", []):
                    audit = entry.get("candidate_audit", {})
                    if audit.get("runtime_future_gt_used") is not False:
                        validation["runtime_future_gt_true_count"] += 1
                    if int(entry["frame"]) <= int(artifact["event_frame"]):
                        validation["causal_boundary_violations"].append({"path": str(path), "frame": entry["frame"]})
            key = (event_id, variant)
            if variant == "M0":
                component_map, probe = {}, {"variant": variant, "score_call_count": 0}
            else:
                if key not in component_cache:
                    component_cache[key] = _trace_components(item, variant)
                    validation["component_replay_count"] += 1
                component_map, probe = component_cache[key]
            # Compare the instrumented replay's matrices with the frozen
            # sidecar before exposing any decomposition rows.
            if variant != "M0":
                tape = build_runtime_tape(item, horizon=100)
                config, _ = variant_config(variant)
                generated = paired_replay(tape, config=config, feat_dim=FEATURE_DIM, write_branch_uses_appearance_memory=True)
                generated_trace = generated["branches"]["memory_write=True"]["future_trace"]
                frozen_trace = artifact["branches"]["memory_write=True"]["future_trace"]
                if len(generated_trace) != len(frozen_trace):
                    raise RuntimeError(f"replay trace length mismatch {event_id} {variant}")
                for left, right in zip(generated_trace, frozen_trace):
                    if int(left["frame"]) != int(right["frame"]):
                        raise RuntimeError(f"replay frame mismatch {event_id} {variant}")
                    for matrix_key in ("base_scores_before_appearance", "appearance_memory_scores", "appearance_score_deltas", "fused_scores"):
                        left_matrix = np.asarray(left["candidate_audit"].get(matrix_key, []), dtype=np.float64)
                        right_matrix = np.asarray(right["candidate_audit"].get(matrix_key, []), dtype=np.float64)
                        if left_matrix.shape != right_matrix.shape:
                            raise RuntimeError(f"matrix shape regression {event_id} {variant} {matrix_key}")
                        diff = float(np.max(np.abs(left_matrix - right_matrix))) if left_matrix.size else 0.0
                        validation["max_matrix_diffs"][matrix_key] = max(validation["max_matrix_diffs"][matrix_key], diff)
                        if diff > 2.0e-5:
                            raise RuntimeError(f"frozen sidecar mismatch {event_id} {variant} {matrix_key}: {diff}")
                del tape, generated
            for row in _record_rows(artifact, item, variant, branch_name, component_map if branch_name == "memory_write=True" else {}, probe, accumulator, validation):
                if row["top1_top2_margin"] is None:
                    validation["zero_margin_count"] += 1
                yield row
            validation["artifact_count"] += 1

    try:
        row_count = atomic_jsonl_stream(table_path, row_iter())
        finished = __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat()
        summary = {
            "protocol": PROTOCOL,
            "status": "PASS",
            "started_at": started,
            "finished_at": finished,
            "input_artifacts": {
                "n38r1_sidecar_count": len(artifacts),
                "n38r1_sidecar_manifest": str(N38R1_MANIFEST.relative_to(ROOT)),
                "n38r1_sidecar_manifest_sha256": sha256(N38R1_MANIFEST),
                "n38r1_summary": str(N38R1_SUMMARY.relative_to(ROOT)),
                "n38r1_summary_sha256": sha256(N38R1_SUMMARY),
                "n37_manifest_sha256": sha256(N37_MANIFEST),
            },
            "table": str(table_path.relative_to(ROOT)),
            "row_count": int(row_count),
            "event_count": 24,
            "variant_count": 5,
            "branch_count": 2,
            "component_replay_count": validation["component_replay_count"],
            "component_source": "existing AppearanceMemory._score_components traced in N39 process; no formula change",
            "matrix_replay_max_abs_diff": dict(validation["max_matrix_diffs"]),
            "runtime_future_gt_true_count": validation["runtime_future_gt_true_count"],
            "causal_boundary_violations": validation["causal_boundary_violations"],
            "zero_margin_observations": validation["zero_margin_count"],
            "distribution_by_dimension": accumulator.summary(),
            "diagnosis": {
                "status": "COMPUTED",
                "interpretation_rule": "delta far below target top1-top2 margin supports association-interface scale limitation; delta near margin with unchanged assignment points to candidate/base geometry or assignment behavior",
                "internal_vs_external": "internal human_weight affects human_positive only; external appearance_score_weight affects appearance_delta and fused_score after the memory total",
            },
        }
        atomic_json(summary_path, summary)
        stage = {
            "stage": "N39-01",
            "status": "PASS",
            "protocol": PROTOCOL,
            "started_at": started,
            "finished_at": finished,
            "artifacts": [str(protocol_path.relative_to(ROOT)), str(table_path.relative_to(ROOT)), str(summary_path.relative_to(ROOT))],
            "row_count": int(row_count),
            "n38r1_artifact_count": len(artifacts),
            "component_replay_count": validation["component_replay_count"],
            "matrix_replay_max_abs_diff": dict(validation["max_matrix_diffs"]),
            "runtime_future_gt_true_count": validation["runtime_future_gt_true_count"],
            "causal_boundary_violations": validation["causal_boundary_violations"],
            "downstream_authorized": True,
            "next_action": "Freeze N39 external lambda and human_weight scans before running smoke.",
        }
        atomic_json(stage_path, stage)
        return stage
    except Exception as exc:
        failure_path = OUT / "attempts" / f"stage_01_attempt{int(attempt)}_failure.json"
        atomic_json(
            failure_path,
            {
                "protocol": PROTOCOL,
                "status": "FAIL",
                "attempt": int(attempt),
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(),
                "artifact_is_failure_evidence": True,
                "partial_validation": {
                    **{key: value for key, value in validation.items() if key != "max_matrix_diffs"},
                    "max_matrix_diffs": dict(validation["max_matrix_diffs"]),
                },
            },
        )
        atomic_json(
            stage_path,
            {
                "stage": "N39-01",
                "status": "FAIL",
                "protocol": PROTOCOL,
                "failure_artifact": str(failure_path.relative_to(ROOT)),
                "error": f"{type(exc).__name__}: {exc}",
                "downstream_authorized": False,
            },
        )
        raise


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--attempt", type=int, default=1)
    args = parser.parse_args()
    result = run(args.attempt)
    print(json.dumps({"status": result["status"], "row_count": result["row_count"], "output": result["artifacts"]}, sort_keys=True))


if __name__ == "__main__":
    main()
