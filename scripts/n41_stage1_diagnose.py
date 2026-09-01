#!/usr/bin/env python3
"""N41-01: frozen parameter-path and appearance-to-assignment diagnosis.

This script is deliberately diagnostic-only.  It reads the completed N39
worker artifacts and the N39 scale-audit table, then writes new N41 artifacts.
It does not modify the production associator, read future GT in a worker, or
start a replay/training job.  GT-derived fields already present in the frozen
N37/N39 post-hoc protocol are used only to identify the controlled target row
and are explicitly marked as post-hoc diagnostics in the outputs.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import sys
import tempfile
import traceback
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sam3_intermot.association.appearance_memory import AppearanceMemory
from sam3_intermot.association.ccam_replay import paired_replay
from sam3_intermot.association.state_manager import StateManager
from scripts.n36_real_eval_common import FEATURE_DIM, variant_config
from scripts.run_n37_replay import build_runtime_tape


OUT = ROOT / "outputs" / "n41"
DIAGNOSTIC = OUT / "diagnostic"
N37_MANIFEST = ROOT / "outputs" / "n37" / "real_event_manifest.json"
N39_WORKER_MANIFEST = ROOT / "outputs" / "n39" / "weight_runs" / "full_attempt1_manifest.json"
N39_SCALE_TABLE = ROOT / "outputs" / "n39" / "scale_audit_table.jsonl"
N39_SCALE_SUMMARY = ROOT / "outputs" / "n39" / "scale_audit_summary.json"
N39_FINAL_GATE = ROOT / "outputs" / "n39" / "n39_final_gate.json"
N39_WEIGHT_RESULTS = ROOT / "outputs" / "n39" / "weight_scan_results.json"
N39_REPORT = ROOT / "docs" / "N39_FINAL_REPORT.md"
N40_STAGE1 = ROOT / "outputs" / "n40" / "stage_01_status.json"
N40_PAUSE = ROOT / "outputs" / "n40" / "stage_02_pause_status.json"

PROTOCOL = "N41_WEIGHTED_ASSOCIATION_INTERFACE_PROBE_STAGE1_V1"
VARIANT = "M2"
EPS = 1.0e-9
HARD_NEGATIVE = -1.0e8
LAMBDA_GRID = (0.25, 0.5, 1.0, 2.0, 4.0, 8.0)
SMOKE_EVENT = "n37-dancetrack0001-0296-authoritative_reassign-001"


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def finite(value: Any) -> bool:
    try:
        return bool(math.isfinite(float(value)))
    except (TypeError, ValueError):
        return False


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def atomic_json(path: Path, value: Any) -> None:
    atomic_bytes(
        path,
        (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8"),
    )


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return value


def config_token(value: float) -> str:
    return f"{float(value):g}".replace("-", "m").replace(".", "p")


def config_id(mode: str, value: float) -> str:
    return f"{mode}_{config_token(value)}"


def load_event_items() -> dict[str, dict[str, Any]]:
    payload = load_json(N37_MANIFEST)
    if payload.get("status") != "PASS" or payload.get("event_count") != 24:
        raise RuntimeError("frozen N37 event manifest is not PASS/24")
    events = payload.get("events")
    if not isinstance(events, list) or len(events) != 24:
        raise RuntimeError("frozen N37 event list is invalid")
    output = {}
    for item in events:
        event = item.get("event") if isinstance(item, dict) else None
        if not isinstance(event, dict) or event.get("event_id") is None:
            raise RuntimeError("N37 event_id missing")
        event_id = str(event["event_id"])
        if event_id in output:
            raise RuntimeError(f"duplicate N37 event_id: {event_id}")
        output[event_id] = item
    return output


def load_worker_records() -> dict[tuple[str, float, str], dict[str, Any]]:
    payload = load_json(N39_WORKER_MANIFEST)
    if payload.get("status") != "PASS" or payload.get("phase") != "full":
        raise RuntimeError("N39 full worker manifest is not PASS/full")
    workers = payload.get("workers")
    if not isinstance(workers, list) or len(workers) != 336:
        raise RuntimeError("N39 full worker manifest does not contain 336 workers")
    output: dict[tuple[str, float, str], dict[str, Any]] = {}
    for record in workers:
        if record.get("returncode") != 0:
            raise RuntimeError(f"N39 worker has nonzero returncode: {record}")
        key = (str(record["mode"]), float(record["value"]), str(record["event_id"]))
        if key in output:
            raise RuntimeError(f"duplicate N39 worker key: {key}")
        output[key] = record
    return output


def worker_path(record: dict[str, Any]) -> Path:
    path = Path(str(record["output"]))
    return path if path.is_absolute() else ROOT / path


def load_worker_artifact(records: dict[tuple[str, float, str], dict[str, Any]], mode: str, value: float, event_id: str) -> tuple[dict[str, Any], Path]:
    key = (str(mode), float(value), str(event_id))
    record = records.get(key)
    if record is None:
        raise KeyError(f"missing N39 worker key: {key}")
    path = worker_path(record)
    artifact = load_json(path)
    if artifact.get("status") != "PASS":
        raise RuntimeError(f"N39 worker artifact is not PASS: {path}")
    if artifact.get("runtime_boundary", {}).get("runtime_future_gt_used") is not False:
        raise RuntimeError(f"runtime future GT flag is not false: {path}")
    return artifact, path


def matrices(audit: dict[str, Any]) -> dict[str, np.ndarray]:
    output = {}
    for key in (
        "base_scores_before_appearance",
        "appearance_memory_scores",
        "appearance_score_deltas",
        "fused_scores",
    ):
        value = np.asarray(audit.get(key, []), dtype=np.float64)
        if value.ndim != 2 or not np.all(np.isfinite(value)):
            raise RuntimeError(f"invalid finite score matrix: {key}, shape={value.shape}")
        output[key] = value
    shapes = {value.shape for value in output.values()}
    if len(shapes) != 1:
        raise RuntimeError(f"score matrix shape mismatch: {sorted(shapes)}")
    public_order = audit.get("public_id_order", [])
    candidates = audit.get("candidates", [])
    if output["fused_scores"].shape != (len(candidates), len(public_order)):
        raise RuntimeError(
            f"score axes do not match candidate/state axes: {output['fused_scores'].shape} vs {(len(candidates), len(public_order))}"
        )
    return output


def audit_for(artifact: dict[str, Any], variant: str = VARIANT, branch: str = "memory_write=True") -> tuple[dict[str, Any], dict[str, Any]]:
    variant_payload = artifact.get("variants", {}).get(variant)
    if not isinstance(variant_payload, dict):
        raise RuntimeError(f"missing variant {variant} in {artifact.get('event_id')}")
    event_payload = variant_payload.get("event_frame_audit")
    if not isinstance(event_payload, dict) or not isinstance(event_payload.get("candidate_audit"), dict):
        raise RuntimeError(f"missing event-frame audit in {artifact.get('event_id')}/{variant}")
    branch_payload = variant_payload.get("branches", {}).get(branch)
    if not isinstance(branch_payload, dict):
        raise RuntimeError(f"missing branch {branch} in {artifact.get('event_id')}/{variant}")
    trace = branch_payload.get("future_trace")
    if not isinstance(trace, list) or not trace:
        raise RuntimeError(f"missing future trace in {artifact.get('event_id')}/{variant}")
    first = trace[0]
    if not isinstance(first, dict) or not isinstance(first.get("candidate_audit"), dict):
        raise RuntimeError(f"missing first future audit in {artifact.get('event_id')}/{variant}")
    return event_payload["candidate_audit"], first["candidate_audit"]


def axis_signature(audit: dict[str, Any]) -> dict[str, Any]:
    return {
        key: copy.deepcopy(audit.get(key))
        for key in (
            "candidate_order",
            "candidate_native_ids",
            "candidate_public_ids",
            "public_id_order",
            "public_id_to_native_tid",
        )
    }


def max_abs_diff(left: np.ndarray, right: np.ndarray) -> float:
    if left.shape != right.shape:
        return float("inf")
    return float(np.max(np.abs(left - right))) if left.size else 0.0


def compare_axes(reference: dict[str, Any], candidate: dict[str, Any], label: str, issues: list[str]) -> None:
    if axis_signature(reference) != axis_signature(candidate):
        issues.append(f"axis_changed:{label}")
    for audit, suffix in ((reference, "reference"), (candidate, label)):
        candidates = audit.get("candidates", [])
        order = audit.get("candidate_order", [])
        native = audit.get("candidate_native_ids", [])
        if len(candidates) != len(order) or len(order) != len(native):
            issues.append(f"candidate_axis_length_invalid:{suffix}")
        if len(order) != len(set(order)):
            issues.append(f"duplicate_candidate_order:{suffix}")
        if len(native) != len(set(native)):
            issues.append(f"duplicate_candidate_native_id:{suffix}")
        if audit.get("candidate_public_id_mapping_complete") is not True:
            issues.append(f"mapping_incomplete:{suffix}")


def causal_audit(artifact: dict[str, Any], event_audit: dict[str, Any], future_audit: dict[str, Any]) -> list[str]:
    event_frame = int(artifact["event_frame"])
    issues: list[str] = []
    runtime = artifact.get("runtime_boundary", {})
    if artifact.get("runtime_boundary", {}).get("runtime_future_gt_used") is not False:
        issues.append("top_level_runtime_future_gt_used")
    if runtime.get("first_memory_read_frame") != event_frame + 1:
        issues.append("first_memory_read_frame_mismatch")
    if event_audit.get("frame") != event_frame:
        issues.append("event_frame_mismatch")
    if event_audit.get("memory_read") is not False or event_audit.get("memory_write") is not False:
        issues.append("event_frame_memory_visible")
    if event_audit.get("current_frame_write_hidden") is not True:
        issues.append("event_frame_write_not_hidden")
    if event_audit.get("runtime_future_gt_used") is not False:
        issues.append("event_frame_runtime_future_gt_used")
    if future_audit.get("frame") != event_frame + 1:
        issues.append("future_t_plus_1_mismatch")
    if future_audit.get("runtime_future_gt_used") is not False:
        issues.append("future_runtime_future_gt_used")
    if future_audit.get("current_frame_write_hidden") is not False:
        issues.append("future_write_boundary_invalid")
    return issues


def install_human_weight(weight: float):
    original_init = StateManager.__init__

    def patched_init(self, config):
        original_init(self, config)
        self.appearance_memory.human_weight = float(weight)

    StateManager.__init__ = patched_init  # type: ignore[method-assign]
    return original_init


def human_component_probe(item: dict[str, Any], weights: Iterable[float]) -> dict[str, Any]:
    """Trace the existing private decomposition on one event+1 frame.

    This is a read-only N41 probe.  The patched value is local to freshly
    created StateManager objects and is restored before returning.
    """
    original_score = AppearanceMemory.score
    results: dict[str, Any] = {}
    try:
        for weight in weights:
            calls: list[dict[str, Any]] = []

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
                        "components": {key: float(value) for key, value in components.items()},
                        "score": float(value),
                    }
                )
                return value

            original_init = install_human_weight(float(weight))
            AppearanceMemory.score = traced_score  # type: ignore[method-assign]
            try:
                tape = build_runtime_tape(item, horizon=1)
                config, _description = variant_config(VARIANT)
                config.appearance_score_weight = 1.0
                replay = paired_replay(
                    tape,
                    config=config,
                    feat_dim=FEATURE_DIM,
                    write_branch_uses_appearance_memory=True,
                )
                if replay.get("status") != "PASS":
                    raise RuntimeError(f"instrumented replay failed at human_weight={weight}: {replay.get('status')}")
                trace = replay["branches"]["memory_write=True"]["future_trace"]
                if len(trace) != 1 or int(trace[0]["frame"]) != int(item["event"]["frame"]) + 1:
                    raise RuntimeError(f"instrumented future boundary failed at human_weight={weight}")
                audit = trace[0]["candidate_audit"]
                n_candidates = len(audit.get("candidates", []))
                public_order = [int(value) for value in audit.get("public_id_order", [])]
                expected_per_branch = n_candidates * len(public_order)
                frame_calls = [call for call in calls if int(call["frame"]) == int(item["event"]["frame"]) + 1]
                # The no-write branch disables appearance memory and therefore
                # does not call AppearanceMemory.score.  Only the write branch
                # contributes scorer calls here; expecting two blocks would
                # incorrectly fail the parameter-path smoke.
                if len(frame_calls) != expected_per_branch:
                    raise RuntimeError(
                        f"score call cardinality mismatch at human_weight={weight}: {len(frame_calls)} != {expected_per_branch}"
                    )
                write_calls = frame_calls
                components: dict[str, dict[str, float]] = {}
                for local_index, call in enumerate(write_calls):
                    candidate_index = local_index // len(public_order)
                    state_index = local_index % len(public_order)
                    if int(call["public_id"]) != public_order[state_index]:
                        raise RuntimeError(f"state order mismatch at human_weight={weight}")
                    components[f"{candidate_index}:{public_order[state_index]}"] = call["components"]
                results[config_token(float(weight))] = {
                    "weight": float(weight),
                    "frame": int(trace[0]["frame"]),
                    "candidate_count": n_candidates,
                    "state_count": len(public_order),
                    "components": components,
                    "audit_matrices": {
                        key: np.asarray(audit[key], dtype=np.float64).tolist()
                        for key in (
                            "base_scores_before_appearance",
                            "appearance_memory_scores",
                            "appearance_score_deltas",
                            "fused_scores",
                        )
                    },
                    "runtime_future_gt_used": False,
                }
            finally:
                StateManager.__init__ = original_init  # type: ignore[method-assign]
    finally:
        AppearanceMemory.score = original_score  # type: ignore[method-assign]
    return results


def human_positive_scaling(probe: dict[str, Any]) -> dict[str, Any]:
    by_weight = {float(payload["weight"]): payload for payload in probe.values()}
    base = by_weight.get(1.0)
    w4 = by_weight.get(4.0)
    w8 = by_weight.get(8.0)
    if not base or not w4 or not w8:
        raise RuntimeError("human component probe did not produce weights 1/4/8")
    residuals4: list[float] = []
    residuals8: list[float] = []
    informative = 0
    for key, component in base["components"].items():
        p1 = float(component.get("positive", 0.0))
        p4 = float(w4["components"].get(key, {}).get("positive", 0.0))
        p8 = float(w8["components"].get(key, {}).get("positive", 0.0))
        if abs(p1) > 1.0e-8:
            informative += 1
            residuals4.append(abs(p4 - 4.0 * p1))
            residuals8.append(abs(p8 - 8.0 * p1))
    return {
        "status": "PASS" if informative and max(residuals4 or [float("inf")]) <= 2.0e-5 and max(residuals8 or [float("inf")]) <= 2.0e-5 else "FAIL",
        "informative_positive_entries": informative,
        "max_positive_scaling_residual_weight4": max(residuals4 or [None]),
        "max_positive_scaling_residual_weight8": max(residuals8 or [None]),
        "expected_relation": "positive_weight4=4*positive_weight1 and positive_weight8=8*positive_weight1 for nonzero eligible terms",
    }


def parameter_smoke(event_items: dict[str, dict[str, Any]], records: dict[tuple[str, float, str], dict[str, Any]]) -> dict[str, Any]:
    event_id = SMOKE_EVENT if SMOKE_EVENT in event_items else sorted(event_items)[0]
    smoke_configs = {
        "lambda_assoc": (0.0, 1.0, 8.0),
        "human_weight": (1.0, 4.0, 8.0),
    }
    paths: dict[str, dict[str, str]] = defaultdict(dict)
    artifacts: dict[str, dict[float, dict[str, Any]]] = defaultdict(dict)
    for mode, values in smoke_configs.items():
        for value in values:
            artifact, path = load_worker_artifact(records, mode, value, event_id)
            artifacts[mode][float(value)] = artifact
            paths[mode][config_token(float(value))] = str(path.relative_to(ROOT))

    issues: list[str] = []
    reference_event, reference_future = audit_for(artifacts["lambda_assoc"][1.0])
    causal_by_config = {}
    for mode, values in smoke_configs.items():
        for value in values:
            artifact = artifacts[mode][float(value)]
            event_audit, future_audit = audit_for(artifact)
            label = f"{mode}_{config_token(float(value))}"
            compare_axes(reference_event, event_audit, f"{label}:event", issues)
            compare_axes(reference_future, future_audit, f"{label}:t_plus_1", issues)
            local_issues = causal_audit(artifact, event_audit, future_audit)
            causal_by_config[label] = local_issues
            issues.extend(f"{label}:{item}" for item in local_issues)

    lambda_artifacts = artifacts["lambda_assoc"]
    lambda_reference = matrices(audit_for(lambda_artifacts[1.0])[1])
    lambda_scaling = {}
    for value, artifact in sorted(lambda_artifacts.items()):
        current = matrices(audit_for(artifact)[1])
        base_diff = max_abs_diff(lambda_reference["base_scores_before_appearance"], current["base_scores_before_appearance"])
        memory_diff = max_abs_diff(lambda_reference["appearance_memory_scores"], current["appearance_memory_scores"])
        delta_residual = max_abs_diff(current["appearance_score_deltas"], float(value) * lambda_reference["appearance_score_deltas"])
        regular = current["base_scores_before_appearance"] > HARD_NEGATIVE
        fused_residual = (
            float(np.max(np.abs(current["fused_scores"][regular] - (current["base_scores_before_appearance"][regular] + current["appearance_score_deltas"][regular]))))
            if np.any(regular)
            else 0.0
        )
        hard = ~regular
        hard_preserved = bool(not np.any(hard) or np.all(current["fused_scores"][hard] <= HARD_NEGATIVE))
        if base_diff > 2.0e-5 or memory_diff > 2.0e-5 or delta_residual > 2.0e-5 or fused_residual > 2.0e-5 or not hard_preserved:
            issues.append(f"lambda_scaling_invalid:{value}")
        lambda_scaling[config_token(float(value))] = {
            "value": float(value),
            "base_max_abs_diff_vs_lambda1": base_diff,
            "memory_total_max_abs_diff_vs_lambda1": memory_diff,
            "appearance_delta_max_abs_residual_vs_lambda_times_lambda1": delta_residual,
            "fused_score_max_abs_residual_on_non_hard_negative": fused_residual,
            "hard_negative_entry_count": int(np.sum(hard)),
            "hard_negative_preserved": hard_preserved,
            "nonzero_appearance_delta_count": int(np.sum(np.abs(current["appearance_score_deltas"]) > EPS)),
        }

    human_artifacts = artifacts["human_weight"]
    human_reference = matrices(audit_for(human_artifacts[1.0])[1])
    human_worker_scaling = {}
    for value, artifact in sorted(human_artifacts.items()):
        current = matrices(audit_for(artifact)[1])
        base_diff = max_abs_diff(human_reference["base_scores_before_appearance"], current["base_scores_before_appearance"])
        branch_memory = artifact.get("variants", {}).get(VARIANT, {}).get("branches", {}).get("memory_write=True", {}).get("appearance_memory", {})
        configured_weight = branch_memory.get("human_weight")
        top_weight = artifact.get("weight_configuration", {}).get("appearance_memory_human_weight")
        if base_diff > 2.0e-5:
            issues.append(f"human_weight_changed_base:{value}")
        if not finite(configured_weight) or abs(float(configured_weight) - float(value)) > 1.0e-9:
            issues.append(f"worker_memory_weight_metadata_mismatch:{value}")
        if not finite(top_weight) or abs(float(top_weight) - float(value)) > 1.0e-9:
            issues.append(f"worker_config_weight_metadata_mismatch:{value}")
        human_worker_scaling[config_token(float(value))] = {
            "value": float(value),
            "base_max_abs_diff_vs_human_weight1": base_diff,
            "worker_memory_summary_human_weight": configured_weight,
            "worker_weight_configuration_human_weight": top_weight,
            "appearance_delta_nonzero_count": int(np.sum(np.abs(current["appearance_score_deltas"]) > EPS)),
        }

    human_probe = human_component_probe(event_items[event_id], (1.0, 4.0, 8.0))
    positive_scaling = human_positive_scaling(human_probe)
    if positive_scaling["status"] != "PASS":
        issues.append("human_positive_scaling_invalid")
    worker_matrix_probe_diffs = {}
    for value in (1.0, 4.0, 8.0):
        artifact_matrix = matrices(audit_for(human_artifacts[value])[1])
        probe_matrix = {
            key: np.asarray(human_probe[config_token(value)]["audit_matrices"][key], dtype=np.float64)
            for key in artifact_matrix
        }
        diffs = {key: max_abs_diff(artifact_matrix[key], probe_matrix[key]) for key in artifact_matrix}
        worker_matrix_probe_diffs[config_token(value)] = diffs
        if any(value_diff > 2.0e-5 for value_diff in diffs.values()):
            issues.append(f"human_worker_matrix_not_reproduced:{value}")

    status = "PASS" if not issues else "FAIL"
    return {
        "status": status,
        "event_id": event_id,
        "variant": VARIANT,
        "frozen_worker_artifacts": paths,
        "lambda_assoc": {
            "values": list(smoke_configs["lambda_assoc"]),
            "scaling": lambda_scaling,
            "relation": "appearance_delta(lambda)=lambda*appearance_delta(lambda=1); base and memory total remain fixed",
        },
        "human_weight": {
            "values": list(smoke_configs["human_weight"]),
            "worker_metadata_and_matrix": human_worker_scaling,
            "instrumented_positive_component": positive_scaling,
            "instrumented_worker_matrix_max_abs_diffs": worker_matrix_probe_diffs,
            "relation": "human_weight scales the existing human-positive term inside AppearanceMemory; lambda_assoc is held at 1",
        },
        "causal_boundary_by_config": causal_by_config,
        "issues": issues,
        "runtime_future_gt_used": False,
        "candidate_order_mapping_checked": True,
        "hard_negative_checked": True,
    }


def percentile(values: list[float], quantile: float) -> float | None:
    if not values:
        return None
    return float(np.quantile(np.asarray(values, dtype=np.float64), quantile))


class PairStats:
    def __init__(self) -> None:
        self.pair_count = 0
        self.appearance_positive = 0
        self.base_wrong = 0
        self.base_wrong_positive = 0
        self.base_wrong_can_correct_at_8 = 0
        self.base_correct = 0
        self.base_correct_pushed_wrong_at_1 = 0
        self.base_correct_pushed_wrong_any_scan = 0
        self.mapping_incomplete = 0
        self.runtime_gt_true = 0
        self.lambda_required: list[float] = []
        self.base_gaps: list[float] = []
        self.appearance_gaps: list[float] = []
        self.fused_gaps: list[float] = []

    def add(self, row: dict[str, Any]) -> None:
        self.pair_count += 1
        base_gap = float(row["base_gap"])
        appearance_gap = float(row["appearance_gap"])
        fused_gap = float(row["fused_gap_lambda1"])
        self.base_gaps.append(base_gap)
        self.appearance_gaps.append(appearance_gap)
        self.fused_gaps.append(fused_gap)
        if appearance_gap > EPS:
            self.appearance_positive += 1
        if base_gap < -EPS:
            self.base_wrong += 1
        if base_gap < -EPS and appearance_gap > EPS:
            self.base_wrong_positive += 1
        required = row.get("lambda_required")
        if finite(required):
            self.lambda_required.append(float(required))
        if row.get("base_wrong_can_correct_at_lambda8") is True:
            self.base_wrong_can_correct_at_8 += 1
        if base_gap > EPS:
            self.base_correct += 1
        if row.get("base_correct_pushed_wrong_at_lambda1") is True:
            self.base_correct_pushed_wrong_at_1 += 1
        if row.get("base_correct_pushed_wrong_any_scanned_lambda") is True:
            self.base_correct_pushed_wrong_any_scan += 1
        if row.get("candidate_mapping_complete") is not True:
            self.mapping_incomplete += 1
        if row.get("runtime_future_gt_used") is True:
            self.runtime_gt_true += 1

    def summary(self) -> dict[str, Any]:
        n = self.pair_count
        return {
            "pair_count": n,
            "appearance_gap_positive_count": self.appearance_positive,
            "appearance_gap_positive_rate": (self.appearance_positive / n) if n else None,
            "base_wrong_count": self.base_wrong,
            "base_wrong_with_positive_appearance_count": self.base_wrong_positive,
            "base_wrong_appearance_can_correct_at_lambda8_count": self.base_wrong_can_correct_at_8,
            "base_wrong_appearance_can_correct_at_lambda8_rate_over_base_wrong": (self.base_wrong_can_correct_at_8 / self.base_wrong) if self.base_wrong else None,
            "base_correct_count": self.base_correct,
            "base_correct_pushed_wrong_at_lambda1_count": self.base_correct_pushed_wrong_at_1,
            "base_correct_pushed_wrong_any_scanned_lambda_count": self.base_correct_pushed_wrong_any_scan,
            "candidate_mapping_incomplete_count": self.mapping_incomplete,
            "runtime_future_gt_true_count": self.runtime_gt_true,
            "base_gap": {
                "median": percentile(self.base_gaps, 0.50),
                "p90": percentile(self.base_gaps, 0.90),
                "p95": percentile(self.base_gaps, 0.95),
            },
            "appearance_gap": {
                "median": percentile(self.appearance_gaps, 0.50),
                "p90": percentile(self.appearance_gaps, 0.90),
                "p95": percentile(self.appearance_gaps, 0.95),
            },
            "fused_gap_lambda1": {
                "median": percentile(self.fused_gaps, 0.50),
                "p90": percentile(self.fused_gaps, 0.90),
                "p95": percentile(self.fused_gaps, 0.95),
            },
            "lambda_required_for_appearance_gap_positive": {
                "finite_count": len(self.lambda_required),
                "median": percentile(self.lambda_required, 0.50),
                "p90": percentile(self.lambda_required, 0.90),
                "p95": percentile(self.lambda_required, 0.95),
                "max": max(self.lambda_required) if self.lambda_required else None,
                "reasonable_0_to_8_count": sum(0.0 <= value <= 8.0 for value in self.lambda_required),
            },
        }


def horizon_labels(offset: int) -> list[str]:
    labels: list[str] = []
    if offset == 1:
        labels.append("event_plus_1")
    if 1 <= offset <= 20:
        labels.append("H20")
    if 1 <= offset <= 50:
        labels.append("H50")
    if 1 <= offset <= 100:
        labels.append("H100")
    return labels


def pair_from_group(event_id: str, frame: int, rows: list[dict[str, Any]], event_item: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    event = event_item["event"]
    target_native = int(event["target_native_tid"])
    target_public = int(event.get("public_id", event.get("canonical_public_id")))
    ordered = sorted(rows, key=lambda row: int(row["candidate_index"]))
    target_rows = [row for row in ordered if row.get("candidate_native_id") is not None and int(row["candidate_native_id"]) == target_native]
    group_audit = {
        "event_id": event_id,
        "frame": int(frame),
        "frame_offset": int(frame) - int(event["frame"]),
        "candidate_rows_for_target_state": len(ordered),
        "target_native_tid": target_native,
        "target_row_count": len(target_rows),
        "target_present": bool(target_rows),
        "duplicate_target_rows": max(0, len(target_rows) - 1),
    }
    if len(target_rows) != 1:
        return [], group_audit
    target = target_rows[0]
    output: list[dict[str, Any]] = []
    offset = int(frame) - int(event["frame"])
    labels = horizon_labels(offset)
    for competitor in ordered:
        if int(competitor["candidate_index"]) == int(target["candidate_index"]):
            continue
        base_gap = float(target["base_score"]) - float(competitor["base_score"])
        appearance_gap = float(target["appearance_delta"]) - float(competitor["appearance_delta"])
        fused_gap = float(target["fused_score"]) - float(competitor["fused_score"])
        lambda_required = (-base_gap / appearance_gap) if appearance_gap > EPS else None
        scan_gaps = {str(value): float(base_gap + value * appearance_gap) for value in LAMBDA_GRID}
        row = {
            "protocol": PROTOCOL,
            "diagnostic_type": "GT_CONTROLLED_TARGET_VS_COMPETITOR_PAIR",
            "event_id": event_id,
            "sequence": str(event["sequence"]),
            "action_type": str(event["action_type"]),
            "event_frame": int(event["frame"]),
            "frame": int(frame),
            "frame_offset_from_event": offset,
            "horizon_labels": labels,
            "variant": "M2",
            "branch": "memory_write=True",
            "target_public_id": target_public,
            "target_native_tid": target_native,
            "target_candidate_index": int(target["candidate_index"]),
            "target_candidate_native_id": target.get("candidate_native_id"),
            "target_candidate_public_id": target.get("candidate_public_id"),
            "competitor_candidate_index": int(competitor["candidate_index"]),
            "competitor_candidate_native_id": competitor.get("candidate_native_id"),
            "competitor_candidate_public_id": competitor.get("candidate_public_id"),
            "target_state_public_id": int(target_public),
            "base_gap": base_gap,
            "appearance_gap": appearance_gap,
            "fused_gap_lambda1": fused_gap,
            "lambda_required": lambda_required,
            "lambda_scan_pairwise_target_minus_competitor": scan_gaps,
            "appearance_gap_positive": bool(appearance_gap > EPS),
            "base_wrong": bool(base_gap < -EPS),
            "base_correct": bool(base_gap > EPS),
            "base_wrong_can_correct_at_lambda8": bool(base_gap < -EPS and appearance_gap > EPS and lambda_required is not None and 0.0 < lambda_required <= 8.0 and scan_gaps["8.0"] > EPS),
            "base_correct_pushed_wrong_at_lambda1": bool(base_gap > EPS and scan_gaps["1.0"] < -EPS),
            "base_correct_pushed_wrong_any_scanned_lambda": bool(base_gap > EPS and any(value < -EPS for value in scan_gaps.values())),
            "target_memory_total": float(target["memory_total"]),
            "competitor_memory_total": float(competitor["memory_total"]),
            "target_human_positive": float(target["human_positive"]),
            "competitor_human_positive": float(competitor["human_positive"]),
            "target_machine_prototype": float(target["machine_prototype"]),
            "competitor_machine_prototype": float(competitor["machine_prototype"]),
            "target_negative": float(target["negative"]),
            "competitor_negative": float(competitor["negative"]),
            "candidate_mapping_complete": bool(target.get("candidate_mapping_complete") and competitor.get("candidate_mapping_complete")),
            "memory_read": bool(target.get("memory_read", False) or competitor.get("memory_read", False)),
            "current_frame_write_hidden": bool(target.get("current_frame_write_hidden", False)),
            "runtime_future_gt_used": bool(target.get("runtime_future_gt_used", False) or competitor.get("runtime_future_gt_used", False)),
            "source_scale_audit_relative_path": "outputs/n39/scale_audit_table.jsonl",
        }
        output.append(row)
    return output, group_audit


def write_pair_diagnostics(event_items: dict[str, dict[str, Any]], table_path: Path, output_path: Path) -> dict[str, Any]:
    if not table_path.is_file():
        raise FileNotFoundError(table_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{output_path.name}.", suffix=".tmp", dir=output_path.parent)
    stats: dict[str, PairStats] = defaultdict(PairStats)
    group_count = 0
    target_present_groups = 0
    target_missing_groups = 0
    duplicate_target_groups = 0
    pair_row_count = 0
    input_rows = 0
    filtered_rows = 0
    current_key: tuple[str, int] | None = None
    current_rows: list[dict[str, Any]] = []
    closed_keys: set[tuple[str, int]] = set()

    def flush(handle) -> None:
        nonlocal group_count, target_present_groups, target_missing_groups, duplicate_target_groups, pair_row_count, current_key, current_rows
        if current_key is None:
            return
        event_id, frame = current_key
        pairs, group_audit = pair_from_group(event_id, frame, current_rows, event_items[event_id])
        group_count += 1
        if group_audit["target_present"]:
            target_present_groups += 1
        else:
            target_missing_groups += 1
        if group_audit["duplicate_target_rows"]:
            duplicate_target_groups += 1
        for row in pairs:
            handle.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
            pair_row_count += 1
            action = str(row["action_type"])
            for horizon in row["horizon_labels"]:
                stats[f"action={action}|horizon={horizon}"].add(row)
                stats[f"all|horizon={horizon}"].add(row)
        current_key = None
        current_rows = []

    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            for line in table_path.open(encoding="utf-8"):
                input_rows += 1
                record = json.loads(line)
                if record.get("event_id") not in event_items:
                    raise RuntimeError(f"scale audit event not in N37 manifest: {record.get('event_id')}")
                if record.get("variant") != "M2" or record.get("branch") != "memory_write=True":
                    continue
                if int(record.get("state_public_id", -1)) != int(event_items[record["event_id"]]["event"].get("public_id", event_items[record["event_id"]]["event"].get("canonical_public_id"))):
                    continue
                filtered_rows += 1
                key = (str(record["event_id"]), int(record["frame"]))
                if current_key != key:
                    flush(handle)
                    if key in closed_keys:
                        raise RuntimeError(f"non-contiguous scale-audit frame group: {key}")
                    if current_key is not None:
                        closed_keys.add(current_key)
                    current_key = key
                current_rows.append(record)
            flush(handle)
            if current_key is not None:
                closed_keys.add(current_key)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, output_path)
        directory_fd = os.open(output_path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)

    return {
        "status": "PASS",
        "input_scale_audit_rows": input_rows,
        "filtered_target_state_rows": filtered_rows,
        "frame_group_count": group_count,
        "target_present_frame_group_count": target_present_groups,
        "target_missing_frame_group_count": target_missing_groups,
        "duplicate_target_frame_group_count": duplicate_target_groups,
        "pair_row_count": pair_row_count,
        "pair_row_count_note": "one JSONL row is one unique event-frame target-vs-competitor pair; grouped summaries intentionally repeat rows across applicable horizon windows",
        "by_group": {key: value.summary() for key, value in sorted(stats.items())},
        "source_variant": "M2",
        "source_branch": "memory_write=True",
        "gt_controlled_posthoc": True,
        "runtime_future_gt_used": False,
        "candidate_pair_definition": "target row is the candidate_native_id equal to frozen N37 event.target_native_tid; each other candidate row is compared in the frozen target public-ID state column",
        "gap_definition": "base_gap=base(target row,target state)-base(competitor row,target state); appearance_gap uses existing appearance_delta; fused_gap_lambda1 is the same difference after lambda=1",
        "lambda_required_definition": "-base_gap/appearance_gap only when appearance_gap>0; positive values are the pairwise lambda needed to close a base disadvantage",
        "scan_grid": list(LAMBDA_GRID),
    }


def stage_protocol(input_hashes: dict[str, str]) -> dict[str, Any]:
    return {
        "protocol": PROTOCOL,
        "stage": "N41-01",
        "scope": "CPU-only frozen N39 parameter path and candidate-pair mechanism diagnosis",
        "production_formula_changed": False,
        "runtime_future_gt_used": False,
        "gt_use": "controlled post-hoc target-row identification from frozen N37 event fields; no GT is passed to replay workers",
        "parameter_probe": {
            "lambda_assoc": [0.0, 1.0, 8.0],
            "human_weight": [1.0, 4.0, 8.0],
            "variant": "M2",
            "event": SMOKE_EVENT,
        },
        "pair_diagnostic": {
            "source": "outputs/n39/scale_audit_table.jsonl",
            "variant": "M2",
            "branch": "memory_write=True",
            "horizons": ["event_plus_1", "H20", "H50", "H100"],
            "lambda_scan_for_pairwise_boundary": list(LAMBDA_GRID),
        },
        "input_hashes": input_hashes,
    }


def run(attempt: int) -> dict[str, Any]:
    started = now()
    OUT.mkdir(parents=True, exist_ok=True)
    DIAGNOSTIC.mkdir(parents=True, exist_ok=True)
    event_items = load_event_items()
    records = load_worker_records()
    input_paths = (
        N37_MANIFEST,
        N39_WORKER_MANIFEST,
        N39_SCALE_TABLE,
        N39_SCALE_SUMMARY,
        N39_FINAL_GATE,
        N39_WEIGHT_RESULTS,
        N39_REPORT,
        N40_STAGE1,
        N40_PAUSE,
    )
    inputs = {str(path.relative_to(ROOT)): sha256(path) for path in input_paths if path.is_file()}
    protocol = stage_protocol(inputs)
    protocol_path = DIAGNOSTIC / "diagnostic_protocol.json"
    atomic_json(protocol_path, protocol)

    try:
        smoke = parameter_smoke(event_items, records)
        smoke_path = DIAGNOSTIC / "parameter_smoke.json"
        atomic_json(smoke_path, smoke)
        pair_path = DIAGNOSTIC / "candidate_pair_diagnostics.jsonl"
        pair_summary = write_pair_diagnostics(event_items, N39_SCALE_TABLE, pair_path)
        pair_summary["input_scale_audit_sha256"] = inputs[str(N39_SCALE_TABLE.relative_to(ROOT))]
        pair_summary_path = DIAGNOSTIC / "candidate_pair_summary.json"
        atomic_json(pair_summary_path, pair_summary)

        diagnostic_gate = {
            "parameter_path_smoke_pass": smoke.get("status") == "PASS",
            "candidate_pair_audit_pass": pair_summary.get("status") == "PASS" and pair_summary.get("runtime_future_gt_used") is False and pair_summary.get("duplicate_target_frame_group_count") == 0,
            "candidate_pair_rows_nonzero": pair_summary.get("pair_row_count", 0) > 0,
            "production_formula_changed": False,
            "downstream_interface_authorized": False,
        }
        # Only the two positive checks below decide whether the diagnostic
        # stage completed.  The other fields are safety/informational facts:
        # production_formula_changed must remain false and downstream
        # authorization must remain false at N41-01.
        required_diagnostic_checks = (
            diagnostic_gate["parameter_path_smoke_pass"],
            diagnostic_gate["candidate_pair_audit_pass"],
            diagnostic_gate["candidate_pair_rows_nonzero"],
        )
        status = "PASS" if all(required_diagnostic_checks) else "FAIL"
        finished = now()
        stage = {
            "stage": "N41-01",
            "status": status,
            "protocol": PROTOCOL,
            "attempt": int(attempt),
            "started_at": started,
            "finished_at": finished,
            "inputs": inputs,
            "artifacts": [
                str(protocol_path.relative_to(ROOT)),
                str((DIAGNOSTIC / "parameter_smoke.json").relative_to(ROOT)),
                str((DIAGNOSTIC / "candidate_pair_diagnostics.jsonl").relative_to(ROOT)),
                str((DIAGNOSTIC / "candidate_pair_summary.json").relative_to(ROOT)),
            ],
            "parameter_smoke": smoke,
            "candidate_pair_summary": {
                "pair_row_count": pair_summary.get("pair_row_count"),
                "frame_group_count": pair_summary.get("frame_group_count"),
                "target_present_frame_group_count": pair_summary.get("target_present_frame_group_count"),
                "target_missing_frame_group_count": pair_summary.get("target_missing_frame_group_count"),
                "duplicate_target_frame_group_count": pair_summary.get("duplicate_target_frame_group_count"),
            },
            "diagnostic_gate": diagnostic_gate,
            "scientific_interpretation": "pending_summary_review",
            "downstream_authorized": False,
            "next_action": "Review candidate pair direction/margin evidence; N41-02 is not authorized by this artifact alone.",
        }
        atomic_json(OUT / "stage_01_status.json", stage)
        return stage
    except Exception as exc:
        failure_path = OUT / "attempts" / f"stage_01_attempt{int(attempt)}_failure.json"
        atomic_json(
            failure_path,
            {
                "protocol": PROTOCOL,
                "stage": "N41-01",
                "attempt": int(attempt),
                "status": "FAIL",
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(),
                "artifact_is_failure_evidence": True,
            },
        )
        atomic_json(
            OUT / "stage_01_status.json",
            {
                "stage": "N41-01",
                "status": "FAIL",
                "protocol": PROTOCOL,
                "attempt": int(attempt),
                "started_at": started,
                "finished_at": now(),
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
    print(
        json.dumps(
            {
                "status": result["status"],
                "stage": result["stage"],
                "pair_row_count": result["candidate_pair_summary"]["pair_row_count"],
                "output": "outputs/n41/stage_01_status.json",
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
