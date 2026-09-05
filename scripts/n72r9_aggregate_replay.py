#!/usr/bin/env python3
"""Audit and aggregate the sealed N72R9 development replay.

The event worker deliberately keeps runtime JSONL separate from posthoc GT
scoring.  This script consumes only the completed event artifacts, verifies
that separation and the frozen frame/candidate contracts, then aggregates the
already-computed posthoc metrics.  It never runs the tracker and never uses
GT to repair or select an event.
"""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = ROOT / "outputs/N72R9/protocol.json"
EVENT_POLICY_PATH = ROOT / "outputs/N72R5/mechanism_rounds/round_06_event_policy/real_event_manifest.json"
DEFAULT_REPLAY_ROOT = ROOT / "outputs/N72R9/replay/full"
DEFAULT_RESULT_PATH = ROOT / "outputs/N72R9/ccam_paired_replay_results.json"
DEFAULT_EVENT_METRICS_PATH = ROOT / "outputs/N72R9/replay/full/event_metrics.jsonl"
DEFAULT_AUDIT_PATH = ROOT / "outputs/N72R9/replay/full/runtime_audit.json"
DEFAULT_STAGE_PATH = ROOT / "outputs/N72R9/stage_07_replay_status.json"

HORIZONS = (20, 50, 100)
VARIANTS = ("BASELINE_B0", "TEMPORAL_CURRENT", "TEMPORAL_REQUERY")
COMPARISONS = {
    "TEMPORAL_CURRENT_vs_BASELINE_B0": ("BASELINE_B0", "TEMPORAL_CURRENT"),
    "TEMPORAL_REQUERY_vs_BASELINE_B0": ("BASELINE_B0", "TEMPORAL_REQUERY"),
    "TEMPORAL_REQUERY_vs_TEMPORAL_CURRENT": ("TEMPORAL_CURRENT", "TEMPORAL_REQUERY"),
}
ACTION_TYPES = (
    "ADD_NEW_IDENTITY",
    "AUTHORITATIVE_REASSIGN",
    "ATOMIC_ID_SWAP",
    "RECOVER_IDENTITY",
)
BOOTSTRAP_SEED = 7290
BOOTSTRAP_REPETITIONS = 2000


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
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


def atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    atomic_write(path, json.dumps(dict(value), ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n")


def atomic_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    atomic_write(
        path,
        "".join(json.dumps(dict(row), ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n" for row in rows),
    )


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object: {path}")
    return value


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise TypeError(f"{path}:{line_number}: expected JSON object")
            rows.append(value)
    return rows


def resolved_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def _contains_forbidden_gt_key(value: Any, location: str, errors: list[str]) -> None:
    forbidden = {"dataset_gt_id", "gt_box", "future_gt", "future_identity_error", "h20", "h50", "h100"}
    if isinstance(value, Mapping):
        for key, nested in value.items():
            key_text = str(key).lower()
            if key_text in forbidden:
                errors.append(f"{location}/{key}")
            _contains_forbidden_gt_key(nested, f"{location}/{key}", errors)
    elif isinstance(value, list):
        for index, nested in enumerate(value):
            _contains_forbidden_gt_key(nested, f"{location}/{index}", errors)


def _runtime_audit(protocol: Mapping[str, Any], batch: Mapping[str, Any], replay_root: Path) -> tuple[dict[str, Any], list[dict[str, Any]], list[str]]:
    expected_events = [str(item["event_id"]) for item in protocol.get("source_event_selection", {}).get("events", [])]
    expected_set = set(expected_events)
    errors: list[str] = []
    failures: list[dict[str, Any]] = []
    batch_ids = [str(item.get("event_id")) for item in batch.get("results", [])]
    duplicate_batch_ids = sorted({event_id for event_id in batch_ids if batch_ids.count(event_id) > 1})
    missing_batch_ids = sorted(expected_set - set(batch_ids))
    unexpected_batch_ids = sorted(set(batch_ids) - expected_set)
    if batch.get("status") != "PASS_N72R9_REPLAY_BATCH":
        errors.append(f"batch_status={batch.get('status')}")
    if len(batch_ids) != len(expected_events):
        errors.append(f"batch_event_count={len(batch_ids)} expected={len(expected_events)}")
    if duplicate_batch_ids:
        errors.append(f"duplicate_batch_event_ids={duplicate_batch_ids}")
    if missing_batch_ids:
        errors.append(f"missing_batch_event_ids={missing_batch_ids}")
    if unexpected_batch_ids:
        errors.append(f"unexpected_batch_event_ids={unexpected_batch_ids}")
    if batch.get("runtime_future_gt_used") is not False:
        errors.append("batch_runtime_future_gt_used_not_false")

    frame_rows = 0
    candidate_rows = 0
    candidate_duplicate_count = 0
    frame_axis_error_count = 0
    matrix_error_count = 0
    mapping_error_count = 0
    runtime_gt_key_count = 0
    seal_error_count = 0
    posthoc_error_count = 0
    requery_stats: dict[str, dict[str, int]] = {variant: {"trigger_count": 0, "applied_count": 0} for variant in VARIANTS}
    runtime_variant_stats: dict[str, dict[str, int]] = {
        variant: {
            "frame_count": 0,
            "future_frame_count": 0,
            "model_score_changed_frame_count": 0,
            "target_assigned_frame_count": 0,
            "memory_read_frame_count": 0,
        }
        for variant in VARIANTS
    }
    runtime_comparison_stats: dict[str, dict[str, int]] = {
        comparison: {"future_frame_count": 0, "target_assignment_change_count": 0}
        for comparison in COMPARISONS
    }
    audited_events: list[dict[str, Any]] = []

    for event_id in expected_events:
        event_dir = replay_root / event_id
        event_errors: list[str] = []
        done_path = event_dir / "done.json"
        seal_path = event_dir / "runtime_event_sealed.json"
        posthoc_path = event_dir / "posthoc.json"
        if not done_path.is_file():
            event_errors.append("missing_done")
        if not seal_path.is_file():
            event_errors.append("missing_runtime_event_sealed")
        if not posthoc_path.is_file():
            event_errors.append("missing_posthoc")
        if event_errors:
            failures.append({"event_id": event_id, "errors": event_errors})
            errors.extend(f"{event_id}:{item}" for item in event_errors)
            continue
        done = read_json(done_path)
        seal = read_json(seal_path)
        posthoc = read_json(posthoc_path)
        if done.get("status") != "PASS_N72R9_RUNTIME_AND_POSTHOC_EVENT":
            event_errors.append(f"done_status={done.get('status')}")
        if done.get("event_id") != event_id:
            event_errors.append("done_event_id_mismatch")
        if done.get("runtime_future_gt_used") is not False:
            event_errors.append("done_runtime_future_gt_used_not_false")
        if done.get("posthoc_gt_used") is not True:
            event_errors.append("done_posthoc_gt_used_not_true")
        if done.get("runtime_event_sealed_sha256") != sha256_file(seal_path):
            event_errors.append("done_seal_hash_mismatch")
        if done.get("posthoc_sha256") != sha256_file(posthoc_path):
            event_errors.append("done_posthoc_hash_mismatch")
        if seal.get("status") != "PASS_N72R9_ALL_VARIANT_RUNTIME_SEALED":
            seal_error_count += 1
            event_errors.append(f"seal_status={seal.get('status')}")
        if seal.get("gt_loaded") is not False or seal.get("posthoc_gt_used") is not False or seal.get("runtime_future_gt_used") is not False:
            seal_error_count += 1
            event_errors.append("seal_gt_boundary_violation")
        if list(seal.get("variants", [])) != list(VARIANTS):
            seal_error_count += 1
            event_errors.append("seal_variant_axis_mismatch")
        _contains_forbidden_gt_key(seal, f"{event_id}/seal", event_errors)
        runtime_gt_key_count += len(event_errors)

        post_event = posthoc.get("event")
        if not isinstance(post_event, Mapping) or post_event.get("event_id") != event_id:
            posthoc_error_count += 1
            event_errors.append("posthoc_event_missing_or_mismatch")
        if posthoc.get("runtime_future_gt_used") is not False or posthoc.get("posthoc_gt_used") is not True:
            posthoc_error_count += 1
            event_errors.append("posthoc_gt_boundary_violation")
        if isinstance(post_event, Mapping):
            for comparison in COMPARISONS:
                for horizon in HORIZONS:
                    if str(horizon) not in post_event.get("comparisons", {}).get(comparison, {}):
                        posthoc_error_count += 1
                        event_errors.append(f"posthoc_missing_metric={comparison}/{horizon}")

        event_frame = int(seal.get("runtime_manifests", {}).get(VARIANTS[0], {}).get("event_frame", -1))
        target_public = seal.get("runtime_manifests", {}).get(VARIANTS[0], {}).get("target_public_id")
        runtime_rows_by_variant: dict[str, list[dict[str, Any]]] = {}
        for variant in VARIANTS:
            manifest_path = event_dir / variant / "runtime_manifest.json"
            if not manifest_path.is_file():
                event_errors.append(f"{variant}:missing_manifest")
                continue
            manifest = read_json(manifest_path)
            if manifest.get("status") != "PASS_N72R9_RUNTIME_ARTIFACT_SEALED":
                event_errors.append(f"{variant}:manifest_status={manifest.get('status')}")
            if manifest.get("event_id") != event_id or manifest.get("variant") != variant:
                event_errors.append(f"{variant}:manifest_identity_mismatch")
            if manifest.get("runtime_future_gt_used") is not False or manifest.get("runtime_gt_read") is not False or manifest.get("posthoc_gt_used") is not False:
                event_errors.append(f"{variant}:manifest_gt_boundary_violation")
            frames_path = resolved_path(str(manifest.get("frames", "")))
            if not frames_path.is_file():
                event_errors.append(f"{variant}:missing_frames")
                continue
            if manifest.get("frames_sha256") != sha256_file(frames_path):
                event_errors.append(f"{variant}:frames_hash_mismatch")
            rows = read_jsonl(frames_path)
            runtime_rows_by_variant[variant] = rows
            frame_rows += len(rows)
            runtime_variant_stats[variant]["frame_count"] += len(rows)
            if len(rows) != 101:
                frame_axis_error_count += 1
                event_errors.append(f"{variant}:frame_count={len(rows)}")
            expected_axis = list(range(event_frame, event_frame + 101))
            actual_axis = [int(row.get("frame", -1)) for row in rows]
            if actual_axis != expected_axis:
                frame_axis_error_count += 1
                event_errors.append(f"{variant}:frame_axis_mismatch")
            if not rows:
                continue
            first = rows[0]
            if first.get("record_kind") != "event_frame_correction" or first.get("candidate_rows") != [] or first.get("candidate_count") != 0:
                frame_axis_error_count += 1
                event_errors.append(f"{variant}:event_frame_payload_violation")
            if first.get("memory_read") is not False or first.get("event_frame_memory_read") is not False:
                frame_axis_error_count += 1
                event_errors.append(f"{variant}:event_frame_memory_read_violation")
            for index, row in enumerate(rows):
                if row.get("runtime_future_gt_used") is not False or row.get("runtime_gt_read") is not False or row.get("posthoc_gt_used") is not False or row.get("public_id_inference") is not False:
                    event_errors.append(f"{variant}:{row.get('frame')}:runtime_flag_violation")
                if int(row.get("event_frame", -1)) != event_frame or int(row.get("target_public_id", -1)) != int(target_public):
                    mapping_error_count += 1
                    event_errors.append(f"{variant}:{row.get('frame')}:authority_axis_violation")
                _contains_forbidden_gt_key(row, f"{event_id}/{variant}/{row.get('frame')}", event_errors)
                if index == 0:
                    continue
                runtime_variant_stats[variant]["future_frame_count"] += 1
                runtime_variant_stats[variant]["model_score_changed_frame_count"] += int(
                    bool((row.get("score_audit") or {}).get("model_score_changed", False))
                )
                runtime_variant_stats[variant]["target_assigned_frame_count"] += int(
                    isinstance(row.get("assignment"), Mapping)
                    and row.get("assignment", {}).get("target_assigned_candidate_uid") is not None
                )
                runtime_variant_stats[variant]["memory_read_frame_count"] += int(bool(row.get("memory_read")))
                if row.get("record_kind") != "future_association_frame" or int(row.get("frame_horizon", -1)) != index:
                    frame_axis_error_count += 1
                    event_errors.append(f"{variant}:{row.get('frame')}:future_row_contract_violation")
                if int(row.get("first_memory_visible_frame", -1)) != event_frame + 1:
                    frame_axis_error_count += 1
                    event_errors.append(f"{variant}:{row.get('frame')}:memory_visibility_violation")
                pool = row.get("candidate_pool")
                if not isinstance(pool, Mapping) or pool.get("runtime_future_gt_used") is not False or pool.get("public_id_inference") is not False:
                    mapping_error_count += 1
                    event_errors.append(f"{variant}:{row.get('frame')}:pool_audit_violation")
                    continue
                pool_rows = list(pool.get("candidate_rows", []))
                output_rows = list(row.get("candidate_rows", []))
                candidate_rows += len(output_rows)
                pool_uids = [str(item.get("candidate_uid")) for item in pool_rows]
                output_uids = [str(item.get("candidate_uid")) for item in output_rows]
                if len(pool_uids) != len(set(pool_uids)) or len(output_uids) != len(set(output_uids)):
                    candidate_duplicate_count += 1
                    event_errors.append(f"{variant}:{row.get('frame')}:candidate_uid_duplicate")
                if pool_uids != output_uids or int(row.get("candidate_count", -1)) != len(output_rows):
                    mapping_error_count += 1
                    event_errors.append(f"{variant}:{row.get('frame')}:candidate_axis_mismatch")
                if any(item.get("public_id") is not None or item.get("public_id_authority") is not None for item in pool_rows):
                    mapping_error_count += 1
                    event_errors.append(f"{variant}:{row.get('frame')}:source_pool_public_authority")
                for item in pool_rows:
                    if not item.get("candidate_source"):
                        mapping_error_count += 1
                        event_errors.append(f"{variant}:{row.get('frame')}:candidate_source_missing")
                score_audit = row.get("score_audit")
                if not isinstance(score_audit, Mapping):
                    matrix_error_count += 1
                    event_errors.append(f"{variant}:{row.get('frame')}:score_audit_missing")
                else:
                    matrix = np.asarray(score_audit.get("fused_score_matrix", []), dtype=np.float64)
                    state_axis = list(score_audit.get("association_state_axis", []))
                    public_axis = list(score_audit.get("public_id_axis", []))
                    if matrix.ndim != 2 or matrix.shape != (len(output_rows), len(state_axis)) or len(state_axis) != len(public_axis) or len(state_axis) != len(set(state_axis)) or len(public_axis) != len(set(public_axis)) or not np.isfinite(matrix).all():
                        matrix_error_count += 1
                        event_errors.append(f"{variant}:{row.get('frame')}:solver_matrix_invalid")
                    for key in ("base_target_scores", "fused_target_scores", "model_score_by_rank"):
                        values = np.asarray(score_audit.get(key, []), dtype=np.float64)
                        if values.size and (values.ndim != 1 or not np.isfinite(values).all()):
                            matrix_error_count += 1
                            event_errors.append(f"{variant}:{row.get('frame')}:{key}_invalid")
                assignment = row.get("assignment")
                solver = assignment.get("solver") if isinstance(assignment, Mapping) else None
                if not isinstance(solver, Mapping) or solver.get("runtime_future_gt_used") is not False:
                    mapping_error_count += 1
                    event_errors.append(f"{variant}:{row.get('frame')}:solver_audit_invalid")
                requery = row.get("requery")
                if isinstance(requery, Mapping):
                    requery_stats[variant]["trigger_count"] += int(bool(requery.get("triggered")))
                    requery_stats[variant]["applied_count"] += int(bool(requery.get("applied")))
                    if requery.get("runtime_future_gt_used") is not False:
                        event_errors.append(f"{variant}:{row.get('frame')}:requery_gt_violation")
            expected_memory_read = variant != "BASELINE_B0"
            if len(rows) > 1 and bool(rows[1].get("memory_read")) != expected_memory_read:
                frame_axis_error_count += 1
                event_errors.append(f"{variant}:t_plus_1_memory_read_mismatch")

        for comparison, (baseline, treatment) in COMPARISONS.items():
            baseline_rows = runtime_rows_by_variant.get(baseline, [])
            treatment_rows = runtime_rows_by_variant.get(treatment, [])
            if len(baseline_rows) == len(treatment_rows) == 101:
                runtime_comparison_stats[comparison]["future_frame_count"] += 100
                for baseline_row, treatment_row in zip(baseline_rows[1:], treatment_rows[1:]):
                    baseline_assignment = (baseline_row.get("assignment") or {}).get("target_assigned_candidate_uid")
                    treatment_assignment = (treatment_row.get("assignment") or {}).get("target_assigned_candidate_uid")
                    runtime_comparison_stats[comparison]["target_assignment_change_count"] += int(
                        baseline_assignment != treatment_assignment
                    )

        if event_errors:
            failures.append({"event_id": event_id, "errors": sorted(set(event_errors))})
            errors.extend(f"{event_id}:{item}" for item in sorted(set(event_errors)))
        else:
            audited_events.append({"event_id": event_id, "event_frame": event_frame, "target_public_id": int(target_public)})

    audit = {
        "schema_version": "N72R9_RUNTIME_AUDIT_V1",
        "status": "PASS_N72R9_RUNTIME_AND_POSTHOC_AUDIT" if not errors else "FAIL_N72R9_RUNTIME_AND_POSTHOC_AUDIT",
        "expected_event_count": len(expected_events),
        "unique_event_count": len(set(expected_events)),
        "audited_event_count": len(audited_events),
        "expected_variant_count": len(expected_events) * len(VARIANTS),
        "expected_runtime_frame_rows": len(expected_events) * len(VARIANTS) * 101,
        "runtime_frame_rows": frame_rows,
        "candidate_rows_audited": candidate_rows,
        "duplicate_event_ids": duplicate_batch_ids,
        "missing_event_ids": missing_batch_ids,
        "unexpected_event_ids": unexpected_batch_ids,
        "failed_events": failures,
        "candidate_duplicate_count": candidate_duplicate_count,
        "frame_axis_error_count": frame_axis_error_count,
        "matrix_error_count": matrix_error_count,
        "mapping_error_count": mapping_error_count,
        "seal_error_count": seal_error_count,
        "posthoc_error_count": posthoc_error_count,
        "runtime_future_gt_used": False,
        "runtime_gt_loaded_before_seal": False,
        "posthoc_gt_only_after_seal": not bool(errors),
        "requery_stats": requery_stats,
        "runtime_variant_stats": runtime_variant_stats,
        "runtime_comparison_stats": runtime_comparison_stats,
        "errors": errors,
        "audited_events": audited_events,
    }
    return audit, failures, errors


def _metric_template() -> dict[str, Any]:
    return {
        "window_frame_count": 0,
        "evaluated_frames": 0,
        "target_gt_visible_frames": 0,
        "target_gt_absent_frames": 0,
        "baseline_iou_sum": 0.0,
        "treatment_iou_sum": 0.0,
        "delta_iou_sum": 0.0,
        "baseline_correct_frames": 0,
        "treatment_correct_frames": 0,
        "baseline_identity_error_frames": 0,
        "treatment_identity_error_frames": 0,
        "identity_error_reduction_sum": 0.0,
        "target_missing_frames": 0,
        "wrong_reassociation_frames": 0,
        "candidate_present_frames": 0,
        "assignment_change_count": 0,
        "target_assignment_change_count": 0,
        "global_common_assignment_change_count": 0,
        "true_correct_crossing_count": 0,
        "true_incorrect_crossing_count": 0,
        "directional_improvement_count": 0,
        "directional_regression_count": 0,
        "neutral_change_count": 0,
        "id_switch_count": 0,
        "recorrection_opportunity_count": 0,
        "raw_switch_count": 0,
        "posthoc_correct_switch_count": 0,
        "posthoc_wrong_switch_count": 0,
        "posthoc_unassessable_switch_count": 0,
        "protected_compared": 0,
        "protected_regression_count": 0,
        "protected_improvement_count": 0,
    }


def _merge_metric(destination: dict[str, Any], source: Mapping[str, Any]) -> None:
    for key, value in source.items():
        if key.endswith(("_sum", "_frames", "_count", "_compared")) or key in {"evaluated_frames", "window_frame_count"}:
            if isinstance(value, (int, float)) and math.isfinite(float(value)):
                destination[key] = destination.get(key, 0) + value


def _finalize_metric(metric: dict[str, Any]) -> dict[str, Any]:
    frames = int(metric.get("evaluated_frames", 0))
    if frames:
        for name, numerator in (
            ("baseline_mean_iou", "baseline_iou_sum"),
            ("treatment_mean_iou", "treatment_iou_sum"),
            ("delta_iou", "delta_iou_sum"),
            ("baseline_identity_error", "baseline_identity_error_frames"),
            ("treatment_identity_error", "treatment_identity_error_frames"),
            ("identity_error_reduction", "identity_error_reduction_sum"),
            ("missing_rate", "target_missing_frames"),
            ("wrong_reassociation_rate", "wrong_reassociation_frames"),
            ("candidate_recall", "candidate_present_frames"),
            ("assignment_change_rate", "assignment_change_count"),
            ("target_assignment_change_rate", "target_assignment_change_count"),
            ("id_switch_rate", "id_switch_count"),
            ("recorrection_rate", "recorrection_opportunity_count"),
        ):
            metric[name] = float(metric[numerator] / frames)
    compared = int(metric.get("protected_compared", 0))
    metric["protected_regression_rate"] = float(metric["protected_regression_count"] / compared) if compared else None
    return metric


def _bootstrap(values_by_sequence: Mapping[str, Sequence[float]], seed: int) -> dict[str, Any]:
    names = sorted(values_by_sequence)
    if not names:
        return {
            "mean": None,
            "lower": None,
            "upper": None,
            "clusters": 0,
            "unit": "independent_sequence",
            "within_cluster_aggregation": "mean_event_value",
            "seed": int(seed),
            "repetitions": BOOTSTRAP_REPETITIONS,
        }
    sequence_values = np.asarray([float(np.mean(values_by_sequence[name])) for name in names], dtype=np.float64)
    if not np.isfinite(sequence_values).all():
        raise ValueError("non-finite sequence metric")
    rng = np.random.default_rng(int(seed))
    samples = np.empty(BOOTSTRAP_REPETITIONS, dtype=np.float64)
    for index in range(BOOTSTRAP_REPETITIONS):
        draw = rng.integers(0, len(sequence_values), size=len(sequence_values))
        samples[index] = float(np.mean(sequence_values[draw]))
    return {
        "mean": float(np.mean(sequence_values)),
        "lower": float(np.quantile(samples, 0.025)),
        "upper": float(np.quantile(samples, 0.975)),
        "clusters": len(names),
        "unit": "independent_sequence",
        "within_cluster_aggregation": "mean_event_value",
        "sequence_means": {name: float(value) for name, value in zip(names, sequence_values)},
        "seed": int(seed),
        "repetitions": BOOTSTRAP_REPETITIONS,
    }


def aggregate_events(events: Sequence[Mapping[str, Any]], comparison: str, horizon: int, action: str | None = None) -> dict[str, Any]:
    selected = [event for event in events if action is None or event["action_type"] == action]
    metric = _metric_template()
    values_by_sequence: dict[str, list[float]] = defaultdict(list)
    for event in selected:
        source = event["comparisons"][comparison][str(horizon)]
        _merge_metric(metric, source)
        values_by_sequence[str(event["sequence"])].append(float(source.get("identity_error_reduction") or 0.0))
    metric = _finalize_metric(metric)
    comparison_index = list(COMPARISONS).index(comparison)
    horizon_index = HORIZONS.index(horizon)
    metric["sequence_cluster_bootstrap_95ci"] = _bootstrap(
        values_by_sequence,
        seed=BOOTSTRAP_SEED + comparison_index * 100 + horizon_index + (0 if action is None else 1000 + ACTION_TYPES.index(action) * 10),
    )
    metric["event_count"] = len(selected)
    metric["independent_sequence_count"] = len({str(event["sequence"]) for event in selected})
    metric["comparison"] = comparison
    metric["horizon"] = int(horizon)
    if action is not None:
        metric["action_type"] = action
    return metric


def _load_event_metrics(replay_root: Path, expected_events: Sequence[str]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    events: list[dict[str, Any]] = []
    refs: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for event_id in expected_events:
        path = replay_root / event_id / "posthoc.json"
        if not path.is_file():
            failures.append({"event_id": event_id, "error": "missing_posthoc"})
            continue
        posthoc = read_json(path)
        event = posthoc.get("event")
        if not isinstance(event, Mapping):
            failures.append({"event_id": event_id, "error": "posthoc_event_missing"})
            continue
        event_copy = dict(event)
        events.append(event_copy)
        refs.append({
            "event_id": event_id,
            "sequence": event_copy.get("sequence"),
            "action_type": event_copy.get("action_type"),
            "posthoc": str(path),
            "posthoc_sha256": sha256_file(path),
            "metric_summary": {
                comparison: {
                    str(horizon): {
                        key: event_copy["comparisons"][comparison][str(horizon)].get(key)
                        for key in (
                            "identity_error_reduction",
                            "delta_iou",
                            "missing_rate",
                            "wrong_reassociation_rate",
                            "candidate_recall",
                            "assignment_change_count",
                            "true_correct_crossing_count",
                            "true_incorrect_crossing_count",
                            "raw_switch_count",
                            "posthoc_correct_switch_count",
                            "posthoc_wrong_switch_count",
                            "protected_regression_count",
                        )
                    }
                    for horizon in HORIZONS
                }
                for comparison in COMPARISONS
            },
        })
    return events, refs, failures


def _requery_milestone_audit(replay_root: Path, expected_events: Sequence[str]) -> dict[str, Any]:
    """Find complete and partial causal requery cases without changing outcomes."""

    counts = {
        "triggered": 0,
        "applied": 0,
        "applied_assignment_changed": 0,
        "applied_raw_binding_switch": 0,
        "applied_different_selector_choice": 0,
        "applied_posthoc_wrong_to_correct": 0,
        "applied_public_id_immutable": 0,
        "applied_safe_memory_update": 0,
        "complete_milestone": 0,
    }
    complete: list[dict[str, Any]] = []
    partial: list[dict[str, Any]] = []
    for event_id in expected_events:
        event_dir = replay_root / event_id
        q_path = event_dir / "TEMPORAL_REQUERY" / "runtime_frames.jsonl"
        c_path = event_dir / "TEMPORAL_CURRENT" / "runtime_frames.jsonl"
        posthoc_path = event_dir / "posthoc.json"
        if not q_path.is_file() or not c_path.is_file() or not posthoc_path.is_file():
            continue
        q_rows = read_jsonl(q_path)
        c_rows = read_jsonl(c_path)
        posthoc = read_json(posthoc_path).get("event", {})
        detail_by_frame = {
            int(item["frame"]): item
            for item in posthoc.get("comparisons", {}).get("TEMPORAL_REQUERY_vs_TEMPORAL_CURRENT", {}).get("100", {}).get("frame_details", [])
        }
        for q_row, c_row in zip(q_rows[1:], c_rows[1:]):
            query = q_row.get("requery") or {}
            if bool(query.get("triggered")):
                counts["triggered"] += 1
            if not bool(query.get("applied")):
                continue
            counts["applied"] += 1
            q_assignment = (q_row.get("assignment") or {}).get("target_assigned_candidate_uid")
            c_assignment = (c_row.get("assignment") or {}).get("target_assigned_candidate_uid")
            assignment_changed = q_assignment != c_assignment
            raw_switch = q_row.get("raw_binding_switch") is not None
            q_selected = (q_row.get("selection_audit") or {}).get("selected_candidate_uid")
            c_selected = (c_row.get("selection_audit") or {}).get("selected_candidate_uid")
            different_selector_choice = q_selected is not None and c_selected is not None and q_selected != c_selected
            immutable = bool(
                (q_row.get("raw_binding_switch") or {}).get("public_id_changed") is False
                and q_row.get("target_public_id") == c_row.get("target_public_id")
                and bool((q_row.get("assignment") or {}).get("solver_public_id_immutable", False))
            )
            safe_memory = q_row.get("trusted_memory_update") in {"NO_TRUSTED_UPDATE", "CAUSAL_TARGET_ASSIGNMENT"}
            detail = detail_by_frame.get(int(q_row["frame"]), {})
            wrong_to_correct = bool(detail.get("baseline_correct") is False and detail.get("treatment_correct") is True)
            counts["applied_assignment_changed"] += int(assignment_changed)
            counts["applied_raw_binding_switch"] += int(raw_switch)
            counts["applied_different_selector_choice"] += int(different_selector_choice)
            counts["applied_posthoc_wrong_to_correct"] += int(wrong_to_correct)
            counts["applied_public_id_immutable"] += int(immutable)
            counts["applied_safe_memory_update"] += int(safe_memory)
            evidence = {
                "event_id": event_id,
                "sequence": q_row.get("sequence"),
                "frame": int(q_row["frame"]),
                "trigger": query,
                "current_assignment_uid": c_assignment,
                "requery_assignment_uid": q_assignment,
                "current_selector_uid": c_selected,
                "requery_selector_uid": q_selected,
                "assignment_changed": assignment_changed,
                "raw_binding_switch": raw_switch,
                "public_id_immutable": immutable,
                "safe_memory_update": safe_memory,
                "posthoc_wrong_to_correct": wrong_to_correct,
                "posthoc_detail": detail,
                "runtime_future_gt_used": False,
                "artifact": str(event_dir),
            }
            complete_case = bool(
                assignment_changed
                and raw_switch
                and different_selector_choice
                and immutable
                and safe_memory
                and wrong_to_correct
            )
            if complete_case:
                counts["complete_milestone"] += 1
                complete.append(evidence)
            elif assignment_changed and raw_switch and wrong_to_correct:
                partial.append(evidence)
    return {
        "schema_version": "N72R9_REQUERY_MILESTONE_AUDIT_V1",
        "definition": {
            "complete_requires": [
                "runtime_uncertainty_triggered_and_requery_applied",
                "selector_choice_changes_to_a_different_candidate",
                "target_assignment_changes",
                "raw_binding_switch_recorded",
                "public_id_immutable",
                "causal_memory_update_is_audited_safe",
                "posthoc_wrong_to_correct",
            ],
            "runtime_future_gt_used": False,
            "posthoc_only_field": "posthoc_wrong_to_correct",
        },
        "counts": counts,
        "complete_cases": complete,
        "partial_cases": partial[:20],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--replay-root", default=str(DEFAULT_REPLAY_ROOT))
    parser.add_argument("--result-path", default=str(DEFAULT_RESULT_PATH))
    parser.add_argument("--event-metrics-path", default=str(DEFAULT_EVENT_METRICS_PATH))
    parser.add_argument("--audit-path", default=str(DEFAULT_AUDIT_PATH))
    parser.add_argument("--stage-path", default=str(DEFAULT_STAGE_PATH))
    args = parser.parse_args()

    protocol = read_json(PROTOCOL_PATH)
    batch_path = Path(args.replay_root) / "batch_manifest.json"
    batch = read_json(batch_path)
    expected_events = [str(item["event_id"]) for item in protocol.get("source_event_selection", {}).get("events", [])]
    audit, audit_failures, audit_errors = _runtime_audit(protocol, batch, Path(args.replay_root))
    events, refs, posthoc_failures = _load_event_metrics(Path(args.replay_root), expected_events)
    milestone = _requery_milestone_audit(Path(args.replay_root), expected_events)
    if posthoc_failures:
        audit_errors.extend(f"posthoc:{item['event_id']}:{item['error']}" for item in posthoc_failures)
        audit["errors"] = sorted(set(audit.get("errors", []) + [str(item) for item in posthoc_failures]))
        audit["status"] = "FAIL_N72R9_RUNTIME_AND_POSTHOC_AUDIT"
    if len(events) != len(expected_events):
        audit_errors.append(f"posthoc_event_count={len(events)} expected={len(expected_events)}")
    if len({str(event.get("event_id")) for event in events}) != len(events):
        audit_errors.append("duplicate_posthoc_event_ids")
    if audit_errors:
        audit["errors"] = sorted(set(audit.get("errors", []) + audit_errors))
        audit["status"] = "FAIL_N72R9_RUNTIME_AND_POSTHOC_AUDIT"

    result: dict[str, Any] = {
        "schema_version": "N72R9_TEMPORAL_REPLAY_AGGREGATE_V1",
        "status": "PASS_N72R9_DEVELOPMENT_REPLAY_AGGREGATED" if not audit["errors"] else "FAIL_N72R9_DEVELOPMENT_REPLAY_AGGREGATION",
        "created_at_utc": now_utc(),
        "protocol": str(PROTOCOL_PATH),
        "protocol_sha256": sha256_file(PROTOCOL_PATH),
        "batch_manifest": str(batch_path),
        "batch_manifest_sha256": sha256_file(batch_path),
        "event_count": len(events),
        "sequence_count": len({str(event["sequence"]) for event in events}),
        "action_counts": {action: sum(str(event.get("action_type")) == action for event in events) for action in ACTION_TYPES},
        "interaction_source": "simulated_from_gt",
        "not_real_human_evidence": True,
        "runtime_future_gt_used": False,
        "posthoc_gt_usage": "posthoc_only_after_runtime_event_seal",
        "fresh_confirmation_available": False,
        "fresh_confirmation_reason": "N72R9 reservation found zero eligible untouched sequences; this is development evidence only",
        "runtime_audit": audit,
        "event_metric_refs": refs,
        "aggregates": {
            comparison: {
                str(horizon): aggregate_events(events, comparison, horizon)
                for horizon in HORIZONS
            }
            for comparison in COMPARISONS
        },
        "by_action": {
            action: {
                comparison: {
                    str(horizon): aggregate_events(events, comparison, horizon, action=action)
                    for horizon in HORIZONS
                }
                for comparison in COMPARISONS
            }
            for action in ACTION_TYPES
        },
        "by_sequence": {
            sequence: {
                comparison: {
                    str(horizon): aggregate_events(
                        [event for event in events if str(event["sequence"]) == sequence], comparison, horizon
                    )
                    for horizon in HORIZONS
                }
                for comparison in COMPARISONS
            }
            for sequence in sorted({str(event["sequence"]) for event in events})
        },
        "requery": {
            "triggered_and_applied_only_from_runtime_uncertainty": True,
            "stats": audit.get("requery_stats", {}),
            "milestone_audit": milestone,
        },
        "gate": {},
    }

    combined = "TEMPORAL_CURRENT_vs_BASELINE_B0"
    requery = "TEMPORAL_REQUERY_vs_TEMPORAL_CURRENT"
    combined_horizons = result["aggregates"][combined]
    requery_horizons = result["aggregates"][requery]
    combined_ci_lower = {horizon: combined_horizons[str(horizon)]["sequence_cluster_bootstrap_95ci"]["lower"] for horizon in HORIZONS}
    requery_ci_lower = {horizon: requery_horizons[str(horizon)]["sequence_cluster_bootstrap_95ci"]["lower"] for horizon in HORIZONS}
    protected_regression_count = int(combined_horizons["20"].get("protected_regression_count", 0))
    combined_signal = all(value is not None and float(value) > 0.0 for value in combined_ci_lower.values())
    requery_signal = all(value is not None and float(value) > 0.0 for value in requery_ci_lower.values())
    strict_gate_pass = bool(not audit["errors"] and requery_signal and protected_regression_count == 0 and result["fresh_confirmation_available"])
    development_signal = bool(not audit["errors"] and combined_signal)
    if strict_gate_pass:
        research_gate = "PASS_N72R9_DEVELOPMENT_AND_CONFIRMATION"
    elif development_signal and not requery_signal:
        research_gate = "FAIL_FUTURE_REQUERY_EFFECT"
    elif development_signal and protected_regression_count > 0:
        research_gate = "FAIL_PROTECTED_IDENTITY_REGRESSION_NO_CONFIRMATION"
    elif development_signal:
        research_gate = "DEVELOPMENT_SIGNAL_NO_FRESH_CONFIRMATION"
    else:
        research_gate = "FAIL_FUTURE_EFFECT"
    result["gate"] = {
        "combined_comparison": combined,
        "requery_incremental_comparison": requery,
        "combined_sequence_cluster_ci_lower": combined_ci_lower,
        "requery_sequence_cluster_ci_lower": requery_ci_lower,
        "combined_ci_lower_gt_zero_all_horizons": combined_signal,
        "requery_ci_lower_gt_zero_all_horizons": requery_signal,
        "strict_ci_lower_gt_zero_all_horizons": requery_signal,
        "requery_complete_milestone_count": milestone["counts"]["complete_milestone"],
        "development_future_effect_signal": development_signal,
        "protected_regression_count_h20": protected_regression_count,
        "runtime_integrity_pass": not bool(audit["errors"]),
        "fresh_confirmation_required_for_final_authorization": True,
        "fresh_confirmation_available": False,
        "research_gate": research_gate,
        "production_authorized": False,
        "strict_gate_pass": strict_gate_pass,
        "interpretation": "Development replay only; no untouched confirmation authorization is possible in this branch.",
    }
    result["event_metric_refs"] = refs

    atomic_json(Path(args.audit_path), audit)
    atomic_jsonl(Path(args.event_metrics_path), refs)
    atomic_json(Path(args.result_path), result)
    stage = {
        "schema_version": "N72R9_STAGE_07_REPLAY_STATUS_V1",
        "status": "PASS_N72R9_DEVELOPMENT_REPLAY_AGGREGATED" if not audit["errors"] else "FAIL_N72R9_DEVELOPMENT_REPLAY_AGGREGATION",
        "created_at_utc": now_utc(),
        "command": "scripts/n72r9_aggregate_replay.py",
        "input_protocol_sha256": result["protocol_sha256"],
        "input_batch_manifest_sha256": result["batch_manifest_sha256"],
        "event_count": len(events),
        "sequence_count": result["sequence_count"],
        "runtime_audit": str(Path(args.audit_path)),
        "result": str(Path(args.result_path)),
        "event_metric_refs": str(Path(args.event_metrics_path)),
        "audit_errors": audit.get("errors", []),
        "research_gate": result["gate"]["research_gate"],
        "production_authorized": False,
        "interaction_source": "simulated_from_gt",
        "not_real_human_evidence": True,
        "runtime_future_gt_used": False,
    }
    atomic_json(Path(args.stage_path), stage)
    print(json.dumps({
        "status": result["status"],
        "event_count": len(events),
        "sequence_count": result["sequence_count"],
        "audit_errors": len(audit.get("errors", [])),
        "research_gate": result["gate"]["research_gate"],
        "production_authorized": False,
        "result": str(Path(args.result_path)),
        "audit": str(Path(args.audit_path)),
        "stage": str(Path(args.stage_path)),
    }, sort_keys=True))
    return 0 if result["status"].startswith("PASS") else 1


if __name__ == "__main__":
    raise SystemExit(main())
