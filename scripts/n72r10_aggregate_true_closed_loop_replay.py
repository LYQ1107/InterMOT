#!/usr/bin/env python3
"""Audit and aggregate the sealed N72R10 E0/E1/E2 replay.

This is a post-run CPU-only auditor.  It never runs association and never
uses GT to repair runtime rows.  GT-bearing posthoc metrics are consumed only
after the runtime seal and its hashes have been verified.
"""

from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in __import__("sys").path:
    __import__("sys").path.insert(0, str(ROOT))

import scripts.n72r9_temporal_replay as legacy  # noqa: E402


PROTOCOL_PATH = ROOT / "outputs/N72R9/protocol.json"
DEFAULT_REPLAY_ROOT = ROOT / "outputs/N72R10/stage_07_replay/attempt_01"
DEFAULT_BATCH_PATH = DEFAULT_REPLAY_ROOT / "batch_manifest.json"
DEFAULT_AUDIT_PATH = ROOT / "outputs/N72R10/stage_07_replay/runtime_audit.json"
DEFAULT_EVENT_METRICS_PATH = ROOT / "outputs/N72R10/stage_07_replay/event_metrics.jsonl"
DEFAULT_RESULT_PATH = ROOT / "outputs/N72R10/ccam_paired_replay_results.json"
DEFAULT_STAGE_PATH = ROOT / "outputs/N72R10/stage_08_status.json"

OLD_VARIANTS = ("BASELINE_B0", "TEMPORAL_CURRENT", "TEMPORAL_REQUERY")
VARIANT_ALIASES = {
    "BASELINE_B0": "E0_B0",
    "TEMPORAL_CURRENT": "E1_TEMPORAL_CURRENT_V2",
    "TEMPORAL_REQUERY": "E2_TRUE_CLOSED_LOOP_REQUERY_V2",
}
HORIZONS = (20, 50, 100)
OLD_COMPARISONS = {
    "TEMPORAL_CURRENT_vs_BASELINE_B0": ("BASELINE_B0", "TEMPORAL_CURRENT"),
    "TEMPORAL_REQUERY_vs_BASELINE_B0": ("BASELINE_B0", "TEMPORAL_REQUERY"),
    "TEMPORAL_REQUERY_vs_TEMPORAL_CURRENT": ("TEMPORAL_CURRENT", "TEMPORAL_REQUERY"),
}
COMPARISON_ALIASES = {
    "TEMPORAL_CURRENT_vs_BASELINE_B0": "E1_vs_E0",
    "TEMPORAL_REQUERY_vs_BASELINE_B0": "E2_vs_E0",
    "TEMPORAL_REQUERY_vs_TEMPORAL_CURRENT": "E2_vs_E1",
}
RUNTIME_FLAGS = ("runtime_future_gt_used", "runtime_gt_read", "posthoc_gt_used", "public_id_inference")
FORBIDDEN_RUNTIME_KEYS = {"dataset_gt_id", "gt_box", "future_gt", "future_identity_error", "gt_target", "gt_id"}


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


def atomic_jsonl(path: Path, rows: list[Mapping[str, Any]]) -> None:
    atomic_write(path, "".join(json.dumps(dict(row), ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n" for row in rows))


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
                raise TypeError(f"{path}:{line_number}: expected object")
            rows.append(value)
    return rows


def resolved_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def _runtime_forbidden_scan(value: Any, location: str, errors: list[str]) -> None:
    if isinstance(value, Mapping):
        for key, nested in value.items():
            key_text = str(key).lower()
            if key_text in FORBIDDEN_RUNTIME_KEYS:
                errors.append(f"{location}/{key}")
            if key_text in RUNTIME_FLAGS and nested is not False:
                errors.append(f"{location}/{key}=not_false")
            _runtime_forbidden_scan(nested, f"{location}/{key}", errors)
    elif isinstance(value, list):
        for index, nested in enumerate(value):
            _runtime_forbidden_scan(nested, f"{location}/{index}", errors)


def _finite_matrix(value: Any, shape: tuple[int, int], label: str, errors: list[str]) -> None:
    matrix = np.asarray(value, dtype=np.float64)
    if matrix.shape != shape or not np.isfinite(matrix).all():
        errors.append(f"{label}:matrix_shape_or_finite={matrix.shape}")


def _audit_event(event_id: str, event_dir: Path, expected_event: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any], list[str]]:
    errors: list[str] = []
    done_path = event_dir / "done.json"
    seal_path = event_dir / "runtime_event_sealed.json"
    posthoc_path = event_dir / "posthoc.json"
    for path in (done_path, seal_path, posthoc_path):
        if not path.is_file():
            errors.append(f"missing:{path.name}")
    if errors:
        return {}, {}, errors
    done, seal, posthoc = read_json(done_path), read_json(seal_path), read_json(posthoc_path)
    if done.get("status") != "PASS_N72R10_RUNTIME_AND_POSTHOC_EVENT":
        errors.append(f"done_status={done.get('status')}")
    if done.get("event_id") != event_id:
        errors.append("done_event_id_mismatch")
    for timestamp_key in ("started_at_utc", "finished_at_utc"):
        timestamp = done.get(timestamp_key)
        if not isinstance(timestamp, str) or not timestamp.strip():
            errors.append(f"done_{timestamp_key}_missing")
    if done.get("runtime_event_sealed_sha256") != sha256_file(seal_path):
        errors.append("done_seal_hash_mismatch")
    if done.get("posthoc_sha256") != sha256_file(posthoc_path):
        errors.append("done_posthoc_hash_mismatch")
    if seal.get("status") != "PASS_N72R10_ALL_VARIANT_RUNTIME_SEALED":
        errors.append(f"seal_status={seal.get('status')}")
    if seal.get("gt_loaded") is not False or seal.get("runtime_future_gt_used") is not False or seal.get("posthoc_gt_used") is not False:
        errors.append("seal_gt_boundary_violation")
    if list(seal.get("variants", [])) != list(OLD_VARIANTS):
        errors.append("seal_variant_axis_mismatch")
    _runtime_forbidden_scan(seal, f"{event_id}/seal", errors)
    if not isinstance(posthoc.get("event"), Mapping) or posthoc["event"].get("event_id") != event_id:
        errors.append("posthoc_event_missing_or_mismatch")
    if posthoc.get("runtime_future_gt_used") is not False or posthoc.get("posthoc_gt_used") is not True:
        errors.append("posthoc_gt_boundary_violation")
    event_frame = int(expected_event["event_frame"])
    target_public = int(expected_event.get("target_public_id") or 0)
    if target_public <= 0:
        target_public = int(seal.get("runtime_manifests", {}).get("BASELINE_B0", {}).get("target_public_id", 0))
    runtime_rows: dict[str, list[dict[str, Any]]] = {}
    candidate_source_counts: Counter[str] = Counter()
    runtime_stats: dict[str, Any] = {}
    future_requery_injected = 0
    for variant in OLD_VARIANTS:
        manifest_path = event_dir / variant / "runtime_manifest.json"
        if not manifest_path.is_file():
            errors.append(f"{variant}:missing_runtime_manifest")
            continue
        manifest = read_json(manifest_path)
        if manifest.get("status") != "PASS_N72R10_RUNTIME_ARTIFACT_SEALED":
            errors.append(f"{variant}:manifest_status={manifest.get('status')}")
        if manifest.get("event_id") != event_id or manifest.get("variant") != variant:
            errors.append(f"{variant}:manifest_identity_mismatch")
        if manifest.get("runtime_future_gt_used") is not False or manifest.get("runtime_gt_read") is not False or manifest.get("posthoc_gt_used") is not False:
            errors.append(f"{variant}:manifest_gt_boundary_violation")
        frames_path = resolved_path(str(manifest.get("frames", "")))
        if not frames_path.is_file():
            errors.append(f"{variant}:missing_frames")
            continue
        if manifest.get("frames_sha256") != sha256_file(frames_path):
            errors.append(f"{variant}:frames_hash_mismatch")
        rows = read_jsonl(frames_path)
        runtime_rows[variant] = rows
        if len(rows) != 101 or [int(row.get("frame", -1)) for row in rows] != list(range(event_frame, event_frame + 101)):
            errors.append(f"{variant}:frame_axis_invalid")
        if not rows:
            errors.append(f"{variant}:empty_runtime_rows")
            continue
        first = rows[0]
        if first.get("record_kind") != "event_frame_correction" or first.get("candidate_rows") != [] or int(first.get("candidate_count", -1)) != 0:
            errors.append(f"{variant}:event_frame_payload_invalid")
        if first.get("memory_read") is not False or first.get("event_frame_memory_read") is not False or first.get("runtime_future_gt_used") is not False:
            errors.append(f"{variant}:event_frame_causal_or_gt_violation")
        model_changed = 0
        target_assigned = 0
        future_rows = 0
        source_counts: Counter[str] = Counter()
        for offset, row in enumerate(rows):
            _runtime_forbidden_scan(row, f"{event_id}/{variant}/{row.get('frame')}", errors)
            if int(row.get("event_frame", -1)) != event_frame or int(row.get("target_public_id", -1)) != target_public:
                errors.append(f"{variant}:{row.get('frame')}:authority_axis_invalid")
            if row.get("public_id_immutable") is not True:
                errors.append(f"{variant}:{row.get('frame')}:public_id_immutable_not_true")
            if offset == 0:
                continue
            future_rows += 1
            if row.get("record_kind") != "future_association_frame" or int(row.get("frame_horizon", -1)) != offset:
                errors.append(f"{variant}:{row.get('frame')}:future_row_contract_invalid")
            if int(row.get("first_memory_visible_frame", -1)) != event_frame + 1:
                errors.append(f"{variant}:{row.get('frame')}:memory_visibility_invalid")
            expected_memory = variant != "BASELINE_B0"
            if bool(row.get("memory_read")) != expected_memory:
                errors.append(f"{variant}:{row.get('frame')}:memory_read_invalid")
            score_audit = row.get("score_audit")
            assignment = row.get("assignment")
            pool = row.get("candidate_pool")
            output_rows = list(row.get("candidate_rows", []))
            if not isinstance(pool, Mapping):
                errors.append(f"{variant}:{row.get('frame')}:candidate_pool_missing")
                continue
            pool_rows = list(pool.get("candidate_rows", []))
            pool_uids = [str(item.get("candidate_uid")) for item in pool_rows]
            output_uids = [str(item.get("candidate_uid")) for item in output_rows]
            if len(pool_uids) != len(set(pool_uids)) or len(output_uids) != len(set(output_uids)) or pool_uids != output_uids:
                errors.append(f"{variant}:{row.get('frame')}:candidate_uid_axis_invalid")
            if int(row.get("candidate_count", -1)) != len(output_rows) or int(pool.get("candidate_count", -1)) != len(pool_rows):
                errors.append(f"{variant}:{row.get('frame')}:candidate_count_invalid")
            if pool.get("runtime_future_gt_used") is not False or pool.get("runtime_gt_read") is not False or pool.get("posthoc_gt_used") is not False or pool.get("public_id_inference") is not False:
                errors.append(f"{variant}:{row.get('frame')}:pool_boundary_invalid")
            for item in pool_rows:
                if item.get("public_id") is not None or item.get("public_id_authority") is not None:
                    errors.append(f"{variant}:{row.get('frame')}:source_pool_public_authority")
                source = str(item.get("candidate_source"))
                source_counts[source] += 1
                candidate_source_counts[source] += 1
                feature = item.get("feature")
                if feature is not None:
                    array = np.asarray(feature, dtype=np.float64).reshape(-1)
                    if array.size != 512 or not np.isfinite(array).all() or float(np.linalg.norm(array)) <= 1.0e-6:
                        errors.append(f"{variant}:{row.get('frame')}:feature_invalid")
            if variant != "TEMPORAL_REQUERY" and source_counts.get("FUTURE_FRAME_REQUERY", 0):
                errors.append(f"{variant}:future_source_outside_E2")
            frame_future_count = sum(
                1 for item in pool_rows if str(item.get("candidate_source")) == "FUTURE_FRAME_REQUERY"
            )
            future_requery_injected += frame_future_count
            if not isinstance(score_audit, Mapping):
                errors.append(f"{variant}:{row.get('frame')}:score_audit_missing")
            else:
                state_axis = list(score_audit.get("association_state_axis", []))
                public_axis = list(score_audit.get("public_id_axis", []))
                _finite_matrix(score_audit.get("fused_score_matrix", []), (len(output_rows), len(state_axis)), f"{variant}:{row.get('frame')}", errors)
                if len(state_axis) != len(public_axis) or len(state_axis) != len(set(state_axis)) or len(public_axis) != len(set(public_axis)):
                    errors.append(f"{variant}:{row.get('frame')}:solver_axis_invalid")
                for key in ("base_target_scores", "fused_target_scores", "model_score_by_rank"):
                    values = np.asarray(score_audit.get(key, []), dtype=np.float64)
                    if values.ndim != 1 or len(values) != len(output_rows) or not np.isfinite(values).all():
                        errors.append(f"{variant}:{row.get('frame')}:{key}_invalid")
                model_changed += int(bool(score_audit.get("model_score_changed", False)))
            if not isinstance(assignment, Mapping) or not isinstance(assignment.get("solver"), Mapping):
                errors.append(f"{variant}:{row.get('frame')}:assignment_solver_missing")
            else:
                solver = assignment["solver"]
                if solver.get("runtime_future_gt_used") is not False:
                    errors.append(f"{variant}:{row.get('frame')}:solver_gt_boundary_invalid")
                public_ids = [item.get("public_id") for item in solver.get("assignment_rows", []) if item.get("public_id") is not None]
                if len(public_ids) != len(set(public_ids)):
                    errors.append(f"{variant}:{row.get('frame')}:duplicate_solver_public_id")
                target_assigned += int(assignment.get("target_assigned_candidate_uid") is not None)
            requery = row.get("requery")
            if isinstance(requery, Mapping) and requery.get("applied"):
                if variant != "TEMPORAL_REQUERY" or requery.get("runtime_future_gt_used") is not False:
                    errors.append(f"{variant}:{row.get('frame')}:invalid_requery_application")
                if str(requery.get("source")) != str(manifest.get("future_requery_provenance", {}).get("candidates")):
                    errors.append(f"{variant}:{row.get('frame')}:future_source_provenance_mismatch")
        runtime_stats[variant] = {
            "future_frame_count": future_rows,
            "model_score_changed_frame_count": model_changed,
            "model_score_changed_rate": float(model_changed / future_rows) if future_rows else None,
            "target_assigned_frame_count": target_assigned,
            "target_assignment_rate": float(target_assigned / future_rows) if future_rows else None,
            "candidate_source_counts": dict(sorted(source_counts.items())),
        }
    if not runtime_rows.keys() >= set(OLD_VARIANTS):
        errors.append("runtime_variant_set_incomplete")
    posthoc_event = dict(posthoc.get("event", {})) if isinstance(posthoc.get("event"), Mapping) else {}
    for old_comparison in OLD_COMPARISONS:
        for horizon in HORIZONS:
            if str(horizon) not in posthoc_event.get("comparisons", {}).get(old_comparison, {}):
                errors.append(f"posthoc_metric_missing:{old_comparison}/{horizon}")
    summary = {
        "event_id": event_id,
        "sequence": str(expected_event["sequence"]),
        "action_type": str(expected_event["action_type"]),
        "event_frame": event_frame,
        "target_public_id": target_public,
        "runtime_stats": runtime_stats,
        "future_requery_candidate_rows_in_E2": future_requery_injected,
        "runtime_future_gt_used": False,
        "posthoc_gt_used": True,
        "runtime_event_sealed": str(seal_path),
        "runtime_event_sealed_sha256": sha256_file(seal_path),
        "posthoc": str(posthoc_path),
        "posthoc_sha256": sha256_file(posthoc_path),
    }
    return summary, posthoc_event, sorted(set(errors))


def _strip_frame_details(value: Mapping[str, Any]) -> dict[str, Any]:
    return {str(key): nested for key, nested in value.items() if str(key) != "frame_details"}


def _sum_source_counts(event_summaries: list[Mapping[str, Any]], variant: str) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for item in event_summaries:
        counts.update(item.get("runtime_stats", {}).get(variant, {}).get("candidate_source_counts", {}))
    return dict(sorted(counts.items()))


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--replay-root", default=str(DEFAULT_REPLAY_ROOT))
    parser.add_argument("--batch-manifest", default=str(DEFAULT_BATCH_PATH))
    parser.add_argument("--audit-output", default=str(DEFAULT_AUDIT_PATH))
    parser.add_argument("--event-metrics-output", default=str(DEFAULT_EVENT_METRICS_PATH))
    parser.add_argument("--result-output", default=str(DEFAULT_RESULT_PATH))
    parser.add_argument("--stage-output", default=str(DEFAULT_STAGE_PATH))
    args = parser.parse_args()
    replay_root = resolved_path(args.replay_root)
    batch_path = resolved_path(args.batch_manifest)
    audit_path = resolved_path(args.audit_output)
    event_metrics_path = resolved_path(args.event_metrics_output)
    result_path = resolved_path(args.result_output)
    stage_path = resolved_path(args.stage_output)
    started = now_utc()
    protocol = read_json(PROTOCOL_PATH)
    expected_events = [dict(item) for item in protocol.get("source_event_selection", {}).get("events", [])]
    expected_by_id = {str(item["event_id"]): item for item in expected_events}
    errors: list[str] = []
    batch = read_json(batch_path) if batch_path.is_file() else {}
    batch_ids = [str(item.get("event_id")) for item in batch.get("results", [])]
    if batch.get("status") != "PASS_N72R10_TRUE_CLOSED_LOOP_REPLAY_BATCH":
        errors.append(f"batch_status={batch.get('status')}")
    if len(batch_ids) != len(expected_events) or len(batch_ids) != len(set(batch_ids)) or set(batch_ids) != set(expected_by_id):
        errors.append("batch_event_key_completeness_failed")
    event_summaries: list[dict[str, Any]] = []
    event_results: list[dict[str, Any]] = []
    for event_id in sorted(expected_by_id):
        summary, posthoc_event, event_errors = _audit_event(event_id, replay_root / event_id, expected_by_id[event_id])
        if event_errors:
            errors.extend(f"{event_id}:{item}" for item in event_errors)
        else:
            event_summaries.append(summary)
            event_results.append(posthoc_event)
    audit = {
        "schema_version": "N72R10_TRUE_CLOSED_LOOP_REPLAY_AUDIT_V1",
        "status": "PASS_N72R10_TRUE_CLOSED_LOOP_REPLAY_AUDIT" if not errors else "FAIL_N72R10_TRUE_CLOSED_LOOP_REPLAY_AUDIT",
        "started_at_utc": started,
        "finished_at_utc": now_utc(),
        "protocol": str(PROTOCOL_PATH),
        "protocol_sha256": sha256_file(PROTOCOL_PATH),
        "batch_manifest": str(batch_path),
        "batch_manifest_sha256": sha256_file(batch_path) if batch_path.is_file() else None,
        "expected_event_count": len(expected_events),
        "audited_event_count": len(event_summaries),
        "unique_event_count": len({item["event_id"] for item in event_summaries}),
        "unique_sequence_count": len({item["sequence"] for item in event_summaries}),
        "action_counts": dict(sorted(Counter(item["action_type"] for item in event_summaries).items())),
        "runtime_candidate_source_counts": _sum_source_counts(event_summaries, "TEMPORAL_REQUERY"),
        "future_requery_candidate_rows_in_E2": int(sum(item.get("future_requery_candidate_rows_in_E2", 0) for item in event_summaries)),
        "failed_event_count": len(expected_events) - len(event_summaries),
        "failures": errors,
        "runtime_future_gt_used": False,
        "posthoc_gt_used": not bool(errors),
        "interaction_source": "simulated_from_gt",
        "not_real_human_evidence": True,
        "event_summaries": event_summaries,
    }
    atomic_json(audit_path, audit)
    if errors:
        failure_stage = {
            "schema_version": "N72R10_STAGE_08_STATUS_V1",
            "stage": "08_CPU_REPLAY_AUDIT_AND_AGGREGATION",
            "status": "FAIL_N72R10_REPLAY_AUDIT",
            "audit": str(audit_path),
            "audit_sha256": sha256_file(audit_path),
            "error_count": len(errors),
            "failures": errors,
            "runtime_future_gt_used": False,
        }
        atomic_json(stage_path, failure_stage)
        print(json.dumps(failure_stage, ensure_ascii=False, sort_keys=True))
        return 1
    event_metric_rows: list[dict[str, Any]] = []
    for summary, event in zip(event_summaries, event_results):
        event_metrics: dict[str, Any] = {}
        for old_comparison, alias in COMPARISON_ALIASES.items():
            event_metrics[alias] = {
                str(horizon): _strip_frame_details(event["comparisons"][old_comparison][str(horizon)])
                for horizon in HORIZONS
            }
        event_metric_rows.append({**summary, "metrics": event_metrics})
    atomic_jsonl(event_metrics_path, event_metric_rows)
    aggregate: dict[str, Any] = {}
    for old_comparison, alias in COMPARISON_ALIASES.items():
        aggregate[alias] = {
            str(horizon): legacy._aggregate(event_results, old_comparison, horizon)
            for horizon in HORIZONS
        }
        aggregate[alias]["by_action"] = {
            action: {str(horizon): legacy._aggregate(event_results, old_comparison, horizon, action=action) for horizon in HORIZONS}
            for action in sorted({str(item["action_type"]) for item in event_results})
        }
    runtime_summary = {
        variant: {
            "future_frame_count": int(sum(item["runtime_stats"].get(variant, {}).get("future_frame_count", 0) for item in event_summaries)),
            "model_score_changed_frame_count": int(sum(item["runtime_stats"].get(variant, {}).get("model_score_changed_frame_count", 0) for item in event_summaries)),
            "target_assigned_frame_count": int(sum(item["runtime_stats"].get(variant, {}).get("target_assigned_frame_count", 0) for item in event_summaries)),
            "candidate_source_counts": _sum_source_counts(event_summaries, variant),
        }
        for variant in OLD_VARIANTS
    }
    result = {
        "schema_version": "N72R10_TRUE_CLOSED_LOOP_PAIRED_REPLAY_V1",
        "status": "PASS_N72R10_TRUE_CLOSED_LOOP_REPLAY_AGGREGATED",
        "created_at_utc": now_utc(),
        "protocol": str(PROTOCOL_PATH),
        "protocol_sha256": sha256_file(PROTOCOL_PATH),
        "future_requery_audit": "outputs/N72R10/stage_03_true_future_requery/batch_integrity_audit.json",
        "future_requery_audit_sha256": sha256_file(ROOT / "outputs/N72R10/stage_03_true_future_requery/batch_integrity_audit.json"),
        "runtime_audit": str(audit_path),
        "runtime_audit_sha256": sha256_file(audit_path),
        "event_metrics": str(event_metrics_path),
        "event_metrics_sha256": sha256_file(event_metrics_path),
        "events": len(event_results),
        "independent_sequences": len({str(item["sequence"]) for item in event_results}),
        "action_counts": dict(sorted(Counter(str(item["action_type"]) for item in event_results).items())),
        "variants": VARIANT_ALIASES,
        "variant_semantics": {
            "E0_B0": "B0 candidate stream, CCAM/requery disabled",
            "E1_TEMPORAL_CURRENT_V2": "trained N72R10 source-aware temporal scorer with current target session only",
            "E2_TRUE_CLOSED_LOOP_REQUERY_V2": "same scorer plus sealed true future-frame re-query source on causal uncertainty",
        },
        "runtime_summary": runtime_summary,
        "metrics": aggregate,
        "runtime_future_gt_used": False,
        "posthoc_gt_used": True,
        "interaction_source": "simulated_from_gt",
        "not_real_human_evidence": True,
        "production_authorized": False,
    }
    atomic_json(result_path, result)
    stage = {
        "schema_version": "N72R10_STAGE_08_STATUS_V1",
        "stage": "08_CPU_REPLAY_AUDIT_AND_AGGREGATION",
        "status": "PASS_N72R10_REPLAY_AUDIT_AND_AGGREGATION",
        "audit": str(audit_path),
        "audit_sha256": sha256_file(audit_path),
        "result": str(result_path),
        "result_sha256": sha256_file(result_path),
        "event_metrics": str(event_metrics_path),
        "event_metrics_sha256": sha256_file(event_metrics_path),
        "expected_event_count": len(expected_events),
        "audited_event_count": len(event_results),
        "unique_event_count": len({str(item["event_id"]) for item in event_results}),
        "unique_sequence_count": len({str(item["sequence"]) for item in event_results}),
        "action_counts": dict(sorted(Counter(str(item["action_type"]) for item in event_results).items())),
        "future_requery_candidate_rows_in_E2": int(sum(item.get("future_requery_candidate_rows_in_E2", 0) for item in event_summaries)),
        "duplicate_event_count": 0,
        "missing_event_count": 0,
        "unavailable_event_count": 0,
        "partial_event_count": 0,
        "runtime_future_gt_used": False,
        "interaction_source": "simulated_from_gt",
        "not_real_human_evidence": True,
        "production_authorized": False,
        "next_stage": "09_FUTURE_EFFECT_GATE",
    }
    atomic_json(stage_path, stage)
    print(json.dumps(stage, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
