#!/usr/bin/env python3
"""Finalize the N72R10 CPU-only gate and audit live re-query milestones.

The runtime artifacts are already sealed by the N72R10 batch auditor.  This
script only reads those artifacts, the posthoc event metrics, and the frozen
training status.  It never runs association, writes runtime rows, or uses GT
before a runtime seal.  Dataset GT is used here only for an offline audit of
whether a selected fresh candidate is a target-quality candidate.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Iterable, Mapping


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REPLAY_ROOT = ROOT / "outputs/N72R10/stage_07_replay/attempt_02"
DEFAULT_RESULT = ROOT / "outputs/N72R10/stage_07_replay_attempt_02/ccam_paired_replay_results.json"
DEFAULT_STAGE08 = ROOT / "outputs/N72R10/stage_08_status_attempt_02.json"
DEFAULT_RUNTIME_AUDIT = ROOT / "outputs/N72R10/stage_07_replay_attempt_02/runtime_audit.json"
DEFAULT_FUTURE_AUDIT = ROOT / "outputs/N72R10/stage_03_true_future_requery/batch_integrity_audit.json"
DEFAULT_CORPUS = ROOT / "outputs/N72R10/training/corpus_manifest.json"
DEFAULT_TRAINING = ROOT / "outputs/N72R10/stage_06_training_status.json"
DEFAULT_STAGE3 = ROOT / "outputs/N72R10/stage_03_status.json"
DEFAULT_MILESTONE = ROOT / "outputs/N72R10/true_requery_milestone_audit.json"
DEFAULT_GATE = ROOT / "outputs/N72R10/stage_09_gate.json"
PROTOCOL = ROOT / "outputs/N72R9/protocol.json"
DATA_ROOT = Path("/data1/LWR/vranlee/DATASETS/JDE/dancetrack")
FUTURE_SOURCE = "FUTURE_FRAME_REQUERY"
HORIZONS = (20, 50, 100)


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


def atomic_write(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(dict(payload), ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n"
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


def box_iou(first: Any, second: Any) -> float:
    if not isinstance(first, (list, tuple)) or not isinstance(second, (list, tuple)):
        return 0.0
    if len(first) != 4 or len(second) != 4:
        return 0.0
    x1, y1, x2, y2 = [float(value) for value in first]
    a1, b1, a2, b2 = [float(value) for value in second]
    ix = max(0.0, min(x2, a2) - max(x1, a1))
    iy = max(0.0, min(y2, b2) - max(y1, b1))
    intersection = ix * iy
    area_first = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_second = max(0.0, a2 - a1) * max(0.0, b2 - b1)
    denominator = area_first + area_second - intersection
    return float(intersection / denominator) if denominator > 0.0 else 0.0


class GroundTruthCache:
    def __init__(self, data_root: Path) -> None:
        self.data_root = data_root
        self._cache: dict[tuple[str, int], dict[int, list[float]]] = {}

    def boxes(self, sequence: str, frame: int) -> dict[int, list[float]]:
        key = (str(sequence), int(frame))
        if key not in self._cache:
            path = self.data_root / "train" / str(sequence) / "gt" / "gt.txt"
            values: dict[int, list[float]] = {}
            with path.open(encoding="utf-8") as handle:
                for line in handle:
                    fields = line.strip().split(",")
                    if len(fields) < 6 or int(fields[0]) - 1 != int(frame):
                        continue
                    identity = int(fields[1])
                    x, y, width, height = [float(value) for value in fields[2:6]]
                    values[identity] = [x, y, x + width, y + height]
            self._cache[key] = values
        return self._cache[key]


def _source_map(rows: Iterable[Mapping[str, Any]]) -> dict[str, str]:
    return {str(row.get("candidate_uid")): str(row.get("candidate_source")) for row in rows}


def _candidate_map(rows: Iterable[Mapping[str, Any]]) -> dict[str, Mapping[str, Any]]:
    return {str(row.get("candidate_uid")): row for row in rows}


def _raw_rebind(row: Mapping[str, Any]) -> tuple[bool, bool]:
    switch = row.get("raw_binding_switch")
    if not isinstance(switch, Mapping):
        return False, False
    changed = (
        switch.get("old_raw_sam_id") != switch.get("new_raw_sam_id")
        or switch.get("old_native_scope") != switch.get("new_native_scope")
    )
    return bool(changed), bool(changed and switch.get("public_id_changed") is False)


def _posthoc_wrong_to_correct(posthoc: Mapping[str, Any]) -> list[int]:
    event = posthoc.get("event")
    if not isinstance(event, Mapping):
        return []
    comparisons = event.get("comparisons")
    if not isinstance(comparisons, Mapping):
        return []
    comparison = comparisons.get("TEMPORAL_REQUERY_vs_TEMPORAL_CURRENT")
    if not isinstance(comparison, Mapping):
        return []
    horizon = comparison.get("100")
    if not isinstance(horizon, Mapping):
        return []
    details = horizon.get("frame_details")
    if not isinstance(details, list):
        return []
    return [
        int(item["frame"])
        for item in details
        if isinstance(item, Mapping)
        and item.get("baseline_correct") is False
        and item.get("treatment_correct") is True
    ]


def audit_event(
    event_id: str,
    expected: Mapping[str, Any],
    event_dir: Path,
    gt_cache: GroundTruthCache,
) -> tuple[dict[str, Any], list[str]]:
    errors: list[str] = []
    e2_path = event_dir / "TEMPORAL_REQUERY" / "runtime_frames.jsonl"
    posthoc_path = event_dir / "posthoc.json"
    if not e2_path.is_file() or not posthoc_path.is_file():
        return {"event_id": event_id, "sequence": expected.get("sequence"), "action_type": expected.get("action_type")}, [
            f"{event_id}:missing_runtime_or_posthoc"
        ]
    rows = read_jsonl(e2_path)
    event_frame = int(expected["event_frame"])
    expected_frames = list(range(event_frame, event_frame + 101))
    if [int(row.get("frame", -1)) for row in rows] != expected_frames:
        errors.append(f"{event_id}:frame_axis_invalid")
    posthoc = read_json(posthoc_path)
    wrong_to_correct = _posthoc_wrong_to_correct(posthoc)
    target_gt_id = int(expected["dataset_gt_id"])
    sequence = str(expected["sequence"])
    counters: Counter[str] = Counter()
    action = str(expected["action_type"])
    records: list[dict[str, Any]] = []
    for row in rows[1:]:
        requery = row.get("requery")
        if not isinstance(requery, Mapping):
            errors.append(f"{event_id}:{row.get('frame')}:missing_requery_record")
            continue
        if requery.get("triggered"):
            counters["trigger_count"] += 1
        if requery.get("applied"):
            counters["applied_count"] += 1
        candidate_rows = list(row.get("candidate_rows", []))
        source_by_uid = _source_map(candidate_rows)
        candidate_by_uid = _candidate_map(candidate_rows)
        future_uids = {uid for uid, source in source_by_uid.items() if source == FUTURE_SOURCE}
        counters["fresh_candidate_count"] += len(future_uids)
        assignment = row.get("assignment")
        if not isinstance(assignment, Mapping):
            errors.append(f"{event_id}:{row.get('frame')}:assignment_missing")
            continue
        selected = assignment.get("target_selected_candidate_uid")
        assigned = assignment.get("target_assigned_candidate_uid")
        selected_uid = None if selected is None else str(selected)
        assigned_uid = None if assigned is None else str(assigned)
        selected_future = selected_uid is not None and source_by_uid.get(selected_uid) == FUTURE_SOURCE
        assigned_future = assigned_uid is not None and source_by_uid.get(assigned_uid) == FUTURE_SOURCE
        same_selected_assignment = bool(selected_future and selected_uid == assigned_uid)
        gt_box = gt_cache.boxes(sequence, int(row["frame"])).get(target_gt_id)
        selected_iou = box_iou(candidate_by_uid.get(selected_uid, {}).get("box_xyxy"), gt_box) if selected_uid else 0.0
        assigned_iou = box_iou(candidate_by_uid.get(assigned_uid, {}).get("box_xyxy"), gt_box) if assigned_uid else 0.0
        if selected_future:
            counters["fresh_selected_count"] += 1
            if selected_iou >= 0.50:
                counters["fresh_selected_target_iou50_count"] += 1
            else:
                counters["fresh_selected_wrong_count"] += 1
        if assigned_future:
            counters["fresh_assigned_any_count"] += 1
        if same_selected_assignment:
            counters["fresh_assigned_target_count"] += 1
            if assigned_iou >= 0.50:
                counters["fresh_assigned_target_iou50_count"] += 1
            else:
                counters["fresh_assigned_wrong_count"] += 1
        if selected_future and selected_iou >= 0.50 and assigned_uid != selected_uid:
            counters["fresh_good_solver_refusal_count"] += 1
        raw_changed, raw_stable = _raw_rebind(row)
        if raw_changed:
            counters["raw_rebind_count"] += 1
        if raw_stable:
            counters["public_stable_rebind_count"] += 1
        solver = assignment.get("solver")
        solver_target_ok = False
        if isinstance(solver, Mapping):
            for solver_row in solver.get("assignment_rows", []):
                if str(solver_row.get("candidate_uid")) == assigned_uid:
                    solver_target_ok = solver_row.get("public_id") == row.get("target_public_id")
                    break
        if same_selected_assignment and solver_target_ok:
            records.append({
                "frame": int(row["frame"]),
                "selected_uid": selected_uid,
                "selected_iou": float(selected_iou),
                "solver_target_ok": True,
            })
        if row.get("runtime_future_gt_used") is not False:
            errors.append(f"{event_id}:{row.get('frame')}:runtime_future_gt_not_false")
    changed_rows = []
    for row in rows[1:]:
        changed, stable = _raw_rebind(row)
        if changed and stable:
            changed_rows.append(int(row["frame"]))
    complete_paths: list[dict[str, Any]] = []
    for record in records:
        rebound_after = [frame for frame in changed_rows if frame >= int(record["frame"])]
        if not rebound_after:
            continue
        rebound_frame = rebound_after[0]
        corrected_after = [frame for frame in wrong_to_correct if frame >= rebound_frame]
        if corrected_after:
            complete_paths.append({
                "selected_frame": int(record["frame"]),
                "rebind_frame": int(rebound_frame),
                "posthoc_wrong_to_correct_frame": int(corrected_after[0]),
                "selected_candidate_uid": record["selected_uid"],
                "selected_target_iou": record["selected_iou"],
            })
    counters["posthoc_wrong_to_correct_count"] = len(wrong_to_correct)
    counters["complete_milestone_count"] = len(complete_paths)
    counters["complete_milestone_event_count"] = int(bool(complete_paths))
    summary = {
        "event_id": event_id,
        "sequence": sequence,
        "action_type": action,
        "event_frame": event_frame,
        "counts": dict(sorted(counters.items())),
        "complete_milestones": complete_paths,
        "wrong_to_correct_frames": wrong_to_correct,
        "runtime_future_gt_used": False,
        "posthoc_gt_used": True,
        "runtime_frames": str(e2_path),
        "runtime_frames_sha256": sha256_file(e2_path),
        "posthoc": str(posthoc_path),
        "posthoc_sha256": sha256_file(posthoc_path),
    }
    return summary, errors


def _metric_summary(metrics: Mapping[str, Any], comparison: str) -> dict[str, Any]:
    source = metrics.get(comparison, {})
    result: dict[str, Any] = {}
    for horizon in HORIZONS:
        value = source.get(str(horizon), {}) if isinstance(source, Mapping) else {}
        ci = value.get("sequence_cluster_bootstrap_95ci", {}) if isinstance(value, Mapping) else {}
        result[str(horizon)] = {
            "identity_error_reduction": value.get("identity_error_reduction"),
            "ci_lower": ci.get("lower"),
            "ci_upper": ci.get("upper"),
            "assignment_change_count": value.get("assignment_change_count"),
            "true_correct_crossing_count": value.get("true_correct_crossing_count"),
            "true_incorrect_crossing_count": value.get("true_incorrect_crossing_count"),
            "protected_regression_count": value.get("protected_regression_count"),
            "protected_compared": value.get("protected_compared"),
            "wrong_reassociation_frames": value.get("wrong_reassociation_frames"),
            "missing_rate": value.get("missing_rate"),
            "delta_iou": value.get("delta_iou"),
        }
    return result


def _all_ci_lower_positive(summary: Mapping[str, Any]) -> bool:
    values = [summary.get(str(horizon), {}).get("ci_lower") for horizon in HORIZONS]
    return all(isinstance(value, (int, float)) and float(value) > 0.0 for value in values)


def _all_crossings_correct(summary: Mapping[str, Any]) -> bool:
    for horizon in HORIZONS:
        value = summary.get(str(horizon), {})
        correct = value.get("true_correct_crossing_count")
        incorrect = value.get("true_incorrect_crossing_count")
        if not isinstance(correct, (int, float)) or not isinstance(incorrect, (int, float)) or correct <= incorrect:
            return False
    return True


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--replay-root", default=str(DEFAULT_REPLAY_ROOT))
    parser.add_argument("--result", default=str(DEFAULT_RESULT))
    parser.add_argument("--stage08", default=str(DEFAULT_STAGE08))
    parser.add_argument("--runtime-audit", default=str(DEFAULT_RUNTIME_AUDIT))
    parser.add_argument("--future-audit", default=str(DEFAULT_FUTURE_AUDIT))
    parser.add_argument("--corpus", default=str(DEFAULT_CORPUS))
    parser.add_argument("--training", default=str(DEFAULT_TRAINING))
    parser.add_argument("--stage03", default=str(DEFAULT_STAGE3))
    parser.add_argument("--milestone-output", default=str(DEFAULT_MILESTONE))
    parser.add_argument("--gate-output", default=str(DEFAULT_GATE))
    args = parser.parse_args()

    replay_root = resolved_path(args.replay_root)
    result_path = resolved_path(args.result)
    stage08_path = resolved_path(args.stage08)
    runtime_audit_path = resolved_path(args.runtime_audit)
    future_audit_path = resolved_path(args.future_audit)
    corpus_path = resolved_path(args.corpus)
    training_path = resolved_path(args.training)
    stage03_path = resolved_path(args.stage03)
    milestone_path = resolved_path(args.milestone_output)
    gate_path = resolved_path(args.gate_output)

    protocol = read_json(PROTOCOL)
    expected_events = {str(item["event_id"]): item for item in protocol.get("source_event_selection", {}).get("events", [])}
    stage03 = read_json(stage03_path)
    stage08 = read_json(stage08_path)
    result = read_json(result_path)
    runtime_audit = read_json(runtime_audit_path)
    future_audit = read_json(future_audit_path)
    corpus = read_json(corpus_path)
    training = read_json(training_path)
    errors: list[str] = []
    if stage03.get("status") != "PASS_TRUE_FUTURE_REQUERY_BATCH":
        errors.append(f"stage03_status={stage03.get('status')}")
    if stage08.get("status") != "PASS_N72R10_REPLAY_AUDIT_AND_AGGREGATION":
        errors.append(f"stage08_status={stage08.get('status')}")
    if stage08.get("expected_event_count") != 32 or stage08.get("audited_event_count") != 32 or stage08.get("unique_event_count") != 32:
        errors.append("stage08_event_completeness_failed")
    for key in ("duplicate_event_count", "missing_event_count", "unavailable_event_count", "partial_event_count"):
        if int(stage08.get(key, -1)) != 0:
            errors.append(f"stage08_{key}={stage08.get(key)}")
    if stage08.get("result_sha256") != sha256_file(result_path):
        errors.append("stage08_result_hash_mismatch")
    if stage08.get("audit_sha256") != sha256_file(runtime_audit_path):
        errors.append("stage08_runtime_audit_hash_mismatch")
    if future_audit.get("status") != "PASS_N72R10_TRUE_FUTURE_REQUERY_BATCH_AUDIT":
        errors.append(f"future_audit_status={future_audit.get('status')}")
    if result.get("runtime_future_gt_used") is not False:
        errors.append("aggregate_runtime_future_gt_not_false")

    gt_cache = GroundTruthCache(DATA_ROOT)
    event_summaries: list[dict[str, Any]] = []
    for event_id in sorted(expected_events):
        summary, event_errors = audit_event(event_id, expected_events[event_id], replay_root / event_id, gt_cache)
        event_summaries.append(summary)
        errors.extend(event_errors)
    if len(event_summaries) != len(expected_events):
        errors.append("event_summary_count_mismatch")

    global_counts: Counter[str] = Counter()
    by_action: dict[str, Counter[str]] = defaultdict(Counter)
    complete_event_ids: list[str] = []
    for summary in event_summaries:
        counts = Counter(summary.get("counts", {}))
        global_counts.update(counts)
        by_action[str(summary.get("action_type"))].update(counts)
        if counts.get("complete_milestone_count", 0) > 0:
            complete_event_ids.append(str(summary["event_id"]))
    milestone = {
        "schema_version": "N72R10_TRUE_REQUERY_MILESTONE_AUDIT_V1",
        "status": "PASS_N72R10_MILESTONE_AUDIT" if not errors else "FAIL_N72R10_MILESTONE_AUDIT_INTEGRITY",
        "created_at_utc": now_utc(),
        "protocol": str(PROTOCOL),
        "protocol_sha256": sha256_file(PROTOCOL),
        "replay_root": str(replay_root),
        "runtime_audit": str(runtime_audit_path),
        "runtime_audit_sha256": sha256_file(runtime_audit_path),
        "future_audit": str(future_audit_path),
        "future_audit_sha256": sha256_file(future_audit_path),
        "expected_event_count": len(expected_events),
        "audited_event_count": len(event_summaries),
        "unique_event_count": len({str(item.get("event_id")) for item in event_summaries}),
        "unique_sequence_count": len({str(item.get("sequence")) for item in event_summaries}),
        "action_counts": dict(sorted(Counter(str(item.get("action_type")) for item in event_summaries).items())),
        "global_counts": dict(sorted(global_counts.items())),
        "by_action": {key: dict(sorted(value.items())) for key, value in sorted(by_action.items())},
        "complete_milestone_event_ids": complete_event_ids,
        "complete_milestone_count": int(global_counts.get("complete_milestone_count", 0)),
        "runtime_future_gt_used": False,
        "interaction_source": "simulated_from_gt",
        "not_real_human_evidence": True,
        "integrity_errors": sorted(set(errors)),
        "event_summaries": event_summaries,
    }
    atomic_write(milestone_path, milestone)

    metrics = result.get("metrics", {})
    e1 = _metric_summary(metrics, "E1_vs_E0")
    e2 = _metric_summary(metrics, "E2_vs_E1")
    e1_protected = {horizon: e1[str(horizon)].get("protected_regression_count") for horizon in HORIZONS}
    e2_protected = {horizon: e2[str(horizon)].get("protected_regression_count") for horizon in HORIZONS}
    protected_increment_not_worse = all(
        isinstance(e1_protected[horizon], (int, float))
        and isinstance(e2_protected[horizon], (int, float))
        and e2_protected[horizon] <= e1_protected[horizon]
        for horizon in HORIZONS
    )
    protected_zero = all(value == 0 for value in e1_protected.values())
    train_split = corpus.get("splits", {}).get("train", {})
    val_split = corpus.get("splits", {}).get("validation", {})
    validation_eval = training.get("validation_evaluation", {})
    validation_source = validation_eval.get("source_diagnostics", {})
    validation_none_accuracy = validation_eval.get("none_accuracy")
    training_checks = {
        "train_examples_at_least_30000": int(train_split.get("example_count", 0)) >= 30000,
        "validation_examples_at_least_5000": int(val_split.get("example_count", 0)) >= 5000,
        "future_requery_train_rows_present": int(train_split.get("future_rows_total", 0)) > 0,
        "future_requery_train_positive_labels_present": int(train_split.get("future_rows_selected_as_label", 0)) > 0,
        "future_requery_validation_rows_present": int(val_split.get("future_rows_total", 0)) > 0,
        "future_requery_validation_positive_labels_present": int(val_split.get("future_rows_selected_as_label", 0)) > 0,
        "validation_none_accuracy_finite_positive": isinstance(validation_none_accuracy, (int, float)) and float(validation_none_accuracy) > 0.0,
    }
    gate_checks = {
        "stage03_true_future_batch_integrity": not errors,
        "stage08_replay_integrity": not errors,
        "module1_e1_ci_lower_positive_h20_h50_h100": _all_ci_lower_positive(e1),
        "module2_e2_minus_e1_ci_lower_positive_h20_h50_h100": _all_ci_lower_positive(e2),
        "module2_correct_crossings_exceed_incorrect_all_horizons": _all_crossings_correct(e2),
        "module2_protected_regression_not_worse_than_e1": protected_increment_not_worse,
        "e1_protected_regression_zero": protected_zero,
        "complete_live_requery_milestone_present": int(global_counts.get("complete_milestone_count", 0)) >= 1,
        "training_future_positive_validation_coverage": training_checks["future_requery_validation_positive_labels_present"],
        "training_distribution_target_reached": training_checks["train_examples_at_least_30000"] and training_checks["validation_examples_at_least_5000"],
    }
    reasons: list[str] = []
    if not gate_checks["e1_protected_regression_zero"]:
        reasons.append("E1_PROTECTED_REGRESSION_NONZERO")
    if not gate_checks["training_future_positive_validation_coverage"]:
        reasons.append("VALIDATION_FUTURE_REQUERY_POSITIVE_LABEL_COVERAGE_ZERO")
    if not gate_checks["training_distribution_target_reached"]:
        reasons.append("TRAINING_DISTRIBUTION_BELOW_PREREGISTERED_TARGET")
    if not gate_checks["complete_live_requery_milestone_present"]:
        reasons.append("NO_COMPLETE_LIVE_REQUERY_MILESTONE")
    if not gate_checks["module2_e2_minus_e1_ci_lower_positive_h20_h50_h100"]:
        reasons.append("E2_MINUS_E1_SEQUENCE_CLUSTER_CI_NOT_STRICTLY_POSITIVE")
    if errors:
        reasons.append("RUNTIME_OR_INPUT_INTEGRITY_ERROR")
    gate = {
        "schema_version": "N72R10_FINAL_GATE_V1",
        "status": "PASS_N72R10_TRUE_CLOSED_LOOP_DEVELOPMENT" if all(gate_checks.values()) else "FAIL_N72R10_DEVELOPMENT_GATE",
        "created_at_utc": now_utc(),
        "research_gate": "PASS_N72R10_TRUE_CLOSED_LOOP_DEVELOPMENT" if all(gate_checks.values()) else "FAIL_FUTURE_EFFECT_OR_DEVELOPMENT_READINESS",
        "terminal_scientific_success": bool(all(gate_checks.values())),
        "production_authorized": False,
        "calibration_authorized": False,
        "selector_authorized": False,
        "decoder_lora_authorized": False,
        "interaction_source": "simulated_from_gt",
        "not_real_human_evidence": True,
        "runtime_future_gt_used": False,
        "historical_n72r9_gate_preserved": "FAIL_FUTURE_REQUERY_EFFECT",
        "protocol": str(PROTOCOL),
        "protocol_sha256": sha256_file(PROTOCOL),
        "stage03": str(stage03_path),
        "stage03_sha256": sha256_file(stage03_path),
        "stage08": str(stage08_path),
        "stage08_sha256": sha256_file(stage08_path),
        "runtime_audit": str(runtime_audit_path),
        "runtime_audit_sha256": sha256_file(runtime_audit_path),
        "result": str(result_path),
        "result_sha256": sha256_file(result_path),
        "milestone_audit": str(milestone_path),
        "milestone_audit_sha256": sha256_file(milestone_path),
        "event_count": len(event_summaries),
        "sequence_count": len({str(item.get("sequence")) for item in event_summaries}),
        "action_counts": dict(sorted(Counter(str(item.get("action_type")) for item in event_summaries).items())),
        "milestone_counts": dict(sorted(global_counts.items())),
        "milestone_by_action": {key: dict(sorted(value.items())) for key, value in sorted(by_action.items())},
        "metrics": {"E1_vs_E0": e1, "E2_vs_E1": e2},
        "protected_regression": {
            "E1_vs_E0": e1_protected,
            "E2_vs_E1": e2_protected,
            "e2_increment_not_worse_than_e1": protected_increment_not_worse,
            "e1_zero": protected_zero,
        },
        "training": {
            "corpus_manifest": str(corpus_path),
            "corpus_manifest_sha256": sha256_file(corpus_path),
            "training_status": str(training_path),
            "training_status_sha256": sha256_file(training_path),
            "train": train_split,
            "validation": val_split,
            "validation_none_accuracy": validation_none_accuracy,
            "validation_source_diagnostics": validation_source,
            "checks": training_checks,
        },
        "gate_checks": gate_checks,
        "reason_codes": reasons,
        "integrity_errors": sorted(set(errors)),
        "next_action": (
            "Do not promote or train downstream modules. Expand the lawful future-requery training/validation event pool, add validation positive FUTURE_FRAME_REQUERY labels, and isolate the target-edge solver refusal before any new production interface."
            if not all(gate_checks.values())
            else "Freeze the successful configuration and seek independent confirmation before downstream authorization."
        ),
    }
    atomic_write(gate_path, gate)
    print(json.dumps({
        "status": gate["status"],
        "research_gate": gate["research_gate"],
        "milestone_count": gate["milestone_counts"].get("complete_milestone_count", 0),
        "selected_future": gate["milestone_counts"].get("fresh_selected_count", 0),
        "assigned_target": gate["milestone_counts"].get("fresh_assigned_target_count", 0),
        "integrity_errors": len(errors),
        "gate_output": str(gate_path),
        "milestone_output": str(milestone_path),
    }, ensure_ascii=False, sort_keys=True))
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
