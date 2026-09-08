#!/usr/bin/env python3
"""Aggregate N72R11R3 component-isolated replay artifacts.

The event children already sealed runtime rows and posthoc metrics.  This
script only reads those artifacts, checks provenance/completeness, and applies
the same independent-sequence bootstrap implementation used by N72R11R2.
It never selects an event, checkpoint, or variant based on a future result.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any, Iterable, Mapping

import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sam3_intermot.evaluation.interaction_effect_metrics import sequence_cluster_bootstrap  # noqa: E402


HORIZONS = (20, 50, 100)
COMPARISONS = ("E1_vs_E0", "E2_vs_E1", "E3_vs_E2", "E3_vs_E0")
BOOTSTRAP_SEED = 7211
BOOTSTRAP_REPETITIONS = 2000
COUNT_KEYS = (
    "evaluated_frames", "window_frame_count", "target_gt_visible_frames", "target_gt_absent_frames",
    "assignment_change_count", "global_common_assignment_change_count", "target_assignment_change_count",
    "true_correct_crossing_count", "true_incorrect_crossing_count", "directional_improvement_count",
    "directional_regression_count", "neutral_change_count", "id_switch_count", "raw_switch_count",
    "posthoc_correct_switch_count", "posthoc_wrong_switch_count", "posthoc_unassessable_switch_count",
    "recorrection_opportunity_count", "candidate_present_frames", "target_missing_frames",
    "baseline_correct_frames", "baseline_identity_error_frames", "treatment_correct_frames",
    "treatment_identity_error_frames", "protected_compared", "protected_regression_count",
    "protected_improvement_count",
)
FLOAT_SUM_KEYS = ("identity_error_reduction_sum", "delta_iou_sum", "baseline_iou_sum", "treatment_iou_sum")


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
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


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object: {path}")
    return value


def resolve(value: Any) -> Path:
    path = Path(str(value))
    return path if path.is_absolute() else ROOT / path


def finite(value: Any, label: str, path: Path) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise ValueError(f"non-finite {label} in {path}: {value!r}")
    return float(value)


def count(value: Any, label: str, path: Path) -> int:
    number = finite(value, label, path)
    result = int(number)
    if number != result or result < 0:
        raise ValueError(f"invalid count {label} in {path}: {value!r}")
    return result


def validate_done(record: Mapping[str, Any]) -> tuple[dict[str, Any], Path]:
    done_path = resolve(record.get("done"))
    if not done_path.is_file():
        raise FileNotFoundError(done_path)
    done = read_json(done_path)
    if done.get("status") != "PASS_N72R11_RUNTIME_AND_POSTHOC_EVENT":
        raise ValueError(f"event done is not sealed PASS: {done_path}")
    if done.get("runtime_future_gt_used") is not False:
        raise ValueError(f"{done_path} has runtime_future_gt_used != false")
    # Older sealed event manifests keep runtime_gt_read in the runtime seal
    # rather than duplicating it at done.json top level.  Validate that
    # authoritative seal explicitly instead of treating an omitted legacy
    # projection as a runtime GT read.
    seal_path = resolve(done.get("runtime_event_sealed"))
    if not seal_path.is_file():
        raise FileNotFoundError(seal_path)
    seal = read_json(seal_path)
    if seal.get("runtime_future_gt_used") is not False or seal.get("runtime_gt_read") is not False:
        raise ValueError(f"runtime seal has a GT flag != false: {seal_path}")
    posthoc_path = resolve(done.get("posthoc"))
    if not posthoc_path.is_file():
        raise FileNotFoundError(posthoc_path)
    posthoc = read_json(posthoc_path)
    if posthoc.get("status") != "PASS_N72R11_POSTHOC_EVENT":
        raise ValueError(f"posthoc is not sealed PASS: {posthoc_path}")
    if posthoc.get("runtime_future_gt_used") is not False or posthoc.get("posthoc_gt_used") is not True:
        raise ValueError(f"posthoc provenance failed: {posthoc_path}")
    event = posthoc.get("event")
    if not isinstance(event, Mapping):
        raise ValueError(f"posthoc event object missing: {posthoc_path}")
    if str(event.get("event_id")) != str(record.get("event_id")) or str(event.get("sequence")) != str(record.get("sequence")):
        raise ValueError(f"manifest/event identity mismatch: {posthoc_path}")
    comparisons = event.get("comparisons")
    if not isinstance(comparisons, Mapping):
        raise ValueError(f"comparisons missing: {posthoc_path}")
    for comparison, horizons in comparisons.items():
        if comparison not in COMPARISONS or not isinstance(horizons, Mapping):
            raise ValueError(f"unknown/malformed comparison {comparison}: {posthoc_path}")
        for horizon in HORIZONS:
            metric = horizons.get(str(horizon))
            if not isinstance(metric, Mapping):
                raise ValueError(f"missing {comparison}/H{horizon}: {posthoc_path}")
            if count(metric.get("evaluated_frames", 0), "evaluated_frames", posthoc_path) <= 0:
                raise ValueError(f"empty metric window: {posthoc_path}")
            for key in COUNT_KEYS:
                count(metric.get(key, 0), key, posthoc_path)
            for key in FLOAT_SUM_KEYS:
                finite(metric.get(key, 0.0), key, posthoc_path)
    return posthoc, posthoc_path


def add_metric(destination: dict[str, Any], metric: Mapping[str, Any], path: Path) -> None:
    for key in COUNT_KEYS:
        destination[key] = destination.get(key, 0) + count(metric.get(key, 0), key, path)
    for key in FLOAT_SUM_KEYS:
        destination[key] = destination.get(key, 0.0) + finite(metric.get(key, 0.0), key, path)


def finalize(total: Mapping[str, Any], *, event_count: int, sequences: Iterable[str], values_by_sequence: Mapping[str, list[float]]) -> dict[str, Any]:
    evaluated = int(total.get("evaluated_frames", 0))
    denominator = max(evaluated, 1)
    protected = int(total.get("protected_compared", 0))
    result = dict(total)
    result.update({
        "event_count": int(event_count),
        "independent_sequence_count": len(set(str(value) for value in sequences)),
        "baseline_future_identity_error": float(total.get("baseline_identity_error_frames", 0) / denominator),
        "treatment_future_identity_error": float(total.get("treatment_identity_error_frames", 0) / denominator),
        "identity_error_reduction": float(total.get("identity_error_reduction_sum", 0.0) / denominator),
        "baseline_mean_iou": float(total.get("baseline_iou_sum", 0.0) / denominator),
        "treatment_mean_iou": float(total.get("treatment_iou_sum", 0.0) / denominator),
        "delta_iou": float(total.get("delta_iou_sum", 0.0) / denominator),
        "assignment_change_rate": float(total.get("assignment_change_count", 0) / denominator),
        "target_assignment_change_rate": float(total.get("target_assignment_change_count", 0) / denominator),
        "id_switch_rate": float(total.get("id_switch_count", 0) / denominator),
        "raw_switch_rate": float(total.get("raw_switch_count", 0) / denominator),
        "recorrection_rate": float(total.get("recorrection_opportunity_count", 0) / denominator),
        "candidate_recall": float(total.get("candidate_present_frames", 0) / denominator),
        "missing_rate": float(total.get("target_missing_frames", 0) / denominator),
        "protected_regression_rate": None if protected == 0 else float(total.get("protected_regression_count", 0) / protected),
        "protected_improvement_rate": None if protected == 0 else float(total.get("protected_improvement_count", 0) / protected),
        "sequence_cluster_bootstrap_95ci": sequence_cluster_bootstrap(
            values_by_sequence, seed=BOOTSTRAP_SEED, repetitions=BOOTSTRAP_REPETITIONS
        ) if values_by_sequence else None,
    })
    return result


def aggregate(items: list[dict[str, Any]]) -> dict[str, Any]:
    total: dict[str, Any] = {key: 0 for key in COUNT_KEYS}
    total.update({key: 0.0 for key in FLOAT_SUM_KEYS})
    values: dict[str, list[float]] = defaultdict(list)
    sequences: list[str] = []
    for item in items:
        add_metric(total, item["metric"], Path(str(item["posthoc"])))
        sequence = str(item["sequence"])
        sequences.append(sequence)
        values[sequence].append(float(item["metric"]["identity_error_reduction"]))
    return finalize(total, event_count=len(items), sequences=sequences, values_by_sequence=values)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest_path = args.manifest if args.manifest.is_absolute() else ROOT / args.manifest
    output_path = args.output if args.output.is_absolute() else ROOT / args.output
    manifest = read_json(manifest_path)
    records = manifest.get("records")
    if not isinstance(records, list):
        raise RuntimeError("component manifest has no records")
    items: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    seen: set[str] = set()
    for record in records:
        if not isinstance(record, Mapping):
            raise RuntimeError("component manifest contains non-object record")
        event_id = str(record.get("event_id"))
        if event_id in seen:
            raise RuntimeError(f"duplicate event record: {event_id}")
        seen.add(event_id)
        if record.get("status") != "PASS":
            failures.append({"event_id": event_id, "status": record.get("status"), "returncode": record.get("returncode"), "log": record.get("log")})
            continue
        try:
            posthoc, posthoc_path = validate_done(record)
            comparisons = posthoc["event"]["comparisons"]
            for comparison, horizons in comparisons.items():
                for horizon in HORIZONS:
                    metric = dict(horizons[str(horizon)])
                    items.append({
                        "event_id": event_id,
                        "sequence": str(record["sequence"]),
                        "action_type": str(record["action_type"]),
                        "comparison": str(comparison),
                        "horizon": int(horizon),
                        "posthoc": str(posthoc_path),
                        "metric": metric,
                    })
        except Exception as exc:
            failures.append({"event_id": event_id, "status": "FAIL_AUDIT", "error_type": type(exc).__name__, "error": str(exc), "done": record.get("done")})
    comparison_names = sorted({str(item["comparison"]) for item in items})
    aggregate_by_comparison: dict[str, dict[str, Any]] = {}
    by_action: dict[str, dict[str, dict[str, Any]]] = {}
    by_sequence: dict[str, dict[str, dict[str, Any]]] = {}
    action_names = sorted({str(record.get("action_type")) for record in records})
    sequence_names = sorted({str(record.get("sequence")) for record in records})
    for comparison in comparison_names:
        aggregate_by_comparison[comparison] = {}
        for horizon in HORIZONS:
            aggregate_by_comparison[comparison][str(horizon)] = aggregate([item for item in items if item["comparison"] == comparison and item["horizon"] == horizon])
    for action in action_names:
        by_action[action] = {}
        for comparison in comparison_names:
            by_action[action][comparison] = {}
            for horizon in HORIZONS:
                by_action[action][comparison][str(horizon)] = aggregate([
                    item for item in items if item["action_type"] == action and item["comparison"] == comparison and item["horizon"] == horizon
                ])
    for sequence in sequence_names:
        by_sequence[sequence] = {}
        for comparison in comparison_names:
            by_sequence[sequence][comparison] = {}
            for horizon in HORIZONS:
                by_sequence[sequence][comparison][str(horizon)] = aggregate([
                    item for item in items if item["sequence"] == sequence and item["comparison"] == comparison and item["horizon"] == horizon
                ])
    gate: dict[str, Any] = {}
    for comparison in comparison_names:
        metric = aggregate_by_comparison[comparison].get("20", {})
        ci = metric.get("sequence_cluster_bootstrap_95ci") or {}
        lower = ci.get("lower")
        gate[comparison] = {
            "h20_lower_ci": lower,
            "strict_lower_ci_gt_zero": bool(isinstance(lower, (int, float)) and float(lower) > 0.0),
            "protected_regression_count": int(metric.get("protected_regression_count", 0)),
            "protected_regression_zero": int(metric.get("protected_regression_count", 0)) == 0,
            "event_count": int(metric.get("event_count", 0)),
        }
    complete = not failures and len(records) == int(manifest.get("event_count", len(records)))
    result = {
        "schema_version": "N72R11R3_COMPONENT_METRICS_V1",
        "status": "PASS_COMPONENT_METRICS" if complete else "PARTIAL_COMPONENT_METRICS_WITH_FAILURES",
        "created_at_utc": now_utc(),
        "source_manifest": str(manifest_path),
        "source_manifest_sha256": sha256_file(manifest_path),
        "protocol": manifest.get("protocol"),
        "protocol_sha256": manifest.get("protocol_sha256"),
        "model_checkpoint": manifest.get("model_checkpoint"),
        "model_checkpoint_sha256": manifest.get("model_checkpoint_sha256"),
        "bridge_checkpoint": manifest.get("bridge_checkpoint"),
        "bridge_checkpoint_sha256": manifest.get("bridge_checkpoint_sha256"),
        "variants": manifest.get("variants"),
        "event_completeness": {
            "required": int(manifest.get("event_count", len(records))),
            "manifest_records": len(records),
            "audited_pass_events": len(records) - len(failures),
            "failed_or_invalid_events": len(failures),
            "duplicate_event_ids": len(records) - len(seen),
            "failures": failures,
        },
        "runtime_future_gt_used": False,
        "interaction_source": "simulated_from_gt",
        "not_real_human_evidence": True,
        "resource_censored_development": manifest.get("resource_censored_development") is True,
        "bootstrap": {"seed": BOOTSTRAP_SEED, "repetitions": BOOTSTRAP_REPETITIONS, "cluster": "independent_sequence"},
        "comparisons": aggregate_by_comparison,
        "by_action": by_action,
        "by_sequence": by_sequence,
        "gate": gate,
    }
    atomic_json(output_path, result)
    print(json.dumps({"status": result["status"], "output": str(output_path), "failures": len(failures), "comparisons": comparison_names}, sort_keys=True))
    return 0 if complete else 1


if __name__ == "__main__":
    raise SystemExit(main())
