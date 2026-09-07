#!/usr/bin/env python3
"""Aggregate the sealed N72R11R2 development replay without changing evidence.

The combined replay audit is the only source of event selection.  This script
reads the posthoc files referenced by its sealed PASS records, keeps the
incomplete event explicit, and computes weighted window summaries plus the
same independent-sequence bootstrap used by the N72R11 worker.  It never
opens a new replay worker and it never promotes a partial replay to a gate
PASS.
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
import sys
import tempfile
from typing import Any, Iterable, Mapping

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_COMBINED = ROOT / "outputs/N72R11R2/formal_replay_resource_censored_combined_manifest.json"
DEFAULT_OUTPUT = ROOT / "outputs/N72R11R2/formal_replay_resource_censored_metrics.json"
HORIZONS = (20, 50, 100)
COMPARISONS = ("E1_vs_E0", "E2_vs_E0", "E2_vs_E1")
BOOTSTRAP_SEED = 7211
BOOTSTRAP_REPETITIONS = 2000

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sam3_intermot.evaluation.interaction_effect_metrics import (  # noqa: E402
    sequence_cluster_bootstrap,
)


COUNT_KEYS = (
    "evaluated_frames",
    "window_frame_count",
    "target_gt_visible_frames",
    "target_gt_absent_frames",
    "assignment_change_count",
    "global_common_assignment_change_count",
    "target_assignment_change_count",
    "true_correct_crossing_count",
    "true_incorrect_crossing_count",
    "directional_improvement_count",
    "directional_regression_count",
    "neutral_change_count",
    "id_switch_count",
    "raw_switch_count",
    "posthoc_correct_switch_count",
    "posthoc_wrong_switch_count",
    "posthoc_unassessable_switch_count",
    "recorrection_opportunity_count",
    "candidate_present_frames",
    "target_missing_frames",
    "baseline_correct_frames",
    "baseline_identity_error_frames",
    "treatment_correct_frames",
    "treatment_identity_error_frames",
    "protected_compared",
    "protected_regression_count",
    "protected_improvement_count",
)
FLOAT_SUM_KEYS = (
    "identity_error_reduction_sum",
    "delta_iou_sum",
    "baseline_iou_sum",
    "treatment_iou_sum",
)


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


def resolve_path(value: Any) -> Path:
    path = Path(str(value))
    return path if path.is_absolute() else ROOT / path


def finite_number(value: Any, *, field: str, path: Path) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise ValueError(f"non-finite {field} in {path}: {value!r}")
    return float(value)


def integer_value(metric: Mapping[str, Any], key: str, *, path: Path) -> int:
    value = metric.get(key, 0)
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise ValueError(f"invalid count {key} in {path}: {value!r}")
    rounded = int(value)
    if float(value) != float(rounded) or rounded < 0:
        raise ValueError(f"non-integral/negative count {key} in {path}: {value!r}")
    return rounded


def validate_posthoc(selected: Mapping[str, Any]) -> tuple[dict[str, Any], Path]:
    posthoc_value = selected.get("posthoc")
    if not posthoc_value:
        raise ValueError(f"selected record has no posthoc path: {selected.get('event_id')}")
    path = resolve_path(posthoc_value)
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = read_json(path)
    if payload.get("status") != "PASS_N72R11_POSTHOC_EVENT":
        raise ValueError(f"posthoc status is not PASS at {path}: {payload.get('status')!r}")
    if payload.get("runtime_future_gt_used") is not False:
        raise ValueError(f"posthoc runtime GT contract failed at {path}")
    if payload.get("posthoc_gt_used") is not True:
        raise ValueError(f"posthoc GT provenance missing at {path}")
    event = payload.get("event")
    if not isinstance(event, Mapping):
        raise ValueError(f"posthoc event object missing at {path}")
    for key in ("event_id", "sequence", "action_type"):
        if str(event.get(key)) != str(selected.get(key)):
            raise ValueError(f"{key} mismatch between manifest and {path}")
    if event.get("runtime_future_gt_used") is not False or event.get("posthoc_gt_used") is not True:
        raise ValueError(f"event-level GT provenance failed at {path}")
    comparisons = event.get("comparisons")
    if not isinstance(comparisons, Mapping):
        raise ValueError(f"comparison object missing at {path}")
    for comparison in COMPARISONS:
        horizons = comparisons.get(comparison)
        if not isinstance(horizons, Mapping):
            raise ValueError(f"missing comparison {comparison} at {path}")
        for horizon in HORIZONS:
            metric = horizons.get(str(horizon))
            if not isinstance(metric, Mapping):
                raise ValueError(f"missing {comparison}/H{horizon} at {path}")
            if integer_value(metric, "evaluated_frames", path=path) <= 0:
                raise ValueError(f"empty evaluated window at {path}: {comparison}/H{horizon}")
            for key in FLOAT_SUM_KEYS:
                finite_number(metric.get(key, 0.0), field=key, path=path)
            for key in COUNT_KEYS:
                integer_value(metric, key, path=path)
    return dict(payload), path


def add_metric(destination: dict[str, Any], metric: Mapping[str, Any]) -> None:
    for key in COUNT_KEYS:
        destination[key] = destination.get(key, 0) + integer_value(metric, key, path=Path("<posthoc>"))
    for key in FLOAT_SUM_KEYS:
        destination[key] = destination.get(key, 0.0) + finite_number(
            metric.get(key, 0.0), field=key, path=Path("<posthoc>")
        )


def finalize_metric(
    total: Mapping[str, Any],
    *,
    event_count: int,
    sequences: Iterable[str],
    event_values_by_sequence: Mapping[str, list[float]],
) -> dict[str, Any]:
    evaluated = int(total.get("evaluated_frames", 0))
    denominator = max(1, evaluated)
    protected = int(total.get("protected_compared", 0))
    result = dict(total)
    result.update(
        {
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
                event_values_by_sequence,
                seed=BOOTSTRAP_SEED,
                repetitions=BOOTSTRAP_REPETITIONS,
            ),
        }
    )
    return result


def aggregate_items(items: list[dict[str, Any]]) -> dict[str, Any]:
    total: dict[str, Any] = {key: 0 for key in COUNT_KEYS}
    total.update({key: 0.0 for key in FLOAT_SUM_KEYS})
    values_by_sequence: dict[str, list[float]] = defaultdict(list)
    sequences: list[str] = []
    for item in items:
        metric = item["metric"]
        add_metric(total, metric)
        sequence = str(item["sequence"])
        sequences.append(sequence)
        values_by_sequence[sequence].append(float(metric["identity_error_reduction"]))
    return finalize_metric(
        total,
        event_count=len(items),
        sequences=sequences,
        event_values_by_sequence=values_by_sequence,
    )


def compact_event_metric(item: Mapping[str, Any]) -> dict[str, Any]:
    metric = item["metric"]
    return {
        "event_id": str(item["event_id"]),
        "sequence": str(item["sequence"]),
        "action_type": str(item["action_type"]),
        "selected_attempt": item.get("selected_attempt"),
        "posthoc": str(item["posthoc"]),
        "horizon": int(item["horizon"]),
        "comparison": str(item["comparison"]),
        "evaluated_frames": int(metric["evaluated_frames"]),
        "baseline_future_identity_error": float(metric["baseline_identity_error"]),
        "treatment_future_identity_error": float(metric["treatment_identity_error"]),
        "identity_error_reduction": float(metric["identity_error_reduction"]),
        "delta_iou": float(metric["delta_iou"]),
        "assignment_change_count": int(metric["assignment_change_count"]),
        "true_correct_crossing_count": int(metric["true_correct_crossing_count"]),
        "true_incorrect_crossing_count": int(metric["true_incorrect_crossing_count"]),
        "directional_improvement_count": int(metric["directional_improvement_count"]),
        "directional_regression_count": int(metric["directional_regression_count"]),
        "protected_regression_count": int(metric["protected_regression_count"]),
        "missing_rate": float(metric["missing_rate"]),
        "id_switch_count": int(metric["id_switch_count"]),
        "recorrection_opportunity_count": int(metric["recorrection_opportunity_count"]),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--combined", type=Path, default=DEFAULT_COMBINED)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    combined = args.combined if args.combined.is_absolute() else ROOT / args.combined
    output = args.output if args.output.is_absolute() else ROOT / args.output
    manifest = read_json(combined)
    selected = manifest.get("selected_records")
    if not isinstance(selected, list):
        raise RuntimeError("combined audit has no selected_records list")
    if manifest.get("status") != "INCOMPLETE_FORMAL_DEVELOPMENT_REPLAY":
        raise RuntimeError(f"unexpected combined audit status: {manifest.get('status')!r}")
    items: list[dict[str, Any]] = []
    seen: set[str] = set()
    for record in selected:
        if not isinstance(record, Mapping) or record.get("status") != "PASS":
            raise RuntimeError("selected_records contains a non-PASS record")
        event_id = str(record.get("event_id"))
        if event_id in seen:
            raise RuntimeError(f"duplicate selected event: {event_id}")
        seen.add(event_id)
        posthoc, path = validate_posthoc(record)
        comparisons = posthoc["event"]["comparisons"]
        for comparison in COMPARISONS:
            for horizon in HORIZONS:
                metric = dict(comparisons[comparison][str(horizon)])
                items.append(
                    {
                        "event_id": event_id,
                        "sequence": str(record["sequence"]),
                        "action_type": str(record["action_type"]),
                        "selected_attempt": record.get("selected_attempt"),
                        "posthoc": path,
                        "comparison": comparison,
                        "horizon": horizon,
                        "metric": metric,
                    }
                )
    required = int(manifest.get("event_count_required", 0))
    missing = [str(value) for value in manifest.get("missing_event_ids", [])]
    action_counts = Counter(str(item["action_type"]) for item in selected)
    sequence_counts = Counter(str(item["sequence"]) for item in selected)
    aggregate: dict[str, dict[str, Any]] = {comparison: {} for comparison in COMPARISONS}
    by_action: dict[str, dict[str, dict[str, Any]]] = {}
    by_sequence: dict[str, dict[str, dict[str, Any]]] = {}
    for comparison in COMPARISONS:
        for horizon in HORIZONS:
            chosen = [item for item in items if item["comparison"] == comparison and item["horizon"] == horizon]
            aggregate[comparison][str(horizon)] = aggregate_items(chosen)
    for action in sorted(action_counts):
        by_action[action] = {comparison: {} for comparison in COMPARISONS}
        for comparison in COMPARISONS:
            for horizon in HORIZONS:
                chosen = [
                    item
                    for item in items
                    if item["action_type"] == action and item["comparison"] == comparison and item["horizon"] == horizon
                ]
                by_action[action][comparison][str(horizon)] = aggregate_items(chosen)
    for sequence in sorted(sequence_counts):
        by_sequence[sequence] = {comparison: {} for comparison in COMPARISONS}
        for comparison in COMPARISONS:
            for horizon in HORIZONS:
                chosen = [
                    item
                    for item in items
                    if item["sequence"] == sequence and item["comparison"] == comparison and item["horizon"] == horizon
                ]
                by_sequence[sequence][comparison][str(horizon)] = aggregate_items(chosen)
    selected_attempt_counts = Counter(str(item.get("selected_attempt")) for item in selected)
    result = {
        "schema_version": "N72R11R2_FORMAL_REPLAY_METRICS_V1",
        "status": "PASS_METRICS_FOR_31_SEALED_EVENTS_INCOMPLETE_FORMAL_REPLAY",
        "created_at_utc": now_utc(),
        "source_combined_manifest": str(combined),
        "source_combined_manifest_sha256": sha256_file(combined),
        "source_status": manifest.get("status"),
        "protocol": manifest.get("protocol"),
        "protocol_sha256": manifest.get("protocol_sha256"),
        "resource_censored_development": manifest.get("resource_censored_development") is True,
        "interaction_source": manifest.get("interaction_source"),
        "not_real_human_evidence": manifest.get("not_real_human_evidence") is True,
        "runtime_future_gt_used": manifest.get("runtime_future_gt_used"),
        "event_completeness": {
            "required_events": required,
            "selected_sealed_pass_events": len(selected),
            "missing_or_failed_events": len(missing),
            "missing_or_failed_event_ids": missing,
            "duplicate_selected_event_ids": [],
            "complete_required_event_gate": len(selected) == required and not missing,
        },
        "selected_attempt_counts": dict(sorted(selected_attempt_counts.items())),
        "action_counts": dict(sorted(action_counts.items())),
        "sequence_counts": dict(sorted(sequence_counts.items())),
        "bootstrap": {
            "seed": BOOTSTRAP_SEED,
            "repetitions": BOOTSTRAP_REPETITIONS,
            "cluster_unit": "independent_sequence",
            "within_cluster_aggregation": "mean_event_identity_error_reduction",
            "implementation": "sam3_intermot.evaluation.interaction_effect_metrics.sequence_cluster_bootstrap",
        },
        "aggregate": aggregate,
        "by_action": by_action,
        "by_sequence": by_sequence,
        "event_metrics": [compact_event_metric(item) for item in items],
        "future_effect_gate": {
            "status": "NOT_EVALUATED_INCOMPLETE_FORMAL_REPLAY",
            "reason": "The fixed formal replay has one sealed child failure; development metrics are descriptive only.",
            "strict_ci_lower_gt_zero": False,
            "protected_identity_regression_required_zero": False,
            "production_authorized": False,
            "v3_bridge_production_authorized": False,
        },
    }
    atomic_json(output, result)
    print(
        json.dumps(
            {
                "status": result["status"],
                "output": str(output),
                "selected_events": len(selected),
                "required_events": required,
                "missing_or_failed": len(missing),
                "action_counts": dict(sorted(action_counts.items())),
                "sequence_count": len(sequence_counts),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
