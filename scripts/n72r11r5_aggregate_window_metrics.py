#!/usr/bin/env python
"""Aggregate a *complete* N72R11R5 interaction-window TrackEval run.

This module intentionally has no metric implementation.  It only consumes the
official TrackEval detailed CSV values materialized by
``n72r11r5_run_window_trackeval.py``.  In particular, it refuses to create an
aggregate when export or any horizon is incomplete.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sam3_intermot.evaluation.window_trackeval import (  # noqa: E402
    FROZEN_METRICS,
    HORIZONS,
    LOGICAL_VARIANTS,
    WindowTrackEvalError,
    sha256_file,
    write_json_atomic,
)


METRICS = ("HOTA", "AssA", "DetA", "IDF1", "MOTA", "IDSW")
HIGHER_IS_BETTER = {"HOTA", "AssA", "DetA", "IDF1", "MOTA"}
COMPARISONS = (("E1A_V3", "E0_BASELINE_B0"), ("E1B_PCTIS", "E0_BASELINE_B0"), ("E1B_PCTIS", "E1A_V3"))
BOOTSTRAP_SEED = 7211
BOOTSTRAP_REPETITIONS = 2000


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise WindowTrackEvalError(f"missing JSON artifact: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise WindowTrackEvalError(f"invalid JSON artifact {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise WindowTrackEvalError(f"expected JSON object: {path}")
    return value


def _finite(value: Any, label: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise WindowTrackEvalError(f"{label} is not numeric: {value!r}") from exc
    if not math.isfinite(result):
        raise WindowTrackEvalError(f"{label} is not finite")
    return result


def _blocked(root: Path, reason: str, **details: Any) -> int:
    payload = {
        "schema_version": "N72R11R5_WINDOW_AGGREGATION_BLOCKED_V1",
        "status": "BLOCKED_INCOMPLETE_WINDOW_TRACKEVAL",
        "evaluation_scope": "INTERACTION_WINDOW_DIAGNOSTIC",
        "official_dancetrack_benchmark_score": False,
        "reason": reason,
        "aggregate_files_written": False,
        **details,
    }
    write_json_atomic(root / "aggregation_blocked.json", payload)
    print(f"AGGREGATION_BLOCKED: {reason}")
    return 1


def _load_complete_inputs(root: Path, project_root: Path) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]], dict[tuple[str, int], dict[str, Any]]]:
    failure_path = root / "export_failure_invalid_assigned_boxes.json"
    if failure_path.is_file():
        failure = _read_json(failure_path)
        if failure.get("status") != "PASS":
            raise WindowTrackEvalError(
                f"export failure artifact blocks aggregation: {failure.get('status')}"
            )

    export = _read_json(root / "export_manifest.json")
    if export.get("status") != "PASS_EXPORT":
        raise WindowTrackEvalError(f"export status is not PASS_EXPORT: {export.get('status')!r}")
    if export.get("source_events") != 32 or export.get("independent_sequence_count") != 18:
        raise WindowTrackEvalError("export does not contain the frozen 32-event/18-sequence matrix")
    if export.get("record_count") != 288:
        raise WindowTrackEvalError(f"export record_count is not 288: {export.get('record_count')!r}")
    if export.get("runtime_future_gt_used") is not False:
        raise WindowTrackEvalError("export runtime_future_gt_used is not false")
    integrity = export.get("integrity")
    if not isinstance(integrity, dict) or integrity.get("duplicate_keys") or integrity.get("missing_keys"):
        raise WindowTrackEvalError("export key integrity is not complete")

    run = _read_json(root / "trackeval_run_manifest.json")
    if run.get("status") != "PASS_TRACKEVAL":
        raise WindowTrackEvalError(f"TrackEval status is not PASS_TRACKEVAL: {run.get('status')!r}")
    if run.get("expected_record_count") != 288 or run.get("record_count") != 288:
        raise WindowTrackEvalError("TrackEval run is not the complete 288-record matrix")
    if run.get("horizons") != list(HORIZONS) or run.get("logical_variants") != list(LOGICAL_VARIANTS):
        raise WindowTrackEvalError("TrackEval run does not cover the frozen horizon/variant matrix")
    if run.get("duplicate_keys"):
        raise WindowTrackEvalError("TrackEval run contains duplicate keys")
    horizon_results = run.get("horizon_results")
    if not isinstance(horizon_results, list) or len(horizon_results) != len(HORIZONS):
        raise WindowTrackEvalError("TrackEval run lacks all three horizon results")
    for result in horizon_results:
        if result.get("status") != "PASS" or result.get("record_count") != 32 * len(LOGICAL_VARIANTS):
            raise WindowTrackEvalError(f"incomplete TrackEval horizon result: {result}")

    event_windows = export.get("event_windows")
    if not isinstance(event_windows, list) or len(event_windows) != 32 * len(HORIZONS):
        raise WindowTrackEvalError("export event_windows is not 96 complete windows")
    windows: dict[tuple[str, int], dict[str, Any]] = {}
    for window in event_windows:
        if not isinstance(window, dict):
            raise WindowTrackEvalError("event_windows contains a non-object")
        key = (window.get("event_id"), window.get("horizon"))
        if key in windows or not isinstance(key[0], str) or key[1] not in HORIZONS:
            raise WindowTrackEvalError(f"invalid or duplicate export window key: {key}")
        windows[key] = window

    rows_path = root / "per_event_metrics.jsonl"
    if not rows_path.is_file():
        raise WindowTrackEvalError("complete per_event_metrics.jsonl is missing")
    rows: list[dict[str, Any]] = []
    with rows_path.open(encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, start=1):
            if not raw.strip():
                continue
            try:
                row = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise WindowTrackEvalError(f"invalid per-event JSONL line {line_number}") from exc
            if not isinstance(row, dict):
                raise WindowTrackEvalError(f"per-event line {line_number} is not an object")
            rows.append(row)
    expected_keys = {
        (event_id, logical, horizon)
        for event_id, horizon in windows
        for logical in LOGICAL_VARIANTS
    }
    observed_keys = set()
    for row in rows:
        key = (row.get("event_id"), row.get("logical_variant"), row.get("horizon"))
        if key in observed_keys:
            raise WindowTrackEvalError(f"duplicate per-event key: {key}")
        observed_keys.add(key)
        if key not in expected_keys:
            raise WindowTrackEvalError(f"unexpected per-event key: {key}")
        metrics = row.get("metrics")
        if not isinstance(metrics, dict):
            raise WindowTrackEvalError(f"missing metrics for {key}")
        for metric in METRICS:
            metrics[metric] = _finite(metrics.get(metric), f"{key}/{metric}")
        window = windows[(key[0], key[2])]
        for field in ("pseudo_sequence", "original_sequence", "action_type", "event_frame"):
            if row.get(field) != window.get(field):
                raise WindowTrackEvalError(f"window metadata mismatch for {key}/{field}")
    if len(rows) != 288 or observed_keys != expected_keys:
        raise WindowTrackEvalError(
            f"per-event matrix incomplete: rows={len(rows)}, missing={len(expected_keys - observed_keys)}"
        )

    # Require the official pooled rows to exist and be finite before producing
    # any descriptive aggregate.
    for result in horizon_results:
        pooled = result.get("pooled")
        if not isinstance(pooled, dict) or set(pooled) != set(LOGICAL_VARIANTS):
            raise WindowTrackEvalError("a horizon lacks pooled official TrackEval rows")
        for logical in LOGICAL_VARIANTS:
            for metric in METRICS:
                _finite(pooled[logical].get(metric), f"pooled/{result.get('horizon')}/{logical}/{metric}")
    return export, run, rows, windows


def _mean(values: Iterable[float]) -> float:
    values = list(values)
    if not values:
        raise WindowTrackEvalError("cannot average an empty value set")
    return float(sum(values) / len(values))


def _bootstrap(sequence_values: Mapping[str, list[float]]) -> dict[str, Any]:
    if len(sequence_values) == 0:
        raise WindowTrackEvalError("no independent sequence clusters for bootstrap")
    sequence_means = {sequence: _mean(values) for sequence, values in sorted(sequence_values.items())}
    values = list(sequence_means.values())
    # The explicit NumPy generator and percentile method are recorded so the
    # paired inference is reproducible and cannot be confused with frame-level
    # or event-level resampling.
    import numpy as np

    array = np.asarray(values, dtype=float)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    indices = rng.integers(0, len(array), size=(BOOTSTRAP_REPETITIONS, len(array)))
    samples = array[indices].mean(axis=1)
    return {
        "unit": "independent_sequence",
        "clusters": len(values),
        "within_cluster_aggregation": "mean_event_value",
        "repetitions": BOOTSTRAP_REPETITIONS,
        "seed": BOOTSTRAP_SEED,
        "mean": float(array.mean()),
        "lower": float(np.quantile(samples, 0.025)),
        "upper": float(np.quantile(samples, 0.975)),
        "sequence_means": sequence_means,
    }


def _pooled_metrics(run: Mapping[str, Any]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for result in sorted(run["horizon_results"], key=lambda item: item["horizon"]):
        horizon = int(result["horizon"])
        output[str(horizon)] = {
            logical: {metric: float(result["pooled"][logical][metric]) for metric in METRICS}
            for logical in LOGICAL_VARIANTS
        }
    return output


def _descriptive_by_action(rows: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[tuple[int, str, str], list[float]] = defaultdict(list)
    counts: dict[tuple[int, str, str], int] = defaultdict(int)
    for row in rows:
        key_base = (int(row["horizon"]), row["logical_variant"], row["action_type"])
        counts[key_base] += 1
        for metric in METRICS:
            grouped[(*key_base, metric)].append(float(row["metrics"][metric]))
    output: dict[str, Any] = {}
    for (horizon, logical, action), count in sorted(counts.items()):
        output.setdefault(str(horizon), {}).setdefault(logical, {})[action] = {
            "event_count": count,
            "per_pseudo_sequence_mean": {
                metric: _mean(grouped[(horizon, logical, action, metric)]) for metric in METRICS
            },
            "aggregation_note": "descriptive arithmetic mean over complete pseudo sequences; not official pooled TrackEval",
        }
    return output


def _paired_deltas(rows: list[dict[str, Any]]) -> dict[str, Any]:
    indexed = {
        (row["event_id"], row["horizon"], row["logical_variant"]): row
        for row in rows
    }
    result: dict[str, Any] = {}
    for treatment, baseline in COMPARISONS:
        comparison_name = f"{treatment}_vs_{baseline}"
        result[comparison_name] = {}
        for horizon in HORIZONS:
            values_by_metric: dict[str, dict[str, list[float]]] = {
                metric: defaultdict(list) for metric in METRICS
            }
            event_deltas: list[dict[str, Any]] = []
            event_ids = sorted({row["event_id"] for row in rows if row["horizon"] == horizon})
            for event_id in event_ids:
                treatment_row = indexed[(event_id, horizon, treatment)]
                baseline_row = indexed[(event_id, horizon, baseline)]
                sequence = treatment_row["original_sequence"]
                delta_row: dict[str, Any] = {
                    "event_id": event_id,
                    "original_sequence": sequence,
                    "action_type": treatment_row["action_type"],
                    "horizon": horizon,
                    "delta": {},
                }
                for metric in METRICS:
                    treatment_value = float(treatment_row["metrics"][metric])
                    baseline_value = float(baseline_row["metrics"][metric])
                    delta = (
                        baseline_value - treatment_value
                        if metric == "IDSW"
                        else treatment_value - baseline_value
                    )
                    values_by_metric[metric][sequence].append(delta)
                    delta_row["delta"][metric] = delta
                event_deltas.append(delta_row)
            metrics_result: dict[str, Any] = {}
            for metric in METRICS:
                metrics_result[metric] = {
                    "direction": "baseline_minus_treatment" if metric == "IDSW" else "treatment_minus_baseline",
                    "event_count": len(event_deltas),
                    "bootstrap_95ci": _bootstrap(values_by_metric[metric]),
                    "event_delta_sum": sum(item["delta"][metric] for item in event_deltas),
                    "positive_event_delta_count": sum(item["delta"][metric] > 0 for item in event_deltas),
                    "negative_event_delta_count": sum(item["delta"][metric] < 0 for item in event_deltas),
                    "zero_event_delta_count": sum(item["delta"][metric] == 0 for item in event_deltas),
                }
            result[comparison_name][str(horizon)] = {
                "treatment": treatment,
                "baseline": baseline,
                "metrics": metrics_result,
                "event_deltas": event_deltas,
            }
    return result


def _identity_interpretation(project_root: Path) -> dict[str, Any]:
    outputs: dict[str, Any] = {
        "schema_version": "N72R11R5_IDENTITY_METRIC_INTERPRETATION_V1",
        "source_metric_type": "N72R11R4 frozen identity-error posthoc metric, not TrackEval",
        "relative_reduction_definition": "(baseline_identity_error - treatment_identity_error) / baseline_identity_error",
        "absolute_reduction_pp_definition": "(baseline_identity_error - treatment_identity_error) * 100",
        "sources": {},
        "by_variant": {},
    }
    source_values: dict[str, dict[str, Any]] = {}
    for logical, relative in FROZEN_METRICS.items():
        path = project_root / relative
        metrics = _read_json(path)
        if metrics.get("complete") is not True:
            raise WindowTrackEvalError(f"frozen identity source is incomplete: {path}")
        outputs["sources"][logical] = {"path": str(path), "sha256": sha256_file(path)}
        source_values[logical] = metrics
    for logical, metrics in source_values.items():
        variant_output: dict[str, Any] = {}
        for horizon in HORIZONS:
            values = metrics.get("by_horizon", {}).get(str(horizon))
            if not isinstance(values, dict):
                raise WindowTrackEvalError(f"frozen identity source lacks H{horizon}: {logical}")
            baseline = _finite(values.get("baseline_future_identity_error"), f"{logical}/H{horizon}/baseline")
            treatment = _finite(values.get("treatment_future_identity_error"), f"{logical}/H{horizon}/treatment")
            reduction = baseline - treatment
            variant_output[str(horizon)] = {
                "baseline_identity_error_percent": baseline * 100.0,
                "treatment_identity_error_percent": treatment * 100.0,
                "absolute_reduction_pp": reduction * 100.0,
                "relative_reduction": None if baseline == 0 else reduction / baseline,
                "baseline_identity_error_frames": values.get("baseline_identity_error_frames"),
                "treatment_identity_error_frames": values.get("treatment_identity_error_frames"),
                "evaluated_frames": values.get("evaluated_frames"),
            }
        outputs["by_variant"][logical] = variant_output
    return outputs


def aggregate(root: Path, project_root: Path) -> dict[str, Any]:
    export, run, rows, windows = _load_complete_inputs(root, project_root)
    pooled = _pooled_metrics(run)
    paired = _paired_deltas(rows)
    identity = _identity_interpretation(project_root)

    # This is a diagnostic sanity summary, not a reimplementation of HOTA or
    # any other TrackEval metric.  A zero detection delta is intentionally
    # reported as undefined rather than divided through.
    association_detection: dict[str, Any] = {}
    for comparison_name, horizons in paired.items():
        association_detection[comparison_name] = {}
        for horizon, payload in horizons.items():
            assa = payload["metrics"]["AssA"]["bootstrap_95ci"]["mean"]
            deta = payload["metrics"]["DetA"]["bootstrap_95ci"]["mean"]
            association_detection[comparison_name][horizon] = {
                "mean_delta_AssA": assa,
                "mean_delta_DetA": deta,
                "AssA_to_DetA_delta_ratio": None if abs(deta) <= 1e-12 else assa / deta,
                "ratio_note": "descriptive ratio only; no causal decomposition",
            }

    pooled_path = root / "pooled_metrics.json"
    paired_path = root / "paired_metric_deltas.json"
    identity_path = root / "identity_metric_interpretation.json"
    write_json_atomic(
        pooled_path,
        {
            "schema_version": "N72R11R5_POOLED_TRACKEVAL_V1",
            "status": "PASS_COMPLETE_INPUT",
            "evaluation_scope": "INTERACTION_WINDOW_DIAGNOSTIC",
            "official_dancetrack_benchmark_score": False,
            "source_trackeval_run_manifest": str(root / "trackeval_run_manifest.json"),
            "source_trackeval_run_manifest_sha256": sha256_file(root / "trackeval_run_manifest.json"),
            "by_horizon": pooled,
            "by_action": _descriptive_by_action(rows),
            "pooled_definition": "official TrackEval COMBINED row per horizon and logical variant",
        },
    )
    write_json_atomic(
        paired_path,
        {
            "schema_version": "N72R11R5_PAIRED_METRIC_DELTAS_V1",
            "status": "PASS_COMPLETE_INPUT",
            "evaluation_scope": "INTERACTION_WINDOW_DIAGNOSTIC",
            "official_dancetrack_benchmark_score": False,
            "bootstrap": {
                "unit": "independent_sequence",
                "seed": BOOTSTRAP_SEED,
                "repetitions": BOOTSTRAP_REPETITIONS,
                "within_cluster_aggregation": "mean_event_value",
            },
            "comparisons": paired,
            "association_detection_sanity": association_detection,
        },
    )
    write_json_atomic(identity_path, identity)
    aggregation = {
        "schema_version": "N72R11R5_WINDOW_AGGREGATION_MANIFEST_V1",
        "status": "PASS_COMPLETE_INPUT",
        "evaluation_scope": "INTERACTION_WINDOW_DIAGNOSTIC",
        "official_dancetrack_benchmark_score": False,
        "event_count": 32,
        "independent_sequence_count": 18,
        "horizons": list(HORIZONS),
        "logical_variants": list(LOGICAL_VARIANTS),
        "per_event_record_count": len(rows),
        "runtime_future_gt_used": False,
        "e2_evaluated": False,
        "e2_status": "NOT_MEASURED_INCOMPLETE_FORMAL_REPLAY",
        "source_export_manifest_sha256": sha256_file(root / "export_manifest.json"),
        "source_trackeval_run_manifest_sha256": sha256_file(root / "trackeval_run_manifest.json"),
        "outputs": {
            "pooled_metrics": {"path": str(pooled_path), "sha256": sha256_file(pooled_path)},
            "paired_metric_deltas": {"path": str(paired_path), "sha256": sha256_file(paired_path)},
            "identity_metric_interpretation": {"path": str(identity_path), "sha256": sha256_file(identity_path)},
        },
    }
    write_json_atomic(root / "aggregation_manifest.json", aggregation)
    return aggregation


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("outputs/N72R11R5"))
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    args = parser.parse_args()
    root = args.root.resolve()
    project_root = args.project_root.resolve()
    try:
        result = aggregate(root, project_root)
    except (OSError, KeyError, TypeError, ValueError, WindowTrackEvalError) as exc:
        return _blocked(root, str(exc), project_root=str(project_root))
    print(f"AGGREGATION_PASS records={result['per_event_record_count']} output={root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
