#!/usr/bin/env python3
"""Aggregate the four-variant N72R12 learned-gate TrackEval matrix."""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone
import argparse
import json
import math
from pathlib import Path
import sys
import tempfile
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sam3_intermot.evaluation.interaction_effect_metrics import sequence_cluster_bootstrap  # noqa: E402

HORIZONS = (20, 50, 100)
EXPECTED_EVENTS = 32
VARIANTS = ("E0_BASELINE_B0", "E1B_PCTIS", "E1C_PCTIS_SAFE", "E1D_PCTIS_LEARNED_SAFE")
TRACK_METRICS = ("HOTA", "AssA", "DetA", "IDF1", "MOTA", "IDSW")
COMPARISONS = (
    ("E1B_vs_E0", "E1B_PCTIS", "E0_BASELINE_B0"),
    ("E1C_vs_E0", "E1C_PCTIS_SAFE", "E0_BASELINE_B0"),
    ("E1D_vs_E0", "E1D_PCTIS_LEARNED_SAFE", "E0_BASELINE_B0"),
    ("E1D_vs_E1B", "E1D_PCTIS_LEARNED_SAFE", "E1B_PCTIS"),
    ("E1D_vs_E1C", "E1D_PCTIS_LEARNED_SAFE", "E1C_PCTIS_SAFE"),
)


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object: {path}")
    return value


def _finite(value: Any, label: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} is non-finite")
    return result


def _bootstrap(values: dict[str, list[float]]) -> dict[str, Any]:
    return sequence_cluster_bootstrap(values, seed=7211, repetitions=2000)


def _atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    try:
        with open(fd, "w", encoding="utf-8", closefd=True) as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
        Path(temporary).replace(path)
    finally:
        if Path(temporary).exists():
            Path(temporary).unlink()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-manifest", type=Path, default=ROOT / "outputs/N72R12/trackeval_learned/trackeval_run_manifest.json")
    parser.add_argument("--causal", type=Path, default=ROOT / "outputs/N72R12/causal/learned_causal_metrics.json")
    parser.add_argument("--training", type=Path, default=ROOT / "outputs/N72R12/gate_training/training_history.json")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/N72R12/trackeval_learned/window_metrics.json")
    args = parser.parse_args()
    run_path = args.run_manifest.resolve()
    causal_path = args.causal.resolve()
    run = _read(run_path)
    causal = _read(causal_path)
    training = _read(args.training.resolve())
    if run.get("status") != "PASS_TRACKEVAL_N72R12_LEARNED" or run.get("record_count") != 384:
        raise RuntimeError("learned TrackEval run is incomplete")
    if causal.get("status") != "PASS_N72R12_LEARNED_CAUSAL_AGGREGATION" or causal.get("event_count") != 32:
        raise RuntimeError("learned causal metrics are incomplete")
    pooled: dict[str, dict[str, dict[str, float]]] = {}
    for horizon_result in run["horizon_results"]:
        horizon = str(horizon_result["horizon"])
        pooled[horizon] = {}
        for variant in VARIANTS:
            pooled[horizon][variant] = {
                metric: _finite(horizon_result["pooled"][variant][metric], f"pooled/{horizon}/{variant}/{metric}")
                for metric in TRACK_METRICS
            }
    rows: list[dict[str, Any]] = []
    with (run_path.parent / "per_event_metrics.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    keys = [(row.get("event_id"), row.get("logical_variant"), int(row.get("horizon", -1))) for row in rows]
    if len(rows) != 384 or len(set(keys)) != 384:
        raise RuntimeError("learned TrackEval per-event matrix is not exactly 384 unique rows")
    comparison_metrics: dict[str, Any] = {}
    for name, treatment, baseline in COMPARISONS:
        comparison_metrics[name] = {}
        for horizon in HORIZONS:
            indexed = {(row["event_id"], int(row["horizon"]), row["logical_variant"]): row for row in rows if int(row["horizon"]) == horizon}
            events = sorted({row["event_id"] for row in rows})
            by_sequence: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
            by_action: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
            deltas: dict[str, list[float]] = {metric: [] for metric in TRACK_METRICS}
            event_deltas: list[dict[str, Any]] = []
            for event_id in events:
                treatment_row = indexed[(event_id, horizon, treatment)]
                baseline_row = indexed[(event_id, horizon, baseline)]
                event_delta: dict[str, Any] = {
                    "event_id": event_id,
                    "sequence": treatment_row["original_sequence"],
                    "action_type": treatment_row["action_type"],
                    "metrics": {},
                }
                for metric in TRACK_METRICS:
                    delta = _finite(treatment_row["metrics"][metric], metric) - _finite(baseline_row["metrics"][metric], metric)
                    deltas[metric].append(delta)
                    by_sequence[str(treatment_row["original_sequence"])][metric].append(delta)
                    by_action[str(treatment_row["action_type"])][metric].append(delta)
                    event_delta["metrics"][metric] = delta
                event_deltas.append(event_delta)
            comparison_metrics[name][str(horizon)] = {
                "treatment": treatment,
                "baseline": baseline,
                "pooled_treatment": pooled[str(horizon)][treatment],
                "pooled_baseline": pooled[str(horizon)][baseline],
                "pooled_delta": {metric: pooled[str(horizon)][treatment][metric] - pooled[str(horizon)][baseline][metric] for metric in TRACK_METRICS},
                "event_mean_delta": {metric: sum(values) / len(values) for metric, values in deltas.items()},
                "sequence_cluster_bootstrap_95ci": {metric: _bootstrap(by_sequence[metric]) for metric in TRACK_METRICS},
                "by_action_event_mean_delta": {action: {metric: sum(values) / len(values) for metric, values in metrics.items()} for action, metrics in sorted(by_action.items())},
                "event_deltas": event_deltas,
            }
    causal_e1d = causal["comparisons"]["E1D_vs_E0"]
    trackeval_h100 = comparison_metrics["E1D_vs_E0"]["100"]["pooled_delta"]
    causal_h20 = causal_e1d["20"]
    learned_gate = training.get("readiness", {})
    strict_gate = {
        "trackeval_h100_hota_improves": bool(trackeval_h100["HOTA"] > 0.0),
        "trackeval_h100_assa_improves": bool(trackeval_h100["AssA"] > 0.0),
        "trackeval_h100_idf1_improves": bool(trackeval_h100["IDF1"] > 0.0),
        "trackeval_h100_idsw_decreases": bool(trackeval_h100["IDSW"] < 0.0),
        "causal_h20_identity_error_reduction_positive": bool(float(causal_h20["identity_error_reduction"]) > 0.0),
        "causal_h20_correct_crossings_gt_incorrect": bool(int(causal_h20["true_correct_crossing_count"]) > int(causal_h20["true_incorrect_crossing_count"])),
        "protected_regression_zero_h20_h50_h100": all(int(causal_e1d[str(horizon)]["protected_regression_count"]) == 0 for horizon in HORIZONS),
    }
    strict_gate["pass"] = all(strict_gate.values())
    payload = {
        "schema_version": "N72R12_LEARNED_WINDOW_METRICS_V1",
        "status": "PASS_N72R12_LEARNED_DETERMINISTIC_MATRIX",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "event_count": EXPECTED_EVENTS,
        "independent_sequence_count": 18,
        "record_count": 384,
        "horizons": list(HORIZONS),
        "variants": list(VARIANTS),
        "pooled": pooled,
        "comparisons": comparison_metrics,
        "causal_gate_input": {"E1D_vs_E0": causal_e1d},
        "learned_gate_readiness": learned_gate,
        "strict_learned_gate": strict_gate,
        "trackeval_commit": run.get("trackeval_commit"),
        "official_dancetrack_benchmark_score": False,
        "runtime_future_gt_used": False,
        "gt_used_only_for_posthoc_evaluation": True,
        "interaction_source": "simulated_from_gt",
        "not_real_human_evidence": True,
        "run_manifest": str(run_path),
        "causal_metrics": str(causal_path),
        "training_history": str(args.training.resolve()),
        "production_authorized": False,
    }
    _atomic(args.output.resolve(), payload)
    print(json.dumps({"status": payload["status"], "strict_gate_pass": strict_gate["pass"], "output": str(args.output.resolve())}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
