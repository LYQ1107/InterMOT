#!/usr/bin/env python3
"""Aggregate the complete N72R12 deterministic CSI replay post-hoc.

All GT access in this script happens after sealed runtime artifacts exist.  It
reuses the historical N72R9 ``_score_pair`` implementation and the frozen
sequence-cluster bootstrap; it does not implement a new metric or alter a
runtime assignment.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import sys
import tempfile
from typing import Any, Mapping

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sam3_intermot.evaluation.interaction_effect_metrics import sequence_cluster_bootstrap  # noqa: E402
from scripts import n72r9_temporal_replay as legacy  # noqa: E402

HORIZONS = (20, 50, 100)
COMPARISONS = ("E1B_vs_E0", "E1C_vs_E0", "E1C_vs_E1B")
COUNT_FIELDS = (
    "evaluated_frames", "target_gt_visible_frames", "target_gt_absent_frames", "baseline_correct_frames",
    "treatment_correct_frames", "baseline_identity_error_frames", "treatment_identity_error_frames",
    "target_missing_frames", "wrong_reassociation_frames", "candidate_present_frames", "assignment_change_count",
    "target_assignment_change_count", "global_common_assignment_change_count", "true_correct_crossing_count",
    "true_incorrect_crossing_count", "directional_improvement_count", "directional_regression_count",
    "neutral_change_count", "id_switch_count", "recorrection_opportunity_count", "raw_switch_count",
    "protected_compared", "protected_regression_count", "protected_improvement_count",
)
SUM_FIELDS = ("baseline_iou_sum", "treatment_iou_sum", "delta_iou_sum", "identity_error_reduction_sum")


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object: {path}")
    return value


def _jsonl(path: Path) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not all(isinstance(row, dict) for row in rows):
        raise TypeError(f"non-object row: {path}")
    return rows


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, value: Any) -> None:
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


def _finite(value: Any, label: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} is non-finite")
    return result


def _metric_aggregate(items: list[dict[str, Any]], horizon: int) -> dict[str, Any]:
    if not items:
        raise RuntimeError(f"empty metric group at H{horizon}")
    totals: dict[str, float] = {field: 0.0 for field in (*COUNT_FIELDS, *SUM_FIELDS)}
    sequence_values: dict[str, list[float]] = defaultdict(list)
    for item in items:
        metric = item["metric"]
        for field in COUNT_FIELDS:
            value = int(metric.get(field, 0))
            if value < 0:
                raise ValueError(f"negative metric count {field}")
            totals[field] += value
        for field in SUM_FIELDS:
            totals[field] += _finite(metric.get(field, 0.0), field)
        sequence_values[str(item["sequence"])].append(_finite(metric.get("identity_error_reduction", 0.0), "identity_error_reduction"))
    evaluated = int(totals["evaluated_frames"])
    denominator = max(evaluated, 1)
    protected = int(totals["protected_compared"])
    result: dict[str, Any] = {field: int(value) if field in COUNT_FIELDS else float(value) for field, value in totals.items()}
    result.update(
        {
            "event_count": len(items),
            "independent_sequence_count": len(sequence_values),
            "baseline_future_identity_error": totals["baseline_identity_error_frames"] / denominator,
            "treatment_future_identity_error": totals["treatment_identity_error_frames"] / denominator,
            "identity_error_reduction": totals["identity_error_reduction_sum"] / denominator,
            "baseline_mean_iou": totals["baseline_iou_sum"] / denominator,
            "treatment_mean_iou": totals["treatment_iou_sum"] / denominator,
            "delta_iou": totals["delta_iou_sum"] / denominator,
            "missing_rate": totals["target_missing_frames"] / denominator,
            "wrong_reassociation_rate": totals["wrong_reassociation_frames"] / denominator,
            "candidate_recall": totals["candidate_present_frames"] / denominator,
            "assignment_change_rate": totals["assignment_change_count"] / denominator,
            "target_assignment_change_rate": totals["target_assignment_change_count"] / denominator,
            "id_switch_rate": totals["id_switch_count"] / denominator,
            "recorrection_rate": totals["recorrection_opportunity_count"] / denominator,
            "protected_regression_rate": None if protected == 0 else totals["protected_regression_count"] / protected,
            "sequence_cluster_bootstrap_95ci": sequence_cluster_bootstrap(sequence_values, seed=7211, repetitions=2000),
        }
    )
    return result


def _score(event: Mapping[str, Any], rows: Mapping[str, list[dict[str, Any]]], baseline: str, treatment: str, horizon: int, gt: Mapping[int, Mapping[int, Any]], protected: Mapping[int, int]) -> dict[str, Any]:
    runtime = {name: {int(row["frame"]): row for row in values[: horizon + 1]} for name, values in rows.items()}
    return legacy._score_pair(
        {
            "event_id": str(event["event_id"]),
            "sequence": str(event["sequence"]),
            "event_frame": int(event["event_frame"]),
            "target_public_id": int(event["target_public_id"]),
            "target_dataset_gt_id": int(event["dataset_gt_id"]),
            "rows": runtime,
        }, baseline, treatment, int(horizon), gt, protected,
    )


def _load_frozen_event(event: Mapping[str, Any], r5r1_record: Mapping[str, Any], safe_record: Mapping[str, Any]) -> dict[str, Any]:
    old_dir = Path(str(r5r1_record["done"])).parent
    new_dir = Path(str(safe_record["done"])).parent
    e0_path = old_dir / "E0_BASELINE_B0/runtime_frames.jsonl"
    e1b_path = old_dir / "E1B_PCTIS_LEGACY/runtime_frames.jsonl"
    e1c_path = new_dir / "E1C_PCTIS_SAFE/runtime_frames.jsonl"
    e0 = _jsonl(e0_path)
    e1b = _jsonl(e1b_path)
    e1c = _jsonl(e1c_path)
    if len(e0) != 101 or len(e1b) != 101 or len(e1c) != 101:
        raise RuntimeError(f"incomplete H100 rows for {event['event_id']}")
    if _sha(e0_path) != _sha(new_dir / "E0_BASELINE_B0/runtime_frames.jsonl"):
        raise RuntimeError(f"E0 regeneration is not byte-equivalent for {event['event_id']}")
    for name, rows in (("E0", e0), ("E1B", e1b), ("E1C", e1c)):
        if any(row.get("runtime_future_gt_used") is not False or row.get("runtime_gt_read") is not False for row in rows):
            raise RuntimeError(f"runtime GT flag in {event['event_id']} {name}")
    posthoc = _read(new_dir / "posthoc.json")
    if posthoc.get("posthoc_gt_used") is not True or posthoc.get("runtime_future_gt_used") is not False:
        raise RuntimeError(f"posthoc sealing invalid for {event['event_id']}")
    return {"e0": e0, "e1b": e1b, "e1c": e1c, "e0_path": str(e0_path), "e1b_path": str(e1b_path), "e1c_path": str(e1c_path), "e1c_done": str(safe_record["done"]), "posthoc": posthoc, "old_dir": str(old_dir), "new_dir": str(new_dir)}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--safe-manifest", type=Path, default=ROOT / "outputs/N72R12/formal_safe/formal_safe_manifest.json")
    parser.add_argument("--r5r1-manifest", type=Path, default=ROOT / "outputs/N72R11R5R1/formal_e1b_manifest.json")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/N72R12/causal/formal_causal_metrics.json")
    args = parser.parse_args()
    safe_path = args.safe_manifest.resolve()
    r5_path = args.r5r1_manifest.resolve()
    safe_manifest = _read(safe_path)
    r5_manifest = _read(r5_path)
    if safe_manifest.get("status") != "PASS_N72R12_FORMAL_RUNTIME" or safe_manifest.get("event_count") != 32:
        raise RuntimeError("safe deterministic manifest is incomplete")
    safe_records = {str(item["event_id"]): item for item in safe_manifest["records"]}
    r5_records = {str(item["event_id"]): item for item in r5_manifest["records"]}
    protocol = _read(ROOT / "outputs/N72R9/protocol.json")
    events = {str(item["event_id"]): item for item in protocol["source_event_selection"]["events"]}
    if set(safe_records) != set(r5_records) or set(safe_records) != set(events) or len(events) != 32:
        raise RuntimeError("N72R12/R5R1/protocol event keys are not identical")
    comparison_items: dict[str, dict[int, list[dict[str, Any]]]] = {name: {horizon: [] for horizon in HORIZONS} for name in COMPARISONS}
    event_rows: list[dict[str, Any]] = []
    diagnostics: Counter[str] = Counter()
    diagnostics_by_action: dict[str, Counter[str]] = defaultdict(Counter)
    rejection_by_action: dict[str, Counter[str]] = defaultdict(Counter)
    for event_id in sorted(events):
        event = dict(events[event_id])
        safe_record = safe_records[event_id]
        r5_record = r5_records[event_id]
        if safe_record.get("status") != "PASS" or r5_record.get("status") != "PASS":
            raise RuntimeError(f"non-PASS source record: {event_id}")
        event_data = _load_frozen_event(event, r5_record, safe_record)
        inputs = legacy._load_rows(event)
        event["target_public_id"] = int(inputs["target_public_id"])
        gt = legacy._load_gt(str(event["sequence"]))
        protected = legacy._protected_map(inputs["rows"]["c0_source"][int(event["event_frame"])], gt, int(event["event_frame"]), int(event["dataset_gt_id"]))
        event_metrics: dict[str, Any] = {}
        for comparison, baseline, treatment, rows in (
            ("E1B_vs_E0", "E0", "E1B", {"E0": event_data["e0"], "E1B": event_data["e1b"]}),
            ("E1C_vs_E0", "E0", "E1C", {"E0": event_data["e0"], "E1C": event_data["e1c"]}),
            ("E1C_vs_E1B", "E1B", "E1C", {"E1B": event_data["e1b"], "E1C": event_data["e1c"]}),
        ):
            event_metrics[comparison] = {}
            for horizon in HORIZONS:
                metric = _score(event, rows, baseline, treatment, horizon, gt, protected)
                event_metrics[comparison][str(horizon)] = metric
                comparison_items[comparison][horizon].append({"event_id": event_id, "sequence": str(event["sequence"]), "action_type": str(event["action_type"]), "metric": metric})
        e1c_rows = event_data["e1c"]
        decision_counts = Counter()
        for row in e1c_rows[1:]:
            cf = row.get("counterfactual_intervention", {})
            decision = str(cf.get("decision"))
            decision_counts[decision] += 1
            diagnostics[f"decision:{decision}"] += 1
            diagnostics_by_action[str(event["action_type"])][f"decision:{decision}"] += 1
            if decision == "KEEP_BASELINE":
                for reason in cf.get("decision_audit", {}).get("rejection_reasons", []):
                    diagnostics[f"rejection:{reason}"] += 1
                    rejection_by_action[str(event["action_type"])][str(reason)] += 1
        stats = _read(Path(str(safe_record["done"]))).get("stats", {}).get("E1C_PCTIS_SAFE", {})
        event_rows.append({
            "event_id": event_id,
            "sequence": str(event["sequence"]),
            "action_type": str(event["action_type"]),
            "metrics": event_metrics,
            "e1c_decision_counts": dict(decision_counts),
            "e1c_stats": stats,
            "sources": {"e0": event_data["e0_path"], "e1b": event_data["e1b_path"], "e1c": event_data["e1c_path"]},
        })
    aggregate: dict[str, Any] = {}
    for comparison in COMPARISONS:
        aggregate[comparison] = {str(horizon): _metric_aggregate(comparison_items[comparison][horizon], horizon) for horizon in HORIZONS}
    action_breakdown: dict[str, Any] = {}
    for action in sorted({item["action_type"] for item in event_rows}):
        action_breakdown[action] = {}
        for comparison in COMPARISONS:
            action_breakdown[action][comparison] = {}
            for horizon in HORIZONS:
                action_breakdown[action][comparison][str(horizon)] = _metric_aggregate([item for item in comparison_items[comparison][horizon] if item["action_type"] == action], horizon)
    sequence_breakdown: dict[str, Any] = {}
    for sequence in sorted({item["sequence"] for item in event_rows}):
        sequence_breakdown[sequence] = {}
        for comparison in COMPARISONS:
            sequence_breakdown[sequence][comparison] = {}
            for horizon in HORIZONS:
                subset = [item for item in comparison_items[comparison][horizon] if item["sequence"] == sequence]
                sequence_breakdown[sequence][comparison][str(horizon)] = _metric_aggregate(subset, horizon) if subset else None
    payload = {
        "schema_version": "N72R12_FORMAL_CAUSAL_METRICS_V1",
        "status": "PASS_N72R12_FORMAL_CAUSAL_AGGREGATION",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "event_count": len(event_rows),
        "independent_sequence_count": len({item["sequence"] for item in event_rows}),
        "horizons": list(HORIZONS),
        "comparisons": aggregate,
        "action_breakdown": action_breakdown,
        "sequence_breakdown": sequence_breakdown,
        "event_rows": event_rows,
        "safe_diagnostics": {
            "overall": dict(diagnostics),
            "by_action": {action: dict(counter) for action, counter in sorted(diagnostics_by_action.items())},
            "rejection_by_action": {action: dict(counter) for action, counter in sorted(rejection_by_action.items())},
        },
        "bootstrap": {"seed": 7211, "repetitions": 2000, "unit": "independent_sequence", "within_sequence": "mean_event_value"},
        "protocol": str(ROOT / "outputs/N72R9/protocol.json"),
        "protocol_sha256": _sha(ROOT / "outputs/N72R9/protocol.json"),
        "safe_manifest": str(safe_path),
        "safe_manifest_sha256": _sha(safe_path),
        "r5r1_manifest": str(r5_path),
        "r5r1_manifest_sha256": _sha(r5_path),
        "runtime_future_gt_used": False,
        "posthoc_gt_used": True,
        "interaction_source": "simulated_from_gt",
        "not_real_human_evidence": True,
        "scientific_effect_gate": "PENDING_TRACKEVAL_AND_STRICT_CSI_GATE",
    }
    _atomic_json(args.output.resolve(), payload)
    print(json.dumps({"status": payload["status"], "events": len(event_rows), "sequences": payload["independent_sequence_count"], "output": str(args.output.resolve())}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
