#!/usr/bin/env python3
"""Posthoc aggregate the fixed N72R12 E1D replay matrix."""

from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime, timezone
import argparse
import json
from pathlib import Path
import sys
import traceback
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import n72r9_temporal_replay as legacy  # noqa: E402
from scripts.n72r12_aggregate_formal_metrics import (  # noqa: E402
    HORIZONS,
    _atomic_json,
    _jsonl,
    _metric_aggregate,
    _read,
    _sha,
    _score,
)

COMPARISONS = (
    ("E1D_vs_E0", "E0", "E1D"),
    ("E1D_vs_E1B", "E1B", "E1D"),
    ("E1D_vs_E1C", "E1C", "E1D"),
)


def _load_rows(path: Path, label: str) -> list[dict[str, Any]]:
    rows = _jsonl(path)
    if len(rows) != 101 or [int(row.get("frame", -1)) for row in rows] != list(range(int(rows[0]["event_frame"]), int(rows[0]["event_frame"]) + 101)):
        raise RuntimeError(f"{label} frame axis is incomplete")
    if any(row.get("runtime_future_gt_used") is not False or row.get("runtime_gt_read") is not False for row in rows):
        raise RuntimeError(f"{label} contains a runtime GT flag")
    return rows


def _event_data(event_id: str, r5: Mapping[str, Any], safe: Mapping[str, Any], learned: Mapping[str, Any]) -> dict[str, Any]:
    old_dir = Path(str(r5["done"])).parent
    safe_dir = Path(str(safe["done"])).parent
    learned_dir = Path(str(learned["e1d_done"])).parent
    e0_path = old_dir / "E0_BASELINE_B0/runtime_frames.jsonl"
    e1b_path = old_dir / "E1B_PCTIS_LEGACY/runtime_frames.jsonl"
    e1c_path = safe_dir / "E1C_PCTIS_SAFE/runtime_frames.jsonl"
    e1d_path = learned_dir / "E1D_PCTIS_LEARNED_SAFE/runtime_frames.jsonl"
    return {
        "e0": _load_rows(e0_path, f"{event_id}:E0"),
        "e1b": _load_rows(e1b_path, f"{event_id}:E1B"),
        "e1c": _load_rows(e1c_path, f"{event_id}:E1C"),
        "e1d": _load_rows(e1d_path, f"{event_id}:E1D"),
        "paths": {"e0": str(e0_path), "e1b": str(e1b_path), "e1c": str(e1c_path), "e1d": str(e1d_path)},
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--learned-manifest", type=Path, default=ROOT / "outputs/N72R12/formal_learned/formal_learned_manifest.json")
    parser.add_argument("--safe-manifest", type=Path, default=ROOT / "outputs/N72R12/formal_safe/formal_safe_manifest.json")
    parser.add_argument("--r5r1-manifest", type=Path, default=ROOT / "outputs/N72R11R5R1/formal_e1b_manifest.json")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/N72R12/causal/learned_causal_metrics.json")
    args = parser.parse_args()
    started = datetime.now(timezone.utc).isoformat()
    try:
        learned_path = args.learned_manifest.resolve()
        safe_path = args.safe_manifest.resolve()
        r5_path = args.r5r1_manifest.resolve()
        learned_manifest = _read(learned_path)
        safe_manifest = _read(safe_path)
        r5_manifest = _read(r5_path)
        if learned_manifest.get("status") != "PASS_N72R12_LEARNED_RUNTIME" or learned_manifest.get("event_count") != 32:
            raise RuntimeError("learned runtime manifest is incomplete")
        if safe_manifest.get("status") != "PASS_N72R12_FORMAL_RUNTIME" or safe_manifest.get("event_count") != 32:
            raise RuntimeError("E1C runtime manifest is incomplete")
        if r5_manifest.get("status") != "PASS_ALL_SELECTED" or r5_manifest.get("event_count") != 32:
            raise RuntimeError("R5R1 reference manifest is incomplete")
        r5_records = {str(item["event_id"]): item for item in r5_manifest["records"]}
        safe_records = {str(item["event_id"]): item for item in safe_manifest["records"]}
        learned_records = {str(item["event_id"]): item for item in learned_manifest["records"]}
        protocol = _read(ROOT / "outputs/N72R9/protocol.json")
        events = {str(item["event_id"]): dict(item) for item in protocol["source_event_selection"]["events"]}
        if len(events) != 32 or set(events) != set(r5_records) or set(events) != set(safe_records) or set(events) != set(learned_records):
            raise RuntimeError("N72R12 learned source event keys are not exactly the frozen 32")
        comparison_items: dict[str, dict[int, list[dict[str, Any]]]] = {name: {horizon: [] for horizon in HORIZONS} for name, _, _ in COMPARISONS}
        event_rows: list[dict[str, Any]] = []
        decision_counts: Counter[str] = Counter()
        deterministic_counts: Counter[str] = Counter()
        learned_counts: Counter[str] = Counter()
        blocked_count = 0
        by_action: dict[str, Counter[str]] = defaultdict(Counter)
        for event_id in sorted(events):
            event = dict(events[event_id])
            data = _event_data(event_id, r5_records[event_id], safe_records[event_id], learned_records[event_id])
            inputs = legacy._load_rows(event)
            event["target_public_id"] = int(inputs["target_public_id"])
            gt = legacy._load_gt(str(event["sequence"]))
            protected = legacy._protected_map(inputs["rows"]["c0_source"][int(event["event_frame"])], gt, int(event["event_frame"]), int(event["dataset_gt_id"]))
            rows_by_name = {"E0": data["e0"], "E1B": data["e1b"], "E1C": data["e1c"], "E1D": data["e1d"]}
            metrics: dict[str, Any] = {}
            for comparison, baseline, treatment in COMPARISONS:
                metrics[comparison] = {}
                for horizon in HORIZONS:
                    metric = _score(event, {baseline: rows_by_name[baseline], treatment: rows_by_name[treatment]}, baseline, treatment, horizon, gt, protected)
                    metrics[comparison][str(horizon)] = metric
                    comparison_items[comparison][horizon].append({"event_id": event_id, "sequence": str(event["sequence"]), "action_type": str(event["action_type"]), "metric": metric})
            gate_decisions = Counter()
            for row in data["e1d"][1:]:
                gate = row.get("counterfactual_intervention", {}).get("learned_gate", {})
                final = row.get("counterfactual_intervention", {})
                learned_decision = str(gate.get("decision"))
                safety_decision = str(gate.get("deterministic_safety_decision", {}).get("decision"))
                final_decision = str(final.get("decision"))
                gate_decisions[f"learned:{learned_decision}"] += 1
                gate_decisions[f"safety:{safety_decision}"] += 1
                gate_decisions[f"final:{final_decision}"] += 1
                learned_counts[learned_decision] += 1
                deterministic_counts[safety_decision] += 1
                decision_counts[final_decision] += 1
                blocked_count += int(learned_decision == "APPLY_PCTIS" and final_decision == "KEEP_BASELINE")
            for key, value in gate_decisions.items():
                by_action[str(event["action_type"])][key] += int(value)
            event_rows.append({"event_id": event_id, "sequence": str(event["sequence"]), "action_type": str(event["action_type"]), "metrics": metrics, "e1d_gate_counts": dict(gate_decisions), "sources": data["paths"]})
        aggregate = {comparison: {str(horizon): _metric_aggregate(comparison_items[comparison][horizon], horizon) for horizon in HORIZONS} for comparison, _, _ in COMPARISONS}
        action_breakdown: dict[str, Any] = {}
        for action in sorted({item["action_type"] for item in event_rows}):
            action_breakdown[action] = {}
            for comparison, _, _ in COMPARISONS:
                action_breakdown[action][comparison] = {}
                for horizon in HORIZONS:
                    subset = [item for item in comparison_items[comparison][horizon] if item["action_type"] == action]
                    action_breakdown[action][comparison][str(horizon)] = _metric_aggregate(subset, horizon)
        sequence_breakdown: dict[str, Any] = {}
        for sequence in sorted({item["sequence"] for item in event_rows}):
            sequence_breakdown[sequence] = {}
            for comparison, _, _ in COMPARISONS:
                sequence_breakdown[sequence][comparison] = {}
                for horizon in HORIZONS:
                    subset = [item for item in comparison_items[comparison][horizon] if item["sequence"] == sequence]
                    sequence_breakdown[sequence][comparison][str(horizon)] = _metric_aggregate(subset, horizon) if subset else None
        payload = {
            "schema_version": "N72R12_LEARNED_CAUSAL_METRICS_V1",
            "status": "PASS_N72R12_LEARNED_CAUSAL_AGGREGATION",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "started_at_utc": started,
            "event_count": len(event_rows),
            "independent_sequence_count": len({item["sequence"] for item in event_rows}),
            "horizons": list(HORIZONS),
            "comparisons": aggregate,
            "action_breakdown": action_breakdown,
            "sequence_breakdown": sequence_breakdown,
            "event_rows": event_rows,
            "safe_diagnostics": {
                "final_decision_counts": dict(decision_counts),
                "deterministic_decision_counts": dict(deterministic_counts),
                "learned_decision_counts": dict(learned_counts),
                "learned_apply_blocked_by_deterministic_or_final_keep": blocked_count,
                "by_action": {action: dict(counter) for action, counter in sorted(by_action.items())},
            },
            "bootstrap": {"seed": 7211, "repetitions": 2000, "unit": "independent_sequence", "within_sequence": "mean_event_value"},
            "protocol": str(ROOT / "outputs/N72R9/protocol.json"),
            "protocol_sha256": _sha(ROOT / "outputs/N72R9/protocol.json"),
            "r5r1_manifest": str(r5_path),
            "r5r1_manifest_sha256": _sha(r5_path),
            "safe_manifest": str(safe_path),
            "safe_manifest_sha256": _sha(safe_path),
            "learned_manifest": str(learned_path),
            "learned_manifest_sha256": _sha(learned_path),
            "runtime_future_gt_used": False,
            "posthoc_gt_used": True,
            "interaction_source": "simulated_from_gt",
            "not_real_human_evidence": True,
            "scientific_effect_gate": "PENDING_LEARNED_TRACKEVAL_AND_STRICT_GATE",
            "production_authorized": False,
            "finished_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        _atomic_json(args.output.resolve(), payload)
        print(json.dumps({"status": payload["status"], "events": payload["event_count"], "sequences": payload["independent_sequence_count"], "output": str(args.output.resolve())}, sort_keys=True))
        return 0
    except Exception as exc:
        failure = {"schema_version": "N72R12_LEARNED_CAUSAL_FAILURE_V1", "status": "FAIL_N72R12_LEARNED_CAUSAL_AGGREGATION", "started_at_utc": started, "finished_at_utc": datetime.now(timezone.utc).isoformat(), "error_type": type(exc).__name__, "error": str(exc), "traceback": traceback.format_exc(), "historical_outputs_modified": False}
        _atomic_json(ROOT / "outputs/N72R12/causal/learned_causal_failure.json", failure)
        print(json.dumps({"status": failure["status"], "error": str(exc)}, sort_keys=True), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
