#!/usr/bin/env python3
"""Post-hoc CSI oracle headroom from the frozen corrected R5R1 streams.

This is deliberately not a runtime replay.  It is allowed to inspect the
sealed R5R1 posthoc GT labels after both E0 and E1B have completed and reports
an upper bound for a selective intervention.  No oracle row is runtime
eligible and no source runtime artifact is modified.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
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
SEED = 7211
REPETITIONS = 2000


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object: {path}")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not all(isinstance(row, dict) for row in rows):
        raise TypeError(f"non-object row: {path}")
    return rows


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


def _event_input(event: Mapping[str, Any]) -> dict[str, Any]:
    # _load_rows only reads the frozen candidate/tape sources and event
    # metadata.  It does not open dataset GT.
    return legacy._load_rows(event)


def _oracle_rows(e0: list[dict[str, Any]], e1b: list[dict[str, Any]], details: list[Mapping[str, Any]], protected_zero: bool) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    detail_by_frame = {int(item["frame"]): item for item in details}
    rows: list[dict[str, Any]] = []
    decisions: list[dict[str, Any]] = []
    for base, proposal in zip(e0, e1b):
        frame = int(base["frame"])
        detail = detail_by_frame.get(frame)
        use_proposal = bool(
            protected_zero
            and detail is not None
            and detail.get("baseline_correct") is False
            and detail.get("treatment_correct") is True
        )
        selected = proposal if use_proposal else base
        rows.append(selected)
        decisions.append(
            {
                "frame": frame,
                "selected_variant": "E1B_PCTIS_LEGACY" if use_proposal else "E0_BASELINE_B0",
                "oracle_condition": "E0_WRONG_E1B_CORRECT_AND_E1B_PROTECTED_REGRESSION_ZERO" if use_proposal else "KEEP_E0",
                "gt_used_for_decision": True,
                "runtime_eligible": False,
                "scientific_runtime_result": False,
                "oracle_upper_bound_only": True,
            }
        )
    return rows, decisions


def _aggregate(items: list[dict[str, Any]], horizon: int) -> dict[str, Any]:
    metrics = [item["metrics"][str(horizon)] for item in items]
    sequence_values: dict[str, list[float]] = defaultdict(list)
    for item in items:
        sequence_values[str(item["sequence"])].append(float(item["metrics"][str(horizon)]["identity_error_reduction"]))
    totals = {
        "evaluated_frames": sum(int(metric.get("evaluated_frames", 0)) for metric in metrics),
        "baseline_identity_error_frames": sum(int(metric.get("baseline_identity_error_frames", 0)) for metric in metrics),
        "treatment_identity_error_frames": sum(int(metric.get("treatment_identity_error_frames", 0)) for metric in metrics),
        "identity_error_reduction_sum": sum(float(metric.get("identity_error_reduction_sum", 0.0)) for metric in metrics),
        "assignment_change_count": sum(int(metric.get("assignment_change_count", 0)) for metric in metrics),
        "true_correct_crossing_count": sum(int(metric.get("true_correct_crossing_count", 0)) for metric in metrics),
        "true_incorrect_crossing_count": sum(int(metric.get("true_incorrect_crossing_count", 0)) for metric in metrics),
        "protected_regression_count": sum(int(metric.get("protected_regression_count", 0)) for metric in metrics),
        "protected_compared": sum(int(metric.get("protected_compared", 0)) for metric in metrics),
        "target_missing_frames": sum(int(metric.get("target_missing_frames", 0)) for metric in metrics),
        "recorrection_opportunity_count": sum(int(metric.get("recorrection_opportunity_count", 0)) for metric in metrics),
    }
    denominator = max(totals["evaluated_frames"], 1)
    return {
        "horizon": horizon,
        "event_count": len(items),
        "independent_sequence_count": len(sequence_values),
        **totals,
        "baseline_future_identity_error": totals["baseline_identity_error_frames"] / denominator,
        "oracle_future_identity_error": totals["treatment_identity_error_frames"] / denominator,
        "identity_error_reduction": totals["identity_error_reduction_sum"] / denominator,
        "assignment_change_count": totals["assignment_change_count"],
        "true_correct_crossing_count": totals["true_correct_crossing_count"],
        "true_incorrect_crossing_count": totals["true_incorrect_crossing_count"],
        "missing_rate": totals["target_missing_frames"] / denominator,
        "recorrection_rate": totals["recorrection_opportunity_count"] / denominator,
        "protected_regression_rate": None if totals["protected_compared"] == 0 else totals["protected_regression_count"] / totals["protected_compared"],
        "sequence_cluster_bootstrap_95ci": sequence_cluster_bootstrap(sequence_values, seed=SEED, repetitions=REPETITIONS),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--r5r1-manifest", type=Path, default=ROOT / "outputs/N72R11R5R1/formal_e1b_manifest.json")
    parser.add_argument("--output-root", type=Path, default=ROOT / "outputs/N72R12/oracle")
    args = parser.parse_args()
    manifest_path = args.r5r1_manifest.resolve()
    output_root = args.output_root.resolve()
    manifest = _read_json(manifest_path)
    records = manifest.get("records")
    if not isinstance(records, list) or len(records) != 32 or len({str(item.get("event_id")) for item in records if isinstance(item, dict)}) != 32:
        raise RuntimeError("frozen R5R1 E1B manifest is not exactly 32 unique records")
    items: list[dict[str, Any]] = []
    event_summaries: list[dict[str, Any]] = []
    for record in sorted(records, key=lambda item: str(item["event_id"])):
        if record.get("status") != "PASS":
            raise RuntimeError(f"R5R1 record is not PASS: {record.get('event_id')}")
        done_path = Path(str(record["done"]))
        event_id = str(record["event_id"])
        event_dir = done_path.parent
        e0_path = event_dir / "E0_BASELINE_B0/runtime_frames.jsonl"
        e1b_path = event_dir / "E1B_PCTIS_LEGACY/runtime_frames.jsonl"
        posthoc_path = event_dir / "posthoc.json"
        e0 = _read_jsonl(e0_path)
        e1b = _read_jsonl(e1b_path)
        posthoc = _read_json(posthoc_path)
        comparison = posthoc["event"]["comparisons"]["E1B_vs_E0"]
        details = comparison["100"]["frame_details"]
        protected_zero = int(comparison["100"].get("protected_regression_count", 0)) == 0
        oracle, decisions = _oracle_rows(e0, e1b, details, protected_zero)
        event = next(item for item in _read_json(ROOT / "outputs/N72R9/protocol.json")["source_event_selection"]["events"] if str(item["event_id"]) == event_id)
        inputs = _event_input(event)
        target_public_id = int(inputs["target_public_id"])
        gt = legacy._load_gt(str(event["sequence"]))
        protected = legacy._protected_map(inputs["rows"]["c0_source"][int(event["event_frame"])], gt, int(event["event_frame"]), int(event["dataset_gt_id"]))
        metrics: dict[str, Any] = {}
        for horizon in HORIZONS:
            metrics[str(horizon)] = legacy._score_pair(
                {
                    "event_id": event_id,
                    "sequence": str(event["sequence"]),
                    "event_frame": int(event["event_frame"]),
                    "target_public_id": target_public_id,
                    "target_dataset_gt_id": int(event["dataset_gt_id"]),
                    "rows": {"E0_BASELINE_B0": {int(row["frame"]): row for row in e0[: horizon + 1]}, "ORACLE_CSI": {int(row["frame"]): row for row in oracle[: horizon + 1]}},
                },
                "E0_BASELINE_B0",
                "ORACLE_CSI",
                horizon,
                gt,
                protected,
            )
        event_artifact = {
            "schema_version": "N72R12_ORACLE_INTERVENTION_EVENT_V1",
            "status": "PASS_POSTHOC_ORACLE_HEADROOM",
            "event_id": event_id,
            "sequence": str(event["sequence"]),
            "action_type": str(event["action_type"]),
            "event_frame": int(event["event_frame"]),
            "horizons": list(HORIZONS),
            "metrics": metrics,
            "decisions": decisions,
            "e1b_protected_regression_count_h100": int(comparison["100"].get("protected_regression_count", 0)),
            "gt_used_for_decision": True,
            "runtime_eligible": False,
            "scientific_runtime_result": False,
            "oracle_upper_bound_only": True,
            "runtime_future_gt_used": False,
            "interaction_source": "simulated_from_gt",
            "not_real_human_evidence": True,
            "sources": {"e0": str(e0_path), "e0_sha256": _sha(e0_path), "e1b": str(e1b_path), "e1b_sha256": _sha(e1b_path), "posthoc": str(posthoc_path), "posthoc_sha256": _sha(posthoc_path)},
        }
        event_out = output_root / "events" / f"{event_id}.json"
        _atomic_json(event_out, event_artifact)
        for horizon in HORIZONS:
            items.append({"event_id": event_id, "sequence": str(event["sequence"]), "action_type": str(event["action_type"]), "metrics": {str(horizon): metrics[str(horizon)]}})
        event_summaries.append({"event_id": event_id, "sequence": str(event["sequence"]), "action_type": str(event["action_type"]), "event_artifact": str(event_out), "selected_e1b_frames_h100": sum(int(item["selected_variant"] == "E1B_PCTIS_LEGACY") for item in decisions), "protected_zero": protected_zero})
    # The item list above intentionally has one item per event per horizon;
    # regroup before aggregate so each horizon sees all 32 events.
    per_horizon_items: dict[int, list[dict[str, Any]]] = {horizon: [] for horizon in HORIZONS}
    for event_summary in event_summaries:
        artifact = _read_json(Path(event_summary["event_artifact"]))
        for horizon in HORIZONS:
            per_horizon_items[horizon].append({"event_id": event_summary["event_id"], "sequence": event_summary["sequence"], "action_type": event_summary["action_type"], "metrics": {str(horizon): artifact["metrics"][str(horizon)]}})
    aggregate = {str(horizon): _aggregate(per_horizon_items[horizon], horizon) for horizon in HORIZONS}
    manifest_out = {
        "schema_version": "N72R12_ORACLE_INTERVENTION_MANIFEST_V1",
        "status": "PASS_POSTHOC_ORACLE_HEADROOM",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "event_count": len(event_summaries),
        "independent_sequence_count": len({item["sequence"] for item in event_summaries}),
        "records": event_summaries,
        "aggregate": aggregate,
        "horizons": list(HORIZONS),
        "bootstrap": {"seed": SEED, "repetitions": REPETITIONS, "unit": "independent_sequence"},
        "gt_used_for_decision": True,
        "runtime_eligible": False,
        "scientific_runtime_result": False,
        "oracle_upper_bound_only": True,
        "runtime_future_gt_used": False,
        "interaction_source": "simulated_from_gt",
        "not_real_human_evidence": True,
        "source_manifest": str(manifest_path),
        "source_manifest_sha256": _sha(manifest_path),
    }
    _atomic_json(output_root / "oracle_manifest.json", manifest_out)
    _atomic_json(output_root / "oracle_headroom.json", {"schema_version": "N72R12_ORACLE_HEADROOM_V1", "status": "PASS_POSTHOC_ORACLE_HEADROOM", "horizons": aggregate, "gt_used_for_decision": True, "runtime_eligible": False, "scientific_runtime_result": False, "oracle_upper_bound_only": True})
    print(json.dumps({"status": manifest_out["status"], "events": len(event_summaries), "output": str(output_root / "oracle_manifest.json")}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
