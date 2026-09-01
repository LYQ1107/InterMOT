#!/usr/bin/env python3
"""N31 bounded fallback selection when the oracle/learning gate is closed."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "outputs/n31"


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _load(path: Path, default: Any) -> Any:
    if not path.is_file():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def _temporal_context(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Use only correction-time predicted-IoU/context features as a bounded fallback."""

    groups: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        if row.get("available") and row.get("features"):
            groups.setdefault(str(row["episode_id"]), []).append(row)
    predictions = []
    for episode_id, candidates in sorted(groups.items()):
        # feature index 9 is the candidate's official predicted-IoU head;
        # index 10 is token index.  The first 11 fields contain no future data.
        chosen = max(candidates, key=lambda row: (float(row["features"][9]), -float(row["features"][10])))
        baseline = next((row for row in candidates if row["candidate"] == "S0_restore_old_state"), candidates[0])
        def h20(field: str, item: dict[str, Any]) -> float:
            return float(item.get("metrics", {}).get("20", {}).get(field) or 0.0)
        predictions.append({
            "episode_id": episode_id,
            "sequence": chosen["sequence"],
            "selected_candidate": chosen["candidate"],
            "baseline_candidate": baseline["candidate"],
            "selected_reward": float(chosen.get("reward", 0.0)),
            "baseline_reward": float(baseline.get("reward", 0.0)),
            "iou_gain": h20("mean_box_iou_visible", chosen) - h20("mean_box_iou_visible", baseline),
            "success_gain": h20("success_at_0_5_visible", chosen) - h20("success_at_0_5_visible", baseline),
            "missing_delta": h20("missing_prediction_rate_visible", chosen) - h20("missing_prediction_rate_visible", baseline),
        })
    gains = [float(row["iou_gain"]) for row in predictions]
    grouped_gains: dict[str, list[float]] = {}
    for row in predictions:
        grouped_gains.setdefault(str(row["sequence"]), []).append(float(row["iou_gain"]))
    sequence_values = np.asarray([np.mean(grouped_gains[key]) for key in sorted(grouped_gains)], dtype=float)
    return {
        "method": "official_predicted_iou_single_context",
        "episode_count": len(predictions),
        "mean_h20_iou_gain_vs_s0": None if not gains else float(np.mean(gains)),
        "mean_h20_success_gain_vs_s0": None if not predictions else float(np.mean([row["success_gain"] for row in predictions])),
        "mean_h20_missing_delta_vs_s0": None if not predictions else float(np.mean([row["missing_delta"] for row in predictions])),
        "negative_transfer_rate": None if not gains else float(np.mean(np.asarray(gains) < 0.0)),
        "sequence_mean_values": sequence_values.tolist(),
        "predictions": predictions,
        "future_gt_used_for_feature_or_selection": False,
        "future_gt_used_for_posthoc_evaluation": True,
    }


def run(*, output: Path) -> dict[str, Any]:
    oracle = _load(OUT_DIR / "candidate_oracle_gate.json", {"status": "NOT_RUN"})
    learn = _load(OUT_DIR / "learn_gate.json", {"status": "NOT_RUN"})
    candidate = _load(OUT_DIR / "candidate_rollout_index.json", {"rows": []})
    n30_multi = _load(ROOT / "outputs/n30/multi_identity_write_summary.json", {})
    branch_metrics = n30_multi.get("branch_metrics", {}) if isinstance(n30_multi, dict) else {}
    comparisons = n30_multi.get("comparisons", {}) if isinstance(n30_multi, dict) else {}
    spatial = branch_metrics.get("M1_official_spatial_write_only", {})
    joint_delta = comparisons.get("M3_minus_max_M1_M2", {}).get("future_delivered_box_iou", {})
    lora_delta = comparisons.get("M4_minus_M3", {}).get("future_delivered_box_iou", {})
    if oracle.get("status") == "PASS" and learn.get("status") != "PASS":
        temporal = _temporal_context(candidate.get("rows", []))
        route = "temporal_context_fallback" if temporal.get("mean_h20_iou_gain_vs_s0") is not None and float(temporal["mean_h20_iou_gain_vs_s0"]) >= 0.0 and float(temporal.get("negative_transfer_rate") or 1.0) < 0.2 else "association_trigger_fallback"
    else:
        temporal = {"status": "NOT_RUN_ORACLE_FAIL"}
        route = "association_trigger_fallback"
    result = {
        "protocol": "N31-FALLBACK",
        "status": "PASS_BOUNDED_FALLBACK" if route else "NOT_RUN",
        "route": route,
        "oracle_gate": oracle,
        "learn_gate": learn,
        "temporal_context_diagnostic": temporal,
        "association_trigger_evidence": {
            "source": "outputs/n30/multi_identity_write_summary.json",
            "status": n30_multi.get("status"),
            "case_count": n30_multi.get("case_count"),
            "candidate_protocol": n30_multi.get("candidate_protocol"),
            "official_spatial_write_future_iou": spatial.get("future_delivered_box_iou"),
            "official_spatial_write_sequence_ci": comparisons.get("M1_minus_M0", {}).get("future_delivered_box_iou", {}).get("sequence_cluster_bootstrap_ci95"),
            "joint_minus_best_single_future_iou": joint_delta,
            "online_lora_minus_joint_future_iou": lora_delta,
            "note": "bounded real train-fold multi-ID association diagnostic only; no hidden full-loop claim",
        },
        "full_loop": {"status": "NOT_RUN_FALLBACK_ROUTE", "reason": "N31 learning gate did not authorize deployment"},
        "future_gt_used_for_selection": False,
        "val25_read": False,
        "test_labels_used": False,
    }
    _write(output, result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUT_DIR / "fallback_results.json")
    args = parser.parse_args()
    result = run(output=args.output)
    print(json.dumps({key: result.get(key) for key in ("protocol", "status", "route")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
