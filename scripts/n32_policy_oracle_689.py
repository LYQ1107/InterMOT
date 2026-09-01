#!/usr/bin/env python3
"""N32-C oracle and gate over all three real policy rollouts."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from scripts.n32_policy_semantics import policy_metric_issues, visible_h20_status

ROOT = Path(__file__).resolve().parents[1]
INDEX = ROOT / "outputs/n32/policy_rollout_index.json"
OUT = ROOT / "outputs/n32/policy_oracle_689.json"
POLICIES = ("K0_KEEP_OLD", "K1_APPLY_ENSURE", "K2_PROMPT_THEN_RESTORE")


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _h20(row: Mapping[str, Any], key: str) -> float | None:
    value = row.get("metrics", {}).get("20", {}).get(key)
    if value is None:
        return None
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if np.isfinite(value) else None


def _mean_defined(rows: Sequence[Mapping[str, Any]], key: str) -> float | None:
    values = [row[key] for row in rows if row.get(key) is not None]
    return float(np.mean(values)) if values else None


def _cluster_ci(rows: Sequence[Mapping[str, Any]], seed: int, draws: int = 2000) -> list[float] | None:
    grouped: dict[str, list[float]] = {}
    for row in rows:
        value = float(row["value"])
        if np.isfinite(value):
            grouped.setdefault(str(row["sequence"]), []).append(value)
    if not grouped:
        return None
    means = np.asarray([np.mean(grouped[key]) for key in sorted(grouped)], dtype=float)
    if len(means) == 1:
        return [float(means[0]), float(means[0])]
    rng = np.random.default_rng(seed)
    sample = means[rng.integers(0, len(means), size=(draws, len(means)))].mean(axis=1)
    return [float(np.quantile(sample, 0.025)), float(np.quantile(sample, 0.975))]


def _entropy(counts: Mapping[str, int]) -> float:
    total = float(sum(counts.values()))
    return float(-sum((count / total) * math.log(count / total) for count in counts.values() if count)) if total else 0.0


def run(*, input_path: Path = INDEX, output_path: Path = OUT) -> dict[str, Any]:
    source = json.loads(input_path.read_text(encoding="utf-8"))
    if source.get("status") != "PASS":
        raise ValueError("N32-C rollout index is not complete; Oracle must not consume a partial index")
    grouped: dict[str, dict[str, Any]] = {}
    for row in source.get("rows", []):
        grouped.setdefault(str(row["episode_id"]), {"sequence": str(row["sequence"]), "learning_split": str(row["learning_split"]), "policies": {}})["policies"][str(row["policy"])] = row
    if len(grouped) != 689 or any(set(item["policies"]) != set(POLICIES) for item in grouped.values()):
        raise ValueError("N32-C Oracle requires 689 episodes with exactly K0/K1/K2")
    if len(source.get("rows", [])) != 2067:
        raise ValueError("N32-C Oracle requires exactly 2067 policy rows")
    episode_rows = []
    for episode_id, item in sorted(grouped.items()):
        values = {}
        for policy in POLICIES:
            row = item["policies"][policy]
            policy_issues = policy_metric_issues(row, require_explicit_visible_status=True)
            if policy_issues:
                raise ValueError(f"invalid policy row in complete index: {episode_id} {policy}: {policy_issues}")
            h20 = row.get("metrics", {}).get("20", {})
            if not isinstance(h20, Mapping) or not visible_h20_status(h20, require_explicit_undefined=True)["valid"]:
                raise ValueError(f"invalid H20 semantic state in complete index: {episode_id} {policy}")
            values[policy] = {
                "h20_iou": _h20(row, "mean_box_iou_visible"),
                "h20_success": _h20(row, "success_at_0_5_visible"),
                "h20_missing": _h20(row, "missing_prediction_rate_visible"),
                "h20_area_drift": _h20(row, "mask_area_drift"),
                "reward": float(row["reward"]),
            }
        iou_defined = all(values[policy]["h20_iou"] is not None for policy in POLICIES)
        episode_rows.append({"episode_id": episode_id, "sequence": item["sequence"], "learning_split": item["learning_split"], "values": values,
                             "h20_iou_winner": max(POLICIES, key=lambda policy: (values[policy]["h20_iou"], -POLICIES.index(policy))) if iou_defined else None,
                             "reward_winner": max(POLICIES, key=lambda policy: (values[policy]["reward"], -POLICIES.index(policy)))})
    mean_iou = {policy: _mean_defined([row["values"][policy] for row in episode_rows], "h20_iou") for policy in POLICIES}
    mean_reward = {policy: float(np.mean([row["values"][policy]["reward"] for row in episode_rows])) for policy in POLICIES}
    best_fixed = max(POLICIES, key=lambda policy: (mean_reward[policy], -POLICIES.index(policy)))
    gains = [
        {"episode_id": row["episode_id"], "sequence": row["sequence"], "learning_split": row["learning_split"], "value": max(value["h20_iou"] for value in row["values"].values()) - row["values"][best_fixed]["h20_iou"]}
        for row in episode_rows
        if row["h20_iou_winner"] is not None
    ]
    reward_gains = [{"episode_id": row["episode_id"], "sequence": row["sequence"], "learning_split": row["learning_split"], "value": max(value["reward"] for value in row["values"].values()) - row["values"][best_fixed]["reward"]} for row in episode_rows]
    winner_counts = {policy: int(sum(row["h20_iou_winner"] == policy for row in episode_rows if row["h20_iou_winner"] is not None)) for policy in POLICIES}
    reward_winner_counts = {policy: int(sum(row["reward_winner"] == policy for row in episode_rows)) for policy in POLICIES}
    sequence_means: dict[str, dict[str, float]] = {}
    for sequence in sorted({row["sequence"] for row in episode_rows}):
        subset = [row for row in episode_rows if row["sequence"] == sequence]
        metric_subset = [row for row in subset if row["h20_iou_winner"] is not None]
        metric_gains = [row["value"] for row in gains if row["sequence"] == sequence]
        sequence_means[sequence] = {
            "episode_count": len(subset),
            "defined_h20_iou_episode_count": len(metric_subset),
            "best_fixed_h20_iou": _mean_defined([row["values"][best_fixed] for row in metric_subset], "h20_iou"),
            "oracle_h20_iou": float(np.mean([max(value["h20_iou"] for value in row["values"].values()) for row in metric_subset])) if metric_subset else None,
            "oracle_minus_best_fixed_h20_iou": float(np.mean(metric_gains)) if metric_gains else None,
            "best_fixed_reward": float(np.mean([row["values"][best_fixed]["reward"] for row in subset])),
            "oracle_reward": float(np.mean([max(value["reward"] for value in row["values"].values()) for row in subset])),
        }
    subset_gates = {}
    for split in ("selection", "calibration"):
        subset = [row for row in episode_rows if row["learning_split"] == split]
        metric_gains = [row["value"] for row in gains if row["learning_split"] == split]
        reward_split_gains = [row["value"] for row in reward_gains if row["learning_split"] == split]
        gap = float(np.mean(metric_gains)) if metric_gains else None
        reward_gap = float(np.mean(reward_split_gains)) if reward_split_gains else None
        subset_gates[split] = {"episode_count": len(subset), "defined_h20_iou_episode_count": len(metric_gains), "h20_iou_oracle_gap": gap, "reward_oracle_gap": reward_gap, "positive_h20_direction": bool(gap is not None and gap > 0.0)}
    ci = _cluster_ci(gains, 320689)
    reward_ci = _cluster_ci(reward_gains, 320690)
    primary_gain = float(np.mean([row["value"] for row in gains])) if gains else None
    positive_sequences = int(sum(value["oracle_minus_best_fixed_h20_iou"] is not None and value["oracle_minus_best_fixed_h20_iou"] > 0.0 for value in sequence_means.values()))
    iou_winner_denominator = len(gains)
    oracle_iou_values = [max(value["h20_iou"] for value in row["values"].values()) for row in episode_rows if row["h20_iou_winner"] is not None]
    oracle_reward_values = [max(value["reward"] for value in row["values"].values()) for row in episode_rows]
    gate_checks = {
        "h20_iou_gain": primary_gain is not None and primary_gain >= 0.01,
        "sequence_cluster_ci_lower_gt_zero": ci is not None and float(ci[0]) > 0.0,
        "winner_not_best_fixed_rate": iou_winner_denominator > 0 and sum(winner_counts[p] for p in POLICIES if p != best_fixed) / iou_winner_denominator >= 0.30,
        "positive_sequence_count": positive_sequences >= 5,
        "selection_oracle_gap_direction_positive": bool(subset_gates["selection"]["positive_h20_direction"]),
        "calibration_oracle_gap_direction_positive": bool(subset_gates["calibration"]["positive_h20_direction"]),
    }
    result = {
        "protocol": "N32-C-POLICY-ORACLE-689",
        "status": "PASS" if all(gate_checks.values()) else "FAIL",
        "source_artifact": str(input_path),
        "source_sha256": _sha(input_path),
        "episode_count": len(episode_rows),
        "policy_order": list(POLICIES),
        "best_fixed_policy": best_fixed,
        "mean_h20_iou": mean_iou,
        "mean_reward": mean_reward,
        "h20_iou_defined_episode_count": len(gains),
        "h20_iou_undefined_episode_count": len(episode_rows) - len(gains),
        "h20_iou_defined_policy_row_count": {policy: int(sum(row["values"][policy]["h20_iou"] is not None for row in episode_rows)) for policy in POLICIES},
        "h20_iou_oracle": float(np.mean(oracle_iou_values)) if oracle_iou_values else None,
        "h20_iou_oracle_gain_vs_best_fixed": primary_gain,
        "reward_oracle": float(np.mean(oracle_reward_values)),
        "reward_oracle_gain_vs_best_fixed": float(np.mean([row["value"] for row in reward_gains])),
        "h20_iou_oracle_sequence_cluster_ci95": ci,
        "reward_oracle_sequence_cluster_ci95": reward_ci,
        "h20_iou_winner_counts": winner_counts,
        "reward_winner_counts": reward_winner_counts,
        "winner_entropy_nats": _entropy(winner_counts),
        "winner_entropy_normalized": _entropy(winner_counts) / math.log(len(POLICIES)),
        "winner_not_best_fixed_rate": sum(winner_counts[p] for p in POLICIES if p != best_fixed) / iou_winner_denominator if iou_winner_denominator else None,
        "positive_sequence_count": positive_sequences,
        "parent_sequence_count": len(sequence_means),
        "sequence_means": sequence_means,
        "selection_calibration_oracle": subset_gates,
        "gate_checks": gate_checks,
        "thresholds": {"h20_iou_gain": 0.01, "sequence_cluster_ci_lower": 0.0, "winner_not_best_fixed_rate": 0.30, "positive_sequence_count": 5},
        "episode_rows": episode_rows,
        "future_gt_used_for_posthoc_oracle": True,
        "future_gt_used_for_selection": False,
        "val25_read": False,
        "test_labels_used": False,
    }
    _write(output_path, result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=INDEX)
    parser.add_argument("--output", type=Path, default=OUT)
    args = parser.parse_args()
    result = run(input_path=args.input, output_path=args.output)
    print(json.dumps({key: result[key] for key in ("protocol", "status", "episode_count", "best_fixed_policy", "h20_iou_oracle_gain_vs_best_fixed", "winner_not_best_fixed_rate", "positive_sequence_count", "selection_calibration_oracle")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
