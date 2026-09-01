#!/usr/bin/env python3
"""N32-A: recompute the independent K0/K1/K2 policy Oracle from N31."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
N31 = ROOT / "outputs/n31/correction_state_ablation.json"
OUT = ROOT / "outputs/n32/policy_oracle_50.json"

POLICIES = {
    "K0_KEEP_OLD": "P0_no_correction_resume_control",
    "K1_APPLY_ENSURE": "P5_current_ensure_path",
    "K2_PROMPT_THEN_RESTORE": "P3_restore_old_state_after_prompt_failure",
}


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _metric(branch: Mapping[str, Any], key: str) -> float:
    value = branch.get("metrics", {}).get("20", {}).get(key)
    if value is None:
        return 0.0 if key == "missing_prediction_rate_visible" else float("nan")
    return float(value)


def _reward(branch: Mapping[str, Any]) -> float:
    metrics = branch.get("metrics", {}).get("20", {})
    iou = float(metrics.get("mean_box_iou_visible") or 0.0)
    missing = float(metrics.get("missing_prediction_rate_visible") if metrics.get("missing_prediction_rate_visible") is not None else 1.0)
    drift = float(metrics.get("mask_area_drift") if metrics.get("mask_area_drift") is not None else 0.0)
    regression = not bool(branch.get("protected_identity_namespace_unchanged", True))
    return float(iou - 0.50 * missing - 0.10 * float(regression) - 0.05 * drift)


def _cluster_ci(rows: Sequence[Mapping[str, Any]], *, value_key: str, seed: int, draws: int = 2000) -> list[float] | None:
    grouped: dict[str, list[float]] = {}
    for row in rows:
        grouped.setdefault(str(row["sequence"]), []).append(float(row[value_key]))
    if not grouped:
        return None
    values = np.asarray([np.mean(grouped[key]) for key in sorted(grouped)], dtype=float)
    if len(values) == 1:
        return [float(values[0]), float(values[0])]
    rng = np.random.default_rng(int(seed))
    sampled = values[rng.integers(0, len(values), size=(int(draws), len(values)))].mean(axis=1)
    return [float(np.quantile(sampled, 0.025)), float(np.quantile(sampled, 0.975))]


def _entropy(counts: Mapping[str, int]) -> float:
    total = float(sum(counts.values()))
    if total <= 0:
        return 0.0
    probs = [count / total for count in counts.values() if count]
    return float(-sum(p * math.log(p) for p in probs))


def run(*, input_path: Path = N31, output_path: Path = OUT) -> dict[str, Any]:
    source = json.loads(input_path.read_text(encoding="utf-8"))
    if source.get("status") != "PASS" or source.get("val25_read") is not False or source.get("test_labels_used") is not False:
        raise ValueError("N31 ablation is not a valid blind PASS source")
    episodes = source.get("episode_results", [])
    if len(episodes) != 50:
        raise ValueError(f"N32-A requires exactly 50 N31 episodes, got {len(episodes)}")

    rows: list[dict[str, Any]] = []
    for episode in episodes:
        values: dict[str, dict[str, float]] = {}
        for policy, branch_name in POLICIES.items():
            branch = episode.get("branches", {}).get(branch_name)
            if not isinstance(branch, Mapping):
                raise ValueError(f"missing {branch_name} in {episode.get('episode_id')}")
            h20 = episode["branches"][branch_name]["metrics"]["20"]
            values[policy] = {
                "h20_iou": float(h20["mean_box_iou_visible"]),
                "h20_success": float(h20["success_at_0_5_visible"]),
                "h20_missing": float(h20["missing_prediction_rate_visible"]),
                "h20_area_drift": float(h20.get("mask_area_drift") or 0.0),
                "reward": _reward(branch),
            }
        iou_winner = max(values, key=lambda policy: (values[policy]["h20_iou"], -list(POLICIES).index(policy)))
        reward_winner = max(values, key=lambda policy: (values[policy]["reward"], -list(POLICIES).index(policy)))
        rows.append({
            "episode_id": str(episode["episode_id"]),
            "sequence": str(episode["sequence"]),
            "split": str(episode.get("split", "train/train_fold")),
            "values": values,
            "h20_iou_winner": iou_winner,
            "reward_winner": reward_winner,
        })

    mean_iou = {policy: float(np.mean([row["values"][policy]["h20_iou"] for row in rows])) for policy in POLICIES}
    mean_reward = {policy: float(np.mean([row["values"][policy]["reward"] for row in rows])) for policy in POLICIES}
    best_fixed = max(POLICIES, key=lambda policy: (mean_reward[policy], -list(POLICIES).index(policy)))
    best_fixed_iou = max(POLICIES, key=lambda policy: (mean_iou[policy], -list(POLICIES).index(policy)))
    primary_gains = [{
        "episode_id": row["episode_id"],
        "sequence": row["sequence"],
        "value": float(max(item["h20_iou"] for item in row["values"].values()) - row["values"][best_fixed]["h20_iou"]),
    } for row in rows]
    reward_gains = [{
        "episode_id": row["episode_id"],
        "sequence": row["sequence"],
        "value": float(max(item["reward"] for item in row["values"].values()) - row["values"][best_fixed]["reward"]),
    } for row in rows]
    sequence_means: dict[str, dict[str, float]] = {}
    for sequence in sorted({row["sequence"] for row in rows}):
        subset = [row for row in rows if row["sequence"] == sequence]
        sequence_means[sequence] = {
            "best_fixed_h20_iou": float(np.mean([row["values"][best_fixed]["h20_iou"] for row in subset])),
            "oracle_h20_iou": float(np.mean([max(item["h20_iou"] for item in row["values"].values()) for row in subset])),
            "oracle_minus_best_fixed_h20_iou": float(np.mean([row["value"] for row in primary_gains if row["sequence"] == sequence])),
            "best_fixed_reward": float(np.mean([row["values"][best_fixed]["reward"] for row in subset])),
            "oracle_reward": float(np.mean([max(item["reward"] for item in row["values"].values()) for row in subset])),
        }
    loso = []
    for sequence in sorted(sequence_means):
        heldout = [row for row in primary_gains if row["sequence"] == sequence]
        train = [row for row in primary_gains if row["sequence"] != sequence]
        loso.append({
            "heldout_sequence": sequence,
            "heldout_episode_count": len(heldout),
            "heldout_oracle_gain": float(np.mean([row["value"] for row in heldout])),
            "training_oracle_gain": float(np.mean([row["value"] for row in train])) if train else None,
        })
    winners = {policy: int(sum(row["h20_iou_winner"] == policy for row in rows)) for policy in POLICIES}
    reward_winners = {policy: int(sum(row["reward_winner"] == policy for row in rows)) for policy in POLICIES}
    positive_sequences = int(sum(value["oracle_minus_best_fixed_h20_iou"] > 0.0 for value in sequence_means.values()))
    oracle_iou = float(np.mean([max(item["h20_iou"] for item in row["values"].values()) for row in rows]))
    oracle_reward = float(np.mean([max(item["reward"] for item in row["values"].values()) for row in rows]))
    gain = float(oracle_iou - mean_iou[best_fixed])
    reward_gain = float(oracle_reward - mean_reward[best_fixed])
    payload = {
        "protocol": "N32-A-POLICY-ORACLE-50",
        "status": "PASS" if gain >= 0.01 and (_cluster_ci(primary_gains, value_key="value", seed=32050) or [-1.0])[0] > 0.0 and sum(winners[p] for p in POLICIES if p != best_fixed) / 50.0 >= 0.30 and positive_sequences >= 5 else "FAIL",
        "source_artifact": str(input_path),
        "source_sha256": _sha256(input_path),
        "episode_count": len(rows),
        "policy_mapping": POLICIES,
        "best_fixed_policy": best_fixed,
        "best_fixed_policy_by_h20_iou": best_fixed_iou,
        "mean_h20_iou": mean_iou,
        "mean_reward": mean_reward,
        "h20_iou_oracle": oracle_iou,
        "h20_iou_oracle_gain_vs_best_fixed": gain,
        "reward_oracle": oracle_reward,
        "reward_oracle_gain_vs_best_fixed": reward_gain,
        "h20_iou_oracle_sequence_cluster_ci95": _cluster_ci(primary_gains, value_key="value", seed=32050),
        "reward_oracle_sequence_cluster_ci95": _cluster_ci(reward_gains, value_key="value", seed=32051),
        "h20_iou_winner_counts": winners,
        "reward_winner_counts": reward_winners,
        "winner_entropy_nats": _entropy(winners),
        "winner_entropy_normalized": float(_entropy(winners) / math.log(len(POLICIES))) if len(POLICIES) > 1 else 0.0,
        "winner_not_best_fixed_rate": float(sum(winners[p] for p in POLICIES if p != best_fixed) / len(rows)),
        "positive_sequence_count": positive_sequences,
        "sequence_means": sequence_means,
        "leave_one_sequence_out_oracle_gain": loso,
        "thresholds": {"h20_iou_gain": 0.01, "sequence_cluster_ci_lower": 0.0, "winner_not_best_fixed_rate": 0.30, "positive_sequence_count": 5},
        "gate_checks": {
            "h20_iou_gain": bool(gain >= 0.01),
            "sequence_cluster_ci_lower_gt_zero": bool((_cluster_ci(primary_gains, value_key="value", seed=32050) or [-1.0])[0] > 0.0),
            "winner_not_best_fixed_rate": bool(sum(winners[p] for p in POLICIES if p != best_fixed) / len(rows) >= 0.30),
            "positive_sequence_count": bool(positive_sequences >= 5),
        },
        "rows": rows,
        "future_gt_used_for_posthoc_oracle": True,
        "future_gt_used_for_selection": False,
        "val25_read": False,
        "test_labels_used": False,
    }
    _write(output_path, payload)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=N31)
    parser.add_argument("--output", type=Path, default=OUT)
    args = parser.parse_args()
    result = run(input_path=args.input, output_path=args.output)
    print(json.dumps({key: result[key] for key in ("protocol", "status", "best_fixed_policy", "h20_iou_oracle_gain_vs_best_fixed", "winner_not_best_fixed_rate", "positive_sequence_count")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
