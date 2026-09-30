#!/usr/bin/env python3
"""Aggregate the frozen N72R20 target-centric identity bridge records."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys
from typing import Any, Iterable

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
HORIZONS = (20, 50, 100)
TIME_BINS = ((1, 5, "1_5"), (6, 20, "6_20"), (21, 50, "21_50"), (51, 100, "51_100"))
METHODS = (
    "B0_HUMAN_ANCHOR_ONLY",
    "B1_EMA_0.90",
    "B2_FROZEN_N72R18_GRU",
    "ORACLE_CORRECT_UPDATE_OFFLINE_ORACLE_DIAGNOSTIC",
)
POLICIES = ("immediate", "2frame")
BOOTSTRAP_REPS = 2000
BOOTSTRAP_SEED = 72020


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object: {path}")
    return value


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise TypeError(f"expected JSON object at {path}:{line_number}")
            rows.append(value)
    return rows


def bootstrap_cluster_ci(values_by_cluster: dict[str, list[float]], *, reps: int = BOOTSTRAP_REPS) -> dict[str, Any]:
    names = sorted(values_by_cluster)
    if not names:
        return {"lower": None, "upper": None, "clusters": 0, "reps": reps, "seed": BOOTSTRAP_SEED}
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    cluster_means = np.asarray(
        [float(np.mean(values_by_cluster[name])) for name in names], dtype=np.float64
    )
    samples = rng.integers(0, len(cluster_means), size=(reps, len(cluster_means)))
    distribution = cluster_means[samples].mean(axis=1)
    return {
        "lower": float(np.percentile(distribution, 2.5)),
        "upper": float(np.percentile(distribution, 97.5)),
        "clusters": len(names),
        "reps": reps,
        "seed": BOOTSTRAP_SEED,
    }


def recovery_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    trajectories: dict[tuple[str, int, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if bool(row.get("target_visible", True)) and bool(row.get("identity_evaluable")):
            key = (str(row["sequence"]), int(row["track_id"]), int(row["anchor_frame"]))
            trajectories[key].append(row)
    failures = 0
    recovered = 0
    recovery_gaps: list[int] = []
    for trajectory in trajectories.values():
        ordered = sorted(trajectory, key=lambda row: int(row["frame"]))
        failed_at = next(
            (
                index
                for index, row in enumerate(ordered)
                if row.get("rank_1", int(row["rank"]) == 1) is False
            ),
            None,
        )
        if failed_at is None:
            continue
        failures += 1
        later = next(
            (
                row
                for row in ordered[failed_at + 1 :]
                if row.get("rank_1", int(row["rank"]) == 1) is True
            ),
            None,
        )
        if later is not None:
            recovered += 1
            recovery_gaps.append(int(later["frame"]) - int(ordered[failed_at]["frame"]))
    return {
        "trajectories_with_rank1_failure": failures,
        "trajectories_with_future_rank1_recovery": recovered,
        "future_identity_recovery_rate": recovered / failures if failures else None,
        "median_recovery_gap": float(np.median(recovery_gaps)) if recovery_gaps else None,
    }


def _summary(rows: list[dict[str, Any]], label: str) -> dict[str, Any]:
    target_visible = [row for row in rows if bool(row.get("target_visible", True))]
    visible = [row for row in target_visible if bool(row.get("candidate_available"))]
    covered = [row for row in target_visible if bool(row.get("target_candidate_available"))]
    evaluable = [row for row in target_visible if bool(row.get("identity_evaluable"))]
    margins = np.asarray([float(row["margin"]) for row in evaluable], dtype=np.float64)
    ranks = np.asarray([int(row["rank"]) for row in evaluable], dtype=np.int64)
    all_updates = [row for row in rows if bool(row.get("update_applied"))]
    updates = [row for row in target_visible if bool(row.get("update_applied"))]
    wrong_writes = [row for row in updates if row.get("wrong_memory_write_posthoc") is True]
    by_sequence: dict[str, list[float]] = defaultdict(list)
    for row in evaluable:
        by_sequence[str(row["sequence"])].append(float(bool(row["win"])))
    return {
        "label": label,
        "runtime_frames": len(rows),
        "target_visible_frames": len(target_visible),
        "candidate_available_frames": len(visible),
        "candidate_pool_availability": len(visible) / len(target_visible) if target_visible else None,
        "covered_target_frames": len(covered),
        "candidate_coverage": len(covered) / len(target_visible) if target_visible else None,
        "identity_evaluable_frames": len(evaluable),
        "hard_negative_identity_win_rate": float(np.mean([bool(row["win"]) for row in evaluable])) if evaluable else None,
        "hard_negative_identity_win_rate_sequence_cluster_ci95": bootstrap_cluster_ci(by_sequence),
        "rank_1_accuracy": float(np.mean(ranks == 1)) if len(ranks) else None,
        "rank_2_accuracy": float(np.mean(ranks <= 2)) if len(ranks) else None,
        "rank_3_accuracy": float(np.mean(ranks <= 3)) if len(ranks) else None,
        "mrr": float(np.mean(1.0 / ranks)) if len(ranks) else None,
        "mean_margin": float(margins.mean()) if len(margins) else None,
        "median_margin": float(np.median(margins)) if len(margins) else None,
        "p10_margin": float(np.percentile(margins, 10)) if len(margins) else None,
        "p25_margin": float(np.percentile(margins, 25)) if len(margins) else None,
        "p75_margin": float(np.percentile(margins, 75)) if len(margins) else None,
        "machine_updates": len(updates),
        "machine_updates_all_runtime_frames": len(all_updates),
        "wrong_memory_writes": len(wrong_writes),
        "wrong_memory_write_rate": len(wrong_writes) / len(updates) if updates else None,
        **recovery_summary(evaluable),
    }


def summarize_method(rows: list[dict[str, Any]], method: str) -> dict[str, Any]:
    method_rows = [row for row in rows if row.get("method") == method]
    horizons = {
        f"H{horizon}": _summary(
            [row for row in method_rows if int(row["gap"]) <= horizon],
            f"H{horizon}",
        )
        for horizon in HORIZONS
    }
    bins = {
        label: _summary(
            [row for row in method_rows if low <= int(row["gap"]) <= high],
            label,
        )
        for low, high, label in TIME_BINS
    }
    return {"method": method, "horizons": horizons, "time_gap_bins": bins}


def paired_delta(
    rows: list[dict[str, Any]],
    left_method: str,
    right_method: str,
    *,
    horizon: int,
    field: str,
) -> dict[str, Any]:
    keyed: dict[tuple[str, int, int, int], dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        if (
            int(row["gap"]) > horizon
            or not bool(row.get("target_visible", True))
            or not bool(row.get("identity_evaluable"))
        ):
            continue
        key = (str(row["sequence"]), int(row["track_id"]), int(row["anchor_frame"]), int(row["frame"]))
        keyed[key][str(row["method"])] = row
    by_sequence: dict[str, list[float]] = defaultdict(list)
    for key, values in keyed.items():
        if left_method not in values or right_method not in values:
            continue
        left = values[left_method].get(field)
        right = values[right_method].get(field)
        if left is None or right is None:
            continue
        by_sequence[key[0]].append(float(bool(left)) - float(bool(right)))
    all_values = [value for values in by_sequence.values() for value in values]
    return {
        "left": left_method,
        "right": right_method,
        "field": field,
        "horizon": horizon,
        "paired_frames": len(all_values),
        "paired_sequence_count": len(by_sequence),
        "delta_left_minus_right": float(np.mean(all_values)) if all_values else None,
        "sequence_cluster_ci95": bootstrap_cluster_ci(by_sequence),
    }


def paired_policy_rate_delta(
    immediate_rows: list[dict[str, Any]],
    confirmed_rows: list[dict[str, Any]],
    *,
    method: str,
    horizon: int,
    numerator_field: str,
) -> dict[str, Any]:
    by_policy: dict[str, dict[str, list[dict[str, Any]]]] = {"immediate": defaultdict(list), "2frame": defaultdict(list)}
    for policy, rows in (("immediate", immediate_rows), ("2frame", confirmed_rows)):
        for row in rows:
            if (
                row.get("method") == method
                and int(row["gap"]) <= horizon
                and bool(row.get("target_visible", True))
                and bool(row.get("update_applied"))
                and row.get(numerator_field) is not None
            ):
                by_policy[policy][str(row["sequence"])].append(row)
    by_sequence: dict[str, list[float]] = defaultdict(list)
    for sequence in sorted(set(by_policy["immediate"]) & set(by_policy["2frame"])):
        immediate_values = by_policy["immediate"][sequence]
        confirmed_values = by_policy["2frame"][sequence]
        immediate_rate = float(np.mean([bool(row[numerator_field]) for row in immediate_values]))
        confirmed_rate = float(np.mean([bool(row[numerator_field]) for row in confirmed_values]))
        by_sequence[sequence].append(immediate_rate - confirmed_rate)
    all_values = [value for values in by_sequence.values() for value in values]
    return {
        "method": method,
        "horizon": horizon,
        "numerator_field": numerator_field,
        "delta_immediate_minus_2frame": float(np.mean(all_values)) if all_values else None,
        "sequence_cluster_ci95": bootstrap_cluster_ci(by_sequence),
    }


def _load_policy_records(records_dir: Path, sequences: Iterable[str], policy: str) -> tuple[list[dict[str, Any]], list[str]]:
    rows: list[dict[str, Any]] = []
    missing: list[str] = []
    for sequence in sequences:
        path = records_dir / f"{sequence}.{policy}.jsonl"
        if not path.is_file():
            missing.append(sequence)
            continue
        rows.extend(load_jsonl(path))
    return rows, missing


def decide(
    *,
    complete: bool,
    immediate: dict[str, Any],
    confirmed: dict[str, Any],
    paired: dict[str, Any],
    contamination_delta: dict[str, Any],
) -> tuple[str | None, str]:
    if not complete:
        return None, "required val sequence/policy records are incomplete"
    coverage = immediate["B0_HUMAN_ANCHOR_ONLY"]["horizons"]["H100"]["candidate_coverage"]
    if coverage is None:
        return None, "H100 candidate coverage is unavailable"
    if coverage < 0.80:
        return "FAIL_CANDIDATE_COVERAGE", "H100 target-visible candidate coverage is below the frozen 80% gate"

    b2 = immediate["B2_FROZEN_N72R18_GRU"]["horizons"]["H100"]
    b0 = immediate["B0_HUMAN_ANCHOR_ONLY"]["horizons"]["H100"]
    delta = paired["delta_left_minus_right"]
    ci_lower = paired["sequence_cluster_ci95"]["lower"]
    identity_gain = (
        delta is not None
        and ci_lower is not None
        and delta > 0.0
        and ci_lower > 0.0
        and b2["rank_1_accuracy"] is not None
        and b0["rank_1_accuracy"] is not None
        and b2["rank_1_accuracy"] - b0["rank_1_accuracy"] >= 0.05
    )

    immediate_b2 = immediate["B2_FROZEN_N72R18_GRU"]["horizons"]["H100"]
    early_b2 = immediate["B2_FROZEN_N72R18_GRU"]["horizons"]["H20"]
    confirmed_b2 = confirmed["B2_FROZEN_N72R18_GRU"]["horizons"]["H100"]
    immediate_wrong = immediate_b2["wrong_memory_write_rate"]
    confirmed_wrong = confirmed_b2["wrong_memory_write_rate"]
    immediate_rank = immediate_b2["rank_1_accuracy"]
    confirmed_rank = confirmed_b2["rank_1_accuracy"]
    contamination_delta_value = contamination_delta["delta_immediate_minus_2frame"]
    contamination_ci_lower = contamination_delta["sequence_cluster_ci95"]["lower"]
    contamination_improved = (
        immediate_wrong is not None
        and confirmed_wrong is not None
        and confirmed_wrong < immediate_wrong
        and contamination_delta_value is not None
        and contamination_ci_lower is not None
        and contamination_ci_lower > 0.0
        and immediate_rank is not None
        and confirmed_rank is not None
        and confirmed_rank >= immediate_rank - 0.05
    )
    if identity_gain and contamination_improved:
        return "PASS_REAL_CANDIDATE_IDENTITY_MEMORY", "all frozen N72R20 gates passed"
    if identity_gain and not contamination_improved:
        return "FAIL_MEMORY_CONTAMINATION", "identity gain exists but the 2-frame contamination gate failed"
    if (
        early_b2["rank_1_accuracy"] is not None
        and immediate_rank is not None
        and immediate_rank < early_b2["rank_1_accuracy"] - 0.05
    ):
        return "FAIL_MEMORY_CONTAMINATION", "the learned memory loses rank-1 performance over the horizon"
    return "FAIL_IDENTITY_DISCRIMINATION", "frozen GRU did not establish the required identity gain over the anchor baseline"


def run(args: argparse.Namespace) -> dict[str, Any]:
    manifest = load_json(Path(args.asset_manifest).expanduser().resolve())
    protocol = load_json(Path(args.protocol).expanduser().resolve())
    if manifest.get("stage") != "N72R20" or protocol.get("stage") != "N72R20":
        raise ValueError("asset manifest and protocol must both be N72R20")
    sequences = list(protocol["val_evaluation_sequences"] if args.split == "val" else protocol["train_dev_sequences"])
    policies = list(POLICIES if args.require_two_policies else ("immediate",))
    records_dir = Path(args.records_dir).expanduser().resolve()
    loaded: dict[str, list[dict[str, Any]]] = {}
    missing: dict[str, list[str]] = {}
    summaries: dict[str, dict[str, dict[str, Any]]] = {}
    for policy in policies:
        rows, policy_missing = _load_policy_records(records_dir, sequences, policy)
        loaded[policy] = rows
        missing[policy] = policy_missing
        summaries[policy] = {method: summarize_method(rows, method) for method in METHODS}
    paired = paired_delta(
        loaded.get("immediate", []),
        "B2_FROZEN_N72R18_GRU",
        "B0_HUMAN_ANCHOR_ONLY",
        horizon=100,
        field="rank_1",
    )
    contamination_delta = paired_policy_rate_delta(
        loaded.get("immediate", []),
        loaded.get("2frame", []),
        method="B2_FROZEN_N72R18_GRU",
        horizon=100,
        numerator_field="wrong_memory_write_posthoc",
    )
    complete = not any(missing.values()) and all(
        summaries[policy]["B2_FROZEN_N72R18_GRU"]["horizons"]["H100"]["target_visible_frames"] > 0
        for policy in policies
    )
    decision = None
    decision_reason = ""
    if args.require_two_policies:
        decision, decision_reason = decide(
            complete=complete,
            immediate=summaries.get("immediate", {}),
            confirmed=summaries.get("2frame", {}),
            paired=paired,
            contamination_delta=contamination_delta,
        )
    result = {
        "stage": "N72R20",
        "status": "FINAL_N72R20_DECISION" if decision is not None else "INCOMPLETE_N72R20_IDENTITY_BRIDGE_AGGREGATION",
        "decision": decision,
        "decision_reason": decision_reason,
        "next_interactive_mot_stage_authorized": decision == "PASS_REAL_CANDIDATE_IDENTITY_MEMORY",
        "split": args.split,
        "sequences": sequences,
        "sequence_count": len(sequences),
        "policies": policies,
        "missing_records": missing,
        "methods": summaries,
        "primary_endpoint": "VAL_H100_HARD_NEGATIVE_IDENTITY_WIN_RATE",
        "paired_b2_vs_b0_h100_rank1": paired,
        "paired_b2_immediate_vs_2frame_wrong_write_rate": contamination_delta,
        "candidate_coverage_gate": 0.80,
        "practical_gain_gate": 0.05,
        "confirmed_rank1_drop_tolerance": 0.05,
        "bootstrap": {"unit": "sequence_cluster", "replicates": BOOTSTRAP_REPS, "seed": BOOTSTRAP_SEED},
        "runtime_future_gt_used": False,
        "runtime_gt_read": False,
        "posthoc_gt_used": True,
        "val_tuning_used": False,
        "records_dir": str(records_dir),
    }
    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "decision": decision, "output": str(output)}, ensure_ascii=False))
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=("train", "val"), default="val")
    parser.add_argument("--records-dir", type=Path, required=True)
    parser.add_argument("--asset-manifest", type=Path, default=ROOT / "outputs/N72R20/asset_manifest.json")
    parser.add_argument("--protocol", type=Path, default=ROOT / "outputs/N72R20/protocol.json")
    parser.add_argument("--require-two-policies", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        run(args)
    except Exception as exc:
        print(f"N72R20 identity bridge aggregation failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
