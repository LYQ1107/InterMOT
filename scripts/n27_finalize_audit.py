#!/usr/bin/env python3
"""Create N27's final reproducibility, bootstrap, and grouped metric artifacts."""

from __future__ import annotations

import csv
import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(".")
OUT = ROOT / "outputs/n27"
DATA = OUT / "data"
REPS = 2000
SEED = 27


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as payload:
        return {key: payload[key].copy() for key in payload.files}


def select(scores: np.ndarray, mask: np.ndarray) -> np.ndarray:
    masked = np.where(mask, scores, -1.0e9)
    result = np.argmax(masked, axis=1).astype(np.int64)
    result[~mask.any(axis=1)] = 5
    return result


def atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = sorted({key for row in rows for key in row})
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def group_summary(values: np.ndarray, groups: np.ndarray, *, seed: int = SEED, repetitions: int = REPS) -> dict[str, Any]:
    values = np.asarray(values, dtype=np.float64)
    groups = np.asarray(groups)
    finite = np.isfinite(values)
    values = values[finite]
    groups = groups[finite]
    unique = np.unique(groups)
    point = float(values.mean()) if len(values) else None
    group_values = [values[groups == group] for group in unique]
    group_means = {str(group): float(chunk.mean()) for group, chunk in zip(unique, group_values) if len(chunk)}
    result: dict[str, Any] = {
        "status": "OK" if len(unique) >= 2 and len(values) else "NOT_COMPUTABLE",
        "groups": int(len(unique)),
        "events": int(len(values)),
        "point": point,
        "group_means": group_means,
        "majority_group_positive": bool(sum(value > 0.0 for value in group_means.values()) > len(group_means) / 2) if group_means else False,
    }
    if len(unique) < 2 or not len(values):
        result["ci95"] = None
        return result
    rng = np.random.default_rng(seed)
    draws = np.empty(repetitions, dtype=np.float64)
    for index in range(repetitions):
        sampled = rng.integers(0, len(group_values), size=len(group_values))
        numerator = sum(float(values_chunk.sum()) for values_chunk in (group_values[item] for item in sampled))
        denominator = sum(len(group_values[item]) for item in sampled)
        draws[index] = numerator / denominator
    result["ci95"] = [float(value) for value in np.quantile(draws, [0.025, 0.975])]
    return result


def masked_softmax(scores: np.ndarray, mask: np.ndarray) -> np.ndarray:
    values = np.asarray(scores, dtype=np.float64).copy()
    values[~mask] = -1.0e4
    values -= values.max(axis=1, keepdims=True)
    probabilities = np.exp(values) * mask
    probabilities /= np.maximum(probabilities.sum(axis=1, keepdims=True), 1.0e-12)
    return probabilities


def response_metrics(arrays: dict[str, np.ndarray]) -> tuple[np.ndarray, dict[str, Any]]:
    target = arrays["target"].astype(np.int64)
    mask = arrays["candidate_mask"].astype(bool)
    present = target < 5
    rejected = arrays["rejected_index"].astype(np.int64)
    pair = arrays["pair_valid"].astype(bool) & present & (rejected >= 0) & (rejected < 5) & (rejected != target)
    current = arrays["apcr_score"].astype(np.float64)
    counterfactual = arrays["cf_apcr_score"].astype(np.float64)
    probability = masked_softmax(current, mask)
    cf_probability = masked_softmax(counterfactual, mask)
    selected = arrays["selected_apcr"].astype(np.int64)
    cf_selected = select(counterfactual, mask)
    if not pair.any():
        return pair, {"pair_events": 0, "target_probability_gain": None, "rejected_selection_delta": None, "target_selection_delta": None}
    safe_target = np.clip(target, 0, 4)
    target_gain = probability[np.arange(len(target)), safe_target] - cf_probability[np.arange(len(target)), safe_target]
    rejected_delta = (selected == rejected).astype(np.float64) - (cf_selected == rejected).astype(np.float64)
    target_delta = (selected == target).astype(np.float64) - (cf_selected == target).astype(np.float64)
    return pair, {
        "pair_events": int(pair.sum()),
        "target_probability_gain": float(target_gain[pair].mean()),
        "rejected_selection_delta": float(rejected_delta[pair].mean()),
        "target_selection_delta": float(target_delta[pair].mean()),
    }


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def role_dataset_name(role: str, dataset_id: int) -> str:
    if "external" in role:
        return {0: "BDD100K", 1: "KITTI", 2: "MOT17", 3: "MOT20"}.get(dataset_id, f"dataset_{dataset_id}")
    if "dance" in role:
        return "DanceTrack"
    return "historical_cal10"


def grouped_rows(role: str, arrays: dict[str, np.ndarray], level: str) -> list[dict[str, Any]]:
    target = arrays["target"].astype(np.int64)
    present = target < 5
    if level == "dataset":
        keys = arrays["dataset"].astype(np.int64)
    else:
        keys = arrays["sequence"].astype(np.int64)
    rows: list[dict[str, Any]] = []
    for key in np.unique(keys):
        chosen = keys == key
        eval_mask = chosen & present
        rows.append({
            "role": role,
            "level": level,
            "dataset_id": int(key) if level == "dataset" else "",
            "dataset": role_dataset_name(role, int(key)) if level == "dataset" else "",
            "sequence_id": int(key) if level == "sequence" else "",
            "events": int(chosen.sum()),
            "candidate_present_events": int(eval_mask.sum()),
            "b10_top1": float(arrays["b10_correct"][eval_mask].mean()) if eval_mask.any() else None,
            "apcr_top1": float(arrays["apcr_correct"][eval_mask].mean()) if eval_mask.any() else None,
            "apcr_minus_b10_top1": float((arrays["apcr_correct"][eval_mask].astype(np.float64) - arrays["b10_correct"][eval_mask].astype(np.float64)).mean()) if eval_mask.any() else None,
            "b10_correction_events": int(arrays["b10_correction_event"][chosen].sum()),
            "apcr_correction_events": int(arrays["apcr_correction_event"][chosen].sum()),
            "pair_events": int((arrays["pair_valid"][chosen] & present[chosen]).sum()),
            "val25_read": False,
        })
    return rows


def main() -> None:
    roles = {
        "external_heldout": DATA / "apcr_rollout_external_heldout.npz",
        "dance_train_real_p2": DATA / "apcr_rollout_dance_train.npz",
        "historical_cal10_ranking_only": DATA / "apcr_rollout_cal10_ranking_only.npz",
    }
    arrays_by_role = {role: load_npz(path) for role, path in roles.items()}

    expected_external = load_npz(DATA / "external_heldout_b10_round0.npz")["selected"].astype(np.int64)
    expected_dance = load_npz(DATA / "dance_train_real_b10_round0.npz")["selected"].astype(np.int64)
    cal_parents = load_jsonl(ROOT / "outputs/n26/dense_dataset/round0_cal10_parents.jsonl")
    expected_cal = np.asarray([5 if int(row["round0_selected"]) < 0 else int(row["round0_selected"]) for row in cal_parents], dtype=np.int64)
    expected = {
        "external_heldout": expected_external,
        "dance_train_real_p2": expected_dance,
        "historical_cal10_ranking_only": expected_cal,
    }
    reproduction: dict[str, Any] = {"formula": "frozen B10 dynamic selection", "val25_read": False, "roles": {}}
    for role, arrays in arrays_by_role.items():
        actual = arrays["selected_b10"].astype(np.int64)
        reference = expected[role]
        mismatch = actual != reference
        reproduction["roles"][role] = {
            "events": int(len(actual)),
            "reference_events": int(len(reference)),
            "selection_mismatch_count": int(mismatch.sum()) if len(actual) == len(reference) else None,
            "selection_mismatch_rate": float(mismatch.mean()) if len(actual) == len(reference) else None,
            "status": "PASS" if len(actual) == len(reference) and not mismatch.any() else "FAIL",
            "ranking_only": role == "historical_cal10_ranking_only",
        }
    atomic_json(OUT / "b10_dynamic_reproduction.json", reproduction)

    bootstrap: dict[str, Any] = {
        "phase": "N27_FINAL_AUDIT",
        "seed": SEED,
        "repetitions": REPS,
        "metric_definition": "sequence/dataset cluster bootstrap resamples complete groups and recomputes pooled conditional-present top-1 or paired response mean",
        "val25_read": False,
        "roles": {},
    }
    for role, arrays in arrays_by_role.items():
        target = arrays["target"].astype(np.int64)
        present = target < 5
        difference = arrays["apcr_correct"].astype(np.float64) - arrays["b10_correct"].astype(np.float64)
        response_pair, response = response_metrics(arrays)
        role_result: dict[str, Any] = {
            "ranking_status": "RANKING_ONLY_NOT_A_SELECTION_RESULT" if "cal10" in role else "FINAL_DYNAMIC_ROLLOUT",
            "events": int(len(target)),
            "candidate_present_events": int(present.sum()),
            "b10_top1": float(arrays["b10_correct"][present].mean()),
            "apcr_top1": float(arrays["apcr_correct"][present].mean()),
            "apcr_minus_b10_top1": group_summary(difference[present], arrays["sequence"][present], seed=SEED),
            "sequence_group_count": int(len(np.unique(arrays["sequence"]))),
            "correction_response": {
                **response,
                "target_probability_gain_bootstrap": group_summary(
                    (masked_softmax(arrays["apcr_score"], arrays["candidate_mask"].astype(bool))[np.arange(len(target)), np.clip(target, 0, 4)] - masked_softmax(arrays["cf_apcr_score"], arrays["candidate_mask"].astype(bool))[np.arange(len(target)), np.clip(target, 0, 4)])[response_pair],
                    arrays["sequence"][response_pair],
                    seed=SEED + 1,
                ) if response_pair.any() else {"status": "NOT_COMPUTABLE"},
            },
        }
        # Add the two paired selection deltas with independently seeded group resamples.
        if response_pair.any():
            rejected = arrays["rejected_index"].astype(np.int64)
            selected = arrays["selected_apcr"].astype(np.int64)
            cf_selected = select(arrays["cf_apcr_score"].astype(np.float64), arrays["candidate_mask"].astype(bool))
            rejected_values = ((selected == rejected).astype(np.float64) - (cf_selected == rejected).astype(np.float64))[response_pair]
            target_values = ((selected == target).astype(np.float64) - (cf_selected == target).astype(np.float64))[response_pair]
            response_result = role_result["correction_response"]
            response_result["rejected_selection_delta_bootstrap"] = group_summary(rejected_values, arrays["sequence"][response_pair], seed=SEED + 2)
            response_result["target_selection_delta_bootstrap"] = group_summary(target_values, arrays["sequence"][response_pair], seed=SEED + 3)
        role_result["dataset_bootstrap"] = group_summary(difference[present], arrays["dataset"][present], seed=SEED + 4)
        bootstrap["roles"][role] = role_result
    atomic_json(OUT / "bootstrap_results.json", bootstrap)

    dataset_rows: list[dict[str, Any]] = []
    sequence_rows: list[dict[str, Any]] = []
    for role, arrays in arrays_by_role.items():
        dataset_rows.extend(grouped_rows(role, arrays, "dataset"))
        sequence_rows.extend(grouped_rows(role, arrays, "sequence"))
    write_csv(OUT / "per_dataset_final.csv", dataset_rows)
    write_csv(OUT / "per_sequence_final.csv", sequence_rows)

    summary = {
        "b10_dynamic_reproduction": {role: value["status"] for role, value in reproduction["roles"].items()},
        "bootstrap_roles": list(bootstrap["roles"]),
        "per_dataset_rows": len(dataset_rows),
        "per_sequence_rows": len(sequence_rows),
        "val25_read": False,
    }
    atomic_json(OUT / "final_audit_summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
