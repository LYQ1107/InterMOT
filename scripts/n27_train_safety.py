#!/usr/bin/env python3
"""Freeze separate existence/commit safety heads before historical cal10."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import pickle
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(".")
OUT = ROOT / "outputs/n27"
DATA = OUT / "data"
SEED = 27
Z95 = 1.6448536269514722


def wilson_lower(successes: int, total: int) -> float:
    if total <= 0:
        return 0.0
    p = successes / total
    z2 = Z95 * Z95
    denominator = 1.0 + z2 / total
    center = p + z2 / (2.0 * total)
    radius = Z95 * math.sqrt(p * (1.0 - p) / total + z2 / (4.0 * total * total))
    return (center - radius) / denominator


def wilson_upper(successes: int, total: int) -> float:
    if total <= 0:
        return 1.0
    p = successes / total
    z2 = Z95 * Z95
    denominator = 1.0 + z2 / total
    center = p + z2 / (2.0 * total)
    radius = Z95 * math.sqrt(p * (1.0 - p) / total + z2 / (4.0 * total * total))
    return (center + radius) / denominator


def load_rollouts() -> dict[str, np.ndarray]:
    paths = [DATA / "apcr_rollout_external_heldout.npz", DATA / "apcr_rollout_dance_train.npz"]
    fields = ["candidate_mask", "target", "target_present", "selected_b10", "selected_apcr", "b10_correct", "apcr_correct", "b10_score", "apcr_score", "b10_margin", "b10_entropy", "apcr_margin", "apcr_entropy", "max_root_similarity", "candidate_count", "detector_score", "dataset", "sequence", "fold"]
    parts: dict[str, list[np.ndarray]] = {field: [] for field in fields}
    for path in paths:
        with np.load(path, allow_pickle=False) as payload:
            for field in fields:
                parts[field].append(payload[field].copy())
    result = {field: np.concatenate(values, axis=0) for field, values in parts.items()}
    result["group"] = np.asarray([f"{int(dataset)}:{int(sequence)}" for dataset, sequence in zip(result["dataset"], result["sequence"])], dtype="U32")
    return result


def design(arrays: dict[str, np.ndarray], method: str) -> np.ndarray:
    score = arrays[f"{method}_score"].astype(np.float32)
    mask = arrays["candidate_mask"].astype(bool)
    max_detector = np.max(np.where(mask, arrays["detector_score"], -1.0), axis=1)
    mean_detector = np.sum(np.where(mask, arrays["detector_score"], 0.0), axis=1) / np.maximum(mask.sum(axis=1), 1)
    top = np.max(np.where(mask, score, -1e4), axis=1)
    margin = arrays[f"{method}_margin"].astype(np.float32)
    entropy = arrays[f"{method}_entropy"].astype(np.float32)
    return np.column_stack([top, margin, entropy, arrays["max_root_similarity"].astype(np.float32), arrays["candidate_count"].astype(np.float32), max_detector, mean_detector])


def make_oof(x: np.ndarray, y_exist: np.ndarray, y_commit: np.ndarray, groups: np.ndarray) -> tuple[np.ndarray, np.ndarray, Pipeline, Pipeline, dict[str, Any]]:
    unique = np.unique(groups)
    group_to_fold = {group: index % min(5, len(unique)) for index, group in enumerate(sorted(unique))}
    folds = np.asarray([group_to_fold[group] for group in groups], dtype=np.int8)
    exist_oof = np.zeros(len(x), dtype=np.float32)
    commit_oof = np.zeros(len(x), dtype=np.float32)
    for fold in sorted(np.unique(folds)):
        train = folds != fold
        test = folds == fold
        exist_model = Pipeline([("scale", StandardScaler()), ("logit", LogisticRegression(C=1.0, max_iter=1000, class_weight="balanced", random_state=SEED))])
        commit_model = Pipeline([("scale", StandardScaler()), ("logit", LogisticRegression(C=1.0, max_iter=1000, class_weight="balanced", random_state=SEED))])
        exist_model.fit(x[train], y_exist[train])
        commit_model.fit(x[train], y_commit[train])
        exist_oof[test] = exist_model.predict_proba(x[test])[:, 1]
        commit_oof[test] = commit_model.predict_proba(x[test])[:, 1]
    full_exist = Pipeline([("scale", StandardScaler()), ("logit", LogisticRegression(C=1.0, max_iter=1000, class_weight="balanced", random_state=SEED))])
    full_commit = Pipeline([("scale", StandardScaler()), ("logit", LogisticRegression(C=1.0, max_iter=1000, class_weight="balanced", random_state=SEED))])
    full_exist.fit(x, y_exist)
    full_commit.fit(x, y_commit)
    return exist_oof, commit_oof, full_exist, full_commit, {"folds": int(len(np.unique(folds))), "groups": int(len(unique)), "group_to_fold": group_to_fold}


def bootstrap_risk_upper(correct: np.ndarray, commits: np.ndarray, groups: np.ndarray, seed: int = SEED, repetitions: int = 1000) -> float:
    unique = np.unique(groups)
    group_correct = np.asarray([int((correct & (groups == group)).sum()) for group in unique], dtype=np.float64)
    group_commits = np.asarray([int((commits & (groups == group)).sum()) for group in unique], dtype=np.float64)
    rng = np.random.default_rng(seed)
    sampled = rng.integers(0, len(unique), size=(repetitions, len(unique)))
    correct_sum = group_correct[sampled].sum(axis=1)
    commit_sum = group_commits[sampled].sum(axis=1)
    precision = np.divide(correct_sum, np.maximum(commit_sum, 1.0))
    risk = np.where(commit_sum > 0, 1.0 - precision, 1.0)
    return float(np.quantile(risk, 0.95, method="linear"))


def threshold_metrics(score: np.ndarray, selected: np.ndarray, target: np.ndarray, group: np.ndarray, threshold: float, bootstrap: bool) -> dict[str, Any]:
    commits = (score >= threshold) & (selected < 5)
    correct = commits & (selected == target)
    absent = target == 5
    absent_false = commits & absent
    commit_count = int(commits.sum())
    sequence_commits = {str(item): int((commits & (group == item)).sum()) for item in np.unique(group) if (commits & (group == item)).any()}
    sequence_count = len(sequence_commits)
    max_sequence_fraction = max(sequence_commits.values()) / max(1, commit_count) if sequence_commits else 1.0
    output = {
        "threshold": float(threshold), "events": len(target), "commits": commit_count, "correct_commits": int(correct.sum()),
        "commit_precision": float(correct.sum() / commit_count) if commit_count else 0.0,
        "precision_lower_wilson_95": wilson_lower(int(correct.sum()), commit_count),
        "coverage": float(commit_count / len(target)), "candidate_set_absent_events": int(absent.sum()),
        "absent_false_accepts": int(absent_false.sum()), "absent_false_accept": float(absent_false.sum() / max(1, absent.sum())),
        "absent_false_accept_upper_wilson_95": wilson_upper(int(absent_false.sum()), int(absent.sum())),
        "sequence_count": sequence_count, "max_sequence_commit_fraction": max_sequence_fraction,
        "not_reject_all": commit_count > 0,
    }
    output["sequence_bootstrap_risk_upper_95"] = bootstrap_risk_upper(correct, commits, group) if bootstrap and commit_count else 1.0
    output["sequence_commits"] = sequence_commits
    return output


def select_threshold(score: np.ndarray, selected: np.ndarray, target: np.ndarray, group: np.ndarray) -> tuple[float, dict[str, Any], list[dict[str, Any]]]:
    # Scan all exact score thresholds in one cumulative pass.  The earlier
    # implementation rebuilt string masks for every threshold, which made a
    # mathematically small scan unnecessarily quadratic.
    finite = np.isfinite(score)
    order = np.argsort(-np.where(finite, score, -np.inf), kind="mergesort")
    sorted_score = score[order]
    eligible = selected[order] < 5
    correct = eligible & (selected[order] == target[order])
    absent_false = eligible & (target[order] == 5)
    end_rows = np.flatnonzero(np.r_[sorted_score[:-1] != sorted_score[1:], True])
    unique_groups, group_index = np.unique(group, return_inverse=True)
    group_matrix = np.zeros((len(order), len(unique_groups)), dtype=np.int8)
    group_matrix[np.arange(len(order)), group_index[order]] = eligible.astype(np.int8)
    cumulative_groups = np.cumsum(group_matrix, axis=0)
    cumulative_commits = np.cumsum(eligible.astype(np.int64))
    cumulative_correct = np.cumsum(correct.astype(np.int64))
    cumulative_absent_false = np.cumsum(absent_false.astype(np.int64))
    point_feasible: list[tuple[float, dict[str, Any], int]] = []
    for end in end_rows:
        commits = int(cumulative_commits[end])
        if commits == 0 or commits / len(target) < 0.05:
            continue
        correct_count = int(cumulative_correct[end])
        absent_count = int(cumulative_absent_false[end])
        group_counts = cumulative_groups[end]
        sequence_count = int(np.count_nonzero(group_counts))
        max_sequence_fraction = float(group_counts.max() / commits) if commits else 1.0
        precision = correct_count / commits
        absent_rate = absent_count / max(1, int((target == 5).sum()))
        if precision < 0.90 or absent_rate > 0.0726 or sequence_count < 5 or max_sequence_fraction > 0.5:
            continue
        point_feasible.append((float(sorted_score[end]), {"coverage": commits / len(target), "commit_precision": precision, "absent_false_accept": absent_rate, "sequence_count": sequence_count, "max_sequence_commit_fraction": max_sequence_fraction}, int(end)))
    point_feasible.sort(key=lambda item: (-item[1]["coverage"], -item[1]["commit_precision"], -item[0]))
    records: list[dict[str, Any]] = []
    chosen: tuple[float, dict[str, Any]] | None = None
    for threshold, point, _ in point_feasible:
        metrics = threshold_metrics(score, selected, target, group, threshold, bootstrap=True)
        records.append(metrics)
        if metrics["precision_lower_wilson_95"] >= 0.90 and metrics["absent_false_accept_upper_wilson_95"] <= 0.0726 and metrics["sequence_bootstrap_risk_upper_95"] <= 0.10:
            chosen = (threshold, metrics)
            break
    if chosen is None:
        return math.inf, {"status": "NO_PRIMARY_GROUP_WILSON_BOOTSTRAP_THRESHOLD", "point_feasible_candidates": len(point_feasible)}, records
    return chosen[0], {"status": "PRIMARY_GROUP_WILSON_BOOTSTRAP_THRESHOLD", "point_feasible_candidates": len(point_feasible), **chosen[1]}, records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="outputs/n27/checkpoints/safety_heads.pkl")
    args = parser.parse_args()
    arrays = load_rollouts()
    heads: dict[str, Any] = {}
    gates: dict[str, Any] = {}
    risk_rows: list[dict[str, Any]] = []
    oof_payload: dict[str, np.ndarray] = {
        "target": arrays["target"].astype(np.int8), "dataset": arrays["dataset"].astype(np.int8),
        "sequence": arrays["sequence"].astype(np.int16),
    }
    for method in ("b10", "apcr"):
        x = design(arrays, method)
        target = arrays["target"].astype(np.int64)
        selected = arrays[f"selected_{method}"].astype(np.int64)
        y_exist = (target < 5).astype(np.int8)
        y_commit = (selected == target).astype(np.int8)
        exist_oof, commit_oof, exist_full, commit_full, cv = make_oof(x, y_exist, y_commit, arrays["group"])
        score = np.minimum(exist_oof, commit_oof)
        oof_payload[f"{method}_selection_score"] = score.astype(np.float32)
        threshold, gate, records = select_threshold(score, selected, target, arrays["group"])
        gates[method] = {"method": method.upper(), "selection_population": {"events": len(target), "external_heldout_events": 12000, "dance_train_events": 1500, "sequence_groups": int(len(np.unique(arrays["group"]))), "sequence_disjoint_oof": cv}, "head_features": ["top frozen policy score", "policy margin", "policy entropy", "max root similarity", "candidate count", "max detector score", "mean detector score"], "threshold": threshold, "calibration": {"name": "GROUP_WILSON_BOOTSTRAP_CERTIFICATE", "confidence": 0.95, "minimum_precision_lower_bound": 0.90, "maximum_absent_false_accept_upper_bound": 0.0726, "maximum_sequence_bootstrap_risk_upper_bound": 0.10, "minimum_source_sequences": 5, "maximum_single_sequence_commit_fraction": 0.5}, "result": gate, "threshold_search_records": records[:64], "threshold_frozen_before_cal10": True, "cal10_read": False, "val25_read": False}
        heads[method] = {"existence": exist_full, "commit": commit_full, "features": ["top", "margin", "entropy", "max_root", "candidate_count", "max_detector", "mean_detector"], "method": method, "threshold": threshold, "gate": gate}
        for record in records[:64]:
            risk_rows.append({"method": method, **{key: value for key, value in record.items() if key != "sequence_commits"}})
    output = ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    with temporary.open("wb") as handle:
        pickle.dump(heads, handle, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(temporary, output)
    oof_path = OUT / "safety_oof.npz"
    oof_temporary = oof_path.with_suffix(".npz.tmp")
    with oof_temporary.open("wb") as handle:
        np.savez_compressed(handle, **oof_payload)
    os.replace(oof_temporary, oof_path)
    for method in ("b10", "apcr"):
        path = OUT / f"{method}_safety_gate.json"
        temporary = path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(gates[method], indent=2, sort_keys=True) + "\n", encoding="utf-8")
        os.replace(temporary, path)
    risk_path = OUT / "risk_coverage.csv"
    fields = sorted({key for row in risk_rows for key in row})
    temporary = risk_path.with_suffix(".csv.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(risk_rows)
    os.replace(temporary, risk_path)
    combined = {"phase": "N27", "heads": str(output.relative_to(ROOT)), "methods": gates, "cal10_read": False, "val25_read": False}
    path = OUT / "b10_safety_gate.json"
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(combined, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)
    print(json.dumps(gates, indent=2, sort_keys=True), flush=True)
    print("N27_SAFETY_HEADS_FROZEN", flush=True)


if __name__ == "__main__":
    main()
