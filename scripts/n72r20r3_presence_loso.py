#!/usr/bin/env python3
"""Presence feature audit, B0-B4 calibration, and LOSO evaluation."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit

from sam3_intermot.association.identity_presence import _b3_score
from scripts.n72r20r3_common import ENCODER_SHA, R2_ASSET_ROOT, ROOT, SEQUENCES, read_zstd_jsonl


R2_C0_DENOMINATOR = 4889
MARGIN_GRID = (0.00, 0.02, 0.05, 0.10)
FEATURE_NAMES = (
    "learned_top1_score",
    "learned_top2_score",
    "learned_margin",
    "learned_top1_minus_mean",
    "learned_score_std",
    "human_anchor_top1_score",
    "learned_state_human_anchor_cosine",
    "candidate_count",
    "predicted_motion_iou_of_base_candidate",
    "frames_since_last_memory_write",
)
AUDIT_FEATURES = FEATURE_NAMES + (
    "learned_score_mean",
    "learned_score_median",
    "learned_score_min",
    "learned_score_max",
    "learned_top1_minus_median",
    "human_anchor_margin",
    "base_assignment_margin",
)


def _div(numerator: float, denominator: float) -> float | None:
    return None if denominator == 0 else float(numerator / denominator)


def _finite(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return float(default)
    return number if np.isfinite(number) else float(default)


def _metric_stats(values: Sequence[float]) -> dict[str, Any]:
    if not values:
        return {"count": 0, "mean": None, "median": None, "min": None, "max": None, "p10": None, "p25": None, "p75": None, "p90": None}
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": int(array.size),
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "min": float(np.min(array)),
        "max": float(np.max(array)),
        "p10": float(np.percentile(array, 10)),
        "p25": float(np.percentile(array, 25)),
        "p75": float(np.percentile(array, 75)),
        "p90": float(np.percentile(array, 90)),
    }


def _auc(scores: Sequence[float], labels: Sequence[bool]) -> float | None:
    positive = sum(bool(label) for label in labels)
    negative = len(labels) - positive
    if positive == 0 or negative == 0:
        return None
    order = sorted(range(len(scores)), key=lambda index: float(scores[index]))
    rank_sum = 0.0
    index = 0
    rank = 1
    while index < len(order):
        end = index + 1
        while end < len(order) and float(scores[order[end]]) == float(scores[order[index]]):
            end += 1
        average_rank = (rank + rank + (end - index) - 1) / 2.0
        rank_sum += sum(average_rank for position in order[index:end] if labels[position])
        rank += end - index
        index = end
    return float((rank_sum - positive * (positive + 1) / 2.0) / (positive * negative))


def _auprc(scores: Sequence[float], labels: Sequence[bool]) -> float | None:
    positive = sum(bool(label) for label in labels)
    if positive == 0:
        return None
    order = sorted(range(len(scores)), key=lambda index: -float(scores[index]))
    true_positive = 0
    false_positive = 0
    previous_recall = 0.0
    area = 0.0
    index = 0
    while index < len(order):
        end = index + 1
        while end < len(order) and float(scores[order[end]]) == float(scores[order[index]]):
            end += 1
        for position in order[index:end]:
            if labels[position]:
                true_positive += 1
            else:
                false_positive += 1
        recall = true_positive / positive
        precision = true_positive / max(1, true_positive + false_positive)
        area += (recall - previous_recall) * precision
        previous_recall = recall
        index = end
    return float(area)


def _load_joined(presence_dir: Path) -> list[dict[str, Any]]:
    runtime = read_zstd_jsonl(presence_dir / "frame_runtime_features.jsonl.zst")
    labels = read_zstd_jsonl(presence_dir / "frame_posthoc_labels.jsonl.zst")
    label_map = {(str(row["sequence"]), int(row["frame"]), str(row["state_condition"])): row for row in labels}
    joined: list[dict[str, Any]] = []
    for row in runtime:
        key = (str(row["sequence"]), int(row["frame"]), str(row["state_condition"]))
        if key not in label_map:
            raise ValueError(f"missing posthoc label for runtime key {key}")
        if row.get("posthoc_gt_used") is not False or row.get("runtime_future_gt_used") is not False:
            raise ValueError("runtime feature table contains forbidden provenance")
        joined.append({"runtime": dict(row), "label": dict(label_map[key])})
    if len(joined) != len(runtime) or len(labels) < len(runtime):
        raise ValueError("runtime/posthoc row count mismatch")
    return joined


def _condition_rows(joined: Sequence[Mapping[str, Any]], condition: str, sequences: Iterable[str] | None = None) -> list[dict[str, Any]]:
    allowed = None if sequences is None else {str(value) for value in sequences}
    return [dict(row) for row in joined if row["runtime"]["state_condition"] == condition and (allowed is None or str(row["runtime"]["sequence"]) in allowed)]


def _score_for_policy(runtime: Mapping[str, Any], policy: Mapping[str, Any]) -> float | None:
    method = str(policy.get("method"))
    if method == "B0_ALWAYS_PRESENT":
        return 1.0 if int(runtime.get("candidate_count", 0)) > 0 else -1.0
    if method == "B1_ABSOLUTE_SCORE":
        return _finite(runtime.get("learned_top1_score"), -np.inf)
    if method == "B2_ABSOLUTE_MARGIN":
        top = _finite(runtime.get("learned_top1_score"), -np.inf)
        margin = runtime.get("learned_margin")
        margin_value = float("inf") if margin is None else _finite(margin, -np.inf)
        return min(top - float(policy["tau_abs"]), margin_value - float(policy["tau_margin"]))
    if method == "B3_DISTRIBUTION":
        return _b3_score(runtime, str(policy["score_feature"]))
    if method == "LOGISTIC_PRESENCE":
        values = np.asarray([_finite(runtime.get(name), 0.0) for name in policy["feature_names"]], dtype=np.float64)
        means = np.asarray(policy["scaler_mean"], dtype=np.float64)
        scales = np.asarray(policy["scaler_scale"], dtype=np.float64)
        weights = np.asarray(policy["weights"], dtype=np.float64)
        if not np.isfinite(values).all():
            return None
        return float(expit(np.dot((values - means) / np.maximum(scales, 1.0e-8), weights) + float(policy["intercept"])))
    raise ValueError(f"unknown policy method {method}")


def _predict(joined: Sequence[Mapping[str, Any]], policy: Mapping[str, Any]) -> list[tuple[bool, str | None]]:
    result: list[tuple[bool, str | None]] = []
    for item in joined:
        runtime = item["runtime"]
        score = _score_for_policy(runtime, policy)
        method = str(policy["method"])
        if method == "B0_ALWAYS_PRESENT":
            present = int(runtime.get("candidate_count", 0)) > 0
        elif method == "B1_ABSOLUTE_SCORE":
            present = score is not None and score >= float(policy["tau_abs"])
        elif method == "B2_ABSOLUTE_MARGIN":
            top = _finite(runtime.get("learned_top1_score"), -np.inf)
            margin = runtime.get("learned_margin")
            margin_value = float("inf") if margin is None else _finite(margin, -np.inf)
            present = top >= float(policy["tau_abs"]) and margin_value >= float(policy["tau_margin"])
        else:
            threshold = float(policy.get("tau_distribution", policy.get("operating_threshold", 0.5)))
            present = score is not None and score >= threshold
        uid = runtime.get("learned_top1_candidate_uid") if present else None
        result.append((bool(present), None if uid in (None, "", "None") else str(uid)))
    return result


def metrics(joined: Sequence[Mapping[str, Any]], policy: Mapping[str, Any], sequence: str | None = None) -> dict[str, Any]:
    predictions = _predict(joined, policy)
    labels = [bool(item["label"]["candidate_set_present"]) for item in joined]
    negative = [not value for value in labels]
    positive = labels
    predicted = [item[0] for item in predictions]
    correct_identification = [
        predicted[index] and item["label"].get("learned_top1_target_iou") is not None and float(item["label"]["learned_top1_target_iou"]) >= 0.50
        for index, item in enumerate(joined)
    ]
    p0 = [item["label"]["taxonomy"] == "P0_TARGET_ABSENT" for item in joined]
    p1 = [str(item["label"]["taxonomy"]).startswith("P1") for item in joined]
    fpr_den = sum(negative)
    pos_den = sum(positive)
    pred_count = sum(predicted)
    accepted_writes = 0
    wrong_writes = 0
    correct_writes = 0
    r2_denominator = 0
    for index, item in enumerate(joined):
        label = item["label"]
        base_uid = label.get("base_candidate_uid")
        base_iou = label.get("base_candidate_target_iou")
        if base_uid not in (None, "", "None") and base_iou is not None and float(base_iou) >= 0.50:
            r2_denominator += 1
        eligible = predicted[index] and base_uid not in (None, "", "None") and base_uid == item["runtime"].get("learned_top1_candidate_uid")
        if eligible:
            accepted_writes += 1
            if base_iou is not None and float(base_iou) >= 0.50:
                correct_writes += 1
            else:
                wrong_writes += 1
    return {
        "method": str(policy.get("method")),
        "policy_id": str(policy.get("policy_id", policy.get("method"))),
        "sequence": sequence,
        "rows": len(joined),
        "candidate_set_negative_count": fpr_den,
        "candidate_set_positive_count": pos_den,
        "predicted_present_count": pred_count,
        "presence_precision": _div(sum(predicted[index] and positive[index] for index in range(len(joined))), pred_count),
        "presence_recall": _div(sum(predicted[index] and positive[index] for index in range(len(joined))), pos_den),
        "false_present_rate": _div(sum(predicted[index] and negative[index] for index in range(len(joined))), fpr_den),
        "false_absent_rate": _div(sum((not predicted[index]) and positive[index] for index in range(len(joined))), pos_den),
        "p0_false_present_rate": _div(sum(predicted[index] for index in range(len(joined)) if p0[index]), sum(p0)),
        "p1_false_present_rate": _div(sum(predicted[index] for index in range(len(joined)) if p1[index]), sum(p1)),
        "p0_count": sum(p0),
        "p1_count": sum(p1),
        "p0_false_present_count": sum(predicted[index] for index in range(len(joined)) if p0[index]),
        "p1_false_present_count": sum(predicted[index] for index in range(len(joined)) if p1[index]),
        "open_set_correct_identification_recall": _div(sum(correct_identification), pos_den),
        "conditional_candidate_accuracy_given_present": _div(sum(correct_identification), pred_count),
        "candidate_coverage": _div(sum(positive[index] for index in range(len(joined)) if bool(item["label"]["target_gt_present"])), sum(bool(item["label"]["target_gt_present"]) for item in joined)),
        "accepted_writes": accepted_writes,
        "wrong_writes": wrong_writes,
        "correct_writes": correct_writes,
        "wrong_write_rate": _div(wrong_writes, accepted_writes),
        "correct_write_retention_vs_r2_c0": _div(correct_writes, r2_denominator),
        "r2_c0_correct_write_denominator_in_rows": r2_denominator,
        "runtime_future_gt_used": False,
    }


def _threshold_candidates(values: Sequence[float]) -> list[float]:
    finite = sorted({float(value) for value in values if np.isfinite(float(value))})
    if not finite:
        return [0.0]
    # Exact unique-score scans are needlessly quadratic on correlated frame
    # tapes.  A fixed 51-quantile grid plus the strict-above-maximum sentinel
    # is a small, deterministic training-only calibration grid.
    if len(finite) > 64:
        quantiles = np.quantile(np.asarray(finite, dtype=np.float64), np.linspace(0.0, 1.0, 51))
        finite = sorted({float(value) for value in quantiles.tolist()})
    return finite + [float(finite[-1] + max(1.0e-6, abs(finite[-1]) * 1.0e-6))]


def _select_threshold(rows: Sequence[dict[str, Any]], score_fn, *, threshold_key: str) -> tuple[float, dict[str, Any]]:
    candidates = _threshold_candidates([score_fn(item["runtime"]) for item in rows if score_fn(item["runtime"]) is not None])
    best: tuple[tuple[float, float, float, float], float, dict[str, Any]] | None = None
    feasible = False
    for threshold in candidates:
        predicted = []
        for item in rows:
            score = score_fn(item["runtime"])
            predicted.append(score is not None and float(score) >= threshold)
        labels = [bool(item["label"]["candidate_set_present"]) for item in rows]
        negative = [not value for value in labels]
        positive = labels
        fpr = _div(sum(predicted[index] and negative[index] for index in range(len(rows))), sum(negative)) or 0.0
        open_recall = _div(
            sum(
                predicted[index]
                and positive[index]
                and item["label"].get("learned_top1_target_iou") is not None
                and float(item["label"]["learned_top1_target_iou"]) >= 0.50
                for index, item in enumerate(rows)
            ),
            sum(positive),
        ) or 0.0
        presence_recall = _div(sum(predicted[index] and positive[index] for index in range(len(rows))), sum(positive)) or 0.0
        if fpr <= 0.02:
            feasible = True
        key = (1.0 if fpr <= 0.02 else 0.0, open_recall, presence_recall, -fpr)
        if best is None or key > best[0]:
            best = (key, float(threshold), {"train_fpr": fpr, "train_open_recall": open_recall, "train_presence_recall": presence_recall})
    assert best is not None
    best[2]["train_gate_infeasible"] = not feasible
    best[2]["threshold_key"] = threshold_key
    return best[1], best[2]


def calibrate_policy(rows: Sequence[dict[str, Any]], method_id: str, condition: str) -> dict[str, Any]:
    if method_id == "B0_ALWAYS_PRESENT":
        return {"method": method_id, "policy_id": f"{condition}:{method_id}", "policy_version": "N72R20R3_B0_V1"}
    if method_id == "B1_ABSOLUTE_SCORE":
        threshold, audit = _select_threshold(rows, lambda runtime: _finite(runtime.get("learned_top1_score"), -np.inf), threshold_key="tau_abs")
        return {"method": method_id, "policy_id": f"{condition}:{method_id}", "policy_version": "N72R20R3_B1_V1", "tau_abs": threshold, "calibration": audit}
    if method_id == "B2_ABSOLUTE_MARGIN":
        best: tuple[tuple[float, float, float, float], float, float, dict[str, Any]] | None = None
        abs_values = _threshold_candidates([_finite(item["runtime"].get("learned_top1_score"), -np.inf) for item in rows])
        for tau_abs in abs_values:
            for tau_margin in MARGIN_GRID:
                def score_fn(runtime: Mapping[str, Any]) -> float:
                    top = _finite(runtime.get("learned_top1_score"), -np.inf)
                    margin = float("inf") if runtime.get("learned_margin") is None else _finite(runtime.get("learned_margin"), -np.inf)
                    return min(top - tau_abs, margin - tau_margin)
                _threshold = 0.0
                predicted = [score_fn(item["runtime"]) >= _threshold for item in rows]
                labels = [bool(item["label"]["candidate_set_present"]) for item in rows]
                negative = [not value for value in labels]
                positive = labels
                fpr = _div(sum(predicted[index] and negative[index] for index in range(len(rows))), sum(negative)) or 0.0
                open_recall = _div(sum(predicted[index] and positive[index] and item["label"].get("learned_top1_target_iou") is not None and float(item["label"]["learned_top1_target_iou"]) >= 0.50 for index, item in enumerate(rows)), sum(positive)) or 0.0
                presence_recall = _div(sum(predicted[index] and positive[index] for index in range(len(rows))), sum(positive)) or 0.0
                key = (1.0 if fpr <= 0.02 else 0.0, open_recall, presence_recall, -fpr)
                if best is None or key > best[0]:
                    best = (key, float(tau_abs), float(tau_margin), {"train_fpr": fpr, "train_open_recall": open_recall, "train_presence_recall": presence_recall, "train_gate_infeasible": fpr > 0.02})
        assert best is not None
        return {"method": method_id, "policy_id": f"{condition}:{method_id}", "policy_version": "N72R20R3_B2_V1", "tau_abs": best[1], "tau_margin": best[2], "calibration": best[3]}
    if method_id.startswith("B3_"):
        score_feature = method_id.removeprefix("B3_")
        threshold, audit = _select_threshold(rows, lambda runtime: _b3_score(runtime, score_feature), threshold_key="tau_distribution")
        return {"method": "B3_DISTRIBUTION", "policy_id": f"{condition}:{method_id}", "policy_version": "N72R20R3_B3_V1", "score_feature": score_feature, "tau_distribution": threshold, "calibration": audit}
    raise ValueError(f"unknown calibration method {method_id}")


def fit_logistic(rows: Sequence[dict[str, Any]], condition: str) -> dict[str, Any] | None:
    if not rows:
        return None
    x = np.asarray([[_finite(item["runtime"].get(name), 0.0) for name in FEATURE_NAMES] for item in rows], dtype=np.float64)
    y = np.asarray([bool(item["label"]["candidate_set_present"]) for item in rows], dtype=np.float64)
    if len(np.unique(y)) < 2:
        return None
    mean = np.mean(x, axis=0)
    scale = np.std(x, axis=0)
    scale[scale < 1.0e-8] = 1.0
    z = (x - mean) / scale
    l2 = 1.0e-3

    def objective(params: np.ndarray) -> tuple[float, np.ndarray]:
        intercept = params[0]
        weights = params[1:]
        logits = np.clip(intercept + z @ weights, -60.0, 60.0)
        probabilities = expit(logits)
        loss = -float(np.mean(y * np.log(np.maximum(probabilities, 1.0e-12)) + (1.0 - y) * np.log(np.maximum(1.0 - probabilities, 1.0e-12)))) + l2 * float(np.dot(weights, weights))
        residual = probabilities - y
        gradient = np.concatenate(([np.mean(residual)], (z.T @ residual) / len(y) + 2.0 * l2 * weights))
        return loss, gradient

    result = minimize(lambda params: objective(params), np.zeros(len(FEATURE_NAMES) + 1), jac=True, method="L-BFGS-B", options={"maxiter": 300, "ftol": 1.0e-12})
    if not result.success and not np.isfinite(result.fun):
        return None
    params = np.asarray(result.x, dtype=np.float64)
    probabilities = expit(np.clip(params[0] + z @ params[1:], -60.0, 60.0))
    calibration_rows = []
    for item, probability in zip(rows, probabilities.tolist()):
        copy = dict(item)
        copy["runtime"] = {**item["runtime"], "_b4_probability": float(probability)}
        calibration_rows.append(copy)
    threshold, audit = _select_threshold(calibration_rows, lambda runtime: float(runtime["_b4_probability"]), threshold_key="operating_threshold")
    return {
        "method": "LOGISTIC_PRESENCE",
        "policy_id": f"{condition}:B4_LOGISTIC_PRESENCE",
        "policy_version": "N72R20R3_B4_LOGISTIC_V1",
        "feature_names": list(FEATURE_NAMES),
        "scaler_mean": mean.tolist(),
        "scaler_scale": scale.tolist(),
        "weights": params[1:].tolist(),
        "intercept": float(params[0]),
        "operating_threshold": float(threshold),
        "calibration": audit,
        "trainable_parameters": int(len(FEATURE_NAMES) + 1),
    }


def feature_audit(joined: Sequence[dict[str, Any]], conditions: Sequence[str]) -> dict[str, Any]:
    output: dict[str, Any] = {"stage": "N72R20R3", "status": "PASS_N72R20R3_PRESENCE_FEATURE_AUDIT", "features": {}, "runtime_future_gt_used": False, "posthoc_gt_used": True}
    for condition in conditions:
        condition_rows = [item for item in joined if item["runtime"]["state_condition"] == condition]
        feature_output: dict[str, Any] = {}
        for feature in AUDIT_FEATURES:
            feature_record: dict[str, Any] = {}
            for group_name, selector in {
                "PRESENT": lambda item: bool(item["label"]["candidate_set_present"]),
                "ABSENT": lambda item: not bool(item["label"]["candidate_set_present"]),
                "P0_ONLY": lambda item: item["label"]["taxonomy"] == "P0_TARGET_ABSENT",
                "P1_ONLY": lambda item: str(item["label"]["taxonomy"]).startswith("P1"),
            }.items():
                values = [_finite(item["runtime"].get(feature), np.nan) for item in condition_rows if selector(item)]
                values = [float(value) for value in values if np.isfinite(value)]
                feature_record[group_name] = _metric_stats(values)
            scores = [_finite(item["runtime"].get(feature), np.nan) for item in condition_rows]
            labels = [bool(item["label"]["candidate_set_present"]) for item in condition_rows]
            valid = [index for index, value in enumerate(scores) if np.isfinite(value)]
            feature_record["auroc_present_vs_absent"] = _auc([float(scores[index]) for index in valid], [labels[index] for index in valid]) if valid else None
            feature_record["auprc_present_vs_absent"] = _auprc([float(scores[index]) for index in valid], [labels[index] for index in valid]) if valid else None
            per_sequence: dict[str, Any] = {}
            for sequence in SEQUENCES:
                selected = [item for item in condition_rows if str(item["runtime"]["sequence"]) == sequence]
                seq_scores = [_finite(item["runtime"].get(feature), np.nan) for item in selected]
                seq_labels = [bool(item["label"]["candidate_set_present"]) for item in selected]
                seq_valid = [index for index, value in enumerate(seq_scores) if np.isfinite(value)]
                per_sequence[sequence] = {
                    "auroc": _auc([float(seq_scores[index]) for index in seq_valid], [seq_labels[index] for index in seq_valid]) if seq_valid else None,
                    "present": _metric_stats([float(seq_scores[index]) for index in seq_valid if seq_labels[index]]),
                    "absent": _metric_stats([float(seq_scores[index]) for index in seq_valid if not seq_labels[index]]),
                }
            feature_record["per_sequence"] = per_sequence
            feature_output[feature] = feature_record
        output["features"][condition] = feature_output
    return output


def _evaluate_loso(joined: Sequence[dict[str, Any]], condition: str, method_ids: Sequence[str], include_b4: bool = False) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    fold_records: list[dict[str, Any]] = []
    fold_policies: dict[str, dict[str, Any]] = {}
    for heldout in SEQUENCES:
        calibration_sequences = [sequence for sequence in SEQUENCES if sequence != heldout]
        train = _condition_rows(joined, condition, calibration_sequences)
        test = _condition_rows(joined, condition, [heldout])
        for method_id in method_ids:
            policy = calibrate_policy(train, method_id, condition)
            policy = {**policy, "heldout_sequence": heldout, "calibration_sequences": list(calibration_sequences)}
            result = metrics(test, policy, sequence=heldout)
            result.update({"condition": condition, "heldout_sequence": heldout, "calibration": policy.get("calibration", {}), "trainable_parameters": 0})
            fold_records.append(result)
            fold_policies[f"{condition}|{heldout}|{policy['policy_id']}"] = policy
        if include_b4:
            policy = fit_logistic(train, condition)
            if policy is not None:
                policy = {**policy, "heldout_sequence": heldout, "calibration_sequences": list(calibration_sequences)}
                result = metrics(test, policy, sequence=heldout)
                result.update({"condition": condition, "heldout_sequence": heldout, "calibration": policy.get("calibration", {}), "trainable_parameters": policy.get("trainable_parameters", 0)})
                fold_records.append(result)
                fold_policies[f"{condition}|{heldout}|{policy['policy_id']}"] = policy
    return fold_records, fold_policies


def aggregate_loso(records: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for record in records:
        groups[(str(record["condition"]), str(record["policy_id"]))].append(record)
    output = []
    for (condition, policy_id), group in sorted(groups.items()):
        numeric_keys = [
            "false_present_rate",
            "p0_false_present_rate",
            "p1_false_present_rate",
            "presence_recall",
            "open_set_correct_identification_recall",
            "conditional_candidate_accuracy_given_present",
            "candidate_coverage",
            "wrong_write_rate",
            "correct_write_retention_vs_r2_c0",
        ]
        pooled: dict[str, Any] = {}
        total_rows = sum(int(item["rows"]) for item in group)
        for key in numeric_keys:
            if key in {"false_present_rate", "p0_false_present_rate", "p1_false_present_rate", "presence_recall", "open_set_correct_identification_recall", "conditional_candidate_accuracy_given_present", "candidate_coverage"}:
                if key == "false_present_rate":
                    numerator = sum((item.get("false_present_rate") or 0.0) * int(item["candidate_set_negative_count"]) for item in group)
                    denominator = sum(int(item["candidate_set_negative_count"]) for item in group)
                elif key == "presence_recall":
                    numerator = sum((item.get("presence_recall") or 0.0) * int(item["candidate_set_positive_count"]) for item in group)
                    denominator = sum(int(item["candidate_set_positive_count"]) for item in group)
                elif key == "open_set_correct_identification_recall":
                    numerator = sum((item.get(key) or 0.0) * int(item["candidate_set_positive_count"]) for item in group)
                    denominator = sum(int(item["candidate_set_positive_count"]) for item in group)
                elif key == "p0_false_present_rate":
                    numerator = sum(int(item.get("p0_false_present_count", 0)) for item in group)
                    denominator = sum(int(item.get("p0_count", 0)) for item in group)
                elif key == "p1_false_present_rate":
                    numerator = sum(int(item.get("p1_false_present_count", 0)) for item in group)
                    denominator = sum(int(item.get("p1_count", 0)) for item in group)
                else:
                    numerator = sum((item.get(key) or 0.0) * int(item["candidate_set_positive_count"]) for item in group)
                    denominator = sum(int(item["candidate_set_positive_count"]) for item in group)
                pooled[key] = _div(numerator, denominator)
            elif key == "wrong_write_rate":
                numerator = sum(int(item["wrong_writes"]) for item in group)
                denominator = sum(int(item["accepted_writes"]) for item in group)
                pooled[key] = _div(numerator, denominator)
            elif key == "correct_write_retention_vs_r2_c0":
                numerator = sum(int(item["correct_writes"]) for item in group)
                denominator = sum(int(item["r2_c0_correct_write_denominator_in_rows"]) for item in group)
                pooled[key] = _div(numerator, denominator)
            else:
                pooled[key] = _div(sum((item.get(key) or 0.0) * int(item["rows"]) for item in group), total_rows)
        macro = {key: _div(sum((item.get(key) or 0.0) for item in group), len(group)) for key in numeric_keys}
        per_sequence = {str(item["heldout_sequence"]): dict(item) for item in group}
        pooled_pass = (pooled["false_present_rate"] is not None and pooled["false_present_rate"] <= 0.02 and pooled["open_set_correct_identification_recall"] is not None and pooled["open_set_correct_identification_recall"] >= 0.60)
        cross_sequence_pass = all(
            item.get("false_present_rate") is not None and item["false_present_rate"] <= 0.02 and item.get("open_set_correct_identification_recall") is not None and item["open_set_correct_identification_recall"] >= 0.60
            for item in group
        )
        pooled["presence_gate_pass"] = bool(pooled_pass)
        pooled["cross_sequence_gate_pass"] = bool(cross_sequence_pass)
        output.append({"condition": condition, "policy_id": policy_id, "pooled": pooled, "macro": macro, "per_sequence": per_sequence, "trainable_parameters": int(group[0].get("trainable_parameters", 0))})
    return output


def choose_policy(aggregates: Sequence[Mapping[str, Any]]) -> dict[str, Any] | None:
    eligible = [item for item in aggregates if item["pooled"].get("presence_gate_pass") and item["pooled"].get("cross_sequence_gate_pass")]
    if not eligible:
        return None
    complexity = {"B0_ALWAYS_PRESENT": 0, "B1_ABSOLUTE_SCORE": 1, "B2_ABSOLUTE_MARGIN": 2, "B3_DISTRIBUTION": 3, "LOGISTIC_PRESENCE": 4}
    def key(item: Mapping[str, Any]) -> tuple[int, float, float]:
        method = str(item["policy_id"]).split(":", 1)[-1]
        if method.startswith("B3_"):
            method_key = "B3_DISTRIBUTION"
        elif method.startswith("B4_"):
            method_key = "LOGISTIC_PRESENCE"
        else:
            method_key = method
        return complexity.get(method_key, 99), -(item["pooled"].get("open_set_correct_identification_recall") or 0.0), item["pooled"].get("false_present_rate") or 1.0
    return dict(sorted(eligible, key=key)[0])


def sequence_cluster_bootstrap(records: Sequence[Mapping[str, Any]], reps: int = 2000, seed: int = 720203) -> list[dict[str, Any]]:
    """Bootstrap held-out sequences, never individual correlated frames."""

    grouped: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[(str(record["condition"]), str(record["policy_id"]))].append(record)
    rng = np.random.default_rng(seed)
    output: list[dict[str, Any]] = []
    for (condition, policy_id), group in sorted(grouped.items()):
        fpr = np.asarray([float(item.get("false_present_rate") or 0.0) for item in group], dtype=np.float64)
        open_recall = np.asarray([float(item.get("open_set_correct_identification_recall") or 0.0) for item in group], dtype=np.float64)
        if len(group) == 0:
            continue
        draws_fpr = np.empty(reps, dtype=np.float64)
        draws_open = np.empty(reps, dtype=np.float64)
        for index in range(reps):
            sample = rng.integers(0, len(group), size=len(group))
            draws_fpr[index] = float(np.mean(fpr[sample]))
            draws_open[index] = float(np.mean(open_recall[sample]))
        output.append(
            {
                "condition": condition,
                "policy_id": policy_id,
                "bootstrap_unit": "sequence",
                "repetitions": reps,
                "seed": seed,
                "macro_false_present_rate": float(np.mean(fpr)),
                "macro_open_set_correct_identification_recall": float(np.mean(open_recall)),
                "macro_false_present_rate_ci95": [float(np.quantile(draws_fpr, 0.025)), float(np.quantile(draws_fpr, 0.975))],
                "macro_open_set_correct_identification_recall_ci95": [float(np.quantile(draws_open, 0.025)), float(np.quantile(draws_open, 0.975))],
            }
        )
    return output


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--presence-dir", type=Path, default=ROOT / "outputs/N72R20R3/presence")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/N72R20R3")
    args = parser.parse_args()
    joined = _load_joined(args.presence_dir)
    conditions = sorted({str(item["runtime"]["state_condition"]) for item in joined})
    audit = feature_audit(joined, conditions)
    (args.presence_dir / "presence_feature_audit.json").write_text(json.dumps(audit, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")

    base_methods = ["B0_ALWAYS_PRESENT", "B1_ABSOLUTE_SCORE", "B2_ABSOLUTE_MARGIN", "B3_top1_minus_mean", "B3_top1_minus_median", "B3_top1_zscore"]
    base_records: list[dict[str, Any]] = []
    base_policies: dict[str, dict[str, Any]] = {}
    for condition in ("S0_HUMAN_ANCHOR_ONLY", "S1_R2_STYLE_CAUSAL_LEARNED_STATE"):
        records, policies = _evaluate_loso(joined, condition, base_methods, include_b4=False)
        base_records.extend(records)
        base_policies.update(policies)
    base_aggregates = aggregate_loso(base_records)
    b4_authorization = {
        "stage": "N72R20R3",
        "runtime_future_gt_used": False,
        "criteria": {
            "usable_separation": "at least one S0/S1 audit feature has pooled AUROC >= 0.60 and PRESENT/ABSENT distributions are non-empty",
            "simple_baselines_limited": "no B0-B3 LOSO method satisfies both pooled safety/usefulness and the all-eight-sequence gate",
        },
    }
    max_auroc = 0.0
    for condition in ("S0_HUMAN_ANCHOR_ONLY", "S1_R2_STYLE_CAUSAL_LEARNED_STATE"):
        for feature in audit["features"].get(condition, {}).values():
            if feature.get("auroc_present_vs_absent") is not None:
                max_auroc = max(max_auroc, float(feature["auroc_present_vs_absent"]))
    simple_pass = any(item["pooled"].get("presence_gate_pass") and item["pooled"].get("cross_sequence_gate_pass") for item in base_aggregates)
    b4_authorization["max_pooled_audit_auroc"] = max_auroc
    b4_authorization["usable_separation"] = bool(max_auroc >= 0.60)
    b4_authorization["simple_baselines_limited"] = not simple_pass
    b4_authorization["authorized"] = bool(b4_authorization["usable_separation"] and b4_authorization["simple_baselines_limited"])
    b4_authorization["reason"] = "Tiny logistic calibrator is authorized only when existing causal features show usable but cross-sequence-limited separation." if b4_authorization["authorized"] else "B4 is not authorized because the pre-registered evidence criteria were not both met."
    (args.output_dir / "B4_AUTHORIZATION.json").write_text(json.dumps(b4_authorization, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    records = list(base_records)
    policies = dict(base_policies)
    if b4_authorization["authorized"]:
        for condition in ("S0_HUMAN_ANCHOR_ONLY", "S1_R2_STYLE_CAUSAL_LEARNED_STATE"):
            b4_records, b4_policies = _evaluate_loso(joined, condition, [], include_b4=True)
            records.extend([item for item in b4_records if item["method"] == "LOGISTIC_PRESENCE"])
            policies.update({key: value for key, value in b4_policies.items() if value.get("method") == "LOGISTIC_PRESENCE"})
    aggregates = aggregate_loso(records)
    bootstrap = sequence_cluster_bootstrap(records)
    per_sequence = [item for record in records for item in [dict(record)]]
    baseline_results = {
        "stage": "N72R20R3",
        "methods": base_methods,
        "state_conditions": ["S0_HUMAN_ANCHOR_ONLY", "S1_R2_STYLE_CAUSAL_LEARNED_STATE"],
        "pooled_all_dev_no_loso": [
            metrics(_condition_rows(joined, condition), calibrate_policy(_condition_rows(joined, condition), method, condition))
            for condition in ("S0_HUMAN_ANCHOR_ONLY", "S1_R2_STYLE_CAUSAL_LEARNED_STATE")
            for method in base_methods
        ],
        "runtime_future_gt_used": False,
    }
    (args.presence_dir / "baseline_results.json").write_text(json.dumps(baseline_results, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    loso_payload = {
        "stage": "N72R20R3",
        "status": "PASS_N72R20R3_LOSO_COMPLETE",
        "fold_count": 8,
        "heldout_sequences": list(SEQUENCES),
        "records": records,
        "aggregates": aggregates,
        "runtime_future_gt_used": False,
        "posthoc_gt_used": True,
    }
    (args.presence_dir / "loso_results.json").write_text(json.dumps(loso_payload, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    (args.presence_dir / "bootstrap_results.json").write_text(json.dumps({"stage": "N72R20R3", "bootstrap_unit": "sequence", "repetitions": 2000, "seed": 720203, "results": bootstrap, "runtime_future_gt_used": False}, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    (args.presence_dir / "per_sequence_results.json").write_text(json.dumps({"stage": "N72R20R3", "records": per_sequence}, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    (args.presence_dir / "fold_policies.json").write_text(json.dumps({"stage": "N72R20R3", "policies": policies, "runtime_future_gt_used": False}, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    selected = choose_policy(aggregates)
    summary = {
        "status": "PASS_N72R20R3_BASELINES_LOSO",
        "fold_count": 8,
        "aggregate_count": len(aggregates),
        "b4_authorized": b4_authorization["authorized"],
        "selected_static_policy": None if selected is None else {"condition": selected["condition"], "policy_id": selected["policy_id"], "pooled": selected["pooled"]},
        "runtime_future_gt_used": False,
    }
    (args.output_dir / "presence_loso_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
