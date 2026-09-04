#!/usr/bin/env python3
"""Train the preregistered small candidate verifier for TVC_V1.

This is the only learned component in the N72R5R1 mechanism loop.  It is a
CPU-only logistic ranker over frozen appearance evidence.  It cannot change
candidate generation, SAM3, the Hungarian solver, or the evaluation labels.
The sequence split and optimizer are written before fitting and are never
selected from future-effect results.
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sam3_intermot.association.branch_public_replay import atomic_json, read_json, sha256_file  # noqa: E402


OUT = Path(os.environ.get("N72R5R1_RUN_ROOT", str(ROOT / "outputs/N72R5R1")))
INPUT_TABLE = Path(
    os.environ.get(
        "N72R5R1_ROUND03_TABLE",
        str(OUT / "controller" / "round_03_feature_separability_attempt2" / "feature_pair_table.jsonl"),
    )
)
ROUND_ROOT = Path(
    os.environ.get(
        "N72R5R1_ROUND04_ROOT",
        str(OUT / "controller" / "round_04_tvc_v1"),
    )
)
PROTOCOL = ROUND_ROOT / "round_04_protocol.json"
MODEL = ROUND_ROOT / "tvc_v1_model.json"
TRAINING = ROUND_ROOT / "training_result.json"
STATUS = ROUND_ROOT / "round_04_status.json"


def _sigmoid(value: np.ndarray) -> np.ndarray:
    clipped = np.clip(value, -60.0, 60.0)
    return 1.0 / (1.0 + np.exp(-clipped))


def _auc(scores: np.ndarray, labels: np.ndarray) -> float | None:
    positives = np.asarray(scores[labels == 1], dtype=np.float64)
    negatives = np.asarray(scores[labels == 0], dtype=np.float64)
    if positives.size == 0 or negatives.size == 0:
        return None
    wins = 0.0
    for value in positives:
        wins += float(np.sum(value > negatives)) + 0.5 * float(np.sum(value == negatives))
    return float(wins / (positives.size * negatives.size))


def _candidate_features(row: Mapping[str, Any], side: str) -> list[float]:
    anchor = row.get(f"{side}_anchor_cosine")
    prototype = row.get(f"{side}_prototype_cosine")
    if anchor is None or not math.isfinite(float(anchor)):
        raise ValueError(f"non-finite anchor feature: {row.get('event_id')}")
    return [
        float(anchor),
        0.0 if prototype is None else float(prototype),
        0.0 if prototype is None else 1.0,
    ]


def _build_examples(rows: Sequence[Mapping[str, Any]], allowed_sequences: set[str]) -> tuple[np.ndarray, np.ndarray, int]:
    features: list[list[float]] = []
    labels: list[int] = []
    pair_count = 0
    for row in rows:
        if str(row.get("sequence")) not in allowed_sequences:
            continue
        features.append(_candidate_features(row, "target"))
        labels.append(1)
        features.append(_candidate_features(row, "competitor"))
        labels.append(0)
        pair_count += 1
    if not features:
        raise ValueError("no training examples after sequence split")
    return np.asarray(features, dtype=np.float64), np.asarray(labels, dtype=np.int64), pair_count


def _standardize(train: np.ndarray, other: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = np.mean(train, axis=0)
    scale = np.std(train, axis=0)
    scale = np.where(scale > 1.0e-8, scale, 1.0)
    return (train - mean) / scale, (other - mean) / scale, np.stack([mean, scale], axis=0)


def _fit(train_x: np.ndarray, train_y: np.ndarray, protocol: Mapping[str, Any]) -> tuple[np.ndarray, float, list[dict[str, Any]]]:
    weights = np.zeros(train_x.shape[1], dtype=np.float64)
    bias = 0.0
    history: list[dict[str, Any]] = []
    lr = float(protocol["learning_rate"])
    l2 = float(protocol["l2"])
    n = float(train_x.shape[0])
    for epoch in range(int(protocol["epochs"])):
        logits = train_x @ weights + bias
        probabilities = _sigmoid(logits)
        if not np.all(np.isfinite(probabilities)):
            raise FloatingPointError(f"non-finite training probability at epoch {epoch}")
        error = probabilities - train_y
        weights -= lr * (train_x.T @ error / n + l2 * weights)
        bias -= lr * float(np.mean(error))
        if epoch in {0, 1, 9, 49, 99, int(protocol["epochs"]) - 1}:
            loss = -float(np.mean(train_y * np.log(np.clip(probabilities, 1.0e-8, 1.0 - 1.0e-8)) + (1 - train_y) * np.log(np.clip(1.0 - probabilities, 1.0e-8, 1.0))))
            history.append({"epoch": epoch + 1, "loss_before_update": loss, "weight_norm": float(np.linalg.norm(weights)), "bias": float(bias)})
    if not np.all(np.isfinite(weights)) or not math.isfinite(bias):
        raise FloatingPointError("non-finite fitted model")
    return weights, bias, history


def _metrics(x: np.ndarray, y: np.ndarray, weights: np.ndarray, bias: float) -> dict[str, Any]:
    scores = x @ weights + bias
    predicted = scores >= 0.0
    return {
        "example_count": int(len(y)),
        "positive_count": int(np.sum(y == 1)),
        "negative_count": int(np.sum(y == 0)),
        "accuracy": float(np.mean(predicted == y)),
        "auc": _auc(scores, y),
        "score_median": float(np.median(scores)),
        "score_p05": float(np.quantile(scores, 0.05)),
        "score_p95": float(np.quantile(scores, 0.95)),
    }


def main() -> int:
    rows = [json.loads(line) for line in INPUT_TABLE.read_text(encoding="utf-8").splitlines() if line.strip()]
    sequences = sorted({str(row["sequence"]) for row in rows})
    if len(sequences) < 4:
        raise RuntimeError(f"need at least four independent sequences, got {len(sequences)}")
    holdout_sequences = sequences[::4]
    train_sequences = [sequence for sequence in sequences if sequence not in set(holdout_sequences)]
    protocol = {
        "schema_version": "N72R5R1_ROUND04_TVC_V1_PROTOCOL_V1",
        "round": "round_04_tvc_v1",
        "model": "three-feature-logistic-candidate-verifier",
        "features": [
            "candidate_vs_event_anchor_cosine",
            "candidate_vs_prefix_target_prototype_cosine_with_zero_if_unavailable",
            "target_prototype_available_indicator",
        ],
        "label": "posthoc target candidate=1, solver competitor=0",
        "sequence_split_rule": "sorted unique pair-table sequences; every fourth sequence is holdout",
        "train_sequences": train_sequences,
        "holdout_sequences": holdout_sequences,
        "seed": 7202,
        "epochs": 400,
        "learning_rate": 0.05,
        "l2": 0.01,
        "runtime_max_abs_residual": 8.0,
        "runtime_formula": "clip(8.0*tanh(logit(candidate)), -8.0, 8.0) added only to target public row",
        "candidate_generation_changed": False,
        "hungarian_solver_changed": False,
        "future_effect_used_for_training_or_selection": False,
        "runtime_future_gt_used": False,
        "posthoc_gt_used_for_labels": True,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    atomic_json(PROTOCOL, protocol)

    train_x_raw, train_y, train_pairs = _build_examples(rows, set(train_sequences))
    holdout_x_raw, holdout_y, holdout_pairs = _build_examples(rows, set(holdout_sequences))
    train_x, holdout_x, stats = _standardize(train_x_raw, holdout_x_raw)
    weights, bias, history = _fit(train_x, train_y, protocol)
    train_metrics = _metrics(train_x, train_y, weights, bias)
    holdout_metrics = _metrics(holdout_x, holdout_y, weights, bias)
    model = {
        "schema_version": "N72R5R1_TVC_V1_MODEL_V1",
        "model": protocol["model"],
        "features": protocol["features"],
        "standardization_mean": stats[0].tolist(),
        "standardization_scale": stats[1].tolist(),
        "weights": weights.tolist(),
        "bias": float(bias),
        "max_abs_residual": float(protocol["runtime_max_abs_residual"]),
        "protocol_sha256": sha256_file(PROTOCOL),
        "posthoc_gt_used_for_labels": True,
        "runtime_future_gt_used": False,
    }
    training = {
        "schema_version": "N72R5R1_ROUND04_TRAINING_RESULT_V1",
        "status": "PASS_TVC_V1_TRAINING",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "input_table": str(INPUT_TABLE),
        "input_table_sha256": sha256_file(INPUT_TABLE),
        "protocol_sha256": sha256_file(PROTOCOL),
        "train_pair_count": train_pairs,
        "holdout_pair_count": holdout_pairs,
        "train_metrics": train_metrics,
        "holdout_metrics": holdout_metrics,
        "optimization_history": history,
        "selection_rule": "model is fixed by protocol; no future-effect metric was used",
        "runtime_future_gt_used": False,
        "posthoc_gt_used": True,
    }
    status = {
        "schema_version": "N72R5R1_ROUND04_STATUS_V1",
        "stage": "04_TVC_V1_TRAINING",
        "status": training["status"],
        "protocol": str(PROTOCOL),
        "model": str(MODEL),
        "training_result": str(TRAINING),
        "train_sequences": train_sequences,
        "holdout_sequences": holdout_sequences,
        "holdout_auc": holdout_metrics["auc"],
        "runtime_future_gt_used": False,
        "posthoc_gt_used": True,
        "next_routing": "TVC_V1_REPLAY_ON_FROZEN_40_EVENTS",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    atomic_json(MODEL, model)
    atomic_json(TRAINING, training)
    atomic_json(STATUS, status)
    print(json.dumps({"status": status["status"], "train_pairs": train_pairs, "holdout_pairs": holdout_pairs, "holdout_auc": holdout_metrics["auc"], "model": str(MODEL)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
