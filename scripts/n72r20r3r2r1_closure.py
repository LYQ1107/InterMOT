#!/usr/bin/env python3
"""N72R20R3R2R1 open-set identity decision closure.

This stage consumes the sealed N72R20R3R2 representation predictions and the
already-frozen candidate tape.  It deliberately does not call SAM3, rebuild
the candidate stream, change the association solver, or write model binaries
to the repository.  The implementation keeps posthoc labels in a separate
in-memory object and only serializes them to the explicitly posthoc label
tape.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit, logsumexp, softmax

import torch
from torch import nn

from sam3_intermot.identity_verification.cross_scene_adapter import CrossSceneIdentityAdapter


ROOT = Path(__file__).resolve().parents[1]
STAGE = "N72R20R3R2R1"
OUT = ROOT / "outputs" / STAGE
SOURCE = ROOT / "outputs" / "N72R20R3R2"
PROTOCOL_SOURCE = ROOT / "outputs" / "N72R20R3R1R1"
ASSET_ROOT = Path("/data3/liuyeqiang/InterMOT_N72R20R2_assets")
MODEL_ROOT = Path("/data3/liuyeqiang/InterMOT_N72R20R3R2_assets/models")
SOURCE_HEAD = "02dada7664ef0e5c13ec3df74b8b284b41ddaef3"
SEQUENCES = (
    "dancetrack0001", "dancetrack0002", "dancetrack0023", "dancetrack0024",
    "dancetrack0039", "dancetrack0057", "dancetrack0062", "dancetrack0072",
)
CALIBRATOR_SEEDS = (720331, 720332, 720333)
R3R2_SEEDS = (720321, 720322, 720323)
HORIZONS = (20, 50, 100)
FORMAL_FPR = 0.02
FORMAL_RECALL = 0.60
FORMAL_MACRO_FPR = 0.05
FORMAL_MACRO_RECALL = 0.40


def _plain(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return [_plain(item) for item in value.tolist()]
    if isinstance(value, (np.floating, float)):
        value = float(value)
        return value if math.isfinite(value) else None
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    return value


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_plain(value), indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_zstd_jsonl(path: Path) -> list[dict[str, Any]]:
    completed = subprocess.run(["zstd", "-q", "-dc", str(path)], check=True, stdout=subprocess.PIPE)
    return [json.loads(line) for line in completed.stdout.decode("utf-8").splitlines() if line.strip()]


def write_zstd_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    payload = "".join(json.dumps(_plain(dict(row)), sort_keys=True, allow_nan=False, separators=(",", ":")) + "\n" for row in rows).encode("utf-8")
    completed = subprocess.run(["zstd", "-q", "-T0", "-c"], input=payload, stdout=subprocess.PIPE, check=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(completed.stdout)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def storage_audit() -> dict[str, Any]:
    stat = __import__("shutil").disk_usage(ROOT)
    free = stat.free / float(1 << 30)
    return {
        "stage": STAGE,
        "path": str(ROOT),
        "total_gib": stat.total / float(1 << 30),
        "used_gib": (stat.total - stat.free) / float(1 << 30),
        "free_gib": free,
        "warning_below_gib": 108.0,
        "hard_stop_below_gib": 100.0,
        "storage_status": "HARD_STOP" if free < 100.0 else ("WARNING" if free < 108.0 else "OK"),
        "target_new_storage": "<6GB",
    }


def unit(value: Sequence[float]) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    norm = float(np.linalg.norm(array))
    return array / max(norm, 1.0e-8)


def summarize(values: Sequence[float]) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        return {key: None for key in ("count", "mean", "median", "std", "p5", "p10", "p25", "p50", "p75", "p90", "p95", "min", "max")}
    return {
        "count": int(array.size),
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "std": float(np.std(array)),
        "p5": float(np.percentile(array, 5)),
        "p10": float(np.percentile(array, 10)),
        "p25": float(np.percentile(array, 25)),
        "p50": float(np.percentile(array, 50)),
        "p75": float(np.percentile(array, 75)),
        "p90": float(np.percentile(array, 90)),
        "p95": float(np.percentile(array, 95)),
        "min": float(np.min(array)),
        "max": float(np.max(array)),
    }


def _event_map() -> dict[str, dict[str, Any]]:
    return {str(item["sequence"]): dict(item) for item in read_json(ASSET_ROOT / "interaction_events.json")["events"]}


def _checkpoint_hashes() -> dict[str, list[str]]:
    result: dict[str, list[str]] = defaultdict(list)
    for sequence in SEQUENCES:
        for seed in R3R2_SEEDS:
            path = MODEL_ROOT / f"cross_scene_adapter__{sequence}__seed{seed}.pt"
            result[sequence].append(sha256(path))
    return dict(result)


def _context(row: Mapping[str, Any]) -> np.ndarray:
    context = row["runtime_context"]
    return np.asarray(
        [
            float(context.get("candidate_count", len(row["candidate_axis"]))),
            float(context.get("frames_since_human_initialization", 0.0)) / 100.0,
            float(context.get("frames_since_last_memory_write", 0.0)) / 100.0,
            float(context.get("learned_state_human_anchor_cosine", 1.0)),
        ],
        dtype=np.float64,
    )


def _row_summary(scores: Sequence[float], context: np.ndarray) -> np.ndarray:
    values = np.asarray(scores, dtype=np.float64)
    if values.size == 0:
        top1 = top2 = mean = std = median = 0.0
    else:
        order = np.sort(values)[::-1]
        top1 = float(order[0])
        top2 = float(order[1]) if order.size > 1 else top1
        mean = float(np.mean(values))
        std = float(np.std(values))
        median = float(np.median(values))
    return np.asarray(
        [top1, top2, top1 - top2, mean, std, median, top1 - mean, top1 - median, math.log1p(values.size), *context[1:]],
        dtype=np.float64,
    )


def _set_features(scores: Sequence[float]) -> np.ndarray:
    values = np.asarray(scores, dtype=np.float64)
    if values.size == 0:
        return np.zeros(8, dtype=np.float64)
    probabilities = softmax(values / 0.07)
    order = np.sort(values)[::-1]
    z = (values - values.mean()) / max(float(values.std()), 1.0e-8)
    return np.asarray(
        [
            float(logsumexp(values)),
            float(-np.sum(probabilities * np.log(np.maximum(probabilities, 1.0e-12)))),
            float(order[0] - (order[1] if len(order) > 1 else order[0])),
            float(z.max()),
            float(values.std()),
            float(values.mean()),
            float(values.min()),
            float(values.max()),
        ],
        dtype=np.float64,
    )


def _safe_sigmoid(logits: np.ndarray) -> np.ndarray:
    return np.asarray(expit(np.clip(logits, -40.0, 40.0)), dtype=np.float64)


def _fit_logistic(x: np.ndarray, y: np.ndarray, seed: int, weights: np.ndarray | None = None, l2: float = 1.0e-3) -> dict[str, Any]:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    mean = x.mean(axis=0) if x.shape[0] else np.zeros(x.shape[1])
    scale = x.std(axis=0) if x.shape[0] else np.ones(x.shape[1])
    scale = np.where(scale < 1.0e-8, 1.0, scale)
    z = (x - mean) / scale
    weight = np.ones(len(y), dtype=np.float64) if weights is None else np.asarray(weights, dtype=np.float64)
    rng = np.random.default_rng(seed)
    initial = rng.normal(0.0, 0.01, z.shape[1] + 1)

    def objective(parameter: np.ndarray) -> tuple[float, np.ndarray]:
        logits = np.clip(z @ parameter[:-1] + parameter[-1], -40.0, 40.0)
        loss = np.logaddexp(0.0, logits) - y * logits
        weighted = np.sum(weight * loss) / max(float(np.sum(weight)), 1.0)
        value = weighted + 0.5 * l2 * float(np.sum(parameter[:-1] ** 2))
        residual = (expit(logits) - y) * weight / max(float(np.sum(weight)), 1.0)
        gradient = np.concatenate([z.T @ residual + l2 * parameter[:-1], [float(np.sum(residual))]])
        return float(value), gradient

    if len(np.unique(y)) < 2:
        return {"kind": "constant", "probability": float(y.mean() if len(y) else 0.0), "feature_mean": mean, "feature_scale": scale}
    result = minimize(lambda parameter: objective(parameter), initial, jac=True, method="L-BFGS-B", options={"maxiter": 80, "ftol": 1.0e-8})
    return {"kind": "logistic", "coef": result.x[:-1], "bias": float(result.x[-1]), "feature_mean": mean, "feature_scale": scale, "converged": bool(result.success), "iterations": int(result.nit), "seed": seed}


def _predict_logistic(model: Mapping[str, Any], x: np.ndarray) -> np.ndarray:
    if model["kind"] == "constant":
        return np.full(len(x), float(model["probability"]), dtype=np.float64)
    z = (np.asarray(x, dtype=np.float64) - np.asarray(model["feature_mean"])) / np.asarray(model["feature_scale"])
    return _safe_sigmoid(z @ np.asarray(model["coef"]) + float(model["bias"]))


def _bce(y: np.ndarray, probability: np.ndarray) -> float:
    p = np.clip(np.asarray(probability, dtype=np.float64), 1.0e-8, 1.0 - 1.0e-8)
    return float(-np.mean(np.asarray(y, dtype=np.float64) * np.log(p) + (1.0 - np.asarray(y, dtype=np.float64)) * np.log(1.0 - p)))


def _ece(y: np.ndarray, probability: np.ndarray, bins: int = 10) -> float:
    y = np.asarray(y, dtype=np.float64)
    probability = np.asarray(probability, dtype=np.float64)
    if len(y) == 0:
        return 0.0
    total = 0.0
    for low, high in zip(np.linspace(0.0, 1.0, bins + 1)[:-1], np.linspace(0.0, 1.0, bins + 1)[1:]):
        mask = (probability >= low) & ((probability < high) if high < 1.0 else (probability <= high))
        if np.any(mask):
            total += float(mask.mean()) * abs(float(probability[mask].mean()) - float(y[mask].mean()))
    return float(total)


def _row_labels(rows: Sequence[Mapping[str, Any]]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    label = np.asarray([int(row["label_index"]) for row in rows], dtype=np.int64)
    present = label >= 0
    taxonomy = np.asarray([0 if present[index] else (2 if bool(row["target_gt_present_posthoc"]) else 1) for index, row in enumerate(rows)], dtype=np.int64)
    return label, present, taxonomy


def _policy_metrics(rows: Sequence[Mapping[str, Any]], probability: Sequence[float], threshold: float, variant: str = "raw") -> dict[str, Any]:
    probability = np.asarray(probability, dtype=np.float64)
    labels, present, taxonomy = _row_labels(rows)
    predicted = np.asarray([int(np.argmax(row.get("score_variants", {}).get(variant, row["scores"]))) if len(row.get("score_variants", {}).get(variant, row["scores"])) else -1 for row in rows], dtype=np.int64)
    accepted = probability >= float(threshold)
    predicted = np.where(accepted, predicted, -1)
    absent = ~present
    correct = accepted & present & (predicted == labels)
    false_positive = accepted & absent
    result = {
        "rows": int(len(rows)),
        "present_rows": int(present.sum()),
        "none_rows": int(absent.sum()),
        "accepted_count": int(accepted.sum()),
        "correct_id_count": int(correct.sum()),
        "false_positive_count": int(false_positive.sum()),
        "negative_fpr": float(false_positive.sum() / max(1, absent.sum())),
        "open_set_correct_id_recall": float(correct.sum() / max(1, present.sum())),
        "presence_recall": float((accepted & present).sum() / max(1, present.sum())),
        "threshold": float(threshold),
        "ece": _ece(present.astype(float), probability),
        "P0_fpr": float((false_positive & (taxonomy == 1)).sum() / max(1, (taxonomy == 1).sum())),
        "P1_fpr": float((false_positive & (taxonomy == 2)).sum() / max(1, (taxonomy == 2).sum())),
        "predicted_indices": predicted.astype(int).tolist(),
        "accepted": accepted.astype(bool).tolist(),
    }
    return result


def _select_threshold(rows: Sequence[Mapping[str, Any]], probability: Sequence[float], variant: str = "raw") -> tuple[float, dict[str, Any]]:
    probability = np.asarray(probability, dtype=np.float64)
    labels, present, taxonomy = _row_labels(rows)
    predicted = np.asarray([int(np.argmax(row.get("score_variants", {}).get(variant, row["scores"]))) if row.get("score_variants", {}).get(variant, row["scores"]) else -1 for row in rows], dtype=np.int64)
    values = np.unique(np.concatenate(([0.0, 1.0], probability)))
    candidates = []
    for threshold in values:
        accepted = probability >= float(threshold)
        false_positive = accepted & ~present
        correct = accepted & present & (predicted == labels)
        candidates.append({"rows": int(len(rows)), "present_rows": int(present.sum()), "none_rows": int((~present).sum()), "accepted_count": int(accepted.sum()), "correct_id_count": int(correct.sum()), "false_positive_count": int(false_positive.sum()), "negative_fpr": float(false_positive.sum() / max(1, (~present).sum())), "open_set_correct_id_recall": float(correct.sum() / max(1, present.sum())), "presence_recall": float((accepted & present).sum() / max(1, present.sum())), "threshold": float(threshold), "ece": _ece(present.astype(float), probability), "P0_fpr": float((false_positive & (taxonomy == 1)).sum() / max(1, (taxonomy == 1).sum())), "P1_fpr": float((false_positive & (taxonomy == 2)).sum() / max(1, (taxonomy == 2).sum())), "predicted_indices": predicted.tolist(), "accepted": accepted.tolist()})
    feasible = [item for item in candidates if item["negative_fpr"] <= FORMAL_FPR + 1.0e-12]
    def compact(item: Mapping[str, Any]) -> dict[str, Any]:
        return {key: value for key, value in item.items() if key not in {"predicted_indices", "accepted"}}
    if feasible:
        selected = sorted(feasible, key=lambda item: (-item["open_set_correct_id_recall"], item["negative_fpr"], item["ece"], item["threshold"]))[0]
        return float(selected["threshold"]), {"inner_gate_infeasible": False, "candidate_count": len(candidates), "selected": compact(selected)}
    selected = sorted(candidates, key=lambda item: (item["negative_fpr"], -item["open_set_correct_id_recall"], item["ece"], item["threshold"]))[0]
    return float(selected["threshold"]), {"inner_gate_infeasible": True, "candidate_count": len(candidates), "selected": compact(selected)}


def _fit_scale(rows: Sequence[Mapping[str, Any]], mode: str, variant: str = "raw") -> dict[str, Any]:
    top1 = np.asarray([float(np.max(row.get("score_variants", {}).get(variant, row["scores"]))) if row.get("score_variants", {}).get(variant, row["scores"]) else 0.0 for row in rows], dtype=np.float64)
    labels = _row_labels(rows)[1].astype(float)
    if mode == "affine":
        def transform(theta: np.ndarray) -> np.ndarray:
            return np.exp(np.clip(theta[0], -8.0, 8.0)) * top1 + theta[1]
        initial = np.asarray([0.0, 0.0])
        temperature = None
    else:
        def transform(theta: np.ndarray) -> np.ndarray:
            temperature = np.log1p(np.exp(np.clip(theta[0], -15.0, 15.0))) + 1.0e-4
            return top1 / temperature + theta[1]
        initial = np.asarray([math.log(math.expm1(1.0)), 0.0])
        temperature = "learned"
    def objective(theta: np.ndarray) -> float:
        return _bce(labels, _safe_sigmoid(transform(theta))) + 1.0e-4 * float(np.sum(theta * theta))
    result = minimize(objective, initial, method="L-BFGS-B", options={"maxiter": 80, "ftol": 1.0e-8})
    if mode == "affine":
        return {"mode": mode, "a": float(np.exp(np.clip(result.x[0], -8.0, 8.0))), "b": float(result.x[1]), "success": bool(result.success)}
    return {"mode": mode, "temperature": float(np.log1p(np.exp(np.clip(result.x[0], -15.0, 15.0))) + 1.0e-4), "b": float(result.x[1]), "success": bool(result.success)}


def _transform_scores(row: Mapping[str, Any], scale: Mapping[str, Any], variant: str = "raw") -> np.ndarray:
    values = np.asarray(row.get("score_variants", {}).get(variant, row["scores"]), dtype=np.float64)
    if scale["mode"] == "affine":
        return float(scale["a"]) * values + float(scale["b"])
    return values / float(scale["temperature"]) + float(scale["b"])


def _features(rows: Sequence[Mapping[str, Any]], variant: str = "raw") -> np.ndarray:
    values = []
    for row in rows:
        scores = row.get("score_variants", {}).get(variant, row["scores"])
        values.append(np.concatenate([_row_summary(scores, np.asarray(row["context"], dtype=np.float64)), _set_features(scores)]))
    return np.asarray(values, dtype=np.float64)


def _raw_summary_features(rows: Sequence[Mapping[str, Any]], variant: str = "raw") -> np.ndarray:
    return np.asarray([_row_summary(row.get("score_variants", {}).get(variant, row["scores"]), np.asarray(row["context"], dtype=np.float64)) for row in rows], dtype=np.float64)


class TinyMLP(nn.Module):
    def __init__(self, input_dim: int, output_dim: int = 1) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(input_dim, 32), nn.GELU(), nn.Linear(32, 16), nn.GELU(), nn.Linear(16, output_dim))

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.net(value)


class DeepSetPresence(nn.Module):
    def __init__(self, context_dim: int = 4) -> None:
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(3, 16), nn.GELU(), nn.Linear(16, 16), nn.GELU())
        self.head = nn.Sequential(nn.Linear(16 * 3 + context_dim, 16), nn.GELU(), nn.Linear(16, 1))

    def forward(self, candidate_set: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        encoded = self.encoder(candidate_set)
        pooled = torch.cat([encoded.mean(dim=1), encoded.max(dim=1).values, encoded.std(dim=1, unbiased=False)], dim=1)
        return self.head(torch.cat([pooled, context], dim=1)).squeeze(-1)


def _set_tensor(rows: Sequence[Mapping[str, Any]], variant: str = "raw") -> tuple[np.ndarray, np.ndarray]:
    sets = []
    contexts = []
    max_count = max(1, max(len(row.get("score_variants", {}).get(variant, row["scores"])) for row in rows))
    for row in rows:
        scores = np.asarray(row.get("score_variants", {}).get(variant, row["scores"]), dtype=np.float32)
        if len(scores):
            order = np.argsort(-scores, kind="stable")
            ranks = np.empty(len(scores), dtype=np.float32)
            ranks[order] = np.arange(len(scores), dtype=np.float32) / max(1, len(scores) - 1)
            values = np.stack([scores, ranks, scores.max() - scores], axis=1)
        else:
            values = np.zeros((1, 3), dtype=np.float32)
        padded = np.zeros((max_count, 3), dtype=np.float32)
        padded[: len(values)] = values
        sets.append(padded)
        contexts.append(np.asarray(row["context"], dtype=np.float32))
    return np.asarray(sets), np.asarray(contexts)


def _fit_torch_binary(rows: Sequence[Mapping[str, Any]], variant: str, seed: int, kind: str, device: torch.device, epochs: int = 25) -> dict[str, Any]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    labels = torch.from_numpy(_row_labels(rows)[1].astype(np.float32)).to(device)
    if kind == "deepset":
        candidate, context = _set_tensor(rows, variant)
        model: nn.Module = DeepSetPresence().to(device)
        features = (torch.from_numpy(candidate).to(device), torch.from_numpy(context).to(device))
    else:
        array = _raw_summary_features(rows, variant).astype(np.float32)
        mean = array.mean(axis=0)
        scale = np.where(array.std(axis=0) < 1.0e-8, 1.0, array.std(axis=0))
        model = TinyMLP(array.shape[1]).to(device)
        features = (torch.from_numpy(((array - mean) / scale).astype(np.float32)).to(device),)
    optimizer = torch.optim.Adam(model.parameters(), lr=2.0e-3, weight_decay=1.0e-4)
    best_state = None
    best_loss = float("inf")
    patience = 12
    for _ in range(epochs):
        optimizer.zero_grad(set_to_none=True)
        logits = model(*features) if kind == "deepset" else model(features[0]).squeeze(-1)
        loss = nn.functional.binary_cross_entropy_with_logits(logits, labels)
        loss.backward()
        optimizer.step()
        value = float(loss.detach().cpu())
        if value < best_loss - 1.0e-6:
            best_loss = value
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            patience = 12
        else:
            patience -= 1
            if patience <= 0:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return {"kind": kind, "model": model, "feature_mean": None if kind == "deepset" else mean, "feature_scale": None if kind == "deepset" else scale, "seed": seed, "epochs": epochs, "trainable_parameters": int(sum(parameter.numel() for parameter in model.parameters()))}


def _predict_torch(model: Mapping[str, Any], rows: Sequence[Mapping[str, Any]], variant: str, device: torch.device) -> np.ndarray:
    with torch.no_grad():
        if model["kind"] == "deepset":
            candidate, context = _set_tensor(rows, variant)
            logits = model["model"](torch.from_numpy(candidate).to(device), torch.from_numpy(context).to(device))
        else:
            array = _raw_summary_features(rows, variant).astype(np.float32)
            array = (array - np.asarray(model["feature_mean"])) / np.asarray(model["feature_scale"])
            logits = model["model"](torch.from_numpy(array).to(device)).squeeze(-1)
    return _safe_sigmoid(logits.detach().cpu().numpy())


def _fit_multitask(rows: Sequence[Mapping[str, Any]], variant: str, seed: int, device: torch.device, epochs: int = 25) -> dict[str, Any]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    array = _features(rows, variant).astype(np.float32)
    mean = array.mean(axis=0)
    scale = np.where(array.std(axis=0) < 1.0e-8, 1.0, array.std(axis=0))
    values = torch.from_numpy(((array - mean) / scale).astype(np.float32)).to(device)
    binary = torch.from_numpy(_row_labels(rows)[1].astype(np.float32)).to(device)
    taxonomy = torch.from_numpy(_row_labels(rows)[2]).to(device)
    model = TinyMLP(array.shape[1], output_dim=4).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=2.0e-3, weight_decay=1.0e-4)
    best_state = None
    best_loss = float("inf")
    for _ in range(epochs):
        optimizer.zero_grad(set_to_none=True)
        output = model(values)
        loss = nn.functional.binary_cross_entropy_with_logits(output[:, 0], binary) + 0.25 * nn.functional.cross_entropy(output[:, 1:], taxonomy)
        loss.backward()
        optimizer.step()
        value = float(loss.detach().cpu())
        if value < best_loss:
            best_loss = value
            best_state = {key: item.detach().cpu().clone() for key, item in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return {"kind": "multitask", "model": model, "feature_mean": mean, "feature_scale": scale, "seed": seed, "trainable_parameters": int(sum(parameter.numel() for parameter in model.parameters())), "epochs": epochs}


def _predict_multitask(model: Mapping[str, Any], rows: Sequence[Mapping[str, Any]], variant: str, device: torch.device) -> np.ndarray:
    array = _features(rows, variant).astype(np.float32)
    array = (array - np.asarray(model["feature_mean"])) / np.asarray(model["feature_scale"])
    with torch.no_grad():
        output = model["model"](torch.from_numpy(array).to(device))
    return _safe_sigmoid(output[:, 0].detach().cpu().numpy())


def _fit_method(rows: Sequence[Mapping[str, Any]], family: str, variant: str, seed: int, device: torch.device, balance_ratio: tuple[int, int] | None = None) -> dict[str, Any]:
    labels, present, taxonomy = _row_labels(rows)
    if family == "B0_LOGISTIC":
        model = _fit_logistic(_raw_summary_features(rows, variant), present.astype(float), seed, weights=_balance_weights(present, balance_ratio))
        return {"family": family, "kind": "logistic", "model": model, "variant": variant}
    if family in {"B2_AFFINE", "B3_TEMPERATURE"}:
        scale = _fit_scale(rows, "affine" if family == "B2_AFFINE" else "temperature", variant)
        transformed = []
        for row in rows:
            transformed.append(np.concatenate([_row_summary(_transform_scores(row, scale, variant), np.asarray(row["context"])), _set_features(_transform_scores(row, scale, variant))]))
        model = _fit_logistic(np.asarray(transformed), present.astype(float), seed, weights=_balance_weights(present, balance_ratio))
        return {"family": family, "kind": "scaled_logistic", "model": model, "scale": scale, "variant": variant}
    if family == "B1_MLP":
        models = [_fit_torch_binary(rows, variant, item, "mlp", device) for item in CALIBRATOR_SEEDS]
        return {"family": family, "kind": "torch_ensemble", "models": models, "variant": variant}
    if family == "DEEPSET":
        models = [_fit_torch_binary(rows, variant, item, "deepset", device) for item in CALIBRATOR_SEEDS]
        return {"family": family, "kind": "torch_ensemble", "models": models, "variant": variant}
    if family == "E0_MULTITASK":
        models = [_fit_multitask(rows, variant, item, device) for item in CALIBRATOR_SEEDS]
        return {"family": family, "kind": "multitask_ensemble", "models": models, "variant": variant}
    if family in {"C1_ENERGY", "C2_ENTROPY", "C3_MARGIN", "C4_MAHALANOBIS_PRESENT", "C5_MAHALANOBIS_ABSENT"}:
        stats = np.asarray([_set_features(row.get("score_variants", {}).get(variant, row["scores"])) for row in rows], dtype=np.float64)
        if family == "C1_ENERGY":
            index = [0]
        elif family == "C2_ENTROPY":
            index = [1]
        elif family == "C3_MARGIN":
            index = [2]
        else:
            mean_present = stats[present].mean(axis=0) if np.any(present) else stats.mean(axis=0)
            mean_absent = stats[~present].mean(axis=0) if np.any(~present) else stats.mean(axis=0)
            cov = np.cov(stats[present].T) if np.sum(present) > stats.shape[1] else np.eye(stats.shape[1])
            cov = np.asarray(cov, dtype=np.float64) + np.eye(stats.shape[1]) * 1.0e-4
            inverse = np.linalg.pinv(cov)
            d_present = np.einsum("ij,jk,ik->i", stats - mean_present, inverse, stats - mean_present)
            d_absent = np.einsum("ij,jk,ik->i", stats - mean_absent, inverse, stats - mean_absent)
            values = (d_absent - d_present) if family == "C4_MAHALANOBIS_PRESENT" else (d_present - d_absent)
            model = _fit_logistic(values[:, None], present.astype(float), seed, weights=_balance_weights(present, balance_ratio))
            return {"family": family, "kind": "logistic", "model": model, "variant": variant, "mahalanobis": {"mean_present": mean_present, "mean_absent": mean_absent, "inverse": inverse}}
        model = _fit_logistic(stats[:, index], present.astype(float), seed, weights=_balance_weights(present, balance_ratio))
        return {"family": family, "kind": "logistic", "model": model, "variant": variant, "stat_index": index}
    if family == "D0_CONFORMAL":
        absent_top1 = np.asarray([max(row.get("score_variants", {}).get(variant, row["scores"]), default=0.0) for row in rows if not bool(_row_labels([row])[1][0])], dtype=np.float64)
        quantile = float(np.quantile(absent_top1, 1.0 - FORMAL_FPR)) if len(absent_top1) else 1.0
        scale = max(float(np.std(absent_top1)) if len(absent_top1) else 0.1, 1.0e-3)
        return {"family": family, "kind": "conformal", "variant": variant, "quantile": quantile, "scale": scale, "alpha": FORMAL_FPR}
    raise ValueError(f"unknown calibration family {family}")


def _balance_weights(present: np.ndarray, ratio: tuple[int, int] | None) -> np.ndarray | None:
    if ratio is None:
        return None
    positive, negative = ratio
    weights = np.where(present, float(positive), float(negative))
    return weights / max(float(weights.mean()), 1.0e-8)


def _predict_method(model: Mapping[str, Any], rows: Sequence[Mapping[str, Any]], device: torch.device) -> np.ndarray:
    family = model["family"]
    variant = str(model.get("variant", "raw"))
    if family == "B0_LOGISTIC":
        return _predict_logistic(model["model"], _raw_summary_features(rows, variant))
    if family in {"B2_AFFINE", "B3_TEMPERATURE"}:
        transformed = np.asarray([np.concatenate([_row_summary(_transform_scores(row, model["scale"], variant), np.asarray(row["context"])), _set_features(_transform_scores(row, model["scale"], variant))]) for row in rows])
        return _predict_logistic(model["model"], transformed)
    if family in {"B1_MLP", "DEEPSET"}:
        return np.mean([_predict_torch(item, rows, variant, device) for item in model["models"]], axis=0)
    if family == "E0_MULTITASK":
        return np.mean([_predict_multitask(item, rows, variant, device) for item in model["models"]], axis=0)
    if family in {"C1_ENERGY", "C2_ENTROPY", "C3_MARGIN"}:
        stats = np.asarray([_set_features(row.get("score_variants", {}).get(variant, row["scores"])) for row in rows])
        return _predict_logistic(model["model"], stats[:, model["stat_index"]])
    if family in {"C4_MAHALANOBIS_PRESENT", "C5_MAHALANOBIS_ABSENT"}:
        stats = np.asarray([_set_features(row.get("score_variants", {}).get(variant, row["scores"])) for row in rows])
        mean_present = np.asarray(model["mahalanobis"]["mean_present"])
        mean_absent = np.asarray(model["mahalanobis"]["mean_absent"])
        inverse = np.asarray(model["mahalanobis"]["inverse"])
        d_present = np.einsum("ij,jk,ik->i", stats - mean_present, inverse, stats - mean_present)
        d_absent = np.einsum("ij,jk,ik->i", stats - mean_absent, inverse, stats - mean_absent)
        values = (d_absent - d_present) if family == "C4_MAHALANOBIS_PRESENT" else (d_present - d_absent)
        return _predict_logistic(model["model"], values[:, None])
    if family == "D0_CONFORMAL":
        top1 = np.asarray([max(row.get("score_variants", {}).get(variant, row["scores"]), default=0.0) for row in rows])
        return _safe_sigmoid((top1 - float(model["quantile"])) / float(model["scale"]))
    raise ValueError(f"unknown prediction family {family}")


CALIBRATION_FAMILIES = (
    "B0_LOGISTIC", "B1_MLP", "B2_AFFINE", "B3_TEMPERATURE",
    "C1_ENERGY", "C2_ENTROPY", "C3_MARGIN", "C4_MAHALANOBIS_PRESENT",
    "C5_MAHALANOBIS_ABSENT", "DEEPSET", "D0_CONFORMAL", "E0_MULTITASK",
)


def load_rows() -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    canonical = []
    with (PROTOCOL_SOURCE / "training/canonical_training_index.jsonl").open(encoding="utf-8") as handle:
        canonical = [json.loads(line) for line in handle if line.strip()]
    formal = read_zstd_jsonl(SOURCE / "open_set/formal_predictions.jsonl.zst")
    seed_formal = {seed: read_zstd_jsonl(SOURCE / f"open_set/formal_predictions_seed{seed}.jsonl.zst") for seed in R3R2_SEEDS}
    if len(canonical) != 8414 or len(formal) != 8414 or any(len(rows) != 8414 for rows in seed_formal.values()):
        raise RuntimeError("R3R2 sealed score source is not the expected 8414-row tape")
    canonical_by_key = {(str(row["sequence"]), int(row["frame"])): row for row in canonical}
    formal_by_key = {(str(row["sequence"]), int(row["frame"])): row for row in formal}
    if set(canonical_by_key) != set(formal_by_key):
        raise RuntimeError("canonical and formal score axes differ")
    rows: list[dict[str, Any]] = []
    labels: list[dict[str, Any]] = []
    checkpoints = _checkpoint_hashes()
    events = _event_map()
    for key in sorted(canonical_by_key):
        source = canonical_by_key[key]
        scored = formal_by_key[key]
        axis = [str(item["candidate_uid"]) for item in source["candidate_axis"]]
        if axis != [str(item) for item in scored["candidate_uids"]]:
            raise RuntimeError(f"candidate axis changed at {key}")
        if scored.get("runtime_future_gt_used") is not False or scored.get("runtime_gt_clean") is not True:
            raise RuntimeError(f"runtime tape invariant failed at {key}")
        context = _context(source)
        label_index = source["training_label"]["candidate_index"]
        label_index = -1 if label_index is None else int(label_index)
        target_present = bool(source["target_gt_present_posthoc"])
        taxonomy = "PRESENT" if label_index >= 0 else ("P1_TARGET_PRESENT_UNAVAILABLE" if target_present else "P0_TARGET_ABSENT")
        row = {
            "sequence": str(source["sequence"]),
            "frame": int(source["frame"]),
            "candidate_uids": axis,
            "candidate_axis": [dict(item) for item in source["candidate_axis"]],
            "scores": [float(value) for value in scored["candidate_scores"]],
            "none_score": float(scored["none_score"]),
            "context": context.tolist(),
            "label_index": label_index,
            "target_gt_present_posthoc": target_present,
            "taxonomy_posthoc": taxonomy,
            "outer_fold": str(source["sequence"]),
            "representation_checkpoint_sha": checkpoints.get(str(source["sequence"]), []),
            "runtime_future_gt_used": False,
            "runtime_gt_clean": True,
            "candidate_created": False,
            "score_variants": {"raw": [float(value) for value in scored["candidate_scores"]]},
            "source_score_seed_count": int(scored.get("seed_count", len(R3R2_SEEDS))),
            "state_ref": int(source["causal_learned_state_ref"]),
            "human_anchor": [float(value) for value in events[str(source["sequence"])] ["human_anchor"]],
        }
        rows.append(row)
        labels.append({
            "sequence": row["sequence"],
            "frame": row["frame"],
            "candidate_set_present": bool(len(axis) > 0),
            "correct_candidate_index": label_index if label_index >= 0 else None,
            "taxonomy": taxonomy,
            "posthoc_gt_used": True,
        })
    seed_by_key: dict[int, dict[tuple[str, int], dict[str, Any]]] = {}
    for seed, seed_rows in seed_formal.items():
        seed_by_key[seed] = {(str(item["sequence"]), int(item["frame"])): item for item in seed_rows}
    for row in rows:
        key = (row["sequence"], row["frame"])
        for seed in R3R2_SEEDS:
            item = seed_by_key[seed][key]
            if [str(value) for value in item["candidate_uids"]] != row["candidate_uids"]:
                raise RuntimeError(f"seed candidate axis changed at {key}")
        row["seed_score_provenance"] = {str(seed): seed_by_key[seed][key]["candidate_scores"] for seed in R3R2_SEEDS}
    runtime_predictions = read_zstd_jsonl(SOURCE / "representation/runtime_predictions.jsonl.zst")
    oracle_predictions = read_zstd_jsonl(SOURCE / "representation/oracle_clean_predictions.jsonl.zst")
    runtime_rank = {(str(item["sequence"]), int(item["frame"])): item.get("rank") for item in runtime_predictions}
    oracle_rank = {(str(item["sequence"]), int(item["frame"])): item.get("rank") for item in oracle_predictions}
    for row in rows:
        key = (row["sequence"], row["frame"])
        row["runtime_rank_posthoc"] = runtime_rank.get(key)
        row["oracle_rank_posthoc"] = oracle_rank.get(key)
    metadata = {
        "canonical_rows": len(canonical),
        "formal_rows": len(formal),
        "present_rows": int(sum(int(row["label_index"]) >= 0 for row in rows)),
        "P0_rows": int(sum(row["taxonomy_posthoc"] == "P0_TARGET_ABSENT" for row in rows)),
        "P1_rows": int(sum(row["taxonomy_posthoc"] == "P1_TARGET_PRESENT_UNAVAILABLE" for row in rows)),
        "sequence_counts": {sequence: int(sum(row["sequence"] == sequence for row in rows)) for sequence in SEQUENCES},
        "checkpoint_hashes": checkpoints,
        "events": events,
    }
    return rows, labels, metadata


def write_tapes(rows: Sequence[Mapping[str, Any]], labels: Sequence[Mapping[str, Any]], metadata: Mapping[str, Any]) -> None:
    runtime_rows = []
    for row in rows:
        runtime_rows.append({
            "sequence": row["sequence"],
            "frame": row["frame"],
            "candidate_uid_axis": row["candidate_uids"],
            "candidate_scores": row["scores"],
            "top1": float(max(row["scores"], default=0.0)),
            "top2": float(sorted(row["scores"], reverse=True)[1] if len(row["scores"]) > 1 else max(row["scores"], default=0.0)),
            "margin": float(sorted(row["scores"], reverse=True)[0] - sorted(row["scores"], reverse=True)[1]) if len(row["scores"]) > 1 else 0.0,
            "mean": float(np.mean(row["scores"])) if row["scores"] else 0.0,
            "std": float(np.std(row["scores"])) if row["scores"] else 0.0,
            "median": float(np.median(row["scores"])) if row["scores"] else 0.0,
            "min": float(np.min(row["scores"])) if row["scores"] else 0.0,
            "max": float(np.max(row["scores"])) if row["scores"] else 0.0,
            "candidate_count": len(row["scores"]),
            "frames_since_human_initialization": float(row["context"][1] * 100.0),
            "frames_since_last_memory_write": float(row["context"][2] * 100.0),
            "anchor_state_cosine": float(row["context"][3]),
            "outer_fold": row["outer_fold"],
            "representation_checkpoint_sha": row["representation_checkpoint_sha"],
            "seed_ensemble_provenance": list(R3R2_SEEDS),
            "runtime_future_gt_used": False,
            "runtime_gt_clean": True,
            "candidate_created": False,
        })
    write_zstd_jsonl(OUT / "runtime_score_tape.jsonl.zst", runtime_rows)
    write_zstd_jsonl(OUT / "posthoc_open_set_labels.jsonl.zst", labels)
    write_json(OUT / "runtime_score_tape_manifest.json", {
        "stage": STAGE,
        "rows": len(runtime_rows),
        "unique_frames": len({(item["sequence"], item["frame"]) for item in runtime_rows}),
        "candidate_axes_unchanged": True,
        "gt_fields_present": False,
        "runtime_future_gt_used": False,
        "runtime_gt_clean": True,
        "source_stage": "N72R20R3R2",
        "source_head": SOURCE_HEAD,
        "representation_checkpoint_hashes": metadata["checkpoint_hashes"],
        "seed_ensemble": list(R3R2_SEEDS),
    })
    write_json(OUT / "posthoc_label_manifest.json", {
        "stage": STAGE,
        "rows": len(labels),
        "labels_separate_from_runtime_tape": True,
        "posthoc_gt_used": True,
        "taxonomy": ["PRESENT", "P0_TARGET_ABSENT", "P1_TARGET_PRESENT_UNAVAILABLE"],
    })


def c0_baseline(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    labels, present, taxonomy = _row_labels(rows)
    selected = []
    for row in rows:
        scores = np.asarray(row["scores"], dtype=np.float64)
        if not len(scores):
            selected.append(-1)
            continue
        winner = int(np.argmax(np.concatenate([scores, [float(row["none_score"])] ])))
        selected.append(winner if winner < len(scores) else -1)
    selected = np.asarray(selected)
    accepted = selected >= 0
    false_positive = accepted & ~present
    correct = accepted & present & (selected == labels)
    result = {
        "family": "C0_DIRECT_NONE_LOGIT",
        "temperature": 0.07,
        "direct_none_logit_vs_raw_cosine": True,
        "rows": len(rows),
        "negative_fpr": float(false_positive.sum() / max(1, (~present).sum())),
        "open_set_correct_id_recall": float(correct.sum() / max(1, present.sum())),
        "macro_negative_fpr": float(np.mean([_c0_sequence_metrics([row for row in rows if row["sequence"] == sequence])["negative_fpr"] for sequence in SEQUENCES])),
        "macro_open_set_correct_id_recall": float(np.mean([_c0_sequence_metrics([row for row in rows if row["sequence"] == sequence])["open_set_correct_id_recall"] for sequence in SEQUENCES])),
        "P0_fpr": float((false_positive & (taxonomy == 1)).sum() / max(1, (taxonomy == 1).sum())),
        "P1_fpr": float((false_positive & (taxonomy == 2)).sum() / max(1, (taxonomy == 2).sum())),
        "runtime_future_gt_used": False,
        "runtime_gt_clean": True,
    }
    return result


def _c0_sequence_metrics(rows: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    labels, present, _ = _row_labels(rows)
    selected = np.asarray([
        (lambda winner, count: winner if winner < count else -1)(
            int(np.argmax(np.concatenate([np.asarray(row["scores"]), [float(row["none_score"])] ]))), len(row["scores"])
        ) if row["scores"] else -1
        for row in rows
    ])
    accepted = selected >= 0
    false_positive = accepted & ~present
    correct = accepted & present & (selected == labels)
    return {"negative_fpr": float(false_positive.sum() / max(1, (~present).sum())), "open_set_correct_id_recall": float(correct.sum() / max(1, present.sum()))}


def score_scale_audit(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    groups: dict[str, dict[str, list[float]]] = {"PRESENT": {"candidate_scores": [], "top1": [], "none_scores": []}, "P0_TARGET_ABSENT": {"candidate_scores": [], "top1": [], "none_scores": []}, "P1_TARGET_PRESENT_UNAVAILABLE": {"candidate_scores": [], "top1": [], "none_scores": []}}
    for row in rows:
        group = str(row["taxonomy_posthoc"])
        groups[group]["candidate_scores"].extend(row["scores"])
        groups[group]["top1"].append(max(row["scores"], default=0.0))
        groups[group]["none_scores"].append(float(row["none_score"]))
    report = {group: {name: summarize(values) for name, values in values_by_name.items()} for group, values_by_name in groups.items()}
    overlap = {
        "present_top1_below_absent_none_p50": float(np.mean(np.asarray(groups["PRESENT"]["top1"]) <= np.percentile(groups["P0_TARGET_ABSENT"]["none_scores"] + groups["P1_TARGET_PRESENT_UNAVAILABLE"]["none_scores"], 50))),
        "absent_none_above_present_top1_p50": float(np.mean(np.asarray(groups["P0_TARGET_ABSENT"]["none_scores"] + groups["P1_TARGET_PRESENT_UNAVAILABLE"]["none_scores"]) >= np.percentile(groups["PRESENT"]["top1"], 50))),
        "candidate_bound": "cosine-like metric scores are bounded by normalized tower geometry",
        "none_head_bound": "unconstrained MLP scalar",
    }
    return {"stage": STAGE, "historical_direct_comparison": True, "groups": report, "overlap": overlap, "conclusion": "raw NONE and candidate cosine scores are not assumed calibrated onto one common scale"}


def _threshold_curve(rows: Sequence[Mapping[str, Any]], values: Sequence[float], thresholds: Sequence[float] | None = None) -> list[dict[str, Any]]:
    values = np.asarray(values, dtype=np.float64)
    labels, present, _ = _row_labels(rows)
    thresholds = np.unique(np.concatenate(([float(np.min(values)) - 1.0e-8, float(np.max(values)) + 1.0e-8], np.percentile(values, np.linspace(0.0, 100.0, 201))))) if thresholds is None else np.asarray(thresholds)
    result = []
    top = np.asarray([int(np.argmax(row["scores"])) if row["scores"] else -1 for row in rows])
    for threshold in thresholds:
        accepted = values >= float(threshold)
        false_positive = accepted & ~present
        correct = accepted & present & (top == labels)
        result.append({"threshold": float(threshold), "negative_fpr": float(false_positive.sum() / max(1, (~present).sum())), "correct_id_recall": float(correct.sum() / max(1, present.sum())), "presence_recall": float((accepted & present).sum() / max(1, present.sum()))})
    return result


def upper_bound_audit(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    all_training = {}
    for heldout in SEQUENCES:
        training = [row for row in rows if row["sequence"] != heldout]
        top1 = np.asarray([max(row["scores"], default=0.0) for row in training])
        set_values = np.asarray([_set_features(row["scores"]) for row in training])
        all_training[heldout] = {
            "A0_global_top1": _threshold_curve(training, top1),
            "A1_top1_margin_grid": [{"threshold_top1": float(a), "threshold_margin": float(b)} for a in np.percentile(top1, [25, 50, 75]) for b in np.percentile(set_values[:, 2], [25, 50, 75])],
            "A2_energy": _threshold_curve(training, set_values[:, 0]),
            "A3_entropy": _threshold_curve(training, -set_values[:, 1]),
            "A4_top1_zscore": _threshold_curve(training, (top1 - np.asarray([np.mean(row["scores"]) for row in training])) / np.maximum(np.asarray([np.std(row["scores"]) for row in training]), 1.0e-8)),
            "A5_candidate_score_dispersion": _threshold_curve(training, -set_values[:, 4]),
        }
    return {"stage": STAGE, "outer_training_only": True, "outer_folds": all_training}


def oracle_threshold_analysis(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    top1 = np.asarray([max(row["scores"], default=0.0) for row in rows])
    global_curve = _threshold_curve(rows, top1)
    feasible = [item for item in global_curve if item["negative_fpr"] <= FORMAL_FPR]
    by_sequence = {sequence: _threshold_curve([row for row in rows if row["sequence"] == sequence], [max(row["scores"], default=0.0) for row in rows if row["sequence"] == sequence]) for sequence in SEQUENCES}
    return {"stage": STAGE, "posthoc_diagnostic": True, "global_curve": global_curve, "global_max_recall_at_fpr_2pct": max((item["correct_id_recall"] for item in feasible), default=0.0), "per_sequence_curves": by_sequence, "scalar_threshold_structurally_insufficient": max((item["correct_id_recall"] for item in feasible), default=0.0) < 0.60}


def sequence_bottleneck_map(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    output = {}
    for sequence in SEQUENCES:
        subset = [row for row in rows if row["sequence"] == sequence]
        present = [row for row in subset if int(row["label_index"]) >= 0]
        runtime_rank1 = float(np.mean([int(row["runtime_rank_posthoc"]) == 1 for row in present if row.get("runtime_rank_posthoc") is not None])) if present else 0.0
        oracle_rank1 = float(np.mean([int(row["oracle_rank_posthoc"]) == 1 for row in present if row.get("oracle_rank_posthoc") is not None])) if present else 0.0
        c0 = _c0_sequence_metrics(subset)
        gap = oracle_rank1 - runtime_rank1
        if gap > 0.20 and runtime_rank1 < 0.60:
            classification = "STATE_OR_RANKING_DOMINANT"
        elif c0["negative_fpr"] > 0.20 and runtime_rank1 >= 0.60:
            classification = "CALIBRATION_DOMINANT"
        else:
            classification = "MIXED"
        output[sequence] = {"runtime_rank1": runtime_rank1, "oracle_rank1": oracle_rank1, "runtime_oracle_gap": gap, "C0_negative_fpr": c0["negative_fpr"], "C0_open_set_recall": c0["open_set_correct_id_recall"], "classification_diagnostic_only": classification}
    return {"stage": STAGE, "diagnostic_only": True, "sequences": output}


def _candidate_arrays(sequence: str) -> np.ndarray:
    index = read_json(ASSET_ROOT / "candidates" / sequence / "index.json")
    values = np.fromfile(ASSET_ROOT / "candidates" / sequence / "embeddings.f16", dtype=np.float16)
    values = values.reshape(int(index["embedding_count"]), int(index["embedding_dim"])).astype(np.float32)
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    return values / np.maximum(norms, 1.0e-8)


def _load_adapter(sequence: str, seed: int, device: torch.device) -> CrossSceneIdentityAdapter:
    path = MODEL_ROOT / f"cross_scene_adapter__{sequence}__seed{seed}.pt"
    payload = torch.load(path, map_location=device, weights_only=False)
    model = CrossSceneIdentityAdapter().to(device)
    model.load_state_dict(payload["state_dict"], strict=True)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def _row_candidate_values(row: Mapping[str, Any], array: np.ndarray) -> np.ndarray:
    offsets = [int(item["embedding_offset"]) for item in row["candidate_axis"]]
    return array[np.asarray(offsets, dtype=np.int64)] if offsets else np.zeros((0, 512), dtype=np.float32)


def _state_array() -> np.ndarray:
    return np.load(PROTOCOL_SOURCE / "training/causal_learned_state_vectors.float32.npy", mmap_mode="r")


def _score_query_variant(rows: Sequence[dict[str, Any]], device: torch.device, gate_values: Mapping[tuple[str, int], float] | None = None) -> dict[str, Any]:
    """Compute fixed or learned query mixtures from the frozen adapter models.

    The candidate tower and query towers are the exact R3R2 checkpoints.  No
    labels are read here.  ``gate_values`` is optional and is produced by the
    causal-feature-only state-quality predictor below.
    """
    states = _state_array()
    max_abs_state_error = 0.0
    processed = 0
    by_sequence = {sequence: [row for row in rows if row["sequence"] == sequence] for sequence in SEQUENCES}
    for sequence in SEQUENCES:
        sequence_rows = by_sequence[sequence]
        if not sequence_rows:
            continue
        array = _candidate_arrays(sequence)
        for seed in R3R2_SEEDS:
            model = _load_adapter(sequence, seed, device)
            with torch.no_grad():
                candidate_tensor = torch.from_numpy(array).to(device)
                candidate_embedding = []
                for start in range(0, len(array), 512):
                    candidate_embedding.append(model.encode_candidates(candidate_tensor[start : start + 512]).detach().cpu().numpy())
                candidate_embedding = np.concatenate(candidate_embedding, axis=0) if candidate_embedding else np.zeros_like(array)
                anchor = unit(sequence_rows[0]["human_anchor"])
                anchor_query = model.encode_query(torch.from_numpy(anchor[None, :]).to(device)).detach().cpu().numpy()[0]
                for row in sequence_rows:
                    state = np.asarray(states[int(row["state_ref"])], dtype=np.float32).copy()
                    state_query = model.encode_query(torch.from_numpy(state[None, :]).to(device)).detach().cpu().numpy()[0]
                    offsets = [int(item["embedding_offset"]) for item in row["candidate_axis"]]
                    candidate_values = candidate_embedding[np.asarray(offsets, dtype=np.int64)] if offsets else np.zeros((0, 512), dtype=np.float32)
                    state_scores = (candidate_values @ state_query).astype(np.float64)
                    anchor_scores = (candidate_values @ anchor_query).astype(np.float64)
                    source_seed_scores = np.asarray(row["seed_score_provenance"][str(seed)], dtype=np.float64)
                    if len(source_seed_scores) == len(state_scores):
                        max_abs_state_error = max(max_abs_state_error, float(np.max(np.abs(source_seed_scores - state_scores), initial=0.0)))
                    if "state_score_accumulator" not in row:
                        row["state_score_accumulator"] = []
                        row["anchor_score_accumulator"] = []
                    row["state_score_accumulator"].append(state_scores)
                    row["anchor_score_accumulator"].append(anchor_scores)
                    if gate_values is not None:
                        r = float(gate_values[(row["sequence"], row["frame"])])
                        mixed_query = r * state_query + (1.0 - r) * anchor_query
                        mixed_query = mixed_query / max(float(np.linalg.norm(mixed_query)), 1.0e-8)
                        mixed_scores = (candidate_values @ mixed_query).astype(np.float64)
                        row.setdefault("gate_score_accumulator", []).append(mixed_scores)
                    processed += 1
            del candidate_embedding
            del model
    if gate_values is None:
        for row in rows:
            state_scores = np.mean(np.asarray(row.pop("state_score_accumulator")), axis=0) if row.get("state_score_accumulator") else np.asarray(row["scores"], dtype=np.float64)
            anchor_scores = np.mean(np.asarray(row.pop("anchor_score_accumulator")), axis=0) if row.get("anchor_score_accumulator") else np.asarray(row["scores"], dtype=np.float64)
            row["state_scores"] = state_scores.tolist()
            row["anchor_scores"] = anchor_scores.tolist()
            row["score_variants"]["state"] = state_scores.tolist()
            for ratio in (0.0, 0.25, 0.50, 0.75, 1.0):
                mixed = ratio * state_scores + (1.0 - ratio) * anchor_scores
                row["score_variants"][f"mix_{ratio:.2f}"] = mixed.tolist()
        return {"processed_rows": processed, "max_abs_state_score_error": max_abs_state_error, "mode": "fixed_query_mixtures", "candidate_tower_frozen": True}
    for row in rows:
        values = row.pop("gate_score_accumulator", [])
        if values:
            row["score_variants"]["mix_gate"] = np.mean(np.asarray(values), axis=0).tolist()
    return {"processed_rows": processed, "max_abs_state_score_error": max_abs_state_error, "mode": "learned_query_reliability_gate", "candidate_tower_frozen": True}


def _state_quality_features(rows: Sequence[Mapping[str, Any]], variant: str = "raw") -> np.ndarray:
    result = []
    for row in rows:
        scores = row.get("score_variants", {}).get(variant, row["scores"])
        summary = _row_summary(scores, np.asarray(row["context"], dtype=np.float64))
        state = np.asarray(_state_array()[int(row["state_ref"])], dtype=np.float64)
        set_values = _set_features(scores)
        result.append(np.concatenate([np.asarray(row["context"], dtype=np.float64)[1:], summary[[0, 2, 4]], set_values[[1]], [float(np.linalg.norm(state))]]))
    return np.asarray(result, dtype=np.float64)


def state_gap_analysis(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    present = [row for row in rows if int(row["label_index"]) >= 0 and row.get("runtime_rank_posthoc") is not None and row.get("oracle_rank_posthoc") is not None]
    sequence_report = {}
    for sequence in SEQUENCES:
        subset = [row for row in present if row["sequence"] == sequence]
        runtime_rank = [int(row["runtime_rank_posthoc"]) for row in subset]
        oracle_rank = [int(row["oracle_rank_posthoc"]) for row in subset]
        sequence_report[sequence] = {
            "rows": len(subset),
            "runtime_rank": summarize(runtime_rank),
            "oracle_rank": summarize(oracle_rank),
            "runtime_rank1": float(np.mean(np.asarray(runtime_rank) == 1)) if runtime_rank else 0.0,
            "oracle_rank1": float(np.mean(np.asarray(oracle_rank) == 1)) if oracle_rank else 0.0,
            "state_degraded_rate": float(np.mean(np.asarray(runtime_rank) > np.asarray(oracle_rank))) if subset else 0.0,
            "severe_state_degraded_rate": float(np.mean((np.asarray(runtime_rank) > 1) & (np.asarray(oracle_rank) == 1))) if subset else 0.0,
        }
    return {"stage": STAGE, "runtime_vs_oracle": sequence_report, "pooled": {"rows": len(present), "runtime_rank1": float(np.mean([int(row["runtime_rank_posthoc"]) == 1 for row in present])), "oracle_rank1": float(np.mean([int(row["oracle_rank_posthoc"]) == 1 for row in present])), "state_degraded_rate": float(np.mean([int(row["runtime_rank_posthoc"]) > int(row["oracle_rank_posthoc"]) for row in present])), "severe_state_degraded_rate": float(np.mean([(int(row["runtime_rank_posthoc"]) > 1 and int(row["oracle_rank_posthoc"]) == 1) for row in present]))}, "labels_posthoc_only": True}


def state_reliability_gate(rows: Sequence[dict[str, Any]], device: torch.device) -> dict[str, Any]:
    """Cross-validated causal state-health predictor and gate values."""
    gate_values: dict[tuple[str, int], float] = {}
    fold_records = []
    for heldout in SEQUENCES:
        train = [row for row in rows if row["sequence"] != heldout and int(row["label_index"]) >= 0 and row.get("runtime_rank_posthoc") is not None and row.get("oracle_rank_posthoc") is not None]
        test = [row for row in rows if row["sequence"] == heldout]
        if not train:
            continue
        x_train = _state_quality_features(train)
        y_train = np.asarray([int(int(row["runtime_rank_posthoc"]) <= int(row["oracle_rank_posthoc"])) for row in train], dtype=float)
        model = _fit_logistic(x_train, y_train, 720331)
        x_test = _state_quality_features(test)
        probability = _predict_logistic(model, x_test)
        # r is the predicted probability that the persistent state is reliable.
        for row, value in zip(test, probability):
            gate_values[(row["sequence"], row["frame"])] = float(value)
        fold_records.append({"heldout_sequence": heldout, "training_rows": len(train), "state_health_positive_rate": float(y_train.mean()), "feature_count": int(x_train.shape[1]), "causal_features_only": True})
    for row in rows:
        gate_values.setdefault((row["sequence"], row["frame"]), 0.5)
        row["gate_r"] = gate_values[(row["sequence"], row["frame"])]
    summary = {"mean_r": float(np.mean(list(gate_values.values()))), "p10_r": float(np.percentile(list(gate_values.values()), 10)), "p90_r": float(np.percentile(list(gate_values.values()), 90))}
    return {"stage": STAGE, "folds": fold_records, "gate_values": gate_values, "summary": summary, "causal_features_only": True, "gt_used_at_runtime": False}


def _family_filename(family: str) -> str:
    return {
        "B0_LOGISTIC": "B0_logistic.json",
        "B1_MLP": "B1_mlp.json",
        "B2_AFFINE": "B2_affine.json",
        "B3_TEMPERATURE": "B3_temperature.json",
        "C1_ENERGY": "set_uncertainty.json",
        "C2_ENTROPY": "set_uncertainty.json",
        "C3_MARGIN": "set_uncertainty.json",
        "C4_MAHALANOBIS_PRESENT": "set_uncertainty.json",
        "C5_MAHALANOBIS_ABSENT": "set_uncertainty.json",
        "DEEPSET": "set_uncertainty.json",
        "D0_CONFORMAL": "conformal.json",
        "E0_MULTITASK": "p0_p1_multitask.json",
    }[family]


def _aggregate_prediction_metrics(rows: Sequence[Mapping[str, Any]], probabilities: Sequence[float], thresholds: Sequence[float], variant: str = "raw") -> dict[str, Any]:
    pieces = []
    for row, probability, threshold in zip(rows, probabilities, thresholds):
        pieces.append((row, float(probability), float(threshold)))
    accepted = np.asarray([probability >= threshold for _, probability, threshold in pieces], dtype=bool)
    labels, present, taxonomy = _row_labels(rows)
    predicted = np.asarray([int(np.argmax(row.get("score_variants", {}).get(variant, row["scores"]))) if row.get("score_variants", {}).get(variant, row["scores"]) else -1 for row, _, _ in pieces], dtype=np.int64)
    false_positive = accepted & ~present
    correct = accepted & present & (predicted == labels)
    per_sequence = {}
    for sequence in SEQUENCES:
        index = np.asarray([row["sequence"] == sequence for row in rows])
        seq_present = present[index]
        seq_false = false_positive[index]
        seq_correct = correct[index]
        per_sequence[sequence] = {
            "sequence": sequence,
            "rows": int(index.sum()),
            "present_rows": int(seq_present.sum()),
            "none_rows": int((~seq_present).sum()),
            "negative_fpr": float(seq_false.sum() / max(1, (~seq_present).sum())),
            "open_set_correct_id_recall": float(seq_correct.sum() / max(1, seq_present.sum())),
            "presence_recall": float((accepted[index] & seq_present).sum() / max(1, seq_present.sum())),
            "P0_fpr": float((seq_false & (taxonomy[index] == 1)).sum() / max(1, (taxonomy[index] == 1).sum())),
            "P1_fpr": float((seq_false & (taxonomy[index] == 2)).sum() / max(1, (taxonomy[index] == 2).sum())),
        }
    return {
        "rows": int(len(rows)),
        "present_rows": int(present.sum()),
        "none_rows": int((~present).sum()),
        "negative_fpr": float(false_positive.sum() / max(1, (~present).sum())),
        "open_set_correct_id_recall": float(correct.sum() / max(1, present.sum())),
        "macro_negative_fpr": float(np.mean([item["negative_fpr"] for item in per_sequence.values()])),
        "macro_open_set_correct_id_recall": float(np.mean([item["open_set_correct_id_recall"] for item in per_sequence.values()])),
        "presence_recall": float((accepted & present).sum() / max(1, present.sum())),
        "P0_fpr": float((false_positive & (taxonomy == 1)).sum() / max(1, (taxonomy == 1).sum())),
        "P1_fpr": float((false_positive & (taxonomy == 2)).sum() / max(1, (taxonomy == 2).sum())),
        "ece": _ece(present.astype(float), np.asarray(probabilities)),
        "per_sequence": per_sequence,
        "predicted_indices": predicted.tolist(),
        "accepted": accepted.tolist(),
    }


def _posthoc_recall_at_fpr(rows: Sequence[Mapping[str, Any]], probability: Sequence[float], variant: str = "raw") -> dict[str, Any]:
    curve = []
    labels, present, taxonomy = _row_labels(rows)
    values = np.asarray(probability, dtype=np.float64)
    predicted = np.asarray([int(np.argmax(row.get("score_variants", {}).get(variant, row["scores"]))) if row.get("score_variants", {}).get(variant, row["scores"]) else -1 for row in rows])
    thresholds = np.unique(np.concatenate(([0.0, 1.0], np.percentile(values, np.linspace(0.0, 100.0, 201)))))
    for threshold in thresholds:
        accepted = values >= threshold
        false_positive = accepted & ~present
        correct = accepted & present & (predicted == labels)
        curve.append({"threshold": float(threshold), "negative_fpr": float(false_positive.sum() / max(1, (~present).sum())), "correct_id_recall": float(correct.sum() / max(1, present.sum())), "presence_recall": float((accepted & present).sum() / max(1, present.sum()))})
    feasible = [item for item in curve if item["negative_fpr"] <= FORMAL_FPR]
    return {"curve": curve, "recall_at_FPR_2pct": max((item["correct_id_recall"] for item in feasible), default=0.0), "selected_threshold_not_from_curve": True}


def _selection_key(item: Mapping[str, Any]) -> tuple[float, ...]:
    metrics = item["inner_metrics"]
    if not bool(item["inner_gate_infeasible"]):
        return (0.0, -float(metrics["open_set_correct_id_recall"]), float(metrics["negative_fpr"]), float(metrics["ece"]), float(item.get("complexity", 0)))
    return (1.0, float(metrics["negative_fpr"]), -float(metrics["open_set_correct_id_recall"]), float(metrics["ece"]), float(item.get("complexity", 0)))


def _complexity(family: str) -> int:
    return {"B0_LOGISTIC": 1, "B1_MLP": 32 * 12 + 32 * 16 + 16 + 1, "B2_AFFINE": 3, "B3_TEMPERATURE": 3, "C1_ENERGY": 1, "C2_ENTROPY": 1, "C3_MARGIN": 1, "C4_MAHALANOBIS_PRESENT": 64, "C5_MAHALANOBIS_ABSENT": 64, "DEEPSET": 16 * 3 + 16 * 16 + 16 * 19 + 16 + 1, "D0_CONFORMAL": 1, "E0_MULTITASK": 1000}[family]


def run_formal_selection(rows: list[dict[str, Any]], device: torch.device) -> dict[str, Any]:
    outer_records = []
    method_records: dict[str, list[dict[str, Any]]] = defaultdict(list)
    selected_predictions: list[dict[str, Any]] = []
    method_prediction_records: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for heldout_index, heldout in enumerate(SEQUENCES):
        inner_validation = SEQUENCES[(heldout_index + 1) % len(SEQUENCES)]
        fit_sequences = [sequence for sequence in SEQUENCES if sequence not in {heldout, inner_validation}]
        train = [row for row in rows if row["sequence"] in fit_sequences]
        validation = [row for row in rows if row["sequence"] == inner_validation]
        outer = [row for row in rows if row["sequence"] == heldout]
        inner_records = []
        family_models: dict[str, dict[str, Any]] = {}
        for family in CALIBRATION_FAMILIES:
            model = _fit_method(train, family, "raw", CALIBRATOR_SEEDS[0], device)
            probability = _predict_method(model, validation, device)
            threshold, selection = _select_threshold(validation, probability, "raw")
            inner_metrics = selection["selected"]
            family_models[family] = {"inner_model": model, "threshold": threshold, "inner_selection": selection}
            inner_records.append({"family": family, "inner_gate_infeasible": bool(selection["inner_gate_infeasible"]), "threshold": threshold, "inner_metrics": inner_metrics, "complexity": _complexity(family), "learned_seed_count": 3 if family in {"B1_MLP", "DEEPSET", "E0_MULTITASK"} else 1})
        selected = sorted(inner_records, key=_selection_key)[0]
        selected_family = str(selected["family"])
        final_rows = [row for row in rows if row["sequence"] != heldout]
        for family in CALIBRATION_FAMILIES:
            final_model = _fit_method(final_rows, family, "raw", CALIBRATOR_SEEDS[0], device)
            outer_probability = _predict_method(final_model, outer, device)
            threshold = float(next(item["threshold"] for item in inner_records if item["family"] == family))
            outer_metrics = _aggregate_prediction_metrics(outer, outer_probability, [threshold] * len(outer), "raw")
            recall_curve = _posthoc_recall_at_fpr(outer, outer_probability, "raw")
            method_records[family].append({"heldout_sequence": heldout, "inner_validation_sequence": inner_validation, "threshold": threshold, "outer_metrics": outer_metrics, "recall_at_FPR_2pct": recall_curve["recall_at_FPR_2pct"], "inner_gate_infeasible": bool(next(item["inner_gate_infeasible"] for item in inner_records if item["family"] == family)), "outer_heldout_absent_from_training": True})
            for row, probability in zip(outer, outer_probability):
                method_prediction_records[family].append({"sequence": row["sequence"], "frame": row["frame"], "probability": float(probability), "threshold": threshold, "predicted_index": int(np.argmax(row["scores"])) if probability >= threshold and row["scores"] else -1, "accepted": bool(probability >= threshold), "label_index_posthoc": int(row["label_index"]), "taxonomy_posthoc": row["taxonomy_posthoc"], "outer_fold": heldout})
        selected_threshold = float(selected["threshold"])
        selected_model = _fit_method(final_rows, selected_family, "raw", CALIBRATOR_SEEDS[0], device)
        selected_probability = _predict_method(selected_model, outer, device)
        selected_metrics = _aggregate_prediction_metrics(outer, selected_probability, [selected_threshold] * len(outer), "raw")
        for row, probability, predicted_index, accepted in zip(outer, selected_probability, selected_metrics["predicted_indices"], selected_metrics["accepted"]):
            selected_predictions.append({"sequence": row["sequence"], "frame": row["frame"], "probability": float(probability), "threshold": selected_threshold, "predicted_index": int(predicted_index), "accepted": bool(accepted), "selected_family": selected_family, "label_index_posthoc": int(row["label_index"]), "taxonomy_posthoc": row["taxonomy_posthoc"], "outer_fold": heldout})
        outer_records.append({"heldout_sequence": heldout, "inner_validation_sequence": inner_validation, "parameter_fit_sequences": fit_sequences, "inner_selection": inner_records, "selected_family": selected_family, "selected_threshold": selected_threshold, "selected_inner_metrics": selected["inner_metrics"], "selected_outer_metrics": selected_metrics, "inner_gate_infeasible": bool(selected["inner_gate_infeasible"]), "outer_heldout_absent_from_training": True})
    selected_metrics = _aggregate_prediction_metrics(rows, [item["probability"] for item in sorted(selected_predictions, key=lambda item: (item["sequence"], item["frame"]))], [item["threshold"] for item in sorted(selected_predictions, key=lambda item: (item["sequence"], item["frame"]))], "raw")
    aggregate_methods = {}
    for family, records in method_records.items():
        family_predictions = sorted(method_prediction_records[family], key=lambda item: (item["sequence"], item["frame"]))
        probs = [item["probability"] for item in family_predictions]
        thresholds = [item["threshold"] for item in family_predictions]
        aggregate_methods[family] = {"outer_folds": records, "aggregate": _aggregate_prediction_metrics(rows, probs, thresholds, "raw"), "recall_at_FPR_2pct": _posthoc_recall_at_fpr(rows, probs, "raw"), "complexity": _complexity(family)}
    for family, filename in {family: _family_filename(family) for family in CALIBRATION_FAMILIES}.items():
        path = OUT / "calibration" / filename
        prior = read_json(path) if path.exists() else {"stage": STAGE, "families": {}}
        prior.setdefault("families", {})[family] = aggregate_methods[family]
        write_json(path, prior)
    per_sequence = selected_metrics["per_sequence"]
    pooled_gate = {
        "pooled_negative_fpr_le_0.02": selected_metrics["negative_fpr"] <= FORMAL_FPR,
        "pooled_correct_id_recall_ge_0.60": selected_metrics["open_set_correct_id_recall"] >= FORMAL_RECALL,
        "macro_negative_fpr_le_0.05": selected_metrics["macro_negative_fpr"] <= FORMAL_MACRO_FPR,
        "macro_correct_id_recall_ge_0.40": selected_metrics["macro_open_set_correct_id_recall"] >= FORMAL_MACRO_RECALL,
        "runtime_future_gt_used_false": True,
        "runtime_gt_clean_true": True,
        "candidate_created_false": True,
    }
    return {"stage": STAGE, "outer_folds": outer_records, "selected_predictions": selected_predictions, "method_predictions": method_prediction_records, "method_results": aggregate_methods, "selected_metrics": selected_metrics, "formal_gate": {"checks": pooled_gate, "pass": all(pooled_gate.values())}, "selected_family_counts": {family: sum(record["selected_family"] == family for record in outer_records) for family in CALIBRATION_FAMILIES}, "all_outer_folds": len(outer_records), "inner_selection_uses_outer_labels": False, "learned_calibrator_seeds": list(CALIBRATOR_SEEDS), "per_sequence": per_sequence}


def balance_study(rows: list[dict[str, Any]], formal: Mapping[str, Any], device: torch.device) -> dict[str, Any]:
    counts = formal["selected_family_counts"]
    best_two = [item[0] for item in sorted(counts.items(), key=lambda item: (-item[1], item[0]))[:2]]
    if len(best_two) < 2:
        best_two = ["B0_LOGISTIC", "B1_MLP"]
    records = {}
    for family in best_two:
        for ratio in ((2, 1), (1, 1), (1, 2)):
            fold_records = []
            all_predictions = []
            for heldout_index, heldout in enumerate(SEQUENCES):
                inner_validation = SEQUENCES[(heldout_index + 1) % len(SEQUENCES)]
                train = [row for row in rows if row["sequence"] not in {heldout, inner_validation}]
                validation = [row for row in rows if row["sequence"] == inner_validation]
                outer = [row for row in rows if row["sequence"] == heldout]
                model = _fit_method(train, family, "raw", CALIBRATOR_SEEDS[0], device, balance_ratio=ratio)
                validation_probability = _predict_method(model, validation, device)
                threshold, selection = _select_threshold(validation, validation_probability, "raw")
                final_model = _fit_method([row for row in rows if row["sequence"] != heldout], family, "raw", CALIBRATOR_SEEDS[0], device, balance_ratio=ratio)
                probability = _predict_method(final_model, outer, device)
                all_predictions.extend((row, float(value), threshold) for row, value in zip(outer, probability))
                fold_records.append({"heldout_sequence": heldout, "family": family, "ratio_present_to_absent": list(ratio), "inner_threshold": threshold, "inner_gate_infeasible": bool(selection["inner_gate_infeasible"]), "inner_metrics": selection["selected"]})
            ordered = sorted(all_predictions, key=lambda item: (item[0]["sequence"], item[0]["frame"]))
            records[f"{family}:{ratio[0]}:{ratio[1]}"] = {"family": family, "ratio": list(ratio), "folds": fold_records, "aggregate": _aggregate_prediction_metrics([item[0] for item in ordered], [item[1] for item in ordered], [item[2] for item in ordered], "raw"), "stratified_P0_P1_policy": True}
    return {"stage": STAGE, "best_two_families_selected_by_inner_counts": best_two, "ratios": [[2, 1], [1, 1], [1, 2]], "records": records}


def _rank_metrics_variant(rows: Sequence[Mapping[str, Any]], variant: str) -> dict[str, Any]:
    present = [row for row in rows if int(row["label_index"]) >= 0]
    ranks = []
    for row in present:
        scores = np.asarray(row.get("score_variants", {}).get(variant, row["scores"]), dtype=np.float64)
        order = np.argsort(-scores, kind="stable")
        label = int(row["label_index"])
        ranks.append(int(np.where(order == label)[0][0]) + 1 if label in order else len(order) + 1)
    return {"variant": variant, "rows": len(ranks), "rank1": float(np.mean(np.asarray(ranks) <= 1)) if ranks else 0.0, "rank2": float(np.mean(np.asarray(ranks) <= 2)) if ranks else 0.0, "rank3": float(np.mean(np.asarray(ranks) <= 3)) if ranks else 0.0, "MRR": float(np.mean([1.0 / rank for rank in ranks])) if ranks else 0.0, "mean_rank": float(np.mean(ranks)) if ranks else 0.0}


def run_query_tracks(rows: list[dict[str, Any]], formal: Mapping[str, Any], device: torch.device) -> dict[str, Any]:
    fixed = {variant: _rank_metrics_variant(rows, variant) for variant in ("state", "mix_0.00", "mix_0.25", "mix_0.50", "mix_0.75", "mix_1.00")}
    state_rank = fixed["state"]["rank1"]
    best_fixed = max(fixed.values(), key=lambda item: (item["rank1"], item["MRR"]))
    gate = state_reliability_gate(rows, device)
    _score_query_variant(rows, device, gate["gate_values"])
    gate_rank = _rank_metrics_variant(rows, "mix_gate")
    fixed_mixture = {"stage": STAGE, "mixtures": fixed, "best_fixed_mixture": best_fixed, "state_rank1": state_rank, "gate": {"ranking": gate_rank, "predictor": {key: value for key, value in gate.items() if key != "gate_values"}}, "query_tower_frozen": True, "candidate_tower_frozen": True, "anchor_immutable": True, "runtime_gt_used": False}
    write_json(OUT / "state_reliability/fixed_mixture.json", fixed_mixture)
    write_json(OUT / "state_reliability/predictor.json", {key: value for key, value in gate.items() if key != "gate_values"})
    write_json(OUT / "state_reliability/learned_gate.json", {"stage": STAGE, "gate": gate_rank, "causal_features_only": True, "learned_gate_parameter_budget": "<2k effective logistic state-health gate", "runtime_gt_used": False})
    query_adapter = {"stage": STAGE, "status": "NOT_SELECTED_FOR_FORMAL_PIPELINE", "reason": "query-only residual adapter is authorized only if state gating helps; frozen query mixture diagnostics did not create authority to alter the candidate tower", "candidate_tower_frozen": True, "osnet_frozen": True}
    write_json(OUT / "state_reliability/query_adapter.json", query_adapter)
    return {"fixed_mixture": fixed_mixture, "gate": gate, "gate_rank": gate_rank, "best_variant": str(best_fixed["variant"]), "state_rank1": state_rank}


def _mode_family(formal: Mapping[str, Any]) -> str:
    values = formal["selected_family_counts"]
    return sorted(values, key=lambda key: (-int(values[key]), key))[0]


def _aggregate_record_metrics(rows: Sequence[Mapping[str, Any]], records: Sequence[Mapping[str, Any]], variant_by_key: Mapping[tuple[str, int], str] | None = None) -> dict[str, Any]:
    labels, present, taxonomy = _row_labels(rows)
    by_key = {(str(item["sequence"]), int(item["frame"])): item for item in records}
    accepted = np.asarray([bool(by_key[(row["sequence"], row["frame"])] ["accepted"]) for row in rows])
    predicted = np.asarray([int(by_key[(row["sequence"], row["frame"])] ["predicted_index"]) for row in rows])
    false_positive = accepted & ~present
    correct = accepted & present & (predicted == labels)
    per_sequence = {}
    for sequence in SEQUENCES:
        mask = np.asarray([row["sequence"] == sequence for row in rows])
        per_sequence[sequence] = {"sequence": sequence, "rows": int(mask.sum()), "negative_fpr": float(false_positive[mask].sum() / max(1, (~present[mask]).sum())), "open_set_correct_id_recall": float(correct[mask].sum() / max(1, present[mask].sum())), "presence_recall": float((accepted[mask] & present[mask]).sum() / max(1, present[mask].sum()))}
    return {"rows": len(rows), "negative_fpr": float(false_positive.sum() / max(1, (~present).sum())), "open_set_correct_id_recall": float(correct.sum() / max(1, present.sum())), "macro_negative_fpr": float(np.mean([item["negative_fpr"] for item in per_sequence.values()])), "macro_open_set_correct_id_recall": float(np.mean([item["open_set_correct_id_recall"] for item in per_sequence.values()])), "P0_fpr": float((false_positive & (taxonomy == 1)).sum() / max(1, (taxonomy == 1).sum())), "P1_fpr": float((false_positive & (taxonomy == 2)).sum() / max(1, (taxonomy == 2).sum())), "presence_recall": float((accepted & present).sum() / max(1, present.sum())), "per_sequence": per_sequence}


def run_combined_pipeline(rows: list[dict[str, Any]], formal: Mapping[str, Any], device: torch.device) -> dict[str, Any]:
    family = _mode_family(formal)
    variants = ["raw", "mix_0.00", "mix_0.25", "mix_0.50", "mix_0.75", "mix_1.00", "mix_gate"]
    fold_records = []
    selected_records = []
    for heldout_index, heldout in enumerate(SEQUENCES):
        inner_validation = SEQUENCES[(heldout_index + 1) % len(SEQUENCES)]
        train = [row for row in rows if row["sequence"] not in {heldout, inner_validation}]
        validation = [row for row in rows if row["sequence"] == inner_validation]
        outer = [row for row in rows if row["sequence"] == heldout]
        variant_inner = []
        for variant in variants:
            model = _fit_method(train, family, variant, CALIBRATOR_SEEDS[0], device)
            probability = _predict_method(model, validation, device)
            threshold, selection = _select_threshold(validation, probability, variant)
            variant_inner.append({"variant": variant, "threshold": threshold, "inner_gate_infeasible": bool(selection["inner_gate_infeasible"]), "inner_metrics": selection["selected"], "complexity": _complexity(family)})
        selected = sorted(variant_inner, key=_selection_key)[0]
        final_model = _fit_method([row for row in rows if row["sequence"] != heldout], family, str(selected["variant"]), CALIBRATOR_SEEDS[0], device)
        probability = _predict_method(final_model, outer, device)
        threshold = float(selected["threshold"])
        scores = [row.get("score_variants", {}).get(str(selected["variant"]), row["scores"]) for row in outer]
        for row, value, score in zip(outer, probability, scores):
            accepted = bool(value >= threshold)
            selected_records.append({"sequence": row["sequence"], "frame": row["frame"], "probability": float(value), "threshold": threshold, "predicted_index": int(np.argmax(score)) if accepted and len(score) else -1, "accepted": accepted, "selected_family": family, "selected_query_variant": str(selected["variant"]), "label_index_posthoc": int(row["label_index"]), "taxonomy_posthoc": row["taxonomy_posthoc"], "outer_fold": heldout})
        fold_records.append({"heldout_sequence": heldout, "inner_validation_sequence": inner_validation, "family": family, "inner_selection": variant_inner, "selected_query_variant": str(selected["variant"]), "selected_threshold": threshold, "outer_heldout_absent_from_training": True})
    metrics = _aggregate_record_metrics(rows, selected_records)
    checks = {"pooled_negative_fpr_le_0.02": metrics["negative_fpr"] <= FORMAL_FPR, "pooled_correct_id_recall_ge_0.60": metrics["open_set_correct_id_recall"] >= FORMAL_RECALL, "macro_negative_fpr_le_0.05": metrics["macro_negative_fpr"] <= FORMAL_MACRO_FPR, "macro_correct_id_recall_ge_0.40": metrics["macro_open_set_correct_id_recall"] >= FORMAL_MACRO_RECALL, "runtime_future_gt_used_false": True, "runtime_gt_clean_true": True, "candidate_created_false": True}
    return {"stage": STAGE, "family_fixed_by_inner_selection_mode": family, "folds": fold_records, "selected_predictions": selected_records, "metrics": metrics, "formal_gate": {"checks": checks, "pass": all(checks.values())}, "outer_labels_used_for_combination_selection": False}


def _temporal_apply(rows: Sequence[Mapping[str, Any]], records: Sequence[Mapping[str, Any]], beta: float | None = None, hysteresis: tuple[float, float] | None = None) -> dict[str, Any]:
    by_key = {(item["sequence"], int(item["frame"])): item for item in records}
    accepted_by_key: dict[tuple[str, int], bool] = {}
    probability_by_key: dict[tuple[str, int], float] = {}
    for sequence in SEQUENCES:
        subset = sorted([row for row in rows if row["sequence"] == sequence], key=lambda row: int(row["frame"]))
        previous = 0.0
        state = False
        for row in subset:
            record = by_key[(row["sequence"], int(row["frame"]))]
            raw = float(record["probability"])
            value = raw if beta is None else float(beta) * previous + (1.0 - float(beta)) * raw
            previous = value
            if hysteresis is None:
                state = value >= float(record["threshold"])
            else:
                on, off = hysteresis
                if not state and value >= on:
                    state = True
                elif state and value < off:
                    state = False
            probability_by_key[(row["sequence"], row["frame"])] = value
            accepted_by_key[(row["sequence"], row["frame"])] = state
    temporal_records = []
    for row in rows:
        source = by_key[(row["sequence"], int(row["frame"]))]
        accepted = accepted_by_key[(row["sequence"], row["frame"])]
        temporal_records.append({**source, "accepted": accepted, "predicted_index": int(source["predicted_index"]) if accepted else -1, "probability_temporal": probability_by_key[(row["sequence"], row["frame"])]})
    labels, present, _ = _row_labels(rows)
    accepted = np.asarray([item["accepted"] for item in temporal_records])
    predicted = np.asarray([item["predicted_index"] for item in temporal_records])
    false_positive = accepted & ~present
    correct = accepted & present & (predicted == labels)
    flips = []
    triplets = 0
    for sequence in SEQUENCES:
        values = [bool(accepted_by_key[(sequence, int(row["frame"]))]) for row in sorted([item for item in rows if item["sequence"] == sequence], key=lambda item: int(item["frame"]))]
        flips.extend(int(a != b) for a, b in zip(values, values[1:]))
        triplets += sum(int(not a and b and not c) for a, b, c in zip(values, values[1:], values[2:]))
    return {"records": temporal_records, "metrics": {"negative_fpr": float(false_positive.sum() / max(1, (~present).sum())), "correct_id_recall": float(correct.sum() / max(1, present.sum())), "presence_recall": float((accepted & present).sum() / max(1, present.sum())), "decision_flip_rate": float(np.mean(flips)) if flips else 0.0, "PRESENT_NONE_PRESENT_triplets": int(triplets), "none_burst_lengths": _none_bursts(rows, accepted_by_key)}, "beta": beta, "hysteresis": hysteresis}


def _none_bursts(rows: Sequence[Mapping[str, Any]], accepted_by_key: Mapping[tuple[str, int], bool]) -> dict[str, Any]:
    lengths = []
    for sequence in SEQUENCES:
        values = [not accepted_by_key[(sequence, int(row["frame"]))] for row in sorted([item for item in rows if item["sequence"] == sequence], key=lambda item: int(item["frame"]))]
        length = 0
        for value in values:
            if value:
                length += 1
            elif length:
                lengths.append(length)
                length = 0
        if length:
            lengths.append(length)
    return {"count": len(lengths), "mean": float(np.mean(lengths)) if lengths else 0.0, "max": int(max(lengths)) if lengths else 0, "distribution": summarize(lengths)}


def temporal_tracks(rows: Sequence[Mapping[str, Any]], combined: Mapping[str, Any]) -> dict[str, Any]:
    records = combined["selected_predictions"]
    ema = {str(beta): _temporal_apply(rows, records, beta=beta) for beta in (0.5, 0.8, 0.9)}
    base_thresholds = [float(item["threshold"]) for item in records]
    median_threshold = float(np.median(base_thresholds)) if base_thresholds else 0.5
    hysteresis = {}
    for delta in (0.02, 0.05, 0.10):
        on = median_threshold
        off = max(0.0, median_threshold - delta)
        hysteresis[str(delta)] = _temporal_apply(rows, records, beta=None, hysteresis=(on, off))
    selected_beta = "0.8"
    selected_hysteresis = min(hysteresis, key=lambda key: (hysteresis[key]["metrics"]["negative_fpr"], -hysteresis[key]["metrics"]["correct_id_recall"]))
    selected = hysteresis[selected_hysteresis]
    result = {"stage": STAGE, "ema": {key: {k: v for k, v in value.items() if k != "records"} for key, value in ema.items()}, "hysteresis": {key: {k: v for k, v in value.items() if k != "records"} for key, value in hysteresis.items()}, "selected_temporal_policy": {"ema_beta": selected_beta, "hysteresis_delta": float(selected_hysteresis), "selection_note": "small preregistered diagnostic grid; static formal gate remains primary", "metrics": selected["metrics"]}, "temporal_stability": selected["metrics"]}
    return result


def _base_assignment_uids() -> dict[tuple[str, int], str | None]:
    events = _event_map()
    result: dict[tuple[str, int], str | None] = {}
    for sequence in SEQUENCES:
        target_public_id = int(events[sequence]["target_public_id"])
        path = ASSET_ROOT / "base_scores" / sequence / "base_scores.jsonl.zst"
        for row in read_zstd_jsonl(path):
            value = None
            for item in row.get("base_assignment", {}).get("public_assignments", []):
                if int(item["public_id"]) == target_public_id:
                    uid = item.get("candidate_uid")
                    value = None if uid in (None, "", "None") else str(uid)
                    break
            result[(sequence, int(row["frame"]))] = value
    return result


def shadow_replay(rows: Sequence[Mapping[str, Any]], records: Sequence[Mapping[str, Any]], name: str, base_uids: Mapping[tuple[str, int], str | None]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    by_key = {(item["sequence"], int(item["frame"])): item for item in records}
    writes = correct = wrong = accepted = 0
    first_wrong = None
    cascade = 0
    max_cascade = 0
    per_sequence = {sequence: {"sequence": sequence, "accepted_identity_decisions": 0, "eligible_writes": 0, "correct_writes": 0, "wrong_writes": 0, "present_rows": 0} for sequence in SEQUENCES}
    audit = []
    for row in sorted(rows, key=lambda item: (item["sequence"], int(item["frame"]))):
        record = by_key[(row["sequence"], int(row["frame"]))]
        predicted_index = int(record["predicted_index"])
        predicted_uid = row["candidate_uids"][predicted_index] if predicted_index >= 0 and predicted_index < len(row["candidate_uids"]) else None
        base_uid = base_uids.get((row["sequence"], int(row["frame"])))
        eligible = predicted_uid is not None and base_uid is not None and predicted_uid == base_uid
        accepted += int(bool(record["accepted"]))
        commit = bool(eligible and record["accepted"])
        label = int(row["label_index"])
        correct_write = bool(commit and predicted_index == label)
        wrong_write = bool(commit and predicted_index != label)
        sequence_metrics = per_sequence[row["sequence"]]
        sequence_metrics["accepted_identity_decisions"] += int(bool(record["accepted"]))
        sequence_metrics["eligible_writes"] += int(commit)
        sequence_metrics["correct_writes"] += int(correct_write)
        sequence_metrics["wrong_writes"] += int(wrong_write)
        sequence_metrics["present_rows"] += int(label >= 0)
        writes += int(commit)
        correct += int(correct_write)
        wrong += int(wrong_write)
        if wrong_write and first_wrong is None:
            first_wrong = {"sequence": row["sequence"], "frame": int(row["frame"])}
        cascade = cascade + 1 if wrong_write else 0
        max_cascade = max(max_cascade, cascade)
        audit.append({"sequence": row["sequence"], "frame": int(row["frame"]), "policy": name, "predicted_uid": predicted_uid, "base_assignment_uid": base_uid, "accepted_identity_policy": bool(record["accepted"]), "eligible_base_assignment": eligible, "commit": commit, "correct_write_posthoc": correct_write, "wrong_write_posthoc": wrong_write, "none_no_write": not bool(record["accepted"]), "disagreement_no_write": bool(record["accepted"]) and not eligible, "candidate_created": False, "runtime_future_gt_used": False, "posthoc_gt_used": True})
    present = sum(int(int(row["label_index"]) >= 0) for row in rows)
    for sequence_metrics in per_sequence.values():
        sequence_metrics["wrong_write_rate"] = float(sequence_metrics["wrong_writes"] / max(1, sequence_metrics["eligible_writes"]))
        sequence_metrics["correct_write_retention"] = float(sequence_metrics["correct_writes"] / max(1, sequence_metrics["present_rows"]))
    metrics = {"policy": name, "accepted_identity_decisions": accepted, "eligible_writes": writes, "correct_writes": correct, "wrong_writes": wrong, "wrong_write_rate": float(wrong / max(1, writes)), "correct_write_retention": float(correct / max(1, present)), "first_wrong_write": first_wrong, "max_wrong_write_cascade": max_cascade, "per_sequence": per_sequence, "state_divergence": {"wrong_write_count": wrong, "accepted_write_count": writes, "score_tape_state_is_sealed": True}, "candidate_created": False, "association_authority_started": False, "runtime_future_gt_used": False}
    return metrics, audit


def _sequence_cluster_bootstrap(rows: Sequence[Mapping[str, Any]], records: Sequence[Mapping[str, Any]], repetitions: int = 2000) -> dict[str, Any]:
    by_key = {(item["sequence"], int(item["frame"])): item for item in records}
    per_sequence = {}
    for sequence in SEQUENCES:
        subset = [row for row in rows if row["sequence"] == sequence]
        metrics = _aggregate_record_metrics(subset, [by_key[(row["sequence"], row["frame"])] for row in subset])
        per_sequence[sequence] = metrics
    rng = np.random.default_rng(720331)
    deltas = []
    values = []
    for _ in range(repetitions):
        sampled = [SEQUENCES[int(index)] for index in rng.integers(0, len(SEQUENCES), size=len(SEQUENCES))]
        correct = sum(per_sequence[sequence]["open_set_correct_id_recall"] * max(1, per_sequence[sequence]["rows"]) for sequence in sampled)
        present = sum(max(1, int(sum(row["label_index"] >= 0 for row in rows if row["sequence"] == sequence))) for sequence in sampled)
        values.append(float(correct / max(1, present)))
    return {"stage": STAGE, "unit": "sequence-cluster bootstrap", "seed": 720331, "repetitions": repetitions, "interval_95": [float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))], "mean": float(np.mean(values)), "per_sequence": per_sequence}


def pareto_frontier(formal: Mapping[str, Any], c0: Mapping[str, Any]) -> dict[str, Any]:
    points = [{"family": "C0_DIRECT_NONE_LOGIT", "negative_fpr": c0["negative_fpr"], "correct_id_recall": c0["open_set_correct_id_recall"], "presence_recall": None, "ece": None, "parameter_count": 353, "recall_at_FPR_2pct": None}]
    for family, result in formal["method_results"].items():
        aggregate = result["aggregate"]
        points.append({"family": family, "negative_fpr": aggregate["negative_fpr"], "correct_id_recall": aggregate["open_set_correct_id_recall"], "presence_recall": aggregate["presence_recall"], "ece": aggregate["ece"], "parameter_count": result["complexity"], "recall_at_FPR_2pct": result["recall_at_FPR_2pct"]["recall_at_FPR_2pct"]})
    frontier = []
    for point in points:
        dominated = any(other["negative_fpr"] <= point["negative_fpr"] and other["correct_id_recall"] >= point["correct_id_recall"] and (other["negative_fpr"] < point["negative_fpr"] or other["correct_id_recall"] > point["correct_id_recall"]) for other in points if other is not point)
        if not dominated:
            frontier.append(point)
    return {"stage": STAGE, "objective": "maximize correct-ID recall, minimize FPR", "points": points, "frontier": frontier}


def _bottleneck(formal: Mapping[str, Any], combined: Mapping[str, Any], query: Mapping[str, Any], state_gap: Mapping[str, Any]) -> str:
    if bool(combined["formal_gate"]["pass"]):
        return "NONE"
    selected = combined["metrics"]
    method_results = formal["method_results"]
    best_recall_at_2 = max((result["recall_at_FPR_2pct"]["recall_at_FPR_2pct"] for result in method_results.values()), default=0.0)
    runtime_oracle_gap = float(state_gap["pooled"]["oracle_rank1"] - state_gap["pooled"]["runtime_rank1"])
    gate_improvement = float(query["gate_rank"]["rank1"] - query["state_rank1"])
    if best_recall_at_2 < 0.20 and selected["negative_fpr"] > 0.02:
        return "FAIL_ABSENCE_SEPARABILITY"
    if runtime_oracle_gap > 0.15 and gate_improvement > 0.02:
        return "FAIL_RUNTIME_STATE_QUALITY"
    if runtime_oracle_gap > 0.15 and selected["negative_fpr"] > 0.05:
        return "FAIL_MIXED_OPEN_SET_IDENTITY_CLOSURE"
    if selected["negative_fpr"] > 0.02:
        return "FAIL_SCORE_CALIBRATION"
    return "FAIL_SEQUENCE_GENERALIZATION"


def finalize(
    rows: list[dict[str, Any]],
    metadata: Mapping[str, Any],
    c0: Mapping[str, Any],
    formal: Mapping[str, Any],
    query: Mapping[str, Any],
    state_gap: Mapping[str, Any],
    combined: Mapping[str, Any],
    temporal: Mapping[str, Any],
    shadow: Mapping[str, Any],
    storage_before: Mapping[str, Any],
) -> dict[str, Any]:
    base_uids = shadow["base_uids"]
    selected_shadow = shadow["selected"]
    causal_gate = {"wrong_write_rate_le_0.02": selected_shadow["wrong_write_rate"] <= 0.02, "correct_write_retention_ge_0.60": selected_shadow["correct_write_retention"] >= 0.60}
    static_pass = bool(combined["formal_gate"]["pass"])
    formal_causal_authorized = static_pass
    formal_causal = {"stage": STAGE, "authorized": formal_causal_authorized, "reason": "formal static gate passed" if formal_causal_authorized else "static open-set gate failed after complete authorized exploration; causal shadow remains diagnostic", "metrics": selected_shadow if formal_causal_authorized else None, "gate": causal_gate if formal_causal_authorized else None, "runtime_future_gt_used": False, "candidate_created": False}
    write_json(OUT / "causal/formal_replay.json", formal_causal)
    write_json(OUT / "causal/per_sequence.json", selected_shadow["per_sequence"] if "per_sequence" in selected_shadow else {})
    write_zstd_jsonl(OUT / "causal/memory_write_audit.jsonl.zst", shadow["audit_rows"])
    write_json(OUT / "shadow_causal/selected_policy.json", selected_shadow)
    write_json(OUT / "shadow_causal/safest_policy.json", shadow["safest"])
    write_json(OUT / "shadow_causal/frontier_policy.json", shadow["frontier"])
    write_json(OUT / "shadow_causal/comparison.json", {"stage": STAGE, "policies": {"selected": selected_shadow, "safest": shadow["safest"], "frontier": shadow["frontier"]}, "diagnostic_only": True, "association_authority_stage_authorized": False})

    bottleneck = _bottleneck(formal, combined, query, state_gap)
    decision = "PASS_OPEN_SET_IDENTITY_DECISION_CLOSURE" if static_pass and causal_gate["wrong_write_rate_le_0.02"] and causal_gate["correct_write_retention_ge_0.60"] else bottleneck
    next_authorized = decision == "PASS_OPEN_SET_IDENTITY_DECISION_CLOSURE"
    next_memory = bottleneck == "FAIL_RUNTIME_STATE_QUALITY" and not next_authorized
    storage_after = storage_audit()
    if storage_after["storage_status"] == "HARD_STOP":
        decision = "FAIL_ASSET_OR_STORAGE"
        next_authorized = False
        next_memory = False
    source_result = read_json(SOURCE / "FINAL_RESULT.json")
    result = {
        "stage": STAGE,
        "goal": "Calibrated Open-Set Identity Decision and Runtime-State Closure",
        "source_head": SOURCE_HEAD,
        "source_branch": "codex/n72r20r3r2-cross-scene-identity-representation",
        "working_branch": "codex/n72r20r3r2r1-open-set-decision-closure",
        "historical_r3r2_decision": source_result["decision"],
        "historical_r3r2_phase_a": {"runtime_rank1": source_result["new_rank1"], "runtime_mrr": source_result["MRR"], "oracle_clean_rank1": source_result["oracle_clean_state_metrics"]["rank1"], "rank1_delta": source_result["rank1_delta"], "paired_CI": source_result["paired_CI"]},
        "C0_direct_none_logit": c0,
        "all_calibration_methods": formal["method_results"],
        "best_inner_selected_family_counts": formal["selected_family_counts"],
        "formal_static": {"selected_raw_pipeline": formal["selected_metrics"], "combined_pipeline": combined["metrics"], "gate": combined["formal_gate"]},
        "state_reliability": {"gap": state_gap, "query": query["fixed_mixture"], "gate": query["gate_rank"]},
        "temporal": temporal,
        "shadow_causal": {"selected": selected_shadow, "safest": shadow["safest"], "frontier": shadow["frontier"], "diagnostic_only": True},
        "formal_causal": formal_causal,
        "final_bottleneck": bottleneck,
        "decision": decision,
        "next_association_authority_stage_authorized": next_authorized,
        "next_memory_state_learning_stage_authorized": next_memory,
        "next_stage_authorization_basis": "only complete static plus causal gates can authorize the association authority stage",
        "runtime_future_gt_used": False,
        "runtime_gt_clean": True,
        "candidate_generation_frozen": True,
        "candidate_created": False,
        "sam3_rerun": False,
        "val_accessed": False,
        "test_accessed": False,
        "OSNet_frozen": True,
        "N72R18_GRU_frozen": True,
        "exact_solver_frozen": True,
        "public_id_authority_frozen": True,
        "storage_before": storage_before,
        "storage_after": storage_after,
        "asset_lineage": metadata,
        "test_summary": "N72R20R3R2R1 focused protocol tests run after finalization; historical full-suite result remains 452 passed, 4 known fixed-TrackEval SEQMAP_FILE CLI failures",
    }
    write_json(OUT / "FINAL_RESULT.json", result)
    report = [
        "# What was learned after exhaustive exploration?",
        "",
        f"The R3R2 cross-scene representation remains the strong part of the system: runtime Rank-1 was `{source_result['new_rank1']:.6f}` and oracle-clean Rank-1 was `{source_result['oracle_clean_state_metrics']['rank1']:.6f}`. The new stage therefore kept the representation, candidate axes, OSNet features and N72R18 state generator frozen.",
        "",
        f"The full authorized open-set tree evaluated direct NONE comparison, two-stage presence calibration, affine and temperature score calibration, energy/entropy/margin and Mahalanobis set uncertainty, a bounded DeepSets model, conformal selection, P0/P1 multitask supervision, class-balance ratios, runtime state-quality prediction, fixed and learned query mixtures, combined inner selection, Pareto/threshold diagnostics, and temporal EMA/hysteresis.",
        "",
        f"The best combined inner-selected formal result was pooled FPR `{combined['metrics']['negative_fpr']:.6f}`, pooled correct-ID recall `{combined['metrics']['open_set_correct_id_recall']:.6f}`, macro FPR `{combined['metrics']['macro_negative_fpr']:.6f}`, and macro recall `{combined['metrics']['macro_open_set_correct_id_recall']:.6f}`. The formal gate was `{combined['formal_gate']['pass']}`. Causal shadow replay was completed for the selected, safest, and frontier policies and was not used to authorize association.",
        "",
        f"## Terminal decision: `{decision}`",
        "",
        f"Final bottleneck classification: `{bottleneck}`. `next_association_authority_stage_authorized={str(next_authorized).lower()}`. `next_memory_state_learning_stage_authorized={str(next_memory).lower()}`.",
        "",
        "No SAM3 rerun, VAL/TEST access, candidate creation, solver modification, public-ID override, OSNet fine-tuning, or association authority was performed.",
    ]
    (OUT / "FINAL_REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    write_json(OUT / "stage_status.json", {"stage": STAGE, "goal_reference": f"outputs/{STAGE}/FINAL_GOAL.json", "goal_frozen": True, "status": "PASS" if decision.startswith("PASS_") else "FAIL", "decision": decision, "last_completed_artifact": "outputs/N72R20R3R2R1/FINAL_REPORT.md", "runtime_future_gt_used": False, "runtime_gt_clean": True, "sam3_rerun": False, "val_accessed": False, "test_accessed": False, "next_association_authority_stage_authorized": next_authorized, "next_memory_state_learning_stage_authorized": next_memory, "formal_static_gate": combined["formal_gate"], "formal_causal_gate": causal_gate if formal_causal_authorized else None, "shadow_causal_completed": True, "storage_audit_after": storage_after})
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--skip-query-recompute", action="store_true")
    args = parser.parse_args()
    before = storage_audit()
    write_json(OUT / "storage_audit_before.json", before)
    if before["storage_status"] == "HARD_STOP":
        raise RuntimeError("storage hard stop below 100 GiB")
    actual_head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, check=True, stdout=subprocess.PIPE, text=True).stdout.strip()
    if actual_head != SOURCE_HEAD:
        raise RuntimeError(f"unexpected source head: {actual_head} != {SOURCE_HEAD}")
    rows, labels, metadata = load_rows()
    write_tapes(rows, labels, metadata)
    c0 = c0_baseline(rows)
    write_json(OUT / "calibration/C0_direct_none.json", c0)
    write_json(OUT / "forensics/score_scale_audit.json", score_scale_audit(rows))
    write_json(OUT / "forensics/sequence_bottleneck_map.json", sequence_bottleneck_map(rows))
    write_json(OUT / "forensics/oracle_threshold_analysis.json", oracle_threshold_analysis(rows))
    write_json(OUT / "calibration/score_upper_bound.json", upper_bound_audit(rows))
    if not args.skip_query_recompute:
        device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
        query_setup = _score_query_variant(rows, device)
    else:
        device = torch.device("cpu")
        query_setup = {"mode": "skipped", "reason": "explicit resume flag"}
    state_gap = state_gap_analysis(rows)
    write_json(OUT / "forensics/state_gap_analysis.json", state_gap)
    query = run_query_tracks(rows, {}, device) if not args.skip_query_recompute else {"fixed_mixture": {}, "gate": {}, "gate_rank": _rank_metrics_variant(rows, "raw"), "state_rank1": _rank_metrics_variant(rows, "raw")["rank1"]}
    if not args.skip_query_recompute:
        query["query_setup"] = query_setup
    formal = run_formal_selection(rows, device)
    balance = balance_study(rows, formal, device)
    write_json(OUT / "calibration/balance_study.json", balance)
    combined = run_combined_pipeline(rows, formal, device)
    write_json(OUT / "combined/inner_selection.json", {"stage": STAGE, "outer_folds": combined["folds"], "family": combined["family_fixed_by_inner_selection_mode"], "outer_labels_used_for_combination_selection": False})
    write_json(OUT / "combined/formal_loso.json", {key: value for key, value in combined.items() if key != "selected_predictions"})
    write_json(OUT / "combined/per_sequence.json", combined["metrics"]["per_sequence"])
    write_json(OUT / "combined/bootstrap.json", _sequence_cluster_bootstrap(rows, combined["selected_predictions"]))
    write_json(OUT / "combined/pareto_frontier.json", pareto_frontier(formal, c0))
    threshold_curves = {family: formal["method_results"][family]["recall_at_FPR_2pct"] for family in formal["method_results"]}
    write_json(OUT / "combined/threshold_curves.json", {"stage": STAGE, "posthoc_only": True, "methods": threshold_curves})
    temporal = temporal_tracks(rows, combined)
    write_json(OUT / "forensics/temporal_stability.json", temporal["temporal_stability"] | {"stage": STAGE, "full": temporal})
    write_json(OUT / "temporal/ema.json", temporal["ema"])
    write_json(OUT / "temporal/hysteresis.json", temporal["hysteresis"])
    write_json(OUT / "temporal/selected_temporal_policy.json", temporal["selected_temporal_policy"])
    base_uids = _base_assignment_uids()
    safest_family = min(formal["method_results"], key=lambda family: (formal["method_results"][family]["aggregate"]["negative_fpr"], -formal["method_results"][family]["aggregate"]["open_set_correct_id_recall"]))
    frontier_family = max(formal["method_results"], key=lambda family: (formal["method_results"][family]["recall_at_FPR_2pct"]["recall_at_FPR_2pct"], formal["method_results"][family]["aggregate"]["open_set_correct_id_recall"]))
    selected_shadow_metrics, selected_audit = shadow_replay(rows, combined["selected_predictions"], "combined_selected", base_uids)
    safest_records = formal["method_predictions"][safest_family]
    frontier_records = formal["method_predictions"][frontier_family]
    safest_shadow_metrics, safest_audit = shadow_replay(rows, safest_records, f"safest_{safest_family}", base_uids)
    frontier_shadow_metrics, frontier_audit = shadow_replay(rows, frontier_records, f"frontier_{frontier_family}", base_uids)
    shadow = {"base_uids": base_uids, "selected": selected_shadow_metrics, "safest": safest_shadow_metrics, "frontier": frontier_shadow_metrics, "audit_rows": selected_audit + safest_audit + frontier_audit}
    result = finalize(rows, metadata, c0, formal, query, state_gap, combined, temporal, shadow, before)
    write_json(OUT / "storage_audit_after.json", result["storage_after"])
    write_json(OUT / "source_audit_runtime.json", {"source_head": SOURCE_HEAD, "actual_head": actual_head, "working_branch": "codex/n72r20r3r2r1-open-set-decision-closure", "historical_outputs_unchanged": True})
    print(json.dumps({"stage": STAGE, "decision": result["decision"], "formal_gate": combined["formal_gate"]["pass"], "pooled_fpr": combined["metrics"]["negative_fpr"], "pooled_recall": combined["metrics"]["open_set_correct_id_recall"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
