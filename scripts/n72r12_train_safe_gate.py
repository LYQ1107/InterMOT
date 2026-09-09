#!/usr/bin/env python3
"""Train the single preregistered N72R12 learned safe-intervention gate.

The corpus is already materialized from the frozen on-policy audit.  This
script trains only the small 37-D gate; it never starts SAM3 and it never
selects a checkpoint using a future replay metric.  The first invocation can
run ``--smoke`` to exercise a real forward/backward/save/restore path before
the fixed ten-epoch training run.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import random
import sys
import tempfile
from typing import Any, Mapping

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sam3_intermot.association.learned_safe_intervention import (  # noqa: E402
    SAFE_GATE_FEATURE_DIM,
    SAFE_GATE_FEATURE_SCHEMA,
    SafeInterventionMLP,
)


CORPUS_ROOT = ROOT / "outputs/N72R12/gate_corpus"
OUTPUT_ROOT = ROOT / "outputs/N72R12/gate_training"
SEED = 7212
EPOCHS = 10
BATCH_SIZE = 256
LEARNING_RATE = 1.0e-3
WEIGHT_DECAY = 1.0e-4
SMOKE_BATCH_SIZE = 256


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_bytes(path: Path, writer: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    os.close(fd)
    try:
        writer(Path(temporary))
        with Path(temporary).open("rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        directory_fd = os.open(path.parent, os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    payload = json.dumps(dict(value), ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n"
    _atomic_bytes(path, lambda target: target.write_text(payload, encoding="utf-8"))


def atomic_torch(path: Path, value: Mapping[str, Any]) -> None:
    _atomic_bytes(path, lambda target: torch.save(dict(value), target))


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object: {path}")
    return value


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise TypeError(f"{path}:{line_number} is not an object")
            rows.append(value)
    return rows


def _finite(value: float, label: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} is non-finite")
    return result


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_split(split: str, manifest: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    split_info = manifest.get("splits", {}).get(split)
    if not isinstance(split_info, Mapping):
        raise ValueError(f"missing corpus split metadata: {split}")
    npz_path = CORPUS_ROOT / f"{split}.npz"
    metadata_path = CORPUS_ROOT / f"{split}_metadata.jsonl"
    if not npz_path.is_file() or not metadata_path.is_file():
        raise FileNotFoundError(f"missing safe-gate corpus files for {split}")
    if split_info.get("npz_sha256") != sha256_file(npz_path) or split_info.get("metadata_sha256") != sha256_file(metadata_path):
        raise ValueError(f"safe-gate corpus hash mismatch for {split}")
    with np.load(npz_path, allow_pickle=False) as source:
        if set(source.files) != {"features", "labels"}:
            raise ValueError(f"unexpected array keys for {split}: {source.files}")
        features = np.asarray(source["features"], dtype=np.float32)
        labels = np.asarray(source["labels"], dtype=np.int64)
    all_metadata = read_jsonl(metadata_path)
    metadata = [row for row in all_metadata if row.get("included_in_training") is True]
    if features.ndim != 2 or features.shape[1] != SAFE_GATE_FEATURE_DIM:
        raise ValueError(f"{split} feature shape is not N x {SAFE_GATE_FEATURE_DIM}: {features.shape}")
    if labels.shape != (features.shape[0],) or not np.isin(labels, [0, 1]).all():
        raise ValueError(f"{split} labels are invalid")
    if len(metadata) != features.shape[0] or not np.isfinite(features).all():
        raise ValueError(
            f"{split} included metadata/features are incomplete or non-finite "
            f"(raw_metadata={len(all_metadata)}, included_metadata={len(metadata)}, arrays={features.shape[0]})"
        )
    expected_sequences = {str(value) for value in manifest.get(f"{split}_sequences", [])}
    actual_sequences = {str(row.get("sequence")) for row in metadata}
    if expected_sequences and actual_sequences != expected_sequences:
        raise ValueError(f"{split} sequence split mismatch")
    if any(row.get("runtime_future_gt_used") is not False or row.get("gt_used_for_offline_label") is not True for row in metadata):
        raise ValueError(f"{split} contains an invalid runtime/label provenance flag")
    return features, labels, metadata


def _binary_metrics(labels: np.ndarray, probabilities: np.ndarray) -> dict[str, Any]:
    labels = np.asarray(labels, dtype=np.int64)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    if labels.shape != probabilities.shape or labels.ndim != 1 or not np.isfinite(probabilities).all():
        raise ValueError("classification metric inputs are invalid")
    predicted = probabilities >= 0.5
    positive = labels == 1
    negative = ~positive
    true_positive = int(np.sum(predicted & positive))
    false_positive = int(np.sum(predicted & negative))
    true_negative = int(np.sum(~predicted & negative))
    false_negative = int(np.sum(~predicted & positive))
    accuracy = float(np.mean(predicted == positive)) if labels.size else 0.0
    precision = true_positive / max(true_positive + false_positive, 1)
    recall = true_positive / max(true_positive + false_negative, 1)
    f1 = 2.0 * precision * recall / max(precision + recall, 1.0e-12)

    order = np.argsort(-probabilities, kind="mergesort")
    sorted_labels = labels[order]
    sorted_scores = probabilities[order]
    positive_count = int(np.sum(positive))
    negative_count = int(np.sum(negative))
    if positive_count and negative_count:
        tp = np.cumsum(sorted_labels == 1, dtype=np.float64)
        fp = np.cumsum(sorted_labels == 0, dtype=np.float64)
        tpr = np.concatenate(([0.0], tp / positive_count, [1.0]))
        fpr = np.concatenate(([0.0], fp / negative_count, [1.0]))
        auroc = float(np.sum((fpr[1:] - fpr[:-1]) * (tpr[1:] + tpr[:-1]) * 0.5))
    else:
        auroc = None
    if positive_count:
        tp = np.cumsum(sorted_labels == 1, dtype=np.float64)
        ranks = np.arange(1, labels.size + 1, dtype=np.float64)
        average_precision = float(np.sum((tp / ranks) * (sorted_labels == 1)) / positive_count)
    else:
        average_precision = None
    unsafe = predicted & ~positive
    predicted_apply_count = int(np.sum(predicted))
    return {
        "count": int(labels.size),
        "positive_count": int(np.sum(positive)),
        "negative_count": int(np.sum(negative)),
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "auroc": auroc,
        "auprc": average_precision,
        "true_positive_count": true_positive,
        "false_positive_count": false_positive,
        "true_negative_count": true_negative,
        "false_negative_count": false_negative,
        "predicted_apply_count": predicted_apply_count,
        "unsafe_apply_count": int(np.sum(unsafe)),
        "unsafe_apply_rate": float(np.sum(unsafe) / max(predicted_apply_count, 1)),
        "threshold": 0.5,
        "score_order_ties": int(np.sum(np.diff(sorted_scores) == 0.0)) if sorted_scores.size > 1 else 0,
    }


def evaluate(model: SafeInterventionMLP, features: np.ndarray, labels: np.ndarray, criterion: nn.Module, device: torch.device) -> tuple[float, dict[str, Any]]:
    model.eval()
    with torch.no_grad():
        tensor = torch.from_numpy(features).to(device=device, dtype=torch.float32)
        target = torch.from_numpy(labels).to(device=device, dtype=torch.float32)
        logits = model(tensor)
        loss = criterion(logits, target)
        probabilities = torch.sigmoid(logits).cpu().numpy()
    loss_value = _finite(loss.item(), "evaluation loss")
    metrics = _binary_metrics(labels, probabilities)
    metrics["loss"] = loss_value
    return loss_value, metrics


def corpus_payload() -> tuple[dict[str, Any], str]:
    manifest_path = CORPUS_ROOT / "manifest.json"
    manifest = read_json(manifest_path)
    if manifest.get("status") != "PASS_N72R12_SAFE_GATE_CORPUS":
        raise ValueError(f"safe-gate corpus is not a PASS artifact: {manifest.get('status')}")
    if manifest.get("feature_dim") != SAFE_GATE_FEATURE_DIM or manifest.get("feature_schema") != list(SAFE_GATE_FEATURE_SCHEMA):
        raise ValueError("safe-gate feature schema mismatch")
    if manifest.get("runtime_future_gt_used") is not False or manifest.get("real_human_evidence") is not False:
        raise ValueError("safe-gate corpus provenance is invalid")
    return manifest, sha256_file(manifest_path)


def run_smoke(manifest: Mapping[str, Any], manifest_sha256: str, device: torch.device) -> dict[str, Any]:
    features, labels, _ = load_split("train", manifest)
    count = min(SMOKE_BATCH_SIZE, features.shape[0])
    set_seed(SEED)
    model = SafeInterventionMLP().to(device)
    model.train()
    positive_count = int(np.sum(labels == 1))
    negative_count = int(np.sum(labels == 0))
    pos_weight = min(max(negative_count / max(positive_count, 1), 1.0), 10.0)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight, dtype=torch.float32, device=device))
    batch = torch.from_numpy(features[:count]).to(device=device, dtype=torch.float32)
    target = torch.from_numpy(labels[:count]).to(device=device, dtype=torch.float32)
    logits = model(batch)
    loss = criterion(logits, target)
    if not torch.isfinite(loss):
        raise RuntimeError("safe-gate smoke loss is non-finite")
    loss.backward()
    gradient_finite = all(parameter.grad is None or bool(torch.isfinite(parameter.grad).all()) for parameter in model.parameters())
    if not gradient_finite:
        raise RuntimeError("safe-gate smoke gradient is non-finite")
    with tempfile.TemporaryDirectory(prefix="n72r12_safe_gate_smoke_") as directory:
        checkpoint = Path(directory) / "safe_gate.pt"
        payload = {
            "schema_version": "N72R12_SAFE_GATE_CHECKPOINT_V1",
            "model_schema": list(SAFE_GATE_FEATURE_SCHEMA),
            "feature_dim": SAFE_GATE_FEATURE_DIM,
            "state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
            "corpus_manifest_sha256": manifest_sha256,
        }
        torch.save(payload, checkpoint)
        restored = SafeInterventionMLP()
        loaded = torch.load(checkpoint, map_location="cpu", weights_only=False)
        restored.load_state_dict(loaded["state_dict"], strict=True)
    result = {
        "schema_version": "N72R12_SAFE_GATE_TRAINING_SMOKE_V1",
        "status": "PASS_N72R12_SAFE_GATE_TRAINING_SMOKE",
        "created_at_utc": now_utc(),
        "device": str(device),
        "feature_dim": SAFE_GATE_FEATURE_DIM,
        "parameter_count": int(sum(parameter.numel() for parameter in model.parameters())),
        "batch_count": count,
        "loss": float(loss.item()),
        "gradient_finite": gradient_finite,
        "save_restore_pass": True,
        "corpus_manifest_sha256": manifest_sha256,
        "runtime_future_gt_used": False,
        "production_authorized": False,
    }
    atomic_json(OUTPUT_ROOT / "smoke.json", result)
    return result


def run_training(manifest: Mapping[str, Any], manifest_sha256: str, device: torch.device) -> dict[str, Any]:
    train_features, train_labels, train_metadata = load_split("train", manifest)
    validation_features, validation_labels, validation_metadata = load_split("validation", manifest)
    if {str(row["sequence"]) for row in train_metadata} & {str(row["sequence"]) for row in validation_metadata}:
        raise ValueError("train/validation sequence leakage")
    set_seed(SEED)
    model = SafeInterventionMLP().to(device)
    positive_count = int(np.sum(train_labels == 1))
    negative_count = int(np.sum(train_labels == 0))
    pos_weight = min(max(negative_count / max(positive_count, 1), 1.0), 10.0)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight, dtype=torch.float32, device=device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    generator = torch.Generator(device="cpu").manual_seed(SEED)
    best_validation_loss = float("inf")
    best_epoch = None
    best_state: dict[str, torch.Tensor] | None = None
    history: list[dict[str, Any]] = []
    for epoch in range(1, EPOCHS + 1):
        model.train()
        permutation = torch.randperm(train_features.shape[0], generator=generator)
        train_losses: list[float] = []
        for start in range(0, train_features.shape[0], BATCH_SIZE):
            indices = permutation[start : start + BATCH_SIZE].numpy()
            batch = torch.from_numpy(train_features[indices]).to(device=device, dtype=torch.float32)
            target = torch.from_numpy(train_labels[indices]).to(device=device, dtype=torch.float32)
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch)
            loss = criterion(logits, target)
            if not torch.isfinite(loss):
                raise RuntimeError(f"non-finite train loss at epoch {epoch}, step {start // BATCH_SIZE + 1}")
            loss.backward()
            if not all(parameter.grad is None or bool(torch.isfinite(parameter.grad).all()) for parameter in model.parameters()):
                raise RuntimeError(f"non-finite gradient at epoch {epoch}, step {start // BATCH_SIZE + 1}")
            optimizer.step()
            train_losses.append(float(loss.item()))
        train_loss, train_metrics = evaluate(model, train_features, train_labels, criterion, device)
        validation_loss, validation_metrics = evaluate(model, validation_features, validation_labels, criterion, device)
        epoch_payload = {
            "epoch": epoch,
            "steps": int(math.ceil(train_features.shape[0] / BATCH_SIZE)),
            "train_loss_mean": float(sum(train_losses) / max(len(train_losses), 1)),
            "train_loss_full": train_loss,
            "validation_loss": validation_loss,
            "train_metrics": train_metrics,
            "validation_metrics": validation_metrics,
        }
        history.append(epoch_payload)
        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            best_epoch = epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    if best_state is None or best_epoch is None:
        raise RuntimeError("training did not produce a validation checkpoint")
    model.load_state_dict(best_state, strict=True)
    _, best_validation_metrics = evaluate(model, validation_features, validation_labels, criterion, device)
    readiness = {
        "precision_at_0.5": float(best_validation_metrics["precision"]),
        "unsafe_apply_rate": float(best_validation_metrics["unsafe_apply_rate"]),
        "predicted_apply_count": int(best_validation_metrics["predicted_apply_count"]),
        "precision_threshold": 0.75,
        "unsafe_apply_rate_threshold": 0.10,
        "predicted_apply_required": True,
        "pass": bool(
            best_validation_metrics["precision"] >= 0.75
            and best_validation_metrics["unsafe_apply_rate"] <= 0.10
            and best_validation_metrics["predicted_apply_count"] > 0
        ),
    }
    checkpoint_payload = {
        "schema_version": "N72R12_SAFE_GATE_CHECKPOINT_V1",
        "status": "PASS_N72R12_SAFE_GATE_TRAINED_CHECKPOINT",
        "model_schema": list(SAFE_GATE_FEATURE_SCHEMA),
        "feature_dim": SAFE_GATE_FEATURE_DIM,
        "architecture": "Linear(37,64)-GELU-LayerNorm(64)-Linear(64,32)-GELU-Linear(32,1)",
        "state_dict": best_state,
        "training_config": {
            "seed": SEED,
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
            "learning_rate": LEARNING_RATE,
            "optimizer": "AdamW",
            "weight_decay": WEIGHT_DECAY,
            "loss": "BCEWithLogitsLoss",
            "train_positive_count": positive_count,
            "train_negative_count": negative_count,
            "pos_weight": pos_weight,
            "checkpoint_selection": "minimum_validation_BCE",
            "threshold": 0.5,
        },
        "best_epoch": best_epoch,
        "best_validation_bce": best_validation_loss,
        "corpus_manifest_sha256": manifest_sha256,
        "runtime_future_gt_used": False,
        "labels_used_only_offline": True,
        "production_authorized": False,
    }
    checkpoint_path = OUTPUT_ROOT / "safe_gate.pt"
    atomic_torch(checkpoint_path, checkpoint_payload)
    history_payload = {
        "schema_version": "N72R12_SAFE_GATE_TRAINING_HISTORY_V1",
        "status": "PASS_N72R12_SAFE_GATE_TRAINING",
        "created_at_utc": now_utc(),
        "training_config": checkpoint_payload["training_config"],
        "train_examples": int(train_features.shape[0]),
        "validation_examples": int(validation_features.shape[0]),
        "train_sequences": sorted({str(row["sequence"]) for row in train_metadata}),
        "validation_sequences": sorted({str(row["sequence"]) for row in validation_metadata}),
        "best_epoch": best_epoch,
        "best_validation_bce": best_validation_loss,
        "validation_metrics_at_best": best_validation_metrics,
        "readiness": readiness,
        "history": history,
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "corpus_manifest_sha256": manifest_sha256,
        "runtime_future_gt_used": False,
        "production_authorized": False,
    }
    atomic_json(OUTPUT_ROOT / "training_history.json", history_payload)
    training_manifest = {
        "schema_version": "N72R12_SAFE_GATE_TRAINING_MANIFEST_V1",
        "status": "PASS_N72R12_SAFE_GATE_TRAINING",
        "created_at_utc": now_utc(),
        "corpus_manifest": str(CORPUS_ROOT / "manifest.json"),
        "corpus_manifest_sha256": manifest_sha256,
        "train_npz_sha256": sha256_file(CORPUS_ROOT / "train.npz"),
        "validation_npz_sha256": sha256_file(CORPUS_ROOT / "validation.npz"),
        "train_metadata_sha256": sha256_file(CORPUS_ROOT / "train_metadata.jsonl"),
        "validation_metadata_sha256": sha256_file(CORPUS_ROOT / "validation_metadata.jsonl"),
        "train_examples": int(train_features.shape[0]),
        "validation_examples": int(validation_features.shape[0]),
        "train_positive_count": positive_count,
        "train_negative_count": negative_count,
        "validation_positive_count": int(np.sum(validation_labels == 1)),
        "validation_negative_count": int(np.sum(validation_labels == 0)),
        "sequence_disjoint": True,
        "training_config": checkpoint_payload["training_config"],
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "readiness": readiness,
        "runtime_future_gt_used": False,
        "interaction_source": "simulated_from_gt",
        "not_real_human_evidence": True,
        "production_authorized": False,
    }
    atomic_json(OUTPUT_ROOT / "training_manifest.json", training_manifest)
    return training_manifest


def main() -> int:
    global CORPUS_ROOT, OUTPUT_ROOT
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus-root", type=Path, default=CORPUS_ROOT)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    CORPUS_ROOT = args.corpus_root.resolve()
    OUTPUT_ROOT = args.output_root.resolve()
    started = now_utc()
    try:
        manifest, manifest_sha256 = corpus_payload()
        device = torch.device(str(args.device))
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(f"requested training CUDA device is unavailable: {device}")
        result = run_smoke(manifest, manifest_sha256, device) if args.smoke else run_training(manifest, manifest_sha256, device)
        result = dict(result)
        result["started_at_utc"] = started
        result["finished_at_utc"] = now_utc()
        if args.smoke:
            atomic_json(OUTPUT_ROOT / "smoke.json", result)
        print(json.dumps({"status": result["status"], "output": str(OUTPUT_ROOT), "device": str(device)}, sort_keys=True))
        return 0
    except Exception as exc:
        failure = {
            "schema_version": "N72R12_SAFE_GATE_TRAINING_FAILURE_V1",
            "status": "FAIL_N72R12_SAFE_GATE_TRAINING",
            "started_at_utc": started,
            "finished_at_utc": now_utc(),
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": __import__("traceback").format_exc(),
            "command": list(sys.argv),
            "historical_outputs_modified": False,
            "runtime_future_gt_used": False,
        }
        OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
        target = OUTPUT_ROOT / "training_failure.json"
        if target.exists():
            attempt = 1
            while (OUTPUT_ROOT / f"training_failure_attempt_{attempt:02d}.json").exists():
                attempt += 1
            target = OUTPUT_ROOT / f"training_failure_attempt_{attempt:02d}.json"
        atomic_json(target, failure)
        print(json.dumps({"status": failure["status"], "failure": str(target), "error": str(exc)}, sort_keys=True), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
