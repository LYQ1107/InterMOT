#!/usr/bin/env python3
"""Train the isolated N72R10 source-aware temporal scorer.

This script consumes only the sealed N72R10 causal tensors.  Dataset GT is
represented by offline labels already stored in the corpus; no SAM3 runtime
or future-GT lookup is performed here.  The checkpoint is research-only and
is never imported by production association.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import random
import tempfile
import traceback
from typing import Any, Mapping

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in __import__("sys").path:
    __import__("sys").path.insert(0, str(ROOT))

from sam3_intermot.reacquisition.models.n72r10_source_temporal_v2 import (  # noqa: E402
    CANDIDATE_FEATURE_DIM,
    DISTRACTOR_MEMORY_SLOTS,
    SOURCE_FEATURE_DIM,
    TEMPORAL_FEATURE_DIM,
    TRUSTED_MEMORY_SLOTS,
    N72R10SourceAwareTemporalIdentityModel,
    n72r9_loss,
)


TRAINING_ROOT = ROOT / "outputs/N72R10/training"
TRAIN_NPZ = TRAINING_ROOT / "train.npz"
VAL_NPZ = TRAINING_ROOT / "validation.npz"
TRAIN_META = TRAINING_ROOT / "train_metadata.jsonl"
VAL_META = TRAINING_ROOT / "validation_metadata.jsonl"
CORPUS_MANIFEST = TRAINING_ROOT / "corpus_manifest.json"
SMOKE_CHECKPOINT = TRAINING_ROOT / "N72R10_source_temporal_smoke.pt"
CHECKPOINT = TRAINING_ROOT / "N72R10SourceAwareTemporalIdentityModel_v2.pt"
HISTORY_PATH = TRAINING_ROOT / "N72R10_training_history.json"
SMOKE_STATUS = ROOT / "outputs/N72R10/stage_05_training_smoke_status.json"
TRAIN_STATUS = ROOT / "outputs/N72R10/stage_06_training_status.json"

SEED = 7210
MAX_EPOCHS = 40
PATIENCE = 8
BATCH_SIZE = 128
LEARNING_RATE = 5.0e-4
WEIGHT_DECAY = 1.0e-4
PAIRWISE_WEIGHT = 0.15
PAIRWISE_MARGIN = 0.20
# Frozen before training.  It addresses the observed NONE under-representation
# without using any future-effect, IDSW, IoU, or replay outcome.
NONE_EXAMPLE_WEIGHT = 2.0
TARGET_EXAMPLE_WEIGHT = 1.0


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
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
    atomic_write(path, json.dumps(dict(value), ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n")


def atomic_torch_save(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    os.close(fd)
    try:
        torch.save(value, temporary)
        with open(temporary, "rb") as handle:
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


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object: {path}")
    return value


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def model_config() -> dict[str, Any]:
    return {
        "candidate_feature_dim": CANDIDATE_FEATURE_DIM,
        "source_feature_dim": SOURCE_FEATURE_DIM,
        "temporal_feature_dim": TEMPORAL_FEATURE_DIM,
        "trusted_slots": TRUSTED_MEMORY_SLOTS,
        "distractor_slots": DISTRACTOR_MEMORY_SLOTS,
        "hidden_dim": 96,
        "layers": 1,
        "heads": 4,
        "dropout": 0.0,
        "architecture": "N72R10_source_conditioned_candidate_set_plus_causal_memory_temporal_context",
    }


def new_model() -> N72R10SourceAwareTemporalIdentityModel:
    config = model_config()
    return N72R10SourceAwareTemporalIdentityModel(
        **{key: value for key, value in config.items() if key in {
            "candidate_feature_dim", "source_feature_dim", "temporal_feature_dim",
            "trusted_slots", "distractor_slots", "hidden_dim", "layers", "heads", "dropout",
        }}
    )


def load_split(npz_path: Path, metadata_path: Path) -> tuple[dict[str, np.ndarray], list[dict[str, Any]]]:
    if not npz_path.is_file() or not metadata_path.is_file():
        raise FileNotFoundError(f"missing split: {npz_path} / {metadata_path}")
    with np.load(npz_path, allow_pickle=False) as loaded:
        arrays = {key: np.asarray(loaded[key]) for key in loaded.files}
    required = {
        "candidate_features", "candidate_mask", "source_features", "trusted_memory", "trusted_mask",
        "distractor_memory", "distractor_mask", "neighbor_feature", "temporal_features", "labels", "candidate_counts",
    }
    missing = required - set(arrays)
    if missing:
        raise RuntimeError(f"split missing keys: {npz_path}: {sorted(missing)}")
    count = len(arrays["labels"])
    if arrays["candidate_features"].shape[:2] != arrays["candidate_mask"].shape:
        raise RuntimeError(f"candidate/mask shape mismatch: {npz_path}")
    if arrays["source_features"].shape[:2] != arrays["candidate_mask"].shape:
        raise RuntimeError(f"source/candidate shape mismatch: {npz_path}")
    if arrays["candidate_features"].shape[0] != count:
        raise RuntimeError(f"candidate row count mismatch: {npz_path}")
    if arrays["trusted_memory"].shape != (count, TRUSTED_MEMORY_SLOTS, 512):
        raise RuntimeError(f"trusted memory shape mismatch: {npz_path}")
    if arrays["distractor_memory"].shape != (count, DISTRACTOR_MEMORY_SLOTS, 512):
        raise RuntimeError(f"distractor memory shape mismatch: {npz_path}")
    if arrays["trusted_mask"].shape != (count, TRUSTED_MEMORY_SLOTS) or arrays["distractor_mask"].shape != (count, DISTRACTOR_MEMORY_SLOTS):
        raise RuntimeError(f"memory mask shape mismatch: {npz_path}")
    if arrays["neighbor_feature"].shape != (count, 512) or arrays["temporal_features"].shape != (count, TEMPORAL_FEATURE_DIM):
        raise RuntimeError(f"context shape mismatch: {npz_path}")
    if arrays["candidate_features"].shape[-1] != CANDIDATE_FEATURE_DIM or arrays["source_features"].shape[-1] != SOURCE_FEATURE_DIM:
        raise RuntimeError(f"feature width mismatch: {npz_path}")
    for key, value in arrays.items():
        if not np.all(np.isfinite(value)):
            raise RuntimeError(f"non-finite split array {key}: {npz_path}")
    candidate_axis = arrays["candidate_features"].shape[1]
    if bool(np.any(arrays["labels"] < 0)) or bool(np.any(arrays["labels"] > candidate_axis)):
        raise RuntimeError(f"label outside candidate/NONE range: {npz_path}")
    metadata = [json.loads(line) for line in metadata_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(metadata) != count:
        raise RuntimeError(f"metadata count mismatch: {npz_path}")
    for index, item in enumerate(metadata):
        if int(item.get("candidate_count", len(item.get("candidate_uids", [])))) != int(arrays["candidate_counts"][index]):
            raise RuntimeError(f"metadata candidate count mismatch at {npz_path}:{index}")
        if item.get("runtime_future_gt_used") is not False:
            raise RuntimeError(f"runtime future GT flag violation at {npz_path}:{index}")
    return arrays, metadata


def batch_tensors(arrays: Mapping[str, np.ndarray], indices: np.ndarray, device: torch.device) -> tuple[torch.Tensor, ...]:
    return (
        torch.as_tensor(arrays["candidate_features"][indices], dtype=torch.float32, device=device),
        torch.as_tensor(arrays["candidate_mask"][indices], dtype=torch.bool, device=device),
        torch.as_tensor(arrays["source_features"][indices], dtype=torch.float32, device=device),
        torch.as_tensor(arrays["trusted_memory"][indices], dtype=torch.float32, device=device),
        torch.as_tensor(arrays["trusted_mask"][indices], dtype=torch.bool, device=device),
        torch.as_tensor(arrays["distractor_memory"][indices], dtype=torch.float32, device=device),
        torch.as_tensor(arrays["distractor_mask"][indices], dtype=torch.bool, device=device),
        torch.as_tensor(arrays["neighbor_feature"][indices], dtype=torch.float32, device=device),
        torch.as_tensor(arrays["temporal_features"][indices], dtype=torch.float32, device=device),
        torch.as_tensor(arrays["labels"][indices], dtype=torch.long, device=device),
    )


def example_weights(labels: torch.Tensor, candidate_axis: int) -> torch.Tensor:
    """Apply the pre-registered NONE correction; no outcome-dependent weight."""
    return torch.where(
        labels == int(candidate_axis),
        torch.full_like(labels, NONE_EXAMPLE_WEIGHT, dtype=torch.float32),
        torch.full_like(labels, TARGET_EXAMPLE_WEIGHT, dtype=torch.float32),
    )


def source_for_prediction(metadata: list[dict[str, Any]], predictions: np.ndarray, labels: np.ndarray, candidate_axis: int) -> dict[str, Any]:
    counts: Counter[str] = Counter()
    correct_by_source: Counter[str] = Counter()
    total_by_source: Counter[str] = Counter()
    for index, item in enumerate(metadata):
        candidate_sources = [str(value) for value in item.get("candidate_sources", [])]
        label = int(labels[index])
        prediction = int(predictions[index])
        label_source = "NONE" if label == candidate_axis else (candidate_sources[label] if label < len(candidate_sources) else "INVALID")
        predicted_source = "NONE" if prediction == candidate_axis else (candidate_sources[prediction] if prediction < len(candidate_sources) else "INVALID")
        counts[label_source] += 1
        total_by_source[label_source] += 1
        if predicted_source == label_source:
            correct_by_source[label_source] += 1
    return {
        "label_source_counts": dict(sorted(counts.items())),
        "label_source_accuracy": {
            key: float(correct_by_source[key] / value) if value else None
            for key, value in sorted(total_by_source.items())
        },
    }


def evaluate(model: torch.nn.Module, arrays: Mapping[str, np.ndarray], metadata: list[dict[str, Any]], device: torch.device) -> dict[str, Any]:
    model.eval()
    losses: list[float] = []
    predictions: list[int] = []
    candidate_axis = arrays["candidate_features"].shape[1]
    with torch.no_grad():
        for start in range(0, len(arrays["labels"]), BATCH_SIZE):
            indices = np.arange(start, min(start + BATCH_SIZE, len(arrays["labels"])), dtype=np.int64)
            values = batch_tensors(arrays, indices, device)
            logits = model(*values[:-1])
            weights = example_weights(values[-1], candidate_axis)
            loss, _ = n72r9_loss(
                logits, values[-1], values[1], pairwise_weight=PAIRWISE_WEIGHT,
                pairwise_margin=PAIRWISE_MARGIN, example_weight=weights,
            )
            losses.append(float(loss.detach().cpu()))
            predictions.extend(torch.argmax(logits, dim=1).detach().cpu().tolist())
    labels = arrays["labels"].astype(np.int64)
    predicted = np.asarray(predictions, dtype=np.int64)
    target = labels < candidate_axis
    none = labels == candidate_axis
    by_action: dict[str, dict[str, Any]] = {}
    for action in sorted({str(item["action_type"]) for item in metadata}):
        selected = np.asarray([i for i, item in enumerate(metadata) if str(item["action_type"]) == action], dtype=np.int64)
        selected_target = selected[labels[selected] < candidate_axis]
        selected_none = selected[labels[selected] == candidate_axis]
        by_action[action] = {
            "examples": int(len(selected)),
            "accuracy": float(np.mean(predicted[selected] == labels[selected])) if len(selected) else None,
            "target_candidate_accuracy": float(np.mean(predicted[selected_target] == labels[selected_target])) if len(selected_target) else None,
            "none_accuracy": float(np.mean(predicted[selected_none] == candidate_axis)) if len(selected_none) else None,
        }
    return {
        "examples": int(len(labels)),
        "loss": float(np.mean(losses)) if losses else None,
        "accuracy": float(np.mean(predicted == labels)) if len(labels) else None,
        "target_candidate_accuracy": float(np.mean(predicted[target] == labels[target])) if bool(np.any(target)) else None,
        "none_accuracy": float(np.mean(predicted[none] == candidate_axis)) if bool(np.any(none)) else None,
        "target_examples": int(np.sum(target)),
        "none_examples": int(np.sum(none)),
        "source_diagnostics": source_for_prediction(metadata, predicted, labels, candidate_axis),
        "by_action": by_action,
    }


def run_smoke(device: torch.device) -> dict[str, Any]:
    arrays, metadata = load_split(TRAIN_NPZ, TRAIN_META)
    set_seed(SEED)
    count = min(8, len(arrays["labels"]))
    indices = np.arange(count, dtype=np.int64)
    model = new_model().to(device)
    model.train()
    values = batch_tensors(arrays, indices, device)
    logits = model(*values[:-1])
    if not bool(torch.isfinite(logits).all()):
        raise RuntimeError("smoke logits are non-finite")
    weights = example_weights(values[-1], arrays["candidate_features"].shape[1])
    loss, parts = n72r9_loss(
        logits, values[-1], values[1], pairwise_weight=PAIRWISE_WEIGHT,
        pairwise_margin=PAIRWISE_MARGIN, example_weight=weights,
    )
    if not bool(torch.isfinite(loss)):
        raise RuntimeError("smoke loss is non-finite")
    loss.backward()
    gradients = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
    if not gradients or not all(bool(torch.isfinite(value).all()) for value in gradients):
        raise RuntimeError("smoke gradients are missing or non-finite")
    atomic_torch_save(SMOKE_CHECKPOINT, {"model_config": model_config(), "state_dict": model.state_dict(), "seed": SEED})
    restored = new_model().to(device)
    payload = torch.load(SMOKE_CHECKPOINT, map_location=device, weights_only=False)
    restored.load_state_dict(payload["state_dict"])
    model.eval()
    restored.eval()
    with torch.no_grad():
        first = model(*values[:-1])
        second = restored(*values[:-1])
    if not bool(torch.allclose(first, second, atol=1.0e-6, rtol=1.0e-6)):
        raise RuntimeError("smoke checkpoint restore changed logits")
    result = {
        "schema_version": "N72R10_SOURCE_AWARE_TRAINING_SMOKE_V1",
        "status": "PASS_N72R10_TRAINING_SMOKE",
        "created_at_utc": now_utc(),
        "device": str(device),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "batch_examples": count,
        "metadata_first_event": str(metadata[0].get("event_id")) if metadata else None,
        "model_config": model_config(),
        "loss": float(loss.detach().cpu()),
        "loss_parts": {key: float(value.cpu()) for key, value in parts.items()},
        "forward_finite": True,
        "backward_finite": True,
        "save_restore_equivalent": True,
        "checkpoint": str(SMOKE_CHECKPOINT),
        "checkpoint_sha256": sha256_file(SMOKE_CHECKPOINT),
        "none_example_weight": NONE_EXAMPLE_WEIGHT,
        "runtime_future_gt_used": False,
        "production_authorized": False,
    }
    atomic_json(SMOKE_STATUS, result)
    return result


def run_train(device: torch.device) -> dict[str, Any]:
    if not SMOKE_STATUS.is_file() or read_json(SMOKE_STATUS).get("status") != "PASS_N72R10_TRAINING_SMOKE":
        raise RuntimeError("N72R10 training smoke must pass before training")
    corpus = read_json(CORPUS_MANIFEST)
    if corpus.get("runtime_future_gt_used") is not False or corpus.get("not_real_human_evidence") is not True:
        raise RuntimeError("corpus provenance gate failed")
    train_arrays, train_metadata = load_split(TRAIN_NPZ, TRAIN_META)
    val_arrays, val_metadata = load_split(VAL_NPZ, VAL_META)
    if not len(train_arrays["labels"]) or not len(val_arrays["labels"]):
        raise RuntimeError("train or validation split is empty")
    set_seed(SEED)
    model = new_model().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    generator = torch.Generator(device="cpu").manual_seed(SEED)
    best_loss = float("inf")
    best_epoch = 0
    stale = 0
    history: list[dict[str, Any]] = []
    candidate_axis = train_arrays["candidate_features"].shape[1]
    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        order = torch.randperm(len(train_arrays["labels"]), generator=generator).numpy()
        train_losses: list[float] = []
        for start in range(0, len(order), BATCH_SIZE):
            indices = order[start : start + BATCH_SIZE]
            values = batch_tensors(train_arrays, indices, device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(*values[:-1])
            weights = example_weights(values[-1], candidate_axis)
            loss, parts = n72r9_loss(
                logits, values[-1], values[1], pairwise_weight=PAIRWISE_WEIGHT,
                pairwise_margin=PAIRWISE_MARGIN, example_weight=weights,
            )
            if not bool(torch.isfinite(loss)):
                raise RuntimeError(f"non-finite training loss at epoch {epoch}")
            loss.backward()
            gradients = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
            if not gradients or not all(bool(torch.isfinite(value).all()) for value in gradients):
                raise RuntimeError(f"non-finite training gradient at epoch {epoch}")
            optimizer.step()
            train_losses.append(float(loss.detach().cpu()))
        validation = evaluate(model, val_arrays, val_metadata, device)
        row = {
            "epoch": epoch,
            "train_loss": float(np.mean(train_losses)),
            "last_batch_cross_entropy": float(parts["cross_entropy"].cpu()),
            "last_batch_hard_negative_ranking": float(parts["hard_negative_ranking"].cpu()),
            "validation": validation,
        }
        history.append(row)
        value = validation["loss"]
        if value is not None and value < best_loss:
            best_loss = float(value)
            best_epoch = epoch
            stale = 0
            atomic_torch_save(CHECKPOINT, {
                "schema_version": "N72R10_SOURCE_AWARE_TEMPORAL_CHECKPOINT_V2",
                "model_config": model_config(),
                "state_dict": model.state_dict(),
                "seed": SEED,
                "epoch": epoch,
                "validation_loss": float(value),
                "none_example_weight": NONE_EXAMPLE_WEIGHT,
                "runtime_future_gt_used": False,
                "production_authorized": False,
            })
        else:
            stale += 1
        if stale >= PATIENCE:
            break
    if not CHECKPOINT.is_file():
        raise RuntimeError("no finite validation-loss checkpoint was saved")
    payload = torch.load(CHECKPOINT, map_location=device, weights_only=False)
    restored = new_model().to(device)
    restored.load_state_dict(payload["state_dict"])
    train_eval = evaluate(restored, train_arrays, train_metadata, device)
    val_eval = evaluate(restored, val_arrays, val_metadata, device)
    history_payload = {
        "schema_version": "N72R10_SOURCE_AWARE_TRAINING_HISTORY_V2",
        "created_at_utc": now_utc(),
        "seed": SEED,
        "model_config": model_config(),
        "history": history,
        "best_epoch": best_epoch,
        "best_validation_loss": best_loss,
        "checkpoint_selection": "minimum_weighted_validation_loss_only",
        "none_example_weight": NONE_EXAMPLE_WEIGHT,
        "future_effect_metrics_used": False,
        "runtime_future_gt_used": False,
    }
    atomic_json(HISTORY_PATH, history_payload)
    result = {
        "schema_version": "N72R10_SOURCE_AWARE_TRAINING_STATUS_V2",
        "status": "PASS_N72R10_SOURCE_AWARE_TEMPORAL_TRAINING",
        "created_at_utc": now_utc(),
        "device": str(device),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "seed": SEED,
        "model_config": model_config(),
        "max_epochs": MAX_EPOCHS,
        "epochs_completed": len(history),
        "patience": PATIENCE,
        "best_epoch": best_epoch,
        "best_validation_loss": best_loss,
        "none_example_weight": NONE_EXAMPLE_WEIGHT,
        "train_evaluation": train_eval,
        "validation_evaluation": val_eval,
        "checkpoint": str(CHECKPOINT),
        "checkpoint_sha256": sha256_file(CHECKPOINT),
        "history": str(HISTORY_PATH),
        "history_sha256": sha256_file(HISTORY_PATH),
        "corpus_manifest": str(CORPUS_MANIFEST),
        "corpus_manifest_sha256": sha256_file(CORPUS_MANIFEST),
        "train_npz_sha256": sha256_file(TRAIN_NPZ),
        "validation_npz_sha256": sha256_file(VAL_NPZ),
        "checkpoint_selected_by_future_effect": False,
        "runtime_future_gt_used": False,
        "interaction_source": "simulated_from_gt",
        "not_real_human_evidence": True,
        "production_authorized": False,
    }
    atomic_json(TRAIN_STATUS, result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("smoke", "train"))
    parser.add_argument("--device", default=os.environ.get("N72R10_DEVICE", "cpu"))
    args = parser.parse_args()
    started = now_utc()
    try:
        device = torch.device(str(args.device))
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("requested CUDA device is unavailable")
        result = run_smoke(device) if args.mode == "smoke" else run_train(device)
        result["started_at_utc"] = started
        print(json.dumps(result, ensure_ascii=False, sort_keys=True))
        return 0
    except Exception as exc:
        status_path = SMOKE_STATUS if args.mode == "smoke" else TRAIN_STATUS
        failure = TRAINING_ROOT / "attempts" / f"{args.mode}_failure_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.json"
        payload = {
            "schema_version": "N72R10_TRAINING_FAILURE_V1",
            "status": f"FAIL_N72R10_TRAINING_{args.mode.upper()}",
            "stage": f"N72R10_TRAINING_{args.mode.upper()}",
            "started_at_utc": started,
            "finished_at_utc": now_utc(),
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
            "runtime_future_gt_used": False,
            "failure_artifact": str(failure),
        }
        atomic_json(failure, payload)
        atomic_json(status_path, payload)
        print(json.dumps({"status": payload["status"], "failure_artifact": str(failure), "error": str(exc)}, sort_keys=True))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
