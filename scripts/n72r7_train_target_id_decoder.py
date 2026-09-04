#!/usr/bin/env python3
"""Train the first learned N72R7 target-candidate decoder.

This script has two explicit modes.  ``smoke`` proves forward/backward and
checkpoint round-trip on sealed tensor shapes.  ``train`` then performs one
fixed sequence-disjoint training run.  It never loads the deferred
confirmation split and never uses replay-effect metrics for optimization or
checkpoint selection.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import tempfile
import traceback
from typing import Any, Mapping

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
from torch import nn

from sam3_intermot.reacquisition.models.target_id_decoder import (  # noqa: E402
    CANDIDATE_FEATURE_DIM,
    CONTEXT_FEATURE_DIM,
    HumanConditionedTargetIDDecoder,
    set_decoder_loss,
)


def resolve_root_path(value: str | None, default: Path) -> Path:
    if value is None:
        return default
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


TRAINING_ROOT = resolve_root_path(os.environ.get("N72R7_TRAINING_ROOT"), ROOT / "outputs/N72R7/training")
PROTOCOL = TRAINING_ROOT / "training_protocol.json"
CORPUS = TRAINING_ROOT / "corpus_manifest.json"
TRAIN_NPZ = TRAINING_ROOT / "train.npz"
VAL_NPZ = TRAINING_ROOT / "validation.npz"
TRAIN_META = TRAINING_ROOT / "train_metadata.jsonl"
VAL_META = TRAINING_ROOT / "validation_metadata.jsonl"
CHECKPOINT = TRAINING_ROOT / "HumanConditionedTargetIDDecoder_v1.pt"
SMOKE_CHECKPOINT = TRAINING_ROOT / "smoke_decoder_roundtrip.pt"
WEIGHT_ARTIFACT = TRAINING_ROOT / "train_hard_negative_weights.json"
STAGE_SMOKE = resolve_root_path(
    os.environ.get("N72R7_TRAINING_SMOKE_STATUS"),
    ROOT / "outputs/N72R7/stage_06_training_smoke_status.json",
)
STAGE = resolve_root_path(
    os.environ.get("N72R7_TRAINING_STATUS"),
    ROOT / "outputs/N72R7/stage_06_status.json",
)
SEED = 7202
MAX_EPOCHS = 80
PATIENCE = 12
BATCH_SIZE = 256
LEARNING_RATE = 2.0e-4
WEIGHT_DECAY = 1.0e-4
PAIRWISE_WEIGHT = 0.15
HARD_NEGATIVE_WEIGHTING = "anchor_cosine_margin_v1"


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
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".pt", dir=str(path.parent))
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


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def load_split(npz_path: Path, metadata_path: Path) -> tuple[dict[str, np.ndarray], list[dict[str, Any]]]:
    if not npz_path.is_file() or not metadata_path.is_file():
        raise FileNotFoundError(f"missing sealed split: {npz_path} / {metadata_path}")
    loaded = np.load(npz_path, allow_pickle=False)
    arrays = {key: np.asarray(loaded[key]) for key in loaded.files}
    required = {"candidate_features", "candidate_mask", "context_features", "labels", "candidate_counts"}
    if not required.issubset(arrays):
        raise RuntimeError(f"split schema missing keys: {npz_path}")
    count = len(arrays["labels"])
    if arrays["candidate_features"].shape[:2] != arrays["candidate_mask"].shape:
        raise RuntimeError(f"candidate tensor/mask mismatch: {npz_path}")
    if arrays["candidate_features"].shape[0] != count or arrays["context_features"].shape[0] != count:
        raise RuntimeError(f"split row count mismatch: {npz_path}")
    if arrays["candidate_features"].shape[-1] != CANDIDATE_FEATURE_DIM or arrays["context_features"].shape[-1] != CONTEXT_FEATURE_DIM:
        raise RuntimeError(f"split feature width mismatch: {npz_path}")
    if not np.isfinite(arrays["candidate_features"]).all() or not np.isfinite(arrays["context_features"]).all():
        raise RuntimeError(f"non-finite sealed tensor: {npz_path}")
    if not np.isfinite(arrays["labels"]).all() or not np.isfinite(arrays["candidate_counts"]).all():
        raise RuntimeError(f"non-finite labels: {npz_path}")
    metadata = [json.loads(line) for line in metadata_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(metadata) != count:
        raise RuntimeError(f"metadata/tensor count mismatch: {npz_path}")
    for index, item in enumerate(metadata):
        if int(item["label_index"]) != int(arrays["candidate_counts"][index]) and item["label_kind"] == "NONE":
            # The metadata keeps the per-example candidate count; the tensor
            # stores the split-wide NONE index.  Both are audited explicitly.
            raise RuntimeError(f"NONE metadata/candidate count mismatch at {npz_path}:{index}")
    return arrays, metadata


def model_config() -> dict[str, Any]:
    return {
        "candidate_feature_dim": CANDIDATE_FEATURE_DIM,
        "context_feature_dim": CONTEXT_FEATURE_DIM,
        "hidden_dim": 256,
        "layers": 2,
        "heads": 4,
        "dropout": 0.0,
        "architecture": "candidate_set_transformer_plus_target_context_cross_attention_plus_explicit_NONE",
    }


def new_model() -> HumanConditionedTargetIDDecoder:
    return HumanConditionedTargetIDDecoder(**{key: value for key, value in model_config().items() if key in {"candidate_feature_dim", "context_feature_dim", "hidden_dim", "layers", "heads", "dropout"}})


def tensor_batch(arrays: Mapping[str, np.ndarray], indices: np.ndarray, device: torch.device) -> tuple[torch.Tensor, ...]:
    return (
        torch.as_tensor(arrays["candidate_features"][indices], dtype=torch.float32, device=device),
        torch.as_tensor(arrays["candidate_mask"][indices], dtype=torch.bool, device=device),
        torch.as_tensor(arrays["context_features"][indices], dtype=torch.float32, device=device),
        torch.as_tensor(arrays["labels"][indices], dtype=torch.long, device=device),
    )


def sealed_hard_negative_weights(arrays: Mapping[str, np.ndarray]) -> tuple[np.ndarray, dict[str, Any]]:
    """Derive fixed train-only weights from frozen features and offline labels.

    The weighting emphasizes examples where the strongest non-target candidate
    is close to, or stronger than, the labeled target in the anchor-cosine
    feature.  It uses no future metric, replay outcome, public ID, or runtime
    state.  The exact per-example values are saved as an audit sidecar.
    """

    candidates = np.asarray(arrays["candidate_features"], dtype=np.float64)
    mask = np.asarray(arrays["candidate_mask"], dtype=bool)
    contexts = np.asarray(arrays["context_features"], dtype=np.float64)
    labels = np.asarray(arrays["labels"], dtype=np.int64)
    counts = np.asarray(arrays["candidate_counts"], dtype=np.int64)
    if candidates.ndim != 3 or mask.ndim != 2 or contexts.ndim != 2 or labels.ndim != 1 or counts.ndim != 1:
        raise ValueError("invalid arrays for sealed hard-negative weighting")
    if candidates.shape[:2] != mask.shape or candidates.shape[0] != len(labels) or len(counts) != len(labels):
        raise ValueError("hard-negative weighting array count mismatch")
    if candidates.shape[-1] < 512 or contexts.shape[-1] < 512:
        raise ValueError("hard-negative weighting requires 512-D appearance prefixes")
    weights = np.ones(len(labels), dtype=np.float64)
    none_index = candidates.shape[1]
    target_examples = 0
    none_examples = 0
    hard_examples = 0
    hardness_values: list[float] = []
    for row_index in range(len(labels)):
        count = int(counts[row_index])
        valid = mask[row_index] & (np.arange(mask.shape[1]) < count)
        if not bool(valid.any()):
            raise ValueError(f"example has no valid candidates at index {row_index}")
        anchor = contexts[row_index, :512]
        anchor_norm = float(np.linalg.norm(anchor))
        if anchor_norm <= 1.0e-8 or not np.isfinite(anchor_norm):
            raise ValueError(f"invalid anchor feature at index {row_index}")
        values = candidates[row_index, :, :512]
        norms = np.linalg.norm(values, axis=1)
        finite = np.isfinite(values).all(axis=1) & np.isfinite(norms) & (norms > 1.0e-8)
        if not bool(np.all(finite[valid])):
            raise ValueError(f"invalid candidate appearance feature at index {row_index}")
        scores = np.full(mask.shape[1], -1.0, dtype=np.float64)
        scores[valid] = values[valid].dot(anchor / anchor_norm) / norms[valid]
        label = int(labels[row_index])
        if label < count:
            if not bool(valid[label]):
                raise ValueError(f"target label points to padded candidate at index {row_index}")
            target_score = float(scores[label])
            negative_scores = scores[valid].copy()
            negative_scores[label] = -1.0
            strongest_negative = float(np.max(negative_scores))
            hardness = float(np.clip(strongest_negative - target_score + 0.05, 0.0, 1.0))
            weights[row_index] = 1.0 + hardness
            target_examples += 1
            hardness_values.append(hardness)
            hard_examples += int(hardness > 0.0)
        elif label == none_index:
            strongest = float(np.max(scores[valid]))
            hardness = float(np.clip(strongest, 0.0, 1.0))
            weights[row_index] = 1.0 + 0.25 * hardness
            none_examples += 1
            hardness_values.append(0.25 * hardness)
        else:
            raise ValueError(f"label outside per-example candidate/NONE range at index {row_index}")
    if not np.isfinite(weights).all() or bool((weights <= 0.0).any()):
        raise ValueError("derived hard-negative weights are non-finite or non-positive")
    return weights.astype(np.float32), {
        "scheme": HARD_NEGATIVE_WEIGHTING,
        "uses_only": ["frozen_candidate_appearance_prefix", "frozen_context_anchor_prefix", "offline_train_label"],
        "future_effect_metrics_used": False,
        "public_id_used": False,
        "target_examples": target_examples,
        "none_examples": none_examples,
        "target_hard_examples": hard_examples,
        "weight_min": float(np.min(weights)) if len(weights) else None,
        "weight_median": float(np.median(weights)) if len(weights) else None,
        "weight_p90": float(np.percentile(weights, 90)) if len(weights) else None,
        "weight_max": float(np.max(weights)) if len(weights) else None,
        "hardness_mean": float(np.mean(hardness_values)) if hardness_values else None,
    }


def evaluate(model: nn.Module, arrays: Mapping[str, np.ndarray], metadata: list[dict[str, Any]], device: torch.device) -> dict[str, Any]:
    model.eval()
    candidate_count = arrays["candidate_features"].shape[1]
    predictions: list[int] = []
    losses: list[float] = []
    with torch.no_grad():
        for start in range(0, len(arrays["labels"]), BATCH_SIZE):
            indices = np.arange(start, min(start + BATCH_SIZE, len(arrays["labels"])))
            candidates, mask, context, labels = tensor_batch(arrays, indices, device)
            logits = model(candidates, mask, context)
            loss, _ = set_decoder_loss(logits, labels, mask, pairwise_weight=PAIRWISE_WEIGHT)
            losses.append(float(loss.detach().cpu()))
            predictions.extend(torch.argmax(logits, dim=1).detach().cpu().tolist())
    labels = arrays["labels"].astype(np.int64)
    predictions_array = np.asarray(predictions, dtype=np.int64)
    if len(predictions_array) != len(labels):
        raise RuntimeError("prediction count mismatch")
    actual_none = labels == candidate_count
    predicted_none = predictions_array == candidate_count
    target_labels = ~actual_none
    target_accuracy = float(np.mean(predictions_array[target_labels] == labels[target_labels])) if bool(target_labels.any()) else None
    none_accuracy = float(np.mean(predicted_none[actual_none])) if bool(actual_none.any()) else None
    all_accuracy = float(np.mean(predictions_array == labels)) if len(labels) else None
    by_action: dict[str, dict[str, Any]] = {}
    for action in sorted({str(item["action_type"]) for item in metadata}):
        indices = np.asarray([i for i, item in enumerate(metadata) if str(item["action_type"]) == action], dtype=np.int64)
        action_labels = labels[indices]
        action_pred = predictions_array[indices]
        action_none = action_labels == candidate_count
        action_target = ~action_none
        by_action[action] = {
            "examples": int(len(indices)),
            "accuracy": float(np.mean(action_pred == action_labels)) if len(indices) else None,
            "target_candidate_accuracy": float(np.mean(action_pred[action_target] == action_labels[action_target])) if bool(action_target.any()) else None,
            "none_accuracy": float(np.mean((action_pred == candidate_count)[action_none])) if bool(action_none.any()) else None,
            "target_examples": int(action_target.sum()),
            "none_examples": int(action_none.sum()),
        }
    return {
        "examples": int(len(labels)),
        "loss": float(np.mean(losses)) if losses else None,
        "accuracy": all_accuracy,
        "target_candidate_accuracy": target_accuracy,
        "none_accuracy": none_accuracy,
        "target_examples": int(target_labels.sum()),
        "none_examples": int(actual_none.sum()),
        "by_action": by_action,
    }


def run_smoke(device: torch.device) -> dict[str, Any]:
    arrays, _ = load_split(TRAIN_NPZ, TRAIN_META)
    model = new_model().to(device)
    set_seed(SEED)
    count = min(8, len(arrays["labels"]))
    candidates, mask, context, labels = tensor_batch(arrays, np.arange(count), device)
    model.train()
    logits = model(candidates, mask, context)
    if not bool(torch.isfinite(logits).all()):
        raise RuntimeError("smoke logits are non-finite")
    smoke_weights = torch.ones(count, dtype=torch.float32, device=device)
    loss, parts = set_decoder_loss(
        logits,
        labels,
        mask,
        pairwise_weight=PAIRWISE_WEIGHT,
        example_weight=smoke_weights,
    )
    if not bool(torch.isfinite(loss)):
        raise RuntimeError("smoke loss is non-finite")
    loss.backward()
    gradients = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
    if not gradients or not all(bool(torch.isfinite(gradient).all()) for gradient in gradients):
        raise RuntimeError("smoke gradients are missing or non-finite")
    atomic_torch_save(SMOKE_CHECKPOINT, {"model_config": model_config(), "state_dict": model.state_dict(), "seed": SEED})
    restored = new_model().to(device)
    payload = torch.load(SMOKE_CHECKPOINT, map_location=device, weights_only=False)
    restored.load_state_dict(payload["state_dict"])
    model.eval()
    restored.eval()
    with torch.no_grad():
        eval_logits = model(candidates, mask, context)
        restored_logits = restored(candidates, mask, context)
    if not bool(torch.allclose(eval_logits, restored_logits, atol=1.0e-6, rtol=1.0e-6)):
        raise RuntimeError("smoke checkpoint round-trip changed logits")
    result = {
        "schema_version": "N72R7_DECODER_TRAINING_SMOKE_V1",
        "status": "PASS_TRAINING_SMOKE",
        "created_at_utc": now_utc(),
        "device": str(device),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "batch_examples": count,
        "model_config": model_config(),
        "loss": float(loss.detach().cpu()),
        "loss_parts": {key: float(value.cpu()) for key, value in parts.items()},
        "hard_negative_weighting": HARD_NEGATIVE_WEIGHTING,
        "checkpoint": str(SMOKE_CHECKPOINT),
        "checkpoint_sha256": sha256_file(SMOKE_CHECKPOINT),
        "forward_finite": True,
        "backward_finite": True,
        "save_restore_equivalent": True,
        "runtime_future_gt_used": False,
        "production_authorized": False,
    }
    atomic_json(STAGE_SMOKE, result)
    return result


def run_train(device: torch.device) -> dict[str, Any]:
    if not STAGE_SMOKE.is_file() or json.loads(STAGE_SMOKE.read_text(encoding="utf-8")).get("status") != "PASS_TRAINING_SMOKE":
        raise RuntimeError("training smoke must pass before the actual training run")
    train_arrays, train_metadata = load_split(TRAIN_NPZ, TRAIN_META)
    val_arrays, val_metadata = load_split(VAL_NPZ, VAL_META)
    if len(train_arrays["labels"]) == 0 or len(val_arrays["labels"]) == 0:
        raise RuntimeError("train or validation split is empty")
    started_at = now_utc()
    train_weights, weight_summary = sealed_hard_negative_weights(train_arrays)
    atomic_json(
        WEIGHT_ARTIFACT,
        {
            "schema_version": "N72R7_SEALED_TRAIN_EXAMPLE_WEIGHTS_V1",
            "status": "PASS_SEALED_TRAIN_ONLY_WEIGHTS",
            "created_at_utc": now_utc(),
            "scheme": HARD_NEGATIVE_WEIGHTING,
            "weights": [float(value) for value in train_weights],
            "summary": weight_summary,
            "train_npz": str(TRAIN_NPZ),
            "train_npz_sha256": sha256_file(TRAIN_NPZ),
            "future_effect_metrics_used": False,
            "runtime_future_gt_used": False,
        },
    )
    set_seed(SEED)
    model = new_model().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    generator = torch.Generator(device="cpu").manual_seed(SEED)
    best_loss = float("inf")
    best_epoch = 0
    stale = 0
    history: list[dict[str, Any]] = []
    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        order = torch.randperm(len(train_arrays["labels"]), generator=generator).numpy()
        losses: list[float] = []
        ce_values: list[float] = []
        pair_values: list[float] = []
        for start in range(0, len(order), BATCH_SIZE):
            indices = order[start : start + BATCH_SIZE]
            candidates, mask, context, labels = tensor_batch(train_arrays, indices, device)
            logits = model(candidates, mask, context)
            batch_weights = torch.as_tensor(train_weights[indices], dtype=torch.float32, device=device)
            loss, parts = set_decoder_loss(
                logits,
                labels,
                mask,
                pairwise_weight=PAIRWISE_WEIGHT,
                example_weight=batch_weights,
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
            ce_values.append(float(parts["cross_entropy"].cpu()))
            pair_values.append(float(parts["hard_negative_ranking"].cpu()))
        validation = evaluate(model, val_arrays, val_metadata, device)
        train_loss = float(np.mean(losses))
        epoch_record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_cross_entropy": float(np.mean(ce_values)),
            "train_hard_negative_ranking": float(np.mean(pair_values)),
            "validation_loss": validation["loss"],
            "validation_accuracy": validation["accuracy"],
            "validation_target_candidate_accuracy": validation["target_candidate_accuracy"],
            "validation_none_accuracy": validation["none_accuracy"],
        }
        history.append(epoch_record)
        validation_loss = float(validation["loss"])
        if validation_loss < best_loss - 1.0e-8:
            best_loss = validation_loss
            best_epoch = epoch
            stale = 0
            atomic_torch_save(
                CHECKPOINT,
                {
                    "schema_version": "N72R7_HUMAN_CONDITIONED_TARGET_ID_DECODER_CHECKPOINT_V1",
                    "model_config": model_config(),
                    "state_dict": model.state_dict(),
                    "seed": SEED,
                    "epoch": epoch,
                    "validation_loss": validation_loss,
                },
            )
        else:
            stale += 1
            if stale >= PATIENCE:
                break
    if best_epoch == 0 or not CHECKPOINT.is_file():
        raise RuntimeError("training did not produce a best checkpoint")
    payload = torch.load(CHECKPOINT, map_location=device, weights_only=False)
    model.load_state_dict(payload["state_dict"])
    final_train = evaluate(model, train_arrays, train_metadata, device)
    final_validation = evaluate(model, val_arrays, val_metadata, device)
    if not np.isfinite(float(final_train["loss"])) or not np.isfinite(float(final_validation["loss"])):
        raise RuntimeError("final decoder metrics are non-finite")
    peak_memory = None
    if torch.cuda.is_available():
        peak_memory = int(torch.cuda.max_memory_allocated())
    manifest = {
        "schema_version": "N72R7_HUMAN_CONDITIONED_TARGET_ID_DECODER_TRAINING_V1",
        "status": "PASS_ACTUAL_TRAINING_COMPLETED",
        "created_at_utc": now_utc(),
        "protocol": str(PROTOCOL),
        "protocol_sha256": sha256_file(PROTOCOL),
        "corpus_manifest": str(CORPUS),
        "corpus_manifest_sha256": sha256_file(CORPUS),
        "train_npz": str(TRAIN_NPZ),
        "train_npz_sha256": sha256_file(TRAIN_NPZ),
        "validation_npz": str(VAL_NPZ),
        "validation_npz_sha256": sha256_file(VAL_NPZ),
        "model_config": model_config(),
        "seed": SEED,
        "device": str(device),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "gpu_count_used": 1 if device.type == "cuda" else 0,
        "precision": "float32",
        "peak_memory_allocated_bytes": peak_memory,
        "optimizer": {"name": "AdamW", "learning_rate": LEARNING_RATE, "weight_decay": WEIGHT_DECAY, "batch_size": BATCH_SIZE},
        "loss": {"primary": "set-level cross entropy over candidates plus explicit NONE", "auxiliary": "hard-negative pairwise hinge", "pairwise_weight": PAIRWISE_WEIGHT, "pairwise_margin": 0.20},
        "hard_negative_weighting": {
            **weight_summary,
            "artifact": str(WEIGHT_ARTIFACT),
            "artifact_sha256": sha256_file(WEIGHT_ARTIFACT),
        },
        "early_stopping": {"metric": "sequence-disjoint validation loss", "max_epochs": MAX_EPOCHS, "patience": PATIENCE, "best_epoch": best_epoch, "best_validation_loss": best_loss},
        "history": history,
        "metrics": {"train": final_train, "validation": final_validation},
        "checkpoint": str(CHECKPOINT),
        "checkpoint_sha256": sha256_file(CHECKPOINT),
        "training_labels_gt_only_offline": True,
        "validation_labels_gt_only_offline": True,
        "confirmation_loaded": False,
        "future_effect_metrics_used_for_selection": False,
        "runtime_future_gt_used": False,
        "public_id_inference": False,
        "sam3_checkpoint_modified": False,
        "production_authorized": False,
        "scientific_result": "TRAINED_CANDIDATE_NOT_YET_EVALUATED_IN_CLOSED_LOOP",
    }
    manifest_path = TRAINING_ROOT / "decoder_training_manifest.json"
    atomic_json(manifest_path, manifest)
    result = {
        "schema_version": "N72R7_STAGE_06_STATUS_V1",
        "status": manifest["status"],
        "started_at_utc": started_at,
        "finished_at_utc": now_utc(),
        "training_manifest": str(manifest_path),
        "training_manifest_sha256": sha256_file(manifest_path),
        "checkpoint": str(CHECKPOINT),
        "checkpoint_sha256": sha256_file(CHECKPOINT),
        "metrics": manifest["metrics"],
        "best_epoch": best_epoch,
        "device": str(device),
        "gpu_count_used": manifest["gpu_count_used"],
        "runtime_future_gt_used": False,
        "production_authorized": False,
        "next_action": "Run learned D1/D2 closed-loop replay with frozen checkpoint; do not promote automatically.",
    }
    atomic_json(STAGE, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("smoke", "train"), default="smoke")
    args = parser.parse_args()
    result: dict[str, Any] = {"status": "FAIL", "mode": args.mode, "started_at_utc": now_utc()}
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if args.mode == "smoke":
            result = run_smoke(device)
        else:
            result = run_train(device)
        print(json.dumps({"status": result["status"], "mode": args.mode, "device": str(device)}, sort_keys=True))
    except Exception as exc:
        result.update({"failure_root_cause": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc(), "finished_at_utc": now_utc(), "device": str(torch.device("cuda" if torch.cuda.is_available() else "cpu"))})
        failure = ROOT / "outputs/N72R7/attempts" / f"n72r7_decoder_{args.mode}_failure_{datetime.now().strftime('%Y%m%dT%H%M%S')}.json"
        atomic_json(failure, result)
        raise


if __name__ == "__main__":
    main()
