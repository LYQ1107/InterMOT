#!/usr/bin/env python3
"""Train the isolated N71 candidate-by-identity scorer.

The training input is the materialized N71 matrix dataset.  Sequence splits,
the optimizer, stopping rule, and checkpoint-selection rule are frozen by the
N71 protocol.  This script never imports the production association runtime
and never uses future replay metrics for sample or checkpoint selection.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import tempfile
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import n71_global_matrix_common as common  # noqa: E402


def atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
        directory_fd = os.open(str(path.parent), os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def atomic_torch_save(path: Path, payload: object) -> None:
    import torch

    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    os.close(fd)
    try:
        torch.save(payload, tmp)
        with open(tmp, "rb") as handle:
            os.fsync(handle.fileno())
        os.replace(tmp, path)
        directory_fd = os.open(str(path.parent), os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def load_protocol() -> tuple[dict[str, Any], Path]:
    path = ROOT / "outputs/N71/protocol.json"
    protocol = json.loads(path.read_text(encoding="utf-8"))
    return protocol, path


def evaluate_split(
    model: Any,
    arrays: dict[str, np.ndarray],
    groups: list[int],
    mean: np.ndarray,
    std: np.ndarray,
    device: Any,
    max_cells: int,
) -> dict[str, float]:
    """Evaluate a split without gradients, with batch-size accounting."""
    import torch

    totals = {"total": 0.0, "cell_bce": 0.0, "group_softmax": 0.0, "hard_negative_margin": 0.0, "none_presence": 0.0}
    weight = 0
    model.eval()
    with torch.no_grad():
        for batch_groups, indices in common.group_batches(arrays, groups, max_cells=max_cells, rng=None):
            _loss, detail = common.group_loss(model, arrays, batch_groups, indices, mean, std, device)
            count = len(indices)
            weight += count
            for key in totals:
                totals[key] += float(detail[key]) * count
    if weight <= 0:
        raise RuntimeError("requested split has no cells")
    return {key: value / weight for key, value in totals.items()}


def cpu_state_dict(model: Any) -> dict[str, Any]:
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def run_training(
    manifest_path: Path,
    checkpoint_path: Path,
    output_path: Path,
    status_path: Path,
    device_name: str,
    attempt: int,
) -> dict[str, Any]:
    import torch

    protocol, protocol_path = load_protocol()
    training = protocol["training"]
    seed = int(training["seed"])
    common.set_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.use_deterministic_algorithms(False)

    device = torch.device(device_name)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError(f"N71 training requires CUDA, got {device}")
    torch.cuda.set_device(device)

    arrays, dataset_manifest = common.load_arrays(manifest_path)
    if dataset_manifest.get("runtime_future_gt_used") is not False:
        raise RuntimeError("dataset manifest permits runtime future GT")
    train_groups = common.group_ids_for_split(arrays, "train")
    validation_groups = common.group_ids_for_split(arrays, "validation")
    holdout_groups = common.group_ids_for_split(arrays, "holdout")
    if not train_groups or not validation_groups or not holdout_groups:
        raise RuntimeError("sequence-disjoint train/validation/holdout split is incomplete")

    # The normalizer is fit only on train groups and is saved with the model.
    mean, std = common.context_normalization(arrays)
    model = common.build_model().to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
    )
    max_epochs = int(training["max_epochs"])
    patience = int(training["early_stopping_patience"])
    batch_cells = int(training["batch_cells"])
    clip_norm = float(training["gradient_clip_norm"])
    best_validation = float("inf")
    best_epoch = -1
    best_state: dict[str, Any] | None = None
    bad_epochs = 0
    history: list[dict[str, Any]] = []
    started = time.time()

    for epoch in range(max_epochs):
        model.train()
        rng = np.random.default_rng(seed + epoch)
        train_sums = {"total": 0.0, "cell_bce": 0.0, "group_softmax": 0.0, "hard_negative_margin": 0.0, "none_presence": 0.0}
        train_weight = 0
        batches = 0
        for batch_groups, indices in common.group_batches(arrays, train_groups, max_cells=batch_cells, rng=rng):
            loss, detail = common.group_loss(model, arrays, batch_groups, indices, mean, std, device)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"nonfinite training loss at epoch={epoch} batch={batches}")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm).detach().cpu())
            if not np.isfinite(grad_norm):
                raise FloatingPointError(f"nonfinite training gradient at epoch={epoch} batch={batches}")
            optimizer.step()
            count = len(indices)
            train_weight += count
            batches += 1
            for key in train_sums:
                train_sums[key] += float(detail[key]) * count
        if train_weight <= 0:
            raise RuntimeError("training produced no batches")
        train_metrics = {key: value / train_weight for key, value in train_sums.items()}
        validation_metrics = evaluate_split(model, arrays, validation_groups, mean, std, device, batch_cells)
        validation_composite = float(validation_metrics["total"])
        if not np.isfinite(validation_composite):
            raise FloatingPointError(f"nonfinite validation composite at epoch={epoch}")
        improved = validation_composite < best_validation
        if improved:
            best_validation = validation_composite
            best_epoch = epoch
            best_state = cpu_state_dict(model)
            bad_epochs = 0
        else:
            bad_epochs += 1
        record = {
            "epoch": epoch,
            "batches": batches,
            "train": train_metrics,
            "validation": validation_metrics,
            "validation_composite": validation_composite,
            "improved": improved,
            "gradient_clip_norm": clip_norm,
            "bad_epochs": bad_epochs,
        }
        history.append(record)
        print(json.dumps({"epoch": epoch, "train_total": train_metrics["total"], "validation_total": validation_composite, "improved": improved, "bad_epochs": bad_epochs}, sort_keys=True), flush=True)
        if bad_epochs >= patience:
            break

    if best_state is None or best_epoch < 0:
        raise RuntimeError("no validation checkpoint was selected")
    model.load_state_dict(best_state)
    # Holdout is deliberately read only after checkpoint selection and is not
    # used to choose weights, thresholds, samples, or branches.
    holdout_metrics = evaluate_split(model, arrays, holdout_groups, mean, std, device, batch_cells)
    if not all(np.isfinite(value) for value in holdout_metrics.values()):
        raise FloatingPointError("nonfinite descriptive holdout loss")

    checkpoint_payload = {
        "schema": "N71_GLOBAL_MATRIX_SCORER_CHECKPOINT_V1",
        "state_dict": best_state,
        "context_mean": mean.tolist(),
        "context_std": std.tolist(),
        "model_metadata": common.model_metadata(model),
        "config": {
            "seed": seed,
            "optimizer": training["optimizer"],
            "learning_rate": float(training["learning_rate"]),
            "weight_decay": float(training["weight_decay"]),
            "max_epochs": max_epochs,
            "early_stopping_patience": patience,
            "batch_cells": batch_cells,
            "gradient_clip_norm": clip_norm,
            "loss_coefficients": training["loss_coefficients"],
            "ranking_margin": float(training["ranking_margin"]),
            "best_epoch": best_epoch,
            "checkpoint_selection": training["checkpoint_selection"],
        },
        "dataset_manifest": str(manifest_path),
        "dataset_manifest_sha256": common.sha256(manifest_path),
        "protocol": str(protocol_path),
        "protocol_sha256": common.sha256(protocol_path),
        "sequence_disjoint_split": True,
        "holdout_used_for_selection": False,
        "runtime_future_gt_used": False,
        "interaction_source": "simulated_from_gt",
        "production_authorized": False,
    }
    atomic_torch_save(checkpoint_path, checkpoint_payload)
    result = {
        "schema": "N71_GLOBAL_MATRIX_TRAINING_V1",
        "status": "PASS_TRAINED_VALIDATION_SELECTED_HOLDOUT_DESCRIPTIVE_ONLY",
        "attempt": int(attempt),
        "started_unix": started,
        "ended_unix": time.time(),
        "duration_seconds": time.time() - started,
        "manifest": str(manifest_path),
        "manifest_sha256": common.sha256(manifest_path),
        "protocol": str(protocol_path),
        "protocol_sha256": common.sha256(protocol_path),
        "device": str(device),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "cuda_device_name": torch.cuda.get_device_name(device),
        "model": common.model_metadata(model),
        "split_group_counts": {"train": len(train_groups), "validation": len(validation_groups), "holdout": len(holdout_groups)},
        "sequence_disjoint_split": True,
        "seed": seed,
        "best_epoch": best_epoch,
        "best_validation_composite": best_validation,
        "history": history,
        "holdout_descriptive_only": holdout_metrics,
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": common.sha256(checkpoint_path),
        "runtime_future_gt_used": False,
        "interaction_source": "simulated_from_gt",
        "numeric_public_id_feature": False,
        "numeric_target_native_id_feature": False,
        "holdout_used_for_selection": False,
        "production_authorized": False,
    }
    atomic_json(output_path, result)
    atomic_json(status_path, {
        "schema": "N71_STAGE_04_STATUS_V1",
        "status": "PASS_T1_GLOBAL_MATRIX_SCORER_TRAINED",
        "training_result": str(output_path),
        "training_result_sha256": common.sha256(output_path),
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": common.sha256(checkpoint_path),
        "manifest": str(manifest_path),
        "manifest_sha256": common.sha256(manifest_path),
        "protocol_sha256": common.sha256(protocol_path),
        "best_epoch": best_epoch,
        "holdout_used_for_selection": False,
        "runtime_future_gt_used": False,
        "interaction_source": "simulated_from_gt",
        "production_authorized": False,
        "next": "paired_replay_on_frozen_N70_24_events_and_temporal_branch",
    })
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=common.DATA_MANIFEST)
    parser.add_argument("--checkpoint", type=Path, default=Path("/path/to/cache/SAM3_InterMOT_N71/training/N71_GLOBAL_MATRIX_SCORER_ATTEMPT1.pt"))
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/N71/training/global_matrix_training_attempt1.json")
    parser.add_argument("--status", type=Path, default=ROOT / "outputs/N71/stage_04_status.json")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--attempt", type=int, default=1)
    args = parser.parse_args()
    manifest = args.manifest.resolve()
    checkpoint = args.checkpoint.resolve()
    output = args.output.resolve()
    status = args.status.resolve()
    try:
        result = run_training(manifest, checkpoint, output, status, args.device, args.attempt)
        print(json.dumps(result, indent=2, sort_keys=True))
    except Exception as exc:
        attempts = ROOT / "outputs/N71/attempts"
        attempts.mkdir(parents=True, exist_ok=True)
        existing = sorted(attempts.glob("n71_global_matrix_training_failure_attempt*.json"))
        failure_path = attempts / f"n71_global_matrix_training_failure_attempt{len(existing) + 1}.json"
        failure = {
            "schema": "N71_GLOBAL_MATRIX_TRAINING_FAILURE_V1",
            "status": "FAIL_PRESERVED",
            "attempt": int(args.attempt),
            "failure_type": type(exc).__name__,
            "failure_message": str(exc),
            "traceback": traceback.format_exc(),
            "manifest": str(manifest),
            "manifest_sha256": common.sha256(manifest),
            "checkpoint_target": str(checkpoint),
            "device": args.device,
            "runtime_future_gt_used": False,
            "production_authorized": False,
        }
        atomic_json(failure_path, failure)
        atomic_json(status, {
            "schema": "N71_STAGE_04_STATUS_V1",
            "status": "FAIL_T1_TRAINING_PRESERVED",
            "failure_artifact": str(failure_path),
            "failure_artifact_sha256": common.sha256(failure_path),
            "failure_type": type(exc).__name__,
            "failure_message": str(exc),
            "manifest": str(manifest),
            "manifest_sha256": common.sha256(manifest),
            "runtime_future_gt_used": False,
            "production_authorized": False,
        })
        print(json.dumps({"status": "FAIL_PRESERVED", "failure_artifact": str(failure_path), "error": f"{type(exc).__name__}: {exc}"}, sort_keys=True))
        raise


if __name__ == "__main__":
    main()
