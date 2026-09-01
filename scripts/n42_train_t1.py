#!/usr/bin/env python3
"""Train the isolated N42 T1 pairwise association calibration head."""

from __future__ import annotations

import hashlib
import json
import os
import sys
import tempfile
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.n36_real_eval_common import atomic_json
from scripts.n42_t1_common import FEATURE_DIM, PairwiseCalibrationHead, checkpoint_digest, feature_contract


TRAIN_DIR = ROOT / "outputs/n42/training"
PROTOCOL_PATH = TRAIN_DIR / "training_protocol.json"
DATASET_PATH = TRAIN_DIR / "pair_dataset.jsonl"
DATASET_MANIFEST = TRAIN_DIR / "dataset_manifest.json"
SMOKE_CKPT = TRAIN_DIR / "t1_smoke_checkpoint.pt"
FULL_CKPT = TRAIN_DIR / "t1_pairwise_calibration.pt"
SMOKE_STATUS = TRAIN_DIR / "smoke_status.json"
FULL_MANIFEST = TRAIN_DIR / "full_training_manifest.json"
STAGE_STATUS = ROOT / "outputs/n42/stage_02_status.json"
FAILURE_PATH = ROOT / "outputs/n42/attempts/training_failure.json"
SEED = 4242


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_torch_save(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    os.close(fd)
    try:
        torch.save(value, name)
        with open(name, "rb") as handle:
            os.fsync(handle.fileno())
        os.replace(name, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def load_dataset() -> dict[str, tuple[np.ndarray, np.ndarray]]:
    arrays: dict[str, list[list[float]]] = {"train": [], "validation": [], "holdout": []}
    labels: dict[str, list[float]] = {"train": [], "validation": [], "holdout": []}
    with DATASET_PATH.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            split = str(row["split"])
            if split not in arrays:
                raise ValueError(f"unknown split {split}")
            feature = np.asarray(row["features"], dtype=np.float32).reshape(-1)
            if feature.size != FEATURE_DIM or not np.all(np.isfinite(feature)):
                raise ValueError("nonfinite or wrong-dimensional training feature")
            arrays[split].append(feature.tolist())
            labels[split].append(float(row["label"]))
    output = {}
    for split in arrays:
        x = np.asarray(arrays[split], dtype=np.float32)
        y = np.asarray(labels[split], dtype=np.float32)
        if x.ndim != 2 or x.shape[1] != FEATURE_DIM or y.ndim != 1 or len(x) != len(y):
            raise ValueError(f"invalid split tensors {split}: {x.shape}/{y.shape}")
        output[split] = (x, y)
    return output


def loader(x: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool, seed: int) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(int(seed))
    return DataLoader(TensorDataset(torch.from_numpy(x), torch.from_numpy(y)), batch_size=batch_size, shuffle=shuffle, generator=generator, drop_last=False)


def evaluate(model: PairwiseCalibrationHead, x: np.ndarray, y: np.ndarray, device: torch.device, batch_size: int = 1024) -> float:
    model.eval()
    criterion = torch.nn.BCEWithLogitsLoss()
    values = []
    with torch.no_grad():
        for xb, yb in loader(x, y, batch_size, False, SEED):
            loss = criterion(model(xb.to(device)), yb.to(device))
            values.append(float(loss.detach().cpu()))
    if not values:
        return float("nan")
    return float(np.mean(values))


def checkpoint_payload(model: PairwiseCalibrationHead, protocol: dict[str, Any], dataset: dict[str, Any], phase: str, epoch: int, train_loss: float, val_loss: float) -> dict[str, Any]:
    return {
        "protocol": "N42_T1_PAIRWISE_CALIBRATION_V1",
        "phase": phase,
        "input_dim": FEATURE_DIM,
        "feature_contract": feature_contract(),
        "architecture": "Linear(23,64)-ReLU-Linear(64,32)-ReLU-Linear(32,1)",
        "state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
        "training_protocol_sha256": sha256(PROTOCOL_PATH),
        "training_dataset_sha256": sha256(DATASET_PATH),
        "dataset_manifest_sha256": sha256(DATASET_MANIFEST),
        "seed": SEED,
        "epoch": int(epoch),
        "train_loss": float(train_loss),
        "validation_loss": float(val_loss),
        "calibration_application_scale": 1.0,
        "runtime_future_gt_used": False,
        "production_authorized": False,
    }


def run_smoke(device: torch.device, data: dict[str, tuple[np.ndarray, np.ndarray]], protocol: dict[str, Any]) -> dict[str, Any]:
    x, y = data["train"]
    count = min(int(protocol["smoke"]["max_rows"]), len(x))
    indices = np.arange(count, dtype=int)
    model = PairwiseCalibrationHead().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(protocol["training"]["learning_rate"]), weight_decay=float(protocol["training"]["weight_decay"]))
    criterion = torch.nn.BCEWithLogitsLoss()
    losses = []
    train_loader = loader(x[indices], y[indices], min(128, count), False, SEED)
    iterator = iter(train_loader)
    for step in range(int(protocol["smoke"]["steps"])):
        try:
            xb, yb = next(iterator)
        except StopIteration:
            iterator = iter(train_loader)
            xb, yb = next(iterator)
        optimizer.zero_grad(set_to_none=True)
        loss = criterion(model(xb.to(device)), yb.to(device))
        if not torch.isfinite(loss):
            raise RuntimeError(f"smoke loss nonfinite at step {step}")
        loss.backward()
        grad_ok = all(parameter.grad is None or torch.all(torch.isfinite(parameter.grad)) for parameter in model.parameters())
        if not grad_ok:
            raise RuntimeError(f"smoke gradient nonfinite at step {step}")
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
    payload = checkpoint_payload(model, protocol, json.loads(DATASET_MANIFEST.read_text()), "smoke", int(protocol["smoke"]["steps"]), float(np.mean(losses)), float(np.mean(losses)))
    atomic_torch_save(SMOKE_CKPT, payload)
    restored = torch.load(SMOKE_CKPT, map_location="cpu", weights_only=False)
    reload_model = PairwiseCalibrationHead()
    reload_model.load_state_dict(restored["state_dict"])
    with torch.no_grad():
        test_out = reload_model(torch.from_numpy(x[: min(4, len(x))]))
    if not torch.all(torch.isfinite(test_out)):
        raise RuntimeError("smoke save/reload output is nonfinite")
    result = {
        "protocol": "N42_T1_PAIRWISE_CALIBRATION_SMOKE_V1",
        "status": "PASS",
        "device": str(device),
        "rows": int(count),
        "steps": int(protocol["smoke"]["steps"]),
        "losses": losses,
        "loss_finite": True,
        "gradient_finite": True,
        "checkpoint": str(SMOKE_CKPT.relative_to(ROOT)),
        "checkpoint_sha256": checkpoint_digest(SMOKE_CKPT),
        "save_reload": "PASS",
        "runtime_future_gt_used": False,
    }
    atomic_json(SMOKE_STATUS, result)
    return result


def run_full(device: torch.device, data: dict[str, tuple[np.ndarray, np.ndarray]], protocol: dict[str, Any]) -> dict[str, Any]:
    train_x, train_y = data["train"]
    val_x, val_y = data["validation"]
    holdout_x, holdout_y = data["holdout"]
    model = PairwiseCalibrationHead().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(protocol["training"]["learning_rate"]), weight_decay=float(protocol["training"]["weight_decay"]))
    criterion = torch.nn.BCEWithLogitsLoss()
    best_val = float("inf")
    best_epoch = 0
    best_payload = None
    wait = 0
    history = []
    max_epochs = int(protocol["training"]["epochs"])
    for epoch in range(1, max_epochs + 1):
        model.train()
        losses = []
        for xb, yb in loader(train_x, train_y, int(protocol["training"]["batch_size"]), True, SEED + epoch):
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(xb.to(device)), yb.to(device))
            if not torch.isfinite(loss):
                raise RuntimeError(f"full train loss nonfinite at epoch {epoch}")
            loss.backward()
            if not all(parameter.grad is None or torch.all(torch.isfinite(parameter.grad)) for parameter in model.parameters()):
                raise RuntimeError(f"full train gradient nonfinite at epoch {epoch}")
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        train_loss = float(np.mean(losses))
        val_loss = evaluate(model, val_x, val_y, device)
        if not np.isfinite(val_loss):
            raise RuntimeError(f"validation loss nonfinite at epoch {epoch}")
        history.append({"epoch": epoch, "train_loss": train_loss, "validation_loss": val_loss})
        if val_loss < best_val:
            best_val = val_loss
            best_epoch = epoch
            wait = 0
            best_payload = checkpoint_payload(model, protocol, json.loads(DATASET_MANIFEST.read_text()), "full", epoch, train_loss, val_loss)
        else:
            wait += 1
            if wait >= int(protocol["training"]["early_stopping_patience"]):
                break
    if best_payload is None:
        raise RuntimeError("no finite validation checkpoint produced")
    atomic_torch_save(FULL_CKPT, best_payload)
    restored = torch.load(FULL_CKPT, map_location=device, weights_only=False)
    model.load_state_dict(restored["state_dict"])
    holdout_loss = evaluate(model, holdout_x, holdout_y, device)
    if not np.isfinite(holdout_loss):
        raise RuntimeError("holdout descriptive loss nonfinite")
    result = {
        "protocol": "N42_T1_PAIRWISE_CALIBRATION_TRAINING_V1",
        "status": "PASS",
        "device": str(device),
        "seed": SEED,
        "configuration": protocol["training"],
        "train_count": int(len(train_x)),
        "validation_count": int(len(val_x)),
        "holdout_count": int(len(holdout_x)),
        "completed_epochs": int(len(history)),
        "best_epoch": int(best_epoch),
        "best_validation_loss": float(best_val),
        "holdout_descriptive_loss_not_used_for_selection": float(holdout_loss),
        "history": history,
        "checkpoint": str(FULL_CKPT.relative_to(ROOT)),
        "checkpoint_sha256": checkpoint_digest(FULL_CKPT),
        "training_protocol_sha256": sha256(PROTOCOL_PATH),
        "dataset_sha256": sha256(DATASET_PATH),
        "runtime_future_gt_used": False,
        "production_authorized": False,
    }
    atomic_json(FULL_MANIFEST, result)
    return result


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("smoke", "full"), required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    started = now()
    try:
        protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
        data = load_dataset()
        torch.manual_seed(SEED)
        np.random.seed(SEED)
        device = torch.device(args.device)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("requested CUDA device is unavailable")
        if args.phase == "smoke":
            result = run_smoke(device, data, protocol)
            atomic_json(STAGE_STATUS, {"protocol": "N42_T1_PAIRWISE_CALIBRATION_TRAINING_V1", "stage": "N42-02-smoke", "status": "SMOKE_PASS", "started_at": started, "finished_at": now(), "smoke": result, "t0_baseline": "not_trained_frozen_baseline", "runtime_future_gt_used": False, "production_authorized": False})
        else:
            result = run_full(device, data, protocol)
            atomic_json(STAGE_STATUS, {"protocol": "N42_T1_PAIRWISE_CALIBRATION_TRAINING_V1", "stage": "N42-02", "status": "TRAINING_PASS", "started_at": started, "finished_at": now(), "smoke_status": str(SMOKE_STATUS.relative_to(ROOT)), "full_training": result, "t0_baseline": "not_trained_frozen_baseline", "runtime_future_gt_used": False, "production_authorized": False})
        print(json.dumps({"status": result["status"], "phase": args.phase, "checkpoint": result.get("checkpoint")}, sort_keys=True), flush=True)
    except Exception as exc:
        failure = {"protocol": "N42_T1_PAIRWISE_CALIBRATION_TRAINING_V1", "phase": args.phase, "status": "FAIL", "started_at": started, "finished_at": now(), "exception": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc(), "failure_preserved": True}
        if not FAILURE_PATH.exists():
            atomic_json(FAILURE_PATH, failure)
        atomic_json(STAGE_STATUS, failure)
        raise


if __name__ == "__main__":
    main()
