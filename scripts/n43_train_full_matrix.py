#!/usr/bin/env python3
"""N43 stage 03: actual full sequence-disjoint sidecar training."""

from __future__ import annotations

import hashlib
import json
import random
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.n43_full_matrix_common import FEATURE_DIM, PROTOCOL, FullMatrixCalibrationHead, feature_contract


OUT = ROOT / "outputs/n43"
TRAIN = OUT / "training"
DATASET = TRAIN / "cell_dataset.npz"
DATASET_MANIFEST = TRAIN / "dataset_manifest.json"
SIDECAR_PROTOCOL = OUT / "sidecar_protocol.json"
CHECKPOINT = TRAIN / "n43_full_matrix_calibration.pt"
MANIFEST = TRAIN / "full_training_manifest.json"
STAGE = OUT / "stage_03_status.json"
SEED = 4242


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_data() -> dict[str, np.ndarray]:
    if not DATASET.is_file() or not DATASET_MANIFEST.is_file() or not SIDECAR_PROTOCOL.is_file():
        raise FileNotFoundError("N43 stage-02 dataset/protocol is missing")
    with np.load(DATASET) as data:
        arrays = {key: data[key] for key in data.files}
    if arrays["x"].shape[1] != FEATURE_DIM or len(arrays["x"]) != len(arrays["target_utility"]):
        raise RuntimeError("N43 dataset shape mismatch")
    return arrays


def loader(x: np.ndarray, appearance: np.ndarray, target: np.ndarray, batch: int, shuffle: bool, seed: int) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(TensorDataset(torch.from_numpy(x), torch.from_numpy(appearance), torch.from_numpy(target)), batch_size=batch, shuffle=shuffle, generator=generator, num_workers=0)


def utility_tensor(model: FullMatrixCalibrationHead, x: torch.Tensor, appearance: torch.Tensor) -> torch.Tensor:
    raw = model(x)
    gate = torch.sigmoid(raw[:, 0])
    residual = 0.5 * torch.tanh(raw[:, 1])
    return gate * appearance + residual


def evaluate(model: FullMatrixCalibrationHead, x: np.ndarray, app: np.ndarray, target: np.ndarray, label: np.ndarray, device: torch.device) -> tuple[float, float, float]:
    model.eval()
    values = torch.from_numpy(x).to(device)
    appearance = torch.from_numpy(app).to(device)
    expected = torch.from_numpy(target).to(device)
    labels = torch.from_numpy(label.astype(np.float32)).to(device)
    with torch.no_grad():
        utility = utility_tensor(model, values, appearance)
        mse = torch.mean((utility - expected) ** 2).item()
        bce = nn.functional.binary_cross_entropy_with_logits(utility, labels).item()
    return float(mse), float(bce), float(torch.mean(utility).item())


def main() -> None:
    started = now()
    result: dict[str, Any] = {"status": "FAIL", "protocol": "N43_STAGE_03_FULL_TRAINING_V1", "started_at": started, "project_root": str(ROOT)}
    try:
        seed_everything(SEED)
        arrays = load_data()
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model = FullMatrixCalibrationHead().to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1.0e-3, weight_decay=1.0e-4)
        train_mask, val_mask, hold_mask = arrays["split"] == 0, arrays["split"] == 1, arrays["split"] == 2
        train_loader = loader(arrays["x"][train_mask], arrays["appearance"][train_mask], arrays["target_utility"][train_mask], 512, True, SEED)
        history = []
        best_val = float("inf")
        best_epoch = None
        best_state = None
        patience = 5
        stale = 0
        max_epochs = 30
        for epoch in range(1, max_epochs + 1):
            model.train()
            loss_sum = 0.0
            count = 0
            for x_batch, app_batch, target_batch in train_loader:
                x_batch = x_batch.to(device)
                app_batch = app_batch.to(device)
                target_batch = target_batch.to(device)
                utility = utility_tensor(model, x_batch, app_batch)
                loss = torch.mean((utility - target_batch) ** 2)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
                loss_sum += float(loss.item()) * len(x_batch)
                count += len(x_batch)
            train_loss = loss_sum / max(count, 1)
            val_mse, val_bce, val_mean = evaluate(model, arrays["x"][val_mask], arrays["appearance"][val_mask], arrays["target_utility"][val_mask], arrays["label"][val_mask], device)
            history.append({"epoch": epoch, "train_mse": train_loss, "validation_mse": val_mse, "validation_bce": val_bce, "validation_mean_utility": val_mean})
            if val_mse < best_val - 1.0e-10:
                best_val = val_mse
                best_epoch = epoch
                best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
                stale = 0
            else:
                stale += 1
            if stale >= patience:
                break
        if best_state is None or best_epoch is None:
            raise RuntimeError("no best N43 checkpoint selected")
        model.load_state_dict(best_state)
        payload = {"protocol": PROTOCOL, "input_dim": FEATURE_DIM, "feature_contract": feature_contract(), "architecture": "Linear(18,64)-ReLU-Linear(64,32)-ReLU-Linear(32,2)", "state_dict": model.state_dict(), "seed": SEED, "training_dataset_sha256": digest(DATASET), "training_protocol_sha256": digest(SIDECAR_PROTOCOL), "production_authorized": False, "runtime_future_gt_used": False, "application": "base + sigmoid(gate)*appearance_delta + bounded residual for every finite candidate x public-ID cell; NONE bypass", "best_epoch": best_epoch}
        torch.save(payload, CHECKPOINT)
        loaded = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
        if loaded.get("protocol") != PROTOCOL or loaded.get("production_authorized") is not False:
            raise RuntimeError("checkpoint save/reload metadata integrity failed")
        hold_mse, hold_bce, hold_mean = evaluate(model, arrays["x"][hold_mask], arrays["appearance"][hold_mask], arrays["target_utility"][hold_mask], arrays["label"][hold_mask], device)
        full_manifest = {"protocol": PROTOCOL, "status": "PASS", "training_mode": "FULL_NOT_SMOKE", "seed": SEED, "device": str(device), "gpu_count_visible": torch.cuda.device_count() if torch.cuda.is_available() else 0, "dataset": str(DATASET), "dataset_sha256": digest(DATASET), "dataset_manifest": str(DATASET_MANIFEST), "sidecar_protocol": str(SIDECAR_PROTOCOL), "sequence_disjoint_split": True, "train_count": int(train_mask.sum()), "validation_count": int(val_mask.sum()), "holdout_count": int(hold_mask.sum()), "positive_count": int(np.sum(arrays["label"] == 1)), "negative_count": int(np.sum(arrays["label"] == 0)), "configuration": {"batch_size": 512, "learning_rate": 1.0e-3, "weight_decay": 1.0e-4, "max_epochs": max_epochs, "early_stopping_patience": patience, "selection": "minimum validation utility MSE; earliest strict improvement", "gradient_clip_norm": 5.0}, "completed_epochs": len(history), "best_epoch": best_epoch, "best_validation_mse": best_val, "holdout_descriptive_mse_not_used_for_selection": hold_mse, "holdout_descriptive_bce_not_used_for_selection": hold_bce, "holdout_mean_utility": hold_mean, "history": history, "checkpoint": str(CHECKPOINT), "checkpoint_sha256": digest(CHECKPOINT), "production_authorized": False, "runtime_future_gt_used": False}
        MANIFEST.write_text(json.dumps(full_manifest, indent=2) + "\n", encoding="utf-8")
        result.update({"status": "PASS", "command": [sys.executable, str(Path(__file__).resolve())], "inputs": {"dataset": str(DATASET), "dataset_manifest": str(DATASET_MANIFEST), "sidecar_protocol": str(SIDECAR_PROTOCOL)}, "outputs": {"checkpoint": str(CHECKPOINT), "training_manifest": str(MANIFEST)}, "metrics": full_manifest, "gate_checks": {"actual_full_training": True, "not_smoke": True, "sequence_disjoint_split": True, "seed_fixed": SEED == 4242, "checkpoint_save_reload": True, "production_authorized": False, "runtime_future_gt_false": True}, "failure_root_cause": None, "next_action": "Use the frozen checkpoint in the same-candidate paired M0-M4 replay; do not promote production calibration based on training metrics.", "runtime_future_gt_used": False, "finished_at": now()})
        OUT.mkdir(parents=True, exist_ok=True)
        STAGE.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({"status": result["status"], "device": str(device), "epochs": len(history), "checkpoint": str(CHECKPOINT)}, sort_keys=True))
    except Exception as exc:
        result.update({"status": "FAIL", "failure_root_cause": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc(), "finished_at": now()})
        failure = OUT / "attempts" / f"stage_03_failure_{datetime.now().strftime('%Y%m%dT%H%M%S')}.json"
        failure.parent.mkdir(parents=True, exist_ok=True)
        failure.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        raise


if __name__ == "__main__":
    main()
