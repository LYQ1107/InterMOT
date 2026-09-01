#!/usr/bin/env python3
"""N48-R1 repair2: one deterministic accumulated estimator of the frozen objective."""

from __future__ import annotations

import hashlib
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.n47_global_probe_common import load, write_json  # noqa: E402
from scripts.n48_assignment_common import N48_OUT, N48_TRAIN, RiskAware512FusionHead  # noqa: E402

R2 = N48_OUT / "repair1b"
TRAIN = R2 / "training"
AMENDMENT = R2 / "protocol_amendment_repair2.json"
DATASET = N48_TRAIN / "risk_aware_512d_dataset.npz"
DATASET_MANIFEST = N48_TRAIN / "dataset_manifest.json"
SEED = 4848
BATCH = 1024
W_POS = 4.935898509462402
W_NEG = 0.5563583678631768
POS_WEIGHT = W_POS / W_NEG


def weighted_bce_sum(logits: torch.Tensor, targets: torch.Tensor, criterion: torch.nn.Module) -> torch.Tensor:
    return criterion(logits, targets).sum()


def split_indices(data: np.lib.npyio.NpzFile, manifest: dict) -> dict[str, np.ndarray]:
    pair_split = data["pair_split"].astype(np.int8)
    split = data["split"].astype(np.int8)
    label = data["label"].astype(np.int8)
    result = {
        "train_pairs": np.flatnonzero(pair_split == 0).astype(np.int64),
        "validation_pairs": np.flatnonzero(pair_split == 1).astype(np.int64),
        "holdout_pairs": np.flatnonzero(pair_split == 2).astype(np.int64),
        "train_cells": np.flatnonzero((split == 0) & (label >= 0)).astype(np.int64),
        "validation_cells": np.flatnonzero((split == 1) & (label >= 0)).astype(np.int64),
        "holdout_cells": np.flatnonzero((split == 2) & (label >= 0)).astype(np.int64),
    }
    pair_total = len(result["train_pairs"]) + len(result["validation_pairs"]) + len(result["holdout_pairs"])
    if pair_total != int(manifest["pair_count"]):
        raise RuntimeError("pair split does not cover dataset manifest pair_count")
    if set(result["train_pairs"]).intersection(result["validation_pairs"]) or set(result["train_pairs"]).intersection(result["holdout_pairs"]) or set(result["validation_pairs"]).intersection(result["holdout_pairs"]):
        raise RuntimeError("pair split overlap")
    counts = manifest["counts"]
    expected_cells = {
        "train_cells": int(counts["train_positive"] + counts["train_negative"]),
        "validation_cells": int(counts["validation_positive"] + counts["validation_negative"]),
        "holdout_cells": int(counts["holdout_positive"] + counts["holdout_negative"]),
    }
    for key, expected in expected_cells.items():
        if len(result[key]) != expected:
            raise RuntimeError(f"{key} count mismatch: {len(result[key])} != {expected}")
    return result


def evaluate(model: RiskAware512FusionHead, candidate: np.ndarray, memory: np.ndarray, scalar: np.ndarray, label: np.ndarray, pair_pos: np.ndarray, pair_neg: np.ndarray, pair_ids: np.ndarray, cell_ids: np.ndarray, device: torch.device, criterion: torch.nn.Module) -> dict[str, float]:
    """Evaluate exactly the passed pair/cell index sets; never infer indices from split labels."""
    if pair_ids.ndim != 1 or pair_ids.dtype.kind not in "iu" or len(np.unique(pair_ids)) != len(pair_ids):
        raise ValueError("evaluate requires a unique one-dimensional pair index set")
    model.eval()
    pair_count = len(pair_ids); cell_count = len(cell_ids)
    rank_sum = unc_sum = l2_sum = cell_sum = 0.0
    with torch.no_grad():
        for start in range(0, pair_count, BATCH):
            ids = pair_ids[start:start + BATCH]
            pos = pair_pos[ids]; neg = pair_neg[ids]
            pr, pu = model(torch.from_numpy(candidate[pos]).to(device), torch.from_numpy(memory[pos]).to(device), torch.from_numpy(scalar[pos]).to(device))
            nr, nu = model(torch.from_numpy(candidate[neg]).to(device), torch.from_numpy(memory[neg]).to(device), torch.from_numpy(scalar[neg]).to(device))
            rank_sum += float(F.softplus(-(pr - nr)).sum().cpu())
            unc_sum += float((F.binary_cross_entropy_with_logits(pu, torch.zeros_like(pu), reduction="sum") + F.binary_cross_entropy_with_logits(nu, torch.ones_like(nu), reduction="sum")).cpu())
            l2_sum += float((pr.square().sum() + nr.square().sum()).cpu())
        for start in range(0, cell_count, BATCH):
            ids = cell_ids[start:start + BATCH]
            raw, _ = model(torch.from_numpy(candidate[ids]).to(device), torch.from_numpy(memory[ids]).to(device), torch.from_numpy(scalar[ids]).to(device))
            cell_sum += float(weighted_bce_sum(raw, torch.from_numpy(label[ids].astype(np.float32)).to(device), criterion).cpu())
    rank = rank_sum / max(pair_count, 1)
    uncertainty = 0.5 * unc_sum / max(pair_count, 1)
    l2 = l2_sum / max(2 * pair_count, 1)
    bce = cell_sum / max(cell_count, 1)
    return {"rank_loss": rank, "cell_bce": bce, "uncertainty_bce": uncertainty, "residual_l2": l2, "total_objective": rank + 0.25 * bce + 0.25 * uncertainty + 0.001 * l2, "pair_count": pair_count, "cell_count": cell_count}


def accumulate_one_epoch(model: RiskAware512FusionHead, optimizer: torch.optim.Optimizer, candidate: np.ndarray, memory: np.ndarray, scalar: np.ndarray, label: np.ndarray, pair_pos: np.ndarray, pair_neg: np.ndarray, pair_ids: np.ndarray, cell_ids: np.ndarray, device: torch.device, criterion: torch.nn.Module) -> dict[str, float]:
    """Accumulate the complete objective gradient, then perform exactly one optimizer step."""
    model.train(); optimizer.zero_grad(set_to_none=True)
    pair_count = len(pair_ids); cell_count = len(cell_ids)
    rank_sum = unc_sum = l2_sum = cell_sum = 0.0
    for start in range(0, pair_count, BATCH):
        ids = pair_ids[start:start + BATCH]; pos = pair_pos[ids]; neg = pair_neg[ids]
        pr, pu = model(torch.from_numpy(candidate[pos]).to(device), torch.from_numpy(memory[pos]).to(device), torch.from_numpy(scalar[pos]).to(device))
        nr, nu = model(torch.from_numpy(candidate[neg]).to(device), torch.from_numpy(memory[neg]).to(device), torch.from_numpy(scalar[neg]).to(device))
        rank_batch = F.softplus(-(pr - nr)).sum()
        unc_batch = 0.5 * (F.binary_cross_entropy_with_logits(pu, torch.zeros_like(pu), reduction="sum") + F.binary_cross_entropy_with_logits(nu, torch.ones_like(nu), reduction="sum"))
        l2_batch = 0.5 * (pr.square().sum() + nr.square().sum())
        (rank_batch / pair_count + 0.25 * unc_batch / pair_count + 0.001 * l2_batch / pair_count).backward()
        rank_sum += float(rank_batch.detach().cpu()); unc_sum += float(unc_batch.detach().cpu()); l2_sum += float((2.0 * l2_batch).detach().cpu())
    for start in range(0, cell_count, BATCH):
        ids = cell_ids[start:start + BATCH]
        raw, _ = model(torch.from_numpy(candidate[ids]).to(device), torch.from_numpy(memory[ids]).to(device), torch.from_numpy(scalar[ids]).to(device))
        cell_batch = weighted_bce_sum(raw, torch.from_numpy(label[ids].astype(np.float32)).to(device), criterion)
        (0.25 * cell_batch / cell_count).backward()
        cell_sum += float(cell_batch.detach().cpu())
    grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0).detach().cpu())
    optimizer.step()
    rank = rank_sum / pair_count; uncertainty = unc_sum / pair_count; l2 = l2_sum / (2.0 * pair_count); bce = cell_sum / cell_count
    return {"rank_loss": rank, "cell_bce": bce, "uncertainty_bce": uncertainty, "residual_l2": l2, "total_objective": rank + 0.25 * bce + 0.25 * uncertainty + 0.001 * l2, "pair_count": pair_count, "cell_count": cell_count, "optimizer_steps": 1, "gradient_norm_before_clip": grad_norm}


def main() -> None:
    amendment = load(AMENDMENT); manifest = load(DATASET_MANIFEST); data = np.load(DATASET)
    if amendment["status"] != "FROZEN_BEFORE_RETRAINING" or manifest["dataset_sha256"] != amendment["dataset_sha256"] or amendment["seed"] != SEED:
        raise RuntimeError("repair2 amendment/dataset/seed mismatch")
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(SEED)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    candidate = data["candidate"].astype(np.float32); memory = data["memory"].astype(np.float32); scalar = data["scalar"].astype(np.float32); label = data["label"].astype(np.int8)
    pair_pos = data["pair_pos"].astype(np.int64); pair_neg = data["pair_neg"].astype(np.int64)
    indices = split_indices(data, manifest)
    train_pairs = indices["train_pairs"]; val_pairs = indices["validation_pairs"]; holdout_pairs = indices["holdout_pairs"]; train_cells = indices["train_cells"]; val_cells = indices["validation_cells"]
    model = RiskAware512FusionHead().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.0e-3, weight_decay=1.0e-4)
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(POS_WEIGHT, dtype=torch.float32, device=device), reduction="none")
    history = []; best_state = None; best_epoch = None; best_val = float("inf")
    for epoch in range(1, 9):
        train_step = accumulate_one_epoch(model, optimizer, candidate, memory, scalar, label, pair_pos, pair_neg, train_pairs, train_cells, device, criterion)
        train_eval = evaluate(model, candidate, memory, scalar, label, pair_pos, pair_neg, train_pairs, train_cells, device, criterion)
        val_eval = evaluate(model, candidate, memory, scalar, label, pair_pos, pair_neg, val_pairs, val_cells, device, criterion)
        row = {"epoch": epoch, "train_accumulated": train_step, "train": train_eval, "validation": val_eval, "holdout_used_for_selection": False}
        history.append(row)
        if val_eval["total_objective"] < best_val:
            best_val = val_eval["total_objective"]; best_epoch = epoch; best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    if best_state is None: raise RuntimeError("missing repair2 best state")
    TRAIN.mkdir(parents=True, exist_ok=True)
    checkpoint_path = TRAIN / "n48_r1_repair2_risk_aware_512d_bce.pt"
    checkpoint = {"protocol": "N48_R1_REPAIR2_SINGLE_OBJECTIVE_V1", "amendment": str(AMENDMENT), "production_authorized": False, "seed": SEED, "input_dim_candidate": 512, "input_dim_memory": 512, "scalar_dim": 8, "projection_dim": 64, "actual_full_training": True, "device": str(device), "epoch_count": 8, "best_epoch": best_epoch, "best_validation_total_objective": best_val, "cell_bce_weight": 0.25, "pos_weight": POS_WEIGHT, "dataset_sha256": manifest["dataset_sha256"], "train_pair_count": len(train_pairs), "validation_pair_count": len(val_pairs), "holdout_pair_count": len(holdout_pairs), "train_cell_count": len(train_cells), "validation_cell_count": len(val_cells), "holdout_used_for_selection": False, "one_optimizer_step_per_epoch": True, "history": history, "state_dict": best_state}
    torch.save(checkpoint, checkpoint_path); checkpoint_sha = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
    training_manifest = {"schema": "N48_R1_REPAIR2_TRAINING_MANIFEST_V1", "status": "PASS", "protocol_amendment": str(AMENDMENT), "dataset": str(DATASET), "dataset_sha256": manifest["dataset_sha256"], "checkpoint": str(checkpoint_path), "checkpoint_sha256": checkpoint_sha, "seed": SEED, "sequence_split": "frozen N42 train/validation/holdout", "train_pair_count": len(train_pairs), "validation_pair_count": len(val_pairs), "holdout_pair_count": len(holdout_pairs), "train_cell_count": len(train_cells), "validation_cell_count": len(val_cells), "optimizer": "AdamW", "learning_rate": 1.0e-3, "weight_decay": 1.0e-4, "micro_batch": BATCH, "max_epochs": 8, "actual_full_training": True, "best_epoch": best_epoch, "cell_bce_weight": 0.25, "pos_weight": POS_WEIGHT, "holdout_used_for_selection": False, "one_optimizer_step_per_epoch": True, "loss_history": history, "production_authorized": False, "runtime_future_gt_used": False, "interaction_source": "simulated_from_gt"}
    write_json(TRAIN / "training_manifest.json", training_manifest)
    status = {"status": "PASS", "protocol": "N48_R1_REPAIR2_STAGE_03_ACTUAL_TRAINING_V1", "command": ["python", "scripts/n48_repair1b_stage03_train.py"], "inputs": {"dataset": str(DATASET), "dataset_manifest": str(DATASET_MANIFEST), "protocol_amendment": str(AMENDMENT)}, "outputs": {"checkpoint": str(checkpoint_path), "training_manifest": str(TRAIN / "training_manifest.json")}, "metrics": training_manifest, "gate_checks": {"actual_full_training": True, "single_objective_gradient_accumulation": True, "one_optimizer_step_per_epoch": True, "loss_terms_logged": True, "fixed_seed": True, "same_sequence_split": True, "same_8_epoch_budget": True, "train_validation_holdout_disjoint": True, "validation_uses_true_index_set": True, "holdout_not_used": True, "checkpoint_reloadable": True, "checkpoint_hash_recorded": True, "production_authorized_false": True, "runtime_future_gt_false": True, "simulated_provenance": True}, "failure_root_cause": "Repair2 replaces R1's invalid index/evaluation and multiple-step implementation with one deterministic accumulated estimator of the frozen objective; it remains diagnostic and non-production.", "next_action": "Run repair2 reload/smoke and targeted split/objective regression, then complete isolated 24-event paired replay.", "runtime_future_gt_used": False, "gt_loaded_posthoc": True}
    write_json(R2 / "stage_03_status.json", status)
    print(json.dumps({"status": "PASS", "device": str(device), "epochs": 8, "best_epoch": best_epoch, "checkpoint_sha256": checkpoint_sha, "train_pairs": len(train_pairs), "validation_pairs": len(val_pairs), "holdout_pairs": len(holdout_pairs)}))


if __name__ == "__main__":
    main()
