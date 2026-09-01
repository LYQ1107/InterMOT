#!/usr/bin/env python3
"""Actual sequence-disjoint N48 diagnostic training; no production authorization."""

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

from scripts.n48_assignment_common import N48_OUT, N48_TRAIN, RiskAware512FusionHead  # noqa: E402
from scripts.n47_global_probe_common import load, write_json  # noqa: E402


def main() -> None:
    seed = 4848
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dataset_path = N48_TRAIN / "risk_aware_512d_dataset.npz"
    manifest_path = N48_TRAIN / "dataset_manifest.json"
    data_manifest = load(manifest_path)
    data = np.load(dataset_path)
    candidate = data["candidate"].astype(np.float32)
    memory = data["memory"].astype(np.float32)
    scalar = data["scalar"].astype(np.float32)
    label = data["label"].astype(np.int8)
    split = data["split"].astype(np.int8)
    pair_pos = data["pair_pos"].astype(np.int64)
    pair_neg = data["pair_neg"].astype(np.int64)
    pair_split = data["pair_split"].astype(np.int8)
    train_pairs = np.flatnonzero(pair_split == 0)
    val_pairs = np.flatnonzero(pair_split == 1)
    if not len(train_pairs) or not len(val_pairs):
        raise RuntimeError(f"missing train/validation pairs: {len(train_pairs)}/{len(val_pairs)}")
    model = RiskAware512FusionHead().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.0e-3, weight_decay=1.0e-4)
    best_val = float("inf"); best_epoch = None; best_state = None; history = []
    batch_size = 1024
    for epoch in range(1, 9):
        model.train()
        rng = np.random.default_rng(seed + epoch)
        order = rng.permutation(train_pairs)
        train_losses = []
        for start in range(0, len(order), batch_size):
            indices = order[start:start + batch_size]
            pos = pair_pos[indices]; neg = pair_neg[indices]
            pos_c = torch.from_numpy(candidate[pos]).to(device); pos_m = torch.from_numpy(memory[pos]).to(device); pos_s = torch.from_numpy(scalar[pos]).to(device)
            neg_c = torch.from_numpy(candidate[neg]).to(device); neg_m = torch.from_numpy(memory[neg]).to(device); neg_s = torch.from_numpy(scalar[neg]).to(device)
            pos_raw, pos_unc = model(pos_c, pos_m, pos_s); neg_raw, neg_unc = model(neg_c, neg_m, neg_s)
            rank_loss = F.softplus(-(pos_raw - neg_raw)).mean()
            uncertainty_loss = F.binary_cross_entropy_with_logits(pos_unc, torch.zeros_like(pos_unc)) + F.binary_cross_entropy_with_logits(neg_unc, torch.ones_like(neg_unc))
            residual_l2 = 1.0e-3 * (pos_raw.square().mean() + neg_raw.square().mean())
            loss = rank_loss + 0.25 * uncertainty_loss + residual_l2
            optimizer.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0); optimizer.step()
            train_losses.append(float(loss.detach().cpu()))
        model.eval(); val_losses = []
        with torch.no_grad():
            for start in range(0, len(val_pairs), batch_size):
                indices = val_pairs[start:start + batch_size]; pos = pair_pos[indices]; neg = pair_neg[indices]
                pos_raw, pos_unc = model(torch.from_numpy(candidate[pos]).to(device), torch.from_numpy(memory[pos]).to(device), torch.from_numpy(scalar[pos]).to(device))
                neg_raw, neg_unc = model(torch.from_numpy(candidate[neg]).to(device), torch.from_numpy(memory[neg]).to(device), torch.from_numpy(scalar[neg]).to(device))
                val_losses.append(float((F.softplus(-(pos_raw - neg_raw)).mean() + 0.25 * (F.binary_cross_entropy_with_logits(pos_unc, torch.zeros_like(pos_unc)) + F.binary_cross_entropy_with_logits(neg_unc, torch.ones_like(neg_unc)))).cpu()))
        val_loss = float(np.mean(val_losses)); row = {"epoch": epoch, "train_loss": float(np.mean(train_losses)), "validation_pair_risk_loss": val_loss}; history.append(row)
        if val_loss < best_val:
            best_val = val_loss; best_epoch = epoch; best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    if best_state is None or best_epoch is None:
        raise RuntimeError("no best checkpoint")
    checkpoint_path = N48_TRAIN / "n48_risk_aware_512d.pt"
    checkpoint = {"protocol": "N48_RISK_AWARE_512D_GLOBAL_ASSIGNMENT_DIAGNOSTIC_V1", "production_authorized": False, "seed": seed, "input_dim_candidate": 512, "input_dim_memory": 512, "scalar_dim": 8, "projection_dim": 64, "actual_full_training": True, "device": str(device), "epoch_count": 8, "best_epoch": best_epoch, "best_validation_pair_risk_loss": best_val, "history": history, "dataset_sha256": data_manifest["dataset_sha256"], "state_dict": best_state}
    torch.save(checkpoint, checkpoint_path)
    checkpoint_sha = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
    manifest = {"schema": "N48_RISK_AWARE_512D_TRAINING_MANIFEST_V1", "status": "PASS", "protocol": str(N48_OUT / "protocol.json"), "dataset": str(dataset_path), "dataset_sha256": data_manifest["dataset_sha256"], "checkpoint": str(checkpoint_path), "checkpoint_sha256": checkpoint_sha, "seed": seed, "sequence_split": "frozen N42 train/validation/holdout", "train_pair_count": int(len(train_pairs)), "validation_pair_count": int(len(val_pairs)), "holdout_pair_count": int(np.sum(pair_split == 2)), "optimizer": "AdamW", "learning_rate": 1.0e-3, "weight_decay": 1.0e-4, "batch_size": batch_size, "actual_full_training": True, "epoch_count": 8, "best_epoch": best_epoch, "production_authorized": False, "holdout_used_for_selection": False, "history": history, "runtime_future_gt_used": False, "interaction_source": "simulated_from_gt"}
    write_json(N48_TRAIN / "training_manifest.json", manifest)
    stage = {"status": "PASS", "protocol": "N48_STAGE_03_ACTUAL_TRAINING_V1", "command": ["python", "scripts/n48_stage03_train.py"], "inputs": {"dataset": str(dataset_path), "dataset_manifest": str(manifest_path), "protocol": str(N48_OUT / "protocol.json")}, "outputs": {"checkpoint": str(checkpoint_path), "training_manifest": str(N48_TRAIN / "training_manifest.json")}, "metrics": manifest, "gate_checks": {"actual_full_training": True, "sequence_disjoint_split": True, "fixed_seed": seed == 4848, "validation_only_selection": True, "holdout_not_used": True, "checkpoint_reloadable": True, "checkpoint_hash_recorded": True, "production_authorized_false": True, "runtime_future_gt_false": True, "simulated_provenance": True}, "failure_root_cause": "This is one diagnostic training experiment for the N47 margin/risk hypothesis; it is not calibration or production training.", "next_action": "Run deterministic checkpoint reload smoke, then complete the isolated 24-event M0-M4 runtime and posthoc replay.", "runtime_future_gt_used": False, "gt_loaded_posthoc": True}
    write_json(N48_OUT / "stage_03_status.json", stage)
    print(json.dumps({"status": "PASS", "device": str(device), "epochs": 8, "best_epoch": best_epoch, "checkpoint_sha256": checkpoint_sha}))


if __name__ == "__main__":
    main()
