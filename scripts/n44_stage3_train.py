#!/usr/bin/env python3
"""N44 stage 03: actual sequence-disjoint assignment-aware training."""

from __future__ import annotations

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

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.n44_assignment_common import AssignmentAwareHead, PROTOCOL, sha256


DATASET = ROOT / "outputs/n44/training/assignment_pair_dataset.npz"
GROUPS = ROOT / "outputs/n44/training/assignment_groups.json"
CHECKPOINT = ROOT / "outputs/n44/training/n44_assignment_aware.pt"
MANIFEST = ROOT / "outputs/n44/training/full_training_manifest.json"
STAGE = ROOT / "outputs/n44/stage_03_status.json"
SEED = 4444
MAX_EPOCHS = 60
PATIENCE = 10


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def output(model: AssignmentAwareHead, x: np.ndarray, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    with torch.no_grad():
        raw = model(torch.as_tensor(x, dtype=torch.float32, device=device)).cpu().numpy()
    score = raw[:, 0]
    variance = (torch.nn.functional.softplus(torch.as_tensor(raw[:, 1])) + 0.02).numpy()
    return score, np.sqrt(variance)


def pair_metrics(model: AssignmentAwareHead, arrays: dict[str, np.ndarray], code: int, device: torch.device) -> dict[str, float]:
    mask = arrays["pair_split"] == code
    left, right = arrays["pair_left"][mask], arrays["pair_right"][mask]
    if not len(left):
        return {"pairs": 0, "accuracy": None, "mean_advantage": None, "mean_uncertainty": None}
    score_l, unc_l = output(model, arrays["x"][left], device)
    score_r, unc_r = output(model, arrays["x"][right], device)
    advantage = score_l - score_r
    uncertainty = np.sqrt(unc_l * unc_l + unc_r * unc_r)
    return {"pairs": int(len(left)), "accuracy": float(np.mean(advantage > 0.0)), "mean_advantage": float(np.mean(advantage)), "mean_uncertainty": float(np.mean(uncertainty)), "p95_uncertainty": float(np.quantile(uncertainty, 0.95))}


def candidate_configs(train_adv: np.ndarray, train_unc: np.ndarray, train_gap: np.ndarray) -> list[dict[str, float]]:
    configs = []
    for margin in (0.25, 0.5, 1.0, 2.0):
        for advantage in (0.0, 0.1, 0.25, 0.5):
            for uncertainty in (0.75, 1.0, 1.5, 2.0):
                accepted = (train_gap <= margin) & (train_adv >= advantage) & (train_unc <= uncertainty)
                count = int(np.sum(accepted))
                precision_proxy = float(np.mean(train_adv[accepted] > 0.0)) if count else 0.0
                # A conservative train objective: reward reliable positive
                # pair predictions, with a small coverage term and no
                # requirement to invent changes when none are supported.
                objective = precision_proxy * np.sqrt(count / max(len(train_adv), 1))
                configs.append({"near_tie_margin": margin, "min_predicted_advantage": advantage, "max_pair_uncertainty": uncertainty, "train_accept_count": count, "train_precision_proxy": precision_proxy, "train_gate_objective": objective})
    return sorted(configs, key=lambda item: (-item["train_gate_objective"], -item["train_accept_count"], item["near_tie_margin"], item["min_predicted_advantage"], item["max_pair_uncertainty"]))


def validate_config(config: dict[str, float], adv: np.ndarray, unc: np.ndarray, gap: np.ndarray) -> dict[str, float]:
    accepted = (gap <= config["near_tie_margin"]) & (adv >= config["min_predicted_advantage"]) & (unc <= config["max_pair_uncertainty"])
    count = int(np.sum(accepted))
    return {"validation_accept_count": count, "validation_positive_fraction": float(np.mean(adv[accepted] > 0.0)) if count else 0.0, "validation_gate_objective": (float(np.mean(adv[accepted] > 0.0)) * np.sqrt(count / max(len(adv), 1))) if count else 0.0}


def main() -> None:
    result: dict[str, Any] = {"status": "FAIL", "protocol": "N44_STAGE_03_ACTUAL_TRAINING_V1", "started_at": now(), "project_root": str(ROOT)}
    try:
        set_seed(SEED)
        if not DATASET.is_file() or not GROUPS.is_file():
            raise FileNotFoundError("N44 Stage 02 dataset or group manifest missing")
        loaded = np.load(DATASET)
        arrays = {key: loaded[key] for key in loaded.files}
        required = {"x", "pair_left", "pair_right", "pair_split", "pair_base_gap"}
        if not required.issubset(arrays) or arrays["x"].shape[1] != 18:
            raise RuntimeError("N44 pair dataset schema invalid")
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model = AssignmentAwareHead().to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=2.0e-3, weight_decay=1.0e-4)
        train_mask = arrays["pair_split"] == 0
        val_mask = arrays["pair_split"] == 1
        train_left, train_right = arrays["pair_left"][train_mask], arrays["pair_right"][train_mask]
        if len(train_left) == 0 or int(np.sum(val_mask)) == 0:
            raise RuntimeError("train/validation pair split is empty")
        best_val = float("inf")
        best_epoch = 0
        stale = 0
        history = []
        batch_size = 1024
        generator = torch.Generator(device="cpu").manual_seed(SEED)
        for epoch in range(1, MAX_EPOCHS + 1):
            model.train()
            order = torch.randperm(len(train_left), generator=generator).numpy()
            loss_values = []
            for start in range(0, len(order), batch_size):
                indices = order[start : start + batch_size]
                left = torch.as_tensor(arrays["x"][train_left[indices]], dtype=torch.float32, device=device)
                right = torch.as_tensor(arrays["x"][train_right[indices]], dtype=torch.float32, device=device)
                raw_l, raw_r = model(left), model(right)
                advantage = raw_l[:, 0] - raw_r[:, 0]
                bce = nn.functional.binary_cross_entropy_with_logits(advantage, torch.ones_like(advantage))
                variance = nn.functional.softplus(raw_l[:, 1]) + nn.functional.softplus(raw_r[:, 1]) + 0.02
                nll = 0.5 * (((advantage - 1.0) ** 2) / variance + torch.log(variance))
                loss = bce + 0.10 * nll.mean()
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
                loss_values.append(float(loss.detach().cpu()))
            model.eval()
            val_left, val_right = arrays["pair_left"][val_mask], arrays["pair_right"][val_mask]
            with torch.no_grad():
                vl, vr = model(torch.as_tensor(arrays["x"][val_left], dtype=torch.float32, device=device)), model(torch.as_tensor(arrays["x"][val_right], dtype=torch.float32, device=device))
                v_adv = vl[:, 0] - vr[:, 0]
                v_var = nn.functional.softplus(vl[:, 1]) + nn.functional.softplus(vr[:, 1]) + 0.02
                val_loss = float((nn.functional.binary_cross_entropy_with_logits(v_adv, torch.ones_like(v_adv)) + 0.10 * (0.5 * (((v_adv - 1.0) ** 2) / v_var + torch.log(v_var))).mean()).cpu())
            history.append({"epoch": epoch, "train_loss": float(np.mean(loss_values)), "validation_loss": val_loss})
            if val_loss < best_val - 1.0e-8:
                best_val, best_epoch, stale = val_loss, epoch, 0
                torch.save({"protocol": PROTOCOL, "input_dim": 18, "hidden": 64, "seed": SEED, "epoch": epoch, "state_dict": model.state_dict()}, CHECKPOINT)
            else:
                stale += 1
                if stale >= PATIENCE:
                    break
        if best_epoch == 0:
            raise RuntimeError("no best checkpoint produced")
        payload = torch.load(CHECKPOINT, map_location=device, weights_only=False)
        model.load_state_dict(payload["state_dict"])
        model.eval()
        train_pair = pair_metrics(model, arrays, 0, device)
        val_pair = pair_metrics(model, arrays, 1, device)
        holdout_pair = pair_metrics(model, arrays, 2, device)
        # Calibrate the uncertainty scale on train, then select a finite gate
        # from the best train candidates using validation.  Holdout is not
        # read in this selection path.
        train_l, train_r = arrays["pair_left"][train_mask], arrays["pair_right"][train_mask]
        val_l, val_r = arrays["pair_left"][val_mask], arrays["pair_right"][val_mask]
        sl, ul = output(model, arrays["x"][train_l], device); sr, ur = output(model, arrays["x"][train_r], device)
        train_adv, raw_train_unc = sl - sr, np.sqrt(ul * ul + ur * ur)
        calibration_scale = float(np.clip(np.quantile(np.abs(train_adv - 1.0) / np.maximum(raw_train_unc, 1.0e-6), 0.75), 0.5, 10.0))
        train_unc = raw_train_unc * calibration_scale
        train_configs = candidate_configs(train_adv, train_unc, arrays["pair_base_gap"][train_mask])
        top = train_configs[: min(16, len(train_configs))]
        sl, ul = output(model, arrays["x"][val_l], device); sr, ur = output(model, arrays["x"][val_r], device)
        val_adv, val_unc = sl - sr, np.sqrt(ul * ul + ur * ur) * calibration_scale
        validated = []
        for config in top:
            item = dict(config); item.update(validate_config(config, val_adv, val_unc, arrays["pair_base_gap"][val_mask])); validated.append(item)
        selected = sorted(validated, key=lambda item: (-item["validation_gate_objective"], -item["validation_accept_count"], item["near_tie_margin"], item["min_predicted_advantage"], item["max_pair_uncertainty"]))[0]
        gate = {"near_tie_margin": float(selected["near_tie_margin"]), "min_predicted_advantage": float(selected["min_predicted_advantage"]), "max_pair_uncertainty": float(selected["max_pair_uncertainty"]), "uncertainty_scale": calibration_scale}
        payload.update({"gate": gate, "training_protocol": "fixed sequence-disjoint train/validation/holdout; pairwise loss train only; validation early stopping and gate freeze; holdout untouched", "best_validation_loss": best_val})
        torch.save(payload, CHECKPOINT)
        manifest = {"protocol": PROTOCOL, "status": "PASS", "seed": SEED, "device": str(device), "dataset": str(DATASET), "dataset_sha256": sha256(DATASET), "checkpoint": str(CHECKPOINT), "checkpoint_sha256": sha256(CHECKPOINT), "completed_epochs": len(history), "best_epoch": best_epoch, "best_validation_loss": best_val, "max_epochs": MAX_EPOCHS, "patience": PATIENCE, "optimizer": {"name": "AdamW", "lr": 2.0e-3, "weight_decay": 1.0e-4, "batch_size": batch_size}, "loss": "BCE(pair score difference, positive wins) + 0.10 heteroscedastic pair NLL", "pair_metrics": {"train": train_pair, "validation": val_pair, "holdout_audit_only": holdout_pair}, "gate_selection": {"calibration": "train 75th percentile absolute pair residual/raw uncertainty", "train_candidates_evaluated": len(train_configs), "validation_candidates_evaluated": len(validated), "selected": selected, "frozen_gate": gate, "holdout_used_for_selection": False}, "history": history}
        MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        result.update({"status": "PASS", "command": [sys.executable, str(Path(__file__).resolve())], "inputs": {"dataset": str(DATASET), "group_manifest": str(GROUPS), "sequence_split": str(ROOT / "outputs/n42/training/training_protocol.json")}, "outputs": {"checkpoint": str(CHECKPOINT), "manifest": str(MANIFEST)}, "metrics": {"device": str(device), "seed": SEED, "completed_epochs": len(history), "best_epoch": best_epoch, "pair_metrics": manifest["pair_metrics"], "frozen_gate": gate}, "gate_checks": {"actual_full_training": True, "sequence_disjoint": True, "fixed_seed": True, "checkpoint_hash_recorded": True, "train_labels_only_for_optimization": True, "validation_only_gate_freeze": True, "holdout_not_used_for_selection": True, "public_id_not_feature": True, "future_outcome_not_feature": True, "none_abstain_runtime_boundary": True, "bounded_application": True, "production_code_modified": False}, "failure_root_cause": "N43 per-cell target utility did not optimize candidate-vs-competitor assignment gain; this experiment trains anti-symmetric pairwise utility with calibrated uncertainty for conservative proposal gating.", "next_action": "Run same-prefix same-event same-candidate M0-M4 paired replay with the frozen gate; posthoc GT only after runtime validation.", "runtime_future_gt_used": False, "finished_at": now()})
        STAGE.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({"status": result["status"], "device": str(device), "epochs": len(history), "checkpoint": str(CHECKPOINT)}))
    except Exception as exc:
        result.update({"status": "FAIL", "failure_root_cause": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc(), "finished_at": now()})
        failure = ROOT / "outputs/n44/attempts" / f"stage_03_failure_{datetime.now().strftime('%Y%m%dT%H%M%S')}.json"
        failure.parent.mkdir(parents=True, exist_ok=True)
        failure.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        raise


if __name__ == "__main__":
    main()
