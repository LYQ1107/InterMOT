#!/usr/bin/env python3
"""N32-G fallback: one causal five-step GRU selector.

This file is invoked only after the static selector Learn Gate fails.  It uses
the same three policy labels, objective, split, seeds, and margin rule; the
only changed capacity is a GRU over the stored last-at-most-five causal
feature snapshots.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import sys
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
INDEX = ROOT / "outputs/n32/policy_rollout_index.json"
OUT_DIR = ROOT / "outputs/n32"
CHECKPOINT_DIR = OUT_DIR / "checkpoints"
POLICIES = ("K0_KEEP_OLD", "K1_APPLY_ENSURE", "K2_PROMPT_THEN_RESTORE")
SEEDS = (3201, 3202, 3203)
THRESHOLDS = (0.0, 0.01, 0.02, 0.05, 0.10)

from sam3_intermot.adaptation.correction_application_selector import CorrectionApplicationSelector  # noqa: E402
from sam3_intermot.adaptation.correction_selector_features import FEATURE_NAMES  # noqa: E402
from scripts.n32_evaluate_selector import _best_fixed, _gate, _load_groups, _summarize  # noqa: E402


class TemporalCorrectionApplicationSelector(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 64) -> None:
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dim + input_dim),
            nn.Linear(hidden_dim + input_dim, 128),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Linear(64, 3),
        )

    def forward(self, sequence: torch.Tensor, current: torch.Tensor) -> torch.Tensor:
        _, hidden = self.gru(sequence)
        return self.head(torch.cat([hidden[-1], current], dim=-1))

    def loss(self, logits: torch.Tensor, rewards: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        values = []
        for index in range(logits.shape[0]):
            listwise = CorrectionApplicationSelector.listwise_kl(logits[index:index + 1], rewards[index:index + 1], 0.10)
            pairwise = CorrectionApplicationSelector.pairwise_margin(logits[index:index + 1], rewards[index:index + 1], 0.01, 0.05)
            values.append((listwise + 0.10 * pairwise) * weights[index])
        return torch.stack(values).mean()


def _default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer, np.bool_)):
        return value.item()
    return value


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True, default=_default) + "\n", encoding="utf-8")
    tmp.replace(path)


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _tape(group: Mapping[str, Any]) -> np.ndarray:
    values = group["rows"][POLICIES[0]].get("temporal_feature_sequence", [])
    array = np.asarray(values, dtype=np.float32)
    if array.ndim != 2 or array.shape[1] != len(FEATURE_NAMES) or not (1 <= array.shape[0] <= 5) or not np.isfinite(array).all():
        raise RuntimeError(f"missing/invalid causal temporal tape for {group['episode_id']}")
    return array


def _arrays(groups: list[dict[str, Any]], mean: np.ndarray, std: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    sequences = []
    rewards = []
    weights = []
    for group in groups:
        tape = (_tape(group) - mean[None, :]) / std[None, :]
        if tape.shape[0] < 5:
            tape = np.concatenate([np.repeat(tape[:1], 5 - tape.shape[0], axis=0), tape], axis=0)
        sequences.append(tape[-5:])
        reward = np.asarray([float(group["rows"][policy]["reward"]) for policy in POLICIES], dtype=np.float32)
        rewards.append(reward)
        weights.append(1.0 + min(max(float(reward.max() - reward[1]), 0.0), 0.50))
    return np.asarray(sequences, dtype=np.float32), np.asarray(rewards, dtype=np.float32), np.asarray(weights, dtype=np.float32)


def _train(sequences: np.ndarray, rewards: np.ndarray, weights: np.ndarray, seed: int, epochs: int) -> tuple[TemporalCorrectionApplicationSelector, list[dict[str, float]]]:
    _seed(seed)
    model = TemporalCorrectionApplicationSelector(len(FEATURE_NAMES))
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.0e-3, weight_decay=1.0e-4)
    x = torch.from_numpy(sequences)
    current = x[:, -1]
    y = torch.from_numpy(rewards)
    w = torch.from_numpy(weights)
    generator = torch.Generator().manual_seed(seed + 191)
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        order = torch.randperm(len(x), generator=generator)
        for start in range(0, len(x), 64):
            idx = order[start:start + 64]
            optimizer.zero_grad(set_to_none=True)
            loss = model.loss(model(x[idx], current[idx]), y[idx], w[idx])
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        model.eval()
        with torch.no_grad():
            logits = model(x, current)
            chosen = logits.argmax(dim=-1)
            best = y.max(dim=-1).values
            selected = y.gather(1, chosen[:, None]).squeeze(1)
            history.append({"epoch": epoch, "loss": float(model.loss(logits, y, w).item()), "action_accuracy": float((selected >= best - 1.0e-6).float().mean().item()), "reward_ratio_to_oracle": float((selected.mean() / best.abs().mean()).item())})
    return model, history


def _actions(model: TemporalCorrectionApplicationSelector, groups: list[dict[str, Any]], mean: np.ndarray, std: np.ndarray, threshold: float | None, base_policy: str) -> tuple[dict[str, str], dict[str, float]]:
    sequences, _, _ = _arrays(groups, mean, std)
    x = torch.from_numpy(sequences)
    with torch.no_grad():
        logits = model(x, x[:, -1]).numpy()
    actions = {}
    margins = {}
    for group, row in zip(groups, logits):
        order = np.argsort(-row, kind="stable")
        margin = float(row[int(order[0])] - row[int(order[1])])
        policy = POLICIES[int(order[0])]
        if threshold is not None and margin <= threshold:
            policy = base_policy
        actions[group["episode_id"]] = policy
        margins[group["episode_id"]] = margin
    return actions, margins


def _select_threshold(model: TemporalCorrectionApplicationSelector, groups: list[dict[str, Any]], mean: np.ndarray, std: np.ndarray, base_policy: str) -> tuple[float, list[dict[str, Any]]]:
    records = []
    for threshold in THRESHOLDS:
        actions, _ = _actions(model, groups, mean, std, threshold, base_policy)
        summary = _summarize(groups, actions, method=f"TEMPORAL_MARGIN_{threshold:g}", base_policy=base_policy)
        records.append({"threshold": threshold, "mean_reward": summary["mean_reward"], "h20_iou": summary["h20_iou"], "summary": summary})
    selected = max(records, key=lambda item: ((item["mean_reward"] if item["mean_reward"] is not None else -1.0e9), (item["h20_iou"] if item["h20_iou"] is not None else -1.0e9), item["threshold"]))
    return float(selected["threshold"]), records


def _overfit(groups: list[dict[str, Any]], mean: np.ndarray, std: np.ndarray, seed: int) -> dict[str, Any]:
    sequences, rewards, weights = _arrays(groups[:20], mean, std)
    _seed(seed)
    initial = TemporalCorrectionApplicationSelector(len(FEATURE_NAMES))
    initial.eval()
    with torch.no_grad():
        initial_loss = float(initial.loss(initial(torch.from_numpy(sequences), torch.from_numpy(sequences)[:, -1]), torch.from_numpy(rewards), torch.from_numpy(weights)).item())
    model, history = _train(sequences, rewards, weights, seed, 500)
    return {"seed": seed, "episode_count": 20, "initial_loss": initial_loss, "final_loss": history[-1]["loss"], "final_action_accuracy": history[-1]["action_accuracy"], "final_reward_ratio_to_oracle": history[-1]["reward_ratio_to_oracle"], "pass": bool(history[-1]["loss"] < initial_loss and history[-1]["action_accuracy"] >= 0.90 and history[-1]["reward_ratio_to_oracle"] >= 0.95)}


def run(*, index_path: Path = INDEX, output_dir: Path = OUT_DIR) -> dict[str, Any]:
    groups = _load_groups(index_path)
    try:
        for group in groups.values():
            _tape(group)
    except RuntimeError as exc:
        result = {"protocol": "N32-G-TEMPORAL-FALLBACK", "status": "NOT_RUN_INPUT_TAPE_UNAVAILABLE", "reason": str(exc), "val25_read": False, "test_labels_used": False}
        _write(output_dir / "temporal_learn_gate.json", result)
        return result
    train = _ordered(groups, "train")
    selection = _ordered(groups, "selection")
    calibration = _ordered(groups, "calibration")
    raw_x = np.concatenate([_tape(group) for group in train], axis=0)
    mean = raw_x.mean(axis=0).astype(np.float32)
    std = np.where(raw_x.std(axis=0) < 1.0e-6, 1.0, raw_x.std(axis=0)).astype(np.float32)
    overfit = [_overfit(train, mean, std, seed) for seed in SEEDS]
    base_policy = _best_fixed(train)
    seed_payload = {}
    selection_seed_summary = []
    for seed in SEEDS:
        seq, rew, weights = _arrays(train, mean, std)
        model, history = _train(seq, rew, weights, seed, 100)
        path = CHECKPOINT_DIR / f"temporal_selector_seed{seed}.pt"
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"state_dict": model.state_dict(), "input_dim": len(FEATURE_NAMES), "feature_names": list(FEATURE_NAMES), "mean": mean, "std": std, "seed": seed, "protocol": "N32-G-TEMPORAL"}, path)
        threshold, threshold_records = _select_threshold(model, selection, mean, std, base_policy)
        raw_actions, raw_margins = _actions(model, selection, mean, std, None, base_policy)
        margin_actions, _ = _actions(model, selection, mean, std, threshold, base_policy)
        cal_raw, _ = _actions(model, calibration, mean, std, None, base_policy)
        cal_margin, _ = _actions(model, calibration, mean, std, threshold, base_policy)
        seed_payload[str(seed)] = {"threshold": threshold, "threshold_records": threshold_records, "selection_raw_actions": raw_actions, "selection_margin_actions": margin_actions, "calibration_raw_actions": cal_raw, "calibration_margin_actions": cal_margin, "selection_raw": _summarize(selection, raw_actions, method="TEMPORAL_RAW", base_policy=base_policy), "selection_margin": _summarize(selection, margin_actions, method="TEMPORAL_MARGIN", base_policy=base_policy), "calibration_raw": _summarize(calibration, cal_raw, method="TEMPORAL_RAW", base_policy=base_policy), "calibration_margin": _summarize(calibration, cal_margin, method="TEMPORAL_MARGIN", base_policy=base_policy), "checkpoint": str(path), "checkpoint_sha256": _sha(path), "formal_final_loss": history[-1]["loss"]}
        selection_seed_summary.append({"seed": seed, "margin": seed_payload[str(seed)]["selection_margin"], "threshold": threshold})
    selected = max(selection_seed_summary, key=lambda item: ((item["margin"]["mean_reward"] if item["margin"]["mean_reward"] is not None else -1.0e9), -item["seed"]))
    selected_seed = int(selected["seed"])
    selected_payload = seed_payload[str(selected_seed)]
    selection_methods = {policy: {group["episode_id"]: policy for group in selection} for policy in POLICIES}
    calibration_methods = {policy: {group["episode_id"]: policy for group in calibration} for policy in POLICIES}
    selection_methods["BEST_FIXED"] = {group["episode_id"]: base_policy for group in selection}
    calibration_methods["BEST_FIXED"] = {group["episode_id"]: base_policy for group in calibration}
    selection_methods["TEMPORAL_RAW"] = selected_payload["selection_raw_actions"]
    selection_methods["TEMPORAL_MARGIN"] = selected_payload["selection_margin_actions"]
    calibration_methods["TEMPORAL_RAW"] = selected_payload["calibration_raw_actions"]
    calibration_methods["TEMPORAL_MARGIN"] = selected_payload["calibration_margin_actions"]
    selection_summaries = {name: _summarize(selection, actions, method=name, base_policy=base_policy) for name, actions in selection_methods.items()}
    calibration_summaries = {name: _summarize(calibration, actions, method=name, base_policy=base_policy) for name, actions in calibration_methods.items()}
    gates = {str(seed): {"selection": _gate(seed_payload[str(seed)]["selection_margin"]), "calibration": _gate(seed_payload[str(seed)]["calibration_margin"])} for seed in SEEDS}
    checks = {"overfit_gate_pass": all(item["pass"] for item in overfit), "at_least_two_of_three_seeds_pass": sum(gates[str(seed)]["selection"]["pass"] and gates[str(seed)]["calibration"]["pass"] for seed in SEEDS) >= 2, "selected_calibration_pass": _gate(selected_payload["calibration_margin"])["pass"]}
    result = {"protocol": "N32-G-TEMPORAL-FALLBACK", "status": "PASS" if all(checks.values()) else "FAIL", "route": "temporal_selector" if all(checks.values()) else "association_fallback", "base_policy": base_policy, "selected_seed": selected_seed, "selected_threshold": selected_payload["threshold"], "overfit": overfit, "seed_gates": gates, "checks": checks, "selection_methods": selection_summaries, "calibration_methods": calibration_summaries, "future_gt_used_for_selector_input": False, "future_gt_used_for_training_labels": True, "val25_read": False, "test_labels_used": False}
    _write(output_dir / "temporal_selector_training.json", {"protocol": "N32-G-TEMPORAL-TRAIN", "status": "PASS" if all(item["pass"] for item in overfit) else "FAIL", "feature_dimension": len(FEATURE_NAMES), "max_sequence_length": 5, "hidden_dimension": 64, "seeds": list(SEEDS), "overfit": overfit, "normalization_fit_split": "train", "future_gt_used_for_selector_input": False, "future_gt_used_for_training_labels": True, "val25_read": False, "test_labels_used": False})
    _write(output_dir / "temporal_overfit_gate.json", {"protocol": "N32-G-TEMPORAL-OVERFIT-GATE", "status": "PASS" if all(item["pass"] for item in overfit) else "FAIL", "results": overfit, "val25_read": False, "test_labels_used": False})
    _write(output_dir / "temporal_selection_results.json", {"protocol": "N32-G-TEMPORAL-SELECTION", "status": "PASS", "selected_seed": selected_seed, "selected_threshold": selected_payload["threshold"], "methods": selection_summaries, "seed_summaries": selection_seed_summary, "future_gt_used_for_selection": False, "val25_read": False, "test_labels_used": False})
    _write(output_dir / "temporal_calibration_results.json", {"protocol": "N32-G-TEMPORAL-CALIBRATION", "status": "PASS", "selected_seed": selected_seed, "selected_threshold": selected_payload["threshold"], "methods": calibration_summaries, "future_gt_used_for_calibration_metrics": True, "val25_read": False, "test_labels_used": False})
    _write(output_dir / "temporal_learn_gate.json", result)
    return result


def _ordered(groups: Mapping[str, dict[str, Any]], split: str) -> list[dict[str, Any]]:
    return sorted([group for group in groups.values() if group["learning_split"] == split], key=lambda group: (group["sequence"], group["episode_id"]))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", type=Path, default=INDEX)
    parser.add_argument("--output-dir", type=Path, default=OUT_DIR)
    args = parser.parse_args()
    result = run(index_path=args.index, output_dir=args.output_dir)
    print(json.dumps({"protocol": result.get("protocol"), "status": result.get("status"), "route": result.get("route")}, indent=2, default=_default))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
