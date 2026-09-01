#!/usr/bin/env python3
"""N32-E/F: train and mechanism-test the strategy-level selector.

The script consumes only the frozen train split and the post-hoc policy
rewards attached to the three real-policy rollouts.  Episode and sequence
identifiers are used for grouping/splitting and are never passed to the model.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
INDEX = ROOT / "outputs/n32/policy_rollout_index.json"
AUDIT = ROOT / "outputs/n32/selector_feature_audit.json"
ORACLE = ROOT / "outputs/n32/policy_oracle_689.json"
FROZEN = ROOT / "outputs/n32/frozen_protocol.json"
OUT_DIR = ROOT / "outputs/n32"
CHECKPOINT_DIR = OUT_DIR / "checkpoints"
POLICIES = ("K0_KEEP_OLD", "K1_APPLY_ENSURE", "K2_PROMPT_THEN_RESTORE")
SEEDS = (3201, 3202, 3203)

from sam3_intermot.adaptation.correction_application_selector import (  # noqa: E402
    CorrectionApplicationSelector,
    parameter_count,
)
from sam3_intermot.adaptation.correction_selector_features import FEATURE_NAMES  # noqa: E402


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True, default=_json_default) + "\n", encoding="utf-8")
    tmp.replace(path)


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    raise TypeError(type(value).__name__)


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _seed(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))


def _load_groups(index_path: Path) -> dict[str, dict[str, Any]]:
    source = json.loads(index_path.read_text(encoding="utf-8"))
    if source.get("status") != "PASS":
        raise RuntimeError("N32-C index is not PASS")
    groups: dict[str, dict[str, Any]] = {}
    for row in source.get("rows", []):
        episode_id = str(row["episode_id"])
        group = groups.setdefault(episode_id, {
            "episode_id": episode_id,
            "sequence": str(row["sequence"]),
            "learning_split": str(row["learning_split"]),
            "rows": {},
        })
        policy = str(row["policy"])
        if policy in group["rows"]:
            raise RuntimeError(f"duplicate policy row: {episode_id} {policy}")
        group["rows"][policy] = row
    if len(groups) != 689 or any(set(group["rows"]) != set(POLICIES) for group in groups.values()):
        raise RuntimeError("selector training requires 689 complete three-policy groups")
    for group in groups.values():
        vectors = [np.asarray(group["rows"][policy]["feature_vector"], dtype=np.float32) for policy in POLICIES]
        if any(vector.shape != (len(FEATURE_NAMES),) for vector in vectors):
            raise RuntimeError(f"bad feature dimension in {group['episode_id']}")
        if not all(np.array_equal(vectors[0], vector) for vector in vectors[1:]):
            raise RuntimeError(f"policy-dependent feature vector in {group['episode_id']}")
        rewards = [group["rows"][policy].get("reward") for policy in POLICIES]
        if any(value is None or not np.isfinite(float(value)) for value in rewards):
            raise RuntimeError(f"missing reward in {group['episode_id']}")
    return groups


def _ordered_groups(groups: Mapping[str, dict[str, Any]], split: str | None = None) -> list[dict[str, Any]]:
    values = [group for group in groups.values() if split is None or group["learning_split"] == split]
    return sorted(values, key=lambda group: (group["sequence"], group["episode_id"]))


def _arrays(groups: Iterable[dict[str, Any]], mean: np.ndarray, std: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    selected = list(groups)
    x = np.asarray([selected_group["rows"][POLICIES[0]]["feature_vector"] for selected_group in selected], dtype=np.float32)
    rewards = np.asarray([[float(selected_group["rows"][policy]["reward"]) for policy in POLICIES] for selected_group in selected], dtype=np.float32)
    weights = 1.0 + np.minimum(np.maximum(rewards.max(axis=1) - rewards[:, 1], 0.0), 0.50)
    x = (x - mean[None, :]) / std[None, :]
    return x, rewards, weights.astype(np.float32)


def _action_accuracy(logits: torch.Tensor, rewards: torch.Tensor) -> float:
    chosen = torch.argmax(logits, dim=-1)
    best = rewards.max(dim=-1).values
    selected = rewards.gather(1, chosen[:, None]).squeeze(1)
    return float((selected >= best - 1.0e-6).float().mean().item())


def _reward_ratio(logits: torch.Tensor, rewards: torch.Tensor) -> float:
    chosen = torch.argmax(logits, dim=-1)
    selected = rewards.gather(1, chosen[:, None]).squeeze(1)
    oracle = rewards.max(dim=-1).values
    denominator = float(oracle.abs().mean().item())
    if denominator < 1.0e-8:
        return 1.0
    return float((selected.mean() / oracle.abs().mean()).item())


def _train_model(
    x: np.ndarray,
    rewards: np.ndarray,
    weights: np.ndarray,
    *,
    seed: int,
    epochs: int,
    batch_size: int,
    record_path: Path | None = None,
) -> tuple[CorrectionApplicationSelector, list[dict[str, Any]]]:
    _seed(seed)
    model = CorrectionApplicationSelector(x.shape[1])
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.0e-3, weight_decay=1.0e-4)
    x_tensor = torch.from_numpy(x)
    reward_tensor = torch.from_numpy(rewards)
    weight_tensor = torch.from_numpy(weights)
    generator = torch.Generator().manual_seed(int(seed) + 91)
    history: list[dict[str, Any]] = []
    for epoch in range(1, int(epochs) + 1):
        model.train()
        order = torch.randperm(x_tensor.shape[0], generator=generator)
        batch_losses: list[float] = []
        for start in range(0, x_tensor.shape[0], int(batch_size)):
            indices = order[start:start + int(batch_size)]
            optimizer.zero_grad(set_to_none=True)
            logits = model(x_tensor[indices])
            loss = model.loss(
                logits,
                reward_tensor[indices],
                temperature=0.10,
                epsilon=0.01,
                margin=0.05,
                pairwise_weight=0.10,
                weights=weight_tensor[indices],
            )
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            batch_losses.append(float(loss.detach().item()))
        model.eval()
        with torch.no_grad():
            logits = model(x_tensor)
            eval_loss = model.loss(logits, reward_tensor, temperature=0.10, epsilon=0.01, margin=0.05, pairwise_weight=0.10, weights=weight_tensor)
            history.append({
                "epoch": epoch,
                "train_loss": float(np.mean(batch_losses)),
                "eval_loss": float(eval_loss.item()),
                "action_accuracy": _action_accuracy(logits, reward_tensor),
                "reward_ratio_to_oracle": _reward_ratio(logits, reward_tensor),
            })
    if record_path is not None:
        record_path.parent.mkdir(parents=True, exist_ok=True)
        with record_path.open("w", encoding="utf-8") as handle:
            for item in history:
                handle.write(json.dumps(item, sort_keys=True) + "\n")
    return model, history


def _overfit(seed: int, groups: list[dict[str, Any]], mean: np.ndarray, std: np.ndarray, epochs: int = 500) -> dict[str, Any]:
    x, rewards, weights = _arrays(groups[:20], mean, std)
    _seed(seed)
    initial = CorrectionApplicationSelector(x.shape[1])
    initial.eval()
    with torch.no_grad():
        initial_loss = float(initial.loss(torch.from_numpy(x), torch.from_numpy(rewards), temperature=0.10, epsilon=0.01, margin=0.05, pairwise_weight=0.10, weights=torch.from_numpy(weights)).item())
    model, history = _train_model(x, rewards, weights, seed=seed, epochs=epochs, batch_size=20)
    final = history[-1]
    checkpoint = {
        "state_dict": model.state_dict(),
        "input_dim": int(x.shape[1]),
        "feature_names": list(FEATURE_NAMES),
        "normalization_mean": torch.from_numpy(mean),
        "normalization_std": torch.from_numpy(std),
        "protocol": "N32-E-OVERFIT",
        "seed": int(seed),
    }
    tmp = CHECKPOINT_DIR / f"selector_overfit_seed{seed}.pt"
    tmp.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, tmp)
    loaded = CorrectionApplicationSelector(x.shape[1])
    loaded.load_state_dict(torch.load(tmp, map_location="cpu", weights_only=False)["state_dict"])
    loaded.eval()
    model.eval()
    with torch.no_grad():
        save_load_delta = float((model(torch.from_numpy(x)) - loaded(torch.from_numpy(x))).abs().max().item())
        shuffled = torch.from_numpy(x.copy())
        shuffled = shuffled[torch.randperm(shuffled.shape[0], generator=torch.Generator().manual_seed(seed + 7000))]
        shuffled_accuracy = _action_accuracy(loaded(shuffled), torch.from_numpy(rewards))
    return {
        "seed": int(seed),
        "episode_count": 20,
        "epochs": int(epochs),
        "initial_loss": initial_loss,
        "final_loss": float(final["eval_loss"]),
        "final_action_accuracy": float(final["action_accuracy"]),
        "final_reward_ratio_to_oracle": float(final["reward_ratio_to_oracle"]),
        "save_load_max_abs_logit_delta": save_load_delta,
        "shuffled_input_action_accuracy": float(shuffled_accuracy),
        "parameter_count": parameter_count(model),
        "checkpoint": str(tmp),
        "pass": bool(
            final["eval_loss"] < initial_loss
            and final["action_accuracy"] >= 0.90
            and final["reward_ratio_to_oracle"] >= 0.95
            and save_load_delta <= 1.0e-6
            and shuffled_accuracy < final["action_accuracy"]
        ),
    }


def run(*, index_path: Path = INDEX, output_dir: Path = OUT_DIR, seeds: tuple[int, ...] = SEEDS) -> dict[str, Any]:
    audit = json.loads(AUDIT.read_text(encoding="utf-8"))
    oracle = json.loads(ORACLE.read_text(encoding="utf-8"))
    frozen = json.loads(FROZEN.read_text(encoding="utf-8"))
    if audit.get("status") != "PASS" or oracle.get("status") not in {"PASS", "FAIL"}:
        raise RuntimeError("N32 selector training requires completed feature audit and policy Oracle")
    groups = _load_groups(index_path)
    train_groups = _ordered_groups(groups, "train")
    if len(train_groups) != 419:
        raise RuntimeError(f"expected 419 train episodes, got {len(train_groups)}")
    raw_x = np.asarray([group["rows"][POLICIES[0]]["feature_vector"] for group in train_groups], dtype=np.float32)
    mean = raw_x.mean(axis=0)
    std = raw_x.std(axis=0)
    std = np.where(std < 1.0e-6, 1.0, std).astype(np.float32)
    overfit_results = [_overfit(int(seed), train_groups, mean, std) for seed in seeds]
    formal_results: list[dict[str, Any]] = []
    x, rewards, weights = _arrays(train_groups, mean, std)
    for seed in seeds:
        model, history = _train_model(
            x,
            rewards,
            weights,
            seed=int(seed),
            epochs=100,
            batch_size=64,
            record_path=output_dir / f"train_metrics_seed{int(seed)}.jsonl",
        )
        checkpoint_path = CHECKPOINT_DIR / f"selector_seed{int(seed)}.pt"
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "state_dict": model.state_dict(),
            "input_dim": int(x.shape[1]),
            "feature_names": list(FEATURE_NAMES),
            "normalization_mean": torch.from_numpy(mean),
            "normalization_std": torch.from_numpy(std),
            "protocol": "N32-F-SELECTOR",
            "seed": int(seed),
            "optimizer": {"name": "AdamW", "learning_rate": 1.0e-3, "weight_decay": 1.0e-4},
            "epochs": 100,
            "batch_size": 64,
            "gradient_clip": 1.0,
            "loss": frozen["selector"]["loss"],
            "best_fixed_policy": oracle.get("best_fixed_policy", "K1_APPLY_ENSURE"),
        }, checkpoint_path)
        formal_results.append({
            "seed": int(seed),
            "train_episode_count": len(train_groups),
            "epochs": 100,
            "parameter_count": parameter_count(model),
            "initial_loss": float(history[0]["eval_loss"]),
            "final_loss": float(history[-1]["eval_loss"]),
            "final_action_accuracy": float(history[-1]["action_accuracy"]),
            "final_reward_ratio_to_oracle": float(history[-1]["reward_ratio_to_oracle"]),
            "checkpoint": str(checkpoint_path),
        })
    result = {
        "protocol": "N32-EF-SELECTOR-TRAIN",
        "status": "PASS" if all(item["pass"] for item in overfit_results) else "FAIL",
        "source_index": str(index_path),
        "source_index_sha256": _sha(index_path),
        "feature_names": list(FEATURE_NAMES),
        "feature_dimension": len(FEATURE_NAMES),
        "train_episode_count": len(train_groups),
        "normalization": {"mean": mean.tolist(), "std": std.tolist(), "fit_split": "train"},
        "overfit": overfit_results,
        "overfit_gate": {
            "episode_count": 20,
            "required_seed_count": len(seeds),
            "all_seeds_pass": all(item["pass"] for item in overfit_results),
            "thresholds": {"action_accuracy": 0.90, "reward_ratio_to_oracle": 0.95, "save_load_max_abs": 1.0e-6},
        },
        "formal_training": formal_results,
        "seeds": [int(seed) for seed in seeds],
        "future_gt_used_for_selector_input": False,
        "future_gt_used_for_training_labels": True,
        "val25_read": False,
        "test_labels_used": False,
    }
    _write(output_dir / "selector_training.json", result)
    _write(output_dir / "overfit_gate.json", {"protocol": "N32-E-OVERFIT-GATE", **result["overfit_gate"], "results": overfit_results, "status": "PASS" if result["status"] == "PASS" else "FAIL", "val25_read": False, "test_labels_used": False})
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", type=Path, default=INDEX)
    parser.add_argument("--output-dir", type=Path, default=OUT_DIR)
    args = parser.parse_args()
    result = run(index_path=args.index, output_dir=args.output_dir)
    print(json.dumps({key: result[key] for key in ("protocol", "status", "train_episode_count", "overfit_gate", "formal_training")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
