#!/usr/bin/env python3
"""Conditional N28-C first-order episodic meta-training.

This module is deliberately inert unless the machine-readable N28-B LCIA
gate has passed.  Episodes contain one chronological B10 correction support
event and later same-identity query parents.  The inner loop updates only the
episode-local LoRA ``B`` factors; the outer first-order update optimizes the
shared relation backbone and ``A`` factors.  No blind ``val25`` artifact is
read.

The automatic transition uses a bounded CPU/default run so that the four-GPU
limit is never exceeded.  A caller may explicitly choose one device, but the
runner rejects a request for more than four visible CUDA devices.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional

import numpy as np
import torch
from torch.nn import functional as F


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from n28_cached_causal_replay import (  # noqa: E402
    NONE_INDEX,
    build_cached_relation_features,
    build_identity_rows,
    load_cache,
    select_rows,
    sha256,
)
from sam3_intermot.adaptation.live_identity_lora import (  # noqa: E402
    LiveIdentityLoRA,
    LiveLoRAConfig,
)


HELDOUT_CACHE = ROOT / "outputs/n27/data/external_heldout_b10_round0.npz"
TRAIN_ROLES = {
    "external_train": ROOT / "outputs/n27/data/external_train_b10_round0.npz",
    "dancetrack_train_real_p2": ROOT / "outputs/n27/data/dance_train_real_b10_round0.npz",
}
MAX_TRAIN_EPISODES_PER_ROLE = 256
MAX_VALIDATION_EPISODES = 128
QUERY_HORIZON = 5
META_SEED = 28
MODEL_DIM = 64
BLOCKS = 2
ALPHA = 8.0
INNER_LEARNING_RATE = 0.05
OUTER_LEARNING_RATE = 1.0e-3


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


@dataclass(frozen=True)
class Episode:
    role: str
    support_row: int
    sequence: int
    identity: int
    frame: int
    target: int
    rejected: Optional[int]
    query_rows: tuple[int, ...]
    relation: np.ndarray
    anchor: np.ndarray
    mask: np.ndarray
    query_relations: tuple[np.ndarray, ...]
    query_anchors: tuple[np.ndarray, ...]
    query_masks: tuple[np.ndarray, ...]
    query_targets: tuple[int, ...]
    weight: float


def build_episodes(
    role: str,
    path: Path,
    *,
    limit: Optional[int],
) -> tuple[list[Episode], dict[str, Any]]:
    arrays = load_cache(path)
    relation = build_cached_relation_features(
        b10_score=arrays["b10_score"],
        root_similarity=arrays["root_similarity"],
        positive_similarity=arrays["positive_similarity"],
        negative_similarity=arrays["negative_similarity"],
        hard_similarity=arrays["hard_similarity"],
        detector_score=arrays["detector_score"],
        candidate_mask=arrays["candidate_mask"],
    ).astype(np.float32)
    mask = arrays["candidate_mask"].astype(bool)
    target = arrays["target"].astype(np.int64)
    present = arrays["target_present"].astype(bool)
    anchor = arrays["b10_score"].astype(np.float32)
    selected = select_rows(anchor, mask).astype(np.int64)
    eligible = present & (target >= 0) & (target < NONE_INDEX)
    feedback = eligible & (selected != target)
    identity_rows = build_identity_rows(arrays)
    episodes: list[Episode] = []
    for (sequence, identity), indices in sorted(identity_rows.items()):
        feedback_positions = [position for position, row in enumerate(indices) if feedback[row]]
        for feedback_position in feedback_positions:
            support_row = int(indices[feedback_position])
            future: list[int] = []
            for query_row in indices[feedback_position + 1 :]:
                if int(arrays["frame"][query_row]) <= int(arrays["frame"][support_row]):
                    continue
                future.append(int(query_row))
                if feedback[query_row] or len(future) >= QUERY_HORIZON:
                    break
            query_rows = tuple(future)
            if not query_rows:
                continue
            rejected = int(selected[support_row]) if int(selected[support_row]) < NONE_INDEX else None
            episodes.append(
                Episode(
                    role=role,
                    support_row=support_row,
                    sequence=int(sequence),
                    identity=int(identity),
                    frame=int(arrays["frame"][support_row]),
                    target=int(target[support_row]),
                    rejected=rejected,
                    query_rows=query_rows,
                    relation=relation[support_row].copy(),
                    anchor=anchor[support_row].copy(),
                    mask=mask[support_row].copy(),
                    query_relations=tuple(relation[row].copy() for row in query_rows),
                    query_anchors=tuple(anchor[row].copy() for row in query_rows),
                    query_masks=tuple(mask[row].copy() for row in query_rows),
                    query_targets=tuple(int(target[row]) for row in query_rows),
                    weight=float(arrays["parent_weight"][support_row]) if "parent_weight" in arrays else 1.0,
                )
            )
    episodes.sort(key=lambda item: (item.sequence, item.identity, item.frame, item.support_row))
    total = len(episodes)
    if limit is not None:
        episodes = episodes[:limit]
    return episodes, {
        "role": role,
        "cache": str(path.relative_to(ROOT)),
        "cache_sha256": sha256(path),
        "candidate_correction_episodes_with_future": total,
        "episodes_used": len(episodes),
        "limit": limit,
        "query_horizon": QUERY_HORIZON,
        "val25_read": False,
    }


def reset_state(state: torch.nn.Module) -> None:
    with torch.no_grad():
        for parameter in state.factors:
            parameter.zero_()
            parameter.grad = None


def scores_for_row(
    model: LiveIdentityLoRA,
    state: torch.nn.Module,
    relation: np.ndarray,
    anchor: np.ndarray,
    mask: np.ndarray,
) -> torch.Tensor:
    relation_tensor = torch.as_tensor(relation, dtype=torch.float32, device=next(model.parameters()).device).unsqueeze(0)
    anchor_tensor = torch.as_tensor(anchor, dtype=torch.float32, device=relation_tensor.device)
    delta = model._forward(relation_tensor, state)[0] - model._forward(relation_tensor, None)[0]
    scores = anchor_tensor + delta
    scores = scores.masked_fill(~torch.as_tensor(mask, dtype=torch.bool, device=scores.device), -1.0e4)
    return torch.cat([scores, anchor_tensor.new_tensor([0.0])])


def support_loss(
    model: LiveIdentityLoRA,
    state: torch.nn.Module,
    episode: Episode,
) -> torch.Tensor:
    scores = scores_for_row(model, state, episode.relation, episode.anchor, episode.mask)
    loss = F.cross_entropy(scores.unsqueeze(0), torch.tensor([episode.target], device=scores.device))
    if episode.rejected is not None:
        loss = loss + F.relu(scores[episode.rejected] - scores[episode.target] + 0.01)
    return loss


def apply_inner_updates(
    model: LiveIdentityLoRA,
    state: torch.nn.Module,
    episode: Episode,
    inner_steps: int,
) -> None:
    for _ in range(inner_steps):
        loss = support_loss(model, state, episode)
        gradients = torch.autograd.grad(
            loss,
            tuple(state.factors),
            create_graph=False,
            allow_unused=True,
        )
        with torch.no_grad():
            for parameter, gradient in zip(state.factors, gradients):
                if gradient is not None:
                    parameter.add_(-INNER_LEARNING_RATE * gradient.clamp(-10.0, 10.0))


def query_loss_and_errors(
    model: LiveIdentityLoRA,
    state: torch.nn.Module,
    episode: Episode,
    *,
    with_grad: bool,
) -> tuple[torch.Tensor, int, int]:
    losses: list[torch.Tensor] = []
    adapted_errors = 0
    baseline_errors = 0
    context = torch.enable_grad() if with_grad else torch.no_grad()
    with context:
        for relation, anchor, mask, target in zip(
            episode.query_relations,
            episode.query_anchors,
            episode.query_masks,
            episode.query_targets,
        ):
            scores = scores_for_row(model, state, relation, anchor, mask)
            losses.append(F.cross_entropy(scores.unsqueeze(0), torch.tensor([target], device=scores.device)))
            adapted = int(torch.argmax(scores).item())
            baseline_scores = torch.as_tensor(anchor, dtype=torch.float32, device=scores.device)
            baseline_scores = baseline_scores.masked_fill(~torch.as_tensor(mask, dtype=torch.bool, device=scores.device), -1.0e4)
            baseline = int(torch.argmax(torch.cat([baseline_scores, baseline_scores.new_tensor([0.0])])).item())
            adapted_errors += int(adapted != target)
            baseline_errors += int(baseline != target)
    return torch.stack(losses).mean(), adapted_errors, baseline_errors


def train_config(
    episodes: list[Episode],
    validation_episodes: list[Episode],
    *,
    rank: int,
    inner_steps: int,
    device: torch.device,
) -> tuple[dict[str, Any], LiveIdentityLoRA]:
    torch.manual_seed(META_SEED + rank * 100 + inner_steps)
    model = LiveIdentityLoRA(
        LiveLoRAConfig(
            input_dim=7,
            d_model=MODEL_DIM,
            rank=rank,
            blocks=BLOCKS,
            alpha=ALPHA,
            seed=META_SEED,
        )
    )
    state = model.ensure_identity("meta_episode")
    model.to(device)
    shared_parameters = [
        parameter
        for name, parameter in model.named_parameters()
        if not name.startswith("fast_states.")
    ]
    for parameter in shared_parameters:
        parameter.requires_grad_(True)
    optimizer = torch.optim.Adam(shared_parameters, lr=OUTER_LEARNING_RATE)
    train_loss_values: list[float] = []
    train_adapted_errors = 0
    train_baseline_errors = 0
    for episode in episodes:
        reset_state(state)
        optimizer.zero_grad(set_to_none=True)
        apply_inner_updates(model, state, episode, inner_steps)
        query_loss, adapted_errors, baseline_errors = query_loss_and_errors(
            model, state, episode, with_grad=True
        )
        (episode.weight * query_loss).backward()
        torch.nn.utils.clip_grad_norm_(shared_parameters, 1.0)
        optimizer.step()
        train_loss_values.append(float(query_loss.detach().cpu()))
        train_adapted_errors += adapted_errors
        train_baseline_errors += baseline_errors

    validation_loss: list[float] = []
    validation_adapted_errors = 0
    validation_baseline_errors = 0
    for episode in validation_episodes:
        reset_state(state)
        apply_inner_updates(model, state, episode, inner_steps)
        query_loss, adapted_errors, baseline_errors = query_loss_and_errors(
            model, state, episode, with_grad=False
        )
        validation_loss.append(float(query_loss.cpu()))
        validation_adapted_errors += adapted_errors
        validation_baseline_errors += baseline_errors

    train_queries = max(1, sum(len(episode.query_rows) for episode in episodes))
    validation_queries = max(1, sum(len(episode.query_rows) for episode in validation_episodes))
    result = {
        "rank": rank,
        "inner_steps": inner_steps,
        "train_episodes": len(episodes),
        "validation_episodes": len(validation_episodes),
        "train_query_loss": float(np.mean(train_loss_values)) if train_loss_values else None,
        "validation_query_loss": float(np.mean(validation_loss)) if validation_loss else None,
        "train_adapted_error": float(train_adapted_errors / train_queries),
        "train_baseline_error": float(train_baseline_errors / train_queries),
        "validation_adapted_error": float(validation_adapted_errors / validation_queries),
        "validation_baseline_error": float(validation_baseline_errors / validation_queries),
        "validation_error_delta": float((validation_adapted_errors - validation_baseline_errors) / validation_queries),
        "first_order": True,
        "inner_updates_B_only": True,
        "outer_updates_shared_backbone_and_A": True,
        "val25_read": False,
    }
    return result, model


def run_n28c(
    b_result: dict[str, Any],
    *,
    output: Path = ROOT / "outputs/n28/n28c_result.json",
    checkpoint_dir: Path = ROOT / "outputs/n28/checkpoints",
    device_name: str = "cpu",
    full_sweep: bool = True,
) -> dict[str, Any]:
    started = time.monotonic()
    gate_passed = bool(b_result.get("primary_lcia_gate", {}).get("passed", False))
    if not gate_passed:
        result = {
            "phase": "N28-C",
            "status": "NOT_AUTHORIZED_N28_B_GATE_FAIL",
            "meta_training_started": False,
            "val25_read": False,
            "blocking_gate": "N28-B primary LCIA gate",
            "blocking_result": b_result.get("primary_lcia_gate", {}),
            "elapsed_seconds": time.monotonic() - started,
        }
        atomic_json(output, result)
        return result

    if torch.cuda.is_available() and device_name.startswith("cuda"):
        requested = int(device_name.split(":", 1)[1]) if ":" in device_name else 0
        if requested < 0 or requested >= torch.cuda.device_count():
            raise ValueError(f"requested CUDA device is unavailable: {device_name}")
        device = torch.device(device_name)
        gpu_count_used = 1
    else:
        device = torch.device("cpu")
        gpu_count_used = 0
    if gpu_count_used > 4:
        raise RuntimeError("N28-C exceeds the four-GPU limit")
    torch.set_num_threads(1)

    train_episodes: list[Episode] = []
    train_audit: list[dict[str, Any]] = []
    for role, path in TRAIN_ROLES.items():
        episodes, audit = build_episodes(role, path, limit=MAX_TRAIN_EPISODES_PER_ROLE)
        train_episodes.extend(episodes)
        train_audit.append(audit)
    validation_episodes, validation_audit = build_episodes(
        "external_heldout", HELDOUT_CACHE, limit=MAX_VALIDATION_EPISODES
    )
    configs = [(rank, steps) for rank in (4, 8, 16) for steps in (1, 5, 10, 20, 40)]
    if not full_sweep:
        configs = [(4, 1), (8, 5), (16, 5)]
    results: list[dict[str, Any]] = []
    best: Optional[tuple[dict[str, Any], LiveIdentityLoRA]] = None
    for rank, steps in configs:
        config_result, model = train_config(
            train_episodes,
            validation_episodes,
            rank=rank,
            inner_steps=steps,
            device=device,
        )
        results.append(config_result)
        if config_result["validation_query_loss"] is not None and (
            best is None or config_result["validation_query_loss"] < best[0]["validation_query_loss"]
        ):
            best = (config_result, model)

    checkpoint_path = None
    best_result = best[0] if best is not None else None
    if best is not None:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_path = checkpoint_dir / "n28c_lcia_best.pt"
        model = best[1]
        model.reset()
        torch.save(
            {
                "phase": "N28-C",
                "model_config": {
                    "input_dim": 7,
                    "d_model": MODEL_DIM,
                    "rank": int(best_result["rank"]),
                    "blocks": BLOCKS,
                    "alpha": ALPHA,
                    "seed": META_SEED,
                },
                "model_state": model.state_dict(),
                "selection": best_result,
                "val25_read": False,
            },
            checkpoint_path,
        )
    result = {
        "phase": "N28-C",
        "status": "META_TRAINING_COMPLETE" if results else "META_TRAINING_NOT_COMPUTABLE",
        "meta_training_started": True,
        "val25_read": False,
        "device": str(device),
        "gpu_count_used": gpu_count_used,
        "max_simultaneous_gpus": 4,
        "first_order": True,
        "train_audit": train_audit,
        "validation_audit": validation_audit,
        "sweep": {
            "ranks": [4, 8, 16],
            "inner_steps": [1, 5, 10, 20, 40] if full_sweep else [1, 5],
            "configs_run": len(configs),
            "max_train_episodes_per_role": MAX_TRAIN_EPISODES_PER_ROLE,
            "max_validation_episodes": MAX_VALIDATION_EPISODES,
        },
        "results": results,
        "best": best_result,
        "checkpoint": str(checkpoint_path.relative_to(ROOT)) if checkpoint_path else None,
        "checkpoint_sha256": sha256(checkpoint_path) if checkpoint_path else None,
        "elapsed_seconds": time.monotonic() - started,
        "transition": "N28-D_NOT_AUTHORIZED_BY_THIS_PHASE",
    }
    atomic_json(output, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--b-result", type=Path, default=ROOT / "outputs/n28/n28b_result.json")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/n28/n28c_result.json")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--short-sweep", action="store_true")
    args = parser.parse_args()
    b_result = json.loads(args.b_result.read_text(encoding="utf-8"))
    result = run_n28c(
        b_result,
        output=args.output,
        device_name=args.device,
        full_sweep=not args.short_sweep,
    )
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    print("N28C_META_TRAINING_COMPLETE" if result["meta_training_started"] else "N28C_NOT_AUTHORIZED", flush=True)


if __name__ == "__main__":
    main()
