#!/usr/bin/env python
"""Train N72R19 LIMN on deterministic mixtures of noisy identity observations."""

from __future__ import annotations

import argparse
import json
import random
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from sam3_intermot.identity_memory.dataset import FrozenIdentityEpisodeDataset
from sam3_intermot.identity_memory.encoder import FEATURE_DIMENSION, FROZEN_ENCODER_NAME, normalize
from sam3_intermot.identity_memory.losses import hard_negative_info_nce, stability_loss
from sam3_intermot.identity_memory.noise import corrupt_observation, training_noise_spec
from sam3_intermot.identity_research.protocol import protocol_sha256
from sam3_intermot.identity_memory.updater import VARIANT_GRU_GATE, build_updater


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--train-protocol",
        default="/data3/liuyeqiang/InterMOT_N72R17_assets/outputs/N72R17/protocol_train.json",
    )
    parser.add_argument(
        "--train-store",
        default="/data3/liuyeqiang/InterMOT_N72R16_assets/outputs/N72R16/embeddings_train",
    )
    parser.add_argument("--output-dir", default="outputs/N72R19")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--stability-lambda", type=float, default=0.01)
    parser.add_argument("--gate-hidden-dim", type=int, default=256)
    parser.add_argument("--gradient-clip", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=7219)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _episode_key(meta, frame: int) -> str:
    anchor = meta.anchor
    return f"{anchor.sequence}|{anchor.track_id}|{anchor.anchor_frame}|{frame}"


def noisy_observation_batch(
    state: torch.Tensor,
    clean_observations: torch.Tensor,
    competitors: torch.Tensor,
    competitor_mask: torch.Tensor,
    target_mask: torch.Tensor,
    metadata: list[object],
    frames: torch.Tensor,
    step: int,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor, list[dict[str, object]]]:
    """Construct a differentiability-safe batch of corrupted observations."""

    observations = clean_observations.clone()
    present = torch.zeros(target_mask.shape, dtype=torch.bool, device=target_mask.device)
    details: list[dict[str, object]] = []
    for batch_index, meta in enumerate(metadata):
        if not bool(target_mask[batch_index]):
            details.append({"kind": "padding", "rate": 0.0, "applied": False, "present": False})
            continue
        frame = int(frames[batch_index].item())
        key = _episode_key(meta, frame)
        spec = training_noise_spec(key, step, seed=seed)
        result = corrupt_observation(
            state[batch_index],
            clean_observations[batch_index],
            competitors[batch_index],
            competitor_mask[batch_index],
            spec,
            key,
            step,
        )
        observations[batch_index] = result.observation
        present[batch_index] = result.present
        details.append(
            {
                "kind": spec.kind,
                "rate": spec.rate,
                "applied": result.applied,
                "present": result.present,
                "replacement_index": result.replacement_index,
                "source": result.source,
            }
        )
    return observations, present, details


def batch_loss(
    model,
    batch: dict[str, object],
    temperature: float,
    stability_lambda: float,
    seed: int,
) -> tuple[torch.Tensor, dict[str, float], Counter[str]]:
    state = normalize(batch["anchor_embeddings"])
    targets = batch["target_embeddings"]
    target_mask = batch["target_mask"]
    competitors = batch["competitor_embeddings"]
    competitor_mask = batch["competitor_mask"]
    metadata = batch["metadata"]
    frames = batch["frames"]
    max_steps = targets.shape[1]

    contrastive_sum = torch.zeros((), device=state.device, dtype=state.dtype)
    stability_sum = torch.zeros((), device=state.device, dtype=state.dtype)
    competitive_count = 0
    update_count = 0
    condition_counts: Counter[str] = Counter()

    for step in range(max_steps):
        observations, present, details = noisy_observation_batch(
            state,
            targets[:, step],
            competitors[:, step],
            competitor_mask[:, step],
            target_mask[:, step],
            metadata,
            frames[:, step],
            step,
            seed,
        )
        for detail in details:
            if detail["kind"] != "padding":
                condition_counts[str(detail["kind"])] += 1

        candidate_state, _reliability, _ = model(state, observations)
        update_mask = target_mask[:, step] & present
        new_state = torch.where(update_mask.unsqueeze(-1), candidate_state, state)

        valid_competitive = target_mask[:, step] & competitor_mask[:, step].any(dim=1)
        if bool(valid_competitive.any()):
            losses = hard_negative_info_nce(
                new_state,
                targets[:, step],
                competitors[:, step],
                competitor_mask[:, step],
                temperature=temperature,
            )
            contrastive_sum = contrastive_sum + losses.masked_select(valid_competitive).sum()
            competitive_count += int(valid_competitive.sum().item())

        step_stability = stability_loss(state, new_state)
        if bool(update_mask.any()):
            stability_sum = stability_sum + step_stability.masked_select(update_mask).sum()
            update_count += int(update_mask.sum().item())
        state = new_state

    if competitive_count == 0:
        raise ValueError("training batch has no competitive future frames")
    contrastive = contrastive_sum / competitive_count
    stability = stability_sum / max(update_count, 1)
    total = contrastive + stability_lambda * stability
    return total, {
        "loss": float(total.detach().item()),
        "contrastive_loss": float(contrastive.detach().item()),
        "stability_loss": float(stability.detach().item()),
        "competitive_frames": float(competitive_count),
        "observed_updates": float(update_count),
    }, condition_counts


def main() -> None:
    args = parse_args()
    if args.epochs < 1 or args.batch_size < 1:
        raise ValueError("epochs and batch-size must be positive")
    set_seed(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")

    dataset = FrozenIdentityEpisodeDataset(args.train_protocol, args.train_store, split="train")
    model = build_updater(
        VARIANT_GRU_GATE,
        feature_dim=FEATURE_DIMENSION,
        gate_hidden_dim=args.gate_hidden_dim,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    output_dir = Path(args.output_dir)
    checkpoint_dir = output_dir / "checkpoints"
    metrics_dir = output_dir / "metrics"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    config = {
        "stage": "N72R19",
        "goal": "Robust Human Identity Memory Learning",
        "goal_family": "Human Identity Memory Learning",
        "goal_reference": "outputs/N72R19/FINAL_GOAL.json",
        "central_question": "Can learned identity memory outperform EMA under noisy observation conditions?",
        "variant": VARIANT_GRU_GATE,
        "feature_encoder": FROZEN_ENCODER_NAME,
        "feature_dimension": FEATURE_DIMENSION,
        "train_protocol": str(Path(args.train_protocol).resolve()),
        "train_protocol_sha256": protocol_sha256(args.train_protocol),
        "train_store": str(Path(args.train_store).resolve()),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "temperature": args.temperature,
        "stability_lambda": args.stability_lambda,
        "gate_hidden_dim": args.gate_hidden_dim,
        "gradient_clip": args.gradient_clip,
        "seed": args.seed,
        "training_observation": "deterministic_mixture_of_clean_wrong_missing_hard_negative",
        "training_condition_probabilities": {
            "clean": 0.2,
            "wrong_identity_injection": 0.3,
            "missing_observation": 0.2,
            "hard_negative_replacement": 0.3,
        },
        "training_noise_rates": [0.1, 0.2, 0.3, 0.5],
        "score_then_update_evaluation": True,
        "uses_sam3": False,
        "uses_mot_association": False,
        "uses_trackeval": False,
        "new_reid_encoder_training": False,
        "raw_crops_saved": False,
    }
    (metrics_dir / "train_robust_config.json").write_text(
        json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    history: list[dict[str, object]] = []
    total_started = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        totals = {"loss": 0.0, "contrastive_loss": 0.0, "stability_loss": 0.0}
        condition_counts: Counter[str] = Counter()
        batches = 0
        epoch_started = time.time()
        for indices in dataset.batch_indices(batch_size=args.batch_size, shuffle=True, seed=args.seed + epoch):
            batch = dataset.make_batch(indices, device=device)
            loss, stats, counts = batch_loss(
                model,
                batch,
                temperature=args.temperature,
                stability_lambda=args.stability_lambda,
                seed=args.seed,
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clip)
            optimizer.step()
            batches += 1
            condition_counts.update(counts)
            for key in totals:
                totals[key] += stats[key]

        row = {
            "epoch": epoch,
            "batches": batches,
            "loss": totals["loss"] / max(batches, 1),
            "contrastive_loss": totals["contrastive_loss"] / max(batches, 1),
            "stability_loss": totals["stability_loss"] / max(batches, 1),
            "condition_steps": dict(sorted(condition_counts.items())),
            "epoch_seconds": time.time() - epoch_started,
        }
        history.append(row)
        print(json.dumps(row, sort_keys=True), flush=True)

    checkpoint_path = checkpoint_dir / "identity_memory_robust_gate.pt"
    torch.save(
        {
            "stage": "N72R19",
            "goal": "Robust Human Identity Memory Learning",
            "goal_family": "Human Identity Memory Learning",
            "goal_reference": "outputs/N72R19/FINAL_GOAL.json",
            "variant": VARIANT_GRU_GATE,
            "config": config,
            "model_state_dict": model.state_dict(),
            "training_history": history,
            "elapsed_seconds": time.time() - total_started,
        },
        checkpoint_path,
    )
    (metrics_dir / "train_robust.json").write_text(
        json.dumps(
            {
                "config": config,
                "history": history,
                "checkpoint": str(checkpoint_path.resolve()),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {"checkpoint": str(checkpoint_path.resolve()), "elapsed_seconds": time.time() - total_started},
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
