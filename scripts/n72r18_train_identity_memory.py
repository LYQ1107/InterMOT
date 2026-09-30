#!/usr/bin/env python
"""Train one N72R18 HIIM variant on frozen N72R17 OSNet episodes."""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch

from sam3_intermot.identity_memory.dataset import FrozenIdentityEpisodeDataset
from sam3_intermot.identity_memory.encoder import FEATURE_DIMENSION, FROZEN_ENCODER_NAME, normalize
from sam3_intermot.identity_memory.losses import hard_negative_info_nce, stability_loss
from sam3_intermot.identity_memory.updater import (
    VARIANT_GRU,
    VARIANT_GRU_GATE,
    build_updater,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=("gru", "gru_gate"), required=True)
    parser.add_argument(
        "--train-protocol",
        default="/data3/liuyeqiang/InterMOT_N72R17_assets/outputs/N72R17/protocol_train.json",
    )
    parser.add_argument(
        "--train-store",
        default="/data3/liuyeqiang/InterMOT_N72R16_assets/outputs/N72R16/embeddings_train",
    )
    parser.add_argument("--output-dir", default="outputs/N72R18")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--stability-lambda", type=float, default=0.01)
    parser.add_argument("--gate-hidden-dim", type=int, default=256)
    parser.add_argument("--gradient-clip", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=7218)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_variant(name: str) -> str:
    return VARIANT_GRU if name == "gru" else VARIANT_GRU_GATE


def batch_loss(
    model,
    batch: dict[str, object],
    temperature: float,
    stability_lambda: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    state = normalize(batch["anchor_embeddings"])
    targets = batch["target_embeddings"]
    target_mask = batch["target_mask"]
    competitors = batch["competitor_embeddings"]
    competitor_mask = batch["competitor_mask"]
    max_steps = targets.shape[1]

    contrastive_sum = torch.zeros((), device=state.device, dtype=state.dtype)
    stability_sum = torch.zeros((), device=state.device, dtype=state.dtype)
    competitive_count = 0
    target_count = 0

    for step in range(max_steps):
        valid_targets = target_mask[:, step]
        valid_competitive = competitor_mask[:, step].any(dim=1)
        if bool(valid_competitive.any()):
            losses = hard_negative_info_nce(
                state,
                targets[:, step],
                competitors[:, step],
                competitor_mask[:, step],
                temperature=temperature,
            )
            contrastive_sum = contrastive_sum + losses.masked_select(valid_competitive).sum()
            competitive_count += int(valid_competitive.sum().item())

        new_state, _, _ = model(state, targets[:, step])
        step_stability = stability_loss(state, new_state)
        if bool(valid_targets.any()):
            stability_sum = stability_sum + step_stability.masked_select(valid_targets).sum()
            target_count += int(valid_targets.sum().item())
        state = torch.where(valid_targets.unsqueeze(-1), new_state, state)

    if competitive_count == 0:
        raise ValueError("training batch has no competitive future frames")
    contrastive = contrastive_sum / competitive_count
    stability = stability_sum / max(target_count, 1)
    total = contrastive + stability_lambda * stability
    return total, {
        "loss": float(total.detach().item()),
        "contrastive_loss": float(contrastive.detach().item()),
        "stability_loss": float(stability.detach().item()),
        "competitive_frames": float(competitive_count),
        "target_updates": float(target_count),
    }


def main() -> None:
    args = parse_args()
    if args.epochs < 1:
        raise ValueError("epochs must be positive")
    set_seed(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")

    variant = resolve_variant(args.variant)
    dataset = FrozenIdentityEpisodeDataset(args.train_protocol, args.train_store, split="train")
    model = build_updater(variant, feature_dim=FEATURE_DIMENSION, gate_hidden_dim=args.gate_hidden_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    output_dir = Path(args.output_dir)
    checkpoint_dir = output_dir / "checkpoints"
    metrics_dir = output_dir / "metrics"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    config = {
        "stage": "N72R18",
        "goal": "Human Identity Memory Learning",
        "central_question": "Can a learned identity memory updater build a more reliable long-term identity representation than a fixed EMA from sparse human-confirmed observations?",
        "variant": variant,
        "feature_encoder": FROZEN_ENCODER_NAME,
        "feature_dimension": FEATURE_DIMENSION,
        "train_protocol": str(Path(args.train_protocol).resolve()),
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
        "score_then_update": True,
        "same_identity_observation_source": "future_gt_crop_offline_causal_replay",
        "uses_sam3": False,
        "uses_mot_association": False,
        "uses_trackeval": False,
    }
    (metrics_dir / f"train_{args.variant}_config.json").write_text(
        json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    history: list[dict[str, object]] = []
    started = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        totals = {"loss": 0.0, "contrastive_loss": 0.0, "stability_loss": 0.0}
        batches = 0
        epoch_started = time.time()
        for indices in dataset.batch_indices(batch_size=args.batch_size, shuffle=True, seed=args.seed + epoch):
            batch = dataset.make_batch(indices, device=device)
            loss, stats = batch_loss(
                model,
                batch,
                temperature=args.temperature,
                stability_lambda=args.stability_lambda,
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clip)
            optimizer.step()
            batches += 1
            for key in totals:
                totals[key] += stats[key]

        row = {
            "epoch": epoch,
            "batches": batches,
            "loss": totals["loss"] / max(batches, 1),
            "contrastive_loss": totals["contrastive_loss"] / max(batches, 1),
            "stability_loss": totals["stability_loss"] / max(batches, 1),
            "epoch_seconds": time.time() - epoch_started,
        }
        history.append(row)
        print(json.dumps(row, sort_keys=True), flush=True)

    checkpoint = {
        "stage": "N72R18",
        "goal": "Human Identity Memory Learning",
        "variant": variant,
        "config": config,
        "model_state_dict": model.state_dict(),
        "training_history": history,
        "elapsed_seconds": time.time() - started,
    }
    checkpoint_path = checkpoint_dir / f"identity_memory_{args.variant}.pt"
    torch.save(checkpoint, checkpoint_path)
    (metrics_dir / f"train_{args.variant}.json").write_text(
        json.dumps({"config": config, "history": history, "checkpoint": str(checkpoint_path.resolve())}, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"checkpoint": str(checkpoint_path.resolve()), "elapsed_seconds": time.time() - started}, sort_keys=True))


if __name__ == "__main__":
    main()
