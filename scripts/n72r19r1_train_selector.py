#!/usr/bin/env python
"""Train a small N72R19R1 ObservationSelector on train sequences only."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import random

import numpy as np
import torch
from torch.nn import functional as F

from sam3_intermot.identity_memory.dataset import FrozenIdentityEpisodeDataset
from sam3_intermot.identity_memory.r1_protocol import condition_slug, load_corruption_manifest
from sam3_intermot.identity_memory.selective import (
    FEATURE_NAMES,
    ObservationSelector,
    build_evidence_batch,
    feature_indices_for_ablation,
    load_frozen_n72r18_gru,
    update_state_soft,
)

DEFAULT_TRAIN_PROTOCOL = "/data3/liuyeqiang/InterMOT_N72R17_assets/outputs/N72R17/protocol_train.json"
DEFAULT_TRAIN_STORE = "/data3/liuyeqiang/InterMOT_N72R16_assets/outputs/N72R16/embeddings_train"
DEFAULT_PROTOCOL = "outputs/N72R19R1/protocol.json"
DEFAULT_MANIFEST = "outputs/N72R19R1/corruption_manifest.json"
DEFAULT_BASE_CHECKPOINT = "outputs/N72R18/checkpoints/identity_memory_gru.pt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("correctness", "future_utility", "ablation"), default="future_utility")
    parser.add_argument("--ablation", default="A5_full_observation_evidence")
    parser.add_argument("--seed", type=int, default=72191)
    parser.add_argument("--train-protocol", default=DEFAULT_TRAIN_PROTOCOL)
    parser.add_argument("--train-store", default=DEFAULT_TRAIN_STORE)
    parser.add_argument("--protocol", default=DEFAULT_PROTOCOL)
    parser.add_argument("--manifest", default=DEFAULT_MANIFEST)
    parser.add_argument("--base-checkpoint", default=DEFAULT_BASE_CHECKPOINT)
    parser.add_argument("--output-dir", default="outputs/N72R19R1/checkpoints")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs-per-phase", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--lambda-selection", type=float, default=0.5)
    parser.add_argument("--lambda-utility", type=float, default=1.0)
    parser.add_argument("--lambda-preservation", type=float, default=0.1)
    parser.add_argument("--utility-scale", type=float, default=0.05)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _phase_conditions(phase: int) -> tuple[str, ...]:
    conditions = [condition_slug("clean")]
    for rate in (0.10, 0.20, 0.30):
        if phase >= 1 or rate == 0.10:
            conditions.extend((condition_slug("wrong_identity", rate), condition_slug("hard_negative", rate)))
    return tuple(conditions)


def _manifest_observations(batch, manifest: dict[str, object], condition: str, device: torch.device):
    targets = batch["target_embeddings"]
    competitors = batch["competitor_embeddings"]
    frames = batch["frames"]
    masks = batch["target_mask"]
    metadata = batch["metadata"]
    observations = targets.clone()
    present = torch.zeros(masks.shape, dtype=torch.bool, device=device)
    applied = torch.zeros(masks.shape, dtype=torch.bool, device=device)
    for step in range(masks.shape[1]):
        for batch_index, meta in enumerate(metadata):
            if not bool(masks[batch_index, step]):
                continue
            frame = int(frames[batch_index, step].item())
            key = f"{meta.anchor.sequence}|{meta.anchor.track_id}|{meta.anchor.anchor_frame}|{frame}"
            row = manifest["rows"][key]["conditions"][condition]
            present[batch_index, step] = bool(row["present"])
            applied[batch_index, step] = bool(row["applied"])
            replacement_index = row.get("replacement_index")
            if replacement_index is not None:
                observations[batch_index, step] = competitors[batch_index, step, int(replacement_index)]
    return observations, present, applied


def _margin(states: torch.Tensor, targets: torch.Tensor, competitors: torch.Tensor, competitor_mask: torch.Tensor):
    positive = (normalize(states) * normalize(targets)).sum(dim=-1)
    negative = torch.einsum("bd,bcd->bc", normalize(states), normalize(competitors))
    negative = negative.masked_fill(~competitor_mask, -torch.inf)
    has_competitor = competitor_mask.any(dim=-1)
    hard = negative.max(dim=-1).values
    margin = positive - hard
    return margin, has_competitor


def normalize(features: torch.Tensor) -> torch.Tensor:
    return F.normalize(features, dim=-1, eps=1e-12)


def _future_utility(
    state: torch.Tensor,
    candidate: torch.Tensor,
    targets: torch.Tensor,
    competitors: torch.Tensor,
    competitor_mask: torch.Tensor,
    target_mask: torch.Tensor,
    step: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    utilities: list[torch.Tensor] = []
    validities: list[torch.Tensor] = []
    for offset in (1, 5, 10):
        future_step = step + offset
        if future_step >= targets.shape[1]:
            continue
        accept_margin, accept_valid = _margin(candidate, targets[:, future_step], competitors[:, future_step], competitor_mask[:, future_step])
        hold_margin, hold_valid = _margin(state, targets[:, future_step], competitors[:, future_step], competitor_mask[:, future_step])
        valid = target_mask[:, future_step] & accept_valid & hold_valid
        utilities.append(accept_margin - hold_margin)
        validities.append(valid)
    if not utilities:
        return torch.zeros(state.shape[0], device=state.device, dtype=state.dtype), torch.zeros(state.shape[0], device=state.device, dtype=torch.bool)
    utility_stack = torch.stack(utilities, dim=0)
    valid_stack = torch.stack(validities, dim=0)
    utility_sum = torch.where(valid_stack, utility_stack, torch.zeros_like(utility_stack)).sum(dim=0)
    count = valid_stack.sum(dim=0)
    return utility_sum / count.clamp_min(1).to(state.dtype), count > 0


def train_epoch(
    selector: ObservationSelector,
    frozen_gru,
    dataset: FrozenIdentityEpisodeDataset,
    manifest: dict[str, object],
    *,
    conditions: tuple[str, ...],
    epoch: int,
    seed: int,
    device: torch.device,
    optimizer: torch.optim.Optimizer,
    mode: str,
    batch_size: int,
    lambda_selection: float,
    lambda_utility: float,
    lambda_preservation: float,
    utility_scale: float,
) -> dict[str, float]:
    selector.train()
    totals = Counter()
    batch_count = 0
    for batch_index, indices in enumerate(dataset.batch_indices(batch_size=batch_size, shuffle=True, seed=seed + epoch)):
        condition = conditions[(batch_index + epoch) % len(conditions)]
        batch = dataset.make_batch(indices, device=device)
        observations, present, applied = _manifest_observations(batch, manifest, condition, device)
        state = normalize(batch["anchor_embeddings"])
        human_anchor = state.clone()
        targets = batch["target_embeddings"]
        target_mask = batch["target_mask"]
        competitors = batch["competitor_embeddings"]
        competitor_mask = batch["competitor_mask"]
        frames = batch["frames"]
        metadata = batch["metadata"]
        batch_size_actual = state.shape[0]
        anchor_frames = torch.tensor([int(meta.anchor.anchor_frame) for meta in metadata], dtype=torch.long, device=device)
        last_frames = anchor_frames.clone()
        trusted_counts = torch.zeros(batch_size_actual, dtype=torch.long, device=device)
        recent_stability = torch.zeros(batch_size_actual, dtype=state.dtype, device=device)
        loss_sum = torch.zeros((), device=device)
        selection_count = 0
        utility_count = 0
        preservation_count = 0

        for step in range(targets.shape[1]):
            valid = target_mask[:, step]
            if not bool(valid.any()):
                continue
            features, candidate, _, _ = build_evidence_batch(
                state.detach(),
                human_anchor,
                observations[:, step],
                competitors[:, step],
                competitor_mask[:, step],
                frames=frames[:, step],
                last_trusted_update_frames=last_frames,
                trusted_update_counts=trusted_counts,
                recent_state_stability=recent_stability,
                frozen_gru=frozen_gru,
            )
            selection_score = selector(features)
            eligible = valid & present[:, step]
            correct_target = (~applied[:, step]).to(state.dtype)
            if bool(eligible.any()):
                selection_loss = F.binary_cross_entropy(selection_score[eligible], correct_target[eligible])
                loss_sum = loss_sum + lambda_selection * selection_loss
                selection_count += int(eligible.sum().item())

            utility, utility_valid = _future_utility(
                state.detach(),
                candidate.detach(),
                targets,
                competitors,
                competitor_mask,
                target_mask,
                step,
            )
            utility_valid = utility_valid & eligible
            if mode == "future_utility" and bool(utility_valid.any()):
                scaled_utility = utility[utility_valid].clamp(-0.5, 0.5) / utility_scale
                utility_loss = -(selection_score[utility_valid] * scaled_utility).mean()
                loss_sum = loss_sum + lambda_utility * utility_loss
                utility_count += int(utility_valid.sum().item())

            clean = eligible & ~applied[:, step]
            if bool(clean.any()):
                soft_state = update_state_soft(state.detach(), candidate.detach(), selection_score)
                preservation_loss = (1.0 - (soft_state[clean] * candidate.detach()[clean]).sum(dim=-1)).mean()
                loss_sum = loss_sum + lambda_preservation * preservation_loss
                preservation_count += int(clean.sum().item())

            soft_state = update_state_soft(state.detach(), candidate.detach(), selection_score)
            update_mask = eligible
            new_state = torch.where(update_mask.unsqueeze(-1), soft_state, state.detach())
            state_change = 1.0 - (state.detach() * normalize(new_state)).sum(dim=-1)
            state = new_state.detach()
            accepted = update_mask & (selection_score >= 0.5)
            last_frames = torch.where(accepted, frames[:, step], last_frames)
            trusted_counts = trusted_counts + accepted.to(trusted_counts.dtype)
            recent_stability = torch.where(
                accepted,
                0.8 * recent_stability + 0.2 * state_change.detach(),
                recent_stability,
            )

        if batch_count == 0 or bool(loss_sum.requires_grad):
            optimizer.zero_grad(set_to_none=True)
            loss_sum.backward()
            torch.nn.utils.clip_grad_norm_(selector.parameters(), 5.0)
            optimizer.step()
        batch_count += 1
        totals["loss"] += float(loss_sum.detach().item())
        totals["selection_count"] += selection_count
        totals["utility_count"] += utility_count
        totals["preservation_count"] += preservation_count
    return {
        "loss": totals["loss"] / max(batch_count, 1),
        "selection_count": float(totals["selection_count"]),
        "utility_count": float(totals["utility_count"]),
        "preservation_count": float(totals["preservation_count"]),
        "batches": float(batch_count),
    }


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    protocol = json.loads(Path(args.protocol).read_text(encoding="utf-8"))
    manifest = load_corruption_manifest(args.manifest)
    train_sequences = protocol["train_dev_split"]["train_sequences"]
    dataset = FrozenIdentityEpisodeDataset(
        args.train_protocol,
        args.train_store,
        split="train",
        sequence_filter=train_sequences,
    )
    frozen_gru = load_frozen_n72r18_gru(args.base_checkpoint, device)
    if args.mode == "ablation":
        feature_indices = feature_indices_for_ablation(args.ablation)
        run_name = args.ablation
    else:
        feature_indices = tuple(range(len(FEATURE_NAMES)))
        run_name = args.mode
    selector = ObservationSelector(feature_indices=feature_indices).to(device)
    optimizer = torch.optim.AdamW(selector.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    history: list[dict[str, object]] = []
    for phase in range(3):
        conditions = _phase_conditions(phase)
        for phase_epoch in range(args.epochs_per_phase):
            epoch = phase * args.epochs_per_phase + phase_epoch + 1
            stats = train_epoch(
                selector,
                frozen_gru,
                dataset,
                manifest,
                conditions=conditions,
                epoch=epoch,
                seed=args.seed,
                device=device,
                optimizer=optimizer,
                mode=args.mode,
                batch_size=args.batch_size,
                lambda_selection=args.lambda_selection,
                lambda_utility=args.lambda_utility,
                lambda_preservation=args.lambda_preservation,
                utility_scale=args.utility_scale,
            )
            row = {"epoch": epoch, "phase": phase + 1, "conditions": list(conditions), **stats}
            history.append(row)
            print(json.dumps(row, sort_keys=True), flush=True)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / f"selector_{run_name}_seed{args.seed}.pt"
    config = {
        "stage": "N72R19R1",
        "goal": "Selective Identity Memory Update",
        "central_question": "Can selective observation updates preserve the strong learned identity memory while preventing corrupted observations from damaging future identity recognition?",
        "mode": args.mode,
        "run_name": run_name,
        "feature_names": list(FEATURE_NAMES),
        "feature_indices": list(feature_indices),
        "hidden_dims": [64, 32],
        "train_sequences": list(train_sequences),
        "train_protocol": str(Path(args.train_protocol).resolve()),
        "manifest": str(Path(args.manifest).resolve()),
        "base_checkpoint": str(Path(args.base_checkpoint).resolve()),
        "epochs_per_phase": args.epochs_per_phase,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "lambda_selection": args.lambda_selection,
        "lambda_utility": args.lambda_utility,
        "lambda_preservation": args.lambda_preservation,
        "utility_scale": args.utility_scale,
        "seed": args.seed,
        "parameter_count": selector.parameter_count,
        "uses_future_utility_as_inference_feature": False,
        "uses_corruption_label_as_inference_feature": False,
    }
    torch.save(
        {
            "stage": "N72R19R1",
            "goal": "Selective Identity Memory Update",
            "config": config,
            "selector_state_dict": selector.state_dict(),
            "history": history,
        },
        checkpoint_path,
    )
    metrics_dir = Path("outputs/N72R19R1/metrics")
    metrics_dir.mkdir(parents=True, exist_ok=True)
    (metrics_dir / f"train_{run_name}_seed{args.seed}.json").write_text(
        json.dumps({"config": config, "history": history, "checkpoint": str(checkpoint_path.resolve())}, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"checkpoint": str(checkpoint_path.resolve()), "parameter_count": selector.parameter_count}, sort_keys=True))


if __name__ == "__main__":
    main()
