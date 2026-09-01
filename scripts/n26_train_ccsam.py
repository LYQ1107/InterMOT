#!/usr/bin/env python3
"""Four-GPU DDP training for the N26 CC-SAM association module."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
from torch import nn
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import ConcatDataset, DataLoader, Dataset, Subset
from torch.utils.data.distributed import DistributedSampler


ROOT = Path(".")
OUT = ROOT / "outputs/n26"
CHECKPOINTS = OUT / "checkpoints"
sys.path.insert(0, str(ROOT / "scripts"))
from n26_ccsam_model import CCSAM, CCSAMConfig, count_parameters  # noqa: E402


LOSS_WEIGHTS = {
    "listwise": 1.0,
    "human_negative": 0.2,
    "positive_contrast": 0.1,
    "correction_response": 0.5,
    "existence": 0.3,
    "risk": 0.5,
}
SELECTION_VALIDATION = [
    "dancetrack0008", "dancetrack0023", "dancetrack0037", "dancetrack0052", "dancetrack0068"
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def seed_everything(seed: int, rank: int = 0) -> None:
    random.seed(seed + rank)
    np.random.seed(seed + rank)
    torch.manual_seed(seed + rank)
    torch.cuda.manual_seed_all(seed + rank)


class DenseDataset(Dataset):
    def __init__(self, path: Path, weight_scale: float = 1.0):
        self.path = path
        with np.load(path, allow_pickle=False) as z:
            self.arrays = {name: z[name].copy() for name in z.files}
        self.weight_scale = float(weight_scale)

    def __len__(self) -> int:
        return len(self.arrays["target"])

    def __getitem__(self, index: int) -> dict[str, Any]:
        parent = int(self.arrays["parent"][index])
        return {
            "candidate_clip": self.arrays["candidate_clip"][index],
            "candidate_scalar": self.arrays["candidate_scalar"][index],
            "candidate_mask": self.arrays["candidate_mask"][index],
            "candidate_label": self.arrays["candidate_label"][index],
            "target": int(self.arrays["target"][index]),
            "existence": float(self.arrays["existence"][index]),
            "existence_mask": bool(self.arrays["existence_mask"][index]),
            "memory_clip": self.arrays["memory_clip"][parent],
            "memory_meta": self.arrays["memory_meta"][parent],
            "memory_mask": self.arrays["memory_mask"][parent],
            "memory_pre_mask": self.arrays["memory_pre_mask"][parent],
            "memory_kind": self.arrays["memory_kind"][parent],
            "pair_valid": bool(self.arrays["pair_valid"][index]),
            "rejected_index": int(self.arrays["rejected_index"][index]),
            "primary_h5": bool(self.arrays["primary_h5"][index]),
            "sample_weight": float(self.arrays["sample_weight"][index]) * self.weight_scale,
            "sequence": int(self.arrays["sequence"][index]),
            "parent": parent,
            "state_index": int(self.arrays["state_index"][index]),
            "index": index,
        }


def to_device(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {name: value.to(device, non_blocking=True) for name, value in batch.items()}


def weighted_mean(values: torch.Tensor, weights: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
    active = weights if mask is None else weights * mask.float()
    denominator = active.sum()
    if float(denominator.detach()) == 0.0:
        return values.sum() * 0.0
    return (values * active).sum() / denominator


def compute_loss(model: nn.Module, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    kwargs = {
        "candidate_clip": batch["candidate_clip"],
        "candidate_scalar": batch["candidate_scalar"],
        "candidate_mask": batch["candidate_mask"],
        "memory_clip": batch["memory_clip"],
        "memory_meta": batch["memory_meta"],
        "memory_mask": batch["memory_mask"],
        "memory_kind": batch["memory_kind"],
    }
    output = model(**kwargs, memory_mode="positive_negative")
    pre_output = model(**{**kwargs, "memory_mask": batch["memory_pre_mask"]}, memory_mode="positive_negative")
    weights = batch["sample_weight"].float()
    target = batch["target"].long()
    listwise = weighted_mean(F.cross_entropy(output["logits"], target, reduction="none"), weights)

    existence_values = F.binary_cross_entropy_with_logits(output["existence_logit"], batch["existence"].float(), reduction="none")
    existence = weighted_mean(existence_values, weights, batch["existence_mask"].bool())
    risk_values = F.binary_cross_entropy_with_logits(output["risk_logits"], batch["candidate_label"].float(), reduction="none")
    risk_state = (risk_values * batch["candidate_mask"].float()).sum(1) / batch["candidate_mask"].sum(1).clamp_min(1)
    risk = weighted_mean(risk_state, weights)

    target_logit = output["logits"].gather(1, target[:, None]).squeeze(1)
    rejected = batch["rejected_index"].long().clamp_min(0)
    rejected_logit = output["candidate_logits"].gather(1, rejected[:, None]).squeeze(1)
    pair = batch["pair_valid"].bool() & batch["rejected_index"].ge(0)
    pair &= target.eq(5) | target.ne(rejected)
    human_negative = weighted_mean(F.relu(rejected_logit - target_logit + 0.2), weights, pair)

    candidate = F.normalize(batch["candidate_clip"].float(), dim=-1)
    memory = F.normalize(batch["memory_clip"].float(), dim=-1)
    similarity = torch.einsum("bkd,bmd->bkm", candidate, memory)
    positive_memory = batch["memory_mask"].bool() & batch["memory_kind"].eq(1)
    positive_similarity = similarity.masked_fill(~positive_memory[:, None, :], -1e4).max(dim=-1).values
    candidate_target = target.lt(5)
    target_index = target.clamp_max(4)
    positive_target = positive_similarity.gather(1, target_index[:, None]).squeeze(1)
    wrong_mask = batch["candidate_mask"].bool() & ~batch["candidate_label"].bool()
    hardest_wrong = positive_similarity.masked_fill(~wrong_mask, -1e4).max(dim=1).values
    positive_pair = candidate_target & positive_memory.any(dim=1) & wrong_mask.any(dim=1)
    positive_contrast = weighted_mean(F.relu(hardest_wrong - positive_target + 0.1), weights, positive_pair)

    pre_target = pre_output["logits"].gather(1, target[:, None]).squeeze(1)
    pre_rejected = pre_output["candidate_logits"].gather(1, rejected[:, None]).squeeze(1)
    response_values = F.relu(rejected_logit - pre_rejected + 0.02) + F.relu(pre_target - target_logit + 0.02)
    correction_response = weighted_mean(response_values, weights, pair)
    components = {
        "listwise": listwise,
        "human_negative": human_negative,
        "positive_contrast": positive_contrast,
        "correction_response": correction_response,
        "existence": existence,
        "risk": risk,
    }
    total = sum(LOSS_WEIGHTS[name] * value for name, value in components.items())
    return total, components, output


def make_loader(dataset: Dataset, batch_size: int, rank: int, world_size: int, epoch: int, workers: int) -> tuple[DataLoader, DistributedSampler]:
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True, seed=26, drop_last=False)
    sampler.set_epoch(epoch)
    loader = DataLoader(dataset, batch_size=batch_size, sampler=sampler, num_workers=workers, pin_memory=True, persistent_workers=workers > 0, drop_last=False)
    return loader, sampler


def train_epoch(model: DDP, dataset: Dataset, optimizer: torch.optim.Optimizer, device: torch.device, epoch: int, args: argparse.Namespace, rank: int, world_size: int) -> dict[str, float]:
    model.train()
    loader, _ = make_loader(dataset, args.batch_size, rank, world_size, epoch, args.workers)
    totals = torch.zeros(11, dtype=torch.float64, device=device)
    started = time.time()
    for batch in loader:
        batch = to_device(batch, device)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            loss, components, output = compute_loss(model, batch)
        loss.backward()
        gradient = torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clip)
        optimizer.step()
        weight = float(batch["sample_weight"].sum().detach())
        correct = output["logits"].argmax(1).eq(batch["target"]).float()
        totals[0] += float(loss.detach()) * weight
        for offset, name in enumerate(LOSS_WEIGHTS, start=1):
            totals[offset] += float(components[name].detach()) * weight
        totals[7] += weight
        totals[8] += float(gradient.detach())
        totals[9] += 1
        totals[10] += float((correct * batch["sample_weight"]).sum().detach())
    dist.all_reduce(totals, op=dist.ReduceOp.SUM)
    elapsed = max(1e-6, time.time() - started)
    denominator = max(1e-8, float(totals[7]))
    result = {"train_total_loss": float(totals[0]) / denominator}
    for offset, name in enumerate(LOSS_WEIGHTS, start=1):
        result[f"train_{name}_loss"] = float(totals[offset]) / denominator
    result.update({
        "train_parent_weight": float(totals[7]),
        "train_gradient_norm": float(totals[8]) / max(1.0, float(totals[9])),
        "train_parent_weighted_accuracy": float(totals[10]) / denominator,
        "train_steps": int(totals[9]),
        "train_states_per_second": len(dataset) / elapsed,
        "epoch_wall_seconds": elapsed,
    })
    return result


@torch.no_grad()
def evaluate(model: nn.Module, dataset: Dataset, device: torch.device, batch_size: int, workers: int) -> dict[str, float]:
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size * 2, shuffle=False, num_workers=workers, pin_memory=True, persistent_workers=workers > 0)
    sums = {name: 0.0 for name in ["total", *LOSS_WEIGHTS]}
    weight_sum = 0.0
    h5_correct = h5_weight = all_correct = 0.0
    for batch in loader:
        batch = to_device(batch, device)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            loss, components, output = compute_loss(model, batch)
        weight = float(batch["sample_weight"].sum())
        sums["total"] += float(loss) * weight
        for name, value in components.items():
            sums[name] += float(value) * weight
        weight_sum += weight
        correct = output["logits"].argmax(1).eq(batch["target"]).float()
        all_correct += float((correct * batch["sample_weight"]).sum())
        h5 = batch["primary_h5"].bool()
        h5_correct += float(correct[h5].sum())
        h5_weight += int(h5.sum())
    result = {"val_total_loss": sums["total"] / max(1e-8, weight_sum)}
    for name in LOSS_WEIGHTS:
        result[f"val_{name}_loss"] = sums[name] / max(1e-8, weight_sum)
    result["val_parent_weighted_accuracy"] = all_correct / max(1e-8, weight_sum)
    result["val_h5_top1"] = h5_correct / max(1.0, h5_weight)
    result["val_parent_weight"] = weight_sum
    result["val_h5_states"] = h5_weight
    return result


def optimizer_for(model: nn.Module, args: argparse.Namespace) -> torch.optim.Optimizer:
    return torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)


def cpu_state(model: nn.Module) -> dict[str, torch.Tensor]:
    target = model.module if isinstance(model, DDP) else model
    return {name: value.detach().cpu() for name, value in target.state_dict().items()}


def save_checkpoint(path: Path, model: nn.Module, optimizer: torch.optim.Optimizer, *, stage: str, epoch: int, args: argparse.Namespace, data_paths: list[Path], extra: dict[str, Any]) -> None:
    payload = {
        "format": "N26_CCSAM_RESUMABLE_V1", "stage": stage, "epoch": epoch, "seed": args.seed,
        "model_config": CCSAMConfig().__dict__, "model_state": cpu_state(model), "optimizer_state": optimizer.state_dict(),
        "loss_weights": LOSS_WEIGHTS, "training_arguments": vars(args),
        "data": [{"path": str(path.relative_to(ROOT)), "sha256": sha256(path)} for path in data_paths],
        "torch_rng_state": torch.get_rng_state(), "cuda_rng_state_all": torch.cuda.get_rng_state_all(),
        "numpy_rng_state": np.random.get_state(), "python_rng_state": random.getstate(),
        "parameter_count": count_parameters(model.module if isinstance(model, DDP) else model),
        **extra,
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def append_curves(path: Path, rows: list[dict[str, Any]]) -> None:
    keys = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def initialize_model(device: torch.device, rank: int, args: argparse.Namespace, init_path: Path | None = None) -> DDP:
    seed_everything(args.seed, 0)
    model = CCSAM(CCSAMConfig()).to(device)
    if init_path is not None:
        checkpoint = torch.load(init_path, map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint["model_state"], strict=True)
    model = DDP(model, device_ids=[rank], output_device=rank, broadcast_buffers=False, find_unused_parameters=False)
    return model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("round0", "round1"), default="round0")
    parser.add_argument("--data", type=Path, default=OUT / "dense_dataset/round0_train30.npz")
    parser.add_argument("--data2", type=Path)
    parser.add_argument("--init", type=Path)
    parser.add_argument("--selection-epochs", type=int, default=12)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=26)
    args = parser.parse_args()
    args.data = args.data.resolve()
    args.data2 = args.data2.resolve() if args.data2 is not None else None
    args.init = args.init.resolve() if args.init is not None else None

    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ["LOCAL_RANK"])
    if world_size != 4:
        raise RuntimeError(f"N26 full training requires exactly four DDP processes, got {world_size}")
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("bfloat16 is not supported on the selected GPU")
    seed_everything(args.seed, rank)
    CHECKPOINTS.mkdir(parents=True, exist_ok=True)
    started_all = time.time()
    curves: list[dict[str, Any]] = []

    base = DenseDataset(args.data, weight_scale=1.0 if args.stage == "round0" else (0.5 if args.data2 else 1.0))
    data_paths = [args.data]
    if args.stage == "round0":
        summary = json.loads(args.data.with_name("round0_train30_summary.json").read_text(encoding="utf-8"))
        name_to_id = {name: index for index, name in enumerate(summary["sequence_names"])}
        validation_ids = {name_to_id[name] for name in SELECTION_VALIDATION}
        train_indices = np.flatnonzero(~np.isin(base.arrays["sequence"], list(validation_ids))).tolist()
        validation_indices = np.flatnonzero(np.isin(base.arrays["sequence"], list(validation_ids))).tolist()
        train_dataset, validation_dataset = Subset(base, train_indices), Subset(base, validation_indices)
        model = initialize_model(device, local_rank, args)
        optimizer = optimizer_for(model, args)
        best_key = (math.inf, math.inf, math.inf)
        best_epoch = 1
        for epoch in range(1, args.selection_epochs + 1):
            train_result = train_epoch(model, train_dataset, optimizer, device, epoch, args, rank, world_size)
            dist.barrier()
            validation_result: dict[str, float] = {}
            if rank == 0:
                validation_result = evaluate(model.module, validation_dataset, device, args.batch_size, args.workers)
                key = (validation_result["val_total_loss"], -validation_result["val_h5_top1"], epoch)
                if key < best_key:
                    best_key, best_epoch = key, epoch
                    save_checkpoint(CHECKPOINTS / "n26_round0_selection_best.pt", model, optimizer, stage="round0_selection", epoch=epoch, args=args, data_paths=data_paths, extra={"validation_sequences": SELECTION_VALIDATION, "selection_key": key})
                row = {"stage": "round0_selection", "epoch": epoch, **train_result, **validation_result, "best_epoch_so_far": best_epoch, "gpu_hours_cumulative": (time.time() - started_all) * world_size / 3600.0}
                curves.append(row)
                append_curves(OUT / "training_curves_round0.csv", curves)
                print(json.dumps(row, sort_keys=True), flush=True)
            holder = [best_epoch]
            dist.broadcast_object_list(holder, src=0)
            best_epoch = int(holder[0])
            dist.barrier()

        del model, optimizer
        torch.cuda.empty_cache()
        dist.barrier()
        model = initialize_model(device, local_rank, args)
        optimizer = optimizer_for(model, args)
        full_dataset: Dataset = base
        for epoch in range(1, best_epoch + 1):
            train_result = train_epoch(model, full_dataset, optimizer, device, 1000 + epoch, args, rank, world_size)
            if rank == 0:
                row = {"stage": "round0_full_fit", "epoch": epoch, "selected_epoch_count": best_epoch, **train_result, "gpu_hours_cumulative": (time.time() - started_all) * world_size / 3600.0}
                curves.append(row)
                save_checkpoint(CHECKPOINTS / "n26_round0_last.pt", model, optimizer, stage="round0_full_fit", epoch=epoch, args=args, data_paths=data_paths, extra={"selected_epoch_count": best_epoch, "selection_validation_sequences": SELECTION_VALIDATION})
                append_curves(OUT / "training_curves_round0.csv", curves)
                print(json.dumps(row, sort_keys=True), flush=True)
            dist.barrier()
        if rank == 0:
            final_path = CHECKPOINTS / "n26_round0_final.pt"
            save_checkpoint(final_path, model, optimizer, stage="round0_complete", epoch=best_epoch, args=args, data_paths=data_paths, extra={"selected_epoch_count": best_epoch, "selection_validation_sequences": SELECTION_VALIDATION, "gpu_hours": (time.time() - started_all) * world_size / 3600.0})
            print(f"N26_ROUND0_DDP_COMPLETE checkpoint={final_path} selected_epochs={best_epoch}", flush=True)
    else:
        if args.data2 is None or args.init is None:
            raise ValueError("round1 requires --data2 and --init")
        second = DenseDataset(args.data2, weight_scale=0.5)
        dataset = ConcatDataset([base, second])
        data_paths.append(args.data2)
        model = initialize_model(device, local_rank, args, args.init)
        optimizer = optimizer_for(model, args)
        for epoch in range(1, args.epochs + 1):
            train_result = train_epoch(model, dataset, optimizer, device, 2000 + epoch, args, rank, world_size)
            if rank == 0:
                row = {"stage": "round1_aggregate_refit", "epoch": epoch, **train_result, "gpu_hours_cumulative": (time.time() - started_all) * world_size / 3600.0}
                curves.append(row)
                save_checkpoint(CHECKPOINTS / "n26_round1_last.pt", model, optimizer, stage="round1_aggregate_refit", epoch=epoch, args=args, data_paths=data_paths, extra={"initial_checkpoint": str(args.init.relative_to(ROOT)), "gpu_hours": (time.time() - started_all) * world_size / 3600.0})
                append_curves(OUT / "training_curves_round1.csv", curves)
                print(json.dumps(row, sort_keys=True), flush=True)
            dist.barrier()
        if rank == 0:
            final_path = CHECKPOINTS / "n26_round1_final.pt"
            save_checkpoint(final_path, model, optimizer, stage="round1_complete", epoch=args.epochs, args=args, data_paths=data_paths, extra={"initial_checkpoint": str(args.init.relative_to(ROOT)), "gpu_hours": (time.time() - started_all) * world_size / 3600.0})
            print(f"N26_ROUND1_DDP_COMPLETE checkpoint={final_path} epochs={args.epochs}", flush=True)
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
