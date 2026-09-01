#!/usr/bin/env python3
"""Four-GPU AMP training for the N27 APCR-S residual.

Only the bounded residual parameters are optimized.  B10 scores and all
identity embeddings were frozen upstream.  The validation selector is
lexicographic: preserve the worst group first, then reward correction
response, then reduce loss.  No cal10 threshold is read or selected here.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
from torch import distributed as torch_dist
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler

from n27_apcr_model import APCRConfig, APCRS, counterfactual_tensors, feature_tensors


ROOT = Path(".")
OUT = ROOT / "outputs/n27"
SEED = 27
NONE_INDEX = 5

MODEL_FIELDS = [
    "candidate_mask", "target", "b10_score", "positive_similarity", "negative_similarity", "hard_similarity",
    "detector_score", "candidate_count", "has_positive", "has_negative", "has_hard", "positive_count", "correction_event",
    "negative_count", "hard_count", "positive_age", "negative_age", "hard_age", "pair_valid", "rejected_index",
    "cf_b10_score", "cf_positive_similarity", "cf_negative_similarity", "cf_has_positive", "cf_has_negative",
    "cf_positive_count", "cf_negative_count", "cf_positive_age", "cf_negative_age", "dataset", "sequence",
    "fold", "frame", "identity",
]


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def init_ddp() -> tuple[int, int, int, torch.device]:
    if "RANK" not in os.environ:
        return 0, 1, 0, torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    rank = int(os.environ["RANK"])
    world = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    backend = "nccl" if torch.cuda.is_available() else "gloo"
    dist.init_process_group(backend=backend)
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    return rank, world, local_rank, torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")


def finish_ddp() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


class ArrayDataset(Dataset):
    def __init__(self, paths: list[Path], indices: np.ndarray | None = None) -> None:
        loaded: dict[str, list[np.ndarray]] = {key: [] for key in MODEL_FIELDS}
        for path in paths:
            with np.load(path, allow_pickle=False) as payload:
                missing = [key for key in MODEL_FIELDS if key not in payload.files]
                if missing:
                    raise RuntimeError(f"{path} missing fields {missing}")
                for key in MODEL_FIELDS:
                    loaded[key].append(payload[key].copy())
        self.arrays = {key: np.concatenate(value, axis=0) for key, value in loaded.items()}
        if indices is not None:
            self.arrays = {key: value[indices] for key, value in self.arrays.items()}
        self.size = len(self.arrays["target"])
        self.tensors = {key: torch.from_numpy(value) for key, value in self.arrays.items()}

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {key: value[index] for key, value in self.tensors.items()}


def concat_arrays(paths: list[Path]) -> dict[str, np.ndarray]:
    result: dict[str, list[np.ndarray]] = {key: [] for key in MODEL_FIELDS}
    for path in paths:
        with np.load(path, allow_pickle=False) as payload:
            for key in MODEL_FIELDS:
                result[key].append(payload[key].copy())
    return {key: np.concatenate(value, axis=0) for key, value in result.items()}


def parse_fold_filter(value: str) -> np.ndarray | None:
    if not value:
        return None
    return np.asarray([int(item) for item in value.split(",") if item.strip()], dtype=np.int8)


def to_device(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device, non_blocking=True) for key, value in batch.items()}


def group_weights(dataset: np.ndarray) -> np.ndarray:
    unique, counts = np.unique(dataset, return_counts=True)
    weights = np.ones(len(dataset), dtype=np.float32)
    for group, count in zip(unique, counts):
        weights[dataset == group] = len(unique) / max(1, len(dataset)) / max(1, count / len(dataset))
    weights /= max(float(weights.mean()), 1e-6)
    return weights


def weighted_mean(values: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    return (values * weights).sum() / weights.sum().clamp_min(1e-6)


def loss_terms(model: torch.nn.Module, batch: dict[str, torch.Tensor], device: torch.device, mode: str) -> tuple[torch.Tensor, dict[str, float]]:
    features = feature_tensors(batch)
    output = model(features, mode=mode)
    scores = output["scores"]
    target = batch["target"].long()
    valid_event = target < NONE_INDEX
    sample_weights = torch.ones_like(target, dtype=torch.float32)
    if "group_weight" in batch:
        sample_weights = batch["group_weight"].float()
    if valid_event.any():
        logits = scores[valid_event]
        y = target[valid_event]
        ce = weighted_mean(F.cross_entropy(logits, y, reduction="none"), sample_weights[valid_event])
    else:
        ce = scores.sum() * 0.0

    anchor_correct = valid_event & (batch["b10_score"].argmax(dim=1) == target)
    margins: list[torch.Tensor] = []
    if anchor_correct.any():
        selected = scores.gather(1, target.clamp(0, 4).unsqueeze(1)).squeeze(1)
        wrong_mask = batch["candidate_mask"].bool().clone()
        wrong_mask.scatter_(1, target.clamp(0, 4).unsqueeze(1), False)
        wrong_max = scores.masked_fill(~wrong_mask, -1e4).max(dim=1).values
        has_wrong = wrong_mask.any(dim=1)
        anchor_rows = anchor_correct & has_wrong
        if anchor_rows.any():
            margins.append(weighted_mean(F.relu(wrong_max[anchor_rows] - selected[anchor_rows] + 0.005), sample_weights[anchor_rows]))
    anchor = margins[0] if margins else scores.sum() * 0.0

    teacher = scores.sum() * 0.0
    teacher_rows = anchor_correct & batch["candidate_mask"].any(dim=1)
    if teacher_rows.any():
        base_model = model.module if isinstance(model, DDP) else model
        temperature = base_model.config.temperature
        teacher_b10 = batch["b10_score"][teacher_rows].masked_fill(~batch["candidate_mask"][teacher_rows].bool(), -1e4)
        teacher_distribution = F.softmax(teacher_b10 / temperature, dim=1)
        teacher = weighted_mean(
            F.kl_div(F.log_softmax(scores[teacher_rows] / temperature, dim=1), teacher_distribution, reduction="none").sum(dim=1),
            sample_weights[teacher_rows],
        )

    response = scores.sum() * 0.0
    pair = batch["pair_valid"].bool() & valid_event
    rejected = batch["rejected_index"].long().clamp(0, 4)
    pair &= rejected != target.clamp(0, 4)
    pair &= batch["candidate_mask"].gather(1, rejected.unsqueeze(1)).squeeze(1)
    if pair.any():
        cf_output = model(counterfactual_tensors(batch), mode=mode)
        current_gap = scores.gather(1, target.clamp(0, 4).unsqueeze(1)).squeeze(1) - scores.gather(1, rejected.unsqueeze(1)).squeeze(1)
        cf_gap = cf_output["scores"].gather(1, target.clamp(0, 4).unsqueeze(1)).squeeze(1) - cf_output["scores"].gather(1, rejected.unsqueeze(1)).squeeze(1)
        response = weighted_mean(F.relu(0.001 - (current_gap[pair] - cf_gap[pair])), sample_weights[pair])

    total = ce + 0.50 * anchor + 0.03 * teacher + 0.20 * response
    return total, {"loss": float(total.detach().item()), "ce": float(ce.detach().item()), "anchor": float(anchor.detach().item()), "teacher": float(teacher.detach().item()), "response": float(response.detach().item()), "valid": float(valid_event.float().mean().item())}


@torch.no_grad()
def evaluate(model: APCRS, arrays: dict[str, np.ndarray], device: torch.device, mode: str, batch_size: int = 2048) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    model.eval()
    n = len(arrays["target"])
    scores = np.full((n, 5), -1e4, dtype=np.float32)
    cf_scores = np.full((n, 5), -1e4, dtype=np.float32)
    for start in range(0, n, batch_size):
        end = min(n, start + batch_size)
        batch = {key: torch.from_numpy(value[start:end]).to(device) for key, value in arrays.items() if key in MODEL_FIELDS}
        output = model(feature_tensors(batch), mode=mode)
        scores[start:end] = output["scores"].float().cpu().numpy()
        cf = model(counterfactual_tensors(batch), mode=mode)
        cf_scores[start:end] = cf["scores"].float().cpu().numpy()
    mask = arrays["candidate_mask"].astype(bool)
    target = arrays["target"].astype(np.int64)
    present = target < NONE_INDEX
    selected = scores.argmax(axis=1)
    b10_selected = arrays["b10_score"].argmax(axis=1)
    b10_correct = b10_selected == target
    correct = selected == target
    dataset_rows: list[dict[str, Any]] = []
    for dataset in sorted(np.unique(arrays["dataset"]).tolist()):
        population = present & (arrays["dataset"] == dataset)
        if not population.any():
            continue
        dataset_rows.append({
            "dataset": int(dataset), "events": int(population.sum()), "b10_top1": float(b10_correct[population].mean()),
            "apcr_top1": float(correct[population].mean()), "anchor_drop": float(b10_correct[population].mean() - correct[population].mean()),
            "corrections": int(arrays["correction_event"][population].sum()),
        })
    sequence_rows: list[dict[str, Any]] = []
    for sequence in sorted(np.unique(arrays["sequence"]).tolist()):
        population = present & (arrays["sequence"] == sequence)
        if not population.any():
            continue
        sequence_rows.append({
            "sequence": int(sequence), "events": int(population.sum()), "b10_top1": float(b10_correct[population].mean()),
            "apcr_top1": float(correct[population].mean()), "anchor_drop": float(b10_correct[population].mean() - correct[population].mean()),
        })
    pair = arrays["pair_valid"].astype(bool) & present
    rejected = arrays["rejected_index"].astype(np.int64)
    pair &= rejected >= 0
    pair &= rejected < 5
    pair &= rejected != target
    current_prob = np.zeros(n, dtype=np.float32)
    cf_prob = np.zeros(n, dtype=np.float32)
    if pair.any():
        current_softmax = np.exp(scores - np.max(scores, axis=1, keepdims=True)) * mask
        current_softmax /= np.maximum(current_softmax.sum(axis=1, keepdims=True), 1e-9)
        cf_softmax = np.exp(cf_scores - np.max(cf_scores, axis=1, keepdims=True)) * mask
        cf_softmax /= np.maximum(cf_softmax.sum(axis=1, keepdims=True), 1e-9)
        current_prob[pair] = current_softmax[pair, target[pair]]
        cf_prob[pair] = cf_softmax[pair, target[pair]]
        rejected_current = selected[pair] == rejected[pair]
        rejected_cf = cf_scores[pair].argmax(axis=1) == rejected[pair]
        rejected_select_delta = float(rejected_current.mean() - rejected_cf.mean())
        target_prob_gain = float((current_prob[pair] - cf_prob[pair]).mean())
    else:
        rejected_select_delta, target_prob_gain = math.nan, math.nan
    group_drops = [row["anchor_drop"] for row in dataset_rows + sequence_rows]
    metrics = {
        "events": n, "present_events": int(present.sum()), "none_events": int((~present).sum()),
        "b10_top1": float(b10_correct[present].mean()) if present.any() else math.nan,
        "apcr_top1": float(correct[present].mean()) if present.any() else math.nan,
        "anchor_drop": float(b10_correct[present].mean() - correct[present].mean()) if present.any() else math.nan,
        "worst_group_anchor_drop": float(max(group_drops)) if group_drops else math.nan,
        "worst_group_apcr_top1": float(min(row["apcr_top1"] for row in dataset_rows)) if dataset_rows else math.nan,
        "correction_response_pairs": int(pair.sum()),
        "correction_target_probability_gain": target_prob_gain,
        "rejected_identity_selection_delta": rejected_select_delta,
        "residual_abs_max": float(np.max(np.abs(scores - arrays["b10_score"])[mask])) if mask.any() else 0.0,
        "residual_positive_max": float(np.max((scores - arrays["b10_score"])[mask])) if mask.any() else 0.0,
        "residual_negative_min": float(np.min((scores - arrays["b10_score"])[mask])) if mask.any() else 0.0,
    }
    model.train()
    return metrics, dataset_rows, sequence_rows


def save_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fields = sorted({key for row in rows for key in row})
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def train(args: argparse.Namespace) -> None:
    rank, world, local_rank, device = init_ddp()
    seed_everything(SEED + rank)
    if rank == 0:
        print(json.dumps({"stage": args.stage, "world": world, "device": str(device), "train": args.train, "val": args.val}, sort_keys=True), flush=True)
    train_paths = [Path(item) for item in args.train.split(",")]
    val_paths = [Path(item) for item in args.val.split(",")]
    train_fold_filter = parse_fold_filter(args.train_folds)
    val_fold_filter = parse_fold_filter(args.val_folds)
    train_indices = None
    if train_fold_filter is not None:
        train_probe = concat_arrays(train_paths)
        train_indices = np.flatnonzero(np.isin(train_probe["fold"], train_fold_filter))
    train_dataset = ArrayDataset(train_paths, indices=train_indices)
    val_arrays = concat_arrays(val_paths)
    if val_fold_filter is not None:
        val_indices = np.flatnonzero(np.isin(val_arrays["fold"], val_fold_filter))
        val_arrays = {key: value[val_indices] for key, value in val_arrays.items()}
    train_weights = group_weights(train_dataset.arrays["dataset"])
    train_dataset.tensors["group_weight"] = torch.from_numpy(train_weights)
    sampler = DistributedSampler(train_dataset, num_replicas=world, rank=rank, shuffle=True, seed=SEED) if world > 1 else None
    loader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=sampler, shuffle=sampler is None, num_workers=0, pin_memory=device.type == "cuda")

    config = APCRConfig(hidden=args.hidden, positive_bound=args.positive_bound, negative_bound=args.negative_bound, hard_bound=args.hard_bound)
    model = APCRS(config).to(device)
    if args.init:
        checkpoint = torch.load(args.init, map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint["model_state"], strict=True)
    # APCR-S deliberately leaves the separately audited ordinary-hard control
    # branch unused during the main both-channel fit.  Let DDP account for
    # those parameters while keeping the branch in the checkpoint for the
    # predeclared ablation.
    wrapped: torch.nn.Module = DDP(model, device_ids=[local_rank], find_unused_parameters=True) if world > 1 and device.type == "cuda" else model
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda" and args.amp_dtype == "float16")
    best_key: tuple[Any, ...] | None = None
    best_metrics: dict[str, Any] | None = None
    curve_rows: list[dict[str, Any]] = []
    dataset_rows_all: list[dict[str, Any]] = []
    sequence_rows_all: list[dict[str, Any]] = []
    start_time = time.monotonic()
    for epoch in range(1, args.epochs + 1):
        if sampler is not None:
            sampler.set_epoch(epoch)
        wrapped.train()
        epoch_values: list[dict[str, float]] = []
        for batch in loader:
            batch = to_device(batch, device)
            optimizer.zero_grad(set_to_none=True)
            dtype = torch.bfloat16 if args.amp_dtype == "bfloat16" else torch.float16
            with torch.autocast(device_type=device.type, dtype=dtype, enabled=device.type == "cuda"):
                loss, values = loss_terms(wrapped, batch, device, args.mode)
            if scaler.is_enabled():
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            epoch_values.append(values)
        if world > 1:
            dist.barrier()
        if rank == 0:
            metric, dataset_rows, sequence_rows = evaluate(model, val_arrays, device, args.mode)
            mean_loss = {key: float(np.mean([row[key] for row in epoch_values])) for key in epoch_values[0]} if epoch_values else {}
            row = {"stage": args.stage, "epoch": epoch, "elapsed_seconds": time.monotonic() - start_time, **mean_loss, **{f"val_{key}": value for key, value in metric.items()}}
            curve_rows.append(row)
            dataset_rows_all = [{"stage": args.stage, "epoch": epoch, **item} for item in dataset_rows]
            sequence_rows_all = [{"stage": args.stage, "epoch": epoch, **item} for item in sequence_rows]
            max_drop = float(metric["worst_group_anchor_drop"])
            feasible = max_drop <= 0.01 + 1e-8
            response = metric["correction_target_probability_gain"]
            response_key = -float(response) if math.isfinite(response) else math.inf
            key = (0 if feasible else 1, max_drop, response_key, -float(metric["apcr_top1"]))
            if not args.finalize and (best_key is None or key < best_key):
                best_key = key
                best_metrics = {"stage": args.stage, "epoch": epoch, "selection_key": list(key), "validation": metric, "config": vars(args)}
                checkpoint_path = OUT / "checkpoints" / f"apcr_s_{args.stage}_best.pt"
                checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save({"stage": args.stage, "epoch": epoch, "model_config": vars(config), "model_state": model.state_dict(), "optimizer_state": optimizer.state_dict(), "metrics": best_metrics}, checkpoint_path)
            print(json.dumps(row, sort_keys=True), flush=True)
        if world > 1:
            dist.barrier()
    if rank == 0:
        save_csv(OUT / f"training_curves_{args.stage}.csv", curve_rows)
        save_csv(OUT / f"per_dataset_{args.stage}.csv", dataset_rows_all)
        save_csv(OUT / f"per_sequence_{args.stage}.csv", sequence_rows_all)
        # Preserve the complete multi-stage record in the canonical files.
        def existing_rows(path: Path) -> list[dict[str, Any]]:
            if not path.is_file():
                return []
            with path.open(encoding="utf-8", newline="") as handle:
                return list(csv.DictReader(handle))
        save_csv(OUT / "training_curves.csv", existing_rows(OUT / "training_curves.csv") + curve_rows)
        save_csv(OUT / "per_dataset.csv", existing_rows(OUT / "per_dataset.csv") + dataset_rows_all)
        save_csv(OUT / "per_sequence.csv", existing_rows(OUT / "per_sequence.csv") + sequence_rows_all)
        if args.finalize:
            final_path = OUT / "checkpoints" / f"apcr_s_{args.stage}_best.pt"
            final_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"stage": args.stage, "epoch": args.epochs, "model_config": vars(config), "model_state": model.state_dict(), "optimizer_state": optimizer.state_dict(), "metrics": {"stage": args.stage, "epoch": args.epochs, "selection": "fixed_epoch_from_sequence_disjoint_cv", "validation_last": curve_rows[-1] if curve_rows else {}, "config": vars(args)}}, final_path)
        summary = {"stage": args.stage, "epochs": args.epochs, "world_size": world, "gpu_hours": (time.monotonic() - start_time) * world / 3600.0, "best": best_metrics, "train_parents": len(train_dataset), "validation_parents": len(val_arrays["target"]), "val25_read": False}
        path = OUT / f"{args.stage}_training_summary.json"
        path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    finish_ddp()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("p1", "p2_cv", "p2"), required=True)
    parser.add_argument("--train", required=True, help="comma-separated NPZ paths")
    parser.add_argument("--val", required=True, help="comma-separated NPZ paths")
    parser.add_argument("--init", default="")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--hidden", type=int, default=24)
    parser.add_argument("--positive-bound", type=float, default=0.03)
    parser.add_argument("--negative-bound", type=float, default=0.03)
    parser.add_argument("--hard-bound", type=float, default=0.03)
    parser.add_argument("--mode", choices=("both", "positive_only", "negative_only", "hard_negative"), default="both")
    parser.add_argument("--amp-dtype", choices=("bfloat16", "float16"), default="bfloat16")
    parser.add_argument("--train-folds", default="", help="optional comma-separated sequence folds")
    parser.add_argument("--val-folds", default="", help="optional comma-separated sequence folds")
    parser.add_argument("--finalize", action="store_true", help="train a fixed number of epochs after CV selection and save the final state")
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
