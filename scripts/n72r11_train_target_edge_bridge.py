#!/usr/bin/env python3
"""Train the isolated N72R11 target-column residual bridge.

The scorer checkpoint is frozen while this head is trained.  The bridge gets
only sealed causal features and offline candidate labels.  Its output is a
residual on the explicit target public column; the exact solver, every other
public column, and the NONE score remain unchanged.
"""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import tempfile
import traceback
from typing import Any, Mapping

import numpy as np
import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in __import__("sys").path:
    __import__("sys").path.insert(0, str(ROOT))

from sam3_intermot.association.target_edge_bridge import (  # noqa: E402
    BRIDGE_INPUT_DIM,
    TargetEdgeBridge,
    fit_residual_scale,
)
from scripts.n72r11_train_v3 import (  # noqa: E402
    CORPUS_ROOT,
    DEFAULT_BATCH_SIZE,
    device_from,
    load_checkpoint,
    load_split,
    model_config,
    now_utc,
    read_json,
    sha256_file,
    set_seed,
)


OUTPUT_ROOT = ROOT / "outputs/N72R11/training_v3"
STAGE_12 = ROOT / "outputs/N72R11/stage_12_bridge_smoke.json"
STAGE_13 = ROOT / "outputs/N72R11/stage_13_bridge_training.json"
BRIDGE_CHECKPOINT = OUTPUT_ROOT / "target_edge_bridge.pt"
BRIDGE_SEED = 7212
BRIDGE_EPOCHS = 3
BRIDGE_LR = 1.0e-3
BRIDGE_WEIGHT_DECAY = 1.0e-4
PROTECTED_WEIGHT = 0.25
POSITIVE_COMPETITION_WEIGHT = 0.25


def atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        directory_fd = os.open(path.parent, os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    atomic_write(path, json.dumps(dict(value), ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n")


def atomic_torch(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    os.close(fd)
    try:
        torch.save(dict(value), temporary)
        with open(temporary, "rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        directory_fd = os.open(path.parent, os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def bridge_features(arrays: Mapping[str, np.ndarray], logits: np.ndarray) -> np.ndarray:
    count, candidates = arrays["candidate_mask"].shape
    if logits.shape != (count, candidates + 1) or not np.isfinite(logits).all():
        raise RuntimeError(f"invalid frozen scorer logits: {logits.shape}")
    source = np.asarray(arrays["source_features"], dtype=np.float32)
    feature = np.concatenate(
        [
            (logits[:, :candidates] - logits[:, candidates, None])[..., None],
            np.asarray(arrays["legacy_target_scores"], dtype=np.float32)[..., None],
            np.asarray(arrays["legacy_best_other_scores"], dtype=np.float32)[..., None],
            (np.asarray(arrays["legacy_target_scores"], dtype=np.float32) - np.asarray(arrays["legacy_best_other_scores"], dtype=np.float32))[..., None],
            np.asarray(arrays["incumbent_target"], dtype=np.float32)[..., None],
            np.asarray(arrays["incumbent_other"], dtype=np.float32)[..., None],
            np.asarray(arrays["confidence"], dtype=np.float32)[..., None],
            np.asarray(arrays["presence"], dtype=np.float32)[..., None],
            np.asarray(arrays["motion_iou"], dtype=np.float32)[..., None],
            source,
        ],
        axis=-1,
    )
    if feature.shape != (count, candidates, BRIDGE_INPUT_DIM) or not np.isfinite(feature).all():
        raise RuntimeError(f"invalid target-edge feature tensor: {feature.shape}")
    return feature.astype(np.float32)


def frozen_scorer_logits(checkpoint: Path, arrays: Mapping[str, np.ndarray], device: torch.device, batch_size: int) -> np.ndarray:
    model, _ = load_checkpoint(checkpoint, device)
    model.eval()
    output: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, int(arrays["labels"].shape[0]), int(batch_size)):
            indices = np.arange(start, min(start + int(batch_size), int(arrays["labels"].shape[0])), dtype=np.int64)
            # Importing here avoids a second copy of the tensor conversion
            # helpers in this bridge-only script.
            from scripts.n72r11_train_v3 import model_forward

            logits = model_forward(model, arrays, indices, device).detach().float().cpu().numpy()
            output.append(logits)
    return np.concatenate(output, axis=0)


def bridge_loss(
    bridge: TargetEdgeBridge,
    features: torch.Tensor,
    legacy_target: torch.Tensor,
    labels: torch.Tensor,
    candidate_mask: torch.Tensor,
    protected_mask: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    delta, calibrated = bridge(features, legacy_target)
    count = int(candidate_mask.shape[1])
    safe_calibrated = calibrated.masked_fill(~candidate_mask, torch.finfo(calibrated.dtype).min)
    none = torch.zeros((calibrated.shape[0], 1), dtype=calibrated.dtype, device=calibrated.device)
    scores = torch.cat([safe_calibrated, none], dim=1)
    ce = F.cross_entropy(scores, labels)
    protected_values = delta.masked_fill(~protected_mask, 0.0)
    protected_denominator = protected_mask.to(delta.dtype).sum().clamp_min(1.0)
    protected_penalty = (torch.relu(protected_values) ** 2).sum() / protected_denominator
    positive = labels < count
    competition = delta.new_zeros(())
    if bool(positive.any()):
        positive_labels = labels[positive]
        positive_delta = delta[positive]
        positive_mask = candidate_mask[positive].clone()
        positive_mask.scatter_(1, positive_labels.unsqueeze(1), False)
        valid = positive_mask.any(dim=1)
        if bool(valid.any()):
            best_other = positive_delta.masked_fill(~positive_mask, torch.finfo(delta.dtype).min).max(dim=1).values
            target_delta = positive_delta[torch.arange(len(positive_labels), device=delta.device), positive_labels]
            competition = F.relu(0.20 - target_delta[valid] + best_other[valid]).mean()
    total = ce + PROTECTED_WEIGHT * protected_penalty + POSITIVE_COMPETITION_WEIGHT * competition
    return total, {"cross_entropy": ce.detach(), "protected_penalty": protected_penalty.detach(), "positive_competition": competition.detach()}


def evaluate_bridge(bridge: TargetEdgeBridge, features: np.ndarray, arrays: Mapping[str, np.ndarray], device: torch.device, batch_size: int) -> dict[str, float]:
    bridge.eval()
    total = 0.0
    correct = 0
    n = int(arrays["labels"].shape[0])
    with torch.no_grad():
        for start in range(0, n, int(batch_size)):
            indices = np.arange(start, min(start + int(batch_size), n), dtype=np.int64)
            feat = torch.as_tensor(features[indices], dtype=torch.float32, device=device)
            target = torch.as_tensor(arrays["legacy_target_scores"][indices], dtype=torch.float32, device=device)
            mask = torch.as_tensor(arrays["candidate_mask"][indices], dtype=torch.bool, device=device)
            labels = torch.as_tensor(arrays["labels"][indices], dtype=torch.long, device=device)
            protected = torch.as_tensor(arrays["protected_candidate_mask"][indices], dtype=torch.bool, device=device)
            loss, _ = bridge_loss(bridge, feat, target, labels, mask, protected)
            delta, calibrated = bridge(feat, target)
            scores = torch.cat([calibrated.masked_fill(~mask, torch.finfo(calibrated.dtype).min), torch.zeros((len(indices), 1), device=device)], dim=1)
            total += float(loss.detach().cpu()) * len(indices)
            correct += int((scores.argmax(dim=1) == labels).sum().detach().cpu())
    return {"loss": total / max(n, 1), "accuracy": correct / max(n, 1), "examples": n}


def stage_base(stage: str) -> dict[str, Any]:
    return {
        "schema_version": "N72R11_STAGE_STATUS_V1",
        "stage": stage,
        "started_at_utc": now_utc(),
        "runtime_future_gt_used": False,
        "interaction_source": "simulated_from_gt",
        "real_human_evidence": False,
        "not_real_human_evidence": True,
        "production_authorized": False,
    }


def main() -> int:
    parser = __import__("argparse").ArgumentParser()
    parser.add_argument("--phase", choices=("smoke", "train"), required=True)
    parser.add_argument("--device", default="cuda:5")
    parser.add_argument("--scorer-checkpoint", type=Path, default=OUTPUT_ROOT / "v3_bootstrap.pt")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    args = parser.parse_args()
    set_seed(BRIDGE_SEED)
    device = device_from(str(args.device))
    stage_path = STAGE_12 if args.phase == "smoke" else STAGE_13
    status = stage_base(f"N72R11-{args.phase.upper()}-BRIDGE")
    try:
        train_arrays, train_metadata, train_summary = load_split("train")
        validation_arrays, validation_metadata, validation_summary = load_split("validation")
        train_logits = frozen_scorer_logits(Path(args.scorer_checkpoint), train_arrays, device, int(args.batch_size))
        validation_logits = frozen_scorer_logits(Path(args.scorer_checkpoint), validation_arrays, device, int(args.batch_size))
        train_features = bridge_features(train_arrays, train_logits)
        validation_features = bridge_features(validation_arrays, validation_logits)
        if args.phase == "smoke":
            scale = fit_residual_scale(
                train_arrays["legacy_target_scores"][train_arrays["candidate_mask"]],
                train_arrays["legacy_best_other_scores"][train_arrays["candidate_mask"]],
            )
            bridge = TargetEdgeBridge(residual_scale=scale).to(device)
            sample = min(4, len(train_features))
            feat = torch.as_tensor(train_features[:sample], dtype=torch.float32, device=device)
            target = torch.as_tensor(train_arrays["legacy_target_scores"][:sample], dtype=torch.float32, device=device)
            delta, calibrated = bridge(feat, target)
            if not torch.isfinite(delta).all() or not torch.isfinite(calibrated).all():
                raise RuntimeError("bridge smoke output is non-finite")
            status.update({
                "status": "PASS_N72R11_BRIDGE_SMOKE",
                "device": str(device),
                "input_dim": BRIDGE_INPUT_DIM,
                "residual_scale": float(scale),
                "sample_count": sample,
                "scorer_checkpoint": str(args.scorer_checkpoint),
                "scorer_checkpoint_sha256": sha256_file(Path(args.scorer_checkpoint)),
                "feature_shapes": {"train": list(train_features.shape), "validation": list(validation_features.shape)},
                "finished_at_utc": now_utc(),
            })
        else:
            target_flat = train_arrays["legacy_target_scores"][train_arrays["candidate_mask"]]
            other_flat = train_arrays["legacy_best_other_scores"][train_arrays["candidate_mask"]]
            scale = fit_residual_scale(target_flat, other_flat)
            bridge = TargetEdgeBridge(residual_scale=scale).to(device)
            optimizer = torch.optim.AdamW(bridge.parameters(), lr=BRIDGE_LR, weight_decay=BRIDGE_WEIGHT_DECAY)
            rng = np.random.default_rng(BRIDGE_SEED)
            history: list[dict[str, Any]] = []
            for epoch in range(BRIDGE_EPOCHS):
                bridge.train()
                order = np.arange(len(train_features), dtype=np.int64)
                rng.shuffle(order)
                running = 0.0
                seen = 0
                for start in range(0, len(order), int(args.batch_size)):
                    indices = order[start : start + int(args.batch_size)]
                    optimizer.zero_grad(set_to_none=True)
                    feat = torch.as_tensor(train_features[indices], dtype=torch.float32, device=device)
                    target = torch.as_tensor(train_arrays["legacy_target_scores"][indices], dtype=torch.float32, device=device)
                    labels = torch.as_tensor(train_arrays["labels"][indices], dtype=torch.long, device=device)
                    mask = torch.as_tensor(train_arrays["candidate_mask"][indices], dtype=torch.bool, device=device)
                    protected = torch.as_tensor(train_arrays["protected_candidate_mask"][indices], dtype=torch.bool, device=device)
                    loss, components = bridge_loss(bridge, feat, target, labels, mask, protected)
                    if not torch.isfinite(loss):
                        raise RuntimeError(f"non-finite bridge loss at epoch={epoch}")
                    loss.backward()
                    if not all(parameter.grad is None or bool(torch.isfinite(parameter.grad).all()) for parameter in bridge.parameters()):
                        raise RuntimeError(f"non-finite bridge gradient at epoch={epoch}")
                    torch.nn.utils.clip_grad_norm_(bridge.parameters(), 5.0)
                    optimizer.step()
                    running += float(loss.detach().cpu()) * len(indices)
                    seen += len(indices)
                validation = evaluate_bridge(bridge, validation_features, validation_arrays, device, int(args.batch_size))
                history.append({"epoch": epoch + 1, "train_loss": running / max(seen, 1), "validation": validation})
            checkpoint_payload = {
                "schema_version": "N72R11_TARGET_EDGE_BRIDGE_CHECKPOINT_V1",
                "state_dict": {key: value.detach().cpu() for key, value in bridge.state_dict().items()},
                "residual_scale": float(scale),
                "architecture": bridge.audit_metadata(),
                "training_config": {
                    "seed": BRIDGE_SEED,
                    "epochs": BRIDGE_EPOCHS,
                    "batch_size": int(args.batch_size),
                    "learning_rate": BRIDGE_LR,
                    "weight_decay": BRIDGE_WEIGHT_DECAY,
                    "protected_weight": PROTECTED_WEIGHT,
                    "positive_competition_weight": POSITIVE_COMPETITION_WEIGHT,
                    "checkpoint_selection": "single preregistered train-only bridge run",
                },
                "scorer_checkpoint": str(args.scorer_checkpoint),
                "scorer_checkpoint_sha256": sha256_file(Path(args.scorer_checkpoint)),
                "train_summary": train_summary,
                "validation_summary": validation_summary,
                "history": history,
                "runtime_future_gt_used": False,
                "interaction_source": "simulated_from_gt",
                "not_real_human_evidence": True,
                "production_authorized": False,
            }
            atomic_torch(BRIDGE_CHECKPOINT, checkpoint_payload)
            status.update({
                "status": "PASS_N72R11_BRIDGE_TRAINING",
                "device": str(device),
                "checkpoint": str(BRIDGE_CHECKPOINT),
                "checkpoint_sha256": sha256_file(BRIDGE_CHECKPOINT),
                "residual_scale": float(scale),
                "train_summary": train_summary,
                "validation_summary": validation_summary,
                "history": history,
                "finished_at_utc": now_utc(),
            })
        atomic_json(stage_path, status)
        print(json.dumps(status, sort_keys=True))
        return 0
    except Exception as exc:
        failure = OUTPUT_ROOT / "attempts" / f"bridge_{args.phase}_failure_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.json"
        payload = {**status, "status": f"FAIL_N72R11_BRIDGE_{str(args.phase).upper()}", "error_type": type(exc).__name__, "error": str(exc), "traceback": traceback.format_exc(), "finished_at_utc": now_utc(), "failure_artifact": str(failure)}
        atomic_json(failure, payload)
        atomic_json(stage_path, payload)
        print(json.dumps({"status": payload["status"], "failure_artifact": str(failure), "error": str(exc)}, sort_keys=True))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
