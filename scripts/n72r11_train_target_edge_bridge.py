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
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in __import__("sys").path:
    __import__("sys").path.insert(0, str(ROOT))

from sam3_intermot.association.target_edge_bridge import (  # noqa: E402
    BRIDGE_INPUT_DIM,
    TargetEdgeBridge,
    build_target_edge_feature_from_scalars,
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
import scripts.n72r11_train_v3 as train_v3_module  # noqa: E402


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
DELTA_L2_WEIGHT = 0.01


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


def bridge_features(
    arrays: Mapping[str, np.ndarray],
    logits: np.ndarray,
    metadata: Sequence[Mapping[str, Any]],
) -> np.ndarray:
    count, candidates = arrays["candidate_mask"].shape
    if len(metadata) != count or logits.shape != (count, candidates + 1) or not np.isfinite(logits).all():
        raise RuntimeError(f"invalid frozen scorer logits: {logits.shape}")
    feature = np.zeros((count, candidates, BRIDGE_INPUT_DIM), dtype=np.float32)
    target_scores = np.asarray(arrays["legacy_target_scores"], dtype=np.float32)
    other_scores = np.asarray(arrays["legacy_best_other_scores"], dtype=np.float32)
    incumbent_target = np.asarray(arrays["incumbent_target"], dtype=np.float32)
    incumbent_other = np.asarray(arrays["incumbent_other"], dtype=np.float32)
    confidence = np.asarray(arrays["confidence"], dtype=np.float32)
    presence = np.asarray(arrays["presence"], dtype=np.float32)
    motion = np.asarray(arrays["motion_iou"], dtype=np.float32)
    for row_index, row in enumerate(metadata):
        actual = len(row.get("candidate_uids", []))
        if actual > candidates:
            raise RuntimeError(f"bridge metadata candidate axis exceeds padded width at {row_index}")
        sources = list(row.get("candidate_sources", []))
        if len(sources) != actual:
            raise RuntimeError(f"bridge metadata source axis mismatch at {row_index}")
        for candidate_index in range(candidates):
            source = sources[candidate_index] if candidate_index < actual else "UNKNOWN"
            feature[row_index, candidate_index] = np.asarray(
                build_target_edge_feature_from_scalars(
                    candidate_logit=float(logits[row_index, candidate_index]),
                    none_logit=float(logits[row_index, candidates]),
                    legacy_target_score=float(target_scores[row_index, candidate_index]),
                    legacy_best_other_score=float(other_scores[row_index, candidate_index]),
                    incumbent_target=float(incumbent_target[row_index, candidate_index]),
                    incumbent_other=float(incumbent_other[row_index, candidate_index]),
                    confidence=float(confidence[row_index, candidate_index]),
                    presence=float(presence[row_index, candidate_index]),
                    motion_iou=float(motion[row_index, candidate_index]),
                    candidate_source=str(source),
                ),
                dtype=np.float32,
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
    legacy_best_other: torch.Tensor,
    labels: torch.Tensor,
    candidate_mask: torch.Tensor,
    protected_mask: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    delta, calibrated = bridge(features, legacy_target)
    if legacy_best_other.shape != legacy_target.shape or not torch.isfinite(legacy_best_other).all():
        raise ValueError("legacy best-other tensor must align with finite legacy target")
    count = int(candidate_mask.shape[1])
    safe_calibrated = calibrated.masked_fill(~candidate_mask, torch.finfo(calibrated.dtype).min)
    none = torch.zeros((calibrated.shape[0], 1), dtype=calibrated.dtype, device=calibrated.device)
    scores = torch.cat([safe_calibrated, none], dim=1)
    ce = F.cross_entropy(scores, labels)
    protected_values = calibrated - legacy_best_other
    protected_values = protected_values.masked_fill(~protected_mask, 0.0)
    protected_denominator = protected_mask.to(delta.dtype).sum().clamp_min(1.0)
    protected_penalty = (torch.relu(protected_values + 0.20) ** 2).sum() / protected_denominator
    positive = labels < count
    competition = delta.new_zeros(())
    if bool(positive.any()):
        positive_labels = labels[positive]
        positive_delta = delta[positive]
        positive_mask = candidate_mask[positive].clone()
        positive_mask.scatter_(1, positive_labels.unsqueeze(1), False)
        valid = positive_mask.any(dim=1)
        if bool(valid.any()):
            target_calibrated = calibrated[positive]
            target_value = target_calibrated[torch.arange(len(positive_labels), device=delta.device), positive_labels]
            target_legacy_other = legacy_best_other[positive][torch.arange(len(positive_labels), device=delta.device), positive_labels]
            competition = F.relu(0.20 - target_value[valid] + target_legacy_other[valid]).mean()
    delta_l2 = (delta.masked_fill(~candidate_mask, 0.0) ** 2).sum() / candidate_mask.to(delta.dtype).sum().clamp_min(1.0)
    total = ce + PROTECTED_WEIGHT * protected_penalty + POSITIVE_COMPETITION_WEIGHT * competition + DELTA_L2_WEIGHT * delta_l2
    return total, {"cross_entropy": ce.detach(), "protected_penalty": protected_penalty.detach(), "positive_competition": competition.detach(), "delta_l2": delta_l2.detach()}


def evaluate_bridge(bridge: TargetEdgeBridge, features: np.ndarray, arrays: Mapping[str, np.ndarray], device: torch.device, batch_size: int) -> dict[str, float]:
    bridge.eval()
    total = 0.0
    correct = 0
    legacy_correct = 0
    legacy_boundary_total = 0
    legacy_boundary_success = 0
    bridge_boundary_success = 0
    protected_total = 0
    protected_safe = 0
    delta_values: list[float] = []
    n = int(arrays["labels"].shape[0])
    with torch.no_grad():
        for start in range(0, n, int(batch_size)):
            indices = np.arange(start, min(start + int(batch_size), n), dtype=np.int64)
            feat = torch.as_tensor(features[indices], dtype=torch.float32, device=device)
            target = torch.as_tensor(arrays["legacy_target_scores"][indices], dtype=torch.float32, device=device)
            other = torch.as_tensor(arrays["legacy_best_other_scores"][indices], dtype=torch.float32, device=device)
            mask = torch.as_tensor(arrays["candidate_mask"][indices], dtype=torch.bool, device=device)
            labels = torch.as_tensor(arrays["labels"][indices], dtype=torch.long, device=device)
            protected = torch.as_tensor(arrays["protected_candidate_mask"][indices], dtype=torch.bool, device=device)
            loss, _ = bridge_loss(bridge, feat, target, other, labels, mask, protected)
            delta, calibrated = bridge(feat, target)
            scores = torch.cat([calibrated.masked_fill(~mask, torch.finfo(calibrated.dtype).min), torch.zeros((len(indices), 1), device=device)], dim=1)
            legacy_scores = torch.cat([target.masked_fill(~mask, torch.finfo(target.dtype).min), torch.zeros((len(indices), 1), device=device)], dim=1)
            total += float(loss.detach().cpu()) * len(indices)
            correct += int((scores.argmax(dim=1) == labels).sum().detach().cpu())
            legacy_correct += int((legacy_scores.argmax(dim=1) == labels).sum().detach().cpu())
            # The boundary diagnostics are target-vs-its-frozen best-other,
            # evaluated only for positive labels and never used for training.
            positive = labels < int(arrays["candidate_mask"].shape[1])
            if bool(positive.any()):
                rows = torch.arange(len(indices), device=device)[positive]
                cols = labels[positive]
                legacy_boundary_total += int(positive.sum().detach().cpu())
                legacy_boundary_success += int((target[rows, cols] > other[rows, cols]).sum().detach().cpu())
                bridge_boundary_success += int((calibrated[rows, cols] > other[rows, cols]).sum().detach().cpu())
            protected_total += int(protected.sum().detach().cpu())
            protected_safe += int(((calibrated <= other + 0.20) & protected).sum().detach().cpu())
            delta_values.extend(delta[mask].detach().float().cpu().tolist())
    return {
        "loss": total / max(n, 1), "accuracy": correct / max(n, 1), "examples": n,
        "legacy_accuracy": legacy_correct / max(n, 1),
        "bridge_accuracy": correct / max(n, 1),
        "legacy_positive_boundary_success": legacy_boundary_success / max(legacy_boundary_total, 1),
        "bridge_positive_boundary_success": bridge_boundary_success / max(legacy_boundary_total, 1),
        "protected_safe_rate": protected_safe / max(protected_total, 1),
        "mean_abs_delta": float(np.mean(np.abs(np.asarray(delta_values, dtype=np.float64)))) if delta_values else 0.0,
        "p95_abs_delta": float(np.percentile(np.abs(np.asarray(delta_values, dtype=np.float64)), 95.0)) if delta_values else 0.0,
    }


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
    global OUTPUT_ROOT, STAGE_12, STAGE_13, BRIDGE_CHECKPOINT

    parser = __import__("argparse").ArgumentParser()
    parser.add_argument("--phase", choices=("smoke", "train"), required=True)
    parser.add_argument("--device", default="cuda:5")
    parser.add_argument("--corpus-root", type=Path, default=CORPUS_ROOT)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--stage-dir", type=Path, default=ROOT / "outputs/N72R11")
    parser.add_argument("--resource-censored", action="store_true")
    parser.add_argument("--scorer-checkpoint", type=Path, default=None)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    args = parser.parse_args()
    corpus_root = args.corpus_root if args.corpus_root.is_absolute() else ROOT / args.corpus_root
    OUTPUT_ROOT = args.output_root if args.output_root.is_absolute() else ROOT / args.output_root
    stage_dir = args.stage_dir if args.stage_dir.is_absolute() else ROOT / args.stage_dir
    STAGE_12 = stage_dir / "stage_12_bridge_smoke.json"
    STAGE_13 = stage_dir / "stage_13_bridge_training.json"
    BRIDGE_CHECKPOINT = OUTPUT_ROOT / "target_edge_bridge.pt"
    scorer_checkpoint = args.scorer_checkpoint
    if scorer_checkpoint is None:
        scorer_checkpoint = OUTPUT_ROOT / "v3_bootstrap.pt"
    elif not scorer_checkpoint.is_absolute():
        scorer_checkpoint = ROOT / scorer_checkpoint
    train_v3_module.CORPUS_ROOT = corpus_root
    set_seed(BRIDGE_SEED)
    device = device_from(str(args.device))
    stage_path = STAGE_12 if args.phase == "smoke" else STAGE_13
    status = stage_base(f"N72R11-{args.phase.upper()}-BRIDGE")
    status["resource_censored_development"] = bool(args.resource_censored)
    try:
        corpus_manifest = read_json(corpus_root / "corpus_manifest.json")
        if bool(corpus_manifest.get("resource_censored_development", False)) != bool(args.resource_censored):
            raise RuntimeError(
                "resource-censored corpus requires --resource-censored and cannot use the default production path"
            )
        train_arrays, train_metadata, train_summary = load_split("train")
        validation_arrays, validation_metadata, validation_summary = load_split("validation")
        train_logits = frozen_scorer_logits(scorer_checkpoint, train_arrays, device, int(args.batch_size))
        validation_logits = frozen_scorer_logits(scorer_checkpoint, validation_arrays, device, int(args.batch_size))
        train_features = bridge_features(train_arrays, train_logits, train_metadata)
        validation_features = bridge_features(validation_arrays, validation_logits, validation_metadata)
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
                "resource_censored_development": bool(args.resource_censored),
                "device": str(device),
                "input_dim": BRIDGE_INPUT_DIM,
                "residual_scale": float(scale),
                "sample_count": sample,
                "scorer_checkpoint": str(scorer_checkpoint),
                "scorer_checkpoint_sha256": sha256_file(scorer_checkpoint),
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
            best_validation_loss = float("inf")
            best_state_dict: dict[str, torch.Tensor] | None = None
            best_epoch = None
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
                    other = torch.as_tensor(train_arrays["legacy_best_other_scores"][indices], dtype=torch.float32, device=device)
                    labels = torch.as_tensor(train_arrays["labels"][indices], dtype=torch.long, device=device)
                    mask = torch.as_tensor(train_arrays["candidate_mask"][indices], dtype=torch.bool, device=device)
                    protected = torch.as_tensor(train_arrays["protected_candidate_mask"][indices], dtype=torch.bool, device=device)
                    loss, components = bridge_loss(bridge, feat, target, other, labels, mask, protected)
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
                if float(validation["loss"]) < best_validation_loss:
                    best_validation_loss = float(validation["loss"])
                    best_epoch = int(epoch + 1)
                    best_state_dict = {key: value.detach().cpu().clone() for key, value in bridge.state_dict().items()}
            if best_state_dict is None or best_epoch is None:
                raise RuntimeError("bridge did not produce a validation checkpoint")
            bridge.load_state_dict(best_state_dict, strict=True)
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
                    "checkpoint_selection": "minimum fixed validation loss; no future effect metrics",
                    "selected_epoch": int(best_epoch),
                    "selected_validation_loss": float(best_validation_loss),
                    "delta_l2_weight": DELTA_L2_WEIGHT,
                },
                "scorer_checkpoint": str(scorer_checkpoint),
                "scorer_checkpoint_sha256": sha256_file(scorer_checkpoint),
                "train_summary": train_summary,
                "validation_summary": validation_summary,
                "history": history,
                "selected_epoch": int(best_epoch),
                "selected_validation_loss": float(best_validation_loss),
                "runtime_future_gt_used": False,
                "interaction_source": "simulated_from_gt",
                "not_real_human_evidence": True,
                "production_authorized": False,
            }
            atomic_torch(BRIDGE_CHECKPOINT, checkpoint_payload)
            status.update({
                "status": "PASS_N72R11_BRIDGE_TRAINING",
                "resource_censored_development": bool(args.resource_censored),
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
