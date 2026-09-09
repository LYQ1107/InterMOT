#!/usr/bin/env python3
"""Train, smoke-test, and audit the isolated N72R11R4 PCTIS candidate.

The script uses only the sealed exact-on-policy corpus and offline labels.  It
never changes candidate generation, the public solver, SAM3, or any historical
N72R* output.  ``self_rollout`` is a causal model-input audit; its labels are
read only for the reported offline readiness counters.
"""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
import traceback
from typing import Any, Mapping, Sequence

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in __import__("sys").path:
    __import__("sys").path.insert(0, str(ROOT))

from sam3_intermot.association.target_edge_interface import select_candidate_from_logits  # noqa: E402
from sam3_intermot.reacquisition.models.n72r11r4_public_competition_temporal import (  # noqa: E402
    PublicCompetitionTemporalIdentityModel,
    initialize_from_v3_checkpoint,
    load_pctis_checkpoint,
    pctis_model_config,
    public_competition_temporal_loss,
)
from sam3_intermot.reacquisition.temporal_state_policy import (  # noqa: E402
    build_temporal_features,
    initialize_temporal_state,
    state_audit,
    state_memory_arrays,
    update_temporal_state,
)
from scripts import n72r11_train_v3 as train_v3  # noqa: E402


SEED = 7211
BATCH_SIZE = 64
INITIAL_EPOCHS = 2
INITIAL_LR = 2.5e-4
FINETUNE_EPOCHS = 1
FINETUNE_LR = 1.25e-4
WEIGHT_DECAY = 1.0e-4
NONE_WEIGHT = 2.0
PAIRWISE_WEIGHT = 0.15
PAIRWISE_MARGIN = 0.20
POSITIVE_BOUNDARY_WEIGHT = 0.25
PROTECTED_BOUNDARY_WEIGHT = 0.25
DELTA_L2_WEIGHT = 0.01
PUBLIC_MARGIN = 0.20
TARGET_ACCURACY_MIN = 0.70
NONE_ACCURACY_MIN = 0.50
POSITIVE_BOUNDARY_MIN = 0.70
PROTECTED_SAFE_MIN = 0.90
FUTURE_COUNT_MIN = 50


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(dict(value), handle, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        directory_fd = os.open(path.parent, os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if temporary.exists():
            temporary.unlink()


def atomic_torch(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    os.close(fd)
    temporary = Path(temporary_name)
    try:
        torch.save(dict(value), temporary)
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        directory_fd = os.open(path.parent, os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if temporary.exists():
            temporary.unlink()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object: {path}")
    return value


def set_seed() -> None:
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    torch.use_deterministic_algorithms(False)


def device_from(value: str) -> torch.device:
    device = torch.device(str(value))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"requested {device}, but CUDA is unavailable")
    if device.type == "cuda" and device.index is not None and device.index >= torch.cuda.device_count():
        raise RuntimeError(f"requested {device}, only {torch.cuda.device_count()} CUDA devices visible")
    return device


def load_data(corpus_root: Path, resource_censored: bool) -> tuple[dict[str, np.ndarray], list[dict[str, Any]], dict[str, Any], dict[str, np.ndarray], list[dict[str, Any]], dict[str, Any]]:
    manifest = read_json(corpus_root / "corpus_manifest.json")
    if bool(manifest.get("resource_censored_development", False)) != bool(resource_censored):
        raise RuntimeError("resource-censored corpus flag does not match the requested isolated path")
    train_v3.CORPUS_ROOT = corpus_root
    train_arrays, train_metadata, train_summary = train_v3.load_split("train")
    validation_arrays, validation_metadata, validation_summary = train_v3.load_split("validation")
    for split, arrays in (("train", train_arrays), ("validation", validation_arrays)):
        if "public_competition_features" not in arrays:
            raise RuntimeError(f"{split} corpus has no six-dimensional public competition features")
        n = int(arrays["labels"].shape[0])
        expected = (n, int(arrays["candidate_features"].shape[1]), 6)
        if tuple(arrays["public_competition_features"].shape) != expected:
            raise RuntimeError(f"{split} competition feature shape mismatch: {arrays['public_competition_features'].shape} != {expected}")
        if not np.isfinite(arrays["public_competition_features"]).all():
            raise RuntimeError(f"{split} competition features contain non-finite values")
    return train_arrays, train_metadata, train_summary, validation_arrays, validation_metadata, validation_summary


def tensor_batch(arrays: Mapping[str, np.ndarray], indices: np.ndarray, device: torch.device) -> tuple[torch.Tensor, ...]:
    def tensor(key: str, dtype: torch.dtype | None = None) -> torch.Tensor:
        value = torch.as_tensor(arrays[key][indices], device=device)
        return value if dtype is None else value.to(dtype=dtype)

    return (
        tensor("candidate_features", torch.float32),
        tensor("candidate_mask", torch.bool),
        tensor("source_features", torch.float32),
        tensor("public_competition_features", torch.float32),
        tensor("human_anchor", torch.float32),
        tensor("recent_trusted_memory", torch.float32),
        tensor("recent_trusted_mask", torch.bool),
        tensor("long_term_trusted_memory", torch.float32),
        tensor("long_term_trusted_mask", torch.bool),
        tensor("distractor_memory", torch.float32),
        tensor("distractor_mask", torch.bool),
        tensor("neighbor_feature", torch.float32),
        tensor("temporal_features", torch.float32),
    )


def labels_and_boundary_batch(arrays: Mapping[str, np.ndarray], indices: np.ndarray, device: torch.device) -> tuple[torch.Tensor, ...]:
    return (
        torch.as_tensor(arrays["labels"][indices], device=device, dtype=torch.long),
        torch.as_tensor(arrays["candidate_mask"][indices], device=device, dtype=torch.bool),
        torch.as_tensor(arrays["legacy_target_scores"][indices], device=device, dtype=torch.float32),
        torch.as_tensor(arrays["legacy_best_other_scores"][indices], device=device, dtype=torch.float32),
        torch.as_tensor(arrays["protected_candidate_mask"][indices], device=device, dtype=torch.bool),
    )


def forward(model: PublicCompetitionTemporalIdentityModel, arrays: Mapping[str, np.ndarray], indices: np.ndarray, device: torch.device) -> torch.Tensor:
    return model(*tensor_batch(arrays, indices, device))


def evaluate(model: PublicCompetitionTemporalIdentityModel, arrays: Mapping[str, np.ndarray], device: torch.device, batch_size: int) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    total = 0
    correct = 0
    with torch.no_grad():
        for start in range(0, int(arrays["labels"].shape[0]), int(batch_size)):
            indices = np.arange(start, min(start + int(batch_size), int(arrays["labels"].shape[0])), dtype=np.int64)
            logits = forward(model, arrays, indices, device)
            labels, mask, target_scores, other_scores, protected = labels_and_boundary_batch(arrays, indices, device)
            loss, _ = public_competition_temporal_loss(logits, labels, mask, target_scores, other_scores, protected)
            count = len(indices)
            total_loss += float(loss.detach().cpu()) * count
            total += count
            correct += int((logits.argmax(dim=1) == labels).sum().detach().cpu())
    return {"loss": total_loss / max(total, 1), "accuracy": correct / max(total, 1), "examples": total}


def train_epochs(
    model: PublicCompetitionTemporalIdentityModel,
    train_arrays: Mapping[str, np.ndarray],
    validation_arrays: Mapping[str, np.ndarray],
    *,
    device: torch.device,
    epochs: int,
    learning_rate: float,
    batch_size: int,
    phase: str,
) -> tuple[list[dict[str, Any]], dict[str, float]]:
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(learning_rate), weight_decay=WEIGHT_DECAY)
    generator = np.random.default_rng(SEED + (1 if phase == "finetune" else 0))
    history: list[dict[str, Any]] = []
    best_loss = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    step = 0
    for epoch in range(int(epochs)):
        model.train()
        order = np.arange(int(train_arrays["labels"].shape[0]), dtype=np.int64)
        generator.shuffle(order)
        running = 0.0
        seen = 0
        for start in range(0, len(order), int(batch_size)):
            indices = order[start : start + int(batch_size)]
            optimizer.zero_grad(set_to_none=True)
            logits = forward(model, train_arrays, indices, device)
            labels, mask, target_scores, other_scores, protected = labels_and_boundary_batch(train_arrays, indices, device)
            loss, components = public_competition_temporal_loss(
                logits,
                labels,
                mask,
                target_scores,
                other_scores,
                protected,
                none_weight=NONE_WEIGHT,
                pairwise_weight=PAIRWISE_WEIGHT,
                pairwise_margin=PAIRWISE_MARGIN,
                positive_boundary_weight=POSITIVE_BOUNDARY_WEIGHT,
                protected_boundary_weight=PROTECTED_BOUNDARY_WEIGHT,
                delta_l2_weight=DELTA_L2_WEIGHT,
                public_margin=PUBLIC_MARGIN,
            )
            if not torch.isfinite(loss):
                raise RuntimeError(f"non-finite PCTIS {phase} loss at epoch={epoch} step={step}")
            loss.backward()
            if not all(parameter.grad is None or bool(torch.isfinite(parameter.grad).all()) for parameter in model.parameters()):
                raise RuntimeError(f"non-finite PCTIS {phase} gradient at epoch={epoch} step={step}")
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            count = len(indices)
            running += float(loss.detach().cpu()) * count
            seen += count
            step += 1
        validation = evaluate(model, validation_arrays, device, batch_size)
        history.append(
            {
                "phase": phase,
                "epoch": epoch + 1,
                "step": step,
                "train_loss": running / max(seen, 1),
                "validation": validation,
                "loss_components_last_batch": {key: float(value.cpu()) for key, value in components.items()},
            }
        )
        if float(validation["loss"]) < best_loss:
            best_loss = float(validation["loss"])
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    if best_state is None:
        raise RuntimeError("PCTIS training produced no validation checkpoint")
    model.load_state_dict(best_state, strict=True)
    return history, evaluate(model, validation_arrays, device, batch_size)


def checkpoint_payload(
    model: PublicCompetitionTemporalIdentityModel,
    *,
    phase: str,
    history: Sequence[Mapping[str, Any]],
    train_summary: Mapping[str, Any],
    validation_summary: Mapping[str, Any],
    initialization_audit: Mapping[str, Any] | None,
    corpus_root: Path,
) -> dict[str, Any]:
    return {
        "schema_version": "N72R11R4_PCTIS_CHECKPOINT_V1",
        "model_config": pctis_model_config(),
        "public_competition_feature_schema": [
            "tanh_legacy_target_score",
            "tanh_legacy_best_other_score",
            "tanh_target_minus_best_other",
            "incumbent_is_target",
            "incumbent_is_other",
            "base_solver_assigns_target",
        ],
        "state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
        "phase": str(phase),
        "seed": SEED,
        "training_config": {
            "batch_size": BATCH_SIZE,
            "initial_epochs": INITIAL_EPOCHS,
            "initial_learning_rate": INITIAL_LR,
            "finetune_epochs": FINETUNE_EPOCHS,
            "finetune_learning_rate": FINETUNE_LR,
            "weight_decay": WEIGHT_DECAY,
            "none_weight": NONE_WEIGHT,
            "pairwise_weight": PAIRWISE_WEIGHT,
            "pairwise_margin": PAIRWISE_MARGIN,
            "positive_boundary_weight": POSITIVE_BOUNDARY_WEIGHT,
            "protected_boundary_weight": PROTECTED_BOUNDARY_WEIGHT,
            "delta_l2_weight": DELTA_L2_WEIGHT,
            "public_margin": PUBLIC_MARGIN,
            "checkpoint_selection": "minimum fixed validation loss; no future effect metrics",
        },
        "train_summary": dict(train_summary),
        "validation_summary": dict(validation_summary),
        "initialization_audit": None if initialization_audit is None else dict(initialization_audit),
        "corpus_root": str(corpus_root),
        "runtime_future_gt_used": False,
        "interaction_source": "simulated_from_gt",
        "not_real_human_evidence": True,
        "production_authorized": False,
    }


def initialization_smoke(
    corpus_root: Path,
    train_arrays: Mapping[str, np.ndarray],
    v3_checkpoint: Path,
    device: torch.device,
    output_root: Path,
) -> dict[str, Any]:
    v3_model, _ = train_v3.load_checkpoint(v3_checkpoint, device)
    pctis = PublicCompetitionTemporalIdentityModel(**pctis_model_config()).to(device)
    audit = initialize_from_v3_checkpoint(pctis, v3_checkpoint)
    pctis.eval()
    v3_model.eval()
    indices = np.arange(min(4, int(train_arrays["labels"].shape[0])), dtype=np.int64)
    zeros = dict(train_arrays)
    zeros["public_competition_features"] = np.zeros_like(train_arrays["public_competition_features"])
    with torch.no_grad():
        v3_logits = train_v3.model_forward(v3_model, train_arrays, indices, device)
        pctis_logits = pctis(*tensor_batch(zeros, indices, device))
    difference = float(torch.max(torch.abs(v3_logits - pctis_logits)).detach().cpu())
    if not torch.allclose(pctis_logits, v3_logits, atol=1.0e-6, rtol=1.0e-5):
        raise RuntimeError(f"PCTIS zero-feature initialization is not V3-equivalent: max_abs_diff={difference}")
    model = PublicCompetitionTemporalIdentityModel(**pctis_model_config()).to(device)
    init_audit = initialize_from_v3_checkpoint(model, v3_checkpoint)
    model.train()
    logits = forward(model, train_arrays, indices, device)
    labels, mask, target_scores, other_scores, protected = labels_and_boundary_batch(train_arrays, indices, device)
    loss, components = public_competition_temporal_loss(logits, labels, mask, target_scores, other_scores, protected)
    if not torch.isfinite(loss):
        raise RuntimeError("PCTIS initialization smoke loss is non-finite")
    loss.backward()
    if not all(parameter.grad is None or bool(torch.isfinite(parameter.grad).all()) for parameter in model.parameters()):
        raise RuntimeError("PCTIS initialization smoke gradient is non-finite")
    smoke_path = output_root / "pctis_initialization_smoke.pt"
    atomic_torch(
        smoke_path,
        checkpoint_payload(
            model,
            phase="initialization_smoke",
            history=[],
            train_summary={"examples": int(train_arrays["labels"].shape[0])},
            validation_summary={},
            initialization_audit=init_audit,
            corpus_root=corpus_root,
        ),
    )
    return {
        "status": "PASS_N72R11R4_PCTIS_INITIALIZATION_SMOKE",
        "examples": len(indices),
        "zero_feature_max_abs_diff": difference,
        "zero_feature_atol": 1.0e-6,
        "zero_feature_rtol": 1.0e-5,
        "loss": float(loss.detach().cpu()),
        "loss_components": {key: float(value.cpu()) for key, value in components.items()},
        "initialization_audit": init_audit,
        "checkpoint": str(smoke_path),
        "checkpoint_sha256": sha256_file(smoke_path),
        "runtime_future_gt_used": False,
        "interaction_source": "simulated_from_gt",
        "not_real_human_evidence": True,
    }


def _safe_feature(value: np.ndarray) -> np.ndarray | None:
    array = np.asarray(value, dtype=np.float32).reshape(-1)
    return None if float(np.linalg.norm(array)) <= 1.0e-6 else array


def pctis_self_rollout(
    model: PublicCompetitionTemporalIdentityModel,
    arrays: Mapping[str, np.ndarray],
    metadata: Sequence[Mapping[str, Any]],
    device: torch.device,
) -> dict[str, Any]:
    model.eval()
    groups: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(metadata):
        groups[str(row["event_id"])].append(index)
    records: list[dict[str, Any]] = []
    target_total = target_correct = none_total = none_correct = 0
    future_total = future_correct = positive_total = positive_success = protected_total = protected_safe = 0
    injection_values: list[float] = []
    by_action: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    with torch.no_grad():
        for event_id in sorted(groups):
            indices = sorted(groups[event_id], key=lambda index: int(metadata[index]["frame"]))
            anchor = np.asarray(arrays["human_anchor"][indices[0]], dtype=np.float32)
            state = initialize_temporal_state(anchor_feature=anchor, anchor_box=metadata[indices[0]].get("anchor_box", [0.0, 0.0, 1.0, 1.0]))
            for index in indices:
                row_meta = metadata[index]
                count = int(arrays["candidate_counts"][index])
                candidate_features = np.asarray(arrays["candidate_features"][index : index + 1, :count], dtype=np.float32)
                source_features = np.asarray(arrays["source_features"][index : index + 1, :count], dtype=np.float32)
                competition = np.asarray(arrays["public_competition_features"][index : index + 1, :count], dtype=np.float32)
                recent, recent_mask, long_term, long_mask, distractors, distractor_mask = state_memory_arrays(state)
                temporal = build_temporal_features(
                    state,
                    frame_horizon=int(row_meta["frame_horizon"]),
                    causal_top_score=float(row_meta.get("causal_top_score", 0.0)),
                    causal_second_score=float(row_meta.get("causal_second_score", 0.0)),
                    causal_margin=float(row_meta.get("causal_margin", 0.0)),
                    has_future_requery=any(str(source) == "FUTURE_FRAME_REQUERY" for source in row_meta.get("candidate_sources", [])),
                )
                input_arrays = {
                    "candidate_features": candidate_features,
                    "candidate_mask": np.ones((1, count), dtype=np.bool_),
                    "source_features": source_features,
                    "public_competition_features": competition,
                    "human_anchor": np.asarray(arrays["human_anchor"][index : index + 1], dtype=np.float32),
                    "recent_trusted_memory": recent[None].astype(np.float32),
                    "recent_trusted_mask": recent_mask[None].astype(np.bool_),
                    "long_term_trusted_memory": long_term[None].astype(np.float32),
                    "long_term_trusted_mask": long_mask[None].astype(np.bool_),
                    "distractor_memory": distractors[None].astype(np.float32),
                    "distractor_mask": distractor_mask[None].astype(np.bool_),
                    "neighbor_feature": np.asarray(arrays["neighbor_feature"][index : index + 1], dtype=np.float32),
                    "temporal_features": temporal[None].astype(np.float32),
                }
                output = model(*tensor_batch(input_arrays, np.asarray([0], dtype=np.int64), device))[0].detach().float().cpu().numpy()
                candidate_logits = output[:count].astype(np.float64)
                none_logit = float(output[count])
                selection = select_candidate_from_logits(candidate_logits, none_logit, [str(uid) for uid in row_meta["candidate_uids"][:count]])
                predicted_index = int(selection["selected_index"])
                label_index = int(arrays["labels"][index])
                action = str(row_meta.get("action_type", "UNKNOWN"))
                label_kind = str(row_meta.get("label_kind", "UNKNOWN"))
                if label_kind == "TARGET_CANDIDATE":
                    target_total += 1
                    target_correct += int(predicted_index == label_index)
                    by_action[action]["target_total"] += 1
                    by_action[action]["target_correct"] += int(predicted_index == label_index)
                    source = str(row_meta.get("candidate_sources", ["UNKNOWN"] * count)[label_index])
                    if source == "FUTURE_FRAME_REQUERY":
                        future_total += 1
                        future_correct += int(predicted_index == label_index)
                        by_action[action]["future_total"] += 1
                        by_action[action]["future_correct"] += int(predicted_index == label_index)
                elif label_kind == "NONE":
                    none_total += 1
                    none_correct += int(predicted_index == count)
                    by_action[action]["none_total"] += 1
                    by_action[action]["none_correct"] += int(predicted_index == count)
                model_scores = candidate_logits - none_logit
                injection = np.maximum(model_scores, 0.0)
                legacy_target = np.asarray(arrays["legacy_target_scores"][index, :count], dtype=np.float64)
                legacy_other = np.asarray(arrays["legacy_best_other_scores"][index, :count], dtype=np.float64)
                fused = legacy_target + injection
                injection_values.extend(float(value) for value in injection if math.isfinite(float(value)))
                if label_kind == "TARGET_CANDIDATE" and 0 <= label_index < count:
                    positive_total += 1
                    positive_ok = bool(fused[label_index] >= legacy_other[label_index] + PUBLIC_MARGIN)
                    positive_success += int(positive_ok)
                    by_action[action]["positive_total"] += 1
                    by_action[action]["positive_success"] += int(positive_ok)
                protected = np.asarray(arrays["protected_candidate_mask"][index, :count], dtype=np.bool_).copy()
                if label_kind == "TARGET_CANDIDATE" and 0 <= label_index < count:
                    protected[label_index] = False
                protected_count = int(protected.sum())
                protected_total += protected_count
                protected_ok = (fused <= legacy_other - PUBLIC_MARGIN) | ~protected
                protected_safe += int(np.logical_and(protected, protected_ok).sum())
                by_action[action]["protected_total"] += protected_count
                by_action[action]["protected_safe"] += int(np.logical_and(protected, protected_ok).sum())
                candidate_rows = [
                    {
                        "candidate_uid": str(row_meta["candidate_uids"][candidate_index]),
                        "box_xyxy": row_meta["candidate_boxes"][candidate_index],
                        "candidate_source": row_meta.get("candidate_sources", ["UNKNOWN"] * count)[candidate_index],
                        "feature": _safe_feature(candidate_features[0, candidate_index, :512]),
                        "feature_available": bool(np.linalg.norm(candidate_features[0, candidate_index, :512]) > 1.0e-6),
                        "official_raw_sam_id": None,
                        "native_scope": None,
                    }
                    for candidate_index in range(count)
                ]
                accepted = bool(selection["accepted"])
                top_index = int(selection["best_candidate_index"])
                selected_uid = None if not accepted else str(row_meta["candidate_uids"][top_index])
                update = update_temporal_state(
                    state,
                    candidates=candidate_rows,
                    target_uid=selected_uid,
                    selected_uid=selected_uid,
                    selected_score=None if not accepted else float(selection["selected_score"]),
                    selected_margin=float(selection["best_minus_second_margin"]),
                    fused_target_scores=fused,
                    frame_horizon=int(row_meta["frame_horizon"]),
                    assigned_candidate=None if not accepted else candidate_rows[top_index],
                    base_top_score=float(row_meta.get("causal_top_score", 0.0)),
                    base_second_score=float(row_meta.get("causal_second_score", 0.0)),
                )
                records.append(
                    {
                        "event_id": event_id,
                        "sequence": str(row_meta["sequence"]),
                        "action_type": action,
                        "frame": int(row_meta["frame"]),
                        "frame_horizon": int(row_meta["frame_horizon"]),
                        "predicted_index": predicted_index,
                        "predicted_uid": None if not accepted else str(row_meta["candidate_uids"][top_index]),
                        "label_index": label_index,
                        "label_kind": label_kind,
                        "candidate_logits": candidate_logits.tolist(),
                        "none_logit": none_logit,
                        "model_scores": model_scores.tolist(),
                        "injection_deltas": injection.tolist(),
                        "legacy_target_scores": legacy_target.tolist(),
                        "legacy_best_other_scores": legacy_other.tolist(),
                        "fused_target_edges": fused.tolist(),
                        "positive_boundary_success": None if label_kind != "TARGET_CANDIDATE" else bool(fused[label_index] >= legacy_other[label_index] + PUBLIC_MARGIN),
                        "protected_candidate_count": protected_count,
                        "protected_safe_count": int(np.logical_and(protected, protected_ok).sum()),
                        "recent_memory_size_before": len(state.recent_trusted),
                        "long_term_memory_size_before": len(state.long_term_trusted),
                        "distractor_memory_size_before": len(state.distractors),
                        "runtime_future_gt_used": False,
                        "gt_used_for_runtime_decision": False,
                        "temporal_state_update": update,
                        "temporal_state_after": state_audit(state),
                    }
                )
    target_accuracy = None if target_total == 0 else target_correct / target_total
    none_accuracy = None if none_total == 0 else none_correct / none_total
    future_accuracy = None if future_total == 0 else future_correct / future_total
    boundary_rate = None if positive_total == 0 else positive_success / positive_total
    protected_rate = None if protected_total == 0 else protected_safe / protected_total
    readiness = {
        "target_candidate_accuracy": target_accuracy,
        "none_accuracy": none_accuracy,
        "future_requery_positive_accuracy": future_accuracy,
        "future_requery_positive_count": future_total,
        "positive_public_boundary_success_rate": boundary_rate,
        "protected_public_safe_rate": protected_rate,
        "mean_injection_delta": None if not injection_values else float(np.mean(injection_values)),
        "p95_injection_delta": None if not injection_values else float(np.percentile(np.asarray(injection_values), 95)),
        "thresholds": {
            "target_candidate_accuracy_min": TARGET_ACCURACY_MIN,
            "none_accuracy_min": NONE_ACCURACY_MIN,
            "positive_public_boundary_success_min": POSITIVE_BOUNDARY_MIN,
            "protected_public_safe_min": PROTECTED_SAFE_MIN,
            "future_requery_positive_count_min_for_coverage_report": FUTURE_COUNT_MIN,
        },
    }
    gate = {
        "target_candidate_accuracy": target_accuracy is not None and target_accuracy >= TARGET_ACCURACY_MIN,
        "none_accuracy": none_accuracy is not None and none_accuracy >= NONE_ACCURACY_MIN,
        "positive_public_boundary_success": boundary_rate is not None and boundary_rate >= POSITIVE_BOUNDARY_MIN,
        "protected_public_safe": protected_rate is not None and protected_rate >= PROTECTED_SAFE_MIN,
    }
    readiness["gate"] = gate
    readiness["all_model_gates_pass"] = all(gate.values())
    readiness["future_source_coverage_status"] = "PASS" if future_total >= FUTURE_COUNT_MIN else "FUTURE_SOURCE_COVERAGE_LIMITED"
    return {
        "schema_version": "N72R11R4_PCTIS_SELF_ROLLOUT_V1",
        "status": "PASS_N72R11R4_PCTIS_READINESS" if readiness["all_model_gates_pass"] else "PCTIS_NOT_READY",
        "example_count": len(records),
        "event_count": len(groups),
        "accepted_candidate_count": sum(int(row["predicted_index"] < len(row["candidate_logits"])) for row in records),
        "readiness": readiness,
        "by_action_counts": {action: dict(values) for action, values in sorted(by_action.items())},
        "records": records,
        "runtime_future_gt_used": False,
        "gt_used_for_runtime_decision": False,
        "interaction_source": "simulated_from_gt",
        "not_real_human_evidence": True,
    }


def main() -> int:
    parser = __import__("argparse").ArgumentParser()
    parser.add_argument("--phase", choices=("smoke", "train", "finetune", "self_rollout"), required=True)
    parser.add_argument("--corpus-root", type=Path, default=ROOT / "outputs/N72R11R4/exact_onpolicy_v3")
    parser.add_argument("--output-root", type=Path, default=ROOT / "outputs/N72R11R4/pctis_training")
    parser.add_argument("--stage-path", type=Path, default=None)
    parser.add_argument("--v3-checkpoint", type=Path, default=None)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--output-checkpoint", type=Path, default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--resource-censored", action="store_true")
    args = parser.parse_args()
    corpus_root = args.corpus_root if args.corpus_root.is_absolute() else ROOT / args.corpus_root
    output_root = args.output_root if args.output_root.is_absolute() else ROOT / args.output_root
    phase = str(args.phase)
    default_stage = {
        "smoke": ROOT / "outputs/N72R11R4/stage_12_pctis_initialization_smoke.json",
        "train": ROOT / "outputs/N72R11R4/stage_13_pctis_training.json",
        "finetune": ROOT / "outputs/N72R11R4/stage_15_pctis_finetune.json",
        "self_rollout": ROOT / "outputs/N72R11R4/stage_14_pctis_readiness.json",
    }[phase]
    stage_path = args.stage_path if args.stage_path is not None else default_stage
    if not stage_path.is_absolute():
        stage_path = ROOT / stage_path
    device = device_from(str(args.device))
    v3_checkpoint = None if args.v3_checkpoint is None else (args.v3_checkpoint if args.v3_checkpoint.is_absolute() else ROOT / args.v3_checkpoint)
    checkpoint = None if args.checkpoint is None else (args.checkpoint if args.checkpoint.is_absolute() else ROOT / args.checkpoint)
    output_checkpoint = None if args.output_checkpoint is None else (args.output_checkpoint if args.output_checkpoint.is_absolute() else ROOT / args.output_checkpoint)
    status: dict[str, Any] = {
        "schema_version": "N72R11R4_PCTIS_STAGE_STATUS_V1",
        "stage": f"N72R11R4-{phase}",
        "started_at_utc": now_utc(),
        "runtime_future_gt_used": False,
        "interaction_source": "simulated_from_gt",
        "real_human_evidence": False,
        "not_real_human_evidence": True,
        "production_authorized": False,
    }
    set_seed()
    try:
        train_arrays, train_metadata, train_summary, validation_arrays, validation_metadata, validation_summary = load_data(corpus_root, bool(args.resource_censored))
        status.update({"corpus_root": str(corpus_root), "train_summary": train_summary, "validation_summary": validation_summary, "device": str(device), "resource_censored_development": bool(args.resource_censored)})
        if phase == "smoke":
            if v3_checkpoint is None or not v3_checkpoint.is_file():
                raise FileNotFoundError(f"PCTIS smoke requires V3 checkpoint: {v3_checkpoint}")
            result = initialization_smoke(corpus_root, train_arrays, v3_checkpoint, device, output_root)
            status.update(result)
        elif phase in {"train", "finetune"}:
            if phase == "train":
                if v3_checkpoint is None or not v3_checkpoint.is_file():
                    raise FileNotFoundError(f"PCTIS initial training requires V3 checkpoint: {v3_checkpoint}")
                model = PublicCompetitionTemporalIdentityModel(**pctis_model_config()).to(device)
                init_audit = initialize_from_v3_checkpoint(model, v3_checkpoint)
                epochs, learning_rate = INITIAL_EPOCHS, INITIAL_LR
                source_checkpoint = v3_checkpoint
            else:
                if checkpoint is None or not checkpoint.is_file():
                    raise FileNotFoundError(f"PCTIS finetune requires checkpoint: {checkpoint}")
                model = load_pctis_checkpoint(checkpoint, device)
                init_audit = None
                epochs, learning_rate = FINETUNE_EPOCHS, FINETUNE_LR
                source_checkpoint = checkpoint
            history, selected_validation = train_epochs(model, train_arrays, validation_arrays, device=device, epochs=epochs, learning_rate=learning_rate, batch_size=int(args.batch_size), phase=phase)
            if output_checkpoint is None:
                output_checkpoint = output_root / ("pctis_initial.pt" if phase == "train" else "pctis_onpolicy_finetuned.pt")
            payload = checkpoint_payload(model, phase=phase, history=history, train_summary=train_summary, validation_summary=selected_validation, initialization_audit=init_audit, corpus_root=corpus_root)
            atomic_torch(output_checkpoint, payload)
            status.update({
                "status": "PASS_N72R11R4_PCTIS_TRAINING" if phase == "train" else "PASS_N72R11R4_PCTIS_ONPOLICY_FINETUNE",
                "source_checkpoint": str(source_checkpoint),
                "source_checkpoint_sha256": sha256_file(source_checkpoint),
                "output_checkpoint": str(output_checkpoint),
                "output_checkpoint_sha256": sha256_file(output_checkpoint),
                "epochs": epochs,
                "learning_rate": learning_rate,
                "batch_size": int(args.batch_size),
                "weight_decay": WEIGHT_DECAY,
                "checkpoint_selection": "minimum fixed validation loss",
                "initialization_audit": init_audit,
                "history": history,
                "selected_validation": selected_validation,
            })
        else:
            if checkpoint is None or not checkpoint.is_file():
                raise FileNotFoundError(f"PCTIS self-rollout requires checkpoint: {checkpoint}")
            model = load_pctis_checkpoint(checkpoint, device)
            rollout = pctis_self_rollout(model, validation_arrays, validation_metadata, device)
            output = output_root / "pctis_self_rollout.json"
            atomic_json(output, {**rollout, "checkpoint": str(checkpoint), "checkpoint_sha256": sha256_file(checkpoint), "created_at_utc": now_utc()})
            readiness = rollout["readiness"]
            status.update({
                "status": rollout["status"],
                "checkpoint": str(checkpoint),
                "checkpoint_sha256": sha256_file(checkpoint),
                "output": str(output),
                "output_sha256": sha256_file(output),
                "example_count": rollout["example_count"],
                "event_count": rollout["event_count"],
                "readiness": readiness,
                "by_action_counts": rollout["by_action_counts"],
            })
        atomic_json(stage_path, status)
        print(json.dumps(status, sort_keys=True))
        return 0
    except Exception as exc:
        failure = output_root / "attempts" / f"{phase}_failure_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.json"
        payload = {**status, "status": f"FAIL_N72R11R4_PCTIS_{phase.upper()}", "error_type": type(exc).__name__, "error": str(exc), "traceback": traceback.format_exc(), "finished_at_utc": now_utc(), "failure_artifact": str(failure)}
        atomic_json(failure, payload)
        atomic_json(stage_path, payload)
        print(json.dumps({"status": payload["status"], "failure_artifact": str(failure), "error": str(exc)}, sort_keys=True))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
