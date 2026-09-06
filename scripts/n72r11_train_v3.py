#!/usr/bin/env python3
"""Train and audit the isolated N72R11 temporal V3 scorer.

The script deliberately keeps the three required operations separate:
``smoke`` checks autograd/checkpoint integrity, ``bootstrap`` trains one
fixed configuration on the pre-sealed sequence split, ``self_rollout`` runs
the scorer with its own causal memory updates, and ``finetune`` performs one
pre-registered continuation from the bootstrap checkpoint.  No future label
is read by the self-rollout or by the model input path; labels in the sealed
corpus are used only by the offline training/evaluation loss.
"""

from __future__ import annotations

from collections import defaultdict
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
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in __import__("sys").path:
    __import__("sys").path.insert(0, str(ROOT))

from sam3_intermot.reacquisition.models.n72r11_temporal_v3 import (  # noqa: E402
    CANDIDATE_FEATURE_DIM,
    DISTRACTOR_SLOTS,
    LONG_TERM_TRUSTED_SLOTS,
    N72R11TemporalIdentityModel,
    RECENT_TRUSTED_SLOTS,
    SOURCE_FEATURE_DIM,
    TEMPORAL_FEATURE_DIM,
    n72r11_loss,
)


CORPUS_ROOT = ROOT / "outputs/N72R11/training_v3"
OUTPUT_ROOT = ROOT / "outputs/N72R11/training_v3"
STAGE_07 = ROOT / "outputs/N72R11/stage_07_training_input_audit.json"
STAGE_08 = ROOT / "outputs/N72R11/stage_08_training_smoke.json"
STAGE_09 = ROOT / "outputs/N72R11/stage_09_bootstrap_training.json"
STAGE_10 = ROOT / "outputs/N72R11/stage_10_causal_self_rollout.json"
STAGE_11 = ROOT / "outputs/N72R11/stage_11_finetune.json"
SEED = 7211
DEFAULT_BATCH_SIZE = 64
BOOTSTRAP_EPOCHS = 2
FINETUNE_EPOCHS = 1
LEARNING_RATE = 5.0e-4
FINETUNE_LEARNING_RATE = 2.5e-4
WEIGHT_DECAY = 1.0e-4
NONE_WEIGHT = 2.0
PAIRWISE_WEIGHT = 0.25
PAIRWISE_MARGIN = 0.20


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


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


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object: {path}")
    return value


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def set_seed(seed: int = SEED) -> None:
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    torch.use_deterministic_algorithms(False)


def device_from(value: str) -> torch.device:
    requested = torch.device(str(value))
    if requested.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"requested {requested}, but CUDA is unavailable")
    if requested.type == "cuda" and requested.index is not None and requested.index >= torch.cuda.device_count():
        raise RuntimeError(f"requested {requested}, only {torch.cuda.device_count()} CUDA devices visible")
    return requested


def model_config() -> dict[str, Any]:
    return {
        "candidate_feature_dim": CANDIDATE_FEATURE_DIM,
        "source_feature_dim": SOURCE_FEATURE_DIM,
        "temporal_feature_dim": TEMPORAL_FEATURE_DIM,
        "recent_trusted_slots": RECENT_TRUSTED_SLOTS,
        "long_term_trusted_slots": LONG_TERM_TRUSTED_SLOTS,
        "distractor_slots": DISTRACTOR_SLOTS,
        "hidden_dim": 256,
        "layers": 3,
        "heads": 8,
        "dropout": 0.0,
    }


def load_split(split: str) -> tuple[dict[str, np.ndarray], list[dict[str, Any]], dict[str, Any]]:
    npz_path = CORPUS_ROOT / f"{split}.npz"
    metadata_path = CORPUS_ROOT / f"{split}_metadata.jsonl"
    if not npz_path.is_file() or not metadata_path.is_file():
        raise FileNotFoundError(f"sealed {split} corpus missing: {npz_path} {metadata_path}")
    with np.load(npz_path, allow_pickle=False) as loaded:
        arrays = {key: loaded[key] for key in loaded.files}
    metadata = read_jsonl(metadata_path)
    n = int(arrays["labels"].shape[0])
    if len(metadata) != n:
        raise RuntimeError(f"{split} metadata/array count mismatch: {len(metadata)} != {n}")
    required = {
        "candidate_features", "candidate_mask", "source_features", "human_anchor",
        "recent_trusted_memory", "recent_trusted_mask", "long_term_trusted_memory", "long_term_trusted_mask",
        "distractor_memory", "distractor_mask", "neighbor_feature", "temporal_features", "labels", "candidate_counts",
        "legacy_target_scores", "legacy_best_other_scores", "incumbent_target", "incumbent_other", "confidence", "presence",
        "motion_iou", "protected_candidate_mask",
    }
    missing = sorted(required - set(arrays))
    if missing:
        raise RuntimeError(f"{split} corpus missing arrays: {missing}")
    candidate_count = int(arrays["candidate_features"].shape[1])
    expected_shapes = {
        "candidate_features": (n, candidate_count, CANDIDATE_FEATURE_DIM),
        "candidate_mask": (n, candidate_count),
        "source_features": (n, candidate_count, SOURCE_FEATURE_DIM),
        "human_anchor": (n, 512),
        "recent_trusted_memory": (n, RECENT_TRUSTED_SLOTS, 512),
        "recent_trusted_mask": (n, RECENT_TRUSTED_SLOTS),
        "long_term_trusted_memory": (n, LONG_TERM_TRUSTED_SLOTS, 512),
        "long_term_trusted_mask": (n, LONG_TERM_TRUSTED_SLOTS),
        "distractor_memory": (n, DISTRACTOR_SLOTS, 512),
        "distractor_mask": (n, DISTRACTOR_SLOTS),
        "neighbor_feature": (n, 512),
        "temporal_features": (n, TEMPORAL_FEATURE_DIM),
        "labels": (n,),
        "candidate_counts": (n,),
        "legacy_target_scores": (n, candidate_count),
        "legacy_best_other_scores": (n, candidate_count),
        "incumbent_target": (n, candidate_count),
        "incumbent_other": (n, candidate_count),
        "confidence": (n, candidate_count),
        "presence": (n, candidate_count),
        "motion_iou": (n, candidate_count),
        "protected_candidate_mask": (n, candidate_count),
    }
    for key, shape in expected_shapes.items():
        if tuple(arrays[key].shape) != shape:
            raise RuntimeError(f"{split} {key} shape {arrays[key].shape} != {shape}")
        if arrays[key].dtype.kind not in "biu" and not np.isfinite(arrays[key]).all():
            raise RuntimeError(f"{split} {key} contains non-finite values")
    if np.any(arrays["candidate_counts"] < 1) or np.any(arrays["candidate_counts"] > candidate_count):
        raise RuntimeError(f"{split} candidate count outside padded axis")
    for index, row in enumerate(metadata):
        if row.get("runtime_future_gt_used") is not False or row.get("runtime_gt_read") is not False:
            raise RuntimeError(f"{split} metadata runtime GT violation at {index}")
        if row.get("interaction_source") != "simulated_from_gt" or row.get("not_real_human_evidence") is not True:
            raise RuntimeError(f"{split} metadata provenance violation at {index}")
        if int(row.get("label_index_raw", -1)) < 0 or int(row["label_index_raw"]) > len(row.get("candidate_uids", [])):
            raise RuntimeError(f"{split} metadata label outside candidate/NONE axis at {index}")
    summary = {
        "split": split,
        "examples": n,
        "max_candidates": candidate_count,
        "sequence_count": len({str(row["sequence"]) for row in metadata}),
        "event_count": len({str(row["event_id"]) for row in metadata}),
        "future_positive_count": sum(row.get("label_kind") == "TARGET_CANDIDATE" for row in metadata),
        "npz_sha256": sha256_file(npz_path),
        "metadata_sha256": sha256_file(metadata_path),
    }
    return arrays, metadata, summary


def tensor_batch(arrays: Mapping[str, np.ndarray], indices: np.ndarray, device: torch.device) -> tuple[torch.Tensor, ...]:
    def t(key: str, dtype: torch.dtype | None = None) -> torch.Tensor:
        value = torch.as_tensor(arrays[key][indices], device=device)
        return value if dtype is None else value.to(dtype=dtype)

    return (
        t("candidate_features", torch.float32), t("candidate_mask", torch.bool), t("source_features", torch.float32),
        t("human_anchor", torch.float32), t("recent_trusted_memory", torch.float32), t("recent_trusted_mask", torch.bool),
        t("long_term_trusted_memory", torch.float32), t("long_term_trusted_mask", torch.bool), t("distractor_memory", torch.float32),
        t("distractor_mask", torch.bool), t("neighbor_feature", torch.float32), t("temporal_features", torch.float32),
    )


def labels_batch(arrays: Mapping[str, np.ndarray], indices: np.ndarray, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    return (
        torch.as_tensor(arrays["labels"][indices], device=device, dtype=torch.long),
        torch.as_tensor(arrays["candidate_mask"][indices], device=device, dtype=torch.bool),
    )


def model_forward(model: nn.Module, arrays: Mapping[str, np.ndarray], indices: np.ndarray, device: torch.device) -> torch.Tensor:
    return model(*tensor_batch(arrays, indices, device))


def evaluate(model: nn.Module, arrays: Mapping[str, np.ndarray], batch_size: int, device: torch.device) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_count = 0
    correct = 0
    with torch.no_grad():
        for start in range(0, int(arrays["labels"].shape[0]), int(batch_size)):
            indices = np.arange(start, min(start + int(batch_size), int(arrays["labels"].shape[0])), dtype=np.int64)
            logits = model_forward(model, arrays, indices, device)
            labels, mask = labels_batch(arrays, indices, device)
            loss, _ = n72r11_loss(logits, labels, mask, none_weight=NONE_WEIGHT, pairwise_weight=PAIRWISE_WEIGHT, pairwise_margin=PAIRWISE_MARGIN)
            total_loss += float(loss.detach().cpu()) * len(indices)
            total_count += len(indices)
            correct += int((logits.argmax(dim=1) == labels).sum().detach().cpu())
    return {"loss": total_loss / max(total_count, 1), "accuracy": correct / max(total_count, 1), "examples": total_count}


def train_epochs(
    model: nn.Module,
    train_arrays: Mapping[str, np.ndarray],
    validation_arrays: Mapping[str, np.ndarray],
    *,
    device: torch.device,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    phase: str,
) -> tuple[list[dict[str, Any]], dict[str, float]]:
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(learning_rate), weight_decay=WEIGHT_DECAY)
    history: list[dict[str, Any]] = []
    generator = np.random.default_rng(SEED + (1 if phase == "finetune" else 0))
    global_step = 0
    for epoch in range(int(epochs)):
        model.train()
        order = np.arange(int(train_arrays["labels"].shape[0]), dtype=np.int64)
        generator.shuffle(order)
        running = 0.0
        seen = 0
        for start in range(0, len(order), int(batch_size)):
            indices = order[start : start + int(batch_size)]
            optimizer.zero_grad(set_to_none=True)
            logits = model_forward(model, train_arrays, indices, device)
            labels, mask = labels_batch(train_arrays, indices, device)
            loss, components = n72r11_loss(
                logits,
                labels,
                mask,
                none_weight=NONE_WEIGHT,
                pairwise_weight=PAIRWISE_WEIGHT,
                pairwise_margin=PAIRWISE_MARGIN,
            )
            if not torch.isfinite(loss):
                raise RuntimeError(f"non-finite {phase} loss at epoch={epoch} step={global_step}")
            loss.backward()
            gradients_finite = all(parameter.grad is None or bool(torch.isfinite(parameter.grad).all()) for parameter in model.parameters())
            if not gradients_finite:
                raise RuntimeError(f"non-finite {phase} gradient at epoch={epoch} step={global_step}")
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            batch_count = len(indices)
            running += float(loss.detach().cpu()) * batch_count
            seen += batch_count
            global_step += 1
        validation = evaluate(model, validation_arrays, batch_size, device)
        history.append(
            {
                "phase": phase,
                "epoch": epoch + 1,
                "step": global_step,
                "train_loss": running / max(seen, 1),
                "validation": validation,
            }
        )
    return history, evaluate(model, validation_arrays, batch_size, device)


def checkpoint_payload(model: nn.Module, *, phase: str, history: Sequence[Mapping[str, Any]], train_summary: Mapping[str, Any], validation_summary: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": "N72R11_V3_SCORER_CHECKPOINT_V1",
        "model_config": model_config(),
        "state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
        "phase": phase,
        "seed": SEED,
        "training_config": {
            "batch_size": DEFAULT_BATCH_SIZE,
            "bootstrap_epochs": BOOTSTRAP_EPOCHS,
            "finetune_epochs": FINETUNE_EPOCHS,
            "learning_rate": LEARNING_RATE,
            "finetune_learning_rate": FINETUNE_LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "none_weight": NONE_WEIGHT,
            "pairwise_weight": PAIRWISE_WEIGHT,
            "pairwise_margin": PAIRWISE_MARGIN,
            "checkpoint_selection": "minimum fixed validation loss; no future effect metrics",
        },
        "train_summary": dict(train_summary),
        "validation_summary": dict(validation_summary),
        "history": [dict(item) for item in history],
        "runtime_future_gt_used": False,
        "interaction_source": "simulated_from_gt",
        "not_real_human_evidence": True,
        "production_authorized": False,
    }


def load_checkpoint(path: Path, device: torch.device) -> tuple[nn.Module, dict[str, Any]]:
    payload = torch.load(path, map_location=device, weights_only=False)
    if not isinstance(payload, Mapping) or not isinstance(payload.get("state_dict"), Mapping):
        raise RuntimeError(f"invalid V3 checkpoint: {path}")
    model = N72R11TemporalIdentityModel(**model_config()).to(device)
    model.load_state_dict(payload["state_dict"], strict=True)
    return model, dict(payload)


def memory_pad(values: Sequence[np.ndarray], slots: int) -> tuple[np.ndarray, np.ndarray]:
    result = np.zeros((int(slots), 512), dtype=np.float32)
    mask = np.zeros(int(slots), dtype=np.bool_)
    for index, value in enumerate(list(values)[-int(slots):]):
        value = np.asarray(value, dtype=np.float32).reshape(-1)
        norm = float(np.linalg.norm(value))
        if value.size != 512 or not np.isfinite(value).all() or norm <= 1.0e-6:
            raise RuntimeError("self-rollout memory feature is invalid")
        result[index] = value / norm
        mask[index] = True
    return result, mask


def run_self_rollout(model: nn.Module, arrays: Mapping[str, np.ndarray], metadata: Sequence[Mapping[str, Any]], device: torch.device) -> dict[str, Any]:
    model.eval()
    groups: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(metadata):
        groups[str(row["event_id"])].append(index)
    records: list[dict[str, Any]] = []
    for event_id in sorted(groups):
        indices = sorted(groups[event_id], key=lambda index: int(metadata[index]["frame"]))
        anchor = np.asarray(arrays["human_anchor"][indices[0]], dtype=np.float32)
        recent: list[np.ndarray] = [anchor]
        long_term: list[np.ndarray] = [anchor]
        distractors: list[np.ndarray] = []
        for index in indices:
            c = int(arrays["candidate_counts"][index])
            candidate_features = np.asarray(arrays["candidate_features"][index : index + 1, :c], dtype=np.float32)
            source_features = np.asarray(arrays["source_features"][index : index + 1, :c], dtype=np.float32)
            recent_array, recent_mask = memory_pad(recent, RECENT_TRUSTED_SLOTS)
            long_array, long_mask = memory_pad(long_term, LONG_TERM_TRUSTED_SLOTS)
            distractor_array, distractor_mask = memory_pad(distractors, DISTRACTOR_SLOTS)
            tensors = (
                torch.as_tensor(candidate_features, device=device),
                torch.ones((1, c), dtype=torch.bool, device=device),
                torch.as_tensor(source_features, device=device),
                torch.as_tensor(arrays["human_anchor"][index : index + 1], dtype=torch.float32, device=device),
                torch.as_tensor(recent_array[None], dtype=torch.float32, device=device),
                torch.as_tensor(recent_mask[None], dtype=torch.bool, device=device),
                torch.as_tensor(long_array[None], dtype=torch.float32, device=device),
                torch.as_tensor(long_mask[None], dtype=torch.bool, device=device),
                torch.as_tensor(distractor_array[None], dtype=torch.float32, device=device),
                torch.as_tensor(distractor_mask[None], dtype=torch.bool, device=device),
                torch.as_tensor(arrays["neighbor_feature"][index : index + 1], dtype=torch.float32, device=device),
                torch.as_tensor(arrays["temporal_features"][index : index + 1], dtype=torch.float32, device=device),
            )
            with torch.no_grad():
                output = model(*tensors)[0].detach().float().cpu().numpy()
            candidate_logits = output[:c]
            none_logit = float(output[c])
            order = np.argsort(-candidate_logits, kind="stable")
            top_index = int(order[0]) if c else c
            top = float(candidate_logits[top_index]) if c else float("-inf")
            second = float(candidate_logits[int(order[1])]) if c > 1 else float("-inf")
            margin = top - second if c > 1 else float("inf")
            accepted = bool(c and top > none_logit and (c == 1 or margin >= 0.20))
            selected_index = top_index if accepted else c
            records.append(
                {
                    "event_id": event_id,
                    "sequence": str(metadata[index]["sequence"]),
                    "frame": int(metadata[index]["frame"]),
                    "frame_horizon": int(metadata[index]["frame_horizon"]),
                    "predicted_index": selected_index,
                    "predicted_uid": None if not accepted else str(metadata[index]["candidate_uids"][top_index]),
                    "none_logit": none_logit,
                    "top_candidate_logit": top,
                    "second_candidate_logit": second,
                    "candidate_margin": margin,
                    "accepted": accepted,
                    "recent_memory_size_before": len(recent),
                    "long_term_memory_size_before": len(long_term),
                    "distractor_memory_size_before": len(distractors),
                    "runtime_future_gt_used": False,
                    "gt_used_for_runtime_decision": False,
                }
            )
            if accepted:
                feature = candidate_features[0, top_index, :512]
                feature_norm = float(np.linalg.norm(feature))
                if feature.size == 512 and np.isfinite(feature).all() and feature_norm > 1.0e-6:
                    recent.append(feature)
                    if margin >= 0.30:
                        long_term.append(feature)
                    recent = recent[-RECENT_TRUSTED_SLOTS:]
                    long_term = long_term[-LONG_TERM_TRUSTED_SLOTS:]
            if c > 1:
                distractor = candidate_features[0, int(order[1]), :512]
                distractor_norm = float(np.linalg.norm(distractor))
                if distractor.size == 512 and np.isfinite(distractor).all() and distractor_norm > 1.0e-6:
                    distractors.append(distractor)
                    distractors = distractors[-DISTRACTOR_SLOTS:]
    predicted = sum(bool(row["accepted"]) for row in records)
    return {
        "schema_version": "N72R11_CAUSAL_SELF_ROLLOUT_V1",
        "status": "PASS_N72R11_CAUSAL_SELF_ROLLOUT",
        "example_count": len(records),
        "event_count": len(groups),
        "accepted_candidate_count": int(predicted),
        "none_count": int(len(records) - predicted),
        "records": records,
        "runtime_future_gt_used": False,
        "gt_used_for_runtime_decision": False,
        "interaction_source": "simulated_from_gt",
        "not_real_human_evidence": True,
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
    global CORPUS_ROOT, OUTPUT_ROOT, STAGE_07, STAGE_08, STAGE_09, STAGE_10, STAGE_11

    parser = __import__("argparse").ArgumentParser()
    parser.add_argument("--phase", choices=("smoke", "bootstrap", "self_rollout", "finetune"), required=True)
    parser.add_argument("--device", default="cuda:5")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--corpus-root", type=Path, default=CORPUS_ROOT)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--stage-dir", type=Path, default=ROOT / "outputs/N72R11")
    parser.add_argument("--resource-censored", action="store_true")
    parser.add_argument("--bootstrap-checkpoint", type=Path, default=None)
    parser.add_argument("--output-checkpoint", type=Path, default=None)
    args = parser.parse_args()
    CORPUS_ROOT = args.corpus_root if args.corpus_root.is_absolute() else ROOT / args.corpus_root
    OUTPUT_ROOT = args.output_root if args.output_root.is_absolute() else ROOT / args.output_root
    stage_dir = args.stage_dir if args.stage_dir.is_absolute() else ROOT / args.stage_dir
    STAGE_07 = stage_dir / "stage_07_training_input_audit.json"
    STAGE_08 = stage_dir / "stage_08_training_smoke.json"
    STAGE_09 = stage_dir / "stage_09_bootstrap_training.json"
    STAGE_10 = stage_dir / "stage_10_causal_self_rollout.json"
    STAGE_11 = stage_dir / "stage_11_finetune.json"
    bootstrap_checkpoint = args.bootstrap_checkpoint
    if bootstrap_checkpoint is None:
        bootstrap_checkpoint = OUTPUT_ROOT / "v3_bootstrap.pt"
    elif not bootstrap_checkpoint.is_absolute():
        bootstrap_checkpoint = ROOT / bootstrap_checkpoint
    set_seed()
    device = device_from(str(args.device))
    phase = str(args.phase)
    stage_path = {"smoke": STAGE_08, "bootstrap": STAGE_09, "self_rollout": STAGE_10, "finetune": STAGE_11}[phase]
    status = stage_base(f"N72R11-{phase}")
    status["resource_censored_development"] = bool(args.resource_censored)
    try:
        corpus_manifest = read_json(CORPUS_ROOT / "corpus_manifest.json")
        corpus_is_resource_censored = bool(corpus_manifest.get("resource_censored_development", False))
        if corpus_is_resource_censored != bool(args.resource_censored):
            raise RuntimeError(
                "resource-censored corpus requires --resource-censored and cannot be used by the default production path"
            )
        train_arrays, train_metadata, train_summary = load_split("train")
        validation_arrays, validation_metadata, validation_summary = load_split("validation")
        if phase == "smoke":
            model = N72R11TemporalIdentityModel(**model_config()).to(device)
            count = min(4, int(train_arrays["labels"].shape[0]))
            indices = np.arange(count, dtype=np.int64)
            model.train()
            logits = model_forward(model, train_arrays, indices, device)
            labels, mask = labels_batch(train_arrays, indices, device)
            loss, components = n72r11_loss(logits, labels, mask, none_weight=NONE_WEIGHT, pairwise_weight=PAIRWISE_WEIGHT, pairwise_margin=PAIRWISE_MARGIN)
            if not torch.isfinite(loss):
                raise RuntimeError("smoke loss is non-finite")
            loss.backward()
            if not all(parameter.grad is None or bool(torch.isfinite(parameter.grad).all()) for parameter in model.parameters()):
                raise RuntimeError("smoke gradient is non-finite")
            checkpoint = OUTPUT_ROOT / "v3_smoke.pt"
            atomic_torch(checkpoint, checkpoint_payload(model, phase="smoke", history=[], train_summary=train_summary, validation_summary=validation_summary))
            restored, _ = load_checkpoint(checkpoint, device)
            restored.eval()
            with torch.no_grad():
                restored_logits = model_forward(restored, train_arrays, indices, device)
            if not torch.isfinite(restored_logits).all():
                raise RuntimeError("smoke restored logits are non-finite")
            status.update({
                "status": "PASS_N72R11_TRAINING_SMOKE",
                "resource_censored_development": bool(args.resource_censored),
                "device": str(device),
                "example_count": count,
                "loss": float(loss.detach().cpu()),
                "loss_components": {key: float(value.cpu()) for key, value in components.items()},
                "checkpoint": str(checkpoint),
                "checkpoint_sha256": sha256_file(checkpoint),
                "train_summary": train_summary,
                "validation_summary": validation_summary,
                "finished_at_utc": now_utc(),
            })
        elif phase == "bootstrap":
            audit = {
                **stage_base("N72R11-07-TRAINING-INPUT-AUDIT"),
                "status": "PASS_N72R11_TRAINING_INPUT_AUDIT",
                "resource_censored_development": bool(args.resource_censored),
                "train_summary": train_summary,
                "validation_summary": validation_summary,
                "fixed_sequence_split": "12 train / 6 validation from event_protocol; no frame randomization",
                "labels": "offline box-IoU labels only; not runtime inputs",
                "finished_at_utc": now_utc(),
            }
            atomic_json(STAGE_07, audit)
            model = N72R11TemporalIdentityModel(**model_config()).to(device)
            history, validation = train_epochs(model, train_arrays, validation_arrays, device=device, epochs=BOOTSTRAP_EPOCHS, batch_size=int(args.batch_size), learning_rate=LEARNING_RATE, phase="bootstrap")
            checkpoint = args.output_checkpoint or (OUTPUT_ROOT / "v3_bootstrap.pt")
            atomic_torch(checkpoint, checkpoint_payload(model, phase="bootstrap", history=history, train_summary=train_summary, validation_summary=validation))
            status.update({
                "status": "PASS_N72R11_BOOTSTRAP_TRAINING",
                "device": str(device),
                "checkpoint": str(checkpoint),
                "checkpoint_sha256": sha256_file(checkpoint),
                "train_summary": train_summary,
                "validation_summary": validation,
                "history": history,
                "finished_at_utc": now_utc(),
            })
        elif phase == "self_rollout":
            checkpoint = bootstrap_checkpoint
            model, payload = load_checkpoint(checkpoint, device)
            rollout = run_self_rollout(model, validation_arrays, validation_metadata, device)
            output = OUTPUT_ROOT / "causal_self_rollout.json"
            atomic_json(output, {**rollout, "checkpoint": str(checkpoint), "checkpoint_sha256": sha256_file(checkpoint), "created_at_utc": now_utc()})
            status.update({
                **{key: value for key, value in rollout.items() if key != "records"},
                "device": str(device),
                "checkpoint": str(checkpoint),
                "checkpoint_sha256": sha256_file(checkpoint),
                "output": str(output),
                "output_sha256": sha256_file(output),
                "finished_at_utc": now_utc(),
            })
        else:
            checkpoint = bootstrap_checkpoint
            model, bootstrap_payload = load_checkpoint(checkpoint, device)
            history, validation = train_epochs(model, train_arrays, validation_arrays, device=device, epochs=FINETUNE_EPOCHS, batch_size=int(args.batch_size), learning_rate=FINETUNE_LEARNING_RATE, phase="finetune")
            output = args.output_checkpoint or (OUTPUT_ROOT / "v3_finetuned.pt")
            atomic_torch(output, checkpoint_payload(model, phase="finetune", history=history, train_summary=train_summary, validation_summary=validation))
            bootstrap_validation = dict(bootstrap_payload.get("validation_summary", {}))
            selected = "finetuned" if float(validation.get("loss", float("inf"))) < float(bootstrap_validation.get("loss", float("inf"))) else "bootstrap"
            selected_path = output if selected == "finetuned" else checkpoint
            status.update({
                "status": "PASS_N72R11_FINETUNE",
                "device": str(device),
                "bootstrap_checkpoint": str(checkpoint),
                "bootstrap_checkpoint_sha256": sha256_file(checkpoint),
                "finetuned_checkpoint": str(output),
                "finetuned_checkpoint_sha256": sha256_file(output),
                "bootstrap_validation": bootstrap_validation,
                "finetuned_validation": validation,
                "checkpoint_selection": "minimum fixed validation loss",
                "selected_checkpoint": str(selected_path),
                "selected_phase": selected,
                "history": history,
                "finished_at_utc": now_utc(),
            })
        atomic_json(stage_path, status)
        print(json.dumps(status, sort_keys=True))
        return 0
    except Exception as exc:
        failure = OUTPUT_ROOT / "attempts" / f"{phase}_failure_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.json"
        payload = {**status, "status": f"FAIL_N72R11_{phase.upper()}", "error_type": type(exc).__name__, "error": str(exc), "traceback": traceback.format_exc(), "finished_at_utc": now_utc(), "failure_artifact": str(failure)}
        atomic_json(failure, payload)
        atomic_json(stage_path, payload)
        print(json.dumps({"status": payload["status"], "failure_artifact": str(failure), "error": str(exc)}, sort_keys=True))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
