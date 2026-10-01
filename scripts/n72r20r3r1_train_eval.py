#!/usr/bin/env python3
"""Train and evaluate the N72R20R3R1 frozen-backbone verifier.

This script owns the reference-only training index, sequence-held-out model
training, static metrics, calibration, and the terminal static decision.  It
does not run SAM3, read DanceTrack val/test, call the association solver, or
perform causal memory updates.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import random
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch
from scipy.special import expit

from sam3_intermot.identity_verification.explicit_none_verifier import ExplicitNoneVerifier
from scripts.n72r20r3_common import R2_ASSET_ROOT, ROOT, SEQUENCES, read_zstd_jsonl


TRAINING_DIR = ROOT / "outputs/N72R20R3R1/training"
OUTPUT_DIR = ROOT / "outputs/N72R20R3R1"
DATASET_ROOT = Path("/data3/liuyeqiang/InterMOT_N72R16_assets/dataset")
SEEDS = (720301, 720302, 720303)
BOOTSTRAP_SEED = 720311
SEQUENCE_COUNT = len(SEQUENCES)
MAX_EPOCHS = 25
PATIENCE = 5
BATCH_SIZE = 1024


@dataclass(frozen=True)
class Example:
    sequence: str
    frame: int
    condition: str
    anchor_ref: int
    state_ref: int
    candidate_offsets: tuple[int, ...]
    candidate_uids: tuple[str, ...]
    label: int
    taxonomy: str
    candidate_set_present: bool
    context: tuple[float, float, float, float]


@dataclass
class Dataset:
    examples: list[Example]
    anchors: np.ndarray
    states: np.ndarray
    candidate_memmaps: dict[str, np.memmap]
    candidate_counts: dict[str, int]

    def close(self) -> None:
        for value in self.candidate_memmaps.values():
            mmap = getattr(value, "_mmap", None)
            if mmap is not None:
                mmap.close()


MODEL_SPECS = (
    ("V1_PAIRWISE_THRESHOLD_NONE", "V1_PAIRWISE", "A2_DUAL_STATE"),
    ("V2_ANCHOR_ONLY", "V2_EXPLICIT_NONE", "A0_ANCHOR_ONLY"),
    ("V2_LEARNED_STATE_ONLY", "V2_EXPLICIT_NONE", "A1_LEARNED_STATE_ONLY"),
    ("V2_DUAL_STATE", "V2_EXPLICIT_NONE", "A2_DUAL_STATE"),
)


def _json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def _seed_everything(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_dataset(training_dir: Path = TRAINING_DIR, asset_root: Path = R2_ASSET_ROOT) -> Dataset:
    manifest = _json(training_dir / "training_index_manifest.json")
    if manifest.get("candidate_features_copied") is not False:
        raise RuntimeError("training index copied candidate features")
    if manifest.get("runtime_future_gt_used") is not False or manifest.get("posthoc_gt_used") is not True:
        raise RuntimeError("training index provenance is invalid")
    anchors = np.load(training_dir / "anchor_vectors.float32.npy", mmap_mode="r")
    states = np.load(training_dir / "state_vectors.float32.npy", mmap_mode="r")
    examples: list[Example] = []
    with (training_dir / "training_index.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            candidates = list(row["candidate_axis"])
            context_row = row["runtime_context"]
            context = (
                math.log1p(float(context_row.get("candidate_count", len(candidates)))),
                float(context_row.get("frames_since_human_initialization", 0.0)) / 100.0,
                float(context_row.get("frames_since_last_memory_write", 0.0)) / 100.0,
                float(context_row.get("learned_state_human_anchor_cosine", 1.0)),
            )
            label = row["training_label"]
            candidate_index = label.get("candidate_index")
            examples.append(
                Example(
                    sequence=str(row["sequence"]),
                    frame=int(row["frame"]),
                    condition=str(row["state_condition"]),
                    anchor_ref=int(row["anchor_ref"]),
                    state_ref=int(row["state_ref"]),
                    candidate_offsets=tuple(int(item["embedding_offset"]) for item in candidates),
                    candidate_uids=tuple(str(item["candidate_uid"]) for item in candidates),
                    label=-1 if candidate_index is None else int(candidate_index),
                    taxonomy=str(row["taxonomy_posthoc"]),
                    candidate_set_present=bool(candidate_index is not None),
                    context=context,
                )
            )
    if sorted({item.sequence for item in examples}) != sorted(SEQUENCES):
        raise RuntimeError("training index sequence axis is not the frozen R3 axis")
    # Keep only the already-sealed float16 candidate tape in a transient RAM
    # cache.  This avoids re-reading the same offsets for every epoch/model;
    # it is never written to outputs or staged as a new asset.
    candidate_memmaps: dict[str, np.ndarray] = {}
    candidate_counts: dict[str, int] = {}
    for sequence in SEQUENCES:
        sequence_root = asset_root / "candidates" / sequence
        index = _json(sequence_root / "index.json")
        count = int(index["embedding_count"])
        sealed = np.memmap(
            sequence_root / "embeddings.f16", mode="r", dtype=np.float16, shape=(count, 512)
        )
        candidate_memmaps[sequence] = np.array(sealed, copy=True)
        del sealed
        candidate_counts[sequence] = count
    if len(examples) != int(manifest["rows"]):
        raise RuntimeError("training index row count changed")
    return Dataset(examples, anchors, states, candidate_memmaps, candidate_counts)


def _sequence_counts(examples: Sequence[Example]) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for item in examples:
        counts[item.sequence] += 1
    return dict(counts)


def _batch(dataset: Dataset, examples: Sequence[Example], device: torch.device) -> tuple[torch.Tensor, ...]:
    batch_size = len(examples)
    max_candidates = max((len(item.candidate_offsets) for item in examples), default=0)
    max_candidates = max(1, max_candidates)
    anchor = np.asarray([dataset.anchors[item.anchor_ref] for item in examples], dtype=np.float32)
    state = np.asarray([dataset.states[item.state_ref] for item in examples], dtype=np.float32)
    candidates = np.zeros((batch_size, max_candidates, 512), dtype=np.float32)
    mask = np.zeros((batch_size, max_candidates), dtype=bool)
    labels = np.full((batch_size,), max_candidates, dtype=np.int64)
    context = np.asarray([item.context for item in examples], dtype=np.float32)
    for index, item in enumerate(examples):
        count = len(item.candidate_offsets)
        if count:
            offsets = np.asarray(item.candidate_offsets, dtype=np.int64)
            candidates[index, :count] = np.asarray(dataset.candidate_memmaps[item.sequence][offsets], dtype=np.float32)
            mask[index, :count] = True
        if item.label >= 0:
            if item.label >= count:
                raise RuntimeError(f"label outside candidate axis: {item.sequence}:{item.frame}")
            labels[index] = item.label
    return (
        torch.from_numpy(anchor).to(device=device),
        torch.from_numpy(state).to(device=device),
        torch.from_numpy(candidates).to(device=device),
        torch.from_numpy(mask).to(device=device),
        torch.from_numpy(labels).to(device=device),
        torch.from_numpy(context).to(device=device),
    )


def _weighted_mean(values: torch.Tensor, examples: Sequence[Example], counts: Mapping[str, int]) -> torch.Tensor:
    weights = torch.tensor(
        [1.0 / float(counts[item.sequence]) for item in examples], dtype=values.dtype, device=values.device
    )
    return (values * weights).sum() / weights.sum().clamp_min(1.0e-8)


def _loss(model: ExplicitNoneVerifier, batch: tuple[torch.Tensor, ...], examples: Sequence[Example], counts: Mapping[str, int]) -> torch.Tensor:
    anchor, state, candidates, mask, labels, context = batch
    output = model(anchor, state, candidates, mask, context)
    if model.mode == "V2_EXPLICIT_NONE":
        losses = torch.nn.functional.cross_entropy(output["logits"], labels, reduction="none")
        return _weighted_mean(losses, examples, counts)
    candidate_logits = output["candidate_logits"]
    targets = torch.zeros_like(candidate_logits)
    for index, item in enumerate(examples):
        if item.label >= 0:
            targets[index, item.label] = 1.0
    raw = torch.nn.functional.binary_cross_entropy_with_logits(candidate_logits, targets, reduction="none")
    raw = raw.masked_fill(~mask, 0.0)
    losses = raw.sum(dim=1) / mask.sum(dim=1).clamp_min(1).to(raw.dtype)
    # Pairwise training is intentionally frame-level: every candidate in an
    # absent frame remains a negative hard example, while sequence totals are
    # balanced by the outer weight.
    return _weighted_mean(losses, examples, counts)


def _run_epoch(
    model: ExplicitNoneVerifier,
    dataset: Dataset,
    examples: Sequence[Example],
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    seed: int,
) -> float:
    training = optimizer is not None
    model.train(training)
    order = list(range(len(examples)))
    if training:
        random.Random(seed).shuffle(order)
    counts = _sequence_counts(examples)
    losses: list[float] = []
    for start in range(0, len(order), BATCH_SIZE):
        batch_examples = [examples[index] for index in order[start : start + BATCH_SIZE]]
        batch = _batch(dataset, batch_examples, device)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            loss = _loss(model, batch, batch_examples, counts)
        if training:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
        losses.append(float(loss.detach().cpu().item()))
    return float(np.mean(losses)) if losses else 0.0


def _raw_predictions(
    model: ExplicitNoneVerifier,
    dataset: Dataset,
    examples: Sequence[Example],
    device: torch.device,
    threshold: float | None = None,
) -> list[dict[str, Any]]:
    model.eval()
    output_rows: list[dict[str, Any]] = []
    for start in range(0, len(examples), BATCH_SIZE):
        batch_examples = list(examples[start : start + BATCH_SIZE])
        batch = _batch(dataset, batch_examples, device)
        anchor, state, candidates, mask, labels, context = batch
        with torch.no_grad():
            output = model(anchor, state, candidates, mask, context)
        candidate_logits = output["candidate_logits"].detach().cpu().numpy()
        if model.mode == "V2_EXPLICIT_NONE":
            probabilities = torch.softmax(output["logits"], dim=1).detach().cpu().numpy()
        else:
            probabilities = expit(candidate_logits)
        for row_index, item in enumerate(batch_examples):
            count = len(item.candidate_uids)
            candidate_values = probabilities[row_index, :count]
            if model.mode == "V2_EXPLICIT_NONE":
                none_probability = float(probabilities[row_index, -1])
                chosen_class = int(np.argmax(probabilities[row_index]))
                predicted_index = chosen_class if chosen_class < count else -1
                class_probability = float(probabilities[row_index, item.label if item.label >= 0 else -1])
                present_probability = 1.0 - none_probability
            else:
                none_probability = 1.0 if count == 0 else float(1.0 - np.max(candidate_values))
                if count == 0:
                    predicted_index = -1
                else:
                    predicted_index = int(np.argmax(candidate_values))
                if predicted_index >= 0 and threshold is not None and float(candidate_values[predicted_index]) < float(threshold):
                    predicted_index = -1
                if item.label >= 0 and count:
                    class_probability = float(candidate_values[item.label])
                else:
                    class_probability = none_probability
                present_probability = 1.0 - none_probability
            candidate_order = np.argsort(-candidate_values, kind="stable") if count else np.asarray([], dtype=np.int64)
            rank = None
            if item.label >= 0 and count:
                found = np.where(candidate_order == item.label)[0]
                rank = None if not len(found) else int(found[0]) + 1
            output_rows.append(
                {
                    "sequence": item.sequence,
                    "frame": item.frame,
                    "state_condition": item.condition,
                    "taxonomy": item.taxonomy,
                    "label_index": item.label,
                    "candidate_count": count,
                    "predicted_index": int(predicted_index),
                    "predicted_uid": None if predicted_index < 0 else item.candidate_uids[predicted_index],
                    "none_probability": none_probability,
                    "present_probability": present_probability,
                    "class_probability": class_probability,
                    "candidate_rank": rank,
                    "runtime_future_gt_used": False,
                    "candidate_created": False,
                }
            )
    return output_rows


def _auc(scores: Sequence[float], labels: Sequence[bool]) -> float | None:
    positive = sum(bool(value) for value in labels)
    negative = len(labels) - positive
    if positive == 0 or negative == 0:
        return None
    order = sorted(range(len(scores)), key=lambda index: float(scores[index]))
    rank_sum = 0.0
    cursor = 0
    rank = 1
    while cursor < len(order):
        end = cursor + 1
        while end < len(order) and float(scores[order[end]]) == float(scores[order[cursor]]):
            end += 1
        average = (rank + end - 1) / 2.0
        rank_sum += sum(average for position in order[cursor:end] if labels[position])
        rank += end - cursor
        cursor = end
    return float((rank_sum - positive * (positive + 1) / 2.0) / (positive * negative))


def _auprc(scores: Sequence[float], labels: Sequence[bool]) -> float | None:
    positive = sum(bool(value) for value in labels)
    if positive == 0:
        return None
    order = sorted(range(len(scores)), key=lambda index: -float(scores[index]))
    tp = fp = 0
    previous_recall = 0.0
    area = 0.0
    cursor = 0
    while cursor < len(order):
        end = cursor + 1
        while end < len(order) and float(scores[order[end]]) == float(scores[order[cursor]]):
            end += 1
        for position in order[cursor:end]:
            if labels[position]:
                tp += 1
            else:
                fp += 1
        recall = tp / positive
        precision = tp / max(1, tp + fp)
        area += (recall - previous_recall) * precision
        previous_recall = recall
        cursor = end
    return float(area)


def _ece(scores: Sequence[float], labels: Sequence[bool], bins: int = 10) -> float | None:
    if not scores:
        return None
    values = np.asarray(scores, dtype=np.float64)
    truth = np.asarray(labels, dtype=np.float64)
    total = float(len(values))
    error = 0.0
    for index in range(bins):
        low = index / bins
        high = (index + 1) / bins
        selected = (values >= low) & (values <= high if index == bins - 1 else values < high)
        if not selected.any():
            continue
        error += float(selected.sum()) / total * abs(float(values[selected].mean()) - float(truth[selected].mean()))
    return float(error)


def _div(numerator: float, denominator: float) -> float | None:
    return None if denominator == 0 else float(numerator / denominator)


def metrics(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"rows": 0}
    present = [int(item["label_index"]) >= 0 for item in rows]
    predicted = [item["predicted_index"] >= 0 for item in rows]
    correct = [predicted[index] and item["predicted_index"] == item["label_index"] for index, item in enumerate(rows)]
    absent = [not value for value in present]
    p0 = [item["taxonomy"] == "P0_TARGET_ABSENT" for item in rows]
    p1 = [str(item["taxonomy"]).startswith("P1") for item in rows]
    neg_count = sum(absent)
    pos_count = sum(present)
    predicted_count = sum(predicted)
    absent_correct = sum((not predicted[index]) and absent[index] for index in range(len(rows)))
    true_probs = [float(item["class_probability"]) for item in rows]
    binary_scores = [float(item["present_probability"]) for item in rows]
    binary_labels = [bool(value) for value in present]
    nll = float(np.mean([-math.log(max(1.0e-12, value)) for value in true_probs]))
    brier = float(np.mean([(binary_scores[index] - float(binary_labels[index])) ** 2 for index in range(len(rows))]))
    result = {
        "rows": len(rows),
        "candidate_set_negative_count": neg_count,
        "candidate_set_positive_count": pos_count,
        "predicted_present_count": predicted_count,
        "false_present_count": sum(predicted[index] and absent[index] for index in range(len(rows))),
        "false_present_rate": _div(sum(predicted[index] and absent[index] for index in range(len(rows))), neg_count),
        "open_set_correct_identification_count": sum(correct),
        "open_set_correct_identification_recall": _div(sum(correct), pos_count),
        "presence_precision": _div(sum(predicted[index] and present[index] for index in range(len(rows))), predicted_count),
        "presence_recall": _div(sum(predicted[index] and present[index] for index in range(len(rows))), pos_count),
        "false_absent_rate": _div(sum((not predicted[index]) and present[index] for index in range(len(rows))), pos_count),
        "conditional_candidate_accuracy_given_present": _div(sum(correct), predicted_count),
        "candidate_top1_accuracy_on_present": _div(sum(correct), pos_count),
        "none_recall_on_absent": _div(absent_correct, neg_count),
        "none_precision": _div(absent_correct, sum(not value for value in predicted)),
        "p0_count": sum(p0),
        "p1_count": sum(p1),
        "p0_false_present_rate": _div(sum(predicted[index] for index in range(len(rows)) if p0[index]), sum(p0)),
        "p1_false_present_rate": _div(sum(predicted[index] for index in range(len(rows)) if p1[index]), sum(p1)),
        "p0_false_present_count": sum(predicted[index] for index in range(len(rows)) if p0[index]),
        "p1_false_present_count": sum(predicted[index] for index in range(len(rows)) if p1[index]),
        "mean_candidate_rank_present": float(np.mean([item["candidate_rank"] for item in rows if item["candidate_rank"] is not None])) if any(item["candidate_rank"] is not None for item in rows) else None,
        "mrr_present": float(np.mean([1.0 / item["candidate_rank"] for item in rows if item["candidate_rank"] is not None])) if any(item["candidate_rank"] is not None for item in rows) else None,
        "nll": nll,
        "brier": brier,
        "ece": _ece(binary_scores, binary_labels),
        "auroc_present": _auc(binary_scores, binary_labels),
        "auprc_present": _auprc(binary_scores, binary_labels),
        "runtime_future_gt_used": False,
        "candidate_created": False,
    }
    return result


def per_sequence_metrics(rows: Sequence[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["sequence"])].append(row)
    result = {}
    for sequence in SEQUENCES:
        result[sequence] = metrics(grouped.get(sequence, []))
        result[sequence]["sequence"] = sequence
        result[sequence]["catastrophic"] = bool(
            result[sequence].get("false_present_rate") is not None
            and result[sequence]["false_present_rate"] > 0.20
            or result[sequence].get("open_set_correct_identification_recall") is not None
            and result[sequence]["open_set_correct_identification_recall"] < 0.01
        )
    return result


def aggregate_seed_predictions(prediction_sets: Sequence[Sequence[dict[str, Any]]]) -> list[dict[str, Any]]:
    if not prediction_sets:
        return []
    by_key: dict[tuple[str, int, str], list[dict[str, Any]]] = defaultdict(list)
    for predictions in prediction_sets:
        for row in predictions:
            by_key[(str(row["sequence"]), int(row["frame"]), str(row["state_condition"]))].append(row)
    result: list[dict[str, Any]] = []
    for key in sorted(by_key):
        values = by_key[key]
        reference = dict(values[0])
        reference["none_probability"] = float(np.mean([float(item["none_probability"]) for item in values]))
        reference["present_probability"] = float(np.mean([float(item["present_probability"]) for item in values]))
        reference["class_probability"] = float(np.mean([float(item["class_probability"]) for item in values]))
        # Candidate identity is only used for the final prediction.  A seed
        # disagreement is treated as NONE unless a strict majority agrees.
        candidate_votes = [item["predicted_uid"] for item in values if item["predicted_uid"] is not None]
        if candidate_votes:
            counts = {uid: candidate_votes.count(uid) for uid in set(candidate_votes)}
            uid, votes = max(counts.items(), key=lambda item: (item[1], item[0]))
            if votes > len(values) / 2:
                reference["predicted_uid"] = uid
                reference["predicted_index"] = next(item["predicted_index"] for item in values if item["predicted_uid"] == uid)
            else:
                reference["predicted_uid"] = None
                reference["predicted_index"] = -1
        else:
            reference["predicted_uid"] = None
            reference["predicted_index"] = -1
        result.append(reference)
    return result


def calibrate_v1(predictions: Sequence[dict[str, Any]]) -> tuple[float, dict[str, Any]]:
    scores = np.asarray([float(item["present_probability"]) for item in predictions], dtype=np.float64)
    if len(scores) > 1:
        thresholds = sorted({0.0, 1.0, *np.quantile(scores, np.linspace(0.0, 1.0, 201)).tolist()})
    else:
        thresholds = [0.0, 1.0]
    labels = np.asarray([int(item["label_index"]) >= 0 for item in predictions], dtype=bool)
    base_predicted = np.asarray([int(item["predicted_index"]) >= 0 for item in predictions], dtype=bool)
    negative = ~labels
    positive_count = int(labels.sum())
    negative_count = int(negative.sum())
    candidates: list[tuple[float, float, float]] = []
    for threshold in thresholds:
        accepted = base_predicted & (scores >= float(threshold))
        false_present_rate = float((accepted & negative).sum() / negative_count) if negative_count else 0.0
        open_recall = float((accepted & labels).sum() / positive_count) if positive_count else 0.0
        candidates.append((float(threshold), false_present_rate, open_recall))
    feasible = [item for item in candidates if item[1] <= 0.02]
    if feasible:
        threshold, _, _ = max(feasible, key=lambda item: (item[2], -item[0]))
    else:
        threshold, _, _ = min(candidates, key=lambda item: (item[1], -float(item[0])))
    cloned = []
    for item in predictions:
        value = dict(item)
        if value["predicted_index"] >= 0 and float(value["present_probability"]) < threshold:
            value["predicted_index"] = -1
            value["predicted_uid"] = None
        cloned.append(value)
    report = metrics(cloned)
    return threshold, {
        "threshold": threshold,
        "calibration_rows": len(predictions),
        "calibration_metrics": report,
        "calibrated_only_on_internal_sequence": True,
    }


def fit_model(
    dataset: Dataset,
    train_examples: Sequence[Example],
    validation_examples: Sequence[Example],
    *,
    mode: str,
    variant: str,
    seed: int,
    device: torch.device,
) -> tuple[ExplicitNoneVerifier, dict[str, Any]]:
    _seed_everything(seed)
    model = ExplicitNoneVerifier(mode=mode, state_variant=variant, projection_dim=32, hidden_dim=48).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.0e-3, weight_decay=1.0e-4)
    best_state = copy.deepcopy(model.state_dict())
    best_validation = float("inf")
    history: list[dict[str, Any]] = []
    stale = 0
    for epoch in range(1, MAX_EPOCHS + 1):
        train_loss = _run_epoch(model, dataset, train_examples, device, optimizer, seed + epoch)
        with torch.no_grad():
            validation_loss = _run_epoch(model, dataset, validation_examples, device, None, seed)
        history.append({"epoch": epoch, "train_loss": train_loss, "validation_loss": validation_loss})
        if validation_loss < best_validation - 1.0e-5:
            best_validation = validation_loss
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
        if stale >= PATIENCE:
            break
    model.load_state_dict(best_state)
    return model, {
        "epochs_run": len(history),
        "best_validation_loss": best_validation,
        "history": history,
        "trainable_parameters": model.trainable_parameters,
        "projection_dim": model.projection_dim,
        "hidden_dim": model.hidden_dim,
    }


def save_checkpoint(
    model: ExplicitNoneVerifier,
    path: Path,
    *,
    source_commit: str,
    heldout: str,
    training_sequences: Sequence[str],
    internal_validation: Sequence[str],
    seed: int,
    normalization: Mapping[str, Any],
) -> str:
    payload = {
        "stage": "N72R20R3R1",
        "architecture": model.mode,
        "state_variant": model.state_variant,
        "source_commit": source_commit,
        "training_sequences": list(training_sequences),
        "heldout_sequence": heldout,
        "internal_validation_sequences": list(internal_validation),
        "seed": int(seed),
        "osnet_sha256": "2809d3227f7d078f6045f7feb874a34d0684f0e0057b264b99adccf7d4519154",
        "n72r18_gru_sha256": "94ba9c44e177b254ee54984195e839f37c78a648c010ffc723dd3f1d553976a2",
        "feature_schema": "frozen 512-D candidate/anchor/state vectors; runtime context whitelist v1",
        "normalization": dict(normalization),
        "trainable_parameters": model.trainable_parameters,
        "runtime_future_gt_used": False,
        "candidate_creation": False,
        "state_dict": model.state_dict(),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)
    return _sha256(path)


def _source_commit() -> str:
    import subprocess

    return subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, check=True, text=True, stdout=subprocess.PIPE).stdout.strip()


def _select_development_variant(dataset: Dataset, device: torch.device, output_dir: Path) -> dict[str, Any]:
    heldout = SEQUENCES[-1]
    internal_validation = [SEQUENCES[-2]]
    train_sequences = [sequence for sequence in SEQUENCES[:-1] if sequence not in internal_validation]
    train_examples = [item for item in dataset.examples if item.sequence in train_sequences]
    validation_examples = [item for item in dataset.examples if item.sequence in internal_validation]
    comparison: list[dict[str, Any]] = []
    for name, mode, variant in MODEL_SPECS:
        model, fit = fit_model(dataset, train_examples, validation_examples, mode=mode, variant=variant, seed=SEEDS[0], device=device)
        validation_predictions = _raw_predictions(model, dataset, validation_examples, device)
        threshold = None
        calibration = None
        if mode == "V1_PAIRWISE":
            threshold, calibration = calibrate_v1(validation_predictions)
            validation_predictions = _raw_predictions(model, dataset, validation_examples, device, threshold=threshold)
        report = metrics(validation_predictions)
        comparison.append(
            {
                "model": name,
                "mode": mode,
                "state_variant": variant,
                "fit": fit,
                "threshold": threshold,
                "calibration": calibration,
                "validation_metrics": report,
                "selection_sequences": {"train": train_sequences, "validation": internal_validation, "development_holdout_not_used": heldout},
            }
        )
    def score(item: Mapping[str, Any]) -> tuple[float, float, float]:
        report = item["validation_metrics"]
        fpr = float(report.get("false_present_rate") or 1.0)
        recall = float(report.get("open_set_correct_identification_recall") or 0.0)
        feasible = 1.0 if fpr <= 0.02 else 0.0
        return feasible, recall, -fpr

    # V1 is a required closed-set/threshold baseline.  The scientific
    # primary architecture must be selected within the explicit-NONE V2
    # family, using only the development sequence-level validation result.
    v2_comparison = [item for item in comparison if item["mode"] == "V2_EXPLICIT_NONE"]
    selected = max(v2_comparison, key=lambda item: (float(item["validation_metrics"].get("open_set_correct_identification_recall") or 0.0), -float(item["validation_metrics"].get("false_present_rate") or 1.0)))
    result = {
        "stage": "N72R20R3R1",
        "status": "PASS_MODEL_SELECTION_COMPLETE",
        "selection_protocol": "one development sequence held out; six sequence training; one sequence internal validation; no formal held-out scores",
        "comparisons": comparison,
        "selected_model": selected["model"],
        "selected_mode": selected["mode"],
        "selected_state_variant": selected["state_variant"],
        "frozen_hyperparameters": {"projection_dim": 32, "hidden_dim": 48, "max_epochs": MAX_EPOCHS, "patience": PATIENCE, "batch_size": BATCH_SIZE, "seed": SEEDS[0]},
    }
    _write_json(output_dir / "training" / "model_selection.json", result)
    return result


def _static_gate(report: Mapping[str, Any]) -> dict[str, Any]:
    catastrophic = [sequence for sequence, item in report["per_sequence"].items() if item.get("catastrophic")]
    checks = {
        "S1_pooled_negative_fpr_le_0.02": (report.get("false_present_rate") is not None and report["false_present_rate"] <= 0.02),
        "S2_pooled_open_recall_ge_0.60": (report.get("open_set_correct_identification_recall") is not None and report["open_set_correct_identification_recall"] >= 0.60),
        "S3_macro_recall_ge_0.40": (report.get("macro_open_set_correct_identification_recall") is not None and report["macro_open_set_correct_identification_recall"] >= 0.40),
        "S4_macro_fpr_le_0.05": (report.get("macro_false_present_rate") is not None and report["macro_false_present_rate"] <= 0.05),
        "S5_runtime_gt_false": report.get("runtime_future_gt_used") is True,
        "S6_all_eight_sequences_reported": len(report.get("per_sequence", {})) == SEQUENCE_COUNT,
        "S7_no_candidate_creation": report.get("candidate_created") is False,
    }
    return {"checks": checks, "pass": bool(all(checks.values())), "catastrophic_sequences": catastrophic, "cross_sequence_warning": len(catastrophic) >= 2}


def _aggregate_report(predictions: Sequence[dict[str, Any]]) -> dict[str, Any]:
    report = metrics(predictions)
    sequence = per_sequence_metrics(predictions)
    recalls = [item["open_set_correct_identification_recall"] for item in sequence.values() if item.get("open_set_correct_identification_recall") is not None]
    fprs = [item["false_present_rate"] for item in sequence.values() if item.get("false_present_rate") is not None]
    report["per_sequence"] = sequence
    report["macro_open_set_correct_identification_recall"] = float(np.mean(recalls)) if recalls else None
    report["macro_false_present_rate"] = float(np.mean(fprs)) if fprs else None
    report["runtime_future_gt_used"] = all(not bool(item.get("runtime_future_gt_used")) for item in predictions)
    report["candidate_created"] = any(bool(item.get("candidate_created")) for item in predictions)
    report["cross_sequence_warning"] = sum(bool(item.get("catastrophic")) for item in sequence.values()) >= 2
    report["static_gate"] = _static_gate(report)
    return report


def _bootstrap(predictions: Sequence[dict[str, Any]], repetitions: int = 2000) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in predictions:
        grouped[str(row["sequence"])].append(row)
    sequences = [sequence for sequence in SEQUENCES if sequence in grouped]
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    # Precompute cluster sufficient statistics.  Re-running the full frame
    # metric function 2,000 times is equivalent but unnecessarily expensive.
    clusters = []
    for sequence in sequences:
        report = metrics(grouped[sequence])
        clusters.append(
            {
                "negative": int(report.get("candidate_set_negative_count", 0)),
                "false_present": int(report.get("false_present_count", 0)),
                "positive": int(report.get("candidate_set_positive_count", 0)),
                "correct": int(report.get("open_set_correct_identification_count", 0)),
                "fpr": float(report.get("false_present_rate") or 0.0),
                "recall": float(report.get("open_set_correct_identification_recall") or 0.0),
            }
        )
    values = {"negative_fpr": [], "open_recall": [], "macro_recall": []}
    for _ in range(repetitions):
        sampled = rng.integers(0, len(clusters), size=len(clusters))
        selected = [clusters[int(index)] for index in sampled]
        negative = sum(item["negative"] for item in selected)
        false_present = sum(item["false_present"] for item in selected)
        positive = sum(item["positive"] for item in selected)
        correct = sum(item["correct"] for item in selected)
        values["negative_fpr"].append(float(false_present / negative) if negative else 0.0)
        values["open_recall"].append(float(correct / positive) if positive else 0.0)
        values["macro_recall"].append(float(np.mean([item["recall"] for item in selected])))
    return {
        "repetitions": repetitions,
        "seed": BOOTSTRAP_SEED,
        "unit": "sequence cluster",
        "intervals_95": {
            name: [float(np.percentile(values[name], 2.5)), float(np.percentile(values[name], 97.5))]
            for name in values
        },
        "means": {name: float(np.mean(value)) for name, value in values.items()},
    }


def _baseline_report() -> dict[str, Any]:
    data = _json(ROOT / "outputs/N72R20R3/presence/loso_results.json")
    records = [
        dict(row)
        for row in data["records"]
        if str(row.get("method")) == "LOGISTIC_PRESENCE" and str(row.get("policy_id", "")).endswith("B4_LOGISTIC_PRESENCE")
    ]
    if not records:
        raise RuntimeError("R3 B4 records are missing")
    per_sequence: dict[str, dict[str, Any]] = {}
    for sequence in SEQUENCES:
        selected = [row for row in records if row.get("sequence") == sequence]
        def _mean_defined(key: str) -> float | None:
            values = [row.get(key) for row in selected if row.get(key) is not None]
            return None if not values else float(np.mean(values))
        per_sequence[sequence] = {
            "false_present_rate": _mean_defined("false_present_rate"),
            "open_set_correct_identification_recall": _mean_defined("open_set_correct_identification_recall"),
            "p0_false_present_rate": _mean_defined("p0_false_present_rate"),
            "p1_false_present_rate": _mean_defined("p1_false_present_rate"),
            "rows": int(sum(row["rows"] for row in selected)),
        }
    pooled = data["aggregates"]
    pooled_rows = [
        row for row in pooled if str(row.get("policy_id", "")).endswith("B4_LOGISTIC_PRESENCE")
    ]
    # The two state conditions have separate pooled values; retain both and a
    # conservative mean for the comparison table.
    return {
        "method": "V0_R3_LOGISTIC_BASELINE",
        "source": "outputs/N72R20R3/presence/loso_results.json",
        "records": len(records),
        "pooled_by_condition": [row["pooled"] for row in pooled_rows],
        "per_sequence": per_sequence,
        "r3_final_decision_unchanged": True,
    }


def run_formal(
    dataset: Dataset,
    selection: Mapping[str, Any],
    device: torch.device,
    output_dir: Path,
    *,
    model_specs: Sequence[tuple[str, str, str]] = MODEL_SPECS,
    seeds: Sequence[int] = SEEDS,
    write_outputs: bool = True,
) -> dict[str, Any]:
    source_commit = _source_commit()
    all_results: list[dict[str, Any]] = []
    prediction_archive: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    checkpoint_records: list[dict[str, Any]] = []
    for name, mode, variant in model_specs:
        for seed in seeds:
            seed_metrics: list[dict[str, Any]] = []
            for heldout in SEQUENCES:
                training_sequences = [sequence for sequence in SEQUENCES if sequence != heldout]
                internal_validation = [training_sequences[-1]]
                fit_sequences = [sequence for sequence in training_sequences if sequence not in internal_validation]
                train_examples = [item for item in dataset.examples if item.sequence in fit_sequences]
                validation_examples = [item for item in dataset.examples if item.sequence in internal_validation]
                heldout_examples = [item for item in dataset.examples if item.sequence == heldout]
                heldout_offset = SEQUENCES.index(heldout) * 17
                model, fit = fit_model(dataset, train_examples, validation_examples, mode=mode, variant=variant, seed=seed + heldout_offset, device=device)
                calibration = None
                threshold = None
                calibration_predictions = _raw_predictions(model, dataset, validation_examples, device)
                if mode == "V1_PAIRWISE":
                    threshold, calibration = calibrate_v1(calibration_predictions)
                predictions = _raw_predictions(model, dataset, heldout_examples, device, threshold=threshold)
                report = _aggregate_report(predictions)
                report.update({"model": name, "mode": mode, "state_variant": variant, "seed": seed, "heldout_sequence": heldout, "fit": fit, "threshold": threshold, "calibration": calibration, "train_sequences": fit_sequences, "internal_validation_sequences": internal_validation})
                seed_metrics.append(report)
                prediction_archive[name][str(seed)].extend(predictions)
                checkpoint_path = output_dir / "models" / f"{name}__{heldout}__seed{seed}.pt"
                checkpoint_sha = save_checkpoint(
                    model,
                    checkpoint_path,
                    source_commit=source_commit,
                    heldout=heldout,
                    training_sequences=fit_sequences,
                    internal_validation=internal_validation,
                    seed=seed,
                    normalization={"vector_normalization": "unit L2", "context": "candidate_count=log1p; relative gaps/100; anchor-state cosine", "heldout_fit": False},
                )
                checkpoint_records.append({"model": name, "heldout": heldout, "seed": seed, "path": str(checkpoint_path), "sha256": checkpoint_sha, "trainable_parameters": model.trainable_parameters})
            pooled_seed = _aggregate_report(prediction_archive[name][str(seed)])
            pooled_seed.update({"model": name, "seed": seed})
            seed_metrics.append(pooled_seed)
            all_results.extend(seed_metrics)
    seed_summary: dict[str, Any] = {}
    primary_reports: dict[str, Any] = {}
    for name, _, _ in model_specs:
        per_seed = [item for item in all_results if item.get("model") == name and "heldout_sequence" not in item]
        fields = ["false_present_rate", "open_set_correct_identification_recall", "macro_false_present_rate", "macro_open_set_correct_identification_recall", "none_recall_on_absent", "nll", "brier", "ece", "auroc_present", "auprc_present"]
        aggregate_predictions = aggregate_seed_predictions([prediction_archive[name][str(seed)] for seed in seeds])
        aggregate = _aggregate_report(aggregate_predictions)
        primary_reports[name] = aggregate
        seed_summary[name] = {
            "seeds": list(seeds),
            "per_seed": per_seed,
            "mean_std": {
                field: {"mean": float(np.mean([float(item[field]) for item in per_seed if item.get(field) is not None])), "std": float(np.std([float(item[field]) for item in per_seed if item.get(field) is not None]))}
                for field in fields
                if any(item.get(field) is not None for item in per_seed)
            },
            "mean_seed_gate": all(bool(item.get("static_gate", {}).get("pass")) for item in per_seed),
            "three_seed_aggregate": aggregate,
        }
    selected_name = str(selection["selected_model"])
    if selected_name not in prediction_archive:
        selected_name = str(model_specs[0][0])
    selected_predictions = aggregate_seed_predictions([prediction_archive[selected_name][str(seed)] for seed in seeds])
    result = {
        "stage": "N72R20R3R1",
        "status": "PASS_FORMAL_LOSO_COMPLETE",
        "source_commit": source_commit,
        "folds": SEQUENCE_COUNT,
        "heldout_sequences": list(SEQUENCES),
        "seeds": list(seeds),
        "models": [name for name, _, _ in model_specs],
        "records": all_results,
        "primary_reports": primary_reports,
        "checkpoint_records": checkpoint_records,
        "runtime_future_gt_used": False,
        "candidate_created": False,
        "prediction_archive": {name: {seed: rows for seed, rows in values.items()} for name, values in prediction_archive.items()},
    }
    if write_outputs:
        _write_json(output_dir / "training" / "seed_results.json", result)
        _write_json(output_dir / "static_eval" / "loso_results.json", {"stage": "N72R20R3R1", "status": "PASS_FORMAL_LOSO_COMPLETE", "reports": primary_reports, "records": all_results, "runtime_future_gt_used": False, "candidate_created": False})
        _write_json(output_dir / "static_eval" / "per_sequence_results.json", {"stage": "N72R20R3R1", "model": selected_name, "records": per_sequence_metrics(selected_predictions)})
        _write_json(output_dir / "static_eval" / "bootstrap.json", _bootstrap(selected_predictions))
        _write_json(output_dir / "static_eval" / "calibration.json", {"model": selected_name, "metrics": primary_reports[selected_name], "calibration": "V1 threshold fitted on internal sequences only; V2 NONE is learned without held-out tuning"})
    return {"formal": result, "seed_summary": seed_summary, "selected_name": selected_name, "selected_predictions": selected_predictions, "baseline": _baseline_report() if write_outputs else None}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--asset-root", type=Path, default=R2_ASSET_ROOT)
    parser.add_argument("--device", default="cuda:5" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--skip-development", action="store_true")
    parser.add_argument("--only-model", choices=[name for name, _, _ in MODEL_SPECS])
    parser.add_argument("--only-seed", type=int, choices=list(SEEDS))
    parser.add_argument("--worker-output-dir", type=Path)
    args = parser.parse_args()
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "static_eval").mkdir(exist_ok=True)
    (args.output_dir / "models").mkdir(exist_ok=True)
    dataset = load_dataset(TRAINING_DIR, args.asset_root)
    try:
        selection_path = OUTPUT_DIR / "training" / "model_selection.json"
        if args.skip_development and selection_path.exists():
            selection = _json(selection_path)
        else:
            selection = _select_development_variant(dataset, device, args.output_dir)
        selected_specs = [spec for spec in MODEL_SPECS if args.only_model is None or spec[0] == args.only_model]
        selected_seeds = [seed for seed in SEEDS if args.only_seed is None or seed == args.only_seed]
        if not selected_specs or not selected_seeds:
            raise ValueError("worker selection is empty")
        worker_mode = args.worker_output_dir is not None
        worker_dir = args.worker_output_dir if worker_mode else args.output_dir
        if worker_mode:
            worker_dir.mkdir(parents=True, exist_ok=True)
            (worker_dir / "models").mkdir(exist_ok=True)
        formal = run_formal(dataset, selection, device, worker_dir, model_specs=selected_specs, seeds=selected_seeds, write_outputs=not worker_mode)
        if worker_mode:
            _write_json(worker_dir / "worker_result.json", formal["formal"])
        selected = formal["selected_predictions"]
        if not worker_mode:
            selected_report = _aggregate_report(selected)
            _write_json(args.output_dir / "static_eval" / "selected_report.json", selected_report)
            print(json.dumps({"status": "PASS_FORMAL_LOSO_COMPLETE", "selected_model": formal["selected_name"], "static_gate": selected_report["static_gate"], "metrics": {key: selected_report.get(key) for key in ("false_present_rate", "open_set_correct_identification_recall", "macro_false_present_rate", "macro_open_set_correct_identification_recall")}}, sort_keys=True))
        else:
            print(json.dumps({"status": "PASS_WORKER_COMPLETE", "model": selected_specs[0][0], "seed": selected_seeds[0]}, sort_keys=True))
    finally:
        dataset.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
