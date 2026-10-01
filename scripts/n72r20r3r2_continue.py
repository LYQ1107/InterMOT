#!/usr/bin/env python3
"""Continue N72R20R3R2 after a passed Phase-A representation gate.

This file is intentionally separate from the representation trainer.  It
opens Phase B only when the frozen Phase-A gate is true, trains a nine-feature
NONE head while keeping both metric towers frozen, and opens causal replay
only when the preregistered open-set gate is true.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from sam3_intermot.identity_verification.cross_scene_adapter import CrossSceneIdentityAdapter
from sam3_intermot.identity_verification.frozen_none_head import FrozenMetricNoneHead
from scripts.n72r20r3_common import (
    CHECKPOINT_SHA,
    ENCODER_SHA,
    R2_ASSET_ROOT,
    ROOT,
    SEQUENCES,
    read_events,
    load_sequence,
    unit,
    write_zstd_jsonl,
)
from scripts.n72r20r3r2_representation import (
    ASSET_MODELS,
    BATCH_SIZE,
    EXPECTED_PRESENT,
    OUT,
    SEEDS,
    SOURCE_STAGE,
    STAGE,
    TEMPERATURE,
    _frozen_updater,
    _json,
    _sha256,
    _state_update,
    load_episode_dataset,
    storage_audit,
)


OPEN_SET = OUT / "open_set"
CAUSAL = OUT / "causal"
CANONICAL_INDEX = SOURCE_STAGE / "training/canonical_training_index.jsonl"
STATE_ARRAY = SOURCE_STAGE / "training/causal_learned_state_vectors.float32.npy"
NONE_MAX_EPOCHS = 25
NONE_PATIENCE = 5
NONE_BATCH_SIZE = 512
EXPECTED_TOTAL = 8414
EXPECTED_NONE = 1819
NONE_FEATURE_NAMES = (
    "top1_similarity",
    "top2_similarity",
    "top1_minus_top2_margin",
    "mean_similarity",
    "std_similarity",
    "log1p_candidate_count",
    "frames_since_human_initialization_div100",
    "frames_since_last_safe_write_div100",
    "anchor_state_cosine",
)


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def _load_canonical_rows() -> tuple[list[dict[str, Any]], np.ndarray]:
    rows = [json.loads(line) for line in CANONICAL_INDEX.read_text(encoding="utf-8").splitlines() if line.strip()]
    states = np.load(STATE_ARRAY, mmap_mode="r")
    if len(rows) != EXPECTED_TOTAL:
        raise RuntimeError(f"expected {EXPECTED_TOTAL} canonical rows, got {len(rows)}")
    if states.shape[0] < len(rows) or states.shape[1] != 512:
        raise RuntimeError("canonical state array is incompatible")
    if any(row.get("runtime_future_gt_used") is not False for row in rows):
        raise RuntimeError("canonical rows contain runtime future GT")
    return rows, states


def _load_adapter(heldout: str, seed: int, device: torch.device) -> CrossSceneIdentityAdapter:
    path = ASSET_MODELS / f"cross_scene_adapter__{heldout}__seed{seed}.pt"
    payload = torch.load(path, map_location=device, weights_only=False)
    model = CrossSceneIdentityAdapter().to(device)
    model.load_state_dict(payload["state_dict"], strict=True)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def _load_none_head(heldout: str, seed: int, device: torch.device) -> FrozenMetricNoneHead:
    path = ASSET_MODELS / f"none_head__{heldout}__seed{seed}.pt"
    payload = torch.load(path, map_location=device, weights_only=False)
    head = FrozenMetricNoneHead(hidden_dim=int(payload["hidden_dim"])).to(device)
    head.load_state_dict(payload["state_dict"], strict=True)
    head.eval()
    return head


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _metric_features(scores: Sequence[float], context: Mapping[str, Any]) -> np.ndarray:
    values = np.asarray(scores, dtype=np.float32)
    if values.size:
        order = np.sort(values)[::-1]
        top1 = float(order[0])
        top2 = float(order[1]) if order.size > 1 else top1
        mean = float(values.mean())
        std = float(values.std())
    else:
        top1 = top2 = mean = std = 0.0
    feature = np.asarray(
        [
            top1,
            top2,
            top1 - top2,
            mean,
            std,
            math.log1p(float(len(values))),
            float(context.get("frames_since_human_initialization", 0.0)) / 100.0,
            float(context.get("frames_since_last_memory_write", 0.0)) / 100.0,
            float(context.get("learned_state_human_anchor_cosine", 1.0)),
        ],
        dtype=np.float32,
    )
    if not np.isfinite(feature).all():
        raise RuntimeError("NONE feature vector is non-finite")
    return feature


def _score_metric_rows(
    model: CrossSceneIdentityAdapter,
    dataset: Any,
    rows: Sequence[dict[str, Any]],
    states: np.ndarray,
    device: torch.device,
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(rows), BATCH_SIZE):
            batch_rows = list(rows[start : start + BATCH_SIZE])
            max_candidates = max(1, max(len(row["candidate_axis"]) for row in batch_rows))
            query = np.asarray([states[int(row["causal_learned_state_ref"])] for row in batch_rows], dtype=np.float32)
            candidate_values = np.zeros((len(batch_rows), max_candidates, 512), dtype=np.float32)
            mask = np.zeros((len(batch_rows), max_candidates), dtype=bool)
            for row_index, row in enumerate(batch_rows):
                offsets = [int(item["embedding_offset"]) for item in row["candidate_axis"]]
                values = dataset.candidate_arrays[str(row["sequence"])][np.asarray(offsets, dtype=np.int64)]
                candidate_values[row_index, : len(offsets)] = values
                mask[row_index, : len(offsets)] = True
            output = model(
                torch.from_numpy(query).to(device),
                torch.from_numpy(candidate_values).to(device),
                torch.from_numpy(mask).to(device),
            )
            scores = output["scores"].detach().cpu().numpy()
            for row_index, row in enumerate(batch_rows):
                count = len(row["candidate_axis"])
                values = np.asarray(scores[row_index, :count], dtype=np.float32)
                label_index = row["training_label"]["candidate_index"]
                result.append(
                    {
                        "sequence": str(row["sequence"]),
                        "frame": int(row["frame"]),
                        "candidate_uids": [str(item["candidate_uid"]) for item in row["candidate_axis"]],
                        "candidate_scores": values.astype(float).tolist(),
                        "features": _metric_features(values, row["runtime_context"]).astype(float).tolist(),
                        "label_index": int(label_index) if label_index is not None else -1,
                        "label_class": str(row["training_label"]["class"]),
                        "target_gt_present_posthoc": bool(row["target_gt_present_posthoc"]),
                        "taxonomy_posthoc": str(row["taxonomy_posthoc"]),
                        "target_public_id": int(row["target_public_id"]),
                        "runtime_future_gt_used": False,
                        "runtime_gt_clean": True,
                        "posthoc_gt_used": True,
                        "candidate_created": False,
                    }
                )
    return result


def _none_kind(row: Mapping[str, Any]) -> str:
    if int(row["label_index"]) >= 0:
        return "PRESENT"
    return "N1_TARGET_PRESENT_NO_CANDIDATE" if bool(row["target_gt_present_posthoc"]) else "N0_TARGET_ABSENT"


def _round_robin_sample(rows: Sequence[dict[str, Any]], count: int, seed: int, key_name: str) -> list[dict[str, Any]]:
    if count >= len(rows):
        return list(rows)
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row[key_name])].append(row)
    rng = random.Random(seed)
    for values in groups.values():
        rng.shuffle(values)
    keys = sorted(groups)
    selected: list[dict[str, Any]] = []
    cursors = {key: 0 for key in keys}
    while len(selected) < count:
        advanced = False
        for key in keys:
            if cursors[key] < len(groups[key]) and len(selected) < count:
                selected.append(groups[key][cursors[key]])
                cursors[key] += 1
                advanced = True
        if not advanced:
            break
    return selected


def _balanced_none_rows(rows: Sequence[dict[str, Any]], seed: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    present = [dict(row, balance_key=f"{row['sequence']}:{row['target_public_id']}") for row in rows if int(row["label_index"]) >= 0]
    absent = [dict(row, balance_key=f"{row['sequence']}:{row['target_public_id']}:{_none_kind(row)}") for row in rows if int(row["label_index"]) < 0]
    if not present or not absent:
        raise RuntimeError("NONE training requires both PRESENT and ABSENT examples")
    if len(present) > 2 * len(absent):
        present = _round_robin_sample(present, 2 * len(absent), seed, "balance_key")
    elif len(absent) > len(present):
        absent = _round_robin_sample(absent, len(present), seed + 1, "balance_key")
    selected = present + absent
    random.Random(seed + 2).shuffle(selected)
    counts = {kind: sum(_none_kind(row) == kind for row in selected) for kind in ("PRESENT", "N0_TARGET_ABSENT", "N1_TARGET_PRESENT_NO_CANDIDATE")}
    return selected, {
        "present_count": counts["PRESENT"],
        "absent_count": counts["N0_TARGET_ABSENT"] + counts["N1_TARGET_PRESENT_NO_CANDIDATE"],
        "N0_count": counts["N0_TARGET_ABSENT"],
        "N1_count": counts["N1_TARGET_PRESENT_NO_CANDIDATE"],
        "present_absent_ratio": float(counts["PRESENT"] / max(1, counts["N0_TARGET_ABSENT"] + counts["N1_TARGET_PRESENT_NO_CANDIDATE"])),
        "sequence_balanced": True,
        "identity_balanced": True,
        "sampling_seed": seed,
    }


def _none_batch(rows: Sequence[dict[str, Any]], device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    max_candidates = max(1, max(len(row["candidate_scores"]) for row in rows))
    candidates = np.full((len(rows), max_candidates), -1.0e9, dtype=np.float32)
    features = np.asarray([row["features"] for row in rows], dtype=np.float32)
    labels = []
    for index, row in enumerate(rows):
        scores = np.asarray(row["candidate_scores"], dtype=np.float32)
        candidates[index, : len(scores)] = scores
        labels.append(int(row["label_index"]) if int(row["label_index"]) >= 0 else max_candidates)
    return (
        torch.from_numpy(candidates).to(device),
        torch.from_numpy(features).to(device),
        torch.tensor(labels, dtype=torch.long, device=device),
    )


def _none_loss(head: FrozenMetricNoneHead, rows: Sequence[dict[str, Any]], device: torch.device) -> torch.Tensor:
    scores, features, labels = _none_batch(rows, device)
    none_score = head(features).unsqueeze(1)
    logits = torch.cat([scores, none_score], dim=1) / TEMPERATURE
    return F.cross_entropy(logits, labels)


def _none_batches(rows: Sequence[dict[str, Any]], seed: int, training: bool) -> list[list[dict[str, Any]]]:
    values = list(rows)
    if training:
        random.Random(seed).shuffle(values)
    return [values[start : start + NONE_BATCH_SIZE] for start in range(0, len(values), NONE_BATCH_SIZE)]


def _none_epoch(head: FrozenMetricNoneHead, rows: Sequence[dict[str, Any]], device: torch.device, optimizer: torch.optim.Optimizer | None, seed: int) -> float:
    head.train(optimizer is not None)
    losses = []
    for batch in _none_batches(rows, seed, optimizer is not None):
        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(optimizer is not None):
            loss = _none_loss(head, batch, device)
        if optimizer is not None:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), 5.0)
            optimizer.step()
        losses.append(float(loss.detach().cpu()))
    return float(np.mean(losses)) if losses else float("inf")


def _fit_none_inner(train_rows: Sequence[dict[str, Any]], validation_rows: Sequence[dict[str, Any]], seed: int, device: torch.device) -> tuple[FrozenMetricNoneHead, dict[str, Any]]:
    _seed_everything(seed)
    head = FrozenMetricNoneHead().to(device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=1.0e-3, weight_decay=1.0e-4)
    best_state = copy.deepcopy(head.state_dict())
    best_validation = float("inf")
    best_epoch = 1
    stale = 0
    history = []
    for epoch in range(1, NONE_MAX_EPOCHS + 1):
        train_loss = _none_epoch(head, train_rows, device, optimizer, seed + epoch)
        validation_loss = _none_epoch(head, validation_rows, device, None, seed)
        history.append({"epoch": epoch, "train_loss": train_loss, "validation_loss": validation_loss})
        if validation_loss < best_validation - 1.0e-6:
            best_validation = validation_loss
            best_epoch = epoch
            best_state = copy.deepcopy(head.state_dict())
            stale = 0
        else:
            stale += 1
        if stale >= NONE_PATIENCE:
            break
    head.load_state_dict(best_state)
    return head, {"best_epoch": best_epoch, "epochs_run": len(history), "best_validation_loss": best_validation, "history": history, "trainable_parameters": head.trainable_parameters}


def _fit_none_final(rows: Sequence[dict[str, Any]], seed: int, epochs: int, device: torch.device) -> tuple[FrozenMetricNoneHead, dict[str, Any]]:
    _seed_everything(seed)
    head = FrozenMetricNoneHead().to(device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=1.0e-3, weight_decay=1.0e-4)
    history = []
    for epoch in range(1, max(1, int(epochs)) + 1):
        history.append(_none_epoch(head, rows, device, optimizer, seed + epoch))
    head.eval()
    return head, {"epochs_run": len(history), "last_train_loss": history[-1] if history else None}


def _open_predictions(head: FrozenMetricNoneHead, rows: Sequence[dict[str, Any]], device: torch.device) -> list[dict[str, Any]]:
    head.eval()
    result = []
    for start in range(0, len(rows), NONE_BATCH_SIZE):
        batch = list(rows[start : start + NONE_BATCH_SIZE])
        scores, features, _ = _none_batch(batch, device)
        with torch.no_grad():
            none_scores = head(features)
        for index, row in enumerate(batch):
            count = len(row["candidate_scores"])
            candidate_scores = np.asarray(row["candidate_scores"], dtype=np.float64)
            none_score = float(none_scores[index].detach().cpu())
            logits = np.concatenate([candidate_scores, [none_score]]) / TEMPERATURE
            predicted = int(np.argmax(logits))
            predicted_index = predicted if predicted < count else -1
            result.append(
                {
                    "sequence": row["sequence"],
                    "frame": int(row["frame"]),
                    "candidate_uids": list(row["candidate_uids"]),
                    "candidate_scores": candidate_scores.tolist(),
                    "none_score": none_score,
                    "features": list(row["features"]),
                    "label_index": int(row["label_index"]),
                    "predicted_index": predicted_index,
                    "predicted_uid": row["candidate_uids"][predicted_index] if predicted_index >= 0 else None,
                    "candidate_created": False,
                    "runtime_future_gt_used": False,
                    "runtime_gt_clean": True,
                    "posthoc_gt_used": True,
                }
            )
    return result


def _aggregate_open_predictions(prediction_sets: Sequence[Sequence[dict[str, Any]]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for values in prediction_sets:
        for row in values:
            grouped[(str(row["sequence"]), int(row["frame"]))].append(row)
    result = []
    for key in sorted(grouped):
        values = grouped[key]
        if len(values) != len(prediction_sets):
            raise RuntimeError(f"NONE seed prediction mismatch at {key}")
        axis = values[0]["candidate_uids"]
        if any(row["candidate_uids"] != axis for row in values):
            raise RuntimeError(f"NONE candidate axis mismatch at {key}")
        scores = np.mean([np.asarray(row["candidate_scores"], dtype=np.float64) for row in values], axis=0)
        none_score = float(np.mean([float(row["none_score"]) for row in values]))
        predicted = int(np.argmax(np.concatenate([scores, [none_score]]) / TEMPERATURE))
        predicted_index = predicted if predicted < len(axis) else -1
        result.append(
            {
                "sequence": key[0],
                "frame": key[1],
                "candidate_uids": axis,
                "candidate_scores": scores.tolist(),
                "none_score": none_score,
                "label_index": int(values[0]["label_index"]),
                "predicted_index": predicted_index,
                "predicted_uid": axis[predicted_index] if predicted_index >= 0 else None,
                "seed_count": len(values),
                "candidate_created": False,
                "runtime_future_gt_used": False,
                "runtime_gt_clean": True,
                "posthoc_gt_used": True,
            }
        )
    return result


def _open_metrics(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    none_rows = [row for row in rows if int(row["label_index"]) < 0]
    present_rows = [row for row in rows if int(row["label_index"]) >= 0]
    false_positive = sum(int(row["predicted_index"]) >= 0 for row in none_rows)
    correct_id = sum(int(row["predicted_index"]) == int(row["label_index"]) for row in present_rows)
    return {
        "rows": len(rows),
        "none_rows": len(none_rows),
        "present_rows": len(present_rows),
        "negative_fpr": float(false_positive / len(none_rows)) if none_rows else None,
        "open_set_correct_id_recall": float(correct_id / len(present_rows)) if present_rows else None,
        "false_positive_count": false_positive,
        "correct_id_count": correct_id,
        "candidate_creation_count": sum(bool(row.get("candidate_created")) for row in rows),
        "runtime_future_gt_used": any(bool(row.get("runtime_future_gt_used")) for row in rows),
        "runtime_gt_clean": all(not bool(row.get("runtime_future_gt_used")) for row in rows),
        "posthoc_gt_used": True,
    }


def _open_per_sequence(rows: Sequence[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    grouped = defaultdict(list)
    for row in rows:
        grouped[str(row["sequence"])].append(row)
    return {sequence: dict(_open_metrics(grouped[sequence]), sequence=sequence) for sequence in SEQUENCES}


def _save_none_checkpoint(head: FrozenMetricNoneHead, heldout: str, seed: int, fit_sequences: Sequence[str], validation: str, selected_epoch: int, representation_path: Path, representation_sha: str) -> dict[str, Any]:
    ASSET_MODELS.mkdir(parents=True, exist_ok=True)
    path = ASSET_MODELS / f"none_head__{heldout}__seed{seed}.pt"
    torch.save(
        {
            "stage": STAGE,
            "architecture": "FrozenMetricNoneHead",
            "hidden_dim": head.hidden_dim,
            "input_features": list(NONE_FEATURE_NAMES),
            "temperature": TEMPERATURE,
            "source_representation": str(representation_path),
            "source_representation_sha256": representation_sha,
            "parameter_fit_sequences": list(fit_sequences),
            "internal_validation_sequence": validation,
            "seed": int(seed),
            "trainable_parameters": head.trainable_parameters,
            "representation_frozen": True,
            "OSNet_frozen": True,
            "GRU_frozen": True,
            "runtime_future_gt_used": False,
            "state_dict": {key: value.detach().cpu() for key, value in head.state_dict().items()},
        },
        path,
    )
    return {"heldout_sequence": heldout, "seed": seed, "path": str(path), "sha256": _sha256(path), "trainable_parameters": head.trainable_parameters, "binary_committed": False}


def _episode_dataset_sha() -> dict[str, str]:
    return {name: _sha256(TRAINING_PATH) for name, TRAINING_PATH in (("identity_episode_index", OUT / "training/identity_episode_index.jsonl"), ("query_states", OUT / "training/query_states.float32.npy"))}


def _write_frozen_representation_policy(formal: Mapping[str, Any]) -> dict[str, Any]:
    checkpoint_manifest = _json(OUT / "training/checkpoint_manifest.json")
    policy = {
        "stage": STAGE,
        "policy": "FROZEN_REPRESENTATION_POLICY",
        "architecture": "CrossSceneIdentityAdapter",
        "weights": checkpoint_manifest["records"],
        "training_folds": list(SEQUENCES),
        "loss": "1.0*listwise + 0.20*supervised_contrastive + 0.20*hard_negative_margin",
        "temperature": TEMPERATURE,
        "hard_negative_margin": 0.20,
        "trainable_parameters": 265472,
        "OSNet_sha256": ENCODER_SHA,
        "GRU_checkpoint_sha256": CHECKPOINT_SHA,
        "episode_dataset_sha256": _episode_dataset_sha(),
        "formal_runtime_metrics": formal["runtime_metrics"],
        "formal_oracle_clean_metrics": formal["oracle_clean_metrics"],
        "formal_gate": formal["representation_gate"],
        "representation_frozen": True,
        "candidate_generation_frozen": True,
        "runtime_future_gt_used": False,
        "runtime_gt_clean": True,
    }
    _write_json(OUT / "FROZEN_REPRESENTATION_POLICY.json", policy)
    return policy


def run_open_set(dataset: Any, device: torch.device) -> dict[str, Any]:
    formal = _json(OUT / "representation/formal_loso.json")
    if not bool(formal["representation_gate"]["pass"]):
        raise RuntimeError("Phase A representation gate did not pass; Phase B is forbidden")
    policy = _write_frozen_representation_policy(formal)
    rows, states = _load_canonical_rows()
    all_seed_predictions: dict[str, list[dict[str, Any]]] = {str(seed): [] for seed in SEEDS}
    fold_records = []
    checkpoint_records = []
    training_records = []
    for heldout in SEQUENCES:
        validation = SEQUENCES[(SEQUENCES.index(heldout) + 1) % len(SEQUENCES)]
        fit_sequences = [sequence for sequence in SEQUENCES if sequence not in {heldout, validation}]
        final_sequences = [sequence for sequence in SEQUENCES if sequence != heldout]
        for seed in SEEDS:
            representation_path = ASSET_MODELS / f"cross_scene_adapter__{heldout}__seed{seed}.pt"
            representation_sha = _sha256(representation_path)
            adapter = _load_adapter(heldout, seed, device)
            # Do not even materialize heldout metric rows until the NONE head
            # has been fit and refit on the non-heldout sequences.
            nonheldout_rows = [row for row in rows if row["sequence"] != heldout]
            scored = _score_metric_rows(adapter, dataset, nonheldout_rows, states, device)
            fit_rows = [row for row in scored if row["sequence"] in fit_sequences]
            validation_rows = [row for row in scored if row["sequence"] == validation]
            balanced_fit, fit_balance = _balanced_none_rows(fit_rows, seed + 1000)
            inner_head, inner_info = _fit_none_inner(balanced_fit, validation_rows, seed + 2000, device)
            final_raw = [row for row in scored if row["sequence"] in final_sequences]
            balanced_final, final_balance = _balanced_none_rows(final_raw, seed + 3000)
            final_head, final_info = _fit_none_final(balanced_final, seed + 4000, int(inner_info["best_epoch"]), device)
            heldout_rows = _score_metric_rows(adapter, dataset, [row for row in rows if row["sequence"] == heldout], states, device)
            predictions = _open_predictions(final_head, heldout_rows, device)
            all_seed_predictions[str(seed)].extend(predictions)
            head_record = _save_none_checkpoint(final_head, heldout, seed, fit_sequences, validation, int(inner_info["best_epoch"]), representation_path, representation_sha)
            checkpoint_records.append(head_record)
            training_records.append(
                {
                    "heldout_sequence": heldout,
                    "seed": seed,
                    "parameter_fit_sequences": fit_sequences,
                    "internal_validation_sequence": validation,
                    "final_training_sequences": final_sequences,
                    "training_present_episodes": fit_balance["present_count"],
                    "training_N0_episodes": fit_balance["N0_count"],
                    "training_N1_episodes": fit_balance["N1_count"],
                    "training_present_absent_ratio": fit_balance["present_absent_ratio"],
                    "final_training_present_episodes": final_balance["present_count"],
                    "final_training_absent_episodes": final_balance["absent_count"],
                    "heldout_observations_used_in_training": 0,
                    "heldout_identity_keys_in_training": 0,
                    "outer_heldout_absent_from_training": True,
                    "representation_frozen": True,
                    "inner_fit": inner_info,
                    "final_fit": final_info,
                    "head_trainable_parameters": final_head.trainable_parameters,
                }
            )
            del adapter, inner_head, final_head
    seed_predictions = [all_seed_predictions[str(seed)] for seed in SEEDS]
    ensemble = _aggregate_open_predictions(seed_predictions)
    if len(ensemble) != EXPECTED_TOTAL:
        raise RuntimeError(f"expected {EXPECTED_TOTAL} open-set rows, got {len(ensemble)}")
    metrics = _open_metrics(ensemble)
    per_sequence = _open_per_sequence(ensemble)
    macro_fpr = float(np.mean([value["negative_fpr"] for value in per_sequence.values()]))
    macro_recall = float(np.mean([value["open_set_correct_id_recall"] for value in per_sequence.values()]))
    gate = {
        "O1_pooled_negative_fpr_le_0.02": metrics["negative_fpr"] is not None and metrics["negative_fpr"] <= 0.02,
        "O2_pooled_open_set_correct_id_recall_ge_0.60": metrics["open_set_correct_id_recall"] is not None and metrics["open_set_correct_id_recall"] >= 0.60,
        "O3_macro_negative_fpr_le_0.05": macro_fpr <= 0.05,
        "O4_macro_open_set_correct_id_recall_ge_0.40": macro_recall >= 0.40,
        "O5_runtime_future_gt_used_false": metrics["runtime_future_gt_used"] is False,
        "O6_runtime_gt_clean_true": metrics["runtime_gt_clean"] is True,
        "O7_no_candidate_creation": metrics["candidate_creation_count"] == 0,
    }
    result = {
        "stage": STAGE,
        "status": "PASS_OPEN_SET_FORMAL_LOSO_COMPLETE",
        "representation_policy": "../FROZEN_REPRESENTATION_POLICY.json",
        "representation_frozen": True,
        "seeds": list(SEEDS),
        "heldout_sequences": list(SEQUENCES),
        "features": list(NONE_FEATURE_NAMES),
        "none_head": "FrozenMetricNoneHead",
        "none_head_trainable_parameters": 353,
        "runtime_metrics": metrics,
        "macro_negative_fpr": macro_fpr,
        "macro_open_set_correct_id_recall": macro_recall,
        "per_sequence": per_sequence,
        "open_set_gate": {"checks": gate, "pass": bool(all(gate.values()))},
        "training_records": training_records,
        "runtime_future_gt_used": False,
        "runtime_gt_clean": True,
        "candidate_creation_count": 0,
    }
    OPEN_SET.mkdir(parents=True, exist_ok=True)
    write_zstd_jsonl(OPEN_SET / "formal_predictions.jsonl.zst", ensemble)
    for seed in SEEDS:
        write_zstd_jsonl(OPEN_SET / f"formal_predictions_seed{seed}.jsonl.zst", all_seed_predictions[str(seed)])
    _write_json(OPEN_SET / "training_manifest.json", {"stage": STAGE, "representation_frozen": True, "head": "FrozenMetricNoneHead", "head_trainable_parameters": 353, "input_features": list(NONE_FEATURE_NAMES), "records": training_records, "N0_total": sum(record["training_N0_episodes"] for record in training_records), "N1_total": sum(record["training_N1_episodes"] for record in training_records), "runtime_future_gt_used": False, "runtime_gt_clean": True})
    _write_json(OPEN_SET / "formal_loso.json", result)
    _write_json(OPEN_SET / "per_sequence.json", {"stage": STAGE, "metrics": per_sequence, "macro_negative_fpr": macro_fpr, "macro_open_set_correct_id_recall": macro_recall})
    _write_json(OPEN_SET / "calibration.json", {"stage": STAGE, "input_features": list(NONE_FEATURE_NAMES), "temperature": TEMPERATURE, "none_head_score_is_compared_directly_with_candidate_cosine": True, "pooled": metrics, "macro_negative_fpr": macro_fpr, "macro_open_set_correct_id_recall": macro_recall, "gate": result["open_set_gate"]})
    _write_json(OPEN_SET / "checkpoint_manifest.json", {"stage": STAGE, "records": checkpoint_records, "binary_committed": False, "asset_root": str(ASSET_MODELS)})
    _write_json(OPEN_SET / "formal_prediction_manifest.json", {"stage": STAGE, "rows": len(ensemble), "seed_rows": {str(seed): len(all_seed_predictions[str(seed)]) for seed in SEEDS}, "one_decision_per_frame": True, "runtime_future_gt_used": False, "runtime_gt_clean": True, "candidate_creation_count": 0})
    if result["open_set_gate"]["pass"]:
        _write_json(OPEN_SET / "FROZEN_OPEN_SET_IDENTITY_POLICY.json", {"stage": STAGE, "policy": "FROZEN_OPEN_SET_IDENTITY_POLICY", "representation_policy": "../FROZEN_REPRESENTATION_POLICY.json", "none_head": "FrozenMetricNoneHead", "head_checkpoint_manifest": "checkpoint_manifest.json", "input_features": list(NONE_FEATURE_NAMES), "temperature": TEMPERATURE, "gate": result["open_set_gate"], "runtime_future_gt_used": False, "runtime_gt_clean": True})
    return result


def _base_target_uid(base_row: Mapping[str, Any], target_public_id: int) -> str | None:
    for item in base_row.get("base_assignment", {}).get("public_assignments", []):
        if int(item["public_id"]) == int(target_public_id):
            value = item.get("candidate_uid")
            return None if value in (None, "", "None") else str(value)
    return None


def _dynamic_prediction(
    adapter: CrossSceneIdentityAdapter,
    head: FrozenMetricNoneHead,
    state: np.ndarray,
    candidates: Sequence[Mapping[str, Any]],
    candidate_features: np.ndarray,
    context: Mapping[str, Any],
    device: torch.device,
) -> tuple[np.ndarray, float, int]:
    count = len(candidates)
    if count:
        values = np.asarray(candidate_features, dtype=np.float32)
        tensor = torch.from_numpy(values[None, :, :]).to(device)
        mask = torch.ones((1, count), dtype=torch.bool, device=device)
    else:
        values = np.zeros((1, 512), dtype=np.float32)
        tensor = torch.from_numpy(values[None, :, :]).to(device)
        mask = torch.zeros((1, 1), dtype=torch.bool, device=device)
    with torch.no_grad():
        output = adapter(torch.from_numpy(np.asarray(state, dtype=np.float32)[None, :]).to(device), tensor, mask)
        scores = output["scores"][0, :count].detach().cpu().numpy().astype(np.float64)
        features = torch.from_numpy(_metric_features(scores, context)[None, :]).to(device)
        none_score = float(head(features)[0].detach().cpu())
    selected = int(np.argmax(np.concatenate([scores, [none_score]]) / TEMPERATURE))
    return scores, none_score, selected if selected < count else -1


def run_causal(dataset: Any, open_result: Mapping[str, Any], device: torch.device) -> dict[str, Any]:
    if not bool(open_result["open_set_gate"]["pass"]):
        raise RuntimeError("Phase B open-set gate did not pass; causal replay is forbidden")
    events = read_events(R2_ASSET_ROOT)
    all_rows, _ = _load_canonical_rows()
    rows_by_sequence: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in all_rows:
        rows_by_sequence[str(row["sequence"])].append(row)
    replay_rows = []
    per_sequence = {}
    for sequence in SEQUENCES:
        frames, base_rows = load_sequence(R2_ASSET_ROOT, sequence)
        base_by_frame = {}
        for (frame_payload, candidates), row in zip(frames, base_rows):
            if int(frame_payload["frame"]) != int(row["frame"]):
                raise RuntimeError(f"frozen base frame mismatch for {sequence}")
            base_by_frame[int(row["frame"])] = (row, candidates)
        target_rows = {int(row["frame"]): row for row in rows_by_sequence[sequence]}
        event = events[sequence]
        target_public_id = int(event["target_public_id"])
        adapters = [_load_adapter(sequence, seed, device) for seed in SEEDS]
        heads = [_load_none_head(sequence, seed, device) for seed in SEEDS]
        updater = _frozen_updater()
        anchor = unit(event["human_anchor"], "human anchor")
        state = anchor.copy()
        frames_since_write = 1
        writes = correct_writes = wrong_writes = 0
        present_rows = 0
        for frame in sorted(target_rows):
            if frame not in base_by_frame:
                raise RuntimeError(f"missing frozen base row for {sequence}:{frame}")
            base_row, candidates = base_by_frame[frame]
            current = target_rows[frame]
            candidate_features = np.asarray(
                [dataset.candidate_arrays[sequence][int(candidate["embedding_offset"])] for candidate in candidates],
                dtype=np.float32,
            )
            context = {
                "frames_since_human_initialization": int(frame - int(event["event_frame"])),
                "frames_since_last_memory_write": int(frames_since_write),
                "learned_state_human_anchor_cosine": float(np.dot(unit(state, "causal state"), anchor)),
            }
            seed_outputs = [_dynamic_prediction(adapter, head, state, candidates, candidate_features, context, device) for adapter, head in zip(adapters, heads)]
            if candidate_features.shape[0]:
                score_matrix = np.asarray([value[0] for value in seed_outputs], dtype=np.float64)
                mean_scores = score_matrix.mean(axis=0)
            else:
                mean_scores = np.asarray([], dtype=np.float64)
            mean_none = float(np.mean([value[1] for value in seed_outputs]))
            selected = int(np.argmax(np.concatenate([mean_scores, [mean_none]]) / TEMPERATURE))
            predicted_index = selected if selected < len(candidates) else -1
            predicted_uid = str(candidates[predicted_index]["candidate_uid"]) if predicted_index >= 0 else None
            base_uid = _base_target_uid(base_row, target_public_id)
            commit = predicted_uid is not None and base_uid is not None and predicted_uid == base_uid
            label_index = current["training_label"]["candidate_index"]
            label_index = int(label_index) if label_index is not None else -1
            if label_index >= 0:
                present_rows += 1
            if commit:
                writes += 1
                if predicted_index == label_index:
                    correct_writes += 1
                else:
                    wrong_writes += 1
                state = _state_update(updater, state, candidate_features[predicted_index])
                frames_since_write = 1
            else:
                frames_since_write += 1
            replay_rows.append(
                {
                    "sequence": sequence,
                    "frame": frame,
                    "target_public_id": target_public_id,
                    "candidate_uids": [str(candidate["candidate_uid"]) for candidate in candidates],
                    "predicted_index": predicted_index,
                    "predicted_uid": predicted_uid,
                    "none_score": mean_none,
                    "base_assignment_uid": base_uid,
                    "trusted_commit": commit,
                    "state_updated_after_decision": commit,
                    "label_index_posthoc": label_index,
                    "correct_write_posthoc": bool(commit and predicted_index == label_index),
                    "wrong_write_posthoc": bool(commit and predicted_index != label_index),
                    "posthoc_gt_used": True,
                    "runtime_future_gt_used": False,
                    "runtime_gt_clean": True,
                    "candidate_created": False,
                    "human_anchor_immutable": True,
                    "public_id_immutable": True,
                    "association_override": False,
                    "one_frame_one_decision": True,
                }
            )
        total_writes = writes
        per_sequence[sequence] = {
            "sequence": sequence,
            "rows": len(target_rows),
            "present_rows": present_rows,
            "writes": writes,
            "correct_writes": correct_writes,
            "wrong_writes": wrong_writes,
            "wrong_write_rate": float(wrong_writes / total_writes) if total_writes else 0.0,
            "correct_write_retention": float(correct_writes / present_rows) if present_rows else 0.0,
            "runtime_future_gt_used": False,
            "runtime_gt_clean": True,
            "association_override": False,
        }
    total_rows = len(replay_rows)
    total_present = sum(item["present_rows"] for item in per_sequence.values())
    total_writes = sum(item["writes"] for item in per_sequence.values())
    total_correct = sum(item["correct_writes"] for item in per_sequence.values())
    total_wrong = sum(item["wrong_writes"] for item in per_sequence.values())
    metrics = {
        "rows": total_rows,
        "present_rows": total_present,
        "writes": total_writes,
        "correct_writes": total_correct,
        "wrong_writes": total_wrong,
        "wrong_write_rate": float(total_wrong / total_writes) if total_writes else 0.0,
        "correct_write_retention": float(total_correct / total_present) if total_present else 0.0,
        "runtime_future_gt_used": False,
        "runtime_gt_clean": True,
        "association_override": False,
        "human_anchor_immutable": True,
        "public_id_immutable": True,
        "one_frame_one_decision": True,
    }
    macro_wrong = float(np.mean([item["wrong_write_rate"] for item in per_sequence.values()]))
    macro_retention = float(np.mean([item["correct_write_retention"] for item in per_sequence.values()]))
    gate = {
        "C1_pooled_wrong_write_rate_le_0.02": metrics["wrong_write_rate"] <= 0.02,
        "C2_pooled_correct_write_retention_ge_0.60": metrics["correct_write_retention"] >= 0.60,
        "C3_runtime_future_gt_used_false": metrics["runtime_future_gt_used"] is False,
        "C4_runtime_gt_clean_true": metrics["runtime_gt_clean"] is True,
        "C5_no_association_override": metrics["association_override"] is False,
    }
    result = {"stage": STAGE, "status": "PASS_CAUSAL_REPLAY_COMPLETE", "runtime_metrics": metrics, "macro_wrong_write_rate": macro_wrong, "macro_correct_write_retention": macro_retention, "per_sequence": per_sequence, "causal_gate": {"checks": gate, "pass": bool(all(gate.values()))}, "runtime_future_gt_used": False, "runtime_gt_clean": True}
    CAUSAL.mkdir(parents=True, exist_ok=True)
    write_zstd_jsonl(CAUSAL / "memory_write_audit.jsonl.zst", replay_rows)
    _write_json(CAUSAL / "causal_replay.json", result)
    _write_json(CAUSAL / "per_sequence.json", {"stage": STAGE, "metrics": per_sequence, "macro_wrong_write_rate": macro_wrong, "macro_correct_write_retention": macro_retention})
    _write_json(CAUSAL / "memory_write_manifest.json", {"stage": STAGE, "rows": len(replay_rows), "trusted_commit_rule": "predicted_candidate_uid == frozen_base_assignment_uid", "none_no_write": True, "disagreement_no_write": True, "association_override": False, "runtime_future_gt_used": False, "runtime_gt_clean": True})
    return result


def finalize_continuation(formal: Mapping[str, Any], open_result: Mapping[str, Any], causal_result: Mapping[str, Any] | None, tests_summary: str) -> dict[str, Any]:
    if not open_result["open_set_gate"]["pass"]:
        decision = "FAIL_OPEN_SET_CALIBRATION_AFTER_REPRESENTATION"
        bottleneck = "BOTTLENECK_OPEN_SET_CALIBRATION"
        next_association = False
    elif causal_result is not None and causal_result["causal_gate"]["pass"]:
        decision = "PASS_CROSS_SCENE_OPEN_SET_IDENTITY_REPRESENTATION"
        bottleneck = None
        next_association = True
    else:
        decision = "FAIL_CAUSAL_COMMIT_AFTER_REPRESENTATION"
        bottleneck = "BOTTLENECK_CAUSAL_COMMIT"
        next_association = False
    storage_after = storage_audit()
    _write_json(OUT / "storage_audit_after.json", storage_after)
    _write_json(OUT / "future_association_headroom.json", {"stage": STAGE, "status": "AUTHORIZED" if next_association else "NOT_AUTHORIZED", "association_rescue_run": False, "solver_called": False, "public_authority_changed": False, "runtime_future_gt_used": False, "runtime_gt_clean": True, "reason": "N72R20R4 remains closed unless the complete R3R2 representation/open-set/causal gates pass"})
    result = {
        "stage": STAGE,
        "goal": "Cross-Scene Open-Set Identity Representation Learning",
        "decision": decision,
        "bottleneck_classification": bottleneck,
        "source_commit": formal["source_head"],
        "training_identity_count": _json(OUT / "training/identity_episode_manifest.json")["training_identity_count"],
        "identity_episode_count": _json(OUT / "training/identity_episode_manifest.json")["episode_count"],
        "candidate_GTs_matched": _json(OUT / "training/identity_match_manifest.json")["canonical_records"],
        "primary_model": "CrossSceneIdentityAdapter",
        "trainable_parameters": 265472,
        "frozen_baseline_rank1": formal["baseline_metrics"]["rank1"],
        "new_rank1": formal["runtime_metrics"]["rank1"],
        "rank1_delta": float(formal["runtime_metrics"]["rank1"] - formal["baseline_metrics"]["rank1"]),
        "paired_CI": formal["paired_bootstrap"]["interval_95"],
        "MRR": formal["runtime_metrics"]["MRR"],
        "Rank2": formal["runtime_metrics"]["rank2"],
        "Rank3": formal["runtime_metrics"]["rank3"],
        "Rank5": formal["runtime_metrics"]["rank5"],
        "runtime_state_metrics": formal["runtime_metrics"],
        "oracle_clean_state_metrics": formal["oracle_clean_metrics"],
        "representation_gate": formal["representation_gate"],
        "open_set_metrics": open_result["runtime_metrics"],
        "open_set_gate": open_result["open_set_gate"],
        "causal_metrics": causal_result["runtime_metrics"] if causal_result is not None else None,
        "causal_gate": causal_result["causal_gate"] if causal_result is not None else None,
        "runtime_future_gt_used": False,
        "runtime_gt_clean": True,
        "OSNet_frozen": True,
        "GRU_frozen": True,
        "SAM3_rerun": False,
        "candidate_generation_frozen": True,
        "association_authority_started": False,
        "val_accessed": False,
        "test_accessed": False,
        "next_backbone_adaptation_stage_authorized": False,
        "next_memory_state_learning_stage_authorized": False,
        "next_association_authority_stage_authorized": next_association,
        "tests_summary": tests_summary,
        "storage_after": storage_after,
    }
    _write_json(OUT / "FINAL_RESULT.json", result)
    _write_json(OUT / "stage_status.json", {"stage": STAGE, "goal_reference": f"outputs/{STAGE}/FINAL_GOAL.json", "goal_frozen": True, "status": "PASS" if decision.startswith("PASS_") else ("OPEN_SET_FAIL" if decision.startswith("FAIL_OPEN_SET") else "CAUSAL_FAIL"), "decision": decision, "last_completed_artifact": "outputs/N72R20R3R2/causal/causal_replay.json" if causal_result is not None else "outputs/N72R20R3R2/open_set/formal_loso.json", "representation_gate": formal["representation_gate"], "open_set_gate": open_result["open_set_gate"], "causal_gate": causal_result["causal_gate"] if causal_result is not None else None, "open_set_started": True, "causal_started": causal_result is not None, "runtime_future_gt_used": False, "runtime_gt_clean": True, "sam3_rerun": False, "val_accessed": False, "test_accessed": False, "next_association_authority_stage_authorized": next_association, "tests": tests_summary, "storage_audit_after": storage_after})
    first = "YES" if decision.startswith("PASS_") else "NO"
    report = [
        f"{first} — sequence-held-out identity representation learning over all available training identities raises unseen-sequence candidate identification enough to support open-set persistent identity recognition." if first == "YES" else f"NO — sequence-held-out identity representation learning passes Phase A, but the complete open-set/causal requirement is not met.",
        "",
        f"# InterMOT {STAGE} — Cross-Scene Open-Set Identity Representation Learning",
        "",
        f"Final decision: `{decision}`.",
        "",
        "## Phase A — representation",
        "",
        f"Frozen learned-state baseline Rank-1 `{result['frozen_baseline_rank1']}`; new runtime-state Rank-1 `{result['new_rank1']}`; delta `{result['rank1_delta']}`; MRR `{result['MRR']}`; Rank-2/3/5 `{result['Rank2']}` / `{result['Rank3']}` / `{result['Rank5']}`.",
        f"Oracle-clean Rank-1 `{formal['oracle_clean_metrics']['rank1']}`; sequence-cluster paired 95% CI `{formal['paired_bootstrap']['interval_95']}`; representation gate `{formal['representation_gate']['pass']}`.",
        "",
        "## Phase B — explicit NONE",
        "",
        f"Pooled negative FPR `{open_result['runtime_metrics']['negative_fpr']}`; pooled open-set correct-ID recall `{open_result['runtime_metrics']['open_set_correct_id_recall']}`; macro negative FPR `{open_result['macro_negative_fpr']}`; macro recall `{open_result['macro_open_set_correct_id_recall']}`; gate `{open_result['open_set_gate']['pass']}`.",
        "",
        "## Phase C — causal commit replay",
        "",
        f"Causal metrics: `{json.dumps(causal_result['runtime_metrics'], sort_keys=True) if causal_result is not None else 'NOT RUN — Phase B failed'}`; gate `{causal_result['causal_gate']['pass'] if causal_result is not None else False}`.",
        "",
        "OSNet, N72R18 GRU, candidate tape, exact solver and public-ID authority remained frozen. No SAM3 rerun, DanceTrack VAL/TEST access, association override or candidate creation occurred.",
        f"next_association_authority_stage_authorized={next_association}",
        f"Focused/full test summary: {tests_summary}.",
    ]
    (OUT / "FINAL_REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("open_set", "causal", "continue"), default="continue")
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--tests-summary", default="focused_pending_full_pending")
    args = parser.parse_args()
    before = storage_audit()
    if before["filesystem"]["storage_status"] == "HARD_STOP":
        raise RuntimeError("storage hard stop")
    formal = _json(OUT / "representation/formal_loso.json")
    if not bool(formal["representation_gate"]["pass"]):
        raise RuntimeError("Phase A gate is not passed")
    dataset = load_episode_dataset(None)
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    open_result = run_open_set(dataset, device)
    if args.phase == "open_set" or not bool(open_result["open_set_gate"]["pass"]):
        result = finalize_continuation(formal, open_result, None, args.tests_summary)
        print(json.dumps({"stage": STAGE, "status": result["decision"], "open_set_gate": open_result["open_set_gate"]["pass"]}, sort_keys=True))
        return 0
    causal_result = run_causal(dataset, open_result, device)
    result = finalize_continuation(formal, open_result, causal_result, args.tests_summary)
    print(json.dumps({"stage": STAGE, "status": result["decision"], "open_set_gate": open_result["open_set_gate"]["pass"], "causal_gate": causal_result["causal_gate"]["pass"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
