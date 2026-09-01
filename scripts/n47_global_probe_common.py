#!/usr/bin/env python3
"""Isolated N47 global candidate-to-public-ID fusion probe.

This module is deliberately separate from production MOT/OVMOT and from the
N44 checkpoint.  It predicts a candidate/public-ID appearance logit from
causal cell/state signals, adds that logit to the frozen branch score matrix,
and runs one global Hungarian assignment with explicit NONE dummy columns.
There is no owner-by-column proposal gate, so candidate swaps are legal.
GT is used only by the offline dataset/posthoc scripts, never by runtime.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from torch import nn


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/n47_global_probe"
TRAIN = OUT / "training"
RUNTIME = OUT / "replay/runtime"
POSTHOC = OUT / "replay/posthoc"
ATTEMPTS = OUT / "attempts"
EVENTS = ROOT / "outputs/n37/real_event_manifest.json"
N42_RUNTIME = ROOT / "outputs/n42/replay/runtime/t0"
N42_PROTOCOL = ROOT / "outputs/n42/training/training_protocol.json"
N43_MAP = ROOT / "outputs/n43/training/dataset_manifest.json"
N46_DIAG = ROOT / "outputs/n46/diagnosis_final_repair2/structural_diagnosis.json"
N46_GATE = ROOT / "outputs/n46/diagnosis_final_repair2/final_gate.json"
CHECKPOINT = TRAIN / "n47_global_fusion_probe.pt"
DATASET = TRAIN / "global_assignment_dataset.npz"
DATASET_MANIFEST = TRAIN / "global_assignment_dataset_manifest.json"
TRAIN_MANIFEST = TRAIN / "training_manifest.json"
PROTOCOL = OUT / "probe_protocol.json"

VARIANTS = ("M0", "M1", "M2", "M3", "M4")
HORIZONS = (20, 50, 100)
SEED = 4747
FEATURE_NAMES = (
    "base_score_tanh",
    "appearance_delta_tanh",
    "appearance_memory_tanh",
    "fused_score_tanh",
    "candidate_confidence",
    "candidate_age_norm",
    "candidate_rank_norm",
    "frame_offset_norm",
)
FEATURE_DIM = len(FEATURE_NAMES)
NONE_SCORE = -1.0e8
HARD_NEGATIVE = -1.0e7
SCORE_SCALE = 1.0
MAX_NEGATIVES_PER_POSITIVE = 8
MAX_EPOCHS = 25
PATIENCE = 5
BATCH_SIZE = 512
LEARNING_RATE = 2.0e-3
WEIGHT_DECAY = 1.0e-4
L2_LOGIT = 1.0e-3
BOOTSTRAP_SEED = 4747
BOOTSTRAP_REPS = 2000


def load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def event_map() -> dict[str, dict[str, Any]]:
    payload = load(EVENTS)
    if payload.get("status") != "PASS" or len(payload.get("events", [])) != 24:
        raise RuntimeError("frozen N37 event manifest is invalid")
    return {str(item["event"]["event_id"]): item["event"] for item in payload["events"]}


def sequence_split() -> dict[str, str]:
    payload = load(N42_PROTOCOL)
    output: dict[str, str] = {}
    for name in ("train", "validation", "holdout"):
        for sequence in payload["sequence_split"][name]:
            sequence = str(sequence)
            if sequence in output:
                raise RuntimeError(f"sequence split overlap: {sequence}")
            output[sequence] = name
    return output


def score_matrix(audit: dict[str, Any], key: str = "fused_scores") -> np.ndarray:
    value = np.asarray(audit.get(key, []), dtype=np.float32)
    if value.ndim != 2 or not np.all(np.isfinite(value)):
        raise ValueError(f"invalid finite score matrix {key}: {value.shape}")
    return value


def candidate_list(audit: dict[str, Any]) -> list[dict[str, Any]]:
    candidates = audit.get("candidates", [])
    if not isinstance(candidates, list):
        raise ValueError("candidate list missing")
    native_ids = [int(x["native_tid"]) for x in candidates]
    if len(native_ids) != len(set(native_ids)):
        raise ValueError("native candidate IDs are not unique")
    return candidates


def hungarian_with_none(scores: np.ndarray) -> np.ndarray:
    """Return candidate row -> public-ID column, or -1 for a NONE dummy."""
    matrix = np.asarray(scores, dtype=np.float32)
    if matrix.ndim != 2 or not np.all(np.isfinite(matrix)):
        raise ValueError("invalid global assignment score matrix")
    # A hard-negative/NONE cell must never win a tie against an explicit NONE
    # dummy.  Keep the input value in the audit, but use a strictly lower
    # working value for the assignment solver.
    working = matrix.copy()
    working[working <= HARD_NEGATIVE] = NONE_SCORE - 16.0
    dummy = np.full((matrix.shape[0], matrix.shape[0]), NONE_SCORE, dtype=np.float32)
    expanded = np.concatenate([working, dummy], axis=1)
    rows, columns = linear_sum_assignment(-expanded)
    result = np.full(matrix.shape[0], -1, dtype=np.int64)
    result[rows] = columns
    return result


def normalize_assignment(assignment: Any, public_id_count: int) -> list[int]:
    return [int(x) if 0 <= int(x) < int(public_id_count) else -1 for x in assignment]


def assignment_public_ids(assignment: Any, pids: list[int]) -> list[int | None]:
    return [int(pids[int(col)]) if 0 <= int(col) < len(pids) else None for col in assignment]


def rows_from_assignment(audit: dict[str, Any], assignment: Any) -> list[dict[str, Any]]:
    candidates = candidate_list(audit)
    pids = [int(x) for x in audit.get("public_id_order", [])]
    normalized = normalize_assignment(assignment, len(pids))
    if len(normalized) != len(candidates):
        raise ValueError("assignment/candidate row count mismatch")
    rows = []
    for index, candidate in enumerate(candidates):
        rows.append({
            "candidate_index": int(candidate.get("index", index)),
            "native_tid": int(candidate["native_tid"]),
            "box": candidate["box"],
            "confidence": float(candidate.get("confidence", 0.0)),
            "public_id": int(pids[normalized[index]]) if normalized[index] >= 0 else None,
        })
    return rows


def feature_row(audit: dict[str, Any], row: int, column: int, frame_offset: int) -> np.ndarray:
    base = score_matrix(audit, "base_scores_before_appearance")
    delta = score_matrix(audit, "appearance_score_deltas")
    memory = score_matrix(audit, "appearance_memory_scores")
    fused = score_matrix(audit, "fused_scores")
    if not (base.shape == delta.shape == memory.shape == fused.shape):
        raise ValueError("source score matrix shapes differ")
    candidates = candidate_list(audit)
    if row >= len(candidates) or column >= base.shape[1]:
        raise IndexError((row, column, base.shape))
    candidate = candidates[row]
    confidence = float(np.clip(candidate.get("confidence", 0.0), 0.0, 1.0))
    age = max(0.0, float(candidate.get("native_age", 0.0)))
    rank = float(row) / max(len(candidates) - 1, 1)
    values = np.asarray([
        np.tanh(float(base[row, column]) / 5.0),
        np.tanh(float(delta[row, column]) / 2.0),
        np.tanh(float(memory[row, column]) / 2.0),
        np.tanh(float(fused[row, column]) / 5.0),
        confidence,
        min(age / 2000.0, 1.0),
        rank,
        min(max(float(frame_offset), 0.0) / 100.0, 1.0),
    ], dtype=np.float32)
    if values.shape != (FEATURE_DIM,) or not np.all(np.isfinite(values)):
        raise ValueError("nonfinite N47 causal feature")
    return values


def feature_matrix(audit: dict[str, Any], frame_offset: int) -> np.ndarray:
    matrix = score_matrix(audit, "fused_scores")
    return np.asarray([feature_row(audit, i, j, frame_offset) for i in range(matrix.shape[0]) for j in range(matrix.shape[1])], dtype=np.float32)


class GlobalFusionHead(nn.Module):
    """Candidate-level additive appearance logit; no public-ID input."""

    def __init__(self, input_dim: int = FEATURE_DIM, hidden: int = 48) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(input_dim, hidden), nn.ReLU(), nn.Linear(hidden, 24), nn.ReLU(), nn.Linear(24, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def load_checkpoint(path: Path = CHECKPOINT, device: str = "cpu") -> tuple[GlobalFusionHead, dict[str, Any]]:
    payload = torch.load(path, map_location=device, weights_only=False)
    if not isinstance(payload, dict) or payload.get("protocol") != "N47_GLOBAL_CANDIDATE_ASSIGNMENT_PROBE_V1":
        raise ValueError("invalid N47 checkpoint protocol")
    if payload.get("production_authorized") is not False:
        raise ValueError("N47 checkpoint must record production_authorized=false")
    model = GlobalFusionHead(int(payload.get("input_dim", FEATURE_DIM)), int(payload.get("hidden", 48)))
    model.load_state_dict(payload["state_dict"])
    model.to(device).eval()
    return model, payload


def predict_logit(model: GlobalFusionHead, features: np.ndarray) -> np.ndarray:
    with torch.no_grad():
        output = model(torch.as_tensor(features, dtype=torch.float32)).cpu().numpy()
    if not np.all(np.isfinite(output)):
        raise ValueError("N47 predicted appearance logit is nonfinite")
    return output.astype(np.float32)


def apply_global_probe(audit: dict[str, Any], model: GlobalFusionHead, frame_offset: int) -> dict[str, Any]:
    """Apply the candidate-level logit to every finite cell, then global Hungarian."""
    base = score_matrix(audit, "fused_scores")
    pids = [int(x) for x in audit.get("public_id_order", [])]
    if base.shape[1] != len(pids):
        raise ValueError("public-ID axis does not match score matrix")
    features = feature_matrix(audit, frame_offset)
    logits = predict_logit(model, features).reshape(base.shape)
    finite = base > HARD_NEGATIVE
    adjusted = base.copy()
    adjusted[finite] = base[finite] + SCORE_SCALE * logits[finite]
    adjusted[~finite] = base[~finite]
    baseline = hungarian_with_none(base)
    plus = hungarian_with_none(adjusted)
    changed_cells = [
        {"candidate_index": int(i), "column": int(j), "public_id": int(pids[j]), "baseline_score": float(base[i, j]), "appearance_logit": float(logits[i, j]), "adjusted_score": float(adjusted[i, j])}
        for i, j in zip(*np.where(np.abs(adjusted - base) > 1.0e-12))
    ]
    hard = ~finite
    return {
        "baseline_scores": base.astype(float).tolist(),
        "predicted_appearance_logit": logits.astype(float).tolist(),
        "adjusted_scores": adjusted.astype(float).tolist(),
        "baseline_assignment": normalize_assignment(baseline, len(pids)),
        "plus_assignment": normalize_assignment(plus, len(pids)),
        "baseline_assignment_public_ids": assignment_public_ids(baseline, pids),
        "plus_assignment_public_ids": assignment_public_ids(plus, pids),
        "changed_cells": changed_cells,
        "assignment_changed": bool(np.any(normalize_assignment(baseline, len(pids)) != normalize_assignment(plus, len(pids)))),
        "hard_negative_preserved": bool(np.array_equal(adjusted[hard], base[hard])),
        "explicit_none": True,
        "swap_allowed": True,
        "runtime_future_gt_used": False,
    }
