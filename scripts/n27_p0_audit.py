#!/usr/bin/env python3
"""P0 regression tests for the frozen B10 anchor and APCR-S constraints."""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import torch

from n27_apcr_model import APCRS, CONTEXT_NAMES, feature_tensors


ROOT = Path(".")
OUT = ROOT / "outputs/n27"


def frozen_b10(root: np.ndarray, candidates: np.ndarray, positive: np.ndarray, negative: np.ndarray) -> np.ndarray:
    root_sim = candidates @ root
    positive_sim = np.max(candidates @ positive.T, axis=1) if len(positive) else np.zeros(len(candidates), dtype=np.float32)
    negative_sim = np.max(candidates @ negative.T, axis=1) if len(negative) else np.zeros(len(candidates), dtype=np.float32)
    base = np.maximum(root_sim, positive_sim) if len(positive) else root_sim
    penalty = np.maximum(0.0, negative_sim - base + 0.02) if len(negative) else np.zeros_like(base)
    return base - 0.8 * penalty


def main() -> None:
    torch.manual_seed(27)
    np.random.seed(27)
    model = APCRS()
    batch_size = 32
    candidate_mask = torch.ones(batch_size, 5, dtype=torch.bool)
    candidate_mask[::7, -1] = False
    b10 = torch.randn(batch_size, 5)
    base = {
        "candidate_mask": candidate_mask,
        "b10_score": b10,
        "positive_similarity": torch.rand(batch_size, 5) * 2 - 1,
        "negative_similarity": torch.rand(batch_size, 5) * 2 - 1,
        "hard_similarity": torch.rand(batch_size, 5) * 2 - 1,
        "detector_score": torch.rand(batch_size, 5),
        "candidate_count": torch.full((batch_size,), 0.8),
        "has_positive": torch.ones(batch_size),
        "has_negative": torch.ones(batch_size),
        "has_hard": torch.ones(batch_size),
        "positive_count": torch.full((batch_size,), 0.5),
        "negative_count": torch.full((batch_size,), 0.5),
        "hard_count": torch.full((batch_size,), 0.5),
        "positive_age": torch.full((batch_size,), 0.2),
        "negative_age": torch.full((batch_size,), 0.2),
        "hard_age": torch.full((batch_size,), 0.2),
    }
    features = feature_tensors(base)
    off = model(features, residual_off=True)
    valid_off_error = float(torch.max(torch.abs(off["scores"][candidate_mask] - b10[candidate_mask])).item())

    no_memory = dict(base)
    no_memory["has_positive"] = torch.zeros(batch_size)
    no_memory["has_negative"] = torch.zeros(batch_size)
    no_memory["has_hard"] = torch.zeros(batch_size)
    no_memory_delta = model(feature_tensors(no_memory))["delta"]
    no_memory_max = float(torch.max(torch.abs(no_memory_delta)).item())

    positive_delta = model(features, mode="positive_only")["delta"]
    negative_delta = model(features, mode="negative_only")["delta"]
    hard_delta = model(features, mode="hard_negative")["delta"]
    signs = {
        "positive_min": float(positive_delta.min().item()),
        "negative_max": float(negative_delta.max().item()),
        "hard_max": float(hard_delta.max().item()),
        "positive_upper": float(positive_delta.max().item()),
        "negative_lower": float(negative_delta.min().item()),
        "hard_lower": float(hard_delta.min().item()),
    }

    context = features["detector_score"].new_zeros(1, 5, len(CONTEXT_NAMES))
    # Feed a constant context directly to the gates so the grid audit changes
    # only the relevant memory similarity.
    context[:, :, 0] = 0.5
    context[:, :, 1] = 1.0
    context[:, :, 2:5] = 1.0
    context[:, :, 5:8] = 0.5
    context[:, :, 8:11] = 0.2
    grid = torch.linspace(-1.0, 1.0, 101).view(1, 101)
    ctx_grid = context[:, :1].expand(1, len(grid[0]), -1)
    positive_gate = model.config.positive_bound * model.positive_gate(ctx_grid, grid)
    negative_gate = -model.config.negative_bound * model.negative_gate(ctx_grid, grid)
    positive_diff = torch.diff(positive_gate, dim=1)
    negative_diff = torch.diff(negative_gate, dim=1)
    monotonic = {
        "positive_min_first_difference": float(positive_diff.min().item()),
        "negative_max_first_difference": float(negative_diff.max().item()),
        "positive_non_decreasing": bool(torch.all(positive_diff >= -1e-7)),
        "negative_penalty_non_decreasing": bool(torch.all(negative_diff <= 1e-7)),
    }

    root = np.random.randn(1280).astype(np.float32)
    root /= np.linalg.norm(root)
    candidates = np.random.randn(5, 1280).astype(np.float32)
    candidates /= np.linalg.norm(candidates, axis=1, keepdims=True)
    positive = np.random.randn(3, 1280).astype(np.float32)
    positive /= np.linalg.norm(positive, axis=1, keepdims=True)
    negative = np.random.randn(4, 1280).astype(np.float32)
    negative /= np.linalg.norm(negative, axis=1, keepdims=True)
    scores_a = frozen_b10(root, candidates, positive, negative)
    positive_base = np.maximum(candidates @ root, np.max(candidates @ positive.T, axis=1))
    negative_max = np.max(candidates @ negative.T, axis=1)
    scores_b = positive_base - 0.8 * np.maximum(0.0, negative_max - positive_base + 0.02)
    anchor_formula_error = float(np.max(np.abs(scores_a - scores_b)))
    audit = {
        "phase": "N27_P0",
        "residual_off_valid_candidate_max_abs_error": valid_off_error,
        "no_memory_delta_max_abs": no_memory_max,
        "sign_and_bound": signs,
        "monotonicity": monotonic,
        "anchor_formula_independent_reproduction_max_abs_error": anchor_formula_error,
        "context_names": list(CONTEXT_NAMES),
        "forbidden_context_names_present": any(name in {"sequence_id", "rank_embedding", "camera_x", "camera_y"} for name in CONTEXT_NAMES),
        "bounds": {"positive": model.config.positive_bound, "negative": model.config.negative_bound, "hard": model.config.hard_bound},
        "val25_read": False,
    }
    path = OUT / "monotonicity_audit.json"
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)
    anchor_path = OUT / "anchor_reproduction.json"
    temporary = anchor_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps({"phase": "N27_P0", "formula": "B10", "max_abs_error": anchor_formula_error, "residual_off_valid_candidate_max_abs_error": valid_off_error, "val25_read": False}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, anchor_path)
    print(json.dumps(audit, indent=2, sort_keys=True), flush=True)
    print("N27_P0_AUDIT_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
