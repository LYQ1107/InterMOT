#!/usr/bin/env python3
"""Shared N71 global candidate-by-identity scorer and explicit-NONE solver.

This module is isolated from the production association classes.  Public IDs
are used only for offline labels and audit output; the neural inputs contain
embeddings, scalar context, and positional/role flags, never an ID value.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Iterable

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import n70_association_common as n70  # noqa: E402


FEATURE_DIM = 512
CONTEXT_DIM = 15
PROJECTION_DIM = 64
DATA_MANIFEST = ROOT / "outputs/N71/training/global_matrix_dataset_manifest_attempt5.json"
DATA_ROOT = Path("/path/to/cache/SAM3_InterMOT_N71/training/N71_GLOBAL_MATRIX_DATASET_ATTEMPT5")
SPLIT_CODE = {"train": 0, "validation": 1, "holdout": 2}
SPLIT_NAMES = {value: key for key, value in SPLIT_CODE.items()}
HYSTERESIS_MARGIN = 0.15
NONE_COLUMN_SENTINEL = -1.0e9


def sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def load_arrays(manifest_path: Path = DATA_MANIFEST) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "PASS_GLOBAL_MATRIX_MATERIALIZED":
        raise RuntimeError("global matrix dataset manifest is not PASS")
    root = Path(manifest["output_root"])
    names = [
        "candidate", "identity_memory", "human_anchor", "hard_negative", "context",
        "label", "none_label", "group", "split", "frame", "variant", "candidate_slot",
        "identity_slot", "target_slot", "target_row", "target_present", "base_score",
        "group_offsets",
    ]
    arrays = {}
    for name in names:
        path = root / f"{name}.npy"
        if not path.is_file():
            raise FileNotFoundError(path)
        arrays[name] = np.load(path, mmap_mode="r", allow_pickle=False)
    cells = int(manifest["cell_count"])
    groups = int(manifest["group_count"])
    if arrays["group_offsets"].shape != (groups + 1,) or int(arrays["group_offsets"][-1]) != cells:
        raise RuntimeError("global matrix offsets do not match manifest")
    if arrays["candidate"].shape != (cells, FEATURE_DIM) or arrays["identity_memory"].shape != (cells, FEATURE_DIM):
        raise RuntimeError("global matrix feature shape mismatch")
    if arrays["context"].shape != (cells, CONTEXT_DIM):
        raise RuntimeError("global matrix context shape mismatch")
    if not np.all(np.isfinite(arrays["candidate"])) or not np.all(np.isfinite(arrays["identity_memory"])):
        raise RuntimeError("global matrix features are nonfinite")
    if not np.all(np.isfinite(arrays["context"])) or not np.all(np.isfinite(arrays["base_score"])):
        raise RuntimeError("global matrix context/base scores are nonfinite")
    return arrays, manifest


def group_ids_for_split(arrays: dict[str, np.ndarray], split_name: str) -> list[int]:
    code = SPLIT_CODE[split_name]
    offsets = arrays["group_offsets"]
    result = []
    for group in range(offsets.size - 1):
        start = int(offsets[group])
        if int(arrays["split"][start]) == code:
            result.append(group)
    return result


def group_batches(arrays: dict[str, np.ndarray], group_ids: Iterable[int], max_cells: int, rng: np.random.Generator | None = None) -> Iterable[tuple[list[int], np.ndarray]]:
    ordered = list(group_ids)
    if rng is not None:
        rng.shuffle(ordered)
    offsets = arrays["group_offsets"]
    current: list[int] = []
    cells = 0
    for group in ordered:
        size = int(offsets[group + 1]) - int(offsets[group])
        if current and cells + size > max_cells:
            indices = np.concatenate([np.arange(int(offsets[g]), int(offsets[g + 1]), dtype=np.int64) for g in current])
            yield current, indices
            current, cells = [], 0
        current.append(group)
        cells += size
    if current:
        indices = np.concatenate([np.arange(int(offsets[g]), int(offsets[g + 1]), dtype=np.int64) for g in current])
        yield current, indices


def context_normalization(arrays: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    offsets = arrays["group_offsets"]
    train_groups = group_ids_for_split(arrays, "train")
    ranges = [np.arange(int(offsets[g]), int(offsets[g + 1]), dtype=np.int64) for g in train_groups]
    indices = np.concatenate(ranges)
    mean = np.asarray(arrays["context"][indices], dtype=np.float32).mean(axis=0, dtype=np.float64).astype(np.float32)
    std = np.asarray(arrays["context"][indices], dtype=np.float32).std(axis=0, dtype=np.float64).astype(np.float32)
    std[std < 1.0e-6] = 1.0
    return mean, std


def set_seed(seed: int) -> None:
    import random
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


class GlobalMatrixScorer:  # subclass is defined lazily to keep CPU audits torch-free
    pass


def build_model():
    import torch.nn as nn
    import torch.nn.functional as F
    import torch

    class _GlobalMatrixScorer(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.candidate_projection = nn.Linear(FEATURE_DIM, PROJECTION_DIM, bias=False)
            self.identity_projection = nn.Linear(FEATURE_DIM, PROJECTION_DIM, bias=False)
            self.anchor_projection = nn.Linear(FEATURE_DIM, PROJECTION_DIM, bias=False)
            self.negative_projection = nn.Linear(FEATURE_DIM, PROJECTION_DIM, bias=False)
            self.context_projection = nn.Sequential(
                nn.Linear(CONTEXT_DIM, 32), nn.LayerNorm(32), nn.ReLU()
            )
            pair_dim = PROJECTION_DIM * 8 + 32
            self.pair_norm = nn.LayerNorm(pair_dim)
            self.pair_head = nn.Sequential(
                nn.Linear(pair_dim, 128), nn.ReLU(), nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 1)
            )
            # NONE is a candidate-level alternative in the augmented
            # Hungarian matrix. It must not vary with the identity column:
            # the per-cell context contains identity-role/base-score fields.
            # The first attempt fed that full context and then selected
            # column zero in the solver, which was not a valid
            # candidate-specific NONE score.
            self.none_head = nn.Sequential(
                nn.Linear(PROJECTION_DIM, 64), nn.ReLU(), nn.Linear(64, 1)
            )

        def forward(self, candidate, identity_memory, human_anchor, hard_negative, context):
            candidate = F.normalize(candidate, dim=-1, eps=1.0e-6)
            identity_memory = F.normalize(identity_memory, dim=-1, eps=1.0e-6)
            human_anchor = F.normalize(human_anchor, dim=-1, eps=1.0e-6)
            hard_negative = F.normalize(hard_negative, dim=-1, eps=1.0e-6)
            c = self.candidate_projection(candidate)
            m = self.identity_projection(identity_memory)
            a = self.anchor_projection(human_anchor)
            h = self.negative_projection(hard_negative)
            context = self.context_projection(context)
            pair = torch.cat([c, m, a, h, c * m, torch.abs(c - m), c * a, torch.abs(c - a), context], dim=-1)
            pair_logit = self.pair_head(self.pair_norm(pair)).squeeze(-1)
            none_logit = self.none_head(c).squeeze(-1)
            return pair_logit, none_logit

    return _GlobalMatrixScorer()


def model_metadata(model: Any) -> dict[str, Any]:
    return {
        "name": "N71_SHARED_PROJECTION_GLOBAL_CANDIDATE_IDENTITY_MATRIX_SCORER",
        "parameter_count": int(sum(p.numel() for p in model.parameters())),
        "feature_dim": FEATURE_DIM,
        "context_dim": CONTEXT_DIM,
        "projection_dim": PROJECTION_DIM,
        "outputs": ["candidate_identity_pair_logit", "candidate_specific_none_logit"],
        "explicit_none": True,
        "numeric_public_id_feature": False,
        "numeric_target_native_id_feature": False,
        "runtime_future_gt_used": False,
    }


def tensors_for_indices(arrays: dict[str, np.ndarray], indices: np.ndarray, mean: np.ndarray, std: np.ndarray, device: Any):
    import torch
    index = np.asarray(indices, dtype=np.int64)
    context = (np.asarray(arrays["context"][index], dtype=np.float32) - mean) / std
    return (
        torch.as_tensor(np.asarray(arrays["candidate"][index], dtype=np.float32), device=device),
        torch.as_tensor(np.asarray(arrays["identity_memory"][index], dtype=np.float32), device=device),
        torch.as_tensor(np.asarray(arrays["human_anchor"][index], dtype=np.float32), device=device),
        torch.as_tensor(np.asarray(arrays["hard_negative"][index], dtype=np.float32), device=device),
        torch.as_tensor(context, device=device),
        torch.as_tensor(np.asarray(arrays["label"][index], dtype=np.float32), device=device),
        torch.as_tensor(np.asarray(arrays["none_label"][index], dtype=np.float32), device=device),
    )


def group_loss(model: Any, arrays: dict[str, np.ndarray], group_list: list[int], indices: np.ndarray, mean: np.ndarray, std: np.ndarray, device: Any, *, include_details: bool = True):
    import torch
    import torch.nn.functional as F
    tensors = tensors_for_indices(arrays, indices, mean, std, device)
    pair_logits, none_logits = model(*tensors[:5])
    labels, none_labels = tensors[5], tensors[6]
    pos_weight = torch.tensor(3.0, device=device)
    cell_bce = F.binary_cross_entropy_with_logits(pair_logits, labels, pos_weight=pos_weight)
    offsets = arrays["group_offsets"]
    group_ce_terms = []
    margin_terms = []
    none_terms = []
    cursor = 0
    for group in group_list:
        start, end = int(offsets[group]), int(offsets[group + 1])
        size = end - start
        local_pair = pair_logits[cursor:cursor + size]
        local_none = none_logits[cursor:cursor + size]
        n = int(np.max(arrays["candidate_slot"][start:end])) + 1
        p = int(np.max(arrays["identity_slot"][start:end])) + 1
        pair_grid = local_pair.reshape(n, p)
        label_grid = labels[cursor:cursor + size].reshape(n, p)
        none_grid = local_none.reshape(n, p)
        targets = []
        for row in range(n):
            positive = (label_grid[row] > 0.5).nonzero(as_tuple=False).reshape(-1)
            targets.append(int(positive[0].detach().cpu()) if positive.numel() else p)
        target_tensor = torch.as_tensor(targets, dtype=torch.long, device=device)
        augmented = torch.cat([pair_grid, none_grid[:, :1]], dim=1)
        group_ce_terms.append(F.cross_entropy(augmented, target_tensor))
        none_target = (label_grid.sum(dim=1) <= 0.5).float()
        none_terms.append(F.binary_cross_entropy_with_logits(none_grid[:, 0], none_target))
        for row, target in enumerate(targets):
            if target < p:
                negative = torch.cat([pair_grid[row, :target], pair_grid[row, target + 1:]])
                if negative.numel():
                    margin_terms.append(F.relu(torch.max(negative) - pair_grid[row, target] + 0.20))
        cursor += size
    group_ce = torch.stack(group_ce_terms).mean() if group_ce_terms else torch.zeros((), device=device)
    margin = torch.stack(margin_terms).mean() if margin_terms else torch.zeros((), device=device)
    none_loss = torch.stack(none_terms).mean() if none_terms else torch.zeros((), device=device)
    total = cell_bce + 0.5 * group_ce + 0.5 * margin + 0.25 * none_loss
    details = {
        "total": float(total.detach().cpu()),
        "cell_bce": float(cell_bce.detach().cpu()),
        "group_softmax": float(group_ce.detach().cpu()),
        "hard_negative_margin": float(margin.detach().cpu()),
        "none_presence": float(none_loss.detach().cpu()),
    }
    return total, details


def explicit_none_hungarian(identity_scores: np.ndarray, none_scores: np.ndarray, public_ids: list[int], candidate_rows: list[dict[str, Any]]) -> dict[str, Any]:
    from scipy.optimize import linear_sum_assignment
    scores = np.asarray(identity_scores, dtype=np.float64)
    none = np.asarray(none_scores, dtype=np.float64).reshape(-1)
    if scores.ndim != 2 or scores.shape[0] != none.size or scores.shape[1] != len(public_ids):
        raise ValueError(f"assignment shape mismatch scores={scores.shape} none={none.shape} public={len(public_ids)}")
    if not np.all(np.isfinite(scores)) or not np.all(np.isfinite(none)):
        raise ValueError("assignment scores are nonfinite")
    n, p = scores.shape
    augmented = np.full((n, p + n), NONE_COLUMN_SENTINEL, dtype=np.float64)
    augmented[:, :p] = scores
    augmented[np.arange(n), p + np.arange(n)] = none
    row_ind, col_ind = linear_sum_assignment(-augmented)
    assigned_public: list[int | None] = [None] * n
    assigned_column: list[int] = [-1] * n
    for row, col in zip(row_ind.tolist(), col_ind.tolist()):
        assigned_column[int(row)] = int(col)
        if col < p:
            assigned_public[int(row)] = int(public_ids[col])
    used = [value for value in assigned_public if value is not None]
    if len(used) != len(set(used)):
        raise RuntimeError("explicit NONE assignment produced duplicate public IDs")
    return {
        "assigned_column": assigned_column,
        "assigned_public_ids": assigned_public,
        "public_id_order": [int(value) for value in public_ids],
        "none_column_start": p,
        "top_assignments": [
            {
                "candidate_index": int(i),
                "native_tid": int(candidate_rows[i].get("native_tid", -1)),
                "assigned_column": int(assigned_column[i]),
                "assigned_public_id": assigned_public[i],
                "assigned_score": float(augmented[i, assigned_column[i]]) if assigned_column[i] >= 0 else None,
                "none_score": float(none[i]),
            }
            for i in range(n)
        ],
        "augmented_score_matrix": augmented.tolist(),
    }


def apply_temporal_guard(assignment: dict[str, Any], identity_scores: np.ndarray, none_scores: np.ndarray, candidate_rows: list[dict[str, Any]], public_ids: list[int], frame: int, *, target_native_id: int | None = None, history: dict[int, tuple[int | None, int]] | None = None, window_frames: int = 3, hysteresis_margin: float = HYSTERESIS_MARGIN) -> tuple[dict[str, Any], dict[int, tuple[int | None, int]], dict[str, Any]]:
    history = {} if history is None else dict(history)
    scores = np.asarray(identity_scores, dtype=np.float64)
    none = np.asarray(none_scores, dtype=np.float64)
    current = list(assignment["assigned_public_ids"])
    current_cols = list(assignment["assigned_column"])
    used = {value for value in current if value is not None}
    retained = []
    for i, row in enumerate(candidate_rows):
        native = int(row.get("native_tid", -1))
        previous = history.get(native)
        if previous is None or int(frame) - int(previous[1]) > int(window_frames):
            continue
        previous_public, _previous_frame = previous
        if previous_public is None or native == target_native_id or previous_public not in public_ids:
            continue
        previous_col = public_ids.index(int(previous_public))
        new_public = current[i]
        if new_public == previous_public:
            continue
        # The guard is conservative: only retain a previous public identity
        # when it remains a finite valid alternative and the new choice does
        # not exceed it by the frozen hysteresis margin.  It never revives a
        # row whose explicit assignment is reserved for the target scope.
        previous_score = float(scores[i, previous_col])
        new_score = float(none[i] if new_public is None else scores[i, public_ids.index(int(new_public))])
        if not np.isfinite(previous_score) or not np.isfinite(new_score) or previous_score < new_score - float(hysteresis_margin):
            continue
        if previous_public in used:
            continue
        current[i] = int(previous_public)
        current_cols[i] = int(previous_col)
        used.add(int(previous_public))
        retained.append({"candidate_index": i, "native_tid": native, "retained_public_id": int(previous_public), "previous_frame": int(_previous_frame), "previous_score": previous_score, "new_score": new_score})
    guarded = dict(assignment)
    guarded["assigned_public_ids"] = current
    guarded["assigned_column"] = current_cols
    guarded["temporal_guard_retained"] = retained
    guarded["temporal_guard_window_frames"] = int(window_frames)
    guarded["temporal_guard_hysteresis_margin"] = float(hysteresis_margin)
    for i, row in enumerate(candidate_rows):
        history[int(row.get("native_tid", -1))] = (current[i], int(frame))
    return guarded, history, {"retained_count": len(retained), "retained": retained, "runtime_future_gt_used": False}


def event_anchor_and_negative(event: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    anchor = np.asarray(event["human_embedding"], dtype=np.float32).reshape(-1)
    negative = np.asarray(event.get("negative_embedding", np.zeros(FEATURE_DIM)), dtype=np.float32).reshape(-1)
    if anchor.shape != (FEATURE_DIM,) or negative.shape != (FEATURE_DIM,):
        raise ValueError("event anchor/negative feature shape mismatch")
    return anchor, negative


def build_replay_cells(frame: dict[str, Any], event: dict[str, Any]) -> dict[str, Any]:
    """Build exactly the frozen materializer feature contract for one cache frame."""
    candidates = frame["candidate_rows"]
    rows = frame["rows"]
    public_ids = [int(value) for value in frame["public_id_order"]]
    candidate = np.asarray(frame["candidate_features_512"], dtype=np.float32)
    memory = np.asarray(frame["memory_vectors_512"], dtype=np.float32)
    memory_valid = np.asarray(frame["memory_valid"], dtype=bool).reshape(-1)
    score = np.asarray(frame["score_matrix"], dtype=np.float32)
    scalar = np.asarray(frame["scalar_features_8"], dtype=np.float32)
    n, p = len(candidates), len(public_ids)
    if candidate.shape != (n, FEATURE_DIM) or memory.shape != (p, FEATURE_DIM) or score.shape != (n, p) or scalar.shape != (n * p, 8):
        raise RuntimeError(f"replay cell shape mismatch {frame.get('event_id')}/{frame.get('frame')}")
    assignment = np.asarray(frame["assignment_columns"], dtype=np.int64).reshape(n)
    target_slot = public_ids.index(int(event["target_public_id"])) if int(event["target_public_id"]) in public_ids else -1
    target_row = next((i for i, row in enumerate(rows) if int(row.get("native_tid", -1)) == int(event["target_native_id"])), -1)
    assigned_grid = (assignment[:, None] == np.arange(p, dtype=np.int64)[None, :]).astype(np.float32)
    occupancy = np.bincount(assignment[assignment >= 0], minlength=p).astype(np.float32) / max(1, n)
    rank = np.arange(n, dtype=np.float32) / max(1, n - 1)
    count_norm = np.full(n, min(1.0, n / 32.0), dtype=np.float32)
    role = np.zeros(p, dtype=np.float32)
    if target_slot >= 0:
        role[target_slot] = 1.0
    anchor, negative = event_anchor_and_negative(event)
    anchors = np.zeros((p, FEATURE_DIM), dtype=np.float32)
    negatives = np.zeros((p, FEATURE_DIM), dtype=np.float32)
    if target_slot >= 0:
        anchors[target_slot] = anchor
        negatives[target_slot] = negative
    scalar_grid = scalar.reshape(n, p, 8)
    context = np.concatenate([
        scalar_grid,
        score[:, :, None],
        np.broadcast_to(role[None, :, None], (n, p, 1)),
        np.broadcast_to(memory_valid.astype(np.float32)[None, :, None], (n, p, 1)),
        assigned_grid[:, :, None],
        np.broadcast_to(rank[:, None, None], (n, p, 1)),
        np.broadcast_to(count_norm[:, None, None], (n, p, 1)),
        np.broadcast_to(occupancy[None, :, None], (n, p, 1)),
    ], axis=2).reshape(n * p, CONTEXT_DIM)
    return {
        "candidate": np.repeat(candidate, p, axis=0),
        "identity_memory": np.tile(memory, (n, 1)),
        "human_anchor": np.tile(anchors, (n, 1)),
        "hard_negative": np.tile(negatives, (n, 1)),
        "context": context,
        "base_score": score,
        "public_ids": public_ids,
        "candidate_rows": candidates,
        "rows": rows,
        "target_slot": target_slot,
        "target_row": target_row,
        "target_present": target_row >= 0,
        "event_frame": int(frame["event_frame"]),
        "frame": int(frame["frame"]),
    }
