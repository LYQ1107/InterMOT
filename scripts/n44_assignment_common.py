#!/usr/bin/env python3
"""N44 isolated assignment-aware sidecar.

The model scores a candidate cell from causal cell/state features only.  Its
pairwise difference is anti-symmetric by construction.  At runtime the exact
baseline matrix is retained unless a bounded, explicitly gated proposal is
accepted; the full matrix is then passed through the same Hungarian-with-NONE
interface.
"""

from __future__ import annotations

import copy
import hashlib
from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from torch import nn

from scripts.n43_full_matrix_common import HARD_NEGATIVE, NONE_SCORE, cell_features


PROTOCOL = "N44_ASSIGNMENT_AWARE_SIDECAR_V1"
FEATURE_DIM = 18
MAX_BOOST = 0.25


class AssignmentAwareHead(nn.Module):
    """Scalar utility plus a learned uncertainty scale; no identity input."""

    def __init__(self, input_dim: int = FEATURE_DIM, hidden: int = 64) -> None:
        super().__init__()
        self.body = nn.Sequential(nn.Linear(input_dim, hidden), nn.ReLU(), nn.Linear(hidden, 32), nn.ReLU())
        self.output = nn.Linear(32, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.output(self.body(x))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_checkpoint(path: Path, device: str = "cpu") -> tuple[AssignmentAwareHead, dict[str, Any]]:
    payload = torch.load(path, map_location=device, weights_only=False)
    if not isinstance(payload, dict) or payload.get("protocol") != PROTOCOL:
        raise ValueError("invalid N44 checkpoint protocol")
    model = AssignmentAwareHead(input_dim=int(payload.get("input_dim", FEATURE_DIM)), hidden=int(payload.get("hidden", 64)))
    model.load_state_dict(payload["state_dict"])
    model.to(device).eval()
    return model, payload


def finite_matrix(audit: dict[str, Any]) -> np.ndarray:
    # The sidecar receives the branch's current fused score as its exact
    # baseline.  This preserves the M1--M4 memory/appearance condition; the
    # sidecar itself contributes only the bounded accepted proposal boost.
    raw = audit.get("fused_scores", audit.get("base_scores_before_appearance", audit.get("base_scores")))
    value = np.asarray(raw, dtype=np.float32)
    if value.ndim != 2 or not np.all(np.isfinite(value)):
        raise ValueError("N44 baseline matrix is invalid")
    return value


def hungarian_with_none(scores: np.ndarray) -> np.ndarray:
    matrix = np.asarray(scores, dtype=np.float32)
    if matrix.ndim != 2 or not np.all(np.isfinite(matrix)):
        raise ValueError("invalid N44 score matrix")
    expanded = np.concatenate([matrix, np.full((matrix.shape[0], matrix.shape[0]), NONE_SCORE, dtype=np.float32)], axis=1)
    rows, columns = linear_sum_assignment(-expanded)
    result = np.full(matrix.shape[0], -1, dtype=int)
    result[rows] = columns
    return result


def predict(model: AssignmentAwareHead, features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    values = torch.as_tensor(np.asarray(features, dtype=np.float32))
    with torch.no_grad():
        raw = model(values).cpu().numpy()
    if not np.all(np.isfinite(raw)):
        raise ValueError("N44 model output is nonfinite")
    # The second output is a log variance.  The pairwise uncertainty is the
    # square root of the sum of the two cell variances.
    score = raw[:, 0].astype(np.float32)
    variance = (torch.nn.functional.softplus(torch.as_tensor(raw[:, 1])) + 0.02).numpy().astype(np.float32)
    return score, variance


def apply_sidecar(
    audit: dict[str, Any],
    model: AssignmentAwareHead,
    frame_offset: int,
    previous_audit: dict[str, Any] | None,
    gate: dict[str, float],
) -> dict[str, Any]:
    """Apply only accepted bounded proposals and recompute global assignment."""
    output = copy.deepcopy(audit)
    base = finite_matrix(audit)
    pids = [int(value) for value in audit.get("public_id_order", [])]
    features = np.asarray(
        [cell_features(audit, i, j, frame_offset, previous_audit) for i in range(base.shape[0]) for j in range(base.shape[1])],
        dtype=np.float32,
    )
    cell_score, cell_variance = predict(model, features)
    cell_score = cell_score.reshape(base.shape)
    cell_variance = cell_variance.reshape(base.shape)
    baseline_assignment = hungarian_with_none(base)
    owner_by_column = {int(column): int(row) for row, column in enumerate(baseline_assignment) if 0 <= int(column) < len(pids)}
    proposals: list[tuple[float, int, int, float]] = []
    near_tie = float(gate["near_tie_margin"])
    min_advantage = float(gate["min_predicted_advantage"])
    max_std = float(gate["max_pair_uncertainty"])
    for column in range(base.shape[1]):
        owner = owner_by_column.get(column)
        # NONE/abstain remains a hard conservative boundary.  The sidecar
        # cannot turn an unmatched public ID into a new assignment.
        if owner is None or base[owner, column] <= HARD_NEGATIVE:
            continue
        for candidate in range(base.shape[0]):
            if candidate == owner or base[candidate, column] <= HARD_NEGATIVE:
                continue
            if base[owner, column] - base[candidate, column] > near_tie:
                continue
            other_assignment = baseline_assignment[candidate]
            if 0 <= int(other_assignment) < len(pids) and int(other_assignment) != column:
                continue
            advantage = float(cell_score[candidate, column] - cell_score[owner, column])
            uncertainty = float(np.sqrt(cell_variance[candidate, column] + cell_variance[owner, column]))
            if advantage >= min_advantage and uncertainty <= max_std:
                proposals.append((advantage, candidate, column, uncertainty))
    # A proposal cannot reserve the same candidate or public ID twice.  This
    # selection is only a gate; assignment is still decided globally below.
    proposals.sort(key=lambda item: (-item[0], item[1], item[2]))
    selected: list[tuple[float, int, int, float]] = []
    used_candidates: set[int] = set()
    used_columns: set[int] = set()
    for proposal in proposals:
        _, candidate, column, _ = proposal
        if candidate in used_candidates or column in used_columns:
            continue
        used_candidates.add(candidate)
        used_columns.add(column)
        selected.append(proposal)
    adjusted = base.copy()
    for _, candidate, column, _ in selected:
        adjusted[candidate, column] = base[candidate, column] + MAX_BOOST
    hard = base <= HARD_NEGATIVE
    adjusted[hard] = base[hard]
    assignment = hungarian_with_none(adjusted)
    mapped = [pids[int(column)] if 0 <= int(column) < len(pids) else None for column in assignment]
    output["fused_scores_before_n44"] = base.astype(float).tolist()
    output["fused_scores"] = adjusted.astype(float).tolist()
    output["scores"] = adjusted.astype(float).tolist()
    output["assignment_before_n44"] = baseline_assignment.astype(int).tolist()
    output["assignment_after_n44"] = assignment.astype(int).tolist()
    output["assignment_after_scope"] = assignment.astype(int).tolist()
    output["assignment"] = assignment.astype(int).tolist()
    output["candidate_public_ids"] = mapped
    output["n44_sidecar"] = {
        "enabled": True,
        "protocol": PROTOCOL,
        "application": "accepted bounded near-tie proposals only; all other cells exact baseline",
        "cell_count": int(base.size),
        "candidate_count": int(base.shape[0]),
        "public_id_count": int(base.shape[1]),
        "proposals_considered": int(len(proposals)),
        "proposals_selected": int(len(selected)),
        "changed_cell_count": int(np.sum(np.abs(adjusted - base) > 1.0e-12)),
        "changed_assignment_count": int(np.sum(baseline_assignment != assignment)),
        "hard_negative_preserved": bool(np.all(adjusted[hard] == base[hard])),
        "none_score": NONE_SCORE,
        "max_boost": MAX_BOOST,
        "gate": {key: float(value) for key, value in gate.items()},
        "runtime_future_gt_used": False,
    }
    output["runtime_future_gt_used"] = False
    return output
