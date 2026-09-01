"""Assignment-coupled challenger interfaces for N28.

The anchor and all challenger deltas are assembled before matching.  The
result is therefore one global target-by-candidate assignment, with one
private NONE/dummy column per target, rather than independent row argmaxes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
import torch

from sam3_intermot.adaptation.correction_rls import CRLSConfig, CorrectionRLS
from sam3_intermot.adaptation.correction_compiler import CorrectionTransaction
from sam3_intermot.adaptation.live_identity_lora import LiveIdentityLoRA
from sam3_intermot.association.online_associator import hungarian_max


NEGATIVE_SCORE = -1.0e9


@dataclass(frozen=True)
class AssignmentResult:
    """A complete assignment in candidate coordinates.

    ``assignment[row]`` is a candidate column or ``-1`` for that row's NONE
    option.  ``matrix`` retains the candidate and dummy columns used by
    Hungarian for auditability.
    """

    matrix: np.ndarray
    assignment: np.ndarray
    selected_scores: np.ndarray


def _candidate_mask(mask: Optional[np.ndarray], rows: int, candidates: int) -> np.ndarray:
    if mask is None:
        return np.ones((rows, candidates), dtype=bool)
    values = np.asarray(mask, dtype=bool)
    if values.ndim == 1:
        if len(values) != candidates:
            raise ValueError("one-dimensional candidate_mask has the wrong length")
        return np.broadcast_to(values[None], (rows, candidates)).copy()
    if values.shape != (rows, candidates):
        raise ValueError("candidate_mask must be [candidates] or [targets, candidates]")
    return values.copy()


def build_assignment_matrix(
    anchor_scores: np.ndarray,
    delta_scores: Optional[np.ndarray] = None,
    *,
    none_scores: Optional[np.ndarray] = None,
    candidate_mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Build ``target x (candidate + target-specific NONE)`` scores."""
    anchor = np.asarray(anchor_scores, dtype=np.float64)
    if anchor.ndim != 2:
        raise ValueError("anchor_scores must be a [target, candidate] matrix")
    targets, candidates = anchor.shape
    delta = np.zeros_like(anchor) if delta_scores is None else np.asarray(delta_scores, dtype=np.float64)
    if delta.shape != anchor.shape:
        raise ValueError("delta_scores must have the same shape as anchor_scores")
    mask = _candidate_mask(candidate_mask, targets, candidates)
    matrix = np.full((targets, candidates + targets), NEGATIVE_SCORE, dtype=np.float64)
    matrix[:, :candidates] = np.where(mask, anchor + delta, NEGATIVE_SCORE)
    if none_scores is None:
        none = np.zeros(targets, dtype=np.float64)
    else:
        none = np.asarray(none_scores, dtype=np.float64).reshape(-1)
        if len(none) != targets:
            raise ValueError("none_scores must have one value per target")
    if targets:
        matrix[np.arange(targets), candidates + np.arange(targets)] = none
    return matrix


def assign_matrix(matrix: np.ndarray, candidates: int) -> AssignmentResult:
    """Run the existing Hungarian helper on a complete coupled matrix."""
    values = np.asarray(matrix, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] < candidates:
        raise ValueError("invalid assignment matrix")
    assignment_columns = hungarian_max(values)
    assignment = np.full(values.shape[0], -1, dtype=int)
    selected = np.full(values.shape[0], NEGATIVE_SCORE, dtype=np.float64)
    for row, column in enumerate(assignment_columns):
        if column < 0:
            continue
        selected[row] = values[row, column]
        if column < candidates:
            assignment[row] = int(column)
    return AssignmentResult(values.copy(), assignment, selected)


def coupled_assignment(
    anchor_scores: np.ndarray,
    delta_scores: Optional[np.ndarray] = None,
    *,
    none_scores: Optional[np.ndarray] = None,
    candidate_mask: Optional[np.ndarray] = None,
) -> AssignmentResult:
    anchor = np.asarray(anchor_scores)
    matrix = build_assignment_matrix(
        anchor,
        delta_scores,
        none_scores=none_scores,
        candidate_mask=candidate_mask,
    )
    return assign_matrix(matrix, anchor.shape[1])


def build_cached_relation_features(
    *,
    b10_score: np.ndarray,
    root_similarity: np.ndarray,
    positive_similarity: np.ndarray,
    negative_similarity: np.ndarray,
    hard_similarity: np.ndarray,
    detector_score: np.ndarray,
    candidate_mask: np.ndarray,
) -> np.ndarray:
    """Pack the legal frozen N27 relation fields for N28-A/B.

    These are candidate-level relation features only.  No target label,
    sequence ID, candidate rank, or future field is introduced here.
    """
    arrays = [
        np.asarray(value, dtype=np.float32)
        for value in (
            b10_score,
            root_similarity,
            positive_similarity,
            negative_similarity,
            hard_similarity,
            detector_score,
        )
    ]
    mask = np.asarray(candidate_mask, dtype=np.float32)
    if any(value.shape != arrays[0].shape for value in arrays[1:]) or mask.shape != arrays[0].shape:
        raise ValueError("cached relation fields must share [target, candidate] shape")
    valid = mask
    return np.stack((*arrays, valid), axis=-1)


class CrlsRelationalChallenger:
    """Zero-reference assignment challenger backed by correction-only RLS."""

    def __init__(self, feature_dim: int, *, residual_scale: float = 2.0, ridge: float = 1.0) -> None:
        self.rls = CorrectionRLS(
            config=CRLSConfig(
                feature_dim=feature_dim,
                residual_scale=residual_scale,
                ridge=ridge,
            )
        )

    def delta(self, relation: np.ndarray, identity_id: Any) -> np.ndarray:
        return self.rls.delta(identity_id, relation)

    def delta_batch(self, relation: np.ndarray, identity_ids: list[Any]) -> np.ndarray:
        return self.rls.delta_batch(relation, identity_ids)

    def score_matrix(
        self,
        anchor_scores: np.ndarray,
        relation: np.ndarray,
        identity_ids: list[Any],
        *,
        none_scores: Optional[np.ndarray] = None,
        candidate_mask: Optional[np.ndarray] = None,
    ) -> tuple[np.ndarray, AssignmentResult]:
        delta = self.delta_batch(relation, identity_ids)
        result = coupled_assignment(
            anchor_scores,
            delta,
            none_scores=none_scores,
            candidate_mask=candidate_mask,
        )
        return delta, result

    def update_from_constraints(self, relation: np.ndarray, labels: np.ndarray, identity_id: Any) -> None:
        self.rls.update(identity_id, relation, labels)

    def update_transaction(
        self,
        transaction: CorrectionTransaction,
        relation_by_identity: dict[Any, np.ndarray],
    ) -> None:
        """Consume only the positive/rejected relations in a legal event.

        Unselected candidates are not silently treated as negatives.  This is
        the C-RLS analogue of the correction compiler's provenance contract.
        """
        for identity_id in transaction.affected_identities:
            relation = np.asarray(relation_by_identity[identity_id])
            rows = []
            labels = []
            for constraint in transaction.for_identity(identity_id):
                if constraint.candidate_index is None:
                    continue
                candidate = int(constraint.candidate_index)
                if candidate >= relation.shape[0]:
                    raise ValueError(f"candidate index {candidate} exceeds relation rows")
                rows.append(relation[candidate])
                labels.append(1.0 if constraint.label else -1.0)
            if rows:
                self.rls.update(identity_id, np.stack(rows), np.asarray(labels))

    def snapshot(self, identity_ids: list[Any]) -> dict[Any, tuple[np.ndarray, np.ndarray]]:
        return self.rls.snapshot(identity_ids)

    def restore(self, snapshot: dict[Any, tuple[np.ndarray, np.ndarray]]) -> None:
        self.rls.restore(snapshot)

    def reset(self, identity_ids: Optional[list[Any]] = None) -> None:
        self.rls.reset(identity_ids)


class LciaRelationalChallenger:
    """Assignment wrapper around the identity-scoped B-only LoRA module."""

    def __init__(self, model: Optional[LiveIdentityLoRA] = None) -> None:
        self.model = model or LiveIdentityLoRA()

    def delta_tensor(self, relation: torch.Tensor, identity_ids: list[Any]) -> torch.Tensor:
        return self.model.delta_batch(relation, identity_ids)

    def delta(self, relation: np.ndarray, identity_id: Any) -> np.ndarray:
        return self.model.delta_numpy(relation, identity_id)

    def delta_batch(self, relation: np.ndarray, identity_ids: list[Any]) -> np.ndarray:
        values = torch.as_tensor(relation, dtype=torch.float32)
        with torch.no_grad():
            return self.model.delta_batch(values, identity_ids).cpu().numpy()

    def score_matrix(
        self,
        anchor_scores: np.ndarray,
        relation: np.ndarray,
        identity_ids: list[Any],
        *,
        none_scores: Optional[np.ndarray] = None,
        candidate_mask: Optional[np.ndarray] = None,
    ) -> tuple[np.ndarray, AssignmentResult]:
        delta = self.delta_batch(relation, identity_ids)
        result = coupled_assignment(
            anchor_scores,
            delta,
            none_scores=none_scores,
            candidate_mask=candidate_mask,
        )
        return delta, result

    def snapshot(self, identity_ids: list[Any]) -> dict[Any, tuple[torch.Tensor, ...]]:
        return self.model.snapshot(identity_ids)

    def restore(self, snapshot: dict[Any, tuple[torch.Tensor, ...]]) -> None:
        self.model.restore(snapshot)

    def reset(self, identity_ids: Optional[list[Any]] = None) -> None:
        self.model.reset(identity_ids)

    def live_parameters(self, identity_ids: list[Any]) -> list[torch.nn.Parameter]:
        return self.model.live_parameters(identity_ids)

    def state_norm(self, identity_ids: list[Any]) -> float:
        return self.model.live_state_norm(identity_ids)
