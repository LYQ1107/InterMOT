"""Chronological online update engine for the N28 challengers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional

import numpy as np
import torch
from torch.nn import functional as F

from sam3_intermot.adaptation.correction_compiler import CorrectionTransaction
from sam3_intermot.adaptation.correction_rls import CorrectionRLS
from sam3_intermot.adaptation.update_validator import UpdateValidation, UpdateValidator
from sam3_intermot.association.relational_challenger import LciaRelationalChallenger


@dataclass(frozen=True)
class SupportExample:
    """One cached target row used for an online support update."""

    identity_id: Any
    anchor_scores: np.ndarray
    relation_features: np.ndarray
    candidate_mask: np.ndarray
    none_score: float = 0.0


@dataclass(frozen=True)
class UpdateConfig:
    learning_rate: float = 0.05
    max_steps: int = 40
    support_margin: float = 0.01
    protected_weight: float = 0.5
    trust_weight: float = 1.0e-4
    gradient_clip: float = 10.0


@dataclass(frozen=True)
class UpdateResult:
    accepted: bool
    steps: int
    loss: float
    validation: UpdateValidation
    rolled_back: bool


class LiveUpdateEngine:
    """Update only the affected identity factors, with atomic rollback."""

    def __init__(
        self,
        challenger: LciaRelationalChallenger,
        *,
        config: Optional[UpdateConfig] = None,
        validator: Optional[UpdateValidator] = None,
    ) -> None:
        self.challenger = challenger
        self.config = config or UpdateConfig()
        self.validator = validator or UpdateValidator()

    @staticmethod
    def _constraints_by_identity(transaction: CorrectionTransaction) -> dict[Any, list]:
        out: dict[Any, list] = {identity_id: [] for identity_id in transaction.affected_identities}
        for constraint in transaction.constraints:
            out.setdefault(constraint.identity_id, []).append(constraint)
        return out

    def _scores(self, example: SupportExample, identity_id: Any) -> torch.Tensor:
        relation = torch.as_tensor(example.relation_features, dtype=torch.float32).unsqueeze(0)
        delta = self.challenger.delta_tensor(relation, [identity_id])[0]
        anchor = torch.as_tensor(example.anchor_scores, dtype=torch.float32)
        scores = anchor + delta
        mask = torch.as_tensor(example.candidate_mask, dtype=torch.bool)
        scores = scores.masked_fill(~mask, -1.0e4)
        return torch.cat([scores, anchor.new_tensor([float(example.none_score)])])

    def _loss(
        self,
        transaction: CorrectionTransaction,
        examples: Mapping[Any, SupportExample],
        snapshots: dict[Any, tuple[torch.Tensor, ...]],
    ) -> tuple[torch.Tensor, list[tuple[Any, int, Optional[int]]]]:
        constraints = self._constraints_by_identity(transaction)
        losses: list[torch.Tensor] = []
        support_pairs: list[tuple[Any, int, Optional[int]]] = []
        for identity_id, identity_constraints in constraints.items():
            if identity_id not in examples:
                raise KeyError(f"missing cached support example for {identity_id}")
            example = examples[identity_id]
            scores = self._scores(example, identity_id)
            positive = [c for c in identity_constraints if c.role in {"positive", "none"}]
            rejected = [c for c in identity_constraints if c.role == "rejected"]
            if positive:
                target = len(example.anchor_scores) if positive[0].candidate_index is None else int(positive[0].candidate_index)
            elif rejected:
                # A reassign must also teach the old identity that the
                # corrected candidate is not hers.  In the absence of a
                # second positive, explicit NONE is the legal local target;
                # this is not a detector-missing label.
                target = len(example.anchor_scores)
            else:
                continue
            if target < 0 or target >= len(scores):
                raise ValueError(f"support target out of range for {identity_id}")
            losses.append(F.cross_entropy(scores.unsqueeze(0), torch.tensor([target])))
            rejected_index = None
            if rejected:
                rejected_index = int(rejected[0].candidate_index)
                losses.append(F.relu(scores[rejected_index] - scores[target] + self.config.support_margin))
            support_pairs.append((identity_id, target, rejected_index))

            # Keep a previously accepted relation close to its transaction
            # snapshot.  This is a local trust term, not an anchor replacement.
            state = self.challenger.model.ensure_identity(identity_id)
            for parameter, before in zip(state.factors, snapshots[identity_id]):
                losses.append(self.config.trust_weight * (parameter - before).pow(2).mean())
        if not losses:
            raise ValueError("transaction has no positive support constraint")
        return torch.stack(losses).sum(), support_pairs

    def apply(
        self,
        transaction: CorrectionTransaction,
        examples: Mapping[Any, SupportExample],
        *,
        protected_examples: Optional[Mapping[Any, SupportExample]] = None,
        force_reject: bool = False,
    ) -> UpdateResult:
        identities = list(transaction.affected_identities)
        snapshot = self.challenger.snapshot(identities)
        snapshots = {identity_id: snapshot[identity_id] for identity_id in identities}
        parameters = self.challenger.live_parameters(identities)
        optimizer = torch.optim.Adam(parameters, lr=self.config.learning_rate)
        before_scores = {
            identity_id: self._scores(examples[identity_id], identity_id).detach().cpu().numpy()
            for identity_id in identities
        }
        last_loss = float("nan")
        last_validation: Optional[UpdateValidation] = None
        try:
            for step in range(1, self.config.max_steps + 1):
                optimizer.zero_grad(set_to_none=True)
                loss, support_pairs = self._loss(transaction, examples, snapshots)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(parameters, self.config.gradient_clip)
                optimizer.step()
                last_loss = float(loss.detach().cpu())
                after_scores = {
                    identity_id: self._scores(examples[identity_id], identity_id).detach().cpu().numpy()
                    for identity_id in identities
                }
                protected_pairs: list[tuple[Any, int]] = []
                if protected_examples:
                    for identity_id, protected in protected_examples.items():
                        # The protected candidate index is the pre-update
                        # anchor winner; only unchanged identities are used in
                        # this first smoke implementation.
                        protected_scores = np.asarray(protected.anchor_scores).reshape(-1)
                        protected_pairs.append((identity_id, int(np.argmax(protected_scores))))
                        if identity_id not in before_scores:
                            before_scores[identity_id] = self._scores(protected, identity_id).detach().cpu().numpy()
                        after_scores[identity_id] = self._scores(protected, identity_id).detach().cpu().numpy()
                last_validation = self.validator.validate(
                    before_scores=before_scores,
                    after_scores=after_scores,
                    support_pairs=support_pairs,
                    protected_pairs=protected_pairs,
                    update_norm=self.challenger.state_norm(identities),
                )
                if last_validation.accepted and not force_reject:
                    return UpdateResult(True, step, last_loss, last_validation, False)
            if last_validation is None:
                last_validation = UpdateValidation(False, True, float("nan"), 0.0, self.challenger.state_norm(identities), ("no_checkpoint",))
        except Exception:
            self.challenger.restore(snapshot)
            raise
        self.challenger.restore(snapshot)
        return UpdateResult(False, self.config.max_steps, last_loss, last_validation, True)


class AtomicChallengerUpdate:
    """Coordinate snapshots of a LoRA and C-RLS challenger."""

    def __init__(self, lcia: Optional[LciaRelationalChallenger], crls: Optional[CorrectionRLS], identities: list[Any]) -> None:
        self.lcia = lcia
        self.crls = crls
        self.identities = identities
        self._lcia_snapshot = lcia.snapshot(identities) if lcia is not None else None
        self._crls_snapshot = crls.snapshot(identities) if crls is not None else None
        self._committed = False

    def commit(self) -> None:
        self._committed = True

    def rollback(self) -> None:
        if self.lcia is not None and self._lcia_snapshot is not None:
            self.lcia.restore(self._lcia_snapshot)
        if self.crls is not None and self._crls_snapshot is not None:
            self.crls.restore(self._crls_snapshot)

    def __enter__(self) -> "AtomicChallengerUpdate":
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        if exc_type is not None or not self._committed:
            self.rollback()
        return False
