"""Compile legal human corrections into assignment-learning transactions.

The compiler is deliberately independent of the SAM3 executor.  Its output is
the only supervision that the N28 live challengers are allowed to consume.
In particular, a reassign is represented by both sides of the decision:
the new identity receives a positive relation and the old identity receives
an explicit rejected relation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Optional


LEGAL_PROVENANCE = frozenset(
    {
        "HUMAN_REASSIGN",
        "HUMAN_ID_SWAP",
        "HUMAN_ADD",
        "HUMAN_DELETE",
        "PROTECTED_ANCHOR",
        # N29 spatial-supervision provenance.  Identity constraints may carry
        # one of these labels when a confirmed mask and ID correction arrive
        # in the same user transaction; the decoder teacher performs the
        # stricter mask-shape/oracle validation.
        "HUMAN_CONFIRMED_MASK",
        "POINT_REFINED_CONFIRMED_MASK",
        "BOX_PROMPTED_CONFIRMED_MASK",
        "BOX_DERIVED_PSEUDO_MASK",
        "GT_MASK_ORACLE",
    }
)


class CorrectionCompilationError(ValueError):
    """Raised when an event cannot be represented as legal supervision."""


@dataclass(frozen=True)
class RelationConstraint:
    """One target/candidate label emitted by a human event.

    ``candidate_index=None`` is the explicit per-target NONE choice.  It is
    not used for an unobserved candidate: candidate missingness is represented
    outside this class and never silently converted to a NONE label.
    """

    identity_id: Any
    candidate_index: Optional[int]
    label: int
    role: str
    provenance: str
    other_identity_id: Any = None

    def __post_init__(self) -> None:
        if self.label not in (0, 1):
            raise CorrectionCompilationError("relation labels must be 0 or 1")
        if self.role not in {"positive", "rejected", "none", "protected"}:
            raise CorrectionCompilationError(f"unknown constraint role: {self.role}")
        if self.role in {"positive", "none", "protected"} and self.label != 1:
            raise CorrectionCompilationError(f"{self.role} must have label 1")
        if self.role == "rejected" and self.label != 0:
            raise CorrectionCompilationError("rejected relations must have label 0")
        if self.candidate_index is not None and int(self.candidate_index) < 0:
            raise CorrectionCompilationError("candidate indices must be non-negative")
        if self.provenance not in LEGAL_PROVENANCE:
            raise CorrectionCompilationError(
                f"illegal supervision provenance: {self.provenance}"
            )


@dataclass(frozen=True)
class CorrectionTransaction:
    """Atomic set of relation constraints emitted by one user action."""

    transaction_id: str
    action: str
    frame: int
    constraints: tuple[RelationConstraint, ...]
    affected_identities: tuple[Any, ...]
    metadata: tuple[tuple[str, str], ...] = ()

    def for_identity(self, identity_id: Any) -> tuple[RelationConstraint, ...]:
        return tuple(c for c in self.constraints if c.identity_id == identity_id)

    @property
    def positive_constraints(self) -> tuple[RelationConstraint, ...]:
        return tuple(c for c in self.constraints if c.role in {"positive", "none", "protected"})

    @property
    def rejected_constraints(self) -> tuple[RelationConstraint, ...]:
        return tuple(c for c in self.constraints if c.role == "rejected")

    def validate(self) -> "CorrectionTransaction":
        """Check that one transaction does not issue contradictory labels."""
        seen: dict[tuple[Any, Optional[int]], RelationConstraint] = {}
        for constraint in self.constraints:
            key = (constraint.identity_id, constraint.candidate_index)
            previous = seen.get(key)
            if previous is not None and previous.label != constraint.label:
                raise CorrectionCompilationError(
                    f"contradictory labels for {key}: {previous.role}/{constraint.role}"
                )
            seen[key] = constraint
        return self


def _transaction(
    action: str,
    frame: int,
    constraints: Iterable[RelationConstraint],
    affected: Iterable[Any],
    metadata: Optional[dict[str, Any]] = None,
) -> CorrectionTransaction:
    constraints_tuple = tuple(constraints)
    affected_tuple = tuple(dict.fromkeys(affected))
    if not affected_tuple:
        raise CorrectionCompilationError("a correction must affect an identity")
    fields = tuple(
        sorted((str(key), str(value)) for key, value in (metadata or {}).items())
    )
    identity_text = ",".join(map(str, affected_tuple))
    transaction_id = f"{action}:{int(frame)}:{identity_text}:{len(constraints_tuple)}"
    return CorrectionTransaction(
        transaction_id=transaction_id,
        action=action,
        frame=int(frame),
        constraints=constraints_tuple,
        affected_identities=affected_tuple,
        metadata=fields,
    ).validate()


def compile_reassign(
    *,
    new_identity_id: Any,
    old_identity_id: Any,
    candidate_index: int,
    frame: int,
    provenance: str = "HUMAN_REASSIGN",
) -> CorrectionTransaction:
    """Compile ``old_id -> new_id`` as an atomic bilateral update."""
    if new_identity_id == old_identity_id:
        raise CorrectionCompilationError("reassign needs distinct old and new identities")
    if provenance != "HUMAN_REASSIGN":
        raise CorrectionCompilationError("reassign must use HUMAN_REASSIGN provenance")
    candidate_index = int(candidate_index)
    return _transaction(
        "REASSIGN",
        frame,
        (
            RelationConstraint(new_identity_id, candidate_index, 1, "positive", provenance, old_identity_id),
            RelationConstraint(old_identity_id, candidate_index, 0, "rejected", provenance, new_identity_id),
        ),
        (new_identity_id, old_identity_id),
        {"candidate_index": candidate_index},
    )


def compile_identity_correction(
    *,
    identity_id: Any,
    positive_candidate: int,
    rejected_candidate: Optional[int] = None,
    frame: int,
    provenance: str = "HUMAN_REASSIGN",
) -> CorrectionTransaction:
    """Compile a cached single-identity correction.

    The full runtime ``REASSIGN`` event is bilateral and is compiled by
    :func:`compile_reassign`.  A cached parent episode does not carry the
    global owner table needed to name the old identity, so its legal
    projection is one target identity with a positive corrected candidate and,
    when the displayed candidate was wrong, one explicitly rejected
    candidate.  This function never creates a negative for an unselected
    candidate and never represents candidate missingness as human absence.
    """
    if provenance not in {"HUMAN_REASSIGN", "HUMAN_ADD"}:
        raise CorrectionCompilationError(
            "single-identity correction must use HUMAN_REASSIGN or HUMAN_ADD"
        )
    positive_candidate = int(positive_candidate)
    if positive_candidate < 0:
        raise CorrectionCompilationError("positive candidate must be non-negative")
    constraints: list[RelationConstraint] = [
        RelationConstraint(
            identity_id,
            positive_candidate,
            1,
            "positive",
            provenance,
        )
    ]
    if rejected_candidate is not None:
        rejected_candidate = int(rejected_candidate)
        if rejected_candidate < 0:
            raise CorrectionCompilationError("rejected candidate must be non-negative")
        if rejected_candidate == positive_candidate:
            raise CorrectionCompilationError(
                "a corrected candidate cannot also be the rejected candidate"
            )
        constraints.append(
            RelationConstraint(
                identity_id,
                rejected_candidate,
                0,
                "rejected",
                provenance,
            )
        )
    return _transaction(
        "IDENTITY_CORRECTION",
        frame,
        constraints,
        (identity_id,),
        {
            "positive_candidate": positive_candidate,
            "rejected_candidate": (
                "none" if rejected_candidate is None else rejected_candidate
            ),
        },
    )


def compile_id_swap(
    *,
    identity_a: Any,
    identity_b: Any,
    candidate_a: int,
    candidate_b: int,
    frame: int,
    provenance: str = "HUMAN_ID_SWAP",
) -> CorrectionTransaction:
    """Compile a two-person switch into two positives and two cross-negatives."""
    if identity_a == identity_b:
        raise CorrectionCompilationError("an ID swap needs distinct identities")
    if int(candidate_a) == int(candidate_b):
        raise CorrectionCompilationError("an ID swap needs distinct candidates")
    if provenance != "HUMAN_ID_SWAP":
        raise CorrectionCompilationError("ID swap must use HUMAN_ID_SWAP provenance")
    candidate_a, candidate_b = int(candidate_a), int(candidate_b)
    return _transaction(
        "ID_SWAP",
        frame,
        (
            RelationConstraint(identity_a, candidate_a, 1, "positive", provenance, identity_b),
            RelationConstraint(identity_a, candidate_b, 0, "rejected", provenance, identity_b),
            RelationConstraint(identity_b, candidate_b, 1, "positive", provenance, identity_a),
            RelationConstraint(identity_b, candidate_a, 0, "rejected", provenance, identity_a),
        ),
        (identity_a, identity_b),
        {"candidate_a": candidate_a, "candidate_b": candidate_b},
    )


def compile_add(*, identity_id: Any, candidate_index: int, frame: int) -> CorrectionTransaction:
    """Compile a human-provided new identity observation."""
    candidate_index = int(candidate_index)
    return _transaction(
        "ADD",
        frame,
        (RelationConstraint(identity_id, candidate_index, 1, "positive", "HUMAN_ADD"),),
        (identity_id,),
        {"candidate_index": candidate_index},
    )


def compile_delete(*, identity_id: Any, frame: int) -> CorrectionTransaction:
    """Compile an explicit human absence/delete as the target's NONE label."""
    return _transaction(
        "DELETE",
        frame,
        (RelationConstraint(identity_id, None, 1, "none", "HUMAN_DELETE"),),
        (identity_id,),
    )


def compile_protected_anchor(
    *, identity_id: Any, candidate_index: int, frame: int
) -> CorrectionTransaction:
    """Create an optional replay/protection label from an earlier correction."""
    candidate_index = int(candidate_index)
    return _transaction(
        "PROTECTED",
        frame,
        (RelationConstraint(identity_id, candidate_index, 1, "protected", "PROTECTED_ANCHOR"),),
        (identity_id,),
        {"candidate_index": candidate_index},
    )
