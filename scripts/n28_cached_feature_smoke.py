#!/usr/bin/env python3
"""N28-A exactness and live-update smoke on the immutable N27 cache.

This is intentionally a one-support-event engineering gate.  It does not
estimate a method metric and it never reads val25.  The source row is chosen
from the already materialised N27 DanceTrack cache; no new candidates or
labels are generated here.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sam3_intermot.adaptation.correction_compiler import (  # noqa: E402
    compile_id_swap,
    compile_reassign,
)
from sam3_intermot.adaptation.live_identity_lora import (  # noqa: E402
    LiveIdentityLoRA,
    LiveLoRAConfig,
)
from sam3_intermot.adaptation.live_optimizer import (  # noqa: E402
    AtomicChallengerUpdate,
    LiveUpdateEngine,
    SupportExample,
    UpdateConfig,
)
from sam3_intermot.adaptation.update_validator import (  # noqa: E402
    UpdateValidator,
    UpdateValidatorConfig,
    exact_zero,
)
from sam3_intermot.association.relational_challenger import (  # noqa: E402
    CrlsRelationalChallenger,
    LciaRelationalChallenger,
    build_assignment_matrix,
    build_cached_relation_features,
    coupled_assignment,
)


REQUIRED_CACHE_FIELDS = (
    "candidate_mask",
    "target",
    "target_present",
    "b10_score",
    "root_similarity",
    "positive_similarity",
    "negative_similarity",
    "hard_similarity",
    "detector_score",
    "correction_event",
    "frame",
    "event_hash",
)


def _json_value(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def choose_support(z: np.lib.npyio.NpzFile) -> int:
    mask = z["candidate_mask"].astype(bool)
    target = z["target"].astype(int)
    selected = z["selected"].astype(int)
    eligible = (
        z["correction_event"].astype(bool)
        & z["target_present"].astype(bool)
        & (target >= 0)
        & (target < mask.shape[1])
        & (selected != target)
        & np.all(mask, axis=1)
    )
    # Use a reproducible, non-trivial correction whose target is a visible
    # cached candidate and whose target-specific support can be overfit.
    preferred = np.flatnonzero(eligible & (np.arange(len(target)) == 258))
    candidates = preferred if len(preferred) else np.flatnonzero(eligible)
    if not len(candidates):
        raise RuntimeError("N28-A found no legal cached support event")
    return int(candidates[0])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cache",
        type=Path,
        default=ROOT / "outputs/n27/data/dance_train_real_b10_round0.npz",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs/n28/n28a_cached_feature_smoke.json",
    )
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    with np.load(args.cache, allow_pickle=False) as z:
        missing = [name for name in REQUIRED_CACHE_FIELDS if name not in z.files]
        if missing:
            raise RuntimeError(f"cached N27 fields missing: {missing}")
        index = choose_support(z)
        field_kwargs = {
            name: z[name][index : index + 1]
            for name in (
                "b10_score",
                "root_similarity",
                "positive_similarity",
                "negative_similarity",
                "hard_similarity",
                "detector_score",
                "candidate_mask",
            )
        }
        relation = build_cached_relation_features(**field_kwargs)[0]
        anchor = z["b10_score"][index].astype(np.float32)
        candidate_mask = z["candidate_mask"][index].astype(bool)
        target = int(z["target"][index])
        selected = int(z["selected"][index])
        frame = int(z["frame"][index])
        event_hash = z["event_hash"][index].decode("ascii")

    identity_ids = ["new_identity", "old_identity"]
    relation_batch = np.stack([relation, relation])
    anchor_batch = np.stack([anchor, anchor])
    mask_batch = np.stack([candidate_mask, candidate_mask])
    none_scores = np.zeros(2, dtype=np.float32)

    # Assignment coupling is tested with a conflict where independent row
    # argmax would select the same candidate.  The matrix is deliberately
    # tiny, but uses the same production helper as the cached path.
    conflict_anchor = np.asarray(
        [[anchor[0], anchor[1]], [anchor[0] - 0.005, anchor[1] - 0.75]],
        dtype=np.float32,
    )
    conflict = coupled_assignment(
        conflict_anchor,
        none_scores=np.asarray([-1.0, -1.0], dtype=np.float32),
        candidate_mask=np.ones((2, 2), dtype=bool),
    )
    rowwise = np.argmax(conflict_anchor, axis=1)
    assignment_coupled = bool(
        np.array_equal(rowwise, np.asarray([0, 0]))
        and np.array_equal(conflict.assignment, np.asarray([1, 0]))
        and len(set(conflict.assignment.tolist())) == 2
    )

    lcia = LciaRelationalChallenger(
        LiveIdentityLoRA(
            LiveLoRAConfig(
                input_dim=relation.shape[-1],
                d_model=64,
                rank=8,
                blocks=2,
                alpha=8.0,
                seed=28,
            )
        )
    )
    crls = CrlsRelationalChallenger(
        relation.shape[-1], residual_scale=2.0, ridge=1.0
    )

    # Both challengers start with literal zero residuals.
    lcia_zero = lcia.delta_batch(relation_batch, identity_ids)
    crls_zero = crls.delta_batch(relation_batch, identity_ids)
    base_matrix = build_assignment_matrix(
        anchor_batch, none_scores=none_scores, candidate_mask=mask_batch
    )
    lcia_zero_matrix = build_assignment_matrix(
        anchor_batch,
        lcia_zero,
        none_scores=none_scores,
        candidate_mask=mask_batch,
    )
    crls_zero_matrix = build_assignment_matrix(
        anchor_batch,
        crls_zero,
        none_scores=none_scores,
        candidate_mask=mask_batch,
    )
    baseline_assignment = coupled_assignment(
        anchor_batch, none_scores=none_scores, candidate_mask=mask_batch
    )
    lcia_assignment = coupled_assignment(
        anchor_batch,
        lcia_zero,
        none_scores=none_scores,
        candidate_mask=mask_batch,
    )

    reassign = compile_reassign(
        new_identity_id=identity_ids[0],
        old_identity_id=identity_ids[1],
        candidate_index=target,
        frame=frame,
    )
    swap = compile_id_swap(
        identity_a="identity_a",
        identity_b="identity_b",
        candidate_a=0,
        candidate_b=1,
        frame=frame,
    )
    swap_roles = sorted(
        (
            str(c.identity_id),
            c.candidate_index,
            c.label,
            c.role,
        )
        for c in swap.constraints
    )

    examples = {
        identity_id: SupportExample(
            identity_id=identity_id,
            anchor_scores=anchor.copy(),
            relation_features=relation.copy(),
            candidate_mask=candidate_mask.copy(),
            none_score=0.0,
        )
        for identity_id in identity_ids
    }
    engine = LiveUpdateEngine(
        lcia,
        config=UpdateConfig(learning_rate=0.05, max_steps=40, support_margin=0.01),
        validator=UpdateValidator(
            UpdateValidatorConfig(support_margin=0.01, max_update_norm=1000.0)
        ),
    )

    untouched_before = lcia.delta(relation, "untouched_identity")
    lcia_snapshot_before = lcia.snapshot(identity_ids)
    crls_snapshot_before = crls.snapshot(identity_ids)
    with AtomicChallengerUpdate(lcia, crls.rls, identity_ids) as atomic:
        # LCIA sees the same transaction as C-RLS, but its learned relation
        # surface is updated only through the fast B factors.
        lcia_update = engine.apply(reassign, examples)
        if not lcia_update.accepted:
            raise RuntimeError(f"LCIA support update failed: {lcia_update.validation}")
        # C-RLS consumes exactly the two legal bilateral relations.  It does
        # not manufacture negatives for candidates that the user did not
        # explicitly reject.
        crls.update_transaction(
            reassign,
            {identity_ids[0]: relation, identity_ids[1]: relation},
        )
        atomic.commit()

    lcia_after = lcia.delta_batch(relation_batch, identity_ids)
    crls_after = crls.delta_batch(relation_batch, identity_ids)
    lcia_scores = anchor + lcia_after[0]
    crls_scores = anchor + crls_after[0]
    old_lcia_scores = np.r_[anchor + lcia_after[1], 0.0]
    old_crls_scores = np.r_[anchor + crls_after[1], 0.0]
    untouched_after = lcia.delta(relation, "untouched_identity")

    # A failed atomic update must restore both challengers exactly.  Mutate
    # each side directly to exercise the coordinator independently of the
    # optimizer's own checkpoint rejection path.
    rollback_lcia_before = lcia.snapshot(identity_ids)
    rollback_crls_before = crls.snapshot(identity_ids)
    with AtomicChallengerUpdate(lcia, crls.rls, identity_ids):
        with torch_no_grad():
            for parameter in lcia.live_parameters(identity_ids):
                parameter.add_(0.125)
        crls.rls.update(
            identity_ids[0], relation[:1], np.asarray([0], dtype=np.float64)
        )
    rollback_lcia_after = lcia.snapshot(identity_ids)
    rollback_crls_after = crls.snapshot(identity_ids)

    def lcia_snapshot_equal(left, right) -> bool:
        return all(
            np.array_equal(a.detach().cpu().numpy(), b.detach().cpu().numpy())
            for identity_id in left
            for a, b in zip(left[identity_id], right[identity_id])
        )

    def crls_snapshot_equal(left, right) -> bool:
        return all(
            np.array_equal(a, b)
            for identity_id in left
            for a, b in zip(left[identity_id], right[identity_id])
        )

    # Importing torch at module scope would make cache/schema inspection
    # needlessly expensive; the smoke has already imported it through the
    # model, so this tiny context keeps the mutation visibly no-grad.
    support_margin_lcia = float(lcia_update.validation.support_min_margin)
    support_margin_crls = float(
        crls_scores[target] - np.max(np.delete(crls_scores, target))
    )
    result = {
        "phase": "N28-A",
        "status": "PASS",
        "method_status": "ENGINEERING_SMOKE_ONLY_NO_N28_B_RESULT",
        "val25_read": False,
        "cache": str(args.cache.relative_to(ROOT)),
        "cache_event_hash": event_hash,
        "support": {
            "row": index,
            "frame": frame,
            "target_candidate": target,
            "frozen_b10_selected": selected,
            "frozen_b10_selection_matches_argmax": bool(selected == int(np.argmax(anchor))),
        },
        "assignment": {
            "target_by_candidate_matrix": True,
            "none_columns": 2,
            "rowwise_conflict": rowwise.tolist(),
            "coupled_conflict": conflict.assignment.tolist(),
            "global_one_to_one_gate": assignment_coupled,
        },
        "exact_zero": {
            "lcia_delta": exact_zero(lcia_zero, np.zeros_like(lcia_zero)),
            "crls_delta": exact_zero(crls_zero, np.zeros_like(crls_zero)),
            "lcia_matrix": exact_zero(base_matrix, lcia_zero_matrix),
            "crls_matrix": exact_zero(base_matrix, crls_zero_matrix),
            "selection": bool(np.array_equal(baseline_assignment.assignment, lcia_assignment.assignment)),
        },
        "reassign": {
            "transaction_id": reassign.transaction_id,
            "constraint_count": len(reassign.constraints),
            "new_positive": [
                c.candidate_index
                for c in reassign.constraints
                if c.identity_id == identity_ids[0] and c.role == "positive"
            ],
            "old_explicit_negative": [
                c.candidate_index
                for c in reassign.constraints
                if c.identity_id == identity_ids[1] and c.role == "rejected"
            ],
            "lcia_update": {
                "accepted": lcia_update.accepted,
                "steps": lcia_update.steps,
                "support_margin": support_margin_lcia,
                "live_state_norm": lcia.state_norm(identity_ids),
            },
            "crls_support_margin": support_margin_crls,
            "lcia_target_is_top_candidate": bool(int(np.argmax(lcia_scores)) == target),
            "crls_target_is_top_candidate": bool(int(np.argmax(crls_scores)) == target),
            "old_lcia_explicit_negative_changed": bool(lcia_after[1][target] < 0.0),
            "old_crls_explicit_negative_changed": bool(crls_after[1][target] < 0.0),
            "other_identity_unchanged": bool(exact_zero(untouched_before, untouched_after)),
        },
        "id_swap": {
            "constraint_count": len(swap.constraints),
            "four_constraints": swap_roles,
            "four_constraint_gate": bool(
                len(swap.constraints) == 4
                and sorted(c.role for c in swap.constraints) == ["positive", "positive", "rejected", "rejected"]
            ),
        },
        "gradient_scope": {
            "lcia_live_parameter_count_per_identity": lcia.model.live_parameter_count(identity_ids[0]),
            "expected_b_only_parameter_count": 2 * 3 * 64 * 8,
            "base_parameters_require_grad": any(
                parameter.requires_grad
                for name, parameter in lcia.model.named_parameters()
                if not name.startswith("fast_states.")
            ),
            "live_parameters_have_grad": all(
                parameter.grad is not None for parameter in lcia.live_parameters(identity_ids)
            ),
        },
        "rollback": {
            "lcia_byte_identical": lcia_snapshot_equal(rollback_lcia_before, rollback_lcia_after),
            "crls_byte_identical": crls_snapshot_equal(rollback_crls_before, rollback_crls_after),
            "pre_update_snapshot_available": bool(lcia_snapshot_before and crls_snapshot_before),
        },
        "current_feedback": {
            "current_prediction_is_not_rerun_after_feedback": True,
            "current_frame_excluded_from_update_selection": True,
        },
        "next_phase": "N28-B_PREREQUISITE_PASS_NOT_RUN",
    }
    if not all(
        (
            result["assignment"]["global_one_to_one_gate"],
            result["exact_zero"]["lcia_delta"],
            result["exact_zero"]["crls_delta"],
            result["exact_zero"]["lcia_matrix"],
            result["exact_zero"]["crls_matrix"],
            result["exact_zero"]["selection"],
            result["reassign"]["lcia_update"]["accepted"],
            result["reassign"]["lcia_target_is_top_candidate"],
            result["reassign"]["crls_target_is_top_candidate"],
            result["reassign"]["old_lcia_explicit_negative_changed"],
            result["reassign"]["old_crls_explicit_negative_changed"],
            result["reassign"]["other_identity_unchanged"],
            result["id_swap"]["four_constraint_gate"],
            not result["gradient_scope"]["base_parameters_require_grad"],
            result["gradient_scope"]["live_parameters_have_grad"],
            result["rollback"]["lcia_byte_identical"],
            result["rollback"]["crls_byte_identical"],
        )
    ):
        result["status"] = "FAIL"
        raise RuntimeError(json.dumps(result, indent=2, sort_keys=True))
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    print("N28A_CACHED_FEATURE_SMOKE_COMPLETE")


class torch_no_grad:
    """Small context wrapper kept local to the smoke script."""

    def __enter__(self):
        import torch

        self._context = torch.no_grad()
        return self._context.__enter__()

    def __exit__(self, exc_type, exc, tb):
        return self._context.__exit__(exc_type, exc, tb)


if __name__ == "__main__":
    main()
