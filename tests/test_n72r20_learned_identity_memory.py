import json
from pathlib import Path

import numpy as np
import pytest
import torch

from sam3_intermot.association.learned_identity_memory import LearnedIdentityMemoryBank
from sam3_intermot.association.learned_identity_state_edge import (
    build_learned_identity_state_edge_matrix,
)
from sam3_intermot.identity_memory.encoder import FEATURE_DIMENSION
from sam3_intermot.identity_memory.updater import VARIANT_GRU, build_updater


CHECKPOINT = Path("outputs/N72R18/checkpoints/identity_memory_gru.pt")
CHECKPOINT_SHA = "94ba9c44e177b254ee54984195e839f37c78a648c010ffc723dd3f1d553976a2"
ENCODER_SHA = "2809d3227f7d078f6045f7feb874a34d0684f0e0057b264b99adccf7d4519154"


def _feature(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    value = rng.normal(size=FEATURE_DIMENSION).astype(np.float32)
    return value / np.linalg.norm(value)


def _bank() -> LearnedIdentityMemoryBank:
    return LearnedIdentityMemoryBank.from_updater(
        build_updater(VARIANT_GRU),
        checkpoint_sha256="unit-checkpoint",
        encoder_sha256=ENCODER_SHA,
    )


def _solvers(uid: str | None, *, public_id: int = 7) -> tuple[dict, dict]:
    status = "EXPLICIT_NONE" if uid is None else "ASSIGNED_TO_PUBLIC_ID"
    item = {
        "public_id": public_id,
        "candidate_uid": uid,
        "status": status,
        "score": 0.0 if uid is None else 1.0,
    }
    return (
        {"public_assignments": [dict(item)], "runtime_future_gt_used": False},
        {"public_assignments": [dict(item)], "runtime_future_gt_used": False},
    )


def test_n72r20_checkpoint_strict_load_and_freeze():
    bank = LearnedIdentityMemoryBank.from_checkpoint(
        CHECKPOINT,
        encoder_sha256=ENCODER_SHA,
        expected_encoder_sha256=ENCODER_SHA,
        expected_checkpoint_sha256=CHECKPOINT_SHA,
    )
    assert bank.checkpoint_sha256 == CHECKPOINT_SHA
    assert all(not parameter.requires_grad for parameter in bank.updater.parameters())
    assert not bank.updater.training


def test_incompatible_encoder_sha_is_rejected():
    with pytest.raises(ValueError, match="encoder SHA mismatch"):
        LearnedIdentityMemoryBank.from_checkpoint(
            CHECKPOINT,
            encoder_sha256="wrong",
            expected_encoder_sha256=ENCODER_SHA,
        )


def test_human_anchor_is_immutable_and_state_starts_at_anchor():
    bank = _bank()
    anchor = _feature(1)
    bank.initialize_public_identity(
        public_id=7,
        association_state_id=70,
        frame=10,
        human_anchor=anchor,
        interaction_source="simulated_from_gt",
    )
    exposed = bank.records[7].human_anchor
    exposed[:] = 0
    assert np.allclose(bank.records[7].human_anchor, anchor)
    assert np.allclose(bank.records[7].current_state, anchor)
    with pytest.raises(ValueError, match="already initialized"):
        bank.initialize_public_identity(
            public_id=7,
            association_state_id=70,
            frame=11,
            human_anchor=_feature(2),
            interaction_source="simulated_from_gt",
        )


def test_score_before_update_and_none_does_not_update_memory():
    bank = _bank()
    anchor = _feature(3)
    bank.initialize_public_identity(
        public_id=7,
        association_state_id=70,
        frame=10,
        human_anchor=anchor,
        interaction_source="simulated_from_gt",
    )
    rows = [{"candidate_uid": "s:11:0", "embedding": anchor, "confidence": 0.9}]
    scored = bank.score_matrix(candidate_rows=rows, public_id_axis=[7], frame=11)
    assert float(scored["state_candidate_scores"][0, 0]) > 0.99
    before = bank.records[7].state_hash()
    base, treatment = _solvers(None)
    result = bank.update_from_consensus(
        frame=11,
        candidate_rows=rows,
        base_solver=base,
        treatment_solver=treatment,
    )
    assert result["updated_public_ids"] == []
    assert bank.records[7].state_hash() == before


def test_disagreement_does_not_update_memory():
    bank = _bank()
    bank.initialize_public_identity(
        public_id=7,
        association_state_id=70,
        frame=10,
        human_anchor=_feature(4),
        interaction_source="simulated_from_gt",
    )
    rows = [{"candidate_uid": "s:11:0", "embedding": _feature(5), "confidence": 0.9}]
    base, _ = _solvers("other")
    _, treatment = _solvers("s:11:0")
    before = bank.records[7].state_hash()
    result = bank.update_from_consensus(
        frame=11,
        candidate_rows=rows,
        base_solver=base,
        treatment_solver=treatment,
    )
    assert result["disagreement_public_ids"] == [7]
    assert bank.records[7].state_hash() == before


def test_consensus_update_is_applied_exactly_once():
    bank = _bank()
    bank.initialize_public_identity(
        public_id=7,
        association_state_id=70,
        frame=10,
        human_anchor=_feature(6),
        interaction_source="simulated_from_gt",
    )
    rows = [{"candidate_uid": "s:11:0", "embedding": _feature(7), "confidence": 0.8}]
    base, treatment = _solvers("s:11:0")
    first = bank.update_from_consensus(
        frame=11,
        candidate_rows=rows,
        base_solver=base,
        treatment_solver=treatment,
    )
    second = bank.update_from_consensus(
        frame=11,
        candidate_rows=rows,
        base_solver=base,
        treatment_solver=treatment,
    )
    assert first["updated_public_ids"] == [7]
    assert second["updated_public_ids"] == []
    assert bank.records[7].machine_update_count == 1
    assert len(bank.update_audit) == 1


def test_public_authority_is_immutable_and_uid_is_not_authority():
    bank = _bank()
    bank.initialize_public_identity(
        public_id=7,
        association_state_id=70,
        frame=10,
        human_anchor=_feature(8),
        interaction_source="simulated_from_gt",
    )
    rows = [{"candidate_uid": "native:999", "embedding": _feature(9), "confidence": 0.9}]
    base, treatment = _solvers("native:999")
    result = bank.update_from_consensus(
        frame=11,
        candidate_rows=rows,
        base_solver=base,
        treatment_solver=treatment,
    )
    assert result["updated_public_ids"] == [7]
    assert bank.records[7].public_id == 7
    assert bank.records[7].association_state_id == 70


def test_learned_edge_preserves_row_max():
    bank = _bank()
    bank.initialize_public_identity(
        public_id=7,
        association_state_id=70,
        frame=10,
        human_anchor=_feature(10),
        interaction_source="simulated_from_gt",
    )
    bank.initialize_public_identity(
        public_id=8,
        association_state_id=80,
        frame=10,
        human_anchor=_feature(11),
        interaction_source="simulated_from_gt",
    )
    rows = [
        {"candidate_uid": "s:11:0", "embedding": _feature(10)},
        {"candidate_uid": "s:11:1", "embedding": _feature(11)},
    ]
    base = np.asarray([[1.2, 0.4], [0.2, 1.1]])
    edge = build_learned_identity_state_edge_matrix(
        bank=bank,
        candidate_rows=rows,
        public_id_axis=[7, 8],
        frame=11,
        base_candidate_public_scores=base,
    )
    assert edge["row_max_preserved"] is True
    assert np.allclose(np.max(base, axis=1), np.max(edge["fused_candidate_public_scores"], axis=1))


def test_snapshot_restore_is_exact():
    bank = _bank()
    bank.initialize_public_identity(
        public_id=7,
        association_state_id=70,
        frame=10,
        human_anchor=_feature(12),
        interaction_source="simulated_from_gt",
    )
    rows = [{"candidate_uid": "s:11:0", "embedding": _feature(13), "confidence": 0.9}]
    base, treatment = _solvers("s:11:0")
    bank.update_from_consensus(
        frame=11,
        candidate_rows=rows,
        base_solver=base,
        treatment_solver=treatment,
    )
    snapshot = bank.snapshot()
    digest = bank.digest()
    bank.records[7]._current_state = _feature(14)
    bank.restore(snapshot)
    assert bank.digest() == digest
    assert bank.records[7].machine_update_count == 1


def test_runtime_future_gt_is_forbidden():
    bank = _bank()
    bank.initialize_public_identity(
        public_id=7,
        association_state_id=70,
        frame=10,
        human_anchor=_feature(15),
        interaction_source="simulated_from_gt",
    )
    rows = [{"candidate_uid": "s:11:0", "embedding": _feature(16), "runtime_future_gt_used": True}]
    with pytest.raises(ValueError, match="runtime_future_gt_used"):
        bank.score_matrix(candidate_rows=rows, public_id_axis=[7], frame=11)
    rows = [{"candidate_uid": "s:11:0", "embedding": _feature(16)}]
    base, treatment = _solvers("s:11:0")
    base["runtime_future_gt_used"] = True
    with pytest.raises(ValueError, match="runtime_future_gt_used"):
        bank.update_from_consensus(
            frame=11,
            candidate_rows=rows,
            base_solver=base,
            treatment_solver=treatment,
        )


def test_goal_and_lineage_are_frozen():
    goal = json.loads(Path("outputs/N72R20/FINAL_GOAL.json").read_text())
    lineage = json.loads(Path("outputs/N72R20/feature_lineage_audit.json").read_text())
    assert goal["goal"] == "Learned Identity Memory Integration into InterMOT"
    assert goal["goal_frozen"] is True
    assert goal["next_full_interactive_mot_stage_authorized"] is False
    assert lineage["feature_lineage_compatible"] is True
