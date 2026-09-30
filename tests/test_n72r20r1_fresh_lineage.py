"""Structural gates for the N72R20R1 fresh-lineage experiment."""

import json
from pathlib import Path

import numpy as np
import pytest

from sam3_intermot.association.learned_identity_memory import LearnedIdentityMemoryBank
from sam3_intermot.association.learned_identity_state_edge import build_learned_identity_state_edge_matrix
from sam3_intermot.association.relative_persistent_state_edge import fuse_row_max_preserving
from sam3_intermot.identity_memory.updater import VARIANT_GRU, build_updater


ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT = ROOT / "outputs/N72R18/checkpoints/identity_memory_gru.pt"
CHECKPOINT_SHA = "94ba9c44e177b254ee54984195e839f37c78a648c010ffc723dd3f1d553976a2"
ENCODER_SHA = "2809d3227f7d078f6045f7feb874a34d0684f0e0057b264b99adccf7d4519154"


def feature(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    value = rng.normal(size=512).astype(np.float32)
    return value / np.linalg.norm(value)


def bank() -> LearnedIdentityMemoryBank:
    return LearnedIdentityMemoryBank.from_updater(
        build_updater(VARIANT_GRU),
        checkpoint_sha256="r1-test-checkpoint",
        encoder_sha256=ENCODER_SHA,
    )


def solver(uid: str | None, public_id: int = 7) -> dict:
    return {
        "public_assignments": [
            {
                "public_id": public_id,
                "candidate_uid": uid,
                "status": "EXPLICIT_NONE" if uid is None else "ASSIGNED_TO_PUBLIC_ID",
                "score": 0.0 if uid is None else 1.0,
            }
        ],
        "runtime_future_gt_used": False,
    }


def initialized() -> LearnedIdentityMemoryBank:
    value = bank()
    value.initialize_public_identity(
        public_id=7,
        association_state_id=70,
        frame=0,
        human_anchor=feature(1),
        interaction_source="simulated_from_gt",
    )
    return value


def test_r1_strict_gru_load() -> None:
    loaded = LearnedIdentityMemoryBank.from_checkpoint(
        CHECKPOINT,
        encoder_sha256=ENCODER_SHA,
        expected_encoder_sha256=ENCODER_SHA,
        expected_checkpoint_sha256=CHECKPOINT_SHA,
    )
    assert loaded.checkpoint_sha256 == CHECKPOINT_SHA
    assert not loaded.updater.training
    assert all(not parameter.requires_grad for parameter in loaded.updater.parameters())


def test_r1_encoder_sha_validation() -> None:
    with pytest.raises(ValueError, match="encoder SHA mismatch"):
        LearnedIdentityMemoryBank.from_checkpoint(
            CHECKPOINT,
            encoder_sha256="wrong",
            expected_encoder_sha256=ENCODER_SHA,
        )


def test_r1_fresh_candidate_uid_uniqueness() -> None:
    validation = json.loads((ROOT / "outputs/N72R20R1/candidate_tape_validation.json").read_text())
    assert validation["status"] == "PASS_N72R20R1_CANDIDATE_TAPE_VALIDATION"
    for sequence in validation["sequences"]:
        assert sequence["unique_candidate_uid_count"] == sequence["candidate_count"]


def test_r1_runtime_future_gt_forbidden() -> None:
    value = initialized()
    rows = [{"candidate_uid": "c", "embedding": feature(2), "runtime_future_gt_used": True}]
    with pytest.raises(ValueError, match="runtime_future_gt_used"):
        value.score_matrix(candidate_rows=rows, public_id_axis=[7], frame=1)


def test_r1_score_before_update() -> None:
    value = initialized()
    rows = [{"candidate_uid": "c", "embedding": feature(3)}]
    before = value.records[7].state_hash()
    scored = value.score_matrix(candidate_rows=rows, public_id_axis=[7], frame=1)
    assert scored["frame"] == 1
    assert value.records[7].state_hash() == before


def test_r1_none_does_not_update() -> None:
    value = initialized()
    rows = [{"candidate_uid": "c", "embedding": feature(4)}]
    before = value.records[7].state_hash()
    result = value.update_from_consensus(
        frame=1,
        candidate_rows=rows,
        base_solver=solver(None),
        treatment_solver=solver(None),
    )
    assert result["updated_public_ids"] == []
    assert value.records[7].state_hash() == before


def test_r1_consensus_update_once() -> None:
    value = initialized()
    rows = [{"candidate_uid": "c", "embedding": feature(5)}]
    first = value.update_from_consensus(frame=1, candidate_rows=rows, base_solver=solver("c"), treatment_solver=solver("c"))
    second = value.update_from_consensus(frame=1, candidate_rows=rows, base_solver=solver("c"), treatment_solver=solver("c"))
    assert first["updated_public_ids"] == [7]
    assert second["updated_public_ids"] == []


def test_r1_disagreement_does_not_update() -> None:
    value = initialized()
    rows = [{"candidate_uid": "c", "embedding": feature(6)}]
    before = value.records[7].state_hash()
    result = value.update_from_consensus(frame=1, candidate_rows=rows, base_solver=solver("other"), treatment_solver=solver("c"))
    assert result["disagreement_public_ids"] == [7]
    assert value.records[7].state_hash() == before


def test_r1_human_anchor_immutable() -> None:
    value = initialized()
    anchor = value.records[7].human_anchor.copy()
    value.records[7].human_anchor[:] = 0
    assert np.allclose(value.records[7].human_anchor, anchor)
    assert np.allclose(value.records[7].current_state, anchor)


def test_r1_public_id_immutable() -> None:
    value = initialized()
    value.update_from_consensus(frame=1, candidate_rows=[{"candidate_uid": "c", "embedding": feature(7)}], base_solver=solver("c"), treatment_solver=solver("c"))
    assert value.records[7].public_id == 7
    assert value.records[7].association_state_id == 70


def test_r1_row_max_preservation() -> None:
    base = np.asarray([[1.2, 0.3], [0.2, 1.1]], dtype=np.float64)
    delta = np.asarray([[0.4, -0.7], [-0.2, 0.5]], dtype=np.float64)
    fused = fuse_row_max_preserving(base, delta)
    assert fused["row_max_preserved"] is True
    assert np.allclose(np.max(base, axis=1), np.max(fused["fused"], axis=1))


def test_r1_snapshot_restore() -> None:
    value = initialized()
    value.update_from_consensus(frame=1, candidate_rows=[{"candidate_uid": "c", "embedding": feature(8)}], base_solver=solver("c"), treatment_solver=solver("c"))
    snapshot = value.snapshot()
    digest = value.digest()
    value.records[7]._current_state = feature(9)
    value.restore(snapshot)
    assert value.digest() == digest


def test_r1_same_tape_reused_by_e0_e1_e2() -> None:
    manifest = json.loads((ROOT / "outputs/N72R20R1/fresh_tape_manifest.json").read_text())
    assert manifest["fresh_tape_sealed"] is True
    assert len(manifest["candidate_tapes"]) == len(manifest["base_score_tapes"]) == 2
    replay = json.loads((ROOT / "outputs/N72R20R1/causal_replay/train_smoke_replay_manifest.json").read_text())
    assert replay["candidate_tape_shared"] is True
    assert replay["base_score_tape_shared"] is True
    assert replay["variants"] == ["FRESH_BASELINE_B0", "FRESH_TRUSTED_RELATIVE_STATE", "FRESH_LEARNED_IDENTITY_TRUSTED"]


def test_r1_feature_lineage_mismatch_fails_closed() -> None:
    with pytest.raises(ValueError, match="checkpoint SHA mismatch"):
        LearnedIdentityMemoryBank.from_checkpoint(
            CHECKPOINT,
            encoder_sha256=ENCODER_SHA,
            expected_encoder_sha256=ENCODER_SHA,
            expected_checkpoint_sha256="wrong-checkpoint-sha",
        )


def test_r1_contamination_audit_is_terminal_and_posthoc_only() -> None:
    audit = json.loads((ROOT / "outputs/N72R20R1/contamination_audit.json").read_text())
    result = json.loads((ROOT / "outputs/N72R20R1/FINAL_RESULT.json").read_text())
    status = json.loads((ROOT / "outputs/N72R20R1/stage_status.json").read_text())
    assert audit["status"] == "FAIL_MEMORY_CONTAMINATION"
    assert audit["runtime_future_gt_used"] is False
    assert audit["posthoc_gt_used"] is True
    assert audit["write_checks"]["candidate_feature_lineage_fixed"] is True
    assert audit["write_checks"]["runtime_gt_flags_clean"] is True
    assert audit["write_checks"]["wrong_memory_write_count_zero"] is False
    assert len(audit["posthoc_wrong_memory_writes"]) == 33
    assert result["final_decision"] == "FAIL_MEMORY_CONTAMINATION"
    assert status["final_decision"] == "FAIL_MEMORY_CONTAMINATION"
    assert status["next_full_interactive_mot_stage_authorized"] is False
