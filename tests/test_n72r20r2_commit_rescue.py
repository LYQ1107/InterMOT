"""Contract tests for the N72R20R2 trusted-memory commit gate."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from sam3_intermot.association.identity_memory_commit import evaluate_commit
from sam3_intermot.association.learned_identity_memory import LearnedIdentityMemoryBank
from sam3_intermot.association.learned_identity_rescue import build_rescue_delta


ROOT = Path(__file__).resolve().parents[1]
R2 = ROOT / "outputs/N72R20R2"
DEV_SEQUENCES = [
    "dancetrack0001",
    "dancetrack0002",
    "dancetrack0023",
    "dancetrack0024",
    "dancetrack0039",
    "dancetrack0057",
    "dancetrack0062",
    "dancetrack0072",
]


def rows() -> list[dict[str, object]]:
    return [
        {"candidate_uid": "a", "feature": [1.0, 0.0], "runtime_future_gt_used": False},
        {"candidate_uid": "b", "feature": [0.0, 1.0], "runtime_future_gt_used": False},
        {"candidate_uid": "c", "feature": [-1.0, 0.0], "runtime_future_gt_used": False},
    ]


def test_goal_is_frozen_to_r2() -> None:
    goal = json.loads((R2 / "FINAL_GOAL.json").read_text(encoding="utf-8"))
    assert goal["stage"] == "N72R20R2"
    assert goal["goal"] == "Trusted Identity Memory Commit and Association Rescue"
    assert goal["goal_frozen"] is True
    assert goal["exact_solver_frozen"] is True
    assert goal["runtime_future_gt_forbidden"] is True


def test_commit_accepts_runtime_learned_top1() -> None:
    result = evaluate_commit(public_id=1, assigned_candidate_uid="a", candidate_rows=rows(), learned_state=[1.0, 0.0], frame=2)
    assert result["commit"] is True
    assert result["reason"] == "COMMIT_ACCEPTED"
    assert result["learned_top1_candidate_uid"] == "a"
    assert result["runtime_future_gt_used"] is False


def test_commit_rejects_none_assignment() -> None:
    result = evaluate_commit(public_id=1, assigned_candidate_uid=None, candidate_rows=rows(), learned_state=[1.0, 0.0], frame=2)
    assert result["commit"] is False
    assert result["reason"] == "NONE_ASSIGNMENT"


def test_commit_rejects_missing_assignment_candidate() -> None:
    result = evaluate_commit(public_id=1, assigned_candidate_uid="missing", candidate_rows=rows(), learned_state=[1.0, 0.0], frame=2)
    assert result["commit"] is False
    assert result["reason"] == "ASSIGNED_CANDIDATE_MISSING"


def test_commit_rejects_missing_state() -> None:
    result = evaluate_commit(public_id=1, assigned_candidate_uid="a", candidate_rows=rows(), learned_state=None, frame=2)
    assert result["commit"] is False
    assert result["reason"] == "LEARNED_STATE_MISSING"


def test_commit_rejects_when_assigned_is_not_top1() -> None:
    result = evaluate_commit(public_id=1, assigned_candidate_uid="b", candidate_rows=rows(), learned_state=[1.0, 0.0], frame=2)
    assert result["commit"] is False
    assert result["reason"] == "ASSIGNED_NOT_LEARNED_TOP1"
    assert result["learned_rank"] == 2


def test_commit_margin_threshold_is_inclusive() -> None:
    result = evaluate_commit(public_id=1, assigned_candidate_uid="a", candidate_rows=rows(), learned_state=[1.0, 0.0], frame=2, margin_threshold=1.0)
    assert result["commit"] is True
    result = evaluate_commit(public_id=1, assigned_candidate_uid="a", candidate_rows=rows(), learned_state=[1.0, 0.0], frame=2, margin_threshold=1.01)
    assert result["commit"] is False
    assert result["reason"] == "LEARNED_MARGIN_BELOW_THRESHOLD"


def test_commit_temporal_confirmation_is_explicit() -> None:
    result = evaluate_commit(public_id=1, assigned_candidate_uid="a", candidate_rows=rows(), learned_state=[1.0, 0.0], frame=2, temporal_confirmed=False, require_temporal_confirmation=True)
    assert result["commit"] is False
    assert result["reason"] == "TEMPORAL_CONFIRMATION_INCOMPLETE"


def test_commit_rejects_runtime_gt_metadata() -> None:
    bad_rows = rows()
    bad_rows[0]["runtime_future_gt_used"] = True
    with pytest.raises(ValueError, match="runtime_future_gt_used"):
        evaluate_commit(public_id=1, assigned_candidate_uid="a", candidate_rows=bad_rows, learned_state=[1.0, 0.0], frame=2)


def test_commit_rejects_invalid_threshold() -> None:
    with pytest.raises(ValueError, match="margin_threshold"):
        evaluate_commit(public_id=1, assigned_candidate_uid="a", candidate_rows=rows(), learned_state=[1.0, 0.0], frame=2, margin_threshold=-0.1)


def test_commit_rejects_all_invalid_features() -> None:
    bad_rows = [{"candidate_uid": "a", "feature": [0.0, 0.0]}]
    result = evaluate_commit(public_id=1, assigned_candidate_uid="a", candidate_rows=bad_rows, learned_state=[1.0, 0.0], frame=2)
    assert result["commit"] is False
    assert result["reason"] == "NO_VALID_FEATURE"


def test_commit_tie_break_is_deterministic_by_uid() -> None:
    result = evaluate_commit(public_id=1, assigned_candidate_uid="a", candidate_rows=rows(), learned_state=[0.0, 1.0], frame=2)
    assert result["learned_top1_candidate_uid"] == "b"


def test_rescue_is_target_only() -> None:
    result = build_rescue_delta(
        base_matrix=[[0.8, 0.4], [0.5, 0.7]],
        public_id_axis=[1, 2],
        target_public_id=1,
        candidate_rows=rows()[:2],
        learned_scores=[0.1, 0.9],
        base_assignment_candidate_uid="a",
        runtime_policy={"lambda": 2.0, "margin_threshold": 0.0},
    )
    assert result["rescue_active"] is True
    assert result["raw_delta_non_target_max_abs"] == 0.0
    delta = np.asarray(result["relative_delta"], dtype=float)
    assert np.allclose(delta[:, 1], 0.0)


def test_rescue_inactive_when_base_already_agrees() -> None:
    result = build_rescue_delta(
        base_matrix=[[0.8, 0.4], [0.5, 0.7]], public_id_axis=[1, 2], target_public_id=1,
        candidate_rows=rows()[:2], learned_scores=[0.9, 0.1], base_assignment_candidate_uid="a",
        runtime_policy={"lambda": 1.0},
    )
    assert result["rescue_active"] is False
    assert result["activation_reason"] == "INACTIVE_BASE_ALREADY_LEARNED_TOP1"
    assert np.allclose(result["relative_delta"], 0.0)


def test_rescue_margin_gate_is_runtime_only() -> None:
    result = build_rescue_delta(
        base_matrix=[[0.8, 0.4], [0.5, 0.7]], public_id_axis=[1, 2], target_public_id=1,
        candidate_rows=rows()[:2], learned_scores=[0.51, 0.5], base_assignment_candidate_uid="b",
        runtime_policy={"lambda": 1.0, "margin_threshold": 0.02},
    )
    assert result["rescue_active"] is False
    assert result["activation_reason"] == "INACTIVE_LEARNED_MARGIN_BELOW_THRESHOLD"


def test_rescue_motion_gate_fails_closed() -> None:
    result = build_rescue_delta(
        base_matrix=[[0.8, 0.4], [0.5, 0.7]], public_id_axis=[1, 2], target_public_id=1,
        candidate_rows=rows()[:2], learned_scores=[0.1, 0.9], base_assignment_candidate_uid="a",
        runtime_policy={"lambda": 1.0, "min_motion_plausibility": 0.5}, motion_plausibility=0.2,
    )
    assert result["rescue_active"] is False
    assert result["activation_reason"] == "INACTIVE_MOTION_BELOW_THRESHOLD"


def test_rescue_preserves_row_maximum() -> None:
    result = build_rescue_delta(
        base_matrix=[[0.8, 0.4], [0.5, 0.7]], public_id_axis=[1, 2], target_public_id=1,
        candidate_rows=rows()[:2], learned_scores=[0.1, 0.9], base_assignment_candidate_uid="a",
        runtime_policy={"lambda": 1.0},
    )
    fused = np.asarray(result["fused_candidate_public_scores"])
    base = np.asarray([[0.8, 0.4], [0.5, 0.7]])
    assert result["row_max_preserved"] is True
    assert np.allclose(np.max(fused, axis=1), np.max(base, axis=1))


def test_rescue_does_not_change_candidate_axis() -> None:
    result = build_rescue_delta(
        base_matrix=[[0.8, 0.4], [0.5, 0.7]], public_id_axis=[1, 2], target_public_id=1,
        candidate_rows=rows()[:2], learned_scores=[0.1, 0.9], base_assignment_candidate_uid="a",
        runtime_policy={"lambda": 1.0},
    )
    assert result["candidate_axis_unchanged"] is True
    assert result["candidate_uids"] == ["a", "b"]
    assert result["public_id_axis"] == [1, 2]


def test_rescue_lambda_scales_target_residual() -> None:
    common = dict(base_matrix=[[0.8, 0.4], [0.5, 0.7]], public_id_axis=[1, 2], target_public_id=1,
                  candidate_rows=rows()[:2], learned_scores=[0.1, 0.9], base_assignment_candidate_uid="a")
    one = build_rescue_delta(runtime_policy={"lambda": 1.0}, **common)
    two = build_rescue_delta(runtime_policy={"lambda": 2.0}, **common)
    assert np.allclose(np.asarray(two["relative_delta"])[:, 0], 2.0 * np.asarray(one["relative_delta"])[:, 0])


def test_rescue_rejects_shape_mismatch() -> None:
    with pytest.raises(ValueError, match="inconsistent"):
        build_rescue_delta(base_matrix=[[1.0]], public_id_axis=[1, 2], target_public_id=1,
                           candidate_rows=rows()[:1], learned_scores=[1.0], base_assignment_candidate_uid="a",
                           runtime_policy={})


def test_rescue_rejects_nonfinite_lambda() -> None:
    with pytest.raises(ValueError, match="lambda"):
        build_rescue_delta(base_matrix=[[1.0]], public_id_axis=[1], target_public_id=1,
                           candidate_rows=rows()[:1], learned_scores=[1.0], base_assignment_candidate_uid="a",
                           runtime_policy={"lambda": float("nan")})


def test_human_anchor_is_immutable_copy() -> None:
    bank = LearnedIdentityMemoryBank.from_updater(torch.nn.Identity())
    bank.initialize_public_identity(public_id=1, association_state_id=7, frame=0, human_anchor=[1.0] + [0.0] * 511, interaction_source="test")
    original = bank.records[1].anchor_hash()
    exposed = bank.records[1].human_anchor
    exposed[0] = 0.0
    assert bank.records[1].anchor_hash() == original
    assert bank.records[1].human_anchor[0] == 1.0


def test_duplicate_public_identity_is_rejected() -> None:
    bank = LearnedIdentityMemoryBank.from_updater(torch.nn.Identity())
    anchor = [1.0] + [0.0] * 511
    bank.initialize_public_identity(public_id=1, association_state_id=7, frame=0, human_anchor=anchor, interaction_source="test")
    with pytest.raises(ValueError, match="already initialized"):
        bank.initialize_public_identity(public_id=1, association_state_id=8, frame=1, human_anchor=anchor, interaction_source="test")


def test_duplicate_association_state_is_rejected() -> None:
    bank = LearnedIdentityMemoryBank.from_updater(torch.nn.Identity())
    anchor = [1.0] + [0.0] * 511
    bank.initialize_public_identity(public_id=1, association_state_id=7, frame=0, human_anchor=anchor, interaction_source="test")
    with pytest.raises(ValueError, match="association state"):
        bank.initialize_public_identity(public_id=2, association_state_id=7, frame=1, human_anchor=anchor, interaction_source="test")


def test_formal_commit_gate_has_no_passing_policy() -> None:
    decision = json.loads((R2 / "commit_gate_decision.json").read_text(encoding="utf-8"))
    assert decision["decision"] == "FAIL_RUNTIME_MEMORY_COMMIT"
    assert decision["passing_policy_count"] == 0
    assert decision["rescue_authorized"] is False
    assert decision["runtime_future_gt_used"] is False
    assert all(not item["safety_pass"] or not item["usefulness_pass"] for item in decision["policy_summary"])


def test_candidate_lineage_is_formally_validated() -> None:
    validation = json.loads((R2 / "candidate_tape_validation.json").read_text(encoding="utf-8"))
    assert validation["status"] == "PASS_N72R20R2_CANDIDATE_TAPE_VALIDATION"
    assert validation["stage"] == "N72R20R2"
    assert len(validation["sequences"]) == 8
    assert validation["total_frames"] == 8422
    assert validation["runtime_future_gt_used"] is False


def test_dataset_lineage_has_exact_dev_sequences() -> None:
    manifest = json.loads((R2 / "asset_manifest.json").read_text(encoding="utf-8"))
    assert manifest["dataset"]["train_sequences"] == DEV_SEQUENCES
    assert manifest["dataset"]["test_present_or_used"] is False
    assert manifest["dataset"]["runtime_future_gt_used"] is False


def test_final_status_blocks_val_and_association() -> None:
    status = json.loads((R2 / "stage_status.json").read_text(encoding="utf-8"))
    assert status["goal_reference"] == "outputs/N72R20R2/FINAL_GOAL.json"
    assert status["final_decision"] == "FAIL_RUNTIME_MEMORY_COMMIT"
    assert status["next_association_stage_authorized"] is False
    assert status["val"]["started"] is False
    assert status["rescue"]["started"] is False


def test_final_result_uses_only_allowed_failure_decision() -> None:
    result = json.loads((R2 / "FINAL_RESULT.json").read_text(encoding="utf-8"))
    assert result["decision"] in {
        "PASS_TRUSTED_IDENTITY_MEMORY_INTEGRATION",
        "FAIL_RUNTIME_MEMORY_COMMIT",
        "FAIL_ASSOCIATION_RESCUE",
        "FAIL_VAL_MEMORY_SAFETY",
        "FAIL_VAL_ASSOCIATION_GAIN",
        "FAIL_RUNTIME_INVARIANT",
        "FAIL_ASSET_OR_STORAGE",
    }
    assert result["decision"] == "FAIL_RUNTIME_MEMORY_COMMIT"
    assert result["next_association_stage_authorized"] is False


def test_previous_stage_results_are_preserved() -> None:
    result = json.loads((R2 / "FINAL_RESULT.json").read_text(encoding="utf-8"))
    assert "N72R20: BLOCKED_NOT_RECOVERED" in result["previous_results_preserved"]
    assert "N72R20R1: FAIL_MEMORY_CONTAMINATION" in result["previous_results_preserved"]


def test_frozen_assets_are_recorded() -> None:
    manifest = json.loads((R2 / "asset_manifest.json").read_text(encoding="utf-8"))
    assert manifest["frozen_identity_assets"]["n72r18_gru"]["sha256"]
    assert manifest["frozen_identity_assets"]["osnet_x1_0_market1501"]["sha256"]
    assert manifest["sam3_checkpoint"]["sha256"]
    assert manifest["protocol"]["assignment_solver"] == "solve_effect_assignment_then_solve_exact_public_assignment"
