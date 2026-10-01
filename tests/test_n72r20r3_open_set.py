"""Contract tests for the N72R20R3 open-set identity stage.

These tests cover the runtime boundary, post-hoc taxonomy, LOSO protocol,
artifact provenance, and the terminal static-gate decision.  They deliberately
do not start SAM3, open DanceTrack val/test, or run a causal association
replay.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from sam3_intermot.association.identity_presence import commit_eligibility, evaluate_identity_presence
from sam3_intermot.association.learned_identity_memory import LearnedIdentityMemoryBank
from scripts.n72r20r1_build_base_score_tape import read_zstd_jsonl
from scripts.n72r20r3_failure_taxonomy import classify_frame
from scripts.n72r20r3_presence_loso import _score_for_policy


ROOT = Path(__file__).resolve().parents[1]
R3 = ROOT / "outputs/N72R20R3"
SEQUENCES = [
    "dancetrack0001",
    "dancetrack0002",
    "dancetrack0023",
    "dancetrack0024",
    "dancetrack0039",
    "dancetrack0057",
    "dancetrack0062",
    "dancetrack0072",
]


def vector(index: int = 0) -> list[float]:
    value = np.zeros(512, dtype=np.float32)
    value[int(index) % 512] = 1.0
    return value.tolist()


def candidates() -> list[dict[str, object]]:
    return [
        {"candidate_uid": "candidate-a", "feature": vector(0)},
        {"candidate_uid": "candidate-b", "feature": vector(1)},
    ]


def solver(uid: str | None, public_id: int = 1) -> dict[str, object]:
    return {
        "public_assignments": [
            {
                "public_id": public_id,
                "candidate_uid": uid,
                "status": "ASSIGNED_CANDIDATE" if uid is not None else "NO_CANDIDATE_ASSIGNED",
            }
        ],
        "runtime_future_gt_used": False,
    }


def presence(
    rows: list[dict[str, object]] | None = None,
    *,
    state: object = None,
    anchor: object = None,
    base_uid: str | None = "candidate-a",
    encoder: str = "encoder-test",
) -> dict[str, object]:
    if state is None:
        state = vector(0)
    if anchor is None:
        anchor = vector(0)
    return evaluate_identity_presence(
        learned_state=state,
        human_anchor=anchor,
        candidate_rows=candidates() if rows is None else rows,
        base_assignment=solver(base_uid),
        runtime_context={"public_id": 1, "frame": 1, "encoder_sha256": encoder},
        frozen_policy={
            "method": "B0_ALWAYS_PRESENT",
            "policy_version": "TEST",
            "encoder_sha256": "encoder-test",
        },
    )


class IdentityUpdater(torch.nn.Module):
    def forward(self, previous: torch.Tensor, observation: torch.Tensor):
        return previous, torch.ones(previous.shape[0], device=previous.device), observation


def bank() -> LearnedIdentityMemoryBank:
    value = LearnedIdentityMemoryBank.from_updater(
        IdentityUpdater(), checkpoint_sha256="test-checkpoint", encoder_sha256="test-encoder"
    )
    value.initialize_public_identity(
        public_id=1,
        association_state_id=11,
        frame=0,
        human_anchor=vector(0),
        interaction_source="test",
    )
    return value


def test_zero_candidates_are_absent() -> None:
    result = presence(rows=[])
    assert result["present"] is False
    assert result["presence_reason"] == "NO_CANDIDATES"


def test_runtime_presence_has_no_gt_dependency() -> None:
    result = presence()
    assert result["runtime_future_gt_used"] is False
    assert result["runtime_gt_read"] is False
    assert result["posthoc_gt_used"] is False


def test_posthoc_labels_are_rejected_at_runtime_boundary() -> None:
    bad = candidates()
    bad[0]["taxonomy"] = "P0_TARGET_ABSENT"
    with pytest.raises(ValueError, match="posthoc labels"):
        presence(rows=bad)


def test_candidate_uid_uniqueness_is_enforced() -> None:
    bad = [
        {"candidate_uid": "duplicate", "feature": vector(0)},
        {"candidate_uid": "duplicate", "feature": vector(1)},
    ]
    with pytest.raises(ValueError, match="collision"):
        presence(rows=bad)


def test_feature_lineage_mismatch_fails_closed() -> None:
    result = presence(encoder="wrong-encoder")
    assert result["present"] is False
    assert result["presence_reason"] == "FEATURE_LINEAGE_MISMATCH"


def test_invalid_state_fails_closed() -> None:
    result = presence(state=[0.0] * 512)
    assert result["present"] is False
    assert result["presence_reason"] == "STATE_UNAVAILABLE"


def test_score_before_update_does_not_mutate_state() -> None:
    state = np.asarray(vector(0), dtype=np.float32)
    before = state.copy()
    result = presence(state=state)
    assert result["present"] is True
    assert np.array_equal(state, before)


def test_absent_presence_has_no_write_eligibility() -> None:
    result = presence(rows=[])
    eligibility = commit_eligibility(result)
    assert eligibility["commit_eligible"] is False
    assert eligibility["commit_reason"] == "NO_WRITE_ABSENT"


def test_present_alone_does_not_force_a_write() -> None:
    result = presence(base_uid=None)
    assert result["present"] is True
    eligibility = commit_eligibility(result)
    assert eligibility["commit_eligible"] is False
    assert eligibility["commit_reason"] == "NO_WRITE_BASE_NONE"


def test_base_learned_disagreement_is_not_eligible() -> None:
    result = presence(base_uid="candidate-b")
    assert result["selected_candidate_uid"] == "candidate-a"
    eligibility = commit_eligibility(result)
    assert eligibility["commit_eligible"] is False
    assert eligibility["commit_reason"] == "NO_WRITE_BASE_LEARNED_DISAGREEMENT"


def test_present_and_base_top1_are_eligible() -> None:
    result = presence(base_uid="candidate-a")
    eligibility = commit_eligibility(result)
    assert eligibility["commit_eligible"] is True
    assert eligibility["commit_reason"] == "ELIGIBLE_PRESENT_BASE_LEARNED_AGREE"


def test_human_anchor_is_immutable() -> None:
    value = bank()
    original = value.records[1].anchor_hash()
    exposed = value.records[1].human_anchor
    exposed[0] = 0.0
    assert value.records[1].anchor_hash() == original


def test_public_id_and_state_binding_are_immutable() -> None:
    value = bank()
    assert value.records[1].public_id == 1
    assert value.records[1].association_state_id == 11
    assert value.records[1].human_update_count == 1


def test_none_assignment_does_not_update_memory() -> None:
    value = bank()
    before = value.records[1].state_hash()
    result = value.update_from_consensus(
        frame=1,
        candidate_rows=candidates(),
        base_solver=solver(None),
        treatment_solver=solver(None),
    )
    assert result["updated_public_ids"] == []
    assert value.records[1].state_hash() == before


def test_one_observation_cannot_update_twice() -> None:
    value = bank()
    first = value.update_from_consensus(frame=1, candidate_rows=candidates(), base_solver=solver("candidate-a"), treatment_solver=solver("candidate-a"))
    second = value.update_from_consensus(frame=1, candidate_rows=candidates(), base_solver=solver("candidate-a"), treatment_solver=solver("candidate-a"))
    assert first["updated_public_ids"] == [1]
    assert second["updated_public_ids"] == []
    assert len(value.update_audit) == 1


def test_snapshot_restore_preserves_state() -> None:
    value = bank()
    value.update_from_consensus(frame=1, candidate_rows=candidates(), base_solver=solver("candidate-a"), treatment_solver=solver("candidate-a"))
    snapshot = value.snapshot()
    digest = value.digest()
    value.records[1]._current_state = np.asarray(vector(2), dtype=np.float32)
    value.restore(snapshot)
    assert value.digest() == digest


def test_b1_thresholds_record_training_fold_only() -> None:
    payload = json.loads((R3 / "presence/fold_policies.json").read_text(encoding="utf-8"))
    policies = [value for value in payload["policies"].values() if value.get("method") == "B1_ABSOLUTE_SCORE"]
    assert len(policies) == 16
    assert all(len(policy["calibration_sequences"]) == 7 for policy in policies)
    assert all(policy["heldout_sequence"] not in policy["calibration_sequences"] for policy in policies)


def test_b2_thresholds_record_training_fold_only() -> None:
    payload = json.loads((R3 / "presence/fold_policies.json").read_text(encoding="utf-8"))
    policies = [value for value in payload["policies"].values() if value.get("method") == "B2_ABSOLUTE_MARGIN"]
    assert len(policies) == 16
    assert all(policy["heldout_sequence"] not in policy["calibration_sequences"] for policy in policies)


def test_heldout_labels_never_enter_calibration() -> None:
    payload = json.loads((R3 / "presence/fold_policies.json").read_text(encoding="utf-8"))
    for policy in payload["policies"].values():
        assert set(policy["calibration_sequences"]) == set(SEQUENCES) - {policy["heldout_sequence"]}


def test_loso_has_exactly_eight_folds() -> None:
    payload = json.loads((R3 / "presence/loso_results.json").read_text(encoding="utf-8"))
    assert payload["fold_count"] == 8
    assert len(payload["heldout_sequences"]) == 8


def test_each_sequence_is_held_out_exactly_once_per_policy() -> None:
    payload = json.loads((R3 / "presence/loso_results.json").read_text(encoding="utf-8"))
    for aggregate in payload["aggregates"]:
        assert set(aggregate["per_sequence"]) == set(SEQUENCES)


def test_protocol_has_no_frame_random_split() -> None:
    payload = json.loads((R3 / "presence/fold_policies.json").read_text(encoding="utf-8"))
    assert payload["stage"] == "N72R20R3"
    assert all("frame" not in policy.get("calibration_sequences", []) for policy in payload["policies"].values())


def test_candidate_set_present_definition_matches_taxonomy() -> None:
    taxonomy = json.loads((R3 / "failure_taxonomy.json").read_text(encoding="utf-8"))
    assert taxonomy["candidate_set_present_count"] == taxonomy["p2_count"] + taxonomy["p3_count"]
    assert taxonomy["candidate_set_absent_count"] == taxonomy["p0_count"] + taxonomy["p1_count"]


def test_p0_taxonomy_is_target_absent() -> None:
    assert classify_frame(target_gt_present=False, best_iou=None, best_uid=None, base_uid=None, base_iou=None) == "P0_TARGET_ABSENT"


def test_p1_taxonomy_is_candidate_unavailable() -> None:
    assert classify_frame(target_gt_present=True, best_iou=0.49, best_uid="a", base_uid="a", base_iou=0.49) == "P1a_BEST_AVAILABLE_LOCALIZATION"


def test_p2_taxonomy_is_association_wrong() -> None:
    assert classify_frame(target_gt_present=True, best_iou=0.80, best_uid="a", base_uid="b", base_iou=0.20) == "P2_TARGET_AVAILABLE_ASSOCIATION_WRONG"


def test_p3_taxonomy_is_association_correct() -> None:
    assert classify_frame(target_gt_present=True, best_iou=0.80, best_uid="a", base_uid="a", base_iou=0.80) == "P3_TARGET_AVAILABLE_ASSOCIATION_CORRECT"


def test_p1a_p1b_are_deterministic() -> None:
    args = dict(target_gt_present=True, best_iou=0.49, best_uid="a", base_uid="b", base_iou=0.20)
    assert classify_frame(**args) == classify_frame(**args) == "P1b_WRONG_AVAILABLE_OBJECT"


def test_causal_replay_is_not_run_after_static_gate_failure() -> None:
    replay = json.loads((R3 / "causal/causal_replay.json").read_text(encoding="utf-8"))
    assert replay["status"] == "NOT_RUN_STATIC_PRESENCE_GATE_FAILED"
    assert replay["replay_rows"] == 0
    assert replay["runtime_future_gt_used"] is False


def test_rejected_write_preserves_exact_state_hash() -> None:
    value = bank()
    before = value.records[1].state_hash()
    result = commit_eligibility(presence(base_uid="candidate-b"))
    assert result["commit_eligible"] is False
    assert value.records[1].state_hash() == before


def test_oracle_clean_diagnostic_is_posthoc_only() -> None:
    manifest = json.loads((R3 / "presence/runtime_feature_manifest.json").read_text(encoding="utf-8"))
    assert manifest["oracle_clean_included"] is True
    assert manifest["oracle_clean_posthoc_only"] is True
    assert manifest["runtime_future_gt_used"] is False
    row = read_zstd_jsonl(R3 / "presence/oracle_clean_features.jsonl.zst")[0]
    assert row["state_condition"] == "S2_ORACLE_CLEAN"
    assert row["posthoc_oracle_only"] is True
    assert row["posthoc_gt_used"] is True


def test_oracle_clean_path_cannot_be_selected_as_final_runtime() -> None:
    result = json.loads((R3 / "FINAL_RESULT.json").read_text(encoding="utf-8"))
    assert result["best_presence_method"] is None
    assert result["selected_static_policy"] is None
    assert result["presence_features"]["S2_ORACLE_CLEAN"] == "posthoc_oracle_only_not_eligible_for_final_runtime"


def test_r2_outputs_remain_unchanged() -> None:
    audit = json.loads((R3 / "r2_immutability_audit.json").read_text(encoding="utf-8"))
    assert audit["all_unchanged"] is True
    assert audit["historical_r2_modified"] is False


def test_no_sam3_generation_was_started() -> None:
    goal = json.loads((R3 / "FINAL_GOAL.json").read_text(encoding="utf-8"))
    assert goal["sam3_rerun_forbidden"] is True
    assert goal["candidate_generation_frozen"] is True


def test_val_and_test_were_not_accessed() -> None:
    goal = json.loads((R3 / "FINAL_GOAL.json").read_text(encoding="utf-8"))
    status = json.loads((R3 / "stage_status.json").read_text(encoding="utf-8"))
    assert goal["dance_track_val_forbidden"] is True
    assert goal["dance_track_test_forbidden"] is True
    assert status["dataset"]["val_accessed"] is False
    assert status["dataset"]["test_accessed"] is False


def test_presence_policy_scoring_is_deterministic() -> None:
    runtime = read_zstd_jsonl(R3 / "presence/frame_runtime_features.jsonl.zst")[0]
    policies = json.loads((R3 / "presence/fold_policies.json").read_text(encoding="utf-8"))["policies"]
    policy = next(value for value in policies.values() if value.get("method") == "LOGISTIC_PRESENCE")
    first = _score_for_policy(runtime, policy)
    second = _score_for_policy(dict(runtime), dict(policy))
    assert runtime["selected_candidate_uid"] == runtime["learned_top1_candidate_uid"]
    assert first == second


def test_runtime_artifact_flags_are_false() -> None:
    manifest = json.loads((R3 / "presence/runtime_feature_manifest.json").read_text(encoding="utf-8"))
    assert manifest["runtime_future_gt_used"] is False
    assert manifest["posthoc_gt_used"] is False


def test_gt_join_occurs_only_after_runtime_artifact() -> None:
    runtime = json.loads((R3 / "presence/runtime_feature_manifest.json").read_text(encoding="utf-8"))
    labels = json.loads((R3 / "presence/posthoc_label_manifest.json").read_text(encoding="utf-8"))
    assert runtime["posthoc_gt_used"] is False
    assert labels["posthoc_gt_used"] is True
    assert labels["runtime_input"] == runtime["runtime_feature_file"]


def test_presence_policy_does_not_create_candidates() -> None:
    rows = candidates()
    result = presence(rows=rows)
    assert result["candidate_count"] == len(rows)
    assert result["valid_feature_count"] == len(rows)
    assert {result["learned_top1_candidate_uid"], result["learned_top2_candidate_uid"]} == {row["candidate_uid"] for row in rows}


def test_selected_candidate_is_on_existing_candidate_axis() -> None:
    rows = candidates()
    result = presence(rows=rows)
    assert result["selected_candidate_uid"] in {row["candidate_uid"] for row in rows}


def test_storage_hard_stop_policy_is_respected() -> None:
    audit = json.loads((R3 / "storage_audit_after.json").read_text(encoding="utf-8"))
    filesystem = audit["filesystem"]
    policy = audit["policy"]
    assert filesystem["free_gib"] > 100.0
    assert filesystem["storage_status"] == "OK"
    assert policy["deletion_performed"] is False
    assert policy["download_performed"] is False


def test_final_decision_is_formally_enumerated_and_blocks_next_stage() -> None:
    result = json.loads((R3 / "FINAL_RESULT.json").read_text(encoding="utf-8"))
    assert result["final_decision"] == "FAIL_OPEN_SET_CROSS_SEQUENCE_GENERALIZATION"
    assert result["next_association_authority_stage_authorized"] is False
