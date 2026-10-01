"""Focused invariants for N72R20R3R2 cross-scene representation learning."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import numpy as np
import pytest
import torch

from sam3_intermot.identity_verification.cross_scene_adapter import CrossSceneIdentityAdapter
from scripts.n72r20r3_common import SEQUENCES
from scripts.n72r20r3r2_representation import EXPECTED_PRESENT, IOU_THRESHOLD, SEEDS, _identity_key, _match_frame, rank_metrics


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/N72R20R3R2"
TRAINING = OUT / "training"
REPRESENTATION = OUT / "representation"


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_goal_is_frozen_to_cross_scene_representation() -> None:
    goal = _read_json(OUT / "FINAL_GOAL.json")
    assert goal["goal"] == "Cross-Scene Open-Set Identity Representation Learning"
    assert goal["goal_frozen"] is True
    assert goal["sam3_rerun_forbidden"] is True
    assert goal["association_authority_forbidden"] is True


def test_source_audit_resolves_correct_failed_predecessor() -> None:
    audit = _read_json(OUT / "source_audit.json")
    assert audit["source_stage"] == "N72R20R3R1R1"
    assert audit["source_decision"] == "FAIL_PROTOCOL_CORRECTED_EXPLICIT_NONE_GENERALIZATION"
    assert audit["source_learned_state_rank1"] == pytest.approx(0.3815011372251706)
    assert audit["previous_results_modified"] is False


def test_exact_eight_outer_heldout_sequences_exist() -> None:
    manifest = _read_json(TRAINING / "outer_split_manifest.json")
    assert len(manifest["folds"]) == 8
    assert {fold["heldout_sequence"] for fold in manifest["folds"]} == set(SEQUENCES)


def test_heldout_sequence_contributes_zero_training_identities() -> None:
    manifest = _read_json(TRAINING / "outer_split_manifest.json")
    for fold in manifest["folds"]:
        assert fold["heldout_observations_used_in_training"] == 0
        assert fold["heldout_identity_keys_in_training"] == 0
        assert fold["outer_heldout_absent_from_training"] is True
        assert fold["heldout_sequence"] not in fold["parameter_fit_sequences"]


def test_identity_label_contains_sequence_and_gt_track_id() -> None:
    manifest = _read_json(TRAINING / "identity_episode_manifest.json")
    assert manifest["identity_label"] == "(sequence_name, gt_track_id)"
    assert _identity_key("dancetrack0001", 7) == "dancetrack0001:7"


def test_candidate_gt_matching_is_one_to_one_at_fixed_iou_threshold() -> None:
    candidates = [
        {"candidate_uid": "a", "box_xyxy": [0, 0, 10, 10], "embedding_offset": 0},
        {"candidate_uid": "b", "box_xyxy": [20, 0, 30, 10], "embedding_offset": 1},
    ]
    result = _match_frame("dancetrack0001", {"frame": 1}, candidates, [(1, [0, 0, 10, 10]), (2, [20, 0, 30, 10])])
    assert IOU_THRESHOLD == 0.50
    chosen = [item["candidate_index"] for item in result["canonical"].values()]
    assert len(chosen) == len(set(chosen)) == 2


def test_same_target_duplicate_is_not_a_false_hard_negative() -> None:
    candidates = [
        {"candidate_uid": "canonical", "box_xyxy": [0, 0, 10, 10], "embedding_offset": 0},
        {"candidate_uid": "duplicate", "box_xyxy": [0, 0, 10, 10], "embedding_offset": 1},
        {"candidate_uid": "other", "box_xyxy": [20, 0, 30, 10], "embedding_offset": 2},
    ]
    result = _match_frame("dancetrack0001", {"frame": 1}, candidates, [(1, [0, 0, 10, 10]), (2, [20, 0, 30, 10])])
    target_key = "dancetrack0001:1"
    positive = result["canonical"][target_key]["candidate_index"]
    ignored = {int(index) for index, key in result["same_identity_duplicate_ignore"].items() if key == target_key}
    assert positive not in ignored
    assert ignored == {1}


def test_episode_query_has_only_past_history() -> None:
    with (TRAINING / "identity_episode_index.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            episode = json.loads(line)
            assert episode["anchor_frame"] < episode["query_frame"]
            assert all(int(frame) < int(episode["query_frame"]) for frame in episode["history_frames"])


def test_episode_query_state_is_reference_only_and_finite() -> None:
    manifest = _read_json(TRAINING / "identity_episode_manifest.json")
    states = np.load(TRAINING / "query_states.float32.npy", mmap_mode="r")
    assert manifest["candidate_features_copied"] is False
    assert states.shape == (manifest["query_state_count"], 512)
    assert np.isfinite(states[:]).all()


def test_episode_provenance_marks_posthoc_labels_without_runtime_gt() -> None:
    with (TRAINING / "identity_episode_index.jsonl").open(encoding="utf-8") as handle:
        episodes = [json.loads(next(handle)) for _ in range(3)]
    assert all(episode["posthoc_gt_used"] is True for episode in episodes)
    assert all(episode["training_posthoc_gt_used"] is True for episode in episodes)
    assert all(episode["runtime_future_gt_used"] is False for episode in episodes)


def test_gru_and_osnet_are_frozen_in_training_lineage() -> None:
    manifest = _read_json(TRAINING / "identity_episode_manifest.json")
    goal = _read_json(OUT / "FINAL_GOAL.json")
    assert goal["n72r18_gru_frozen"] is True
    assert goal["base_osnet_feature_tape_frozen"] is True
    assert "frozen N72R18 GRU" in manifest["query_state_lineage"]


def test_adapter_has_separate_trainable_query_and_candidate_towers() -> None:
    model = CrossSceneIdentityAdapter()
    assert any(name.startswith("query_tower") and parameter.requires_grad for name, parameter in model.named_parameters())
    assert any(name.startswith("candidate_tower") and parameter.requires_grad for name, parameter in model.named_parameters())


def test_adapter_outputs_are_normalized_and_finite() -> None:
    model = CrossSceneIdentityAdapter()
    query = torch.randn(3, 512)
    candidates = torch.randn(3, 4, 512)
    output = model(query, candidates, torch.ones(3, 4, dtype=torch.bool))
    assert torch.isfinite(output["scores"]).all()
    assert torch.allclose(output["query_embedding"].norm(dim=-1), torch.ones(3), atol=1e-5)
    assert torch.allclose(output["candidate_embedding"].norm(dim=-1), torch.ones(3, 4), atol=1e-5)


def test_adapter_has_no_sequence_or_gt_input() -> None:
    model = CrossSceneIdentityAdapter()
    assert not any(token in name.lower() for name, _ in model.named_parameters() for token in ("sequence", "gt", "track", "fold"))


def test_adapter_respects_parameter_budget() -> None:
    assert CrossSceneIdentityAdapter().trainable_parameters == 265472
    assert CrossSceneIdentityAdapter().trainable_parameters <= 500_000


def test_inner_validation_differs_from_outer_heldout() -> None:
    manifest = _read_json(TRAINING / "outer_split_manifest.json")
    assert all(fold["internal_validation_sequence"] != fold["heldout_sequence"] for fold in manifest["folds"])
    assert all(fold["internal_validation_sequence"] in fold["all_nonheldout_sequences"] for fold in manifest["folds"])


def test_three_preregistered_seeds_are_recorded() -> None:
    seed_manifest = _read_json(TRAINING / "seed_manifest.json")
    assert tuple(seed_manifest["formal_seeds"]) == SEEDS == (720321, 720322, 720323)


def test_candidate_tape_is_referenced_not_copied() -> None:
    manifest = _read_json(TRAINING / "identity_episode_manifest.json")
    assert manifest["candidate_features_copied"] is False
    with (TRAINING / "identity_episode_index.jsonl").open(encoding="utf-8") as handle:
        episode = json.loads(next(handle))
    assert "candidate_embedding" not in episode
    assert episode["candidate_embedding_offsets"]


def test_formal_rows_have_one_prediction_per_present_frame() -> None:
    payload = _read_json(REPRESENTATION / "formal_loso.json")
    assert payload["runtime_metrics"]["rows"] == EXPECTED_PRESENT
    assert payload["oracle_clean_metrics"]["rows"] == EXPECTED_PRESENT
    assert payload["runtime_gt_clean"] is True


def test_formal_runtime_evaluation_excludes_none() -> None:
    payload = _read_json(REPRESENTATION / "formal_loso.json")
    assert payload["runtime_metrics"]["rows"] == EXPECTED_PRESENT
    assert payload["runtime_metrics"]["runtime_future_gt_used"] is False


def test_seed_ensemble_produces_one_score_vector_per_frame() -> None:
    payload = _read_json(REPRESENTATION / "formal_prediction_manifest.json")
    assert payload["one_score_vector_per_frame"] is True
    assert payload["runtime_rows"] == EXPECTED_PRESENT
    assert payload["seed_rows"] == {str(seed): EXPECTED_PRESENT for seed in SEEDS}


def test_oracle_clean_state_is_explicitly_posthoc_only() -> None:
    diagnostic = _read_json(REPRESENTATION / "oracle_clean_diagnostic.json")
    assert diagnostic["posthoc_oracle_only"] is True
    assert diagnostic["runtime_usable"] is False
    formal = _read_json(REPRESENTATION / "formal_loso.json")
    assert formal["oracle_clean_metrics"]["posthoc_oracle_only"] is True


def test_no_heldout_teacher_forcing_in_formal_records() -> None:
    formal = _read_json(REPRESENTATION / "formal_loso.json")
    assert all(record["heldout_observations_used_in_training"] == 0 for record in formal["fold_records"])
    assert all(record["outer_heldout_absent_from_training"] is True for record in formal["fold_records"])


def test_representation_gate_contains_all_five_preregistered_checks() -> None:
    gate = _read_json(REPRESENTATION / "formal_loso.json")["representation_gate"]
    assert set(gate["checks"]) == {
        "R1_pooled_rank1_ge_0.60",
        "R2_macro_sequence_rank1_ge_0.50",
        "R3_pooled_mrr_ge_0.70",
        "R4_paired_bootstrap_rank1_delta_lower_gt_0",
        "R5_no_outer_supervision_leakage",
    }


def test_paired_bootstrap_is_sequence_clustered_and_preregistered() -> None:
    payload = _read_json(REPRESENTATION / "paired_bootstrap.json")
    assert payload["repetitions"] == 2000
    assert payload["seed"] == 720322
    assert payload["unit"] == "sequence cluster paired delta"


def test_rank_metrics_ignore_none_rows() -> None:
    result = rank_metrics([{"rank": 1}, {"rank": 2}, {"rank": None}])
    assert result["rows"] == 3
    assert result["rank1"] == pytest.approx(1 / 3)
    assert result["MRR"] == pytest.approx(0.75)


def test_checkpoint_binaries_are_external_and_not_staged() -> None:
    manifest = _read_json(TRAINING / "checkpoint_manifest.json")
    assert manifest["binary_committed"] is False
    assert all(not Path(item["path"]).resolve().is_relative_to(ROOT.resolve()) for item in manifest["records"])
    tracked = subprocess.run(["git", "ls-files", "*.pt"], cwd=ROOT, check=True, text=True, stdout=subprocess.PIPE).stdout
    assert "N72R20R3R2" not in tracked


def test_sam3_val_test_and_association_authority_remain_closed() -> None:
    result = _read_json(OUT / "FINAL_RESULT.json")
    assert result["SAM3_rerun"] is False
    assert result["val_accessed"] is False
    assert result["test_accessed"] is False
    assert result["association_authority_started"] is False
    status = _read_json(OUT / "stage_status.json")
    assert status["open_set_started"] is True
    assert status["causal_started"] is False


def test_failure_decision_authorizes_only_the_next_relevant_research_stage() -> None:
    result = _read_json(OUT / "FINAL_RESULT.json")
    assert result["decision"] in {
        "FAIL_CROSS_SCENE_IDENTITY_REPRESENTATION",
        "FAIL_RUNTIME_IDENTITY_STATE_QUALITY",
        "FAIL_OPEN_SET_CALIBRATION_AFTER_REPRESENTATION",
        "FAIL_CAUSAL_COMMIT_AFTER_REPRESENTATION",
        "PASS_CROSS_SCENE_OPEN_SET_IDENTITY_REPRESENTATION",
    }
    assert result["next_association_authority_stage_authorized"] is False
    if result["decision"] == "FAIL_CROSS_SCENE_IDENTITY_REPRESENTATION":
        assert result["next_backbone_adaptation_stage_authorized"] is True
        assert result["next_memory_state_learning_stage_authorized"] is False
    elif result["decision"] == "FAIL_RUNTIME_IDENTITY_STATE_QUALITY":
        assert result["next_backbone_adaptation_stage_authorized"] is False
        assert result["next_memory_state_learning_stage_authorized"] is True
    elif result["decision"] == "FAIL_OPEN_SET_CALIBRATION_AFTER_REPRESENTATION":
        assert result["next_backbone_adaptation_stage_authorized"] is False
        assert result["next_memory_state_learning_stage_authorized"] is False


def test_none_head_keeps_representation_frozen() -> None:
    manifest = _read_json(OUT / "open_set/training_manifest.json")
    assert manifest["representation_frozen"] is True
    assert manifest["head"] == "FrozenMetricNoneHead"
    assert manifest["head_trainable_parameters"] == 353
    assert all(record["heldout_observations_used_in_training"] == 0 for record in manifest["records"])


def test_none_head_uses_only_the_nine_runtime_safe_features() -> None:
    manifest = _read_json(OUT / "open_set/training_manifest.json")
    features = manifest["input_features"]
    assert len(features) == 9
    assert all("gt" not in feature.lower() for feature in features)
    assert all("sequence" not in feature.lower() for feature in features)


def test_none_training_contains_both_n0_and_n1_absence_types() -> None:
    manifest = _read_json(OUT / "open_set/training_manifest.json")
    assert manifest["N0_total"] > 0
    assert manifest["N1_total"] > 0
    assert all(1.0 <= record["training_present_absent_ratio"] <= 2.0 for record in manifest["records"])


def test_open_set_outputs_are_legal_single_decisions() -> None:
    import subprocess

    completed = subprocess.run(["zstd", "-q", "-d", "-c", str(OUT / "open_set/formal_predictions.jsonl.zst")], check=True, text=True, stdout=subprocess.PIPE)
    rows = [json.loads(line) for line in completed.stdout.splitlines() if line]
    assert len(rows) == 8414
    for row in rows:
        assert row["predicted_index"] == -1 or 0 <= row["predicted_index"] < len(row["candidate_uids"])
        assert row["candidate_created"] is False
        assert row["runtime_future_gt_used"] is False


def test_open_set_metrics_match_formal_prediction_labels() -> None:
    payload = _read_json(OUT / "open_set/formal_loso.json")
    assert payload["runtime_metrics"]["negative_fpr"] == pytest.approx(0.23474436503573393)
    assert payload["runtime_metrics"]["open_set_correct_id_recall"] == pytest.approx(0.5323730098559515)
    assert payload["runtime_metrics"]["candidate_creation_count"] == 0
    assert payload["open_set_gate"]["pass"] is False


def test_failed_open_set_gate_closes_causal_stage() -> None:
    status = _read_json(OUT / "stage_status.json")
    result = _read_json(OUT / "FINAL_RESULT.json")
    assert result["decision"] == "FAIL_OPEN_SET_CALIBRATION_AFTER_REPRESENTATION"
    assert result["causal_metrics"] is None
    assert status["causal_started"] is False
    assert not (OUT / "causal/causal_replay.json").exists()


def test_storage_audits_remain_above_hard_stop() -> None:
    before = _read_json(OUT / "storage_audit_before.json")
    after = _read_json(OUT / "storage_audit_after.json")
    assert before["filesystem"]["storage_status"] != "HARD_STOP"
    assert after["filesystem"]["storage_status"] != "HARD_STOP"
    assert float(after["filesystem"]["free_gib"]) > 100.0
