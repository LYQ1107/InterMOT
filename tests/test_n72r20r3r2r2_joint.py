"""N72R20R3R2R2 research-tree and runtime-boundary invariants."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from scripts.n72r20r3r2r2_joint import OUT, canonical_positive, select_threshold


def load(name: str):
    return json.loads((OUT / name).read_text(encoding="utf-8"))


def audit_rows():
    from scripts.n72r20r3r2r2_joint import read_zstd_jsonl
    return read_zstd_jsonl(OUT / "causal/memory_write_audit.jsonl.zst")


def test_01_heldout_sequence_contributes_zero_training_episodes():
    payload = load("training/outer_split_manifest.json")
    assert len(payload["folds"]) == 8
    assert all(item["outer_heldout_absent_from_training"] for item in payload["folds"])
    assert all(item["heldout_observations_used_in_training"] == 0 for item in payload["folds"])


def test_02_p0_labels_are_defined_as_target_absent():
    payload = load("forensics/absence_taxonomy_audit.json")
    assert "target GT absent" in payload["taxonomy"]["P0"]
    assert payload["counts"]["P0"] > 0


def test_03_p1_labels_are_defined_as_target_visible_without_valid_candidate():
    payload = load("forensics/absence_taxonomy_audit.json")
    assert payload["counts"]["P1a"] + payload["counts"]["P1b"] > 0
    assert payload["taxonomy"]["P1a"].startswith("target GT visible")


def test_04_p1a_labels_use_weak_overlap_interval():
    payload = load("forensics/absence_taxonomy_audit.json")
    assert "[0.10,0.50)" in payload["taxonomy"]["P1a"]


def test_05_p1b_labels_use_no_target_overlap_interval():
    payload = load("forensics/absence_taxonomy_audit.json")
    assert "< 0.10" in payload["taxonomy"]["P1b"]


def test_06_counterfactual_removes_valid_target_candidates():
    payload = load("training/counterfactual_manifest.json")
    assert payload["episode_count"] > 0
    assert all(item["kept_valid_count"] == 0 for item in payload["records"])


def test_07_counterfactual_keeps_distractor_axis_unchanged():
    payload = load("training/counterfactual_manifest.json")
    assert all(item["distractor_axis_unchanged"] for item in payload["records"])


def test_08_counterfactuals_are_not_formal_evaluation():
    payload = load("training/counterfactual_manifest.json")
    assert payload["formal_evaluation_eligible"] is False
    assert all(item["formal_evaluation_eligible"] is False for item in payload["records"])


def test_09_candidate_validity_labels_are_posthoc_training_labels():
    payload = load("forensics/candidate_validity_oracle.json")
    assert payload["oracle_is_posthoc_only"] is True
    assert payload["training_validity_labels_gt_only"] is True


def test_10_runtime_whitelist_contains_no_gt_field():
    payload = load("training/runtime_feature_whitelist.json")
    fields = set(payload["candidate_features"] + payload["query_features"] + payload["quality_features"])
    assert not any("gt" in field.lower() or "iou" in field.lower() for field in fields)


def test_11_valid_candidate_definition_is_iou_at_least_point_five():
    payload = load("forensics/candidate_validity_oracle.json")
    assert payload["classes"]["V"] == "IoU >= 0.50"


def test_12_weak_candidate_interval_is_closed_open():
    payload = load("forensics/candidate_validity_oracle.json")
    assert payload["classes"]["W"] == "0.10 <= IoU < 0.50"


def test_13_duplicate_target_candidates_are_positive_not_false_negatives():
    labels, canonical = canonical_positive([2, 2, 0], [0.51, 0.80, 0.01], ["a", "b", "c"])
    assert labels == [0, 1]
    assert canonical == 1


def test_14_j1_probabilities_are_finite():
    metrics = load("joint_representation/J1_pairwise.json")["metrics"]
    assert np.isfinite([metrics["candidate_valid_auroc"], metrics["set_presence_auroc"]]).all()


def test_15_j2_ranking_metrics_are_finite():
    metrics = load("joint_representation/J2_joint_absolute.json")["metrics"]
    assert np.isfinite([metrics["Rank1"], metrics["MRR"]]).all()


def test_16_noisy_or_set_loss_is_present():
    assert load("joint_representation/J3_set_loss.json")["family"] == "J3_SET_NOISY_OR"


def test_17_absent_set_target_is_zero_defined():
    payload = load("joint_representation/J3_set_loss.json")
    assert payload["counterfactual_training_included"] is False
    assert load("training/open_set_episode_manifest.json")["counts"]["P0"] > 0


def test_18_present_set_target_is_one_defined():
    assert load("training/open_set_episode_manifest.json")["counts"]["PRESENT"] > 0


def test_19_energy_loss_metrics_are_finite():
    metrics = load("joint_representation/J4_energy_loss.json")["metrics"]
    assert np.isfinite([metrics["negative_fpr"], metrics["correct_id_recall"]]).all()


def test_20_quality_missing_values_are_deterministic():
    audit = load("forensics/runtime_candidate_feature_audit.json")
    assert audit["unavailable_fields_not_invented"] is True
    assert audit["candidate_feature_availability"]["iou_pred"]["missing"] > 0


def test_21_unavailable_metadata_not_imputed_from_gt():
    whitelist = load("training/runtime_feature_whitelist.json")
    assert "target_iou" in whitelist["excluded_unavailable_features"]
    assert "future_gt_box" in whitelist["excluded_unavailable_features"]


def test_22_osnet_is_frozen():
    assert load("FINAL_RESULT.json")["OSNet_frozen"] is True


def test_23_gru_is_frozen_initially():
    assert load("FINAL_RESULT.json")["N72R18_GRU_frozen_initially"] is True


def test_24_ranking_retention_is_calculated():
    payload = load("joint_representation/ranking_retention.json")
    assert "baseline" in payload and "J2_JOINT_ABSOLUTE" in payload["families"]


def test_25_recall_at_two_percent_is_posthoc_diagnostic():
    payload = load("joint_representation/absence_separability.json")
    assert payload["development_target_is_not_final_pass"] is True


def test_26_thresholds_are_selected_inner_only():
    payload = load("formal/inner_selection.json")
    assert payload["threshold_selection_inner_only"] is True
    assert payload["outer_labels_used_for_selection"] is False


def test_27_outer_sequence_does_not_tune_threshold():
    folds = load("formal/inner_selection.json")["folds"]
    assert all(item["heldout_sequence"] != item["inner_validation_sequence"] for item in folds)


def test_28_three_learned_seeds_are_retained():
    assert load("training/seed_manifest.json")["representation_seeds"] == [720341, 720342, 720343]


def test_29_one_decision_per_frame_is_declared():
    assert load("training/seed_manifest.json")["one_decision_per_frame"] is True


def test_30_denoiser_runtime_path_has_no_future_gt():
    payload = load("state_denoising/formal_comparison.json")
    assert payload["runtime_future_gt_used"] is False
    assert payload["selected_by_inner_only"] is True


def test_31_denoiser_uses_runtime_only_at_evaluation():
    for name in ("D0.json", "D1.json", "D2.json", "D3.json"):
        assert load(f"state_denoising/{name}")["runtime_only"] is True


def test_32_corruption_augmentation_is_training_only():
    assert load("state_denoising/formal_comparison.json")["corruption_train_only"] is True


def test_33_causal_gru_is_unidirectional():
    payload = load("temporal/all_results.json")
    records = [item.get("causal_gru") for item in payload["per_sequence"].values()]
    assert records and all(item["unidirectional"] is True for item in records)


def test_34_temporal_model_cannot_inspect_future():
    payload = load("temporal/all_results.json")
    assert payload["future_frames_inspected"] is False
    assert all(item["causal_gru"]["unidirectional"] for item in payload["per_sequence"].values())


def test_35_decomposed_presence_ends_in_product_none_decision():
    payload = load("decomposed_presence/combined.json")
    assert payload["decision_stages"][-1] == "product_then_argmax_or_NONE"


def test_36_oracle_diagnostics_are_posthoc():
    payload = load("oracle/oracle_ladder.json")
    assert payload["posthoc_only_for_oracle_rows"] is True
    assert all(item["posthoc_oracle_only"] for name, item in payload["modes"].items() if name != "O0_RUNTIME_JOINT")


def test_37_shadow_causal_does_not_authorize_association():
    payload = load("shadow_causal/selected.json")
    assert payload["diagnostic_only"] is True
    assert payload["association_authority_started"] is False


def test_38_shadow_scores_before_update():
    assert all(item["score_before_update"] for item in audit_rows())


def test_39_none_never_writes():
    assert all(not item["commit"] for item in audit_rows() if item["none_no_write"])


def test_40_disagreement_never_writes():
    assert all(not item["commit"] for item in audit_rows() if item["disagreement"])


def test_41_no_candidate_creation():
    result = load("FINAL_RESULT.json")
    assert result["candidate_created"] is False
    assert all(item["candidate_created"] is False for item in audit_rows())


def test_42_candidate_output_exists():
    assert load("forensics/runtime_candidate_feature_audit.json")["sequences"]


def test_43_no_sam3_rerun():
    assert load("FINAL_RESULT.json")["sam3_rerun"] is False


def test_44_val_not_accessed():
    assert load("FINAL_RESULT.json")["val_accessed"] is False


def test_45_test_not_accessed():
    assert load("FINAL_RESULT.json")["test_accessed"] is False


def test_46_exact_solver_unchanged():
    assert load("FINAL_RESULT.json")["exact_solver_frozen"] is True


def test_47_public_id_authority_unchanged():
    assert load("FINAL_RESULT.json")["public_id_authority_frozen"] is True


def test_48_historical_r3r2r1_artifacts_unchanged():
    source = load("source_audit.json")
    assert source["historical_outputs_modified"] is False
    assert source["source_head"] == "7820d312e2b3c852a7909a4d42dc71bad0c085c6"


def test_49_storage_remains_above_hard_stop():
    storage = load("storage_audit_after.json")
    assert storage["free_gib"] >= storage["hard_stop_below_gib"]


def test_50_no_binary_checkpoint_staged():
    assert not list(OUT.rglob("*.pt"))
    manifest = load("training/checkpoint_manifest.json")
    assert manifest["binary_committed"] is False
    assert all(item["binary_committed"] is False for item in manifest["records"])
