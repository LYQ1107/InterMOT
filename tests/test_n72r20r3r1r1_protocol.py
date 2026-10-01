"""Focused regression tests for the N72R20R3R1R1 protocol repair."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import numpy as np
import pytest
import torch

from sam3_intermot.identity_verification.explicit_none_verifier import ExplicitNoneVerifier
from scripts.n72r20r3r1r1_protocol import (
    EXPECTED_SEQUENCE_COUNTS,
    OUT,
    SEEDS,
    SEQUENCES,
)


ROOT = Path(__file__).resolve().parents[1]


def _read_json(path: str) -> dict:
    return json.loads((OUT / path).read_text())


def _read_zstd(path: Path) -> list[dict]:
    result = subprocess.run(["zstd", "-q", "-d", "-c", str(path)], check=True, stdout=subprocess.PIPE)
    return [json.loads(line) for line in result.stdout.decode().splitlines()]


def test_goal_is_frozen_and_primary_is_preregistered() -> None:
    goal = _read_json("FINAL_GOAL.json")
    assert goal["stage"] == "N72R20R3R1R1"
    assert goal["goal"] == "Protocol-Corrected Explicit-NONE Identity Verification"
    assert goal["one_real_frame_one_decision"] is True
    assert goal["primary_model_pre_frozen"] == "V2_DUAL_STATE"
    assert goal["goal_frozen"] is True
    assert goal["sam3_rerun_forbidden"] is True
    assert goal["dance_track_val_forbidden"] is True
    assert goal["dance_track_test_forbidden"] is True


def test_canonical_index_has_exact_frame_and_label_counts() -> None:
    manifest = _read_json("training/canonical_training_index_manifest.json")
    assert manifest["rows"] == 8414
    assert manifest["unique_sequence_frame_keys"] == 8414
    assert manifest["label_counts"] == {"CANDIDATE": 6595, "NONE": 1819}
    assert manifest["state_condition_dimension"] is False
    assert manifest["candidate_features_copied"] is False


def test_canonical_rows_have_unique_keys_and_no_state_condition() -> None:
    keys = []
    with (OUT / "training/canonical_training_index.jsonl").open() as handle:
        for line in handle:
            row = json.loads(line)
            keys.append((row["sequence"], row["frame"]))
            assert "state_condition" not in row
            assert "human_anchor_ref" in row
            assert "causal_learned_state_ref" in row
            assert row["runtime_future_gt_used"] is False
            assert row["posthoc_gt_used"] is True
    assert len(keys) == 8414
    assert len(set(keys)) == 8414


@pytest.mark.parametrize("sequence,expected", sorted(EXPECTED_SEQUENCE_COUNTS.items()))
def test_canonical_sequence_count(sequence: str, expected: int) -> None:
    count = 0
    with (OUT / "training/canonical_training_index.jsonl").open() as handle:
        for line in handle:
            if json.loads(line)["sequence"] == sequence:
                count += 1
    assert count == expected


def test_canonical_state_lineage_is_pre_update() -> None:
    first = json.loads((OUT / "training/canonical_training_index.jsonl").read_text().splitlines()[0])
    lineage = first["state_lineage"]
    assert lineage["source_state_condition"] == "S1_R2_STYLE_CAUSAL_LEARNED_STATE"
    assert lineage["causal_order"] == "state_before_current_frame_score_then_optional_future_update"
    assert lineage["reconstructed_provenance_only"] is True


def test_split_manifest_has_eight_outer_folds_and_no_frame_split() -> None:
    payload = _read_json("training/split_manifest.json")
    assert len(payload["folds"]) == 8
    assert all(item["frame_random_split"] is False for item in payload["folds"])
    assert all(item["outer_heldout_absent_from_fit"] for item in payload["folds"])
    assert all(item["outer_heldout_absent_from_validation"] for item in payload["folds"])


def test_inner_validation_is_cyclic_and_differs_from_outer() -> None:
    payload = _read_json("training/inner_validation_rule.json")
    assert payload["frozen_before_formal_evaluation"] is True
    assert [item["validation"] for item in payload["folds"]] == [
        "dancetrack0002", "dancetrack0023", "dancetrack0024", "dancetrack0039",
        "dancetrack0057", "dancetrack0062", "dancetrack0072", "dancetrack0001",
    ]
    assert all(item["heldout"] != item["validation"] for item in payload["folds"])


def test_architecture_selection_is_disabled_and_primary_fixed() -> None:
    selection = _read_json("training/model_selection.json")
    assert selection["primary_model"] == "V2_DUAL_STATE"
    assert selection["formal_architecture_selection"] is False
    assert selection["architecture_selection_from_outer_scores"] is False


def test_seed_manifest_is_preregistered() -> None:
    manifest = _read_json("training/seed_manifest.json")
    assert manifest["formal_seeds"] == list(SEEDS)
    assert manifest["bootstrap_seed"] == 720312


@pytest.mark.parametrize("variant", ["A0_ANCHOR_ONLY", "A1_LEARNED_STATE_ONLY", "A2_DUAL_STATE"])
def test_all_ablation_variants_have_finite_outputs(variant: str) -> None:
    model = ExplicitNoneVerifier(mode="V2_EXPLICIT_NONE", state_variant=variant)
    output = model(torch.randn(2, 512), torch.randn(2, 512), torch.randn(2, 4, 512), torch.ones(2, 4, dtype=torch.bool), torch.randn(2, 4))
    assert torch.isfinite(output["logits"]).all()


def test_a0_does_not_consume_learned_state_or_memory_timing() -> None:
    model = ExplicitNoneVerifier(mode="V2_EXPLICIT_NONE", state_variant="A0_ANCHOR_ONLY")
    anchor = torch.randn(1, 512)
    candidates = torch.randn(1, 3, 512)
    mask = torch.ones(1, 3, dtype=torch.bool)
    first = model(anchor, torch.randn(1, 512), candidates, mask, torch.tensor([[2.0, 0.1, 0.0, 1.0]]))["logits"]
    second = model(anchor, torch.randn(1, 512) * 100.0, candidates, mask, torch.tensor([[2.0, 0.1, 99.0, -0.5]]))["logits"]
    assert torch.allclose(first, second)


def test_a1_does_not_consume_anchor_pairwise_evidence() -> None:
    model = ExplicitNoneVerifier(mode="V2_EXPLICIT_NONE", state_variant="A1_LEARNED_STATE_ONLY")
    state = torch.randn(1, 512)
    candidates = torch.randn(1, 3, 512)
    mask = torch.ones(1, 3, dtype=torch.bool)
    first = model(torch.randn(1, 512), state, candidates, mask, torch.tensor([[2.0, 0.1, 0.5, 1.0]]))["logits"]
    second = model(torch.randn(1, 512) * 100.0, state, candidates, mask, torch.tensor([[2.0, 0.1, 0.5, -0.9]]))["logits"]
    assert torch.allclose(first, second)


def test_a2_receives_both_state_sources() -> None:
    model = ExplicitNoneVerifier(mode="V2_EXPLICIT_NONE", state_variant="A2_DUAL_STATE")
    assert model.state_variant == "A2_DUAL_STATE"
    assert model.pairwise[0].in_features == 2 * (2 * model.projection_dim + 2) + 1


def test_formal_manifest_has_one_prediction_per_frame() -> None:
    manifest = _read_json("static_eval/formal_predictions_manifest.json")
    assert manifest["outer_folds"] == 8
    assert manifest["per_seed_rows"] == 8414
    assert manifest["primary_ensemble_rows"] == 8414
    assert manifest["primary_positive"] == 6595
    assert manifest["primary_negative"] == 1819


def test_primary_predictions_are_unique_and_legal() -> None:
    rows = _read_zstd(OUT / "static_eval/primary_predictions.jsonl.zst")
    assert len(rows) == 8414
    assert len({(row["sequence"], row["frame"]) for row in rows}) == 8414
    assert sum(int(row["label_index"]) >= 0 for row in rows) == 6595
    assert sum(int(row["label_index"]) < 0 for row in rows) == 1819
    assert all(row["runtime_future_gt_used"] is False for row in rows)
    assert all(row["candidate_created"] is False for row in rows)
    assert all(row["predicted_uid"] is None or row["predicted_uid"] in row["candidate_uids"] for row in rows)


def test_three_seed_aggregation_is_not_concatenated() -> None:
    assert _read_json("static_eval/formal_predictions_manifest.json")["primary_ensemble_rows"] == 8414
    assert len(_read_zstd(OUT / "static_eval/primary_predictions.jsonl.zst")) != 3 * 8414


@pytest.mark.parametrize("seed", [720301, 720302, 720303])
def test_each_primary_seed_has_one_unique_prediction_per_frame(seed: int) -> None:
    rows = _read_zstd(OUT / f"static_eval/seed_predictions_V2_DUAL_STATE_seed{seed}.jsonl.zst")
    assert len(rows) == 8414
    assert len({(row["sequence"], row["frame"]) for row in rows}) == 8414


def test_outer_heldout_is_absent_from_fit_and_validation_records() -> None:
    records = _read_json("training/formal_records.json")["records"]
    assert len(records) == 96
    assert all(record["outer_heldout_absent_from_fit"] for record in records)
    assert all(record["outer_heldout_absent_from_validation"] for record in records)
    assert all(record["heldout_sequence"] not in record["parameter_fit_sequences"] for record in records)
    assert all(record["heldout_sequence"] != record["internal_validation_sequence"] for record in records)


def test_v1_threshold_is_calibrated_only_on_inner_validation() -> None:
    records = _read_json("training/formal_records.json")["records"]
    v1 = [record for record in records if record["model"] == "V1_PAIRWISE_THRESHOLD_NONE"]
    assert len(v1) == 24
    assert all(record["v1_threshold_calibration_source"] == "internal_validation_sequence_only" for record in v1)
    assert all(record["architecture_selection_from_outer_scores"] is False for record in records)


def test_canonical_vectors_have_expected_dimensions() -> None:
    anchors = np.load(OUT / "training/human_anchor_vectors.float32.npy", mmap_mode="r")
    states = np.load(OUT / "training/causal_learned_state_vectors.float32.npy", mmap_mode="r")
    assert anchors.shape == (8, 512)
    assert states.shape == (8414, 512)
    assert np.isfinite(anchors).all()
    assert np.isfinite(states).all()


def test_no_forbidden_runtime_stage_was_accessed() -> None:
    manifest = _read_json("static_eval/formal_predictions_manifest.json")
    result = _read_json("FINAL_RESULT.json")
    assert manifest["candidate_generation_rerun"] is False
    assert manifest["sam3_rerun"] is False
    assert manifest["val_accessed"] is False
    assert manifest["test_accessed"] is False
    assert result["association_rescue_run"] is False


def test_exact_solver_file_is_not_modified() -> None:
    changed = subprocess.run(["git", "diff", "--name-only"], cwd=ROOT, check=True, text=True, stdout=subprocess.PIPE).stdout.splitlines()
    assert "sam3_intermot/association/effect_assignment.py" not in changed


def test_checkpoint_asset_budget_and_external_location() -> None:
    manifest = _read_json("training/checkpoint_manifest.json")
    assert all("InterMOT_N72R20R3R1R1_assets/models" in record["path"] for record in manifest["records"])
    assert sum(Path(record["path"]).stat().st_size for record in manifest["records"]) < 3 * 1024**3


def test_runtime_provenance_contract_is_correct() -> None:
    manifest = _read_json("training/canonical_training_index_manifest.json")
    report = _read_json("static_eval/primary_result.json")
    assert manifest["runtime_future_gt_used"] is False
    assert manifest["runtime_gt_clean"] is True
    assert report["metrics"]["runtime_future_gt_used"] is False
    assert report["metrics"]["runtime_gt_clean"] is True
    assert report["static_gate"]["checks"]["S5_runtime_future_gt_false"] is True
    assert report["static_gate"]["checks"]["S5b_runtime_gt_clean"] is True


def test_closed_set_ranking_ignores_none_class() -> None:
    rows = _read_zstd(OUT / "static_eval/primary_predictions.jsonl.zst")
    assert all(row["closed_set_top1_index"] is None or row["closed_set_top1_index"] < row["candidate_count"] for row in rows)
    ranking = _read_json("static_eval/ranking_diagnostics.json")
    assert ranking["closed_set_candidate_top1"] is not None
    assert ranking["closed_set_MRR"] is not None
    assert ranking["top3"] is not None
    assert ranking["top5"] is not None


def test_secondary_rank2_rank3_metrics_are_reported() -> None:
    metrics = _read_json("static_eval/primary_result.json")["metrics"]
    assert metrics["rank_1_identity_accuracy"] == metrics["closed_set_candidate_top1_accuracy"]
    assert metrics["rank_2_identity_accuracy"] >= metrics["rank_1_identity_accuracy"]
    assert metrics["rank_3_identity_accuracy"] >= metrics["rank_2_identity_accuracy"]


def test_oracle_diagnostics_are_posthoc_only() -> None:
    oracle = _read_json("static_eval/oracle_diagnostics.json")
    assert oracle["posthoc_oracle_only"] is True
    assert oracle["runtime_usable"] is False


def test_bootstrap_is_sequence_clustered_and_preregistered() -> None:
    bootstrap = _read_json("static_eval/bootstrap.json")
    assert bootstrap["repetitions"] == 2000
    assert bootstrap["seed"] == 720312
    assert bootstrap["unit"] == "sequence cluster"
    assert {"negative_fpr", "open_recall", "closed_set_top1", "macro_recall"}.issubset(bootstrap["intervals_95"])


def test_static_gate_is_terminal_failure_and_classified() -> None:
    result = _read_json("FINAL_RESULT.json")
    assert result["decision"] == "FAIL_PROTOCOL_CORRECTED_EXPLICIT_NONE_GENERALIZATION"
    assert result["bottleneck_classification"] == "BOTTLENECK_CANDIDATE_IDENTITY_REPRESENTATION"
    assert result["next_representation_learning_stage_authorized"] is True
    assert result["next_association_authority_stage_authorized"] is False


def test_causal_and_association_stages_were_not_run() -> None:
    status = _read_json("stage_status.json")
    causal = _read_json("causal/causal_replay.json")
    headroom = _read_json("future_association_headroom.json")
    assert status["causal_replay"] == "NOT_RUN_STATIC_GATE_FAILED"
    assert causal["causal_commit_replay_authorized"] is False
    assert headroom["association_rescue_run"] is False
    assert headroom["solver_called"] is False


def test_frozen_runtime_boundaries_are_recorded() -> None:
    goal = _read_json("FINAL_GOAL.json")
    result = _read_json("FINAL_RESULT.json")
    for key in ("osnet_frozen", "n72r18_gru_frozen", "exact_solver_frozen", "public_id_authority_frozen"):
        assert goal.get({"osnet_frozen": "osnet_frozen", "n72r18_gru_frozen": "n72r18_gru_frozen", "exact_solver_frozen": "exact_solver_frozen", "public_id_authority_frozen": "public_id_authority_frozen"}[key]) is True
        assert result[key] is True
    assert result["sam3_rerun"] is False
    assert result["val_accessed"] is False
    assert result["test_accessed"] is False


def test_checkpoint_manifest_tracks_hashes_without_commit_binaries() -> None:
    manifest = _read_json("training/checkpoint_manifest.json")
    assert len(manifest["records"]) == 96
    assert all(record["binary_committed"] is False for record in manifest["records"])
    assert all(len(record["sha256"]) == 64 for record in manifest["records"])
    assert all(record["trainable_parameters"] < 200000 for record in manifest["records"])


def test_historical_r3r1_is_immutable_and_source_head_recorded() -> None:
    audit = _read_json("r3r1_immutability_audit.json")
    result = _read_json("FINAL_RESULT.json")
    assert audit["all_unchanged"] is True
    assert audit["historical_r3r1_modified"] is False
    assert result["source_head"] == "e1c294597e53d743128f5ba24d9770e96a5b7be7"
    assert result["historical_reported_final_commit"] == "abbe3779ccbebf749a91fffa3c9c38168ae648e4"


def test_storage_audit_is_above_hard_stop() -> None:
    storage = _read_json("storage_audit_after.json")
    assert storage["filesystem"]["storage_status"] == "OK"
    assert storage["filesystem"]["free_gib"] > 100.0


def test_final_report_answers_the_protocol_question_first() -> None:
    first_word = (OUT / "FINAL_REPORT.md").read_text().split()[0]
    assert first_word in {"YES", "NO"}
    assert first_word == "NO"
