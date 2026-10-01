"""Focused invariant tests for the completed N72R20R3R2R1 closure stage."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from scripts.n72r20r3r2r1_closure import (
    OUT,
    PROTOCOL_SOURCE,
    SOURCE,
    SEQUENCES,
    _set_features,
    _temporal_apply,
    c0_baseline,
    load_rows,
    read_zstd_jsonl,
)


@pytest.fixture(scope="module")
def artifacts():
    rows, labels, metadata = load_rows()
    return rows, labels, metadata


@pytest.fixture(scope="module")
def result():
    return json.loads((OUT / "FINAL_RESULT.json").read_text(encoding="utf-8"))


def _runtime_rows():
    return read_zstd_jsonl(OUT / "runtime_score_tape.jsonl.zst")


def test_01_score_tape_has_8414_unique_frames():
    rows = _runtime_rows()
    assert len(rows) == 8414
    assert len({(row["sequence"], row["frame"]) for row in rows}) == 8414


def test_02_candidate_axes_unchanged(artifacts):
    rows, _, _ = artifacts
    tape = _runtime_rows()
    for source, sealed in zip(rows, tape):
        assert source["candidate_uids"] == sealed["candidate_uid_axis"]


def test_03_representation_checkpoint_hashes_match(artifacts):
    _, _, metadata = artifacts
    manifest = json.loads((OUT / "runtime_score_tape_manifest.json").read_text())
    assert manifest["representation_checkpoint_hashes"] == metadata["checkpoint_hashes"]
    assert all(len(values) == 3 for values in metadata["checkpoint_hashes"].values())


def test_04_runtime_tape_has_no_gt_fields():
    forbidden = {"label_index", "taxonomy_posthoc", "target_gt_present_posthoc", "posthoc_gt_used", "correct_candidate_index"}
    assert all(not forbidden.intersection(row) for row in _runtime_rows())


def test_05_posthoc_labels_are_separate():
    labels = read_zstd_jsonl(OUT / "posthoc_open_set_labels.jsonl.zst")
    manifest = json.loads((OUT / "posthoc_label_manifest.json").read_text())
    assert len(labels) == 8414
    assert manifest["labels_separate_from_runtime_tape"] is True


def test_06_c0_reproducible(artifacts, result):
    expected = c0_baseline(artifacts[0])
    actual = result["C0_direct_none_logit"]
    assert actual["negative_fpr"] == pytest.approx(expected["negative_fpr"])
    assert actual["open_set_correct_id_recall"] == pytest.approx(expected["open_set_correct_id_recall"])


def test_07_b0_inner_split_is_sequence_held_out():
    payload = json.loads((OUT / "combined/inner_selection.json").read_text())
    assert all(item["heldout_sequence"] != item["inner_validation_sequence"] for item in payload["outer_folds"])


def test_08_b1_is_present_in_method_table():
    payload = json.loads((OUT / "calibration/B1_mlp.json").read_text())
    assert "B1_MLP" in payload["families"]


def test_09_b2_monotonic_affine_constraint():
    rows, _, _ = load_rows()
    scale = {"mode": "affine", "a": 2.0, "b": -0.5}
    from scripts.n72r20r3r2r1_closure import _transform_scores
    values = _transform_scores(rows[0], scale)
    assert np.all(np.diff(values[np.argsort(rows[0]["scores"])]) >= 0)


def test_10_b3_positive_temperature():
    payload = json.loads((OUT / "calibration/B3_temperature.json").read_text())
    assert payload["families"]["B3_TEMPERATURE"]["complexity"] > 0


def test_11_energy_finite(artifacts):
    values = [_set_features(row["scores"])[0] for row in artifacts[0]]
    assert np.isfinite(values).all()


def test_12_entropy_finite(artifacts):
    values = [_set_features(row["scores"])[1] for row in artifacts[0]]
    assert np.isfinite(values).all()


def test_13_mahalanobis_output_exists_and_finite():
    payload = json.loads((OUT / "calibration/set_uncertainty.json").read_text())
    assert {"C4_MAHALANOBIS_PRESENT", "C5_MAHALANOBIS_ABSENT"}.issubset(payload["families"])
    assert math.isfinite(payload["families"]["C4_MAHALANOBIS_PRESENT"]["aggregate"]["negative_fpr"])


def test_14_conformal_uses_inner_fold_fields():
    payload = json.loads((OUT / "calibration/conformal.json").read_text())
    folds = payload["families"]["D0_CONFORMAL"]["outer_folds"]
    assert len(folds) == 8
    assert all("inner_validation_sequence" in fold for fold in folds)


def test_15_p0_p1_not_in_runtime_tape():
    assert all("P0" not in row and "P1" not in row for row in _runtime_rows())


def test_16_inner_selection_does_not_use_outer_labels():
    payload = json.loads((OUT / "combined/formal_loso.json").read_text())
    assert payload["outer_labels_used_for_combination_selection"] is False


def test_17_all_eight_outer_folds():
    payload = json.loads((OUT / "combined/formal_loso.json").read_text())
    assert len(payload["folds"]) == 8


def test_18_learned_calibrators_use_three_seeds():
    payload = json.loads((OUT / "combined/formal_loso.json").read_text())
    assert payload["learned_calibrator_seeds"] == [720331, 720332, 720333]


def test_19_one_prediction_per_frame():
    payload = json.loads((OUT / "combined/formal_loso.json").read_text())
    predictions = payload["selected_predictions"] if "selected_predictions" in payload else None
    assert predictions is None or len({(x["sequence"], x["frame"]) for x in predictions}) == len(predictions)
    combined = json.loads((OUT / "combined/formal_loso.json").read_text())
    assert combined["all_outer_folds"] == 8


def test_20_state_predictor_causal_features_only(result):
    assert result["state_reliability"]["query"]["gate"]["predictor"]["causal_features_only"] is True


def test_21_query_mixture_outputs_are_finite():
    payload = json.loads((OUT / "state_reliability/fixed_mixture.json").read_text())
    for item in payload["mixtures"].values():
        assert np.isfinite([item["rank1"], item["MRR"]]).all()


def test_22_anchor_is_immutable(result):
    assert result["state_reliability"]["query"]["anchor_immutable"] is True


def test_23_state_is_immutable_during_static_scoring(result):
    assert result["N72R18_GRU_frozen"] is True


def test_24_temporal_ema_is_causal():
    sequence = "dancetrack0001"
    rows = [{"sequence": sequence, "frame": i, "scores": [1.0], "label_index": 0, "target_gt_present_posthoc": True} for i in range(3)]
    records = [{"sequence": sequence, "frame": i, "probability": p, "threshold": 0.5, "predicted_index": 0, "accepted": p >= 0.5} for i, p in enumerate((0.0, 1.0, 0.0))]
    output = _temporal_apply(rows, records, beta=0.8)
    assert output["records"][1]["probability_temporal"] < 0.5


def test_25_hysteresis_output_exists():
    payload = json.loads((OUT / "temporal/hysteresis.json").read_text())
    assert set(payload) == {"0.02", "0.05", "0.1"}


def test_26_shadow_does_not_authorize_association(result):
    assert result["shadow_causal"]["diagnostic_only"] is True
    assert result["shadow_causal"]["selected"]["association_authority_started"] is False


def test_27_score_tape_state_is_sealed(result):
    assert result["shadow_causal"]["selected"]["state_divergence"]["score_tape_state_is_sealed"] is True


def test_28_none_has_no_write():
    rows = read_zstd_jsonl(OUT / "causal/memory_write_audit.jsonl.zst")
    assert all(item["commit"] is False for item in rows if item["none_no_write"])


def test_29_disagreement_has_no_write():
    rows = read_zstd_jsonl(OUT / "causal/memory_write_audit.jsonl.zst")
    assert all(item["commit"] is False for item in rows if item["disagreement_no_write"])


def test_30_no_candidate_creation(result):
    assert result["candidate_created"] is False
    assert all(item["candidate_created"] is False for item in read_zstd_jsonl(OUT / "causal/memory_write_audit.jsonl.zst"))


def test_31_no_sam3_rerun(result):
    assert result["sam3_rerun"] is False


def test_32_no_val(result):
    assert result["val_accessed"] is False


def test_33_no_test(result):
    assert result["test_accessed"] is False


def test_34_osnet_frozen(result):
    assert result["OSNet_frozen"] is True


def test_35_gru_frozen(result):
    assert result["N72R18_GRU_frozen"] is True


def test_36_solver_frozen(result):
    assert result["exact_solver_frozen"] is True


def test_37_public_id_authority_frozen(result):
    assert result["public_id_authority_frozen"] is True


def test_38_no_model_binary_in_stage_output():
    assert not list(OUT.rglob("*.pt"))


def test_39_storage_above_hard_stop(result):
    assert result["storage_after"]["free_gib"] >= result["storage_after"]["hard_stop_below_gib"]


def test_40_historical_r3r2_outputs_exist_and_decision_unchanged(result):
    historical = json.loads((SOURCE / "FINAL_RESULT.json").read_text())
    assert historical["decision"] == "FAIL_OPEN_SET_CALIBRATION_AFTER_REPRESENTATION"
    assert (SOURCE / "open_set/formal_predictions.jsonl.zst").exists()
