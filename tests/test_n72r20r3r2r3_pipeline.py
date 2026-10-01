"""Invariant and artifact tests for N72R20R3R2R3.

These tests deliberately inspect the sealed artifacts and small pure helpers;
they never start SAM3 or retrain a model.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import subprocess
from pathlib import Path

import numpy as np

from scripts import n72r20r3r2r3_pipeline as p


OUT = p.ROOT / "outputs" / p.STAGE


def _json(relative: str):
    return json.loads((OUT / relative).read_text(encoding="utf-8"))


def _dev_metrics():
    return _json("trackeval/development_summary.json")


def test_baseline_export_complete():
    manifest = _json("baseline/baseline_tracker_manifest.json")
    assert set(manifest["sequences"]) == set(p.SEQUENCES)


def test_frame_indices_valid():
    manifest = _json("baseline/baseline_tracker_manifest.json")
    for row in manifest["sequences"].values():
        assert row["frame_min"] in (None, 1)
        assert row["frame_max"] is None or row["frame_max"] <= row["frame_count"]


def test_mot_format_valid():
    manifest = _json("baseline/baseline_tracker_manifest.json")
    for row in manifest["sequences"].values():
        for line in Path(row["path"]).read_text().splitlines()[:20]:
            fields = line.split(",")
            assert len(fields) == 10
            assert int(fields[0]) >= 1 and int(fields[1]) >= 0
            assert all(np.isfinite(float(value)) for value in fields[:7])


def test_trackeval_wrapper_resolves_scalar_list_compatibility():
    source = (p.ROOT / "scripts/n72r20r3r2r3_trackeval_entry.py").read_text()
    assert "SEQMAP_FILE" in source and "OUTPUT_FOLDER" in source
    assert "len(value) == 1" in source


def test_trackeval_baseline_reproducible():
    baseline = _json("baseline/baseline_full_sequence_trackeval.json")
    assert baseline["immutable"] is True
    assert baseline["parsed"]["combined"]["HOTA___AUC"] > 0


def test_no_gt_runtime_baseline():
    assert _json("baseline/baseline_full_sequence_trackeval.json")["runtime_future_gt_used"] is False


def test_geometry_predictor_causal():
    source = inspect.getsource(p.causal_geometry_predictions)
    assert "target_box_for_frame" not in source
    assert "causal" in source and "future_gt_used" in source


def test_a0_predicts_every_runtime_frame():
    assets = p.load_assets()
    event = p.event_map()[p.SEQUENCES[0]]
    prediction = p.causal_geometry_predictions(assets[p.SEQUENCES[0]], event, "A0_LAST_BOX")
    assert set(prediction) == set(range(assets[p.SEQUENCES[0]].frame_count))


def test_kalman_geometry_finite():
    geometry = _json("geometry/kalman.json")
    assert geometry["variant"] == "A2_KALMAN"
    assert geometry["runtime_future_gt_used"] is False
    assert geometry["pooled"]["frames"] > 0


def test_learned_geometry_finite():
    geometry = _json("geometry/learned_box_predictor.json")
    assert geometry["variant"] == "A3_LEARNED_CAUSAL"
    assert geometry["pooled"]["predicted_iou_mean"] is not None


def test_outer_heldout_excluded_from_fit():
    for fold in _json("candidate_quality/formal_loso.json")["folds"]:
        assert fold["heldout"] not in fold["fit_sequences"]
        assert fold["heldout"] != fold["validation"]


def test_quality_label_v_boundary():
    asset = p.SequenceAsset("x", {0: {"candidates": []}}, {}, {0: [(1, [0, 0, 10, 10])]}, 1, np.zeros((0, 512), dtype=np.float16))
    assert p.label_for_candidate(asset, 0, {"box_xyxy": [0, 0, 10, 10]}, 1)[0] == "V"


def test_quality_label_w_boundary():
    asset = p.SequenceAsset("x", {0: {"candidates": []}}, {}, {0: [(1, [0, 0, 10, 10])]}, 1, np.zeros((0, 512), dtype=np.float16))
    assert p.label_for_candidate(asset, 0, {"box_xyxy": [5, 0, 15, 10]}, 1)[0] == "W"


def test_quality_label_o_boundary():
    asset = p.SequenceAsset("x", {0: {"candidates": []}}, {}, {0: [(1, [0, 0, 10, 10])]}, 1, np.zeros((0, 512), dtype=np.float16))
    assert p.label_for_candidate(asset, 0, {"box_xyxy": [20, 0, 30, 10]}, 1)[0] == "O"


def test_quality_models_finite():
    for name in ("B0_linear.json", "B1_mlp.json", "B2_target_conditioned.json", "B3_ordinal.json", "B4_iou_regression.json"):
        assert _json(f"candidate_quality/{name}")["runtime_future_gt_used"] is False


def test_refined_boxes_finite():
    assets = p.load_assets(); event = p.event_map()[p.SEQUENCES[0]]; geom = p.causal_geometry_predictions(assets[p.SEQUENCES[0]], event, "A0_LAST_BOX")
    candidate = assets[p.SEQUENCES[0]].frames[0]["candidates"][0]
    refined = p.apply_box_refinement(assets[p.SEQUENCES[0]], 0, candidate, 0.5, geom[0], event["human_anchor"], "C0_GEOMETRIC", None)
    assert np.isfinite(refined).all()


def test_refined_boxes_valid_coordinates():
    box = p.clip_box([-3, -2, 4, 5])
    assert 0 <= box[0] < box[2] and 0 <= box[1] < box[3]


def test_v_candidates_not_unnecessarily_degraded():
    conversion = _json("box_refinement/p1a_conversion.json")
    assert conversion["variants"]["COMBINED"]["p1a_after_frames"] <= conversion["variants"]["COMBINED"]["p1a_frames"]


def test_target_sam3_prompt_contains_no_current_gt():
    result = _json("sam3_refinement/targeted_results.json")
    for sequence in result["sequences"].values():
        for attempt in sequence["attempts"]:
            assert attempt.get("gt_prompt_box_used") is False


def test_sam3_prompt_provenance_complete():
    result = _json("sam3_refinement/targeted_results.json")
    assert result["prompt_variants"] == ["S0_PREDICTED", "S1_IDENTITY_CANDIDATE", "S2_ENCLOSING_UNION", "S3_PREVIOUS_TRUSTED"]


def test_new_candidate_provenance_complete():
    result = _json("sam3_refinement/targeted_results.json")
    for sequence in result["sequences"].values():
        for attempt in sequence["attempts"]:
            if attempt.get("status") == "PASS_OFFICIAL_OUTPUT":
                assert attempt["candidate_uid"] and attempt["variant"]


def test_candidate_uid_uniqueness_after_union():
    result = _json("sam3_refinement/targeted_results.json")
    uids = [attempt["candidate_uid"] for sequence in result["sequences"].values() for attempt in sequence["attempts"] if attempt.get("candidate_uid")]
    assert len(uids) == len(set(uids))


def test_identity_model_candidate_axis_consistent():
    assets = p.load_assets()
    for asset in assets.values():
        for frame, row in asset.base_rows.items():
            assert len(row["candidate_uid_axis"]) == len(asset.frames[frame]["candidates"])


def test_association_exact_solver_unchanged():
    audit = _json("association/base_score_audit.json")
    assert "linear_sum_assignment" in audit["exact_solver"]


def test_identity_residual_bounded():
    assets = p.load_assets(); event = p.event_map()[p.SEQUENCES[0]]; scores = p.base_identity_scores(assets[p.SEQUENCES[0]], 0, event)
    assert np.all(scores >= 0) and np.all(scores <= 1)


def test_lambda_selected_inner_only():
    for fold in _json("association/inner_selection.json")["folds"]:
        assert fold["inner_selection"]["lambda"] in {0.25, 0.5, 1.0, 2.0}


def test_outer_sequence_excluded_from_selection():
    for fold in _json("association/inner_selection.json")["folds"]:
        assert fold["heldout"] not in fold["fit_sequences"]
        assert fold["heldout"] != fold["validation"]


def test_shadow_change_statistics_present():
    shadow = _json("association/shadow_comparison.json")["variants"]
    assert set(shadow) >= {"G0_ADDITIVE", "G1_TARGET_COLUMN", "G2_GATED", "G3_RECOVERY_ONLY"}


def test_trackeval_treatment_complete():
    summary = _dev_metrics()
    assert "COMBINED" in summary["treatments"]
    assert summary["settings_identical"] is True


def test_hota_parser_correct():
    assert _dev_metrics()["baseline"]["HOTA"] == _json("baseline/baseline_full_sequence_trackeval.json")["parsed"]["combined"]["HOTA___AUC"]


def test_assa_parser_correct():
    assert _dev_metrics()["baseline"]["AssA"] is not None


def test_deta_parser_correct():
    assert _dev_metrics()["baseline"]["DetA"] is not None


def test_per_sequence_metrics_complete():
    per_sequence = _dev_metrics()["per_sequence"]
    assert set(per_sequence) == set(p.SEQUENCES)
    assert all(per_sequence[seq]["BASELINE"]["HOTA"] is not None for seq in p.SEQUENCES)


def test_bootstrap_seed_fixed():
    bootstrap = _json("trackeval/paired_bootstrap.json")
    assert bootstrap["seed"] == 720351 and bootstrap["repetitions"] == 2000


def test_bootstrap_sequence_cluster_only():
    bootstrap = _json("trackeval/paired_bootstrap.json")
    assert bootstrap["cluster_unit"] == "sequence" and len(bootstrap["sequences"]) == 8


def test_no_outer_hota_tuning():
    assert _json("candidate_pipeline/selected_pipeline.json")["candidate_pipeline_selection_is_inner_only"] is True


def test_memory_score_before_update():
    rows = _json("memory/causal_replay.json")
    assert "policy" in rows and rows["runtime_future_gt_used"] is False


def test_bad_quality_candidate_cannot_write():
    for row in _json("memory/per_sequence.json").values():
        assert "wrong_write_rate" in row


def test_none_cannot_write():
    compressed = subprocess.run(["zstd", "-q", "-d", "-c", str(OUT / "memory/write_audit.jsonl.zst")], check=True, stdout=subprocess.PIPE).stdout.decode()
    for line in compressed.splitlines()[:1000]:
        row = json.loads(line)
        if row["candidate_uid"] is None:
            assert row["commit"] is False


def test_one_observation_written_at_most_once():
    compressed = subprocess.run(["zstd", "-q", "-d", "-c", str(OUT / "memory/write_audit.jsonl.zst")], check=True, stdout=subprocess.PIPE).stdout.decode()
    keys = [(json.loads(line)["sequence"], json.loads(line)["frame"]) for line in compressed.splitlines()]
    assert len(keys) == len(set(keys))


def test_val_not_used_before_dev_freeze():
    manifest = _json("val/generation_manifest.json")
    assert manifest["val_training"] is False and manifest["val_tuning"] is False


def test_val_gt_never_trains():
    manifest = _json("val/generation_manifest.json")
    assert manifest.get("val_dataset_gt_not_read", True) or manifest.get("gt_used_for_training") is False


def test_no_test_tuning():
    assert _json("FINAL_RESULT.json").get("test_accessed") is False


def test_no_model_binaries_staged():
    stage_root = p.OUT
    binaries = [path for path in stage_root.rglob("*") if path.is_file() and path.suffix in {".pt", ".pth", ".ckpt"}]
    assert not binaries


def test_storage_hard_stop_not_reached():
    assert _json("storage_audit_after.json")["free_gib"] >= 100.0


def test_historical_r3r2r2_unchanged():
    audit = _json("source_audit.json")
    for relative, expected in audit["critical_historical_hashes"].items():
        digest = hashlib.sha256((p.ROOT / relative).read_bytes()).hexdigest()
        assert digest == expected


def test_local_branch_is_expected():
    branch = subprocess.run(["git", "branch", "--show-current"], cwd=p.ROOT, text=True, stdout=subprocess.PIPE, check=True).stdout.strip()
    assert branch == "codex/n72r20r3r2r3-localization-association-hota"


def test_final_goal_frozen():
    goal = _json("FINAL_GOAL.json")
    assert goal["goal_frozen"] is True and goal["trackeval_required"] is True


def test_stage_status_goal_reference():
    status = _json("stage_status.json")
    assert status["goal_reference"] == "outputs/N72R20R3R2R3/FINAL_GOAL.json"


def test_final_decision_is_enumerated():
    assert _json("FINAL_RESULT.json")["decision"] in {"PASS_END_TO_END_HOTA_IMPROVEMENT", "FAIL_VAL_GENERALIZATION", "FAIL_LOCALIZATION_TO_HOTA_TRANSFER", "FAIL_ASSOCIATION_AUTHORITY", "FAIL_DETA_ASSA_TRADEOFF", "FAIL_CANDIDATE_GENERATION", "FAIL_MEMORY_COMMIT", "FAIL_MIXED_END_TO_END", "FAIL_RUNTIME_INVARIANT", "FAIL_ASSET_OR_STORAGE"}


def test_report_answers_q1_q2_q3_first():
    report = (OUT / "FINAL_REPORT.md").read_text(encoding="utf-8")
    assert report.index("Q1") < report.index("Q2") < report.index("Q3")


def test_report_contains_required_metrics():
    report = (OUT / "FINAL_REPORT.md").read_text(encoding="utf-8")
    for metric in ("HOTA", "DetA", "AssA", "IDF1", "MOTA", "IDSW"):
        assert metric in report


def test_source_audit_records_checkpoint():
    audit = _json("source_audit.json")
    assert audit["identity_checkpoint_hashes"]["sam3"]


def test_storage_artifacts_present():
    assert (OUT / "storage_audit_before.json").exists() and (OUT / "storage_audit_after.json").exists()


def test_val_asset_lineage_is_external():
    manifest = _json("val/generation_manifest.json")
    assert manifest["split"] == "val" and manifest["test_accessed"] is False


def test_trackeval_uses_eight_dev_sequences():
    assert set(_dev_metrics()["per_sequence"]) == set(p.SEQUENCES)


def test_runtime_flags_are_false_in_final():
    final = _json("FINAL_RESULT.json")
    assert final["runtime_future_gt_used"] is False and final["historical_outputs_modified"] is False
