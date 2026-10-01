"""Focused regression tests for N72R20R3R1 explicit-NONE verification."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from sam3_intermot.association.identity_presence import _assignment_uid
from sam3_intermot.identity_verification.explicit_none_verifier import (
    ExplicitNoneVerifier,
    predict_open_set_identity,
)
from scripts.n72r20r3r1_train_eval import _aggregate_report, _static_gate


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/N72R20R3R1"


def _vector(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    value = rng.normal(size=512).astype(np.float32)
    return value / np.linalg.norm(value)


def _rows(count: int = 3) -> list[dict[str, object]]:
    return [{"candidate_uid": f"seq:1:{index}:{index}", "feature": _vector(index + 10)} for index in range(count)]


def _runtime() -> dict[str, object]:
    return {"frames_since_human_initialization": 4, "frames_since_last_memory_write": 2, "learned_state_human_anchor_cosine": 0.9}


def test_public_id_assignment_uid_is_resolved_against_requested_public_id() -> None:
    solver = {"public_assignments": [{"public_id": 100001, "candidate_uid": "A"}, {"public_id": 100003, "candidate_uid": "C"}]}
    assert _assignment_uid(solver, 100001) == "A"
    assert _assignment_uid(solver, 100003) == "C"
    assert _assignment_uid(solver, 100002) is None


@pytest.mark.parametrize("mode", ["V1_PAIRWISE", "V2_EXPLICIT_NONE"])
@pytest.mark.parametrize("variant", ["A0_ANCHOR_ONLY", "A1_LEARNED_STATE_ONLY", "A2_DUAL_STATE"])
@pytest.mark.parametrize("candidate_count", [0, 1, 5])
def test_model_supports_variable_candidate_counts_and_finite_logits(mode: str, variant: str, candidate_count: int) -> None:
    model = ExplicitNoneVerifier(mode=mode, state_variant=variant)
    candidates = torch.randn(2, max(1, candidate_count), 512)
    mask = torch.zeros(2, max(1, candidate_count), dtype=torch.bool)
    if candidate_count:
        mask[:, :candidate_count] = True
    output = model(torch.randn(2, 512), torch.randn(2, 512), candidates, mask, torch.zeros(2, 4))
    assert torch.isfinite(output["candidate_logits"]).all()
    if mode == "V2_EXPLICIT_NONE":
        assert output["logits"].shape == (2, max(1, candidate_count) + 1)
        assert torch.isfinite(output["none_logit"]).all()


def test_v2_has_a_true_none_class() -> None:
    model = ExplicitNoneVerifier(mode="V2_EXPLICIT_NONE", state_variant="A2_DUAL_STATE")
    output = model(torch.randn(1, 512), torch.randn(1, 512), torch.randn(1, 3, 512), torch.ones(1, 3, dtype=torch.bool), torch.zeros(1, 4))
    assert output["logits"].shape[-1] == 4
    assert output["none_logit"].shape == (1,)


def test_v1_is_candidate_only_and_threshold_can_return_none() -> None:
    model = ExplicitNoneVerifier(mode="V1_PAIRWISE", state_variant="A2_DUAL_STATE")
    result = predict_open_set_identity(
        human_anchor=_vector(1), learned_state=_vector(2), candidate_rows=_rows(), runtime_context=_runtime(), verifier=model, device="cpu", threshold=1.1
    )
    assert result["selected_candidate_uid"] is None
    assert result["runtime_future_gt_used"] is False


def test_empty_candidate_set_fails_closed_to_none() -> None:
    model = ExplicitNoneVerifier(mode="V2_EXPLICIT_NONE", state_variant="A2_DUAL_STATE")
    result = predict_open_set_identity(human_anchor=_vector(1), learned_state=_vector(2), candidate_rows=[], runtime_context=_runtime(), verifier=model, device="cpu")
    assert result["selected_candidate_uid"] is None
    assert result["candidate_created"] is False


def test_candidate_uid_collision_is_rejected() -> None:
    model = ExplicitNoneVerifier(mode="V2_EXPLICIT_NONE", state_variant="A2_DUAL_STATE")
    rows = [_rows()[0], dict(_rows()[1], candidate_uid=_rows()[0]["candidate_uid"])]
    with pytest.raises(ValueError, match="collision"):
        predict_open_set_identity(human_anchor=_vector(1), learned_state=_vector(2), candidate_rows=rows, runtime_context=_runtime(), verifier=model, device="cpu")


def test_runtime_candidate_gt_key_is_rejected() -> None:
    model = ExplicitNoneVerifier(mode="V2_EXPLICIT_NONE", state_variant="A2_DUAL_STATE")
    row = dict(_rows()[0], target_iou=0.8)
    with pytest.raises(ValueError, match="posthoc/GT"):
        predict_open_set_identity(human_anchor=_vector(1), learned_state=_vector(2), candidate_rows=[row], runtime_context=_runtime(), verifier=model, device="cpu")


def test_runtime_context_gt_key_is_rejected() -> None:
    model = ExplicitNoneVerifier(mode="V2_EXPLICIT_NONE", state_variant="A2_DUAL_STATE")
    with pytest.raises(ValueError, match="posthoc/GT"):
        predict_open_set_identity(human_anchor=_vector(1), learned_state=_vector(2), candidate_rows=_rows(), runtime_context={"taxonomy": "P0"}, verifier=model, device="cpu")


def test_predict_does_not_mutate_anchor_or_candidates() -> None:
    model = ExplicitNoneVerifier(mode="V2_EXPLICIT_NONE", state_variant="A2_DUAL_STATE")
    anchor = _vector(1)
    rows = _rows()
    before = [np.asarray(row["feature"]).copy() for row in rows]
    predict_open_set_identity(human_anchor=anchor, learned_state=_vector(2), candidate_rows=rows, runtime_context=_runtime(), verifier=model, device="cpu")
    assert np.array_equal(anchor, _vector(1))
    assert all(np.array_equal(before[index], rows[index]["feature"]) for index in range(len(rows)))


def test_model_size_and_projection_constraints() -> None:
    for mode in ("V1_PAIRWISE", "V2_EXPLICIT_NONE"):
        for variant in ("A0_ANCHOR_ONLY", "A1_LEARNED_STATE_ONLY", "A2_DUAL_STATE"):
            model = ExplicitNoneVerifier(mode=mode, state_variant=variant, projection_dim=64, hidden_dim=64)
            assert model.trainable_parameters < 200_000
            assert model.projection_dim <= 64 and model.hidden_dim <= 64


def test_frozen_backbone_boundary_has_no_osnet_or_gru_parameters() -> None:
    model = ExplicitNoneVerifier()
    assert not any("osnet" in name.lower() or "gru" in name.lower() for name, _ in model.named_parameters())
    anchor = torch.randn(1, 512)
    state = torch.randn(1, 512)
    candidates = torch.randn(1, 2, 512)
    model(anchor, state, candidates, torch.ones(1, 2, dtype=torch.bool), torch.zeros(1, 4))
    assert anchor.requires_grad is False and state.requires_grad is False and candidates.requires_grad is False


def test_training_index_has_reference_only_candidates_and_none_labels() -> None:
    manifest = json.loads((OUT / "training/training_index_manifest.json").read_text())
    assert manifest["candidate_features_copied"] is False
    assert manifest["label_counts"] == {"CANDIDATE": 6595, "NONE": 1819}
    assert manifest["rows"] == 16828


def test_runtime_whitelist_forbids_posthoc_fields() -> None:
    whitelist = json.loads((OUT / "training/runtime_feature_whitelist.json").read_text())
    assert whitelist["runtime_future_gt_used"] is False
    assert {"target_iou", "taxonomy_posthoc", "training_label"}.issubset(set(whitelist["forbidden_runtime_features"]))


def test_split_manifest_is_exactly_eight_sequence_loso_folds() -> None:
    payload = json.loads((OUT / "training/split_manifest.json").read_text())
    folds = payload["folds"]
    assert len(folds) == 8
    assert {item["heldout_sequence"] for item in folds} == {
        "dancetrack0001", "dancetrack0002", "dancetrack0023", "dancetrack0024", "dancetrack0039", "dancetrack0057", "dancetrack0062", "dancetrack0072"
    }
    assert all(item["frame_random_split"] is False for item in folds)


def test_model_selection_is_v2_only_for_primary_architecture() -> None:
    selection = json.loads((OUT / "training/model_selection.json").read_text())
    assert selection["selected_mode"] == "V2_EXPLICIT_NONE"
    assert selection["selected_model"] == "V2_DUAL_STATE"


def test_formal_results_have_three_preregistered_seeds() -> None:
    payload = json.loads((OUT / "static_eval/loso_results.json").read_text())
    assert payload["reports"]["V2_DUAL_STATE"]["static_gate"]["checks"]["S6_all_eight_sequences_reported"] is True
    assert payload["seed_summary"]["V2_DUAL_STATE"]["seeds"] == [720301, 720302, 720303]


def test_static_gate_is_deterministically_failed() -> None:
    report = json.loads((OUT / "static_eval/selected_report.json").read_text())
    assert report["static_gate"]["pass"] is False
    assert report["static_gate"]["checks"]["S5_runtime_gt_false"] is True
    assert report["static_gate"]["checks"]["S7_no_candidate_creation"] is True


def test_static_gate_rejects_bad_fpr_or_recall() -> None:
    rows = [
        {"label_index": -1, "predicted_index": 0, "taxonomy": "P0_TARGET_ABSENT", "present_probability": 0.9, "class_probability": 0.1, "candidate_rank": None, "sequence": "dancetrack0001", "runtime_future_gt_used": False, "candidate_created": False},
        {"label_index": 0, "predicted_index": -1, "taxonomy": "P3_TARGET_AVAILABLE_ASSOCIATION_CORRECT", "present_probability": 0.1, "class_probability": 0.9, "candidate_rank": 1, "sequence": "dancetrack0001", "runtime_future_gt_used": False, "candidate_created": False},
    ]
    report = _aggregate_report(rows)
    assert _static_gate(report)["pass"] is False


def test_r3_immutability_audit_passes() -> None:
    payload = json.loads((OUT / "r3_immutability_audit.json").read_text())
    assert payload["all_unchanged"] is True
    assert payload["historical_r3_modified"] is False


def test_postfinal_audit_decision_unchanged() -> None:
    payload = json.loads((OUT / "r3_postfinal_audit/corrected_r3_diagnostics.json").read_text())
    assert payload["R3_FINAL_DECISION_UNCHANGED"] is True
    assert payload["gate_relevant_metric_changed"] is False
    assert payload["old_candidate_presence_diagnostic"] != payload["corrected_candidate_presence_diagnostic"]


def test_storage_remains_above_hard_stop() -> None:
    payload = json.loads((OUT / "storage_audit_after.json").read_text())
    assert payload["filesystem"]["storage_status"] == "OK"
    assert float(payload["filesystem"]["free_gib"]) > 100.0


def test_exact_solver_file_was_not_modified() -> None:
    import subprocess

    changed = subprocess.run(["git", "diff", "--name-only"], cwd=ROOT, check=True, text=True, stdout=subprocess.PIPE).stdout.splitlines()
    assert "sam3_intermot/association/effect_assignment.py" not in changed


def test_primary_report_has_all_eight_per_sequence_rows() -> None:
    report = json.loads((OUT / "static_eval/selected_report.json").read_text())
    assert len(report["per_sequence"]) == 8
    assert report["runtime_future_gt_used"] is True
    assert report["candidate_created"] is False


def test_bootstrap_is_sequence_clustered_and_preregistered() -> None:
    payload = json.loads((OUT / "static_eval/bootstrap.json").read_text())
    assert payload["repetitions"] == 2000
    assert payload["seed"] == 720311
    assert payload["unit"] == "sequence cluster"


def test_checkpoint_manifest_contains_lineage_and_small_models() -> None:
    payload = json.loads((OUT / "training/checkpoint_manifest.json").read_text())
    assert len(payload["records"]) == 96
    assert all(int(item["trainable_parameters"]) < 200_000 for item in payload["records"])
    assert all(Path(item["canonical_path"]).exists() for item in payload["records"])


def test_none_class_probabilities_are_normalized() -> None:
    model = ExplicitNoneVerifier(mode="V2_EXPLICIT_NONE", state_variant="A2_DUAL_STATE")
    output = model(torch.randn(4, 512), torch.randn(4, 512), torch.randn(4, 5, 512), torch.ones(4, 5, dtype=torch.bool), torch.zeros(4, 4))
    probabilities = torch.softmax(output["logits"], dim=1)
    assert torch.allclose(probabilities.sum(dim=1), torch.ones(4), atol=1.0e-5)


def test_runtime_output_never_creates_candidate() -> None:
    model = ExplicitNoneVerifier(mode="V2_EXPLICIT_NONE", state_variant="A2_DUAL_STATE")
    result = predict_open_set_identity(human_anchor=_vector(4), learned_state=_vector(5), candidate_rows=_rows(2), runtime_context=_runtime(), verifier=model, device="cpu")
    assert result["candidate_created"] is False
    assert result["selected_candidate_uid"] in {None, "seq:1:0:0", "seq:1:1:1"}


def test_model_selection_does_not_use_val_or_test() -> None:
    selection_text = (OUT / "training/model_selection.json").read_text().lower()
    assert "dancetrack/test" not in selection_text
    assert "val" not in selection_text or "validation" in selection_text


def test_causal_authority_is_not_granted_by_static_failure() -> None:
    result = json.loads((OUT / "FINAL_RESULT.json").read_text())
    assert result["final_decision"] == "FAIL_EXPLICIT_NONE_CROSS_SEQUENCE_VERIFICATION"
    assert result["next_association_authority_stage_authorized"] is False
    assert result["association_rescue_run"] is False
