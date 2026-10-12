import numpy as np
import pytest
import torch
from scripts import n72r21r2_state_feature_support_v1 as audit
from sam3_intermot.one_click.event_authority_models import EventAuthorityHead


def fixture():
    model = EventAuthorityHead("SMALL_MLP", hidden=32)
    initial = {k: v.detach().clone() for k, v in model.state_dict().items()}
    with torch.no_grad():
        for p in model.parameters(): p.add_(.2)
    mean, scale = np.zeros(32, np.float32), np.ones(32, np.float32)
    saved = {"schema": "N72R21R2_EVENT_AUTHORITY_V1", "family": "SMALL_MLP",
        "objective": "L3_HARM_AWARE", "seed": 1, "epoch": 2, "hidden": 32,
        "model": model.state_dict(), "feature_names": list(audit.FEATURE_NAMES),
        "training_state_source": "TREATMENT_STATE", "protocol_sha256": "p",
        "actual_model_generated_on_policy": False,
        "authority_status": "TRAINED_UNCALIBRATED_NOT_DEPLOYABLE",
        "FIT_mean": mean.tolist(), "FIT_scale": scale.tolist()}
    fit = {"mode": "TREATMENT_STATE", "family": "SMALL_MLP", "objective": "L3_HARM_AWARE",
        "seed": 1, "selected_epoch": 2, "protocol_sha256": "p", "parameter_count": 1411,
        "changed_weight_elements": 1411, "nonzero_gradient_steps": 2,
        "actual_model_generated_on_policy": False, "association_authority": False,
        "status": "COMPLETE_ACTUAL_STATE_SOURCE_CONTRAST_UNCALIBRATED_NOT_DEPLOYABLE"}
    return saved, initial, fit, mean, scale


def test_exact_actual_source_mode_normalizer_and_frozen_predictor():
    saved, initial, fit, mean, scale = fixture()
    predictor = audit.bound_predictor(saved, initial, fit, "TREATMENT_STATE", 1, mean, scale)
    assert not predictor.model.training and not any(p.requires_grad for p in predictor.model.parameters())


@pytest.mark.parametrize("which,key,value", [
    ("fit", "mode", "MIXED_STATE"), ("fit", "family", "TREATMENT_STATE"),
    ("fit", "seed", 2), ("fit", "objective", "L0_CURRENT_CORRECTNESS"),
    ("fit", "actual_model_generated_on_policy", True), ("fit", "association_authority", True),
    ("fit", "nonzero_gradient_steps", 0), ("fit", "status", "FAIL_MEASURED_STATE_SOURCE_SUPERVISION_DIVERSITY"),
    ("saved", "authority_status", "DEPLOYABLE"), ("saved", "FIT_mean", [1.] * 32),
    ("saved", "FIT_scale", [2.] * 32), ("saved", "training_state_source", "BASELINE_STATE")])
def test_wrong_source_architecture_or_authority_and_normalizer_rejected(which, key, value):
    saved, initial, fit, mean, scale = fixture()
    (fit if which == "fit" else saved)[key] = value
    with pytest.raises((ValueError, RuntimeError)):
        audit.bound_predictor(saved, initial, fit, "TREATMENT_STATE", 1, mean, scale)


def row(frame=1, sample=True, effective=False):
    return {"frame": frame, "runtime_GT_read": False, "runtime_future_GT_used": False,
        "authority": {"causal_feature_sample": sample, "effective_assignment_change": effective,
            "all_current_choices": [{"branch": "KEEP"}, {"branch": "REJECT_TARGET"}],
            "own_primary_feature_vector": [0.] * 32,
            "causal_previous_feature_vectors_H3": [[0.] * 32 for _ in range(3)]}}


def test_exact_every16_postclick_or_effective_action_sampling_not_replaced():
    assert audit.sampled_choices(row(1), {"frame": 0})
    assert audit.sampled_choices(row(17), {"frame": 0})
    assert audit.sampled_choices(row(2, False), {"frame": 0}) is None
    assert audit.sampled_choices(row(2, True, True), {"frame": 0})
    for r in (row(1, False), row(2, True), row(2, False, True), row(0, True)):
        with pytest.raises(ValueError): audit.sampled_choices(r, {"frame": 0})


@pytest.mark.parametrize("change", ["GT", "future", "duplicate", "no_KEEP", "primary", "past", "nonfinite", "not_bool"])
def test_truth_or_mismatched_choices_history_and_sampling_flags_rejected(change):
    r = row()
    if change == "GT": r["runtime_GT_read"] = True
    elif change == "future": r["runtime_future_GT_used"] = True
    elif change == "duplicate": r["authority"]["all_current_choices"].append({"branch": "KEEP"})
    elif change == "no_KEEP": r["authority"]["all_current_choices"] = [{"branch": "REJECT_TARGET"}]
    elif change == "primary": r["authority"]["own_primary_feature_vector"] = [0.] * 31
    elif change == "past": r["authority"]["causal_previous_feature_vectors_H3"] = [[0.] * 32] * 2
    elif change == "nonfinite": r["authority"]["own_primary_feature_vector"][0] = np.nan
    else: r["authority"]["causal_feature_sample"] = 1
    with pytest.raises(ValueError): audit.sampled_choices(r, {"frame": 0})


def test_each_point_requires_three_seeds_all_inner_videos_and_all_three_seals(tmp_path, monkeypatch):
    monkeypatch.setattr(audit, "OUT", tmp_path)
    p = {"fitted_modes": list(audit.MODES), "points": {"P0": {}}, "seeds": [1, 2, 3],
        "split": {"inner": ["seq" + str(i) for i in range(8)]}}
    paths = audit.required_point_paths("TREATMENT_STATE", "P0", p)
    assert len(paths) == len(set(paths)) == 72
    assert all("STATE__TREATMENT_STATE__L3_HARM_AWARE__seed" in str(path) for path in paths)
    for mode, point in [("BASELINE_STATE", "P0"), ("TREATMENT_STATE", "new")]:
        with pytest.raises(ValueError): audit.required_point_paths(mode, point, p)


def test_no_ready_seed_or_video_subset_can_open_fit_source(tmp_path, monkeypatch):
    p = {"source_sha256": {}, "frozen": True, "goal": audit.GOAL,
        "state_policy_protocol_sha256": "x", "state_fit_protocol_sha256": "x",
        "original_MAIN_support_protocol_sha256": "x", "fitted_modes": list(audit.MODES),
        "seeds": [1, 2, 3], "points": {"P0": {}}, "split": {"fit": ["f"], "inner": ["i"]}}
    monkeypatch.setattr(audit, "OUT", tmp_path)
    monkeypatch.setattr(audit.Evidence, "json", lambda self, path: p)
    monkeypatch.setattr(audit.Evidence, "digest", lambda self, path: "x")
    monkeypatch.setattr(audit, "preregistration", lambda: {"split": p["split"]})
    def forbidden(*args, **kwargs): raise AssertionError("FIT labels cannot open for an unready point")
    monkeypatch.setattr(audit.source, "load_records", forbidden)
    with pytest.raises(RuntimeError, match="not ready"):
        audit.run("TREATMENT_STATE", "P0")
