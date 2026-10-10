from copy import deepcopy
import numpy as np
import pytest
import torch
from sam3_intermot.one_click.intervention_features import FEATURE_NAMES, feature_vector
from sam3_intermot.one_click.event_authority_learning import (CURRENT_DIM, OBJECTIVES, encode_runtime_input,
    objective_target, correlated_sequence_weights, fit_normalizer, normalize_inputs)
from sam3_intermot.one_click.event_authority_models import FAMILIES, EventAuthorityHead, independent_head_loss


def runtime():
    features = {k: 0. for k in FEATURE_NAMES}
    return {"features": features, "feature_vector": feature_vector(features).tolist(),
            "causal_previous_feature_vectors": {"H3": [[0.] * 32] * 3, "H8": [[0.] * 32] * 8}}


def labels():
    current = dict(target_correct=1, N01=1, N10=0, non_target_damage=0, verified_OTHER_writes=0, UNKNOWN_writes=0)
    return {"current_t": current, "future": {"H" + str(h): dict(complete=True, risk_label=False, benefit_label=True,
        normalized_target_value_label=.1, normalized_global_proxy_label=-.2, UNKNOWN=0, verified_OTHER=0, KEEP_verified_OTHER=0) for h in (20, 50, 100)},
        "raw_frame_components": [dict(KEEP_outcome="TARGET")] * 101}


def test_runtime_encoder_rejects_truth_but_has_distinct_delayed_policy_axis():
    r = runtime()
    keep, past = encode_runtime_input(r, "KEEP")
    delay, _ = encode_runtime_input(r, "DELAYED_CHALLENGER")
    assert keep.shape == delay.shape == (CURRENT_DIM,) and past.shape == (3, 32)
    assert np.array_equal(keep[:32], delay[:32]) and not np.array_equal(keep, delay)
    r["offline_supervision_labels"] = {"GT": 1}
    with pytest.raises(ValueError):
        encode_runtime_input(r, "KEEP")


def test_runtime_encoder_checks_immutable_feature_axis_and_past_shape():
    r = runtime()
    r["features"]["target_gt_identity"] = 12
    with pytest.raises(ValueError):
        encode_runtime_input(r, "KEEP")
    r = runtime()
    r["feature_vector"][0] = 1.
    with pytest.raises(ValueError):
        encode_runtime_input(r, "KEEP")
    r = runtime()
    r["causal_previous_feature_vectors"]["H3"].append([0.] * 32)
    with pytest.raises(ValueError):
        encode_runtime_input(r, "KEEP")


def test_incomplete_futures_not_negative_even_when_current_is_correct():
    r = labels()
    r["future"]["H100"]["complete"] = False
    assert objective_target(r, OBJECTIVES[0])["benefit"] == 1.
    assert all(objective_target(r, objective) is None for objective in OBJECTIVES[1:])


def test_actual_L5_cannot_substitute_global_proxy_and_harm_includes_current():
    r = labels()
    assert objective_target(r, OBJECTIVES[4])["value"] == -.2
    assert objective_target(r, OBJECTIVES[5]) is None
    actual = {"complete_H100": True, "L5_value_label": .03}
    assert objective_target(r, OBJECTIVES[5], actual_trajectory_utility=actual)["value"] == .03
    r["current_t"]["non_target_damage"] = 1
    target = objective_target(r, OBJECTIVES[5], actual_trajectory_utility=actual)
    assert target["risk"] == 1. and target["benefit"] == 0.


def test_equal_video_then_observed_interval_weights_do_not_inflate_long_errors():
    def row(seq, group, event):
        return dict(sequence=seq, episode_uid=seq + "click0", correlation_group_not_proven_causal_origin=group, event_uid=event)
    rows = [row("A", "long", "event1")] * 10 + [row("A", "short", "event2"), row("B", "only", "event3")]
    weights = correlated_sequence_weights(rows)
    assert np.isclose(weights[:10].sum(), .25) and np.isclose(weights[10], .25) and np.isclose(weights[11], .5)
    values = np.zeros((len(rows), CURRENT_DIM), np.float32)
    values[-1, 0] = 10.
    mean, scale = fit_normalizer(values, weights)
    assert mean[0] == 5.
    future_inner = np.full((1, CURRENT_DIM), 1e6)
    assert mean[0] == 5.  # INNER is never passed to fit_normalizer.
    normalized, past = normalize_inputs(future_inner, np.zeros((1, 3, 32)), mean, scale)
    assert np.count_nonzero(past) == 0 and normalized[0, 0] > 1000


@pytest.mark.parametrize("family", FAMILIES)
def test_all_families_have_real_gradients_distinct_keep_and_causal_input_axes(family):
    torch.manual_seed(730101)
    model = EventAuthorityHead(family)
    current, past, keep = torch.randn(4, CURRENT_DIM), torch.randn(4, 3, 32), torch.randn(4, CURRENT_DIM)
    output = model(current, past, keep_current=keep)
    assert output.shape == (4, 3) and torch.isfinite(output).all()
    target = torch.tensor([[1., 0., .1], [0., 1., -.2], [0., 0., 0.], [1., 0., .3]])
    independent_head_loss(output, target).mean().backward()
    assert sum(float(p.grad.abs().sum()) for p in model.parameters() if p.grad is not None) > 0
    if family in ("PAIRWISE_RANKER", "COUNTERFACTUAL_ACTION_VALUE"):
        assert torch.equal(model(keep, past, keep_current=keep)[:, 2], torch.zeros(4))
        with pytest.raises(ValueError):
            model(current, past)


def test_relational_family_is_not_implicitly_authorized():
    with pytest.raises(ValueError):
        EventAuthorityHead("RELATIONAL_CONDITIONAL")
