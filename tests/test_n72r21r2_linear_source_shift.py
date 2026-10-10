import numpy as np
import pytest
from sam3_intermot.evaluation.linear_source_shift import constant_feature_decomposition, score_only_pass


def test_constant_support_attribution_is_read_only_and_not_deployment_policy():
    fit = np.array([[0, 0, 1], [0, 1, 0]], np.float32)
    own = np.array([[20, 1, 0]], np.float32)
    before = own.copy()
    weight = np.array([[-1, 2, 0], [1, -5, 0], [0, 1, 0]], np.float32)
    const, full, contribution, residual = constant_feature_decomposition(fit, own, weight, np.zeros(3), causal_dim=2)
    assert const.tolist() == [0] and np.array_equal(own, before)
    assert np.allclose(full, contribution + residual)
    point = dict(claim_min=.8, risk_max=.02)
    assert score_only_pass(full, point).tolist() == [False]
    assert score_only_pass(residual, point).tolist() == [True]  # Not a real action.


def test_invalid_linear_axes_and_nonfinite_inputs_rejected():
    with pytest.raises(ValueError):
        constant_feature_decomposition(np.ones((2, 40)), np.ones((2, 39)), np.ones((3, 40)), np.zeros(3))
    with pytest.raises(ValueError):
        score_only_pass([[float("nan"), 0., 1.]], dict(claim_min=.8, risk_max=.02))
