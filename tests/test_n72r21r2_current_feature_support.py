import numpy as np
import pytest
from sam3_intermot.evaluation.current_feature_support import (
    support_comparison, unpadded_history, prediction_error)


def test_constant_FIT_feature_shift_is_measured_not_zeroed():
    own = np.array([[4., .5], [0., 2.]], np.float32)
    before = own.copy()
    r = support_comparison([[0., 0.], [0., 1.]], own, ["pending", "cos"], [0., .5], [.05, .5])
    assert r["features"]["pending"]["FIT_exactly_constant"]
    assert r["features"]["pending"]["outside_FIT_marginal_range_rows"] == 1
    assert r["features"]["pending"]["max_abs_actual_frozen_normalized_value"] == 80.
    assert r["any_marginal_outside_rows"] == 2
    np.testing.assert_array_equal(own, before)
    assert r["rows_not_independent_scientific_units"] and r["marginal_membership_is_not_joint_support"]


def test_exact_endpoints_are_not_marginal_outliers():
    r = support_comparison([[0.], [1.]], [[0.], [1.]], ["a"], [.5], [.5])
    assert r["any_marginal_outside_rows"] == 0


def test_empty_failed_video_not_imputed_as_successful_zero():
    r = support_comparison([[1.]], np.empty((0, 1)), ["a"], [1.], [.05])
    assert r["actual_rows_correlated"] == 0
    assert r["features"]["a"]["actual_quantiles_0_10_25_50_75_90_100"] is None
    assert r["features"]["a"]["outside_FIT_marginal_range_fraction"] is None


@pytest.mark.parametrize("bad", [np.nan, np.inf])
def test_nonfinite_features_rejected(bad):
    with pytest.raises(ValueError):
        support_comparison([[0.]], [[bad]], ["a"], [0.], [.05])


@pytest.mark.parametrize("mean,scale", [([0.], [0.]), ([0.], [-1.]), ([0.], [np.nan]), ([0., 0.], [1.])])
def test_invalid_frozen_normalizer_rejected(mean, scale):
    with pytest.raises(ValueError):
        support_comparison([[0.]], [[0.]], ["a"], mean, scale)


def test_padding_removed_only_when_whole_vector_exact_zero():
    h = np.array([[[0., 0.], [0., 2.], [1., 0.]]])
    result, padding = unpadded_history(h, 2)
    assert padding == 1
    np.testing.assert_array_equal(result, [[0., 2.], [1., 0.]])


def test_wrong_history_length_rejected():
    with pytest.raises(ValueError):
        unpadded_history(np.zeros((1, 8, 2)), 2)


def test_actual_three_prediction_values_numeric_not_bitwise_proof():
    a = {"beneficial": .3, "harmful": .6, "value": .2}
    assert prediction_error(a, a) == 0
    assert prediction_error(a, {**a, "value": .1}) == pytest.approx(.1)
    with pytest.raises(ValueError):
        prediction_error(a, {**a, "future_label": 0})
