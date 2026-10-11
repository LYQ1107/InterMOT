import numpy as np
import pytest
from scripts import n72r21r2_state_source_support_v1 as support


def row(sequence="a", dtype=np.float32):
    return {"sequence": sequence, "episode_uid": sequence + "click", "event_uid": "event",
        "correlation_group_not_proven_causal_origin": "interval", "state_source": "BASELINE_STATE",
        "target": {"benefit": 0., "risk": 1.}, "x": np.array([1., 2.], dtype=dtype)}


def test_full_row_fingerprint_includes_dtype_and_shape():
    first = row()
    assert support.row_digest(first) != support.row_digest(row(dtype=np.float64))
    second = row(); second["x"] = second["x"].reshape(1, 2)
    assert support.row_digest(first) != support.row_digest(second)


def test_full_row_fingerprint_includes_target_not_only_features():
    first, second = row(), row(); second["target"]["benefit"] = 1.
    assert support.row_digest(first) != support.row_digest(second)


def test_common_inner_fingerprint_preserves_order():
    a, b = row("a"), row("b")
    assert support.record_digest([a, b]) != support.record_digest([b, a])


def test_describe_weights_macro_sequences_not_row_count():
    a, b = row("a"), row("b")
    b["event_uid"] = "b-event"
    result = support.describe([a, b])
    assert result["hierarchical_sequence_weight_mass"] == {"a": .5, "b": .5}
    assert result["target_counts"] == {"benefit": 0., "risk": 2.}
    assert result["correlation_controls_NOT_independent_causal_roots"]


@pytest.mark.parametrize("fit,inner,benefit,risk,expected", [
    (1, 1, 0., 1., True), (1, 1, 1., 1., False), (0, 1, 0., 1., False),
    (1, 0, 0., 1., False), (1, 1, 0., 0., False),
])
def test_only_measured_zero_benefit_support_permits_no_fit_receipt(fit, inner, benefit, risk, expected):
    value = {"FIT": {"rows": fit, "target_counts": {"benefit": benefit, "risk": risk}}, "INNER": {"rows": inner}}
    assert support.no_fit_supported(value) is expected


def test_never_execute_positive_mode_as_supervision_only_failure():
    value = {"FIT": {"rows": 1, "target_counts": {"benefit": 1., "risk": 1.}}, "INNER": {"rows": 1}}
    with pytest.raises(ValueError, match="zero-benefit"):
        support.seal_original_no_optimizer_failure(1, value)


def test_nonfinite_features_never_hash_as_valid_evidence():
    value = row(); value["x"][0] = np.nan
    with pytest.raises(ValueError):
        support.row_digest(value)
