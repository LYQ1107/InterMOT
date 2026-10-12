import pytest
from scripts.n72r21r2_native_deployment_support_v1 import fit_support, distribution


def row(branch="KEEP", value=3, role="FIT"):
    return {"sequence": "v", "episode_uid": "click", "event_uid": "probe", "frame": 9,
            "branch": branch, "role": role, "observables": {"observed_native_streak": value}}


def test_source_probe_once_not_branch_inflation():
    assert fit_support([row(), row("RESCUE")]) == ([3, 3], [3])


def test_inconsistent_source_probe_rejected():
    with pytest.raises(ValueError, match="inconsistent"):
        fit_support([row(), row("RESCUE", 4)])


def test_inner_not_optimizer_support():
    with pytest.raises(ValueError, match="FIT"):
        fit_support([row(role="INNER")])


def test_marginal_integer_support_and_empty_video_are_distinct():
    d = distribution([0, 3, 8], [1, 5])
    assert d["FIT_min_max"] == [1, 5] and d["below_FIT_min"] == d["above_FIT_max"] == 1
    assert distribution([], [1, 5])["quantiles_0_10_25_50_75_90_100"] is None


@pytest.mark.parametrize("values,reference", [([True], None), ([-1], None), ([2.], None), ([2], [])])
def test_proxy_or_missing_reference_not_exact_counter_support(values, reference):
    with pytest.raises(ValueError):
        distribution(values, reference)
