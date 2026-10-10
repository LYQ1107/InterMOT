import pytest
from scripts.n72r21r2_component_global_utility_v1 import paired_counts, sign, EPSILON


def sources(*, complete=True, safe=True, target_value=10):
    labels = {"current_t": {"N10": 0, "non_target_damage": 0, "verified_OTHER_writes": 0, "UNKNOWN_writes": 0},
              "future": {"H100": {"complete": complete, "benefit_label": safe if complete else None,
                                  "risk_label": not safe if complete else None,
                                  "raw_target_value": target_value, "raw_global_component_proxy": target_value}}}
    utility = {"complete_H100": complete, "actual_nine_metrics": {"HOTA": .5, "AssA": .5} if complete else None,
               "paired_delta_vs_own_KEEP": {"HOTA": .01, "AssA": .02} if complete else None,
               "L5_value_label": .015 if complete else None}
    return labels, utility


def test_positive_component_is_not_automatically_positive_global_association():
    labels, utility = sources()
    utility["paired_delta_vs_own_KEEP"]["AssA"] = -.01
    result = paired_counts(labels, utility)
    assert result["safe_component_positive_arms_not_roots"] == 1
    assert result["safe_component_positive_and_both_global_metrics_positive_arms"] == 0
    assert result["safe_component_positive_but_HOTA_or_AssA_negative_arms"] == 1
    assert result["positive_target_value_but_negative_AssA_arms"] == 1


def test_global_metric_improvement_does_not_remove_current_harm():
    labels, utility = sources()
    labels["current_t"]["non_target_damage"] = 1
    result = paired_counts(labels, utility)
    assert result["safe_component_positive_arms_not_roots"] == 0
    assert result["harmful_component_but_both_global_metrics_positive_arms"] == 1


def test_incomplete_future_is_retained_without_sign_or_negative_reward():
    result = paired_counts(*sources(complete=False))
    assert result == {"all_arm_records": 1, "incomplete_H100_excluded_from_sign_comparison": 1}


@pytest.mark.parametrize("kind", ("truncated_as_zero", "complete_component_truncated_utility", "truncated_component_complete_utility"))
def test_boundary_or_fake_reward_mismatch_is_rejected(kind):
    labels, utility = sources(complete=False)
    if kind == "truncated_as_zero": utility["L5_value_label"] = 0.
    elif kind == "complete_component_truncated_utility": labels["future"]["H100"]["complete"] = True
    else: utility["complete_H100"] = True
    with pytest.raises(ValueError): paired_counts(labels, utility)


def test_api_cli_tolerance_is_not_a_policy_operating_point():
    assert sign(EPSILON) == sign(-EPSILON) == "ZERO_WITHIN_API_CLI_TOLERANCE"
    assert sign(2 * EPSILON) == "POSITIVE" and sign(-2 * EPSILON) == "NEGATIVE"


def test_both_global_metrics_positive_is_descriptive_not_root_or_gate_pass():
    result = paired_counts(*sources())
    assert result["safe_component_positive_and_both_global_metrics_positive_arms"] == 1
    assert not any("PASS" in key or "independent" in key for key in result)
