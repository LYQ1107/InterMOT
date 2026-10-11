"""Narrow M9 delivery gate semantics, not real write-risk event evidence."""
import copy
import pytest
from scripts.n72r21r2_memory_delivery_v1 import point_disposition

LIMITS = {"wrong_plus_UNKNOWN_max": .02, "correct_retention_min": .6}


def summary(accepted=0, correct=0, other=0, unknown=0, opportunities=100, qualifies=False):
    return {"exposure_counts_seed_averaged_NOT_independent_sample_size": {
        "accepted_writes": accepted, "write_TARGET": correct, "write_VERIFIED_OTHER": other,
        "write_UNKNOWN": unknown, "correct_C0_committed_observations": opportunities},
        "pooled_exposure_rates": {"wrong_or_UNKNOWN_write_rate": (other + unknown) / accepted if accepted else None,
            "correct_write_retention": correct / opportunities if opportunities else None},
        "empirical_pooled_G4_point_conditions_only": qualifies}


def test_zero_writes_have_undefined_risk_and_no_memory_authorization():
    result = point_disposition(summary(), LIMITS)
    assert result["wrong_plus_UNKNOWN_exposure_rate"] is None
    assert result["correct_write_retention"] == 0
    assert not result["necessary_empirical_G4_point_conditions"] and not result["M_B_authorized"]
    assert "ZERO_WRITES_RISK_UNDEFINED_NOT_SAFETY" in result["reasons"]


def test_unknown_and_rejected_correct_opportunities_remain_in_denominators():
    result = point_disposition(summary(60, 57, 0, 3), LIMITS)
    assert result["wrong_plus_UNKNOWN_exposure_rate"] == .05
    assert result["correct_write_retention"] == .57
    assert len(result["reasons"]) == 2 and not result["necessary_empirical_G4_point_conditions"]


def test_fractional_seed_means_never_become_independent_physical_counts():
    result = point_disposition(summary(60.5, 59.5, .5, .5, qualifies=False), LIMITS)
    assert result["correct_write_retention"] == .595
    assert result["independent_causal_event_risk_CI"] is None
    assert not result["population_two_percent_safety_established"]


def test_exact_gate_boundary_is_only_necessary_point_not_population_safety():
    result = point_disposition(summary(100, 98, 1, 1, qualifies=True), LIMITS)
    assert result["necessary_empirical_G4_point_conditions"]
    assert result["disposition"] == "NECESSARY_POINT_PASS_UNCERTAINTY_NOT_QUALIFIED"
    assert not result["M_B_authorized"] and not result["population_two_percent_safety_established"]


@pytest.mark.parametrize("mutation", ["negative", "nan", "bool", "unknown_omitted", "retention_exceeds", "stored_rate", "stored_gate"])
def test_inconsistent_or_invalid_original_counts_cannot_be_published(mutation):
    row = copy.deepcopy(summary(10, 8, 1, 1))
    counts = row["exposure_counts_seed_averaged_NOT_independent_sample_size"]
    if mutation == "negative": counts["write_UNKNOWN"] = -1
    if mutation == "nan": counts["accepted_writes"] = float("nan")
    if mutation == "bool": counts["accepted_writes"] = True
    if mutation == "unknown_omitted": counts["accepted_writes"] = 9
    if mutation == "retention_exceeds": counts["correct_C0_committed_observations"] = 7
    if mutation == "stored_rate": row["pooled_exposure_rates"]["wrong_or_UNKNOWN_write_rate"] = 0
    if mutation == "stored_gate": row["empirical_pooled_G4_point_conditions_only"] = True
    with pytest.raises(ValueError): point_disposition(row, LIMITS)
