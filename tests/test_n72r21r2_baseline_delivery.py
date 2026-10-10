from copy import deepcopy
import pytest
from sam3_intermot.evaluation.baseline_delivery import historical_summary
from sam3_intermot.evaluation.learned_policy_evidence import METRICS


def fixture():
    actual = {}
    for case, gain in (("CLICK_C0", 0), ("ACIB_FULL_SEED1", .1), ("ACIB_FULL_SEED2", .2), ("ACIB_FULL_SEED3", .3)):
        values = {k: (.4 + gain if k not in ("FP", "FN", "IDSW") else 4) for k in METRICS}
        actual[case] = {**values, "per_sequence": {s: deepcopy(values) for s in ("a", "b")}}
    return actual


def test_eight_replays_are_two_videos_not_eight_or_best_seed():
    result = historical_summary(fixture(), ["a", "b"], [1, 2, 3])
    assert result["actual_replay_cells_NOT_videos"] == 8
    assert result["independent_video_cluster_count_UPPER_BOUND"] == 2
    assert result["paired_all_seed_equal_video_delta"]["macro"]["HOTA"] == pytest.approx(.2)
    assert result["macro_FULL_all_seed_metrics"]["HOTA"] == pytest.approx(.6)
    assert not result["fresh_FIT_INNER_or_confirmation_population"]
    assert not result["scientific_generalization_PASS"]


def test_legacy_combined_values_are_not_reused_as_equal_video_macro():
    actual = fixture(); actual["CLICK_C0"]["HOTA"] = .9
    result = historical_summary(actual, ["a", "b"], [1, 2, 3])
    assert result["macro_C0_metrics"]["HOTA"] == .4
    assert result["original_combined_metric_summaries_separate_NOT_equal_video_means"]["CLICK_C0"]["HOTA"] == .9


@pytest.mark.parametrize("corruption", ("missing_seed", "missing_video", "duplicate_seed", "duplicate_video", "missing_metric", "nonfinite", "percent_units", "negative_count", "fractional_count", "boolean_count"))
def test_missing_scope_or_invalid_metrics_never_qualify(corruption):
    actual = fixture(); sequences, seeds = ["a", "b"], [1, 2, 3]
    if corruption == "missing_seed": del actual["ACIB_FULL_SEED2"]
    elif corruption == "missing_video": del actual["ACIB_FULL_SEED2"]["per_sequence"]["b"]
    elif corruption == "duplicate_seed": seeds = [1, 1, 3]
    elif corruption == "duplicate_video": sequences = ["a", "a"]
    elif corruption == "missing_metric": del actual["ACIB_FULL_SEED2"]["per_sequence"]["b"]["AssA"]
    elif corruption == "nonfinite": actual["ACIB_FULL_SEED2"]["per_sequence"]["b"]["AssA"] = float("nan")
    elif corruption == "negative_count": actual["ACIB_FULL_SEED2"]["per_sequence"]["b"]["FP"] = -1
    elif corruption == "fractional_count": actual["ACIB_FULL_SEED2"]["per_sequence"]["b"]["FP"] = .5
    elif corruption == "boolean_count": actual["ACIB_FULL_SEED2"]["per_sequence"]["b"]["FP"] = True
    else: actual["ACIB_FULL_SEED2"]["per_sequence"]["b"]["AssA"] = 80
    with pytest.raises(ValueError): historical_summary(actual, sequences, seeds)
