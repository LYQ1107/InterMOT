import pytest
from scripts.n72r21r2_open_set_delivery_v1 import family_video_intervals, original_grid_summary


def test_seed_averages_within_video_do_not_triple_independent_clusters():
    seeds = [1, 2, 3]
    models = [{"model": str(seed), "family": "MLP", "seed": seed} for seed in seeds]
    strategies = {}
    for seed in seeds:
        rows = {"a": {"rates": {"recall": float(seed != 1)}}, "b": {"rates": {"recall": 0.}}}
        strategies[str(seed)] = {role: {"per_sequence": rows} for role in ("fit", "inner")}
    split = {"fit": ["a", "b"], "inner": ["a", "b"]}
    result = family_video_intervals(strategies, models, seeds, split)["MLP"]["inner"]
    assert result["video_cluster_CI95"]["recall"]["video_clusters"] == 2
    assert result["video_cluster_CI95"]["recall"]["mean"] == pytest.approx(1/3)
    assert result["defined_seed_counts_per_video"]["a"]["recall"] == 3
    with pytest.raises(ValueError): family_video_intervals(strategies, models[:2], seeds, split)


def test_no_exposure_undefined_rates_not_imputed_as_correct_or_safety():
    models = [{"model": str(seed), "family": "MLP", "seed": seed} for seed in (1, 2, 3)]
    strategies = {str(seed): {role: {"per_sequence": {"empty": {"rates": {"recall": None}}}} for role in ("fit", "inner")} for seed in (1, 2, 3)}
    result = family_video_intervals(strategies, models, [1,2,3], {"fit": ["empty"], "inner": ["empty"]})
    assert result["MLP"]["fit"]["video_cluster_CI95"]["recall"]["CI95"] is None


def test_row_fpr_measured_grid_not_physical_absence_safety_or_new_selection():
    original = {"status": "CALIBRATION_ABSTAIN"}
    point = {"selection": {"probability_min": .5}, "nonvacuous": True, "empirical_current_safe": False,
             "result": {"counts": {}, "micro": {"verified_OTHER_row_FPR": .001, "target_recall_given_available": .2}}}
    result = original_grid_summary({"selection": original, "all_INNER_operating_points": [point]})
    assert result["Recall_at_verified_OTHER_row_FPR2_original_measured_GRID_ONLY"] == .2
    assert result["selection_unchanged"] == original
    assert result["NOT_physical_absence_FPR_or_accepted_claim_contamination_bound"]
