from copy import deepcopy
import pytest
from sam3_intermot.evaluation.whole_video_density import full_video_summary
from sam3_intermot.evaluation.learned_policy_evidence import METRICS


def row(s, click, delta, seed="FIXED", density="MEDIUM"):
    return dict(sequence=s, episode_uid=click, seed=seed, density=density,
        metrics={k: .3 for k in METRICS}, deltas={k: delta for k in METRICS},
        counts=dict(target_correct_frames=4, positive_available_frames=10, physically_visible_frames=20))


def test_one_video_many_clicks_is_one_cluster_and_singleton_has_no_informative_ci():
    result = full_video_summary([row("a", str(i), i) for i in range(4)], ["a", "failed"])
    assert result["overall"]["usable_video_clusters"] == 1
    assert result["overall"]["macro_deltas"]["HOTA"] == 1.5
    assert result["overall"]["paired_video_bootstrap95_delta"] is None
    assert result["missing_or_failed_initialization_videos_retained"] == ["failed"]
    assert result["overall"]["macro_target_claim_recall_available"] == .4
    assert result["overall"]["counts_summed_over_video_exposures_NOT_independent_events"]["positive_available_frames"] == 40


def test_seed_and_click_average_inside_video_not_global_row_pooling():
    rows = [row("a", str(c), seed, seed=seed) for seed in (1, 2, 3) for c in range(4)]
    rows += [row("b", "one", 10, seed=seed) for seed in (1, 2, 3)]
    result = full_video_summary(rows, ["a", "b"], expected_seeds=(1, 2, 3))
    assert result["overall"]["usable_video_clusters"] == 2
    assert result["overall"]["macro_deltas"]["HOTA"] == 6.
    with pytest.raises(ValueError, match="All registered seeds"):
        full_video_summary(rows[:-1], ["a", "b"], expected_seeds=(1, 2, 3))


def test_original_density_is_fixed_and_not_masked_or_unopened_video():
    first = row("a", "0", 0., density="SPARSE")
    output = full_video_summary([first], ["a"])
    assert output["by_frozen_whole_video_density"]["SPARSE"]["insufficient_density_generalization_support"]
    assert output["by_frozen_whole_video_density"]["CROWDED"]["macro_metrics"] is None
    with pytest.raises(ValueError, match="change density"):
        full_video_summary([first, row("a", "1", 0., density="CROWDED")], ["a"])
    with pytest.raises(ValueError, match="unplanned"):
        full_video_summary([first], ["unopened"])


def test_no_duplicate_seed_click_metric_omission_or_rate_as_count():
    r = row("a", "0", 0.)
    with pytest.raises(ValueError, match="Repeated"):
        full_video_summary([r, deepcopy(r)], ["a"])
    missing = deepcopy(r); del missing["metrics"]["IDSW"]
    with pytest.raises(ValueError, match="nine"):
        full_video_summary([missing], ["a"])
    rate = deepcopy(r); rate["counts"]["recall"] = .8
    with pytest.raises(ValueError, match="not rates"):
        full_video_summary([rate], ["a"])
