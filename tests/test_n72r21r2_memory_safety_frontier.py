import pytest
from sam3_intermot.evaluation.memory_safety_frontier import cluster_summary, checked_counts, paired_recognition
from scripts.n72r21r2_memory_frontier import case_group


def example(video, seed=1, accepted=60, other=0, unknown=0, identity=3, episode="click0"):
    return {"sequence": video, "seed": seed, "episode_uid": video + episode, "target_gt_identity": identity,
            "max_bank_drift": .1, "raw_counts": {"accepted_writes": accepted, "write_TARGET": accepted-other-unknown,
            "write_VERIFIED_OTHER": other, "write_UNKNOWN": unknown, "correct_C0_committed_observations": 100,
            "positive_available_frames": 100, "competitive_available_frames": 100,
            "rank1_correct_when_positive_available": 50, "competitive_rank1_correct": 50,
            "physically_visible_frames": 100, "identity_claim_TARGET": 50, "identity_claim_NONE": 50,
            "identity_correct_claim_available": 50, "identity_correct_claim_visible": 50}}


def test_seeds_and_clicks_never_inflate_video_cluster_count():
    rows = [example(v, s) for v in ("v1", "v2") for s in (1, 2, 3)]
    summary = cluster_summary(rows, "video", repetitions=20)
    assert summary["cluster_count"] == 2
    assert summary["exposure_counts_seed_averaged_NOT_independent_sample_size"]["accepted_writes"] == 120
    assert summary["empirical_pooled_G4_point_conditions_only"]
    assert not summary["population_two_percent_safety_established"]
    assert summary["independent_causal_event_95_CI"] is None
    assert summary["macro_bootstrap"]["wrong_or_UNKNOWN_write_rate"]["empirical_interval_degenerate"]
    assert summary["conditional_independence_ANY_bad_cluster_binomial_95_CI_NOT_per_write_risk"][1] > .8
    identity = cluster_summary(rows, "video_local_identity", repetitions=20)
    assert identity["cluster_count"] == 2  # Same numeric GT-ID in another video is NOT same verified person.


def test_abstention_is_not_zero_risk_or_memory_success():
    summary = cluster_summary([example("v1", accepted=0), example("v2", accepted=0)], "video", repetitions=10)
    assert summary["pooled_exposure_rates"]["wrong_or_UNKNOWN_write_rate"] is None
    assert summary["macro_bootstrap"]["wrong_or_UNKNOWN_write_rate"]["percentile_95_CI"] is None
    assert not summary["empirical_pooled_G4_point_conditions_only"]
    assert summary["clusters_with_nonzero_actual_writes"] == 0


def test_unknown_counts_as_write_risk_but_not_verified_other():
    row = example("v1", accepted=100, unknown=10)
    summary = cluster_summary([row], "video", repetitions=10)
    assert summary["pooled_exposure_rates"]["wrong_or_UNKNOWN_write_rate"] == .1
    assert summary["pooled_exposure_rates"]["verified_OTHER_write_rate"] == 0
    assert not summary["empirical_pooled_G4_point_conditions_only"]
    row["raw_counts"]["accepted_writes"] = 90
    with pytest.raises(ValueError, match="UNKNOWN"):
        checked_counts(row["raw_counts"])


def test_paired_recognition_requires_same_opportunities_and_frozen_references():
    frozen = [example(v, "FIXED", accepted=0) for v in ("v1", "v2")]
    rows = [example(v, s) for v in ("v1", "v2") for s in (1, 2, 3)]
    for r in rows:
        r["raw_counts"]["competitive_rank1_correct"] += 10
    result = paired_recognition(rows, frozen, repetitions=20)
    assert result["target_rank1_competitive"]["mean_delta"] == pytest.approx(.1)
    assert result["target_rank1_competitive"]["defined_video_clusters"] == 2
    rows[0]["raw_counts"]["competitive_available_frames"] = 99
    with pytest.raises(ValueError, match="opportunities"):
        paired_recognition(rows, frozen, repetitions=20)
    with pytest.raises(ValueError, match="twice"):
        cluster_summary([frozen[0], frozen[0]], "video", repetitions=10)


def test_case_family_keeps_point_and_seed_distinct():
    assert case_group("MLP__seed730101__SELECTED_CURRENT_RISK_ONLY") == ("MLP__SELECTED_CURRENT_RISK_ONLY", 730101)
    assert case_group("LOGISTIC__seed730103__FIXED_P50_UNQUALIFIED_DIAGNOSTIC") == ("LOGISTIC__FIXED_P50_UNQUALIFIED_DIAGNOSTIC", 730103)
    assert case_group("FROZEN") == ("FROZEN", "FIXED")
