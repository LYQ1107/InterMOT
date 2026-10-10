"""Descriptive memory uncertainty without treating frames/seeds as people.

These summaries do not authorize association, make zero-write safety claims,
or turn degenerate empirical bootstrap intervals into population guarantees.
"""
from collections import defaultdict
import numpy as np
from scipy.stats import beta


COUNT_NAMES = (
    "accepted_writes", "write_TARGET", "write_VERIFIED_OTHER", "write_UNKNOWN",
    "correct_C0_committed_observations", "positive_available_frames",
    "competitive_available_frames", "rank1_correct_when_positive_available",
    "competitive_rank1_correct", "physically_visible_frames",
    "identity_correct_claim_available", "identity_correct_claim_visible",
    "identity_claim_TARGET", "identity_claim_VERIFIED_OTHER", "identity_claim_UNKNOWN",
    "identity_claim_NONE", "false_presence_no_positive_candidate", "rollback_events",
)


def ratio(numerator, denominator):
    return float(numerator / denominator) if denominator > 0 else None


def checked_counts(raw):
    result = {name: int(raw.get(name, 0)) for name in COUNT_NAMES}
    if any(float(raw.get(name, 0)) != value or value < 0 for name, value in result.items()):
        raise ValueError("Actual counts must be nonnegative integers")
    if result["accepted_writes"] != sum(result[k] for k in ("write_TARGET", "write_VERIFIED_OTHER", "write_UNKNOWN")):
        raise ValueError("UNKNOWN writes must remain in the accepted-risk denominator")
    if result["write_TARGET"] > result["correct_C0_committed_observations"]:
        raise ValueError("Correct retention cannot exceed actual C0 positive opportunities")
    for n, d in (("rank1_correct_when_positive_available", "positive_available_frames"),
                 ("competitive_rank1_correct", "competitive_available_frames"),
                 ("identity_correct_claim_available", "positive_available_frames"),
                 ("identity_correct_claim_visible", "physically_visible_frames")):
        if result[n] > result[d]:
            raise ValueError("Recognition numerator exceeds actual denominator")
    frames = sum(result["identity_claim_" + k] for k in ("TARGET", "VERIFIED_OTHER", "UNKNOWN", "NONE"))
    if frames < result["positive_available_frames"] or frames < result["physically_visible_frames"]:
        raise ValueError("Frame-level claim census is incomplete")
    if result["false_presence_no_positive_candidate"] > frames - result["positive_available_frames"]:
        raise ValueError("False-presence denominator is not actual absent-positive frames")
    return result


def rates(counts):
    c = counts
    claims = sum(c["identity_claim_" + k] for k in ("TARGET", "VERIFIED_OTHER", "UNKNOWN"))
    frames = claims + c["identity_claim_NONE"]
    return {
        "wrong_or_UNKNOWN_write_rate": ratio(c["write_VERIFIED_OTHER"] + c["write_UNKNOWN"], c["accepted_writes"]),
        "verified_OTHER_write_rate": ratio(c["write_VERIFIED_OTHER"], c["accepted_writes"]),
        "UNKNOWN_write_rate": ratio(c["write_UNKNOWN"], c["accepted_writes"]),
        "correct_write_retention": ratio(c["write_TARGET"], c["correct_C0_committed_observations"]),
        "target_rank1_available": ratio(c["rank1_correct_when_positive_available"], c["positive_available_frames"]),
        "target_rank1_competitive": ratio(c["competitive_rank1_correct"], c["competitive_available_frames"]),
        "target_claim_recall_available": ratio(c["identity_correct_claim_available"], c["positive_available_frames"]),
        "target_claim_recall_visible": ratio(c["identity_correct_claim_visible"], c["physically_visible_frames"]),
        "identity_claim_precision": ratio(c["identity_claim_TARGET"], claims),
        "identity_claim_verified_OTHER_rate": ratio(c["identity_claim_VERIFIED_OTHER"], claims),
        "identity_claim_UNKNOWN_rate": ratio(c["identity_claim_UNKNOWN"], claims),
        "false_presence_without_positive_candidate_rate": ratio(c["false_presence_no_positive_candidate"], frames - c["positive_available_frames"]),
    }


def _mean_defined(values):
    present = [v for v in values if v is not None]
    return float(np.mean(present)) if present else None


def cluster_units(rows, level):
    if level not in ("video", "video_local_identity"):
        raise ValueError("No independent event roots can be inferred from contiguous frames")
    groups = defaultdict(lambda: defaultdict(list))
    seen = set()
    for row in rows:
        unique = (row["sequence"], row["episode_uid"], row["seed"])
        if unique in seen:
            raise ValueError("A click/seed replica may not be counted twice")
        seen.add(unique)
        key = row["sequence"] if level == "video" else row["sequence"] + "/identity" + str(row["target_gt_identity"])
        groups[key][row["seed"]].append(row)
    units = []
    for key, seeds in sorted(groups.items()):
        totals, per_seed = [], {}
        for seed, examples in sorted(seeds.items(), key=lambda item: str(item[0])):
            c = {name: sum(checked_counts(e["raw_counts"])[name] for e in examples) for name in COUNT_NAMES}
            totals.append(c)
            per_seed[str(seed)] = {"raw_counts": c, "rates": rates(c), "sole_clicks": len(examples),
                                   "max_bank_drift": max(e["max_bank_drift"] for e in examples)}
        average = {name: float(np.mean([c[name] for c in totals])) for name in COUNT_NAMES}
        unit_rates = {name: _mean_defined([r["rates"][name] for r in per_seed.values()]) for name in rates(average)}
        units.append({"cluster_uid": key, "seeds": per_seed, "seed_mean_exposure_counts": average,
                      "seed_mean_rates": unit_rates, "actual_seed_replicas": len(seeds),
                      "nonabstaining_seed_replicas": sum(c["accepted_writes"] > 0 for c in totals),
                      "any_actual_wrong_or_UNKNOWN_write": any(c["write_VERIFIED_OTHER"] + c["write_UNKNOWN"] > 0 for c in totals)})
    return units


def cluster_summary(rows, level, *, repetitions=2000, seed=730104):
    """Resample entire units; repeated clicks/seeds remain inside their unit."""
    units = cluster_units(rows, level)
    if repetitions < 1:
        raise ValueError("Bootstrap needs positive repetitions")
    names = list(rates({name: 0 for name in COUNT_NAMES}))
    count_matrix = np.asarray([[u["seed_mean_exposure_counts"][k] for k in COUNT_NAMES] for u in units], float)
    averages = {name: 0. for name in COUNT_NAMES} if not units else dict(zip(COUNT_NAMES, np.sum(count_matrix, axis=0).tolist()))
    pooled = rates(averages)
    macro = {name: _mean_defined([u["seed_mean_rates"][name] for u in units]) for name in names}
    macro_draws = {name: [] for name in names}
    pooled_draws = {name: [] for name in names}
    rng = np.random.default_rng(seed)
    for _ in range(repetitions if len(units) >= 2 else 0):
        chosen = rng.integers(0, len(units), len(units))
        draw_counts = dict(zip(COUNT_NAMES, np.sum(count_matrix[chosen], axis=0).tolist()))
        draw_pooled = rates(draw_counts)
        for name in names:
            value = _mean_defined([units[i]["seed_mean_rates"][name] for i in chosen])
            if value is not None:
                macro_draws[name].append(value)
            if draw_pooled[name] is not None:
                pooled_draws[name].append(draw_pooled[name])

    def intervals(draws):
        return {name: {"percentile_95_CI": np.quantile(values, [.025, .975]).tolist() if values else None,
                       "defined_resamples": len(values), "requested_resamples": repetitions,
                       "defined_clusters": sum(u["seed_mean_rates"][name] is not None for u in units),
                       "empirical_interval_degenerate": bool(values) and min(values) == max(values)}
                for name, values in draws.items()}

    exposed = [u for u in units if u["nonabstaining_seed_replicas"] > 0]
    harmed = sum(u["any_actual_wrong_or_UNKNOWN_write"] for u in exposed)
    # This binomial interval is for ANY bad-write cluster, NOT per-write risk.
    # It is conditional on between-cluster independence; identity clusters
    # nested in one video especially do not establish that assumption.
    if exposed:
        n = len(exposed)
        any_bad_ci = [float(beta.ppf(.025, harmed, n - harmed + 1)) if harmed else 0.,
                      float(beta.ppf(.975, harmed + 1, n - harmed)) if harmed < n else 1.]
    else:
        any_bad_ci = None
    point_gate = (averages["accepted_writes"] > 0 and pooled["wrong_or_UNKNOWN_write_rate"] <= .02
                  and pooled["correct_write_retention"] is not None and pooled["correct_write_retention"] >= .6)
    return {"cluster_level": level, "cluster_count": len(units), "clusters": units,
            "exposure_counts_seed_averaged_NOT_independent_sample_size": averages,
            "macro_rates_equal_clusters_seeds_inside": macro,
            "pooled_exposure_rates": pooled, "macro_bootstrap": intervals(macro_draws),
            "pooled_cluster_bootstrap": intervals(pooled_draws), "bootstrap_seed": seed,
            "clusters_with_nonzero_actual_writes": len(exposed), "clusters_with_any_bad_write": harmed,
            "conditional_independence_ANY_bad_cluster_binomial_95_CI_NOT_per_write_risk": any_bad_ci,
            "empirical_pooled_G4_point_conditions_only": bool(point_gate),
            "population_two_percent_safety_established": False, "association_stage_authorized": False,
            "zero_write_rate_is_UNDEFINED_not_zero": True,
            "between_video_recording_and_person_independence_unproven": True,
            "identity_level_is_video_local_GT_identity_not_verified_cross_video_person": level == "video_local_identity",
            "independent_causal_event_95_CI": None,
            "independent_causal_event_CI_status": "UNAVAILABLE_NO_VERIFIED_INDEPENDENT_WRITE_EVENT_ROOTS; frames, bursts, observed intervals and seeds are not substituted",
            "risk_CI_NOT_population_guarantee_even_if_empirical_interval_zero": True}


def paired_recognition(rows, frozen_rows, *, repetitions=2000, seed=730104):
    """Macro video recognition deltas, averaging head seeds inside videos."""
    treatment = cluster_units(rows, "video")
    reference = {u["cluster_uid"]: u for u in cluster_units(frozen_rows, "video")}
    names = ("target_rank1_available", "target_rank1_competitive", "target_claim_recall_available", "target_claim_recall_visible")
    deltas = {name: {} for name in names}
    for u in treatment:
        baseline = reference[u["cluster_uid"]]
        for denominator in ("positive_available_frames", "competitive_available_frames", "physically_visible_frames"):
            if u["seed_mean_exposure_counts"][denominator] != baseline["seed_mean_exposure_counts"][denominator]:
                raise ValueError("Memory recognition comparison changed C0 opportunities/clicks")
        for name in names:
            a, b = u["seed_mean_rates"][name], baseline["seed_mean_rates"][name]
            if a is not None and b is not None:
                deltas[name][u["cluster_uid"]] = a - b
    result = {}
    for name, by_video in deltas.items():
        values = np.asarray(list(by_video.values()), float)
        rng = np.random.default_rng(seed)
        draws = [float(np.mean(values[rng.integers(0, len(values), len(values))]))
                 for _ in range(repetitions if len(values) >= 2 else 0)]
        result[name] = {"paired_video_deltas": by_video, "mean_delta": float(np.mean(values)) if len(values) else None,
                        "paired_video_cluster_95_CI": np.quantile(draws, [.025, .975]).tolist() if draws else None,
                        "defined_video_clusters": len(values), "selection_INNER_not_independent_confirmation": True}
    return result
