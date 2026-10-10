"""Legacy two-video controls and fresh controls are separate populations."""
import math
import numpy as np
from .learned_policy_evidence import METRICS, video_macro


def historical_summary(actual, sequences, seeds):
    names = ["ACIB_FULL_SEED" + str(seed) for seed in seeds]
    if len(set(sequences)) != len(sequences) or len(set(seeds)) != len(seeds) or not seeds or not sequences:
        raise ValueError("Exact unique historical video/seed census required")
    if set(actual) != {"CLICK_C0", *names}:
        raise ValueError("All original historical seeds required; no best-seed subset")
    for metrics in actual.values():
        if set(metrics["per_sequence"]) != set(sequences):
            raise ValueError("Every original historical video is required")
        for values in [metrics, *metrics["per_sequence"].values()]:
            if any(k not in values or not math.isfinite(values[k]) for k in METRICS):
                raise ValueError("All nine finite actual metrics required")
            if any(not 0 <= values[k] <= 1 for k in ("HOTA", "AssA", "DetA", "LocA", "IDF1")):
                raise ValueError("Fraction metrics cannot be percentages")
            if any(isinstance(values[k], bool) or values[k] < 0 or values[k] != int(values[k]) for k in ("FP", "FN", "IDSW")):
                raise ValueError("Count metrics must be nonnegative integers")
    cells, per_video = [], {}
    for sequence in sequences:
        base = {k: actual["CLICK_C0"]["per_sequence"][sequence][k] for k in METRICS}
        full = {k: float(np.mean([actual[name]["per_sequence"][sequence][k] for name in names])) for k in METRICS}
        per_video[sequence] = {"C0": base, "ACIB_FULL_all_seed_mean": full}
        for seed, name in zip(seeds, names, strict=True):
            cells.append({"sequence": sequence, "seed": seed, "valid_initializations": 1,
                "paired_deltas_vs_C0": {"sole_legacy_click": {k: actual[name]["per_sequence"][sequence][k] - base[k] for k in METRICS}}})
    return {"historical_videos": list(sequences), "historical_seeds": list(seeds),
        "actual_replay_cells_NOT_videos": len(sequences) * (1 + len(seeds)),
        "independent_video_cluster_count_UPPER_BOUND": len(sequences),
        "same_recording_or_person_independence_NOT_proven": True,
        "per_video": per_video, "paired_all_seed_equal_video_delta": video_macro(cells, seeds, sequences),
        "macro_C0_metrics": {k: float(np.mean([v["C0"][k] for v in per_video.values()])) for k in METRICS},
        "macro_FULL_all_seed_metrics": {k: float(np.mean([v["ACIB_FULL_all_seed_mean"][k] for v in per_video.values()])) for k in METRICS},
        "original_combined_metric_summaries_separate_NOT_equal_video_means": actual,
        "fresh_FIT_INNER_or_confirmation_population": False,
        "historical_density_NOT_imputed_from_fresh_video_labels": True,
        "scientific_generalization_PASS": False}
