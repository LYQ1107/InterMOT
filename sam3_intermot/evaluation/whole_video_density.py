"""Whole-video density tables: clicks/seeds never become extra videos."""
from collections import defaultdict
import math
import numpy as np
from .learned_policy_evidence import METRICS

DENSITIES = ("SPARSE", "MEDIUM", "CROWDED", "MIXED")


def full_video_summary(rows, planned_sequences, *, expected_seeds=("FIXED",), bootstrap_seed=730104, repetitions=2000):
    if not planned_sequences or len(set(planned_sequences)) != len(planned_sequences):
        raise ValueError("Frozen unique planned video census required")
    by_video = defaultdict(lambda: defaultdict(list))
    seen = set()
    for row in rows:
        if row["sequence"] not in planned_sequences or row["density"] not in DENSITIES:
            raise ValueError("No unplanned video or retuned density label")
        unique = row["sequence"], row["seed"], row["episode_uid"]
        if unique in seen:
            raise ValueError("Repeated click/seed cannot increase video support")
        seen.add(unique)
        if set(row["metrics"]) != set(METRICS) or set(row["deltas"]) != set(METRICS):
            raise ValueError("All nine actual full-video metrics and deltas required")
        if not all(math.isfinite(row[group][key]) for group in ("metrics", "deltas") for key in METRICS):
            raise ValueError("Actual finite full-video metrics required")
        if any(isinstance(v, bool) or not math.isfinite(v) or int(v) != v or v < 0 for v in row.get("counts", {}).values()):
            raise ValueError("Counts must be actual nonnegative integer exposures, not rates")
        by_video[row["sequence"]][row["seed"]].append(row)
    videos = []
    for sequence, seed_rows in sorted(by_video.items()):
        if set(seed_rows) != set(expected_seeds):
            raise ValueError("All registered seeds required; no best successful subset")
        flat = [r for values in seed_rows.values() for r in values]
        labels = {r["density"] for r in flat}
        if len(labels) != 1:
            raise ValueError("One video cannot change density between policies/seeds")
        click_sets = [{r["episode_uid"] for r in values} for values in seed_rows.values()]
        if any(clicks != click_sets[0] for clicks in click_sets):
            raise ValueError("Seeds may not silently retain different valid clicks")
        video = {"sequence": sequence, "density": flat[0]["density"],
            "actual_seed_replicas": len(seed_rows), "sole_clicks": len(click_sets[0])}
        for group in ("metrics", "deltas"):
            video[group] = {key: float(np.mean([np.mean([r[group][key] for r in values]) for values in seed_rows.values()])) for key in METRICS}
        counts = sorted({key for r in flat for key in r.get("counts", {})})
        video["seed_mean_counts_clicks_summed"] = {key: float(np.mean([sum(r.get("counts", {}).get(key, 0) for r in values) for values in seed_rows.values()])) for key in counts}
        values = video["seed_mean_counts_clicks_summed"]
        correct = values.get("target_correct_frames", 0.)
        video["target_claim_recall_available"] = correct / values["positive_available_frames"] if values.get("positive_available_frames", 0) > 0 else None
        video["target_claim_recall_visible"] = correct / values["physically_visible_frames"] if values.get("physically_visible_frames", 0) > 0 else None
        videos.append(video)

    def aggregate(subset):
        if not subset:
            return {"usable_video_clusters": 0, "macro_metrics": None, "macro_deltas": None,
                    "paired_video_bootstrap95_delta": None, "insufficient_density_generalization_support": True}
        metric_matrix = np.asarray([[r["metrics"][k] for k in METRICS] for r in subset])
        delta_matrix = np.asarray([[r["deltas"][k] for k in METRICS] for r in subset])
        # A singleton has an actual descriptive mean but no informative cluster CI.
        rng = np.random.default_rng(bootstrap_seed)
        boot = delta_matrix[rng.integers(len(subset), size=(repetitions, len(subset)))].mean(1) if len(subset) >= 2 else None
        return {"usable_video_clusters": len(subset),
            "macro_metrics": dict(zip(METRICS, metric_matrix.mean(0).tolist(), strict=True)),
            "macro_deltas": dict(zip(METRICS, delta_matrix.mean(0).tolist(), strict=True)),
            "paired_video_bootstrap95_delta": {k: np.quantile(boot[:, i], [.025, .975]).tolist() for i, k in enumerate(METRICS)} if boot is not None else None,
            "insufficient_density_generalization_support": len(subset) < 3,
            "improved_HOTA_videos": sum(r["deltas"]["HOTA"] > 0 for r in subset),
            "degraded_HOTA_videos": sum(r["deltas"]["HOTA"] < 0 for r in subset),
            "minimum_video_delta_HOTA": float(delta_matrix[:, 0].min()),
            "macro_target_claim_recall_available": float(np.mean([r["target_claim_recall_available"] for r in subset if r["target_claim_recall_available"] is not None])) if any(r["target_claim_recall_available"] is not None for r in subset) else None,
            "macro_target_claim_recall_visible": float(np.mean([r["target_claim_recall_visible"] for r in subset if r["target_claim_recall_visible"] is not None])) if any(r["target_claim_recall_visible"] is not None for r in subset) else None,
            "counts_summed_over_video_exposures_NOT_independent_events": {key: sum(r["seed_mean_counts_clicks_summed"].get(key, 0) for r in subset) for key in sorted({key for r in subset for key in r["seed_mean_counts_clicks_summed"]})}}
    return {"planned_sequences": list(planned_sequences), "usable_sequences": [r["sequence"] for r in videos],
        "missing_or_failed_initialization_videos_retained": [s for s in planned_sequences if s not in by_video],
        "per_video": videos, "overall": aggregate(videos),
        "by_frozen_whole_video_density": {label: aggregate([r for r in videos if r["density"] == label]) for label in DENSITIES},
        "aggregation": "Equal videos; mean clicks inside seed, then mean all seed replicas inside video; frame counts sum clicks and average seeds, not independent-event counts",
        "bootstrap_seed": bootstrap_seed, "bootstrap_resamples": repetitions,
        "whole_original_video_metrics_NOT_posthoc_masked": True,
        "independent_confirmation_or_person_disjointness_claim": False}
