"""Complete-grid full-MOT evidence, never a frame-as-event qualification."""
from collections import Counter
import math
import numpy as np
from .learned_policy_evidence import METRICS
from .whole_video_density import full_video_summary

ONSET_KEYS = ("effective_decisions", "complete_H100_decisions", "incomplete_H100_decisions",
              "beneficial_complete_H100_decisions", "risky_decisions", "severe_non_target_harm_decisions")


def count(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0 or int(value) != value:
        raise ValueError("Actual nonnegative integer exposure counts required")
    return int(value)


def onset_summary(cell):
    """Recompute raw one-shot labels, not adaptive-policy rewards or roots."""
    calculated = Counter({key: 0 for key in ONSET_KEYS})
    episodes = cell["own_prefix_one_shot_labels"]
    if len({e["episode_uid"] for e in episodes}) != len(episodes):
        raise ValueError("No duplicate click labels")
    for episode in episodes:
        if len({e["frame"] for e in episode["onsets"]}) != len(episode["onsets"]):
            raise ValueError("No duplicate one-shot action labels")
        for onset in episode["onsets"]:
            labels = onset["labels"]
            future = labels["future"]["H100"]
            rows = labels["raw_frame_components"]
            if not rows or rows[0]["frame"] != onset["frame"] or any(r["frame"] != onset["frame"] + i for i, r in enumerate(rows)):
                raise ValueError("Actual paired original one-shot frame axis required")
            complete = bool(future["complete"])
            if future["observed_future_frames"] != len(rows) - 1 or (complete and len(rows) != 101):
                raise ValueError("Complete H100 needs current plus100 actual future frames")
            risk = any(any(count(r[k]) for k in ("N10", "non_target_damage", "verified_OTHER_writes", "UNKNOWN_writes")) for r in rows)
            calculated["effective_decisions"] += 1
            calculated["complete_H100_decisions"] += complete
            calculated["incomplete_H100_decisions"] += not complete
            calculated["beneficial_complete_H100_decisions"] += bool(complete and future["benefit_label"] and not risk)
            calculated["risky_decisions"] += risk
            calculated["severe_non_target_harm_decisions"] += bool(future["severe_non_target_harm_public_ids"])
    if set(cell["own_onset_summary"]) != set(ONSET_KEYS) or any(count(cell["own_onset_summary"][k]) != calculated[k] for k in ONSET_KEYS):
        raise ValueError("Reported one-shot onset counters must equal all actual raw labels")
    if count(cell["effective_direct_decisions_NOT_independent_onsets"]) != calculated["effective_decisions"]:
        raise ValueError("Every actual effective decision needs its own-prefix audit")
    if {e["episode_uid"] for e in episodes} != set(cell["per_episode_target_components"]):
        raise ValueError("Every valid click needs both full-video and own-onset components")
    return dict(calculated)


def summarize_complete_group(cells, baselines, initializations, densities, *, seeds, sequences):
    if not seeds or len(set(seeds)) != len(seeds) or not sequences or len(set(sequences)) != len(sequences):
        raise ValueError("Frozen unique seeds and full video census required")
    expected = {(s, seed) for s in sequences for seed in seeds}
    keys = [(c["sequence"], c["seed"]) for c in cells]
    if len(keys) != len(set(keys)) or set(keys) != expected:
        raise ValueError("Exactly all registered videos/seeds required; no successful subset")
    rows, base_rows, per_seed = [], [], {seed: {"onsets_NOT_independent": Counter(), "counts_NOT_independent": Counter(), "action_videos": set()} for seed in seeds}
    for cell in cells:
        sequence, seed = cell["sequence"], cell["seed"]
        if cell["density"]["whole_video_density"] != densities[sequence]:
            raise ValueError("Original frozen density cannot change with model effects")
        events = {e["episode_uid"]: e for e in initializations[sequence]["inputs"] if not e["initialization_failure"]}
        if cell["valid_initializations"] != len(events) or set(cell["per_episode_target_components"]) != set(events):
            raise ValueError("Valid/failed original clicks cannot be silently replaced")
        expected_trackers = {cell["experiment_uid"] + "__click" + str(e["slot"]) for e in events.values()}
        if set(cell["all_nine_metrics"]) != expected_trackers or set(cell["paired_deltas_vs_C0"]) != expected_trackers:
            raise ValueError("Actual nine-metric tracker coverage must equal all valid clicks")
        onsets = onset_summary(cell)
        per_seed[seed]["onsets_NOT_independent"].update(onsets)
        if onsets["effective_decisions"]:
            per_seed[seed]["action_videos"].add(sequence)
        for episode, event in events.items():
            name = cell["experiment_uid"] + "__click" + str(event["slot"])
            reported = cell["all_nine_metrics"][name]
            if not set(METRICS) <= set(reported) or set(reported) - set(METRICS) - {"tracker", "per_sequence"}:
                raise ValueError("Nine scalar metrics plus explicitly separated pinned metadata required")
            metrics = {k: reported[k] for k in METRICS}
            base = baselines[sequence]["actual_nine_metrics"]["CLICK_C0__click" + str(event["slot"])]
            if set(metrics) != set(METRICS) or any(k not in base or not math.isfinite(metrics[k]) or not math.isfinite(base[k]) for k in METRICS):
                raise ValueError("All nine finite full-video metrics required")
            for values in (metrics, base):
                if any(not 0 <= values[k] <= 1 for k in ("HOTA", "AssA", "DetA", "LocA", "IDF1")):
                    raise ValueError("Normalized 0-to-1 fraction metrics required, not percentages")
                for k in ("IDSW", "FP", "FN"): count(values[k])
            delta = {k: metrics[k] - base[k] for k in METRICS}
            if delta != cell["paired_deltas_vs_C0"][name]:
                raise ValueError("Paired deltas must use the same exact C0 click input")
            components = cell["per_episode_target_components"][episode]
            if any(count(v) != v for v in components.values()):
                raise ValueError("Frame components are counts, not per-event rates")
            per_seed[seed]["counts_NOT_independent"].update(components)
            # Counter files omit unobserved zero keys; never impute missing exposures.
            available, visible = (components[k] for k in ("positive_available_frames", "physically_visible_frames"))
            actual_counts = {**components, "target_correct_frames": components.get("TARGET_frames", 0),
                             "verified_OTHER_takeover_frames": components.get("VERIFIED_OTHER_frames", 0)}
            base_counts = {"positive_available_frames": available, "physically_visible_frames": visible,
                           "target_correct_frames": components.get("C0_TARGET_frames", 0),
                           "verified_OTHER_takeover_frames": components.get("C0_VERIFIED_OTHER_frames", 0)}
            common = {"sequence": sequence, "seed": seed, "episode_uid": episode, "density": densities[sequence]}
            rows.append({**common, "metrics": metrics, "deltas": delta, "counts": actual_counts})
            base_rows.append({**common, "metrics": {k: base[k] for k in METRICS}, "deltas": {k: 0. for k in METRICS}, "counts": base_counts})
    actual = full_video_summary(rows, sequences, expected_seeds=seeds)
    baseline = full_video_summary(base_rows, sequences, expected_seeds=seeds)
    for seed, values in per_seed.items():
        values["onsets_NOT_independent"] = dict(values["onsets_NOT_independent"])
        values["counts_NOT_independent"] = dict(values["counts_NOT_independent"])
        values["action_videos"] = sorted(values["action_videos"])
    counts = {k: float(np.mean([v["counts_NOT_independent"].get(k, 0) for v in per_seed.values()])) for k in {k for v in per_seed.values() for k in v["counts_NOT_independent"]}}
    onsets = {k: float(np.mean([v["onsets_NOT_independent"][k] for v in per_seed.values()])) for k in ONSET_KEYS}
    return {"actual_full_video": actual, "same_input_full_C0": baseline,
            "per_seed_all_videos_summed_NOT_independent_counts": per_seed,
            "seed_mean_full_video_components_NOT_event_counts": counts,
            "seed_mean_one_shot_decisions_NOT_independent_roots": onsets,
            "independent_G1_event_precision_recall": None,
            "independent_G1_event_metric_status": "UNAVAILABLE_UNTIL_CAUSAL_ROOT_PROOF; not substituted by frames/windows/seeds",
            "confirmation_authorized": False}


def development_gates(summary, gates):
    """Negative evidence is usable; positive proxies never imply root proof."""
    values = summary["actual_full_video"]["overall"]
    delta = values["macro_deltas"]
    base = summary["same_input_full_C0"]["overall"]["macro_metrics"]
    if delta is None:
        return {"G1": {"status": "FAIL_NO_USABLE_INITIALIZATION"}, "G2": {"status": "UNDEFINED_NO_ACTUAL_METRICS"}, "confirmation_authorized": False}
    g2 = gates["G2"]
    counts = summary["seed_mean_full_video_components_NOT_event_counts"]
    checks = {"HOTA": delta["HOTA"] >= g2["delta_HOTA_min"], "AssA": delta["AssA"] > g2["delta_AssA_gt"],
              "DetA": delta["DetA"] >= g2["delta_DetA_min"], "IDF1": delta["IDF1"] >= g2["delta_IDF1_min"],
              "IDSW": delta["IDSW"] <= g2["IDSW_growth_max_fraction"] * base["IDSW"],
              "wrong_takeover": counts.get("VERIFIED_OTHER_frames", 0) - counts.get("C0_VERIFIED_OTHER_frames", 0) <= g2["wrong_person_takeover_growth_max"]}
    failures, per_seed, g1 = [], summary["per_seed_all_videos_summed_NOT_independent_counts"], gates["G1"]
    # Overcounting ALL actions (including duplicate seeds/clicks) is only an
    # upper bound. Even ten such rows cannot establish ten independent roots.
    action_upper_bound = sum(v["onsets_NOT_independent"]["effective_decisions"] for v in per_seed.values())
    video_upper_bound = len({s for v in per_seed.values() for s in v["action_videos"]})
    severe = sum(v["onsets_NOT_independent"]["severe_non_target_harm_decisions"] for v in per_seed.values())
    if action_upper_bound < g1["independent_beneficial_correction_onsets_min"]:
        failures.append("ALL_ACTION_UPPER_BOUND_BELOW_INDEPENDENT_BENEFICIAL_ROOT_MINIMUM")
    if video_upper_bound < g1["supporting_sequences_min"]:
        failures.append("ACTION_VIDEO_UPPER_BOUND_BELOW_MINIMUM")
    if counts.get("N01_frames", 0) <= counts.get("N10_frames", 0):
        failures.append("N01_NOT_GREATER_THAN_N10")
    if severe > g1["severe_non_target_harm_onsets_max"]:
        failures.append("OBSERVED_SEVERE_OTHER_PERSON_HARM")
    ci = values["paired_video_bootstrap95_delta"]
    return {"G1": {"status": "FAIL_OBSERVED_NONVACUITY_OR_HARM_BOUND" if failures else "UNPROVEN_INDEPENDENT_CAUSAL_ROOTS_REQUIRED",
                    "negative_evidence": failures, "independent_roots_proven": False,
                    "all_seed_click_effective_actions_UPPER_BOUND_ONLY": action_upper_bound,
                    "action_video_support_UPPER_BOUND_ONLY": video_upper_bound,
                    "frames_clicks_seeds_or_spaced_windows_NOT_independent_roots": True},
            "G2": {"status": "PASS_NUMERIC_TARGETS_ONLY_NOT_RESEARCH_SUCCESS" if all(checks.values()) else "FAIL_NUMERIC_MOT_TARGETS",
                    "checks": checks, "scale": "0_to_1 fractions; IDSW/FP/FN counts", "paired_video_95_CI_reported_separately": ci,
                    "HOTA_CI_strictly_positive": ci is not None and ci["HOTA"][0] > 0,
                    "zero_C0_IDSW_allows_no_increase": base["IDSW"] == 0},
            "G0": "SEPARATE_ACTUAL_SOURCE_RUNTIME_ENGINEERING_AUDIT_REQUIRED",
            "G3": "NOT_RUN_CONFIRMATION_UNAUTHORIZED", "G4": "SEPARATE_NONZERO_WRITE_SAFETY_AUDIT_REQUIRED",
            "scientific_success": None, "confirmation_authorized": False, "next_association_stage_authorized": False}
