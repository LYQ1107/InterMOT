"""Compact, lossless decision evidence and video-cluster development selection.

Frame components, overlapping onset windows and repeated seeds are never
independent correction roots. This module cannot authorize confirmation.
"""
from copy import deepcopy
import math
import numpy as np
from sam3_intermot.one_click.causal_state_fingerprint import fingerprint, serial

METRICS = ("HOTA", "AssA", "DetA", "LocA", "IDF1", "IDSW", "MOTA", "FP", "FN")
DECISION_FIELDS = ("frame", "outputs", "target_uid", "target_public_id", "selected_action",
                   "state_before", "state_after", "births", "deaths")


def compact_policy_result(result, *, primary_vector=None, past_vectors=None, sample=False):
    """Never drop global outputs/actions/tensor seals; don't duplicate matrices."""
    row = {key: deepcopy(result[key]) for key in DECISION_FIELDS}
    for key in ("full_tracker_state_before_sha256", "full_tracker_state_after_sha256",
                "full_learned_state_after_sha256"):
        if key in result:
            row[key] = result[key]
    row.update(identity_memory_write=bool(result["joint_identity_memory_write"]),
               identity_memory_write_UID=result["joint_memory_write_candidate_uid"],
               runtime_GT_read=False, runtime_future_GT_used=False)
    authority = result.get("authority", {})
    effective = bool(authority.get("effective_assignment_change", False))
    keep_outputs = authority.get("own_unmodified_KEEP_outputs", result["outputs"])
    public = result["target_public_id"]
    keep_uid = next((o["candidate_uid"] for o in keep_outputs if o["public_id"] == public), None)
    row["authority"] = {key: deepcopy(authority[key]) for key in
        ("chosen_branch", "approved", "prediction", "features", "point", "support_ablation_policy") if key in authority}
    row["authority"].update(effective_assignment_change=effective,
        own_KEEP_target_uid=keep_uid, own_KEEP_global_outputs_sha256=fingerprint(keep_outputs),
        own_KEEP_outputs=deepcopy(keep_outputs) if effective else None,
        causal_feature_sample=bool(sample or effective))
    if sample or effective:
        row["authority"]["all_current_choices"] = deepcopy(authority.get("all_current_choices", []))
        row["authority"]["own_primary_feature_vector"] = deepcopy(primary_vector)
        row["authority"]["causal_previous_feature_vectors_H3"] = deepcopy(past_vectors)
    return serial(row)


def assert_decision_preserved(original, compact):
    for key in DECISION_FIELDS:
        if serial(original[key]) != compact[key]:
            raise AssertionError("Decision evidence changed: " + key)
    for key in ("full_tracker_state_before_sha256", "full_tracker_state_after_sha256"):
        if key in original and original[key] != compact[key]:
            raise AssertionError("Tensor evidence changed")
    a = original.get("authority", {})
    keep = a.get("own_unmodified_KEEP_outputs", original["outputs"])
    assert fingerprint(keep) == compact["authority"]["own_KEEP_global_outputs_sha256"]
    if a.get("effective_assignment_change"):
        assert serial(keep) == compact["authority"]["own_KEEP_outputs"]


def density_from_counts(counts):
    counts = list(counts)
    if not counts or any(int(n) != n or n < 0 for n in counts):
        raise ValueError("Original nonempty candidate-count frame axis required")
    groups = {"SPARSE": sum(n <= 4 for n in counts), "MEDIUM": sum(5 <= n <= 8 for n in counts),
              "CROWDED": sum(n >= 9 for n in counts)}
    return {"original_frames": len(counts), "counts": groups,
            "fractions": {k: v / len(counts) for k, v in groups.items()},
            "whole_video_density": next((k for k, v in groups.items() if v >= .8 * len(counts)), "MIXED"),
            "GT_used": False, "not_posthoc_masked_TrackEval": True}


def video_macro(cells, seeds, expected_sequences):
    """Average clicks, then all registered seeds inside video, not across rows."""
    by_video = {}
    for sequence in expected_sequences:
        selected = [c for c in cells if c["sequence"] == sequence]
        if len(selected) != len(seeds) or {c["seed"] for c in selected} != set(seeds):
            raise ValueError("Every fixed video and seed must be retained")
        seed_values = []
        for cell in selected:
            delta = list(cell["paired_deltas_vs_C0"].values())
            if not delta:
                if cell["valid_initializations"] != 0:
                    raise ValueError("Missing actual full-video metrics")
                continue  # Failed initialization retained, never an invented zero.
            if any(set(d) != set(METRICS) or not all(math.isfinite(d[k]) for k in METRICS) for d in delta):
                raise ValueError("All nine actual finite metrics required")
            seed_values.append({k: float(np.mean([d[k] for d in delta])) for k in METRICS})
        if seed_values:
            if len(seed_values) != len(seeds):
                raise ValueError("Do not average only the best successful seeds")
            by_video[sequence] = {k: float(np.mean([v[k] for v in seed_values])) for k in METRICS}
    if not by_video:
        return {"per_video": {}, "macro": None, "bootstrap95": None}
    names = sorted(by_video)
    matrix = np.asarray([[by_video[n][k] for k in METRICS] for n in names])
    rng = np.random.default_rng(730104)
    boot = matrix[rng.integers(len(names), size=(2000, len(names)))].mean(axis=1)
    return {"per_video": by_video, "macro": {k: float(v) for k, v in zip(METRICS, matrix.mean(axis=0), strict=True)},
            "bootstrap95": {k: np.quantile(boot[:, i], [.025, .975]).tolist() for i, k in enumerate(METRICS)},
            "improved_HOTA_videos": sum(v["HOTA"] > 0 for v in by_video.values()),
            "degraded_HOTA_videos": sum(v["HOTA"] < 0 for v in by_video.values()),
            "worst_video_delta_HOTA": float(matrix[:, 0].min()),
            "independent_cluster": "VIDEO; clicks and seeds averaged inside each video"}


def choose_inner_point(by_point, seeds, sequences):
    """Shared operating point for all seeds; development choice, NEVER G1 PASS."""
    candidates = []
    for name, cells in sorted(by_point.items()):
        aggregate = video_macro(cells, seeds, sequences)
        actions = sum(c["effective_direct_decisions_NOT_independent_onsets"] for c in cells)
        nonempty_videos = {c["sequence"] for c in cells if c["effective_direct_decisions_NOT_independent_onsets"]}
        components = [v for c in cells for v in c["per_episode_target_components"].values()]
        n01, n10 = (sum(r.get(key, 0) for r in components) for key in ("N01_frames", "N10_frames"))
        # These are frame/onset proxies, not independent G1 event qualifications.
        severe = sum(c["own_onset_summary"]["severe_non_target_harm_decisions"] for c in cells)
        risk = sum(c["own_onset_summary"]["risky_decisions"] for c in cells)
        correct = sum(c["own_onset_summary"]["beneficial_complete_H100_decisions"] for c in cells)
        incomplete = sum(c["own_onset_summary"].get("incomplete_H100_decisions", 0) for c in cells)
        eligible = bool(actions and len(nonempty_videos) >= 3 and n01 > n10 and correct and not severe and not risk and not incomplete)
        candidates.append({"point": name, "aggregate": aggregate, "effective_decisions": actions,
            "nonempty_videos": len(nonempty_videos), "N01_frames": n01, "N10_frames": n10,
            "risky_decisions": risk, "severe_non_target_harm_decisions": severe,
            "incomplete_H100_decisions": incomplete,
            "beneficial_complete_H100_decisions": correct, "development_nonvacuous_harm_screen": eligible})
    if not candidates:
        raise ValueError("No complete operating points")
    def order(c):
        macro = c["aggregate"]["macro"]
        return (not c["development_nonvacuous_harm_screen"], c["severe_non_target_harm_decisions"],
                c["risky_decisions"], c["incomplete_H100_decisions"], -(macro["HOTA"] if macro else -1.),
                -(macro["AssA"] if macro else -1.), -(macro["IDF1"] if macro else -1.), c["point"])
    chosen = min(candidates, key=order)
    return {"selected_point": chosen["point"], "candidates": candidates,
            "status": "INNER_SHARED_POINT_DEVELOPMENT_ONLY" if chosen["development_nonvacuous_harm_screen"] else
                      "NO_NONVACUOUS_POINT_FIXED_DIAGNOSTIC_SELECTION_NOT_QUALIFIED",
            "all_seeds_retained": list(seeds), "independent_G1_roots_proven": False,
            "confirmation_authorized": False, "next_stage_authorized": False}
