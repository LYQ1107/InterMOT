"""Offline paired raw components; never inputs to an online decision."""
from collections import Counter
import math
from .causal_identity_events import candidate_identity_outcome
from .causal_identity_windows import future_window
from .safe_intervention_events import assignment_map, contiguous_intervals

SUM_FIELDS = ("target_correct", "KEEP_correct", "N01", "N10", "verified_OTHER", "KEEP_verified_OTHER", "UNKNOWN", "NONE",
              "positive_available", "physically_visible", "non_target_damage", "non_target_benefit", "non_target_verified_OTHER_damage",
              "non_target_UNKNOWN_damage", "non_target_NONE_damage", "same_target_fragment_displacements", "ownership_changed",
              "births", "deaths", "memory_writes", "correct_writes", "verified_OTHER_writes", "UNKNOWN_writes", "native_diff_vs_KEEP")


def distance(a, b):
    return math.sqrt(sum((float(x) - float(y)) ** 2 for x, y in zip(a, b, strict=True)))


def label_actual_branch(actual_rows, keep_rows, matched, target, prestate_origins, visible):
    if not actual_rows or len(actual_rows) != len(keep_rows):
        raise ValueError("Paired original frame axis required")
    start = int(keep_rows[0]["frame"])
    public = int(keep_rows[0]["target_public_id"])
    components = []
    target_publics, keep_target_publics = [], []
    for offset, (actual, keep) in enumerate(zip(actual_rows, keep_rows, strict=True)):
        frame = int(actual["frame"])
        assert frame == keep["frame"] == start + offset
        truth = matched[frame]
        am, bm = assignment_map(actual), assignment_map(keep)
        assert len(am) == len(actual["outputs"]) and len(set(am.values())) == len(am)
        assert set(am.values()) == set(bm.values()) == set(truth), "All current candidates must be globally owned"
        uid, base_uid = am.get(public), bm.get(public)
        outcome = candidate_identity_outcome(uid, truth, target)
        base = candidate_identity_outcome(base_uid, truth, target)
        harmed, helped, fragments = [], [], []
        harmful_outcomes = Counter()
        for p, identity in prestate_origins.items():
            p = int(p)
            if p == public:
                continue
            a, b = truth.get(am.get(p)), truth.get(bm.get(p))
            if identity == target:
                if b == identity and a != identity:
                    fragments.append(p)
                continue
            if b == identity and a != identity:
                harmed.append(p)
                harm = candidate_identity_outcome(am.get(p), truth, identity)
                harmful_outcomes[harm] += 1
            if b != identity and a == identity:
                helped.append(p)
        write = bool(actual["identity_memory_write"])
        write_uid = actual["identity_memory_write_UID"]
        if write and write_uid != uid:
            raise ValueError("Memory write must reference actually committed target UID")
        write_outcome = candidate_identity_outcome(write_uid, truth, target) if write else None
        if write and write_uid is None:
            raise ValueError("NONE cannot write an observation")
        motion = actual.get("target_motion_state")
        keep_motion = keep.get("target_motion_state")
        full_motion = motion is not None and keep_motion is not None
        actor_box, keep_actor_box = actual.get("actor_state", {}).get("last_box"), keep.get("actor_state", {}).get("last_box")
        components.append({"frame": frame, "offset": offset, "outcome": outcome, "KEEP_outcome": base,
                           "target_correct": int(outcome == "TARGET"), "KEEP_correct": int(base == "TARGET"),
                           "N01": int(outcome == "TARGET" and base != "TARGET"), "N10": int(base == "TARGET" and outcome != "TARGET"),
                           "verified_OTHER": int(outcome == "VERIFIED_OTHER"), "KEEP_verified_OTHER": int(base == "VERIFIED_OTHER"),
                           "UNKNOWN": int(outcome == "UNKNOWN"), "NONE": int(outcome == "NONE"),
                           "positive_available": int(target in truth.values()), "physically_visible": int(visible.get(frame, False)),
                           "non_target_damage": len(harmed), "non_target_benefit": len(helped),
                           "non_target_verified_OTHER_damage": harmful_outcomes["VERIFIED_OTHER"],
                           "non_target_UNKNOWN_damage": harmful_outcomes["UNKNOWN"], "non_target_NONE_damage": harmful_outcomes["NONE"],
                           "harmed_public_ids": sorted(harmed), "helped_public_ids": sorted(helped),
                           "same_target_fragment_displacements": len(fragments), "ownership_changed": int(am != bm),
                           "births": len(actual["births"]), "deaths": len(actual["deaths"]), "memory_writes": int(write),
                           "correct_writes": int(write_outcome == "TARGET"), "verified_OTHER_writes": int(write_outcome == "VERIFIED_OTHER"),
                           "UNKNOWN_writes": int(write_outcome == "UNKNOWN"), "native_diff_vs_KEEP": int(actual["native_after"] != keep["native_after"]),
                           "prototype_anchor_cosine_difference": abs(actual["prototype_anchor_cosine"] - keep["prototype_anchor_cosine"]),
                           "prototype_vector_changed_vs_KEEP": None if "target_prototype_sha256" not in actual else actual["target_prototype_sha256"] != keep["target_prototype_sha256"],
                           "motion_box_L2_vs_KEEP": distance(motion["last_box"], keep_motion["last_box"]) if full_motion else None,
                           "motion_velocity_L2_vs_KEEP": distance(motion["velocity"], keep_motion["velocity"]) if full_motion else None,
                           "actor_box_L2_proxy_vs_KEEP": distance(actor_box, keep_actor_box) if actor_box is not None and keep_actor_box is not None else None,
                           "motion_measurement_status": "ACTUAL_TRACKER_BOX_AND_VELOCITY" if full_motion else "NOT_AVAILABLE_V1_TRACE_ACTOR_BOX_PROXY_ONLY",
                           "tracker_semantic_divergence_vs_KEEP": actual["state_after"] != keep["state_after"],
                           "tracker_tensor_divergence_vs_KEEP": None if "full_tracker_state_after_sha256" not in actual else actual["full_tracker_state_after_sha256"] != keep["full_tracker_state_after_sha256"]})
        target_publics.append(next((p for p, u in am.items() if truth[u] == target), None))
        keep_target_publics.append(next((p for p, u in bm.items() if truth[u] == target), None))
    windows = {}
    for horizon in (1, 5, 20, 50, 100):
        rows, complete = future_window(components, start, horizon)
        counts = {key: sum(r[key] for r in rows) for key in SUM_FIELDS}
        affected = Counter(p for r in rows for p in r["harmed_public_ids"])
        severe = sorted(p for p, n in affected.items() if n >= 20)
        target_value = counts["N01"] - counts["N10"]
        risk = bool(counts["N10"] or counts["non_target_damage"] or counts["verified_OTHER_writes"] or counts["UNKNOWN_writes"])
        global_proxy = target_value - counts["non_target_damage"] + counts["non_target_benefit"] - counts["verified_OTHER_writes"] - counts["UNKNOWN_writes"]
        windows["H" + str(horizon)] = {**counts, "complete": complete, "observed_future_frames": len(rows), "future_offsets": [1, horizon],
                                      "affected_public_frame_counts": dict(affected), "severe_non_target_harm_public_ids": severe,
                                      "raw_target_value": target_value, "raw_global_component_proxy": global_proxy,
                                      "risk_label": risk if complete else None, "benefit_label": bool(global_proxy > 0 and not risk) if complete else None,
                                      "normalized_target_value_label": target_value / horizon if complete else None,
                                      "normalized_global_proxy_label": global_proxy / horizon if complete else None,
                                      "not_actual_HOTA_AssA_or_full_policy_reward": True,
                                      "max_prototype_anchor_cosine_difference": max((r["prototype_anchor_cosine_difference"] for r in rows), default=0.),
                                      "max_motion_box_L2_vs_KEEP": max((r["motion_box_L2_vs_KEEP"] for r in rows if r["motion_box_L2_vs_KEEP"] is not None), default=None),
                                      "max_motion_velocity_L2_vs_KEEP": max((r["motion_velocity_L2_vs_KEEP"] for r in rows if r["motion_velocity_L2_vs_KEEP"] is not None), default=None)}
    takeover = contiguous_intervals(r["frame"] for r in components if r["outcome"] == "VERIFIED_OTHER")
    changed = [r["frame"] for r in components if r["ownership_changed"]]
    return {"current_t": components[0], "future": windows, "raw_frame_components": components,
            "first_global_ownership_divergent_frame": min(changed, default=None),
            "first_semantic_state_divergent_frame": next((r["frame"] for r in components if r["tracker_semantic_divergence_vs_KEEP"]), None),
            "first_tensor_state_divergent_frame": next((r["frame"] for r in components if r["tracker_tensor_divergence_vs_KEEP"]), None),
            "first_future_target_recovery_frame": next((r["frame"] for r in components[1:] if r["target_correct"]), None),
            "verified_OTHER_takeover_intervals": takeover, "max_observed_verified_OTHER_takeover_duration": max((r["frames"] for r in takeover), default=0),
            "N01_propagated_intervals_not_action_onsets": contiguous_intervals(r["frame"] for r in components[1:] if r["N01"]),
            "N10_propagated_intervals_not_action_onsets": contiguous_intervals(r["frame"] for r in components[1:] if r["N10"]),
            "target_fragmentation_public_transitions": sum(a is not None and b is not None and a != b for a, b in zip(target_publics, target_publics[1:])),
            "KEEP_target_fragmentation_public_transitions": sum(a is not None and b is not None and a != b for a, b in zip(keep_target_publics, keep_target_publics[1:])),
            "prestate_origin_proxy_not_global_IDF1": True, "UNKNOWN_not_verified_OTHER": True,
            "overlapping_branch_windows_not_independent_correction_events": True}
