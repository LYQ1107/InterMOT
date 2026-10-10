"""Exact own-policy onset pairs; future branches never follow teacher actions."""
from copy import deepcopy
from .causal_state_fingerprint import fingerprint, full_tracker_fingerprint
from .matched_event_observer import causal_bridge_fingerprint
from .joint_intervention_primitives import step_keep


def complete_learned_state_fingerprint(bridge):
    return fingerprint({"causal_core": causal_bridge_fingerprint(bridge),
                        "event_feature_history": deepcopy(bridge.event_feature_history),
                        "event_branch_pending": deepcopy(bridge.event_branch_pending),
                        "point": deepcopy(bridge.point), "support_policy": getattr(bridge, "support_policy", None)})


def chronological_nonoverlapping_candidates(frames, horizon=100):
    result, next_allowed = [], -1
    for frame in sorted(set(int(f) for f in frames)):
        if frame >= next_allowed:
            result.append(frame)
            next_allowed = frame + horizon + 1
    return result  # Candidate spacing is NOT proof of distinct causal origins.


def onset_pair(bridge, frames, frame, observed, compact, horizon=100):
    if frame != bridge.tracker.frame + 1 or frame <= bridge.event_frame or int(observed["frame"]) != frame:
        raise ValueError("Exact next original post-click frame required")
    before = complete_learned_state_fingerprint(bridge)
    assert full_tracker_fingerprint(bridge) == observed["full_tracker_state_before_sha256"]
    if "full_learned_state_before_sha256" in observed:
        assert before == observed["full_learned_state_before_sha256"]
    branches = {}
    for name in ("OWN_KEEP", "ACTUAL_CURRENT_THEN_KEEP"):
        arm = bridge.clone()
        assert complete_learned_state_fingerprint(arm) == before
        rows = []
        for payload, candidates in frames[frame:min(frame + horizon + 1, len(frames))]:
            current = int(payload["frame"])
            assert current == arm.tracker.frame + 1
            start = full_tracker_fingerprint(arm)
            result = arm.step(current, candidates) if current == frame and name == "ACTUAL_CURRENT_THEN_KEEP" else step_keep(arm, current, candidates)
            if current == frame and name == "ACTUAL_CURRENT_THEN_KEEP":
                assert result["outputs"] == observed["outputs"] and result["selected_action"] == observed["selected_action"]
                assert full_tracker_fingerprint(arm) == observed["full_tracker_state_after_sha256"]
                if "full_learned_state_after_sha256" in observed:
                    assert complete_learned_state_fingerprint(arm) == observed["full_learned_state_after_sha256"]
            rows.append(compact(result, arm, start))
        branches[name] = rows
        assert complete_learned_state_fingerprint(bridge) == before
    return branches
