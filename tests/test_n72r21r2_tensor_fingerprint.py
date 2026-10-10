import numpy as np
from sam3_intermot.one_click.causal_state_fingerprint import full_tracker_fingerprint, tracker_state
from sam3_intermot.one_click.mot_bridge import MOTIdentityBridge
from sam3_intermot.association.causal_identity_tracker import state_digest


def bridge():
    vector = np.zeros(512, np.float32)
    vector[0] = 1.
    result = MOTIdentityBridge({"event_frame": 0, "human_anchor": vector, "target_candidate_uid": "start", "target_box_xyxy": [0, 0, 20, 80]}, frames=10)
    result.configure_fps(20.)
    result.step(0, [{"candidate_uid": "start", "feature": vector, "box_xyxy": [0, 0, 20, 80], "native_tid": 1}])
    return result


def test_tensor_mutations_detected_even_when_legacy_semantic_hash_same():
    current = bridge()
    before = full_tracker_fingerprint(current)
    semantic = state_digest(current.tracker.states)
    state = current.tracker.states[current.tracker.target_public]
    state.prototype[1] = .25
    state.last_box[0] = .5
    state.velocity[1] = 2.
    assert state_digest(current.tracker.states) == semantic
    assert full_tracker_fingerprint(current) != before


def test_full_snapshot_clone_does_not_share_mutable_tensors():
    current = bridge()
    cloned_tracker = current.tracker.clone()
    state = cloned_tracker.states[cloned_tracker.target_public]
    state.prototype[1] = .5
    assert current.tracker.states[current.tracker.target_public].prototype[1] == 0.
    assert "prototype" in tracker_state(current)["states"][str(current.tracker.target_public)]
