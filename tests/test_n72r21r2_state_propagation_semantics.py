"""Synthetic two-person unit probes of real global commit/state feedback.

No GT fields enter runtime rows. Symbolic p/q truth is confined to assertions;
these are not scientific sequences, correction roots or MOT results.
"""
import numpy as np
from tests.test_n72r21r2_matched_event_observer import setup
from sam3_intermot.one_click.safe_mot_bridge import SafeMOTIdentityBridge
from sam3_intermot.one_click.intervention_gate import GatePolicy
from sam3_intermot.one_click.joint_intervention_primitives import prepare_proposal, commit_action, step_keep
from sam3_intermot.association.opportunity_solver import action_for_candidate
from sam3_intermot.evaluation.complete_mot_state import complete_mot_fingerprint


def scene(native_confusion):
    click, actor, positive = setup()
    other = np.zeros(512, np.float32)
    other[1] = 1.
    initial = [dict(candidate_uid="0", feature=positive, box_xyxy=[0, 0, 20, 80], native_tid=1, conf=.99),
               dict(candidate_uid="q0", feature=other, box_xyxy=[60, 0, 80, 80], native_tid=2, conf=.99)]
    root = SafeMOTIdentityBridge(click, actor, policy=GatePolicy(family="shadow"), frames=5)
    root.configure_fps(20.)
    root.step(0, initial)
    current = [{**row, "candidate_uid": "p1" if i == 0 else "q1",
                "native_tid": 3 - row["native_tid"] if native_confusion else row["native_tid"]}
               for i, row in enumerate(initial)]
    future = [{**row, "candidate_uid": row["candidate_uid"][0] + "2"} for row in current]
    return root, current, future


def global_pair(native_confusion, selected):
    root, current, future = scene(native_confusion)
    before = complete_mot_fingerprint(root)
    changed, own_keep = root.clone(), root.clone()
    prepared = prepare_proposal(changed, 1, current)
    action = action_for_candidate(changed.tracker.target_public, selected, prepared["preview"]["solver"])
    actual = commit_action(changed, 1, current, prepared, action)
    reference = step_keep(own_keep, 1, current)
    assert actual["global_assignment_changed"] and action.family == "GLOBAL_SWAP"
    assert {o["candidate_uid"] for o in actual["outputs"]} == {row["candidate_uid"] for row in current}
    assert len({o["public_id"] for o in actual["outputs"]}) == len(current) == 2
    assert not actual["runtime_gt_read"] and not actual["runtime_future_gt_used"]
    assert not actual["joint_identity_memory_write"] and actual["extra_clicks"] == 0
    assert complete_mot_fingerprint(root) == before
    assert complete_mot_fingerprint(changed) != complete_mot_fingerprint(own_keep)
    return root, changed, own_keep, current, future, actual, reference


def test_correct_global_swap_updates_real_state_and_future_KEEP_recovers_target():
    root, changed, own_keep, current, future, actual, reference = global_pair(True, "p1")
    assert reference["target_uid"] == "q1" and actual["target_uid"] == "p1"
    state = changed.tracker.states[changed.tracker.target_public]
    assert state.last_native_tid == 2 and state.last_seen_frame == 1
    assert np.array_equal(state.last_box, current[0]["box_xyxy"])
    assert np.array_equal(state.prototype, root.tracker.states[root.tracker.target_public].prototype)
    assert changed.identity.last_frame == 1 and list(changed.identity.last_box) == current[0]["box_xyxy"]
    assert not np.shares_memory(state.prototype, own_keep.tracker.states[own_keep.tracker.target_public].prototype)
    repaired, unrepaired = step_keep(changed, 2, future), step_keep(own_keep, 2, future)
    assert repaired["target_uid"] == "p2" and unrepaired["target_uid"] == "q2"
    assert repaired["state_after"] != unrepaired["state_after"]


def test_wrong_global_swap_updates_real_state_and_causes_future_target_loss():
    root, changed, own_keep, current, future, actual, reference = global_pair(False, "q1")
    assert reference["target_uid"] == "p1" and actual["target_uid"] == "q1"
    state = changed.tracker.states[changed.tracker.target_public]
    assert state.last_native_tid == 2 and state.last_seen_frame == 1
    assert np.array_equal(state.last_box, current[1]["box_xyxy"])
    assert state.prototype[1] > 0. and root.tracker.states[root.tracker.target_public].prototype[1] == 0.
    assert changed.identity.last_frame == 1 and list(changed.identity.last_box) == current[1]["box_xyxy"]
    harmed, untouched = step_keep(changed, 2, future), step_keep(own_keep, 2, future)
    assert untouched["target_uid"] == "p2" and harmed["target_uid"] is None
    assert state.last_seen_frame == 1 and state.state == state.LOST
    assert harmed["state_after"] != untouched["state_after"]
    assert len(harmed["outputs"]) == len({o["candidate_uid"] for o in harmed["outputs"]}) == 2
