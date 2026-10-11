from copy import deepcopy
from collections import deque
import numpy as np
import pytest
from sam3_intermot.one_click.mot_bridge import MOTIdentityBridge
from sam3_intermot.one_click.causal_state_fingerprint import full_tracker_fingerprint
from sam3_intermot.evaluation.complete_mot_state import canonical, complete_mot_snapshot, complete_mot_fingerprint, assert_complete_mot_AA


def bridge():
    a = np.zeros(512, np.float32); a[0] = 1.
    b = MOTIdentityBridge(dict(event_frame=0, human_anchor=a, target_candidate_uid="a", target_box_xyxy=[0, 0, 20, 80]), frames=8)
    b.configure_fps(20.)
    b.step(0, [dict(candidate_uid="a", feature=a, box_xyxy=[0, 0, 20, 80], native_tid=1)])
    return b


@pytest.mark.parametrize("field,value", [("pending", [{"frame": 1}]), ("native_streak", 4),
    ("observed_native_streak", 4), ("last_native_key", ("s", 1)), ("last_observation_frame", 3),
    ("contradiction_streak", 5), ("trust_snapshot", {"frame": 2}), ("adapter_calls", 1)])
def test_mutations_omitted_by_legacy_subset_are_detected(field, value):
    a, b = bridge(), bridge()
    assert_complete_mot_AA(a, b)
    before = full_tracker_fingerprint(b)
    setattr(b.tracker, field, value)
    assert before == full_tracker_fingerprint(b)
    with pytest.raises(AssertionError): assert_complete_mot_AA(a, b)
    assert complete_mot_fingerprint(a) != complete_mot_fingerprint(b)


def test_every_tracker_attribute_and_identity_state_attribute_is_covered():
    b = bridge(); snapshot = complete_mot_snapshot(b)
    assert snapshot["all_tracker_attribute_names"] == sorted(vars(b.tracker))
    before = complete_mot_fingerprint(b)
    state = b.tracker.states[b.tracker.target_public]
    state.audit_extra_mutable = [1]
    assert complete_mot_fingerprint(b) != before
    cloned = deepcopy(b)
    assert_complete_mot_AA(b, cloned)
    cloned.tracker.states[cloned.tracker.target_public].prototype[1] = .5
    with pytest.raises(AssertionError): assert_complete_mot_AA(b, cloned)


def test_array_dtype_shape_bytes_and_dictionary_key_types_are_not_lost():
    assert canonical(np.array([1], np.float32)) != canonical(np.array([1], np.float64))
    assert canonical(np.array([1], np.float32)) != canonical(np.array([[1]], np.float32))
    assert canonical({1: "a", "1": "b"}) != canonical({"1": "b"})
    assert canonical({"a": 1, "b": 2}) == canonical({"b": 2, "a": 1})
    with pytest.raises(TypeError): canonical(object())
    with pytest.raises(ValueError): canonical(float("nan"))
    with pytest.raises(ValueError): canonical(np.array([object()], dtype=object))
    with pytest.raises(ValueError): canonical(np.array([float("nan")]))
    assert canonical(deque([1], maxlen=2)) != canonical(deque([1], maxlen=3))


def test_raw_C0_shadow_pair_uses_independent_raw_click_arrays_and_complete_AA(monkeypatch):
    from scripts import n72r21r2_complete_mot_AA_v1 as audit
    from sam3_intermot.one_click.acib_memory import ACIBMemoryNetwork
    from sam3_intermot.one_click.acib_trusted_runtime import TrustedACIBRecognizer
    from scripts.n72r21_t2_replay import DecisionCapture
    import torch
    anchor = np.zeros(512, np.float32); anchor[0] = 1.00001
    torch.set_num_threads(1); torch.manual_seed(7)
    actor = TrustedACIBRecognizer(DecisionCapture(ACIBMemoryNetwork()).eval(), anchor, "unit", policy="P0")
    actor.start_recording("unit", fps=20, width=100, height=100, initial_frame=0, initial_box=[0, 0, 20, 80])
    monkeypatch.setattr(audit, "make_actor", lambda *args: (actor, {}))
    event = dict(frame=0, clicked_candidate_uid="a0", box_xyxy=[0, 0, 20, 80], fps=20)
    left, right = audit.pair(event, anchor, 4)
    assert not np.shares_memory(left.tracker.event["human_anchor"], right.tracker.event["human_anchor"])
    assert left.tracker.event["human_anchor"][0] == anchor[0]
    for frame in range(4):
        rows = [dict(candidate_uid="a"+str(frame), feature=anchor, box_xyxy=[0, 0, 20, 80], native_tid=1, conf=.99)]
        assert_complete_mot_AA(left, right)
        a, b = left.step(frame, rows), right.step(frame, rows)
        assert a["outputs"] == b["outputs"]
        assert_complete_mot_AA(left, right)


def test_manager_allocator_maps_and_shared_state_relationship_are_not_omitted():
    a, b = bridge(), bridge()
    legacy = full_tracker_fingerprint(b)
    b.tracker.manager.next_pid += 1
    assert full_tracker_fingerprint(b) == legacy
    with pytest.raises(AssertionError): assert_complete_mot_AA(a, b)
    b = bridge(); state = b.tracker.states[b.tracker.target_public]
    b.tracker.manager.states[state.pid] = deepcopy(state)
    with pytest.raises(AssertionError): complete_mot_snapshot(b)


def test_scalar_types_numpy_dtypes_and_signed_zero_do_not_collapse():
    assert canonical(np.float64(1)) != canonical(np.float32(1))
    a, b = bridge(), bridge()
    b.tracker.audit_scalar = -0.0; a.tracker.audit_scalar = 0.0
    with pytest.raises(AssertionError): assert_complete_mot_AA(a, b)
    b.tracker.audit_scalar = 1.0; a.tracker.audit_scalar = 1
    with pytest.raises(AssertionError): assert_complete_mot_AA(a, b)
