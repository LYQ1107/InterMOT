from dataclasses import replace
import numpy as np
import pytest

from sam3_intermot.association.causal_identity_tracker import CausalIdentityTracker
from sam3_intermot.association.identity_authority import AuthorityConfig


def feature(i):
    v = np.zeros(512, dtype=np.float32)
    v[i] = 1
    return v


def candidate(uid, i, native, x=0):
    return {"candidate_uid": uid, "feature": feature(i), "box_xyxy": [x, 0, x+10, 20], "native_tid": native, "confidence": 0.9, "conf": 0.9}


def event(frame=0):
    return {"event_frame": frame, "human_anchor": feature(0), "target_candidate_uid": "a" if frame == 0 else "click", "target_public_id": 100001, "target_association_state_id": 1, "target_box_xyxy": [0, 0, 10, 20]}


def test_identity_off_a_a_and_pre_event_complete():
    c = AuthorityConfig()
    a = CausalIdentityTracker(config=c, event=event(2))
    b = CausalIdentityTracker(config=replace(c, strength=1), event=event(2))
    for f in range(4):
        rows = [candidate("click" if f == 2 else f"a{f}", 0, 1), candidate(f"b{f}", 1, 2, 100)]
        ra, rb = a.step(rows, f), b.step(rows, f)
        assert ra["outputs"] == rb["outputs"]
        assert ra["state_after"] == rb["state_after"]
        assert len(ra["outputs"]) == 2
        assert ra["authority"] == 0
        if f < 2:
            assert ra["target_public_id"] is None


def test_intervention_state_changes_next_frame_matrix_and_global_ids():
    a = CausalIdentityTracker(config=AuthorityConfig(), event=event())
    b = CausalIdentityTracker(config=AuthorityConfig(mode="fixed", source="raw", strength=10), event=event())
    r0 = [candidate("a", 0, 1), candidate("b", 1, 2, 100)]
    a.step(r0, 0); b.step(r0, 0)
    r1 = [candidate("x", 1, 1), candidate("y", 0, 2, 100)]
    ra = a.step(r1, 1); rb = b.step(r1, 1)
    assert ra["target_uid"] != rb["target_uid"]
    assert ra["state_after"] != rb["state_after"]
    assert len(set(row["candidate_uid"] for row in rb["outputs"])) == len(rb["outputs"])
    r2 = [candidate("u", 1, 1), candidate("v", 0, 2, 100)]
    ra2, rb2 = a.step(r2, 2), b.step(r2, 2)
    assert not np.array_equal(ra2["base_matrix"], rb2["base_matrix"])


def test_no_future_gt_or_fake_feature_or_repeated_frame():
    tracker = CausalIdentityTracker(config=AuthorityConfig(), event=event())
    with pytest.raises(ValueError, match="GT"):
        tracker.step([{**candidate("a", 0, 1), "gt_id": 3}], 0)
    with pytest.raises(ValueError, match="real finite"):
        tracker.step([{**candidate("a", 0, 1), "feature": np.zeros(512)}], 0)
    tracker.step([candidate("a", 0, 1)], 0)
    with pytest.raises(ValueError, match="increasing"):
        tracker.step([], 0)


def test_clone_state_independence_and_birth_death():
    tracker = CausalIdentityTracker(config=AuthorityConfig(), event=event(), max_lost_gap=1)
    tracker.step([candidate("a", 0, 1), candidate("b", 1, 2, 100)], 0)
    branch = tracker.clone()
    assert branch.states[100001] is not tracker.states[100001]
    branch.states[100001].prototype[:] = feature(4)
    assert np.array_equal(tracker.states[100001].prototype, feature(0))
    tracker.step([], 1)
    r = tracker.step([], 2)
    assert 100002 in r["deaths"]
    assert 100001 not in r["deaths"]
