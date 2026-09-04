"""Focused N72R6 unit tests; fixtures are toy/non-scientific only."""

from types import SimpleNamespace

import numpy as np
import pytest

from sam3_intermot.association.target_scoped_merge import (
    TARGET_CANDIDATE_KIND,
    apply_human_anchor_verification_gate,
    apply_target_exclusive_constraints,
    merge_main_and_target_candidates,
)
from sam3_intermot.association.identity_state import IdentityState
from sam3_intermot.identity.correction_epoch import (
    apply_epoch_to_identity_state,
    make_correction_epoch,
)


def _target(uid="target:0"):
    return {
        "candidate_uid": uid,
        "candidate_index": 0,
        "candidate_kind": TARGET_CANDIDATE_KIND,
        "event_id": "toy-event",
        "frame": 10,
        "public_id": None,
        "human_target_scope_public_id": 1001,
        "target_session_scope": "toy-target-scope",
        "runtime_future_gt_used": False,
        "runtime_gt_read": False,
        "posthoc_gt_used": False,
        "public_id_inference": False,
    }


def test_merge_is_lossless_for_main_and_reindexes_only_target_row():
    main = [{"candidate_uid": "main:0", "candidate_index": 0, "box_xyxy": [0, 0, 1, 1]}]
    merged, audit = merge_main_and_target_candidates(
        main,
        [_target()],
        event_id="toy-event",
        frame=10,
        target_public_id=1001,
        target_session_scope="toy-target-scope",
    )
    assert merged[0] == main[0]
    assert merged[1]["candidate_uid"] == "target:0"
    assert merged[1]["target_session_candidate_index"] == 0
    assert merged[1]["candidate_index"] == 1
    assert audit["main_rows_copied_without_relabel"] is True


def test_target_public_column_is_exclusive_and_none_remains_explicit():
    class Runtime:
        def get_identity_by_state_id(self, state_id):
            return {1: SimpleNamespace(public_id=1001), 2: SimpleNamespace(public_id=1002)}[state_id]

    states = [SimpleNamespace(pid=1), SimpleNamespace(pid=2)]
    main = [{"candidate_uid": "main:0", "candidate_index": 0}]
    merged, _ = merge_main_and_target_candidates(
        main,
        [_target()],
        event_id="toy-event",
        frame=10,
        target_public_id=1001,
        target_session_scope="toy-target-scope",
    )
    fused, audit = apply_target_exclusive_constraints(
        np.asarray([[2.0, 1.0], [0.5, 0.25]]),
        states,
        merged,
        Runtime(),
        target_public_id=1001,
        force_target_uid="target:0",
    )
    assert fused[0, 0] < -1e5
    assert fused[1, 0] > 1e5
    assert fused[1, 1] < -1e5
    assert audit["explicit_none_preserved"] is True


def test_frozen_target_main_row_is_shadowed_to_none_without_outer_birth():
    class Runtime:
        def get_identity_by_state_id(self, state_id):
            return {1: SimpleNamespace(public_id=1001), 2: SimpleNamespace(public_id=1002)}[state_id]

    states = [SimpleNamespace(pid=1), SimpleNamespace(pid=2)]
    main = [{"candidate_uid": "main:target", "candidate_index": 0}]
    merged, _ = merge_main_and_target_candidates(
        main,
        [],
        event_id="toy-event",
        frame=11,
        target_public_id=1001,
        target_session_scope="toy-target-scope",
    )
    fused, audit = apply_target_exclusive_constraints(
        np.asarray([[2.0, 9.0]]),
        states,
        merged,
        Runtime(),
        target_public_id=1001,
        shadowed_main_uids=["main:target"],
    )
    assert np.all(fused[0] < -1e5)
    assert audit["shadowed_main_rows_forced_none"] == 1


def test_fallback_allows_only_frozen_target_main_row_to_target_public():
    class Runtime:
        def get_identity_by_state_id(self, state_id):
            return {1: SimpleNamespace(public_id=1001), 2: SimpleNamespace(public_id=1002)}[state_id]

    states = [SimpleNamespace(pid=1), SimpleNamespace(pid=2)]
    main = [
        {"candidate_uid": "main:target", "candidate_index": 0},
        {"candidate_uid": "main:protected", "candidate_index": 1},
    ]
    merged, _ = merge_main_and_target_candidates(
        main,
        [],
        event_id="toy-event",
        frame=11,
        target_public_id=1001,
        target_session_scope="toy-target-scope",
    )
    base = np.asarray([[4.0, 1.0], [9.0, 8.0]])
    fused, audit = apply_target_exclusive_constraints(
        base,
        states,
        merged,
        Runtime(),
        target_public_id=1001,
        shadowed_main_uids=["main:target"],
        fallback_main_uid="main:target",
    )
    assert fused[0, 0] == base[0, 0]
    assert fused[0, 1] < -1e5
    assert fused[1, 0] < -1e5
    assert audit["fallback_main_row_allowed_target_public_or_none"] is True
    assert audit["main_rows_blocked_from_target_public"] == 1


def test_human_anchor_gate_rejects_target_candidate_to_none_only():
    accepted_candidate = _target("target:accepted")
    accepted_candidate["feature"] = np.eye(512, dtype=np.float32)[0].tolist()
    rejected_candidate = _target("target:rejected")
    rejected_candidate["feature"] = np.eye(512, dtype=np.float32)[1].tolist()
    accepted, accepted_audit = apply_human_anchor_verification_gate(
        [accepted_candidate],
        np.eye(512, dtype=np.float32)[0],
        threshold=0.9,
        event_id="toy-event",
        frame=11,
    )
    assert [item["candidate_uid"] for item in accepted] == ["target:accepted"]
    assert accepted_audit["rejected_candidate_count"] == 0
    rejected, rejected_audit = apply_human_anchor_verification_gate(
        [rejected_candidate],
        np.eye(512, dtype=np.float32)[0],
        threshold=0.9,
        event_id="toy-event",
        frame=11,
    )
    assert rejected == []
    assert rejected_audit["rejected_candidate_count"] == 1
    assert rejected_audit["rejected_to_explicit_none"] is True
    assert rejected_audit["main_candidate_fallback"] is False


def test_correction_epoch_clears_native_constraints_and_reanchors_motion():
    state = IdentityState(1, np.eye(512, dtype=np.float32)[0], np.zeros(4), 10, native_tid=7, native_scope="old")
    state.add_positive(7, native_scope="old")
    state.add_negative(8, native_scope="old")
    anchor = np.eye(512, dtype=np.float32)[1]
    epoch = make_correction_epoch(
        epoch_id="toy-epoch",
        public_id=1001,
        start_frame=10,
        human_anchor=anchor,
        target_session_scope="toy-target-scope",
        previous_native_tid=7,
        previous_native_scope="old",
    )
    apply_epoch_to_identity_state(state, epoch, anchor, [1, 2, 3, 4], target_native_tid=1)
    assert not state.positive_native_tids
    assert not state.negative_native_tids
    assert state.last_native_tid == 1
    assert state.last_native_scope == "toy-target-scope"
    assert state.correction_epoch_id == "toy-epoch"
    assert np.allclose(state.velocity, 0.0)
