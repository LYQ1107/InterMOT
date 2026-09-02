"""N72R3 persistent-public-identity contract tests.

These are deterministic toy fixtures for architecture and causal contracts;
they are not DanceTrack results and must not be counted as experiments.
"""

from __future__ import annotations

import numpy as np
import pytest

from sam3_intermot.association.state_manager import StateManager, StateManagerConfig
from sam3_intermot.backend.output_types import PromptObjectObservation
from sam3_intermot.identity.handover import PersistentLineageHandover
from sam3_intermot.identity.persistent_runtime import SequencePersistentIdentityRuntime
from sam3_intermot.identity.persistent_snapshot import PersistentRuntimeSnapshot
from sam3_intermot.identity.public_authority import PublicAuthorityBridge
from sam3_intermot.tracking.track import TrackState


def _obs(sam_id: int, box=(10.0, 10.0, 30.0, 50.0)) -> PromptObjectObservation:
    return PromptObjectObservation(
        frame_idx=0,
        sam_object_id=sam_id,
        raw_sam_object_id=sam_id,
        mask=np.ones((8, 8), dtype=bool),
        box_xyxy=np.asarray(box, dtype=float),
        confidence=0.9,
    )


def test_public_id_belongs_to_identity_not_candidate_and_survives_two_sessions() -> None:
    runtime = SequencePersistentIdentityRuntime("toy-1007", public_id_start=1000)
    runtime.begin_new_sam_session("session-A")
    record = runtime.create_identity(
        10,
        _obs(17),
        public_id=1007,
        candidate_uid="session-A:candidate-17",
        session_id="session-A",
        adapter_external_id=1700,
        raw_sam_id=17,
    )
    runtime.bind_candidate(
        record,
        "session-A:candidate-17",
        _obs(17),
        19,
        session_id="session-A",
        adapter_external_id=1700,
        raw_sam_id=17,
    )
    lineage = record.identity_lineage_id
    mot_id = record.mot_track_id
    snapshot = PersistentRuntimeSnapshot.capture(
        runtime, snapshot_frame=19, next_window_start=20
    )

    boundary = runtime.begin_new_sam_session("session-B", boundary_frame=19)
    assert boundary["identities_deleted"] is False
    assert record.status == "LOST"
    assert record.public_id == 1007
    assert record.mot_track_id == mot_id == 1007
    assert record.identity_lineage_id == lineage
    assert runtime.manager.get(1007).sam_object_id is None
    assert runtime.manager.get(1007).state == TrackState.LOST
    none_rows = runtime.record_frame_decisions(20, {})
    assert none_rows[0]["status"] == "NO_CANDIDATE_ASSIGNED"
    assert none_rows[0]["public_id"] == 1007

    runtime.reactivate(
        record,
        "session-B:candidate-8",
        _obs(8, box=(12.0, 10.0, 32.0, 50.0)),
        28,
        session_id="session-B",
        adapter_external_id=8008,
        raw_sam_id=88,
    )
    assert record.status == "ACTIVE"
    assert record.public_id == 1007
    assert record.mot_track_id == 1007
    assert record.identity_lineage_id == lineage
    assert record.current_raw_sam_id == 88
    assert runtime.manager.get(1007).sam_object_id == 8
    assert runtime.audit()["invariant_violations"] == []


def test_persistent_snapshot_is_exactly_window_start_minus_one() -> None:
    runtime = SequencePersistentIdentityRuntime("toy-snapshot", public_id_start=1000)
    runtime.begin_new_sam_session("A")
    runtime.create_identity(4, _obs(17), public_id=1007, session_id="A")
    with pytest.raises(ValueError, match="window_B.frame_start - 1"):
        PersistentRuntimeSnapshot.capture(runtime, snapshot_frame=4, next_window_start=20)
    snapshot = PersistentRuntimeSnapshot.capture(
        runtime, snapshot_frame=19, next_window_start=20
    )
    clone = SequencePersistentIdentityRuntime(
        "toy-snapshot",
        authority_bridge=PublicAuthorityBridge("n72r3-persistent:toy-snapshot", "toy-snapshot"),
    )
    snapshot.restore_into(clone)
    restored = clone.get_identity_by_public_id(1007)
    assert restored is not None
    assert restored.public_id == 1007
    assert clone.manager is clone.track_manager
    assert clone.audit()["auxiliary_track_manager_count"] == 0


def test_external_authority_state_manager_cannot_birth_public_identity() -> None:
    bridge = PublicAuthorityBridge("toy-run", "toy-external")
    manager = StateManager(
        StateManagerConfig(
            external_identity_authority=True,
            variant="reid",
            score_threshold=100.0,
        ),
        public_authority_resolver=bridge,
    )
    seed = {
        "feat": np.eye(512, dtype=np.float32)[0],
        "box": np.asarray([0, 0, 10, 20], dtype=float),
        "native_tid": 17,
    }
    manager.register_identity_state(7, 1007, seed, 0)
    with pytest.raises(RuntimeError, match="forbids StateManager-local births"):
        manager._new_pid()
    candidate = dict(seed)
    candidate.update({"obs_id": 0, "candidate_uid": "session-B:candidate-8"})
    manager.rollout_frame(1, [candidate])
    assert set(manager.states) == {7}
    assert len(manager.unmatched_candidates) == 1
    assert manager.unmatched_candidates[0]["reason"] == "OUTER_BIRTH_DECISION_REQUIRED"


def test_persistent_bridge_is_immutable_and_boundary_none_is_legal() -> None:
    bridge = PublicAuthorityBridge("toy-run", "toy-bridge")
    first = bridge.bind_identity_state(
        association_state_id=7,
        public_id=1007,
        mot_track_id=1007,
        lineage_id=77,
        created_frame=0,
        transaction_id="create-1007",
    )
    assert bridge.bind_identity_state(
        association_state_id=7,
        public_id=1007,
        mot_track_id=1007,
        lineage_id=77,
        created_frame=0,
        transaction_id="idempotent",
    ) == first
    with pytest.raises(ValueError, match="immutable public authority"):
        bridge.bind_identity_state(
            association_state_id=7,
            public_id=1008,
            mot_track_id=1008,
            lineage_id=77,
            created_frame=0,
            transaction_id="illegal-switch",
        )
    bridge.record_identity_no_candidate(7, 20)
    assert bridge.resolve_public_authority(
        association_state_id=7, frame_idx=20
    ).status == "EXPLICIT_NONE"
    assert bridge.resolve_public_authority(
        association_state_id=7, frame_idx=21
    ).public_id == 1007


def test_heuristic_handover_never_creates_exact_authority() -> None:
    handover = PersistentLineageHandover("run", "seq")
    transactions = handover.match_overlap(
        [
            {
                "frame_idx": 10,
                "mot_track_id": 4,
                "public_id": 1004,
                "lineage_id": 9,
                "raw_sam_id": 15,
                "adapter_id": 150,
                "box": [1, 1, 5, 5],
                "feature": [1.0, 0.0],
            }
        ],
        [
            {
                "frame_idx": 10,
                "mot_track_id": 1,
                "raw_sam_id": 88,
                "adapter_id": 880,
                "box": [1.1, 1.0, 5.1, 5.0],
                "feature": [0.99, 0.01],
            }
        ],
        from_session="s0",
        to_session="s1",
        from_segment="seg0",
        to_segment="seg1",
        frame_boundary=10,
    )
    assert len(transactions) == 1
    assert transactions[0].status.startswith("HEURISTIC")
    assert transactions[0].status != "PASS"
    assert transactions[0].authority_eligible is False
    assert handover.audit()["authority_eligible_transaction_count"] == 0
