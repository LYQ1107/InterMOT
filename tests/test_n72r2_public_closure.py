"""Focused toy contract tests; no scientific result is produced here."""

import numpy as np

from sam3_intermot.backend.output_types import PromptObjectObservation
from sam3_intermot.association.public_assignment import (
    solve_exact_public_assignment,
    validate_exact_public_assignment,
)
from sam3_intermot.identity.handover import PersistentLineageHandover
from sam3_intermot.identity.public_authority import PublicAuthorityBridge
from sam3_intermot.interaction.n72r2_simulated_observer import (
    ACTION_TYPES,
    N72R2SimulatedHumanObserver,
)
from sam3_intermot.interaction.simulator import GTFrame
from sam3_intermot.interaction.continuous_observer import GTFrameAccessor
from sam3_intermot.tracking.track_manager import TrackManager
from sam3_intermot.identity.lineage import IdentityLineageRegistry


def _obs(frame=0, sam_id=10, box=(1, 1, 5, 5)):
    return PromptObjectObservation(
        frame_idx=frame,
        sam_object_id=sam_id,
        mask=np.ones((8, 8), dtype=bool),
        box_xyxy=np.asarray(box, dtype=float),
        confidence=0.9,
    )


def test_track_manager_public_authority_is_not_association_pid():
    manager = TrackManager()
    lineage = IdentityLineageRegistry().create(0)
    track = manager.create_track(0, _obs(), lineage.lineage_id, mot_track_id=77)
    bridge = PublicAuthorityBridge("run", "seq")
    binding = bridge.bind_track(
        frame_idx=0,
        candidate_uid="uid-0",
        association_state_id=3,
        track=track,
        binding_transaction_id="txn-0",
    )
    assert binding.status == "EXACT"
    assert bridge.resolve_public_authority(association_state_id=3).public_id == 77
    assert bridge.resolve_public_authority(association_state_id=99).status == "NO_PUBLIC_AUTHORITY"
    assert bridge.resolve_public_authority(association_state_id=3, source_run_id="other").status == "SOURCE_RUN_MISMATCH"
    assert bridge.audit()["public_id_is_association_state_id"] is False


def test_handover_uses_distinct_raw_ids_and_persistent_lineage():
    handover = PersistentLineageHandover("run", "seq")
    left = [{
        "frame_idx": 10, "mot_track_id": 4, "public_id": 1004, "lineage_id": 9,
        "raw_sam_id": 15, "adapter_id": 150, "box": [1, 1, 5, 5],
        "feature": [1.0, 0.0],
    }]
    right = [{
        "frame_idx": 10, "mot_track_id": 1, "raw_sam_id": 88, "adapter_id": 880,
        "box": [1.1, 1.0, 5.1, 5.0], "feature": [0.99, 0.01],
    }]
    txns = handover.match_overlap(
        left, right, from_session="s0", to_session="s1",
        from_segment="seg0", to_segment="seg1", frame_boundary=10,
    )
    assert len(txns) == 1
    assert txns[0].status == "PASS"
    assert txns[0].old_raw_sam_id != txns[0].new_raw_sam_id
    assert txns[0].public_id == 1004
    assert handover.audit()["raw_id_equality_used_for_match"] is False


def test_simulated_observer_enforces_gt_after_prediction_and_t1_memory():
    accessor = GTFrameAccessor({5: GTFrame(boxes=[np.asarray([1, 1, 5, 5])], gt_ids=[2])})
    observer = N72R2SimulatedHumanObserver(accessor, "seq", "event-5")
    observer.begin_prediction(5)
    try:
        observer.read_current_gt_for_simulation()
    except RuntimeError:
        pass
    else:
        raise AssertionError("GT was read before Y_pre freeze")
    observer.freeze_prediction({"candidate": "pre"})
    current = observer.read_current_gt_for_simulation()
    observer.simulate_action("AUTHORITATIVE_CORRECT", public_id=17, current_gt_input=current)
    observer.freeze_post({"candidate": "post"})
    observer.write_memory(17, embedding=np.ones(4, dtype=np.float32), source="current_frame_authoritative_roi")
    # The event-frame branch must not even issue a memory read.  The observer
    # exposes the first legal read only at t+1.
    assert observer.read_memory(6, 17) is not None
    audit = observer.audit_dict()
    assert audit["event_frame_read_hidden"] is True
    assert audit["first_memory_read_offset"] == 1
    assert audit["runtime_future_gt_used"] is False


def test_simulated_observer_declares_all_six_action_contracts():
    for index, action_type in enumerate(ACTION_TYPES):
        frame = 10 + index
        accessor = GTFrameAccessor(
            {frame: GTFrame(boxes=[np.asarray([1, 1, 5, 5])], gt_ids=[index + 1])}
        )
        observer = N72R2SimulatedHumanObserver(accessor, "toy", f"event-{frame}")
        observer.begin_prediction(frame)
        observer.freeze_prediction({"frame": frame, "pre": True})
        current = observer.read_current_gt_for_simulation()
        command = observer.simulate_action(
            action_type,
            public_id=None if action_type == "AUTHORITATIVE_DELETE" else 100 + index,
            current_gt_input=current,
        )
        observer.freeze_post({"frame": frame, "post": True})
        if action_type != "AUTHORITATIVE_DELETE":
            observer.write_memory(
                100 + index,
                embedding=np.ones(4, dtype=np.float32),
                source="current_frame_authoritative_roi",
            )
        assert command["action_type"] == action_type
        assert command["interaction_source"] == "simulated_from_gt"
        assert command["runtime_future_gt_used"] is False
        assert observer.audit_dict()["gt_read_future"] == 0


def test_exact_public_assignment_keeps_state_and_public_axes_distinct():
    artifact = solve_exact_public_assignment(
        [{"candidate_uid": "c0", "candidate_index": 0}, {"candidate_uid": "c1", "candidate_index": 1}],
        [[0.9, 0.1], [0.2, 0.8]],
        [7, 8],
        [101, 202],
        none_score=0.0,
        source_run_id="toy-run",
        session_id="toy-session",
    )
    assert validate_exact_public_assignment(artifact) == []
    assert [row["public_id"] for row in artifact["assignment_rows"]] == [101, 202]
    assert artifact["association_state_id_is_public_id"] is False
    assert artifact["none_column_count"] == 2


def test_exact_public_assignment_can_explicitly_reject_all_candidates():
    artifact = solve_exact_public_assignment(
        [{"candidate_uid": "c0"}, {"candidate_uid": "c1"}],
        [[-1.0, -2.0], [-3.0, -4.0]],
        [7, 8],
        [101, 202],
        none_score=0.0,
    )
    assert artifact["explicit_none_count"] == 2
    assert all(row["status"] == "EXPLICIT_NONE" for row in artifact["assignment_rows"])


def test_public_axis_missing_candidate_is_not_candidate_explicit_none():
    artifact = solve_exact_public_assignment(
        [{"candidate_uid": "c0"}], [[-1.0, -2.0]], [7, 8], [101, 202], none_score=0.0
    )
    assert artifact["public_assignments"][0]["status"] == "NO_CANDIDATE_ASSIGNED"
    assert artifact["public_assignments"][1]["status"] == "NO_CANDIDATE_ASSIGNED"


def test_exact_public_assignment_rejects_duplicate_authority_axis():
    try:
        solve_exact_public_assignment(
            [{"candidate_uid": "c0"}], [[1.0, 0.5]], [7, 8], [101, 101]
        )
    except ValueError as exc:
        assert "duplicate" in str(exc)
    else:
        raise AssertionError("duplicate public authority was accepted")
