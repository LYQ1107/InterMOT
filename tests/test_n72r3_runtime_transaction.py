"""Focused N72R3 atomic backend/identity/memory rollback tests."""

from __future__ import annotations

import numpy as np
import pytest

from sam3_intermot.association.state_manager import StateManager, StateManagerConfig
from sam3_intermot.backend.mock_backend import MockBackend
from sam3_intermot.identity.persistent_runtime import SequencePersistentIdentityRuntime
from sam3_intermot.interaction.runtime_transactions import (
    RuntimeInteractionError,
    RuntimeInteractionTransaction,
)


def _fixture():
    backend = MockBackend(frame_h=128, frame_w=128, seed=72)
    session_id = backend.start_video("toy-n72r3")
    initial = backend.add_box(0, 1, np.asarray([10.0, 10.0, 30.0, 40.0]))
    runtime = SequencePersistentIdentityRuntime("toy-n72r3", public_id_start=1007)
    record = runtime.create_identity(
        0,
        initial,
        public_id=1007,
        candidate_uid="seed-candidate",
        session_id=session_id,
        adapter_external_id=1,
        raw_sam_id=1,
    )
    state_manager = StateManager(
        StateManagerConfig(
            external_identity_authority=True,
            use_appearance_memory=True,
        )
    )
    state_manager.register_from_persistent_identity(
        record,
        {"feat": np.ones(512, dtype=np.float32), "box": [10.0, 10.0, 30.0, 40.0], "native_tid": 1},
        0,
    )
    return backend, session_id, runtime, record, state_manager


def _transaction(backend, runtime, state_manager, event_id):
    return RuntimeInteractionTransaction(
        backend=backend,
        persistent_runtime=runtime,
        state_manager=state_manager,
        event_id=event_id,
    )


def test_backend_success_manager_failure_restores_backend_and_identity() -> None:
    backend, session_id, runtime, record, state_manager = _fixture()
    before_box = backend._objects[1]["prompt_box"].copy()
    before_record = record.as_dict()
    holder = {}

    def backend_step():
        holder["obs"] = backend.correct_object(
            1, 1, box_xyxy=np.asarray([20.0, 20.0, 45.0, 55.0])
        )
        return holder["obs"]

    def identity_step():
        runtime.bind_candidate(
            record,
            "candidate-after-correction",
            holder["obs"],
            1,
            session_id=session_id,
            adapter_external_id=101,
            raw_sam_id=201,
        )
        raise RuntimeError("injected manager failure")

    with pytest.raises(RuntimeInteractionError, match="injected manager failure"):
        _transaction(backend, runtime, state_manager, "tx-manager-failure").execute(
            backend_step, identity_step, lambda: True
        )

    np.testing.assert_array_equal(backend._objects[1]["prompt_box"], before_box)
    assert record.as_dict() == before_record
    assert sorted(backend._objects) == [1]
    assert runtime.manager._sam_to_track == {1: 1007}


def test_manager_success_memory_failure_restores_all_prior_phases() -> None:
    backend, session_id, runtime, record, state_manager = _fixture()
    before_box = backend._objects[1]["prompt_box"].copy()
    holder = {}

    def backend_step():
        holder["obs"] = backend.correct_object(
            1, 1, box_xyxy=np.asarray([22.0, 21.0, 46.0, 56.0])
        )
        return holder["obs"]

    def identity_step():
        return runtime.bind_candidate(
            record,
            "candidate-after-correction",
            holder["obs"],
            1,
            session_id=session_id,
            adapter_external_id=102,
            raw_sam_id=202,
        )

    def memory_step():
        state_manager.appearance_memory.update_from_human(
            1007, 1, np.ones(512, dtype=np.float32), quality=1.0
        )
        raise RuntimeError("injected memory failure")

    with pytest.raises(RuntimeInteractionError, match="injected memory failure"):
        _transaction(backend, runtime, state_manager, "tx-memory-failure").execute(
            backend_step, identity_step, memory_step
        )

    np.testing.assert_array_equal(backend._objects[1]["prompt_box"], before_box)
    assert record.last_candidate_uid == "seed-candidate"
    assert record.current_raw_sam_id == 1
    assert state_manager.appearance_memory.records == {}
    assert sorted(backend._objects) == [1]
    assert runtime.manager._sam_to_track == {1: 1007}


def test_add_allocation_failure_restores_allocator_and_removes_ghost_sam_object() -> None:
    backend, session_id, runtime, record, state_manager = _fixture()
    holder = {}

    def backend_step():
        holder["obs"] = backend.add_box(
            2, 99, np.asarray([60.0, 60.0, 80.0, 90.0])
        )
        return holder["obs"]

    def identity_step():
        # 1007 is already occupied; the outer allocator must reject this birth.
        return runtime.create_identity(
            2,
            holder["obs"],
            public_id=1007,
            candidate_uid="ghost-birth",
            session_id=session_id,
        )

    with pytest.raises(RuntimeInteractionError, match="public_id is unavailable"):
        _transaction(backend, runtime, state_manager, "tx-allocation-failure").execute(
            backend_step, identity_step, lambda: True
        )

    assert sorted(backend._objects) == [1]
    assert runtime.audit()["identity_count"] == 1
    assert runtime.get_identity_by_public_id(1007) is record
    assert runtime._public_allocator.snapshot() == {"next_id": 1008, "allocated": [1007]}
