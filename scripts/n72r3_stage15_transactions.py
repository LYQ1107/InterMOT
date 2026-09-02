#!/usr/bin/env python3
"""Run the N72R3 atomic backend/identity/memory fault-injection contract."""

from __future__ import annotations

import json
import os
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "N72R3"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sam3_intermot.association.state_manager import StateManager, StateManagerConfig  # noqa: E402
from sam3_intermot.backend.mock_backend import MockBackend  # noqa: E402
from sam3_intermot.identity.persistent_runtime import SequencePersistentIdentityRuntime  # noqa: E402
from sam3_intermot.interaction.runtime_transactions import (  # noqa: E402
    RuntimeInteractionError,
    RuntimeInteractionTransaction,
)


def atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        directory_fd = os.open(path.parent, os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def make_fixture():
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
        StateManagerConfig(external_identity_authority=True, use_appearance_memory=True)
    )
    state_manager.register_from_persistent_identity(
        record,
        {"feat": np.ones(512, dtype=np.float32), "box": [10.0, 10.0, 30.0, 40.0], "native_tid": 1},
        0,
    )
    return backend, session_id, runtime, record, state_manager


def tx(backend, runtime, state_manager, event_id: str) -> RuntimeInteractionTransaction:
    return RuntimeInteractionTransaction(
        backend=backend,
        persistent_runtime=runtime,
        state_manager=state_manager,
        event_id=event_id,
    )


def run_case(name: str) -> dict:
    backend, session_id, runtime, record, state_manager = make_fixture()
    before_box = backend._objects[1]["prompt_box"].copy()
    holder = {}

    if name == "backend_succeeds_manager_fails":
        def backend_step():
            holder["obs"] = backend.correct_object(1, 1, box_xyxy=np.asarray([20.0, 20.0, 45.0, 55.0]))
            return holder["obs"]

        def identity_step():
            runtime.bind_candidate(record, "candidate-after-correction", holder["obs"], 1, session_id=session_id, adapter_external_id=101, raw_sam_id=201)
            raise RuntimeError("injected manager failure")

        expected = "injected manager failure"
        memory_step = lambda: True
    elif name == "manager_succeeds_memory_fails":
        def backend_step():
            holder["obs"] = backend.correct_object(1, 1, box_xyxy=np.asarray([22.0, 21.0, 46.0, 56.0]))
            return holder["obs"]

        def identity_step():
            return runtime.bind_candidate(record, "candidate-after-correction", holder["obs"], 1, session_id=session_id, adapter_external_id=102, raw_sam_id=202)

        def memory_step():
            state_manager.appearance_memory.update_from_human(1007, 1, np.ones(512, dtype=np.float32), quality=1.0)
            raise RuntimeError("injected memory failure")

        expected = "injected memory failure"
    else:
        def backend_step():
            holder["obs"] = backend.add_box(2, 99, np.asarray([60.0, 60.0, 80.0, 90.0]))
            return holder["obs"]

        def identity_step():
            return runtime.create_identity(2, holder["obs"], public_id=1007, candidate_uid="ghost-birth", session_id=session_id)

        expected = "public_id is unavailable"
        memory_step = lambda: True

    transaction = tx(backend, runtime, state_manager, f"n72r3-{name}")
    caught = None
    try:
        transaction.execute(backend_step, identity_step, memory_step)
    except RuntimeInteractionError as exc:
        caught = str(exc)
    rollback_ok = (
        caught is not None
        and expected in caught
        and np.array_equal(backend._objects[1]["prompt_box"], before_box)
        and sorted(backend._objects) == [1]
        and runtime.audit()["identity_count"] == 1
        and runtime.get_identity_by_public_id(1007) is record
        and runtime.manager._sam_to_track == {1: 1007}
        and state_manager.appearance_memory.records == {}
        and runtime._public_allocator.snapshot() == {"next_id": 1008, "allocated": [1007]}
    )
    return {
        "case": name,
        "expected_failure_substring": expected,
        "caught_failure": caught,
        "transaction_audit": transaction.audit(),
        "rollback_verified": bool(rollback_ok),
        "ghost_sam_object_ids": sorted(set(backend._objects) - {1}),
        "runtime_future_gt_used": False,
    }


def main() -> int:
    started = datetime.now(timezone.utc).isoformat()
    cases = [run_case(name) for name in (
        "backend_succeeds_manager_fails",
        "manager_succeeds_memory_fails",
        "add_allocation_fails",
    )]
    passed = all(item["rollback_verified"] for item in cases)
    artifact = {
        "schema_version": "N72R3_STAGE15_ATOMIC_TRANSACTION_AUDIT_V1",
        "stage": "15_BACKEND_IDENTITY_MEMORY_ATOMIC_TRANSACTION",
        "status": "PASS_STAGE15_ATOMIC_TRANSACTION_TOY_FAULT_INJECTION" if passed else "FAIL_STAGE15_ATOMIC_TRANSACTION",
        "created_at_utc": started,
        "cases": cases,
        "backend_snapshot_contract": "sam3_state_snapshot_for_live_sam3_python_object_for_mock_only",
        "official_backend_fault_injection": "PENDING_STAGE16_LIVE_BACKEND_SMOKE",
        "runtime_future_gt_used": False,
        "scientific_result": "NOT_A_SCIENTIFIC_RESULT",
    }
    atomic_json(OUT / "transactions" / "stage15_atomic_transaction_audit.json", artifact)
    atomic_json(OUT / "stage_15_status.json", {
        "schema_version": "N72R3_STAGE_STATUS_V1",
        "stage": artifact["stage"],
        "status": artifact["status"],
        "created_at_utc": started,
        "artifact": str(OUT / "transactions" / "stage15_atomic_transaction_audit.json"),
        "case_count": len(cases),
        "rollback_verified_count": sum(int(item["rollback_verified"]) for item in cases),
        "official_backend_fault_injection": artifact["official_backend_fault_injection"],
        "runtime_future_gt_used": False,
        "scientific_result": "NOT_A_SCIENTIFIC_RESULT",
    })
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
