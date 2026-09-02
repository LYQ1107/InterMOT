"""Four action classes and causal transaction guards on toy state only."""

from __future__ import annotations

import hashlib

import numpy as np
import pytest

from sam3_intermot.identity.add_transaction import AddIdentityTransaction
from sam3_intermot.identity.namespace import IdentityNamespace
from sam3_intermot.interaction.real_human_v2 import finalize_ui_submission, new_server_session, validate_real_human_event_v2
from sam3_intermot.interaction.runtime_transactions import RuntimeCausalError, RuntimeCausalGuard


def _base_event(action: str, *, public_id: int | None = 101) -> dict:
    event = {
        "event_id": f"toy-{action}",
        "sequence": "toy-seq",
        "split": "train",
        "session_id": "toy-session",
        "annotator_id_hash": "a" * 64,
        "timestamp": "2026-09-02T01:02:03+08:00",
        "frame_hash_sha256": "b" * 64,
        "candidate_tape_ref": "candidate/window-0.jsonl",
        "event_frame": 10,
        "prefix_range": [0, 9],
        "future_ranges": {"H20": [11, 30], "H50": [11, 60], "H100": [11, 110]},
        "action_type": action,
        "interaction_source": "ui_submission",
        "human_confirmed": True,
        "runtime_future_gt_used": False,
        "public_id": public_id,
        "human_input": {
            "kind": "ID_SELECTION",
            "origin": "external_human_ui",
            "human_confirmed": True,
            "raw_payload_ref": "requests/toy.json",
            "raw_payload_sha256": "c" * 64,
            "selected_public_ids": [101, 102],
        },
    }
    if action == "ADD_NEW_IDENTITY":
        event["public_id"] = None
        event["public_id_source"] = None
        event["human_input"] = {
            "kind": "BOX", "origin": "external_human_ui", "human_confirmed": True,
            "raw_payload_ref": "requests/toy.json", "raw_payload_sha256": "c" * 64,
            "box_xyxy": [1.0, 2.0, 7.0, 9.0],
        }
    elif action == "RECOVER_IDENTITY":
        event["public_id_source"] = "human_selected_existing_public"
        event["human_input"] = {
            "kind": "BOX", "origin": "external_human_ui", "human_confirmed": True,
            "raw_payload_ref": "requests/toy.json", "raw_payload_sha256": "c" * 64,
            "box_xyxy": [1.0, 2.0, 7.0, 9.0],
        }
    else:
        event["public_id_source"] = "human_selected_existing_public"
    if action == "AUTHORITATIVE_REASSIGN":
        event["source_public_id"] = 101
        event["destination_public_id"] = 102
    if action == "ATOMIC_ID_SWAP":
        event["other_public_id"] = 102
        event["atomic_transaction_confirmed"] = True
    return event


def test_v2_schema_covers_add_reassign_swap_and_recover() -> None:
    for action in ("ADD_NEW_IDENTITY", "AUTHORITATIVE_REASSIGN", "ATOMIC_ID_SWAP", "RECOVER_IDENTITY"):
        result = validate_real_human_event_v2(_base_event(action))
        assert result["valid"], (action, result["errors"])


def test_add_allocator_transaction_never_accepts_user_public_id_and_rolls_back() -> None:
    namespace = IdentityNamespace()
    before = namespace.mutable_state_hash()
    transaction = AddIdentityTransaction(namespace)
    ok, result, error = transaction.execute(10, lambda preview: {"allocated": preview.public_mot_id})
    assert ok and error is None
    assert result["allocated"] == 1000
    assert namespace.public_id_for(1) == 1000

    namespace2 = IdentityNamespace()
    before2 = namespace2.mutable_state_hash()
    transaction2 = AddIdentityTransaction(namespace2)
    ok, result, error = transaction2.execute(10, lambda _: (_ for _ in ()).throw(RuntimeError("toy failure")))
    assert not ok and result is None and "toy failure" in str(error)
    assert namespace2.mutable_state_hash() == before2
    assert before != namespace.mutable_state_hash()


def test_runtime_causal_guard_enforces_correction_then_t_plus_one_read() -> None:
    guard = RuntimeCausalGuard("toy-event", "ADD_NEW_IDENTITY", 10, "toy-session")
    digest = hashlib.sha256(b"toy feature").hexdigest()
    guard.record_spatial_correction(10, backend_prompt_route="native_box", correction_id="correction-1")
    guard.write_memory(10, memory_key="public-1000", feature_sha256=digest, source="human_roi")
    with pytest.raises(RuntimeCausalError, match="not visible"):
        guard.read_memory(10, memory_key="public-1000")
    guard.read_memory(11, memory_key="public-1000")
    guard.record_future_frame(11)
    result = guard.finalize(expected_first_future_frame=11)
    assert result["status"] == "PASS_RUNTIME_CAUSAL_BOUNDARY"
    assert result["event_frame_read"] is False
    assert result["runtime_future_gt_used"] is False


def test_runtime_guard_rejects_gt_and_wrong_order() -> None:
    guard = RuntimeCausalGuard("toy-event", "AUTHORITATIVE_CORRECT", 10, "toy-session")
    with pytest.raises(RuntimeCausalError, match="GT"):
        guard.record_spatial_correction(10, backend_prompt_route="native_box", correction_id="c", gt_box=[1, 2, 3, 4])
    with pytest.raises(RuntimeCausalError, match="spatial correction"):
        guard.write_memory(10, memory_key="k", feature_sha256="a" * 64, source="human_roi")
