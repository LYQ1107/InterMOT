"""Same-run mapping/assignment toy tests; public IDs are explicit only."""

from __future__ import annotations

import numpy as np

from sam3_intermot.association.assignment_sidecar import build_assignment_sidecar, validate_assignment_sidecar
from sam3_intermot.association.state_manager import StateManager, StateManagerConfig
from sam3_intermot.provenance.mapping_v2 import HandoverLedger, PublicAuthorityResolver


def _candidate(index: int, uid: str, local: str, global_id: str) -> dict:
    return {
        "candidate_index": index,
        "candidate_uid": uid,
        "source_run_id": "toy-run",
        "session_id": "toy-session",
        "segment_id": "segment-0",
        "window_id": "window-0",
        "chunk_id": "chunk-0",
        "official_raw_sam_id": 17 + index,
        "adapter_external_id": 9001 + index,
        "segment_local_id": local,
        "sequence_global_id": global_id,
    }


def test_handover_does_not_turn_missing_continuity_into_exact_mapping() -> None:
    ledger = HandoverLedger("toy-run", "toy-seq", "toy-session", "segment-1")
    local, global_id = ledger.axes_for(17, previous_global_id="previous-global")
    assert local.startswith("seg-")
    assert global_id is None
    assert ledger.audit()["status"] == "PARTIAL_MULTI_CHUNK_HANDOVER_UNRESOLVED"


def test_assignment_sidecar_separates_state_axis_from_public_authority() -> None:
    manager = StateManager(StateManagerConfig(score_threshold=-100.0, variant="reid"))
    seed = [
        {"obs_id": i, "candidate_uid": f"seed-{i}", "source_run_id": "toy-run", "session_id": "toy-session", "segment_id": "segment-0", "window_id": "window-0", "chunk_id": "chunk-0", "official_raw_sam_id": 17 + i, "adapter_external_id": 9001 + i, "segment_local_id": f"local-{i}", "sequence_global_id": f"global-{i}", "feat": np.eye(512, dtype=np.float32)[i], "has_feat": 1.0, "box": np.asarray([i, 0, i + 5, 5], dtype=float), "native_tid": 9001 + i, "native_age": 0.0, "conf": 0.9}
        for i in range(2)
    ]
    manager.rollout_frame(0, seed)
    manager.candidate_log.clear()
    current = [dict(item, candidate_uid=f"current-{i}", native_age=1.0) for i, item in enumerate(seed)]
    manager.rollout_frame(1, current)
    audit = manager.candidate_log[-1]
    candidates = [_candidate(i, f"current-{i}", f"local-{i}", f"global-{i}") for i in range(2)]

    unresolved = build_assignment_sidecar(candidates, audit, source_run_id="toy-run", session_id="toy-session")
    assert unresolved["public_id_axis"] == [None, None]
    assert unresolved["public_authority_present"] is False
    assert all(item["status"] == "CANDIDATE_ONLY_NO_PUBLIC_AUTHORITY" for item in unresolved["candidate_assignment_rows"])

    resolver = PublicAuthorityResolver(source_run_id="toy-run", session_id="toy-session")
    resolver.bind(1, 101, source="explicit_runtime_assignment", transaction_id="toy-bind-1")
    resolver.bind(2, 102, source="explicit_runtime_assignment", transaction_id="toy-bind-2")
    exact = build_assignment_sidecar(candidates, audit, resolver=resolver, source_run_id="toy-run", session_id="toy-session")
    assert exact["public_id_axis"] == [101, 102]
    assert validate_assignment_sidecar(exact) == []


def test_public_resolver_rejects_conflicting_or_duplicate_bindings() -> None:
    resolver = PublicAuthorityResolver(source_run_id="toy-run", session_id="toy-session")
    resolver.bind(1, 101, source="explicit_runtime_assignment", transaction_id="first")
    try:
        resolver.bind(2, 101, source="explicit_runtime_assignment", transaction_id="collision")
    except ValueError as exc:
        assert "already bound" in str(exc)
    else:  # pragma: no cover - assertion makes the failure explicit
        raise AssertionError("duplicate public binding was accepted")
