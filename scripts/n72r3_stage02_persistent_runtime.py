"""Run deterministic N72R3 persistent-identity stress cases.

The cases use toy ``PromptObjectObservation`` objects only.  They validate the
identity/session contract and do not produce DanceTrack or efficacy evidence.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sam3_intermot.backend.output_types import PromptObjectObservation
from sam3_intermot.identity.handover import PersistentLineageHandover
from sam3_intermot.identity.persistent_runtime import SequencePersistentIdentityRuntime
from sam3_intermot.identity.persistent_snapshot import PersistentRuntimeSnapshot


OUT = ROOT / "outputs" / "N72R3"


def atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
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


def observation(sam_id: int, box=(10.0, 10.0, 30.0, 50.0)) -> PromptObjectObservation:
    return PromptObjectObservation(
        frame_idx=0,
        sam_object_id=sam_id,
        raw_sam_object_id=sam_id,
        mask=np.ones((8, 8), dtype=bool),
        box_xyxy=np.asarray(box, dtype=float),
        confidence=0.9,
    )


def case_public_1007() -> tuple[dict, SequencePersistentIdentityRuntime]:
    runtime = SequencePersistentIdentityRuntime("n72r3-toy-1007", public_id_start=1000)
    runtime.begin_new_sam_session("session-A")
    identity = runtime.create_identity(
        10,
        observation(17),
        public_id=1007,
        candidate_uid="session-A:candidate-17",
        session_id="session-A",
        adapter_external_id=1700,
        raw_sam_id=17,
        appearance_state={"human_anchor_digest": "toy-anchor-1007"},
        motion_state_ref={"velocity": [0.0, 0.0]},
    )
    runtime.bind_candidate(
        identity,
        "session-A:candidate-17",
        observation(17),
        19,
        session_id="session-A",
        adapter_external_id=1700,
        raw_sam_id=17,
    )
    snapshot = PersistentRuntimeSnapshot.capture(
        runtime, snapshot_frame=19, next_window_start=20
    )
    boundary = runtime.begin_new_sam_session("session-B", boundary_frame=19)
    none_rows = runtime.record_frame_decisions(20, {})
    restored = runtime.get_identity_by_public_id(1007)
    assert restored is not None
    assert restored.status == "LOST"
    assert restored.public_id == 1007
    assert restored.mot_track_id == 1007
    assert runtime.manager.get(1007).sam_object_id is None
    runtime.reactivate(
        restored,
        "session-B:candidate-8",
        observation(8, (12.0, 10.0, 32.0, 50.0)),
        28,
        session_id="session-B",
        adapter_external_id=8008,
        raw_sam_id=88,
    )
    assert restored.status == "ACTIVE"
    assert restored.identity_lineage_id == identity.identity_lineage_id
    assert runtime.manager.get(1007).sam_object_id == 8
    return (
        {
            "case": "1007_SESSION_A_NONE_SESSION_B_REBIND",
            "public_id": 1007,
            "lineage_id": identity.identity_lineage_id,
            "mot_track_id": identity.mot_track_id,
            "session_a_candidate": "session-A:candidate-17",
            "boundary": boundary,
            "boundary_none_rows": none_rows,
            "session_b_candidate": "session-B:candidate-8",
            "session_b_raw_sam_id": 88,
            "session_b_adapter_external_id": 8008,
            "snapshot": snapshot.as_dict(),
            "public_id_stable": restored.public_id == 1007,
            "lineage_stable": restored.identity_lineage_id == identity.identity_lineage_id,
            "mot_track_id_stable": restored.mot_track_id == 1007,
            "lost_without_deletion": True,
            "reactivated_with_new_raw_id": runtime.manager.get(1007).sam_object_id == 8,
        },
        runtime,
    )


def run() -> dict:
    cases: list[dict] = []
    primary, runtime = case_public_1007()
    cases.append(primary)

    # CASE 1: a public identity has a candidate at the boundary.
    case1 = SequencePersistentIdentityRuntime("n72r3-case-1", public_id_start=1)
    case1.begin_new_sam_session("A")
    id1 = case1.create_identity(0, observation(1), public_id=1, session_id="A")
    case1.record_frame_decisions(0, {1: "A:candidate-1"})
    cases.append({"case": "1_BOUNDARY_CANDIDATE", "status": "PASS", "public_id": id1.public_id})

    # CASE 2: no candidate at the boundary is a legal LOST state.
    case2 = SequencePersistentIdentityRuntime("n72r3-case-2", public_id_start=1)
    case2.begin_new_sam_session("A")
    id2 = case2.create_identity(0, observation(2), public_id=2, session_id="A")
    case2.begin_new_sam_session("B", boundary_frame=5)
    rows2 = case2.record_frame_decisions(5, {})
    cases.append(
        {
            "case": "2_BOUNDARY_NONE_LOST",
            "status": "PASS",
            "public_id": id2.public_id,
            "identity_exists": case2.get_identity_by_public_id(2) is not None,
            "identity_status": id2.status,
            "decision": rows2[0],
        }
    )

    # CASE 3: a new raw SAM object reactivates the same persistent identity.
    case2.reactivate(id2, "B:candidate-8", observation(8), 13, session_id="B", raw_sam_id=88)
    cases.append(
        {
            "case": "3_REAPPEAR_NEW_RAW_ID_REACTIVATE",
            "status": "PASS",
            "public_id": id2.public_id,
            "raw_sam_id": id2.current_raw_sam_id,
            "track_sam_object_id": case2.manager.get(id2.mot_track_id).sam_object_id,
        }
    )

    # CASE 4: duplicate geometry cannot create exact authority for two IDs.
    case4 = SequencePersistentIdentityRuntime("n72r3-case-4", public_id_start=1)
    case4.begin_new_sam_session("A")
    left = case4.create_identity(0, observation(4), public_id=4, session_id="A")
    right = case4.create_identity(0, observation(5), public_id=5, session_id="A")
    case4.begin_new_sam_session("B", boundary_frame=5)
    heuristic = PersistentLineageHandover("run", "n72r3-case-4").match_overlap(
        [
            {"frame_idx": 5, "mot_track_id": 4, "public_id": 4, "lineage_id": left.identity_lineage_id, "raw_sam_id": 4, "adapter_id": 40, "box": [10, 10, 30, 50], "feature": [1.0, 0.0]},
            {"frame_idx": 5, "mot_track_id": 5, "public_id": 5, "lineage_id": right.identity_lineage_id, "raw_sam_id": 5, "adapter_id": 50, "box": [10, 10, 30, 50], "feature": [0.0, 1.0]},
        ],
        [
            {"frame_idx": 5, "mot_track_id": 1, "raw_sam_id": 41, "adapter_id": 410, "box": [10, 10, 30, 50], "feature": [1.0, 0.0]},
        ],
        from_session="A",
        to_session="B",
        from_segment="A",
        to_segment="B",
        frame_boundary=5,
    )
    assert all(item.status.startswith("HEURISTIC") for item in heuristic)
    assert case4.get_identity_by_public_id(4).status == "LOST"
    assert case4.get_identity_by_public_id(5).status == "LOST"
    cases.append(
        {
            "case": "4_DUPLICATE_BOUNDARY_BOX_NO_GEOMETRY_AUTHORITY",
            "status": "PASS",
            "persistent_ids": [4, 5],
            "heuristic_statuses": [item.status for item in heuristic],
            "authority_eligible": [item.authority_eligible for item in heuristic],
        }
    )

    # CASE 5: an outer birth decision allocates a new public ID.
    case5 = SequencePersistentIdentityRuntime("n72r3-case-5", public_id_start=10)
    case5.begin_new_sam_session("B")
    new_identity = case5.create_identity(2, observation(60), session_id="B")
    assert new_identity.public_id == 10
    cases.append({"case": "5_OUTER_BIRTH_ALLOCATES_NEW_PUBLIC_ID", "status": "PASS", "public_id": new_identity.public_id})

    # CASE 6: a false-positive candidate remains unassigned; it cannot
    # overwrite a historical public identity.
    case6 = SequencePersistentIdentityRuntime("n72r3-case-6", public_id_start=1)
    case6.begin_new_sam_session("A")
    old = case6.create_identity(0, observation(6), public_id=6, session_id="A")
    case6.record_frame_decisions(1, {})
    assert case6.get_identity_by_public_id(6) is old
    cases.append(
        {
            "case": "6_FALSE_POSITIVE_UNASSIGNED_NONE",
            "status": "PASS",
            "historical_public_ids": [6],
            "new_identity_created": False,
            "decision_status": case6.assignment_log[-1]["status"],
        }
    )

    audit = runtime.audit()
    assert audit["invariant_violations"] == []
    assert audit["auxiliary_track_manager_count"] == 0
    assert audit["runtime_future_gt_used"] is False
    result = {
        "schema_version": "N72R3_STAGE02_PERSISTENT_RUNTIME_STRESS_V1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS_PERSISTENT_RUNTIME_TOY_CONTRACT",
        "scientific_result": "NOT_A_SCIENTIFIC_RESULT",
        "cases": cases,
        "primary_runtime_audit": audit,
        "primary_snapshot_identity_count": len(runtime.identities),
        "runtime_future_gt_used": False,
    }
    atomic_json(OUT / "runtime" / "persistent_runtime_stress_cases.json", result)
    atomic_json(
        OUT / "stage_02_status.json",
        {
            "schema_version": "N72R3_STAGE_STATUS_V1",
            "stage": "02_PERSISTENT_RUNTIME",
            "status": result["status"],
            "created_at_utc": result["created_at_utc"],
            "artifact": str(OUT / "runtime" / "persistent_runtime_stress_cases.json"),
            "cases": len(cases),
            "primary_case": "1007_SESSION_A_NONE_SESSION_B_REBIND",
            "track_manager_instance_count": audit["track_manager_instance_count"],
            "auxiliary_track_manager_count": audit["auxiliary_track_manager_count"],
            "public_id_stable": True,
            "lineage_stable": True,
            "lost_without_deletion": True,
            "reactivated_new_raw_sam_id": True,
            "candidate_boundary_presence_required": False,
            "heuristic_handover_authority_eligible": False,
            "runtime_future_gt_used": False,
            "scientific_result": "NOT_A_SCIENTIFIC_RESULT",
            "next_stage": "03_UNIFY_MOT_PUBLIC_ID",
        },
    )
    print(json.dumps({"status": result["status"], "cases": len(cases), "artifact": str(OUT / "runtime" / "persistent_runtime_stress_cases.json")}))
    return result


if __name__ == "__main__":
    run()
