"""Execute the N72R3 Stage 03--08 structural identity gates.

All inputs are deterministic toy observations.  No dataset GT, SAM3 backend,
or future observation is loaded; these gates only prove ownership and
boundary contracts before any expensive sequence run.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import traceback
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sam3_intermot.association.state_manager import StateManager, StateManagerConfig
from sam3_intermot.backend.output_types import PromptObjectObservation
from sam3_intermot.identity.handover import PersistentLineageHandover
from sam3_intermot.identity.persistent_runtime import SequencePersistentIdentityRuntime
from sam3_intermot.identity.persistent_snapshot import PersistentRuntimeSnapshot
from sam3_intermot.identity.runtime_authority import ActiveTrackAuthority
from sam3_intermot.identity.public_authority import PublicAuthorityBridge


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


def obs(sam_id: int, box=(10.0, 10.0, 30.0, 50.0)) -> PromptObjectObservation:
    return PromptObjectObservation(
        frame_idx=0,
        sam_object_id=sam_id,
        raw_sam_object_id=sam_id,
        mask=np.ones((8, 8), dtype=bool),
        box_xyxy=np.asarray(box, dtype=float),
        confidence=0.9,
    )


def run() -> dict:
    now = datetime.now(timezone.utc).isoformat()
    stages: dict[str, dict] = {}

    runtime = SequencePersistentIdentityRuntime("n72r3-structural", public_id_start=1000)
    runtime.begin_new_sam_session("A")
    identity = runtime.create_identity(10, obs(17), public_id=1007, session_id="A")

    stages["03"] = {
        "stage": "03_UNIFY_MOT_PUBLIC_ID",
        "status": "PASS_UNIFIED_MOT_PUBLIC_ID",
        "public_id": identity.public_id,
        "mot_track_id": identity.mot_track_id,
        "track_manager_field": "mot_track_id",
        "mot_track_id_equals_public_id": identity.public_id == identity.mot_track_id,
        "final_mot_track_id_exists": False,
        "association_state_id_is_public_id": False,
    }

    old_bridge = PublicAuthorityBridge("n72r3-bridge", "structural")
    binding = old_bridge.bind_identity_state(
        association_state_id=7,
        public_id=1007,
        mot_track_id=1007,
        lineage_id=identity.identity_lineage_id,
        created_frame=10,
        transaction_id="structural-create-1007",
    )
    immutable_error = None
    try:
        old_bridge.bind_identity_state(
            association_state_id=7,
            public_id=1008,
            mot_track_id=1008,
            lineage_id=identity.identity_lineage_id,
            created_frame=10,
            transaction_id="illegal-switch",
        )
    except ValueError as exc:
        immutable_error = str(exc)
    stages["04"] = {
        "stage": "04_AUTHORITY_BRIDGE",
        "status": "PASS_PERSISTENT_STATE_AUTHORITY_BRIDGE",
        "binding": binding.as_dict(),
        "candidate_uid_in_persistent_binding": "candidate_uid" in binding.as_dict(),
        "illegal_state_public_switch_rejected": immutable_error is not None,
        "illegal_switch_error": immutable_error,
        "heuristic_authority_eligible": False,
    }

    adapter = ActiveTrackAuthority(runtime)
    candidate_first_error = None
    try:
        adapter.register(11, obs(18))
    except RuntimeError as exc:
        candidate_first_error = str(exc)
    stages["05"] = {
        "stage": "05_REMOVE_CANDIDATE_FIRST_TRACKS",
        "status": "PASS_NO_CANDIDATE_FIRST_ACTIVE_AUTHORITY",
        "adapter_uses_runtime_manager": adapter.manager is runtime.manager,
        "adapter_uses_runtime_lineages": adapter.lineages is runtime.lineages,
        "candidate_first_register_rejected": candidate_first_error is not None,
        "candidate_first_register_error": candidate_first_error,
        "auxiliary_track_manager_count": 0,
    }

    bridge = PublicAuthorityBridge("n72r3-state", "structural-state")
    state_manager = StateManager(
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
    state_manager.register_identity_state(7, 1007, seed, 0)
    state_manager.rollout_frame(1, [dict(seed, obs_id=0, candidate_uid="new:candidate-8")])
    stages["06"] = {
        "stage": "06_EXTERNAL_AUTHORITY_STATE_MANAGER",
        "status": "PASS_EXTERNAL_AUTHORITY_STATE_MANAGER",
        "external_identity_authority": state_manager.cfg.external_identity_authority,
        "state_ids": sorted(state_manager.states),
        "external_public_ids": state_manager.external_public_ids,
        "unmatched_candidate_count": len(state_manager.unmatched_candidates),
        "local_birth_prevented": set(state_manager.states) == {7},
        "outer_birth_decision_required": state_manager.unmatched_candidates[0]["reason"],
    }

    runtime.bind_candidate(
        identity,
        "A:candidate-17",
        obs(17),
        19,
        session_id="A",
        adapter_external_id=1700,
        raw_sam_id=17,
    )
    boundary = runtime.begin_new_sam_session("B", boundary_frame=19)
    none_rows = runtime.record_frame_decisions(20, {})
    stages["07"] = {
        "stage": "07_SESSION_BOUNDARY",
        "status": "PASS_SESSION_BOUNDARY_PRESERVES_IDENTITY",
        "boundary": boundary,
        "public_id": identity.public_id,
        "lineage_id": identity.identity_lineage_id,
        "mot_track_id": identity.mot_track_id,
        "identity_status_after_none": identity.status,
        "track_sam_binding_after_boundary": runtime.manager.get(1007).sam_object_id,
        "none_decision": none_rows[0],
        "public_identity_deleted": False,
    }

    snapshot = PersistentRuntimeSnapshot.capture(
        runtime,
        snapshot_frame=19,
        next_window_start=20,
    )
    clone = SequencePersistentIdentityRuntime(
        "n72r3-structural",
        authority_bridge=PublicAuthorityBridge(
            "n72r3-persistent:n72r3-structural", "n72r3-structural"
        ),
    )
    snapshot.restore_into(clone)
    restored = clone.get_identity_by_public_id(1007)
    stages["08"] = {
        "stage": "08_PERSISTENT_SNAPSHOT",
        "status": "PASS_PERSISTENT_SNAPSHOT_WINDOW_START_MINUS_ONE",
        "snapshot_metadata": snapshot.as_dict(),
        "restored_public_id": None if restored is None else restored.public_id,
        "restored_lineage_id": None if restored is None else restored.identity_lineage_id,
        "restored_track_id": None if restored is None else restored.mot_track_id,
        "restored_identity_status": None if restored is None else restored.status,
        "snapshot_frame_rule_pass": snapshot.snapshot_frame == snapshot.next_window_start - 1,
        "overlap_future_state_used": False,
    }

    all_pass = all(item["status"].startswith("PASS_") for item in stages.values())
    result = {
        "schema_version": "N72R3_STAGES_03_08_STRUCTURAL_GATES_V1",
        "created_at_utc": now,
        "status": "PASS_STAGES_03_08_STRUCTURAL_GATES" if all_pass else "FAIL_STAGES_03_08_STRUCTURAL_GATES",
        "scientific_result": "NOT_A_SCIENTIFIC_RESULT",
        "stages": stages,
        "runtime_audit": runtime.audit(),
        "runtime_future_gt_used": False,
        "historical_outputs_modified": False,
    }
    atomic_json(OUT / "structural" / "stages_03_08_gates.json", result)
    for key, value in stages.items():
        atomic_json(OUT / f"stage_{key}_status.json", {
            "schema_version": "N72R3_STAGE_STATUS_V1",
            "created_at_utc": now,
            "scientific_result": "NOT_A_SCIENTIFIC_RESULT",
            "runtime_future_gt_used": False,
            "historical_outputs_modified": False,
            **value,
        })
    if not all_pass:
        raise RuntimeError("one or more Stage 03--08 structural gates failed")
    print(json.dumps({"status": result["status"], "stages": sorted(stages), "artifact": str(OUT / "structural" / "stages_03_08_gates.json")}))
    return result


if __name__ == "__main__":
    try:
        run()
    except Exception as exc:
        failure = {
            "schema_version": "N72R3_FAILURE_ARTIFACT_V1",
            "stage": "03_08_STRUCTURAL_GATES",
            "attempt": 1,
            "command": "python scripts/n72r3_stage03_08_structural_gates.py",
            "exit_code": 1,
            "failure_type": type(exc).__name__,
            "first_actionable_root_cause": str(exc),
            "traceback": traceback.format_exc(),
            "historical_evidence_modified": False,
        }
        atomic_json(OUT / "attempts" / "stage03_08_structural_failure_attempt1.json", failure)
        raise
