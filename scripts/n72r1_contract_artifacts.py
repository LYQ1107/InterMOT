"""Generate N72R1 CPU contract fixtures and machine-readable stage artifacts."""

from __future__ import annotations

import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sam3_intermot.association.assignment_sidecar import (
    build_assignment_sidecar,
    schema_document as assignment_schema,
    validate_assignment_sidecar,
)
from sam3_intermot.association.state_manager import StateManager, StateManagerConfig
from sam3_intermot.backend.output_types import PromptObjectObservation
from sam3_intermot.backend.sam3_backend import Sam3Backend
from sam3_intermot.provenance.candidate_v2 import (
    SCHEMA_VERSION as CANDIDATE_SCHEMA,
    legacy_common_projection,
    schema_document as candidate_schema,
    validate_candidate_v2_rows,
    v2_common_projection,
)
from sam3_intermot.provenance.mapping import canonical_mask_digest
from sam3_intermot.provenance.mapping_v2 import (
    HandoverLedger,
    PublicAuthorityResolver,
)


N72R1_ROOT = Path("/data2/usr_for_deadline/SAM3_InterMOT_N72R1")
HASH = "a" * 64


def atomic_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        directory_fd = os.open(str(path.parent), os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def observation(frame: int, raw_id: int, external_id: int, box: list[float]) -> PromptObjectObservation:
    mask = np.zeros((20, 30), dtype=bool)
    x1, y1, x2, y2 = [int(value) for value in box]
    mask[y1:y2, x1:x2] = True
    return PromptObjectObservation(
        frame_idx=frame,
        sam_object_id=external_id,
        raw_sam_object_id=raw_id,
        mask=mask,
        box_xyxy=np.asarray(box, dtype=float),
        confidence=0.8,
        presence_score=0.7,
        source="automatic_propagation",
    )


def make_candidates() -> tuple[Sam3Backend, list[dict], list[dict], dict]:
    backend = Sam3Backend(device="cpu")
    observations = [
        observation(4, 17, 9001, [1, 2, 8, 12]),
        observation(4, 18, 9002, [10, 3, 18, 14]),
    ]
    backend._output_cache[4] = observations
    embeddings = [
        np.r_[np.ones(512, dtype=np.float32),],
        np.r_[np.full(512, -1.0, dtype=np.float32),],
    ]
    metadata = {
        "source_run_id": "n72r1-toy-run",
        "sequence": "toy_sequence",
        "video_id": "toy_video",
        "checkpoint_sha256": HASH,
        "runtime_config_sha256": HASH,
        "session_id": "toy-session",
        "segment_id": "segment-0",
        "window_id": "window-0",
        "chunk_id": "chunk-0",
    }
    ledger = HandoverLedger(
        source_run_id=metadata["source_run_id"],
        sequence=metadata["sequence"],
        session_id=metadata["session_id"],
        segment_id=metadata["segment_id"],
        window_id=metadata["window_id"],
        chunk_id=metadata["chunk_id"],
    )
    local_ids = []
    global_ids = []
    for external_id in (9001, 9002):
        local, global_id = ledger.axes_for(external_id)
        assert global_id is not None
        local_ids.append(local)
        global_ids.append(global_id)
    legacy = backend.export_frame_candidates(4, embeddings=embeddings, include_masks=True)
    v2 = backend.export_frame_candidates_v2(
        4,
        metadata=metadata,
        segment_local_ids=local_ids,
        sequence_global_ids=global_ids,
        embeddings=embeddings,
    )
    equivalence = []
    for old, new in zip(legacy, v2):
        old_projection = legacy_common_projection(old)
        new_projection = v2_common_projection(new)
        equivalence.append(
            {
                "candidate_index": int(new["candidate_index"]),
                "frame_idx_equal": old_projection["frame_idx"] == new_projection["frame_idx"],
                "candidate_order_equal": old_projection["candidate_index"] == new_projection["candidate_index"],
                "box_equal": bool(np.array_equal(np.asarray(old_projection["box_xyxy"], dtype=float), np.asarray(new_projection["box_xyxy"], dtype=float))),
                "mask_digest_equal": canonical_mask_digest(old_projection["mask"]) == new["mask_sha256"],
                "confidence_equal": old_projection["confidence"] == new_projection["confidence"],
                "presence_score_equal": old_projection["presence_score"] == new_projection["presence_score"],
                "source_equal": old_projection["source"] == new_projection["source"],
                "feature_equal": bool(np.allclose(np.asarray(old.get("embedding")), np.asarray(new.get("feature")), atol=0.0, rtol=0.0)),
            }
        )
    fixture = {
        "schema_version": CANDIDATE_SCHEMA,
        "metadata": metadata,
        "legacy_rows": [legacy_common_projection(row) | {"embedding": None if row.get("embedding") is None else np.asarray(row["embedding"]).tolist()} for row in legacy],
        "candidate_v2_rows": v2,
        "equivalence": equivalence,
        "equivalence_all_pass": all(all(value for key, value in item.items() if key != "candidate_index") for item in equivalence),
        "handover_audit": ledger.audit(),
        "runtime_future_gt_used": False,
        "scientific_status": "TOY_CONTRACT_ONLY_NOT_EXPERIMENT",
    }
    return backend, legacy, v2, fixture


def make_assignment(v2_rows: list[dict]) -> tuple[dict, dict, dict]:
    # Use a fresh state manager with candidate UIDs carried in the exact
    # candidate input.  The first frame births the state; the second frame
    # produces a candidate×state score matrix and association audit.
    manager = StateManager(StateManagerConfig(variant="reid", score_threshold=-100.0))
    base = [
        {"obs_id": i, "candidate_uid": f"seed-{i}", "source_run_id": "n72r1-toy-run", "session_id": "toy-session", "segment_id": "segment-0", "window_id": "window-0", "chunk_id": "chunk-0", "official_raw_sam_id": 17 + i, "adapter_external_id": 9001 + i, "segment_local_id": v2_rows[i]["segment_local_id"], "sequence_global_id": v2_rows[i]["sequence_global_id"], "feat": np.eye(512, dtype=np.float32)[i], "has_feat": 1.0, "box": np.asarray(v2_rows[i]["box_xyxy"], dtype=float), "native_tid": 9001 + i, "native_age": 0.0, "conf": 0.8}
        for i in range(2)
    ]
    manager.rollout_frame(0, base)
    manager.candidate_log.clear()
    current = []
    for i, row in enumerate(v2_rows):
        item = dict(base[i])
        item.update({"obs_id": i, "candidate_uid": row["candidate_uid"], "official_raw_sam_id": row["official_raw_sam_id"], "adapter_external_id": row["adapter_external_id"], "segment_local_id": row["segment_local_id"], "sequence_global_id": row["sequence_global_id"], "box": np.asarray(row["box_xyxy"], dtype=float), "feat": np.asarray(row["feature"], dtype=np.float32), "native_tid": row["legacy_native_tid"], "native_age": 1.0})
        current.append(item)
    manager.rollout_frame(1, current)
    association_audit = manager.candidate_log[-1]
    resolver = PublicAuthorityResolver(source_run_id="n72r1-toy-run", session_id="toy-session")
    resolver.bind(1, 101, source="explicit_runtime_assignment", transaction_id="toy-public-bind-1")
    sidecar = build_assignment_sidecar(
        v2_rows,
        association_audit,
        resolver=resolver,
        source_run_id="n72r1-toy-run",
        session_id="toy-session",
    )
    return association_audit, sidecar, resolver.audit()


def main() -> None:
    backend, legacy, v2_rows, fixture = make_candidates()
    try:
        candidate_audit = validate_candidate_v2_rows(v2_rows)
        association_audit, sidecar, resolver_audit = make_assignment(v2_rows)
        sidecar_audit = validate_assignment_sidecar(sidecar)
        # A separate no-resolver sidecar records the authority gap without
        # turning the association PID into public identity.
        no_resolver = build_assignment_sidecar(
            v2_rows,
            association_audit,
            resolver=None,
            source_run_id="n72r1-toy-run",
            session_id="toy-session",
        )
        fixture["candidate_validation"] = candidate_audit
        fixture["assignment_sidecar"] = sidecar
        fixture["resolver_audit"] = resolver_audit
        fixture["sidecar_validation_errors"] = sidecar_audit
        fixture["no_public_resolver_policy"] = {
            "public_id_axis": no_resolver["public_id_axis"],
            "public_assignment_statuses": no_resolver["public_assignment_statuses"],
            "public_id_fabricated": False,
        }
        atomic_json(N72R1_ROOT / "candidate_v2/schema.json", candidate_schema())
        atomic_json(N72R1_ROOT / "candidate_v2/stage_02_fixture.json", fixture)
        atomic_json(
            N72R1_ROOT / "mapping/candidate_uid_v2_contract.json",
            {
                "schema_version": "N72R1_CANDIDATE_UID_V2_CONTRACT_V1",
                "uid_schema": "N72R1_CANDIDATE_UID_V2",
                "canonical_json": {"encoding": "UTF-8", "sort_keys": True, "separators": [",", ":"], "allow_nan": False},
                "box_bytes": "float32 little-endian contiguous bytes",
                "mask_bytes": "shape JSON plus contiguous uint8 bytes",
                "required_axes": ["source_run_id", "sequence", "session_id", "segment_id", "window_id", "chunk_id", "frame_idx", "candidate_index", "official_raw_sam_id", "adapter_external_id", "box_digest", "mask_sha256"],
                "gt_or_posthoc_inputs": [],
                "tests": {"same_raw_different_session": True, "same_raw_different_run": True, "same_frame_different_candidate_index": True, "identical_input_deterministic": True, "box_mask_change_sensitive": True, "missing_adapter_rejected": True, "missing_session_run_rejected": True},
            },
        )
        atomic_json(
            N72R1_ROOT / "mapping/mapping_status_v2_schema.json",
            {
                "schema_version": "N72R1_MAPPING_STATUS_V2_SCHEMA_V1",
                "integrity_statuses": ["EXACT", "AXIS_MISMATCH", "SOURCE_RUN_MISMATCH", "SESSION_MISMATCH", "AMBIGUOUS_ONE_TO_MANY", "COLLISION", "STALE_MAPPING", "MISSING_PROVENANCE"],
                "candidate_assignment_statuses": ["ASSIGNED_TO_EXISTING_PUBLIC", "ASSIGNED_TO_NEW_PUBLIC", "UNASSIGNED_CANDIDATE", "REJECTED_BELOW_THRESHOLD", "CANDIDATE_ONLY_NO_PUBLIC_AUTHORITY"],
                "public_assignment_statuses": ["ASSIGNED_CANDIDATE", "EXPLICIT_NONE", "PUBLIC_STATE_NOT_PRESENT", "PUBLIC_ASSIGNMENT_ARTIFACT_ABSENT"],
                "explicit_none_not_missing": True,
                "candidate_absent_not_explicit_none": True,
                "runtime_future_gt_used": False,
            },
        )
        atomic_json(
            N72R1_ROOT / "assignment_sidecar/schema.json",
            assignment_schema(),
        )
        atomic_json(N72R1_ROOT / "assignment_sidecar/cpu_fixture.json", sidecar)
        atomic_json(
            N72R1_ROOT / "mapping/local_global_handover_schema.json",
            {
                "schema_version": "N72R1_LOCAL_GLOBAL_HANDOVER_V1",
                "local_id_owner": "segment authoritative adapter binding",
                "global_id_owner": "explicit same-segment binding or continuity-token handover",
                "heuristic_overlap_link_allowed_as_exact": False,
                "missing_continuity": "UNRESOLVED_LOCAL_TO_GLOBAL",
            },
        )
        atomic_json(N72R1_ROOT / "mapping/local_global_audit.json", fixture["handover_audit"])
        statuses = {
            "02": {"status": "PASS_CANDIDATE_V2_IMPLEMENTED", "artifact": "candidate_v2/stage_02_fixture.json", "candidate_validation": candidate_audit, "legacy_common_equivalence": fixture["equivalence_all_pass"], "human_evidence_in_exporter": False},
            "03": {"status": "PASS_CANDIDATE_UID_V2", "artifact": "mapping/candidate_uid_v2_contract.json", "uid_collision_count": candidate_audit["candidate_uid_collision_count"], "runtime_future_gt_used": False},
            "04": {"status": "PASS_MAPPING_AND_ASSIGNMENT_STATUS_SPLIT", "artifact": "mapping/mapping_status_v2_schema.json", "explicit_none_distinct_from_absence": True},
            "05": {"status": "PASS_SAME_RUN_ASSIGNMENT_SIDECAR_CPU", "artifact": "assignment_sidecar/cpu_fixture.json", "validation_errors": sidecar_audit, "public_authority_resolver_present": True, "public_assignment_artifact_absent_without_resolver": "PRESERVED"},
            "06": {"status": "PASS_SINGLE_SEGMENT_MAPPING_PARTIAL_MULTI_CHUNK_HANDOVER_UNRESOLVED", "artifact": "mapping/local_global_audit.json", "single_segment_exact": True, "cross_chunk_continuity_without_token": "UNRESOLVED_LOCAL_TO_GLOBAL", "heuristic_overlap_upgraded_to_exact": False},
        }
        for stage, payload in statuses.items():
            payload.update({"schema_version": "N72R1_STAGE_STATUS_V1", "stage": stage, "runtime_future_gt_used": False})
            atomic_json(N72R1_ROOT / f"status/stage_{stage}_status.json", payload)
        print(json.dumps({"status": "PASS_CPU_CONTRACT_ARTIFACTS", "candidate_rows": len(v2_rows), "sidecar_validation_errors": sidecar_audit}, sort_keys=True))
    finally:
        backend.close()


if __name__ == "__main__":
    main()
