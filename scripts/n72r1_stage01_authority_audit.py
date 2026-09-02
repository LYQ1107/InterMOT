"""Audit identity namespaces before implementing N72R1 joins."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
N72R1_ROOT = Path("/data2/usr_for_deadline/SAM3_InterMOT_N72R1")


def sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def evidence(path: str, needles: list[str]) -> list[dict[str, object]]:
    file_path = ROOT / path
    if not file_path.is_file():
        return [{"path": path, "missing": True}]
    lines = file_path.read_text(encoding="utf-8", errors="replace").splitlines()
    result = []
    for needle in needles:
        matches = [index + 1 for index, line in enumerate(lines) if needle in line]
        result.append({"path": path, "needle": needle, "lines": matches[:12]})
    return result


def main() -> None:
    audit_root = N72R1_ROOT / "audits"
    status_root = N72R1_ROOT / "status"
    audit_root.mkdir(parents=True, exist_ok=True)
    status_root.mkdir(parents=True, exist_ok=True)

    source_files = {
        "backend": "sam3_intermot/backend/sam3_backend.py",
        "output_types": "sam3_intermot/backend/output_types.py",
        "state_manager": "sam3_intermot/association/state_manager.py",
        "n35_export": "scripts/run_n35_export_tape.py",
        "n36_export": "scripts/run_n36_export_chunk.py",
        "continuous_observer": "sam3_intermot/interaction/continuous_observer.py",
        "mot_export": "sam3_intermot/evaluation/mot_export.py",
        "frame_output": "sam3_intermot/evaluation/frame_output.py",
        "identity_namespace": "sam3_intermot/identity/namespace.py",
        "identity_registry": "sam3_intermot/identity/registry.py",
        "track_manager": "sam3_intermot/tracking/track_manager.py",
    }
    file_hashes = {name: {"path": path, "sha256": sha256(ROOT / path)} for name, path in source_files.items()}

    observations = evidence(
        source_files["backend"],
        [
            "raw_sam_object_id=int(oid)",
            "obs.sam_object_id = ext",
            '"native_tid": int(observation.sam_object_id)',
            '"native_id_source": "official_out_obj_ids"',
        ],
    )
    state_evidence = evidence(
        source_files["state_manager"],
        [
            "def _new_pid",
            'public_id_order = [int(s.pid) for s in states]',
            '"public_id": int(states[state_index].pid)',
            'st = IdentityState(pid, obs["feat"]',
        ],
    )
    active_evidence = evidence(
        source_files["continuous_observer"],
        [
            "ObjectIdentityRegistry",
            "self.manager.create_track",
            "track.mot_track_id",
            "IdentityLineageRegistry",
        ],
    )
    export_evidence = evidence(
        source_files["mot_export"],
        [
            "observation_to_mot_row",
            "track_id",
        ],
    )
    namespace_evidence = evidence(
        source_files["identity_namespace"],
        [
            "lineage_to_public",
            "PublicTrackIDAllocator",
            "create_user",
        ],
    )

    audit = {
        "schema_version": "N72R1_PUBLIC_ID_AUTHORITY_AUDIT_V1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_root": str(ROOT),
        "source_files": file_hashes,
        "final_mot_public_id_source": (
            "TrackManager.mot_track_id, emitted by FrameOutputAssembler/standard MOT export "
            "in the continuous-observer runtime; this is distinct from StateManager.pid."
        ),
        "state_manager_pid_semantics": (
            "Association-local state identifier allocated by StateManager._new_pid. "
            "N35/N36 candidate exports place it in public-shaped fields, but no explicit "
            "bridge to TrackManager.mot_track_id or PublicTrackIDAllocator is present."
        ),
        "track_manager_id_semantics": (
            "TrackManager.mot_track_id is the ID passed to FrameOutputAssembler and MOT export "
            "in the active interaction runtime; it is not a SAM raw ID."
        ),
        "identity_namespace_used_in_active_path": False,
        "explicit_pid_to_public_bridge_exists": False,
        "candidate_and_assignment_same_run": False,
        "candidate_branch_semantics": "official SAM3 observations -> StateManager association state; no proven final public resolver",
        "active_output_semantics": "continuous observer -> TrackManager/FrameOutputAssembler -> MOT track_id",
        "candidate_public_ids_field_semantics": "StateManager association-local pid in N35/N36 audits, not proven public ID",
        "native_tid_semantics": "adapter-visible stable SAM ID after binding in backend legacy export; official raw out_obj_ids are separate raw_sam_object_id",
        "first_missing_join_edge": "StateManager association_state_id -> active TrackManager.mot_track_id/public authority",
        "unsafe_numeric_equivalences": [
            "StateManager.pid == public_mot_id",
            "StateManager.pid == TrackManager.mot_track_id",
            "sam_object_id == public_mot_id",
            "raw_sam_object_id == public_mot_id",
            "native_tid == final MOT public ID",
            "identity_lineage_id == public_mot_id without IdentityNamespace/allocator bridge",
        ],
        "evidence": {
            "backend_raw_and_stable": observations,
            "state_manager": state_evidence,
            "active_runtime": active_evidence,
            "mot_export": export_evidence,
            "identity_namespace": namespace_evidence,
        },
        "n72r1_rule": "Until an explicit runtime resolver is supplied, sidecars use association_state_id and public_id=null with PUBLIC_ASSIGNMENT_ARTIFACT_ABSENT.",
    }
    audit_path = audit_root / "public_id_authority_audit.json"
    audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    lines = [
        "# Public-ID authority audit",
        "",
        "## Result",
        "",
        "The active MOT output path uses `TrackManager.mot_track_id` through the frame-output/MOT serializer. The candidate exporter and `StateManager` use a separate association-local PID. No explicit PID-to-public bridge is present in the audited active candidate path.",
        "",
        "Therefore N72R1 must not write a `StateManager.pid` as `public_id`. Without a supplied resolver, the V2 assignment sidecar records `association_state_id`, leaves `public_id` null, and uses `PUBLIC_ASSIGNMENT_ARTIFACT_ABSENT`.",
        "",
        "## Namespace decisions",
        "",
        f"- Final MOT public-ID source: `{audit['final_mot_public_id_source']}`",
        f"- StateManager PID: `{audit['state_manager_pid_semantics']}`",
        f"- IdentityNamespace active-path usage: `{audit['identity_namespace_used_in_active_path']}`",
        f"- Explicit bridge: `{audit['explicit_pid_to_public_bridge_exists']}`",
        f"- Candidate/assignment same authoritative run: `{audit['candidate_and_assignment_same_run']}`",
        f"- First missing join edge: `{audit['first_missing_join_edge']}`",
        "",
        "The complete machine-readable evidence and source hashes are in `public_id_authority_audit.json`.",
    ]
    (audit_root / "PUBLIC_ID_AUTHORITY_AUDIT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    namespace_table = {
        "schema_version": "N72R1_ID_NAMESPACE_TABLE_V1",
        "rows": [
            {"name": "official_raw_sam_id", "owner": "SAM3 official response", "may_be_public": False},
            {"name": "adapter_external_id", "owner": "Sam3Backend stable binding", "may_be_public": False},
            {"name": "segment_local_id", "owner": "N72R1 segment adapter/handover", "may_be_public": False},
            {"name": "sequence_global_id", "owner": "N72R1 explicit handover ledger", "may_be_public": False},
            {"name": "association_state_id", "owner": "StateManager._new_pid", "may_be_public": False},
            {"name": "mot_track_id", "owner": "TrackManager/FrameOutputAssembler", "may_be_public": True, "authority_condition": "same active runtime output path"},
            {"name": "public_mot_id", "owner": "IdentityNamespace/PublicTrackIDAllocator when explicitly bridged", "may_be_public": True, "authority_condition": "explicit resolver transaction"},
            {"name": "dataset_gt_id", "owner": "offline evaluator only", "may_be_public": False},
        ],
    }
    (audit_root / "id_namespace_table.json").write_text(json.dumps(namespace_table, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    status = {
        "schema_version": "N72R1_STAGE_STATUS_V1",
        "stage": "01",
        "status": "PASS_AUTHORITY_AUDITED_BRIDGE_UNPROVEN",
        "audit": str(audit_path),
        "namespace_table": str(audit_root / "id_namespace_table.json"),
        "explicit_pid_to_public_bridge_exists": False,
        "next_stage": "02",
        "downstream_rule": "Do not emit StateManager.pid as public_id; use association_state_id until explicit resolver is supplied.",
        "runtime_future_gt_used": False,
    }
    (status_root / "stage_01_status.json").write_text(json.dumps(status, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(status, sort_keys=True))


if __name__ == "__main__":
    main()
