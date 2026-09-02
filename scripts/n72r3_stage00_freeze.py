"""Freeze the N72R3 persistent-public-identity protocol.

The N72R3 worktree is an isolated copy of the N72R2 proven code.  Historical
outputs, the shared checkpoint, and the official SAM3 checkout are inventory
inputs only; this script writes only the new N72R3 output root.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "outputs" / "N72R3"
SOURCE = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/Interactive/SAM3_InterMOT")
N72R2 = Path("/data2/usr_for_deadline/SAM3_InterMOT_N72R2")
N72R1 = Path("/data2/usr_for_deadline/SAM3_InterMOT_N72R1")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
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


def file_record(path: Path) -> dict[str, object]:
    if not path.is_file():
        return {"path": str(path), "exists": False}
    stat = path.stat()
    return {
        "path": str(path),
        "exists": True,
        "size": stat.st_size,
        "sha256": sha256(path),
    }


def records(paths: Iterable[Path]) -> list[dict[str, object]]:
    return [file_record(path) for path in paths]


def git_state(path: Path) -> dict[str, object]:
    try:
        result = subprocess.run(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError as exc:
        return {"is_repository": False, "error": str(exc)}
    return {
        "is_repository": result.returncode == 0,
        "head": result.stdout.strip() if result.returncode == 0 else None,
        "stderr": result.stderr.strip() if result.returncode else None,
    }


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    now = datetime.now(timezone.utc).isoformat()
    n72r2_report = N72R2 / "worktree/docs/N72R2_FINAL_REPORT.md"
    n72r2_gate = N72R2 / "outputs/N72R2/n72r2_final_gate.json"
    historical = [
        n72r2_report,
        n72r2_gate,
        N72R2 / "outputs/N72R2/stage_02_status_final.json",
        N72R1 / "reports/N72R1_FINAL_REPORT.md",
        SOURCE / "docs/N72_FINAL_REPORT.md",
        SOURCE / "docs/N71_FINAL_REPORT.md",
        SOURCE / "docs/N70_FINAL_REPORT.md",
        SOURCE / "AGENTS.md",
        SOURCE / "research_log.md",
    ]
    source_files = [
        ROOT / "sam3_intermot/identity/runtime_authority.py",
        ROOT / "sam3_intermot/identity/public_authority.py",
        ROOT / "sam3_intermot/identity/handover.py",
        ROOT / "sam3_intermot/tracking/track_manager.py",
        ROOT / "sam3_intermot/association/state_manager.py",
        ROOT / "sam3_intermot/interaction/continuous_observer.py",
        ROOT / "sam3_intermot/identity/namespace.py",
        ROOT / "sam3_intermot/identity/lineage.py",
        ROOT / "sam3_intermot/backend/sam3_state_snapshot.py",
    ]
    checkpoint = SOURCE / "checkpoints/sam3.1_mirror/sam3.1_multiplex.pt"
    protection_path = OUTPUT / "protection_manifest.json"
    protection = {
        "schema_version": "N72R3_PROTECTION_MANIFEST_V1",
        "created_at_utc": now,
        "source_root": str(SOURCE),
        "input_n72r2_root": str(N72R2),
        "new_root": str(ROOT.parent),
        "source_git": git_state(SOURCE),
        "historical_inputs": records(historical),
        "worktree_baseline": records(source_files),
        "checkpoint": file_record(checkpoint),
        "third_party_root": str(SOURCE / "third_party/sam3"),
        "policy": {
            "historical_outputs_read_only": True,
            "source_root_not_modified": True,
            "third_party_sam3_not_modified": True,
            "checkpoint_not_modified": True,
            "all_new_artifacts_under": str(OUTPUT),
            "no_real_human_tape_created": True,
        },
    }
    atomic_json(protection_path, protection)
    protocol = {
        "schema_version": "N72R3_PROTOCOL_V1",
        "name": "N72R3_PERSISTENT_PUBLIC_IDENTITY_ACROSS_INDEPENDENT_SAM_SESSIONS",
        "created_at_utc": now,
        "source_root": str(SOURCE),
        "input_n72r2_root": str(N72R2),
        "isolated_worktree": str(ROOT),
        "output_root": str(OUTPUT),
        "old_n72r2_13_of_13_gate": "RETIRED_AS_INCORRECT_IDENTITY_PREREQUISITE",
        "public_identity_boundary_rule": (
            "persistent public identity survives session boundaries independently of candidate presence"
        ),
        "identity_axes": {
            "sam_raw_id": "session_local",
            "adapter_external_id": "session_local",
            "candidate_uid": "observation_or_run_local",
            "sam_mask_state": "session_local",
            "public_id": "sequence_persistent",
            "mot_track_id": "sequence_persistent_and_equal_to_public_id",
            "identity_lineage_id": "sequence_persistent",
            "association_identity_state": "sequence_persistent",
            "appearance_memory": "sequence_persistent",
            "motion_lost_history": "sequence_persistent",
        },
        "invariants": {
            "I1_public_id_belongs_to_identity": True,
            "I2_public_id_immutable_within_lineage": True,
            "I3_state_has_at_most_one_public_id": True,
            "I4_session_reset_preserves_identity_state": True,
            "I5_none_at_boundary_is_valid_lost_state": True,
            "I6_new_candidate_reactivates_existing_identity": True,
            "I7_heuristics_never_create_exact_authority": True,
            "I8_gt_identity_never_runtime_public_id": True,
            "I9_current_gt_only_after_y_pre_freeze_for_simulator": True,
            "I10_future_gt_never_runtime": True,
        },
        "frozen_inputs": {
            "candidate_definition": "N72R1 Candidate V2; no candidate-generation change",
            "checkpoint": file_record(checkpoint),
            "n72r2_final_report": file_record(n72r2_report),
            "n72r2_final_gate": file_record(n72r2_gate),
        },
        "fixed_windows": {
            "frame_window_length": 160,
            "overlap_frames": 20,
            "handover_gate": [1, 2, 6],
            "snapshot_frame_rule": "window_B_start_minus_one",
            "overlap_is_diagnostic_only": True,
        },
        "future_windows": [20, 50, 100],
        "bootstrap": {"unit": "independent_sequence", "repetitions": 2000, "seed": 7202},
        "actions": [
            "AUTHORITATIVE_CORRECT",
            "AUTHORITATIVE_REASSIGN",
            "ATOMIC_ID_SWAP",
            "ADD_NEW_IDENTITY",
            "RECOVER_IDENTITY",
            "AUTHORITATIVE_DELETE",
        ],
        "variants": [
            "NO_INTERVENTION",
            "M0_CURRENT_FRAME_CORRECTION_ONLY",
            "M1_HUMAN_EMA_PROTOTYPE",
            "M2_POSITIVE_HUMAN_ANCHORS",
            "M3_NEGATIVE_COMPETITOR_BANK",
            "M4_RELIABILITY_AGE_ADMISSION",
        ],
        "runtime_causal_contract": {
            "runtime_future_gt_used": False,
            "future_gt_runtime_reads": False,
            "event_frame_memory_read": False,
            "first_memory_read_offset": 1,
            "posthoc_gt_only_after_runtime_artifact_frozen": True,
            "simulated_oracle_reads_current_gt_only_after_y_pre_freeze": True,
        },
        "resource_limits": {
            "max_gpus": 4,
            "one_sequence_or_frame_range_per_gpu": True,
            "independent_process_per_session_window": True,
            "oom_sharding": [160, 100, 50],
        },
        "non_goals": [
            "candidate_recall_as_identity_continuity_prerequisite",
            "heuristic_iou_or_appearance_exact_authority",
            "checkpoint_change",
            "candidate_definition_change",
            "hungarian_solver_change",
            "metric_or_bootstrap_change",
            "real_human_tape_claim",
            "training_before_structural_and_future_effect_gates",
        ],
        "stage_sequence": [
            "00_FREEZE_PROTOCOL",
            "01_N72R2_AUTHORITY_SEMANTICS_AUDIT",
            "02_PERSISTENT_RUNTIME",
            "03_UNIFY_MOT_PUBLIC_ID",
            "04_AUTHORITY_BRIDGE",
            "05_REMOVE_CANDIDATE_FIRST_TRACKS",
            "06_EXTERNAL_AUTHORITY_STATE_MANAGER",
            "07_SESSION_BOUNDARY",
            "08_PERSISTENT_SNAPSHOT",
            "09_TWO_WINDOW_GATE",
            "10_SIX_WINDOW_GATE",
            "11_OPTIONAL_CANDIDATE_RECOVERY",
            "12_GT_AUDIT_FIX",
            "13_SIMULATED_HUMAN_ORACLE",
            "14_EVENT_POLICY",
            "15_ATOMIC_RUNTIME_TRANSACTION",
            "16_OFFICIAL_CURRENT_FRAME_CORRECTION",
            "17_REAL_APPEARANCE_MEMORY",
            "18_EXACT_PUBLIC_BASELINE",
            "19_CANDIDATE_RECALL_DIAGNOSIS",
            "20_GT_SIMULATED_EFFECT_EXPERIMENT",
            "21_TARGET_SCOPED_ASSOCIATION",
            "22_STRICT_FUTURE_EFFECT_GATE",
        ],
        "protection_manifest": str(protection_path),
    }
    protocol_path = OUTPUT / "protocol.json"
    atomic_json(protocol_path, protocol)
    atomic_json(
        OUTPUT / "stage_00_status.json",
        {
            "schema_version": "N72R3_STAGE_STATUS_V1",
            "stage": "00_FREEZE_PROTOCOL_AND_INPUTS",
            "status": "PASS_PROTOCOL_FROZEN",
            "created_at_utc": now,
            "protocol": str(protocol_path),
            "protection_manifest": str(protection_path),
            "historical_outputs_read_only": True,
            "interaction_source": "simulated_from_gt_allowed_but_not_real_human",
            "real_human_tape_used": False,
            "retired_gate": "old_n72r2_13_of_13_gate",
            "next_stage": "01_N72R2_AUTHORITY_SEMANTICS_AUDIT",
            "environment": {
                "python": platform.python_version(),
                "platform": platform.platform(),
            },
        },
    )
    print(json.dumps({"status": "PASS_PROTOCOL_FROZEN", "protocol": str(protocol_path)}))


if __name__ == "__main__":
    main()
