#!/usr/bin/env python3
"""Materialize the N72R2 evidence ledger after the handover gate.

This script never overwrites historical N36--N72R1 evidence and never fills
unrun research metrics.  It records the exact same-run smoke separately from
the unresolved cross-session candidate-recall gate.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path("/data2/usr_for_deadline/SAM3_InterMOT_N72R2")
WORKTREE = ROOT / "worktree"
OUT = ROOT / "outputs/N72R2"
WORKTREE_OUT = WORKTREE / "outputs/N72R2"


def atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
        directory_fd = os.open(str(path.parent), os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def maybe_sha256(path: Path) -> dict[str, Any]:
    return {
        "path": str(path),
        "exists": path.is_file(),
        "sha256": sha256(path) if path.is_file() else None,
    }


def now_utc() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def main() -> None:
    exact_path = OUT / "bridge/stage_01_exact_public_assignment_attempt2/done.json"
    exact = read_json(exact_path)
    exact_failure_path = OUT / "bridge/stage_01_exact_public_assignment_attempt1/failure.json"
    initial_handover_path = OUT / "handover/overlap_audit/handover_gate.json"
    seed_recovery_path = OUT / "handover/overlap_audit_seed_recovery_attempt3/handover_gate.json"
    bulk_done_path = OUT / "handover/second_window_0416_0575_bulk_rebind_attempt1/done.json"
    bulk_gate_path = OUT / "handover/overlap_audit_bulk_rebind_attempt1/handover_gate.json"
    bulk_done = read_json(bulk_done_path)
    bulk_gate = read_json(bulk_gate_path)

    stage_01_exact = {
        "schema_version": "N72R2_STAGE_STATUS_V1",
        "stage": "01_PUBLIC_AUTHORITY_BRIDGE",
        "status": "PASS_SAME_RUN_EXACT_PUBLIC_ASSIGNMENT",
        "artifact": str(exact_path),
        "sequence": exact.get("sequence"),
        "window_id": exact.get("window_id"),
        "frame_count": exact.get("frame_count"),
        "candidate_row_count": exact.get("candidate_row_count"),
        "same_run_mapping_coverage": exact.get("public_authority_audit", {}).get("mapping_coverage"),
        "public_authority_source": exact.get("public_authority_audit", {}).get("binding_source"),
        "association_state_id_is_public_id": exact.get("public_authority_audit", {}).get("association_state_ids_are_public"),
        "exact_public_assignment_pass_frame_count": exact.get("exact_public_assignment_pass_frame_count"),
        "exact_public_assignment_failure_frame_count": exact.get("exact_public_assignment_failure_count") or 0,
        "none_column_solver_enabled": True,
        "runtime_future_gt_used": False,
        "research_efficacy": "NOT_RUN",
        "historical_exact_smoke_failure": str(exact_failure_path),
        "created_at_utc": now_utc(),
    }
    atomic_json(OUT / "stage_01_status_exact_public_attempt2.json", stage_01_exact)

    unresolved = []
    for item in bulk_done.get("past_state_rebind_audit", {}).get("failures", []):
        unresolved.append(
            {
                "object_id": item.get("object_id"),
                "public_id": item.get("public_id") if item.get("public_id") is not None else int(item["object_id"]) - 100000,
                "reason": item.get("reason"),
            }
        )
    stage_02_final = {
        "schema_version": "N72R2_STAGE_STATUS_V1",
        "stage": "02_MULTI_WINDOW_SEGMENT_HANDOVER",
        "status": "BLOCKED_PUBLIC_MAPPING",
        "historical_status_artifact": str(OUT / "stage_02_status.json"),
        "gate_1_single_window": stage_01_exact,
        "gate_2_overlap": {
            "status": "FAIL_CANDIDATE_RECALL",
            "initial": str(initial_handover_path),
            "initial_overlap_mapping_coverage": read_json(initial_handover_path).get("overlap_mapping_coverage"),
            "per_object_recovery": str(seed_recovery_path),
            "bulk_rebind_done": str(bulk_done_path),
            "bulk_rebind_gate": str(bulk_gate_path),
            "bulk_requested": bulk_done.get("past_state_rebind_audit", {}).get("requested_count"),
            "bulk_first_prompt_observed": bulk_done.get("past_state_rebind_audit", {}).get("attempts", [{}])[0].get("observed_count"),
            "bulk_sanitized_prompt_observed": bulk_done.get("past_state_rebind_audit", {}).get("attempts", [{}, {}])[-1].get("observed_count"),
            "bulk_recovered": bulk_done.get("past_state_rebind_audit", {}).get("recovered_count"),
            "bulk_unresolved": unresolved,
            "bulk_overlap_mapping_coverage": bulk_gate.get("overlap_mapping_coverage"),
            "bulk_transactions": bulk_gate.get("transaction_count"),
            "bulk_overlap_previous_tracks": bulk_gate.get("overlap_previous_track_count"),
            "bulk_overlap_next_tracks": bulk_gate.get("overlap_next_track_count"),
            "raw_id_equality_used_for_match": bulk_gate.get("raw_id_equality_used_for_match"),
            "runtime_future_gt_used": bulk_gate.get("runtime_future_gt_used"),
        },
        "gate_6_fixed_windows": {
            "status": "NOT_RUN_BLOCKED_AT_OVERLAP_GATE",
            "required_count": 6,
            "completed_count": 1,
        },
        "candidate_recall": {
            "prior_overlap_tracks": bulk_gate.get("overlap_previous_track_count"),
            "bulk_rebind_requested": bulk_done.get("past_state_rebind_audit", {}).get("requested_count"),
            "bulk_rebind_recovered": bulk_done.get("past_state_rebind_audit", {}).get("recovered_count"),
            "bulk_rebind_failed": bulk_done.get("past_state_rebind_audit", {}).get("failure_count"),
            "bulk_mapped_overlap_tracks": bulk_gate.get("mapped_previous_track_count"),
            "bulk_missing_overlap_tracks": (
                int(bulk_gate.get("overlap_previous_track_count", 0))
                - int(bulk_gate.get("mapped_previous_track_count", 0))
            ),
            "same_frame_duplicate_box_ambiguity_present": True,
            "past_state_source_is_gt": False,
        },
        "downstream_authorized": False,
        "research_efficacy": "NOT_RUN",
        "real_human_event_count": 0,
        "interaction_source": "simulated_from_gt_not_started",
        "runtime_future_gt_used": False,
        "created_at_utc": now_utc(),
    }
    atomic_json(OUT / "stage_02_status_final.json", stage_02_final)

    common = {
        "schema_version": "N72R2_STAGE_STATUS_V1",
        "prerequisite_blocker": "STAGE_02_PUBLIC_MAPPING_AND_CANDIDATE_RECALL",
        "real_human_event_count": 0,
        "interaction_source": "simulated_from_gt_not_started",
        "runtime_future_gt_used": False,
        "research_efficacy": "NOT_RUN",
        "downstream_authorized": False,
        "created_at_utc": now_utc(),
    }
    stage_specs = {
        3: {
            "stage": "03_CAUSAL_SIMULATED_HUMAN_OBSERVER",
            "status": "BLOCKED_PUBLIC_MAPPING",
            "local_implementation": "READY_LOCAL_ONLY",
            "official_runtime": "NOT_RUN_PREREQUISITE",
            "toy_contract_tests": 8,
        },
        4: {"stage": "04_SIX_ACTION_RUNTIME_TRANSACTIONS", "status": "NOT_RUN_BLOCKED_PREREQUISITE"},
        5: {"stage": "05_OFFICIAL_CURRENT_FRAME_SPATIAL_CORRECTION", "status": "NOT_RUN_BLOCKED_PREREQUISITE", "correction_accuracy": None},
        6: {"stage": "06_PUBLIC_ID_APPEARANCE_MEMORY", "status": "NOT_RUN_BLOCKED_PREREQUISITE", "untouched_bitwise_audit": None},
        7: {"stage": "07_EXACT_PUBLIC_BASELINE", "status": "NOT_RUN_BLOCKED_PREREQUISITE", "single_window_solver_smoke": str(exact_path)},
        8: {"stage": "08_M0_M4_NO_WRITE_VARIANTS", "status": "NOT_RUN_BLOCKED_PREREQUISITE"},
        9: {"stage": "09_TARGET_SCOPED_ASSOCIATION", "status": "NOT_RUN_BLOCKED_PREREQUISITE"},
        10: {"stage": "10_STRICT_FUTURE_EFFECT_GATE", "status": "NOT_RUN_BLOCKED_PREREQUISITE", "h20": None, "h50": None, "h100": None, "cluster_bootstrap": None},
    }
    stage_paths: dict[str, str] = {}
    for number, spec in stage_specs.items():
        payload = dict(common)
        payload.update(spec)
        path = OUT / f"stage_{number:02d}_status.json"
        atomic_json(path, payload)
        stage_paths[f"stage_{number:02d}"] = str(path)

    modified_code = [
        WORKTREE / "sam3_intermot/backend/sam3_backend.py",
        WORKTREE / "sam3_intermot/association/public_assignment.py",
        WORKTREE / "sam3_intermot/identity/public_authority.py",
        WORKTREE / "sam3_intermot/identity/handover.py",
        WORKTREE / "sam3_intermot/identity/runtime_authority.py",
        WORKTREE / "sam3_intermot/interaction/n72r2_simulated_observer.py",
        WORKTREE / "scripts/n72r2_stage00_freeze.py",
        WORKTREE / "scripts/n72r2_stage01_authority_smoke.py",
        WORKTREE / "scripts/n72r2_stage02_handover.py",
        WORKTREE / "scripts/n72r2_stage02_status.py",
        WORKTREE / "tests/test_n72r2_public_closure.py",
    ]
    code_hashes = {str(path.relative_to(WORKTREE)): maybe_sha256(path) for path in modified_code}

    final_gate = {
        "schema_version": "N72R2_FINAL_GATE_V1",
        "final_status": "BLOCKED_CANDIDATE_RECALL",
        "best_round": "ROUND_0_BASELINE",
            "protocol": str(WORKTREE_OUT / "protocol.json"),
        "stage_statuses": {
            "stage_00": str(OUT / "stage_00_status.json"),
            "stage_01": str(OUT / "stage_01_status_exact_public_attempt2.json"),
            "stage_02": str(OUT / "stage_02_status_final.json"),
            **stage_paths,
        },
        "public_mapping": {
            "same_run_coverage": exact.get("public_authority_audit", {}).get("mapping_coverage"),
            "same_run_exact_solver_frames": exact.get("exact_public_assignment_pass_frame_count"),
            "cross_window_overlap_coverage": bulk_gate.get("overlap_mapping_coverage"),
            "cross_window_handover_pass": False,
            "handover_required_fixed_window_count": 6,
            "handover_completed_fixed_window_count": 1,
            "association_state_id_used_as_public_id": False,
            "raw_id_equality_used": False,
        },
        "candidate_recall": {
            "prior_overlap_tracks": bulk_gate.get("overlap_previous_track_count"),
            "bulk_rebind_requested": bulk_done.get("past_state_rebind_audit", {}).get("requested_count"),
            "bulk_rebind_recovered": bulk_done.get("past_state_rebind_audit", {}).get("recovered_count"),
            "bulk_mapped_overlap_tracks": bulk_gate.get("mapped_previous_track_count"),
            "unresolved_public_ids": [item.get("public_id") for item in unresolved],
            "root_cause": "official_box_only_multi_object_rebind_cannot uniquely recover all persisted objects; same-frame duplicate box authority ambiguity remains",
        },
        "simulated_human": {
            "events": 0,
            "independent_sequences": 0,
            "actions": {action: 0 for action in read_json(WORKTREE_OUT / "protocol.json").get("actions", [])},
            "interaction_source": "simulated_from_gt_not_started",
            "real_human_tape_used": False,
            "runtime_future_gt_used": False,
        },
        "current_frame_correction": {"status": "NOT_RUN_BLOCKED_PREREQUISITE", "accuracy": None},
        "future": {
            "status": "NOT_RUN_BLOCKED_PREREQUISITE",
            "H20": None,
            "H50": None,
            "H100": None,
            "id_switch": None,
            "missing": None,
            "wrong_reassociation": None,
            "re_correction": None,
        },
        "assignment": {"changes": None, "correct": None, "incorrect": None, "neutral": None},
        "safety": {"untouched_regression": None, "protected_identity_regression": None},
        "statistics": {"sequence_cluster_bootstrap": None, "ci95": None, "repetitions": 2000},
        "runtime_audit": {
            "gt_read_before_prediction": None,
            "gt_read_future": 0,
            "gt_used_for_model_decision": 0,
            "gt_used_for_scheduler": 0,
            "event_frame_memory_read": None,
            "t1_first_memory_read": None,
        },
        "root_cause_history": [
            {
                "round": "ROUND_0_BASELINE",
                "finding": "same-run TrackManager authority and exact public+NONE solver pass on one fixed window",
                "gate": "PASS_LOCAL_WINDOW_ONLY",
            },
            {
                "round": "STAGE_02_RECOVERY_ATTEMPTS_1_TO_3",
                "finding": "per-object past-state seed did not recover the prior candidate set",
                "gate": "FAIL_CANDIDATE_RECALL",
            },
            {
                "round": "STAGE_02_BULK_REBIND_TARGETED_SMOKE",
                "finding": "official multi-box prompt improved recovery but remained incomplete and ambiguous",
                "gate": "FAIL_CANDIDATE_RECALL",
            },
        ],
        "failed_branches": [
            "cross_session_candidate_recall",
            "multi_window_public_lineage_handover",
            "full_loop",
            "M0_M4_future_replay",
            "strict_future_effect_gate",
            "training_or_production_authorization",
        ],
        "authorization": {
            "full_loop": False,
            "future_replay": False,
            "calibration": False,
            "selector": False,
            "decoder_lora": False,
            "production": False,
        },
        "confirmation": {"status": "NOT_RUN_BLOCKED_PREREQUISITE"},
        "input_and_code_hashes": {
            "protocol": maybe_sha256(WORKTREE_OUT / "protocol.json"),
            "protection_manifest": maybe_sha256(WORKTREE_OUT / "protection_manifest.json"),
            "exact_public_smoke": maybe_sha256(exact_path),
            "bulk_rebind_smoke": maybe_sha256(bulk_done_path),
            "bulk_rebind_gate": maybe_sha256(bulk_gate_path),
            "modified_code": code_hashes,
        },
        "files": {
            "final_gate": str(OUT / "n72r2_final_gate.json"),
            "final_report": str(WORKTREE / "docs/N72R2_FINAL_REPORT.md"),
            "stage_01_exact": str(OUT / "stage_01_status_exact_public_attempt2.json"),
            "stage_02_final": str(OUT / "stage_02_status_final.json"),
            "bulk_rebind_done": str(bulk_done_path),
            "bulk_rebind_gate": str(bulk_gate_path),
        },
        "next_decision": "Provide or implement a proven official multi-object/session rebind that preserves every persisted candidate authority; do not infer unresolved public identities from raw IDs, geometry, or future GT. After a candidate-complete handover, rerun only the frozen 1→2→6 gate before any simulated event or effect replay.",
        "created_at_utc": now_utc(),
    }
    atomic_json(OUT / "n72r2_final_gate.json", final_gate)
    print(json.dumps({
        "final_status": final_gate["final_status"],
        "same_run_mapping": final_gate["public_mapping"]["same_run_coverage"],
        "cross_window_mapping": final_gate["public_mapping"]["cross_window_overlap_coverage"],
        "bulk_rebind_requested": final_gate["candidate_recall"]["bulk_rebind_requested"],
        "bulk_rebind_recovered": final_gate["candidate_recall"]["bulk_rebind_recovered"],
        "bulk_mapped_overlap_tracks": final_gate["candidate_recall"]["bulk_mapped_overlap_tracks"],
        "events": 0,
        "downstream_authorized": False,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
