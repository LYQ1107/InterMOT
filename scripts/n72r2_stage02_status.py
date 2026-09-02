#!/usr/bin/env python3
"""Write the auditable N72R2 Stage 02 handover decision."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any


ROOT = Path("/data2/usr_for_deadline/SAM3_InterMOT_N72R2")


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


def main() -> None:
    out_root = ROOT / "outputs/N72R2"
    bridge_done = read_json(out_root / "bridge/stage_01_smoke_attempt4/done.json")
    partial = read_json(out_root / "handover/overlap_audit/handover_gate.json")
    recovered_done = read_json(
        out_root / "handover/second_window_0416_0575_seed_recovery_attempt3/done.json"
    )
    recovered_partial = read_json(
        out_root / "handover/overlap_audit_seed_recovery_attempt3/handover_gate.json"
    )
    status = {
        "schema_version": "N72R2_STAGE_STATUS_V1",
        "stage": "02_MULTI_WINDOW_SEGMENT_HANDOVER",
        "status": "BLOCKED_PUBLIC_MAPPING",
        "gate_1_single_window": {
            "status": "PASS",
            "artifact": str(out_root / "bridge/stage_01_smoke_attempt4/done.json"),
            "candidate_rows": bridge_done.get("candidate_row_count"),
            "public_mapping_status": bridge_done.get("public_mapping_status"),
            "mapping_coverage": bridge_done.get("public_authority_audit", {}).get("mapping_coverage"),
            "runtime_future_gt_used": bridge_done.get("runtime_future_gt_used"),
        },
        "gate_2_overlap": {
            "status": "FAIL_CANDIDATE_RECALL",
            "initial_artifact": str(out_root / "handover/overlap_audit/handover_gate.json"),
            "initial": {
                "overlap_mapping_coverage": partial.get("overlap_mapping_coverage"),
                "previous_tracks": partial.get("overlap_previous_track_count"),
                "next_tracks": partial.get("overlap_next_track_count"),
                "transactions": partial.get("transaction_count"),
            },
            "recovery_attempts": [
                {
                    "attempt": 1,
                    "status": "FAIL_OFFICIAL_SAM3_SEED_NOT_RETURNED",
                    "artifact": str(out_root / "handover/second_window_0416_0575_seed_recovery_attempt1/failure.json"),
                    "root_cause": "Official SAM3 returned no observation for the first past-state seed; strict runner preserved the traceback.",
                },
                {
                    "attempt": 2,
                    "status": recovered_done.get("status"),
                    "artifact": str(out_root / "handover/second_window_0416_0575_seed_recovery_attempt2/done.json"),
                    "candidate_rows": recovered_done.get("candidate_row_count"),
                    "seed_requested": recovered_done.get("past_state_seed_count"),
                    "seed_recovered": recovered_done.get("past_state_seed_recovered_count"),
                    "seed_failed": recovered_done.get("past_state_seed_failure_count"),
                    "root_cause": "Failed seeds were removed without re-establishing the concept cache; this produced an honest zero-row partial artifact.",
                },
                {
                    "attempt": 3,
                    "status": recovered_done.get("status"),
                    "artifact": str(out_root / "handover/second_window_0416_0575_seed_recovery_attempt3/done.json"),
                    "candidate_rows": recovered_done.get("candidate_row_count"),
                    "seed_requested": recovered_done.get("past_state_seed_count"),
                    "seed_recovered": recovered_done.get("past_state_seed_recovered_count"),
                    "seed_failed": recovered_done.get("past_state_seed_failure_count"),
                    "root_cause": "Official box-only reinitialization could not distinguish the prior session's seed boxes; the concept fallback retained only the original seven candidates.",
                },
            ],
            "recovered_artifact": str(out_root / "handover/overlap_audit_seed_recovery_attempt3/handover_gate.json"),
            "recovered": {
                "overlap_mapping_coverage": recovered_partial.get("overlap_mapping_coverage"),
                "previous_tracks": recovered_partial.get("overlap_previous_track_count"),
                "next_tracks": recovered_partial.get("overlap_next_track_count"),
                "transactions": recovered_partial.get("transaction_count"),
            },
            "raw_id_equality_used_for_match": False,
            "runtime_future_gt_used": False,
        },
        "gate_6_fixed_windows": {
            "status": "NOT_RUN_BLOCKED_AT_OVERLAP_GATE",
            "required_count": 6,
            "completed_count": 1,
            "note": "The existing N71 six-window plan remains read-only. Further independent exports cannot repair an unresolved exact overlap authority join and are not used to claim handover completion.",
        },
        "candidate_recall": {
            "prior_overlap_tracks": 13,
            "recovered_overlap_tracks": 7,
            "missing_track_count": 6,
            "repeated_same_box_authority_ambiguity": True,
            "past_state_seed_source_is_gt": False,
        },
        "downstream_authorized": False,
        "research_efficacy": "NOT_RUN",
        "real_human_event_count": 0,
        "interaction_source": "simulated_from_gt_not_started",
        "next_minimum_action": "Implement and unit-test an explicit past-state/session rebind contract or provide a proven official multi-object reinitialization primitive; do not infer missing public IDs from raw SAM IDs or geometry alone.",
        "runtime_future_gt_used": False,
    }
    atomic_json(out_root / "stage_02_status.json", status)
    print(json.dumps({"status": status["status"], "overlap_coverage": recovered_partial.get("overlap_mapping_coverage"), "missing_tracks": status["candidate_recall"]["missing_track_count"]}, sort_keys=True))


if __name__ == "__main__":
    main()
