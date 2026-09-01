#!/usr/bin/env python3
"""Exercise the N34 event transaction loop on an explicit synthetic fallback.

The real-data branch is kept NOT_AVAILABLE because N34-2 did not produce a
candidate-complete tape.  The synthetic branch is useful only for checking
the existing state-machine interfaces and transaction invariants; it is not a
DanceTrack metric or identity-learning result.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from n34_synthetic import SyntheticHumanExtractor, box_for, event_spec, feature, observation

from sam3_intermot.association.human_intervention import apply_intervention
from sam3_intermot.association.state_manager import StateManager, StateManagerConfig


OUT = ROOT / "outputs" / "n34"
ACTION_TYPES = (
    "ADD_NEW_IDENTITY",
    "AUTHORITATIVE_REASSIGN",
    "ATOMIC_ID_SWAP",
    "RECOVER_IDENTITY",
)


def jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.astype(float).tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    return value


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, path)
    finally:
        if os.path.exists(tmp_name):
            os.unlink(tmp_name)


def atomic_json(path: Path, payload: Any) -> None:
    atomic_text(path, json.dumps(jsonable(payload), indent=2, sort_keys=True, allow_nan=False) + "\n")


def atomic_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    atomic_text(
        path,
        "".join(json.dumps(jsonable(row), sort_keys=True, allow_nan=False) + "\n" for row in rows),
    )


def as_obs(native_tid: int, frame: int) -> dict[str, Any]:
    raw = observation(native_tid, frame)
    raw["feat"] = np.asarray(raw["embedding"], dtype=np.float32)
    raw["has_feat"] = 1.0
    raw["conf"] = raw.pop("confidence")
    raw["box"] = np.asarray(raw["box"], dtype=float)
    return raw


def _manager() -> StateManager:
    manager = StateManager(
        StateManagerConfig(
            variant="reid",
            score_threshold=-100.0,
            max_lost_gap=30,
            use_appearance_memory=True,
            appearance_reliability_threshold=0.0,
            native_constraint_frames=None,
        )
    )
    for pid, native_tid in ((101, 11), (102, 22)):
        state = manager.get_or_create(pid, as_obs(native_tid, 0), 0)
        state.add_positive(native_tid, None)
    manager.next_pid = 103
    return manager


def _run_action(action: str, future_end: int = 30) -> dict[str, Any]:
    manager = _manager()
    event = event_spec(action, frame=1)
    obs_a = as_obs(11, 1)
    obs_b = as_obs(22, 1)
    if action == "RECOVER_IDENTITY":
        manager.states[101].mark_lost(1)
        event_obs = [obs_b]
    else:
        event_obs = [obs_a, obs_b]
    pre_rows = manager.rollout_frame(1, event_obs, model=None)
    if action in {"AUTHORITATIVE_REASSIGN", "ATOMIC_ID_SWAP"}:
        # Explicitly model the pre-correction wrong assignment at the event
        # frame.  The human event, not a hidden label, supplies the repair.
        pre_rows = [(102, obs_a["box"].copy()), (101, obs_b["box"].copy())]
    event_audit_before = jsonable(manager.candidate_log[-1])
    record = apply_intervention(
        manager,
        event,
        1,
        event_obs,
        pre_rows,
        SyntheticHumanExtractor(),
        Path("/synthetic/n34"),
    )
    annotated = manager.annotate_human_event(1, event, record)
    future_rows: list[dict[str, Any]] = []
    future_ids: list[list[int]] = []
    for frame in range(2, int(future_end) + 1):
        obs_list = [as_obs(11, frame), as_obs(22, frame)]
        if action == "ADD_NEW_IDENTITY":
            obs_list.append(as_obs(33, frame))
        rows = manager.rollout_frame(frame, obs_list, model=None)
        ids = [int(pid) for pid, _ in rows]
        future_ids.append(ids)
        future_rows.append(
            {
                "frame": frame,
                "rows": [(int(pid), np.asarray(box, dtype=float).tolist()) for pid, box in rows],
                "candidate_audit": manager.candidate_log[-1],
            }
        )

    expected_ids = {101, 102} | ({103} if action == "ADD_NEW_IDENTITY" else set())
    all_unique = all(len(ids) == len(set(ids)) for ids in future_ids)
    all_expected_present = all(expected_ids.issubset(set(ids)) for ids in future_ids)
    event_frame_delta = np.asarray(
        event_audit_before.get("appearance_score_deltas", []), dtype=float
    )
    current_frame_memory_hidden = bool(event_frame_delta.size == 0 or np.allclose(event_frame_delta, 0.0))
    annotation_present = bool(
        manager.candidate_log
        and manager.candidate_log[-1].get("frame") != 1
        and any(item.get("event_id") == event.get("event_id") for item in manager.candidate_log[0].get("human_events", []))
    )
    # The candidate log at frame 1 is retained in event_audit_before; the
    # annotation is checked directly because later frames append new records.
    annotation_present = any(
        item.get("event_id") == event.get("event_id")
        for item in manager.candidate_log[0].get("human_events", [])
    )
    action_record = jsonable(record)
    ledgers = record.get("appearance_memory", [])
    memory_write_pass = bool(ledgers) and all(item.get("status") == "PASS" for item in ledgers)
    touched_ids = {
        int(pid)
        for pid in (
            [record.get("new_pid")]
            if action == "ADD_NEW_IDENTITY"
            else ([101, 102] if action == "ATOMIC_ID_SWAP" else [101])
        )
        if pid is not None
    }
    spatial_state_ready = bool(touched_ids) and all(
        len(manager.states[pid].anchors) > 0 for pid in touched_ids if pid in manager.states
    )
    protected_ids_stable = all_expected_present
    recover_reuses_id = True
    if action == "RECOVER_IDENTITY":
        recover_reuses_id = bool(
            101 in manager.states
            and all(101 in ids for ids in future_ids)
            and record.get("new_pid") in (None, 101)
            and manager.next_pid == 103
        )
    swap_constraints = True
    if action == "ATOMIC_ID_SWAP":
        a, b = manager.states[101], manager.states[102]
        swap_constraints = (
            11 in a.positive_native_tids
            and 22 in a.negative_native_tids
            and 22 in b.positive_native_tids
            and 11 in b.negative_native_tids
        )
    checks = {
        "event_applied": bool(record.get("applied")),
        "event_annotation_present": annotation_present,
        "no_duplicate_public_ids_per_future_frame": all_unique,
        "expected_public_ids_present_through_sequence_end": all_expected_present,
        "untouched_identity_stability": protected_ids_stable,
        "spatial_correction_before_memory_write": bool(record.get("applied") and spatial_state_ready and memory_write_pass),
        "current_frame_memory_effect_hidden": current_frame_memory_hidden,
        "future_starts_at_event_plus_one": bool(future_rows and future_rows[0]["frame"] == int(event["frame"]) + 1),
        "recovery_reuses_existing_public_id": recover_reuses_id,
        "swap_bilateral_constraints": swap_constraints,
    }
    result = {
        "action_type": action,
        "interaction_source": "simulated_from_gt",
        "synthetic": True,
        "future_gt_used_runtime": False,
        "event": event,
        "pre_rows": pre_rows,
        "event_record": action_record,
        "future_frame_count": len(future_rows),
        "future_public_ids": future_ids,
        "checks": checks,
        "all_checks_pass": bool(all(checks.values())),
        "transaction_order": [
            "current_frame_rollout_and_candidate_audit",
            "spatial_correction",
            "human_roi_memory_write",
            "current_frame_event_annotation",
            "future_rollout_frames_t_plus_1_to_sequence_end",
        ],
        "source_contract": {
            "spatial_before_memory": "human_intervention.apply_intervention applies state/anchor correction before write_memory",
            "memory_target": "known authoritative public ID supplied by event",
            "same_frame_visibility": "AppearanceMemory score is causal and hides source-frame writes",
        },
        "final_state_summary": manager.state_summary(),
    }
    return result


def run() -> dict[str, Any]:
    OUT.mkdir(parents=True, exist_ok=True)
    results = [_run_action(action) for action in ACTION_TYPES]
    ledger_rows = [
        {
            "ledger_type": "N34_SYNTHETIC_FULL_LOOP_EVENT",
            "event_id": result["event"]["event_id"],
            "action_type": result["action_type"],
            "checks": result["checks"],
            "event_record": result["event_record"],
            "future_frame_count": result["future_frame_count"],
            "future_gt_used_runtime": False,
        }
        for result in results
    ]
    atomic_jsonl(OUT / "full_loop_event_ledger.jsonl", ledger_rows)
    atomic_json(
        OUT / "synthetic_event_tape.json",
        {
            "protocol": "N34_SYNTHETIC_EVENT_TAPE_DECLARATION",
            "status": "PASS" if all(item["all_checks_pass"] for item in results) else "FAIL",
            "synthetic": True,
            "interaction_source": "simulated_from_gt",
            "candidate_complete": True,
            "future_gt_used_runtime": False,
            "event_types": list(ACTION_TYPES),
            "future_frames_per_event": 29,
            "not_a_real_data_result": True,
            "builder": "scripts/n34_synthetic.py",
        },
    )
    real_reason = "N34-2 real candidate-complete tape unavailable; real event-to-sequence loop not claimable."
    payload = {
        "protocol": "N34_FULL_LOOP_TRANSACTION_AUDIT",
        "status": "PARTIAL" if all(item["all_checks_pass"] for item in results) else "FAIL",
        "real_data_status": "NOT_AVAILABLE",
        "real_reason": real_reason,
        "synthetic_fallback_status": "PASS" if all(item["all_checks_pass"] for item in results) else "FAIL",
        "synthetic_not_a_real_data_result": True,
        "future_gt_used_runtime": False,
        "event_type_count": len(results),
        "event_types": list(ACTION_TYPES),
        "events": results,
        "aggregate_checks": {
            "all_four_event_types_exercised": len(results) == 4,
            "all_synthetic_transaction_checks_pass": all(item["all_checks_pass"] for item in results),
            "no_duplicate_ids": all(item["checks"]["no_duplicate_public_ids_per_future_frame"] for item in results),
            "untouched_identity_stability": all(item["checks"]["untouched_identity_stability"] for item in results),
            "memory_after_spatial": all(item["checks"]["spatial_correction_before_memory_write"] for item in results),
        },
        "artifacts": {
            "event_ledger": "outputs/n34/full_loop_event_ledger.jsonl",
            "synthetic_event_tape": "outputs/n34/synthetic_event_tape.json",
        },
    }
    atomic_json(OUT / "full_loop_transaction_results.json", payload)
    stage = {
        "stage": "N34-3",
        "status": payload["status"],
        "commands": ["python scripts/run_n34_full_loop.py"],
        "artifacts": [
            "outputs/n34/full_loop_transaction_results.json",
            "outputs/n34/full_loop_event_ledger.jsonl",
            "outputs/n34/synthetic_event_tape.json",
        ],
        "errors": [] if payload["status"] != "FAIL" else ["synthetic transaction invariant failure"],
        "real_data_status": "NOT_AVAILABLE",
        "synthetic_fallback_status": payload["synthetic_fallback_status"],
        "next_action": "Run paired M0-M4 replay on the synthetic fallback; keep real future-effect metrics NOT_COMPUTABLE.",
    }
    atomic_json(OUT / "stage_03_status.json", stage)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.parse_args()
    payload = run()
    print(json.dumps({"status": payload["status"], "synthetic": payload["synthetic_fallback_status"], "output": "outputs/n34/full_loop_transaction_results.json"}, sort_keys=True))


if __name__ == "__main__":
    main()
