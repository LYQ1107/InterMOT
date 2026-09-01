#!/usr/bin/env python3
"""Run the N36 real human-correction transaction loop to sequence end.

The event manifest is an offline artifact.  At runtime this runner receives
only the event's public ID and human box (plus the explicit simulated human
feature crop performed by ``human_evidence``); it never loads future GT.
Candidate observations are streamed from the already validated real tape.
"""

from __future__ import annotations

import argparse
import copy
import gc
import json
import sys
import traceback
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sam3_intermot.association.ccam_replay import _manager_from_prefix
from sam3_intermot.association.human_intervention import (
    HumanFeatureExtractor,
    apply_intervention,
)
from sam3_intermot.association.state_manager import StateManagerConfig

from scripts.n36_real_eval_common import (
    DATA_ROOT,
    FEATURE_DIM,
    atomic_json,
    atomic_jsonl,
    event_source_path,
    iter_rows,
    jsonable,
    load_manifest,
    observations_for_row,
    variant_config,
)


OUT = ROOT / "outputs/n36"
EVENT_MANIFEST = OUT / "real_event_manifest.json"
LEDGER = OUT / "full_loop_event_ledger.jsonl"
RESULT = OUT / "full_loop_results.json"
STAGE = OUT / "stage_03_status.json"
HUMAN_CHECKPOINT = ROOT / "outputs/n9/checkpoints/osnet_x1_0_market1501.pth"
FULL_LOOP_VARIANT = "M3"


def audit_summary(audit: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(audit, dict):
        return {"present": False, "mapping_complete": False, "memory_enabled": False}
    deltas = np.asarray(audit.get("appearance_score_deltas", []), dtype=float)
    return {
        "present": True,
        "mapping_complete": bool(audit.get("candidate_public_id_mapping_complete", False)),
        "candidate_count": len(audit.get("candidates", [])),
        "memory_enabled": bool(audit.get("appearance_memory_enabled", False)),
        "current_frame_memory_delta_max_abs": (
            float(np.max(np.abs(deltas))) if deltas.size else 0.0
        ),
    }


def display_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path.resolve())


def runtime_event_view(event: dict[str, Any]) -> dict[str, Any]:
    """Strip offline GT/feature metadata before invoking the live path."""
    allowed = {
        "event_id",
        "event_type",
        "action_type",
        "frame",
        "public_id",
        "canonical_public_id",
        "current_public_id",
        "other_canonical_public_id",
        "other_auto_tid",
        "gt_box",
        "other_gt_box",
        "quality",
        "mask",
        "other_mask",
        "other_quality",
    }
    return {key: copy.deepcopy(value) for key, value in event.items() if key in allowed}


def run_one(
    item: dict[str, Any],
    extractor: HumanFeatureExtractor,
    variant: str = FULL_LOOP_VARIANT,
) -> dict[str, Any]:
    event = copy.deepcopy(item["event"])
    sequence = str(event["sequence"])
    event_frame = int(event["frame"])
    sequence_end = int(item["sequence_frame_count"]) - 1
    config, description = variant_config(variant)
    # The event prefix is serialized pure Python state.  It was generated from
    # frames < event_frame and contains no GT fields.
    manager = _manager_from_prefix(item["prefix_state"], event_frame, config, FEATURE_DIM)
    source = event_source_path(item)
    runtime_event = runtime_event_view(event)
    event_row = None
    event_observations = None
    first_future_audit = None
    future_frame_count = 0
    future_candidate_count = 0
    duplicate_public_id_frames = []
    mapping_failures = []
    last_state_summary = None
    for _line_no, row in iter_rows(source, event_frame, sequence_end):
        frame = int(row["frame"])
        if row.get("runtime_future_gt_used") is not False or row.get("runtime_gt_read") is not False:
            mapping_failures.append(f"frame_{frame}:runtime_gt_flag_not_false")
        if frame == event_frame:
            event_row = row
            event_observations = observations_for_row(row)
            pre_rows = manager.rollout_frame(frame, event_observations, model=None)
            current_audit = manager.candidate_log[-1] if manager.candidate_log else None
            before_summary = audit_summary(current_audit)
            event_record = apply_intervention(
                manager,
                runtime_event,
                event_frame,
                event_observations,
                pre_rows,
                extractor,
                DATA_ROOT / "train" / sequence / "img1",
            )
            annotation_present = manager.annotate_human_event(event_frame, runtime_event, event_record)
            manager.mark_scope(event_record.get("scope_pids", []), event_frame)
            after_summary = audit_summary(manager.candidate_log[-1] if manager.candidate_log else None)
            action_record = jsonable(event_record)
            human_ledgers = event_record.get("appearance_memory", [])
            memory_write_pass = bool(human_ledgers) and all(
                entry.get("status") == "PASS" for entry in human_ledgers
            )
            touched_ids = {
                int(value)
                for value in (
                    event_record.get("scope_pids", [])
                    if event_record.get("scope_pids")
                    else [event.get("public_id")]
                )
                if value is not None
            }
            # ``scope_pids`` deliberately includes the old/wrong identity for
            # AUTHORITATIVE_REASSIGN.  Only IDs in the transaction's ``adds``
            # list receive the new spatial anchor; requiring an anchor on the
            # protected old ID would be an audit bug.
            anchor_target_ids = {
                int(pid) for pid, _box in event_record.get("adds", []) if pid is not None
            }
            spatial_ready = bool(event_record.get("applied")) and bool(anchor_target_ids) and all(
                int(pid) in manager.states and len(manager.states[int(pid)].anchors) > 0
                for pid in anchor_target_ids
            )
            if len({int(pid) for pid, _box in pre_rows}) != len(pre_rows):
                duplicate_public_id_frames.append(event_frame)
            if not before_summary["present"]:
                mapping_failures.append("event_frame:candidate_audit_missing")
            # Keep the current-frame audit intact long enough to prove that the
            # write did not affect its scores, then retain only the latest
            # record so a long real sequence does not accumulate a giant log.
            if len(manager.candidate_log) > 1:
                manager.candidate_log = [manager.candidate_log[-1]]
            event_checks = {
                "current_spatial_correction_applied": bool(event_record.get("applied")),
                "event_annotation_present": bool(annotation_present),
                "public_native_mapping_audit_present": bool(before_summary["present"]),
                "current_frame_memory_effect_hidden": bool(
                    before_summary["current_frame_memory_delta_max_abs"] <= 1e-8
                ),
                "spatial_correction_before_memory_write": bool(spatial_ready and memory_write_pass),
                "human_feature_write_pass": bool(memory_write_pass),
                "runtime_future_gt_used_false": event.get("future_gt_used_runtime") is False,
            }
            event_start_summary = {
                "pre_rows_count": len(pre_rows),
                "pre_public_ids": [int(pid) for pid, _box in pre_rows],
                "audit": before_summary,
                "event_record": action_record,
                "checks_at_event": event_checks,
            }
            continue
        if event_row is None:
            mapping_failures.append(f"event_frame_{event_frame}_not_seen_before_{frame}")
            continue
        observations = observations_for_row(row)
        rows = manager.rollout_frame(frame, observations, model=None)
        audit = manager.candidate_log[-1] if manager.candidate_log else None
        if first_future_audit is None:
            first_future_audit = audit_summary(audit)
        if len({int(pid) for pid, _box in rows}) != len(rows):
            duplicate_public_id_frames.append(frame)
        if not isinstance(audit, dict) or not audit.get("candidate_public_id_mapping_complete", False):
            if observations:
                mapping_failures.append(f"frame_{frame}:public_mapping_incomplete")
        future_frame_count += 1
        future_candidate_count += len(observations)
        last_state_summary = manager.state_summary()
        if len(manager.candidate_log) > 1:
            manager.candidate_log = [manager.candidate_log[-1]]

    if event_row is None or event_observations is None:
        raise RuntimeError(f"event frame {event_frame} was not found in {source}")
    expected_future_frames = max(0, sequence_end - event_frame)
    event_checks = event_start_summary["checks_at_event"]
    event_checks.update(
        {
            "future_starts_at_event_plus_one": bool(first_future_audit is not None and future_frame_count > 0),
            "future_processed_to_sequence_end": future_frame_count == expected_future_frames,
            "future_memory_audit_enabled": bool(first_future_audit and first_future_audit["memory_enabled"]),
            "no_duplicate_public_ids_per_future_frame": not duplicate_public_id_frames,
            "all_nonempty_future_mappings_complete": not mapping_failures,
        }
    )
    passed = bool(all(event_checks.values()) and future_frame_count == expected_future_frames)
    result = {
        "event_id": event["event_id"],
        "sequence": sequence,
        "event_frame": event_frame,
        "event_type": event["event_type"],
        "public_id": int(event["public_id"]),
        "interaction_source": "simulated_from_gt",
        "synthetic": False,
        "runtime_future_gt_used": False,
        "variant": variant,
        "variant_description": description,
        "status": "PASS" if passed else "FAIL",
        "future_frame_count": future_frame_count,
        "expected_future_frame_count": expected_future_frames,
        "future_candidate_count": future_candidate_count,
        "duplicate_public_id_frames": duplicate_public_id_frames,
        "mapping_failures": mapping_failures,
        "checks": event_checks,
        "event_start": event_start_summary,
        "first_future_audit": first_future_audit,
        "final_state_summary": last_state_summary or manager.state_summary(),
        "event_contract": {
            "runtime_input": ["public_id", "human_box"],
            "future_gt_loaded_by_runner": False,
            "human_feature_source": event.get("human_feature_source"),
        },
    }
    del manager, event_observations, event_row
    gc.collect()
    return result


def run(manifest_path: Path = EVENT_MANIFEST, variant: str = FULL_LOOP_VARIANT) -> dict[str, Any]:
    manifest = load_manifest(manifest_path)
    events = manifest.get("events", [])
    extractor = HumanFeatureExtractor(HUMAN_CHECKPOINT)
    ledger_rows: list[dict[str, Any]] = []
    for item in events:
        try:
            result = run_one(item, extractor, variant=variant)
        except Exception as exc:
            event = item.get("event", {})
            result = {
                "event_id": event.get("event_id"),
                "sequence": event.get("sequence"),
                "event_frame": event.get("frame"),
                "event_type": event.get("event_type"),
                "public_id": event.get("public_id"),
                "synthetic": False,
                "runtime_future_gt_used": False,
                "variant": variant,
                "status": "FAIL",
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(),
            }
        ledger_rows.append(result)
        atomic_jsonl(LEDGER, ledger_rows)
        print(json.dumps({"event_id": result.get("event_id"), "status": result.get("status"), "sequence": result.get("sequence")}, sort_keys=True), flush=True)
    sequence_count = len({row.get("sequence") for row in ledger_rows if row.get("sequence")})
    pass_count = sum(row.get("status") == "PASS" for row in ledger_rows)
    status = "PASS" if pass_count == len(events) and sequence_count >= 6 else ("PARTIAL" if pass_count else "FAIL")
    payload = {
        "protocol": "N36_REAL_FULL_LOOP_TRANSACTION_AUDIT",
        "status": status,
        "real_data_status": status,
        "synthetic": False,
        "split": "train/train_fold",
        "event_count": len(events),
        "event_pass_count": pass_count,
        "independent_sequence_count": sequence_count,
        "variant": variant,
        "runtime_future_gt_used": False,
        "future_gt_used_only_offline_event_generation": True,
        "events": ledger_rows,
        "artifacts": {
            "event_manifest": display_path(manifest_path),
            "event_ledger": display_path(LEDGER),
            "result": display_path(RESULT),
        },
        "aggregate_checks": {
            "at_least_six_independent_sequences": sequence_count >= 6,
            "all_events_pass": pass_count == len(events),
            "all_spatial_corrections_applied": all(row.get("checks", {}).get("current_spatial_correction_applied", False) for row in ledger_rows if row.get("status") == "PASS"),
            "all_memory_writes_after_spatial": all(row.get("checks", {}).get("spatial_correction_before_memory_write", False) for row in ledger_rows if row.get("status") == "PASS"),
            "all_current_frame_writes_hidden": all(row.get("checks", {}).get("current_frame_memory_effect_hidden", False) for row in ledger_rows if row.get("status") == "PASS"),
            "all_sequences_reached_end": all(row.get("checks", {}).get("future_processed_to_sequence_end", False) for row in ledger_rows if row.get("status") == "PASS"),
        },
    }
    atomic_json(RESULT, payload)
    stage = {
        "stage": "N36-05",
        "status": status,
        "real_data_status": status,
        "commands": ["python scripts/run_n36_full_loop.py --manifest outputs/n36/real_event_manifest.json"],
        "artifacts": [display_path(RESULT), display_path(LEDGER)],
        "event_count": len(events),
        "event_pass_count": pass_count,
        "independent_sequence_count": sequence_count,
        "runtime_future_gt_used": False,
        "errors": [row.get("error") for row in ledger_rows if row.get("status") == "FAIL"],
        "next_action": "Run N36 real M0-M4 paired replay only after this full-loop artifact is reviewed; do not train before the future-effect gate.",
    }
    atomic_json(STAGE, stage)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=EVENT_MANIFEST)
    parser.add_argument("--variant", default=FULL_LOOP_VARIANT, choices=("M1", "M2", "M3", "M4"))
    args = parser.parse_args()
    payload = run(args.manifest, args.variant)
    print(json.dumps({"status": payload["status"], "event_count": payload["event_count"], "event_pass_count": payload["event_pass_count"], "output": display_path(RESULT)}, sort_keys=True))


if __name__ == "__main__":
    main()
