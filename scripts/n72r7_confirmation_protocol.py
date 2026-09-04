#!/usr/bin/env python3
"""Freeze the sequence-disjoint N72R7 confirmation inputs.

The two confirmation events were reserved before N72R7 development replay,
but their old B1 branches did not receive an authoritative public ID.  This
module completes that *event input* explicitly without deriving an ID from GT,
raw SAM IDs, or replay outcomes.  Runtime workers consume only this sealed
protocol; GT is loaded later by the posthoc scorer.
"""

from __future__ import annotations

from datetime import datetime, timezone
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
import traceback
from typing import Any, Mapping

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
EVENT_POLICY = ROOT / "outputs/N72R5/mechanism_rounds/round_06_event_policy/real_event_manifest.json"
STAGE08 = ROOT / "outputs/N72R5R1/controller/round_05_branch_isolation_v0/full/stage08_runtime_manifest.json"
OUTPUT = ROOT / "outputs/N72R7/confirmation/confirmation_protocol.json"
EXPECTED = (
    "n72r5-pool-n37-dancetrack0020-0035-add_new_identity-032",
    "n72r5-pool-n37-dancetrack0049-0008-atomic_id_swap-001",
)


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object: {path}")
    return value


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise TypeError(f"{path}:{line_number}: expected object")
            rows.append(value)
    return rows


def resolve(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(json.dumps(dict(payload), ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n")
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


def _explicit_pairs(row: Mapping[str, Any]) -> list[tuple[int, int]]:
    state_axis = [int(value) for value in row.get("association_state_axis", [])]
    by_state: dict[int, int] = {}
    # The public_id_axis also contains explicit NONE columns for candidates
    # that do not have a live association state.  The solver's assigned rows
    # are the authoritative event-frame state->public mapping; do not infer a
    # state from every candidate/identity row.
    for item in row.get("solver", {}).get("assignment_rows", []):
        if str(item.get("status")) != "ASSIGNED_TO_PUBLIC_ID":
            continue
        state = item.get("association_state_id")
        public = item.get("public_id")
        if state is not None and public is not None and int(state) in state_axis:
            by_state[int(state)] = int(public)
    if set(state_axis) != set(by_state):
        # Compatibility fallback for older frozen rows whose solver audit is
        # absent, still restricted to the explicitly live state axis.
        by_state = {}
        for item in row.get("identity_rows", []):
            state = item.get("association_state_id")
            public = item.get("public_id")
            if state is not None and public is not None and int(state) in state_axis:
                by_state[int(state)] = int(public)
    if set(state_axis) != set(by_state):
        raise RuntimeError(
            "B0 event row lacks a complete explicit state-to-public axis: "
            f"state_axis={len(state_axis)} mapped={len(by_state)} "
            f"public_axis={len(row.get('public_id_axis', []))}"
        )
    pairs = [(state, by_state[state]) for state in state_axis]
    if len({public for _, public in pairs}) != len(pairs):
        raise RuntimeError("B0 event row has duplicate public authority")
    return pairs


def build_protocol() -> dict[str, Any]:
    policy = read_json(EVENT_POLICY)
    stage08 = read_json(STAGE08)
    policy_by_id = {str(item["event_id"]): dict(item) for item in policy.get("events", [])}
    stage_by_id = {str(item["event_id"]): dict(item) for item in stage08.get("events", [])}
    if set(EXPECTED) - set(policy_by_id) or set(EXPECTED) - set(stage_by_id):
        raise RuntimeError("confirmation events are absent from frozen N72R5 inputs")

    specs: list[dict[str, Any]] = []
    for event_id in EXPECTED:
        event = policy_by_id[event_id]
        stage_event = stage_by_id[event_id]
        action = str(event["action_type"])
        event_frame = int(event["event_frame"])
        future_window = [int(event_frame + 1), int(event_frame + 100)]
        branches = {str(item.get("branch")): dict(item) for item in stage_event.get("branches", [])}
        main = branches.get("B0_NO_INTERVENTION")
        if main is None:
            raise RuntimeError(f"missing frozen B0 branch: {event_id}")
        main_path = resolve(str(main["output"]))
        if not main_path.is_file():
            raise FileNotFoundError(main_path)
        rows = read_jsonl(main_path)
        event_rows = [row for row in rows if int(row.get("frame", -1)) == event_frame]
        if len(event_rows) != 1:
            raise RuntimeError(f"B0 event-frame row count is not one: {event_id}")
        pairs = _explicit_pairs(event_rows[0])
        public_axis = [public for _, public in pairs]
        state_axis = [state for state, _ in pairs]
        if action == "ADD_NEW_IDENTITY":
            if event_id != EXPECTED[0]:
                raise RuntimeError(f"unexpected ADD confirmation event: {event_id}")
            target_public = max(public_axis) + 1
            target_state = max(state_axis) + 1
            other_public = None
            authority = "explicit_simulated_human_add_allocator_value_after_frozen_prefix_axis"
            if target_public in public_axis or target_state in state_axis:
                raise RuntimeError("ADD confirmation authority collides with frozen prefix axis")
        elif action == "ATOMIC_ID_SWAP":
            if event_id != EXPECTED[1]:
                raise RuntimeError(f"unexpected ATOMIC confirmation event: {event_id}")
            # These are explicit event inputs from the frozen public-state
            # axis.  They are deliberately not computed from dataset_gt_id or
            # raw SAM IDs; the posthoc scorer joins GT only after sealing.
            target_public = 1003
            other_public = 1004
            target_state = next((state for state, public in pairs if public == target_public), None)
            other_state = next((state for state, public in pairs if public == other_public), None)
            if target_state is None or other_state is None or target_public == other_public:
                raise RuntimeError("ATOMIC confirmation public-state authority is not distinct/available")
            authority = "explicit_simulated_human_atomic_public_state_pair_from_frozen_prefix"
        else:
            raise RuntimeError(f"unsupported confirmation action: {action}")
        if not isinstance(event.get("current_gt_box"), list) or len(event["current_gt_box"]) != 4:
            raise RuntimeError(f"confirmation event lacks frozen simulated human box: {event_id}")
        if event.get("runtime_future_gt_used") is not False or event.get("runtime_gt_read") is not False:
            raise RuntimeError(f"confirmation event contains a runtime GT flag: {event_id}")
        specs.append({
            "event_id": event_id,
            "sequence": str(event["sequence"]),
            "event_frame": event_frame,
            "action_type": action,
            "current_gt_box": [float(value) for value in event["current_gt_box"]],
            "prefix_range": [int(value) for value in event["prefix_range"]],
            "future_window": future_window,
            "main_b0_path": str(main_path),
            "main_b0_sha256": sha256_file(main_path),
            "main_b0_event_row_sha256": hashlib.sha256(
                json.dumps(event_rows[0], ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
            ).hexdigest(),
            "target_public_id": int(target_public),
            "target_association_state_id": int(target_state),
            "other_public_id": None if other_public is None else int(other_public),
            "public_axis_at_event": public_axis,
            "authority": authority,
            "public_id_from_gt_id": False,
            "public_id_from_raw_sam_id": False,
            "public_id_inference": False,
            "original_stage08_b1_status": str(
                branches.get("B1_SPATIAL_CORRECTION_ONLY", {}).get("action_precondition_status")
            ),
            "interaction_source": "simulated_from_gt",
            "not_real_human_evidence": True,
            "runtime_future_gt_used": False,
            "runtime_gt_read": False,
            "posthoc_gt_used": False,
        })
    body: dict[str, Any] = {
        "schema_version": "N72R7_SEQUENCE_DISJOINT_CONFIRMATION_PROTOCOL_V1",
        "created_at_utc": now_utc(),
        "event_policy": str(EVENT_POLICY),
        "event_policy_sha256": sha256_file(EVENT_POLICY),
        "stage08_manifest": str(STAGE08),
        "stage08_manifest_sha256": sha256_file(STAGE08),
        "events": specs,
        "sequence_count": len({item["sequence"] for item in specs}),
        "confirmation_sequences": sorted({item["sequence"] for item in specs}),
        "selection_rule": "pre-registered deferred sequences only; no development/replay outcome fields",
        "candidate_definition": "frozen N72R5 B0 stream plus one fresh official target-session stream from explicit event box",
        "best_frozen_mechanism": "N72R7 R2 learned target/NONE selector with D1 B0-only and D2 B0+target-session pool",
        "checkpoint_and_solver_frozen": True,
        "metric_definition_frozen": True,
        "runtime_future_gt_used": False,
        "runtime_gt_read": False,
        "posthoc_gt_used": False,
        "interaction_source": "simulated_from_gt",
        "not_real_human_evidence": True,
        "confirmation_not_used_for_training_or_architecture_selection": True,
    }
    body["protocol_sha256"] = hashlib.sha256(
        json.dumps(body, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    ).hexdigest()
    return body


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--attempt", type=int, default=1)
    args = parser.parse_args()
    output = args.output if args.output.is_absolute() else ROOT / args.output
    try:
        payload = build_protocol()
        atomic_json(output, payload)
        print(json.dumps({"status": "PASS_N72R7_CONFIRMATION_PROTOCOL_FROZEN", "event_count": len(payload["events"]), "protocol_sha256": payload["protocol_sha256"]}, sort_keys=True))
        return 0
    except Exception as exc:
        failure = {
            "schema_version": "N72R7_CONFIRMATION_PROTOCOL_FAILURE_V1",
            "status": "FAIL_PROTOCOL_FREEZE",
            "attempt": int(args.attempt),
            "exception_type": type(exc).__name__,
            "exception": str(exc),
            "traceback": traceback.format_exc(),
            "output_not_written": True,
        }
        failure_path = output.parent / "attempts" / f"protocol_freeze_failure_attempt{int(args.attempt)}.json"
        atomic_json(failure_path, failure)
        print(json.dumps({"status": failure["status"], "failure_artifact": str(failure_path), "exception": str(exc)}, sort_keys=True))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
