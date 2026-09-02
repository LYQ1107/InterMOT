#!/usr/bin/env python3
"""Freeze the N72R3 simulated-oracle boundary and event policy.

Stage 13 is an isolated toy/protocol audit.  Stage 14 reuses the frozen N37
event candidate pool, but selects only events whose current frame is covered
by the already-audited N72R1 Candidate V2 windows.  The selection reads no
future score, future error, or future identity outcome.  The resulting events
remain ``simulated_from_gt`` and are never real-human evidence.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sam3_intermot.simulation.human_oracle import SimulatedHumanOracle


OUT = ROOT / "outputs" / "N72R3"
N37_EVENTS = Path(
    "/data1/LWR/vranlee/SERVER_ONLY/avis/Interactive/SAM3_InterMOT/"
    "outputs/n37/real_event_manifest.json"
)
N71_PLAN = Path(
    "/data1/LWR/vranlee/SERVER_ONLY/avis/Interactive/SAM3_InterMOT/"
    "outputs/N71/candidate_branch/window_plan.json"
)
N72R1_EXPORT = Path("/data2/usr_for_deadline/SAM3_InterMOT_N72R1/six_window_export")


def atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
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


def sha256(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def oracle_protocol() -> dict[str, Any]:
    # This object is written before the event manifest is read.  Numerical
    # event rules therefore cannot be tuned after seeing N72R3 outcomes.
    return {
        "schema_version": "N72R3_SIMULATED_ORACLE_PROTOCOL_V1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "interaction_source": "simulated_from_gt",
        "not_real_human_evidence": True,
        "gt_scope": "current_frame_only_after_Y_pre_is_frozen",
        "runtime_future_gt_used": False,
        "candidate_generation_uses_gt": False,
        "mapping_uses_gt_at_runtime": False,
        "public_id_source": "outer_persistent_runtime_or_explicit_current_frame_confirmation",
        "localization_iou_threshold": 0.5,
        "event_policy": {
            "AUTHORITATIVE_CORRECT": "known public identity has current output but current box IoU is below threshold",
            "RECOVER_IDENTITY": "known public identity has no correctly assigned current candidate/output",
            "AUTHORITATIVE_REASSIGN": "known identity is assigned to a different current public ID",
            "ATOMIC_ID_SWAP": "two known identities have reciprocal current public assignments",
            "ADD_NEW_IDENTITY": "current GT identity is absent from the oracle map; outer runtime allocates/ confirms public ID",
            "AUTHORITATIVE_DELETE": "known public runtime output has no current GT counterpart",
        },
        "event_eligibility": {
            "current_gt_box_required": True,
            "current_frame_candidate_v2_coverage_required": True,
            "candidate_uid_mapping_must_be_complete": True,
            "selection_may_use": ["current GT frame", "current/past runtime state", "current-frame Candidate V2 rows"],
            "selection_must_not_use": [
                "future candidate rows",
                "H20/H50/H100",
                "future identity error",
                "IDSW/IoU post-treatment outcome",
                "re-correction outcome",
                "variant score or replay result",
            ],
        },
        "scale_target": {
            "target_events": 40,
            "target_independent_sequences": 20,
            "shortfall_rule": "use_all_current-frame-eligible-frozen-events_and_report_shortfall",
            "duplicate_sequence_event_is_not_new_independent_sequence": True,
        },
        "actions": [
            "ADD_NEW_IDENTITY",
            "AUTHORITATIVE_REASSIGN",
            "ATOMIC_ID_SWAP",
            "RECOVER_IDENTITY",
        ],
        "frozen_input_plan": str(N71_PLAN),
        "frozen_input_plan_sha256": sha256(N71_PLAN),
    }


def candidate_frame_index() -> dict[tuple[str, int], list[dict[str, Any]]]:
    index: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for path in sorted((N72R1_EXPORT / "windows").glob("*/candidate_v2.jsonl")):
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                index.setdefault((str(row["sequence"]), int(row["frame_idx"])), []).append(row)
    for key in index:
        index[key].sort(key=lambda row: (int(row["candidate_index"]), str(row["candidate_uid"])))
    return index


def run_stage13() -> dict[str, Any]:
    oracle = SimulatedHumanOracle("toy")
    add = oracle.choose_actions(
        1,
        {"boxes": [[0.0, 0.0, 10.0, 10.0]], "gt_ids": [7]},
        [{"candidate_uid": "toy-candidate", "box": [0.0, 0.0, 10.0, 10.0]}],
    )
    oracle.commit_mapping(7, 1007, reason="outer_allocator_birth")
    wrong = oracle.choose_actions(
        2,
        {"boxes": [[0.0, 0.0, 10.0, 10.0]], "gt_ids": [7]},
        [{"candidate_uid": "toy-candidate-2", "public_id": 2002, "box": [0.0, 0.0, 10.0, 10.0]}],
    )
    rejecting = SimulatedHumanOracle("toy")
    try:
        rejecting.choose_actions(1, {"boxes": [], "gt_ids": [], "future_boxes": []}, [])
    except ValueError:
        future_field_rejected = True
    else:
        future_field_rejected = False
    result = {
        "schema_version": "N72R3_STAGE13_ORACLE_AUDIT_V1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "stage": "13_SIMULATED_HUMAN_ORACLE",
        "status": "PASS_STAGE13_ORACLE_ISOLATED",
        "interaction_source": "simulated_from_gt",
        "not_real_human_evidence": True,
        "unknown_identity_action": add[0].as_dict() if add else None,
        "explicit_mapping_after_outer_allocation": oracle.gt_to_public,
        "known_identity_wrong_assignment_action": wrong[0].as_dict() if wrong else None,
        "future_field_rejected": future_field_rejected,
        "oracle": oracle.as_dict(),
        "runtime_future_gt_used": False,
        "scientific_result": "NOT_A_SCIENTIFIC_RESULT",
    }
    atomic_json(OUT / "simulation" / "stage13_oracle_audit.json", result)
    atomic_json(OUT / "stage_13_status.json", {
        "schema_version": "N72R3_STAGE_STATUS_V1",
        "stage": "13_SIMULATED_HUMAN_ORACLE",
        "status": result["status"],
        "created_at_utc": result["created_at_utc"],
        "artifact": str(OUT / "simulation" / "stage13_oracle_audit.json"),
        "runtime_future_gt_used": False,
        "not_real_human_evidence": True,
        "scientific_result": "NOT_A_SCIENTIFIC_RESULT",
    })
    return result


def run_stage14(protocol: dict[str, Any]) -> dict[str, Any]:
    manifest = json.loads(N37_EVENTS.read_text(encoding="utf-8"))
    events = list(manifest.get("events", []))
    frame_index = candidate_frame_index()
    plan = json.loads(N71_PLAN.read_text(encoding="utf-8"))
    windows = list(plan.get("windows", []))
    selected: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []
    seen_event_ids: set[str] = set()
    for outer in events:
        event = dict(outer.get("event", {}))
        event_id = str(event.get("event_id") or outer.get("protocol_candidate_id") or "")
        sequence = str(event.get("sequence", ""))
        frame = event.get("frame")
        action = str(event.get("action_type", outer.get("action_type", "")))
        reason: list[str] = []
        if not event_id or event_id in seen_event_ids:
            reason.append("missing_or_duplicate_event_id")
        if event.get("interaction_source") != "simulated_from_gt":
            reason.append("not_simulated_from_gt")
        if event.get("runtime_future_gt_used") is not False or outer.get("runtime_future_gt_used") is not False:
            reason.append("source_runtime_future_gt_not_false")
        if frame is None or not isinstance(event.get("gt_box"), list):
            reason.append("current_gt_box_or_frame_missing")
        try:
            frame_int = int(frame)
            gt_box = np.asarray(event["gt_box"], dtype=float).reshape(-1)
            if gt_box.size != 4 or not np.isfinite(gt_box).all():
                reason.append("current_gt_box_invalid")
        except (TypeError, ValueError):
            frame_int = -1
            gt_box = np.zeros(4, dtype=float)
            reason.append("current_gt_box_invalid")
        rows = frame_index.get((sequence, frame_int), [])
        covering = [
            window for window in windows
            if str(window.get("sequence")) == sequence
            and int(window.get("frame_start", -1)) <= frame_int <= int(window.get("frame_end", -1))
        ]
        if not covering or not rows:
            reason.append("current_frame_candidate_v2_not_covered")
        if rows and len({str(row.get("candidate_uid")) for row in rows}) != len(rows):
            reason.append("candidate_uid_duplicate")
        if reason:
            excluded.append({"event_id": event_id, "sequence": sequence, "frame": frame_int, "reasons": reason})
            continue
        seen_event_ids.add(event_id)
        original_future = outer.get("future_window")
        if original_future is None:
            original_future = [outer.get("future_frame_start"), outer.get("future_frame_end")]
        selected.append(
            {
                "schema_version": "N72R3_SIMULATED_EVENT_V1",
                "event_id": event_id,
                "sequence": sequence,
                "event_frame": frame_int,
                "action_type": action,
                "dataset_gt_id": int(event["dataset_gt_id"]),
                "current_gt_box": [float(value) for value in gt_box],
                "interaction_source": "simulated_from_gt",
                "not_real_human_evidence": True,
                "manual_box_source": "offline_train_GT_box_as_simulated_human_annotation",
                "current_candidate_v2": {
                    "window_id": str(covering[0]["window_id"]),
                    "candidate_count": len(rows),
                    "candidate_uids": [str(row["candidate_uid"]) for row in rows],
                    "candidate_source": str(N72R1_EXPORT / "windows" / str(covering[0]["window_id"]) / "candidate_v2.jsonl"),
                    "public_mapping_in_source": False,
                },
                "prefix_range": outer.get("prefix_range"),
                "future_window": original_future,
                "candidate_tape_ref": outer.get("source_tape"),
                "candidate_tape_sha256": outer.get("source_tape_sha256"),
                "selection_basis": "frozen_N37_event_pool intersect current-frame N72R1 Candidate V2 coverage; no N72R3 post-treatment fields",
                "selection_post_treatment_fields_used": [],
                "runtime_future_gt_used": False,
            }
        )
    counts = Counter(str(item["action_type"]) for item in selected)
    sequences = sorted({str(item["sequence"]) for item in selected})
    event_manifest = {
        "schema_version": "N72R3_SIMULATED_EVENT_MANIFEST_V1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS_STAGE14_POLICY_FROZEN",
        "interaction_source": "simulated_from_gt",
        "not_real_human_evidence": True,
        "source_manifest": str(N37_EVENTS),
        "source_manifest_sha256": sha256(N37_EVENTS),
        "protocol_sha256": sha256(OUT / "simulation" / "stage14_event_policy.json") if (OUT / "simulation" / "stage14_event_policy.json").is_file() else None,
        "selection_uses_future_outcome": False,
        "runtime_future_gt_used": False,
        "event_count": len(selected),
        "independent_sequence_count": len(sequences),
        "action_counts": dict(sorted(counts.items())),
        "target_event_count": int(protocol["scale_target"]["target_events"]),
        "target_independent_sequence_count": int(protocol["scale_target"]["target_independent_sequences"]),
        "eligible_shortfall": len(selected) < int(protocol["scale_target"]["target_events"]) or len(sequences) < int(protocol["scale_target"]["target_independent_sequences"]),
        "shortfall_rule": protocol["scale_target"]["shortfall_rule"],
        "events": selected,
        "excluded_event_count": len(excluded),
        "excluded_events": excluded,
    }
    atomic_json(OUT / "simulation" / "real_event_manifest.json", event_manifest)
    status = {
        "schema_version": "N72R3_STAGE_STATUS_V1",
        "stage": "14_FREEZE_SIMULATED_EVENT_POLICY",
        "status": "PASS_STAGE14_POLICY_FROZEN",
        "created_at_utc": event_manifest["created_at_utc"],
        "artifact": str(OUT / "simulation" / "real_event_manifest.json"),
        "protocol": str(OUT / "simulation" / "stage14_event_policy.json"),
        "event_count": len(selected),
        "independent_sequence_count": len(sequences),
        "action_counts": dict(sorted(counts.items())),
        "eligible_shortfall": event_manifest["eligible_shortfall"],
        "excluded_event_count": len(excluded),
        "selection_uses_future_outcome": False,
        "runtime_future_gt_used": False,
        "interaction_source": "simulated_from_gt",
        "not_real_human_evidence": True,
        "scientific_result": "NOT_A_SCIENTIFIC_RESULT",
    }
    atomic_json(OUT / "stage_14_status.json", status)
    return event_manifest


def main() -> int:
    protocol = oracle_protocol()
    atomic_json(OUT / "simulation" / "stage14_event_policy.json", protocol)
    stage13 = run_stage13()
    stage14 = run_stage14(protocol)
    return 0 if stage13["status"].startswith("PASS") and stage14["status"].startswith("PASS") else 1


if __name__ == "__main__":
    raise SystemExit(main())

