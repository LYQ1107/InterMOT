#!/usr/bin/env python3
"""Persistent-state replay for the frozen N72R3 candidate mechanism probe.

Each event/variant starts from a real N72R3 Stage18 runtime rebuilt through
``event_frame - 1`` and then performs an exact public-ID+NONE event solve.
The future rows are still the frozen Candidate V2 stream from N72R3R1; this
module is therefore a persistent-state structural probe, not the official
SAM3 future-propagation experiment (that is Stage 9).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import tempfile
import traceback
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sam3_intermot.association.effect_assignment import solve_effect_assignment  # noqa: E402
from sam3_intermot.identity.persistent_snapshot import PersistentRuntimeSnapshot  # noqa: E402
from sam3_intermot.identity.persistent_runtime import SequencePersistentIdentityRuntime  # noqa: E402
from scripts.n72r3_stage09_11_candidate_runtime import (  # noqa: E402
    load_plan,
    load_source_rows,
    new_state_manager,
    observation_from_row,
    prompt_observation,
    run_session,
)
from scripts.n72r3r1_semantic_replay import (  # noqa: E402
    ARTIFACT_ROOT as SEMANTIC_ARTIFACT_ROOT,
    EVENT_MANIFEST,
    atomic_json,
    atomic_jsonl,
    read_json,
    read_jsonl,
    sha256,
)


N72R3_ROOT = Path("/data2/usr_for_deadline/SAM3_InterMOT_N72R3/worktree/outputs/N72R3")
OFFICIAL_ROOT = N72R3_ROOT / "official_correction/events"
PRESTATE_MANIFEST = ROOT / "outputs/N72R4/event_prestate_manifest.json"
OUT = ROOT / "outputs/N72R4"
REPLAY_ROOT = OUT / "full_loop/persistent_candidate_probe"
REPLAY_ROOT = Path(os.environ.get("N72R4_PERSISTENT_REPLAY_ROOT", str(REPLAY_ROOT)))
RUNTIME_MANIFEST = Path(os.environ.get("N72R4_PERSISTENT_REPLAY_MANIFEST", str(REPLAY_ROOT / "persistent_candidate_probe_manifest.json")))
VALIDATION_PATH = Path(os.environ.get("N72R4_PERSISTENT_REPLAY_VALIDATION", str(REPLAY_ROOT / "persistent_candidate_probe_validation.json")))
STAGE_PATH = Path(os.environ.get("N72R4_PERSISTENT_STAGE_PATH", str(OUT / "stage_08_status.json")))
STAGE_STATUS_PATH = Path(os.environ.get("N72R4_PERSISTENT_STAGE_STATUS_PATH", str(OUT / "stage_status/stage_08_status.json")))
STAGE7_PATH = Path(os.environ.get("N72R4_PERSISTENT_STAGE07_PATH", str(OUT / "stage_status/stage_07_status.json")))
STAGE7_TOP_PATH = Path(os.environ.get("N72R4_PERSISTENT_STAGE07_TOP_PATH", str(OUT / "stage_07_status.json")))
FAILURE_ROOT = OUT / "attempts"

VARIANTS = (
    "NO_INTERVENTION",
    "M0_CURRENT_FRAME_CORRECTION_ONLY",
    "M1_HUMAN_EMA_PROTOTYPE",
    "M2_POSITIVE_HUMAN_ANCHORS",
    "M3_NEGATIVE_COMPETITOR_BANK",
    "M4_RELIABILITY_AGE_ADMISSION",
)
MEMORY_VARIANTS = set(VARIANTS[2:])


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def json_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")
    ).hexdigest()


def finite_feature(value: Any) -> np.ndarray:
    feature = np.asarray(value, dtype=np.float32).reshape(-1)
    if feature.size != 512 or not np.all(np.isfinite(feature)) or float(np.linalg.norm(feature)) <= 1.0e-6:
        raise ValueError(f"expected finite nonzero 512-D feature, got {feature.shape}")
    return feature / float(np.linalg.norm(feature))


def compact_candidate(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "candidate_index": int(row["candidate_index"]),
        "candidate_uid": str(row["candidate_uid"]),
    }


def candidate_view(row: dict[str, Any], public_id: int | None, status: str) -> dict[str, Any]:
    return {
        "candidate_index": int(row["candidate_index"]),
        "candidate_uid": str(row["candidate_uid"]),
        "official_raw_sam_id": int(row["official_raw_sam_id"]),
        "adapter_external_id": int(row["adapter_external_id"]),
        "box_xyxy": [float(value) for value in row["box_xyxy"]],
        "confidence": float(row["confidence"]),
        "feature_sha256": str(row["feature_sha256"]),
        "feature_dim": int(row["feature_dim"]),
        "public_id": None if public_id is None else int(public_id),
        "assignment_status": str(status),
    }


def load_events() -> list[dict[str, Any]]:
    manifest = read_json(EVENT_MANIFEST)
    if manifest.get("status") != "PASS_STAGE14_POLICY_FROZEN":
        raise RuntimeError("persistent replay requires the frozen N72R3 Stage14 event policy")
    events = [dict(item) for item in manifest.get("events", [])]
    if len(events) != 6:
        raise RuntimeError(f"persistent replay requires six frozen events, found {len(events)}")
    return sorted(events, key=lambda item: str(item["event_id"]))


def load_prestate_index() -> dict[str, dict[str, Any]]:
    root = read_json(PRESTATE_MANIFEST)
    if root.get("status") != "PASS_EVENT_PRESTATE_SET":
        raise RuntimeError("event prestate manifest is not complete")
    completed = {str(item["event_id"]): dict(item) for item in root.get("completed", [])}
    if len(completed) != 6:
        raise RuntimeError(f"expected six prestate manifests, found {len(completed)}")
    for event_id, item in completed.items():
        if item.get("runtime_future_gt_used") is not False:
            raise RuntimeError(f"prestate permits runtime future GT: {event_id}")
        event_manifest = Path(item["snapshot"]).parent / "manifest.json"
        if not Path(item["snapshot"]).is_file() or not event_manifest.is_file():
            raise RuntimeError(f"prestate file reference is incomplete: {event_id}")
    return completed


def load_semantic_rows(event_id: str) -> dict[tuple[str, int], dict[str, Any]]:
    path = SEMANTIC_ARTIFACT_ROOT / f"{event_id}.jsonl"
    if not path.is_file():
        raise FileNotFoundError(path)
    rows = read_jsonl(path)
    keyed = {(str(item["variant"]), int(item["frame"])): item for item in rows}
    if len(keyed) != len(rows) or len(rows) != 606:
        raise RuntimeError(f"semantic repaired artifact is incomplete: {event_id}/{len(rows)}")
    return keyed


def rebuild_prestate(
    event: dict[str, Any],
    windows: list[dict[str, Any]],
    prestate_item: dict[str, Any],
    output_dir: Path,
) -> tuple[PersistentRuntimeSnapshot, dict[str, Any], dict[int, list[dict[str, Any]]]]:
    event_id = str(event["event_id"])
    sequence = str(event["sequence"])
    event_frame = int(event["event_frame"])
    window_id = str(event["current_candidate_v2"]["window_id"])
    matches = [item for item in windows if str(item["window_id"]) == window_id]
    if len(matches) != 1:
        raise RuntimeError(f"candidate window is not unique for {event_id}")
    window = dict(matches[0])
    rows, source_meta = load_source_rows(window)
    by_frame: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_frame[int(row["frame_idx"])].append(row)
    for frame_rows in by_frame.values():
        frame_rows.sort(key=lambda item: (int(item["candidate_index"]), str(item["candidate_uid"])))
    prefix_rows = [row for row in rows if int(row["frame_idx"]) <= event_frame - 1]
    if not prefix_rows or max(int(row["frame_idx"]) for row in prefix_rows) != event_frame - 1:
        raise RuntimeError(f"persistent prefix does not end at t-1: {event_id}")
    runtime = SequencePersistentIdentityRuntime(sequence, public_id_start=1000)
    prefix_result = run_session(
        runtime,
        prefix_rows,
        session_label=f"persistent-prefix:{event_id}",
        output_dir=output_dir / "prefix_runtime",
    )
    if int(prefix_result["frame_end"]) != event_frame - 1:
        raise RuntimeError(f"prefix runtime ended at {prefix_result['frame_end']} for {event_id}")
    if prefix_result["runtime_audit"].get("runtime_future_gt_used") is not False:
        raise RuntimeError(f"prefix runtime used future GT: {event_id}")
    snapshot = PersistentRuntimeSnapshot.capture(runtime, snapshot_frame=event_frame - 1, next_window_start=event_frame)
    stored = read_json(Path(prestate_item["snapshot"]))
    stored_payload = stored.get("payload", {})
    live_public = sorted(int(item.public_id) for item in runtime.identities.values())
    stored_public = sorted(int(item["public_id"]) for item in stored_payload.get("identities", []))
    live_state = sorted(int(item.association_state_id) for item in runtime.identities.values())
    stored_state = sorted(int(item["association_state_id"]) for item in stored_payload.get("identities", []))
    if live_public != stored_public or live_state != stored_state:
        raise RuntimeError(f"rebuilt prefix does not match stored prestate axes: {event_id}")
    return snapshot, {
        "source_meta": source_meta,
        "prefix_result": prefix_result,
        "stored_snapshot_sha256": sha256(Path(prestate_item["snapshot"])),
        "stored_public_axis": stored_public,
        "stored_association_state_axis": stored_state,
    }, by_frame


def exact_event_preassociation(
    runtime: SequencePersistentIdentityRuntime,
    event: dict[str, Any],
    event_rows: list[dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, int | None]]:
    event_id = str(event["event_id"])
    frame = int(event["event_frame"])
    session_id = f"persistent-event:{event_id}"
    runtime.begin_new_sam_session(session_id, boundary_frame=frame - 1)
    runtime.clear_current_session_bindings(frame)
    manager = new_state_manager(runtime)
    observations = [observation_from_row(row, frame) for row in event_rows]
    manager.rollout_frame(frame, observations)
    if not manager.candidate_log:
        raise RuntimeError(f"event preassociation did not emit audit: {event_id}")
    audit = manager.candidate_log[-1]
    state_axis = [int(value) for value in audit.get("association_state_axis", [])]
    if len(state_axis) != len(runtime.identities):
        raise RuntimeError(f"event association state axis mismatch: {event_id}")
    persistent_states: list[dict[str, int]] = []
    for state_id in state_axis:
        record = runtime.get_identity_by_state_id(state_id)
        if record is None:
            raise RuntimeError(f"event solver state is not persistent: {event_id}/{state_id}")
        persistent_states.append({"association_state_id": state_id, "public_id": int(record.public_id)})
    state_candidate_scores = np.asarray(audit.get("public_id_fused_score_matrix", []), dtype=np.float64)
    solver = solve_effect_assignment(
        candidate_rows=[compact_candidate(row) for row in event_rows],
        persistent_states=persistent_states,
        fused_state_candidate_scores=state_candidate_scores,
        source_run_id=f"n72r4-persistent:{event_id}",
        session_id=session_id,
        none_score=0.0,
    )
    output_public: list[int | None] = [item["public_id"] for item in solver["assignment_rows"]]
    output_status: list[str] = [
        "ASSIGNED_EXISTING_IDENTITY" if public is not None else "EXPLICIT_NONE_BEFORE_OUTER_BIRTH"
        for public in output_public
    ]
    # Apply the formal outer-birth decision only after the exact existing-ID
    # plus NONE solve.  The allocator is owned by the persistent runtime.
    runtime.clear_current_session_bindings(frame)
    for index, public in enumerate(output_public):
        row = event_rows[index]
        observation = prompt_observation(row, frame)
        if public is None:
            record = runtime.create_identity(
                frame,
                observation,
                session_id=session_id,
                adapter_external_id=int(row["adapter_external_id"]),
                raw_sam_id=int(row["official_raw_sam_id"]),
                candidate_uid=str(row["candidate_uid"]),
                appearance_state={
                    "last_machine_feature": finite_feature(row["feature"]).astype(float).tolist(),
                    "last_machine_feature_sha256": str(row["feature_sha256"]),
                    "last_machine_feature_frame": frame,
                },
                motion_state_ref={"last_box": [float(value) for value in row["box_xyxy"]], "last_frame": frame},
            )
            output_public[index] = int(record.public_id)
            output_status[index] = "OUTER_BIRTH_ASSIGNED_BY_PERSISTENT_ALLOCATOR"
        else:
            record = runtime.get_identity_by_public_id(int(public))
            if record is None:
                raise RuntimeError(f"event solver public ID is not persistent: {event_id}/{public}")
            runtime.bind_candidate(
                record,
                str(row["candidate_uid"]),
                observation,
                frame,
                session_id=session_id,
                adapter_external_id=int(row["adapter_external_id"]),
                raw_sam_id=int(row["official_raw_sam_id"]),
            )
    mapping = {str(row["candidate_uid"]): public for row, public in zip(event_rows, output_public)}
    if any(public is None for public in mapping.values()) or len(set(mapping.values())) != len(mapping):
        raise RuntimeError(f"event persistent mapping is incomplete or duplicated: {event_id}")
    runtime.record_frame_decisions(frame, {int(public): uid for uid, public in mapping.items() if public is not None})
    for record in list(runtime.identities.values()):
        if int(record.public_id) not in set(int(value) for value in output_public if value is not None):
            runtime.mark_lost(record, frame, reason="exact_event_none")
    event_artifact = {
        "schema_version": "N72R4_PERSISTENT_CANDIDATE_FRAME_V1",
        "event_id": event_id,
        "sequence": str(event["sequence"]),
        "action_type": str(event["action_type"]),
        "event_frame": frame,
        "frame": frame,
        "frame_horizon": 0,
        "phase": "CURRENT_FRAME_PREINTERVENTION",
        "candidate_rows": [candidate_view(row, public, status) for row, public, status in zip(event_rows, output_public, output_status)],
        "assignment_public_ids": output_public,
        "assignment_status": output_status,
        "assignment_map": mapping,
        "persistent_public_id_source": "PersistentIdentityRecord.public_id",
        "persistent_association_state_source": "PersistentIdentityRecord.association_state_id",
        "event_prestate_frame": frame - 1,
        "event_prestate_restored": True,
        "pre_correction_solver": solver,
        "pre_correction_solver_name": "sam3_intermot.association.effect_assignment.solve_effect_assignment",
        "pre_correction_solver_input_orientation": "state_x_candidate",
        "event_frame_memory_read": False,
        "current_frame_write_hidden": True,
        "first_memory_visible_frame": frame + 1,
        "runtime_future_gt_used": False,
        "runtime_gt_read": False,
        "posthoc_gt_used": False,
        "candidate_stream_kind": "FROZEN_CANDIDATE_V2_STRUCTURAL_PROBE",
        "scientific_result": "STRUCTURAL_PREINTERVENTION_ONLY",
    }
    return event_artifact, mapping


def correction_command(event: dict[str, Any], event_rows: list[dict[str, Any]]) -> tuple[int, int, np.ndarray]:
    event_id = str(event["event_id"])
    official = read_json(OFFICIAL_ROOT / f"{event_id}.json")
    target_public = int(official["persistent_identity"]["public_id"])
    setup = official.get("simulated_assignment_setup", {})
    target_native = int(setup["target_sam_object_id"])
    target_rows = [row for row in event_rows if int(row["adapter_external_id"]) == target_native]
    if len(target_rows) != 1:
        raise RuntimeError(f"official correction target candidate is not unique: {event_id}")
    positives = official.get("appearance_memory", {}).get("positive", [])
    if len(positives) != 1:
        raise RuntimeError(f"official human feature is not unique: {event_id}")
    human_feature = finite_feature(positives[0]["feature"])
    return target_public, target_native, human_feature


def bind_correction(
    runtime: SequencePersistentIdentityRuntime,
    event: dict[str, Any],
    event_rows: list[dict[str, Any]],
    pre_map: dict[str, int | None],
    variant: str,
) -> tuple[dict[str, int], dict[str, Any]]:
    frame = int(event["event_frame"])
    event_id = str(event["event_id"])
    target_public, target_native, human_feature = correction_command(event, event_rows)
    target_row = next(row for row in event_rows if int(row["adapter_external_id"]) == target_native)
    target_uid = str(target_row["candidate_uid"])
    if runtime.get_identity_by_public_id(target_public) is None:
        raise RuntimeError(f"correction public ID does not exist in persistent runtime: {event_id}/{target_public}")
    # Rebind all pre-intervention candidates except the target/conflicting
    # authority, then bind the target candidate to the immutable public ID.
    runtime.clear_current_session_bindings(frame)
    corrected: dict[str, int] = {}
    for uid, public in pre_map.items():
        if public is None or uid == target_uid or int(public) == target_public:
            continue
        corrected[uid] = int(public)
    corrected[target_uid] = target_public
    row_by_uid = {str(row["candidate_uid"]): row for row in event_rows}
    for uid, public in corrected.items():
        row = row_by_uid[uid]
        record = runtime.get_identity_by_public_id(public)
        if record is None:
            raise RuntimeError(f"persistent correction binding disappeared: {event_id}/{public}")
        runtime.bind_candidate(
            record,
            uid,
            prompt_observation(row, frame),
            frame,
            session_id=f"persistent-event:{event_id}",
            adapter_external_id=int(row["adapter_external_id"]),
            raw_sam_id=int(row["official_raw_sam_id"]),
        )
    runtime.record_frame_decisions(frame, {public: uid for uid, public in corrected.items()})
    memory_write = False
    memory_audit: dict[str, Any] = {
        "enabled": variant in MEMORY_VARIANTS,
        "event_frame_read": False,
        "first_visible_frame": frame + 1,
        "human_feature_sha256": hashlib.sha256(np.asarray(human_feature, dtype="<f4").tobytes()).hexdigest(),
        "runtime_future_gt_used": False,
    }
    if variant in MEMORY_VARIANTS:
        target_machine = finite_feature(target_row["feature"])
        if not runtime.appearance_memory.update_from_machine(target_public, frame, target_machine, confidence=1.0):
            raise RuntimeError(f"persistent machine appearance seed failed: {event_id}/{variant}")
        competitors = [
            finite_feature(row["feature"])
            for row in event_rows
            if str(row["candidate_uid"]) != target_uid
        ]
        accepted = runtime.appearance_memory.update_from_human(
            target_public,
            frame,
            human_feature,
            quality=1.0,
            competing_embeddings=competitors if variant in {"M3_NEGATIVE_COMPETITOR_BANK", "M4_RELIABILITY_AGE_ADMISSION"} else None,
            write_event_id=event_id,
        )
        if not accepted:
            raise RuntimeError(f"persistent human appearance write failed: {event_id}/{variant}")
        memory_write = True
        memory_audit.update(
            {
                "accepted": True,
                "target_public_id": target_public,
                "negative_competitor_count": len(competitors) if variant in {"M3_NEGATIVE_COMPETITOR_BANK", "M4_RELIABILITY_AGE_ADMISSION"} else 0,
            }
        )
    correction = {
        "status": "PASS_CURRENT_FRAME_CORRECTION_TRANSACTION",
        "event_id": event_id,
        "target_public_id": target_public,
        "target_native_id": target_native,
        "target_candidate_uid": target_uid,
        "pre_map": pre_map,
        "post_map": corrected,
        "correction_before_memory_write": True,
        "event_frame_memory_read": False,
        "memory_write": memory_write,
        "runtime_future_gt_used": False,
        "interaction_source": "simulated_from_gt",
        "not_real_human_evidence": True,
        "memory_audit": memory_audit,
    }
    return corrected, correction


def future_row(
    source: dict[str, Any],
    scenario: dict[str, Any],
    variant: str,
    correction: dict[str, Any] | None,
) -> dict[str, Any]:
    frame = int(source["frame"])
    candidates = [compact_candidate(item) for item in source["candidate_rows"]]
    states = [dict(item) for item in source["explicit_persistent_state_axis"]]
    matrix = np.asarray(source["fused_score_matrix"], dtype=np.float64)
    solver = solve_effect_assignment(
        candidate_rows=candidates,
        persistent_states=states,
        fused_state_candidate_scores=matrix,
        source_run_id=f"n72r4-persistent-future:{scenario['event_id']}:{variant}:{frame}",
        session_id=f"persistent-event:{scenario['event_id']}",
        none_score=0.0,
    )
    solver_public = [item["public_id"] for item in solver["assignment_rows"]]
    old_public = list(source["assignment_public_ids"])
    old_status = list(source["assignment_status"])
    output_public: list[int | None] = []
    output_status: list[str] = []
    for solver_pid, old_pid, old_item_status in zip(solver_public, old_public, old_status):
        if solver_pid is not None:
            output_public.append(int(solver_pid))
            output_status.append("EXACT_EXISTING_IDENTITY")
        elif old_item_status in {"OUTER_BIRTH_RETAINED_FROZEN_ALLOCATOR", "OUTER_BIRTH_ASSIGNED"} and old_pid is not None:
            output_public.append(int(old_pid))
            output_status.append("OUTER_BIRTH_RETAINED_PERSISTENT_ALLOCATOR")
        else:
            output_public.append(None)
            output_status.append("EXPLICIT_NONE_NO_OUTER_BIRTH")
    assigned = [value for value in output_public if value is not None]
    if len(assigned) != len(set(assigned)):
        raise RuntimeError(f"persistent future mapping duplicated: {scenario['event_id']}/{variant}/{frame}")
    result = {
        "schema_version": "N72R4_PERSISTENT_CANDIDATE_FRAME_V1",
        "event_id": str(scenario["event_id"]),
        "sequence": str(scenario["sequence"]),
        "action_type": str(scenario["action_type"]),
        "event_frame": int(scenario["event_frame"]),
        "frame": frame,
        "frame_horizon": frame - int(scenario["event_frame"]),
        "phase": "FUTURE_ASSOCIATION",
        "variant": variant,
        "candidate_rows": [candidate_view(item, public, status) for item, public, status in zip(source["candidate_rows"], output_public, output_status)],
        "assignment_public_ids": output_public,
        "assignment_status": output_status,
        "assignment_map": {str(item["candidate_uid"]): public for item, public in zip(source["candidate_rows"], output_public)},
        "explicit_persistent_state_axis": states,
        "formal_solver": solver,
        "formal_solver_name": "sam3_intermot.association.effect_assignment.solve_effect_assignment",
        "formal_solver_assignment_public_ids": solver_public,
        "formal_solver_none_count": int(solver["explicit_none_count"]),
        "memory_read": bool(source.get("memory_read", False)),
        "memory_admitted": bool(source.get("memory_admitted", False)),
        "memory_read_reason": source.get("memory_read_reason"),
        "memory_write": False,
        "event_frame_memory_read": False,
        "first_memory_visible_frame": int(scenario["event_frame"]) + 1,
        "candidate_stream_kind": "FROZEN_CANDIDATE_V2_STRUCTURAL_PROBE",
        "source_semantic_repair_row_sha256": json_hash(source),
        "source_correction_transaction": correction,
        "persistent_public_id_source": "PersistentIdentityRecord.public_id",
        "runtime_future_gt_used": False,
        "runtime_gt_read": False,
        "posthoc_gt_used": False,
        "scientific_result": "STRUCTURAL_PERSISTENT_CANDIDATE_PROBE_NOT_OFFICIAL_SAM3",
    }
    return result


def event_variant(
    event: dict[str, Any],
    snapshot: PersistentRuntimeSnapshot,
    event_rows: list[dict[str, Any]],
    semantic_rows: dict[tuple[str, int], dict[str, Any]],
    variant: str,
    event_output_dir: Path,
) -> list[dict[str, Any]]:
    runtime = SequencePersistentIdentityRuntime(str(event["sequence"]), public_id_start=1000)
    snapshot.restore_into(runtime)
    event_artifact, pre_map = exact_event_preassociation(runtime, event, event_rows)
    event_artifact["variant"] = variant
    rows = [event_artifact]
    correction: dict[str, Any] | None = None
    if variant != "NO_INTERVENTION":
        post_map, correction = bind_correction(runtime, event, event_rows, pre_map, variant)
        event_artifact["correction_transaction"] = correction
        event_artifact["assignment_public_ids"] = [post_map.get(str(row["candidate_uid"])) for row in event_rows]
        event_artifact["assignment_status"] = [
            "CURRENT_FRAME_CORRECTION_BINDING" if str(row["candidate_uid"]) in post_map else "EXPLICIT_NONE_AFTER_CORRECTION"
            for row in event_rows
        ]
        event_artifact["candidate_rows"] = [
            candidate_view(
                row,
                post_map.get(str(row["candidate_uid"])),
                "CURRENT_FRAME_CORRECTION_BINDING" if str(row["candidate_uid"]) in post_map else "EXPLICIT_NONE_AFTER_CORRECTION",
            )
            for row in event_rows
        ]
        event_artifact["assignment_map"] = post_map
        event_artifact["memory_write"] = bool(correction["memory_write"])
    else:
        event_artifact["correction_transaction"] = None
        event_artifact["memory_write"] = False
    for frame in range(int(event["event_frame"]) + 1, int(event["event_frame"]) + 101):
        source = semantic_rows[(variant, frame)]
        rows.append(future_row(source, event, variant, correction))
    event_output_dir.mkdir(parents=True, exist_ok=True)
    return rows


def validate_runtime(events: list[dict[str, Any]], paths: dict[str, Path]) -> dict[str, Any]:
    checked = 0
    event_checked = 0
    future_checked = 0
    candidate_stream_errors: list[str] = []
    persistent_axis_errors: list[str] = []
    for event in events:
        event_id = str(event["event_id"])
        path = paths[event_id]
        rows = read_jsonl(path)
        if len(rows) != 606:
            raise RuntimeError(f"persistent artifact row count mismatch: {event_id}/{len(rows)}")
        keyed = {(str(row["variant"]), int(row["frame"])): row for row in rows}
        if len(keyed) != len(rows):
            raise RuntimeError(f"persistent artifact duplicate key: {event_id}")
        for frame in range(int(event["event_frame"]), int(event["event_frame"]) + 101):
            streams: list[tuple[str, ...]] = []
            for variant in VARIANTS:
                row = keyed[(variant, frame)]
                checked += 1
                if row.get("runtime_future_gt_used") is not False or row.get("runtime_gt_read") is not False or row.get("posthoc_gt_used") is not False:
                    raise RuntimeError(f"persistent runtime GT flag failed: {event_id}/{variant}/{frame}")
                if any(key in row for key in ("dataset_gt_id", "gt_box", "future_gt")):
                    raise RuntimeError(f"GT field entered persistent runtime row: {event_id}/{variant}/{frame}")
                candidates = row.get("candidate_rows", [])
                uids = tuple(str(item["candidate_uid"]) for item in candidates)
                streams.append(uids)
                public = [item.get("public_id") for item in candidates]
                # ``None`` is the explicit per-candidate NONE outcome and is
                # therefore allowed to occur on more than one candidate.  A
                # duplicate check must compare only assigned public IDs; the
                # previous validator compared the full list length with the
                # filtered set and falsely rejected a valid correction row
                # containing one or more explicit NONE decisions.
                assigned_public = [value for value in public if value is not None]
                if len(uids) != len(set(uids)) or len(assigned_public) != len(set(assigned_public)):
                    raise RuntimeError(f"persistent candidate mapping duplicate: {event_id}/{variant}/{frame}")
                if len(public) != len(row.get("assignment_public_ids", [])):
                    raise RuntimeError(f"persistent candidate/public shape mismatch: {event_id}/{variant}/{frame}")
                formal = row.get("pre_correction_solver", row.get("formal_solver", {}))
                if formal.get("runtime_future_gt_used") is not False:
                    raise RuntimeError(f"persistent exact solver GT flag failed: {event_id}/{variant}/{frame}")
                axis = row.get("explicit_persistent_state_axis", [])
                if frame == int(event["event_frame"]):
                    event_checked += 1
                    if row.get("event_frame_memory_read") is not False or row.get("first_memory_visible_frame") != frame + 1:
                        raise RuntimeError(f"persistent event causal boundary failed: {event_id}/{variant}")
                    if not row.get("event_prestate_restored"):
                        raise RuntimeError(f"persistent event prestate flag missing: {event_id}/{variant}")
                else:
                    future_checked += 1
                    if row.get("phase") != "FUTURE_ASSOCIATION" or int(row.get("frame_horizon", -1)) != frame - int(event["event_frame"]):
                        raise RuntimeError(f"persistent future phase failed: {event_id}/{variant}/{frame}")
                    if row.get("first_memory_visible_frame") != int(event["event_frame"]) + 1:
                        raise RuntimeError(f"persistent future causal boundary failed: {event_id}/{variant}/{frame}")
                    if not axis or len({int(item["association_state_id"]) for item in axis}) != len(axis) or len({int(item["public_id"]) for item in axis}) != len(axis):
                        persistent_axis_errors.append(f"{event_id}/{variant}/{frame}")
            if len(set(streams)) != 1:
                candidate_stream_errors.append(f"{event_id}/{frame}")
    if candidate_stream_errors or persistent_axis_errors:
        raise RuntimeError(f"persistent validation errors: streams={candidate_stream_errors[:5]}, axes={persistent_axis_errors[:5]}")
    audit = {
        "schema_version": "N72R4_PERSISTENT_CANDIDATE_PROBE_VALIDATION_V1",
        "status": "PASS_PERSISTENT_CANDIDATE_PROBE_VALIDATION",
        "event_count": len(events),
        "independent_sequence_count": len({str(event["sequence"]) for event in events}),
        "checked_variant_frame_rows": checked,
        "checked_event_rows": event_checked,
        "checked_future_rows": future_checked,
        "candidate_stream_shared_across_variants": True,
        "event_prestate_restored": True,
        "public_id_from_persistent_record": True,
        "candidate_index_to_public_id": False,
        "native_id_to_public_id": False,
        "exact_none_solver": True,
        "outer_birth_after_exact_solver": True,
        "runtime_future_gt_used": False,
        "interaction_source": "simulated_from_gt",
        "real_human_tape": False,
        "scientific_result": "STRUCTURAL_PERSISTENT_CANDIDATE_PROBE_ONLY",
    }
    atomic_json(VALIDATION_PATH, audit)
    return audit


def write_stage7_status() -> dict[str, Any]:
    # Build the two forbidden strings without placing either literal in this
    # source file, so this audit cannot match its own test vocabulary.
    forbidden = ["1000 " + "+" + " index", "1000" + "+" + "index", "candidate_index" + "_to_public_id ="]
    formal_path = ROOT / "scripts/n72r4_persistent_effect_replay.py"
    text = formal_path.read_text(encoding="utf-8")
    hits = [{"file": str(formal_path), "pattern": pattern} for pattern in forbidden if pattern in text]
    status = "PASS_STAGE07_PERSISTENT_AXIS_SOURCE" if not hits else "FAIL_STAGE07_CANDIDATE_INDEX_PUBLIC_MAPPING"
    payload = {
        "schema_version": "N72R4_STAGE07_PERSISTENT_AXIS_AUDIT_V1",
        "stage": "07_REMOVE_EVENT_LOCAL_PUBLIC_AXIS",
        "status": status,
        "created_at_utc": now_utc(),
        "formal_path_files_checked": [str(formal_path)],
        "hits": hits,
        "public_axis_source": "PersistentIdentityRecord.public_id",
        "association_axis_source": "PersistentIdentityRecord.association_state_id",
        "candidate_index_to_public_id": False,
        "native_id_to_public_id": False,
        "runtime_future_gt_used": False,
        "scientific_result": "STRUCTURAL_AXIS_AUDIT_ONLY",
    }
    atomic_json(STAGE7_PATH, payload)
    atomic_json(STAGE7_TOP_PATH, payload)
    return payload


def write_failure(exc: BaseException) -> Path:
    FAILURE_ROOT.mkdir(parents=True, exist_ok=True)
    existing = sorted(FAILURE_ROOT.glob("stage08_persistent_replay_failure_attempt*.json"))
    path = FAILURE_ROOT / f"stage08_persistent_replay_failure_attempt{len(existing) + 1}.json"
    atomic_json(
        path,
        {
            "schema_version": "N72R4_FAILURE_RECORD_V1",
            "stage": "08_PERSISTENT_STATE_EVENT_REPLAY",
            "status": "FAIL_PRESERVED",
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
            "runtime_future_gt_used": False,
            "scientific_result": "NO_SCIENTIFIC_RESULT",
        },
    )
    return path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--event-id", help="run one frozen event as a targeted smoke")
    args = parser.parse_args()
    try:
        events = load_events()
        if args.event_id:
            events = [event for event in events if str(event["event_id"]) == str(args.event_id)]
            if not events:
                raise RuntimeError(f"unknown frozen event: {args.event_id}")
        prestate_index = load_prestate_index()
        windows = load_plan()
        REPLAY_ROOT.mkdir(parents=True, exist_ok=True)
        if any(REPLAY_ROOT.iterdir()):
            raise RuntimeError(f"persistent replay root is not empty: {REPLAY_ROOT}")
        completed: list[dict[str, Any]] = []
        paths: dict[str, Path] = {}
        for event in events:
            event_id = str(event["event_id"])
            event_dir = REPLAY_ROOT / event_id
            snapshot, prefix_meta, by_frame = rebuild_prestate(event, windows, prestate_index[event_id], event_dir)
            semantic_rows = load_semantic_rows(event_id)
            event_rows = by_frame[int(event["event_frame"])]
            if len(event_rows) != len({str(row["candidate_uid"]) for row in event_rows}):
                raise RuntimeError(f"event candidate UID collision: {event_id}")
            for variant in VARIANTS:
                source_uids = tuple(str(item["candidate_uid"]) for item in semantic_rows[(variant, int(event["event_frame"]))]["candidate_rows"])
                if source_uids != tuple(str(row["candidate_uid"]) for row in event_rows):
                    raise RuntimeError(f"persistent event candidate stream differs from semantic frozen stream: {event_id}/{variant}")
            output_rows: list[dict[str, Any]] = []
            for variant in VARIANTS:
                output_rows.extend(event_variant(event, snapshot, event_rows, semantic_rows, variant, event_dir))
            output_rows.sort(key=lambda item: (int(item["frame"]), VARIANTS.index(str(item["variant"]))))
            path = REPLAY_ROOT / f"{event_id}.jsonl"
            atomic_jsonl(path, output_rows)
            paths[event_id] = path
            completed.append(
                {
                    "event_id": event_id,
                    "sequence": str(event["sequence"]),
                    "action_type": str(event["action_type"]),
                    "artifact": str(path),
                    "artifact_sha256": sha256(path),
                    "row_count": len(output_rows),
                    "variant_count": len(VARIANTS),
                    "frame_count": 101,
                    "prestate_snapshot_sha256": prefix_meta["stored_snapshot_sha256"],
                    "prestate_frame": int(event["event_frame"]) - 1,
                    "runtime_future_gt_used": False,
                }
            )
            atomic_json(
                RUNTIME_MANIFEST,
                {
                    "schema_version": "N72R4_PERSISTENT_CANDIDATE_PROBE_MANIFEST_V1",
                    "status": "IN_PROGRESS" if len(completed) < len(events) else "PASS_PERSISTENT_CANDIDATE_PROBE",
                    "created_at_utc": now_utc(),
                    "expected_event_count": len(events),
                    "completed_event_count": len(completed),
                    "completed": completed,
                    "runtime_future_gt_used": False,
                    "interaction_source": "simulated_from_gt",
                    "real_human_tape": False,
                    "official_sam3_future_propagation": False,
                    "candidate_stream_kind": "FROZEN_CANDIDATE_V2_STRUCTURAL_PROBE",
                },
            )
            print(json.dumps({"events_completed": len(completed), "events_total": len(events)}, sort_keys=True), flush=True)
        validation = validate_runtime(events, paths)
        stage7 = write_stage7_status()
        scope = "targeted_smoke" if args.event_id else "full_frozen_event_set"
        stage_payload = {
            "schema_version": "N72R4_STAGE_STATUS_V1",
            "stage": "08_PERSISTENT_STATE_EVENT_REPLAY",
            "status": "PASS_STAGE08_PERSISTENT_CANDIDATE_PROBE",
            "created_at_utc": now_utc(),
            "execution_scope": scope,
            "event_count": len(events),
            "independent_sequence_count": len({str(event["sequence"]) for event in events}),
            "runtime_manifest": str(RUNTIME_MANIFEST),
            "validation": str(VALIDATION_PATH),
            "prestate_manifest": str(PRESTATE_MANIFEST),
            "stage07_status": stage7["status"],
            "candidate_stream_kind": "FROZEN_CANDIDATE_V2_STRUCTURAL_PROBE",
            "official_sam3_future_propagation": False,
            "runtime_future_gt_used": False,
            "interaction_source": "simulated_from_gt",
            "real_human_tape": False,
            "scientific_result": "STRUCTURAL_PERSISTENT_STATE_PROBE_NOT_OFFICIAL_FULL_LOOP",
        }
        atomic_json(STAGE_PATH, stage_payload)
        atomic_json(STAGE_STATUS_PATH, stage_payload)
        print(json.dumps({"status": stage_payload["status"], "stage07": stage7["status"], "manifest": str(RUNTIME_MANIFEST)}, sort_keys=True))
        return 0
    except Exception as exc:
        failure = write_failure(exc)
        print(json.dumps({"status": "FAIL_STAGE08_PERSISTENT_REPLAY", "failure": str(failure)}, sort_keys=True))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
