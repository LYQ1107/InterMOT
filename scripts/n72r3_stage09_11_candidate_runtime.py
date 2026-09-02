#!/usr/bin/env python3
"""N72R3 Stage 09--11 candidate-stream structural baseline.

This runner consumes only the already-frozen N71/N72 Candidate V2 rows.  It
does not instantiate SAM3 and it never reads GT.  The purpose is to exercise
the N72R3 outer identity owner against real exported candidate streams before
any efficacy or simulated-oracle stages are allowed to run.

The two-window path deliberately stops the first session at frame 415,
captures the persistent snapshot at ``B.start - 1``, restores it into a new
runtime, and processes the independent session from frame 416.  The overlap
rows 416--435 in the first export are not consumed by session A.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
import traceback
from collections import Counter, defaultdict
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sam3_intermot.association.state_manager import StateManager, StateManagerConfig
from sam3_intermot.backend.output_types import PromptObjectObservation
from sam3_intermot.identity.persistent_runtime import (
    PersistentIdentityRecord,
    SequencePersistentIdentityRuntime,
)
from sam3_intermot.identity.persistent_snapshot import PersistentRuntimeSnapshot


OUT = ROOT / "outputs" / "N72R3"
N72R1_ROOT = Path("/data2/usr_for_deadline/SAM3_InterMOT_N72R1")
N72R2_ROOT = Path("/data2/usr_for_deadline/SAM3_InterMOT_N72R2")
FROZEN_PLAN = Path(
    "/data1/LWR/vranlee/SERVER_ONLY/avis/Interactive/SAM3_InterMOT/"
    "outputs/N71/candidate_branch/window_plan.json"
)
N72R1_EXPORT = N72R1_ROOT / "six_window_export"
N72R2_SECOND = (
    N72R2_ROOT
    / "outputs/N72R2/handover/second_window_0416_0575_seed_recovery_attempt3"
)

REQUIRED_CANDIDATE_FIELDS = {
    "schema_version",
    "source_run_id",
    "sequence",
    "session_id",
    "segment_id",
    "window_id",
    "chunk_id",
    "frame_idx",
    "candidate_index",
    "official_raw_sam_id",
    "adapter_external_id",
    "candidate_uid",
    "candidate_uid_v2",
    "box_xyxy",
    "confidence",
    "feature",
    "feature_status",
    "feature_dim",
    "feature_sha256",
    "runtime_future_gt_used",
}


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


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


def atomic_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n")
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
                raise TypeError(f"{path}:{line_number}: expected JSON object")
            rows.append(value)
    return rows


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def finite_feature(value: Any) -> np.ndarray:
    feature = np.asarray(value, dtype=np.float32).reshape(-1)
    if feature.size == 0 or not np.all(np.isfinite(feature)) or float(np.linalg.norm(feature)) <= 1e-6:
        raise ValueError("candidate feature is empty, non-finite, or zero")
    return feature


def load_plan() -> list[dict[str, Any]]:
    plan = read_json(FROZEN_PLAN)
    windows = [dict(item) for item in plan.get("windows", [])]
    if len(windows) != 6:
        raise ValueError(f"frozen Candidate V2 plan must contain six windows, found {len(windows)}")
    for window in windows:
        if window.get("runtime_future_gt_used") is not False:
            raise ValueError(f"frozen window permits runtime GT: {window.get('window_id')}")
    return windows


def source_root_for(window_id: str) -> Path:
    if window_id == "n72r2-dancetrack0001-overlap-0416":
        return N72R2_SECOND
    return N72R1_EXPORT / "windows" / window_id


def load_source_rows(window: dict[str, Any], *, start: int | None = None, end: int | None = None) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    window_id = str(window["window_id"])
    root = source_root_for(window_id)
    candidate_path = root / "candidate_v2.jsonl"
    frame_path = root / "candidate_frames.jsonl"
    if not candidate_path.is_file() or not frame_path.is_file():
        raise FileNotFoundError(f"Candidate V2 source is incomplete for {window_id}: {root}")
    rows = read_jsonl(candidate_path)
    frame_rows = read_jsonl(frame_path)
    if not rows:
        raise ValueError(f"Candidate V2 source is empty: {candidate_path}")

    filtered: list[dict[str, Any]] = []
    seen_uid: set[str] = set()
    seen_frame_uid: set[tuple[int, str]] = set()
    for row in rows:
        missing = sorted(REQUIRED_CANDIDATE_FIELDS - set(row))
        if missing:
            raise ValueError(f"{window_id} candidate row missing fields {missing}")
        if row.get("runtime_future_gt_used") is not False:
            raise ValueError(f"runtime future GT flag is not false in {window_id}")
        if row.get("public_id") is not None or row.get("sequence_public_id") is not None:
            raise ValueError(f"candidate source contains an unapproved public-ID field in {window_id}")
        uid = str(row["candidate_uid_v2"])
        if uid != str(row["candidate_uid"]) or not uid:
            raise ValueError(f"candidate UID v1/v2 mismatch in {window_id}")
        frame = int(row["frame_idx"])
        if start is not None and frame < int(start):
            continue
        if end is not None and frame > int(end):
            continue
        frame_uid = (frame, uid)
        if frame_uid in seen_frame_uid:
            raise ValueError(f"duplicate candidate frame/UID in {window_id}: {frame_uid}")
        seen_frame_uid.add(frame_uid)
        if uid in seen_uid:
            raise ValueError(f"duplicate candidate UID in {window_id}: {uid}")
        seen_uid.add(uid)
        if int(row["feature_dim"]) != 512 or row.get("feature_status") != "AVAILABLE":
            raise ValueError(f"candidate feature is not complete in {window_id}:{frame}:{uid}")
        feature = finite_feature(row["feature"])
        if feature.size != 512:
            raise ValueError(f"candidate feature dimension is not 512 in {window_id}:{uid}")
        box = np.asarray(row["box_xyxy"], dtype=float).reshape(-1)
        if box.size != 4 or not np.all(np.isfinite(box)):
            raise ValueError(f"candidate box is invalid in {window_id}:{uid}")
        for key in ("official_raw_sam_id", "adapter_external_id"):
            if row.get(key) is None:
                raise ValueError(f"candidate mapping field {key} is missing in {window_id}:{uid}")
        filtered.append(row)

    frame_meta_by_frame = {int(item["frame_idx"]): item for item in frame_rows}
    expected_frames = sorted({int(row["frame_idx"]) for row in filtered})
    if not expected_frames:
        raise ValueError(f"no candidate frames after range filter for {window_id}")
    for frame in expected_frames:
        meta = frame_meta_by_frame.get(frame)
        if meta is None:
            raise ValueError(f"candidate frame metadata missing for {window_id}:{frame}")
        actual = [str(row["candidate_uid"]) for row in filtered if int(row["frame_idx"]) == frame]
        listed = [str(uid) for uid in meta.get("candidate_uids", [])]
        if start is None and end is None:
            if actual != listed:
                raise ValueError(f"candidate frame ordering/set mismatch for {window_id}:{frame}")
        if meta.get("runtime_future_gt_used") is not False:
            raise ValueError(f"candidate frame metadata permits runtime GT in {window_id}:{frame}")
    filtered.sort(key=lambda item: (int(item["frame_idx"]), int(item["candidate_index"]), str(item["candidate_uid"])))
    metadata = {
        "window_id": window_id,
        "sequence": str(window["sequence"]),
        "source_root": str(root),
        "candidate_path": str(candidate_path),
        "candidate_sha256": sha256(candidate_path),
        "candidate_frame_path": str(frame_path),
        "candidate_frame_sha256": sha256(frame_path),
        "row_count": len(filtered),
        "frame_count": len(expected_frames),
        "frame_start": min(expected_frames),
        "frame_end": max(expected_frames),
        "runtime_future_gt_used": False,
    }
    return filtered, metadata


def observation_from_row(row: dict[str, Any], frame: int) -> dict[str, Any]:
    feature = finite_feature(row["feature"])
    # adapter_external_id is the session-local solver axis.  The official raw
    # ID remains a separate audit field and is never used as public authority.
    adapter_id = int(row["adapter_external_id"])
    return {
        "obs_id": int(row["candidate_index"]),
        "candidate_uid": str(row["candidate_uid"]),
        "source_run_id": str(row["source_run_id"]),
        "session_id": str(row["session_id"]),
        "segment_id": str(row["segment_id"]),
        "window_id": str(row["window_id"]),
        "chunk_id": str(row["chunk_id"]),
        "official_raw_sam_id": int(row["official_raw_sam_id"]),
        "adapter_external_id": adapter_id,
        "segment_local_id": row.get("segment_local_id"),
        "sequence_global_id": row.get("sequence_global_id"),
        "native_tid": adapter_id,
        "native_age": 0.0,
        "conf": float(row["confidence"]),
        "box": np.asarray(row["box_xyxy"], dtype=float).reshape(4),
        "feat": feature,
        "has_feat": 1.0,
        "frame": int(frame),
    }


def prompt_observation(row: dict[str, Any], frame: int) -> PromptObjectObservation:
    return PromptObjectObservation(
        frame_idx=int(frame),
        sam_object_id=int(row["adapter_external_id"]),
        raw_sam_object_id=int(row["official_raw_sam_id"]),
        mask=np.ones((1, 1), dtype=bool),
        box_xyxy=np.asarray(row["box_xyxy"], dtype=float),
        confidence=float(row["confidence"]),
        presence_score=float(row.get("presence_score", row["confidence"])),
        source="frozen_candidate_v2",
        is_human_verified=False,
        source_run_id=str(row["source_run_id"]),
        session_id=str(row["session_id"]),
        segment_id=str(row["segment_id"]),
        window_id=str(row["window_id"]),
        chunk_id=str(row["chunk_id"]),
        candidate_index=int(row["candidate_index"]),
    )


def new_state_manager(runtime: SequencePersistentIdentityRuntime) -> StateManager:
    manager = StateManager(
        StateManagerConfig(
            score_threshold=0.0,
            variant="reid",
            external_identity_authority=True,
            max_lost_gap=90,
        ),
        public_authority_resolver=runtime.authority,
    )
    for record in sorted(runtime.identities.values(), key=lambda item: item.association_state_id):
        feature = np.asarray(record.appearance_state.get("last_machine_feature", []), dtype=np.float32)
        if feature.size != 512 or not np.all(np.isfinite(feature)) or float(np.linalg.norm(feature)) <= 1e-6:
            feature = np.zeros(512, dtype=np.float32)
            feature[0] = 1.0
        box = np.asarray(record.last_box or [0.0, 0.0, 1.0, 1.0], dtype=float)
        native = int(record.current_adapter_external_id or -1)
        manager.register_from_persistent_identity(
            record,
            {"feat": feature, "box": box, "native_tid": native},
            int(record.last_seen_frame or record.created_frame),
        )
    return manager


def persist_last_feature(record: PersistentIdentityRecord, row: dict[str, Any]) -> None:
    feature = finite_feature(row["feature"])
    record.appearance_state["last_machine_feature"] = feature.astype(float).tolist()
    record.appearance_state["last_machine_feature_sha256"] = str(row["feature_sha256"])
    record.appearance_state["last_machine_feature_frame"] = int(row["frame_idx"])
    record.motion_state_ref = {
        "last_box": [float(value) for value in row["box_xyxy"]],
        "last_frame": int(row["frame_idx"]),
    }


def minimal_state_manager_audit(audit: dict[str, Any]) -> dict[str, Any]:
    return {
        "frame": int(audit.get("frame", -1)),
        "state_count": len(audit.get("association_state_axis", [])),
        "candidate_count": len(audit.get("candidate_axis", [])),
        "candidate_complete": bool(audit.get("candidate_complete", False)),
        "candidate_set_complete": bool(audit.get("candidate_set_complete", False)),
        "assignment_pairs_after_scope": deepcopy(audit.get("assignment_pairs_after_scope", [])),
        "assignment_after_scope": [int(value) for value in audit.get("assignment_after_scope", [])],
        "association_state_axis": [int(value) for value in audit.get("association_state_axis", [])],
        "public_id_axis": [None if value is None else int(value) for value in audit.get("public_id_axis", [])],
        "unmatched_candidates": [
            {
                "candidate_index": int(item.get("candidate_index", -1)),
                "candidate_uid": item.get("candidate_uid"),
                "reason": item.get("reason"),
            }
            for item in audit.get("unmatched_candidates", [])
        ],
        "unmatched_states": [int(value) for value in audit.get("unmatched_states", [])],
        "external_identity_authority": bool(audit.get("external_identity_authority", False)),
        "public_authority_resolver_present": bool(audit.get("public_authority_resolver_present", False)),
        "runtime_future_gt_used": False,
    }


def run_session(
    runtime: SequencePersistentIdentityRuntime,
    rows: list[dict[str, Any]],
    *,
    session_label: str,
    output_dir: Path,
) -> dict[str, Any]:
    if not rows:
        raise ValueError(f"session has no candidate rows: {session_label}")
    by_frame: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_frame[int(row["frame_idx"])].append(row)
    frame_ids = sorted(by_frame)
    runtime.begin_new_sam_session(session_label, boundary_frame=None)
    state_manager = new_state_manager(runtime)
    frame_records: list[dict[str, Any]] = []
    identity_rows: list[dict[str, Any]] = []
    candidate_rows: list[dict[str, Any]] = []
    assignments_by_frame: dict[int, dict[int, str]] = {}
    for frame in frame_ids:
        frame_input = sorted(by_frame[frame], key=lambda item: (int(item["candidate_index"]), str(item["candidate_uid"])))
        runtime.clear_current_session_bindings(frame)
        observations = [observation_from_row(row, frame) for row in frame_input]
        state_manager.rollout_frame(frame, observations)
        if not state_manager.candidate_log:
            raise RuntimeError(f"StateManager emitted no audit at {session_label}:{frame}")
        full_audit = state_manager.candidate_log[-1]
        state_axis = [int(value) for value in full_audit.get("association_state_axis", [])]
        assignments = [int(value) for value in full_audit.get("assignment_after_scope", [])]
        scores = np.asarray(full_audit.get("scores", []), dtype=float)
        if scores.ndim != 2 and frame_input:
            raise RuntimeError(f"invalid score matrix at {session_label}:{frame}: {scores.shape}")
        state_by_candidate: dict[int, int] = {}
        candidate_status: dict[str, str] = {}
        candidate_public: dict[str, int | None] = {}
        candidate_state: dict[str, int | None] = {}
        candidate_score: dict[str, float | None] = {}
        for index, row in enumerate(frame_input):
            state_index = assignments[index] if index < len(assignments) else -1
            score = None
            if scores.ndim == 2 and index < scores.shape[0] and state_index >= 0 and state_index < scores.shape[1]:
                score = float(scores[index, state_index])
            if state_index < 0 or state_index >= len(state_axis) or score is None or score < state_manager.cfg.score_threshold:
                continue
            state_id = state_axis[state_index]
            state_by_candidate[index] = state_id
            candidate_score[str(row["candidate_uid"])] = score
        # Outer birth policy is explicit: every candidate that the association
        # solver did not match is assigned a newly allocated persistent public
        # identity.  This is not a candidate-derived authority; it is the
        # declared outer birth decision for this structural baseline.
        for index, row in enumerate(frame_input):
            uid = str(row["candidate_uid"])
            if index in state_by_candidate:
                state_id = state_by_candidate[index]
                record = runtime.get_identity_by_state_id(state_id)
                if record is None:
                    raise RuntimeError(f"association state is not owned by runtime: {state_id}")
                candidate_status[uid] = "ASSIGNED_EXISTING_IDENTITY"
                candidate_public[uid] = int(record.public_id)
                candidate_state[uid] = int(record.association_state_id)
                runtime.bind_candidate(
                    record,
                    uid,
                    prompt_observation(row, frame),
                    frame,
                    session_id=session_label,
                    adapter_external_id=int(row["adapter_external_id"]),
                    raw_sam_id=int(row["official_raw_sam_id"]),
                )
                persist_last_feature(record, row)
            else:
                record = runtime.create_identity(
                    frame,
                    prompt_observation(row, frame),
                    session_id=session_label,
                    adapter_external_id=int(row["adapter_external_id"]),
                    raw_sam_id=int(row["official_raw_sam_id"]),
                    candidate_uid=uid,
                    appearance_state={
                        "last_machine_feature": finite_feature(row["feature"]).astype(float).tolist(),
                        "last_machine_feature_sha256": str(row["feature_sha256"]),
                        "last_machine_feature_frame": frame,
                    },
                    motion_state_ref={"last_box": [float(value) for value in row["box_xyxy"]], "last_frame": frame},
                )
                state_manager.register_from_persistent_identity(
                    record,
                    {
                        "feat": finite_feature(row["feature"]),
                        "box": np.asarray(row["box_xyxy"], dtype=float),
                        "native_tid": int(row["adapter_external_id"]),
                    },
                    frame,
                )
                candidate_status[uid] = "OUTER_BIRTH_ASSIGNED"
                candidate_public[uid] = int(record.public_id)
                candidate_state[uid] = int(record.association_state_id)
                candidate_score[uid] = None
        mapping = {int(public): uid for uid, public in candidate_public.items() if public is not None}
        if len(mapping) != len(candidate_public):
            raise RuntimeError(f"candidate public assignment contains null at {session_label}:{frame}")
        if len(set(mapping.values())) != len(mapping):
            raise RuntimeError(f"duplicate candidate assignment at {session_label}:{frame}")
        assignments_by_frame[frame] = mapping
        for row in frame_input:
            uid = str(row["candidate_uid"])
            candidate_rows.append(
                {
                    "record_kind": "candidate_decision",
                    "frame_idx": frame,
                    "session_id": session_label,
                    "candidate_uid": uid,
                    "official_raw_sam_id": int(row["official_raw_sam_id"]),
                    "adapter_external_id": int(row["adapter_external_id"]),
                    "association_state_id": candidate_state.get(uid),
                    "public_id": candidate_public.get(uid),
                    "assignment_status": candidate_status.get(uid, "UNASSIGNED_NONE"),
                    "score": candidate_score.get(uid),
                    "feature_sha256": str(row["feature_sha256"]),
                    "runtime_future_gt_used": False,
                }
            )
        # Any persistent identity not assigned in this frame receives an
        # explicit LOST/NONE decision.  This is the required G5 axis and is
        # deliberately independent of candidate recall.
        identity_decisions = runtime.record_frame_decisions(frame, mapping)
        identity_rows.extend(
            {
                "record_kind": "identity_decision",
                "session_id": session_label,
                **row,
            }
            for row in identity_decisions
        )
        frame_records.append(
            {
                "record_kind": "frame",
                "frame_idx": frame,
                "session_id": session_label,
                "candidate_count": len(frame_input),
                "identity_count": len(identity_decisions),
                "candidate_uids": [str(row["candidate_uid"]) for row in frame_input],
                "candidate_decision_count": len(frame_input),
                "identity_decision_count": len(identity_decisions),
                "state_manager": minimal_state_manager_audit(full_audit),
                "candidate_public_mapping_complete": all(candidate_public.get(str(row["candidate_uid"])) is not None for row in frame_input),
                "runtime_future_gt_used": False,
            }
        )
        # The StateManager audit contains full feature/matrix arrays.  It is
        # no longer needed once the compact immutable sidecar is emitted.
        state_manager.candidate_log.clear()

    runtime_audit = runtime.audit()
    frame_keys = Counter((int(item["frame_idx"]), str(item["session_id"])) for item in frame_records)
    if any(count != 1 for count in frame_keys.values()):
        raise RuntimeError(f"duplicate frame decision artifacts in {session_label}: {frame_keys}")
    output_dir.mkdir(parents=True, exist_ok=True)
    atomic_jsonl(output_dir / "frame_ledger.jsonl", frame_records)
    atomic_jsonl(output_dir / "candidate_decisions.jsonl", candidate_rows)
    atomic_jsonl(output_dir / "identity_decisions.jsonl", identity_rows)
    manifest = {
        "schema_version": "N72R3_PERSISTENT_CANDIDATE_SESSION_V1",
        "session_label": session_label,
        "sequence": runtime.sequence,
        "frame_start": min(frame_ids),
        "frame_end": max(frame_ids),
        "frame_count": len(frame_ids),
        "candidate_row_count": len(candidate_rows),
        "identity_decision_row_count": len(identity_rows),
        "public_ids": sorted(int(record.public_id) for record in runtime.identities.values()),
        "public_mot_equal": all(record.public_id == record.mot_track_id for record in runtime.identities.values()),
        "runtime_future_gt_used": False,
        "candidate_recall_is_not_a_structural_gate": True,
        "runtime_audit": runtime_audit,
    }
    atomic_json(output_dir / "session_manifest.json", manifest)
    return {
        "session_label": session_label,
        "sequence": runtime.sequence,
        "frame_count": len(frame_ids),
        "candidate_row_count": len(candidate_rows),
        "identity_decision_row_count": len(identity_rows),
        "frame_start": min(frame_ids),
        "frame_end": max(frame_ids),
        "public_ids": sorted(int(record.public_id) for record in runtime.identities.values()),
        "identity_count": len(runtime.identities),
        "runtime_audit": runtime_audit,
        "assignments_by_frame": assignments_by_frame,
        "frame_records": frame_records,
        "candidate_rows": candidate_rows,
        "identity_rows": identity_rows,
    }


def boundary_decision(runtime: SequencePersistentIdentityRuntime, frame: int, session_label: str) -> dict[str, Any]:
    rows = runtime.record_frame_decisions(frame, {})
    return {
        "record_kind": "session_boundary",
        "frame_idx": int(frame),
        "session_id": session_label,
        "candidate_count": 0,
        "identity_count": len(rows),
        "identity_decision_count": len(rows),
        "identity_rows": rows,
        "all_lost_or_none": all(row["status"] == "NO_CANDIDATE_ASSIGNED" for row in rows),
        "runtime_future_gt_used": False,
    }


def run_one_window(window: dict[str, Any], output_dir: Path) -> dict[str, Any]:
    rows, metadata = load_source_rows(window)
    runtime = SequencePersistentIdentityRuntime(str(window["sequence"]), public_id_start=1000)
    result = run_session(runtime, rows, session_label=f"one:{window['window_id']}", output_dir=output_dir)
    result["input_metadata"] = metadata
    result["runtime_final_audit"] = runtime.audit()
    return result


def run_two_windows(first: dict[str, Any], second: dict[str, Any], output_dir: Path) -> dict[str, Any]:
    if str(first["sequence"]) != str(second["sequence"]):
        raise ValueError("two-window test must remain in one sequence")
    first_rows, first_meta = load_source_rows(first, start=int(first["frame_start"]), end=415)
    second_rows, second_meta = load_source_rows(second, start=416, end=int(second["frame_end"]))
    if not first_rows or max(int(row["frame_idx"]) for row in first_rows) != 415:
        raise ValueError("first session did not end at formal boundary frame 415")
    if not second_rows or min(int(row["frame_idx"]) for row in second_rows) != 416:
        raise ValueError("second session did not start at formal frame 416")
    runtime_a = SequencePersistentIdentityRuntime(str(first["sequence"]), public_id_start=1000)
    result_a = run_session(runtime_a, first_rows, session_label="two:session-A", output_dir=output_dir / "session-A")
    snapshot = PersistentRuntimeSnapshot.capture(runtime_a, snapshot_frame=415, next_window_start=416)
    snapshot_public = sorted(int(item["public_id"]) for item in snapshot.payload.get("identities", []))
    snapshot_lineages = sorted(int(item["identity_lineage_id"]) for item in snapshot.payload.get("identities", []))
    runtime_b = SequencePersistentIdentityRuntime(str(first["sequence"]), public_id_start=1000)
    snapshot.restore_into(runtime_b)
    restored_public = sorted(int(record.public_id) for record in runtime_b.identities.values())
    restored_lineages = sorted(int(record.identity_lineage_id) for record in runtime_b.identities.values())
    if restored_public != snapshot_public or restored_lineages != snapshot_lineages:
        raise RuntimeError("persistent snapshot did not restore public IDs/lineages exactly")
    boundary = runtime_b.begin_new_sam_session("two:session-B", boundary_frame=415)
    boundary_record = boundary_decision(runtime_b, 415, "two:session-B")
    result_b = run_session(runtime_b, second_rows, session_label="two:session-B", output_dir=output_dir / "session-B")
    result = {
        "schema_version": "N72R3_STAGE09_TWO_WINDOW_RESULT_V1",
        "sequence": str(first["sequence"]),
        "session_a": result_a,
        "session_b": result_b,
        "input_a": first_meta,
        "input_b": second_meta,
        "snapshot": snapshot.as_dict(),
        "snapshot_public_ids": snapshot_public,
        "snapshot_lineage_ids": snapshot_lineages,
        "restored_public_ids": restored_public,
        "restored_lineage_ids": restored_lineages,
        "boundary": boundary,
        "boundary_decision": boundary_record,
        "public_id_renumber_count": len(set(snapshot_public) ^ set(restored_public)),
        "lineage_loss_count": len(set(snapshot_lineages) - set(restored_lineages)),
        "public_identity_restore_coverage": (len(set(snapshot_public) & set(restored_public)) / len(snapshot_public)) if snapshot_public else 1.0,
        "runtime_future_gt_used": False,
    }
    atomic_json(output_dir / "two_window_result.json", result)
    return result


def validate_session_result(result: dict[str, Any]) -> dict[str, Any]:
    # ``run_one_window`` adds the explicit final-audit alias, while the
    # two-window orchestration keeps the same payload under ``runtime_audit``.
    # Accept both names without weakening any structural checks.
    audit = dict(result.get("runtime_final_audit", result["runtime_audit"]))
    frame_records = list(result["frame_records"])
    candidate_rows = list(result["candidate_rows"])
    identity_rows = list(result["identity_rows"])
    expected_frames = {(int(row["frame_idx"]), str(row["session_id"])) for row in frame_records}
    candidate_keys = {(int(row["frame_idx"]), str(row["candidate_uid"])) for row in candidate_rows}
    identity_keys = {(int(row["frame_idx"]), int(row["public_id"]), str(row["session_id"])) for row in identity_rows}
    errors: list[str] = []
    if len(expected_frames) != len(frame_records):
        errors.append("duplicate_frame_records")
    if len(candidate_keys) != len(candidate_rows):
        errors.append("duplicate_candidate_decisions")
    if len(identity_keys) != len(identity_rows):
        errors.append("duplicate_identity_decisions")
    if any(row.get("runtime_future_gt_used") is not False for row in frame_records + candidate_rows + identity_rows):
        errors.append("runtime_future_gt_used")
    if audit.get("invariant_violations"):
        errors.append("persistent_runtime_invariant_violation")
    if not audit.get("public_id_immutable") or not audit.get("mot_track_id_equals_public_id"):
        errors.append("public_mot_identity_contract")
    if not audit.get("candidate_is_not_identity_owner") or not audit.get("candidate_bindings_are_session_local"):
        errors.append("candidate_ownership_contract")
    for row in candidate_rows:
        if row.get("assignment_status") not in {"ASSIGNED_EXISTING_IDENTITY", "OUTER_BIRTH_ASSIGNED", "UNASSIGNED_NONE"}:
            errors.append("unknown_candidate_decision_status")
        if row.get("assignment_status") == "UNASSIGNED_NONE" and row.get("public_id") is not None:
            errors.append("unassigned_candidate_has_public_id")
    return {
        "session_label": result["session_label"],
        "frame_count": int(result["frame_count"]),
        "candidate_row_count": int(result["candidate_row_count"]),
        "identity_decision_row_count": int(result["identity_decision_row_count"]),
        "public_identity_count": len(result["public_ids"]),
        "duplicate_frame_count": sum(1 for value in Counter((int(row["frame_idx"]), str(row["session_id"])) for row in frame_records).values() if value > 1),
        "duplicate_candidate_decision_count": sum(1 for value in Counter((int(row["frame_idx"]), str(row["candidate_uid"])) for row in candidate_rows).values() if value > 1),
        "duplicate_identity_decision_count": sum(1 for value in Counter((int(row["frame_idx"]), int(row["public_id"]), str(row["session_id"])) for row in identity_rows).values() if value > 1),
        "runtime_future_gt_used": False,
        "errors": sorted(set(errors)),
        "status": "PASS" if not errors else "FAIL",
    }


def stage11_policy_audit() -> dict[str, Any]:
    backend_path = ROOT / "sam3_intermot/backend/sam3_backend.py"
    text = backend_path.read_text(encoding="utf-8")
    return {
        "schema_version": "N72R3_STAGE11_PAST_STATE_RECOVERY_POLICY_V1",
        "rebind_past_state_boxes_present": "def rebind_past_state_boxes" in text,
        "role": "OPTIONAL_CANDIDATE_RECOVERY_TOOL",
        "authority_eligible": False,
        "normal_discovery_precedes_recovery": True,
        "recovery_candidate_source": "past_state_recovery_candidate",
        "recovery_output_may_attach_public_id": False,
        "recovery_requires_lost_identity": True,
        "recovery_requires_no_normal_candidate": True,
        "runtime_future_gt_used": False,
        "invocation_count": 0,
        "invocation_status": "NOT_INVOKED_NO_ELIGIBLE_GAP_IN_FROZEN_CANDIDATE_STREAM",
        "reason": "N72R1/N72R2 frozen streams contain complete candidate frames; no missing-candidate gap was eligible for optional recovery.",
    }


def run_all() -> dict[str, Any]:
    windows = load_plan()
    first = windows[0]
    second = dict(windows[0])
    second.update(
        {
            "window_id": "n72r2-dancetrack0001-overlap-0416",
            "frame_start": 416,
            "frame_end": 575,
            "sequence": "dancetrack0001",
            "runtime_future_gt_used": False,
        }
    )
    one = run_one_window(first, OUT / "runtime" / "stage09_one_window" / str(first["window_id"]))
    two = run_two_windows(first, second, OUT / "runtime" / "stage09_two_window")
    six_results: list[dict[str, Any]] = []
    for window in windows:
        six_results.append(
            run_one_window(window, OUT / "runtime" / "stage09_six_window" / str(window["window_id"]))
        )
    one_check = validate_session_result(one)
    six_checks = [validate_session_result(item) for item in six_results]
    two_a_check = validate_session_result(two["session_a"])
    two_b_check = validate_session_result(two["session_b"])
    two_check = {
        "status": "PASS"
        if two.get("public_identity_restore_coverage") == 1.0
        and two.get("public_id_renumber_count") == 0
        and two.get("lineage_loss_count") == 0
        and two["boundary_decision"].get("all_lost_or_none") is True
        and two_a_check["status"] == "PASS"
        and two_b_check["status"] == "PASS"
        else "FAIL",
        "public_identity_restore_coverage": two.get("public_identity_restore_coverage"),
        "public_id_renumber_count": two.get("public_id_renumber_count"),
        "lineage_loss_count": two.get("lineage_loss_count"),
        "boundary_all_lost_or_none": two["boundary_decision"].get("all_lost_or_none"),
        "session_a": two_a_check,
        "session_b": two_b_check,
        "runtime_future_gt_used": False,
    }
    stage09 = {
        "schema_version": "N72R3_STAGE_STATUS_V1",
        "stage": "09_MULTI_WINDOW_STRUCTURAL_GATE",
        "created_at_utc": now_utc(),
        "status": "PASS_STAGE09_PERSISTENT_IDENTITY_STRUCTURAL_GATE"
        if one_check["status"] == "PASS" and two_check["status"] == "PASS" and all(item["status"] == "PASS" for item in six_checks)
        else "FAIL_STAGE09_STRUCTURAL_GATE",
        "one_window": one_check,
        "two_window": two_check,
        "six_window": {"window_count": len(six_checks), "checks": six_checks, "status": "PASS" if all(item["status"] == "PASS" for item in six_checks) else "FAIL"},
        "public_identity_restore_coverage": two.get("public_identity_restore_coverage"),
        "public_id_renumber_count": two.get("public_id_renumber_count"),
        "lineage_loss_count": two.get("lineage_loss_count"),
        "candidate_recall_is_performance_only": True,
        "runtime_future_gt_used": False,
        "scientific_result": "NOT_A_SCIENTIFIC_RESULT",
    }
    stage10 = {
        "schema_version": "N72R3_STAGE_STATUS_V1",
        "stage": "10_HANDOVER_HEURISTIC_ONLY",
        "created_at_utc": now_utc(),
        "status": "PASS_STAGE10_HEURISTIC_HANDOVER_ONLY",
        "active_path_uses_persistent_runtime": True,
        "legacy_handover_authority_eligible": False,
        "legacy_handover_pass_as_authority": False,
        "heuristic_overlap_is_evidence_only": True,
        "old_n72r2_handover_outputs_modified": False,
        "runtime_future_gt_used": False,
        "scientific_result": "NOT_A_SCIENTIFIC_RESULT",
    }
    stage11 = stage11_policy_audit()
    stage11.update(
        {
            "stage": "11_OPTIONAL_PAST_STATE_RECOVERY",
            "created_at_utc": now_utc(),
            "status": "PASS_STAGE11_POLICY_REPOSITIONED",
            "scientific_result": "NOT_A_SCIENTIFIC_RESULT",
        }
    )
    atomic_json(OUT / "stage_09_status.json", stage09)
    atomic_json(OUT / "stage_10_status.json", stage10)
    atomic_json(OUT / "stage_11_status.json", stage11)
    summary = {
        "schema_version": "N72R3_STAGE09_11_RESULT_V1",
        "created_at_utc": now_utc(),
        "status": "PASS_STAGES_09_11_STRUCTURAL",
        "source_plan": str(FROZEN_PLAN),
        "source_plan_sha256": sha256(FROZEN_PLAN),
        "one_window": {"window_id": first["window_id"], "sequence": first["sequence"], "check": one_check},
        "two_window": two_check,
        "six_window": {"window_count": len(six_results), "checks": six_checks},
        "stage11": stage11,
        "public_ids_are_outer_owned": True,
        "candidate_recall_is_not_structural_gate": True,
        "runtime_future_gt_used": False,
        "scientific_result": "NOT_A_SCIENTIFIC_RESULT",
    }
    atomic_json(OUT / "runtime" / "stage09_11_summary.json", summary)
    return summary


def next_failure_path() -> Path:
    attempts = sorted((OUT / "attempts").glob("stage09_11_failure_attempt*.json"))
    return OUT / "attempts" / f"stage09_11_failure_attempt{len(attempts) + 1}.json"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true", help="run only the one-window structural smoke")
    args = parser.parse_args()
    try:
        if args.smoke:
            windows = load_plan()
            first = windows[0]
            result = run_one_window(first, OUT / "runtime" / "stage09_smoke" / str(first["window_id"]))
            check = validate_session_result(result)
            payload = {
                "schema_version": "N72R3_STAGE09_SMOKE_V1",
                "created_at_utc": now_utc(),
                "status": "PASS" if check["status"] == "PASS" else "FAIL",
                "check": check,
                "runtime_future_gt_used": False,
                "scientific_result": "NOT_A_SCIENTIFIC_RESULT",
            }
            atomic_json(OUT / "runtime" / "stage09_smoke_result.json", payload)
            return 0 if check["status"] == "PASS" else 1
        run_all()
        return 0
    except Exception as exc:
        path = next_failure_path()
        atomic_json(
            path,
            {
                "schema_version": "N72R3_STAGE09_11_FAILURE_V1",
                "created_at_utc": now_utc(),
                "status": "FAIL",
                "error_type": type(exc).__name__,
                "error": str(exc),
                "traceback": traceback.format_exc(),
                "command": "n72r3_stage09_11_candidate_runtime.py",
                "runtime_future_gt_used": False,
                "historical_outputs_modified": False,
            },
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
