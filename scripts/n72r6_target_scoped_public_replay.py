#!/usr/bin/env python3
"""N72R6 Stage 05 C0/C1 target-scoped public-association replay.

C0 is the copied frozen N72R5R1 B0 public assignment. C1 replays the same
event from the frozen t-1 persistent snapshot, adds one independent target
candidate stream, applies a correction epoch, and submits a target-exclusive
matrix to the existing exact public solver. This runner does not load GT;
future scoring is performed by a separate posthoc program.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import traceback
from typing import Any, Mapping, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sam3_intermot.association.branch_public_replay import (  # noqa: E402
    HORIZON,
    apply_exact_frame,
    atomic_json,
    atomic_jsonl,
    candidate_obs,
    exact_solve,
    load_stage07_event_rows,
    new_runtime,
    new_state_manager,
    read_json,
    read_jsonl,
    score_existing,
    sha256_file,
)
from sam3_intermot.association.target_scoped_merge import (  # noqa: E402
    TARGET_CANDIDATE_KIND,
    apply_human_anchor_verification_gate,
    apply_target_exclusive_constraints,
    merge_main_and_target_candidates,
)
from sam3_intermot.identity.correction_epoch import apply_epoch_to_identity_state  # noqa: E402

STAGE08 = ROOT / "outputs/N72R5R1/controller/round_05_branch_isolation_v0/full/stage08_runtime_manifest.json"
STAGE07_ROOT = ROOT / "outputs/N72R5/mechanism_rounds/round_07_official_full_loop_attempt5"
PRESTATE_ROOT = ROOT / "outputs/N72R5R1/controller/round_05_branch_isolation_v0/full/event_prestate"
TARGET_STREAM_MANIFEST = ROOT / "outputs/N72R6/target_correction_stream/target_stream_manifest.json"
OUT = ROOT / "outputs/N72R6/public_replay"
HORIZON = 100


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def digest_json(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                     separators=(",", ":"), allow_nan=False).encode("utf-8")).hexdigest()


def resolve(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def finite_feature(value: Any, label: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float32).reshape(-1)
    if result.size != 512 or not np.all(np.isfinite(result)):
        raise ValueError(f"{label}: expected finite 512-D feature")
    norm = float(np.linalg.norm(result))
    if norm <= 1.0e-6:
        raise ValueError(f"{label}: zero-norm feature")
    return result / norm


def load_events() -> list[dict[str, Any]]:
    payload = read_json(STAGE08)
    result: list[dict[str, Any]] = []
    for item in payload.get("events", []):
        branches = {str(branch.get("branch")): branch for branch in item.get("branches", [])
                    if isinstance(branch, dict)}
        b1 = branches.get("B1_SPATIAL_CORRECTION_ONLY")
        b0 = branches.get("B0_NO_INTERVENTION")
        if not b1 or b1.get("action_precondition_status") != "APPLIED":
            continue
        if b0 is None or b1.get("target_public_id") is None:
            raise ValueError(f"eligible event lacks main/target authority: {item.get('event_id')}")
        event = dict(item)
        event["target_public_id"] = int(b1["target_public_id"])
        event["main_output"] = str(b0["output"])
        result.append(event)
    result.sort(key=lambda item: str(item["event_id"]))
    if len(result) != 32 or len({str(item["event_id"]) for item in result}) != 32:
        raise ValueError(f"expected exactly 32 eligible events, found {len(result)}")
    return result


def load_selected_streams(manifest_path: Path | None = None) -> dict[str, dict[str, Any]]:
    path = TARGET_STREAM_MANIFEST if manifest_path is None else manifest_path
    payload = read_json(path)
    accepted_statuses = {
        "PASS_32_OF_32_TARGET_STREAMS_VALIDATED",
        "PASS_N72R6_TARGET_SESSION_RECOVERY_32_OF_32_VALIDATED",
    }
    if payload.get("status") not in accepted_statuses:
        raise RuntimeError(f"target stream manifest is not validated: {payload.get('status')}")
    if payload.get("status") == "PASS_N72R6_TARGET_SESSION_RECOVERY_32_OF_32_VALIDATED":
        if payload.get("replay_ready") is not True or payload.get("target_session_recovery_mode") is not True:
            raise RuntimeError("recovery target stream manifest is not replay-ready")
    rows = [dict(item) for item in payload.get("selected", [])]
    if len(rows) != 32 or len({str(item["event_id"]) for item in rows}) != 32:
        raise RuntimeError("target stream selection is not 32 unique events")
    return {str(item["event_id"]): item for item in rows}


def read_target_stream(selected: Mapping[str, Any]) -> tuple[dict[str, Any], dict[int, list[dict[str, Any]]]]:
    done = read_json(resolve(str(selected["done"])))
    rows = read_jsonl(resolve(str(done["frames"])))
    if len(rows) != HORIZON + 1:
        raise ValueError(f"target stream frame count mismatch: {done['event_id']}")
    by_frame: dict[int, list[dict[str, Any]]] = {}
    for row in rows:
        frame = int(row["frame"])
        if frame in by_frame:
            raise ValueError(f"duplicate target frame: {done['event_id']}:{frame}")
        if any(row.get(flag) is not False for flag in ("runtime_future_gt_used", "runtime_gt_read", "posthoc_gt_used")):
            raise ValueError(f"target stream GT flag is not false: {done['event_id']}:{frame}")
        candidates = row.get("candidate_rows")
        if not isinstance(candidates, list) or len(candidates) > 1:
            raise ValueError(f"target stream candidate cardinality invalid: {done['event_id']}:{frame}")
        by_frame[frame] = [dict(item) for item in candidates]
    expected = set(range(int(done["event_frame"]), int(done["end_frame"]) + 1))
    if set(by_frame) != expected:
        raise ValueError(f"target stream frame range mismatch: {done['event_id']}")
    return done, by_frame


def load_main_stream(event: Mapping[str, Any]) -> tuple[dict[int, list[dict[str, Any]]], str]:
    rows_by_branch, _, paths = load_stage07_event_rows(
        STAGE07_ROOT, str(event["event_id"]), int(event["event_frame"]), str(event["sequence"])
    )
    main = rows_by_branch["B0_NO_INTERVENTION"]
    if len(main) != HORIZON + 1:
        raise ValueError(f"main stream frame count mismatch: {event['event_id']}")
    return main, str(paths["B0_NO_INTERVENTION"])


def public_semantic(row: Mapping[str, Any]) -> list[dict[str, Any]]:
    result = []
    for item in row.get("candidate_rows", []):
        result.append({
            "candidate_index": int(item["candidate_index"]),
            "official_raw_sam_id": int(item["official_raw_sam_id"]),
            "adapter_external_id": int(item["adapter_external_id"]),
            "box_xyxy": [float(value) for value in item["box_xyxy"]],
            "feature_sha256": str(item["feature_sha256"]),
            "public_id": None if item.get("public_id") is None else int(item["public_id"]),
            "assignment_status": str(item.get("assignment_status", "")),
        })
    return sorted(result, key=lambda item: (item["candidate_index"], item["official_raw_sam_id"]))


def target_candidate(row: Mapping[str, Any], event_id: str, done: Mapping[str, Any]) -> dict[str, Any]:
    if row.get("candidate_kind") != TARGET_CANDIDATE_KIND:
        raise ValueError(f"target candidate kind mismatch: {event_id}:{row.get('frame')}")
    candidate = dict(row)
    raw_sam_id = int(candidate["official_raw_sam_id"])
    candidate.update({
        "event_id": event_id,
        "target_session_scope": str(done["target_session_scope"]),
        "native_scope": str(done["target_session_scope"]),
        "native_tid_scope": str(done["target_session_scope"]),
        "native_tid": int(candidate["adapter_external_id"]),
        # TrackManager's legacy binding table is keyed by an integer and is
        # not scope-aware.  Keep the official raw/adapter/native IDs intact,
        # but use a deterministic target-session-only handle for that internal
        # table so adapter ID 1 from the isolated session cannot collide with
        # the main session's adapter ID 1.  This handle is never an authority.
        "binding_sam_id": -(2_000_000 + raw_sam_id),
        "binding_handle_source": "target_session_scope_disambiguation_only",
        "presence_score": float(candidate.get("presence_score") or candidate["confidence"]),
        "public_id": None,
        "runtime_future_gt_used": False,
        "runtime_gt_read": False,
        "posthoc_gt_used": False,
        "public_id_inference": False,
        "source_kind": "N72R6_INDEPENDENT_TARGET_SESSION",
    })
    return candidate


def existing_frame(
    runtime: Any, manager: Any, candidates: Sequence[Mapping[str, Any]], *, frame: int,
    event_id: str, session_id: str, event_frame: int, source_path: str,
) -> dict[str, Any]:
    runtime.clear_current_session_bindings(int(frame), reason="n72r6_target_scoped_frame_refresh")
    states, base, score_audit = score_existing(manager, candidates, int(frame))
    solver = exact_solve(
        runtime, states, candidates, base, event_id=event_id,
        branch="C1_Y_PRE_REFERENCE", frame=int(frame), session_id=session_id,
    )
    applied = apply_exact_frame(
        runtime, manager, states, candidates, base, base, solver,
        frame=int(frame), event_id=event_id, branch="C1_Y_PRE_REFERENCE",
        session_id=session_id, event_frame=int(event_frame), source_path=source_path,
        candidate_role="PRE_INTERVENTION_Y_PRE", birth_none=True,
    )
    applied["frame_record"]["score_audit"] = deepcopy(score_audit)
    return applied


def register_target_if_new(
    runtime: Any, manager: Any, target: Mapping[str, Any], *, target_public: int,
    event_frame: int, session_id: str,
) -> Any:
    record = runtime.get_identity_by_public_id(int(target_public))
    if record is not None:
        return record
    record = runtime.create_identity(
        int(event_frame), candidate_obs(target, int(event_frame), source="simulated_human_correction"),
        public_id=int(target_public), session_id=session_id,
        adapter_external_id=int(target["adapter_external_id"]),
        raw_sam_id=int(target["official_raw_sam_id"]), candidate_uid=str(target["candidate_uid"]),
        appearance_state={
            "last_machine_feature": list(target["feature"]),
            "last_machine_feature_sha256": str(target["feature_sha256"]),
            "last_machine_feature_frame": int(event_frame),
        },
        motion_state_ref={"last_box": list(target["box_xyxy"]), "last_frame": int(event_frame)},
        native_scope=str(target["target_session_scope"]),
    )
    manager.register_from_persistent_identity(
        record,
        {
            "feat": np.asarray(target["feature"], dtype=np.float32),
            "box": np.asarray(target["box_xyxy"], dtype=float),
            "native_tid": int(target["native_tid"]),
            "native_scope": str(target["target_session_scope"]),
        },
        int(event_frame),
    )
    return record


def target_frame(
    runtime: Any, manager: Any, main_candidates: Sequence[Mapping[str, Any]],
    target_candidates: Sequence[Mapping[str, Any]], *, frame: int, event_id: str,
    event_frame: int, session_id: str, target_public: int, target_scope: str,
    source_main: str, target_source: str, force_target_uid: str | None,
    y_pre_hash: str, shadowed_main_uids: Sequence[str] = (),
    human_anchor: np.ndarray | None = None,
    human_anchor_gate_threshold: float | None = None,
    allow_human_anchor_main_fallback: bool = False,
) -> dict[str, Any]:
    raw_target_candidates = [dict(item) for item in target_candidates]
    gate_audit: dict[str, Any] = {
        "schema_version": "N72R6_HUMAN_ANCHOR_VERIFICATION_GATE_V1",
        "enabled": False,
        "event_id": str(event_id),
        "frame": int(frame),
        "input_candidate_count": len(raw_target_candidates),
        "accepted_candidate_count": len(raw_target_candidates),
        "rejected_candidate_count": 0,
        "accepted": raw_target_candidates[0].get("candidate_uid") if raw_target_candidates else None,
        "rejected": [],
        "rejected_to_explicit_none": False,
        "main_candidate_fallback": False,
        "runtime_future_gt_used": False,
        "public_id_inference": False,
    }
    if human_anchor_gate_threshold is not None:
        if human_anchor is None:
            raise ValueError("human-anchor gate requires the explicit human anchor")
        if int(frame) > int(event_frame):
            target_candidates, gate_audit = apply_human_anchor_verification_gate(
                raw_target_candidates,
                human_anchor,
                threshold=float(human_anchor_gate_threshold),
                event_id=str(event_id),
                frame=int(frame),
            )
            gate_audit["enabled"] = True
        else:
            target_candidates = raw_target_candidates
    fallback_audit: dict[str, Any] = {
        "enabled": bool(allow_human_anchor_main_fallback and human_anchor_gate_threshold is not None),
        "input_shadowed_main_count": len(shadowed_main_uids),
        "selected_main_candidate_uid": None,
        "selected_human_anchor_cosine": None,
        "threshold": human_anchor_gate_threshold,
        "candidate_scores": [],
        "runtime_future_gt_used": False,
        "public_id_inference": False,
    }
    fallback_main_uid: str | None = None
    if (
        allow_human_anchor_main_fallback
        and human_anchor_gate_threshold is not None
        and human_anchor is not None
        and int(frame) > int(event_frame)
        and not target_candidates
    ):
        anchor = np.asarray(human_anchor, dtype=np.float32).reshape(-1)
        anchor_norm = float(np.linalg.norm(anchor))
        if anchor.size != 512 or not np.all(np.isfinite(anchor)) or anchor_norm <= 1.0e-6:
            raise ValueError("human-anchor main fallback requires a finite non-zero 512-D anchor")
        anchor = anchor / anchor_norm
        by_uid = {str(item["candidate_uid"]): item for item in main_candidates}
        scored: list[tuple[float, str]] = []
        for uid in shadowed_main_uids:
            candidate = by_uid.get(str(uid))
            if candidate is None:
                raise ValueError(f"fallback shadowed main UID is absent from frame: {event_id}:{frame}:{uid}")
            feature = np.asarray(candidate.get("feature"), dtype=np.float32).reshape(-1)
            norm = float(np.linalg.norm(feature))
            if feature.size != 512 or not np.all(np.isfinite(feature)) or norm <= 1.0e-6:
                raise ValueError(f"fallback main candidate feature is invalid: {event_id}:{frame}:{uid}")
            cosine = float(np.dot(feature / norm, anchor))
            scored.append((cosine, str(uid)))
        scored.sort(key=lambda item: (-item[0], item[1]))
        fallback_audit["candidate_scores"] = [
            {
                "candidate_uid": uid,
                "human_anchor_cosine": score,
                "accepted": score >= float(human_anchor_gate_threshold),
            }
            for score, uid in scored
        ]
        if scored and scored[0][0] >= float(human_anchor_gate_threshold):
            fallback_main_uid = scored[0][1]
            fallback_audit["selected_main_candidate_uid"] = fallback_main_uid
            fallback_audit["selected_human_anchor_cosine"] = scored[0][0]
    merged, merge_audit = merge_main_and_target_candidates(
        main_candidates, target_candidates, event_id=event_id, frame=int(frame),
        target_public_id=int(target_public), target_session_scope=str(target_scope),
    )
    reason = "n72r6_after_y_pre_before_target_merge" if frame == event_frame else "n72r6_target_frame_refresh"
    runtime.clear_current_session_bindings(int(frame), reason=reason)
    states, base, score_audit = score_existing(manager, merged, int(frame))
    fused, exclusive_audit = apply_target_exclusive_constraints(
        base, states, merged, runtime, target_public_id=int(target_public),
        force_target_uid=force_target_uid,
        shadowed_main_uids=shadowed_main_uids,
        fallback_main_uid=fallback_main_uid,
    )
    solver = exact_solve(
        runtime, states, merged, fused, event_id=event_id, branch="C1_TARGET_SCOPED",
        frame=int(frame), session_id=session_id,
    )
    target_uids = [str(item["candidate_uid"]) for item in target_candidates]
    applied = apply_exact_frame(
        runtime, manager, states, merged, base, fused, solver,
        frame=int(frame), event_id=event_id, branch="C1_TARGET_SCOPED",
        session_id=session_id, event_frame=int(event_frame),
        source_path=f"main={source_main};target={target_source}",
        candidate_role=("EVENT_Y_POST_TARGET_SCOPED" if frame == event_frame
                        else "FUTURE_ASSOCIATION_TARGET_EXCLUSIVE"),
        birth_none=True, memory_read=False, freeze_public_ids={int(target_public)},
        persistence_mode="FREEZE_MACHINE_PROTOTYPE_AFTER_EVENT",
        birth_none_excluded_uids=[*target_uids, *[str(value) for value in shadowed_main_uids]],
    )
    record = applied["frame_record"]
    record.update({
        "n72r6_variant": "C1_TARGET_SCOPED_NO_TVC",
        "shared_y_pre_semantic_hash": str(y_pre_hash),
        "target_scoped_merge": merge_audit,
        "target_exclusive_constraint": exclusive_audit,
        "score_audit": deepcopy(score_audit),
        "target_session_candidate_source": str(target_source),
        "target_scope_public_id": int(target_public),
        "target_session_raw_candidate_count": len(raw_target_candidates),
        "target_session_candidate_gate": gate_audit,
        "human_anchor_main_fallback": fallback_audit,
        "runtime_future_gt_used": False,
        "posthoc_gt_used": False,
        "public_id_inference": False,
    })
    return applied


def copy_c0(rows: Sequence[Mapping[str, Any]], event_id: str) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for item in rows:
        row = deepcopy(dict(item))
        if row.get("event_id") != event_id:
            raise ValueError(f"C0 event mismatch: {event_id}")
        row.update({
            "n72r6_variant": "C0_MAIN_FROZEN_B0",
            "target_scoped_merge": None,
            "target_exclusive_constraint": None,
            "runtime_future_gt_used": False,
            "posthoc_gt_used": False,
            "public_id_inference": False,
        })
        result.append(row)
    return result


def run_event(
    event: Mapping[str, Any],
    selected: Mapping[str, Any],
    output_root: Path,
    gate_protocol: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    event_id = str(event["event_id"])
    event_frame = int(event["event_frame"])
    target_public = int(event["target_public_id"])
    event_root = output_root / event_id
    manifest_path = event_root / "event_manifest.json"
    if manifest_path.is_file():
        existing = read_json(manifest_path)
        if existing.get("status") == "PASS_N72R6_C0_C1_EVENT_REPLAY":
            existing["skipped_existing"] = True
            return existing
        raise RuntimeError(f"refusing incomplete event artifact: {event_root}")
    event_root.mkdir(parents=True, exist_ok=True)

    done, target_by_frame = read_target_stream(selected)
    main_by_frame, main_source = load_main_stream(event)
    frozen_rows = read_jsonl(resolve(str(event["main_output"])))
    frozen_by_frame = {int(row["frame"]): row for row in frozen_rows}
    if len(frozen_rows) != HORIZON + 1 or set(frozen_by_frame) != set(main_by_frame):
        raise ValueError(f"frozen B0 frame axis mismatch: {event_id}")
    ypre_row = frozen_by_frame[event_frame]
    ypre_hash = str(ypre_row.get("y_pre_semantic_hash") or ypre_row.get("shared_y_pre_semantic_hash"))
    if not ypre_hash:
        raise ValueError(f"frozen Y_pre hash missing: {event_id}")
    ypre_candidate_hash = digest_json([
        {key: item.get(key) for key in (
            "candidate_uid", "candidate_index", "official_raw_sam_id",
            "adapter_external_id", "box_xyxy", "feature_sha256")}
        for item in ypre_row.get("candidate_rows", [])
    ])
    if str(done.get("main_y_pre_candidate_content_sha256")) != ypre_candidate_hash:
        raise ValueError(f"target stream/main Y_pre candidate hash mismatch: {event_id}")
    prestate_path = PRESTATE_ROOT / event_id / "persistent_runtime_snapshot.json"
    if not prestate_path.is_file():
        raise FileNotFoundError(prestate_path)
    prestate = read_json(prestate_path)
    c0_rows = copy_c0(frozen_rows, event_id)

    runtime = new_runtime(str(event["sequence"]), event_id)
    runtime.restore(deepcopy(prestate))
    # A target public identity must remain re-bindable over the full frozen
    # H100 window.  The legacy 90-frame lifecycle gap would terminate its
    # association state before the window ends when the target stream is
    # absent, contradicting the N72R6 LOST-but-persistent contract.  This is
    # recorded in the event manifest and does not alter C0 or metric code.
    manager = new_state_manager(runtime, max_lost_gap=HORIZON + 1)
    session_id = f"{event_id}:N72R6:C1_TARGET_SCOPED"
    runtime.begin_new_sam_session(session_id, boundary_frame=event_frame - 1)
    ypre_applied = existing_frame(
        runtime, manager, main_by_frame[event_frame], frame=event_frame,
        event_id=event_id, session_id=session_id, event_frame=event_frame,
        source_path=main_source,
    )
    if public_semantic(ypre_applied["frame_record"]) != public_semantic(ypre_row):
        raise RuntimeError(f"C1 Y_pre assignment differs from frozen B0: {event_id}")

    target_event_rows = target_by_frame[event_frame]
    if len(target_event_rows) != 1:
        raise RuntimeError(f"target event row is not singleton: {event_id}")
    target_event = target_candidate(target_event_rows[0], event_id, done)
    anchor = read_json(resolve(str(done["human_anchor"])))
    human_anchor = finite_feature(anchor["feature"], f"{event_id}:human_anchor")
    target_record = register_target_if_new(
        runtime, manager, target_event, target_public=target_public,
        event_frame=event_frame, session_id=session_id,
    )
    epoch = runtime.begin_correction_epoch(
        target_record, epoch_id=str(done["correction_epoch_id"]), frame_idx=event_frame,
        human_anchor=human_anchor, authoritative_box=anchor["box_xyxy"],
        target_session_scope=str(done["target_session_scope"]),
        target_native_tid=int(target_event["native_tid"]),
    )
    target_state = manager.states.get(int(target_record.association_state_id))
    if target_state is None:
        raise RuntimeError(f"target public has no association state: {event_id}:{target_public}")
    apply_epoch_to_identity_state(
        target_state, epoch, human_anchor, anchor["box_xyxy"],
        target_native_tid=int(target_event["native_tid"]),
    )

    c1_rows: list[dict[str, Any]] = []
    target_source = str(done["frames"])
    gate_threshold = None if gate_protocol is None else float(gate_protocol["threshold"])
    allow_human_anchor_main_fallback = bool(
        gate_protocol is not None and gate_protocol.get("main_candidate_fallback", False)
    )
    gate_rejected_count = 0
    fallback_used_count = 0
    for frame in range(event_frame, event_frame + HORIZON + 1):
        target_rows = [target_candidate(item, event_id, done) for item in target_by_frame[frame]]
        shadowed_main_uids = [
            str(item["candidate_uid"])
            for item in frozen_by_frame[frame].get("candidate_rows", [])
            if str(item.get("solver_status", "")) == "ASSIGNED_TO_PUBLIC_ID"
            and item.get("solver_public_id") is not None
            and int(item["solver_public_id"]) == int(target_public)
        ]
        applied = target_frame(
            runtime, manager, main_by_frame[frame], target_rows, frame=frame,
            event_id=event_id, event_frame=event_frame, session_id=session_id,
            target_public=target_public, target_scope=str(done["target_session_scope"]),
            source_main=main_source, target_source=target_source,
            force_target_uid=(str(target_rows[0]["candidate_uid"])
                              if frame == event_frame and target_rows else None),
            y_pre_hash=ypre_hash,
            shadowed_main_uids=shadowed_main_uids,
            human_anchor=human_anchor,
            human_anchor_gate_threshold=gate_threshold,
            allow_human_anchor_main_fallback=allow_human_anchor_main_fallback,
        )
        row = applied["frame_record"]
        gate_audit = row["target_session_candidate_gate"]
        gate_rejected_count += int(gate_audit.get("rejected_candidate_count", 0))
        fallback_used_count += int(
            row["human_anchor_main_fallback"].get("selected_main_candidate_uid") is not None
        )
        row.update({
            "correction_epoch": epoch.as_dict(),
            "human_anchor_sha256": str(anchor["feature_sha256"]),
            "target_session_future_candidate_present": bool(
                any(item.get("candidate_kind") == TARGET_CANDIDATE_KIND
                    for item in row.get("candidate_rows", []))
            ),
            "target_session_future_candidate_raw_present": bool(target_rows),
            "target_session_future_candidate_count": int(
                sum(item.get("candidate_kind") == TARGET_CANDIDATE_KIND
                    for item in row.get("candidate_rows", []))
            ),
            "target_session_future_candidate_raw_count": len(target_rows),
            "event_frame_memory_read": False,
        })
        c1_rows.append(row)
    if len(c1_rows) != HORIZON + 1:
        raise RuntimeError(f"C1 frame count mismatch: {event_id}")
    for index, row in enumerate(c1_rows):
        if int(row["frame"]) != event_frame + index:
            raise RuntimeError(f"C1 frame axis mismatch: {event_id}:{row.get('frame')}")
        if row.get("runtime_future_gt_used") is not False or row.get("posthoc_gt_used") is not False:
            raise RuntimeError(f"C1 GT flag is not false: {event_id}:{row['frame']}")

    c0_path = event_root / "C0_MAIN_FROZEN_B0.jsonl"
    c1_path = event_root / "C1_TARGET_SCOPED_NO_TVC.jsonl"
    atomic_jsonl(c0_path, c0_rows)
    atomic_jsonl(c1_path, c1_rows)
    manifest = {
        "schema_version": "N72R6_TARGET_SCOPED_EVENT_REPLAY_V1",
        "status": "PASS_N72R6_C0_C1_EVENT_REPLAY",
        "event_id": event_id,
        "sequence": str(event["sequence"]),
        "action_type": str(event["action_type"]),
        "event_frame": event_frame,
        "horizon": HORIZON,
        "target_public_id": target_public,
        "c0_variant": "C0_MAIN_FROZEN_B0",
        "c1_variant": "C1_TARGET_SCOPED_NO_TVC",
        "c1_state_retention_max_lost_gap": HORIZON + 1,
        "c1_state_retention_policy": "KEEP_TARGET_ASSOCIATION_STATE_REBINDABLE_THROUGH_H100",
        "same_t_minus_1_snapshot": str(prestate_path),
        "same_y_pre_semantic_hash": ypre_hash,
        "same_y_pre_candidate_content_hash": ypre_candidate_hash,
        "c1_ypre_assignment_matches_frozen_b0": True,
        "c0": {"path": str(c0_path), "sha256": sha256_file(c0_path), "frame_count": len(c0_rows)},
        "c1": {"path": str(c1_path), "sha256": sha256_file(c1_path), "frame_count": len(c1_rows)},
        "target_stream_done": str(resolve(str(selected["done"]))),
        "target_stream_frames": str(resolve(str(done["frames"]))),
        "target_event_candidate_count": 1,
        "target_future_candidate_rows": sum(
            len(target_by_frame[frame]) for frame in range(event_frame + 1, event_frame + HORIZON + 1)
        ),
        "human_anchor_gate_enabled": gate_protocol is not None,
        "human_anchor_gate_threshold": gate_threshold,
        "human_anchor_gate_protocol": None if gate_protocol is None else {
            "status": gate_protocol.get("status"),
            "path": gate_protocol.get("path"),
            "sha256": gate_protocol.get("sha256"),
        },
        "human_anchor_gate_rejected_future_candidate_count": int(gate_rejected_count),
        "human_anchor_main_fallback_enabled": allow_human_anchor_main_fallback,
        "human_anchor_main_fallback_used_future_frame_count": int(fallback_used_count),
        "correction_epoch": epoch.as_dict(),
        "target_public_id_immutable": True,
        "target_candidate_public_id_null_before_solver": True,
        "main_candidates_relabelled": False,
        "main_candidates_can_claim_target_public": False,
        "target_candidates_domain": [target_public, "NONE"],
        "event_frame_memory_read": False,
        "first_memory_visible_frame": event_frame + 1,
        "runtime_future_gt_used": False,
        "runtime_gt_read": False,
        "posthoc_gt_used": False,
        "interaction_source": "simulated_from_gt",
        "not_real_human_evidence": True,
        "runtime_invariants": runtime.audit(),
        "created_at_utc": now_utc(),
    }
    atomic_json(manifest_path, manifest)
    return manifest


def write_failure(output_root: Path, event_id: str, exc: BaseException) -> Path:
    attempts = output_root.parent / "attempts"
    attempts.mkdir(parents=True, exist_ok=True)
    path = attempts / f"{event_id}.failure.json"
    if path.exists():
        index = 2
        while (attempts / f"{event_id}.failure.attempt{index}.json").exists():
            index += 1
        path = attempts / f"{event_id}.failure.attempt{index}.json"
    atomic_json(
        path,
        {
            "schema_version": "N72R6_TARGET_SCOPED_REPLAY_FAILURE_V1",
            "status": "FAIL_N72R6_C0_C1_EVENT_REPLAY",
            "event_id": event_id,
            "failure_type": type(exc).__name__,
            "message": str(exc),
            "traceback": traceback.format_exc(),
            "runtime_future_gt_used": False,
            "historical_outputs_modified": False,
            "created_at_utc": now_utc(),
        },
    )
    return path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--event-id", action="append", default=[])
    parser.add_argument("--output-root", type=Path, default=OUT / "attempt_1")
    parser.add_argument(
        "--target-stream-manifest",
        type=Path,
        default=None,
        help="validated 32-event target stream manifest; defaults to the frozen C1 manifest",
    )
    parser.add_argument(
        "--human-anchor-gate-protocol",
        type=Path,
        default=None,
        help="registered human-ROI verification protocol; omitted for un-gated C1",
    )
    args = parser.parse_args()
    global TARGET_STREAM_MANIFEST
    if args.target_stream_manifest is not None:
        TARGET_STREAM_MANIFEST = resolve(args.target_stream_manifest)
    gate_protocol: dict[str, Any] | None = None
    if args.human_anchor_gate_protocol is not None:
        gate_path = resolve(args.human_anchor_gate_protocol)
        gate_protocol = read_json(gate_path)
        if gate_protocol.get("status") not in {
            "PASS_N72R6_HUMAN_ANCHOR_GATE_PROTOCOL_REGISTERED",
            "PASS_N72R6_HUMAN_ANCHOR_FALLBACK_PROTOCOL_REGISTERED",
        }:
            raise SystemExit(f"human-anchor gate protocol is not registered: {gate_protocol.get('status')}")
        threshold = gate_protocol.get("threshold")
        if not isinstance(threshold, (int, float)) or not np.isfinite(float(threshold)):
            raise SystemExit("human-anchor gate protocol threshold is not finite")
        gate_protocol = {
            **gate_protocol,
            "path": str(gate_path),
            "sha256": sha256_file(gate_path),
        }
    output_root = args.output_root if args.output_root.is_absolute() else ROOT / args.output_root
    events = load_events()
    selected = load_selected_streams(TARGET_STREAM_MANIFEST)
    requested = set(args.event_id)
    if requested:
        unknown = requested - {str(item["event_id"]) for item in events}
        if unknown:
            raise SystemExit(f"unknown event IDs: {sorted(unknown)}")
        events = [item for item in events if str(item["event_id"]) in requested]
    results: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for event in events:
        event_id = str(event["event_id"])
        try:
            result = run_event(event, selected[event_id], output_root, gate_protocol)
            results.append(result)
            print(json.dumps({"event_id": event_id, "status": result.get("status"),
                              "skipped_existing": bool(result.get("skipped_existing", False))}))
        except Exception as exc:
            path = write_failure(output_root, event_id, exc)
            failures.append({"event_id": event_id, "artifact": str(path),
                             "failure_type": type(exc).__name__, "message": str(exc)})
            print(json.dumps({"event_id": event_id, "status": "FAIL",
                              "failure_artifact": str(path)}))
    status = "PASS_N72R6_C0_C1_REPLAY" if len(results) == len(events) and not failures else "PARTIAL_N72R6_C0_C1_REPLAY"
    batch = {
        "schema_version": "N72R6_TARGET_SCOPED_REPLAY_BATCH_V1",
        "status": status,
        "requested_event_count": len(events),
        "completed_event_count": len(results),
        "failed_event_count": len(failures),
        "results": results,
        "failures": failures,
        "stage08_manifest": str(STAGE08),
        "stage08_manifest_sha256": sha256_file(STAGE08),
        "target_stream_manifest": str(TARGET_STREAM_MANIFEST),
        "target_stream_manifest_sha256": sha256_file(TARGET_STREAM_MANIFEST),
        "target_stream_manifest_status": read_json(TARGET_STREAM_MANIFEST).get("status"),
        "target_session_recovery_mode": bool(read_json(TARGET_STREAM_MANIFEST).get("target_session_recovery_mode", False)),
        "target_stream_replay_ready": bool(read_json(TARGET_STREAM_MANIFEST).get("replay_ready", False)),
        "human_anchor_gate_enabled": gate_protocol is not None,
        "human_anchor_gate_protocol": None if gate_protocol is None else {
            "status": gate_protocol.get("status"),
            "path": gate_protocol.get("path"),
            "sha256": gate_protocol.get("sha256"),
            "threshold": gate_protocol.get("threshold"),
            "main_candidate_fallback": bool(gate_protocol.get("main_candidate_fallback", False)),
        },
        "runtime_future_gt_used": False,
        "created_at_utc": now_utc(),
    }
    atomic_json(output_root / "replay_batch_status.json", batch)
    return 0 if status.startswith("PASS") else 1


if __name__ == "__main__":
    raise SystemExit(main())
