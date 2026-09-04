#!/usr/bin/env python3
"""Run the frozen D1/D2 replay for the two N72R7 confirmation events.

The confirmation events are deliberately outside the N72R6 32-event replay
set.  This worker therefore builds a small, read-only frozen manifest from
the N72R5R1 public-assignment rows and the independently generated N72R7
target-session stream.  It never loads dataset GT; GT is used only by the
separate posthoc scorer after these runtime artifacts are sealed.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import traceback
from typing import Any, Mapping

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.n72r7_dev_replay import (  # noqa: E402
    DEV_PROTOCOL,
    HORIZON,
    atomic_json,
    read_json,
    read_jsonl,
    run_event,
    sha256_file,
)
from scripts.n72r7_learned_replay import LearnedTargetCandidateSelector  # noqa: E402


CONFIRMATION_PROTOCOL = ROOT / "outputs/N72R7/confirmation/confirmation_protocol.json"
REPLAY_PROTOCOL = ROOT / "outputs/N72R7/training_v2/learned_replay_protocol.json"
STAGE08 = ROOT / "outputs/N72R5R1/controller/round_05_branch_isolation_v0/full/stage08_runtime_manifest.json"
PUBLIC_ASSIGNMENT_ROOT = STAGE08.parent / "public_assignment"


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def _finite_vector(value: Any, *, location: str) -> np.ndarray:
    vector = np.asarray(value, dtype=np.float64).reshape(-1)
    if vector.size == 0 or not np.all(np.isfinite(vector)):
        raise RuntimeError(f"non-finite or empty feature at {location}")
    norm = float(np.linalg.norm(vector))
    if not math.isfinite(norm) or norm <= 0.0:
        raise RuntimeError(f"zero/non-finite feature norm at {location}")
    return vector


def _validate_association_rows(path: Path, *, event_id: str, label: str, event_frame: int) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    rows = read_jsonl(path)
    expected = list(range(event_frame, event_frame + HORIZON + 1))
    if len(rows) != HORIZON + 1 or [int(row.get("frame", -1)) for row in rows] != expected:
        raise RuntimeError(f"{label} frame axis is not {event_id}:{expected[0]}..{expected[-1]}")
    for index, row in enumerate(rows):
        location = f"{label}:{event_id}:{row.get('frame')}"
        if row.get("event_id") != event_id or int(row.get("event_frame", -1)) != event_frame:
            raise RuntimeError(f"{location} event identity mismatch")
        for flag in ("runtime_future_gt_used", "runtime_gt_read", "posthoc_gt_used", "public_id_inference"):
            if row.get(flag) is not False:
                raise RuntimeError(f"{location} causal flag {flag} is not false")
        candidates = list(row.get("candidate_rows", []))
        if int(row.get("candidate_count", -1)) != len(candidates):
            raise RuntimeError(f"{location} candidate count mismatch")
        uids = [str(item.get("candidate_uid")) for item in candidates]
        if any(uid in {"None", ""} for uid in uids) or len(uids) != len(set(uids)):
            raise RuntimeError(f"{location} candidate UID is missing or duplicated")
        for candidate in candidates:
            feature = candidate.get("feature")
            if feature is not None:
                _finite_vector(feature, location=f"{location}/{candidate['candidate_uid']}")
        base = np.asarray(row.get("base_score_matrix", []), dtype=np.float64)
        fused = np.asarray(row.get("fused_score_matrix", []), dtype=np.float64)
        states = list(row.get("association_state_axis", []))
        publics = list(row.get("public_id_axis", []))
        if base.ndim != 2 or fused.shape != base.shape or base.shape != (len(candidates), len(states)):
            raise RuntimeError(f"{location} score matrix shape mismatch: {base.shape}/{fused.shape}")
        if len(states) == 0 or len(states) != len(set(states)) or not np.all(np.isfinite(base)) or not np.all(np.isfinite(fused)):
            raise RuntimeError(f"{location} score matrix/association axis invalid")
        if len(publics) == 0 or len(publics) != len(set(publics)):
            raise RuntimeError(f"{location} public axis invalid")
        if index == 0 and row.get("memory_read") is not False:
            raise RuntimeError(f"{location} event frame read memory")
    return rows


def _validate_target_stream(
    root: Path,
    *,
    event_id: str,
    event_frame: int,
    target_public_id: int,
    target_attempt: int,
) -> tuple[Path, Path, dict[str, Any]]:
    event_root = root / f"attempt_{int(target_attempt)}" / event_id
    done_path = event_root / "done.json"
    frames_path = event_root / "frames.jsonl"
    mapping_path = event_root / "target_session_frame_mapping.json"
    anchor_path = event_root / "human_anchor.json"
    done = read_json(done_path)
    if done.get("status") != "PASS_TARGET_STREAM_COMPLETE":
        raise RuntimeError(f"target stream is not PASS: {done_path}:{done.get('status')}")
    if int(done.get("event_frame", -1)) != event_frame or int(done.get("target_public_id", -1)) != target_public_id:
        raise RuntimeError(f"target stream authority mismatch: {event_id}")
    if done.get("target_candidate_present_event_frame") is not True or done.get("target_candidate_present_event_plus_one") is not True:
        raise RuntimeError(f"target stream lacks event/event+1 candidate: {event_id}")
    if done.get("runtime_future_gt_used") is not False or done.get("runtime_gt_read") is not False or done.get("public_id_inference") is not False:
        raise RuntimeError(f"target stream causal flags invalid: {event_id}")
    audit = done.get("target_session_audit")
    if not isinstance(audit, Mapping) or audit.get("event_frame_memory_read") is not False or int(audit.get("first_memory_visible_frame", -1)) != event_frame + 1:
        raise RuntimeError(f"target stream memory boundary invalid: {event_id}")
    if int(done.get("frame_count", -1)) != HORIZON + 1:
        raise RuntimeError(f"target stream frame count invalid: {event_id}")
    _finite_vector(read_json(anchor_path).get("feature"), location=f"{event_id}:human_anchor")
    rows = read_jsonl(frames_path)
    expected = list(range(event_frame, event_frame + HORIZON + 1))
    if len(rows) != HORIZON + 1 or [int(row.get("frame", -1)) for row in rows] != expected:
        raise RuntimeError(f"target stream frame axis invalid: {event_id}")
    for index, row in enumerate(rows):
        location = f"target:{event_id}:{row.get('frame')}"
        for flag in ("runtime_future_gt_used", "runtime_gt_read", "posthoc_gt_used", "public_id_inference"):
            if row.get(flag) is not False:
                raise RuntimeError(f"{location} flag {flag} is not false")
        if row.get("candidate_set_complete") is not True:
            raise RuntimeError(f"{location} candidate set is incomplete")
        candidates = list(row.get("candidate_rows", []))
        if int(row.get("candidate_count", -1)) != len(candidates):
            raise RuntimeError(f"{location} candidate count mismatch")
        if len({str(item.get("candidate_uid")) for item in candidates}) != len(candidates):
            raise RuntimeError(f"{location} duplicate target candidate UID")
        if any(item.get("public_id") is not None for item in candidates):
            raise RuntimeError(f"{location} target source owns public ID before replay solver")
        if index == 0:
            if row.get("is_event_frame") is not True or row.get("is_future_frame") is not False or len(candidates) == 0 or row.get("memory_read") is not False:
                raise RuntimeError(f"{location} event-frame contract invalid")
        else:
            if row.get("is_event_frame") is not False or row.get("is_future_frame") is not True:
                raise RuntimeError(f"{location} future-frame contract invalid")
    mapping = read_json(mapping_path)
    mapping_rows = list(mapping.get("mapping", []))
    if mapping.get("runtime_future_gt_used") is not False or len(mapping_rows) != HORIZON + 1:
        raise RuntimeError(f"target mapping contract invalid: {event_id}")
    if [int(item.get("global_frame", -1)) for item in mapping_rows] != expected or [int(item.get("local_frame", -1)) for item in mapping_rows] != list(range(HORIZON + 1)):
        raise RuntimeError(f"target mapping frame axis invalid: {event_id}")
    if any(not str(item.get("source_frame_sha256", "")) for item in mapping_rows):
        raise RuntimeError(f"target mapping source hash missing: {event_id}")
    return frames_path, done_path, done


def _frozen_manifest(spec: Mapping[str, Any], target_root: Path, target_attempt: int) -> tuple[dict[str, Any], dict[str, Any]]:
    event_id = str(spec["event_id"])
    event_frame = int(spec["event_frame"])
    target_public_id = int(spec["target_public_id"])
    assignment_dir = PUBLIC_ASSIGNMENT_ROOT / event_id
    c0_path = assignment_dir / "B0_NO_INTERVENTION.jsonl"
    c1_path = assignment_dir / "B1_SPATIAL_CORRECTION_ONLY.jsonl"
    c0_rows = _validate_association_rows(c0_path, event_id=event_id, label="N72R5R1_B0", event_frame=event_frame)
    c1_rows = _validate_association_rows(c1_path, event_id=event_id, label="N72R5R1_B1", event_frame=event_frame)
    if str(spec.get("main_b0_path")) != str(c0_path) or str(spec.get("main_b0_sha256")) != sha256_file(c0_path):
        raise RuntimeError(f"confirmation protocol B0 provenance mismatch: {event_id}")
    target_frames, target_done, done = _validate_target_stream(
        target_root,
        event_id=event_id,
        event_frame=event_frame,
        target_public_id=target_public_id,
        target_attempt=target_attempt,
    )
    c0_sha = sha256_file(c0_path)
    c1_sha = sha256_file(c1_path)
    target_sha = sha256_file(target_frames)
    target_done_sha = sha256_file(target_done)
    frozen = {
        "event_id": event_id,
        "sequence": str(spec["sequence"]),
        "event_frame": event_frame,
        "target_public_id": target_public_id,
        "c0": {"path": str(c0_path), "sha256": c0_sha},
        "c1": {"path": str(c1_path), "sha256": c1_sha},
        "target_stream_frames": str(target_frames),
        "target_stream_frames_sha256": target_sha,
        "target_stream_done": str(target_done),
        "target_stream_done_sha256": target_done_sha,
    }
    authority = {
        "association_state_id": int(spec["target_association_state_id"]),
        "public_id": target_public_id,
        "other_public_id": spec.get("other_public_id"),
        "source": str(spec["authority"]),
    }
    return frozen, {
        "event_id": event_id,
        "sequence": str(spec["sequence"]),
        "action_type": str(spec["action_type"]),
        "event_frame": event_frame,
        "target_public_id": target_public_id,
        "target_authority": authority,
        "c0_sha256": c0_sha,
        "c1_sha256": c1_sha,
        "target_frames_sha256": target_sha,
        "target_done_sha256": target_done_sha,
        "target_session_status": done.get("status"),
        "event_frame_candidate_present": done.get("target_candidate_present_event_frame"),
        "event_plus_one_candidate_present": done.get("target_candidate_present_event_plus_one"),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--event-id", required=True)
    parser.add_argument("--target-root", type=Path, required=True)
    parser.add_argument("--target-attempt", type=int, default=2)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--attempt", type=int, default=1)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    target_root = _path(args.target_root)
    output_root = _path(args.output_root)
    event_id = str(args.event_id)
    result: dict[str, Any] = {
        "schema_version": "N72R7_CONFIRMATION_REPLAY_WORKER_V1",
        "status": "FAIL",
        "event_id": event_id,
        "attempt": int(args.attempt),
        "target_attempt": int(args.target_attempt),
        "started_at_utc": now_utc(),
        "runtime_future_gt_used": False,
        "posthoc_gt_used": False,
        "interaction_source": "simulated_from_gt",
        "not_real_human_evidence": True,
    }
    try:
        protocol = read_json(CONFIRMATION_PROTOCOL)
        events = {str(item["event_id"]): item for item in protocol.get("events", [])}
        if event_id not in events:
            raise RuntimeError(f"event is absent from frozen confirmation protocol: {event_id}")
        spec = events[event_id]
        frozen, input_audit = _frozen_manifest(spec, target_root, int(args.target_attempt))
        dev_protocol = read_json(DEV_PROTOCOL)
        replay_protocol = read_json(REPLAY_PROTOCOL)
        checkpoint = _path(str(replay_protocol["checkpoint"]))
        if sha256_file(checkpoint) != str(replay_protocol["checkpoint_sha256"]):
            raise RuntimeError("learned replay checkpoint hash mismatch")
        device = torch.device(str(args.device))
        event = {
            "event_id": event_id,
            "sequence": str(spec["sequence"]),
            "action_type": str(spec["action_type"]),
        }
        authority_pair = (
            int(spec["target_association_state_id"]),
            int(spec["target_public_id"]),
        )
        d1_root = output_root / "D1"
        d2_root = output_root / "D2"
        d1_selector = LearnedTargetCandidateSelector(checkpoint, device=device, protocol=replay_protocol)
        d1_manifest = run_event(
            event,
            frozen,
            variant="D1",
            output_root=d1_root,
            selector=d1_selector,
            protocol=dev_protocol,
            target_authority_pair=authority_pair,
        )
        del d1_selector
        d2_selector = LearnedTargetCandidateSelector(checkpoint, device=device, protocol=replay_protocol)
        d2_manifest = run_event(
            event,
            frozen,
            variant="D2",
            output_root=d2_root,
            selector=d2_selector,
            protocol=dev_protocol,
            target_authority_pair=authority_pair,
        )
        del d2_selector
        result.update({
            "status": "PASS_N72R7_CONFIRMATION_REPLAY",
            "sequence": str(spec["sequence"]),
            "action_type": str(spec["action_type"]),
            "event_frame": int(spec["event_frame"]),
            "target_public_id": int(spec["target_public_id"]),
            "target_authority": input_audit["target_authority"],
            "input_audit": input_audit,
            "frozen_manifest": frozen,
            "protocol": str(CONFIRMATION_PROTOCOL),
            "protocol_sha256": sha256_file(CONFIRMATION_PROTOCOL),
            "dev_protocol": str(DEV_PROTOCOL),
            "dev_protocol_sha256": sha256_file(DEV_PROTOCOL),
            "replay_protocol": str(REPLAY_PROTOCOL),
            "replay_protocol_sha256": sha256_file(REPLAY_PROTOCOL),
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": str(replay_protocol["checkpoint_sha256"]),
            "D1_manifest": str(d1_root / event_id / "event_manifest.json"),
            "D1_manifest_sha256": sha256_file(d1_root / event_id / "event_manifest.json"),
            "D2_manifest": str(d2_root / event_id / "event_manifest.json"),
            "D2_manifest_sha256": sha256_file(d2_root / event_id / "event_manifest.json"),
            "variants": {
                "D1": "frozen B0 pool plus learned target/NONE selector; no target-session row",
                "D2": "same frozen B0 pool plus current official target-session row",
            },
            "device": str(device),
            "runtime_future_gt_used": False,
            "posthoc_gt_used": False,
            "finished_at_utc": now_utc(),
        })
        atomic_json(output_root / event_id / "worker_status.json", result)
        print(json.dumps({"event_id": event_id, "status": result["status"]}, sort_keys=True))
        return 0
    except Exception as exc:
        result.update({
            "status": "FAIL_N72R7_CONFIRMATION_REPLAY",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
            "finished_at_utc": now_utc(),
        })
        atomic_json(output_root / "attempts" / f"{event_id}.attempt{int(args.attempt)}.failure.json", result)
        print(json.dumps({"event_id": event_id, "status": result["status"], "error": str(exc)}, sort_keys=True))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
