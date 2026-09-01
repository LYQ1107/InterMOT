#!/usr/bin/env python3
"""Freeze and materialize the N37 real DanceTrack event protocol.

Stage A is deliberately split into two boundaries:

* ``scan`` is one offline N8 discovery process for exactly one sequence.  It
  records only compact event candidates and the native-to-N8 map needed to
  rebuild a prefix.  The process does not retain the large N8 audit/post-row
  collections used by the historical N36 builder.
* ``assemble`` is a small parent coordinator.  It launches one child scan per
  sequence, selects events using a frozen deterministic quota rule, and
  launches one isolated materializer per selected sequence.  No replay result
  or future metric is consulted by selection.

All inputs are the already validated N36 train/train_fold candidate tape and
the offline train annotations.  The annotation boundary is explicit: GT is
used to freeze the simulated human current-frame box and to score event
eligibility, never by the later full-loop/replay runtime.
"""

from __future__ import annotations

import argparse
import copy
import gc
import hashlib
import json
import os
import resource
import subprocess
import sys
import traceback
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sam3_intermot.association.ccam_replay import _manager_from_prefix
from sam3_intermot.association.human_intervention import HumanFeatureExtractor
from sam3_intermot.association.state_manager import StateManagerConfig
from sam3_intermot.datasets.dancetrack import DanceTrackDataset
from sam3_intermot.interaction.n8_temporal_observer import N8Config, N8TemporalObserver
from sam3_intermot.interaction.simulator import GTFrame

from scripts.n36_tape_common import DATA_ROOT, atomic_json, box_iou, display_path, iter_jsonl
from scripts.run_n36_build_events import (
    best_candidate,
    build_event as build_n36_event,
    build_prefix,
    current_rows,
    event_box,
    load_backbone,
    load_gt,
    make_runtime_probe,
)


OUT = ROOT / "outputs/n37"
DEFAULT_SEQUENCE_LIST = ROOT / "outputs/n34/selected_sequences.json"
DEFAULT_TAPE_ROOT = ROOT / "outputs/n36/real_tape/frames"
DEFAULT_MANIFEST = OUT / "real_event_manifest.json"
DEFAULT_PROTOCOL = OUT / "event_protocol.json"
DEFAULT_STAGE = OUT / "stage_01_status.json"
SCAN_DIR = OUT / "event_candidates"
MATERIALIZED_DIR = OUT / "event_materialized"
SELECTION_FILE = OUT / "selected_event_refs.json"
FAILURE_EVIDENCE = OUT / "event_scan_failure_evidence.json"
HUMAN_CHECKPOINT = ROOT / "outputs/n9/checkpoints/osnet_x1_0_market1501.pth"

FEATURE_DIM = 512
HORIZON = 100
TARGET_EVENTS = 24
MIN_INDEPENDENT_SEQUENCES = 12
MIN_PER_ACTION = 4
MIN_MULTI_GT = 2
MIN_MULTI_CANDIDATES = 2
MAX_POOL_PER_ACTION = 128
ACTION_TYPES = (
    "AUTHORITATIVE_REASSIGN",
    "ATOMIC_ID_SWAP",
    "RECOVER_IDENTITY",
    "ADD_NEW_IDENTITY",
)

_GT_CACHE: dict[str, dict[int, GTFrame]] = {}


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


def digest_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def digest_feature(value: Any) -> str:
    vector = np.asarray(value, dtype=np.float32).reshape(-1)
    return hashlib.sha256(vector.tobytes()).hexdigest()


def cached_gt_frames(sequence: str) -> dict[int, GTFrame]:
    """Cache only the offline event-frame GT used by Stage-A audits."""
    if sequence not in _GT_CACHE:
        _GT_CACHE[sequence] = load_gt(
            DanceTrackDataset(str(DATA_ROOT), sequences=[sequence], split="train"), sequence
        )
    return _GT_CACHE[sequence]


def load_sequences(path: Path) -> list[str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    sequences = payload.get("sequences", [])
    output = [
        str(item["sequence"])
        for item in sequences
        if isinstance(item, dict) and item.get("sequence")
    ]
    if len(output) != 24 or len(set(output)) != 24:
        raise RuntimeError(f"N36 selected sequence list is not exactly 24 unique sequences: {output}")
    return sorted(output)


def event_ref(sequence: str, frame: int, action: str, ordinal: int) -> str:
    return f"n37-{sequence}-{int(frame):04d}-{action.lower()}-{int(ordinal):03d}"


def compact_pre_rows(rows: list[tuple[int, np.ndarray]]) -> list[list[Any]]:
    return [
        [int(pid), np.asarray(box, dtype=float).reshape(-1).tolist()]
        for pid, box in rows
    ]


def compact_event_candidate(
    event: dict[str, Any],
    *,
    sequence: str,
    frame: int,
    ordinal: int,
    pre_rows: list[tuple[int, np.ndarray]],
    prefix_ids: set[int],
    gt_count: int,
    candidate_count: int,
    frame_count: int,
) -> dict[str, Any]:
    action = str(event["action_type"])
    n8_event = copy.deepcopy(event)
    ref = event_ref(sequence, frame, action, ordinal)
    canonical = n8_event.get("canonical_public_id")
    other = n8_event.get("other_canonical_public_id")
    # N8's serialized frame is one-based for human-facing reports.  The
    # protocol stores the tape's zero-based frame explicitly and retains the
    # original value only as provenance.
    n8_event["frame"] = int(frame)
    return {
        "candidate_id": ref,
        "sequence": sequence,
        "frame": int(frame),
        "action_type": action,
        "event_type": str(n8_event.get("event_type", "")),
        "dataset_gt_id": None if n8_event.get("dataset_gt_id") is None else int(n8_event["dataset_gt_id"]),
        "current_public_id": None if n8_event.get("current_public_id") is None else int(n8_event["current_public_id"]),
        "canonical_public_id": None if canonical is None else int(canonical),
        "other_dataset_gt_id": None if n8_event.get("other_dataset_gt_id") is None else int(n8_event["other_dataset_gt_id"]),
        "other_canonical_public_id": None if other is None else int(other),
        "target_auto_tid": None if n8_event.get("target_auto_tid") is None else int(n8_event["target_auto_tid"]),
        "other_auto_tid": None if n8_event.get("other_auto_tid") is None else int(n8_event["other_auto_tid"]),
        "iou": None if n8_event.get("iou") is None else float(n8_event["iou"]),
        "gt_box": None if n8_event.get("gt_box") is None else list(n8_event["gt_box"]),
        "pre_box": None if n8_event.get("pre_box") is None else list(n8_event["pre_box"]),
        "seen_before": bool(n8_event.get("seen_before", False)),
        "gap_length": n8_event.get("gap_length"),
        "accepted": bool(n8_event.get("accepted", False)),
        "n8_event": n8_event,
        "pre_rows": compact_pre_rows(pre_rows),
        "prefix_ids_n8": sorted(int(pid) for pid in prefix_ids),
        "canonical_in_prefix_n8": (
            action == "ADD_NEW_IDENTITY"
            or canonical is not None and int(canonical) in prefix_ids
        ),
        "other_in_prefix_n8": (
            action != "ATOMIC_ID_SWAP"
            or other is not None and int(other) in prefix_ids
        ),
        "gt_visible_count": int(gt_count),
        "candidate_count": int(candidate_count),
        "multi_identity_context": bool(
            int(gt_count) >= MIN_MULTI_GT
            and int(candidate_count) >= MIN_MULTI_CANDIDATES
        ),
        "h100_available": bool(int(frame) + HORIZON < int(frame_count)),
        "frame_count": int(frame_count),
    }


def scan_sequence(sequence: str, tape_root: Path, output: Path) -> dict[str, Any]:
    """Discover compact N8 candidates in a single isolated process."""
    path = tape_root / f"{sequence}.jsonl"
    if not path.is_file():
        raise FileNotFoundError(path)
    backbone, frame_count = load_backbone(path)
    dataset = DanceTrackDataset(str(DATA_ROOT), sequences=[sequence], split="train")
    gt_frames = load_gt(dataset, sequence)
    observer = N8TemporalObserver(
        backbone,
        gt_frames,
        frame_count,
        N8Config(budget=-1, match_iou_threshold=0.5, sequence=sequence),
        sequence=sequence,
    )
    candidates: list[dict[str, Any]] = []
    eligible_counts: Counter[str] = Counter()
    action_ordinals: Counter[str] = Counter()
    map_before_by_frame: dict[str, dict[str, int]] = {}
    prefix_ids: set[int] = set()
    n8_all_counts: Counter[str] = Counter()
    n8_accepted_counts: Counter[str] = Counter()
    for frame in range(frame_count):
        raw = backbone.get(frame, [])
        map_before_by_frame[str(frame)] = {
            str(int(native)): int(public)
            for native, public in observer.canonical_map.items()
        }
        pre_rows = observer._assemble_pre(raw)
        gt = gt_frames.get(frame, GTFrame())
        events = observer._detect_errors(frame, raw, gt)
        selected = [
            event
            for event in events
            if event.get("interaction_required")
            and event.get("action_type") in ACTION_TYPES
        ]
        # _apply_events mutates accepted status and supplies the canonical PID
        # for ADD_NEW_IDENTITY.  Capture candidates only after that operation.
        observer._apply_events(frame, events)
        for event in selected:
            action = str(event["action_type"])
            n8_all_counts[action] += 1
            if bool(event.get("accepted", False)):
                n8_accepted_counts[action] += 1
            action_ordinals[action] += 1
            canonical = event.get("canonical_public_id")
            other = event.get("other_canonical_public_id")
            eligible = (
                frame + HORIZON < frame_count
                and len(gt.gt_ids) >= MIN_MULTI_GT
                and len(raw) >= MIN_MULTI_CANDIDATES
                and (
                    action == "ADD_NEW_IDENTITY"
                    or canonical is not None and int(canonical) in prefix_ids
                )
                and (
                    action != "ATOMIC_ID_SWAP"
                    or other is not None and int(other) in prefix_ids
                )
            )
            if eligible:
                eligible_counts[action] += 1
                # The protocol consumes only the earliest deterministic pool
                # candidates.  Keep the complete per-action count separately
                # while bounding JSON/Python memory for a large sequence.
                if eligible_counts[action] <= MAX_POOL_PER_ACTION:
                    candidates.append(
                        compact_event_candidate(
                            event,
                            sequence=sequence,
                            frame=frame,
                            ordinal=action_ordinals[action],
                            pre_rows=pre_rows,
                            prefix_ids=prefix_ids,
                            gt_count=len(gt.gt_ids),
                            candidate_count=len(raw),
                            frame_count=frame_count,
                        )
                    )
        # This is the only retained state needed to attest that a canonical
        # N8 public ID has appeared before a later correction.  Do not retain
        # observer.pre_rows/post_rows/observer_audit/verified_errors.
        prefix_ids.update(int(pid) for pid, _box in pre_rows)
        # These N8 collections are useful for the historical full audit but
        # are not part of N37's frozen event-selection contract.  Keeping them
        # across thousands of frames was the actionable cause of the prior
        # process-level termination when several sequences shared one Python
        # interpreter.
        observer.verified_errors.clear()
        observer.interaction_events.clear()
        observer.observer_audit.clear()
        observer.state_hashes.clear()
        observer.invariant_violations.clear()
        del events, selected, pre_rows, raw, gt
        # Full collection on every frame turns a long sequence into a
        # quadratic-looking audit path.  Reference counting already releases
        # the per-frame objects; periodic collection is sufficient after the
        # explicit observer-log clears above.
        if frame % 128 == 0:
            gc.collect()
    gc.collect()
    if not candidates:
        raise RuntimeError(f"no eligible multi-identity N37 candidates for {sequence}")
    payload = {
        "protocol": "N37_REAL_EVENT_DISCOVERY_CHILD",
        "status": "PASS",
        "sequence": sequence,
        "split": "train/train_fold",
        "source_tape": display_path(path),
        "source_tape_sha256": digest_file(path),
        "frame_count": int(frame_count),
        "candidate_count": len(candidates),
        "eligible_candidate_count": int(sum(eligible_counts.values())),
        "n8_all_action_counts": {key: int(value) for key, value in sorted(n8_all_counts.items())},
        "n8_accepted_action_counts": {key: int(value) for key, value in sorted(n8_accepted_counts.items())},
        "eligible_action_counts": {
            action: int(eligible_counts[action]) for action in ACTION_TYPES
        },
        "candidate_pool_cap_per_action": MAX_POOL_PER_ACTION,
        "candidate_pool_truncated": bool(
            any(eligible_counts[action] > MAX_POOL_PER_ACTION for action in ACTION_TYPES)
        ),
        "map_before_by_frame": map_before_by_frame,
        "candidates": candidates,
        "runtime_future_gt_used": False,
        "maxrss_kb": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
        "retained_n8_large_collections": False,
    }
    atomic_json(output, jsonable(payload))
    print(
        json.dumps(
            {
                "mode": "scan",
                "sequence": sequence,
                "status": payload["status"],
                "eligible": payload["candidate_count"],
                "maxrss_kb": payload["maxrss_kb"],
                "output": display_path(output),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return payload


def public_id(n8_pid: int) -> int:
    return 100_000 + int(n8_pid)


def _best_observation_by_box(observations: list[dict[str, Any]], box: Any) -> tuple[dict[str, Any] | None, float]:
    if box is None or not observations:
        return None, 0.0
    values = [
        float(max(0.0, box_iou(np.asarray(box, dtype=float), item["box"])))
        for item in observations
    ]
    index = int(np.argmax(np.asarray(values, dtype=float)))
    return observations[index], float(values[index])


def static_precondition(
    candidate: dict[str, Any],
    scan_payload: dict[str, Any],
    tape_root: Path,
) -> dict[str, Any]:
    """Check current-frame transaction legality without human extraction/replay.

    This is deliberately limited to the prefix and event frame.  It catches
    invalid atomic probes (for example both corrected targets resolving to the
    same current public ID) before the expensive human crop materializer.
    """
    sequence = str(candidate["sequence"])
    frame = int(candidate["frame"])
    path = tape_root / f"{sequence}.jsonl"
    map_by_frame = {
        int(current_frame): {int(native): int(pid) for native, pid in mapping.items()}
        for current_frame, mapping in scan_payload.get("map_before_by_frame", {}).items()
    }
    prefix = build_prefix(path, frame, map_by_frame)
    observations = current_rows(path, frame)
    action = str(candidate["action_type"])
    # Use the same offline event-frame GT lookup as build_n36_event.  The
    # frozen candidate's ``pre_rows`` are prefix/probe observations, not the
    # second identity's event-frame GT box; using them here can select the
    # target observation twice and produce a false legal result.
    gt_frames = cached_gt_frames(sequence)
    target_box = event_box(gt_frames[frame], int(candidate["dataset_gt_id"]))
    target_obs, target_iou = _best_observation_by_box(observations, target_box)
    if action == "AUTHORITATIVE_REASSIGN":
        if target_obs is None or target_iou < 0.3:
            return {"valid": False, "reason": f"target_candidate_iou={target_iou:.6f}"}
        probe_rows = make_runtime_probe(prefix, observations, frame)
        if not probe_rows:
            return {"valid": False, "reason": "empty_current_public_probe"}
        current_pid = int(
            probe_rows[
                int(
                    np.argmax(
                        np.asarray(
                            [
                                box_iou(target_obs["box"], box)
                                for _pid, box in probe_rows
                            ],
                            dtype=float,
                        )
                    )
                )
            ][0]
        )
        if current_pid == public_id(int(candidate["canonical_public_id"])):
            return {"valid": False, "reason": "target_already_canonical"}
        return {"valid": True, "current_public_id": current_pid}
    if action == "ATOMIC_ID_SWAP":
        other_gid = candidate.get("other_dataset_gt_id")
        if other_gid is None:
            return {"valid": False, "reason": "missing_other_dataset_gt_id"}
        other_box = event_box(gt_frames[frame], int(other_gid))
        other_obs, other_iou = _best_observation_by_box(observations, other_box)
        if target_obs is None or other_obs is None or target_iou < 0.3 or other_iou < 0.3:
            return {
                "valid": False,
                "reason": f"candidate_iou={target_iou:.6f}/{other_iou:.6f}",
                "target_iou": float(target_iou),
                "other_iou": float(other_iou),
            }
        probe_rows = make_runtime_probe(prefix, observations, frame)
        if not probe_rows:
            return {"valid": False, "reason": "empty_current_public_probe"}

        def probed_pid(observation: dict[str, Any]) -> int:
            scores = np.asarray(
                [
                    box_iou(observation["box"], box)
                    for _pid, box in probe_rows
                ],
                dtype=float,
            )
            return int(probe_rows[int(np.argmax(scores))][0])

        target_pid = probed_pid(target_obs)
        other_pid = probed_pid(other_obs)
        diagnostics = {
            "target_current_public_id": target_pid,
            "other_current_public_id": other_pid,
            "target_native_tid": int(target_obs["native_tid"]),
            "other_native_tid": int(other_obs["native_tid"]),
            "target_iou": float(target_iou),
            "other_iou": float(other_iou),
            "target_gt_box": np.asarray(target_box, dtype=float).tolist(),
            "other_gt_box": np.asarray(other_box, dtype=float).tolist(),
        }
        if target_pid == other_pid:
            return {"valid": False, "reason": "target_other_share_current_public_id", **diagnostics}
        if target_pid == public_id(int(candidate["canonical_public_id"])):
            return {"valid": False, "reason": "target_already_canonical", **diagnostics}
        if other_pid == public_id(int(candidate["other_canonical_public_id"])):
            return {"valid": False, "reason": "other_already_canonical", **diagnostics}
        return {"valid": True, **diagnostics}
    return {"valid": True}


def _reference_builder_atomic_precondition(
    candidate: dict[str, Any],
    scan_payload: dict[str, Any],
    tape_root: Path,
) -> dict[str, Any]:
    """Recompute the ATOMIC_ID_SWAP checks exactly as ``build_event`` does.

    This is an audit-only duplicate of the builder's precondition block.  It
    intentionally does not construct a human feature or write an event, so a
    consistency audit cannot select a replacement or mutate the manifest.
    """
    sequence = str(candidate["sequence"])
    frame = int(candidate["frame"])
    path = tape_root / f"{sequence}.jsonl"
    map_by_frame = {
        int(current_frame): {int(native): int(pid) for native, pid in mapping.items()}
        for current_frame, mapping in scan_payload.get("map_before_by_frame", {}).items()
    }
    prefix = build_prefix(path, frame, map_by_frame)
    observations = current_rows(path, frame)
    gt_frames = cached_gt_frames(sequence)
    target_box = event_box(gt_frames[frame], int(candidate["dataset_gt_id"]))
    target_index, target_iou = best_candidate(observations, target_box)
    target_obs = None if target_index is None else observations[target_index]
    other_box = event_box(gt_frames[frame], int(candidate["other_dataset_gt_id"]))
    other_index, other_iou = best_candidate(observations, other_box)
    other_obs = None if other_index is None else observations[other_index]
    probe_rows = make_runtime_probe(prefix, observations, frame)
    if target_obs is None or other_obs is None or target_iou < 0.3 or other_iou < 0.3:
        return {
            "valid": False,
            "reason": f"candidate_iou={target_iou:.6f}/{other_iou:.6f}",
            "target_iou": float(target_iou),
            "other_iou": float(other_iou),
            "target_native_tid": None if target_obs is None else int(target_obs["native_tid"]),
            "other_native_tid": None if other_obs is None else int(other_obs["native_tid"]),
            "target_gt_box": np.asarray(target_box, dtype=float).tolist(),
            "other_gt_box": np.asarray(other_box, dtype=float).tolist(),
            "probe_rows": [[int(pid), np.asarray(box, dtype=float).tolist()] for pid, box in probe_rows],
        }
    if not probe_rows:
        return {"valid": False, "reason": "empty_current_public_probe"}

    def probed_pid(observation: dict[str, Any]) -> int:
        values = [box_iou(observation["box"], box) for _pid, box in probe_rows]
        return int(probe_rows[int(np.argmax(np.asarray(values, dtype=float)))][0])

    target_pid = probed_pid(target_obs)
    other_pid = probed_pid(other_obs)
    diagnostics = {
        "target_current_public_id": target_pid,
        "other_current_public_id": other_pid,
        "target_native_tid": int(target_obs["native_tid"]),
        "other_native_tid": int(other_obs["native_tid"]),
        "target_iou": float(target_iou),
        "other_iou": float(other_iou),
        "target_gt_box": np.asarray(target_box, dtype=float).tolist(),
        "other_gt_box": np.asarray(other_box, dtype=float).tolist(),
        "probe_rows": [[int(pid), np.asarray(box, dtype=float).tolist()] for pid, box in probe_rows],
    }
    if target_pid == other_pid:
        return {"valid": False, "reason": "target_other_share_current_public_id", **diagnostics}
    if target_pid == public_id(int(candidate["canonical_public_id"])):
        return {"valid": False, "reason": "target_already_canonical", **diagnostics}
    if other_pid == public_id(int(candidate["other_canonical_public_id"])):
        return {"valid": False, "reason": "other_already_canonical", **diagnostics}
    return {"valid": True, **diagnostics}


def consistency_audit(
    scan_path: Path,
    sequence: str,
    action: str,
    tape_root: Path,
    output: Path,
) -> dict[str, Any]:
    """Compare N37 static and N36 builder preconditions without selecting."""
    scan_payload = json.loads(scan_path.read_text(encoding="utf-8"))
    candidates = [
        item
        for item in scan_payload.get("candidates", [])
        if str(item.get("action_type")) == action
    ]
    candidates.sort(key=lambda item: (int(item["frame"]), str(item["candidate_id"])))
    rows = []
    for candidate in candidates:
        static = static_precondition(candidate, scan_payload, tape_root)
        reference = _reference_builder_atomic_precondition(candidate, scan_payload, tape_root)
        compare_fields = (
            "valid",
            "reason",
            "target_current_public_id",
            "other_current_public_id",
            "target_native_tid",
            "other_native_tid",
            "target_iou",
            "other_iou",
        )
        mismatches = [
            field
            for field in compare_fields
            if static.get(field) != reference.get(field)
        ]
        rows.append(
            {
                "candidate_id": str(candidate["candidate_id"]),
                "sequence": sequence,
                "frame": int(candidate["frame"]),
                "action_type": action,
                "candidate_fields": {
                    "dataset_gt_id": int(candidate["dataset_gt_id"]),
                    "other_dataset_gt_id": int(candidate["other_dataset_gt_id"]),
                    "canonical_public_id": public_id(int(candidate["canonical_public_id"])),
                    "other_canonical_public_id": public_id(int(candidate["other_canonical_public_id"])),
                    "n8_target_auto_tid": candidate.get("target_auto_tid"),
                    "n8_other_auto_tid": candidate.get("other_auto_tid"),
                },
                "static_precondition": static,
                "builder_reference_precondition": reference,
                "field_mismatches": mismatches,
                "replacement_selected": False,
                "pass_assigned": False,
            }
        )
    result = {
        "protocol": "N37_ATOMIC_PRECONDITION_CONSISTENCY_AUDIT_V1",
        "status": "AUDIT_ONLY",
        "selection_performed": False,
        "pass_assigned": False,
        "source_scan_artifact": display_path(scan_path),
        "source_tape": display_path(tape_root / f"{sequence}.jsonl"),
        "source_tape_sha256": digest_file(tape_root / f"{sequence}.jsonl"),
        "sequence": sequence,
        "action_type": action,
        "candidate_count": len(rows),
        "candidates": rows,
        "all_code_paths_match": all(not row["field_mismatches"] for row in rows),
        "all_candidates_rejected_by_builder_precondition": all(
            not row["builder_reference_precondition"].get("valid", False) for row in rows
        ),
        "legacy_static_bug": {
            "description": "The earlier static probe used pre_rows[other_auto_tid] as the other event-frame box; build_event uses offline event-frame GT for other_dataset_gt_id.",
            "fixed_in_script": True,
            "n36_builder_modified": False,
        },
        "runtime_future_gt_used": False,
        "replay_metrics_used": False,
        "protocol_or_quota_changed": False,
    }
    result["output"] = display_path(output)
    atomic_json(output, jsonable(result))
    print(json.dumps({"mode": "consistency_audit", **{key: result[key] for key in ("status", "candidate_count", "all_code_paths_match", "all_candidates_rejected_by_builder_precondition", "output")}}, sort_keys=True), flush=True)
    return result


def global_atomic_pool_audit(
    sequence_list_path: Path,
    tape_root: Path,
    selection_path: Path,
    output: Path,
) -> dict[str, Any]:
    """Audit every stored frozen atomic candidate without materializing it.

    The audit uses the predeclared sequence/frame/candidate-id order.  It
    records a deterministic replacement recommendation only; it does not
    alter ``selected_event_refs.json`` or assign any candidate PASS.
    """
    sequences = load_sequences(sequence_list_path)
    sequence_order = {sequence: index for index, sequence in enumerate(sequences)}
    pools: dict[str, dict[str, Any]] = {}
    all_candidates: list[dict[str, Any]] = []
    pool_counts: dict[str, Any] = {}
    for sequence in sequences:
        scan_path = SCAN_DIR / f"{sequence}.json"
        payload = json.loads(scan_path.read_text(encoding="utf-8"))
        candidates = [
            item for item in payload.get("candidates", [])
            if str(item.get("action_type")) == "ATOMIC_ID_SWAP"
        ]
        pools[sequence] = payload
        all_candidates.extend(candidates)
        pool_counts[sequence] = {
            "stored_pool_count": len(candidates),
            "eligible_action_count": int(payload.get("eligible_action_counts", {}).get("ATOMIC_ID_SWAP", 0)),
            "candidate_pool_truncated": bool(payload.get("candidate_pool_truncated", False)),
        }
    all_candidates.sort(key=lambda item: candidate_sort_key(item, sequence_order))
    frozen_selection = json.loads(selection_path.read_text(encoding="utf-8"))
    selected = frozen_selection.get("selected", [])
    selected_ids = {str(item.get("candidate_id")) for item in selected}
    selected_by_sequence: dict[str, list[int]] = {}
    for item in selected:
        selected_by_sequence.setdefault(str(item.get("sequence")), []).append(int(item["frame"]))
    failed_id = "n37-dancetrack0015-0772-atomic_id_swap-001"
    results: list[dict[str, Any]] = []
    reason_counts: Counter[str] = Counter()
    for index, candidate in enumerate(all_candidates, start=1):
        sequence = str(candidate["sequence"])
        scan_payload = pools[sequence]
        try:
            precondition = static_precondition(candidate, scan_payload, tape_root)
            audit_error = None
        except Exception as exc:
            precondition = {"valid": False, "reason": "audit_exception"}
            audit_error = f"{type(exc).__name__}: {exc}"
        same_sequence_window_ok = all(
            abs(int(candidate["frame"]) - existing_frame) > HORIZON
            for existing_frame in selected_by_sequence.get(sequence, [])
            if str(candidate["candidate_id"]) not in selected_ids
        )
        already_selected = str(candidate["candidate_id"]) in selected_ids
        is_other_sequence = sequence != "dancetrack0015"
        eligible = bool(
            precondition.get("valid", False)
            and str(candidate["candidate_id"]) != failed_id
            and is_other_sequence
            and not already_selected
            and same_sequence_window_ok
        )
        if eligible:
            reason = "DETERMINISTIC_REPLACEMENT_ELIGIBLE"
        elif audit_error:
            reason = "audit_exception"
        elif not precondition.get("valid", False):
            reason = str(precondition.get("reason", "precondition_rejected"))
        elif str(candidate["candidate_id"]) == failed_id:
            reason = "original_failed_candidate"
        elif not is_other_sequence:
            reason = "same_failed_sequence_excluded"
        elif already_selected:
            reason = "candidate_already_in_frozen_selection"
        elif not same_sequence_window_ok:
            reason = "future_window_overlap_with_frozen_event"
        else:
            reason = "not_eligible"
        reason_counts[reason] += 1
        results.append(
            {
                "global_order": index,
                "candidate_id": str(candidate["candidate_id"]),
                "sequence": sequence,
                "frame": int(candidate["frame"]),
                "action_type": "ATOMIC_ID_SWAP",
                "precondition": precondition,
                "audit_error": audit_error,
                "already_in_frozen_selection": already_selected,
                "other_than_failed_sequence": is_other_sequence,
                "same_sequence_future_window_compatible": same_sequence_window_ok,
                "replacement_eligible": eligible,
                "pass_assigned": False,
                "replay_metrics_used": False,
            }
        )
        if index % 64 == 0:
            print(json.dumps({"mode": "atomic_pool_audit", "audited": index, "total": len(all_candidates)}, sort_keys=True), flush=True)
    recommended = next((row for row in results if row["replacement_eligible"]), None)
    swapped_sequences = {str(item.get("sequence")) for item in selected if str(item.get("candidate_id")) != failed_id}
    if recommended is not None:
        swapped_sequences.add(str(recommended["sequence"]))
    result = {
        "protocol": "N37_GLOBAL_ATOMIC_POOL_AUDIT_V1",
        "status": "AUDIT_ONLY",
        "selection_performed": False,
        "pass_assigned": False,
        "source_sequence_list": display_path(sequence_list_path),
        "source_scan_dir": display_path(SCAN_DIR),
        "source_selection": display_path(selection_path),
        "sequence_count": len(sequences),
        "sequences": sequences,
        "stored_atomic_candidate_count": len(all_candidates),
        "sequences_with_stored_atomic_candidates": sum(value["stored_pool_count"] > 0 for value in pool_counts.values()),
        "pool_counts": pool_counts,
        "global_order": "sorted sequence order from frozen N34 list, then frame ascending, then candidate_id ascending",
        "prior_failed_command": {
            "status": "FAIL_RETAINED",
            "command_issue": "selected_sequences.json entries use 'sequence', not 'name'",
            "exception": "KeyError: 'name'",
            "no_candidate_audit_started": True,
            "no_protocol_artifact_modified": True,
        },
        "failed_slot": {
            "candidate_id": failed_id,
            "sequence": "dancetrack0015",
            "action_type": "ATOMIC_ID_SWAP",
            "replacement_scope": "other frozen sequence atomic pool only",
        },
        "reason_counts": dict(sorted(reason_counts.items())),
        "replacement_eligible_count": sum(bool(row["replacement_eligible"]) for row in results),
        "first_deterministic_replacement_recommendation": recommended,
        "replacement_would_preserve_action_count": True,
        "replacement_would_preserve_event_count": recommended is not None,
        "independent_sequence_count_after_replacement": len(swapped_sequences),
        "future_window_rule_preserved": recommended is None or bool(recommended["same_sequence_future_window_compatible"]),
        "candidates": results,
        "protocol_integrity": {
            "candidate_tape_changed": False,
            "checkpoint_changed": False,
            "action_quota_changed": False,
            "event_position_or_future_window_scored": False,
            "replay_metrics_used": False,
            "runtime_future_gt_used": False,
            "cross_sequence_event_composition": False,
            "candidate_pass_assigned": False,
        },
    }
    result["output"] = display_path(output)
    atomic_json(output, jsonable(result))
    print(json.dumps({
        "mode": "atomic_pool_audit",
        "status": result["status"],
        "sequence_count": result["sequence_count"],
        "stored_atomic_candidate_count": result["stored_atomic_candidate_count"],
        "replacement_eligible_count": result["replacement_eligible_count"],
        "recommendation": None if recommended is None else recommended["candidate_id"],
        "output": result["output"],
    }, sort_keys=True), flush=True)
    return result


def materialize_event(
    scan_payload: dict[str, Any],
    candidate_id: str,
    tape_root: Path,
    output: Path,
) -> dict[str, Any]:
    candidates = {
        str(item["candidate_id"]): item for item in scan_payload.get("candidates", [])
    }
    if candidate_id not in candidates:
        raise KeyError(f"candidate_id {candidate_id} absent from scan payload")
    candidate = candidates[candidate_id]
    sequence = str(scan_payload["sequence"])
    frame = int(candidate["frame"])
    path = tape_root / f"{sequence}.jsonl"
    if digest_file(path) != str(scan_payload["source_tape_sha256"]):
        raise RuntimeError(f"source tape changed after scan for {sequence}")
    # build_prefix expects the map at every frame before the event, so retain
    # the full child-produced map artifact but only convert keys here.
    map_before_by_frame = {
        int(current_frame): {int(native): int(pid) for native, pid in mapping.items()}
        for current_frame, mapping in scan_payload.get("map_before_by_frame", {}).items()
    }
    prefix = build_prefix(path, frame, map_before_by_frame)
    observations = current_rows(path, frame)
    gt_frames = load_gt(
        DanceTrackDataset(str(DATA_ROOT), sequences=[sequence], split="train"), sequence
    )
    info = {
        "event": copy.deepcopy(candidate["n8_event"]),
        "event_id": candidate_id,
        "action_type": str(candidate["action_type"]),
        "dataset_gt_id": int(candidate["dataset_gt_id"]),
        "frame": frame,
        "frame_count": int(scan_payload["frame_count"]),
        "pre_rows": [
            (int(pid), np.asarray(box, dtype=float))
            for pid, box in candidate.get("pre_rows", [])
        ],
    }
    item = build_n36_event(
        info,
        prefix,
        observations,
        gt_frames,
        sequence,
        DATA_ROOT / "train" / sequence / "img1",
        HumanFeatureExtractor(HUMAN_CHECKPOINT),
    )
    event = item["event"]
    event["event_id"] = candidate_id
    event["frame"] = frame
    event["n8_candidate_id"] = candidate_id
    event["n8_reported_frame"] = int(candidate["n8_event"].get("frame", frame))
    event["manual_box_source"] = "offline_train_GT_box_as_simulated_human_annotation"
    event["manual_box_is_synthetic"] = False
    event["runtime_future_gt_used"] = False
    event["future_gt_used_runtime"] = False
    event["source_tape_sha256"] = str(scan_payload["source_tape_sha256"])
    event["multi_identity_context"] = {
        "gt_visible_count": int(candidate["gt_visible_count"]),
        "candidate_count": int(candidate["candidate_count"]),
        "candidate_competitor_count": max(0, int(candidate["candidate_count"]) - 1),
        "protocol_requirement": "at least two visible GT identities and two current candidate rows",
    }
    # The N36 builder used a report-only offset for ADD events.  N37 uses the
    # actual public PID the full-loop manager will allocate after its current
    # frame births, so an empty-prefix ADD remains a valid transaction.
    if str(candidate["action_type"]) == "ADD_NEW_IDENTITY":
        config = StateManagerConfig(
            variant="reid",
            score_threshold=-100.0,
            max_lost_gap=90,
            use_appearance_memory=True,
            appearance_anchor_cap=8,
            appearance_negative_cap=16,
        )
        probe = _manager_from_prefix(prefix, frame, config, FEATURE_DIM)
        probe.rollout_frame(frame, observations, model=None)
        expected_pid = int(probe.next_pid)
        event["public_id"] = expected_pid
        event["canonical_public_id"] = expected_pid
        event["spatial_corrections"] = [
            dict(correction, public_id=expected_pid)
            for correction in event.get("spatial_corrections", [])
        ]
        del probe
    item.update(
        {
            "event": event,
            "source_tape": display_path(path),
            "source_tape_sha256": str(scan_payload["source_tape_sha256"]),
            "sequence_frame_count": int(scan_payload["frame_count"]),
            "prefix_range": [0, max(-1, frame - 1)],
            "future_window": [frame + 1, min(int(scan_payload["frame_count"]) - 1, frame + HORIZON)],
            "future_frame_start": frame + 1,
            "future_frame_end": min(int(scan_payload["frame_count"]) - 1, frame + HORIZON),
            "candidate_complete_source": True,
            "runtime_gt_used": False,
            "runtime_future_gt_used": False,
            "real_data": True,
            "synthetic": False,
            "action_type": str(candidate["action_type"]),
            "protocol_candidate_id": candidate_id,
            "multi_identity_context": event["multi_identity_context"],
            "human_embedding_source": {
                "type": "explicit_human_roi_image_crop",
                "checkpoint": str(HUMAN_CHECKPOINT),
                "feature_dim": FEATURE_DIM,
                "digest": event.get("human_feature_digest"),
                "machine_candidate_embedding_used_as_human": False,
            },
        }
    )
    atomic_json(output, jsonable(item))
    print(
        json.dumps(
            {
                "mode": "materialize",
                "sequence": sequence,
                "candidate_id": candidate_id,
                "action": candidate["action_type"],
                "status": "PASS",
                "output": display_path(output),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return item


def preflight_failure_evidence() -> dict[str, Any]:
    return {
        "status": "RETAINED_FAILURE_EVIDENCE",
        "scope": "pre-N37 read-only event discovery",
        "original_n36_artifacts_modified": False,
        "attempts": [
            {
                "attempt": "adhoc_all_sequence_survey_1",
                "observed_output_last_completed": "dancetrack0008",
                "traceback": None,
                "shell_epilogue": False,
                "exit_code": None,
                "fact": "process disappeared before dancetrack0012 and before summary; no PASS was assigned",
            },
            {
                "attempt": "isolated_dancetrack0012_diagnostic",
                "exit_code": 0,
                "traceback": None,
                "maxrss_kb": 398944,
                "fact": "single sequence completes normally in an isolated Python process",
            },
            {
                "attempt": "adhoc_first_five_sequence_reproduction",
                "observed_output_last_completed": "dancetrack0008",
                "traceback": None,
                "shell_epilogue": False,
                "exit_code": None,
                "last_reported_maxrss_kb": 385000,
                "fact": "same process-level disappearance reproduced before dancetrack0012",
            },
        ],
        "root_cause_assessment": {
            "category": "process_level_resource_termination",
            "evidence": "isolated dancetrack0012 exit 0 versus repeated multi-sequence disappearance without Python traceback or shell epilogue",
            "likely_mechanism": "N8 observer retained/allocator-held per-sequence temporal audit structures across one interpreter",
            "not_attributed_to": ["N36 real tape corruption", "SAM3 runtime", "future metric", "event selection outcome"],
        },
        "minimal_repair": {
            "implemented_by": "scripts/run_n37_event_protocol.py",
            "one_child_process_per_sequence": True,
            "child_retains_full_pre_rows": False,
            "child_retains_full_post_rows": False,
            "child_retains_observer_audit": False,
            "child_retains_verified_errors": False,
            "selection_or_replay_definition_changed": False,
        },
        "n37_repair_attempts": [
            {
                "attempt": "scan_smoke_attempt2",
                "sequence": "dancetrack0012",
                "status": "FAIL_NO_ARTIFACT",
                "traceback": None,
                "root_cause": "per-frame full gc.collect made the isolated child exceed the command window before final output",
            },
            {
                "attempt": "scan_smoke_attempt3",
                "sequence": "dancetrack0012",
                "status": "FAIL_NO_ARTIFACT",
                "traceback": None,
                "root_cause": "same per-frame full-GC slow path; no Python traceback",
            },
            {
                "attempt": "scan_smoke_final",
                "sequence": "dancetrack0012",
                "status": "PASS",
                "exit_code": 0,
                "artifact": "outputs/n37/event_candidates/dancetrack0012_smoke_final.json",
                "maxrss_kb": 369184,
            },
            {
                "attempt": "materialize_smoke",
                "sequence": "dancetrack0012",
                "status": "PASS",
                "exit_code": 0,
                "artifact": "outputs/n37/event_materialized/dancetrack0012_smoke_materialized.json",
                "action": "AUTHORITATIVE_REASSIGN",
                "human_crop_source_verified": True,
            },
        ],
    }


def candidate_sort_key(item: dict[str, Any], sequence_order: dict[str, int]) -> tuple[Any, ...]:
    return (
        int(sequence_order[str(item["sequence"])]),
        int(item["frame"]),
        str(item["candidate_id"]),
    )


def select_candidates(
    pools: dict[str, dict[str, Any]],
    sequences: list[str],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Select exactly one event per sequence where possible.

    The rule is frozen before materialization/replay: first satisfy four
    distinct-sequence slots for every action in fixed action order, then fill
    remaining sequence slots by sequence order and frame order.  No score,
    outcome, or future GT window is inspected.
    """
    order = {sequence: index for index, sequence in enumerate(sequences)}
    all_candidates = [
        candidate
        for sequence in sequences
        for candidate in pools[sequence].get("candidates", [])
    ]
    all_candidates.sort(key=lambda item: candidate_sort_key(item, order))
    selected: list[dict[str, Any]] = []
    selected_ids: set[str] = set()
    selected_sequences: set[str] = set()
    action_slots: list[dict[str, Any]] = []
    for action in ACTION_TYPES:
        action_selected = 0
        # Fill all four quota slots, preferring a new sequence for every slot.
        for candidate in all_candidates:
            if action_selected >= MIN_PER_ACTION:
                break
            if candidate["action_type"] != action:
                continue
            sequence = str(candidate["sequence"])
            if sequence in selected_sequences or candidate["candidate_id"] in selected_ids:
                continue
            selected.append(candidate)
            selected_ids.add(str(candidate["candidate_id"]))
            selected_sequences.add(sequence)
            action_selected += 1
            action_slots.append(
                {
                    "action": action,
                    "candidate_id": candidate["candidate_id"],
                    "sequence": sequence,
                    "slot_type": "minimum_action_quota",
                }
            )
    action_counts = Counter(item["action_type"] for item in selected)
    if any(action_counts[action] < MIN_PER_ACTION for action in ACTION_TYPES):
        raise RuntimeError(
            "frozen protocol cannot satisfy per-action quota: "
            + json.dumps(dict(action_counts), sort_keys=True)
        )
    # Fill to target with the earliest unused sequence candidates.  With the
    # N36 24-sequence source this normally yields one event per sequence.
    for candidate in all_candidates:
        if len(selected) >= TARGET_EVENTS:
            break
        if candidate["candidate_id"] in selected_ids:
            continue
        sequence = str(candidate["sequence"])
        if sequence in selected_sequences:
            continue
        selected.append(candidate)
        selected_ids.add(str(candidate["candidate_id"]))
        selected_sequences.add(sequence)
        action_slots.append(
            {
                "action": candidate["action_type"],
                "candidate_id": candidate["candidate_id"],
                "sequence": sequence,
                "slot_type": "unique_sequence_fill",
            }
        )
    # If some source sequence has no candidate, the target can still be met by
    # a deterministic second event only when its future window is disjoint.
    # This fallback is not expected for the selected 24 tapes; retain the rule
    # explicitly instead of silently duplicating a sequence.
    if len(selected) < TARGET_EVENTS:
        for candidate in all_candidates:
            if len(selected) >= TARGET_EVENTS:
                break
            if candidate["candidate_id"] in selected_ids:
                continue
            sequence = str(candidate["sequence"])
            frames = [int(item["frame"]) for item in selected if item["sequence"] == sequence]
            if frames and any(abs(int(candidate["frame"]) - frame) <= HORIZON for frame in frames):
                continue
            selected.append(candidate)
            selected_ids.add(str(candidate["candidate_id"]))
            action_slots.append(
                {
                    "action": candidate["action_type"],
                    "candidate_id": candidate["candidate_id"],
                    "sequence": sequence,
                    "slot_type": "disjoint_second_event_fallback",
                }
            )
    selected.sort(key=lambda item: candidate_sort_key(item, order))
    action_counts = Counter(item["action_type"] for item in selected)
    sequence_counts = Counter(str(item["sequence"]) for item in selected)
    if len(selected) != TARGET_EVENTS:
        raise RuntimeError(f"frozen protocol selected {len(selected)} instead of {TARGET_EVENTS}")
    if len(sequence_counts) < MIN_INDEPENDENT_SEQUENCES:
        raise RuntimeError(
            f"frozen protocol selected only {len(sequence_counts)} independent sequences"
        )
    return selected, {
        "selection_rule": {
            "candidate_source": "N8 verified interaction events from N36 real train tape",
            "sort": "sequence order, zero-based event frame, candidate_id",
            "minimum_action_quota": MIN_PER_ACTION,
            "target_event_count": TARGET_EVENTS,
            "minimum_independent_sequences": MIN_INDEPENDENT_SEQUENCES,
            "multi_identity_requirement": {
                "gt_visible_count_at_least": MIN_MULTI_GT,
                "candidate_count_at_least": MIN_MULTI_CANDIDATES,
            },
            "h100_required": True,
            "same_sequence_event_gap_at_least": HORIZON + 1,
            "future_metric_or_replay_used": False,
        },
        "action_slots": action_slots,
        "selected_action_counts": {action: int(action_counts[action]) for action in ACTION_TYPES},
        "selected_sequence_counts": {sequence: int(sequence_counts[sequence]) for sequence in sorted(sequence_counts)},
        "selected_candidate_ids": [str(item["candidate_id"]) for item in selected],
    }


def run_child_scan(sequence: str, tape_root: Path, output: Path) -> dict[str, Any]:
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--mode",
        "scan",
        "--sequence",
        sequence,
        "--tape-root",
        str(tape_root),
        "--output",
        str(output),
    ]
    environment = dict(os.environ)
    environment["PYTHONHASHSEED"] = "0"
    completed = subprocess.run(
        command,
        cwd=str(ROOT),
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"scan child failed for {sequence}, exit={completed.returncode}\n"
            f"stdout={completed.stdout[-4000:]}\n"
            f"stderr={completed.stderr[-4000:]}"
        )
    return json.loads(output.read_text(encoding="utf-8"))


def run_child_materialize(
    sequence: str,
    scan_path: Path,
    candidate_id: str,
    tape_root: Path,
    output: Path,
) -> dict[str, Any]:
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--mode",
        "materialize",
        "--sequence",
        sequence,
        "--scan-artifact",
        str(scan_path),
        "--candidate-id",
        candidate_id,
        "--tape-root",
        str(tape_root),
        "--output",
        str(output),
    ]
    environment = dict(os.environ)
    environment["PYTHONHASHSEED"] = "0"
    completed = subprocess.run(
        command,
        cwd=str(ROOT),
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"materializer child failed for {sequence}/{candidate_id}, exit={completed.returncode}\n"
            f"stdout={completed.stdout[-4000:]}\n"
            f"stderr={completed.stderr[-4000:]}"
        )
    return json.loads(output.read_text(encoding="utf-8"))


def validate_manifest(payload: dict[str, Any]) -> dict[str, Any]:
    events = payload.get("events", [])
    keys = [str(item.get("event", {}).get("event_id")) for item in events]
    action_counts = Counter(str(item.get("event", {}).get("action_type")) for item in events)
    sequence_counts = Counter(str(item.get("event", {}).get("sequence")) for item in events)
    issues = []
    if len(keys) != len(set(keys)):
        issues.append("duplicate_event_id")
    if len(events) != TARGET_EVENTS:
        issues.append(f"event_count:{len(events)}")
    if len(sequence_counts) < MIN_INDEPENDENT_SEQUENCES:
        issues.append(f"independent_sequence_count:{len(sequence_counts)}")
    for action in ACTION_TYPES:
        if action_counts[action] < MIN_PER_ACTION:
            issues.append(f"action_quota:{action}:{action_counts[action]}")
    for item in events:
        event = item.get("event", {})
        if item.get("runtime_future_gt_used") is not False:
            issues.append(f"{event.get('event_id')}:runtime_future_gt_used")
        if item.get("synthetic") is not False:
            issues.append(f"{event.get('event_id')}:synthetic")
        context = item.get("multi_identity_context", {})
        if int(context.get("gt_visible_count", 0)) < MIN_MULTI_GT:
            issues.append(f"{event.get('event_id')}:not_multi_gt")
        if int(context.get("candidate_count", 0)) < MIN_MULTI_CANDIDATES:
            issues.append(f"{event.get('event_id')}:not_multi_candidate")
        human = item.get("human_embedding_source", {})
        if human.get("machine_candidate_embedding_used_as_human") is not False:
            issues.append(f"{event.get('event_id')}:human_machine_substitute")
        feature = np.asarray(event.get("human_embedding", []), dtype=float)
        if feature.size != FEATURE_DIM or not np.all(np.isfinite(feature)) or np.linalg.norm(feature) <= 1e-6:
            issues.append(f"{event.get('event_id')}:human_feature_invalid")
    return {
        "valid": not issues,
        "issues": issues,
        "event_count": len(events),
        "independent_sequence_count": len(sequence_counts),
        "action_counts": {action: int(action_counts[action]) for action in ACTION_TYPES},
        "sequence_counts": {sequence: int(sequence_counts[sequence]) for sequence in sorted(sequence_counts)},
        "duplicate_event_id_count": len(keys) - len(set(keys)),
    }


def assemble(
    sequence_list_path: Path,
    tape_root: Path,
    manifest_output: Path,
    protocol_output: Path,
    stage_output: Path,
    selection_input: Path | None = None,
) -> dict[str, Any]:
    sequences = load_sequences(sequence_list_path)
    OUT.mkdir(parents=True, exist_ok=True)
    SCAN_DIR.mkdir(parents=True, exist_ok=True)
    MATERIALIZED_DIR.mkdir(parents=True, exist_ok=True)
    atomic_json(FAILURE_EVIDENCE, preflight_failure_evidence())
    pools: dict[str, dict[str, Any]] = {}
    failures: list[dict[str, Any]] = []
    for sequence in sequences:
        scan_path = SCAN_DIR / f"{sequence}.json"
        try:
            reusable = None
            if scan_path.is_file():
                try:
                    existing = json.loads(scan_path.read_text(encoding="utf-8"))
                    if (
                        existing.get("status") == "PASS"
                        and existing.get("sequence") == sequence
                        and existing.get("source_tape") == display_path(tape_root / f"{sequence}.jsonl")
                    ):
                        reusable = existing
                except (OSError, json.JSONDecodeError):
                    reusable = None
            pools[sequence] = reusable if reusable is not None else run_child_scan(sequence, tape_root, scan_path)
            print(
                json.dumps(
                    {
                        "stage": "A_scan",
                        "sequence": sequence,
                        "status": pools[sequence].get("status"),
                        "eligible": pools[sequence].get("candidate_count"),
                        "reused_existing": reusable is not None,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
        except Exception as exc:
            failure = {
                "sequence": sequence,
                "status": "FAIL",
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(),
            }
            failures.append(failure)
            atomic_json(OUT / f"event_scan_failure_{sequence}.json", failure)
            break
    if failures:
        stage = {
            "stage": "N37-A",
            "status": "BLOCKED",
            "real_data_status": "BLOCKED",
            "reason": "sequence_scan_failure",
            "failures": failures,
            "failure_evidence": display_path(FAILURE_EVIDENCE),
            "next_action": "retain traceback and repair only the first actionable scan root cause before materialization",
        }
        atomic_json(stage_output, stage)
        raise RuntimeError(json.dumps(stage, sort_keys=True))
    selection_artifact = SELECTION_FILE
    if selection_input is None:
        selected, selection_audit = select_candidates(pools, sequences)
        atomic_json(
            SELECTION_FILE,
            {
                "protocol": "N37_REAL_EVENT_SELECTION_V1",
                "status": "PASS",
                "selected": selected,
                "audit": selection_audit,
            },
        )
    else:
        selection_artifact = selection_input
        repaired_selection = json.loads(selection_input.read_text(encoding="utf-8"))
        selected = repaired_selection.get("selected", [])
        selection_audit = repaired_selection.get("audit", {})
        selected_ids = [str(item.get("candidate_id")) for item in selected]
        selected_counts = Counter(str(item.get("action_type")) for item in selected)
        if len(selected) != TARGET_EVENTS or len(selected_ids) != len(set(selected_ids)):
            raise RuntimeError("repaired selection is not exactly 24 unique candidates")
        if any(selected_counts[action] < MIN_PER_ACTION for action in ACTION_TYPES):
            raise RuntimeError(
                "repaired selection violates action quota: "
                + json.dumps(dict(selected_counts), sort_keys=True)
            )
        if len({str(item.get("sequence")) for item in selected}) < MIN_INDEPENDENT_SEQUENCES:
            raise RuntimeError("repaired selection violates independent sequence minimum")
        available_ids = {
            str(item.get("candidate_id"))
            for payload in pools.values()
            for item in payload.get("candidates", [])
        }
        missing_ids = sorted(set(selected_ids) - available_ids)
        if missing_ids:
            raise RuntimeError("repaired selection references missing scan candidates: " + json.dumps(missing_ids))
        selection_audit = dict(selection_audit)
        selection_audit["selection_input"] = display_path(selection_input)
        selection_audit["replay_metrics_used"] = False
        selection_audit["runtime_future_gt_used"] = False
    materialized: list[dict[str, Any]] = []
    materialization_failures: list[dict[str, Any]] = []
    for candidate in selected:
        sequence = str(candidate["sequence"])
        scan_path = SCAN_DIR / f"{sequence}.json"
        item_path = MATERIALIZED_DIR / f"{candidate['candidate_id']}.json"
        try:
            item = run_child_materialize(
                sequence,
                scan_path,
                str(candidate["candidate_id"]),
                tape_root,
                item_path,
            )
            materialized.append(item)
            print(
                json.dumps(
                    {
                        "stage": "A_materialize",
                        "sequence": sequence,
                        "candidate_id": candidate["candidate_id"],
                        "action": candidate["action_type"],
                        "status": "PASS",
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
        except Exception as exc:
            failure = {
                "candidate_id": candidate["candidate_id"],
                "sequence": sequence,
                "action_type": candidate["action_type"],
                "status": "FAIL",
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(),
            }
            materialization_failures.append(failure)
            atomic_json(OUT / f"event_materialization_failure_{candidate['candidate_id']}.json", failure)
            break
    if materialization_failures:
        stage = {
            "stage": "N37-A",
            "status": "BLOCKED",
            "real_data_status": "BLOCKED",
            "reason": "event_materialization_failure",
            "selection_audit": selection_audit,
            "failures": materialization_failures,
            "failure_evidence": display_path(FAILURE_EVIDENCE),
            "next_action": "retain the first materialization traceback and repair before full-loop",
        }
        atomic_json(stage_output, stage)
        raise RuntimeError(json.dumps(stage, sort_keys=True))
    events = sorted(materialized, key=lambda item: (str(item["event"]["sequence"]), int(item["event"]["frame"]), str(item["event"]["event_id"])))
    manifest = {
        "protocol": "N37_REAL_EVENT_TAPE_EXPANSION_V1",
        "status": "PASS",
        "split": "train/train_fold",
        "synthetic": False,
        "real_data_status": "PASS",
        "candidate_tape_source": "outputs/n36/real_tape/tape_manifest.json (read-only); per-sequence frame tapes",
        "candidate_tape_reused_without_reexport": True,
        "gt_used_only_offline_event_generation": True,
        "runtime_future_gt_used": False,
        "event_count": len(events),
        "independent_sequence_count": len({item["event"]["sequence"] for item in events}),
        "action_counts": {
            action: sum(item["event"]["action_type"] == action for item in events)
            for action in ACTION_TYPES
        },
        "events": events,
        "protocol_artifacts": {
            "event_protocol": display_path(protocol_output),
            "selection": display_path(selection_artifact),
            "scan_dir": display_path(SCAN_DIR),
            "materialized_dir": display_path(MATERIALIZED_DIR),
            "preflight_failure_evidence": display_path(FAILURE_EVIDENCE),
        },
        "human_feature": {
            "extractor": "HumanFeatureExtractor",
            "checkpoint": str(HUMAN_CHECKPOINT),
            "feature_dim": FEATURE_DIM,
            "source": "explicit real image crop from offline human GT box; no machine embedding substitution",
        },
        "selection_audit": selection_audit,
    }
    validation = validate_manifest(manifest)
    manifest["manifest_validation"] = validation
    if not validation["valid"]:
        manifest["status"] = "FAIL"
        manifest["real_data_status"] = "FAIL"
    atomic_json(manifest_output, jsonable(manifest))
    protocol = {
        "protocol": "N37_REAL_EVENT_SELECTION_V1",
        "status": "PASS" if validation["valid"] else "FAIL",
        "frozen_before_replay": True,
        "source_sequences": sequences,
        "source_tape_manifest": "outputs/n36/real_tape/tape_manifest.json",
        "source_split": "train/train_fold",
        "selection": selection_audit,
        "event_count": len(events),
        "independent_sequence_count": len({item["event"]["sequence"] for item in events}),
        "action_counts": manifest["action_counts"],
        "candidate_pool_counts": {
            sequence: pools[sequence].get("eligible_action_counts", {})
            for sequence in sequences
        },
        "future_metric_or_replay_used": False,
        "runtime_future_gt_used": False,
        "event_positions_frozen_from": "N8 event type/frame and tape completeness only; no N37 replay outcome",
        "nonoverlap": selection_audit["selection_rule"]["same_sequence_event_gap_at_least"],
        "failure_evidence": display_path(FAILURE_EVIDENCE),
    }
    atomic_json(protocol_output, jsonable(protocol))
    stage = {
        "stage": "N37-01",
        "status": "PASS" if validation["valid"] else "FAIL",
        "real_data_status": "PASS" if validation["valid"] else "FAIL",
        "event_count": len(events),
        "independent_sequence_count": len({item["event"]["sequence"] for item in events}),
        "action_counts": manifest["action_counts"],
        "candidate_tape_reused_without_reexport": True,
        "runtime_future_gt_used": False,
        "manifest_validation": validation,
        "artifacts": [display_path(protocol_output), display_path(manifest_output), display_path(FAILURE_EVIDENCE)],
        "next_action": "Run N37 real full-loop for every frozen event; no replay/training before full-loop PASS.",
    }
    atomic_json(stage_output, stage)
    print(json.dumps(stage, sort_keys=True), flush=True)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("scan", "materialize", "consistency_audit", "atomic_pool_audit", "assemble"), default="assemble")
    parser.add_argument("--sequence")
    parser.add_argument("--sequence-list", type=Path, default=DEFAULT_SEQUENCE_LIST)
    parser.add_argument("--tape-root", type=Path, default=DEFAULT_TAPE_ROOT)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--scan-artifact", type=Path)
    parser.add_argument("--candidate-id")
    parser.add_argument("--manifest-output", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--protocol-output", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--stage-output", type=Path, default=DEFAULT_STAGE)
    parser.add_argument("--selection-input", type=Path)
    args = parser.parse_args()
    if args.mode == "scan":
        if not args.sequence or args.output is None:
            parser.error("scan requires --sequence and --output")
        scan_sequence(args.sequence, args.tape_root, args.output)
        return
    if args.mode == "materialize":
        if not args.sequence or args.scan_artifact is None or not args.candidate_id or args.output is None:
            parser.error("materialize requires --sequence, --scan-artifact, --candidate-id and --output")
        payload = json.loads(args.scan_artifact.read_text(encoding="utf-8"))
        materialize_event(payload, args.candidate_id, args.tape_root, args.output)
        return
    if args.mode == "consistency_audit":
        if not args.sequence or args.scan_artifact is None or args.output is None:
            parser.error("consistency_audit requires --sequence, --scan-artifact and --output")
        consistency_audit(
            args.scan_artifact,
            args.sequence,
            "ATOMIC_ID_SWAP",
            args.tape_root,
            args.output,
        )
        return
    if args.mode == "atomic_pool_audit":
        if args.output is None:
            parser.error("atomic_pool_audit requires --output")
        global_atomic_pool_audit(
            args.sequence_list,
            args.tape_root,
            SELECTION_FILE,
            args.output,
        )
        return
    assemble(
        args.sequence_list,
        args.tape_root,
        args.manifest_output,
        args.protocol_output,
        args.stage_output,
        args.selection_input,
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"N37_EVENT_PROTOCOL_ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        traceback.print_exc()
        raise
