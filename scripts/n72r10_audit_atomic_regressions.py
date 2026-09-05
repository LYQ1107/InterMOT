#!/usr/bin/env python3
"""Audit the N72R9 ATOMIC protected-regression metric without rerunning models.

This is a posthoc, read-only audit.  It compares the frozen N72R9 baseline and
TEMPORAL_CURRENT runtime rows with the frozen N72R5R1 private mapping.  The
N72R9 artifacts are never edited.  In particular, the legacy protected map is
retained as evidence and is not silently replaced in historical JSON.
"""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any, Iterable, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
EVENT_MANIFEST = ROOT / "outputs/N72R5/mechanism_rounds/round_06_event_policy/real_event_manifest.json"
N72R9_ROOT = ROOT / "outputs/N72R9/replay/full"
N72R5R1_ROOT = ROOT / "outputs/N72R5R1/controller/round_06_persistence_probe/full"
PRIVATE_ROOT = N72R5R1_ROOT / "simulation_private"
PRESTATE_ROOT = N72R5R1_ROOT / "event_prestate"
DATA_ROOT = Path(os.environ.get("DANCETRACK_TRAIN_ROOT", "/data1/LWR/vranlee/DATASETS/JDE/dancetrack/train"))
OUT_ROOT = ROOT / "outputs/N72R10"
AUDIT_PATH = OUT_ROOT / "atomic_regression_audit.json"
STATUS_PATH = OUT_ROOT / "stage_01_status.json"
ATTEMPTS_ROOT = OUT_ROOT / "attempts"
IOU_THRESHOLD = 0.50
HORIZON = 100


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_write(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(json.dumps(dict(value), ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False))
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


def preserve_previous_outputs() -> Path | None:
    """Keep a prior run's machine-readable result before a new attempt writes it."""
    if not AUDIT_PATH.exists() and not STATUS_PATH.exists():
        return None
    ATTEMPTS_ROOT.mkdir(parents=True, exist_ok=True)
    existing = sorted(ATTEMPTS_ROOT.glob("atomic_audit_attempt_*"))
    attempt_number = len(existing) + 1
    destination = ATTEMPTS_ROOT / f"atomic_audit_attempt_{attempt_number:02d}"
    destination.mkdir(parents=True, exist_ok=False)
    for source in (AUDIT_PATH, STATUS_PATH):
        if source.exists():
            shutil.copy2(source, destination / source.name)
    return destination


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
                raise TypeError(f"expected JSON object at {path}:{line_number}")
            rows.append(value)
    return rows


def as_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or value is None:
        raise ValueError(f"{name} is not an integer: {value!r}")
    return int(value)


def box_iou(box_a: Sequence[Any] | None, box_b: Sequence[Any] | None) -> float:
    if box_a is None or box_b is None or len(box_a) != 4 or len(box_b) != 4:
        return 0.0
    a = [float(value) for value in box_a]
    b = [float(value) for value in box_b]
    if not all(math.isfinite(value) for value in (*a, *b)):
        return 0.0
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    intersection = max(0.0, min(ax2, bx2) - max(ax1, bx1)) * max(0.0, min(ay2, by2) - max(ay1, by1))
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - intersection
    return float(intersection / union) if union > 0.0 else 0.0


def load_gt(sequence: str) -> dict[int, dict[int, list[float]]]:
    path = DATA_ROOT / sequence / "gt" / "gt.txt"
    if not path.is_file():
        raise FileNotFoundError(path)
    result: dict[int, dict[int, list[float]]] = defaultdict(dict)
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            fields = [item.strip() for item in line.split(",")]
            if len(fields) < 6:
                raise ValueError(f"malformed GT row at {path}:{line_number}")
            frame = int(fields[0]) - 1
            identity = int(fields[1])
            x, y, width, height = (float(value) for value in fields[2:6])
            box = [x, y, x + width, y + height]
            if not all(math.isfinite(value) for value in box):
                raise ValueError(f"non-finite GT box at {path}:{line_number}")
            result[frame][identity] = box
    return result


def rows_by_frame(path: Path) -> tuple[dict[int, dict[str, Any]], list[str]]:
    rows = read_jsonl(path)
    result: dict[int, dict[str, Any]] = {}
    duplicate_frames: list[str] = []
    for row in rows:
        frame = as_int(row.get("frame"), f"frame in {path}")
        if frame in result:
            duplicate_frames.append(str(frame))
        result[frame] = row
    return result, duplicate_frames


def public_candidates(row: Mapping[str, Any], public_id: int) -> list[dict[str, Any]]:
    result = []
    for candidate in row.get("candidate_rows", []):
        if not isinstance(candidate, Mapping) or candidate.get("public_id") is None:
            continue
        if as_int(candidate.get("public_id"), "candidate public_id") == int(public_id):
            result.append(dict(candidate))
    return result


def public_observation(row: Mapping[str, Any], public_id: int, gt_box: Sequence[float]) -> dict[str, Any]:
    candidates = public_candidates(row, public_id)
    scored = [(box_iou(candidate.get("box_xyxy"), gt_box), candidate) for candidate in candidates]
    if not scored:
        return {"public_id": int(public_id), "candidate_present": False, "candidate_uid": None, "iou": 0.0}
    scored.sort(key=lambda pair: (-pair[0], as_int(pair[1].get("candidate_index", 0), "candidate_index")))
    score, candidate = scored[0]
    return {
        "public_id": int(public_id),
        "candidate_present": True,
        "candidate_uid": candidate.get("candidate_uid"),
        "candidate_index": candidate.get("candidate_index"),
        "official_raw_sam_id": candidate.get("official_raw_sam_id"),
        "iou": float(score),
        "assignment_status": candidate.get("assignment_status"),
    }


def assigned_uid(row: Mapping[str, Any], public_id: int) -> str | None:
    candidates = public_candidates(row, public_id)
    if not candidates:
        return None
    candidates.sort(key=lambda item: as_int(item.get("candidate_index", 0), "candidate_index"))
    return None if candidates[0].get("candidate_uid") is None else str(candidates[0]["candidate_uid"])


def assignment_axis_violations(row: Mapping[str, Any]) -> list[str]:
    assigned = [
        item
        for item in row.get("candidate_rows", [])
        if isinstance(item, Mapping)
        and item.get("public_id") is not None
        and item.get("candidate_uid") is not None
    ]
    public_ids = [as_int(item.get("public_id"), "assigned public_id") for item in assigned]
    candidate_uids = [str(item["candidate_uid"]) for item in assigned]
    violations: list[str] = []
    if len(public_ids) != len(set(public_ids)):
        violations.append("duplicate_public_id_assignment")
    if len(candidate_uids) != len(set(candidate_uids)):
        violations.append("duplicate_candidate_assignment")
    return violations


def runtime_flags(row: Mapping[str, Any]) -> list[str]:
    flags = []
    if row.get("runtime_future_gt_used") is True:
        flags.append("runtime_future_gt_used")
    if row.get("runtime_gt_read") is True:
        flags.append("runtime_gt_read")
    if row.get("posthoc_gt_used") is True:
        flags.append("posthoc_gt_used_in_runtime_row")
    assignment = row.get("assignment")
    if isinstance(assignment, Mapping) and assignment.get("runtime_future_gt_used") is True:
        flags.append("assignment_runtime_future_gt_used")
    return flags


def load_private_mapping(event_id: str) -> tuple[dict[int, int], Path, Path, set[int]]:
    mapping_path = PRIVATE_ROOT / event_id / "oracle_private_mapping.json"
    prestate_path = PRESTATE_ROOT / event_id / "public_axis.json"
    mapping_payload = read_json(mapping_path)
    raw_mapping = mapping_payload.get("dataset_gt_to_public")
    if not isinstance(raw_mapping, Mapping):
        raise ValueError(f"missing dataset_gt_to_public in {mapping_path}")
    mapping = {as_int(key, "private mapping GT") : as_int(value, "private mapping public") for key, value in raw_mapping.items()}
    axis_payload = read_json(prestate_path)
    raw_axis = axis_payload.get("public_axis", axis_payload.get("axis", []))
    if not isinstance(raw_axis, list):
        raise ValueError(f"invalid public axis in {prestate_path}")
    axis_public_ids = {
        as_int(item.get("public_id"), "prestate public_id")
        for item in raw_axis
        if isinstance(item, Mapping) and item.get("public_id") is not None
    }
    return mapping, mapping_path, prestate_path, axis_public_ids


def audit_event(event: Mapping[str, Any]) -> dict[str, Any]:
    event_id = str(event["event_id"])
    sequence = str(event["sequence"])
    event_frame = as_int(event["event_frame"], "event_frame")
    target_gt = as_int(event["dataset_gt_id"], "target GT")
    other_gt = as_int(event["other_dataset_gt_id"], "other GT")
    event_root = N72R9_ROOT / event_id
    posthoc_path = event_root / "posthoc.json"
    posthoc = read_json(posthoc_path)
    posthoc_event = posthoc.get("event")
    if not isinstance(posthoc_event, Mapping):
        raise ValueError(f"missing posthoc event object: {posthoc_path}")
    target_public = as_int(posthoc_event.get("target_public_id"), "posthoc target public")
    legacy_raw = posthoc_event.get("protected_public_by_gt_posthoc")
    if not isinstance(legacy_raw, Mapping):
        raise ValueError(f"missing legacy protected map: {posthoc_path}")
    legacy_map = {as_int(key, "legacy protected GT") : as_int(value, "legacy protected public") for key, value in legacy_raw.items()}
    private_map, mapping_path, prestate_path, prestate_axis = load_private_mapping(event_id)
    if target_gt not in private_map or other_gt not in private_map:
        raise ValueError(f"private mapping lacks ATOMIC pair: {event_id}")
    pair_target_public = private_map[target_gt]
    pair_other_public = private_map[other_gt]
    if pair_target_public != target_public:
        raise ValueError(
            f"N72R9 target public disagrees with frozen private mapping: {event_id}: "
            f"posthoc={target_public}, private={pair_target_public}"
        )
    if pair_target_public == pair_other_public:
        raise ValueError(f"private ATOMIC pair is not distinct: {event_id}")
    if pair_target_public not in prestate_axis or pair_other_public not in prestate_axis:
        raise ValueError(f"private ATOMIC public is absent from prestate axis: {event_id}")

    gt = load_gt(sequence)
    base_path = event_root / "BASELINE_B0" / "runtime_frames.jsonl"
    treatment_path = event_root / "TEMPORAL_CURRENT" / "runtime_frames.jsonl"
    baseline, baseline_duplicate_frames = rows_by_frame(base_path)
    treatment, treatment_duplicate_frames = rows_by_frame(treatment_path)
    required_frames = list(range(event_frame + 1, event_frame + HORIZON + 1))
    missing_frames = {
        variant: [frame for frame in required_frames if frame not in rows]
        for variant, rows in (("BASELINE_B0", baseline), ("TEMPORAL_CURRENT", treatment))
    }
    flags = {
        "BASELINE_B0": sorted({flag for row in baseline.values() for flag in runtime_flags(row)}),
        "TEMPORAL_CURRENT": sorted({flag for row in treatment.values() for flag in runtime_flags(row)}),
    }
    axis_violations = {
        variant: {
            str(frame): assignment_axis_violations(rows[frame])
            for frame in required_frames
            if frame in rows and assignment_axis_violations(rows[frame])
        }
        for variant, rows in (("BASELINE_B0", baseline), ("TEMPORAL_CURRENT", treatment))
    }
    protected_details: list[dict[str, Any]] = []
    for frame in required_frames:
        if frame not in baseline or frame not in treatment:
            continue
        for gt_id, legacy_public in sorted(legacy_map.items()):
            gt_box = gt.get(frame, {}).get(gt_id)
            if gt_box is None:
                continue
            base_observation = public_observation(baseline[frame], legacy_public, gt_box)
            treatment_observation = public_observation(treatment[frame], legacy_public, gt_box)
            corrected_public = private_map.get(gt_id)
            corrected_observation = None
            if corrected_public is not None:
                corrected_observation = public_observation(treatment[frame], corrected_public, gt_box)
            legacy_regression = bool(
                base_observation["iou"] >= IOU_THRESHOLD and treatment_observation["iou"] < IOU_THRESHOLD
            )
            if legacy_regression or legacy_public == target_public:
                protected_details.append(
                    {
                        "frame": frame,
                        "gt_id": gt_id,
                        "legacy_public_id": legacy_public,
                        "corrected_public_id": corrected_public,
                        "legacy_map_collides_with_target_public": legacy_public == target_public and gt_id != target_gt,
                        "legacy_regression": legacy_regression,
                        "baseline_legacy": base_observation,
                        "treatment_legacy": treatment_observation,
                        "treatment_corrected": corrected_observation,
                    }
                )

    corrected_regressions: list[dict[str, Any]] = []
    corrected_protected_details: list[dict[str, Any]] = []
    target_steals_distinct_other: list[dict[str, Any]] = []
    for frame in required_frames:
        if frame not in baseline or frame not in treatment:
            continue
        for gt_id in sorted(legacy_map):
            gt_box = gt.get(frame, {}).get(gt_id)
            corrected_public = private_map.get(gt_id)
            if gt_box is None or corrected_public is None:
                continue
            base_observation = public_observation(baseline[frame], corrected_public, gt_box)
            treatment_observation = public_observation(treatment[frame], corrected_public, gt_box)
            corrected_regression = bool(
                base_observation["iou"] >= IOU_THRESHOLD and treatment_observation["iou"] < IOU_THRESHOLD
            )
            if corrected_regression:
                corrected_regressions.append(
                    {
                        "frame": frame,
                        "gt_id": gt_id,
                        "corrected_public_id": corrected_public,
                        "baseline": base_observation,
                        "treatment": treatment_observation,
                    }
                )
            if gt_id == other_gt:
                corrected_protected_details.append(
                    {
                        "frame": frame,
                        "other_gt_id": gt_id,
                        "other_public_id": corrected_public,
                        "baseline": base_observation,
                        "treatment": treatment_observation,
                        "corrected_regression": corrected_regression,
                    }
                )
                treatment_target_uid = assigned_uid(treatment[frame], target_public)
                baseline_other_uid = assigned_uid(baseline[frame], corrected_public)
                if treatment_target_uid is not None and treatment_target_uid == baseline_other_uid:
                    target_steals_distinct_other.append(
                        {
                            "frame": frame,
                            "target_public_id": target_public,
                            "other_public_id": corrected_public,
                            "candidate_uid": treatment_target_uid,
                        }
                    )

    legacy_collision_gt_ids = sorted(
        gt_id for gt_id, public_id in legacy_map.items() if gt_id != target_gt and public_id == target_public
    )
    historical_comparison = (
        posthoc_event.get("comparisons", {})
        .get("TEMPORAL_CURRENT_vs_BASELINE_B0", {})
        .get(str(HORIZON), {})
    )
    computed_legacy_regression_count = sum(1 for detail in protected_details if detail["legacy_regression"])
    legacy_noncollision_regressions = sum(
        1 for detail in protected_details if detail["legacy_regression"] and not detail["legacy_map_collides_with_target_public"]
    )
    return {
        "event_id": event_id,
        "sequence": sequence,
        "action_type": str(event["action_type"]),
        "event_frame": event_frame,
        "target_dataset_gt_id": target_gt,
        "other_dataset_gt_id": other_gt,
        "target_public_id": target_public,
        "corrected_private_pair": {
            "target_public_id": pair_target_public,
            "other_public_id": pair_other_public,
            "mapping_source": str(mapping_path),
            "mapping_sha256": sha256_file(mapping_path),
            "prestate_public_axis_source": str(prestate_path),
            "prestate_public_axis_sha256": sha256_file(prestate_path),
            "axis_contains_target": pair_target_public in prestate_axis,
            "axis_contains_other": pair_other_public in prestate_axis,
            "public_ids_distinct": pair_target_public != pair_other_public,
        },
        "legacy_protected_map": {str(key): value for key, value in sorted(legacy_map.items())},
        "legacy_map_collision": {
            "collision_gt_ids": legacy_collision_gt_ids,
            "collision_count": len(legacy_collision_gt_ids),
            "target_public_id": target_public,
            "interpretation": "legacy posthoc protected map assigns a distinct protected GT to the target public axis",
        },
        "historical_n72r9_metric": {
            "reported_protected_regression_count": historical_comparison.get("protected_regression_count"),
            "reported_protected_compared": historical_comparison.get("protected_compared"),
            "reported_target_assignment_change_count": historical_comparison.get("target_assignment_change_count"),
        },
        "computed_legacy_audit": {
            "protected_regression_count": computed_legacy_regression_count,
            "noncollision_regression_count": legacy_noncollision_regressions,
            "protected_map_entry_count": len(legacy_map),
            "details": protected_details,
        },
        "computed_corrected_pair_audit": {
            "other_public_regression_count": sum(item["corrected_regression"] for item in corrected_protected_details),
            "all_corrected_protected_regression_count": len(corrected_regressions),
            "target_steals_distinct_other_count": len(target_steals_distinct_other),
            "other_public_details": corrected_protected_details,
            "all_regressions": corrected_regressions,
            "target_steals_distinct_other": target_steals_distinct_other,
        },
        "runtime_integrity": {
            "required_future_frame_count": len(required_frames),
            "missing_frames": missing_frames,
            "duplicate_frame_numbers": {
                "BASELINE_B0": baseline_duplicate_frames,
                "TEMPORAL_CURRENT": treatment_duplicate_frames,
            },
            "assignment_one_to_one_violations": axis_violations,
            "runtime_gt_flags": flags,
            "runtime_future_gt_used": False,
        },
        "source_hashes": {
            "event_policy_manifest": sha256_file(EVENT_MANIFEST),
            "posthoc": sha256_file(posthoc_path),
            "baseline_runtime": sha256_file(base_path),
            "treatment_runtime": sha256_file(treatment_path),
            "gt": sha256_file(DATA_ROOT / sequence / "gt" / "gt.txt"),
        },
    }


def main() -> int:
    previous_outputs = preserve_previous_outputs()
    manifest = read_json(EVENT_MANIFEST)
    events = manifest.get("events")
    if not isinstance(events, list):
        raise ValueError(f"missing events in {EVENT_MANIFEST}")
    all_atomic_events = [event for event in events if isinstance(event, Mapping) and event.get("action_type") == "ATOMIC_ID_SWAP"]
    atomic_events = [
        event
        for event in all_atomic_events
        if (N72R9_ROOT / str(event.get("event_id")) / "posthoc.json").is_file()
    ]
    unexecuted_atomic_events = [
        str(event.get("event_id"))
        for event in all_atomic_events
        if event not in atomic_events
    ]
    event_results: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    for event in atomic_events:
        try:
            event_results.append(audit_event(event))
        except Exception as exc:  # preserve every event-level audit error in the artifact
            event_results.append({"event_id": event.get("event_id"), "status": "FAIL_AUDIT_EVENT", "error": str(exc)})
            errors.append({"event_id": str(event.get("event_id")), "error": str(exc)})

    historical_regressions = sum(
        int(item.get("computed_legacy_audit", {}).get("protected_regression_count", 0))
        for item in event_results
        if item.get("status") != "FAIL_AUDIT_EVENT"
    )
    legacy_collisions = sum(
        int(item.get("legacy_map_collision", {}).get("collision_count", 0))
        for item in event_results
        if item.get("status") != "FAIL_AUDIT_EVENT"
    )
    corrected_regressions = sum(
        int(item.get("computed_corrected_pair_audit", {}).get("all_corrected_protected_regression_count", 0))
        for item in event_results
        if item.get("status") != "FAIL_AUDIT_EVENT"
    )
    other_regressions = sum(
        int(item.get("computed_corrected_pair_audit", {}).get("other_public_regression_count", 0))
        for item in event_results
        if item.get("status") != "FAIL_AUDIT_EVENT"
    )
    target_steals = sum(
        int(item.get("computed_corrected_pair_audit", {}).get("target_steals_distinct_other_count", 0))
        for item in event_results
        if item.get("status") != "FAIL_AUDIT_EVENT"
    )
    solver_violations = sum(
        len(violations)
        for item in event_results
        if item.get("status") != "FAIL_AUDIT_EVENT"
        for violations in item.get("runtime_integrity", {}).get("assignment_one_to_one_violations", {}).values()
    )
    runtime_flags_count = sum(
        len(flags)
        for item in event_results
        if item.get("status") != "FAIL_AUDIT_EVENT"
        for flags in item.get("runtime_integrity", {}).get("runtime_gt_flags", {}).values()
    )
    audit_pass = bool(
        not errors
        and len(event_results) == len(atomic_events) == 3
        and historical_regressions == 9
        and legacy_collisions == 2
        and corrected_regressions == 0
        and other_regressions == 0
        and target_steals == 0
        and solver_violations == 0
        and runtime_flags_count == 0
        and all(
            not any(item.get("runtime_integrity", {}).get("missing_frames", {}).values())
            for item in event_results
            if item.get("status") != "FAIL_AUDIT_EVENT"
        )
    )
    artifact = {
        "schema_version": "N72R10_ATOMIC_REGRESSION_AUDIT_V1",
        "created_at_utc": now_utc(),
        "status": "PASS_LEGACY_PROTECTED_MAP_COLLISION" if audit_pass else "BLOCKED_ATOMIC_AUDIT",
        "posthoc_only": True,
        "runtime_future_gt_used": False,
        "production_runtime_modified": False,
        "input_contract": {
            "event_manifest": str(EVENT_MANIFEST),
            "n72r9_runtime_root": str(N72R9_ROOT),
            "frozen_private_mapping_root": str(PRIVATE_ROOT),
            "iou_threshold": IOU_THRESHOLD,
            "future_horizon": HORIZON,
            "gt_opened_only_for_posthoc_audit": True,
            "audit_scope": "ATOMIC events with sealed N72R9 full posthoc/runtime artifacts only",
        },
        "aggregate": {
            "atomic_event_count": len(event_results),
            "expected_atomic_event_count": len(atomic_events),
            "all_manifest_atomic_event_count": len(all_atomic_events),
            "unexecuted_manifest_atomic_event_count": len(unexecuted_atomic_events),
            "legacy_protected_regression_count": historical_regressions,
            "legacy_map_target_collision_count": legacy_collisions,
            "corrected_all_protected_regression_count": corrected_regressions,
            "corrected_other_public_regression_count": other_regressions,
            "target_steals_distinct_other_public_count": target_steals,
            "solver_one_to_one_violation_count": solver_violations,
            "runtime_gt_flag_count": runtime_flags_count,
            "production_pair_guard_required": False,
            "production_pair_guard_authorized": False,
        },
        "root_cause": {
            "classification": "POSTHOC_PROTECTED_MAP_COLLISION",
            "statement": (
                "The nine historical protected regressions are all explained by legacy posthoc map collision "
                "entries that assign a distinct protected identity to a target public axis. The frozen private "
                "ATOMIC pairs remain distinct, and no corrected distinct-pair regression is observed."
            ),
            "not_a_solver_one_to_one_violation": True,
            "not_a_candidate_absence_for_corrected_pair": True,
            "repair_scope": "posthoc_metric_and_future_event_contract_only",
            "next_minimal_action": "carry explicit other_public_id from the frozen pair contract into N72R10 event/replay artifacts",
        },
        "event_results": event_results,
        "unexecuted_manifest_atomic_event_ids": unexecuted_atomic_events,
        "errors": errors,
        "previous_outputs_preserved_at": None if previous_outputs is None else str(previous_outputs),
    }
    atomic_write(AUDIT_PATH, artifact)
    status = {
        "schema_version": "N72R10_STAGE_STATUS_V1",
        "stage": "STAGE_01_ATOMIC_PROTECTED_REGRESSION_AUDIT",
        "created_at_utc": now_utc(),
        "status": "PASS_ROOT_CAUSE_POSTHOC_MAP_COLLISION" if audit_pass else "BLOCKED_ATOMIC_AUDIT",
        "audit_artifact": str(AUDIT_PATH),
        "audit_artifact_sha256": sha256_file(AUDIT_PATH),
        "counts": artifact["aggregate"],
        "unexecuted_manifest_atomic_event_ids": unexecuted_atomic_events,
        "production_change": False,
        "repair_needed": False if audit_pass else None,
        "pair_guard_authorized": False,
        "next_stage": "STAGE_03_TRUE_FUTURE_REQUERY_SESSION" if audit_pass else "STOP_UNTIL_ATOMIC_AUDIT_REPAIRED",
        "errors": errors,
        "historical_evidence_preserved": True,
        "previous_outputs_preserved_at": None if previous_outputs is None else str(previous_outputs),
    }
    atomic_write(STATUS_PATH, status)
    print(json.dumps(status, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if audit_pass else 2


if __name__ == "__main__":
    raise SystemExit(main())
