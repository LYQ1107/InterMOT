#!/usr/bin/env python3
"""Posthoc 40-event appearance separability audit for N72R5R1.

The audit consumes sealed Stage07 candidate features and the independent
Stage08 public-assignment sidecars.  GT is opened only here, after runtime
artifacts are complete, to identify the target candidate and never changes a
runtime assignment.  No model, threshold, candidate stream, or solver is
modified.
"""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import sys
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sam3_intermot.association.branch_public_replay import (  # noqa: E402
    BRANCHES,
    HORIZON,
    IOU_THRESHOLD,
    atomic_json,
    atomic_jsonl,
    best_candidate_for_box,
    load_gt,
    load_stage07_event_rows,
    read_json,
    sha256_file,
)


EVENT_MANIFEST = ROOT / "outputs/N72R5/mechanism_rounds/round_06_event_policy/real_event_manifest.json"
STAGE07_ROOT = ROOT / "outputs/N72R5/mechanism_rounds/round_07_official_full_loop_attempt5"
OUT = Path(os.environ.get("N72R5R1_RUN_ROOT", str(ROOT / "outputs/N72R5R1")))
STAGE08_MANIFEST = OUT / "stage08_runtime_manifest.json"
ROUND_ROOT = Path(
    os.environ.get(
        "N72R5R1_ROUND03_ROOT",
        str(OUT / "controller" / "round_03_feature_separability"),
    )
)
TABLE = ROUND_ROOT / "feature_pair_table.jsonl"
SUMMARY = ROUND_ROOT / "feature_separability_summary.json"
STATUS = ROUND_ROOT / "round_03_status.json"

TREATMENT_BRANCHES = list(BRANCHES[1:])
HORIZONS = (1, 20, 50, 100)
TEMPORAL_K = 5


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _cosine(left: Sequence[float] | None, right: Sequence[float] | None) -> float | None:
    if left is None or right is None:
        return None
    a = np.asarray(left, dtype=np.float64).reshape(-1)
    b = np.asarray(right, dtype=np.float64).reshape(-1)
    if a.size == 0 or a.size != b.size or not np.all(np.isfinite(a)) or not np.all(np.isfinite(b)):
        return None
    an = float(np.linalg.norm(a))
    bn = float(np.linalg.norm(b))
    if an <= 1.0e-12 or bn <= 1.0e-12:
        return None
    return float(np.dot(a, b) / (an * bn))


def _finite_rate(values: Sequence[bool | None]) -> float | None:
    defined = [bool(value) for value in values if value is not None]
    return None if not defined else float(sum(defined) / len(defined))


def _quantiles(values: Sequence[float]) -> dict[str, float | None]:
    if not values:
        return {"median": None, "p90": None, "p95": None, "max": None}
    array = np.asarray(values, dtype=np.float64)
    return {
        "median": float(np.median(array)),
        "p90": float(np.quantile(array, 0.90)),
        "p95": float(np.quantile(array, 0.95)),
        "max": float(np.max(array)),
    }


def _done(root: Path, event_id: str, branch: str) -> dict[str, Any]:
    return read_json(root / "public_assignment" / event_id / f"{branch}.done.json")


def _assignment_map(row: Mapping[str, Any]) -> dict[str, int | None]:
    result: dict[str, int | None] = {}
    for item in row.get("solver", {}).get("assignment_rows", []):
        uid = str(item["candidate_uid"])
        value = item.get("public_id")
        result[uid] = None if value is None else int(value)
    return result


def _feature_map(candidates: Iterable[Mapping[str, Any]]) -> dict[str, np.ndarray]:
    result: dict[str, np.ndarray] = {}
    for candidate in candidates:
        uid = str(candidate["candidate_uid"])
        if uid in result:
            raise ValueError(f"duplicate candidate feature UID: {uid}")
        feature = np.asarray(candidate["feature"], dtype=np.float64).reshape(-1)
        if feature.size != 512 or not np.all(np.isfinite(feature)) or float(np.linalg.norm(feature)) <= 1.0e-12:
            raise ValueError(f"invalid candidate feature: {uid}")
        result[uid] = feature
    return result


def _snapshot_features(snapshot: Mapping[str, Any]) -> dict[int, np.ndarray]:
    result: dict[int, np.ndarray] = {}
    for item in snapshot.get("identities", []):
        public = item.get("public_id")
        feature = item.get("appearance_state", {}).get("last_machine_feature")
        if public is None or feature is None:
            continue
        array = np.asarray(feature, dtype=np.float64).reshape(-1)
        if array.size == 512 and np.all(np.isfinite(array)) and float(np.linalg.norm(array)) > 1.0e-12:
            result[int(public)] = array
    return result


def _history_feature(
    frame: int,
    event_frame: int,
    public_id: int | None,
    sidecar_by_frame: Mapping[int, Mapping[str, Any]],
    feature_by_frame: Mapping[int, Mapping[str, np.ndarray]],
) -> np.ndarray | None:
    if public_id is None:
        return None
    values: list[np.ndarray] = []
    for previous in range(int(frame) - 1, max(int(event_frame) - 1, int(frame) - TEMPORAL_K - 1), -1):
        row = sidecar_by_frame.get(previous)
        if row is None:
            continue
        assignment = _assignment_map(row)
        uid = next((candidate_uid for candidate_uid, value in assignment.items() if value == int(public_id)), None)
        if uid is None:
            continue
        feature = feature_by_frame.get(previous, {}).get(uid)
        if feature is not None:
            values.append(feature)
    if not values:
        return None
    mean = np.mean(np.stack(values, axis=0), axis=0)
    norm = float(np.linalg.norm(mean))
    return None if norm <= 1.0e-12 else mean / norm


def _score_at_public(row: Mapping[str, Any], candidate_index: int, public_id: int) -> float | None:
    public_axis = [int(value) for value in row.get("solver", {}).get("public_id_axis", [])]
    try:
        column = public_axis.index(int(public_id))
    except ValueError:
        return None
    matrix = np.asarray(row.get("fused_score_matrix", []), dtype=np.float64)
    if matrix.ndim != 2 or candidate_index < 0 or candidate_index >= matrix.shape[0] or column >= matrix.shape[1]:
        return None
    value = float(matrix[candidate_index, column])
    return value if math.isfinite(value) else None


def _candidate_index_by_uid(row: Mapping[str, Any]) -> dict[str, int]:
    return {str(item["candidate_uid"]): int(item["candidate_index"]) for item in row.get("candidate_rows", [])}


def _pair_row(
    event: Mapping[str, Any],
    branch: str,
    frame: int,
    target_public: int,
    target_uid: str,
    target_iou: float,
    target_feature: np.ndarray,
    competitor_uid: str,
    competitor_feature: np.ndarray,
    competitor_source: str,
    competitor_public: int | None,
    sidecar: Mapping[str, Any],
    anchor: np.ndarray | None,
    target_prototype: np.ndarray | None,
    competitor_prototype: np.ndarray | None,
    target_history: np.ndarray | None,
    competitor_history: np.ndarray | None,
    assignment: Mapping[str, int | None],
) -> dict[str, Any]:
    target_assignment = assignment.get(target_uid)
    target_index = _candidate_index_by_uid(sidecar).get(target_uid, -1)
    competitor_index = _candidate_index_by_uid(sidecar).get(competitor_uid, -1)
    target_score = _score_at_public(sidecar, target_index, target_public)
    competitor_score = _score_at_public(sidecar, competitor_index, target_public)
    fused_gap = None if target_score is None or competitor_score is None else float(target_score - competitor_score)
    anchor_target = _cosine(target_feature, anchor)
    anchor_competitor = _cosine(competitor_feature, anchor)
    prototype_target = _cosine(target_feature, target_prototype)
    prototype_competitor = _cosine(competitor_feature, target_prototype)
    temporal_target = _cosine(target_feature, target_history)
    temporal_competitor = _cosine(competitor_feature, competitor_history)
    return {
        "event_id": str(event["event_id"]),
        "sequence": str(event["sequence"]),
        "action_type": str(event["action_type"]),
        "branch": branch,
        "event_frame": int(event["event_frame"]),
        "frame": int(frame),
        "horizon": int(frame - int(event["event_frame"])),
        "target_gt_id": int(event["dataset_gt_id"]),
        "target_public_id": int(target_public),
        "target_candidate_uid": target_uid,
        "target_candidate_iou": float(target_iou),
        "target_candidate_present": bool(target_iou >= IOU_THRESHOLD),
        "target_assigned_public_id": target_assignment,
        "target_correctly_assigned": bool(target_assignment == int(target_public)),
        "competitor_candidate_uid": competitor_uid,
        "competitor_source": competitor_source,
        "competitor_assigned_public_id": competitor_public,
        "target_anchor_cosine": anchor_target,
        "competitor_anchor_cosine": anchor_competitor,
        "anchor_direction_correct": None if anchor_target is None or anchor_competitor is None else bool(anchor_target > anchor_competitor),
        "target_prototype_cosine": prototype_target,
        "competitor_prototype_cosine": prototype_competitor,
        "prototype_direction_correct": None if prototype_target is None or prototype_competitor is None else bool(prototype_target > prototype_competitor),
        "target_temporal_cosine": temporal_target,
        "competitor_temporal_cosine": temporal_competitor,
        "temporal_direction_correct": None if temporal_target is None or temporal_competitor is None else bool(temporal_target > temporal_competitor),
        "base_or_fused_target_column_gap": fused_gap,
        "target_candidate_feature_sha256": str(sidecar["candidate_rows"][target_index]["feature_sha256"]) if target_index >= 0 else None,
        "competitor_feature_sha256": str(sidecar["candidate_rows"][competitor_index]["feature_sha256"]) if competitor_index >= 0 else None,
        "runtime_future_gt_used": False,
        "posthoc_gt_used": True,
        "human_anchor_semantics": "stage07_corrected_candidate_feature_not_explicit_human_roi_embedding",
    }


def _summarize(rows: Sequence[Mapping[str, Any]], key_fields: Sequence[str]) -> dict[str, Any]:
    groups: dict[tuple[Any, ...], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[tuple(row.get(field) for field in key_fields)].append(row)
    result: dict[str, Any] = {}
    for key, group in sorted(groups.items(), key=lambda item: str(item[0])):
        label = "|".join(f"{field}={value}" for field, value in zip(key_fields, key))
        gaps = [float(row["base_or_fused_target_column_gap"]) for row in group if row.get("base_or_fused_target_column_gap") is not None]
        result[label] = {
            "key": {field: value for field, value in zip(key_fields, key)},
            "pair_count": len(group),
            "target_present_count": sum(bool(row.get("target_candidate_present")) for row in group),
            "target_present_wrong_assignment_count": sum(bool(row.get("target_candidate_present")) and not bool(row.get("target_correctly_assigned")) for row in group),
            "anchor_direction_rate": _finite_rate([row.get("anchor_direction_correct") for row in group]),
            "prototype_direction_rate": _finite_rate([row.get("prototype_direction_correct") for row in group]),
            "temporal_direction_rate": _finite_rate([row.get("temporal_direction_correct") for row in group]),
            "gap_quantiles": _quantiles(gaps),
        }
    return result


def main() -> int:
    manifest = read_json(STAGE08_MANIFEST)
    if manifest.get("status") != "PASS_N72R5R1_EXACT_PUBLIC_ASSOCIATION" or manifest.get("failures"):
        raise RuntimeError("Stage08 must be a complete PASS before posthoc separability audit")
    events = [dict(item) for item in read_json(EVENT_MANIFEST).get("events", [])]
    if len(events) != 40:
        raise RuntimeError(f"expected 40 frozen events, got {len(events)}")

    rows: list[dict[str, Any]] = []
    unavailable: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    events_audited: set[str] = set()
    branches_audited: set[tuple[str, str]] = set()

    for event in sorted(events, key=lambda item: str(item["event_id"])):
        event_id = str(event["event_id"])
        sequence = str(event["sequence"])
        event_frame = int(event["event_frame"])
        try:
            gt_frames = load_gt(sequence)
            target_gt_id = int(event["dataset_gt_id"])
            target_public_done = _done(OUT, event_id, TREATMENT_BRANCHES[0])
            target_public_value = target_public_done.get("target_public_id")
            if target_public_value is None:
                unavailable.append({"event_id": event_id, "reason": "target_public_unresolved"})
                continue
            target_public = int(target_public_value)
            snapshot = read_json(OUT / "event_prestate" / event_id / "persistent_runtime_snapshot.json")
            snapshot_features = _snapshot_features(snapshot)
            target_prototype = snapshot_features.get(target_public)
            stage_rows, _, _ = load_stage07_event_rows(STAGE07_ROOT, event_id, event_frame, sequence)
            for branch in TREATMENT_BRANCHES:
                done = _done(OUT, event_id, branch)
                sidecar_rows = _read_jsonl(Path(done["output"]))
                sidecar_by_frame = {int(row["frame"]): row for row in sidecar_rows}
                if set(sidecar_by_frame) != set(range(event_frame, event_frame + HORIZON + 1)):
                    raise ValueError(f"sidecar frame coverage mismatch: {event_id}/{branch}")
                feature_by_frame = {int(frame): _feature_map(candidates) for frame, candidates in stage_rows[branch].items()}
                anchor_uid = done.get("target_candidate_uid")
                anchor = feature_by_frame[event_frame].get(str(anchor_uid)) if anchor_uid is not None else None
                branches_audited.add((event_id, branch))
                for frame in range(event_frame + 1, event_frame + HORIZON + 1):
                    sidecar = sidecar_by_frame[frame]
                    candidates = stage_rows[branch][frame]
                    target_box = gt_frames.get(frame, {}).get(target_gt_id, {}).get("box")
                    if target_box is None:
                        # The posthoc target is not visible in this frame;
                        # absence is a valid observation, not an invalid box.
                        continue
                    target_iou, target_candidate = best_candidate_for_box(candidates, target_box or [])
                    if target_candidate is None or target_iou < IOU_THRESHOLD:
                        continue
                    target_uid = str(target_candidate["candidate_uid"])
                    target_feature = np.asarray(target_candidate["feature"], dtype=np.float64)
                    assignment = _assignment_map(sidecar)
                    assigned_target_uid = next((uid for uid, value in assignment.items() if value == target_public), None)
                    candidate_index = _candidate_index_by_uid(sidecar)
                    public_axis = [int(value) for value in sidecar.get("solver", {}).get("public_id_axis", [])]
                    competitor_uid: str | None = None
                    competitor_source: str | None = None
                    if assigned_target_uid is not None and assigned_target_uid != target_uid:
                        competitor_uid = assigned_target_uid
                        competitor_source = "actual_solver_target_public_owner"
                    elif target_uid in candidate_index and target_public in public_axis:
                        matrix = np.asarray(sidecar.get("fused_score_matrix", []), dtype=np.float64)
                        column = public_axis.index(target_public)
                        ranked = [
                            (float(matrix[index, column]), uid)
                            for uid, index in candidate_index.items()
                            if uid != target_uid and matrix.ndim == 2 and index < matrix.shape[0] and column < matrix.shape[1]
                        ]
                        ranked.sort(key=lambda item: (-item[0], item[1]))
                        if ranked:
                            competitor_uid = ranked[0][1]
                            competitor_source = "best_alternative_target_public_column"
                    if competitor_uid is None:
                        continue
                    competitor_feature = feature_by_frame[frame].get(competitor_uid)
                    if competitor_feature is None:
                        continue
                    competitor_public = assignment.get(competitor_uid)
                    target_history = _history_feature(frame, event_frame, target_public, sidecar_by_frame, feature_by_frame)
                    competitor_history = _history_feature(frame, event_frame, competitor_public, sidecar_by_frame, feature_by_frame)
                    competitor_prototype = snapshot_features.get(int(competitor_public)) if competitor_public is not None else None
                    rows.append(
                        _pair_row(
                            event,
                            branch,
                            frame,
                            target_public,
                            target_uid,
                            float(target_iou),
                            target_feature,
                            competitor_uid,
                            competitor_feature,
                            str(competitor_source),
                            competitor_public,
                            sidecar,
                            anchor,
                            target_prototype,
                            competitor_prototype,
                            target_history,
                            competitor_history,
                            assignment,
                        )
                    )
                events_audited.add(event_id)
        except Exception as exc:
            errors.append({"event_id": event_id, "sequence": sequence, "failure_type": type(exc).__name__, "message": str(exc)})

    overall = {
        "pair_count": len(rows),
        "target_present_count": sum(bool(row["target_candidate_present"]) for row in rows),
        "target_present_wrong_assignment_count": sum(bool(row["target_candidate_present"]) and not bool(row["target_correctly_assigned"]) for row in rows),
        "anchor_direction_rate": _finite_rate([row.get("anchor_direction_correct") for row in rows]),
        "prototype_direction_rate": _finite_rate([row.get("prototype_direction_correct") for row in rows]),
        "temporal_direction_rate": _finite_rate([row.get("temporal_direction_correct") for row in rows]),
        "gap_quantiles": _quantiles([float(row["base_or_fused_target_column_gap"]) for row in rows if row.get("base_or_fused_target_column_gap") is not None]),
    }
    summary = {
        "schema_version": "N72R5R1_ROUND03_FEATURE_SEPARABILITY_SUMMARY_V1",
        "status": "PASS_FEATURE_SEPARABILITY_AUDIT" if not errors else "BLOCKED_FEATURE_SEPARABILITY_AUDIT",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "input": {
            "event_manifest_sha256": sha256_file(EVENT_MANIFEST),
            "stage07_manifest_sha256": sha256_file(STAGE07_ROOT / "official_full_loop_manifest.json"),
            "stage08_manifest_sha256": sha256_file(STAGE08_MANIFEST),
            "events_expected": 40,
            "events_audited": len(events_audited),
            "branches_audited": len(branches_audited),
            "horizon_frames": HORIZON,
        },
        "overall": overall,
        "by_horizon": _summarize(rows, ("horizon",)),
        "by_action": _summarize(rows, ("action_type",)),
        "by_branch": _summarize(rows, ("branch",)),
        "by_sequence": _summarize(rows, ("sequence",)),
        "unavailable_events": unavailable,
        "errors": errors,
        "semantics": {
            "posthoc_gt_used": True,
            "runtime_future_gt_used": False,
            "feature_source": "sealed Stage07 normalized candidate feature vectors",
            "anchor_source": "stage07 corrected candidate feature; no explicit human ROI embedding was present",
            "target_candidate_rule": "maximum current-frame target GT box IoU, posthoc only",
            "competitor_rule": "actual solver owner of target public ID, otherwise highest target-public-column alternative",
            "temporal_rule": f"mean normalized feature of up to previous {TEMPORAL_K} solver-assigned frames within the event window",
        },
    }
    status = {
        "schema_version": "N72R5R1_ROUND03_STATUS_V1",
        "stage": "03_FEATURE_SEPARABILITY_AUDIT",
        "status": summary["status"],
        "round": "round_03_feature_separability",
        "summary": str(SUMMARY),
        "table": str(TABLE),
        "event_count_expected": 40,
        "event_count_audited": len(events_audited),
        "branch_count_audited": len(branches_audited),
        "pair_count": len(rows),
        "unavailable_event_count": len(unavailable),
        "error_count": len(errors),
        "posthoc_gt_used": True,
        "runtime_future_gt_used": False,
        "next_routing": "TVC_V1_SMALL_VERIFIER_ONLY_IF_SEPARABILITY_SUPPORTS" if not errors else "REPAIR_ROUND03_AUDIT",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    atomic_jsonl(TABLE, rows)
    atomic_json(SUMMARY, summary)
    atomic_json(STATUS, status)
    print(json.dumps({"status": summary["status"], "pairs": len(rows), "events": len(events_audited), "branches": len(branches_audited), "unavailable": len(unavailable), "errors": len(errors), "output": str(SUMMARY)}, ensure_ascii=False))
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
