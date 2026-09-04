#!/usr/bin/env python3
"""Posthoc audit of spatial-correction persistence and candidate-stream drift.

The audit consumes the repaired N72R5R1 public-assignment sidecars and the
sealed Stage07 feature streams.  GT is used only here to label the already
completed runtime rows.  No runtime map, candidate stream, solver, or
checkpoint is changed.
"""

from __future__ import annotations

from collections import Counter
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
    box_iou,
    load_gt,
    load_stage07_event_rows,
    read_json,
    read_jsonl,
    sha256_file,
)


EVENT_MANIFEST = ROOT / "outputs/N72R5/mechanism_rounds/round_06_event_policy/real_event_manifest.json"
DEFAULT_RUN_ROOT = ROOT / "outputs/N72R5R1/controller/round_05_branch_isolation/full"
RUN_ROOT = Path(os.environ.get("N72R5R1_PERSISTENCE_RUN_ROOT", str(DEFAULT_RUN_ROOT)))
STAGE07_ROOT = ROOT / "outputs/N72R5/mechanism_rounds/round_07_official_full_loop_attempt5"
AUDIT_ROOT = Path(
    os.environ.get(
        "N72R5R1_ROUND06_ROOT",
        str(ROOT / "outputs/N72R5R1/controller/round_06_persistence_audit"),
    )
)
TABLE = AUDIT_ROOT / "persistence_event_table.jsonl"
SUMMARY = AUDIT_ROOT / "persistence_audit_summary.json"
STATUS = AUDIT_ROOT / "round_06_status.json"

B0 = "B0_NO_INTERVENTION"
B1 = "B1_SPATIAL_CORRECTION_ONLY"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def _rows(path: Path) -> dict[int, dict[str, Any]]:
    rows = read_jsonl(path)
    result = {int(row["frame"]): row for row in rows}
    if len(result) != HORIZON + 1:
        raise ValueError(f"expected {HORIZON + 1} frame rows: {path}")
    return result


def _branch_output(event_result: Mapping[str, Any], branch: str) -> Path:
    value = next((item for item in event_result.get("branches", []) if str(item.get("branch")) == branch), None)
    if value is None:
        raise ValueError(f"missing branch result: {event_result.get('event_id')}/{branch}")
    return Path(str(value["output"]))


def _branch_diagnostic(event_result: Mapping[str, Any], branch: str) -> dict[str, Any]:
    value = next((item for item in event_result.get("branches", []) if str(item.get("branch")) == branch), None)
    if value is None:
        raise ValueError(f"missing branch result: {event_result.get('event_id')}/{branch}")
    diagnostic = value.get("treatment_diagnostic")
    return dict(diagnostic) if isinstance(diagnostic, Mapping) else {}


def _best_target(row: Mapping[str, Any], box: Sequence[float] | None) -> dict[str, Any]:
    if box is None:
        return {"visible": False, "present": False, "iou": None, "candidate": None}
    best_iou = -1.0
    best: Mapping[str, Any] | None = None
    for candidate in row.get("candidate_rows", []):
        value = float(box_iou(candidate.get("box_xyxy"), box))
        index = int(candidate.get("candidate_index", 0))
        best_index = None if best is None else int(best.get("candidate_index", 0))
        if value > best_iou or (value == best_iou and (best is None or index < int(best_index))):
            best_iou = value
            best = candidate
    return {
        "visible": True,
        "present": bool(best is not None and best_iou >= IOU_THRESHOLD),
        "iou": float(max(0.0, best_iou)),
        "candidate": None if best is None else dict(best),
    }


def _assigned_target(row: Mapping[str, Any], public_id: int) -> dict[str, Any] | None:
    matches = [item for item in row.get("candidate_rows", []) if item.get("public_id") is not None and int(item["public_id"]) == int(public_id)]
    if len(matches) > 1:
        raise ValueError(f"duplicate public assignment in frame {row.get('event_id')}/{row.get('branch')}/{row.get('frame')}: {public_id}")
    return None if not matches else dict(matches[0])


def _frame_eval(row: Mapping[str, Any], box: Sequence[float] | None, public_id: int) -> dict[str, Any]:
    target = _best_target(row, box)
    assigned = _assigned_target(row, public_id)
    assigned_iou = None if assigned is None or box is None else float(box_iou(assigned.get("box_xyxy"), box))
    target_candidate = target.get("candidate")
    target_uid = None if target_candidate is None else str(target_candidate.get("candidate_uid"))
    assigned_uid = None if assigned is None else str(assigned.get("candidate_uid"))
    return {
        "visible": bool(target["visible"]),
        "candidate_present": bool(target["present"]),
        "target_candidate_iou": target.get("iou"),
        "target_candidate_uid": target_uid,
        "target_candidate_public_id": None if target_candidate is None or target_candidate.get("public_id") is None else int(target_candidate["public_id"]),
        "target_public_assigned_uid": assigned_uid,
        "target_public_assigned_iou": assigned_iou,
        "target_public_assigned": assigned is not None,
        "target_correct": bool(target["present"] and target_candidate is not None and target_candidate.get("public_id") is not None and int(target_candidate["public_id"]) == int(public_id)),
        "target_public_overwritten_by_non_target": bool(assigned is not None and (target_uid is None or assigned_uid != target_uid)),
        "target_public_lost_or_none": bool(assigned is None),
        "assigned_candidate_feature_sha256": None if assigned is None else assigned.get("feature_sha256"),
        "target_candidate_feature_sha256": None if target_candidate is None else target_candidate.get("feature_sha256"),
    }


def _stream_key(candidate: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        int(candidate.get("official_raw_sam_id", -1)),
        int(candidate.get("adapter_external_id", -1)),
        tuple(round(float(value), 5) for value in candidate.get("box_xyxy", [])),
        str(candidate.get("feature_sha256")),
    )


def _stream_delta(left: Mapping[str, Any], right: Mapping[str, Any]) -> dict[str, Any]:
    left_set = {_stream_key(item) for item in left.get("candidate_rows", [])}
    right_set = {_stream_key(item) for item in right.get("candidate_rows", [])}
    union = left_set | right_set
    return {
        "left_count": len(left_set),
        "right_count": len(right_set),
        "intersection_count": len(left_set & right_set),
        "added_to_right": len(right_set - left_set),
        "removed_from_left": len(left_set - right_set),
        "symmetric_difference_count": len(left_set ^ right_set),
        "jaccard": None if not union else float(len(left_set & right_set) / len(union)),
        "identical": bool(left_set == right_set),
    }


def _cosine(left: Sequence[float] | None, right: Sequence[float] | None) -> float | None:
    if left is None or right is None:
        return None
    a = np.asarray(left, dtype=np.float64).reshape(-1)
    b = np.asarray(right, dtype=np.float64).reshape(-1)
    if a.size == 0 or a.size != b.size or not np.all(np.isfinite(a)) or not np.all(np.isfinite(b)):
        return None
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na <= 1.0e-12 or nb <= 1.0e-12:
        return None
    return float(np.dot(a, b) / (na * nb))


def _rate(values: Iterable[bool]) -> float | None:
    values = list(values)
    return None if not values else float(sum(bool(value) for value in values) / len(values))


def _compact_horizon(frames: Sequence[Mapping[str, Any]], horizon: int) -> dict[str, Any]:
    selected = list(frames[: int(horizon)])
    visible = [item for item in selected if item["visible"]]
    errors = [item for item in visible if not item["target_correct"]]
    overwrites = [item for item in selected if item["target_public_overwritten_by_non_target"]]
    lost = [item for item in selected if item["target_public_lost_or_none"]]
    stream_delta = [item["stream_delta"] for item in selected]
    first_error = next((int(item["horizon"]) for item in errors), None)
    first_overwrite = next((int(item["horizon"]) for item in overwrites), None)
    first_lost = next((int(item["horizon"]) for item in lost), None)
    return {
        "horizon": int(horizon),
        "frame_count": len(selected),
        "visible_frame_count": len(visible),
        "target_correct_rate": _rate(item["target_correct"] for item in visible),
        "candidate_present_rate": _rate(item["candidate_present"] for item in visible),
        "target_public_overwrite_count": len(overwrites),
        "target_public_lost_or_none_count": len(lost),
        "first_identity_error_horizon": first_error,
        "first_overwrite_horizon": first_overwrite,
        "first_lost_or_none_horizon": first_lost,
        "stream_delta_frame_count": sum(not bool(item["stream_delta"]["identical"]) for item in selected),
        "stream_delta_rate": _rate(not bool(item["stream_delta"]["identical"]) for item in selected),
    }


def main() -> int:
    required = [
        EVENT_MANIFEST,
        RUN_ROOT / "stage08_runtime_manifest.json",
        RUN_ROOT / "stage09_validation.json",
        RUN_ROOT / "stage10_effect_scoring.json",
        STAGE07_ROOT / "official_full_loop_manifest.json",
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        payload = {
            "schema_version": "N72R5R1_ROUND06_PERSISTENCE_AUDIT_V1",
            "status": "BLOCKED_MISSING_PERSISTENCE_INPUT",
            "missing_inputs": missing,
            "runtime_future_gt_used": False,
            "created_at_utc": _now(),
        }
        atomic_json(SUMMARY, payload)
        atomic_json(STATUS, {**payload, "stage": "06_SPATIAL_CORRECTION_PERSISTENCE_AUDIT"})
        print(json.dumps(payload, ensure_ascii=False))
        return 1

    policy = read_json(EVENT_MANIFEST)
    events = {str(item["event_id"]): dict(item) for item in policy.get("events", [])}
    manifest = read_json(RUN_ROOT / "stage08_runtime_manifest.json")
    results = {str(item["event_id"]): dict(item) for item in manifest.get("events", [])}
    event_rows: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    unavailable: list[dict[str, Any]] = []
    applied_event_ids: list[str] = []
    not_applied_event_ids: list[str] = []

    for event_id in sorted(events):
        event = events[event_id]
        result = results.get(event_id)
        private_path = RUN_ROOT / "simulation_private" / event_id / "oracle_private_mapping.json"
        try:
            if result is None or not private_path.is_file():
                unavailable.append({"event_id": event_id, "reason": "missing_event_result_or_private_mapping"})
                continue
            private = read_json(private_path)
            mapping = {int(key): int(value) for key, value in (private.get("dataset_gt_to_public") or {}).items()}
            target_gt = int(event["dataset_gt_id"])
            target_public = mapping.get(target_gt)
            if target_public is None:
                unavailable.append({"event_id": event_id, "reason": "target_public_unresolved"})
                continue
            gt_frames = load_gt(str(event["sequence"]))
            b0_rows = _rows(_branch_output(result, B0))
            b1_rows = _rows(_branch_output(result, B1))
            b1_diag = _branch_diagnostic(result, B1)
            action_status = str(b1_diag.get("status", "UNKNOWN"))
            applied = bool(b1_diag.get("human_intervention_applied") is True)
            if applied:
                applied_event_ids.append(event_id)
            else:
                not_applied_event_ids.append(event_id)

            stage07_by_branch, _, _ = load_stage07_event_rows(
                STAGE07_ROOT,
                event_id,
                int(event["event_frame"]),
                str(event["sequence"]),
            )
        except Exception:
            # Keep a complete traceback rather than silently dropping an
            # event from the posthoc mechanism audit.
            import traceback

            errors.append(
                {
                    "event_id": event_id,
                    "failure_type": "PERSISTENCE_AUDIT_EVENT_ERROR",
                    "traceback": traceback.format_exc(),
                }
            )
            continue

        # The sealed Stage07 loader returns normalized rows keyed by frame.
        # Build a UID->feature map over the full B1 event/future range.
        feature_by_uid: dict[str, np.ndarray] = {}
        for frame_candidates in stage07_by_branch[B1].values():
            for candidate in frame_candidates:
                feature_by_uid[str(candidate["candidate_uid"])] = np.asarray(candidate["feature"], dtype=np.float64)

        event_frame = int(event["event_frame"])
        event_box = gt_frames.get(event_frame, {}).get(target_gt, {}).get("box")
        frame_records: list[dict[str, Any]] = []
        for frame in range(event_frame + 1, event_frame + HORIZON + 1):
            b0 = b0_rows[frame]
            b1 = b1_rows[frame]
            box = gt_frames.get(frame, {}).get(target_gt, {}).get("box")
            b0_eval = _frame_eval(b0, box, int(target_public))
            b1_eval = _frame_eval(b1, box, int(target_public))
            delta = _stream_delta(b0, b1)
            b1_uid = b1_eval.get("target_candidate_uid")
            event_uid = b1_diag.get("target_candidate_uid")
            cosine = _cosine(feature_by_uid.get(str(event_uid)), feature_by_uid.get(str(b1_uid))) if event_uid and b1_uid else None
            frame_records.append(
                {
                    "frame": int(frame),
                    "horizon": int(frame - event_frame),
                    "b0": b0_eval,
                    "b1": b1_eval,
                    "stream_delta": delta,
                    "b0_candidate_stream_source": b0.get("candidate_stream_source"),
                    "b1_candidate_stream_source": b1.get("candidate_stream_source"),
                    "b1_event_anchor_to_target_candidate_cosine": cosine,
                    "runtime_future_gt_used": False,
                }
            )

        event_plus_one = frame_records[0]
        b1_future_uids = [item["b1"].get("target_candidate_uid") for item in frame_records if item["b1"].get("target_candidate_uid")]
        b1_stream_changed = [item for item in frame_records if not item["stream_delta"]["identical"]]
        event_rows.append(
            {
                "event_id": event_id,
                "sequence": str(event["sequence"]),
                "action_type": str(event["action_type"]),
                "event_frame": event_frame,
                "target_gt_id": target_gt,
                "target_public_id": int(target_public),
                "b1_action_status": action_status,
                "b1_human_intervention_applied": applied,
                "b1_target_candidate_iou_at_event": b1_diag.get("target_candidate_iou"),
                "b1_event_target_candidate_uid": b1_diag.get("target_candidate_uid"),
                "event_box_available_posthoc": event_box is not None,
                "event_plus_one": {
                    "b0": event_plus_one["b0"],
                    "b1": event_plus_one["b1"],
                    "stream_delta": event_plus_one["stream_delta"],
                    "b1_event_anchor_to_target_candidate_cosine": event_plus_one["b1_event_anchor_to_target_candidate_cosine"],
                },
                "b1_target_candidate_uid_change_count_over_h100": int(sum(1 for left, right in zip(b1_future_uids, b1_future_uids[1:]) if left != right)),
                "b1_candidate_stream_changed_frame_count_h100": len(b1_stream_changed),
                "b1_candidate_stream_changed_rate_h100": float(len(b1_stream_changed) / len(frame_records)),
                "horizons": {
                    "20": {"b0": _compact_horizon([{**item["b0"], "horizon": item["horizon"], "stream_delta": item["stream_delta"]} for item in frame_records], 20), "b1": _compact_horizon([{**item["b1"], "horizon": item["horizon"], "stream_delta": item["stream_delta"]} for item in frame_records], 20)},
                    "50": {"b0": _compact_horizon([{**item["b0"], "horizon": item["horizon"], "stream_delta": item["stream_delta"]} for item in frame_records], 50), "b1": _compact_horizon([{**item["b1"], "horizon": item["horizon"], "stream_delta": item["stream_delta"]} for item in frame_records], 50)},
                    "100": {"b0": _compact_horizon([{**item["b0"], "horizon": item["horizon"], "stream_delta": item["stream_delta"]} for item in frame_records], 100), "b1": _compact_horizon([{**item["b1"], "horizon": item["horizon"], "stream_delta": item["stream_delta"]} for item in frame_records], 100)},
                },
                "frame_records": frame_records,
                "runtime_future_gt_used": False,
            }
        )

    # Aggregate only applied B1 events for causal persistence conclusions;
    # retain non-applied events separately so they cannot be mistaken for a
    # successful intervention.
    applied_rows = [row for row in event_rows if row["b1_human_intervention_applied"]]
    all_frames = [frame for row in applied_rows for frame in row["frame_records"]]
    def _side(name: str) -> list[Mapping[str, Any]]:
        return [frame[name] for frame in all_frames]

    def _count(predicate: Any) -> int:
        return int(sum(bool(predicate(item)) for item in all_frames))

    event_plus_one_b1 = [row["event_plus_one"]["b1"] for row in applied_rows]
    event_plus_one_b0 = [row["event_plus_one"]["b0"] for row in applied_rows]
    stream_event_plus_one = [row["event_plus_one"]["stream_delta"] for row in applied_rows]
    overwrite_count = _count(lambda item: item["b1"]["target_public_overwritten_by_non_target"])
    lost_count = _count(lambda item: item["b1"]["target_public_lost_or_none"])
    target_error_count = _count(lambda item: item["b1"]["visible"] and not item["b1"]["target_correct"])
    target_present_count = _count(lambda item: item["b1"]["visible"] and item["b1"]["candidate_present"])
    target_correct_count = _count(lambda item: item["b1"]["target_correct"])
    frame_count = len(all_frames)
    summary = {
        "schema_version": "N72R5R1_ROUND06_PERSISTENCE_AUDIT_V1",
        "status": "PASS_PERSISTENCE_AUDIT_COMPLETED" if not errors else "BLOCKED_PERSISTENCE_AUDIT_EVENT_ERRORS",
        "inputs": {
            "event_manifest": str(EVENT_MANIFEST),
            "event_manifest_sha256": sha256_file(EVENT_MANIFEST),
            "stage08_runtime_manifest": str(RUN_ROOT / "stage08_runtime_manifest.json"),
            "stage08_runtime_manifest_sha256": sha256_file(RUN_ROOT / "stage08_runtime_manifest.json"),
            "stage09_validation": str(RUN_ROOT / "stage09_validation.json"),
            "stage10_effect_scoring": str(RUN_ROOT / "stage10_effect_scoring.json"),
            "stage07_manifest": str(STAGE07_ROOT / "official_full_loop_manifest.json"),
        },
        "coverage": {
            "expected_event_count": len(events),
            "event_count_with_public_mapping": len(event_rows),
            "applied_b1_event_count": len(applied_rows),
            "not_applied_b1_event_count": len(not_applied_event_ids),
            "not_applied_b1_event_ids": not_applied_event_ids,
            "unavailable_event_count": len(unavailable),
            "unavailable_events": unavailable,
            "audit_error_count": len(errors),
            "runtime_future_gt_used": False,
        },
        "event_plus_one": {
            "applied_event_count": len(applied_rows),
            "b0_target_correct_count": int(sum(item["target_correct"] for item in event_plus_one_b0)),
            "b1_target_correct_count": int(sum(item["target_correct"] for item in event_plus_one_b1)),
            "b0_target_candidate_absent_count": int(sum(not item["candidate_present"] for item in event_plus_one_b0)),
            "b1_target_candidate_absent_count": int(sum(not item["candidate_present"] for item in event_plus_one_b1)),
            "b1_target_public_overwrite_count": int(sum(item["target_public_overwritten_by_non_target"] for item in event_plus_one_b1)),
            "b1_target_public_lost_or_none_count": int(sum(item["target_public_lost_or_none"] for item in event_plus_one_b1)),
            "candidate_stream_changed_count": int(sum(not item["identical"] for item in stream_event_plus_one)),
            "candidate_stream_identical_rate": _rate(item["identical"] for item in stream_event_plus_one),
        },
        "applied_b1_h100": {
            "frame_count": frame_count,
            "target_candidate_present_count": target_present_count,
            "target_correct_count": target_correct_count,
            "target_error_count": target_error_count,
            "target_error_rate_over_visible": None if not target_present_count + target_error_count else float(target_error_count / max(1, target_present_count + target_error_count)),
            "target_public_overwrite_count": overwrite_count,
            "target_public_lost_or_none_count": lost_count,
            "target_public_overwrite_rate": None if not frame_count else float(overwrite_count / frame_count),
            "target_public_lost_or_none_rate": None if not frame_count else float(lost_count / frame_count),
            "candidate_stream_changed_frame_count": int(sum(row["b1_candidate_stream_changed_frame_count_h100"] for row in applied_rows)),
            "candidate_stream_changed_frame_rate": None if not frame_count else float(sum(row["b1_candidate_stream_changed_frame_count_h100"] for row in applied_rows) / frame_count),
        },
        "horizon_event_aggregates": {
            str(horizon): {
                "b0_target_correct_rate": _rate(row["horizons"][str(horizon)]["b0"]["target_correct_rate"] is not None and row["horizons"][str(horizon)]["b0"]["target_correct_rate"] > 0.0 for row in applied_rows),
                "b1_target_correct_rate_mean": None if not applied_rows else float(np.mean([row["horizons"][str(horizon)]["b1"]["target_correct_rate"] for row in applied_rows if row["horizons"][str(horizon)]["b1"]["target_correct_rate"] is not None])),
                "b1_first_error_count": int(sum(row["horizons"][str(horizon)]["b1"]["first_identity_error_horizon"] is not None for row in applied_rows)),
                "b1_first_overwrite_count": int(sum(row["horizons"][str(horizon)]["b1"]["first_overwrite_horizon"] is not None for row in applied_rows)),
                "b1_first_lost_or_none_count": int(sum(row["horizons"][str(horizon)]["b1"]["first_lost_or_none_horizon"] is not None for row in applied_rows)),
                "b1_stream_delta_frame_count": int(sum(row["horizons"][str(horizon)]["b1"]["stream_delta_frame_count"] for row in applied_rows)),
            }
            for horizon in (20, 50, 100)
        },
        "mechanism_conclusion": {
            "spatial_correction_persistence_or_candidate_stream_drift_supported": bool(
                len(applied_rows) > 0
                and (overwrite_count > 0 or lost_count > 0 or any(not item["stream_delta"]["identical"] for item in all_frames))
            ),
            "candidate_stream_drift_is_observed": bool(any(not item["stream_delta"]["identical"] for item in all_frames)),
            "target_public_overwrite_is_observed": bool(overwrite_count > 0),
            "target_public_loss_is_observed": bool(lost_count > 0),
            "next_action": "RUN_ONE_FIXED_IDENTITY_SCOPED_PERSISTENCE_PROBE" if applied_rows else "BLOCKED_NO_APPLIED_SPATIAL_EVENTS",
            "production_promotion": False,
            "calibration_or_lora_authorized": False,
        },
        "errors": errors,
        "runtime_future_gt_used": False,
        "posthoc_gt_opened": True,
        "created_at_utc": _now(),
    }
    atomic_jsonl(TABLE, event_rows)
    atomic_json(SUMMARY, summary)
    status = {
        "schema_version": "N72R5R1_ROUND06_STATUS_V1",
        "stage": "06_SPATIAL_CORRECTION_PERSISTENCE_AUDIT",
        "status": summary["status"],
        "event_count": len(event_rows),
        "applied_event_count": len(applied_rows),
        "audit_error_count": len(errors),
        "summary": str(SUMMARY),
        "table": str(TABLE),
        "next_action": summary["mechanism_conclusion"]["next_action"],
        "runtime_future_gt_used": False,
        "created_at_utc": _now(),
    }
    atomic_json(STATUS, status)
    print(json.dumps(status, ensure_ascii=False))
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
