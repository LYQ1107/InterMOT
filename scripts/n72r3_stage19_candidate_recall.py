#!/usr/bin/env python3
"""Posthoc Candidate V2 recall diagnosis for the frozen N72R3 events.

This process runs after the exact-public baseline.  It reads GT only here,
after the candidate/runtime artifacts are frozen, and never sends GT to a
runtime, association solver, or event selector.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import sys
import tempfile
import traceback
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.n72r3_stage09_11_candidate_runtime import load_plan, load_source_rows  # noqa: E402


OUT = ROOT / "outputs/N72R3"
EVENT_MANIFEST = OUT / "simulation/real_event_manifest.json"
BASELINE_STATUS = OUT / "stage_18_status.json"
RECALL_ROOT = OUT / "candidate_recall"
FRAME_PATH = RECALL_ROOT / "frame_diagnosis.jsonl"
RESULT_PATH = RECALL_ROOT / "stage19_candidate_recall.json"
STATUS_PATH = OUT / "stage_19_status.json"
FAILURE_PATH = OUT / "attempts/stage19_candidate_recall_failure.json"
HORIZONS = (20, 50, 100)
IOU_THRESHOLD = 0.5


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


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def iou(left: Any, right: Any) -> float:
    a = np.asarray(left, dtype=float).reshape(-1)
    b = np.asarray(right, dtype=float).reshape(-1)
    if a.size != 4 or b.size != 4 or not np.all(np.isfinite(a)) or not np.all(np.isfinite(b)):
        return 0.0
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter
    return float(inter / union) if union > 0.0 else 0.0


def center(box: list[float]) -> tuple[float, float]:
    return ((float(box[0]) + float(box[2])) / 2.0, (float(box[1]) + float(box[3])) / 2.0)


def load_gt(sequence: str) -> dict[int, dict[int, dict[str, Any]]]:
    path = Path("/data1/LWR/vranlee/DATASETS/JDE/dancetrack") / "train" / sequence / "gt/gt.txt"
    if not path.is_file():
        raise FileNotFoundError(path)
    result: dict[int, dict[int, dict[str, Any]]] = defaultdict(dict)
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            parts = [item.strip() for item in line.split(",")]
            if len(parts) < 6:
                raise ValueError(f"{path}:{line_number}: malformed GT row")
            frame_one_based, dataset_id = int(parts[0]), int(parts[1])
            x, y, width, height = [float(item) for item in parts[2:6]]
            box = [x, y, x + width, y + height]
            mark = float(parts[6]) if len(parts) > 6 else None
            class_id = float(parts[7]) if len(parts) > 7 else None
            visibility = float(parts[8]) if len(parts) > 8 else None
            result[frame_one_based - 1][dataset_id] = {
                "box": box,
                "mark": mark,
                "class_id": class_id,
                "visibility": visibility,
            }
    return result


def size_bin(box: list[float]) -> str:
    area = max(0.0, float(box[2] - box[0])) * max(0.0, float(box[3] - box[1]))
    if area < 32.0 * 32.0:
        return "small_area_lt_1024"
    if area < 96.0 * 96.0:
        return "medium_area_1024_to_9216"
    return "large_area_ge_9216"


def occlusion_bin(gt: dict[str, Any] | None) -> str:
    if gt is None:
        return "gt_absent"
    visibility = gt.get("visibility")
    if visibility is None or not math.isfinite(float(visibility)):
        return "visibility_unavailable"
    return "visibility_lt_0_5" if float(visibility) < 0.5 else "visibility_ge_0_5"


def motion_bin(gt_by_frame: dict[int, dict[int, dict[str, Any]]], frame: int, dataset_id: int) -> tuple[str, float | None]:
    current = gt_by_frame.get(frame, {}).get(dataset_id)
    previous = gt_by_frame.get(frame - 5, {}).get(dataset_id)
    if current is None or previous is None:
        return "motion_unavailable", None
    c0 = center(current["box"])
    c1 = center(previous["box"])
    speed = math.hypot(c0[0] - c1[0], c0[1] - c1[1]) / 5.0
    if speed < 5.0:
        return "low_lt_5px_per_frame", speed
    if speed < 20.0:
        return "medium_5_to_20px_per_frame", speed
    return "high_ge_20px_per_frame", speed


def gt_gap_before(gt_by_frame: dict[int, dict[int, dict[str, Any]]], frame: int, dataset_id: int) -> int:
    gap = 0
    cursor = frame - 1
    while cursor >= 0 and dataset_id not in gt_by_frame.get(cursor, {}):
        gap += 1
        cursor -= 1
    return gap


def candidate_lost_age(
    rows_by_frame: dict[int, list[dict[str, Any]]],
    gt_by_frame: dict[int, dict[int, dict[str, Any]]],
    frame: int,
    dataset_id: int,
) -> int | None:
    current = gt_by_frame.get(frame, {}).get(dataset_id)
    if current is None:
        return None
    age = 0
    cursor = frame
    while cursor >= 0:
        gt = gt_by_frame.get(cursor, {}).get(dataset_id)
        if gt is None:
            break
        best = max((iou(row["box_xyxy"], gt["box"]) for row in rows_by_frame.get(cursor, [])), default=0.0)
        if best >= IOU_THRESHOLD:
            return age
        age += 1
        cursor -= 1
    return age


def event_summary_template() -> dict[str, Any]:
    return {
        "target_gt_present_frames": 0,
        "candidate_present_frames": 0,
        "candidate_absent_frames": 0,
        "gt_absent_frames": 0,
    }


def update_summary(summary: dict[str, Any], record: dict[str, Any]) -> None:
    if record["target_gt_present"]:
        summary["target_gt_present_frames"] += 1
        if record["candidate_present"]:
            summary["candidate_present_frames"] += 1
        else:
            summary["candidate_absent_frames"] += 1
    else:
        summary["gt_absent_frames"] += 1


def finalize_summary(summary: dict[str, Any]) -> dict[str, Any]:
    denominator = int(summary["target_gt_present_frames"])
    result = dict(summary)
    result["candidate_recall"] = (float(summary["candidate_present_frames"]) / denominator) if denominator else None
    return result


def add_dimension(
    target: dict[str, dict[str, Any]],
    dimension: str,
    value: str,
    record: dict[str, Any],
) -> None:
    bucket = target.setdefault(f"{dimension}={value}", event_summary_template())
    update_summary(bucket, record)


def main() -> int:
    try:
        baseline = json.loads(BASELINE_STATUS.read_text(encoding="utf-8"))
        if not str(baseline.get("status", "")).startswith("PASS"):
            raise RuntimeError("Stage 19 requires a passing Stage 18 exact-public baseline")
        events_payload = json.loads(EVENT_MANIFEST.read_text(encoding="utf-8"))
        events = [dict(item) for item in events_payload.get("events", [])]
        plan = load_plan()
        plan_by_window = {str(item["window_id"]): dict(item) for item in plan}
        frame_records: list[dict[str, Any]] = []
        event_records: list[dict[str, Any]] = []
        aggregate = {str(horizon): event_summary_template() for horizon in HORIZONS}
        by_action = {str(action): {str(horizon): event_summary_template() for horizon in HORIZONS} for action in sorted({str(item["action_type"]) for item in events})}
        by_dimension = {str(horizon): {} for horizon in HORIZONS}
        input_records: list[dict[str, Any]] = []

        for event in events:
            event_id = str(event["event_id"])
            sequence = str(event["sequence"])
            event_frame = int(event["event_frame"])
            dataset_id = int(event["dataset_gt_id"])
            window_id = str(event["current_candidate_v2"]["window_id"])
            window = plan_by_window[window_id]
            rows, metadata = load_source_rows(window)
            gt_by_frame = load_gt(sequence)
            rows_by_frame: dict[int, list[dict[str, Any]]] = defaultdict(list)
            for row in rows:
                rows_by_frame[int(row["frame_idx"])].append(row)
            for frame_rows in rows_by_frame.values():
                frame_rows.sort(key=lambda item: (int(item["candidate_index"]), str(item["candidate_uid"])))
            gt_event = gt_by_frame.get(event_frame, {}).get(dataset_id)
            if gt_event is None or iou(event["current_gt_box"], gt_event["box"]) < 0.999:
                raise ValueError(f"event current box does not match posthoc GT: {event_id}")
            input_records.append(
                {
                    "event_id": event_id,
                    "window_id": window_id,
                    "sequence": sequence,
                    "candidate_path": metadata["candidate_path"],
                    "candidate_sha256": metadata["candidate_sha256"],
                    "candidate_frame_sha256": metadata["candidate_frame_sha256"],
                    "frame_start": metadata["frame_start"],
                    "frame_end": metadata["frame_end"],
                    "row_count": metadata["row_count"],
                    "runtime_future_gt_used": False,
                }
            )
            event_dimensions = {
                "action_type": str(event["action_type"]),
                "sequence": sequence,
                "session_boundary_relation": "event_at_window_start" if event_frame == int(window["frame_start"]) else "within_independent_session_window",
                "target_size": size_bin(gt_event["box"]),
                "gap_length": f"{min(gt_gap_before(gt_by_frame, event_frame, dataset_id), 20)}plus" if gt_gap_before(gt_by_frame, event_frame, dataset_id) > 20 else str(gt_gap_before(gt_by_frame, event_frame, dataset_id)),
                "occlusion": occlusion_bin(gt_event),
                "motion": motion_bin(gt_by_frame, event_frame, dataset_id)[0],
            }
            event_horizon_stats = {}
            for horizon in HORIZONS:
                horizon_start = event_frame + 1
                horizon_end = event_frame + horizon
                expected = list(range(horizon_start, horizon_end + 1))
                available = [frame for frame in expected if frame in rows_by_frame]
                if len(available) != len(expected):
                    raise ValueError(f"candidate stream does not cover H{horizon}: {event_id} missing {len(expected) - len(available)} frames")
                horizon_stats = event_summary_template()
                for frame in available:
                    gt = gt_by_frame.get(frame, {}).get(dataset_id)
                    candidates = rows_by_frame.get(frame, [])
                    ious = [(iou(row["box_xyxy"], gt["box"]) if gt is not None else 0.0, row) for row in candidates]
                    best_iou, best_row = max(ious, key=lambda item: (item[0], -int(item[1]["candidate_index"])), default=(0.0, None))
                    target_present = gt is not None
                    motion_label, motion_speed = motion_bin(gt_by_frame, frame, dataset_id)
                    record = {
                        "event_id": event_id,
                        "sequence": sequence,
                        "action_type": str(event["action_type"]),
                        "event_frame": event_frame,
                        "frame_idx": int(frame),
                        "horizon": int(horizon),
                        "dataset_gt_id": dataset_id,
                        "target_gt_present": target_present,
                        "candidate_present": bool(target_present and best_iou >= IOU_THRESHOLD),
                        "candidate_absent": bool(target_present and best_iou < IOU_THRESHOLD),
                        "best_candidate_iou": float(best_iou),
                        "best_candidate_uid": None if best_row is None else str(best_row["candidate_uid"]),
                        "best_candidate_index": None if best_row is None else int(best_row["candidate_index"]),
                        "candidate_frame_row_count": len(candidates),
                        "occlusion": occlusion_bin(gt),
                        "target_size": size_bin(gt["box"]) if gt is not None else "gt_absent",
                        "gap_length": str(gt_gap_before(gt_by_frame, frame, dataset_id)) if gt is not None else None,
                        "session_boundary_relation": event_dimensions["session_boundary_relation"],
                        "lost_age": candidate_lost_age(rows_by_frame, gt_by_frame, frame, dataset_id),
                        "motion": motion_label,
                        "motion_speed_px_per_frame": None if motion_speed is None else float(motion_speed),
                        "window_id": window_id,
                        "candidate_recall_is_performance_only": True,
                        "runtime_future_gt_used": False,
                        "gt_usage": "posthoc_only",
                    }
                    frame_records.append(record)
                    update_summary(horizon_stats, record)
                    for dimension, value in event_dimensions.items():
                        add_dimension(by_dimension[str(horizon)], dimension, value, record)
                    add_dimension(by_dimension[str(horizon)], "frame_occlusion", record["occlusion"], record)
                    add_dimension(by_dimension[str(horizon)], "frame_motion", record["motion"], record)
                event_horizon_stats[str(horizon)] = finalize_summary(horizon_stats)
                update_summary(aggregate[str(horizon)], {"target_gt_present": horizon_stats["target_gt_present_frames"] > 0, "candidate_present": horizon_stats["candidate_present_frames"] > 0})
                # Aggregate over frame records below; this event-level marker
                # is intentionally not used for the final denominator.
                action_summary = by_action[str(event["action_type"])][str(horizon)]
                for record in frame_records[-len(available):]:
                    update_summary(action_summary, record)
            event_record = {
                "event_id": event_id,
                "sequence": sequence,
                "event_frame": event_frame,
                "action_type": str(event["action_type"]),
                "dataset_gt_id": dataset_id,
                "event_candidate_best_iou": max((iou(row["box_xyxy"], gt_event["box"]) for row in rows_by_frame.get(event_frame, [])), default=0.0),
                "event_candidate_present": max((iou(row["box_xyxy"], gt_event["box"]) for row in rows_by_frame.get(event_frame, [])), default=0.0) >= IOU_THRESHOLD,
                "event_dimensions": event_dimensions,
                "horizons": event_horizon_stats,
                "runtime_future_gt_used": False,
                "gt_usage": "posthoc_only",
            }
            event_records.append(event_record)

        atomic_jsonl(FRAME_PATH, frame_records)
        for horizon in HORIZONS:
            bucket = aggregate[str(horizon)]
            bucket["target_gt_present_frames"] = sum(1 for row in frame_records if int(row["horizon"]) == horizon and row["target_gt_present"])
            bucket["candidate_present_frames"] = sum(1 for row in frame_records if int(row["horizon"]) == horizon and row["candidate_present"])
            bucket["candidate_absent_frames"] = sum(1 for row in frame_records if int(row["horizon"]) == horizon and row["candidate_absent"])
            bucket["gt_absent_frames"] = sum(1 for row in frame_records if int(row["horizon"]) == horizon and not row["target_gt_present"])
            aggregate[str(horizon)] = finalize_summary(bucket)
            for dimension, buckets in by_dimension[str(horizon)].items():
                by_dimension[str(horizon)][dimension] = finalize_summary(buckets)
            for action in by_action:
                action_bucket = by_action[action][str(horizon)]
                by_action[action][str(horizon)] = finalize_summary(action_bucket)
        result = {
            "schema_version": "N72R3_STAGE19_CANDIDATE_RECALL_V1",
            "status": "PASS_STAGE19_CANDIDATE_RECALL_DIAGNOSTIC",
            "created_at_utc": now_utc(),
            "event_count": len(events),
            "independent_sequence_count": len({str(item["sequence"]) for item in events}),
            "iou_threshold": IOU_THRESHOLD,
            "horizons": list(HORIZONS),
            "event_recall": event_records,
            "by_horizon": aggregate,
            "by_action": by_action,
            "by_dimension": by_dimension,
            "classification_definitions": {
                "candidate_present": "max candidate box IoU to posthoc target GT >= 0.5",
                "candidate_absent": "target GT exists but every candidate box IoU < 0.5",
                "target_gt_absent": "no posthoc GT row for selected dataset_gt_id; excluded from recall denominator",
                "session_boundary_relation": "event at frozen window start versus within an independent session window",
                "lost_age": "consecutive prior GT-present frames without a candidate at IoU >= 0.5",
                "gap_length": "consecutive prior frames without a GT row for the selected dataset ID",
                "occlusion": "fixed raw DanceTrack visibility split at 0.5; unavailable is retained",
                "target_size": "fixed pixel-area bins small<1024, medium<9216, large>=9216",
                "motion": "fixed 5-frame center speed bins <5, 5-20, >=20 pixels/frame",
            },
            "input_records": input_records,
            "frame_artifact": str(FRAME_PATH),
            "candidate_recall_is_performance_only": True,
            "runtime_future_gt_used": False,
            "gt_usage": "posthoc_only_after_stage18_runtime_artifacts_frozen",
            "scientific_result": "PERFORMANCE_DIAGNOSTIC_NOT_FUTURE_EFFECT",
        }
        atomic_json(RESULT_PATH, result)
        status = {
            "schema_version": "N72R3_STAGE_STATUS_V1",
            "stage": "19_CANDIDATE_RECALL_DIAGNOSIS",
            "status": "PASS_STAGE19_CANDIDATE_RECALL_DIAGNOSTIC",
            "event_count": len(events),
            "independent_sequence_count": len({str(item["sequence"]) for item in events}),
            "horizons": list(HORIZONS),
            "by_horizon": aggregate,
            "candidate_recall_is_performance_only": True,
            "runtime_future_gt_used": False,
            "gt_usage": "posthoc_only",
            "result_artifact": str(RESULT_PATH),
            "frame_artifact": str(FRAME_PATH),
            "scientific_result": "PERFORMANCE_DIAGNOSTIC_NOT_FUTURE_EFFECT",
        }
        atomic_json(STATUS_PATH, status)
        print(json.dumps({"status": status["status"], "result": str(RESULT_PATH)}, sort_keys=True))
        return 0
    except Exception as exc:
        failure = {
            "schema_version": "N72R3_FAILURE_RECORD_V1",
            "stage": "19_CANDIDATE_RECALL_DIAGNOSIS",
            "status": "FAIL_STAGE19_CANDIDATE_RECALL",
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
            "runtime_future_gt_used": False,
            "gt_usage": "posthoc_only",
            "scientific_result": "NO_SCIENTIFIC_RESULT",
        }
        atomic_json(FAILURE_PATH, failure)
        atomic_json(
            STATUS_PATH,
            {
                "schema_version": "N72R3_STAGE_STATUS_V1",
                "stage": "19_CANDIDATE_RECALL_DIAGNOSIS",
                "status": "BLOCKED_STAGE19_CANDIDATE_RECALL",
                "failure_artifact": str(FAILURE_PATH),
                "runtime_future_gt_used": False,
                "scientific_result": "NO_SCIENTIFIC_RESULT",
            },
        )
        print(json.dumps({"status": failure["status"], "failure": str(FAILURE_PATH)}, sort_keys=True))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
