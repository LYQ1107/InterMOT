#!/usr/bin/env python3
"""Audit frozen N72R9 candidate sources before the positive-area replay.

This is a source-only audit.  It does not open dataset GT, run a model, or
change any frozen N72R9/N72R11R4 artifact.  Zero-area rows are reported as
input evidence; filtering is left to the opt-in candidate-pool policy used by
the corrected replay.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime, timezone
import argparse
import json
import math
from pathlib import Path
import sys
from typing import Any, Mapping

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import n72r9_temporal_replay as legacy  # noqa: E402


OUTPUT_PATH = ROOT / "outputs/N72R11R5R1/geometry_source_audit.json"
SOURCE_KEYS = (
    "c0_source",
    "c1_source",
    "target_stream_source",
    "requery_source",
)
SOURCE_LABELS = {
    "c0_source": "MAIN_B0_CANDIDATE",
    "c1_source": "C1_AUTHORITY_SOURCE",
    "target_stream_source": "TARGET_SESSION_CURRENT_RAW",
    "requery_source": "TARGET_SESSION_REQUERY",
}


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(nested) for key, nested in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        result = float(value)
        return result if math.isfinite(result) else None
    return value


def _box_geometry(candidate: Mapping[str, Any]) -> tuple[bool, list[float] | None, str | None]:
    value = candidate.get("box_xyxy", candidate.get("box"))
    try:
        array = np.asarray(value, dtype=np.float64).reshape(-1)
    except (TypeError, ValueError):
        return False, None, "box_not_numeric"
    if array.size != 4:
        return False, None, "box_not_four_values"
    if not np.all(np.isfinite(array)):
        return False, [float(item) if math.isfinite(float(item)) else None for item in array], "box_nonfinite"
    box = [float(item) for item in array]
    if box[2] <= box[0] or box[3] <= box[1]:
        return False, box, "box_non_positive_area"
    return True, box, None


def _candidate_record(
    *,
    event: Mapping[str, Any],
    source_key: str,
    source_label: str,
    frame: int,
    index: int,
    candidate: Mapping[str, Any],
    geometry_valid: bool,
    box: list[float] | None,
    reason: str | None,
) -> dict[str, Any]:
    return {
        "event_id": str(event["event_id"]),
        "sequence": str(event["sequence"]),
        "source_key": source_key,
        "source_label": source_label,
        "frame": int(frame),
        "candidate_index": int(candidate.get("candidate_index", index)),
        "candidate_uid": None if candidate.get("candidate_uid") is None else str(candidate["candidate_uid"]),
        "official_raw_sam_id": candidate.get("official_raw_sam_id"),
        "native_scope": candidate.get("native_scope", candidate.get("native_tid_scope")),
        "box_xyxy": box,
        "geometry_valid": bool(geometry_valid),
        "rejection_reason": reason,
    }


def _source_rows(inputs: Mapping[str, Any], source_key: str, frame: int) -> list[Mapping[str, Any]]:
    row = inputs["rows"][source_key][int(frame)]
    return [item for item in row.get("candidate_rows", []) if isinstance(item, Mapping)]


def audit() -> dict[str, Any]:
    protocol = legacy.read_json(legacy.PROTOCOL_PATH)
    events = list(protocol.get("source_event_selection", {}).get("events", []))
    event_records: list[dict[str, Any]] = []
    invalid_records: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    totals = Counter()
    invalid_by_source = Counter()
    invalid_by_sequence = Counter()
    invalid_by_frame = Counter()
    empty_frames: list[dict[str, Any]] = []

    for event in events:
        event_id = str(event.get("event_id"))
        try:
            inputs = legacy._load_rows(event)
        except Exception as exc:  # preserve the first actionable source failure in the artifact
            errors.append(
                {
                    "event_id": event_id,
                    "sequence": str(event.get("sequence")),
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            )
            continue
        event_total = Counter()
        event_invalid = Counter()
        event_empty: list[dict[str, Any]] = []
        for frame in range(int(inputs["event_frame"]), int(inputs["event_frame"]) + legacy.HORIZON + 1):
            valid_by_key: dict[str, int] = {}
            for source_key in SOURCE_KEYS:
                candidates = _source_rows(inputs, source_key, frame)
                source_label = SOURCE_LABELS[source_key]
                valid_count = 0
                for index, candidate in enumerate(candidates):
                    valid, box, reason = _box_geometry(candidate)
                    valid_count += int(valid)
                    totals["total_source_candidates"] += 1
                    event_total["total_source_candidates"] += 1
                    totals[f"total_{source_key}"] += 1
                    event_total[f"total_{source_key}"] += 1
                    if not valid:
                        record = _candidate_record(
                            event=event,
                            source_key=source_key,
                            source_label=source_label,
                            frame=frame,
                            index=index,
                            candidate=candidate,
                            geometry_valid=valid,
                            box=box,
                            reason=reason,
                        )
                        invalid_records.append(record)
                        invalid_by_source[source_key] += 1
                        invalid_by_sequence[str(event["sequence"])] += 1
                        invalid_by_frame[f"{event_id}:{frame}"] += 1
                        event_invalid[source_key] += 1
                        event_invalid["invalid_geometry_candidate_count"] += 1
                        totals["invalid_geometry_candidate_count"] += 1
                valid_by_key[source_key] = valid_count
            # Solver pools that can be constructed from these source axes.  C1
            # is an authority/score source and is intentionally not a pool row.
            pool_sizes = {
                "baseline_c0": valid_by_key["c0_source"],
                "target_c0_plus_target": valid_by_key["c0_source"] + valid_by_key["target_stream_source"],
                "requery_c0_plus_target_plus_requery": (
                    valid_by_key["c0_source"]
                    + valid_by_key["target_stream_source"]
                    + valid_by_key["requery_source"]
                ),
            }
            for pool_name, pool_size in pool_sizes.items():
                if frame > int(inputs["event_frame"]) and pool_size == 0:
                    item = {
                        "event_id": event_id,
                        "sequence": str(event["sequence"]),
                        "frame": int(frame),
                        "pool": pool_name,
                        "valid_candidate_count_after_filter": 0,
                    }
                    event_empty.append(item)
                    empty_frames.append(item)
        event_records.append(
            {
                "event_id": event_id,
                "sequence": str(event["sequence"]),
                "event_frame": int(inputs["event_frame"]),
                "future_window": [int(inputs["event_frame"]) + 1, int(inputs["event_frame"]) + legacy.HORIZON],
                "total_source_candidates": int(event_total["total_source_candidates"]),
                "invalid_geometry_candidate_count": int(event_invalid["invalid_geometry_candidate_count"]),
                "invalid_by_source": {key: int(value) for key, value in sorted(event_invalid.items()) if key != "invalid_geometry_candidate_count"},
                "empty_solver_pool_frame_count": len(event_empty),
                "empty_solver_pool_frames": event_empty,
                "runtime_future_gt_used": False,
            }
        )

    # De-duplicate defensively by pool/frame while retaining each distinct
    # empty solver-pool condition.
    unique_empty = {
        (item["event_id"], item["frame"], item["pool"]): item for item in empty_frames
    }
    status = "PASS_GEOMETRY_SOURCE_AUDIT" if len(events) == 32 and not errors else "FAIL_GEOMETRY_SOURCE_AUDIT"
    return {
        "schema_version": "N72R11R5R1_GEOMETRY_SOURCE_AUDIT_V1",
        "status": status,
        "created_at_utc": now_utc(),
        "protocol_path": str(legacy.PROTOCOL_PATH),
        "protocol_sha256": legacy.sha256_file(legacy.PROTOCOL_PATH),
        "event_count": len(events),
        "events_loaded": len(event_records),
        "event_load_error_count": len(errors),
        "event_load_errors": errors,
        "total_source_candidates": int(totals["total_source_candidates"]),
        "invalid_geometry_candidate_count": int(totals["invalid_geometry_candidate_count"]),
        "invalid_geometry_by_source_key": {key: int(value) for key, value in sorted(invalid_by_source.items())},
        "invalid_geometry_by_sequence": {key: int(value) for key, value in sorted(invalid_by_sequence.items())},
        "invalid_geometry_by_event_frame": {key: int(value) for key, value in sorted(invalid_by_frame.items())},
        "invalid_geometry_candidates": invalid_records,
        "frames_where_solver_pool_becomes_empty": list(unique_empty.values()),
        "empty_solver_pool_frame_count": len(unique_empty),
        "empty_solver_pool_count_by_pool": dict(
            sorted(Counter(item["pool"] for item in unique_empty.values()).items())
        ),
        "runtime_future_gt_used": False,
        "posthoc_gt_used": False,
        "event_records": event_records,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    args = parser.parse_args()
    output = args.output if args.output.is_absolute() else ROOT / args.output
    result = audit()
    legacy.atomic_json(output, _jsonable(result))
    print(json.dumps({"status": result["status"], "output": str(output), "events": result["events_loaded"], "invalid": result["invalid_geometry_candidate_count"], "empty_solver_pool_frames": result["empty_solver_pool_frame_count"]}, sort_keys=True))
    return 0 if result["status"] == "PASS_GEOMETRY_SOURCE_AUDIT" else 1


if __name__ == "__main__":
    raise SystemExit(main())
