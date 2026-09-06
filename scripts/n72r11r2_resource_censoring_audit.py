#!/usr/bin/env python3
"""Build the explicit N72R11R2 resource-censored development audit.

The classification deliberately uses only the sealed execution manifest and
the frozen schedule.  It does not read future identity metrics, IoU, H20, or
any other tracking outcome when deciding whether an item is censored.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any


DEFAULT_MERGED = Path("outputs/N72R11R1/secondary_batch_merged_attempt_05.json")
DEFAULT_SCHEDULE = Path("outputs/N72R11/secondary_event_manifest.json")
DATA_ROOT = Path("/data1/LWR/vranlee/DATASETS/JDE/dancetrack")
FROZEN_HORIZON = 100


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if temporary.exists():
            temporary.unlink()


def _image_count(sequence: str) -> int:
    image_dir = DATA_ROOT / "train" / sequence / "img1"
    return len(list(image_dir.glob("*.jpg")))


def _window_fields(event: dict[str, Any]) -> dict[str, Any]:
    sequence = str(event["sequence"])
    secondary = int(event["secondary_frame"])
    original = int(event["original_event_frame"])
    image_count = _image_count(sequence)
    end_frame = min(secondary + FROZEN_HORIZON, original + FROZEN_HORIZON)
    if image_count:
        end_frame = min(end_frame, image_count - 1)
    return {
        "secondary_frame": secondary,
        "original_event_frame": original,
        "secondary_offset": int(event["secondary_offset"]),
        "future_window_end_frame_inclusive": int(end_frame),
        "future_window_length": max(0, int(end_frame - secondary)),
        "total_window_frame_count": max(0, int(end_frame - secondary + 1)),
        "sequence_image_count": image_count,
    }


def _group_key(record: dict[str, Any], schedule: dict[str, Any]) -> tuple[str, str, int, int]:
    event = schedule[record["event_id"]]
    fields = _window_fields(event)
    return (
        str(event["sequence"]),
        str(event["action_type"]),
        int(fields["secondary_offset"]),
        int(fields["future_window_length"]),
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--merged-manifest", type=Path, default=DEFAULT_MERGED)
    parser.add_argument("--schedule", type=Path, default=DEFAULT_SCHEDULE)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/N72R11R2/resource_censoring_audit.json"),
    )
    args = parser.parse_args()

    merged = json.loads(args.merged_manifest.read_text(encoding="utf-8"))
    schedule_payload = json.loads(args.schedule.read_text(encoding="utf-8"))
    schedule = {str(event["event_id"]): event for event in schedule_payload["events"]}
    records = list(merged.get("records", []))
    unique_ids = [str(record.get("event_id")) for record in records]
    duplicate_ids = sorted(
        event_id for event_id, count in Counter(unique_ids).items() if count > 1
    )
    missing_schedule = sorted(
        event_id for event_id in unique_ids if event_id not in schedule
    )

    retained: list[dict[str, Any]] = []
    censored: list[dict[str, Any]] = []
    invalid: list[dict[str, Any]] = []
    for record in records:
        event_id = str(record.get("event_id"))
        event = schedule.get(event_id)
        if event is None:
            invalid.append({"event_id": event_id, "reason": "missing_schedule"})
            continue
        window = _window_fields(event)
        common = {
            "event_id": event_id,
            "sequence": str(event["sequence"]),
            "action_type": str(event["action_type"]),
            "secondary_offset": int(event["secondary_offset"]),
            "future_window_length": int(window["future_window_length"]),
            "secondary_frame": int(window["secondary_frame"]),
            "future_window_end_frame_inclusive": int(
                window["future_window_end_frame_inclusive"]
            ),
            "runtime_future_gt_used": bool(record.get("runtime_future_gt_used", False)),
            "status": str(record.get("status")),
            "failure_type": record.get("failure_type"),
        }
        if (
            record.get("status") == "PASS"
            and record.get("runtime_future_gt_used", False) is False
        ):
            retained.append({**common, "classification": "RETAINED_EXECUTABLE"})
        elif (
            record.get("status") == "FAIL_CHILD"
            and record.get("failure_type") == "OutOfMemoryError"
            and record.get("runtime_future_gt_used", False) is False
        ):
            censored.append({**common, "classification": "RESOURCE_CENSORED_CUDA_OOM"})
        else:
            invalid.append(
                {
                    **common,
                    "reason": "not_an_allowed_resource_censor_candidate",
                }
            )

    def grouped(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        groups: defaultdict[tuple[str, str, int, int], list[str]] = defaultdict(list)
        for row in rows:
            groups[
                (
                    str(row["sequence"]),
                    str(row["action_type"]),
                    int(row["secondary_offset"]),
                    int(row["future_window_length"]),
                )
            ].append(str(row["event_id"]))
        return [
            {
                "sequence": key[0],
                "action_type": key[1],
                "secondary_offset": key[2],
                "future_window_length": key[3],
                "count": len(event_ids),
                "event_ids": sorted(event_ids),
            }
            for key, event_ids in sorted(groups.items())
        ]

    action_counts = {
        "retained": dict(sorted(Counter(row["action_type"] for row in retained).items())),
        "resource_censored": dict(
            sorted(Counter(row["action_type"] for row in censored).items())
        ),
    }
    result = {
        "schema_version": "N72R11R2_RESOURCE_CENSORING_AUDIT_V1",
        "status": (
            "PASS_RESOURCE_CENSORED_DEVELOPMENT"
            if len(records) == 527
            and len(retained) == 516
            and len(censored) == 11
            and not duplicate_ids
            and not missing_schedule
            and not invalid
            else "FAIL_RESOURCE_CENSORING_AUDIT"
        ),
        "protocol": {
            "required_count": 527,
            "frozen_horizon": FROZEN_HORIZON,
            "schedule_end_rule": "min(secondary_frame+100, original_event_frame+100), clipped to available train images",
            "classification_uses_future_tracking_outcomes": False,
            "allowed_resource_failure": "FAIL_CHILD + OutOfMemoryError + runtime_future_gt_used=false",
            "pass_records_are_never_dropped": True,
            "development_only": True,
        },
        "source": {
            "merged_manifest": str(args.merged_manifest.resolve()),
            "merged_manifest_sha256": _sha256(args.merged_manifest),
            "schedule": str(args.schedule.resolve()),
            "schedule_sha256": _sha256(args.schedule),
        },
        "counts": {
            "required": 527,
            "observed_records": len(records),
            "unique_records": len(set(unique_ids)),
            "duplicate_count": len(duplicate_ids),
            "missing_schedule_count": len(missing_schedule),
            "retained_executable": len(retained),
            "resource_censored_cuda_oom": len(censored),
            "invalid_or_unclassified": len(invalid),
        },
        "action_counts": action_counts,
        "retained_by_sequence_action_offset_window": grouped(retained),
        "censored_by_sequence_action_offset_window": grouped(censored),
        "retained": retained,
        "resource_censored": censored,
        "duplicate_event_ids": duplicate_ids,
        "missing_schedule_event_ids": missing_schedule,
        "invalid_records": invalid,
        "posthoc_metrics_read_for_censor_decision": False,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    _atomic_json(args.output, result)
    print(json.dumps({"status": result["status"], "counts": result["counts"]}, sort_keys=True))
    return 0 if result["status"] == "PASS_RESOURCE_CENSORED_DEVELOPMENT" else 2


if __name__ == "__main__":
    raise SystemExit(main())
