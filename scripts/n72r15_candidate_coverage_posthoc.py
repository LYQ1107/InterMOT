#!/usr/bin/env python3
"""Posthoc candidate coverage diagnosis for the N72R15 sealed replay."""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
from typing import Any, Mapping

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import n72r11_on_demand_replay as replay  # noqa: E402


HORIZONS = (20, 50, 100)
IOU_THRESHOLD = 0.50
VARIANTS = ("E0_BASELINE_B0", "E1I_HUMAN_RELATIVE_STATE", "E1J_TRUSTED_GLOBAL_RELATIVE_STATE")
DEFAULT_MANIFEST = ROOT / "outputs/N72R15/formal/formal_manifest.json"
DEFAULT_OUTPUT = ROOT / "outputs/N72R15/candidate_coverage.json"


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object: {path}")
    return value


def _iou(a: Any, b: Any) -> float:
    left = max(float(a[0]), float(b[0]))
    top = max(float(a[1]), float(b[1]))
    right = min(float(a[2]), float(b[2]))
    bottom = min(float(a[3]), float(b[3]))
    inter = max(0.0, right - left) * max(0.0, bottom - top)
    area_a = max(0.0, float(a[2]) - float(a[0])) * max(0.0, float(a[3]) - float(a[1]))
    area_b = max(0.0, float(b[2]) - float(b[0])) * max(0.0, float(b[3]) - float(b[1]))
    union = area_a + area_b - inter
    return inter / union if union > 0.0 else 0.0


def _load_event_records(manifest: Mapping[str, Any]) -> list[dict[str, Any]]:
    if manifest.get("status") != "PASS_N72R15_FORMAL_REPLAY" or int(manifest.get("event_count", -1)) != 32:
        raise RuntimeError("formal manifest is not complete")
    records = list(manifest.get("events", []))
    if len(records) != 32 or len({str(item["event_id"]) for item in records}) != 32:
        raise RuntimeError("formal event keys are not unique/complete")
    return [dict(item) for item in records]


def _coverage_bucket() -> dict[str, Any]:
    return {
        "target_visible_frames": 0,
        "main_covered_frames": 0,
        "target_session_covered_frames": 0,
        "target_session_only_rescue_frames": 0,
        "neither_source_coverage_failure_frames": 0,
        "main_coverage": None,
        "target_session_coverage": None,
        "target_session_only_rescue": None,
        "neither_source_failure": None,
    }


def _finish_bucket(bucket: dict[str, Any]) -> dict[str, Any]:
    denom = int(bucket["target_visible_frames"])
    if denom:
        bucket["main_coverage"] = float(bucket["main_covered_frames"] / denom)
        bucket["target_session_coverage"] = float(bucket["target_session_covered_frames"] / denom)
        bucket["target_session_only_rescue"] = float(bucket["target_session_only_rescue_frames"] / denom)
        bucket["neither_source_failure"] = float(bucket["neither_source_coverage_failure_frames"] / denom)
    return bucket


def coverage(manifest_path: Path = DEFAULT_MANIFEST, output_path: Path = DEFAULT_OUTPUT) -> dict[str, Any]:
    manifest = read_json(manifest_path)
    event_records = _load_event_records(manifest)
    protocol = read_json(ROOT / "outputs/N72R9/protocol.json")
    protocol_events = {str(item["event_id"]): dict(item) for item in protocol["source_event_selection"]["events"]}
    records: list[dict[str, Any]] = []
    aggregate = {
        str(horizon): {"all": _coverage_bucket(), "by_action": defaultdict(_coverage_bucket), "by_sequence": defaultdict(_coverage_bucket)}
        for horizon in HORIZONS
    }
    for event_record in sorted(event_records, key=lambda item: str(item["event_id"])):
        event_id = str(event_record["event_id"])
        event = protocol_events[event_id]
        inputs = replay._load_inputs(event, horizon=100)
        gt = replay.legacy._load_gt(str(event["sequence"]))
        target_gid = int(event["dataset_gt_id"])
        event_result = {
            "event_id": event_id,
            "sequence": str(event["sequence"]),
            "action_type": str(event["action_type"]),
            "event_frame": int(event["event_frame"]),
            "target_dataset_gt_id": target_gid,
            "horizons": {},
            "runtime_future_gt_used": False,
            "posthoc_gt_used": True,
        }
        for horizon in HORIZONS:
            bucket = _coverage_bucket()
            frame_rows: list[dict[str, Any]] = []
            for frame in range(int(event["event_frame"]) + 1, int(event["event_frame"]) + horizon + 1):
                target = gt.get(frame, {}).get(target_gid)
                if target is None:
                    continue
                main_rows = list(inputs["rows"]["c0_source"][frame].get("candidate_rows", []))
                target_rows = list(inputs["rows"]["target_stream_source"][frame].get("candidate_rows", []))
                main_best = max((_iou(row.get("box_xyxy", row.get("box")), target["box"]) for row in main_rows), default=0.0)
                target_best = max((_iou(row.get("box_xyxy", row.get("box")), target["box"]) for row in target_rows), default=0.0)
                main_hit, target_hit = main_best >= IOU_THRESHOLD, target_best >= IOU_THRESHOLD
                bucket["target_visible_frames"] += 1
                bucket["main_covered_frames"] += int(main_hit)
                bucket["target_session_covered_frames"] += int(target_hit)
                bucket["target_session_only_rescue_frames"] += int(not main_hit and target_hit)
                bucket["neither_source_coverage_failure_frames"] += int(not main_hit and not target_hit)
                frame_rows.append({
                    "frame": frame,
                    "main_best_iou": float(main_best),
                    "target_session_best_iou": float(target_best),
                    "main_covered": main_hit,
                    "target_session_covered": target_hit,
                    "target_session_only_rescue": bool(not main_hit and target_hit),
                    "neither_source_coverage_failure": bool(not main_hit and not target_hit),
                    "runtime_future_gt_used": False,
                })
            _finish_bucket(bucket)
            event_result["horizons"][str(horizon)] = bucket | {"frames": frame_rows}
            for key in ("all", "by_action", "by_sequence"):
                dest = aggregate[str(horizon)]["all"] if key == "all" else aggregate[str(horizon)][key][str(event["action_type"] if key == "by_action" else event["sequence"])]
                for count_key in ("target_visible_frames", "main_covered_frames", "target_session_covered_frames", "target_session_only_rescue_frames", "neither_source_coverage_failure_frames"):
                    dest[count_key] += bucket[count_key]
        records.append(event_result)
    aggregate_out: dict[str, Any] = {}
    for horizon in HORIZONS:
        aggregate_out[str(horizon)] = {}
        for key in ("all", "by_action", "by_sequence"):
            values = aggregate[str(horizon)][key]
            if key == "all":
                aggregate_out[str(horizon)][key] = _finish_bucket(values)
            else:
                aggregate_out[str(horizon)][key] = {name: _finish_bucket(bucket) for name, bucket in sorted(values.items())}
    output = {
        "schema_version": "N72R15_CANDIDATE_COVERAGE_POSTHOC_V1",
        "status": "PASS_N72R15_CANDIDATE_COVERAGE_POSTHOC",
        "created_at_utc": now_utc(),
        "source_formal_manifest": str(manifest_path),
        "source_formal_manifest_sha256": sha256_file(manifest_path),
        "iou_threshold": IOU_THRESHOLD,
        "horizons": list(HORIZONS),
        "aggregate": aggregate_out,
        "events": records,
        "runtime_future_gt_used": False,
        "posthoc_gt_used": True,
        "diagnostic_only": True,
        "historical_outputs_modified": False,
    }
    atomic_json(output_path, output)
    return output


def main() -> int:
    parser = __import__("argparse").ArgumentParser()
    parser.add_argument("--formal-manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    try:
        result = coverage(args.formal_manifest.resolve(), args.output.resolve())
        print(json.dumps({"status": result["status"], "event_count": len(result["events"]), "output": str(args.output.resolve())}, sort_keys=True))
        return 0
    except Exception as exc:
        failure = {"schema_version": "N72R15_FAILURE_V1", "status": "FAIL_N72R15_CANDIDATE_COVERAGE", "error_type": type(exc).__name__, "error": str(exc), "created_at_utc": now_utc(), "runtime_future_gt_used": False}
        atomic_json(args.output.resolve().with_name("candidate_coverage_failure.json"), failure)
        print(json.dumps(failure, sort_keys=True))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
