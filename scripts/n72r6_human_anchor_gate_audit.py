#!/usr/bin/env python3
"""Audit human-ROI anchor similarity for the recovered target streams.

Runtime stream/provenance fields are validated before any GT file is opened.
GT is used only afterward to label the diagnostic candidate geometry; no label
is used to select the fixed gate threshold or to alter a replay.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import sys
from typing import Any, Iterable, Mapping

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sam3_intermot.association.branch_public_replay import (  # noqa: E402
    atomic_json,
    atomic_jsonl,
    box_iou,
    read_json,
    read_jsonl,
    sha256_file,
)


GT_ROOT = Path("/data1/LWR/vranlee/DATASETS/JDE/dancetrack/train")
DEFAULT_TARGET_MANIFEST = ROOT / "outputs/N72R6/recovery_target_stream_manifest_attempt3.json"
DEFAULT_REPLAY_ROOT = ROOT / "outputs/N72R6/public_replay/recovery_attempt_1"
DEFAULT_PROTOCOL = ROOT / "outputs/N72R6/human_anchor_gate_protocol.json"
DEFAULT_TABLE = ROOT / "outputs/N72R6/human_anchor_gate_candidate_table.jsonl"
DEFAULT_SUMMARY = ROOT / "outputs/N72R6/human_anchor_gate_audit.json"
DEFAULT_STATUS = ROOT / "outputs/N72R6/stage_08_human_anchor_gate_audit_status.json"


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def resolve(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else ROOT / value


def unit_feature(value: Any, label: str) -> np.ndarray:
    feature = np.asarray(value, dtype=np.float32).reshape(-1)
    if feature.size != 512 or not np.all(np.isfinite(feature)):
        raise ValueError(f"{label}: expected finite 512-D feature")
    norm = float(np.linalg.norm(feature))
    if norm <= 1.0e-6:
        raise ValueError(f"{label}: zero-norm feature")
    return feature / norm


def finite_stats(values: Iterable[float]) -> dict[str, Any]:
    array = np.asarray(list(values), dtype=np.float64)
    if array.size == 0:
        return {"count": 0, "min": None, "p05": None, "median": None, "p95": None, "max": None}
    return {
        "count": int(array.size),
        "min": float(np.min(array)),
        "p05": float(np.quantile(array, 0.05)),
        "median": float(np.median(array)),
        "p95": float(np.quantile(array, 0.95)),
        "max": float(np.max(array)),
    }


def load_gt(sequence: str) -> dict[int, dict[int, list[float]]]:
    path = GT_ROOT / sequence / "gt" / "gt.txt"
    if not path.is_file():
        raise FileNotFoundError(path)
    result: dict[int, dict[int, list[float]]] = defaultdict(dict)
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            parts = [item.strip() for item in line.split(",")]
            if len(parts) < 6:
                raise ValueError(f"malformed GT row: {path}:{line_number}")
            frame = int(parts[0]) - 1
            gt_id = int(parts[1])
            x, y, width, height = (float(item) for item in parts[2:6])
            result[frame][gt_id] = [x, y, x + width, y + height]
    return result


def horizon_group(horizon: int) -> list[str]:
    return [
        name for name, maximum in (("event_plus_one", 1), ("H20", 20), ("H50", 50), ("H100", 100))
        if horizon <= maximum
    ]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-manifest", default=str(DEFAULT_TARGET_MANIFEST))
    parser.add_argument("--replay-root", default=str(DEFAULT_REPLAY_ROOT))
    parser.add_argument("--protocol", default=str(DEFAULT_PROTOCOL))
    parser.add_argument("--table-output", default=str(DEFAULT_TABLE))
    parser.add_argument("--summary-output", default=str(DEFAULT_SUMMARY))
    parser.add_argument("--status-output", default=str(DEFAULT_STATUS))
    args = parser.parse_args()

    target_manifest_path = resolve(args.target_manifest)
    replay_root = resolve(args.replay_root)
    protocol_path = resolve(args.protocol)
    table_path = resolve(args.table_output)
    summary_path = resolve(args.summary_output)
    status_path = resolve(args.status_output)
    protocol = read_json(protocol_path)
    if protocol.get("status") != "PASS_N72R6_HUMAN_ANCHOR_GATE_PROTOCOL_REGISTERED":
        raise RuntimeError(f"unregistered gate protocol: {protocol.get('status')}")
    threshold = float(protocol["threshold"])
    if not math.isfinite(threshold) or not -1.0 <= threshold <= 1.0:
        raise ValueError("gate threshold is not finite in [-1, 1]")

    target_manifest = read_json(target_manifest_path)
    if target_manifest.get("status") != "PASS_N72R6_TARGET_SESSION_RECOVERY_32_OF_32_VALIDATED":
        raise RuntimeError(f"target manifest is not complete: {target_manifest.get('status')}")
    selected = target_manifest.get("selected", [])
    if len(selected) != 32 or len({str(item.get("event_id")) for item in selected}) != 32:
        raise RuntimeError("target manifest does not contain 32 unique events")
    event_policy = read_json(ROOT / "outputs/N72R5/mechanism_rounds/round_06_event_policy/real_event_manifest.json")
    events = {str(item["event_id"]): dict(item) for item in event_policy.get("events", [])}
    selected_ids = {str(item["event_id"]) for item in selected}
    if not selected_ids.issubset(set(events)):
        raise RuntimeError("target manifest contains an event absent from the frozen event policy")

    # First validate all stream/provenance fields.  Only this phase is allowed
    # to precede opening GT; no posthoc label can affect the runtime inputs.
    pending: list[dict[str, Any]] = []
    for selected_item in sorted(selected, key=lambda item: str(item["event_id"])):
        event_id = str(selected_item["event_id"])
        done_path = resolve(str(selected_item["done"]))
        done = read_json(done_path)
        frames_path = resolve(str(done["frames"]))
        anchor_path = resolve(str(done["human_anchor"]))
        anchor_payload = read_json(anchor_path)
        anchor = unit_feature(anchor_payload.get("feature"), f"{event_id}:human_anchor")
        rows = read_jsonl(frames_path)
        event_frame = int(done["event_frame"])
        end_frame = int(done["end_frame"])
        if [int(row.get("frame", -1)) for row in rows] != list(range(event_frame, end_frame + 1)):
            raise ValueError(f"non-contiguous target stream: {event_id}")
        if len(rows) != 101:
            raise ValueError(f"target stream horizon mismatch: {event_id}")
        for row in rows:
            frame = int(row["frame"])
            if any(row.get(flag) is not False for flag in ("runtime_future_gt_used", "runtime_gt_read", "posthoc_gt_used")):
                raise ValueError(f"runtime GT flag is not false: {event_id}:{frame}")
            candidates = row.get("candidate_rows")
            if not isinstance(candidates, list) or len(candidates) > 1:
                raise ValueError(f"target candidate cardinality invalid: {event_id}:{frame}")
            if frame <= event_frame or not candidates:
                continue
            candidate = candidates[0]
            if candidate.get("candidate_kind") != "TARGET_CORRECTION_SESSION_CANDIDATE":
                raise ValueError(f"target candidate kind mismatch: {event_id}:{frame}")
            if any(candidate.get(flag) is not False for flag in ("runtime_future_gt_used", "runtime_gt_read", "posthoc_gt_used", "public_id_inference")):
                raise ValueError(f"candidate GT/mapping flag is not false: {event_id}:{frame}")
            feature = unit_feature(candidate.get("feature"), f"{event_id}:{frame}:candidate")
            cosine = float(np.dot(feature, anchor))
            pending.append({
                "event_id": event_id,
                "sequence": str(done["sequence"]),
                "action_type": str(events[event_id]["action_type"]),
                "frame": frame,
                "horizon": frame - event_frame,
                "candidate_uid": str(candidate["candidate_uid"]),
                "candidate_feature_sha256": str(candidate["feature_sha256"]),
                "human_anchor_sha256": str(anchor_payload.get("feature_sha256")),
                "human_anchor_cosine": cosine,
                "accepted_by_fixed_gate": bool(cosine >= threshold),
                "runtime_future_gt_used": False,
                "posthoc_gt_used": False,
                "candidate_box": [float(value) for value in candidate["box_xyxy"]],
            })

    # Posthoc-only labeling starts here.
    gt_cache: dict[str, dict[int, dict[int, list[float]]]] = {}
    for event_id in sorted(events):
        sequence = str(events[event_id]["sequence"])
        gt_cache[sequence] = load_gt(sequence)
    for item in pending:
        event = events[item["event_id"]]
        gt_box = gt_cache[item["sequence"]].get(item["frame"], {}).get(int(event["dataset_gt_id"]))
        item["candidate_iou_to_posthoc_gt"] = None if gt_box is None else float(box_iou(item["candidate_box"], gt_box))
        item["posthoc_spatial_label"] = (
            "UNDEFINED_NOT_VISIBLE" if gt_box is None
            else ("SPATIAL_HIT" if item["candidate_iou_to_posthoc_gt"] >= 0.5 else "SPATIAL_DRIFT")
        )
        item["posthoc_gt_used"] = True

    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in pending:
        groups["all"].append(item)
        groups[f"action:{item['action_type']}"].append(item)
        for group in horizon_group(int(item["horizon"])):
            groups[f"horizon:{group}"].append(item)
            groups[f"action:{item['action_type']}:horizon:{group}"].append(item)
    summaries: dict[str, Any] = {}
    for name, rows in sorted(groups.items()):
        cosine_values = [float(row["human_anchor_cosine"]) for row in rows]
        visible = [row for row in rows if row["posthoc_spatial_label"] != "UNDEFINED_NOT_VISIBLE"]
        hits = [row for row in visible if row["posthoc_spatial_label"] == "SPATIAL_HIT"]
        accepted = [row for row in rows if row["accepted_by_fixed_gate"]]
        accepted_visible = [row for row in accepted if row["posthoc_spatial_label"] != "UNDEFINED_NOT_VISIBLE"]
        accepted_hits = [row for row in accepted_visible if row["posthoc_spatial_label"] == "SPATIAL_HIT"]
        summaries[name] = {
            "candidate_count": len(rows),
            "human_anchor_cosine": finite_stats(cosine_values),
            "fixed_gate_accept_count": len(accepted),
            "fixed_gate_accept_rate": None if not rows else float(len(accepted) / len(rows)),
            "visible_count": len(visible),
            "spatial_hit_count": len(hits),
            "spatial_hit_rate": None if not visible else float(len(hits) / len(visible)),
            "accepted_visible_count": len(accepted_visible),
            "accepted_spatial_hit_rate": None if not accepted_visible else float(len(accepted_hits) / len(accepted_visible)),
        }

    summary = {
        "schema_version": "N72R6_HUMAN_ANCHOR_GATE_AUDIT_V1",
        "status": "PASS_N72R6_HUMAN_ANCHOR_GATE_AUDIT",
        "event_count": len(selected),
        "sequence_count": len({str(item["sequence"]) for item in selected}),
        "candidate_row_count": len(pending),
        "fixed_threshold": threshold,
        "fixed_threshold_source": str(protocol_path),
        "fixed_threshold_source_sha256": sha256_file(protocol_path),
        "groups": summaries,
        "runtime_future_gt_used": False,
        "posthoc_gt_used": True,
        "interaction_source": "simulated_from_gt",
        "not_real_human_evidence": True,
        "inputs": {
            "target_manifest": str(target_manifest_path),
            "target_manifest_sha256": sha256_file(target_manifest_path),
            "replay_root": str(replay_root),
            "event_manifest": str(ROOT / "outputs/N72R5/mechanism_rounds/round_06_event_policy/real_event_manifest.json"),
        },
        "created_at_utc": now_utc(),
    }
    atomic_jsonl(table_path, pending)
    atomic_json(summary_path, summary)
    atomic_json(
        status_path,
        {
            "schema_version": "N72R6_STAGE_STATUS_V1",
            "stage": "N72R6-08_HUMAN_ANCHOR_GATE_AUDIT",
            "status": summary["status"],
            "summary": str(summary_path),
            "table": str(table_path),
            "event_count": summary["event_count"],
            "candidate_row_count": summary["candidate_row_count"],
            "fixed_threshold": threshold,
            "runtime_future_gt_used": False,
            "posthoc_gt_used": True,
            "created_at_utc": now_utc(),
        },
    )
    print(json.dumps({"status": summary["status"], "event_count": summary["event_count"], "candidate_row_count": summary["candidate_row_count"], "threshold": threshold}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
