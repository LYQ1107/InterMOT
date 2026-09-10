#!/usr/bin/env python3
"""Posthoc state-to-edge-to-solver propagation audit for N72R15."""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import tempfile
from typing import Any, Mapping

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sam3_intermot.evaluation.interaction_effect_metrics import metric_record  # noqa: E402
from scripts import n72r11_on_demand_replay as replay  # noqa: E402


VARIANTS = ("E0_BASELINE_B0", "E1I_HUMAN_RELATIVE_STATE", "E1J_TRUSTED_GLOBAL_RELATIVE_STATE")
TREATMENTS = VARIANTS[1:]
HORIZON = 100
IOU_THRESHOLD = 0.50
DEFAULT_MANIFEST = ROOT / "outputs/N72R15/formal/formal_manifest.json"
DEFAULT_OUTPUT = ROOT / "outputs/N72R15/state_propagation_audit.json"


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


def _assignment_map(row: Mapping[str, Any]) -> dict[int, str | None]:
    result: dict[int, str | None] = {}
    for item in row.get("assignment", {}).get("public_assignments", []):
        if item.get("public_id") is None:
            continue
        public = int(item["public_id"])
        if public in result:
            raise RuntimeError(f"duplicate public assignment {public}")
        uid = item.get("candidate_uid")
        result[public] = None if uid in (None, "", "None") else str(uid)
    return result


def _edge(row: Mapping[str, Any]) -> np.ndarray:
    state_info = row.get("persistent_state_association", {})
    # N72R15 runtime rows persist matrices with explicit ``*_matrix`` names.
    # Keep the fallback for early smoke artifacts, but audit the canonical
    # formal schema first instead of silently treating a present edge as empty.
    raw_value = state_info.get("relative_delta_matrix", state_info.get("relative_delta", []))
    value = np.asarray(raw_value, dtype=np.float64)
    if value.ndim != 2 or not np.isfinite(value).all():
        raise RuntimeError(f"invalid relative edge at {row.get('event_id')}:{row.get('frame')}")
    return value


def _target_uid(row: Mapping[str, Any], target_public: int) -> str | None:
    values = [item for item in row.get("candidate_rows", []) if item.get("public_id") is not None and int(item["public_id"]) == int(target_public)]
    if len(values) > 1:
        raise RuntimeError("duplicate target public assignment")
    return None if not values else str(values[0]["candidate_uid"])


def _audit_event(event_record: Mapping[str, Any], protocol_event: Mapping[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    paths = {str(item["variant"]): Path(str(item["frames"])) for item in event_record["variants"]}
    rows = {variant: [json.loads(line) for line in paths[variant].read_text(encoding="utf-8").splitlines() if line.strip()] for variant in VARIANTS}
    event_frame = int(event_record["event_frame"])
    target_public = int(event_record.get("target_public_id", protocol_event.get("target_public_id", 0)))
    target_gid = int(protocol_event["dataset_gt_id"])
    inputs = replay._load_inputs(protocol_event, horizon=100)
    gt = replay.legacy._load_gt(str(protocol_event["sequence"]))
    protected = replay.legacy._protected_map(inputs["rows"]["c0_source"][event_frame], gt, event_frame, target_gid)
    records: list[dict[str, Any]] = []
    baseline = rows[VARIANTS[0]]
    for variant in VARIANTS:
        current = rows[variant]
        state_changed_rows = [row for row in current[1:] if row.get("persistent_state_association", {}).get("state_changed") is True]
        transitions: list[dict[str, Any]] = []
        for index in range(1, HORIZON):
            row = current[index]
            if row.get("persistent_state_association", {}).get("state_changed") is not True:
                continue
            next_row = current[index + 1]
            left, right = _edge(row), _edge(next_row)
            edge_changed = bool(left.shape != right.shape or np.any(np.abs(left - right) > 1.0e-9))
            base_map = _assignment_map(baseline[index])
            current_map = _assignment_map(row)
            solver_changed = base_map != current_map
            base_uid, current_uid = _target_uid(baseline[index], target_public), _target_uid(row, target_public)
            target_changed = base_uid != current_uid
            target_gt = gt.get(int(row["frame"]), {}).get(target_gid)
            change_type = "UNASSESSABLE_NO_TARGET_GT"
            if target_gt is not None:
                base_iou, _ = replay.legacy._public_box_for_gt(baseline[index], target_public, target_gt["box"])
                current_iou, _ = replay.legacy._public_box_for_gt(row, target_public, target_gt["box"])
                record = metric_record(
                    baseline_iou=float(base_iou),
                    treatment_iou=float(current_iou),
                    baseline_correct=bool(base_iou >= IOU_THRESHOLD),
                    treatment_correct=bool(current_iou >= IOU_THRESHOLD),
                    assignment_changed=target_changed,
                )
                change_type = str(record["assignment_change_type"])
            transitions.append({
                "frame": int(row["frame"]),
                "next_frame": int(next_row["frame"]),
                "state_changed": True,
                "edge_changed_next_frame": edge_changed,
                "solver_changed_from_b0": solver_changed,
                "target_assignment_changed_from_b0": target_changed,
                "assignment_change_type": change_type,
                "runtime_future_gt_used": False,
            })
        solver_changed_count = 0
        correct = incorrect = 0
        assignment_changed_count = 0
        cardinality_changed_count = 0
        disagreement_trusted_write_count = 0
        for index in range(1, HORIZON + 1):
            base_map, current_map = _assignment_map(baseline[index]), _assignment_map(current[index])
            solver_changed_count += int(base_map != current_map)
            assignment_changed_count += int(_target_uid(baseline[index], target_public) != _target_uid(current[index], target_public))
            base_count = sum(value is not None for value in base_map.values())
            current_count = sum(value is not None for value in current_map.values())
            cardinality_changed_count += int(base_count != current_count)
            info = current[index].get("persistent_state_association", {})
            changed = set(info.get("treatment_changed_public_ids", []))
            writes = set(current[index].get("state_update", {}).get("machine_memory_write_public_ids", []))
            disagreement_trusted_write_count += int(bool(changed & writes))
            target_gt = gt.get(int(current[index]["frame"]), {}).get(target_gid)
            if target_gt is not None and _target_uid(baseline[index], target_public) != _target_uid(current[index], target_public):
                base_iou, _ = replay.legacy._public_box_for_gt(baseline[index], target_public, target_gt["box"])
                current_iou, _ = replay.legacy._public_box_for_gt(current[index], target_public, target_gt["box"])
                record = metric_record(
                    baseline_iou=float(base_iou), treatment_iou=float(current_iou),
                    baseline_correct=bool(base_iou >= IOU_THRESHOLD), treatment_correct=bool(current_iou >= IOU_THRESHOLD),
                    assignment_changed=True,
                )
                correct += int(record["true_correct_crossing"])
                incorrect += int(record["true_incorrect_crossing"])
        records.append({
            "event_id": str(event_record["event_id"]),
            "sequence": str(event_record["sequence"]),
            "action_type": str(event_record["action_type"]),
            "variant": variant,
            "event_frame": event_frame,
            "state_changed_frame_count": len(state_changed_rows),
            "next_edge_checked_count": len(transitions),
            "next_edge_changed_count": sum(item["edge_changed_next_frame"] for item in transitions),
            "state_changed_but_next_edge_unchanged_count": sum(not item["edge_changed_next_frame"] for item in transitions),
            "solver_changed_frame_count": solver_changed_count,
            "target_assignment_changed_count": assignment_changed_count,
            "correct_assignment_change_count": correct,
            "incorrect_assignment_change_count": incorrect,
            "assignment_cardinality_changed_frame_count": cardinality_changed_count,
            "disagreement_trusted_write_count": disagreement_trusted_write_count,
            "row_max_preservation_failure_count": sum(not bool(row.get("persistent_state_association", {}).get("row_max_preserved", True)) for row in current[1:]),
            "transitions": transitions,
            "runtime_future_gt_used": False,
            "posthoc_gt_used": True,
        })
    return {"event_id": str(event_record["event_id"]), "sequence": str(event_record["sequence"]), "action_type": str(event_record["action_type"])}, records


def audit(manifest_path: Path = DEFAULT_MANIFEST, output_path: Path = DEFAULT_OUTPUT) -> dict[str, Any]:
    manifest = read_json(manifest_path)
    if manifest.get("status") != "PASS_N72R15_FORMAL_REPLAY":
        raise RuntimeError("formal replay is not PASS")
    protocol = read_json(ROOT / "outputs/N72R9/protocol.json")
    protocol_events = {str(item["event_id"]): dict(item) for item in protocol["source_event_selection"]["events"]}
    all_records: list[dict[str, Any]] = []
    for event_record in manifest["events"]:
        _, records = _audit_event(event_record, protocol_events[str(event_record["event_id"])])
        all_records.extend(records)
    if len(all_records) != 32 * len(VARIANTS):
        raise RuntimeError(f"propagation record count {len(all_records)} != {32 * len(VARIANTS)}")
    summaries: dict[str, Any] = {}
    for variant in VARIANTS:
        selected = [item for item in all_records if item["variant"] == variant]
        by_action: dict[str, list[dict[str, Any]]] = defaultdict(list)
        by_sequence: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for item in selected:
            by_action[item["action_type"]].append(item)
            by_sequence[item["sequence"]].append(item)
        summary = {
            "event_count": len(selected),
            "independent_sequence_count": len(by_sequence),
            "state_changed_frame_count": sum(item["state_changed_frame_count"] for item in selected),
            "next_edge_checked_count": sum(item["next_edge_checked_count"] for item in selected),
            "next_edge_changed_count": sum(item["next_edge_changed_count"] for item in selected),
            "state_changed_but_next_edge_unchanged_count": sum(item["state_changed_but_next_edge_unchanged_count"] for item in selected),
            "solver_changed_frame_count": sum(item["solver_changed_frame_count"] for item in selected),
            "target_assignment_changed_count": sum(item["target_assignment_changed_count"] for item in selected),
            "correct_assignment_change_count": sum(item["correct_assignment_change_count"] for item in selected),
            "incorrect_assignment_change_count": sum(item["incorrect_assignment_change_count"] for item in selected),
            "assignment_cardinality_changed_frame_count": sum(item["assignment_cardinality_changed_frame_count"] for item in selected),
            "disagreement_trusted_write_count": sum(item["disagreement_trusted_write_count"] for item in selected),
            "row_max_preservation_failure_count": sum(item["row_max_preservation_failure_count"] for item in selected),
            "action_breakdown": {
                action: {
                    "event_count": len(items),
                    "state_changed_frame_count": sum(item["state_changed_frame_count"] for item in items),
                    "next_edge_changed_count": sum(item["next_edge_changed_count"] for item in items),
                    "solver_changed_frame_count": sum(item["solver_changed_frame_count"] for item in items),
                    "correct_assignment_change_count": sum(item["correct_assignment_change_count"] for item in items),
                    "incorrect_assignment_change_count": sum(item["incorrect_assignment_change_count"] for item in items),
                }
                for action, items in sorted(by_action.items())
            },
            "sequence_breakdown": {
                sequence: {
                    "event_count": len(items),
                    "state_changed_frame_count": sum(item["state_changed_frame_count"] for item in items),
                    "next_edge_changed_count": sum(item["next_edge_changed_count"] for item in items),
                    "solver_changed_frame_count": sum(item["solver_changed_frame_count"] for item in items),
                }
                for sequence, items in sorted(by_sequence.items())
            },
        }
        summary["status"] = "PASS_N72R15_STATE_PROPAGATION" if (
            variant == "E0_BASELINE_B0"
            or (summary["next_edge_checked_count"] > 0 and summary["state_changed_but_next_edge_unchanged_count"] == 0 and summary["row_max_preservation_failure_count"] == 0)
        ) else "BLOCKED_N72R15_STATE_PROPAGATION"
        summaries[variant] = summary
    output = {
        "schema_version": "N72R15_STATE_PROPAGATION_AUDIT_V1",
        "status": "PASS_N72R15_STATE_PROPAGATION" if all(item["status"] == "PASS_N72R15_STATE_PROPAGATION" for item in summaries.values()) else "BLOCKED_N72R15_STATE_PROPAGATION",
        "created_at_utc": now_utc(),
        "source_formal_manifest": str(manifest_path),
        "source_formal_manifest_sha256": sha256_file(manifest_path),
        "variants": list(VARIANTS),
        "records": all_records,
        "summary": summaries,
        "runtime_future_gt_used": False,
        "posthoc_gt_used": True,
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
        result = audit(args.formal_manifest.resolve(), args.output.resolve())
        print(json.dumps({"status": result["status"], "record_count": len(result["records"]), "output": str(args.output.resolve())}, sort_keys=True))
        return 0 if result["status"].startswith("PASS") else 2
    except Exception as exc:
        failure = {"schema_version": "N72R15_FAILURE_V1", "status": "FAIL_N72R15_STATE_PROPAGATION_AUDIT", "error_type": type(exc).__name__, "error": str(exc), "traceback": __import__("traceback").format_exc(), "created_at_utc": now_utc(), "runtime_future_gt_used": False}
        atomic_json(args.output.resolve().with_name("state_propagation_audit_failure.json"), failure)
        print(json.dumps(failure, sort_keys=True))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
