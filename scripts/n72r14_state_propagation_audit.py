#!/usr/bin/env python3
"""Audit causal state-to-edge propagation in sealed N72R14 runtime rows."""

from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import tempfile
import sys
from typing import Any, Mapping, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

MANIFEST = ROOT / "outputs/N72R14/formal_manifest.json"
OUTPUT = ROOT / "outputs/N72R14/state_causal_propagation_audit.json"
VARIANTS = ("E1G_TARGET_STATE_ONLY", "E1H_PERSISTENT_GLOBAL_STATE")


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


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


def _edge_array(row: Mapping[str, Any]) -> np.ndarray:
    edge = row.get("persistent_state_association", {}).get("edge", {})
    value = np.asarray(edge.get("delta", []), dtype=np.float64)
    if value.ndim != 2 or not np.isfinite(value).all():
        raise ValueError(f"invalid edge matrix at {row.get('event_id')}:{row.get('frame')}")
    return value


def _target_uid(row: Mapping[str, Any], target_public: int) -> str | None:
    values = [item for item in row.get("candidate_rows", []) if item.get("public_id") is not None and int(item["public_id"]) == int(target_public)]
    if len(values) > 1:
        raise ValueError(f"duplicate target public assignment at {row.get('event_id')}:{row.get('frame')}")
    return None if not values else str(values[0]["candidate_uid"])


def _mean_abs_common(left: np.ndarray, right: np.ndarray) -> float:
    if left.shape != right.shape:
        return 1.0
    return float(np.mean(np.abs(right - left))) if left.size else 0.0


def _event_audit(event_record: Mapping[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    event_id = str(event_record["event_id"])
    rows_by_variant: dict[str, list[dict[str, Any]]] = {}
    for variant_record in event_record.get("variants", []):
        variant = str(variant_record["variant"])
        path = Path(str(variant_record["frames"]))
        rows_by_variant[variant] = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    baseline = rows_by_variant["E0_BASELINE_B0"]
    event_frame = int(event_record["event_frame"])
    target_public = int(baseline[0]["target_public_id"])
    records: list[dict[str, Any]] = []
    for variant in VARIANTS:
        rows = rows_by_variant[variant]
        state_change_frames = [
            row for row in rows[1:] if row.get("persistent_state_association", {}).get("state_changed") is True
        ]
        transitions: list[dict[str, Any]] = []
        for index, row in enumerate(rows[1:-1], start=1):
            state_info = row.get("persistent_state_association", {})
            state_changed = state_info.get("state_changed") is True
            if not state_changed:
                continue
            next_row = rows[index + 1]
            left, right = _edge_array(row), _edge_array(next_row)
            changed = bool(left.shape != right.shape or np.any(np.abs(left - right) > 1.0e-9))
            baseline_uid = _target_uid(baseline[index], target_public)
            variant_uid = _target_uid(row, target_public)
            target_assignment_changed = bool(baseline_uid != variant_uid)
            target_col = None
            target_edge_change = None
            non_target_edge_change = None
            edge = row.get("persistent_state_association", {}).get("edge", {})
            axis = [int(value) for value in edge.get("public_id_axis", [])]
            if target_public in axis:
                target_col = axis.index(target_public)
                next_edge = next_row.get("persistent_state_association", {}).get("edge", {})
                next_axis = [int(value) for value in next_edge.get("public_id_axis", [])]
                next_array = _edge_array(next_row)
                if target_public in next_axis and target_col < left.shape[1]:
                    next_col = next_axis.index(target_public)
                    target_edge_change = _mean_abs_common(left[:, target_col:target_col + 1], next_array[:, next_col:next_col + 1])
                other = np.delete(left, target_col, axis=1)
                next_other = np.delete(next_array, next_axis.index(target_public), axis=1) if target_public in next_axis else next_array
                non_target_edge_change = _mean_abs_common(other, next_other)
            transitions.append({
                "frame": int(row["frame"]),
                "next_frame": int(next_row["frame"]),
                "state_changed": True,
                "next_edge_changed": changed,
                "edge_shape": list(left.shape),
                "next_edge_shape": list(right.shape),
                "edge_mean_abs_change": _mean_abs_common(left, right),
                "target_edge_mean_abs_change": target_edge_change,
                "non_target_edge_mean_abs_change": non_target_edge_change,
                "target_assignment_changed_from_b0": target_assignment_changed,
                "runtime_future_gt_used": False,
            })
        records.append({
            "event_id": event_id,
            "sequence": str(event_record["sequence"]),
            "action_type": str(event_record["action_type"]),
            "variant": variant,
            "event_frame": event_frame,
            "state_changed_frame_count": len(state_change_frames),
            "next_edge_checked_count": len(transitions),
            "next_edge_changed_count": int(sum(item["next_edge_changed"] for item in transitions)),
            "state_changed_but_next_edge_unchanged_count": int(sum(not item["next_edge_changed"] for item in transitions)),
            "target_assignment_changed_count": int(sum(item["target_assignment_changed_from_b0"] for item in transitions)),
            "mean_edge_l1_change": float(np.mean([item["edge_mean_abs_change"] for item in transitions])) if transitions else 0.0,
            "mean_target_edge_change": float(np.mean([item["target_edge_mean_abs_change"] for item in transitions if item["target_edge_mean_abs_change"] is not None])) if any(item["target_edge_mean_abs_change"] is not None for item in transitions) else 0.0,
            "mean_non_target_edge_change": float(np.mean([item["non_target_edge_mean_abs_change"] for item in transitions if item["non_target_edge_mean_abs_change"] is not None])) if any(item["non_target_edge_mean_abs_change"] is not None for item in transitions) else 0.0,
            "transitions": transitions,
            "runtime_future_gt_used": False,
        })
    return {"event_id": event_id, "sequence": str(event_record["sequence"]), "action_type": str(event_record["action_type"])}, records


def aggregate(manifest_path: Path = MANIFEST, output_path: Path = OUTPUT) -> dict[str, Any]:
    manifest = read_json(manifest_path)
    if manifest.get("status") != "PASS_N72R14_FORMAL_REPLAY":
        raise RuntimeError("formal replay is not PASS")
    all_records: list[dict[str, Any]] = []
    for event in manifest.get("events", []):
        _, records = _event_audit(event)
        all_records.extend(records)
    if len(all_records) != 32 * len(VARIANTS):
        raise RuntimeError(f"propagation record count {len(all_records)} != 64")
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
            "mean_edge_l1_change": float(np.mean([item["mean_edge_l1_change"] for item in selected])),
            "mean_target_edge_change": float(np.mean([item["mean_target_edge_change"] for item in selected])),
            "mean_non_target_edge_change": float(np.mean([item["mean_non_target_edge_change"] for item in selected])),
            "action_breakdown": {
                action: {
                    "event_count": len(items),
                    "state_changed_frame_count": sum(item["state_changed_frame_count"] for item in items),
                    "next_edge_checked_count": sum(item["next_edge_checked_count"] for item in items),
                    "next_edge_changed_count": sum(item["next_edge_changed_count"] for item in items),
                    "target_assignment_changed_count": sum(item["target_assignment_changed_count"] for item in items),
                }
                for action, items in sorted(by_action.items())
            },
            "sequence_breakdown": {
                sequence: {
                    "event_count": len(items),
                    "state_changed_frame_count": sum(item["state_changed_frame_count"] for item in items),
                    "next_edge_changed_count": sum(item["next_edge_changed_count"] for item in items),
                }
                for sequence, items in sorted(by_sequence.items())
            },
        }
        summary["status"] = "PASS_STATE_CONNECTED_TO_SOLVER" if (
            summary["state_changed_frame_count"] > 0
            and summary["next_edge_checked_count"] > 0
            and summary["state_changed_but_next_edge_unchanged_count"] == 0
        ) else "BLOCKED_STATE_NOT_CONNECTED_TO_SOLVER"
        summaries[variant] = summary
    status = "PASS_N72R14_STATE_CAUSAL_PROPAGATION" if all(item["status"] == "PASS_STATE_CONNECTED_TO_SOLVER" for item in summaries.values()) else "BLOCKED_STATE_NOT_CONNECTED_TO_SOLVER"
    output = {
        "schema_version": "N72R14_STATE_CAUSAL_PROPAGATION_AUDIT_V1",
        "status": status,
        "created_at_utc": now_utc(),
        "source_manifest": str(manifest_path),
        "source_manifest_sha256": sha256_file(manifest_path),
        "variants": list(VARIANTS),
        "records": all_records,
        "summary": summaries,
        "runtime_future_gt_used": False,
        "historical_outputs_modified": False,
    }
    atomic_json(output_path, output)
    return output


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    manifest_path = args.manifest.resolve()
    output_path = args.output.resolve()
    try:
        result = aggregate(manifest_path, output_path)
        print(json.dumps({"status": result["status"], "output": str(output_path), "record_count": len(result["records"])}, sort_keys=True))
        return 0 if result["status"].startswith("PASS") else 2
    except Exception as exc:
        failure = {"schema_version": "N72R14_FAILURE_V1", "status": "FAIL_N72R14_STATE_PROPAGATION_AUDIT", "error_type": type(exc).__name__, "error": str(exc), "created_at_utc": now_utc(), "runtime_future_gt_used": False}
        atomic_json(output_path.with_name("state_causal_propagation_audit_failure.json"), failure)
        print(json.dumps(failure, sort_keys=True))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
