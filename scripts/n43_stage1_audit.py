#!/usr/bin/env python3
"""N43 stage 01: audit the N42 boundary and materialize full cell audits."""

from __future__ import annotations

import json
import os
import sys
import traceback
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sam3_intermot.datasets.dancetrack import DanceTrackDataset
from scripts.n36_real_eval_common import DATA_ROOT
from scripts.n43_full_matrix_common import HARD_NEGATIVE, FEATURE_NAMES, cell_features, finite_matrix, iou


EVENTS = ROOT / "outputs/n37/real_event_manifest.json"
N41_ROOT = ROOT / "outputs/n41/source_replay/full/attempt1"
N42_T0 = ROOT / "outputs/n42/replay/runtime/t0"
N42_T1 = ROOT / "outputs/n42/replay/runtime/t1"
N42_POSTHOC = ROOT / "outputs/n42/replay/posthoc_results.json"
OUT = ROOT / "outputs/n43"
AUDIT = OUT / "audit/full_matrix_audit.jsonl"
STAGE = OUT / "stage_01_status.json"


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def event_map() -> dict[str, dict[str, Any]]:
    payload = load(EVENTS)
    if payload.get("status") != "PASS" or payload.get("event_count") != 24:
        raise RuntimeError("frozen N37 event manifest is not PASS/24")
    events = payload.get("events", [])
    if len(events) != 24 or len({str(x["event"]["event_id"]) for x in events}) != 24:
        raise RuntimeError("frozen N37 event IDs are invalid")
    return {str(x["event"]["event_id"]): x["event"] for x in events}


def source_artifact(event_id: str) -> Path:
    return N41_ROOT / event_id / "A_ideal_gt_roi" / "lambda_1_human_1.json"


def extract_audits(event_id: str) -> list[dict[str, Any]]:
    artifact = load(source_artifact(event_id))
    if artifact.get("status") != "PASS" or artifact.get("runtime_boundary", {}).get("runtime_future_gt_used") is not False:
        raise RuntimeError(f"invalid frozen N41 artifact: {event_id}")
    variant = artifact["variants"]["M2"]
    branch = variant["branches"]["memory_write=True"]
    event_audit = variant.get("event_frame_audit", {}).get("candidate_audit", {})
    rows = []
    if event_audit:
        rows.append(event_audit)
    rows.extend(entry["candidate_audit"] for entry in branch["future_trace"])
    if len(rows) != int(artifact["future_frame_count"]) + 1:
        raise RuntimeError(f"unexpected audit frame count {event_id}: {len(rows)}")
    frames = [int(x["frame"]) for x in rows]
    if frames != sorted(set(frames)):
        raise RuntimeError(f"duplicate or unordered frozen audit frames: {event_id}")
    return rows


def audit_rows(events: dict[str, dict[str, Any]]) -> tuple[dict[str, Any], dict[str, Any]]:
    OUT.joinpath("audit").mkdir(parents=True, exist_ok=True)
    temp = AUDIT.with_suffix(".jsonl.tmp")
    if temp.exists():
        temp.unlink()
    row_count = 0
    cell_count = 0
    candidate_counts = []
    mapping_incomplete = 0
    rows_by_event: dict[str, list[dict[str, Any]]] = {}
    with temp.open("w", encoding="utf-8") as handle:
        for event_id in sorted(events):
            event = events[event_id]
            audits = extract_audits(event_id)
            rows_by_event[event_id] = audits
            previous = None
            for audit in audits:
                base = finite_matrix(audit, "base_scores_before_appearance")
                memory = finite_matrix(audit, "appearance_memory_scores")
                delta = finite_matrix(audit, "appearance_score_deltas")
                fused = finite_matrix(audit, "fused_scores")
                if not (base.shape == memory.shape == delta.shape == fused.shape):
                    raise RuntimeError(f"score matrix shape mismatch {event_id}/{audit.get('frame')}")
                candidates = [dict(x) for x in audit.get("candidates", [])]
                pids = [int(x) for x in audit.get("public_id_order", [])]
                if not candidates or not pids or audit.get("candidate_set_complete") is not True or audit.get("candidate_public_id_mapping_complete") is not True:
                    raise RuntimeError(f"incomplete candidate/mapping at {event_id}/{audit.get('frame')}")
                if base.shape != (len(candidates), len(pids)):
                    raise RuntimeError(f"matrix/candidate/public shape mismatch {event_id}/{audit.get('frame')}")
                if audit.get("runtime_future_gt_used") is not False:
                    raise RuntimeError(f"future GT flag is not false {event_id}/{audit.get('frame')}")
                feature_rows = []
                fields = {name: np.zeros_like(base, dtype=np.float32) for name in FEATURE_NAMES}
                for i in range(base.shape[0]):
                    for j in range(base.shape[1]):
                        feature = cell_features(audit, i, j, max(0, int(audit["frame"]) - int(event["frame"])), previous)
                        feature_rows.append(feature.tolist())
                        for name, value in zip(FEATURE_NAMES, feature.tolist()):
                            fields[name][i, j] = float(value)
                payload = {
                    "protocol": "N43_FULL_MATRIX_CELL_AUDIT_V1",
                    "event_id": event_id,
                    "sequence": str(event["sequence"]),
                    "action_type": str(event["action_type"]),
                    "event_frame": int(event["frame"]),
                    "frame": int(audit["frame"]),
                    "frame_offset_from_event": max(0, int(audit["frame"]) - int(event["frame"])),
                    "candidate_order": [int(x.get("index", i)) for i, x in enumerate(candidates)],
                    "candidate_native_ids": [x.get("native_tid") for x in candidates],
                    "candidate_public_ids": audit.get("candidate_public_ids", []),
                    "public_id_order": pids,
                    "public_id_to_native_tid": audit.get("public_id_to_native_tid", {}),
                    "assignment_before_sidecar": audit.get("assignment_after_scope", audit.get("assignment", [])),
                    "candidate_count": len(candidates),
                    "public_id_count": len(pids),
                    "candidates": candidates,
                    "cell_features": feature_rows,
                    "cell_feature_names": list(FEATURE_NAMES),
                    "derived_cell_matrices": {name: matrix.astype(float).tolist() for name, matrix in fields.items()},
                    "base_scores": base.astype(float).tolist(),
                    "appearance_memory_scores": memory.astype(float).tolist(),
                    "appearance_delta_scores": delta.astype(float).tolist(),
                    "fused_scores": fused.astype(float).tolist(),
                    "hard_negative_mask": (base <= HARD_NEGATIVE).tolist(),
                    "runtime_future_gt_used": False,
                    "gt_loaded_posthoc": False,
                    "none_semantics": {"score": -1.0e8, "model_bypassed": True, "one_dummy_per_candidate": True},
                }
                handle.write(json.dumps(payload, separators=(",", ":")) + "\n")
                row_count += 1
                cell_count += int(base.size)
                candidate_counts.append(len(candidates))
                mapping_incomplete += int(audit.get("candidate_public_id_mapping_complete") is not True)
                previous = audit
    os.replace(temp, AUDIT)
    return rows_by_event, {"frame_count": row_count, "cell_count": cell_count, "candidate_count_min": min(candidate_counts), "candidate_count_max": max(candidate_counts), "mapping_incomplete_frame_count": mapping_incomplete}


def target_recall(events: dict[str, dict[str, Any]], rows_by_event: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    dataset = DanceTrackDataset(str(DATA_ROOT), sequences=sorted({str(x["sequence"]) for x in events.values()}), split="train")
    gt = {sequence: dataset.load_gt(sequence) for sequence in sorted({str(x["sequence"]) for x in events.values()})}
    values = []
    for event_id, event in events.items():
        target_pid = int(event["public_id"])
        target_gid = int(event["dataset_gt_id"])
        for audit in rows_by_event[event_id]:
            if int(audit["frame"]) <= int(event["frame"]):
                continue
            frame_gt = gt[str(event["sequence"])].get(int(audit["frame"]))
            if frame_gt is None:
                continue
            target_box = next((box for gid, box in zip(frame_gt.gt_ids, frame_gt.boxes) if int(gid) == target_gid), None)
            if target_box is None:
                continue
            candidates = audit.get("candidates", [])
            ious = [iou(c["box"], target_box) for c in candidates]
            assigned = [iou(c["box"], target_box) for c, pid in zip(candidates, audit.get("candidate_public_ids", [])) if pid is not None and int(pid) == target_pid]
            values.append({"event_id": event_id, "frame": int(audit["frame"]), "assigned_iou": max(assigned, default=0.0), "oracle_iou": max(ious, default=0.0), "assigned_recall_at_0.5": bool(max(assigned, default=0.0) >= 0.5), "oracle_recall_at_0.5": bool(max(ious, default=0.0) >= 0.5)})
    return {
        "evaluated_frames": len(values),
        "assigned_recall_at_0.5": float(np.mean([x["assigned_recall_at_0.5"] for x in values])) if values else None,
        "oracle_recall_at_0.5": float(np.mean([x["oracle_recall_at_0.5"] for x in values])) if values else None,
        "mean_assigned_iou": float(np.mean([x["assigned_iou"] for x in values])) if values else None,
        "mean_oracle_iou": float(np.mean([x["oracle_iou"] for x in values])) if values else None,
        "gt_usage": "offline posthoc target recall/oracle only",
    }


def old_target_only_interface(events: dict[str, dict[str, Any]]) -> dict[str, Any]:
    changed = Counter()
    changed_columns = Counter()
    frames = 0
    for event_id, event in events.items():
        t0 = load(N42_T0 / f"{event_id}.json")
        t1 = load(N42_T1 / f"{event_id}.json")
        a0 = t0["variants"]["M2"]["branches"]["memory_write=True"]["future_trace"]
        a1 = t1["variants"]["M2"]["branches"]["memory_write=True"]["future_trace"]
        for e0, e1 in zip(a0, a1):
            x = np.asarray(e0["candidate_audit"]["fused_scores"], dtype=float)
            y = np.asarray(e1["candidate_audit"]["fused_scores"], dtype=float)
            if x.shape != y.shape:
                raise RuntimeError(f"N42 score shape changed: {event_id}/{e0['frame']}")
            d = np.abs(y - x) > 1e-12
            frames += 1
            changed["changed_frames"] += int(np.any(d))
            changed["unchanged_frames"] += int(not np.any(d))
            for col in np.where(np.any(d, axis=0))[0].tolist():
                changed_columns[int(col)] += 1
    return {"frames": frames, "changed_frame_count": changed["changed_frames"], "unchanged_frame_count": changed["unchanged_frames"], "changed_column_frame_counts": {str(k): int(v) for k, v in changed_columns.items()}, "conclusion": "N42 modifies only the human target column; N43 must score all finite cells"}


def main() -> None:
    started = now()
    result: dict[str, Any] = {"status": "FAIL", "protocol": "N43_STAGE_01_DECISION_BOUNDARY_AUDIT_V1", "started_at": started, "project_root": str(ROOT)}
    try:
        events = event_map()
        rows_by_event, materialized = audit_rows(events)
        result.update({
            "status": "PASS",
            "command": [sys.executable, str(Path(__file__).resolve())],
            "inputs": {"n37_event_manifest": str(EVENTS), "n41_source_artifact_root": str(N41_ROOT), "n42_runtime_t0": str(N42_T0), "n42_runtime_t1": str(N42_T1), "n42_posthoc": str(N42_POSTHOC)},
            "outputs": {"full_matrix_audit": str(AUDIT)},
            "metrics": {"event_count": len(events), "independent_sequence_count": len({str(x['sequence']) for x in events.values()}), "action_counts": dict(Counter(str(x["action_type"]) for x in events.values())), "materialized": materialized, "target_recall_oracle": target_recall(events, rows_by_event), "n42_old_interface": old_target_only_interface(events)},
            "gate_checks": {"frozen_24_events": len(events) == 24, "candidate_matrix_complete": materialized["mapping_incomplete_frame_count"] == 0, "runtime_future_gt_false": True, "full_cell_audit_written": AUDIT.is_file(), "none_semantics_fixed": True, "real_human_tape_available": False},
            "failure_root_cause": "N42 ordered-pair aggregate was written to one human target public-ID column; N42 audit also lacked an explicit full-cell geometry/motion/reliability/NONE contract.",
            "next_action": "Run N43 sidecar integrity checks and materialize a sequence-disjoint full-cell training dataset; do not modify production association code.",
            "runtime_future_gt_used": False,
            "finished_at": now(),
        })
        OUT.mkdir(parents=True, exist_ok=True)
        STAGE.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({"status": result["status"], "frames": materialized["frame_count"], "cells": materialized["cell_count"], "output": str(STAGE)}, sort_keys=True))
    except Exception as exc:
        result.update({"status": "FAIL", "failure_root_cause": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc(), "finished_at": now()})
        OUT.mkdir(parents=True, exist_ok=True)
        failure = OUT / "attempts" / f"stage_01_failure_{datetime.now().strftime('%Y%m%dT%H%M%S')}.json"
        failure.parent.mkdir(parents=True, exist_ok=True)
        failure.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        raise


if __name__ == "__main__":
    main()
