#!/usr/bin/env python3
"""Recompute N45 attribution after normalizing both branches to their ID axis.

The original N45 result remains immutable.  This repair is isolated under
outputs/n46/n45_attribution_repair and uses the frozen N42 source plus the
repair2 runtime sidecar assignments.
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sam3_intermot.datasets.dancetrack import DanceTrackDataset
from scripts.n36_real_eval_common import DATA_ROOT
from scripts.n43_full_matrix_common import iou
from scripts.n45_stage3_attribution_replay import aggregate, events, load


EVENTS = ROOT / "outputs/n37/real_event_manifest.json"
N42 = ROOT / "outputs/n42/replay/runtime/t0"
N43_MAP = ROOT / "outputs/n43/training/dataset_manifest.json"
RUNTIME = ROOT / "outputs/n46/diagnosis_repair2/events"
OLD_N45_RUNTIME = ROOT / "outputs/n45/replay/runtime"
OLD_N45_RESULT = ROOT / "outputs/n45/replay/attribution_results.json"
OUT = ROOT / "outputs/n46/n45_attribution_repair"
POSTHOC = OUT / "posthoc_events"
RESULT = OUT / "normalized_attribution_results.json"
STATUS = OUT / "status.json"
VARIANTS = ("M0", "M1", "M2", "M3", "M4")
HORIZONS = (20, 50, 100)


def axis_assignment(audit: dict[str, Any], values: list[int] | None = None) -> list[int | None]:
    pids = [int(x) for x in audit.get("public_id_order", [])]
    assignment = values if values is not None else audit.get("assignment_after_scope", audit.get("assignment", []))
    return [pids[int(col)] if 0 <= int(col) < len(pids) else None for col in assignment]


def rows_from_assignment(audit: dict[str, Any], values: list[int] | None = None) -> dict[str, Any]:
    pids = axis_assignment(audit, values)
    rows = [{"native_tid": c.get("native_tid"), "box": c.get("box"), "confidence": c.get("confidence"), "public_id": pids[i] if i < len(pids) else None} for i, c in enumerate(audit.get("candidates", []))]
    return {"frame": int(audit["frame"]), "rows": rows}


def by_native(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {str(x["native_tid"]): x.get("public_id") for x in rows if x.get("native_tid") is not None}


def pid_iou(rows: list[dict[str, Any]], pid: int, target: Any) -> float:
    return max((float(iou(x["box"], target)) for x in rows if x.get("public_id") is not None and int(x["public_id"]) == pid), default=0.0)


def compare(reference: list[dict[str, Any]], treated: list[dict[str, Any]], event: dict[str, Any], mapping: dict[int, int], gt: dict[int, Any], horizon: int) -> dict[str, Any]:
    target_pid, target_gid = int(event["public_id"]), int(event["dataset_gt_id"])
    details = []
    for ref, yes in zip(reference[:horizon], treated[:horizon]):
        frame = gt.get(int(ref["frame"]))
        if frame is None:
            continue
        target = next((box for gid, box in zip(frame.gt_ids, frame.boxes) if int(gid) == target_gid), None)
        if target is None:
            continue
        ri, yi = pid_iou(ref["rows"], target_pid, target), pid_iou(yes["rows"], target_pid, target)
        changed = by_native(ref["rows"]) != by_native(yes["rows"])
        details.append({"frame": int(ref["frame"]), "reference_iou": ri, "treated_iou": yi, "target_iou_delta": yi - ri, "reference_error": ri < 0.5, "treated_error": yi < 0.5, "assignment_changed": changed, "assignment_change_correct": bool(changed and yi > ri + 1e-9), "assignment_change_incorrect": bool(changed and yi < ri - 1e-9), "assignment_change_neutral": bool(changed and abs(yi - ri) <= 1e-9), "assignment_no_change": not changed})
    if not details:
        return {"evaluated_frames": 0, "identity_utility": None, "target_iou_delta": None, "future_identity_error_reduction": None, "recorrection_proxy_reduction": None, "assignment_change_count": 0, "assignment_change_correct_count": 0, "assignment_change_incorrect_count": 0, "assignment_change_neutral_count": 0, "assignment_no_change_count": 0, "untouched_regression": {"status": "NOT_COMPUTABLE"}, "frame_details": []}
    ri = float(np.mean([x["reference_iou"] for x in details])); yi = float(np.mean([x["treated_iou"] for x in details])); re = float(np.mean([x["reference_error"] for x in details])); ye = float(np.mean([x["treated_error"] for x in details]))
    rr = sum(x["reference_error"] and (i == 0 or not details[i - 1]["reference_error"]) for i, x in enumerate(details)); yr = sum(x["treated_error"] and (i == 0 or not details[i - 1]["treated_error"]) for i, x in enumerate(details))
    untouched = []
    for ref, yes in zip(reference[:horizon], treated[:horizon]):
        frame = gt.get(int(ref["frame"]))
        if frame is None:
            continue
        boxes = {int(gid): box for gid, box in zip(frame.gt_ids, frame.boxes)}
        for pid, gid in mapping.items():
            if int(pid) == target_pid or int(gid) not in boxes:
                continue
            untouched.append(pid_iou(yes["rows"], int(pid), boxes[int(gid)]) - pid_iou(ref["rows"], int(pid), boxes[int(gid)]))
    untouched_result = {"compared": len(untouched), "mean_iou_delta": float(np.mean(untouched)) if untouched else None, "regression_count_delta_lt_minus_0.05": int(sum(x < -0.05 for x in untouched)), "all_no_obvious_regression": bool(untouched) and not any(x < -0.05 for x in untouched), "status": "PASS" if untouched and not any(x < -0.05 for x in untouched) else "FAIL" if untouched else "NOT_COMPUTABLE"}
    return {"evaluated_frames": len(details), "identity_utility": 0.5 * (yi - ri) + 0.5 * (re - ye), "target_iou_delta": yi - ri, "future_identity_error_reduction": re - ye, "recorrection_proxy_reduction": int(rr - yr), "assignment_change_count": int(sum(x["assignment_changed"] for x in details)), "assignment_change_correct_count": int(sum(x["assignment_change_correct"] for x in details)), "assignment_change_incorrect_count": int(sum(x["assignment_change_incorrect"] for x in details)), "assignment_change_neutral_count": int(sum(x["assignment_change_neutral"] for x in details)), "assignment_no_change_count": int(sum(x["assignment_no_change"] for x in details)), "untouched_regression": untouched_result, "frame_details": details}


def validate_runtime(event_map: dict[str, dict[str, Any]]) -> dict[str, Any]:
    files = sorted(RUNTIME.glob("*.json"))
    if len(files) != 24 or {p.stem for p in files} != set(event_map):
        raise RuntimeError("repair2 runtime event set is incomplete")
    checked = 0
    for event_id in sorted(event_map):
        payload = load(RUNTIME / f"{event_id}.json")
        if payload.get("runtime_future_gt_used") is not False:
            raise RuntimeError(f"runtime GT flag {event_id}")
        for variant in VARIANTS:
            frames = payload["variants"][variant]
            if len(frames) != 100:
                raise RuntimeError(f"runtime length {event_id}/{variant}")
            for frame in frames:
                if frame.get("runtime_future_gt_used") is not False:
                    raise RuntimeError(f"frame GT flag {event_id}/{variant}/{frame.get('frame')}")
                checked += 1
    return {"status": "PASS", "event_count": 24, "trace_rows_checked": checked, "runtime_future_gt_used": False, "gt_loaded": False}


def main() -> None:
    event_map = events(); validation = validate_runtime(event_map)
    raw_maps = load(N43_MAP)["public_to_gt_mapping"]
    sequences = sorted({str(e["sequence"]) for e in event_map.values()})
    dataset = DanceTrackDataset(str(DATA_ROOT), sequences=sequences, split="train")
    # This is the first GT load, after the complete runtime validation above.
    gt = {sequence: dataset.load_gt(sequence) for sequence in sequences}
    OUT.mkdir(parents=True, exist_ok=True); POSTHOC.mkdir(parents=True, exist_ok=True)
    rows = {("memory", v, h): [] for v in VARIANTS for h in HORIZONS} | {("incremental", v, h): [] for v in VARIANTS for h in HORIZONS}
    axis_mismatch_count = 0; axis_examples = []
    for event_id, event in sorted(event_map.items()):
        source = load(N42 / f"{event_id}.json"); runtime = load(RUNTIME / f"{event_id}.json"); mapping = {int(pid): int(gid) for pid, gid in raw_maps.get(event_id, {}).items()}; event_out = {"event_id": event_id, "sequence": str(event["sequence"]), "interaction_source": "simulated_from_gt", "not_real_human_evidence": True, "runtime_future_gt_used": False, "gt_loaded_posthoc": True, "axis_reconciliation": {"candidate_public_id_rows_not_used_for_assignment": True, "mismatch_rows": 0}, "variants": {}}
        for variant in VARIANTS:
            no_audits = source["variants"][variant]["branches"]["memory_write=False"]["future_trace"]; write_audits = source["variants"][variant]["branches"]["memory_write=True"]["future_trace"]; diag_frames = runtime["variants"][variant]
            no_rows = [rows_from_assignment(x["candidate_audit"]) for x in no_audits]; write_rows = [rows_from_assignment(x["candidate_audit"]) for x in write_audits]; plus_rows = [rows_from_assignment(w["candidate_audit"], d["plus_assignment"]) for w, d in zip(write_audits, diag_frames)]
            for idx, audit in enumerate(write_audits):
                raw = audit["candidate_audit"].get("candidate_public_ids", [])
                normalized = axis_assignment(audit["candidate_audit"])
                if raw != normalized:
                    axis_mismatch_count += 1
                    if len(axis_examples) < 5: axis_examples.append({"event_id": event_id, "variant": variant, "frame": int(audit["frame"]), "raw_candidate_public_ids": raw, "normalized_assignment_public_ids": normalized})
            horizons = {}
            for h in HORIZONS:
                memory = compare(no_rows, write_rows, event, mapping, gt[str(event["sequence"])], h); incremental = compare(write_rows, plus_rows, event, mapping, gt[str(event["sequence"])], h)
                app = {"proposals_considered": int(sum(len(x["proposals"]) for x in diag_frames[:h])), "proposals_selected": int(sum(int(x["selected_count"]) for x in diag_frames[:h])), "selected_but_no_assignment_change": int(sum(int(x["selected_but_no_assignment_change"]) for x in diag_frames[:h])), "changed_cells": int(sum(len(x["changed_cells"]) for x in diag_frames[:h])), "assignment_changed": int(sum(int(x["assignment_changed_count"]) for x in diag_frames[:h]))}
                horizons[str(h)] = {"memory_effect_no_write_to_write_baseline": memory, "n44_incremental_effect_write_baseline_to_write_plus_n44": incremental, "application_counts_through_horizon": app}
                rows[("memory", variant, h)].append({"event_id": event_id, "sequence": str(event["sequence"]), **memory}); rows[("incremental", variant, h)].append({"event_id": event_id, "sequence": str(event["sequence"]), **incremental})
            event_out["variants"][variant] = {"horizons": horizons, "runtime_frame_count": len(diag_frames)}
        event_out["axis_reconciliation"]["mismatch_rows"] = sum(1 for variant in VARIANTS for audit in source["variants"][variant]["branches"]["memory_write=True"]["future_trace"] if audit["candidate_audit"].get("candidate_public_ids", []) != axis_assignment(audit["candidate_audit"]))
        (POSTHOC / f"{event_id}.json").write_text(json.dumps(event_out, indent=2) + "\n", encoding="utf-8")
    aggregates = {effect: {v: {str(h): aggregate(rows[(effect, v, h)]) for h in HORIZONS} for v in VARIANTS} for effect in ("memory", "incremental")}
    old = load(OLD_N45_RESULT)
    result = {"schema": "N45_NORMALIZED_ATTRIBUTION_REPAIR_V1", "status": "PASS", "protocol": {"source": "frozen N42 no/write plus repair2 runtime", "normalization": "both branches mapped from assignment columns through current public_id_order; candidate_public_ids are not used as assignment evidence", "runtime_future_gt_used": False, "gt_loaded_only_after_runtime_validation": True, "bootstrap": "sequence_mean_then_equal_sequence_cluster_bootstrap", "seed": 4444, "replicates": 2000}, "inputs": {"n42_runtime": str(N42), "repair2_runtime": str(RUNTIME), "old_n45_result_preserved": str(OLD_N45_RESULT), "old_n45_runtime": str(OLD_N45_RUNTIME)}, "outputs": {"posthoc_events": str(POSTHOC), "result": str(RESULT), "status": str(STATUS)}, "runtime_validation": validation, "event_count": 24, "independent_sequence_count": len({str(e["sequence"]) for e in event_map.values()}), "variant_count": 5, "horizons": list(HORIZONS), "interaction_source": "simulated_from_gt", "real_human_tape_created": False, "id_switch_metric": "NOT_COMPUTABLE_NO_COMPLETE_TRACKEVAL_SEQUENCE_INPUT", "axis_reconciliation": {"write_source_frames_with_candidate_public_id_axis_mismatch": axis_mismatch_count, "examples": axis_examples, "both_assignment_maps_normalized": True, "old_n45_result_modified": False}, "effects": aggregates, "old_n45_m2_increment_for_comparison": old["effects"]["incremental"]["M2"], "attribution": {"memory_effect": "write_baseline minus no_write", "n44_incremental_effect": "write_plus_n44 minus write_baseline", "corrected_assignment_semantics": "frame-level native-ID mapping after assignment+axis normalization"}}
    RESULT.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    status = {"status": "PASS", "protocol": "N45_NORMALIZED_ATTRIBUTION_REPAIR_STAGE_V1", "command": ["python", "scripts/n45_normalized_attribution_repair.py"], "inputs": result["inputs"], "outputs": result["outputs"], "metrics": {"runtime_validation": validation, "axis_reconciliation": result["axis_reconciliation"], "memory_effect": aggregates["memory"], "n44_incremental_effect": aggregates["incremental"]}, "gate_checks": {"runtime_validated_before_gt": True, "runtime_future_gt_false": True, "all_24_events": True, "all_5_variants": True, "all_100_frames": True, "assignment_plus_axis_normalization": True, "old_n45_preserved": True, "equal_sequence_bootstrap": True, "simulated_provenance_explicit": True, "standard_mot_not_computable": True}, "failure_root_cause": "N45's old posthoc baseline mapping used candidate_public_ids that can contain IDs outside public_id_order; this repair normalizes both branches through assignment columns and the active axis.", "next_action": "Use this normalized result for scientific attribution, keep old N45 result labelled legacy/provisional, and do not authorize production without real human tape/full-loop.", "runtime_future_gt_used": False, "gt_loaded_posthoc": True, "finished_at": datetime.now(timezone.utc).isoformat()}
    STATUS.write_text(json.dumps(status, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "axis_mismatch_frames": axis_mismatch_count, "output": str(RESULT)}))


if __name__ == "__main__":
    main()
