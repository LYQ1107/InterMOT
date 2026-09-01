#!/usr/bin/env python3
"""N44 stage 04: same-prefix/event/candidate paired replay and posthoc CI."""

from __future__ import annotations

import copy
import json
import sys
import traceback
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
from scripts.n44_assignment_common import apply_sidecar, load_checkpoint, sha256


EVENTS = ROOT / "outputs/n37/real_event_manifest.json"
N42_T0 = ROOT / "outputs/n42/replay/runtime/t0"
N43_MAP = ROOT / "outputs/n43/training/dataset_manifest.json"
CHECKPOINT = ROOT / "outputs/n44/training/n44_assignment_aware.pt"
OUT = ROOT / "outputs/n44/replay"
RUNTIME = OUT / "runtime"
POSTHOC = OUT / "posthoc_events"
RESULT = OUT / "paired_replay_results.json"
STAGE = ROOT / "outputs/n44/stage_04_status.json"
VARIANTS = ("M0", "M1", "M2", "M3", "M4")
HORIZONS = (20, 50, 100)
BOOTSTRAP_SEED = 4444
BOOTSTRAP_REPS = 2000


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def event_map() -> dict[str, dict[str, Any]]:
    payload = load(EVENTS)
    if payload.get("status") != "PASS" or len(payload.get("events", [])) != 24:
        raise RuntimeError("N37 event manifest invalid")
    return {str(item["event"]["event_id"]): item["event"] for item in payload["events"]}


def source(event_id: str) -> dict[str, Any]:
    payload = load(N42_T0 / f"{event_id}.json")
    if payload.get("status") != "PASS" or payload.get("runtime_boundary", {}).get("runtime_future_gt_used") is not False:
        raise RuntimeError(f"invalid N42 runtime source {event_id}")
    return payload


def slim(audit: dict[str, Any]) -> dict[str, Any]:
    candidates = audit.get("candidates", [])
    pids = list(audit.get("candidate_public_ids", []))
    rows = [{"native_tid": c.get("native_tid"), "box": c.get("box"), "confidence": c.get("confidence"), "public_id": pid} for c, pid in zip(candidates, pids)]
    valid_mapping = len(rows) == len(candidates) and all(x.get("public_id") is None or isinstance(x.get("public_id"), int) for x in rows)
    return {"frame": int(audit["frame"]), "rows": rows, "candidate_count": len(rows), "candidate_set_complete": audit.get("candidate_set_complete") is True, "mapping_complete_or_explicit_none": bool(valid_mapping), "assignment": list(audit.get("assignment", audit.get("assignment_after_scope", []))), "public_id_order": list(audit.get("public_id_order", [])), "n44_sidecar": audit.get("n44_sidecar"), "runtime_future_gt_used": False}


def runtime(events: dict[str, dict[str, Any]]) -> dict[str, Any]:
    model, checkpoint = load_checkpoint(CHECKPOINT, "cpu")
    gate = checkpoint.get("gate")
    if not isinstance(gate, dict):
        raise RuntimeError("N44 checkpoint has no frozen gate")
    RUNTIME.mkdir(parents=True, exist_ok=True)
    application_totals = {"frames": 0, "proposals_considered": 0, "proposals_selected": 0, "changed_cells": 0, "changed_assignments": 0}
    for event_id in sorted(events):
        payload = source(event_id)
        event = events[event_id]
        variants = {}
        for variant in VARIANTS:
            src_variant = payload["variants"][variant]
            no_source = src_variant["branches"]["memory_write=False"]
            write_source = src_variant["branches"]["memory_write=True"]
            no_trace, write_trace = [], []
            previous = src_variant.get("event_frame_audit", {}).get("candidate_audit", {})
            for no_entry, write_entry in zip(no_source["future_trace"], write_source["future_trace"]):
                if int(no_entry["frame"]) != int(write_entry["frame"]):
                    raise RuntimeError(f"prefix/frame mismatch {event_id}/{variant}")
                no_audit, write_audit = no_entry["candidate_audit"], write_entry["candidate_audit"]
                no_trace.append(slim(no_audit))
                if variant == "M0":
                    calibrated = copy.deepcopy(no_audit)
                else:
                    calibrated = apply_sidecar(write_audit, model, int(write_audit["frame"]) - int(event["frame"]), previous, gate)
                    sidecar_stats = calibrated.get("n44_sidecar", {})
                    application_totals["frames"] += 1
                    for source_key, target_key in (("proposals_considered", "proposals_considered"), ("proposals_selected", "proposals_selected"), ("changed_cell_count", "changed_cells"), ("changed_assignment_count", "changed_assignments")):
                        application_totals[target_key] += int(sidecar_stats.get(source_key, 0))
                write_trace.append(slim(calibrated))
                previous = write_audit
            if len(no_trace) != 100 or len(write_trace) != 100 or [x["frame"] for x in no_trace] != [x["frame"] for x in write_trace]:
                raise RuntimeError(f"future trace incomplete or changed {event_id}/{variant}")
            variants[variant] = {"no_write": no_trace, "write": write_trace, "memory_variant": variant, "sidecar_enabled": variant != "M0", "runtime_future_gt_used": False}
        out = {"protocol": "N44_FROZEN_CANDIDATE_ASSIGNMENT_AWARE_PAIRED_RUNTIME_V1", "status": "PASS", "event_id": event_id, "sequence": str(event["sequence"]), "action_type": str(event["action_type"]), "event_frame": int(event["frame"]), "future_frame_start": int(event["frame"]) + 1, "future_frame_count": 100, "interaction_source": "simulated_from_gt", "not_real_human_evidence": True, "prefix_contract": "same N42 frozen prefix/event/candidate stream", "checkpoint": str(CHECKPOINT), "checkpoint_sha256": sha256(CHECKPOINT), "frozen_gate": gate, "variants": variants, "runtime_boundary": {"runtime_future_gt_used": False, "gt_loaded_in_worker": False, "future_gt_fields_sent": [], "event_frame_memory_read": False, "first_future_frame": int(event["frame"]) + 1}}
        (RUNTIME / f"{event_id}.json").write_text(json.dumps(out, separators=(",", ":")) + "\n", encoding="utf-8")
    return {"status": "PASS", "event_count": len(events), "runtime_artifacts": str(RUNTIME), "checkpoint": str(CHECKPOINT), "checkpoint_sha256": sha256(CHECKPOINT), "application_totals": application_totals, "runtime_future_gt_used": False}


def validate(events: dict[str, dict[str, Any]]) -> dict[str, Any]:
    checked = 0
    for event_id in sorted(events):
        payload = load(RUNTIME / f"{event_id}.json")
        if payload.get("status") != "PASS" or payload.get("runtime_boundary", {}).get("runtime_future_gt_used") is not False:
            raise RuntimeError(f"runtime boundary invalid {event_id}")
        for variant in VARIANTS:
            for branch in ("no_write", "write"):
                trace = payload["variants"][variant][branch]
                if len(trace) != 100 or [int(row["frame"]) for row in trace] != list(range(int(trace[0]["frame"]), int(trace[0]["frame"]) + 100)):
                    raise RuntimeError(f"frame gap/duplicate {event_id}/{variant}/{branch}")
                for row in trace:
                    if row.get("runtime_future_gt_used") is not False or row.get("candidate_set_complete") is not True or row.get("mapping_complete_or_explicit_none") is not True:
                        raise RuntimeError(f"candidate/boundary invalid {event_id}/{variant}/{branch}/{row['frame']}")
                    natives = [int(x["native_tid"]) for x in row["rows"] if x.get("native_tid") is not None]
                    if len(natives) != len(set(natives)):
                        raise RuntimeError(f"duplicate native id {event_id}/{variant}/{branch}/{row['frame']}")
                    checked += 1
                if branch == "write":
                    no_trace = payload["variants"][variant]["no_write"]
                    for no_row, write_row in zip(no_trace, trace):
                        no_keys = [(x.get("native_tid"), x.get("box"), x.get("confidence")) for x in no_row["rows"]]
                        write_keys = [(x.get("native_tid"), x.get("box"), x.get("confidence")) for x in write_row["rows"]]
                        if no_keys != write_keys or no_row.get("candidate_count") != write_row.get("candidate_count"):
                            raise RuntimeError(f"candidate stream changed {event_id}/{variant}/{write_row['frame']}")
    return {"status": "PASS", "artifact_count": len(events), "trace_rows_checked": checked, "duplicate_or_missing_frame": False, "candidate_complete": True, "runtime_future_gt_used": False}


def by_native(row: dict[str, Any]) -> dict[str, Any]:
    return {str(x["native_tid"]): x.get("public_id") for x in row["rows"] if x.get("native_tid") is not None}


def pid_iou(row: dict[str, Any], pid: int, target: Any) -> float:
    return max((iou(x["box"], target) for x in row["rows"] if x.get("public_id") is not None and int(x["public_id"]) == pid), default=0.0)


def metric(no_trace: list[dict[str, Any]], yes_trace: list[dict[str, Any]], event: dict[str, Any], mapping: dict[int, int], gt: dict[int, Any], horizon: int) -> dict[str, Any]:
    pid, gid = int(event["public_id"]), int(event["dataset_gt_id"])
    details = []
    for no, yes in zip(no_trace[:horizon], yes_trace[:horizon]):
        frame = gt.get(int(no["frame"]))
        if frame is None:
            continue
        target = next((box for item, box in zip(frame.gt_ids, frame.boxes) if int(item) == gid), None)
        if target is None:
            continue
        ni, yi = pid_iou(no, pid, target), pid_iou(yes, pid, target)
        changed = by_native(no) != by_native(yes)
        details.append({"frame": int(no["frame"]), "no_write_iou": ni, "write_iou": yi, "no_write_error": ni < 0.5, "write_error": yi < 0.5, "assignment_changed": changed, "assignment_change_correct": bool(changed and yi > ni + 1e-9), "assignment_change_incorrect": bool(changed and yi < ni - 1e-9), "assignment_no_change": not changed})
    if not details:
        return {"evaluated_frames": 0, "identity_utility": None, "future_identity_error_reduction": None, "target_mean_iou_no_write": None, "target_mean_iou_write": None, "assignment_change_count": 0, "assignment_change_correct_count": 0, "assignment_change_incorrect_count": 0, "assignment_no_change_count": 0, "recorrection_proxy_reduction": None, "untouched_regression": {"status": "NOT_COMPUTABLE"}, "frame_details": []}
    ni = float(np.mean([x["no_write_iou"] for x in details])); yi = float(np.mean([x["write_iou"] for x in details]))
    ne = float(np.mean([x["no_write_error"] for x in details])); ye = float(np.mean([x["write_error"] for x in details]))
    nr = sum(x["no_write_error"] and (i == 0 or not details[i - 1]["no_write_error"]) for i, x in enumerate(details))
    yr = sum(x["write_error"] and (i == 0 or not details[i - 1]["write_error"]) for i, x in enumerate(details))
    others = [p for p in mapping if int(p) != pid]
    deltas = []
    for no, yes in zip(no_trace[:horizon], yes_trace[:horizon]):
        frame = gt.get(int(no["frame"]))
        if frame is None:
            continue
        boxes = {int(k): box for k, box in zip(frame.gt_ids, frame.boxes)}
        for other in others:
            if mapping[other] not in boxes:
                continue
            deltas.append(pid_iou(yes, int(other), boxes[mapping[other]]) - pid_iou(no, int(other), boxes[mapping[other]]))
    untouched = {"compared": len(deltas), "mean_iou_delta": float(np.mean(deltas)) if deltas else None, "regression_count_delta_lt_minus_0.05": int(sum(x < -0.05 for x in deltas)), "all_no_obvious_regression": bool(deltas) and not any(x < -0.05 for x in deltas), "status": "PASS" if deltas and not any(x < -0.05 for x in deltas) else "FAIL" if deltas else "NOT_COMPUTABLE"}
    return {"evaluated_frames": len(details), "identity_utility": 0.5 * (yi - ni) + 0.5 * (ne - ye), "future_identity_error_reduction": ne - ye, "target_mean_iou_no_write": ni, "target_mean_iou_write": yi, "recorrection_proxy_reduction": int(nr - yr), "assignment_change_count": int(sum(x["assignment_changed"] for x in details)), "assignment_change_correct_count": int(sum(x["assignment_change_correct"] for x in details)), "assignment_change_incorrect_count": int(sum(x["assignment_change_incorrect"] for x in details)), "assignment_no_change_count": int(sum(x["assignment_no_change"] for x in details)), "untouched_regression": untouched, "frame_details": details}


def posthoc(events: dict[str, dict[str, Any]], validation: dict[str, Any]) -> dict[str, Any]:
    # This function is called only after validate(); GT is loaded here and
    # nowhere in runtime().
    raw_maps = load(N43_MAP)["public_to_gt_mapping"]
    sequences = sorted({str(event["sequence"]) for event in events.values()})
    dataset = DanceTrackDataset(str(DATA_ROOT), sequences=sequences, split="train")
    gt = {sequence: dataset.load_gt(sequence) for sequence in sequences}
    rows: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for event_id, event in sorted(events.items()):
        mapping = {int(pid): int(gid) for pid, gid in raw_maps.get(event_id, {}).items()}
        payload = load(RUNTIME / f"{event_id}.json")
        event_result = {"event_id": event_id, "sequence": str(event["sequence"]), "action_type": str(event["action_type"]), "interaction_source": "simulated_from_gt", "not_real_human_evidence": True, "mapping_used_for_posthoc_only": mapping, "horizons": {}}
        for variant in VARIANTS:
            event_result["horizons"][variant] = {}
            for horizon in HORIZONS:
                value = metric(payload["variants"][variant]["no_write"], payload["variants"][variant]["write"], event, mapping, gt[str(event["sequence"])], horizon)
                event_result["horizons"][variant][str(horizon)] = value
                rows[(variant, horizon)].append({"sequence": str(event["sequence"]), "action_type": str(event["action_type"]), **value})
        (POSTHOC / f"{event_id}.json").parent.mkdir(parents=True, exist_ok=True)
        (POSTHOC / f"{event_id}.json").write_text(json.dumps(event_result, indent=2) + "\n", encoding="utf-8")
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    aggregates = {}
    for variant in VARIANTS:
        aggregates[variant] = {}
        for horizon in HORIZONS:
            selected = rows[(variant, horizon)]
            by_sequence: dict[str, list[float]] = defaultdict(list)
            for item in selected:
                if item["identity_utility"] is not None:
                    by_sequence[item["sequence"]].append(float(item["identity_utility"]))
            cluster_values = {key: float(np.mean(value)) for key, value in by_sequence.items() if value}
            names = sorted(cluster_values)
            boot = [float(np.mean([cluster_values[name] for name in rng.choice(names, size=len(names), replace=True)])) for _ in range(BOOTSTRAP_REPS)] if names else []
            untouched = [item["untouched_regression"] for item in selected]
            utility = [item["identity_utility"] for item in selected if item["identity_utility"] is not None]
            aggregates[variant][str(horizon)] = {"event_count": len(selected), "independent_sequence_count": len(names), "identity_utility": float(np.mean(utility)) if utility else None, "future_identity_error_reduction": float(np.mean([x["future_identity_error_reduction"] for x in selected])) if selected else None, "recorrection_proxy_reduction": float(np.mean([x["recorrection_proxy_reduction"] for x in selected])) if selected else None, "assignment_change_count": int(sum(x["assignment_change_count"] for x in selected)), "assignment_change_correct_count": int(sum(x["assignment_change_correct_count"] for x in selected)), "assignment_change_incorrect_count": int(sum(x["assignment_change_incorrect_count"] for x in selected)), "assignment_no_change_count": int(sum(x["assignment_no_change_count"] for x in selected)), "untouched_regression": {"all_no_obvious_regression": bool(untouched) and all(x.get("all_no_obvious_regression", False) for x in untouched), "mean_iou_delta": float(np.mean([x["mean_iou_delta"] for x in untouched if x.get("mean_iou_delta") is not None])) if untouched else None}, "sequence_cluster_bootstrap_95ci": {"lower": float(np.quantile(boot, 0.025)) if boot else None, "upper": float(np.quantile(boot, 0.975)) if boot else None, "seed": BOOTSTRAP_SEED, "replicates": BOOTSTRAP_REPS, "clusters": len(names), "cluster_weighting": "equal_sequence_mean", "cluster_mean_identity_utility": float(np.mean(list(cluster_values.values()))) if cluster_values else None, "event_weighted_identity_utility": float(np.mean(utility)) if utility else None}}
    final = {"protocol": "N44_ASSIGNMENT_AWARE_PAIRED_REPLAY_AND_POSTHOC_V1", "status": "PASS", "event_count": len(events), "independent_sequence_count": len({str(x["sequence"]) for x in events.values()}), "variant_count": 5, "horizons": list(HORIZONS), "runtime_validation": validation, "runtime_future_gt_used": False, "gt_loaded_only_after_runtime_validation": True, "interaction_source": "simulated_from_gt", "real_human_tape_created": False, "id_switch_metric": "NOT_COMPUTABLE_NO_COMPLETE_TRACKEVAL_SEQUENCE_INPUT", "aggregates": aggregates, "posthoc_event_artifact_count": len(events), "bootstrap_protocol": "sequence_mean_then_equal_sequence_cluster_bootstrap"}
    RESULT.write_text(json.dumps(final, indent=2) + "\n", encoding="utf-8")
    return final


def main() -> None:
    status: dict[str, Any] = {"status": "FAIL", "protocol": "N44_STAGE_04_PAIRED_REPLAY_V1", "started_at": now(), "project_root": str(ROOT)}
    try:
        events = event_map()
        run = runtime(events)
        validation = validate(events)
        final = posthoc(events, validation)
        status.update({"status": "PASS", "command": [sys.executable, str(Path(__file__).resolve())], "inputs": {"n37_event_manifest": str(EVENTS), "n42_frozen_runtime": str(N42_T0), "n43_offline_mapping_manifest": str(N43_MAP), "n44_checkpoint": str(CHECKPOINT)}, "outputs": {"runtime_dir": str(RUNTIME), "posthoc_dir": str(POSTHOC), "result": str(RESULT)}, "metrics": {"runtime": run, "replay": final["aggregates"]}, "gate_checks": {"same_prefix": True, "same_events": True, "same_candidates": True, "runtime_future_gt_false": True, "posthoc_gt_only_after_runtime_validation": True, "all_24_events": True, "all_5_variants": True, "all_horizons": True, "sequence_cluster_equal_weight": True, "untouched_regression_reported": True, "standard_mot": "NOT_COMPUTABLE_NO_COMPLETE_TRACKEVAL_SEQUENCE_INPUT", "real_human_tape_available": False, "real_full_loop": False}, "failure_root_cause": "N43 changed finite cells without assignment-aware global utility; N44 applies only frozen-gate, bounded proposals and leaves all other cells at the current branch baseline. This replay remains simulated_from_gt and is not a real SAM3 full-loop.", "next_action": "Run post-replay targeted regressions, perform up to three evidence-preserving N40 real-tape feasibility checks, then write the strict final gate/report without production authorization.", "runtime_future_gt_used": False, "finished_at": now()})
        STAGE.write_text(json.dumps(status, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({"status": "PASS", "events": len(events), "result": str(RESULT)}))
    except Exception as exc:
        status.update({"status": "FAIL", "failure_root_cause": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc(), "finished_at": now()})
        failure = ROOT / "outputs/n44/attempts" / f"stage_04_failure_{datetime.now().strftime('%Y%m%dT%H%M%S')}.json"
        failure.parent.mkdir(parents=True, exist_ok=True)
        failure.write_text(json.dumps(status, indent=2) + "\n", encoding="utf-8")
        raise


if __name__ == "__main__":
    main()
