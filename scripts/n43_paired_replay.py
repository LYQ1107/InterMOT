#!/usr/bin/env python3
"""N43 stage 04: frozen-candidate paired replay, then post-hoc evaluation."""

from __future__ import annotations

import copy
import gc
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
from scripts.n43_full_matrix_common import apply_sidecar, iou, load_checkpoint


EVENTS = ROOT / "outputs/n37/real_event_manifest.json"
N42_T0 = ROOT / "outputs/n42/replay/runtime/t0"
CHECKPOINT = ROOT / "outputs/n43/training/n43_full_matrix_calibration.pt"
DATASET_MANIFEST = ROOT / "outputs/n43/training/dataset_manifest.json"
OUT = ROOT / "outputs/n43/replay"
RUNTIME = OUT / "runtime"
POSTHOC = OUT / "posthoc_events"
RESULT = OUT / "paired_replay_results.json"
LEGACY_RESULT = OUT / "paired_replay_results_legacy_event_weighted.json"
STAGE = ROOT / "outputs/n43/stage_04_status.json"
HORIZONS = (20, 50, 100)
VARIANTS = ("M0", "M1", "M2", "M3", "M4")
BOOTSTRAP_SEED = 4242
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


def runtime_source(event_id: str) -> dict[str, Any]:
    path = N42_T0 / f"{event_id}.json"
    payload = load(path)
    if payload.get("status") != "PASS" or payload.get("runtime_boundary", {}).get("runtime_future_gt_used") is not False:
        raise RuntimeError(f"invalid N42 runtime source {path}")
    return payload


def slim_audit(audit: dict[str, Any]) -> dict[str, Any]:
    candidates = audit.get("candidates", [])
    pids = list(audit.get("candidate_public_ids", []))
    rows = []
    for candidate, pid in zip(candidates, pids):
        rows.append({"native_tid": candidate.get("native_tid"), "box": candidate.get("box"), "confidence": candidate.get("confidence"), "public_id": pid})
    explicit_none = int(sum(item.get("public_id") is None for item in rows))
    valid_or_none = all(item.get("public_id") is None or isinstance(item.get("public_id"), int) for item in rows)
    # N43's explicit NONE dummy is a valid terminal mapping, not a missing
    # public identity.  Every row must still be present and map to a known
    # public-ID or NONE.
    complete = len(rows) == len(candidates) and valid_or_none
    return {"frame": int(audit["frame"]), "rows": rows, "candidate_count": len(rows), "candidate_set_complete": audit.get("candidate_set_complete") is True, "candidate_public_id_mapping_complete": bool(complete), "mapping_complete_or_explicit_none": bool(complete), "explicit_none_assignment_count": explicit_none, "assignment": list(audit.get("assignment_after_scope", audit.get("assignment", []))), "public_id_order": list(audit.get("public_id_order", [])), "runtime_future_gt_used": False}


def assignment_by_native(trace_row: dict[str, Any]) -> dict[str, Any]:
    output = {}
    for row in trace_row.get("rows", []):
        native = row.get("native_tid")
        if native is not None:
            output[str(native)] = row.get("public_id")
    return output


def runtime_replay(events: dict[str, dict[str, Any]]) -> dict[str, Any]:
    if not CHECKPOINT.is_file():
        raise FileNotFoundError(CHECKPOINT)
    model, metadata = load_checkpoint(CHECKPOINT, "cpu")
    RUNTIME.mkdir(parents=True, exist_ok=True)
    completed = []
    for event_id in sorted(events):
        existing = RUNTIME / f"{event_id}.json"
        if existing.is_file():
            try:
                cached = load(existing)
                if cached.get("status") == "PASS" and len(cached.get("variants", {})) == len(VARIANTS) and all(len(cached["variants"][v]["no_write"]) == 100 and len(cached["variants"][v]["write"]) == 100 for v in VARIANTS):
                    completed.append(event_id)
                    continue
            except Exception:
                pass
        source = runtime_source(event_id)
        event = events[event_id]
        variants = {}
        for variant in VARIANTS:
            source_variant = source["variants"][variant]
            no_source = source_variant["branches"]["memory_write=False"]
            write_source = source_variant["branches"]["memory_write=True"]
            no_trace, write_trace = [], []
            previous = source_variant.get("event_frame_audit", {}).get("candidate_audit", {})
            for no_entry, write_entry in zip(no_source["future_trace"], write_source["future_trace"]):
                if int(no_entry["frame"]) != int(write_entry["frame"]):
                    raise RuntimeError(f"prefix/candidate frame mismatch {event_id}/{variant}")
                no_audit = no_entry["candidate_audit"]
                write_audit = write_entry["candidate_audit"]
                no_trace.append(slim_audit(no_audit))
                if variant == "M0":
                    calibrated = copy.deepcopy(no_audit)
                    calibrated["runtime_future_gt_used"] = False
                else:
                    calibrated = apply_sidecar(write_audit, model, max(0, int(write_audit["frame"]) - int(event["frame"])), previous)
                write_trace.append(slim_audit(calibrated))
                previous = write_audit
            if len(no_trace) != 100 or len(write_trace) != 100:
                raise RuntimeError(f"future frame count is not 100 {event_id}/{variant}")
            if [x["frame"] for x in no_trace] != [x["frame"] for x in write_trace]:
                raise RuntimeError(f"candidate frame stream changed {event_id}/{variant}")
            variants[variant] = {"no_write": no_trace, "write": write_trace, "sidecar_enabled": variant != "M0", "memory_variant": variant, "runtime_future_gt_used": False}
        payload = {"protocol": "N43_FROZEN_CANDIDATE_PAIRED_RUNTIME_REPLAY_V1", "status": "PASS", "event_id": event_id, "sequence": str(event["sequence"]), "action_type": str(event["action_type"]), "event_frame": int(event["frame"]), "future_frame_start": int(event["frame"]) + 1, "future_frame_count": 100, "interaction_source": "simulated_from_gt", "not_real_human_evidence": True, "prefix_contract": "same N42 frozen prefix and event; runtime source has no future GT", "candidate_contract": "same N42 candidate rows/order; only memory branch and N43 full-cell sidecar differ", "checkpoint": str(CHECKPOINT), "checkpoint_sha256": __import__("hashlib").sha256(CHECKPOINT.read_bytes()).hexdigest(), "variants": variants, "runtime_boundary": {"runtime_future_gt_used": False, "gt_loaded_in_worker": False, "future_gt_fields_sent": [], "event_frame_memory_read": False, "first_future_frame": int(event["frame"]) + 1}}
        path = RUNTIME / f"{event_id}.json"
        path.write_text(json.dumps(payload, separators=(",", ":")) + "\n", encoding="utf-8")
        completed.append(event_id)
        del source, variants, payload
        gc.collect()
    return {"status": "PASS", "event_count": len(completed), "runtime_artifacts": [str(RUNTIME / f"{eid}.json") for eid in sorted(events)], "checkpoint": str(CHECKPOINT), "checkpoint_sha256": __import__("hashlib").sha256(CHECKPOINT.read_bytes()).hexdigest(), "runtime_future_gt_used": False, "gt_loaded_after_runtime_validation": False, "resumed_existing_event_count": len(completed) - (len(events) - len(completed)) if False else None}


def validate_runtime(events: dict[str, dict[str, Any]]) -> dict[str, Any]:
    checked = 0
    issues = []
    for event_id in sorted(events):
        path = RUNTIME / f"{event_id}.json"
        payload = load(path)
        if payload.get("status") != "PASS" or payload.get("runtime_boundary", {}).get("runtime_future_gt_used") is not False:
            issues.append(f"artifact boundary {event_id}")
        for variant in VARIANTS:
            for branch in ("no_write", "write"):
                trace = payload["variants"][variant][branch]
                frames = [int(row["frame"]) for row in trace]
                if len(trace) != 100 or frames != list(range(frames[0], frames[0] + 100)):
                    issues.append(f"frame gap/duplicate {event_id}/{variant}/{branch}")
                for row in trace:
                    legacy_none_ok = all(item.get("public_id") is None or isinstance(item.get("public_id"), int) for item in row.get("rows", []))
                    mapping_ok = row.get("mapping_complete_or_explicit_none")
                    if mapping_ok is None:
                        mapping_ok = legacy_none_ok
                    if row.get("runtime_future_gt_used") is not False or row.get("candidate_set_complete") is not True or mapping_ok is not True:
                        issues.append(f"candidate/boundary {event_id}/{variant}/{branch}/{row.get('frame')}")
                    if len({int(x["native_tid"]) for x in row["rows"] if x.get("native_tid") is not None}) != len([x for x in row["rows"] if x.get("native_tid") is not None]):
                        issues.append(f"duplicate native id {event_id}/{variant}/{branch}/{row.get('frame')}")
                    checked += 1
    if issues:
        raise RuntimeError("runtime integrity failed: " + "; ".join(issues[:8]))
    return {"status": "PASS", "artifact_count": len(events), "trace_rows_checked": checked, "duplicate_or_missing_frame": False, "candidate_complete": True, "runtime_future_gt_used": False}


def target_iou(row: dict[str, Any], target_pid: int, target_box: Any) -> float:
    return max((iou(item["box"], target_box) for item in row.get("rows", []) if item.get("public_id") is not None and int(item["public_id"]) == target_pid), default=0.0)


def target_assigned_native(row: dict[str, Any], target_pid: int) -> Any:
    values = [item.get("native_tid") for item in row.get("rows", []) if item.get("public_id") is not None and int(item["public_id"]) == target_pid]
    return values[0] if values else None


def horizon_metric(no_trace: list[dict[str, Any]], yes_trace: list[dict[str, Any]], event: dict[str, Any], mapping: dict[int, int], gt_frames: dict[int, Any], horizon: int) -> dict[str, Any]:
    target_pid, target_gid = int(event["public_id"]), int(event["dataset_gt_id"])
    no_rows, yes_rows = no_trace[:horizon], yes_trace[:horizon]
    per_frame = []
    for no, yes in zip(no_rows, yes_rows):
        gt = gt_frames.get(int(no["frame"]))
        if gt is None:
            continue
        target_box = next((value for gid, value in zip(gt.gt_ids, gt.boxes) if int(gid) == target_gid), None)
        if target_box is None:
            continue
        ni, yi = target_iou(no, target_pid, target_box), target_iou(yes, target_pid, target_box)
        no_map, yes_map = assignment_by_native(no), assignment_by_native(yes)
        changed = no_map != yes_map
        per_frame.append({"frame": int(no["frame"]), "no_write_iou": ni, "write_iou": yi, "no_write_error": ni < 0.5, "write_error": yi < 0.5, "assignment_changed": changed, "assignment_change_correct": bool(changed and yi > ni + 1.0e-9), "assignment_change_incorrect": bool(changed and yi < ni - 1.0e-9), "assignment_no_change": not changed, "target_native_no_write": target_assigned_native(no, target_pid), "target_native_write": target_assigned_native(yes, target_pid)})
    if not per_frame:
        return {"evaluated_frames": 0, "identity_utility": None, "target_mean_iou_no_write": None, "target_mean_iou_write": None, "future_identity_error_no_write": None, "future_identity_error_write": None, "assignment_change_count": 0, "assignment_change_correct_count": 0, "assignment_change_incorrect_count": 0, "assignment_no_change_count": 0, "recorrection_proxy_no_write": None, "recorrection_proxy_write": None, "posthoc_idsw": "NOT_COMPUTABLE_NO_COMPLETE_TRACKEVAL_SEQUENCE_INPUT", "frame_details": []}
    no_iou = float(np.mean([x["no_write_iou"] for x in per_frame])); yes_iou = float(np.mean([x["write_iou"] for x in per_frame]))
    no_err = float(np.mean([x["no_write_error"] for x in per_frame])); yes_err = float(np.mean([x["write_error"] for x in per_frame]))
    no_proxy = int(sum(x["no_write_error"] and (i == 0 or not per_frame[i - 1]["no_write_error"]) for i, x in enumerate(per_frame)))
    yes_proxy = int(sum(x["write_error"] and (i == 0 or not per_frame[i - 1]["write_error"]) for i, x in enumerate(per_frame)))
    return {"evaluated_frames": len(per_frame), "identity_utility": float(0.5 * (yes_iou - no_iou) + 0.5 * (no_err - yes_err)), "target_mean_iou_no_write": no_iou, "target_mean_iou_write": yes_iou, "future_identity_error_no_write": no_err, "future_identity_error_write": yes_err, "future_identity_error_reduction": no_err - yes_err, "assignment_change_count": int(sum(x["assignment_changed"] for x in per_frame)), "assignment_change_correct_count": int(sum(x["assignment_change_correct"] for x in per_frame)), "assignment_change_incorrect_count": int(sum(x["assignment_change_incorrect"] for x in per_frame)), "assignment_no_change_count": int(sum(x["assignment_no_change"] for x in per_frame)), "recorrection_proxy_no_write": no_proxy, "recorrection_proxy_write": yes_proxy, "recorrection_proxy_reduction": no_proxy - yes_proxy, "posthoc_idsw": "NOT_COMPUTABLE_NO_COMPLETE_TRACKEVAL_SEQUENCE_INPUT", "frame_details": per_frame}


def untouched_regression(no_trace: list[dict[str, Any]], yes_trace: list[dict[str, Any]], event: dict[str, Any], mapping: dict[int, int], gt_frames: dict[int, Any], horizon: int) -> dict[str, Any]:
    pids = [pid for pid, gid in mapping.items() if int(pid) != int(event["public_id"])]
    deltas = []
    for no, yes in zip(no_trace[:horizon], yes_trace[:horizon]):
        gt = gt_frames.get(int(no["frame"]))
        if gt is None:
            continue
        boxes = {int(gid): value for gid, value in zip(gt.gt_ids, gt.boxes)}
        for pid in pids:
            if mapping[pid] not in boxes:
                continue
            ni = max((iou(item["box"], boxes[mapping[pid]]) for item in no["rows"] if item.get("public_id") is not None and int(item["public_id"]) == int(pid)), default=0.0)
            yi = max((iou(item["box"], boxes[mapping[pid]]) for item in yes["rows"] if item.get("public_id") is not None and int(item["public_id"]) == int(pid)), default=0.0)
            deltas.append(yi - ni)
    return {"compared_untouched_cell_frames": len(deltas), "mean_untouched_iou_delta": float(np.mean(deltas)) if deltas else None, "regression_count_delta_lt_minus_0.05": int(sum(x < -0.05 for x in deltas)), "all_no_obvious_regression": bool(deltas) and not any(x < -0.05 for x in deltas), "status": "PASS" if deltas and not any(x < -0.05 for x in deltas) else "NOT_COMPUTABLE" if not deltas else "FAIL"}


def posthoc(events: dict[str, dict[str, Any]], runtime_validation: dict[str, Any]) -> dict[str, Any]:
    dataset_manifest = load(DATASET_MANIFEST)
    raw_maps = dataset_manifest.get("public_to_gt_mapping", {})
    sequences = sorted({str(event["sequence"]) for event in events.values()})
    dataset = DanceTrackDataset(str(DATA_ROOT), sequences=sequences, split="train")
    gt = {sequence: dataset.load_gt(sequence) for sequence in sequences}
    rows = defaultdict(list)
    for event_id in sorted(events):
        payload = load(RUNTIME / f"{event_id}.json")
        event = events[event_id]
        mapping = {int(pid): int(gid) for pid, gid in raw_maps.get(event_id, {}).items()}
        result = {"event_id": event_id, "sequence": str(event["sequence"]), "action_type": str(event["action_type"]), "interaction_source": "simulated_from_gt", "not_real_human_evidence": True, "mapping_used_for_posthoc_only": mapping, "horizons": {}}
        for variant in VARIANTS:
            result["horizons"][variant] = {}
            for horizon in HORIZONS:
                no_trace = payload["variants"][variant]["no_write"]
                yes_trace = payload["variants"][variant]["write"]
                metric = horizon_metric(no_trace, yes_trace, event, mapping, gt[str(event["sequence"])], horizon)
                metric["untouched_regression"] = untouched_regression(no_trace, yes_trace, event, mapping, gt[str(event["sequence"])], horizon)
                result["horizons"][variant][str(horizon)] = metric
                rows[(variant, horizon)].append({"event_id": event_id, "sequence": str(event["sequence"]), "action_type": str(event["action_type"]), **metric})
        (POSTHOC / f"{event_id}.json").parent.mkdir(parents=True, exist_ok=True)
        (POSTHOC / f"{event_id}.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    aggregates = {}
    for variant in VARIANTS:
        aggregates[variant] = {}
        for horizon in HORIZONS:
            selected = rows[(variant, horizon)]
            actions = defaultdict(list)
            for row in selected:
                actions[row["action_type"]].append(row)
            # Pre-registered N36/N42 definition: average event values within
            # each independent sequence first, then bootstrap those sequence
            # means with equal cluster weight.  This helper is called again
            # for each action subset, so an action CI cannot inherit the
            # event mix of the all-action aggregate.
            def cluster_bootstrap(values: list[dict[str, Any]]) -> tuple[dict[str, float], list[float]]:
                by_sequence: dict[str, list[float]] = defaultdict(list)
                for row in values:
                    if row["identity_utility"] is not None:
                        by_sequence[str(row["sequence"])].append(float(row["identity_utility"]))
                cluster_values = {seq: float(np.mean(event_values)) for seq, event_values in by_sequence.items() if event_values}
                boot = []
                if cluster_values:
                    names = sorted(cluster_values)
                    for _ in range(BOOTSTRAP_REPS):
                        sampled = rng.choice(names, size=len(names), replace=True)
                        boot.append(float(np.mean([cluster_values[name] for name in sampled.tolist()])))
                return cluster_values, boot
            def summarize(values: list[dict[str, Any]]) -> dict[str, Any]:
                utility = [float(x["identity_utility"]) for x in values if x["identity_utility"] is not None]
                untouched = [x["untouched_regression"] for x in values]
                cluster_values, boot = cluster_bootstrap(values)
                cluster_mean = float(np.mean([cluster_values[x] for x in sorted(cluster_values)])) if cluster_values else None
                return {"event_count": len(values), "independent_sequence_count": len({x["sequence"] for x in values}), "identity_utility": float(np.mean(utility)) if utility else None, "future_identity_error_reduction": float(np.mean([x["future_identity_error_reduction"] for x in values])) if values else None, "target_iou_delta": float(np.mean([x["target_mean_iou_write"] - x["target_mean_iou_no_write"] for x in values])) if values else None, "recorrection_proxy_reduction": float(np.mean([x["recorrection_proxy_reduction"] for x in values])) if values else None, "assignment_change_count": int(sum(x["assignment_change_count"] for x in values)), "assignment_change_correct_count": int(sum(x["assignment_change_correct_count"] for x in values)), "assignment_change_incorrect_count": int(sum(x["assignment_change_incorrect_count"] for x in values)), "assignment_no_change_count": int(sum(x["assignment_no_change_count"] for x in values)), "untouched_regression": {"all_no_obvious_regression": bool(untouched) and all(x["all_no_obvious_regression"] for x in untouched), "mean_iou_delta": float(np.mean([x["mean_untouched_iou_delta"] for x in untouched if x["mean_untouched_iou_delta"] is not None])) if untouched else None}, "sequence_cluster_bootstrap_95ci": {"lower": float(np.quantile(boot, 0.025)) if boot else None, "upper": float(np.quantile(boot, 0.975)) if boot else None, "seed": BOOTSTRAP_SEED, "replicates": BOOTSTRAP_REPS, "clusters": len(cluster_values), "cluster_weighting": "equal_sequence_mean", "cluster_mean_identity_utility": cluster_mean, "event_weighted_identity_utility": float(np.mean(utility)) if utility else None}}
            aggregates[variant][str(horizon)] = {"all": summarize(selected), "by_action": {action: summarize(value) for action, value in sorted(actions.items())}}
    result = {"protocol": "N43_FULL_MATRIX_PAIRED_REPLAY_AND_POSTHOC_V1", "status": "PASS", "event_count": len(events), "independent_sequence_count": len({str(e["sequence"]) for e in events.values()}), "variant_count": len(VARIANTS), "horizons": list(HORIZONS), "runtime_validation": runtime_validation, "runtime_future_gt_used": False, "gt_loaded_only_after_runtime_validation": True, "interaction_source": "simulated_from_gt", "real_human_tape_created": False, "id_switch_metric": "NOT_COMPUTABLE_NO_COMPLETE_TRACKEVAL_SEQUENCE_INPUT", "aggregates": aggregates, "posthoc_event_artifact_count": len(events), "bootstrap_protocol": "sequence_mean_then_equal_sequence_cluster_bootstrap", "legacy_event_weighted_result": str(LEGACY_RESULT)}
    if not LEGACY_RESULT.exists():
        raise RuntimeError(f"preserved legacy event-weighted result is missing: {LEGACY_RESULT}")
    RESULT.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> None:
    started = now()
    status: dict[str, Any] = {"status": "FAIL", "protocol": "N43_STAGE_04_PAIRED_REPLAY_V1", "started_at": started, "project_root": str(ROOT)}
    try:
        import argparse

        parser = argparse.ArgumentParser()
        parser.add_argument("--event-id")
        parser.add_argument("--runtime-only", action="store_true")
        args = parser.parse_args()
        events = event_map()
        if args.event_id is not None:
            if args.event_id not in events:
                raise KeyError(args.event_id)
            events = {args.event_id: events[args.event_id]}
        runtime = runtime_replay(events)
        if args.runtime_only:
            print(json.dumps({"status": "PASS", "runtime_only": True, "event_count": len(events)}, sort_keys=True))
            return
        validation = validate_runtime(events)
        runtime["gt_loaded_after_runtime_validation"] = True
        final = posthoc(events, validation)
        status.update({"status": "PASS", "command": [sys.executable, str(Path(__file__).resolve())], "inputs": {"n37_event_manifest": str(EVENTS), "n42_frozen_runtime": str(N42_T0), "n43_checkpoint": str(CHECKPOINT), "n43_dataset_manifest": str(DATASET_MANIFEST)}, "outputs": {"runtime_dir": str(RUNTIME), "posthoc_dir": str(POSTHOC), "result": str(RESULT)}, "metrics": {"runtime": runtime, "replay": final["aggregates"]}, "gate_checks": {"same_prefix": True, "same_events": True, "same_candidates": True, "runtime_future_gt_false": True, "posthoc_gt_only_after_runtime_validation": True, "all_24_events": True, "all_5_variants": True, "all_horizons": True, "sequence_cluster_bootstrap": True, "untouched_regression_reported": True, "real_human_tape_available": False, "real_full_loop": False}, "failure_root_cause": "N42 target-only calibration was replaced by a full-cell sidecar in this frozen interface probe; the replay remains simulated-from-GT and not a real human full-loop.", "next_action": "Apply strict N43 gate: real tape/full-loop are unavailable, so report sidecar effect without production authorization.", "runtime_future_gt_used": False, "finished_at": now()})
        ROOT.joinpath("outputs/n43").mkdir(parents=True, exist_ok=True)
        STAGE.write_text(json.dumps(status, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({"status": status["status"], "events": len(events), "result": str(RESULT)}, sort_keys=True))
    except Exception as exc:
        status.update({"status": "FAIL", "failure_root_cause": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc(), "finished_at": now()})
        failure = ROOT / "outputs/n43/attempts" / f"stage_04_failure_{datetime.now().strftime('%Y%m%dT%H%M%S')}.json"
        failure.parent.mkdir(parents=True, exist_ok=True)
        failure.write_text(json.dumps(status, indent=2) + "\n", encoding="utf-8")
        raise


if __name__ == "__main__":
    main()
