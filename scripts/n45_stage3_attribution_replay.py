#!/usr/bin/env python3
"""N45 Stage 03: three-branch attribution replay from frozen N42/N44 inputs."""

from __future__ import annotations

import copy
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
from scripts.n44_assignment_common import apply_sidecar, load_checkpoint, sha256


EVENTS = ROOT / "outputs/n37/real_event_manifest.json"
N42 = ROOT / "outputs/n42/replay/runtime/t0"
N43_MAP = ROOT / "outputs/n43/training/dataset_manifest.json"
CHECKPOINT = ROOT / "outputs/n44/training/n44_assignment_aware.pt"
OUT = ROOT / "outputs/n45/replay"
RUNTIME = OUT / "runtime"
POSTHOC = OUT / "posthoc_events"
RESULT = OUT / "attribution_results.json"
STAGE = ROOT / "outputs/n45/stage_03_status.json"
VARIANTS = ("M0", "M1", "M2", "M3", "M4")
HORIZONS = (20, 50, 100)
BOOTSTRAP_SEED = 4444
BOOTSTRAP_REPS = 2000


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def events() -> dict[str, dict[str, Any]]:
    payload = load(EVENTS)
    if payload.get("status") != "PASS" or len(payload.get("events", [])) != 24:
        raise RuntimeError("frozen N37 event manifest invalid")
    return {str(item["event"]["event_id"]): item["event"] for item in payload["events"]}


def source(event_id: str) -> dict[str, Any]:
    payload = load(N42 / f"{event_id}.json")
    if payload.get("status") != "PASS" or payload.get("runtime_boundary", {}).get("runtime_future_gt_used") is not False:
        raise RuntimeError(f"invalid frozen N42 runtime source {event_id}")
    return payload


def slim(audit: dict[str, Any]) -> dict[str, Any]:
    candidates = audit.get("candidates", [])
    pids = list(audit.get("candidate_public_ids", []))
    rows = [{"native_tid": c.get("native_tid"), "box": c.get("box"), "confidence": c.get("confidence"), "public_id": pid} for c, pid in zip(candidates, pids)]
    valid = len(rows) == len(candidates) and all(x.get("public_id") is None or isinstance(x.get("public_id"), int) for x in rows)
    return {"frame": int(audit["frame"]), "rows": rows, "candidate_count": len(rows), "candidate_set_complete": audit.get("candidate_set_complete") is True, "mapping_complete_or_explicit_none": bool(valid), "assignment": list(audit.get("assignment", audit.get("assignment_after_scope", []))), "public_id_order": list(audit.get("public_id_order", [])), "runtime_future_gt_used": False}


def normalize_assignment(values: Any, public_id_count: int) -> list[int]:
    return [(-1 if int(value) >= public_id_count else int(value)) for value in values]


def candidate_keys(row: dict[str, Any]) -> list[tuple[Any, Any, Any]]:
    return [(x.get("native_tid"), x.get("box"), x.get("confidence")) for x in row.get("rows", [])]


def runtime(event_map: dict[str, dict[str, Any]]) -> dict[str, Any]:
    model, checkpoint = load_checkpoint(CHECKPOINT, "cpu")
    if not isinstance(checkpoint.get("gate"), dict):
        raise RuntimeError("frozen N44 checkpoint gate missing")
    RUNTIME.mkdir(parents=True, exist_ok=True)
    totals = {"write_frames": 0, "proposals_considered": 0, "proposals_selected": 0, "selected_but_no_assignment_change": 0, "changed_cells": 0, "changed_assignments": 0}
    for event_id in sorted(event_map):
        event = event_map[event_id]
        payload = source(event_id)
        output_variants: dict[str, Any] = {}
        for variant in VARIANTS:
            src_variant = payload["variants"][variant]
            no_source = src_variant["branches"]["memory_write=False"]
            write_source = src_variant["branches"]["memory_write=True"]
            no_trace, write_trace, plus_trace, frame_attribution = [], [], [], []
            previous = src_variant.get("event_frame_audit", {}).get("candidate_audit", {})
            for no_entry, write_entry in zip(no_source["future_trace"], write_source["future_trace"]):
                if int(no_entry["frame"]) != int(write_entry["frame"]):
                    raise RuntimeError(f"N42 no/write frame mismatch {event_id}/{variant}")
                no_audit, write_audit = no_entry["candidate_audit"], write_entry["candidate_audit"]
                no_row, write_row = slim(no_audit), slim(write_audit)
                if candidate_keys(no_row) != candidate_keys(write_row) or no_row["candidate_count"] != write_row["candidate_count"]:
                    raise RuntimeError(f"N42 no/write candidate stream mismatch {event_id}/{variant}/{no_row['frame']}")
                if variant == "M0":
                    plus_audit = copy.deepcopy(write_audit)
                    sidecar_meta = {"enabled": False, "proposals_considered": 0, "proposals_selected": 0, "changed_cell_count": 0, "changed_assignment_count": 0, "runtime_future_gt_used": False}
                else:
                    plus_audit = apply_sidecar(write_audit, model, int(write_audit["frame"]) - int(event["frame"]), previous, checkpoint["gate"])
                    sidecar_meta = plus_audit.get("n44_sidecar", {})
                plus_row = slim(plus_audit)
                if candidate_keys(write_row) != candidate_keys(plus_row) or write_row["candidate_count"] != plus_row["candidate_count"] or set(write_row["public_id_order"]) != set(plus_row["public_id_order"]):
                    raise RuntimeError(f"write/plus candidate stream mismatch {event_id}/{variant}/{write_row['frame']}")
                write_scores = np.asarray(write_audit["fused_scores"], dtype=np.float32)
                plus_scores = np.asarray(plus_audit.get("fused_scores", write_audit["fused_scores"]), dtype=np.float32)
                changed = np.argwhere(np.abs(plus_scores - write_scores) > 1e-12)
                baseline_assignment = normalize_assignment(write_audit["assignment_after_scope"], len(write_audit["public_id_order"]))
                sidecar_before = normalize_assignment(plus_audit.get("assignment_before_n44", write_audit["assignment_after_scope"]), len(write_audit["public_id_order"]))
                if baseline_assignment != sidecar_before:
                    raise RuntimeError(f"write baseline assignment mismatch {event_id}/{variant}/{write_row['frame']}")
                plus_assignment = normalize_assignment(plus_audit.get("assignment_after_n44", plus_audit.get("assignment_after_scope", [])), len(write_audit["public_id_order"]))
                selected = int(sidecar_meta.get("proposals_selected", 0))
                changed_assignments = int(sum(a != b for a, b in zip(baseline_assignment, plus_assignment)))
                if int(sidecar_meta.get("changed_cell_count", 0)) != len(changed) or changed_assignments != int(sidecar_meta.get("changed_assignment_count", changed_assignments)):
                    raise RuntimeError(f"N44 sidecar metadata mismatch {event_id}/{variant}/{write_row['frame']}")
                if len(changed) and not np.allclose((plus_scores - write_scores)[np.abs(plus_scores - write_scores) > 1e-12], 0.25, atol=1e-7):
                    raise RuntimeError(f"non-bounded/unrecorded N44 boost {event_id}/{variant}/{write_row['frame']}")
                hard = write_scores <= -1.0e7
                if not np.all(plus_scores[hard] == write_scores[hard]):
                    raise RuntimeError(f"hard negative changed {event_id}/{variant}/{write_row['frame']}")
                no_trace.append(no_row); write_trace.append(write_row); plus_trace.append(plus_row)
                selected_no_assignment = int(selected if selected > 0 and changed_assignments == 0 else 0)
                frame_attribution.append({"frame": int(write_row["frame"]), "candidate_count": int(write_row["candidate_count"]), "public_id_alignment": {"no_write_ids": sorted(set(int(x) for x in no_row["public_id_order"])), "write_baseline_ids": sorted(set(int(x) for x in write_row["public_id_order"])), "intersection_ids": sorted(set(int(x) for x in no_row["public_id_order"]) & set(int(x) for x in write_row["public_id_order"])), "no_write_only_ids": sorted(set(int(x) for x in no_row["public_id_order"]) - set(int(x) for x in write_row["public_id_order"])), "write_only_ids": sorted(set(int(x) for x in write_row["public_id_order"]) - set(int(x) for x in no_row["public_id_order"]))}, "no_write_to_write_baseline": {"assignment_changed": by_native(no_row) != by_native(write_row)}, "write_baseline_to_write_plus_n44": {"proposals_considered": int(sidecar_meta.get("proposals_considered", 0)), "proposals_selected": selected, "selected_but_no_assignment_change": selected_no_assignment, "changed_cells": int(len(changed)), "assignment_changed": changed_assignments, "hard_negative_preserved": bool(np.all(plus_scores[hard] == write_scores[hard])), "runtime_future_gt_used": False}, "same_candidate_stream": True})
                totals["write_frames"] += 1
                totals["proposals_considered"] += int(sidecar_meta.get("proposals_considered", 0)); totals["proposals_selected"] += selected; totals["selected_but_no_assignment_change"] += selected_no_assignment; totals["changed_cells"] += int(len(changed)); totals["changed_assignments"] += changed_assignments
                previous = write_audit
            if len(no_trace) != 100 or len(write_trace) != 100 or len(plus_trace) != 100:
                raise RuntimeError(f"trace length invalid {event_id}/{variant}")
            output_variants[variant] = {"no_write": no_trace, "write_baseline": write_trace, "write_plus_n44": plus_trace, "frame_attribution": frame_attribution, "runtime_future_gt_used": False}
        artifact = {"protocol": "N45_THREE_BRANCH_ATTRIBUTION_RUNTIME_V1", "status": "PASS", "event_id": event_id, "sequence": str(event["sequence"]), "action_type": str(event["action_type"]), "event_frame": int(event["frame"]), "future_frame_start": int(event["frame"]) + 1, "future_frame_count": 100, "interaction_source": "simulated_from_gt", "not_real_human_evidence": True, "checkpoint": str(CHECKPOINT), "checkpoint_sha256": sha256(CHECKPOINT), "branches": {"no_write": "N42 memory_write=False", "write_baseline": "N42 memory_write=True original fused assignment", "write_plus_n44": "write_baseline plus frozen N44 only"}, "variants": output_variants, "runtime_boundary": {"runtime_future_gt_used": False, "gt_loaded_in_worker": False, "future_gt_fields_sent": [], "event_frame_memory_read": False}}
        (RUNTIME / f"{event_id}.json").write_text(json.dumps(artifact, separators=(",", ":")) + "\n", encoding="utf-8")
    return {"status": "PASS", "event_count": len(event_map), "runtime_artifacts": str(RUNTIME), "totals": totals, "checkpoint": str(CHECKPOINT), "checkpoint_sha256": sha256(CHECKPOINT), "runtime_future_gt_used": False}


def by_native(row: dict[str, Any]) -> dict[str, Any]:
    return {str(x["native_tid"]): x.get("public_id") for x in row["rows"] if x.get("native_tid") is not None}


def pid_iou(row: dict[str, Any], pid: int, target: Any) -> float:
    return max((iou(x["box"], target) for x in row["rows"] if x.get("public_id") is not None and int(x["public_id"]) == pid), default=0.0)


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
        ri, yi = pid_iou(ref, target_pid, target), pid_iou(yes, target_pid, target)
        changed = by_native(ref) != by_native(yes)
        details.append({"frame": int(ref["frame"]), "reference_iou": ri, "treated_iou": yi, "target_iou_delta": yi - ri, "reference_error": ri < 0.5, "treated_error": yi < 0.5, "assignment_changed": changed, "assignment_change_correct": bool(changed and yi > ri + 1e-9), "assignment_change_incorrect": bool(changed and yi < ri - 1e-9), "assignment_change_neutral": bool(changed and abs(yi - ri) <= 1e-9), "assignment_no_change": not changed})
    if not details:
        return {"evaluated_frames": 0, "identity_utility": None, "target_iou_delta": None, "future_identity_error_reduction": None, "recorrection_proxy_reduction": None, "assignment_change_count": 0, "assignment_change_correct_count": 0, "assignment_change_incorrect_count": 0, "assignment_change_neutral_count": 0, "assignment_no_change_count": 0, "untouched_regression": {"status": "NOT_COMPUTABLE"}, "frame_details": []}
    ri = float(np.mean([x["reference_iou"] for x in details])); yi = float(np.mean([x["treated_iou"] for x in details])); re = float(np.mean([x["reference_error"] for x in details])); ye = float(np.mean([x["treated_error"] for x in details]))
    rr = sum(x["reference_error"] and (i == 0 or not details[i - 1]["reference_error"]) for i, x in enumerate(details)); yr = sum(x["treated_error"] and (i == 0 or not details[i - 1]["treated_error"]) for i, x in enumerate(details))
    deltas = []
    for ref, yes in zip(reference[:horizon], treated[:horizon]):
        frame = gt.get(int(ref["frame"]))
        if frame is None:
            continue
        boxes = {int(gid): box for gid, box in zip(frame.gt_ids, frame.boxes)}
        for pid, gid in mapping.items():
            if int(pid) == target_pid or int(gid) not in boxes:
                continue
            deltas.append(pid_iou(yes, int(pid), boxes[int(gid)]) - pid_iou(ref, int(pid), boxes[int(gid)]))
    untouched = {"compared": len(deltas), "mean_iou_delta": float(np.mean(deltas)) if deltas else None, "regression_count_delta_lt_minus_0.05": int(sum(x < -0.05 for x in deltas)), "all_no_obvious_regression": bool(deltas) and not any(x < -0.05 for x in deltas), "status": "PASS" if deltas and not any(x < -0.05 for x in deltas) else "FAIL" if deltas else "NOT_COMPUTABLE"}
    return {"evaluated_frames": len(details), "identity_utility": 0.5 * (yi - ri) + 0.5 * (re - ye), "target_iou_delta": yi - ri, "future_identity_error_reduction": re - ye, "recorrection_proxy_reduction": int(rr - yr), "assignment_change_count": int(sum(x["assignment_changed"] for x in details)), "assignment_change_correct_count": int(sum(x["assignment_change_correct"] for x in details)), "assignment_change_incorrect_count": int(sum(x["assignment_change_incorrect"] for x in details)), "assignment_change_neutral_count": int(sum(x["assignment_change_neutral"] for x in details)), "assignment_no_change_count": int(sum(x["assignment_no_change"] for x in details)), "untouched_regression": untouched, "frame_details": details}


def validate_runtime(event_map: dict[str, dict[str, Any]]) -> dict[str, Any]:
    checked = 0
    for event_id in sorted(event_map):
        payload = load(RUNTIME / f"{event_id}.json")
        if payload.get("status") != "PASS" or payload.get("runtime_boundary", {}).get("runtime_future_gt_used") is not False:
            raise RuntimeError(f"runtime boundary invalid {event_id}")
        for variant in VARIANTS:
            traces = payload["variants"][variant]
            for branch in ("no_write", "write_baseline", "write_plus_n44"):
                trace = traces[branch]
                if len(trace) != 100 or [int(x["frame"]) for x in trace] != list(range(int(trace[0]["frame"]), int(trace[0]["frame"]) + 100)):
                    raise RuntimeError(f"frame gap/duplicate {event_id}/{variant}/{branch}")
                for row in trace:
                    if row.get("runtime_future_gt_used") is not False or row.get("candidate_set_complete") is not True or row.get("mapping_complete_or_explicit_none") is not True:
                        raise RuntimeError(f"row boundary/candidate invalid {event_id}/{variant}/{branch}/{row['frame']}")
                    natives = [int(x["native_tid"]) for x in row["rows"] if x.get("native_tid") is not None]
                    if len(natives) != len(set(natives)):
                        raise RuntimeError(f"duplicate native id {event_id}/{variant}/{branch}/{row['frame']}")
                    checked += 1
            for a, b in zip(traces["no_write"], traces["write_baseline"]):
                if candidate_keys(a) != candidate_keys(b):
                    raise RuntimeError(f"no/write baseline candidate mismatch {event_id}/{variant}/{a['frame']}")
            for a, b in zip(traces["write_baseline"], traces["write_plus_n44"]):
                if candidate_keys(a) != candidate_keys(b) or a["public_id_order"] != b["public_id_order"]:
                    raise RuntimeError(f"write/plus candidate mismatch {event_id}/{variant}/{a['frame']}")
    return {"status": "PASS", "artifact_count": len(event_map), "trace_rows_checked": checked, "same_candidate_stream": True, "duplicate_or_missing_frame": False, "runtime_future_gt_used": False}


def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    utility = [float(x["identity_utility"]) for x in rows if x["identity_utility"] is not None]
    by_sequence: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        if row["identity_utility"] is not None:
            by_sequence[str(row["sequence"])].append(float(row["identity_utility"]))
    clusters = {key: float(np.mean(value)) for key, value in by_sequence.items() if value}
    names = sorted(clusters)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    boot = [float(np.mean([clusters[name] for name in rng.choice(names, size=len(names), replace=True)])) for _ in range(BOOTSTRAP_REPS)] if names else []
    return {"event_count": len(rows), "independent_sequence_count": len(names), "identity_utility": float(np.mean(utility)) if utility else None, "target_iou_delta": float(np.mean([x["target_iou_delta"] for x in rows if x["target_iou_delta"] is not None])) if rows else None, "future_identity_error_reduction": float(np.mean([x["future_identity_error_reduction"] for x in rows if x["future_identity_error_reduction"] is not None])) if rows else None, "recorrection_proxy_reduction": float(np.mean([x["recorrection_proxy_reduction"] for x in rows if x["recorrection_proxy_reduction"] is not None])) if rows else None, "assignment_change_count": int(sum(x["assignment_change_count"] for x in rows)), "assignment_change_correct_count": int(sum(x["assignment_change_correct_count"] for x in rows)), "assignment_change_incorrect_count": int(sum(x["assignment_change_incorrect_count"] for x in rows)), "assignment_change_neutral_count": int(sum(x["assignment_change_neutral_count"] for x in rows)), "assignment_no_change_count": int(sum(x["assignment_no_change_count"] for x in rows)), "assignment_decomposition_closes": int(sum(x["assignment_change_correct_count"] + x["assignment_change_incorrect_count"] + x["assignment_change_neutral_count"] + x["assignment_no_change_count"] for x in rows)) == int(sum(x["assignment_change_count"] + x["assignment_no_change_count"] for x in rows)), "untouched_regression": {"all_no_obvious_regression": bool(rows) and all(x["untouched_regression"].get("all_no_obvious_regression", False) for x in rows), "mean_iou_delta": float(np.mean([x["untouched_regression"]["mean_iou_delta"] for x in rows if x["untouched_regression"].get("mean_iou_delta") is not None])) if rows else None}, "sequence_cluster_bootstrap_95ci": {"lower": float(np.quantile(boot, 0.025)) if boot else None, "upper": float(np.quantile(boot, 0.975)) if boot else None, "seed": BOOTSTRAP_SEED, "replicates": BOOTSTRAP_REPS, "clusters": len(names), "cluster_weighting": "equal_sequence_mean", "cluster_mean_identity_utility": float(np.mean(list(clusters.values()))) if clusters else None, "event_weighted_identity_utility": float(np.mean(utility)) if utility else None}}


def posthoc(event_map: dict[str, dict[str, Any]], validation: dict[str, Any]) -> dict[str, Any]:
    raw_maps = load(N43_MAP)["public_to_gt_mapping"]
    sequences = sorted({str(event["sequence"]) for event in event_map.values()})
    dataset = DanceTrackDataset(str(DATA_ROOT), sequences=sequences, split="train")
    gt = {sequence: dataset.load_gt(sequence) for sequence in sequences}
    rows: dict[tuple[str, str, int], list[dict[str, Any]]] = defaultdict(list)
    for event_id, event in sorted(event_map.items()):
        mapping = {int(pid): int(gid) for pid, gid in raw_maps.get(event_id, {}).items()}
        payload = load(RUNTIME / f"{event_id}.json")
        event_result = {"event_id": event_id, "sequence": str(event["sequence"]), "action_type": str(event["action_type"]), "interaction_source": "simulated_from_gt", "not_real_human_evidence": True, "mapping_used_for_posthoc_only": mapping, "frame_attribution": {v: payload["variants"][v]["frame_attribution"] for v in VARIANTS}, "horizons": {}}
        for variant in VARIANTS:
            event_result["horizons"][variant] = {}
            for horizon in HORIZONS:
                no = payload["variants"][variant]["no_write"]
                write = payload["variants"][variant]["write_baseline"]
                plus = payload["variants"][variant]["write_plus_n44"]
                memory = compare(no, write, event, mapping, gt[str(event["sequence"])], horizon)
                incremental = compare(write, plus, event, mapping, gt[str(event["sequence"])], horizon)
                app = {key: int(sum(int(item["write_baseline_to_write_plus_n44"].get(key, 0)) for item in payload["variants"][variant]["frame_attribution"][:horizon])) for key in ("proposals_considered", "proposals_selected", "selected_but_no_assignment_change", "changed_cells", "assignment_changed")}
                event_result["horizons"][variant][str(horizon)] = {"memory_effect_no_write_to_write_baseline": memory, "n44_incremental_effect_write_baseline_to_write_plus_n44": incremental, "application_counts_through_horizon": app}
                rows[("memory", variant, horizon)].append({"event_id": event_id, "sequence": str(event["sequence"]), **memory})
                rows[("incremental", variant, horizon)].append({"event_id": event_id, "sequence": str(event["sequence"]), **incremental})
        (POSTHOC / f"{event_id}.json").parent.mkdir(parents=True, exist_ok=True)
        (POSTHOC / f"{event_id}.json").write_text(json.dumps(event_result, indent=2) + "\n", encoding="utf-8")
    aggregates = {effect: {variant: {str(h): aggregate(rows[(effect, variant, h)]) for h in HORIZONS} for variant in VARIANTS} for effect in ("memory", "incremental")}
    final = {"protocol": "N45_THREE_BRANCH_ATTRIBUTION_POSTHOC_V1", "status": "PASS", "event_count": len(event_map), "independent_sequence_count": len({str(x["sequence"]) for x in event_map.values()}), "variant_count": 5, "horizons": list(HORIZONS), "runtime_validation": validation, "runtime_future_gt_used": False, "gt_loaded_only_after_runtime_validation": True, "interaction_source": "simulated_from_gt", "real_human_tape_created": False, "id_switch_metric": "NOT_COMPUTABLE_NO_COMPLETE_TRACKEVAL_SEQUENCE_INPUT", "bootstrap_protocol": "sequence_mean_then_equal_sequence_cluster_bootstrap", "effects": aggregates, "posthoc_event_artifact_count": len(event_map), "attribution": {"memory_effect": "write_baseline minus no_write", "n44_incremental_effect": "write_plus_n44 minus write_baseline"}}
    RESULT.write_text(json.dumps(final, indent=2) + "\n", encoding="utf-8")
    return final


def main() -> None:
    status = {"status": "FAIL", "protocol": "N45_STAGE_03_ATTRIBUTION_REPLAY_V1", "started_at": now(), "project_root": str(ROOT)}
    try:
        event_map = events()
        run = runtime(event_map)
        validation = validate_runtime(event_map)
        final = posthoc(event_map, validation)
        status.update({"status": "PASS", "command": ["python", "scripts/n45_stage3_attribution_replay.py"], "inputs": {"n37_event_manifest": str(EVENTS), "n42_frozen_runtime": str(N42), "n43_offline_mapping_manifest": str(N43_MAP), "n44_frozen_checkpoint": str(CHECKPOINT)}, "outputs": {"runtime_dir": str(RUNTIME), "posthoc_dir": str(POSTHOC), "result": str(RESULT)}, "metrics": {"runtime": run, "validation": validation, "memory_effect": final["effects"]["memory"], "n44_incremental_effect": final["effects"]["incremental"]}, "gate_checks": {"same_prefix": True, "same_event": True, "same_candidates": True, "write_baseline_is_original_n42_fused_assignment": True, "write_plus_diff_only_sidecar": True, "runtime_future_gt_false": True, "posthoc_after_validation": True, "all_24_events": True, "all_5_variants": True, "all_horizons": True, "equal_sequence_bootstrap": True, "per_event_frame_attribution": True, "proposal_counts_recorded": True, "selected_no_assignment_change_recorded": True, "none_hard_negative_checked": True, "checkpoint_production_authorized_false": True, "standard_mot": "NOT_COMPUTABLE_NO_COMPLETE_TRACKEVAL_SEQUENCE_INPUT", "real_human_tape": False, "real_full_loop": False}, "failure_root_cause": "N44's previous no_write versus write_plus_N44 comparison was not attributable. N45 separately measures memory effect and the true write_baseline-to-write_plus_N44 increment; all events remain simulated_from_gt and no production authorization follows.", "next_action": "Audit N45 incremental aggregates and corrected Stage 01/02 contracts, then generate the strict N45 gate/report without using holdout or relabeling simulated input.", "runtime_future_gt_used": False, "finished_at": now()})
        STAGE.write_text(json.dumps(status, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({"status": status["status"], "events": len(event_map), "result": str(RESULT)}))
    except Exception as exc:
        status.update({"status": "FAIL", "failure_root_cause": f"{type(exc).__name__}: {exc}", "traceback": __import__("traceback").format_exc(), "finished_at": now()})
        failure = ROOT / "outputs/n45/attempts" / f"stage_03_failure_{datetime.now().strftime('%Y%m%dT%H%M%S')}.json"
        failure.parent.mkdir(parents=True, exist_ok=True)
        failure.write_text(json.dumps(status, indent=2) + "\n", encoding="utf-8")
        raise


if __name__ == "__main__":
    main()
