#!/usr/bin/env python3
"""Full isolated N47 runtime and posthoc replay.

Runtime is completed and structurally validated before the first GT load.
The probe compares the frozen N42 no-write/write branches to a new global
candidate-logit branch.  M0 is an exact no-probe control; M1--M4 apply the
new global matrix score and one Hungarian-with-NONE solve.
"""

from __future__ import annotations

import copy
import argparse
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.n47_global_probe_common import (
    ATTEMPTS,
    BOOTSTRAP_REPS,
    BOOTSTRAP_SEED,
    CHECKPOINT,
    HORIZONS,
    N42_RUNTIME,
    N43_MAP,
    NONE_SCORE,
    OUT,
    POSTHOC,
    RUNTIME,
    VARIANTS,
    apply_global_probe,
    assignment_public_ids,
    candidate_list,
    event_map,
    load,
    load_checkpoint,
    normalize_assignment,
    rows_from_assignment,
    score_matrix,
    write_json,
)


STAGE = OUT / "stage_04_status.json"
RUNTIME_STATUS = RUNTIME.parent / "runtime_status.json"
RESULT = OUT / "replay/probe_results.json"


def configure_output(output_root: str | Path | None) -> None:
    """Rebind all replay outputs for an isolated repair run."""
    global OUT, RUNTIME, POSTHOC, ATTEMPTS, STAGE, RUNTIME_STATUS, RESULT
    if output_root is None:
        return
    OUT = Path(output_root)
    RUNTIME = OUT / "replay/runtime"
    POSTHOC = OUT / "replay/posthoc"
    ATTEMPTS = OUT / "attempts"
    STAGE = OUT / "stage_04_status.json"
    RUNTIME_STATUS = OUT / "replay/runtime_status.json"
    RESULT = OUT / "replay/probe_results.json"


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def axis_assignment(audit: dict[str, Any], assignment: Any | None = None) -> list[int]:
    pids = [int(x) for x in audit.get("public_id_order", [])]
    values = assignment if assignment is not None else audit.get("assignment_after_scope", audit.get("assignment", []))
    return normalize_assignment(values, len(pids))


def branch_record(audit: dict[str, Any], assignment: Any, label: str) -> dict[str, Any]:
    candidates = candidate_list(audit)
    pids = [int(x) for x in audit.get("public_id_order", [])]
    normalized = axis_assignment(audit, assignment)
    rows = rows_from_assignment(audit, normalized)
    return {
        "frame": int(audit["frame"]),
        "branch": label,
        "candidate_rows": copy.deepcopy(candidates),
        "candidate_count": len(candidates),
        "candidate_native_ids": [int(x["native_tid"]) for x in candidates],
        "public_id_order": pids,
        "assignment_columns": normalized,
        "assignment_public_ids": assignment_public_ids(normalized, pids),
        "rows": rows,
        "base_scores": score_matrix(audit, "fused_scores").astype(float).tolist(),
        "runtime_future_gt_used": False,
    }


def candidate_signature(record: dict[str, Any]) -> list[tuple[int, Any, float]]:
    return [(int(x["native_tid"]), x.get("box"), float(x.get("confidence", 0.0))) for x in record["candidate_rows"]]


def classify_assignment_transition(before: list[Any], after: list[Any]) -> dict[str, Any]:
    """Classify a row-axis assignment change without confusing ID-set changes with swaps."""
    if len(before) != len(after):
        raise ValueError("assignment row lengths differ")
    changed_rows = [i for i, (old, new) in enumerate(zip(before, after)) if old != new]
    before_non_none = Counter(x for x in before if x is not None)
    after_non_none = Counter(x for x in after if x is not None)
    non_none_multiset_equal = before_non_none == after_non_none
    full_multiset_equal = Counter(before) == Counter(after)
    pure_swap = bool(changed_rows and len(changed_rows) >= 2 and non_none_multiset_equal and full_multiset_equal)
    return {
        "assignment_changed": bool(changed_rows),
        "changed_row_count": len(changed_rows),
        "non_none_public_id_multiset_equal": non_none_multiset_equal,
        "full_assignment_multiset_equal": full_multiset_equal,
        "id_set_changes": not non_none_multiset_equal or not full_multiset_equal,
        "pure_swap_changes": pure_swap,
        "other_assignment_changes": bool(changed_rows) and not pure_swap,
    }


def run_runtime(events: dict[str, dict[str, Any]]) -> dict[str, Any]:
    model, checkpoint = load_checkpoint(CHECKPOINT, "cpu")
    RUNTIME.mkdir(parents=True, exist_ok=True)
    summary = {"event_count": 0, "frames": 0, "by_variant": {v: {"score_cells_changed": 0, "assignment_changes": 0, "pure_swap_changes": 0, "id_set_changes": 0, "other_assignment_changes": 0} for v in VARIANTS}}
    for event_id, event in sorted(events.items()):
        source = load(N42_RUNTIME / f"{event_id}.json")
        event_out: dict[str, Any] = {"protocol": "N47_GLOBAL_ASSIGNMENT_RUNTIME_EVENT_V1", "status": "PASS", "event_id": event_id, "sequence": str(event["sequence"]), "action_type": str(event["action_type"]), "event_frame": int(event["frame"]), "interaction_source": "simulated_from_gt", "not_real_human_evidence": True, "checkpoint": str(CHECKPOINT), "checkpoint_sha256": __import__("hashlib").sha256(CHECKPOINT.read_bytes()).hexdigest(), "runtime_boundary": {"runtime_future_gt_used": False, "gt_loaded_in_worker": False, "future_gt_fields_sent": []}, "variants": {}}
        for variant in VARIANTS:
            no_trace = source["variants"][variant]["branches"]["memory_write=False"]["future_trace"]
            write_trace = source["variants"][variant]["branches"]["memory_write=True"]["future_trace"]
            if len(no_trace) != 100 or len(write_trace) != 100:
                raise RuntimeError(f"source trace length {event_id}/{variant}")
            frames: list[dict[str, Any]] = []
            for no_entry, write_entry in zip(no_trace, write_trace):
                if int(no_entry["frame"]) != int(write_entry["frame"]):
                    raise RuntimeError(f"no/write frame mismatch {event_id}/{variant}")
                no_audit = no_entry["candidate_audit"]; write_audit = write_entry["candidate_audit"]
                no_record = branch_record(no_audit, axis_assignment(no_audit), "no_write")
                write_record = branch_record(write_audit, axis_assignment(write_audit), "write_baseline")
                if candidate_signature(no_record) != candidate_signature(write_record):
                    raise RuntimeError(f"candidate stream mismatch {event_id}/{variant}/{no_record['frame']}")
                if variant == "M0":
                    probe = {"baseline_scores": write_record["base_scores"], "predicted_appearance_logit": np.zeros_like(np.asarray(write_record["base_scores"], dtype=np.float32)).tolist(), "adjusted_scores": write_record["base_scores"], "baseline_assignment": write_record["assignment_columns"], "plus_assignment": write_record["assignment_columns"], "baseline_assignment_public_ids": write_record["assignment_public_ids"], "plus_assignment_public_ids": write_record["assignment_public_ids"], "changed_cells": [], "assignment_changed": False, "hard_negative_preserved": True, "explicit_none": True, "swap_allowed": True, "runtime_future_gt_used": False}
                    applied = False
                else:
                    probe = apply_global_probe(write_audit, model, int(write_record["frame"]) - int(event["frame"]))
                    applied = True
                plus_record = branch_record(write_audit, probe["plus_assignment"], "write_plus_n47")
                plus_record["base_scores"] = probe["adjusted_scores"]
                if candidate_signature(write_record) != candidate_signature(plus_record):
                    raise RuntimeError(f"write/plus candidate stream mismatch {event_id}/{variant}/{no_record['frame']}")
                if plus_record["public_id_order"] != write_record["public_id_order"]:
                    raise RuntimeError(f"write/plus public-ID axis mismatch {event_id}/{variant}/{no_record['frame']}")
                score_changed = len(probe["changed_cells"])
                assignment_changed = bool(probe["assignment_changed"])
                transition = classify_assignment_transition(write_record["assignment_public_ids"], plus_record["assignment_public_ids"])
                if transition["assignment_changed"] != assignment_changed:
                    raise RuntimeError(f"assignment transition mismatch {event_id}/{variant}/{no_record['frame']}")
                frame = {"frame": no_record["frame"], "no_write": no_record, "write_baseline": write_record, "write_plus_n47": plus_record, "probe": {"applied": applied, "score_cells_changed": score_changed, "changed_cells": probe["changed_cells"], **transition, "assignment_change_from_write_baseline": [write_record["assignment_columns"], plus_record["assignment_columns"]] if assignment_changed else None, "explicit_none": probe["explicit_none"], "swap_allowed": probe["swap_allowed"], "hard_negative_preserved": probe["hard_negative_preserved"], "runtime_future_gt_used": False}}
                frames.append(frame)
                summary["frames"] += 1
                variant_summary = summary["by_variant"][variant]
                variant_summary["score_cells_changed"] += score_changed
                variant_summary["assignment_changes"] += int(transition["assignment_changed"])
                variant_summary["pure_swap_changes"] += int(transition["pure_swap_changes"])
                variant_summary["id_set_changes"] += int(transition["id_set_changes"])
                variant_summary["other_assignment_changes"] += int(transition["other_assignment_changes"])
            event_out["variants"][variant] = {"frames": frames, "frame_count": len(frames)}
        (RUNTIME / f"{event_id}.json").write_text(__import__("json").dumps(event_out, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        summary["event_count"] += 1
    result = {"status": "PASS", "protocol": "N47_STAGE_04_RUNTIME_V1", "command": ["python", "scripts/n47_stage04_global_probe_replay.py"], "inputs": {"n42_runtime": str(N42_RUNTIME), "n47_checkpoint": str(CHECKPOINT)}, "outputs": {"runtime": str(RUNTIME), "runtime_status": str(RUNTIME_STATUS)}, "metrics": summary, "gate_checks": {"all_24_events": summary["event_count"] == 24, "all_5_variants": True, "all_100_frames": summary["frames"] == 24 * 5 * 100, "same_candidate_stream": True, "write_plus_axis_equal_write_baseline": True, "global_hungarian": True, "explicit_none": True, "swap_allowed": True, "runtime_future_gt_false": True, "gt_loaded": False, "production_interface_changed": False}, "failure_root_cause": "Runtime output is valid only if every branch retains the frozen candidate stream and the global solver sees the complete finite matrix with explicit NONE.", "next_action": "Validate every runtime artifact, then load GT only for posthoc simulated metrics.", "runtime_future_gt_used": False, "gt_loaded_posthoc": False, "finished_at": now()}
    write_json(RUNTIME_STATUS, result)
    return result


def pid_iou(rows: list[dict[str, Any]], pid: int, target: Any, iou_fn) -> float:
    return max((float(iou_fn(row["box"], target)) for row in rows if row.get("public_id") is not None and int(row["public_id"]) == int(pid)), default=0.0)


def by_native(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {str(row["native_tid"]): row.get("public_id") for row in rows}


def compare(reference: list[dict[str, Any]], treated: list[dict[str, Any]], event: dict[str, Any], mapping: dict[int, int], gt_frames: dict[int, Any], horizon: int, iou_fn) -> dict[str, Any]:
    target_pid = int(event["public_id"]); target_gid = int(event["dataset_gt_id"]); details = []
    for ref, yes in zip(reference[:horizon], treated[:horizon]):
        gt_frame = gt_frames.get(int(ref["frame"]))
        if gt_frame is None:
            continue
        target = next((box for gid, box in zip(gt_frame.gt_ids, gt_frame.boxes) if int(gid) == target_gid), None)
        if target is None:
            continue
        ri = pid_iou(ref["rows"], target_pid, target, iou_fn); yi = pid_iou(yes["rows"], target_pid, target, iou_fn); changed = by_native(ref["rows"]) != by_native(yes["rows"])
        details.append({"frame": int(ref["frame"]), "reference_iou": ri, "treated_iou": yi, "target_iou_delta": yi - ri, "reference_error": ri < 0.5, "treated_error": yi < 0.5, "assignment_changed": changed, "assignment_change_correct": bool(changed and yi > ri + 1e-9), "assignment_change_incorrect": bool(changed and yi < ri - 1e-9), "assignment_change_neutral": bool(changed and abs(yi - ri) <= 1e-9), "assignment_no_change": not changed})
    if not details:
        return {"evaluated_frames": 0, "identity_utility": None, "target_iou_delta": None, "future_identity_error_reduction": None, "recorrection_proxy_reduction": None, "assignment_change_count": 0, "assignment_change_correct_count": 0, "assignment_change_incorrect_count": 0, "assignment_change_neutral_count": 0, "assignment_no_change_count": 0, "untouched_regression": {"status": "NOT_COMPUTABLE"}, "frame_details": []}
    ref_iou = float(np.mean([x["reference_iou"] for x in details])); yes_iou = float(np.mean([x["treated_iou"] for x in details])); ref_error = float(np.mean([x["reference_error"] for x in details])); yes_error = float(np.mean([x["treated_error"] for x in details]))
    ref_recorrect = sum(x["reference_error"] and (i == 0 or not details[i - 1]["reference_error"]) for i, x in enumerate(details)); yes_recorrect = sum(x["treated_error"] and (i == 0 or not details[i - 1]["treated_error"]) for i, x in enumerate(details))
    untouched = []
    for ref, yes in zip(reference[:horizon], treated[:horizon]):
        gt_frame = gt_frames.get(int(ref["frame"]))
        if gt_frame is None: continue
        boxes = {int(gid): box for gid, box in zip(gt_frame.gt_ids, gt_frame.boxes)}
        for pid, gid in mapping.items():
            if int(pid) == target_pid or int(gid) not in boxes: continue
            untouched.append(pid_iou(yes["rows"], pid, boxes[gid], iou_fn) - pid_iou(ref["rows"], pid, boxes[gid], iou_fn))
    untouched_result = {"compared": len(untouched), "mean_iou_delta": float(np.mean(untouched)) if untouched else None, "regression_count_delta_lt_minus_0.05": int(sum(x < -0.05 for x in untouched)), "all_no_obvious_regression": bool(untouched) and not any(x < -0.05 for x in untouched), "status": "PASS" if untouched and not any(x < -0.05 for x in untouched) else "FAIL" if untouched else "NOT_COMPUTABLE"}
    return {"evaluated_frames": len(details), "identity_utility": 0.5 * (yes_iou - ref_iou) + 0.5 * (ref_error - yes_error), "target_iou_delta": yes_iou - ref_iou, "future_identity_error_reduction": ref_error - yes_error, "recorrection_proxy_reduction": int(ref_recorrect - yes_recorrect), "assignment_change_count": int(sum(x["assignment_changed"] for x in details)), "assignment_change_correct_count": int(sum(x["assignment_change_correct"] for x in details)), "assignment_change_incorrect_count": int(sum(x["assignment_change_incorrect"] for x in details)), "assignment_change_neutral_count": int(sum(x["assignment_change_neutral"] for x in details)), "assignment_no_change_count": int(sum(x["assignment_no_change"] for x in details)), "untouched_regression": untouched_result, "frame_details": details}


def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    valid = [row for row in rows if row["identity_utility"] is not None]
    by_sequence: dict[str, list[float]] = defaultdict(list)
    for row in valid: by_sequence[str(row["sequence"])].append(float(row["identity_utility"]))
    clusters = {sequence: float(np.mean(values)) for sequence, values in by_sequence.items() if values}; names = sorted(clusters)
    rng = np.random.default_rng(BOOTSTRAP_SEED); boot = [float(np.mean([clusters[name] for name in rng.choice(names, len(names), replace=True)])) for _ in range(BOOTSTRAP_REPS)] if names else []
    return {"event_count": len(valid), "independent_sequence_count": len(names), "identity_utility": float(np.mean([x["identity_utility"] for x in valid])) if valid else None, "target_iou_delta": float(np.mean([x["target_iou_delta"] for x in valid])) if valid else None, "future_identity_error_reduction": float(np.mean([x["future_identity_error_reduction"] for x in valid])) if valid else None, "recorrection_proxy_reduction": float(np.mean([x["recorrection_proxy_reduction"] for x in valid])) if valid else None, "assignment_change_count": int(sum(x["assignment_change_count"] for x in valid)), "assignment_change_correct_count": int(sum(x["assignment_change_correct_count"] for x in valid)), "assignment_change_incorrect_count": int(sum(x["assignment_change_incorrect_count"] for x in valid)), "assignment_change_neutral_count": int(sum(x["assignment_change_neutral_count"] for x in valid)), "assignment_no_change_count": int(sum(x["assignment_no_change_count"] for x in valid)), "untouched_regression": {"all_no_obvious_regression": bool(valid) and all(x["untouched_regression"].get("all_no_obvious_regression", False) for x in valid), "mean_iou_delta": float(np.mean([x["untouched_regression"]["mean_iou_delta"] for x in valid if x["untouched_regression"].get("mean_iou_delta") is not None])) if valid else None}, "sequence_cluster_bootstrap_95ci": {"lower": float(np.quantile(boot, 0.025)) if boot else None, "upper": float(np.quantile(boot, 0.975)) if boot else None, "seed": BOOTSTRAP_SEED, "replicates": BOOTSTRAP_REPS, "clusters": len(names), "cluster_weighting": "equal_sequence_mean", "cluster_mean_identity_utility": float(np.mean(list(clusters.values()))) if clusters else None, "event_weighted_identity_utility": float(np.mean([x["identity_utility"] for x in valid])) if valid else None}}


def validate_runtime(events: dict[str, dict[str, Any]]) -> dict[str, Any]:
    files = sorted(RUNTIME.glob("*.json"))
    if len(files) != 24 or {p.stem for p in files} != set(events): raise RuntimeError("N47 runtime event set incomplete")
    frames_checked = 0
    for event_id, event in sorted(events.items()):
        payload = load(RUNTIME / f"{event_id}.json")
        if payload.get("runtime_boundary", {}).get("runtime_future_gt_used") is not False: raise RuntimeError(f"runtime GT flag {event_id}")
        for variant in VARIANTS:
            frames = payload["variants"][variant]["frames"]
            if len(frames) != 100: raise RuntimeError(f"runtime frames {event_id}/{variant}")
            expected = list(range(int(frames[0]["frame"]), int(frames[0]["frame"]) + 100))
            if [int(x["frame"]) for x in frames] != expected: raise RuntimeError(f"frame gap {event_id}/{variant}")
            for frame in frames:
                if frame.get("probe", {}).get("runtime_future_gt_used") is not False: raise RuntimeError(f"frame GT flag {event_id}/{variant}/{frame['frame']}")
                no, write, plus = frame["no_write"], frame["write_baseline"], frame["write_plus_n47"]
                if candidate_signature(write) != candidate_signature(plus): raise RuntimeError(f"candidate rows changed {event_id}/{variant}/{frame['frame']}")
                if write["public_id_order"] != plus["public_id_order"]: raise RuntimeError(f"axis changed {event_id}/{variant}/{frame['frame']}")
                if len(set(write["candidate_native_ids"])) != len(write["candidate_native_ids"]): raise RuntimeError(f"native IDs not unique {event_id}/{variant}/{frame['frame']}")
                frames_checked += 1
    return {"status": "PASS", "event_count": 24, "frames_checked": frames_checked, "runtime_future_gt_used": False, "gt_loaded": False}


def posthoc(events: dict[str, dict[str, Any]], validation: dict[str, Any]) -> dict[str, Any]:
    from sam3_intermot.datasets.dancetrack import DanceTrackDataset
    from scripts.n36_real_eval_common import DATA_ROOT
    from scripts.n43_full_matrix_common import iou
    mapping_all = load(N43_MAP)["public_to_gt_mapping"]
    sequences = sorted({str(x["sequence"]) for x in events.values()}); dataset = DanceTrackDataset(str(DATA_ROOT), sequences=sequences, split="train"); gt = {sequence: dataset.load_gt(sequence) for sequence in sequences}
    rows: dict[tuple[str, str, int], list[dict[str, Any]]] = defaultdict(list); event_results = {}
    POSTHOC.mkdir(parents=True, exist_ok=True)
    for event_id, event in sorted(events.items()):
        mapping = {int(pid): int(gid) for pid, gid in mapping_all.get(event_id, {}).items()}; runtime = load(RUNTIME / f"{event_id}.json"); output = {"protocol": "N47_GLOBAL_ASSIGNMENT_POSTHOC_EVENT_V1", "status": "PASS", "event_id": event_id, "sequence": str(event["sequence"]), "interaction_source": "simulated_from_gt", "not_real_human_evidence": True, "runtime_future_gt_used": False, "gt_loaded_posthoc": True, "variants": {}}
        for variant in VARIANTS:
            frames = runtime["variants"][variant]["frames"]; no = [x["no_write"] for x in frames]; write = [x["write_baseline"] for x in frames]; plus = [x["write_plus_n47"] for x in frames]; output["variants"][variant] = {"runtime_frame_count": len(frames), "horizons": {}}
            for horizon in HORIZONS:
                memory = compare(no, write, event, mapping, gt[str(event["sequence"])], horizon, iou); incremental = compare(write, plus, event, mapping, gt[str(event["sequence"])], horizon, iou)
                app = {"score_cells_changed": int(sum(len(x["probe"]["changed_cells"]) for x in frames[:horizon])), "assignment_changes": int(sum(bool(x["probe"]["assignment_changed"]) for x in frames[:horizon])), "swap_allowed": True}
                output["variants"][variant]["horizons"][str(horizon)] = {"memory_effect_no_write_to_write_baseline": memory, "n47_incremental_effect_write_baseline_to_write_plus_n47": incremental, "application_counts_through_horizon": app}
                rows[("memory", variant, horizon)].append({"event_id": event_id, "sequence": str(event["sequence"]), **memory}); rows[("incremental", variant, horizon)].append({"event_id": event_id, "sequence": str(event["sequence"]), **incremental})
        (POSTHOC / f"{event_id}.json").write_text(__import__("json").dumps(output, indent=2, allow_nan=False) + "\n", encoding="utf-8"); event_results[event_id] = output
    aggregates = {effect: {variant: {str(h): aggregate(rows[(effect, variant, h)]) for h in HORIZONS} for variant in VARIANTS} for effect in ("memory", "incremental")}
    result = {"schema": "N47_GLOBAL_ASSIGNMENT_PROBE_RESULT_V1", "status": "PASS", "protocol": {"source": "frozen N42 no/write branches", "runtime_future_gt_used": False, "gt_loaded_only_after_runtime_validation": True, "bootstrap": "sequence_mean_then_equal_sequence_cluster_bootstrap", "seed": BOOTSTRAP_SEED, "replicates": BOOTSTRAP_REPS}, "inputs": {"n42_runtime": str(N42_RUNTIME), "n47_checkpoint": str(CHECKPOINT)}, "outputs": {"runtime": str(RUNTIME), "posthoc": str(POSTHOC), "result": str(RESULT)}, "event_count": 24, "variant_count": 5, "horizons": list(HORIZONS), "interaction_source": "simulated_from_gt", "real_human_tape_created": False, "id_switch_metric": "NOT_COMPUTABLE_NO_COMPLETE_TRACKEVAL_SEQUENCE_INPUT", "runtime_validation": validation, "effects": aggregates, "attribution": {"memory_effect": "write_baseline minus no_write", "n47_incremental_effect": "write_plus_n47 minus write_baseline"}}
    write_json(RESULT, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=None)
    args = parser.parse_args()
    configure_output(args.output_root)
    status = {"status": "FAIL", "protocol": "N47_STAGE_04_GLOBAL_PROBE_REPLAY_V1", "started_at": now()}
    try:
        events = event_map(); runtime = run_runtime(events); validation = validate_runtime(events); final = posthoc(events, validation)
        status.update({"status": "PASS", "command": ["python", "scripts/n47_stage04_global_probe_replay.py", *( ["--output-root", str(args.output_root)] if args.output_root else [])], "inputs": {"n42_runtime": str(N42_RUNTIME), "n47_checkpoint": str(CHECKPOINT)}, "outputs": {"runtime": str(RUNTIME), "posthoc": str(POSTHOC), "result": str(RESULT)}, "metrics": {"runtime": runtime["metrics"], "validation": validation, "effects": final["effects"]}, "gate_checks": {"runtime_complete": True, "posthoc_after_runtime_validation": True, "same_prefix_event_candidates": True, "all_24_events": True, "all_5_variants": True, "all_horizons": True, "global_hungarian": True, "explicit_none": True, "swap_allowed": True, "runtime_future_gt_false": True, "gt_loaded_posthoc": True, "equal_sequence_bootstrap": True, "standard_mot_not_computable": True, "simulated_provenance": True, "production_authorized": False}, "failure_root_cause": "This is a simulated structural probe; efficacy is interpreted only from the frozen write-baseline-to-global-assignment increment, separately from memory effect.", "next_action": "Run integrity checks and finalize the semantic gate; real human tape/full-loop remain hard gates.", "runtime_future_gt_used": False, "gt_loaded_posthoc": True, "finished_at": now()})
        write_json(STAGE, status); print(__import__("json").dumps({"status": "PASS", "runtime_frames": runtime["metrics"]["frames"], "result": str(RESULT)}))
    except Exception as exc:
        status.update({"status": "FAIL_PRESERVED", "failure_root_cause": f"{type(exc).__name__}: {exc}", "outputs": {}, "metrics": {}, "gate_checks": {"false_pass": False}, "next_action": "Preserve this failure, fix only the first actionable issue, smoke and rerun without changing the frozen protocol.", "runtime_future_gt_used": False, "finished_at": now()})
        ATTEMPTS.mkdir(parents=True, exist_ok=True); write_json(ATTEMPTS / f"stage_04_failure_{datetime.now().strftime('%Y%m%dT%H%M%S')}.json", status); write_json(STAGE, status); raise


if __name__ == "__main__":
    main()
