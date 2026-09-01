#!/usr/bin/env python3
"""Run the single frozen N71 normalized base/pair fusion probe.

This continuation consumes already audited N71 runtime artifacts.  It does
not run model inference, change candidate streams, or use GT while creating
the new assignments.  GT is loaded only after the new CPU branch artifacts
have passed their own structural audit.
"""
from __future__ import annotations

import argparse
import copy
import gc
import json
import os
import sys
import tempfile
import traceback
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import n70_association_common as n70  # noqa: E402
from scripts import n71_global_matrix_common as global_common  # noqa: E402
from scripts.n71_replay_global_matrix import (  # noqa: E402
    DATA_ROOT,
    HORIZONS,
    VARIANTS,
    BOOTSTRAP_REPS,
    BOOTSTRAP_SEEDS,
    DEFAULT_RUNTIME_ROOT,
    DATASET_MANIFEST,
    PROTOCOL,
    atomic_json,
    atomic_jsonl,
    array_hash,
    box_iou,
    candidate_audit,
    load_event_details,
    normalized_assignment_columns,
    posthoc_outcome,
    row_score_audit,
    sha256_file,
    summarize_method,
    validate_runtime_artifacts,
)


AMENDMENT = ROOT / "outputs/N71/protocol_amendment_attempt2.json"
DEFAULT_PROBE_ROOT = Path("/path/to/cache/SAM3_InterMOT_N71/normalized_fusion_probe_attempt1")
DEFAULT_MANIFEST = ROOT / "outputs/N71/replay/normalized_fusion_probe_manifest_attempt1.json"
DEFAULT_RESULT = ROOT / "outputs/N71/replay/normalized_fusion_probe_results_attempt1.json"
DEFAULT_STAGE = ROOT / "outputs/N71/stage_06_status.json"


def bootstrap_cluster_means(cluster_means: dict[str, float], seed: int) -> dict[str, Any]:
    if not cluster_means:
        return {"sequence_count": 0, "mean": None, "ci95": [None, None], "seed": int(seed), "repetitions": BOOTSTRAP_REPS, "cluster_means": {}}
    values = np.asarray(list(cluster_means.values()), dtype=np.float64)
    rng = np.random.default_rng(seed)
    draws = values[rng.integers(0, len(values), size=(BOOTSTRAP_REPS, len(values)))]
    means = draws.mean(axis=1)
    return {"sequence_count": len(values), "mean": float(np.mean(values)), "ci95": [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))], "seed": int(seed), "repetitions": BOOTSTRAP_REPS, "cluster_means": dict(sorted((str(key), float(value)) for key, value in cluster_means.items()))}


def run_cpu_probe(source_runtime_root: Path, output_root: Path, manifest_path: Path, events: dict[str, dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if output_root.exists() and any(output_root.iterdir()):
        raise RuntimeError(f"normalized probe output root is not empty: {output_root}")
    source_rows, source_audit = validate_runtime_artifacts(runtime_root=source_runtime_root, events=events, event_limit=None)
    # ``source_rows`` is already fully validated and contains no GT.  Group
    # them back into frame records so temporal state is advanced in frame
    # order independently for every event and variant.
    by_event: defaultdict[str, defaultdict[int, dict[str, Any]]] = defaultdict(lambda: defaultdict(dict))
    for info in source_rows:
        frame_row = info["frame_row"]
        by_event[str(frame_row["event_id"])][int(frame_row["frame"])][str(info["variant"])] = info
    probe_root = output_root / "event_artifacts"
    probe_root.mkdir(parents=True, exist_ok=True)
    completed = []
    output_rows: list[dict[str, Any]] = []
    for event_id in sorted(by_event):
        event = events[event_id]
        histories = {variant: {} for variant in VARIANTS}
        rows_for_event: list[dict[str, Any]] = []
        for frame_number in sorted(by_event[event_id]):
            for variant in VARIANTS:
                info = by_event[event_id][frame_number][variant]
                value = info["variant_data"]
                base = np.asarray(value["base_score_matrix"], dtype=np.float64)
                pair = np.asarray(value["global_matrix"]["score_matrix"], dtype=np.float64)
                none = np.asarray(value["global_none_scores"], dtype=np.float64)
                n, p = base.shape
                if pair.shape != (n, p) or none.shape != (n,) or not np.all(np.isfinite(base)) or not np.all(np.isfinite(pair)) or not np.all(np.isfinite(none)):
                    raise RuntimeError(f"normalized probe source matrix malformed: {event_id}/{variant}/{frame_number}")
                base_aug = np.concatenate([base.reshape(-1), np.zeros(n, dtype=np.float64)])
                pair_aug = np.concatenate([pair.reshape(-1), none])
                base_std = float(np.std(base_aug))
                pair_std = float(np.std(pair_aug))
                base_norm = (base_aug - float(np.mean(base_aug))) / max(base_std, 1.0e-6)
                pair_norm = (pair_aug - float(np.mean(pair_aug))) / max(pair_std, 1.0e-6)
                fused = (base_norm[: n * p] + pair_norm[: n * p]).reshape(n, p)
                fused_none = base_norm[n * p :] + pair_norm[n * p :]
                if not np.all(np.isfinite(fused)) or not np.all(np.isfinite(fused_none)):
                    raise FloatingPointError(f"normalized fusion is nonfinite: {event_id}/{variant}/{frame_number}")
                candidates = value["candidate_rows_audit"]
                public_ids = [int(item) for item in value["public_id_order"]]
                candidate_rows = [{"native_tid": int(item["native_tid"])} for item in candidates]
                fused_assignment = global_common.explicit_none_hungarian(fused, fused_none, public_ids, candidate_rows)
                guarded, histories[variant], temporal_audit = global_common.apply_temporal_guard(
                    fused_assignment,
                    fused,
                    fused_none,
                    candidate_rows,
                    public_ids,
                    frame_number,
                    target_native_id=None,
                    history=histories[variant],
                    window_frames=3,
                    hysteresis_margin=global_common.HYSTERESIS_MARGIN,
                )
                delta = fused - base
                normalized = normalized_assignment_columns(fused_assignment)
                normalized_temporal = normalized_assignment_columns(guarded)
                branch = {
                    "schema": "N71_NORMALIZED_FUSION_VARIANT_FRAME_V1",
                    "method": "GLOBAL_NORMALIZED_FUSION",
                    "temporal_method": "GLOBAL_NORMALIZED_FUSION_TEMPORAL",
                    "event_id": event_id,
                    "variant": variant,
                    "frame": frame_number,
                    "candidate_count": n,
                    "identity_count": p,
                    "score_matrix": fused.astype(float).tolist(),
                    "none_scores": fused_none.astype(float).tolist(),
                    "assignment_columns": normalized,
                    "assignment_public_ids": fused_assignment["assigned_public_ids"],
                    "temporal_assignment_columns": normalized_temporal,
                    "temporal_assignment_public_ids": guarded["assigned_public_ids"],
                    "score_audit": row_score_audit(fused, np.asarray(normalized, dtype=np.int64)),
                    "temporal_score_audit": row_score_audit(fused, np.asarray(normalized_temporal, dtype=np.int64)),
                    "base_mean": float(np.mean(base_aug)),
                    "base_std": base_std,
                    "pair_mean": float(np.mean(pair_aug)),
                    "pair_std": pair_std,
                    "score_cells_changed": int(np.sum(np.abs(delta) > 1.0e-12)),
                    "max_abs_score_delta": float(np.max(np.abs(delta))) if delta.size else 0.0,
                    "none_column_spread": float(np.max(np.abs(fused_none - fused_none[:1]))) if fused_none.size else 0.0,
                    "temporal_guard": temporal_audit,
                    "source_runtime_frame_sha256": str(info["variant_data"]["source_frame_json_sha256"]),
                    "candidate_rows_mapping_sha256": array_hash(np.asarray([int(item["native_tid"]) for item in candidates], dtype=np.int64), np.int64),
                    "runtime_future_gt_used": False,
                    "runtime_gt_read": False,
                    "target_native_id_sent_to_runtime": False,
                    "interaction_source": "simulated_from_gt",
                    "production_authorized": False,
                }
                rows_for_event.append(branch)
                output_rows.append(branch)
        artifact_path = probe_root / f"{event_id}.jsonl"
        atomic_jsonl(artifact_path, rows_for_event)
        if len(rows_for_event) != 500:
            raise RuntimeError(f"normalized probe event artifact count mismatch: {event_id}")
        completed.append({"event_id": event_id, "sequence": event["sequence"], "artifact": str(artifact_path), "artifact_sha256": sha256_file(artifact_path), "variant_frame_count": len(rows_for_event)})
        atomic_json(manifest_path, {"schema": "N71_NORMALIZED_FUSION_PROBE_MANIFEST_V1", "status": "IN_PROGRESS", "runtime_source_root": str(source_runtime_root), "probe_root": str(probe_root), "completed_event_count": len(completed), "event_count": len(events), "variant_frame_count": len(output_rows), "completed": completed, "runtime_future_gt_used": False, "interaction_source": "simulated_from_gt", "production_authorized": False})
        print(json.dumps({"events_completed": len(completed), "events_total": len(events), "variant_frames": len(output_rows)}, sort_keys=True), flush=True)
    final_manifest = {"schema": "N71_NORMALIZED_FUSION_PROBE_MANIFEST_V1", "status": "PASS_CPU_ASSIGNMENT_ARTIFACTS", "runtime_source_root": str(source_runtime_root), "probe_root": str(probe_root), "completed_event_count": len(completed), "event_count": len(events), "frame_count": len(output_rows) // len(VARIANTS), "variant_frame_count": len(output_rows), "completed": completed, "source_runtime_audit": source_audit, "protocol_amendment_sha256": sha256_file(AMENDMENT), "runtime_future_gt_used": False, "interaction_source": "simulated_from_gt", "production_authorized": False}
    atomic_json(manifest_path, final_manifest)
    return output_rows, final_manifest


def validate_probe_artifacts(probe_root: Path, events: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    files = sorted(probe_root.glob("*.jsonl"))
    if {path.stem for path in files} != set(events):
        raise RuntimeError(f"normalized probe file set mismatch: {len(files)} vs {len(events)}")
    rows: list[dict[str, Any]] = []
    seen: set[tuple[str, str, int]] = set()
    for path in files:
        event_id = path.stem
        lines = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        if len(lines) != 500:
            raise RuntimeError(f"normalized probe rows mismatch: {event_id} {len(lines)}")
        for row in lines:
            key = (event_id, str(row.get("variant")), int(row.get("frame", -1)))
            if key in seen:
                raise RuntimeError(f"duplicate normalized probe key: {key}")
            seen.add(key)
            if row.get("runtime_future_gt_used") is not False or row.get("runtime_gt_read") is not False or row.get("target_native_id_sent_to_runtime") is not False:
                raise RuntimeError(f"normalized probe runtime contract failed: {key}")
            n, p = int(row["candidate_count"]), int(row["identity_count"])
            if str(row["variant"]) not in VARIANTS or np.asarray(row["score_matrix"], dtype=np.float64).shape != (n, p) or np.asarray(row["none_scores"], dtype=np.float64).shape != (n,):
                raise RuntimeError(f"normalized probe matrix axes failed: {key}")
            # NONE is a candidate-level alternative.  Its score is expected
            # to be a length-n vector and may legitimately vary by candidate;
            # the source replay's zero-spread assertion applied to an older
            # model output and must not be reused as a validator condition.
            if len(row["assignment_columns"]) != n or len(row["temporal_assignment_columns"]) != n or not np.isfinite(float(row["none_column_spread"])):
                raise RuntimeError(f"normalized probe assignment axes failed: {key}")
            rows.append(row)
    if len(seen) != len(events) * 100 * len(VARIANTS):
        raise RuntimeError(f"normalized probe unique row count failed: {len(seen)}")
    return rows


def load_gt_after_runtime_audit(events: dict[str, dict[str, Any]]) -> dict[str, dict[int, dict[int, list[float]]]]:
    from sam3_intermot.datasets.dancetrack import DanceTrackDataset

    sequences = sorted({str(event["sequence"]) for event in events.values()})
    dataset = DanceTrackDataset(str(DATA_ROOT), sequences=sequences, split="train")
    output: dict[str, dict[int, dict[int, list[float]]]] = {}
    for sequence in sequences:
        output[sequence] = {}
        for frame, gt_frame in dataset.load_gt(sequence).items():
            output[sequence][int(frame)] = {int(identity): [float(value) for value in box] for identity, box in zip(gt_frame.gt_ids, gt_frame.boxes)}
    return output


def score_probe(rows: list[dict[str, Any]], source_rows: list[dict[str, Any]], events: dict[str, dict[str, Any]], protocol: dict[str, Any]) -> dict[str, Any]:
    # This function is called only after validate_probe_artifacts.  Its GT
    # load is deliberately below the runtime/branch validation boundary.
    gt_by_sequence = load_gt_after_runtime_audit(events)
    split_by_sequence: dict[str, str] = {}
    for split_name, values in protocol.get("sequence_split", {}).items():
        if not isinstance(values, (list, tuple)):
            continue
        for sequence in values:
            split_by_sequence[str(sequence)] = str(split_name)
    source_index = {(str(info["frame_row"]["event_id"]), str(info["variant"]), int(info["frame_row"]["frame"])): info for info in source_rows}
    probe_index = {(str(row["event_id"]), str(row["variant"]), int(row["frame"])): row for row in rows}
    outcomes: list[dict[str, Any]] = []
    for key, branch in sorted(probe_index.items()):
        info = source_index[key]
        value = copy.deepcopy(info["variant_data"])
        value["global_matrix"] = {
            "score_matrix": branch["score_matrix"],
            "assignment_columns": branch["assignment_columns"],
            "assignment_public_ids": branch["assignment_public_ids"],
            "score_audit": branch["score_audit"],
            "score_cells_changed": branch["score_cells_changed"],
            "max_abs_score_delta": branch["max_abs_score_delta"],
        }
        value["global_matrix_temporal"] = {
            "score_matrix": branch["score_matrix"],
            "assignment_columns": branch["temporal_assignment_columns"],
            "assignment_public_ids": branch["temporal_assignment_public_ids"],
            "score_audit": branch["temporal_score_audit"],
            "score_cells_changed": branch["score_cells_changed"],
            "max_abs_score_delta": branch["max_abs_score_delta"],
        }
        frame_info = {"frame_row": info["frame_row"], "variant": info["variant"], "variant_data": value, "event": info["event"]}
        for method, assignment in (("GLOBAL_NORMALIZED_FUSION", np.asarray(branch["assignment_columns"], dtype=np.int64)), ("GLOBAL_NORMALIZED_FUSION_TEMPORAL", np.asarray(branch["temporal_assignment_columns"], dtype=np.int64))):
            outcome = posthoc_outcome(frame_info, method, assignment, gt_by_sequence=gt_by_sequence)
            outcome["split"] = split_by_sequence.get(str(outcome["sequence"]), "UNKNOWN")
            outcomes.append(outcome)
    methods = {method: summarize_method(outcomes, method, split_by_sequence) for method in ("GLOBAL_NORMALIZED_FUSION", "GLOBAL_NORMALIZED_FUSION_TEMPORAL")}
    gate_details: dict[str, Any] = {}
    for method, summary in methods.items():
        gate_details[method] = {}
        for horizon in HORIZONS:
            item = summary["horizons"][str(horizon)]
            lower = item["sequence_cluster_bootstrap_utility"]["ci95"][0]
            gate_details[method][str(horizon)] = {"ci_lower_bound": lower, "strict_positive": bool(lower is not None and lower > 0.0), "candidate_present_improvement_count": item["candidate_present_improvement_count"], "baseline_wrong_reassociation_count": item["baseline_wrong_reassociation_count"], "wrong_reassociation_count": item["wrong_reassociation_count"], "new_wrong_reassociation_count": item["new_wrong_reassociation_count"], "untouched_regression_total": item["untouched_regression_total"]}
    required = tuple(methods)
    strict_ci = all(gate_details[method][str(horizon)]["strict_positive"] for method in required for horizon in HORIZONS)
    candidate_improvement = all(any(gate_details[method][str(horizon)]["candidate_present_improvement_count"] > 0 for horizon in HORIZONS) for method in required)
    no_new_wrong = all(methods[method]["new_wrong_reassociation_total"] == 0 for method in required)
    untouched_safe = all(methods[method]["untouched_regression_total"] == 0 for method in required)
    assignment_crossing = all(methods[method]["all_frame_assignment_change_rate"] is not None and methods[method]["all_frame_assignment_change_rate"] > 0.0 for method in required)
    effect_beyond_first = all(methods[method]["assignment_changes_after_event_plus_one"] > 0 for method in required)
    integrity = bool(outcomes) and all(row["runtime_future_gt_used"] is False and row["posthoc_gt_used"] is True for row in outcomes)
    research_gate = "PASS_FUTURE_EFFECT" if all((strict_ci, candidate_improvement, no_new_wrong, untouched_safe, assignment_crossing, effect_beyond_first, integrity)) else "FAIL_FUTURE_EFFECT"
    return {
        "schema": "N71_NORMALIZED_FUSION_PROBE_RESULTS_V1",
        "status": "PASS_EXECUTION_FUTURE_EFFECT_PASS" if research_gate == "PASS_FUTURE_EFFECT" else "PASS_EXECUTION_FAIL_FUTURE_EFFECT",
        "event_count": len(events),
        "independent_sequence_count": len({str(event["sequence"]) for event in events.values()}),
        "runtime_frame_count": len(rows) // len(VARIANTS),
        "variant_frame_count": len(rows),
        "methods": methods,
        "gate": {"research_gate": research_gate, "candidate_present_improvement": candidate_improvement, "strict_ci_lower_bound_positive_all_horizons": strict_ci, "no_new_wrong_reassociation": no_new_wrong, "untouched_regression_safe": untouched_safe, "real_assignment_crossing": assignment_crossing, "memory_effect_not_first_frame_only": effect_beyond_first, "runtime_integrity": integrity, "gate_details": gate_details, "runtime_future_gt_used": False, "posthoc_gt_loaded_after_runtime_audit": True, "interaction_source": "simulated_from_gt", "real_human_tape": False, "production_authorized": False},
        "bootstrap": {"repetitions": BOOTSTRAP_REPS, "seed_by_horizon": {str(key): value for key, value in BOOTSTRAP_SEEDS.items()}, "clusters": "independent sequence"},
        "runtime_future_gt_used": False,
        "posthoc_gt_used": True,
        "interaction_source": "simulated_from_gt",
        "real_human_tape": False,
        "production_authorized": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runtime-root", type=Path, default=DEFAULT_RUNTIME_ROOT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--result", type=Path, default=DEFAULT_RESULT)
    parser.add_argument("--stage-status", type=Path, default=DEFAULT_STAGE)
    parser.add_argument("--probe-root", type=Path, default=None)
    parser.add_argument("--attempt", default="1")
    args = parser.parse_args()
    try:
        events = load_event_details()
        protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
        source_rows, source_audit = validate_runtime_artifacts(args.runtime_root.resolve(), events, event_limit=None)
        probe_output_root = args.probe_root.resolve() if args.probe_root is not None else (DEFAULT_PROBE_ROOT if str(args.attempt) == "1" else DEFAULT_PROBE_ROOT.parent / f"normalized_fusion_probe_attempt{args.attempt}")
        probe_rows, probe_manifest = run_cpu_probe(args.runtime_root.resolve(), probe_output_root, args.manifest.resolve(), events)
        # The probe output is separate from the source runtime output.  Audit
        # its keys before importing the dataset/GT for the posthoc stage.
        validated_probe_rows = validate_probe_artifacts(Path(probe_manifest["probe_root"]), events)
        if len(validated_probe_rows) != len(probe_rows):
            raise RuntimeError("normalized probe in-memory/on-disk row count mismatch")
        probe_audit = {"schema": "N71_NORMALIZED_FUSION_PROBE_AUDIT_V1", "status": "PASS", "event_count": len(events), "variant_frame_count": len(validated_probe_rows), "duplicate_keys": 0, "missing_keys": 0, "none_column_spread_max": float(max(row["none_column_spread"] for row in validated_probe_rows)), "runtime_future_gt_used": False, "production_authorized": False}
        audit_path = args.result.resolve().parent / (args.result.resolve().stem + "_audit.json")
        atomic_json(audit_path, probe_audit)
        results = score_probe(validated_probe_rows, source_rows, events, protocol)
        results.update({"created_at_utc": datetime_now(), "runtime_manifest": str(ROOT / "outputs/N71/replay/global_matrix_runtime_manifest_attempt1.json"), "runtime_source_audit": source_audit, "probe_manifest": str(args.manifest.resolve()), "probe_manifest_sha256": sha256_file(args.manifest.resolve()), "probe_audit": str(audit_path), "probe_audit_sha256": sha256_file(audit_path), "protocol_amendment": str(AMENDMENT), "protocol_amendment_sha256": sha256_file(AMENDMENT), "dataset_manifest": str(DATASET_MANIFEST), "dataset_manifest_sha256": sha256_file(DATASET_MANIFEST)})
        atomic_json(args.result.resolve(), results)
        atomic_json(args.stage_status.resolve(), {"schema": "N71_STAGE_06_STATUS_V1", "status": "PASS_FIXED_NORMALIZED_FUSION_PROBE_POSTHOC_COMPLETE", "probe_manifest": str(args.manifest.resolve()), "probe_manifest_sha256": sha256_file(args.manifest.resolve()), "probe_audit": str(audit_path), "probe_audit_sha256": sha256_file(audit_path), "probe_results": str(args.result.resolve()), "probe_results_sha256": sha256_file(args.result.resolve()), "event_count": len(events), "independent_sequence_count": len({str(event["sequence"]) for event in events.values()}), "variant_frame_count": len(validated_probe_rows), "research_gate": results["gate"]["research_gate"], "runtime_future_gt_used": False, "posthoc_gt_loaded_after_runtime_audit": True, "interaction_source": "simulated_from_gt", "production_authorized": False})
        print(json.dumps({"status": results["status"], "research_gate": results["gate"]["research_gate"], "result": str(args.result.resolve()), "probe_manifest": str(args.manifest.resolve())}, sort_keys=True), flush=True)
    except Exception as exc:
        attempts = ROOT / "outputs/N71/attempts"
        attempts.mkdir(parents=True, exist_ok=True)
        existing = sorted(attempts.glob("n71_normalized_fusion_probe_failure_attempt*.json"))
        path = attempts / f"n71_normalized_fusion_probe_failure_attempt{len(existing) + 1}.json"
        atomic_json(path, {"schema": "N71_NORMALIZED_FUSION_PROBE_FAILURE_V1", "status": "FAIL_PRESERVED", "attempt": str(args.attempt), "failure_type": type(exc).__name__, "failure_message": str(exc), "traceback": traceback.format_exc(), "runtime_root": str(args.runtime_root.resolve()), "protocol_amendment_sha256": sha256_file(AMENDMENT), "runtime_future_gt_used": False, "interaction_source": "simulated_from_gt", "production_authorized": False})
        print(traceback.format_exc(), file=sys.stderr, flush=True)
        print(json.dumps({"status": "FAIL_PRESERVED", "failure_artifact": str(path)}, sort_keys=True), file=sys.stderr, flush=True)
        raise


def datetime_now() -> str:
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat()


if __name__ == "__main__":
    main()
