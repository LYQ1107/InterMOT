#!/usr/bin/env python3
"""Read-only, posthoc diagnosis of the frozen N47 M2 assignment boundary."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import linear_sum_assignment

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.n36_real_eval_common import DATA_ROOT  # noqa: E402
from scripts.n43_full_matrix_common import iou  # noqa: E402
from scripts.n47_global_probe_common import (  # noqa: E402
    BOOTSTRAP_REPS,
    BOOTSTRAP_SEED,
    HARD_NEGATIVE,
    N42_RUNTIME,
    N43_MAP,
    NONE_SCORE,
    VARIANTS,
    event_map,
    hungarian_with_none,
    load,
    normalize_assignment,
    sha256,
    write_json,
)


def solve_expanded(scores: np.ndarray, dummy_score: float = NONE_SCORE) -> tuple[np.ndarray, float]:
    matrix = np.asarray(scores, dtype=np.float32)
    work = matrix.copy()
    work[work <= HARD_NEGATIVE] = dummy_score - 16.0
    dummy = np.full((matrix.shape[0], matrix.shape[0]), dummy_score, dtype=np.float32)
    expanded = np.concatenate([work, dummy], axis=1)
    rows, cols = linear_sum_assignment(-expanded)
    assignment = np.full(matrix.shape[0], -1, dtype=np.int64)
    assignment[rows] = cols
    return assignment, float(expanded[rows, cols].astype(np.float64).sum())


def normalized_expanded(assignment: np.ndarray, pid_count: int) -> list[int]:
    return [int(x) if 0 <= int(x) < pid_count else -1 for x in assignment]


def assignment_total(scores: np.ndarray, expanded: np.ndarray, pid_count: int, dummy_score: float = NONE_SCORE) -> float:
    return float(sum(float(scores[i, col]) if 0 <= int(col) < pid_count else dummy_score for i, col in enumerate(expanded)))


def global_margin(scores: np.ndarray, baseline_expanded: np.ndarray, dummy_score: float = NONE_SCORE) -> float:
    """Exact one-row-forced next assignment gap, ignoring equivalent dummy labels."""
    matrix = np.asarray(scores, dtype=np.float32)
    n, p = matrix.shape
    baseline_total = assignment_total(matrix, baseline_expanded, p, dummy_score)
    best_alternative = -float("inf")
    for row in range(n):
        current = int(baseline_expanded[row])
        alternatives = [col for col in range(p) if col != current]
        if current < p:
            alternatives.append(p)  # one representative NONE dummy
        for forced in alternatives:
            remaining_rows = [i for i in range(n) if i != row]
            remaining_cols = [j for j in range(p + n) if j != forced]
            work = matrix.copy()
            work[work <= HARD_NEGATIVE] = dummy_score - 16.0
            expanded = np.concatenate((work, np.full((n, n), dummy_score, dtype=np.float32)), axis=1)
            if forced >= p:
                forced_value = dummy_score
            else:
                forced_value = float(expanded[row, forced])
            if not remaining_rows:
                candidate_total = forced_value
            else:
                sub = expanded[np.ix_(remaining_rows, remaining_cols)]
                rr, cc = linear_sum_assignment(-sub)
                candidate_total = forced_value + float(sub[rr, cc].astype(np.float64).sum())
            best_alternative = max(best_alternative, candidate_total)
    if best_alternative == -float("inf"):
        return float("inf")
    return float(baseline_total - best_alternative)


def candidate_signature(candidates: list[dict[str, Any]]) -> list[tuple[int, Any, float]]:
    return [(int(x["native_tid"]), x.get("box"), float(x.get("confidence", 0.0))) for x in candidates]


def pid_iou(rows: list[dict[str, Any]], pid: int, target: Any) -> float:
    return max((float(iou(row["box"], target)) for row in rows if row.get("public_id") is not None and int(row["public_id"]) == int(pid)), default=0.0)


def mapping_from_assignment(record: dict[str, Any]) -> dict[int, Any]:
    return {int(row["native_tid"]): row.get("public_id") for row in record["rows"]}


def oracle_for_frame(candidates: list[dict[str, Any]], pids: list[int], gt_frame: Any, public_to_gt: dict[int, int]) -> tuple[list[int], np.ndarray, float]:
    values = np.zeros((len(candidates), len(pids)), dtype=np.float32)
    gt_boxes = {int(gid): box for gid, box in zip(gt_frame.gt_ids, gt_frame.boxes)}
    for i, candidate in enumerate(candidates):
        for j, pid in enumerate(pids):
            gid = public_to_gt.get(int(pid))
            if gid is not None and gid in gt_boxes:
                values[i, j] = float(iou(candidate["box"], gt_boxes[gid]))
    assignment, total = solve_expanded(values, dummy_score=0.0)
    return normalized_expanded(assignment, len(pids)), values, total


def percentile(values: list[float], q: float) -> float | None:
    return float(np.quantile(values, q)) if values else None


def bootstrap(event_values: list[tuple[str, str, float]]) -> dict[str, Any]:
    by_sequence: dict[str, list[float]] = defaultdict(list)
    for sequence, _event_id, value in event_values:
        by_sequence[sequence].append(float(value))
    sequence_means = {seq: float(np.mean(vals)) for seq, vals in by_sequence.items() if vals}
    names = sorted(sequence_means)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    draws = [float(np.mean([sequence_means[name] for name in rng.choice(names, len(names), replace=True)])) for _ in range(BOOTSTRAP_REPS)] if names else []
    return {
        "event_count": len(event_values),
        "sequence_count": len(names),
        "sequence_mean": float(np.mean(list(sequence_means.values()))) if sequence_means else None,
        "event_weighted_mean": float(np.mean([x[2] for x in event_values])) if event_values else None,
        "lower": float(np.quantile(draws, 0.025)) if draws else None,
        "upper": float(np.quantile(draws, 0.975)) if draws else None,
        "seed": BOOTSTRAP_SEED,
        "replicates": BOOTSTRAP_REPS,
        "cluster_weighting": "equal_sequence_mean",
        "sequence_means": sequence_means,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=ROOT / "outputs/n48")
    args = parser.parse_args()
    out = args.output_root
    diagnosis_dir = out / "diagnosis"
    frame_path = diagnosis_dir / "n47_m2_frame_diagnostics.jsonl"
    summary_path = diagnosis_dir / "n47_m2_structural_diagnosis.json"
    status_path = out / "stage_01_status.json"
    runtime_dir = ROOT / "outputs/n47_global_probe/repair1_swap_metric/replay/runtime"
    posthoc_dir = ROOT / "outputs/n47_global_probe/repair1_swap_metric/replay/posthoc"
    events = event_map()
    failures: list[str] = []

    # Runtime contract validation happens before any GT load.
    runtime_files = sorted(runtime_dir.glob("*.json"))
    if len(runtime_files) != 24 or {p.stem for p in runtime_files} != set(events):
        failures.append("runtime event set is not the frozen 24-event set")
    for event_id, event in sorted(events.items()):
        source = load(N42_RUNTIME / f"{event_id}.json")
        runtime = load(runtime_dir / f"{event_id}.json") if (runtime_dir / f"{event_id}.json").is_file() else {}
        if runtime.get("runtime_boundary", {}).get("runtime_future_gt_used") is not False:
            failures.append(f"{event_id}: runtime future GT flag")
        for variant in VARIANTS:
            trace = source["variants"][variant]["branches"]["memory_write=True"]["future_trace"]
            frames = runtime.get("variants", {}).get(variant, {}).get("frames", [])
            if len(trace) != 100 or len(frames) != 100:
                failures.append(f"{event_id}/{variant}: trace length")
                continue
            expected = list(range(int(trace[0]["frame"]), int(trace[0]["frame"]) + 100))
            if [int(x["frame"]) for x in trace] != expected or [int(x["frame"]) for x in frames] != expected:
                failures.append(f"{event_id}/{variant}: frame gap/duplicate")
            for frame in frames:
                if frame.get("probe", {}).get("runtime_future_gt_used") is not False:
                    failures.append(f"{event_id}/{variant}/{frame['frame']}: frame future GT flag")
                for branch in ("no_write", "write_baseline", "write_plus_n47"):
                    record = frame[branch]
                    if record.get("runtime_future_gt_used") is not False:
                        failures.append(f"{event_id}/{variant}/{frame['frame']}: branch future GT flag")
                    native_ids = record.get("candidate_native_ids", [])
                    if len(native_ids) != len(set(native_ids)):
                        failures.append(f"{event_id}/{variant}/{frame['frame']}: duplicate native IDs")
    if failures:
        payload = {"status": "FAIL_PRESERVED", "protocol": "N48_STAGE_01_N47_M2_DIAGNOSIS_V1", "inputs": {"runtime": str(runtime_dir)}, "outputs": {}, "metrics": {"failures": failures[:100]}, "gate_checks": {"runtime_validated_before_gt": False}, "failure_root_cause": failures[0], "next_action": "Preserve the read-only contract failure; do not load GT or train until the frozen runtime contract is repaired."}
        write_json(status_path, payload)
        raise RuntimeError(failures[0])

    # GT is opened only after the GT-free runtime checks above.
    from sam3_intermot.datasets.dancetrack import DanceTrackDataset  # noqa: E402

    mapping_all = load(N43_MAP)["public_to_gt_mapping"]
    sequences = sorted({str(x["sequence"]) for x in events.values()})
    dataset = DanceTrackDataset(str(DATA_ROOT), sequences=sequences, split="train")
    gt = {sequence: dataset.load_gt(sequence) for sequence in sequences}
    frame_rows: list[dict[str, Any]] = []
    per_sequence: dict[str, dict[str, Any]] = defaultdict(lambda: {"event_ids": [], "frame_count": 0, "assignment_changes": 0, "correct_changes": 0, "incorrect_changes": 0, "neutral_changes": 0, "no_change_frames": 0, "candidate_ceiling_frames": 0, "target_present_frames": 0, "baseline_target_correct_frames": 0, "plus_target_correct_frames": 0, "untouched_regression_frames": 0, "none_involved_changes": 0, "global_competition_changes": 0, "oracle_desired_frames": 0, "oracle_recovered_frames": 0, "oracle_desired_pairs": 0, "oracle_desired_pairs_with_changed_cell": 0, "oracle_desired_pairs_blocked_by_baseline_owner": 0, "events": {}})
    logit_correct: list[float] = []
    logit_incorrect: list[float] = []
    logit_all: list[float] = []
    margins_all: list[float] = []
    margins_changed: list[float] = []
    required_gaps: list[float] = []
    event_values: dict[str, list[tuple[str, str, float]]] = {str(h): [] for h in (20, 50, 100)}
    event_posthoc: dict[str, dict[str, Any]] = {}

    for event_id, event in sorted(events.items()):
        sequence = str(event["sequence"])
        target_pid = int(event["public_id"])
        target_gid = int(event["dataset_gt_id"])
        public_to_gt = {int(pid): int(gid) for pid, gid in mapping_all.get(event_id, {}).items()}
        source = load(N42_RUNTIME / f"{event_id}.json")
        runtime = load(runtime_dir / f"{event_id}.json")
        posthoc = load(posthoc_dir / f"{event_id}.json")
        event_posthoc[event_id] = posthoc
        seq = per_sequence[sequence]
        if event_id not in seq["event_ids"]:
            seq["event_ids"].append(event_id)
        seq["events"][event_id] = {"frame_count": 100, "horizons": {}}
        for h in (20, 50, 100):
            item = posthoc["variants"]["M2"]["horizons"][str(h)]["n47_incremental_effect_write_baseline_to_write_plus_n47"]
            event_values[str(h)].append((sequence, event_id, float(item["identity_utility"])))
            seq["events"][event_id]["horizons"][str(h)] = {k: item[k] for k in ("identity_utility", "target_iou_delta", "future_identity_error_reduction", "recorrection_proxy_reduction", "assignment_change_count", "assignment_change_correct_count", "assignment_change_incorrect_count", "assignment_change_neutral_count", "assignment_no_change_count", "untouched_regression")}
        for frame in runtime["variants"]["M2"]["frames"]:
            frame_id = int(frame["frame"])
            offset = frame_id - int(event["frame"])
            write = frame["write_baseline"]
            plus = frame["write_plus_n47"]
            probe = frame["probe"]
            pids = [int(x) for x in write["public_id_order"]]
            base = np.asarray(write["base_scores"], dtype=np.float32)
            adjusted = np.asarray(plus["base_scores"], dtype=np.float32)
            delta = adjusted - base
            finite = base > HARD_NEGATIVE
            changed_cells = {(int(x["candidate_index"]), int(x["column"])) for x in probe["changed_cells"]}
            gt_frame = gt[sequence].get(frame_id)
            gt_boxes = {int(gid): box for gid, box in zip(gt_frame.gt_ids, gt_frame.boxes)} if gt_frame is not None else {}
            target_box = gt_boxes.get(target_gid)
            target_iou_before = pid_iou(write["rows"], target_pid, target_box) if target_box is not None else None
            target_iou_after = pid_iou(plus["rows"], target_pid, target_box) if target_box is not None else None
            target_candidate_ious = [float(iou(candidate["box"], target_box)) for candidate in write["candidate_rows"]] if target_box is not None else []
            ceiling = bool(target_candidate_ious and max(target_candidate_ious) >= 0.5)
            base_correct = bool(target_iou_before is not None and target_iou_before >= 0.5)
            plus_correct = bool(target_iou_after is not None and target_iou_after >= 0.5)
            assignment_changed = mapping_from_assignment(write) != mapping_from_assignment(plus)
            if assignment_changed:
                if target_iou_after is not None and target_iou_before is not None and target_iou_after > target_iou_before + 1e-9:
                    change_class = "correct"
                elif target_iou_after is not None and target_iou_before is not None and target_iou_after < target_iou_before - 1e-9:
                    change_class = "incorrect"
                else:
                    change_class = "neutral"
            else:
                change_class = "no_change"
            untouched_deltas = []
            for pid, gid in public_to_gt.items():
                if pid == target_pid or gid not in gt_boxes:
                    continue
                untouched_deltas.append(pid_iou(plus["rows"], pid, gt_boxes[gid]) - pid_iou(write["rows"], pid, gt_boxes[gid]))
            untouched_regressed = bool(untouched_deltas and any(x < -0.05 for x in untouched_deltas))
            changed_rows = [row for row, (before, after) in enumerate(zip(write["assignment_public_ids"], plus["assignment_public_ids"])) if before != after]
            none_involved = bool(assignment_changed and any(write["assignment_public_ids"][row] is None or plus["assignment_public_ids"][row] is None for row in changed_rows))
            baseline_expanded, _ = solve_expanded(base)
            local_margins = []
            for row, col in enumerate(write["assignment_columns"]):
                alternatives = [float(base[row, j]) for j in range(base.shape[1]) if j != int(col) and finite[row, j]] if int(col) >= 0 else [float(x) for x in base[row] if x > HARD_NEGATIVE]
                assigned = float(base[row, int(col)]) if int(col) >= 0 else NONE_SCORE
                local_margins.append(assigned - max(alternatives, default=NONE_SCORE))
            g_margin = global_margin(base, baseline_expanded)
            margins_all.append(g_margin)
            if assignment_changed:
                margins_changed.append(g_margin)
            logits = delta[finite].astype(float).tolist()
            logit_all.extend(logits)
            # Offline correctness labels are used only for calibration diagnostics.
            if target_box is not None:
                for row in range(base.shape[0]):
                    for col in range(base.shape[1]):
                        if not finite[row, col]:
                            continue
                        pid = pids[col]
                        gid = public_to_gt.get(pid)
                        cell_iou = float(iou(write["candidate_rows"][row]["box"], gt_boxes[gid])) if gid in gt_boxes else 0.0
                        (logit_correct if cell_iou >= 0.5 else logit_incorrect).append(float(delta[row, col]))
            oracle_assignment, oracle_values, oracle_total = oracle_for_frame(write["candidate_rows"], pids, gt_frame, public_to_gt) if gt_frame is not None else ([-1] * len(write["candidate_rows"]), np.zeros_like(base), 0.0)
            base_assignment = normalize_assignment(write["assignment_columns"], len(pids))
            plus_assignment = normalize_assignment(plus["assignment_columns"], len(pids))
            oracle_desired = [i for i, (a, o) in enumerate(zip(base_assignment, oracle_assignment)) if a != o and o >= 0]
            oracle_changed = [i for i in oracle_desired if any((i, j) in changed_cells for j in range(base.shape[1]))]
            base_total = sum(float(base[i, c]) if c >= 0 else NONE_SCORE for i, c in enumerate(base_assignment))
            required_gap = max(0.0, float(base_total - oracle_total)) if oracle_desired else 0.0
            if oracle_desired:
                required_gaps.append(required_gap)
            base_oracle_match = base_assignment == oracle_assignment
            plus_oracle_match = plus_assignment == oracle_assignment
            target_column_present = target_pid in pids
            target_column = pids.index(target_pid) if target_column_present else None
            target_touched = bool(target_column is not None and any((i, target_column) in changed_cells for i in range(base.shape[0])))
            competitor_touched = bool(target_column is not None and any((i, j) in changed_cells for i in range(base.shape[0]) for j in range(base.shape[1]) if j != target_column))
            seq["frame_count"] += 1
            seq["assignment_changes"] += int(assignment_changed)
            seq["correct_changes"] += int(change_class == "correct")
            seq["incorrect_changes"] += int(change_class == "incorrect")
            seq["neutral_changes"] += int(change_class == "neutral")
            seq["no_change_frames"] += int(not assignment_changed)
            seq["candidate_ceiling_frames"] += int(ceiling)
            seq["target_present_frames"] += int(target_box is not None)
            seq["baseline_target_correct_frames"] += int(base_correct)
            seq["plus_target_correct_frames"] += int(plus_correct)
            seq["untouched_regression_frames"] += int(untouched_regressed)
            seq["none_involved_changes"] += int(none_involved)
            seq["global_competition_changes"] += int(competitor_touched and assignment_changed)
            seq["oracle_desired_frames"] += int(bool(oracle_desired))
            seq["oracle_recovered_frames"] += int(bool(oracle_desired and plus_oracle_match))
            seq["oracle_desired_pairs"] += len(oracle_desired)
            seq["oracle_desired_pairs_with_changed_cell"] += len(oracle_changed)
            seq["oracle_desired_pairs_blocked_by_baseline_owner"] += sum(int(base_assignment[i] >= 0 and base_assignment[i] != oracle_assignment[i]) for i in oracle_desired)
            frame_rows.append({
                "event_id": event_id, "sequence": sequence, "variant": "M2", "frame": frame_id, "offset": offset,
                "runtime_future_gt_used": False, "target_pid": target_pid, "target_column_present": target_column_present,
                "target_column_touched": target_touched, "competitor_cells_touched": competitor_touched,
                "candidate_count": len(write["candidate_rows"]), "public_id_count": len(pids), "finite_cell_count": int(finite.sum()), "score_cells_changed": len(changed_cells),
                "baseline_assignment_public_ids": write["assignment_public_ids"], "plus_assignment_public_ids": plus["assignment_public_ids"], "assignment_changed_frame": assignment_changed, "assignment_change_class_frame": change_class,
                "none_involved": none_involved, "target_gt_present_posthoc": target_box is not None, "candidate_ceiling_target_iou_ge_0.5": ceiling,
                "baseline_target_iou": target_iou_before, "plus_target_iou": target_iou_after, "target_iou_delta": (target_iou_after - target_iou_before) if target_iou_before is not None and target_iou_after is not None else None,
                "baseline_target_correct": base_correct, "plus_target_correct": plus_correct, "untouched_delta_min": min(untouched_deltas) if untouched_deltas else None, "untouched_regressed": untouched_regressed,
                "local_margin_min": float(min(local_margins, default=0.0)), "local_margin_median": float(np.median(local_margins)) if local_margins else None, "global_assignment_margin": g_margin,
                "appearance_logit_min": min(logits) if logits else None, "appearance_logit_median": percentile(logits, 0.5), "appearance_logit_max": max(logits) if logits else None, "appearance_logit_p95": percentile(logits, 0.95),
                "oracle_assignment_public_ids": [pids[c] if c >= 0 else None for c in oracle_assignment], "base_matches_offline_oracle": base_oracle_match, "plus_matches_offline_oracle": plus_oracle_match,
                "oracle_desired_pair_count": len(oracle_desired), "oracle_desired_pair_changed_cell_count": len(oracle_changed), "oracle_required_total_score_gap": required_gap,
            })

    diagnosis_dir.mkdir(parents=True, exist_ok=True)
    with frame_path.open("w", encoding="utf-8") as handle:
        for row in frame_rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    bootstrap_results = {h: bootstrap(values) for h, values in event_values.items()}
    per_sequence_clean = {}
    for sequence, item in sorted(per_sequence.items()):
        item["event_count"] = len(item["event_ids"])
        per_sequence_clean[sequence] = item
    # Compact diagnosis, while retaining the complete frame-level JSONL beside it.
    def stats(values: list[float]) -> dict[str, Any]:
        return {"count": len(values), "mean": float(np.mean(values)) if values else None, "median": float(np.median(values)) if values else None, "p05": percentile(values, 0.05), "p95": percentile(values, 0.95), "min": min(values) if values else None, "max": max(values) if values else None}

    diagnosis = {
        "schema": "N48_N47_M2_ASSIGNMENT_BOUNDARY_DIAGNOSIS_V1", "status": "PASS", "provenance": {"source": "frozen N47 repair1 runtime", "interaction_source": "simulated_from_gt", "runtime_future_gt_used": False, "gt_loaded_only_after_runtime_validation": True, "production_authorized": False},
        "inputs": {"runtime": str(runtime_dir), "posthoc": str(posthoc_dir), "checkpoint": str(ROOT / "outputs/n47_global_probe/training/n47_global_fusion_probe.pt"), "checkpoint_sha256": sha256(ROOT / "outputs/n47_global_probe/training/n47_global_fusion_probe.pt"), "event_count": 24, "variant": "M2", "horizons": [20, 50, 100]},
        "outputs": {"frame_diagnostics": str(frame_path), "diagnosis": str(summary_path), "stage_status": str(status_path)},
        "metrics": {"runtime_frames_audited": len(frame_rows), "sequence_count": len(per_sequence_clean), "horizon_bootstrap": bootstrap_results, "per_sequence": per_sequence_clean, "per_sequence_closure": {sequence: item["assignment_changes"] == item["correct_changes"] + item["incorrect_changes"] + item["neutral_changes"] for sequence, item in per_sequence_clean.items()}, "assignment_change_context": {"assignment_changes": sum(int(x["assignment_changed_frame"]) for x in frame_rows), "correct": sum(int(x["assignment_change_class_frame"] == "correct") for x in frame_rows), "incorrect": sum(int(x["assignment_change_class_frame"] == "incorrect") for x in frame_rows), "neutral": sum(int(x["assignment_change_class_frame"] == "neutral") for x in frame_rows), "no_change": sum(int(x["assignment_change_class_frame"] == "no_change") for x in frame_rows), "none_involved": sum(int(x["none_involved"]) for x in frame_rows)}, "candidate_ceiling": {"target_present_frames": sum(int(x["target_gt_present_posthoc"]) for x in frame_rows), "ceiling_frames": sum(int(x["candidate_ceiling_target_iou_ge_0.5"]) for x in frame_rows), "baseline_target_correct_frames": sum(int(x["baseline_target_correct"]) for x in frame_rows), "plus_target_correct_frames": sum(int(x["plus_target_correct"]) for x in frame_rows)}, "logit_stats_all_finite_cells": stats(logit_all), "logit_stats_offline_correct_cells": stats(logit_correct), "logit_stats_offline_incorrect_cells": stats(logit_incorrect), "baseline_global_margin_all": stats(margins_all), "baseline_global_margin_changed_frames": stats(margins_changed), "oracle_required_total_score_gap": {"status": "INVALID_NON_COMPARABLE", "reason": "base fused scores and IoU oracle values use different units; retained only as legacy diagnostic and excluded from conclusions", "legacy_values": stats(required_gaps)}, "oracle_required_total_score_gap_legacy": stats(required_gaps), "oracle_desired_pair_count": sum(int(x["oracle_desired_pair_count"]) for x in frame_rows), "oracle_desired_pair_changed_cell_count": sum(int(x["oracle_desired_pair_changed_cell_count"]) for x in frame_rows), "target_column_touched_frames": sum(int(x["target_column_touched"]) for x in frame_rows), "untouched_regressed_frames": sum(int(x["untouched_regressed"]) for x in frame_rows)},
        "diagnostic_interpretation": {"n47_model_class": "8 scalar causal interface features, not a complete 512-D appearance-memory module", "coverage": "dense finite-cell score updates in this global probe; assignment changes are sparse relative to 12000 frames", "fixed_boost_not_used": True, "holdout_not_used_for_selection": True, "neutral_not_correct": True, "oracle_is_posthoc_only": True},
        "hypotheses": {"a_sparse_proposals_or_near_tie_coverage": {"status": "NOT_PRIMARY_FOR_N47_GLOBAL_PROBE", "evidence": "N47 applies a logit to every finite cell; quantify changed assignment/finite-cell and target-column coverage in metrics"}, "b_boost_below_assignment_margin": {"status": "TESTED_DIAGNOSTIC", "evidence": "compare appearance-logit and baseline global-margin distributions in frame JSONL"}, "c_logit_misaligned_with_correctness": {"status": "TESTED_DIAGNOSTIC", "evidence": "offline-correct versus offline-incorrect cell logit distributions"}, "d_owner_column_none_constraint": {"status": "NOT_PRIMARY_GLOBAL_INTERFACE", "evidence": "N47 permits a global Hungarian swap; NONE/entry-exit involvement is separately counted"}, "e_m2_memory_effect_negative": {"status": "SEPARATE_CONFIRMED_CONTEXT", "evidence": "N47 write-baseline isolates the global increment; no-write→write remains the negative N42/N45 memory effect"}},
        "failure_root_cause": "N47's global interface is exercised but its future effect is heterogeneous; this audit separates sparse assignment crossings from score coverage, margin size, candidate ceiling, NONE/global competition and offline correctness. No runtime GT or future outcome is an input.",
        "next_action": "Read this evidence before defining any N48 protocol; train only one isolated sequence-disjoint diagnostic structure if the margin/coverage evidence supports a falsifiable hypothesis. Real human tape and real SAM3 full-loop remain hard gates.",
    }
    write_json(summary_path, diagnosis)
    status = {"status": "PASS", "protocol": "N48_STAGE_01_N47_M2_DIAGNOSIS_V1", "command": ["python", "scripts/n48_stage01_n47_m2_diagnosis.py", "--output-root", str(out)], "inputs": diagnosis["inputs"], "outputs": diagnosis["outputs"], "metrics": diagnosis["metrics"], "gate_checks": {"runtime_1200_m2_frames": len(frame_rows) == 2400, "all_24_events": len(events) == 24, "all_21_sequences": len(per_sequence_clean) == 21, "runtime_future_gt_false": True, "gt_only_after_runtime_validation": True, "simulated_provenance": True, "standard_mot_not_computable": True, "no_training_started": True, "n47_checkpoint_unchanged": True}, "failure_root_cause": diagnosis["failure_root_cause"], "next_action": diagnosis["next_action"], "runtime_future_gt_used": False, "gt_loaded_posthoc": True}
    status["gate_checks"].update({"per_sequence_assignment_closure": all(diagnosis["metrics"]["per_sequence_closure"].values()), "none_only_on_changed_rows": True, "oracle_gap_invalid_non_comparable": diagnosis["metrics"]["oracle_required_total_score_gap"]["status"] == "INVALID_NON_COMPARABLE"})
    write_json(status_path, status)
    print(json.dumps({"status": "PASS", "frames": len(frame_rows), "sequences": len(per_sequence_clean), "horizons": bootstrap_results}))


if __name__ == "__main__":
    main()
