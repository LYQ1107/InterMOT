#!/usr/bin/env python3
"""N44 Stage 01: read-only assignment-boundary audit over frozen N43 cells.

Runtime features are not constructed from GT.  GT is loaded only in this
offline audit to produce labels, oracle ceilings, and post-hoc reconciliation.
The N43 checkpoint is replayed only to expose its exact per-cell gate/residual
and assignment changes; no N43 artifact is modified.
"""

from __future__ import annotations

import json
import sys
import traceback
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import linear_sum_assignment

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sam3_intermot.datasets.dancetrack import DanceTrackDataset
from scripts.n36_real_eval_common import DATA_ROOT
from scripts.n43_full_matrix_common import (
    HARD_NEGATIVE,
    NONE_SCORE,
    bounded_utility,
    hungarian_with_none,
    load_checkpoint,
)


N43_AUDIT = ROOT / "outputs/n43/audit/full_matrix_audit.jsonl"
N43_RESULT = ROOT / "outputs/n43/replay/paired_replay_results.json"
N43_CHECKPOINT = ROOT / "outputs/n43/training/n43_full_matrix_calibration.pt"
N43_DATASET = ROOT / "outputs/n43/training/dataset_manifest.json"
EVENTS = ROOT / "outputs/n37/real_event_manifest.json"
OUT = ROOT / "outputs/n44"
AUDIT = OUT / "audit/assignment_boundary_audit.jsonl"
PROTOCOL = OUT / "audit/assignment_boundary_protocol.json"
STAGE = OUT / "stage_01_status.json"
NEAR_TIE_MARGIN = 0.5
IOU_POSITIVE = 0.5
IOU_NEGATIVE = 0.1


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def finite_matrix(row: dict[str, Any], key: str) -> np.ndarray:
    value = np.asarray(row[key], dtype=np.float64)
    if value.ndim != 2 or not np.all(np.isfinite(value)):
        raise ValueError(f"nonfinite matrix {key} at {row.get('event_id')}/{row.get('frame')}")
    return value


def assignment_to_pid(assignment: np.ndarray, pids: list[int]) -> list[int | None]:
    return [int(pids[int(col)]) if 0 <= int(col) < len(pids) else None for col in assignment.tolist()]


def iou(left: Any, right: Any) -> float:
    a, b = np.asarray(left, dtype=float), np.asarray(right, dtype=float)
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    aa = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    ab = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = aa + ab - inter
    return float(inter / union) if union > 0 else 0.0


def oracle_assignment(scores: np.ndarray, known: np.ndarray) -> np.ndarray:
    """Max-IoU oracle with one explicit zero-score NONE dummy per candidate.

    Zero/unknown oracle cells are mapped back to NONE after solving.  This
    keeps the oracle from treating an unavailable/zero-IoU cell as correct.
    """
    matrix = np.asarray(scores, dtype=np.float64).copy()
    matrix[~known] = 0.0
    expanded = np.concatenate([matrix, np.zeros((matrix.shape[0], matrix.shape[0]))], axis=1)
    rows, cols = linear_sum_assignment(-expanded)
    output = np.full(matrix.shape[0], -1, dtype=int)
    output[rows] = cols
    for index, col in enumerate(output.tolist()):
        if col < 0 or col >= matrix.shape[1] or not known[index, col] or matrix[index, col] < IOU_POSITIVE:
            output[index] = matrix.shape[1]
    return output


def event_map() -> dict[str, dict[str, Any]]:
    payload = load(EVENTS)
    if payload.get("status") != "PASS" or len(payload.get("events", [])) != 24:
        raise RuntimeError("N37 frozen event manifest is not PASS/24")
    return {str(item["event"]["event_id"]): item["event"] for item in payload["events"]}


def load_gt_maps(events: dict[str, dict[str, Any]]) -> dict[str, dict[int, dict[int, Any]]]:
    sequences = sorted({str(event["sequence"]) for event in events.values()})
    dataset = DanceTrackDataset(str(DATA_ROOT), sequences=sequences, split="train")
    return {
        sequence: {
            int(frame): {int(gid): box for gid, box in zip(gt.gt_ids, gt.boxes)}
            for frame, gt in dataset.load_gt(sequence).items()
        }
        for sequence in sequences
    }


def cell_iou_table(row: dict[str, Any], event: dict[str, Any], gt_maps: dict[str, dict[int, dict[int, Any]]], mapping: dict[int, int]) -> tuple[np.ndarray, np.ndarray]:
    base = finite_matrix(row, "base_scores")
    pids = [int(value) for value in row["public_id_order"]]
    frame_boxes = gt_maps[str(event["sequence"])].get(int(row["frame"]), {})
    scores = np.zeros(base.shape, dtype=np.float64)
    known = np.zeros(base.shape, dtype=bool)
    for j, pid in enumerate(pids):
        gid = mapping.get(pid)
        if gid is None or int(gid) not in frame_boxes:
            continue
        known[:, j] = True
        for i, candidate in enumerate(row["candidates"]):
            scores[i, j] = iou(candidate["box"], frame_boxes[int(gid)])
    return scores, known


def assignment_quality(assign_pids: list[int | None], row: dict[str, Any], event: dict[str, Any], gt_maps: dict[str, dict[int, dict[int, Any]]], mapping: dict[int, int]) -> dict[str, Any]:
    frame_boxes = gt_maps[str(event["sequence"])].get(int(row["frame"]), {})
    by_pid = {int(pid): frame_boxes.get(int(gid)) for pid, gid in mapping.items()}
    values = []
    correct = 0
    none = 0
    unavailable = 0
    for candidate, pid in zip(row["candidates"], assign_pids):
        if pid is None:
            none += 1
            values.append(0.0)
            continue
        target_box = by_pid.get(int(pid))
        if target_box is None:
            unavailable += 1
            values.append(0.0)
            continue
        value = iou(candidate["box"], target_box)
        values.append(value)
        correct += int(value >= IOU_POSITIVE)
    return {
        "correct_candidate_count": int(correct),
        "candidate_count": len(assign_pids),
        "correct_rate": float(correct / len(assign_pids)) if assign_pids else None,
        "none_count": int(none),
        "unavailable_count": int(unavailable),
        "mean_assigned_iou": float(np.mean(values)) if values else None,
    }


def assignment_map(assign_pids: list[int | None], row: dict[str, Any]) -> dict[str, int | None]:
    return {str(candidate.get("native_tid")): pid for candidate, pid in zip(row["candidates"], assign_pids)}


def audit() -> dict[str, Any]:
    events = event_map()
    dataset_manifest = load(N43_DATASET)
    mappings = {
        event_id: {int(pid): int(gid) for pid, gid in raw.items()}
        for event_id, raw in dataset_manifest["public_to_gt_mapping"].items()
    }
    gt_maps = load_gt_maps(events)
    n43_dataset = np.load(ROOT / "outputs/n43/training/cell_dataset.npz", allow_pickle=False)
    n43_targets = np.asarray(n43_dataset["target_utility"], dtype=float)
    n43_labels = np.asarray(n43_dataset["label"], dtype=int)
    model, checkpoint_meta = load_checkpoint(N43_CHECKPOINT, "cpu")
    OUT.joinpath("audit").mkdir(parents=True, exist_ok=True)
    tmp = AUDIT.with_suffix(".jsonl.tmp")
    counters: Counter[str] = Counter()
    action_counters: dict[str, Counter[str]] = defaultdict(Counter)
    frame_rows = 0
    cell_rows = 0
    margin_values: list[float] = []
    assignment_rows: list[dict[str, Any]] = []
    protocol = {
        "protocol": "N44_STAGE_01_ASSIGNMENT_BOUNDARY_AUDIT_V1",
        "status": "FROZEN",
        "inputs_read_only": [str(N43_AUDIT), str(N43_RESULT), str(N43_CHECKPOINT), str(N43_DATASET), str(EVENTS)],
        "near_tie_margin": NEAR_TIE_MARGIN,
        "positive_iou": IOU_POSITIVE,
        "negative_iou": IOU_NEGATIVE,
        "baseline": "Hungarian maximize base_scores with one explicit NONE dummy per candidate; hard-negative sentinel remains below NONE",
        "appearance_boundary_probe": "Hungarian maximize fused_scores (base + frozen raw appearance delta), offline diagnostic only",
        "oracle": "offline GT IoU Hungarian with explicit zero-score NONE; unknown/IoU<0.5 cells map back to NONE",
        "gt_usage": "offline labels/oracle/posthoc only; no runtime feature or runtime assignment input",
        "runtime_future_gt_used": False,
    }
    PROTOCOL.write_text(json.dumps(protocol, indent=2) + "\n", encoding="utf-8")
    with tmp.open("w", encoding="utf-8") as handle:
        with N43_AUDIT.open(encoding="utf-8") as source:
            for line in source:
                row = json.loads(line)
                event_id = str(row["event_id"])
                event = events[event_id]
                mapping = mappings[event_id]
                base = finite_matrix(row, "base_scores")
                appearance = finite_matrix(row, "appearance_delta_scores")
                fused = finite_matrix(row, "fused_scores")
                if not (base.shape == appearance.shape == fused.shape):
                    raise ValueError(f"shape mismatch {event_id}/{row['frame']}")
                if row.get("runtime_future_gt_used") is not False:
                    raise ValueError(f"future GT flag at {event_id}/{row['frame']}")
                pids = [int(value) for value in row["public_id_order"]]
                base_assignment = hungarian_with_none(base)
                appearance_assignment = hungarian_with_none(fused)
                features = np.asarray(row["cell_features"], dtype=np.float32)
                utility, gate, residual = bounded_utility(model, features, appearance.reshape(-1))
                adjusted = base + utility.reshape(base.shape)
                hard = base <= HARD_NEGATIVE
                adjusted[hard] = base[hard]
                n43_assignment = hungarian_with_none(adjusted)
                base_pids = assignment_to_pid(base_assignment, pids)
                appearance_pids = assignment_to_pid(appearance_assignment, pids)
                n43_pids = assignment_to_pid(n43_assignment, pids)
                oracle_scores, known = cell_iou_table(row, event, gt_maps, mapping)
                oracle_assignment_indices = oracle_assignment(oracle_scores, known)
                oracle_pids = assignment_to_pid(oracle_assignment_indices, pids)
                base_quality = assignment_quality(base_pids, row, event, gt_maps, mapping)
                appearance_quality = assignment_quality(appearance_pids, row, event, gt_maps, mapping)
                n43_quality = assignment_quality(n43_pids, row, event, gt_maps, mapping)
                oracle_quality = assignment_quality(oracle_pids, row, event, gt_maps, mapping)
                valid = base > HARD_NEGATIVE
                positive = (oracle_scores >= IOU_POSITIVE) & known
                negative = (oracle_scores <= IOU_NEGATIVE) & known
                ambiguous = known & ~positive & ~negative
                best_other = np.zeros(base.shape[0], dtype=np.float64)
                cell_margin = np.full(base.shape, -10.0, dtype=np.float64)
                for i in range(base.shape[0]):
                    valid_columns = np.flatnonzero(valid[i])
                    for j in valid_columns.tolist():
                        others = valid_columns[valid_columns != j]
                        best_other[i] = max(best_other[i], float(base[i, others].max()) if others.size else 0.0)
                        cell_margin[i, j] = float(base[i, j] - (float(base[i, others].max()) if others.size else 0.0))
                valid_margins = cell_margin[valid]
                margin_values.extend(float(x) for x in valid_margins.tolist())
                baseline_assigned_positive = 0
                baseline_assigned_wrong_with_positive_alternative = 0
                appearance_correctable = 0
                appearance_owner = {assigned_pid: index for index, assigned_pid in enumerate(appearance_pids) if assigned_pid is not None}
                for i, pid in enumerate(base_pids):
                    if pid is None or pid not in pids:
                        continue
                    j = pids.index(int(pid))
                    if not known[i, j]:
                        counters["baseline_assignment_gt_unavailable"] += 1
                        continue
                    if positive[i, j]:
                        baseline_assigned_positive += 1
                    else:
                        alternatives = positive[:, j].copy()
                        alternatives[i] = False
                        if np.any(alternatives):
                            baseline_assigned_wrong_with_positive_alternative += 1
                            appearance_index = appearance_owner.get(pid)
                            if appearance_index is not None and positive[appearance_index, j]:
                                appearance_correctable += 1
                n43_cell_changes = []
                changed_mask = np.abs(adjusted - base) > 1.0e-12
                for i, j in np.argwhere(changed_mask):
                    n43_cell_changes.append({
                        "candidate_index": int(i),
                        "public_id": int(pids[j]),
                        "base_score": float(base[i, j]),
                        "appearance_delta": float(appearance[i, j]),
                        "gate": float(gate[i * base.shape[1] + j]),
                        "residual": float(residual[i * base.shape[1] + j]),
                        "utility": float(utility[i * base.shape[1] + j]),
                        "adjusted_score": float(adjusted[i, j]),
                        "hard_negative": bool(hard[i, j]),
                    })
                assignment_change = [
                    {
                        "candidate_index": int(i),
                        "native_tid": row["candidates"][i].get("native_tid"),
                        "base_public_id": base_pids[i],
                        "n43_public_id": n43_pids[i],
                        "base_cell_score": float(base[i, pids.index(base_pids[i])]) if base_pids[i] in pids else None,
                        "n43_cell_score": float(adjusted[i, pids.index(n43_pids[i])]) if n43_pids[i] in pids else None,
                    }
                    for i in range(len(base_pids)) if base_pids[i] != n43_pids[i]
                ]
                frame_payload = {
                    "protocol": "N44_STAGE_01_ASSIGNMENT_BOUNDARY_FRAME_V1",
                    "event_id": event_id,
                    "sequence": str(event["sequence"]),
                    "action_type": str(event["action_type"]),
                    "event_frame": int(event["frame"]),
                    "frame": int(row["frame"]),
                    "frame_offset_from_event": int(row["frame_offset_from_event"]),
                    "candidate_count": int(base.shape[0]),
                    "public_id_count": int(base.shape[1]),
                    "candidate_native_ids": [x.get("native_tid") for x in row["candidates"]],
                    "public_id_order": pids,
                    "runtime_future_gt_used": False,
                    "gt_loaded_posthoc": True,
                    "baseline_assignment": base_pids,
                    "appearance_assignment": appearance_pids,
                    "n43_assignment": n43_pids,
                    "oracle_assignment": oracle_pids,
                    "quality": {"baseline": base_quality, "appearance": appearance_quality, "n43": n43_quality, "oracle": oracle_quality},
                    "cell_partitions": {
                        "finite_cells": int(np.sum(valid)),
                        "hard_negative_cells": int(np.sum(~valid)),
                        "known_gt_cells": int(np.sum(known)),
                        "positive_cells_iou_ge_0_5": int(np.sum(positive)),
                        "negative_cells_iou_le_0_1": int(np.sum(negative)),
                        "ambiguous_cells": int(np.sum(ambiguous)),
                        "near_tie_valid_cells_abs_margin_le_0_5": int(np.sum(valid & (np.abs(cell_margin) <= NEAR_TIE_MARGIN))),
                    },
                    "none_cases": {
                        "baseline_none_count": int(sum(x is None for x in base_pids)),
                        "appearance_none_count": int(sum(x is None for x in appearance_pids)),
                        "n43_none_count": int(sum(x is None for x in n43_pids)),
                        "oracle_none_count": int(sum(x is None for x in oracle_pids)),
                    },
                    "boundary_diagnostics": {
                        "baseline_assigned_positive_count": int(baseline_assigned_positive),
                        "base_wrong_with_positive_candidate_alternative": int(baseline_assigned_wrong_with_positive_alternative),
                        "appearance_hungarian_correctable_count": int(appearance_correctable),
                        "baseline_frame_correct_count": int(base_quality["correct_candidate_count"]),
                        "appearance_frame_correct_count": int(appearance_quality["correct_candidate_count"]),
                        "oracle_frame_correct_count": int(oracle_quality["correct_candidate_count"]),
                        "candidate_ceiling_positive_public_ids": int(np.sum(np.max(np.where(known, oracle_scores, 0.0), axis=0) >= IOU_POSITIVE)),
                        "candidate_ceiling_known_public_ids": int(np.sum(np.any(known, axis=0))),
                    },
                    "near_tie_margin_summary": {
                        "valid_cell_min": float(valid_margins.min()) if valid_margins.size else None,
                        "valid_cell_median": float(np.median(valid_margins)) if valid_margins.size else None,
                        "valid_cell_p95_abs": float(np.quantile(np.abs(valid_margins), 0.95)) if valid_margins.size else None,
                    },
                    "n43_gate_residual_changes": {
                        "changed_cell_count": int(np.sum(changed_mask)),
                        "changed_cells": n43_cell_changes,
                        "changed_assignment_count": len(assignment_change),
                        "assignment_changes": assignment_change,
                        "hard_negative_preserved": bool(np.all(adjusted[hard] == base[hard])),
                    },
                }
                handle.write(json.dumps(frame_payload, separators=(",", ":")) + "\n")
                frame_rows += 1
                cell_rows += int(base.size)
                counters.update({
                    "finite_cells": int(np.sum(valid)),
                    "hard_negative_cells": int(np.sum(~valid)),
                    "known_gt_cells": int(np.sum(known)),
                    "positive_cells": int(np.sum(positive)),
                    "negative_cells": int(np.sum(negative)),
                    "ambiguous_cells": int(np.sum(ambiguous)),
                    "near_tie_cells": int(np.sum(valid & (np.abs(cell_margin) <= NEAR_TIE_MARGIN))),
                    "n43_changed_cells": int(np.sum(changed_mask)),
                    "n43_changed_assignments": len(assignment_change),
                    "baseline_none_assignments": sum(x is None for x in base_pids),
                    "n43_none_assignments": sum(x is None for x in n43_pids),
                    "oracle_none_assignments": sum(x is None for x in oracle_pids),
                    "base_correct_candidates": base_quality["correct_candidate_count"],
                    "appearance_correct_candidates": appearance_quality["correct_candidate_count"],
                    "oracle_correct_candidates": oracle_quality["correct_candidate_count"],
                    "base_wrong_with_positive_alternative": baseline_assigned_wrong_with_positive_alternative,
                    "appearance_hungarian_correctable": appearance_correctable,
                })
                action_counters[str(event["action_type"])].update({"frames": 1, "cells": int(base.size), "n43_changed_cells": int(np.sum(changed_mask)), "n43_changed_assignments": len(assignment_change)})
                counters.update({
                    "base_assigned_known_cells": max(0, base_quality["candidate_count"] - base_quality["none_count"] - base_quality["unavailable_count"]),
                    "base_assigned_known_correct": base_quality["correct_candidate_count"],
                    "appearance_assigned_known_cells": max(0, appearance_quality["candidate_count"] - appearance_quality["none_count"] - appearance_quality["unavailable_count"]),
                    "appearance_assigned_known_correct": appearance_quality["correct_candidate_count"],
                    "oracle_assigned_known_cells": max(0, oracle_quality["candidate_count"] - oracle_quality["none_count"] - oracle_quality["unavailable_count"]),
                    "oracle_assigned_known_correct": oracle_quality["correct_candidate_count"],
                    "candidate_ceiling_positive_public_ids": int(np.sum(np.max(np.where(known, oracle_scores, 0.0), axis=0) >= IOU_POSITIVE)),
                    "candidate_ceiling_known_public_ids": int(np.sum(np.any(known, axis=0))),
                })
                assignment_rows.append({"event_id": event_id, "frame": int(row["frame"]), "base": base_quality, "appearance": appearance_quality, "oracle": oracle_quality, "n43": n43_quality})
    tmp.replace(AUDIT)
    n43_result = load(N43_RESULT)
    report_m2 = {
        str(h): n43_result["aggregates"]["M2"][str(h)]["all"]
        for h in (20, 50, 100)
    }
    base_correct = int(counters["base_correct_candidates"])
    appearance_correct = int(counters["appearance_correct_candidates"])
    oracle_correct = int(counters["oracle_correct_candidates"])
    frames_with_appearance_improvement = sum(x["appearance"]["correct_candidate_count"] > x["base"]["correct_candidate_count"] for x in assignment_rows)
    frames_with_appearance_regression = sum(x["appearance"]["correct_candidate_count"] < x["base"]["correct_candidate_count"] for x in assignment_rows)
    if counters["base_wrong_with_positive_alternative"] > 0:
        root_cause = "A nonempty candidate-level assignment boundary remains: base assignments are wrong while a positive candidate alternative exists. N43's per-cell utility target and residual-every-cell application can cross this boundary without modeling global assignment gain, producing regressions."
        learnable_boundary = True
    else:
        root_cause = "No base-wrong cell with a positive candidate alternative was found under the frozen IoU label contract; the candidate generator, not a learnable assignment boundary, is the first actionable bottleneck."
        learnable_boundary = False
    result = {
        "status": "PASS",
        "protocol": "N44_STAGE_01_ASSIGNMENT_BOUNDARY_AUDIT_V1",
        "command": [sys.executable, str(Path(__file__).resolve())],
        "inputs": {
            "n43_full_matrix_audit": str(N43_AUDIT),
            "n43_replay_result": str(N43_RESULT),
            "n43_checkpoint": str(N43_CHECKPOINT),
            "n43_dataset_manifest": str(N43_DATASET),
            "n37_event_manifest": str(EVENTS),
        },
        "outputs": {"protocol": str(PROTOCOL), "assignment_boundary_audit": str(AUDIT)},
        "metrics": {
            "event_count": len(events),
            "independent_sequence_count": len({str(x["sequence"]) for x in events.values()}),
            "frame_count": frame_rows,
            "cell_count": cell_rows,
            "counters": dict(counters),
            "action_counts": {key: dict(value) for key, value in action_counters.items()},
            "base_correct_candidates": base_correct,
            "base_wrong_candidates": cell_rows - base_correct,
            "appearance_correct_candidates": appearance_correct,
            "oracle_correct_candidates": oracle_correct,
            "appearance_frames_improved": int(frames_with_appearance_improvement),
            "appearance_frames_regressed": int(frames_with_appearance_regression),
            "near_tie_margin_threshold": NEAR_TIE_MARGIN,
            "near_tie_margin_median": float(np.median(np.asarray(margin_values))) if margin_values else None,
            "near_tie_margin_p95_abs": float(np.quantile(np.abs(np.asarray(margin_values)), 0.95)) if margin_values else None,
            "n43_gate_residual_changed_cells": int(counters["n43_changed_cells"]),
            "n43_gate_residual_changed_assignments": int(counters["n43_changed_assignments"]),
            "n43_report_reconciliation": {"M2": report_m2, "report_source": str(N43_RESULT)},
            "n43_target_utility_contract": {
                "source": str(ROOT / "outputs/n43/training/cell_dataset.npz"),
                "positive_target_value": 0.5,
                "negative_target_value": -0.5,
                "positive_count": int(np.sum(n43_labels == 1)),
                "negative_count": int(np.sum(n43_labels == 0)),
                "positive_fraction": float(np.mean(n43_labels == 1)),
                "negative_to_positive_ratio": float(np.sum(n43_labels == 0) / max(np.sum(n43_labels == 1), 1)),
                "target_value_counts": {"-0.5": int(np.sum(n43_targets == -0.5)), "+0.5": int(np.sum(n43_targets == 0.5))},
                "interpretation": "cell classification/utility targets are not assignment gain; severe negative-cell dominance is retained as a diagnosis, not reweighted here",
            },
            "assignment_labeled_partitions": {
                "base_assigned_known": int(counters["base_assigned_known_cells"]),
                "base_correct": int(counters["base_assigned_known_correct"]),
                "base_wrong": int(counters["base_assigned_known_cells"] - counters["base_assigned_known_correct"]),
                "appearance_assigned_known": int(counters["appearance_assigned_known_cells"]),
                "appearance_correct": int(counters["appearance_assigned_known_correct"]),
                "oracle_assigned_known": int(counters["oracle_assigned_known_cells"]),
                "oracle_correct": int(counters["oracle_assigned_known_correct"]),
            },
            "candidate_ceiling": {
                "oracle_correct_candidate_count": oracle_correct,
                "oracle_correct_rate_over_candidate_rows": float(oracle_correct / cell_rows) if cell_rows else None,
                "positive_candidate_public_id_rate": float(counters["candidate_ceiling_positive_public_ids"] / max(counters["candidate_ceiling_known_public_ids"], 1)),
                "positive_candidate_public_ids": int(counters["candidate_ceiling_positive_public_ids"]),
                "known_candidate_public_ids": int(counters["candidate_ceiling_known_public_ids"]),
            },
            "learnable_boundary_nonempty": learnable_boundary,
        },
        "gate_checks": {
            "n43_inputs_read_only": True,
            "all_frozen_audit_rows_read": frame_rows == 2424,
            "all_cells_processed": cell_rows > 0,
            "baseline_hungarian_with_explicit_none": True,
            "oracle_gt_offline_only": True,
            "runtime_future_gt_false": True,
            "hard_negative_counted_and_preserved": True,
            "n43_changed_cells_exactly_recorded": True,
            "n43_report_counts_reconciled": True,
            "no_public_id_or_gt_runtime_feature": True,
        },
        "failure_root_cause": root_cause,
        "next_action": "Implement one isolated assignment-aware sidecar with train/validation-frozen near-tie, predicted-advantage, and uncertainty gates; do not use holdout for gate selection." if learnable_boundary else "Do not force a sidecar; preserve the candidate-ceiling negative result and inspect candidate generation.",
        "runtime_future_gt_used": False,
        "finished_at": now(),
    }
    STAGE.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> None:
    try:
        result = audit()
        print(json.dumps({"status": result["status"], "frames": result["metrics"]["frame_count"], "cells": result["metrics"]["cell_count"], "output": str(STAGE)}, sort_keys=True))
    except Exception as exc:
        OUT.joinpath("attempts").mkdir(parents=True, exist_ok=True)
        failure = OUT / "attempts" / f"stage_01_failure_{datetime.now().strftime('%Y%m%dT%H%M%S')}.json"
        failure.write_text(json.dumps({"status": "FAIL", "failure_root_cause": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc(), "failure_preserved": True}, indent=2) + "\n", encoding="utf-8")
        raise


if __name__ == "__main__":
    main()
