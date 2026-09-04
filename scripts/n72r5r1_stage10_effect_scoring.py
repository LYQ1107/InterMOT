#!/usr/bin/env python3
"""N72R5R1 Stage 10: posthoc effect scoring after a sealed Stage08/09 run.

GT is opened only after the new public-runtime validator has passed.  The
runtime rows themselves contain no dataset identity or future GT.  All
metrics are calculated from the public sidecar plus an isolated simulation
map, and sequence-cluster bootstrap follows the frozen N72R5 definition.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sam3_intermot.evaluation.interaction_effect_metrics import (  # noqa: E402
    BOOTSTRAP_REPETITIONS,
    BOOTSTRAP_SEED,
    metric_record,
    sequence_cluster_bootstrap,
)
from sam3_intermot.association.branch_public_replay import (  # noqa: E402
    BRANCHES,
    HORIZON,
    IOU_THRESHOLD,
    atomic_json,
    box_iou,
    now_utc,
    read_json,
    read_jsonl,
    sha256_file,
)

OUT = Path(os.environ.get("N72R5R1_RUN_ROOT", str(ROOT / "outputs/N72R5R1")))
VALIDATION = OUT / "stage09_validation.json"
RUNTIME_MANIFEST = OUT / "stage08_runtime_manifest.json"
EVENT_MANIFEST = ROOT / "outputs/N72R5/mechanism_rounds/round_06_event_policy/real_event_manifest.json"
EFFECT = OUT / "stage10_effect_scoring.json"
STATUS = OUT / "stage_10_status.json"
GT_ROOT = Path("/data1/LWR/vranlee/DATASETS/JDE/dancetrack/train")

PAIRS = {
    "B1_MINUS_B0": ("B1_SPATIAL_CORRECTION_ONLY", "B0_NO_INTERVENTION"),
    "B2_MINUS_B1": ("B2_SPATIAL_CORRECTION_PLUS_IMAGE_RECOVERY", "B1_SPATIAL_CORRECTION_ONLY"),
    "B3_MINUS_B1": ("B3_SPATIAL_CORRECTION_PLUS_TVC", "B1_SPATIAL_CORRECTION_ONLY"),
    "B4_MINUS_B2": ("B4_SPATIAL_CORRECTION_PLUS_RECOVERY_PLUS_TVC", "B2_SPATIAL_CORRECTION_PLUS_IMAGE_RECOVERY"),
    "B4_MINUS_B0": ("B4_SPATIAL_CORRECTION_PLUS_RECOVERY_PLUS_TVC", "B0_NO_INTERVENTION"),
}
HORIZONS = (20, 50, 100)


def _gt(sequence: str) -> dict[int, dict[int, list[float]]]:
    path = GT_ROOT / sequence / "gt" / "gt.txt"
    if not path.is_file():
        raise FileNotFoundError(path)
    result: dict[int, dict[int, list[float]]] = defaultdict(dict)
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            parts = [item.strip() for item in line.split(",")]
            if len(parts) < 6:
                raise ValueError(f"malformed GT row: {path}:{line_number}")
            frame = int(parts[0]) - 1
            gt_id = int(parts[1])
            x, y, width, height = (float(item) for item in parts[2:6])
            result[frame][gt_id] = [x, y, x + width, y + height]
    return result


def _best_candidate(row: Mapping[str, Any], gt_box: Sequence[float]) -> tuple[float, int | None, str | None]:
    best = (0.0, None, None)
    best_index: int | None = None
    for candidate in row.get("candidate_rows", []):
        score = float(box_iou(candidate.get("box_xyxy"), gt_box))
        public = None if candidate.get("public_id") is None else int(candidate["public_id"])
        uid = None if candidate.get("candidate_uid") is None else str(candidate["candidate_uid"])
        candidate_index = int(candidate.get("candidate_index", 0))
        if score > best[0] or (score == best[0] and (best_index is None or candidate_index < best_index)):
            best = (score, public, uid)
            best_index = candidate_index
    return best


def _frame_target(
    row: Mapping[str, Any],
    gt_box: Sequence[float] | None,
    target_public: int | None,
) -> dict[str, Any]:
    if gt_box is None:
        return {
            "visible": False,
            "candidate_present": False,
            "missing": False,
            "wrong_reassociation": False,
            "correct": False,
            "iou": None,
            "assigned_public_id": None,
            "candidate_uid": None,
        }
    iou, public, uid = _best_candidate(row, gt_box)
    present = bool(iou >= IOU_THRESHOLD)
    correct = bool(present and target_public is not None and public == int(target_public))
    return {
        "visible": True,
        "candidate_present": present,
        "missing": not present,
        "wrong_reassociation": bool(present and not correct),
        "correct": correct,
        "iou": float(iou),
        "assigned_public_id": public,
        "candidate_uid": uid,
    }


def _rate(values: Sequence[float]) -> float | None:
    return None if not values else float(np.mean(np.asarray(values, dtype=float)))


def _branch_metrics(
    rows: Mapping[int, Mapping[str, Any]],
    gt_frames: Mapping[int, Mapping[int, Sequence[float]]],
    *,
    target_gt: int,
    target_public: int | None,
    event_frame: int,
    horizon: int,
) -> dict[str, Any]:
    frame_rows: list[dict[str, Any]] = []
    last_assigned: int | None = None
    id_switches = 0
    for frame in range(event_frame + 1, event_frame + int(horizon) + 1):
        row = rows.get(frame)
        if row is None:
            raise RuntimeError(f"missing sidecar frame {frame}")
        item = _frame_target(row, gt_frames.get(frame, {}).get(int(target_gt)), target_public)
        if item["visible"] and item["assigned_public_id"] is not None:
            current = int(item["assigned_public_id"])
            if last_assigned is not None and current != last_assigned:
                id_switches += 1
            last_assigned = current
        frame_rows.append({"frame": frame, **item})
    visible = [item for item in frame_rows if item["visible"]]
    present = [item for item in visible if item["candidate_present"]]
    ious = [float(item["iou"]) for item in visible if item["iou"] is not None]
    wrong_or_missing = [item for item in visible if item["wrong_reassociation"] or item["missing"]]
    recorrection = [
        index
        for index, item in enumerate(frame_rows)
        if item["visible"] and (item["wrong_reassociation"] or item["missing"])
        and (index == 0 or not (frame_rows[index - 1]["wrong_reassociation"] or frame_rows[index - 1]["missing"]))
    ]
    return {
        "horizon": int(horizon),
        "visible_frame_count": len(visible),
        "candidate_recall": _rate([float(item["candidate_present"]) for item in visible]),
        "missing_rate": _rate([float(item["missing"]) for item in visible]),
        "wrong_reassociation_rate": _rate([float(item["wrong_reassociation"]) for item in visible]),
        "identity_accuracy": _rate([float(item["correct"]) for item in visible]),
        "identity_error_rate": _rate([float(not item["correct"]) for item in visible]),
        "mean_iou": _rate(ious),
        "id_switch_count": int(id_switches),
        "recorrection_opportunity_count": int(len(recorrection)),
        "frame_rows": frame_rows,
    }


def _protected_regression(
    baseline_rows: Mapping[int, Mapping[str, Any]],
    treatment_rows: Mapping[int, Mapping[str, Any]],
    gt_frames: Mapping[int, Mapping[int, Sequence[float]]],
    gt_to_public: Mapping[int, int],
    *,
    protected_ids: Sequence[int],
    event_frame: int,
    horizon: int,
) -> dict[str, Any]:
    baseline_correct = 0
    regression = 0
    total = 0
    for frame in range(event_frame + 1, event_frame + int(horizon) + 1):
        for gt_id in protected_ids:
            box = gt_frames.get(frame, {}).get(int(gt_id))
            if box is None or int(gt_id) not in gt_to_public:
                continue
            b = _frame_target(baseline_rows[frame], box, int(gt_to_public[gt_id]))["correct"]
            t = _frame_target(treatment_rows[frame], box, int(gt_to_public[gt_id]))["correct"]
            total += 1
            baseline_correct += int(b)
            regression += int(b and not t)
    return {
        "protected_visible_evaluations": int(total),
        "baseline_correct_count": int(baseline_correct),
        "regression_count": int(regression),
        "regression_rate": None if total == 0 else float(regression / total),
    }


def _load_sidecar(path: str) -> dict[int, dict[str, Any]]:
    rows = read_jsonl(Path(path))
    return {int(row["frame"]): row for row in rows}


def _not_authorized(reason: str, *, validation: Mapping[str, Any] | None = None) -> int:
    payload = {
        "schema_version": "N72R5R1_STAGE10_STATUS_V1",
        "stage": "10_POSTHOC_FUTURE_EFFECT_SCORING",
        "status": "NOT_AUTHORIZED_STAGE09_BLOCKED",
        "effect_evaluated": False,
        "posthoc_gt_opened": False,
        "reason": reason,
        "stage09_validation_status": None if validation is None else validation.get("status"),
        "runtime_future_gt_used": False,
        "created_at_utc": now_utc(),
    }
    atomic_json(STATUS, payload)
    print(json.dumps(payload, ensure_ascii=False))
    return 1


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--allow-nonstrict", action="store_true", help="debug only; still records non-authorized effect")
    args = parser.parse_args()
    if not VALIDATION.is_file() or not RUNTIME_MANIFEST.is_file():
        return _not_authorized("Stage08/Stage09 artifact is missing")
    validation = read_json(VALIDATION)
    if validation.get("strict_pass") is not True:
        return _not_authorized("Stage09 strict public-runtime validation did not pass", validation=validation)
    manifest = read_json(RUNTIME_MANIFEST)
    if manifest.get("status") != "PASS_N72R5R1_EXACT_PUBLIC_ASSOCIATION":
        return _not_authorized("Stage08 full public-association manifest did not pass", validation=validation)

    # Only now is GT opened.  It remains a posthoc-only object and is never
    # passed into the runtime/solver.
    event_policy = read_json(EVENT_MANIFEST)
    events = {str(item["event_id"]): dict(item) for item in event_policy.get("events", [])}
    result_by_event = {str(item["event_id"]): item for item in manifest.get("events", [])}
    all_metrics: list[dict[str, Any]] = []
    pair_records: dict[str, list[dict[str, Any]]] = defaultdict(list)
    protected_records: dict[str, list[dict[str, Any]]] = defaultdict(list)
    unavailable: list[dict[str, Any]] = []
    for event_id in sorted(events):
        event = events[event_id]
        private_path = OUT / "simulation_private" / event_id / "oracle_private_mapping.json"
        if not private_path.is_file() or event_id not in result_by_event:
            unavailable.append({"event_id": event_id, "reason": "missing_private_or_event_result"})
            continue
        private = read_json(private_path)
        gt_to_public = {int(k): int(v) for k, v in private.get("dataset_gt_to_public", {}).items()}
        target_gt = int(event["dataset_gt_id"])
        target_public = gt_to_public.get(target_gt)
        if target_public is None:
            unavailable.append({"event_id": event_id, "reason": "target_public_unresolved"})
            continue
        gt_frames = _gt(str(event["sequence"]))
        result = result_by_event[event_id]
        branch_rows: dict[str, dict[int, dict[str, Any]]] = {}
        branch_public: dict[str, int | None] = {}
        for branch_result in result.get("branches", []):
            branch = str(branch_result["branch"])
            branch_rows[branch] = _load_sidecar(str(branch_result["output"]))
            branch_public[branch] = None if branch_result.get("target_public_id") is None else int(branch_result["target_public_id"])
        for branch in BRANCHES:
            if branch not in branch_rows:
                unavailable.append({"event_id": event_id, "reason": f"missing_branch:{branch}"})
        for pair_name, (treatment_branch, baseline_branch) in PAIRS.items():
            if treatment_branch not in branch_rows or baseline_branch not in branch_rows:
                continue
            treatment_public = branch_public[treatment_branch] or target_public
            baseline_public = branch_public[baseline_branch] or target_public
            protected = [gt_id for gt_id in gt_to_public if gt_id not in {target_gt, event.get("other_dataset_gt_id")}]
            for horizon in HORIZONS:
                baseline = _branch_metrics(branch_rows[baseline_branch], gt_frames, target_gt=target_gt, target_public=baseline_public, event_frame=int(event["event_frame"]), horizon=horizon)
                treatment = _branch_metrics(branch_rows[treatment_branch], gt_frames, target_gt=target_gt, target_public=treatment_public, event_frame=int(event["event_frame"]), horizon=horizon)
                base_error = baseline["identity_error_rate"]
                treat_error = treatment["identity_error_rate"]
                reduction = None if base_error is None or treat_error is None else float(base_error - treat_error)
                first_base = baseline["frame_rows"][0]
                first_treat = treatment["frame_rows"][0]
                first = metric_record(
                    baseline_iou=float(first_base["iou"] or 0.0),
                    treatment_iou=float(first_treat["iou"] or 0.0),
                    baseline_correct=bool(first_base["correct"]),
                    treatment_correct=bool(first_treat["correct"]),
                    assignment_changed=first_base.get("assigned_public_id") != first_treat.get("assigned_public_id"),
                )
                record = {
                    "event_id": event_id,
                    "sequence": str(event["sequence"]),
                    "action_type": str(event["action_type"]),
                    "pair": pair_name,
                    "baseline_branch": baseline_branch,
                    "treatment_branch": treatment_branch,
                    "horizon": int(horizon),
                    "target_gt_id": target_gt,
                    "target_public_id": int(target_public),
                    "baseline": {key: value for key, value in baseline.items() if key != "frame_rows"},
                    "treatment": {key: value for key, value in treatment.items() if key != "frame_rows"},
                    "identity_error_reduction": reduction,
                    "first_future_frame": first,
                    "runtime_future_gt_used": False,
                }
                pair_records[pair_name].append(record)
                if horizon == 20:
                    protected_records[pair_name].append(_protected_regression(branch_rows[baseline_branch], branch_rows[treatment_branch], gt_frames, gt_to_public, protected_ids=protected, event_frame=int(event["event_frame"]), horizon=horizon))
                all_metrics.append(record)

    summaries: dict[str, Any] = {}
    for pair_name, records in pair_records.items():
        by_horizon: dict[str, Any] = {}
        for horizon in HORIZONS:
            subset = [record for record in records if int(record["horizon"]) == horizon and record.get("identity_error_reduction") is not None]
            values_by_sequence: dict[str, list[float]] = defaultdict(list)
            action_values: dict[str, list[float]] = defaultdict(list)
            true_correct = 0
            true_incorrect = 0
            assignment_changes = 0
            for record in subset:
                value = float(record["identity_error_reduction"])
                values_by_sequence[str(record["sequence"])].append(value)
                action_values[str(record["action_type"])].append(value)
                first = record["first_future_frame"]
                true_correct += int(first.get("true_correct_crossing", False))
                true_incorrect += int(first.get("true_incorrect_crossing", False))
                assignment_changes += int(first.get("assignment_change_type") != "UNCHANGED")
            bootstrap = sequence_cluster_bootstrap(values_by_sequence, seed=BOOTSTRAP_SEED, repetitions=BOOTSTRAP_REPETITIONS) if values_by_sequence else {"mean": None, "lower": None, "upper": None, "clusters": 0, "seed": BOOTSTRAP_SEED, "repetitions": BOOTSTRAP_REPETITIONS}
            by_horizon[str(horizon)] = {
                "event_count": len(subset),
                "sequence_cluster_bootstrap": bootstrap,
                "mean_identity_error_reduction": None if not subset else float(np.mean([record["identity_error_reduction"] for record in subset])),
                "true_correct_crossings": int(true_correct),
                "true_incorrect_crossings": int(true_incorrect),
                "assignment_changes": int(assignment_changes),
                "assignment_change_rate": None if not subset else float(assignment_changes / len(subset)),
                "by_action": {action: {"event_count": len(values), "mean_identity_error_reduction": float(np.mean(values))} for action, values in sorted(action_values.items())},
            }
        regressions = protected_records.get(pair_name, [])
        by_horizon["protected_identity_regression_h20"] = {
            "event_count": len(regressions),
            "regression_count": int(sum(item["regression_count"] for item in regressions)),
            "visible_evaluations": int(sum(item["protected_visible_evaluations"] for item in regressions)),
        }
        summaries[pair_name] = by_horizon
    primary = summaries.get("B4_MINUS_B0", {}).get("20", {})
    primary_bootstrap = primary.get("sequence_cluster_bootstrap", {})
    protected = summaries.get("B4_MINUS_B0", {}).get("protected_identity_regression_h20", {})
    gate = {
        "status": "PASS_GT_SIMULATED_FUTURE_EFFECT_CONFIRMED" if (
            bool(primary.get("true_correct_crossings", 0) > 0)
            and float(primary.get("mean_identity_error_reduction") or 0.0) > 0.0
            and float(primary_bootstrap.get("lower") if primary_bootstrap.get("lower") is not None else -math.inf) > 0.0
            and int(primary.get("true_incorrect_crossings", 0)) <= int(primary.get("true_correct_crossings", 0))
            and int(protected.get("regression_count", 0)) == 0
            and not unavailable
        ) else "FAIL_FUTURE_EFFECT",
        "primary_pair": "B4_MINUS_B0",
        "primary_horizon": 20,
        "runtime_future_gt_used": False,
        "protected_regression_count": int(protected.get("regression_count", 0)),
        "unavailable_event_count": len(unavailable),
    }
    payload = {
        "schema_version": "N72R5R1_STAGE10_EFFECT_V1",
        "status": gate["status"],
        "gate": gate,
        "summaries": summaries,
        "event_metrics": all_metrics,
        "unavailable_events": unavailable,
        "bootstrap": {"seed": BOOTSTRAP_SEED, "repetitions": BOOTSTRAP_REPETITIONS, "unit": "independent_sequence", "within_cluster_aggregation": "mean_event_value"},
        "runtime_future_gt_used": False,
        "interaction_source": "simulated_from_gt",
        "not_real_human_evidence": True,
        "stage08_runtime_manifest_sha256": sha256_file(RUNTIME_MANIFEST),
        "stage09_validation_sha256": sha256_file(VALIDATION),
        "created_at_utc": now_utc(),
    }
    atomic_json(EFFECT, payload)
    atomic_json(
        STATUS,
        {
            "schema_version": "N72R5R1_STAGE_STATUS_V1",
            "stage": "10_POSTHOC_FUTURE_EFFECT_SCORING",
            "status": gate["status"],
            "effect_evaluated": True,
            "posthoc_gt_opened": True,
            "runtime_future_gt_used": False,
            "effect_artifact": str(EFFECT),
            "created_at_utc": now_utc(),
        },
    )
    print(json.dumps({"status": gate["status"], "pairs": len(summaries), "event_metrics": len(all_metrics), "unavailable": len(unavailable)}, ensure_ascii=False))
    return 0 if gate["status"] == "PASS_GT_SIMULATED_FUTURE_EFFECT_CONFIRMED" else 1


if __name__ == "__main__":
    raise SystemExit(main())
