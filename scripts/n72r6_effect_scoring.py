#!/usr/bin/env python3
"""N72R6 posthoc C1-C0 effect scoring.

The C0/C1 replay is sealed before this module opens dataset GT.  No GT value,
dataset identity, or private mapping is passed to the runtime replay.  This
script only scores the completed public-assignment sidecars and keeps the
target-session candidate recall separate from identity correctness.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
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

from sam3_intermot.association.branch_public_replay import (  # noqa: E402
    HORIZON,
    IOU_THRESHOLD,
    box_iou,
    now_utc,
    read_json,
    read_jsonl,
    sha256_file,
    atomic_json,
)
from sam3_intermot.evaluation.interaction_effect_metrics import (  # noqa: E402
    BOOTSTRAP_REPETITIONS,
    BOOTSTRAP_SEED,
    metric_record,
    sequence_cluster_bootstrap,
)


REPLAY_ROOT = ROOT / "outputs/N72R6/public_replay/attempt_4"
REPLAY_BATCH = REPLAY_ROOT / "replay_batch_status.json"
STAGE05 = ROOT / "outputs/N72R6/stage_05_status.json"
PROTOCOL = ROOT / "outputs/N72R6/protocol.json"
STAGE08 = ROOT / "outputs/N72R5R1/controller/round_05_branch_isolation_v0/full/stage08_runtime_manifest.json"
EVENT_MANIFEST = ROOT / "outputs/N72R5/mechanism_rounds/round_06_event_policy/real_event_manifest.json"
PRIVATE_ROOT = ROOT / "outputs/N72R5R1/simulation_private"
GT_ROOT = Path("/data1/LWR/vranlee/DATASETS/JDE/dancetrack/train")
EFFECT = ROOT / "outputs/N72R6/ccam_paired_replay_results.json"
STATUS = ROOT / "outputs/N72R6/stage_06_status.json"
CONTROLLER = ROOT / "outputs/N72R6/CONTROLLER_STATUS.json"


def _resolve(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def _gt(sequence: str) -> dict[int, dict[int, list[float]]]:
    """Read GT only after ``validate_runtime_artifacts`` has returned."""

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


def _finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _effective_public(candidate: Mapping[str, Any]) -> int | None:
    """Use solver output, never the outer birth label on explicit NONE rows."""

    solver_status = str(candidate.get("solver_status", ""))
    if solver_status in {"EXPLICIT_NONE", "UNASSIGNED", "NOT_ASSIGNED"}:
        return None
    if "solver_public_id" in candidate:
        value = candidate.get("solver_public_id")
        return None if value is None else int(value)
    value = candidate.get("public_id")
    return None if value is None else int(value)


def _best_candidate(
    row: Mapping[str, Any],
    gt_box: Sequence[float],
    *,
    assigned_only: bool = False,
) -> dict[str, Any] | None:
    best: dict[str, Any] | None = None
    for candidate in row.get("candidate_rows", []):
        public_id = _effective_public(candidate)
        if assigned_only and public_id is None:
            continue
        iou = float(box_iou(candidate.get("box_xyxy"), gt_box))
        item = {
            "candidate_uid": None if candidate.get("candidate_uid") is None else str(candidate["candidate_uid"]),
            "candidate_index": int(candidate.get("candidate_index", 0)),
            "iou": iou,
            "public_id": public_id,
            "candidate_kind": candidate.get("candidate_kind"),
        }
        if best is None or iou > float(best["iou"]) or (
            iou == float(best["iou"]) and int(item["candidate_index"]) < int(best["candidate_index"])
        ):
            best = item
    return best


def _target_frame(
    row: Mapping[str, Any],
    gt_box: Sequence[float] | None,
    target_public: int,
) -> dict[str, Any]:
    if gt_box is None:
        return {
            "visible": False,
            "geometry_present": False,
            "assigned_present": False,
            "missing": False,
            "wrong_reassociation": False,
            "correct": False,
            "identity_error": False,
            "geometry_iou": None,
            "assigned_iou": None,
            "assigned_public_id": None,
            "geometry_candidate_uid": None,
            "assigned_candidate_uid": None,
            "target_session_candidate_present": False,
            "target_session_candidate_iou": None,
        }
    geometry = _best_candidate(row, gt_box)
    assigned = _best_candidate(row, gt_box, assigned_only=True)
    target_rows = [
        item for item in row.get("candidate_rows", [])
        if str(item.get("candidate_kind", "")) == "TARGET_CORRECTION_SESSION_CANDIDATE"
    ]
    target_geometry = _best_candidate({"candidate_rows": target_rows}, gt_box)
    geometry_iou = None if geometry is None else float(geometry["iou"])
    assigned_iou = None if assigned is None else float(assigned["iou"])
    assigned_present = bool(assigned is not None and assigned_iou is not None and assigned_iou >= IOU_THRESHOLD)
    geometry_present = bool(geometry is not None and geometry_iou is not None and geometry_iou >= IOU_THRESHOLD)
    assigned_public = None if assigned is None else assigned["public_id"]
    correct = bool(assigned_present and assigned_public == int(target_public))
    return {
        "visible": True,
        "geometry_present": geometry_present,
        "assigned_present": assigned_present,
        "missing": not assigned_present,
        "wrong_reassociation": bool(assigned_present and not correct),
        "correct": correct,
        "identity_error": not correct,
        "geometry_iou": geometry_iou,
        "assigned_iou": assigned_iou,
        "assigned_public_id": assigned_public,
        "geometry_candidate_uid": None if geometry is None else geometry["candidate_uid"],
        "assigned_candidate_uid": None if assigned is None else assigned["candidate_uid"],
        "target_session_candidate_present": bool(target_geometry is not None),
        "target_session_candidate_iou": None if target_geometry is None else float(target_geometry["iou"]),
    }


def _rate(values: Sequence[float]) -> float | None:
    return None if not values else float(np.mean(np.asarray(values, dtype=np.float64)))


def _branch_metrics(
    rows: Mapping[int, Mapping[str, Any]],
    gt_frames: Mapping[int, Mapping[int, Sequence[float]]],
    *,
    target_gt: int,
    target_public: int,
    event_frame: int,
    horizon: int,
) -> dict[str, Any]:
    frame_rows: list[dict[str, Any]] = []
    previous_public: int | None = None
    id_switches = 0
    for frame in range(event_frame + 1, event_frame + int(horizon) + 1):
        row = rows.get(frame)
        if row is None:
            raise RuntimeError(f"missing replay frame {frame}")
        item = _target_frame(row, gt_frames.get(frame, {}).get(int(target_gt)), target_public)
        current = item["assigned_public_id"]
        if current is not None:
            if previous_public is not None and int(current) != int(previous_public):
                id_switches += 1
            previous_public = int(current)
        frame_rows.append({"frame": int(frame), **item})
    visible = [item for item in frame_rows if item["visible"]]
    errors = [item for item in visible if item["identity_error"]]
    missing = [item for item in visible if item["missing"]]
    wrong = [item for item in visible if item["wrong_reassociation"]]
    geometry_ious = [float(item["geometry_iou"]) for item in visible if item["geometry_iou"] is not None]
    assigned_ious = [float(item["assigned_iou"]) for item in visible if item["assigned_iou"] is not None]
    recorrection = [
        index for index, item in enumerate(frame_rows)
        if item["visible"] and item["identity_error"]
        and (index == 0 or not frame_rows[index - 1]["identity_error"])
    ]
    return {
        "horizon": int(horizon),
        "visible_frame_count": len(visible),
        "geometry_candidate_recall": _rate([float(item["geometry_present"]) for item in visible]),
        "assigned_candidate_recall": _rate([float(item["assigned_present"]) for item in visible]),
        "missing_rate": _rate([float(item["missing"]) for item in visible]),
        "wrong_reassociation_rate": _rate([float(item["wrong_reassociation"]) for item in visible]),
        "identity_accuracy": _rate([float(item["correct"]) for item in visible]),
        "identity_error_rate": _rate([float(item["identity_error"]) for item in visible]),
        "mean_geometry_iou": _rate(geometry_ious),
        "mean_assigned_iou": _rate(assigned_ious),
        "id_switch_count": int(id_switches),
        "recorrection_opportunity_count": int(len(recorrection)),
        "target_session_candidate_frame_count": int(sum(item["target_session_candidate_present"] for item in visible)),
        "frame_rows": frame_rows,
    }


def _public_for_gt(candidate_rows: Mapping[int, Mapping[str, Any]], gt_box: Sequence[float], gt_to_public: Mapping[int, int], gt_id: int) -> dict[str, Any]:
    item = _target_frame(candidate_rows, gt_box, int(gt_to_public[gt_id]))
    return item


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
    treatment_correct = 0
    regression = 0
    total = 0
    for frame in range(event_frame + 1, event_frame + int(horizon) + 1):
        for gt_id in protected_ids:
            box = gt_frames.get(frame, {}).get(int(gt_id))
            if box is None or int(gt_id) not in gt_to_public:
                continue
            b = _public_for_gt(baseline_rows[frame], box, gt_to_public, int(gt_id))["correct"]
            t = _public_for_gt(treatment_rows[frame], box, gt_to_public, int(gt_id))["correct"]
            total += 1
            baseline_correct += int(b)
            treatment_correct += int(t)
            regression += int(b and not t)
    return {
        "protected_visible_evaluations": int(total),
        "baseline_correct_count": int(baseline_correct),
        "treatment_correct_count": int(treatment_correct),
        "regression_count": int(regression),
        "regression_rate": None if total == 0 else float(regression / total),
    }


def _validate_runtime_artifacts() -> tuple[dict[str, Any], dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    """Validate all sealed replay inputs before any GT file is opened."""

    stage05 = read_json(STAGE05)
    if stage05.get("status") != "PASS_TARGET_SCOPED_C0_C1_REPLAY_AUDITED":
        raise RuntimeError(f"stage05 is not PASS: {stage05.get('status')}")
    batch = read_json(REPLAY_BATCH)
    if batch.get("status") != "PASS_N72R6_C0_C1_REPLAY" or int(batch.get("completed_event_count", -1)) != 32:
        raise RuntimeError(f"replay batch is not complete: {batch.get('status')}")
    event_policy = read_json(EVENT_MANIFEST)
    events = {str(item["event_id"]): dict(item) for item in event_policy.get("events", [])}
    stage08 = read_json(STAGE08)
    eligible = {}
    for item in stage08.get("events", []):
        branches = {str(branch.get("branch")): branch for branch in item.get("branches", [])}
        b1 = branches.get("B1_SPATIAL_CORRECTION_ONLY")
        if b1 and b1.get("action_precondition_status") == "APPLIED":
            eligible[str(item["event_id"])] = dict(item)
    if len(eligible) != 32:
        raise RuntimeError(f"expected 32 eligible stage08 events, found {len(eligible)}")
    manifests: dict[str, dict[str, Any]] = {}
    for path in sorted(REPLAY_ROOT.glob("*/event_manifest.json")):
        manifest = read_json(path)
        event_id = str(manifest.get("event_id"))
        if event_id in manifests:
            raise RuntimeError(f"duplicate replay event manifest: {event_id}")
        if manifest.get("status") != "PASS_N72R6_C0_C1_EVENT_REPLAY":
            raise RuntimeError(f"replay event is not PASS: {event_id}")
        if event_id not in eligible or event_id not in events:
            raise RuntimeError(f"unexpected replay event: {event_id}")
        for flag in ("runtime_future_gt_used", "runtime_gt_read", "posthoc_gt_used", "event_frame_memory_read"):
            if manifest.get(flag) is not False:
                raise RuntimeError(f"runtime flag is not false: {event_id}:{flag}")
        for key in ("c0", "c1"):
            if int(manifest[key].get("frame_count", -1)) != HORIZON + 1:
                raise RuntimeError(f"{key} frame count mismatch: {event_id}")
            output = _resolve(str(manifest[key]["path"]))
            if not output.is_file() or sha256_file(output) != str(manifest[key]["sha256"]):
                raise RuntimeError(f"{key} hash mismatch: {event_id}")
            rows = read_jsonl(output)
            expected = list(range(int(manifest["event_frame"]), int(manifest["event_frame"]) + HORIZON + 1))
            if [int(row["frame"]) for row in rows] != expected:
                raise RuntimeError(f"{key} frame axis mismatch: {event_id}")
            for row in rows:
                for flag in ("runtime_future_gt_used", "runtime_gt_read", "posthoc_gt_used"):
                    if row.get(flag) is not False:
                        raise RuntimeError(f"row GT flag is not false: {event_id}:{key}:{row.get('frame')}:{flag}")
        manifests[event_id] = manifest
    if len(manifests) != 32:
        raise RuntimeError(f"expected 32 unique replay manifests, found {len(manifests)}")
    return stage05, events, manifests


def _mean(records: Sequence[Mapping[str, Any]], path: tuple[str, ...]) -> float | None:
    values: list[float] = []
    for record in records:
        value: Any = record
        for key in path:
            value = value.get(key) if isinstance(value, Mapping) else None
        if _finite(value):
            values.append(float(value))
    return _rate(values)


def _aggregate(records: Sequence[dict[str, Any]], protected: Sequence[dict[str, Any]]) -> dict[str, Any]:
    by_horizon: dict[str, Any] = {}
    for horizon in (20, 50, 100):
        subset = [item for item in records if int(item["horizon"]) == horizon]
        values_by_sequence: dict[str, list[float]] = defaultdict(list)
        action_values: dict[str, list[float]] = defaultdict(list)
        for item in subset:
            value = float(item["identity_error_reduction"])
            values_by_sequence[str(item["sequence"])].append(value)
            action_values[str(item["action_type"])].append(value)
        bootstrap = sequence_cluster_bootstrap(
            values_by_sequence, seed=BOOTSTRAP_SEED, repetitions=BOOTSTRAP_REPETITIONS
        ) if values_by_sequence else {
            "mean": None, "lower": None, "upper": None, "clusters": 0,
            "unit": "independent_sequence", "seed": BOOTSTRAP_SEED,
            "repetitions": BOOTSTRAP_REPETITIONS,
        }
        first = [item["first_future_frame"] for item in subset]
        changes = [item for item in first if item["assignment_change_type"] != "UNCHANGED"]
        by_horizon[str(horizon)] = {
            "event_count": len(subset),
            "mean_identity_error_reduction": _mean(subset, ("identity_error_reduction",)),
            "mean_missing_rate_reduction": _mean(subset, ("missing_rate_reduction",)),
            "mean_wrong_reassociation_rate_reduction": _mean(subset, ("wrong_reassociation_rate_reduction",)),
            "baseline_identity_error_rate": _mean(subset, ("baseline", "identity_error_rate")),
            "treatment_identity_error_rate": _mean(subset, ("treatment", "identity_error_rate")),
            "baseline_missing_rate": _mean(subset, ("baseline", "missing_rate")),
            "treatment_missing_rate": _mean(subset, ("treatment", "missing_rate")),
            "baseline_mean_geometry_iou": _mean(subset, ("baseline", "mean_geometry_iou")),
            "treatment_mean_geometry_iou": _mean(subset, ("treatment", "mean_geometry_iou")),
            "baseline_recorrection_opportunities": int(sum(item["baseline"]["recorrection_opportunity_count"] for item in subset)),
            "treatment_recorrection_opportunities": int(sum(item["treatment"]["recorrection_opportunity_count"] for item in subset)),
            "baseline_id_switches": int(sum(item["baseline"]["id_switch_count"] for item in subset)),
            "treatment_id_switches": int(sum(item["treatment"]["id_switch_count"] for item in subset)),
            "sequence_cluster_bootstrap": bootstrap,
            "assignment_changes": int(len(changes)),
            "assignment_change_rate": None if not subset else float(len(changes) / len(subset)),
            "true_correct_crossings": int(sum(item["assignment_change_type"] == "TRUE_CORRECT_CROSSING" for item in first)),
            "true_incorrect_crossings": int(sum(item["assignment_change_type"] == "TRUE_INCORRECT_CROSSING" for item in first)),
            "directional_improvements": int(sum(item["assignment_change_type"] == "DIRECTIONAL_IMPROVEMENT" for item in first)),
            "directional_regressions": int(sum(item["assignment_change_type"] == "DIRECTIONAL_REGRESSION" for item in first)),
            "by_action": {
                action: {
                    "event_count": len(values),
                    "mean_identity_error_reduction": float(np.mean(np.asarray(values, dtype=np.float64))),
                }
                for action, values in sorted(action_values.items())
            },
        }
    by_horizon["protected_identity_regression_h20"] = {
        "event_count": len(protected),
        "regression_count": int(sum(item["regression_count"] for item in protected)),
        "visible_evaluations": int(sum(item["protected_visible_evaluations"] for item in protected)),
        "baseline_correct_count": int(sum(item["baseline_correct_count"] for item in protected)),
        "treatment_correct_count": int(sum(item["treatment_correct_count"] for item in protected)),
    }
    return by_horizon


def main() -> int:
    global REPLAY_ROOT, REPLAY_BATCH, STAGE05
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=str(EFFECT))
    parser.add_argument("--replay-root", default=str(REPLAY_ROOT))
    parser.add_argument("--stage05", default=str(STAGE05))
    args = parser.parse_args()

    REPLAY_ROOT = _resolve(args.replay_root)
    REPLAY_BATCH = REPLAY_ROOT / "replay_batch_status.json"
    STAGE05 = _resolve(args.stage05)

    # This is the hard causal boundary: all runtime artifacts and their flags
    # are checked before _gt() is called below.
    stage05, events, replay = _validate_runtime_artifacts()

    # Posthoc-only inputs.  These files never enter C0/C1 replay.
    all_records: list[dict[str, Any]] = []
    protected_by_horizon: dict[str, list[dict[str, Any]]] = defaultdict(list)
    target_stream_present = 0
    target_stream_rows = 0
    target_stream_assigned = 0
    target_stream_missing_events: list[str] = []
    gt_hashes: dict[str, str] = {}
    for event_id in sorted(replay):
        event = events[event_id]
        manifest = replay[event_id]
        target_public = int(manifest["target_public_id"])
        target_gt = int(event["dataset_gt_id"])
        private_path = PRIVATE_ROOT / event_id / "oracle_private_mapping.json"
        if not private_path.is_file():
            raise FileNotFoundError(private_path)
        private = read_json(private_path)
        gt_to_public = {int(key): int(value) for key, value in private["dataset_gt_to_public"].items()}
        if gt_to_public.get(target_gt) != target_public:
            raise RuntimeError(f"posthoc target mapping mismatch: {event_id}")
        gt_frames = _gt(str(event["sequence"]))
        gt_path = GT_ROOT / str(event["sequence"]) / "gt" / "gt.txt"
        gt_hashes[str(event["sequence"])] = sha256_file(gt_path)
        c0_rows = {int(row["frame"]): row for row in read_jsonl(_resolve(str(manifest["c0"]["path"]))) }
        c1_rows = {int(row["frame"]): row for row in read_jsonl(_resolve(str(manifest["c1"]["path"]))) }
        target_future_frames = 0
        target_future_assigned = 0
        for frame in range(int(manifest["event_frame"]) + 1, int(manifest["event_frame"]) + HORIZON + 1):
            target_rows = [
                item for item in c1_rows[frame].get("candidate_rows", [])
                if str(item.get("candidate_kind", "")) == "TARGET_CORRECTION_SESSION_CANDIDATE"
            ]
            target_stream_rows += len(target_rows)
            target_future_frames += int(bool(target_rows))
            target_future_assigned += int(any(_effective_public(item) == target_public for item in target_rows))
        target_stream_present += int(target_future_frames > 0)
        target_stream_assigned += target_future_assigned
        if target_future_frames == 0:
            target_stream_missing_events.append(event_id)

        protected_ids = [
            gt_id for gt_id in sorted(gt_to_public)
            if int(gt_id) not in {target_gt, event.get("other_dataset_gt_id")}
        ]
        for horizon in (20, 50, 100):
            baseline = _branch_metrics(
                c0_rows, gt_frames, target_gt=target_gt, target_public=target_public,
                event_frame=int(manifest["event_frame"]), horizon=horizon,
            )
            treatment = _branch_metrics(
                c1_rows, gt_frames, target_gt=target_gt, target_public=target_public,
                event_frame=int(manifest["event_frame"]), horizon=horizon,
            )
            baseline_error = baseline["identity_error_rate"]
            treatment_error = treatment["identity_error_rate"]
            reduction = None if baseline_error is None or treatment_error is None else float(baseline_error - treatment_error)
            first_base = baseline["frame_rows"][0]
            first_treat = treatment["frame_rows"][0]
            first_assignment = first_base.get("assigned_public_id") != first_treat.get("assigned_public_id")
            first = metric_record(
                baseline_iou=float(first_base.get("geometry_iou") or 0.0),
                treatment_iou=float(first_treat.get("geometry_iou") or 0.0),
                baseline_correct=bool(first_base["correct"]),
                treatment_correct=bool(first_treat["correct"]),
                assignment_changed=bool(first_assignment),
            )
            first.update({
                "baseline_assigned_public_id": first_base.get("assigned_public_id"),
                "treatment_assigned_public_id": first_treat.get("assigned_public_id"),
                "baseline_geometry_iou": first_base.get("geometry_iou"),
                "treatment_geometry_iou": first_treat.get("geometry_iou"),
                "baseline_missing": first_base["missing"],
                "treatment_missing": first_treat["missing"],
            })
            all_records.append({
                "event_id": event_id,
                "sequence": str(event["sequence"]),
                "action_type": str(event["action_type"]),
                "event_frame": int(manifest["event_frame"]),
                "target_gt_id": target_gt,
                "target_public_id": target_public,
                "horizon": int(horizon),
                "baseline_branch": "C0_MAIN_BASELINE",
                "treatment_branch": "C1_TARGET_SCOPED_CORRECTION",
                "baseline": {key: value for key, value in baseline.items() if key != "frame_rows"},
                "treatment": {key: value for key, value in treatment.items() if key != "frame_rows"},
                "identity_error_reduction": reduction,
                "missing_rate_reduction": None if baseline["missing_rate"] is None or treatment["missing_rate"] is None else float(baseline["missing_rate"] - treatment["missing_rate"]),
                "wrong_reassociation_rate_reduction": None if baseline["wrong_reassociation_rate"] is None or treatment["wrong_reassociation_rate"] is None else float(baseline["wrong_reassociation_rate"] - treatment["wrong_reassociation_rate"]),
                "first_future_frame": first,
                "runtime_future_gt_used": False,
                "posthoc_gt_used": True,
            })
            if horizon == 20:
                protected_by_horizon["C1_MINUS_C0"].append(_protected_regression(
                    c0_rows, c1_rows, gt_frames, gt_to_public,
                    protected_ids=protected_ids,
                    event_frame=int(manifest["event_frame"]), horizon=horizon,
                ))

    summary = {"C1_MINUS_C0": _aggregate(all_records, protected_by_horizon["C1_MINUS_C0"])}
    primary = summary["C1_MINUS_C0"]["20"]
    bootstrap = primary["sequence_cluster_bootstrap"]
    protected = summary["C1_MINUS_C0"]["protected_identity_regression_h20"]
    lower = bootstrap.get("lower")
    strict_positive = bool(lower is not None and float(lower) > 0.0)
    runtime_clean = bool(stage05.get("runtime_future_gt_used") is False)
    gate_status = "PASS_GT_SIMULATED_TARGET_SCOPED_EFFECT_CONFIRMED" if (
        strict_positive
        and float(primary.get("mean_identity_error_reduction") or 0.0) > 0.0
        and int(primary.get("true_correct_crossings", 0)) > int(primary.get("true_incorrect_crossings", 0))
        and int(protected.get("regression_count", 0)) == 0
        and runtime_clean
    ) else "FAIL_FUTURE_EFFECT"
    if not strict_positive:
        next_root_cause = "TARGET_SESSION_PROPAGATION_FAILURE" if target_stream_rows < HORIZON * len(replay) * 0.5 else "TARGET_SCOPED_NO_STABLE_TARGET_GAIN"
    elif int(protected.get("regression_count", 0)) > 0:
        next_root_cause = "TARGET_SHADOW_DEDUP_OR_PROTECTED_ASSIGNMENT_REGRESSION"
    elif int(primary.get("true_correct_crossings", 0)) <= int(primary.get("true_incorrect_crossings", 0)):
        next_root_cause = "TARGET_SESSION_IDENTITY_DRIFT"
    else:
        next_root_cause = "NONE"

    payload = {
        "schema_version": "N72R6_STAGE06_EFFECT_V1",
        "status": gate_status,
        "primary_pair": "C1_MINUS_C0",
        "primary_horizon": 20,
        "gate": {
            "status": gate_status,
            "strict_sequence_cluster_ci_lower_gt_zero": strict_positive,
            "sequence_cluster_ci_lower": lower,
            "true_correct_crossings": int(primary.get("true_correct_crossings", 0)),
            "true_incorrect_crossings": int(primary.get("true_incorrect_crossings", 0)),
            "protected_regression_count_h20": int(protected.get("regression_count", 0)),
            "runtime_future_gt_used": False,
        },
        "summaries": summary,
        "event_metrics": all_records,
        "target_session_coverage": {
            "event_count": len(replay),
            "future_window_frame_count": HORIZON * len(replay),
            "future_candidate_present_event_count": int(target_stream_present),
            "future_candidate_present_frame_count": int(target_stream_rows),
            "future_candidate_assigned_target_row_count": int(target_stream_assigned),
            "candidate_recall_over_all_future_frames": float(target_stream_rows / (HORIZON * len(replay))),
            "assigned_target_rate_given_candidate": None if target_stream_rows == 0 else float(target_stream_assigned / target_stream_rows),
            "events_without_future_target_candidate": target_stream_missing_events,
            "root_cause_signal": next_root_cause,
        },
        "sequence_cluster_bootstrap": {
            "seed": BOOTSTRAP_SEED,
            "repetitions": BOOTSTRAP_REPETITIONS,
            "unit": "independent_sequence",
            "within_cluster_aggregation": "mean_event_value",
        },
        "inputs": {
            "protocol": str(PROTOCOL),
            "protocol_sha256": sha256_file(PROTOCOL),
            "stage05_status": str(STAGE05),
            "stage05_sha256": sha256_file(STAGE05),
            "replay_batch": str(REPLAY_BATCH),
            "replay_batch_sha256": sha256_file(REPLAY_BATCH),
            "gt_sequences": sorted(gt_hashes),
            "gt_sha256_by_sequence": gt_hashes,
        },
        "runtime_future_gt_used": False,
        "posthoc_gt_opened": True,
        "interaction_source": "simulated_from_gt",
        "not_real_human_evidence": True,
        "c2_status": "NOT_RUN_PENDING_C1_GATE_AND_C2_HUMAN_ANCHOR_PROTOCOL",
        "created_at_utc": now_utc(),
    }
    output = _resolve(args.output)
    # Preserve the first posthoc result before replacing the default artifact
    # with the shadow-dedup regression result.  The old replay sidecars remain
    # immutable, but the top-level effect path is intentionally the latest
    # sealed attempt.
    previous_effect = ROOT / "outputs/N72R6/ccam_paired_replay_results_attempt4_pre_shadow.json"
    if output == EFFECT and output.is_file() and not previous_effect.exists():
        atomic_json(previous_effect, read_json(output))
    previous_status = ROOT / "outputs/N72R6/stage_06_status_attempt4_pre_shadow.json"
    if STATUS.is_file() and not previous_status.exists():
        atomic_json(previous_status, read_json(STATUS))
    atomic_json(output, payload)
    status = {
        "schema_version": "N72R6_STAGE_STATUS_V1",
        "stage": "N72R6-06_POSTHOC_C1_MINUS_C0_EFFECT",
        "status": gate_status,
        "effect_evaluated": True,
        "posthoc_gt_opened": True,
        "runtime_future_gt_used": False,
        "event_count": len(replay),
        "sequence_count": int(primary["sequence_cluster_bootstrap"].get("clusters", 0)),
        "event_metric_row_count": len(all_records),
        "c1_minus_c0_h20_mean": primary.get("mean_identity_error_reduction"),
        "c1_minus_c0_h20_ci": primary["sequence_cluster_bootstrap"],
        "protected_regression_h20": protected,
        "target_session_coverage": payload["target_session_coverage"],
        "next_root_cause": next_root_cause,
        "c2_status": payload["c2_status"],
        "effect_artifact": str(output),
        "created_at_utc": now_utc(),
    }
    atomic_json(STATUS, status)
    atomic_json(CONTROLLER, {
        "schema_version": "N72R6_CONTROLLER_STATUS_V1",
        "current_stage": "N72R6-06",
        "architecture_mode": "TARGET_SCOPED_CORRECTION",
        "eligible_events": len(replay),
        "target_session_jobs_completed": len(replay),
        "C1_minus_C0_H20": primary.get("mean_identity_error_reduction"),
        "C1_minus_C0_H50": summary["C1_MINUS_C0"]["50"].get("mean_identity_error_reduction"),
        "C1_minus_C0_H100": summary["C1_MINUS_C0"]["100"].get("mean_identity_error_reduction"),
        "protected_regression": protected,
        "target_session_candidate_recall": payload["target_session_coverage"],
        "true_correct_crossing": primary.get("true_correct_crossings"),
        "true_incorrect_crossing": primary.get("true_incorrect_crossings"),
        "next_root_cause": next_root_cause,
        "runtime_future_gt_used": False,
        "created_at_utc": now_utc(),
    })
    print(json.dumps({
        "status": gate_status,
        "event_count": len(replay),
        "event_metric_rows": len(all_records),
        "C1_minus_C0_H20": primary.get("mean_identity_error_reduction"),
        "C1_minus_C0_H20_CI": [bootstrap.get("lower"), bootstrap.get("upper")],
        "protected_regression_h20": protected.get("regression_count"),
        "target_future_candidate_rows": target_stream_rows,
        "target_future_candidate_recall": payload["target_session_coverage"]["candidate_recall_over_all_future_frames"],
        "next_root_cause": next_root_cause,
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
