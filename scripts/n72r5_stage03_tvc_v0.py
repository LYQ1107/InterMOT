#!/usr/bin/env python3
"""N72R5 Stage 03: preregistered, non-learning Target-vs-Competitor v0.

This is a mechanism probe on the sealed N72R4 corrected stream.  It keeps the
candidate stream, checkpoint-derived features, explicit-NONE solver, and
evaluation definition fixed.  TVC_V0 only edits the target public-ID row of
the state-by-candidate score matrix.  GT is opened only after every runtime
artifact has passed the runtime-only validator.

The experiment is deliberately not a production implementation and does not
scan weights.  Its one frozen intervention is a robustly normalized,
bounded target-vs-competitor residual:

    clip((cos(candidate, human_anchor)
          + cos(candidate, target_persistent_prototype)
          - max_j cos(candidate, competitor_prototype_j))
         / robust_scale, -TRUST_RADIUS, TRUST_RADIUS)

``robust_scale`` is computed from the event-frame current score matrix before
any future row is inspected.  All non-target state rows are copied bitwise
from the baseline matrix before the single exact solver call.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import tempfile
from typing import Any, Iterable

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sam3_intermot.association.effect_assignment import solve_effect_assignment  # noqa: E402
from sam3_intermot.evaluation.interaction_effect_metrics import (  # noqa: E402
    BOOTSTRAP_REPETITIONS,
    BOOTSTRAP_SEED,
    metric_record,
    sequence_cluster_bootstrap,
)
from scripts.n72r4_stage10_cpu_analysis import (  # noqa: E402
    current_post_anchor,
    load_stage16,
)
from scripts.n72r5_stage01_decision_boundary import (  # noqa: E402
    VARIANTS as N72R4_VARIANTS,
    atomic_json,
    atomic_jsonl,
    box_iou,
    enrich_mechanism_row,
    load_gt,
    load_inputs,
    load_official_stream,
    normalized_feature,
    public_axis,
    reconstruct_states,
    sha256,
    update_reconstructed_states,
)


N72R4_ROOT = Path(
    os.environ.get(
        "N72R4_INPUT_ROOT",
        "/data2/usr_for_deadline/SAM3_InterMOT_N72R3R1/worktree/outputs/N72R4",
    )
)
OUT = ROOT / "outputs" / "N72R5"
ROUND_ROOT = Path(
    os.environ.get(
        "N72R5_STAGE03_ROOT",
        str(OUT / "mechanism_rounds" / "round_03_tvc_v0"),
    )
)
ARTIFACT_ROOT = ROUND_ROOT / "artifacts"
PROTOCOL_PATH = ROUND_ROOT / "tvc_v0_protocol.json"
RUNTIME_MANIFEST = ROUND_ROOT / "runtime_manifest.json"
RUNTIME_VALIDATION = ROUND_ROOT / "runtime_validation.json"
METRICS_PATH = ROUND_ROOT / "metrics.json"
GATE_PATH = ROUND_ROOT / "gate.json"
STAGE_STATUS = OUT / "stage_status" / "stage_03_status.json"

TVC_NAME = "TVC_V0_TARGET_VS_COMPETITOR"
TVC_TRUST_RADIUS = 1.0
TVC_COMPETITOR_TOP_K = 3
TVC_MAD_SCALE_FACTOR = 1.4826
TVC_SCALE_EPS = 1.0e-6
IOU_THRESHOLD = 0.5
HORIZONS = (20, 50, 100)
TVC_VARIANTS = ("M0_CURRENT_FRAME_CORRECTION_ONLY", TVC_NAME)


@dataclass(frozen=True)
class SolverState:
    association_state_id: int
    public_id: int


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def json_digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    ).hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object: {path}")
    return value


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise TypeError(f"{path}:{line_number}: expected object")
            rows.append(value)
    return rows


def unit(value: Any) -> np.ndarray:
    vector = np.asarray(value, dtype=np.float64).reshape(-1)
    if vector.size != 512 or not np.all(np.isfinite(vector)):
        raise ValueError(f"expected finite 512-D vector, got {vector.shape}")
    norm = float(np.linalg.norm(vector))
    if norm <= 1.0e-12:
        raise ValueError("zero-norm appearance vector")
    return vector / norm


def state_copy(value: Any) -> Any:
    """Copy the frozen Stage01 audit state without sharing mutable arrays."""
    return type(value)(
        public_id=int(value.public_id),
        last_box=None if value.last_box is None else value.last_box.copy(),
        last_feature=None if value.last_feature is None else value.last_feature.copy(),
        velocity=value.velocity.copy(),
        last_frame=int(value.last_frame),
        last_native=None if value.last_native is None else int(value.last_native),
    )


def output_candidate_rows(rows: list[dict[str, Any]], mapping: dict[str, int | None]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for row in rows:
        item = dict(row)
        uid = str(row["candidate_uid"])
        assigned = mapping.get(uid)
        item["candidate_uid"] = uid
        item["public_id"] = None if assigned is None else int(assigned)
        item["assignment_status"] = "EXPLICIT_NONE" if assigned is None else "ASSIGNED_TO_PUBLIC_ID"
        item.pop("dataset_gt_id", None)
        item.pop("gt_box", None)
        item.pop("future_gt", None)
        return_item = item
        output.append(return_item)
    return output


def exact_solver(
    rows: list[dict[str, Any]],
    matrix: np.ndarray,
    state_axis: list[int],
    public_values: list[int],
    event_id: str,
    frame: int,
    variant: str,
) -> dict[str, Any]:
    if matrix.shape != (len(state_axis), len(rows)):
        raise RuntimeError(f"score shape mismatch: {event_id}/{variant}/{frame}/{matrix.shape}/{len(rows)}")
    solver_states = [
        SolverState(association_state_id=int(state), public_id=int(public))
        for state, public in zip(state_axis, public_values)
    ]
    return solve_effect_assignment(
        candidate_rows=[{"candidate_uid": str(row["candidate_uid"])} for row in rows],
        persistent_states=solver_states,
        fused_state_candidate_scores=matrix,
        source_run_id=f"n72r5-stage03:{event_id}:{variant}:{frame}",
        session_id=f"n72r5-stage03:{event_id}:{variant}",
        none_score=0.0,
    )


def assignment_map(artifact: dict[str, Any], rows: list[dict[str, Any]]) -> dict[str, int | None]:
    expected = {str(row["candidate_uid"]) for row in rows}
    result: dict[str, int | None] = {uid: None for uid in expected}
    assignments = artifact.get("assignment_rows")
    if not isinstance(assignments, list) or len(assignments) != len(rows):
        raise RuntimeError("exact solver assignment row count mismatch")
    for item in assignments:
        uid = str(item.get("candidate_uid"))
        if uid not in expected:
            raise RuntimeError(f"exact solver returned unknown candidate UID: {uid}")
        value = item.get("public_id")
        result[uid] = None if value is None else int(value)
    return result


def assigned_candidate_by_public(mapping: dict[str, int | None]) -> dict[int, str]:
    result: dict[int, str] = {}
    for uid, public in mapping.items():
        if public is None:
            continue
        if public in result:
            raise RuntimeError(f"duplicate assigned public ID: {public}")
        result[int(public)] = str(uid)
    return result


def state_score_matrix(states: dict[int, Any], publics: list[int], rows: list[dict[str, Any]], frame: int) -> np.ndarray:
    # Importing the frozen Stage10 component avoids introducing another score
    # definition.  The source stream and this recomputation are audited below.
    from scripts.n72r4_stage10_cpu_analysis import association_score

    return np.asarray(
        [[association_score(states[public], row, frame) for row in rows] for public in publics],
        dtype=np.float64,
    )


def robust_scale_from_event_matrix(matrix: np.ndarray) -> dict[str, float]:
    values = np.asarray(matrix, dtype=np.float64).reshape(-1)
    values = values[np.isfinite(values)]
    if values.size == 0:
        raise RuntimeError("event-frame score distribution is empty")
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    scale = max(TVC_SCALE_EPS, TVC_MAD_SCALE_FACTOR * mad)
    return {
        "median": median,
        "mad": mad,
        "mad_scale_factor": TVC_MAD_SCALE_FACTOR,
        "scale": scale,
        "epsilon": TVC_SCALE_EPS,
        "source": "event_frame_current_state_x_candidate_score_matrix_only",
        "future_metrics_excluded": True,
    }


def human_and_persistent_prototypes(event: dict[str, Any], official_event_row: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, str, str]:
    stage16 = load_stage16(str(event["event_id"]))
    positive = (stage16.get("appearance_memory") or {}).get("positive") or []
    if not positive:
        raise RuntimeError(f"human anchor missing: {event['event_id']}")
    record = positive[0]
    if str(record.get("source")) not in {"human", "current_frame_simulated_human_box_roi"}:
        raise RuntimeError(f"human anchor provenance invalid: {event['event_id']}/{record.get('source')}")
    human = unit(record.get("feature"))
    target_post, _ = current_post_anchor(stage16, official_event_row)
    machine = unit(target_post["feature"])
    # This is the exact non-learning prototype update used by the frozen
    # N72R4 memory replay: machine seed followed by a 0.2 human update.
    persistent = unit(0.8 * machine + 0.2 * human)
    human_hash = hashlib.sha256(np.asarray(human, dtype="<f8").tobytes()).hexdigest()
    persistent_hash = hashlib.sha256(np.asarray(persistent, dtype="<f8").tobytes()).hexdigest()
    return human, persistent, human_hash, persistent_hash


def select_competitors(
    event: dict[str, Any],
    event_row: dict[str, Any],
    states: dict[int, Any],
    public_values: list[int],
    event_matrix: np.ndarray,
) -> dict[str, Any]:
    target_public = int(event["target_public_id"])
    target_index = public_values.index(target_public)
    candidates = list(event_row.get("candidate_rows", []))
    target_pre = [row for row in candidates if row.get("public_id") is not None and int(row["public_id"]) == target_public]
    target_pre_uid = None if not target_pre else str(target_pre[0]["candidate_uid"])
    target_pre_col = None
    if target_pre_uid is not None:
        for index, row in enumerate(candidates):
            if str(row["candidate_uid"]) == target_pre_uid:
                target_pre_col = index
                break
    scores: list[dict[str, Any]] = []
    for index, public in enumerate(public_values):
        if int(public) == target_public:
            continue
        state = states[int(public)]
        if state.last_feature is None:
            continue
        max_score = float(np.max(event_matrix[index])) if event_matrix.shape[1] else 0.0
        occupancy_score = None if target_pre_col is None else float(event_matrix[index, target_pre_col])
        scores.append(
            {
                "public_id": int(public),
                "max_event_frame_base_score": max_score,
                "target_pre_candidate_score": occupancy_score,
                "has_persistent_feature": True,
            }
        )
    if not scores:
        raise RuntimeError(f"no feature-bearing competitor identities: {event['event_id']}")
    action = str(event["action_type"])
    if action == "AUTHORITATIVE_REASSIGN" and target_pre_col is not None:
        ranked = sorted(
            scores,
            key=lambda item: (
                -(float(item["target_pre_candidate_score"]) if item["target_pre_candidate_score"] is not None else -math.inf),
                -float(item["max_event_frame_base_score"]),
                int(item["public_id"]),
            ),
        )
        rationale = "AUTHORITATIVE_REASSIGN_target_pre_candidate_occupancy_then_event_score"
    else:
        ranked = sorted(scores, key=lambda item: (-float(item["max_event_frame_base_score"]), int(item["public_id"])))
        rationale = "runtime_event_frame_base_score_top_k"
    selected = ranked[:TVC_COMPETITOR_TOP_K]
    return {
        "action_type": action,
        "target_public_id": target_public,
        "target_pre_candidate_uid": target_pre_uid,
        "target_pre_candidate_column": target_pre_col,
        "top_k": TVC_COMPETITOR_TOP_K,
        "selected_public_ids": [int(item["public_id"]) for item in selected],
        "ranked_candidates": ranked,
        "rationale": rationale,
        "selection_inputs": [
            "event_frame_candidate_rows",
            "event_frame_base_score_matrix",
            "persistent_public_identity_axis",
        ],
        "posthoc_fields_excluded": [
            "dataset_gt_id",
            "future_identity_error",
            "future_iou",
            "future_reward",
            "future_replay_outcome",
        ],
        "runtime_future_gt_used": False,
    }


def tvc_components(
    rows: list[dict[str, Any]],
    states: dict[int, Any],
    target_public: int,
    competitor_publics: list[int],
    human: np.ndarray,
    persistent: np.ndarray,
    scale: float,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    if target_public not in states:
        raise RuntimeError(f"target public state missing: {target_public}")
    prototypes = {
        public: unit(states[public].last_feature)
        for public in competitor_publics
        if states[public].last_feature is not None
    }
    residual = np.zeros(len(rows), dtype=np.float64)
    details: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        feature = unit(row["feature"])
        human_similarity = float(np.dot(feature, human))
        persistent_similarity = float(np.dot(feature, persistent))
        competitor_scores = [
            (float(np.dot(feature, prototype)), int(public))
            for public, prototype in prototypes.items()
        ]
        if competitor_scores:
            best_competitor_similarity, best_competitor = max(
                competitor_scores,
                key=lambda item: (item[0], -item[1]),
            )
        else:
            best_competitor_similarity, best_competitor = 0.0, None
        relative_margin = human_similarity + persistent_similarity - best_competitor_similarity
        normalized_margin = relative_margin / max(float(scale), TVC_SCALE_EPS)
        bounded = float(np.clip(normalized_margin, -TVC_TRUST_RADIUS, TVC_TRUST_RADIUS))
        residual[index] = bounded
        details.append(
            {
                "candidate_uid": str(row["candidate_uid"]),
                "candidate_index": int(row["candidate_index"]),
                "human_target_similarity": human_similarity,
                "persistent_target_similarity": persistent_similarity,
                "best_competitor_similarity": best_competitor_similarity,
                "best_competitor_public_id": best_competitor,
                "relative_margin": relative_margin,
                "normalized_margin": normalized_margin,
                "bounded_target_row_residual": bounded,
                "scale": float(scale),
                "trust_radius": TVC_TRUST_RADIUS,
            }
        )
    return residual, details


def validate_runtime_rows(events: list[dict[str, Any]], rows_by_event: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    checked_rows = 0
    checked_frames = 0
    checked_candidates = 0
    score_cells = 0
    non_target_cells_checked = 0
    errors: list[str] = []
    for event in events:
        event_id = str(event["event_id"])
        rows = rows_by_event.get(event_id, [])
        expected = {(variant, frame) for variant in TVC_VARIANTS for frame in range(int(event["event_frame"]), int(event["event_frame"]) + 101)}
        actual = {(str(row.get("variant")), int(row.get("frame", -1))) for row in rows}
        if actual != expected:
            errors.append(f"{event_id}:frame_variant_key_set_mismatch")
        by_key = {(str(row["variant"]), int(row["frame"])): row for row in rows}
        for variant, frame in sorted(expected):
            row = by_key.get((variant, frame))
            if row is None:
                continue
            checked_rows += 1
            checked_frames += 1
            if row.get("runtime_future_gt_used") is not False or row.get("runtime_gt_read") is not False or row.get("posthoc_gt_used") is not False:
                errors.append(f"{event_id}/{variant}/{frame}:gt_boundary")
            if {"dataset_gt_id", "gt_box", "future_gt", "future_identity_error", "reward"}.intersection(row):
                errors.append(f"{event_id}/{variant}/{frame}:posthoc_field_in_runtime")
            candidates = row.get("candidate_rows")
            if not isinstance(candidates, list):
                errors.append(f"{event_id}/{variant}/{frame}:candidate_rows_missing")
                continue
            uids = [str(item.get("candidate_uid")) for item in candidates]
            if len(uids) != len(set(uids)) or any(uid in {"", "None"} for uid in uids):
                errors.append(f"{event_id}/{variant}/{frame}:candidate_uid_invalid")
            checked_candidates += len(candidates)
            base = np.asarray(row.get("base_score_matrix"), dtype=np.float64)
            fused = np.asarray(row.get("fused_score_matrix"), dtype=np.float64)
            appearance = np.asarray(row.get("appearance_score_matrix"), dtype=np.float64)
            state_axis = row.get("association_state_axis", [])
            if base.shape != (len(state_axis), len(candidates)) or fused.shape != base.shape or appearance.shape != base.shape:
                errors.append(f"{event_id}/{variant}/{frame}:matrix_shape")
                continue
            if not np.isfinite(base).all() or not np.isfinite(fused).all() or not np.isfinite(appearance).all():
                errors.append(f"{event_id}/{variant}/{frame}:matrix_nonfinite")
            score_cells += int(base.size)
            if frame == int(event["event_frame"]):
                if row.get("solver_executed") is not False or row.get("memory_read") is not False:
                    errors.append(f"{event_id}/{variant}/{frame}:event_frame_causal_boundary")
            else:
                if row.get("solver_executed") is not True or not isinstance(row.get("solver"), dict):
                    errors.append(f"{event_id}/{variant}/{frame}:solver_missing")
            if variant == TVC_NAME:
                target_idx = int(row["target_state_index"])
                non_target = [index for index in range(base.shape[0]) if index != target_idx]
                if non_target and not np.array_equal(base[non_target, :], fused[non_target, :]):
                    errors.append(f"{event_id}/{frame}:non_target_score_row_changed")
                if non_target:
                    non_target_cells_checked += int(len(non_target) * base.shape[1])
                expected_fused = base.copy()
                expected_fused[target_idx, :] += appearance[target_idx, :]
                if not np.array_equal(expected_fused, fused):
                    errors.append(f"{event_id}/{frame}:tvc_fused_reconstruction_mismatch")
            else:
                if np.any(appearance != 0.0) or not np.array_equal(base, fused):
                    errors.append(f"{event_id}/{frame}:m0_not_identity")
    return {
        "schema_version": "N72R5_STAGE03_RUNTIME_VALIDATION_V1",
        "status": "PASS_STAGE03_RUNTIME_ARTIFACT_VALIDATION" if not errors else "FAIL_STAGE03_RUNTIME_ARTIFACT_VALIDATION",
        "event_count": len(events),
        "expected_rows": len(events) * len(TVC_VARIANTS) * 101,
        "checked_rows": checked_rows,
        "checked_frames": checked_frames,
        "checked_candidates": checked_candidates,
        "score_cells": score_cells,
        "non_target_score_cells_checked": non_target_cells_checked,
        "errors": errors,
        "runtime_future_gt_used": False,
        "posthoc_gt_used": False,
    }


def _assignment_map_from_row(row: dict[str, Any]) -> dict[str, int | None]:
    return {
        str(candidate["candidate_uid"]): None if candidate.get("public_id") is None else int(candidate["public_id"])
        for candidate in row.get("candidate_rows", [])
    }


def _public_iou(row: dict[str, Any], public_id: int, box: list[float]) -> float:
    return max(
        (
            box_iou(candidate["box_xyxy"], box)
            for candidate in row.get("candidate_rows", [])
            if candidate.get("public_id") is not None and int(candidate["public_id"]) == int(public_id)
        ),
        default=0.0,
    )


def _candidate_best(row: dict[str, Any], box: list[float]) -> tuple[float, dict[str, Any] | None]:
    ranked = sorted(
        ((box_iou(candidate["box_xyxy"], box), candidate) for candidate in row.get("candidate_rows", [])),
        key=lambda item: (-item[0], str(item[1].get("candidate_uid"))),
    )
    return (float(ranked[0][0]), ranked[0][1]) if ranked else (0.0, None)


def _wrong_reassociation(row: dict[str, Any], target_public: int, gt_frame: dict[int, list[float]], target_gid: int) -> bool:
    target_candidate = next(
        (
            candidate
            for candidate in row.get("candidate_rows", [])
            if candidate.get("public_id") is not None and int(candidate["public_id"]) == int(target_public)
        ),
        None,
    )
    if target_candidate is None:
        return False
    return any(
        int(gid) != int(target_gid) and box_iou(target_candidate["box_xyxy"], box) >= IOU_THRESHOLD
        for gid, box in gt_frame.items()
    )


def _protected_public_by_gt(event_row: dict[str, Any], gt_frame: dict[int, list[float]], target_gid: int) -> dict[int, int]:
    result: dict[int, int] = {}
    for gid, box in gt_frame.items():
        if int(gid) == int(target_gid):
            continue
        best_iou, candidate = _candidate_best(event_row, box)
        if best_iou >= IOU_THRESHOLD and candidate is not None and candidate.get("public_id") is not None:
            public = int(candidate["public_id"])
            if public not in result.values():
                result[int(gid)] = public
    return result


def metric_template() -> dict[str, float | int]:
    return {
        "evaluated_frames": 0,
        "target_iou_sum": 0.0,
        "baseline_target_iou_sum": 0.0,
        "target_correct_frames": 0,
        "baseline_target_correct_frames": 0,
        "target_missing_frames": 0,
        "baseline_target_missing_frames": 0,
        "target_identity_error_frames": 0,
        "baseline_identity_error_frames": 0,
        "wrong_reassociation_frames": 0,
        "baseline_wrong_reassociation_frames": 0,
        "candidate_present_frames": 0,
        "baseline_candidate_present_frames": 0,
        "assignment_change_count": 0,
        "true_correct_crossing_count": 0,
        "true_incorrect_crossing_count": 0,
        "directional_improvement_count": 0,
        "directional_regression_count": 0,
        "neutral_change_count": 0,
        "protected_compared": 0,
        "protected_regression_count": 0,
        "protected_improvement_count": 0,
        "identity_error_reduction_sum": 0.0,
        "delta_iou_sum": 0.0,
        "solver_coupled_collateral_count": 0,
    }


def finalize_metric(metric: dict[str, Any]) -> dict[str, Any]:
    denom = max(1, int(metric["evaluated_frames"]))
    metric["target_mean_iou"] = float(metric["target_iou_sum"] / denom)
    metric["baseline_target_mean_iou"] = float(metric["baseline_target_iou_sum"] / denom)
    metric["delta_iou_mean_vs_m0"] = float(metric["delta_iou_sum"] / denom)
    metric["identity_error_reduction"] = float(metric["identity_error_reduction_sum"] / denom)
    metric["future_identity_error"] = float(metric["target_identity_error_frames"] / denom)
    metric["baseline_future_identity_error"] = float(metric["baseline_identity_error_frames"] / denom)
    metric["missing_rate"] = float(metric["target_missing_frames"] / denom)
    metric["baseline_missing_rate"] = float(metric["baseline_target_missing_frames"] / denom)
    metric["wrong_reassociation_rate"] = float(metric["wrong_reassociation_frames"] / denom)
    metric["baseline_wrong_reassociation_rate"] = float(metric["baseline_wrong_reassociation_frames"] / denom)
    metric["candidate_recall"] = float(metric["candidate_present_frames"] / denom)
    metric["baseline_candidate_recall"] = float(metric["baseline_candidate_present_frames"] / denom)
    metric["assignment_change_rate"] = float(metric["assignment_change_count"] / denom)
    protected = int(metric["protected_compared"])
    metric["protected_regression_rate"] = None if protected == 0 else float(metric["protected_regression_count"] / protected)
    metric["protected_improvement_rate"] = None if protected == 0 else float(metric["protected_improvement_count"] / protected)
    return metric


def score_event(event: dict[str, Any], rows: list[dict[str, Any]]) -> dict[str, Any]:
    gt = load_gt(str(event["sequence"]))
    target_public = int(event["target_public_id"])
    target_gid = int(event["dataset_gt_id"])
    event_frame = int(event["event_frame"])
    by_key = {(str(row["variant"]), int(row["frame"])): row for row in rows}
    event_row = by_key[(TVC_VARIANTS[0], event_frame)]
    protected = _protected_public_by_gt(event_row, gt.get(event_frame, {}), target_gid)
    horizons: dict[str, Any] = {}
    for horizon in HORIZONS:
        metric = metric_template()
        frame_details: list[dict[str, Any]] = []
        previous_treatment_uid: str | None = None
        previous_baseline_uid: str | None = None
        for frame in range(event_frame + 1, event_frame + horizon + 1):
            gt_box = gt.get(frame, {}).get(target_gid)
            if gt_box is None:
                continue
            baseline = by_key[(TVC_VARIANTS[0], frame)]
            treatment = by_key[(TVC_NAME, frame)]
            baseline_iou = _public_iou(baseline, target_public, gt_box)
            treatment_iou = _public_iou(treatment, target_public, gt_box)
            baseline_best_iou, baseline_best = _candidate_best(baseline, gt_box)
            treatment_best_iou, treatment_best = _candidate_best(treatment, gt_box)
            baseline_correct = bool(baseline_iou >= IOU_THRESHOLD)
            treatment_correct = bool(treatment_iou >= IOU_THRESHOLD)
            baseline_missing = not any(value == target_public for value in _assignment_map_from_row(baseline).values())
            treatment_missing = not any(value == target_public for value in _assignment_map_from_row(treatment).values())
            baseline_wrong = _wrong_reassociation(baseline, target_public, gt.get(frame, {}), target_gid)
            treatment_wrong = _wrong_reassociation(treatment, target_public, gt.get(frame, {}), target_gid)
            baseline_map = _assignment_map_from_row(baseline)
            treatment_map = _assignment_map_from_row(treatment)
            changed_uids = {
                uid
                for uid in set(baseline_map) | set(treatment_map)
                if baseline_map.get(uid) != treatment_map.get(uid)
            }
            assignment_changed = bool(changed_uids)
            record = metric_record(
                baseline_iou=baseline_iou,
                treatment_iou=treatment_iou,
                baseline_correct=baseline_correct,
                treatment_correct=treatment_correct,
                assignment_changed=assignment_changed,
            )
            metric["evaluated_frames"] += 1
            metric["target_iou_sum"] += treatment_iou
            metric["baseline_target_iou_sum"] += baseline_iou
            metric["target_correct_frames"] += int(treatment_correct)
            metric["baseline_target_correct_frames"] += int(baseline_correct)
            metric["target_missing_frames"] += int(treatment_missing)
            metric["baseline_target_missing_frames"] += int(baseline_missing)
            metric["target_identity_error_frames"] += int(not treatment_correct)
            metric["baseline_identity_error_frames"] += int(not baseline_correct)
            metric["wrong_reassociation_frames"] += int(treatment_wrong)
            metric["baseline_wrong_reassociation_frames"] += int(baseline_wrong)
            metric["candidate_present_frames"] += int(treatment_best_iou >= IOU_THRESHOLD)
            metric["baseline_candidate_present_frames"] += int(baseline_best_iou >= IOU_THRESHOLD)
            metric["assignment_change_count"] += int(assignment_changed)
            metric["true_correct_crossing_count"] += int(record["true_correct_crossing"])
            metric["true_incorrect_crossing_count"] += int(record["true_incorrect_crossing"])
            metric["directional_improvement_count"] += int(record["directional_improvement"])
            metric["directional_regression_count"] += int(record["directional_regression"])
            metric["neutral_change_count"] += int(record["assignment_change_type"] == "NEUTRAL_CHANGE")
            metric["identity_error_reduction_sum"] += float(record["identity_error_reduction"])
            metric["delta_iou_sum"] += float(record["delta_iou"])
            target_related = {
                uid
                for uid in changed_uids
                if baseline_map.get(uid) == target_public or treatment_map.get(uid) == target_public
            }
            metric["solver_coupled_collateral_count"] += int(bool(changed_uids - target_related))
            for gid, public in protected.items():
                protected_box = gt.get(frame, {}).get(int(gid))
                if protected_box is None:
                    continue
                baseline_other_iou = _public_iou(baseline, public, protected_box)
                treatment_other_iou = _public_iou(treatment, public, protected_box)
                metric["protected_compared"] += 1
                metric["protected_regression_count"] += int(baseline_other_iou >= IOU_THRESHOLD and treatment_other_iou < IOU_THRESHOLD)
                metric["protected_improvement_count"] += int(treatment_other_iou >= IOU_THRESHOLD and baseline_other_iou < IOU_THRESHOLD)
            frame_details.append(
                {
                    "frame": frame,
                    "target_iou": treatment_iou,
                    "baseline_m0_target_iou": baseline_iou,
                    "target_correct": treatment_correct,
                    "baseline_m0_target_correct": baseline_correct,
                    "target_missing": treatment_missing,
                    "baseline_m0_target_missing": baseline_missing,
                    "candidate_best_iou": treatment_best_iou,
                    "baseline_m0_candidate_best_iou": baseline_best_iou,
                    "wrong_reassociation": treatment_wrong,
                    "baseline_m0_wrong_reassociation": baseline_wrong,
                    "assignment_changed": assignment_changed,
                    "assignment_change_type": record["assignment_change_type"],
                    "identity_error_reduction": record["identity_error_reduction"],
                    "delta_iou": record["delta_iou"],
                    "changed_candidate_uids": sorted(changed_uids),
                    "runtime_future_gt_used": False,
                    "posthoc_gt_used": True,
                }
            )
        horizons[str(horizon)] = {**finalize_metric(metric), "frame_details": frame_details}
    return {
        "event_id": str(event["event_id"]),
        "sequence": str(event["sequence"]),
        "action_type": str(event["action_type"]),
        "event_frame": event_frame,
        "target_public_id": target_public,
        "target_dataset_gt_id_posthoc": target_gid,
        "competitor_public_ids": event["_tvc_competitor_selection"]["selected_public_ids"],
        "protected_public_by_gt_posthoc": {str(gid): int(public) for gid, public in sorted(protected.items())},
        "horizons": horizons,
        "runtime_future_gt_used": False,
        "gt_usage": "posthoc_only_after_runtime_artifact_validation",
        "interaction_source": "simulated_from_gt",
        "not_real_human_evidence": True,
    }


def aggregate(event_metrics: list[dict[str, Any]], horizon: int, action: str | None = None) -> dict[str, Any]:
    selected = [item for item in event_metrics if action is None or item["action_type"] == action]
    total = metric_template()
    for item in selected:
        value = item["horizons"][str(horizon)]
        for key in total:
            total[key] += value.get(key, 0)
    result = finalize_metric(total)
    result["event_count"] = len(selected)
    result["independent_sequence_count"] = len({item["sequence"] for item in selected})
    sequence_values: dict[str, list[float]] = defaultdict(list)
    for item in selected:
        sequence_values[str(item["sequence"])].append(float(item["horizons"][str(horizon)]["identity_error_reduction"]))
    result["sequence_cluster_bootstrap_95ci"] = sequence_cluster_bootstrap(
        sequence_values,
        seed=BOOTSTRAP_SEED,
        repetitions=BOOTSTRAP_REPETITIONS,
    )
    return result


def build_event_runtime(event: dict[str, Any], by_key: dict[tuple[str, int], dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    event_id = str(event["event_id"])
    event_frame = int(event["event_frame"])
    source_event = by_key[(N72R4_VARIANTS[0], event_frame)]
    official_stream = load_official_stream(event_id, event_frame)
    official_event = official_stream[event_frame]
    enriched_event = enrich_mechanism_row(source_event, official_event)
    state_axis, public_values = public_axis(enriched_event)
    states_for_selection = reconstruct_states(enriched_event, public_values, event_frame, official_event)
    event_base = np.asarray(enriched_event["base_score_matrix"], dtype=np.float64)
    if event_base.shape != (len(public_values), len(enriched_event["candidate_rows"])):
        raise RuntimeError(f"event base shape invalid: {event_id}")
    scale = robust_scale_from_event_matrix(event_base)
    human, persistent, human_hash, persistent_hash = human_and_persistent_prototypes(event, official_event)
    selection = select_competitors(event, enriched_event, states_for_selection, public_values, event_base)
    event["_tvc_competitor_selection"] = selection
    rows: list[dict[str, Any]] = []
    # The event frame is a correction/write boundary, not a TVC read.  The
    # frozen mapping is retained for both paired branches, and the first TVC
    # score is at event+1.
    event_candidates = [dict(candidate) for candidate in enriched_event["candidate_rows"]]
    event_mapping = {
        str(candidate["candidate_uid"]): None if candidate.get("public_id") is None else int(candidate["public_id"])
        for candidate in event_candidates
    }
    for variant in TVC_VARIANTS:
        rows.append(
            {
                "schema_version": "N72R5_STAGE03_TVC_V0_FRAME_V1",
                "event_id": event_id,
                "sequence": str(event["sequence"]),
                "action_type": str(event["action_type"]),
                "event_frame": event_frame,
                "frame": event_frame,
                "frame_horizon": 0,
                "phase": "CURRENT_FRAME_CORRECTION_AND_TVC_WRITE",
                "variant": variant,
                "candidate_stream_kind": "N72R4_FROZEN_OFFICIAL_CORRECTED_EVENT_FRAME",
                "candidate_stream_sha256": str(official_event["frame_hash_sha256"]),
                "candidate_rows": output_candidate_rows(event_candidates, event_mapping),
                "association_state_axis": state_axis,
                "public_id_order": public_values,
                "target_public_id": int(event["target_public_id"]),
                "target_state_index": public_values.index(int(event["target_public_id"])),
                "base_score_matrix": event_base.tolist(),
                "appearance_score_matrix": np.zeros_like(event_base).tolist(),
                "fused_score_matrix": event_base.tolist(),
                "solver_executed": False,
                "solver": None,
                "assignment_map": event_mapping,
                "memory_write": True,
                "memory_read": False,
                "tvc_write": variant == TVC_NAME,
                "tvc_visible_from_frame": event_frame + 1,
                "tvc_components_by_candidate": [],
                "tvc_normalization": scale,
                "tvc_competitor_selection": selection,
                "human_anchor_feature_sha256": human_hash,
                "persistent_target_feature_sha256": persistent_hash,
                "causal_boundary": {
                    "event_frame_memory_read": False,
                    "event_frame_tvc_read": False,
                    "write_after_spatial_correction": True,
                    "first_tvc_visible_frame": event_frame + 1,
                    "runtime_future_gt_used": False,
                },
                "runtime_future_gt_used": False,
                "runtime_gt_read": False,
                "posthoc_gt_used": False,
            }
        )
    baseline_states = {public: state_copy(value) for public, value in states_for_selection.items()}
    tvc_states = {public: state_copy(value) for public, value in states_for_selection.items()}
    target_public = int(event["target_public_id"])
    for frame in range(event_frame + 1, event_frame + 101):
        source_row = enrich_mechanism_row(by_key[(N72R4_VARIANTS[0], frame)], official_stream[frame])
        candidates = [dict(candidate) for candidate in source_row["candidate_rows"]]
        frozen_base = np.asarray(source_row["base_score_matrix"], dtype=np.float64)
        recomputed_baseline = state_score_matrix(baseline_states, public_values, candidates, frame)
        baseline_base_error = float(np.max(np.abs(frozen_base - recomputed_baseline))) if frozen_base.size else 0.0
        if baseline_base_error > 1.0e-4:
            raise RuntimeError(f"frozen baseline score reconstruction mismatch: {event_id}/{frame}/{baseline_base_error}")
        baseline_solver = exact_solver(candidates, frozen_base, state_axis, public_values, event_id, frame, TVC_VARIANTS[0])
        baseline_mapping = assignment_map(baseline_solver, candidates)
        update_reconstructed_states(baseline_states, {"candidate_rows": candidates, "frame": frame}, baseline_mapping)

        tvc_base = state_score_matrix(tvc_states, public_values, candidates, frame)
        residual, details = tvc_components(
            candidates,
            tvc_states,
            target_public,
            [int(value) for value in selection["selected_public_ids"]],
            human,
            persistent,
            float(scale["scale"]),
        )
        tvc_appearance = np.zeros_like(tvc_base)
        target_index = public_values.index(target_public)
        tvc_appearance[target_index, :] = residual
        tvc_fused = tvc_base.copy()
        tvc_fused[target_index, :] += residual
        tvc_solver = exact_solver(candidates, tvc_fused, state_axis, public_values, event_id, frame, TVC_NAME)
        tvc_mapping = assignment_map(tvc_solver, candidates)
        update_reconstructed_states(tvc_states, {"candidate_rows": candidates, "frame": frame}, tvc_mapping)
        common = {
            "event_id": event_id,
            "sequence": str(event["sequence"]),
            "action_type": str(event["action_type"]),
            "event_frame": event_frame,
            "frame": frame,
            "frame_horizon": frame - event_frame,
            "phase": "FUTURE_ASSOCIATION",
            "candidate_stream_kind": "N72R4_FROZEN_OFFICIAL_CORRECTED_FUTURE_STREAM",
            "candidate_stream_sha256": str(official_stream[frame]["frame_hash_sha256"]),
            "association_state_axis": state_axis,
            "public_id_order": public_values,
            "target_public_id": target_public,
            "target_state_index": target_index,
            "tvc_normalization": scale,
            "tvc_competitor_selection": selection,
            "human_anchor_feature_sha256": human_hash,
            "persistent_target_feature_sha256": persistent_hash,
            "causal_boundary": {
                "event_frame_tvc_read": False,
                "first_tvc_visible_frame": event_frame + 1,
                "runtime_future_gt_used": False,
            },
            "runtime_future_gt_used": False,
            "runtime_gt_read": False,
            "posthoc_gt_used": False,
        }
        rows.append(
            {
                **common,
                "schema_version": "N72R5_STAGE03_TVC_V0_FRAME_V1",
                "variant": TVC_VARIANTS[0],
                "candidate_rows": output_candidate_rows(candidates, baseline_mapping),
                "base_score_matrix": frozen_base.tolist(),
                "appearance_score_matrix": np.zeros_like(frozen_base).tolist(),
                "fused_score_matrix": frozen_base.tolist(),
                "solver_executed": True,
                "solver": baseline_solver,
                "assignment_map": baseline_mapping,
                "memory_write": False,
                "memory_read": False,
                "tvc_write": False,
                "tvc_components_by_candidate": [],
                "baseline_reconstruction_max_abs_error": baseline_base_error,
                "target_scoped_non_target_rows_bitwise_equal": True,
                "solver_coupled_collateral": False,
            }
        )
        rows.append(
            {
                **common,
                "schema_version": "N72R5_STAGE03_TVC_V0_FRAME_V1",
                "variant": TVC_NAME,
                "candidate_rows": output_candidate_rows(candidates, tvc_mapping),
                "base_score_matrix": tvc_base.tolist(),
                "appearance_score_matrix": tvc_appearance.tolist(),
                "fused_score_matrix": tvc_fused.tolist(),
                "solver_executed": True,
                "solver": tvc_solver,
                "assignment_map": tvc_mapping,
                "memory_write": False,
                "memory_read": True,
                "tvc_write": True,
                "tvc_components_by_candidate": details,
                "target_scoped_non_target_rows_bitwise_equal": bool(
                    np.array_equal(tvc_base[[index for index in range(tvc_base.shape[0]) if index != target_index], :], tvc_fused[[index for index in range(tvc_fused.shape[0]) if index != target_index], :])
                ) if tvc_base.shape[0] > 1 else True,
                "solver_coupled_collateral": False,
            }
        )
    rows.sort(key=lambda row: (int(row["frame"]), TVC_VARIANTS.index(str(row["variant"]))))
    return rows, {
        "event_id": event_id,
        "sequence": str(event["sequence"]),
        "action_type": str(event["action_type"]),
        "event_frame": event_frame,
        "future_frame_count": 100,
        "variant_count": len(TVC_VARIANTS),
        "row_count": len(rows),
        "tvc_name": TVC_NAME,
        "tvc_normalization": scale,
        "competitor_selection": selection,
        "human_anchor_feature_sha256": human_hash,
        "persistent_target_feature_sha256": persistent_hash,
        "runtime_future_gt_used": False,
        "interaction_source": "simulated_from_gt",
        "not_real_human_evidence": True,
    }


def run(args: argparse.Namespace) -> int:
    if ROUND_ROOT.exists() and any(ROUND_ROOT.iterdir()):
        raise RuntimeError(f"refusing to overwrite non-empty Stage03 round root: {ROUND_ROOT}")
    ROUND_ROOT.mkdir(parents=True, exist_ok=True)
    protocol = {
        "schema_version": "N72R5_STAGE03_TVC_V0_PROTOCOL_V1",
        "status": "FROZEN_BEFORE_RUNTIME_AND_POSTHOC_SCORING",
        "stage": "03_TVC_V0_TARGET_VS_COMPETITOR",
        "name": TVC_NAME,
        "hypothesis": "target-vs-competitor human appearance residual can cross the exact assignment boundary when candidate is present",
        "formula": "clip((cos(candidate,human_anchor)+cos(candidate,target_persistent_prototype)-max_j cos(candidate,competitor_prototype_j))/robust_scale,-R,R)",
        "trust_radius": TVC_TRUST_RADIUS,
        "competitor_top_k": TVC_COMPETITOR_TOP_K,
        "normalization": {
            "method": "median_absolute_deviation",
            "mad_scale_factor": TVC_MAD_SCALE_FACTOR,
            "epsilon": TVC_SCALE_EPS,
            "source": "event_frame_current_state_x_candidate_score_matrix",
            "future_outcome_fields_excluded": True,
        },
        "candidate_stream": "N72R4_frozen_official_corrected_stream",
        "checkpoint_changed": False,
        "candidate_definition_changed": False,
        "solver": "sam3_intermot.association.effect_assignment.solve_effect_assignment_only",
        "solver_changed": False,
        "score_scan": False,
        "runtime_future_gt_used": False,
        "interaction_source": "simulated_from_gt",
        "not_real_human_evidence": True,
        "selection_post_treatment_fields_forbidden": [
            "future_identity_error",
            "future_iou",
            "future_reward",
            "H20",
            "H50",
            "H100",
            "replay_outcome",
        ],
        "created_at_utc": now_utc(),
    }
    atomic_json(PROTOCOL_PATH, protocol)
    events, loaded, _ = load_inputs()
    if args.event_limit:
        events = events[: int(args.event_limit)]
    if not events:
        raise RuntimeError("no frozen N72R4 events selected")
    runtime_manifest = {
        "schema_version": "N72R5_STAGE03_TVC_V0_RUNTIME_MANIFEST_V1",
        "status": "BUILDING_RUNTIME_ARTIFACTS",
        "event_count": len(events),
        "event_ids": [str(event["event_id"]) for event in events],
        "protocol_sha256": sha256(PROTOCOL_PATH),
        "inputs": {
            "stage11_results": str(N72R4_ROOT / "metrics" / "corrected_stream_m1_m4_results_attempt1.json"),
            "stage11_results_sha256": sha256(N72R4_ROOT / "metrics" / "corrected_stream_m1_m4_results_attempt1.json"),
            "official_corrected_root": str(N72R4_ROOT / "official_corrected" / "full_attempt2"),
        },
        "runtime_future_gt_used": False,
    }
    atomic_json(RUNTIME_MANIFEST, runtime_manifest)
    rows_by_event: dict[str, list[dict[str, Any]]] = {}
    summaries: list[dict[str, Any]] = []
    for event in events:
        rows, summary = build_event_runtime(event, loaded[str(event["event_id"])])
        path = ARTIFACT_ROOT / f"{event['event_id']}.jsonl"
        atomic_jsonl(path, rows)
        summary["artifact_path"] = str(path)
        summary["artifact_sha256"] = sha256(path)
        summaries.append(summary)
        rows_by_event[str(event["event_id"])] = rows
    runtime_manifest["status"] = "RUNTIME_ARTIFACTS_WRITTEN"
    runtime_manifest["event_summaries"] = summaries
    runtime_manifest["runtime_future_gt_used"] = False
    atomic_json(RUNTIME_MANIFEST, runtime_manifest)
    validation = validate_runtime_rows(events, rows_by_event)
    atomic_json(RUNTIME_VALIDATION, validation)
    if validation["status"] != "PASS_STAGE03_RUNTIME_ARTIFACT_VALIDATION":
        atomic_json(
            GATE_PATH,
            {
                "schema_version": "N72R5_STAGE03_TVC_V0_GATE_V1",
                "status": "BLOCKED_STAGE03_RUNTIME_VALIDATION",
                "runtime_validation": str(RUNTIME_VALIDATION),
                "future_effect_gate": "NOT_EVALUATED",
                "training_authorized": False,
                "production_authorized": False,
                "runtime_future_gt_used": False,
            },
        )
        atomic_json(
            STAGE_STATUS,
            {
                "schema_version": "N72R5_STAGE_STATUS_V1",
                "stage": "03_TVC_V0_TARGET_VS_COMPETITOR",
                "status": "BLOCKED_STAGE03_RUNTIME_VALIDATION",
                "runtime_validation": str(RUNTIME_VALIDATION),
                "runtime_future_gt_used": False,
                "posthoc_gt_used": False,
                "training_authorized": False,
                "production_authorized": False,
            },
        )
        return 1
    # GT is deliberately loaded only here, after the runtime validator has
    # checked every event/variant/frame artifact.
    event_metrics = [score_event(event, rows_by_event[str(event["event_id"])]) for event in events]
    aggregate_metrics = {str(horizon): aggregate(event_metrics, horizon) for horizon in HORIZONS}
    by_action = {
        action: {str(horizon): aggregate(event_metrics, horizon, action=action) for horizon in HORIZONS}
        for action in sorted({str(event["action_type"]) for event in events})
    }
    score_change_count = 0
    assignment_change_count = 0
    true_correct = 0
    true_incorrect = 0
    for item in event_metrics:
        for value in item["horizons"].values():
            score_change_count += sum(
                1
                for detail in value["frame_details"]
                if detail["assignment_change_type"] != "UNCHANGED"
            )
            assignment_change_count += int(value["assignment_change_count"])
            true_correct += int(value["true_correct_crossing_count"])
            true_incorrect += int(value["true_incorrect_crossing_count"])
    if true_correct > 0 and true_incorrect == 0:
        gate_status = "PASS_TVC_V0_CORRECT_CROSSING_ROUTE_TO_EXPANDED_VALIDATION"
        route = "EXPANDED_VALIDATION"
    elif assignment_change_count > 0 and true_correct == 0:
        gate_status = "FAIL_TVC_V0_SCORE_OR_ASSIGNMENT_CHANGE_WITHOUT_TRUE_CORRECT_CROSSING_ROUTE_TO_FEATURE_OR_BOUNDARY_DIAGNOSIS"
        route = "FEATURE_OR_ASSOCIATION_BOUNDARY_DIAGNOSIS"
    else:
        gate_status = "FAIL_TVC_V0_NO_CORRECT_CROSSING_ROUTE_TO_FEATURE_OR_BOUNDARY_DIAGNOSIS"
        route = "FEATURE_OR_ASSOCIATION_BOUNDARY_DIAGNOSIS"
    metrics = {
        "schema_version": "N72R5_STAGE03_TVC_V0_METRICS_V1",
        "status": "PASS_STAGE03_POSTHOC_SCORING",
        "event_count": len(events),
        "independent_sequence_count": len({str(event["sequence"]) for event in events}),
        "variants": list(TVC_VARIANTS),
        "horizons": list(HORIZONS),
        "event_metrics": event_metrics,
        "aggregate": aggregate_metrics,
        "by_action": by_action,
        "mechanism_counts": {
            "score_or_assignment_change_observations_across_horizons": score_change_count,
            "assignment_change_observations_across_horizons": assignment_change_count,
            "true_correct_crossing_observations_across_horizons": true_correct,
            "true_incorrect_crossing_observations_across_horizons": true_incorrect,
        },
        "runtime_validation": validation,
        "runtime_future_gt_used": False,
        "gt_usage": "posthoc_only_after_runtime_artifact_validation",
        "interaction_source": "simulated_from_gt",
        "not_real_human_evidence": True,
    }
    atomic_json(METRICS_PATH, metrics)
    gate = {
        "schema_version": "N72R5_STAGE03_TVC_V0_GATE_V1",
        "status": gate_status,
        "route": route,
        "runtime_valid": True,
        "posthoc_scored": True,
        "event_count": len(events),
        "independent_sequence_count": len({str(event["sequence"]) for event in events}),
        "true_correct_crossing_count": true_correct,
        "true_incorrect_crossing_count": true_incorrect,
        "assignment_change_count": assignment_change_count,
        "training_authorized": False,
        "production_authorized": False,
        "runtime_future_gt_used": False,
        "interaction_source": "simulated_from_gt",
        "not_real_human_evidence": True,
        "next_step": "feature_separability_and_association_boundary_diagnosis_before_any_learned_TVC",
    }
    atomic_json(GATE_PATH, gate)
    atomic_json(
        STAGE_STATUS,
        {
            "schema_version": "N72R5_STAGE_STATUS_V1",
            "stage": "03_TVC_V0_TARGET_VS_COMPETITOR",
            "status": gate_status,
            "protocol": str(PROTOCOL_PATH),
            "runtime_manifest": str(RUNTIME_MANIFEST),
            "runtime_validation": str(RUNTIME_VALIDATION),
            "metrics": str(METRICS_PATH),
            "gate": str(GATE_PATH),
            "event_count": len(events),
            "independent_sequence_count": len({str(event["sequence"]) for event in events}),
            "runtime_future_gt_used": False,
            "posthoc_gt_used": True,
            "interaction_source": "simulated_from_gt",
            "not_real_human_evidence": True,
            "training_authorized": False,
            "production_authorized": False,
        },
    )
    print(json.dumps({"status": gate_status, "metrics": str(METRICS_PATH), "gate": str(GATE_PATH)}, sort_keys=True), flush=True)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--event-limit", type=int, default=0)
    args = parser.parse_args()
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
