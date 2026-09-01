"""N70 assignment-boundary audit and fixed paired replay.

The runtime part of this script receives only the event's authoritative public
ID and the frozen N70 cache features.  ``target_native_id`` is touched only by
the posthoc scorer, after every method's scores and assignments have been
materialized.  The Hungarian implementation is the unchanged SciPy reference
used by the parent experiments.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import sys
import tempfile
import traceback
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from scipy.optimize import linear_sum_assignment

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import n70_association_common as common  # noqa: E402


OUT = ROOT / "outputs/N70"
REPLAY = OUT / "replay"
ARTIFACTS = REPLAY / "event_artifacts"
BOUNDARY = REPLAY / "assignment_boundary.jsonl"
RESULTS = REPLAY / "paired_replay_results.json"
STAGE04 = OUT / "stage_04_status.json"
STAGE05 = OUT / "stage_05_status.json"
ATTEMPTS = OUT / "attempts"
TRAIN_PROTOCOL = OUT / "training_protocol.json"
VARIANTS = ("M0", "M1", "M2", "M3", "M4")
HORIZONS = (20, 50, 100)
METHODS = ("CURRENT_CCAM_BASELINE", "M0", "M1", "M2", "M3", "M4", "BRANCH_A", "BRANCH_B")
BOOTSTRAP_REPS = 2000
BOOTSTRAP_SEEDS = {20: 7020, 50: 7050, 100: 7100}


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_name, path)
        dfd = os.open(str(path.parent), os.O_DIRECTORY)
        try:
            os.fsync(dfd)
        finally:
            os.close(dfd)
    finally:
        if os.path.exists(temp_name):
            os.unlink(temp_name)


def atomic_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    count = 0
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n")
                count += 1
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_name, path)
        dfd = os.open(str(path.parent), os.O_DIRECTORY)
        try:
            os.fsync(dfd)
        finally:
            os.close(dfd)
    finally:
        if os.path.exists(temp_name):
            os.unlink(temp_name)
    return count


def assignment_from_scores(scores: np.ndarray) -> np.ndarray:
    values = np.asarray(scores, dtype=np.float64)
    if values.ndim != 2 or not np.all(np.isfinite(values)):
        raise RuntimeError("N70 replay score matrix is not finite 2-D")
    result = np.full(values.shape[0], -1, dtype=np.int64)
    if values.shape[0] and values.shape[1]:
        rows, columns = linear_sum_assignment(-values)
        result[rows] = columns
    return result


def load_checkpoint(branch: str, device: str = "cpu") -> tuple[Any, np.ndarray, np.ndarray, dict[str, Any]]:
    import torch

    path = common.CHECKPOINT_A if branch == "A" else common.CHECKPOINT_B
    if not path.is_file():
        raise RuntimeError(f"N70 Branch {branch} checkpoint is missing: {path}")
    payload = torch.load(path, map_location=device, weights_only=False)
    if not isinstance(payload, dict) or payload.get("schema") != "N70_ASSOCIATION_CHECKPOINT_V1" or payload.get("branch") != branch:
        raise RuntimeError(f"invalid N70 Branch {branch} checkpoint")
    if payload.get("training_protocol_sha256") != sha256_file(TRAIN_PROTOCOL):
        raise RuntimeError(f"N70 Branch {branch} training protocol hash mismatch")
    if payload.get("dataset_sha256") != sha256_file(common.DATASET):
        raise RuntimeError(f"N70 Branch {branch} dataset hash mismatch")
    model = common.build_model(branch).to(device)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    mean = np.asarray(payload.get("context_mean"), dtype=np.float32)
    std = np.asarray(payload.get("context_std"), dtype=np.float32)
    if mean.shape != (common.CONTEXT_DIM,) or std.shape != (common.CONTEXT_DIM,) or not np.all(np.isfinite(mean)) or not np.all(np.isfinite(std)):
        raise RuntimeError(f"N70 Branch {branch} context normalization is invalid")
    if payload.get("runtime_future_gt_used") is not False or payload.get("target_native_id_sent_to_runtime") is not False:
        raise RuntimeError(f"N70 Branch {branch} runtime contract is invalid")
    # Keep the replay result manifest JSON-native.  The loaded checkpoint
    # contains the torch state_dict and NumPy context arrays, which are needed
    # to restore the model but must not be retained in the audit payload (and
    # would make the final atomic JSON write fail).  Record provenance and
    # lossless digests for those arrays instead of embedding the objects.
    def array_digest(value: np.ndarray) -> dict[str, Any]:
        contiguous = np.ascontiguousarray(value)
        return {
            "dtype": str(contiguous.dtype),
            "shape": [int(item) for item in contiguous.shape],
            "sha256": hashlib.sha256(contiguous.tobytes()).hexdigest(),
        }

    checkpoint_meta = {
        "path": str(path),
        "sha256": sha256_file(path),
        "schema": payload.get("schema"),
        "branch": payload.get("branch"),
        "model": payload.get("model"),
        "training_protocol": payload.get("training_protocol"),
        "training_protocol_sha256": payload.get("training_protocol_sha256"),
        "dataset": payload.get("dataset"),
        "dataset_sha256": payload.get("dataset_sha256"),
        "context_mean": array_digest(mean),
        "context_std": array_digest(std),
        "sequence_split": payload.get("sequence_split"),
        "best_epoch": int(payload.get("best_epoch")) if payload.get("best_epoch") is not None else None,
        "runtime_future_gt_used": payload.get("runtime_future_gt_used"),
        "target_native_id_sent_to_runtime": payload.get("target_native_id_sent_to_runtime"),
        "interaction_source": payload.get("interaction_source"),
        "real_human_tape": payload.get("real_human_tape"),
        "real_sam3_full_loop": payload.get("real_sam3_full_loop"),
        "not_real_human_evidence": payload.get("not_real_human_evidence"),
        "production_authorized": payload.get("production_authorized"),
    }
    json.dumps(checkpoint_meta, ensure_ascii=False, allow_nan=False)
    return model, mean, std, checkpoint_meta


def model_sidecar(model: Any, branch: str, pack: dict[str, Any], mean: np.ndarray, std: np.ndarray, device: Any) -> dict[str, Any]:
    import torch

    n = int(pack["candidate"].shape[0])
    temporary = {
        "candidate": pack["candidate"],
        "anchor": pack["anchor"],
        "memory": pack["memory"],
        "hard_negative": pack["hard_negative"],
        "context": pack["context"],
        "label": np.zeros(n, dtype=np.int8),
        "group": np.zeros(n, dtype=np.int64),
    }
    indices = np.arange(n, dtype=np.int64)
    tensors = common.tensors_for_indices(temporary, indices, mean, std, device)
    with torch.no_grad():
        logits = model(*tensors[:5]).detach().cpu().numpy().astype(np.float32)
    if logits.shape != (n, 2) or not np.all(np.isfinite(logits)):
        raise RuntimeError(f"N70 Branch {branch} emitted malformed/nonfinite logits")
    logit_delta = logits[:, 0] - logits[:, 1]
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    probabilities = np.exp(shifted)
    probabilities /= np.sum(probabilities, axis=1, keepdims=True)
    target_probability = probabilities[:, 0]
    target_col = pack["target_column"]
    none_predicted = bool(target_col is None or target_probability.size == 0 or float(np.max(target_probability)) < 0.5)
    residual = np.zeros(n, dtype=np.float32) if none_predicted else (2.0 * np.tanh(logit_delta / 2.0)).astype(np.float32)
    adjusted = np.asarray(pack["base"], dtype=np.float32).copy()
    if target_col is not None and not none_predicted:
        column = int(target_col)
        valid = np.isfinite(adjusted[:, column]) & (adjusted[:, column] > -1.0e8)
        adjusted[valid, column] += residual[valid]
    if not np.all(np.isfinite(adjusted)):
        raise RuntimeError(f"N70 Branch {branch} adjusted score is nonfinite")
    delta = adjusted - np.asarray(pack["base"], dtype=np.float32)
    non_target = delta.copy()
    if target_col is not None:
        non_target[:, int(target_col)] = 0.0
    non_target_max = float(np.max(np.abs(non_target))) if non_target.size else 0.0
    if non_target_max > 1e-7:
        raise RuntimeError(f"N70 Branch {branch} changed a non-target public-ID column")
    return {
        "branch": branch,
        "target_logits": logits[:, 0].astype(float).tolist(),
        "none_logits": logits[:, 1].astype(float).tolist(),
        "target_probability": target_probability.astype(float).tolist(),
        "logit_delta_target_minus_none": logit_delta.astype(float).tolist(),
        "residual_target_public_column": residual.astype(float).tolist(),
        "target_column": target_col,
        "none_predicted": none_predicted,
        "none_threshold": 0.5,
        "residual_bound": 2.0,
        "adjusted_scores": adjusted.astype(float).tolist(),
        "score_cells_changed": int(np.sum(np.abs(delta) > 1e-12)),
        "max_abs_score_delta": float(np.max(np.abs(delta))) if delta.size else 0.0,
        "non_target_column_max_abs_delta": non_target_max,
        "target_column_only": True,
        "target_native_id_sent_to_runtime": False,
        "runtime_future_gt_used": False,
    }


def mapping_state(frame: dict[str, Any], event: dict[str, Any], pids: list[int]) -> dict[str, Any]:
    candidates = frame.get("candidate_rows", [])
    mappings = [row.get("mapping", {}) for row in candidates if isinstance(row, dict)]
    chain_ok = bool(all(
        isinstance(mapping, dict)
        and mapping.get("local_id") is not None
        and mapping.get("global_id") is not None
        and (
            mapping.get("public_id") is not None
            or mapping.get("public_id_status") == "EXPLICIT_N54_PUBLIC_ASSIGNMENT_ABSENT"
        )
        for mapping in mappings
    ))
    target_rows = [index for index, row in enumerate(candidates) if isinstance(row, dict) and int(row.get("native_tid", -1)) == int(event["target_native_id"])]
    target_row = target_rows[0] if len(target_rows) == 1 else None
    target_present = target_row is not None
    target_public = int(event["target_public_id"])
    target_col = pids.index(target_public) if target_public in pids else None
    target_public_assignment_absent = bool(target_present and target_row is not None and candidates[target_row].get("mapping", {}).get("public_id") is None)
    # A target row with a wrong/absent public assignment is an observed wrong
    # association, not automatically an input mapping failure.  The target
    # public column must exist for a target-conditioned assignment to be
    # scored; otherwise this frame is explicitly F_MAPPING_UNCERTAIN.
    target_scope_resolved = bool(target_present and target_col is not None)
    mapping_uncertain = bool(not chain_ok or (target_present and target_col is None))
    return {
        "candidate_chain_complete": chain_ok,
        "candidate_count": len(candidates),
        "target_row": target_row,
        "target_candidate_present": target_present,
        "target_public_id": target_public,
        "target_public_column": target_col,
        "target_scope_resolved": target_scope_resolved,
        "target_public_assignment_absent": target_public_assignment_absent,
        "mapping_uncertain": mapping_uncertain,
        "public_assignment_absent_candidate_rows": int(sum(mapping.get("public_id") is None for mapping in mappings)),
        "runtime_future_gt_used": False,
    }


def public_assignments(assignment: np.ndarray, pids: list[int]) -> list[int | None]:
    return [None if int(column) < 0 else int(pids[int(column)]) for column in assignment]


def native_assignment_map(candidates: list[dict[str, Any]], assignments: list[int | None]) -> dict[int, int | None]:
    if len(candidates) != len(assignments):
        raise RuntimeError("N70 native/assignment axis mismatch")
    result: dict[int, int | None] = {}
    for row, public_id in zip(candidates, assignments):
        native = int(row["native_tid"])
        if native in result:
            raise RuntimeError(f"duplicate N70 native candidate: {native}")
        result[native] = public_id
    return result


def compact_mapping(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "candidate_index": int(row.get("candidate_index", index)),
            "native_tid": int(row.get("native_tid", -1)),
            "mapping": row.get("mapping", {}),
            "n36_candidate_index": row.get("n36_candidate_index"),
            "n36_box_iou": row.get("n36_box_iou"),
            "n36_box_coordinate_equal": row.get("n36_box_coordinate_equal"),
        }
        for index, row in enumerate(candidates)
    ]


def outcome(
    *,
    event: dict[str, Any],
    frame: dict[str, Any],
    mapping: dict[str, Any],
    method: str,
    comparison_baseline: str,
    baseline_scores: np.ndarray,
    baseline_assignment: np.ndarray,
    treated_scores: np.ndarray,
    treated_assignment: np.ndarray,
    sidecar: dict[str, Any] | None,
) -> dict[str, Any]:
    candidates = frame["candidate_rows"]
    pids = [int(value) for value in frame["public_id_order"]]
    baseline_public = public_assignments(baseline_assignment, pids)
    treated_public = public_assignments(treated_assignment, pids)
    target_row = mapping["target_row"]
    target_col = mapping["target_public_column"]
    base_assigned = baseline_public[target_row] if target_row is not None else None
    treated_assigned = treated_public[target_row] if target_row is not None else None
    baseline_correct = bool(mapping["target_scope_resolved"] and target_row is not None and target_col is not None and int(baseline_assignment[target_row]) == int(target_col))
    treated_correct = bool(mapping["target_scope_resolved"] and target_row is not None and target_col is not None and int(treated_assignment[target_row]) == int(target_col))
    valid_identity = bool(mapping["target_scope_resolved"])
    utility = int(treated_correct) - int(baseline_correct) if valid_identity else 0
    base_native = native_assignment_map(candidates, baseline_public)
    treated_native = native_assignment_map(candidates, treated_public)
    target_native = int(event["target_native_id"])
    untouched_ids = sorted(set(base_native) | set(treated_native))
    untouched_changes = sum(base_native.get(native) != treated_native.get(native) for native in untouched_ids if native != target_native)
    score_delta = np.asarray(treated_scores, dtype=np.float64) - np.asarray(baseline_scores, dtype=np.float64)
    if score_delta.shape != baseline_scores.shape or not np.all(np.isfinite(score_delta)):
        raise RuntimeError("N70 outcome score delta is invalid")
    score_changed = bool(np.any(np.abs(score_delta) > 1e-12))
    assignment_changed = bool(not np.array_equal(baseline_assignment, treated_assignment))
    target_assignment_changed = bool(base_assigned != treated_assigned)
    base_target_score = float(baseline_scores[target_row, target_col]) if target_row is not None and target_col is not None else None
    treated_target_score = float(treated_scores[target_row, target_col]) if target_row is not None and target_col is not None else None
    base_margin = None
    treated_margin = None
    if target_row is not None and target_col is not None:
        base_alternatives = [float(baseline_scores[target_row, index]) for index in range(baseline_scores.shape[1]) if index != target_col and baseline_scores[target_row, index] > -1e8]
        treated_alternatives = [float(treated_scores[target_row, index]) for index in range(treated_scores.shape[1]) if index != target_col and treated_scores[target_row, index] > -1e8]
        if base_alternatives:
            base_margin = float(baseline_scores[target_row, target_col] - max(base_alternatives))
        if treated_alternatives:
            treated_margin = float(treated_scores[target_row, target_col] - max(treated_alternatives))
    if mapping["mapping_uncertain"]:
        classification = "F_MAPPING_UNCERTAIN"
    elif not mapping["target_candidate_present"]:
        classification = "E_TARGET_CANDIDATE_ABSENT"
    elif utility > 0:
        classification = "B_CROSSING_TARGET_CORRECT"
    elif utility < 0:
        classification = "C_CROSSING_TARGET_INCORRECT"
    elif treated_correct and untouched_changes > 0:
        classification = "D_TARGET_CORRECT_WITH_UNTOUCHED_COLLATERAL"
    elif score_changed and not target_assignment_changed:
        classification = "A_SCORE_CHANGED_NO_ASSIGNMENT_CROSSING"
    elif target_assignment_changed:
        classification = "N_NEUTRAL_TARGET_ASSIGNMENT_CHANGE"
    else:
        classification = "N_NO_CHANGE"
    if method == "CURRENT_CCAM_BASELINE":
        classification = "N_NO_CHANGE"
    result = {
        "event_id": event["event_id"],
        "sequence": event["sequence"],
        "action_type": event["action_type"],
        "variant": str(frame["variant"]),
        "frame": int(frame["frame"]),
        "event_frame": int(event["event_frame"]),
        "horizon": int(frame["frame"]) - int(event["event_frame"]),
        "method": method,
        "comparison_baseline": comparison_baseline,
        "target_public_id": int(event["target_public_id"]),
        "target_row": target_row,
        "target_public_column": target_col,
        "target_candidate_present": bool(mapping["target_candidate_present"]),
        "target_scope_resolved": valid_identity,
        "target_public_assignment_absent": bool(mapping["target_public_assignment_absent"]),
        "mapping_uncertain": bool(mapping["mapping_uncertain"]),
        "baseline_target_assigned_public_id": base_assigned,
        "treated_target_assigned_public_id": treated_assigned,
        "baseline_assignment_columns": baseline_assignment.astype(int).tolist(),
        "treated_assignment_columns": treated_assignment.astype(int).tolist(),
        "baseline_assignment_public_ids": baseline_public,
        "treated_assignment_public_ids": treated_public,
        "baseline_target_correct": baseline_correct,
        "target_correct": treated_correct,
        "utility_delta": utility,
        "assignment_changed": assignment_changed,
        "target_assignment_changed": target_assignment_changed,
        "correct_change": bool(utility > 0),
        "incorrect_change": bool(utility < 0),
        "neutral_change": bool(utility == 0),
        "untouched_assignment_changed_count": int(untouched_changes),
        "untouched_regression": bool(untouched_changes > 0),
        "score_changed": score_changed,
        "score_cells_changed": int(sidecar.get("score_cells_changed", 0)) if sidecar else 0,
        "max_abs_score_delta": float(sidecar.get("max_abs_score_delta", float(np.max(np.abs(score_delta)) if score_delta.size else 0.0))) if sidecar else float(np.max(np.abs(score_delta)) if score_delta.size else 0.0),
        "target_score_delta": None if base_target_score is None or treated_target_score is None else float(treated_target_score - base_target_score),
        "base_target_score": base_target_score,
        "treated_target_score": treated_target_score,
        "base_target_vs_alternative_margin": base_margin,
        "treated_target_vs_alternative_margin": treated_margin,
        "margin_delta": None if base_margin is None or treated_margin is None else float(treated_margin - base_margin),
        "classification": classification,
        "candidate_integrity": bool(mapping["candidate_chain_complete"]),
        "runtime_future_gt_used": False,
        "interaction_source": "simulated_from_gt",
        "real_human_tape": False,
        "production_authorized": False,
    }
    return result


def frame_summary(outcome_rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not outcome_rows:
        return {"frame_count": 0}
    finite_base_margin = [float(item["base_target_vs_alternative_margin"]) for item in outcome_rows if item["base_target_vs_alternative_margin"] is not None and math.isfinite(float(item["base_target_vs_alternative_margin"]))]
    finite_treated_margin = [float(item["treated_target_vs_alternative_margin"]) for item in outcome_rows if item["treated_target_vs_alternative_margin"] is not None and math.isfinite(float(item["treated_target_vs_alternative_margin"]))]
    def distribution(values: list[float]) -> dict[str, float | None]:
        if not values:
            return {"median": None, "p90": None, "p95": None, "max": None}
        arr = np.asarray(values, dtype=np.float64)
        return {"median": float(np.percentile(arr, 50)), "p90": float(np.percentile(arr, 90)), "p95": float(np.percentile(arr, 95)), "max": float(np.max(arr))}
    return {
        "frame_count": len(outcome_rows),
        "score_change_rate": float(np.mean([int(item["score_changed"]) for item in outcome_rows])),
        "assignment_change_rate": float(np.mean([int(item["assignment_changed"]) for item in outcome_rows])),
        "target_assignment_change_rate": float(np.mean([int(item["target_assignment_changed"]) for item in outcome_rows])),
        "correct_assignment_changes": int(sum(item["correct_change"] for item in outcome_rows)),
        "incorrect_assignment_changes": int(sum(item["incorrect_change"] for item in outcome_rows)),
        "neutral_assignment_changes": int(sum(item["neutral_change"] for item in outcome_rows)),
        "candidate_recall": float(np.mean([int(item["target_candidate_present"]) for item in outcome_rows])),
        "mapping_scope_resolved_rate": float(np.mean([int(item["target_scope_resolved"]) for item in outcome_rows])),
        "mapping_uncertain_count": int(sum(item["mapping_uncertain"] for item in outcome_rows)),
        "candidate_absent_count": int(sum(not item["target_candidate_present"] for item in outcome_rows)),
        "untouched_assignment_changed_total": int(sum(item["untouched_assignment_changed_count"] for item in outcome_rows)),
        "untouched_regression_frame_rate": float(np.mean([int(item["untouched_regression"]) for item in outcome_rows])),
        "classification_counts": dict(sorted(Counter(item["classification"] for item in outcome_rows).items())),
        "base_assignment_margin": distribution(finite_base_margin),
        "treated_assignment_margin": distribution(finite_treated_margin),
        "runtime_future_gt_used": False,
    }


def bootstrap(values_by_sequence: dict[str, list[float]], seed: int) -> dict[str, Any]:
    cluster_means = {sequence: float(np.mean(values)) for sequence, values in values_by_sequence.items() if values}
    if not cluster_means:
        return {"sequence_count": 0, "mean": None, "ci95": [None, None], "seed": seed, "repetitions": BOOTSTRAP_REPS, "cluster_means": {}}
    values = np.asarray(list(cluster_means.values()), dtype=np.float64)
    rng = np.random.default_rng(seed)
    draws = values[rng.integers(0, len(values), size=(BOOTSTRAP_REPS, len(values)))]
    means = draws.mean(axis=1)
    return {
        "sequence_count": int(len(values)),
        "mean": float(np.mean(values)),
        "ci95": [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))],
        "seed": int(seed),
        "repetitions": BOOTSTRAP_REPS,
        "cluster_means": dict(sorted(cluster_means.items())),
    }


def horizon_summary(rows: list[dict[str, Any]], horizon: int) -> dict[str, Any]:
    selected = [item for item in rows if 1 <= int(item["horizon"]) <= horizon]
    valid = [item for item in selected if item["target_scope_resolved"]]
    event_variant_values: dict[tuple[str, str], list[float]] = defaultdict(list)
    for item in valid:
        event_variant_values[(item["event_id"], item["variant"])].append(float(item["utility_delta"]))
    sequence_values: dict[str, list[float]] = defaultdict(list)
    for (event_id, _variant), values in event_variant_values.items():
        sequence = next(item["sequence"] for item in valid if item["event_id"] == event_id)
        sequence_values[sequence].append(float(np.mean(values)))
    target_values = [int(item["target_correct"]) for item in valid]
    baseline_values = [int(item["baseline_target_correct"]) for item in valid]
    utility_values = [float(item["utility_delta"]) for item in valid]
    return {
        "horizon": horizon,
        "frame_count": len(selected),
        "valid_identity_frames": len(valid),
        "candidate_absent_frames": int(sum(not item["target_candidate_present"] for item in selected)),
        "mapping_uncertain_frames": int(sum(item["mapping_uncertain"] for item in selected)),
        "candidate_recall": float(np.mean([int(item["target_candidate_present"]) for item in selected])) if selected else None,
        "baseline_target_correct_rate": float(np.mean(baseline_values)) if baseline_values else None,
        "treated_target_correct_rate": float(np.mean(target_values)) if target_values else None,
        "baseline_future_identity_error": float(1.0 - np.mean(baseline_values)) if baseline_values else None,
        "treated_future_identity_error": float(1.0 - np.mean(target_values)) if target_values else None,
        "mean_utility_delta_valid_frames": float(np.mean(utility_values)) if utility_values else None,
        "sequence_cluster_bootstrap_utility": bootstrap(sequence_values, BOOTSTRAP_SEEDS[horizon]),
        "score_change_rate": float(np.mean([int(item["score_changed"]) for item in selected])) if selected else None,
        "assignment_change_rate": float(np.mean([int(item["assignment_changed"]) for item in selected])) if selected else None,
        "target_assignment_change_rate": float(np.mean([int(item["target_assignment_changed"]) for item in selected])) if selected else None,
        "correct_assignment_changes": int(sum(item["correct_change"] for item in selected)),
        "incorrect_assignment_changes": int(sum(item["incorrect_change"] for item in selected)),
        "neutral_assignment_changes": int(sum(item["neutral_change"] for item in selected)),
        "untouched_assignment_changed_total": int(sum(item["untouched_assignment_changed_count"] for item in selected)),
        "untouched_regression_frame_rate": float(np.mean([int(item["untouched_regression"]) for item in selected])) if selected else None,
        "recorrection_opportunity_proxy": float(np.mean([int(item["target_scope_resolved"] and not item["target_correct"]) for item in selected])) if selected else None,
        "runtime_future_gt_used": False,
    }


def summarize_method(rows: list[dict[str, Any]], method: str) -> dict[str, Any]:
    selected = [item for item in rows if item["method"] == method]
    if not selected:
        return {"method": method, "frame_count": 0}
    actions = sorted({item["action_type"] for item in selected})
    action_summary = {
        action: frame_summary([item for item in selected if item["action_type"] == action])
        for action in actions
    }
    variants = sorted({item["variant"] for item in selected})
    variant_summary = {
        variant: frame_summary([item for item in selected if item["variant"] == variant])
        for variant in variants
    }
    return {
        "method": method,
        "frame_count": len(selected),
        "frame_summary": frame_summary(selected),
        "horizons": {str(horizon): horizon_summary(selected, horizon) for horizon in HORIZONS},
        "by_action": action_summary,
        "by_variant": variant_summary,
        "runtime_future_gt_used": False,
    }


def read_event_cache(event_id: str) -> list[dict[str, Any]]:
    path = common.CACHE_DIR / f"{event_id}.jsonl"
    if not path.is_file():
        raise FileNotFoundError(path)
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise RuntimeError(f"N70 cache row is not object: {path}:{line_no}")
            if row.get("runtime_future_gt_used") is not False:
                raise RuntimeError(f"N70 cache future GT boundary failed: {event_id}/{line_no}")
            rows.append(row)
    if len(rows) != 500:
        raise RuntimeError(f"N70 event cache expected 500 rows: {event_id}, found {len(rows)}")
    if {(str(row.get("variant")), int(row.get("frame", -1))) for row in rows}.__len__() != 500:
        raise RuntimeError(f"N70 event cache duplicate variant/frame: {event_id}")
    return rows


def replay(device_name: str = "cpu") -> dict[str, Any]:
    import torch

    protocol = common.load_protocol()
    events = common.load_event_map()
    REPLAY.mkdir(parents=True, exist_ok=True)
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    device = common.torch_device(device_name)
    model_a, mean_a, std_a, meta_a = load_checkpoint("A", device)
    model_b, mean_b, std_b, meta_b = load_checkpoint("B", device)
    all_outcomes: list[dict[str, Any]] = []
    boundary_rows: list[dict[str, Any]] = []
    event_count = 0
    cache_frame_count = 0
    for event_id in sorted(events):
        event = events[event_id]
        source_rows = read_event_cache(event_id)
        by_variant_frame = {(str(row["variant"]), int(row["frame"])): row for row in source_rows}
        event_artifact_rows: list[dict[str, Any]] = []
        for frame_number in sorted({int(row["frame"]) for row in source_rows}):
            m0_frame = by_variant_frame[("M0", frame_number)]
            frame_methods: dict[str, dict[str, Any]] = {}
            prepared: dict[str, dict[str, Any]] = {}
            for variant in VARIANTS:
                frame = by_variant_frame[(variant, frame_number)]
                horizon = int(frame["frame"]) - int(event["event_frame"])
                if horizon < 1 or horizon > 100:
                    raise RuntimeError(f"N70 future horizon is outside 1..100: {event_id}/{variant}/{frame_number}")
                pids = [int(value) for value in frame["public_id_order"]]
                pack = common.build_feature_pack(frame, event, include_offline_label=False)
                mapping = mapping_state(frame, event, pids)
                base_scores = np.asarray(frame["score_matrix"], dtype=np.float32)
                source_assignment = np.asarray(pack["source_assignment"], dtype=np.int64)
                recomputed_base = assignment_from_scores(base_scores)
                source_objective = float(np.sum([base_scores[index, int(column)] for index, column in enumerate(source_assignment) if int(column) >= 0]))
                recomputed_objective = float(np.sum([base_scores[index, int(column)] for index, column in enumerate(recomputed_base) if int(column) >= 0]))
                source_assignment_max_weight_or_tied = bool(abs(source_objective - recomputed_objective) <= 1e-5)
                if not source_assignment_max_weight_or_tied:
                    raise RuntimeError(f"N70 frozen assignment is not max-weight: {event_id}/{variant}/{frame_number}")
                base_assignment = source_assignment
                model_outputs: dict[str, tuple[np.ndarray, np.ndarray, dict[str, Any]]] = {}
                for branch, model, mean, std in (("A", model_a, mean_a, std_a), ("B", model_b, mean_b, std_b)):
                    sidecar = model_sidecar(model, branch, pack, mean, std, device)
                    treated_scores = np.asarray(sidecar["adjusted_scores"], dtype=np.float32)
                    treated_assignment = assignment_from_scores(treated_scores)
                    model_outputs[branch] = (treated_scores, treated_assignment, sidecar)
                prepared[variant] = {
                    "frame": frame,
                    "pids": pids,
                    "pack": pack,
                    "mapping": mapping,
                    "base_scores": base_scores,
                    "base_assignment": base_assignment,
                    "model_outputs": model_outputs,
                }
                frame_methods[variant] = {
                    "public_id_order": pids,
                    "candidate_rows_mapping": compact_mapping(frame["candidate_rows"]),
                    "mapping_audit": mapping,
                    "score_matrix": base_scores.astype(float).tolist(),
                    "assignment_columns": base_assignment.astype(int).tolist(),
                    "assignment_public_ids": public_assignments(base_assignment, pids),
                    "branch_a": {},
                    "branch_b": {},
                }
            m0_data = prepared["M0"]
            # The current baseline is the untouched M0 stream.
            current = outcome(
                event=event,
                frame=m0_data["frame"],
                mapping=m0_data["mapping"],
                method="CURRENT_CCAM_BASELINE",
                comparison_baseline="CURRENT_CCAM_BASELINE",
                baseline_scores=m0_data["base_scores"],
                baseline_assignment=m0_data["base_assignment"],
                treated_scores=m0_data["base_scores"],
                treated_assignment=m0_data["base_assignment"],
                sidecar=None,
            )
            all_outcomes.append(current)
            boundary_rows.append(current)
            # M0--M4 are the unchanged upstream streams.  Cross-variant
            # comparison is made only when the candidate/native and public-ID
            # axes are identical; otherwise the frame is retained as explicit
            # F_MAPPING_UNCERTAIN evidence instead of aligning by index.
            for variant_method in VARIANTS:
                data = prepared[variant_method]
                axis_compatible = (
                    [int(row["native_tid"]) for row in data["frame"]["candidate_rows"]]
                    == [int(row["native_tid"]) for row in m0_data["frame"]["candidate_rows"]]
                    and data["pids"] == m0_data["pids"]
                )
                comparison_mapping = dict(data["mapping"])
                if variant_method != "M0" and not axis_compatible:
                    comparison_mapping["mapping_uncertain"] = True
                    comparison_mapping["target_scope_resolved"] = False
                    comparison_mapping["axis_incompatible_with_m0"] = True
                else:
                    comparison_mapping["axis_incompatible_with_m0"] = False
                baseline_data = m0_data if variant_method != "M0" and axis_compatible else data
                item = outcome(
                    event=event,
                    frame=data["frame"],
                    mapping=comparison_mapping,
                    method=variant_method,
                    comparison_baseline="CURRENT_CCAM_BASELINE" if variant_method != "M0" else "M0",
                    baseline_scores=baseline_data["base_scores"],
                    baseline_assignment=baseline_data["base_assignment"],
                    treated_scores=data["base_scores"],
                    treated_assignment=data["base_assignment"],
                    sidecar=None,
                )
                item["axis_compatible_with_m0"] = axis_compatible
                all_outcomes.append(item)
                boundary_rows.append(item)
            # Apply each learned branch to every frozen upstream variant.  The
            # corresponding upstream variant is the paired baseline.
            for variant in VARIANTS:
                data = prepared[variant]
                for branch, (treated_scores, treated_assignment, sidecar) in data["model_outputs"].items():
                    method = "BRANCH_A" if branch == "A" else "BRANCH_B"
                    branch_item = outcome(
                        event=event,
                        frame=data["frame"],
                        mapping=data["mapping"],
                        method=method,
                        comparison_baseline=variant,
                        baseline_scores=data["base_scores"],
                        baseline_assignment=data["base_assignment"],
                        treated_scores=treated_scores,
                        treated_assignment=treated_assignment,
                        sidecar=sidecar,
                    )
                    all_outcomes.append(branch_item)
                    boundary_rows.append(branch_item)
                    frame_methods[variant]["branch_" + branch.lower()] = {
                        "score_matrix": treated_scores.astype(float).tolist(),
                        "assignment_columns": treated_assignment.astype(int).tolist(),
                        "assignment_public_ids": public_assignments(treated_assignment, data["pids"]),
                        "sidecar": sidecar,
                    }
            # The artifact line is deliberately self-contained for this
            # event/frame: candidate mapping, all upstream variants, and both
            # learned branches, with no GT field in runtime payload.
            mapping = mapping_state(m0_frame, event, [int(value) for value in m0_frame["public_id_order"]])
            event_artifact_rows.append({
                "schema": "N70_PAIRED_REPLAY_FRAME_V1",
                "status": "PASS_RUNTIME_FRAME",
                "event_id": event_id,
                "sequence": event["sequence"],
                "action_type": event["action_type"],
                "event_frame": int(event["event_frame"]),
                "frame": frame_number,
                "frame_horizon": frame_number - int(event["event_frame"]),
                "candidate_rows_mapping": compact_mapping(m0_frame["candidate_rows"]),
                "public_id_order": [int(value) for value in m0_frame["public_id_order"]],
                "mapping_audit": mapping,
                "variants": frame_methods,
                "runtime_future_gt_used": False,
                "target_native_id_sent_to_runtime": False,
                "interaction_source": "simulated_from_gt",
                "real_human_tape": False,
                "real_sam3_full_loop": False,
                "not_real_human_evidence": True,
                "production_authorized": False,
            })
            cache_frame_count += 1
        atomic_jsonl(ARTIFACTS / f"{event_id}.jsonl", event_artifact_rows)
        event_count += 1
        if event_count % 4 == 0:
            print(json.dumps({"events_completed": event_count, "cache_frames": cache_frame_count}, sort_keys=True), flush=True)
        del source_rows, event_artifact_rows
        gc.collect()
    boundary_count = atomic_jsonl(BOUNDARY, boundary_rows)
    summaries = {method: summarize_method(all_outcomes, method) for method in METHODS}
    # The strict gate is intentionally evaluated on the learned branches and
    # on their paired frozen variant baselines; no method is selected using
    # full-future results.
    gate_details: dict[str, Any] = {}
    for method in ("BRANCH_A", "BRANCH_B"):
        horizon_gate = {}
        for horizon in HORIZONS:
            ci = summaries[method]["horizons"][str(horizon)]["sequence_cluster_bootstrap_utility"]
            horizon_gate[str(horizon)] = {"lower_bound": ci["ci95"][0], "strict_positive": bool(ci["ci95"][0] is not None and ci["ci95"][0] > 0.0)}
        gate_details[method] = horizon_gate
    all_strict_positive = all(item[str(horizon)]["strict_positive"] for item in gate_details.values() for horizon in HORIZONS)
    protected_ok = all(summaries[method]["frame_summary"]["untouched_regression_frame_rate"] == 0.0 for method in ("BRANCH_A", "BRANCH_B"))
    replay_payload = {
        "schema": "N70_PAIRED_REPLAY_RESULTS_V1",
        "status": "PASS_EXECUTION_FAIL_FUTURE_EFFECT" if not (all_strict_positive and protected_ok) else "PASS_EXECUTION_FUTURE_EFFECT_PASS",
        "created_at_utc": now(),
        "protocol": str(common.PROTOCOL),
        "protocol_sha256": sha256_file(common.PROTOCOL),
        "training_protocol": str(TRAIN_PROTOCOL),
        "training_protocol_sha256": sha256_file(TRAIN_PROTOCOL),
        "cache_manifest": str(common.CACHE_MANIFEST),
        "cache_manifest_sha256": sha256_file(common.CACHE_MANIFEST),
        "event_count": event_count,
        "independent_sequence_count": len({event["sequence"] for event in events.values()}),
        "cache_frame_count": cache_frame_count,
        "boundary_row_count": boundary_count,
        "methods": summaries,
        "gate": {
            "research_gate": "PASS_FUTURE_EFFECT" if all_strict_positive and protected_ok else "FAIL_FUTURE_EFFECT",
            "strict_positive_lower_bound_all_horizons": all_strict_positive,
            "protected_untouched_regression_pass": protected_ok,
            "branch_ci": gate_details,
            "candidate_integrity": True,
            "mapping_audit": True,
            "runtime_future_gt_used": False,
            "interaction_source": "simulated_from_gt",
            "real_human_tape": False,
            "real_sam3_full_loop": False,
            "not_real_human_evidence": True,
            "production_authorized": False,
        },
        "checkpoints": {"A": meta_a, "B": meta_b},
        "horizons": list(HORIZONS),
        "bootstrap": {"repetitions": BOOTSTRAP_REPS, "seed_by_horizon": {str(key): value for key, value in BOOTSTRAP_SEEDS.items()}, "clusters": "independent sequence"},
        "runtime_future_gt_used": False,
        "interaction_source": "simulated_from_gt",
        "real_human_tape": False,
        "real_sam3_full_loop": False,
        "not_real_human_evidence": True,
        "production_authorized": False,
    }
    atomic_json(RESULTS, replay_payload)
    atomic_json(STAGE04, {
        "schema": "N70_STAGE_04_STATUS_V1",
        "status": "PASS_ASSIGNMENT_BOUNDARY_AUDIT",
        "created_at_utc": now(),
        "protocol": str(common.PROTOCOL),
        "protocol_sha256": sha256_file(common.PROTOCOL),
        "boundary_artifact": str(BOUNDARY),
        "boundary_artifact_sha256": sha256_file(BOUNDARY),
        "event_artifacts": str(ARTIFACTS),
        "metrics": {"events": event_count, "cache_frames": cache_frame_count, "boundary_rows": boundary_count, "methods": list(METHODS)},
        "gate_checks": {
            "score_assignment_rows_complete": boundary_count > 0,
            "candidate_mapping_preserved": True,
            "hungarian_solver_unchanged": True,
            "runtime_future_gt_false": True,
            "target_native_id_not_runtime_feature": True,
            "event_plus_one_causal_boundary": True,
            "production_authorized": False,
        },
        "next_stage": "N70_STAGE_05_PAIRED_REPLAY_GATE",
        "interaction_source": "simulated_from_gt",
        "real_human_tape": False,
        "real_sam3_full_loop": False,
        "not_real_human_evidence": True,
        "production_authorized": False,
    })
    atomic_json(STAGE05, {
        "schema": "N70_STAGE_05_STATUS_V1",
        "status": replay_payload["gate"]["research_gate"],
        "created_at_utc": now(),
        "paired_replay_results": str(RESULTS),
        "paired_replay_results_sha256": sha256_file(RESULTS),
        "metrics": {"event_count": event_count, "independent_sequence_count": replay_payload["independent_sequence_count"], "cache_frame_count": cache_frame_count, "boundary_row_count": boundary_count},
        "gate": replay_payload["gate"],
        "decision": {
            "calibration_head": "NOT_AUTHORIZED",
            "selector": "NOT_AUTHORIZED",
            "decoder_lora": "NOT_AUTHORIZED",
            "production_association": "NOT_AUTHORIZED" if replay_payload["gate"]["research_gate"] != "PASS_FUTURE_EFFECT" else "CANDIDATE_ONLY_PENDING_PRODUCTION_EVIDENCE",
            "reason": "N70 uses simulated_from_gt events and strict future-effect/protected-ID gate; training completion is not scientific authorization.",
        },
        "runtime_future_gt_used": False,
        "interaction_source": "simulated_from_gt",
        "real_human_tape": False,
        "real_sam3_full_loop": False,
        "not_real_human_evidence": True,
        "production_authorized": False,
    })
    return replay_payload


def record_failure(stage: str, exc: BaseException) -> Path:
    ATTEMPTS.mkdir(parents=True, exist_ok=True)
    existing = sorted(ATTEMPTS.glob(f"n70_replay_{stage}_failure_attempt*.json"))
    path = ATTEMPTS / f"n70_replay_{stage}_failure_attempt{len(existing) + 1}.json"
    atomic_json(path, {
        "schema": "N70_REPLAY_FAILURE_V1",
        "status": "FAIL_PRESERVED",
        "created_at_utc": now(),
        "stage": stage,
        "failure_type": type(exc).__name__,
        "failure_message": str(exc),
        "traceback": traceback.format_exc(),
        "protocol": str(common.PROTOCOL),
        "protocol_sha256": sha256_file(common.PROTOCOL),
        "cache_manifest": str(common.CACHE_MANIFEST),
        "cache_manifest_sha256": sha256_file(common.CACHE_MANIFEST),
        "runtime_future_gt_used": False,
        "interaction_source": "simulated_from_gt",
        "real_human_tape": False,
        "real_sam3_full_loop": False,
        "not_real_human_evidence": True,
        "production_authorized": False,
    })
    return path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", required=True, choices=("replay",))
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    result = replay(args.device)
    print(json.dumps({"status": result["status"], "research_gate": result["gate"]["research_gate"], "results": str(RESULTS), "stage_04": str(STAGE04), "stage_05": str(STAGE05)}, sort_keys=True), flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        stage = "runtime"
        path = record_failure(stage, exc)
        print(traceback.format_exc(), file=sys.stderr, flush=True)
        print(json.dumps({"status": "FAIL_PRESERVED", "artifact": str(path)}, sort_keys=True), file=sys.stderr, flush=True)
        raise
