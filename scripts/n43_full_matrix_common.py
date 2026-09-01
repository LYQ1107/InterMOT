"""N43 independent full candidate x public-ID calibration sidecar.

This module deliberately has no import path into the production associator.  It
turns the frozen N41/N42 audit into a complete cell interface and applies a
bounded residual to every finite candidate/public-ID cell.  GT is not needed
for feature construction or inference; it is used only by the dataset and
post-hoc evaluation scripts.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from torch import nn


NONE_SCORE = -1.0e8
HARD_NEGATIVE = -1.0e7
FEATURE_NAMES = (
    "base_score_tanh",
    "memory_score_tanh",
    "appearance_delta_tanh",
    "fused_score_tanh",
    "geometry_iou_to_native_ref",
    "motion_compatibility_from_previous_native",
    "cell_margin_tanh",
    "cell_reliability",
    "candidate_age_norm",
    "candidate_confidence",
    "candidate_center_x_norm",
    "candidate_center_y_norm",
    "candidate_width_norm",
    "candidate_height_norm",
    "candidate_rank_norm",
    "state_has_native_ref",
    "state_native_age_norm",
    "frame_offset_norm",
)
FEATURE_DIM = len(FEATURE_NAMES)
FRAME_WIDTH = 1280.0
FRAME_HEIGHT = 720.0
PROTOCOL = "N43_FULL_MATRIX_CALIBRATION_SIDECAR_V1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def finite_matrix(audit: dict[str, Any], key: str) -> np.ndarray:
    aliases = {
        "base_scores_before_appearance": "base_scores",
        "appearance_score_deltas": "appearance_delta_scores",
    }
    raw = audit.get(key, audit.get(aliases.get(key, "__missing__"), []))
    value = np.asarray(raw, dtype=np.float64)
    if value.ndim != 2 or not np.all(np.isfinite(value)):
        raise ValueError(f"{key} is not a finite matrix: {value.shape}")
    return value


def box(candidate: dict[str, Any]) -> np.ndarray:
    value = np.asarray(candidate.get("box", []), dtype=np.float64).reshape(-1)
    if value.size != 4 or not np.all(np.isfinite(value)):
        raise ValueError("invalid/nonfinite candidate box")
    return value


def iou(left: Any, right: Any) -> float:
    a, b = np.asarray(left, dtype=float), np.asarray(right, dtype=float)
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter
    return float(inter / union) if union > 0 else 0.0


def candidate_map(audit: dict[str, Any]) -> dict[int, dict[str, Any]]:
    output = {}
    for pos, item in enumerate(audit.get("candidates", [])):
        if not isinstance(item, dict):
            continue
        index = int(item.get("index", pos))
        output[index] = item
    return output


def state_reference_boxes(audit: dict[str, Any]) -> dict[int, dict[str, Any]]:
    """Map each public ID to the candidate carrying its native ID, if any."""
    candidates = candidate_map(audit)
    by_native = {
        int(item["native_tid"]): item
        for item in candidates.values()
        if item.get("native_tid") is not None and _valid_box(item.get("box"))
    }
    refs = {}
    for pid, native in (audit.get("public_id_to_native_tid") or {}).items():
        if native is not None and int(native) in by_native:
            refs[int(pid)] = by_native[int(native)]
    return refs


def previous_boxes(previous_audit: dict[str, Any] | None) -> dict[int, dict[str, Any]]:
    if not previous_audit:
        return {}
    return {int(item["native_tid"]): item for item in audit_candidates(previous_audit) if item.get("native_tid") is not None and _valid_box(item.get("box"))}


def audit_candidates(audit: dict[str, Any]) -> list[dict[str, Any]]:
    return [item for item in audit.get("candidates", []) if isinstance(item, dict)]


def _valid_box(value: Any) -> bool:
    array = np.asarray(value if value is not None else [], dtype=float).reshape(-1)
    return bool(array.size == 4 and np.all(np.isfinite(array)) and array[2] > array[0] and array[3] > array[1])


def cell_features(
    audit: dict[str, Any],
    row_index: int,
    column_index: int,
    frame_offset: int,
    previous_audit: dict[str, Any] | None = None,
) -> np.ndarray:
    base = finite_matrix(audit, "base_scores_before_appearance")
    memory = finite_matrix(audit, "appearance_memory_scores")
    delta = finite_matrix(audit, "appearance_score_deltas")
    fused = finite_matrix(audit, "fused_scores")
    if not (base.shape == memory.shape == delta.shape == fused.shape):
        raise ValueError("score matrix shape mismatch")
    if row_index < 0 or row_index >= base.shape[0] or column_index < 0 or column_index >= base.shape[1]:
        raise IndexError((row_index, column_index, base.shape))
    candidates = candidate_map(audit)
    candidate = candidates.get(int(row_index))
    if candidate is None:
        raise ValueError(f"missing candidate index {row_index}")
    current_box = box(candidate)
    refs = state_reference_boxes(audit)
    pids = [int(value) for value in audit.get("public_id_order", [])]
    pid = pids[column_index]
    reference = refs.get(pid)
    geometry = iou(current_box, box(reference)) if reference is not None and reference is not candidate else 0.0
    previous = previous_boxes(previous_audit)
    native = candidate.get("native_tid")
    if native is not None and int(native) in previous:
        old = box(previous[int(native)])
        center = (current_box[:2] + current_box[2:]) / 2.0
        old_center = (old[:2] + old[2:]) / 2.0
        distance = float(np.linalg.norm((center - old_center) / np.asarray([FRAME_WIDTH, FRAME_HEIGHT])))
        motion = float(np.exp(-distance / 0.05))
    else:
        motion = 0.0
    row_values = base[row_index]
    # Keep column coordinates attached to the compressed validity mask.  The
    # previous implementation searched for the current *value* in valid_row;
    # repeated scores and hard-negative columns could therefore remove the
    # wrong entry or index into an empty array.
    valid_columns = np.flatnonzero(row_values > HARD_NEGATIVE)
    if row_values[column_index] > HARD_NEGATIVE:
        other_columns = valid_columns[valid_columns != int(column_index)]
        best_other = float(np.max(row_values[other_columns])) if other_columns.size else 0.0
        margin = float(row_values[column_index] - best_other)
    else:
        best_other = 0.0
        margin = -10.0
    confidence = float(candidate.get("confidence", 0.0))
    age = max(0.0, float(candidate.get("native_age", 0.0)))
    rank = float(candidate.get("index", row_index)) / max(len(candidates) - 1, 1)
    ref_age = max(0.0, float(reference.get("native_age", 0.0))) if reference is not None else 0.0
    width, height = current_box[2] - current_box[0], current_box[3] - current_box[1]
    feature = np.asarray(
        [
            np.tanh(float(base[row_index, column_index]) / 5.0),
            np.tanh(float(memory[row_index, column_index]) / 2.0),
            np.tanh(float(delta[row_index, column_index]) / 2.0),
            np.tanh(float(fused[row_index, column_index]) / 5.0),
            geometry,
            motion,
            np.tanh(margin / 5.0),
            float(bool(candidate.get("feature_available", candidate.get("has_feat", False)))) * float(np.clip(confidence, 0.0, 1.0)),
            min(age / 2000.0, 1.0),
            float(np.clip(confidence, 0.0, 1.0)),
            float(((current_box[0] + current_box[2]) / 2.0) / FRAME_WIDTH),
            float(((current_box[1] + current_box[3]) / 2.0) / FRAME_HEIGHT),
            float(width / FRAME_WIDTH),
            float(height / FRAME_HEIGHT),
            rank,
            float(reference is not None),
            min(ref_age / 2000.0, 1.0),
            min(max(float(frame_offset), 0.0) / 100.0, 1.0),
        ],
        dtype=np.float32,
    )
    if feature.shape != (FEATURE_DIM,) or not np.all(np.isfinite(feature)):
        raise ValueError("nonfinite N43 cell feature")
    return feature


class FullMatrixCalibrationHead(nn.Module):
    """Bounded full-cell gate/residual head.

    The output is not an absolute replacement score: ``utility`` is bounded
    and applied to every valid cell as ``base + sigmoid(gate)*appearance_delta
    + residual``.  The NONE sentinel bypasses the model.
    """

    def __init__(self, input_dim: int = FEATURE_DIM, hidden1: int = 64, hidden2: int = 32) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(input_dim, hidden1), nn.ReLU(), nn.Linear(hidden1, hidden2), nn.ReLU(), nn.Linear(hidden2, 2))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features)


def bounded_utility(model: FullMatrixCalibrationHead, features: np.ndarray, appearance_delta: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    values = np.asarray(features, dtype=np.float32)
    appearance = np.asarray(appearance_delta, dtype=np.float32).reshape(-1)
    if values.ndim != 2 or values.shape[1] != FEATURE_DIM or appearance.shape[0] != values.shape[0]:
        raise ValueError("invalid N43 model input")
    with torch.no_grad():
        raw = model(torch.as_tensor(values)).detach().cpu().numpy()
    if not np.all(np.isfinite(raw)):
        raise ValueError("N43 model output is nonfinite")
    gate = 1.0 / (1.0 + np.exp(-np.clip(raw[:, 0], -40.0, 40.0)))
    residual = 0.5 * np.tanh(np.clip(raw[:, 1], -40.0, 40.0))
    utility = gate * appearance + residual
    return utility.astype(np.float32), gate.astype(np.float32), residual.astype(np.float32)


def hungarian_with_none(scores: np.ndarray) -> np.ndarray:
    matrix = np.asarray(scores, dtype=np.float32)
    if matrix.ndim != 2 or not np.all(np.isfinite(matrix)):
        raise ValueError("invalid score matrix")
    none = np.full((matrix.shape[0], matrix.shape[0]), NONE_SCORE, dtype=np.float32)
    expanded = np.concatenate([matrix, none], axis=1)
    rows, columns = linear_sum_assignment(-expanded)
    output = np.full(matrix.shape[0], -1, dtype=int)
    output[rows] = columns
    return output


def checkpoint_digest(path: Path) -> str:
    return sha256(path)


def feature_contract() -> dict[str, Any]:
    return {
        "feature_names": list(FEATURE_NAMES),
        "input_dim": FEATURE_DIM,
        "none_semantics": {"score": NONE_SCORE, "model_bypassed": True, "one_dummy_per_candidate": True},
        "hard_negative_sentinel": HARD_NEGATIVE,
        "runtime_gt_used": False,
        "public_id_in_feature": False,
        "target_identity_in_feature": False,
        "future_outcome_in_feature": False,
    }


def load_checkpoint(path: Path, device: str = "cpu") -> tuple[FullMatrixCalibrationHead, dict[str, Any]]:
    payload = torch.load(path, map_location=device, weights_only=False)
    if not isinstance(payload, dict) or payload.get("protocol") != PROTOCOL:
        raise ValueError("invalid N43 checkpoint protocol")
    model = FullMatrixCalibrationHead(input_dim=int(payload.get("input_dim", FEATURE_DIM)))
    model.load_state_dict(payload["state_dict"])
    model.to(device)
    model.eval()
    return model, payload


def cell_key(audit: dict[str, Any], row: int, col: int) -> tuple[int, int]:
    pids = [int(value) for value in audit.get("public_id_order", [])]
    return int(row), int(pids[col])


def apply_sidecar(
    audit: dict[str, Any],
    model: FullMatrixCalibrationHead,
    frame_offset: int,
    previous_audit: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return a new full-cell score/assignment audit; never mutates input."""
    import copy

    output = copy.deepcopy(audit)
    base = finite_matrix(audit, "base_scores_before_appearance")
    delta = finite_matrix(audit, "appearance_score_deltas")
    features = np.asarray(
        [cell_features(audit, i, j, frame_offset, previous_audit) for i in range(base.shape[0]) for j in range(base.shape[1])],
        dtype=np.float32,
    )
    utility, gate, residual = bounded_utility(model, features, delta.reshape(-1))
    utility_matrix = utility.reshape(base.shape)
    adjusted = base + utility_matrix
    hard = base <= HARD_NEGATIVE
    adjusted[hard] = base[hard]
    assignment = hungarian_with_none(adjusted)
    pids = [int(value) for value in audit.get("public_id_order", [])]
    mapped = [pids[int(column)] if 0 <= int(column) < len(pids) else None for column in assignment.tolist()]
    output["n43_sidecar"] = {
        "enabled": True,
        "protocol": PROTOCOL,
        "application": "all candidate x public-ID finite cells",
        "cell_count": int(base.size),
        "changed_cell_count": int(np.sum(np.abs(adjusted - base) > 1.0e-12)),
        "changed_column_count": int(np.sum(np.any(np.abs(adjusted - base) > 1.0e-12, axis=0))),
        "hard_negative_preserved": bool(np.all(adjusted[hard] == base[hard])),
        "none_score": NONE_SCORE,
        "none_assignments": int(np.sum(assignment >= len(pids))),
        "gate_min": float(np.min(gate)) if gate.size else None,
        "gate_max": float(np.max(gate)) if gate.size else None,
        "residual_min": float(np.min(residual)) if residual.size else None,
        "residual_max": float(np.max(residual)) if residual.size else None,
        "runtime_future_gt_used": False,
    }
    output["fused_scores_before_n43"] = base.astype(float).tolist()
    output["fused_scores"] = adjusted.astype(float).tolist()
    output["scores"] = adjusted.astype(float).tolist()
    output["public_id_score_matrix"] = adjusted.T.astype(float).tolist()
    output["public_id_fused_score_matrix"] = adjusted.T.astype(float).tolist()
    output["assignment_before_n43"] = list(np.asarray(audit.get("assignment_after_scope", audit.get("assignment", [])), dtype=int).tolist())
    output["assignment_after_n43_with_none"] = assignment.astype(int).tolist()
    output["assignment_after_scope"] = [int(value) for value in assignment.tolist()]
    output["assignment"] = [int(value) for value in assignment.tolist()]
    output["candidate_public_ids"] = mapped
    output["candidate_public_id_mapping_complete"] = bool(all(value is not None for value in mapped))
    output["runtime_future_gt_used"] = False
    return output
