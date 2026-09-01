"""Isolated N42 T1 pairwise calibration head and frozen feature contract.

The module is intentionally independent of the production associator.  It
operates on the already recorded base/memory/fused score audit and returns a
bounded preference signal for a human-specified public-ID column.  It never
uses GT at inference time.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn


FEATURE_NAMES = (
    "base_gap_left_minus_right",
    "memory_gap_left_minus_right",
    "appearance_delta_gap_left_minus_right",
    "fused_gap_left_minus_right",
    "left_center_x_norm",
    "left_center_y_norm",
    "left_width_norm",
    "left_height_norm",
    "right_center_x_norm",
    "right_center_y_norm",
    "right_width_norm",
    "right_height_norm",
    "center_dx_norm",
    "center_dy_norm",
    "log_area_ratio_left_right",
    "left_confidence",
    "right_confidence",
    "left_native_age_norm",
    "right_native_age_norm",
    "frame_offset_norm",
    "candidate_count_norm",
    "left_rank_norm",
    "right_rank_norm",
)
FEATURE_DIM = len(FEATURE_NAMES)
FRAME_WIDTH = 1280.0
FRAME_HEIGHT = 720.0


class PairwiseCalibrationHead(nn.Module):
    """Small T1 scorer: positive logit means left candidate is preferable."""

    def __init__(self, input_dim: int = FEATURE_DIM, hidden1: int = 64, hidden2: int = 32) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden1),
            nn.ReLU(),
            nn.Linear(hidden1, hidden2),
            nn.ReLU(),
            nn.Linear(hidden2, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features).squeeze(-1)


def feature_contract() -> dict[str, Any]:
    return {
        "feature_names": list(FEATURE_NAMES),
        "input_dim": FEATURE_DIM,
        "frame_width": FRAME_WIDTH,
        "frame_height": FRAME_HEIGHT,
        "normalization": {
            "native_age": "min(age/2000,1)",
            "frame_offset": "min(offset/100,1)",
            "candidate_count": "min(count/20,1)",
            "rank": "index/max(candidate_count-1,1)",
            "box": "x/1280,y/720,width/1280,height/720",
        },
        "runtime_gt_used": False,
    }


def _matrix(audit: dict[str, Any], key: str) -> np.ndarray:
    value = np.asarray(audit.get(key, []), dtype=np.float64)
    if value.ndim != 2 or not np.all(np.isfinite(value)):
        raise ValueError(f"invalid {key} matrix: {value.shape}")
    return value


def _box_values(candidate: dict[str, Any]) -> tuple[float, float, float, float, float, float, float]:
    box = np.asarray(candidate.get("box", []), dtype=float).reshape(-1)
    if box.size != 4 or not np.all(np.isfinite(box)):
        raise ValueError("candidate box is invalid")
    x1, y1, x2, y2 = [float(x) for x in box]
    width, height = max(0.0, x2 - x1), max(0.0, y2 - y1)
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    return cx, cy, width, height, float(candidate.get("confidence", 1.0)), float(candidate.get("native_age", 0.0)), float(candidate.get("index", 0))


def pair_feature_from_audit(
    audit: dict[str, Any],
    left_index: int,
    right_index: int,
    state_public_id: int,
    frame_offset: int,
) -> np.ndarray | None:
    """Build the exact 23-D causal feature used in training and inference."""
    candidates = audit.get("candidates", [])
    if not isinstance(candidates, list):
        return None
    by_index = {int(item.get("index", i)): item for i, item in enumerate(candidates) if isinstance(item, dict)}
    if int(left_index) not in by_index or int(right_index) not in by_index or int(left_index) == int(right_index):
        return None
    order = [int(value) for value in audit.get("public_id_order", [])]
    try:
        column = order.index(int(state_public_id))
    except ValueError:
        return None
    base = _matrix(audit, "base_scores_before_appearance")
    memory = _matrix(audit, "appearance_memory_scores")
    delta = _matrix(audit, "appearance_score_deltas")
    fused = _matrix(audit, "fused_scores")
    li, ri = int(left_index), int(right_index)
    if max(li, ri) >= base.shape[0] or column >= base.shape[1] or base.shape != memory.shape or base.shape != delta.shape or base.shape != fused.shape:
        return None
    left, right = by_index[li], by_index[ri]
    lcx, lcy, lw, lh, lconf, lage, lrank = _box_values(left)
    rcx, rcy, rw, rh, rconf, rage, rrank = _box_values(right)
    n_candidates = max(1, len(candidates))
    feature = np.asarray(
        [
            base[li, column] - base[ri, column],
            memory[li, column] - memory[ri, column],
            delta[li, column] - delta[ri, column],
            fused[li, column] - fused[ri, column],
            lcx / FRAME_WIDTH,
            lcy / FRAME_HEIGHT,
            lw / FRAME_WIDTH,
            lh / FRAME_HEIGHT,
            rcx / FRAME_WIDTH,
            rcy / FRAME_HEIGHT,
            rw / FRAME_WIDTH,
            rh / FRAME_HEIGHT,
            (lcx - rcx) / FRAME_WIDTH,
            (lcy - rcy) / FRAME_HEIGHT,
            float(np.log((lw * lh + 1.0) / (rw * rh + 1.0))),
            lconf,
            rconf,
            min(max(lage, 0.0) / 2000.0, 1.0),
            min(max(rage, 0.0) / 2000.0, 1.0),
            min(max(float(frame_offset), 0.0) / 100.0, 1.0),
            min(n_candidates / 20.0, 1.0),
            lrank / max(n_candidates - 1, 1),
            rrank / max(n_candidates - 1, 1),
        ],
        dtype=np.float32,
    )
    if feature.size != FEATURE_DIM or not np.all(np.isfinite(feature)):
        return None
    return feature


def checkpoint_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_checkpoint(path: Path, device: str = "cpu") -> tuple[PairwiseCalibrationHead, dict[str, Any]]:
    payload = torch.load(path, map_location=device, weights_only=False)
    if not isinstance(payload, dict) or payload.get("protocol") != "N42_T1_PAIRWISE_CALIBRATION_V1":
        raise ValueError("invalid N42 T1 checkpoint protocol")
    model = PairwiseCalibrationHead(input_dim=int(payload.get("input_dim", FEATURE_DIM)))
    model.load_state_dict(payload["state_dict"])
    model.to(device)
    model.eval()
    return model, payload


def preference_from_model(model: PairwiseCalibrationHead, features: np.ndarray) -> np.ndarray:
    values = np.asarray(features, dtype=np.float32)
    if values.ndim != 2 or values.shape[1] != FEATURE_DIM:
        raise ValueError(f"invalid feature batch {values.shape}")
    with torch.no_grad():
        logits = model(torch.as_tensor(values)).detach().cpu().numpy()
    if not np.all(np.isfinite(logits)):
        raise ValueError("T1 logits are nonfinite")
    return (2.0 / (1.0 + np.exp(-np.clip(logits, -40.0, 40.0))) - 1.0).astype(np.float32).reshape(-1)


def model_metadata(protocol_hash: str, dataset_hash: str, seed: int) -> dict[str, Any]:
    return {
        "protocol": "N42_T1_PAIRWISE_CALIBRATION_V1",
        "input_dim": FEATURE_DIM,
        "feature_contract": feature_contract(),
        "architecture": "Linear(23,64)-ReLU-Linear(64,32)-ReLU-Linear(32,1)",
        "calibration_application": "mean ordered-pair preference for each candidate added only to the direct human-specified target public-ID column; fixed scale=1.0; hard-negative entries remain -1e8",
        "training_protocol_sha256": protocol_hash,
        "training_dataset_sha256": dataset_hash,
        "seed": int(seed),
        "runtime_future_gt_used": False,
    }
