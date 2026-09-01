#!/usr/bin/env python3
"""Isolated N48 512-D risk-aware assignment diagnostic utilities."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.n36_real_eval_common import DATA_ROOT  # noqa: E402
from scripts.n43_full_matrix_common import iou  # noqa: E402
from scripts.n47_global_probe_common import (  # noqa: E402
    HARD_NEGATIVE,
    N42_RUNTIME,
    N43_MAP,
    event_map,
    load,
    normalize_assignment,
    sha256,
    write_json,
)

N36_FRAMES = ROOT / "outputs/n36/real_tape/frames"
N47_RUNTIME = ROOT / "outputs/n47_global_probe/repair1_swap_metric/replay/runtime"
N42_PROTOCOL = ROOT / "outputs/n42/training/training_protocol.json"
N48_OUT = ROOT / "outputs/n48"
N48_TRAIN = N48_OUT / "training"
SEED = 4848
FEATURE_DIM = 512
SCALAR_DIM = 8
PIDS = "public_id_order"


def sequence_split() -> dict[str, str]:
    payload = load(N42_PROTOCOL)
    result = {}
    for split in ("train", "validation", "holdout"):
        for seq in payload["sequence_split"][split]:
            result[str(seq)] = split
    return result


def load_n36_sequence(sequence: str) -> dict[int, dict[str, Any]]:
    path = N36_FRAMES / f"{sequence}.jsonl"
    if not path.is_file():
        raise FileNotFoundError(path)
    return {int(item["frame"]): item for line in path.read_text(encoding="utf-8").splitlines() if line.strip() for item in [json.loads(line)]}


def candidate_features(n42_candidates: list[dict[str, Any]], n36_frame: dict[str, Any]) -> np.ndarray:
    source = n36_frame.get("candidates", [])
    result = []
    used: set[int] = set()
    for candidate in n42_candidates:
        box = np.asarray(candidate["box"], dtype=np.float64)
        matches = []
        for index, item in enumerate(source):
            if index in used:
                continue
            other = np.asarray(item["box"], dtype=np.float64)
            matches.append((float(np.max(np.abs(box - other))), index, item))
        if not matches:
            raise ValueError("candidate feature match missing")
        error, index, item = min(matches, key=lambda x: x[0])
        if error > 1e-3:
            raise ValueError(f"candidate box mismatch {error}")
        vector = np.asarray(item.get("machine_embedding", []), dtype=np.float32).reshape(-1)
        if vector.size != FEATURE_DIM or not np.all(np.isfinite(vector)) or float(np.linalg.norm(vector)) <= 1e-6:
            raise ValueError("candidate machine embedding invalid")
        used.add(index)
        result.append(vector / float(np.linalg.norm(vector)))
    return np.asarray(result, dtype=np.float32)


def gt_boxes(gt_frame: Any) -> dict[int, Any]:
    return {int(gid): box for gid, box in zip(gt_frame.gt_ids, gt_frame.boxes)} if gt_frame is not None else {}


def make_memory_snapshot(event: dict[str, Any], pids: list[int], n36_frames: dict[int, dict[str, Any]], gt: dict[int, Any], n42_event_frame: dict[str, Any]) -> tuple[dict[int, np.ndarray], dict[int, bool]]:
    source = n36_frames.get(int(event["frame"]))
    if source is None:
        raise ValueError(f"missing N36 event frame {event['event_id']}")
    source_candidates = source.get("candidates", [])
    memories: dict[int, np.ndarray] = {}
    valid: dict[int, bool] = {}
    boxes = gt_boxes(gt.get(int(event["frame"])))
    mapping = {int(pid): int(gid) for pid, gid in load(N43_MAP)["public_to_gt_mapping"].get(str(event["event_id"]), {}).items()}
    for pid in pids:
        target = boxes.get(mapping.get(int(pid)))
        best = None
        if target is not None:
            for item in source_candidates:
                value = float(iou(item["box"], target))
                if best is None or value > best[0]:
                    best = (value, item)
        if best is not None and best[0] >= 0.5:
            vector = np.asarray(best[1].get("machine_embedding", []), dtype=np.float32).reshape(-1)
            if vector.size == FEATURE_DIM and np.all(np.isfinite(vector)) and float(np.linalg.norm(vector)) > 1e-6:
                memories[int(pid)] = vector / float(np.linalg.norm(vector)); valid[int(pid)] = True; continue
        memories[int(pid)] = np.zeros(FEATURE_DIM, dtype=np.float32); valid[int(pid)] = False
    return memories, valid


def scalar_features(base: np.ndarray, candidates: list[dict[str, Any]], memory_valid: np.ndarray, frame_offset: int) -> np.ndarray:
    output = []
    for row in range(base.shape[0]):
        values = base[row]
        finite = values > HARD_NEGATIVE
        ordered = np.sort(values[finite])[::-1]
        margin = float(ordered[0] - ordered[1]) if len(ordered) > 1 else float(ordered[0] - (-1.0e8)) if len(ordered) else 0.0
        confidence = float(np.clip(candidates[row].get("confidence", 0.0), 0.0, 1.0))
        age = min(max(float(candidates[row].get("native_age", 0.0)), 0.0) / 2000.0, 1.0)
        rank = float(row) / max(base.shape[0] - 1, 1)
        output.append([float(np.tanh(np.max(values[finite]) / 5.0)) if np.any(finite) else -1.0, float(np.tanh(margin / 5.0)), confidence, age, rank, min(max(float(frame_offset), 0.0) / 100.0, 1.0), float(np.mean(memory_valid)), 1.0])
    return np.asarray(output, dtype=np.float32)


class RiskAware512FusionHead(nn.Module):
    def __init__(self, projection_dim: int = 64):
        super().__init__()
        self.candidate = nn.Sequential(nn.Linear(FEATURE_DIM, projection_dim), nn.ReLU())
        self.memory = nn.Sequential(nn.Linear(FEATURE_DIM, projection_dim), nn.ReLU())
        self.trunk = nn.Sequential(nn.Linear(projection_dim * 3 + SCALAR_DIM, 128), nn.ReLU(), nn.Linear(128, 64), nn.ReLU())
        self.residual = nn.Linear(64, 1)
        self.uncertainty = nn.Linear(64, 1)

    def forward(self, candidate: torch.Tensor, memory: torch.Tensor, scalars: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        c = self.candidate(candidate)
        m = self.memory(memory)
        h = self.trunk(torch.cat((c, m, torch.abs(c - m), scalars), dim=-1))
        return self.residual(h).squeeze(-1), self.uncertainty(h).squeeze(-1)


def solve_with_none(scores: np.ndarray) -> np.ndarray:
    matrix = np.asarray(scores, dtype=np.float32)
    work = matrix.copy(); work[work <= HARD_NEGATIVE] = -1.0e8 - 16.0
    expanded = np.concatenate((work, np.full((matrix.shape[0], matrix.shape[0]), -1.0e8, dtype=np.float32)), axis=1)
    rows, cols = linear_sum_assignment(-expanded)
    result = np.full(matrix.shape[0], -1, dtype=np.int64); result[rows] = cols
    return normalize_assignment(result, matrix.shape[1])


def _solve_with_none_expanded(scores: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """Return the row assignment including dummy columns and its total score."""
    matrix = np.asarray(scores, dtype=np.float32)
    if matrix.ndim != 2 or not np.all(np.isfinite(matrix)):
        raise ValueError("invalid global assignment score matrix")
    working = matrix.copy()
    working[working <= HARD_NEGATIVE] = -1.0e8 - 16.0
    expanded = np.concatenate((working, np.full((matrix.shape[0], matrix.shape[0]), -1.0e8, dtype=np.float32)), axis=1)
    rows, columns = linear_sum_assignment(-expanded)
    assignment = np.full(matrix.shape[0], -1, dtype=np.int64)
    assignment[rows] = columns
    total = float(sum(float(expanded[row, col]) for row, col in zip(rows, columns)))
    return assignment, expanded, total


def global_assignment_margin(scores: np.ndarray) -> float:
    """Exact gap to the best one-row-forced alternative, including explicit NONE."""
    matrix = np.asarray(scores, dtype=np.float32)
    assignment, expanded, baseline_total = _solve_with_none_expanded(matrix)
    n, p = matrix.shape
    best_alternative = -float("inf")
    for row in range(n):
        current = int(assignment[row])
        alternatives = [column for column in range(p) if column != current]
        if current < p:
            alternatives.append(p)  # one representative NONE dummy
        for forced in alternatives:
            remaining_rows = [index for index in range(n) if index != row]
            remaining_columns = [index for index in range(p + n) if index != forced]
            forced_value = float(expanded[row, forced])
            if not remaining_rows:
                candidate_total = forced_value
            else:
                sub = expanded[np.ix_(remaining_rows, remaining_columns)]
                rr, cc = linear_sum_assignment(-sub)
                candidate_total = forced_value + float(sum(float(sub[r, c]) for r, c in zip(rr, cc)))
            best_alternative = max(best_alternative, candidate_total)
    return float(baseline_total - best_alternative) if best_alternative != -float("inf") else float("inf")


def load_checkpoint(path: Path, device: str = "cpu") -> tuple[RiskAware512FusionHead, dict[str, Any]]:
    payload = torch.load(path, map_location=device, weights_only=False)
    if payload.get("protocol") not in {"N48_RISK_AWARE_512D_GLOBAL_ASSIGNMENT_DIAGNOSTIC_V1", "N48_R1_RISK_AWARE_512D_WITH_CELL_BCE_V1", "N48_R1_REPAIR2_SINGLE_OBJECTIVE_V1"} or payload.get("production_authorized") is not False:
        raise ValueError("invalid/non-authorized N48 checkpoint")
    model = RiskAware512FusionHead(int(payload.get("projection_dim", 64))); model.load_state_dict(payload["state_dict"]); model.to(device).eval()
    return model, payload


def runtime_sidecar(model: RiskAware512FusionHead, candidate: np.ndarray, memories: np.ndarray, scalars: np.ndarray, base: np.ndarray, memory_valid: np.ndarray, device: str = "cpu") -> dict[str, Any]:
    with torch.no_grad():
        raw, uncertainty_raw = model(torch.as_tensor(candidate, dtype=torch.float32, device=device), torch.as_tensor(memories, dtype=torch.float32, device=device), torch.as_tensor(scalars, dtype=torch.float32, device=device))
    residual = (0.25 * torch.tanh(raw)).cpu().numpy().astype(np.float32)
    uncertainty = torch.sigmoid(uncertainty_raw).cpu().numpy().astype(np.float32)
    accepted = np.zeros_like(residual, dtype=bool)
    baseline_assignment = solve_with_none(base)
    global_margin = global_assignment_margin(base)
    finite = base > HARD_NEGATIVE
    # The frozen gate is evaluated against the whole baseline assignment gap.
    for row in range(base.shape[0]):
        for col in range(base.shape[1]):
            index = row * base.shape[1] + col
            accepted[index] = bool(finite[row, col] and memory_valid[col] and global_margin <= 2.0 and abs(float(residual[index])) >= 0.05 and float(uncertainty[index]) <= 0.35)
    adjusted = base.copy()
    flat = adjusted.reshape(-1); flat[accepted] += residual[accepted]
    adjusted = flat.reshape(base.shape)
    return {"raw_residual": raw.cpu().numpy().astype(float).tolist(), "bounded_residual": residual.astype(float).tolist(), "uncertainty": uncertainty.astype(float).tolist(), "accepted": accepted.tolist(), "adjusted_scores": adjusted.astype(float).tolist(), "baseline_assignment": baseline_assignment, "global_assignment_margin": global_margin, "plus_assignment": solve_with_none(adjusted), "runtime_future_gt_used": False}


def checkpoint_sha(path: Path) -> str:
    return sha256(path)
