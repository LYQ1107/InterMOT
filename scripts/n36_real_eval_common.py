"""Shared CPU-side helpers for the N36 real event/full-loop evaluations."""

from __future__ import annotations

import json
import os
import tempfile
from copy import deepcopy
from pathlib import Path
from typing import Any, Iterable

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = Path("/path/to/dancetrack")
FEATURE_DIM = 512
HORIZONS = (20, 50, 100)


def jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.astype(float).tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    return value


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_name, path)
    finally:
        if os.path.exists(temp_name):
            os.unlink(temp_name)


def atomic_json(path: Path, payload: Any) -> None:
    atomic_text(
        path,
        json.dumps(jsonable(payload), indent=2, sort_keys=True, allow_nan=False) + "\n",
    )


def atomic_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    atomic_text(
        path,
        "".join(
            json.dumps(jsonable(row), sort_keys=True, allow_nan=False) + "\n"
            for row in rows
        ),
    )


def load_manifest(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("status") != "PASS":
        raise RuntimeError(f"event manifest is not PASS: {payload.get('status')}")
    return payload


def observation_from_candidate(candidate: dict[str, Any], index: int) -> dict[str, Any]:
    feature = np.asarray(candidate.get("machine_embedding"), dtype=np.float32).reshape(-1)
    if feature.size != FEATURE_DIM or not np.all(np.isfinite(feature)):
        raise ValueError(f"machine feature invalid at candidate index {index}")
    norm = float(np.linalg.norm(feature))
    if norm <= 1e-6:
        raise ValueError(f"machine feature is zero at candidate index {index}")
    return {
        "obs_id": int(candidate.get("candidate_index", index)),
        "box": np.asarray(candidate["box"], dtype=float).copy(),
        "feat": feature / norm,
        "has_feat": 1.0,
        "native_tid": int(candidate.get("sequence_global_native_id", candidate["native_tid"])),
        "native_age": float(candidate.get("native_age", 0.0)),
        "conf": float(candidate.get("confidence", 1.0)),
    }


def replay_candidate(candidate: dict[str, Any], index: int) -> dict[str, Any]:
    """Convert a runtime candidate to the restricted replay schema."""
    obs = observation_from_candidate(candidate, index)
    return {
        "obs_id": int(obs["obs_id"]),
        "box": obs["box"].tolist(),
        "embedding": obs["feat"].tolist(),
        "native_tid": int(obs["native_tid"]),
        "native_age": float(obs["native_age"]),
        "confidence": float(obs["conf"]),
    }


def iter_rows(path: Path, start: int | None = None, end: int | None = None):
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            frame = int(row["frame"])
            if start is not None and frame < int(start):
                continue
            if end is not None and frame > int(end):
                break
            yield line_no, row


def event_source_path(item: dict[str, Any]) -> Path:
    return ROOT / str(item["source_tape"])


def observations_for_row(row: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        observation_from_candidate(candidate, index)
        for index, candidate in enumerate(row.get("candidates", []))
    ]


def build_replay_tape(item: dict[str, Any], horizon: int = 100) -> dict[str, Any]:
    event = deepcopy(item["event"])
    event_frame = int(event["frame"])
    frames = []
    expected_end = min(
        int(item["sequence_frame_count"]) - 1,
        event_frame + int(horizon),
    )
    for _line_no, row in iter_rows(event_source_path(item), event_frame + 1, expected_end):
        frames.append(
            {
                "frame": int(row["frame"]),
                "candidates": [
                    replay_candidate(candidate, index)
                    for index, candidate in enumerate(row.get("candidates", []))
                ],
                "candidate_complete": True,
                "candidate_set_complete": True,
            }
        )
    return {
        "protocol": "N36_REAL_CCAM_PAIRED_REPLAY_TAPE",
        "synthetic": False,
        "sequence": event["sequence"],
        "interaction_source": "simulated_from_gt",
        "runtime_gt_used": False,
        "future_gt_used_runtime": False,
        "candidate_complete": True,
        "candidate_set_complete": True,
        "prefix_state": deepcopy(item["prefix_state"]),
        "event": event,
        "frames": frames,
        "source_candidate_tape": item["source_tape"],
        "event_frame": event_frame,
        "future_frame_end": expected_end,
    }


def finite_iou(left: Any, right: Any) -> float:
    a = np.asarray(left, dtype=float).reshape(-1)
    b = np.asarray(right, dtype=float).reshape(-1)
    if a.size != 4 or b.size != 4 or not np.all(np.isfinite(a)) or not np.all(np.isfinite(b)):
        return 0.0
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter
    return float(inter / union) if union > 0 else 0.0


def assign_predictions(
    rows: list[list[Any]] | list[tuple[int, Any]],
    gt_ids: list[int],
    gt_boxes: list[Any],
    threshold: float = 0.5,
) -> dict[int, tuple[int, float]]:
    """Match output public IDs to GT only for post-hoc metric computation."""
    predictions = []
    for row in rows:
        try:
            predictions.append((int(row[0]), np.asarray(row[1], dtype=float)))
        except (TypeError, ValueError, IndexError):
            continue
    if not predictions or not gt_ids:
        return {}
    scores = np.asarray(
        [[finite_iou(box, gt_box) for gt_box in gt_boxes] for _pid, box in predictions],
        dtype=float,
    )
    pairs: list[tuple[int, int]] = []
    try:
        from scipy.optimize import linear_sum_assignment

        row_indices, col_indices = linear_sum_assignment(-scores)
        pairs = list(zip(row_indices.tolist(), col_indices.tolist()))
    except Exception:
        candidates = sorted(
            (
                float(scores[pred_index, gt_index]),
                int(pred_index),
                int(gt_index),
            )
            for pred_index in range(scores.shape[0])
            for gt_index in range(scores.shape[1])
        )
        pairs = []
        used_predictions: set[int] = set()
        used_gt: set[int] = set()
        for score, pred_index, gt_index in reversed(candidates):
            if pred_index in used_predictions or gt_index in used_gt:
                continue
            used_predictions.add(pred_index)
            used_gt.add(gt_index)
            pairs.append((pred_index, gt_index))
    output: dict[int, tuple[int, float]] = {}
    for pred_index, gt_index in pairs:
        score = float(scores[pred_index, gt_index])
        if score >= float(threshold):
            output[int(gt_ids[gt_index])] = (int(predictions[pred_index][0]), score)
    return output


def gt_box_map(gt_frames: dict[int, Any], frame: int) -> tuple[list[int], list[np.ndarray]]:
    gt = gt_frames.get(int(frame))
    if gt is None:
        return [], []
    return [int(value) for value in gt.gt_ids], [np.asarray(box, dtype=float) for box in gt.boxes]


def target_box(gt_frames: dict[int, Any], frame: int, gid: int) -> np.ndarray | None:
    gt = gt_frames.get(int(frame))
    if gt is None:
        return None
    for current_gid, box in zip(gt.gt_ids, gt.boxes):
        if int(current_gid) == int(gid):
            return np.asarray(box, dtype=float)
    return None


def evaluate_trace(
    trace: list[dict[str, Any]],
    gt_frames: dict[int, Any],
    event: dict[str, Any],
    horizons: Iterable[int] = HORIZONS,
    match_threshold: float = 0.5,
) -> dict[str, Any]:
    """Compute post-hoc identity metrics without feeding labels to replay."""
    target_gid = int(event["dataset_gt_id"])
    target_pid = int(event["public_id"])
    all_horizon_metrics: dict[str, Any] = {}
    per_gt_by_horizon: dict[str, Any] = {}
    for horizon in horizons:
        selected = trace[: int(horizon)]
        target_ious: list[float] = []
        target_missing = 0
        target_visible = 0
        switches = 0
        last_pid_by_gt: dict[int, int] = {}
        error_active = False
        recorrection_opportunities = 0
        per_gt: dict[int, dict[str, Any]] = {}
        first_correct_latency = None
        for entry in selected:
            frame = int(entry["frame"])
            rows = entry.get("rows", [])
            gt_ids, gt_boxes = gt_box_map(gt_frames, frame)
            assignment = assign_predictions(rows, gt_ids, gt_boxes, threshold=match_threshold)
            for gid, (pid, _iou) in assignment.items():
                previous = last_pid_by_gt.get(gid)
                if previous is not None and previous != int(pid):
                    switches += 1
                last_pid_by_gt[gid] = int(pid)
            for gid, box in zip(gt_ids, gt_boxes):
                record = per_gt.setdefault(
                    int(gid),
                    {"visible_frames": 0, "missing": 0, "ious": [], "id_switches": 0},
                )
                record["visible_frames"] += 1
                match = assignment.get(int(gid))
                if match is None:
                    record["missing"] += 1
                else:
                    record["ious"].append(float(match[1]))
            box = target_box(gt_frames, frame, target_gid)
            if box is None:
                continue
            target_visible += 1
            target_rows = [row for row in rows if int(row[0]) == target_pid]
            target_iou = max((finite_iou(row[1], box) for row in target_rows), default=0.0)
            target_ious.append(float(target_iou))
            target_is_error = not target_rows or target_iou < match_threshold
            if target_is_error:
                target_missing += 1
            if target_is_error and not error_active:
                recorrection_opportunities += 1
            error_active = target_is_error
            if first_correct_latency is None and not target_is_error:
                first_correct_latency = frame - int(event["frame"])
        for gid, record in per_gt.items():
            visible = int(record["visible_frames"])
            record["mean_iou"] = (
                float(np.mean(record["ious"])) if record["ious"] else None
            )
            record["missing_rate"] = (
                float(record["missing"] / visible) if visible else None
            )
            del record["ious"]
        per_gt_by_horizon[str(horizon)] = per_gt
        all_horizon_metrics[str(horizon)] = {
            "target_gt_id": target_gid,
            "target_public_id": target_pid,
            "visible_frames": int(target_visible),
            "target_mean_iou": float(np.mean(target_ious)) if target_ious else None,
            "target_missing_rate": (
                float(target_missing / target_visible) if target_visible else None
            ),
            "id_switch_count": int(switches),
            "posthoc_recorrection_opportunity_count": int(recorrection_opportunities),
            "recovery_latency_frames": (
                None if first_correct_latency is None else int(first_correct_latency)
            ),
        }
    return {
        "horizons": all_horizon_metrics,
        "per_gt": per_gt_by_horizon,
        "metric_semantics": {
            "source": "offline GT post-hoc scoring only",
            "target_iou": "IoU of the event public_id row against the event dataset GT identity",
            "missing_rate": "target public_id row absent or below IoU threshold on visible GT frames",
            "recorrection_count": "posthoc contiguous identity-error opportunities, not observed human clicks",
            "id_switch_count": "posthoc Hungarian GT-to-public matching at IoU threshold",
        },
    }


def variant_config(name: str):
    from sam3_intermot.association.state_manager import StateManagerConfig

    specs = {
        "M0": {
            "description": "K1 only; CCAM disabled",
            "use_appearance_memory": False,
            "appearance_anchor_cap": 0,
            "appearance_negative_cap": 0,
        },
        "M1": {
            "description": "human EMA prototype only",
            "use_appearance_memory": True,
            "appearance_anchor_cap": 0,
            "appearance_negative_cap": 0,
        },
        "M2": {
            "description": "human EMA prototype plus positive anchors",
            "use_appearance_memory": True,
            "appearance_anchor_cap": 8,
            "appearance_negative_cap": 0,
        },
        "M3": {
            "description": "positive anchors plus negative competitor bank",
            "use_appearance_memory": True,
            "appearance_anchor_cap": 8,
            "appearance_negative_cap": 16,
        },
        "M4": {
            "description": "M3 plus reliability/age gate",
            "use_appearance_memory": True,
            "appearance_anchor_cap": 8,
            "appearance_negative_cap": 16,
            "appearance_reliability_threshold": 0.5,
            "appearance_decay_frames": 60.0,
        },
    }
    if name not in specs:
        raise KeyError(name)
    spec = specs[name]
    config = StateManagerConfig(
        variant="reid",
        score_threshold=-100.0,
        max_lost_gap=90,
        **{key: value for key, value in spec.items() if key != "description"},
    )
    return config, spec["description"]
