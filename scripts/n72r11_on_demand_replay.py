#!/usr/bin/env python3
"""N72R11 causal E0/E1/E2 replay with an on-demand live re-query.

The runtime half of this worker never opens dataset GT.  E0 is the sealed
N72R9 B0 stream, E1 uses the isolated N72R11 scorer/target-edge bridge, and
E2 adds :class:`LiveFutureRequeryController` only after the frozen causal
uncertainty rule fires.  Posthoc GT scoring starts only after all runtime
files have been atomically sealed.

The ``--smoke`` mode is an engineering check for the real controller and is
explicitly not a scientific effect result.  It uses a three-frame suffix and
an auditable deterministic probe selection when no trained N72R11 models are
provided.  Formal runs require both a V3 scorer and a train-only bridge
checkpoint.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
from datetime import datetime, timezone
import argparse
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import tempfile
import time
import traceback
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sam3_intermot.association.effect_assignment import solve_effect_assignment  # noqa: E402
from sam3_intermot.association.target_edge_bridge import (  # noqa: E402
    TargetEdgeBridge,
    build_target_edge_feature,
)
from sam3_intermot.reacquisition.live_requery_controller import (  # noqa: E402
    LiveFutureRequeryController,
)
from sam3_intermot.reacquisition.models.n72r11_temporal_v3 import (  # noqa: E402
    CANDIDATE_FEATURE_DIM,
    DISTRACTOR_SLOTS,
    LONG_TERM_TRUSTED_SLOTS,
    MEMORY_FEATURE_DIM,
    N72R11TemporalIdentityModel,
    RECENT_TRUSTED_SLOTS,
    SOURCE_FEATURE_DIM,
    TEMPORAL_FEATURE_DIM,
)
from sam3_intermot.reacquisition.target_candidate_pool import (  # noqa: E402
    FUTURE_FRAME_REQUERY,
    MAIN_B0_CANDIDATE,
    TARGET_SESSION_CURRENT_RAW,
    build_candidate_pool,
    build_candidate_pool_with_future_requery,
    serializable_candidate,
)
from sam3_intermot.reacquisition.target_id_features import candidate_feature_vector  # noqa: E402
from scripts import n72r9_temporal_replay as legacy  # noqa: E402
from scripts.n72r5_stage07_official_full_loop import (  # noqa: E402
    CHECKPOINT,
    DATA_ROOT,
    FrozenMachineOSNetN72R5,
    MACHINE_CHECKPOINT,
    image_files,
)
from sam3_intermot.backend.sam3_backend import Sam3Backend  # noqa: E402


PROTOCOL_PATH = ROOT / "outputs/N72R9/protocol.json"
DEFAULT_OUTPUT_ROOT = ROOT / "outputs/N72R11/stage_03_replay"
DEFAULT_SMOKE_ROOT = ROOT / "outputs/N72R11/stage_04_live_smoke"
HORIZON = 100
HORIZONS = (20, 50, 100)
UNCERTAINTY_MARGIN = 0.25
ADMISSION_SCORE = 0.50
ADMISSION_MARGIN = 0.20
BOOTSTRAP_SEED = 7211
BOOTSTRAP_REPETITIONS = 2000
SOURCE_NAMES = (
    MAIN_B0_CANDIDATE,
    TARGET_SESSION_CURRENT_RAW,
    "STATIC_EVENT_REQUERY",
    FUTURE_FRAME_REQUERY,
    "UNKNOWN",
)


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        directory_fd = os.open(path.parent, os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def atomic_json(path: Path, value: Any) -> None:
    atomic_write(path, json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n")


def atomic_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    atomic_write(
        path,
        "".join(json.dumps(dict(row), ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n" for row in rows),
    )


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
                raise TypeError(f"{path}:{line_number} is not an object")
            rows.append(value)
    return rows


def _path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def _finite(value: Any, label: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} is not finite")
    return result


def _unit(value: Any, label: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32).reshape(-1)
    if array.size != MEMORY_FEATURE_DIM or not np.all(np.isfinite(array)):
        raise ValueError(f"{label} is not finite {MEMORY_FEATURE_DIM}-D")
    norm = float(np.linalg.norm(array))
    if norm <= 1.0e-6:
        raise ValueError(f"{label} has zero norm")
    return (array / norm).astype(np.float32)


def _source_vector(source: str) -> np.ndarray:
    vector = np.zeros(len(SOURCE_NAMES), dtype=np.float32)
    try:
        vector[SOURCE_NAMES.index(str(source))] = 1.0
    except ValueError:
        vector[-1] = 1.0
    return vector


def _memory_array(values: Sequence[np.ndarray], slots: int) -> tuple[np.ndarray, np.ndarray]:
    array = np.zeros((int(slots), MEMORY_FEATURE_DIM), dtype=np.float32)
    mask = np.zeros(int(slots), dtype=np.bool_)
    for index, value in enumerate(list(values)[-int(slots):]):
        array[index] = _unit(value, f"memory[{index}]")
        mask[index] = True
    return array, mask


def _mean_feature(values: Sequence[np.ndarray], fallback: np.ndarray) -> np.ndarray:
    if not values:
        return _unit(fallback, "neighbor fallback")
    result = np.mean(np.stack([_unit(value, "neighbor feature") for value in values], axis=0), axis=0)
    norm = float(np.linalg.norm(result))
    return (result / max(norm, 1.0e-6)).astype(np.float32)


def _state_objects(pairs: Sequence[tuple[int, int]]) -> list[Any]:
    return [
        type(
            "N72R11ReplayState",
            (),
            {"association_state_id": int(state_id), "public_id": int(public_id)},
        )()
        for state_id, public_id in pairs
    ]


def _strip_target_authority(row: Mapping[str, Any]) -> dict[str, Any]:
    """Remove post-solver authority from a target-session input row.

    N72R6 stores the public ID in some derived C1 audit rows.  The frozen
    target stream itself is authority-free, but accepting both representations
    makes the adapter explicit and prevents accidental authority leakage into
    the candidate pool.
    """

    result = deepcopy(dict(row))
    for key in (
        "public_id",
        "public_id_authority",
        "solver_public_id",
        "solver_association_state_id",
        "assigned_public_id",
        "assignment_status",
        "solver_status",
        "solver_score",
    ):
        result.pop(key, None)
    result["public_id"] = None
    result["public_id_inference"] = False
    result["runtime_future_gt_used"] = False
    result["runtime_gt_read"] = False
    result["posthoc_gt_used"] = False
    return result


def _load_inputs(event: Mapping[str, Any], *, horizon: int) -> dict[str, Any]:
    """Load only frozen runtime inputs; no dataset GT is opened here."""

    inputs = legacy._load_rows(event)
    event_frame = int(inputs["event_frame"])
    if int(horizon) < 1 or int(horizon) > HORIZON:
        raise ValueError("horizon must be in 1..100")
    baseline_path = ROOT / "outputs/N72R9/replay/full" / str(inputs["event_id"]) / "BASELINE_B0/runtime_frames.jsonl"
    if not baseline_path.is_file():
        raise FileNotFoundError(baseline_path)
    baseline_rows = read_jsonl(baseline_path)
    expected = list(range(event_frame, event_frame + HORIZON + 1))
    if len(baseline_rows) != HORIZON + 1 or [int(row.get("frame", -1)) for row in baseline_rows] != expected:
        raise RuntimeError(f"frozen N72R9 E0 frame axis is incomplete: {inputs['event_id']}")
    if any(row.get("runtime_future_gt_used") is not False for row in baseline_rows):
        raise RuntimeError(f"frozen N72R9 E0 contains a runtime GT flag: {inputs['event_id']}")
    inputs = dict(inputs)
    inputs["baseline_rows"] = baseline_rows[: int(horizon) + 1]
    inputs["horizon"] = int(horizon)
    inputs["rows"] = {
        key: {int(frame): value for frame, value in values.items()}
        for key, values in dict(inputs["rows"]).items()
    }
    return inputs


def _base_for_pool(
    inputs: Mapping[str, Any],
    frame: int,
    pool: Sequence[Mapping[str, Any]],
) -> tuple[list[tuple[int, int]], np.ndarray]:
    pairs, vectors = legacy._base_vectors(inputs, int(frame), pool)
    if not pool:
        raise RuntimeError(f"empty candidate pool at {inputs['event_id']}:{frame}")
    matrix = np.stack([np.asarray(vectors[str(item["candidate_uid"])], dtype=np.float64) for item in pool], axis=0)
    if matrix.ndim != 2 or not np.isfinite(matrix).all():
        raise RuntimeError(f"invalid base matrix at {inputs['event_id']}:{frame}")
    return list(pairs), matrix


def _find_target_uid(solver: Mapping[str, Any], target_public: int) -> str | None:
    matches = [
        row
        for row in solver.get("assignment_rows", [])
        if row.get("public_id") is not None and int(row["public_id"]) == int(target_public)
    ]
    if len(matches) > 1:
        raise RuntimeError(f"duplicate target public assignment at {target_public}")
    return None if not matches else str(matches[0]["candidate_uid"])


def _solver_rows(pool: Sequence[Mapping[str, Any]], solver: Mapping[str, Any]) -> list[dict[str, Any]]:
    decisions = {str(row["candidate_uid"]): row for row in solver.get("assignment_rows", [])}
    if set(decisions) != {str(item["candidate_uid"]) for item in pool}:
        raise RuntimeError("solver candidate axis does not cover the complete pool")
    result: list[dict[str, Any]] = []
    for candidate in pool:
        decision = decisions[str(candidate["candidate_uid"])]
        item = serializable_candidate(candidate, include_feature=False)
        feature = candidate.get("feature")
        norm = None if feature is None else float(np.linalg.norm(np.asarray(feature, dtype=np.float32).reshape(-1)))
        public_id = decision.get("public_id")
        item.update(
            {
                "feature_finite": bool(feature is not None and norm is not None and math.isfinite(norm) and norm > 1.0e-6),
                "feature_norm": norm,
                "solver_public_id": None if public_id is None else int(public_id),
                "solver_association_state_id": decision.get("association_state_id"),
                "solver_status": str(decision["status"]),
                "solver_score": float(decision["score"]),
                "public_id": None if public_id is None else int(public_id),
                "public_id_authority": "exact_global_solver_output" if public_id is not None else None,
                "assigned_public_id": None if public_id is None else int(public_id),
                "assignment_status": "EXPLICIT_NONE" if public_id is None else "ASSIGNED_TO_PUBLIC_ID",
                "runtime_future_gt_used": False,
                "runtime_gt_read": False,
                "posthoc_gt_used": False,
            }
        )
        result.append(item)
    return result


def _model_values(
    inputs: Mapping[str, Any],
    frame: int,
    pool: Sequence[Mapping[str, Any]],
    base_matrix: np.ndarray,
    public_axis: Sequence[int],
    trusted_recent: Sequence[np.ndarray],
    trusted_long: Sequence[np.ndarray],
    distractors: Sequence[np.ndarray],
    predicted_box: Sequence[float],
    previous_raw: int | None,
    previous_scope: str | None,
    previous_score: float,
    previous_uncertainty: float,
    trusted_age: int,
) -> dict[str, Any]:
    width, height = legacy._dimensions(str(inputs["sequence"]), int(frame))
    target_public = int(inputs["target_public_id"])
    if target_public not in [int(value) for value in public_axis]:
        raise RuntimeError(f"target public axis missing at {inputs['event_id']}:{frame}")
    target_col = [int(value) for value in public_axis].index(target_public)
    base_scores = base_matrix[:, target_col].astype(np.float64)
    candidate_values = np.stack(
        [
            candidate_feature_vector(
                candidate,
                anchor_feature=inputs["anchor"],
                anchor_box=inputs["anchor_box"],
                predicted_box=predicted_box,
                previous_raw_sam_id=previous_raw,
                previous_native_scope=previous_scope,
                image_width=width,
                image_height=height,
                candidate_count=len(pool),
                base_target_score=float(base_scores[index]),
            )
            for index, candidate in enumerate(pool)
        ],
        axis=0,
    ).astype(np.float32)
    causal_scores = np.asarray(
        [
            legacy._causal_score(
                candidate,
                float(base_scores[index]),
                trusted_recent[-1:] or trusted_long[-1:],
                predicted_box,
            )
            for index, candidate in enumerate(pool)
        ],
        dtype=np.float64,
    )
    order = sorted(range(len(pool)), key=lambda index: (-float(causal_scores[index]), str(pool[index]["candidate_uid"])))
    top = float(causal_scores[order[0]]) if order else 0.0
    second = float(causal_scores[order[1]]) if len(order) > 1 else 0.0
    margin = float(top - second)
    neighbor = _mean_feature(
        [candidate_values[index, :MEMORY_FEATURE_DIM] for index in order[1:] if np.linalg.norm(candidate_values[index, :MEMORY_FEATURE_DIM]) > 1.0e-6],
        _unit(inputs["anchor"], "anchor"),
    )
    recent_array, recent_mask = _memory_array(trusted_recent, RECENT_TRUSTED_SLOTS)
    long_array, long_mask = _memory_array(trusted_long, LONG_TERM_TRUSTED_SLOTS)
    distractor_array, distractor_mask = _memory_array(distractors, DISTRACTOR_SLOTS)
    temporal = np.asarray(
        [
            float(frame - int(inputs["event_frame"])) / float(HORIZON),
            float(np.tanh(top)),
            float(np.tanh(second)),
            float(np.tanh(margin)),
            float(np.clip(previous_score, -1.0, 1.0)),
            float(np.clip(previous_uncertainty, 0.0, 1.0)),
            float(min(int(trusted_age), HORIZON)) / float(HORIZON),
            float(any(str(candidate["candidate_source"]) == TARGET_SESSION_CURRENT_RAW for candidate in pool)),
        ],
        dtype=np.float32,
    )
    source_values = np.stack([_source_vector(str(candidate["candidate_source"])) for candidate in pool], axis=0).astype(np.float32)
    return {
        "candidate_values": candidate_values,
        "source_values": source_values,
        "human_anchor": _unit(inputs["anchor"], "human anchor"),
        "recent_array": recent_array,
        "recent_mask": recent_mask,
        "long_array": long_array,
        "long_mask": long_mask,
        "distractor_array": distractor_array,
        "distractor_mask": distractor_mask,
        "neighbor": neighbor,
        "temporal": temporal,
        "base_scores": base_scores,
        "causal_scores": causal_scores,
        "causal_order": order,
        "causal_margin": margin,
        "target_col": target_col,
    }


def _heuristic_logits(values: Mapping[str, Any], pool: Sequence[Mapping[str, Any]], anchor: np.ndarray) -> tuple[np.ndarray, float]:
    scores = []
    for index, candidate in enumerate(pool):
        feature = candidate.get("feature")
        cosine = 0.0 if feature is None else float(np.dot(_unit(feature, "heuristic candidate"), anchor))
        score = float(values["base_scores"][index]) + 0.30 * cosine + 0.05 * float(candidate.get("presence_score", 0.0))
        scores.append(score)
    return np.asarray(scores, dtype=np.float64), 0.0


def _model_logits(
    model: torch.nn.Module | None,
    values: Mapping[str, Any],
    pool: Sequence[Mapping[str, Any]],
    device: torch.device,
    anchor: np.ndarray,
) -> tuple[np.ndarray, float]:
    if model is None:
        return _heuristic_logits(values, pool, anchor)
    tensors = (
        torch.as_tensor(values["candidate_values"][None], dtype=torch.float32, device=device),
        torch.ones((1, len(pool)), dtype=torch.bool, device=device),
        torch.as_tensor(values["source_values"][None], dtype=torch.float32, device=device),
        torch.as_tensor(values["human_anchor"][None], dtype=torch.float32, device=device),
        torch.as_tensor(values["recent_array"][None], dtype=torch.float32, device=device),
        torch.as_tensor(values["recent_mask"][None], dtype=torch.bool, device=device),
        torch.as_tensor(values["long_array"][None], dtype=torch.float32, device=device),
        torch.as_tensor(values["long_mask"][None], dtype=torch.bool, device=device),
        torch.as_tensor(values["distractor_array"][None], dtype=torch.float32, device=device),
        torch.as_tensor(values["distractor_mask"][None], dtype=torch.bool, device=device),
        torch.as_tensor(values["neighbor"][None], dtype=torch.float32, device=device),
        torch.as_tensor(values["temporal"][None], dtype=torch.float32, device=device),
    )
    with torch.no_grad():
        logits = model(*tensors)[0].detach().float().cpu().numpy()
    candidate_logits = np.asarray(logits[: len(pool)], dtype=np.float64)
    none_logit = _finite(logits[len(pool)], "none_logit")
    if candidate_logits.shape != (len(pool),) or not np.isfinite(candidate_logits).all():
        raise RuntimeError("N72R11 model returned an invalid candidate logit vector")
    return candidate_logits, none_logit


def _score_pool(
    inputs: Mapping[str, Any],
    frame: int,
    pool: Sequence[Mapping[str, Any]],
    model: torch.nn.Module | None,
    bridge: TargetEdgeBridge | None,
    device: torch.device,
    trusted_recent: Sequence[np.ndarray],
    trusted_long: Sequence[np.ndarray],
    distractors: Sequence[np.ndarray],
    predicted_box: Sequence[float],
    previous_raw: int | None,
    previous_scope: str | None,
    previous_score: float,
    previous_uncertainty: float,
    trusted_age: int,
) -> dict[str, Any]:
    pairs, base_matrix = _base_for_pool(inputs, frame, pool)
    state_axis = [int(pair[0]) for pair in pairs]
    public_axis = [int(pair[1]) for pair in pairs]
    values = _model_values(
        inputs,
        frame,
        pool,
        base_matrix,
        public_axis,
        trusted_recent,
        trusted_long,
        distractors,
        predicted_box,
        previous_raw,
        previous_scope,
        previous_score,
        previous_uncertainty,
        trusted_age,
    )
    logits, none_logit = _model_logits(model, values, pool, device, _unit(inputs["anchor"], "anchor"))
    fused = base_matrix.copy()
    target_col = int(values["target_col"])
    bridge_deltas = np.zeros(len(pool), dtype=np.float64)
    calibrated_target = base_matrix[:, target_col].copy()
    if bridge is not None:
        legacy_scores_by_public = [
            {int(public_axis[column]): float(base_matrix[index, column]) for column in range(len(public_axis))}
            for index in range(len(pool))
        ]
        feature_rows = [
            build_target_edge_feature(
                candidate,
                candidate_logit=float(logits[index]),
                none_logit=float(none_logit),
                legacy_target_score=float(base_matrix[index, target_col]),
                legacy_public_scores=legacy_scores_by_public[index],
                target_public_id=int(inputs["target_public_id"]),
            )
            for index, candidate in enumerate(pool)
        ]
        feature_tensor = torch.as_tensor(feature_rows, dtype=torch.float32, device=device)
        legacy_tensor = torch.as_tensor(base_matrix[:, target_col], dtype=torch.float32, device=device)
        with torch.no_grad():
            delta, calibrated = bridge(feature_tensor, legacy_tensor)
        bridge_deltas = delta.detach().float().cpu().numpy().astype(np.float64)
        calibrated_target = calibrated.detach().float().cpu().numpy().astype(np.float64)
        if not np.isfinite(bridge_deltas).all() or not np.isfinite(calibrated_target).all():
            raise RuntimeError("target-edge bridge returned non-finite values")
        fused[:, target_col] = calibrated_target
    solver = solve_effect_assignment(
        candidate_rows=pool,
        persistent_states=_state_objects(pairs),
        fused_state_candidate_scores=fused.T,
        source_run_id=f"n72r11:{frame}:{inputs['event_id']}",
        session_id=f"n72r11:{inputs['event_id']}",
        none_score=0.0,
    )
    target_uid = _find_target_uid(solver, int(inputs["target_public_id"]))
    scores = (logits - float(none_logit)).astype(np.float64)
    order = sorted(range(len(pool)), key=lambda index: (-float(scores[index]), str(pool[index]["candidate_uid"])))
    best_index = order[0] if order else None
    second_index = order[1] if len(order) > 1 else None
    best_score = None if best_index is None else float(scores[best_index])
    second_score = None if second_index is None else float(scores[second_index])
    model_margin = None if best_score is None else float(best_score - max(0.0, second_score or 0.0))
    selected_uid = (
        None
        if best_index is None or best_score is None or best_score < ADMISSION_SCORE or (model_margin is not None and model_margin < ADMISSION_MARGIN)
        else str(pool[best_index]["candidate_uid"])
    )
    selection = {
        "selected_candidate_uid": selected_uid,
        "selected_score": best_score,
        "second_candidate_uid": None if second_index is None else str(pool[second_index]["candidate_uid"]),
        "second_score": second_score,
        "best_minus_second_margin": model_margin,
        "none_logit": float(none_logit),
        "ranked_candidates": [
            {
                "candidate_uid": str(pool[index]["candidate_uid"]),
                "candidate_source": str(pool[index]["candidate_source"]),
                "model_logit": float(logits[index]),
                "model_score": float(scores[index]),
                "runtime_future_gt_used": False,
            }
            for index in order
        ],
        "candidate_count": len(pool),
        "score_changed": bool(any(abs(float(value)) > 1.0e-9 for value in scores)),
        "runtime_future_gt_used": False,
        "public_id_inference": False,
    }
    return {
        "pairs": pairs,
        "state_axis": state_axis,
        "public_axis": public_axis,
        "base_matrix": base_matrix,
        "fused_matrix": fused,
        "values": values,
        "logits": logits,
        "none_logit": float(none_logit),
        "bridge_deltas": bridge_deltas,
        "calibrated_target": calibrated_target,
        "solver": solver,
        "target_uid": target_uid,
        "selection": selection,
    }


def _make_backend(device: str) -> Sam3Backend:
    return Sam3Backend(
        checkpoint_path=str(CHECKPOINT),
        max_num_objects=16,
        multiplex_count=16,
        use_fa3=False,
        use_rope_real=True,
        compile=False,
        warm_up=False,
        session_expiration_sec=1200,
        output_prob_thresh=0.30,
        async_loading_frames=False,
        device=device,
        official_batched_grounding_batch_size=1,
    )


def _make_live_controller(inputs: Mapping[str, Any], *, end_frame: int, device: str) -> LiveFutureRequeryController:
    sequence = str(inputs["sequence"])
    paths = image_files(DATA_ROOT / "train" / sequence)
    if int(end_frame) >= len(paths):
        raise RuntimeError(f"image coverage is incomplete for live controller: {sequence}:{end_frame}")
    encoder = FrozenMachineOSNetN72R5(device)

    def feature_fn(frame: int, box: Sequence[float]) -> np.ndarray:
        return np.asarray(encoder.encode(paths[int(frame)], [list(box)])[0], dtype=np.float32)

    return LiveFutureRequeryController(
        backend_factory=lambda: _make_backend(device),
        sequence=sequence,
        event_id=str(inputs["event_id"]),
        event_frame=int(inputs["event_frame"]),
        target_public_id=int(inputs["target_public_id"]),
        frame_paths=paths,
        feature_fn=feature_fn,
        end_frame=int(end_frame),
    )


def _event_frame_row(inputs: Mapping[str, Any], variant: str) -> dict[str, Any]:
    event_frame = int(inputs["event_frame"])
    return {
        "schema_version": "N72R11_ON_DEMAND_RUNTIME_FRAME_V1",
        "record_kind": "event_frame_correction",
        "variant": variant,
        "event_id": str(inputs["event_id"]),
        "sequence": str(inputs["sequence"]),
        "event_frame": event_frame,
        "frame": event_frame,
        "frame_horizon": 0,
        "target_public_id": int(inputs["target_public_id"]),
        "candidate_rows": [],
        "candidate_count": 0,
        "candidate_pool": None,
        "assignment": None,
        "score_audit": None,
        "selection_audit": None,
        "memory_read": False,
        "memory_write": True,
        "event_frame_memory_read": False,
        "first_memory_visible_frame": event_frame + 1,
        "requery": {"triggered": False, "applied": False, "runtime_future_gt_used": False},
        "runtime_future_gt_used": False,
        "runtime_gt_read": False,
        "posthoc_gt_used": False,
        "public_id_inference": False,
        "public_id_immutable": True,
    }


def _strip_runtime_rows(rows: Sequence[Mapping[str, Any]], horizon: int) -> list[dict[str, Any]]:
    values = [deepcopy(dict(row)) for row in rows[: int(horizon) + 1]]
    if len(values) != int(horizon) + 1:
        raise RuntimeError("runtime row suffix is incomplete")
    return values


def _run_temporal_variant(
    inputs: Mapping[str, Any],
    *,
    variant: str,
    model: torch.nn.Module | None,
    bridge: TargetEdgeBridge | None,
    device: torch.device,
    enable_live: bool,
    force_trigger: bool,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if variant not in {"E1_V3_BRIDGE", "E2_V3_BRIDGE_LIVE_ON_DEMAND"}:
        raise ValueError(f"unknown N72R11 temporal variant: {variant}")
    event_frame = int(inputs["event_frame"])
    target_public = int(inputs["target_public_id"])
    horizon = int(inputs["horizon"])
    rows: list[dict[str, Any]] = [_event_frame_row(inputs, variant)]
    trusted_recent: list[np.ndarray] = [_unit(inputs["anchor"], "human anchor")]
    trusted_long: list[np.ndarray] = [_unit(inputs["anchor"], "human anchor")]
    distractors: list[np.ndarray] = []
    predicted_box = [float(value) for value in inputs["anchor_box"]]
    event_target = list(inputs["rows"]["target_stream_source"].get(event_frame, {}).get("candidate_rows", []))
    previous_raw = None if not event_target else event_target[0].get("official_raw_sam_id")
    previous_raw = None if previous_raw is None else int(previous_raw)
    previous_scope = None if not event_target else event_target[0].get("native_tid_scope")
    previous_scope = None if previous_scope is None else str(previous_scope)
    previous_score = 0.0
    previous_uncertainty = 1.0
    trusted_age = 0
    live_controller = None
    if variant.startswith("E2") and enable_live:
        live_controller = _make_live_controller(inputs, end_frame=event_frame + horizon, device=str(device))
    stats = {
        "trigger_count": 0,
        "probe_candidate_count": 0,
        "selected_fresh_count": 0,
        "selected_fresh_solver_rejected_count": 0,
        "live_future_candidate_rows": 0,
        "model_score_changed_frame_count": 0,
        "target_assigned_frame_count": 0,
        "assignment_changed_from_pre_rescue_count": 0,
    }
    try:
        for frame in range(event_frame + 1, event_frame + horizon + 1):
            c0_row = inputs["rows"]["c0_source"][frame]
            target_row = inputs["rows"]["target_stream_source"][frame]
            main_candidates = list(c0_row.get("candidate_rows", []))
            current_candidates = [_strip_target_authority(item) for item in target_row.get("candidate_rows", [])]
            if len(current_candidates) > 1:
                raise RuntimeError(f"target stream is not singleton at {inputs['event_id']}:{frame}")
            pre_pool, pre_pool_audit = build_candidate_pool(
                main_candidates,
                current_candidates,
                sequence=str(inputs["sequence"]),
                frame=frame,
                include_target_session=True,
            )
            pre = _score_pool(
                inputs,
                frame,
                pre_pool,
                model,
                bridge,
                device,
                trusted_recent,
                trusted_long,
                distractors,
                predicted_box,
                previous_raw,
                previous_scope,
                previous_score,
                previous_uncertainty,
                trusted_age,
            )
            base_matrix = pre["base_matrix"]
            target_col = [int(value) for value in pre["public_axis"]].index(target_public)
            base_target_scores = base_matrix[:, target_col]
            base_order = sorted(range(len(pre_pool)), key=lambda index: (-float(base_target_scores[index]), str(pre_pool[index]["candidate_uid"])))
            base_top = float(base_target_scores[base_order[0]]) if base_order else 0.0
            base_second = float(base_target_scores[base_order[1]]) if len(base_order) > 1 else 0.0
            base_margin = float(base_top - base_second)
            base_target_uid = _find_target_uid(
                solve_effect_assignment(
                    candidate_rows=pre_pool,
                    persistent_states=_state_objects(pre["pairs"]),
                    fused_state_candidate_scores=base_matrix.T,
                    source_run_id=f"n72r11:pre:{inputs['event_id']}:{frame}",
                    session_id=f"n72r11:pre:{inputs['event_id']}",
                    none_score=0.0,
                ),
                target_public,
            )
            uncertain = bool(base_target_uid is None or base_margin < UNCERTAINTY_MARGIN)
            triggered = False
            applied = False
            probe_audit: dict[str, Any] | None = None
            selection_for_commit: dict[str, Any] | None = None
            if variant.startswith("E2") and live_controller is not None and (uncertain or force_trigger and frame == event_frame + 1):
                triggered = True
                stats["trigger_count"] += 1
                causal_state = {
                    "previous_raw_sam_id": previous_raw,
                    "previous_native_scope": previous_scope,
                    "previous_score": float(previous_score),
                    "previous_uncertainty": float(previous_uncertainty),
                    "trusted_age": int(trusted_age),
                    "source_candidate_uid": str(base_target_uid) if base_target_uid is not None else None,
                    "runtime_future_gt_used": False,
                    "runtime_gt_read": False,
                    "posthoc_gt_used": False,
                }
                session, probe_rows = live_controller.probe(
                    frame=frame,
                    predicted_box=predicted_box,
                    causal_state=causal_state,
                )
                stats["probe_candidate_count"] += len(probe_rows)
                probe_pool, probe_pool_audit = build_candidate_pool_with_future_requery(
                    main_candidates,
                    current_candidates,
                    probe_rows,
                    sequence=str(inputs["sequence"]),
                    frame=frame,
                )
                probe = _score_pool(
                    inputs,
                    frame,
                    probe_pool,
                    model,
                    bridge,
                    device,
                    trusted_recent,
                    trusted_long,
                    distractors,
                    predicted_box,
                    previous_raw,
                    previous_scope,
                    previous_score,
                    previous_uncertainty,
                    trusted_age,
                )
                probe_target_uid = probe["target_uid"]
                fresh = next(
                    (
                        item
                        for item in probe_pool
                        if str(item["candidate_uid"]) == str(probe_target_uid)
                        and str(item["candidate_source"]) == FUTURE_FRAME_REQUERY
                    ),
                    None,
                )
                selection_for_commit = {
                    "selector": "N72R11_V3_BRIDGE_EXACT_SOLVER",
                    "base_assignment_uid": base_target_uid,
                    "base_assignment_margin": base_margin,
                    "probe_solver_target_uid": probe_target_uid,
                    "probe_solver_target_source": None if fresh is None else str(fresh["candidate_source"]),
                    "uncertainty_threshold": UNCERTAINTY_MARGIN,
                    "runtime_future_gt_used": False,
                    "public_id_inference": False,
                }
                if fresh is not None:
                    live_controller.commit(
                        session=session,
                        selected_candidate_uid=str(fresh["candidate_uid"]),
                        selection_audit=selection_for_commit,
                        none_score=0.0,
                        margin=base_margin,
                    )
                    applied = True
                    stats["selected_fresh_count"] += 1
                else:
                    live_controller.commit(
                        session=session,
                        selected_candidate_uid=None,
                        selection_audit=selection_for_commit,
                        none_score=0.0,
                        margin=base_margin,
                    )
                    stats["selected_fresh_solver_rejected_count"] += 1
                probe_audit = live_controller.audit()
            active_rows = [] if live_controller is None else live_controller.active_candidates(frame)
            stats["live_future_candidate_rows"] += len(active_rows)
            if active_rows:
                pool, pool_audit = build_candidate_pool_with_future_requery(
                    main_candidates,
                    current_candidates,
                    active_rows,
                    sequence=str(inputs["sequence"]),
                    frame=frame,
                )
            else:
                pool, pool_audit = pre_pool, pre_pool_audit
            scored = _score_pool(
                inputs,
                frame,
                pool,
                model,
                bridge,
                device,
                trusted_recent,
                trusted_long,
                distractors,
                predicted_box,
                previous_raw,
                previous_scope,
                previous_score,
                previous_uncertainty,
                trusted_age,
            )
            target_uid = scored["target_uid"]
            stats["model_score_changed_frame_count"] += int(scored["selection"].get("score_changed", False))
            stats["target_assigned_frame_count"] += int(target_uid is not None)
            stats["assignment_changed_from_pre_rescue_count"] += int(target_uid != base_target_uid)
            assigned = next((item for item in pool if str(item["candidate_uid"]) == str(target_uid)), None)
            selected_feature = None if assigned is None else assigned.get("feature")
            if selected_feature is not None:
                feature = _unit(selected_feature, "assigned target feature")
                trusted_recent.append(feature)
                trusted_recent = trusted_recent[-RECENT_TRUSTED_SLOTS:]
                if target_uid is not None and assigned.get("candidate_source") != FUTURE_FRAME_REQUERY:
                    trusted_long.append(feature)
                    trusted_long = trusted_long[-LONG_TERM_TRUSTED_SLOTS:]
                trusted_age = 0
                new_box = [float(value) for value in assigned["box_xyxy"]]
                old_center = np.asarray([(predicted_box[0] + predicted_box[2]) / 2.0, (predicted_box[1] + predicted_box[3]) / 2.0])
                new_center = np.asarray([(new_box[0] + new_box[2]) / 2.0, (new_box[1] + new_box[3]) / 2.0])
                predicted_box = new_box
                del old_center, new_center
            else:
                trusted_age += 1
            for candidate in pool:
                if target_uid is not None and str(candidate["candidate_uid"]) == str(target_uid):
                    continue
                if candidate.get("feature") is not None and len(distractors) < DISTRACTOR_SLOTS:
                    distractors.append(_unit(candidate["feature"], "distractor feature"))
                    break
            if assigned is not None:
                previous_raw = assigned.get("official_raw_sam_id")
                previous_raw = None if previous_raw is None else int(previous_raw)
                previous_scope = assigned.get("native_scope")
                previous_scope = None if previous_scope is None else str(previous_scope)
            previous_score = float(np.max(scored["fused_matrix"][:, target_col])) if scored["fused_matrix"].size else 0.0
            previous_uncertainty = float(1.0 / (1.0 + max(base_margin, 0.0)))
            source_rows = [serializable_candidate(candidate, include_feature=False) for candidate in pool]
            for source_row in source_rows:
                source_row["public_id"] = None
                source_row["public_id_authority"] = None
            rows.append(
                {
                    "schema_version": "N72R11_ON_DEMAND_RUNTIME_FRAME_V1",
                    "record_kind": "future_association_frame",
                    "variant": variant,
                    "event_id": str(inputs["event_id"]),
                    "sequence": str(inputs["sequence"]),
                    "event_frame": event_frame,
                    "frame": frame,
                    "frame_horizon": frame - event_frame,
                    "target_public_id": target_public,
                    "candidate_rows": _solver_rows(pool, scored["solver"]),
                    "candidate_count": len(pool),
                    "candidate_pool": {**pool_audit, "candidate_rows": source_rows},
                    "assignment": {
                        "target_public_id": target_public,
                        "target_assigned_candidate_uid": target_uid,
                        "target_base_assigned_candidate_uid": base_target_uid,
                        "target_selected_candidate_uid": scored["selection"].get("selected_candidate_uid"),
                        "target_selector_and_solver_agree": bool(scored["selection"].get("selected_candidate_uid") is not None and str(scored["selection"].get("selected_candidate_uid")) == str(target_uid)),
                        "solver": scored["solver"],
                        "solver_public_id_immutable": True,
                        "runtime_future_gt_used": False,
                    },
                    "score_audit": {
                        "association_state_axis": scored["state_axis"],
                        "public_id_axis": scored["public_axis"],
                        "fused_score_matrix": scored["fused_matrix"].astype(float).tolist(),
                        "base_score_matrix": scored["base_matrix"].astype(float).tolist(),
                        "base_target_scores": scored["base_matrix"][:, target_col].astype(float).tolist(),
                        "fused_target_scores": scored["fused_matrix"][:, target_col].astype(float).tolist(),
                        "model_logit_by_candidate": scored["logits"].astype(float).tolist(),
                        "model_score_by_candidate": (scored["logits"] - float(scored["none_logit"])).astype(float).tolist(),
                        "none_logit": float(scored["none_logit"]),
                        "bridge_target_delta_by_candidate": scored["bridge_deltas"].astype(float).tolist(),
                        "bridge_calibrated_target_by_candidate": scored["calibrated_target"].astype(float).tolist(),
                        "base_top1_score": base_top,
                        "base_top2_score": base_second,
                        "base_assignment_margin": base_margin,
                        "causal_margin": float(scored["values"]["causal_margin"]),
                        "runtime_future_gt_used": False,
                    },
                    "selection_audit": {
                        **scored["selection"],
                        "event_frame_memory_read": False,
                        "frame": frame,
                        "memory_read": True,
                        "runtime_future_gt_used": False,
                    },
                    "memory_read": True,
                    "memory_write": bool(target_uid is not None),
                    "event_frame_memory_read": False,
                    "first_memory_visible_frame": event_frame + 1,
                    "requery": {
                        "triggered": triggered,
                        "applied": applied,
                        "source": FUTURE_FRAME_REQUERY if active_rows else None,
                        "active_candidate_count": len(active_rows),
                        "base_assignment_uid": base_target_uid,
                        "base_assignment_margin": base_margin,
                        "threshold": UNCERTAINTY_MARGIN,
                        "selection_audit": selection_for_commit,
                        "controller_audit": probe_audit,
                        "runtime_future_gt_used": False,
                    },
                    "trusted_memory_update": "CAUSAL_TARGET_ASSIGNMENT" if target_uid is not None else "NO_TRUSTED_UPDATE",
                    "distractor_memory_update_count": len(distractors),
                    "runtime_future_gt_used": False,
                    "runtime_gt_read": False,
                    "posthoc_gt_used": False,
                    "public_id_inference": False,
                    "public_id_immutable": True,
                }
            )
        if live_controller is not None:
            controller_audit = live_controller.audit()
        else:
            controller_audit = {
                "enabled": False,
                "runtime_future_gt_used": False,
                "event_frame_memory_read": False,
                "first_memory_visible_frame": event_frame + 1,
            }
    finally:
        if live_controller is not None:
            controller_audit = live_controller.audit()
            live_controller.close()
        else:
            controller_audit = locals().get("controller_audit", {"enabled": False})
        del live_controller
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    _validate_runtime(rows, inputs, variant)
    stats["runtime_future_gt_used"] = False
    stats["controller_audit"] = controller_audit
    return rows, stats


def _runtime_forbidden_scan(value: Any, location: str, errors: list[str]) -> None:
    forbidden = {"dataset_gt_id", "gt_box", "future_identity_error", "future_iou", "future_gt", "gt_target", "gt_id"}
    flags = {"runtime_future_gt_used", "runtime_gt_read", "posthoc_gt_used", "public_id_inference"}
    if isinstance(value, Mapping):
        for key, nested in value.items():
            key_text = str(key).lower()
            if key_text in forbidden:
                errors.append(f"{location}/{key}")
            if key_text in flags and nested is not False:
                errors.append(f"{location}/{key}=not_false")
            _runtime_forbidden_scan(nested, f"{location}/{key}", errors)
    elif isinstance(value, list):
        for index, nested in enumerate(value):
            _runtime_forbidden_scan(nested, f"{location}/{index}", errors)


def _validate_runtime(rows: Sequence[Mapping[str, Any]], inputs: Mapping[str, Any], variant: str) -> None:
    event_frame = int(inputs["event_frame"])
    expected = list(range(event_frame, event_frame + int(inputs["horizon"]) + 1))
    if len(rows) != len(expected) or [int(row.get("frame", -1)) for row in rows] != expected:
        raise RuntimeError(f"{variant} frame axis is invalid")
    errors: list[str] = []
    for index, row in enumerate(rows):
        _runtime_forbidden_scan(row, f"{variant}/{row.get('frame')}", errors)
        if int(row.get("event_frame", -1)) != event_frame or int(row.get("target_public_id", -1)) != int(inputs["target_public_id"]):
            errors.append(f"{variant}/{row.get('frame')}:authority_axis")
        if row.get("public_id_immutable") is not True:
            errors.append(f"{variant}/{row.get('frame')}:public_id_immutable")
        if index == 0:
            if row.get("candidate_rows") != [] or row.get("candidate_count") != 0 or row.get("memory_read") is not False or row.get("event_frame_memory_read") is not False:
                errors.append(f"{variant}/event_frame:causal_boundary")
            continue
        pool = row.get("candidate_pool")
        score = row.get("score_audit")
        assignment = row.get("assignment")
        if not isinstance(pool, Mapping) or not isinstance(score, Mapping) or not isinstance(assignment, Mapping):
            errors.append(f"{variant}/{row.get('frame')}:missing_audit")
            continue
        pool_rows = list(pool.get("candidate_rows", []))
        output_rows = list(row.get("candidate_rows", []))
        pool_uids = [str(item.get("candidate_uid")) for item in pool_rows]
        output_uids = [str(item.get("candidate_uid")) for item in output_rows]
        if pool_uids != output_uids or len(pool_uids) != len(set(pool_uids)):
            errors.append(f"{variant}/{row.get('frame')}:candidate_axis")
        if int(row.get("candidate_count", -1)) != len(output_rows) or int(pool.get("candidate_count", -1)) != len(pool_rows):
            errors.append(f"{variant}/{row.get('frame')}:candidate_count")
        if pool.get("public_id_inference") is not False or pool.get("runtime_future_gt_used") is not False:
            errors.append(f"{variant}/{row.get('frame')}:pool_boundary")
        if any(item.get("public_id") is not None or item.get("public_id_authority") is not None for item in pool_rows):
            errors.append(f"{variant}/{row.get('frame')}:source_authority")
        matrix = np.asarray(score.get("fused_score_matrix", []), dtype=np.float64)
        state_axis = list(score.get("association_state_axis", []))
        public_axis = list(score.get("public_id_axis", []))
        if matrix.shape != (len(output_rows), len(state_axis)) or len(state_axis) != len(public_axis) or not np.isfinite(matrix).all():
            errors.append(f"{variant}/{row.get('frame')}:score_matrix")
        solver = assignment.get("solver")
        if not isinstance(solver, Mapping) or solver.get("runtime_future_gt_used") is not False:
            errors.append(f"{variant}/{row.get('frame')}:solver_boundary")
        requery = row.get("requery")
        if isinstance(requery, Mapping) and requery.get("applied") and str(requery.get("source")) != FUTURE_FRAME_REQUERY:
            errors.append(f"{variant}/{row.get('frame')}:requery_source")
        if variant.startswith("E1") and isinstance(requery, Mapping) and requery.get("applied"):
            errors.append(f"{variant}/{row.get('frame')}:live_source_in_E1")
    if errors:
        raise RuntimeError("runtime validation failed: " + "; ".join(sorted(set(errors))[:16]))


def _load_v3_checkpoint(path: Path, device: torch.device) -> torch.nn.Module:
    payload = torch.load(path, map_location=device, weights_only=False)
    if not isinstance(payload, Mapping) or not isinstance(payload.get("state_dict"), Mapping):
        raise RuntimeError(f"invalid N72R11 V3 checkpoint: {path}")
    config = dict(payload.get("model_config", {}))
    allowed = {
        "candidate_feature_dim": CANDIDATE_FEATURE_DIM,
        "source_feature_dim": SOURCE_FEATURE_DIM,
        "temporal_feature_dim": TEMPORAL_FEATURE_DIM,
        "recent_trusted_slots": RECENT_TRUSTED_SLOTS,
        "long_term_trusted_slots": LONG_TERM_TRUSTED_SLOTS,
        "distractor_slots": DISTRACTOR_SLOTS,
        "hidden_dim": 256,
        "layers": 3,
        "heads": 8,
        "dropout": 0.0,
    }
    for key, default in allowed.items():
        if key not in config:
            config[key] = default
    model = N72R11TemporalIdentityModel(**{key: config[key] for key in allowed})
    model.load_state_dict(payload["state_dict"], strict=True)
    model.to(device)
    model.eval()
    return model


def _load_bridge_checkpoint(path: Path, device: torch.device) -> TargetEdgeBridge:
    payload = torch.load(path, map_location=device, weights_only=False)
    if not isinstance(payload, Mapping) or not isinstance(payload.get("state_dict"), Mapping):
        raise RuntimeError(f"invalid N72R11 bridge checkpoint: {path}")
    scale = float(payload.get("residual_scale"))
    bridge = TargetEdgeBridge(residual_scale=scale)
    bridge.load_state_dict(payload["state_dict"], strict=True)
    bridge.to(device)
    bridge.eval()
    return bridge


def _posthoc_score(inputs: Mapping[str, Any], runtime_rows: Mapping[str, Sequence[Mapping[str, Any]]]) -> dict[str, Any]:
    """Score sealed runtime outputs using the historical N72R9 metric code."""

    if int(inputs["horizon"]) < 100:
        return {
            "status": "NOT_SCIENTIFIC_SMOKE",
            "reason": "horizon_shorter_than_formal_H100",
            "runtime_future_gt_used": False,
            "posthoc_gt_used": False,
        }
    gt = legacy._load_gt(str(inputs["sequence"]))
    target_gid = None
    protocol = read_json(PROTOCOL_PATH)
    event = next(item for item in protocol["source_event_selection"]["events"] if str(item["event_id"]) == str(inputs["event_id"]))
    target_gid = int(event["dataset_gt_id"])
    protected = legacy._protected_map(inputs["rows"]["c0_source"][int(inputs["event_frame"])], gt, int(inputs["event_frame"]), target_gid)
    event_payload: dict[str, Any] = {
        "event_id": str(inputs["event_id"]),
        "sequence": str(inputs["sequence"]),
        "action_type": str(event["action_type"]),
        "event_frame": int(inputs["event_frame"]),
        "target_public_id": int(inputs["target_public_id"]),
        "target_dataset_gt_id": target_gid,
        "protected_public_by_gt_posthoc": protected,
        "comparisons": {},
        "runtime_future_gt_used": False,
        "posthoc_gt_used": True,
        "interaction_source": "simulated_from_gt",
        "not_real_human_evidence": True,
    }
    names = {"E0_BASELINE_B0": "E0_BASELINE_B0", "E1_V3_BRIDGE": "E1_V3_BRIDGE", "E2_V3_BRIDGE_LIVE_ON_DEMAND": "E2_V3_BRIDGE_LIVE_ON_DEMAND"}
    for comparison, (baseline, treatment) in (
        ("E1_vs_E0", ("E0_BASELINE_B0", "E1_V3_BRIDGE")),
        ("E2_vs_E0", ("E0_BASELINE_B0", "E2_V3_BRIDGE_LIVE_ON_DEMAND")),
        ("E2_vs_E1", ("E1_V3_BRIDGE", "E2_V3_BRIDGE_LIVE_ON_DEMAND")),
    ):
        event_payload["comparisons"][comparison] = {}
        for horizon in HORIZONS:
            event_payload["comparisons"][comparison][str(horizon)] = legacy._score_pair(
                {
                    "event_id": str(inputs["event_id"]),
                    "sequence": str(inputs["sequence"]),
                    "event_frame": int(inputs["event_frame"]),
                    "target_public_id": int(inputs["target_public_id"]),
                    "target_dataset_gt_id": target_gid,
                    "rows": {key: {int(row["frame"]): row for row in values} for key, values in runtime_rows.items()},
                },
                baseline,
                treatment,
                int(horizon),
                gt,
                protected,
            )
    return {
        "schema_version": "N72R11_ON_DEMAND_POSTHOC_EVENT_V1",
        "status": "PASS_N72R11_POSTHOC_EVENT",
        "event": event_payload,
        "runtime_future_gt_used": False,
        "posthoc_gt_used": True,
        "interaction_source": "simulated_from_gt",
        "not_real_human_evidence": True,
    }


def run_event(
    event: Mapping[str, Any],
    *,
    output_root: Path,
    device: str,
    model_checkpoint: Path | None,
    bridge_checkpoint: Path | None,
    horizon: int,
    enable_live: bool,
    force_trigger: bool,
    smoke: bool,
) -> dict[str, Any]:
    inputs = _load_inputs(event, horizon=horizon)
    event_dir = output_root / str(inputs["event_id"])
    done_path = event_dir / "done.json"
    if done_path.exists():
        raise RuntimeError(f"refusing to overwrite existing N72R11 event artifact: {done_path}")
    device_obj = torch.device(device)
    if device_obj.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"requested CUDA device is unavailable: {device}")
    model = None if smoke and model_checkpoint is None else _load_v3_checkpoint(model_checkpoint, device_obj) if model_checkpoint is not None else None
    bridge = None if smoke and bridge_checkpoint is None else _load_bridge_checkpoint(bridge_checkpoint, device_obj) if bridge_checkpoint is not None else None
    baseline_rows = _strip_runtime_rows(inputs["baseline_rows"], horizon)
    if baseline_rows[0].get("candidate_rows") != []:
        raise RuntimeError("frozen E0 event frame is not empty")
    started = now_utc()
    runtime_rows: dict[str, list[dict[str, Any]]] = {
        "E0_BASELINE_B0": baseline_rows,
    }
    stats: dict[str, Any] = {}
    for variant in ("E1_V3_BRIDGE", "E2_V3_BRIDGE_LIVE_ON_DEMAND"):
        rows, variant_stats = _run_temporal_variant(
            inputs,
            variant=variant,
            model=model,
            bridge=bridge,
            device=device_obj,
            enable_live=bool(enable_live and variant.startswith("E2")),
            force_trigger=bool(force_trigger),
        )
        runtime_rows[variant] = rows
        stats[variant] = variant_stats
    manifests: dict[str, Any] = {}
    for variant, rows in runtime_rows.items():
        frames_path = event_dir / variant / "runtime_frames.jsonl"
        atomic_jsonl(frames_path, rows)
        manifests[variant] = {
            "schema_version": "N72R11_ON_DEMAND_RUNTIME_MANIFEST_V1",
            "status": "PASS_N72R11_RUNTIME_ARTIFACT_SEALED",
            "event_id": str(inputs["event_id"]),
            "sequence": str(inputs["sequence"]),
            "variant": variant,
            "event_frame": int(inputs["event_frame"]),
            "target_public_id": int(inputs["target_public_id"]),
            "frame_count": len(rows),
            "frames": str(frames_path),
            "frames_sha256": sha256_file(frames_path),
            "runtime_future_gt_used": False,
            "runtime_gt_read": False,
            "posthoc_gt_used": False,
            "interaction_source": "simulated_from_gt",
            "not_real_human_evidence": True,
        }
        atomic_json(event_dir / variant / "runtime_manifest.json", manifests[variant])
    seal = {
        "schema_version": "N72R11_ON_DEMAND_RUNTIME_EVENT_SEALED_V1",
        "status": "PASS_N72R11_ALL_RUNTIME_SEALED",
        "event_id": str(inputs["event_id"]),
        "variants": list(runtime_rows),
        "runtime_manifests": manifests,
        "runtime_future_gt_used": False,
        "runtime_gt_read": False,
        "posthoc_gt_used": False,
        "gt_loaded": False,
        "created_at_utc": now_utc(),
    }
    seal_path = event_dir / "runtime_event_sealed.json"
    atomic_json(seal_path, seal)
    posthoc = _posthoc_score(inputs, runtime_rows)
    posthoc_path = event_dir / "posthoc.json"
    atomic_json(posthoc_path, posthoc)
    done = {
        "schema_version": "N72R11_ON_DEMAND_EVENT_DONE_V1",
        "status": "PASS_N72R11_RUNTIME_AND_POSTHOC_EVENT",
        "event_id": str(inputs["event_id"]),
        "runtime_event_sealed": str(seal_path),
        "runtime_event_sealed_sha256": sha256_file(seal_path),
        "posthoc": str(posthoc_path),
        "posthoc_sha256": sha256_file(posthoc_path),
        "stats": stats,
        "model_checkpoint": None if model_checkpoint is None else str(model_checkpoint),
        "bridge_checkpoint": None if bridge_checkpoint is None else str(bridge_checkpoint),
        "horizon": int(horizon),
        "runtime_future_gt_used": False,
        "posthoc_gt_used": bool(posthoc.get("posthoc_gt_used") is True),
        "interaction_source": "simulated_from_gt",
        "not_real_human_evidence": True,
        "production_authorized": False,
        "scientific_effect_result": None if smoke else posthoc,
        "started_at_utc": started,
        "finished_at_utc": now_utc(),
    }
    atomic_json(done_path, done)
    return done


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--event-id", required=True)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--model-checkpoint", type=Path, default=None)
    parser.add_argument("--bridge-checkpoint", type=Path, default=None)
    parser.add_argument("--horizon", type=int, default=HORIZON)
    parser.add_argument("--enable-live", action="store_true")
    parser.add_argument("--force-trigger", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    protocol = read_json(PROTOCOL_PATH)
    matches = [item for item in protocol.get("source_event_selection", {}).get("events", []) if str(item["event_id"]) == str(args.event_id)]
    if len(matches) != 1:
        raise SystemExit(f"event is not in frozen N72R9 protocol: {args.event_id}")
    output_root = args.output_root if args.output_root.is_absolute() else ROOT / args.output_root
    try:
        result = run_event(
            matches[0],
            output_root=output_root,
            device=str(args.device),
            model_checkpoint=None if args.model_checkpoint is None else (_path(args.model_checkpoint)),
            bridge_checkpoint=None if args.bridge_checkpoint is None else (_path(args.bridge_checkpoint)),
            horizon=int(args.horizon),
            enable_live=bool(args.enable_live),
            force_trigger=bool(args.force_trigger),
            smoke=bool(args.smoke),
        )
        print(json.dumps({"status": result["status"], "event_id": result["event_id"], "output": str(output_root / str(args.event_id))}, sort_keys=True))
        return 0
    except Exception as exc:
        failure_dir = output_root / "attempts"
        failure_dir.mkdir(parents=True, exist_ok=True)
        failure_path = failure_dir / f"{args.event_id}.failure.json"
        if failure_path.exists():
            index = 2
            while (failure_dir / f"{args.event_id}.failure.attempt{index}.json").exists():
                index += 1
            failure_path = failure_dir / f"{args.event_id}.failure.attempt{index}.json"
        atomic_json(
            failure_path,
            {
                "schema_version": "N72R11_ON_DEMAND_REPLAY_FAILURE_V1",
                "status": "FAIL_N72R11_ON_DEMAND_REPLAY_EVENT",
                "event_id": str(args.event_id),
                "error_type": type(exc).__name__,
                "error": str(exc),
                "traceback": traceback.format_exc(),
                "command": list(sys.argv),
                "device": str(args.device),
                "runtime_future_gt_used": False,
                "historical_outputs_modified": False,
                "finished_at_utc": now_utc(),
            },
        )
        print(json.dumps({"status": "FAIL_N72R11_ON_DEMAND_REPLAY_EVENT", "failure_artifact": str(failure_path), "error": str(exc)}, sort_keys=True), file=sys.stderr)
        return 1
    finally:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    raise SystemExit(main())
