#!/usr/bin/env python3
"""N71 GT-free global-matrix replay followed by isolated posthoc scoring.

The first half of this program only consumes the frozen N70 cache and the
trained N71 scorer.  It writes complete per-frame audit artifacts without
loading DanceTrack annotations.  Only after every runtime artifact has passed
the structural/causal audit does the second half load GT for posthoc scoring.
This is a simulated-from-GT mechanism experiment; it is not a production
tracker and does not claim real-human efficacy.
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
import time
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

from scripts import n70_association_common as n70  # noqa: E402
from scripts import n71_global_matrix_common as global_common  # noqa: E402


HORIZONS = (20, 50, 100)
VARIANTS = ("M0", "M1", "M2", "M3", "M4")
METHODS = ("RAW_BASELINE", "BASE_EXPLICIT_NONE", "GLOBAL_MATRIX", "GLOBAL_MATRIX_TEMPORAL")
BOOTSTRAP_REPS = 2000
BOOTSTRAP_SEEDS = {20: 7020, 50: 7050, 100: 7100}
DATA_ROOT = Path("/path/to/dancetrack")
CHECKPOINT = Path("/path/to/cache/SAM3_InterMOT_N71/training/N71_GLOBAL_MATRIX_SCORER_ATTEMPT2.pt")
DATASET_MANIFEST = ROOT / "outputs/N71/training/global_matrix_dataset_manifest_attempt5.json"
PROTOCOL = ROOT / "outputs/N71/protocol.json"
DEFAULT_RUNTIME_ROOT = Path("/path/to/cache/SAM3_InterMOT_N71/replay_attempt1")
DEFAULT_RUNTIME_MANIFEST = ROOT / "outputs/N71/replay/global_matrix_runtime_manifest_attempt1.json"
DEFAULT_RESULT = ROOT / "outputs/N71/replay/global_matrix_replay_results_attempt1.json"
DEFAULT_STAGE05 = ROOT / "outputs/N71/stage_05_status.json"


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


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def json_hash(value: Any) -> str:
    return sha256_bytes(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8"))


def array_hash(value: Any, dtype: Any | None = None) -> str:
    array = np.ascontiguousarray(np.asarray(value, dtype=dtype))
    return sha256_bytes(array.tobytes())


def atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        directory_fd = os.open(str(path.parent), os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def atomic_jsonl(path: Path, rows: list[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    count = 0
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n")
                count += 1
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        directory_fd = os.open(str(path.parent), os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return count


def load_event_details() -> dict[str, dict[str, Any]]:
    """Load frozen intervention metadata, including GT only for later scoring."""
    events = n70.load_event_map()
    raw = json.loads((ROOT / "outputs/n37/real_event_manifest.json").read_text(encoding="utf-8"))
    for item in raw.get("events", []):
        event = item.get("event", item)
        event_id = str(item.get("protocol_candidate_id") or event.get("event_id"))
        if event_id not in events:
            raise RuntimeError(f"event metadata not in N70 event map: {event_id}")
        events[event_id]["dataset_gt_id"] = int(event["dataset_gt_id"])
        events[event_id]["source_event_manifest_sha256"] = sha256_file(ROOT / "outputs/n37/real_event_manifest.json")
    if len(events) != 24 or any("dataset_gt_id" not in value for value in events.values()):
        raise RuntimeError("frozen N37 event metadata is incomplete")
    return events


def read_event_cache(event_id: str) -> list[dict[str, Any]]:
    path = n70.CACHE_DIR / f"{event_id}.jsonl"
    if not path.is_file():
        raise FileNotFoundError(path)
    rows: list[dict[str, Any]] = []
    seen: set[tuple[str, int]] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"invalid frozen cache JSON {path}:{line_no}: {exc}") from exc
            if not isinstance(row, dict):
                raise RuntimeError(f"cache row is not object: {path}:{line_no}")
            key = (str(row.get("variant")), int(row.get("frame", -1)))
            if key in seen:
                raise RuntimeError(f"duplicate frozen cache key: {event_id}/{key}")
            seen.add(key)
            rows.append(row)
    if len(rows) != 500 or set(key[0] for key in seen) != set(VARIANTS):
        raise RuntimeError(f"frozen event cache must contain 500 rows and five variants: {event_id}")
    frame_sets = {variant: {frame for current_variant, frame in seen if current_variant == variant} for variant in VARIANTS}
    if len({tuple(sorted(value)) for value in frame_sets.values()}) != 1:
        raise RuntimeError(f"variant frame sets differ: {event_id}")
    return rows


def assignment_from_scores(scores: np.ndarray) -> np.ndarray:
    values = np.asarray(scores, dtype=np.float64)
    if values.ndim != 2 or not np.all(np.isfinite(values)):
        raise RuntimeError("base score matrix is not finite 2-D")
    result = np.full(values.shape[0], -1, dtype=np.int64)
    if values.shape[0] and values.shape[1]:
        rows, columns = linear_sum_assignment(-values)
        result[rows] = columns
    return result


def assignment_public(assignment: np.ndarray | list[int], public_ids: list[int]) -> list[int | None]:
    return [None if int(column) < 0 or int(column) >= len(public_ids) else int(public_ids[int(column)]) for column in assignment]


def objective(scores: np.ndarray, assignment: np.ndarray) -> float:
    return float(sum(float(scores[row, int(column)]) for row, column in enumerate(assignment) if int(column) >= 0))


def row_score_audit(scores: np.ndarray, assignment: np.ndarray) -> dict[str, Any]:
    values = np.asarray(scores, dtype=np.float64)
    assigned = np.asarray(assignment, dtype=np.int64)
    top1: list[float | None] = []
    top2: list[float | None] = []
    top_margin: list[float | None] = []
    assigned_scores: list[float | None] = []
    assignment_margin: list[float | None] = []
    for row in range(values.shape[0]):
        current = values[row]
        order = np.argsort(-current, kind="stable")
        first = float(current[order[0]]) if order.size else None
        second = float(current[order[1]]) if order.size > 1 else None
        top1.append(first)
        top2.append(second)
        top_margin.append(None if first is None or second is None else first - second)
        column = int(assigned[row]) if row < assigned.size else -1
        if column < 0 or column >= values.shape[1]:
            assigned_scores.append(None)
            assignment_margin.append(None)
            continue
        assigned_value = float(current[column])
        alternatives = [float(current[index]) for index in range(values.shape[1]) if index != column]
        assigned_scores.append(assigned_value)
        assignment_margin.append(assigned_value - max(alternatives) if alternatives else None)
    return {
        "top1": top1,
        "top2": top2,
        "top1_top2_margin": top_margin,
        "assigned_score": assigned_scores,
        "assigned_vs_best_alternative_margin": assignment_margin,
        "finite": True,
    }


def validate_source_frame(frame: dict[str, Any], event: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[int]]:
    if frame.get("runtime_future_gt_used") is not False or frame.get("runtime_gt_read") is not False:
        raise RuntimeError(f"runtime future GT boundary failed: {frame.get('event_id')}/{frame.get('variant')}/{frame.get('frame')}")
    if frame.get("candidate_set_complete") is not True:
        raise RuntimeError(f"frozen candidate set is incomplete: {frame.get('event_id')}/{frame.get('frame')}")
    candidates = frame.get("candidate_rows")
    rows = frame.get("rows")
    public_ids = [int(value) for value in frame.get("public_id_order", [])]
    if not isinstance(candidates, list) or not isinstance(rows, list) or len(candidates) != len(rows):
        raise RuntimeError("candidate/row axis is not a list with equal length")
    n, p = len(candidates), len(public_ids)
    candidate_features = np.asarray(frame.get("candidate_features_512"), dtype=np.float32)
    memory = np.asarray(frame.get("memory_vectors_512"), dtype=np.float32)
    score = np.asarray(frame.get("score_matrix"), dtype=np.float32)
    scalar = np.asarray(frame.get("scalar_features_8"), dtype=np.float32)
    memory_valid = np.asarray(frame.get("memory_valid"), dtype=bool).reshape(-1)
    if n <= 0 or p <= 0 or candidate_features.shape != (n, 512) or memory.shape != (p, 512) or score.shape != (n, p) or scalar.shape != (n * p, 8) or memory_valid.shape != (p,):
        raise RuntimeError(f"frozen frame shape mismatch n={n} p={p} candidate={candidate_features.shape} memory={memory.shape} score={score.shape} scalar={scalar.shape}")
    for name, value in (("candidate_features", candidate_features), ("memory", memory), ("score", score), ("scalar", scalar)):
        if not np.all(np.isfinite(value)):
            raise RuntimeError(f"frozen frame {name} contains nonfinite values")
    if len(set(public_ids)) != p:
        raise RuntimeError("duplicate public identity columns")
    if [int(row.get("index", -1)) for row in candidates] != list(range(n)):
        raise RuntimeError("candidate order/index is not stable")
    if [int(row.get("candidate_index", -1)) for row in rows] != list(range(n)):
        raise RuntimeError("row order/index is not stable")
    native_ids = [int(row.get("native_tid", -1)) for row in candidates]
    if len(set(native_ids)) != n:
        raise RuntimeError("duplicate runtime native candidate IDs")
    assignment = np.asarray(frame.get("assignment_columns"), dtype=np.int64).reshape(-1)
    if assignment.shape != (n,) or np.any((assignment < -1) | (assignment >= p)):
        raise RuntimeError("frozen assignment axis is malformed")
    if int(frame.get("frame", -1)) <= int(event["event_frame"]):
        raise RuntimeError("replay cache includes event frame or an earlier frame")
    expected_horizon = int(frame["frame"]) - int(event["event_frame"])
    if expected_horizon != int(frame.get("frame_horizon", expected_horizon)) or not 1 <= expected_horizon <= 100:
        raise RuntimeError(f"frozen future horizon is malformed: {expected_horizon}")
    return candidate_features, memory, score, scalar, memory_valid, public_ids


def build_runtime_cells(frame: dict[str, Any], event: dict[str, Any]) -> dict[str, Any]:
    """Build model inputs without touching the offline target native ID."""
    candidates = frame["candidate_rows"]
    candidate_features, memory, score, scalar, memory_valid, public_ids = validate_source_frame(frame, event)
    n, p = len(candidates), len(public_ids)
    target_slot = public_ids.index(int(event["target_public_id"])) if int(event["target_public_id"]) in public_ids else -1
    role = np.zeros(p, dtype=np.float32)
    if target_slot >= 0:
        role[target_slot] = 1.0
    anchor = np.asarray(event["human_embedding"], dtype=np.float32).reshape(-1)
    negative = np.asarray(event.get("negative_embedding", np.zeros(512)), dtype=np.float32).reshape(-1)
    if anchor.shape != (512,) or negative.shape != (512,) or not np.all(np.isfinite(anchor)) or not np.all(np.isfinite(negative)):
        raise RuntimeError("event anchor/negative input is malformed")
    anchors = np.zeros((p, 512), dtype=np.float32)
    negatives = np.zeros((p, 512), dtype=np.float32)
    if target_slot >= 0:
        anchors[target_slot] = anchor
        negatives[target_slot] = negative
    assignment = np.asarray(frame["assignment_columns"], dtype=np.int64).reshape(n)
    assigned_grid = (assignment[:, None] == np.arange(p, dtype=np.int64)[None, :]).astype(np.float32)
    occupancy = np.bincount(assignment[assignment >= 0], minlength=p).astype(np.float32) / max(1, n)
    rank = np.arange(n, dtype=np.float32) / max(1, n - 1)
    count_norm = np.full(n, min(1.0, n / 32.0), dtype=np.float32)
    scalar_grid = scalar.reshape(n, p, 8)
    context = np.concatenate([
        scalar_grid,
        score[:, :, None],
        np.broadcast_to(role[None, :, None], (n, p, 1)),
        np.broadcast_to(memory_valid.astype(np.float32)[None, :, None], (n, p, 1)),
        assigned_grid[:, :, None],
        np.broadcast_to(rank[:, None, None], (n, p, 1)),
        np.broadcast_to(count_norm[:, None, None], (n, p, 1)),
        np.broadcast_to(occupancy[None, :, None], (n, p, 1)),
    ], axis=2).reshape(n * p, global_common.CONTEXT_DIM)
    if not np.all(np.isfinite(context)):
        raise RuntimeError("runtime context is nonfinite")
    return {
        "candidate": np.repeat(candidate_features, p, axis=0),
        "identity_memory": np.tile(memory, (n, 1)),
        "human_anchor": np.tile(anchors, (n, 1)),
        "hard_negative": np.tile(negatives, (n, 1)),
        "context": context,
        "base_score": score,
        "public_ids": public_ids,
        "candidate_rows": candidates,
        "memory_valid": memory_valid,
        "target_slot": target_slot,
    }


def candidate_audit(frame: dict[str, Any], candidate_features: np.ndarray) -> list[dict[str, Any]]:
    result = []
    for index, row in enumerate(frame["candidate_rows"]):
        feature = np.asarray(row.get("machine_embedding"), dtype=np.float32).reshape(-1)
        if feature.shape != (512,) or not np.all(np.isfinite(feature)):
            # The frozen frame-level feature is authoritative for inference;
            # still record a row-level absence explicitly instead of inventing
            # a feature from another candidate.
            feature_finite = False
            feature_norm = None
            feature_digest = None
        else:
            feature_finite = True
            feature_norm = float(np.linalg.norm(feature))
            feature_digest = array_hash(feature, np.float32)
        result.append({
            "candidate_index": int(row.get("index", index)),
            "obs_id": int(row.get("obs_id", index)),
            "native_tid": int(row.get("native_tid", -1)),
            "box": [float(value) for value in row.get("box", [])],
            "confidence": float(row.get("confidence", 0.0)),
            "feature_available": bool(row.get("feature_available", False)),
            "feature_finite": feature_finite,
            "feature_norm": feature_norm,
            "machine_embedding_sha256": feature_digest,
            "candidate_feature_512_sha256": array_hash(candidate_features[index], np.float32),
            "mask_sha256": json_hash(row.get("mask")),
            "mapping": row.get("mapping", {}),
        })
    return result


def load_model(device_name: str) -> tuple[Any, np.ndarray, np.ndarray, dict[str, Any]]:
    import torch

    protocol_sha = sha256_file(PROTOCOL)
    dataset_sha = sha256_file(DATASET_MANIFEST)
    payload = torch.load(CHECKPOINT, map_location=device_name, weights_only=False)
    if not isinstance(payload, dict) or payload.get("schema") != "N71_GLOBAL_MATRIX_SCORER_CHECKPOINT_V1":
        raise RuntimeError("N71 trained checkpoint schema is invalid")
    if payload.get("protocol_sha256") != protocol_sha or payload.get("dataset_manifest_sha256") != dataset_sha:
        raise RuntimeError("N71 trained checkpoint provenance hash mismatch")
    if payload.get("runtime_future_gt_used") is not False or payload.get("production_authorized") is not False:
        raise RuntimeError("N71 checkpoint runtime/production contract is invalid")
    model = global_common.build_model().to(device_name)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    mean = np.asarray(payload.get("context_mean"), dtype=np.float32)
    std = np.asarray(payload.get("context_std"), dtype=np.float32)
    if mean.shape != (global_common.CONTEXT_DIM,) or std.shape != (global_common.CONTEXT_DIM,) or not np.all(np.isfinite(mean)) or not np.all(np.isfinite(std)) or np.any(std <= 0):
        raise RuntimeError("N71 checkpoint context normalizer is invalid")
    metadata = {
        "path": str(CHECKPOINT),
        "sha256": sha256_file(CHECKPOINT),
        "schema": payload.get("schema"),
        "model": payload.get("model_metadata"),
        "protocol_sha256": payload.get("protocol_sha256"),
        "dataset_manifest_sha256": payload.get("dataset_manifest_sha256"),
        "best_epoch": payload.get("config", {}).get("best_epoch"),
        "sequence_disjoint_split": payload.get("sequence_disjoint_split"),
        "context_mean_sha256": array_hash(mean, np.float32),
        "context_std_sha256": array_hash(std, np.float32),
        "runtime_future_gt_used": False,
        "production_authorized": False,
    }
    return model, mean, std, metadata


def normalized_assignment_columns(assignment: dict[str, Any]) -> list[int]:
    """Expose NONE as -1; the solver-internal NONE columns are p+i."""
    identity_count = len(assignment.get("public_id_order", []))
    return [int(column) if 0 <= int(column) < identity_count else -1 for column in assignment["assigned_column"]]


def runtime_method_payload(name: str, assignment: dict[str, Any], score_matrix: np.ndarray, score_audit: dict[str, Any], *, score_cells_changed: int, max_abs_score_delta: float, temporal: dict[str, Any] | None = None) -> dict[str, Any]:
    normalized = normalized_assignment_columns(assignment)
    return {
        "method": name,
        "score_matrix": np.asarray(score_matrix, dtype=np.float32).astype(float).tolist(),
        "assignment_columns": normalized,
        "assignment_public_ids": assignment["assigned_public_ids"],
        "none_assignment_count": int(sum(value is None for value in assignment["assigned_public_ids"])),
        "score_audit": score_audit,
        "score_cells_changed": int(score_cells_changed),
        "max_abs_score_delta": float(max_abs_score_delta),
        "temporal_guard": temporal,
        "runtime_future_gt_used": False,
    }


def run_runtime(events: dict[str, dict[str, Any]], model: Any, mean: np.ndarray, std: np.ndarray, checkpoint_meta: dict[str, Any], device_name: str, runtime_root: Path, manifest_path: Path, event_limit: int | None) -> dict[str, Any]:
    import torch

    if runtime_root.exists() and any(runtime_root.iterdir()):
        raise RuntimeError(f"runtime output root is not empty; use a new attempt root: {runtime_root}")
    artifact_root = runtime_root / "event_artifacts"
    artifact_root.mkdir(parents=True, exist_ok=True)
    selected_ids = sorted(events)[:event_limit] if event_limit is not None else sorted(events)
    if not selected_ids:
        raise RuntimeError("runtime replay selected no events")
    completed: list[dict[str, Any]] = []
    started = time.time()
    for event_number, event_id in enumerate(selected_ids, 1):
        event = events[event_id]
        source_rows = read_event_cache(event_id)
        by_key = {(str(row["variant"]), int(row["frame"])): row for row in source_rows}
        frames = sorted({int(row["frame"]) for row in source_rows})
        if len(frames) != 100:
            raise RuntimeError(f"expected exactly 100 future frames: {event_id}, found {len(frames)}")
        histories: dict[str, dict[int, tuple[int | None, int]]] = {variant: {} for variant in VARIANTS}
        artifact_rows: list[dict[str, Any]] = []
        for frame_number in frames:
            variants: dict[str, Any] = {}
            for variant in VARIANTS:
                frame = by_key[(variant, frame_number)]
                candidate_features, _memory, base_score, _scalar, _memory_valid, public_ids = validate_source_frame(frame, event)
                cells = build_runtime_cells(frame, event)
                n, p = len(cells["candidate_rows"]), len(public_ids)
                context = (cells["context"].astype(np.float32) - mean) / std
                tensors = (
                    torch.as_tensor(cells["candidate"], dtype=torch.float32, device=device_name),
                    torch.as_tensor(cells["identity_memory"], dtype=torch.float32, device=device_name),
                    torch.as_tensor(cells["human_anchor"], dtype=torch.float32, device=device_name),
                    torch.as_tensor(cells["hard_negative"], dtype=torch.float32, device=device_name),
                    torch.as_tensor(context, dtype=torch.float32, device=device_name),
                )
                with torch.inference_mode():
                    pair_logits, none_logits = model(*tensors)
                pair = pair_logits.detach().cpu().numpy().astype(np.float32).reshape(n, p)
                none_grid = none_logits.detach().cpu().numpy().astype(np.float32).reshape(n, p)
                if not np.all(np.isfinite(pair)) or not np.all(np.isfinite(none_grid)):
                    raise FloatingPointError(f"nonfinite global scorer output: {event_id}/{variant}/{frame_number}")
                none_column_spread = float(np.max(np.abs(none_grid - none_grid[:, :1]))) if none_grid.size else 0.0
                if none_column_spread > 1.0e-6:
                    raise RuntimeError(f"candidate-specific NONE contract failed: spread={none_column_spread}")
                none_scores = none_grid[:, 0]
                source_assignment = np.asarray(frame["assignment_columns"], dtype=np.int64).reshape(n)
                recomputed = assignment_from_scores(base_score)
                if abs(objective(base_score, source_assignment) - objective(base_score, recomputed)) > 1.0e-5:
                    raise RuntimeError(f"frozen base assignment is not max-weight: {event_id}/{variant}/{frame_number}")
                base_public = assignment_public(source_assignment, public_ids)
                explicit_base = global_common.explicit_none_hungarian(base_score, np.zeros(n, dtype=np.float64), public_ids, cells["candidate_rows"])
                global_assignment = global_common.explicit_none_hungarian(pair, none_scores, public_ids, cells["candidate_rows"])
                guarded, histories[variant], temporal_audit = global_common.apply_temporal_guard(
                    global_assignment,
                    pair,
                    none_scores,
                    cells["candidate_rows"],
                    public_ids,
                    frame_number,
                    target_native_id=None,
                    history=histories[variant],
                    window_frames=3,
                    hysteresis_margin=global_common.HYSTERESIS_MARGIN,
                )
                delta = pair.astype(np.float64) - base_score.astype(np.float64)
                if not np.all(np.isfinite(delta)):
                    raise FloatingPointError("global/base score delta is nonfinite")
                common_candidate_audit = candidate_audit(frame, candidate_features)
                variants[variant] = {
                    "schema": "N71_GLOBAL_MATRIX_VARIANT_FRAME_V1",
                    "variant": variant,
                    "candidate_count": n,
                    "identity_count": p,
                    "candidate_order": frame.get("candidate_order"),
                    "public_id_order": [int(value) for value in public_ids],
                    "candidate_rows_audit": common_candidate_audit,
                    "memory_valid": [bool(value) for value in cells["memory_valid"]],
                    "candidate_feature_512_sha256": array_hash(candidate_features, np.float32),
                    "memory_vectors_512_sha256": array_hash(frame["memory_vectors_512"], np.float32),
                    "scalar_features_8_sha256": array_hash(frame["scalar_features_8"], np.float32),
                    "runtime_context_15_sha256": array_hash(cells["context"], np.float32),
                    "base_score_matrix": base_score.astype(float).tolist(),
                    "base_rectangular": {
                        "assignment_columns": source_assignment.astype(int).tolist(),
                        "assignment_public_ids": base_public,
                        "score_audit": row_score_audit(base_score, source_assignment),
                        "solver_recomputed_assignment_columns": recomputed.astype(int).tolist(),
                        "source_assignment_max_weight_or_tied": True,
                    },
                    "base_explicit_none": runtime_method_payload(
                        "BASE_EXPLICIT_NONE",
                        explicit_base,
                        base_score,
                        row_score_audit(base_score, np.asarray(normalized_assignment_columns(explicit_base), dtype=np.int64)),
                        score_cells_changed=0,
                        max_abs_score_delta=0.0,
                    ),
                    "global_matrix": runtime_method_payload(
                        "GLOBAL_MATRIX",
                        global_assignment,
                        pair,
                        row_score_audit(pair, np.asarray(normalized_assignment_columns(global_assignment), dtype=np.int64)),
                        score_cells_changed=int(np.sum(np.abs(delta) > 1.0e-12)),
                        max_abs_score_delta=float(np.max(np.abs(delta))) if delta.size else 0.0,
                    ),
                    "global_none_scores": none_scores.astype(float).tolist(),
                    "global_none_column_spread": none_column_spread,
                    "global_matrix_temporal": runtime_method_payload(
                        "GLOBAL_MATRIX_TEMPORAL",
                        guarded,
                        pair,
                        row_score_audit(pair, np.asarray(normalized_assignment_columns(guarded), dtype=np.int64)),
                        score_cells_changed=int(np.sum(np.abs(delta) > 1.0e-12)),
                        max_abs_score_delta=float(np.max(np.abs(delta))) if delta.size else 0.0,
                        temporal=temporal_audit,
                    ),
                    "source_frame_json_sha256": json_hash(frame),
                    "source_provenance": frame.get("source_provenance"),
                    "source_provenance_sha256": json_hash(frame.get("source_provenance")),
                    "candidate_set_complete": True,
                    "runtime_future_gt_used": False,
                    "runtime_gt_read": False,
                    "target_native_id_sent_to_runtime": False,
                    "event_frame_memory_read": False,
                    "memory_write_current_frame_hidden": True,
                    "first_memory_visible_at_event_plus_one": frame_number == int(event["event_frame"]) + 1,
                    "interaction_source": "simulated_from_gt",
                    "real_human_tape": False,
                    "production_authorized": False,
                }
                del tensors, cells, pair_logits, none_logits
                gc.collect()
            artifact_rows.append({
                "schema": "N71_GLOBAL_MATRIX_RUNTIME_FRAME_V1",
                "status": "PASS_RUNTIME_FRAME",
                "event_id": event_id,
                "sequence": event["sequence"],
                "action_type": event["action_type"],
                "event_frame": int(event["event_frame"]),
                "frame": frame_number,
                "frame_horizon": frame_number - int(event["event_frame"]),
                "variants": variants,
                "runtime_future_gt_used": False,
                "runtime_gt_read": False,
                "target_native_id_sent_to_runtime": False,
                "interaction_source": "simulated_from_gt",
                "real_human_tape": False,
                "not_real_human_evidence": True,
                "production_authorized": False,
            })
        artifact_path = artifact_root / f"{event_id}.jsonl"
        count = atomic_jsonl(artifact_path, artifact_rows)
        if count != 100:
            raise RuntimeError(f"atomic event artifact count mismatch: {event_id} {count}")
        completed.append({
            "event_id": event_id,
            "sequence": event["sequence"],
            "artifact": str(artifact_path),
            "artifact_sha256": sha256_file(artifact_path),
            "frame_count": count,
            "variant_frame_count": count * len(VARIANTS),
            "runtime_future_gt_used": False,
        })
        atomic_json(manifest_path, {
            "schema": "N71_GLOBAL_MATRIX_RUNTIME_MANIFEST_V1",
            "status": "IN_PROGRESS" if event_number < len(selected_ids) else "PASS_RUNTIME_ARTIFACTS",
            "created_at_utc": now(),
            "runtime_root": str(runtime_root),
            "artifact_root": str(artifact_root),
            "checkpoint": checkpoint_meta,
            "protocol_sha256": sha256_file(PROTOCOL),
            "dataset_manifest_sha256": sha256_file(DATASET_MANIFEST),
            "selected_event_count": len(selected_ids),
            "completed_event_count": event_number,
            "frame_count": event_number * 100,
            "variant_frame_count": event_number * 100 * len(VARIANTS),
            "completed": completed,
            "runtime_future_gt_used": False,
            "interaction_source": "simulated_from_gt",
            "production_authorized": False,
        })
        print(json.dumps({"events_completed": event_number, "events_total": len(selected_ids), "frames": event_number * 100}, sort_keys=True), flush=True)
        del source_rows, artifact_rows
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return {
        "schema": "N71_GLOBAL_MATRIX_RUNTIME_MANIFEST_V1",
        "status": "PASS_RUNTIME_ARTIFACTS",
        "created_at_utc": now(),
        "runtime_root": str(runtime_root),
        "artifact_root": str(runtime_root / "event_artifacts"),
        "checkpoint": checkpoint_meta,
        "selected_event_count": len(selected_ids),
        "completed_event_count": len(completed),
        "frame_count": len(completed) * 100,
        "variant_frame_count": len(completed) * 100 * len(VARIANTS),
        "duration_seconds": time.time() - started,
        "completed": completed,
        "runtime_future_gt_used": False,
        "interaction_source": "simulated_from_gt",
        "production_authorized": False,
    }


def validate_runtime_artifacts(runtime_root: Path, events: dict[str, dict[str, Any]], event_limit: int | None) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    selected_ids = sorted(events)[:event_limit] if event_limit is not None else sorted(events)
    artifact_root = runtime_root / "event_artifacts"
    if not artifact_root.is_dir():
        raise RuntimeError(f"runtime artifact directory missing: {artifact_root}")
    files = sorted(artifact_root.glob("*.jsonl"))
    expected_names = {f"{event_id}.jsonl" for event_id in selected_ids}
    if {path.name for path in files} != expected_names:
        raise RuntimeError(f"runtime artifact file set mismatch: expected={len(expected_names)} found={len(files)}")
    frame_rows: list[dict[str, Any]] = []
    seen: set[tuple[str, str, int]] = set()
    for path in files:
        event_id = path.stem
        event = events[event_id]
        with path.open("r", encoding="utf-8") as handle:
            rows = [json.loads(line) for line in handle if line.strip()]
        if len(rows) != 100:
            raise RuntimeError(f"runtime artifact frame count mismatch: {event_id} {len(rows)}")
        for row in rows:
            if row.get("schema") != "N71_GLOBAL_MATRIX_RUNTIME_FRAME_V1" or row.get("status") != "PASS_RUNTIME_FRAME":
                raise RuntimeError(f"runtime artifact status/schema failed: {event_id}")
            if row.get("event_id") != event_id or row.get("runtime_future_gt_used") is not False or row.get("runtime_gt_read") is not False or row.get("target_native_id_sent_to_runtime") is not False:
                raise RuntimeError(f"runtime artifact causal/provenance failed: {event_id}/{row.get('frame')}")
            frame = int(row["frame"])
            horizon = int(row["frame_horizon"])
            if horizon != frame - int(event["event_frame"]) or not 1 <= horizon <= 100:
                raise RuntimeError(f"runtime horizon failed: {event_id}/{frame}")
            variants = row.get("variants")
            if not isinstance(variants, dict) or set(variants) != set(VARIANTS):
                raise RuntimeError(f"runtime variant set failed: {event_id}/{frame}")
            for variant in VARIANTS:
                key = (event_id, variant, frame)
                if key in seen:
                    raise RuntimeError(f"duplicate runtime artifact key: {key}")
                seen.add(key)
                value = variants[variant]
                if value.get("runtime_future_gt_used") is not False or value.get("runtime_gt_read") is not False or value.get("target_native_id_sent_to_runtime") is not False:
                    raise RuntimeError(f"variant runtime GT boundary failed: {key}")
                n, p = int(value["candidate_count"]), int(value["identity_count"])
                pids = value.get("public_id_order", [])
                candidates = value.get("candidate_rows_audit", [])
                if n <= 0 or p <= 0 or len(candidates) != n or len(pids) != p or len(set(int(x) for x in pids)) != p:
                    raise RuntimeError(f"runtime matrix axes malformed: {key}")
                for candidate_index, candidate in enumerate(candidates):
                    if int(candidate.get("candidate_index", -1)) != candidate_index:
                        raise RuntimeError(f"runtime candidate order failed: {key}")
                base = np.asarray(value["base_score_matrix"], dtype=np.float64)
                if base.shape != (n, p) or not np.all(np.isfinite(base)):
                    raise RuntimeError(f"runtime base matrix malformed: {key}")
                base_assignment = np.asarray(value["base_rectangular"]["assignment_columns"], dtype=np.int64)
                if base_assignment.shape != (n,) or np.any((base_assignment < -1) | (base_assignment >= p)):
                    raise RuntimeError(f"runtime base assignment malformed: {key}")
                for method_key in ("base_explicit_none", "global_matrix", "global_matrix_temporal"):
                    method = value[method_key]
                    assignment = np.asarray(method["assignment_columns"], dtype=np.int64)
                    matrix = np.asarray(method["score_matrix"], dtype=np.float64)
                    if assignment.shape != (n,) or matrix.shape != (n, p) or not np.all(np.isfinite(matrix)) or np.any((assignment < -1) | (assignment >= p)):
                        raise RuntimeError(f"runtime method axis/matrix malformed: {key}/{method_key}")
                    public = assignment_public(assignment, [int(x) for x in pids])
                    if public != method.get("assignment_public_ids"):
                        raise RuntimeError(f"runtime public assignment mapping failed: {key}/{method_key}")
                    if method.get("runtime_future_gt_used") is not False:
                        raise RuntimeError(f"runtime method GT boundary failed: {key}/{method_key}")
                if float(value.get("global_none_column_spread", 1.0)) > 1.0e-6:
                    raise RuntimeError(f"candidate-specific NONE spread in artifact: {key}")
                frame_rows.append({"frame_row": row, "variant": variant, "variant_data": value, "event": event})
    if len(seen) != len(selected_ids) * 100 * len(VARIANTS):
        raise RuntimeError(f"runtime unique key count mismatch: {len(seen)}")
    audit = {
        "schema": "N71_GLOBAL_MATRIX_RUNTIME_AUDIT_V1",
        "status": "PASS",
        "runtime_root": str(runtime_root),
        "event_count": len(selected_ids),
        "frame_count": len(selected_ids) * 100,
        "variant_frame_count": len(seen),
        "duplicate_keys": 0,
        "missing_keys": 0,
        "candidate_axis_errors": 0,
        "runtime_future_gt_used": False,
        "runtime_gt_read": False,
        "target_native_id_sent_to_runtime": False,
        "event_frame_memory_read": False,
        "first_future_frame_is_event_plus_one": True,
        "candidate_mapping_preserved": True,
        "interaction_source": "simulated_from_gt",
        "production_authorized": False,
    }
    return frame_rows, audit


def box_iou(first: Any, second: Any) -> float:
    a = np.asarray(first, dtype=np.float64).reshape(4)
    b = np.asarray(second, dtype=np.float64).reshape(4)
    if not np.all(np.isfinite(a)) or not np.all(np.isfinite(b)):
        return 0.0
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter
    return float(inter / union) if union > 0 else 0.0


def posthoc_outcome(frame_info: dict[str, Any], method: str, treated_assignment: np.ndarray, *, gt_by_sequence: dict[str, dict[int, dict[int, list[float]]]]) -> dict[str, Any]:
    row = frame_info["frame_row"]
    value = frame_info["variant_data"]
    event = frame_info["event"]
    variant = frame_info["variant"]
    candidates = value["candidate_rows_audit"]
    public_ids = [int(item) for item in value["public_id_order"]]
    base_assignment = np.asarray(value["base_rectangular"]["assignment_columns"], dtype=np.int64)
    n = len(candidates)
    if treated_assignment.shape != (n,):
        raise RuntimeError("posthoc assignment shape mismatch")
    target_native = int(event["target_native_id"])
    target_rows = [index for index, candidate in enumerate(candidates) if int(candidate.get("native_tid", -1)) == target_native]
    target_row = target_rows[0] if len(target_rows) == 1 else None
    target_public = int(event["target_public_id"])
    target_col = public_ids.index(target_public) if target_public in public_ids else None
    target_present = target_row is not None
    target_scope = target_col is not None
    base_public = assignment_public(base_assignment, public_ids)
    treated_public = assignment_public(treated_assignment, public_ids)
    base_correct = bool(target_scope and target_present and int(base_assignment[target_row]) == int(target_col))
    treated_correct = bool(target_scope and target_present and int(treated_assignment[target_row]) == int(target_col))
    utility = int(treated_correct) - int(base_correct) if target_scope and target_present else 0
    target_assigned_base = base_public[target_row] if target_row is not None else None
    target_assigned_treated = treated_public[target_row] if target_row is not None else None
    native_ids = [int(candidate.get("native_tid", -1)) for candidate in candidates]
    untouched_assignment_changed = 0
    untouched_regression_count = 0
    for index, candidate in enumerate(candidates):
        if int(candidate.get("native_tid", -1)) == target_native:
            continue
        if base_public[index] != treated_public[index]:
            untouched_assignment_changed += 1
        mapping = candidate.get("mapping") if isinstance(candidate.get("mapping"), dict) else {}
        mapped_public = mapping.get("public_id")
        if mapped_public is not None and base_public[index] == int(mapped_public) and treated_public[index] != int(mapped_public):
            untouched_regression_count += 1
    gt_frame = gt_by_sequence.get(str(event["sequence"]), {}).get(int(row["frame"]), {})
    gt_id = int(event["dataset_gt_id"])
    gt_box = gt_frame.get(gt_id)
    target_box = candidates[target_row].get("box") if target_row is not None else None
    target_visible = gt_box is not None
    target_iou = box_iou(target_box, gt_box) if target_visible and target_box is not None else None
    mapping_complete = all(
        isinstance(candidate.get("mapping"), dict)
        and candidate["mapping"].get("local_id") is not None
        and candidate["mapping"].get("global_id") is not None
        and (candidate["mapping"].get("public_id") is not None or candidate["mapping"].get("public_id_status") == "EXPLICIT_N54_PUBLIC_ASSIGNMENT_ABSENT")
        for candidate in candidates
    )
    is_global_method = method.startswith("GLOBAL")
    global_method = value["global_matrix"]
    score_audit = global_method.get("score_audit", {})
    base_audit = value["base_rectangular"].get("score_audit", {})
    base_margin = base_audit.get("assigned_vs_best_alternative_margin", [None] * n)
    treated_margin = score_audit.get("assigned_vs_best_alternative_margin", [None] * n)
    target_score_delta = None
    if target_row is not None and target_col is not None:
        global_matrix = np.asarray(global_method["score_matrix"], dtype=np.float64)
        base_matrix = np.asarray(value["base_score_matrix"], dtype=np.float64)
        target_score_delta = float(global_matrix[target_row, target_col] - base_matrix[target_row, target_col])
    baseline_wrong_reassociation = bool(target_present and target_assigned_base is not None and target_assigned_base != target_public)
    treated_wrong_reassociation = bool(target_present and target_assigned_treated is not None and target_assigned_treated != target_public)
    new_wrong_reassociation = bool(treated_wrong_reassociation and not baseline_wrong_reassociation)
    corrected_wrong_reassociation = bool(baseline_wrong_reassociation and treated_correct)
    classification = "N_NO_CHANGE"
    if method == "RAW_BASELINE":
        classification = "N_BASELINE_REFERENCE"
    elif method == "BASE_EXPLICIT_NONE":
        classification = "N_EXPLICIT_NONE_BASELINE"
    elif not target_scope or not mapping_complete:
        classification = "F_MAPPING_UNCERTAIN"
    elif not target_present:
        classification = "E_TARGET_CANDIDATE_ABSENT"
    elif utility > 0:
        classification = "B_CROSSING_TARGET_CORRECT"
    elif utility < 0:
        classification = "C_CROSSING_TARGET_INCORRECT"
    elif treated_correct and untouched_regression_count > 0:
        classification = "D_TARGET_CORRECT_WITH_UNTOUCHED_COLLATERAL"
    elif bool(np.any(np.abs(np.asarray(value["global_matrix"]["score_matrix"], dtype=np.float64) - np.asarray(value["base_score_matrix"], dtype=np.float64)) > 1.0e-12)) and target_assigned_base == target_assigned_treated:
        classification = "A_SCORE_CHANGED_NO_TARGET_CROSSING"
    elif target_assigned_base != target_assigned_treated:
        classification = "N_NEUTRAL_TARGET_ASSIGNMENT_CHANGE"
    return {
        "event_id": str(row["event_id"]),
        "sequence": str(row["sequence"]),
        "split": "UNKNOWN",
        "action_type": str(row["action_type"]),
        "variant": variant,
        "method": method,
        "frame": int(row["frame"]),
        "event_frame": int(row["event_frame"]),
        "horizon": int(row["frame_horizon"]),
        "target_public_id": target_public,
        "target_native_id_posthoc_only": target_native,
        "target_dataset_gt_id_posthoc_only": gt_id,
        "target_row": target_row,
        "target_public_column": target_col,
        "target_candidate_present": target_present,
        "target_scope_resolved": target_scope,
        "target_visible_posthoc": target_visible,
        "target_iou_if_visible": target_iou,
        "baseline_target_assigned_public_id": target_assigned_base,
        "treated_target_assigned_public_id": target_assigned_treated,
        "baseline_target_correct": base_correct,
        "target_correct": treated_correct,
        "utility_delta": utility,
        "target_assignment_changed": target_assigned_base != target_assigned_treated,
        "assignment_changed": not np.array_equal(base_assignment, treated_assignment),
        "correct_assignment_change": utility > 0,
        "incorrect_assignment_change": utility < 0,
        "neutral_assignment_change": utility == 0,
        "baseline_wrong_reassociation": baseline_wrong_reassociation,
        "target_wrong_reassociation": treated_wrong_reassociation,
        "new_wrong_reassociation": new_wrong_reassociation,
        "corrected_wrong_reassociation": corrected_wrong_reassociation,
        "target_missing_after_treatment": not treated_correct,
        "untouched_assignment_changed_count": untouched_assignment_changed,
        "untouched_regression_count": untouched_regression_count,
        "untouched_regression": untouched_regression_count > 0,
        "mapping_complete": mapping_complete,
        "score_changed": bool(global_method.get("score_cells_changed", 0) > 0) if is_global_method else False,
        "score_cells_changed": int(global_method.get("score_cells_changed", 0)) if is_global_method else 0,
        "max_abs_score_delta": float(global_method.get("max_abs_score_delta", 0.0)) if is_global_method else 0.0,
        "target_score_delta": target_score_delta if is_global_method else None,
        "base_assignment_margin": base_margin[target_row] if target_row is not None and target_row < len(base_margin) else None,
        "treated_assignment_margin": treated_margin[target_row] if target_row is not None and target_row < len(treated_margin) else None,
        "recorrection_opportunity_proxy": bool(target_present and target_visible and not treated_correct),
        "classification": classification,
        "runtime_future_gt_used": False,
        "posthoc_gt_used": True,
        "interaction_source": "simulated_from_gt",
        "real_human_tape": False,
        "production_authorized": False,
    }


def bootstrap_sequence(event_values: dict[tuple[str, str], list[float]], seed: int) -> dict[str, Any]:
    by_sequence: defaultdict[str, list[float]] = defaultdict(list)
    for (event_id, _variant), values in event_values.items():
        if values:
            sequence = str(event_id).split("-", 2)[1] if "-" in event_id else event_id
            # The event ID prefix is not used as the sequence label in the
            # artifact; callers replace this key below when constructing the
            # proper sequence-indexed map.
            by_sequence[sequence].append(float(np.mean(values)))
    cluster_means = {key: float(np.mean(value)) for key, value in by_sequence.items() if value}
    return bootstrap_cluster_means(cluster_means, seed)


def bootstrap_cluster_means(cluster_means: dict[str, float], seed: int) -> dict[str, Any]:
    if not cluster_means:
        return {"sequence_count": 0, "mean": None, "ci95": [None, None], "seed": int(seed), "repetitions": BOOTSTRAP_REPS, "cluster_means": {}}
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
        "cluster_means": dict(sorted((str(key), float(value)) for key, value in cluster_means.items())),
    }


def summarize_method(rows: list[dict[str, Any]], method: str, split_by_sequence: dict[str, str], *, include_breakdowns: bool = True) -> dict[str, Any]:
    selected = [row for row in rows if row["method"] == method]
    if not selected:
        return {"method": method, "frame_count": 0}
    for row in selected:
        row["split"] = split_by_sequence.get(str(row["sequence"]), "UNKNOWN")
    horizon_summary: dict[str, Any] = {}
    for horizon in HORIZONS:
        window = [row for row in selected if 1 <= int(row["horizon"]) <= horizon]
        scoped = [row for row in window if row["target_scope_resolved"]]
        candidate_present = [row for row in scoped if row["target_candidate_present"]]
        visible = [row for row in candidate_present if row["target_visible_posthoc"]]
        event_values: dict[tuple[str, str], list[float]] = defaultdict(list)
        seq_values: defaultdict[str, list[float]] = defaultdict(list)
        for row in candidate_present:
            event_values[(str(row["event_id"]), str(row["variant"]))].append(float(row["utility_delta"]))
            seq_values[str(row["sequence"])].append(float(row["utility_delta"]))
        event_means = {key: float(np.mean(value)) for key, value in event_values.items() if value}
        sequence_means = {sequence: float(np.mean(values)) for sequence, values in seq_values.items() if values}
        treated_correct_rate = float(np.mean([int(row["target_correct"]) for row in candidate_present])) if candidate_present else None
        baseline_correct_rate = float(np.mean([int(row["baseline_target_correct"]) for row in candidate_present])) if candidate_present else None
        ious = [float(row["target_iou_if_visible"]) for row in visible if row["target_iou_if_visible"] is not None and math.isfinite(float(row["target_iou_if_visible"])) and row["target_correct"]]
        horizon_summary[str(horizon)] = {
            "horizon": horizon,
            "frame_count": len(window),
            "identity_scope_frames": len(scoped),
            "candidate_present_frames": len(candidate_present),
            "candidate_absent_frames": int(sum(not row["target_candidate_present"] for row in scoped)),
            "target_visible_frames": len(visible),
            "candidate_present_rate": float(np.mean([int(row["target_candidate_present"]) for row in scoped])) if scoped else None,
            "baseline_target_correct_rate": baseline_correct_rate,
            "treated_target_correct_rate": treated_correct_rate,
            "baseline_future_identity_error": None if baseline_correct_rate is None else 1.0 - baseline_correct_rate,
            "treated_future_identity_error": None if treated_correct_rate is None else 1.0 - treated_correct_rate,
            "mean_utility_delta_candidate_present": float(np.mean([float(row["utility_delta"]) for row in candidate_present])) if candidate_present else None,
            "candidate_present_improvement_count": int(sum(row["correct_assignment_change"] for row in candidate_present)),
            "candidate_present_harm_count": int(sum(row["incorrect_assignment_change"] for row in candidate_present)),
            "baseline_wrong_reassociation_count": int(sum(row["baseline_wrong_reassociation"] for row in window)),
            "wrong_reassociation_count": int(sum(row["target_wrong_reassociation"] for row in window)),
            "new_wrong_reassociation_count": int(sum(row["new_wrong_reassociation"] for row in window)),
            "corrected_wrong_reassociation_count": int(sum(row["corrected_wrong_reassociation"] for row in window)),
            "recorrection_opportunity_proxy_rate": float(np.mean([int(row["recorrection_opportunity_proxy"]) for row in candidate_present])) if candidate_present else None,
            "target_iou_if_correct_mean": float(np.mean(ious)) if ious else None,
            "assignment_change_rate": float(np.mean([int(row["assignment_changed"]) for row in window])) if window else None,
            "target_assignment_change_rate": float(np.mean([int(row["target_assignment_changed"]) for row in window])) if window else None,
            "score_change_rate": float(np.mean([int(row["score_changed"]) for row in window])) if window else None,
            "correct_assignment_changes": int(sum(row["correct_assignment_change"] for row in window)),
            "incorrect_assignment_changes": int(sum(row["incorrect_assignment_change"] for row in window)),
            "neutral_assignment_changes": int(sum(row["neutral_assignment_change"] for row in window)),
            "untouched_assignment_changed_total": int(sum(int(row["untouched_assignment_changed_count"]) for row in window)),
            "untouched_regression_total": int(sum(int(row["untouched_regression_count"]) for row in window)),
            "untouched_regression_frame_rate": float(np.mean([int(row["untouched_regression"]) for row in window])) if window else None,
            "sequence_cluster_bootstrap_utility": bootstrap_cluster_means(sequence_means, BOOTSTRAP_SEEDS[horizon]),
            "event_variant_utility_means": {f"{event_id}|{variant}": float(value) for (event_id, variant), value in sorted(event_means.items())},
            "runtime_future_gt_used": False,
        }
    actions = sorted({str(row["action_type"]) for row in selected})
    variants = sorted({str(row["variant"]) for row in selected})
    by_action = {
        action: summarize_method([row for row in selected if row["action_type"] == action], method, split_by_sequence, include_breakdowns=False)["horizons"]
        for action in actions
    } if include_breakdowns else {}
    by_variant = {
        variant: summarize_method([row for row in selected if row["variant"] == variant], method, split_by_sequence, include_breakdowns=False)["horizons"]
        for variant in variants
    } if include_breakdowns else {}
    post_first = [row for row in selected if int(row["horizon"]) > 1]
    return {
        "method": method,
        "frame_count": len(selected),
        "horizons": horizon_summary,
        "by_action": by_action,
        "by_variant": by_variant,
        "all_frame_score_change_rate": float(np.mean([int(row["score_changed"]) for row in selected])) if selected else None,
        "all_frame_assignment_change_rate": float(np.mean([int(row["assignment_changed"]) for row in selected])) if selected else None,
        "assignment_changes_after_event_plus_one": int(sum(row["assignment_changed"] for row in post_first)),
        "score_changes_after_event_plus_one": int(sum(row["score_changed"] for row in post_first)),
        "baseline_wrong_reassociation_total": int(sum(row["baseline_wrong_reassociation"] for row in selected)),
        "wrong_reassociation_total": int(sum(row["target_wrong_reassociation"] for row in selected)),
        "new_wrong_reassociation_total": int(sum(row["new_wrong_reassociation"] for row in selected)),
        "corrected_wrong_reassociation_total": int(sum(row["corrected_wrong_reassociation"] for row in selected)),
        "untouched_regression_total": int(sum(int(row["untouched_regression_count"]) for row in selected)),
        "classification_counts": dict(sorted(Counter(str(row["classification"]) for row in selected).items())),
        "runtime_future_gt_used": False,
    }


def posthoc_score(frame_rows: list[dict[str, Any]], events: dict[str, dict[str, Any]], protocol: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Load GT only after ``validate_runtime_artifacts`` has passed."""
    from sam3_intermot.datasets.dancetrack import DanceTrackDataset

    sequences = sorted({str(event["sequence"]) for event in events.values()})
    dataset = DanceTrackDataset(str(DATA_ROOT), sequences=sequences, split="train")
    raw_gt = {sequence: dataset.load_gt(sequence) for sequence in sequences}
    gt_by_sequence: dict[str, dict[int, dict[int, list[float]]]] = {}
    for sequence, frames in raw_gt.items():
        gt_by_sequence[sequence] = {}
        for frame_number, gt_frame in frames.items():
            gt_by_sequence[sequence][int(frame_number)] = {int(identity): [float(value) for value in box] for identity, box in zip(gt_frame.gt_ids, gt_frame.boxes)}
    split_by_sequence: dict[str, str] = {}
    for split_name, values in protocol.get("sequence_split", {}).items():
        # The frozen protocol also stores boolean metadata under this key;
        # only the three sequence-list fields participate in the split map.
        if not isinstance(values, (list, tuple)):
            continue
        for sequence in values:
            split_by_sequence[str(sequence)] = str(split_name)
    all_outcomes: list[dict[str, Any]] = []
    for frame_info in frame_rows:
        frame_info["event"]["dataset_gt_id"] = int(frame_info["event"]["dataset_gt_id"])
        value = frame_info["variant_data"]
        base = np.asarray(value["base_rectangular"]["assignment_columns"], dtype=np.int64)
        explicit = np.asarray(value["base_explicit_none"]["assignment_columns"], dtype=np.int64)
        global_assignment = np.asarray(value["global_matrix"]["assignment_columns"], dtype=np.int64)
        temporal = np.asarray(value["global_matrix_temporal"]["assignment_columns"], dtype=np.int64)
        for method, assignment in (("RAW_BASELINE", base), ("BASE_EXPLICIT_NONE", explicit), ("GLOBAL_MATRIX", global_assignment), ("GLOBAL_MATRIX_TEMPORAL", temporal)):
            outcome = posthoc_outcome(frame_info, method, assignment, gt_by_sequence=gt_by_sequence)
            outcome["split"] = split_by_sequence.get(str(outcome["sequence"]), "UNKNOWN")
            all_outcomes.append(outcome)
    methods = {method: summarize_method(all_outcomes, method, split_by_sequence) for method in METHODS}
    required = ("GLOBAL_MATRIX", "GLOBAL_MATRIX_TEMPORAL")
    gate_details: dict[str, Any] = {}
    for method in required:
        horizon_checks = {}
        summary = methods[method]
        for horizon in HORIZONS:
            item = summary["horizons"][str(horizon)]
            ci = item["sequence_cluster_bootstrap_utility"]
            lower = ci["ci95"][0]
            horizon_checks[str(horizon)] = {
                "ci_lower_bound": lower,
                "strict_positive": bool(lower is not None and lower > 0.0),
                "candidate_present_improvement_count": item["candidate_present_improvement_count"],
                "baseline_wrong_reassociation_count": item["baseline_wrong_reassociation_count"],
                "wrong_reassociation_count": item["wrong_reassociation_count"],
                "new_wrong_reassociation_count": item["new_wrong_reassociation_count"],
                "untouched_regression_total": item["untouched_regression_total"],
            }
        gate_details[method] = horizon_checks
    strict_ci = all(gate_details[method][str(horizon)]["strict_positive"] for method in required for horizon in HORIZONS)
    candidate_improvement = all(any(gate_details[method][str(horizon)]["candidate_present_improvement_count"] > 0 for horizon in HORIZONS) for method in required)
    # Strict safety concerns newly introduced wrong associations, not the
    # frozen baseline errors that the candidate was meant to repair. Report
    # both quantities above and gate only on the treatment-induced increase.
    no_wrong = all(methods[method]["new_wrong_reassociation_total"] == 0 for method in required)
    untouched_safe = all(methods[method]["untouched_regression_total"] == 0 for method in required)
    assignment_crossing = all(methods[method]["all_frame_assignment_change_rate"] is not None and methods[method]["all_frame_assignment_change_rate"] > 0.0 for method in required)
    effect_beyond_first = all(methods[method]["assignment_changes_after_event_plus_one"] > 0 for method in required)
    integrity = bool(frame_rows) and all(row["runtime_future_gt_used"] is False and row["posthoc_gt_used"] is True for row in all_outcomes)
    research_gate = "PASS_FUTURE_EFFECT" if all((strict_ci, candidate_improvement, no_wrong, untouched_safe, assignment_crossing, effect_beyond_first, integrity)) else "FAIL_FUTURE_EFFECT"
    results = {
        "schema": "N71_GLOBAL_MATRIX_REPLAY_RESULTS_V1",
        "status": "PASS_EXECUTION_FUTURE_EFFECT_PASS" if research_gate == "PASS_FUTURE_EFFECT" else "PASS_EXECUTION_FAIL_FUTURE_EFFECT",
        "created_at_utc": now(),
        "event_count": len(events),
        "independent_sequence_count": len({str(event["sequence"]) for event in events.values()}),
        "runtime_frame_count": len(frame_rows) // len(VARIANTS),
        "variant_frame_count": len(frame_rows),
        "methods": methods,
        "gate": {
            "research_gate": research_gate,
            "candidate_present_improvement": candidate_improvement,
            "strict_ci_lower_bound_positive_all_horizons": strict_ci,
            "no_wrong_reassociation": no_wrong,
            "untouched_regression_safe": untouched_safe,
            "real_assignment_crossing": assignment_crossing,
            "memory_effect_not_first_frame_only": effect_beyond_first,
            "runtime_integrity": integrity,
            "gate_details": gate_details,
            "runtime_future_gt_used": False,
            "posthoc_gt_loaded_after_runtime_audit": True,
            "interaction_source": "simulated_from_gt",
            "real_human_tape": False,
            "production_authorized": False,
        },
        "bootstrap": {"repetitions": BOOTSTRAP_REPS, "seed_by_horizon": {str(key): value for key, value in BOOTSTRAP_SEEDS.items()}, "clusters": "independent sequence"},
        "runtime_future_gt_used": False,
        "posthoc_gt_used": True,
        "interaction_source": "simulated_from_gt",
        "real_human_tape": False,
        "not_real_human_evidence": True,
        "production_authorized": False,
    }
    return all_outcomes, results


def write_failure(exc: BaseException, attempt: str, runtime_root: Path) -> Path:
    attempts = ROOT / "outputs/N71/attempts"
    attempts.mkdir(parents=True, exist_ok=True)
    existing = sorted(attempts.glob("n71_global_matrix_replay_failure_attempt*.json"))
    path = attempts / f"n71_global_matrix_replay_failure_attempt{len(existing) + 1}.json"
    atomic_json(path, {
        "schema": "N71_GLOBAL_MATRIX_REPLAY_FAILURE_V1",
        "status": "FAIL_PRESERVED",
        "attempt": attempt,
        "failure_type": type(exc).__name__,
        "failure_message": str(exc),
        "traceback": traceback.format_exc(),
        "runtime_root": str(runtime_root),
        "checkpoint": str(CHECKPOINT),
        "checkpoint_sha256": sha256_file(CHECKPOINT),
        "protocol_sha256": sha256_file(PROTOCOL),
        "dataset_manifest_sha256": sha256_file(DATASET_MANIFEST),
        "runtime_future_gt_used": False,
        "interaction_source": "simulated_from_gt",
        "production_authorized": False,
    })
    return path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--runtime-root", type=Path, default=DEFAULT_RUNTIME_ROOT)
    parser.add_argument("--runtime-manifest", type=Path, default=DEFAULT_RUNTIME_MANIFEST)
    parser.add_argument("--result", type=Path, default=DEFAULT_RESULT)
    parser.add_argument("--stage-status", type=Path, default=DEFAULT_STAGE05)
    parser.add_argument("--event-limit", type=int, default=None)
    parser.add_argument("--attempt", default="1")
    parser.add_argument("--smoke-only", action="store_true")
    parser.add_argument("--posthoc-only", action="store_true", help="reuse an existing runtime artifact root; do not rerun model inference")
    args = parser.parse_args()
    runtime_root = args.runtime_root.resolve()
    runtime_manifest = args.runtime_manifest.resolve()
    result_path = args.result.resolve()
    stage_path = args.stage_status.resolve()
    try:
        events = load_event_details()
        protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
        if args.posthoc_only:
            frame_rows, runtime_audit = validate_runtime_artifacts(runtime_root, events, args.event_limit)
            audit_path = result_path.parent / (result_path.stem + "_runtime_audit.json")
            atomic_json(audit_path, runtime_audit)
            outcomes, results = posthoc_score(frame_rows, events, protocol)
            results.update({
                "runtime_manifest": str(runtime_manifest),
                "runtime_manifest_sha256": sha256_file(runtime_manifest),
                "runtime_audit": str(audit_path),
                "runtime_audit_sha256": sha256_file(audit_path),
                "checkpoint": {"path": str(CHECKPOINT), "sha256": sha256_file(CHECKPOINT)},
                "protocol": str(PROTOCOL),
                "protocol_sha256": sha256_file(PROTOCOL),
                "dataset_manifest": str(DATASET_MANIFEST),
                "dataset_manifest_sha256": sha256_file(DATASET_MANIFEST),
            })
            atomic_json(result_path, results)
            atomic_json(stage_path, {
                "schema": "N71_STAGE_05_STATUS_V1",
                "status": "PASS_RUNTIME_AUDIT_POSTHOC_REPLAY_COMPLETE",
                "runtime_manifest": str(runtime_manifest),
                "runtime_manifest_sha256": sha256_file(runtime_manifest),
                "runtime_audit": str(audit_path),
                "runtime_audit_sha256": sha256_file(audit_path),
                "replay_results": str(result_path),
                "replay_results_sha256": sha256_file(result_path),
                "event_count": len(events),
                "independent_sequence_count": len({str(event["sequence"]) for event in events.values()}),
                "runtime_frame_count": len(frame_rows) // len(VARIANTS),
                "variant_frame_count": len(frame_rows),
                "research_gate": results["gate"]["research_gate"],
                "runtime_future_gt_used": False,
                "posthoc_gt_loaded_after_runtime_audit": True,
                "interaction_source": "simulated_from_gt",
                "production_authorized": False,
            })
            print(json.dumps({"status": results["status"], "research_gate": results["gate"]["research_gate"], "result": str(result_path), "runtime_manifest": str(runtime_manifest)}, sort_keys=True), flush=True)
            return
        import torch

        if not torch.cuda.is_available() or not str(args.device).startswith("cuda"):
            raise RuntimeError(f"N71 replay requires CUDA, got {args.device}")
        model, mean, std, checkpoint_meta = load_model(args.device)
        runtime_result = run_runtime(events, model, mean, std, checkpoint_meta, args.device, runtime_root, runtime_manifest, args.event_limit)
        frame_rows, runtime_audit = validate_runtime_artifacts(runtime_root, events, args.event_limit)
        audit_path = result_path.parent / (result_path.stem + "_runtime_audit.json")
        atomic_json(audit_path, runtime_audit)
        if args.smoke_only:
            smoke_result = {
                "schema": "N71_GLOBAL_MATRIX_REPLAY_SMOKE_V1",
                "status": "PASS_RUNTIME_AUDIT_SMOKE",
                "runtime_manifest": str(runtime_manifest),
                "runtime_manifest_sha256": sha256_file(runtime_manifest),
                "runtime_audit": str(audit_path),
                "runtime_audit_sha256": sha256_file(audit_path),
                "event_count": args.event_limit,
                "frame_count": len(frame_rows) // len(VARIANTS),
                "variant_frame_count": len(frame_rows),
                "checkpoint": checkpoint_meta,
                "runtime_future_gt_used": False,
                "posthoc_gt_scored": False,
                "interaction_source": "simulated_from_gt",
                "production_authorized": False,
            }
            atomic_json(result_path, smoke_result)
            atomic_json(stage_path, {
                "schema": "N71_STAGE_05_STATUS_V1",
                "status": "PASS_TARGETED_REPLAY_SMOKE_RUNTIME_AUDITED",
                "runtime_manifest": str(runtime_manifest),
                "runtime_manifest_sha256": sha256_file(runtime_manifest),
                "runtime_audit": str(audit_path),
                "runtime_audit_sha256": sha256_file(audit_path),
                "event_count": args.event_limit,
                "frame_count": len(frame_rows) // len(VARIANTS),
                "posthoc_gt_scored": False,
                "runtime_future_gt_used": False,
                "production_authorized": False,
            })
            print(json.dumps(smoke_result, indent=2, sort_keys=True), flush=True)
            return
        outcomes, results = posthoc_score(frame_rows, events, protocol)
        results.update({
            "runtime_manifest": str(runtime_manifest),
            "runtime_manifest_sha256": sha256_file(runtime_manifest),
            "runtime_audit": str(audit_path),
            "runtime_audit_sha256": sha256_file(audit_path),
            "checkpoint": checkpoint_meta,
            "protocol": str(PROTOCOL),
            "protocol_sha256": sha256_file(PROTOCOL),
            "dataset_manifest": str(DATASET_MANIFEST),
            "dataset_manifest_sha256": sha256_file(DATASET_MANIFEST),
        })
        atomic_json(result_path, results)
        atomic_json(stage_path, {
            "schema": "N71_STAGE_05_STATUS_V1",
            "status": "PASS_RUNTIME_AUDIT_POSTHOC_REPLAY_COMPLETE",
            "runtime_manifest": str(runtime_manifest),
            "runtime_manifest_sha256": sha256_file(runtime_manifest),
            "runtime_audit": str(audit_path),
            "runtime_audit_sha256": sha256_file(audit_path),
            "replay_results": str(result_path),
            "replay_results_sha256": sha256_file(result_path),
            "event_count": len(events),
            "independent_sequence_count": len({str(event["sequence"]) for event in events.values()}),
            "runtime_frame_count": len(frame_rows) // len(VARIANTS),
            "variant_frame_count": len(frame_rows),
            "research_gate": results["gate"]["research_gate"],
            "runtime_future_gt_used": False,
            "posthoc_gt_loaded_after_runtime_audit": True,
            "interaction_source": "simulated_from_gt",
            "production_authorized": False,
        })
        print(json.dumps({"status": results["status"], "research_gate": results["gate"]["research_gate"], "result": str(result_path), "runtime_manifest": str(runtime_manifest)}, sort_keys=True), flush=True)
    except Exception as exc:
        failure = write_failure(exc, str(args.attempt), runtime_root)
        print(traceback.format_exc(), file=sys.stderr, flush=True)
        print(json.dumps({"status": "FAIL_PRESERVED", "failure_artifact": str(failure)}, sort_keys=True), file=sys.stderr, flush=True)
        raise


if __name__ == "__main__":
    main()
