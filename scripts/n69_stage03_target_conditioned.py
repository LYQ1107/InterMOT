"""N69 target-conditioned association sidecar.

The module contains four deliberately separated operations:

* ``materialize`` builds an offline, sequence-disjoint dataset from the frozen
  N54 candidate stream.  GT-derived target-native IDs are used only for labels.
* ``smoke`` checks one real frozen event/frame, a CUDA forward/backward pass,
  checkpoint round-trip, and the event-frame causal boundary.
* ``train`` fits a new low-rank target-conditioned scorer in the isolated N69
  output directory.  It never changes SAM3, Hungarian, or candidate generation.
* ``replay`` emits GT-free runtime sidecars; ``score`` joins the sidecars with
  the offline mapping audit and evaluates the paired result posthoc.

The runtime sidecar is target-column scoped because the event already supplies
the authoritative public ID.  It does not receive ``target_native_id`` and it
does not read future GT.  A target-native row is joined only by the posthoc
scorer through the N69 mapping audit.
"""

from __future__ import annotations

import argparse
import copy
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

from scripts.n69_mapping_contract import MAPPING_VERSION  # noqa: E402


OUT = ROOT / "outputs/n69"
TRAIN_DIR = OUT / "training"
REPLAY_DIR = OUT / "replay"
ARTIFACT_DIR = REPLAY_DIR / "event_artifacts"
DIAG_DIR = OUT / "diagnosis"
ATTEMPTS = OUT / "attempts"

N69_PROTOCOL = OUT / "protocol.json"
MODEL_PROTOCOL = OUT / "stage_03_protocol.json"
N37_EVENTS = ROOT / "outputs/n37/real_event_manifest.json"
N54_RUNTIME = ROOT / "outputs/n54/replay/runtime"
N54_STATUS = ROOT / "outputs/n54/replay/runtime_status.json"
N68_RESULTS = ROOT / "outputs/n68/replay/paired_replay_results.json"
N69_CACHE_MANIFEST = OUT / "cache/candidate_cache_manifest.json"
MAPPING_ROWS = DIAG_DIR / "mapping_audit.jsonl"
MAPPING_SUMMARY = DIAG_DIR / "mapping_summary.json"

DATASET = TRAIN_DIR / "n69_target_conditioned_dataset.npz"
DATASET_MANIFEST = TRAIN_DIR / "n69_target_conditioned_dataset_manifest.json"
SMOKE_JSON = TRAIN_DIR / "n69_model_smoke.json"
SMOKE_CKPT = TRAIN_DIR / "n69_model_smoke.pt"
CHECKPOINT = TRAIN_DIR / "n69_target_conditioned_scorer.pt"
TRAIN_MANIFEST = TRAIN_DIR / "n69_target_conditioned_training_manifest.json"

RUNTIME_STATUS = REPLAY_DIR / "runtime_status.json"
RESULTS = REPLAY_DIR / "paired_replay_results.json"
POSTHOC_STATUS = REPLAY_DIR / "posthoc_score_status.json"
ASSIGNMENT_DIAG = REPLAY_DIR / "assignment_diagnostics.jsonl"
STAGE03 = OUT / "stage_03_status.json"
STAGE04 = OUT / "stage_04_status.json"
STAGE05 = OUT / "stage_05_status.json"

VARIANTS = ("M0", "M1", "M2", "M3", "M4")
HORIZONS = (20, 50, 100)
EVENT_COUNT = 24
FRAMES_PER_EVENT = 100
FEAT_DIM = 512
PROJECTION_DIM = 64
SEED = 6901
BOOTSTRAP_SEED = 6908
BOOTSTRAP_REPS = 2000

SPLIT_CODE = {"train": 0, "validation": 1, "holdout": 2}
SPLIT_NAME = {value: key for key, value in SPLIT_CODE.items()}

SCALAR_NAMES = [
    "candidate_human_cosine",
    "candidate_target_memory_cosine",
    "candidate_hard_negative_cosine",
    "human_target_memory_cosine",
    "candidate_hard_negative_margin",
    "target_memory_valid",
    "target_column_present",
    "target_column_max_score_tanh",
    "target_column_margin_tanh",
    "candidate_rank_norm",
    "candidate_target_column_score_tanh",
    "candidate_row_best_score_tanh",
    "candidate_row_margin_tanh",
    "candidate_current_public_is_target",
    "candidate_base_assignment_is_target",
    "candidate_confidence",
    "candidate_native_age_norm",
    "candidate_box_width_norm",
    "candidate_box_height_norm",
    "candidate_center_x_norm",
    "candidate_center_y_norm",
    "frame_offset_norm",
    "candidate_count_norm",
    "public_count_norm",
    "valid_memory_fraction",
    "hard_negative_memory_valid",
] + [f"frozen_scalar_{i}" for i in range(8)]
SCALAR_DIM = len(SCALAR_NAMES)


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str | None:
    if not path.is_file():
        return None
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
        dfd = os.open(str(path.parent), os.O_DIRECTORY)
        try:
            os.fsync(dfd)
        finally:
            os.close(dfd)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def atomic_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
        dfd = os.open(str(path.parent), os.O_DIRECTORY)
        try:
            os.fsync(dfd)
        finally:
            os.close(dfd)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def atomic_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".npz", dir=str(path.parent))
    os.close(fd)
    try:
        np.savez_compressed(tmp, **arrays)
        with open(tmp, "rb") as handle:
            os.fsync(handle.fileno())
        os.replace(tmp, path)
        dfd = os.open(str(path.parent), os.O_DIRECTORY)
        try:
            os.fsync(dfd)
        finally:
            os.close(dfd)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return value


def finite(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return float(default)
    return result if math.isfinite(result) else float(default)


def unit(value: Any, dim: int = FEAT_DIM) -> np.ndarray:
    try:
        arr = np.asarray(value, dtype=np.float32).reshape(-1)
    except (TypeError, ValueError):
        return np.zeros(dim, dtype=np.float32)
    if arr.size != dim or not np.all(np.isfinite(arr)):
        return np.zeros(dim, dtype=np.float32)
    norm = float(np.linalg.norm(arr))
    return arr / norm if norm > 1e-6 else np.zeros(dim, dtype=np.float32)


def feature_digest(value: Any) -> str:
    return hashlib.sha256(np.asarray(value, dtype=np.float32).reshape(-1).tobytes()).hexdigest()


def cosine(left: np.ndarray, right: np.ndarray) -> float:
    a, b = unit(left), unit(right)
    if not np.any(a) or not np.any(b):
        return 0.0
    return float(np.dot(a, b))


def score_tanh(value: float, scale: float = 5.0) -> float:
    return float(np.tanh(float(value) / float(scale)))


def sigmoid_np(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    return (1.0 / (1.0 + np.exp(-np.clip(values, -40.0, 40.0)))).astype(np.float32)


def load_event_map() -> dict[str, dict[str, Any]]:
    payload = load_json(N37_EVENTS)
    events: dict[str, dict[str, Any]] = {}
    for item in payload.get("events", []):
        event = item.get("event", {})
        event_id = str(item.get("protocol_candidate_id") or event.get("event_id"))
        if not isinstance(event, dict) or not event_id or event_id == "None":
            raise RuntimeError("N37 event is not addressable")
        if event.get("interaction_source") != "simulated_from_gt" or item.get("interaction_source") != "simulated_from_gt":
            raise RuntimeError(f"N69 requires simulated event provenance: {event_id}")
        target_public = event.get("public_id", event.get("canonical_public_id"))
        target_native = event.get("target_native_tid")
        if target_public is None or target_native is None:
            raise RuntimeError(f"explicit offline target mapping missing: {event_id}")
        events[event_id] = {
            "event_id": event_id,
            "sequence": str(event["sequence"]),
            "event_frame": int(event["frame"]),
            "future_start": int(item["future_frame_start"]),
            "future_end": int(item["future_frame_end"]),
            "target_public_id": int(target_public),
            "target_native_id": int(target_native),
            "human_embedding": event.get("human_embedding"),
            "action_type": str(item.get("action_type") or event.get("action_type")),
            "interaction_source": "simulated_from_gt",
        }
    if len(events) != EVENT_COUNT:
        raise RuntimeError(f"expected {EVENT_COUNT} events, found {len(events)}")
    return events


def sequence_split() -> dict[str, str]:
    protocol = load_json(N69_PROTOCOL)
    result: dict[str, str] = {}
    for name, values in protocol["data"]["sequence_split"].items():
        for sequence in values:
            sequence = str(sequence)
            if sequence in result:
                raise RuntimeError(f"sequence appears in multiple splits: {sequence}")
            result[sequence] = str(name)
    return result


def ensure_model_protocol() -> dict[str, Any]:
    split = sequence_split()
    grouped: dict[str, list[str]] = {"train": [], "validation": [], "holdout": []}
    for sequence, name in split.items():
        grouped[name].append(sequence)
    for values in grouped.values():
        values.sort()
    payload = {
        "schema": "N69_STAGE_03_TARGET_CONDITIONED_PROTOCOL_V1",
        "status": "FROZEN_BEFORE_DATASET_MATERIALIZATION",
        "created_at_utc": now(),
        "parent_protocol": str(N69_PROTOCOL),
        "parent_protocol_sha256": sha256_file(N69_PROTOCOL),
        "model": {
            "name": "shared-low-rank-target-conditioned-listwise-none-scorer",
            "raw_embedding_dim": FEAT_DIM,
            "projection_dim": PROJECTION_DIM,
            "scalar_dim": SCALAR_DIM,
            "shared_projection": True,
            "projected_terms": ["candidate", "human_anchor", "target_memory", "hard_negative", "candidate*anchor", "abs(candidate-anchor)", "candidate*memory", "abs(candidate-memory)", "candidate*hard_negative", "abs(candidate-hard_negative)"],
            "output_logits": ["target", "none"],
            "hidden": [128, 64],
            "numeric_public_id_feature": False,
            "target_native_id_feature": False,
            "raw_gt_feature": False,
        },
        "loss": {
            "candidate_target_none_cross_entropy": {"coefficient": 1.0, "target_class_weight": 2.0, "none_class_weight": 1.0},
            "within_frame_hard_negative_softplus": {"coefficient": 0.5, "margin": 0.2},
            "explicit_none_frame_bce": {"coefficient": 0.5},
            "temporal_target_logit_consistency": {"coefficient": 0.1, "loss": "smooth_l1", "pair_source": "adjacent frozen frames with target candidate present"},
            "no_op_absent_frame_penalty": {"coefficient": 0.05, "target_max_probability": 0.5},
        },
        "training": {
            "seed": SEED,
            "optimizer": "AdamW",
            "learning_rate": 0.0005,
            "weight_decay": 0.0001,
            "batch_size": 512,
            "max_epochs": 30,
            "early_stopping_patience": 5,
            "selection": "earliest minimum validation composite loss; holdout evaluated once after selection",
            "holdout_used_for_selection": False,
            "normalization": "train-sequence context mean/std only; raw embeddings normalized inside shared projection",
        },
        "application": {
            "residual": "2.0*tanh((target_logit-none_logit)/2.0)",
            "residual_bound": 2.0,
            "none_threshold": 0.5,
            "target_column_only": True,
            "other_public_columns_unchanged": True,
            "hungarian_solver_changed": False,
            "candidate_generation_changed": False,
            "checkpoint_changed": False,
        },
        "sequence_split": grouped,
        "runtime_boundary": {
            "event_frame_memory_read": False,
            "first_memory_visible_frame": "event_frame+1",
            "runtime_future_gt_used": False,
            "target_native_id_sent_to_runtime": False,
            "gt_loaded_only_offline_labels_and_posthoc": True,
        },
        "variants": {"upstream": list(VARIANTS), "methods": ["CURRENT_CCAM_BASELINE", "N69_TARGET_CONDITIONED"]},
        "provenance": {"interaction_source": "simulated_from_gt", "real_human_tape": False, "real_sam3_full_loop": False, "not_real_human_evidence": True, "production_authorized": False},
    }
    if MODEL_PROTOCOL.is_file():
        existing = load_json(MODEL_PROTOCOL)
        for key, value in payload.items():
            if key == "created_at_utc":
                continue
            if existing.get(key) != value:
                raise RuntimeError(f"N69 Stage03 protocol differs at {key}")
        return existing
    atomic_json(MODEL_PROTOCOL, payload)
    return payload


def assignment_from_scores(scores: np.ndarray) -> np.ndarray:
    scores = np.asarray(scores, dtype=np.float64)
    if scores.ndim != 2 or not np.all(np.isfinite(scores)):
        raise RuntimeError("nonfinite assignment score matrix")
    result = np.full(scores.shape[0], -1, dtype=np.int64)
    if scores.shape[0] == 0 or scores.shape[1] == 0:
        return result
    rows, cols = linear_sum_assignment(-scores)
    result[rows] = cols
    return result


def normalize_assignment(value: Any, n: int, p: int) -> np.ndarray:
    result = np.asarray(value, dtype=np.int64).reshape(-1)
    if result.size != n or np.any((result < -1) | (result >= p)):
        raise RuntimeError(f"invalid assignment shape/range ({result.size}, {n}, {p})")
    return result


def finite_rank(values: np.ndarray, index: int) -> int:
    valid = values[np.isfinite(values) & (values > -1.0e8)]
    if index < 0 or index >= values.size or not np.isfinite(values[index]) or values[index] <= -1.0e8:
        return int(values.size + 1)
    return int(1 + np.sum(valid > values[index]))


def top_margin(values: np.ndarray) -> float:
    valid = np.sort(values[np.isfinite(values) & (values > -1.0e8)])[::-1]
    return float(valid[0] - valid[1]) if valid.size >= 2 else 0.0


def validate_runtime_frame(frame: dict[str, Any], variant: str, event: dict[str, Any]) -> dict[str, Any]:
    branch = frame.get("write_baseline")
    if not isinstance(branch, dict):
        raise RuntimeError(f"missing write_baseline {event['event_id']}/{variant}/{frame.get('frame')}")
    candidates = branch.get("candidate_rows", [])
    pids = branch.get("public_id_order", [])
    base = np.asarray(branch.get("score_matrix"), dtype=np.float32)
    features = np.asarray(frame.get("candidate_features_512"), dtype=np.float32)
    memory = np.asarray(frame.get("memory_vectors_512"), dtype=np.float32)
    valid = np.asarray(frame.get("memory_valid"), dtype=bool).reshape(-1)
    if not isinstance(candidates, list) or not isinstance(pids, list):
        raise RuntimeError(f"candidate/public axis is not list {event['event_id']}/{variant}/{frame.get('frame')}")
    n, p = len(candidates), len(pids)
    if base.shape != (n, p) or not np.all(np.isfinite(base)):
        raise RuntimeError(f"invalid base score shape/finite {event['event_id']}/{variant}/{frame.get('frame')}")
    if features.shape != (n, FEAT_DIM) or not np.all(np.isfinite(features)):
        raise RuntimeError(f"invalid candidate embedding {event['event_id']}/{variant}/{frame.get('frame')}")
    if memory.shape != (p, FEAT_DIM) or not np.all(np.isfinite(memory)) or valid.size != p:
        raise RuntimeError(f"invalid target memory axis {event['event_id']}/{variant}/{frame.get('frame')}")
    scalar = np.asarray(frame.get("scalar_features_8"), dtype=np.float32)
    if scalar.shape != (n * p, 8) or not np.all(np.isfinite(scalar)):
        raise RuntimeError(f"invalid scalar contract {event['event_id']}/{variant}/{frame.get('frame')}")
    if frame.get("runtime_future_gt_used") is not False or branch.get("runtime_future_gt_used") is not False:
        raise RuntimeError(f"runtime future GT boundary failed {event['event_id']}/{variant}/{frame.get('frame')}")
    pids = [int(value) for value in pids]
    if len(set(pids)) != len(pids):
        raise RuntimeError(f"duplicate public ID axis {event['event_id']}/{variant}/{frame.get('frame')}")
    assignment = normalize_assignment(branch.get("assignment_columns"), n, p)
    recomputed = assignment_from_scores(base)
    if not np.array_equal(assignment, recomputed):
        source_score = sum(float(base[row, col]) for row, col in enumerate(assignment) if col >= 0)
        recomputed_score = sum(float(base[row, col]) for row, col in enumerate(recomputed) if col >= 0)
        if abs(source_score - recomputed_score) > 1.0e-5:
            raise RuntimeError(f"source assignment is not max-weight {event['event_id']}/{variant}/{frame.get('frame')}")
    return {"branch": branch, "candidates": candidates, "pids": pids, "base": base, "candidate_features": features, "memory_vectors": memory, "memory_valid": valid, "scalar": scalar.reshape(n, p, 8), "assignment": assignment}


def build_feature_arrays(frame: dict[str, Any], event: dict[str, Any], *, include_offline_label: bool) -> dict[str, Any]:
    """Build exactly the runtime feature contract; labels are opt-in offline."""

    variant = str(frame.get("variant", "unknown"))
    validated = validate_runtime_frame(frame, variant, event)
    branch = validated["branch"]
    candidates = validated["candidates"]
    pids = validated["pids"]
    base = validated["base"]
    candidate_raw = np.asarray(validated["candidate_features"], dtype=np.float32)
    memory_raw = np.asarray(validated["memory_vectors"], dtype=np.float32)
    memory_valid = validated["memory_valid"]
    scalar_matrix = validated["scalar"]
    source_assignment = validated["assignment"]
    human_raw = np.asarray(event.get("human_embedding"), dtype=np.float32).reshape(-1)
    if human_raw.size != FEAT_DIM or not np.all(np.isfinite(human_raw)) or not np.any(unit(human_raw)):
        raise RuntimeError(f"invalid human anchor {event['event_id']}")
    human = unit(human_raw)
    target_pid = int(event["target_public_id"])
    target_col = pids.index(target_pid) if target_pid in pids else None
    memory_units = np.asarray([unit(value) for value in memory_raw], dtype=np.float32)
    valid_memory = np.asarray([bool(memory_valid[col]) and bool(np.any(memory_units[col])) for col in range(len(pids))], dtype=bool)
    target_memory = memory_units[target_col].copy() if target_col is not None and valid_memory[target_col] else np.zeros(FEAT_DIM, dtype=np.float32)
    target_memory_valid = bool(target_col is not None and valid_memory[target_col])
    human_memory_cosine = cosine(human, target_memory) if target_memory_valid else 0.0
    target_values = base[:, target_col] if target_col is not None else np.zeros(len(candidates), dtype=np.float32)
    target_max = float(np.max(target_values[target_values > -1.0e8])) if np.any(target_values > -1.0e8) else 0.0
    target_margin = top_margin(target_values)
    valid_memory_fraction = float(np.mean(valid_memory)) if valid_memory.size else 0.0
    raw_candidates: list[np.ndarray] = []
    raw_anchors: list[np.ndarray] = []
    raw_memories: list[np.ndarray] = []
    raw_hard: list[np.ndarray] = []
    context: list[list[float]] = []
    labels: list[int] = []
    target_rows: list[int] = []
    hard_memory_flags: list[bool] = []
    for row, candidate in enumerate(candidates):
        c_raw = np.asarray(candidate_raw[row], dtype=np.float32)
        c = unit(c_raw)
        other_cols = [col for col in range(len(pids)) if col != target_col and valid_memory[col]]
        if other_cols:
            hard_col = max(other_cols, key=lambda col: float(np.dot(c, memory_units[col])))
            hard = memory_units[hard_col].copy()
            hard_valid = True
        else:
            hard = np.zeros(FEAT_DIM, dtype=np.float32)
            hard_valid = False
        candidate_human = float(np.dot(c, human)) if np.any(c) else 0.0
        candidate_memory = float(np.dot(c, target_memory)) if target_memory_valid and np.any(c) else 0.0
        candidate_hard = float(np.dot(c, hard)) if hard_valid and np.any(c) else 0.0
        row_values = base[row]
        finite_row = row_values[np.isfinite(row_values) & (row_values > -1.0e8)]
        row_best = float(np.max(finite_row)) if finite_row.size else 0.0
        row_margin = top_margin(row_values)
        rank = finite_rank(target_values, row) if target_col is not None else len(candidates) + 1
        box = candidate.get("box", [0.0, 0.0, 0.0, 0.0]) if isinstance(candidate, dict) else [0.0] * 4
        x1, y1, x2, y2 = [finite(value) for value in list(box)[:4]]
        current_public = None
        mapped_rows = branch.get("rows", [])
        if row < len(mapped_rows) and isinstance(mapped_rows[row], dict):
            value = mapped_rows[row].get("public_id")
            current_public = int(value) if value is not None else None
        base_assigned_target = bool(source_assignment[row] >= 0 and pids[int(source_assignment[row])] == target_pid)
        scalar_values = scalar_matrix[row, target_col].tolist() if target_col is not None else [0.0] * 8
        values = [
            candidate_human,
            candidate_memory,
            candidate_hard,
            human_memory_cosine,
            candidate_memory - candidate_hard,
            1.0 if target_memory_valid else 0.0,
            1.0 if target_col is not None else 0.0,
            score_tanh(target_max),
            score_tanh(target_margin),
            float((rank - 1) / max(len(candidates) - 1, 1)) if target_col is not None else 1.0,
            score_tanh(float(target_values[row])),
            score_tanh(row_best),
            score_tanh(row_margin),
            1.0 if current_public == target_pid else 0.0,
            1.0 if base_assigned_target else 0.0,
            float(np.clip(finite(candidate.get("confidence")), 0.0, 1.0)),
            float(np.clip(finite(candidate.get("native_age")) / 2000.0, 0.0, 1.0)),
            float(np.clip((x2 - x1) / 1920.0, 0.0, 1.0)),
            float(np.clip((y2 - y1) / 1080.0, 0.0, 1.0)),
            float(np.clip(((x1 + x2) * 0.5) / 1920.0, 0.0, 1.0)),
            float(np.clip(((y1 + y2) * 0.5) / 1080.0, 0.0, 1.0)),
            float(np.clip((int(frame["frame"]) - int(event["event_frame"])) / 100.0, 0.0, 1.0)),
            float(np.clip(len(candidates) / 20.0, 0.0, 1.0)),
            float(np.clip(len(pids) / 32.0, 0.0, 1.0)),
            valid_memory_fraction,
            1.0 if hard_valid else 0.0,
        ] + [finite(value) for value in scalar_values]
        if len(values) != SCALAR_DIM or not np.all(np.isfinite(values)):
            raise RuntimeError(f"N69 scalar feature contract violation {event['event_id']}/{frame.get('frame')}/{row}")
        raw_candidates.append(c_raw)
        raw_anchors.append(human_raw.copy())
        raw_memories.append(target_memory.copy())
        raw_hard.append(hard)
        context.append(values)
        hard_memory_flags.append(hard_valid)
        if include_offline_label:
            labels.append(int(candidate.get("native_tid") == int(event["target_native_id"])))
            if labels[-1]:
                target_rows.append(row)
    target_row = target_rows[0] if target_rows else None
    result = {
        "candidate": np.asarray(raw_candidates, dtype=np.float32),
        "anchor": np.asarray(raw_anchors, dtype=np.float32),
        "memory": np.asarray(raw_memories, dtype=np.float32),
        "hard_negative": np.asarray(raw_hard, dtype=np.float32),
        "context": np.asarray(context, dtype=np.float32),
        "base": base.astype(np.float32),
        "pids": pids,
        "source_assignment": source_assignment.astype(np.int64),
        "target_column": target_col,
        "target_memory_valid": target_memory_valid,
        "hard_negative_memory_valid": hard_memory_flags,
        "candidate_feature_digests": [feature_digest(value) for value in candidate_raw],
        "memory_feature_digests": [feature_digest(value) for value in memory_raw],
        "human_feature_digest": feature_digest(human_raw),
        "target_row_offline": target_row,
        "target_present_offline": target_row is not None,
        "labels": np.asarray(labels, dtype=np.int64) if include_offline_label else None,
        "runtime_future_gt_used": False,
    }
    if result["candidate"].shape[0] != len(candidates):
        raise RuntimeError("N69 feature/candidate count mismatch")
    return result


def iter_source_frames(events: dict[str, dict[str, Any]]) -> Iterable[tuple[str, dict[str, Any], str, dict[str, Any]]]:
    for event_id in sorted(events):
        event = events[event_id]
        path = N54_RUNTIME / f"{event_id}.json"
        if not path.is_file():
            raise RuntimeError(f"missing frozen N54 source {path}")
        with path.open("r", encoding="utf-8") as handle:
            source = json.load(handle)
        if source.get("event_id") != event_id or source.get("runtime_boundary", {}).get("runtime_future_gt_used") is not False:
            raise RuntimeError(f"N54 event/provenance mismatch {event_id}")
        for variant in VARIANTS:
            frames = source.get("variants", {}).get(variant, {}).get("frames", [])
            if len(frames) != FRAMES_PER_EVENT:
                raise RuntimeError(f"N54 frame denominator mismatch {event_id}/{variant}")
            expected = list(range(event["event_frame"] + 1, event["event_frame"] + 1 + FRAMES_PER_EVENT))
            actual = [int(item.get("frame", -1)) for item in frames]
            if actual != expected:
                raise RuntimeError(f"N54 future range mismatch {event_id}/{variant}")
            for raw in frames:
                frame = dict(raw)
                frame["variant"] = variant
                yield event_id, event, variant, frame
        del source
        gc.collect()


def materialize() -> dict[str, Any]:
    ensure_model_protocol()
    events = load_event_map()
    split = sequence_split()
    candidates: list[np.ndarray] = []
    anchors: list[np.ndarray] = []
    memories: list[np.ndarray] = []
    hard_negatives: list[np.ndarray] = []
    contexts: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    groups: list[np.ndarray] = []
    group_split: list[int] = []
    group_has_positive: list[int] = []
    event_indices: list[np.ndarray] = []
    variant_indices: list[np.ndarray] = []
    frame_numbers: list[np.ndarray] = []
    native_ids: list[np.ndarray] = []
    sequence_values: list[np.ndarray] = []
    event_id_values: list[np.ndarray] = []
    target_columns: list[np.ndarray] = []
    target_rows: list[np.ndarray] = []
    temporal_left: list[int] = []
    temporal_right: list[int] = []
    target_present_frames = 0
    target_absent_frames = 0
    group_id = 0
    example_offset = 0
    previous_positive: dict[tuple[str, str], int | None] = {}
    per_split_examples: Counter[str] = Counter()
    per_split_positive: Counter[str] = Counter()
    action_counts: Counter[str] = Counter()
    for event_id, event, variant, frame in iter_source_frames(events):
        features = build_feature_arrays(frame, event, include_offline_label=True)
        n = features["candidate"].shape[0]
        y = features["labels"]
        if n == 0:
            raise RuntimeError(f"empty candidate frame {event_id}/{variant}/{frame['frame']}")
        split_name = split[event["sequence"]]
        group_id_array = np.full(n, group_id, dtype=np.int64)
        split_code = SPLIT_CODE[split_name]
        has_positive = int(np.any(y == 1))
        candidate_native = np.asarray([int(row.get("native_tid")) for row in frame["write_baseline"]["candidate_rows"]], dtype=np.int64)
        candidates.append(features["candidate"])
        anchors.append(features["anchor"])
        memories.append(features["memory"])
        hard_negatives.append(features["hard_negative"])
        contexts.append(features["context"])
        labels.append(y)
        groups.append(group_id_array)
        group_split.append(split_code)
        group_has_positive.append(has_positive)
        event_indices.append(np.full(n, len(event_indices), dtype=np.int64))
        variant_indices.append(np.full(n, VARIANTS.index(variant), dtype=np.int8))
        frame_numbers.append(np.full(n, int(frame["frame"]), dtype=np.int32))
        native_ids.append(candidate_native)
        sequence_values.append(np.full(n, event["sequence"], dtype="U64"))
        event_id_values.append(np.full(n, event_id, dtype="U128"))
        target_columns.append(np.full(n, -1 if features["target_column"] is None else int(features["target_column"]), dtype=np.int32))
        target_rows.append(np.full(n, -1 if features["target_row_offline"] is None else int(features["target_row_offline"]), dtype=np.int32))
        if has_positive:
            target_present_frames += 1
            positive_row = int(np.where(y == 1)[0][0])
            previous_key = (event_id, variant)
            if previous_positive.get(previous_key) is not None:
                temporal_left.append(int(previous_positive[previous_key]))
                temporal_right.append(example_offset + positive_row)
            previous_positive[previous_key] = example_offset + positive_row
        else:
            target_absent_frames += 1
            previous_positive[(event_id, variant)] = None
        per_split_examples[split_name] += n
        per_split_positive[split_name] += int(np.sum(y == 1))
        action_counts[event["action_type"]] += 1
        example_offset += n
        group_id += 1
        if group_id % 250 == 0:
            print(json.dumps({"materialized_groups": group_id, "examples": example_offset}, sort_keys=True), flush=True)
    if group_id != EVENT_COUNT * len(VARIANTS) * FRAMES_PER_EVENT:
        raise RuntimeError(f"expected 12000 frame groups, found {group_id}")
    arrays = {
        "candidate": np.concatenate(candidates, axis=0),
        "anchor": np.concatenate(anchors, axis=0),
        "memory": np.concatenate(memories, axis=0),
        "hard_negative": np.concatenate(hard_negatives, axis=0),
        "context": np.concatenate(contexts, axis=0),
        "label": np.concatenate(labels, axis=0),
        "group": np.concatenate(groups, axis=0),
        "group_split": np.asarray(group_split, dtype=np.int8),
        "group_has_positive": np.asarray(group_has_positive, dtype=np.int8),
        "event_index": np.concatenate(event_indices, axis=0),
        "variant_index": np.concatenate(variant_indices, axis=0),
        "frame": np.concatenate(frame_numbers, axis=0),
        "native_id": np.concatenate(native_ids, axis=0),
        "sequence": np.concatenate(sequence_values, axis=0),
        "event_id": np.concatenate(event_id_values, axis=0),
        "target_column": np.concatenate(target_columns, axis=0),
        "target_row_offline": np.concatenate(target_rows, axis=0),
        "temporal_left": np.asarray(temporal_left, dtype=np.int64),
        "temporal_right": np.asarray(temporal_right, dtype=np.int64),
    }
    if not all(np.all(np.isfinite(value)) for key, value in arrays.items() if np.issubdtype(value.dtype, np.number)):
        raise RuntimeError("N69 dataset contains nonfinite numeric values")
    atomic_npz(DATASET, arrays)
    manifest = {
        "schema": "N69_TARGET_CONDITIONED_DATASET_MANIFEST_V1",
        "status": "PASS_DATASET_MATERIALIZED",
        "created_at_utc": now(),
        "protocol": str(MODEL_PROTOCOL),
        "protocol_sha256": sha256_file(MODEL_PROTOCOL),
        "dataset": str(DATASET),
        "dataset_sha256": sha256_file(DATASET),
        "shape": {key: list(value.shape) for key, value in arrays.items()},
        "examples": int(arrays["label"].shape[0]),
        "groups": int(group_id),
        "frames": int(group_id),
        "positive_examples": int(np.sum(arrays["label"] == 1)),
        "negative_examples": int(np.sum(arrays["label"] == 0)),
        "target_present_frames": target_present_frames,
        "target_absent_frames": target_absent_frames,
        "temporal_pairs": int(len(temporal_left)),
        "examples_by_sequence_split": dict(sorted(per_split_examples.items())),
        "positive_by_sequence_split": dict(sorted(per_split_positive.items())),
        "action_frame_counts": dict(sorted(action_counts.items())),
        "feature_contract": {
            "raw_candidate_dim": FEAT_DIM,
            "raw_human_anchor_dim": FEAT_DIM,
            "raw_target_memory_dim": FEAT_DIM,
            "raw_hard_negative_dim": FEAT_DIM,
            "scalar_names": SCALAR_NAMES,
            "scalar_dim": SCALAR_DIM,
            "target_native_id_used_only_for_offline_label": True,
            "numeric_public_id_feature": False,
            "runtime_future_gt_used": False,
        },
        "input_hashes": {
            "n37_event_manifest": sha256_file(N37_EVENTS),
            "n54_runtime_status": sha256_file(N54_STATUS),
            "n69_mapping_summary": sha256_file(MAPPING_SUMMARY),
            "n69_cache_manifest": sha256_file(N69_CACHE_MANIFEST),
        },
        "interaction_source": "simulated_from_gt",
        "real_human_tape": False,
        "real_sam3_full_loop": False,
        "not_real_human_evidence": True,
        "runtime_future_gt_used": False,
        "production_authorized": False,
    }
    atomic_json(DATASET_MANIFEST, manifest)
    return manifest


def set_all_seeds(seed: int) -> None:
    import random

    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def torch_device(name: str):
    import torch

    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    return torch.device(name)


def build_model():
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    class TargetConditionedScorer(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.shared_projection = nn.Linear(FEAT_DIM, PROJECTION_DIM, bias=False)
            self.norm = nn.LayerNorm(PROJECTION_DIM * 10 + SCALAR_DIM)
            self.scorer = nn.Sequential(
                nn.Linear(PROJECTION_DIM * 10 + SCALAR_DIM, 128),
                nn.ReLU(),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Linear(64, 2),
            )

        def forward(self, candidate, anchor, memory, hard_negative, context):
            # Normalization is part of the frozen model contract, not a
            # dataset-specific feature rewrite.
            candidate = F.normalize(candidate, dim=-1, eps=1e-6)
            anchor = F.normalize(anchor, dim=-1, eps=1e-6)
            memory = F.normalize(memory, dim=-1, eps=1e-6)
            hard_negative = F.normalize(hard_negative, dim=-1, eps=1e-6)
            c = self.shared_projection(candidate)
            a = self.shared_projection(anchor)
            m = self.shared_projection(memory)
            h = self.shared_projection(hard_negative)
            features = torch.cat(
                [c, a, m, h, c * a, torch.abs(c - a), c * m, torch.abs(c - m), c * h, torch.abs(c - h), context],
                dim=-1,
            )
            return self.scorer(self.norm(features))

    return TargetConditionedScorer()


def load_dataset() -> dict[str, np.ndarray]:
    if not DATASET.is_file() or not DATASET_MANIFEST.is_file():
        raise RuntimeError("N69 dataset is missing; run --mode materialize first")
    payload = np.load(DATASET, allow_pickle=False)
    arrays = {key: payload[key] for key in payload.files}
    required = {"candidate", "anchor", "memory", "hard_negative", "context", "label", "group", "group_split", "group_has_positive", "temporal_left", "temporal_right"}
    missing = sorted(required - set(arrays))
    if missing:
        raise RuntimeError(f"N69 dataset missing arrays: {missing}")
    if arrays["candidate"].shape[1:] != (FEAT_DIM,) or arrays["anchor"].shape != arrays["candidate"].shape or arrays["memory"].shape != arrays["candidate"].shape or arrays["hard_negative"].shape != arrays["candidate"].shape:
        raise RuntimeError("N69 raw embedding shapes are inconsistent")
    if arrays["context"].shape[1:] != (SCALAR_DIM,) or arrays["context"].shape[0] != arrays["label"].shape[0]:
        raise RuntimeError("N69 context/label shapes are inconsistent")
    numeric = [arrays[key] for key in ("candidate", "anchor", "memory", "hard_negative", "context")]
    if not all(np.all(np.isfinite(value)) for value in numeric):
        raise RuntimeError("N69 dataset numeric arrays are not finite")
    if not np.array_equal(np.unique(arrays["group"]), np.arange(arrays["group_split"].size, dtype=arrays["group"].dtype)):
        raise RuntimeError("N69 group IDs are not contiguous")
    return arrays


def group_index(arrays: dict[str, np.ndarray]) -> dict[int, np.ndarray]:
    groups = arrays["group"]
    result: dict[int, np.ndarray] = {}
    for gid in range(int(arrays["group_split"].size)):
        indices = np.where(groups == gid)[0]
        if indices.size == 0:
            raise RuntimeError(f"empty N69 group {gid}")
        result[gid] = indices
    return result


def group_batches(group_ids: list[int], groups: dict[int, np.ndarray], batch_size: int, rng: np.random.Generator | None) -> Iterable[np.ndarray]:
    ordered = list(group_ids)
    if rng is not None:
        rng.shuffle(ordered)
    current: list[int] = []
    current_size = 0
    for gid in ordered:
        size = int(groups[gid].size)
        if current and current_size + size > batch_size:
            yield np.concatenate([groups[value] for value in current])
            current = []
            current_size = 0
        current.append(gid)
        current_size += size
    if current:
        yield np.concatenate([groups[value] for value in current])


def context_normalization(arrays: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    train_mask = np.isin(arrays["group"], np.where(arrays["group_split"] == SPLIT_CODE["train"])[0])
    if not np.any(train_mask):
        raise RuntimeError("N69 training split is empty")
    mean = arrays["context"][train_mask].mean(axis=0, dtype=np.float64).astype(np.float32)
    std = arrays["context"][train_mask].std(axis=0, dtype=np.float64).astype(np.float32)
    std[std < 1e-6] = 1.0
    return mean, std


def tensors_for_indices(arrays: dict[str, np.ndarray], indices: np.ndarray, mean: np.ndarray, std: np.ndarray, device: Any):
    import torch

    index = np.asarray(indices, dtype=np.int64)
    return (
        torch.as_tensor(arrays["candidate"][index], dtype=torch.float32, device=device),
        torch.as_tensor(arrays["anchor"][index], dtype=torch.float32, device=device),
        torch.as_tensor(arrays["memory"][index], dtype=torch.float32, device=device),
        torch.as_tensor(arrays["hard_negative"][index], dtype=torch.float32, device=device),
        torch.as_tensor((arrays["context"][index] - mean) / std, dtype=torch.float32, device=device),
        torch.as_tensor(arrays["label"][index], dtype=torch.long, device=device),
        torch.as_tensor(arrays["group"][index], dtype=torch.long, device=device),
    )


def batch_loss(model: Any, arrays: dict[str, np.ndarray], indices: np.ndarray, mean: np.ndarray, std: np.ndarray, device: Any) -> tuple[Any, dict[str, float]]:
    import torch
    import torch.nn.functional as F

    candidate, anchor, memory, hard_negative, context, labels, group_ids = tensors_for_indices(arrays, indices, mean, std, device)
    logits = model(candidate, anchor, memory, hard_negative, context)
    # The offline dataset keeps the auditable convention 1=target/positive,
    # 0=non-target.  The model contract is class 0=target and class 1=NONE,
    # which is also the convention used by replay.  Convert only at this CE
    # boundary; ranking, NONE and evaluation continue to use raw labels.
    model_labels = 1 - labels
    ce = F.cross_entropy(logits, model_labels, weight=torch.tensor([2.0, 1.0], device=device))
    target_logits = logits[:, 0]
    target_prob = torch.softmax(logits, dim=-1)[:, 0]
    unique_groups = torch.unique(group_ids)
    ranking_terms: list[Any] = []
    none_terms: list[Any] = []
    noop_terms: list[Any] = []
    for gid in unique_groups.tolist():
        mask = group_ids == int(gid)
        group_labels = labels[mask]
        group_target_logits = target_logits[mask]
        group_probs = target_prob[mask]
        if torch.any(group_labels == 1) and torch.any(group_labels == 0):
            positive = torch.max(group_target_logits[group_labels == 1])
            negative = torch.max(group_target_logits[group_labels == 0])
            ranking_terms.append(F.softplus(negative - positive + 0.2))
        max_probability = torch.max(group_probs)
        has_positive = bool(torch.any(group_labels == 1).detach().cpu())
        none_terms.append(F.binary_cross_entropy(max_probability, torch.tensor(float(has_positive), device=device)))
        if not has_positive:
            noop_terms.append(F.relu(max_probability - 0.5).square())
    ranking = torch.stack(ranking_terms).mean() if ranking_terms else torch.zeros((), device=device)
    none_loss = torch.stack(none_terms).mean() if none_terms else torch.zeros((), device=device)
    noop = torch.stack(noop_terms).mean() if noop_terms else torch.zeros((), device=device)
    total = ce + 0.5 * ranking + 0.5 * none_loss + 0.05 * noop
    return total, {"total": float(total.detach().cpu()), "cross_entropy": float(ce.detach().cpu()), "ranking": float(ranking.detach().cpu()), "none": float(none_loss.detach().cpu()), "no_op": float(noop.detach().cpu())}


def evaluate(model: Any, arrays: dict[str, np.ndarray], split_name: str, groups: dict[int, np.ndarray], mean: np.ndarray, std: np.ndarray, device: Any) -> dict[str, Any]:
    import torch
    import torch.nn.functional as F

    split_value = SPLIT_CODE[split_name]
    group_ids = [gid for gid, value in enumerate(arrays["group_split"].tolist()) if int(value) == split_value]
    predictions: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    group_predictions: list[float] = []
    group_targets: list[float] = []
    ranking_values: list[float] = []
    model.eval()
    with torch.no_grad():
        for batch_indices in group_batches(group_ids, groups, 4096, None):
            tensors = tensors_for_indices(arrays, batch_indices, mean, std, device)
            logits = model(*tensors[:5])
            probs = torch.softmax(logits, dim=-1)[:, 0].cpu().numpy()
            predictions.append(probs)
            labels.append(arrays["label"][batch_indices].astype(np.int64))
            for gid in np.unique(arrays["group"][batch_indices]):
                local = arrays["group"][batch_indices] == gid
                local_labels = arrays["label"][batch_indices][local]
                local_probs = probs[local]
                group_predictions.append(float(np.max(local_probs)))
                group_targets.append(float(np.any(local_labels == 1)))
                pos = local_probs[local_labels == 1]
                neg = local_probs[local_labels == 0]
                if pos.size and neg.size:
                    ranking_values.append(float(max(0.0, 0.2 - float(np.max(pos)) + float(np.max(neg)))))
    y = np.concatenate(labels) if labels else np.zeros(0, dtype=np.int64)
    p = np.concatenate(predictions) if predictions else np.zeros(0, dtype=np.float32)
    eps = 1e-7
    bce = float(np.mean(-(y * np.log(np.clip(p, eps, 1 - eps)) + (1 - y) * np.log(np.clip(1 - p, eps, 1 - eps))))) if y.size else None
    try:
        from sklearn.metrics import roc_auc_score

        auc = float(roc_auc_score(y, p)) if np.unique(y).size == 2 else None
    except Exception:
        auc = None
    group_bce = float(np.mean(-(np.asarray(group_targets) * np.log(np.clip(group_predictions, eps, 1 - eps)) + (1 - np.asarray(group_targets)) * np.log(np.clip(1 - np.asarray(group_predictions), eps, 1 - eps))))) if group_predictions else None
    composite = None if bce is None else bce + 0.5 * (float(np.mean(ranking_values)) if ranking_values else 0.0) + 0.5 * (group_bce or 0.0)
    return {
        "split": split_name,
        "examples": int(y.size),
        "positive": int(np.sum(y == 1)),
        "finite_predictions": bool(np.all(np.isfinite(p))),
        "bce": bce,
        "auc": auc,
        "accuracy_at_0_5": float(np.mean((p >= 0.5) == (y == 1))) if y.size else None,
        "group_none_bce": group_bce,
        "group_none_accuracy": float(np.mean((np.asarray(group_predictions) >= 0.5) == (np.asarray(group_targets) >= 0.5))) if group_predictions else None,
        "ranking_proxy": float(np.mean(ranking_values)) if ranking_values else None,
        "composite": composite,
        "probability_range": [float(np.min(p)), float(np.max(p))] if p.size else [None, None],
    }


def temporal_train_step(model: Any, arrays: dict[str, np.ndarray], mean: np.ndarray, std: np.ndarray, device: Any, optimizer: Any, pair_indices: np.ndarray) -> float | None:
    import torch
    import torch.nn.functional as F

    if pair_indices.size == 0:
        return None
    model.train()
    values: list[float] = []
    for start in range(0, pair_indices.shape[0], 1024):
        left = pair_indices[start : start + 1024, 0]
        right = pair_indices[start : start + 1024, 1]
        li = tensors_for_indices(arrays, left, mean, std, device)
        ri = tensors_for_indices(arrays, right, mean, std, device)
        optimizer.zero_grad(set_to_none=True)
        left_logits = model(*li[:5])[:, 0]
        right_logits = model(*ri[:5])[:, 0]
        loss = F.smooth_l1_loss(left_logits, right_logits)
        (0.1 * loss).backward()
        optimizer.step()
        values.append(float(loss.detach().cpu()))
    return float(np.mean(values)) if values else None


def smoke(device_name: str = "cuda") -> dict[str, Any]:
    ensure_model_protocol()
    events = load_event_map()
    event_id = sorted(events)[0]
    event = events[event_id]
    path = N54_RUNTIME / f"{event_id}.json"
    with path.open("r", encoding="utf-8") as handle:
        source = json.load(handle)
    raw = dict(source["variants"]["M2"]["frames"][0])
    raw["variant"] = "M2"
    if int(raw["frame"]) != event["event_frame"] + 1:
        raise RuntimeError("N69 smoke did not start at event+1")
    features = build_feature_arrays(raw, event, include_offline_label=False)
    device = torch_device(device_name)
    set_all_seeds(SEED)
    model = build_model().to(device)
    mean = np.zeros(SCALAR_DIM, dtype=np.float32)
    std = np.ones(SCALAR_DIM, dtype=np.float32)
    import torch
    import torch.nn.functional as F

    indices = np.arange(min(4, features["candidate"].shape[0]), dtype=np.int64)
    temporary_arrays = {
        "candidate": features["candidate"],
        "anchor": features["anchor"],
        "memory": features["memory"],
        "hard_negative": features["hard_negative"],
        "context": features["context"],
        "label": np.zeros(features["candidate"].shape[0], dtype=np.int64),
        "group": np.zeros(features["candidate"].shape[0], dtype=np.int64),
    }
    tensors = tensors_for_indices(temporary_arrays, indices, mean, std, device)
    logits = model(*tensors[:5])
    loss = F.cross_entropy(logits, torch.zeros(logits.shape[0], dtype=torch.long, device=device))
    loss.backward()
    if not torch.isfinite(loss):
        raise RuntimeError("N69 smoke loss is nonfinite")
    atomic_json(SMOKE_JSON, {
        "schema": "N69_TARGET_CONDITIONED_MODEL_SMOKE_V1",
        "status": "PASS",
        "created_at_utc": now(),
        "event_id": event_id,
        "sequence": event["sequence"],
        "frame": int(raw["frame"]),
        "event_frame": event["event_frame"],
        "causal_boundary": {"event_frame_memory_read": False, "first_visible_frame": event["event_frame"] + 1, "smoke_frame_is_event_plus_one": True},
        "model_protocol_sha256": sha256_file(MODEL_PROTOCOL),
        "input_shapes": {key: list(features[key].shape) for key in ("candidate", "anchor", "memory", "hard_negative", "context")},
        "parameter_count": int(sum(parameter.numel() for parameter in model.parameters())),
        "loss": float(loss.detach().cpu()),
        "finite_loss": bool(torch.isfinite(loss).item()),
        "runtime_future_gt_used": False,
        "interaction_source": "simulated_from_gt",
        "not_real_human_evidence": True,
        "production_authorized": False,
    })
    torch.save({"state_dict": model.state_dict(), "protocol_sha256": sha256_file(MODEL_PROTOCOL), "parameter_count": int(sum(parameter.numel() for parameter in model.parameters()))}, SMOKE_CKPT)
    reloaded = build_model().to(device)
    reloaded.load_state_dict(torch.load(SMOKE_CKPT, map_location=device, weights_only=False)["state_dict"])
    with torch.no_grad():
        reload_logits = reloaded(*tensors[:5])
    reload_error = float(torch.max(torch.abs(logits.detach() - reload_logits)).cpu())
    result = load_json(SMOKE_JSON)
    result["checkpoint"] = str(SMOKE_CKPT)
    result["reload_max_abs_error"] = reload_error
    result["reload_pass"] = reload_error < 1e-6
    atomic_json(SMOKE_JSON, result)
    if not result["reload_pass"]:
        raise RuntimeError(f"N69 smoke reload mismatch: {reload_error}")
    return result


def train(device_name: str = "cuda") -> dict[str, Any]:
    import torch

    protocol = ensure_model_protocol()
    if not SMOKE_JSON.is_file():
        raise RuntimeError("N69 training requires a completed CUDA smoke artifact")
    smoke_result = load_json(SMOKE_JSON)
    if smoke_result.get("status") != "PASS" or smoke_result.get("reload_pass") is not True:
        raise RuntimeError("N69 training requires smoke status=PASS and reload_pass=true")
    if smoke_result.get("model_protocol_sha256") != sha256_file(MODEL_PROTOCOL):
        raise RuntimeError("N69 smoke protocol hash does not match frozen model protocol")
    arrays = load_dataset()
    set_all_seeds(SEED)
    device = torch_device(device_name)
    groups = group_index(arrays)
    mean, std = context_normalization(arrays)
    model = build_model().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5.0e-4, weight_decay=1.0e-4)
    split_groups = {
        name: [gid for gid, value in enumerate(arrays["group_split"].tolist()) if int(value) == SPLIT_CODE[name]]
        for name in ("train", "validation", "holdout")
    }
    if any(not value for value in split_groups.values()):
        raise RuntimeError(f"N69 sequence split has empty group partition: {split_groups}")
    pair_indices = np.stack([arrays["temporal_left"], arrays["temporal_right"]], axis=1) if arrays["temporal_left"].size else np.zeros((0, 2), dtype=np.int64)
    rng = np.random.default_rng(SEED)
    best_state: dict[str, Any] | None = None
    best_epoch = 0
    best_val: float | None = None
    bad_epochs = 0
    history: list[dict[str, Any]] = []
    total_optimizer_steps = 0
    max_epochs = 30
    for epoch in range(1, max_epochs + 1):
        model.train()
        epoch_values: list[dict[str, float]] = []
        for batch_indices in group_batches(split_groups["train"], groups, 512, rng):
            optimizer.zero_grad(set_to_none=True)
            loss, components = batch_loss(model, arrays, batch_indices, mean, std, device)
            if not torch.isfinite(loss):
                raise RuntimeError(f"N69 nonfinite training loss at epoch {epoch}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            total_optimizer_steps += 1
            epoch_values.append(components)
        temporal_loss = temporal_train_step(model, arrays, mean, std, device, optimizer, pair_indices)
        if temporal_loss is not None:
            total_optimizer_steps += int(math.ceil(pair_indices.shape[0] / 1024))
        train_eval = evaluate(model, arrays, "train", groups, mean, std, device)
        validation_eval = evaluate(model, arrays, "validation", groups, mean, std, device)
        record = {
            "epoch": epoch,
            "train_loss": {key: float(np.mean([item[key] for item in epoch_values])) for key in epoch_values[0]} if epoch_values else {},
            "temporal_loss_unweighted": temporal_loss,
            "train": train_eval,
            "validation": validation_eval,
            "optimizer_steps_total": total_optimizer_steps,
        }
        history.append(record)
        val_score = validation_eval.get("composite")
        print(json.dumps({"epoch": epoch, "train_composite": train_eval.get("composite"), "validation_composite": val_score, "validation_auc": validation_eval.get("auc"), "temporal_loss": temporal_loss}, sort_keys=True), flush=True)
        if val_score is not None and (best_val is None or val_score < best_val - 1e-9):
            best_val = float(val_score)
            best_epoch = epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= 5:
                break
    if best_state is None:
        raise RuntimeError("N69 training did not produce a finite validation checkpoint")
    model.load_state_dict(best_state)
    holdout_eval = evaluate(model, arrays, "holdout", groups, mean, std, device)
    checkpoint_payload = {
        "schema": "N69_TARGET_CONDITIONED_SCORER_CHECKPOINT_V1",
        "state_dict": best_state,
        "protocol_sha256": sha256_file(MODEL_PROTOCOL),
        "feature_names": SCALAR_NAMES,
        "scalar_mean": mean,
        "scalar_std": std,
        "raw_embedding_dim": FEAT_DIM,
        "projection_dim": PROJECTION_DIM,
        "scalar_dim": SCALAR_DIM,
        "best_epoch": best_epoch,
        "production_authorized": False,
        "runtime_future_gt_used": False,
        "interaction_source": "simulated_from_gt",
    }
    TRAIN_DIR.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{CHECKPOINT.name}.", suffix=".pt", dir=str(TRAIN_DIR))
    os.close(fd)
    try:
        torch.save(checkpoint_payload, tmp)
        os.replace(tmp, CHECKPOINT)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)
    manifest = {
        "schema": "N69_TARGET_CONDITIONED_TRAINING_MANIFEST_V1",
        "status": "PASS_ACTUAL_GPU_TRAINING_COMPLETED",
        "created_at_utc": now(),
        "protocol": str(MODEL_PROTOCOL),
        "protocol_sha256": sha256_file(MODEL_PROTOCOL),
        "dataset": str(DATASET),
        "dataset_sha256": sha256_file(DATASET),
        "checkpoint": str(CHECKPOINT),
        "checkpoint_sha256": sha256_file(CHECKPOINT),
        "device": str(device),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "cuda_device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
        "seed": SEED,
        "architecture": "shared Linear(512->64) for candidate/anchor/memory/hard-negative; 10 projected terms + 34 context -> 128 -> 64 -> 2",
        "parameter_count": int(sum(parameter.numel() for parameter in model.parameters())),
        "raw_embedding_dim": FEAT_DIM,
        "projection_dim": PROJECTION_DIM,
        "scalar_dim": SCALAR_DIM,
        "train_sequence_count": len(protocol["sequence_split"]["train"]),
        "validation_sequence_count": len(protocol["sequence_split"]["validation"]),
        "holdout_sequence_count": len(protocol["sequence_split"]["holdout"]),
        "best_epoch": best_epoch,
        "best_validation_composite": best_val,
        "history": history,
        "holdout_evaluated_once_after_selection": True,
        "holdout": holdout_eval,
        "normalization_mean": mean.astype(float).tolist(),
        "normalization_std": std.astype(float).tolist(),
        "temporal_pairs": int(pair_indices.shape[0]),
        "optimizer_steps_total": total_optimizer_steps,
        "gt_loaded_for_offline_labels": True,
        "target_native_id_used_only_for_offline_labels": True,
        "numeric_public_id_feature": False,
        "target_native_id_feature": False,
        "runtime_future_gt_used": False,
        "interaction_source": "simulated_from_gt",
        "not_real_human_evidence": True,
        "real_human_tape": False,
        "real_sam3_full_loop": False,
        "production_authorized": False,
    }
    atomic_json(TRAIN_MANIFEST, manifest)
    atomic_json(STAGE03, {
        "schema": "N69_STAGE_03_STATUS_V1",
        "status": "PASS_ACTUAL_GPU_TRAINING_COMPLETED",
        "created_at_utc": now(),
        "protocol": str(MODEL_PROTOCOL),
        "protocol_sha256": sha256_file(MODEL_PROTOCOL),
        "dataset_manifest": str(DATASET_MANIFEST),
        "dataset_manifest_sha256": sha256_file(DATASET_MANIFEST),
        "dataset_sha256": sha256_file(DATASET),
        "smoke_artifact": str(SMOKE_JSON),
        "smoke_artifact_sha256": sha256_file(SMOKE_JSON),
        "training_manifest": str(TRAIN_MANIFEST),
        "training_manifest_sha256": sha256_file(TRAIN_MANIFEST),
        "checkpoint": str(CHECKPOINT),
        "checkpoint_sha256": sha256_file(CHECKPOINT),
        "training_device": str(device),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "input_sequence_split": protocol["sequence_split"],
        "training": {
            "actual_gpu_training": bool(device.type == "cuda"),
            "architecture": manifest["architecture"],
            "parameter_count": manifest["parameter_count"],
            "best_epoch": best_epoch,
            "optimizer_steps_total": total_optimizer_steps,
            "holdout_evaluated_once_after_selection": True,
        },
        "runtime_future_gt_used": False,
        "interaction_source": "simulated_from_gt",
        "real_human_tape": False,
        "real_sam3_full_loop": False,
        "production_authorized": False,
        "next_stage": "N69_STAGE_04_PAIRED_REPLAY",
    })
    return manifest


def load_trained_model(device_name: str = "cpu") -> tuple[Any, np.ndarray, np.ndarray, Any]:
    import torch

    if not CHECKPOINT.is_file() or not TRAIN_MANIFEST.is_file():
        raise RuntimeError("N69 trained checkpoint is missing")
    payload = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    if payload.get("protocol_sha256") != sha256_file(MODEL_PROTOCOL):
        raise RuntimeError("N69 checkpoint protocol hash mismatch")
    if payload.get("feature_names") != SCALAR_NAMES:
        raise RuntimeError("N69 checkpoint scalar feature contract mismatch")
    device = torch_device(device_name)
    model = build_model().to(device)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    mean = np.asarray(payload["scalar_mean"], dtype=np.float32)
    std = np.asarray(payload["scalar_std"], dtype=np.float32)
    if mean.shape != (SCALAR_DIM,) or std.shape != (SCALAR_DIM,) or not np.all(np.isfinite(mean)) or not np.all(np.isfinite(std)):
        raise RuntimeError("N69 checkpoint normalization contract invalid")
    return model, mean, std, device


def runtime_branch_payload(branch: dict[str, Any], assignment: np.ndarray, scores: np.ndarray, label: str, pids: list[int]) -> dict[str, Any]:
    candidates = branch.get("candidate_rows", [])
    rows = branch.get("rows", [])
    assignment_public = [pids[int(col)] if int(col) >= 0 else None for col in assignment]
    return {
        "branch": label,
        "frame": int(branch.get("frame")),
        "candidate_count": len(candidates),
        "candidate_rows": [dict(row) for row in candidates],
        "rows": [dict(row) for row in rows],
        "public_id_order": list(pids),
        "assignment_columns": assignment.astype(int).tolist(),
        "assignment_public_ids": assignment_public,
        "score_matrix": np.asarray(scores, dtype=np.float32).astype(float).tolist(),
        "runtime_future_gt_used": False,
    }


def apply_model_sidecar(model: Any, features: dict[str, Any], mean: np.ndarray, std: np.ndarray, device: Any) -> dict[str, Any]:
    import torch

    n = int(features["candidate"].shape[0])
    temporary = {
        "candidate": features["candidate"],
        "anchor": features["anchor"],
        "memory": features["memory"],
        "hard_negative": features["hard_negative"],
        "context": features["context"],
        "label": np.zeros(n, dtype=np.int64),
        "group": np.zeros(n, dtype=np.int64),
    }
    indices = np.arange(n, dtype=np.int64)
    tensors = tensors_for_indices(temporary, indices, mean, std, device)
    with torch.no_grad():
        logits = model(*tensors[:5]).detach().cpu().numpy().astype(np.float32)
    if logits.shape != (n, 2) or not np.all(np.isfinite(logits)):
        raise RuntimeError("N69 runtime model emitted nonfinite or malformed logits")
    logits_delta = logits[:, 0] - logits[:, 1]
    probabilities = np.exp(logits - np.max(logits, axis=1, keepdims=True))
    probabilities = probabilities / np.sum(probabilities, axis=1, keepdims=True)
    target_probability = probabilities[:, 0].astype(np.float32)
    none = bool(features["target_column"] is None or target_probability.size == 0 or float(np.max(target_probability)) < 0.5)
    residual = np.zeros(n, dtype=np.float32) if none else (2.0 * np.tanh(logits_delta / 2.0)).astype(np.float32)
    adjusted = np.asarray(features["base"], dtype=np.float32).copy()
    target_col = features["target_column"]
    if target_col is not None and not none:
        valid = np.isfinite(adjusted[:, int(target_col)]) & (adjusted[:, int(target_col)] > -1.0e8)
        adjusted[valid, int(target_col)] += residual[valid]
    if not np.all(np.isfinite(adjusted)):
        raise RuntimeError("N69 adjusted score matrix is nonfinite")
    score_delta = adjusted - np.asarray(features["base"], dtype=np.float32)
    if target_col is None:
        other_column_max = 0.0
    else:
        other = score_delta.copy()
        other[:, int(target_col)] = 0.0
        other_column_max = float(np.max(np.abs(other))) if other.size else 0.0
    if other_column_max > 1.0e-7:
        raise RuntimeError("N69 target-conditioned sidecar changed a non-target public-ID column")
    return {
        "target_logits": logits[:, 0].astype(float).tolist(),
        "none_logits": logits[:, 1].astype(float).tolist(),
        "target_probability": target_probability.astype(float).tolist(),
        "logit_delta_target_minus_none": logits_delta.astype(float).tolist(),
        "residual_target_public_column": residual.astype(float).tolist(),
        "target_column": target_col,
        "none_predicted": none,
        "none_threshold": 0.5,
        "residual_bound": 2.0,
        "adjusted_scores": adjusted.astype(float).tolist(),
        "score_cells_changed": int(np.sum(np.abs(score_delta) > 1.0e-12)),
        "max_abs_score_delta": float(np.max(np.abs(score_delta))) if score_delta.size else 0.0,
        "non_target_column_max_abs_delta": other_column_max,
        "target_column_only": True,
        "runtime_future_gt_used": False,
    }


def replay(device_name: str = "cuda") -> dict[str, Any]:
    ensure_model_protocol()
    events = load_event_map()
    model, mean, std, device = load_trained_model(device_name)
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    completed = 0
    runtime_frames = 0
    for event_id in sorted(events):
        event = events[event_id]
        path = N54_RUNTIME / f"{event_id}.json"
        if not path.is_file():
            raise RuntimeError(f"N69 source runtime missing: {path}")
        with path.open("r", encoding="utf-8") as handle:
            source = json.load(handle)
        artifact = {
            "schema": "N69_TARGET_CONDITIONED_RUNTIME_EVENT_V1",
            "status": "PASS",
            "created_at_utc": now(),
            "event_id": event_id,
            "sequence": event["sequence"],
            "action_type": event["action_type"],
            "event_frame": event["event_frame"],
            "first_event_memory_visible_frame": event["event_frame"] + 1,
            "target_public_id_event_input": event["target_public_id"],
            "mapping_version": MAPPING_VERSION,
            "checkpoint": str(CHECKPOINT),
            "checkpoint_sha256": sha256_file(CHECKPOINT),
            "runtime_boundary": {
                "event_frame_memory_read": False,
                "first_memory_visible_frame": event["event_frame"] + 1,
                "target_native_id_sent_to_runtime": False,
                "gt_loaded_in_worker": False,
                "future_gt_fields_sent": [],
                "runtime_future_gt_used": False,
            },
            "interaction_source": "simulated_from_gt",
            "not_real_human_evidence": True,
            "real_human_tape": False,
            "real_sam3_full_loop": False,
            "production_authorized": False,
            "variants": {},
        }
        for variant in VARIANTS:
            frames = source.get("variants", {}).get(variant, {}).get("frames", [])
            if len(frames) != FRAMES_PER_EVENT:
                raise RuntimeError(f"N69 replay frame denominator mismatch {event_id}/{variant}")
            expected = list(range(event["event_frame"] + 1, event["event_frame"] + 1 + FRAMES_PER_EVENT))
            if [int(item.get("frame", -1)) for item in frames] != expected:
                raise RuntimeError(f"N69 replay future range mismatch {event_id}/{variant}")
            frame_outputs: list[dict[str, Any]] = []
            for raw in frames:
                frame = dict(raw)
                frame["variant"] = variant
                features = build_feature_arrays(frame, event, include_offline_label=False)
                target_col = features["target_column"]
                baseline_assignment = features["source_assignment"]
                baseline_branch = runtime_branch_payload(frame["write_baseline"], baseline_assignment, features["base"], "CURRENT_CCAM_BASELINE", features["pids"])
                sidecar = apply_model_sidecar(model, features, mean, std, device)
                adjusted = np.asarray(sidecar["adjusted_scores"], dtype=np.float32)
                new_assignment = assignment_from_scores(adjusted) if target_col is not None else baseline_assignment.copy()
                new_branch = runtime_branch_payload(frame["write_baseline"], new_assignment, adjusted, "N69_TARGET_CONDITIONED", features["pids"])
                frame_outputs.append({
                    "frame": int(frame["frame"]),
                    "upstream_variant": variant,
                    "feature_audit": {
                        "candidate_count": int(features["candidate"].shape[0]),
                        "public_id_order": features["pids"],
                        "target_public_id": event["target_public_id"],
                        "target_column": target_col,
                        "target_memory_valid": bool(features["target_memory_valid"]),
                        "candidate_feature_sha256": features["candidate_feature_digests"],
                        "memory_feature_sha256": features["memory_feature_digests"],
                        "human_feature_sha256": features["human_feature_digest"],
                        "runtime_future_gt_used": False,
                    },
                    "methods": {
                        "CURRENT_CCAM_BASELINE": {
                            "assignment": baseline_branch,
                            "sidecar": {"reason": "frozen_write_baseline", "target_column": target_col, "score_cells_changed": 0, "runtime_future_gt_used": False},
                            "assignment_recomputed_from_adjusted_scores": False,
                            "runtime_future_gt_used": False,
                        },
                        "N69_TARGET_CONDITIONED": {
                            "assignment": new_branch,
                            "sidecar": sidecar,
                            "assignment_recomputed_from_adjusted_scores": target_col is not None,
                            "runtime_future_gt_used": False,
                        },
                    },
                    "candidate_stream_same_across_methods": True,
                    "public_id_axis_same_across_methods": True,
                    "mapping_sidecar_not_runtime_gt": True,
                    "event_frame_memory_read": False,
                    "first_event_memory_visible_frame": event["event_frame"] + 1,
                    "is_future_frame": True,
                    "runtime_future_gt_used": False,
                })
                runtime_frames += 1
            artifact["variants"][variant] = {"frame_count": len(frame_outputs), "frames": frame_outputs}
        atomic_json(ARTIFACT_DIR / f"{event_id}.json", artifact)
        completed += 1
        print(json.dumps({"replayed_events": completed, "event_id": event_id, "runtime_frames": runtime_frames}, sort_keys=True), flush=True)
        del source
        gc.collect()
    status = {
        "schema": "N69_TARGET_CONDITIONED_RUNTIME_STATUS_V1",
        "status": "PASS_RUNTIME_REPLAY",
        "created_at_utc": now(),
        "protocol": str(MODEL_PROTOCOL),
        "protocol_sha256": sha256_file(MODEL_PROTOCOL),
        "checkpoint": str(CHECKPOINT),
        "checkpoint_sha256": sha256_file(CHECKPOINT),
        "outputs": {"event_artifacts": str(ARTIFACT_DIR)},
        "metrics": {"event_count": completed, "frames": runtime_frames, "expected_events": EVENT_COUNT, "expected_frames": EVENT_COUNT * len(VARIANTS) * FRAMES_PER_EVENT},
        "gate_checks": {
            "all_24_events": completed == EVENT_COUNT,
            "all_5_variants": True,
            "all_100_frames": runtime_frames == EVENT_COUNT * len(VARIANTS) * FRAMES_PER_EVENT,
            "same_candidate_stream": True,
            "same_public_id_axis": True,
            "target_column_only": True,
            "event_frame_memory_read_false": True,
            "first_memory_visible_at_event_plus_one": True,
            "target_native_id_sent_to_runtime": False,
            "runtime_future_gt_false": True,
            "production_authorized": False,
        },
        "interaction_source": "simulated_from_gt",
        "not_real_human_evidence": True,
        "real_human_tape": False,
        "real_sam3_full_loop": False,
        "runtime_future_gt_used": False,
        "production_authorized": False,
    }
    atomic_json(RUNTIME_STATUS, status)
    atomic_json(STAGE04, {
        "schema": "N69_STAGE_04_STATUS_V1",
        "status": "PASS_RUNTIME_REPLAY_COMPLETE",
        "created_at_utc": now(),
        "runtime_status": str(RUNTIME_STATUS),
        "event_artifacts": str(ARTIFACT_DIR),
        "gate_checks": status["gate_checks"],
        "runtime_future_gt_used": False,
        "interaction_source": "simulated_from_gt",
        "real_human_tape": False,
        "real_sam3_full_loop": False,
        "production_authorized": False,
        "next_action": "Run independent posthoc scorer and strict synthetic/production gate; do not use future outcomes in runtime.",
    })
    return status


def load_mapping_index() -> dict[tuple[str, str, int], dict[str, Any]]:
    if not MAPPING_ROWS.is_file():
        raise RuntimeError("N69 mapping audit rows are missing")
    result: dict[tuple[str, str, int], dict[str, Any]] = {}
    with MAPPING_ROWS.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            key = (str(row["event_id"]), str(row["variant"]), int(row["frame"]))
            if key in result:
                raise RuntimeError(f"duplicate N69 mapping audit key {key}")
            result[key] = row
    expected = EVENT_COUNT * len(VARIANTS) * FRAMES_PER_EVENT
    if len(result) != expected:
        raise RuntimeError(f"N69 mapping index expected {expected} rows, found {len(result)}")
    return result


def assignment_public_ids(branch: dict[str, Any]) -> list[int | None]:
    values = branch.get("assignment_public_ids")
    if not isinstance(values, list):
        raise RuntimeError("runtime branch assignment_public_ids missing")
    return [None if value is None else int(value) for value in values]


def native_assignment_map(branch: dict[str, Any]) -> dict[int, int | None]:
    rows = branch.get("candidate_rows", [])
    assignments = assignment_public_ids(branch)
    if len(rows) != len(assignments):
        raise RuntimeError("runtime branch native/assignment axis mismatch")
    result: dict[int, int | None] = {}
    for index, row in enumerate(rows):
        native = int(row["native_tid"])
        if native in result:
            raise RuntimeError(f"duplicate native ID in runtime branch: {native}")
        result[native] = assignments[index]
    return result


def score_cell(matrix: Any, row: int | None, col: int | None) -> float | None:
    if row is None or col is None:
        return None
    try:
        value = float(matrix[row][col])
    except (IndexError, TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def frame_outcome(frame: dict[str, Any], event: dict[str, Any], mapping: dict[str, Any], method: str) -> dict[str, Any]:
    baseline = frame["methods"]["CURRENT_CCAM_BASELINE"]["assignment"]
    branch = frame["methods"][method]["assignment"]
    base_assignments = assignment_public_ids(baseline)
    assignments = assignment_public_ids(branch)
    if len(base_assignments) != len(assignments):
        raise RuntimeError(f"assignment vector mismatch {event['event_id']}/{frame['upstream_variant']}/{frame['frame']}")
    target_pid = int(event["target_public_id"])
    target_row_raw = mapping["reconciled_mapping"]["target_row"]
    target_row = None if target_row_raw is None else int(target_row_raw)
    target_present = target_row is not None
    target_assigned = assignments[target_row] if target_present and target_row < len(assignments) else None
    base_assigned = base_assignments[target_row] if target_present and target_row < len(base_assignments) else None
    target_correct = bool(target_present and target_assigned == target_pid)
    baseline_correct = bool(target_present and base_assigned == target_pid)
    utility = int(target_correct) - int(baseline_correct)
    baseline_native = native_assignment_map(baseline)
    treated_native = native_assignment_map(branch)
    native_ids = sorted(set(baseline_native) | set(treated_native))
    untouched = [native for native in native_ids if target_row is None or native != int(baseline["candidate_rows"][target_row]["native_tid"])]
    untouched_changed = sum(baseline_native.get(native) != treated_native.get(native) for native in untouched)
    assignment_changed = assignments != base_assignments
    target_assignment_changed = base_assigned != target_assigned
    sidecar = frame["methods"][method]["sidecar"]
    base_scores = np.asarray(baseline["score_matrix"], dtype=np.float64)
    new_scores = np.asarray(branch["score_matrix"], dtype=np.float64)
    if base_scores.shape != new_scores.shape or not np.all(np.isfinite(base_scores)) or not np.all(np.isfinite(new_scores)):
        raise RuntimeError("N69 posthoc score matrix invalid")
    target_col_raw = mapping["reconciled_mapping"]["target_public_column"]
    target_col = None if target_col_raw is None else int(target_col_raw)
    base_target_score = score_cell(base_scores, target_row, target_col)
    new_target_score = score_cell(new_scores, target_row, target_col)
    base_margin = None
    new_margin = None
    if target_row is not None and target_col is not None and target_row < base_scores.shape[0] and target_col < base_scores.shape[1]:
        base_alts = [float(base_scores[target_row, col]) for col in range(base_scores.shape[1]) if col != target_col and base_scores[target_row, col] > -1.0e8]
        new_alts = [float(new_scores[target_row, col]) for col in range(new_scores.shape[1]) if col != target_col and new_scores[target_row, col] > -1.0e8]
        base_margin = float(base_scores[target_row, target_col] - max(base_alts)) if base_alts else None
        new_margin = float(new_scores[target_row, target_col] - max(new_alts)) if new_alts else None
    changed_score = int(sidecar.get("score_cells_changed", 0)) > 0
    target_delta = None if base_target_score is None or new_target_score is None else float(new_target_score - base_target_score)
    if target_present and target_row is not None and target_col is not None:
        matrix_delta = new_scores - base_scores
        other_delta = matrix_delta.copy()
        other_delta[:, target_col] = 0.0
        non_target_delta = float(np.max(np.abs(other_delta))) if other_delta.size else 0.0
    else:
        non_target_delta = 0.0
    if utility > 0:
        classification = "B_CORRECT_TARGET_BOUNDARY_CROSSING"
    elif utility < 0:
        classification = "C_WRONG_TARGET_BOUNDARY_CROSSING"
    elif changed_score and not target_assignment_changed:
        classification = "A_SCORE_CHANGED_NO_TARGET_ASSIGNMENT_CHANGE"
    elif target_assignment_changed:
        classification = "N_NEUTRAL_TARGET_ASSIGNMENT_CHANGE"
    else:
        classification = "N_NO_CHANGE"
    if untouched_changed > 0:
        classification_with_untouched = "D_UNTOUCHED_ID_ASSIGNMENT_CHANGED"
    else:
        classification_with_untouched = classification
    return {
        "event_id": event["event_id"],
        "sequence": event["sequence"],
        "action_type": event["action_type"],
        "variant": frame["upstream_variant"],
        "frame": int(frame["frame"]),
        "horizon": int(frame["frame"]) - int(event["event_frame"]),
        "method": method,
        "target_public_id": target_pid,
        "target_row_posthoc": target_row,
        "target_candidate_present": target_present,
        "baseline_target_assigned_public_id": base_assigned,
        "target_assigned_public_id": target_assigned,
        "baseline_target_correct": baseline_correct,
        "target_correct": target_correct,
        "utility_delta_vs_current_ccam": utility,
        "assignment_changed": bool(assignment_changed),
        "target_assignment_changed": bool(target_assignment_changed),
        "correct_change": bool(utility > 0),
        "incorrect_change": bool(utility < 0),
        "neutral_change": bool(utility == 0),
        "untouched_assignment_changed_count": int(untouched_changed),
        "untouched_regression": bool(untouched_changed > 0),
        "score_changed": bool(changed_score),
        "score_cells_changed": int(sidecar.get("score_cells_changed", 0)),
        "max_abs_score_delta": float(sidecar.get("max_abs_score_delta", 0.0)),
        "target_score_delta": target_delta,
        "target_base_score": base_target_score,
        "target_new_score": new_target_score,
        "base_target_vs_distractor_margin": base_margin,
        "new_target_vs_distractor_margin": new_margin,
        "target_margin_delta": None if base_margin is None or new_margin is None else new_margin - base_margin,
        "non_target_column_max_abs_delta": non_target_delta,
        "classification": classification,
        "classification_with_untouched": classification_with_untouched,
        "mapping_target_scope_resolved": bool(mapping["reconciled_mapping"]["target_row_resolved"]),
        "mapping_old_conflict": bool(mapping["reconciled_mapping"]["conflict_rows_claiming_target_public"]),
        "candidate_integrity": bool(mapping["candidate_integrity"]["structural_valid"]),
        "runtime_future_gt_used": False,
    }


def bootstrap_ci(values_by_sequence: dict[str, list[float]], seed: int = BOOTSTRAP_SEED, reps: int = BOOTSTRAP_REPS) -> dict[str, Any]:
    cluster_means = {sequence: float(np.mean(values)) for sequence, values in values_by_sequence.items() if values}
    if not cluster_means:
        return {"sequence_count": 0, "mean": None, "ci95": [None, None], "seed": seed, "repetitions": reps, "cluster_means": {}}
    values = np.asarray(list(cluster_means.values()), dtype=np.float64)
    rng = np.random.default_rng(seed)
    draws = values[rng.integers(0, len(values), size=(reps, len(values)))]
    means = draws.mean(axis=1)
    return {
        "sequence_count": int(len(values)),
        "mean": float(np.mean(values)),
        "ci95": [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))],
        "seed": seed,
        "repetitions": reps,
        "cluster_means": dict(sorted(cluster_means.items())),
    }


def summarize_outcomes(outcomes: list[dict[str, Any]], method: str) -> dict[str, Any]:
    selected = [item for item in outcomes if item["method"] == method]
    if not selected:
        return {}
    by_horizon: dict[str, Any] = {}
    for horizon in HORIZONS:
        event_variant: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
        for item in selected:
            if 1 <= int(item["horizon"]) <= horizon:
                event_variant[(item["event_id"], item["variant"])].append(item)
        per_sequence: dict[str, list[float]] = defaultdict(list)
        event_variant_values: list[float] = []
        for (event_id, _variant), values in event_variant.items():
            value = float(np.mean([float(item["utility_delta_vs_current_ccam"]) for item in values]))
            event_variant_values.append(value)
            per_sequence[values[0]["sequence"]].append(value)
        by_horizon[str(horizon)] = {
            "mean_utility_delta_raw_event_variant": float(np.mean(event_variant_values)) if event_variant_values else None,
            "sequence_cluster_bootstrap": bootstrap_ci(per_sequence, seed=BOOTSTRAP_SEED + horizon),
            "event_variant_count": len(event_variant_values),
        }
    target_correct = np.asarray([int(item["target_correct"]) for item in selected], dtype=np.float64)
    baseline_correct = np.asarray([int(item["baseline_target_correct"]) for item in selected], dtype=np.float64)
    transitions = Counter(item["classification"] for item in selected)
    return {
        "frame_count": len(selected),
        "target_correct_rate": float(np.mean(target_correct)),
        "baseline_target_correct_rate": float(np.mean(baseline_correct)),
        "future_identity_error_rate": float(1.0 - np.mean(target_correct)),
        "target_candidate_recall": float(np.mean([int(item["target_candidate_present"]) for item in selected])),
        "score_change_frame_rate": float(np.mean([int(item["score_changed"]) for item in selected])),
        "assignment_change_rate": float(np.mean([int(item["assignment_changed"]) for item in selected])),
        "target_assignment_change_rate": float(np.mean([int(item["target_assignment_changed"]) for item in selected])),
        "correct_changes": int(sum(item["correct_change"] for item in selected)),
        "incorrect_changes": int(sum(item["incorrect_change"] for item in selected)),
        "neutral_changes": int(sum(item["neutral_change"] for item in selected)),
        "classification_counts": dict(sorted(transitions.items())),
        "untouched_assignment_changed_total": int(sum(item["untouched_assignment_changed_count"] for item in selected)),
        "untouched_regression_frame_rate": float(np.mean([int(item["untouched_regression"]) for item in selected])),
        "re_correction_opportunity_proxy": float(np.mean([int(item["target_candidate_present"] and not item["target_correct"]) for item in selected])),
        "candidate_absent_frame_count": int(sum(not item["target_candidate_present"] for item in selected)),
        "mapping_target_scope_resolved_rate": float(np.mean([int(item["mapping_target_scope_resolved"]) for item in selected])),
        "mapping_old_conflict_frame_count": int(sum(item["mapping_old_conflict"] for item in selected)),
        "mean_max_abs_score_delta": float(np.mean([item["max_abs_score_delta"] for item in selected])),
        "mean_target_score_delta": float(np.mean([item["target_score_delta"] for item in selected if item["target_score_delta"] is not None])) if any(item["target_score_delta"] is not None for item in selected) else None,
        "mean_base_target_vs_distractor_margin": float(np.mean([item["base_target_vs_distractor_margin"] for item in selected if item["base_target_vs_distractor_margin"] is not None])) if any(item["base_target_vs_distractor_margin"] is not None for item in selected) else None,
        "mean_new_target_vs_distractor_margin": float(np.mean([item["new_target_vs_distractor_margin"] for item in selected if item["new_target_vs_distractor_margin"] is not None])) if any(item["new_target_vs_distractor_margin"] is not None for item in selected) else None,
        "future_iou": None,
        "future_idsw": None,
        "runtime_future_gt_used": False,
        "horizons": by_horizon,
    }


def load_parent_upstream_summary() -> dict[str, Any]:
    if not N68_RESULTS.is_file():
        return {"available": False, "reason": "N68 paired result missing"}
    parent = load_json(N68_RESULTS)
    return {
        "available": True,
        "path": str(N68_RESULTS),
        "sha256": sha256_file(N68_RESULTS),
        "methods_by_upstream_variant": {
            variant: {
                "target_conditioned_baseline": parent.get("by_upstream_variant", {}).get(variant, {}).get("CURRENT_CCAM_BASELINE"),
                "learned_local_parent": parent.get("by_upstream_variant", {}).get(variant, {}).get("LEARNED_LOCAL_ASSOCIATION"),
            }
            for variant in VARIANTS
        },
        "interpretation": "Read-only parent M0-M4 upstream baseline context; N69 does not rewrite or select from this result.",
    }


def score_replay() -> dict[str, Any]:
    runtime = load_json(RUNTIME_STATUS)
    if runtime.get("status") != "PASS_RUNTIME_REPLAY":
        raise RuntimeError("N69 runtime replay is not complete")
    events = load_event_map()
    mapping_index = load_mapping_index()
    artifacts = sorted(ARTIFACT_DIR.glob("*.json"))
    if len(artifacts) != EVENT_COUNT:
        raise RuntimeError(f"N69 expected {EVENT_COUNT} runtime artifacts, found {len(artifacts)}")
    outcomes: list[dict[str, Any]] = []
    diag_rows: list[dict[str, Any]] = []
    for artifact_path in artifacts:
        artifact = load_json(artifact_path)
        event_id = str(artifact.get("event_id"))
        if event_id not in events:
            raise RuntimeError(f"unknown N69 runtime event artifact {event_id}")
        event = events[event_id]
        if artifact.get("runtime_boundary", {}).get("runtime_future_gt_used") is not False:
            raise RuntimeError(f"runtime future GT boundary failed {event_id}")
        for variant in VARIANTS:
            frames = artifact.get("variants", {}).get(variant, {}).get("frames", [])
            if len(frames) != FRAMES_PER_EVENT:
                raise RuntimeError(f"N69 artifact frame denominator failed {event_id}/{variant}")
            for frame in frames:
                frame_key = (event_id, variant, int(frame["frame"]))
                if frame_key not in mapping_index:
                    raise RuntimeError(f"N69 mapping row missing for {frame_key}")
                mapping = mapping_index[frame_key]
                if frame.get("runtime_future_gt_used") is not False:
                    raise RuntimeError(f"N69 frame future GT boundary failed {frame_key}")
                for method in ("CURRENT_CCAM_BASELINE", "N69_TARGET_CONDITIONED"):
                    outcome = frame_outcome(frame, event, mapping, method)
                    outcomes.append(outcome)
                    if method == "N69_TARGET_CONDITIONED":
                        diag_rows.append(outcome)
    methods = {method: summarize_outcomes(outcomes, method) for method in ("CURRENT_CCAM_BASELINE", "N69_TARGET_CONDITIONED")}
    by_action: dict[str, Any] = {}
    for action in sorted({event["action_type"] for event in events.values()}):
        by_action[action] = {method: summarize_outcomes([item for item in outcomes if item["action_type"] == action], method) for method in methods}
    by_variant: dict[str, Any] = {}
    for variant in VARIANTS:
        by_variant[variant] = {method: summarize_outcomes([item for item in outcomes if item["variant"] == variant], method) for method in methods}
    by_sequence: dict[str, Any] = {}
    for sequence in sorted({event["sequence"] for event in events.values()}):
        by_sequence[sequence] = {method: summarize_outcomes([item for item in outcomes if item["sequence"] == sequence], method) for method in methods}
    new_summary = methods["N69_TARGET_CONDITIONED"]
    lower = {str(horizon): new_summary["horizons"][str(horizon)]["sequence_cluster_bootstrap"]["ci95"][0] for horizon in HORIZONS}
    mapping_summary = load_json(MAPPING_SUMMARY)
    runtime_gate = {
        "runtime_complete": runtime.get("status") == "PASS_RUNTIME_REPLAY",
        "all_events_24": runtime.get("metrics", {}).get("event_count") == EVENT_COUNT,
        "all_frames_12000": runtime.get("metrics", {}).get("frames") == EVENT_COUNT * len(VARIANTS) * FRAMES_PER_EVENT,
        "runtime_future_gt_false": runtime.get("runtime_future_gt_used") is False,
        "target_column_only": runtime.get("gate_checks", {}).get("target_column_only") is True,
        "candidate_frame_integrity_100": mapping_summary.get("candidate_frame_integrity_100") is True,
        "target_scope_mapping_100_on_available_candidates": mapping_summary.get("target_scope_mapping_100_on_available_candidates") is True,
        "mapping_formal_provenance_100": mapping_summary.get("full_native_local_global_public_provenance") is True,
    }
    synthetic_gate = {
        "status": "PASS" if all(value is True for value in runtime_gate.values()) and all(value > 0.0 for value in lower.values()) and new_summary["correct_changes"] > new_summary["incorrect_changes"] and new_summary["untouched_regression_frame_rate"] == 0.0 else "FAIL_FUTURE_EFFECT",
        "strict_lower_ci_by_horizon": lower,
        "correct_changes_gt_incorrect_changes": new_summary["correct_changes"] > new_summary["incorrect_changes"],
        "untouched_regression_safe": new_summary["untouched_regression_frame_rate"] == 0.0,
        "formal_mapping_provenance_100": runtime_gate["mapping_formal_provenance_100"],
        "candidate_frame_integrity_100": runtime_gate["candidate_frame_integrity_100"],
        "production_authorized": False,
    }
    result = {
        "schema": "N69_TARGET_CONDITIONED_PAIRED_RESULTS_V1",
        "status": "N69_SIMULATED_FUTURE_EFFECT_EVALUATED",
        "created_at_utc": now(),
        "protocol": str(MODEL_PROTOCOL),
        "protocol_sha256": sha256_file(MODEL_PROTOCOL),
        "runtime_status": str(RUNTIME_STATUS),
        "event_count": EVENT_COUNT,
        "variant_count": len(VARIANTS),
        "frame_count": len(outcomes),
        "interaction_source": "simulated_from_gt",
        "not_real_human_evidence": True,
        "real_human_tape": False,
        "real_sam3_full_loop": False,
        "runtime_future_gt_used": False,
        "gt_loaded_only_posthoc": True,
        "evaluation_boundary": {
            "candidate_generation_changed": False,
            "hungarian_solver_changed": False,
            "same_candidate_stream": True,
            "same_public_id_axis": True,
            "target_column_only": True,
            "mapping_version": MAPPING_VERSION,
            "future_iou_available": False,
            "future_idsw_available": False,
        },
        "methods": methods,
        "by_action_type": by_action,
        "by_upstream_variant": by_variant,
        "by_sequence": by_sequence,
        "frozen_n68_upstream_context": load_parent_upstream_summary(),
        "runtime_gate": runtime_gate,
        "synthetic_science_gate": synthetic_gate,
        "production_evidence_gate": {"status": "BLOCKED_NO_REAL_HUMAN_TAPE_OR_REAL_SAM3_FULL_LOOP", "real_human_tape": False, "real_sam3_full_loop": False, "production_authorized": False},
        "failure_root_cause": "N69 tests a new raw-512-D target-conditioned scorer after mapping target-scope reconciliation. Any positive point estimate remains synthetic GT-simulated; strict gate requires positive sequence-cluster lower CI and zero untouched regression.",
        "outputs": {"event_artifacts": str(ARTIFACT_DIR), "paired_results": str(RESULTS), "assignment_diagnostics": str(ASSIGNMENT_DIAG)},
    }
    atomic_jsonl(ASSIGNMENT_DIAG, diag_rows)
    atomic_json(RESULTS, result)
    atomic_json(POSTHOC_STATUS, {
        "schema": "N69_POSTHOC_SCORE_STATUS_V1",
        "status": "PASS_POSTHOC_SCORED_STRICT_GATE_RECORDED",
        "created_at_utc": now(),
        "paired_results": str(RESULTS),
        "assignment_diagnostics": str(ASSIGNMENT_DIAG),
        "event_count": EVENT_COUNT,
        "runtime_frames_per_method": EVENT_COUNT * len(VARIANTS) * FRAMES_PER_EVENT,
        "runtime_future_gt_used": False,
        "gt_loaded_only_posthoc": True,
        "production_authorized": False,
    })
    atomic_json(STAGE04, {
        "schema": "N69_STAGE_04_STATUS_V1",
        "status": "PASS_PAIRED_REPLAY_POSTHOC_SCORED",
        "created_at_utc": now(),
        "runtime_status": str(RUNTIME_STATUS),
        "paired_results": str(RESULTS),
        "assignment_diagnostics": str(ASSIGNMENT_DIAG),
        "gate_checks": runtime_gate,
        "synthetic_science_gate": synthetic_gate,
        "runtime_future_gt_used": False,
        "interaction_source": "simulated_from_gt",
        "real_human_tape": False,
        "real_sam3_full_loop": False,
        "production_authorized": False,
        "next_action": "If the no-trimming strict gate fails, classify the first boundary failure and run only a pre-registered isolated alternative; do not run TACT or downstream learning.",
    })
    atomic_json(STAGE05, {
        "schema": "N69_STAGE_05_STATUS_V1",
        "status": "PASS_SYNTHETIC_REPLAY_GATE_REPORTED" if synthetic_gate["status"] == "PASS" else "FAIL_NO_TRIMMING_STRICT_FUTURE_EFFECT",
        "created_at_utc": now(),
        "paired_results": str(RESULTS),
        "synthetic_science_gate": synthetic_gate,
        "production_evidence_gate": result["production_evidence_gate"],
        "interaction_source": "simulated_from_gt",
        "real_human_tape": False,
        "real_sam3_full_loop": False,
        "runtime_future_gt_used": False,
        "production_authorized": False,
        "next_action": "Do not train calibration, selector, decoder LoRA, or modify production. Preserve this result and use its boundary diagnosis to select at most one isolated N69 alternative if justified.",
    })
    return result


def record_failure(stage: str, exc: BaseException) -> None:
    ATTEMPTS.mkdir(parents=True, exist_ok=True)
    existing = sorted(ATTEMPTS.glob(f"{stage}_failure_attempt*.json"))
    atomic_json(ATTEMPTS / f"{stage}_failure_attempt{len(existing) + 1}.json", {
        "schema": "N69_FAILURE_ARTIFACT_V1",
        "status": "FAIL_PRESERVED",
        "stage": stage,
        "created_at_utc": now(),
        "failure_root_cause": f"{type(exc).__name__}: {exc}",
        "traceback": traceback.format_exc(),
        "runtime_future_gt_used": False,
        "interaction_source": "simulated_from_gt",
        "real_human_tape": False,
        "real_sam3_full_loop": False,
        "production_authorized": False,
        "next_action": "Preserve this failure, repair only the first actionable root cause, and rerun the same frozen N69 unit.",
    })


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", required=True, choices=("protocol", "materialize", "smoke", "train", "replay", "score"))
    parser.add_argument("--device", default="cpu", choices=("cpu", "cuda"))
    args = parser.parse_args()
    if args.mode == "protocol":
        payload = ensure_model_protocol()
        print(json.dumps({"status": payload["status"], "path": str(MODEL_PROTOCOL), "sha256": sha256_file(MODEL_PROTOCOL)}, sort_keys=True))
    elif args.mode == "materialize":
        print(json.dumps(materialize(), sort_keys=True))
    elif args.mode == "smoke":
        print(json.dumps(smoke(args.device), sort_keys=True))
    elif args.mode == "train":
        payload = train(args.device)
        print(json.dumps({"status": payload["status"], "checkpoint": payload["checkpoint"], "best_epoch": payload["best_epoch"], "holdout": payload["holdout"]}, sort_keys=True))
    elif args.mode == "replay":
        print(json.dumps(replay(args.device), sort_keys=True))
    else:
        result = score_replay()
        print(json.dumps({"status": result["synthetic_science_gate"]["status"], "paired_results": str(RESULTS), "methods": {key: {"correct": value.get("correct_changes"), "incorrect": value.get("incorrect_changes"), "h20": value.get("horizons", {}).get("20", {}).get("mean_utility_delta_raw_event_variant")} for key, value in result["methods"].items()}}, sort_keys=True))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        record_failure(f"stage_03_{sys.argv[sys.argv.index('--mode') + 1] if '--mode' in sys.argv else 'unknown'}", exc)
        raise
