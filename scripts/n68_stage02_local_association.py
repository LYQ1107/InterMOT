"""N68 isolated identity-scoped local association experiment.

This module is intentionally a sidecar.  It does not import or modify the
production tracker, SAM3, the Hungarian implementation, or any MOT/OVMOT
configuration.  The runtime replay reads the frozen N54/N37 simulated event
stream and the target public ID supplied by the event.  Future GT is never an
input to feature construction or scoring; the target-native labels are loaded
only by dataset materialisation/posthoc scoring.

The command is split into protocol, materialise, smoke, train, replay, and
score modes so the causal/runtime boundary is visible in the artifacts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import shutil
import tempfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from scipy.optimize import linear_sum_assignment


ROOT = Path(__file__).resolve().parents[1]
N37_EVENTS = ROOT / "outputs/n37/real_event_manifest.json"
N42_PROTOCOL = ROOT / "outputs/n42/training/training_protocol.json"
N54_RUNTIME = ROOT / "outputs/n54/replay/runtime"
OUT = ROOT / "outputs/n68"
DIAG = OUT / "diagnosis"
TRAIN = OUT / "training"
REPLAY = OUT / "replay"
REPLAY_RUNTIME = REPLAY / "runtime"
REPLAY_ARTIFACTS = REPLAY / "event_artifacts"
ATTEMPTS = OUT / "attempts"

PROTOCOL = OUT / "stage_02_protocol.json"
DATASET = TRAIN / "n68_local_association_dataset.npz"
DATASET_MANIFEST = TRAIN / "n68_local_association_dataset_manifest.json"
SMOKE_CHECKPOINT = TRAIN / "n68_local_association_smoke.pt"
CHECKPOINT = TRAIN / "n68_identity_local_head.pt"
TRAINING_MANIFEST = TRAIN / "n68_identity_local_head_training_manifest.json"
REPLAY_STATUS = REPLAY / "runtime_status.json"
SCORED_RESULTS = REPLAY / "paired_replay_results.json"
SCORE_STATUS = REPLAY / "posthoc_score_status.json"
STAGE_02 = OUT / "stage_02_status.json"
STAGE_03 = OUT / "stage_03_status.json"
STAGE_04 = OUT / "stage_04_status.json"
STAGE_05 = OUT / "stage_05_status.json"
STAGE_06 = OUT / "stage_06_status.json"

EVENT_COUNT = 24
VARIANTS = ("M0", "M1", "M2", "M3", "M4")
MODES = ("CURRENT_CCAM_BASELINE", "FIXED_LOCAL_PROJECTION", "LEARNED_LOCAL_ASSOCIATION")
HORIZONS = (20, 50, 100)
SEED = 6801
BOOTSTRAP_SEED = 6808
BOOTSTRAP_REPS = 2000

ENGINEERED_FEATURE_NAMES = [
    "candidate_human_cosine",
    "candidate_target_memory_cosine",
    "human_target_memory_cosine",
    "candidate_human_minus_hard_negative_cosine",
    "candidate_target_memory_minus_competitor_memory_cosine",
    "target_memory_valid",
    "target_column_present",
    "target_column_max_score_tanh",
    "target_column_max_margin_tanh",
    "candidate_rank_in_target_column_norm",
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
    "valid_memory_fraction",
]
SCALAR_FEATURE_NAMES = [f"frozen_scalar_{i}" for i in range(8)]
FEATURE_NAMES = ENGINEERED_FEATURE_NAMES + SCALAR_FEATURE_NAMES


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
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


def atomic_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".jsonl", dir=str(path.parent))
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


def preserve_existing_file(path: Path, destination: Path) -> None:
    """Preserve a prior generated artifact without overwriting evidence."""
    if not path.is_file() or destination.is_file():
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{destination.name}.", suffix=".tmp", dir=str(destination.parent))
    os.close(fd)
    try:
        shutil.copyfile(path, tmp)
        with open(tmp, "rb") as handle:
            os.fsync(handle.fileno())
        os.replace(tmp, destination)
        dfd = os.open(str(destination.parent), os.O_DIRECTORY)
        try:
            os.fsync(dfd)
        finally:
            os.close(dfd)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def finite(value: Any, default: float = 0.0) -> float:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return float(default)
    return value if math.isfinite(value) else float(default)


def unit(value: Any, dim: int = 512) -> np.ndarray | None:
    try:
        arr = np.asarray(value, dtype=np.float32).reshape(-1)
    except (TypeError, ValueError):
        return None
    if arr.size != dim or not np.all(np.isfinite(arr)):
        return None
    norm = float(np.linalg.norm(arr))
    if norm <= 1.0e-6:
        return None
    return arr / norm


def cosine(left: Any, right: Any) -> float:
    a, b = unit(left), unit(right)
    if a is None or b is None:
        return 0.0
    return float(np.clip(np.dot(a, b), -1.0, 1.0))


def vector_digest(value: Any) -> str:
    arr = np.asarray(value, dtype=np.float32).reshape(-1)
    return sha256_bytes(arr.tobytes())


def score_tanh(value: float, scale: float = 5.0) -> float:
    return float(np.tanh(finite(value) / scale))


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_event_map() -> dict[str, dict[str, Any]]:
    payload = load_json(N37_EVENTS)
    if payload.get("status") not in {"PASS", "PARTIAL"}:
        raise RuntimeError(f"unexpected N37 manifest status: {payload.get('status')}")
    result: dict[str, dict[str, Any]] = {}
    for item in payload.get("events", []):
        event = item.get("event")
        if not isinstance(event, dict):
            raise RuntimeError("N37 event item does not contain an event record")
        event_id = str(item.get("protocol_candidate_id") or event.get("event_id"))
        target_pid = event.get("canonical_public_id", event.get("public_id"))
        target_native = event.get("target_native_tid")
        if target_pid is None or target_native is None:
            raise RuntimeError(f"event {event_id} lacks explicit offline target identity")
        if event_id in result:
            raise RuntimeError(f"duplicate frozen event {event_id}")
        result[event_id] = {
            "manifest_item": item,
            "event": event,
            "event_id": event_id,
            "sequence": str(event["sequence"]),
            "event_frame": int(event["frame"]),
            "target_public_id": int(target_pid),
            "target_native_tid": int(target_native),
            "human_embedding": event.get("human_embedding"),
            "competing_embeddings": event.get("competing_embeddings", []),
        }
    if len(result) != EVENT_COUNT:
        raise RuntimeError(f"expected {EVENT_COUNT} frozen N37 events, found {len(result)}")
    return result


def load_sequence_split() -> dict[str, str]:
    protocol = load_json(N42_PROTOCOL)
    split: dict[str, str] = {}
    for name in ("train", "validation", "holdout"):
        for sequence in protocol["sequence_split"][name]:
            sequence = str(sequence)
            if sequence in split:
                raise RuntimeError(f"sequence appears in multiple splits: {sequence}")
            split[sequence] = name
    return split


def protocol_payload() -> dict[str, Any]:
    split = load_sequence_split()
    by_split: dict[str, list[str]] = {"train": [], "validation": [], "holdout": []}
    for sequence, name in split.items():
        by_split[name].append(sequence)
    for values in by_split.values():
        values.sort()
    return {
        "schema": "N68_STAGE_02_LOCAL_ASSOCIATION_PROTOCOL_V1",
        "status": "FROZEN_BEFORE_MATERIALIZATION",
        "created_at_utc": now(),
        "hypothesis": "A known public ID needs a target-conditioned local candidate/NONE decision whose bounded residual can reach the existing global Hungarian boundary without changing candidate generation or solver.",
        "branch": "identity_scoped_local_association_without_causal_trimming",
        "frozen_inputs": {
            "n37_event_manifest": {"path": str(N37_EVENTS), "sha256": sha256_file(N37_EVENTS), "role": "offline simulated target public ID, human anchor, and target-native labels"},
            "n42_sequence_split": {"path": str(N42_PROTOCOL), "sha256": sha256_file(N42_PROTOCOL), "role": "sequence-disjoint train/validation/holdout split"},
            "n54_runtime": {"path": str(N54_RUNTIME), "role": "immutable candidate, memory, scalar, matrix and mapping stream"},
        },
        "feature_contract": {
            "feature_names": FEATURE_NAMES,
            "input_dim": len(FEATURE_NAMES),
            "candidate_embedding_dim": 512,
            "human_anchor_embedding_dim": 512,
            "numeric_public_id_feature": False,
            "target_native_id_feature": False,
            "raw_gt_feature": False,
            "backbone_frozen": True,
            "candidate_generation_changed": False,
            "hungarian_solver_changed": False,
        },
        "fixed_sidecar": {
            "target_column_only": True,
            "residual_bound": 2.0,
            "none_threshold": 0.5,
            "fixed_projection_signal": "0.7*candidate_human_cosine + 0.3*candidate_target_memory_cosine",
            "fixed_projection_logit_scale": 4.0,
            "fixed_projection_name": "FIXED_LOCAL_PROJECTION",
            "learned_residual": "residual_bound*tanh(local_head_logit)",
            "abstention": "if max candidate probability < none_threshold, subtract residual_bound/4 from the target column; this is an explicit NONE audit outcome, not a solver replacement",
        },
        "variants": {"upstream_frozen_variants": list(VARIANTS), "sidecar_modes": list(MODES)},
        "training": {
            "seed": SEED,
            "optimizer": "AdamW",
            "learning_rate": 0.001,
            "weight_decay": 0.0001,
            "batch_size": 512,
            "max_epochs": 30,
            "early_stopping_patience": 5,
            "selection": "earliest minimum validation BCE; holdout is evaluated once after selection",
            "loss": "class-weighted candidate targetness BCE plus fixed-weight within-frame max-positive-vs-max-negative softplus ranking loss",
            "ranking_loss_weight": 0.25,
            "architecture": "MLP(input_dim->64->32->1), ReLU, no dropout",
            "normalization": "train-sequence mean/std only",
            "holdout_used_for_selection": False,
            "threshold_scan": False,
            "seed_scan": False,
        },
        "sequence_split": by_split,
        "labels": {
            "positive": "candidate native_tid equals the event's explicit offline target_native_tid",
            "negative": "all other frozen candidate rows",
            "none": "frame has no positive candidate row; only used offline for abstention scoring",
            "gt_used": "offline event/label materialization and posthoc scoring only",
        },
        "runtime_boundary": {
            "runtime_future_gt_used": False,
            "gt_loaded_in_runtime_replay": False,
            "simulated_from_gt": True,
            "not_real_human_evidence": True,
            "real_human_tape": False,
            "real_sam3_full_loop": False,
            "production_authorized": False,
        },
    }


def ensure_protocol() -> dict[str, Any]:
    payload = protocol_payload()
    if PROTOCOL.is_file():
        existing = load_json(PROTOCOL)
        # The timestamp is intentionally ignored; all semantic fields must
        # remain frozen after the first write.
        for key in payload:
            if key == "created_at_utc":
                continue
            if existing.get(key) != payload.get(key):
                raise RuntimeError(f"frozen N68 Stage 02 protocol differs at {key}")
        return existing
    atomic_json(PROTOCOL, payload)
    return payload


def source_path(event_id: str) -> Path:
    path = N54_RUNTIME / f"{event_id}.json"
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def candidate_signature(rows: list[dict[str, Any]]) -> list[tuple[Any, Any, float]]:
    return [(row.get("native_tid"), row.get("box"), finite(row.get("confidence"))) for row in rows]


def public_ids(branch: dict[str, Any]) -> list[int]:
    values = branch.get("public_id_order")
    if not isinstance(values, list):
        raise RuntimeError("missing public_id_order")
    if any(value is None for value in values):
        raise RuntimeError("N54 public-ID axis contains null")
    return [int(value) for value in values]


def assignment_from_scores(scores: np.ndarray) -> np.ndarray:
    if scores.ndim != 2 or not np.all(np.isfinite(scores)):
        raise RuntimeError("nonfinite local assignment matrix")
    n, p = scores.shape
    result = np.full(n, -1, dtype=np.int64)
    if n == 0 or p == 0:
        return result
    rows, cols = linear_sum_assignment(-scores)
    result[rows] = cols
    return result


def normalize_assignment(value: Any, n: int, p: int) -> np.ndarray:
    arr = np.asarray(value, dtype=np.int64).reshape(-1)
    if arr.size != n:
        raise RuntimeError(f"assignment length {arr.size} != candidate count {n}")
    if np.any((arr >= p) | (arr < -1)):
        raise RuntimeError("assignment column out of range")
    return arr


def finite_rank(values: np.ndarray, index: int) -> int:
    valid = values[np.isfinite(values) & (values > -1.0e8)]
    if index < 0 or index >= values.size or not np.isfinite(values[index]) or values[index] <= -1.0e8:
        return values.size + 1
    return 1 + int(np.sum(valid > values[index]))


def top_margin(values: np.ndarray) -> float:
    valid = np.sort(values[np.isfinite(values) & (values > -1.0e8)])[::-1]
    if valid.size < 2:
        return 0.0
    return float(valid[0] - valid[1])


def target_physical_row(rows: list[dict[str, Any]], target_native: int) -> int | None:
    hits = [idx for idx, row in enumerate(rows) if int(row.get("native_tid")) == int(target_native)]
    if len(hits) > 1:
        raise RuntimeError(f"duplicate target native row {target_native}")
    return hits[0] if hits else None


def validate_frame_structure(frame: dict[str, Any], variant: str, event: dict[str, Any]) -> tuple[dict[str, Any], np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    write = frame.get("write_baseline")
    no_write = frame.get("no_write")
    if not isinstance(write, dict) or not isinstance(no_write, dict):
        raise RuntimeError(f"missing N54 branches {event['event_id']}/{variant}/{frame.get('frame')}")
    candidates = write.get("candidate_rows", [])
    no_candidates = no_write.get("candidate_rows", [])
    if candidate_signature(candidates) != candidate_signature(no_candidates):
        raise RuntimeError(f"candidate stream differs between branches {event['event_id']}/{variant}/{frame.get('frame')}")
    pids = public_ids(write)
    # N54's no_write branch is an intentionally different memory-state
    # counterfactual and may retire stale public IDs.  N68 local sidecar
    # evaluation is defined on the current write_baseline axis; requiring the
    # no_write axis to match would confuse a legitimate counterfactual state
    # change with a candidate/mapping defect.  The write_baseline and
    # write_plus branches, which are the actual paired local-association
    # inputs, must still share their axis exactly.
    plus = frame.get("write_plus_n54r1")
    if isinstance(plus, dict):
        if candidate_signature(plus.get("candidate_rows", [])) != candidate_signature(candidates):
            raise RuntimeError(f"write/plus candidate stream differs {event['event_id']}/{variant}/{frame.get('frame')}")
        if public_ids(plus) != pids:
            raise RuntimeError(f"write/plus public-ID axis differs {event['event_id']}/{variant}/{frame.get('frame')}")
    n, p = len(candidates), len(pids)
    base = np.asarray(write.get("score_matrix"), dtype=np.float32)
    candidate_features = np.asarray(frame.get("candidate_features_512"), dtype=np.float32)
    memory_vectors = np.asarray(frame.get("memory_vectors_512"), dtype=np.float32)
    memory_valid = np.asarray(frame.get("memory_valid"), dtype=bool).reshape(-1)
    if base.shape != (n, p) or not np.all(np.isfinite(base)):
        raise RuntimeError(f"invalid N54 baseline score matrix {event['event_id']}/{variant}/{frame.get('frame')}")
    if candidate_features.shape != (n, 512) or not np.all(np.isfinite(candidate_features)):
        raise RuntimeError(f"invalid N54 candidate feature matrix {event['event_id']}/{variant}/{frame.get('frame')}")
    if memory_vectors.shape != (p, 512) or not np.all(np.isfinite(memory_vectors)):
        raise RuntimeError(f"invalid N54 memory feature matrix {event['event_id']}/{variant}/{frame.get('frame')}")
    if memory_valid.size != p:
        raise RuntimeError(f"invalid N54 memory-valid axis {event['event_id']}/{variant}/{frame.get('frame')}")
    scalar = np.asarray(frame.get("scalar_features_8"), dtype=np.float32)
    if scalar.shape != (n * p, 8) or not np.all(np.isfinite(scalar)):
        raise RuntimeError(f"invalid N54 scalar feature matrix {event['event_id']}/{variant}/{frame.get('frame')}")
    if frame.get("runtime_future_gt_used") is not False:
        raise RuntimeError(f"N54 runtime future GT boundary is not false {event['event_id']}/{variant}/{frame.get('frame')}")
    if write.get("runtime_future_gt_used") is not False or no_write.get("runtime_future_gt_used") is not False:
        raise RuntimeError(f"N54 branch future GT boundary is not false {event['event_id']}/{variant}/{frame.get('frame')}")
    source_assignment = normalize_assignment(write.get("assignment_columns"), n, p)
    recomputed = assignment_from_scores(base)
    # Ties are possible in the frozen matrix; the source assignment is the
    # authoritative frozen result, but a large mismatch is an integrity error.
    if not np.array_equal(source_assignment, recomputed):
        source_score = sum(float(base[r, c]) for r, c in enumerate(source_assignment) if c >= 0)
        recomputed_score = sum(float(base[r, c]) for r, c in enumerate(recomputed) if c >= 0)
        if abs(source_score - recomputed_score) > 1.0e-5:
            raise RuntimeError(f"frozen assignment is not max-weight {event['event_id']}/{variant}/{frame.get('frame')}")
    return write, base, candidate_features, memory_vectors, memory_valid


def feature_matrix(frame: dict[str, Any], event: dict[str, Any], target_pid: int) -> tuple[np.ndarray, dict[str, Any]]:
    variant = str(frame.get("variant", "unknown"))
    write, base, candidate_features, memory_vectors, memory_valid = validate_frame_structure(frame, variant, event)
    candidates = write["candidate_rows"]
    pids = public_ids(write)
    n, p = base.shape
    target_col = pids.index(target_pid) if target_pid in pids else None
    human = unit(event.get("human_embedding"))
    if human is None:
        raise RuntimeError(f"invalid simulated human anchor {event['event_id']}")
    competing = [unit(value) for value in event.get("competing_embeddings", [])]
    competing = [value for value in competing if value is not None]
    candidate_units = [unit(value) for value in candidate_features]
    candidate_units = [value if value is not None else np.zeros(512, dtype=np.float32) for value in candidate_units]
    memory_units = [unit(value) for value in memory_vectors]
    valid_memory = np.asarray([bool(memory_valid[i]) and memory_units[i] is not None for i in range(p)], dtype=bool)
    memory_units = [value if value is not None else np.zeros(512, dtype=np.float32) for value in memory_units]
    target_memory = memory_units[target_col] if target_col is not None else np.zeros(512, dtype=np.float32)
    target_memory_is_valid = bool(target_col is not None and valid_memory[target_col])
    target_human_memory_cos = cosine(human, target_memory) if target_memory_is_valid else 0.0
    scalar = np.asarray(frame["scalar_features_8"], dtype=np.float32).reshape(n, p, 8)
    source_assignment = normalize_assignment(write.get("assignment_columns"), n, p)
    target_column_values = base[:, target_col] if target_col is not None else np.zeros(n, dtype=np.float32)
    target_column_rank = [finite_rank(target_column_values, row) for row in range(n)] if target_col is not None else [n + 1] * n
    target_column_max = float(np.max(target_column_values[target_column_values > -1.0e8])) if target_col is not None and np.any(target_column_values > -1.0e8) else 0.0
    target_column_margin = top_margin(target_column_values) if target_col is not None else 0.0
    target_rows = write.get("rows", [])
    rows: list[list[float]] = []
    for row, candidate in enumerate(candidates):
        cfeat = candidate_units[row]
        candidate_human = float(np.dot(cfeat, human))
        hard_negative_cos = max((float(np.dot(cfeat, value)) for value in competing), default=0.0)
        competitor_memory_cos = max(
            (float(np.dot(cfeat, memory_units[col])) for col in range(p) if col != target_col and valid_memory[col]),
            default=0.0,
        )
        target_memory_cos = float(np.dot(cfeat, target_memory)) if target_memory_is_valid else 0.0
        row_values = base[row]
        finite_row = row_values[np.isfinite(row_values) & (row_values > -1.0e8)]
        row_best = float(np.max(finite_row)) if finite_row.size else 0.0
        row_margin = top_margin(row_values)
        box = candidate.get("box", [0, 0, 0, 0])
        x1, y1, x2, y2 = [finite(value) for value in box[:4]]
        width = max(0.0, x2 - x1)
        height = max(0.0, y2 - y1)
        current_public = target_rows[row].get("public_id") if row < len(target_rows) and isinstance(target_rows[row], dict) else None
        assigned_target = bool(source_assignment[row] >= 0 and pids[int(source_assignment[row])] == target_pid)
        scalar_values = scalar[row, target_col].tolist() if target_col is not None else [0.0] * 8
        values = [
            candidate_human,
            target_memory_cos,
            target_human_memory_cos,
            candidate_human - hard_negative_cos,
            target_memory_cos - competitor_memory_cos,
            1.0 if target_memory_is_valid else 0.0,
            1.0 if target_col is not None else 0.0,
            score_tanh(target_column_max),
            score_tanh(target_column_margin),
            float((target_column_rank[row] - 1) / max(n - 1, 1)) if target_col is not None else 1.0,
            score_tanh(float(target_column_values[row])),
            score_tanh(row_best),
            score_tanh(row_margin),
            1.0 if current_public is not None and int(current_public) == target_pid else 0.0,
            1.0 if assigned_target else 0.0,
            float(np.clip(finite(candidate.get("confidence")), 0.0, 1.0)),
            float(np.clip(finite(candidate.get("native_age")) / 2000.0, 0.0, 1.0)),
            float(np.clip(width / 1920.0, 0.0, 1.0)),
            float(np.clip(height / 1080.0, 0.0, 1.0)),
            float(np.clip(((x1 + x2) * 0.5) / 1920.0, 0.0, 1.0)),
            float(np.clip(((y1 + y2) * 0.5) / 1080.0, 0.0, 1.0)),
            float(np.clip((int(frame["frame"]) - int(event["event_frame"])) / 100.0, 0.0, 1.0)),
            float(np.clip(n / 20.0, 0.0, 1.0)),
            float(np.mean(valid_memory)) if p else 0.0,
        ] + [finite(value) for value in scalar_values]
        if len(values) != len(FEATURE_NAMES) or not np.all(np.isfinite(values)):
            raise RuntimeError(f"feature contract violation {event['event_id']}/{frame.get('frame')}/{row}")
        rows.append(values)
    matrix = np.asarray(rows, dtype=np.float32)
    audit = {
        "frame": int(frame["frame"]),
        "candidate_count": n,
        "public_id_order": pids,
        "target_public_id": int(target_pid),
        "target_column": target_col,
        "target_column_present": target_col is not None,
        "target_memory_valid": target_memory_is_valid,
        "target_native_row_posthoc": target_physical_row(candidates, int(event["target_native_tid"])),
        "candidate_native_ids": [int(row["native_tid"]) for row in candidates],
        "candidate_feature_sha256": [vector_digest(value) for value in candidate_features],
        "memory_feature_sha256": [vector_digest(value) for value in memory_vectors],
        "human_feature_sha256": vector_digest(event["human_embedding"]),
        "source_assignment_columns": source_assignment.astype(int).tolist(),
        "source_assignment_public_ids": [pids[int(col)] if col >= 0 else None for col in source_assignment],
        "runtime_future_gt_used": False,
    }
    return matrix, audit


def iter_source_frames(events: dict[str, dict[str, Any]]) -> Iterable[tuple[dict[str, Any], str, dict[str, Any]]]:
    for event_id in sorted(events):
        event = events[event_id]
        source = load_json(source_path(event_id))
        if source.get("event_id") != event_id:
            raise RuntimeError(f"source event mismatch for {event_id}")
        if source.get("sequence") != event["sequence"] or int(source.get("event_frame")) != event["event_frame"]:
            raise RuntimeError(f"source event metadata mismatch for {event_id}")
        for variant in VARIANTS:
            frames = source.get("variants", {}).get(variant, {}).get("frames", [])
            if len(frames) != 100:
                raise RuntimeError(f"expected 100 future frames for {event_id}/{variant}, found {len(frames)}")
            seen: set[int] = set()
            for raw in frames:
                frame = dict(raw)
                frame["variant"] = variant
                frame_id = int(frame["frame"])
                if frame_id in seen or frame_id != event["event_frame"] + len(seen) + 1:
                    raise RuntimeError(f"non-contiguous/duplicate N54 future frame {event_id}/{variant}/{frame_id}")
                seen.add(frame_id)
                yield event, variant, frame


def materialise() -> dict[str, Any]:
    protocol = ensure_protocol()
    events = load_event_map()
    sequence_split = load_sequence_split()
    expected_files = sorted(N54_RUNTIME.glob("*.json"))
    if len(expected_files) != EVENT_COUNT:
        raise RuntimeError(f"expected {EVENT_COUNT} N54 event files, found {len(expected_files)}")
    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    sequences: list[str] = []
    event_ids: list[str] = []
    variants: list[str] = []
    frames: list[int] = []
    rows: list[int] = []
    group_ids: list[int] = []
    target_pids: list[int] = []
    target_natives: list[int] = []
    groups = 0
    frame_count = 0
    target_present_frames = 0
    positive_by_split: Counter[str] = Counter()
    examples_by_split: Counter[str] = Counter()
    frame_by_variant: Counter[str] = Counter()
    for event, variant, frame in iter_source_frames(events):
        target_pid = int(event["target_public_id"])
        matrix, _audit = feature_matrix(frame, event, target_pid)
        candidate_rows = frame["write_baseline"]["candidate_rows"]
        labels = np.asarray([int(int(row["native_tid"]) == int(event["target_native_tid"])) for row in candidate_rows], dtype=np.int8)
        if labels.sum() > 1:
            raise RuntimeError(f"multiple target-native positives {event['event_id']}/{frame['frame']}")
        split = sequence_split.get(event["sequence"])
        if split is None:
            raise RuntimeError(f"sequence absent from frozen split: {event['sequence']}")
        xs.append(matrix)
        ys.append(labels)
        n = matrix.shape[0]
        sequences.extend([event["sequence"]] * n)
        event_ids.extend([event["event_id"]] * n)
        variants.extend([variant] * n)
        frames.extend([int(frame["frame"])] * n)
        rows.extend(list(range(n)))
        group_ids.extend([groups] * n)
        target_pids.extend([target_pid] * n)
        target_natives.extend([int(event["target_native_tid"])] * n)
        groups += 1
        frame_count += 1
        frame_by_variant[variant] += 1
        if labels.sum():
            target_present_frames += 1
        positive_by_split[split] += int(labels.sum())
        examples_by_split[split] += n
        if frame_count % 100 == 0:
            print(json.dumps({"materialised_frames": frame_count, "event": event["event_id"], "variant": variant}, sort_keys=True), flush=True)
    if frame_count != EVENT_COUNT * len(VARIANTS) * 100:
        raise RuntimeError(f"frame denominator mismatch: {frame_count}")
    x = np.concatenate(xs, axis=0).astype(np.float32)
    y = np.concatenate(ys, axis=0).astype(np.int8)
    if x.ndim != 2 or x.shape[1] != len(FEATURE_NAMES) or not np.all(np.isfinite(x)):
        raise RuntimeError("materialised feature array violates finite/input-dim contract")
    if int(y.sum()) <= 0:
        raise RuntimeError("no positive target candidate labels")
    arrays = {
        "x": x,
        "y": y,
        "sequence": np.asarray(sequences),
        "event_id": np.asarray(event_ids),
        "variant": np.asarray(variants),
        "frame": np.asarray(frames, dtype=np.int32),
        "candidate_row": np.asarray(rows, dtype=np.int32),
        "group_id": np.asarray(group_ids, dtype=np.int32),
        "target_public_id": np.asarray(target_pids, dtype=np.int64),
        "target_native_tid_label_only": np.asarray(target_natives, dtype=np.int64),
    }
    atomic_npz(DATASET, arrays)
    manifest = {
        "schema": "N68_STAGE_02_LOCAL_ASSOCIATION_DATASET_MANIFEST_V1",
        "status": "PASS_DATASET_MATERIALIZED",
        "created_at_utc": now(),
        "protocol_sha256": sha256_file(PROTOCOL),
        "dataset": str(DATASET),
        "dataset_sha256": sha256_file(DATASET),
        "input_hashes": {"n37_event_manifest": sha256_file(N37_EVENTS), "n42_protocol": sha256_file(N42_PROTOCOL)},
        "shape": {"examples": int(x.shape[0]), "input_dim": int(x.shape[1]), "frames": frame_count, "groups": groups},
        "positive_examples": int(y.sum()),
        "negative_examples": int((y == 0).sum()),
        "target_present_frame_count": target_present_frames,
        "target_absent_frame_count": frame_count - target_present_frames,
        "examples_by_sequence_split": dict(examples_by_split),
        "positive_by_sequence_split": dict(positive_by_split),
        "frames_by_variant": dict(frame_by_variant),
        "feature_names": FEATURE_NAMES,
        "target_native_tid_is_label_only": True,
        "raw_gt_loaded": False,
        "offline_label_source": "N37 simulated event manifest target_native_tid; no raw GT loaded by this materializer",
        "runtime_future_gt_used": False,
        "interaction_source": "simulated_from_gt",
        "not_real_human_evidence": True,
        "real_human_tape": False,
        "real_sam3_full_loop": False,
        "production_authorized": False,
    }
    atomic_json(DATASET_MANIFEST, manifest)
    status = {
        "schema": "N68_STAGE_02_DATASET_STATUS_V1",
        "status": "PASS_DATASET_MATERIALIZED",
        "created_at_utc": now(),
        "outputs": {"protocol": str(PROTOCOL), "dataset": str(DATASET), "manifest": str(DATASET_MANIFEST)},
        "metrics": {"frames": frame_count, "groups": groups, "examples": int(x.shape[0]), "positive": int(y.sum()), "negative": int((y == 0).sum()), "target_present_frames": target_present_frames},
        "gate_checks": {"all_24_events": True, "all_5_variants": True, "all_100_frames": frame_count == 12000, "finite_features": True, "positive_labels": int(y.sum()) > 0, "sequence_disjoint_split": True, "raw_gt_loaded": False, "runtime_future_gt_used": False, "production_authorized": False},
        "provenance": {"interaction_source": "simulated_from_gt", "not_real_human_evidence": True, "real_human_tape": False, "real_sam3_full_loop": False},
    }
    atomic_json(OUT / "stage_02_dataset_status.json", status)
    return status


def set_all_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.use_deterministic_algorithms(False)
    except Exception:
        pass


def torch_model(input_dim: int):
    import torch.nn as nn
    return nn.Sequential(nn.Linear(input_dim, 64), nn.ReLU(), nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 1))


def torch_device(requested: str):
    import torch
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested for N68 smoke/train but is unavailable")
    return torch.device(requested)


def dataset_arrays() -> dict[str, np.ndarray]:
    if not DATASET.is_file() or not DATASET_MANIFEST.is_file():
        raise RuntimeError("N68 dataset is missing; run --mode materialize first")
    data = np.load(DATASET, allow_pickle=False)
    result = {key: data[key] for key in data.files}
    data.close()
    if result["x"].shape[1] != len(FEATURE_NAMES) or not np.all(np.isfinite(result["x"])):
        raise RuntimeError("invalid N68 dataset feature contract")
    return result


def standardized_arrays(data: dict[str, np.ndarray], sequence_split: dict[str, str]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    split = np.asarray([sequence_split[str(value)] for value in data["sequence"]])
    train_mask = split == "train"
    mean = data["x"][train_mask].mean(axis=0).astype(np.float32)
    std = data["x"][train_mask].std(axis=0).astype(np.float32)
    std[std < 1.0e-6] = 1.0
    x = ((data["x"] - mean) / std).astype(np.float32)
    return x, mean, std, split


def auc_score(y_true: np.ndarray, scores: np.ndarray) -> float | None:
    if len(np.unique(y_true)) < 2:
        return None
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(order) + 1, dtype=np.float64)
    positives = y_true == 1
    n_pos = int(positives.sum())
    n_neg = int((~positives).sum())
    return float((ranks[positives].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)) if n_pos and n_neg else None


def evaluate_logits(model: Any, x: np.ndarray, y: np.ndarray, indices: np.ndarray, device: Any, batch_size: int = 8192) -> dict[str, Any]:
    import torch
    import torch.nn.functional as F
    model.eval()
    values: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(indices), batch_size):
            batch = indices[start : start + batch_size]
            logits = model(torch.as_tensor(x[batch], dtype=torch.float32, device=device)).squeeze(-1)
            values.append(logits.detach().cpu().numpy())
    logits_np = np.concatenate(values) if values else np.empty(0, dtype=np.float32)
    labels = y[indices].astype(np.float32)
    tensor_logits = torch.as_tensor(logits_np)
    tensor_labels = torch.as_tensor(labels)
    bce = float(F.binary_cross_entropy_with_logits(tensor_logits, tensor_labels).item()) if len(labels) else None
    probs = 1.0 / (1.0 + np.exp(-np.clip(logits_np, -60.0, 60.0)))
    return {"examples": int(len(labels)), "positive": int(labels.sum()), "bce": bce, "auc": auc_score(labels.astype(np.int8), probs), "accuracy_at_0_5": float(np.mean((probs >= 0.5) == (labels >= 0.5))) if len(labels) else None, "probability_range": [float(probs.min()), float(probs.max())] if len(probs) else [None, None], "finite_predictions": bool(np.all(np.isfinite(logits_np)))}


def pairwise_batch_loss(logits: Any, labels: Any, groups: Any):
    import torch
    import torch.nn.functional as F
    losses = []
    for group in torch.unique(groups):
        mask = groups == group
        positive = logits[mask & (labels > 0.5)]
        negative = logits[mask & (labels <= 0.5)]
        if positive.numel() and negative.numel():
            losses.append(F.softplus(1.0 - positive.max() + negative.max()))
    if not losses:
        return logits.sum() * 0.0
    return torch.stack(losses).mean()


def smoke(device_name: str = "cuda") -> dict[str, Any]:
    import torch
    protocol = ensure_protocol()
    data = dataset_arrays()
    split_map = load_sequence_split()
    x, mean, std, split = standardized_arrays(data, split_map)
    train_indices = np.flatnonzero(split == "train")[:512]
    if len(train_indices) < 8:
        raise RuntimeError("N68 smoke has too few train examples")
    set_all_seeds(SEED)
    device = torch_device(device_name)
    model = torch_model(len(FEATURE_NAMES)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.0001)
    labels = torch.as_tensor(data["y"][train_indices], dtype=torch.float32, device=device)
    values = torch.as_tensor(x[train_indices], dtype=torch.float32, device=device)
    groups = torch.as_tensor(data["group_id"][train_indices], dtype=torch.long, device=device)
    pos = float(labels.sum().item()); neg = float(labels.numel() - pos)
    if pos <= 0 or neg <= 0:
        raise RuntimeError("N68 smoke lacks both target classes")
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=torch.as_tensor([neg / pos], device=device))
    history = []
    for step in range(3):
        optimizer.zero_grad(set_to_none=True)
        logits = model(values).squeeze(-1)
        bce = criterion(logits, labels)
        rank = pairwise_batch_loss(logits, labels, groups)
        loss = bce + 0.25 * rank
        if not torch.isfinite(loss):
            raise RuntimeError(f"nonfinite N68 smoke loss at step {step}")
        loss.backward()
        optimizer.step()
        history.append({"step": step + 1, "loss": float(loss.item()), "bce": float(bce.item()), "ranking": float(rank.item())})
    payload = {"schema": "N68_STAGE_02_LOCAL_HEAD_SMOKE_V1", "status": "PASS", "created_at_utc": now(), "protocol_sha256": sha256_file(PROTOCOL), "device": str(device), "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu", "parameter_count": int(sum(parameter.numel() for parameter in model.parameters())), "steps": history, "save_reload": True, "runtime_future_gt_used": False, "interaction_source": "simulated_from_gt", "not_real_human_evidence": True, "production_authorized": False}
    fd, tmp = tempfile.mkstemp(prefix=f".{SMOKE_CHECKPOINT.name}.", suffix=".pt", dir=str(SMOKE_CHECKPOINT.parent))
    os.close(fd)
    try:
        torch.save({"state_dict": model.state_dict(), "input_dim": len(FEATURE_NAMES), "mean": mean, "std": std, "protocol_sha256": sha256_file(PROTOCOL), "production_authorized": False}, tmp)
        reloaded = torch_model(len(FEATURE_NAMES)).to(device)
        reloaded.load_state_dict(torch.load(tmp, map_location=device, weights_only=False)["state_dict"])
        reloaded.eval()
        with torch.no_grad():
            if not torch.allclose(model(values), reloaded(values), atol=1.0e-6, rtol=1.0e-6):
                raise RuntimeError("N68 smoke save/reload changed logits")
        os.replace(tmp, SMOKE_CHECKPOINT)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)
    atomic_json(TRAIN / "n68_local_head_smoke.json", payload)
    return payload


def train(device_name: str = "cuda") -> dict[str, Any]:
    import torch
    protocol = ensure_protocol()
    data = dataset_arrays()
    split_map = load_sequence_split()
    x, mean, std, split = standardized_arrays(data, split_map)
    train_indices = np.flatnonzero(split == "train")
    validation_indices = np.flatnonzero(split == "validation")
    holdout_indices = np.flatnonzero(split == "holdout")
    if not len(train_indices) or not len(validation_indices) or not len(holdout_indices):
        raise RuntimeError("N68 sequence split lacks train/validation/holdout examples")
    set_all_seeds(SEED)
    device = torch_device(device_name)
    model = torch_model(len(FEATURE_NAMES)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.0001)
    generator = torch.Generator().manual_seed(SEED)
    train_ds = torch.utils.data.TensorDataset(torch.as_tensor(x[train_indices]), torch.as_tensor(data["y"][train_indices], dtype=torch.float32), torch.as_tensor(data["group_id"][train_indices], dtype=torch.long))
    loader = torch.utils.data.DataLoader(train_ds, batch_size=512, shuffle=True, generator=generator, drop_last=False)
    labels_train = data["y"][train_indices]
    pos = float(labels_train.sum()); neg = float(len(labels_train) - pos)
    if pos <= 0 or neg <= 0:
        raise RuntimeError("N68 training lacks both target classes")
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=torch.as_tensor([neg / pos], dtype=torch.float32, device=device))
    history: list[dict[str, Any]] = []
    best_val = float("inf")
    best_epoch = None
    best_state = None
    patience = 0
    for epoch in range(1, 31):
        model.train()
        loss_sum = 0.0; bce_sum = 0.0; rank_sum = 0.0; count = 0
        for batch_x, batch_y, batch_groups in loader:
            batch_x = batch_x.to(device); batch_y = batch_y.to(device); batch_groups = batch_groups.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch_x).squeeze(-1)
            bce = criterion(logits, batch_y)
            rank = pairwise_batch_loss(logits, batch_y, batch_groups)
            loss = bce + 0.25 * rank
            if not torch.isfinite(loss):
                raise RuntimeError(f"nonfinite N68 training loss at epoch {epoch}")
            loss.backward()
            optimizer.step()
            batch_n = int(batch_y.numel())
            loss_sum += float(loss.item()) * batch_n; bce_sum += float(bce.item()) * batch_n; rank_sum += float(rank.item()) * batch_n; count += batch_n
        train_eval = evaluate_logits(model, x, data["y"], train_indices, device)
        val_eval = evaluate_logits(model, x, data["y"], validation_indices, device)
        record = {"epoch": epoch, "train_loss": loss_sum / max(count, 1), "train_bce_weighted": bce_sum / max(count, 1), "train_ranking": rank_sum / max(count, 1), "train": train_eval, "validation": val_eval}
        history.append(record)
        print(json.dumps({"epoch": epoch, "train_loss": record["train_loss"], "validation_bce": val_eval["bce"], "validation_auc": val_eval["auc"]}, sort_keys=True), flush=True)
        val_bce = float(val_eval["bce"])
        if val_bce < best_val - 1.0e-12:
            best_val = val_bce; best_epoch = epoch; best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}; patience = 0
        else:
            patience += 1
        if patience >= 5:
            break
    if best_state is None or best_epoch is None:
        raise RuntimeError("N68 training did not produce a selected validation checkpoint")
    model.load_state_dict(best_state)
    holdout_eval = evaluate_logits(model, x, data["y"], holdout_indices, device)
    payload = {"schema": "N68_STAGE_02_LOCAL_HEAD_TRAINING_MANIFEST_V1", "status": "PASS_TRAINED_ISOLATED_SIDEcar", "created_at_utc": now(), "protocol": str(PROTOCOL), "protocol_sha256": sha256_file(PROTOCOL), "dataset": str(DATASET), "dataset_sha256": sha256_file(DATASET), "device": str(device), "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"), "cuda_device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu", "seed": SEED, "architecture": "MLP(input_dim->64->32->1)", "input_dim": len(FEATURE_NAMES), "parameter_count": int(sum(parameter.numel() for parameter in model.parameters())), "train_sequence_count": int(len(set(data["sequence"][train_indices].tolist()))), "validation_sequence_count": int(len(set(data["sequence"][validation_indices].tolist()))), "holdout_sequence_count": int(len(set(data["sequence"][holdout_indices].tolist()))), "best_epoch": int(best_epoch), "best_validation_bce": best_val, "history": history, "holdout_evaluated_once_after_selection": True, "holdout": holdout_eval, "normalization_mean": mean.astype(float).tolist(), "normalization_std": std.astype(float).tolist(), "checkpoint": str(CHECKPOINT), "runtime_future_gt_used": False, "gt_loaded_for_offline_labels": True, "interaction_source": "simulated_from_gt", "not_real_human_evidence": True, "real_human_tape": False, "real_sam3_full_loop": False, "production_authorized": False}
    fd, tmp = tempfile.mkstemp(prefix=f".{CHECKPOINT.name}.", suffix=".pt", dir=str(CHECKPOINT.parent)); os.close(fd)
    try:
        torch.save({"schema": "N68_IDENTITY_LOCAL_HEAD_CHECKPOINT_V1", "state_dict": best_state, "input_dim": len(FEATURE_NAMES), "feature_names": FEATURE_NAMES, "mean": mean, "std": std, "protocol_sha256": sha256_file(PROTOCOL), "best_epoch": best_epoch, "production_authorized": False, "runtime_future_gt_used": False, "interaction_source": "simulated_from_gt"}, tmp)
        os.replace(tmp, CHECKPOINT)
    finally:
        if os.path.exists(tmp): os.unlink(tmp)
    payload["checkpoint_sha256"] = sha256_file(CHECKPOINT)
    atomic_json(TRAINING_MANIFEST, payload)
    return payload


def load_trained_model(device_name: str = "cpu") -> tuple[Any, np.ndarray, np.ndarray, Any]:
    import torch
    if not CHECKPOINT.is_file() or not TRAINING_MANIFEST.is_file():
        raise RuntimeError("N68 trained checkpoint/manifest missing")
    payload = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    if payload.get("production_authorized") is not False or payload.get("runtime_future_gt_used") is not False:
        raise RuntimeError("invalid N68 checkpoint provenance")
    if payload.get("feature_names") != FEATURE_NAMES:
        raise RuntimeError("N68 checkpoint feature contract mismatch")
    device = torch_device(device_name)
    model = torch_model(len(FEATURE_NAMES)).to(device)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model, np.asarray(payload["mean"], dtype=np.float32), np.asarray(payload["std"], dtype=np.float32), device


def sigmoid(value: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(value, -60.0, 60.0)))


def sidecar_output(mode: str, x_raw: np.ndarray, base: np.ndarray, target_col: int | None, model: Any = None, mean: np.ndarray | None = None, std: np.ndarray | None = None, device: Any = None) -> dict[str, Any]:
    n, p = base.shape
    residual = np.zeros(n, dtype=np.float32)
    probabilities = np.zeros(n, dtype=np.float32)
    if mode == "CURRENT_CCAM_BASELINE":
        logits = np.zeros(n, dtype=np.float32)
        probabilities.fill(0.0)
        abstained = False
        reason = "baseline_no_local_residual"
    else:
        if mode == "FIXED_LOCAL_PROJECTION":
            signal = 0.7 * x_raw[:, 0] + 0.3 * x_raw[:, 1]
            logits = (4.0 * signal).astype(np.float32)
        elif mode == "LEARNED_LOCAL_ASSOCIATION":
            import torch
            if model is None or mean is None or std is None:
                raise RuntimeError("learned sidecar model is missing")
            with torch.no_grad():
                logits = model(torch.as_tensor((x_raw - mean) / std, dtype=torch.float32, device=device)).squeeze(-1).detach().cpu().numpy().astype(np.float32)
        else:
            raise RuntimeError(f"unknown N68 sidecar mode {mode}")
        probabilities = sigmoid(logits).astype(np.float32)
        abstained = bool(probabilities.size == 0 or float(np.max(probabilities)) < 0.5)
        if abstained:
            residual.fill(-0.5)
            reason = "explicit_none_threshold_below_0_5"
        else:
            residual = (2.0 * np.tanh(logits)).astype(np.float32)
            reason = "target_conditioned_residual_applied"
    adjusted = base.copy()
    if target_col is not None and mode != "CURRENT_CCAM_BASELINE":
        for row in range(n):
            if base[row, target_col] > -1.0e8 and np.isfinite(base[row, target_col]):
                adjusted[row, target_col] += residual[row]
    if not np.all(np.isfinite(adjusted)):
        raise RuntimeError("N68 sidecar produced nonfinite score matrix")
    return {"logits": logits.astype(float).tolist(), "probabilities": probabilities.astype(float).tolist(), "residual_target_column": residual.astype(float).tolist(), "target_column": target_col, "abstained_none": abstained, "reason": reason, "adjusted_scores": adjusted.astype(float).tolist(), "score_cells_changed": int(np.sum(np.abs(adjusted - base) > 1.0e-12)), "runtime_future_gt_used": False}


def branch_summary(branch: dict[str, Any], assignment: np.ndarray, scores: np.ndarray, pids: list[int], label: str) -> dict[str, Any]:
    return {"branch": label, "candidate_native_ids": [int(row["native_tid"]) for row in branch["candidate_rows"]], "candidate_rows": branch["candidate_rows"], "candidate_count": len(branch["candidate_rows"]), "public_id_order": pids, "assignment_columns": assignment.astype(int).tolist(), "assignment_public_ids": [pids[int(col)] if col >= 0 else None for col in assignment], "score_matrix": scores.astype(float).tolist(), "runtime_future_gt_used": False}


def replay(device_name: str = "cpu") -> dict[str, Any]:
    protocol = ensure_protocol()
    events = load_event_map()
    model, mean, std, device = load_trained_model(device_name)
    REPLAY_ARTIFACTS.mkdir(parents=True, exist_ok=True)
    completed = 0
    frame_total = 0
    method_cells = Counter()
    for event_id in sorted(events):
        event = events[event_id]
        source = load_json(source_path(event_id))
        output = {"schema": "N68_STAGE_02_LOCAL_ASSOCIATION_RUNTIME_EVENT_V1", "status": "PASS", "created_at_utc": now(), "event_id": event_id, "sequence": event["sequence"], "event_frame": event["event_frame"], "action_type": event["event"].get("action_type"), "target_public_id_event_input": event["target_public_id"], "target_native_tid_posthoc_label_only": event["target_native_tid"], "interaction_source": "simulated_from_gt", "not_real_human_evidence": True, "real_human_tape": False, "real_sam3_full_loop": False, "checkpoint": str(CHECKPOINT), "checkpoint_sha256": sha256_file(CHECKPOINT), "runtime_boundary": {"runtime_future_gt_used": False, "gt_loaded_in_worker": False, "target_native_tid_used_for_runtime_features": False, "future_gt_fields_sent": []}, "variants": {}}
        for variant in VARIANTS:
            frames_out: list[dict[str, Any]] = []
            frames = source["variants"][variant]["frames"]
            for raw in frames:
                frame = dict(raw); frame["variant"] = variant
                write, base, _cf, _mv, _valid = validate_frame_structure(frame, variant, event)
                x_raw, feature_audit = feature_matrix(frame, event, event["target_public_id"])
                pids = public_ids(write)
                target_col = pids.index(event["target_public_id"]) if event["target_public_id"] in pids else None
                source_assignment = normalize_assignment(write["assignment_columns"], len(write["candidate_rows"]), len(pids))
                methods: dict[str, Any] = {}
                for mode in MODES:
                    sidecar = sidecar_output(mode, x_raw, base, target_col, model, mean, std, device)
                    adjusted = np.asarray(sidecar["adjusted_scores"], dtype=np.float32)
                    assignment = source_assignment.copy() if mode == "CURRENT_CCAM_BASELINE" else assignment_from_scores(adjusted)
                    methods[mode] = {"sidecar": sidecar, "assignment": branch_summary(write, assignment, adjusted, pids, mode), "assignment_recomputed_from_adjusted_scores": True, "runtime_future_gt_used": False}
                    method_cells[mode] += int(sidecar["score_cells_changed"])
                frames_out.append({"frame": int(frame["frame"]), "candidate_feature_source": frame.get("candidate_feature_source"), "feature_audit": feature_audit, "candidate_features_digest": [vector_digest(value) for value in frame["candidate_features_512"]], "methods": methods, "candidate_stream_same_across_methods": True, "public_id_axis_same_across_methods": True, "memory_current_frame_write_hidden": int(frame["frame"]) == event["event_frame"], "first_event_memory_visible_frame": event["event_frame"] + 1, "runtime_future_gt_used": False})
                frame_total += 1
            output["variants"][variant] = {"frame_count": len(frames_out), "frames": frames_out}
        path = REPLAY_ARTIFACTS / f"{event_id}.json"
        atomic_json(path, output)
        completed += 1
        print(json.dumps({"replayed_events": completed, "event_id": event_id, "frames": frame_total}, sort_keys=True), flush=True)
    status = {"schema": "N68_STAGE_02_LOCAL_ASSOCIATION_RUNTIME_STATUS_V1", "status": "PASS_RUNTIME_REPLAY", "created_at_utc": now(), "protocol": str(PROTOCOL), "protocol_sha256": sha256_file(PROTOCOL), "checkpoint": str(CHECKPOINT), "checkpoint_sha256": sha256_file(CHECKPOINT), "outputs": {"event_artifacts": str(REPLAY_ARTIFACTS)}, "metrics": {"event_count": completed, "frames": frame_total, "expected_frames": EVENT_COUNT * len(VARIANTS) * 100, "score_cells_changed_by_mode": dict(method_cells)}, "gate_checks": {"all_24_events": completed == EVENT_COUNT, "all_5_variants": True, "all_100_frames": frame_total == EVENT_COUNT * len(VARIANTS) * 100, "same_candidate_stream": True, "same_public_id_axis": True, "same_hungarian_solver": True, "target_column_only_residual": True, "runtime_future_gt_false": True, "gt_loaded_in_worker": False, "production_authorized": False}, "interaction_source": "simulated_from_gt", "not_real_human_evidence": True, "real_human_tape": False, "real_sam3_full_loop": False, "runtime_future_gt_used": False}
    atomic_json(REPLAY_STATUS, status)
    return status


def assignment_map(branch: dict[str, Any]) -> dict[int, int | None]:
    rows = branch["candidate_rows"]
    assigned = branch["assignment_public_ids"]
    return {int(row["native_tid"]): assigned[index] for index, row in enumerate(rows)}


def frame_outcome(frame: dict[str, Any], method: str, baseline: dict[str, Any], event: dict[str, Any]) -> dict[str, Any]:
    branch = frame["methods"][method]["assignment"]
    base_branch = baseline["assignment"]
    target_native = int(event["target_native_tid"])
    target_pid = int(event["target_public_id"])
    rows = branch["candidate_rows"]
    target_row = target_physical_row(rows, target_native)
    target_present = target_row is not None
    target_assigned = branch["assignment_public_ids"][target_row] if target_present else None
    base_rows = base_branch["candidate_rows"]
    base_target_row = target_physical_row(base_rows, target_native)
    base_present = base_target_row is not None
    base_assigned = base_branch["assignment_public_ids"][base_target_row] if base_present else None
    correct = bool(target_present and target_assigned == target_pid)
    base_correct = bool(base_present and base_assigned == target_pid)
    base_map = assignment_map(base_branch)
    treated_map = assignment_map(branch)
    untouched_native = sorted(set(base_map) | set(treated_map))
    untouched_native = [native for native in untouched_native if native != target_native]
    untouched_changed = sum(base_map.get(native) != treated_map.get(native) for native in untouched_native)
    assignment_changed = branch["assignment_public_ids"] != base_branch["assignment_public_ids"]
    delta_utility = int(correct) - int(base_correct)
    sidecar = frame["methods"][method]["sidecar"]
    target_scores = np.asarray(branch["score_matrix"], dtype=np.float32)
    target_margin = None
    target_rank = None
    if target_present:
        target_col = sidecar.get("target_column")
        if target_col is not None and int(target_col) < target_scores.shape[1]:
            target_rank = finite_rank(target_scores[:, int(target_col)], target_row)
            alternatives = [float(target_scores[target_row, col]) for col in range(target_scores.shape[1]) if col != int(target_col) and target_scores[target_row, col] > -1.0e8]
            target_margin = float(target_scores[target_row, int(target_col)] - max(alternatives)) if alternatives else None
    return {"frame": int(frame["frame"]), "target_present": target_present, "target_assigned_public_id": target_assigned, "target_correct": correct, "baseline_target_present": base_present, "baseline_target_assigned_public_id": base_assigned, "baseline_target_correct": base_correct, "utility_delta_vs_current_ccam": delta_utility, "assignment_changed_vs_current_ccam": bool(assignment_changed), "correct_change": bool(delta_utility > 0), "incorrect_change": bool(delta_utility < 0), "neutral_change": bool(delta_utility == 0), "untouched_assignment_changed_count": int(untouched_changed), "untouched_regression": bool(untouched_changed > 0), "candidate_recall": target_present, "none_predicted": bool(sidecar.get("abstained_none", False)), "none_correct_posthoc": bool(bool(sidecar.get("abstained_none", False)) == (not target_present)), "score_cells_changed": int(sidecar.get("score_cells_changed", 0)), "target_rank": target_rank, "target_vs_distractor_margin": target_margin, "assignment_public_ids": branch["assignment_public_ids"], "runtime_future_gt_used": False}


def bootstrap_ci(values_by_sequence: dict[str, list[float]], seed: int = BOOTSTRAP_SEED, reps: int = BOOTSTRAP_REPS) -> dict[str, Any]:
    sequence_values = {seq: float(np.mean(values)) for seq, values in values_by_sequence.items() if values}
    if not sequence_values:
        return {"sequence_count": 0, "mean": None, "ci95": [None, None], "seed": seed, "repetitions": reps}
    values = np.asarray(list(sequence_values.values()), dtype=np.float64)
    rng = np.random.default_rng(seed)
    samples = values[rng.integers(0, len(values), size=(reps, len(values)))]
    means = samples.mean(axis=1)
    return {"sequence_count": int(len(values)), "mean": float(values.mean()), "ci95": [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))], "seed": seed, "repetitions": reps, "cluster_means": {key: float(value) for key, value in sorted(sequence_values.items())}}


def summarize_outcomes(outcomes: list[tuple[str, str, str, dict[str, Any]]], method: str) -> dict[str, Any]:
    selected = [(sequence, action, variant, outcome) for sequence, action, variant, outcome in outcomes if outcome["method"] == method]
    if not selected:
        return {}
    by_horizon: dict[str, Any] = {}
    for horizon in HORIZONS:
        horizon_values: list[float] = []
        by_sequence: dict[str, list[float]] = defaultdict(list)
        by_event_variant: dict[tuple[str, str], list[float]] = defaultdict(list)
        for sequence, action, variant, item in selected:
            by_event_variant[(item["event_id"], variant)].append(float(item["utility_delta_vs_current_ccam"]))
        for (event_id, variant), values in by_event_variant.items():
            # The list is naturally one value per frame; selecting the first H
            # frames enforces the frozen future window without using outcomes
            # for event selection.
            value = float(np.mean(values[:horizon]))
            event_sequence = next(seq for seq, act, var, out in selected if out["event_id"] == event_id and var == variant)
            by_sequence[event_sequence].append(value)
            horizon_values.append(value)
        by_horizon[str(horizon)] = {"mean_utility_delta_raw_event_variant": float(np.mean(horizon_values)) if horizon_values else None, "sequence_cluster_bootstrap": bootstrap_ci(by_sequence, seed=BOOTSTRAP_SEED + horizon)}
    target_correct = [int(item["target_correct"]) for _, _, _, item in selected]
    baseline_correct = [int(item["baseline_target_correct"]) for _, _, _, item in selected]
    deltas = [int(item["utility_delta_vs_current_ccam"]) for _, _, _, item in selected]
    transitions = Counter("correct" if value > 0 else "incorrect" if value < 0 else "neutral" for value in deltas)
    assignment_changes = sum(bool(item["assignment_changed_vs_current_ccam"]) for _, _, _, item in selected)
    score_changes = sum(int(item["score_cells_changed"]) > 0 for _, _, _, item in selected)
    untouched_changed = sum(int(item["untouched_assignment_changed_count"]) for _, _, _, item in selected)
    none_count = sum(bool(item["none_predicted"]) for _, _, _, item in selected)
    none_correct = sum(bool(item["none_correct_posthoc"]) for _, _, _, item in selected)
    return {"frame_count": len(selected), "target_correct_rate": float(np.mean(target_correct)), "baseline_target_correct_rate": float(np.mean(baseline_correct)), "future_identity_error_rate": float(1.0 - np.mean(target_correct)), "target_candidate_recall": float(np.mean([int(item["candidate_recall"]) for _, _, _, item in selected])), "score_change_frame_rate": float(score_changes / len(selected)), "assignment_change_rate": float(assignment_changes / len(selected)), "correct_changes": int(transitions["correct"]), "incorrect_changes": int(transitions["incorrect"]), "neutral_changes": int(transitions["neutral"]), "none_predicted_count": int(none_count), "none_accuracy_posthoc": float(none_correct / len(selected)), "untouched_assignment_changed_total": int(untouched_changed), "untouched_regression_frame_rate": float(sum(bool(item["untouched_regression"]) for _, _, _, item in selected) / len(selected)), "re_correction_opportunity_proxy": float(sum(bool(item["target_present"] and not item["target_correct"]) for _, _, _, item in selected) / len(selected)), "mean_target_rank": float(np.mean([item["target_rank"] for _, _, _, item in selected if item["target_rank"] is not None])) if any(item["target_rank"] is not None for _, _, _, item in selected) else None, "mean_target_vs_distractor_margin": float(np.mean([item["target_vs_distractor_margin"] for _, _, _, item in selected if item["target_vs_distractor_margin"] is not None])) if any(item["target_vs_distractor_margin"] is not None for _, _, _, item in selected) else None, "future_iou": None, "future_idsw": None, "id_switch_proxy_target_assignment": None, "horizons": by_horizon, "runtime_future_gt_used": False}


def score_replay() -> dict[str, Any]:
    if not REPLAY_STATUS.is_file():
        raise RuntimeError("N68 runtime replay status is missing")
    # The first scoring attempt is retained before the corrected scorer can
    # replace its output.  This is intentionally a full-file snapshot, not a
    # hand-edited rewrite of the erroneous counts.
    preserve_existing_file(
        SCORED_RESULTS,
        ATTEMPTS / "stage_02_posthoc_scoring_attempt1_preserved.json",
    )
    status = load_json(REPLAY_STATUS)
    if status.get("status") != "PASS_RUNTIME_REPLAY":
        raise RuntimeError("N68 runtime replay did not pass")
    events = load_event_map()
    artifact_paths = sorted(REPLAY_ARTIFACTS.glob("*.json"))
    if len(artifact_paths) != EVENT_COUNT:
        raise RuntimeError(f"expected {EVENT_COUNT} N68 runtime event artifacts, found {len(artifact_paths)}")
    all_outcomes: list[tuple[str, str, str, dict[str, Any]]] = []
    event_summaries: dict[str, Any] = {}
    for path in artifact_paths:
        artifact = load_json(path)
        event_id = str(artifact["event_id"])
        if event_id not in events:
            raise RuntimeError(f"unknown N68 replay event artifact {event_id}")
        event = events[event_id]
        if artifact.get("runtime_boundary", {}).get("runtime_future_gt_used") is not False:
            raise RuntimeError(f"runtime future GT boundary failed in {event_id}")
        event_summaries[event_id] = {}
        for variant in VARIANTS:
            variant_payload = artifact["variants"].get(variant)
            if not isinstance(variant_payload, dict) or int(variant_payload.get("frame_count", -1)) != 100:
                raise RuntimeError(f"N68 replay frame denominator failed {event_id}/{variant}")
            frames = variant_payload["frames"]
            for frame in frames:
                # Baseline is paired at the same frame and upstream variant.
                # Reusing the event's first frame would manufacture assignment
                # transitions and corrupt correct/incorrect/neutral counts.
                frame_baseline = frame["methods"]["CURRENT_CCAM_BASELINE"]
                for method in MODES:
                    outcome = frame_outcome(frame, method, frame_baseline, event)
                    outcome["event_id"] = event_id
                    outcome["sequence"] = event["sequence"]
                    outcome["action_type"] = event["event"].get("action_type")
                    outcome["variant"] = variant
                    outcome["method"] = method
                    all_outcomes.append((event["sequence"], str(event["event"].get("action_type")), variant, outcome))
            for method in MODES:
                event_summaries[event_id].setdefault(method, {})[variant] = summarize_outcomes(all_outcomes[-(100 * len(MODES)):], method) if False else None
    by_method = {method: summarize_outcomes(all_outcomes, method) for method in MODES}
    by_action: dict[str, dict[str, Any]] = {}
    action_values = sorted({action for _, action, _, _ in all_outcomes})
    for action in action_values:
        action_outcomes = [item for item in all_outcomes if item[1] == action]
        by_action[action] = {method: summarize_outcomes(action_outcomes, method) for method in MODES}
    by_variant: dict[str, dict[str, Any]] = {}
    for variant in VARIANTS:
        variant_outcomes = [item for item in all_outcomes if item[2] == variant]
        by_variant[variant] = {method: summarize_outcomes(variant_outcomes, method) for method in MODES}
    by_sequence: dict[str, dict[str, Any]] = {}
    for sequence in sorted({item[0] for item in all_outcomes}):
        seq_outcomes = [item for item in all_outcomes if item[0] == sequence]
        by_sequence[sequence] = {method: summarize_outcomes(seq_outcomes, method) for method in MODES}
    baseline = by_method["CURRENT_CCAM_BASELINE"]
    gate_by_mode: dict[str, Any] = {}
    for method in MODES:
        if method == "CURRENT_CCAM_BASELINE":
            gate_by_mode[method] = {"future_effect": False, "reason": "reference_baseline"}
            continue
        summary = by_method[method]
        lower_bounds = {h: summary["horizons"][str(h)]["sequence_cluster_bootstrap"]["ci95"][0] for h in HORIZONS}
        gate_by_mode[method] = {"future_effect": bool(all(value is not None and value > 0.0 for value in lower_bounds.values()) and summary["correct_changes"] > summary["incorrect_changes"] and summary["untouched_regression_frame_rate"] == 0.0), "strict_lower_ci_by_horizon": lower_bounds, "correct_changes_gt_incorrect_changes": summary["correct_changes"] > summary["incorrect_changes"], "untouched_regression_safe": summary["untouched_regression_frame_rate"] == 0.0, "real_human_tape": False, "production_authorized": False}
    result = {"schema": "N68_STAGE_02_LOCAL_ASSOCIATION_PAIRED_RESULTS_V1", "status": "N68_SIMULATED_FUTURE_EFFECT_EVALUATED", "created_at_utc": now(), "protocol": str(PROTOCOL), "protocol_sha256": sha256_file(PROTOCOL), "runtime_status": str(REPLAY_STATUS), "event_count": EVENT_COUNT, "variant_count": len(VARIANTS), "frame_count": len(all_outcomes), "interaction_source": "simulated_from_gt", "not_real_human_evidence": True, "real_human_tape": False, "real_sam3_full_loop": False, "runtime_future_gt_used": False, "gt_loaded_only_posthoc": True, "evaluation_boundary": {"candidate_generation_changed": False, "hungarian_solver_changed": False, "same_candidate_stream": True, "same_public_id_axis": True, "target_column_only_residual": True, "future_iou_available": False, "future_idsw_available": False, "id_switch_reported_as_proxy_only": False}, "methods": by_method, "by_action_type": by_action, "by_upstream_variant": by_variant, "by_sequence": by_sequence, "gate_by_mode": gate_by_mode, "strict_future_effect_gate": {"status": "PASS" if any(value.get("future_effect") for key, value in gate_by_mode.items() if key != "CURRENT_CCAM_BASELINE") else "FAIL_FUTURE_EFFECT", "baseline": baseline, "calibration_authorized": False, "selector_authorized": False, "decoder_lora_authorized": False, "production_authorized": False}, "failure_root_cause": "N68 evaluates an isolated target-conditioned local association sidecar against the frozen N54 stream. Any positive loss or score change is not identity efficacy; gate requires positive utility at all horizons, correct changes exceeding incorrect changes, and zero untouched regression.", "next_action": "If no sidecar meets the strict gate, preserve this branch as failed and do not enter TACT/calibration/selector/LoRA; diagnose mapping/scope and collect real human tape.", "outputs": {"event_artifacts": str(REPLAY_ARTIFACTS), "paired_results": str(SCORED_RESULTS)}}
    atomic_json(SCORED_RESULTS, result)
    atomic_json(SCORE_STATUS, {"schema": "N68_STAGE_02_LOCAL_ASSOCIATION_POSTHOC_STATUS_V1", "status": "PASS_POSTHOC_SCORED_STRICT_GATE_REPORTED", "created_at_utc": now(), "paired_results": str(SCORED_RESULTS), "event_count": EVENT_COUNT, "frame_outcome_count": len(all_outcomes), "runtime_future_gt_used": False, "gt_loaded_only_posthoc": True, "production_authorized": False})
    return result


def update_stage_statuses(result: dict[str, Any] | None = None) -> None:
    training = load_json(TRAINING_MANIFEST) if TRAINING_MANIFEST.is_file() else None
    replay = load_json(REPLAY_STATUS) if REPLAY_STATUS.is_file() else None
    scored = result if result is not None else (load_json(SCORED_RESULTS) if SCORED_RESULTS.is_file() else None)
    stage02 = {"schema": "N68_STAGE_02_STATUS_V1", "status": "PASS_ISOLATED_LOCAL_HEAD_TRAINED", "created_at_utc": now(), "protocol": str(PROTOCOL), "protocol_sha256": sha256_file(PROTOCOL), "dataset_manifest": str(DATASET_MANIFEST), "training_manifest": str(TRAINING_MANIFEST), "smoke": str(TRAIN / "n68_local_head_smoke.json"), "checkpoint": str(CHECKPOINT), "training": training, "gate_checks": {"dataset_materialized": DATASET.is_file(), "smoke_pass": (TRAIN / "n68_local_head_smoke.json").is_file(), "actual_training_completed": training is not None, "sequence_disjoint_split": True, "holdout_selected_after_training": bool(training and training.get("holdout_evaluated_once_after_selection")), "production_authorized": False}, "interaction_source": "simulated_from_gt", "not_real_human_evidence": True, "real_human_tape": False, "real_sam3_full_loop": False, "runtime_future_gt_used": False, "production_authorized": False}
    atomic_json(STAGE_02, stage02)
    if scored is not None:
        strict = scored.get("strict_future_effect_gate", {})
        stage03 = {"schema": "N68_STAGE_03_STATUS_V1", "status": "FAIL_NO_TRIMMING_STRICT_FUTURE_EFFECT" if strict.get("status") == "FAIL_FUTURE_EFFECT" else "PASS_NO_TRIMMING_STRICT_GATE", "created_at_utc": now(), "paired_results": str(SCORED_RESULTS), "gate_by_mode": scored.get("gate_by_mode"), "failure_root_cause": scored.get("failure_root_cause"), "next_action": "Do not enter TACT or downstream learning unless a no-trimming branch passes all frozen gates; current simulated event source is not real human evidence.", "production_authorized": False, "runtime_future_gt_used": False, "interaction_source": "simulated_from_gt", "not_real_human_evidence": True}
        atomic_json(STAGE_03, stage03)
        atomic_json(STAGE_04, {"schema": "N68_STAGE_04_STATUS_V1", "status": "NOT_RUN", "reason": "No N68 no-trimming sidecar has passed the strict future-effect gate; TACT causal trimming is not authorized.", "created_at_utc": now(), "parent": str(STAGE_03), "production_authorized": False, "runtime_future_gt_used": False})
        atomic_json(STAGE_05, {"schema": "N68_STAGE_05_STATUS_V1", "status": "BLOCKED_PENDING_REAL_HUMAN_TAPE", "reason": "N37/N54/N68 evidence is simulated_from_gt; no real human event tape or real SAM3 full-loop evidence is present.", "created_at_utc": now(), "real_human_tape": False, "real_sam3_full_loop": False, "production_authorized": False, "runtime_future_gt_used": False})
        atomic_json(STAGE_06, {"schema": "N68_STAGE_06_STATUS_V1", "status": "NOT_RUN", "reason": "Calibration, selector, and decoder LoRA remain unauthorized because simulated future-effect and real-evidence gates are not complete.", "created_at_utc": now(), "production_authorized": False, "runtime_future_gt_used": False})


def record_failure(stage: str, exc: BaseException) -> None:
    ATTEMPTS.mkdir(parents=True, exist_ok=True)
    existing = sorted(ATTEMPTS.glob(f"{stage}_failure_attempt*.json"))
    path = ATTEMPTS / f"{stage}_failure_attempt{len(existing) + 1}.json"
    atomic_json(path, {"schema": "N68_STAGE_FAILURE_V1", "status": "FAIL_PRESERVED", "created_at_utc": now(), "stage": stage, "failure_root_cause": f"{type(exc).__name__}: {exc}", "protocol": str(PROTOCOL), "runtime_future_gt_used": False, "interaction_source": "simulated_from_gt", "not_real_human_evidence": True, "production_authorized": False, "next_action": "Preserve this failure, repair only the first actionable root cause, and rerun the same frozen input."})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", required=True, choices=("protocol", "materialize", "smoke", "train", "replay", "score"))
    parser.add_argument("--device", default="cpu", choices=("cpu", "cuda"))
    args = parser.parse_args()
    if args.mode == "protocol":
        ensure_protocol()
        print(json.dumps({"status": "PASS_PROTOCOL_FROZEN", "path": str(PROTOCOL), "sha256": sha256_file(PROTOCOL)}, sort_keys=True))
    elif args.mode == "materialize":
        print(json.dumps(materialise(), sort_keys=True))
    elif args.mode == "smoke":
        print(json.dumps(smoke(args.device), sort_keys=True))
    elif args.mode == "train":
        payload = train(args.device)
        print(json.dumps({"status": payload["status"], "checkpoint": payload["checkpoint"], "best_epoch": payload["best_epoch"], "holdout": payload["holdout"]}, sort_keys=True))
    elif args.mode == "replay":
        print(json.dumps(replay(args.device), sort_keys=True))
    elif args.mode == "score":
        result = score_replay()
        update_stage_statuses(result)
        print(json.dumps({"status": result["strict_future_effect_gate"]["status"], "paired_results": str(SCORED_RESULTS), "gate_by_mode": result["gate_by_mode"]}, sort_keys=True))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        record_failure("stage_02_local_association", exc)
        raise
