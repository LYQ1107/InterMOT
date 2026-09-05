#!/usr/bin/env python3
"""Run one isolated N72R9 temporal replay event.

Runtime and posthoc scoring are intentionally separated.  The runtime phase
never opens GT and writes sealed per-variant JSONL first.  Only after those
files and hashes exist does the posthoc phase load train GT to score the
already completed runtime outputs.
"""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import tempfile
import traceback
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in __import__("sys").path:
    __import__("sys").path.insert(0, str(ROOT))

from sam3_intermot.association.effect_assignment import solve_effect_assignment  # noqa: E402
from sam3_intermot.evaluation.interaction_effect_metrics import (  # noqa: E402
    AssignmentChangeType,
    metric_record,
    sequence_cluster_bootstrap,
)
from sam3_intermot.reacquisition.models.n72r9_source_temporal import (  # noqa: E402
    N72R9SourceAwareTemporalIdentityModel,
)
from sam3_intermot.reacquisition.target_candidate_pool import (  # noqa: E402
    build_candidate_pool,
    build_candidate_pool_with_requery,
    serializable_candidate,
)
from sam3_intermot.reacquisition.target_id_features import candidate_feature_vector  # noqa: E402


PROTOCOL_PATH = ROOT / "outputs/N72R9/protocol.json"
EVENT_POLICY = ROOT / "outputs/N72R5/mechanism_rounds/round_06_event_policy/real_event_manifest.json"
DATA_ROOT = Path("/data1/LWR/vranlee/DATASETS/JDE/dancetrack")
CHECKPOINT = ROOT / "outputs/N72R9/training/N72R9SourceAwareTemporalIdentityModel_v1.pt"
HORIZON = 100
HORIZONS = (20, 50, 100)
IOU_THRESHOLD = 0.50
VARIANTS = ("BASELINE_B0", "TEMPORAL_CURRENT", "TEMPORAL_REQUERY")
COMPARISONS = {
    "TEMPORAL_CURRENT_vs_BASELINE_B0": ("BASELINE_B0", "TEMPORAL_CURRENT"),
    "TEMPORAL_REQUERY_vs_BASELINE_B0": ("BASELINE_B0", "TEMPORAL_REQUERY"),
    "TEMPORAL_REQUERY_vs_TEMPORAL_CURRENT": ("TEMPORAL_CURRENT", "TEMPORAL_REQUERY"),
}
SOURCE_NAMES = (
    "MAIN_B0_CANDIDATE",
    "TARGET_SESSION_CURRENT_RAW",
    "TARGET_SESSION_REQUERY",
    "UNKNOWN",
)
MEMORY_SLOTS = 4
UNCERTAINTY_MARGIN = 0.25
ADMISSION_SCORE = 0.50
ADMISSION_MARGIN = 0.20
INJECTION_SCALE = 1.0
BOOTSTRAP_SEED = 7290
BOOTSTRAP_REPETITIONS = 2000


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


def atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    atomic_write(path, json.dumps(dict(value), ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n")


def atomic_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    atomic_write(path, "".join(json.dumps(dict(row), ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n" for row in rows))


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


def _finite_box(value: Any, label: str) -> list[float]:
    box = np.asarray(value, dtype=np.float64).reshape(-1)
    if box.size != 4 or not np.all(np.isfinite(box)):
        raise ValueError(f"{label} box is not finite xyxy")
    return [float(item) for item in box]


def _box_iou(left: Any, right: Any) -> float:
    a = np.asarray(left, dtype=np.float64).reshape(-1)
    b = np.asarray(right, dtype=np.float64).reshape(-1)
    if a.size != 4 or b.size != 4 or not np.all(np.isfinite(a)) or not np.all(np.isfinite(b)):
        return 0.0
    if a[2] <= a[0] or a[3] <= a[1] or b[2] <= b[0] or b[3] <= b[1]:
        return 0.0
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    intersection = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - intersection
    return float(intersection / union) if union > 0.0 else 0.0


def _unit(value: Any, label: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32).reshape(-1)
    if array.size != 512 or not np.all(np.isfinite(array)):
        raise ValueError(f"{label} is not finite 512-D")
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


def _memory_array(values: Sequence[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    array = np.zeros((MEMORY_SLOTS, 512), dtype=np.float32)
    mask = np.zeros(MEMORY_SLOTS, dtype=np.bool_)
    for index, value in enumerate(list(values)[-MEMORY_SLOTS:]):
        array[index] = _unit(value, f"memory[{index}]")
        mask[index] = True
    return array, mask


def _mean_feature(values: Sequence[np.ndarray], fallback: np.ndarray) -> np.ndarray:
    if not values:
        return _unit(fallback, "neighbor fallback")
    result = np.mean(np.stack([_unit(value, "neighbor feature") for value in values], axis=0), axis=0)
    return (result / max(float(np.linalg.norm(result)), 1.0e-6)).astype(np.float32)


def _explicit_pairs(row: Mapping[str, Any], label: str) -> list[tuple[int, int]]:
    states = [int(value) for value in row.get("association_state_axis", [])]
    if not states or len(states) != len(set(states)):
        raise ValueError(f"{label} has invalid association state axis")
    by_state: dict[int, int] = {}
    for item in row.get("identity_rows", []):
        state_id, public_id = item.get("association_state_id"), item.get("public_id")
        if state_id is None or public_id is None:
            continue
        state_id, public_id = int(state_id), int(public_id)
        if state_id in by_state and by_state[state_id] != public_id:
            raise ValueError(f"{label} authority conflict at state {state_id}")
        by_state[state_id] = public_id
    for item in row.get("candidate_rows", []):
        state_id, public_id = item.get("solver_association_state_id"), item.get("solver_public_id")
        if state_id is None or public_id is None:
            continue
        state_id, public_id = int(state_id), int(public_id)
        if state_id in by_state and by_state[state_id] != public_id:
            raise ValueError(f"{label} candidate authority conflict at state {state_id}")
        by_state[state_id] = public_id
    declared = [int(value) for value in row.get("public_id_axis", [])]
    if all(state_id in by_state for state_id in states):
        publics = [by_state[state_id] for state_id in states]
    elif len(declared) == len(states):
        publics = declared
    else:
        raise ValueError(f"{label} cannot map active state columns to public IDs")
    if len(publics) != len(set(publics)):
        raise ValueError(f"{label} mapped public axis has duplicates")
    return list(zip(states, publics))


def _matrix_row_vectors(
    row: Mapping[str, Any],
    target_pairs: Sequence[tuple[int, int]],
    label: str,
) -> dict[str, np.ndarray]:
    candidates = list(row.get("candidate_rows", []))
    matrix = np.asarray(row.get("base_score_matrix", []), dtype=np.float64)
    source_pairs = _explicit_pairs(row, label)
    if matrix.shape != (len(candidates), len(source_pairs)) or not np.all(np.isfinite(matrix)):
        raise ValueError(f"{label} score matrix/authority shape mismatch: {matrix.shape}")
    source_by_public = {public_id: index for index, (_, public_id) in enumerate(source_pairs)}
    result: dict[str, np.ndarray] = {}
    for index, candidate in enumerate(candidates):
        uid = str(candidate["candidate_uid"])
        vector = np.zeros(len(target_pairs), dtype=np.float64)
        for target_index, (_, public_id) in enumerate(target_pairs):
            source_index = source_by_public.get(public_id)
            if source_index is not None:
                vector[target_index] = float(matrix[index, source_index])
        result[uid] = vector
    return result


def _load_rows(event: Mapping[str, Any]) -> dict[str, Any]:
    event_manifest = read_json(_path(str(event["source_event_manifest"])))
    if event_manifest.get("status") != "PASS_N72R7_CLOSED_LOOP_EVENT_REPLAY":
        raise RuntimeError(f"source event manifest is not PASS: {event['event_id']}")
    target_public = event_manifest.get("target_public_id")
    if target_public is None:
        raise RuntimeError(f"source event lacks target public authority: {event['event_id']}")
    target_public = int(target_public)
    paths = {key: _path(str(event[key])) for key in ("c0_source", "c1_source", "target_stream_source", "requery_source")}
    hash_keys = {"c0_source": "c0_source_sha256", "c1_source": "c1_source_sha256", "target_stream_source": "target_stream_source_sha256", "requery_source": "requery_source_sha256"}
    for key, path in paths.items():
        if not path.is_file() or sha256_file(path) != str(event[hash_keys[key]]):
            raise RuntimeError(f"frozen input hash mismatch: {event['event_id']}:{key}")
    rows = {key: read_jsonl(path) for key, path in paths.items()}
    event_frame = int(event["event_frame"])
    expected = list(range(event_frame, event_frame + HORIZON + 1))
    for key, values in rows.items():
        if [int(value["frame"]) for value in values] != expected or len(values) != HORIZON + 1:
            raise RuntimeError(f"{event['event_id']} {key} frame axis is not event..event+100")
        for value in values:
            if value.get("runtime_future_gt_used") is not False or value.get("runtime_gt_read") is True:
                raise RuntimeError(f"{event['event_id']} {key} runtime GT flag violation")
    anchor_path = paths["target_stream_source"].parent / "human_anchor.json"
    anchor_payload = read_json(anchor_path)
    anchor = _unit(anchor_payload.get("feature"), f"{event['event_id']} human anchor")
    anchor_box = _finite_box(anchor_payload.get("box_xyxy"), f"{event['event_id']} human anchor")
    by_frame = {key: {int(value["frame"]): value for value in values} for key, values in rows.items()}
    return {
        "event_id": str(event["event_id"]),
        "sequence": str(event["sequence"]),
        "event_frame": event_frame,
        "future_window": [event_frame + 1, event_frame + HORIZON],
        "target_public_id": target_public,
        "anchor": anchor,
        "anchor_box": anchor_box,
        "rows": by_frame,
        "source_paths": {key: str(value) for key, value in paths.items()},
        "source_hashes": {key: str(event[hash_keys[key]]) for key in paths},
    }


def _dimensions(sequence: str, frame: int) -> tuple[int, int]:
    path = DATA_ROOT / "train" / sequence / "img1" / f"{int(frame) + 1:08d}.jpg"
    if not path.is_file():
        raise FileNotFoundError(path)
    with Image.open(path) as image:
        return int(image.width), int(image.height)


def _base_vectors(
    inputs: Mapping[str, Any],
    frame: int,
    pool: Sequence[Mapping[str, Any]],
) -> tuple[list[tuple[int, int]], dict[str, np.ndarray]]:
    c0 = inputs["rows"]["c0_source"][frame]
    c1 = inputs["rows"]["c1_source"][frame]
    c0_pairs = _explicit_pairs(c0, f"{inputs['event_id']}:C0:{frame}")
    c1_pairs = _explicit_pairs(c1, f"{inputs['event_id']}:C1:{frame}")
    replay_pairs = list(c0_pairs)
    if inputs["target_public_id"] not in {public_id for _, public_id in replay_pairs}:
        target_pair = next((pair for pair in c1_pairs if pair[1] == inputs["target_public_id"]), None)
        if target_pair is None:
            raise RuntimeError(f"target authority is missing at {inputs['event_id']}:{frame}")
        if target_pair[0] in {state_id for state_id, _ in replay_pairs}:
            raise RuntimeError(f"target state authority collides at {inputs['event_id']}:{frame}")
        replay_pairs.append(target_pair)
    c0_vectors = _matrix_row_vectors(c0, replay_pairs, f"{inputs['event_id']}:C0:{frame}")
    c1_vectors = _matrix_row_vectors(c1, replay_pairs, f"{inputs['event_id']}:C1:{frame}")
    result: dict[str, np.ndarray] = {}
    for candidate in pool:
        uid = str(candidate["candidate_uid"])
        if uid in c0_vectors:
            result[uid] = c0_vectors[uid]
        elif uid in c1_vectors:
            result[uid] = c1_vectors[uid]
        else:
            result[uid] = np.zeros(len(replay_pairs), dtype=np.float64)
    return replay_pairs, result


def _causal_score(candidate: Mapping[str, Any], base_score: float, trusted: Sequence[np.ndarray], predicted_box: Sequence[float]) -> float:
    similarity = 0.0
    if candidate.get("feature") is not None and trusted:
        similarity = float(np.dot(_unit(candidate["feature"], "causal feature"), _unit(trusted[-1], "causal memory")))
    presence = float(np.clip(float(candidate.get("presence_score", 0.0)), 0.0, 1.0))
    return float(np.tanh(base_score) + 0.30 * similarity + 0.15 * _box_iou(candidate["box_xyxy"], predicted_box) + 0.05 * presence)


def _model_inputs(
    inputs: Mapping[str, Any],
    frame: int,
    pool: Sequence[Mapping[str, Any]],
    base_vectors: Mapping[str, np.ndarray],
    replay_pairs: Sequence[tuple[int, int]],
    trusted: Sequence[np.ndarray],
    distractors: Sequence[np.ndarray],
    predicted_box: Sequence[float],
    previous_raw: int | None,
    previous_scope: str | None,
    previous_score: float,
    previous_uncertainty: float,
    trusted_age: int,
    dimensions: tuple[int, int],
) -> dict[str, Any]:
    width, height = dimensions
    target_col = [public_id for _, public_id in replay_pairs].index(inputs["target_public_id"])
    base_scores = {str(candidate["candidate_uid"]): float(base_vectors[str(candidate["candidate_uid"])][target_col]) for candidate in pool}
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
                base_target_score=base_scores[str(candidate["candidate_uid"])],
            )
            for candidate in pool
        ],
        axis=0,
    ).astype(np.float32)
    causal_scores = np.asarray(
        [_causal_score(candidate, base_scores[str(candidate["candidate_uid"])], trusted, predicted_box) for candidate in pool],
        dtype=np.float64,
    )
    order = sorted(range(len(pool)), key=lambda index: (-causal_scores[index], str(pool[index]["candidate_uid"])))
    top = float(causal_scores[order[0]]) if order else 0.0
    second = float(causal_scores[order[1]]) if len(order) > 1 else 0.0
    margin = float(top - second)
    neighbor = _mean_feature(
        [candidate_values[index, :512] for index in order[1:] if np.linalg.norm(candidate_values[index, :512]) > 1.0e-6],
        inputs["anchor"],
    )
    trusted_array, trusted_mask = _memory_array(trusted)
    distractor_array, distractor_mask = _memory_array(distractors)
    temporal = np.asarray(
        [
            float(frame - inputs["event_frame"]) / float(HORIZON),
            float(np.tanh(top)),
            float(np.tanh(second)),
            float(np.tanh(margin)),
            float(np.clip(previous_score, -1.0, 1.0)),
            float(np.clip(previous_uncertainty, 0.0, 1.0)),
            float(min(trusted_age, HORIZON)) / float(HORIZON),
            float(any(str(candidate["candidate_source"]) == "TARGET_SESSION_CURRENT_RAW" for candidate in pool)),
        ],
        dtype=np.float32,
    )
    source_values = np.stack([_source_vector(str(candidate["candidate_source"])) for candidate in pool], axis=0).astype(np.float32)
    return {
        "candidate_values": candidate_values,
        "source_values": source_values,
        "trusted_array": trusted_array,
        "trusted_mask": trusted_mask,
        "distractor_array": distractor_array,
        "distractor_mask": distractor_mask,
        "neighbor": neighbor,
        "temporal": temporal,
        "base_scores": base_scores,
        "causal_scores": causal_scores,
        "causal_order": order,
        "causal_margin": margin,
    }


def _select_model(model: torch.nn.Module, values: Mapping[str, Any], device: torch.device, pool: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    candidates = torch.as_tensor(values["candidate_values"][None], dtype=torch.float32, device=device)
    mask = torch.ones((1, len(pool)), dtype=torch.bool, device=device)
    sources = torch.as_tensor(values["source_values"][None], dtype=torch.float32, device=device)
    trusted = torch.as_tensor(values["trusted_array"][None], dtype=torch.float32, device=device)
    trusted_mask = torch.as_tensor(values["trusted_mask"][None], dtype=torch.bool, device=device)
    distractors = torch.as_tensor(values["distractor_array"][None], dtype=torch.float32, device=device)
    distractor_mask = torch.as_tensor(values["distractor_mask"][None], dtype=torch.bool, device=device)
    neighbor = torch.as_tensor(values["neighbor"][None], dtype=torch.float32, device=device)
    temporal = torch.as_tensor(values["temporal"][None], dtype=torch.float32, device=device)
    with torch.no_grad():
        logits = model(candidates, mask, sources, trusted, trusted_mask, distractors, distractor_mask, neighbor, temporal)[0].detach().float().cpu().numpy()
    none_logit = float(logits[len(pool)])
    scores = [float(value - none_logit) for value in logits[: len(pool)]]
    order = sorted(range(len(pool)), key=lambda index: (-scores[index], str(pool[index]["candidate_uid"])))
    best_index = order[0] if order else None
    second_index = order[1] if len(order) > 1 else None
    best = None if best_index is None else float(scores[best_index])
    second = None if second_index is None else float(scores[second_index])
    margin = None if best is None else float(best - max(0.0, second if second is not None else 0.0))
    selected = None if best_index is None or best is None or best < ADMISSION_SCORE or (margin is not None and margin < ADMISSION_MARGIN) else str(pool[best_index]["candidate_uid"])
    ranked = [
        {
            "candidate_uid": str(pool[index]["candidate_uid"]),
            "candidate_source": str(pool[index]["candidate_source"]),
            "model_score": float(scores[index]),
            "model_logit": float(logits[index]),
            "none_logit": none_logit,
            "runtime_future_gt_used": False,
        }
        for index in order
    ]
    return {
        "selected_candidate_uid": selected,
        "selected_score": best,
        "second_candidate_uid": None if second_index is None else str(pool[second_index]["candidate_uid"]),
        "second_score": second,
        "best_minus_second_margin": margin,
        "none_logit": none_logit,
        "ranked_candidates": ranked,
        "candidate_count": len(pool),
        "score_changed": bool(any(abs(value) > 1.0e-9 for value in scores)),
        "runtime_future_gt_used": False,
        "public_id_inference": False,
    }


def _find_target_uid(solver: Mapping[str, Any], target_public: int) -> str | None:
    matches = [row for row in solver.get("assignment_rows", []) if row.get("public_id") is not None and int(row["public_id"]) == int(target_public)]
    if len(matches) > 1:
        raise RuntimeError(f"duplicate target public assignment: {target_public}")
    return None if not matches else str(matches[0]["candidate_uid"])


def _solver_rows(pool: Sequence[Mapping[str, Any]], solver: Mapping[str, Any]) -> list[dict[str, Any]]:
    decisions = {str(row["candidate_uid"]): row for row in solver["assignment_rows"]}
    result: list[dict[str, Any]] = []
    for candidate in pool:
        uid = str(candidate["candidate_uid"])
        decision = decisions[uid]
        item = serializable_candidate(candidate, include_feature=False)
        feature = candidate.get("feature")
        norm = None if feature is None else float(np.linalg.norm(np.asarray(feature, dtype=np.float32).reshape(-1)))
        public_id = decision.get("public_id")
        item.update(
            {
                "feature_finite": bool(feature is not None and norm is not None and np.isfinite(norm) and norm > 1.0e-6),
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
                "posthoc_gt_used": False,
            }
        )
        result.append(item)
    return result


def _state_objects(pairs: Sequence[tuple[int, int]]) -> list[Any]:
    return [type("N72R9ReplayState", (), {"association_state_id": int(state_id), "public_id": int(public_id)})() for state_id, public_id in pairs]


def _validate_runtime_rows(rows: Sequence[Mapping[str, Any]], inputs: Mapping[str, Any], variant: str) -> None:
    if len(rows) != HORIZON + 1:
        raise RuntimeError(f"{variant} runtime row count is not 101")
    expected = list(range(int(inputs["event_frame"]), int(inputs["event_frame"]) + HORIZON + 1))
    if [int(row.get("frame", -1)) for row in rows] != expected:
        raise RuntimeError(f"{variant} runtime frame axis mismatch")
    for index, row in enumerate(rows):
        for flag in ("runtime_future_gt_used", "runtime_gt_read", "posthoc_gt_used", "public_id_inference"):
            if row.get(flag) is not False:
                raise RuntimeError(f"{variant}:{row.get('frame')} flag {flag} is not false")
        if int(row.get("event_frame", -1)) != int(inputs["event_frame"]) or int(row.get("target_public_id", -1)) != int(inputs["target_public_id"]):
            raise RuntimeError(f"{variant}:{row.get('frame')} authority mismatch")
        if index == 0:
            if row.get("candidate_rows") != [] or row.get("candidate_count") != 0 or row.get("memory_read") is not False or row.get("event_frame_memory_read") is not False:
                raise RuntimeError(f"{variant} event frame violates causal boundary")
            continue
        if row.get("record_kind") != "future_association_frame" or int(row.get("frame_horizon", -1)) != index:
            raise RuntimeError(f"{variant}:{row.get('frame')} future row contract failed")
        if int(row.get("first_memory_visible_frame", -1)) != int(inputs["event_frame"]) + 1:
            raise RuntimeError(f"{variant}:{row.get('frame')} memory visibility failed")
        pool = row.get("candidate_pool")
        if not isinstance(pool, Mapping) or pool.get("runtime_future_gt_used") is not False or pool.get("public_id_inference") is not False:
            raise RuntimeError(f"{variant}:{row.get('frame')} candidate pool audit failed")
        pool_rows = list(pool.get("candidate_rows", []))
        output_rows = list(row.get("candidate_rows", []))
        if len(pool_rows) != len(output_rows) or int(row.get("candidate_count", -1)) != len(output_rows):
            raise RuntimeError(f"{variant}:{row.get('frame')} candidate count mismatch")
        pool_uids = [str(item.get("candidate_uid")) for item in pool_rows]
        output_uids = [str(item.get("candidate_uid")) for item in output_rows]
        if pool_uids != output_uids or len(pool_uids) != len(set(pool_uids)):
            raise RuntimeError(f"{variant}:{row.get('frame')} candidate UID order/collision")
        if any(item.get("public_id") is not None for item in pool_rows):
            raise RuntimeError(f"{variant}:{row.get('frame')} source pool carries public authority")
        score_audit = row.get("score_audit")
        if not isinstance(score_audit, Mapping):
            raise RuntimeError(f"{variant}:{row.get('frame')} score audit missing")
        matrix = np.asarray(score_audit.get("fused_score_matrix", []), dtype=np.float64)
        state_axis = list(score_audit.get("association_state_axis", []))
        public_axis = list(score_audit.get("public_id_axis", []))
        if matrix.ndim != 2 or matrix.shape != (len(output_rows), len(state_axis)) or len(state_axis) != len(public_axis) or not np.all(np.isfinite(matrix)):
            raise RuntimeError(f"{variant}:{row.get('frame')} fused matrix audit failed")
        assignment = row.get("assignment")
        if not isinstance(assignment, Mapping) or not isinstance(assignment.get("solver"), Mapping) or assignment["solver"].get("runtime_future_gt_used") is not False:
            raise RuntimeError(f"{variant}:{row.get('frame')} solver audit failed")


def _run_variant(inputs: Mapping[str, Any], variant: str, model: torch.nn.Module | None, device: torch.device) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    event_frame = int(inputs["event_frame"])
    target_public = int(inputs["target_public_id"])
    rows: list[dict[str, Any]] = [
        {
            "schema_version": "N72R9_TEMPORAL_RUNTIME_FRAME_V1",
            "record_kind": "event_frame_correction",
            "variant": variant,
            "event_id": str(inputs["event_id"]),
            "sequence": str(inputs["sequence"]),
            "event_frame": event_frame,
            "frame": event_frame,
            "frame_horizon": 0,
            "target_public_id": target_public,
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
            "raw_binding_switch": None,
            "trusted_memory_update": "HUMAN_ANCHOR_INITIALIZED",
            "distractor_memory_update_count": 0,
            "requery": {"triggered": False, "applied": False, "runtime_future_gt_used": False},
            "runtime_future_gt_used": False,
            "runtime_gt_read": False,
            "posthoc_gt_used": False,
            "public_id_inference": False,
            "public_id_immutable": True,
        }
    ]
    dimensions = _dimensions(str(inputs["sequence"]), event_frame)
    target_event_candidates = list(inputs["rows"]["target_stream_source"][event_frame].get("candidate_rows", []))
    previous_raw = None if not target_event_candidates else target_event_candidates[0].get("official_raw_sam_id")
    previous_raw = None if previous_raw is None else int(previous_raw)
    previous_scope = None if not target_event_candidates else str(target_event_candidates[0].get("native_tid_scope"))
    trusted: list[np.ndarray] = [_unit(inputs["anchor"], "anchor")]
    distractors: list[np.ndarray] = []
    predicted_box = list(inputs["anchor_box"])
    velocity = np.zeros(2, dtype=np.float64)
    previous_score = 0.0
    previous_uncertainty = 1.0
    trusted_age = 0
    requery_triggers = 0
    requery_applied = 0
    model_score_changed = 0
    assignment_count = 0
    for frame in range(event_frame + 1, event_frame + HORIZON + 1):
        c0 = inputs["rows"]["c0_source"][frame]
        c1 = inputs["rows"]["c1_source"][frame]
        target_row = inputs["rows"]["target_stream_source"][frame]
        main = list(c0.get("candidate_rows", []))
        target = list(target_row.get("candidate_rows", []))
        if variant == "BASELINE_B0":
            pool, pool_audit = build_candidate_pool(main, (), sequence=str(inputs["sequence"]), frame=frame, include_target_session=False)
        else:
            pool, pool_audit = build_candidate_pool(main, target, sequence=str(inputs["sequence"]), frame=frame, include_target_session=True)
        replay_pairs, base_vectors = _base_vectors(inputs, frame, pool)
        state_objects = _state_objects(replay_pairs)
        state_axis = [pair[0] for pair in replay_pairs]
        public_axis = [pair[1] for pair in replay_pairs]
        base_matrix = np.stack([base_vectors[str(candidate["candidate_uid"])] for candidate in pool], axis=0)
        target_col = public_axis.index(target_public)
        base_target_scores = base_matrix[:, target_col]
        base_order = sorted(range(len(pool)), key=lambda index: (-float(base_target_scores[index]), str(pool[index]["candidate_uid"])))
        base_top = float(base_target_scores[base_order[0]]) if base_order else 0.0
        base_second = float(base_target_scores[base_order[1]]) if len(base_order) > 1 else 0.0
        base_margin = float(base_top - base_second)
        base_solver = solve_effect_assignment(
            candidate_rows=pool,
            persistent_states=state_objects,
            fused_state_candidate_scores=base_matrix.T,
            source_run_id=f"n72r9:{variant}:base:{inputs['event_id']}:{frame}",
            session_id=f"n72r9:{variant}:{inputs['event_id']}",
            none_score=0.0,
        )
        base_assigned_uid = _find_target_uid(base_solver, target_public)
        uncertain = bool(base_assigned_uid is None or base_margin < UNCERTAINTY_MARGIN)
        triggered = False
        applied = False
        requery_rows: list[dict[str, Any]] = []
        if variant == "TEMPORAL_REQUERY" and uncertain:
            triggered = True
            requery_triggers += 1
            requery_rows = list(inputs["rows"]["requery_source"][frame].get("candidate_rows", []))
            if requery_rows:
                pool, pool_audit = build_candidate_pool_with_requery(main, target, requery_rows, sequence=str(inputs["sequence"]), frame=frame)
                replay_pairs, base_vectors = _base_vectors(inputs, frame, pool)
                state_objects = _state_objects(replay_pairs)
                state_axis = [pair[0] for pair in replay_pairs]
                public_axis = [pair[1] for pair in replay_pairs]
                base_matrix = np.stack([base_vectors[str(candidate["candidate_uid"])] for candidate in pool], axis=0)
                target_col = public_axis.index(target_public)
                applied = True
                requery_applied += 1
        if variant == "BASELINE_B0":
            selection = {
                "selected_candidate_uid": None,
                "selected_score": None,
                "score_changed": False,
                "runtime_future_gt_used": False,
                "public_id_inference": False,
            }
            model_scores = np.zeros(len(pool), dtype=np.float64)
        else:
            values = _model_inputs(
                inputs, frame, pool, base_vectors, replay_pairs, trusted, distractors,
                predicted_box, previous_raw, previous_scope, previous_score,
                previous_uncertainty, trusted_age, dimensions,
            )
            selection = _select_model(model, values, device, pool)
            model_scores = np.asarray([float(item.get("model_score", 0.0)) for item in selection.get("ranked_candidates", [])], dtype=np.float64)
            model_score_changed += int(bool(selection.get("score_changed")))
        selected_uid = selection.get("selected_candidate_uid")
        fused = base_matrix.copy()
        injected_delta = 0.0
        if selected_uid is not None and target_col < fused.shape[1]:
            selected_index = next(index for index, candidate in enumerate(pool) if str(candidate["candidate_uid"]) == str(selected_uid))
            selected_score = float(selection.get("selected_score") or 0.0)
            injected_delta = float(INJECTION_SCALE * max(selected_score, 0.0))
            fused[selected_index, target_col] += injected_delta
        solver = solve_effect_assignment(
            candidate_rows=pool,
            persistent_states=state_objects,
            fused_state_candidate_scores=fused.T,
            source_run_id=f"n72r9:{variant}:{inputs['event_id']}:{frame}",
            session_id=f"n72r9:{variant}:{inputs['event_id']}",
            none_score=0.0,
        )
        target_uid = _find_target_uid(solver, target_public)
        assignment_count += int(target_uid is not None)
        assigned_candidate = next((candidate for candidate in pool if str(candidate["candidate_uid"]) == str(target_uid)), None)
        old_raw, old_scope = previous_raw, previous_scope
        binding_switch = None
        if assigned_candidate is not None:
            new_raw = assigned_candidate.get("official_raw_sam_id")
            new_raw = None if new_raw is None else int(new_raw)
            new_scope = assigned_candidate.get("native_scope")
            new_scope = None if new_scope is None else str(new_scope)
            if old_raw != new_raw or old_scope != new_scope:
                binding_switch = {
                    "public_id": target_public,
                    "frame": frame,
                    "old_raw_sam_id": old_raw,
                    "new_raw_sam_id": new_raw,
                    "old_native_scope": old_scope,
                    "new_native_scope": new_scope,
                    "public_id_changed": False,
                    "runtime_future_gt_used": False,
                }
            if assigned_candidate.get("feature") is not None:
                new_box = [float(value) for value in assigned_candidate["box_xyxy"]]
                old_center = np.asarray([(predicted_box[0] + predicted_box[2]) / 2.0, (predicted_box[1] + predicted_box[3]) / 2.0])
                new_center = np.asarray([(new_box[0] + new_box[2]) / 2.0, (new_box[1] + new_box[3]) / 2.0])
                velocity = 0.5 * velocity + 0.5 * (new_center - old_center)
                predicted_box = new_box
                if selected_uid is not None and str(selected_uid) == str(target_uid) and float(selection.get("selected_score") or -1.0) >= ADMISSION_SCORE and float(selection.get("best_minus_second_margin") or -1.0) >= ADMISSION_MARGIN:
                    trusted.append(_unit(assigned_candidate["feature"], "trusted update"))
                    trusted_age = 0
                else:
                    trusted_age += 1
            else:
                trusted_age += 1
            previous_raw, previous_scope = new_raw, new_scope
        else:
            trusted_age += 1
        if selected_uid is not None and (assigned_candidate is None or str(selected_uid) != str(target_uid)):
            selected_candidate = next(candidate for candidate in pool if str(candidate["candidate_uid"]) == str(selected_uid))
            if selected_candidate.get("feature") is not None:
                distractors.append(_unit(selected_candidate["feature"], "distractor update"))
        trusted = trusted[-MEMORY_SLOTS:]
        distractors = distractors[-MEMORY_SLOTS:]
        previous_score = float(np.max(fused[:, target_col])) if len(fused) else 0.0
        previous_uncertainty = float(1.0 / (1.0 + max(base_margin, 0.0)))
        source_rows = [serializable_candidate(candidate, include_feature=False) for candidate in pool]
        for source_row in source_rows:
            source_row["public_id"] = None
            source_row["public_id_authority"] = None
        output_rows = _solver_rows(pool, solver)
        rows.append(
            {
                "schema_version": "N72R9_TEMPORAL_RUNTIME_FRAME_V1",
                "record_kind": "future_association_frame",
                "variant": variant,
                "event_id": str(inputs["event_id"]),
                "sequence": str(inputs["sequence"]),
                "event_frame": event_frame,
                "frame": frame,
                "frame_horizon": frame - event_frame,
                "target_public_id": target_public,
                "candidate_rows": output_rows,
                "candidate_count": len(pool),
                "candidate_pool": {
                    **pool_audit,
                    "candidate_rows": source_rows,
                },
                "assignment": {
                    "target_public_id": target_public,
                    "target_assigned_candidate_uid": target_uid,
                    "target_base_assigned_candidate_uid": base_assigned_uid,
                    "target_selected_candidate_uid": selected_uid,
                    "target_selector_and_solver_agree": bool(selected_uid is not None and str(selected_uid) == str(target_uid)),
                    "solver": solver,
                    "solver_public_id_immutable": True,
                    "runtime_future_gt_used": False,
                },
                "score_audit": {
                    "association_state_axis": state_axis,
                    "public_id_axis": public_axis,
                    "fused_score_matrix": fused.astype(float).tolist(),
                    "base_target_scores": base_matrix[:, target_col].astype(float).tolist(),
                    "fused_target_scores": fused[:, target_col].astype(float).tolist(),
                    "model_score_by_rank": model_scores.astype(float).tolist(),
                    "appearance_injection_delta": injected_delta,
                    "base_top1_score": base_top,
                    "base_top2_score": base_second,
                    "base_assignment_margin": base_margin,
                    "model_score_changed": bool(selection.get("score_changed", False)),
                    "runtime_future_gt_used": False,
                },
                "selection_audit": {
                    **selection,
                    "event_frame_memory_read": False,
                    "frame": frame,
                    "memory_read": variant != "BASELINE_B0",
                    "runtime_future_gt_used": False,
                },
                "memory_read": variant != "BASELINE_B0",
                "memory_write": variant != "BASELINE_B0" and target_uid is not None,
                "event_frame_memory_read": False,
                "first_memory_visible_frame": event_frame + 1,
                "raw_binding_switch": binding_switch,
                "trusted_memory_update": "CAUSAL_TARGET_ASSIGNMENT" if target_uid is not None and selected_uid == target_uid else "NO_TRUSTED_UPDATE",
                "distractor_memory_update_count": len(distractors),
                "requery": {
                    "triggered": triggered,
                    "applied": applied,
                    "source": inputs["source_paths"]["requery_source"] if applied else None,
                    "source_sha256": inputs["source_hashes"]["requery_source"] if applied else None,
                    "base_assignment_uid": base_assigned_uid,
                    "base_assignment_margin": base_margin,
                    "threshold": UNCERTAINTY_MARGIN,
                    "runtime_future_gt_used": False,
                },
                "runtime_future_gt_used": False,
                "runtime_gt_read": False,
                "posthoc_gt_used": False,
                "public_id_inference": False,
                "public_id_immutable": True,
            }
        )
    stats = {
        "requery_trigger_count": requery_triggers,
        "requery_applied_count": requery_applied,
        "model_score_changed_frame_count": model_score_changed,
        "target_assigned_frame_count": assignment_count,
        "runtime_future_gt_used": False,
    }
    _validate_runtime_rows(rows, inputs, variant)
    return rows, stats


def _load_gt(sequence: str) -> dict[int, dict[int, dict[str, Any]]]:
    path = DATA_ROOT / "train" / sequence / "gt" / "gt.txt"
    if not path.is_file():
        raise FileNotFoundError(path)
    result: dict[int, dict[int, dict[str, Any]]] = defaultdict(dict)
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            fields = [item.strip() for item in line.split(",")]
            if len(fields) < 6:
                raise ValueError(f"malformed GT row {path}:{line_number}")
            frame, identity = int(fields[0]) - 1, int(fields[1])
            x, y, width, height = [float(value) for value in fields[2:6]]
            box = [x, y, x + width, y + height]
            if not np.all(np.isfinite(np.asarray(box, dtype=np.float64))):
                raise ValueError(f"non-finite GT box {path}:{line_number}")
            result[frame][identity] = {"box": box}
    return result


def _candidate_best_iou(row: Mapping[str, Any], gt_box: Sequence[float]) -> tuple[float, dict[str, Any] | None]:
    values = [(_box_iou(item.get("box_xyxy"), gt_box), item) for item in row.get("candidate_rows", [])]
    return max(values, key=lambda value: (value[0], -int(value[1].get("candidate_index", 0))), default=(0.0, None))


def _public_box_for_gt(row: Mapping[str, Any], public_id: int, gt_box: Sequence[float]) -> tuple[float, dict[str, Any] | None]:
    values = [item for item in row.get("candidate_rows", []) if item.get("public_id") is not None and int(item["public_id"]) == int(public_id)]
    return max(((_box_iou(item.get("box_xyxy"), gt_box), item) for item in values), key=lambda value: (value[0], -int(value[1].get("candidate_index", 0))), default=(0.0, None))


def _target_binding(row: Mapping[str, Any], target_public: int) -> tuple[str | None, dict[str, Any] | None]:
    values = [item for item in row.get("candidate_rows", []) if item.get("public_id") is not None and int(item["public_id"]) == int(target_public)]
    if len(values) > 1:
        raise RuntimeError(f"duplicate posthoc target assignment: {row.get('event_id')}:{row.get('frame')}")
    return (None, None) if not values else (str(values[0]["candidate_uid"]), values[0])


def _public_map(row: Mapping[str, Any]) -> dict[str, int | None]:
    return {str(item["candidate_uid"]): item.get("public_id") for item in row.get("candidate_rows", [])}


def _metric_template() -> dict[str, Any]:
    return {
        "window_frame_count": 0, "evaluated_frames": 0, "target_gt_visible_frames": 0, "target_gt_absent_frames": 0,
        "baseline_iou_sum": 0.0, "treatment_iou_sum": 0.0, "delta_iou_sum": 0.0,
        "baseline_correct_frames": 0, "treatment_correct_frames": 0, "baseline_identity_error_frames": 0, "treatment_identity_error_frames": 0,
        "identity_error_reduction_sum": 0.0, "target_missing_frames": 0, "wrong_reassociation_frames": 0, "candidate_present_frames": 0,
        "assignment_change_count": 0, "target_assignment_change_count": 0, "global_common_assignment_change_count": 0,
        "true_correct_crossing_count": 0, "true_incorrect_crossing_count": 0, "directional_improvement_count": 0, "directional_regression_count": 0, "neutral_change_count": 0,
        "id_switch_count": 0, "recorrection_opportunity_count": 0, "raw_switch_count": 0,
        "posthoc_correct_switch_count": 0, "posthoc_wrong_switch_count": 0, "posthoc_unassessable_switch_count": 0,
        "protected_compared": 0, "protected_regression_count": 0, "protected_improvement_count": 0,
        "baseline_identity_error": None, "treatment_identity_error": None, "identity_error_reduction": None,
        "baseline_mean_iou": None, "treatment_mean_iou": None, "delta_iou": None,
        "missing_rate": None, "wrong_reassociation_rate": None, "candidate_recall": None,
        "assignment_change_rate": None, "target_assignment_change_rate": None, "id_switch_rate": None, "recorrection_rate": None, "protected_regression_rate": None,
    }


def _finalize_metric(metric: dict[str, Any]) -> dict[str, Any]:
    frames = int(metric["evaluated_frames"])
    if frames:
        for name, numerator in (
            ("baseline_mean_iou", "baseline_iou_sum"), ("treatment_mean_iou", "treatment_iou_sum"), ("delta_iou", "delta_iou_sum"),
            ("baseline_identity_error", "baseline_identity_error_frames"), ("treatment_identity_error", "treatment_identity_error_frames"),
            ("identity_error_reduction", "identity_error_reduction_sum"), ("missing_rate", "target_missing_frames"),
            ("wrong_reassociation_rate", "wrong_reassociation_frames"), ("candidate_recall", "candidate_present_frames"),
            ("assignment_change_rate", "assignment_change_count"), ("target_assignment_change_rate", "target_assignment_change_count"),
            ("id_switch_rate", "id_switch_count"), ("recorrection_rate", "recorrection_opportunity_count"),
        ):
            metric[name] = float(metric[numerator] / frames)
    if int(metric["protected_compared"]):
        metric["protected_regression_rate"] = float(metric["protected_regression_count"] / metric["protected_compared"])
    return metric


def _score_pair(
    event: Mapping[str, Any],
    baseline_name: str,
    treatment_name: str,
    horizon: int,
    gt_frames: Mapping[int, Mapping[int, Any]],
    protected: Mapping[int, int],
) -> dict[str, Any]:
    metric = _metric_template()
    metric["window_frame_count"] = int(horizon)
    baseline_rows = event["rows"][baseline_name]
    treatment_rows = event["rows"][treatment_name]
    event_frame = int(event["event_frame"])
    target_public = int(event["target_public_id"])
    target_gid = int(event["target_dataset_gt_id"])
    previous_uid: str | None = None
    previous_error = False
    details: list[dict[str, Any]] = []
    for offset in range(1, horizon + 1):
        frame = event_frame + offset
        baseline, treatment = baseline_rows[frame], treatment_rows[frame]
        gt_target = gt_frames.get(frame, {}).get(target_gid)
        baseline_uid, _ = _target_binding(baseline, target_public)
        treatment_uid, treatment_candidate = _target_binding(treatment, target_public)
        raw_switch = treatment.get("raw_binding_switch") is not None
        metric["raw_switch_count"] += int(raw_switch)
        if gt_target is None:
            metric["target_gt_absent_frames"] += 1
            metric["posthoc_unassessable_switch_count"] += int(raw_switch)
            continue
        metric["evaluated_frames"] += 1
        metric["target_gt_visible_frames"] += 1
        gt_box = gt_target["box"]
        baseline_iou, _ = _public_box_for_gt(baseline, target_public, gt_box)
        treatment_iou, _ = _public_box_for_gt(treatment, target_public, gt_box)
        baseline_correct, treatment_correct = baseline_iou >= IOU_THRESHOLD, treatment_iou >= IOU_THRESHOLD
        best_iou, _ = _candidate_best_iou(treatment, gt_box)
        treatment_missing = treatment_uid is None
        wrong_reassociation = bool(
            treatment_candidate is not None and any(
                int(other_id) != target_gid and _box_iou(treatment_candidate.get("box_xyxy"), other_item["box"]) >= IOU_THRESHOLD
                for other_id, other_item in gt_frames.get(frame, {}).items()
            )
        )
        changed = baseline_uid != treatment_uid
        baseline_map, treatment_map = _public_map(baseline), _public_map(treatment)
        common = set(baseline_map) & set(treatment_map)
        common_changed = sum(int(baseline_map[uid] != treatment_map[uid]) for uid in common)
        record = metric_record(
            baseline_iou=baseline_iou,
            treatment_iou=treatment_iou,
            baseline_correct=baseline_correct,
            treatment_correct=treatment_correct,
            assignment_changed=changed,
        )
        metric["baseline_iou_sum"] += baseline_iou
        metric["treatment_iou_sum"] += treatment_iou
        metric["delta_iou_sum"] += float(record["delta_iou"])
        metric["baseline_correct_frames"] += int(baseline_correct)
        metric["treatment_correct_frames"] += int(treatment_correct)
        metric["baseline_identity_error_frames"] += int(not baseline_correct)
        metric["treatment_identity_error_frames"] += int(not treatment_correct)
        metric["identity_error_reduction_sum"] += float(record["identity_error_reduction"])
        metric["target_missing_frames"] += int(treatment_missing)
        metric["wrong_reassociation_frames"] += int(wrong_reassociation)
        metric["candidate_present_frames"] += int(best_iou >= IOU_THRESHOLD)
        metric["assignment_change_count"] += int(changed)
        metric["target_assignment_change_count"] += int(changed)
        metric["global_common_assignment_change_count"] += int(common_changed > 0)
        change_type = record["assignment_change_type"]
        metric["true_correct_crossing_count"] += int(change_type == AssignmentChangeType.TRUE_CORRECT_CROSSING.value)
        metric["true_incorrect_crossing_count"] += int(change_type == AssignmentChangeType.TRUE_INCORRECT_CROSSING.value)
        metric["directional_improvement_count"] += int(change_type == AssignmentChangeType.DIRECTIONAL_IMPROVEMENT.value)
        metric["directional_regression_count"] += int(change_type == AssignmentChangeType.DIRECTIONAL_REGRESSION.value)
        metric["neutral_change_count"] += int(change_type == AssignmentChangeType.NEUTRAL_CHANGE.value)
        current_error = not treatment_correct
        recorrection = bool(current_error and not previous_error)
        id_switch = bool(previous_uid is not None and treatment_uid is not None and treatment_uid != previous_uid)
        metric["id_switch_count"] += int(id_switch)
        metric["recorrection_opportunity_count"] += int(recorrection)
        if raw_switch:
            metric["posthoc_correct_switch_count"] += int(treatment_correct)
            metric["posthoc_wrong_switch_count"] += int(not treatment_correct)
        for protected_gid, protected_pid in protected.items():
            other = gt_frames.get(frame, {}).get(int(protected_gid))
            if other is None:
                continue
            baseline_other_iou, _ = _public_box_for_gt(baseline, protected_pid, other["box"])
            treatment_other_iou, _ = _public_box_for_gt(treatment, protected_pid, other["box"])
            baseline_other_correct, treatment_other_correct = baseline_other_iou >= IOU_THRESHOLD, treatment_other_iou >= IOU_THRESHOLD
            metric["protected_compared"] += 1
            metric["protected_regression_count"] += int(baseline_other_correct and not treatment_other_correct)
            metric["protected_improvement_count"] += int(treatment_other_correct and not baseline_other_correct)
        details.append({
            "frame": frame,
            "baseline_target_candidate_uid": baseline_uid,
            "treatment_target_candidate_uid": treatment_uid,
            "target_assignment_changed": changed,
            "global_common_assignment_changed_count": common_changed,
            "baseline_target_iou": float(baseline_iou),
            "treatment_target_iou": float(treatment_iou),
            "baseline_correct": baseline_correct,
            "treatment_correct": treatment_correct,
            "target_missing": treatment_missing,
            "candidate_recall_present": bool(best_iou >= IOU_THRESHOLD),
            "wrong_reassociation": wrong_reassociation,
            "id_switch": id_switch,
            "recorrection_opportunity": recorrection,
            "raw_binding_switch": raw_switch,
            "assignment_change_type": change_type,
            "true_correct_crossing": bool(record["true_correct_crossing"]),
            "true_incorrect_crossing": bool(record["true_incorrect_crossing"]),
            "runtime_future_gt_used": False,
        })
        previous_uid = treatment_uid or previous_uid
        previous_error = current_error
    metric = _finalize_metric(metric)
    metric["frame_details"] = details
    return metric


def _protected_map(c0_event_row: Mapping[str, Any], gt_frames: Mapping[int, Mapping[int, Any]], event_frame: int, target_gid: int) -> dict[int, int]:
    result: dict[int, int] = {}
    for gt_id, item in gt_frames.get(event_frame, {}).items():
        if int(gt_id) == int(target_gid):
            continue
        best_iou, candidate = _candidate_best_iou(c0_event_row, item["box"])
        if candidate is not None and best_iou >= IOU_THRESHOLD and candidate.get("public_id") is not None:
            result[int(gt_id)] = int(candidate["public_id"])
    return result


def _merge_metric(destination: dict[str, Any], source: Mapping[str, Any]) -> None:
    for key, value in source.items():
        if key.endswith(("_sum", "_frames", "_count", "_compared")) or key in {"evaluated_frames", "window_frame_count"}:
            if isinstance(value, (int, float)):
                destination[key] = destination.get(key, 0) + value


def _aggregate(events: Sequence[Mapping[str, Any]], comparison: str, horizon: int, action: str | None = None) -> dict[str, Any]:
    selected = [item for item in events if action is None or item["action_type"] == action]
    metric = _metric_template()
    values_by_sequence: dict[str, list[float]] = defaultdict(list)
    for event in selected:
        source = event["comparisons"][comparison][str(horizon)]
        _merge_metric(metric, source)
        values_by_sequence[str(event["sequence"])].append(float(source.get("identity_error_reduction") or 0.0))
    metric = _finalize_metric(metric)
    metric["sequence_cluster_bootstrap_95ci"] = sequence_cluster_bootstrap(
        values_by_sequence,
        seed=BOOTSTRAP_SEED + list(COMPARISONS).index(comparison) * 100 + HORIZONS.index(horizon),
        repetitions=BOOTSTRAP_REPETITIONS,
    )
    metric["event_count"] = len(selected)
    metric["independent_sequence_count"] = len({str(item["sequence"]) for item in selected})
    metric["comparison"] = comparison
    metric["horizon"] = int(horizon)
    return metric


def _posthoc_score(inputs: Mapping[str, Any], runtime_rows: Mapping[str, list[dict[str, Any]]]) -> dict[str, Any]:
    policy = read_json(EVENT_POLICY)
    policy_events = {str(item["event_id"]): item for item in policy.get("events", [])}
    policy_event = policy_events.get(str(inputs["event_id"]))
    if policy_event is None:
        raise RuntimeError(f"event absent from frozen policy: {inputs['event_id']}")
    gt = _load_gt(str(inputs["sequence"]))
    target_gid = int(policy_event["dataset_gt_id"])
    c0_event = inputs["rows"]["c0_source"][int(inputs["event_frame"])]
    protected = _protected_map(c0_event, gt, int(inputs["event_frame"]), target_gid)
    by_frame = {variant: {int(row["frame"]): row for row in runtime_rows[variant]} for variant in VARIANTS}
    event_result: dict[str, Any] = {
        "event_id": str(inputs["event_id"]),
        "sequence": str(inputs["sequence"]),
        "action_type": str(policy_event["action_type"]),
        "event_frame": int(inputs["event_frame"]),
        "target_public_id": int(inputs["target_public_id"]),
        "target_dataset_gt_id": target_gid,
        "protected_public_by_gt_posthoc": protected,
        "comparisons": {},
        "interaction_source": "simulated_from_gt",
        "not_real_human_evidence": True,
        "runtime_future_gt_used": False,
        "gt_usage": "posthoc_only_after_runtime_artifact_seal",
    }
    for comparison, (baseline, treatment) in COMPARISONS.items():
        event_result["comparisons"][comparison] = {}
        for horizon in HORIZONS:
            event_result["comparisons"][comparison][str(horizon)] = _score_pair(
                {"event_id": inputs["event_id"], "sequence": inputs["sequence"], "event_frame": inputs["event_frame"], "target_public_id": inputs["target_public_id"], "target_dataset_gt_id": target_gid, "rows": by_frame},
                baseline, treatment, horizon, gt, protected,
            )
    aggregate = {comparison: {str(horizon): _aggregate([event_result], comparison, horizon) for horizon in HORIZONS} for comparison in COMPARISONS}
    return {
        "schema_version": "N72R9_TEMPORAL_POSTHOC_EVENT_V1",
        "status": "PASS_N72R9_POSTHOC_EVENT",
        "created_at_utc": now_utc(),
        "event": event_result,
        "aggregate_single_event": aggregate,
        "runtime_future_gt_used": False,
        "posthoc_gt_used": True,
        "interaction_source": "simulated_from_gt",
        "not_real_human_evidence": True,
    }


def _load_model(device: torch.device) -> torch.nn.Module:
    payload = torch.load(CHECKPOINT, map_location=device, weights_only=False)
    config = dict(payload.get("model_config", {}))
    required = {"candidate_feature_dim", "source_feature_dim", "temporal_feature_dim", "trusted_slots", "distractor_slots", "hidden_dim", "layers", "heads", "dropout"}
    if not required.issubset(config):
        raise RuntimeError("N72R9 checkpoint config is incomplete")
    model = N72R9SourceAwareTemporalIdentityModel(**{key: config[key] for key in required if key != "candidate_feature_dim" or True}).to(device)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--event-id", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    output_root = _path(args.output_root)
    event_id = str(args.event_id)
    started = now_utc()
    try:
        protocol = read_json(PROTOCOL_PATH)
        events = {str(item["event_id"]): item for item in protocol.get("source_event_selection", {}).get("events", [])}
        if event_id not in events:
            raise RuntimeError(f"event is not in N72R9 frozen development protocol: {event_id}")
        inputs = _load_rows(events[event_id])
        device = torch.device(str(args.device))
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("requested CUDA device is unavailable")
        model = _load_model(device)
        runtime_rows: dict[str, list[dict[str, Any]]] = {}
        runtime_manifests: dict[str, dict[str, Any]] = {}
        stats: dict[str, Any] = {}
        event_dir = output_root / event_id
        for variant in VARIANTS:
            rows, variant_stats = _run_variant(inputs, variant, None if variant == "BASELINE_B0" else model, device)
            frames_path = event_dir / variant / "runtime_frames.jsonl"
            atomic_jsonl(frames_path, rows)
            manifest = {
                "schema_version": "N72R9_TEMPORAL_RUNTIME_MANIFEST_V1",
                "status": "PASS_N72R9_RUNTIME_ARTIFACT_SEALED",
                "event_id": event_id,
                "sequence": inputs["sequence"],
                "variant": variant,
                "event_frame": inputs["event_frame"],
                "target_public_id": inputs["target_public_id"],
                "frame_count": len(rows),
                "frames": str(frames_path),
                "frames_sha256": sha256_file(frames_path),
                "input_source_hashes": inputs["source_hashes"],
                "stats": variant_stats,
                "runtime_future_gt_used": False,
                "runtime_gt_read": False,
                "posthoc_gt_used": False,
                "interaction_source": "simulated_from_gt",
                "not_real_human_evidence": True,
            }
            atomic_json(event_dir / variant / "runtime_manifest.json", manifest)
            runtime_rows[variant] = rows
            runtime_manifests[variant] = manifest
            stats[variant] = variant_stats
        sealed = {
            "schema_version": "N72R9_TEMPORAL_RUNTIME_EVENT_SEALED_V1",
            "status": "PASS_N72R9_ALL_VARIANT_RUNTIME_SEALED",
            "event_id": event_id,
            "variants": list(VARIANTS),
            "runtime_manifests": runtime_manifests,
            "runtime_future_gt_used": False,
            "posthoc_gt_used": False,
            "gt_loaded": False,
            "created_at_utc": now_utc(),
        }
        atomic_json(event_dir / "runtime_event_sealed.json", sealed)
        posthoc = _posthoc_score(inputs, runtime_rows)
        atomic_json(event_dir / "posthoc.json", posthoc)
        done = {
            "schema_version": "N72R9_TEMPORAL_EVENT_DONE_V1",
            "status": "PASS_N72R9_RUNTIME_AND_POSTHOC_EVENT",
            "event_id": event_id,
            "runtime_event_sealed": str(event_dir / "runtime_event_sealed.json"),
            "runtime_event_sealed_sha256": sha256_file(event_dir / "runtime_event_sealed.json"),
            "posthoc": str(event_dir / "posthoc.json"),
            "posthoc_sha256": sha256_file(event_dir / "posthoc.json"),
            "stats": stats,
            "runtime_future_gt_used": False,
            "posthoc_gt_used": True,
            "interaction_source": "simulated_from_gt",
            "not_real_human_evidence": True,
            "production_authorized": False,
            "started_at_utc": started,
            "finished_at_utc": now_utc(),
        }
        atomic_json(event_dir / "done.json", done)
        print(json.dumps({"status": done["status"], "event_id": event_id, "output": str(event_dir)}, sort_keys=True))
        return 0
    except Exception as exc:
        failure = output_root / "attempts" / f"{event_id}.failure.json"
        payload = {
            "schema_version": "N72R9_TEMPORAL_REPLAY_FAILURE_V1",
            "status": "FAIL_N72R9_TEMPORAL_REPLAY_EVENT",
            "event_id": event_id,
            "started_at_utc": started,
            "finished_at_utc": now_utc(),
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
            "runtime_future_gt_used": False,
            "historical_outputs_modified": False,
        }
        atomic_json(failure, payload)
        print(json.dumps({"status": payload["status"], "event_id": event_id, "failure_artifact": str(failure), "error": str(exc)}, sort_keys=True))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
