#!/usr/bin/env python3
"""Build the independent N72R9 source-aware temporal training corpus.

The input streams are frozen N72R7/N72R6 artifacts.  The feature tensors are
constructed without public IDs or GT; GT is read only after each causal
candidate/state tensor has been made, to attach an offline training label.
The causal state update is deliberately teacher-free: it uses only the
previous state and current candidate observations.
"""

from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import tempfile
import traceback
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in __import__("sys").path:
    __import__("sys").path.insert(0, str(ROOT))

from sam3_intermot.reacquisition.target_candidate_pool import build_candidate_pool  # noqa: E402
from sam3_intermot.reacquisition.target_id_features import (  # noqa: E402
    CANDIDATE_FEATURE_DIM,
    candidate_feature_vector,
)


PROTOCOL_PATH = ROOT / "outputs/N72R9/protocol.json"
OUTPUT_ROOT = ROOT / "outputs/N72R9/training"
STAGE_PATH = ROOT / "outputs/N72R9/stage_03_corpus_status.json"
DATA_ROOT = Path("/data1/LWR/vranlee/DATASETS/JDE/dancetrack")
HORIZON = 100
IOU_THRESHOLD = 0.50
SOURCE_NAMES = (
    "MAIN_B0_CANDIDATE",
    "TARGET_SESSION_CURRENT_RAW",
    "TARGET_SESSION_REQUERY",
    "UNKNOWN",
)
SOURCE_FEATURE_DIM = len(SOURCE_NAMES)
MEMORY_SLOTS = 4
TEMPORAL_FEATURE_DIM = 8


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
    atomic_write(
        path,
        "".join(json.dumps(dict(row), ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n" for row in rows),
    )


def atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".npz", dir=str(path.parent))
    os.close(fd)
    try:
        np.savez_compressed(temporary, **arrays)
        with open(temporary, "rb") as handle:
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


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object: {path}")
    return value


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def box_iou(left: Sequence[float], right: Sequence[float]) -> float:
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


def unit(value: Any, label: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32).reshape(-1)
    if array.size != 512 or not np.all(np.isfinite(array)):
        raise ValueError(f"{label} is not finite 512-D")
    norm = float(np.linalg.norm(array))
    if norm <= 1.0e-6:
        raise ValueError(f"{label} has zero norm")
    return (array / norm).astype(np.float32)


def load_gt(sequence: str) -> dict[int, dict[int, list[float]]]:
    path = DATA_ROOT / "train" / sequence / "gt" / "gt.txt"
    if not path.is_file():
        raise FileNotFoundError(path)
    result: dict[int, dict[int, list[float]]] = {}
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            fields = [item.strip() for item in line.split(",")]
            if len(fields) < 6:
                raise ValueError(f"malformed GT row {path}:{line_number}")
            frame = int(fields[0]) - 1
            identity = int(fields[1])
            x, y, width, height = [float(item) for item in fields[2:6]]
            box = [x, y, x + width, y + height]
            if not np.all(np.isfinite(np.asarray(box, dtype=np.float64))):
                raise ValueError(f"non-finite GT box {path}:{line_number}")
            result.setdefault(frame, {})[identity] = box
    return result


def dimensions(sequence: str, frame: int) -> tuple[int, int]:
    image = DATA_ROOT / "train" / sequence / "img1" / f"{int(frame) + 1:08d}.jpg"
    if not image.is_file():
        raise FileNotFoundError(image)
    from PIL import Image

    with Image.open(image) as handle:
        return int(handle.width), int(handle.height)


def source_vector(source: str) -> np.ndarray:
    result = np.zeros(SOURCE_FEATURE_DIM, dtype=np.float32)
    try:
        result[SOURCE_NAMES.index(str(source))] = 1.0
    except ValueError:
        result[-1] = 1.0
    return result


def authority_axis(row: Mapping[str, Any], label: str) -> tuple[list[int], list[int]]:
    states = [int(value) for value in row.get("association_state_axis", [])]
    declared_publics = [int(value) for value in row.get("public_id_axis", [])]
    if len(states) != len(set(states)) or len(declared_publics) != len(set(declared_publics)):
        raise ValueError(f"{label} has invalid explicit authority axes")
    if not states:
        raise ValueError(f"{label} has empty authority axes")
    # Historical rows intentionally expose the full persistent public axis,
    # while the score matrix columns contain only the active association-state
    # axis.  Recover the matrix-aligned public axis from explicit bindings;
    # never pair values by numeric position or infer public IDs from a state ID.
    by_state: dict[int, int] = {}
    for item in row.get("identity_rows", []):
        state_id = item.get("association_state_id")
        public_id = item.get("public_id")
        if state_id is None or public_id is None:
            continue
        state_id, public_id = int(state_id), int(public_id)
        if state_id in by_state and by_state[state_id] != public_id:
            raise ValueError(f"{label} has conflicting identity authority for state {state_id}")
        by_state[state_id] = public_id
    for item in row.get("candidate_rows", []):
        state_id = item.get("solver_association_state_id")
        public_id = item.get("solver_public_id")
        if state_id is None or public_id is None:
            continue
        state_id, public_id = int(state_id), int(public_id)
        if state_id in by_state and by_state[state_id] != public_id:
            raise ValueError(f"{label} has conflicting candidate authority for state {state_id}")
        by_state[state_id] = public_id
    if all(state_id in by_state for state_id in states):
        publics = [by_state[state_id] for state_id in states]
    elif len(declared_publics) == len(states):
        publics = declared_publics
    else:
        missing = [state_id for state_id in states if state_id not in by_state]
        raise ValueError(f"{label} lacks explicit state-to-public bindings: {missing}")
    if len(publics) != len(set(publics)):
        raise ValueError(f"{label} matrix-aligned public axis contains duplicates")
    return states, publics


def base_scores_for_public(row: Mapping[str, Any], public_id: int, label: str) -> dict[str, float]:
    candidates = list(row.get("candidate_rows", []))
    matrix = np.asarray(row.get("base_score_matrix", []), dtype=np.float64)
    _, publics = authority_axis(row, label)
    if matrix.shape != (len(candidates), len(publics)):
        raise ValueError(f"{label} base score shape {matrix.shape} does not match candidates/authority")
    if not np.all(np.isfinite(matrix)):
        raise ValueError(f"{label} has non-finite base scores")
    if int(public_id) not in publics:
        return {str(item["candidate_uid"]): 0.0 for item in candidates}
    column = publics.index(int(public_id))
    return {str(item["candidate_uid"]): float(matrix[index, column]) for index, item in enumerate(candidates)}


def best_label(candidates: Sequence[Mapping[str, Any]], target_box: Sequence[float] | None) -> tuple[int, str, float, bool]:
    if target_box is None:
        return len(candidates), "TARGET_NOT_VISIBLE", 0.0, False
    ious = [box_iou(item["box_xyxy"], target_box) for item in candidates]
    best = max(ious, default=0.0)
    if not ious or best < IOU_THRESHOLD:
        return len(candidates), "VISIBLE_NO_CANDIDATE_IOU_0.50", float(best), True
    index = max(range(len(ious)), key=lambda value: (ious[value], -value))
    return int(index), "HIGHEST_IOU_TARGET_CANDIDATE", float(best), True


def padded_memory(values: Sequence[np.ndarray], limit: int) -> tuple[np.ndarray, np.ndarray]:
    array = np.zeros((limit, 512), dtype=np.float32)
    mask = np.zeros(limit, dtype=np.bool_)
    for index, value in enumerate(list(values)[-limit:]):
        array[index] = unit(value, f"memory[{index}]")
        mask[index] = True
    return array, mask


def normalized_mean(values: Sequence[np.ndarray], fallback: np.ndarray) -> np.ndarray:
    if not values:
        return unit(fallback, "neighbor fallback")
    result = np.mean(np.stack([unit(value, "neighbor") for value in values], axis=0), axis=0)
    norm = float(np.linalg.norm(result))
    return (result / max(norm, 1.0e-6)).astype(np.float32)


def causal_selection_score(
    candidate: Mapping[str, Any],
    base_score: float,
    trusted: Sequence[np.ndarray],
    predicted_box: Sequence[float],
) -> float:
    feature = candidate.get("feature")
    similarity = 0.0
    if feature is not None and trusted:
        value = unit(feature, "causal candidate feature")
        similarity = float(np.dot(value, unit(trusted[-1], "causal trusted feature")))
    motion = box_iou(candidate["box_xyxy"], predicted_box)
    presence = float(np.clip(float(candidate.get("presence_score", 0.0)), 0.0, 1.0))
    return float(np.tanh(base_score) + 0.30 * similarity + 0.15 * motion + 0.05 * presence)


def event_arrays(event: Mapping[str, Any], gt: Mapping[int, Mapping[int, Sequence[float]]]) -> tuple[list[np.ndarray], list[dict[str, Any]], dict[str, Any]]:
    event_id = str(event["event_id"])
    event_manifest_path = Path(str(event["source_event_manifest"]))
    event_manifest = read_json(event_manifest_path)
    if event_manifest.get("status") != "PASS_N72R7_CLOSED_LOOP_EVENT_REPLAY":
        raise RuntimeError(f"frozen event manifest is not PASS: {event_id}")
    sequence = str(event["sequence"])
    event_frame = int(event["event_frame"])
    target_public_id = event_manifest.get("target_public_id")
    if target_public_id is None:
        raise RuntimeError(f"frozen event has no explicit target public axis value: {event_id}")
    target_public_id = int(target_public_id)
    c0_rows = read_jsonl(Path(str(event["c0_source"])))
    c1_rows = read_jsonl(Path(str(event["c1_source"])))
    target_rows = read_jsonl(Path(str(event["target_stream_source"])))
    expected = list(range(event_frame, event_frame + HORIZON + 1))
    for name, rows in (("c0", c0_rows), ("c1", c1_rows), ("target", target_rows)):
        if [int(row["frame"]) for row in rows] != expected or len(rows) != HORIZON + 1:
            raise RuntimeError(f"{event_id} {name} frame axis is incomplete")
        if any(row.get("runtime_future_gt_used") is not False for row in rows):
            raise RuntimeError(f"{event_id} {name} runtime future GT flag is not false")
    anchor_path = Path(str(event["target_stream_source"])).parent / "human_anchor.json"
    anchor_payload = read_json(anchor_path)
    anchor = unit(anchor_payload.get("feature"), f"{event_id} human anchor")
    anchor_box = [float(value) for value in event["current_gt_box"]]
    width, height = dimensions(sequence, event_frame)
    c0_by_frame = {int(row["frame"]): row for row in c0_rows}
    c1_by_frame = {int(row["frame"]): row for row in c1_rows}
    target_by_frame = {int(row["frame"]): row for row in target_rows}
    event_target_rows = list(target_by_frame[event_frame].get("candidate_rows", []))
    previous_raw = None if not event_target_rows else event_target_rows[0].get("official_raw_sam_id")
    previous_scope = None if not event_target_rows else event_target_rows[0].get("native_tid_scope")
    previous_raw = None if previous_raw is None else int(previous_raw)
    previous_scope = None if previous_scope is None else str(previous_scope)
    predicted_box = list(anchor_box)
    velocity = np.zeros(2, dtype=np.float64)
    trusted: list[np.ndarray] = [anchor]
    distractors: list[np.ndarray] = []
    previous_score = 0.0
    previous_uncertainty = 1.0
    trusted_age = 0
    examples: list[np.ndarray] = []
    source_arrays: list[np.ndarray] = []
    trusted_arrays: list[np.ndarray] = []
    trusted_masks: list[np.ndarray] = []
    distractor_arrays: list[np.ndarray] = []
    distractor_masks: list[np.ndarray] = []
    neighbor_arrays: list[np.ndarray] = []
    temporal_arrays: list[np.ndarray] = []
    labels: list[int] = []
    metadata: list[dict[str, Any]] = []
    label_counts: Counter[str] = Counter()
    source_counts: Counter[str] = Counter()
    for frame in range(event_frame + 1, event_frame + HORIZON + 1):
        c0 = c0_by_frame[frame]
        c1 = c1_by_frame[frame]
        target_row = target_by_frame[frame]
        main_raw = list(c0.get("candidate_rows", []))
        target_raw = list(target_row.get("candidate_rows", []))
        pool, _ = build_candidate_pool(
            main_raw,
            target_raw,
            sequence=sequence,
            frame=frame,
            include_target_session=True,
        )
        if not pool:
            raise RuntimeError(f"{event_id}:{frame} has empty candidate pool")
        c0_scores = base_scores_for_public(c0, target_public_id, f"{event_id}:c0:{frame}")
        c1_scores = base_scores_for_public(c1, target_public_id, f"{event_id}:c1:{frame}")
        base_scores = {
            str(candidate["candidate_uid"]): float(
                c0_scores.get(
                    str(candidate["candidate_uid"]),
                    c1_scores.get(str(candidate["candidate_uid"]), 0.0),
                )
            )
            for candidate in pool
        }
        candidate_vectors = np.stack(
            [
                candidate_feature_vector(
                    candidate,
                    anchor_feature=anchor,
                    anchor_box=anchor_box,
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
        if candidate_vectors.shape != (len(pool), CANDIDATE_FEATURE_DIM) or not np.all(np.isfinite(candidate_vectors)):
            raise RuntimeError(f"{event_id}:{frame} candidate feature tensor invalid")
        causal_scores = np.asarray(
            [causal_selection_score(candidate, base_scores[str(candidate["candidate_uid"])], trusted, predicted_box) for candidate in pool],
            dtype=np.float64,
        )
        order = sorted(range(len(pool)), key=lambda index: (-causal_scores[index], str(pool[index]["candidate_uid"])))
        selected_index = int(order[0])
        second_score = float(causal_scores[order[1]]) if len(order) > 1 else 0.0
        top_score = float(causal_scores[selected_index])
        margin = float(top_score - second_score)
        valid_features = [candidate_vectors[index, :512] for index in range(len(pool)) if np.linalg.norm(candidate_vectors[index, :512]) > 1.0e-6]
        neighbor_values = [candidate_vectors[index, :512] for index in order[1:] if np.linalg.norm(candidate_vectors[index, :512]) > 1.0e-6]
        neighbor = normalized_mean(neighbor_values, anchor)
        trusted_array, trusted_mask = padded_memory(trusted, MEMORY_SLOTS)
        distractor_array, distractor_mask = padded_memory(distractors, MEMORY_SLOTS)
        temporal = np.asarray(
            [
                float(frame - event_frame) / float(HORIZON),
                float(np.tanh(top_score)),
                float(np.tanh(second_score)),
                float(np.tanh(margin)),
                float(np.clip(previous_score, -1.0, 1.0)),
                float(np.clip(previous_uncertainty, 0.0, 1.0)),
                float(min(trusted_age, HORIZON)) / float(HORIZON),
                float(any(str(candidate["candidate_source"]) == "TARGET_SESSION_CURRENT_RAW" for candidate in pool)),
            ],
            dtype=np.float32,
        )
        target_box = gt.get(frame, {}).get(int(event["dataset_gt_id"]))
        label, reason, best_iou_value, visible = best_label(pool, target_box)
        label_counts[reason] += 1
        examples.append(candidate_vectors)
        source_arrays.append(np.stack([source_vector(str(candidate["candidate_source"])) for candidate in pool], axis=0))
        trusted_arrays.append(trusted_array)
        trusted_masks.append(trusted_mask)
        distractor_arrays.append(distractor_array)
        distractor_masks.append(distractor_mask)
        neighbor_arrays.append(neighbor)
        temporal_arrays.append(temporal)
        labels.append(int(label))
        for candidate in pool:
            source_counts[str(candidate["candidate_source"])] += 1
        metadata.append(
            {
                "event_id": event_id,
                "sequence": sequence,
                "split": str(event["split"]),
                "action_type": str(event["action_type"]),
                "event_frame": event_frame,
                "frame": frame,
                "frame_horizon": frame - event_frame,
                "candidate_uids": [str(candidate["candidate_uid"]) for candidate in pool],
                "candidate_sources": [str(candidate["candidate_source"]) for candidate in pool],
                "candidate_feature_sha256": [str(candidate.get("feature_sha256")) for candidate in pool],
                "label_index": int(label),
                "label_kind": "NONE" if label == len(pool) else "TARGET_CANDIDATE",
                "label_reason": reason,
                "posthoc_target_visible": bool(visible),
                "posthoc_best_iou": float(best_iou_value),
                "gt_used_offline": True,
                "runtime_future_gt_used": False,
                "public_id_inference": False,
                "not_real_human_evidence": True,
                "causal_state_update": "base_score_plus_previous_trusted_similarity_geometry_presence",
            }
        )
        selected = pool[selected_index]
        selected_feature = selected.get("feature")
        if selected_feature is not None:
            selected_feature = unit(selected_feature, f"{event_id}:{frame} selected feature")
            old_center = np.asarray([(predicted_box[0] + predicted_box[2]) / 2.0, (predicted_box[1] + predicted_box[3]) / 2.0])
            new_box = [float(value) for value in selected["box_xyxy"]]
            new_center = np.asarray([(new_box[0] + new_box[2]) / 2.0, (new_box[1] + new_box[3]) / 2.0])
            velocity = 0.5 * velocity + 0.5 * (new_center - old_center)
            predicted_box = new_box
            if top_score > 0.0 and margin >= 0.20:
                trusted.append(selected_feature)
                trusted_age = 0
            else:
                trusted_age += 1
            if len(order) > 1:
                second = pool[order[1]].get("feature")
                if second is not None:
                    distractors.append(unit(second, f"{event_id}:{frame} distractor feature"))
        else:
            trusted_age += 1
        previous_raw = selected.get("official_raw_sam_id")
        previous_raw = None if previous_raw is None else int(previous_raw)
        previous_scope = selected.get("native_scope")
        previous_scope = None if previous_scope is None else str(previous_scope)
        previous_score = top_score
        previous_uncertainty = float(1.0 / (1.0 + max(margin, 0.0)))
        trusted = trusted[-MEMORY_SLOTS:]
        distractors = distractors[-MEMORY_SLOTS:]
    return (
        examples,
        metadata,
        {
            "source_counts": dict(sorted(source_counts.items())),
            "label_counts": dict(sorted(label_counts.items())),
            "arrays": {
                "source_features": source_arrays,
                "trusted_memory": trusted_arrays,
                "trusted_mask": trusted_masks,
                "distractor_memory": distractor_arrays,
                "distractor_mask": distractor_masks,
                "neighbor_feature": neighbor_arrays,
                "temporal_features": temporal_arrays,
                "labels": labels,
            },
        },
    )


def build_split(events: Sequence[Mapping[str, Any]], gt_by_sequence: Mapping[str, Mapping[int, Mapping[int, Sequence[float]]]]) -> dict[str, Any]:
    examples: list[np.ndarray] = []
    source_features: list[np.ndarray] = []
    trusted_memory: list[np.ndarray] = []
    trusted_mask: list[np.ndarray] = []
    distractor_memory: list[np.ndarray] = []
    distractor_mask: list[np.ndarray] = []
    neighbor_feature: list[np.ndarray] = []
    temporal_features: list[np.ndarray] = []
    labels: list[int] = []
    metadata: list[dict[str, Any]] = []
    label_counts: Counter[str] = Counter()
    source_counts: Counter[str] = Counter()
    max_candidates = 0
    for event in sorted(events, key=lambda item: str(item["event_id"])):
        event_examples, event_metadata, details = event_arrays(event, gt_by_sequence[str(event["sequence"])])
        arrays = details["arrays"]
        examples.extend(event_examples)
        source_features.extend(arrays["source_features"])
        trusted_memory.extend(arrays["trusted_memory"])
        trusted_mask.extend(arrays["trusted_mask"])
        distractor_memory.extend(arrays["distractor_memory"])
        distractor_mask.extend(arrays["distractor_mask"])
        neighbor_feature.extend(arrays["neighbor_feature"])
        temporal_features.extend(arrays["temporal_features"])
        labels.extend(arrays["labels"])
        metadata.extend(event_metadata)
        label_counts.update(details["label_counts"])
        source_counts.update(details["source_counts"])
        max_candidates = max(max_candidates, max(len(value) for value in event_examples))
    count = len(examples)
    padded_candidates = np.zeros((count, max_candidates, CANDIDATE_FEATURE_DIM), dtype=np.float32)
    padded_sources = np.zeros((count, max_candidates, SOURCE_FEATURE_DIM), dtype=np.float32)
    candidate_mask = np.zeros((count, max_candidates), dtype=np.bool_)
    for index, value in enumerate(examples):
        width = len(value)
        padded_candidates[index, :width] = value
        padded_sources[index, :width] = source_features[index]
        candidate_mask[index, :width] = True
    none_index = max_candidates
    encoded_labels = np.asarray(
        [none_index if int(label) == int(len(metadata[index]["candidate_uids"])) else int(label) for index, label in enumerate(labels)],
        dtype=np.int64,
    )
    arrays = {
        "candidate_features": padded_candidates,
        "candidate_mask": candidate_mask,
        "source_features": padded_sources,
        "trusted_memory": np.stack(trusted_memory, axis=0).astype(np.float32),
        "trusted_mask": np.stack(trusted_mask, axis=0).astype(np.bool_),
        "distractor_memory": np.stack(distractor_memory, axis=0).astype(np.float32),
        "distractor_mask": np.stack(distractor_mask, axis=0).astype(np.bool_),
        "neighbor_feature": np.stack(neighbor_feature, axis=0).astype(np.float32),
        "temporal_features": np.stack(temporal_features, axis=0).astype(np.float32),
        "labels": encoded_labels,
        "candidate_counts": np.asarray([len(item["candidate_uids"]) for item in metadata], dtype=np.int64),
    }
    for key, value in arrays.items():
        if not np.all(np.isfinite(value)):
            raise RuntimeError(f"non-finite corpus array: {key}")
    return {
        "arrays": arrays,
        "metadata": metadata,
        "summary": {
            "event_count": len(events),
            "sequence_count": len({str(event["sequence"]) for event in events}),
            "example_count": count,
            "max_candidates": max_candidates,
            "candidate_feature_dim": CANDIDATE_FEATURE_DIM,
            "source_feature_dim": SOURCE_FEATURE_DIM,
            "trusted_memory_slots": MEMORY_SLOTS,
            "distractor_memory_slots": MEMORY_SLOTS,
            "temporal_feature_dim": TEMPORAL_FEATURE_DIM,
            "label_counts": dict(sorted(label_counts.items())),
            "source_counts": dict(sorted(source_counts.items())),
            "runtime_future_gt_used": False,
            "gt_used_only_offline_label_generation": True,
        },
    }


def canonical_hash(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")).hexdigest()


def main() -> int:
    started = now_utc()
    base_status: dict[str, Any] = {
        "schema_version": "N72R9_TEMPORAL_CORPUS_STATUS_V1",
        "stage": "N72R9_SOURCE_AWARE_TEMPORAL_CORPUS",
        "started_at_utc": started,
        "runtime_future_gt_used": False,
        "interaction_source": "simulated_from_gt",
        "not_real_human_evidence": True,
    }
    try:
        protocol = read_json(PROTOCOL_PATH)
        protocol_hash = str(protocol.get("protocol_sha256"))
        protocol_body = dict(protocol)
        protocol_body.pop("protocol_sha256", None)
        if canonical_hash(protocol_body) != protocol_hash:
            raise RuntimeError("N72R9 protocol hash mismatch")
        events = list(protocol.get("source_event_selection", {}).get("events", []))
        if len(events) != 32 or len({str(item["event_id"]) for item in events}) != len(events):
            raise RuntimeError(f"expected 32 unique frozen D2 events, found {len(events)}")
        gt_by_sequence = {sequence: load_gt(sequence) for sequence in sorted({str(item["sequence"]) for item in events})}
        split_outputs: dict[str, dict[str, Any]] = {}
        for split in ("train", "validation"):
            selected = [item for item in events if str(item["split"]) == split]
            if not selected:
                raise RuntimeError(f"empty {split} event split")
            split_outputs[split] = build_split(selected, gt_by_sequence)
            arrays = split_outputs[split]["arrays"]
            atomic_npz(OUTPUT_ROOT / f"{split}.npz", **arrays)
            atomic_jsonl(OUTPUT_ROOT / f"{split}_metadata.jsonl", split_outputs[split]["metadata"])
        corpus = {
            "schema_version": "N72R9_SOURCE_AWARE_TEMPORAL_CORPUS_V1",
            "created_at_utc": now_utc(),
            "status": "PASS_N72R9_SOURCE_AWARE_CORPUS_SEALED",
            "protocol": str(PROTOCOL_PATH),
            "protocol_sha256": protocol_hash,
            "events": len(events),
            "interaction_source": "simulated_from_gt",
            "not_real_human_evidence": True,
            "gt_used_only_offline_label_generation": True,
            "runtime_future_gt_used": False,
            "source_feature_names": list(SOURCE_NAMES),
            "splits": {},
        }
        for split, value in split_outputs.items():
            npz_path = OUTPUT_ROOT / f"{split}.npz"
            metadata_path = OUTPUT_ROOT / f"{split}_metadata.jsonl"
            corpus["splits"][split] = {
                **value["summary"],
                "npz": str(npz_path),
                "npz_sha256": sha256_file(npz_path),
                "metadata": str(metadata_path),
                "metadata_sha256": sha256_file(metadata_path),
            }
        atomic_json(OUTPUT_ROOT / "corpus_manifest.json", corpus)
        result = {
            **base_status,
            "status": "PASS_N72R9_SOURCE_AWARE_CORPUS_SEALED",
            "finished_at_utc": now_utc(),
            "protocol": str(PROTOCOL_PATH),
            "protocol_sha256": protocol_hash,
            "corpus_manifest": str(OUTPUT_ROOT / "corpus_manifest.json"),
            "train_examples": int(split_outputs["train"]["summary"]["example_count"]),
            "validation_examples": int(split_outputs["validation"]["summary"]["example_count"]),
            "source_coverage": {
                "MAIN_B0_CANDIDATE": int(corpus["splits"]["train"]["source_counts"].get("MAIN_B0_CANDIDATE", 0) + corpus["splits"]["validation"]["source_counts"].get("MAIN_B0_CANDIDATE", 0)),
                "TARGET_SESSION_CURRENT_RAW": int(corpus["splits"]["train"]["source_counts"].get("TARGET_SESSION_CURRENT_RAW", 0) + corpus["splits"]["validation"]["source_counts"].get("TARGET_SESSION_CURRENT_RAW", 0)),
                "TARGET_SESSION_REQUERY": 0,
            },
            "fresh_confirmation_authorized": False,
            "production_authorized": False,
        }
        atomic_json(STAGE_PATH, result)
        print(json.dumps(result, sort_keys=True))
        return 0
    except Exception as exc:
        failure = OUTPUT_ROOT / "attempts" / f"corpus_failure_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.json"
        failure_payload = {
            **base_status,
            "status": "FAIL_N72R9_SOURCE_AWARE_CORPUS",
            "finished_at_utc": now_utc(),
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }
        atomic_json(failure, failure_payload)
        atomic_json(STAGE_PATH, {**failure_payload, "failure_artifact": str(failure)})
        print(json.dumps({"status": failure_payload["status"], "failure_artifact": str(failure), "error": str(exc)}, sort_keys=True))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
