#!/usr/bin/env python3
"""Seal the N72R11 causal training corpus from completed secondary events.

The secondary interaction workers are the only source of fresh future
candidate observations.  This builder consumes their immutable ``done.json``
artifacts, reconstructs the *causal* target state without reading GT, and
attaches labels only after every runtime feature/state row has been sealed in
memory.  The resulting arrays are intentionally public-ID-free model inputs;
the accompanying metadata retains enough provenance for an offline audit and
for the target-edge bridge.

This is a research artifact builder, not a production tracker.  In
particular, all events remain ``simulated_from_gt`` and no model checkpoint is
created here.
"""

from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
import traceback
from typing import Any, Mapping, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in __import__("sys").path:
    __import__("sys").path.insert(0, str(ROOT))

from sam3_intermot.reacquisition.target_candidate_pool import (  # noqa: E402
    FUTURE_FRAME_REQUERY,
    build_candidate_pool_with_future_requery,
)
from sam3_intermot.reacquisition.target_id_features import (  # noqa: E402
    CANDIDATE_FEATURE_DIM,
    candidate_feature_vector,
)
from sam3_intermot.reacquisition.temporal_state_policy import (  # noqa: E402
    TEMPORAL_FEATURE_SCHEMA,
    TEMPORAL_FEATURE_DIM as SHARED_TEMPORAL_FEATURE_DIM,
    TemporalIdentityState,
    build_temporal_features,
    initialize_temporal_state,
    state_audit,
    state_memory_arrays,
    update_temporal_state,
)


SCHEDULE_PATH = ROOT / "outputs/N72R11/secondary_event_manifest.json"
PROTOCOL_PATH = ROOT / "outputs/N72R11/event_protocol.json"
BATCH_MANIFEST_PATH = ROOT / "outputs/N72R11/secondary_interactions_attempt_02/batch_manifest_attempt_02.json"
OUTPUT_ROOT = ROOT / "outputs/N72R11/training_v3"
STAGE_PATH = ROOT / "outputs/N72R11/stage_06_status.json"
DATA_ROOT = Path("/data1/LWR/vranlee/DATASETS/JDE/dancetrack")
HORIZON = 100
IOU_THRESHOLD = 0.50
SOURCE_NAMES = (
    "MAIN_B0_CANDIDATE",
    "TARGET_SESSION_CURRENT_RAW",
    "STATIC_EVENT_REQUERY",
    "FUTURE_FRAME_REQUERY",
    "UNKNOWN",
)
SOURCE_FEATURE_DIM = len(SOURCE_NAMES)
MEMORY_DIM = 512
RECENT_SLOTS = 4
LONG_TERM_SLOTS = 4
DISTRACTOR_SLOTS = 8
TEMPORAL_DIM = SHARED_TEMPORAL_FEATURE_DIM
CAUSAL_MARGIN_ADMISSION = 0.20


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


def atomic_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
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


def unit(value: Any, label: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32).reshape(-1)
    if array.size != MEMORY_DIM or not np.all(np.isfinite(array)):
        raise ValueError(f"{label} must be finite {MEMORY_DIM}-D")
    norm = float(np.linalg.norm(array))
    if norm <= 1.0e-6:
        raise ValueError(f"{label} has zero norm")
    return (array / norm).astype(np.float32)


def box_xyxy(value: Any, label: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64).reshape(-1)
    if result.size != 4 or not np.all(np.isfinite(result)) or result[2] <= result[0] or result[3] <= result[1]:
        raise ValueError(f"{label} is not a finite positive XYXY box")
    return result


def finite_box_xyxy(value: Any, label: str) -> np.ndarray:
    """Validate finite candidate coordinates, including zero-area observations."""
    result = np.asarray(value, dtype=np.float64).reshape(-1)
    if result.size != 4 or not np.all(np.isfinite(result)):
        raise ValueError(f"{label} is not a finite XYXY box")
    return result


def box_iou(left: Any, right: Any) -> float:
    a = finite_box_xyxy(left, "left box")
    b = finite_box_xyxy(right, "right box")
    if a[2] <= a[0] or a[3] <= a[1] or b[2] <= b[0] or b[3] <= b[1]:
        return 0.0
    intersection = max(0.0, min(float(a[2]), float(b[2])) - max(float(a[0]), float(b[0]))) * max(
        0.0, min(float(a[3]), float(b[3])) - max(float(a[1]), float(b[1]))
    )
    area_a = float(a[2] - a[0]) * float(a[3] - a[1])
    area_b = float(b[2] - b[0]) * float(b[3] - b[1])
    union = area_a + area_b - intersection
    return float(intersection / union) if union > 0.0 else 0.0


def source_vector(source: str) -> np.ndarray:
    result = np.zeros(SOURCE_FEATURE_DIM, dtype=np.float32)
    try:
        result[SOURCE_NAMES.index(str(source))] = 1.0
    except ValueError:
        result[-1] = 1.0
    return result


def load_gt(sequence: str) -> dict[int, dict[int, list[float]]]:
    path = DATA_ROOT / "train" / str(sequence) / "gt" / "gt.txt"
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


def image_dimensions(sequence: str, frame: int) -> tuple[int, int]:
    path = DATA_ROOT / "train" / str(sequence) / "img1" / f"{int(frame) + 1:08d}.jpg"
    if not path.is_file():
        raise FileNotFoundError(path)
    from PIL import Image

    with Image.open(path) as handle:
        return int(handle.width), int(handle.height)


def active_public_axis(row: Mapping[str, Any], label: str) -> list[int]:
    solver = row.get("solver")
    if not isinstance(solver, Mapping):
        raise ValueError(f"{label} lacks solver audit")
    axis = [int(value) for value in solver.get("public_id_axis", [])]
    if not axis or len(axis) != len(set(axis)):
        raise ValueError(f"{label} has invalid active public axis")
    return axis


def base_score_vectors(row: Mapping[str, Any], target_public_id: int, label: str) -> tuple[dict[str, float], dict[str, float]]:
    candidates = list(row.get("candidate_rows", []))
    axis = active_public_axis(row, label)
    matrix = np.asarray(row.get("base_score_matrix", []), dtype=np.float64)
    if matrix.shape != (len(candidates), len(axis)):
        raise ValueError(f"{label} base matrix {matrix.shape} does not match {(len(candidates), len(axis))}")
    if not np.all(np.isfinite(matrix)):
        raise ValueError(f"{label} base matrix is non-finite")
    target_column = axis.index(int(target_public_id)) if int(target_public_id) in axis else None
    target: dict[str, float] = {}
    other: dict[str, float] = {}
    for index, candidate in enumerate(candidates):
        uid = str(candidate.get("candidate_uid"))
        target[uid] = 0.0 if target_column is None else float(matrix[index, target_column])
        alternatives = [float(matrix[index, col]) for col, public_id in enumerate(axis) if public_id != int(target_public_id)]
        other[uid] = max(alternatives, default=0.0)
    return target, other


def pad_memory(values: Sequence[np.ndarray], slots: int) -> tuple[np.ndarray, np.ndarray]:
    array = np.zeros((int(slots), MEMORY_DIM), dtype=np.float32)
    mask = np.zeros(int(slots), dtype=np.bool_)
    for index, value in enumerate(list(values)[-int(slots):]):
        array[index] = unit(value, f"memory[{index}]")
        mask[index] = True
    return array, mask


def normalized_mean(values: Sequence[np.ndarray], fallback: np.ndarray) -> np.ndarray:
    if not values:
        return unit(fallback, "neighbor fallback")
    result = np.mean(np.stack([unit(value, "neighbor") for value in values], axis=0), axis=0)
    norm = float(np.linalg.norm(result))
    if norm <= 1.0e-6:
        return unit(fallback, "neighbor fallback")
    return (result / norm).astype(np.float32)


def causal_score(candidate: Mapping[str, Any], base_score: float, trusted: Sequence[np.ndarray], predicted_box: Sequence[float]) -> float:
    feature = candidate.get("feature")
    similarity = 0.0
    if feature is not None and trusted:
        similarity = float(np.dot(unit(feature, "causal candidate feature"), unit(trusted[-1], "causal trusted feature")))
    presence = float(candidate.get("presence_score", candidate.get("confidence", 0.0)) or 0.0)
    if not math.isfinite(presence):
        raise ValueError("causal candidate presence is non-finite")
    return float(
        np.tanh(float(base_score))
        + 0.30 * similarity
        + 0.15 * box_iou(candidate["box_xyxy"], predicted_box)
        + 0.05 * float(np.clip(presence, 0.0, 1.0))
    )


def load_secondary_artifact(item: Mapping[str, Any], record: Mapping[str, Any]) -> tuple[dict[int, dict[str, Any]], dict[int, list[dict[str, Any]]], dict[str, Any]]:
    event_id = str(item["event_id"])
    returncode = record.get("returncode")
    if record.get("status") != "PASS" or (returncode is not None and int(returncode) != 0):
        raise RuntimeError(f"batch record is not PASS: {event_id}:{record.get('status')}")
    # R1 has two sealed manifest schemas: child records carry returncode/log,
    # while the retry merge carries an explicit hash-protected done_artifact
    # and intentionally omits returncode.  Prefer the explicit artifact and
    # fall back to the child log layout without accepting an unsealed record.
    explicit_done = record.get("done_artifact") or record.get("done")
    done_path = Path(str(explicit_done)) if explicit_done else Path()
    if not done_path.is_file():
        log_value = record.get("log")
        if not isinstance(log_value, str) or not log_value:
            raise FileNotFoundError(f"sealed secondary done artifact is not declared: {event_id}")
        done_path = Path(log_value).parent.parent / event_id / "done.json"
    if not done_path.is_file():
        raise FileNotFoundError(f"missing secondary done artifact: {done_path}")
    declared_hash = record.get("done_sha256")
    if declared_hash is not None and sha256_file(done_path) != str(declared_hash):
        raise RuntimeError(f"batch done artifact hash mismatch: {event_id}")
    done = read_json(done_path)
    if done.get("status") != "PASS_N72R11_SECONDARY_INTERACTION":
        raise RuntimeError(f"secondary done status is not PASS: {event_id}")
    if str(done.get("event_id")) != event_id or done.get("runtime_future_gt_used") is not False or done.get("runtime_gt_read") is not False:
        raise RuntimeError(f"secondary done identity/GT audit invalid: {event_id}")
    if done.get("interaction_source") != "simulated_from_gt" or done.get("not_real_human_evidence") is not True:
        raise RuntimeError(f"secondary provenance invalid: {event_id}")
    stream_path = Path(str(done["target_stream"]))
    mapping_path = Path(str(done["mapping"]))
    live_path = Path(str(done["live_requery"]))
    anchor_path = Path(str(done["human_anchor"]))
    for path, key in ((stream_path, "target_stream_sha256"), (mapping_path, "mapping_sha256"), (live_path, "live_requery_sha256"), (anchor_path, "human_anchor_sha256")):
        if not path.is_file():
            raise FileNotFoundError(path)
        if sha256_file(path) != str(done[key]):
            raise RuntimeError(f"secondary artifact hash mismatch: {event_id}:{path.name}")
    target_rows = read_jsonl(stream_path)
    secondary_frame = int(item["secondary_frame"])
    end_frame = int(done["end_frame"])
    expected_frames = list(range(secondary_frame, end_frame + 1))
    if [int(row.get("frame", -1)) for row in target_rows] != expected_frames:
        raise RuntimeError(f"target stream frame axis incomplete: {event_id}")
    target_by_frame: dict[int, dict[str, Any]] = {}
    for row in target_rows:
        frame = int(row["frame"])
        if frame in target_by_frame:
            raise RuntimeError(f"duplicate target frame: {event_id}:{frame}")
        for flag in ("runtime_future_gt_used", "runtime_gt_read", "posthoc_gt_used"):
            if row.get(flag) is not False:
                raise RuntimeError(f"target stream violates {flag}: {event_id}:{frame}")
        candidates = row.get("candidate_rows")
        if not isinstance(candidates, list):
            raise RuntimeError(f"target stream lacks candidate rows: {event_id}:{frame}")
        uids = [str(value.get("candidate_uid")) for value in candidates]
        if len(uids) != len(set(uids)):
            raise RuntimeError(f"duplicate target candidate UID: {event_id}:{frame}")
        target_by_frame[frame] = row
    if not target_by_frame.get(secondary_frame, {}).get("candidate_rows"):
        raise RuntimeError(f"event frame has no target candidate: {event_id}")
    live = read_json(live_path)
    if live.get("runtime_future_gt_used") is not False or live.get("runtime_gt_read") is not False or live.get("posthoc_gt_used") is not False:
        raise RuntimeError(f"live requery violates GT boundary: {event_id}")
    future_rows = list(live.get("future_rows", []))
    rows_by_trigger: dict[int, list[dict[str, Any]]] = {}
    seen_keys: set[tuple[int, int, str]] = set()
    for row in future_rows:
        frame = int(row.get("frame", -1))
        trigger = int(row.get("trigger_frame", -1))
        if not secondary_frame < trigger <= frame <= end_frame:
            raise RuntimeError(f"future requery row has invalid frame/trigger: {event_id}:{trigger}:{frame}")
        for flag in ("runtime_future_gt_used", "runtime_gt_read", "posthoc_gt_used", "public_id_inference"):
            if row.get(flag) is not False:
                raise RuntimeError(f"future requery row violates {flag}: {event_id}:{frame}")
        if row.get("public_id") is not None or row.get("candidate_kind") != "FUTURE_FRAME_REQUERY_CANDIDATE":
            raise RuntimeError(f"future requery row carries invalid authority/kind: {event_id}:{frame}")
        key = (trigger, frame, str(row.get("candidate_uid")))
        if key in seen_keys:
            raise RuntimeError(f"duplicate future requery row: {event_id}:{key}")
        seen_keys.add(key)
        rows_by_trigger.setdefault(trigger, []).append(dict(row))
    trigger_records = list(live.get("trigger_records", []))
    if not trigger_records:
        raise RuntimeError(f"live requery has no trigger records: {event_id}")
    by_trigger_record: dict[int, dict[str, Any]] = {}
    for record_item in trigger_records:
        frame = int(record_item.get("frame", -1))
        if not secondary_frame < frame <= end_frame or frame in by_trigger_record:
            raise RuntimeError(f"invalid/duplicate live trigger record: {event_id}:{frame}")
        by_trigger_record[frame] = record_item
        expected = len(rows_by_trigger.get(frame, [])) if record_item.get("triggered") else 0
        if int(record_item.get("future_row_count", 0)) != expected:
            raise RuntimeError(f"trigger/future row count mismatch: {event_id}:{frame}")
    active_by_frame: dict[int, list[dict[str, Any]]] = {}
    for frame in range(secondary_frame + 1, end_frame + 1):
        latest_trigger = max((value for value in by_trigger_record if value <= frame), default=None)
        if latest_trigger is None or not by_trigger_record[latest_trigger].get("triggered"):
            active_by_frame[frame] = []
            continue
        active_by_frame[frame] = [dict(value) for value in rows_by_trigger.get(latest_trigger, []) if int(value["frame"]) == frame]
    anchor = read_json(anchor_path)
    if anchor.get("runtime_future_gt_used") is not False or anchor.get("posthoc_gt_used") is not False:
        raise RuntimeError(f"human anchor violates GT boundary: {event_id}")
    if int(anchor.get("event_frame", -1)) != secondary_frame or int(anchor.get("public_id", -1)) != int(item["target_public_id"]):
        raise RuntimeError(f"human anchor event/public mismatch: {event_id}")
    unit(anchor.get("feature"), f"{event_id} human anchor")
    return target_by_frame, active_by_frame, {
        "done": done,
        "anchor": anchor,
        "event_frame": secondary_frame,
        "end_frame": end_frame,
        "future_requery_row_count": len(future_rows),
        "selected_trigger_count": sum(1 for value in trigger_records if value.get("triggered")),
    }


def _label_for_pool(candidates: Sequence[Mapping[str, Any]], target_box: Sequence[float] | None) -> tuple[int, float, bool, str]:
    if target_box is None:
        return len(candidates), 0.0, False, "TARGET_NOT_VISIBLE"
    ious = [box_iou(candidate["box_xyxy"], target_box) for candidate in candidates]
    best = max(ious, default=0.0)
    if best < IOU_THRESHOLD:
        return len(candidates), float(best), True, "VISIBLE_NO_CANDIDATE_IOU_0.50"
    index = max(range(len(ious)), key=lambda value: (ious[value], -value))
    return int(index), float(best), True, "HIGHEST_IOU_TARGET_CANDIDATE"


def build_event_runtime(item: Mapping[str, Any], record: Mapping[str, Any]) -> tuple[list[dict[str, Any]], dict[str, list[np.ndarray]], dict[str, int]]:
    event_id = str(item["event_id"])
    target_by_frame, active_by_frame, details = load_secondary_artifact(item, record)
    done = details["done"]
    c0_path = Path(str(item["c0_source"]))
    if not c0_path.is_file() or sha256_file(c0_path) != str(item["c0_source_sha256"]):
        raise RuntimeError(f"frozen C0 hash mismatch: {event_id}")
    c0_rows = read_jsonl(c0_path)
    c0_by_frame: dict[int, dict[str, Any]] = {}
    for row in c0_rows:
        frame = int(row.get("frame", -1))
        if frame in c0_by_frame:
            raise RuntimeError(f"duplicate C0 frame: {event_id}:{frame}")
        if row.get("runtime_future_gt_used") is not False or row.get("runtime_gt_read") is not False or row.get("posthoc_gt_used") is not False:
            raise RuntimeError(f"C0 violates GT boundary: {event_id}:{frame}")
        c0_by_frame[frame] = row
    start = int(item["secondary_frame"])
    end = int(done["end_frame"])
    if any(frame not in c0_by_frame for frame in range(start, end + 1)):
        raise RuntimeError(f"C0 does not cover secondary window: {event_id}")
    anchor = unit(details["anchor"]["feature"], f"{event_id} anchor")
    anchor_box = box_xyxy(item["current_target_box_posthoc_selection_only"], f"{event_id} anchor box")
    width, height = image_dimensions(str(item["sequence"]), start)
    event_frame_candidates = list(target_by_frame.get(start, {}).get("candidate_rows", []))
    initial_raw = event_frame_candidates[0].get("official_raw_sam_id") if event_frame_candidates else None
    initial_scope = event_frame_candidates[0].get("native_scope", event_frame_candidates[0].get("native_tid_scope")) if event_frame_candidates else None
    state = initialize_temporal_state(
        anchor_feature=anchor,
        anchor_box=anchor_box,
        previous_raw_sam_id=None if initial_raw is None else int(initial_raw),
        previous_native_scope=None if initial_scope is None else str(initial_scope),
    )
    rows: list[dict[str, Any]] = []
    arrays: dict[str, list[np.ndarray]] = {
        "candidate_features": [],
        "candidate_mask": [],
        "source_features": [],
        "human_anchor": [],
        "recent_trusted_memory": [],
        "recent_trusted_mask": [],
        "long_term_trusted_memory": [],
        "long_term_trusted_mask": [],
        "distractor_memory": [],
        "distractor_mask": [],
        "neighbor_feature": [],
        "temporal_features": [],
        "legacy_target_scores": [],
        "legacy_best_other_scores": [],
        "incumbent_target": [],
        "incumbent_other": [],
        "confidence": [],
        "presence": [],
        "motion_iou": [],
        "protected_candidate_mask": [],
    }
    label_counts: Counter[str] = Counter()
    for frame in range(start + 1, end + 1):
        c0 = c0_by_frame[frame]
        main_rows = [dict(value) for value in c0.get("candidate_rows", [])]
        current_rows = [dict(value) for value in target_by_frame[frame].get("candidate_rows", [])]
        future_rows = active_by_frame.get(frame, [])
        pool, pool_audit = build_candidate_pool_with_future_requery(
            main_rows,
            current_rows,
            future_rows,
            sequence=str(item["sequence"]),
            frame=frame,
        )
        if not pool:
            raise RuntimeError(f"empty causal candidate pool: {event_id}:{frame}")
        target_scores_raw, other_scores_raw = base_score_vectors(c0, int(item["target_public_id"]), f"{event_id}:{frame}")
        main_uid_set = {str(value["candidate_uid"]) for value in main_rows}
        target_scores = {str(candidate["candidate_uid"]): float(target_scores_raw.get(str(candidate["candidate_uid"]), 0.0)) for candidate in pool}
        other_scores = {str(candidate["candidate_uid"]): float(other_scores_raw.get(str(candidate["candidate_uid"]), 0.0)) for candidate in pool}
        candidate_vectors = np.stack(
            [
                candidate_feature_vector(
                    candidate,
                    anchor_feature=anchor,
                    anchor_box=anchor_box,
                    predicted_box=state.predicted_box,
                    previous_raw_sam_id=state.previous_raw_sam_id,
                    previous_native_scope=state.previous_native_scope,
                    image_width=width,
                    image_height=height,
                    candidate_count=len(pool),
                    base_target_score=target_scores[str(candidate["candidate_uid"])],
                )
                for candidate in pool
            ],
            axis=0,
        ).astype(np.float32)
        if candidate_vectors.shape != (len(pool), CANDIDATE_FEATURE_DIM) or not np.all(np.isfinite(candidate_vectors)):
            raise RuntimeError(f"invalid causal candidate feature tensor: {event_id}:{frame}")
        causal_values = np.asarray(
            [causal_score(candidate, target_scores[str(candidate["candidate_uid"])], state.recent_trusted, state.predicted_box) for candidate in pool],
            dtype=np.float64,
        )
        order = sorted(range(len(pool)), key=lambda index: (-float(causal_values[index]), str(pool[index]["candidate_uid"])))
        top_index = int(order[0])
        second_score = float(causal_values[order[1]]) if len(order) > 1 else 0.0
        top_score = float(causal_values[top_index])
        margin = float(top_score - second_score)
        neighbor = normalized_mean(
            [candidate_vectors[index, :MEMORY_DIM] for index in order[1:] if np.linalg.norm(candidate_vectors[index, :MEMORY_DIM]) > 1.0e-6],
            anchor,
        )
        recent_array, recent_mask, long_array, long_mask, distractor_array, distractor_mask = state_memory_arrays(state)
        temporal = build_temporal_features(
            state,
            frame_horizon=frame - start,
            causal_top_score=top_score,
            causal_second_score=second_score,
            causal_margin=margin,
            has_future_requery=any(str(candidate["candidate_source"]) == FUTURE_FRAME_REQUERY for candidate in pool),
        )
        pool_uids = [str(candidate["candidate_uid"]) for candidate in pool]
        row = {
            "event_id": event_id,
            "sequence": str(item["sequence"]),
            "split": str(item["split"]),
            "action_type": str(item["action_type"]),
            "event_frame": start,
            "frame": frame,
            "frame_horizon": frame - start,
            "target_public_id_for_offline_audit": int(item["target_public_id"]),
            "target_dataset_gt_id_for_offline_label": int(item["target_dataset_gt_id"]),
            "anchor_box": [float(value) for value in anchor_box],
            "candidate_uids": pool_uids,
            "candidate_boxes": [list(map(float, candidate["box_xyxy"])) for candidate in pool],
            "candidate_sources": [str(candidate["candidate_source"]) for candidate in pool],
            "candidate_feature_sha256": [None if candidate.get("feature_sha256") is None else str(candidate["feature_sha256"]) for candidate in pool],
            "candidate_incumbent_public_ids": [candidate.get("incumbent_public_id_if_any") for candidate in pool],
            "candidate_confidence": [float(candidate["confidence"]) for candidate in pool],
            "candidate_presence": [float(candidate["presence_score"]) for candidate in pool],
            "base_target_scores": [target_scores[uid] for uid in pool_uids],
            "base_best_other_scores": [other_scores[uid] for uid in pool_uids],
            "causal_selected_index": top_index,
            "causal_selected_uid": pool_uids[top_index],
            "causal_top_score": top_score,
            "causal_second_score": second_score,
            "causal_margin": margin,
            "causal_state_update": "shared_temporal_state_policy; base_target_plus_anchor_similarity_geometry_presence_only",
            "temporal_feature_schema": list(TEMPORAL_FEATURE_SCHEMA),
            "active_live_candidate_count": len(future_rows),
            "pool_audit": pool_audit,
            "runtime_future_gt_used": False,
            "runtime_gt_read": False,
            "public_id_inference": False,
            "gt_used_offline": True,
            "interaction_source": "simulated_from_gt",
            "not_real_human_evidence": True,
        }
        rows.append(row)
        arrays["candidate_features"].append(candidate_vectors)
        arrays["candidate_mask"].append(np.ones(len(pool), dtype=np.bool_))
        arrays["source_features"].append(np.stack([source_vector(str(candidate["candidate_source"])) for candidate in pool], axis=0))
        arrays["human_anchor"].append(anchor.copy())
        arrays["recent_trusted_memory"].append(recent_array)
        arrays["recent_trusted_mask"].append(recent_mask)
        arrays["long_term_trusted_memory"].append(long_array)
        arrays["long_term_trusted_mask"].append(long_mask)
        arrays["distractor_memory"].append(distractor_array)
        arrays["distractor_mask"].append(distractor_mask)
        arrays["neighbor_feature"].append(neighbor)
        arrays["temporal_features"].append(temporal)
        arrays["legacy_target_scores"].append(np.asarray([target_scores[uid] for uid in pool_uids], dtype=np.float32))
        arrays["legacy_best_other_scores"].append(np.asarray([other_scores[uid] for uid in pool_uids], dtype=np.float32))
        arrays["incumbent_target"].append(np.asarray([float(candidate.get("incumbent_public_id_if_any") == int(item["target_public_id"])) for candidate in pool], dtype=np.float32))
        arrays["incumbent_other"].append(np.asarray([float(candidate.get("incumbent_public_id_if_any") is not None and candidate.get("incumbent_public_id_if_any") != int(item["target_public_id"])) for candidate in pool], dtype=np.float32))
        arrays["confidence"].append(np.asarray([float(candidate["confidence"]) for candidate in pool], dtype=np.float32))
        arrays["presence"].append(np.asarray([float(candidate["presence_score"]) for candidate in pool], dtype=np.float32))
        arrays["motion_iou"].append(np.asarray([box_iou(candidate["box_xyxy"], state.predicted_box) for candidate in pool], dtype=np.float32))
        arrays["protected_candidate_mask"].append(np.asarray([bool(candidate.get("incumbent_public_id_if_any") is not None and candidate.get("incumbent_public_id_if_any") != int(item["target_public_id"])) for candidate in pool], dtype=np.bool_))
        selected = pool[top_index]
        state_update = update_temporal_state(
            state,
            candidates=pool,
            target_uid=pool_uids[top_index],
            selected_uid=pool_uids[top_index],
            selected_score=top_score,
            selected_margin=margin,
            fused_target_scores=[target_scores[uid] for uid in pool_uids],
            frame_horizon=frame - start,
            assigned_candidate=selected,
            base_top_score=top_score,
            base_second_score=second_score,
        )
        row["temporal_state_update"] = state_update
        row["temporal_state_after"] = state_audit(state)
    return rows, arrays, {"label_count_placeholder": len(rows), "future_requery_rows": int(details["future_requery_row_count"]), "selected_trigger_count": int(details["selected_trigger_count"])}


def attach_offline_labels(rows: Sequence[dict[str, Any]], gt: Mapping[int, Mapping[int, Sequence[float]]]) -> Counter[str]:
    counts: Counter[str] = Counter()
    # Runtime state/tensors have already been constructed.  This function is
    # the only place that consults the dataset GT for labels.
    for row in rows:
        target_box = gt.get(int(row["frame"]), {}).get(int(row["target_dataset_gt_id_for_offline_label"]))
        label, best_iou, visible, reason = _label_for_pool(
            [{"box_xyxy": box} for box in row["candidate_boxes"]],
            target_box,
        )
        row["label_index_raw"] = int(label)
        row["label_kind"] = "NONE" if label == len(row["candidate_uids"]) else "TARGET_CANDIDATE"
        row["posthoc_target_visible"] = bool(visible)
        row["posthoc_best_iou"] = float(best_iou)
        row["label_reason"] = reason
        counts[reason] += 1
    return counts


def stack_split(event_payloads: Sequence[tuple[list[dict[str, Any]], dict[str, list[np.ndarray]], dict[str, int]]], gt_by_sequence: Mapping[str, Mapping[int, Mapping[int, Sequence[float]]]]) -> tuple[dict[str, np.ndarray], list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    collected: dict[str, list[np.ndarray]] = {key: [] for key in (
        "candidate_features", "candidate_mask", "source_features", "human_anchor",
        "recent_trusted_memory", "recent_trusted_mask", "long_term_trusted_memory", "long_term_trusted_mask",
        "distractor_memory", "distractor_mask", "neighbor_feature", "temporal_features", "legacy_target_scores",
        "legacy_best_other_scores", "incumbent_target", "incumbent_other", "confidence", "presence", "motion_iou",
        "protected_candidate_mask",
    )}
    for event_rows, event_arrays, _details in event_payloads:
        attach_offline_labels(event_rows, gt_by_sequence[str(event_rows[0]["sequence"])])
        rows.extend(event_rows)
        for key in collected:
            collected[key].extend(event_arrays[key])
    if not rows:
        raise RuntimeError("empty split corpus")
    max_candidates = max(len(row["candidate_uids"]) for row in rows)
    count = len(rows)
    candidate_features = np.zeros((count, max_candidates, CANDIDATE_FEATURE_DIM), dtype=np.float32)
    candidate_mask = np.zeros((count, max_candidates), dtype=np.bool_)
    source_features = np.zeros((count, max_candidates, SOURCE_FEATURE_DIM), dtype=np.float32)
    padded_float_names = ("legacy_target_scores", "legacy_best_other_scores", "incumbent_target", "incumbent_other", "confidence", "presence", "motion_iou")
    padded_float: dict[str, np.ndarray] = {key: np.zeros((count, max_candidates), dtype=np.float32) for key in padded_float_names}
    padded_bool = np.zeros((count, max_candidates), dtype=np.bool_)
    for index in range(count):
        width = len(collected["candidate_features"][index])
        candidate_features[index, :width] = collected["candidate_features"][index]
        candidate_mask[index, :width] = True
        source_features[index, :width] = collected["source_features"][index]
        for key in padded_float_names:
            padded_float[key][index, :width] = collected[key][index]
        padded_bool[index, :width] = collected["protected_candidate_mask"][index]
    labels = np.asarray(
        [int(row["label_index_raw"]) if int(row["label_index_raw"]) < len(row["candidate_uids"]) else max_candidates for row in rows],
        dtype=np.int64,
    )
    arrays: dict[str, np.ndarray] = {
        "candidate_features": candidate_features,
        "candidate_mask": candidate_mask,
        "source_features": source_features,
        "human_anchor": np.stack(collected["human_anchor"], axis=0).astype(np.float32),
        "recent_trusted_memory": np.stack(collected["recent_trusted_memory"], axis=0).astype(np.float32),
        "recent_trusted_mask": np.stack(collected["recent_trusted_mask"], axis=0).astype(np.bool_),
        "long_term_trusted_memory": np.stack(collected["long_term_trusted_memory"], axis=0).astype(np.float32),
        "long_term_trusted_mask": np.stack(collected["long_term_trusted_mask"], axis=0).astype(np.bool_),
        "distractor_memory": np.stack(collected["distractor_memory"], axis=0).astype(np.float32),
        "distractor_mask": np.stack(collected["distractor_mask"], axis=0).astype(np.bool_),
        "neighbor_feature": np.stack(collected["neighbor_feature"], axis=0).astype(np.float32),
        "temporal_features": np.stack(collected["temporal_features"], axis=0).astype(np.float32),
        "labels": labels,
        "candidate_counts": np.asarray([len(row["candidate_uids"]) for row in rows], dtype=np.int64),
        **padded_float,
        "protected_candidate_mask": padded_bool,
    }
    for key, value in arrays.items():
        if value.dtype.kind not in "biu" and not np.all(np.isfinite(value)):
            raise RuntimeError(f"non-finite corpus array: {key}")
    return arrays, rows, {
        "example_count": count,
        "event_count": len({str(row["event_id"]) for row in rows}),
        "sequence_count": len({str(row["sequence"]) for row in rows}),
        "max_candidates": max_candidates,
        "label_counts": dict(sorted(Counter(str(row["label_reason"]) for row in rows).items())),
        "future_positive_count": sum(1 for row in rows if row["label_kind"] == "TARGET_CANDIDATE"),
        "candidate_source_counts": dict(sorted(Counter(source for row in rows for source in row["candidate_sources"]).items())),
    }


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-manifest", type=Path, default=BATCH_MANIFEST_PATH)
    parser.add_argument("--resource-censored", action="store_true")
    parser.add_argument(
        "--resource-audit",
        type=Path,
        default=ROOT / "outputs/N72R11R2/resource_censoring_audit.json",
    )
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--stage-path", type=Path, default=STAGE_PATH)
    args = parser.parse_args()
    batch_manifest_path = args.batch_manifest if args.batch_manifest.is_absolute() else ROOT / args.batch_manifest
    resource_audit_path = args.resource_audit if args.resource_audit.is_absolute() else ROOT / args.resource_audit
    output_root = args.output_root if args.output_root.is_absolute() else ROOT / args.output_root
    stage_path = args.stage_path if args.stage_path.is_absolute() else ROOT / args.stage_path
    resource_censored = bool(args.resource_censored)
    started = now_utc()
    base_status: dict[str, Any] = {
        "schema_version": "N72R11_STAGE_06_STATUS_V1",
        "stage": "N72R11-06-CAUSAL_CORPUS",
        "started_at_utc": started,
        "runtime_future_gt_used": False,
        "interaction_source": "simulated_from_gt",
        "real_human_evidence": False,
        "not_real_human_evidence": True,
        "resource_censored_development": resource_censored,
    }
    try:
        schedule = read_json(SCHEDULE_PATH)
        all_events = [dict(value) for value in schedule.get("events", []) if value.get("status") == "ELIGIBLE"]
        if len(all_events) != 527 or len({str(value["event_id"]) for value in all_events}) != 527:
            raise RuntimeError(f"expected 527 eligible secondary events, found {len(all_events)}")
        batch = read_json(batch_manifest_path)
        accepted_batch_statuses = {"PASS_ALL_SELECTED", "PASS_ALL_SELECTED_AFTER_RETRY"}
        batch_records = {str(value["event_id"]): value for value in batch.get("records", [])}
        if resource_censored:
            resource_audit = read_json(resource_audit_path)
            audit_counts = resource_audit.get("counts", {})
            if resource_audit.get("status") != "PASS_RESOURCE_CENSORED_DEVELOPMENT":
                raise RuntimeError(f"resource censor audit is not PASS: {resource_audit.get('status')}")
            if audit_counts.get("required") != 527 or audit_counts.get("retained_executable") != 516 or audit_counts.get("resource_censored_cuda_oom") != 11:
                raise RuntimeError(f"resource censor counts invalid: {audit_counts}")
            retained_ids = {str(row["event_id"]) for row in resource_audit.get("retained", [])}
            events = [item for item in all_events if str(item["event_id"]) in retained_ids]
            if len(events) != 516 or len(retained_ids) != 516:
                raise RuntimeError(f"resource-censored event count mismatch: {len(events)}")
            if len(batch_records) != 527:
                raise RuntimeError(f"sealed merged batch record count mismatch: {len(batch_records)}")
            batch_records = {
                event_id: value
                for event_id, value in batch_records.items()
                if str(value.get("status")) == "PASS"
            }
            if len(batch_records) != 516:
                raise RuntimeError(f"resource-censored executable batch count mismatch: {len(batch_records)}")
            accepted_batch_statuses = {"PARTIAL_WITH_FAILURES"}
        else:
            events = all_events
            if batch.get("status") not in accepted_batch_statuses or batch.get("counts") != {"PASS": 527}:
                raise RuntimeError(f"secondary batch is not complete PASS_ALL_SELECTED: {batch.get('status')} {batch.get('counts')}")
            if len(batch_records) != 527:
                raise RuntimeError(f"secondary batch record count mismatch: {len(batch_records)}")
            if any(str(value.get("status")) != "PASS" for value in batch_records.values()):
                raise RuntimeError("secondary batch contains a non-PASS record")
        if batch.get("status") not in accepted_batch_statuses:
            raise RuntimeError(f"secondary batch status is not accepted: {batch.get('status')}")
        protocol_path = PROTOCOL_PATH
        protocol = read_json(protocol_path)
        train_sequences = [str(value) for value in protocol.get("train_sequences", [])]
        validation_sequences = [str(value) for value in protocol.get("validation_sequences", [])]
        if len(train_sequences) != 12 or len(validation_sequences) != 6 or set(train_sequences) & set(validation_sequences):
            raise RuntimeError("fixed 12/6 sequence split is invalid")
        event_payloads: dict[str, list[tuple[list[dict[str, Any]], dict[str, list[np.ndarray]], dict[str, int]]]] = {"train": [], "validation": []}
        gt_by_sequence = {sequence: load_gt(sequence) for sequence in train_sequences + validation_sequences}
        event_audit: list[dict[str, Any]] = []
        for item in sorted(events, key=lambda value: str(value["event_id"])):
            sequence = str(item["sequence"])
            split = "train" if sequence in train_sequences else "validation" if sequence in validation_sequences else None
            if split is None:
                raise RuntimeError(f"event sequence is outside fixed split: {item['event_id']}:{sequence}")
            payload = build_event_runtime(item, batch_records[str(item["event_id"])])
            event_payloads[split].append(payload)
            event_audit.append({
                "event_id": str(item["event_id"]),
                "sequence": sequence,
                "split": split,
                "action_type": str(item["action_type"]),
                "runtime_future_gt_used": False,
                "event_example_count": len(payload[0]),
                "future_requery_row_count": int(payload[2]["future_requery_rows"]),
                "selected_trigger_count": int(payload[2]["selected_trigger_count"]),
            })
        split_summaries: dict[str, Any] = {}
        for split in ("train", "validation"):
            arrays, rows, summary = stack_split(event_payloads[split], gt_by_sequence)
            atomic_npz(output_root / f"{split}.npz", **arrays)
            atomic_jsonl(output_root / f"{split}_metadata.jsonl", rows)
            split_summaries[split] = {
                **summary,
                "npz": str(output_root / f"{split}.npz"),
                "npz_sha256": sha256_file(output_root / f"{split}.npz"),
                "metadata": str(output_root / f"{split}_metadata.jsonl"),
                "metadata_sha256": sha256_file(output_root / f"{split}_metadata.jsonl"),
            }
        atomic_jsonl(output_root / "event_audit.jsonl", event_audit)
        manifest = {
            "schema_version": "N72R11_CAUSAL_V3_CORPUS_V1",
            "status": "PASS_N72R11_RESOURCE_CENSORED_CAUSAL_CORPUS" if resource_censored else "PASS_N72R11_CAUSAL_CORPUS_SEALED",
            "created_at_utc": now_utc(),
            "source_schedule": str(SCHEDULE_PATH),
            "source_schedule_sha256": sha256_file(SCHEDULE_PATH),
            "source_batch_manifest": str(batch_manifest_path),
            "source_batch_manifest_sha256": sha256_file(batch_manifest_path),
            "source_protocol": str(protocol_path),
            "source_protocol_sha256": sha256_file(protocol_path),
            "event_count": len(events),
            "required_event_count": 527,
            "resource_censored_development": resource_censored,
            "resource_censored_count": 11 if resource_censored else 0,
            "sequence_count": len(set(train_sequences + validation_sequences)),
            "train_sequences": train_sequences,
            "validation_sequences": validation_sequences,
            "interaction_source": "simulated_from_gt",
            "real_human_evidence": False,
            "runtime_future_gt_used": False,
            "gt_used_only_offline_label_generation": True,
            "candidate_feature_dim": CANDIDATE_FEATURE_DIM,
            "source_feature_names": list(SOURCE_NAMES),
            "source_feature_dim": SOURCE_FEATURE_DIM,
            "memory": {"recent_trusted_slots": RECENT_SLOTS, "long_term_trusted_slots": LONG_TERM_SLOTS, "distractor_slots": DISTRACTOR_SLOTS},
            "temporal_feature_schema": list(TEMPORAL_FEATURE_SCHEMA),
            "causal_selector": "base_target_plus_anchor_similarity_geometry_presence; selected state only; no GT",
            "offline_label": "highest candidate box IoU to dataset GT >= 0.50, else NONE; attached after runtime tensors",
            "splits": split_summaries,
            "event_audit": str(output_root / "event_audit.jsonl"),
            "event_audit_sha256": sha256_file(output_root / "event_audit.jsonl"),
        }
        atomic_json(output_root / "corpus_manifest.json", manifest)
        result = {
            **base_status,
            "status": "PASS_N72R11_RESOURCE_CENSORED_CAUSAL_CORPUS" if resource_censored else "PASS_N72R11_CAUSAL_CORPUS_SEALED",
            "finished_at_utc": now_utc(),
            "corpus_manifest": str(output_root / "corpus_manifest.json"),
            "corpus_manifest_sha256": sha256_file(output_root / "corpus_manifest.json"),
            "event_count": len(events),
            "train_examples": int(split_summaries["train"]["example_count"]),
            "validation_examples": int(split_summaries["validation"]["example_count"]),
            "train_future_positive_count": int(split_summaries["train"]["future_positive_count"]),
            "validation_future_positive_count": int(split_summaries["validation"]["future_positive_count"]),
            "split_sequence_counts": {"train": 12, "validation": 6},
            "production_authorized": False,
        }
        atomic_json(stage_path, result)
        print(json.dumps(result, sort_keys=True))
        return 0
    except Exception as exc:
        failure = output_root / "attempts" / f"corpus_failure_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.json"
        failure_payload = {
            **base_status,
            "status": "FAIL_N72R11_CAUSAL_CORPUS",
            "finished_at_utc": now_utc(),
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }
        atomic_json(failure, failure_payload)
        atomic_json(stage_path, {**failure_payload, "failure_artifact": str(failure)})
        print(json.dumps({"status": failure_payload["status"], "failure_artifact": str(failure), "error": str(exc)}, sort_keys=True))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
