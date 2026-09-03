#!/usr/bin/env python3
"""N72R5 image-grounded recovery probe.

This stage is deliberately a small, isolated SAM3 worker.  It does not use
the posthoc target box to decide when to recover: the trigger list is built
from the sealed Stage01 runtime assignment audit and contains only frames at
which the target public identity was assigned to explicit NONE.  At a trigger
the worker predicts a box from the persistent state and asks the official
SAM3 adapter for a current-image segmentation.  The returned observation is
still an unauthorised candidate; only the one explicit-NONE public-ID solver
may assign it later.

The worker never loads GT.  A separate parent-side posthoc scorer is run only
after every worker artifact has passed runtime validation.  Each trigger is a
fresh Python process and a fresh official SAM3 session.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import traceback
from typing import Any, Iterable, Sequence

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sam3_intermot.association.effect_assignment import solve_effect_assignment  # noqa: E402
from sam3_intermot.backend.sam3_backend import Sam3Backend  # noqa: E402
from scripts.n72r4_official_future_pair import (  # noqa: E402
    CHECKPOINT,
    DATA_ROOT,
    FROZEN_N72R3_ROOT,
    collect_pre_event,
    correction_target,
    image_files,
    install_official_shape_audit,
    load_event,
    load_window,
    official_cached_observations,
    semantic_pre_view,
)


N72R4_ROOT = Path(
    os.environ.get(
        "N72R4_INPUT_ROOT",
        "/data2/usr_for_deadline/SAM3_InterMOT_N72R3R1/worktree/outputs/N72R4",
    )
)
STAGE01_TABLE = ROOT / "outputs/N72R5/mechanism_rounds/round_01_decision_boundary/posthoc/decision_boundary_failure_inventory.jsonl"
ROUND_ROOT = Path(
    os.environ.get(
        "N72R5_STAGE02_ROOT",
        str(ROOT / "outputs/N72R5/mechanism_rounds/round_02_image_grounded_recovery"),
    )
)
TRIGGER_MANIFEST = ROUND_ROOT / "trigger_manifest.json"
ARTIFACT_ROOT = ROUND_ROOT / "runtime"
WORKER_LOG_ROOT = ROUND_ROOT / "worker_logs"
RESULTS_PATH = ROUND_ROOT / "metrics.json"
GATE_PATH = ROUND_ROOT / "gate.json"
STAGE_STATUS = ROOT / "outputs/N72R5/stage_status/stage_02_status.json"
CONTROLLER_STATUS = ROOT / "outputs/N72R5/CONTROLLER_STATUS.json"
HUMAN_STATUS = ROOT / "outputs/N72R5/HUMAN_READABLE_STATUS.md"
MACHINE_ENCODER_CHECKPOINT = Path(
    os.environ.get(
        "N72R5_MACHINE_ENCODER_CHECKPOINT",
        "/data1/LWR/vranlee/SERVER_ONLY/avis/Interactive/SAM3_InterMOT/outputs/n9/checkpoints/osnet_x1_0_market1501.pth",
    )
)

IOU_THRESHOLD = 0.5
NONE_SCORE = 0.0
SAM_CONTEXT_WINDOW = 200
SIM_WEIGHT = 1.5
IOU_WEIGHT = 1.0
NATIVE_WEIGHT = 0.5
NATIVE_BONUS = 3.0
GAP_WEIGHT = 0.1


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"unsupported JSON value: {type(value).__name__}")


def atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False, default=json_default)
            handle.write("\n")
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


def atomic_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n")
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
                raise TypeError(f"{path}:{line_number}: expected object")
            rows.append(value)
    return rows


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def json_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    ).hexdigest()


def finite_feature(value: Any) -> np.ndarray:
    feature = np.asarray(value, dtype=np.float32).reshape(-1)
    if feature.size != 512 or not np.all(np.isfinite(feature)):
        raise ValueError(f"feature must be finite 512-D, got {feature.shape}")
    norm = float(np.linalg.norm(feature))
    if norm <= 1.0e-6:
        raise ValueError("feature norm is zero")
    return feature / norm


def box_iou(left: Any, right: Any) -> float:
    a = np.asarray(left, dtype=np.float64).reshape(-1)
    b = np.asarray(right, dtype=np.float64).reshape(-1)
    if a.size != 4 or b.size != 4 or not np.all(np.isfinite(a)) or not np.all(np.isfinite(b)):
        return 0.0
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter
    return float(inter / union) if union > 0.0 else 0.0


@dataclass
class State:
    public_id: int
    last_box: np.ndarray | None
    last_feature: np.ndarray | None
    velocity: np.ndarray
    last_frame: int
    last_native: int | None
    status: str = "ACTIVE"


def copy_state(state: State) -> State:
    return State(
        public_id=int(state.public_id),
        last_box=None if state.last_box is None else state.last_box.copy(),
        last_feature=None if state.last_feature is None else state.last_feature.copy(),
        velocity=state.velocity.copy(),
        last_frame=int(state.last_frame),
        last_native=None if state.last_native is None else int(state.last_native),
        status=str(state.status),
    )


def predicted_box(state: State, frame: int) -> np.ndarray | None:
    if state.last_box is None:
        return None
    gap = max(0, int(frame) - int(state.last_frame))
    result = state.last_box.astype(np.float64, copy=True)
    result[[0, 2]] += state.velocity[0] * gap
    result[[1, 3]] += state.velocity[1] * gap
    return result


def update_state(state: State, row: dict[str, Any], frame: int) -> None:
    new_box = np.asarray(row["box_xyxy"], dtype=np.float64).reshape(4)
    new_feature = finite_feature(row["feature"])
    old_center = (
        np.asarray([(state.last_box[0] + state.last_box[2]) / 2.0, (state.last_box[1] + state.last_box[3]) / 2.0])
        if state.last_box is not None
        else np.asarray([(new_box[0] + new_box[2]) / 2.0, (new_box[1] + new_box[3]) / 2.0])
    )
    new_center = np.asarray([(new_box[0] + new_box[2]) / 2.0, (new_box[1] + new_box[3]) / 2.0])
    delta = (new_center - old_center) / max(1, int(frame) - int(state.last_frame))
    state.velocity = 0.8 * state.velocity + 0.2 * delta
    state.last_box = new_box.copy()
    state.last_feature = new_feature.copy()
    state.last_frame = int(frame)
    state.last_native = int(row["adapter_external_id"])
    state.status = "ACTIVE"


def association_score(state: State, row: dict[str, Any], frame: int) -> float:
    feature = finite_feature(row["feature"])
    similarity = 0.0 if state.last_feature is None else float(np.dot(feature, state.last_feature))
    predicted = predicted_box(state, frame)
    geometry = 0.0 if predicted is None else box_iou(predicted, row["box_xyxy"])
    native_same = float(state.last_native is not None and int(row["adapter_external_id"]) == int(state.last_native))
    gap = min(1.0, max(0, int(frame) - int(state.last_frame)) / 200.0)
    return float(
        SIM_WEIGHT * similarity
        + IOU_WEIGHT * geometry
        + (NATIVE_WEIGHT + NATIVE_BONUS) * native_same
        - GAP_WEIGHT * gap
    )


class MachineOSNet:
    feature_dim = 512

    def __init__(self, device: str) -> None:
        if not MACHINE_ENCODER_CHECKPOINT.is_file():
            raise FileNotFoundError(MACHINE_ENCODER_CHECKPOINT)
        from PIL import Image
        from torchreid.reid.utils.feature_extractor import FeatureExtractor

        self._image = Image
        self.extractor = FeatureExtractor(
            model_name="osnet_x1_0",
            model_path=str(MACHINE_ENCODER_CHECKPOINT),
            image_size=(256, 128),
            device=device,
            verbose=False,
        )

    def encode(self, image_path: Path, boxes: Sequence[Sequence[float]]) -> np.ndarray:
        if not boxes:
            return np.zeros((0, self.feature_dim), dtype=np.float32)
        with self._image.open(image_path) as handle:
            image = handle.convert("RGB")
            crops = []
            for box in boxes:
                x1, y1, x2, y2 = [int(round(float(value))) for value in box]
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(image.width, x2), min(image.height, y2)
                crops.append(
                    np.asarray(image.crop((x1, y1, x2, y2)), dtype=np.uint8)
                    if x2 > x1 and y2 > y1
                    else np.zeros((8, 8, 3), dtype=np.uint8)
                )
        with torch.no_grad():
            values = self.extractor(crops).detach().float().cpu().numpy()
        values = np.asarray(values, dtype=np.float32).reshape(len(boxes), -1)
        if values.shape != (len(boxes), 512) or not np.all(np.isfinite(values)):
            raise RuntimeError(f"machine ROI feature invalid: {values.shape}")
        norms = np.linalg.norm(values, axis=1, keepdims=True)
        if np.any(norms <= 1.0e-6):
            raise RuntimeError("machine ROI feature norm is zero")
        return values / norms


def candidate_rows(backend: Sam3Backend, encoder: MachineOSNet, image_path: Path, frame: int, observations: list[Any]) -> list[dict[str, Any]]:
    """Use the existing official candidate exporter plus independent machine features."""
    boxes = [np.asarray(item.box_xyxy, dtype=float).copy() for item in observations]
    features = encoder.encode(image_path, boxes)
    backend._output_cache[int(frame)] = [item.copy() for item in observations]
    try:
        exported = backend.export_frame_candidates(
            int(frame), embeddings=features, include_masks=True, include_raw_provenance=True
        )
    finally:
        backend._output_cache.pop(int(frame), None)
    if len(exported) != len(observations):
        raise RuntimeError(f"official exporter changed candidate count at {frame}")
    rows: list[dict[str, Any]] = []
    seen: set[int] = set()
    for index, item in enumerate(exported):
        raw = item.get("raw_native_id")
        native_axis = int(raw) if raw is not None else int(item["native_tid"])
        if native_axis in seen:
            raise RuntimeError(f"duplicate official candidate axis {native_axis} at {frame}")
        seen.add(native_axis)
        row = dict(item)
        row["candidate_index"] = int(index)
        row["official_raw_sam_id"] = None if raw is None else int(raw)
        row["raw_native_id"] = None if raw is None else int(raw)
        row["native_tid"] = native_axis
        row["adapter_external_id"] = int(item["adapter_external_id"])
        row["feature"] = np.asarray(features[index], dtype=np.float32).tolist()
        row["feature_dim"] = 512
        row["feature_sha256"] = hashlib.sha256(np.asarray(features[index], dtype="<f4").tobytes()).hexdigest()
        row["candidate_uid"] = f"official:stage02:frame:{int(frame)}:raw:{native_axis}:index:{int(index)}"
        row["public_id"] = None
        row["authority_eligible"] = False
        rows.append(row)
    return rows


def solver_states(states: dict[int, State], public_to_state: dict[int, int]) -> tuple[list[int], list[dict[str, int]]]:
    publics = sorted(int(public) for public in states)
    if set(publics) != set(int(public) for public in public_to_state):
        raise RuntimeError("persistent public axis changed")
    values = [
        {"association_state_id": int(public_to_state[public]), "public_id": int(public)}
        for public in publics
    ]
    if len({item["association_state_id"] for item in values}) != len(values):
        raise RuntimeError("persistent state axis duplicated")
    return publics, values


def solve_rows(rows: list[dict[str, Any]], states: dict[int, State], public_to_state: dict[int, int], event_id: str, frame: int, suffix: str) -> tuple[dict[str, Any], np.ndarray]:
    publics, explicit_states = solver_states(states, public_to_state)
    matrix = np.asarray(
        [[association_score(states[public], row, frame) for row in rows] for public in publics],
        dtype=np.float64,
    )
    if not np.all(np.isfinite(matrix)):
        raise RuntimeError(f"nonfinite recovery association matrix: {event_id}/{frame}/{suffix}")
    solver = solve_effect_assignment(
        candidate_rows=rows,
        persistent_states=explicit_states,
        fused_state_candidate_scores=matrix,
        source_run_id=f"n72r5-stage02:{event_id}:{frame}:{suffix}",
        session_id=f"n72r5-stage02:{event_id}:{frame}",
        none_score=NONE_SCORE,
    )
    return solver, matrix


def assignment_map(solver: dict[str, Any]) -> dict[str, int | None]:
    return {str(item["candidate_uid"]): (None if item.get("public_id") is None else int(item["public_id"])) for item in solver.get("assignment_rows", [])}


def assigned_rows(rows: list[dict[str, Any]], solver: dict[str, Any]) -> list[dict[str, Any]]:
    mapping = assignment_map(solver)
    output: list[dict[str, Any]] = []
    for row in rows:
        item = dict(row)
        value = mapping.get(str(row["candidate_uid"]))
        item["public_id"] = value
        item["assignment_status"] = "EXPLICIT_NONE" if value is None else "ASSIGNED_TO_PUBLIC_ID"
        output.append(item)
    return output


def apply_solver_to_states(rows: list[dict[str, Any]], solver: dict[str, Any], states: dict[int, State], frame: int) -> None:
    mapping = assignment_map(solver)
    assigned = set(value for value in mapping.values() if value is not None)
    by_uid = {str(row["candidate_uid"]): row for row in rows}
    for public, state in states.items():
        uid = next((candidate_uid for candidate_uid, value in mapping.items() if value == public), None)
        if uid is None:
            state.status = "LOST"
            continue
        update_state(state, by_uid[uid], frame)


def frozen_event_row(event_id: str, event_frame: int) -> dict[str, Any]:
    path = N72R4_ROOT / "mechanism_probe/corrected_stream_attempt1" / f"{event_id}.jsonl"
    rows = [row for row in read_jsonl(path) if str(row.get("variant")) == "M0_CURRENT_FRAME_CORRECTION_ONLY" and int(row.get("frame", -1)) == event_frame]
    if len(rows) != 1:
        raise RuntimeError(f"frozen mechanism event row is not unique: {event_id}/{event_frame}")
    row = rows[0]
    if row.get("runtime_future_gt_used") is not False or row.get("runtime_gt_read") is not False:
        raise RuntimeError(f"frozen mechanism event row violates GT boundary: {event_id}")
    return row


def initialize_states(pre_rows: list[dict[str, Any]], post_rows: list[dict[str, Any]], event_row: dict[str, Any], event_frame: int) -> tuple[dict[int, State], dict[int, int]]:
    publics = [int(value) for value in event_row["public_id_order"]]
    state_axis = [int(value) for value in event_row["association_state_axis"]]
    if len(publics) != len(state_axis) or len(set(publics)) != len(publics) or len(set(state_axis)) != len(state_axis):
        raise RuntimeError("frozen event public/state axis is not one-to-one")
    states = {public: State(public, None, None, np.zeros(2, dtype=np.float64), event_frame, None) for public in publics}
    raw_to_public = {
        int(candidate["official_raw_sam_id"]): int(candidate["public_id"])
        for candidate in event_row["candidate_rows"]
        if candidate.get("public_id") is not None
    }
    pre_by_raw = {int(row["official_raw_sam_id"]): row for row in pre_rows if row.get("official_raw_sam_id") is not None}
    if set(pre_by_raw) != set(raw_to_public):
        raise RuntimeError(f"fresh Y_pre raw axis differs from frozen mapping: {sorted(set(pre_by_raw) ^ set(raw_to_public))}")
    for raw, public in raw_to_public.items():
        row = pre_by_raw[raw]
        states[public].last_box = np.asarray(row["box_xyxy"], dtype=np.float64).reshape(4)
        states[public].last_feature = finite_feature(row["feature"])
        states[public].last_native = int(row["adapter_external_id"])
    post_by_raw = {int(row["official_raw_sam_id"]): row for row in post_rows if row.get("official_raw_sam_id") is not None}
    public_to_post_raw = event_row.get("correction", {}).get("public_to_post_raw", {})
    if not isinstance(public_to_post_raw, dict):
        raise RuntimeError("frozen public_to_post_raw mapping missing")
    for public_text, raw_value in public_to_post_raw.items():
        public = int(public_text)
        raw = int(raw_value)
        if public not in states or raw not in post_by_raw:
            raise RuntimeError(f"fresh post-correction mapping cannot be resolved: {public}/{raw}")
        update_state(states[public], post_by_raw[raw], event_frame)
    public_to_state = {public: state for public, state in zip(publics, state_axis)}
    return states, public_to_state


def make_backend() -> Sam3Backend:
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
        device="cuda",
    )


def expected_prefix_hash(event_id: str) -> str | None:
    path = N72R4_ROOT / "official_corrected/full_attempt2" / f"{event_id}.done.json"
    if not path.is_file():
        return None
    return read_json(path).get("y_pre_semantic_hash")


def runtime_trigger(entry: dict[str, Any]) -> dict[str, Any]:
    event_id = str(entry["event_id"])
    event_frame = int(entry["event_frame"])
    frame = int(entry["trigger_frame"])
    event = load_event(event_id)
    if str(event["sequence"]) != str(entry["sequence"]):
        raise RuntimeError("trigger sequence disagrees with frozen event manifest")
    window = load_window(event)
    if not int(window["frame_start"]) <= event_frame <= int(window["frame_end"]):
        raise RuntimeError(f"event outside frozen window: {event_id}")
    sequence_dir = DATA_ROOT / "train" / str(event["sequence"])
    paths = image_files(sequence_dir)
    if frame >= len(paths):
        raise RuntimeError(f"trigger frame outside image sequence: {event_id}/{frame}")
    frozen_row = frozen_event_row(event_id, event_frame)
    public_order = [int(value) for value in frozen_row["public_id_order"]]
    target_public = int(entry["target_public_id"])
    if target_public not in public_order:
        raise RuntimeError(f"trigger target public not in frozen axis: {event_id}/{target_public}")
    backend: Sam3Backend | None = None
    encoder: MachineOSNet | None = None
    started = now_utc()
    started_clock = time.time()
    try:
        backend = make_backend()
        session_id = backend.start_video(str(sequence_dir / "img1"))
        install_official_shape_audit(backend)
        pre = collect_pre_event(backend, window, event_frame)
        if not pre:
            raise RuntimeError("fresh official Y_pre is empty")
        pre_hash = json_hash(semantic_pre_view(pre))
        frozen_hash = expected_prefix_hash(event_id)
        if frozen_hash is not None and pre_hash != frozen_hash:
            raise RuntimeError(f"PREFIX_HASH_MISMATCH expected={frozen_hash} actual={pre_hash}")
        backend.reconcile_official_tracker_to_visible_outputs(event_frame)
        action, target_sam, human_box = correction_target(event)
        for observation in pre:
            backend.register_detected_observation(observation)
        if action == "RECOVER_IDENTITY":
            existing = [int(key) for key in getattr(backend, "_objects", {})]
            prompt_target = max(existing, default=0) + 1000
            correction = backend.add_box(event_frame, prompt_target, human_box)
            correction_route = "official_backend.add_box"
        else:
            prompt_target = int(target_sam)
            if prompt_target not in getattr(backend, "_objects", {}):
                raise RuntimeError(f"correction target absent from fresh Y_pre: {event_id}/{prompt_target}")
            correction = backend.correct_object(event_frame, prompt_target, box_xyxy=human_box)
            correction_route = "official_backend.correct_object"
        official_post = official_cached_observations(backend, event_frame)
        encoder = MachineOSNet("cuda:0")
        pre_rows = candidate_rows(backend, encoder, paths[event_frame], event_frame, [item.copy() for item in pre])
        post_rows = candidate_rows(backend, encoder, paths[event_frame], event_frame, [item.copy() for item in official_post])
        states, public_to_state = initialize_states(pre_rows, post_rows, frozen_row, event_frame)
        # The pinned multiplex planner does not support repeatedly stopping
        # and reopening propagation at every frame: after an early stop it can
        # enter a cache-fetch path with a missing detector frame.  Use one
        # continuous official stream through the frozen trigger frame, then
        # perform the current-image recovery prompt at that boundary.
        outputs = backend.propagate(
            event_frame,
            frame,
            start_frame_index=event_frame,
            max_frame_num_to_track=SAM_CONTEXT_WINDOW,
            keep_masks=True,
            cache_outputs=True,
        )
        expected_frames = set(range(event_frame, frame + 1))
        observed_frames = {int(value) for value in outputs}
        if observed_frames != expected_frames:
            raise RuntimeError(
                f"fresh official prefix-to-trigger coverage mismatch: "
                f"missing={sorted(expected_frames - observed_frames)[:8]} "
                f"extra={sorted(observed_frames - expected_frames)[:8]}"
            )
        for current in range(event_frame + 1, frame):
            observations = outputs.get(current, backend.get_frame_outputs(current)) or []
            rows = candidate_rows(backend, encoder, paths[current], current, [item.copy() for item in observations])
            states_before = {public: copy_state(value) for public, value in states.items()}
            solver, _ = solve_rows(rows, states_before, public_to_state, event_id, current, "prefix")
            apply_solver_to_states(rows, solver, states, current)
        current = frame
        observations = outputs.get(current, backend.get_frame_outputs(current)) or []
        baseline_rows = candidate_rows(backend, encoder, paths[current], current, [item.copy() for item in observations])
        states_before = {public: copy_state(value) for public, value in states.items()}
        baseline_solver, baseline_matrix = solve_rows(baseline_rows, states_before, public_to_state, event_id, current, "R0")
        all_future: dict[int, dict[str, Any]] = {}
        recovery_candidate_uid = None
        recovery_box = None
        prompt_observation_count = 0
        if True:
            baseline_map = assignment_map(baseline_solver)
            if target_public in set(value for value in baseline_map.values() if value is not None):
                trigger_observed = False
                recovery_status = "NOT_TRIGGERED_TARGET_ASSIGNED_IN_FRESH_RUNTIME"
                treatment_rows = baseline_rows
                treatment_solver = baseline_solver
                recovery_candidate_uid = None
                recovery_box = None
            else:
                trigger_observed = True
                target_state = states_before[target_public]
                recovery_box_array = predicted_box(target_state, current)
                recovery_box = None if recovery_box_array is None else [float(value) for value in recovery_box_array]
                if recovery_box_array is None:
                    recovery_status = "TRIGGERED_NO_PERSISTENT_BOX"
                    treatment_rows = baseline_rows
                    treatment_solver = baseline_solver
                    recovery_candidate_uid = None
                else:
                    recovery_object_id = 1_000_000 + int(current)
                    recovered = backend.seed_box_from_past_state(current, recovery_object_id, recovery_box_array)
                    if recovered is None:
                        recovery_status = "TRIGGERED_OFFICIAL_SAM3_RETURNED_NO_RECOVERY_OBSERVATION"
                        treatment_rows = baseline_rows
                        treatment_solver = baseline_solver
                        recovery_candidate_uid = None
                    else:
                        prompt_outputs = official_cached_observations(backend, current)
                        prompt_observation_count = len(prompt_outputs)
                        prompt_rows = candidate_rows(backend, encoder, paths[current], current, [item.copy() for item in prompt_outputs])
                        recovered_raw = recovered.raw_sam_object_id
                        matches = [row for row in prompt_rows if recovered_raw is not None and row.get("official_raw_sam_id") == int(recovered_raw)]
                        if len(matches) != 1:
                            raise RuntimeError(f"recovery observation/raw mapping is not unique: {event_id}/{current}")
                        recovery_row = dict(matches[0])
                        recovery_row.update(
                            {
                                "candidate_kind": "IMAGE_GROUNDED_RECOVERY_CANDIDATE",
                                "authority_eligible": False,
                                "recovery_source": "official_sam3_seed_box_from_persistent_prediction_current_image",
                                "search_region": {
                                    "type": "predicted_box",
                                    "scale": 1.0,
                                    "box_xyxy": recovery_box,
                                },
                                "public_id": None,
                            }
                        )
                        # The official add-prompt response is allowed to expose
                        # only the newly seeded object.  It is not a complete
                        # replacement candidate set.  Retain the complete
                        # pre-prompt current-image official stream and append
                        # exactly the new image-grounded candidate; this keeps
                        # untouched candidates available to the global solver.
                        treatment_rows = [dict(row) for row in baseline_rows]
                        for index, row in enumerate(treatment_rows):
                            row.setdefault("candidate_kind", "OFFICIAL_SAM3_CANDIDATE")
                            row.setdefault("authority_eligible", False)
                            row["candidate_index"] = int(index)
                        recovery_row["candidate_index"] = len(treatment_rows)
                        recovery_row["candidate_uid"] = f"recovery:stage02:{event_id}:frame:{current}:raw:{recovery_row.get('official_raw_sam_id')}"
                        recovery_candidate_uid = str(recovery_row["candidate_uid"])
                        treatment_rows.append(recovery_row)
                        treatment_solver, treatment_matrix = solve_rows(treatment_rows, states_before, public_to_state, event_id, current, "R1")
                        recovery_status = "IMAGE_GROUNDED_RECOVERY_CANDIDATE_RETURNED_BASELINE_CANDIDATES_RETAINED"
            baseline_output = assigned_rows(baseline_rows, baseline_solver)
            treatment_output = assigned_rows(treatment_rows, treatment_solver)
            recovery_assignment = None if recovery_candidate_uid is None else assignment_map(treatment_solver).get(recovery_candidate_uid)
            all_future[current] = {
                "frame": int(current),
                "frame_horizon": int(current - event_frame),
                "trigger_expected_from_stage01_runtime": True,
                "trigger_observed_in_fresh_runtime": bool(trigger_observed),
                "recovery_status": recovery_status,
                "predicted_search_box": recovery_box,
                "recovery_candidate_uid": recovery_candidate_uid,
                "recovery_candidate_assigned_public_id": recovery_assignment,
                "recovery_prompt_observation_count": int(prompt_observation_count) if trigger_observed else 0,
                "treatment_candidate_stream_complete": True,
                "baseline_candidates_retained_in_treatment": bool(recovery_candidate_uid is not None),
                "baseline_solver": baseline_solver,
                "treatment_solver": treatment_solver,
                "baseline_score_matrix": baseline_matrix.astype(float).tolist(),
                "baseline_candidate_rows": baseline_output,
                "treatment_candidate_rows": treatment_output,
                "causal_boundary": {
                    "event_frame_memory_read": False,
                    "recovery_current_frame_memory_read": False,
                    "recovery_first_visible_frame": int(current),
                    "runtime_future_gt_used": False,
                },
                "runtime_future_gt_used": False,
                "runtime_gt_read": False,
                "posthoc_gt_used": False,
            }
        if not all_future:
            raise RuntimeError("target trigger frame was not reached")
        frame_record = all_future[frame]
        result = {
            "schema_version": "N72R5_STAGE02_IMAGE_GROUNDED_RECOVERY_WORKER_V1",
            "status": "PASS_RUNTIME_IMAGE_GROUNDED_RECOVERY_PROBE",
            "event_id": event_id,
            "sequence": str(event["sequence"]),
            "action_type": str(event["action_type"]),
            "event_frame": event_frame,
            "trigger_frame": frame,
            "frame_horizon": frame - event_frame,
            "target_public_id": target_public,
            "session_id": str(session_id),
            "checkpoint_sha256": sha256_file(CHECKPOINT),
            "machine_encoder_checkpoint_sha256": sha256_file(MACHINE_ENCODER_CHECKPOINT),
            "y_pre_semantic_hash": pre_hash,
            "frozen_y_pre_semantic_hash": frozen_hash,
            "y_pre_candidate_count": len(pre_rows),
            "post_correction_candidate_count": len(post_rows),
            "correction": {
                "route": correction_route,
                "prompt_target_id": int(prompt_target),
                "frozen_target_sam_id": int(target_sam),
                "human_box_used_after_y_pre_freeze": np.asarray(human_box, dtype=float).tolist(),
                "official_observation_box": np.asarray(correction.box_xyxy, dtype=float).tolist(),
                "event_frame_memory_read": False,
                "first_future_frame": event_frame + 1,
                "runtime_future_gt_used": False,
            },
            "future": frame_record,
            "runtime_memory_policy": backend.runtime_memory_policy(),
            "official_shape_audit": list(getattr(getattr(backend, "_predictor", None).model, "_n72r4_shape_audit", [])),
            "runtime_future_gt_used": False,
            "runtime_gt_read": False,
            "posthoc_gt_used": False,
            "interaction_source": "simulated_from_gt",
            "not_real_human_evidence": True,
            "public_id_inference": False,
            "elapsed_sec": time.time() - started_clock,
            "started_at_utc": started,
            "finished_at_utc": now_utc(),
        }
        output = Path(os.environ["N72R5_STAGE02_WORKER_OUTPUT"]).resolve()
        atomic_json(output, result)
        atomic_json(output.with_suffix(".done.json"), {"status": result["status"], "artifact": str(output), "artifact_sha256": sha256_file(output), "runtime_future_gt_used": False})
        return result
    finally:
        if backend is not None:
            try:
                backend.close()
            except Exception:
                pass
        encoder = None
        backend = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def load_trigger_manifest() -> dict[str, Any]:
    if not STAGE01_TABLE.is_file():
        raise FileNotFoundError(STAGE01_TABLE)
    rows = read_jsonl(STAGE01_TABLE)
    triggers: list[dict[str, Any]] = []
    for row in rows:
        # This selection uses only the sealed runtime assignment result.  GT
        # fields in the Stage01 table are deliberately ignored here.
        if row.get("baseline_target_candidate_uid") is not None:
            continue
        if row.get("runtime_future_gt_used") is not False or row.get("posthoc_gt_used") is not True:
            raise RuntimeError(f"Stage01 provenance invalid for trigger source: {row.get('event_id')}/{row.get('frame')}")
        triggers.append(
            {
                "event_id": str(row["event_id"]),
                "sequence": str(row["sequence"]),
                "action_type": str(row["action_type"]),
                "event_frame": int(row["event_frame"]),
                "trigger_frame": int(row["frame"]),
                "frame_horizon": int(row["frame_horizon"]),
                "target_public_id": int(row["target_public_id"]),
                "trigger_rule": "exact_solver_target_public_id_unassigned_explicit_NONE_or_no_candidate",
                "gt_used_for_trigger": False,
                "future_metric_used_for_trigger": False,
                "source_runtime_record_sha256": json_hash(row),
            }
        )
    keys = [(item["event_id"], item["trigger_frame"]) for item in triggers]
    if len(keys) != len(set(keys)):
        raise RuntimeError("duplicate image recovery trigger keys")
    return {
        "schema_version": "N72R5_STAGE02_TRIGGER_MANIFEST_V1",
        "status": "PASS_STAGE01_RUNTIME_TRIGGER_FREEZE",
        "stage": "02_IMAGE_GROUNDED_RECOVERY",
        "trigger_count": len(triggers),
        "triggers": sorted(triggers, key=lambda item: (item["event_id"], item["trigger_frame"])),
        "selection_inputs": {
            "source_stage01_table": str(STAGE01_TABLE),
            "source_stage01_sha256": sha256_file(STAGE01_TABLE),
            "ignored_posthoc_fields": ["target_gt_present", "correct_candidate_uid", "target_gt_box_posthoc"],
        },
        "rule": "recover only when exact public-ID+NONE runtime solve leaves target public unassigned; no unconditional per-frame query",
        "candidate_public_authority": "none_before_exact_solver",
        "runtime_future_gt_used": False,
        "interaction_source": "simulated_from_gt",
        "not_real_human_evidence": True,
        "created_at_utc": now_utc(),
    }


def validate_worker_artifact(path: Path, trigger: dict[str, Any]) -> dict[str, Any]:
    result = read_json(path)
    errors: list[str] = []
    for key, expected in (("event_id", trigger["event_id"]), ("trigger_frame", trigger["trigger_frame"]), ("target_public_id", trigger["target_public_id"])):
        if result.get(key) != expected:
            errors.append(f"{key}_mismatch")
    if result.get("status") != "PASS_RUNTIME_IMAGE_GROUNDED_RECOVERY_PROBE":
        errors.append("worker_status_not_pass")
    if result.get("runtime_future_gt_used") is not False or result.get("runtime_gt_read") is not False or result.get("posthoc_gt_used") is not False:
        errors.append("runtime_gt_boundary_invalid")
    future = result.get("future")
    if not isinstance(future, dict):
        errors.append("future_record_missing")
    else:
        if future.get("runtime_future_gt_used") is not False or future.get("runtime_gt_read") is not False or future.get("posthoc_gt_used") is not False:
            errors.append("future_gt_boundary_invalid")
        if future.get("causal_boundary", {}).get("event_frame_memory_read") is not False:
            errors.append("event_frame_memory_read_invalid")
        if future.get("treatment_candidate_stream_complete") is not True:
            errors.append("treatment_candidate_stream_incomplete")
        recovery_uid = future.get("recovery_candidate_uid")
        if recovery_uid is not None:
            rows = future.get("treatment_candidate_rows", [])
            matches = [row for row in rows if str(row.get("candidate_uid")) == str(recovery_uid)]
            if len(matches) != 1:
                errors.append("recovery_candidate_uid_not_unique")
            elif matches[0].get("authority_eligible") is not False:
                errors.append("recovery_candidate_has_authority")
            if future.get("baseline_candidates_retained_in_treatment") is not True:
                errors.append("baseline_candidates_not_retained")
            if len({str(row.get("candidate_uid")) for row in rows}) != len(rows):
                errors.append("treatment_candidate_uid_duplicate")
            raw_axis = [row.get("official_raw_sam_id") for row in rows if row.get("official_raw_sam_id") is not None]
            if len(raw_axis) != len(set(int(value) for value in raw_axis)):
                errors.append("treatment_raw_candidate_axis_duplicate")
            if int(future.get("recovery_prompt_observation_count", 0)) < 1:
                errors.append("official_recovery_prompt_observation_missing")
    return {"path": str(path), "event_id": trigger["event_id"], "trigger_frame": trigger["trigger_frame"], "status": "PASS" if not errors else "FAIL", "errors": errors, "runtime_future_gt_used": False}


def posthoc_iou_scoring(trigger_manifest: dict[str, Any], validations: list[dict[str, Any]]) -> dict[str, Any]:
    """Score only after worker runtime artifacts have been validated."""
    def load_gt(sequence: str) -> dict[int, dict[int, list[float]]]:
        path = DATA_ROOT / "train" / sequence / "gt" / "gt.txt"
        result: dict[int, dict[int, list[float]]] = defaultdict(dict)
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                values = [item.strip() for item in line.split(",")]
                frame = int(values[0]) - 1
                identity = int(values[1])
                x, y, width, height = [float(item) for item in values[2:6]]
                result[frame][identity] = [x, y, x + width, y + height]
        return result

    output: list[dict[str, Any]] = []
    for trigger, validation in zip(trigger_manifest["triggers"], validations):
        artifact = read_json(Path(validation["path"]))
        target_gt_id = int(load_event(str(trigger["event_id"]))["dataset_gt_id"])
        gt_box = load_gt(str(trigger["sequence"])).get(int(trigger["trigger_frame"]), {}).get(target_gt_id)
        future = artifact["future"]
        baseline_rows = future.get("baseline_candidate_rows", [])
        treatment_rows = future.get("treatment_candidate_rows", [])
        recovery_rows = [row for row in treatment_rows if row.get("candidate_kind") == "IMAGE_GROUNDED_RECOVERY_CANDIDATE"]
        baseline_best = max((box_iou(row["box_xyxy"], gt_box) for row in baseline_rows), default=0.0) if gt_box is not None else None
        treatment_best = max((box_iou(row["box_xyxy"], gt_box) for row in treatment_rows), default=0.0) if gt_box is not None else None
        recovery_best = max((box_iou(row["box_xyxy"], gt_box) for row in recovery_rows), default=0.0) if gt_box is not None else None
        output.append(
            {
                "event_id": trigger["event_id"],
                "sequence": trigger["sequence"],
                "action_type": trigger["action_type"],
                "frame": trigger["trigger_frame"],
                "target_gt_id_posthoc": target_gt_id,
                "target_gt_present_posthoc": gt_box is not None,
                "baseline_best_candidate_iou": baseline_best,
                "treatment_best_candidate_iou": treatment_best,
                "recovery_candidate_best_iou": recovery_best,
                "baseline_candidate_recalled": baseline_best is not None and baseline_best >= IOU_THRESHOLD,
                "recovery_candidate_recalled": recovery_best is not None and recovery_best >= IOU_THRESHOLD,
                "recovery_candidate_assigned_public_id": future.get("recovery_candidate_assigned_public_id"),
                "trigger_observed_in_fresh_runtime": future.get("trigger_observed_in_fresh_runtime"),
                "runtime_future_gt_used": False,
                "posthoc_gt_used": True,
            }
        )
    return {
        "schema_version": "N72R5_STAGE02_RECOVERY_POSTHOC_RESULTS_V1",
        "status": "PASS_STAGE02_POSTHOC_RECOVERY_RECALL_AUDIT",
        "trigger_count": len(output),
        "trigger_results": output,
        "candidate_recovery_gain_count": sum(int(item["recovery_candidate_recalled"] and not item["baseline_candidate_recalled"]) for item in output),
        "candidate_recovery_recall_count": sum(int(item["recovery_candidate_recalled"]) for item in output),
        "baseline_candidate_recall_count": sum(int(item["baseline_candidate_recalled"]) for item in output),
        "runtime_future_gt_used": False,
        "posthoc_gt_used": True,
        "interaction_source": "simulated_from_gt",
        "not_real_human_evidence": True,
    }


def worker_main(args: argparse.Namespace) -> int:
    trigger = next(item for item in load_trigger_manifest()["triggers"] if item["event_id"] == args.event_id and item["trigger_frame"] == args.frame)
    output = args.output.resolve()
    os.environ["N72R5_STAGE02_WORKER_OUTPUT"] = str(output)
    try:
        result = runtime_trigger(trigger)
        print(json.dumps({"status": result["status"], "event_id": args.event_id, "frame": args.frame, "output": str(output)}, sort_keys=True), flush=True)
        return 0
    except Exception as exc:
        failure = output.with_suffix(".failure.json")
        atomic_json(
            failure,
            {
                "schema_version": "N72R5_STAGE02_FAILURE_V1",
                "stage": "02_IMAGE_GROUNDED_RECOVERY",
                "status": "FAIL_PRESERVED",
                "event_id": args.event_id,
                "trigger_frame": int(args.frame),
                "error_type": type(exc).__name__,
                "error": str(exc),
                "traceback": traceback.format_exc(),
                "runtime_future_gt_used": False,
                "scientific_result": "NO_SCIENTIFIC_RESULT",
            },
        )
        print(json.dumps({"status": "FAIL", "event_id": args.event_id, "frame": args.frame, "failure": str(failure), "error": str(exc)}, sort_keys=True), flush=True)
        return 1


def orchestrator_main(args: argparse.Namespace) -> int:
    started = now_utc()
    if ROUND_ROOT.exists() and any(ROUND_ROOT.iterdir()):
        raise RuntimeError(f"refusing to overwrite non-empty Stage02 round root: {ROUND_ROOT}")
    ROUND_ROOT.mkdir(parents=True, exist_ok=True)
    manifest = load_trigger_manifest()
    atomic_json(TRIGGER_MANIFEST, manifest)
    if not manifest["triggers"]:
        raise RuntimeError("no runtime target-NONE triggers were frozen")
    if args.only:
        selected = [
            item
            for item in manifest["triggers"]
            if item["event_id"] == args.only
            and (args.trigger_frame is None or item["trigger_frame"] == int(args.trigger_frame))
        ]
    else:
        selected = manifest["triggers"]
    if not selected:
        raise RuntimeError(f"requested smoke event not in trigger manifest: {args.only}")
    records: list[dict[str, Any]] = []
    for trigger in selected:
        output = ARTIFACT_ROOT / f"{trigger['event_id']}.frame{trigger['trigger_frame']}.json"
        log = WORKER_LOG_ROOT / f"{trigger['event_id']}.frame{trigger['trigger_frame']}.log"
        log.parent.mkdir(parents=True, exist_ok=True)
        environment = os.environ.copy()
        environment["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
        environment["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
        command = [str(Path(sys.executable)), str(Path(__file__).resolve()), "--worker", "--event-id", trigger["event_id"], "--frame", str(trigger["trigger_frame"]), "--output", str(output)]
        with log.open("w", encoding="utf-8") as handle:
            completed = subprocess.run(command, cwd=str(ROOT), env=environment, stdout=handle, stderr=subprocess.STDOUT, text=True, check=False)
        done = output.with_suffix(".done.json")
        failure = output.with_suffix(".failure.json")
        if done.is_file():
            payload = read_json(done)
        elif failure.is_file():
            payload = read_json(failure)
        else:
            payload = {"status": "MISSING_ARTIFACT"}
        records.append({"event_id": trigger["event_id"], "trigger_frame": trigger["trigger_frame"], "return_code": int(completed.returncode), "output": str(output), "log": str(log), "done": str(done) if done.is_file() else None, "failure": str(failure) if failure.is_file() else None, "status": payload.get("status"), "runtime_future_gt_used": False})
        if completed.returncode != 0:
            break
    selected_validations = []
    for trigger in selected[: len(records)]:
        record = next(item for item in records if item["event_id"] == trigger["event_id"] and item["trigger_frame"] == trigger["trigger_frame"])
        selected_validations.append(validate_worker_artifact(Path(record["output"]), trigger) if Path(record["output"]).is_file() else {"status": "FAIL", "path": record["output"], "errors": ["artifact_missing"]})
    all_runtime_pass = len(records) == len(selected) and all(item["return_code"] == 0 and item["status"] == "PASS_RUNTIME_IMAGE_GROUNDED_RECOVERY_PROBE" for item in records) and all(item["status"] == "PASS" for item in selected_validations)
    result = {
        "schema_version": "N72R5_STAGE02_RECOVERY_RUNTIME_RESULTS_V1",
        "status": "PASS_STAGE02_RUNTIME_TARGETED_SMOKE" if all_runtime_pass and len(selected) < len(manifest["triggers"]) else ("PASS_STAGE02_RUNTIME_FULL_TRIGGER_SET" if all_runtime_pass else "BLOCKED_STAGE02_RUNTIME_RECOVERY"),
        "execution_scope": "targeted_smoke" if len(selected) < len(manifest["triggers"]) else "full_runtime_trigger_set",
        "started_at_utc": started,
        "finished_at_utc": now_utc(),
        "trigger_manifest": str(TRIGGER_MANIFEST),
        "trigger_manifest_sha256": sha256_file(TRIGGER_MANIFEST),
        "worker_records": records,
        "runtime_validations": selected_validations,
        "runtime_future_gt_used": False,
        "posthoc_gt_used": False,
        "official_sam3_runtime": True,
        "candidate_public_authority_before_solver": False,
        "scientific_result": "RUNTIME_RECOVERY_CANDIDATE_PROBE_ONLY_NO_EFFECT_CLAIM",
    }
    atomic_json(RESULTS_PATH.with_name("runtime_results.json"), result)
    if not all_runtime_pass:
        atomic_json(GATE_PATH, {"status": "BLOCKED_STAGE02_RUNTIME_RECOVERY", "runtime_valid": False, "runtime_results": str(RESULTS_PATH.with_name("runtime_results.json")), "runtime_future_gt_used": False})
        atomic_json(STAGE_STATUS, {"schema_version": "N72R5_STAGE_STATUS_V1", "stage": "02_IMAGE_GROUNDED_RECOVERY", "status": "BLOCKED_STAGE02_RUNTIME_RECOVERY", "runtime_results": str(RESULTS_PATH.with_name("runtime_results.json")), "failed_workers": [item for item in records if item["return_code"] != 0], "runtime_future_gt_used": False, "posthoc_gt_used": False})
        return 1
    if len(selected) < len(manifest["triggers"]):
        atomic_json(GATE_PATH, {"status": "PASS_STAGE02_TARGETED_SMOKE_ONLY", "runtime_valid": True, "full_trigger_set_authorized": False, "remaining_trigger_count": len(manifest["triggers"]) - len(selected), "runtime_results": str(RESULTS_PATH.with_name("runtime_results.json")), "runtime_future_gt_used": False})
        print(json.dumps({"status": "PASS_STAGE02_TARGETED_SMOKE_ONLY", "runtime_results": str(RESULTS_PATH.with_name("runtime_results.json"))}, sort_keys=True), flush=True)
        return 0
    posthoc = posthoc_iou_scoring(manifest, selected_validations)
    final = {**result, "posthoc": posthoc, "posthoc_gt_used": True, "status": "PASS_STAGE02_COMPLETE_CANDIDATE_RECALL_AUDIT"}
    atomic_json(RESULTS_PATH, final)
    gate = {
        "schema_version": "N72R5_STAGE02_GATE_V1",
        "status": "PASS_STAGE02_ROUTE_TO_TVC" if posthoc["candidate_recovery_gain_count"] > 0 else "PASS_STAGE02_RECOVERY_NO_CANDIDATE_RECALL_GAIN_ROUTE_TO_TVC",
        "runtime_valid": True,
        "trigger_count": len(manifest["triggers"]),
        "candidate_recovery_gain_count": posthoc["candidate_recovery_gain_count"],
        "candidate_recovery_recall_count": posthoc["candidate_recovery_recall_count"],
        "baseline_candidate_recall_count": posthoc["baseline_candidate_recall_count"],
        "future_effect_gate": "NOT_EVALUATED_STAGE02",
        "tvc_authorized": True,
        "training_authorized": False,
        "production_authorized": False,
        "runtime_future_gt_used": False,
    }
    atomic_json(GATE_PATH, gate)
    atomic_json(STAGE_STATUS, {"schema_version": "N72R5_STAGE_STATUS_V1", "stage": "02_IMAGE_GROUNDED_RECOVERY", "status": gate["status"], "runtime_results": str(RESULTS_PATH), "gate": str(GATE_PATH), "trigger_count": len(manifest["triggers"]), "candidate_recovery_gain_count": posthoc["candidate_recovery_gain_count"], "runtime_future_gt_used": False, "posthoc_gt_used": True, "interaction_source": "simulated_from_gt", "not_real_human_evidence": True, "production_authorized": False})
    print(json.dumps({"status": gate["status"], "results": str(RESULTS_PATH), "gate": str(GATE_PATH)}, sort_keys=True), flush=True)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--event-id")
    parser.add_argument("--frame", type=int)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--only")
    parser.add_argument("--trigger-frame", type=int)
    parser.add_argument("--gpu", type=int, default=1)
    args = parser.parse_args()
    if args.worker:
        if args.event_id is None or args.frame is None or args.output is None:
            raise SystemExit("--worker requires --event-id, --frame and --output")
        if not torch.cuda.is_available():
            raise RuntimeError("image-grounded recovery worker requires CUDA")
        return worker_main(args)
    if not torch.cuda.is_available():
        raise RuntimeError("image-grounded recovery requires CUDA")
    return orchestrator_main(args)


if __name__ == "__main__":
    raise SystemExit(main())
