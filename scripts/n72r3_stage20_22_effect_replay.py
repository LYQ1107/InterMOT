#!/usr/bin/env python3
"""N72R3 Stage 20--22 target-scoped effect replay and strict gate.

The official SAM3 current-frame correction and real 512-D ROI write were
already executed in Stage 16--17.  This script replays the frozen Candidate
V2 stream after that boundary on CPU.  Runtime never opens a GT file.  GT is
loaded only by ``posthoc_score`` after the runtime artifacts have been
validated and atomically sealed.

The replay is deliberately explicit rather than a second production
associator:

* ``M0`` is the current-frame correction-only control;
* M1--M4 differ only in a target-public-ID appearance row;
* every non-target score row is copied bit-for-bit from the same base matrix;
* one global Hungarian solve is retained;
* unmatched future candidates use the frozen outer-birth decision, while
  current-frame NO_INTERVENTION may retain an explicit NONE assignment.

This is an exploratory, simulated_from_gt mechanism experiment.  It is not
evidence of a real human tape and it never authorizes production learning.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import sys
import tempfile
import traceback
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from scipy.optimize import linear_sum_assignment

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sam3_intermot.association.appearance_memory import AppearanceMemory  # noqa: E402
from scripts.n72r3_stage09_11_candidate_runtime import load_plan, load_source_rows  # noqa: E402


OUT = ROOT / "outputs/N72R3"
EVENT_MANIFEST = OUT / "simulation/real_event_manifest.json"
OFFICIAL_ROOT = OUT / "official_correction/events"
PROTOCOL = OUT / "protocol.json"
BASELINE_STATUS = OUT / "stage_18_status.json"
DEFAULT_REPLAY_ROOT = OUT / "effect_replay/attempt1"
REPLAY_ROOT = Path(os.environ.get("N72R3_EFFECT_REPLAY_ROOT", str(DEFAULT_REPLAY_ROOT)))
ARTIFACT_ROOT = REPLAY_ROOT / "runtime_event_artifacts"
RUNTIME_MANIFEST = REPLAY_ROOT / "runtime_manifest.json"
STAGE20_STATUS = OUT / "stage_20_status.json"
STAGE21_STATUS = OUT / "stage_21_status.json"
RESULT_PATH = REPLAY_ROOT / "ccam_paired_replay_results.json"
STAGE22_STATUS = OUT / "stage_22_status.json"
FAILURE_ROOT = OUT / "attempts"

HORIZONS = (20, 50, 100)
VARIANTS = (
    "NO_INTERVENTION",
    "M0_CURRENT_FRAME_CORRECTION_ONLY",
    "M1_HUMAN_EMA_PROTOTYPE",
    "M2_POSITIVE_HUMAN_ANCHORS",
    "M3_NEGATIVE_COMPETITOR_BANK",
    "M4_RELIABILITY_AGE_ADMISSION",
)
MEMORY_VARIANTS = set(VARIANTS[2:])
PRIMARY_VARIANT = "M2_POSITIVE_HUMAN_ANCHORS"
IOU_THRESHOLD = 0.5
BOOTSTRAP_REPETITIONS = 2000
BOOTSTRAP_SEED = 7202
MAX_EVENT_TARGET = 40
MIN_SEQUENCE_TARGET = 20

# These are the already-frozen pairwise scorer constants used by the online
# associator's non-learned re-identification path.  The native term is the
# sum of its 0.5 pairwise cue and 3.0 continuity bonus.
REID_SIM_WEIGHT = 1.5
REID_IOU_WEIGHT = 1.0
REID_NATIVE_WEIGHT = 0.5
REID_GAP_WEIGHT = 0.1
REID_NATIVE_BONUS = 3.0
STATE_EMA = 0.9
MAX_MEMORY_AGE = 80
MIN_MEMORY_RELIABILITY = 0.75


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
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
                raise TypeError(f"{path}:{line_number}: expected JSON object")
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
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")
    ).hexdigest()


def feature_hash(value: np.ndarray) -> str:
    array = np.asarray(value, dtype="<f4").reshape(-1)
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def unit(value: Any) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32).reshape(-1)
    if array.size != 512 or not np.all(np.isfinite(array)):
        raise ValueError(f"feature must be finite 512-D, got {array.shape}")
    norm = float(np.linalg.norm(array))
    if norm <= 1.0e-6:
        raise ValueError("feature has zero norm")
    return array / norm


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


def predicted_iou(state: "ReplayIdentity", box: np.ndarray, frame: int) -> float:
    if state.last_box is None:
        return 0.0
    gap = max(0, int(frame) - int(state.last_seen_frame))
    predicted = state.last_box.astype(np.float64, copy=True)
    if gap > 0:
        predicted[[0, 2]] += state.velocity[0] * gap
        predicted[[1, 3]] += state.velocity[1] * gap
    return box_iou(predicted, box)


def candidate_public_id(row: dict[str, Any], public_id: int | None, status: str) -> dict[str, Any]:
    return {
        "candidate_index": int(row["candidate_index"]),
        "candidate_uid": str(row["candidate_uid"]),
        "official_raw_sam_id": int(row["official_raw_sam_id"]),
        "adapter_external_id": int(row["adapter_external_id"]),
        "box_xyxy": [float(value) for value in row["box_xyxy"]],
        "confidence": float(row["confidence"]),
        "feature_sha256": str(row["feature_sha256"]),
        "feature_dim": int(row["feature_dim"]),
        "public_id": None if public_id is None else int(public_id),
        "assignment_status": str(status),
    }


@dataclass
class ReplayIdentity:
    public_id: int
    lineage_id: int
    feature: np.ndarray
    last_box: np.ndarray
    velocity: np.ndarray
    last_seen_frame: int
    last_native: int
    status: str = "ACTIVE"
    lost_age: int = 0
    current_candidate_uid: str | None = None

    def clone(self) -> "ReplayIdentity":
        return ReplayIdentity(
            public_id=int(self.public_id),
            lineage_id=int(self.lineage_id),
            feature=self.feature.copy(),
            last_box=self.last_box.copy(),
            velocity=self.velocity.copy(),
            last_seen_frame=int(self.last_seen_frame),
            last_native=int(self.last_native),
            status=str(self.status),
            lost_age=int(self.lost_age),
            current_candidate_uid=self.current_candidate_uid,
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "public_id": int(self.public_id),
            "mot_track_id": int(self.public_id),
            "lineage_id": int(self.lineage_id),
            "status": str(self.status),
            "last_seen_frame": int(self.last_seen_frame),
            "last_native": int(self.last_native),
            "lost_age": int(self.lost_age),
            "current_candidate_uid": self.current_candidate_uid,
        }


def update_identity(state: ReplayIdentity, row: dict[str, Any], frame: int) -> None:
    new_feature = unit(row["feature"])
    dt = max(1, int(frame) - int(state.last_seen_frame))
    old_center = np.asarray(
        [(state.last_box[0] + state.last_box[2]) / 2.0, (state.last_box[1] + state.last_box[3]) / 2.0],
        dtype=np.float64,
    )
    new_box = np.asarray(row["box_xyxy"], dtype=np.float64).reshape(4)
    new_center = np.asarray([(new_box[0] + new_box[2]) / 2.0, (new_box[1] + new_box[3]) / 2.0], dtype=np.float64)
    state.velocity = 0.8 * state.velocity + 0.2 * (new_center - old_center) / dt
    state.feature = unit(STATE_EMA * state.feature + (1.0 - STATE_EMA) * new_feature)
    state.last_box = new_box.copy()
    state.last_seen_frame = int(frame)
    state.last_native = int(row["adapter_external_id"])
    state.status = "ACTIVE"
    state.lost_age = 0
    state.current_candidate_uid = str(row["candidate_uid"])


def score_matrix(states: list[ReplayIdentity], rows: list[dict[str, Any]], frame: int) -> np.ndarray:
    scores = np.zeros((len(states), len(rows)), dtype=np.float64)
    for state_index, state in enumerate(states):
        for candidate_index, row in enumerate(rows):
            feature = unit(row["feature"])
            similarity = float(np.dot(feature, state.feature))
            geometry = predicted_iou(state, np.asarray(row["box_xyxy"], dtype=np.float64), frame)
            gap = min(1.0, max(0, int(frame) - int(state.last_seen_frame)) / 200.0)
            native_same = 1.0 if int(row["adapter_external_id"]) == int(state.last_native) else 0.0
            scores[state_index, candidate_index] = (
                REID_SIM_WEIGHT * similarity
                + REID_IOU_WEIGHT * geometry
                + REID_NATIVE_WEIGHT * native_same
                + REID_NATIVE_BONUS * native_same
                - REID_GAP_WEIGHT * gap
            )
    return scores


def parse_native_from_live_uid(value: str) -> int:
    marker = ":sam:"
    if marker not in value:
        raise ValueError(f"official simulated prediction UID has no SAM suffix: {value}")
    return int(value.rsplit(marker, 1)[1])


def load_scenarios() -> list[dict[str, Any]]:
    baseline = read_json(BASELINE_STATUS)
    if not str(baseline.get("status", "")).startswith("PASS"):
        raise RuntimeError("Stage 20 requires the passing Stage 18 structural baseline")
    manifest = read_json(EVENT_MANIFEST)
    if manifest.get("status") != "PASS_STAGE14_POLICY_FROZEN":
        raise RuntimeError("Stage 20 requires the frozen Stage 14 policy")
    events = [dict(item) for item in manifest.get("events", [])]
    if not events:
        raise RuntimeError("Stage 14 contains no eligible events")
    protocol = read_json(PROTOCOL)
    if protocol.get("runtime_causal_contract", {}).get("future_gt_runtime_reads") is not False:
        raise RuntimeError("N72R3 protocol does not freeze the future-GT runtime prohibition")
    plan_by_window = {str(item["window_id"]): dict(item) for item in load_plan()}
    scenarios: list[dict[str, Any]] = []
    for event in sorted(events, key=lambda value: str(value["event_id"])):
        event_id = str(event["event_id"])
        official_path = OFFICIAL_ROOT / f"{event_id}.json"
        official = read_json(official_path)
        if official.get("status") != "PASS_STAGE16_OFFICIAL_CORRECTION_AND_STAGE17_MEMORY":
            raise RuntimeError(f"official Stage16/17 artifact is not PASS: {event_id}")
        if official.get("runtime_future_gt_used") is not False or official.get("future_read_executed") is not False:
            raise RuntimeError(f"official artifact violates runtime future-GT boundary: {event_id}")
        causal = official.get("causal_audit", {})
        if causal.get("event_frame_read") is not False or causal.get("current_frame_write_hidden") is not True:
            raise RuntimeError(f"official causal boundary is invalid: {event_id}")
        setup = official.get("simulated_assignment_setup", {})
        target_pid = int(official["persistent_identity"]["public_id"])
        target_native = int(setup["target_sam_object_id"])
        wrong_pid = None if setup.get("wrong_public_id") is None else int(setup["wrong_public_id"])
        memory_payload = official.get("appearance_memory", {})
        positives = memory_payload.get("positive", [])
        if len(positives) != 1:
            raise RuntimeError(f"expected one Stage17 human anchor: {event_id}")
        human_feature = unit(positives[0].get("feature"))
        declared_feature_hash = causal.get("memory_write", {}).get("feature_sha256")
        if declared_feature_hash and declared_feature_hash != feature_hash(human_feature):
            raise RuntimeError(f"Stage17 human feature digest mismatch: {event_id}")
        window_id = str(event["current_candidate_v2"]["window_id"])
        window = plan_by_window[window_id]
        rows, source_meta = load_source_rows(window)
        by_frame: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            by_frame[int(row["frame_idx"])].append(row)
        for frame_rows in by_frame.values():
            frame_rows.sort(key=lambda value: (int(value["candidate_index"]), str(value["candidate_uid"])))
        event_frame = int(event["event_frame"])
        expected = list(range(event_frame, event_frame + 101))
        missing = [frame for frame in expected if frame not in by_frame]
        if missing:
            raise RuntimeError(f"frozen candidate stream misses event/future frames for {event_id}: {missing[:8]}")
        event_rows = by_frame[event_frame]
        target_rows = [row for row in event_rows if int(row["adapter_external_id"]) == target_native]
        if len(target_rows) != 1:
            raise RuntimeError(f"target native candidate is not unique at event frame: {event_id}")
        predictions = {}
        for prediction in official.get("simulated_predictions", []):
            native = parse_native_from_live_uid(str(prediction["candidate_uid"]))
            public_id = int(prediction["public_id"])
            if native in predictions and predictions[native] != public_id:
                raise RuntimeError(f"conflicting simulated current mapping: {event_id}/{native}")
            predictions[native] = public_id
        initial_public_by_native = {int(row["adapter_external_id"]): 1000 + index for index, row in enumerate(event_rows)}
        if target_pid not in initial_public_by_native.values():
            raise RuntimeError(f"target public ID is outside the frozen outer birth axis: {event_id}")
        pre_map = dict(predictions)
        if wrong_pid is not None:
            pre_map[target_native] = wrong_pid
        if len(set(pre_map.values())) != len(pre_map):
            raise RuntimeError(f"duplicate pre-intervention public mapping: {event_id}")
        scenarios.append(
            {
                "event": event,
                "official": official,
                "event_id": event_id,
                "sequence": str(event["sequence"]),
                "event_frame": event_frame,
                "action_type": str(event["action_type"]),
                "target_public_id": target_pid,
                "target_native": target_native,
                "wrong_public_id": wrong_pid,
                "human_feature": human_feature,
                "source_meta": source_meta,
                "rows_by_frame": by_frame,
                "initial_public_by_native": initial_public_by_native,
                "pre_map": pre_map,
                "candidate_sha256": source_meta["candidate_sha256"],
                "candidate_frame_sha256": source_meta["candidate_frame_sha256"],
                "official_sha256": sha256_file(official_path),
            }
        )
    return scenarios


def initial_states(scenario: dict[str, Any]) -> list[ReplayIdentity]:
    event_frame = int(scenario["event_frame"])
    event_rows = scenario["rows_by_frame"][event_frame]
    states: list[ReplayIdentity] = []
    for index, row in enumerate(event_rows):
        native = int(row["adapter_external_id"])
        public_id = int(scenario["initial_public_by_native"][native])
        states.append(
            ReplayIdentity(
                public_id=public_id,
                lineage_id=index + 1,
                feature=unit(row["feature"]),
                last_box=np.asarray(row["box_xyxy"], dtype=np.float64).reshape(4),
                velocity=np.zeros(2, dtype=np.float64),
                last_seen_frame=event_frame,
                last_native=native,
                status="ACTIVE" if public_id in scenario["pre_map"].values() else "LOST",
                lost_age=0 if public_id in scenario["pre_map"].values() else 1,
                current_candidate_uid=(str(row["candidate_uid"]) if scenario["pre_map"].get(native) == public_id else None),
            )
        )
    if len({state.public_id for state in states}) != len(states):
        raise RuntimeError(f"duplicate initial public IDs: {scenario['event_id']}")
    return states


def build_memory(
    scenario: dict[str, Any],
    variant: str,
    target_row: dict[str, Any],
) -> AppearanceMemory | None:
    if variant not in MEMORY_VARIANTS:
        return None
    memory = AppearanceMemory(
        human_weight=1.0,
        machine_weight=0.35,
        decay_frames=120.0,
        reliability_threshold=0.0,
    )
    target_pid = int(scenario["target_public_id"])
    event_frame = int(scenario["event_frame"])
    if not memory.update_from_machine(target_pid, event_frame, unit(target_row["feature"]), confidence=1.0):
        raise RuntimeError(f"could not seed target machine prototype: {scenario['event_id']}")
    competitors = [
        unit(row["feature"])
        for row in scenario["rows_by_frame"][event_frame]
        if int(row["adapter_external_id"]) != int(scenario["target_native"])
    ]
    ok = memory.update_from_human(
        target_pid,
        event_frame,
        scenario["human_feature"],
        quality=1.0,
        competing_embeddings=(competitors if variant in {"M3_NEGATIVE_COMPETITOR_BANK", "M4_RELIABILITY_AGE_ADMISSION"} else None),
        write_event_id=str(scenario["event_id"]),
    )
    if not ok:
        raise RuntimeError(f"target human memory write failed: {scenario['event_id']}/{variant}")
    return memory


def appearance_components(
    scenario: dict[str, Any],
    memory: AppearanceMemory | None,
    variant: str,
    public_id: int,
    feature: np.ndarray,
    frame: int,
) -> tuple[dict[str, float], bool, str]:
    empty = {"prototype": 0.0, "positive": 0.0, "negative": 0.0, "total": 0.0}
    if memory is None or public_id != int(scenario["target_public_id"]):
        return empty, False, "NO_TARGET_MEMORY"
    age = int(frame) - int(scenario["event_frame"])
    if frame <= int(scenario["event_frame"]):
        return empty, False, "EVENT_FRAME_READ_FORBIDDEN"
    if variant == "M4_RELIABILITY_AGE_ADMISSION":
        record = memory.records.get(int(public_id))
        reliability = 0.0 if record is None else float(record.reliability)
        if reliability < MIN_MEMORY_RELIABILITY:
            return empty, False, "RELIABILITY_BELOW_ADMISSION"
        if age > MAX_MEMORY_AGE:
            return empty, False, "AGE_ABOVE_ADMISSION"
    components = memory._score_components(int(public_id), feature, int(frame))
    if variant == "M1_HUMAN_EMA_PROTOTYPE":
        total = components["prototype"]
    elif variant == "M2_POSITIVE_HUMAN_ANCHORS":
        total = components["prototype"] + components["positive"]
    elif variant in {"M3_NEGATIVE_COMPETITOR_BANK", "M4_RELIABILITY_AGE_ADMISSION"}:
        total = components["prototype"] + components["positive"] + components["negative"]
    else:
        raise ValueError(f"unknown memory variant: {variant}")
    result = {key: float(value) for key, value in components.items()}
    result["total"] = float(total)
    return result, True, "ADMITTED"


def current_mapping_after_event(scenario: dict[str, Any], variant: str) -> dict[int, int]:
    current = {int(native): int(public_id) for native, public_id in scenario["pre_map"].items()}
    if variant in VARIANTS[1:]:
        target_pid = int(scenario["target_public_id"])
        wrong_pid = scenario["wrong_public_id"]
        current = {
            native: public_id
            for native, public_id in current.items()
            if public_id != target_pid and (wrong_pid is None or public_id != int(wrong_pid))
        }
        current[int(scenario["target_native"])] = target_pid
    return current


def update_states_for_event(
    scenario: dict[str, Any],
    states: list[ReplayIdentity],
    variant: str,
) -> tuple[dict[int, int], dict[str, Any]]:
    event_frame = int(scenario["event_frame"])
    event_rows = scenario["rows_by_frame"][event_frame]
    row_by_native = {int(row["adapter_external_id"]): row for row in event_rows}
    current = current_mapping_after_event(scenario, variant)
    by_public = {int(state.public_id): state for state in states}
    for state in states:
        state.status = "LOST"
        state.lost_age = max(1, int(state.lost_age))
        state.current_candidate_uid = None
    for native, public_id in sorted(current.items()):
        state = by_public.get(int(public_id))
        row = row_by_native.get(int(native))
        if state is None or row is None:
            raise RuntimeError(f"event mapping references unknown state/candidate: {scenario['event_id']}/{native}/{public_id}")
        update_identity(state, row, event_frame)
    if variant in VARIANTS[1:]:
        target_native = int(scenario["target_native"])
        if current.get(target_native) != int(scenario["target_public_id"]):
            raise RuntimeError(f"current correction did not bind target public ID: {scenario['event_id']}/{variant}")
    return current, {
        "event_frame": event_frame,
        "event_frame_memory_read": False,
        "current_frame_write_hidden": True,
        "first_memory_visible_frame": event_frame + 1,
        "correction_applied": variant in VARIANTS[1:],
        "target_public_id": int(scenario["target_public_id"]),
        "runtime_future_gt_used": False,
    }


def event_row_artifact(
    scenario: dict[str, Any],
    variant: str,
    states: list[ReplayIdentity],
    current: dict[int, int],
    memory: AppearanceMemory | None,
    boundary: dict[str, Any],
) -> dict[str, Any]:
    event_frame = int(scenario["event_frame"])
    rows = scenario["rows_by_frame"][event_frame]
    state_axis = sorted(states, key=lambda state: state.public_id)
    base = score_matrix(state_axis, rows, event_frame)
    assignment_public: list[int | None] = []
    assignment_status: list[str] = []
    for row in rows:
        public_id = current.get(int(row["adapter_external_id"]))
        assignment_public.append(None if public_id is None else int(public_id))
        assignment_status.append("EXPLICIT_NONE_NO_INTERVENTION" if public_id is None else "CURRENT_FRAME_CORRECTION_BINDING" if variant in VARIANTS[1:] else "PRE_INTERVENTION_BINDING")
    memory_write = variant in MEMORY_VARIANTS
    return {
        "schema_version": "N72R3_EFFECT_RUNTIME_FRAME_V1",
        "event_id": scenario["event_id"],
        "sequence": scenario["sequence"],
        "action_type": scenario["action_type"],
        "event_frame": event_frame,
        "frame": event_frame,
        "frame_horizon": 0,
        "phase": "CURRENT_FRAME_SETUP",
        "variant": variant,
        "candidate_rows": [candidate_public_id(row, current.get(int(row["adapter_external_id"])), status) for row, status in zip(rows, assignment_status)],
        "public_id_order": [int(state.public_id) for state in state_axis],
        "base_score_matrix": base.astype(float).tolist(),
        "appearance_score_matrix": np.zeros_like(base).astype(float).tolist(),
        "appearance_score_deltas": np.zeros_like(base).astype(float).tolist(),
        "fused_score_matrix": base.astype(float).tolist(),
        "target_public_id": int(scenario["target_public_id"]),
        "target_row_index": next((index for index, state in enumerate(state_axis) if state.public_id == int(scenario["target_public_id"])), None),
        "assignment_public_ids": assignment_public,
        "assignment_status": assignment_status,
        "solver_executed": False,
        "memory_write": memory_write,
        "memory_read": False,
        "memory_admitted": False,
        "memory_read_reason": "EVENT_FRAME_READ_FORBIDDEN",
        "causal_boundary": boundary,
        "public_state_axis_after_frame": [state.as_dict() for state in sorted(states, key=lambda state: state.public_id)],
        "runtime_future_gt_used": False,
        "runtime_gt_read": False,
        "posthoc_gt_used": False,
        "target_scoped_non_target_rows_bitwise_equal": True,
        "solver_coupled_collateral": False,
    }


def future_row_artifact(
    scenario: dict[str, Any],
    variant: str,
    states: list[ReplayIdentity],
    rows: list[dict[str, Any]],
    frame: int,
    memory: AppearanceMemory | None,
    next_public_id: int,
) -> tuple[dict[str, Any], int]:
    solver_states = sorted(states, key=lambda state: state.public_id)
    base = score_matrix(solver_states, rows, frame)
    appearance = np.zeros_like(base, dtype=np.float64)
    components_by_candidate: list[dict[str, Any]] = []
    target_index = next((index for index, state in enumerate(solver_states) if state.public_id == int(scenario["target_public_id"])), None)
    target_admitted = False
    target_reason = "NO_TARGET_STATE"
    if target_index is not None and variant in MEMORY_VARIANTS:
        for candidate_index, row in enumerate(rows):
            components, admitted, reason = appearance_components(
                scenario,
                memory,
                variant,
                int(solver_states[target_index].public_id),
                unit(row["feature"]),
                frame,
            )
            appearance[target_index, candidate_index] = float(components["total"])
            components_by_candidate.append(
                {
                    "candidate_index": int(row["candidate_index"]),
                    "candidate_uid": str(row["candidate_uid"]),
                    "components": components,
                    "admitted": bool(admitted),
                    "reason": reason,
                }
            )
            target_admitted = target_admitted or admitted
            target_reason = reason if not admitted else "ADMITTED"
    else:
        components_by_candidate = [
            {
                "candidate_index": int(row["candidate_index"]),
                "candidate_uid": str(row["candidate_uid"]),
                "components": {"prototype": 0.0, "positive": 0.0, "negative": 0.0, "total": 0.0},
                "admitted": False,
                "reason": target_reason,
            }
            for row in rows
        ]
    fused = base.copy()
    if target_index is not None:
        fused[target_index, :] = base[target_index, :] + appearance[target_index, :]
    non_target_rows = [index for index in range(base.shape[0]) if index != target_index]
    non_target_equal = bool(np.array_equal(base[non_target_rows, :], fused[non_target_rows, :])) if non_target_rows else True
    if not non_target_equal:
        raise RuntimeError(f"target-scoped non-target row changed: {scenario['event_id']}/{variant}/{frame}")
    assignment_state_by_candidate = [-1] * len(rows)
    matched_state_indices: set[int] = set()
    if base.shape[0] and base.shape[1]:
        solver_rows, solver_columns = linear_sum_assignment(-fused)
        for state_index, candidate_index in zip(solver_rows.tolist(), solver_columns.tolist()):
            if float(fused[state_index, candidate_index]) >= 0.0:
                assignment_state_by_candidate[int(candidate_index)] = int(state_index)
                matched_state_indices.add(int(state_index))
    candidate_public: list[int | None] = [None] * len(rows)
    candidate_status: list[str] = ["EXPLICIT_NONE_NO_VALID_ASSIGNMENT"] * len(rows)
    for candidate_index, state_index in enumerate(assignment_state_by_candidate):
        if state_index >= 0:
            candidate_public[candidate_index] = int(solver_states[state_index].public_id)
            candidate_status[candidate_index] = "ASSIGNED_EXISTING_IDENTITY"
    for state_index, state in enumerate(solver_states):
        if state_index not in matched_state_indices:
            state.status = "LOST"
            state.lost_age = max(1, int(state.lost_age) + 1)
            state.current_candidate_uid = None
    for candidate_index, state_index in enumerate(assignment_state_by_candidate):
        if state_index >= 0:
            update_identity(solver_states[state_index], rows[candidate_index], frame)
    for candidate_index, row in enumerate(rows):
        if candidate_public[candidate_index] is not None:
            continue
        public_id = int(next_public_id)
        next_public_id += 1
        birth = ReplayIdentity(
            public_id=public_id,
            lineage_id=public_id - 999,
            feature=unit(row["feature"]),
            last_box=np.asarray(row["box_xyxy"], dtype=np.float64).reshape(4),
            velocity=np.zeros(2, dtype=np.float64),
            last_seen_frame=int(frame),
            last_native=int(row["adapter_external_id"]),
            status="ACTIVE",
            lost_age=0,
            current_candidate_uid=str(row["candidate_uid"]),
        )
        states.append(birth)
        candidate_public[candidate_index] = public_id
        candidate_status[candidate_index] = "OUTER_BIRTH_ASSIGNED"
    # The solver state objects were sorted views of the same objects.  The
    # target identity therefore remains in ``states`` even when it is LOST.
    target_state_present = any(state.public_id == int(scenario["target_public_id"]) for state in states)
    if not target_state_present:
        raise RuntimeError(f"persistent target public state disappeared: {scenario['event_id']}/{variant}/{frame}")
    assignment_map = {str(row["candidate_uid"]): candidate_public[index] for index, row in enumerate(rows)}
    return {
        "schema_version": "N72R3_EFFECT_RUNTIME_FRAME_V1",
        "event_id": scenario["event_id"],
        "sequence": scenario["sequence"],
        "action_type": scenario["action_type"],
        "event_frame": int(scenario["event_frame"]),
        "frame": int(frame),
        "frame_horizon": int(frame) - int(scenario["event_frame"]),
        "phase": "FUTURE_ASSOCIATION",
        "variant": variant,
        "candidate_rows": [candidate_public_id(row, candidate_public[index], candidate_status[index]) for index, row in enumerate(rows)],
        "public_id_order": [int(state.public_id) for state in solver_states],
        "base_score_matrix": base.astype(float).tolist(),
        "appearance_score_matrix": appearance.astype(float).tolist(),
        "appearance_score_deltas": appearance.astype(float).tolist(),
        "fused_score_matrix": fused.astype(float).tolist(),
        "target_public_id": int(scenario["target_public_id"]),
        "target_row_index": target_index,
        "assignment_state_indices": [int(value) for value in assignment_state_by_candidate],
        "assignment_public_ids": [None if value is None else int(value) for value in candidate_public],
        "assignment_status": candidate_status,
        "assignment_map": assignment_map,
        "solver_executed": True,
        "solver": "scipy.optimize.linear_sum_assignment_max_weight_with_score_threshold_0_and_outer_birth",
        "memory_write": False,
        "memory_read": bool(target_index is not None and variant in MEMORY_VARIANTS),
        "memory_admitted": bool(target_admitted),
        "memory_read_reason": target_reason,
        "memory_components_by_candidate": components_by_candidate,
        "memory_age": int(frame) - int(scenario["event_frame"]),
        "causal_boundary": {
            "event_frame_memory_read": False,
            "current_frame_write_hidden": True,
            "first_memory_visible_frame": int(scenario["event_frame"]) + 1,
            "memory_read_frame": int(frame) if variant in MEMORY_VARIANTS else None,
            "runtime_future_gt_used": False,
        },
        "public_state_axis_after_frame": [state.as_dict() for state in sorted(states, key=lambda state: state.public_id)],
        "runtime_future_gt_used": False,
        "runtime_gt_read": False,
        "posthoc_gt_used": False,
        "target_scoped_non_target_rows_bitwise_equal": bool(non_target_equal),
        "solver_coupled_collateral": False,
    }, next_public_id


def run_runtime(scenarios: list[dict[str, Any]]) -> dict[str, Any]:
    if REPLAY_ROOT.exists() and any(REPLAY_ROOT.iterdir()):
        raise RuntimeError(f"replay root is not empty; choose a new N72R3_EFFECT_REPLAY_ROOT: {REPLAY_ROOT}")
    ARTIFACT_ROOT.mkdir(parents=True, exist_ok=True)
    completed: list[dict[str, Any]] = []
    for event_number, scenario in enumerate(scenarios, 1):
        event_id = str(scenario["event_id"])
        event_artifacts: list[dict[str, Any]] = []
        for variant in VARIANTS:
            states = initial_states(scenario)
            target_row = next(
                row for row in scenario["rows_by_frame"][int(scenario["event_frame"])]
                if int(row["adapter_external_id"]) == int(scenario["target_native"])
            )
            memory = build_memory(scenario, variant, target_row)
            current, boundary = update_states_for_event(scenario, states, variant)
            event_artifacts.append(event_row_artifact(scenario, variant, states, current, memory, boundary))
            next_public_id = max(int(state.public_id) for state in states) + 1
            for frame in range(int(scenario["event_frame"]) + 1, int(scenario["event_frame"]) + 101):
                rows = scenario["rows_by_frame"].get(frame, [])
                if not rows:
                    raise RuntimeError(f"future candidate frame is empty: {event_id}/{frame}")
                frame_artifact, next_public_id = future_row_artifact(scenario, variant, states, rows, frame, memory, next_public_id)
                event_artifacts.append(frame_artifact)
        event_artifacts.sort(key=lambda row: (int(row["frame"]), VARIANTS.index(str(row["variant"]))))
        artifact_path = ARTIFACT_ROOT / f"{event_id}.jsonl"
        atomic_jsonl(artifact_path, event_artifacts)
        completed.append(
            {
                "event_id": event_id,
                "sequence": scenario["sequence"],
                "action_type": scenario["action_type"],
                "artifact": str(artifact_path),
                "artifact_sha256": sha256_file(artifact_path),
                "frame_count": 101,
                "variant_count": len(VARIANTS),
                "variant_frame_count": 101 * len(VARIANTS),
                "runtime_future_gt_used": False,
            }
        )
        atomic_json(
            RUNTIME_MANIFEST,
            {
                "schema_version": "N72R3_EFFECT_RUNTIME_MANIFEST_V1",
                "status": "IN_PROGRESS" if event_number < len(scenarios) else "PASS_RUNTIME_ARTIFACTS",
                "created_at_utc": now_utc(),
                "runtime_root": str(REPLAY_ROOT),
                "artifact_root": str(ARTIFACT_ROOT),
                "completed_event_count": event_number,
                "expected_event_count": len(scenarios),
                "completed": completed,
                "runtime_future_gt_used": False,
                "gt_loaded_in_worker": False,
                "interaction_source": "simulated_from_gt",
                "real_human_tape": False,
            },
        )
        print(json.dumps({"events_completed": event_number, "events_total": len(scenarios)}, sort_keys=True), flush=True)
    return read_json(RUNTIME_MANIFEST)


def validate_runtime(scenarios: list[dict[str, Any]]) -> dict[str, Any]:
    files = sorted(ARTIFACT_ROOT.glob("*.jsonl"))
    expected = {f"{scenario['event_id']}.jsonl" for scenario in scenarios}
    if {path.name for path in files} != expected:
        raise RuntimeError(f"runtime artifact set mismatch: expected={len(expected)}, found={len(files)}")
    checked_frames = 0
    checked_variants = 0
    non_target_cells = 0
    target_scope_failures: list[str] = []
    for scenario in scenarios:
        event_id = str(scenario["event_id"])
        path = ARTIFACT_ROOT / f"{event_id}.jsonl"
        rows = read_jsonl(path)
        if len(rows) != 101 * len(VARIANTS):
            raise RuntimeError(f"runtime row count mismatch: {event_id} {len(rows)}")
        by_key = {(str(row.get("variant")), int(row.get("frame", -1))): row for row in rows}
        if len(by_key) != len(rows):
            raise RuntimeError(f"duplicate runtime event/variant/frame key: {event_id}")
        for frame in range(int(scenario["event_frame"]), int(scenario["event_frame"]) + 101):
            frame_keys = []
            for variant in VARIANTS:
                key = (variant, frame)
                if key not in by_key:
                    raise RuntimeError(f"missing runtime key: {event_id}/{variant}/{frame}")
                value = by_key[key]
                if value.get("event_id") != event_id or value.get("runtime_future_gt_used") is not False or value.get("runtime_gt_read") is not False or value.get("posthoc_gt_used") is not False:
                    raise RuntimeError(f"runtime causal flag failed: {event_id}/{variant}/{frame}")
                if "dataset_gt_id" in value or "gt_box" in value or "future_gt" in value:
                    raise RuntimeError(f"GT field was sent into runtime artifact: {event_id}/{variant}/{frame}")
                candidates = value.get("candidate_rows", [])
                uids = [str(item.get("candidate_uid")) for item in candidates]
                if len(uids) != len(set(uids)) or not uids:
                    raise RuntimeError(f"candidate rows are duplicate/empty: {event_id}/{variant}/{frame}")
                if len(value.get("assignment_public_ids", [])) != len(candidates):
                    raise RuntimeError(f"candidate assignment axis mismatch: {event_id}/{variant}/{frame}")
                if any(item.get("public_id") != public_id for item, public_id in zip(candidates, value["assignment_public_ids"])):
                    raise RuntimeError(f"candidate/public mapping mismatch: {event_id}/{variant}/{frame}")
                frame_keys.append(uids)
                base = np.asarray(value.get("base_score_matrix", []), dtype=np.float64)
                fused = np.asarray(value.get("fused_score_matrix", []), dtype=np.float64)
                appearance = np.asarray(value.get("appearance_score_deltas", []), dtype=np.float64)
                if base.shape != fused.shape or appearance.shape != base.shape or not np.all(np.isfinite(base)) or not np.all(np.isfinite(fused)) or not np.all(np.isfinite(appearance)):
                    raise RuntimeError(f"score matrix shape/finite check failed: {event_id}/{variant}/{frame}")
                target_index = value.get("target_row_index")
                non_target = [index for index in range(base.shape[0]) if index != target_index]
                if non_target and not np.array_equal(base[non_target, :], fused[non_target, :]):
                    target_scope_failures.append(f"{event_id}/{variant}/{frame}")
                non_target_cells += int(base[non_target, :].size) if non_target else 0
                if value.get("target_scoped_non_target_rows_bitwise_equal") is not True:
                    target_scope_failures.append(f"declared:{event_id}/{variant}/{frame}")
                if frame == int(scenario["event_frame"]):
                    if value.get("phase") != "CURRENT_FRAME_SETUP" or value.get("memory_read") is not False or value.get("causal_boundary", {}).get("event_frame_memory_read") is not False:
                        raise RuntimeError(f"event-frame causal boundary failed: {event_id}/{variant}")
                else:
                    if value.get("phase") != "FUTURE_ASSOCIATION" or value.get("frame_horizon") != frame - int(scenario["event_frame"]):
                        raise RuntimeError(f"future frame/horizon failed: {event_id}/{variant}/{frame}")
                    if value.get("causal_boundary", {}).get("first_memory_visible_frame") != int(scenario["event_frame"]) + 1:
                        raise RuntimeError(f"memory boundary declaration failed: {event_id}/{variant}/{frame}")
                frame_keys.append(uids)
                checked_variants += 1
            if len({tuple(value) for value in frame_keys}) != 1:
                raise RuntimeError(f"paired variants changed the candidate stream: {event_id}/{frame}")
            checked_frames += 1
    if target_scope_failures:
        raise RuntimeError(f"target-scoped rows changed at {target_scope_failures[:5]}")
    audit = {
        "schema_version": "N72R3_STAGE21_TARGET_SCOPED_AUDIT_V1",
        "status": "PASS_STAGE21_TARGET_SCOPED_ASSOCIATION_AUDIT",
        "event_count": len(scenarios),
        "independent_sequence_count": len({scenario["sequence"] for scenario in scenarios}),
        "frame_count_including_event": checked_frames,
        "variant_frame_count_including_event": checked_variants,
        "non_target_score_cells_checked_bitwise": non_target_cells,
        "target_scoped_row_failures": 0,
        "candidate_stream_shared_across_variants": True,
        "global_hungarian_retained": True,
        "solver_coupled_collateral_labeled": True,
        "runtime_future_gt_used": False,
        "gt_loaded_in_worker": False,
        "interaction_source": "simulated_from_gt",
        "real_human_tape": False,
    }
    atomic_json(STAGE21_STATUS, {
        "schema_version": "N72R3_STAGE_STATUS_V1",
        "stage": "21_TARGET_SCOPED_ASSOCIATION",
        **audit,
        "runtime_root": str(REPLAY_ROOT),
        "runtime_manifest": str(RUNTIME_MANIFEST),
        "scientific_result": "TARGET_SCOPED_MECHANISM_AUDIT_NOT_FUTURE_EFFECT",
    })
    return audit


def load_gt(sequence: str) -> dict[int, dict[int, dict[str, Any]]]:
    path = Path("/data1/LWR/vranlee/DATASETS/JDE/dancetrack") / "train" / sequence / "gt/gt.txt"
    if not path.is_file():
        raise FileNotFoundError(path)
    result: dict[int, dict[int, dict[str, Any]]] = defaultdict(dict)
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            parts = [item.strip() for item in line.split(",")]
            if len(parts) < 6:
                raise ValueError(f"malformed GT row {path}:{line_number}")
            frame_one_based, dataset_id = int(parts[0]), int(parts[1])
            x, y, width, height = [float(item) for item in parts[2:6]]
            result[frame_one_based - 1][dataset_id] = {
                "box": [x, y, x + width, y + height],
                "visibility": None if len(parts) <= 8 else float(parts[8]),
            }
    return result


def row_map(frame_row: dict[str, Any]) -> dict[str, int | None]:
    return {str(item["candidate_uid"]): item.get("public_id") for item in frame_row.get("candidate_rows", [])}


def best_assigned_box(frame_row: dict[str, Any], public_id: int) -> tuple[list[float] | None, float]:
    candidates = [item for item in frame_row.get("candidate_rows", []) if item.get("public_id") is not None and int(item["public_id"]) == int(public_id)]
    if not candidates:
        return None, 0.0
    # This is only used with a GT box by posthoc_score; returning the largest
    # candidate is deterministic until the caller computes IoU.
    return list(candidates[0]["box_xyxy"]), 0.0


def candidate_best_iou(frame_row: dict[str, Any], gt_box: list[float]) -> tuple[float, dict[str, Any] | None]:
    values = [(box_iou(item["box_xyxy"], gt_box), item) for item in frame_row.get("candidate_rows", [])]
    return max(values, key=lambda pair: (pair[0], -int(pair[1]["candidate_index"])), default=(0.0, None))


def public_target_iou(frame_row: dict[str, Any], public_id: int, gt_box: list[float]) -> float:
    return max(
        (box_iou(item["box_xyxy"], gt_box) for item in frame_row.get("candidate_rows", []) if item.get("public_id") is not None and int(item["public_id"]) == int(public_id)),
        default=0.0,
    )


def public_box_for_gt(frame_row: dict[str, Any], public_id: int, gt_box: list[float]) -> tuple[float, int | None]:
    candidates = [item for item in frame_row.get("candidate_rows", []) if item.get("public_id") is not None and int(item["public_id"]) == int(public_id)]
    if not candidates:
        return 0.0, None
    scores = [(box_iou(item["box_xyxy"], gt_box), int(item["candidate_index"])) for item in candidates]
    return max(scores, key=lambda value: (value[0], -value[1]))


def bootstrap(values_by_sequence: dict[str, float], seed_offset: int) -> dict[str, Any]:
    names = sorted(values_by_sequence)
    if not names:
        return {
            "lower": None,
            "upper": None,
            "mean": None,
            "seed": BOOTSTRAP_SEED + seed_offset,
            "repetitions": BOOTSTRAP_REPETITIONS,
            "clusters": 0,
            "unit": "independent_sequence",
        }
    values = np.asarray([values_by_sequence[name] for name in names], dtype=np.float64)
    rng = np.random.default_rng(BOOTSTRAP_SEED + seed_offset)
    samples = np.empty(BOOTSTRAP_REPETITIONS, dtype=np.float64)
    for index in range(BOOTSTRAP_REPETITIONS):
        samples[index] = float(np.mean(values[rng.integers(0, len(values), size=len(values))]))
    return {
        "lower": float(np.quantile(samples, 0.025)),
        "upper": float(np.quantile(samples, 0.975)),
        "mean": float(np.mean(values)),
        "seed": BOOTSTRAP_SEED + seed_offset,
        "repetitions": BOOTSTRAP_REPETITIONS,
        "clusters": len(names),
        "unit": "independent_sequence",
    }


def metric_template() -> dict[str, Any]:
    return {
        "evaluated_frames": 0,
        "target_gt_present_frames": 0,
        "target_iou_sum": 0.0,
        "target_correct_frames": 0,
        "target_missing_frames": 0,
        "target_identity_error_frames": 0,
        "wrong_reassociation_frames": 0,
        "candidate_present_frames": 0,
        "id_switch_count": 0,
        "recorrection_opportunity_count": 0,
        "assignment_change_count": 0,
        "assignment_change_correct_count": 0,
        "assignment_change_incorrect_count": 0,
        "assignment_change_neutral_count": 0,
        "solver_coupled_collateral_count": 0,
        "protected_compared": 0,
        "protected_regression_count": 0,
        "protected_improvement_count": 0,
        "identity_utility": 0.0,
        "candidate_recall": None,
        "target_mean_iou": None,
        "future_identity_error": None,
        "missing_rate": None,
        "id_switch_rate": None,
        "wrong_reassociation_rate": None,
        "recorrection_rate": None,
        "protected_regression_rate": None,
    }


def finalize_metric(metric: dict[str, Any]) -> dict[str, Any]:
    frames = int(metric["evaluated_frames"])
    if frames:
        metric["target_mean_iou"] = float(metric["target_iou_sum"] / frames)
        metric["future_identity_error"] = float(metric["target_identity_error_frames"] / frames)
        metric["missing_rate"] = float(metric["target_missing_frames"] / frames)
        metric["id_switch_rate"] = float(metric["id_switch_count"] / frames)
        metric["wrong_reassociation_rate"] = float(metric["wrong_reassociation_frames"] / frames)
        metric["recorrection_rate"] = float(metric["recorrection_opportunity_count"] / frames)
        metric["candidate_recall"] = float(metric["candidate_present_frames"] / frames)
    compared = int(metric["protected_compared"])
    if compared:
        metric["protected_regression_rate"] = float(metric["protected_regression_count"] / compared)
    metric["identity_utility"] = float(metric["identity_utility"] / max(1, frames))
    return metric


def posthoc_score(scenarios: list[dict[str, Any]], validation: dict[str, Any]) -> dict[str, Any]:
    # This is the first point at which the GT files are opened.  The runtime
    # manifest and every candidate/score artifact have already passed the
    # causal validator above.
    gt_by_sequence = {scenario["sequence"]: load_gt(scenario["sequence"]) for scenario in scenarios}
    by_event_variant_frame: dict[tuple[str, str, int], dict[str, Any]] = {}
    for scenario in scenarios:
        rows = read_jsonl(ARTIFACT_ROOT / f"{scenario['event_id']}.jsonl")
        for row in rows:
            by_event_variant_frame[(scenario["event_id"], str(row["variant"]), int(row["frame"]))] = row

    event_metrics: list[dict[str, Any]] = []
    aggregate: dict[str, dict[str, dict[str, Any]]] = {variant: {} for variant in VARIANTS}
    action_aggregate: dict[str, dict[str, dict[str, dict[str, Any]]]] = defaultdict(lambda: defaultdict(dict))
    sequence_values: dict[tuple[str, str], dict[str, float]] = {}
    for scenario in scenarios:
        event_id = scenario["event_id"]
        event_frame = int(scenario["event_frame"])
        target_pid = int(scenario["target_public_id"])
        target_gid = int(scenario["event"]["dataset_gt_id"])
        gt_frames = gt_by_sequence[scenario["sequence"]]
        m0_event = by_event_variant_frame[(event_id, "M0_CURRENT_FRAME_CORRECTION_ONLY", event_frame)]
        m0_event_map = row_map(m0_event)
        protected_public_by_gt: dict[int, int] = {}
        for gt_id, gt_item in gt_frames.get(event_frame, {}).items():
            if int(gt_id) == target_gid:
                continue
            best_iou, best_candidate = candidate_best_iou(m0_event, gt_item["box"])
            if best_candidate is not None and best_iou >= IOU_THRESHOLD and best_candidate.get("public_id") is not None:
                protected_public_by_gt[int(gt_id)] = int(best_candidate["public_id"])
        event_result: dict[str, Any] = {
            "event_id": event_id,
            "sequence": scenario["sequence"],
            "action_type": scenario["action_type"],
            "event_frame": event_frame,
            "target_public_id": target_pid,
            "target_dataset_gt_id": target_gid,
            "interaction_source": "simulated_from_gt",
            "not_real_human_evidence": True,
            "protected_public_by_gt_posthoc": protected_public_by_gt,
            "horizons": {},
            "runtime_future_gt_used": False,
            "gt_usage": "posthoc_only_after_runtime_validation",
        }
        for variant in VARIANTS:
            variant_frames = [by_event_variant_frame[(event_id, variant, event_frame + offset)] for offset in range(1, 101)]
            baseline_frames = [by_event_variant_frame[(event_id, "M0_CURRENT_FRAME_CORRECTION_ONLY", event_frame + offset)] for offset in range(1, 101)]
            previous_observed_pid: int | None = None
            previous_error = False
            horizon_metrics: dict[str, dict[str, Any]] = {}
            for horizon in HORIZONS:
                metric = metric_template()
                frame_details: list[dict[str, Any]] = []
                previous_observed_pid = None
                previous_error = False
                for offset in range(1, horizon + 1):
                    frame = event_frame + offset
                    treatment_row = variant_frames[offset - 1]
                    baseline_row = baseline_frames[offset - 1]
                    gt_target = gt_frames.get(frame, {}).get(target_gid)
                    if gt_target is None:
                        continue
                    gt_box = gt_target["box"]
                    treatment_iou = public_target_iou(treatment_row, target_pid, gt_box)
                    baseline_iou = public_target_iou(baseline_row, target_pid, gt_box)
                    treatment_best_iou, treatment_best_candidate = candidate_best_iou(treatment_row, gt_box)
                    baseline_best_iou, baseline_best_candidate = candidate_best_iou(baseline_row, gt_box)
                    treatment_correct = treatment_iou >= IOU_THRESHOLD
                    baseline_correct = baseline_iou >= IOU_THRESHOLD
                    treatment_missing = not any(
                        item.get("public_id") is not None and int(item["public_id"]) == target_pid
                        for item in treatment_row.get("candidate_rows", [])
                    )
                    wrong = False
                    target_candidate = next(
                        (item for item in treatment_row.get("candidate_rows", []) if item.get("public_id") is not None and int(item["public_id"]) == target_pid),
                        None,
                    )
                    if target_candidate is not None:
                        for other_id, other_item in gt_frames.get(frame, {}).items():
                            if int(other_id) == target_gid:
                                continue
                            if box_iou(target_candidate["box_xyxy"], other_item["box"]) >= IOU_THRESHOLD:
                                wrong = True
                                break
                    observed_pid = None
                    if treatment_best_candidate is not None and treatment_best_iou >= IOU_THRESHOLD and treatment_best_candidate.get("public_id") is not None:
                        observed_pid = int(treatment_best_candidate["public_id"])
                    switch = previous_observed_pid is not None and observed_pid is not None and observed_pid != previous_observed_pid
                    previous_observed_pid = observed_pid if observed_pid is not None else previous_observed_pid
                    error = not treatment_correct
                    recorrection = bool(error and not previous_error)
                    previous_error = error
                    baseline_map = row_map(baseline_row)
                    treatment_map = row_map(treatment_row)
                    changed = baseline_map != treatment_map
                    target_direction = treatment_iou - baseline_iou
                    change_correct = bool(changed and (target_direction > 1.0e-9 or (treatment_correct and not baseline_correct)))
                    change_incorrect = bool(changed and (target_direction < -1.0e-9 or (baseline_correct and not treatment_correct)))
                    change_neutral = bool(changed and not change_correct and not change_incorrect)
                    collateral = bool(changed and any(baseline_map.get(uid) != treatment_map.get(uid) for uid in set(baseline_map) - {str(target_candidate.get("candidate_uid")) if target_candidate else "__none__"}))
                    metric["evaluated_frames"] += 1
                    metric["target_gt_present_frames"] += 1
                    metric["target_iou_sum"] += float(treatment_iou)
                    metric["target_correct_frames"] += int(treatment_correct)
                    metric["target_missing_frames"] += int(treatment_missing)
                    metric["target_identity_error_frames"] += int(not treatment_correct)
                    metric["wrong_reassociation_frames"] += int(wrong)
                    metric["candidate_present_frames"] += int(treatment_best_iou >= IOU_THRESHOLD)
                    metric["id_switch_count"] += int(switch)
                    metric["recorrection_opportunity_count"] += int(recorrection)
                    metric["assignment_change_count"] += int(changed)
                    metric["assignment_change_correct_count"] += int(change_correct)
                    metric["assignment_change_incorrect_count"] += int(change_incorrect)
                    metric["assignment_change_neutral_count"] += int(change_neutral)
                    metric["solver_coupled_collateral_count"] += int(collateral)
                    metric["identity_utility"] += 0.5 * (treatment_iou - baseline_iou) + 0.5 * (int(baseline_correct) - int(treatment_correct))
                    for protected_gt, protected_pid in protected_public_by_gt.items():
                        gt_other = gt_frames.get(frame, {}).get(protected_gt)
                        if gt_other is None:
                            continue
                        baseline_protected_iou, _ = public_box_for_gt(baseline_row, protected_pid, gt_other["box"])
                        treatment_protected_iou, _ = public_box_for_gt(treatment_row, protected_pid, gt_other["box"])
                        baseline_protected_correct = baseline_protected_iou >= IOU_THRESHOLD
                        treatment_protected_correct = treatment_protected_iou >= IOU_THRESHOLD
                        metric["protected_compared"] += 1
                        metric["protected_regression_count"] += int(baseline_protected_correct and not treatment_protected_correct)
                        metric["protected_improvement_count"] += int(treatment_protected_correct and not baseline_protected_correct)
                    frame_details.append(
                        {
                            "frame": frame,
                            "target_iou": float(treatment_iou),
                            "baseline_m0_target_iou": float(baseline_iou),
                            "target_correct": treatment_correct,
                            "baseline_m0_target_correct": baseline_correct,
                            "target_missing": treatment_missing,
                            "candidate_recall_present": bool(treatment_best_iou >= IOU_THRESHOLD),
                            "wrong_reassociation": wrong,
                            "id_switch": switch,
                            "recorrection_opportunity": recorrection,
                            "assignment_changed": changed,
                            "assignment_change_correct": change_correct,
                            "assignment_change_incorrect": change_incorrect,
                            "assignment_change_neutral": change_neutral,
                            "solver_coupled_collateral": collateral,
                            "runtime_future_gt_used": False,
                        }
                    )
                metric = finalize_metric(metric)
                horizon_metrics[str(horizon)] = {**metric, "frame_details": frame_details}
            event_result["horizons"][variant] = horizon_metrics
            for horizon in HORIZONS:
                value = horizon_metrics[str(horizon)]
                aggregate.setdefault(variant, {})
                action_aggregate[scenario["action_type"]].setdefault(variant, {})
                action_aggregate[scenario["action_type"]].setdefault(variant, {})
                sequence_values[(variant, str(horizon))] = {
                    **sequence_values.get((variant, str(horizon)), {}),
                    scenario["sequence"]: float(value["identity_utility"]),
                }
        event_metrics.append(event_result)

    for variant in VARIANTS:
        for horizon in HORIZONS:
            selected = [event["horizons"][variant][str(horizon)] for event in event_metrics]
            sums = metric_template()
            for value in selected:
                for key in (
                    "evaluated_frames", "target_gt_present_frames", "target_iou_sum", "target_correct_frames", "target_missing_frames",
                    "target_identity_error_frames", "wrong_reassociation_frames", "candidate_present_frames", "id_switch_count",
                    "recorrection_opportunity_count", "assignment_change_count", "assignment_change_correct_count",
                    "assignment_change_incorrect_count", "assignment_change_neutral_count", "solver_coupled_collateral_count",
                    "protected_compared", "protected_regression_count", "protected_improvement_count", "identity_utility",
                ):
                    sums[key] += value[key]
            sums = finalize_metric(sums)
            ci = bootstrap(sequence_values[(variant, str(horizon))], HORIZONS.index(horizon) + VARIANTS.index(variant) * 10)
            sums["sequence_cluster_bootstrap_95ci"] = ci
            sums["event_count"] = len(selected)
            sums["independent_sequence_count"] = len({event["sequence"] for event in event_metrics})
            aggregate[variant][str(horizon)] = sums
            for action in sorted({event["action_type"] for event in event_metrics}):
                action_selected = [event["horizons"][variant][str(horizon)] for event in event_metrics if event["action_type"] == action]
                action_sum = metric_template()
                for value in action_selected:
                    for key in (
                        "evaluated_frames", "target_gt_present_frames", "target_iou_sum", "target_correct_frames", "target_missing_frames",
                        "target_identity_error_frames", "wrong_reassociation_frames", "candidate_present_frames", "id_switch_count",
                        "recorrection_opportunity_count", "assignment_change_count", "assignment_change_correct_count",
                        "assignment_change_incorrect_count", "assignment_change_neutral_count", "solver_coupled_collateral_count",
                        "protected_compared", "protected_regression_count", "protected_improvement_count", "identity_utility",
                    ):
                        action_sum[key] += value[key]
                action_aggregate[action].setdefault(variant, {})[str(horizon)] = finalize_metric(action_sum)

    primary_h20 = aggregate[PRIMARY_VARIANT]["20"]
    gate_by_variant: dict[str, Any] = {}
    for variant in MEMORY_VARIANTS:
        main = aggregate[variant]["20"]
        gate_by_variant[variant] = {
            "primary_horizon": 20,
            "identity_utility_ci_lower": main["sequence_cluster_bootstrap_95ci"]["lower"],
            "ci_lower_strictly_positive": bool(main["sequence_cluster_bootstrap_95ci"]["lower"] is not None and main["sequence_cluster_bootstrap_95ci"]["lower"] > 0.0),
            "correct_assignment_changes": int(main["assignment_change_correct_count"]),
            "incorrect_assignment_changes": int(main["assignment_change_incorrect_count"]),
            "protected_regression_count": int(main["protected_regression_count"]),
            "public_state_continuity": True,
            "runtime_future_gt_used": False,
        }
    strict_primary = gate_by_variant[PRIMARY_VARIANT]
    strict_gate = bool(
        strict_primary["ci_lower_strictly_positive"]
        and strict_primary["correct_assignment_changes"] > 0
        and strict_primary["incorrect_assignment_changes"] == 0
        and strict_primary["protected_regression_count"] == 0
        and validation["runtime_future_gt_used"] is False
    )
    result = {
        "schema_version": "N72R3_STAGE22_EFFECT_RESULTS_V1",
        "status": "PASS_EXECUTION_FAIL_FUTURE_EFFECT" if not strict_gate else "PASS_EXECUTION_FUTURE_EFFECT_PASS",
        "created_at_utc": now_utc(),
        "event_count": len(scenarios),
        "independent_sequence_count": len({scenario["sequence"] for scenario in scenarios}),
        "event_quota_target": MAX_EVENT_TARGET,
        "sequence_quota_target": MIN_SEQUENCE_TARGET,
        "eligible_events_exhausted": len(scenarios) < MAX_EVENT_TARGET,
        "eligible_event_shortfall": max(0, MAX_EVENT_TARGET - len(scenarios)),
        "horizons": list(HORIZONS),
        "variants": list(VARIANTS),
        "aggregate": aggregate,
        "action_aggregate": action_aggregate,
        "event_metrics": event_metrics,
        "gate": {
            "research_gate": "PASS_FUTURE_EFFECT" if strict_gate else "FAIL_FUTURE_EFFECT",
            "primary_variant": PRIMARY_VARIANT,
            "primary_horizon": 20,
            "by_variant": gate_by_variant,
            "strict_primary_gate": strict_gate,
            "candidate_completeness": True,
            "mapping_completeness": True,
            "public_state_continuity": True,
            "solver_coupled_collateral_labeled": True,
            "runtime_future_gt_used": False,
            "posthoc_gt_loaded_after_runtime_validation": True,
            "interaction_source": "simulated_from_gt",
            "real_human_tape": False,
            "production_authorized": False,
        },
        "bootstrap_protocol": {
            "repetitions": BOOTSTRAP_REPETITIONS,
            "seed": BOOTSTRAP_SEED,
            "cluster_unit": "independent_sequence",
            "identity_utility": "0.5*(treated_target_iou-baseline_M0_target_iou)+0.5*(baseline_M0_correct-treated_correct), per evaluated target-GT frame, then sequence mean",
        },
        "runtime_validation": validation,
        "runtime_future_gt_used": False,
        "gt_usage": "posthoc_only_after_runtime_artifacts_frozen",
        "interaction_source": "simulated_from_gt",
        "real_human_tape": False,
        "not_real_human_evidence": True,
        "production_authorized": False,
        "scientific_result": "EXPLORATORY_FUTURE_EFFECT_GATE_ONLY",
    }
    atomic_json(RESULT_PATH, result)
    return result


def write_stage20(scenarios: list[dict[str, Any]], runtime: dict[str, Any]) -> None:
    atomic_json(
        STAGE20_STATUS,
        {
            "schema_version": "N72R3_STAGE_STATUS_V1",
            "stage": "20_GT_SIMULATED_EFFECT_EXPERIMENT",
            "status": "PASS_STAGE20_RUNTIME_EXECUTION",
            "event_count": len(scenarios),
            "independent_sequence_count": len({scenario["sequence"] for scenario in scenarios}),
            "event_quota_target": MAX_EVENT_TARGET,
            "sequence_quota_target": MIN_SEQUENCE_TARGET,
            "eligible_events_exhausted": len(scenarios) < MAX_EVENT_TARGET,
            "event_quota_status": "ELIGIBLE_EXHAUSTED_BELOW_TARGET" if len(scenarios) < MAX_EVENT_TARGET else "TARGET_MET",
            "variants": list(VARIANTS),
            "horizons": list(HORIZONS),
            "runtime_manifest": str(RUNTIME_MANIFEST),
            "runtime_root": str(REPLAY_ROOT),
            "candidate_streams": [scenario["candidate_sha256"] for scenario in scenarios],
            "runtime_future_gt_used": False,
            "gt_loaded_in_worker": False,
            "interaction_source": "simulated_from_gt",
            "real_human_tape": False,
            "scientific_result": "EXPLORATORY_RUNTIME_ARTIFACTS_NOT_FUTURE_EFFECT",
        },
    )


def write_failure(exc: BaseException) -> Path:
    FAILURE_ROOT.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = FAILURE_ROOT / f"stage20_22_effect_replay_failure_{stamp}.json"
    atomic_json(
        path,
        {
            "schema_version": "N72R3_FAILURE_RECORD_V1",
            "stage": "20_22_EFFECT_REPLAY",
            "status": "FAIL_PRESERVED",
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
            "runtime_root": str(REPLAY_ROOT),
            "runtime_future_gt_used": False,
            "scientific_result": "NO_SCIENTIFIC_RESULT",
        },
    )
    return path


def main() -> int:
    try:
        scenarios = load_scenarios()
        runtime = run_runtime(scenarios)
        write_stage20(scenarios, runtime)
        validation = validate_runtime(scenarios)
        result = posthoc_score(scenarios, validation)
        atomic_json(
            STAGE22_STATUS,
            {
                "schema_version": "N72R3_STAGE_STATUS_V1",
                "stage": "22_STRICT_FUTURE_EFFECT_GATE",
                "status": "PASS_STAGE22_FUTURE_EFFECT_GATE" if result["gate"]["research_gate"] == "PASS_FUTURE_EFFECT" else "FAIL_STAGE22_FUTURE_EFFECT_GATE",
                "research_gate": result["gate"]["research_gate"],
                "event_count": result["event_count"],
                "independent_sequence_count": result["independent_sequence_count"],
                "primary_variant": PRIMARY_VARIANT,
                "primary_horizon": 20,
                "result_artifact": str(RESULT_PATH),
                "runtime_manifest": str(RUNTIME_MANIFEST),
                "gate": result["gate"],
                "runtime_future_gt_used": False,
                "interaction_source": "simulated_from_gt",
                "real_human_tape": False,
                "production_authorized": False,
                "scientific_result": "EXPLORATORY_EFFECT_GATE_NOT_PRODUCTION_AUTHORIZATION",
            },
        )
        print(json.dumps({"status": result["status"], "research_gate": result["gate"]["research_gate"], "result": str(RESULT_PATH)}, sort_keys=True))
        return 0
    except Exception as exc:
        failure = write_failure(exc)
        print(json.dumps({"status": "FAIL_STAGE20_22_EFFECT_REPLAY", "failure": str(failure)}, sort_keys=True))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
