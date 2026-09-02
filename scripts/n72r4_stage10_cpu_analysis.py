#!/usr/bin/env python3
"""CPU-only Stage 10/12 analysis for the official N72R4 future branches.

The official SAM3 worker intentionally does not infer public identity.  This
module therefore adds an explicit, posthoc association adapter for evaluation
only.  Its authority comes from the frozen Stage-18 persistent runtime and
the current-frame correction observation; candidate index, raw SAM ID, and
adapter ID are never promoted to a public ID.

The runtime artifacts are validated before GT is opened.  Candidate recall is
then scored posthoc from the same official NO and corrected streams.  The
mapping adapter uses the frozen non-learned pairwise cues and an explicit
NONE outcome, and never reads GT or future metrics while assigning candidates.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import tempfile
from typing import Any, Iterable

import numpy as np
from scipy.optimize import linear_sum_assignment

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


FROZEN_N72R3 = Path(
    os.environ.get(
        "N72R4_FROZEN_N72R3_ROOT",
        "/data2/usr_for_deadline/SAM3_InterMOT_N72R3/worktree/outputs/N72R3",
    )
)
EVENT_MANIFEST = FROZEN_N72R3 / "simulation/real_event_manifest.json"
STAGE16_EVENT_ROOT = FROZEN_N72R3 / "official_correction/events"
STAGE18_ROOT = FROZEN_N72R3 / "baseline/stage18_persistent_public/full_eligible"
PAIR_MANIFEST = ROOT / "outputs/N72R4/official_future_pair_manifest_attempt2.json"
NO_ROOT = ROOT / "outputs/N72R4/official_no_intervention/full_attempt2"
M0_ROOT = ROOT / "outputs/N72R4/official_corrected/full_attempt2"
OUT = ROOT / "outputs/N72R4"
DEFAULT_AUDIT = OUT / "mechanism_probe/public_mapping_audit.jsonl"
DEFAULT_RECALL = OUT / "candidate_recall/no_vs_m0_candidate_recall.json"
DEFAULT_STATUS = OUT / "stage_status/stage_10_status.json"

HORIZONS = (20, 50, 100)
IOU_THRESHOLD = 0.5

# These are the already frozen non-learned pairwise cues used by the N72R3
# CPU probe.  They are used only by the posthoc association adapter; they do
# not alter the official SAM3 branch or production online association.
SIM_WEIGHT = 1.5
IOU_WEIGHT = 1.0
NATIVE_WEIGHT = 0.5
NATIVE_BONUS = 3.0
GAP_WEIGHT = 0.1
NONE_SCORE = 0.0


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
                raise TypeError(f"{path}:{line_number}: expected object")
            rows.append(value)
    return rows


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def finite_feature(value: Any) -> np.ndarray:
    feature = np.asarray(value, dtype=np.float32).reshape(-1)
    if feature.size != 512 or not np.all(np.isfinite(feature)):
        raise ValueError(f"official candidate feature is not finite 512-D: {feature.shape}")
    norm = float(np.linalg.norm(feature))
    if norm <= 1.0e-6:
        raise ValueError("official candidate feature has zero norm")
    return feature / norm


def feature_digest(feature: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(feature, dtype="<f4").tobytes()).hexdigest()


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


def center(box: Any) -> np.ndarray:
    value = np.asarray(box, dtype=np.float64).reshape(4)
    return np.asarray([(value[0] + value[2]) / 2.0, (value[1] + value[3]) / 2.0], dtype=np.float64)


@dataclass
class State:
    public_id: int
    last_box: np.ndarray | None
    last_feature: np.ndarray | None
    velocity: np.ndarray
    last_frame: int
    last_native: int | None
    status: str = "ACTIVE"


def state_copy(value: State) -> State:
    return State(
        public_id=int(value.public_id),
        last_box=None if value.last_box is None else value.last_box.copy(),
        last_feature=None if value.last_feature is None else value.last_feature.copy(),
        velocity=value.velocity.copy(),
        last_frame=int(value.last_frame),
        last_native=None if value.last_native is None else int(value.last_native),
        status=str(value.status),
    )


def predicted_box(state: State, frame: int) -> np.ndarray | None:
    if state.last_box is None:
        return None
    gap = max(0, int(frame) - int(state.last_frame))
    result = state.last_box.astype(np.float64, copy=True)
    result[[0, 2]] += state.velocity[0] * gap
    result[[1, 3]] += state.velocity[1] * gap
    return result


def association_score(state: State, row: dict[str, Any], frame: int) -> float:
    feature = finite_feature(row["feature"])
    similarity = 0.0 if state.last_feature is None else float(np.dot(feature, state.last_feature))
    predicted = predicted_box(state, frame)
    geometry = 0.0 if predicted is None else box_iou(predicted, row["box_xyxy"])
    native = int(row["adapter_external_id"])
    native_same = 1.0 if state.last_native is not None and native == int(state.last_native) else 0.0
    gap = min(1.0, max(0, int(frame) - int(state.last_frame)) / 200.0)
    return float(
        SIM_WEIGHT * similarity
        + IOU_WEIGHT * geometry
        + NATIVE_WEIGHT * native_same
        + NATIVE_BONUS * native_same
        - GAP_WEIGHT * gap
    )


def update_state(state: State, row: dict[str, Any], frame: int) -> None:
    new_box = np.asarray(row["box_xyxy"], dtype=np.float64).reshape(4)
    new_feature = finite_feature(row["feature"])
    old_center = center(state.last_box) if state.last_box is not None else center(new_box)
    delta = (center(new_box) - old_center) / max(1, int(frame) - int(state.last_frame))
    state.velocity = 0.8 * state.velocity + 0.2 * delta
    state.last_box = new_box.copy()
    state.last_feature = new_feature.copy()
    state.last_frame = int(frame)
    state.last_native = int(row["adapter_external_id"])
    state.status = "ACTIVE"


def event_id_from_path(path: Path) -> str:
    return path.name.removesuffix(".jsonl")


def event_file(root: Path, event_id: str) -> Path:
    path = root / f"{event_id}.jsonl"
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def ensure_official_candidate(row: dict[str, Any], event_id: str, frame: int) -> None:
    forbidden = {"dataset_gt_id", "gt_box", "future_gt", "public_id_inference_result"}
    if forbidden.intersection(row):
        raise RuntimeError(f"posthoc GT/identity field entered official row: {event_id}/{frame}")
    if row.get("runtime_future_gt_used") is not False or row.get("runtime_gt_read") is not False:
        raise RuntimeError(f"official row violates runtime GT boundary: {event_id}/{frame}")
    candidates = row.get("candidates")
    # A complete official propagation frame may legitimately contain zero
    # observations (for example after a tracker loses every object).  That is
    # different from a missing candidate list/frame and is represented by the
    # explicit NONE outcome in the posthoc adapter.
    if not isinstance(candidates, list):
        raise RuntimeError(f"official candidate set is missing: {event_id}/{frame}")
    raw_ids = []
    for candidate in candidates:
        for key in ("official_raw_sam_id", "raw_native_id", "native_tid", "adapter_external_id", "adapter_visible_id"):
            if key not in candidate or candidate[key] is None:
                raise RuntimeError(f"official candidate lacks {key}: {event_id}/{frame}")
        raw_ids.append(int(candidate["official_raw_sam_id"]))
        box = np.asarray(candidate.get("box_xyxy"), dtype=np.float64).reshape(-1)
        if box.size != 4 or not np.all(np.isfinite(box)):
            raise RuntimeError(f"official candidate box is invalid: {event_id}/{frame}")
        finite_feature(candidate.get("feature"))
    if len(raw_ids) != len(set(raw_ids)):
        raise RuntimeError(f"duplicate official raw candidate IDs: {event_id}/{frame}")
    if row.get("candidate_set_complete") is not True:
        raise RuntimeError(f"official candidate set is not marked complete: {event_id}/{frame}")
    if row.get("public_id_inference") is not False:
        raise RuntimeError(f"official worker inferred public IDs: {event_id}/{frame}")


def load_events() -> list[dict[str, Any]]:
    manifest = read_json(EVENT_MANIFEST)
    if manifest.get("status") != "PASS_STAGE14_POLICY_FROZEN":
        raise RuntimeError("N72R4 Stage 10 requires the frozen N72R3 event manifest")
    events = [dict(item) for item in manifest.get("events", [])]
    if len(events) != 6:
        raise RuntimeError(f"expected six frozen N72R3 events, found {len(events)}")
    for event in events:
        if event.get("interaction_source") != "simulated_from_gt" or event.get("runtime_future_gt_used") is not False:
            raise RuntimeError(f"event provenance/casual flag invalid: {event.get('event_id')}")
    return sorted(events, key=lambda value: str(value["event_id"]))


def stage18_dir(sequence: str, event_frame: int) -> Path:
    matches = sorted(STAGE18_ROOT.glob(f"n71-{sequence}-{int(event_frame):04d}"))
    if len(matches) != 1:
        raise RuntimeError(f"Stage18 persistent mapping is not unique: {sequence}/{event_frame}/{matches}")
    return matches[0]


def load_stage18_public_map(sequence: str, event_frame: int) -> dict[int, int]:
    path = stage18_dir(sequence, event_frame) / "candidate_decisions.jsonl"
    rows = read_jsonl(path)
    selected = [row for row in rows if int(row.get("frame_idx", -1)) == int(event_frame)]
    if not selected:
        raise RuntimeError(f"Stage18 has no event-frame candidate decisions: {path}")
    result: dict[int, int] = {}
    for row in selected:
        if row.get("runtime_future_gt_used") is not False:
            raise RuntimeError(f"Stage18 mapping used future GT: {path}")
        raw = row.get("official_raw_sam_id")
        public = row.get("public_id")
        if raw is None or public is None:
            raise RuntimeError(f"Stage18 mapping has missing raw/public axis: {path}")
        raw_id, public_id = int(raw), int(public)
        if raw_id in result and result[raw_id] != public_id:
            raise RuntimeError(f"Stage18 raw ID maps to multiple public IDs: {path}/{raw_id}")
        result[raw_id] = public_id
    if len(result) != len(set(result.values())):
        raise RuntimeError(f"Stage18 public axis is not one-to-one: {path}")
    return result


def load_branch_rows(root: Path, event_id: str) -> dict[int, dict[str, Any]]:
    rows = read_jsonl(event_file(root, event_id))
    if len(rows) != 101:
        raise RuntimeError(f"official branch must contain event+100 frames: {event_id}/{len(rows)}")
    result: dict[int, dict[str, Any]] = {}
    for row in rows:
        frame = int(row.get("frame", -1))
        if frame in result:
            raise RuntimeError(f"duplicate official frame: {event_id}/{frame}")
        if row.get("event_id") != event_id or row.get("branch") not in {"B0_NO_INTERVENTION", "B1_CURRENT_FRAME_CORRECTION"}:
            raise RuntimeError(f"official branch identity mismatch: {event_id}/{frame}")
        ensure_official_candidate(row, event_id, frame)
        result[frame] = row
    first = min(result)
    event_frame = int(result[first]["event_frame"])
    expected = set(range(event_frame, event_frame + 101))
    if set(result) != expected or first != event_frame:
        raise RuntimeError(f"official frame range is incomplete: {event_id}")
    return result


def load_stage16(event_id: str) -> dict[str, Any]:
    path = STAGE16_EVENT_ROOT / f"{event_id}.json"
    artifact = read_json(path)
    if artifact.get("status") != "PASS_STAGE16_OFFICIAL_CORRECTION_AND_STAGE17_MEMORY":
        raise RuntimeError(f"Stage16 correction artifact is not PASS: {event_id}")
    if artifact.get("runtime_future_gt_used") is not False:
        raise RuntimeError(f"Stage16 artifact violates runtime GT boundary: {event_id}")
    return artifact


def current_post_anchor(stage16: dict[str, Any], corrected_event_row: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    correction = corrected_event_row.get("correction") or {}
    post_candidates = correction.get("post_candidates")
    post_observation = correction.get("post_observation")
    if not isinstance(post_candidates, list) or not post_candidates or not isinstance(post_observation, dict):
        raise RuntimeError(f"corrected branch lacks complete current-frame post candidates: {stage16['event_id']}")
    human_box = post_observation.get("box_xyxy")
    if human_box is None:
        raise RuntimeError(f"corrected branch lacks official post-observation box: {stage16['event_id']}")
    scored = sorted(
        ((box_iou(candidate["box_xyxy"], human_box), int(candidate["official_raw_sam_id"]), candidate) for candidate in post_candidates),
        key=lambda item: (-item[0], item[1]),
    )
    if not scored or scored[0][0] < IOU_THRESHOLD:
        raise RuntimeError(f"current correction post-observation cannot anchor a candidate: {stage16['event_id']}")
    if len(scored) > 1 and math.isclose(scored[0][0], scored[1][0], rel_tol=0.0, abs_tol=1.0e-12):
        raise RuntimeError(f"current correction post-observation has an ambiguous candidate anchor: {stage16['event_id']}")
    return scored[0][2], [dict(item) for item in post_candidates]


def make_row_key(branch: str, event_id: str, frame: int, raw_id: int) -> str:
    return f"official:{branch}:{event_id}:frame:{int(frame)}:raw:{int(raw_id)}"


def row_view(branch: str, event_id: str, row: dict[str, Any]) -> list[dict[str, Any]]:
    result = []
    for candidate in row["candidates"]:
        result.append(
            {
                "candidate_key": make_row_key(branch, event_id, int(row["frame"]), int(candidate["official_raw_sam_id"])),
                "candidate_index": int(candidate["candidate_index"]),
                "official_raw_sam_id": int(candidate["official_raw_sam_id"]),
                "raw_native_id": int(candidate["raw_native_id"]),
                "native_tid": int(candidate["native_tid"]),
                "adapter_external_id": int(candidate["adapter_external_id"]),
                "adapter_visible_id": int(candidate["adapter_visible_id"]),
                "box_xyxy": [float(value) for value in candidate["box_xyxy"]],
                "feature": finite_feature(candidate["feature"]),
                "feature_sha256": str(candidate["feature_sha256"]),
                "confidence": float(candidate["confidence"]),
                "source": str(candidate.get("source", "automatic_propagation")),
            }
        )
    return result


def prestate_rows_by_public(event_row: dict[str, Any], public_by_raw: dict[int, int], branch: str, event_id: str) -> dict[int, dict[str, Any]]:
    rows = row_view(branch, event_id, event_row)
    result: dict[int, dict[str, Any]] = {}
    for row in rows:
        raw = int(row["official_raw_sam_id"])
        if raw not in public_by_raw:
            raise RuntimeError(f"official event candidate has no Stage18 public mapping: {event_id}/{raw}")
        public_id = int(public_by_raw[raw])
        if public_id in result:
            raise RuntimeError(f"event public ID is duplicated: {event_id}/{public_id}")
        result[public_id] = row
    if set(result) != set(public_by_raw.values()):
        raise RuntimeError(f"Stage18/public event axes do not match: {event_id}")
    return result


def initialise_b0(event_row: dict[str, Any], public_by_raw: dict[int, int], event_id: str) -> dict[int, State]:
    states: dict[int, State] = {}
    for public_id, row in prestate_rows_by_public(event_row, public_by_raw, "B0_NO_INTERVENTION", event_id).items():
        states[public_id] = State(
            public_id=public_id,
            last_box=np.asarray(row["box_xyxy"], dtype=np.float64),
            last_feature=finite_feature(row["feature"]),
            velocity=np.zeros(2, dtype=np.float64),
            last_frame=int(event_row["frame"]),
            last_native=int(row["adapter_external_id"]),
        )
    return states


def initialise_b1(
    event_row: dict[str, Any],
    post_candidates: list[dict[str, Any]],
    target_post: dict[str, Any],
    public_by_raw: dict[int, int],
    target_public: int,
    event_id: str,
) -> tuple[dict[int, State], dict[int, int]]:
    pre_by_public = prestate_rows_by_public(event_row, public_by_raw, "B1_CURRENT_FRAME_CORRECTION", event_id)
    states = {
        public_id: State(
            public_id=public_id,
            last_box=np.asarray(row["box_xyxy"], dtype=np.float64),
            last_feature=finite_feature(row["feature"]),
            velocity=np.zeros(2, dtype=np.float64),
            last_frame=int(event_row["frame"]),
            last_native=int(row["adapter_external_id"]),
            status="LOST",
        )
        for public_id, row in pre_by_public.items()
    }
    remaining_public = [public for public in sorted(pre_by_public) if public != int(target_public)]
    target_raw_id = int(target_post["official_raw_sam_id"])
    remaining_candidates = [
        candidate
        for candidate in post_candidates
        if int(candidate["official_raw_sam_id"]) != target_raw_id
    ]
    public_to_post: dict[int, int] = {int(target_public): int(target_post["official_raw_sam_id"])}
    if remaining_public and remaining_candidates:
        cost = np.zeros((len(remaining_public), len(remaining_candidates)), dtype=np.float64)
        for public_index, public_id in enumerate(remaining_public):
            previous = pre_by_public[public_id]
            previous_feature = finite_feature(previous["feature"])
            for candidate_index, candidate in enumerate(remaining_candidates):
                candidate_feature = finite_feature(candidate["feature"])
                cost[public_index, candidate_index] = (
                    1.0 - box_iou(previous["box_xyxy"], candidate["box_xyxy"])
                    + 0.25 * (1.0 - float(np.dot(previous_feature, candidate_feature)))
                    + 1.0e-9 * int(candidate["official_raw_sam_id"])
                )
        assigned_public, assigned_candidate = linear_sum_assignment(cost)
        for public_index, candidate_index in zip(assigned_public.tolist(), assigned_candidate.tolist()):
            public_to_post[int(remaining_public[public_index])] = int(remaining_candidates[candidate_index]["official_raw_sam_id"])
    post_by_raw = {int(candidate["official_raw_sam_id"]): candidate for candidate in post_candidates}
    for public_id, raw_id in public_to_post.items():
        candidate = post_by_raw[int(raw_id)]
        update_state(states[int(public_id)], candidate, int(event_row["frame"]))
    return states, public_to_post


def associate_frame(states: dict[int, State], rows: list[dict[str, Any]], frame: int) -> tuple[list[int | None], list[dict[str, Any]]]:
    ordered_states = [states[public] for public in sorted(states)]
    assignment: list[int | None] = [None] * len(rows)
    audit: list[dict[str, Any]] = []
    if ordered_states and rows:
        scores = np.asarray(
            [[association_score(state, row, frame) for row in rows] for state in ordered_states], dtype=np.float64
        )
        state_indices, candidate_indices = linear_sum_assignment(-scores)
        matched_states: set[int] = set()
        matched_candidates: set[int] = set()
        for state_index, candidate_index in zip(state_indices.tolist(), candidate_indices.tolist()):
            score = float(scores[state_index, candidate_index])
            if score < NONE_SCORE:
                continue
            assignment[candidate_index] = int(ordered_states[state_index].public_id)
            matched_states.add(state_index)
            matched_candidates.add(candidate_index)
            audit.append(
                {
                    "candidate_index": int(rows[candidate_index]["candidate_index"]),
                    "candidate_raw_id": int(rows[candidate_index]["official_raw_sam_id"]),
                    "public_id": int(ordered_states[state_index].public_id),
                    "score": score,
                    "status": "ASSIGNED_EXISTING_IDENTITY",
                }
            )
        for state_index, state in enumerate(ordered_states):
            if state_index not in matched_states:
                state.status = "LOST"
        for candidate_index, row in enumerate(rows):
            if candidate_index not in matched_candidates:
                audit.append(
                    {
                        "candidate_index": int(row["candidate_index"]),
                        "candidate_raw_id": int(row["official_raw_sam_id"]),
                        "public_id": None,
                        "score": None,
                        "status": "EXPLICIT_NONE_UNMAPPED_OFFICIAL_CANDIDATE",
                    }
                )
    else:
        for state in ordered_states:
            state.status = "LOST"
        audit = [
            {
                "candidate_index": int(row["candidate_index"]),
                "candidate_raw_id": int(row["official_raw_sam_id"]),
                "public_id": None,
                "score": None,
                "status": "EXPLICIT_NONE_UNMAPPED_OFFICIAL_CANDIDATE",
            }
            for row in rows
        ]
    for item in audit:
        if item["public_id"] is not None:
            update_state(states[int(item["public_id"])], rows[next(index for index, row in enumerate(rows) if int(row["official_raw_sam_id"]) == int(item["candidate_raw_id"]))], frame)
    if len([item["public_id"] for item in audit if item["public_id"] is not None]) != len(
        {item["public_id"] for item in audit if item["public_id"] is not None}
    ):
        raise RuntimeError(f"posthoc mapping duplicated public ID at frame {frame}")
    return assignment, audit


def map_branch(
    branch_name: str,
    event: dict[str, Any],
    branch_rows: dict[int, dict[str, Any]],
    stage16: dict[str, Any],
    public_by_raw: dict[int, int],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    event_id = str(event["event_id"])
    event_frame = int(event["event_frame"])
    event_row = branch_rows[event_frame]
    target_public = int(stage16["persistent_identity"]["public_id"])
    target_post: dict[str, Any] | None = None
    public_to_post: dict[int, int] = {}
    if branch_name == "B0_NO_INTERVENTION":
        states = initialise_b0(event_row, public_by_raw, event_id)
    else:
        target_post, post_candidates = current_post_anchor(stage16, event_row)
        states, public_to_post = initialise_b1(
            event_row,
            post_candidates,
            target_post,
            public_by_raw,
            target_public,
            event_id,
        )
    output: list[dict[str, Any]] = []
    for frame in range(event_frame, event_frame + 101):
        row = branch_rows[frame]
        candidates = row_view(branch_name, event_id, row)
        if frame == event_frame:
            # Y_pre is before the current-frame correction for both branches.
            # It is retained as a causal audit, not as the post-correction
            # initialization used for future frames.
            mapping = [public_by_raw.get(int(candidate["official_raw_sam_id"])) for candidate in candidates]
            mapping_audit = [
                {
                    "candidate_index": int(candidate["candidate_index"]),
                    "candidate_raw_id": int(candidate["official_raw_sam_id"]),
                    "public_id": None if public is None else int(public),
                    "score": None,
                    "status": "PRESTATE_PERSISTENT_MAPPING",
                }
                for candidate, public in zip(candidates, mapping)
            ]
        else:
            mapping, mapping_audit = associate_frame(states, candidates, frame)
        if len([item for item in mapping if item is not None]) != len({item for item in mapping if item is not None}):
            raise RuntimeError(f"candidate/public mapping is not one-to-one: {event_id}/{branch_name}/{frame}")
        output.append(
            {
                "schema_version": "N72R4_POSTHOC_PUBLIC_MAPPING_V1",
                "event_id": event_id,
                "sequence": str(event["sequence"]),
                "branch": branch_name,
                "event_frame": event_frame,
                "frame": int(frame),
                "frame_horizon": int(frame - event_frame),
                "candidate_stream_sha256": str(row["y_pre_semantic_hash"] if frame == event_frame else row["frame_hash_sha256"]),
                "candidate_rows": [
                    {
                        "candidate_key": candidate["candidate_key"],
                        "candidate_index": int(candidate["candidate_index"]),
                        "official_raw_sam_id": int(candidate["official_raw_sam_id"]),
                        "adapter_external_id": int(candidate["adapter_external_id"]),
                        "box_xyxy": candidate["box_xyxy"],
                        "feature_sha256": candidate["feature_sha256"],
                        "public_id": None if public is None else int(public),
                        "assignment_status": next(
                            item["status"] for item in mapping_audit if int(item["candidate_index"]) == int(candidate["candidate_index"])
                        ),
                    }
                    for candidate, public in zip(candidates, mapping)
                ],
                "assignment_audit": mapping_audit,
                "public_id_authority": (
                    "N72R3_STAGE18_PERSISTENT_RUNTIME_PRESTATE"
                    if frame == event_frame
                    else "POSTHOC_EXPLICIT_NONE_ASSOCIATION_FROM_PERSISTENT_PRESTATE"
                ),
                "target_public_id": target_public,
                "event_frame_memory_read": False,
                "first_future_frame": event_frame + 1,
                "candidate_index_to_public_id": False,
                "official_raw_sam_id_to_public_id": False,
                "adapter_id_to_public_id": False,
                "runtime_future_gt_used": False,
                "posthoc_gt_used": False,
            }
        )
    return output, {
        "branch": branch_name,
        "event_id": event_id,
        "target_public_id": target_public,
        "event_frame": event_frame,
        "post_correction_target_raw_id": None if target_post is None else int(target_post["official_raw_sam_id"]),
        "post_correction_public_to_raw": {str(public): int(raw) for public, raw in sorted(public_to_post.items())},
        "future_frames": 100,
        "unmapped_candidate_count": sum(
            1 for row in output for candidate in row["candidate_rows"] if candidate["public_id"] is None
        ),
        "duplicate_public_frame_count": 0,
        "mapping_authority_is_persistent": True,
        "candidate_index_to_public_id": False,
        "raw_sam_id_to_public_id": False,
        "runtime_future_gt_used": False,
        "posthoc_gt_used": False,
    }


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


def score_recall(
    events: list[dict[str, Any]],
    branch_rows_by_event: dict[tuple[str, str], dict[int, dict[str, Any]]],
    mapping_by_event: dict[tuple[str, str], list[dict[str, Any]]],
) -> dict[str, Any]:
    gt_by_sequence = {str(event["sequence"]): load_gt(str(event["sequence"])) for event in events}
    event_details: list[dict[str, Any]] = []
    for event in events:
        event_id = str(event["event_id"])
        sequence = str(event["sequence"])
        event_frame = int(event["event_frame"])
        target_gid = int(event["dataset_gt_id"])
        event_result: dict[str, Any] = {
            "event_id": event_id,
            "sequence": sequence,
            "action_type": str(event["action_type"]),
            "event_frame": event_frame,
            "target_dataset_gt_id": target_gid,
            "interaction_source": "simulated_from_gt",
            "not_real_human_evidence": True,
            "horizons": {},
            "runtime_future_gt_used": False,
            "gt_usage": "posthoc_only_after_official_runtime_validation",
        }
        gt_frames = gt_by_sequence[sequence]
        for branch in ("B0_NO_INTERVENTION", "B1_CURRENT_FRAME_CORRECTION"):
            official_rows = branch_rows_by_event[(event_id, branch)]
            mapped_rows = {int(row["frame"]): row for row in mapping_by_event[(event_id, branch)]}
            horizon_result: dict[str, Any] = {}
            for horizon in HORIZONS:
                evaluated = 0
                target_present = 0
                candidate_present = 0
                best_iou_sum = 0.0
                frame_details: list[dict[str, Any]] = []
                for frame in range(event_frame + 1, event_frame + horizon + 1):
                    gt = gt_frames.get(frame, {}).get(target_gid)
                    if gt is None:
                        continue
                    evaluated += 1
                    target_present += 1
                    official_candidates = official_rows[frame]["candidates"]
                    best_iou = max((box_iou(candidate["box_xyxy"], gt["box"]) for candidate in official_candidates), default=0.0)
                    target_row = mapped_rows[frame]
                    target_iou = max(
                        (
                            box_iou(candidate["box_xyxy"], gt["box"])
                            for candidate in target_row["candidate_rows"]
                            if candidate.get("public_id") is not None
                            and int(candidate["public_id"]) == int(target_row["target_public_id"])
                        ),
                        default=0.0,
                    )
                    candidate_hit = best_iou >= IOU_THRESHOLD
                    candidate_present += int(candidate_hit)
                    best_iou_sum += float(best_iou)
                    frame_details.append(
                        {
                            "frame": int(frame),
                            "target_gt_present": True,
                            "candidate_best_iou": float(best_iou),
                            "candidate_present": bool(candidate_hit),
                            "mapped_target_iou": float(target_iou),
                            "mapped_target_present": bool(
                                any(
                                    candidate.get("public_id") is not None
                                    and int(candidate["public_id"]) == int(target_row["target_public_id"])
                                    for candidate in target_row["candidate_rows"]
                                )
                            ),
                            "runtime_future_gt_used": False,
                            "posthoc_gt_used": True,
                        }
                    )
                horizon_result[str(horizon)] = {
                    "evaluated_frames": int(evaluated),
                    "target_gt_present_frames": int(target_present),
                    "candidate_present_frames": int(candidate_present),
                    "candidate_recall": None if evaluated == 0 else float(candidate_present / evaluated),
                    "candidate_best_iou_mean": None if evaluated == 0 else float(best_iou_sum / evaluated),
                    "frame_details": frame_details,
                }
            event_result["horizons"][branch] = horizon_result
        event_details.append(event_result)

    aggregate: dict[str, dict[str, Any]] = {}
    for branch in ("B0_NO_INTERVENTION", "B1_CURRENT_FRAME_CORRECTION"):
        aggregate[branch] = {}
        for horizon in HORIZONS:
            selected = [event["horizons"][branch][str(horizon)] for event in event_details]
            denom = sum(int(value["evaluated_frames"]) for value in selected)
            hits = sum(int(value["candidate_present_frames"]) for value in selected)
            best_iou_sum = sum(
                float(value["candidate_best_iou_mean"]) * int(value["evaluated_frames"])
                for value in selected
                if value["candidate_best_iou_mean"] is not None
            )
            by_sequence = {
                str(event["sequence"]): float(event["horizons"][branch][str(horizon)]["candidate_recall"])
                for event in event_details
                if event["horizons"][branch][str(horizon)]["candidate_recall"] is not None
            }
            aggregate[branch][str(horizon)] = {
                "evaluated_frames": int(denom),
                "candidate_present_frames": int(hits),
                "candidate_recall": None if denom == 0 else float(hits / denom),
                "candidate_best_iou_mean": None if denom == 0 else float(best_iou_sum / denom),
                "event_count": len(selected),
                "independent_sequence_count": len(by_sequence),
                "sequence_values": by_sequence,
            }
    delta: dict[str, Any] = {}
    for horizon in HORIZONS:
        no = aggregate["B0_NO_INTERVENTION"][str(horizon)]["candidate_recall"]
        m0 = aggregate["B1_CURRENT_FRAME_CORRECTION"][str(horizon)]["candidate_recall"]
        delta[str(horizon)] = None if no is None or m0 is None else float(m0 - no)
    action_breakdown: dict[str, Any] = {}
    for action in sorted({str(event["action_type"]) for event in event_details}):
        action_breakdown[action] = {}
        for branch in ("B0_NO_INTERVENTION", "B1_CURRENT_FRAME_CORRECTION"):
            action_breakdown[action][branch] = {}
            for horizon in HORIZONS:
                selected = [event for event in event_details if event["action_type"] == action]
                values = [event["horizons"][branch][str(horizon)] for event in selected]
                denom = sum(int(value["evaluated_frames"]) for value in values)
                hits = sum(int(value["candidate_present_frames"]) for value in values)
                action_breakdown[action][branch][str(horizon)] = {
                    "evaluated_frames": int(denom),
                    "candidate_present_frames": int(hits),
                    "candidate_recall": None if denom == 0 else float(hits / denom),
                    "event_count": len(selected),
                    "independent_sequence_count": len({str(event["sequence"]) for event in selected}),
                }
    return {
        "schema_version": "N72R4_STAGE10_12_CANDIDATE_RECALL_V1",
        "status": "PASS_STAGE10_NO_VS_M0_POSTHOC_RECALL",
        "events": event_details,
        "aggregate": aggregate,
        "m0_minus_no_candidate_recall": delta,
        "by_action": action_breakdown,
        "thresholds": {"candidate_iou": IOU_THRESHOLD, "future_horizons": list(HORIZONS)},
        "official_pair_manifest": str(PAIR_MANIFEST),
        "runtime_future_gt_used": False,
        "gt_usage": "posthoc_only_after_official_runtime_validation",
        "scientific_result": "NO_VS_M0_CANDIDATE_AVAILABILITY_ONLY_NOT_FUTURE_EFFECT_GATE",
    }


def freeze_protocol() -> Path:
    path = OUT / "protocol.json"
    if path.exists():
        return path
    payload = {
        "schema_version": "N72R4_PROTOCOL_V1",
        "created_at_utc": now_utc(),
        "source": "N72R3_frozen_six_event_protocol",
        "event_manifest": str(EVENT_MANIFEST),
        "official_pair_manifest": str(PAIR_MANIFEST),
        "interaction_source": "simulated_from_gt",
        "real_human_tape": False,
        "candidate_iou_threshold": IOU_THRESHOLD,
        "future_horizons": list(HORIZONS),
        "mapping": {
            "authority": "N72R3_STAGE18_PERSISTENT_RUNTIME_PRESTATE",
            "correction_target_anchor": "official_current_correction_post_observation_box_max_iou",
            "future_solver": "frozen_pairwise_geometry_feature_native_cue_with_explicit_NONE",
            "similarity_weight": SIM_WEIGHT,
            "iou_weight": IOU_WEIGHT,
            "native_weight": NATIVE_WEIGHT,
            "native_bonus": NATIVE_BONUS,
            "gap_weight": GAP_WEIGHT,
            "none_score": NONE_SCORE,
            "candidate_index_to_public_id": False,
            "official_raw_sam_id_to_public_id": False,
            "adapter_id_to_public_id": False,
            "new_public_ids_created": False,
        },
        "post_treatment_fields_for_assignment": [],
        "runtime_future_gt_used": False,
        "protocol_freeze_reason": "protocol is fixed before posthoc GT scoring; no future outcome is used for mapping",
    }
    atomic_json(path, payload)
    return path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit-path", type=Path, default=DEFAULT_AUDIT)
    parser.add_argument("--recall-path", type=Path, default=DEFAULT_RECALL)
    parser.add_argument("--status-path", type=Path, default=DEFAULT_STATUS)
    args = parser.parse_args()
    started = now_utc()
    try:
        protocol_path = freeze_protocol()
        pair = read_json(PAIR_MANIFEST)
        if pair.get("status") != "PASS_OFFICIAL_PAIRED_FUTURE_STREAM" or pair.get("official_future_propagation") is not True:
            raise RuntimeError("Stage09 attempt2 official paired manifest is not a passing future stream")
        if pair.get("runtime_future_gt_used") is not False or pair.get("public_id_inference") is not False:
            raise RuntimeError("Stage09 official paired manifest violates public/GT boundary")
        events = load_events()
        branch_rows_by_event: dict[tuple[str, str], dict[int, dict[str, Any]]] = {}
        mapping_by_event: dict[tuple[str, str], list[dict[str, Any]]] = {}
        mapping_summaries: list[dict[str, Any]] = []
        audit_rows: list[dict[str, Any]] = []
        for event in events:
            event_id = str(event["event_id"])
            stage16 = load_stage16(event_id)
            public_by_raw = load_stage18_public_map(str(event["sequence"]), int(event["event_frame"]))
            no_rows = load_branch_rows(NO_ROOT, event_id)
            m0_rows = load_branch_rows(M0_ROOT, event_id)
            event_frame = int(event["event_frame"])
            if no_rows[event_frame]["y_pre_semantic_hash"] != m0_rows[event_frame]["y_pre_semantic_hash"]:
                raise RuntimeError(f"NO/M0 event-frame candidate stream mismatch: {event_id}")
            for branch, rows in (("B0_NO_INTERVENTION", no_rows), ("B1_CURRENT_FRAME_CORRECTION", m0_rows)):
                branch_rows_by_event[(event_id, branch)] = rows
            no_mapped, no_summary = map_branch("B0_NO_INTERVENTION", event, no_rows, stage16, public_by_raw)
            m0_mapped, m0_summary = map_branch("B1_CURRENT_FRAME_CORRECTION", event, m0_rows, stage16, public_by_raw)
            mapping_by_event[(event_id, "B0_NO_INTERVENTION")] = no_mapped
            mapping_by_event[(event_id, "B1_CURRENT_FRAME_CORRECTION")] = m0_mapped
            mapping_summaries.extend([no_summary, m0_summary])
            for mapped in no_mapped + m0_mapped:
                audit_rows.append(mapped)
        recall = score_recall(events, branch_rows_by_event, mapping_by_event)
        recall["inputs"] = {
            "pair_manifest_sha256": sha256_file(PAIR_MANIFEST),
            "event_manifest_sha256": sha256_file(EVENT_MANIFEST),
            "stage16_event_count": len(events),
            "official_no_root": str(NO_ROOT),
            "official_corrected_root": str(M0_ROOT),
        }
        atomic_json(args.recall_path, recall)
        atomic_jsonl(args.audit_path, audit_rows)
        protocol_hash = sha256_file(protocol_path)
        status = {
            "schema_version": "N72R4_STAGE_STATUS_V1",
            "stage": "10_NO_VS_M0_DECOMPOSITION_AND_12_CANDIDATE_RECALL",
            "status": "PASS_STAGE10_NO_VS_M0_POSTHOC_RECALL",
            "started_at_utc": started,
            "finished_at_utc": now_utc(),
            "event_count": len(events),
            "independent_sequence_count": len({str(event["sequence"]) for event in events}),
            "official_pair_manifest": str(PAIR_MANIFEST),
            "official_pair_manifest_sha256": sha256_file(PAIR_MANIFEST),
            "protocol": str(protocol_path),
            "protocol_sha256": protocol_hash,
            "public_mapping_audit": str(args.audit_path),
            "candidate_recall": str(args.recall_path),
            "mapping_summaries": mapping_summaries,
            "candidate_index_to_public_id": False,
            "official_raw_sam_id_to_public_id": False,
            "adapter_id_to_public_id": False,
            "explicit_none_retained": True,
            "runtime_future_gt_used": False,
            "posthoc_gt_used": True,
            "interaction_source": "simulated_from_gt",
            "not_real_human_evidence": True,
            "scientific_result": "NO_VS_M0_CANDIDATE_AVAILABILITY_ONLY_NOT_FUTURE_EFFECT_GATE",
        }
        atomic_json(args.status_path, status)
        print(
            json.dumps(
                {
                    "status": status["status"],
                    "events": len(events),
                    "audit_rows": len(audit_rows),
                    "recall_path": str(args.recall_path),
                },
                sort_keys=True,
            ),
            flush=True,
        )
        return 0
    except Exception as exc:
        failure_root = OUT / "attempts" / "stage10"
        failure_root.mkdir(parents=True, exist_ok=True)
        failure_path = failure_root / f"stage10_failure_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.json"
        atomic_json(
            failure_path,
            {
                "schema_version": "N72R4_FAILURE_V1",
                "stage": "10_NO_VS_M0_DECOMPOSITION_AND_12_CANDIDATE_RECALL",
                "status": "FAIL",
                "started_at_utc": started,
                "finished_at_utc": now_utc(),
                "error_type": type(exc).__name__,
                "error": str(exc),
                "traceback": __import__("traceback").format_exc(),
                "runtime_future_gt_used": False,
            },
        )
        print(json.dumps({"status": "FAIL", "failure_artifact": str(failure_path), "error": str(exc)}, sort_keys=True), flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
