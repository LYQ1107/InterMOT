"""Shared loaders and post-hoc-safe helpers for N72R20R3."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any, Mapping, Sequence

import numpy as np

from sam3_intermot.association.online_associator import predicted_iou
from sam3_intermot.association.learned_identity_memory import LearnedIdentityMemoryBank
from scripts.n72r20r1_build_base_score_tape import (
    load_candidate_frames,
    read_json,
    read_zstd_jsonl,
    unit,
)
from scripts.n72r20r1_build_base_score_tape import (
    initialize_states as initialize_base_states,
    update_base_states,
)
from scripts.n72r20r2_forensics import gt_by_frame, iou


ROOT = Path(__file__).resolve().parents[1]
R2_ASSET_ROOT = Path("/data3/liuyeqiang/InterMOT_N72R20R2_assets")
DATASET_ROOT = Path("/data3/liuyeqiang/InterMOT_N72R16_assets/dataset")
CHECKPOINT = ROOT / "outputs/N72R18/checkpoints/identity_memory_gru.pt"
CHECKPOINT_SHA = "94ba9c44e177b254ee54984195e839f37c78a648c010ffc723dd3f1d553976a2"
ENCODER_SHA = "2809d3227f7d078f6045f7feb874a34d0684f0e0057b264b99adccf7d4519154"
SEQUENCES = (
    "dancetrack0001",
    "dancetrack0002",
    "dancetrack0023",
    "dancetrack0024",
    "dancetrack0039",
    "dancetrack0057",
    "dancetrack0062",
    "dancetrack0072",
)
IOU_THRESHOLD = 0.50


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_zstd_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    payload = "".join(
        json.dumps(dict(row), sort_keys=True, allow_nan=False, separators=(",", ":")) + "\n"
        for row in rows
    ).encode("utf-8")
    completed = subprocess.run(
        ["zstd", "-q", "-T0", "-c"],
        input=payload,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(completed.stdout)


def read_events(asset_root: Path = R2_ASSET_ROOT) -> dict[str, dict[str, Any]]:
    payload = read_json(asset_root / "interaction_events.json")
    return {str(item["sequence"]): dict(item) for item in payload["events"]}


def read_labels(asset_root: Path = R2_ASSET_ROOT) -> dict[str, dict[str, Any]]:
    payload = read_json(asset_root / "posthoc_event_labels.json")
    return {str(item["sequence"]): dict(item) for item in payload["labels"]}


def load_sequence(asset_root: Path, sequence: str) -> tuple[list[tuple[dict[str, Any], list[dict[str, Any]]]], list[dict[str, Any]]]:
    frames = load_candidate_frames(asset_root, sequence)
    base_rows = read_zstd_jsonl(asset_root / "base_scores" / sequence / "base_scores.jsonl.zst")
    if len(frames) != len(base_rows):
        raise ValueError(f"{sequence}: candidate/base frame count mismatch")
    for (frame_row, candidates), base_row in zip(frames, base_rows):
        if int(frame_row["frame"]) != int(base_row["frame"]):
            raise ValueError(f"{sequence}: candidate/base frame axis mismatch")
        if base_row.get("runtime_future_gt_used") is not False:
            raise ValueError(f"{sequence}: base tape has runtime GT")
        if len(candidates) != int(base_row.get("candidate_count", len(candidates))):
            raise ValueError(f"{sequence}: candidate count mismatch")
    return frames, base_rows


def initialize_learned_bank(event: Mapping[str, Any]) -> LearnedIdentityMemoryBank:
    bank = LearnedIdentityMemoryBank.from_checkpoint(
        CHECKPOINT,
        encoder_sha256=ENCODER_SHA,
        expected_encoder_sha256=ENCODER_SHA,
        expected_checkpoint_sha256=CHECKPOINT_SHA,
    )
    bank.initialize_public_identity(
        public_id=int(event["target_public_id"]),
        association_state_id=int(event["target_association_state_id"]),
        frame=int(event["event_frame"]),
        human_anchor=unit(event["human_anchor"], "human anchor"),
        interaction_source="simulated_from_gt",
    )
    return bank


def public_assignment_item(solver: Mapping[str, Any], public_id: int) -> dict[str, Any]:
    for item in solver.get("public_assignments", []):
        if int(item["public_id"]) == int(public_id):
            return dict(item)
    raise ValueError(f"public ID {public_id} is absent from solver")


def public_assignment_uid(solver: Mapping[str, Any], public_id: int) -> str | None:
    item = public_assignment_item(solver, public_id)
    uid = item.get("candidate_uid")
    return None if uid in (None, "", "None") else str(uid)


def candidate_by_uid(candidates: Sequence[Mapping[str, Any]]) -> dict[str, Mapping[str, Any]]:
    result = {str(row["candidate_uid"]): row for row in candidates}
    if len(result) != len(candidates):
        raise ValueError("candidate UID collision")
    return result


def target_iou_for_uid(
    uid: str | None,
    candidates: Sequence[Mapping[str, Any]],
    target_boxes: Sequence[Sequence[float]],
) -> float | None:
    if uid is None:
        return None
    row = candidate_by_uid(candidates).get(str(uid))
    if row is None:
        return None
    return float(max((iou(row["box_xyxy"], box) for box in target_boxes), default=0.0))


def best_target_candidate(
    candidates: Sequence[Mapping[str, Any]],
    target_boxes: Sequence[Sequence[float]],
) -> tuple[str | None, float | None]:
    if not candidates or not target_boxes:
        return None, None
    best = max(
        ((target_iou_for_uid(str(row["candidate_uid"]), candidates, target_boxes) or 0.0, str(row["candidate_uid"])) for row in candidates),
        key=lambda item: (float(item[0]), item[1]),
    )
    return best[1], float(best[0])


def target_boxes_for_frame(gt: Mapping[int, Sequence[tuple[int, Sequence[float]]]], frame: int, target_gt_id: int) -> list[list[float]]:
    return [list(box) for track_id, box in gt.get(int(frame), []) if int(track_id) == int(target_gt_id)]


def base_assignment_margin(base_row: Mapping[str, Any], public_id: int, association_state_id: int | None = None) -> float | None:
    uid = public_assignment_uid(base_row["base_assignment"], public_id)
    if uid is None:
        return None
    axis = [int(value) for value in base_row["association_state_axis"]]
    matrix = np.asarray(base_row["base_score_matrix"], dtype=np.float64)
    try:
        column = axis.index(int(public_id if association_state_id is None else association_state_id))
    except ValueError:
        return None
    candidates = [str(value) for value in base_row["candidate_uid_axis"]]
    if uid not in candidates:
        return None
    row_index = candidates.index(uid)
    assigned = float(matrix[row_index, column])
    alternatives = [0.0] + [float(matrix[index, column]) for index in range(matrix.shape[0]) if index != row_index]
    return assigned - max(alternatives)


def target_base_context(
    *,
    states: Mapping[int, Any],
    candidates: Sequence[Mapping[str, Any]],
    base_row: Mapping[str, Any],
    public_id: int,
    association_state_id: int,
    frame: int,
) -> dict[str, Any]:
    uid = public_assignment_uid(base_row["base_assignment"], public_id)
    by_uid = candidate_by_uid(candidates)
    state = states[int(public_id)]
    predicted = None
    native = None
    if uid is not None and uid in by_uid:
        candidate = by_uid[uid]
        predicted = float(predicted_iou(state, np.asarray(candidate["box_xyxy"], dtype=np.float64), int(frame)))
        native = bool(
            int(candidate.get("native_tid", candidate.get("raw_native_id", -1))) == int(state.last_native_tid)
            and state.last_native_scope in (None, "")
            and candidate.get("native_scope") in (None, "")
        )
    return {
        "base_assignment_margin": base_assignment_margin(base_row, public_id, association_state_id),
        "predicted_motion_iou_of_base_candidate": predicted,
        "native_continuity_of_base_candidate": native,
        "base_candidate_present": uid in by_uid if uid is not None else False,
    }


def update_base_for_frame(
    *,
    states: Mapping[int, Any],
    event: Mapping[str, Any],
    candidates: list[dict[str, Any]],
    base_solver: Mapping[str, Any],
    frame: int,
) -> dict[str, Any]:
    return update_base_states(
        states=states,
        solver=base_solver,
        rows=candidates,
        frame=int(frame),
        target_public=int(event["target_public_id"]),
        event_frame=int(event["event_frame"]),
    )


__all__ = [
    "CHECKPOINT",
    "CHECKPOINT_SHA",
    "DATASET_ROOT",
    "ENCODER_SHA",
    "IOU_THRESHOLD",
    "R2_ASSET_ROOT",
    "ROOT",
    "SEQUENCES",
    "best_target_candidate",
    "candidate_by_uid",
    "gt_by_frame",
    "initialize_base_states",
    "initialize_learned_bank",
    "iou",
    "load_sequence",
    "public_assignment_item",
    "public_assignment_uid",
    "read_events",
    "read_labels",
    "read_json",
    "read_zstd_jsonl",
    "sha256",
    "target_base_context",
    "target_boxes_for_frame",
    "target_iou_for_uid",
    "unit",
    "update_base_for_frame",
    "write_zstd_jsonl",
]
