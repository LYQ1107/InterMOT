#!/usr/bin/env python3
"""Forensic reconstruction of the sealed N72R20R1 train-smoke lineage.

The replay reconstructs the frozen base and N72R18 learned-memory state without
opening GT.  A separate post-hoc join adds target/competitor labels, rescue
categories, contamination details, and exact-solver decision-boundary probes.
No candidate tape, solver, public-ID authority, or model is modified.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any, Mapping, Sequence

import numpy as np

from sam3_intermot.association.effect_assignment import solve_effect_assignment
from sam3_intermot.association.learned_identity_memory import LearnedIdentityMemoryBank
from sam3_intermot.association.learned_identity_state_edge import (
    build_learned_identity_state_edge_matrix,
)
from sam3_intermot.association.online_associator import (
    native_same,
    predicted_iou,
    score_matrix_pairwise,
)
from scripts.n72r20r1_build_base_score_tape import (
    assignable_states,
    initialize_states,
    load_candidate_frames,
    read_json,
    read_zstd_jsonl,
    unit,
    update_base_states,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ASSET_ROOT = Path("/data3/liuyeqiang/InterMOT_N72R20R1_assets")
DEFAULT_DATASET_ROOT = Path("/data3/liuyeqiang/InterMOT_N72R16_assets/dataset")
CHECKPOINT = ROOT / "outputs/N72R18/checkpoints/identity_memory_gru.pt"
CHECKPOINT_SHA = "94ba9c44e177b254ee54984195e839f37c78a648c010ffc723dd3f1d553976a2"
ENCODER_SHA = "2809d3227f7d078f6045f7feb874a34d0684f0e0057b264b99adccf7d4519154"
IOU_THRESHOLD = 0.50
HORIZONS = (20, 50, 100)
BASE_REID_WEIGHTS = {"sim": 1.5, "iou": 1.0, "native": 0.5, "gap": 0.1}
BASE_NATIVE_BONUS = 3.0
BASE_POSITIVE_BONUS = 5.0
BASE_EMA = 0.9


@dataclass(frozen=True)
class AxisState:
    association_state_id: int
    public_id: int


def iou(left: Any, right: Any) -> float:
    a = np.asarray(left, dtype=np.float64)
    b = np.asarray(right, dtype=np.float64)
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    intersection = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - intersection
    return float(intersection / union) if union > 0.0 else 0.0


def cosine(left: Any, right: Any) -> float:
    a = np.asarray(left, dtype=np.float32).reshape(-1)
    b = np.asarray(right, dtype=np.float32).reshape(-1)
    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(a, b) / denominator) if denominator > 1.0e-8 else 0.0


def stats(values: Sequence[float]) -> dict[str, Any]:
    finite = np.asarray([float(value) for value in values if value is not None and np.isfinite(value)], dtype=np.float64)
    if finite.size == 0:
        return {"count": 0, "min": None, "median": None, "p75": None, "p90": None, "p95": None, "max": None, "mean": None}
    return {
        "count": int(finite.size),
        "min": float(np.min(finite)),
        "median": float(np.median(finite)),
        "p75": float(np.percentile(finite, 75)),
        "p90": float(np.percentile(finite, 90)),
        "p95": float(np.percentile(finite, 95)),
        "max": float(np.max(finite)),
        "mean": float(np.mean(finite)),
    }


def gt_by_frame(path: Path) -> dict[int, list[tuple[int, list[float]]]]:
    result: dict[int, list[tuple[int, list[float]]]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        values = [item.strip() for item in line.split(",")]
        if len(values) < 6:
            continue
        frame = int(float(values[0])) - 1
        track_id = int(float(values[1]))
        x, y, width, height = (float(item) for item in values[2:6])
        result.setdefault(frame, []).append((track_id, [x, y, x + width, y + height]))
    return result


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_zstd_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    payload = "".join(json.dumps(dict(row), sort_keys=True, allow_nan=False) + "\n" for row in rows).encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    compressed = subprocess.run(
        ["zstd", "-q", "-T0", "-c"], input=payload, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True
    )
    path.write_bytes(compressed.stdout)


def assignment_for_public(solver: Mapping[str, Any], public_id: int) -> dict[str, Any]:
    for item in solver.get("public_assignments", []):
        if int(item["public_id"]) == int(public_id):
            return dict(item)
    raise ValueError(f"public ID {public_id} missing from solver artifact")


def candidate_order(scores: np.ndarray, candidates: Sequence[Mapping[str, Any]]) -> list[int]:
    return sorted(range(len(candidates)), key=lambda index: (-float(scores[index]), str(candidates[index]["candidate_uid"])))


def rank_for_uid(scores: np.ndarray, candidates: Sequence[Mapping[str, Any]], uid: str | None) -> int | None:
    if uid is None:
        return None
    axis = {str(row["candidate_uid"]): index for index, row in enumerate(candidates)}
    index = axis.get(str(uid))
    if index is None:
        return None
    return int(candidate_order(scores, candidates).index(index) + 1)


def target_iou(candidate: Mapping[str, Any] | None, target_boxes: Sequence[Sequence[float]]) -> float | None:
    if candidate is None:
        return None
    return float(max((iou(candidate["box_xyxy"], box) for box in target_boxes), default=0.0))


def posthoc_target_candidate(candidates: Sequence[Mapping[str, Any]], target_boxes: Sequence[Sequence[float]]) -> tuple[dict[str, Any] | None, float | None]:
    if not target_boxes or not candidates:
        return None, None
    ranked = sorted(
        ((float(max(iou(row["box_xyxy"], box) for box in target_boxes)), row) for row in candidates),
        key=lambda item: (-item[0], str(item[1]["candidate_uid"])),
    )
    value, row = ranked[0]
    return row, value


def hard_negative_candidates(
    candidates: Sequence[Mapping[str, Any]],
    all_gt: Sequence[tuple[int, Sequence[float]]],
    target_gt_id: int,
    target_uid: str | None,
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for candidate in candidates:
        if target_uid is not None and str(candidate["candidate_uid"]) == str(target_uid):
            continue
        overlap = max(
            (iou(candidate["box_xyxy"], box) for track_id, box in all_gt if int(track_id) != int(target_gt_id)),
            default=0.0,
        )
        if overlap >= IOU_THRESHOLD:
            result.append(dict(candidate))
    return result


def simple_candidate_rows(candidates: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [
        {"candidate_uid": str(row["candidate_uid"]), "candidate_index": int(row.get("candidate_index", index))}
        for index, row in enumerate(candidates)
    ]


def solver_target_uid(artifact: Mapping[str, Any], public_id: int) -> str | None:
    item = assignment_for_public(artifact, public_id)
    uid = item.get("candidate_uid")
    return None if uid in (None, "", "None") else str(uid)


def solve_with_target_delta(
    *,
    base_matrix: np.ndarray,
    candidates: Sequence[Mapping[str, Any]],
    state_axis: Sequence[int],
    public_axis: Sequence[int],
    target_public: int,
    target_candidate_index: int,
    delta: float,
    frame: int,
    sequence: str,
) -> str | None:
    modified = np.asarray(base_matrix, dtype=np.float64).copy()
    target_col = list(public_axis).index(int(target_public))
    modified[target_candidate_index, target_col] += float(delta)
    states = [AxisState(int(state), int(public)) for state, public in zip(state_axis, public_axis)]
    artifact = solve_effect_assignment(
        candidate_rows=simple_candidate_rows(candidates),
        persistent_states=states,
        fused_state_candidate_scores=modified.T,
        source_run_id=f"n72r20r2:boundary:{sequence}:{frame}:{delta:.8f}",
        session_id=f"n72r20r2:{sequence}",
        none_score=0.0,
    )
    return solver_target_uid(artifact, target_public)


def required_target_delta(
    *,
    base_matrix: np.ndarray,
    candidates: Sequence[Mapping[str, Any]],
    state_axis: Sequence[int],
    public_axis: Sequence[int],
    target_public: int,
    desired_uid: str,
    frame: int,
    sequence: str,
) -> float | None:
    uid_to_index = {str(row["candidate_uid"]): index for index, row in enumerate(candidates)}
    target_index = uid_to_index.get(str(desired_uid))
    if target_index is None:
        return None
    if solve_with_target_delta(
        base_matrix=base_matrix,
        candidates=candidates,
        state_axis=state_axis,
        public_axis=public_axis,
        target_public=target_public,
        target_candidate_index=target_index,
        delta=0.0,
        frame=frame,
        sequence=sequence,
    ) == str(desired_uid):
        return 0.0
    high = 1.0
    while high <= 256.0:
        if solve_with_target_delta(
            base_matrix=base_matrix,
            candidates=candidates,
            state_axis=state_axis,
            public_axis=public_axis,
            target_public=target_public,
            target_candidate_index=target_index,
            delta=high,
            frame=frame,
            sequence=sequence,
        ) == str(desired_uid):
            low = 0.0
            for _ in range(42):
                middle = (low + high) / 2.0
                if solve_with_target_delta(
                    base_matrix=base_matrix,
                    candidates=candidates,
                    state_axis=state_axis,
                    public_axis=public_axis,
                    target_public=target_public,
                    target_candidate_index=target_index,
                    delta=middle,
                    frame=frame,
                    sequence=sequence,
                ) == str(desired_uid):
                    high = middle
                else:
                    low = middle
            return float(high)
        high *= 2.0
    return None


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


def contiguous_runs(frames: Sequence[int], max_gap: int) -> list[dict[str, Any]]:
    ordered = sorted(int(frame) for frame in frames)
    if not ordered:
        return []
    runs: list[list[int]] = [[ordered[0]]]
    for frame in ordered[1:]:
        if frame - runs[-1][-1] <= int(max_gap):
            runs[-1].append(frame)
        else:
            runs.append([frame])
    return [
        {
            "start_frame": int(run[0]),
            "end_frame": int(run[-1]),
            "observed_frames": [int(value) for value in run],
            "observed_count": len(run),
            "span": int(run[-1] - run[0] + 1),
            "max_gap": int(max_gap),
        }
        for run in runs
    ]


def state_vectors_cascade(
    rows: Sequence[Mapping[str, Any]],
    vectors_before: Mapping[tuple[str, int], np.ndarray],
    vectors_after: Mapping[tuple[str, int], np.ndarray],
) -> dict[str, Any]:
    by_sequence: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_sequence.setdefault(str(row["sequence"]), []).append(dict(row))
    output: dict[str, Any] = {}
    all_wrong_details: list[dict[str, Any]] = []
    for sequence, sequence_rows in sorted(by_sequence.items()):
        sequence_rows.sort(key=lambda item: int(item["frame"]))
        wrong = [item for item in sequence_rows if item.get("base_correct") is False]
        wrong_frames = [int(item["frame"]) for item in wrong]
        first_wrong = wrong[0] if wrong else None
        first_frame = None if first_wrong is None else int(first_wrong["frame"])
        exact = contiguous_runs(wrong_frames, 1)
        near = contiguous_runs(wrong_frames, 2)
        details: list[dict[str, Any]] = []
        for item in wrong:
            key = (sequence, int(item["frame"]))
            before = vectors_before.get(key)
            after = vectors_after.get(key)
            details.append(
                {
                    "sequence": sequence,
                    "frame": int(item["frame"]),
                    "base_candidate_uid": item.get("base_target_candidate_uid"),
                    "base_candidate_iou": item.get("base_candidate_target_iou"),
                    "learned_rank_of_base_candidate": item.get("learned_rank_of_base_candidate"),
                    "base_candidate_is_learned_top1": item.get("base_candidate_is_learned_top1"),
                    "learned_margin": item.get("learned_margin"),
                    "learned_score_of_assigned_candidate": item.get("learned_score_of_base_candidate"),
                    "learned_score_of_posthoc_target_candidate": item.get("learned_score_of_posthoc_target_candidate"),
                    "posthoc_target_candidate_uid": item.get("posthoc_target_candidate_uid"),
                    "human_anchor_score_of_assigned_candidate": item.get("human_anchor_score_of_base_candidate"),
                    "native_continuity": item.get("native_continuity"),
                    "predicted_motion_iou": item.get("predicted_motion_iou"),
                    "presence": item.get("base_candidate_presence"),
                    "candidate_count": item.get("candidate_count"),
                    "previous_state_hash": item.get("memory_state_hash"),
                    "state_hash_after_write": item.get("memory_state_hash_after"),
                    "state_similarity_before_after": item.get("memory_state_similarity_before_after"),
                    "frames_since_last_safe_write": item.get("frames_since_last_committed_write"),
                    "target_vs_hard_negative_margin": item.get("learned_target_vs_hard_negative_margin"),
                    "state_similarity_to_first_wrong_after": None
                    if first_wrong is None or after is None
                    else cosine(after, vectors_after.get((sequence, first_frame), after)),
                }
            )
        all_wrong_details.extend(details)
        first_key = None if first_wrong is None else (sequence, first_frame)
        first_after = None if first_key is None else vectors_after.get(first_key)
        before_margins = [item.get("learned_target_vs_hard_negative_margin") for item in sequence_rows if first_frame is not None and int(item["frame"]) < first_frame]
        after_margins = [item.get("learned_target_vs_hard_negative_margin") for item in sequence_rows if first_frame is not None and int(item["frame"]) > first_frame]
        state_sim_after_first = []
        if first_after is not None and first_frame is not None:
            for item in sequence_rows:
                if int(item["frame"]) > first_frame:
                    vector = vectors_after.get((sequence, int(item["frame"])))
                    if vector is not None:
                        state_sim_after_first.append(cosine(first_after, vector))
        output[sequence] = {
            "wrong_write_count": len(wrong),
            "wrong_write_rate_over_future_frames": None if not sequence_rows else len(wrong) / len(sequence_rows),
            "first_wrong_write_frame": first_frame,
            "exact_contiguous_runs": exact,
            "near_contiguous_runs_gap_le_2": near,
            "first_wrong_state_similarity_before_after": None if first_wrong is None else first_wrong.get("memory_state_similarity_before_after"),
            "first_wrong_target_vs_hard_negative_margin": None if first_wrong is None else first_wrong.get("learned_target_vs_hard_negative_margin"),
            "target_vs_hard_negative_margin_before_first_wrong": stats([value for value in before_margins if value is not None]),
            "target_vs_hard_negative_margin_after_first_wrong": stats([value for value in after_margins if value is not None]),
            "state_similarity_to_first_wrong_after": stats(state_sim_after_first),
            "later_wrong_write_count": sum(int(item["frame"]) > int(first_frame) for item in wrong) if first_frame is not None else 0,
            "wrong_write_details": details,
        }
    return {"sequences": output, "wrong_write_details": all_wrong_details}


def markdown_cascade(payload: Mapping[str, Any]) -> str:
    lines = [
        "# N72R20R2 contamination cascade analysis",
        "",
        "This is a post-hoc join against the sealed R1 train-smoke tape. Runtime",
        "reconstruction used no GT; GT only labels the recorded writes afterward.",
        "The wrong-write criterion is target-candidate IoU < 0.50.",
        "",
        "| Sequence | Wrong writes | Rate | First wrong frame | Exact runs | Near runs (gap≤2) |",
        "|---|---:|---:|---:|---|---|",
    ]
    for sequence, item in payload["sequences"].items():
        exact = ", ".join(f"{run['start_frame']}-{run['end_frame']}" for run in item["exact_contiguous_runs"]) or "none"
        near = ", ".join(f"{run['start_frame']}-{run['end_frame']}" for run in item["near_contiguous_runs_gap_le_2"]) or "none"
        rate = "—" if item["wrong_write_rate_over_future_frames"] is None else f"{100.0 * item['wrong_write_rate_over_future_frames']:.3f}%"
        first = "—" if item["first_wrong_write_frame"] is None else str(item["first_wrong_write_frame"])
        lines.append(f"| {sequence} | {item['wrong_write_count']} | {rate} | {first} | {exact} | {near} |")
    lines.extend(
        [
            "",
            "The detailed JSON records learned rank, learned margin, anchor score,",
            "native continuity, predicted motion IoU, state hashes, state drift,",
            "and target-vs-hard-negative margins for every wrong write.",
            "",
            "Interpretation: the R1 assignment shadow did not change public",
            "assignments. These are base-assignment observations admitted by a",
            "non-discriminative consensus rule, not errors caused by a learned",
            "assignment change.",
        ]
    )
    return "\n".join(lines) + "\n"


def process_sequence(
    *,
    asset_root: Path,
    dataset_root: Path,
    sequence: str,
    event: Mapping[str, Any],
    label: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[tuple[str, int], np.ndarray], dict[tuple[str, int], np.ndarray]]:
    candidate_frames = load_candidate_frames(asset_root, sequence)
    base_rows = read_zstd_jsonl(asset_root / "base_scores" / sequence / "base_scores.jsonl.zst")
    if len(candidate_frames) != len(base_rows):
        raise ValueError(f"{sequence}: candidate/base frame count mismatch")
    gt = gt_by_frame(dataset_root / "train" / sequence / "gt" / "gt.txt")
    target_public = int(event["target_public_id"])
    target_gt_id = int(label["target_gt_id"])
    public_axis = [int(value) for value in base_rows[0]["public_id_axis"]]
    state_axis = [int(value) for value in base_rows[0]["association_state_axis"]]
    target_col = public_axis.index(target_public)
    base_states = initialize_states(event, candidate_frames[0][1])
    ordered_states = assignable_states([dict(item) for item in event["state_bindings"]], base_states)
    learned_bank = initialize_learned_bank(event)
    anchor = unit(event["human_anchor"], "human anchor")
    table: list[dict[str, Any]] = []
    vectors_before: dict[tuple[str, int], np.ndarray] = {}
    vectors_after: dict[tuple[str, int], np.ndarray] = {}
    boundary_rows: list[dict[str, Any]] = []
    base_score_values: list[float] = []
    learned_score_values: list[float] = []
    relative_delta_values: list[float] = []
    reconstruction_failures = 0
    for (frame_payload, candidates), base_row in zip(candidate_frames, base_rows):
        frame = int(frame_payload["frame"])
        if int(base_row["frame"]) != frame:
            raise ValueError(f"{sequence}:{frame}: frame axis mismatch")
        base_matrix = np.asarray(base_row["base_score_matrix"], dtype=np.float64)
        score_audit: dict[str, Any] = {}
        reconstructed = np.asarray(
            score_matrix_pairwise(
                ordered_states,
                candidates,
                frame,
                model=None,
                reid_weights=BASE_REID_WEIGHTS,
                positive_bonus=BASE_POSITIVE_BONUS,
                native_bonus=BASE_NATIVE_BONUS,
                score_audit=score_audit,
            ),
            dtype=np.float64,
        )
        if reconstructed.shape != base_matrix.shape or not np.allclose(reconstructed, base_matrix, atol=1.0e-5, rtol=0.0):
            reconstruction_failures += 1
        if frame == int(event["event_frame"]):
            update_base_states(
                states=base_states,
                solver=base_row["base_assignment"],
                rows=candidates,
                frame=frame,
                target_public=target_public,
                event_frame=int(event["event_frame"]),
            )
            continue

        target_state = base_states[target_public]
        base_item = assignment_for_public(base_row["base_assignment"], target_public)
        base_uid = None if base_item.get("candidate_uid") in (None, "", "None") else str(base_item["candidate_uid"])
        by_uid = {str(row["candidate_uid"]): row for row in candidates}
        uid_to_index = {str(row["candidate_uid"]): index for index, row in enumerate(candidates)}
        base_candidate = by_uid.get(base_uid) if base_uid is not None else None
        target_boxes = [box for track_id, box in gt.get(frame, []) if int(track_id) == target_gt_id]
        all_gt = list(gt.get(frame, []))
        target_candidate, target_candidate_iou = posthoc_target_candidate(candidates, target_boxes)
        target_uid = None if target_candidate is None else str(target_candidate["candidate_uid"])
        hard_negatives = hard_negative_candidates(candidates, all_gt, target_gt_id, target_uid)

        state_hash_before = learned_bank.records[target_public].state_hash()
        anchor_hash = learned_bank.records[target_public].anchor_hash()
        state_vector_before = learned_bank.records[target_public].current_state
        frames_since_write = max(0, frame - int(learned_bank.records[target_public].last_update_frame))
        learned_score_payload = learned_bank.score_matrix(candidate_rows=candidates, public_id_axis=[target_public], frame=frame)
        learned_scores = np.asarray(learned_score_payload["state_candidate_scores"], dtype=np.float64)[0]
        learned_order = candidate_order(learned_scores, candidates)
        learned_top1_index = learned_order[0] if learned_order else None
        learned_top2_index = learned_order[1] if len(learned_order) > 1 else None
        learned_top1_uid = None if learned_top1_index is None else str(candidates[learned_top1_index]["candidate_uid"])
        learned_top2_uid = None if learned_top2_index is None else str(candidates[learned_top2_index]["candidate_uid"])
        learned_top1_score = None if learned_top1_index is None else float(learned_scores[learned_top1_index])
        learned_top2_score = None if learned_top2_index is None else float(learned_scores[learned_top2_index])
        learned_margin = None if learned_top1_score is None or learned_top2_score is None else learned_top1_score - learned_top2_score
        learned_base_score = None if base_candidate is None else float(learned_scores[uid_to_index[base_uid]])
        human_scores = np.asarray([float(np.dot(anchor, unit(row["feature"], "candidate feature"))) for row in candidates], dtype=np.float64)
        learned_target_score = None if target_candidate is None else float(learned_scores[uid_to_index[target_uid]])
        hard_negative_score = max((float(learned_scores[uid_to_index[str(row["candidate_uid"])]]) for row in hard_negatives), default=None)
        target_vs_hard_margin = None if learned_target_score is None or hard_negative_score is None else learned_target_score - hard_negative_score
        target_col_values = base_matrix[:, target_col]
        base_score = float(base_matrix[uid_to_index[base_uid], target_col]) if base_candidate is not None else float(base_row.get("explicit_none_score", 0.0))
        base_order = candidate_order(target_col_values, candidates)
        base_rank = None if base_candidate is None else int(base_order.index(uid_to_index[base_uid]) + 1)
        base_other_max = max(
            [float(target_col_values[index]) for index in range(len(candidates)) if base_candidate is None or index != uid_to_index[base_uid]]
            + [float(base_row.get("explicit_none_score", 0.0))]
        )
        base_assignment_margin = base_score - base_other_max
        target_iou_for_base = target_iou(base_candidate, target_boxes)
        target_iou_for_top1 = target_iou(None if learned_top1_index is None else candidates[learned_top1_index], target_boxes)
        base_correct = bool(target_iou_for_base is not None and target_iou_for_base >= IOU_THRESHOLD)
        learned_correct = bool(target_iou_for_top1 is not None and target_iou_for_top1 >= IOU_THRESHOLD)
        native_continuity = None if base_candidate is None else float(native_same(target_state, base_candidate))
        predicted_motion = None if base_candidate is None else float(predicted_iou(target_state, np.asarray(base_candidate["box_xyxy"], dtype=np.float64), frame))
        category = "C11" if base_correct and learned_correct else "C10" if base_correct and not learned_correct else "C01" if not base_correct and learned_correct else "C00"
        relative_edge = build_learned_identity_state_edge_matrix(
            bank=learned_bank,
            candidate_rows=candidates,
            public_id_axis=public_axis,
            frame=frame,
            state_scope="HUMAN_INITIALIZED_ONLY",
            base_candidate_public_scores=base_matrix,
        )
        relative_delta = np.asarray(relative_edge["relative_delta"], dtype=np.float64)
        relative_delta_values.extend(relative_delta.reshape(-1).tolist())
        base_score_values.extend(base_matrix.reshape(-1).tolist())
        learned_score_values.extend(learned_scores.tolist())
        required_delta = None
        if category == "C01" and learned_top1_uid is not None:
            required_delta = required_target_delta(
                base_matrix=base_matrix,
                candidates=candidates,
                state_axis=state_axis,
                public_axis=public_axis,
                target_public=target_public,
                desired_uid=learned_top1_uid,
                frame=frame,
                sequence=sequence,
            )
        boundary_rows.append(
            {
                "sequence": sequence,
                "frame": frame,
                "category": category,
                "base_assignment_margin": float(base_assignment_margin),
                "required_target_column_delta_for_learned_top1": required_delta,
                "learned_margin": learned_margin,
                "base_target_score": base_score,
                "learned_top1_score": learned_top1_score,
                "learned_top2_score": learned_top2_score,
                "learned_relative_delta_min": float(np.min(relative_delta)) if relative_delta.size else None,
                "learned_relative_delta_max": float(np.max(relative_delta)) if relative_delta.size else None,
            }
        )
        row: dict[str, Any] = {
            "stage": "N72R20R2",
            "source_stage": "N72R20R1",
            "sequence": sequence,
            "frame": frame,
            "gap": frame - int(event["event_frame"]),
            "base_target_candidate_uid": base_uid,
            "base_target_score": base_score,
            "base_target_rank": base_rank,
            "learned_top1_candidate_uid": learned_top1_uid,
            "learned_top1_score": learned_top1_score,
            "learned_top2_candidate_uid": learned_top2_uid,
            "learned_top2_score": learned_top2_score,
            "learned_margin": learned_margin,
            "learned_score_of_base_candidate": learned_base_score,
            "learned_rank_of_base_candidate": rank_for_uid(learned_scores, candidates, base_uid),
            "base_candidate_is_learned_top1": bool(base_uid is not None and base_uid == learned_top1_uid),
            "human_anchor_score_of_base_candidate": None if base_candidate is None else float(human_scores[uid_to_index[base_uid]]),
            "human_anchor_score_of_learned_top1": None if learned_top1_index is None else float(human_scores[learned_top1_index]),
            "base_candidate_presence": None if base_candidate is None else float(base_candidate.get("presence", base_candidate.get("confidence", 0.0))),
            "base_candidate_iou_pred": None if base_candidate is None else base_candidate.get("iou_pred"),
            "native_continuity": native_continuity,
            "predicted_motion_iou": predicted_motion,
            "base_assignment_margin": float(base_assignment_margin),
            "candidate_count": len(candidates),
            "memory_state_hash": state_hash_before,
            "human_anchor_hash": anchor_hash,
            "frames_since_last_committed_write": frames_since_write,
            "posthoc_target_candidate_uid": target_uid,
            "learned_score_of_posthoc_target_candidate": learned_target_score,
            "learned_rank_of_posthoc_target_candidate": rank_for_uid(learned_scores, candidates, target_uid),
            "learned_target_vs_hard_negative_margin": target_vs_hard_margin,
            "base_candidate_target_iou": target_iou_for_base,
            "learned_top1_target_iou": target_iou_for_top1,
            "base_correct": base_correct,
            "learned_top1_correct": learned_correct,
            "category": category,
            "target_visible": bool(target_boxes),
            "hard_negative_count": len(hard_negatives),
            "runtime_future_gt_used": False,
            "runtime_gt_read": False,
            "posthoc_gt_used": True,
            "base_reconstruction_matches_sealed_tape": bool(reconstructed.shape == base_matrix.shape and np.allclose(reconstructed, base_matrix, atol=1.0e-5, rtol=0.0)),
            "base_native_bonus_contribution": None if base_candidate is None else float(BASE_NATIVE_BONUS if native_same(target_state, base_candidate) else 0.0),
            "base_positive_bonus_contribution": None if base_candidate is None else float(BASE_POSITIVE_BONUS if target_state.has_positive(int(base_candidate["native_tid"]), frame, base_candidate.get("native_scope")) else 0.0),
            "learned_relative_delta_min": float(np.min(relative_delta)) if relative_delta.size else None,
            "learned_relative_delta_max": float(np.max(relative_delta)) if relative_delta.size else None,
            "learned_state_edge_scale": float(relative_edge["state_edge_scale"]),
            "memory_state_hash_after": None,
            "memory_state_similarity_before_after": None,
        }
        bank_update = learned_bank.update_from_consensus(
            frame=frame,
            candidate_rows=candidates,
            base_solver=base_row["base_assignment"],
            treatment_solver=base_row["base_assignment"],
        )
        state_vector_after = learned_bank.records[target_public].current_state
        row["memory_state_hash_after"] = learned_bank.records[target_public].state_hash()
        row["memory_state_similarity_before_after"] = cosine(state_vector_before, state_vector_after)
        row["memory_write"] = bool(bank_update.get("updated_public_ids"))
        row["memory_write_source"] = "R1_C0_BASE_ASSIGNMENT_REPLAY"
        vectors_before[(sequence, frame)] = state_vector_before.copy()
        vectors_after[(sequence, frame)] = state_vector_after.copy()
        table.append(row)
        update_base_states(
            states=base_states,
            solver=base_row["base_assignment"],
            rows=candidates,
            frame=frame,
            target_public=target_public,
            event_frame=int(event["event_frame"]),
        )
    wrong = [row for row in table if row.get("base_correct") is False]
    return table, {
        "sequence": sequence,
        "frames": len(table),
        "base_reconstruction_failures": reconstruction_failures,
        "wrong_write_count": len(wrong),
        "target_gt_id": target_gt_id,
        "target_public_id": target_public,
        "candidate_tape_sha256": sha256(asset_root / "candidates" / sequence / "metadata.jsonl.zst"),
        "base_score_tape_sha256": sha256(asset_root / "base_scores" / sequence / "base_scores.jsonl.zst"),
        "runtime_future_gt_used": False,
        "posthoc_gt_used": True,
        "boundary_rows": boundary_rows,
        "base_score_values": base_score_values,
        "learned_score_values": learned_score_values,
        "relative_delta_values": relative_delta_values,
    }, vectors_before, vectors_after


def aggregate_categories(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    total = len(rows)
    for category in ("C00", "C01", "C10", "C11"):
        selected = [row for row in rows if row.get("category") == category]
        output[category] = {
            "count": len(selected),
            "percentage_of_future_frames": None if total == 0 else len(selected) / total,
            "learned_margin": stats([row.get("learned_margin") for row in selected if row.get("learned_margin") is not None]),
            "base_assignment_margin": stats([row.get("base_assignment_margin") for row in selected if row.get("base_assignment_margin") is not None]),
            "motion_consistency": stats([row.get("predicted_motion_iou") for row in selected if row.get("predicted_motion_iou") is not None]),
        }
    return output


def by_horizon(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for horizon in HORIZONS:
        selected = [row for row in rows if int(row["gap"]) <= horizon]
        total = len(selected)
        result[f"H{horizon}"] = {
            category: {
                "count": sum(row.get("category") == category for row in selected),
                "percentage": None if total == 0 else sum(row.get("category") == category for row in selected) / total,
            }
            for category in ("C00", "C01", "C10", "C11")
        }
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--asset-root", type=Path, default=DEFAULT_ASSET_ROOT)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--sequences", nargs="+", default=["dancetrack0001", "dancetrack0002"])
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/N72R20R2/forensics")
    args = parser.parse_args()
    asset_root = args.asset_root.resolve()
    dataset_root = args.dataset_root.resolve()
    events_payload = read_json(asset_root / "interaction_events.json")
    labels_payload = read_json(asset_root / "posthoc_event_labels.json")
    events = {str(item["sequence"]): dict(item) for item in events_payload["events"]}
    labels = {str(item["sequence"]): dict(item) for item in labels_payload["labels"]}
    all_rows: list[dict[str, Any]] = []
    sequence_summaries: list[dict[str, Any]] = []
    all_boundary_rows: list[dict[str, Any]] = []
    all_base_scores: list[float] = []
    all_learned_scores: list[float] = []
    all_relative: list[float] = []
    vectors_before: dict[tuple[str, int], np.ndarray] = {}
    vectors_after: dict[tuple[str, int], np.ndarray] = {}
    for sequence in args.sequences:
        sequence = str(sequence)
        table, summary, before, after = process_sequence(
            asset_root=asset_root,
            dataset_root=dataset_root,
            sequence=sequence,
            event=events[sequence],
            label=labels[sequence],
        )
        all_rows.extend(table)
        sequence_summaries.append({key: value for key, value in summary.items() if key not in {"boundary_rows", "base_score_values", "learned_score_values", "relative_delta_values"}})
        all_boundary_rows.extend(summary["boundary_rows"])
        all_base_scores.extend(summary["base_score_values"])
        all_learned_scores.extend(summary["learned_score_values"])
        all_relative.extend(summary["relative_delta_values"])
        vectors_before.update(before)
        vectors_after.update(after)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    table_path = args.output_dir / "frame_decision_table.jsonl.zst"
    write_zstd_jsonl(table_path, all_rows)
    categories = aggregate_categories(all_rows)
    sequence_categories = {
        sequence: aggregate_categories([row for row in all_rows if str(row["sequence"]) == sequence])
        for sequence in args.sequences
    }
    summary = {
        "stage": "N72R20R2",
        "source_stage": "N72R20R1",
        "status": "PASS_N72R20R2_FORENSICS",
        "sequences": sequence_summaries,
        "future_frame_count": len(all_rows),
        "categories": categories,
        "categories_by_sequence": sequence_categories,
        "categories_by_horizon": by_horizon(all_rows),
        "candidate_tape_shared": True,
        "base_score_tape_shared": True,
        "runtime_future_gt_used": False,
        "posthoc_gt_used": True,
        "frame_decision_table": str(table_path),
        "frame_decision_table_sha256": sha256(table_path),
        "wrong_write_count": sum(row.get("base_correct") is False for row in all_rows),
        "r1_expected_wrong_write_count": 33,
    }
    (args.output_dir / "forensics_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")

    cascade = state_vectors_cascade(all_rows, vectors_before, vectors_after)
    cascade_payload = {
        "stage": "N72R20R2",
        "source_stage": "N72R20R1",
        "status": "PASS_N72R20R2_CONTAMINATION_CASCADE_ANALYSIS",
        "wrong_write_definition": "base assigned candidate target IoU < 0.50",
        "wrong_write_count": len(cascade["wrong_write_details"]),
        "near_contiguous_definition": "successive wrong frames with gap <= 2",
        **cascade,
        "runtime_future_gt_used": False,
        "posthoc_gt_used": True,
    }
    cascade_path = ROOT / "outputs/N72R20R2/contamination_cascade_analysis.json"
    cascade_path.write_text(json.dumps(cascade_payload, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    (ROOT / "outputs/N72R20R2/contamination_cascade_analysis.md").write_text(markdown_cascade(cascade_payload), encoding="utf-8")

    c01_boundary = [row for row in all_boundary_rows if row["category"] == "C01"]
    boundary_payload = {
        "stage": "N72R20R2",
        "source_stage": "N72R20R1",
        "status": "PASS_N72R20R2_DECISION_BOUNDARY_AUDIT",
        "base_scorer": {
            "reid_weights": BASE_REID_WEIGHTS,
            "native_bonus": BASE_NATIVE_BONUS,
            "positive_bonus": BASE_POSITIVE_BONUS,
            "ema": BASE_EMA,
        },
        "learned_edge": {
            "implementation": "build_learned_identity_state_edge_matrix",
            "relative_delta": "tanh(relative)",
            "state_edge_scale": 1.0,
            "row_max_preservation": True,
        },
        "base_target_column_score_range": stats(all_base_scores),
        "learned_identity_score_range": stats(all_learned_scores),
        "learned_relative_delta_range": stats(all_relative),
        "all_frame_base_assignment_margin": stats([row["base_assignment_margin"] for row in all_boundary_rows]),
        "c01_recoverable_frame_count": len(c01_boundary),
        "c01_required_target_column_delta": stats([
            row["required_target_column_delta_for_learned_top1"]
            for row in c01_boundary
            if row["required_target_column_delta_for_learned_top1"] is not None
        ]),
        "c01_rows": c01_boundary,
        "runtime_future_gt_used": False,
        "posthoc_gt_used": True,
        "interpretation": "Required delta is an offline probe using the existing exact solver; it is not a new solver or a runtime rule.",
    }
    (ROOT / "outputs/N72R20R2/decision_boundary_audit.json").write_text(json.dumps(boundary_payload, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": summary["status"], "future_frames": len(all_rows), "categories": {key: value["count"] for key, value in categories.items()}, "wrong_writes": cascade_payload["wrong_write_count"], "c01": len(c01_boundary)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
