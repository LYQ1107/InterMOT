#!/usr/bin/env python3
"""Evaluate frozen N72R18 memory on a real SAM3 candidate cache.

The candidate cache is produced without GT.  This evaluator reads GT only
after the cache exists to label target coverage, hard negatives, and memory
contamination.  It is target-centric and deliberately does not run MOT or a
global assignment solver.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any, Iterable

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

HORIZONS = (20, 50, 100)
TIME_BINS = ((1, 5, "1_5"), (6, 20, "6_20"), (21, 50, "21_50"), (51, 100, "51_100"))
METHODS = (
    "B0_HUMAN_ANCHOR_ONLY",
    "B1_EMA_0.90",
    "B2_FROZEN_N72R18_GRU",
    "ORACLE_CORRECT_UPDATE_OFFLINE_ORACLE_DIAGNOSTIC",
)
IOU_THRESHOLD = 0.50
CONFIRMATION_CONTINUITY_IOU = 0.10


def normalize(vector: np.ndarray) -> np.ndarray:
    array = np.asarray(vector, dtype=np.float32).reshape(-1)
    norm = float(np.linalg.norm(array))
    if array.size == 0 or not np.isfinite(norm) or norm <= 1e-8:
        raise ValueError("identity feature must be finite and non-zero")
    return array / norm


def xyxy_from_tlwh(box: Iterable[float]) -> tuple[float, float, float, float]:
    left, top, width, height = [float(value) for value in box]
    return left, top, left + width, top + height


def box_iou_xyxy(left: Iterable[float], right: Iterable[float]) -> float:
    lx1, ly1, lx2, ly2 = [float(value) for value in left]
    rx1, ry1, rx2, ry2 = [float(value) for value in right]
    ix1, iy1 = max(lx1, rx1), max(ly1, ry1)
    ix2, iy2 = min(lx2, rx2), min(ly2, ry2)
    intersection = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    left_area = max(0.0, lx2 - lx1) * max(0.0, ly2 - ly1)
    right_area = max(0.0, rx2 - rx1) * max(0.0, ry2 - ry1)
    union = left_area + right_area - intersection
    return intersection / union if union > 0.0 else 0.0


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object: {path}")
    return value


def source_protocol_path(protocol: dict[str, Any], split: str) -> Path:
    source = protocol.get("protocol_sources", {}).get(split, {})
    path = source.get("path") if isinstance(source, dict) else None
    if not path:
        raise ValueError(f"N72R20 protocol has no frozen {split} source protocol")
    result = Path(path).expanduser().resolve()
    if not result.is_file():
        raise FileNotFoundError(f"missing frozen {split} source protocol: {result}")
    return result


def read_zstd_jsonl(path: Path) -> dict[int, list[dict[str, Any]]]:
    if not path.is_file():
        raise FileNotFoundError(f"missing candidate metadata: {path}")
    process = subprocess.Popen(
        ["zstd", "-q", "-d", "-c", "--", str(path)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    assert process.stdout is not None
    by_frame: dict[int, list[dict[str, Any]]] = {}
    try:
        for line_number, line in enumerate(process.stdout, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            frame = int(row["frame"])
            candidates = [dict(item) for item in row.get("candidates", []) if bool(item.get("valid", True))]
            by_frame[frame] = candidates
    finally:
        process.stdout.close()
    stderr = process.stderr.read() if process.stderr is not None else ""
    return_code = process.wait()
    if return_code != 0:
        raise RuntimeError(f"zstd failed for {path} with code {return_code}: {stderr[-2000:]}")
    return by_frame


def load_feature_matrix(candidate_dir: Path) -> np.memmap:
    path = candidate_dir / "embeddings.f16"
    if not path.is_file():
        raise FileNotFoundError(f"missing candidate embeddings: {path}")
    item_bytes = 2 * 512
    if path.stat().st_size % item_bytes:
        raise ValueError(f"candidate embedding file is not an integer number of 512-D float16 rows: {path}")
    return np.memmap(path, mode="r", dtype=np.float16, shape=(path.stat().st_size // item_bytes, 512))


def attach_features(rows: dict[int, list[dict[str, Any]]], feature_matrix: np.memmap) -> None:
    for frame_rows in rows.values():
        for row in frame_rows:
            offset = int(row["embedding_offset"])
            dimension = int(row.get("embedding_dim", 512))
            if dimension != 512 or offset < 0 or offset >= len(feature_matrix):
                raise ValueError(f"invalid embedding offset/dimension in candidate row: {row}")
            row["_feature"] = normalize(np.asarray(feature_matrix[offset], dtype=np.float32))


def load_sequence(dataset_root: Path, split: str, sequence: str):
    from sam3_intermot.identity_probe.dataset import discover_sequences

    sequences = [item for item in discover_sequences(dataset_root, split) if item.name == sequence]
    if len(sequences) != 1:
        raise ValueError(f"expected one {split} sequence named {sequence}, found {len(sequences)}")
    return sequences[0]


def load_anchor_encoder(checkpoint: Path, device: str):
    from scripts.n72r20_candidate_stream_smoke import MachineOSNet

    return MachineOSNet(checkpoint, device)


def candidate_image_box(row: dict[str, Any]) -> tuple[float, float, float, float]:
    return tuple(float(value) for value in row["box_xyxy"])


def _top_index(scores: np.ndarray) -> int:
    if scores.size == 0:
        raise ValueError("cannot select from an empty candidate set")
    # Official candidate order is the deterministic tie-break.
    return int(np.argmax(scores))


def _gru_state(model: torch.nn.Module, state: np.ndarray, observation: np.ndarray) -> np.ndarray:
    try:
        device = next(model.parameters()).device
    except StopIteration:
        device = torch.device("cpu")
    state_tensor = torch.from_numpy(state).float().unsqueeze(0).to(device)
    observation_tensor = torch.from_numpy(observation).float().unsqueeze(0).to(device)
    with torch.no_grad():
        new_state, _reliability, _candidate = model(state_tensor, observation_tensor)
    return normalize(new_state.squeeze(0).cpu().numpy())


def load_frozen_gru(checkpoint: Path, device: str) -> torch.nn.Module:
    from sam3_intermot.identity_memory.selective import load_frozen_n72r18_gru

    return load_frozen_n72r18_gru(checkpoint, torch.device(device))


def _anchor_box_xyxy(anchor: dict[str, Any]) -> tuple[float, float, float, float]:
    return xyxy_from_tlwh(anchor["tlwh"])


def _future_target_xyxy(future: dict[str, Any]) -> tuple[float, float, float, float]:
    return xyxy_from_tlwh(future["target"]["tlwh"])


def _method_record(
    *,
    method: str,
    sequence: str,
    track_id: int,
    anchor_frame: int,
    future_frame: int,
    rows: list[dict[str, Any]],
    state: np.ndarray,
    target_box: tuple[float, float, float, float] | None,
    update_allowed: bool,
    competitor_boxes: Iterable[tuple[float, float, float, float]] = (),
) -> tuple[dict[str, Any], np.ndarray]:
    target_ious = np.asarray(
        [box_iou_xyxy(candidate_image_box(row), target_box) if target_box is not None else 0.0 for row in rows],
        dtype=np.float32,
    )
    target_indices = [index for index, value in enumerate(target_ious) if float(value) >= IOU_THRESHOLD]
    positive_index = max(target_indices, key=lambda index: (float(target_ious[index]), -index)) if target_indices else None
    target_index_set = set(target_indices)
    competitor_boxes = tuple(competitor_boxes)
    negative_indices = [
        index
        for index, row in enumerate(rows)
        if index not in target_index_set
        and any(box_iou_xyxy(candidate_image_box(row), box) >= IOU_THRESHOLD for box in competitor_boxes)
    ]
    features = np.asarray([row["_feature"] for row in rows], dtype=np.float32) if rows else np.empty((0, 512), dtype=np.float32)
    scores = features @ normalize(state) if len(rows) else np.empty((0,), dtype=np.float32)
    selected_index = _top_index(scores) if len(rows) else None
    update_index = selected_index
    update_source = "machine_selected_top1" if selected_index is not None else None
    if method == "ORACLE_CORRECT_UPDATE_OFFLINE_ORACLE_DIAGNOSTIC":
        update_index = positive_index
        update_source = "posthoc_gt_correct_candidate" if positive_index is not None else None
    update_applied = bool(update_allowed and update_index is not None)
    next_state = normalize(state)
    if method == "B1_EMA_0.90" and update_applied:
        next_state = normalize(0.90 * next_state + 0.10 * features[update_index])
    positive_score = None if positive_index is None else float(scores[positive_index])
    hard_negative_score = max((float(scores[index]) for index in negative_indices), default=None)
    identity_evaluable = positive_score is not None and hard_negative_score is not None
    rank = None
    if identity_evaluable:
        rank = 1 + sum(float(scores[index]) > positive_score + 1e-12 for index in negative_indices)
    selected_target = (
        None if target_box is None else selected_index in target_indices if selected_index is not None else None
    )
    update_target = None if target_box is None else update_index in target_indices if update_index is not None else None
    sorted_scores = np.sort(scores)[::-1] if len(scores) else np.empty((0,), dtype=np.float32)
    selection_margin = (
        float(sorted_scores[0] - sorted_scores[1]) if len(sorted_scores) >= 2 else None
    )
    row = {
        "method": method,
        "sequence": sequence,
        "track_id": int(track_id),
        "anchor_frame": int(anchor_frame),
        "frame": int(future_frame),
        "gap": int(future_frame - anchor_frame),
        "candidate_count": len(rows),
        "candidate_available": bool(rows),
        "target_candidate_count": len(target_indices),
        "target_candidate_available": bool(target_indices),
        "hard_negative_count": len(negative_indices),
        "hard_negative_available": bool(negative_indices),
        "identity_evaluable": bool(identity_evaluable),
        "positive_score": positive_score,
        "hard_negative_score": hard_negative_score,
        "margin": None if not identity_evaluable else positive_score - hard_negative_score,
        "win": None if not identity_evaluable else bool(positive_score > hard_negative_score),
        "rank": rank,
        "rank_1": None if rank is None else bool(rank == 1),
        "rank_2": None if rank is None else bool(rank <= 2),
        "rank_3": None if rank is None else bool(rank <= 3),
        "selected_candidate_index": selected_index,
        "selected_candidate_is_target_posthoc": selected_target,
        "selected_candidate_uid": (
            rows[selected_index].get("candidate_uid") if selected_index is not None else None
        ),
        "selected_candidate_box_xyxy": (
            list(candidate_image_box(rows[selected_index])) if selected_index is not None else None
        ),
        "selection_margin_top1_top2": selection_margin,
        "update_candidate_index": update_index,
        "update_candidate_is_target_posthoc": update_target,
        "update_source": update_source,
        "update_applied": update_applied,
        "wrong_memory_write_posthoc": (
            None if update_applied and target_box is None else bool(update_applied and not bool(update_target))
        ),
        "target_visible": target_box is not None,
        "runtime_future_gt_used": False,
        "posthoc_gt_used": True,
    }
    return row, next_state


def _selection_snapshot(record: dict[str, Any], rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    index = record.get("selected_candidate_index")
    if index is None:
        return None
    return {
        "frame": int(record["frame"]),
        "row": rows[int(index)],
        "selection_margin": record.get("selection_margin_top1_top2"),
    }


def _candidate_continuous(previous: dict[str, Any], current: dict[str, Any]) -> bool:
    for key in ("adapter_external_id", "raw_native_id"):
        previous_id = previous.get(key)
        current_id = current.get(key)
        if previous_id is not None and current_id is not None:
            return previous_id == current_id
    return box_iou_xyxy(candidate_image_box(previous), candidate_image_box(current)) >= CONFIRMATION_CONTINUITY_IOU


def _confirmed_update_allowed(
    previous: dict[str, Any] | None,
    current: dict[str, Any] | None,
    *,
    current_frame: int,
    threshold: float,
) -> bool:
    if previous is None or current is None:
        return False
    if int(previous["frame"]) != int(current_frame) - 1:
        return False
    previous_margin = previous.get("selection_margin")
    current_margin = current.get("selection_margin")
    if previous_margin is None or current_margin is None:
        return False
    if float(previous_margin) <= threshold or float(current_margin) <= threshold:
        return False
    return _candidate_continuous(previous["row"], current["row"])


def evaluate_anchor(
    *,
    sequence_name: str,
    anchor_record: dict[str, Any],
    sequence,
    candidates: dict[int, list[dict[str, Any]]],
    encoder,
    gru: torch.nn.Module,
    update_policy: str,
    confirmation_margin_threshold: float | None,
) -> list[dict[str, Any]]:
    anchor = dict(anchor_record["anchor"])
    anchor_frame = int(anchor_record["anchor_frame"])
    anchor_image = sequence.image_path(anchor_frame)
    anchor_vector = normalize(encoder.encode(anchor_image, [_anchor_box_xyxy(anchor)])[0])
    states = {method: anchor_vector.copy() for method in METHODS}
    previous_selections: dict[str, dict[str, Any] | None] = {method: None for method in METHODS}
    rows: list[dict[str, Any]] = []
    end_frame = min(int(sequence.seq_length), anchor_frame + 100)
    track_id = int(anchor["track_id"])
    for future_frame in range(anchor_frame + 1, end_frame + 1):
        gap = future_frame - anchor_frame
        candidate_rows = candidates.get(future_frame - 1, [])
        frame_boxes = tuple(sequence.boxes(future_frame))
        target = next((box for box in frame_boxes if int(box.track_id) == track_id), None)
        target_box = xyxy_from_tlwh(target.tlwh) if target is not None else None
        competitor_boxes = tuple(
            xyxy_from_tlwh(box.tlwh)
            for box in frame_boxes
            if int(box.track_id) != track_id and int(box.track_id) >= 0
        )
        for method in METHODS:
            provisional, _ = _method_record(
                method=method,
                sequence=sequence_name,
                track_id=track_id,
                anchor_frame=anchor_frame,
                future_frame=future_frame,
                rows=candidate_rows,
                state=states[method],
                target_box=target_box,
                competitor_boxes=competitor_boxes,
                update_allowed=False,
            )
            current_selection = _selection_snapshot(provisional, candidate_rows)
            if method == "B0_HUMAN_ANCHOR_ONLY":
                update_allowed = False
            elif method == "ORACLE_CORRECT_UPDATE_OFFLINE_ORACLE_DIAGNOSTIC":
                update_allowed = bool(provisional["target_candidate_available"])
            elif update_policy == "immediate":
                update_allowed = bool(provisional["selected_candidate_index"] is not None)
            else:
                if confirmation_margin_threshold is None:
                    raise ValueError("2-frame update policy requires a frozen confirmation margin threshold")
                update_allowed = _confirmed_update_allowed(
                    previous_selections[method],
                    current_selection,
                    current_frame=future_frame,
                    threshold=confirmation_margin_threshold,
                )
            if update_allowed:
                record, next_state = _method_record(
                    method=method,
                    sequence=sequence_name,
                    track_id=track_id,
                    anchor_frame=anchor_frame,
                    future_frame=future_frame,
                    rows=candidate_rows,
                    state=states[method],
                    target_box=target_box,
                    competitor_boxes=competitor_boxes,
                    update_allowed=True,
                )
            else:
                record, next_state = provisional, normalize(states[method])
            if (
                method in {"B2_FROZEN_N72R18_GRU", "ORACLE_CORRECT_UPDATE_OFFLINE_ORACLE_DIAGNOSTIC"}
                and record["update_applied"]
                and record["update_candidate_index"] is not None
            ):
                updated = candidate_rows[int(record["update_candidate_index"])]
                next_state = _gru_state(gru, states[method], updated["_feature"])
            states[method] = next_state
            rows.append(record)
            previous_selections[method] = current_selection
    return rows


def _bootstrap_sequence_ci(records: list[dict[str, Any]], reps: int = 2000, seed: int = 72020) -> dict[str, Any]:
    eligible = [row for row in records if row.get("identity_evaluable")]
    by_sequence: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in eligible:
        by_sequence[str(row["sequence"])].append(row)
    names = sorted(by_sequence)
    if not names:
        return {"lower": None, "upper": None, "clusters": 0, "reps": reps, "seed": seed}
    rng = np.random.default_rng(seed)
    values: list[float] = []
    for _ in range(reps):
        selected = rng.integers(0, len(names), size=len(names))
        cluster_rates = []
        for index in selected:
            cluster = by_sequence[names[int(index)]]
            cluster_rates.append(float(np.mean([bool(item["win"]) for item in cluster])))
        values.append(float(np.mean(cluster_rates)))
    return {
        "lower": float(np.percentile(values, 2.5)),
        "upper": float(np.percentile(values, 97.5)),
        "clusters": len(names),
        "reps": reps,
        "seed": seed,
    }


def _recovery_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    trajectories: dict[tuple[str, int, int], list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        if bool(row.get("target_visible", True)) and bool(row.get("identity_evaluable")):
            key = (str(row["sequence"]), int(row["track_id"]), int(row["anchor_frame"]))
            trajectories[key].append(row)
    failures = 0
    recovered = 0
    recovery_gaps: list[int] = []
    for trajectory in trajectories.values():
        ordered = sorted(trajectory, key=lambda row: int(row["frame"]))
        failed_at = next((index for index, row in enumerate(ordered) if row["rank_1"] is False), None)
        if failed_at is None:
            continue
        failures += 1
        later = next((row for row in ordered[failed_at + 1 :] if row["rank_1"] is True), None)
        if later is not None:
            recovered += 1
            recovery_gaps.append(int(later["frame"]) - int(ordered[failed_at]["frame"]))
    return {
        "trajectories_with_rank1_failure": failures,
        "trajectories_with_future_rank1_recovery": recovered,
        "future_identity_recovery_rate": recovered / failures if failures else None,
        "median_recovery_gap": float(np.median(recovery_gaps)) if recovery_gaps else None,
    }


def summarize_method(records: list[dict[str, Any]], method: str) -> dict[str, Any]:
    method_rows = [row for row in records if row["method"] == method]

    def summary(subset: list[dict[str, Any]], label: str) -> dict[str, Any]:
        target_visible = [row for row in subset if bool(row.get("target_visible", True))]
        visible = [row for row in target_visible if row["candidate_available"]]
        covered = [row for row in target_visible if row["target_candidate_available"]]
        evaluable = [row for row in target_visible if row["identity_evaluable"]]
        margins = np.asarray([float(row["margin"]) for row in evaluable], dtype=np.float32)
        ranks = np.asarray([int(row["rank"]) for row in evaluable], dtype=np.int64)
        all_updates = [row for row in subset if row["update_applied"]]
        updates = [row for row in target_visible if row["update_applied"]]
        wrong_writes = [row for row in updates if row["wrong_memory_write_posthoc"] is True]
        return {
            "label": label,
            "runtime_frames": len(subset),
            "target_visible_frames": len(target_visible),
            "candidate_available_frames": len(visible),
            "candidate_pool_availability": len(visible) / len(target_visible) if target_visible else None,
            "covered_target_frames": len(covered),
            "candidate_coverage": len(covered) / len(target_visible) if target_visible else None,
            "identity_evaluable_frames": len(evaluable),
            "hard_negative_identity_win_rate": float(np.mean([bool(row["win"]) for row in evaluable])) if evaluable else None,
            "hard_negative_identity_win_rate_sequence_cluster_ci95": _bootstrap_sequence_ci(evaluable),
            "rank_1_accuracy": float(np.mean(ranks == 1)) if len(ranks) else None,
            "rank_2_accuracy": float(np.mean(ranks <= 2)) if len(ranks) else None,
            "rank_3_accuracy": float(np.mean(ranks <= 3)) if len(ranks) else None,
            "mrr": float(np.mean(1.0 / ranks)) if len(ranks) else None,
            "mean_margin": float(margins.mean()) if len(margins) else None,
            "median_margin": float(np.median(margins)) if len(margins) else None,
            "p10_margin": float(np.percentile(margins, 10)) if len(margins) else None,
            "p25_margin": float(np.percentile(margins, 25)) if len(margins) else None,
            "p75_margin": float(np.percentile(margins, 75)) if len(margins) else None,
            "machine_updates": len(updates),
            "machine_updates_all_runtime_frames": len(all_updates),
            "wrong_memory_writes": len(wrong_writes),
            "wrong_memory_write_rate": len(wrong_writes) / len(updates) if updates else None,
            **_recovery_summary(evaluable),
        }

    horizons = {f"H{horizon}": summary([row for row in method_rows if int(row["gap"]) <= horizon], f"H{horizon}") for horizon in HORIZONS}
    bins = {
        label: summary([row for row in method_rows if low <= int(row["gap"]) <= high], label)
        for low, high, label in TIME_BINS
    }
    return {"method": method, "horizons": horizons, "time_gap_bins": bins}


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.update_policy == "2frame" and args.confirmation_margin_threshold is None:
        raise ValueError("--confirmation-margin-threshold is required for the 2-frame update diagnostic")
    if args.confirmation_margin_threshold is not None and args.confirmation_margin_threshold < 0.0:
        raise ValueError("confirmation margin threshold must be non-negative")
    manifest = load_json(Path(args.asset_manifest).expanduser().resolve())
    if manifest.get("stage") != "N72R20":
        raise ValueError("asset manifest is not N72R20")
    protocol = load_json(Path(args.protocol).expanduser().resolve())
    dataset_root = Path(args.dataset_root or manifest["DANCETRACK_ROOT"]).expanduser().resolve()
    candidate_value = args.candidate_root or os.environ.get("N72R20_ASSET_ROOT")
    if not candidate_value:
        raise ValueError("provide --candidate-root or set N72R20_ASSET_ROOT")
    candidate_root = Path(candidate_value).expanduser().resolve()
    gru_path = Path(args.gru_checkpoint or manifest["frozen_identity_assets"]["n72r18_gru"]["path"]).expanduser().resolve()
    osnet_path = Path(args.osnet_checkpoint or manifest["frozen_identity_assets"]["osnet_x1_0_market1501"]["path"]).expanduser().resolve()
    if not gru_path.is_file() or not osnet_path.is_file():
        raise FileNotFoundError("frozen N72R18 GRU and OSNet checkpoints are required")
    source_path = source_protocol_path(protocol, args.split)
    source = load_json(source_path)
    sequence_name = str(args.sequence)
    sequence = load_sequence(dataset_root, args.split, sequence_name)
    anchors = [item for item in source.get("anchors", []) if str(item.get("anchor", {}).get("sequence")) == sequence_name]
    if args.max_anchors > 0:
        anchors = anchors[: int(args.max_anchors)]
    if not anchors:
        raise ValueError(f"no frozen anchors for {args.split}:{sequence_name}")
    candidate_dir = candidate_root / "candidates" / sequence_name
    candidates = read_zstd_jsonl(candidate_dir / "metadata.jsonl.zst")
    feature_matrix = load_feature_matrix(candidate_dir)
    attach_features(candidates, feature_matrix)
    encoder = load_anchor_encoder(osnet_path, args.anchor_device)
    gru = load_frozen_gru(gru_path, args.anchor_device)
    records: list[dict[str, Any]] = []
    skipped = 0
    for anchor in anchors:
        try:
            records.extend(
                evaluate_anchor(
                    sequence_name=sequence_name,
                    anchor_record=anchor,
                    sequence=sequence,
                    candidates=candidates,
                    encoder=encoder,
                    gru=gru,
                    update_policy=args.update_policy,
                    confirmation_margin_threshold=args.confirmation_margin_threshold,
                )
            )
        except (FileNotFoundError, ValueError, RuntimeError) as exc:
            if args.skip_unavailable_anchors:
                skipped += 1
                continue
            raise RuntimeError(f"anchor evaluation failed for {sequence_name}:{anchor.get('anchor_frame')}") from exc
    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    records_path = output.with_suffix(".jsonl")
    with records_path.open("w", encoding="utf-8") as handle:
        for row in records:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n")
    result = {
        "stage": "N72R20",
        "status": "PASS_N72R20_IDENTITY_BRIDGE_EVAL",
        "split": args.split,
        "sequence": sequence_name,
        "anchor_count": len(anchors),
        "skipped_anchor_count": skipped,
        "record_count": len(records),
        "methods": {method: summarize_method(records, method) for method in METHODS},
        "candidate_coverage_and_identity_discrimination_are_separate": True,
        "runtime_future_gt_used": False,
        "runtime_gt_read": False,
        "posthoc_gt_used": True,
        "interaction_source": "simulated_from_gt",
        "candidate_iou_threshold": IOU_THRESHOLD,
        "confirmation_continuity_iou": CONFIRMATION_CONTINUITY_IOU,
        "update_policy": args.update_policy,
        "confirmation_margin_threshold": args.confirmation_margin_threshold,
        "records": str(records_path),
        "candidate_metadata": str(candidate_dir / "metadata.jsonl.zst"),
        "candidate_embeddings": str(candidate_dir / "embeddings.f16"),
        "frozen_protocol_source": str(source_path),
        "frozen_n72r18_gru": str(gru_path),
        "frozen_osnet": str(osnet_path),
        "val_tuning_used": False,
    }
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "sequence": sequence_name, "records": len(records), "output": str(output)}, ensure_ascii=False))
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sequence", required=True)
    parser.add_argument("--split", choices=("train", "val"), default="train")
    parser.add_argument("--asset-manifest", type=Path, default=ROOT / "outputs/N72R20/asset_manifest.json")
    parser.add_argument("--protocol", type=Path, default=ROOT / "outputs/N72R20/protocol.json")
    parser.add_argument("--dataset-root", type=Path)
    parser.add_argument("--candidate-root", type=Path)
    parser.add_argument("--gru-checkpoint", type=Path)
    parser.add_argument("--osnet-checkpoint", type=Path)
    parser.add_argument("--anchor-device", default="cpu")
    parser.add_argument("--update-policy", choices=("immediate", "2frame"), default="immediate")
    parser.add_argument("--confirmation-margin-threshold", type=float)
    parser.add_argument("--max-anchors", type=int, default=0)
    parser.add_argument("--skip-unavailable-anchors", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        run(args)
    except Exception as exc:
        print(f"N72R20 identity bridge evaluation failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
