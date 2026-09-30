#!/usr/bin/env python3
"""Post-hoc identity shadow metrics for the fresh R1 train tape.

The runtime tape is already sealed before this script opens GT.  The shadow
branch never changes the assignment: the frozen E0 assignment is used only as
the causal trusted observation for the diagnostic learned-memory update.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any, Mapping

import numpy as np

from sam3_intermot.association.learned_identity_memory import LearnedIdentityMemoryBank
from scripts.n72r20r1_build_base_score_tape import load_candidate_frames, read_json, read_zstd_jsonl, unit


ROOT = Path(__file__).resolve().parents[1]
ASSET_ROOT = Path("/data3/liuyeqiang/InterMOT_N72R20R1_assets")
DATASET_ROOT = Path("/data3/liuyeqiang/InterMOT_N72R16_assets/dataset")
CHECKPOINT = ROOT / "outputs/N72R18/checkpoints/identity_memory_gru.pt"
CHECKPOINT_SHA = "94ba9c44e177b254ee54984195e839f37c78a648c010ffc723dd3f1d553976a2"
ENCODER_SHA = "2809d3227f7d078f6045f7feb874a34d0684f0e0057b264b99adccf7d4519154"
HORIZONS = (20, 50, 100)
BOOTSTRAP_REPS = 2000
BOOTSTRAP_SEED = 720201
IOU_THRESHOLD = 0.50


def iou(left: Any, right: Any) -> float:
    a = np.asarray(left, dtype=np.float64)
    b = np.asarray(right, dtype=np.float64)
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    intersection = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - intersection
    return float(intersection / union) if union > 0 else 0.0


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


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


def metric_records(
    *,
    candidate_frames: list[tuple[dict[str, Any], list[dict[str, Any]]]],
    base_rows: list[dict[str, Any]],
    event: Mapping[str, Any],
    gt: Mapping[int, list[tuple[int, list[float]]]],
) -> dict[str, list[dict[str, Any]]]:
    anchor = unit(event["human_anchor"], "human anchor")
    target_public = int(event["target_public_id"])
    bank = LearnedIdentityMemoryBank.from_checkpoint(
        CHECKPOINT,
        encoder_sha256=ENCODER_SHA,
        expected_encoder_sha256=ENCODER_SHA,
        expected_checkpoint_sha256=CHECKPOINT_SHA,
    )
    bank.initialize_public_identity(
        public_id=target_public,
        association_state_id=int(event["target_association_state_id"]),
        frame=int(event["event_frame"]),
        human_anchor=anchor,
        interaction_source="simulated_from_gt",
    )
    records: dict[str, list[dict[str, Any]]] = {"HUMAN_ANCHOR_ONLY": [], "LEARNED_MEMORY_SHADOW": []}
    for (frame_row, candidates), base_row in zip(candidate_frames, base_rows):
        frame = int(frame_row["frame"])
        if frame == int(event["event_frame"]):
            continue
        # Candidate frame 0 corresponds to DanceTrack GT frame 1; gt_by_frame
        # stores the same zero-based convention.
        target_boxes = [box for track_id, box in gt.get(frame, []) if int(track_id) == int(event["posthoc_target_gt_id"])]
        all_boxes = list(gt.get(frame, []))
        target_box = target_boxes[0] if target_boxes else None
        target_positive: dict[str, Any] | None = None
        hard_negatives: list[dict[str, Any]] = []
        if target_box is not None:
            target_positive = max(candidates, key=lambda row: (iou(row["box_xyxy"], target_box), str(row["candidate_uid"])), default=None)
            if target_positive is not None and iou(target_positive["box_xyxy"], target_box) < IOU_THRESHOLD:
                target_positive = None
            for candidate in candidates:
                other_iou = max((iou(candidate["box_xyxy"], box) for track_id, box in all_boxes if int(track_id) != int(event["posthoc_target_gt_id"])), default=0.0)
                if other_iou >= IOU_THRESHOLD and (target_positive is None or str(candidate["candidate_uid"]) != str(target_positive["candidate_uid"])):
                    hard_negatives.append(candidate)
        score = bank.score_matrix(candidate_rows=candidates, public_id_axis=[target_public], frame=frame)
        learned_values = np.asarray(score["state_candidate_scores"], dtype=np.float64)[0] if candidates else np.zeros(0, dtype=np.float64)
        by_uid = {str(row["candidate_uid"]): index for index, row in enumerate(candidates)}
        positive_uid = None if target_positive is None else str(target_positive["candidate_uid"])
        for method in records:
            values = np.asarray(
                [float(np.dot(anchor, unit(row["feature"], "candidate feature"))) for row in candidates]
                if method == "HUMAN_ANCHOR_ONLY"
                else learned_values,
                dtype=np.float64,
            )
            eligible = bool(target_positive is not None and hard_negatives)
            item: dict[str, Any] = {
                "sequence": str(event["sequence"]),
                "frame": frame,
                "gap": frame - int(event["event_frame"]),
                "method": method,
                "target_visible": bool(target_box is not None),
                "candidate_available": bool(target_positive is not None),
                "hard_negative_available": bool(hard_negatives),
                "candidate_count": len(candidates),
                "runtime_future_gt_used": False,
                "posthoc_gt_used": True,
            }
            if eligible:
                positive_index = by_uid[positive_uid]
                order = sorted(range(len(candidates)), key=lambda index: (-float(values[index]), str(candidates[index]["candidate_uid"])))
                rank = order.index(positive_index) + 1
                negative_scores = [float(values[by_uid[str(row["candidate_uid"])]]) for row in hard_negatives]
                positive_score = float(values[positive_index])
                hard_negative_score = max(negative_scores)
                item.update(
                    {
                        "identity_evaluable": True,
                        "positive_score": positive_score,
                        "hard_negative_score": hard_negative_score,
                        "margin": positive_score - hard_negative_score,
                        "win": bool(positive_score > hard_negative_score),
                        "rank": int(rank),
                        "rank_1": bool(rank <= 1),
                        "rank_2": bool(rank <= 2),
                        "rank_3": bool(rank <= 3),
                        "mrr": 1.0 / float(rank),
                        "positive_candidate_uid": positive_uid,
                        "hard_negative_count": len(hard_negatives),
                    }
                )
            else:
                item.update(
                    {
                        "identity_evaluable": False,
                        "positive_score": None,
                        "hard_negative_score": None,
                        "margin": None,
                        "win": None,
                        "rank": None,
                        "rank_1": None,
                        "rank_2": None,
                        "rank_3": None,
                        "mrr": None,
                    }
                )
            records[method].append(item)

        base_solver = base_row["base_assignment"]
        bank.update_from_consensus(
            frame=frame,
            candidate_rows=candidates,
            base_solver=base_solver,
            treatment_solver=base_solver,
        )
    return records


def aggregate(records: list[dict[str, Any]], horizon: int) -> dict[str, Any]:
    selected = [item for item in records if int(item["gap"]) <= int(horizon)]
    evaluable = [item for item in selected if item.get("identity_evaluable")]
    def rate(key: str) -> float | None:
        return None if not evaluable else float(np.mean([float(item[key]) for item in evaluable]))
    margins = np.asarray([float(item["margin"]) for item in evaluable], dtype=np.float64)
    return {
        "horizon": int(horizon),
        "runtime_frames": len(selected),
        "target_visible_frames": sum(bool(item["target_visible"]) for item in selected),
        "candidate_available_frames": sum(bool(item["candidate_available"]) for item in selected),
        "hard_negative_available_frames": sum(bool(item["hard_negative_available"]) for item in selected),
        "identity_evaluable_frames": len(evaluable),
        "candidate_coverage": None if not selected else float(sum(bool(item["candidate_available"]) for item in selected) / len(selected)),
        "hard_negative_identity_win_rate": rate("win"),
        "rank_1_accuracy": rate("rank_1"),
        "rank_2_accuracy": rate("rank_2"),
        "rank_3_accuracy": rate("rank_3"),
        "mrr": rate("mrr"),
        "mean_margin": None if not evaluable else float(np.mean(margins)),
        "median_margin": None if not evaluable else float(np.median(margins)),
        "p10_margin": None if not evaluable else float(np.percentile(margins, 10)),
        "p25_margin": None if not evaluable else float(np.percentile(margins, 25)),
        "p75_margin": None if not evaluable else float(np.percentile(margins, 75)),
    }


def bootstrap_delta(per_sequence: Mapping[str, Mapping[str, float]], key: str) -> dict[str, Any]:
    names = sorted(per_sequence)
    if not names:
        return {"lower": None, "upper": None, "clusters": 0, "reps": BOOTSTRAP_REPS, "seed": BOOTSTRAP_SEED}
    values = np.asarray([float(per_sequence[name][key]) for name in names], dtype=np.float64)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    sampled = rng.integers(0, len(values), size=(BOOTSTRAP_REPS, len(values)))
    means = np.mean(values[sampled], axis=1)
    return {
        "lower": float(np.percentile(means, 2.5)),
        "upper": float(np.percentile(means, 97.5)),
        "clusters": len(values),
        "reps": BOOTSTRAP_REPS,
        "seed": BOOTSTRAP_SEED,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--asset-root", type=Path, default=ASSET_ROOT)
    parser.add_argument("--dataset-root", type=Path, default=DATASET_ROOT)
    parser.add_argument("--sequences", nargs="+", default=["dancetrack0001", "dancetrack0002"])
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/N72R20R1/shadow/train_smoke_identity_metrics.json")
    args = parser.parse_args()
    asset_root = args.asset_root.resolve()
    events_payload = read_json(asset_root / "interaction_events.json")
    labels_payload = read_json(asset_root / "posthoc_event_labels.json")
    events = {str(item["sequence"]): dict(item) for item in events_payload["events"]}
    labels = {str(item["sequence"]): dict(item) for item in labels_payload["labels"]}
    all_records: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for sequence in args.sequences:
        event = dict(events[str(sequence)])
        event["posthoc_target_gt_id"] = int(labels[str(sequence)]["target_gt_id"])
        candidates = load_candidate_frames(asset_root, str(sequence))
        base_rows = read_zstd_jsonl(asset_root / "base_scores" / str(sequence) / "base_scores.jsonl.zst")
        gt = gt_by_frame(args.dataset_root.resolve() / "train" / str(sequence) / "gt" / "gt.txt")
        all_records[str(sequence)] = metric_records(candidate_frames=candidates, base_rows=base_rows, event=event, gt=gt)

    methods = ["HUMAN_ANCHOR_ONLY", "LEARNED_MEMORY_SHADOW"]
    metrics: dict[str, Any] = {}
    for method in methods:
        per_sequence: dict[str, dict[str, Any]] = {}
        horizons: dict[str, Any] = {}
        for horizon in HORIZONS:
            by_sequence: dict[str, dict[str, Any]] = {}
            for sequence, records in all_records.items():
                by_sequence[sequence] = aggregate(records[method], horizon)
            horizons[f"H{horizon}"] = {
                "aggregate": aggregate([item for records in all_records.values() for item in records[method]], horizon),
                "per_sequence": by_sequence,
            }
        metrics[method] = {"horizons": horizons}
    comparison: dict[str, Any] = {}
    for horizon in HORIZONS:
        h = f"H{horizon}"
        learned = metrics["LEARNED_MEMORY_SHADOW"]["horizons"][h]["aggregate"]
        anchor = metrics["HUMAN_ANCHOR_ONLY"]["horizons"][h]["aggregate"]
        delta = {
            "hard_negative_win_rate": None if learned["hard_negative_identity_win_rate"] is None else float(learned["hard_negative_identity_win_rate"] - anchor["hard_negative_identity_win_rate"]),
            "rank_1_accuracy": None if learned["rank_1_accuracy"] is None else float(learned["rank_1_accuracy"] - anchor["rank_1_accuracy"]),
            "mrr": None if learned["mrr"] is None else float(learned["mrr"] - anchor["mrr"]),
            "mean_margin": None if learned["mean_margin"] is None else float(learned["mean_margin"] - anchor["mean_margin"]),
        }
        per_sequence_delta = {}
        for sequence in all_records:
            l = metrics["LEARNED_MEMORY_SHADOW"]["horizons"][h]["per_sequence"][sequence]
            a = metrics["HUMAN_ANCHOR_ONLY"]["horizons"][h]["per_sequence"][sequence]
            per_sequence_delta[sequence] = {
                "hard_negative_win_rate": float(l["hard_negative_identity_win_rate"] - a["hard_negative_identity_win_rate"]),
                "rank_1_accuracy": float(l["rank_1_accuracy"] - a["rank_1_accuracy"]),
                "mrr": float(l["mrr"] - a["mrr"]),
                "mean_margin": float(l["mean_margin"] - a["mean_margin"]),
            }
        comparison[h] = {
            "learned_minus_anchor": delta,
            "h100_sequence_cluster_ci95": bootstrap_delta(per_sequence_delta, "hard_negative_win_rate") if horizon == 100 else None,
        }
    h100_delta = comparison["H100"]["learned_minus_anchor"]["hard_negative_win_rate"]
    decision = "PASS_LEARNED_MEMORY_CANDIDATE_SIGNAL" if h100_delta is not None and h100_delta > 0.0 else "FAIL_LEARNED_MEMORY_CANDIDATE_SIGNAL"
    payload = {
        "stage": "N72R20R1",
        "status": decision,
        "shadow_type": "POST_ASSIGNMENT_SHADOW_BASE_DRIVEN",
        "methods": methods,
        "metrics": metrics,
        "comparison": comparison,
        "decision_basis": "H100 learned-memory hard-negative win rate strictly exceeds human-anchor-only baseline on the same fresh candidate tape",
        "candidate_tape_shared": True,
        "base_score_tape_shared": True,
        "gt_role": "posthoc_evaluation_truth_only",
        "runtime_future_gt_used": False,
        "posthoc_gt_used": True,
        "bootstrap": {"unit": "sequence_cluster", "replicates": BOOTSTRAP_REPS, "seed": BOOTSTRAP_SEED},
        "candidate_tape_sha256": {
            str(sequence): sha256(asset_root / "candidates" / str(sequence) / "metadata.jsonl.zst")
            for sequence in args.sequences
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": decision, "h100_delta": h100_delta}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
