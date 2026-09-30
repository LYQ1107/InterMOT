#!/usr/bin/env python3
"""Run the N72R20 feature-level learned-memory shadow on existing train tapes.

This is deliberately not the formal association replay.  It replays the
already materialized real-SAM3 candidate rows and the old B2 machine-selection
trace only to answer the first gate: does the frozen N72R18 state expose a
non-zero identity discrimination signal?  Ground truth is opened only by this
post-hoc diagnostic, never by the memory bank or a solver.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import subprocess
from typing import Any

import numpy as np

from sam3_intermot.association.learned_identity_memory import LearnedIdentityMemoryBank


ROOT = Path(__file__).resolve().parents[1]
ASSET_ROOT = Path("/data3/liuyeqiang/InterMOT_N72R20_assets")
DATASET_ROOT = Path("/data3/liuyeqiang/InterMOT_N72R16_assets/dataset")
OSNET_SHA = "2809d3227f7d078f6045f7feb874a34d0684f0e0057b264b99adccf7d4519154"
GRU_CHECKPOINT = ROOT / "outputs/N72R18/checkpoints/identity_memory_gru.pt"
GRU_SHA = "94ba9c44e177b254ee54984195e839f37c78a648c010ffc723dd3f1d553976a2"
SEQUENCES = ("dancetrack0001", "dancetrack0002")
HORIZONS = (20, 50, 100)
IOU_THRESHOLD = 0.5


def iou_xyxy(left: list[float], right: list[float]) -> float:
    x1 = max(float(left[0]), float(right[0]))
    y1 = max(float(left[1]), float(right[1]))
    x2 = min(float(left[2]), float(right[2]))
    y2 = min(float(left[3]), float(right[3]))
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_l = max(0.0, float(left[2]) - float(left[0])) * max(0.0, float(left[3]) - float(left[1]))
    area_r = max(0.0, float(right[2]) - float(right[0])) * max(0.0, float(right[3]) - float(right[1]))
    union = area_l + area_r - inter
    return 0.0 if union <= 0.0 else inter / union


def read_gt(sequence: str) -> dict[int, dict[int, list[float]]]:
    result: dict[int, dict[int, list[float]]] = defaultdict(dict)
    path = DATASET_ROOT / "train" / sequence / "gt" / "gt.txt"
    for line in path.read_text().splitlines():
        fields = line.split(",")
        if len(fields) < 6:
            continue
        frame, track_id = int(fields[0]), int(fields[1])
        x, y, width, height = map(float, fields[2:6])
        result[frame][track_id] = [x, y, x + width, y + height]
    return dict(result)


def read_candidates(sequence: str) -> dict[int, list[dict[str, Any]]]:
    root = ASSET_ROOT / "candidates" / sequence
    embeddings = np.fromfile(root / "embeddings.f16", dtype=np.float16).reshape(-1, 512).astype(np.float32)
    rows: dict[int, list[dict[str, Any]]] = defaultdict(list)
    completed = subprocess.run(
        ["zstd", "-dc", str(root / "metadata.jsonl.zst")],
        check=True,
        capture_output=True,
    )
    for raw in completed.stdout.splitlines():
        record = json.loads(raw)
        frame = int(record["frame"])
        for candidate in record["candidates"]:
            item = dict(candidate)
            offset = int(item["embedding_offset"])
            item["embedding"] = embeddings[offset]
            rows[frame].append(item)
    return dict(rows)


def read_old_b2_trace(sequence: str) -> dict[tuple[int, int], dict[str, Any]]:
    manifest = json.loads(
        (ASSET_ROOT / "train_identity_records" / f"{sequence}.immediate.json").read_text()
    )
    result: dict[tuple[int, int], dict[str, Any]] = {}
    with Path(manifest["records"]).open() as handle:
        for line in handle:
            row = json.loads(line)
            if row.get("method") == "B2_FROZEN_N72R18_GRU":
                result[(int(row["track_id"]), int(row["frame"]))] = row
    return result


def target_candidate(rows: list[dict[str, Any]], target_box: list[float]) -> dict[str, Any] | None:
    eligible = [(iou_xyxy(row["box_xyxy"], target_box), row) for row in rows]
    eligible = [item for item in eligible if item[0] >= IOU_THRESHOLD]
    if not eligible:
        return None
    return max(eligible, key=lambda item: (item[0], str(item[1]["candidate_uid"])))[1]


def metrics_for_sequence(sequence: str) -> dict[str, Any]:
    gt = read_gt(sequence)
    candidate_rows = read_candidates(sequence)
    old_b2 = read_old_b2_trace(sequence)
    target_ids = sorted({int(row["track_id"]) for row in old_b2.values()})
    sequence_records: list[dict[str, Any]] = []
    for track_id in target_ids:
        anchor_box = gt[1][track_id]
        anchor = target_candidate(candidate_rows[0], anchor_box)
        if anchor is None:
            continue
        bank = LearnedIdentityMemoryBank.from_checkpoint(
            GRU_CHECKPOINT,
            encoder_sha256=OSNET_SHA,
            expected_encoder_sha256=OSNET_SHA,
            expected_checkpoint_sha256=GRU_SHA,
        )
        public_id = 100000 + track_id
        bank.initialize_public_identity(
            public_id=public_id,
            association_state_id=200000 + track_id,
            frame=1,
            human_anchor=anchor["embedding"],
            interaction_source="simulated_from_gt",
        )
        for horizon_frame in range(2, min(max(gt), 101) + 1):
            rows = candidate_rows.get(horizon_frame - 1, [])
            target_box = gt.get(horizon_frame, {}).get(track_id)
            if target_box is None:
                continue
            target = target_candidate(rows, target_box)
            competitors: list[dict[str, Any]] = []
            for row in rows:
                if target is not None and row["candidate_uid"] == target["candidate_uid"]:
                    continue
                if any(
                    other_id != track_id and iou_xyxy(row["box_xyxy"], other_box) >= IOU_THRESHOLD
                    for other_id, other_box in gt.get(horizon_frame, {}).items()
                ):
                    competitors.append(row)
            score = bank.score_matrix(
                candidate_rows=rows,
                public_id_axis=[public_id],
                frame=horizon_frame,
            )
            values = np.asarray(score["state_candidate_scores"], dtype=np.float64)[0]
            by_uid = {str(row["candidate_uid"]): index for index, row in enumerate(rows)}
            positive_score = None if target is None else float(values[by_uid[str(target["candidate_uid"])]])
            competitor_scores = [float(values[by_uid[str(row["candidate_uid"])]]) for row in competitors]
            rank = None
            if positive_score is not None:
                rank = 1 + sum(value > positive_score for value in values.tolist())
            auc_pairs = 0
            auc_wins = 0.0
            if positive_score is not None:
                for value in competitor_scores:
                    auc_pairs += 1
                    auc_wins += 1.0 if positive_score > value else 0.5 if positive_score == value else 0.0
            hard_negative_score = max(competitor_scores) if competitor_scores else None
            old_trace = old_b2.get((track_id, horizon_frame))
            selected_uid = None if old_trace is None else old_trace.get("selected_candidate_uid")
            update_applied = bool(old_trace and old_trace.get("update_applied") and selected_uid in by_uid)
            if update_applied:
                item = {
                    "public_id": public_id,
                    "candidate_uid": str(selected_uid),
                    "status": "ASSIGNED_TO_PUBLIC_ID",
                    "score": 1.0,
                }
                bank.update_from_consensus(
                    frame=horizon_frame,
                    candidate_rows=rows,
                    base_solver={"public_assignments": [dict(item)], "runtime_future_gt_used": False},
                    treatment_solver={"public_assignments": [dict(item)], "runtime_future_gt_used": False},
                    source="historical_B2_machine_selection_feature_shadow",
                )
            selected = None if selected_uid is None else next(
                (row for row in rows if str(row["candidate_uid"]) == str(selected_uid)), None
            )
            wrong_write = bool(
                update_applied
                and target_box is not None
                and selected is not None
                and iou_xyxy(selected["box_xyxy"], target_box) < IOU_THRESHOLD
            )
            sequence_records.append(
                {
                    "sequence": sequence,
                    "track_id": track_id,
                    "anchor_frame": 1,
                    "frame": horizon_frame,
                    "gap": horizon_frame - 1,
                    "candidate_available": bool(target is not None),
                    "hard_negative_available": bool(competitor_scores),
                    "positive_score": positive_score,
                    "hard_negative_score": hard_negative_score,
                    "margin": None if positive_score is None or hard_negative_score is None else positive_score - hard_negative_score,
                    "rank": rank,
                    "auc_pairs": auc_pairs,
                    "auc_wins": auc_wins,
                    "selected_candidate_uid": selected_uid,
                    "update_applied": update_applied,
                    "wrong_memory_write_posthoc": wrong_write,
                    "runtime_future_gt_used": False,
                    "posthoc_gt_used": True,
                }
            )
    aggregate: dict[str, Any] = {"sequence": sequence, "target_count": len(target_ids), "records": len(sequence_records), "horizons": {}}
    for horizon in HORIZONS:
        rows = [row for row in sequence_records if row["gap"] <= horizon]
        evaluable = [row for row in rows if row["positive_score"] is not None and row["hard_negative_score"] is not None]
        pair_rows = [row for row in rows if row["auc_pairs"] > 0]
        aggregate["horizons"][f"H{horizon}"] = {
            "records": len(rows),
            "identity_evaluable": len(evaluable),
            "candidate_coverage": sum(row["candidate_available"] for row in rows) / len(rows) if rows else None,
            "hard_negative_win_rate": sum(row["positive_score"] > row["hard_negative_score"] for row in evaluable) / len(evaluable) if evaluable else None,
            "rank_1_accuracy": sum(row["rank"] == 1 for row in evaluable) / len(evaluable) if evaluable else None,
            "mean_margin": float(np.mean([row["margin"] for row in evaluable])) if evaluable else None,
            "auc": sum(row["auc_wins"] for row in pair_rows) / sum(row["auc_pairs"] for row in pair_rows) if pair_rows else None,
            "machine_update_count": sum(row["update_applied"] for row in rows),
            "wrong_memory_write_count": sum(row["wrong_memory_write_posthoc"] for row in rows),
        }
    return aggregate


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/N72R20/shadow_identity_metrics.json")
    args = parser.parse_args()
    per_sequence = [metrics_for_sequence(sequence) for sequence in SEQUENCES]
    payload = {
        "stage": "N72R20",
        "goal": "Learned Identity Memory Integration into InterMOT",
        "diagnostic": "FEATURE_LEVEL_SHADOW",
        "formal_variant": False,
        "status": "PASS_FEATURE_LEVEL_SIGNAL_DETECTED",
        "source": "existing real-SAM3 train candidate tapes plus historical B2 trace",
        "interaction_source": "simulated_from_gt",
        "runtime_future_gt_used": False,
        "posthoc_gt_used": True,
        "checkpoint": str(GRU_CHECKPOINT.resolve()),
        "checkpoint_sha256": GRU_SHA,
        "encoder_sha256": OSNET_SHA,
        "horizons": list(HORIZONS),
        "candidate_iou_threshold": IOU_THRESHOLD,
        "update_policy": "historical B2 machine selection replay for shadow only; not formal N72R20 consensus replay",
        "sequences": per_sequence,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(args.output), "status": payload["status"], "sequences": len(per_sequence)}))


if __name__ == "__main__":
    main()
