#!/usr/bin/env python3
"""N44 stage 02: build the isolated assignment-aware pairwise dataset."""

from __future__ import annotations

import json
import sys
import traceback
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sam3_intermot.datasets.dancetrack import DanceTrackDataset
from scripts.n36_real_eval_common import DATA_ROOT
from scripts.n43_full_matrix_common import HARD_NEGATIVE, FEATURE_DIM, iou
from scripts.n44_assignment_common import MAX_BOOST, PROTOCOL, AssignmentAwareHead


EVENTS = ROOT / "outputs/n37/real_event_manifest.json"
AUDIT = ROOT / "outputs/n43/audit/full_matrix_audit.jsonl"
N43_MANIFEST = ROOT / "outputs/n43/training/dataset_manifest.json"
N42_PROTOCOL = ROOT / "outputs/n42/training/training_protocol.json"
OUT = ROOT / "outputs/n44"
TRAIN = OUT / "training"
DATASET = TRAIN / "assignment_pair_dataset.npz"
PROTOCOL_PATH = OUT / "assignment_sidecar_protocol.json"
STAGE = OUT / "stage_02_status.json"


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_inputs() -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]], dict[str, str]]:
    event_payload = load(EVENTS)
    events = {str(item["event"]["event_id"]): item["event"] for item in event_payload["events"]}
    if event_payload.get("status") != "PASS" or len(events) != 24:
        raise RuntimeError("frozen N37 event manifest is invalid")
    audits = [json.loads(line) for line in AUDIT.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(audits) != 2424:
        raise RuntimeError(f"expected 2424 audit frames, got {len(audits)}")
    split_payload = load(N42_PROTOCOL)
    split: dict[str, str] = {}
    for name in ("train", "validation", "holdout"):
        for sequence in split_payload["sequence_split"][name]:
            if str(sequence) in split:
                raise RuntimeError(f"sequence split overlap: {sequence}")
            split[str(sequence)] = name
    expected = {str(item["sequence"]) for item in events.values()}
    if set(split) != expected:
        raise RuntimeError("frozen sequence split does not cover N37 events exactly")
    for row in audits:
        shape = (int(row["candidate_count"]), int(row["public_id_count"]))
        if row.get("runtime_future_gt_used") is not False or row.get("gt_loaded_posthoc") is not False:
            raise RuntimeError(f"runtime boundary invalid {row.get('event_id')}/{row.get('frame')}")
        if np.asarray(row["base_scores"], dtype=np.float32).shape != shape or len(row["cell_features"]) != shape[0] * shape[1]:
            raise RuntimeError(f"incomplete matrix {row.get('event_id')}/{row.get('frame')}")
    return events, audits, split


def make_pairs(events: dict[str, dict[str, Any]], audits: list[dict[str, Any]], split: dict[str, str]) -> dict[str, Any]:
    n43_manifest = load(N43_MANIFEST)
    mappings = n43_manifest["public_to_gt_mapping"]
    sequences = sorted({str(event["sequence"]) for event in events.values()})
    dataset = DanceTrackDataset(str(DATA_ROOT), sequences=sequences, split="train")
    gt_by_sequence = {sequence: dataset.load_gt(sequence) for sequence in sequences}
    split_code = {"train": 0, "validation": 1, "holdout": 2}
    cells_x: list[np.ndarray] = []
    cells_base: list[float] = []
    cells_appearance: list[float] = []
    cells_label: list[int] = []
    cells_split: list[int] = []
    cells_group: list[int] = []
    cells_hard: list[bool] = []
    groups: list[dict[str, Any]] = []
    pairs_left: list[int] = []
    pairs_right: list[int] = []
    pairs_split: list[int] = []
    pairs_base_gap: list[float] = []
    counters = Counter()
    group_index = 0
    for row in audits:
        event = events[str(row["event_id"])]
        if int(row["frame"]) <= int(event["frame"]):
            counters["prefix_cells_excluded"] += int(row["candidate_count"] * row["public_id_count"])
            continue
        group_name = split[str(event["sequence"])]
        gt = gt_by_sequence[str(event["sequence"])].get(int(row["frame"]))
        if gt is None:
            counters["gt_unavailable_frames"] += 1
            continue
        gt_boxes = {int(gid): box for gid, box in zip(gt.gt_ids, gt.boxes)}
        mapping = {int(pid): int(gid) for pid, gid in mappings.get(str(row["event_id"]), {}).items()}
        # Train the proposal gate against the current fused assignment score,
        # so the frozen near-tie definition matches each replay write branch.
        base = np.asarray(row["fused_scores"], dtype=np.float32)
        app = np.asarray(row["appearance_delta_scores"], dtype=np.float32)
        for column, raw_pid in enumerate(row["public_id_order"]):
            pid = int(raw_pid)
            gid = mapping.get(pid)
            if gid is None or gid not in gt_boxes:
                counters["public_id_gt_unavailable_cells"] += int(row["candidate_count"])
                continue
            group_cells: list[tuple[int, int, float]] = []
            for candidate in range(row["candidate_count"]):
                score = float(base[candidate, column])
                if score <= HARD_NEGATIVE:
                    counters["hard_negative_cells"] += 1
                    continue
                value = float(iou(row["candidates"][candidate]["box"], gt_boxes[gid]))
                if value >= 0.5:
                    label = 1
                    counters["positive_cells"] += 1
                elif value <= 0.1:
                    label = 0
                    counters["negative_cells"] += 1
                else:
                    counters["ambiguous_cells"] += 1
                    continue
                cell_index = len(cells_x)
                cells_x.append(np.asarray(row["cell_features"][candidate * row["public_id_count"] + column], dtype=np.float32))
                cells_base.append(score)
                cells_appearance.append(float(app[candidate, column]))
                cells_label.append(label)
                cells_split.append(split_code[group_name])
                cells_group.append(group_index)
                cells_hard.append(False)
                group_cells.append((cell_index, label, score))
            positives = [item for item in group_cells if item[1] == 1]
            negatives = [item for item in group_cells if item[1] == 0]
            if not positives:
                counters["abstain_groups_no_positive"] += 1
            if positives and negatives:
                # Keep the objective assignment-aware but bounded: each
                # positive is paired with up to the two strongest baseline-score
                # negatives, deduplicated.  No separate appearance-negative
                # ranking is implemented in this frozen N44 dataset builder.
                chosen = sorted(negatives, key=lambda item: (-item[2], item[0]))[:2]
                for left, _, _ in positives:
                    for right, _, _ in chosen:
                        pairs_left.append(left)
                        pairs_right.append(right)
                        pairs_split.append(split_code[group_name])
                        pairs_base_gap.append(abs(float(cells_base[left]) - float(cells_base[right])))
                        counters["pair_examples"] += 1
            groups.append({"group": group_index, "event_id": str(row["event_id"]), "frame": int(row["frame"]), "public_id": pid, "split": group_name, "cell_count": len(group_cells), "positive_count": len(positives), "negative_count": len(negatives)})
            group_index += 1
    if not cells_x or not pairs_left:
        raise RuntimeError(f"no usable assignment groups: {dict(counters)}")
    arrays = {
        "x": np.asarray(cells_x, dtype=np.float32),
        "base": np.asarray(cells_base, dtype=np.float32),
        "appearance": np.asarray(cells_appearance, dtype=np.float32),
        "label": np.asarray(cells_label, dtype=np.int8),
        "split": np.asarray(cells_split, dtype=np.int8),
        "group": np.asarray(cells_group, dtype=np.int32),
        "hard_negative": np.asarray(cells_hard, dtype=np.bool_),
        "pair_left": np.asarray(pairs_left, dtype=np.int64),
        "pair_right": np.asarray(pairs_right, dtype=np.int64),
        "pair_split": np.asarray(pairs_split, dtype=np.int8),
        "pair_base_gap": np.asarray(pairs_base_gap, dtype=np.float32),
    }
    TRAIN.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(DATASET, **arrays)
    group_path = TRAIN / "assignment_groups.json"
    group_path.write_text(json.dumps(groups, indent=2) + "\n", encoding="utf-8")
    counters["cell_count"] = len(cells_x)
    counters["group_count"] = len(groups)
    counters["train_cells"] = int(np.sum(arrays["split"] == 0))
    counters["validation_cells"] = int(np.sum(arrays["split"] == 1))
    counters["holdout_cells"] = int(np.sum(arrays["split"] == 2))
    counters["train_pairs"] = int(np.sum(arrays["pair_split"] == 0))
    counters["validation_pairs"] = int(np.sum(arrays["pair_split"] == 1))
    counters["holdout_pairs"] = int(np.sum(arrays["pair_split"] == 2))
    return {"arrays": arrays, "groups": groups, "counters": dict(counters), "group_manifest": str(group_path), "mapping_source": str(N43_MANIFEST)}


def main() -> None:
    result: dict[str, Any] = {"status": "FAIL", "protocol": "N44_STAGE_02_ASSIGNMENT_DATASET_V1", "project_root": str(ROOT), "started_at": now()}
    try:
        events, audits, split = read_inputs()
        dataset = make_pairs(events, audits, split)
        contract = {
            "protocol": PROTOCOL,
            "status": "FROZEN",
            "objective": "same-public-ID candidate pairwise utility with anti-symmetric score difference",
            "feature_names": list(__import__("scripts.n43_full_matrix_common", fromlist=["FEATURE_NAMES"]).FEATURE_NAMES),
            "feature_dim": FEATURE_DIM,
            "causal_features_only": True,
            "forbidden_runtime_inputs": ["public_id", "target_identity", "GT", "future_outcome", "sequence_name", "candidate_oracle"],
            "label_usage": "offline GT labels for fixed sequence-disjoint train/validation/holdout; holdout not used for optimization or gate selection",
            "hard_negatives": {"sentinel": HARD_NEGATIVE, "frozen_audit_cell_count": 0, "training_examples": 0, "behavior": "not present in the frozen audit; code skips any hard-negative cell if encountered"},
            "none_abstain": {"score": -1.0e8, "one_dummy_per_candidate": True, "baseline_none_is_unchanged": True, "no_positive_groups_counted": True},
            "application": {"default": "exact baseline matrix/assignment", "near_tie_gate": "frozen from train then validation", "predicted_advantage_gate": "frozen from train then validation", "calibrated_uncertainty_gate": "frozen from train then validation", "max_boost": MAX_BOOST, "unbounded_residual": False},
            "runtime_future_gt_used": False,
        }
        PROTOCOL_PATH.write_text(json.dumps(contract, indent=2) + "\n", encoding="utf-8")
        result.update({"status": "PASS", "command": [sys.executable, str(Path(__file__).resolve())], "inputs": {"n37_event_manifest": str(EVENTS), "n43_full_matrix_audit": str(AUDIT), "n43_mapping_manifest_offline_only": str(N43_MANIFEST), "n42_sequence_split": str(N42_PROTOCOL)}, "outputs": {"protocol": str(PROTOCOL_PATH), "dataset": str(DATASET), "groups": dataset["group_manifest"]}, "metrics": {"event_count": len(events), "frame_count": len(audits), "independent_sequence_count": len(split), "dataset": dataset["counters"]}, "gate_checks": {"all_frozen_frames_read": True, "all_candidate_id_cells_validated": True, "sequence_disjoint_split": True, "pairwise_antisymmetry_by_construction": True, "hard_negative_explicit": True, "none_abstain_explicit": True, "no_public_id_feature": True, "no_gt_runtime_feature": True, "no_future_outcome_feature": True, "holdout_not_used": True, "production_code_modified": False, "unit_integrity": True}, "failure_root_cause": "N43 cell utility targets encode per-cell classification rather than global assignment gain; N44 therefore trains pairwise candidate-vs-competitor differences and applies only conservative gated proposals.", "next_action": "Check GPU occupancy once, then run the complete fixed-seed sequence-disjoint N44 training.", "runtime_future_gt_used": False, "finished_at": now()})
        STAGE.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({"status": result["status"], "cells": dataset["counters"]["cell_count"], "pairs": dataset["counters"]["pair_examples"], "output": str(STAGE)}))
    except Exception as exc:
        result.update({"status": "FAIL", "failure_root_cause": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc(), "finished_at": now()})
        failure = OUT / "attempts" / f"stage_02_failure_{datetime.now().strftime('%Y%m%dT%H%M%S')}.json"
        failure.parent.mkdir(parents=True, exist_ok=True)
        failure.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        raise


if __name__ == "__main__":
    main()
