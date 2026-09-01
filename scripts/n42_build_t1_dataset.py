#!/usr/bin/env python3
"""Build the frozen, sequence-split N42 T1 training dataset.

GT is read only here to create offline pairwise labels.  The resulting
dataset contains audit features and labels, not future GT input for a runtime
worker.  The split and label rule are frozen before any rows are materialized.
"""

from __future__ import annotations

import hashlib
import json
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sam3_intermot.datasets.dancetrack import DanceTrackDataset
from scripts.n36_real_eval_common import DATA_ROOT, atomic_json, atomic_jsonl, finite_iou
from scripts.n42_t1_common import FEATURE_NAMES, pair_feature_from_audit


N37_MANIFEST = ROOT / "outputs/n37/real_event_manifest.json"
N41_PAIR_TABLE = ROOT / "outputs/n41/diagnostic/candidate_pair_diagnostics.jsonl"
N41_ARTIFACT_ROOT = ROOT / "outputs/n41/source_replay/full/attempt1"
N42_SOURCE_MANIFEST = ROOT / "outputs/n42/diagnostic/source_embedding_manifest.json"
OUT = ROOT / "outputs/n42/training"
PROTOCOL_PATH = OUT / "training_protocol.json"
DATASET_PATH = OUT / "pair_dataset.jsonl"
MANIFEST_PATH = OUT / "dataset_manifest.json"
FAILURE_PATH = ROOT / "outputs/n42/attempts/dataset_build_failure.json"

PROTOCOL = "N42_T1_PAIRWISE_CALIBRATION_TRAINING_V1"
SEED = 4242
LABEL_MARGIN = 0.05
TRAIN_CONFIG = {
    "epochs": 30,
    "batch_size": 256,
    "learning_rate": 1.0e-3,
    "weight_decay": 1.0e-4,
    "early_stopping_patience": 5,
    "selection": "minimum_validation_bce; ties keep earliest epoch",
    "calibration_application_scale": 1.0,
}


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_events() -> tuple[dict[str, Any], list[dict[str, Any]]]:
    payload = json.loads(N37_MANIFEST.read_text(encoding="utf-8"))
    events = payload.get("events")
    if payload.get("status") != "PASS" or payload.get("event_count") != 24 or not isinstance(events, list) or len(events) != 24:
        raise RuntimeError("frozen N37 manifest is not PASS/24")
    if len({str(x["event"]["event_id"]) for x in events}) != 24:
        raise RuntimeError("duplicate N37 event IDs")
    if len({str(x["event"]["sequence"]) for x in events}) != 21:
        raise RuntimeError("N37 sequence count is not 21")
    return payload, events


def build_split(events: list[dict[str, Any]]) -> dict[str, list[str]]:
    sequences = sorted({str(item["event"]["sequence"]) for item in events})
    if len(sequences) != 21:
        raise RuntimeError("expected exactly 21 independent sequences")
    return {"train": sequences[:14], "validation": sequences[14:17], "holdout": sequences[17:]}


def protocol_payload(n37: dict[str, Any], events: list[dict[str, Any]]) -> dict[str, Any]:
    split = build_split(events)
    return {
        "protocol": PROTOCOL,
        "status": "FROZEN_BEFORE_DATASET_MATERIALIZATION",
        "created_at": now(),
        "seed": SEED,
        "input_contract": {
            "n37_manifest": str(N37_MANIFEST.relative_to(ROOT)),
            "n37_manifest_sha256": sha256(N37_MANIFEST),
            "n41_candidate_pair_table": str(N41_PAIR_TABLE.relative_to(ROOT)),
            "n41_candidate_pair_table_sha256": sha256(N41_PAIR_TABLE),
            "n42_corrected_source_manifest": str(N42_SOURCE_MANIFEST.relative_to(ROOT)),
            "n42_corrected_source_manifest_sha256": sha256(N42_SOURCE_MANIFEST),
            "source_branch": "N41 M2 memory_write=True lambda_assoc=1 human_weight=1 audit matrices",
        },
        "sequence_split": {
            "method": "lexicographic sequence order; first 14 train, next 3 validation, last 4 holdout",
            "train": split["train"],
            "validation": split["validation"],
            "holdout": split["holdout"],
            "frame_random_split": False,
            "future_outcome_or_metric_used_for_split": False,
        },
        "label_rule": {
            "source": "offline current/future-frame DanceTrack GT only during dataset construction",
            "target_iou_minus_competitor_iou_threshold": LABEL_MARGIN,
            "positive": "target_iou - competitor_iou >= 0.05",
            "negative": "target_iou - competitor_iou <= -0.05",
            "discard": "absolute difference < 0.05",
            "h20_h50_h100_or_idsw_used": False,
        },
        "feature_contract": {
            "feature_names": list(FEATURE_NAMES),
            "input_dim": len(FEATURE_NAMES),
            "runtime_gt_used": False,
            "public_id_in_feature": False,
            "candidate_generation_changed": False,
            "hungarian_solver_changed": False,
        },
        "training": TRAIN_CONFIG,
        "smoke": {"max_rows": 512, "steps": 3, "must_save_and_reload": True},
        "runtime_boundary": {
            "runtime_future_gt_used": False,
            "gt_loaded_only_for_offline_label_materialization": True,
            "model_selection_uses_holdout": False,
        },
    }


def load_event_map(events: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {str(item["event"]["event_id"]): item for item in events}


def artifact_for(event_id: str) -> Path:
    return N41_ARTIFACT_ROOT / event_id / "A_ideal_gt_roi" / "lambda_1_human_1.json"


def load_audits(events: list[dict[str, Any]]) -> dict[str, dict[int, dict[str, Any]]]:
    output: dict[str, dict[int, dict[str, Any]]] = {}
    for item in events:
        event_id = str(item["event"]["event_id"])
        path = artifact_for(event_id)
        if not path.is_file():
            raise FileNotFoundError(path)
        artifact = json.loads(path.read_text(encoding="utf-8"))
        if artifact.get("status") != "PASS" or artifact.get("runtime_boundary", {}).get("runtime_future_gt_used") is not False:
            raise RuntimeError(f"N41 worker artifact is not a valid frozen runtime input: {path}")
        trace = artifact["variants"]["M2"]["branches"]["memory_write=True"]["future_trace"]
        by_frame = {}
        for entry in trace:
            frame = int(entry["frame"])
            if frame in by_frame:
                raise RuntimeError(f"duplicate artifact frame {event_id}/{frame}")
            audit = entry.get("candidate_audit")
            if not isinstance(audit, dict):
                raise RuntimeError(f"missing candidate audit {event_id}/{frame}")
            by_frame[frame] = audit
        output[event_id] = by_frame
    return output


def main() -> None:
    started = now()
    try:
        n37, events = load_events()
        protocol = protocol_payload(n37, events)
        if PROTOCOL_PATH.exists():
            existing = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
            fields = ("protocol", "seed", "input_contract", "sequence_split", "label_rule", "feature_contract", "training", "smoke", "runtime_boundary")
            if {key: existing.get(key) for key in fields} != {key: protocol.get(key) for key in fields}:
                raise RuntimeError("existing training protocol differs; refusing to overwrite")
        else:
            atomic_json(PROTOCOL_PATH, protocol)
        split = protocol["sequence_split"]
        split_groups = {name: list(split[name]) for name in ("train", "validation", "holdout")}
        split_by_sequence = {sequence: name for name, values in split_groups.items() for sequence in values}
        audits = load_audits(events)
        event_map = load_event_map(events)
        dataset = DanceTrackDataset(str(DATA_ROOT), sequences=sorted(split_by_sequence), split="train")
        gt_by_sequence = {sequence: dataset.load_gt(sequence) for sequence in sorted(split_by_sequence)}
        rows: list[dict[str, Any]] = []
        counters = {"input_pair_rows": 0, "future_pair_rows": 0, "feature_unavailable": 0, "gt_unavailable": 0, "ambiguous_label_discarded": 0, "positive": 0, "negative": 0}
        with N41_PAIR_TABLE.open(encoding="utf-8") as handle:
            for line_no, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                pair = json.loads(line)
                counters["input_pair_rows"] += 1
                if int(pair.get("frame_offset_from_event", 0)) <= 0:
                    continue
                counters["future_pair_rows"] += 1
                event_id = str(pair["event_id"])
                item = event_map.get(event_id)
                if item is None:
                    raise RuntimeError(f"pair row event is not in frozen N37 manifest: line {line_no}")
                sequence = str(item["event"]["sequence"])
                frame = int(pair["frame"])
                audit = audits[event_id].get(frame)
                if audit is None:
                    raise RuntimeError(f"pair frame missing from frozen worker artifact: {event_id}/{frame}")
                if pair.get("runtime_future_gt_used") is not False:
                    raise RuntimeError(f"pair row runtime future GT flag is not false: line {line_no}")
                state_pid = int(pair["target_state_public_id"])
                feature = pair_feature_from_audit(audit, int(pair["target_candidate_index"]), int(pair["competitor_candidate_index"]), state_pid, int(pair["frame_offset_from_event"]))
                if feature is None:
                    counters["feature_unavailable"] += 1
                    continue
                gt = gt_by_sequence[sequence].get(frame)
                if gt is None:
                    counters["gt_unavailable"] += 1
                    continue
                target_gid = int(item["event"]["dataset_gt_id"])
                target_box = next((box for gid, box in zip(gt.gt_ids, gt.boxes) if int(gid) == target_gid), None)
                if target_box is None:
                    counters["gt_unavailable"] += 1
                    continue
                by_index = {int(c.get("index", i)): c for i, c in enumerate(audit.get("candidates", []))}
                left = by_index.get(int(pair["target_candidate_index"]))
                right = by_index.get(int(pair["competitor_candidate_index"]))
                if left is None or right is None:
                    counters["feature_unavailable"] += 1
                    continue
                left_iou = finite_iou(left["box"], target_box)
                right_iou = finite_iou(right["box"], target_box)
                difference = float(left_iou - right_iou)
                if abs(difference) < LABEL_MARGIN:
                    counters["ambiguous_label_discarded"] += 1
                    continue
                label = 1 if difference > 0.0 else 0
                counters["positive" if label else "negative"] += 1
                rows.append({
                    "event_id": event_id,
                    "sequence": sequence,
                    "action_type": str(item["event"]["action_type"]),
                    "frame": frame,
                    "event_frame": int(item["event"]["frame"]),
                    "frame_offset_from_event": int(pair["frame_offset_from_event"]),
                    "state_public_id": state_pid,
                    "left_candidate_index": int(pair["target_candidate_index"]),
                    "right_candidate_index": int(pair["competitor_candidate_index"]),
                    "left_iou_posthoc": float(left_iou),
                    "right_iou_posthoc": float(right_iou),
                    "iou_difference_posthoc": difference,
                    "label": label,
                    "features": feature.astype(float).tolist(),
                    "split": split_by_sequence[sequence],
                    "runtime_future_gt_used": False,
                    "label_source": "offline_DanceTrack_GT_posthoc_only",
                })
        if not rows or not counters["positive"] or not counters["negative"]:
            raise RuntimeError(f"dataset has insufficient labeled rows: {counters}")
        rows.sort(key=lambda row: (row["split"], row["sequence"], row["event_id"], row["frame"], row["left_candidate_index"], row["right_candidate_index"]))
        atomic_jsonl(DATASET_PATH, rows)
        split_counts = {name: sum(1 for row in rows if row["split"] == name) for name in ("train", "validation", "holdout")}
        split_positive = {name: sum(int(row["label"]) for row in rows if row["split"] == name) for name in split_counts}
        manifest = {
            "protocol": PROTOCOL,
            "status": "PASS",
            "created_at": now(),
            "finished_at": now(),
            "dataset": str(DATASET_PATH.relative_to(ROOT)),
            "dataset_sha256": sha256(DATASET_PATH),
            "protocol_sha256": sha256(PROTOCOL_PATH),
            "row_count": len(rows),
            "split_counts": split_counts,
            "split_positive_counts": split_positive,
            "split_negative_counts": {name: split_counts[name] - split_positive[name] for name in split_counts},
            "event_count": 24,
            "independent_sequence_count": 21,
            "action_counts": {action: sum(1 for row in rows if row["action_type"] == action) for action in ("ADD_NEW_IDENTITY", "AUTHORITATIVE_REASSIGN", "ATOMIC_ID_SWAP", "RECOVER_IDENTITY")},
            "counters": counters,
            "gt_usage": "offline label materialization only; no H20/H50/H100/IDSW/future outcome used",
            "runtime_future_gt_used": False,
            "holdout_used_for_training_or_selection": False,
        }
        atomic_json(MANIFEST_PATH, manifest)
        print(json.dumps({"status": "PASS", "row_count": len(rows), "split_counts": split_counts, "counters": counters}, sort_keys=True), flush=True)
    except Exception as exc:
        failure = {"protocol": PROTOCOL, "status": "FAIL", "started_at": started, "finished_at": now(), "exception": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc(), "failure_preserved": True}
        if not FAILURE_PATH.exists():
            atomic_json(FAILURE_PATH, failure)
        raise


if __name__ == "__main__":
    main()
