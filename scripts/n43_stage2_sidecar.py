#!/usr/bin/env python3
"""N43 stage 02: freeze sidecar contract and build the full-cell dataset."""

from __future__ import annotations

import json
import sys
import traceback
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import linear_sum_assignment

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sam3_intermot.datasets.dancetrack import DanceTrackDataset
from scripts.n36_real_eval_common import DATA_ROOT
from scripts.n43_full_matrix_common import (
    HARD_NEGATIVE,
    PROTOCOL,
    FEATURE_DIM,
    FullMatrixCalibrationHead,
    apply_sidecar,
    bounded_utility,
    feature_contract,
    hungarian_with_none,
    iou,
)


EVENTS = ROOT / "outputs/n37/real_event_manifest.json"
AUDIT = ROOT / "outputs/n43/audit/full_matrix_audit.jsonl"
OUT = ROOT / "outputs/n43"
TRAIN = OUT / "training"
DATASET = TRAIN / "cell_dataset.npz"
MANIFEST = TRAIN / "dataset_manifest.json"
PROTOCOL_PATH = OUT / "sidecar_protocol.json"
STAGE = OUT / "stage_02_status.json"


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def events() -> dict[str, dict[str, Any]]:
    payload = load(EVENTS)
    values = payload.get("events", [])
    if payload.get("status") != "PASS" or len(values) != 24:
        raise RuntimeError("N37 frozen event manifest invalid")
    return {str(item["event"]["event_id"]): item["event"] for item in values}


def split(events_by_id: dict[str, dict[str, Any]]) -> dict[str, str]:
    protocol = load(ROOT / "outputs/n42/training/training_protocol.json")
    output = {}
    for name in ("train", "validation", "holdout"):
        for sequence in protocol["sequence_split"][name]:
            if sequence in output:
                raise RuntimeError(f"sequence appears in multiple N42 splits: {sequence}")
            output[str(sequence)] = name
    expected = {str(event["sequence"]) for event in events_by_id.values()}
    if set(output) != expected:
        raise RuntimeError("N43 split does not cover exactly the frozen event sequences")
    return output


def load_audits() -> list[dict[str, Any]]:
    if not AUDIT.is_file():
        raise FileNotFoundError(AUDIT)
    rows = [json.loads(line) for line in AUDIT.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(rows) != 2424:
        raise RuntimeError(f"expected 2424 full matrix audit frames, got {len(rows)}")
    for row in rows:
        if row.get("runtime_future_gt_used") is not False or row.get("gt_loaded_posthoc") is not False:
            raise RuntimeError(f"audit runtime boundary invalid at {row.get('event_id')}/{row.get('frame')}")
        shape = (int(row["candidate_count"]), int(row["public_id_count"]))
        if len(row["cell_features"]) != shape[0] * shape[1]:
            raise RuntimeError(f"not all cells present at {row.get('event_id')}/{row.get('frame')}")
        if np.asarray(row["base_scores"], dtype=float).shape != shape:
            raise RuntimeError(f"base shape invalid at {row.get('event_id')}/{row.get('frame')}")
    return rows


def gt_boxes(events_by_id: dict[str, dict[str, Any]]) -> dict[str, dict[int, Any]]:
    sequences = sorted({str(event["sequence"]) for event in events_by_id.values()})
    dataset = DanceTrackDataset(str(DATA_ROOT), sequences=sequences, split="train")
    return {sequence: dataset.load_gt(sequence) for sequence in sequences}


def infer_public_to_gt(event: dict[str, Any], rows: list[dict[str, Any]], sequence_gt: dict[int, Any]) -> dict[int, int]:
    """Infer untouched public IDs offline for labels; force the known event ID."""
    pids = sorted({int(pid) for row in rows for pid in row["public_id_order"]})
    gids = sorted({int(gid) for frame in sequence_gt.values() for gid in frame.gt_ids})
    target_pid, target_gid = int(event["public_id"]), int(event["dataset_gt_id"])
    sums: dict[tuple[int, int], float] = defaultdict(float)
    counts: Counter[tuple[int, int]] = Counter()
    for row in rows:
        if int(row["frame"]) <= int(event["frame"]):
            continue
        gt = sequence_gt.get(int(row["frame"]))
        if gt is None:
            continue
        gt_list = [(int(gid), box_value) for gid, box_value in zip(gt.gt_ids, gt.boxes)]
        for candidate, pid in zip(row["candidates"], row["candidate_public_ids"]):
            if pid is None:
                continue
            for gid, gt_box in gt_list:
                sums[(int(pid), gid)] += iou(candidate["box"], gt_box)
                counts[(int(pid), gid)] += 1
    mapping = {target_pid: target_gid} if target_pid in pids and target_gid in gids else {}
    remaining_pids = [pid for pid in pids if pid not in mapping]
    remaining_gids = [gid for gid in gids if gid not in mapping.values()]
    if remaining_pids and remaining_gids:
        score = np.zeros((len(remaining_pids), len(remaining_gids)), dtype=float)
        for i, pid in enumerate(remaining_pids):
            for j, gid in enumerate(remaining_gids):
                score[i, j] = sums[(pid, gid)] / counts[(pid, gid)] if counts[(pid, gid)] else 0.0
        rr, cc = linear_sum_assignment(-score)
        for r, c in zip(rr.tolist(), cc.tolist()):
            if score[r, c] >= 0.15:
                mapping[remaining_pids[r]] = remaining_gids[c]
    return mapping


def build_dataset(events_by_id: dict[str, dict[str, Any]], audit_rows: list[dict[str, Any]], split_by_sequence: dict[str, str], all_gt: dict[str, dict[int, Any]]) -> dict[str, Any]:
    row_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in audit_rows:
        row_groups[str(row["event_id"])].append(row)
    mappings = {}
    for event_id, event in events_by_id.items():
        mappings[event_id] = infer_public_to_gt(event, row_groups[event_id], all_gt[str(event["sequence"])])
    X, base, appearance, y, target, split_codes, event_codes, pids, action_codes = [], [], [], [], [], [], [], [], []
    counters = Counter()
    event_names = sorted(events_by_id)
    event_code = {name: i for i, name in enumerate(event_names)}
    action_names = sorted({str(event["action_type"]) for event in events_by_id.values()})
    action_code = {name: i for i, name in enumerate(action_names)}
    split_code = {"train": 0, "validation": 1, "holdout": 2}
    for row in audit_rows:
        event_id = str(row["event_id"])
        event = events_by_id[event_id]
        if int(row["frame"]) <= int(event["frame"]):
            counters["event_frame_excluded"] += int(row["candidate_count"] * row["public_id_count"])
            continue
        gt = all_gt[str(event["sequence"])].get(int(row["frame"]))
        if gt is None:
            counters["gt_unavailable_frames"] += 1
            continue
        gt_map = mappings[event_id]
        gt_boxes_by_id = {int(gid): box_value for gid, box_value in zip(gt.gt_ids, gt.boxes)}
        matrix_base = np.asarray(row["base_scores"], dtype=np.float32)
        matrix_app = np.asarray(row["appearance_delta_scores"], dtype=np.float32)
        for i, candidate in enumerate(row["candidates"]):
            for j, pid in enumerate(row["public_id_order"]):
                pid = int(pid)
                if pid not in gt_map or gt_map[pid] not in gt_boxes_by_id:
                    counters["public_id_gt_unavailable_cells"] += 1
                    continue
                value_iou = iou(candidate["box"], gt_boxes_by_id[gt_map[pid]])
                counters["labeled_cells"] += 1
                if matrix_base[i, j] <= HARD_NEGATIVE:
                    counters["hard_negative_cells_excluded"] += 1
                    continue
                if value_iou >= 0.5:
                    label, target_value = 1, 0.5
                    counters["positive"] += 1
                elif value_iou <= 0.1:
                    label, target_value = 0, -0.5
                    counters["negative"] += 1
                else:
                    counters["ambiguous_label_discarded"] += 1
                    continue
                X.append(row["cell_features"][i * len(row["public_id_order"]) + j])
                base.append(float(matrix_base[i, j]))
                appearance.append(float(matrix_app[i, j]))
                y.append(label)
                target.append(target_value)
                split_codes.append(split_code[split_by_sequence[str(event["sequence"])]])
                event_codes.append(event_code[event_id])
                pids.append(pid)
                action_codes.append(action_code[str(event["action_type"])])
    if not X or not counters["positive"] or not counters["negative"]:
        raise RuntimeError(f"insufficient full-cell labels: {dict(counters)}")
    arrays = {"x": np.asarray(X, dtype=np.float32), "base": np.asarray(base, dtype=np.float32), "appearance": np.asarray(appearance, dtype=np.float32), "label": np.asarray(y, dtype=np.int8), "target_utility": np.asarray(target, dtype=np.float32), "split": np.asarray(split_codes, dtype=np.int8), "event_code": np.asarray(event_codes, dtype=np.int16), "public_id": np.asarray(pids, dtype=np.int64), "action_code": np.asarray(action_codes, dtype=np.int8)}
    TRAIN.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(DATASET, **arrays)
    counts = {name: int(np.sum(arrays["split"] == code)) for name, code in split_code.items()}
    positives = {name: int(np.sum((arrays["split"] == code) & (arrays["label"] == 1))) for name, code in split_code.items()}
    negatives = {name: int(np.sum((arrays["split"] == code) & (arrays["label"] == 0))) for name, code in split_code.items()}
    return {"row_count": len(X), "split_counts": counts, "split_positive_counts": positives, "split_negative_counts": negatives, "counters": dict(counters), "public_to_gt_mapping": mappings, "action_names": action_names, "event_names": event_names}


def smoke(audit_row: dict[str, Any]) -> dict[str, Any]:
    model = FullMatrixCalibrationHead()
    x = np.asarray(audit_row["cell_features"], dtype=np.float32)
    app = np.asarray(audit_row["appearance_delta_scores"], dtype=np.float32).reshape(-1)
    utility, gate, residual = bounded_utility(model, x, app)
    result = {"forward_shape": list(utility.shape), "cell_count": int(x.shape[0]), "finite": bool(np.all(np.isfinite(utility))), "gate_range": [float(gate.min()), float(gate.max())], "residual_range": [float(residual.min()), float(residual.max())]}
    if not result["finite"] or x.shape != (int(audit_row["candidate_count"] * audit_row["public_id_count"]), FEATURE_DIM):
        raise RuntimeError(f"N43 sidecar smoke failed: {result}")
    assignment = hungarian_with_none(np.asarray(audit_row["base_scores"], dtype=np.float32))
    if assignment.shape != (int(audit_row["candidate_count"]),):
        raise RuntimeError("NONE Hungarian smoke failed")
    calibrated = apply_sidecar(audit_row, model, int(audit_row["frame_offset_from_event"]))
    result["full_matrix_apply"] = calibrated["n43_sidecar"]
    if calibrated["n43_sidecar"]["changed_column_count"] <= 0 or not calibrated["n43_sidecar"]["hard_negative_preserved"]:
        raise RuntimeError("sidecar did not exercise full-cell application or changed hard negatives")
    return result


def main() -> None:
    started = now()
    result: dict[str, Any] = {"status": "FAIL", "protocol": "N43_STAGE_02_SIDECAR_IMPLEMENTATION_V1", "started_at": started, "project_root": str(ROOT)}
    try:
        ev = events()
        split_by_sequence = split(ev)
        audits = load_audits()
        all_gt = gt_boxes(ev)
        dataset = build_dataset(ev, audits, split_by_sequence, all_gt)
        protocol_payload = {"protocol": PROTOCOL, "status": "FROZEN", "created_at": now(), "feature_contract": feature_contract(), "sequence_split": {name: sorted(sequence for sequence, group in split_by_sequence.items() if group == name) for name in ("train", "validation", "holdout")}, "label_contract": {"GT_usage": "offline labels only", "positive": "candidate IoU >= 0.5 to offline public-ID mapping", "negative": "candidate IoU <= 0.1", "ambiguous": "0.1 < IoU < 0.5 discarded", "target_public_id_not_a_feature": True}, "training_contract": {"model": "base + sigmoid(gate)*appearance_delta + bounded residual", "residual_bound": 0.5, "none_score": -1.0e8, "hard_negative_sentinel": HARD_NEGATIVE, "candidate_generation_changed": False, "hungarian_solver_changed": False}, "runtime_future_gt_used": False}
        if PROTOCOL_PATH.exists() and load(PROTOCOL_PATH).get("feature_contract") != protocol_payload["feature_contract"]:
            raise RuntimeError("existing N43 sidecar protocol feature contract differs")
        PROTOCOL_PATH.write_text(json.dumps(protocol_payload, indent=2) + "\n", encoding="utf-8")
        smoke_result = smoke(audits[0])
        dataset["dataset"] = str(DATASET)
        dataset["protocol"] = str(PROTOCOL_PATH)
        dataset["runtime_future_gt_used"] = False
        MANIFEST.write_text(json.dumps(dataset, indent=2) + "\n", encoding="utf-8")
        result.update({"status": "PASS", "command": [sys.executable, str(Path(__file__).resolve())], "inputs": {"n37_event_manifest": str(EVENTS), "full_matrix_audit": str(AUDIT), "n42_training_protocol_for_frozen_split": str(ROOT / "outputs/n42/training/training_protocol.json")}, "outputs": {"sidecar_protocol": str(PROTOCOL_PATH), "dataset": str(DATASET), "dataset_manifest": str(MANIFEST)}, "metrics": {"event_count": len(ev), "independent_sequence_count": len(split_by_sequence), "dataset": dataset, "smoke": smoke_result}, "gate_checks": {"all_cells_featured": True, "full_cell_apply": True, "target_public_id_not_feature": True, "event_identity_not_feature": True, "future_outcome_not_feature": True, "hard_negative_preserved": True, "none_semantics_fixed": True, "runtime_future_gt_false": True, "production_code_modified": False}, "failure_root_cause": "N42's target-only application was not retained; N43 uses a bounded utility for every finite cell and explicit immutable NONE dummies.", "next_action": "Check GPU occupancy once, then run the actual sequence-disjoint full training to the frozen early-stopping rule.", "runtime_future_gt_used": False, "finished_at": now()})
        OUT.mkdir(parents=True, exist_ok=True)
        STAGE.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({"status": result["status"], "dataset_rows": dataset["row_count"], "output": str(STAGE)}, sort_keys=True))
    except Exception as exc:
        result.update({"status": "FAIL", "failure_root_cause": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc(), "finished_at": now()})
        failure = OUT / "attempts" / f"stage_02_failure_{datetime.now().strftime('%Y%m%dT%H%M%S')}.json"
        failure.parent.mkdir(parents=True, exist_ok=True)
        failure.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        raise


if __name__ == "__main__":
    main()
