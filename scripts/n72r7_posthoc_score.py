#!/usr/bin/env python3
"""Validate sealed N72R7 replays, then score them with posthoc train GT.

The runtime validation phase intentionally does not open the dataset GT.  GT
is loaded only after all D1/D2 event artifacts and their hashes have passed
the causal/integrity checks.
"""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import traceback
from typing import Any, Mapping, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sam3_intermot.evaluation.interaction_effect_metrics import (  # noqa: E402
    AssignmentChangeType,
    metric_record,
    sequence_cluster_bootstrap,
)


N72R6_ROOT = ROOT / "outputs/N72R6/public_replay/human_anchor_fallback_attempt1"
N72R7_ROOT = ROOT / "outputs/N72R7"
TARGET_MANIFEST = ROOT / "outputs/N72R6/recovery_target_stream_manifest_attempt3.json"
EVENT_POLICY = ROOT / "outputs/N72R5/mechanism_rounds/round_06_event_policy/real_event_manifest.json"
D1_ROOT = N72R7_ROOT / "dev_replay/d1_full_attempt2"
D2_ROOT = N72R7_ROOT / "dev_replay/d2_full_attempt1"
POSTHOC_ROOT = N72R7_ROOT / "posthoc_attempt2"
RUNTIME_VALIDATION_PATH = POSTHOC_ROOT / "runtime_validation.json"
RESULT_PATH = POSTHOC_ROOT / "n72r7_d1_d2_posthoc_results.json"
EVENT_METRICS_PATH = POSTHOC_ROOT / "event_metrics.jsonl"
STAGE_PATH = N72R7_ROOT / "stage_05_status_attempt2.json"
CONTROLLER_PATH = N72R7_ROOT / "CONTROLLER_STATUS_attempt2.json"
HUMAN_STATUS_PATH = N72R7_ROOT / "HUMAN_READABLE_STATUS_attempt2.md"
FAILURE_ROOT = N72R7_ROOT / "attempts"

HORIZONS = (20, 50, 100)
IOU_THRESHOLD = 0.5
BOOTSTRAP_SEED = 7202
BOOTSTRAP_REPETITIONS = 2000
EVENT_VARIANTS = ("D0", "D1", "D2")
COMPARISONS = {
    "D1_vs_D0": ("D0", "D1"),
    "D2_vs_D0": ("D0", "D2"),
    "D2_vs_D1": ("D1", "D2"),
}


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        directory_fd = os.open(path.parent, os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if temporary.exists():
            temporary.unlink()


def atomic_json(path: Path, payload: Any) -> None:
    atomic_write(path, json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n")


def atomic_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    atomic_write(
        path,
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n" for row in rows),
    )


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


def _path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def _assert_runtime_value_clean(value: Any, location: str) -> None:
    forbidden = {"dataset_gt_id", "gt_box", "future_gt", "future_identity_error", "h20", "h50", "h100"}
    if isinstance(value, Mapping):
        for key, nested in value.items():
            key_text = str(key).lower()
            if key_text in forbidden:
                raise RuntimeError(f"forbidden posthoc/runtime field {location}/{key}")
            _assert_runtime_value_clean(nested, f"{location}/{key}")
    elif isinstance(value, list):
        for index, nested in enumerate(value):
            _assert_runtime_value_clean(nested, f"{location}/{index}")


def _box_iou(left: Any, right: Any) -> float:
    a = np.asarray(left, dtype=np.float64).reshape(-1)
    b = np.asarray(right, dtype=np.float64).reshape(-1)
    if a.size != 4 or b.size != 4 or not np.all(np.isfinite(a)) or not np.all(np.isfinite(b)):
        return 0.0
    if a[2] <= a[0] or a[3] <= a[1] or b[2] <= b[0] or b[3] <= b[1]:
        return 0.0
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    intersection = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    union = area_a + area_b - intersection
    return float(intersection / union) if union > 0.0 else 0.0


def _candidate_best_iou(row: Mapping[str, Any], gt_box: Sequence[float]) -> tuple[float, dict[str, Any] | None]:
    values = [(_box_iou(item.get("box_xyxy"), gt_box), item) for item in row.get("candidate_rows", [])]
    return max(values, key=lambda pair: (pair[0], -int(pair[1].get("candidate_index", 0))), default=(0.0, None))


def _public_box_for_gt(row: Mapping[str, Any], public_id: int, gt_box: Sequence[float]) -> tuple[float, dict[str, Any] | None]:
    candidates = [
        item
        for item in row.get("candidate_rows", [])
        if item.get("public_id") is not None and int(item["public_id"]) == int(public_id)
    ]
    if not candidates:
        return 0.0, None
    return max(
        ((_box_iou(item.get("box_xyxy"), gt_box), item) for item in candidates),
        key=lambda pair: (pair[0], -int(pair[1].get("candidate_index", 0))),
    )


def _public_map(row: Mapping[str, Any]) -> dict[str, int | None]:
    return {str(item["candidate_uid"]): item.get("public_id") for item in row.get("candidate_rows", [])}


def _target_binding(row: Mapping[str, Any], target_public: int) -> tuple[str | None, dict[str, Any] | None]:
    matches = [
        item
        for item in row.get("candidate_rows", [])
        if item.get("public_id") is not None and int(item["public_id"]) == int(target_public)
    ]
    if len(matches) > 1:
        raise RuntimeError(f"duplicate target public assignment: {row.get('event_id')}/{row.get('frame')}")
    if not matches:
        return None, None
    return str(matches[0]["candidate_uid"]), matches[0]


def _metric_template() -> dict[str, Any]:
    return {
        "window_frame_count": 0,
        "evaluated_frames": 0,
        "target_gt_visible_frames": 0,
        "target_gt_absent_frames": 0,
        "baseline_iou_sum": 0.0,
        "treatment_iou_sum": 0.0,
        "delta_iou_sum": 0.0,
        "baseline_correct_frames": 0,
        "treatment_correct_frames": 0,
        "baseline_identity_error_frames": 0,
        "treatment_identity_error_frames": 0,
        "identity_error_reduction_sum": 0.0,
        "target_missing_frames": 0,
        "wrong_reassociation_frames": 0,
        "candidate_present_frames": 0,
        "assignment_change_count": 0,
        "target_assignment_change_count": 0,
        "global_common_assignment_change_count": 0,
        "true_correct_crossing_count": 0,
        "true_incorrect_crossing_count": 0,
        "directional_improvement_count": 0,
        "directional_regression_count": 0,
        "neutral_change_count": 0,
        "id_switch_count": 0,
        "recorrection_opportunity_count": 0,
        "raw_switch_count": 0,
        "posthoc_correct_switch_count": 0,
        "posthoc_wrong_switch_count": 0,
        "posthoc_unassessable_switch_count": 0,
        "protected_compared": 0,
        "protected_regression_count": 0,
        "protected_improvement_count": 0,
        "baseline_identity_error": None,
        "treatment_identity_error": None,
        "identity_error_reduction": None,
        "baseline_mean_iou": None,
        "treatment_mean_iou": None,
        "delta_iou": None,
        "missing_rate": None,
        "wrong_reassociation_rate": None,
        "candidate_recall": None,
        "assignment_change_rate": None,
        "target_assignment_change_rate": None,
        "id_switch_rate": None,
        "recorrection_rate": None,
        "protected_regression_rate": None,
    }


def _finalize_metric(metric: dict[str, Any]) -> dict[str, Any]:
    frames = int(metric["evaluated_frames"])
    metric["window_frame_count"] = int(metric.get("window_frame_count", 0))
    if frames:
        metric["baseline_mean_iou"] = float(metric["baseline_iou_sum"] / frames)
        metric["treatment_mean_iou"] = float(metric["treatment_iou_sum"] / frames)
        metric["delta_iou"] = float(metric["delta_iou_sum"] / frames)
        metric["baseline_identity_error"] = float(metric["baseline_identity_error_frames"] / frames)
        metric["treatment_identity_error"] = float(metric["treatment_identity_error_frames"] / frames)
        metric["identity_error_reduction"] = float(metric["identity_error_reduction_sum"] / frames)
        metric["missing_rate"] = float(metric["target_missing_frames"] / frames)
        metric["wrong_reassociation_rate"] = float(metric["wrong_reassociation_frames"] / frames)
        metric["candidate_recall"] = float(metric["candidate_present_frames"] / frames)
        metric["assignment_change_rate"] = float(metric["assignment_change_count"] / frames)
        metric["target_assignment_change_rate"] = float(metric["target_assignment_change_count"] / frames)
        metric["id_switch_rate"] = float(metric["id_switch_count"] / frames)
        metric["recorrection_rate"] = float(metric["recorrection_opportunity_count"] / frames)
    if int(metric["protected_compared"]):
        metric["protected_regression_rate"] = float(metric["protected_regression_count"] / metric["protected_compared"])
    return metric


def _merge_metric(destination: dict[str, Any], source: Mapping[str, Any]) -> None:
    for key, value in source.items():
        if key.endswith("_sum") or key.endswith("_frames") or key.endswith("_count") or key.endswith("_compared") or key == "evaluated_frames" or key == "window_frame_count":
            if isinstance(value, (int, float)):
                destination[key] = destination.get(key, 0) + value


def _load_gt(sequence: str) -> dict[int, dict[int, dict[str, Any]]]:
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
            box = [x, y, x + width, y + height]
            if not np.all(np.isfinite(np.asarray(box, dtype=np.float64))):
                raise ValueError(f"non-finite GT box {path}:{line_number}")
            result[frame_one_based - 1][dataset_id] = {"box": box}
    return result


def _load_runtime_scenarios() -> dict[str, dict[str, Any]]:
    target = read_json(TARGET_MANIFEST)
    if target.get("status") != "PASS_N72R6_TARGET_SESSION_RECOVERY_32_OF_32_VALIDATED":
        raise RuntimeError(f"target stream manifest is not frozen PASS: {target.get('status')}")
    selected = {str(item["event_id"]): dict(item) for item in target.get("selected", [])}
    if len(selected) != 32:
        raise RuntimeError(f"expected 32 selected target events, found {len(selected)}")
    scenarios: dict[str, dict[str, Any]] = {}
    for event_id in sorted(selected):
        replay_manifest_path = N72R6_ROOT / event_id / "event_manifest.json"
        replay_manifest = read_json(replay_manifest_path)
        if replay_manifest.get("status") != "PASS_N72R6_C0_C1_EVENT_REPLAY":
            raise RuntimeError(f"frozen N72R6 event is not PASS: {event_id}")
        c0_path = _path(str(replay_manifest["c0"]["path"]))
        if sha256(c0_path) != str(replay_manifest["c0"]["sha256"]):
            raise RuntimeError(f"frozen C0 hash mismatch: {event_id}")
        c0_rows = read_jsonl(c0_path)
        expected_frames = list(range(int(replay_manifest["event_frame"]), int(replay_manifest["event_frame"]) + 101))
        if [int(row["frame"]) for row in c0_rows] != expected_frames:
            raise RuntimeError(f"frozen C0 frame axis mismatch: {event_id}")
        scenarios[event_id] = {
            "event_id": event_id,
            "sequence": str(replay_manifest["sequence"]),
            "event_frame": int(replay_manifest["event_frame"]),
            "target_public_id": int(replay_manifest["target_public_id"]),
            "action_type": str(selected[event_id].get("action_type")),
            "c0_rows": c0_rows,
            "c0_manifest": replay_manifest,
        }
    return scenarios


def _validate_d1_d2_batch(root: Path, variant: str, scenarios: Mapping[str, Any]) -> dict[str, list[dict[str, Any]]]:
    batch_path = root / "batch_manifest.json"
    batch = read_json(batch_path)
    accepted_batch_statuses = {
        "PASS_N72R7_INDEPENDENT_DEV_BATCH",
        "PASS_N72R7_LEARNED_DECODER_BATCH",
    }
    if batch.get("status") not in accepted_batch_statuses:
        raise RuntimeError(f"{variant} batch is not PASS: {batch.get('status')}")
    if int(batch.get("requested_event_count", -1)) != 32 or int(batch.get("completed_event_count", -1)) != 32 or int(batch.get("failed_event_count", -1)) != 0:
        raise RuntimeError(f"{variant} batch count is incomplete")
    if batch.get("independent_process_per_event") is not True or int(batch.get("max_concurrent_processes", -1)) != 1:
        raise RuntimeError(f"{variant} independent-process provenance is invalid")
    records = {(str(item["event_id"]), str(item["variant"])): item for item in batch.get("results", [])}
    if len(records) != 32 or set(event_id for event_id, _ in records) != set(scenarios):
        raise RuntimeError(f"{variant} batch result keys are not exactly the frozen 32 events")
    output: dict[str, list[dict[str, Any]]] = {}
    for event_id in sorted(scenarios):
        record = records[(event_id, variant)]
        manifest_path = _path(str(record["event_manifest"]))
        manifest = read_json(manifest_path)
        if manifest.get("status") != "PASS_N72R7_CLOSED_LOOP_EVENT_REPLAY" or manifest.get("variant") != variant:
            raise RuntimeError(f"{variant} event manifest is not PASS: {event_id}")
        frames_path = _path(str(manifest["frames"]))
        if sha256(frames_path) != str(manifest["frames_sha256"]):
            raise RuntimeError(f"{variant} frames hash mismatch: {event_id}")
        rows = read_jsonl(frames_path)
        scenario = scenarios[event_id]
        expected_frames = list(range(int(scenario["event_frame"]), int(scenario["event_frame"]) + 101))
        if len(rows) != 101 or [int(row["frame"]) for row in rows] != expected_frames:
            raise RuntimeError(f"{variant} frame axis mismatch: {event_id}")
        for row_index, row in enumerate(rows):
            location = f"{variant}/{event_id}/{row.get('frame')}"
            _assert_runtime_value_clean(row, location)
            for flag in ("runtime_future_gt_used", "runtime_gt_read", "posthoc_gt_used", "public_id_inference"):
                if row.get(flag) is not False:
                    raise RuntimeError(f"{location} causal flag {flag} is not false")
            if int(row.get("event_frame", -1)) != int(scenario["event_frame"]) or int(row.get("target_public_id", -1)) != int(scenario["target_public_id"]):
                raise RuntimeError(f"{location} frozen event/target authority mismatch")
            if row_index == 0:
                if row.get("candidate_rows") != [] or row.get("candidate_count") != 0:
                    raise RuntimeError(f"{location} event frame must not run future candidates")
                if row.get("memory_read") is not False or row.get("event_frame_memory_read") is not False:
                    raise RuntimeError(f"{location} event-frame memory boundary failed")
            else:
                if row.get("record_kind") != "future_association_frame" or row.get("memory_read") is not True:
                    raise RuntimeError(f"{location} future association/memory read marker failed")
                if int(row.get("frame_horizon", -1)) != row_index or int(row.get("first_memory_visible_frame", -1)) != int(scenario["event_frame"]) + 1:
                    raise RuntimeError(f"{location} future horizon/memory boundary failed")
                pool = row.get("candidate_pool")
                if not isinstance(pool, Mapping) or pool.get("public_id_inference") is not False or pool.get("runtime_future_gt_used") is not False:
                    raise RuntimeError(f"{location} candidate pool audit failed")
                pool_rows = pool.get("candidate_rows", [])
                candidate_rows = row.get("candidate_rows", [])
                if int(row.get("candidate_count", -1)) != len(candidate_rows) or int(pool.get("candidate_count", -1)) != len(pool_rows):
                    raise RuntimeError(f"{location} candidate count mismatch")
                pool_uids = [str(item["candidate_uid"]) for item in pool_rows]
                output_uids = [str(item["candidate_uid"]) for item in candidate_rows]
                if pool_uids != output_uids or len(output_uids) != len(set(output_uids)):
                    raise RuntimeError(f"{location} candidate UID/pool order mismatch")
                if any(item.get("public_id") is not None or item.get("public_id_authority") is not None for item in pool_rows):
                    raise RuntimeError(f"{location} source candidate owns public authority before solver")
                assigned = [item.get("public_id") for item in candidate_rows if item.get("public_id") is not None]
                if len(assigned) != len(set(assigned)):
                    raise RuntimeError(f"{location} duplicate output public assignment")
                score_audit = row.get("score_audit")
                if not isinstance(score_audit, Mapping):
                    raise RuntimeError(f"{location} score audit is missing")
                matrix = np.asarray(score_audit.get("fused_score_matrix", []), dtype=np.float64)
                state_axis = list(score_audit.get("association_state_axis", []))
                public_axis = list(score_audit.get("public_id_axis", []))
                if matrix.ndim != 2 or matrix.shape[0] != len(candidate_rows) or matrix.shape[1] != len(state_axis) or len(state_axis) != len(public_axis) or not np.all(np.isfinite(matrix)):
                    raise RuntimeError(f"{location} score/authority matrix audit failed")
                assignment = row.get("assignment")
                formal = assignment.get("solver") if isinstance(assignment, Mapping) else None
                if not isinstance(formal, Mapping) or formal.get("runtime_future_gt_used") is not False:
                    raise RuntimeError(f"{location} exact solver audit failed")
                if assignment.get("solver_public_id_immutable") is not True:
                    raise RuntimeError(f"{location} public ID immutability audit failed")
            output.setdefault(event_id, []).append(row)
        # D1/D2 manifests must agree with the frozen target authority and keep
        # source provenance explicit; action_type is joined only posthoc.
        if int(manifest.get("target_public_id", -1)) != int(scenario["target_public_id"]):
            raise RuntimeError(f"{variant} target public authority mismatch: {event_id}")
    return output


def validate_runtime(scenarios: Mapping[str, Any]) -> dict[str, Any]:
    # No dataset GT is loaded in this function.
    d1 = _validate_d1_d2_batch(D1_ROOT, "D1", scenarios)
    d2 = _validate_d1_d2_batch(D2_ROOT, "D2", scenarios)
    for event_id in sorted(scenarios):
        c0_rows = scenarios[event_id]["c0_rows"]
        d1_rows, d2_rows = d1[event_id], d2[event_id]
        for row_index in range(1, 101):
            c0_uids = [str(item["candidate_uid"]) for item in c0_rows[row_index].get("candidate_rows", [])]
            d1_uids = [str(item["candidate_uid"]) for item in d1_rows[row_index].get("candidate_rows", [])]
            d2_uids = [str(item["candidate_uid"]) for item in d2_rows[row_index].get("candidate_rows", [])]
            if d1_uids != c0_uids or d2_uids[: len(c0_uids)] != c0_uids:
                raise RuntimeError(f"B0 candidate pool was not preserved: {event_id}/{c0_rows[row_index]['frame']}")
            d2_sources = [str(item.get("candidate_source")) for item in d2_rows[row_index].get("candidate_rows", [])]
            if any(source != "MAIN_B0_CANDIDATE" for source in [str(item.get("candidate_source")) for item in d1_rows[row_index].get("candidate_rows", [])]):
                raise RuntimeError(f"D1 contains non-B0 source: {event_id}/{c0_rows[row_index]['frame']}")
            if d2_sources.count("TARGET_SESSION_CURRENT_RAW") > 1:
                raise RuntimeError(f"D2 target-session source duplicated: {event_id}/{c0_rows[row_index]['frame']}")
    audit = {
        "schema_version": "N72R7_RUNTIME_VALIDATION_V1",
        "status": "PASS_N72R7_RUNTIME_VALIDATION",
        "event_count": len(scenarios),
        "independent_sequence_count": len({item["sequence"] for item in scenarios.values()}),
        "checked_runtime_rows_per_variant": len(scenarios) * 101,
        "checked_future_rows_per_variant": len(scenarios) * 100,
        "d0_source": "frozen_N72R6_C0_MAIN_FROZEN_B0",
        "d1_source": str(D1_ROOT),
        "d2_source": str(D2_ROOT),
        "d1_d2_b0_candidate_stream_equal": True,
        "d2_target_session_candidate_source_only": True,
        "event_frame_memory_read": False,
        "first_memory_visible_frame": "event_frame+1",
        "exact_global_solver": True,
        "public_id_inference": False,
        "runtime_future_gt_used": False,
        "gt_loaded_in_worker": False,
        "posthoc_gt_not_loaded_during_validation": True,
        "interaction_source": "simulated_from_gt",
        "real_human_evidence": False,
        "created_at_utc": now_utc(),
    }
    atomic_json(RUNTIME_VALIDATION_PATH, audit)
    return {"audit": audit, "rows": {"D0": {key: value["c0_rows"] for key, value in scenarios.items()}, "D1": d1, "D2": d2}}


def _protected_map(event: Mapping[str, Any], gt_frames: Mapping[int, Mapping[int, Any]]) -> dict[int, int]:
    event_frame = int(event["event_frame"])
    target_gid = int(event["target_dataset_gt_id"])
    protected: dict[int, int] = {}
    event_row = event["rows"]["D0"][event_frame]
    for gt_id, gt_item in gt_frames.get(event_frame, {}).items():
        if int(gt_id) == target_gid:
            continue
        best_iou, best_candidate = _candidate_best_iou(event_row, gt_item["box"])
        if best_candidate is not None and best_iou >= IOU_THRESHOLD and best_candidate.get("public_id") is not None:
            protected[int(gt_id)] = int(best_candidate["public_id"])
    return protected


def _score_pair(
    event: Mapping[str, Any],
    baseline_name: str,
    treatment_name: str,
    horizon: int,
    gt_frames: Mapping[int, Mapping[int, Any]],
    protected: Mapping[int, int],
) -> dict[str, Any]:
    metric = _metric_template()
    metric["window_frame_count"] = int(horizon)
    details: list[dict[str, Any]] = []
    event_frame = int(event["event_frame"])
    target_public = int(event["target_public_id"])
    target_gid = int(event["target_dataset_gt_id"])
    baseline_rows = event["rows"][baseline_name]
    treatment_rows = event["rows"][treatment_name]
    previous_treatment_uid: str | None = None
    previous_treatment_error = False
    for offset in range(1, horizon + 1):
        frame = event_frame + offset
        baseline = baseline_rows[frame]
        treatment = treatment_rows[frame]
        gt_target = gt_frames.get(frame, {}).get(target_gid)
        treatment_uid, treatment_candidate = _target_binding(treatment, target_public)
        baseline_uid, _ = _target_binding(baseline, target_public)
        raw_switch = treatment.get("raw_binding_switch") is not None
        metric["raw_switch_count"] += int(raw_switch)
        if gt_target is None:
            metric["target_gt_absent_frames"] += 1
            if raw_switch:
                metric["posthoc_unassessable_switch_count"] += 1
            continue
        metric["evaluated_frames"] += 1
        metric["target_gt_visible_frames"] += 1
        gt_box = gt_target["box"]
        baseline_iou, _ = _public_box_for_gt(baseline, target_public, gt_box)
        treatment_iou, _ = _public_box_for_gt(treatment, target_public, gt_box)
        baseline_correct = baseline_iou >= IOU_THRESHOLD
        treatment_correct = treatment_iou >= IOU_THRESHOLD
        best_iou, best_candidate = _candidate_best_iou(treatment, gt_box)
        treatment_missing = treatment_uid is None
        wrong_reassociation = bool(
            treatment_candidate is not None
            and any(
                int(other_id) != target_gid and _box_iou(treatment_candidate.get("box_xyxy"), other_item["box"]) >= IOU_THRESHOLD
                for other_id, other_item in gt_frames.get(frame, {}).items()
            )
        )
        target_assignment_changed = baseline_uid != treatment_uid
        baseline_map = _public_map(baseline)
        treatment_map = _public_map(treatment)
        common_uids = set(baseline_map) & set(treatment_map)
        global_common_changed = sum(int(baseline_map[uid] != treatment_map[uid]) for uid in common_uids)
        record = metric_record(
            baseline_iou=baseline_iou,
            treatment_iou=treatment_iou,
            baseline_correct=baseline_correct,
            treatment_correct=treatment_correct,
            assignment_changed=target_assignment_changed,
        )
        metric["baseline_iou_sum"] += baseline_iou
        metric["treatment_iou_sum"] += treatment_iou
        metric["delta_iou_sum"] += float(record["delta_iou"])
        metric["baseline_correct_frames"] += int(baseline_correct)
        metric["treatment_correct_frames"] += int(treatment_correct)
        metric["baseline_identity_error_frames"] += int(not baseline_correct)
        metric["treatment_identity_error_frames"] += int(not treatment_correct)
        metric["identity_error_reduction_sum"] += float(record["identity_error_reduction"])
        metric["target_missing_frames"] += int(treatment_missing)
        metric["wrong_reassociation_frames"] += int(wrong_reassociation)
        metric["candidate_present_frames"] += int(best_iou >= IOU_THRESHOLD)
        metric["assignment_change_count"] += int(target_assignment_changed)
        metric["target_assignment_change_count"] += int(target_assignment_changed)
        metric["global_common_assignment_change_count"] += int(global_common_changed > 0)
        change_type = record["assignment_change_type"]
        metric["true_correct_crossing_count"] += int(change_type == AssignmentChangeType.TRUE_CORRECT_CROSSING.value)
        metric["true_incorrect_crossing_count"] += int(change_type == AssignmentChangeType.TRUE_INCORRECT_CROSSING.value)
        metric["directional_improvement_count"] += int(change_type == AssignmentChangeType.DIRECTIONAL_IMPROVEMENT.value)
        metric["directional_regression_count"] += int(change_type == AssignmentChangeType.DIRECTIONAL_REGRESSION.value)
        metric["neutral_change_count"] += int(change_type == AssignmentChangeType.NEUTRAL_CHANGE.value)
        treatment_error = not treatment_correct
        recorrection = bool(treatment_error and not previous_treatment_error)
        id_switch = bool(previous_treatment_uid is not None and treatment_uid is not None and treatment_uid != previous_treatment_uid)
        metric["id_switch_count"] += int(id_switch)
        metric["recorrection_opportunity_count"] += int(recorrection)
        if raw_switch:
            if treatment_correct:
                metric["posthoc_correct_switch_count"] += 1
            else:
                metric["posthoc_wrong_switch_count"] += 1
        for protected_gid, protected_pid in protected.items():
            gt_other = gt_frames.get(frame, {}).get(int(protected_gid))
            if gt_other is None:
                continue
            baseline_protected_iou, _ = _public_box_for_gt(baseline, protected_pid, gt_other["box"])
            treatment_protected_iou, _ = _public_box_for_gt(treatment, protected_pid, gt_other["box"])
            baseline_protected_correct = baseline_protected_iou >= IOU_THRESHOLD
            treatment_protected_correct = treatment_protected_iou >= IOU_THRESHOLD
            metric["protected_compared"] += 1
            metric["protected_regression_count"] += int(baseline_protected_correct and not treatment_protected_correct)
            metric["protected_improvement_count"] += int(treatment_protected_correct and not baseline_protected_correct)
        details.append(
            {
                "frame": frame,
                "baseline_target_candidate_uid": baseline_uid,
                "treatment_target_candidate_uid": treatment_uid,
                "target_assignment_changed": target_assignment_changed,
                "global_common_assignment_changed_count": global_common_changed,
                "baseline_target_iou": float(baseline_iou),
                "treatment_target_iou": float(treatment_iou),
                "baseline_correct": baseline_correct,
                "treatment_correct": treatment_correct,
                "target_missing": treatment_missing,
                "candidate_recall_present": bool(best_iou >= IOU_THRESHOLD),
                "wrong_reassociation": wrong_reassociation,
                "id_switch": id_switch,
                "recorrection_opportunity": recorrection,
                "raw_binding_switch": raw_switch,
                "assignment_change_type": change_type,
                "true_correct_crossing": bool(record["true_correct_crossing"]),
                "true_incorrect_crossing": bool(record["true_incorrect_crossing"]),
                "runtime_future_gt_used": False,
            }
        )
        previous_treatment_uid = treatment_uid or previous_treatment_uid
        previous_treatment_error = treatment_error
    metric = _finalize_metric(metric)
    metric["frame_details"] = details
    return metric


def _aggregate(
    event_results: Sequence[Mapping[str, Any]],
    comparison: str,
    horizon: int,
    *,
    action: str | None = None,
) -> dict[str, Any]:
    metric = _metric_template()
    values_by_sequence: dict[str, list[float]] = defaultdict(list)
    selected = [item for item in event_results if action is None or item["action_type"] == action]
    for event in selected:
        source = event["comparisons"][comparison][str(horizon)]
        _merge_metric(metric, source)
        values_by_sequence[str(event["sequence"])].append(float(source["identity_error_reduction"] or 0.0))
    metric = _finalize_metric(metric)
    seed_offset = list(COMPARISONS).index(comparison) * 100 + HORIZONS.index(horizon)
    metric["sequence_cluster_bootstrap_95ci"] = sequence_cluster_bootstrap(
        values_by_sequence,
        seed=BOOTSTRAP_SEED + seed_offset,
        repetitions=BOOTSTRAP_REPETITIONS,
    )
    metric["event_count"] = len(selected)
    metric["independent_sequence_count"] = len({str(item["sequence"]) for item in selected})
    metric["comparison"] = comparison
    metric["horizon"] = int(horizon)
    return metric


def posthoc_score(runtime: Mapping[str, Any]) -> dict[str, Any]:
    # This is the first function that opens the train GT files.
    policy = read_json(EVENT_POLICY)
    policy_events = {str(item["event_id"]): item for item in policy.get("events", [])}
    scenarios = _load_runtime_scenarios()
    rows = runtime["rows"]
    gt_by_sequence: dict[str, Any] = {}
    event_results: list[dict[str, Any]] = []
    for event_id in sorted(scenarios):
        scenario = dict(scenarios[event_id])
        policy_event = policy_events.get(event_id)
        if policy_event is None:
            raise RuntimeError(f"selected event missing from frozen policy: {event_id}")
        scenario["target_dataset_gt_id"] = int(policy_event["dataset_gt_id"])
        scenario["action_type"] = str(policy_event["action_type"])
        scenario["rows"] = {
            variant: {int(row["frame"]): row for row in rows[variant][event_id]}
            for variant in EVENT_VARIANTS
        }
        sequence = str(scenario["sequence"])
        if sequence not in gt_by_sequence:
            gt_by_sequence[sequence] = _load_gt(sequence)
        protected = _protected_map(scenario, gt_by_sequence[sequence])
        event_result: dict[str, Any] = {
            "event_id": event_id,
            "sequence": sequence,
            "action_type": scenario["action_type"],
            "event_frame": int(scenario["event_frame"]),
            "target_public_id": int(scenario["target_public_id"]),
            "target_dataset_gt_id": int(scenario["target_dataset_gt_id"]),
            "interaction_source": "simulated_from_gt",
            "not_real_human_evidence": True,
            "protected_public_by_gt_posthoc": protected,
            "comparisons": {},
            "runtime_future_gt_used": False,
            "gt_usage": "posthoc_only_after_runtime_validation_and_artifact_seal",
        }
        for comparison, (baseline_name, treatment_name) in COMPARISONS.items():
            event_result["comparisons"][comparison] = {}
            for horizon in HORIZONS:
                event_result["comparisons"][comparison][str(horizon)] = _score_pair(
                    scenario,
                    baseline_name,
                    treatment_name,
                    horizon,
                    gt_by_sequence[sequence],
                    protected,
                )
        event_results.append(event_result)
    aggregate: dict[str, dict[str, dict[str, Any]]] = {comparison: {} for comparison in COMPARISONS}
    action_aggregate: dict[str, dict[str, dict[str, dict[str, Any]]]] = defaultdict(lambda: defaultdict(dict))
    for comparison in COMPARISONS:
        for horizon in HORIZONS:
            aggregate[comparison][str(horizon)] = _aggregate(event_results, comparison, horizon)
            for action in sorted({str(item["action_type"]) for item in event_results}):
                action_aggregate[action][comparison][str(horizon)] = _aggregate(
                    event_results, comparison, horizon, action=action
                )
    gate_by_comparison: dict[str, Any] = {}
    for comparison in ("D1_vs_D0", "D2_vs_D0"):
        metric = aggregate[comparison]["20"]
        ci = metric["sequence_cluster_bootstrap_95ci"]
        gate_by_comparison[comparison] = {
            "primary_horizon": 20,
            "identity_error_reduction_mean": metric["identity_error_reduction"],
            "identity_error_reduction_ci_lower": ci["lower"],
            "ci_lower_strictly_positive": bool(ci["lower"] is not None and ci["lower"] > 0.0),
            "true_correct_crossings": int(metric["true_correct_crossing_count"]),
            "true_incorrect_crossings": int(metric["true_incorrect_crossing_count"]),
            "correct_crossings_exceed_incorrect": int(metric["true_correct_crossing_count"]) > int(metric["true_incorrect_crossing_count"]),
            "protected_regression_count": int(metric["protected_regression_count"]),
            "protected_regression_controlled": int(metric["protected_regression_count"]) == 0,
            "runtime_future_gt_used": False,
        }
    strict_any = any(
        item["ci_lower_strictly_positive"]
        and item["correct_crossings_exceed_incorrect"]
        and item["protected_regression_controlled"]
        for item in gate_by_comparison.values()
    )
    result = {
        "schema_version": "N72R7_POSTHOC_RESULTS_V1",
        "status": "PASS_DEVELOPMENT_FUTURE_EFFECT" if strict_any else "PASS_EXECUTION_FAIL_FUTURE_EFFECT",
        "created_at_utc": now_utc(),
        "event_count": len(event_results),
        "independent_sequence_count": len({item["sequence"] for item in event_results}),
        "variants": list(EVENT_VARIANTS),
        "horizons": list(HORIZONS),
        "aggregate": aggregate,
        "action_aggregate": action_aggregate,
        "event_metrics": event_results,
        "gate": {
            "research_gate": "PASS_GT_SIMULATED_CLOSED_LOOP_REACQUISITION_CONFIRMED" if strict_any else "FAIL_FUTURE_EFFECT",
            "by_comparison": gate_by_comparison,
            "strict_primary_horizon": 20,
            "strict_ci_lower_requirement": "> 0",
            "true_crossing_taxonomy": "metric_record target binding only",
            "protected_regression_checked": True,
            "candidate_pool_completeness_checked": True,
            "runtime_future_gt_used": False,
            "interaction_source": "simulated_from_gt",
            "real_human_evidence": False,
            "production_authorized": False,
        },
        "bootstrap_protocol": {
            "repetitions": BOOTSTRAP_REPETITIONS,
            "seed": BOOTSTRAP_SEED,
            "cluster_unit": "independent_sequence",
            "within_cluster_aggregation": "mean_event_identity_error_reduction",
            "multiple_events_within_sequence_preserved": True,
        },
        "runtime_validation": read_json(RUNTIME_VALIDATION_PATH),
        "gt_usage": "posthoc_only_after_runtime_validation_and_artifact_seal",
        "candidate_pool_oracle_input": str(N72R7_ROOT / "forensic/union_pool_oracle.json"),
        "scientific_result": "DEVELOPMENT_ONLY_NO_INDEPENDENT_CONFIRMATION",
        "production_authorized": False,
    }
    atomic_json(RESULT_PATH, result)
    atomic_jsonl(EVENT_METRICS_PATH, event_results)
    return result


def write_status(result: Mapping[str, Any], runtime: Mapping[str, Any]) -> None:
    gate = result["gate"]
    d1 = result["aggregate"]["D1_vs_D0"]["20"]
    d2 = result["aggregate"]["D2_vs_D0"]["20"]
    controller = {
        "schema_version": "N72R7_CONTROLLER_STATUS_V1",
        "current_round": "N72R7_NONLEARNING_CLOSED_LOOP_REPLAY",
        "current_stage": "05_NONLEARNING_D1_D2_POSTHOC_SCORING",
        "candidate_pool_oracle_recall": {"b0": 0.8578274760383386, "union": 0.8936102236421726},
        "fixed_raw_recall": 0.0,
        "B0_pool_recall": 0.8578274760383386,
        "union_pool_recall": 0.8936102236421726,
        "selector_validation_accuracy": None,
        "NONE_accuracy": None,
        "raw_switch_count": {
            "D1": int(d1.get("raw_switch_count", 0)),
            "D2": int(d2.get("raw_switch_count", 0)),
        },
        "successful_reacquisition_count": {
            "D1": int(d1.get("posthoc_correct_switch_count", 0)),
            "D2": int(d2.get("posthoc_correct_switch_count", 0)),
        },
        "trusted_memory_updates": None,
        "distractor_memory_updates": None,
        "H20": {"D1_vs_D0": d1, "D2_vs_D0": d2},
        "H50": {comparison: result["aggregate"][comparison]["50"] for comparison in COMPARISONS},
        "H100": {comparison: result["aggregate"][comparison]["100"] for comparison in COMPARISONS},
        "true_correct_crossing": {comparison: result["aggregate"][comparison]["20"]["true_correct_crossing_count"] for comparison in COMPARISONS},
        "true_incorrect_crossing": {comparison: result["aggregate"][comparison]["20"]["true_incorrect_crossing_count"] for comparison in COMPARISONS},
        "protected_regression": {comparison: result["aggregate"][comparison]["20"]["protected_regression_count"] for comparison in COMPARISONS},
        "runtime_future_gt_used": False,
        "gpu_allocation": {"mode": "CPU_ONLY_FROZEN_ARTIFACT_REPLAY", "max_concurrent_gpus": 0},
        "next_root_cause": "pending posthoc interpretation",
        "research_gate": gate["research_gate"],
        "runtime_validation": runtime["audit"],
        "interaction_source": "simulated_from_gt",
        "real_human_evidence": False,
        "created_at_utc": now_utc(),
    }
    atomic_json(CONTROLLER_PATH, controller)
    d1_ci = result["aggregate"]["D1_vs_D0"]["20"]["sequence_cluster_bootstrap_95ci"]
    d2_ci = result["aggregate"]["D2_vs_D0"]["20"]["sequence_cluster_bootstrap_95ci"]
    human = "\n".join(
        [
            "# N72R7 Controller Status",
            "",
            f"- Stage: 05 non-learning D1/D2 posthoc scoring",
            f"- Runtime validation: {runtime['audit']['status']}",
            f"- Events/sequences: {result['event_count']}/{result['independent_sequence_count']}",
            f"- D1−D0 H20 identity-error reduction: {d1.get('identity_error_reduction')} CI [{d1_ci.get('lower')}, {d1_ci.get('upper')}]",
            f"- D2−D0 H20 identity-error reduction: {d2.get('identity_error_reduction')} CI [{d2_ci.get('lower')}, {d2_ci.get('upper')}]",
            f"- Research gate: {gate['research_gate']}",
            "- Evidence source: simulated_from_gt; not a real-human study.",
            "- Production/calibration/LoRA authorization: false.",
        ]
    ) + "\n"
    atomic_write(HUMAN_STATUS_PATH, human)
    atomic_json(
        STAGE_PATH,
        {
            "schema_version": "N72R7_STAGE_STATUS_V1",
            "stage": "05_NONLEARNING_D1_D2_CAUSAL_REPLAY_AND_POSTHOC_SCORING",
            "status": "PASS_STAGE05_EXECUTION",
            "created_at_utc": now_utc(),
            "runtime_validation": str(RUNTIME_VALIDATION_PATH),
            "result_artifact": str(RESULT_PATH),
            "event_metrics": str(EVENT_METRICS_PATH),
            "event_count": result["event_count"],
            "independent_sequence_count": result["independent_sequence_count"],
            "research_gate": gate["research_gate"],
            "production_authorized": False,
            "training_authorized": gate["research_gate"] != "PASS_GT_SIMULATED_CLOSED_LOOP_REACQUISITION_CONFIRMED",
            "runtime_future_gt_used": False,
            "posthoc_gt_used": True,
            "interaction_source": "simulated_from_gt",
            "real_human_evidence": False,
            "historical_evidence_preserved": True,
        },
    )


def write_failure(exc: BaseException) -> Path:
    FAILURE_ROOT.mkdir(parents=True, exist_ok=True)
    existing = sorted(FAILURE_ROOT.glob("n72r7_posthoc_score_failure_attempt*.json"))
    path = FAILURE_ROOT / f"n72r7_posthoc_score_failure_attempt{len(existing) + 1}.json"
    atomic_json(
        path,
        {
            "schema_version": "N72R7_FAILURE_RECORD_V1",
            "status": "FAIL_PRESERVED",
            "stage": "05_NONLEARNING_D1_D2_CAUSAL_REPLAY_AND_POSTHOC_SCORING",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
            "runtime_future_gt_used": False,
            "posthoc_gt_used": False,
            "created_at_utc": now_utc(),
        },
    )
    return path


def main() -> int:
    try:
        scenarios = _load_runtime_scenarios()
        runtime = validate_runtime(scenarios)
        result = posthoc_score(runtime)
        write_status(result, runtime)
        print(
            json.dumps(
                {
                    "status": result["status"],
                    "research_gate": result["gate"]["research_gate"],
                    "runtime_validation": str(RUNTIME_VALIDATION_PATH),
                    "result": str(RESULT_PATH),
                },
                sort_keys=True,
            )
        )
        return 0
    except Exception as exc:
        failure = write_failure(exc)
        print(json.dumps({"status": "FAIL_N72R7_POSTHOC_SCORE", "failure": str(failure)}, sort_keys=True))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
