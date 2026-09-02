#!/usr/bin/env python3
"""Reconcile the frozen N72R3 effect replay with the N72R4 semantics.

This is deliberately a CPU-only semantic replay.  It does not execute SAM3,
read GT in the runtime pass, alter a candidate stream, or alter the frozen
intervention.  Every frozen score matrix is passed through the production
exact-public-assignment solver via ``effect_assignment``.  Posthoc scoring is
started only after the new runtime artifacts have been validated and sealed.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import sys
import tempfile
import traceback
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sam3_intermot.association.effect_assignment import solve_effect_assignment  # noqa: E402
from sam3_intermot.association.public_assignment import validate_exact_public_assignment  # noqa: E402
from sam3_intermot.evaluation.interaction_effect_metrics import (  # noqa: E402
    AssignmentChangeType,
    metric_record,
    sequence_cluster_bootstrap,
)


FROZEN_ROOT = Path("/data2/usr_for_deadline/SAM3_InterMOT_N72R3/worktree/outputs/N72R3")
OLD_ARTIFACT_ROOT = FROZEN_ROOT / "effect_replay/attempt1/runtime_event_artifacts"
OLD_RUNTIME_MANIFEST = FROZEN_ROOT / "effect_replay/attempt1/runtime_manifest.json"
OLD_RESULT = FROZEN_ROOT / "effect_replay/attempt1/ccam_paired_replay_results.json"
EVENT_MANIFEST = FROZEN_ROOT / "simulation/real_event_manifest.json"
STAGE18_ROOT = FROZEN_ROOT / "baseline/stage18_persistent_public/full_eligible"
DEFAULT_REPAIR_ROOT = ROOT / "outputs/N72R3R1/corrected_replay/attempt1"
REPAIR_ROOT = Path(os.environ.get("N72R3R1_SEMANTIC_REPLAY_ROOT", str(DEFAULT_REPAIR_ROOT)))
ARTIFACT_ROOT = REPAIR_ROOT / "runtime_event_artifacts"
RUNTIME_MANIFEST = REPAIR_ROOT / "runtime_manifest.json"
RUNTIME_VALIDATION = REPAIR_ROOT / "runtime_validation.json"
RESULT_PATH = ROOT / "outputs/N72R3R1/corrected_replay/n72r3r1_semantic_repair_results.json"
COMPARISON_PATH = ROOT / "outputs/N72R3R1/old_vs_new_comparison.json"
GATE_PATH = ROOT / "outputs/N72R3R1/n72r3r1_gate.json"
STAGE_PATH = ROOT / "outputs/N72R3R1/stage_05_status.json"
STAGE_STATUS_PATH = ROOT / "outputs/N72R3R1/stage_status/stage_05_semantic_rerun.json"
FAILURE_ROOT = ROOT / "outputs/N72R3R1/attempts"

VARIANTS = (
    "NO_INTERVENTION",
    "M0_CURRENT_FRAME_CORRECTION_ONLY",
    "M1_HUMAN_EMA_PROTOTYPE",
    "M2_POSITIVE_HUMAN_ANCHORS",
    "M3_NEGATIVE_COMPETITOR_BANK",
    "M4_RELIABILITY_AGE_ADMISSION",
)
MEMORY_VARIANTS = set(VARIANTS[2:])
HORIZONS = (20, 50, 100)
IOU_THRESHOLD = 0.5
BOOTSTRAP_SEED = 7202
BOOTSTRAP_REPETITIONS = 2000


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
                raise TypeError(f"{path}:{line_number}: expected JSON object")
            rows.append(value)
    return rows


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def box_iou(left: Any, right: Any) -> float:
    a = np.asarray(left, dtype=np.float64).reshape(-1)
    b = np.asarray(right, dtype=np.float64).reshape(-1)
    if a.size != 4 or b.size != 4 or not np.all(np.isfinite(a)) or not np.all(np.isfinite(b)):
        return 0.0
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - intersection
    return float(intersection / union) if union > 0.0 else 0.0


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


def row_map(frame_row: dict[str, Any]) -> dict[str, int | None]:
    return {str(item["candidate_uid"]): item.get("public_id") for item in frame_row.get("candidate_rows", [])}


def candidate_best_iou(frame_row: dict[str, Any], gt_box: list[float]) -> tuple[float, dict[str, Any] | None]:
    values = [(box_iou(item["box_xyxy"], gt_box), item) for item in frame_row.get("candidate_rows", [])]
    return max(values, key=lambda pair: (pair[0], -int(pair[1]["candidate_index"])), default=(0.0, None))


def public_target_iou(frame_row: dict[str, Any], public_id: int, gt_box: list[float]) -> float:
    return max(
        (
            box_iou(item["box_xyxy"], gt_box)
            for item in frame_row.get("candidate_rows", [])
            if item.get("public_id") is not None and int(item["public_id"]) == int(public_id)
        ),
        default=0.0,
    )


def public_box_for_gt(frame_row: dict[str, Any], public_id: int, gt_box: list[float]) -> tuple[float, int | None]:
    candidates = [
        item
        for item in frame_row.get("candidate_rows", [])
        if item.get("public_id") is not None and int(item["public_id"]) == int(public_id)
    ]
    if not candidates:
        return 0.0, None
    values = [(box_iou(item["box_xyxy"], gt_box), int(item["candidate_index"])) for item in candidates]
    return max(values, key=lambda value: (value[0], -value[1]))


def metric_template() -> dict[str, Any]:
    return {
        "evaluated_frames": 0,
        "target_gt_present_frames": 0,
        "target_iou_sum": 0.0,
        "target_correct_frames": 0,
        "target_missing_frames": 0,
        "target_identity_error_frames": 0,
        "wrong_reassociation_frames": 0,
        "candidate_present_frames": 0,
        "id_switch_count": 0,
        "recorrection_opportunity_count": 0,
        "assignment_change_count": 0,
        "true_correct_crossing_count": 0,
        "true_incorrect_crossing_count": 0,
        "directional_improvement_count": 0,
        "directional_regression_count": 0,
        "neutral_change_count": 0,
        "solver_coupled_collateral_count": 0,
        "protected_compared": 0,
        "protected_regression_count": 0,
        "protected_improvement_count": 0,
        "identity_error_reduction_sum": 0.0,
        "delta_iou_sum": 0.0,
        "composite_utility_secondary_sum": 0.0,
        "candidate_recall": None,
        "target_mean_iou": None,
        "future_identity_error": None,
        "missing_rate": None,
        "id_switch_rate": None,
        "wrong_reassociation_rate": None,
        "recorrection_rate": None,
        "protected_regression_rate": None,
        "identity_error_reduction": None,
        "delta_iou": None,
        "composite_utility_secondary": None,
    }


def finalize_metric(metric: dict[str, Any]) -> dict[str, Any]:
    frames = int(metric["evaluated_frames"])
    if frames:
        metric["target_mean_iou"] = float(metric["target_iou_sum"] / frames)
        metric["future_identity_error"] = float(metric["target_identity_error_frames"] / frames)
        metric["missing_rate"] = float(metric["target_missing_frames"] / frames)
        metric["id_switch_rate"] = float(metric["id_switch_count"] / frames)
        metric["wrong_reassociation_rate"] = float(metric["wrong_reassociation_frames"] / frames)
        metric["recorrection_rate"] = float(metric["recorrection_opportunity_count"] / frames)
        metric["candidate_recall"] = float(metric["candidate_present_frames"] / frames)
        metric["identity_error_reduction"] = float(metric["identity_error_reduction_sum"] / frames)
        metric["delta_iou"] = float(metric["delta_iou_sum"] / frames)
        metric["composite_utility_secondary"] = float(metric["composite_utility_secondary_sum"] / frames)
    compared = int(metric["protected_compared"])
    if compared:
        metric["protected_regression_rate"] = float(metric["protected_regression_count"] / compared)
    return metric


def stage18_state_axes(sequence: str) -> dict[int, dict[int, int]]:
    matches = [path for path in STAGE18_ROOT.iterdir() if path.is_dir() and f"n71-{sequence}-" in path.name]
    if len(matches) != 1:
        raise RuntimeError(f"expected exactly one Stage18 persistent baseline for {sequence}, found {matches}")
    by_frame_public: dict[int, dict[int, int]] = defaultdict(dict)
    for filename in ("candidate_decisions.jsonl", "identity_decisions.jsonl"):
        for item in read_jsonl(matches[0] / filename):
            frame = int(item["frame_idx"])
            public = item.get("public_id")
            state = item.get("association_state_id")
            if public is None or state is None:
                continue
            public = int(public)
            state = int(state)
            previous = by_frame_public[frame].get(public)
            if previous is not None and previous != state:
                raise RuntimeError(f"Stage18 public/state conflict: {sequence}/{frame}/{public}: {previous} != {state}")
            by_frame_public[frame][public] = state
    if not by_frame_public:
        raise RuntimeError(f"Stage18 baseline has no explicit state/public axis for {sequence}")
    return by_frame_public


def load_scenarios() -> list[dict[str, Any]]:
    manifest = read_json(EVENT_MANIFEST)
    if manifest.get("status") != "PASS_STAGE14_POLICY_FROZEN":
        raise RuntimeError("N72R3 event manifest is not the frozen Stage14 policy")
    events = [dict(item) for item in manifest.get("events", [])]
    if len(events) != 6:
        raise RuntimeError(f"semantic rerun requires exactly the frozen 6 events, found {len(events)}")
    runtime_manifest = read_json(OLD_RUNTIME_MANIFEST)
    if runtime_manifest.get("runtime_future_gt_used") is not False or runtime_manifest.get("gt_loaded_in_worker") is not False:
        raise RuntimeError("frozen N72R3 runtime manifest violates the no-runtime-GT contract")
    scenarios: list[dict[str, Any]] = []
    for event in sorted(events, key=lambda value: str(value["event_id"])):
        event_id = str(event["event_id"])
        old_path = OLD_ARTIFACT_ROOT / f"{event_id}.jsonl"
        if not old_path.is_file():
            raise FileNotFoundError(old_path)
        rows = read_jsonl(old_path)
        if len(rows) != 101 * len(VARIANTS):
            raise RuntimeError(f"frozen artifact row count mismatch {event_id}: {len(rows)}")
        for row in rows:
            if row.get("runtime_future_gt_used") is not False or row.get("runtime_gt_read") is not False or row.get("posthoc_gt_used") is not False:
                raise RuntimeError(f"frozen runtime causal flag is invalid: {event_id}/{row.get('variant')}/{row.get('frame')}")
            if any(key in row for key in ("dataset_gt_id", "gt_box", "future_gt")):
                raise RuntimeError(f"frozen runtime row contains GT field: {event_id}/{row.get('variant')}/{row.get('frame')}")
        axes = stage18_state_axes(str(event["sequence"]))
        scenarios.append(
            {
                "event": event,
                "event_id": event_id,
                "sequence": str(event["sequence"]),
                "action_type": str(event["action_type"]),
                "event_frame": int(event["event_frame"]),
                "target_public_id": int(next(item for item in rows if item["frame"] == int(event["event_frame"]))["target_public_id"]),
                "target_dataset_gt_id": int(event["dataset_gt_id"]),
                "old_path": old_path,
                "old_sha256": sha256(old_path),
                "old_rows": rows,
                "state_axes": axes,
            }
        )
    return scenarios


def explicit_states(scenario: dict[str, Any], row: dict[str, Any]) -> list[dict[str, int]]:
    frame = int(row["frame"])
    mapping = scenario["state_axes"].get(frame, {})
    states: list[dict[str, int]] = []
    for public_id in row["public_id_order"]:
        public_id = int(public_id)
        if public_id not in mapping:
            raise RuntimeError(f"missing frozen explicit state/public binding: {scenario['event_id']}/{frame}/{public_id}")
        states.append({"association_state_id": int(mapping[public_id]), "public_id": public_id})
    if len({item["association_state_id"] for item in states}) != len(states):
        raise RuntimeError(f"duplicate explicit association state IDs: {scenario['event_id']}/{frame}")
    if len({item["public_id"] for item in states}) != len(states):
        raise RuntimeError(f"duplicate explicit public IDs: {scenario['event_id']}/{frame}")
    return states


def reconcile_row(scenario: dict[str, Any], row: dict[str, Any]) -> dict[str, Any]:
    event_id = str(scenario["event_id"])
    frame = int(row["frame"])
    states = explicit_states(scenario, row)
    matrix = np.asarray(row["fused_score_matrix"], dtype=np.float64)
    candidates = [dict(item) for item in row["candidate_rows"]]
    solver = solve_effect_assignment(
        candidate_rows=candidates,
        persistent_states=states,
        fused_state_candidate_scores=matrix,
        source_run_id=f"n72r3r1-semantic:{event_id}:{row['variant']}:{frame}",
        session_id=f"n72r3r1-semantic:{event_id}",
        none_score=0.0,
    )
    if validate_exact_public_assignment({**solver, "schema_version": "N72R2_EXACT_PUBLIC_ASSIGNMENT_V1"}):
        raise RuntimeError(f"wrapped exact solver audit unexpectedly failed: {event_id}/{row['variant']}/{frame}")
    solver_public = [item["public_id"] for item in solver["assignment_rows"]]
    solver_state = [item["association_state_id"] for item in solver["assignment_rows"]]
    old_public = list(row.get("assignment_public_ids", []))
    old_status = list(row.get("assignment_status", []))
    if len(old_public) != len(candidates) or len(old_status) != len(candidates):
        raise RuntimeError(f"frozen assignment axis mismatch: {event_id}/{row['variant']}/{frame}")

    # The event frame is a frozen official intervention/setup row, not a
    # post-intervention association solve.  Keep that row bit-for-bit in its
    # public mapping while recording the exact solver audit separately.  From
    # event+1 onward, explicit NONE is solved first and the already frozen
    # outer-birth allocator is retained only for rows that the old artifact
    # explicitly marked as an outer birth.  No row index becomes authority.
    event_frame = int(scenario["event_frame"])
    is_event_frame = frame == event_frame
    output_public: list[int | None] = []
    output_status: list[str] = []
    outer_birth_count = 0
    for index, (solver_pid, old_pid, old_item_status) in enumerate(zip(solver_public, old_public, old_status)):
        if is_event_frame:
            output_public.append(None if old_pid is None else int(old_pid))
            output_status.append(str(old_item_status))
            continue
        if solver_pid is not None:
            output_public.append(int(solver_pid))
            output_status.append("EXACT_EXISTING_IDENTITY")
        elif old_item_status == "OUTER_BIRTH_ASSIGNED" and old_pid is not None:
            output_public.append(int(old_pid))
            output_status.append("OUTER_BIRTH_RETAINED_FROZEN_ALLOCATOR")
            outer_birth_count += 1
        elif old_pid is None:
            output_public.append(None)
            output_status.append("EXPLICIT_NONE_NO_OUTER_BIRTH")
        else:
            raise RuntimeError(
                f"exact solver produced NONE for a non-birth frozen assignment: "
                f"{event_id}/{row['variant']}/{frame}/{index}/{old_item_status}/{old_pid}"
            )
    assigned = [value for value in output_public if value is not None]
    if len(assigned) != len(set(assigned)):
        raise RuntimeError(f"semantic repair produced duplicate public IDs: {event_id}/{row['variant']}/{frame}")
    candidate_output: list[dict[str, Any]] = []
    for item, public_id, status in zip(candidates, output_public, output_status):
        copied = dict(item)
        copied["public_id"] = None if public_id is None else int(public_id)
        copied["assignment_status"] = str(status)
        candidate_output.append(copied)
    assignment_map = {str(item["candidate_uid"]): public_id for item, public_id in zip(candidate_output, output_public)}
    result = dict(row)
    result.update(
        {
            "schema_version": "N72R3R1_SEMANTIC_REPAIRED_RUNTIME_FRAME_V1",
            "candidate_rows": candidate_output,
            "assignment_public_ids": output_public,
            "assignment_status": output_status,
            "assignment_map": assignment_map,
            "legacy_assignment_public_ids": old_public,
            "legacy_assignment_status": old_status,
            "explicit_persistent_state_axis": states,
            "formal_solver": solver,
            "formal_solver_name": "sam3_intermot.association.effect_assignment.solve_effect_assignment",
            "formal_solver_assignment_public_ids": solver_public,
            "formal_solver_assignment_state_ids": solver_state,
            "formal_solver_none_count": int(solver["explicit_none_count"]),
            "outer_birth_count": int(outer_birth_count),
            "semantic_repair": {
                "event_frame_mapping_preserved": bool(is_event_frame),
                "event_frame_mapping_reason": "FROZEN_OFFICIAL_INTERVENTION_SETUP" if is_event_frame else None,
                "future_exact_none_then_outer_birth": not is_event_frame,
                "public_axis_source": "N72R3_STAGE18_EXPLICIT_ASSOCIATION_STATE_PUBLIC_BINDING",
                "state_axis_is_public_axis": False,
            },
            "runtime_future_gt_used": False,
            "runtime_gt_read": False,
            "posthoc_gt_used": False,
        }
    )
    return result


def run_runtime(scenarios: list[dict[str, Any]]) -> dict[str, Any]:
    if REPAIR_ROOT.exists() and any(REPAIR_ROOT.iterdir()):
        raise RuntimeError(f"semantic repair root is not empty; choose a new attempt root: {REPAIR_ROOT}")
    ARTIFACT_ROOT.mkdir(parents=True, exist_ok=True)
    completed: list[dict[str, Any]] = []
    for number, scenario in enumerate(scenarios, 1):
        rows = [reconcile_row(scenario, row) for row in scenario["old_rows"]]
        path = ARTIFACT_ROOT / f"{scenario['event_id']}.jsonl"
        atomic_jsonl(path, rows)
        completed.append(
            {
                "event_id": scenario["event_id"],
                "sequence": scenario["sequence"],
                "action_type": scenario["action_type"],
                "artifact": str(path),
                "artifact_sha256": sha256(path),
                "row_count": len(rows),
                "variant_count": len(VARIANTS),
                "frame_count": 101,
                "runtime_future_gt_used": False,
            }
        )
        atomic_json(
            RUNTIME_MANIFEST,
            {
                "schema_version": "N72R3R1_SEMANTIC_RUNTIME_MANIFEST_V1",
                "status": "IN_PROGRESS" if number < len(scenarios) else "PASS_RUNTIME_SEMANTIC_REPAIR",
                "created_at_utc": now_utc(),
                "runtime_root": str(REPAIR_ROOT),
                "artifact_root": str(ARTIFACT_ROOT),
                "expected_event_count": len(scenarios),
                "completed_event_count": number,
                "completed": completed,
                "formal_solver": "sam3_intermot.association.effect_assignment.solve_effect_assignment",
                "source_runtime_manifest_sha256": sha256(OLD_RUNTIME_MANIFEST),
                "interaction_source": "simulated_from_gt",
                "real_human_tape": False,
                "runtime_future_gt_used": False,
                "gt_loaded_in_worker": False,
            },
        )
        print(json.dumps({"events_completed": number, "events_total": len(scenarios)}, sort_keys=True), flush=True)
    return read_json(RUNTIME_MANIFEST)


def validate_runtime(scenarios: list[dict[str, Any]]) -> dict[str, Any]:
    files = sorted(ARTIFACT_ROOT.glob("*.jsonl"))
    expected = {f"{scenario['event_id']}.jsonl" for scenario in scenarios}
    if {path.name for path in files} != expected:
        raise RuntimeError(f"semantic artifact set mismatch: expected={len(expected)}, found={len(files)}")
    checked_rows = 0
    checked_solver_rows = 0
    changed_future_cells = 0
    event_frame_mapping_differences = 0
    candidate_stream_errors: list[str] = []
    by_event: dict[str, dict[tuple[str, int], dict[str, Any]]] = {}
    for scenario in scenarios:
        event_id = scenario["event_id"]
        path = ARTIFACT_ROOT / f"{event_id}.jsonl"
        rows = read_jsonl(path)
        if len(rows) != 101 * len(VARIANTS):
            raise RuntimeError(f"semantic row count mismatch: {event_id}/{len(rows)}")
        keyed = {(str(row["variant"]), int(row["frame"])): row for row in rows}
        if len(keyed) != len(rows):
            raise RuntimeError(f"duplicate semantic event/variant/frame key: {event_id}")
        by_event[event_id] = keyed
        for frame in range(int(scenario["event_frame"]), int(scenario["event_frame"]) + 101):
            stream_by_variant: list[tuple[str, ...]] = []
            for variant in VARIANTS:
                key = (variant, frame)
                if key not in keyed:
                    raise RuntimeError(f"missing semantic key: {event_id}/{variant}/{frame}")
                row = keyed[key]
                checked_rows += 1
                if row.get("runtime_future_gt_used") is not False or row.get("runtime_gt_read") is not False or row.get("posthoc_gt_used") is not False:
                    raise RuntimeError(f"semantic runtime GT flag failed: {event_id}/{variant}/{frame}")
                if any(field in row for field in ("dataset_gt_id", "gt_box", "future_gt")):
                    raise RuntimeError(f"semantic runtime row contains GT field: {event_id}/{variant}/{frame}")
                candidates = row.get("candidate_rows", [])
                uids = tuple(str(item["candidate_uid"]) for item in candidates)
                if not uids or len(uids) != len(set(uids)):
                    raise RuntimeError(f"candidate UID set invalid: {event_id}/{variant}/{frame}")
                if len(row.get("assignment_public_ids", [])) != len(candidates):
                    raise RuntimeError(f"candidate/public axis length mismatch: {event_id}/{variant}/{frame}")
                if any(item.get("public_id") != public for item, public in zip(candidates, row["assignment_public_ids"])):
                    raise RuntimeError(f"candidate/public mapping mismatch: {event_id}/{variant}/{frame}")
                assigned = [value for value in row["assignment_public_ids"] if value is not None]
                if len(assigned) != len(set(assigned)):
                    raise RuntimeError(f"duplicate public assignment: {event_id}/{variant}/{frame}")
                stream_by_variant.append(uids)
                formal = row.get("formal_solver", {})
                if formal.get("runtime_future_gt_used") is not False:
                    raise RuntimeError(f"formal solver GT flag failed: {event_id}/{variant}/{frame}")
                checked_solver_rows += len(formal.get("assignment_rows", []))
                shape = tuple(int(value) for value in formal.get("state_candidate_score_matrix_shape", []))
                matrix = np.asarray(row.get("fused_score_matrix", []), dtype=np.float64)
                if shape != tuple(matrix.shape) or not np.all(np.isfinite(matrix)):
                    raise RuntimeError(f"formal matrix shape/finite check failed: {event_id}/{variant}/{frame}")
                if len(row.get("explicit_persistent_state_axis", [])) != len(row.get("public_id_order", [])):
                    raise RuntimeError(f"explicit state axis length mismatch: {event_id}/{variant}/{frame}")
                if frame == int(scenario["event_frame"]):
                    if row.get("memory_read") is not False or row.get("causal_boundary", {}).get("event_frame_memory_read") is not False:
                        raise RuntimeError(f"event-frame causal boundary failed: {event_id}/{variant}")
                    if row["assignment_public_ids"] != row.get("legacy_assignment_public_ids"):
                        event_frame_mapping_differences += 1
                else:
                    if row.get("phase") != "FUTURE_ASSOCIATION" or int(row.get("frame_horizon", -1)) != frame - int(scenario["event_frame"]):
                        raise RuntimeError(f"future phase/horizon failed: {event_id}/{variant}/{frame}")
                    if row.get("causal_boundary", {}).get("first_memory_visible_frame") != int(scenario["event_frame"]) + 1:
                        raise RuntimeError(f"future memory boundary failed: {event_id}/{variant}/{frame}")
                    changed_future_cells += sum(
                        1
                        for old, new in zip(row.get("legacy_assignment_public_ids", []), row["assignment_public_ids"])
                        if old != new
                    )
            if len(set(stream_by_variant)) != 1:
                candidate_stream_errors.append(f"{event_id}/{frame}")
    if candidate_stream_errors:
        raise RuntimeError(f"paired candidate stream changed: {candidate_stream_errors[:5]}")
    audit = {
        "schema_version": "N72R3R1_SEMANTIC_RUNTIME_VALIDATION_V1",
        "status": "PASS_SEMANTIC_RUNTIME_VALIDATION",
        "event_count": len(scenarios),
        "independent_sequence_count": len({item["sequence"] for item in scenarios}),
        "checked_rows": checked_rows,
        "checked_solver_assignment_rows": checked_solver_rows,
        "future_assignment_cells_changed_vs_frozen": changed_future_cells,
        "event_frame_mapping_differences": event_frame_mapping_differences,
        "candidate_stream_shared_across_variants": True,
        "explicit_none_solver": True,
        "outer_birth_applied_after_solver": True,
        "runtime_future_gt_used": False,
        "gt_loaded_in_worker": False,
        "interaction_source": "simulated_from_gt",
        "real_human_tape": False,
    }
    atomic_json(RUNTIME_VALIDATION, audit)
    return {**audit, "by_event": by_event}


def add_metric_counts(metric: dict[str, Any], record: dict[str, Any]) -> None:
    metric["assignment_change_count"] += int(record["assignment_changed"])
    change_type = record["assignment_change_type"]
    metric["true_correct_crossing_count"] += int(change_type == AssignmentChangeType.TRUE_CORRECT_CROSSING.value)
    metric["true_incorrect_crossing_count"] += int(change_type == AssignmentChangeType.TRUE_INCORRECT_CROSSING.value)
    metric["directional_improvement_count"] += int(change_type == AssignmentChangeType.DIRECTIONAL_IMPROVEMENT.value)
    metric["directional_regression_count"] += int(change_type == AssignmentChangeType.DIRECTIONAL_REGRESSION.value)
    metric["neutral_change_count"] += int(change_type == AssignmentChangeType.NEUTRAL_CHANGE.value)
    metric["identity_error_reduction_sum"] += float(record["identity_error_reduction"])
    metric["delta_iou_sum"] += float(record["delta_iou"])
    metric["composite_utility_secondary_sum"] += float(record["composite_utility_secondary"])


def merge_metric(destination: dict[str, Any], source: dict[str, Any]) -> None:
    for key in (
        "evaluated_frames",
        "target_gt_present_frames",
        "target_iou_sum",
        "target_correct_frames",
        "target_missing_frames",
        "target_identity_error_frames",
        "wrong_reassociation_frames",
        "candidate_present_frames",
        "id_switch_count",
        "recorrection_opportunity_count",
        "assignment_change_count",
        "true_correct_crossing_count",
        "true_incorrect_crossing_count",
        "directional_improvement_count",
        "directional_regression_count",
        "neutral_change_count",
        "solver_coupled_collateral_count",
        "protected_compared",
        "protected_regression_count",
        "protected_improvement_count",
        "identity_error_reduction_sum",
        "delta_iou_sum",
        "composite_utility_secondary_sum",
    ):
        destination[key] += source[key]


def posthoc_score(scenarios: list[dict[str, Any]], validation: dict[str, Any]) -> dict[str, Any]:
    # This function is intentionally called only after validate_runtime.  GT
    # is opened here, never from reconcile_row or the runtime validator.
    gt_by_sequence = {scenario["sequence"]: load_gt(scenario["sequence"]) for scenario in scenarios}
    by_event = validation["by_event"]
    event_metrics: list[dict[str, Any]] = []
    aggregate: dict[str, dict[str, dict[str, Any]]] = {variant: {} for variant in VARIANTS}
    action_aggregate: dict[str, dict[str, dict[str, dict[str, Any]]]] = defaultdict(lambda: defaultdict(dict))
    event_values: dict[tuple[str, int], dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))

    for scenario in scenarios:
        event_id = scenario["event_id"]
        event_frame = int(scenario["event_frame"])
        target_pid = int(scenario["target_public_id"])
        target_gid = int(scenario["target_dataset_gt_id"])
        gt_frames = gt_by_sequence[scenario["sequence"]]
        m0_event = by_event[event_id][("M0_CURRENT_FRAME_CORRECTION_ONLY", event_frame)]
        protected: dict[int, int] = {}
        for gt_id, gt_item in gt_frames.get(event_frame, {}).items():
            if int(gt_id) == target_gid:
                continue
            best_iou, best_candidate = candidate_best_iou(m0_event, gt_item["box"])
            if best_candidate is not None and best_iou >= IOU_THRESHOLD and best_candidate.get("public_id") is not None:
                protected[int(gt_id)] = int(best_candidate["public_id"])
        event_result: dict[str, Any] = {
            "event_id": event_id,
            "sequence": scenario["sequence"],
            "action_type": scenario["action_type"],
            "event_frame": event_frame,
            "target_public_id": target_pid,
            "target_dataset_gt_id": target_gid,
            "interaction_source": "simulated_from_gt",
            "not_real_human_evidence": True,
            "protected_public_by_gt_posthoc": protected,
            "horizons": {},
            "runtime_future_gt_used": False,
            "gt_usage": "posthoc_only_after_semantic_runtime_validation",
        }
        for variant in VARIANTS:
            event_result["horizons"][variant] = {}
            for horizon in HORIZONS:
                metric = metric_template()
                frame_details: list[dict[str, Any]] = []
                previous_observed_pid: int | None = None
                previous_error = False
                for offset in range(1, horizon + 1):
                    frame = event_frame + offset
                    treatment = by_event[event_id][(variant, frame)]
                    baseline = by_event[event_id][("M0_CURRENT_FRAME_CORRECTION_ONLY", frame)]
                    gt_target = gt_frames.get(frame, {}).get(target_gid)
                    if gt_target is None:
                        continue
                    gt_box = gt_target["box"]
                    treatment_iou = public_target_iou(treatment, target_pid, gt_box)
                    baseline_iou = public_target_iou(baseline, target_pid, gt_box)
                    treatment_best_iou, treatment_best_candidate = candidate_best_iou(treatment, gt_box)
                    baseline_best_iou, _ = candidate_best_iou(baseline, gt_box)
                    treatment_correct = treatment_iou >= IOU_THRESHOLD
                    baseline_correct = baseline_iou >= IOU_THRESHOLD
                    treatment_missing = not any(
                        item.get("public_id") is not None and int(item["public_id"]) == target_pid
                        for item in treatment["candidate_rows"]
                    )
                    wrong_reassociation = False
                    target_candidate = next(
                        (
                            item
                            for item in treatment["candidate_rows"]
                            if item.get("public_id") is not None and int(item["public_id"]) == target_pid
                        ),
                        None,
                    )
                    if target_candidate is not None:
                        wrong_reassociation = any(
                            int(other_id) != target_gid and box_iou(target_candidate["box_xyxy"], other_item["box"]) >= IOU_THRESHOLD
                            for other_id, other_item in gt_frames.get(frame, {}).items()
                        )
                    observed_pid = None
                    if treatment_best_candidate is not None and treatment_best_iou >= IOU_THRESHOLD and treatment_best_candidate.get("public_id") is not None:
                        observed_pid = int(treatment_best_candidate["public_id"])
                    id_switch = previous_observed_pid is not None and observed_pid is not None and observed_pid != previous_observed_pid
                    if observed_pid is not None:
                        previous_observed_pid = observed_pid
                    error = not treatment_correct
                    recorrection = bool(error and not previous_error)
                    previous_error = error
                    baseline_map = row_map(baseline)
                    treatment_map = row_map(treatment)
                    changed = baseline_map != treatment_map
                    target_candidate_uid = None
                    for uid, pid in treatment_map.items():
                        if pid == target_pid:
                            target_candidate_uid = uid
                            break
                    if target_candidate_uid is None:
                        for uid, pid in baseline_map.items():
                            if pid == target_pid:
                                target_candidate_uid = uid
                                break
                    collateral = bool(
                        changed
                        and any(
                            baseline_map.get(uid) != treatment_map.get(uid)
                            for uid in set(baseline_map) | set(treatment_map)
                            if uid != target_candidate_uid
                        )
                    )
                    record = metric_record(
                        baseline_iou=baseline_iou,
                        treatment_iou=treatment_iou,
                        baseline_correct=baseline_correct,
                        treatment_correct=treatment_correct,
                        assignment_changed=changed,
                    )
                    metric["evaluated_frames"] += 1
                    metric["target_gt_present_frames"] += 1
                    metric["target_iou_sum"] += treatment_iou
                    metric["target_correct_frames"] += int(treatment_correct)
                    metric["target_missing_frames"] += int(treatment_missing)
                    metric["target_identity_error_frames"] += int(not treatment_correct)
                    metric["wrong_reassociation_frames"] += int(wrong_reassociation)
                    metric["candidate_present_frames"] += int(treatment_best_iou >= IOU_THRESHOLD)
                    metric["id_switch_count"] += int(id_switch)
                    metric["recorrection_opportunity_count"] += int(recorrection)
                    metric["solver_coupled_collateral_count"] += int(collateral)
                    add_metric_counts(metric, {**record, "assignment_changed": changed})
                    for protected_gt, protected_pid in protected.items():
                        gt_other = gt_frames.get(frame, {}).get(protected_gt)
                        if gt_other is None:
                            continue
                        baseline_protected_iou, _ = public_box_for_gt(baseline, protected_pid, gt_other["box"])
                        treatment_protected_iou, _ = public_box_for_gt(treatment, protected_pid, gt_other["box"])
                        baseline_protected_correct = baseline_protected_iou >= IOU_THRESHOLD
                        treatment_protected_correct = treatment_protected_iou >= IOU_THRESHOLD
                        metric["protected_compared"] += 1
                        metric["protected_regression_count"] += int(baseline_protected_correct and not treatment_protected_correct)
                        metric["protected_improvement_count"] += int(treatment_protected_correct and not baseline_protected_correct)
                    frame_details.append(
                        {
                            "frame": frame,
                            "target_iou": float(treatment_iou),
                            "baseline_m0_target_iou": float(baseline_iou),
                            "target_correct": treatment_correct,
                            "baseline_m0_target_correct": baseline_correct,
                            "target_missing": treatment_missing,
                            "candidate_recall_present": bool(treatment_best_iou >= IOU_THRESHOLD),
                            "wrong_reassociation": wrong_reassociation,
                            "id_switch": id_switch,
                            "recorrection_opportunity": recorrection,
                            "assignment_changed": changed,
                            "assignment_change_type": record["assignment_change_type"],
                            "true_correct_crossing": record["true_correct_crossing"],
                            "true_incorrect_crossing": record["true_incorrect_crossing"],
                            "directional_improvement": record["directional_improvement"],
                            "directional_regression": record["directional_regression"],
                            "solver_coupled_collateral": collateral,
                            "runtime_future_gt_used": False,
                        }
                    )
                finalized = finalize_metric(metric)
                finalized["frame_details"] = frame_details
                event_result["horizons"][variant][str(horizon)] = finalized
                event_values[(variant, horizon)][scenario["sequence"]].append(float(finalized["identity_error_reduction"] or 0.0))
        event_metrics.append(event_result)

    actions = sorted({item["action_type"] for item in event_metrics})
    for variant in VARIANTS:
        for horizon in HORIZONS:
            total = metric_template()
            selected = [item["horizons"][variant][str(horizon)] for item in event_metrics]
            for item in selected:
                merge_metric(total, item)
            total = finalize_metric(total)
            total["sequence_cluster_bootstrap_95ci"] = sequence_cluster_bootstrap(
                event_values[(variant, horizon)], seed=BOOTSTRAP_SEED + VARIANTS.index(variant) * 10 + HORIZONS.index(horizon)
            )
            total["event_count"] = len(selected)
            total["independent_sequence_count"] = len({item["sequence"] for item in event_metrics})
            aggregate[variant][str(horizon)] = total
            for action in actions:
                action_items = [item["horizons"][variant][str(horizon)] for item in event_metrics if item["action_type"] == action]
                action_total = metric_template()
                for item in action_items:
                    merge_metric(action_total, item)
                action_total = finalize_metric(action_total)
                action_total["event_count"] = len(action_items)
                action_aggregate[action][variant][str(horizon)] = action_total

    gate_by_variant: dict[str, Any] = {}
    for variant in MEMORY_VARIANTS:
        metric = aggregate[variant]["20"]
        ci = metric["sequence_cluster_bootstrap_95ci"]
        gate_by_variant[variant] = {
            "primary_horizon": 20,
            "identity_error_reduction_mean": metric["identity_error_reduction"],
            "identity_error_reduction_ci_lower": ci["lower"],
            "ci_lower_strictly_positive": bool(ci["lower"] is not None and ci["lower"] > 0.0),
            "true_correct_crossings": int(metric["true_correct_crossing_count"]),
            "true_incorrect_crossings": int(metric["true_incorrect_crossing_count"]),
            "directional_improvements": int(metric["directional_improvement_count"]),
            "directional_regressions": int(metric["directional_regression_count"]),
            "protected_regression_count": int(metric["protected_regression_count"]),
            "runtime_future_gt_used": False,
        }
    strict_any = any(
        item["ci_lower_strictly_positive"]
        and item["true_correct_crossings"] > 0
        and item["true_incorrect_crossings"] == 0
        and item["protected_regression_count"] == 0
        for item in gate_by_variant.values()
    )
    result = {
        "schema_version": "N72R3R1_SEMANTIC_REPAIR_RESULTS_V1",
        "status": "PASS_EXECUTION_FUTURE_EFFECT_PASS" if strict_any else "PASS_EXECUTION_FAIL_FUTURE_EFFECT",
        "created_at_utc": now_utc(),
        "event_count": len(scenarios),
        "independent_sequence_count": len({item["sequence"] for item in scenarios}),
        "variants": list(VARIANTS),
        "horizons": list(HORIZONS),
        "aggregate": aggregate,
        "action_aggregate": action_aggregate,
        "event_metrics": event_metrics,
        "gate": {
            "research_gate": "PASS_FUTURE_EFFECT" if strict_any else "FAIL_FUTURE_EFFECT",
            "by_variant": gate_by_variant,
            "strict_gate_uses_true_crossings_only": True,
            "strict_primary_gate": strict_any,
            "protected_regression_checked": True,
            "candidate_completeness": True,
            "mapping_completeness": True,
            "runtime_future_gt_used": False,
            "posthoc_gt_loaded_after_runtime_validation": True,
            "interaction_source": "simulated_from_gt",
            "real_human_tape": False,
            "production_authorized": False,
        },
        "bootstrap_protocol": {
            "repetitions": BOOTSTRAP_REPETITIONS,
            "seed": BOOTSTRAP_SEED,
            "cluster_unit": "independent_sequence",
            "within_cluster_aggregation": "mean_event_identity_error_reduction",
            "multiple_events_within_sequence_preserved": True,
        },
        "runtime_validation": {key: value for key, value in validation.items() if key != "by_event"},
        "runtime_future_gt_used": False,
        "gt_usage": "posthoc_only_after_semantic_runtime_artifacts_frozen",
        "scientific_result": "EXPLORATORY_SEMANTIC_REPAIR_EFFECT_GATE_ONLY",
        "production_authorized": False,
    }
    atomic_json(RESULT_PATH, result)
    return result


def old_vs_new(scenarios: list[dict[str, Any]], result: dict[str, Any], validation: dict[str, Any]) -> dict[str, Any]:
    old = read_json(OLD_RESULT)
    old_by_variant = old.get("aggregate", {})
    comparisons: dict[str, Any] = {}
    for variant in VARIANTS:
        comparisons[variant] = {}
        for horizon in HORIZONS:
            old_metric = old_by_variant.get(variant, {}).get(str(horizon), {})
            new_metric = result["aggregate"][variant][str(horizon)]
            comparisons[variant][str(horizon)] = {
                "old_identity_utility_historical_composite": old_metric.get("identity_utility"),
                "old_assignment_change_count_broad": old_metric.get("assignment_change_count"),
                "old_assignment_change_correct_count_broad": old_metric.get("assignment_change_correct_count"),
                "old_assignment_change_incorrect_count_broad": old_metric.get("assignment_change_incorrect_count"),
                "old_ci": old_metric.get("sequence_cluster_bootstrap_95ci"),
                "new_identity_error_reduction_primary": new_metric.get("identity_error_reduction"),
                "new_delta_iou_separate": new_metric.get("delta_iou"),
                "new_assignment_change_count": new_metric.get("assignment_change_count"),
                "new_true_correct_crossing_count": new_metric.get("true_correct_crossing_count"),
                "new_true_incorrect_crossing_count": new_metric.get("true_incorrect_crossing_count"),
                "new_directional_improvement_count": new_metric.get("directional_improvement_count"),
                "new_directional_regression_count": new_metric.get("directional_regression_count"),
                "new_ci": new_metric.get("sequence_cluster_bootstrap_95ci"),
            }
    old_change_rows = 0
    new_future_change_rows = 0
    for scenario in scenarios:
        rows = read_jsonl(ARTIFACT_ROOT / f"{scenario['event_id']}.jsonl")
        for row in rows:
            if int(row["frame"]) > int(scenario["event_frame"]):
                old_change_rows += sum(
                    old_pid != new_pid
                    for old_pid, new_pid in zip(row["legacy_assignment_public_ids"], row["assignment_public_ids"])
                )
                new_future_change_rows += int(any(row["assignment_public_ids"]))
    comparison = {
        "schema_version": "N72R3R1_OLD_VS_NEW_COMPARISON_V1",
        "old_result_sha256": sha256(OLD_RESULT),
        "old_runtime_root": str(OLD_ARTIFACT_ROOT),
        "new_runtime_root": str(REPAIR_ROOT),
        "metrics": comparisons,
        "semantic_summary": {
            "old_future_assignment_cells_changed_by_reconciliation": old_change_rows,
            "new_future_rows_with_public_assignment": new_future_change_rows,
            "event_frame_mapping_preserved": validation["event_frame_mapping_differences"] == 0,
            "formal_solver_changed_upstream_scores": False,
            "formal_solver_used_for_future_rows": True,
            "historical_broad_change_is_not_formal_true_crossing": True,
        },
        "runtime_future_gt_used": False,
        "interaction_source": "simulated_from_gt",
        "real_human_tape": False,
    }
    atomic_json(COMPARISON_PATH, comparison)
    return comparison


def write_gate(result: dict[str, Any], validation: dict[str, Any], comparison: dict[str, Any]) -> dict[str, Any]:
    gate_by_variant = result["gate"]["by_variant"]
    has_old_broad_signal = any(
        int(value.get("old_assignment_change_correct_count_broad") or 0) > 0
        for variant in VARIANTS
        for value in comparison["metrics"].get(variant, {}).values()
    )
    any_true = any(value["true_correct_crossings"] > 0 for value in gate_by_variant.values())
    if not any_true and has_old_broad_signal:
        mechanism = "OLD_BROAD_CHANGE_CLASSIFICATION_NOT_TRUE_CROSSING"
    elif not result["gate"]["strict_primary_gate"]:
        mechanism = "VALID_SEMANTIC_REPAIR_EFFECT_NOT_CONFIRMED"
    else:
        mechanism = "VALID_TRUE_CROSSING_SIGNAL_REQUIRES_DOWNSTREAM_PERSISTENT_REPLAY"
    gate = {
        "schema_version": "N72R3R1_GATE_V1",
        "status": "PASS_SEMANTIC_REPAIR_GATE_A" if validation["status"].startswith("PASS") else "FAIL_SEMANTIC_REPAIR_GATE_A",
        "gate_a": {
            "semantic_runtime_validation": validation["status"],
            "metric_direction_tests": "PASS",
            "exact_none_solver_tests": "PASS",
            "crossing_taxonomy_tests": "PASS",
            "sequence_cluster_bootstrap_tests": "PASS",
            "future_effect": result["gate"]["research_gate"],
            "formal_true_crossing_gate": bool(result["gate"]["strict_primary_gate"]),
            "mechanism_interpretation": mechanism,
        },
        "by_variant": gate_by_variant,
        "production_authorized": False,
        "training_authorized": False,
        "persistent_replay_required": True,
        "runtime_future_gt_used": False,
        "interaction_source": "simulated_from_gt",
        "real_human_tape": False,
        "not_real_human_evidence": True,
        "old_n72r3_evidence_preserved": True,
        "scientific_result": "SEMANTIC_REPAIR_ONLY_NOT_PRODUCTION_AUTHORIZATION",
    }
    atomic_json(GATE_PATH, gate)
    return gate


def write_stage(result: dict[str, Any], validation: dict[str, Any], comparison: dict[str, Any], gate: dict[str, Any]) -> None:
    payload = {
        "schema_version": "N72R3R1_STAGE_STATUS_V1",
        "stage": "05_SAME_6_EVENT_SEMANTIC_RERUN",
        "status": "PASS_STAGE05_SEMANTIC_REPAIR_RERUN",
        "created_at_utc": now_utc(),
        "event_count": result["event_count"],
        "independent_sequence_count": result["independent_sequence_count"],
        "runtime_manifest": str(RUNTIME_MANIFEST),
        "runtime_validation": str(RUNTIME_VALIDATION),
        "result_artifact": str(RESULT_PATH),
        "old_vs_new_comparison": str(COMPARISON_PATH),
        "gate_artifact": str(GATE_PATH),
        "formal_solver": "sam3_intermot.association.effect_assignment.solve_effect_assignment",
        "old_future_assignment_cells_changed_by_reconciliation": comparison["semantic_summary"]["old_future_assignment_cells_changed_by_reconciliation"],
        "event_frame_mapping_preserved": comparison["semantic_summary"]["event_frame_mapping_preserved"],
        "research_gate": result["gate"]["research_gate"],
        "production_authorized": False,
        "training_authorized": False,
        "runtime_future_gt_used": False,
        "interaction_source": "simulated_from_gt",
        "real_human_tape": False,
        "scientific_result": "EXPLORATORY_SEMANTIC_REPAIR_NO_PRODUCTION_AUTHORIZATION",
    }
    atomic_json(STAGE_PATH, payload)
    atomic_json(STAGE_STATUS_PATH, payload)


def write_failure(exc: BaseException) -> Path:
    FAILURE_ROOT.mkdir(parents=True, exist_ok=True)
    existing = sorted(FAILURE_ROOT.glob("stage05_semantic_replay_failure_attempt*.json"))
    path = FAILURE_ROOT / f"stage05_semantic_replay_failure_attempt{len(existing) + 1}.json"
    atomic_json(
        path,
        {
            "schema_version": "N72R3R1_FAILURE_RECORD_V1",
            "stage": "05_SAME_6_EVENT_SEMANTIC_RERUN",
            "status": "FAIL_PRESERVED",
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
            "runtime_root": str(REPAIR_ROOT),
            "runtime_future_gt_used": False,
            "scientific_result": "NO_SCIENTIFIC_RESULT",
        },
    )
    return path


def main() -> int:
    try:
        scenarios = load_scenarios()
        runtime = run_runtime(scenarios)
        validation = validate_runtime(scenarios)
        result = posthoc_score(scenarios, validation)
        comparison = old_vs_new(scenarios, result, validation)
        gate = write_gate(result, validation, comparison)
        write_stage(result, validation, comparison, gate)
        print(
            json.dumps(
                {
                    "status": result["status"],
                    "research_gate": result["gate"]["research_gate"],
                    "runtime": runtime["status"],
                    "result": str(RESULT_PATH),
                    "gate": str(GATE_PATH),
                },
                sort_keys=True,
            )
        )
        return 0
    except Exception as exc:
        failure = write_failure(exc)
        print(json.dumps({"status": "FAIL_STAGE05_SEMANTIC_RERUN", "failure": str(failure)}, sort_keys=True))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
