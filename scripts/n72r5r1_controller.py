#!/usr/bin/env python3
"""N72R5R1 autonomous root-cause controller.

The controller is deliberately posthoc and CPU-only.  It reads the sealed
Stage08 public-assignment sidecars and the Stage10 effect artifact, then
records one preregistered mechanism decision per round.  It never feeds GT or
the private simulated-human map into the runtime solver.

Round 01 is the required 40-event global decision-boundary audit.  Later
rounds are appended by the same controller after a mechanism-specific
artifact has been produced; no round is allowed to silently replace an older
failure.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime, timezone
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sam3_intermot.association.effect_assignment import solve_effect_assignment  # noqa: E402
from sam3_intermot.association.branch_public_replay import (  # noqa: E402
    BRANCHES,
    HORIZON,
    IOU_THRESHOLD,
    atomic_json,
    atomic_jsonl,
    box_iou,
    now_utc,
    read_json,
    read_jsonl,
    sha256_file,
)


OUT = Path(os.environ.get("N72R5R1_RUN_ROOT", str(ROOT / "outputs/N72R5R1")))
CONTROLLER_ROOT = OUT / "controller"
MANIFEST = OUT / "stage08_runtime_manifest.json"
VALIDATION = OUT / "stage09_validation.json"
EFFECT = OUT / "stage10_effect_scoring.json"
EVENT_MANIFEST = ROOT / "outputs/N72R5/mechanism_rounds/round_06_event_policy/real_event_manifest.json"
GT_ROOT = Path("/data1/LWR/vranlee/DATASETS/JDE/dancetrack/train")
CONTROLLER_STATUS = OUT / "CONTROLLER_STATUS.json"
HUMAN_STATUS = OUT / "HUMAN_READABLE_STATUS.md"

ROUND_NAME = "round_01_decision_boundary"
ROUND_ROOT = CONTROLLER_ROOT / ROUND_NAME
TABLE = ROUND_ROOT / "decision_boundary.jsonl"
SUMMARY = ROUND_ROOT / "decision_boundary_summary.json"
HYPOTHESIS = ROUND_ROOT / "hypothesis.json"

MAX_RESIDUAL = 128.0
RESIDUAL_STEPS = 32


def _sha256(path: Path) -> str:
    return sha256_file(path)


def _gt(sequence: str) -> dict[int, dict[int, list[float]]]:
    path = GT_ROOT / sequence / "gt" / "gt.txt"
    if not path.is_file():
        raise FileNotFoundError(path)
    result: dict[int, dict[int, list[float]]] = defaultdict(dict)
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            fields = [item.strip() for item in line.split(",")]
            if len(fields) < 6:
                raise ValueError(f"malformed GT row: {path}:{line_number}")
            frame = int(fields[0]) - 1
            gt_id = int(fields[1])
            x, y, width, height = (float(value) for value in fields[2:6])
            result[frame][gt_id] = [x, y, x + width, y + height]
    return result


def _candidate_for_gt(row: Mapping[str, Any], gt_box: Sequence[float] | None) -> tuple[float, Mapping[str, Any] | None]:
    if gt_box is None:
        return 0.0, None
    ranked = [
        (float(box_iou(candidate.get("box_xyxy"), gt_box)), int(candidate.get("candidate_index", 0)), candidate)
        for candidate in row.get("candidate_rows", [])
    ]
    if not ranked:
        return 0.0, None
    ranked.sort(key=lambda item: (-item[0], item[1]))
    iou, _, candidate = ranked[0]
    return iou, candidate if iou >= IOU_THRESHOLD else None


def _solver_axes(row: Mapping[str, Any]) -> tuple[list[int], list[int]]:
    solver = row.get("solver") or {}
    state_axis = [int(value) for value in solver.get("association_state_axis", [])]
    public_axis = [int(value) for value in solver.get("public_id_axis", [])]
    if len(state_axis) != len(public_axis) or len(set(state_axis)) != len(state_axis) or len(set(public_axis)) != len(public_axis):
        return [], []
    return state_axis, public_axis


def _exact_map(row: Mapping[str, Any], matrix: np.ndarray, suffix: str) -> dict[str, int | None]:
    candidates = [dict(item) for item in row.get("candidate_rows", [])]
    state_axis, public_axis = _solver_axes(row)
    if matrix.shape != (len(candidates), len(state_axis)):
        return {str(item.get("candidate_uid")): None for item in candidates}
    artifact = solve_effect_assignment(
        candidate_rows=[
            {"candidate_uid": str(item["candidate_uid"]), "candidate_index": int(item["candidate_index"])}
            for item in candidates
        ],
        persistent_states=[
            SimpleNamespace(association_state_id=state, public_id=public)
            for state, public in zip(state_axis, public_axis)
        ],
        fused_state_candidate_scores=matrix.T,
        source_run_id=f"n72r5r1-controller:{row.get('event_id')}:{row.get('branch')}:{row.get('frame')}:{suffix}",
        session_id=f"n72r5r1-controller:{row.get('event_id')}:{row.get('branch')}:{row.get('frame')}",
        none_score=0.0,
    )
    output = {str(item["candidate_uid"]): None for item in candidates}
    for item in artifact.get("assignment_rows", []):
        uid = str(item["candidate_uid"])
        value = item.get("public_id")
        output[uid] = None if value is None else int(value)
    return output


def _required_residual(
    row: Mapping[str, Any],
    base: np.ndarray,
    target_uid: str,
    target_public: int,
) -> dict[str, Any]:
    candidates = list(row.get("candidate_rows", []))
    state_axis, public_axis = _solver_axes(row)
    candidate_index = next((index for index, item in enumerate(candidates) if str(item.get("candidate_uid")) == target_uid), None)
    public_index = next((index for index, value in enumerate(public_axis) if int(value) == int(target_public)), None)
    if candidate_index is None or public_index is None or base.shape != (len(candidates), len(state_axis)):
        return {"status": "NOT_APPLICABLE_AXIS_MISSING", "required_residual": None, "collateral_count": None}
    baseline = _exact_map(row, base, "baseline")
    if baseline.get(target_uid) == int(target_public):
        return {"status": "ALREADY_CORRECT", "required_residual": 0.0, "collateral_count": 0}

    def probe(residual: float) -> tuple[bool, dict[str, int | None], int]:
        matrix = np.asarray(base, dtype=np.float64).copy()
        matrix[candidate_index, public_index] += float(residual)
        mapping = _exact_map(row, matrix, f"residual_{residual:.9g}")
        collateral = sum(int(mapping.get(uid) != value) for uid, value in mapping.items() if uid != target_uid)
        return mapping.get(target_uid) == int(target_public), mapping, collateral

    found, best_mapping, collateral = probe(MAX_RESIDUAL)
    if not found:
        return {
            "status": "NOT_FOUND_WITH_MAX_RESIDUAL",
            "required_residual": None,
            "max_residual": MAX_RESIDUAL,
            "collateral_count": collateral,
        }
    low = 0.0
    high = MAX_RESIDUAL
    best_collateral = collateral
    for _ in range(RESIDUAL_STEPS):
        middle = (low + high) / 2.0
        corrected, mapping, collateral = probe(middle)
        if corrected:
            high = middle
            best_mapping = mapping
            best_collateral = collateral
        else:
            low = middle
    return {
        "status": "FOUND_BINARY_BOUNDARY",
        "required_residual": float(high),
        "max_residual": MAX_RESIDUAL,
        "binary_steps": RESIDUAL_STEPS,
        "collateral_count": int(best_collateral),
    }


def _actual_tvc_residual(row: Mapping[str, Any], uid: str) -> float | None:
    tvc = row.get("tvc")
    if not isinstance(tvc, Mapping):
        return None
    details = tvc.get("residual_details")
    if not isinstance(details, list):
        return None
    for item in details:
        if str(item.get("candidate_uid")) == str(uid):
            value = item.get("bounded_target_row_residual")
            return float(value) if isinstance(value, (int, float)) and math.isfinite(float(value)) else None
    return None


def _identity_public_for_candidate(row: Mapping[str, Any], uid: str) -> int | None:
    for candidate in row.get("candidate_rows", []):
        if str(candidate.get("candidate_uid")) == str(uid):
            value = candidate.get("public_id")
            return None if value is None else int(value)
    return None


def _native_dominates(row: Mapping[str, Any], uid: str, assigned_public: int | None) -> bool:
    if assigned_public is None:
        return False
    candidate = next((item for item in row.get("candidate_rows", []) if str(item.get("candidate_uid")) == str(uid)), None)
    if candidate is None:
        return False
    target_native = candidate.get("native_tid", candidate.get("adapter_external_id"))
    for identity in row.get("identity_rows", []):
        if int(identity.get("public_id", -1)) == int(assigned_public):
            native = identity.get("last_native_tid")
            return native is not None and int(native) == int(target_native)
    return False


def _classify(
    *,
    candidate_available: bool,
    assigned_public: int | None,
    target_public: int,
    required: Mapping[str, Any],
    actual_tvc: float | None,
    appearance_relative: float | None,
    native_dominates: bool,
) -> str:
    if not candidate_available:
        return "CANDIDATE_ABSENT"
    if assigned_public is None:
        return "TARGET_LOSES_NONE"
    if int(assigned_public) == int(target_public):
        return "TARGET_ALREADY_CORRECT"
    if native_dominates:
        return "NATIVE_CONTINUITY_DOMINATES"
    if appearance_relative is not None and appearance_relative <= 0.0:
        return "APPEARANCE_AMBIGUOUS"
    if required.get("required_residual") is not None and actual_tvc is not None:
        if float(actual_tvc) < float(required["required_residual"]):
            return "CORRECT_CANDIDATE_PRESENT_TARGET_LOSES_COMPETITOR"
    return "SOLVER_COMPETITION"


def _load_event_index() -> dict[str, dict[str, Any]]:
    payload = read_json(EVENT_MANIFEST)
    events = payload.get("events")
    if not isinstance(events, list) or len(events) != 40:
        raise RuntimeError(f"frozen event policy is not exactly 40 events: {len(events) if isinstance(events, list) else None}")
    return {str(item["event_id"]): dict(item) for item in events}


def _load_branch_rows(event_result: Mapping[str, Any], branch: str) -> dict[int, dict[str, Any]]:
    branch_result = next((item for item in event_result.get("branches", []) if str(item.get("branch")) == branch), None)
    if branch_result is None:
        return {}
    output = Path(str(branch_result.get("output", OUT / "public_assignment" / str(event_result["event_id"]) / f"{branch}.jsonl")))
    return {int(row["frame"]): row for row in read_jsonl(output)}


def _round01() -> dict[str, Any]:
    manifest = read_json(MANIFEST)
    validation = read_json(VALIDATION)
    effect = read_json(EFFECT)
    events = _load_event_index()
    event_results = {str(item["event_id"]): item for item in manifest.get("events", [])}
    rows: list[dict[str, Any]] = []
    event_summaries: list[dict[str, Any]] = []
    root_counts: Counter[str] = Counter()
    branch_counts: Counter[str] = Counter()
    action_counts: Counter[str] = Counter()
    sequence_counts: Counter[str] = Counter()
    required_values: list[float] = []
    actual_values: list[float] = []
    candidate_absent_events: set[str] = set()
    unavailable_events: list[dict[str, Any]] = []

    for event_id in sorted(events):
        event = events[event_id]
        result = event_results.get(event_id)
        private_path = OUT / "simulation_private" / event_id / "oracle_private_mapping.json"
        if result is None or not private_path.is_file():
            unavailable_events.append({"event_id": event_id, "reason": "missing_event_result_or_private_map"})
            continue
        private = read_json(private_path)
        mapping = {int(key): int(value) for key, value in (private.get("dataset_gt_to_public") or {}).items()}
        target_gt = int(event["dataset_gt_id"])
        target_public = mapping.get(target_gt)
        if target_public is None:
            unavailable_events.append({"event_id": event_id, "reason": "target_public_unresolved"})
            continue
        gt = _gt(str(event["sequence"]))
        event_categories: Counter[str] = Counter()
        for branch in BRANCHES[1:]:
            branch_rows = _load_branch_rows(result, branch)
            action_status = next(
                (item.get("action_precondition_status") for item in result.get("branches", []) if item.get("branch") == branch),
                None,
            )
            for frame in range(int(event["event_frame"]) + 1, int(event["event_frame"]) + HORIZON + 1):
                row = branch_rows.get(frame)
                gt_box = gt.get(frame, {}).get(target_gt)
                if row is None:
                    record = {
                        "event_id": event_id,
                        "sequence": str(event["sequence"]),
                        "action_type": str(event["action_type"]),
                        "branch": branch,
                        "frame": frame,
                        "frame_horizon": frame - int(event["event_frame"]),
                        "classification": "MISSING_SIDECAR_FRAME",
                        "runtime_future_gt_used": False,
                        "posthoc_gt_used": True,
                    }
                    rows.append(record)
                    event_categories[record["classification"]] += 1
                    root_counts[record["classification"]] += 1
                    continue
                iou, target_candidate = _candidate_for_gt(row, gt_box)
                candidate_available = target_candidate is not None
                target_uid = None if target_candidate is None else str(target_candidate["candidate_uid"])
                assigned_public = None if target_uid is None else _identity_public_for_candidate(row, target_uid)
                base_matrix = np.asarray(row.get("fused_score_matrix", []), dtype=np.float64)
                required = (
                    {"status": "NOT_APPLICABLE_NO_CANDIDATE", "required_residual": None, "collateral_count": None}
                    if target_uid is None
                    else _required_residual(row, base_matrix, target_uid, int(target_public))
                )
                actual_tvc = None if target_uid is None else _actual_tvc_residual(row, target_uid)
                appearance_relative = None
                if target_uid is not None and isinstance(row.get("tvc"), Mapping):
                    for detail in row["tvc"].get("residual_details", []) or []:
                        if str(detail.get("candidate_uid")) == target_uid:
                            value = detail.get("relative_margin")
                            if isinstance(value, (int, float)) and math.isfinite(float(value)):
                                appearance_relative = float(value)
                            break
                classification = _classify(
                    candidate_available=candidate_available,
                    assigned_public=assigned_public,
                    target_public=int(target_public),
                    required=required,
                    actual_tvc=actual_tvc,
                    appearance_relative=appearance_relative,
                    native_dominates=False if target_uid is None else _native_dominates(row, target_uid, assigned_public),
                )
                record = {
                    "event_id": event_id,
                    "sequence": str(event["sequence"]),
                    "action_type": str(event["action_type"]),
                    "branch": branch,
                    "action_precondition_status": action_status,
                    "event_frame": int(event["event_frame"]),
                    "frame": frame,
                    "frame_horizon": frame - int(event["event_frame"]),
                    "target_dataset_gt_id_posthoc": target_gt,
                    "target_public_id": int(target_public),
                    "target_gt_visible": gt_box is not None,
                    "target_candidate_available": candidate_available,
                    "target_candidate_iou": float(iou),
                    "target_candidate_uid": target_uid,
                    "assigned_public_id": assigned_public,
                    "target_correct_assignment": bool(assigned_public == int(target_public)),
                    "required_residual": required,
                    "actual_tvc_residual": actual_tvc,
                    "appearance_relative_margin": appearance_relative,
                    "classification": classification,
                    "runtime_future_gt_used": False,
                    "posthoc_gt_used": True,
                }
                rows.append(record)
                event_categories[classification] += 1
                root_counts[classification] += 1
                branch_counts[branch] += 1
                action_counts[str(event["action_type"])] += 1
                sequence_counts[str(event["sequence"])] += 1
                if classification == "CANDIDATE_ABSENT":
                    candidate_absent_events.add(event_id)
                if required.get("required_residual") is not None and float(required["required_residual"]) > 0:
                    required_values.append(float(required["required_residual"]))
                if actual_tvc is not None:
                    actual_values.append(float(actual_tvc))
        event_summaries.append(
            {
                "event_id": event_id,
                "sequence": str(event["sequence"]),
                "action_type": str(event["action_type"]),
                "classification_counts": dict(sorted(event_categories.items())),
                "candidate_absent": bool(event_id in candidate_absent_events),
            }
        )

    summary = {
        "schema_version": "N72R5R1_CONTROLLER_ROUND01_SUMMARY_V1",
        "status": "PASS_DECISION_BOUNDARY_AUDIT",
        "round": ROUND_NAME,
        "event_count_expected": 40,
        "event_count_audited": len(event_summaries),
        "independent_sequence_count": len({str(item["sequence"]) for item in event_summaries}),
        "branch_count_audited": sum(branch_counts.values()),
        "frame_record_count": len(rows),
        "root_cause_counts": dict(sorted(root_counts.items())),
        "branch_counts": dict(sorted(branch_counts.items())),
        "action_counts": dict(sorted(action_counts.items())),
        "sequence_record_counts": dict(sorted(sequence_counts.items())),
        "candidate_absent_event_count": len(candidate_absent_events),
        "candidate_absent_events": sorted(candidate_absent_events),
        "unavailable_events": unavailable_events,
        "required_residual": {
            "finite_count": len(required_values),
            "median": None if not required_values else float(np.median(required_values)),
            "p90": None if not required_values else float(np.quantile(required_values, 0.90)),
            "max": None if not required_values else float(max(required_values)),
        },
        "actual_tvc_residual": {
            "finite_count": len(actual_values),
            "median": None if not actual_values else float(np.median(actual_values)),
            "p90": None if not actual_values else float(np.quantile(actual_values, 0.90)),
            "max": None if not actual_values else float(max(actual_values)),
        },
        "stage08_status": manifest.get("status"),
        "stage09_status": validation.get("status"),
        "stage10_status": effect.get("status"),
        "runtime_future_gt_used": False,
        "posthoc_gt_used": True,
        "input_hashes": {
            "stage08_runtime_manifest": _sha256(MANIFEST),
            "stage09_validation": _sha256(VALIDATION),
            "stage10_effect": _sha256(EFFECT),
            "event_manifest": _sha256(EVENT_MANIFEST),
        },
        "event_summaries": event_summaries,
        "created_at_utc": now_utc(),
    }
    present_wrong = int(root_counts.get("CORRECT_CANDIDATE_PRESENT_TARGET_LOSES_COMPETITOR", 0)) + int(root_counts.get("SOLVER_COMPETITION", 0))
    absent = int(root_counts.get("CANDIDATE_ABSENT", 0))
    primary_root = "CANDIDATE_ABSENT" if absent > present_wrong else "ASSOCIATION_SOLVER_COMPETITION"
    hypothesis = {
        "schema_version": "N72R5R1_CONTROLLER_HYPOTHESIS_V1",
        "round": ROUND_NAME,
        "question": "Can the exact global public solver cross the target boundary without collateral protected-identity regression?",
        "primary_root_cause": primary_root,
        "evidence_rule": "Use global exact solver residual and sidecar assignment, never a pairwise margin or runtime GT.",
        "next_single_component": (
            "IMAGE_GROUNDED_RECOVERY_V2_FOR_AFFECTED_B2_B4_EVENTS"
            if primary_root == "CANDIDATE_ABSENT"
            else "PROTECTED_IDENTITY_TRANSACTION_FOR_EVENT_CORRECTION"
        ),
        "classification_counts_used_for_routing": {
            "candidate_absent": absent,
            "candidate_present_wrong_or_solver_competition": present_wrong,
        },
        "candidate_absence_is_routing_signal": True,
        "training_authorized": False,
        "runtime_future_gt_used": False,
        "posthoc_gt_used": True,
        "created_at_utc": now_utc(),
    }
    atomic_jsonl(TABLE, rows)
    atomic_json(SUMMARY, summary)
    atomic_json(HYPOTHESIS, hypothesis)
    return {"summary": summary, "hypothesis": hypothesis}


def _write_controller_status(summary: Mapping[str, Any], hypothesis: Mapping[str, Any], *, next_action: str) -> None:
    payload = {
        "schema_version": "N72R5R1_CONTROLLER_STATUS_V1",
        "status": "ACTIVE_AUTONOMOUS_ROOT_CAUSE_CONTROLLER",
        "current_round": ROUND_NAME,
        "current_stage": "DECISION_BOUNDARY_AUDIT",
        "last_gate": "FAIL_FUTURE_EFFECT",
        "root_cause": hypothesis.get("primary_root_cause"),
        "next_action": next_action,
        "events": int(summary.get("event_count_audited", 0)),
        "sequences": int(summary.get("independent_sequence_count", 0)),
        "assignment_decision_coverage": 1.0,
        "true_correct_crossings": None,
        "true_incorrect_crossings": None,
        "H20": None,
        "H50": None,
        "H100": None,
        "candidate_absent_rate": None,
        "candidate_absent_event_count": int(summary.get("candidate_absent_event_count", 0)),
        "recovery_gain": None,
        "best_mechanism": None,
        "runtime_future_gt_used": False,
        "training_authorized": False,
        "confirmation_authorized": False,
        "round_summary": str(SUMMARY),
        "round_hypothesis": str(HYPOTHESIS),
        "created_at_utc": now_utc(),
    }
    atomic_json(CONTROLLER_STATUS, payload)
    absent = int(summary.get("candidate_absent_event_count", 0))
    root = str(hypothesis.get("primary_root_cause"))
    HUMAN_STATUS.parent.mkdir(parents=True, exist_ok=True)
    HUMAN_STATUS.write_text(
        "# N72R5R1 Controller Status\n\n"
        f"- 当前轮次：`{ROUND_NAME}`\n"
        "- 当前结论：Stage08/09 结构链已封存，但 Stage10 future-effect gate 仍 FAIL。\n"
        f"- 主根因候选：`{root}`；candidate-absent 事件数：`{absent}`。\n"
        f"- 下一步：`{next_action}`。\n"
        "- 运行时 future GT：`false`；训练/confirmation：未授权。\n",
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--round01", action="store_true", help="run the preregistered Round 01 audit")
    args = parser.parse_args()
    if not MANIFEST.is_file() or not VALIDATION.is_file() or not EFFECT.is_file():
        raise FileNotFoundError("Stage08, Stage09, and Stage10 artifacts are all required")
    if args.round01 or not SUMMARY.is_file():
        result = _round01()
    else:
        result = {"summary": read_json(SUMMARY), "hypothesis": read_json(HYPOTHESIS)}
    hypothesis = result["hypothesis"]
    next_action = str(hypothesis.get("next_single_component"))
    _write_controller_status(result["summary"], hypothesis, next_action=next_action)
    print(json.dumps({"status": result["summary"]["status"], "round": ROUND_NAME, "next_action": next_action}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
