#!/usr/bin/env python3
"""Post-hoc scoring and mechanism summaries for the N41-02 source replay.

All runtime workers must have completed before this script imports
DanceTrack GT.  GT is used only here for post-hoc metrics and never fed back
to an event, source, weight or branch.  The script keeps the raw worker
artifacts unchanged and writes aggregate/per-event post-hoc evidence under
``outputs/n41/source_replay``.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import sys
import traceback
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sam3_intermot.datasets.dancetrack import DanceTrackDataset
from scripts.n36_real_eval_common import DATA_ROOT, HORIZONS, atomic_json, finite_iou
from scripts.n38r1_sidecar_common import protocol_hash
from scripts.n41_stage2_run_source_replay import (
    PROTOCOL as RUNTIME_PROTOCOL,
    SOURCES,
    candidate_signature,
    finite_matrices,
    load_json,
)
from scripts.run_n36_replay import (
    BOOTSTRAP_REPLICATES,
    BOOTSTRAP_SEED,
    cluster_bootstrap,
    event_variant_summary,
    protected_regression,
)
from scripts.run_n37_replay import add_identity_error_aliases


N37_MANIFEST = ROOT / "outputs" / "n37" / "real_event_manifest.json"
FULL_MANIFEST = ROOT / "outputs" / "n41" / "source_replay" / "full_attempt1_manifest.json"
SOURCE_MANIFEST = ROOT / "outputs" / "n41" / "source_replay" / "source_embedding_manifest.json"
PROTOCOL_PATH = ROOT / "outputs" / "n41" / "source_replay" / "source_protocol.json"
OUT = ROOT / "outputs" / "n41" / "source_replay"
EVENT_OUT = OUT / "posthoc_event_results"
RESULT = OUT / "posthoc_source_results.json"
STAGE = ROOT / "outputs" / "n41" / "stage_03_status.json"
PROTOCOL = "N41_GT_CONTROLLED_APPEARANCE_SOURCE_ABLATION_POSTHOC_V1"
VARIANTS = ("M0", "M1", "M2", "M3", "M4")


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def finite(value: Any) -> bool:
    try:
        return bool(math.isfinite(float(value)))
    except (TypeError, ValueError):
        return False


def load_event_map() -> dict[str, dict[str, Any]]:
    payload = load_json(N37_MANIFEST)
    events = payload.get("events", [])
    if payload.get("status") != "PASS" or payload.get("event_count") != 24 or len(events) != 24:
        raise RuntimeError("N37 manifest is not frozen PASS/24")
    output = {}
    for item in events:
        event_id = str(item["event"]["event_id"])
        if event_id in output:
            raise RuntimeError(f"duplicate N37 event: {event_id}")
        output[event_id] = item
    if len({str(item["event"]["sequence"]) for item in output.values()}) != 21:
        raise RuntimeError("N37 independent sequence count is not 21")
    return output


def load_full_manifest(event_map: dict[str, dict[str, Any]]) -> tuple[dict[str, Any], dict[tuple[str, str, str], dict[str, Any]]]:
    manifest = load_json(FULL_MANIFEST)
    required = {
        "status": "FULL_RUNTIME_PASS",
        "event_count": 24,
        "source_count": 3,
        "configuration_count": 2,
        "worker_count": 144,
        "expected_worker_count": 144,
        "all_workers_returncode_zero": True,
        "all_artifact_audits_pass": True,
        "all_batch_candidate_stream_checks_pass": True,
        "runtime_future_gt_used": False,
        "gt_loaded_in_supervisor": False,
    }
    for key, expected in required.items():
        if manifest.get(key) != expected:
            raise RuntimeError(f"full runtime manifest gate failed: {key}={manifest.get(key)!r} expected {expected!r}")
    records = manifest.get("worker_manifests")
    if not isinstance(records, list) or len(records) != 144:
        raise RuntimeError("full runtime worker manifest is not exactly 144 rows")
    output: dict[tuple[str, str, str], dict[str, Any]] = {}
    for record in records:
        key = (str(record["event_id"]), str(record["source_id"]), str(record["config_id"]))
        if key in output:
            raise RuntimeError(f"duplicate full runtime worker key: {key}")
        if key[0] not in event_map:
            raise RuntimeError(f"worker event not in N37 manifest: {key}")
        if record.get("returncode") != 0 or record.get("artifact_audit", {}).get("status") != "PASS":
            raise RuntimeError(f"worker/audit not PASS: {key}")
        path = ROOT / str(record["output"])
        if not path.is_file():
            raise FileNotFoundError(path)
        output[key] = record
    return manifest, output


def load_source_info(event_id: str, source_id: str) -> dict[str, Any]:
    payload = load_json(SOURCE_MANIFEST)
    for entry in payload.get("events", []):
        if str(entry.get("event_id")) == event_id:
            value = entry.get("sources", {}).get(source_id)
            if isinstance(value, dict):
                return value
    raise KeyError(f"source sidecar entry missing: {event_id}/{source_id}")


def build_comparison(variant_payload: dict[str, Any]) -> list[dict[str, Any]]:
    left_trace = variant_payload["branches"]["memory_write=False"]["future_trace"]
    right_trace = variant_payload["branches"]["memory_write=True"]["future_trace"]
    if len(left_trace) != len(right_trace):
        raise RuntimeError("paired traces have different lengths")
    output = []
    for left, right in zip(left_trace, right_trace):
        left_audit = left["candidate_audit"]
        right_audit = right["candidate_audit"]
        left_scores = np.asarray(left_audit.get("fused_scores", []), dtype=float)
        right_scores = np.asarray(right_audit.get("fused_scores", []), dtype=float)
        aligned_deltas = []
        left_native = [int(value) for value in left_audit.get("candidate_native_ids", [])]
        right_native = [int(value) for value in right_audit.get("candidate_native_ids", [])]
        left_states = [int(value) for value in left_audit.get("public_id_order", [])]
        right_states = [int(value) for value in right_audit.get("public_id_order", [])]
        left_map = {(native, pid): float(left_scores[i, j]) for i, native in enumerate(left_native) for j, pid in enumerate(left_states) if i < left_scores.shape[0] and j < left_scores.shape[1]}
        right_map = {(native, pid): float(right_scores[i, j]) for i, native in enumerate(right_native) for j, pid in enumerate(right_states) if i < right_scores.shape[0] and j < right_scores.shape[1]}
        common = sorted(set(left_map) & set(right_map))
        aligned_deltas = [right_map[key] - left_map[key] for key in common]
        left_assignment = {
            int(native): (left_states[int(column)] if 0 <= int(column) < len(left_states) else None)
            for native, column in zip(left_native, left_audit.get("assignment_after_scope", left_audit.get("assignment", [])))
        }
        right_assignment = {
            int(native): (right_states[int(column)] if 0 <= int(column) < len(right_states) else None)
            for native, column in zip(right_native, right_audit.get("assignment_after_scope", right_audit.get("assignment", [])))
        }
        output.append({
            "frame": int(left["frame"]),
            "score_shape_equal": bool(left_scores.shape == right_scores.shape),
            "aligned_score_pair_count": len(common),
            "max_abs_score_delta": float(max(abs(value) for value in aligned_deltas)) if aligned_deltas else None,
            "score_changed": bool(any(abs(value) > 1e-12 for value in aligned_deltas)),
            "assignment_changed": bool(left_assignment != right_assignment),
            "no_write_assignment_by_native": left_assignment,
            "write_assignment_by_native": right_assignment,
            "candidate_stream_signature_equal": bool(candidate_signature(left_audit) == candidate_signature(right_audit)),
        })
    return output


def target_iou(trace_entry: dict[str, Any], event: dict[str, Any], gt_frames: dict[int, Any]) -> float | None:
    frame = int(trace_entry["frame"])
    gt = gt_frames.get(frame)
    if gt is None:
        return None
    target_gid = int(event["dataset_gt_id"])
    target_box = None
    for gid, box in zip(gt.gt_ids, gt.boxes):
        if int(gid) == target_gid:
            target_box = box
            break
    if target_box is None:
        return None
    target_pid = int(event["public_id"])
    return max(
        (finite_iou(row[1], target_box) for row in trace_entry.get("rows", []) if int(row[0]) == target_pid),
        default=0.0,
    )


def transition_diagnostics(
    comparison: list[dict[str, Any]],
    no_trace: list[dict[str, Any]],
    yes_trace: list[dict[str, Any]],
    event: dict[str, Any],
    gt_frames: dict[int, Any],
) -> dict[str, Any]:
    by_frame = {int(row["frame"]): row for row in comparison}
    output: dict[str, Any] = {}
    for horizon in HORIZONS:
        count = min(int(horizon), len(no_trace), len(yes_trace))
        rows = []
        for index in range(count):
            left, right = no_trace[index], yes_trace[index]
            frame = int(left["frame"])
            row = by_frame.get(frame, {})
            left_iou = target_iou(left, event, gt_frames)
            right_iou = target_iou(right, event, gt_frames)
            changed = bool(row.get("assignment_changed", False))
            correct = bool(changed and left_iou is not None and right_iou is not None and right_iou > left_iou + 1e-9)
            incorrect = bool(changed and left_iou is not None and right_iou is not None and right_iou < left_iou - 1e-9)
            rows.append({
                "frame": frame,
                "score_changed": bool(row.get("score_changed", False)),
                "assignment_changed": changed,
                "correct_assignment_change": correct,
                "incorrect_assignment_change": incorrect,
                "target_iou_no_write": left_iou,
                "target_iou_write": right_iou,
                "max_abs_score_delta": row.get("max_abs_score_delta"),
                "aligned_score_pair_count": row.get("aligned_score_pair_count", 0),
            })
        score_count = sum(int(row["score_changed"]) for row in rows)
        assignment_count = sum(int(row["assignment_changed"]) for row in rows)
        correct_count = sum(int(row["correct_assignment_change"]) for row in rows)
        incorrect_count = sum(int(row["incorrect_assignment_change"]) for row in rows)
        output[str(horizon)] = {
            "evaluated_future_frames": len(rows),
            "score_changed_count": score_count,
            "score_change_rate": float(score_count / len(rows)) if rows else None,
            "assignment_changed_count": assignment_count,
            "assignment_change_rate": float(assignment_count / len(rows)) if rows else None,
            "correct_assignment_change_count": correct_count,
            "incorrect_assignment_change_count": incorrect_count,
            "correct_change_rate_over_evaluated": float(correct_count / len(rows)) if rows else None,
            "incorrect_change_rate_over_evaluated": float(incorrect_count / len(rows)) if rows else None,
            "score_changed_without_assignment_change_count": sum(int(row["score_changed"] and not row["assignment_changed"]) for row in rows),
            "assignment_changed_without_aligned_score_count": sum(int(row["assignment_changed"] and row["aligned_score_pair_count"] == 0) for row in rows),
            "frame_details": rows,
        }
    return output


def run() -> dict[str, Any]:
    started = now()
    event_map = load_event_map()
    full_manifest, records = load_full_manifest(event_map)
    protocol = load_json(PROTOCOL_PATH)
    if protocol.get("status") != "FROZEN_BEFORE_SOURCE_GENERATION_AND_REPLAY":
        raise RuntimeError("source protocol not frozen")
    datasets = DanceTrackDataset(str(DATA_ROOT), sequences=sorted({str(item["event"]["sequence"]) for item in event_map.values()}), split="train")
    event_rows: dict[tuple[str, str, str], dict[str, Any]] = {}
    action_group: dict[tuple[str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for key in sorted(records):
        event_id, source_id, config_id = key
        record = records[key]
        artifact_path = ROOT / str(record["output"])
        artifact = load_json(artifact_path)
        if artifact.get("status") != "PASS":
            raise RuntimeError(f"worker artifact changed/not PASS: {artifact_path}")
        item = event_map[event_id]
        event = item["event"]
        gt_frames = datasets.load_gt(str(event["sequence"]))
        source_info = load_source_info(event_id, source_id)
        result = {
            "event_id": event_id,
            "sequence": str(event["sequence"]),
            "action_type": str(event["action_type"]),
            "event_frame": int(event["frame"]),
            "source_id": source_id,
            "source_role": source_info.get("role"),
            "source_feature_sha256": source_info.get("feature_sha256"),
            "config_id": config_id,
            "interaction_source": "simulated_from_gt",
            "not_real_human_evidence": True,
            "runtime_future_gt_used": False,
            "variants": {},
        }
        for variant in VARIANTS:
            variant_payload = artifact.get("variants", {}).get(variant)
            if not isinstance(variant_payload, dict) or variant_payload.get("status") != "PASS":
                raise RuntimeError(f"missing PASS variant: {artifact_path}/{variant}")
            comparison = build_comparison(variant_payload)
            replay = {
                "status": "PASS",
                "candidate_complete": True,
                "branches": variant_payload["branches"],
                "comparison": comparison,
            }
            summary = event_variant_summary(str(event["action_type"]), event, variant, replay, gt_frames)
            add_identity_error_aliases(summary)
            no_trace = variant_payload["branches"]["memory_write=False"]["future_trace"]
            yes_trace = variant_payload["branches"]["memory_write=True"]["future_trace"]
            summary["source_id"] = source_id
            summary["config_id"] = config_id
            summary["source_role"] = source_info.get("role")
            summary["source_feature_sha256"] = source_info.get("feature_sha256")
            summary["posthoc_gt_loaded_after_all_runtime_workers"] = True
            summary["transition_diagnostics"] = transition_diagnostics(comparison, no_trace, yes_trace, event, gt_frames)
            summary["runtime_boundary"] = {
                "runtime_future_gt_used": False,
                "gt_loaded_in_worker": False,
                "gt_used_here": "posthoc_metrics_and_assignment_direction_only",
            }
            result["variants"][variant] = summary
            action_group[(source_id, config_id, variant, str(event["action_type"]))].append(summary)
        event_rows[key] = result
        atomic_json(EVENT_OUT / event_id / source_id / f"{config_id}.json", result)
    groups: dict[str, Any] = {}
    for source_id in SOURCES:
        groups[source_id] = {}
        for config in protocol.get("weight_grid", []):
            config_id = str(config["config_id"])
            groups[source_id][config_id] = {}
            group_event_rows = [value for (event_id, sid, cid), value in event_rows.items() if sid == source_id and cid == config_id]
            for variant in VARIANTS:
                summaries = [value["variants"][variant] for value in group_event_rows]
                bootstrap = {str(h): cluster_bootstrap(summaries, h) for h in HORIZONS}
                regressions = [
                    protected_regression(
                        summary["no_write_metrics"],
                        summary["write_metrics"],
                        event_map[summary["event_id"]]["event"],
                        horizon=20,
                    )
                    for summary in summaries
                ]
                transition = {}
                for horizon in HORIZONS:
                    horizon_rows = [summary["transition_diagnostics"][str(horizon)] for summary in summaries]
                    total = sum(int(row["evaluated_future_frames"]) for row in horizon_rows)
                    fields = [
                        "score_changed_count",
                        "assignment_changed_count",
                        "correct_assignment_change_count",
                        "incorrect_assignment_change_count",
                        "score_changed_without_assignment_change_count",
                        "assignment_changed_without_aligned_score_count",
                    ]
                    counts = {field: sum(int(row[field]) for row in horizon_rows) for field in fields}
                    transition[str(horizon)] = {
                        **counts,
                        "evaluated_future_frames": total,
                        "score_change_rate": float(counts["score_changed_count"] / total) if total else None,
                        "assignment_change_rate": float(counts["assignment_changed_count"] / total) if total else None,
                        "correct_assignment_change_rate": float(counts["correct_assignment_change_count"] / total) if total else None,
                        "incorrect_assignment_change_rate": float(counts["incorrect_assignment_change_count"] / total) if total else None,
                    }
                action_counts = defaultdict(dict)
                for action in ("ADD_NEW_IDENTITY", "AUTHORITATIVE_REASSIGN", "ATOMIC_ID_SWAP", "RECOVER_IDENTITY"):
                    action_summaries = [row for row in summaries if row["action_type"] == action]
                    action_counts[action] = {
                        "event_count": len(action_summaries),
                        "sequence_count": len({row["sequence"] for row in action_summaries}),
                        "h20": {
                            "identity_utility_mean": float(np.mean([row["horizon_deltas"]["20"]["identity_utility_delta"] for row in action_summaries if finite(row["horizon_deltas"]["20"].get("identity_utility_delta"))])) if any(finite(row["horizon_deltas"]["20"].get("identity_utility_delta")) for row in action_summaries) else None,
                            "assignment_changed_count": sum(row["transition_diagnostics"]["20"]["assignment_changed_count"] for row in action_summaries),
                            "correct_assignment_change_count": sum(row["transition_diagnostics"]["20"]["correct_assignment_change_count"] for row in action_summaries),
                            "incorrect_assignment_change_count": sum(row["transition_diagnostics"]["20"]["incorrect_assignment_change_count"] for row in action_summaries),
                        },
                    }
                groups[source_id][config_id][variant] = {
                    "status": "PASS",
                    "event_count": len(summaries),
                    "independent_sequence_count": len({row["sequence"] for row in summaries}),
                    "action_counts": dict(action_counts),
                    "sequence_cluster_bootstrap": bootstrap,
                    "transition_diagnostics": transition,
                    "protected_regression": regressions,
                    "protected_no_obvious_regression": bool(regressions) and all(row["no_obvious_regression"] for row in regressions),
                    "future_effect_gate": {
                        "status": (
                            "PASS" if variant in ("M2", "M3", "M4") and all(
                            finite(bootstrap[str(h)].get("lower")) and float(bootstrap[str(h)]["lower"]) > 0.0
                            for h in (20, 50, 100)
                            ) and bool(regressions) and all(row["no_obvious_regression"] for row in regressions)
                            else ("FAIL_FUTURE_EFFECT" if variant in ("M2", "M3", "M4") else "NOT_A_GATE_VARIANT")
                        ),
                        "strict_requirement": "M2/M3/M4 lower CI strictly > 0 at H20/H50/H100; protected regression absent",
                        "not_used_for_source_or_config_selection": True,
                    },
                }
    payload = {
        "protocol": PROTOCOL,
        "status": "COMPLETED_POSTHOC_DIAGNOSTIC",
        "started_at": started,
        "finished_at": now(),
        "runtime_manifest": str(FULL_MANIFEST.relative_to(ROOT)),
        "runtime_manifest_sha256": sha256(FULL_MANIFEST),
        "source_manifest": str(SOURCE_MANIFEST.relative_to(ROOT)),
        "source_manifest_sha256": sha256(SOURCE_MANIFEST),
        "source_protocol_sha256": sha256(PROTOCOL_PATH),
        "frozen_n38_protocol_hash": protocol_hash(),
        "event_count": 24,
        "independent_sequence_count": 21,
        "source_count": 3,
        "configuration_count": 2,
        "posthoc_event_result_count": len(event_rows),
        "posthoc_variant_result_count": len(event_rows) * 5,
        "runtime_future_gt_used": False,
        "gt_loaded_only_after_runtime_scan": True,
        "interaction_source": "simulated_from_gt",
        "not_real_human_tape": True,
        "source_configurations": groups,
        "runtime_gate": {
            "full_runtime_status": full_manifest["status"],
            "worker_count": full_manifest["worker_count"],
            "all_runtime_checks_pass": True,
            "gt_loaded_in_supervisor": False,
        },
        "interpretation_contract": {
            "score_change_is_not_assignment_change": True,
            "assignment_change_is_not_correct_change": True,
            "correct_direction_is_posthoc_target_iou_improvement": True,
            "source_A_is_mechanism_upper_bound_not_real_human": True,
            "source_B_is_frozen_N37_simulated_from_gt_path": True,
            "source_C_is_controlled_corruption_not_real_human": True,
        },
        "event_results": [event_rows[key] for key in sorted(event_rows)],
    }
    atomic_json(RESULT, payload)
    stage = {
        "stage": "N41-02_POSTHOC",
        "status": "PASS_RUNTIME_COMPLETED_DIAGNOSTIC_GATE_PENDING_DECISION",
        "protocol": PROTOCOL,
        "result": str(RESULT.relative_to(ROOT)),
        "event_count": 24,
        "independent_sequence_count": 21,
        "source_count": 3,
        "configuration_count": 2,
        "runtime_worker_count": 144,
        "posthoc_event_result_count": len(event_rows),
        "posthoc_variant_result_count": len(event_rows) * 5,
        "runtime_future_gt_used": False,
        "gt_loaded_only_after_runtime_scan": True,
        "interaction_source": "simulated_from_gt",
        "real_human_tape_created": False,
        "all_runtime_checks_pass": True,
        "downstream_authorized": False,
        "next_action": "Review frozen source/action transition and sequence-cluster diagnostics; implement a new production fusion interface only if the preregistered N41-03 diagnostic criteria are supported.",
    }
    atomic_json(STAGE, stage)
    return stage


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=RESULT)
    args = parser.parse_args()
    try:
        result = run()
        print(json.dumps({"status": result["status"], "result": result["result"], "posthoc_event_result_count": result["posthoc_event_result_count"]}, sort_keys=True), flush=True)
    except Exception as exc:
        failure = OUT / "posthoc_attempt1_failure.json"
        atomic_json(failure, {
            "protocol": PROTOCOL,
            "status": "FAIL_POSTHOC",
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
            "runtime_future_gt_used": False,
            "artifact_is_failure_evidence": True,
        })
        raise


if __name__ == "__main__":
    main()
