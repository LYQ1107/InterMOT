#!/usr/bin/env python3
"""Post-hoc score the completed N39 weight scans.

This process is intentionally separate from ``n39_weight_worker.py``.  It is
called only after every runtime worker has produced a PASS artifact, and only
then imports DanceTrack GT.  No GT-derived value is sent back to a worker or
used to select an event/configuration.
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
from scripts.run_n36_replay import (
    BOOTSTRAP_REPLICATES,
    BOOTSTRAP_SEED,
    cluster_bootstrap,
    event_variant_summary,
    finite,
    protected_regression,
)
from scripts.run_n37_replay import add_identity_error_aliases


N37_MANIFEST = ROOT / "outputs/n37/real_event_manifest.json"
N37_FULL_LOOP = ROOT / "outputs/n37/full_loop_results.json"
N36_TAPE = ROOT / "outputs/n36/real_tape/tape_manifest.json"
OUT = ROOT / "outputs/n39"
PROTOCOL = "N39_WEIGHTED_ASSOCIATION_INTERFACE_PROBE_POSTHOC_V1"
VARIANTS = ("M0", "M1", "M2", "M3", "M4")
MODES = ("lambda_assoc", "human_weight")
EPS = 1.0e-9


def root_path(path: Path) -> Path:
    """Normalize CLI paths before relative-to-ROOT reporting."""
    value = Path(path)
    return value if value.is_absolute() else ROOT / value


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"expected object JSON: {path}")
    return value


def value_token(value: float) -> str:
    return f"{float(value):g}".replace("-", "m").replace(".", "p")


def config_id(mode: str, value: float) -> str:
    return f"{mode}_{value_token(value)}"


def assignment_public_vector(audit: dict[str, Any]) -> list[int | None]:
    public_ids = [int(value) for value in audit.get("public_id_order", [])]
    direct = audit.get("candidate_public_ids")
    if isinstance(direct, list) and len(direct) == len(audit.get("candidates", [])):
        return [None if value is None else int(value) for value in direct]
    assignments = audit.get("assignment_after_scope", audit.get("assignment", []))
    if not isinstance(assignments, list):
        return []
    output: list[int | None] = []
    for value in assignments:
        try:
            index = int(value)
        except (TypeError, ValueError):
            output.append(None)
            continue
        output.append(public_ids[index] if 0 <= index < len(public_ids) else None)
    return output


def aligned_fused_delta(no_audit: dict[str, Any], yes_audit: dict[str, Any]) -> tuple[float | None, bool]:
    """Align fused scores by public ID and candidate order, preserving axis changes."""
    no_states = [int(value) for value in no_audit.get("public_id_order", [])]
    yes_states = [int(value) for value in yes_audit.get("public_id_order", [])]
    no_matrix = np.asarray(no_audit.get("fused_scores", []), dtype=float)
    yes_matrix = np.asarray(yes_audit.get("fused_scores", []), dtype=float)
    no_candidates = [int(value) for value in no_audit.get("candidate_order", [])]
    yes_candidates = [int(value) for value in yes_audit.get("candidate_order", [])]
    if no_candidates != yes_candidates:
        return None, False
    common_states = [pid for pid in no_states if pid in set(yes_states)]
    if not common_states:
        return None, bool(no_states != yes_states)
    no_index = {pid: index for index, pid in enumerate(no_states)}
    yes_index = {pid: index for index, pid in enumerate(yes_states)}
    if no_matrix.ndim != 2 or yes_matrix.ndim != 2 or no_matrix.shape[0] != yes_matrix.shape[0]:
        return None, bool(no_states != yes_states)
    values = []
    for pid in common_states:
        left = no_matrix[:, no_index[pid]]
        right = yes_matrix[:, yes_index[pid]]
        if left.shape != right.shape or not np.all(np.isfinite(left)) or not np.all(np.isfinite(right)):
            return None, bool(no_states != yes_states)
        values.append(float(np.max(np.abs(right - left))) if left.size else 0.0)
    max_delta = max(values) if values else 0.0
    return float(max_delta), bool(no_states != yes_states)


def state_top_margin(audit: dict[str, Any], target_pid: int) -> tuple[float | None, float | None]:
    fused = np.asarray(audit.get("fused_scores", []), dtype=float)
    public_ids = [int(value) for value in audit.get("public_id_order", [])]
    if fused.ndim != 2 or target_pid not in public_ids:
        return None, None
    state_index = public_ids.index(int(target_pid))
    values = fused[:, state_index]
    if values.size < 2 or not np.all(np.isfinite(values)):
        return None, None
    ordered = np.sort(values)[::-1]
    top_margin = float(ordered[0] - ordered[1])
    target_row = None
    for index, pid in enumerate(assignment_public_vector(audit)):
        if pid == int(target_pid):
            target_row = index
            break
    assignment_margin = None
    if target_row is not None and fused.shape[1] > 1:
        alternatives = [index for index in range(fused.shape[1]) if index != state_index]
        if alternatives:
            assignment_margin = float(fused[target_row, state_index] - np.max(fused[target_row, alternatives]))
    return top_margin, assignment_margin


def target_iou(trace_entry: dict[str, Any], gt_frames: dict[int, Any], target_gid: int, target_pid: int) -> tuple[float | None, bool]:
    frame = int(trace_entry["frame"])
    gt_frame = gt_frames.get(frame)
    if gt_frame is None:
        return None, False
    target_box = None
    for gid, box in zip(gt_frame.gt_ids, gt_frame.boxes):
        if int(gid) == int(target_gid):
            target_box = np.asarray(box, dtype=float)
            break
    if target_box is None:
        return None, False
    rows = [row for row in trace_entry.get("rows", []) if int(row[0]) == int(target_pid)]
    value = max((finite_iou(row[1], target_box) for row in rows), default=0.0)
    return float(value), True


def transition_diagnostics(
    no_trace: list[dict[str, Any]],
    yes_trace: list[dict[str, Any]],
    event: dict[str, Any],
    gt_frames: dict[int, Any],
) -> dict[str, Any]:
    if [int(row["frame"]) for row in no_trace] != [int(row["frame"]) for row in yes_trace]:
        raise RuntimeError(f"paired frame mismatch for {event['event_id']}")
    target_pid = int(event["public_id"])
    target_gid = int(event["dataset_gt_id"])
    output: dict[str, Any] = {}
    for horizon in HORIZONS:
        frame_rows = []
        for no_entry, yes_entry in zip(no_trace[: int(horizon)], yes_trace[: int(horizon)]):
            no_audit = no_entry["candidate_audit"]
            yes_audit = yes_entry["candidate_audit"]
            score_delta, axes_changed = aligned_fused_delta(no_audit, yes_audit)
            no_assignment = assignment_public_vector(no_audit)
            yes_assignment = assignment_public_vector(yes_audit)
            assignment_changed = no_assignment != yes_assignment
            no_iou, visible = target_iou(no_entry, gt_frames, target_gid, target_pid)
            yes_iou, visible_yes = target_iou(yes_entry, gt_frames, target_gid, target_pid)
            if visible != visible_yes:
                raise RuntimeError(f"posthoc visibility mismatch at frame {no_entry['frame']}")
            correct_direction = None
            if assignment_changed and visible and no_iou is not None and yes_iou is not None:
                if yes_iou > no_iou + EPS:
                    correct_direction = "CORRECT_ASSIGNMENT_CHANGE"
                elif yes_iou < no_iou - EPS:
                    correct_direction = "INCORRECT_ASSIGNMENT_CHANGE"
                else:
                    correct_direction = "NEUTRAL_ASSIGNMENT_CHANGE"
            top_margin, assignment_margin = state_top_margin(yes_audit, target_pid)
            frame_rows.append(
                {
                    "frame": int(no_entry["frame"]),
                    "score_delta_max_abs_aligned": score_delta,
                    "score_changed": bool(score_delta is not None and score_delta > EPS),
                    "state_axis_changed": bool(axes_changed),
                    "assignment_changed": bool(assignment_changed),
                    "no_write_assignment_public_ids": no_assignment,
                    "write_assignment_public_ids": yes_assignment,
                    "target_iou_no_write": no_iou,
                    "target_iou_write": yes_iou,
                    "target_visible": bool(visible),
                    "correct_assignment_change": correct_direction,
                    "target_top1_top2_margin_write": top_margin,
                    "target_assignment_margin_write": assignment_margin,
                    "runtime_future_gt_used": bool(no_audit.get("runtime_future_gt_used", False) or yes_audit.get("runtime_future_gt_used", False)),
                }
            )
        finite_deltas = [row["score_delta_max_abs_aligned"] for row in frame_rows if finite(row["score_delta_max_abs_aligned"])]
        top_margins = [row["target_top1_top2_margin_write"] for row in frame_rows if finite(row["target_top1_top2_margin_write"])]
        assignment_margins = [row["target_assignment_margin_write"] for row in frame_rows if finite(row["target_assignment_margin_write"])]
        visible_rows = [row for row in frame_rows if row["target_visible"]]
        output[str(horizon)] = {
            "frame_count": len(frame_rows),
            "score_changed_frame_count": sum(row["score_changed"] for row in frame_rows),
            "score_change_rate": float(sum(row["score_changed"] for row in frame_rows) / len(frame_rows)) if frame_rows else None,
            "assignment_changed_frame_count": sum(row["assignment_changed"] for row in frame_rows),
            "assignment_change_rate": float(sum(row["assignment_changed"] for row in frame_rows) / len(frame_rows)) if frame_rows else None,
            "correct_assignment_change_count": sum(row["correct_assignment_change"] == "CORRECT_ASSIGNMENT_CHANGE" for row in frame_rows),
            "incorrect_assignment_change_count": sum(row["correct_assignment_change"] == "INCORRECT_ASSIGNMENT_CHANGE" for row in frame_rows),
            "neutral_assignment_change_count": sum(row["correct_assignment_change"] == "NEUTRAL_ASSIGNMENT_CHANGE" for row in frame_rows),
            "target_visible_frame_count": len(visible_rows),
            "mean_score_delta_max_abs_aligned": float(np.mean(finite_deltas)) if finite_deltas else None,
            "max_score_delta_max_abs_aligned": float(np.max(finite_deltas)) if finite_deltas else None,
            "mean_target_top1_top2_margin_write": float(np.mean(top_margins)) if top_margins else None,
            "mean_target_assignment_margin_write": float(np.mean(assignment_margins)) if assignment_margins else None,
            "state_axis_changed_frame_count": sum(row["state_axis_changed"] for row in frame_rows),
            "runtime_future_gt_true_count": sum(row["runtime_future_gt_used"] for row in frame_rows),
            "frames": frame_rows,
        }
    return output


def compact_config_result(
    raw: dict[str, Any],
    item: dict[str, Any],
    variant: str,
    gt_frames: dict[int, Any],
) -> dict[str, Any]:
    event = item["event"]
    replay = {
        "status": raw.get("status"),
        "candidate_complete": True,
        "branches": raw["variants"][variant]["branches"],
        "comparison": raw["variants"][variant].get("comparison", []),
    }
    summary = event_variant_summary(str(event["action_type"]), event, variant, replay, gt_frames)
    add_identity_error_aliases(summary)
    summary["posthoc_gt_loaded_after_runtime_artifact_completion"] = True
    summary["interaction_source"] = "simulated_from_gt"
    summary["transition_diagnostics"] = transition_diagnostics(
        replay["branches"]["memory_write=False"]["future_trace"],
        replay["branches"]["memory_write=True"]["future_trace"],
        event,
        gt_frames,
    )
    return summary


def run(manifest_path: Path, result_path: Path, stage_path: Path) -> dict[str, Any]:
    manifest_path = root_path(manifest_path)
    result_path = root_path(result_path)
    stage_path = root_path(stage_path)
    started = now()
    scan_manifest = load_json(manifest_path)
    if scan_manifest.get("status") != "PASS" or scan_manifest.get("phase") != "full":
        raise RuntimeError(f"full N39 runtime scan is not PASS: {manifest_path}")
    workers = scan_manifest.get("workers", [])
    if len(workers) != 336 or scan_manifest.get("worker_count") != 336:
        raise RuntimeError(f"expected 336 full-scan workers, found {len(workers)}")
    event_payload = load_json(N37_MANIFEST)
    if event_payload.get("status") != "PASS" or event_payload.get("event_count") != 24:
        raise RuntimeError("N37 event manifest is not frozen PASS/24")
    event_items = {str(item["event"]["event_id"]): item for item in event_payload["events"]}
    expected_keys = {(mode, float(value), event_id) for mode in MODES for value in (0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0) for event_id in event_items}
    seen_keys = set()
    for record in workers:
        key = (str(record["mode"]), float(record["value"]), str(record["event_id"]))
        if key in seen_keys:
            raise RuntimeError(f"duplicate worker key: {key}")
        seen_keys.add(key)
        if record.get("returncode") != 0:
            raise RuntimeError(f"nonzero worker returncode: {record}")
    if seen_keys != expected_keys:
        raise RuntimeError(f"worker key coverage mismatch missing={len(expected_keys-seen_keys)} extra={len(seen_keys-expected_keys)}")

    sequences = sorted({str(item["event"]["sequence"]) for item in event_items.values()})
    dataset = DanceTrackDataset(str(DATA_ROOT), sequences=sequences, split="train")
    gt_cache: dict[str, dict[int, Any]] = {}
    by_config: dict[str, dict[str, Any]] = {}
    raw_by_key: dict[tuple[str, float, str], dict[str, Any]] = {}
    for record in workers:
        raw_path = ROOT / str(record["output"])
        raw = load_json(raw_path)
        if raw.get("status") != "PASS":
            raise RuntimeError(f"worker artifact is not PASS: {raw_path}")
        key = (str(record["mode"]), float(record["value"]), str(record["event_id"]))
        raw_by_key[key] = raw

    # Runtime workers are all complete before the first GT load in this loop.
    for mode in MODES:
        for value in (0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0):
            cid = config_id(mode, value)
            event_rows: list[dict[str, Any]] = []
            variant_rows: dict[str, list[dict[str, Any]]] = {name: [] for name in VARIANTS}
            action_counts: dict[str, int] = defaultdict(int)
            errors = []
            for event_id, item in sorted(event_items.items()):
                event = item["event"]
                key = (mode, float(value), event_id)
                raw = raw_by_key[key]
                sequence = str(event["sequence"])
                if sequence not in gt_cache:
                    gt_cache[sequence] = dataset.load_gt(sequence)
                gt_frames = gt_cache[sequence]
                action_row = {
                    "event_id": event_id,
                    "sequence": sequence,
                    "action_type": str(event["action_type"]),
                    "event_frame": int(event["frame"]),
                    "status": "PASS",
                    "variants": {},
                }
                action_counts[str(event["action_type"])] += 1
                for variant in VARIANTS:
                    try:
                        summary = compact_config_result(raw, item, variant, gt_frames)
                        action_row["variants"][variant] = summary
                        variant_rows[variant].append(summary)
                    except Exception as exc:
                        failure = {
                            "event_id": event_id,
                            "variant": variant,
                            "status": "FAIL_POSTHOC",
                            "error": f"{type(exc).__name__}: {exc}",
                            "traceback": traceback.format_exc(),
                        }
                        errors.append(failure)
                        action_row["variants"][variant] = failure
                        action_row["status"] = "FAIL"
                event_rows.append(action_row)
            bootstrap = {
                variant: {str(horizon): cluster_bootstrap(rows, horizon) for horizon in HORIZONS}
                for variant, rows in variant_rows.items()
            }
            gate_checks = {}
            for variant in VARIANTS:
                regression_rows = []
                for event_row in event_rows:
                    summary = event_row["variants"].get(variant, {})
                    if summary.get("status") != "PASS":
                        continue
                    regression_rows.append(
                        protected_regression(
                            summary["no_write_metrics"],
                            summary["write_metrics"],
                            event_items[event_row["event_id"]]["event"],
                            horizon=20,
                        )
                    )
                gate_checks[variant] = {
                    "sequence_cluster_h20_lower_ci": bootstrap[variant]["20"].get("lower"),
                    "sequence_cluster_h50_lower_ci": bootstrap[variant]["50"].get("lower"),
                    "sequence_cluster_h100_lower_ci": bootstrap[variant]["100"].get("lower"),
                    "protected_regression": regression_rows,
                    "protected_no_obvious_regression": bool(regression_rows) and all(row["no_obvious_regression"] for row in regression_rows),
                    "score_change_rate_h20_mean": float(np.mean([row["transition_diagnostics"]["20"]["score_change_rate"] for row in variant_rows[variant] if finite(row["transition_diagnostics"]["20"]["score_change_rate"])])) if any(finite(row["transition_diagnostics"]["20"]["score_change_rate"]) for row in variant_rows[variant]) else None,
                    "assignment_change_rate_h20_mean": float(np.mean([row["transition_diagnostics"]["20"]["assignment_change_rate"] for row in variant_rows[variant] if finite(row["transition_diagnostics"]["20"]["assignment_change_rate"])])) if any(finite(row["transition_diagnostics"]["20"]["assignment_change_rate"]) for row in variant_rows[variant]) else None,
                    "correct_assignment_change_count_h20": sum(row["transition_diagnostics"]["20"]["correct_assignment_change_count"] for row in variant_rows[variant]),
                    "incorrect_assignment_change_count_h20": sum(row["transition_diagnostics"]["20"]["incorrect_assignment_change_count"] for row in variant_rows[variant]),
                }
            complete = not errors and all(len(row["variants"]) == 5 and row["status"] == "PASS" for row in event_rows)
            checks = {
                "runtime_worker_scan_complete": True,
                "all_24_events_posthoc_scored": len(event_rows) == 24,
                "all_five_variants_per_event": complete,
                "runtime_future_gt_used_false": all(
                    summary.get("runtime_future_gt_used") is False
                    and all(
                        row.get("runtime_future_gt_true_count", 0) == 0
                        for horizon in HORIZONS
                        for row in summary.get("transition_diagnostics", {}).values()
                    )
                    for event_row in event_rows
                    for summary in event_row["variants"].values()
                    if summary.get("status") == "PASS"
                ),
                "no_duplicate_or_missing_event_keys": len(event_rows) == 24 and len({row["event_id"] for row in event_rows}) == 24,
            }
            for variant in ("M2", "M3", "M4"):
                checks[f"{variant}_h20_lower_ci_gt_zero"] = bool(
                    finite(gate_checks[variant]["sequence_cluster_h20_lower_ci"])
                    and gate_checks[variant]["sequence_cluster_h20_lower_ci"] > 0.0
                )
                checks[f"{variant}_protected_no_obvious_regression"] = gate_checks[variant]["protected_no_obvious_regression"]
            gate_pass = bool(complete and all(checks.values()))
            by_config[cid] = {
                "mode": mode,
                "value": float(value),
                "status": "PASS" if complete else "FAIL_POSTHOC",
                "event_count": len(event_rows),
                "independent_sequence_count": len({row["sequence"] for row in event_rows}),
                "action_counts": dict(action_counts),
                "events": event_rows,
                "sequence_cluster_bootstrap": bootstrap,
                "gate_checks": gate_checks,
                "future_effect_gate": {
                    "status": "PASS" if gate_pass else "NOT_AUTHORIZED",
                    "checks": checks,
                    "strict_h20_lower_ci_requirement": "M2/M3/M4 lower CI strictly > 0",
                    "cluster_unit": "independent sequence",
                    "seed": BOOTSTRAP_SEED,
                    "replicates": BOOTSTRAP_REPLICATES,
                },
            }

    full_loop = load_json(N37_FULL_LOOP)
    upstream = {
        "n37_event_manifest_pass_24": True,
        "n37_full_loop_pass": full_loop.get("status") == "PASS" and full_loop.get("event_pass_count") == full_loop.get("event_count"),
        "n36_real_tape_pass": load_json(N36_TAPE).get("status") == "PASS",
        "runtime_workers_pass_336": len(workers) == 336,
    }
    config_gates = {cid: result["future_effect_gate"] for cid, result in by_config.items()}
    all_config_gates = bool(upstream["n37_full_loop_pass"] and upstream["n36_real_tape_pass"] and all(item["status"] == "PASS" for item in config_gates.values()))
    payload = {
        "protocol": PROTOCOL,
        "status": "PASS" if all_config_gates else "COMPLETED_GATE_FAILED",
        "started_at": started,
        "finished_at": now(),
        "runtime_scan_manifest": str(manifest_path.relative_to(ROOT)),
        "runtime_future_gt_used": False,
        "gt_loaded_only_after_runtime_scan": True,
        "event_count": 24,
        "independent_sequence_count": 21,
        "configuration_count": len(by_config),
        "configuration_ids": sorted(by_config),
        "upstream_checks": upstream,
        "configurations": by_config,
        "global_future_effect_gate": {
            "status": "PASS" if all_config_gates else "FAIL",
            "all_preregistered_configurations_must_pass": True,
            "any_configuration_selected_by_result": False,
            "calibration_head": "AUTHORIZED" if all_config_gates else "NOT_AUTHORIZED",
            "selector": "NOT_AUTHORIZED",
            "decoder_lora": "AUTHORIZED_PILOT_ONLY" if all_config_gates else "NOT_AUTHORIZED",
        },
        "interpretation": {
            "assignment_change_is_distinct_from_score_change": True,
            "correct_direction_is_posthoc_target_iou_improvement_only": True,
            "interaction_source": "simulated_from_gt",
            "not_historical_human_clicks": True,
        },
        "artifacts": {
            "result": str(result_path.relative_to(ROOT)),
            "runtime_scan_manifest": str(manifest_path.relative_to(ROOT)),
            "scale_audit": "outputs/n39/scale_audit_summary.json",
        },
    }
    atomic_json(result_path, payload)
    stage = {
        "stage": "N39-03",
        "status": payload["status"],
        "protocol": PROTOCOL,
        "result": str(result_path.relative_to(ROOT)),
        "event_count": 24,
        "independent_sequence_count": 21,
        "configuration_count": len(by_config),
        "upstream_checks": upstream,
        "global_future_effect_gate": payload["global_future_effect_gate"],
        "runtime_future_gt_used": False,
        "gt_loaded_only_after_runtime_scan": True,
        "downstream_authorized": all_config_gates,
        "next_action": "Do not train calibration/selector/LoRA; write final N39 report with weight-interface diagnosis." if not all_config_gates else "Run only the separately gated calibration pilot protocol.",
    }
    atomic_json(stage_path, stage)
    return stage


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--result", type=Path, default=OUT / "weight_scan_results.json")
    parser.add_argument("--stage", type=Path, default=OUT / "stage_03_status.json")
    args = parser.parse_args()
    try:
        result = run(root_path(args.manifest), root_path(args.result), root_path(args.stage))
        print(json.dumps({"status": result["status"], "result": result["result"], "configuration_count": result["configuration_count"]}, sort_keys=True), flush=True)
    except Exception as exc:
        failure_path = OUT / "attempts" / "stage_03_posthoc_failure.json"
        atomic_json(failure_path, {"protocol": PROTOCOL, "status": "FAIL", "error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc(), "artifact_is_failure_evidence": True})
        atomic_json(root_path(args.stage), {"stage": "N39-03", "status": "FAIL", "protocol": PROTOCOL, "failure_artifact": str(failure_path.relative_to(ROOT)), "downstream_authorized": False})
        raise


if __name__ == "__main__":
    main()
