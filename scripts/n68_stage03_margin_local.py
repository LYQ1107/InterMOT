"""N68 Stage 03: one isolated no-trimming margin-aware local interface.

This is a separately frozen ablation after the first N68 local head failed its
strict future-effect gate.  It retains the trained target-conditioned head,
but changes only how its candidate confidence is projected into the known
target public-ID column.  Candidate generation, public-ID axes, score rows,
and the global Hungarian solver remain frozen.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

# Keep both ``python scripts/file.py`` and ``python -m scripts.file`` entry
# points reproducible without changing any production package import path.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.n68_stage02_local_association import (
    ATTEMPTS,
    CHECKPOINT,
    EVENT_COUNT,
    FEATURE_NAMES,
    HORIZONS,
    N37_EVENTS,
    N42_PROTOCOL,
    N54_RUNTIME,
    PROTOCOL as STAGE02_PROTOCOL,
    REPLAY_STATUS as STAGE02_REPLAY_STATUS,
    VARIANTS,
    assignment_from_scores,
    atomic_json,
    branch_summary,
    bootstrap_ci,
    candidate_signature,
    finite_rank,
    frame_outcome,
    feature_matrix,
    load_event_map,
    load_json,
    load_sequence_split,
    load_trained_model,
    normalize_assignment,
    score_tanh,
    sha256_file,
    sigmoid,
    source_path,
    summarize_outcomes,
    target_physical_row,
    validate_frame_structure,
)


OUT = ROOT / "outputs/n68"
REPLAY = OUT / "replay"
ARTIFACTS = REPLAY / "stage03_event_artifacts"
PROTOCOL = OUT / "stage_03_protocol.json"
RUNTIME_STATUS = REPLAY / "stage03_runtime_status.json"
RESULTS = REPLAY / "stage03_paired_replay_results.json"
SCORE_STATUS = REPLAY / "stage03_posthoc_score_status.json"
STAGE03 = OUT / "stage_03_status.json"

MODE_BASELINE = "CURRENT_CCAM_BASELINE"
MODE_LEARNED = "LEARNED_LOCAL_ASSOCIATION"
MODE_MARGIN = "LEARNED_MARGIN_AWARE_COLUMN"
STAGE3_MODES = (MODE_BASELINE, MODE_LEARNED, MODE_MARGIN)
BOOTSTRAP_SEED = 6818
BOOTSTRAP_REPS = 2000


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def protocol_payload() -> dict[str, Any]:
    split = load_sequence_split()
    grouped: dict[str, list[str]] = {"train": [], "validation": [], "holdout": []}
    for sequence, name in split.items():
        grouped[name].append(sequence)
    for value in grouped.values():
        value.sort()
    return {
        "schema": "N68_STAGE_03_MARGIN_AWARE_LOCAL_PROTOCOL_V1",
        "status": "FROZEN_BEFORE_REPLAY",
        "created_at_utc": now(),
        "parent": {"stage_02_protocol": str(STAGE02_PROTOCOL), "stage_02_protocol_sha256": sha256_file(STAGE02_PROTOCOL), "stage_02_checkpoint": str(CHECKPOINT), "stage_02_checkpoint_sha256": sha256_file(CHECKPOINT)},
        "branch": "LEARNED_MARGIN_AWARE_COLUMN",
        "motivation": "Stage 02 score changes rarely crossed the global assignment boundary; test a fixed margin-aware projection without causal trimming or a solver change.",
        "formula": {
            "model_logit": "stage_02 learned targetness logit for each candidate row",
            "probability": "sigmoid(model_logit)",
            "row_best": "max finite current write_baseline score across public-ID columns for that candidate row",
            "gap": "max(0, row_best - current target-column score)",
            "residual_scale": "clip(0.5 + gap, 0.5, residual_bound)",
            "residual": "clip((2*probability-1)*residual_scale, -residual_bound, residual_bound)",
            "none": "if max probability < 0.5, use residual=-0.5 for every finite target-column cell",
            "residual_bound": 2.0,
            "none_threshold": 0.5,
            "application": "target public-ID column only; hard-negative cells are unchanged",
        },
        "variants": {"upstream": list(VARIANTS), "sidecar_modes": list(STAGE3_MODES)},
        "split": grouped,
        "frozen_constraints": {"checkpoint_changed": False, "candidate_generation_changed": False, "hungarian_solver_changed": False, "public_id_axis_changed": False, "future_gt_runtime": False, "causal_trimming": False, "production_authorized": False},
        "provenance": {"interaction_source": "simulated_from_gt", "not_real_human_evidence": True, "real_human_tape": False, "real_sam3_full_loop": False, "runtime_future_gt_used": False},
    }


def ensure_protocol() -> dict[str, Any]:
    payload = protocol_payload()
    if PROTOCOL.is_file():
        existing = load_json(PROTOCOL)
        for key, value in payload.items():
            if key == "created_at_utc":
                continue
            if existing.get(key) != value:
                raise RuntimeError(f"frozen Stage 03 protocol differs at {key}")
        return existing
    atomic_json(PROTOCOL, payload)
    return payload


def stage3_sidecar(mode: str, x_raw: np.ndarray, base: np.ndarray, target_col: int | None, model: Any, mean: np.ndarray, std: np.ndarray, device: Any) -> dict[str, Any]:
    import torch

    n, _p = base.shape
    if mode == MODE_BASELINE:
        logits = np.zeros(n, dtype=np.float32)
        probs = np.zeros(n, dtype=np.float32)
        residual = np.zeros(n, dtype=np.float32)
        none = False
        reason = "baseline_no_local_residual"
    else:
        with torch.no_grad():
            logits = model(torch.as_tensor((x_raw - mean) / std, dtype=torch.float32, device=device)).squeeze(-1).detach().cpu().numpy().astype(np.float32)
        probs = sigmoid(logits).astype(np.float32)
        if mode == MODE_LEARNED:
            none = bool(probs.size == 0 or float(np.max(probs)) < 0.5)
            residual = np.full(n, -0.5, dtype=np.float32) if none else (2.0 * np.tanh(logits)).astype(np.float32)
            reason = "stage02_target_conditioned_residual" if not none else "stage02_explicit_none"
        elif mode == MODE_MARGIN:
            none = bool(probs.size == 0 or float(np.max(probs)) < 0.5)
            if none:
                residual = np.full(n, -0.5, dtype=np.float32)
            else:
                finite_rows = np.where(np.isfinite(base) & (base > -1.0e8), base, -1.0e8)
                row_best = np.max(finite_rows, axis=1)
                target_values = base[:, target_col] if target_col is not None else np.zeros(n, dtype=np.float32)
                gaps = np.maximum(0.0, row_best - target_values)
                scale = np.clip(0.5 + gaps, 0.5, 2.0)
                residual = np.clip((2.0 * probs - 1.0) * scale, -2.0, 2.0).astype(np.float32)
            reason = "fixed_margin_aware_target_column_projection" if not none else "margin_branch_explicit_none"
        else:
            raise RuntimeError(f"unknown Stage 03 mode {mode}")
    adjusted = base.copy()
    if target_col is not None and mode != MODE_BASELINE:
        for row in range(n):
            if base[row, target_col] > -1.0e8 and np.isfinite(base[row, target_col]):
                adjusted[row, target_col] += residual[row]
    if not np.all(np.isfinite(adjusted)):
        raise RuntimeError("Stage 03 produced nonfinite adjusted scores")
    return {"logits": logits.astype(float).tolist(), "probabilities": probs.astype(float).tolist(), "residual_target_column": residual.astype(float).tolist(), "target_column": target_col, "abstained_none": none, "reason": reason, "adjusted_scores": adjusted.astype(float).tolist(), "score_cells_changed": int(np.sum(np.abs(adjusted - base) > 1.0e-12)), "runtime_future_gt_used": False}


def replay(device_name: str = "cpu") -> dict[str, Any]:
    ensure_protocol()
    events = load_event_map()
    model, mean, std, device = load_trained_model(device_name)
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    completed = 0
    frame_count = 0
    for event_id in sorted(events):
        event = events[event_id]
        source = load_json(source_path(event_id))
        artifact = {"schema": "N68_STAGE_03_MARGIN_AWARE_RUNTIME_EVENT_V1", "status": "PASS", "created_at_utc": now(), "event_id": event_id, "sequence": event["sequence"], "event_frame": event["event_frame"], "action_type": event["event"].get("action_type"), "target_public_id_event_input": event["target_public_id"], "target_native_tid_posthoc_label_only": event["target_native_tid"], "interaction_source": "simulated_from_gt", "not_real_human_evidence": True, "real_human_tape": False, "real_sam3_full_loop": False, "checkpoint": str(CHECKPOINT), "checkpoint_sha256": sha256_file(CHECKPOINT), "runtime_boundary": {"runtime_future_gt_used": False, "gt_loaded_in_worker": False, "target_native_tid_used_for_runtime_features": False, "future_gt_fields_sent": []}, "variants": {}}
        for variant in VARIANTS:
            frames_out = []
            for raw in source["variants"][variant]["frames"]:
                frame = dict(raw); frame["variant"] = variant
                write, base, _cf, _mv, _valid = validate_frame_structure(frame, variant, event)
                x_raw, feature_audit = feature_matrix(frame, event, event["target_public_id"])
                pids = [int(value) for value in write["public_id_order"]]
                target_col = pids.index(event["target_public_id"]) if event["target_public_id"] in pids else None
                source_assignment = normalize_assignment(write["assignment_columns"], len(write["candidate_rows"]), len(pids))
                methods = {}
                for mode in STAGE3_MODES:
                    sidecar = stage3_sidecar(mode, x_raw, base, target_col, model, mean, std, device)
                    adjusted = np.asarray(sidecar["adjusted_scores"], dtype=np.float32)
                    assignment = source_assignment.copy() if mode == MODE_BASELINE else assignment_from_scores(adjusted)
                    methods[mode] = {"sidecar": sidecar, "assignment": branch_summary(write, assignment, adjusted, pids, mode), "assignment_recomputed_from_adjusted_scores": True, "runtime_future_gt_used": False}
                frames_out.append({"frame": int(frame["frame"]), "feature_audit": feature_audit, "candidate_feature_sha256": feature_audit["candidate_feature_sha256"], "methods": methods, "candidate_stream_same_across_methods": True, "public_id_axis_same_across_methods": True, "memory_current_frame_write_hidden": int(frame["frame"]) == event["event_frame"], "first_event_memory_visible_frame": event["event_frame"] + 1, "runtime_future_gt_used": False})
                frame_count += 1
            artifact["variants"][variant] = {"frame_count": len(frames_out), "frames": frames_out}
        atomic_json(ARTIFACTS / f"{event_id}.json", artifact)
        completed += 1
        print(json.dumps({"replayed_events": completed, "event_id": event_id, "frames": frame_count}, sort_keys=True), flush=True)
    status = {"schema": "N68_STAGE_03_MARGIN_AWARE_RUNTIME_STATUS_V1", "status": "PASS_RUNTIME_REPLAY", "created_at_utc": now(), "protocol": str(PROTOCOL), "protocol_sha256": sha256_file(PROTOCOL), "checkpoint": str(CHECKPOINT), "checkpoint_sha256": sha256_file(CHECKPOINT), "outputs": {"event_artifacts": str(ARTIFACTS)}, "metrics": {"event_count": completed, "frames": frame_count, "expected_frames": EVENT_COUNT * len(VARIANTS) * 100}, "gate_checks": {"all_24_events": completed == EVENT_COUNT, "all_5_variants": True, "all_100_frames": frame_count == EVENT_COUNT * len(VARIANTS) * 100, "same_candidate_stream": True, "same_public_id_axis": True, "same_hungarian_solver": True, "target_column_only": True, "runtime_future_gt_false": True, "gt_loaded_in_worker": False, "production_authorized": False}, "interaction_source": "simulated_from_gt", "not_real_human_evidence": True, "real_human_tape": False, "real_sam3_full_loop": False, "runtime_future_gt_used": False}
    atomic_json(RUNTIME_STATUS, status)
    return status


def score_replay() -> dict[str, Any]:
    status = load_json(RUNTIME_STATUS)
    if status.get("status") != "PASS_RUNTIME_REPLAY":
        raise RuntimeError("Stage 03 runtime replay is not PASS")
    events = load_event_map()
    paths = sorted(ARTIFACTS.glob("*.json"))
    if len(paths) != EVENT_COUNT:
        raise RuntimeError(f"Stage 03 expected {EVENT_COUNT} artifacts, found {len(paths)}")
    all_outcomes: list[tuple[str, str, str, dict[str, Any]]] = []
    for path in paths:
        artifact = load_json(path)
        event_id = artifact["event_id"]
        if event_id not in events:
            raise RuntimeError(f"unknown Stage 03 event {event_id}")
        event = events[event_id]
        for variant in VARIANTS:
            frames = artifact["variants"][variant]["frames"]
            if len(frames) != 100:
                raise RuntimeError(f"Stage 03 frame denominator {event_id}/{variant}")
            for frame in frames:
                if frame.get("runtime_future_gt_used") is not False:
                    raise RuntimeError(f"Stage 03 frame GT boundary {event_id}/{variant}/{frame['frame']}")
                baseline = frame["methods"][MODE_BASELINE]
                for mode in STAGE3_MODES:
                    outcome = frame_outcome(frame, mode, baseline, event)
                    outcome.update({"event_id": event_id, "sequence": event["sequence"], "action_type": event["event"].get("action_type"), "variant": variant, "method": mode})
                    all_outcomes.append((event["sequence"], str(event["event"].get("action_type")), variant, outcome))
    methods = {mode: summarize_outcomes(all_outcomes, mode) for mode in STAGE3_MODES}
    by_action: dict[str, Any] = {}
    for action in sorted({item[1] for item in all_outcomes}):
        subset = [item for item in all_outcomes if item[1] == action]
        by_action[action] = {mode: summarize_outcomes(subset, mode) for mode in STAGE3_MODES}
    by_variant: dict[str, Any] = {}
    for variant in VARIANTS:
        subset = [item for item in all_outcomes if item[2] == variant]
        by_variant[variant] = {mode: summarize_outcomes(subset, mode) for mode in STAGE3_MODES}
    by_sequence: dict[str, Any] = {}
    for sequence in sorted({item[0] for item in all_outcomes}):
        subset = [item for item in all_outcomes if item[0] == sequence]
        by_sequence[sequence] = {mode: summarize_outcomes(subset, mode) for mode in STAGE3_MODES}
    gate_by_mode: dict[str, Any] = {MODE_BASELINE: {"future_effect": False, "reason": "reference_baseline"}}
    for mode in (MODE_LEARNED, MODE_MARGIN):
        summary = methods[mode]
        lower = {str(h): summary["horizons"][str(h)]["sequence_cluster_bootstrap"]["ci95"][0] for h in HORIZONS}
        gate_by_mode[mode] = {"future_effect": bool(all(value is not None and value > 0.0 for value in lower.values()) and summary["correct_changes"] > summary["incorrect_changes"] and summary["untouched_regression_frame_rate"] == 0.0), "strict_lower_ci_by_horizon": lower, "correct_changes_gt_incorrect_changes": summary["correct_changes"] > summary["incorrect_changes"], "untouched_regression_safe": summary["untouched_regression_frame_rate"] == 0.0, "production_authorized": False, "real_human_tape": False}
    result = {"schema": "N68_STAGE_03_MARGIN_AWARE_PAIRED_RESULTS_V1", "status": "N68_SIMULATED_FUTURE_EFFECT_EVALUATED", "created_at_utc": now(), "protocol": str(PROTOCOL), "protocol_sha256": sha256_file(PROTOCOL), "runtime_status": str(RUNTIME_STATUS), "event_count": EVENT_COUNT, "variant_count": len(VARIANTS), "frame_count": len(all_outcomes), "interaction_source": "simulated_from_gt", "not_real_human_evidence": True, "real_human_tape": False, "real_sam3_full_loop": False, "runtime_future_gt_used": False, "gt_loaded_only_posthoc": True, "evaluation_boundary": {"candidate_generation_changed": False, "hungarian_solver_changed": False, "same_candidate_stream": True, "same_public_id_axis": True, "target_column_only": True, "future_iou_available": False, "future_idsw_available": False}, "methods": methods, "by_action_type": by_action, "by_upstream_variant": by_variant, "by_sequence": by_sequence, "gate_by_mode": gate_by_mode, "strict_future_effect_gate": {"status": "PASS" if any(gate_by_mode[mode].get("future_effect") for mode in (MODE_LEARNED, MODE_MARGIN)) else "FAIL_FUTURE_EFFECT", "calibration_authorized": False, "selector_authorized": False, "decoder_lora_authorized": False, "production_authorized": False}, "failure_root_cause": "Stage 03 tests one pre-registered margin-aware target-column projection after Stage 02. It is still a simulated_from_gt diagnostic and cannot authorize production without real human evidence and strict independent-sequence future effect.", "outputs": {"event_artifacts": str(ARTIFACTS), "paired_results": str(RESULTS)}}
    atomic_json(RESULTS, result)
    atomic_json(SCORE_STATUS, {"schema": "N68_STAGE_03_MARGIN_AWARE_POSTHOC_STATUS_V1", "status": "PASS_POSTHOC_SCORED_STRICT_GATE_REPORTED", "created_at_utc": now(), "paired_results": str(RESULTS), "event_count": EVENT_COUNT, "frame_outcome_count": len(all_outcomes), "runtime_future_gt_used": False, "gt_loaded_only_posthoc": True, "production_authorized": False})
    atomic_json(STAGE03, {"schema": "N68_STAGE_03_STATUS_V1", "status": "PASS_NO_TRIMMING_STRICT_GATE" if any(value.get("future_effect") for key, value in gate_by_mode.items() if key != MODE_BASELINE) else "FAIL_NO_TRIMMING_BRANCHES_STRICT_FUTURE_EFFECT", "created_at_utc": now(), "protocol": str(PROTOCOL), "protocol_sha256": sha256_file(PROTOCOL), "stage_02_results": str(OUT / "replay/paired_replay_results.json"), "stage_03_results": str(RESULTS), "gate_by_mode": gate_by_mode, "failure_root_cause": result["failure_root_cause"], "next_action": "Do not run TACT, calibration, selector, or LoRA unless a no-trimming branch passes the strict gate; current evidence remains simulated and mapping/scope limited.", "interaction_source": "simulated_from_gt", "not_real_human_evidence": True, "real_human_tape": False, "real_sam3_full_loop": False, "runtime_future_gt_used": False, "production_authorized": False})
    return result


def record_failure(stage: str, exc: BaseException) -> None:
    ATTEMPTS.mkdir(parents=True, exist_ok=True)
    existing = sorted(ATTEMPTS.glob(f"{stage}_failure_attempt*.json"))
    atomic_json(ATTEMPTS / f"{stage}_failure_attempt{len(existing)+1}.json", {"schema": "N68_STAGE_FAILURE_V1", "status": "FAIL_PRESERVED", "created_at_utc": now(), "stage": stage, "failure_root_cause": f"{type(exc).__name__}: {exc}", "protocol": str(PROTOCOL), "runtime_future_gt_used": False, "interaction_source": "simulated_from_gt", "not_real_human_evidence": True, "production_authorized": False, "next_action": "Preserve this failure and repair only the first actionable root cause."})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", required=True, choices=("protocol", "replay", "score"))
    parser.add_argument("--device", default="cpu", choices=("cpu", "cuda"))
    args = parser.parse_args()
    if args.mode == "protocol":
        ensure_protocol(); print(json.dumps({"status": "PASS_PROTOCOL_FROZEN", "path": str(PROTOCOL), "sha256": sha256_file(PROTOCOL)}, sort_keys=True))
    elif args.mode == "replay":
        print(json.dumps(replay(args.device), sort_keys=True))
    else:
        result = score_replay(); print(json.dumps({"status": result["strict_future_effect_gate"]["status"], "paired_results": str(RESULTS), "gate_by_mode": result["gate_by_mode"]}, sort_keys=True))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        record_failure("stage_03_margin_local", exc)
        raise
