#!/usr/bin/env python3
"""Run one isolated N42 T0/T1 frozen-candidate association replay worker.

The worker reuses a single frozen N41 event artifact and recomputes only the
Hungarian score/assignment interface for T1.  It never imports DanceTrack GT,
never changes the candidate stream, and never modifies a production module.
The direct public ID comes from the frozen human-event protocol; it is not
inferred from a candidate or from future labels.
"""

from __future__ import annotations

import argparse
import copy
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

from sam3_intermot.association.online_associator import hungarian_max
from scripts.n36_real_eval_common import atomic_json
from scripts.n42_t1_common import load_checkpoint, pair_feature_from_audit, preference_from_model


N37_MANIFEST = ROOT / "outputs/n37/real_event_manifest.json"
N41_ARTIFACT_ROOT = ROOT / "outputs/n41/source_replay/full/attempt1"
N42_SOURCE_MANIFEST = ROOT / "outputs/n42/diagnostic/source_embedding_manifest.json"
DEFAULT_CHECKPOINT = ROOT / "outputs/n42/training/t1_pairwise_calibration.pt"
VARIANTS = ("M0", "M1", "M2", "M3", "M4")
HORIZONS = (20, 50, 100)
PROTOCOL = "N42_T1_FROZEN_CANDIDATE_ASSOCIATION_REPLAY_V1"


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected object: {path}")
    return value


def event_item(event_id: str) -> dict[str, Any]:
    manifest = load_json(N37_MANIFEST)
    if manifest.get("status") != "PASS" or manifest.get("event_count") != 24:
        raise RuntimeError("N37 frozen manifest is not PASS/24")
    matches = [item for item in manifest.get("events", []) if str(item.get("event", {}).get("event_id")) == str(event_id)]
    if len(matches) != 1:
        raise KeyError(f"expected one frozen event, found {len(matches)}: {event_id}")
    return matches[0]


def artifact_path(event_id: str) -> Path:
    return N41_ARTIFACT_ROOT / str(event_id) / "A_ideal_gt_roi" / "lambda_1_human_1.json"


def candidate_rows(audit: dict[str, Any]) -> list[list[Any]]:
    candidates = audit.get("candidates", [])
    pids = audit.get("candidate_public_ids", [])
    rows = []
    for index, candidate in enumerate(candidates):
        pid = pids[index] if index < len(pids) else None
        if pid is None:
            continue
        rows.append([int(pid), list(np.asarray(candidate.get("box", []), dtype=float).reshape(-1))])
    return rows


def assignment_pairs(audit: dict[str, Any], assignment: np.ndarray, scores: np.ndarray) -> list[dict[str, Any]]:
    candidates = audit.get("candidates", [])
    pids = [int(value) for value in audit.get("public_id_order", [])]
    output = []
    for index, state_index in enumerate(np.asarray(assignment, dtype=int).tolist()):
        if state_index < 0 or state_index >= len(pids) or index >= len(candidates):
            continue
        output.append({
            "candidate_index": int(index),
            "candidate_obs_id": int(candidates[index].get("obs_id", index)),
            "native_tid": int(candidates[index].get("native_tid", -1)),
            "state_index": int(state_index),
            "public_id": int(pids[state_index]),
            "score": float(scores[index, state_index]),
        })
    return output


def calibrate_audit(audit: dict[str, Any], event_frame: int, target_pid: int, model: Any | None) -> dict[str, Any]:
    output = copy.deepcopy(audit)
    if model is None:
        output["t1_calibration"] = {"enabled": False, "applied": False, "reason": "T0_BASELINE"}
        return output
    public_order = [int(value) for value in output.get("public_id_order", [])]
    if int(target_pid) not in public_order:
        output["t1_calibration"] = {"enabled": True, "applied": False, "reason": "target_public_id_column_absent", "target_public_id": int(target_pid), "runtime_future_gt_used": False}
        return output
    base = np.asarray(output.get("base_scores_before_appearance", []), dtype=np.float64)
    memory = np.asarray(output.get("appearance_memory_scores", []), dtype=np.float64)
    delta = np.asarray(output.get("appearance_score_deltas", []), dtype=np.float64)
    fused = np.asarray(output.get("fused_scores", []), dtype=np.float64)
    if base.ndim != 2 or memory.shape != base.shape or delta.shape != base.shape or fused.shape != base.shape or not np.all(np.isfinite(base)) or not np.all(np.isfinite(memory)) or not np.all(np.isfinite(delta)) or not np.all(np.isfinite(fused)):
        raise RuntimeError(f"invalid score matrices at frame {output.get('frame')}")
    n_candidates = base.shape[0]
    target_column = public_order.index(int(target_pid))
    frame = int(output["frame"])
    offset = max(0, frame - int(event_frame))
    feature_rows: list[np.ndarray] = []
    feature_pairs: list[tuple[int, int]] = []
    for left in range(n_candidates):
        for right in range(n_candidates):
            if left == right:
                continue
            feature = pair_feature_from_audit(output, left, right, int(target_pid), offset)
            if feature is not None:
                feature_rows.append(feature)
                feature_pairs.append((left, right))
    preferences = preference_from_model(model, np.asarray(feature_rows, dtype=np.float32)) if feature_rows else np.zeros(0, dtype=np.float32)
    per_candidate: dict[int, list[float]] = {index: [] for index in range(n_candidates)}
    for (left, _right), preference in zip(feature_pairs, preferences.tolist()):
        per_candidate[left].append(float(preference))
    aggregate = np.asarray([float(np.mean(per_candidate[index])) if per_candidate[index] else 0.0 for index in range(n_candidates)], dtype=np.float64)
    adjusted = fused.copy()
    applied = 0
    for index in range(n_candidates):
        # Preserve the official hard-negative sentinel exactly.  The T1
        # term is a bounded preference and has fixed application scale 1.0.
        if adjusted[index, target_column] > -1.0e7:
            adjusted[index, target_column] += float(aggregate[index])
            applied += 1
    baseline_hard = base <= -1.0e7
    adjusted[baseline_hard] = fused[baseline_hard]
    assignment = hungarian_max(adjusted.astype(np.float32))
    output["t1_calibration"] = {
        "enabled": True,
        "applied": bool(applied > 0),
        "target_public_id": int(target_pid),
        "target_column": int(target_column),
        "ordered_pair_count": int(len(feature_pairs)),
        "candidate_adjustment": aggregate.astype(float).tolist(),
        "application_scale": 1.0,
        "hard_negative_preserved": bool(np.all(adjusted[baseline_hard] == fused[baseline_hard])),
        "runtime_future_gt_used": False,
    }
    output["fused_scores_before_t1"] = fused.astype(float).tolist()
    output["fused_scores"] = adjusted.astype(float).tolist()
    output["scores"] = adjusted.astype(float).tolist()
    output["public_id_score_matrix"] = adjusted.T.astype(float).tolist()
    output["public_id_fused_score_matrix"] = adjusted.T.astype(float).tolist()
    output["assignment_before_t1"] = list(np.asarray(audit.get("assignment_after_scope", audit.get("assignment", [])), dtype=int).tolist())
    output["assignment_after_scope"] = list(np.asarray(assignment, dtype=int).tolist())
    output["assignment"] = list(np.asarray(assignment, dtype=int).tolist())
    output["assignment_pairs"] = assignment_pairs(output, assignment, adjusted)
    output["assignment_pairs_after_scope"] = assignment_pairs(output, assignment, adjusted)
    # Candidate public IDs are the recomputed assigned state IDs; if the
    # matrix has more rows than states, retain the frozen birth mapping for
    # that unmatched row rather than inventing an identity.
    frozen_pids = list(audit.get("candidate_public_ids", []))
    pids = []
    for index, state_index in enumerate(assignment.tolist()):
        if 0 <= int(state_index) < len(public_order):
            pids.append(int(public_order[int(state_index)]))
        elif index < len(frozen_pids) and frozen_pids[index] is not None:
            pids.append(int(frozen_pids[index]))
        else:
            pids.append(None)
    output["candidate_public_ids"] = pids
    output["candidate_public_id_mapping_complete"] = bool(all(value is not None for value in pids))
    output["t1_candidate_mapping_recomputed"] = True
    return output


def build_trace(variant_payload: dict[str, Any], branch_name: str, event_frame: int, target_pid: int, model: Any | None, apply_t1: bool) -> dict[str, Any]:
    source_branch = variant_payload["branches"][branch_name]
    trace = []
    for entry in source_branch.get("future_trace", []):
        audit = entry.get("candidate_audit", {})
        calibrated = calibrate_audit(audit, event_frame, target_pid, model if apply_t1 else None)
        rows = candidate_rows(calibrated)
        trace.append({
            "frame": int(entry["frame"]),
            "rows": rows,
            "candidate_audit": calibrated,
        })
    return {
        "memory_write": bool(source_branch.get("memory_write", False)),
        "memory_read": bool(source_branch.get("memory_read", False)),
        "status": "PASS",
        "future_trace": trace,
        "state_summary": copy.deepcopy(source_branch.get("state_summary", {})),
        "appearance_memory": copy.deepcopy(source_branch.get("appearance_memory", {})),
        "replay_kind": "frozen_candidate_state_interface_probe",
    }


def run(event_id: str, mode: str, output: Path, checkpoint: Path | None) -> dict[str, Any]:
    started = now()
    if mode not in ("t0", "t1"):
        raise ValueError(mode)
    item = event_item(event_id)
    event = item["event"]
    event_frame = int(event["frame"])
    target_pid = int(event["public_id"])
    artifact_file = artifact_path(event_id)
    artifact = load_json(artifact_file)
    if artifact.get("status") != "PASS" or artifact.get("runtime_boundary", {}).get("runtime_future_gt_used") is not False:
        raise RuntimeError(f"frozen N41 artifact is not valid: {artifact_file}")
    model = None
    checkpoint_sha = None
    if mode == "t1":
        if checkpoint is None or not checkpoint.is_file():
            raise FileNotFoundError(checkpoint or DEFAULT_CHECKPOINT)
        model, metadata = load_checkpoint(checkpoint, "cpu")
        checkpoint_sha = __import__("hashlib").sha256(checkpoint.read_bytes()).hexdigest()
        if metadata.get("production_authorized") is not False:
            raise RuntimeError("T1 checkpoint metadata does not explicitly remain non-production")
    variants = {}
    for variant in VARIANTS:
        source_variant = artifact["variants"][variant]
        event_audit = copy.deepcopy(source_variant["event_frame_audit"])
        event_audit["t1_calibration"] = {"enabled": mode == "t1", "applied": False, "reason": "event_frame; current frame never reads new memory", "runtime_future_gt_used": False}
        variants[variant] = {
            "description": source_variant.get("description"),
            "status": "PASS",
            "event_frame_audit": event_audit,
            "branches": {
                "memory_write=False": build_trace(source_variant, "memory_write=False", event_frame, target_pid, model, False),
                "memory_write=True": build_trace(source_variant, "memory_write=True", event_frame, target_pid, model, mode == "t1" and variant != "M0"),
            },
        }
    payload = {
        "protocol": PROTOCOL,
        "status": "PASS",
        "mode": mode,
        "replay_kind": "frozen_candidate_state_interface_probe",
        "event_id": event_id,
        "sequence": str(event["sequence"]),
        "action_type": str(event["action_type"]),
        "event_frame": event_frame,
        "future_frame_start": event_frame + 1,
        "future_frame_end": int(artifact["future_frame_end"]),
        "future_frame_count": int(artifact["future_frame_count"]),
        "target_public_id_direct_from_human_event": target_pid,
        "source_artifact": str(artifact_file.relative_to(ROOT)),
        "source_artifact_sha256": __import__("hashlib").sha256(artifact_file.read_bytes()).hexdigest(),
        "corrected_source_manifest": str(N42_SOURCE_MANIFEST.relative_to(ROOT)),
        "corrected_source_manifest_sha256": __import__("hashlib").sha256(N42_SOURCE_MANIFEST.read_bytes()).hexdigest(),
        "checkpoint": None if checkpoint is None else str(checkpoint.relative_to(ROOT)),
        "checkpoint_sha256": checkpoint_sha,
        "interaction_source": "simulated_from_gt",
        "not_real_human_evidence": True,
        "runtime_boundary": {
            "runtime_future_gt_used": False,
            "gt_loaded_in_worker": False,
            "future_gt_fields_sent": [],
            "event_frame_memory_read": False,
            "event_frame_calibration_applied": False,
            "first_future_frame": event_frame + 1,
        },
        "candidate_stream_contract": {
            "reused_frozen_n41_candidate_stream": True,
            "candidate_order_unchanged": True,
            "candidate_frame_range_complete": True,
            "public_native_mapping_input_unchanged": True,
            "hungarian_solver": "scipy linear_sum_assignment via frozen project helper",
            "production_code_modified": False,
        },
        "variants": variants,
        "started_at": started,
        "finished_at": now(),
    }
    atomic_json(output, payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--event-id", required=True)
    parser.add_argument("--mode", choices=("t0", "t1"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    args = parser.parse_args()
    try:
        result = run(args.event_id, args.mode, args.output, args.checkpoint if args.mode == "t1" else None)
        print(json.dumps({"status": result["status"], "mode": args.mode, "event_id": args.event_id, "output": str(args.output)}, sort_keys=True), flush=True)
    except Exception as exc:
        failure = {"protocol": PROTOCOL, "status": "FAIL", "mode": args.mode, "event_id": args.event_id, "exception": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc(), "failure_preserved": True}
        failure_path = ROOT / "outputs/n42/attempts" / f"replay_worker_{args.mode}_{args.event_id.replace('/', '_')}_failure.json"
        if not failure_path.exists():
            atomic_json(failure_path, failure)
        raise


if __name__ == "__main__":
    main()
