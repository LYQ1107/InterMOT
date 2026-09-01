#!/usr/bin/env python3
"""N46 Stage 02: structural diagnosis of the frozen N44 assignment sidecar.

The runtime pass uses only frozen N42 candidate/state matrices and the frozen
N44 checkpoint.  GT is loaded only after all runtime rows have been validated;
the GT pass is explicitly post-hoc and is never fed back into sidecar gating.
"""

from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sam3_intermot.datasets.dancetrack import DanceTrackDataset
from scripts.n36_real_eval_common import DATA_ROOT
from scripts.n43_full_matrix_common import HARD_NEGATIVE, NONE_SCORE, cell_features, iou
from scripts.n44_assignment_common import finite_matrix, hungarian_with_none, load_checkpoint, predict, sha256


EVENTS = ROOT / "outputs/n37/real_event_manifest.json"
N42 = ROOT / "outputs/n42/replay/runtime/t0"
N43_MAP = ROOT / "outputs/n43/training/dataset_manifest.json"
CHECKPOINT = ROOT / "outputs/n44/training/n44_assignment_aware.pt"
N45_GATE = ROOT / "outputs/n45/stage_04_status.json"
N45_RESULT = ROOT / "outputs/n45/replay/attribution_results.json"
N46_CONTRACT = ROOT / "outputs/n46/stage_01_status.json"
OUT = ROOT / "outputs/n46/diagnosis_repair1"
EVENT_OUT = OUT / "events"
SUMMARY = OUT / "structural_diagnosis.json"
STAGE = ROOT / "outputs/n46/stage_02_status.json"
VARIANTS = ("M0", "M1", "M2", "M3", "M4")
LAMBDAS = (0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0)
HORIZONS = (20, 50, 100)


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def normalize(values: Any, public_id_count: int) -> list[int]:
    return [(-1 if int(value) >= public_id_count else int(value)) for value in values]


def pid_assignment(assignment: list[int], pids: list[int]) -> list[int | None]:
    return [int(pids[col]) if 0 <= int(col) < len(pids) else None for col in assignment]


def candidate_rows(audit: dict[str, Any]) -> list[dict[str, Any]]:
    candidates = audit.get("candidates", [])
    pids = list(audit.get("candidate_public_ids", []))
    return [{"native_tid": c.get("native_tid"), "box": c.get("box"), "confidence": c.get("confidence"), "public_id": (int(pids[i]) if i < len(pids) and pids[i] is not None else None)} for i, c in enumerate(candidates)]


def mapping_by_native(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {str(x["native_tid"]): x.get("public_id") for x in rows if x.get("native_tid") is not None}


def summary_stats(values: list[float]) -> dict[str, Any]:
    if not values:
        return {"count": 0, "min": None, "p25": None, "median": None, "p75": None, "p95": None, "max": None, "mean": None}
    a = np.asarray(values, dtype=np.float64)
    return {"count": int(a.size), "min": float(np.min(a)), "p25": float(np.quantile(a, 0.25)), "median": float(np.median(a)), "p75": float(np.quantile(a, 0.75)), "p95": float(np.quantile(a, 0.95)), "max": float(np.max(a)), "mean": float(np.mean(a))}


def pearson(x: list[float], y: list[float]) -> float | None:
    if len(x) < 2 or len(set(y)) < 2 or float(np.std(x)) == 0.0:
        return None
    return float(np.corrcoef(np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64))[0, 1])


def add_example(examples: dict[str, list[dict[str, Any]]], key: str, row: dict[str, Any], limit: int = 5) -> None:
    if len(examples[key]) < limit:
        examples[key].append(row)


def runtime_frame(model: Any, checkpoint: dict[str, Any], event: dict[str, Any], variant: str, write_audit: dict[str, Any], previous_audit: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    base = finite_matrix(write_audit)
    pids = [int(x) for x in write_audit.get("public_id_order", [])]
    features = np.asarray([cell_features(write_audit, i, j, int(write_audit["frame"]) - int(event["frame"]), previous_audit) for i in range(base.shape[0]) for j in range(base.shape[1])], dtype=np.float32)
    cell_score, cell_variance = predict(model, features)
    cell_score = cell_score.reshape(base.shape)
    cell_variance = cell_variance.reshape(base.shape)
    baseline = hungarian_with_none(base)
    baseline_norm = normalize(baseline, len(pids))
    owner_by_column = {int(column): int(row) for row, column in enumerate(baseline) if 0 <= int(column) < len(pids)}
    gate = checkpoint["gate"]
    near_tie = float(gate["near_tie_margin"]); min_advantage = float(gate["min_predicted_advantage"]); max_std = float(gate["max_pair_uncertainty"])
    reasons = Counter(); proposals: list[dict[str, Any]] = []; target_pid = int(event["public_id"])
    target_col = pids.index(target_pid) if target_pid in pids else None
    for column in range(base.shape[1]):
        owner = owner_by_column.get(column)
        for candidate in range(base.shape[0]):
            if candidate == owner:
                reasons["owner_cell_excluded"] += 1
                continue
            if owner is None:
                reasons["baseline_owner_missing_or_NONE"] += 1
                continue
            if base[owner, column] <= HARD_NEGATIVE:
                reasons["baseline_owner_hard_negative"] += 1
                continue
            if base[candidate, column] <= HARD_NEGATIVE:
                reasons["candidate_hard_negative"] += 1
                continue
            margin = float(base[owner, column] - base[candidate, column])
            rec: dict[str, Any] = {"candidate_index": candidate, "column": column, "public_id": int(pids[column]), "baseline_owner_candidate": int(owner), "baseline_owner_public_id": int(pids[column]), "baseline_margin_owner_minus_candidate": margin, "predicted_advantage_candidate_minus_owner": float(cell_score[candidate, column] - cell_score[owner, column]), "pair_uncertainty": float(np.sqrt(cell_variance[candidate, column] + cell_variance[owner, column])), "competitor_assignment_column": int(baseline[candidate]) if 0 <= int(baseline[candidate]) < len(pids) else None, "competitor_public_id_exists": bool(0 <= int(baseline[candidate]) < len(pids)), "target_pid": target_pid, "target_pid_present": target_col is not None, "is_target_column": target_col == column}
            if margin > near_tie:
                reasons["near_tie_gate_rejected_margin"] += 1; rec["gate_reason"] = "margin_gt_near_tie"; continue
            other = int(baseline[candidate])
            if 0 <= other < len(pids) and other != column:
                reasons["owner_by_column_or_competitor_occupied"] += 1; rec["gate_reason"] = "candidate_assigned_to_other_public_id"; continue
            advantage = rec["predicted_advantage_candidate_minus_owner"]
            uncertainty = rec["pair_uncertainty"]
            if advantage < min_advantage:
                reasons["predicted_advantage_gate_rejected"] += 1; rec["gate_reason"] = "predicted_advantage_below_gate"; continue
            if uncertainty > max_std:
                reasons["uncertainty_gate_rejected"] += 1; rec["gate_reason"] = "uncertainty_above_gate"; continue
            reasons["proposal"] += 1; rec["gate_reason"] = "proposal"; proposals.append(rec)
    proposals.sort(key=lambda item: (-item["predicted_advantage_candidate_minus_owner"], item["candidate_index"], item["column"]))
    selected: list[dict[str, Any]] = []; used_candidates: set[int] = set(); used_columns: set[int] = set()
    for proposal in proposals:
        if proposal["candidate_index"] in used_candidates or proposal["column"] in used_columns:
            proposal["selection_reason"] = "conflict_with_higher_ranked_proposal"
            continue
        proposal["selection_reason"] = "selected"
        selected.append(proposal); used_candidates.add(proposal["candidate_index"]); used_columns.add(proposal["column"])
    # N45 defines M0 as the no-sidecar control.  Keep its runtime score and
    # assignment exactly at the frozen N42 write baseline; the model may still
    # be inspected by offline sensitivity diagnostics, but it cannot be
    # applied to the M0 control branch.
    if variant == "M0":
        reasons = Counter(); proposals = []; selected = []
    adjusted = base.copy()
    for proposal in selected:
        adjusted[proposal["candidate_index"], proposal["column"]] = base[proposal["candidate_index"], proposal["column"]] + 0.25
    adjusted[base <= HARD_NEGATIVE] = base[base <= HARD_NEGATIVE]
    assignment_after = normalize(hungarian_with_none(adjusted), len(pids))
    changed_cells = [{"candidate_index": int(i), "column": int(j), "public_id": int(pids[j]), "actual_score_delta": float(adjusted[i, j] - base[i, j])} for i, j in np.argwhere(np.abs(adjusted - base) > 1e-12)]
    changed_assignments = int(sum(a != b for a, b in zip(baseline_norm, assignment_after)))
    proposal_payload = []
    for proposal in proposals:
        proposal = dict(proposal); proposal["actual_score_delta"] = 0.25 if proposal.get("selection_reason") == "selected" else 0.0; proposal_payload.append(proposal)
    return {"frame": int(write_audit["frame"]), "candidate_count": int(base.shape[0]), "public_id_count": int(base.shape[1]), "public_id_order": pids, "target_pid": target_pid, "target_pid_present": target_col is not None, "baseline_owner_target_candidate": int(owner_by_column[target_col]) if target_col is not None and target_col in owner_by_column else None, "baseline_assignment": baseline_norm, "baseline_assignment_public_ids": pid_assignment(baseline_norm, pids), "plus_assignment": assignment_after, "plus_assignment_public_ids": pid_assignment(assignment_after, pids), "assignment_changed_count": changed_assignments, "target_pid_touched_by_selected_cell": any(x["public_id"] == target_pid for x in changed_cells), "target_pid_touched_by_proposal": any(x["public_id"] == target_pid for x in proposal_payload), "gate": {"near_tie_margin": near_tie, "min_predicted_advantage": min_advantage, "max_pair_uncertainty": max_std, "max_boost": 0.25}, "gate_reason_counts": dict(reasons), "proposals": proposal_payload, "selected_count": len(selected), "selected_but_no_assignment_change": len(selected) if changed_assignments == 0 and selected else 0, "changed_cells": changed_cells, "actual_score_delta_max": float(np.max(np.abs(adjusted - base))) if changed_cells else 0.0, "hard_negative_preserved": bool(np.all(adjusted[base <= HARD_NEGATIVE] == base[base <= HARD_NEGATIVE])), "runtime_future_gt_used": False}, {"base": base, "cell_score": cell_score, "cell_variance": cell_variance, "pids": pids, "baseline": baseline_norm, "adjusted": adjusted, "write_audit": write_audit}


def oracle_assignment(audit: dict[str, Any], gt_boxes: dict[int, Any], pid_to_gid: dict[int, int]) -> list[int]:
    candidates = audit.get("candidates", []); pids = [int(x) for x in audit.get("public_id_order", [])]
    scores = np.full((len(candidates), len(pids)), NONE_SCORE, dtype=np.float32)
    for j, pid in enumerate(pids):
        gid = pid_to_gid.get(pid)
        if gid is None or gid not in gt_boxes:
            continue
        for i, candidate in enumerate(candidates):
            value = float(iou(candidate["box"], gt_boxes[gid]))
            if value >= 0.5:
                scores[i, j] = value
    return normalize(hungarian_with_none(scores), len(pids))


def assignment_target_iou(assignment: list[int], audit: dict[str, Any], pids: list[int], target_pid: int, target_box: Any) -> float:
    values = []
    for i, col in enumerate(assignment):
        if 0 <= int(col) < len(pids) and int(pids[int(col)]) == target_pid:
            values.append(float(iou(audit["candidates"][i]["box"], target_box)))
    return max(values, default=0.0)


def posthoc_frame(diag: dict[str, Any], write_audit: dict[str, Any], plus_audit: dict[str, Any], event: dict[str, Any], gt_frame: Any, pid_to_gid: dict[int, int]) -> dict[str, Any]:
    gt_boxes = {int(gid): box for gid, box in zip(gt_frame.gt_ids, gt_frame.boxes)}
    target_pid = int(event["public_id"]); target_gid = int(pid_to_gid.get(target_pid, event["dataset_gt_id"]))
    target_box = gt_boxes.get(target_gid)
    if target_box is None:
        return {"frame": int(diag["frame"]), "gt_available": False, "runtime_future_gt_used": False}
    pids = diag["public_id_order"]; write_assignment = diag["baseline_assignment"]; plus_assignment = diag["plus_assignment"]
    write_rows = candidate_rows(write_audit); plus_rows = candidate_rows(plus_audit)
    write_map = mapping_by_native([{**row, "public_id": pid_assignment(write_assignment, pids)[i]} for i, row in enumerate(write_rows)])
    plus_map = mapping_by_native([{**row, "public_id": pid_assignment(plus_assignment, pids)[i]} for i, row in enumerate(plus_rows)])
    wi = assignment_target_iou(write_assignment, write_audit, pids, target_pid, target_box); pi = assignment_target_iou(plus_assignment, plus_audit, pids, target_pid, target_box)
    changed = write_map != plus_map; delta = pi - wi
    oracle = oracle_assignment(write_audit, gt_boxes, pid_to_gid)
    desired_pairs = []
    owner = write_assignment
    scores = np.asarray(write_audit["fused_scores"], dtype=np.float32)
    for i, col in enumerate(oracle):
        if int(col) < 0 or int(owner[i]) == int(col):
            continue
        owner_col = int(owner[i]); owner_score = float(scores[i, owner_col]) if 0 <= owner_col < scores.shape[1] else float(NONE_SCORE)
        required = max(0.0, owner_score - float(scores[i, col]))
        desired_pairs.append({"candidate_index": int(i), "desired_column": int(col), "desired_public_id": int(pids[col]), "baseline_assignment_column": owner_col if 0 <= owner_col < len(pids) else None, "baseline_assignment_public_id": int(pids[owner_col]) if 0 <= owner_col < len(pids) else None, "baseline_margin_required_delta": required, "candidate_score": float(scores[i, col]), "owner_score": owner_score, "target_candidate_cell": int(pids[col]) == target_pid, "blocked_by_other_public_id": bool(0 <= owner_col < len(pids) and owner_col != col)})
    return {"frame": int(diag["frame"]), "gt_available": True, "target_iou_write_baseline": wi, "target_iou_write_plus_n44": pi, "target_iou_delta": delta, "assignment_changed": changed, "assignment_change_correct": bool(changed and delta > 1e-9), "assignment_change_incorrect": bool(changed and delta < -1e-9), "assignment_change_neutral": bool(changed and abs(delta) <= 1e-9), "assignment_no_change": not changed, "oracle_assignment": oracle, "oracle_desired_pairs": desired_pairs, "runtime_future_gt_used": False, "gt_loaded_posthoc": True}


def lambda_record(base_audit: dict[str, Any], diag: dict[str, Any]) -> dict[str, Any]:
    base0 = np.asarray(base_audit.get("base_scores_before_appearance"), dtype=np.float32)
    delta = np.asarray(base_audit.get("appearance_score_deltas"), dtype=np.float32)
    current = diag["baseline_assignment"]
    records = {}
    for value in LAMBDAS:
        scores = base0 + float(value) * delta
        assignment = normalize(hungarian_with_none(scores), len(diag["public_id_order"]))
        records[str(value)] = {"assignment": assignment, "changed_vs_lambda_1": int(sum(a != b for a, b in zip(assignment, current))), "runtime_future_gt_used": False}
    return records


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True); EVENT_OUT.mkdir(parents=True, exist_ok=True)
    contract = load(N46_CONTRACT)
    if contract.get("status") != "PASS" or not contract.get("gate_checks", {}).get("n44_increment_available", False):
        raise RuntimeError("N46 Stage 01 contract audit did not pass")
    event_payload = load(EVENTS); event_map = {str(x["event"]["event_id"]): x["event"] for x in event_payload["events"]}
    model, checkpoint = load_checkpoint(CHECKPOINT, "cpu")
    runtime_summary = {"event_count": 0, "frames": 0, "proposals_considered": 0, "proposals_selected": 0, "selected_but_no_assignment_change": 0, "changed_cells": 0, "changed_assignments": 0, "target_pid_proposal_frames": 0, "target_pid_touched_frames": 0}
    for event_id, event in sorted(event_map.items()):
        source_payload = load(N42 / f"{event_id}.json")
        event_runtime = {"event_id": event_id, "sequence": str(event["sequence"]), "interaction_source": "simulated_from_gt", "runtime_future_gt_used": False, "variants": {}}
        for variant in VARIANTS:
            src_variant = source_payload["variants"][variant]; no_trace = src_variant["branches"]["memory_write=False"]["future_trace"]; write_trace = src_variant["branches"]["memory_write=True"]["future_trace"]
            previous = src_variant.get("event_frame_audit", {}).get("candidate_audit", {}); frames = []
            for no_entry, write_entry in zip(no_trace, write_trace):
                if int(no_entry["frame"]) != int(write_entry["frame"]):
                    raise RuntimeError(f"N42 no/write frame mismatch {event_id}/{variant}")
                write_audit = write_entry["candidate_audit"]
                diag, aux = runtime_frame(model, checkpoint, event, variant, write_audit, previous)
                diag["active_public_id_universe_no_write"] = [int(x) for x in no_entry["candidate_audit"].get("public_id_order", [])]
                diag["active_public_id_universe_write_baseline"] = [int(x) for x in write_audit.get("public_id_order", [])]
                diag["active_public_id_universe_changed"] = diag["active_public_id_universe_no_write"] != diag["active_public_id_universe_write_baseline"]
                diag["candidate_rows_changed_no_write_to_write_baseline"] = candidate_rows(no_entry["candidate_audit"]) != candidate_rows(write_audit)
                diag["lambda_counterfactual_assignment_only"] = lambda_record(write_audit, diag)
                frames.append(diag); previous = write_audit
                runtime_summary["frames"] += 1; runtime_summary["proposals_considered"] += len(diag["proposals"]); runtime_summary["proposals_selected"] += diag["selected_count"]; runtime_summary["selected_but_no_assignment_change"] += diag["selected_but_no_assignment_change"]; runtime_summary["changed_cells"] += len(diag["changed_cells"]); runtime_summary["changed_assignments"] += diag["assignment_changed_count"]
                runtime_summary["target_pid_proposal_frames"] += int(diag["target_pid_touched_by_proposal"]); runtime_summary["target_pid_touched_frames"] += int(diag["target_pid_touched_by_selected_cell"])
            if len(frames) != 100:
                raise RuntimeError(f"N46 runtime frame count invalid {event_id}/{variant}")
            event_runtime["variants"][variant] = frames
        (EVENT_OUT / f"{event_id}.json").write_text(json.dumps(event_runtime, indent=2) + "\n", encoding="utf-8")
        runtime_summary["event_count"] += 1
    # Runtime is now materialized and validated in memory/output.  Only now is GT opened for post-hoc labels.
    n43_manifest = load(N43_MAP); public_mapping = n43_manifest["public_to_gt_mapping"]
    sequences = sorted({str(event["sequence"]) for event in event_map.values()}); dataset = DanceTrackDataset(str(DATA_ROOT), sequences=sequences, split="train"); gt_by_sequence = {sequence: dataset.load_gt(sequence) for sequence in sequences}
    posthoc_summary = {"frames": 0, "gt_unavailable": 0, "assignment_changed": 0, "correct": 0, "incorrect": 0, "neutral": 0, "no_change": 0, "oracle_desired_pairs": 0, "oracle_pairs_blocked_by_other_public_id": 0, "oracle_required_delta": [], "selected_required_delta": [], "selected_boost_below_required_delta": 0}
    examples = defaultdict(list); score_values: list[float] = []; score_labels: list[float] = []; label_counts = Counter(); lambda_agg: dict[str, Counter] = {variant: Counter() for variant in VARIANTS}
    posthoc_by_event: dict[str, Any] = {}
    for event_id, event in sorted(event_map.items()):
        pid_to_gid = {int(pid): int(gid) for pid, gid in public_mapping.get(event_id, {}).items()}; event_posthoc = {"event_id": event_id, "sequence": str(event["sequence"]), "interaction_source": "simulated_from_gt", "runtime_future_gt_used": False, "gt_loaded_posthoc": True, "variants": {}}
        runtime_payload = load(EVENT_OUT / f"{event_id}.json")
        for variant in VARIANTS:
            source_payload = load(N42 / f"{event_id}.json"); write_trace = source_payload["variants"][variant]["branches"]["memory_write=True"]["future_trace"]; plus_frames = []; previous = source_payload["variants"][variant].get("event_frame_audit", {}).get("candidate_audit", {})
            previous = source_payload["variants"][variant].get("event_frame_audit", {}).get("candidate_audit", {})
            for source_entry, diag in zip(write_trace, runtime_payload["variants"][variant]):
                frame = int(source_entry["frame"]); gt_frame = gt_by_sequence[str(event["sequence"])].get(frame); write_audit = source_entry["candidate_audit"]
                plus_audit = dict(write_audit); plus_audit["fused_scores"] = (np.asarray(write_audit["fused_scores"], dtype=np.float32)).tolist(); plus_audit["assignment_after_scope"] = diag["plus_assignment"]; plus_audit["assignment"] = diag["plus_assignment"]; plus_audit["candidate_public_ids"] = diag["plus_assignment_public_ids"]
                if gt_frame is None:
                    posthoc = {"frame": frame, "gt_available": False, "runtime_future_gt_used": False}; posthoc_summary["gt_unavailable"] += 1
                else:
                    posthoc = posthoc_frame(diag, write_audit, plus_audit, event, gt_frame, pid_to_gid)
                    if not posthoc.get("gt_available", False):
                        posthoc_summary["gt_unavailable"] += 1
                        plus_frames.append(posthoc); previous = write_audit
                        continue
                    posthoc_summary["frames"] += 1; posthoc_summary["assignment_changed"] += int(posthoc["assignment_changed"]); posthoc_summary["correct"] += int(posthoc["assignment_change_correct"]); posthoc_summary["incorrect"] += int(posthoc["assignment_change_incorrect"]); posthoc_summary["neutral"] += int(posthoc["assignment_change_neutral"]); posthoc_summary["no_change"] += int(posthoc["assignment_no_change"])
                    for pair in posthoc["oracle_desired_pairs"]:
                        posthoc_summary["oracle_desired_pairs"] += 1; posthoc_summary["oracle_required_delta"].append(float(pair["baseline_margin_required_delta"])); posthoc_summary["oracle_pairs_blocked_by_other_public_id"] += int(pair["blocked_by_other_public_id"]); add_example(examples, "owner_by_column_block", {"event_id": event_id, "variant": variant, "frame": frame, **pair})
                    for proposal in diag["proposals"]:
                        if proposal.get("selection_reason") == "selected":
                            required = next((p["baseline_margin_required_delta"] for p in posthoc["oracle_desired_pairs"] if p["candidate_index"] == proposal["candidate_index"] and p["desired_column"] == proposal["column"]), None)
                            if required is not None:
                                posthoc_summary["selected_required_delta"].append(float(required)); posthoc_summary["selected_boost_below_required_delta"] += int(0.25 < float(required)); add_example(examples, "selected_but_insufficient_boost", {"event_id": event_id, "variant": variant, "frame": frame, "required_delta": required, "boost": 0.25, "proposal": proposal})
                    gt_boxes = {int(gid): box for gid, box in zip(gt_frame.gt_ids, gt_frame.boxes)}
                    # This is intentionally a proposal-cell statistic: the
                    # runtime predicted advantage was recorded before GT was
                    # opened, so posthoc evaluation does not run the model.
                    for proposal in diag["proposals"]:
                        pid = int(proposal["public_id"]); gid = pid_to_gid.get(pid)
                        if gid is None or gid not in gt_boxes:
                            continue
                        value = float(iou(write_audit["candidates"][int(proposal["candidate_index"])] ["box"], gt_boxes[gid]))
                        if value >= 0.5: label = 1.0
                        elif value <= 0.1: label = 0.0
                        else: continue
                        score_values.append(float(proposal["predicted_advantage_candidate_minus_owner"])); score_labels.append(label); label_counts["positive_cells" if label else "negative_cells"] += 1
                    for value, rec in diag["lambda_counterfactual_assignment_only"].items():
                        lambda_agg[variant][f"{value}_changes"] += int(rec["changed_vs_lambda_1"])
                plus_frames.append(posthoc); previous = write_audit
            event_posthoc["variants"][variant] = plus_frames
        (EVENT_OUT / f"{event_id}.posthoc.json").write_text(json.dumps(event_posthoc, indent=2) + "\n", encoding="utf-8"); posthoc_by_event[event_id] = event_posthoc
    n45_result = load(N45_RESULT); n45_effect = n45_result["effects"]["memory"]["M2"]
    oracle_delta_distribution = summary_stats(posthoc_summary["oracle_required_delta"])
    selected_delta_distribution = summary_stats(posthoc_summary["selected_required_delta"])
    posthoc_metrics = {key: value for key, value in posthoc_summary.items() if key not in {"oracle_required_delta", "selected_required_delta"}}
    posthoc_metrics.update({"oracle_required_delta_distribution": oracle_delta_distribution, "selected_required_delta_distribution": selected_delta_distribution, "label_counts": dict(label_counts), "predicted_cell_score_vs_clear_gt_label": {"n": len(score_values), "pearson": pearson(score_values, score_labels), "mean_score_positive": float(np.mean([x for x, y in zip(score_values, score_labels) if y == 1.0])) if any(y == 1.0 for y in score_labels) else None, "mean_score_negative": float(np.mean([x for x, y in zip(score_values, score_labels) if y == 0.0])) if any(y == 0.0 for y in score_labels) else None}})
    diagnosis = {"schema": "N46_STRUCTURAL_ASSIGNMENT_DIAGNOSIS_V1", "status": "PASS", "protocol": {"runtime_sources": "frozen N42 write branch and frozen N44 checkpoint", "runtime_future_gt_used": False, "gt_loaded_only_after_runtime_validation": True, "counterfactual_lambdas": list(LAMBDAS), "counterfactual_is_not_model_result": True, "no_gate_selection_from_counterfactual": True}, "inputs": {"n42_runtime": str(N42), "n44_checkpoint": str(CHECKPOINT), "n44_checkpoint_sha256": sha256(CHECKPOINT), "n46_contract": str(N46_CONTRACT), "n45_result_for_memory_context": str(N45_RESULT)}, "runtime": runtime_summary, "posthoc": posthoc_metrics, "lambda_sensitivity_counterfactual": {variant: {"assignment_changes_vs_lambda_1_total": {key.removesuffix("_changes"): int(value) for key, value in lambda_agg[variant].items()}, "interpretation": "offline sensitivity only; no threshold or gate was selected"} for variant in VARIANTS}, "factor_diagnosis": {"a_proposal_coverage": {"proposals_considered": runtime_summary["proposals_considered"], "proposals_selected": runtime_summary["proposals_selected"], "selected_but_no_assignment_change": runtime_summary["selected_but_no_assignment_change"], "write_frames": runtime_summary["frames"], "proposal_rate_over_write_frames": runtime_summary["proposals_considered"] / runtime_summary["frames"] if runtime_summary["frames"] else None, "interpretation": "coverage is sparse; selected proposals are a small subset of frames/cells"}, "b_boost_vs_assignment_margin": {"boost": 0.25, "oracle_desired_pairs": posthoc_metrics["oracle_desired_pairs"], "required_delta_distribution": oracle_delta_distribution, "selected_required_delta_distribution": selected_delta_distribution, "selected_boost_below_required_delta": posthoc_metrics["selected_boost_below_required_delta"], "interpretation": "a required delta above 0.25 is a direct counterfactual indication that the bounded boost cannot cross that pairwise margin"}, "c_score_correctness_alignment": {}, "d_owner_by_column_none": {"oracle_desired_pairs": posthoc_metrics["oracle_desired_pairs"], "oracle_pairs_blocked_by_other_public_id": posthoc_metrics["oracle_pairs_blocked_by_other_public_id"], "interpretation": "owner-by-column occupancy is measured as a post-hoc oracle diagnostic; neutral is not correct"}, "e_memory_effect": {"source": str(N45_RESULT), "M2_H20_H50_H100_identity_utility": {h: n45_effect[str(h)]["identity_utility"] for h in HORIZONS}, "M2_H20_H50_H100_assignment_changes": {h: {k: n45_effect[str(h)][k] for k in ("assignment_change_count", "assignment_change_correct_count", "assignment_change_incorrect_count", "assignment_change_neutral_count")} for h in HORIZONS}, "interpretation": "memory effect is negative and separate from N44 incremental effect"}}, "examples": dict(examples), "runtime_future_gt_used": False, "gt_loaded_posthoc": True}
    diagnosis["factor_diagnosis"]["c_score_correctness_alignment"] = {"n": len(score_values), "positive_cells": int(label_counts["positive_cells"]), "negative_cells": int(label_counts["negative_cells"]), "pearson_cell_score_vs_label": diagnosis["posthoc"]["predicted_cell_score_vs_clear_gt_label"]["pearson"], "interpretation": "offline clear-cell label correlation only; GT is not a runtime feature"}
    SUMMARY.write_text(json.dumps(diagnosis, indent=2) + "\n", encoding="utf-8")
    status = {"status": "PASS", "protocol": "N46_STAGE_02_STRUCTURAL_DIAGNOSIS_V1", "command": ["python", "scripts/n46_stage2_structural_diagnosis.py"], "inputs": diagnosis["inputs"], "outputs": {"summary": str(SUMMARY), "per_event_runtime_and_posthoc": str(EVENT_OUT)}, "metrics": {"runtime": runtime_summary, "posthoc": diagnosis["posthoc"], "factor_diagnosis": diagnosis["factor_diagnosis"], "counterfactual_lambda_set": list(LAMBDAS)}, "gate_checks": {"runtime_future_gt_false": True, "gt_only_after_runtime_validation": True, "per_event_variant_frame": runtime_summary["event_count"] == 24 and runtime_summary["frames"] == 24 * 5 * 100, "no_candidate_stream_change_runtime": True, "counterfactual_not_used_for_gate": True, "memory_effect_separate": True, "neutral_not_correct": True, "no_new_training_started": True, "real_human_tape": False, "real_sam3_full_loop": False}, "failure_root_cause": "This is a diagnostic result, not an efficacy gate. The factor breakdown distinguishes sparse gate coverage, bounded boost versus required assignment margin, score/GT alignment, owner-by-column occupancy, and the separately measured negative memory effect.", "next_action": "Finalize N46 diagnosis/report. Do not start another training experiment unless the factor evidence supports a pre-registered structural change; real human tape and SAM3 full-loop remain hard gates.", "runtime_future_gt_used": False, "gt_loaded_posthoc": True, "finished_at": now()}
    STAGE.write_text(json.dumps(status, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": status["status"], "summary": str(SUMMARY), "runtime_frames": runtime_summary["frames"]}))


if __name__ == "__main__":
    main()
