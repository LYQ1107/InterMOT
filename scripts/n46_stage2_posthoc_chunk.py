#!/usr/bin/env python3
"""Bounded posthoc continuation for the completed N46 runtime diagnosis."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sam3_intermot.datasets.dancetrack import DanceTrackDataset
from scripts.n36_real_eval_common import DATA_ROOT
from scripts.n43_full_matrix_common import iou
from scripts.n46_stage2_structural_diagnosis import EVENTS, N42, N43_MAP, N46_CONTRACT, EVENT_OUT as RUNTIME_OUT, VARIANTS
from scripts.n46_stage2_structural_diagnosis import load, posthoc_frame, summary_stats


OUT = ROOT / "outputs/n46/diagnosis_final/events"
CHUNK_OUT = ROOT / "outputs/n46/posthoc_chunks"


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=int, required=True)
    parser.add_argument("--end", type=int, required=True)
    parser.add_argument("--runtime-dir", type=Path, default=RUNTIME_OUT)
    parser.add_argument("--output-dir", type=Path, default=OUT)
    parser.add_argument("--chunk-status-dir", type=Path, default=CHUNK_OUT)
    args = parser.parse_args()
    runtime_out = args.runtime_dir if args.runtime_dir.is_absolute() else ROOT / args.runtime_dir
    output_dir = args.output_dir if args.output_dir.is_absolute() else ROOT / args.output_dir
    chunk_status_dir = args.chunk_status_dir if args.chunk_status_dir.is_absolute() else ROOT / args.chunk_status_dir
    contract = load(N46_CONTRACT)
    if contract.get("status") != "PASS" or not contract.get("gate_checks", {}).get("n44_increment_available", False):
        raise RuntimeError("N46 Stage 01 contract is not PASS")
    event_payload = load(EVENTS); all_events = {str(x["event"]["event_id"]): x["event"] for x in event_payload["events"]}; ids = sorted(all_events)
    runtime_files = sorted(path for path in runtime_out.glob("*.json") if not path.name.endswith(".posthoc.json"))
    if len(runtime_files) != 24 or {x.stem for x in runtime_files} != set(ids):
        raise RuntimeError("all 24 runtime diagnostics must exist before posthoc GT load")
    for path in runtime_files:
        payload = load(path)
        if payload.get("runtime_future_gt_used") is not False or set(payload.get("variants", {})) != set(VARIANTS):
            raise RuntimeError(f"runtime validation failed before posthoc: {path.name}")
        if any(len(payload["variants"][v]) != 100 for v in VARIANTS):
            raise RuntimeError(f"runtime trace length failed before posthoc: {path.name}")
    start, end = max(0, args.start), min(len(ids), args.end)
    if start >= end:
        raise ValueError("empty event chunk")
    mapping = load(N43_MAP)["public_to_gt_mapping"]
    selected_ids = ids[start:end]; sequences = sorted({str(all_events[x]["sequence"]) for x in selected_ids}); dataset = DanceTrackDataset(str(DATA_ROOT), sequences=sequences, split="train"); gt = {sequence: dataset.load_gt(sequence) for sequence in sequences}
    output_dir.mkdir(parents=True, exist_ok=True); chunk_status_dir.mkdir(parents=True, exist_ok=True)
    completed = []
    for event_id in selected_ids:
        event = all_events[event_id]; pids_to_gt = {int(pid): int(gid) for pid, gid in mapping.get(event_id, {}).items()}; runtime_payload = load(runtime_out / f"{event_id}.json"); source_payload = load(N42 / f"{event_id}.json"); event_posthoc = {"event_id": event_id, "sequence": str(event["sequence"]), "interaction_source": "simulated_from_gt", "runtime_future_gt_used": False, "gt_loaded_posthoc": True, "variants": {}}
        local = {"frames": 0, "gt_unavailable": 0, "assignment_changed": 0, "correct": 0, "incorrect": 0, "neutral": 0, "no_change": 0, "oracle_desired_pairs": 0, "oracle_pairs_blocked_by_other_public_id": 0, "oracle_required_delta": [], "selected_required_delta": [], "selected_boost_below_required_delta": 0, "score_values": [], "score_labels": [], "label_counts": Counter(), "lambda_changes": {v: Counter() for v in VARIANTS}}
        for variant in VARIANTS:
            write_trace = source_payload["variants"][variant]["branches"]["memory_write=True"]["future_trace"]; previous = source_payload["variants"][variant].get("event_frame_audit", {}).get("candidate_audit", {}); frames = []
            for source_entry, diag in zip(write_trace, runtime_payload["variants"][variant]):
                frame = int(source_entry["frame"]); write_audit = source_entry["candidate_audit"]; gt_frame = gt[str(event["sequence"])].get(frame); plus_audit = dict(write_audit); plus_audit["candidate_public_ids"] = diag["plus_assignment_public_ids"]; plus_audit["assignment"] = diag["plus_assignment"]; plus_audit["assignment_after_scope"] = diag["plus_assignment"]
                if gt_frame is None:
                    posthoc = {"frame": frame, "gt_available": False, "runtime_future_gt_used": False}; local["gt_unavailable"] += 1
                else:
                    posthoc = posthoc_frame(diag, write_audit, plus_audit, event, gt_frame, pids_to_gt); local["frames"] += int(posthoc.get("gt_available", False))
                    if posthoc.get("gt_available", False):
                        local["assignment_changed"] += int(posthoc["assignment_changed"]); local["correct"] += int(posthoc["assignment_change_correct"]); local["incorrect"] += int(posthoc["assignment_change_incorrect"]); local["neutral"] += int(posthoc["assignment_change_neutral"]); local["no_change"] += int(posthoc["assignment_no_change"])
                        for pair in posthoc["oracle_desired_pairs"]:
                            local["oracle_desired_pairs"] += 1; local["oracle_required_delta"].append(float(pair["baseline_margin_required_delta"])); local["oracle_pairs_blocked_by_other_public_id"] += int(pair["blocked_by_other_public_id"])
                        for proposal in diag["proposals"]:
                            if proposal.get("selection_reason") != "selected":
                                continue
                            required = next((p["baseline_margin_required_delta"] for p in posthoc["oracle_desired_pairs"] if p["candidate_index"] == proposal["candidate_index"] and p["desired_column"] == proposal["column"]), None)
                            if required is not None:
                                local["selected_required_delta"].append(float(required)); local["selected_boost_below_required_delta"] += int(0.25 < float(required))
                        boxes = {int(gid): box for gid, box in zip(gt_frame.gt_ids, gt_frame.boxes)}
                        for proposal in diag["proposals"]:
                            pid = int(proposal["public_id"]); gid = pids_to_gt.get(pid)
                            if gid is None or gid not in boxes:
                                continue
                            value = float(iou(write_audit["candidates"][int(proposal["candidate_index"])] ["box"], boxes[gid]))
                            if value >= 0.5: label = 1.0
                            elif value <= 0.1: label = 0.0
                            else: continue
                            local["score_values"].append(float(proposal["predicted_advantage_candidate_minus_owner"])); local["score_labels"].append(label); local["label_counts"]["positive_cells" if label else "negative_cells"] += 1
                        for value, rec in diag["lambda_counterfactual_assignment_only"].items():
                            local["lambda_changes"][variant][f"{value}_changes"] += int(rec["changed_vs_lambda_1"])
                frames.append(posthoc); previous = write_audit
            event_posthoc["variants"][variant] = frames
        serial_local = {key: (dict(value) if isinstance(value, Counter) else value) for key, value in local.items() if key not in {"oracle_required_delta", "selected_required_delta", "score_values", "score_labels", "label_counts", "lambda_changes"}}
        serial_local.update({"oracle_required_delta_distribution": summary_stats(local["oracle_required_delta"]), "selected_required_delta_distribution": summary_stats(local["selected_required_delta"]), "score_values": local["score_values"], "score_labels": local["score_labels"], "label_counts": dict(local["label_counts"]), "lambda_changes": {v: dict(c) for v, c in local["lambda_changes"].items()}})
        event_posthoc["summary"] = serial_local; (output_dir / f"{event_id}.posthoc.json").write_text(json.dumps(event_posthoc, indent=2) + "\n", encoding="utf-8"); completed.append(event_id)
    chunk = {"status": "PASS", "protocol": "N46_STAGE_02_POSTHOC_CHUNK_V1", "command": ["python", "scripts/n46_stage2_posthoc_chunk.py", "--start", str(start), "--end", str(end), "--runtime-dir", str(runtime_out), "--output-dir", str(output_dir), "--chunk-status-dir", str(chunk_status_dir)], "inputs": {"n46_contract": str(N46_CONTRACT), "n46_runtime_diagnostics": str(runtime_out), "n42_frozen_runtime": str(N42), "n43_offline_mapping": str(N43_MAP)}, "outputs": {"event_posthoc": str(output_dir)}, "metrics": {"start": start, "end": end, "event_count": len(completed), "events": completed}, "gate_checks": {"all_runtime_24_validated_before_gt": True, "runtime_future_gt_false": True, "gt_loaded_posthoc": True, "simulated_provenance_explicit": True}, "failure_root_cause": "Chunked continuation preserves the fixed posthoc definition while bounding execution duration.", "next_action": "Run remaining posthoc chunks, then assemble Stage 02 without changing thresholds, seeds, metrics or checkpoint.", "runtime_future_gt_used": False, "finished_at": now()}
    (chunk_status_dir / f"chunk_{start:02d}_{end:02d}_status.json").write_text(json.dumps(chunk, indent=2) + "\n", encoding="utf-8"); print(json.dumps({"status": "PASS", "start": start, "end": end, "events": len(completed)}))


if __name__ == "__main__":
    main()
