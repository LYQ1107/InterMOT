#!/usr/bin/env python3
"""Independent integrity audit for the complete N47 runtime/posthoc output."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.n47_global_probe_common import CHECKPOINT, N42_RUNTIME, VARIANTS, event_map, hungarian_with_none, load, normalize_assignment, score_matrix, sha256, write_json


OUT = ROOT / "outputs/n47_global_probe/stage_04_integrity.json"
RUNTIME = ROOT / "outputs/n47_global_probe/replay/runtime"
POSTHOC = ROOT / "outputs/n47_global_probe/replay/posthoc"
RESULT = ROOT / "outputs/n47_global_probe/replay/probe_results.json"


def candidate_signature(candidates):
    return [(int(x["native_tid"]), x.get("box"), float(x.get("confidence", 0.0))) for x in candidates]


def main() -> None:
    events = event_map(); failures = []; runtime_frames = 0; posthoc_frames = 0
    runtime_files = sorted(RUNTIME.glob("*.json")); posthoc_files = sorted(POSTHOC.glob("*.json"))
    if len(runtime_files) != 24 or {p.stem for p in runtime_files} != set(events): failures.append("runtime event set")
    if len(posthoc_files) != 24 or {p.stem for p in posthoc_files} != set(events): failures.append("posthoc event set")
    for event_id, event in sorted(events.items()):
        source = load(N42_RUNTIME / f"{event_id}.json")
        runtime = load(RUNTIME / f"{event_id}.json")
        if runtime.get("runtime_boundary", {}).get("runtime_future_gt_used") is not False: failures.append(f"{event_id}: runtime provenance")
        for variant in VARIANTS:
            source_no = source["variants"][variant]["branches"]["memory_write=False"]["future_trace"]
            source_write = source["variants"][variant]["branches"]["memory_write=True"]["future_trace"]
            frames = runtime["variants"][variant]["frames"]
            if len(frames) != 100: failures.append(f"{event_id}/{variant}: 100 frames")
            for source_no_entry, source_write_entry, frame in zip(source_no, source_write, frames):
                if int(source_no_entry["frame"]) != int(frame["frame"]) or int(source_write_entry["frame"]) != int(frame["frame"]): failures.append(f"{event_id}/{variant}/{frame['frame']}: source frame alignment")
                no, write, plus, probe = frame["no_write"], frame["write_baseline"], frame["write_plus_n47"], frame["probe"]
                if candidate_signature(no["candidate_rows"]) != candidate_signature(source_no_entry["candidate_audit"]["candidates"]): failures.append(f"{event_id}/{variant}/{frame['frame']}: no candidate stream")
                if candidate_signature(write["candidate_rows"]) != candidate_signature(source_write_entry["candidate_audit"]["candidates"]): failures.append(f"{event_id}/{variant}/{frame['frame']}: write candidate stream")
                if candidate_signature(write["candidate_rows"]) != candidate_signature(plus["candidate_rows"]): failures.append(f"{event_id}/{variant}/{frame['frame']}: plus candidate stream")
                for record in (no, write, plus):
                    if len(record["candidate_native_ids"]) != len(set(record["candidate_native_ids"])): failures.append(f"{event_id}/{variant}/{frame['frame']}: duplicate native IDs")
                    if record.get("runtime_future_gt_used") is not False: failures.append(f"{event_id}/{variant}/{frame['frame']}: branch GT flag")
                    matrix = np.asarray(record["base_scores"], dtype=np.float32)
                    if matrix.shape != (record["candidate_count"], len(record["public_id_order"])) or not np.all(np.isfinite(matrix)): failures.append(f"{event_id}/{variant}/{frame['frame']}: score schema")
                    expected = normalize_assignment(hungarian_with_none(matrix), len(record["public_id_order"]))
                    if expected != record["assignment_columns"]: failures.append(f"{event_id}/{variant}/{frame['frame']}: Hungarian mismatch {record['branch']}")
                write_matrix = np.asarray(write["base_scores"], dtype=np.float32); plus_matrix = np.asarray(plus["base_scores"], dtype=np.float32)
                changed = np.abs(plus_matrix - write_matrix) > 1.0e-12
                listed = {(int(x["candidate_index"]), int(x["column"])) for x in probe["changed_cells"]}
                actual = {(int(i), int(j)) for i, j in zip(*np.where(changed))}
                if listed != actual: failures.append(f"{event_id}/{variant}/{frame['frame']}: changed-cell list")
                if np.any(plus_matrix[write_matrix <= -1.0e7] != write_matrix[write_matrix <= -1.0e7]): failures.append(f"{event_id}/{variant}/{frame['frame']}: hard negative changed")
                if bool(probe["assignment_changed"]) != (write["assignment_columns"] != plus["assignment_columns"]): failures.append(f"{event_id}/{variant}/{frame['frame']}: assignment flag")
                runtime_frames += 1
    result = load(RESULT)
    if result.get("status") != "PASS" or result.get("event_count") != 24 or result.get("variant_count") != 5: failures.append("result envelope")
    if result.get("protocol", {}).get("runtime_future_gt_used") is not False or result.get("protocol", {}).get("gt_loaded_only_after_runtime_validation") is not True: failures.append("result provenance")
    for path in posthoc_files:
        payload = load(path)
        if payload.get("runtime_future_gt_used") is not False or payload.get("gt_loaded_posthoc") is not True: failures.append(f"{path.name}: posthoc provenance")
        for variant in VARIANTS:
            if payload["variants"][variant]["runtime_frame_count"] != 100: failures.append(f"{path.name}/{variant}: posthoc frame count")
            for h in (20, 50, 100):
                for effect in ("memory_effect_no_write_to_write_baseline", "n47_incremental_effect_write_baseline_to_write_plus_n47"):
                    item = payload["variants"][variant]["horizons"][str(h)][effect]
                    needed = ("identity_utility", "target_iou_delta", "future_identity_error_reduction", "recorrection_proxy_reduction", "assignment_change_count", "assignment_change_correct_count", "assignment_change_incorrect_count", "assignment_change_neutral_count", "assignment_no_change_count", "untouched_regression", "frame_details")
                    if any(key not in item for key in needed): failures.append(f"{path.name}/{variant}/H{h}/{effect}: metric schema")
                    if int(item["assignment_change_count"]) != int(item["assignment_change_correct_count"]) + int(item["assignment_change_incorrect_count"]) + int(item["assignment_change_neutral_count"]): failures.append(f"{path.name}/{variant}/H{h}/{effect}: decomposition")
                    if int(item["assignment_change_count"]) + int(item["assignment_no_change_count"]) != int(item["evaluated_frames"]): failures.append(f"{path.name}/{variant}/H{h}/{effect}: closure")
                    posthoc_frames += int(item["evaluated_frames"])
    checkpoint = load_checkpoint_meta = __import__("torch").load(CHECKPOINT, map_location="cpu", weights_only=False)
    if checkpoint.get("production_authorized") is not False: failures.append("checkpoint authorization")
    if sha256(ROOT / "outputs/n44/training/n44_assignment_aware.pt") != "0b5e750f5d9569f71ae887595c1d88d4d625f120f8a3811f2598a852cf82348f": failures.append("N44 checkpoint changed")
    report = {"status": "PASS" if not failures else "FAIL", "protocol": "N47_STAGE_04_FULL_INTEGRITY_V1", "inputs": {"runtime": str(RUNTIME), "posthoc": str(POSTHOC), "result": str(RESULT), "checkpoint": str(CHECKPOINT)}, "outputs": {"integrity": str(OUT)}, "metrics": {"event_count": len(events), "runtime_frames": runtime_frames, "posthoc_evaluated_frames_sum_over_effects_variants_horizons": posthoc_frames, "checkpoint_sha256": sha256(CHECKPOINT), "n44_checkpoint_sha256": sha256(ROOT / "outputs/n44/training/n44_assignment_aware.pt"), "failures": failures[:100]}, "gate_checks": {"exact_24_events": not any("event set" in x for x in failures), "12000_runtime_frames": runtime_frames == 12000, "all_variants_horizons": not any("frame count" in x or "metric schema" in x for x in failures), "candidate_stream_unchanged": not any("candidate stream" in x for x in failures), "hungarian_recomputed": not any("Hungarian mismatch" in x for x in failures), "hard_negative_preserved": not any("hard negative" in x for x in failures), "runtime_future_gt_false": not any("provenance" in x or "GT flag" in x for x in failures), "assignment_decomposition": not any("decomposition" in x or "closure" in x for x in failures), "checkpoint_authorized_false": not any("authorization" in x for x in failures), "n44_checkpoint_untouched": not any("N44 checkpoint" in x for x in failures), "equal_sequence_bootstrap": result.get("protocol", {}).get("bootstrap") == "sequence_mean_then_equal_sequence_cluster_bootstrap"}, "failure_root_cause": "Full integrity requires exact frozen candidate streams, independently recomputed global Hungarian/NONE assignments, preserved hard negatives, complete posthoc schemas, and direct runtime GT=false provenance.", "next_action": "If PASS, finalize the semantic gate without treating positive holdout/offline numbers as production evidence; if FAIL, preserve and repair only the first concrete defect.", "runtime_future_gt_used": False, "gt_loaded_posthoc": True}
    write_json(OUT, report); print(json.dumps({"status": report["status"], "runtime_frames": runtime_frames, "posthoc_frames": posthoc_frames, "failures": len(failures)}))
    if failures: raise SystemExit(1)


if __name__ == "__main__":
    main()
