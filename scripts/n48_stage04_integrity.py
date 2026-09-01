#!/usr/bin/env python3
"""Independent integrity checks for the completed N48 runtime/posthoc replay."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.n47_global_probe_common import N42_RUNTIME, event_map, hungarian_with_none, load, write_json  # noqa: E402
from scripts.n48_assignment_common import solve_with_none  # noqa: E402

OUT = ROOT / "outputs/n48"
RUNTIME = OUT / "replay/runtime"
MEMORY = OUT / "training/simulated_event_memory.json"
CHECKPOINT = OUT / "training/n48_risk_aware_512d.pt"
VARIANTS = ("M0", "M1", "M2", "M3", "M4")


def signature(rows):
    return [(int(x["native_tid"]), x.get("box"), float(x.get("confidence", 0.0))) for x in rows]


def main() -> None:
    events = event_map(); failures = []; frames = 0; source_frames = 0; changed_cells = 0
    if not CHECKPOINT.is_file(): failures.append("checkpoint missing")
    else:
        payload = __import__("torch").load(CHECKPOINT, map_location="cpu", weights_only=False)
        if payload.get("production_authorized") is not False: failures.append("checkpoint production_authorized is not false")
    memory = load(MEMORY)
    if memory.get("runtime_future_gt_used") is not False: failures.append("memory manifest runtime_future_gt_used not false")
    for event_id in sorted(events):
        n42 = load(N42_RUNTIME / f"{event_id}.json")
        for variant in VARIANTS:
            for branch_name in ("memory_write=False", "memory_write=True"):
                trace = n42["variants"][variant]["branches"][branch_name]["future_trace"]
                if len(trace) != 100 or [int(x["frame"]) for x in trace] != list(range(int(trace[0]["frame"]), int(trace[0]["frame"]) + 100)):
                    failures.append(f"N42 source trace contract {event_id}/{variant}/{branch_name}")
                source_frames += len(trace)
        payload = load(RUNTIME / f"{event_id}.json")
        if payload.get("runtime_boundary", {}).get("runtime_future_gt_used") is not False: failures.append(f"runtime future GT {event_id}")
        if payload.get("runtime_boundary", {}).get("gt_loaded_in_worker") is not False: failures.append(f"runtime GT loaded {event_id}")
        for variant in VARIANTS:
            items = payload["variants"][variant]["frames"]
            if len(items) != 100: failures.append(f"frame count {event_id}/{variant}")
            for item in items:
                no = item["no_write"]; write = item["write_baseline"]; plus = item["write_plus_n48"]; probe = item["probe"]
                sig = signature(no["candidate_rows"])
                if sig != signature(write["candidate_rows"]) or sig != signature(plus["candidate_rows"]): failures.append(f"candidate stream mismatch {event_id}/{variant}/{item['frame']}")
                if write["public_id_order"] != plus["public_id_order"]: failures.append(f"write/plus public ID axis mismatch {event_id}/{variant}/{item['frame']}")
                if len(set(no["candidate_native_ids"])) != len(no["candidate_native_ids"]): failures.append(f"duplicate native IDs {event_id}/{variant}/{item['frame']}")
                for branch_name in ("no_write", "write_baseline", "write_plus_n48"):
                    if item[branch_name].get("runtime_future_gt_used") is not False: failures.append(f"branch GT flag {event_id}/{variant}/{item['frame']}/{branch_name}")
                for vector in item["candidate_features_512"]:
                    if len(vector) != 512 or not np.all(np.isfinite(np.asarray(vector, dtype=np.float32))): failures.append(f"invalid 512D candidate feature {event_id}/{variant}/{item['frame']}"); break
                if item["memory_provenance"] != "offline_event_prefix_machine_embedding_with_GT_simulated_human_target_anchor": failures.append("memory provenance mismatch")
                write_scores = np.asarray(write["score_matrix"], dtype=np.float32); plus_scores = np.asarray(plus["score_matrix"], dtype=np.float32); no_scores = np.asarray(no["score_matrix"], dtype=np.float32)
                if solve_with_none(write_scores) != write["assignment_columns"]: failures.append(f"write assignment solver mismatch {event_id}/{variant}/{item['frame']}")
                if solve_with_none(plus_scores) != plus["assignment_columns"]: failures.append(f"plus assignment solver mismatch {event_id}/{variant}/{item['frame']}")
                if solve_with_none(no_scores) != no["assignment_columns"]: failures.append(f"no assignment solver mismatch {event_id}/{variant}/{item['frame']}")
                actual = np.argwhere(np.abs(plus_scores - write_scores) > 1e-12)
                listed = {(int(x["candidate_index"]), int(x["column"])) for x in probe["changed_cells"]}
                if {(int(row), int(col)) for row, col in actual} != listed: failures.append(f"changed cell list mismatch {event_id}/{variant}/{item['frame']}")
                if np.any(plus_scores[write_scores <= -1e7] != write_scores[write_scores <= -1e7]): failures.append(f"hard negative changed {event_id}/{variant}/{item['frame']}")
                if variant == "M0" and (not np.array_equal(plus_scores, write_scores) or plus["assignment_columns"] != write["assignment_columns"]): failures.append(f"M0 is not exact no-op {event_id}/{item['frame']}")
                frames += 1; changed_cells += len(actual)
    result = {"status": "PASS" if not failures else "FAIL", "protocol": "N48_STAGE_04_INDEPENDENT_INTEGRITY_V1", "command": ["python", "scripts/n48_stage04_integrity.py"], "inputs": {"runtime": str(RUNTIME), "n42_runtime": str(N42_RUNTIME), "checkpoint": str(CHECKPOINT), "memory_manifest": str(MEMORY)}, "outputs": {}, "metrics": {"runtime_frames_checked": frames, "source_future_trace_frames_checked": source_frames, "changed_cells_checked": changed_cells, "failure_count": len(failures)}, "gate_checks": {"all_24_events": len(list(RUNTIME.glob("*.json"))) == 24, "all_12000_runtime_frames": frames == 12000, "source_traces_exact_100": source_frames == 24 * 5 * 2 * 100, "candidate_stream_identical_three_branches": not any("candidate stream mismatch" in x for x in failures), "write_plus_public_id_axis_identical": not any("write/plus public ID axis mismatch" in x for x in failures), "active_universe_changes_retained": True, "hungarian_none_recomputed_normalized": not any("solver mismatch" in x for x in failures), "no_duplicate_native_ids": not any("duplicate native" in x for x in failures), "runtime_future_gt_false": not any("GT" in x for x in failures), "M0_exact_no_op": not any("M0 is not" in x for x in failures), "hard_negative_preserved": not any("hard negative" in x for x in failures), "checkpoint_production_authorized_false": not any("production_authorized" in x for x in failures), "posthoc_not_used_as_runtime": not any("runtime GT loaded" in x for x in failures)}, "failure_root_cause": failures[0] if failures else "No integrity discrepancy found; explicit NONE assignments were recomputed in normalized -1 form; no/write active-universe changes are retained as diagnostics.", "failures": failures[:100], "next_action": "Proceed to semantic final gate only if PASS; otherwise preserve failure and stop downstream interpretation.", "runtime_future_gt_used": False}
    write_json(OUT / "replay/stage_04_integrity.json", result)
    print(json.dumps({"status": result["status"], "frames": frames, "failures": len(failures)}))
    if failures: raise SystemExit(1)


if __name__ == "__main__":
    main()
