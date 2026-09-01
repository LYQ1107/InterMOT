#!/usr/bin/env python3
"""Targeted regression: repaired N46 diagnostics must reproduce frozen N45 counts."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNTIME = ROOT / "outputs/n46/diagnosis_repair2/events"
POSTHOC = ROOT / "outputs/n46/diagnosis_repair2/posthoc_events"
N45_RUNTIME = ROOT / "outputs/n45/replay/runtime"
N45_RESULT = ROOT / "outputs/n45/replay/attribution_results.json"
OUT = ROOT / "outputs/n46/diagnosis_final_repair2/repair2_alignment_regression.json"
VARIANTS = ("M0", "M1", "M2", "M3", "M4")
HORIZONS = (20, 50, 100)


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    n45_result = load(N45_RESULT)
    runtime_files = sorted(RUNTIME.glob("*.json")); posthoc_files = sorted(POSTHOC.glob("*.posthoc.json"))
    checks = {"runtime_event_count_24": len(runtime_files) == 24, "posthoc_event_count_24": len(posthoc_files) == 24, "m0_no_sidecar": True, "runtime_totals_match_n45": True, "assignment_decomposition_matches_n45": True, "all_runtime_future_gt_false": True, "posthoc_marked_gt_only": True, "posthoc_available_frame_schema": True}
    runtime_totals = {v: {"proposals_considered": 0, "proposals_selected": 0, "selected_but_no_assignment_change": 0, "changed_cells": 0, "changed_assignments": 0} for v in VARIANTS}
    frame_keys = ("assignment_changed", "assignment_change_correct", "assignment_change_incorrect", "assignment_change_neutral", "assignment_no_change")
    expected_keys = ("assignment_change_count", "assignment_change_correct_count", "assignment_change_incorrect_count", "assignment_change_neutral_count", "assignment_no_change_count")
    actual_assignment = {v: {h: {k: 0 for k in frame_keys} for h in HORIZONS} for v in VARIANTS}
    expected_runtime = {"M0": {"proposals_considered": 0, "proposals_selected": 0, "selected_but_no_assignment_change": 0, "changed_cells": 0, "changed_assignments": 0}, "M1": {"proposals_considered": 7, "proposals_selected": 4, "selected_but_no_assignment_change": 1, "changed_cells": 4, "changed_assignments": 6}, "M2": {"proposals_considered": 11, "proposals_selected": 4, "selected_but_no_assignment_change": 2, "changed_cells": 4, "changed_assignments": 4}, "M3": {"proposals_considered": 5, "proposals_selected": 3, "selected_but_no_assignment_change": 1, "changed_cells": 3, "changed_assignments": 4}, "M4": {"proposals_considered": 5, "proposals_selected": 3, "selected_but_no_assignment_change": 1, "changed_cells": 3, "changed_assignments": 4}}
    for path in runtime_files:
        payload = load(path)
        checks["all_runtime_future_gt_false"] &= payload.get("runtime_future_gt_used") is False
        for v in VARIANTS:
            frames = payload["variants"][v]
            checks["all_runtime_future_gt_false"] &= len(frames) == 100 and all(x.get("runtime_future_gt_used") is False for x in frames)
            for x in frames:
                runtime_totals[v]["proposals_considered"] += len(x["proposals"]); runtime_totals[v]["proposals_selected"] += int(x["selected_count"]); runtime_totals[v]["selected_but_no_assignment_change"] += int(x["selected_but_no_assignment_change"]); runtime_totals[v]["changed_cells"] += len(x["changed_cells"]); runtime_totals[v]["changed_assignments"] += int(x["assignment_changed_count"])
                if v == "M0":
                    checks["m0_no_sidecar"] &= not x["proposals"] and not x["changed_cells"] and int(x["assignment_changed_count"]) == 0
    checks["runtime_totals_match_n45"] = runtime_totals == expected_runtime
    for path in posthoc_files:
        payload = load(path)
        for v in VARIANTS:
            frames = payload["variants"][v]
            checks["posthoc_marked_gt_only"] &= len(frames) == 100
            for h in HORIZONS:
                for x in frames[:h]:
                    if x.get("gt_available", False):
                        if any(key not in x for key in actual_assignment[v][h]):
                            checks["posthoc_available_frame_schema"] = False
                            continue
                        for key in actual_assignment[v][h]:
                            actual_assignment[v][h][key] += int(x[key])
                    else:
                        checks["posthoc_marked_gt_only"] &= x.get("runtime_future_gt_used") is False
    for v in VARIANTS:
        for h in HORIZONS:
            expected = n45_result["effects"]["incremental"][v][str(h)]
            for frame_key, expected_key in zip(frame_keys, expected_keys):
                if actual_assignment[v][h][frame_key] != int(expected[expected_key]):
                    checks["assignment_decomposition_matches_n45"] = False
    result = {"status": "PASS" if all(checks.values()) else "FAIL", "protocol": "N46_STAGE_02_REPAIR2_ALIGNMENT_REGRESSION_V1", "inputs": {"repair2_runtime": str(RUNTIME), "repair2_posthoc": str(POSTHOC), "n45_runtime": str(N45_RUNTIME), "n45_result": str(N45_RESULT)}, "outputs": {"regression": str(OUT)}, "metrics": {"runtime_totals": runtime_totals, "expected_runtime_totals": expected_runtime, "assignment_decomposition": actual_assignment, "runtime_rows": len(runtime_files) * 5 * 100, "posthoc_rows": len(posthoc_files) * 5 * 100}, "checks": checks, "failure_root_cause": "Repair2 must reproduce the frozen N45 attribution contract; M0 is a no-sidecar control and only M1-M4 apply N44.", "next_action": "Use the repair2 diagnosis only after the full integrity check passes; preserve repair1 as the caught-failure record.", "runtime_future_gt_used": False, "gt_loaded_posthoc": True}
    OUT.parent.mkdir(parents=True, exist_ok=True); OUT.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8"); print(json.dumps({"status": result["status"], "output": str(OUT)}))
    if result["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
