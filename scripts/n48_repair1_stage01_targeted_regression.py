#!/usr/bin/env python3
"""Targeted regression for repaired N48 Stage-01 accounting semantics."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main() -> None:
    path = ROOT / "outputs/n48/repair1/diagnosis/n47_m2_structural_diagnosis.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    metrics = payload["metrics"]
    per_sequence = metrics["per_sequence"]
    closure = {sequence: item["assignment_changes"] == item["correct_changes"] + item["incorrect_changes"] + item["neutral_changes"] for sequence, item in per_sequence.items()}
    frames = [json.loads(line) for line in (ROOT / "outputs/n48/repair1/diagnosis/n47_m2_frame_diagnostics.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    none_bad = [row for row in frames if row["none_involved"] and not row["assignment_changed_frame"]]
    names = set(frames[0]) if frames else set()
    checks = {
        "all_sequence_closures": bool(closure) and all(closure.values()),
        "frame_target_naming_explicit": "assignment_changed_frame" in names and "assignment_change_class_frame" in names,
        "legacy_frame_names_not_used": "assignment_changed" not in names and "assignment_change_class" not in names,
        "none_only_changed_frames": not none_bad,
        "oracle_gap_invalid_non_comparable": metrics["oracle_required_total_score_gap"]["status"] == "INVALID_NON_COMPARABLE",
        "frozen_counts_reconciled": metrics["assignment_change_context"] == {"assignment_changes": 455, "correct": 24, "incorrect": 40, "neutral": 391, "no_change": 1945, "none_involved": 1},
    }
    result = {"status": "PASS" if all(checks.values()) else "FAIL", "protocol": "N48_R1_STAGE_01_TARGETED_REGRESSION_V1", "command": ["python", "scripts/n48_repair1_stage01_targeted_regression.py"], "inputs": {"diagnosis": str(path)}, "outputs": {}, "metrics": {"sequence_count": len(per_sequence), "frame_count": len(frames), "none_involved_changes": metrics["assignment_change_context"]["none_involved"]}, "gate_checks": checks, "failure_root_cause": "Repaired Stage-01 target-level assignment closure and changed-row NONE semantics are valid." if all(checks.values()) else "Stage-01 accounting regression failed.", "next_action": "Proceed to N48-R1 training only if PASS." if all(checks.values()) else "Preserve failure and do not train.", "runtime_future_gt_used": False}
    out = ROOT / "outputs/n48/repair1/stage_01_targeted_regression.json"; out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "checks": checks}))
    if result["status"] != "PASS": raise SystemExit(1)


if __name__ == "__main__":
    main()
