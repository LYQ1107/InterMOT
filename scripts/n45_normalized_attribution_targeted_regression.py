#!/usr/bin/env python3
"""Integrity regression for the isolated normalized N45 attribution result."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULT = ROOT / "outputs/n46/n45_attribution_repair/normalized_attribution_results.json"
POSTHOC = ROOT / "outputs/n46/n45_attribution_repair/posthoc_events"
OUT = ROOT / "outputs/n46/n45_attribution_repair/targeted_regression.json"
VARIANTS = ("M0", "M1", "M2", "M3", "M4")
HORIZONS = (20, 50, 100)


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    result = load(RESULT); files = sorted(POSTHOC.glob("*.json")); checks = {"result_status_pass": result.get("status") == "PASS", "event_count_24": result.get("event_count") == 24, "posthoc_event_count_24": len(files) == 24, "five_variants": result.get("variant_count") == 5, "all_horizons": result.get("horizons") == [20, 50, 100], "runtime_future_gt_false": result.get("protocol", {}).get("runtime_future_gt_used") is False and result.get("runtime_validation", {}).get("runtime_future_gt_used") is False, "gt_after_runtime_validation": result.get("protocol", {}).get("gt_loaded_only_after_runtime_validation") is True, "axis_normalization_declared": result.get("axis_reconciliation", {}).get("both_assignment_maps_normalized") is True, "old_result_unmodified": result.get("axis_reconciliation", {}).get("old_n45_result_modified") is False, "equal_sequence_bootstrap": result.get("protocol", {}).get("bootstrap") == "sequence_mean_then_equal_sequence_cluster_bootstrap", "simulated_provenance": result.get("interaction_source") == "simulated_from_gt" and result.get("real_human_tape_created") is False, "standard_mot_not_computable": result.get("id_switch_metric") == "NOT_COMPUTABLE_NO_COMPLETE_TRACKEVAL_SEQUENCE_INPUT"}
    required = ("identity_utility", "target_iou_delta", "future_identity_error_reduction", "recorrection_proxy_reduction", "assignment_change_count", "assignment_change_correct_count", "assignment_change_incorrect_count", "assignment_change_neutral_count", "assignment_no_change_count", "untouched_regression")
    per_event = 0
    for path in files:
        payload = load(path); per_event += 1
        checks["runtime_future_gt_false"] &= payload.get("runtime_future_gt_used") is False
        checks["gt_after_runtime_validation"] &= payload.get("gt_loaded_posthoc") is True
        checks["five_variants"] &= set(payload.get("variants", {})) == set(VARIANTS)
        for v in VARIANTS:
            checks["all_horizons"] &= set(payload["variants"][v].get("horizons", {})) == {str(h) for h in HORIZONS}
            for h in HORIZONS:
                for effect in ("memory_effect_no_write_to_write_baseline", "n44_incremental_effect_write_baseline_to_write_plus_n44"):
                    item = payload["variants"][v]["horizons"][str(h)][effect]
                    checks["all_horizons"] &= all(k in item for k in required)
                    checks["runtime_future_gt_false"] &= all(frame.get("runtime_future_gt_used", False) is False for frame in item.get("frame_details", [])) if item.get("frame_details") else checks["runtime_future_gt_false"]
    # Regression focuses on the corrected axis semantics and known frozen M2
    # outcome, not on selecting a new gate or reading holdout.
    m2 = result["effects"]["incremental"]["M2"]
    checks["corrected_m2_utility_zero"] = all(float(m2[str(h)]["identity_utility"]) == 0.0 for h in HORIZONS)
    checks["corrected_m2_changes_are_neutral"] = all(int(m2[str(h)]["assignment_change_correct_count"]) == 0 and int(m2[str(h)]["assignment_change_incorrect_count"]) == 0 and int(m2[str(h)]["assignment_change_neutral_count"]) == int(m2[str(h)]["assignment_change_count"]) for h in HORIZONS)
    checks["corrected_m2_change_counts"] = [int(m2[str(h)]["assignment_change_count"]) for h in HORIZONS] == [1, 2, 2]
    result_payload = {"status": "PASS" if all(checks.values()) else "FAIL", "protocol": "N45_NORMALIZED_ATTRIBUTION_TARGETED_REGRESSION_V1", "inputs": {"normalized_result": str(RESULT), "normalized_posthoc": str(POSTHOC)}, "outputs": {"regression": str(OUT)}, "metrics": {"posthoc_events_checked": per_event, "corrected_m2_increment": {str(h): {k: m2[str(h)][k] for k in ("identity_utility", "assignment_change_count", "assignment_change_correct_count", "assignment_change_incorrect_count", "assignment_change_neutral_count")} for h in HORIZONS}, "axis_mismatch_frames_recorded": result["axis_reconciliation"]["write_source_frames_with_candidate_public_id_axis_mismatch"]}, "checks": checks, "failure_root_cause": "Both N45 attribution branches are compared through assignment columns and the active public-ID axis; raw axis-outside candidate_public_ids are retained only as a diagnosed legacy mismatch.", "next_action": "Use normalized attribution for scientific interpretation; keep old N45 result as legacy and retain real-input gates.", "runtime_future_gt_used": False, "gt_loaded_posthoc": True}
    OUT.write_text(json.dumps(result_payload, indent=2) + "\n", encoding="utf-8"); print(json.dumps({"status": result_payload["status"], "output": str(OUT)}))
    if result_payload["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
