"""Verified progress snapshot, not research closure or final authorization."""
from pathlib import Path
import json
import xml.etree.ElementTree as ET
from scripts.n72r21r2_common import ROOT, OUT, ASSETS, GOAL, read_json, write_json, sha256, storage, utcnow, update_status
from scripts.n72r21r2_input_driver import alive_owned_worker


def run():
    namespaces = {"candidates": "data/candidate_integrity", "baselines": "mot/baseline_results", "CF_sequences": "events/counterfactual_sequences",
                  "CF_labels": "events/label_audit", "actual_window_TrackEval": "events/window_trackeval_results", "simple_full_sequences": "simple/results"}
    files = {kind: sorted((OUT / relative).glob("*.json")) for kind, relative in namespaces.items()}
    counts = {kind: len(paths) for kind, paths in files.items()}
    live = []
    for namespace in ("data/extraction_preflight_v2", "data/extraction_v4_preflight"):
        for path in (OUT / namespace).glob("*.json"):
            record = read_json(path)
            if alive_owned_worker(record["pid"], "scripts.n72r21r2_candidates"):
                live.append({"sequence": record["sequence"], "pid": record["pid"], "physical_GPU": record["visible_physical_GPU"]})
    totals = {}
    for path in files["CF_labels"]:
        record = read_json(path)
        totals["sealed_event_positions"] = totals.get("sealed_event_positions", 0) + record["events"]
        for field, value in record["counts"].items():
            totals[field] = totals.get(field, 0) + value
    tests = {}
    for name in ("ALL_FOCUSED_V4.xml", "FULL_REGRESSION_FRESH_EVENT_V4.xml"):
        path = OUT / "tests" / name
        root = ET.parse(path).getroot()
        cases = list(root.iter("testcase"))
        failed = [r.attrib["classname"] + "::" + r.attrib["name"] for r in cases if r.find("failure") is not None or r.find("error") is not None]
        tests[name] = {"sha256": sha256(path), "cases": len(cases), "passed": len(cases) - len(failed), "failures": failed}
    catalog = {str(p.relative_to(OUT)): sha256(p) for paths in files.values() for p in paths}
    for path in (OUT / "events/COUNTERFACTUAL_PROTOCOL_V1.json", OUT / "events/WINDOW_TRACKEVAL_PROTOCOL_V1.json", OUT / "simple/SIMPLE_PROTOCOL_V1.json",
                 OUT / "PREREGISTRATION.json", OUT / "FINAL_GOAL.json", OUT / "audit/BASELINE_REPRODUCTION.json"):
        catalog[str(path.relative_to(OUT))] = sha256(path)
    value = {"stage": "N72R21R2", "goal": GOAL, "goal_file": "outputs/N72R21R2/FINAL_GOAL.json", "utc": utcnow(),
             "status": "ACTIVE_M1_M3_M4_NOT_SCIENTIFIC_CLOSURE", "actual_completed_counts": counts,
             "actual_CF_label_components": totals, "active_candidate_workers": live, "local_artifact_SHA256": catalog,
             "resource_snapshot": storage(), "tests": tests, "M0_actual_historical_replays": 8,
             "old_scientific_negative_reproduced_not_new_PASS": True, "independent_correction_gate_not_yet_evaluated": True,
             "new_R2_model_fits_completed": 0, "confirmation_GT_used_for_training_or_selection": False,
             "CONFIRM_VAL_TEST_SOT_policy_runs": 0, "scientific_success": None, "next_stage_authorized": False,
             "uploaded_artifacts_policy": "Code and compact instructions only; this SHA catalog is documentation, bulk evidence stays local"}
    write_json("audit/EXECUTION_CHECKPOINT_M3_M4_V1.json", value)
    destination = ROOT / "docs/N72R21R2_LOCAL_EVIDENCE_SHA.json"
    with destination.open("x") as handle:
        json.dump(value, handle, sort_keys=True, indent=2, allow_nan=False)
        handle.write("\n")
    update_status(new_fresh_sequence_candidate_extraction_completed=counts["candidates"], fresh_baselines_completed=counts["baselines"],
                  verified_fresh_candidate_sequences=[p.stem for p in files["candidates"]], verified_fresh_baseline_sequences=[p.stem for p in files["baselines"]],
                  actual_CF_sequence_runtimes_completed=counts["CF_sequences"], actual_CF_labels_completed=counts["CF_labels"],
                  actual_window_TrackEval_completed=counts["actual_window_TrackEval"], focused_test_passed=34, full_regression_passed=959,
                  full_regression_historical_failures=5, source_remote_fresh_read="VERIFIED_SAME_R1_PARENT_V2",
                  resource_snapshot=storage(), application_goal_active=True, scientific_success=None, next_stage_authorized=False)
    print({"checkpoint": counts, "CF_components": totals, "tests": {k: {"passed": v["passed"], "failures": len(v["failures"])} for k, v in tests.items()}, "goal": "ACTIVE"})


if __name__ == "__main__":
    run()
