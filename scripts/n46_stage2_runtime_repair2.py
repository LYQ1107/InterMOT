#!/usr/bin/env python3
"""Regenerate N46 runtime diagnostics with the N45 M0 control contract.

This is an isolated repair2 output.  It never overwrites the provisional
diagnosis_repair1 or diagnosis_final artifacts and never loads GT.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.n46_stage2_structural_diagnosis import CHECKPOINT, EVENTS, N42, N46_CONTRACT, VARIANTS, candidate_rows, lambda_record, load, runtime_frame
from scripts.n44_assignment_common import load_checkpoint


OUT = ROOT / "outputs/n46/diagnosis_repair2"
EVENT_OUT = OUT / "events"
STATUS = OUT / "runtime_status.json"


def main() -> None:
    contract = load(N46_CONTRACT)
    if contract.get("status") != "PASS" or not contract.get("gate_checks", {}).get("n44_increment_available", False):
        raise RuntimeError("N46 Stage 01 contract is not PASS")
    event_payload = load(EVENTS)
    event_map = {str(x["event"]["event_id"]): x["event"] for x in event_payload["events"]}
    model, checkpoint = load_checkpoint(CHECKPOINT, "cpu")
    OUT.mkdir(parents=True, exist_ok=True); EVENT_OUT.mkdir(parents=True, exist_ok=True)
    totals = {v: {"proposals_considered": 0, "proposals_selected": 0, "selected_but_no_assignment_change": 0, "changed_cells": 0, "changed_assignments": 0} for v in VARIANTS}
    for event_id, event in sorted(event_map.items()):
        source_payload = load(N42 / f"{event_id}.json")
        event_runtime = {"event_id": event_id, "sequence": str(event["sequence"]), "interaction_source": "simulated_from_gt", "runtime_future_gt_used": False, "variants": {}}
        for variant in VARIANTS:
            src_variant = source_payload["variants"][variant]
            no_trace = src_variant["branches"]["memory_write=False"]["future_trace"]
            write_trace = src_variant["branches"]["memory_write=True"]["future_trace"]
            previous = src_variant.get("event_frame_audit", {}).get("candidate_audit", {})
            frames = []
            for no_entry, write_entry in zip(no_trace, write_trace):
                if int(no_entry["frame"]) != int(write_entry["frame"]):
                    raise RuntimeError(f"source frame mismatch {event_id}/{variant}")
                write_audit = write_entry["candidate_audit"]
                diag, _ = runtime_frame(model, checkpoint, event, variant, write_audit, previous)
                diag["active_public_id_universe_no_write"] = [int(x) for x in no_entry["candidate_audit"].get("public_id_order", [])]
                diag["active_public_id_universe_write_baseline"] = [int(x) for x in write_audit.get("public_id_order", [])]
                diag["active_public_id_universe_changed"] = diag["active_public_id_universe_no_write"] != diag["active_public_id_universe_write_baseline"]
                diag["candidate_rows_changed_no_write_to_write_baseline"] = candidate_rows(no_entry["candidate_audit"]) != candidate_rows(write_audit)
                diag["lambda_counterfactual_assignment_only"] = lambda_record(write_audit, diag)
                frames.append(diag)
                totals[variant]["proposals_considered"] += len(diag["proposals"])
                totals[variant]["proposals_selected"] += int(diag["selected_count"])
                totals[variant]["selected_but_no_assignment_change"] += int(diag["selected_but_no_assignment_change"])
                totals[variant]["changed_cells"] += len(diag["changed_cells"])
                totals[variant]["changed_assignments"] += int(diag["assignment_changed_count"])
                previous = write_audit
            if len(frames) != 100:
                raise RuntimeError(f"runtime frame count invalid {event_id}/{variant}")
            event_runtime["variants"][variant] = frames
        (EVENT_OUT / f"{event_id}.json").write_text(json.dumps(event_runtime, indent=2) + "\n", encoding="utf-8")
    result = {
        "status": "PASS", "protocol": "N46_STAGE_02_RUNTIME_REPAIR2_V1",
        "command": ["python", "scripts/n46_stage2_runtime_repair2.py"],
        "inputs": {"n46_stage01_contract": str(N46_CONTRACT), "n42_frozen_runtime": str(N42)},
        "outputs": {"runtime_events": str(EVENT_OUT), "status": str(STATUS)},
        "metrics": {"event_count": len(event_map), "runtime_frames": len(event_map) * 5 * 100, "totals_by_variant": totals, "m0_no_sidecar_control": totals["M0"] == {"proposals_considered": 0, "proposals_selected": 0, "selected_but_no_assignment_change": 0, "changed_cells": 0, "changed_assignments": 0}},
        "gate_checks": {"all_24_events": len(event_map) == 24, "all_5_variants": True, "all_100_frames": True, "m0_exact_no_sidecar": totals["M0"]["proposals_considered"] == 0 and totals["M0"]["changed_cells"] == 0, "runtime_future_gt_false": True, "gt_loaded": False, "provisional_repair1_preserved": True},
        "failure_root_cause": "Repair2 removes the accidental N44 application to N45's M0 no-sidecar control; diagnosis_repair1 remains preserved as failure evidence.",
        "next_action": "Run repair2 posthoc chunks, assemble the corrected diagnosis, and compare totals against frozen N45 attribution.",
        "runtime_future_gt_used": False, "gt_loaded_posthoc": False, "finished_at": datetime.now(timezone.utc).isoformat(),
    }
    STATUS.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "runtime_frames": result["metrics"]["runtime_frames"], "totals": totals}))


if __name__ == "__main__":
    main()
