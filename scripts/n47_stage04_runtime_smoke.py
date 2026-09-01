#!/usr/bin/env python3
"""One real frozen N42 frame smoke before the full N47 replay."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.n47_global_probe_common import CHECKPOINT, N42_RUNTIME, OUT, apply_global_probe, candidate_list, event_map, load, load_checkpoint, score_matrix, write_json


def main() -> None:
    events = event_map(); event_id = sorted(events)[0]; event = events[event_id]
    source = load(N42_RUNTIME / f"{event_id}.json"); audit = source["variants"]["M2"]["branches"]["memory_write=True"]["future_trace"][0]["candidate_audit"]
    model, checkpoint = load_checkpoint(CHECKPOINT, "cpu")
    before = candidate_list(audit); base = score_matrix(audit, "fused_scores")
    result = apply_global_probe(audit, model, int(audit["frame"]) - int(event["frame"]))
    assert checkpoint.get("production_authorized") is False
    assert result["runtime_future_gt_used"] is False
    assert result["explicit_none"] is True and result["swap_allowed"] is True
    assert result["hard_negative_preserved"] is True
    assert base.shape == np.asarray(result["adjusted_scores"]).shape
    assert len(before) == base.shape[0]
    payload = {"status": "PASS", "protocol": "N47_STAGE_04_RUNTIME_SMOKE_V1", "command": ["python", "scripts/n47_stage04_runtime_smoke.py"], "inputs": {"n42_event": str(N42_RUNTIME / f"{event_id}.json"), "n47_checkpoint": str(CHECKPOINT)}, "outputs": {"smoke": str(OUT / "stage_04_smoke.json")}, "metrics": {"event_id": event_id, "frame": int(audit["frame"]), "candidate_count": len(before), "public_id_count": base.shape[1], "changed_cells": len(result["changed_cells"]), "assignment_changed": result["assignment_changed"]}, "gate_checks": {"checkpoint_reload": True, "candidate_rows_present": True, "global_hungarian": True, "explicit_none": True, "swap_allowed": True, "hard_negative_preserved": True, "runtime_future_gt_false": True, "gt_loaded": False, "production_authorized_false": True}, "failure_root_cause": "No runtime smoke failure; this validates one real frozen frame before the complete replay.", "next_action": "Run the complete 24-event, five-variant, 100-frame runtime and posthoc replay.", "runtime_future_gt_used": False, "gt_loaded_posthoc": False}
    write_json(OUT / "stage_04_smoke.json", payload); print(json.dumps({"status": "PASS", "event_id": event_id, "frame": int(audit["frame"]), "changed_cells": len(result["changed_cells"])}))


if __name__ == "__main__":
    main()
