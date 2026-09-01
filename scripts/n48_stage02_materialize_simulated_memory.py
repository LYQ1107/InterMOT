#!/usr/bin/env python3
"""Materialize simulated prefix memory input before the GT-free runtime."""

from __future__ import annotations

import sys
from pathlib import Path

import json

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.n36_real_eval_common import DATA_ROOT  # noqa: E402
from scripts.n47_global_probe_common import N43_MAP, event_map, load, write_json  # noqa: E402
from scripts.n48_assignment_common import N36_FRAMES, N47_RUNTIME, load_n36_sequence, make_memory_snapshot  # noqa: E402


def main() -> None:
    events = event_map(); mapping = load(N43_MAP)["public_to_gt_mapping"]
    sequences = sorted({str(item["sequence"]) for item in events.values()})
    from sam3_intermot.datasets.dancetrack import DanceTrackDataset
    dataset = DanceTrackDataset(str(DATA_ROOT), sequences=sequences, split="train")
    gt = {sequence: dataset.load_gt(sequence) for sequence in sequences}
    n36 = {sequence: load_n36_sequence(sequence) for sequence in sequences}
    output = {"schema": "N48_SIMULATED_EVENT_MEMORY_MANIFEST_V1", "status": "PASS", "interaction_source": "simulated_from_gt", "runtime_future_gt_used": False, "gt_loaded_for_offline_simulation": True, "events": {}}
    for event_id, event in sorted(events.items()):
        sequence = str(event["sequence"]); first = load(N47_RUNTIME / f"{event_id}.json")["variants"]["M2"]["frames"][0]["write_baseline"]; pids = [int(x) for x in first["public_id_order"]]
        memories, valid = make_memory_snapshot(event, pids, n36[sequence], gt[sequence], first)
        output["events"][event_id] = {"sequence": sequence, "event_frame": int(event["frame"]), "public_id_order_at_first_future_frame": pids, "memory_vectors": {str(pid): memories[pid].astype(float).tolist() for pid in pids}, "memory_valid": {str(pid): bool(valid[pid]) for pid in pids}, "memory_source": "offline_event_prefix_machine_embedding_with_GT_simulated_human_target_anchor", "runtime_future_gt_used": False}
    path = ROOT / "outputs/n48/training/simulated_event_memory.json"
    write_json(path, output)
    print(json.dumps({"status": "PASS", "events": len(output["events"]), "runtime_future_gt_used": False, "interaction_source": "simulated_from_gt"}))


if __name__ == "__main__":
    main()
