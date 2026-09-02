#!/usr/bin/env python3
"""Freeze the N72R3R1 semantic-repair protocol without changing N72R3."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
N72R3_ROOT = Path("/data2/usr_for_deadline/SAM3_InterMOT_N72R3/worktree")
OUT = ROOT / "outputs/N72R3R1"
PROTOCOL = OUT / "protocol.json"
STATUS = OUT / "stage_status/stage_00_status.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def main() -> int:
    frozen = {
        "n72r3_final_report": N72R3_ROOT / "docs/N72R3_FINAL_REPORT.md",
        "n72r3_final_gate": N72R3_ROOT / "outputs/N72R3/n72r3_final_gate.json",
        "n72r3_protocol": N72R3_ROOT / "outputs/N72R3/protocol.json",
        "n72r3_event_manifest": N72R3_ROOT / "outputs/N72R3/simulation/real_event_manifest.json",
        "n72r3_effect_results": N72R3_ROOT / "outputs/N72R3/effect_replay/attempt1/ccam_paired_replay_results.json",
        "n72r3_runtime_manifest": N72R3_ROOT / "outputs/N72R3/effect_replay/attempt1/runtime_manifest.json",
        "n72r3_public_assignment": ROOT / "sam3_intermot/association/public_assignment.py",
        "n72r3_effect_replay_source": ROOT / "scripts/n72r3_stage20_22_effect_replay.py",
    }
    missing = [str(path) for path in frozen.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("missing frozen inputs: " + ", ".join(missing))
    source_hashes = {name: {"path": str(path), "sha256": sha256(path)} for name, path in frozen.items()}
    n72r3_gate = json.loads(frozen["n72r3_final_gate"].read_text(encoding="utf-8"))
    old_effect = json.loads(frozen["n72r3_effect_results"].read_text(encoding="utf-8"))
    event_manifest = json.loads(frozen["n72r3_event_manifest"].read_text(encoding="utf-8"))
    events = list(event_manifest.get("events", []))
    protocol = {
        "schema_version": "N72R3R1_SEMANTIC_REPAIR_PROTOCOL_V1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "name": "N72R3R1_SEMANTIC_REPAIR_RERUN",
        "purpose": "Repair exact-NONE assignment, metric direction, crossing taxonomy and sequence bootstrap only.",
        "frozen_input": {
            "source_experiment": "N72R3",
            "event_count": len(events),
            "independent_sequence_count": len({str(item.get("sequence")) for item in events}),
            "event_ids": sorted(str(item["event_id"]) for item in events),
            "candidate_stream_and_official_artifacts_unchanged": True,
            "old_effect_result_status": old_effect.get("status"),
            "old_research_gate": n72r3_gate.get("research_gate"),
        },
        "semantic_repairs": {
            "explicit_none_solver": {
                "implementation": "sam3_intermot.association.public_assignment.solve_exact_public_assignment",
                "wrapper": "sam3_intermot.association.effect_assignment.solve_effect_assignment",
                "matrix_orientation": "state_x_candidate_transposed_to_candidate_x_state",
                "none_score": 0.0,
                "outer_birth_after_existing_id_plus_none": True,
            },
            "primary_metric": "identity_error_reduction = baseline_error - treatment_error",
            "secondary_metric": "delta_iou = treatment_iou - baseline_iou",
            "optional_composite": "0.5*delta_iou + 0.5*(treatment_correct-baseline_correct)",
            "crossing_taxonomy": "AssignmentChangeType with true crossings separated from directional changes",
            "bootstrap": {"unit": "independent_sequence", "within_sequence": "mean_all_events", "seed": 7202, "repetitions": 2000},
        },
        "unchanged_protocol": {
            "variants": old_effect.get("variants", []),
            "horizons": old_effect.get("horizons", []),
            "event_actions_and_frames": True,
            "candidate_stream": True,
            "memory_weights": True,
            "human_roi_feature": True,
            "checkpoint": True,
            "thresholds_and_age": True,
            "runtime_future_gt_used": False,
            "interaction_source": "simulated_from_gt",
            "real_human_tape": False,
        },
        "prohibited_before_stage_05": [
            "training",
            "new_model",
            "memory_weight_change",
            "candidate_recovery",
            "event_expansion",
            "future_effect_claim",
        ],
        "frozen_input_hashes": source_hashes,
        "historical_evidence_read_only": True,
        "n72r3_root_unchanged": True,
    }
    atomic_json(PROTOCOL, protocol)
    atomic_json(
        STATUS,
        {
            "schema_version": "N72R3R1_STAGE_STATUS_V1",
            "stage": "00_PROTOCOL_FREEZE",
            "status": "PASS_PROTOCOL_FROZEN",
            "protocol": str(PROTOCOL),
            "event_count": len(events),
            "independent_sequence_count": len({str(item.get("sequence")) for item in events}),
            "runtime_future_gt_used": False,
            "historical_evidence_read_only": True,
            "scientific_result": "NO_SCIENTIFIC_RESULT",
        },
    )
    print(json.dumps({"status": "PASS_PROTOCOL_FROZEN", "protocol": str(PROTOCOL)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
