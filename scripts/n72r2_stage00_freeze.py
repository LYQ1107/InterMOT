"""Freeze the N72R2 protocol and protected-input inventory.

This script is deliberately independent of the source checkout.  N72R2 runs in
an isolated worktree copied from the proven N72R1 code; historical outputs and
the official SAM3 checkout are referenced by hash and are never rewritten.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parents[1]
N72R2 = ROOT / "outputs" / "N72R2"
SOURCE = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/Interactive/SAM3_InterMOT")
N72R1 = Path("/data2/usr_for_deadline/SAM3_InterMOT_N72R1")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, sort_keys=True, allow_nan=False)
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
        dfd = os.open(path.parent, os.O_DIRECTORY)
        try:
            os.fsync(dfd)
        finally:
            os.close(dfd)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def file_record(path: Path) -> dict[str, object]:
    if not path.is_file():
        return {"path": str(path), "exists": False}
    stat = path.stat()
    return {
        "path": str(path),
        "exists": True,
        "size": stat.st_size,
        "sha256": sha256(path),
    }


def records(paths: Iterable[Path]) -> list[dict[str, object]]:
    return [file_record(path) for path in paths]


def git_state(path: Path) -> dict[str, object]:
    try:
        p = subprocess.run(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError as exc:
        return {"is_repository": False, "error": str(exc)}
    return {
        "is_repository": p.returncode == 0,
        "head": p.stdout.strip() if p.returncode == 0 else None,
        "stderr": p.stderr.strip() if p.returncode else None,
    }


def main() -> None:
    N72R2.mkdir(parents=True, exist_ok=True)
    now = datetime.now(timezone.utc).isoformat()
    historical = [
        N72R1 / "reports/N72R1_FINAL_REPORT.md",
        N72R1 / "n72r1_final_status.json",
        N72R1 / "status/stage16_status.json",
        N72R1 / "audits/stage16_six_window_integrity.json",
        SOURCE / "AGENTS.md",
        SOURCE / "docs/N72_FINAL_REPORT.md",
        SOURCE / "research_log.md",
    ]
    key_worktree = [
        ROOT / "sam3_intermot/association/state_manager.py",
        ROOT / "sam3_intermot/association/assignment_sidecar.py",
        ROOT / "sam3_intermot/tracking/track_manager.py",
        ROOT / "sam3_intermot/identity/registry.py",
        ROOT / "sam3_intermot/identity/namespace.py",
        ROOT / "sam3_intermot/backend/base.py",
        ROOT / "sam3_intermot/backend/sam3_backend.py",
        ROOT / "sam3_intermot/interaction/continuous_observer.py",
        ROOT / "scripts/n72r1_stage16_six_window_export.py",
    ]
    checkpoint_root = SOURCE / "checkpoints"
    checkpoint_files = sorted(p for p in checkpoint_root.rglob("*") if p.is_file())
    protected = {
        "schema_version": "N72R2_PROTECTION_MANIFEST_V1",
        "created_at_utc": now,
        "source_root": str(SOURCE),
        "n72r1_root": str(N72R1),
        "n72r2_root": str(ROOT.parent),
        "source_git": git_state(SOURCE),
        "historical_inputs": records(historical),
        "worktree_baseline": records(key_worktree),
        "checkpoint_inventory": records(checkpoint_files),
        "third_party_root": str(SOURCE / "third_party/sam3"),
        "policy": {
            "historical_outputs_read_only": True,
            "source_root_not_modified": True,
            "third_party_sam3_not_modified": True,
            "checkpoint_not_modified": True,
            "all_new_artifacts_under": str(N72R2),
        },
    }
    protection_path = N72R2 / "protection_manifest.json"
    atomic_json(protection_path, protected)

    protocol = {
        "schema_version": "N72R2_PROTOCOL_V1",
        "name": "N72R2_PUBLIC_ID_CLOSURE_AND_AUTONOMOUS_GT_SIMULATED_EFFECT_LOOP",
        "created_at_utc": now,
        "source_root": str(SOURCE),
        "isolated_worktree": str(ROOT),
        "output_root": str(N72R2),
        "interaction_source": "simulated_from_gt",
        "real_human_tape_used": False,
        "frozen_inputs": {
            "base_code": "N72R1 proven worktree; historical evidence is read-only",
            "candidate_definition": "N72R1 Candidate V2, no candidate-generation change",
            "checkpoint": file_record(SOURCE / "checkpoints/sam3.1_mirror/sam3.1_multiplex.pt"),
            "n72r1_evidence": records(historical),
        },
        "runtime_causal_contract": {
            "predict_y_pre_before_gt_read": True,
            "current_gt_read_only_after_prediction_frozen": True,
            "future_gt_runtime_reads": False,
            "future_gt_model_decision": False,
            "future_gt_scheduler": False,
            "event_frame_memory_read": False,
            "first_memory_read_offset": 1,
            "posthoc_gt_only_after_runtime_artifact_frozen": True,
        },
        "public_authority_contract": {
            "authority_order": [
                "TrackManager.final_mot_track_id",
                "IdentityNamespace.lineage_to_public",
                "proven_transaction",
            ],
            "association_state_id_is_public_id": False,
            "unmapped_rows_excluded_from_efficacy_denominator": True,
            "unmapped_rows_counted": True,
            "mapping_coverage_required_for_eligible_rows": 1.0,
        },
        "fixed_windows": {
            "initial_overlap_smoke": {"window_count": 1, "overlap_frames": 20},
            "handover_gate": [1, 2, 6],
            "frame_window_length": 160,
            "overlap_frames": 20,
            "source_plan": str(N72R1 / "protocol/n72r1_protocol.json"),
        },
        "actions": [
            "AUTHORITATIVE_CORRECT",
            "AUTHORITATIVE_REASSIGN",
            "ATOMIC_ID_SWAP",
            "ADD_NEW_IDENTITY",
            "RECOVER_IDENTITY",
            "AUTHORITATIVE_DELETE",
        ],
        "variants": [
            "M0_K1_ONLY_NO_MEMORY",
            "M1_HUMAN_EMA_PROTOTYPE",
            "M2_POSITIVE_HUMAN_ANCHORS",
            "M3_NEGATIVE_COMPETITOR_BANK",
            "M4_RELIABILITY_AGE_ADMISSION",
            "NO_WRITE_CONTROL",
        ],
        "future_windows": [20, 50, 100],
        "bootstrap": {"unit": "independent_sequence", "repetitions": 2000, "seed": 7202},
        "event_selection": {
            "selection_stage": "pre_treatment_current_frame_or_past_only",
            "allowed_runtime_inputs": ["Y_pre", "current/past state", "current frame GT after Y_pre freeze"],
            "forbidden_selection_inputs": ["future GT", "future metrics", "treatment outcomes", "IDSW after event"],
            "minimum_target_events": 40,
            "preferred_independent_sequences": 20,
            "use_all_eligible_if_below_preferred": True,
        },
        "resource_limits": {
            "max_gpus": 4,
            "one_sequence_or_frame_range_per_gpu": True,
            "independent_process_per_window": True,
            "oom_sharding": [160, 100, 50],
        },
        "non_goals": [
            "real_human_tape_claim",
            "checkpoint_change",
            "candidate_generation_change",
            "hungarian_solver_change",
            "threshold_or_metric_change",
            "calibration_selector_or_lora_before_gate",
        ],
        "protection_manifest": str(protection_path),
    }
    protocol_path = N72R2 / "protocol.json"
    atomic_json(protocol_path, protocol)
    atomic_json(
        N72R2 / "stage_00_status.json",
        {
            "schema_version": "N72R2_STAGE_STATUS_V1",
            "stage": "00_FREEZE_PROTOCOL_AND_INPUTS",
            "status": "PASS_BASELINE_FROZEN",
            "created_at_utc": now,
            "protocol": str(protocol_path),
            "protection_manifest": str(protection_path),
            "historical_inputs_read_only": True,
            "interaction_source": "simulated_from_gt",
            "real_human_tape_used": False,
            "next_stage": "01_PUBLIC_AUTHORITY_BRIDGE",
            "environment": {"python": platform.python_version(), "platform": platform.platform()},
        },
    )
    print(json.dumps({"status": "PASS_BASELINE_FROZEN", "protocol": str(protocol_path)}))


if __name__ == "__main__":
    main()
