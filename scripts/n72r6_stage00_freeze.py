#!/usr/bin/env python3
"""Freeze the N72R6 input contract without executing a model."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/N72R6"
EVENT_MANIFEST = ROOT / "outputs/N72R5/mechanism_rounds/round_06_event_policy/real_event_manifest.json"
STAGE08 = ROOT / "outputs/N72R5R1/controller/round_05_branch_isolation_v0/full/stage08_runtime_manifest.json"
STAGE07 = ROOT / "outputs/N72R5/mechanism_rounds/round_07_official_full_loop_attempt5/official_full_loop_manifest.json"
CHECKPOINT = ROOT / "checkpoints/sam3.1_mirror/sam3.1_multiplex.pt"


def sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: object) -> None:
    import os
    import tempfile

    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp, path)
    finally:
        if os.path.exists(temp):
            os.unlink(temp)


def git(args: list[str]) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def main() -> int:
    # The execution host carries a legacy Git that lacks branch
    # --show-current; rev-parse is equivalent and available there.
    branch = git(["rev-parse", "--abbrev-ref", "HEAD"])
    commit = git(["rev-parse", "HEAD"])
    status = git(["status", "--porcelain"])
    event_payload = json.loads(EVENT_MANIFEST.read_text(encoding="utf-8"))
    stage08_payload = json.loads(STAGE08.read_text(encoding="utf-8"))
    events = list(event_payload.get("events", []))
    applied = []
    for item in stage08_payload.get("events", []):
        branches = {str(row.get("branch")): row for row in item.get("branches", [])}
        row = branches.get("B1_SPATIAL_CORRECTION_ONLY")
        if row and str(row.get("action_precondition_status")) == "APPLIED":
            applied.append(str(item.get("event_id")))
    protocol = {
        "schema_version": "N72R6_FROZEN_PROTOCOL_V1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "research_question": "target_scoped_correction_preserves_public_identity_and_protects_untouched_stream",
        "baseline": {
            "branch": branch,
            "commit": commit,
            "expected_parent": "164da11d63b8578f4caf208befb70a5f783b9482",
            "worktree_status_at_freeze": status,
        },
        "inputs": {
            "event_manifest": str(EVENT_MANIFEST),
            "event_manifest_sha256": sha256(EVENT_MANIFEST),
            "stage08_runtime_manifest": str(STAGE08),
            "stage08_runtime_manifest_sha256": sha256(STAGE08),
            "stage07_manifest": str(STAGE07),
            "stage07_manifest_sha256": sha256(STAGE07),
            "checkpoint": str(CHECKPOINT),
            "checkpoint_sha256": sha256(CHECKPOINT),
        },
        "frozen_definitions": {
            "horizon": 100,
            "event_frame_correction_is_not_future": True,
            "first_memory_visible_frame": "event_frame+1",
            "runtime_future_gt_used": False,
            "main_stream": "N72R5R1_B0_NO_INTERVENTION_READ_ONLY",
            "target_session": "independent_sam3_backend_one_object",
            "target_candidate_compatibility": ["target_public_id", "NONE"],
            "non_target_candidate_compatibility": "target_candidate_public_id_forbidden_when_target_candidate_present",
            "weights_unchanged": True,
            "solver": "solve_effect_assignment_with_explicit_none",
            "branches": [
                "C0_MAIN_BASELINE",
                "C1_TARGET_SCOPED_CORRECTION",
                "C2_TARGET_SCOPED_CORRECTION_PLUS_TVC_V1",
            ],
        },
        "selection": {
            "source": "N72R5R1_corrected_V0_stage08",
            "eligibility": "B1_SPATIAL_CORRECTION_ONLY action_precondition_status == APPLIED",
            "eligible_event_count": len(sorted(set(applied))),
            "eligible_event_ids": sorted(set(applied)),
            "ineligible_events_preserved": len(events) - len(set(applied)),
        },
        "prohibited": [
            "future_gt_runtime_read",
            "whole_active_set_reprompt_for_target",
            "target_candidate_to_non_target_public_id",
            "main_candidate_reclaim_target_when_target_candidate_present",
            "rerun_N72R5_stage07_workers",
            "change_checkpoint_candidate_definition_solver_or_metrics",
        ],
    }
    protocol_hash = hashlib.sha256(
        json.dumps(protocol, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    ).hexdigest()
    protocol["protocol_sha256"] = protocol_hash
    atomic_json(OUT / "protocol.json", protocol)
    atomic_json(
        OUT / "stage_00_status.json",
        {
            "schema_version": "N72R6_STAGE_STATUS_V1",
            "stage": "Stage00_FREEZE",
            "status": "PASS_FROZEN_INPUTS" if not status else "PASS_FROZEN_WITH_WORKTREE_CHANGES",
            "protocol_sha256": protocol_hash,
            "branch": branch,
            "commit": commit,
            "worktree_status": status,
            "event_count": len(events),
            "eligible_event_count": len(sorted(set(applied))),
            "runtime_future_gt_used": False,
            "historical_outputs_read_only": True,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
        },
    )
    print(json.dumps({"status": "PASS", "eligible_event_count": len(set(applied)), "protocol_sha256": protocol_hash}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
