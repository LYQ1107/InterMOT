#!/usr/bin/env python3
"""Audit whether the protected event transaction changed the sealed replay.

This is a CPU-only, post-run comparison.  It never opens GT and never edits
either Stage08 result.  The old N72R5R1 output is the immutable comparison
point; the protected-transaction run is selected with N72R5R1_RUN_ROOT.
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
import sys

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sam3_intermot.association.branch_public_replay import (  # noqa: E402
    BRANCHES,
    atomic_json,
    read_json,
    sha256_file,
)


OUT = Path(os.environ.get("N72R5R1_RUN_ROOT", str(ROOT / "outputs/N72R5R1")))
BASE = ROOT / "outputs/N72R5R1"
AUDIT_ROOT = OUT / "controller" / "round_02_protected_transaction_audit"


def _rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _semantic(row: dict[str, Any]) -> list[tuple[Any, ...]]:
    return [
        (item.get("candidate_uid"), item.get("public_id"), item.get("assignment_status"))
        for item in row.get("solver", {}).get("assignment_rows", [])
    ]


def _done(root: Path, event_id: str, branch: str) -> dict[str, Any]:
    return read_json(root / "public_assignment" / event_id / f"{branch}.done.json")


def main() -> int:
    current_manifest_path = OUT / "stage08_runtime_manifest.json"
    base_manifest_path = BASE / "stage08_runtime_manifest.json"
    current_manifest = read_json(current_manifest_path)
    base_manifest = read_json(base_manifest_path)
    events = [dict(item) for item in current_manifest.get("events", [])]
    if len(events) != 40 or current_manifest.get("failures"):
        raise RuntimeError("protected run is not a complete 40-event Stage08 input")

    counts = {
        "events_compared": 0,
        "branches_compared": 0,
        "event_assignment_changed": 0,
        "future_assignment_changed_at_event_plus_1": 0,
        "future_assignment_changed_anywhere": 0,
        "applied_branches_with_protected_locks": 0,
        "applied_branches_with_zero_event_change": 0,
        "action_status_changed": 0,
    }
    differences: list[dict[str, Any]] = []
    lock_counts: list[int] = []

    for event in events:
        event_id = str(event["event_id"])
        event_seen = False
        for branch in BRANCHES:
            current_done = _done(OUT, event_id, branch)
            base_done = _done(BASE, event_id, branch)
            current_rows = _rows(Path(current_done["output"]))
            base_rows = _rows(Path(base_done["output"]))
            if len(current_rows) != len(base_rows):
                differences.append({"event_id": event_id, "branch": branch, "kind": "frame_count", "current": len(current_rows), "base": len(base_rows)})
                continue
            event_seen = True
            counts["branches_compared"] += 1
            lock_count = int(current_done.get("protected_pre_treatment_lock_count", 0))
            if lock_count > 0:
                counts["applied_branches_with_protected_locks"] += 1
                lock_counts.append(lock_count)
            event_frame = int(current_done["event_frame"])
            current_by_frame = {int(row["frame"]): row for row in current_rows}
            base_by_frame = {int(row["frame"]): row for row in base_rows}
            if _semantic(current_by_frame[event_frame]) != _semantic(base_by_frame[event_frame]):
                counts["event_assignment_changed"] += 1
                differences.append({"event_id": event_id, "branch": branch, "kind": "event_assignment"})
            elif lock_count > 0:
                counts["applied_branches_with_zero_event_change"] += 1
            if _semantic(current_by_frame[event_frame + 1]) != _semantic(base_by_frame[event_frame + 1]):
                counts["future_assignment_changed_at_event_plus_1"] += 1
                differences.append({"event_id": event_id, "branch": branch, "kind": "event_plus_1_assignment"})
            changed_future = [
                frame
                for frame in sorted(current_by_frame)
                if frame > event_frame and _semantic(current_by_frame[frame]) != _semantic(base_by_frame[frame])
            ]
            if changed_future:
                counts["future_assignment_changed_anywhere"] += 1
                differences.append({"event_id": event_id, "branch": branch, "kind": "future_assignment", "frames": changed_future[:20]})
            if current_done.get("action_precondition_status") != base_done.get("action_precondition_status"):
                counts["action_status_changed"] += 1
                differences.append({"event_id": event_id, "branch": branch, "kind": "action_status", "current": current_done.get("action_precondition_status"), "base": base_done.get("action_precondition_status")})
        if event_seen:
            counts["events_compared"] += 1

    current_effect = OUT / "stage10_effect_scoring.json"
    base_effect = BASE / "stage10_effect_scoring.json"
    effect_comparison = {
        "base_sha256": sha256_file(base_effect),
        "protected_sha256": sha256_file(current_effect),
        "base_status": read_json(base_effect).get("status"),
        "protected_status": read_json(current_effect).get("status"),
        "primary_base": read_json(base_effect).get("summaries", {}).get("B4_MINUS_B0", {}).get("20", {}),
        "primary_protected": read_json(current_effect).get("summaries", {}).get("B4_MINUS_B0", {}).get("20", {}),
    }
    result = {
        "schema_version": "N72R5R1_ROUND02_PROTECTED_AUDIT_V1",
        "status": "PASS_PROTECTED_RUN_AUDITED_NO_BEHAVIOR_CHANGE" if not differences else "FAIL_PROTECTED_RUN_DIFFERS_FROM_BASELINE",
        "round": "round_02_protected_transaction",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "base_stage08_manifest_sha256": sha256_file(base_manifest_path),
        "protected_stage08_manifest_sha256": sha256_file(current_manifest_path),
        "counts": counts,
        "protected_lock_count": {
            "finite_count": len(lock_counts),
            "median": None if not lock_counts else sum(lock_counts) / len(lock_counts),
            "min": None if not lock_counts else min(lock_counts),
            "max": None if not lock_counts else max(lock_counts),
        },
        "effect_comparison": effect_comparison,
        "differences": differences,
        "interpretation": {
            "mechanism_effective": bool(differences),
            "reason_if_no_effect": "All protected locks matched the pre-treatment exact assignment and no event/future semantic row changed." if not differences else None,
            "runtime_future_gt_used": False,
            "posthoc_gt_used": False,
            "next_routing": "FEATURE_SEPARABILITY_AUDIT_ON_CANDIDATE_PRESENT_ERRORS" if not differences else "REPAIR_PROTECTED_TRANSACTION_DIFF",
        },
    }
    atomic_json(AUDIT_ROOT / "round_02_mechanism_audit.json", result)
    print(json.dumps({"status": result["status"], "counts": counts, "output": str(AUDIT_ROOT / "round_02_mechanism_audit.json")}, ensure_ascii=False))
    return 0 if not differences else 1


if __name__ == "__main__":
    raise SystemExit(main())
