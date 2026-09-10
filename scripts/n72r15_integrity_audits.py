#!/usr/bin/env python3
"""CPU-only integrity audits for N72R15 state propagation and assignment cardinality."""

from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import tempfile
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[1]
VARIANTS = ("E0_BASELINE_B0", "E1I_HUMAN_RELATIVE_STATE", "E1J_TRUSTED_GLOBAL_RELATIVE_STATE")
HORIZON = 100


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    try:
        with __import__("os").fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            __import__("os").fsync(handle.fileno())
        __import__("os").replace(temporary, path)
    finally:
        if __import__("os").path.exists(temporary):
            __import__("os").unlink(temporary)


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object: {path}")
    return value


def read_rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _paths(manifest: Mapping[str, Any]) -> dict[str, dict[str, list[dict[str, Any]]]]:
    result: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for event in manifest.get("events", []):
        event_id = str(event["event_id"])
        if event_id in result:
            raise RuntimeError(f"duplicate event {event_id}")
        result[event_id] = {}
        for variant in event.get("variants", []):
            name = str(variant["variant"])
            if name in result[event_id] or name not in VARIANTS:
                raise RuntimeError(f"duplicate/unknown variant {event_id}/{name}")
            rows = read_rows(Path(str(variant["frames"])))
            if len(rows) != HORIZON + 1:
                raise RuntimeError(f"{event_id}/{name}: expected 101 rows")
            result[event_id][name] = rows
        if set(result[event_id]) != set(VARIANTS):
            raise RuntimeError(f"{event_id}: variant set incomplete")
    if len(result) != 32:
        raise RuntimeError(f"expected 32 events, got {len(result)}")
    return result


def _set_of_ints(value: Any, label: str) -> set[int]:
    if value is None:
        return set()
    if not isinstance(value, list):
        raise RuntimeError(f"{label} must be a list")
    return {int(item) for item in value}


def _assignment_count(row: Mapping[str, Any]) -> int:
    assignment = row.get("assignment")
    if not isinstance(assignment, Mapping):
        raise RuntimeError(f"{row.get('event_id')}:{row.get('frame')}: missing assignment")
    rows = assignment.get("assignment_rows", [])
    return sum(item.get("public_id") is not None for item in rows)


def self_reinforcement(manifest_path: Path, output_path: Path) -> dict[str, Any]:
    manifest = read_json(manifest_path)
    if manifest.get("status") != "PASS_N72R15_FORMAL_REPLAY":
        raise RuntimeError("formal replay is not PASS")
    data = _paths(manifest)
    violations: list[dict[str, Any]] = []
    counts: dict[str, dict[str, int]] = {
        variant: {
            "future_rows": 0,
            "state_changed_rows": 0,
            "machine_write_rows": 0,
            "motion_write_rows": 0,
            "disagreement_rows": 0,
            "disagreement_trusted_write_rows": 0,
            "changed_and_machine_write_rows": 0,
        }
        for variant in VARIANTS
    }
    by_action: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for event in manifest["events"]:
        event_id = str(event["event_id"])
        action = str(event["action_type"])
        for variant in VARIANTS:
            rows = data[event_id][variant]
            if rows[0].get("memory_read") is not False or rows[0].get("event_frame_memory_read") is not False:
                violations.append({"event_id": event_id, "variant": variant, "frame": int(rows[0]["frame"]), "reason": "event_frame_memory_read"})
            for row in rows[1:]:
                frame = int(row["frame"])
                info = row.get("persistent_state_association", {})
                update = row.get("state_update") or {}
                counts[variant]["future_rows"] += 1
                counts[variant]["state_changed_rows"] += int(info.get("state_changed") is True)
                machine = _set_of_ints(update.get("machine_memory_write_public_ids", []), "machine writes")
                motion = _set_of_ints(update.get("motion_state_write_public_ids", []), "motion writes")
                consensus = _set_of_ints(update.get("consensus_public_ids", []), "consensus ids")
                disagreement = _set_of_ints(update.get("disagreement_public_ids", []), "disagreement ids")
                changed = _set_of_ints(info.get("treatment_changed_public_ids", []), "changed ids")
                counts[variant]["machine_write_rows"] += int(bool(machine))
                counts[variant]["motion_write_rows"] += int(bool(motion))
                counts[variant]["disagreement_rows"] += int(bool(disagreement))
                counts[variant]["disagreement_trusted_write_rows"] += int(bool(disagreement & machine))
                counts[variant]["changed_and_machine_write_rows"] += int(bool(changed & machine))
                by_action[action][f"{variant}_disagreement_trusted_write_rows"] += int(bool(disagreement & machine))
                by_action[action][f"{variant}_changed_and_machine_write_rows"] += int(bool(changed & machine))
                if row.get("runtime_future_gt_used") is not False or info.get("runtime_future_gt_used") is not False or update.get("runtime_future_gt_used") is not False:
                    violations.append({"event_id": event_id, "variant": variant, "frame": frame, "reason": "runtime_future_gt_used"})
                if variant == "E1I_HUMAN_RELATIVE_STATE" and (machine or motion):
                    violations.append({"event_id": event_id, "variant": variant, "frame": frame, "reason": "H1_future_machine_or_motion_write", "machine": sorted(machine), "motion": sorted(motion)})
                if variant == "E1J_TRUSTED_GLOBAL_RELATIVE_STATE":
                    if disagreement & machine:
                        violations.append({"event_id": event_id, "variant": variant, "frame": frame, "reason": "disagreement_trusted_write", "ids": sorted(disagreement & machine)})
                    if not machine.issubset(consensus):
                        violations.append({"event_id": event_id, "variant": variant, "frame": frame, "reason": "machine_write_without_consensus", "machine": sorted(machine), "consensus": sorted(consensus)})
                    if disagreement and machine:
                        violations.append({"event_id": event_id, "variant": variant, "frame": frame, "reason": "G2_disagreement_write_nonempty"})
    output = {
        "schema_version": "N72R15_SELF_REINFORCEMENT_AUDIT_V1",
        "status": "PASS_N72R15_SELF_REINFORCEMENT_AUDIT" if not violations else "FAIL_N72R15_SELF_REINFORCEMENT_AUDIT",
        "created_at_utc": now_utc(),
        "source_formal_manifest": str(manifest_path),
        "source_formal_manifest_sha256": sha256_file(manifest_path),
        "events": 32,
        "variants": list(VARIANTS),
        "counts": counts,
        "by_action": {action: dict(values) for action, values in sorted(by_action.items())},
        "violation_count": len(violations),
        "violations": violations,
        "runtime_future_gt_used": False,
        "historical_outputs_modified": False,
    }
    atomic_json(output_path, output)
    return output


def assignment_cardinality(manifest_path: Path, output_path: Path) -> dict[str, Any]:
    manifest = read_json(manifest_path)
    if manifest.get("status") != "PASS_N72R15_FORMAL_REPLAY":
        raise RuntimeError("formal replay is not PASS")
    data = _paths(manifest)
    violations: list[dict[str, Any]] = []
    counts: dict[str, int] = defaultdict(int)
    action_counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for event in manifest["events"]:
        event_id = str(event["event_id"])
        action = str(event["action_type"])
        rows_by_variant = data[event_id]
        for offset in range(HORIZON + 1):
            frame_axes = {}
            candidate_axes = {}
            public_axes = {}
            assigned = {}
            for variant in VARIANTS:
                row = rows_by_variant[variant][offset]
                frame_axes[variant] = int(row.get("frame", -1))
                candidates = row.get("candidate_rows", [])
                uids = [str(item.get("candidate_uid")) for item in candidates]
                candidate_axes[variant] = uids
                info = row.get("persistent_state_association", {})
                score_audit = row.get("score_audit") or {}
                public_axes[variant] = [int(value) for value in info.get("public_id_axis", score_audit.get("public_id_axis", []))]
                if len(uids) != len(set(uids)) or any(value in {"None", ""} for value in uids):
                    violations.append({"event_id": event_id, "frame": frame_axes[variant], "variant": variant, "reason": "candidate_uid_axis_invalid"})
                if len(public_axes[variant]) != len(set(public_axes[variant])):
                    violations.append({"event_id": event_id, "frame": frame_axes[variant], "variant": variant, "reason": "public_id_axis_duplicate"})
                pool_audit = row.get("candidate_pool") or {}
                if offset > 0 and (pool_audit.get("target_session_candidate_in_solver") is not False or info.get("target_session_candidate_in_solver") is not False):
                    violations.append({"event_id": event_id, "frame": frame_axes[variant], "variant": variant, "reason": "target_session_candidate_in_solver"})
                if any(row.get(key) is not False for key in ("runtime_future_gt_used", "runtime_gt_read", "posthoc_gt_used")):
                    violations.append({"event_id": event_id, "frame": frame_axes[variant], "variant": variant, "reason": "runtime_flag"})
                count = _assignment_count(row) if offset > 0 else None
                assigned[variant] = count
                if offset > 0:
                    audit = row.get("score_audit", {})
                    solver_count = int(row.get("assignment", {}).get("assigned_public_count", -1))
                    audit_count = int(audit.get("treatment_assigned_public_count", -2))
                    if count != solver_count or count != audit_count:
                        violations.append({"event_id": event_id, "frame": frame_axes[variant], "variant": variant, "reason": "assigned_count_inconsistent", "rows": count, "solver": solver_count, "audit": audit_count})
                    base_count = int(audit.get("base_assigned_public_count", -1))
                    base_assignment_count = sum(item.get("public_id") is not None for item in row.get("base_assignment", {}).get("assignment_rows", []))
                    if base_count != base_assignment_count:
                        violations.append({"event_id": event_id, "frame": frame_axes[variant], "variant": variant, "reason": "base_assigned_count_inconsistent", "rows": base_assignment_count, "audit": base_count})
                    expected_changed = base_count != audit_count
                    if bool(audit.get("assignment_cardinality_changed")) != expected_changed:
                        violations.append({"event_id": event_id, "frame": frame_axes[variant], "variant": variant, "reason": "cardinality_flag_inconsistent"})
                    counts[f"{variant}_future_rows"] += 1
                    counts[f"{variant}_cardinality_changed_rows"] += int(expected_changed)
                    action_counts[action][f"{variant}_cardinality_changed_rows"] += int(expected_changed)
            frame = frame_axes[VARIANTS[0]]
            if len(set(frame_axes.values())) != 1:
                violations.append({"event_id": event_id, "frame": frame, "reason": "cross_variant_frame_axis_mismatch", "axes": frame_axes})
            if offset > 0:
                if len({tuple(candidate_axes[v]) for v in VARIANTS}) != 1:
                    violations.append({"event_id": event_id, "frame": frame, "reason": "cross_variant_candidate_axis_mismatch"})
                if len({tuple(public_axes[v]) for v in VARIANTS}) != 1:
                    violations.append({"event_id": event_id, "frame": frame, "reason": "cross_variant_public_axis_mismatch"})
                base_count = assigned[VARIANTS[0]]
                for variant in VARIANTS[1:]:
                    counts[f"{variant}_vs_b0_cardinality_delta_rows"] += int(assigned[variant] != base_count)
                    action_counts[action][f"{variant}_vs_b0_cardinality_delta_rows"] += int(assigned[variant] != base_count)
    output = {
        "schema_version": "N72R15_ASSIGNMENT_CARDINALITY_AUDIT_V1",
        "status": "PASS_N72R15_ASSIGNMENT_CARDINALITY_AUDIT" if not violations else "FAIL_N72R15_ASSIGNMENT_CARDINALITY_AUDIT",
        "created_at_utc": now_utc(),
        "source_formal_manifest": str(manifest_path),
        "source_formal_manifest_sha256": sha256_file(manifest_path),
        "events": 32,
        "variants": list(VARIANTS),
        "counts": dict(counts),
        "by_action": {action: dict(values) for action, values in sorted(action_counts.items())},
        "violation_count": len(violations),
        "violations": violations,
        "runtime_future_gt_used": False,
        "historical_outputs_modified": False,
    }
    atomic_json(output_path, output)
    return output


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    try:
        args.output_root.mkdir(parents=True, exist_ok=True)
        self_result = self_reinforcement(args.manifest.resolve(), (args.output_root / "self_reinforcement_audit.json").resolve())
        cardinality_result = assignment_cardinality(args.manifest.resolve(), (args.output_root / "assignment_cardinality_audit.json").resolve())
        print(json.dumps({"self_reinforcement": self_result["status"], "assignment_cardinality": cardinality_result["status"]}, sort_keys=True))
        return 0 if self_result["status"].startswith("PASS") and cardinality_result["status"].startswith("PASS") else 2
    except Exception as exc:
        failure = {
            "schema_version": "N72R15_FAILURE_V1",
            "status": "FAIL_N72R15_INTEGRITY_AUDITS",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": __import__("traceback").format_exc(),
            "created_at_utc": now_utc(),
            "runtime_future_gt_used": False,
        }
        atomic_json(args.output_root.resolve() / "integrity_audits_failure.json", failure)
        print(json.dumps(failure, sort_keys=True))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
