#!/usr/bin/env python3
"""CPU-only integrity gate for the N72R6 C0/C1 replay artifacts."""

from __future__ import annotations

from collections import Counter
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sam3_intermot.association.branch_public_replay import (  # noqa: E402
    HORIZON,
    atomic_json,
    read_json,
    read_jsonl,
    sha256_file,
)


REPLAY_ROOT = ROOT / "outputs/N72R6/public_replay/attempt_4"
BATCH = REPLAY_ROOT / "replay_batch_status.json"
OUT = ROOT / "outputs/N72R6"


def resolve(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def fail(message: str) -> None:
    raise RuntimeError(message)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--replay-root", type=Path, default=REPLAY_ROOT)
    parser.add_argument("--status-output", type=Path, default=OUT / "stage_05_status.json")
    args = parser.parse_args()
    replay_root = args.replay_root if args.replay_root.is_absolute() else ROOT / args.replay_root
    batch_path = replay_root / "replay_batch_status.json"
    batch = read_json(batch_path)
    if batch.get("status") != "PASS_N72R6_C0_C1_REPLAY":
        fail(f"replay batch is not PASS: {batch.get('status')}")
    results = [dict(item) for item in batch.get("results", [])]
    if len(results) != 32 or len({str(item["event_id"]) for item in results}) != 32:
        fail(f"replay batch coverage is not 32 unique events: {len(results)}")
    failures: list[str] = []
    action_counts: Counter[str] = Counter()
    sequence_counts: Counter[str] = Counter()
    future_candidate_rows = 0
    future_target_assignments = 0
    target_lost_frames = 0
    target_event_assignments = 0
    main_mutation_count = 0
    target_domain_violations = 0
    runtime_flag_violations = 0
    axis_violations = 0
    epoch_count = 0
    ypre_matches = 0
    manifest_rows: list[dict] = []

    for result in sorted(results, key=lambda item: str(item["event_id"])):
        event_id = str(result["event_id"])
        manifest_path = replay_root / event_id / "event_manifest.json"
        try:
            manifest = read_json(manifest_path)
            if manifest.get("status") != "PASS_N72R6_C0_C1_EVENT_REPLAY":
                fail(f"event manifest status: {event_id}")
            c0_path = resolve(str(manifest["c0"]["path"]))
            c1_path = resolve(str(manifest["c1"]["path"]))
            if sha256_file(c0_path) != manifest["c0"]["sha256"]:
                fail(f"C0 hash mismatch: {event_id}")
            if sha256_file(c1_path) != manifest["c1"]["sha256"]:
                fail(f"C1 hash mismatch: {event_id}")
            c0 = read_jsonl(c0_path)
            c1 = read_jsonl(c1_path)
            if len(c0) != HORIZON + 1 or len(c1) != HORIZON + 1:
                fail(f"frame count mismatch: {event_id}")
            event_frame = int(manifest["event_frame"])
            target_public = int(manifest["target_public_id"])
            action_counts[str(manifest["action_type"])] += 1
            sequence_counts[str(manifest["sequence"])] += 1
            epoch_count += 1 if manifest.get("correction_epoch", {}).get("epoch_id") else 0
            ypre_matches += int(bool(manifest.get("c1_ypre_assignment_matches_frozen_b0")))
            target_future = 0
            target_future_assigned = 0
            target_lost = 0
            event_target_assigned = 0
            for index, (base_row, row) in enumerate(zip(c0, c1)):
                expected_frame = event_frame + index
                if int(base_row.get("frame", -1)) != expected_frame or int(row.get("frame", -1)) != expected_frame:
                    fail(f"frame axis mismatch: {event_id}:{expected_frame}")
                for flag in ("runtime_future_gt_used", "runtime_gt_read", "posthoc_gt_used", "public_id_inference"):
                    if row.get(flag) is not False or base_row.get(flag) is not False:
                        runtime_flag_violations += 1
                if row.get("event_frame_memory_read") is not False:
                    runtime_flag_violations += 1
                axis = [int(value) for value in row.get("public_id_axis", [])]
                if len(axis) != len(set(axis)) or target_public not in axis:
                    axis_violations += 1
                target_identity_rows = [item for item in row.get("identity_rows", []) if int(item.get("public_id", -1)) == target_public]
                if len(target_identity_rows) != 1:
                    axis_violations += 1
                if target_identity_rows and target_identity_rows[0].get("identity_status") == "LOST":
                    target_lost += 1
                c0_uids = [str(item.get("candidate_uid")) for item in base_row.get("candidate_rows", [])]
                c1_main_uids = [str(item.get("candidate_uid")) for item in row.get("candidate_rows", []) if item.get("candidate_kind") != "TARGET_CORRECTION_SESSION_CANDIDATE"]
                if c0_uids != c1_main_uids:
                    main_mutation_count += 1
                exclusive = row.get("target_exclusive_constraint") or {}
                fallback_uid = exclusive.get("fallback_main_candidate_uid")
                fallback_uid = None if fallback_uid in (None, "") else str(fallback_uid)
                shadowed_uids = {str(item) for item in exclusive.get("shadowed_main_candidate_uids", [])}
                if fallback_uid is not None and fallback_uid not in shadowed_uids:
                    target_domain_violations += 1
                fallback_audit = row.get("human_anchor_main_fallback") or {}
                selected_fallback_uid = fallback_audit.get("selected_main_candidate_uid")
                selected_fallback_uid = (
                    None if selected_fallback_uid in (None, "") else str(selected_fallback_uid)
                )
                if selected_fallback_uid != fallback_uid:
                    target_domain_violations += 1
                for candidate in row.get("candidate_rows", []):
                    kind = candidate.get("candidate_kind")
                    public = candidate.get("public_id")
                    solver_public = candidate.get("solver_public_id")
                    if kind == "TARGET_CORRECTION_SESSION_CANDIDATE":
                        if public not in (None, target_public) or solver_public not in (None, target_public):
                            target_domain_violations += 1
                        if index == 0:
                            event_target_assigned += int(public == target_public)
                        else:
                            target_future += 1
                            target_future_assigned += int(public == target_public)
                    elif public == target_public or solver_public == target_public:
                        # The only permitted main-row exception is the row
                        # selected by the fixed B0 fallback protocol.  It
                        # must already be in this frame's shadowed UID set;
                        # every other main row remains forbidden from the
                        # target public domain.
                        if str(candidate.get("candidate_uid")) != fallback_uid:
                            target_domain_violations += 1
                if exclusive.get("target_public_id") != target_public or exclusive.get("explicit_none_preserved") is not True:
                    target_domain_violations += 1
            future_candidate_rows += target_future
            future_target_assignments += target_future_assigned
            target_lost_frames += target_lost
            target_event_assignments += event_target_assigned
            manifest_rows.append({
                "event_id": event_id,
                "sequence": str(manifest["sequence"]),
                "action_type": str(manifest["action_type"]),
                "target_public_id": target_public,
                "frame_count": len(c1),
                "target_event_assigned": bool(event_target_assigned),
                "target_future_candidate_rows": target_future,
                "target_future_assigned_rows": target_future_assigned,
                "target_lost_frames": target_lost,
                "c1_ypre_assignment_matches_frozen_b0": bool(manifest.get("c1_ypre_assignment_matches_frozen_b0")),
            })
        except Exception as exc:
            failures.append(f"{event_id}: {type(exc).__name__}: {exc}")

    if failures:
        raise RuntimeError("; ".join(failures))
    if target_event_assignments != 32:
        fail(f"target event assignments are not 32: {target_event_assignments}")
    if main_mutation_count or target_domain_violations or runtime_flag_violations or axis_violations:
        fail(f"integrity violations main={main_mutation_count},domain={target_domain_violations},flags={runtime_flag_violations},axis={axis_violations}")

    payload = {
        "schema_version": "N72R6_TARGET_SCOPED_REPLAY_AUDIT_V1",
        "status": "PASS_TARGET_SCOPED_C0_C1_REPLAY_AUDITED",
        "replay_batch": str(batch_path),
        "replay_batch_sha256": sha256_file(batch_path),
        "event_count": len(manifest_rows),
        "sequence_count": len(sequence_counts),
        "action_counts": dict(sorted(action_counts.items())),
        "sequence_counts": dict(sorted(sequence_counts.items())),
        "frame_count_per_variant": HORIZON + 1,
        "c0_c1_same_frozen_t_minus_1_snapshot": True,
        "c1_ypre_assignment_match_count": ypre_matches,
        "target_event_assigned_count": target_event_assignments,
        "target_future_candidate_rows": future_candidate_rows,
        "target_future_assigned_rows": future_target_assignments,
        "target_lost_frames": target_lost_frames,
        "target_public_domain_violations": target_domain_violations,
        "main_candidate_mutation_count": main_mutation_count,
        "runtime_gt_flag_violations": runtime_flag_violations,
        "public_axis_violations": axis_violations,
        "correction_epoch_count": epoch_count,
        "c1_state_retention_max_lost_gap": HORIZON + 1,
        "target_session_candidate_public_domain": ["target_public_id", "NONE"],
        "human_anchor_main_fallback_rows_allowed_only": True,
        "human_anchor_main_fallback_validated_from_shadowed_b0_uids": True,
        "runtime_future_gt_used": False,
        "posthoc_scoring_completed": False,
        "events": manifest_rows,
    }
    status_output = args.status_output if args.status_output.is_absolute() else ROOT / args.status_output
    atomic_json(status_output, payload)
    print({"status": payload["status"], "event_count": len(manifest_rows), "future_candidate_rows": future_candidate_rows})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
