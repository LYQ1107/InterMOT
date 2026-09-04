#!/usr/bin/env python3
"""N72R5R1 Stage 09: validate only the new public-runtime axis.

Stage07's candidate/frame audit is a frozen input.  This validator checks the
Stage08 sidecar's public/state authority, explicit assignment semantics,
causal flags, and one-to-one constraints.  It never opens GT and never turns
an incomplete Stage08 set into a pass.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sam3_intermot.association.branch_public_replay import (  # noqa: E402
    BRANCHES,
    HORIZON,
    atomic_json,
    json_hash,
    now_utc,
    read_json,
    read_jsonl,
    sha256_file,
)

OUT = Path(os.environ.get("N72R5R1_RUN_ROOT", str(ROOT / "outputs/N72R5R1")))
PUBLIC_ROOT = OUT / "public_assignment"
RUNTIME_MANIFEST = OUT / "stage08_runtime_manifest.json"
EVENT_MANIFEST = ROOT / "outputs/N72R5/mechanism_rounds/round_06_event_policy/real_event_manifest.json"
VALIDATION = OUT / "stage09_validation.json"
STATUS = OUT / "stage_09_status.json"

FORBIDDEN_RUNTIME_KEYS = {
    "dataset_gt_id",
    "other_dataset_gt_id",
    "gt_box",
    "future_gt",
    "future_identity_error",
    "reward",
    "target_gt_id",
}
FORMAL_CANDIDATE_STATUSES = {"ASSIGNED_TO_PUBLIC_ID", "EXPLICIT_NONE"}
LEGACY_CANDIDATE_STATUSES = {"OUTER_BIRTH_ASSIGNED"}
IDENTITY_STATUSES = {"ASSIGNED", "NO_CANDIDATE_ASSIGNED"}


def _forbidden_paths(value: Any, path: str = "") -> list[str]:
    found: list[str] = []
    if isinstance(value, Mapping):
        for key, child in value.items():
            child_path = f"{path}.{key}" if path else str(key)
            if str(key) in FORBIDDEN_RUNTIME_KEYS:
                found.append(child_path)
            found.extend(_forbidden_paths(child, child_path))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            found.extend(_forbidden_paths(child, f"{path}[{index}]"))
    return found


def _finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def _semantic_candidate(row: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        int(row.get("candidate_index", -1)),
        int(row.get("official_raw_sam_id", -1)),
        int(row.get("adapter_external_id", -1)),
        tuple(round(float(value), 7) for value in row.get("box_xyxy", [])),
        str(row.get("feature_sha256")),
        None if row.get("public_id") is None else int(row["public_id"]),
    )


def validate_frame(
    row: Mapping[str, Any],
    *,
    event_id: str,
    branch: str,
    event_frame: int,
    expected_y_pre_hash: str | None,
) -> tuple[list[str], Counter[str]]:
    errors: list[str] = []
    counts: Counter[str] = Counter()
    frame = row.get("frame")
    prefix = f"{event_id}/{branch}/{frame}"
    if row.get("event_id") != event_id or row.get("branch") != branch:
        errors.append(f"{prefix}:event_branch_mismatch")
    if not isinstance(frame, int):
        errors.append(f"{prefix}:frame_not_int")
    for key in ("runtime_future_gt_used", "runtime_gt_read", "posthoc_gt_used", "public_id_inference"):
        if row.get(key) is not False:
            errors.append(f"{prefix}:{key}_not_false")
    leaked = _forbidden_paths(row)
    if leaked:
        errors.append(f"{prefix}:runtime_gt_field:{leaked[:3]}")
    candidates = row.get("candidate_rows")
    identities = row.get("identity_rows")
    if not isinstance(candidates, list):
        errors.append(f"{prefix}:candidate_rows_not_list")
        candidates = []
    if not isinstance(identities, list):
        errors.append(f"{prefix}:identity_rows_not_list")
        identities = []
    if row.get("candidate_count") != len(candidates):
        errors.append(f"{prefix}:candidate_count_mismatch")
    if row.get("identity_count") != len(identities):
        errors.append(f"{prefix}:identity_count_mismatch")
    if row.get("assignment_decision_coverage") != 1.0:
        errors.append(f"{prefix}:assignment_decision_coverage_not_one")
    candidate_uids = [str(item.get("candidate_uid")) for item in candidates if isinstance(item, Mapping)]
    if len(candidate_uids) != len(set(candidate_uids)):
        errors.append(f"{prefix}:duplicate_candidate_uid")
    public_values: list[int] = []
    formal_statuses: set[str] = set()
    legacy_birth_count = 0
    for candidate in candidates:
        if not isinstance(candidate, Mapping):
            errors.append(f"{prefix}:candidate_not_object")
            continue
        status = str(candidate.get("assignment_status"))
        formal_statuses.add(status)
        if status not in FORMAL_CANDIDATE_STATUSES | LEGACY_CANDIDATE_STATUSES:
            errors.append(f"{prefix}:invalid_candidate_assignment_status:{status}")
        if status == "OUTER_BIRTH_ASSIGNED":
            legacy_birth_count += 1
        public = candidate.get("public_id")
        if public is None:
            if status not in {"EXPLICIT_NONE"}:
                errors.append(f"{prefix}:null_public_without_explicit_none")
            counts["explicit_none_candidate_rows"] += 1
        else:
            try:
                public_values.append(int(public))
            except (TypeError, ValueError):
                errors.append(f"{prefix}:public_id_not_int")
            if status not in {"ASSIGNED_TO_PUBLIC_ID", "OUTER_BIRTH_ASSIGNED"}:
                errors.append(f"{prefix}:non_null_public_without_assignment_status")
            counts["assigned_candidate_rows"] += 1
    if len(public_values) != len(set(public_values)):
        errors.append(f"{prefix}:duplicate_public_id_for_candidates")
    counts["legacy_outer_birth_status_rows"] += legacy_birth_count
    assigned_by_public = {int(candidate["public_id"]): str(candidate["candidate_uid"]) for candidate in candidates if isinstance(candidate, Mapping) and candidate.get("public_id") is not None}
    identity_public_values: list[int] = []
    identity_uids: list[str] = []
    for identity in identities:
        if not isinstance(identity, Mapping):
            errors.append(f"{prefix}:identity_not_object")
            continue
        status = str(identity.get("status"))
        if status not in IDENTITY_STATUSES:
            errors.append(f"{prefix}:invalid_identity_status:{status}")
        public = identity.get("public_id")
        try:
            identity_public_values.append(int(public))
        except (TypeError, ValueError):
            errors.append(f"{prefix}:identity_public_id_invalid")
            continue
        if int(public) in identity_public_values[:-1]:
            errors.append(f"{prefix}:duplicate_identity_public_id")
        uid = identity.get("candidate_uid")
        if status == "ASSIGNED":
            if uid is None:
                errors.append(f"{prefix}:assigned_identity_without_candidate")
            else:
                uid = str(uid)
                identity_uids.append(uid)
                if uid not in set(candidate_uids):
                    errors.append(f"{prefix}:identity_candidate_not_in_candidate_axis")
                if assigned_by_public.get(int(public)) != uid:
                    errors.append(f"{prefix}:identity_candidate_public_mismatch")
        elif uid is not None:
            errors.append(f"{prefix}:none_identity_has_candidate")
    if len(identity_public_values) != len(set(identity_public_values)):
        errors.append(f"{prefix}:identity_public_axis_collision")
    if len(identity_uids) != len(set(identity_uids)):
        errors.append(f"{prefix}:candidate_owned_by_two_identities")
    if row.get("public_id_immutability") is not True:
        errors.append(f"{prefix}:public_id_immutability_not_true")
    if row.get("candidate_index_to_public_id") is not False or row.get("raw_sam_id_to_public_id") is not False:
        errors.append(f"{prefix}:forbidden_numeric_public_mapping_flag")
    public_axis = row.get("public_id_axis", [])
    state_axis = row.get("association_state_axis", [])
    if len(public_axis) != len(set(public_axis)):
        errors.append(f"{prefix}:public_axis_collision")
    if len(state_axis) != len(set(state_axis)):
        errors.append(f"{prefix}:state_axis_collision")
    if set(public_values) - set(int(value) for value in public_axis):
        errors.append(f"{prefix}:candidate_public_missing_from_public_axis")
    if frame == event_frame:
        expected_role = "PRE_INTERVENTION_Y_PRE" if branch == "B0_NO_INTERVENTION" else "POST_INTERVENTION_Y_POST"
        if row.get("candidate_role") != expected_role:
            errors.append(f"{prefix}:event_candidate_role_mismatch")
        if row.get("event_frame_memory_read") is not False or row.get("memory_read") is not False:
            errors.append(f"{prefix}:event_memory_read_not_hidden")
        if expected_y_pre_hash is not None and str(row.get("shared_y_pre_semantic_hash")) != str(expected_y_pre_hash):
            errors.append(f"{prefix}:shared_y_pre_hash_mismatch")
    elif frame == event_frame + 1 and row.get("first_memory_visible_frame") != event_frame + 1:
        errors.append(f"{prefix}:first_memory_visible_frame_mismatch")
    if row.get("frame_horizon") != int(frame - event_frame):
        errors.append(f"{prefix}:frame_horizon_mismatch")
    if not _finite(row.get("assignment_decision_coverage")):
        errors.append(f"{prefix}:coverage_nonfinite")
    return errors, counts


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args()
    manifest = read_json(RUNTIME_MANIFEST)
    event_policy = read_json(EVENT_MANIFEST)
    expected_events = {str(item["event_id"]): dict(item) for item in event_policy.get("events", [])}
    expected_keys = {(event_id, branch) for event_id in expected_events for branch in BRANCHES}
    results = manifest.get("events", []) if isinstance(manifest.get("events"), list) else []
    observed_keys: set[tuple[str, str]] = set()
    duplicate_keys: list[list[str]] = []
    all_errors: list[str] = []
    counts: Counter[str] = Counter()
    artifact_summaries: list[dict[str, Any]] = []
    for event_result in results:
        if not isinstance(event_result, Mapping):
            all_errors.append("manifest_event_not_object")
            continue
        event_id = str(event_result.get("event_id"))
        event_frame = int(event_result.get("event_frame", expected_events.get(event_id, {}).get("event_frame", -1)))
        expected_hash = event_result.get("y_pre_semantic_hash")
        for branch_result in event_result.get("branches", []):
            if not isinstance(branch_result, Mapping):
                all_errors.append(f"{event_id}:branch_result_not_object")
                continue
            branch = str(branch_result.get("branch"))
            key = (event_id, branch)
            if key in observed_keys:
                duplicate_keys.append([event_id, branch])
                continue
            observed_keys.add(key)
            output = Path(str(branch_result.get("output", PUBLIC_ROOT / event_id / f"{branch}.jsonl")))
            if not output.is_file():
                all_errors.append(f"{event_id}/{branch}:missing_output")
                continue
            try:
                rows = read_jsonl(output)
            except Exception as exc:
                all_errors.append(f"{event_id}/{branch}:parse_error:{type(exc).__name__}:{exc}")
                continue
            expected_frames = list(range(event_frame, event_frame + HORIZON + 1))
            observed_frames = [row.get("frame") for row in rows]
            if observed_frames != expected_frames:
                all_errors.append(f"{event_id}/{branch}:frame_coverage_mismatch")
            frame_errors: list[str] = []
            for row in rows:
                errors, local_counts = validate_frame(
                    row,
                    event_id=event_id,
                    branch=branch,
                    event_frame=event_frame,
                    expected_y_pre_hash=expected_hash,
                )
                frame_errors.extend(errors)
                counts.update(local_counts)
            all_errors.extend(frame_errors)
            done_path = Path(str(branch_result.get("done", output.with_suffix(".done.json"))))
            if done_path.is_file():
                done = read_json(done_path)
                if done.get("output_sha256") != sha256_file(output):
                    all_errors.append(f"{event_id}/{branch}:output_hash_mismatch")
                if done.get("runtime_future_gt_used") is not False:
                    all_errors.append(f"{event_id}/{branch}:done_runtime_gt_not_false")
                if done.get("runtime_invariants", {}).get("invariant_violations"):
                    all_errors.append(f"{event_id}/{branch}:runtime_invariant_violation")
            else:
                all_errors.append(f"{event_id}/{branch}:missing_done_manifest")
            artifact_summaries.append(
                {
                    "event_id": event_id,
                    "branch": branch,
                    "output": str(output),
                    "row_count": len(rows),
                    "frame_error_count": len(frame_errors),
                    "legacy_birth_rows": sum(1 for row in rows for candidate in row.get("candidate_rows", []) if candidate.get("assignment_status") == "OUTER_BIRTH_ASSIGNED"),
                }
            )
    missing_keys = sorted(expected_keys - observed_keys)
    extra_keys = sorted(observed_keys - expected_keys)
    if missing_keys:
        all_errors.append(f"missing_branch_keys:{len(missing_keys)}")
    if extra_keys:
        all_errors.append(f"extra_branch_keys:{len(extra_keys)}")
    if duplicate_keys:
        all_errors.append("duplicate_branch_keys")
    full_coverage = not missing_keys and not extra_keys and not duplicate_keys and len(observed_keys) == len(expected_keys)
    strict_pass = bool(full_coverage and not all_errors and manifest.get("runtime_future_gt_used") is False)
    payload = {
        "schema_version": "N72R5R1_STAGE09_VALIDATION_V1",
        "status": "PASS_N72R5R1_PUBLIC_RUNTIME_VALIDATION" if strict_pass else "BLOCKED_N72R5R1_PUBLIC_RUNTIME_VALIDATION",
        "strict_pass": strict_pass,
        "allow_partial_requested": bool(args.allow_partial),
        "expected_event_count": len(expected_events),
        "expected_branch_count": len(expected_keys),
        "observed_branch_count": len(observed_keys),
        "duplicate_branch_keys": duplicate_keys,
        "missing_branch_keys": [list(key) for key in missing_keys],
        "extra_branch_keys": [list(key) for key in extra_keys],
        "assignment_decision_coverage_definition": "formal candidate decision ASSIGNED_TO_PUBLIC_ID or EXPLICIT_NONE; identity decision ASSIGNED or NO_CANDIDATE_ASSIGNED",
        "counts": dict(sorted(counts.items())),
        "errors": all_errors[:2000],
        "error_count": len(all_errors),
        "artifact_summaries": artifact_summaries,
        "stage08_runtime_manifest_sha256": sha256_file(RUNTIME_MANIFEST),
        "stage06_event_manifest_sha256": sha256_file(EVENT_MANIFEST),
        "runtime_future_gt_used": False,
        "posthoc_gt_opened": False,
        "created_at_utc": now_utc(),
    }
    atomic_json(VALIDATION, payload)
    atomic_json(
        STATUS,
        {
            "schema_version": "N72R5R1_STAGE_STATUS_V1",
            "stage": "09_PUBLIC_RUNTIME_VALIDATION",
            "status": payload["status"],
            "strict_pass": strict_pass,
            "expected_branch_count": len(expected_keys),
            "observed_branch_count": len(observed_keys),
            "error_count": len(all_errors),
            "runtime_future_gt_used": False,
            "posthoc_gt_opened": False,
            "validation": str(VALIDATION),
            "minimal_next_action": None if strict_pass else "complete or repair every missing Stage08 event branch before Stage10",
            "created_at_utc": now_utc(),
        },
    )
    print(json.dumps({"status": payload["status"], "strict_pass": strict_pass, "observed_branches": len(observed_keys), "expected_branches": len(expected_keys), "errors": len(all_errors)}, ensure_ascii=False))
    return 0 if strict_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
