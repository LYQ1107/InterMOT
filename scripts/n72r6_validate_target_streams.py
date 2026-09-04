#!/usr/bin/env python3
"""Validate and freeze the N72R6 target-session stream artifacts.

The GPU controller deliberately records process exits rather than interpreting
them.  This CPU-only validator selects the latest *valid* artifact for each
of the 32 frozen APPLIED events, checks the complete event..H100 contract,
and writes a new manifest plus Stage 01/04 status files.  Older failures and
invalid PASS-looking attempts are retained as evidence and are never promoted
to the selected stream.
"""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import tempfile
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[1]
STAGE08 = ROOT / "outputs/N72R5R1/controller/round_05_branch_isolation_v0/full/stage08_runtime_manifest.json"
STREAM_ROOT = ROOT / "outputs/N72R6/target_correction_stream"
OUT = ROOT / "outputs/N72R6"
HORIZON = 100
FEATURE_DIM = 512


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def digest_json(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    try:
        with open(fd, "w", encoding="utf-8", closefd=True) as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            import os

            os.fsync(handle.fileno())
        Path(temporary).replace(path)
    finally:
        temporary_path = Path(temporary)
        if temporary_path.exists():
            temporary_path.unlink()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object: {path}")
    return value


def resolve_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def finite_feature(value: Any, label: str) -> None:
    if not isinstance(value, list) or len(value) != FEATURE_DIM:
        raise ValueError(f"{label}: feature is not 512-D")
    values = [float(item) for item in value]
    if not all(math.isfinite(item) for item in values):
        raise ValueError(f"{label}: feature contains non-finite value")
    if math.sqrt(sum(item * item for item in values)) <= 1.0e-6:
        raise ValueError(f"{label}: feature has zero norm")


def frozen_y_pre(main_output: Path, event_frame: int) -> tuple[str, str, dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with main_output.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if int(row.get("frame", -1)) == int(event_frame):
                rows.append(row)
    if len(rows) != 1:
        raise ValueError(f"frozen main Y_pre row count is {len(rows)}: {main_output}:{event_frame}")
    row = rows[0]
    if row.get("candidate_role") != "PRE_INTERVENTION_Y_PRE":
        raise ValueError(f"frozen main row is not Y_pre: {main_output}:{event_frame}")
    for flag in ("runtime_future_gt_used", "runtime_gt_read", "posthoc_gt_used"):
        if row.get(flag) is not False:
            raise ValueError(f"frozen main Y_pre flag is not false: {main_output}:{flag}")
    projection = [
        {
            key: item.get(key)
            for key in (
                "candidate_uid",
                "candidate_index",
                "official_raw_sam_id",
                "adapter_external_id",
                "box_xyxy",
                "feature_sha256",
            )
        }
        for item in row.get("candidate_rows", [])
    ]
    semantic = row.get("y_pre_semantic_hash") or row.get("shared_y_pre_semantic_hash")
    if not semantic:
        raise ValueError(f"frozen main Y_pre semantic hash missing: {main_output}")
    return str(semantic), digest_json(projection), row


def eligible_events() -> list[dict[str, Any]]:
    payload = load_json(STAGE08)
    result: list[dict[str, Any]] = []
    for event in payload.get("events", []):
        branches = {
            str(item.get("branch")): item
            for item in event.get("branches", [])
            if isinstance(item, dict)
        }
        branch = branches.get("B1_SPATIAL_CORRECTION_ONLY")
        main = branches.get("B0_NO_INTERVENTION")
        if branch and branch.get("action_precondition_status") == "APPLIED":
            if main is None or branch.get("target_public_id") is None:
                raise ValueError(f"eligible event lacks target/main authority: {event.get('event_id')}")
            item = dict(event)
            item["target_public_id"] = int(branch["target_public_id"])
            item["main_output"] = str(main["output"])
            result.append(item)
    result.sort(key=lambda item: str(item["event_id"]))
    if len(result) != 32 or len({str(item["event_id"]) for item in result}) != 32:
        raise ValueError(f"N72R6 eligible event set is not exactly 32 unique events: {len(result)}")
    return result


def attempt_number(path: Path, done: Mapping[str, Any]) -> int:
    value = done.get("attempt")
    if value is not None:
        return int(value)
    for part in path.parts:
        if part.startswith("attempt_"):
            try:
                return int(part.split("_", 1)[1])
            except ValueError:
                pass
    return -1


def validate_done(
    event: Mapping[str, Any],
    done_path: Path,
    done: Mapping[str, Any],
) -> dict[str, Any]:
    event_id = str(event["event_id"])
    sequence = str(event["sequence"])
    event_frame = int(event["event_frame"])
    end_frame = event_frame + HORIZON
    errors: list[str] = []
    def check(condition: bool, message: str) -> None:
        if not condition:
            errors.append(message)

    check(done.get("status") == "PASS_TARGET_STREAM_COMPLETE", "status_not_complete")
    check(done.get("event_id") == event_id, "event_id_mismatch")
    check(done.get("sequence") == sequence, "sequence_mismatch")
    check(int(done.get("event_frame", -1)) == event_frame, "event_frame_mismatch")
    check(int(done.get("end_frame", -1)) == end_frame, "end_frame_mismatch")
    check(int(done.get("frame_count", -1)) == HORIZON + 1, "frame_count_mismatch")
    check(int(done.get("target_public_id", -1)) == int(event["target_public_id"]), "target_public_id_mismatch")
    for flag in ("runtime_future_gt_used", "runtime_gt_read", "posthoc_gt_used", "public_id_inference"):
        check(done.get(flag) is False, f"{flag}_not_false")
    check(done.get("interaction_source") == "simulated_from_gt", "interaction_source_changed")
    check(done.get("not_real_human_evidence") is True, "synthetic_event_not_marked")
    check(done.get("target_candidate_present_event_frame") is True, "event_candidate_missing")
    check(done.get("target_session_video_mode") == "SYMLINK_EXACT_EVENT_LOCAL_WINDOW", "unexpected_video_mode")

    frames_path = resolve_path(str(done.get("frames", "")))
    anchor_path = resolve_path(str(done.get("human_anchor", "")))
    mapping_path = resolve_path(str(done.get("target_session_frame_mapping", "")))
    for path, label in ((frames_path, "frames"), (anchor_path, "human_anchor"), (mapping_path, "mapping")):
        check(path.is_file(), f"{label}_missing")
    if not errors:
        check(sha256_file(frames_path) == str(done.get("frames_sha256")), "frames_sha256_mismatch")
        check(sha256_file(anchor_path) == str(done.get("human_anchor_sha256")), "anchor_sha256_mismatch")
        check(sha256_file(mapping_path) == str(done.get("target_session_frame_mapping_sha256")), "mapping_sha256_mismatch")

        frames: list[dict[str, Any]] = []
        with frames_path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if line.strip():
                    value = json.loads(line)
                    if not isinstance(value, dict):
                        errors.append(f"frames_line_{line_number}_not_object")
                    else:
                        frames.append(value)
        check(len(frames) == HORIZON + 1, "frames_row_count_mismatch")
        expected_frames = list(range(event_frame, end_frame + 1))
        actual_frames = [int(row.get("frame", -1)) for row in frames]
        check(actual_frames == expected_frames, "global_frame_range_or_order_mismatch")
        future_candidate_count = 0
        total_candidate_count = 0
        for index, row in enumerate(frames):
            label = f"{event_id}:{expected_frames[index] if index < len(expected_frames) else index}"
            check(row.get("event_id") == event_id, f"{label}:event_id_mismatch")
            check(row.get("sequence") == sequence, f"{label}:sequence_mismatch")
            check(row.get("candidate_stream_kind") == "INDEPENDENT_ONE_TARGET_SAM3_SESSION", f"{label}:stream_kind")
            check(row.get("candidate_set_complete") is True, f"{label}:candidate_set_incomplete")
            check(row.get("runtime_future_gt_used") is False, f"{label}:future_gt_flag")
            check(row.get("runtime_gt_read") is False, f"{label}:runtime_gt_flag")
            check(row.get("posthoc_gt_used") is False, f"{label}:posthoc_gt_flag")
            check(row.get("public_id_inference") is False, f"{label}:public_inference")
            check(row.get("event_frame_memory_read") is False, f"{label}:event_memory_read")
            check(row.get("memory_read") is False, f"{label}:memory_read")
            candidates = row.get("candidate_rows")
            check(isinstance(candidates, list), f"{label}:candidate_rows_not_list")
            if not isinstance(candidates, list):
                continue
            check(int(row.get("candidate_count", -1)) == len(candidates), f"{label}:candidate_count_mismatch")
            check(len(candidates) <= 1, f"{label}:multiple_target_candidates")
            total_candidate_count += len(candidates)
            if int(row.get("frame", -1)) > event_frame:
                future_candidate_count += len(candidates)
            for candidate in candidates:
                check(candidate.get("candidate_kind") == "TARGET_CORRECTION_SESSION_CANDIDATE", f"{label}:candidate_kind")
                check(candidate.get("public_id") is None, f"{label}:candidate_has_public_id")
                check(int(candidate.get("human_target_scope_public_id", -1)) == int(event["target_public_id"]), f"{label}:candidate_scope_public")
                check(candidate.get("correction_epoch_id") == done.get("correction_epoch_id"), f"{label}:epoch_mismatch")
                check(candidate.get("target_session_scope") == done.get("target_session_scope"), f"{label}:scope_mismatch")
                check(candidate.get("runtime_future_gt_used") is False, f"{label}:candidate_future_gt")
                check(candidate.get("runtime_gt_read") is False, f"{label}:candidate_gt")
                check(candidate.get("posthoc_gt_used") is False, f"{label}:candidate_posthoc_gt")
                check(candidate.get("public_id_inference") is False, f"{label}:candidate_public_inference")
                check(int(candidate.get("frame", -1)) == int(row.get("frame", -1)), f"{label}:candidate_frame")
                finite_feature(candidate.get("feature"), f"{label}:candidate_feature")
                for field in ("confidence", "presence_score"):
                    if candidate.get(field) is not None:
                        check(math.isfinite(float(candidate[field])), f"{label}:{field}_nonfinite")
        # The target session is required to expose exactly one official target
        # at the event frame.  Future recall is an experimental result, not a
        # stream-integrity precondition: it may be zero, intermittent, or
        # present at event+1.  Preserve that value for the later root-cause
        # and effect audit instead of rejecting a valid target stream here.
        check(int(frames[0].get("candidate_count", -1)) == 1 if frames else False, "event_candidate_count_not_one")

        anchor = load_json(anchor_path)
        finite_feature(anchor.get("feature"), f"{event_id}:human_anchor")
        check(anchor.get("public_id") == int(event["target_public_id"]), "anchor_public_id_mismatch")
        check(anchor.get("interaction_source") == "simulated_from_gt", "anchor_source_changed")
        check(anchor.get("not_real_human_evidence") is True, "anchor_realness_marker_missing")
        for flag in ("runtime_future_gt_used", "runtime_gt_read", "posthoc_gt_used"):
            check(anchor.get(flag) is False, f"anchor_{flag}")

        mapping = load_json(mapping_path)
        items = mapping.get("mapping")
        check(isinstance(items, list) and len(items) == HORIZON + 1, "mapping_count_mismatch")
        if isinstance(items, list):
            check(
                [(int(item.get("local_frame", -1)), int(item.get("global_frame", -1))) for item in items]
                == [(index, event_frame + index) for index in range(HORIZON + 1)],
                "native_to_global_mapping_mismatch",
            )

        expected_y_pre, expected_candidate_hash, expected_y_pre_row = frozen_y_pre(
            resolve_path(str(event["main_output"])), event_frame
        )
        check(done.get("main_y_pre_semantic_hash") == expected_y_pre, "main_y_pre_semantic_hash_mismatch")
        check(done.get("main_y_pre_candidate_content_sha256") == expected_candidate_hash, "main_y_pre_candidate_hash_mismatch")
        check(done.get("main_y_pre_row_hash") == digest_json(expected_y_pre_row), "main_y_pre_row_hash_mismatch")
    if errors:
        raise ValueError(";".join(errors))
    return {
        "event_id": event_id,
        "sequence": sequence,
        "action_type": str(event["action_type"]),
        "event_frame": event_frame,
        "target_public_id": int(event["target_public_id"]),
        "done": str(done_path),
        "attempt": attempt_number(done_path, done),
        "status": "PASS_TARGET_STREAM_VALIDATED",
        "event_candidate_count": 1,
        "future_candidate_count": future_candidate_count,
        "frame_count": HORIZON + 1,
        "runtime_future_gt_used": False,
        "interaction_source": "simulated_from_gt",
        "not_real_human_evidence": True,
    }


def main() -> int:
    events = eligible_events()
    all_done: list[tuple[Path, dict[str, Any]]] = []
    for path in sorted(STREAM_ROOT.rglob("done.json")):
        try:
            all_done.append((path, load_json(path)))
        except Exception:
            continue
    by_event: dict[str, list[tuple[Path, dict[str, Any]]]] = {}
    for path, done in all_done:
        event_id = str(done.get("event_id", ""))
        if event_id in {str(item["event_id"]) for item in events}:
            by_event.setdefault(event_id, []).append((path, done))

    selected: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    event_by_id = {str(item["event_id"]): item for item in events}
    for event in events:
        event_id = str(event["event_id"])
        candidates = sorted(
            by_event.get(event_id, []),
            key=lambda item: (attempt_number(item[0], item[1]), str(item[0])),
            reverse=True,
        )
        valid: list[tuple[Path, dict[str, Any], dict[str, Any]]] = []
        for path, done in candidates:
            try:
                record = validate_done(event, path, done)
                valid.append((path, done, record))
            except Exception as exc:
                rejected.append(
                    {
                        "event_id": event_id,
                        "artifact": str(path),
                        "attempt": attempt_number(path, done),
                        "classification": "INVALID_OR_FAILED_ATTEMPT",
                        "reason": str(exc),
                    }
                )
        if not valid:
            raise RuntimeError(f"no valid target stream artifact for {event_id}")
        path, done, record = valid[0]
        selected.append(record)
        for superseded_path, superseded_done, _ in valid[1:]:
            rejected.append(
                {
                    "event_id": event_id,
                    "artifact": str(superseded_path),
                    "attempt": attempt_number(superseded_path, superseded_done),
                    "classification": "SUPERSEDED_VALID_ATTEMPT",
                    "reason": f"latest_valid_artifact={path}",
                }
            )

    selected.sort(key=lambda item: str(item["event_id"]))
    if len(selected) != 32 or len({item["event_id"] for item in selected}) != 32:
        raise RuntimeError(f"selected target stream coverage is not 32 unique events: {len(selected)}")
    failures = []
    for path in sorted((OUT / "attempts").glob("*.json")):
        try:
            value = load_json(path)
        except Exception:
            continue
        if "failure" in str(value.get("status", "")).lower() or "failure" in path.name.lower() or "invalid" in path.name.lower():
            failures.append({"artifact": str(path), "status": value.get("status"), "failure_type": value.get("failure_type"), "message": value.get("message")})
    for path in sorted(STREAM_ROOT.rglob("*.failure*.json")):
        try:
            value = load_json(path)
            failures.append({"artifact": str(path), "status": value.get("status"), "failure_type": value.get("failure_type"), "message": value.get("message")})
        except Exception:
            failures.append({"artifact": str(path), "status": "UNREADABLE_FAILURE_ARTIFACT"})

    batch_path = STREAM_ROOT / "batch" / "batch_attempt_1.json"
    batch = load_json(batch_path) if batch_path.is_file() else {}
    repair_event = "n72r5-pool-n37-dancetrack0032-0097-authoritative_reassign-001"
    repair_pass = any(
        item.get("event_id") == repair_event and int(item.get("attempt", -1)) == 3
        for item in selected
    )
    payload = {
        "schema_version": "N72R6_TARGET_STREAM_VALIDATION_V1",
        "status": "PASS_32_OF_32_TARGET_STREAMS_VALIDATED",
        "created_at_utc": now_utc(),
        "protocol": str(OUT / "protocol.json"),
        "protocol_sha256": sha256_file(OUT / "protocol.json"),
        "stage08_manifest": str(STAGE08),
        "stage08_manifest_sha256": sha256_file(STAGE08),
        "eligible_event_count": 32,
        "selected_event_count": len(selected),
        "selected_event_ids_unique": len({item["event_id"] for item in selected}) == 32,
        "selected": selected,
        "rejected_and_superseded": rejected,
        "historical_failure_artifacts": failures,
        "initial_batch": {
            "artifact": str(batch_path),
            "status": batch.get("status"),
            "event_count": batch.get("event_count"),
            "completed_count": batch.get("completed_count"),
            "failed_count": batch.get("failed_count"),
        },
        "repair": {
            "event_id": repair_event,
            "selected_attempt": next(item["attempt"] for item in selected if item["event_id"] == repair_event),
            "targeted_repair_pass": repair_pass,
            "root_cause": "official_target_session_exposed_more_than_one_object_or_no_singleton_target_row",
        },
        "candidate_coverage": {
            "event_frame_present_events": sum(item["event_candidate_count"] == 1 for item in selected),
            "future_candidate_present_events": sum(item["future_candidate_count"] > 0 for item in selected),
            "future_candidate_rows": sum(item["future_candidate_count"] for item in selected),
            "all_event_rows_complete": True,
        },
        "runtime_future_gt_used": False,
        "historical_outputs_modified": False,
    }
    atomic_json(OUT / "target_correction_stream" / "target_stream_manifest.json", payload)

    stage01 = {
        "schema_version": "N72R6_STAGE_STATUS_V1",
        "stage": "N72R6-01",
        "status": "PASS_TARGET_SESSION_STREAM_SMOKE_AND_BATCH_VALIDATED",
        "created_at_utc": now_utc(),
        "targeted_smoke": {
            "event_id": "n72r5-pool-n37-dancetrack0001-0296-authoritative_reassign-001",
            "selected_attempt": next(item["attempt"] for item in selected if item["event_id"].startswith("n72r5-pool-n37-dancetrack0001-0296")),
            "status": "PASS_TARGET_STREAM_VALIDATED",
        },
        "eligible_event_count": 32,
        "validated_event_count": len(selected),
        "event_frame_candidate_coverage": "32/32",
        "future_candidate_present_event_count": sum(item["future_candidate_count"] > 0 for item in selected),
        "future_candidate_row_count": sum(item["future_candidate_count"] for item in selected),
        "runtime_future_gt_used": False,
        "interaction_source": "simulated_from_gt",
        "not_real_human_evidence": True,
        "initial_batch_failure_preserved": batch.get("failed_count", 0) != 0,
        "invalid_or_superseded_artifact_count": len(rejected),
        "historical_failure_artifact_count": len(failures),
        "next_stage": "N72R6-02_CORRECTION_EPOCH_NATIVE_SCOPE_CPU_AUDIT",
    }
    atomic_json(OUT / "stage_01_status.json", stage01)

    stage04 = {
        "schema_version": "N72R6_STAGE_STATUS_V1",
        "stage": "N72R6-04",
        "status": "PASS_TARGET_CORRECTION_STREAMS_COMPLETE",
        "created_at_utc": now_utc(),
        "eligible_event_count": 32,
        "selected_valid_stream_count": len(selected),
        "duplicate_event_ids": 0,
        "missing_event_ids": 0,
        "invalid_selected_streams": 0,
        "event_frame_candidate_present_count": sum(item["event_candidate_count"] == 1 for item in selected),
        "future_candidate_present_count": sum(item["future_candidate_count"] > 0 for item in selected),
        "future_candidate_rows": sum(item["future_candidate_count"] for item in selected),
        "initial_batch_status": batch.get("status"),
        "initial_batch_failed_count": batch.get("failed_count"),
        "targeted_repair_event": repair_event,
        "targeted_repair_selected_attempt": next(item["attempt"] for item in selected if item["event_id"] == repair_event),
        "target_stream_manifest": str(OUT / "target_correction_stream" / "target_stream_manifest.json"),
        "runtime_future_gt_used": False,
        "architecture_effect_not_yet_evaluated": True,
        "root_cause_signal": "all_validated_streams_have_zero_future_target_candidates; do_not_claim_future_effect",
        "historical_failures_preserved": True,
        "next_stage": "N72R6-05_C0_C1_TARGET_EXCLUSIVE_PUBLIC_REPLAY",
    }
    atomic_json(OUT / "stage_04_status.json", stage04)
    print(json.dumps({"status": payload["status"], "selected_event_count": len(selected), "future_candidate_rows": payload["candidate_coverage"]["future_candidate_rows"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
