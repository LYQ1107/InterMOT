"""Lossless ingestion and validation for externally collected human events.

This module is deliberately independent of the GT-driven observers and of the
SAM3 runtime.  It validates an event *before* a runner is allowed to turn it
into a :class:`HumanInteraction`.  ``source='human'`` on that lower-level
dataclass is not evidence of provenance; this validator requires an explicit
UI/session/annotator record and a losslessly recoverable human input.

The validator is CPU-only and never imports a dataset reader, simulator, or
future evaluator.  Candidate tapes are read as ordinary JSON/JSONL artifacts
and are checked for frame completeness, duplicate keys, and native/local/global
mapping consistency.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import tempfile
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Optional


PROTOCOL_ID = "N40_REAL_HUMAN_EVENT_TAPE_V1"
CANDIDATE_TAPE_SCHEMA = "N40_CANDIDATE_TAPE_V1"
ALLOWED_ACTIONS = frozenset(
    {
        "ADD_NEW_IDENTITY",
        "AUTHORITATIVE_REASSIGN",
        "ATOMIC_ID_SWAP",
        "RECOVER_IDENTITY",
    }
)
ALLOWED_INPUT_KINDS = frozenset({"BOX", "CLICK", "CONFIRMED_MASK"})
ALLOWED_MASK_FORMATS = frozenset({"lossless_RLE", "lossless_PNG", "lossless_NPZ"})
HASH_RE = re.compile(r"^[0-9a-fA-F]{64}$")
PLACEHOLDER_VALUES = frozenset(
    {
        "",
        "none",
        "null",
        "unknown",
        "simulated_oracle",
        "simulated_from_gt",
        "offline_labeling",
        "correction_simulator",
        "sim_gt_current_frame",
    }
)
FORBIDDEN_GT_KEYS = frozenset(
    {
        "gt",
        "gt_box",
        "gt_id",
        "dataset_gt_id",
        "dataset_identity",
        "ground_truth",
        "future_gt",
        "future_identity",
        "posthoc_gt",
        "gt_used",
        "future_gt_used",
    }
)
FORBIDDEN_SOURCE_TOKENS = (
    "simulated_from_gt",
    "sim_gt",
    "simulated_oracle",
    "correction_simulator",
    "offline_labeling",
    "oracle",
)


def _is_finite_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def _is_hash(value: Any) -> bool:
    return isinstance(value, str) and HASH_RE.fullmatch(value) is not None


def _nonempty_string(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _safe_name(value: Any) -> str:
    text = str(value) if value is not None else "unknown"
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", text).strip("._")
    return text[:80] or "unknown"


def _atomic_write_json(path: Path, payload: Any) -> None:
    """Write one failure/report artifact without exposing a partial JSON file."""

    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        directory_fd = os.open(str(path.parent), os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _issue(code: str, path: str, message: str) -> dict[str, str]:
    return {"code": code, "path": path, "message": message}


@dataclass
class ValidationResult:
    """Machine-readable result for one event."""

    valid: bool
    event_id: Optional[str] = None
    errors: list[dict[str, str]] = field(default_factory=list)
    warnings: list[dict[str, str]] = field(default_factory=list)
    candidate_frames_checked: int = 0
    candidate_rows_checked: int = 0
    normalized_record: Optional[dict[str, Any]] = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "valid": self.valid,
            "event_id": self.event_id,
            "errors": self.errors,
            "warnings": self.warnings,
            "candidate_frames_checked": self.candidate_frames_checked,
            "candidate_rows_checked": self.candidate_rows_checked,
        }


def _walk_forbidden_keys(value: Any, path: str = "") -> list[dict[str, str]]:
    issues: list[dict[str, str]] = []
    if isinstance(value, dict):
        for key, child in value.items():
            key_text = str(key)
            if key_text.lower() in FORBIDDEN_GT_KEYS:
                issues.append(
                    _issue(
                        "gt_derived_field_forbidden",
                        f"{path}/{key_text}",
                        "runtime event/tape records must not contain GT or future-identity fields",
                    )
                )
            issues.extend(_walk_forbidden_keys(child, f"{path}/{key_text}"))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            issues.extend(_walk_forbidden_keys(child, f"{path}/{index}"))
    return issues


def _walk_forbidden_sources(value: Any, path: str = "") -> list[dict[str, str]]:
    issues: list[dict[str, str]] = []
    if isinstance(value, dict):
        for key, child in value.items():
            key_text = str(key)
            if key_text.lower() in {
                "source",
                "interaction_source",
                "mask_origin",
                "provenance",
                "source_kind",
            } and isinstance(child, str):
                lowered = child.strip().lower()
                if lowered in PLACEHOLDER_VALUES or any(
                    token in lowered for token in FORBIDDEN_SOURCE_TOKENS
                ):
                    issues.append(
                        _issue(
                            "non_real_source_forbidden",
                            f"{path}/{key_text}",
                            f"forbidden simulated/GT-derived source value: {child}",
                        )
                    )
            issues.extend(_walk_forbidden_sources(child, f"{path}/{key_text}"))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            issues.extend(_walk_forbidden_sources(child, f"{path}/{index}"))
    return issues


def _parse_timestamp(value: Any, path: str, errors: list[dict[str, str]]) -> Optional[datetime]:
    if not _nonempty_string(value):
        errors.append(_issue("missing_provenance", path, "ISO-8601 timestamp is required"))
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        errors.append(_issue("invalid_timestamp", path, "timestamp is not valid ISO-8601"))
        return None
    if parsed.tzinfo is None:
        errors.append(_issue("timestamp_timezone_missing", path, "timestamp must include timezone"))
    return parsed


def _require(record: dict[str, Any], key: str, errors: list[dict[str, str]]) -> Any:
    if key not in record:
        errors.append(_issue("required_field_missing", f"/{key}", "required field is missing"))
        return None
    return record[key]


def _require_public_id(record: dict[str, Any], key: str, errors: list[dict[str, str]]) -> Any:
    value = _require(record, key, errors)
    if value is None:
        return value
    if isinstance(value, bool) or not isinstance(value, (int, str)):
        errors.append(_issue("invalid_public_id", f"/{key}", "public ID must be an integer or non-empty string"))
    elif isinstance(value, str) and not value.strip():
        errors.append(_issue("invalid_public_id", f"/{key}", "public ID cannot be empty"))
    return value


def _validate_input(record: dict[str, Any], errors: list[dict[str, str]]) -> None:
    human_input = _require(record, "human_input", errors)
    if not isinstance(human_input, dict):
        errors.append(_issue("invalid_human_input", "/human_input", "human_input must be an object"))
        return
    kind = human_input.get("kind")
    if kind not in ALLOWED_INPUT_KINDS:
        errors.append(_issue("invalid_human_input_kind", "/human_input/kind", "kind must be BOX, CLICK or CONFIRMED_MASK"))
        return
    if human_input.get("origin") != "human_ui":
        errors.append(_issue("human_origin_not_proven", "/human_input/origin", "raw input must be emitted by the external human UI"))
    if human_input.get("human_confirmed") is not True:
        errors.append(_issue("human_confirmation_missing", "/human_input/human_confirmed", "raw input must be explicitly confirmed by a human"))
    if not _is_hash(human_input.get("raw_payload_sha256")):
        errors.append(_issue("raw_input_digest_missing", "/human_input/raw_payload_sha256", "lossless raw input digest is required"))

    if kind == "BOX":
        box = human_input.get("box_xyxy")
        if not isinstance(box, list) or len(box) != 4 or not all(_is_finite_number(x) for x in box):
            errors.append(_issue("invalid_human_box", "/human_input/box_xyxy", "box must contain four finite coordinates"))
        elif float(box[2]) <= float(box[0]) or float(box[3]) <= float(box[1]):
            errors.append(_issue("invalid_human_box", "/human_input/box_xyxy", "box must have positive width and height"))
    elif kind == "CLICK":
        points = human_input.get("click_points")
        if not isinstance(points, list) or not points:
            errors.append(_issue("invalid_clicks", "/human_input/click_points", "at least one click is required"))
        else:
            positive = 0
            for index, point in enumerate(points):
                if not isinstance(point, dict) or not _is_finite_number(point.get("x")) or not _is_finite_number(point.get("y")):
                    errors.append(_issue("invalid_click", f"/human_input/click_points/{index}", "click needs finite x and y"))
                    continue
                if point.get("label") not in {"positive", "negative"}:
                    errors.append(_issue("invalid_click_label", f"/human_input/click_points/{index}/label", "label must be positive or negative"))
                positive += int(point.get("label") == "positive")
            if positive == 0:
                errors.append(_issue("positive_click_missing", "/human_input/click_points", "at least one positive click is required"))
    else:
        mask = human_input.get("confirmed_mask")
        if not isinstance(mask, dict):
            errors.append(_issue("confirmed_mask_missing", "/human_input/confirmed_mask", "confirmed mask object is required"))
        else:
            if mask.get("format") not in ALLOWED_MASK_FORMATS:
                errors.append(_issue("invalid_mask_format", "/human_input/confirmed_mask/format", "mask must use a lossless format"))
            if not _nonempty_string(mask.get("payload_ref")):
                errors.append(_issue("mask_payload_missing", "/human_input/confirmed_mask/payload_ref", "lossless mask payload reference is required"))
            if not _is_hash(mask.get("sha256")):
                errors.append(_issue("mask_digest_missing", "/human_input/confirmed_mask/sha256", "mask digest is required"))
            shape = mask.get("frame_shape")
            if not isinstance(shape, list) or len(shape) != 2 or not all(isinstance(x, int) and not isinstance(x, bool) and x > 0 for x in shape):
                errors.append(_issue("invalid_mask_shape", "/human_input/confirmed_mask/frame_shape", "mask frame_shape must be [height,width]"))
            if mask.get("mask_origin") != "human_ui_confirmed":
                errors.append(_issue("machine_mask_not_allowed", "/human_input/confirmed_mask/mask_origin", "confirmed mask must originate in the human UI"))
            if mask.get("machine_candidate_mask") is not False:
                errors.append(_issue("machine_mask_not_allowed", "/human_input/confirmed_mask/machine_candidate_mask", "machine candidate masks cannot be relabeled as confirmed masks"))


def _validate_ranges(record: dict[str, Any], frame: int, errors: list[dict[str, str]]) -> set[int]:
    prefix = _require(record, "prefix_range", errors)
    if not isinstance(prefix, list) or len(prefix) != 2 or not all(isinstance(x, int) and not isinstance(x, bool) for x in prefix):
        errors.append(_issue("invalid_prefix_range", "/prefix_range", "prefix_range must be [start,end] integer frames"))
    elif prefix[0] < 0 or prefix[0] > prefix[1] or prefix[1] != frame - 1:
        errors.append(_issue("invalid_prefix_range", "/prefix_range", "prefix must end at event_frame-1"))

    future = _require(record, "future_ranges", errors)
    expected: set[int] = set()
    if not isinstance(future, dict):
        errors.append(_issue("invalid_future_ranges", "/future_ranges", "future_ranges must be an object"))
        return expected
    for horizon in (20, 50, 100):
        key = f"H{horizon}"
        item = future.get(key)
        if not isinstance(item, list) or len(item) != 2 or not all(isinstance(x, int) and not isinstance(x, bool) for x in item):
            errors.append(_issue("invalid_future_range", f"/future_ranges/{key}", "range must be [event_frame+1,event_frame+H]"))
            continue
        if item != [frame + 1, frame + horizon]:
            errors.append(_issue("future_boundary_mismatch", f"/future_ranges/{key}", "future range does not match frozen horizon"))
        expected.update(range(frame + 1, frame + horizon + 1))
    return expected


def _validate_candidate_mapping(row: dict[str, Any], index: int, errors: list[dict[str, str]]) -> tuple[Any, ...] | None:
    path = f"/candidate_tape/rows/{index}"
    for key in ("candidate_native_id", "candidate_local_id", "sequence_global_id", "mapping"):
        if key not in row:
            errors.append(_issue("candidate_mapping_missing", f"{path}/{key}", "native/local/global mapping fields are required"))
    mapping = row.get("mapping")
    if not isinstance(mapping, dict):
        errors.append(_issue("candidate_mapping_unresolved", f"{path}/mapping", "mapping must be an object"))
        return None
    for key in ("native_id", "local_id", "sequence_global_id", "public_id", "resolution_status"):
        if key not in mapping:
            errors.append(_issue("candidate_mapping_missing", f"{path}/mapping/{key}", "mapping field is required"))
    if mapping.get("native_id") != row.get("candidate_native_id"):
        errors.append(_issue("candidate_mapping_mismatch", f"{path}/mapping/native_id", "native mapping disagrees with row"))
    if mapping.get("local_id") != row.get("candidate_local_id"):
        errors.append(_issue("candidate_mapping_mismatch", f"{path}/mapping/local_id", "local mapping disagrees with row"))
    if mapping.get("sequence_global_id") != row.get("sequence_global_id"):
        errors.append(_issue("candidate_mapping_mismatch", f"{path}/mapping/sequence_global_id", "global mapping disagrees with row"))
    if mapping.get("resolution_status") not in {"RESOLVED", "UNASSIGNED"}:
        errors.append(_issue("candidate_mapping_unresolved", f"{path}/mapping/resolution_status", "status must be RESOLVED or UNASSIGNED"))
    if mapping.get("resolution_status") == "RESOLVED" and mapping.get("public_id") is None:
        errors.append(_issue("candidate_mapping_unresolved", f"{path}/mapping/public_id", "resolved candidate must have a public ID"))
    if not _nonempty_string(row.get("candidate_local_id")) or not _nonempty_string(row.get("sequence_global_id")):
        errors.append(_issue("candidate_mapping_unresolved", path, "local/global IDs cannot be empty"))
    return (
        row.get("frame_id"),
        row.get("candidate_native_id"),
        row.get("candidate_local_id"),
        row.get("sequence_global_id"),
    )


def _validate_candidate_tape(
    record: dict[str, Any], tape: Any, expected_future_frames: set[int], errors: list[dict[str, str]]
) -> tuple[int, int]:
    if not isinstance(tape, dict):
        errors.append(_issue("candidate_tape_unavailable", "/candidate_tape", "candidate tape must be a decoded object"))
        return 0, 0
    if tape.get("schema") != CANDIDATE_TAPE_SCHEMA:
        errors.append(_issue("candidate_tape_schema_mismatch", "/candidate_tape/schema", f"expected {CANDIDATE_TAPE_SCHEMA}"))
    if tape.get("sequence") != record.get("sequence"):
        errors.append(_issue("candidate_sequence_mismatch", "/candidate_tape/sequence", "candidate tape sequence differs from event"))
    manifest = tape.get("frame_manifest")
    rows = tape.get("rows")
    if not isinstance(manifest, list):
        errors.append(_issue("candidate_frame_manifest_missing", "/candidate_tape/frame_manifest", "explicit frame manifest is required"))
        manifest = []
    if not isinstance(rows, list):
        errors.append(_issue("candidate_rows_missing", "/candidate_tape/rows", "candidate rows must be a list"))
        rows = []
    manifest_counts: dict[int, int] = {}
    for index, item in enumerate(manifest):
        if not isinstance(item, dict) or not isinstance(item.get("frame_id"), int) or isinstance(item.get("frame_id"), bool):
            errors.append(_issue("invalid_candidate_frame_manifest", f"/candidate_tape/frame_manifest/{index}", "frame_id must be an integer"))
            continue
        frame_id = item["frame_id"]
        if frame_id in manifest_counts:
            errors.append(_issue("duplicate_candidate_frame", f"/candidate_tape/frame_manifest/{index}", f"frame {frame_id} is listed more than once"))
        count = item.get("candidate_count")
        if not isinstance(count, int) or isinstance(count, bool) or count < 0:
            errors.append(_issue("invalid_candidate_count", f"/candidate_tape/frame_manifest/{index}/candidate_count", "candidate_count must be a non-negative integer"))
            count = -1
        manifest_counts[frame_id] = count
    required_frames = {record["event_frame"]} | expected_future_frames
    missing_manifest = sorted(required_frames - set(manifest_counts))
    for frame_id in missing_manifest:
        errors.append(_issue("missing_candidate_frame", "/candidate_tape/frame_manifest", f"required frame {frame_id} is absent"))

    rows_by_frame: dict[int, int] = {}
    keys: set[tuple[Any, ...]] = set()
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            errors.append(_issue("invalid_candidate_row", f"/candidate_tape/rows/{index}", "candidate row must be an object"))
            continue
        frame_id = row.get("frame_id")
        if not isinstance(frame_id, int) or isinstance(frame_id, bool):
            errors.append(_issue("invalid_candidate_frame", f"/candidate_tape/rows/{index}/frame_id", "frame_id must be an integer"))
            continue
        rows_by_frame[frame_id] = rows_by_frame.get(frame_id, 0) + 1
        if frame_id not in manifest_counts:
            errors.append(_issue("candidate_frame_not_manifested", f"/candidate_tape/rows/{index}/frame_id", f"frame {frame_id} is not in frame_manifest"))
        key = _validate_candidate_mapping(row, index, errors)
        if key is not None:
            if key in keys:
                errors.append(_issue("duplicate_candidate_key", f"/candidate_tape/rows/{index}", f"duplicate candidate key {key}"))
            keys.add(key)
        if row.get("runtime_future_gt_used") is not False:
            errors.append(_issue("runtime_future_gt_used", f"/candidate_tape/rows/{index}/runtime_future_gt_used", "runtime future GT must be exactly false"))
        box = row.get("box_xyxy")
        if not isinstance(box, list) or len(box) != 4 or not all(_is_finite_number(x) for x in box):
            errors.append(_issue("invalid_candidate_box", f"/candidate_tape/rows/{index}/box_xyxy", "candidate box must contain four finite values"))
        confidence = row.get("candidate_confidence")
        if not _is_finite_number(confidence):
            errors.append(_issue("invalid_candidate_confidence", f"/candidate_tape/rows/{index}/candidate_confidence", "candidate confidence must be finite"))
        mask_digest = row.get("mask_sha256")
        if not _is_hash(mask_digest):
            errors.append(_issue("candidate_mask_digest_missing", f"/candidate_tape/rows/{index}/mask_sha256", "candidate mask digest is required"))
        feature = row.get("feature")
        if not isinstance(feature, dict) or feature.get("dim") != 512 or feature.get("finite") is not True or not _is_finite_number(feature.get("norm")) or float(feature.get("norm")) <= 0 or not _is_hash(feature.get("sha256")):
            errors.append(_issue("invalid_candidate_feature", f"/candidate_tape/rows/{index}/feature", "candidate feature must be finite 512-D with positive norm and digest"))

    for frame_id, expected_count in manifest_counts.items():
        if expected_count >= 0 and rows_by_frame.get(frame_id, 0) != expected_count:
            errors.append(_issue("candidate_count_mismatch", f"/candidate_tape/frame_manifest/{frame_id}", f"manifest says {expected_count} rows, found {rows_by_frame.get(frame_id, 0)}"))
    return len(manifest_counts), len(rows)


def validate_real_human_event(
    record: Any,
    *,
    candidate_tape: Any = None,
) -> ValidationResult:
    """Validate one external real-human event and its already-loaded tape.

    ``candidate_tape`` is passed by the adapter after resolving
    ``candidate_tape_ref``.  Keeping it separate prevents an event JSONL file
    from silently embedding a partial or un-audited future stream.
    """

    errors: list[dict[str, str]] = []
    warnings: list[dict[str, str]] = []
    if not isinstance(record, dict):
        return ValidationResult(False, errors=[_issue("invalid_event_record", "/", "event must be a JSON object")])
    errors.extend(_walk_forbidden_keys(record))
    errors.extend(_walk_forbidden_sources(record))

    event_id = record.get("event_id")
    event_id_text = str(event_id) if event_id is not None else None
    if not _nonempty_string(event_id):
        errors.append(_issue("event_id_missing", "/event_id", "stable event_id is required"))
    required = (
        "sequence",
        "split",
        "event_frame",
        "public_id",
        "action_type",
        "interaction_source",
        "human_confirmed",
        "annotator_id_hash",
        "session_id",
        "event_start_timestamp",
        "event_end_timestamp",
        "frame_image_sha256",
        "candidate_tape_ref",
        "prefix_range",
        "future_ranges",
        "spatial_correction",
        "mapping_audit",
        "memory_audit",
        "human_embedding",
        "runtime_future_gt_used",
    )
    for key in required:
        _require(record, key, errors)

    sequence = record.get("sequence")
    if not _nonempty_string(sequence):
        errors.append(_issue("sequence_missing", "/sequence", "sequence is required"))
    if record.get("split") not in {"train", "train_fold"}:
        errors.append(_issue("forbidden_split", "/split", "real tape collection only accepts train/train_fold"))
    frame = record.get("event_frame")
    if not isinstance(frame, int) or isinstance(frame, bool) or frame < 0:
        errors.append(_issue("invalid_event_frame", "/event_frame", "event_frame must be a non-negative integer"))
        frame = 0
    _require_public_id(record, "public_id", errors)
    if record.get("public_id_source") != "human_direct":
        errors.append(_issue("public_id_not_direct", "/public_id_source", "public_id must be supplied directly by the human"))
    action = record.get("action_type")
    if action not in ALLOWED_ACTIONS:
        errors.append(_issue("invalid_action_type", "/action_type", "action must be one of the four N40 actions"))
    if action == "AUTHORITATIVE_REASSIGN":
        current = _require_public_id(record, "current_public_id", errors)
        if current == record.get("public_id"):
            errors.append(_issue("invalid_reassign_ids", "/current_public_id", "current and destination public IDs must differ"))
    elif action == "ATOMIC_ID_SWAP":
        other = _require_public_id(record, "other_public_id", errors)
        if other == record.get("public_id"):
            errors.append(_issue("invalid_swap_ids", "/other_public_id", "swap public IDs must differ"))
        if record.get("atomic_transaction_confirmed") is not True:
            errors.append(_issue("atomic_confirmation_missing", "/atomic_transaction_confirmed", "human must explicitly confirm both sides of a swap"))
    elif action == "RECOVER_IDENTITY" and record.get("recovery_confirmed") is not True:
        errors.append(_issue("recovery_confirmation_missing", "/recovery_confirmed", "recovery must be explicitly confirmed"))
    elif action == "ADD_NEW_IDENTITY" and record.get("new_identity_confirmed") is not True:
        errors.append(_issue("add_confirmation_missing", "/new_identity_confirmed", "new identity must be explicitly confirmed"))

    if record.get("interaction_source") != "real_human":
        errors.append(_issue("real_human_source_missing", "/interaction_source", "interaction_source must be exactly real_human"))
    if record.get("human_confirmed") is not True:
        errors.append(_issue("human_confirmation_missing", "/human_confirmed", "human_confirmed must be exactly true"))
    for key in ("annotator_id_hash", "session_id"):
        value = record.get(key)
        if not _nonempty_string(value) or str(value).strip().lower() in PLACEHOLDER_VALUES:
            errors.append(_issue("missing_provenance", f"/{key}", "non-placeholder annotator/session provenance is required"))
    start = _parse_timestamp(record.get("event_start_timestamp"), "/event_start_timestamp", errors)
    end = _parse_timestamp(record.get("event_end_timestamp"), "/event_end_timestamp", errors)
    if start is not None and end is not None and end < start:
        errors.append(_issue("timestamp_order_invalid", "/event_end_timestamp", "event end precedes event start"))
    if not _is_hash(record.get("annotator_id_hash")):
        errors.append(_issue("annotator_digest_invalid", "/annotator_id_hash", "annotator identifier must be a SHA-256 pseudonym"))
    if not _is_hash(record.get("frame_image_sha256")):
        errors.append(_issue("frame_digest_missing", "/frame_image_sha256", "source frame digest is required"))
    if not _nonempty_string(record.get("candidate_tape_ref")):
        errors.append(_issue("candidate_tape_ref_missing", "/candidate_tape_ref", "candidate_tape_ref is required"))
    if record.get("runtime_future_gt_used") is not False:
        errors.append(_issue("runtime_future_gt_used", "/runtime_future_gt_used", "runtime future GT must be exactly false"))

    _validate_input(record, errors)
    expected_future_frames = _validate_ranges(record, frame, errors)

    spatial = record.get("spatial_correction")
    if not isinstance(spatial, dict):
        errors.append(_issue("spatial_correction_missing", "/spatial_correction", "spatial correction audit is required"))
    else:
        if spatial.get("status") != "PASS":
            errors.append(_issue("spatial_correction_not_pass", "/spatial_correction/status", "event is not ready until official spatial correction passes"))
        if spatial.get("current_frame_output_frozen_before") is not True:
            errors.append(_issue("current_output_order_invalid", "/spatial_correction/current_frame_output_frozen_before", "current output must be frozen before correction"))
        if spatial.get("correction_before_memory_write") is not True:
            errors.append(_issue("memory_order_invalid", "/spatial_correction/correction_before_memory_write", "correction must precede memory write"))
        if spatial.get("backend_prompt_route") not in {"native_box", "box_fallback_from_click", "box_fallback_from_mask"}:
            errors.append(_issue("unsupported_prompt_route", "/spatial_correction/backend_prompt_route", "route must state the real official backend limitation/fallback"))

    mapping = record.get("mapping_audit")
    if not isinstance(mapping, dict):
        errors.append(_issue("mapping_audit_missing", "/mapping_audit", "public/native/local/global mapping audit is required"))
    else:
        for key in ("native_id", "local_id", "sequence_global_id", "public_id"):
            if key not in mapping:
                errors.append(_issue("mapping_audit_missing", f"/mapping_audit/{key}", "mapping field is required"))
        if mapping.get("public_id") != record.get("public_id"):
            errors.append(_issue("mapping_public_id_mismatch", "/mapping_audit/public_id", "mapping must refer to the directly supplied public ID"))
        if mapping.get("stable") is not True or mapping.get("duplicate_public_id") is not False or mapping.get("status") != "PASS":
            errors.append(_issue("mapping_audit_failed", "/mapping_audit", "mapping must be stable, unique and PASS"))

    memory = record.get("memory_audit")
    if not isinstance(memory, dict):
        errors.append(_issue("memory_audit_missing", "/memory_audit", "causal memory audit is required"))
    else:
        if memory.get("event_frame_read") is not False:
            errors.append(_issue("event_frame_memory_leak", "/memory_audit/event_frame_read", "event frame must not read newly written memory"))
        if memory.get("current_frame_write_hidden") is not True:
            errors.append(_issue("event_frame_memory_leak", "/memory_audit/current_frame_write_hidden", "current-frame write must be hidden from current-frame association"))
        if memory.get("first_visible_frame") != frame + 1:
            errors.append(_issue("future_boundary_invalid", "/memory_audit/first_visible_frame", "memory must first be visible at event_frame+1"))
        if memory.get("write_after_spatial_correction") is not True:
            errors.append(_issue("memory_order_invalid", "/memory_audit/write_after_spatial_correction", "memory write must follow spatial correction"))

    embedding = record.get("human_embedding")
    if not isinstance(embedding, dict):
        errors.append(_issue("human_embedding_missing", "/human_embedding", "human embedding provenance is required"))
    else:
        for key in ("source_kind", "derived_from", "feature_dim", "finite", "norm", "sha256"):
            if key not in embedding:
                errors.append(_issue("human_embedding_field_missing", f"/human_embedding/{key}", "embedding provenance field is required"))
        if embedding.get("derived_from") not in ALLOWED_INPUT_KINDS:
            errors.append(_issue("human_embedding_source_invalid", "/human_embedding/derived_from", "embedding must derive from the raw human input kind"))
        if embedding.get("feature_dim") != 512 or embedding.get("finite") is not True or not _is_finite_number(embedding.get("norm")) or float(embedding.get("norm")) <= 0 or not _is_hash(embedding.get("sha256")):
            errors.append(_issue("human_embedding_invalid", "/human_embedding", "human embedding must be finite 512-D with positive norm and digest"))
        if embedding.get("source_kind", "").lower().find("machine") >= 0 or embedding.get("source_kind", "").lower().find("candidate") >= 0:
            errors.append(_issue("machine_embedding_substitution", "/human_embedding/source_kind", "machine candidate embedding cannot substitute for human ROI evidence"))

    frames_checked = rows_checked = 0
    if candidate_tape is None:
        errors.append(_issue("candidate_tape_unavailable", "/candidate_tape_ref", "adapter could not load candidate_tape_ref"))
    else:
        frames_checked, rows_checked = _validate_candidate_tape(record, candidate_tape, expected_future_frames, errors)

    valid = not errors
    normalized = dict(record) if valid else None
    return ValidationResult(
        valid=valid,
        event_id=event_id_text,
        errors=errors,
        warnings=warnings,
        candidate_frames_checked=frames_checked,
        candidate_rows_checked=rows_checked,
        normalized_record=normalized,
    )


def load_candidate_tape(path: Path) -> dict[str, Any]:
    """Load only the N40 candidate-tape container; never consults dataset GT."""

    if path.suffix.lower() == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("candidate tape JSON root must be an object")
        return payload
    rows: list[dict[str, Any]] = []
    manifest: Optional[list[dict[str, Any]]] = None
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        item = json.loads(line)
        if not isinstance(item, dict):
            raise ValueError(f"candidate tape line {line_number} is not an object")
        if item.get("record_type") == "frame_manifest":
            if manifest is not None:
                raise ValueError("candidate tape contains duplicate frame_manifest records")
            manifest = item.get("frame_manifest")
        else:
            rows.append(item)
    if manifest is None:
        raise ValueError("JSONL candidate tape needs one frame_manifest record")
    return {"schema": CANDIDATE_TAPE_SCHEMA, "frame_manifest": manifest, "rows": rows}


class RealHumanEventAdapter:
    """Import JSONL events and retain every rejected record as an attempt artifact."""

    def __init__(self, *, candidate_root: Path | str | None = None, failure_dir: Path | str = "outputs/n40/attempts"):
        self.candidate_root = Path(candidate_root).resolve() if candidate_root is not None else None
        self.failure_dir = Path(failure_dir)

    def _resolve_candidate_ref(self, reference: str) -> Path:
        path = Path(reference)
        if not path.is_absolute():
            base = self.candidate_root or Path.cwd().resolve()
            path = base / path
        path = path.resolve()
        if self.candidate_root is not None:
            try:
                path.relative_to(self.candidate_root)
            except ValueError as exc:
                raise ValueError("candidate_tape_ref escapes candidate_root") from exc
        return path

    def _failure_path(self, index: int, event_id: Any) -> Path:
        self.failure_dir.mkdir(parents=True, exist_ok=True)
        base = self.failure_dir / f"attempt_{index:06d}_{_safe_name(event_id)}.json"
        if not base.exists():
            return base
        suffix = 1
        while True:
            candidate = self.failure_dir / f"attempt_{index:06d}_{_safe_name(event_id)}_{suffix}.json"
            if not candidate.exists():
                return candidate
            suffix += 1

    def validate_record(self, record: Any) -> ValidationResult:
        tape = None
        if isinstance(record, dict) and _nonempty_string(record.get("candidate_tape_ref")):
            try:
                tape = load_candidate_tape(self._resolve_candidate_ref(record["candidate_tape_ref"]))
            except Exception:
                tape = None
        return validate_real_human_event(record, candidate_tape=tape)

    def validate_jsonl(self, input_path: Path | str) -> dict[str, Any]:
        input_path = Path(input_path)
        accepted: list[dict[str, Any]] = []
        failures: list[dict[str, Any]] = []
        seen_event_ids: set[str] = set()
        total = 0
        with input_path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                total += 1
                try:
                    record = json.loads(line)
                except Exception as exc:
                    result = ValidationResult(False, errors=[_issue("invalid_json", f"/line/{line_number}", f"{type(exc).__name__}: {exc}")])
                    record = {"raw_line": line.rstrip("\n")}
                else:
                    result = self.validate_record(record)
                    event_id = result.event_id
                    if result.valid and event_id is not None and event_id in seen_event_ids:
                        result.valid = False
                        result.normalized_record = None
                        result.errors.append(_issue("duplicate_event_id", "/event_id", f"event_id {event_id} occurs more than once"))
                    if result.valid and event_id is not None:
                        seen_event_ids.add(event_id)
                if result.valid:
                    accepted.append(result.normalized_record)
                    continue
                failure = {
                    "schema": "N40_REAL_HUMAN_EVENT_ATTEMPT_V1",
                    "input_path": str(input_path.resolve()),
                    "line_number": line_number,
                    "event_id": result.event_id,
                    "status": "REJECTED",
                    "validation": result.as_dict(),
                    "record": record,
                }
                failure_path = self._failure_path(line_number, result.event_id)
                _atomic_write_json(failure_path, failure)
                failures.append({"line_number": line_number, "event_id": result.event_id, "path": str(failure_path), "errors": result.errors})
        return {
            "schema": "N40_REAL_HUMAN_EVENT_IMPORT_REPORT_V1",
            "status": "PASS" if not failures else "FAIL_INPUT_SCHEMA",
            "input_path": str(input_path.resolve()),
            "total_records": total,
            "accepted_records": len(accepted),
            "rejected_records": len(failures),
            "duplicate_event_ids": len([x for x in failures if any(e["code"] == "duplicate_event_id" for e in x["errors"])]),
            "accepted": accepted,
            "failures": failures,
            "runtime_future_gt_used": False,
            "downstream_authorized": False,
        }


__all__ = [
    "ALLOWED_ACTIONS",
    "ALLOWED_INPUT_KINDS",
    "CANDIDATE_TAPE_SCHEMA",
    "PROTOCOL_ID",
    "RealHumanEventAdapter",
    "ValidationResult",
    "load_candidate_tape",
    "validate_real_human_event",
]
