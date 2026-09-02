"""N72 external real-human event recorder and strict validator.

The module accepts only records emitted by an external UI/annotator.  It does
not import a dataset reader, simulator, GT evaluator, or tracker state.  A
lower-level ``HumanInteraction(source='human')`` is intentionally irrelevant:
real provenance is established only by this contract and its raw payload
digests.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import tempfile
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Mapping

from sam3_intermot.provenance.mapping import (
    MAPPING_STATUSES,
    canonical_candidate_uid,
)


PROTOCOL_ID = "N72_REAL_HUMAN_EVENT_TAPE_V1"
CANDIDATE_TAPE_SCHEMA = "N72_CANDIDATE_TAPE_V1"
ACTION_ALIASES = {
    "REASSIGN": "AUTHORITATIVE_REASSIGN",
    "SWAP": "ATOMIC_ID_SWAP",
    "AUTHORITATIVE_REASSIGN": "AUTHORITATIVE_REASSIGN",
    "ATOMIC_ID_SWAP": "ATOMIC_ID_SWAP",
    "DELETE": "DELETE",
    "ADD_NEW_IDENTITY": "ADD_NEW_IDENTITY",
    "RECOVER_IDENTITY": "RECOVER_IDENTITY",
}
INPUT_KINDS = frozenset({"BOX", "CLICK", "CONFIRMED_MASK"})
MASK_FORMATS = frozenset({"lossless_RLE", "lossless_PNG", "lossless_NPZ"})
REQUIRED_HORIZONS = (20, 50, 100)
HASH_RE = re.compile(r"^[0-9a-fA-F]{64}$")
FORBIDDEN_KEYS = frozenset(
    {
        "gt",
        "gt_box",
        "gt_id",
        "dataset_gt_id",
        "dataset_identity",
        "ground_truth",
        "future_gt",
        "future_gt_used",
        "future_identity",
        "posthoc_gt",
        "reward",
        "selected_candidate",
        "candidate_outcome",
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
PLACEHOLDER_VALUES = frozenset({"", "none", "null", "unknown", "placeholder"})


def _issue(code: str, path: str, message: str) -> dict[str, str]:
    return {"code": code, "path": path, "message": message}


def _hash(value: Any) -> bool:
    return isinstance(value, str) and HASH_RE.fullmatch(value) is not None


def _text(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _finite_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def _int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return int(value)
    if isinstance(value, str) and value.strip():
        try:
            return int(value)
        except ValueError:
            return None
    return None


def _walk_forbidden(value: Any, path: str = "") -> list[dict[str, str]]:
    issues: list[dict[str, str]] = []
    if isinstance(value, Mapping):
        for key, child in value.items():
            key_text = str(key)
            if key_text.lower() in FORBIDDEN_KEYS:
                issues.append(_issue("GT_DERIVED_FIELD_FORBIDDEN", f"{path}/{key_text}", "GT/future-label fields are not allowed in runtime event input"))
            if key_text.lower() in {"source", "interaction_source", "origin", "mask_origin", "provenance", "source_kind"} and isinstance(child, str):
                lowered = child.strip().lower()
                if lowered in PLACEHOLDER_VALUES or any(token in lowered for token in FORBIDDEN_SOURCE_TOKENS):
                    issues.append(_issue("NON_REAL_SOURCE_FORBIDDEN", f"{path}/{key_text}", f"non-real source value is forbidden: {child}"))
            issues.extend(_walk_forbidden(child, f"{path}/{key_text}"))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            issues.extend(_walk_forbidden(child, f"{path}/{index}"))
    return issues


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
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


def _resolve(ref: Any, root: Path | None) -> Path | None:
    if not _text(ref):
        return None
    path = Path(str(ref))
    if not path.is_absolute():
        path = (root or Path.cwd()) / path
    return path.resolve()


def load_candidate_tape(path: Path) -> dict[str, Any]:
    """Load only a declared N72 candidate container, never GT."""

    if path.suffix.lower() == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("candidate tape root must be an object")
        return payload
    rows: list[dict[str, Any]] = []
    frame_manifest = None
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            item = json.loads(line)
            if not isinstance(item, dict):
                raise ValueError(f"candidate tape line {line_number} is not an object")
            if item.get("record_type") == "frame_manifest":
                if frame_manifest is not None:
                    raise ValueError("duplicate frame_manifest")
                frame_manifest = item.get("frame_manifest")
            else:
                rows.append(item)
    if frame_manifest is None:
        raise ValueError("JSONL candidate tape needs one frame_manifest record")
    return {"schema": CANDIDATE_TAPE_SCHEMA, "frame_manifest": frame_manifest, "rows": rows}


def _validate_human_input(record: Mapping[str, Any], errors: list[dict[str, str]]) -> None:
    human_input = record.get("human_input")
    if not isinstance(human_input, Mapping):
        errors.append(_issue("HUMAN_INPUT_MISSING", "/human_input", "external UI human_input object is required"))
        return
    kind = human_input.get("kind")
    if kind not in INPUT_KINDS:
        errors.append(_issue("HUMAN_INPUT_KIND_INVALID", "/human_input/kind", "kind must be BOX, CLICK or CONFIRMED_MASK"))
        return
    if human_input.get("origin") != "external_human_ui":
        errors.append(_issue("HUMAN_ORIGIN_NOT_PROVEN", "/human_input/origin", "raw input must come directly from the external human UI"))
    if human_input.get("human_confirmed") is not True:
        errors.append(_issue("HUMAN_CONFIRMATION_MISSING", "/human_input/human_confirmed", "raw input needs explicit human confirmation"))
    if not _text(human_input.get("raw_payload_ref")) or not _hash(human_input.get("raw_payload_sha256")):
        errors.append(_issue("RAW_INPUT_PROVENANCE_MISSING", "/human_input", "raw payload reference and SHA-256 are required"))
    if kind == "BOX":
        box = human_input.get("box_xyxy")
        if not isinstance(box, list) or len(box) != 4 or not all(_finite_number(value) for value in box) or float(box[2]) <= float(box[0]) or float(box[3]) <= float(box[1]):
            errors.append(_issue("HUMAN_BOX_INVALID", "/human_input/box_xyxy", "BOX needs four finite coordinates with positive area"))
    elif kind == "CLICK":
        points = human_input.get("click_points")
        if not isinstance(points, list) or not points:
            errors.append(_issue("HUMAN_CLICK_INVALID", "/human_input/click_points", "at least one click is required"))
        else:
            positives = 0
            for index, point in enumerate(points):
                if not isinstance(point, Mapping) or not _finite_number(point.get("x")) or not _finite_number(point.get("y")):
                    errors.append(_issue("HUMAN_CLICK_INVALID", f"/human_input/click_points/{index}", "click needs finite x and y"))
                elif point.get("label") not in {"positive", "negative"}:
                    errors.append(_issue("HUMAN_CLICK_LABEL_INVALID", f"/human_input/click_points/{index}/label", "click label must be positive or negative"))
                else:
                    positives += int(point.get("label") == "positive")
            if positives == 0:
                errors.append(_issue("POSITIVE_CLICK_MISSING", "/human_input/click_points", "at least one positive click is required"))
    else:
        mask = human_input.get("confirmed_mask")
        if not isinstance(mask, Mapping):
            errors.append(_issue("CONFIRMED_MASK_MISSING", "/human_input/confirmed_mask", "confirmed mask object is required"))
        else:
            if mask.get("format") not in MASK_FORMATS:
                errors.append(_issue("MASK_FORMAT_INVALID", "/human_input/confirmed_mask/format", "confirmed mask must use a lossless format"))
            if not _text(mask.get("payload_ref")) or not _hash(mask.get("sha256")):
                errors.append(_issue("MASK_PAYLOAD_PROVENANCE_MISSING", "/human_input/confirmed_mask", "lossless mask payload reference and digest are required"))
            shape = mask.get("frame_shape")
            if not isinstance(shape, list) or len(shape) != 2 or not all(isinstance(value, int) and not isinstance(value, bool) and value > 0 for value in shape):
                errors.append(_issue("MASK_SHAPE_INVALID", "/human_input/confirmed_mask/frame_shape", "frame_shape must be positive [height,width]"))
            if mask.get("mask_origin") != "human_ui_confirmed" or mask.get("machine_candidate_mask") is not False:
                errors.append(_issue("MACHINE_MASK_NOT_ALLOWED", "/human_input/confirmed_mask", "machine candidate mask cannot be relabeled as human confirmed mask"))


def _validate_candidate_tape(record: Mapping[str, Any], tape: Any, errors: list[dict[str, str]]) -> dict[str, Any]:
    if not isinstance(tape, Mapping):
        errors.append(_issue("CANDIDATE_TAPE_UNAVAILABLE", "/candidate_tape_ref", "candidate tape could not be decoded"))
        return {"frame_count": 0, "candidate_row_count": 0, "mapping_status_counts": {}}
    if tape.get("schema") != CANDIDATE_TAPE_SCHEMA:
        errors.append(_issue("CANDIDATE_TAPE_SCHEMA_INVALID", "/candidate_tape/schema", f"schema must be {CANDIDATE_TAPE_SCHEMA}"))
    if tape.get("sequence") != record.get("sequence"):
        errors.append(_issue("CANDIDATE_SEQUENCE_MISMATCH", "/candidate_tape/sequence", "candidate tape sequence differs from event"))
    manifest = tape.get("frame_manifest")
    rows = tape.get("rows")
    if not isinstance(manifest, list) or not isinstance(rows, list):
        errors.append(_issue("CANDIDATE_CONTAINER_INVALID", "/candidate_tape", "frame_manifest and rows lists are required"))
        return {"frame_count": 0, "candidate_row_count": len(rows) if isinstance(rows, list) else 0, "mapping_status_counts": {}}
    event_frame = int(record["event_frame"])
    required_frames = {event_frame}
    for horizon in REQUIRED_HORIZONS:
        required_frames.update(range(event_frame + 1, event_frame + horizon + 1))
    manifest_counts: dict[int, int] = {}
    for index, item in enumerate(manifest):
        if not isinstance(item, Mapping) or _int(item.get("frame_id")) is None or _int(item.get("candidate_count")) is None or _int(item.get("candidate_count")) < 0:
            errors.append(_issue("FRAME_MANIFEST_INVALID", f"/candidate_tape/frame_manifest/{index}", "frame_id and non-negative candidate_count are required"))
            continue
        frame = int(item["frame_id"])
        if frame in manifest_counts:
            errors.append(_issue("DUPLICATE_CANDIDATE_FRAME", f"/candidate_tape/frame_manifest/{index}", f"frame {frame} appears more than once"))
        manifest_counts[frame] = int(item["candidate_count"])
    missing_frames = sorted(required_frames - set(manifest_counts))
    for frame in missing_frames:
        errors.append(_issue("MISSING_CANDIDATE_FRAME", "/candidate_tape/frame_manifest", f"required frame {frame} is missing"))

    rows_by_frame: Counter[int] = Counter()
    candidate_uids: Counter[str] = Counter()
    raw_frame_keys: Counter[tuple[int, int]] = Counter()
    mapping_status_counts: Counter[str] = Counter()
    event_frame_hash = record.get("frame_image_sha256")
    for index, row in enumerate(rows):
        path = f"/candidate_tape/rows/{index}"
        if not isinstance(row, Mapping):
            errors.append(_issue("CANDIDATE_ROW_INVALID", path, "candidate row must be an object"))
            continue
        frame = _int(row.get("frame_id"))
        if frame is None:
            errors.append(_issue("CANDIDATE_FRAME_INVALID", f"{path}/frame_id", "frame_id must be an integer"))
            continue
        rows_by_frame[frame] += 1
        if frame not in manifest_counts:
            errors.append(_issue("CANDIDATE_FRAME_NOT_MANIFESTED", f"{path}/frame_id", f"frame {frame} is not in frame_manifest"))
        if row.get("runtime_future_gt_used") is not False:
            errors.append(_issue("RUNTIME_FUTURE_GT_USED", f"{path}/runtime_future_gt_used", "must be exactly false"))
        if not _hash(row.get("frame_hash_sha256")) or (frame == event_frame and row.get("frame_hash_sha256") != event_frame_hash):
            errors.append(_issue("FRAME_HASH_INVALID", f"{path}/frame_hash_sha256", "frame hash is missing or event-frame hash disagrees"))
        box = row.get("box_xyxy")
        if not isinstance(box, list) or len(box) != 4 or not all(_finite_number(value) for value in box):
            errors.append(_issue("CANDIDATE_BOX_INVALID", f"{path}/box_xyxy", "candidate box needs four finite values"))
        if not _hash(row.get("mask_sha256")):
            errors.append(_issue("CANDIDATE_MASK_DIGEST_MISSING", f"{path}/mask_sha256", "candidate mask digest is required"))
        feature = row.get("feature")
        if not isinstance(feature, Mapping) or feature.get("dim") != 512 or feature.get("finite") is not True or not _finite_number(feature.get("norm")) or float(feature.get("norm")) <= 0 or not _hash(feature.get("sha256")):
            errors.append(_issue("CANDIDATE_FEATURE_INVALID", f"{path}/feature", "candidate feature provenance must be finite 512-D with positive norm and digest"))

        raw = _int(row.get("raw_native_id"))
        adapter = _int(row.get("adapter_external_id"))
        local = row.get("segment_local_id")
        global_id = row.get("sequence_global_id")
        if raw is None or adapter is None or not _text(local) or not _text(global_id):
            errors.append(_issue("CANDIDATE_MAPPING_AXIS_MISSING", f"{path}/mapping", "raw native, adapter external, segment local and sequence global IDs are required"))
        else:
            raw_frame_keys[(frame, raw)] += 1
            expected_uid = canonical_candidate_uid(
                sequence=str(record["sequence"]),
                frame=frame,
                raw_native_id=raw,
                adapter_external_id=adapter,
                segment_local_id=str(local),
                sequence_global_id=str(global_id),
            )
            uid = row.get("candidate_uid")
            if uid != expected_uid:
                errors.append(_issue("CANDIDATE_UID_MISMATCH", f"{path}/candidate_uid", "candidate_uid does not match canonical identity axes"))
            candidate_uids[str(uid)] += 1
        mapping = row.get("mapping")
        if not isinstance(mapping, Mapping):
            errors.append(_issue("CANDIDATE_MAPPING_MISSING", f"{path}/mapping", "mapping object is required"))
        else:
            status = mapping.get("status")
            mapping_status_counts[str(status)] += 1
            if status not in MAPPING_STATUSES:
                errors.append(_issue("CANDIDATE_MAPPING_STATUS_INVALID", f"{path}/mapping/status", "unknown exact mapping status"))
            if status == "EXACT":
                if _int(mapping.get("public_id")) is None or not _text(mapping.get("source")) or mapping.get("source") not in {"identity_registry_binding", "explicit_runtime_assignment", "direct_user_public_id", "frozen_provenance_mapping"}:
                    errors.append(_issue("CANDIDATE_MAPPING_SOURCE_INVALID", f"{path}/mapping", "EXACT requires an allowed authoritative source and public_id"))
            elif mapping.get("public_id") is not None:
                errors.append(_issue("NONEXACT_PUBLIC_ID_PRESENT", f"{path}/mapping/public_id", "non-EXACT mapping must preserve public_id as null"))

    for frame, expected in manifest_counts.items():
        if rows_by_frame.get(frame, 0) != expected:
            errors.append(_issue("CANDIDATE_COUNT_MISMATCH", f"/candidate_tape/frame_manifest/{frame}", f"manifest={expected}, rows={rows_by_frame.get(frame, 0)}"))
    for uid, count in candidate_uids.items():
        if count > 1:
            errors.append(_issue("DUPLICATE_CANDIDATE_UID", "/candidate_tape/rows", f"candidate_uid {uid} occurs {count} times"))
    for key, count in raw_frame_keys.items():
        if count > 1:
            errors.append(_issue("DUPLICATE_RAW_ID_IN_FRAME", "/candidate_tape/rows", f"raw ID key {key} occurs {count} times"))
    return {
        "frame_count": len(manifest_counts),
        "required_frame_count": len(required_frames),
        "candidate_row_count": len(rows),
        "candidate_uid_count": len(candidate_uids),
        "mapping_status_counts": dict(sorted(mapping_status_counts.items())),
        "missing_required_frames": missing_frames,
        "runtime_future_gt_used_count": sum(int(row.get("runtime_future_gt_used") is not False) for row in rows if isinstance(row, Mapping)),
    }


def validate_real_human_event(record: Any, *, candidate_tape: Any = None, raw_root: Path | None = None) -> dict[str, Any]:
    """Validate one external event and return a JSON-safe audit result."""

    errors: list[dict[str, str]] = []
    if not isinstance(record, Mapping):
        return {"valid": False, "event_id": None, "errors": [_issue("EVENT_NOT_OBJECT", "/", "event must be a JSON object")], "warnings": []}
    errors.extend(_walk_forbidden(record))
    event_id = record.get("event_id")
    if not _text(event_id):
        errors.append(_issue("EVENT_ID_MISSING", "/event_id", "stable event_id is required"))
    if record.get("interaction_source") != "real_human":
        errors.append(_issue("REAL_HUMAN_SOURCE_MISSING", "/interaction_source", "interaction_source must be exactly real_human"))
    if record.get("test_fixture") is True:
        errors.append(_issue("TEST_FIXTURE_NOT_REAL_HUMAN", "/test_fixture", "test fixtures cannot enter the real-human tape"))
    if record.get("human_confirmed") is not True:
        errors.append(_issue("HUMAN_CONFIRMATION_MISSING", "/human_confirmed", "human_confirmed must be exactly true"))
    if record.get("runtime_future_gt_used") is not False:
        errors.append(_issue("RUNTIME_FUTURE_GT_USED", "/runtime_future_gt_used", "runtime future GT must be exactly false"))
    for key in ("sequence", "split", "ui_version", "session_id", "annotator_id_hash", "event_start_timestamp", "event_end_timestamp", "frame_image_ref", "frame_image_sha256", "candidate_tape_ref", "public_id_source", "human_embedding", "spatial_correction", "mapping_audit", "memory_audit"):
        if key not in record:
            errors.append(_issue("REQUIRED_FIELD_MISSING", f"/{key}", "required field is missing"))
    if not _text(record.get("sequence")) or record.get("split") not in {"train", "train_fold"}:
        errors.append(_issue("SEQUENCE_OR_SPLIT_INVALID", "/sequence", "sequence and train/train_fold split are required"))
    frame = _int(record.get("event_frame"))
    if frame is None or frame < 0:
        errors.append(_issue("EVENT_FRAME_INVALID", "/event_frame", "event_frame must be a non-negative integer"))
        frame = 0
    public_id = _int(record.get("public_id"))
    if public_id is None or public_id < 0:
        errors.append(_issue("PUBLIC_ID_INVALID", "/public_id", "public_id must be a non-negative integer supplied by the human"))
    if record.get("public_id_source") != "human_direct":
        errors.append(_issue("PUBLIC_ID_NOT_DIRECT", "/public_id_source", "public_id must be directly supplied by the human"))
    action_raw = record.get("action_type")
    action = ACTION_ALIASES.get(str(action_raw))
    if action is None:
        errors.append(_issue("ACTION_INVALID", "/action_type", "unsupported action type"))
    elif action == "AUTHORITATIVE_REASSIGN" and _int(record.get("current_public_id")) == public_id:
        errors.append(_issue("REASSIGN_IDS_EQUAL", "/current_public_id", "source and destination public IDs must differ"))
    elif action == "ATOMIC_ID_SWAP":
        if _int(record.get("other_public_id")) == public_id:
            errors.append(_issue("SWAP_IDS_EQUAL", "/other_public_id", "swap IDs must differ"))
        if record.get("atomic_transaction_confirmed") is not True:
            errors.append(_issue("ATOMIC_CONFIRMATION_MISSING", "/atomic_transaction_confirmed", "swap needs explicit two-sided confirmation"))
    elif action == "DELETE" and record.get("delete_confirmed") is not True:
        errors.append(_issue("DELETE_CONFIRMATION_MISSING", "/delete_confirmed", "delete needs explicit confirmation"))
    elif action == "ADD_NEW_IDENTITY" and record.get("new_identity_confirmed") is not True:
        errors.append(_issue("ADD_CONFIRMATION_MISSING", "/new_identity_confirmed", "new identity needs explicit confirmation"))
    elif action == "RECOVER_IDENTITY" and record.get("recovery_confirmed") is not True:
        errors.append(_issue("RECOVERY_CONFIRMATION_MISSING", "/recovery_confirmed", "recovery needs explicit confirmation"))

    if not _hash(record.get("annotator_id_hash")) or not _text(record.get("session_id")) or not _text(record.get("ui_version")):
        errors.append(_issue("PROVENANCE_INVALID", "/provenance", "annotator hash, session and UI version are required"))
    for key in ("event_start_timestamp", "event_end_timestamp"):
        value = record.get(key)
        if not isinstance(value, str) or not value.strip():
            errors.append(_issue("TIMESTAMP_INVALID", f"/{key}", "ISO-8601 timestamp is required"))
        else:
            try:
                datetime.fromisoformat(value.replace("Z", "+00:00"))
            except ValueError:
                errors.append(_issue("TIMESTAMP_INVALID", f"/{key}", "timestamp is not ISO-8601"))
    if not _hash(record.get("frame_image_sha256")):
        errors.append(_issue("FRAME_HASH_INVALID", "/frame_image_sha256", "source frame SHA-256 is required"))
    if raw_root is not None:
        for key in ("frame_image_ref",):
            path = _resolve(record.get(key), raw_root)
            if path is None or not path.is_file() or _sha256(path) != record.get("frame_image_sha256"):
                errors.append(_issue("RAW_FRAME_UNAVAILABLE", f"/{key}", "raw frame reference is unavailable or digest mismatches"))

    _validate_human_input(record, errors)
    human_input = record.get("human_input")
    if isinstance(human_input, Mapping) and raw_root is not None:
        raw_path = _resolve(human_input.get("raw_payload_ref"), raw_root)
        if raw_path is None or not raw_path.is_file() or _sha256(raw_path) != human_input.get("raw_payload_sha256"):
            errors.append(_issue("RAW_INPUT_UNAVAILABLE", "/human_input/raw_payload_ref", "raw UI payload is unavailable or digest mismatches"))
        if human_input.get("kind") == "CONFIRMED_MASK":
            mask = human_input.get("confirmed_mask", {})
            mask_path = _resolve(mask.get("payload_ref"), raw_root)
            if mask_path is None or not mask_path.is_file() or _sha256(mask_path) != mask.get("sha256"):
                errors.append(_issue("RAW_MASK_UNAVAILABLE", "/human_input/confirmed_mask/payload_ref", "raw confirmed mask is unavailable or digest mismatches"))

    prefix = record.get("prefix_range")
    if not isinstance(prefix, list) or len(prefix) != 2 or any(_int(value) is None for value in prefix) or int(prefix[1]) != frame - 1 or int(prefix[0]) < 0 or int(prefix[0]) > int(prefix[1]):
        errors.append(_issue("PREFIX_RANGE_INVALID", "/prefix_range", "prefix must end at event_frame-1"))
    future_ranges = record.get("future_ranges")
    expected_future = {frame + offset for offset in range(1, 101)}
    if not isinstance(future_ranges, Mapping):
        errors.append(_issue("FUTURE_RANGES_INVALID", "/future_ranges", "H20/H50/H100 ranges are required"))
    else:
        expected_future = set()
        for horizon in REQUIRED_HORIZONS:
            item = future_ranges.get(f"H{horizon}")
            if not isinstance(item, list) or len(item) != 2 or item != [frame + 1, frame + horizon]:
                errors.append(_issue("FUTURE_BOUNDARY_INVALID", f"/future_ranges/H{horizon}", "range must be [event_frame+1,event_frame+horizon]"))
            else:
                expected_future.update(range(frame + 1, frame + horizon + 1))

    spatial = record.get("spatial_correction")
    if not isinstance(spatial, Mapping):
        errors.append(_issue("SPATIAL_CORRECTION_MISSING", "/spatial_correction", "official spatial correction audit is required"))
    else:
        if spatial.get("status") != "PASS" or spatial.get("current_frame_output_frozen_before") is not True or spatial.get("correction_before_memory_write") is not True:
            errors.append(_issue("SPATIAL_ORDER_INVALID", "/spatial_correction", "current output must be frozen before correction and correction before memory write"))
        if spatial.get("backend_prompt_route") not in {"native_box", "box_fallback_from_click", "box_fallback_from_mask"}:
            errors.append(_issue("PROMPT_ROUTE_INVALID", "/spatial_correction/backend_prompt_route", "route must state the official backend limitation/fallback"))
    mapping = record.get("mapping_audit")
    if not isinstance(mapping, Mapping):
        errors.append(_issue("EVENT_MAPPING_MISSING", "/mapping_audit", "exact event mapping audit is required"))
    else:
        if mapping.get("status") != "EXACT" or _int(mapping.get("public_id")) != public_id or mapping.get("source") not in {"identity_registry_binding", "explicit_runtime_assignment", "direct_user_public_id", "frozen_provenance_mapping"} or mapping.get("stable") is not True:
            errors.append(_issue("EVENT_MAPPING_NOT_EXACT", "/mapping_audit", "event mapping needs one authoritative exact source"))
        if any(mapping.get(key) is None for key in ("raw_native_id", "adapter_external_id", "segment_local_id", "sequence_global_id")):
            errors.append(_issue("EVENT_MAPPING_AXIS_MISSING", "/mapping_audit", "all raw/adapter/local/global axes are required"))
    memory = record.get("memory_audit")
    if not isinstance(memory, Mapping) or memory.get("event_frame_read") is not False or memory.get("current_frame_write_hidden") is not True or memory.get("first_visible_frame") != frame + 1 or memory.get("write_after_spatial_correction") is not True:
        errors.append(_issue("CAUSAL_MEMORY_BOUNDARY_INVALID", "/memory_audit", "event-frame read must be false and first visible frame event+1 after correction"))
    embedding = record.get("human_embedding")
    if not isinstance(embedding, Mapping) or embedding.get("derived_from") not in INPUT_KINDS or embedding.get("source_kind") in {"machine_candidate", "candidate_embedding"} or embedding.get("feature_dim") != 512 or embedding.get("finite") is not True or not _finite_number(embedding.get("norm")) or float(embedding.get("norm")) <= 0 or not _hash(embedding.get("sha256")):
        errors.append(_issue("HUMAN_EMBEDDING_INVALID", "/human_embedding", "embedding must be a finite 512-D human-input-derived feature with digest"))

    tape_audit = _validate_candidate_tape(record, candidate_tape, errors)
    return {
        "valid": not errors,
        "event_id": None if event_id is None else str(event_id),
        "canonical_action_type": action,
        "errors": errors,
        "warnings": [],
        "candidate_tape_audit": tape_audit,
        "runtime_future_gt_used": False,
        "real_human_evidence": not bool(record.get("test_fixture")),
    }


class N72RealHumanEventAdapter:
    """Validate an external JSONL event file and preserve every rejection."""

    def __init__(self, *, candidate_root: Path | str | None = None, raw_root: Path | str | None = None, failure_dir: Path | str = "outputs/N72/human_tape/attempts") -> None:
        self.candidate_root = Path(candidate_root).resolve() if candidate_root is not None else None
        self.raw_root = Path(raw_root).resolve() if raw_root is not None else None
        self.failure_dir = Path(failure_dir)

    def _failure_path(self, index: int, event_id: Any) -> Path:
        safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(event_id or "unknown"))[:80] or "unknown"
        path = self.failure_dir / f"attempt_{index:06d}_{safe}.json"
        suffix = 1
        while path.exists():
            path = self.failure_dir / f"attempt_{index:06d}_{safe}_{suffix}.json"
            suffix += 1
        return path

    def validate_record(self, record: Any) -> dict[str, Any]:
        tape = None
        if isinstance(record, Mapping):
            path = _resolve(record.get("candidate_tape_ref"), self.candidate_root)
            if path is not None and path.is_file():
                try:
                    tape = load_candidate_tape(path)
                except Exception:
                    tape = None
        return validate_real_human_event(record, candidate_tape=tape, raw_root=self.raw_root)

    def validate_jsonl(self, input_path: Path | str) -> dict[str, Any]:
        input_path = Path(input_path).resolve()
        accepted: list[dict[str, Any]] = []
        failures: list[dict[str, Any]] = []
        seen: set[str] = set()
        total = 0
        with input_path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                total += 1
                try:
                    record = json.loads(line)
                    audit = self.validate_record(record)
                except Exception as exc:
                    record = {"raw_line": line.rstrip("\n")}
                    audit = {"valid": False, "event_id": None, "errors": [_issue("VALIDATOR_EXCEPTION", f"/line/{line_number}", f"{type(exc).__name__}: {exc}")], "warnings": []}
                event_id = audit.get("event_id")
                if audit.get("valid") and event_id in seen:
                    audit["valid"] = False
                    audit["errors"].append(_issue("DUPLICATE_EVENT_ID", "/event_id", f"event_id {event_id} occurs more than once"))
                if audit.get("valid") and event_id is not None:
                    seen.add(str(event_id))
                if audit.get("valid"):
                    accepted.append(record)
                else:
                    failure = {
                        "schema": "N72_REAL_HUMAN_EVENT_ATTEMPT_V1",
                        "status": "REJECTED",
                        "input_path": str(input_path),
                        "line_number": line_number,
                        "event_id": event_id,
                        "validation": audit,
                        "record": record,
                    }
                    failure_path = self._failure_path(line_number, event_id)
                    _atomic_json(failure_path, failure)
                    failures.append({"line_number": line_number, "event_id": event_id, "path": str(failure_path), "errors": audit.get("errors", [])})
        return {
            "schema": "N72_REAL_HUMAN_EVENT_IMPORT_REPORT_V1",
            "status": "PASS" if not failures else "FAIL_INPUT_SCHEMA",
            "input_path": str(input_path),
            "total_records": total,
            "accepted_records": len(accepted),
            "rejected_records": len(failures),
            "accepted": accepted,
            "failures": failures,
            "runtime_future_gt_used": False,
            "real_human_tape": bool(accepted),
            "downstream_authorized": False,
        }


class N72RealHumanTapeRecorder:
    """Append externally emitted JSON records without truncating prior input."""

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)

    def append_record(self, record: Mapping[str, Any]) -> None:
        if not isinstance(record, Mapping):
            raise TypeError("record must be a JSON object")
        encoded = (json.dumps(dict(record), ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("ab") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())


__all__ = [
    "ACTION_ALIASES",
    "CANDIDATE_TAPE_SCHEMA",
    "INPUT_KINDS",
    "N72RealHumanEventAdapter",
    "N72RealHumanTapeRecorder",
    "PROTOCOL_ID",
    "load_candidate_tape",
    "validate_real_human_event",
]
