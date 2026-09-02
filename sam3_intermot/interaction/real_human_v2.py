"""Action-specific, server-authenticated real-human event V2 contract."""

from __future__ import annotations

import hashlib
import json
import math
import secrets
from datetime import datetime
from typing import Any, Mapping


SCHEMA_VERSION = "N72R1_REAL_HUMAN_EVENT_V2"
SPATIAL_ACTIONS = frozenset({"AUTHORITATIVE_CORRECT", "ADD_NEW_IDENTITY", "RECOVER_IDENTITY"})
IDENTITY_ACTIONS = frozenset({"AUTHORITATIVE_REASSIGN", "ATOMIC_ID_SWAP", "AUTHORITATIVE_DELETE"})
ACTION_ALIASES = {
    "CORRECT": "AUTHORITATIVE_CORRECT",
    "AUTHORITATIVE_CORRECT": "AUTHORITATIVE_CORRECT",
    "REASSIGN": "AUTHORITATIVE_REASSIGN",
    "AUTHORITATIVE_REASSIGN": "AUTHORITATIVE_REASSIGN",
    "SWAP": "ATOMIC_ID_SWAP",
    "ATOMIC_ID_SWAP": "ATOMIC_ID_SWAP",
    "DELETE": "AUTHORITATIVE_DELETE",
    "AUTHORITATIVE_DELETE": "AUTHORITATIVE_DELETE",
    "ADD": "ADD_NEW_IDENTITY",
    "ADD_NEW_IDENTITY": "ADD_NEW_IDENTITY",
    "RECOVER": "RECOVER_IDENTITY",
    "RECOVER_IDENTITY": "RECOVER_IDENTITY",
}
INPUT_KINDS = frozenset({"BOX", "CLICK", "CONFIRMED_MASK", "ID_SELECTION", "NONE"})
FORBIDDEN_KEYS = frozenset({
    "gt", "gt_id", "gt_box", "dataset_gt_id", "dataset_identity", "ground_truth", "future_gt",
    "future_identity", "posthoc_gt", "reward", "iou", "future_effect", "selected_candidate",
})
FORBIDDEN_TOKENS = ("simulated_from_gt", "sim_gt", "oracle", "ground_truth", "future_gt")


def _issue(code: str, path: str, message: str) -> dict[str, str]:
    return {"code": code, "path": path, "message": message}


def _text(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


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


def _finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def _hash(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(char in "0123456789abcdefABCDEF" for char in value)


def _walk_forbidden(value: Any, path: str = "") -> list[dict[str, str]]:
    issues: list[dict[str, str]] = []
    if isinstance(value, Mapping):
        for key, child in value.items():
            key_text = str(key)
            if key_text.lower() in FORBIDDEN_KEYS or key_text.lower().startswith("future_gt"):
                issues.append(_issue("GT_DERIVED_FIELD_FORBIDDEN", f"{path}/{key_text}", "runtime human input cannot contain GT or future-label fields"))
            if isinstance(child, str) and any(token in child.lower() for token in FORBIDDEN_TOKENS):
                issues.append(_issue("NON_REAL_SOURCE_FORBIDDEN", f"{path}/{key_text}", "synthetic/oracle provenance cannot enter real-human input"))
            issues.extend(_walk_forbidden(child, f"{path}/{key_text}"))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            issues.extend(_walk_forbidden(child, f"{path}/{index}"))
    return issues


def canonical_action(value: Any) -> str | None:
    return ACTION_ALIASES.get(str(value).strip().upper()) if value is not None else None


def _validate_spatial_input(human_input: Mapping[str, Any], errors: list[dict[str, str]]) -> None:
    kind = human_input.get("kind")
    if kind not in {"BOX", "CLICK", "CONFIRMED_MASK"}:
        errors.append(_issue("SPATIAL_INPUT_KIND_INVALID", "/human_input/kind", "spatial actions require BOX, CLICK, or CONFIRMED_MASK"))
        return
    if human_input.get("origin") != "external_human_ui":
        errors.append(_issue("HUMAN_INPUT_ORIGIN_INVALID", "/human_input/origin", "raw input must come directly from the external UI"))
    if human_input.get("human_confirmed") is not True:
        errors.append(_issue("HUMAN_CONFIRMATION_MISSING", "/human_input/human_confirmed", "explicit human confirmation is required"))
    if not _text(human_input.get("raw_payload_ref")) or not _hash(human_input.get("raw_payload_sha256")):
        errors.append(_issue("RAW_INPUT_PROVENANCE_MISSING", "/human_input", "raw payload reference and SHA-256 are required"))
    if kind == "BOX":
        box = human_input.get("box_xyxy")
        if not isinstance(box, list) or len(box) != 4 or not all(_finite(value) for value in box) or float(box[2]) <= float(box[0]) or float(box[3]) <= float(box[1]):
            errors.append(_issue("BOX_INVALID", "/human_input/box_xyxy", "BOX requires four finite coordinates with positive area"))
    elif kind == "CLICK":
        clicks = human_input.get("click_points")
        if not isinstance(clicks, list) or not clicks:
            errors.append(_issue("CLICK_INVALID", "/human_input/click_points", "at least one click is required"))
        elif not any(isinstance(item, Mapping) and item.get("label") == "positive" for item in clicks):
            errors.append(_issue("POSITIVE_CLICK_MISSING", "/human_input/click_points", "at least one positive click is required"))
    else:
        mask = human_input.get("confirmed_mask")
        if not isinstance(mask, Mapping):
            errors.append(_issue("CONFIRMED_MASK_MISSING", "/human_input/confirmed_mask", "confirmed mask payload is required"))
        else:
            if mask.get("format") not in {"lossless_PNG", "lossless_NPZ", "lossless_RLE"}:
                errors.append(_issue("MASK_FORMAT_INVALID", "/human_input/confirmed_mask/format", "mask must be lossless"))
            if not _text(mask.get("payload_ref")) or not _hash(mask.get("sha256")):
                errors.append(_issue("MASK_PROVENANCE_MISSING", "/human_input/confirmed_mask", "lossless mask reference and digest are required"))
            shape = mask.get("frame_shape")
            if not isinstance(shape, list) or len(shape) != 2 or not all(isinstance(value, int) and value > 0 for value in shape):
                errors.append(_issue("MASK_SHAPE_INVALID", "/human_input/confirmed_mask/frame_shape", "positive frame shape is required"))
            if mask.get("mask_origin") != "human_ui_confirmed" or mask.get("machine_candidate_mask") is not False:
                errors.append(_issue("MACHINE_MASK_NOT_ALLOWED", "/human_input/confirmed_mask", "machine candidate masks cannot be relabeled as human confirmed"))


def _validate_identity_input(human_input: Mapping[str, Any], errors: list[dict[str, str]]) -> None:
    """Validate direct ID selections without treating them as spatial evidence."""

    if human_input.get("kind") != "ID_SELECTION":
        errors.append(_issue("ID_SELECTION_KIND_INVALID", "/human_input/kind", "identity actions require ID_SELECTION"))
        return
    if human_input.get("origin") != "external_human_ui":
        errors.append(_issue("HUMAN_INPUT_ORIGIN_INVALID", "/human_input/origin", "ID selection must come directly from the external UI"))
    if human_input.get("human_confirmed") is not True:
        errors.append(_issue("HUMAN_CONFIRMATION_MISSING", "/human_input/human_confirmed", "explicit human confirmation is required"))
    if not _text(human_input.get("raw_payload_ref")) or not _hash(human_input.get("raw_payload_sha256")):
        errors.append(_issue("RAW_INPUT_PROVENANCE_MISSING", "/human_input", "raw ID-selection payload reference and SHA-256 are required"))
    selected = human_input.get("selected_public_ids")
    if not isinstance(selected, list) or not selected or any(_int(value) is None or int(value) < 0 for value in selected):
        errors.append(_issue("ID_SELECTION_INVALID", "/human_input/selected_public_ids", "selected public IDs must be a non-empty list of non-negative integers"))


def validate_real_human_event_v2(record: Any, *, require_server_auth: bool = False) -> dict[str, Any]:
    errors: list[dict[str, str]] = []
    if not isinstance(record, Mapping):
        return {"schema_version": SCHEMA_VERSION, "valid": False, "errors": [_issue("EVENT_NOT_OBJECT", "/", "event must be an object")], "canonical_action": None}
    errors.extend(_walk_forbidden(record))
    action = canonical_action(record.get("action_type", record.get("action")))
    if action is None:
        errors.append(_issue("ACTION_INVALID", "/action_type", "unsupported action"))
    for key in ("event_id", "sequence", "session_id", "annotator_id_hash", "timestamp", "frame_hash_sha256", "candidate_tape_ref"):
        if not _text(record.get(key)) and not _hash(record.get(key)):
            errors.append(_issue("PROVENANCE_FIELD_MISSING", f"/{key}", "required provenance field is missing"))
    for key in ("prefix_range", "future_ranges"):
        if key not in record or record.get(key) is None:
            errors.append(_issue("PROVENANCE_FIELD_MISSING", f"/{key}", "required temporal range is missing"))
    frame = _int(record.get("event_frame"))
    if frame is None or frame < 0:
        errors.append(_issue("EVENT_FRAME_INVALID", "/event_frame", "non-negative event_frame is required"))
    if record.get("split") not in {"train", "train_fold"}:
        errors.append(_issue("SPLIT_INVALID", "/split", "only train/train_fold are permitted"))
    source = record.get("interaction_source")
    if source not in {"ui_submission", "real_human"}:
        errors.append(_issue("INTERACTION_SOURCE_INVALID", "/interaction_source", "only server-bound UI submission or server-finalized real_human is allowed"))
    if record.get("test_fixture") is True or record.get("synthetic_fixture") is True:
        errors.append(_issue("FIXTURE_NOT_REAL_HUMAN", "/test_fixture", "fixtures cannot enter real-human tape"))
    if record.get("human_confirmed") is not True:
        errors.append(_issue("HUMAN_CONFIRMATION_MISSING", "/human_confirmed", "event-level confirmation is required"))
    if record.get("runtime_future_gt_used", False) is not False:
        errors.append(_issue("RUNTIME_FUTURE_GT_USED", "/runtime_future_gt_used", "runtime_future_gt_used must be false"))
    if require_server_auth and (source != "real_human" or record.get("server_generated_real_human") is not True or not _text(record.get("server_session_nonce"))):
        errors.append(_issue("SERVER_AUTH_MISSING", "/server_generated_real_human", "only the confirmed server UI path can finalize real_human"))
    if not _hash(record.get("annotator_id_hash")) or not _hash(record.get("frame_hash_sha256")):
        errors.append(_issue("PROVENANCE_HASH_INVALID", "/provenance", "annotator and frame SHA-256 are required"))
    try:
        datetime.fromisoformat(str(record.get("timestamp")).replace("Z", "+00:00"))
    except ValueError:
        errors.append(_issue("TIMESTAMP_INVALID", "/timestamp", "timestamp must be ISO-8601"))
    if record.get("prefix_range") is not None:
        prefix = record.get("prefix_range")
        if frame is not None and (not isinstance(prefix, list) or len(prefix) != 2 or prefix[1] != frame - 1 or prefix[0] < 0):
            errors.append(_issue("PREFIX_RANGE_INVALID", "/prefix_range", "prefix must end at event_frame-1"))
    if record.get("future_ranges") is not None and frame is not None:
        ranges = record.get("future_ranges")
        if not isinstance(ranges, Mapping):
            errors.append(_issue("FUTURE_RANGES_INVALID", "/future_ranges", "future ranges must be an object"))
        else:
            for horizon in (20, 50, 100):
                if ranges.get(f"H{horizon}") != [frame + 1, frame + horizon]:
                    errors.append(_issue("FUTURE_BOUNDARY_INVALID", f"/future_ranges/H{horizon}", "future range must begin at event_frame+1"))
    if record.get("prefix_range") is None:
        errors.append(_issue("PREFIX_RANGE_MISSING", "/prefix_range", "prefix range is required for an ingestible event"))
    if record.get("future_ranges") is None:
        errors.append(_issue("FUTURE_RANGES_MISSING", "/future_ranges", "H20/H50/H100 future ranges are required"))
    if action is not None:
        target_public_id = _int(record.get("public_id"))
        if action == "ADD_NEW_IDENTITY":
            if record.get("public_id") is not None:
                errors.append(_issue("ADD_PUBLIC_ID_INJECTION", "/public_id", "ADD public ID is allocated by the server, never entered by the user"))
            if record.get("public_id_source") not in {None, "system_allocator"}:
                errors.append(_issue("ADD_PUBLIC_ID_SOURCE_INVALID", "/public_id_source", "ADD source must be system_allocator after runtime allocation"))
        else:
            if target_public_id is None or target_public_id < 0:
                errors.append(_issue("PUBLIC_ID_INVALID", "/public_id", "existing identity action needs a direct public ID"))
            if record.get("public_id_source") not in {"human_selected_existing_public", "direct_user_public_id"}:
                errors.append(_issue("PUBLIC_ID_SOURCE_INVALID", "/public_id_source", "existing public ID source must be explicit human selection"))
        human_input = record.get("human_input")
        if action in SPATIAL_ACTIONS:
            if not isinstance(human_input, Mapping):
                errors.append(_issue("SPATIAL_INPUT_MISSING", "/human_input", "spatial action requires a raw human input"))
            else:
                _validate_spatial_input(human_input, errors)
        elif action in IDENTITY_ACTIONS:
            if not isinstance(human_input, Mapping):
                errors.append(_issue("IDENTITY_INPUT_MISSING", "/human_input", "identity action requires a direct UI selection"))
            else:
                _validate_identity_input(human_input, errors)
            if action in {"AUTHORITATIVE_REASSIGN", "ATOMIC_ID_SWAP"} and (not isinstance(human_input, Mapping) or human_input.get("kind") != "ID_SELECTION"):
                errors.append(_issue("ID_SELECTION_MISSING", "/human_input", "identity reassignment/swap requires ID_SELECTION, not ROI"))
            if action == "AUTHORITATIVE_REASSIGN":
                source_id = _int(record.get("source_public_id", record.get("current_public_id")))
                destination_id = _int(record.get("destination_public_id", record.get("other_public_id", record.get("public_id"))))
                if source_id is None or destination_id is None or source_id == destination_id:
                    errors.append(_issue("REASSIGN_IDS_INVALID", "/source_public_id", "source and destination public IDs must differ"))
            elif action == "ATOMIC_ID_SWAP":
                other = _int(record.get("other_public_id", record.get("destination_public_id")))
                if target_public_id is None or other is None or target_public_id == other:
                    errors.append(_issue("SWAP_IDS_INVALID", "/other_public_id", "swap public IDs must differ"))
                if record.get("atomic_transaction_confirmed") is not True:
                    errors.append(_issue("ATOMIC_CONFIRMATION_MISSING", "/atomic_transaction_confirmed", "swap requires two-sided confirmation"))
            elif action == "AUTHORITATIVE_DELETE" and record.get("delete_confirmed") is not True:
                errors.append(_issue("DELETE_CONFIRMATION_MISSING", "/delete_confirmed", "delete requires explicit confirmation"))
    return {
        "schema_version": SCHEMA_VERSION,
        "valid": not errors,
        "canonical_action": action,
        "errors": errors,
        "runtime_future_gt_used": False,
        "server_generated_real_human": record.get("server_generated_real_human") is True,
    }


def new_server_session(annotator_id: str, *, ui_version: str = "n72r1-ui-v1") -> dict[str, str]:
    if not _text(annotator_id):
        raise ValueError("annotator_id is required")
    nonce = secrets.token_urlsafe(24)
    session_id = "n72r1-session-" + secrets.token_hex(12)
    annotator_hash = hashlib.sha256(annotator_id.encode("utf-8")).hexdigest()
    return {"session_id": session_id, "server_session_nonce": nonce, "annotator_id_hash": annotator_hash, "ui_version": ui_version}


def finalize_ui_submission(record: Mapping[str, Any], session: Mapping[str, Any]) -> dict[str, Any]:
    if record.get("session_id") != session.get("session_id") or not _text(session.get("server_session_nonce")):
        raise ValueError("submission session does not match server session")
    if record.get("human_confirmed") is not True:
        raise ValueError("the confirm-and-submit action is required")
    finalized = dict(record)
    finalized["schema_version"] = SCHEMA_VERSION
    finalized["interaction_source"] = "real_human"
    finalized["server_generated_real_human"] = True
    finalized["server_session_nonce"] = str(session["server_session_nonce"])
    finalized["annotator_id_hash"] = str(session["annotator_id_hash"])
    audit = validate_real_human_event_v2(finalized, require_server_auth=True)
    if not audit["valid"]:
        raise ValueError(json.dumps(audit["errors"], sort_keys=True))
    return finalized


__all__ = [
    "ACTION_ALIASES",
    "IDENTITY_ACTIONS",
    "INPUT_KINDS",
    "SCHEMA_VERSION",
    "SPATIAL_ACTIONS",
    "canonical_action",
    "finalize_ui_submission",
    "new_server_session",
    "validate_real_human_event_v2",
]
