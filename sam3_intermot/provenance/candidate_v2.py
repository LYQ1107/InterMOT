"""Mandatory candidate provenance and UID V2 contracts for N72R1.

The builder is deliberately independent of GT, posthoc metrics, and human
events.  It receives the official raw SAM axis from the parsed observation and
requires the caller to supply independently audited local/global axes.  It
never infers a public identity.
"""

from __future__ import annotations

import hashlib
import math
from collections import Counter, defaultdict
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from sam3_intermot.backend.output_types import PromptObjectObservation
from sam3_intermot.provenance.mapping import (
    canonical_box_digest,
    canonical_candidate_uid_v2,
    canonical_mask_digest,
)


SCHEMA_VERSION = "N72R1_CANDIDATE_V2"
CANDIDATE_V2_SCHEMA = SCHEMA_VERSION
LEGACY_NATIVE_TID_SEMANTICS = "adapter_visible_stable_id_after_binding"
REQUIRED_FIELDS = (
    "schema_version",
    "source_run_id",
    "sequence",
    "video_id",
    "checkpoint_sha256",
    "runtime_config_sha256",
    "session_id",
    "segment_id",
    "window_id",
    "chunk_id",
    "frame_idx",
    "candidate_index",
    "official_raw_sam_id",
    "adapter_external_id",
    "legacy_native_tid",
    "legacy_native_tid_semantics",
    "segment_local_id",
    "sequence_global_id",
    "candidate_uid",
    "candidate_uid_v2",
    "box_xyxy",
    "box_digest",
    "mask_shape",
    "mask_sha256",
    "confidence",
    "presence_score",
    "source",
    "feature_status",
    "feature_source",
    "feature_dim",
    "feature_sha256",
    "runtime_future_gt_used",
)
LEGACY_COMMON_FIELDS = (
    "frame_idx",
    "native_tid",
    "box_xyxy",
    "mask",
    "confidence",
    "presence_score",
    "source",
    "embedding_status",
    "feature_source",
    "is_human_verified",
    "candidate_index",
)


def _text(value: Any) -> str | None:
    return value.strip() if isinstance(value, str) and value.strip() else None


def _finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def _digest_feature(feature: Any) -> tuple[np.ndarray | None, str | None, float | None, int | None]:
    if feature is None:
        return None, None, None, None
    try:
        vector = np.asarray(feature, dtype=np.float32).reshape(-1)
    except (TypeError, ValueError):
        return None, None, None, None
    if vector.size == 0 or not np.all(np.isfinite(vector)):
        return None, None, None, int(vector.size)
    norm = float(np.linalg.norm(vector))
    if norm <= 1e-6:
        return None, None, norm, int(vector.size)
    vector = (vector / norm).astype(np.float32)
    return vector, hashlib.sha256(vector.tobytes()).hexdigest(), float(np.linalg.norm(vector)), int(vector.size)


def _metadata_required(metadata: Mapping[str, Any]) -> list[str]:
    required = (
        "source_run_id",
        "sequence",
        "video_id",
        "checkpoint_sha256",
        "runtime_config_sha256",
        "session_id",
        "segment_id",
        "window_id",
        "chunk_id",
    )
    return [key for key in required if _text(metadata.get(key)) is None]


def build_candidate_v2_row(
    observation: PromptObjectObservation,
    *,
    metadata: Mapping[str, Any],
    candidate_index: int,
    segment_local_id: str,
    sequence_global_id: str,
    feature: Any = None,
    feature_status: str | None = None,
    feature_source: str | None = None,
) -> dict[str, Any]:
    """Build one mandatory V2 row from one parsed official observation.

    ``segment_local_id`` and ``sequence_global_id`` are required arguments to
    prevent the common but unsafe ``raw_id == local_id == global_id`` shortcut.
    The caller must obtain them from a same-run adapter/handover ledger.
    """

    missing = _metadata_required(metadata)
    if missing:
        raise ValueError(f"candidate V2 metadata missing: {','.join(missing)}")
    if not isinstance(candidate_index, int) or isinstance(candidate_index, bool) or candidate_index < 0:
        raise ValueError("candidate_index must be a non-negative integer")
    local = _text(segment_local_id)
    global_id = _text(sequence_global_id)
    if local is None or global_id is None:
        raise ValueError("segment_local_id and sequence_global_id are mandatory audited axes")
    try:
        box = np.asarray(observation.box_xyxy, dtype="<f4").reshape(-1)
        if box.size != 4 or not np.all(np.isfinite(box)):
            raise ValueError
        mask = np.asarray(observation.mask)
        if mask.ndim != 2:
            raise ValueError
    except (TypeError, ValueError) as exc:
        raise ValueError("observation box/mask is not finite and two-dimensional") from exc

    raw = observation.raw_sam_object_id
    raw_id = None if raw is None else int(raw)
    adapter_id = int(observation.sam_object_id)
    box_list = [float(value) for value in box.tolist()]
    box_digest = canonical_box_digest(box)
    mask_sha256 = canonical_mask_digest(mask)
    vector, feature_sha256, feature_norm, feature_dim = _digest_feature(feature)
    status = feature_status or ("AVAILABLE" if vector is not None else "NOT_EXPOSED")
    source = feature_source or ("machine_candidate_embedding" if vector is not None else "official_response_no_embedding")
    if "human" in source.lower() or "human" in status.lower():
        raise ValueError("human evidence cannot enter the machine candidate exporter")
    uid = None
    if raw_id is not None:
        uid = canonical_candidate_uid_v2(
            source_run_id=str(metadata["source_run_id"]),
            sequence=str(metadata["sequence"]),
            session_id=str(metadata["session_id"]),
            segment_id=str(metadata["segment_id"]),
            window_id=str(metadata["window_id"]),
            chunk_id=str(metadata["chunk_id"]),
            frame_idx=int(observation.frame_idx),
            candidate_index=int(candidate_index),
            official_raw_sam_id=raw_id,
            adapter_external_id=adapter_id,
            box_digest=box_digest,
            mask_sha256=mask_sha256,
        )
    row = {
        "schema_version": SCHEMA_VERSION,
        "source_run_id": str(metadata["source_run_id"]),
        "sequence": str(metadata["sequence"]),
        "video_id": str(metadata["video_id"]),
        "checkpoint_sha256": str(metadata["checkpoint_sha256"]),
        "runtime_config_sha256": str(metadata["runtime_config_sha256"]),
        "session_id": str(metadata["session_id"]),
        "segment_id": str(metadata["segment_id"]),
        "window_id": str(metadata["window_id"]),
        "chunk_id": str(metadata["chunk_id"]),
        "frame_idx": int(observation.frame_idx),
        "candidate_index": int(candidate_index),
        "official_raw_sam_id": raw_id,
        "official_raw_sam_id_source": "official_out_obj_ids" if raw_id is not None else "UNAVAILABLE_NOT_OFFICIAL_OBSERVATION",
        "adapter_external_id": adapter_id,
        "legacy_native_tid": adapter_id,
        "legacy_native_tid_semantics": LEGACY_NATIVE_TID_SEMANTICS,
        "segment_local_id": local,
        "sequence_global_id": global_id,
        "candidate_uid": uid,
        "candidate_uid_v2": uid,
        "box_xyxy": box_list,
        "box_digest": box_digest,
        "mask_shape": [int(value) for value in mask.shape],
        "mask_sha256": mask_sha256,
        "confidence": float(observation.confidence),
        "presence_score": None if observation.presence_score is None else float(observation.presence_score),
        "source": str(observation.source),
        "feature_status": status,
        "feature_source": source,
        "feature_dim": feature_dim,
        "feature_norm": feature_norm,
        "feature_sha256": feature_sha256,
        "runtime_future_gt_used": False,
        "is_human_verified": bool(observation.is_human_verified),
    }
    if vector is not None:
        row["feature"] = vector.tolist()
    else:
        row["feature"] = None
    return row


def legacy_common_projection(legacy_row: Mapping[str, Any]) -> dict[str, Any]:
    """Return the fields used for exact V2-vs-legacy equivalence checks."""

    result = {}
    for key in LEGACY_COMMON_FIELDS:
        value = legacy_row.get(key)
        if isinstance(value, np.ndarray):
            value = value.tolist()
        result[key] = value
    return result


def v2_common_projection(v2_row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "frame_idx": v2_row.get("frame_idx"),
        "native_tid": v2_row.get("legacy_native_tid"),
        "box_xyxy": v2_row.get("box_xyxy"),
        "mask": v2_row.get("mask"),
        "confidence": v2_row.get("confidence"),
        "presence_score": v2_row.get("presence_score"),
        "source": v2_row.get("source"),
        "embedding_status": "MACHINE_ROI_FALLBACK" if v2_row.get("feature_status") == "AVAILABLE" else v2_row.get("feature_status"),
        "feature_source": v2_row.get("feature_source"),
        "is_human_verified": v2_row.get("is_human_verified"),
        "candidate_index": v2_row.get("candidate_index"),
    }


def validate_candidate_v2_row(row: Mapping[str, Any], *, require_official_raw: bool = True) -> list[str]:
    errors: list[str] = []
    if not isinstance(row, Mapping):
        return ["row_not_mapping"]
    errors.extend(f"missing:{key}" for key in REQUIRED_FIELDS if key not in row)
    if row.get("schema_version") != SCHEMA_VERSION:
        errors.append("schema_version_mismatch")
    if require_official_raw and row.get("official_raw_sam_id") is None:
        errors.append("official_raw_sam_id_missing")
    for key in ("source_run_id", "sequence", "video_id", "session_id", "segment_id", "window_id", "chunk_id", "segment_local_id", "sequence_global_id"):
        if _text(row.get(key)) is None:
            errors.append(f"text_axis_invalid:{key}")
    for key in ("frame_idx", "candidate_index", "adapter_external_id", "legacy_native_tid"):
        value = row.get(key)
        if not isinstance(value, int) or isinstance(value, bool) or (key in {"frame_idx", "candidate_index"} and value < 0):
            errors.append(f"integer_axis_invalid:{key}")
    if row.get("legacy_native_tid_semantics") != LEGACY_NATIVE_TID_SEMANTICS:
        errors.append("legacy_native_tid_semantics_mismatch")
    box = row.get("box_xyxy")
    if not isinstance(box, list) or len(box) != 4 or not all(_finite(value) for value in box):
        errors.append("box_invalid")
    if not isinstance(row.get("mask_shape"), list) or len(row.get("mask_shape", [])) != 2 or not all(isinstance(v, int) and v > 0 for v in row.get("mask_shape", [])):
        errors.append("mask_shape_invalid")
    for key in ("box_digest", "mask_sha256", "checkpoint_sha256", "runtime_config_sha256"):
        value = row.get(key)
        if not isinstance(value, str) or len(value) != 64:
            errors.append(f"digest_invalid:{key}")
    if not isinstance(row.get("candidate_uid"), str) or row.get("candidate_uid") != row.get("candidate_uid_v2"):
        errors.append("candidate_uid_missing_or_mismatch")
    if row.get("runtime_future_gt_used") is not False:
        errors.append("runtime_future_gt_used_not_false")
    confidence = row.get("confidence")
    if not _finite(confidence):
        errors.append("confidence_invalid")
    feature = row.get("feature")
    if row.get("feature_status") == "AVAILABLE":
        if not isinstance(feature, list) or len(feature) != 512 or not all(_finite(value) for value in feature):
            errors.append("available_feature_invalid")
        if row.get("feature_dim") != 512 or not isinstance(row.get("feature_sha256"), str) or len(row.get("feature_sha256")) != 64:
            errors.append("available_feature_provenance_invalid")
    if "human" in str(row.get("feature_source", "")).lower():
        errors.append("human_feature_in_machine_row")
    return errors


def validate_candidate_v2_rows(rows: Iterable[Mapping[str, Any]], *, require_official_raw: bool = True) -> dict[str, Any]:
    records = [dict(row) for row in rows]
    errors: list[dict[str, Any]] = []
    uid_counts: Counter[str] = Counter()
    frame_indices: defaultdict[tuple[str, int], list[int]] = defaultdict(list)
    raw_coverage = 0
    for index, row in enumerate(records):
        row_errors = validate_candidate_v2_row(row, require_official_raw=require_official_raw)
        errors.extend({"row": index, "code": error} for error in row_errors)
        if row.get("candidate_uid") is not None:
            uid_counts[str(row.get("candidate_uid"))] += 1
        frame_indices[(str(row.get("source_run_id")), int(row.get("frame_idx", -1)))].append(int(row.get("candidate_index", -1)))
        raw_coverage += int(row.get("official_raw_sam_id") is not None)
    duplicate_uids = {uid: count for uid, count in uid_counts.items() if count > 1}
    errors.extend({"code": "duplicate_candidate_uid", "candidate_uid": uid, "count": count} for uid, count in sorted(duplicate_uids.items()))
    duplicate_order = {
        key: values for key, values in frame_indices.items() if len(values) != len(set(values)) or values != list(range(len(values)))
    }
    errors.extend({"code": "candidate_order_invalid", "frame": list(key), "indices": values} for key, values in sorted(duplicate_order.items(), key=str))
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "PASS" if not errors else "FAIL_CANDIDATE_V2",
        "row_count": len(records),
        "unique_candidate_uid_count": len(uid_counts),
        "candidate_uid_collision_count": sum(count - 1 for count in duplicate_uids.values()),
        "raw_id_coverage": None if not records else raw_coverage / len(records),
        "source_run_count": len({str(row.get("source_run_id")) for row in records}),
        "session_count": len({str(row.get("session_id")) for row in records}),
        "errors": errors,
    }


def schema_document() -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "required_fields": list(REQUIRED_FIELDS),
        "legacy_native_tid_semantics": LEGACY_NATIVE_TID_SEMANTICS,
        "uid_contract": "N72R1_CANDIDATE_UID_V2 over source_run/session/segment/window/chunk/frame/index/raw/adapter/box_digest/mask_sha256",
        "public_id_policy": "candidate rows never infer public_id; same-run assignment sidecar supplies it only via an explicit resolver",
        "runtime_future_gt_used": False,
        "human_evidence_in_exporter": False,
    }


__all__ = [
    "CANDIDATE_UID_V2_SCHEMA",
    "LEGACY_NATIVE_TID_SEMANTICS",
    "REQUIRED_FIELDS",
    "SCHEMA_VERSION",
    "build_candidate_v2_row",
    "legacy_common_projection",
    "schema_document",
    "validate_candidate_v2_row",
    "validate_candidate_v2_rows",
    "v2_common_projection",
]
