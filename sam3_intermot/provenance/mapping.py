"""Lossless, exact identity-axis mapping for N72.

The mapper only reconciles identifiers supplied by an authoritative source.
It intentionally has no image, GT, appearance, temporal, or IoU fallback.  A
missing source is a first-class result, not an invitation to infer a public ID.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter, defaultdict
from typing import Any, Iterable, Mapping


MAPPING_STATUSES = (
    "EXACT",
    "UNMAPPED_NO_SOURCE",
    "AMBIGUOUS_ONE_TO_MANY",
    "COLLISION",
    "AXIS_MISMATCH",
    "STALE_MAPPING",
    "CANDIDATE_ABSENT",
    "PUBLIC_ASSIGNMENT_ABSENT",
)


def _int_or_none(value: Any) -> int | None:
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


def _text_or_none(value: Any) -> str | None:
    if isinstance(value, str) and value.strip():
        return value.strip()
    if isinstance(value, int) and not isinstance(value, bool):
        return str(value)
    return None


def _finite_frame(value: Any) -> int | None:
    frame = _int_or_none(value)
    return frame if frame is not None and frame >= 0 else None


def canonical_candidate_uid(
    *,
    sequence: str,
    frame: int,
    raw_native_id: int,
    adapter_external_id: int | None,
    segment_local_id: str | int,
    sequence_global_id: str | int,
) -> str:
    """Return a deterministic candidate-row UID over every identity axis."""

    payload = {
        "schema": "N72_CANDIDATE_UID_V1",
        "sequence": str(sequence),
        "frame": int(frame),
        "raw_native_id": int(raw_native_id),
        "adapter_external_id": None if adapter_external_id is None else int(adapter_external_id),
        "segment_local_id": str(segment_local_id),
        "sequence_global_id": str(sequence_global_id),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _normalise_source(source: Any) -> tuple[str, int] | None:
    if not isinstance(source, Mapping):
        return None
    name = source.get("source", source.get("mapping_source"))
    public_id = _int_or_none(source.get("public_id"))
    if not isinstance(name, str) or not name.strip() or public_id is None:
        return None
    return name.strip(), public_id


def resolve_exact_mapping(
    row: Mapping[str, Any],
    *,
    exact_sources: Iterable[Mapping[str, Any]] = (),
    public_assignment_absent: bool = False,
) -> dict[str, Any]:
    """Resolve one row using only explicitly supplied exact source records.

    ``exact_sources`` is deliberately a list of provenance records rather
    than a lookup built from geometry or labels.  The caller must prove that
    each source key refers to this candidate; this function only checks the
    identifier axes and source agreement.
    """

    sequence = _text_or_none(row.get("sequence"))
    frame = _finite_frame(row.get("frame"))
    raw = _int_or_none(row.get("raw_native_id", row.get("raw_sam_native_id")))
    adapter = _int_or_none(row.get("adapter_external_id"))
    local = _text_or_none(row.get("segment_local_id", row.get("local_id")))
    global_id = _text_or_none(row.get("sequence_global_id", row.get("global_id")))
    axis_errors: list[str] = []
    if sequence is None:
        axis_errors.append("sequence_invalid")
    if frame is None:
        axis_errors.append("frame_invalid")
    if raw is None:
        axis_errors.append("raw_native_id_missing_or_invalid")
    if local is None:
        axis_errors.append("segment_local_id_missing_or_invalid")
    if global_id is None:
        axis_errors.append("sequence_global_id_missing_or_invalid")
    if axis_errors:
        return {
            "status": "AXIS_MISMATCH",
            "candidate_uid": None,
            "sequence": sequence,
            "frame": frame,
            "raw_native_id": raw,
            "adapter_external_id": adapter,
            "segment_local_id": local,
            "sequence_global_id": global_id,
            "public_id": None,
            "resolution_sources": [],
            "errors": axis_errors,
        }

    uid = canonical_candidate_uid(
        sequence=sequence,
        frame=frame,
        raw_native_id=raw,
        adapter_external_id=adapter,
        segment_local_id=local,
        sequence_global_id=global_id,
    )
    sources: list[dict[str, Any]] = []
    for item in exact_sources:
        normalized = _normalise_source(item)
        if normalized is not None:
            name, public_id = normalized
            sources.append({"source": name, "public_id": public_id})

    public_ids = sorted({int(item["public_id"]) for item in sources})
    if len(public_ids) > 1:
        status = "AMBIGUOUS_ONE_TO_MANY"
        public_id = None
    elif len(public_ids) == 1:
        status = "EXACT"
        public_id = public_ids[0]
    elif public_assignment_absent:
        status = "PUBLIC_ASSIGNMENT_ABSENT"
        public_id = None
    else:
        status = "UNMAPPED_NO_SOURCE"
        public_id = None
    return {
        "status": status,
        "candidate_uid": uid,
        "sequence": sequence,
        "frame": frame,
        "raw_native_id": raw,
        "adapter_external_id": adapter,
        "segment_local_id": local,
        "sequence_global_id": global_id,
        "public_id": public_id,
        "resolution_sources": sources,
        "errors": [],
    }


def validate_mapping_batch(
    rows: Iterable[Mapping[str, Any]],
    *,
    require_raw_coverage: bool = False,
    require_public_assignment: bool = False,
) -> dict[str, Any]:
    """Audit uniqueness and exactness of already-resolved candidate rows."""

    records = [dict(row) for row in rows]
    uid_counts = Counter(str(row.get("candidate_uid")) for row in records)
    frame_raw_counts = Counter(
        (str(row.get("sequence")), int(row.get("frame", -1)), int(row.get("raw_native_id", -1)))
        for row in records
        if row.get("raw_native_id") is not None
    )
    frame_public: defaultdict[tuple[str, int, int], list[str]] = defaultdict(list)
    for row in records:
        public_id = row.get("public_id")
        if public_id is not None:
            frame_public[(str(row.get("sequence")), int(row.get("frame", -1)), int(public_id))].append(
                str(row.get("candidate_uid"))
            )

    status_counts = Counter(str(row.get("status", "MISSING_STATUS")) for row in records)
    errors: list[dict[str, Any]] = []
    duplicate_uids = sorted(uid for uid, count in uid_counts.items() if count > 1)
    for uid in duplicate_uids:
        errors.append({"code": "DUPLICATE_CANDIDATE_UID", "candidate_uid": uid, "count": uid_counts[uid]})
    duplicate_frame_raw = sorted(
        {key for key, count in frame_raw_counts.items() if count > 1},
        key=str,
    )
    for key in duplicate_frame_raw:
        errors.append({"code": "DUPLICATE_RAW_ID_IN_FRAME", "key": list(key), "count": frame_raw_counts[key]})
    public_collisions = sorted(
        (key, values) for key, values in frame_public.items() if len(values) > 1
    )
    for key, values in public_collisions:
        errors.append({"code": "PUBLIC_ID_COLLISION_IN_FRAME", "key": list(key), "candidate_uids": values})
    invalid_statuses = sorted(status for status in status_counts if status not in MAPPING_STATUSES)
    for status in invalid_statuses:
        errors.append({"code": "UNKNOWN_MAPPING_STATUS", "status": status})
    if require_raw_coverage:
        missing_raw = [
            str(row.get("candidate_uid"))
            for row in records
            if row.get("raw_native_id") is None
        ]
        if missing_raw:
            errors.append({"code": "RAW_NATIVE_COVERAGE_INCOMPLETE", "count": len(missing_raw), "candidate_uids": missing_raw[:20]})
    if require_public_assignment:
        missing_public = [
            str(row.get("candidate_uid"))
            for row in records
            if row.get("status") != "EXACT" or row.get("public_id") is None
        ]
        if missing_public:
            errors.append({"code": "PUBLIC_ASSIGNMENT_COVERAGE_INCOMPLETE", "count": len(missing_public), "candidate_uids": missing_public[:20]})
    return {
        "status": "PASS" if not errors else "FAIL_MAPPING_INTEGRITY",
        "row_count": len(records),
        "unique_candidate_uid_count": len(uid_counts),
        "status_counts": dict(sorted(status_counts.items())),
        "duplicate_candidate_uid_count": len(duplicate_uids),
        "duplicate_raw_id_frame_count": len(duplicate_frame_raw),
        "public_collision_frame_count": len(public_collisions),
        "errors": errors,
        "raw_native_coverage": (
            None
            if not records
            else float(sum(row.get("raw_native_id") is not None for row in records) / len(records))
        ),
        "public_assignment_coverage": (
            None
            if not records
            else float(sum(row.get("status") == "EXACT" and row.get("public_id") is not None for row in records) / len(records))
        ),
    }


__all__ = [
    "MAPPING_STATUSES",
    "canonical_candidate_uid",
    "resolve_exact_mapping",
    "validate_mapping_batch",
]
