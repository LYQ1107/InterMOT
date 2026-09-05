"""Lossless, public-ID-free candidate pooling for N72R7.

The pool is a source adapter, not an identity authority.  Historical solver
labels on B0 rows are retained only as ``incumbent_public_id_if_any`` for
diagnosis; the normalized candidate always carries ``public_id=None``.  A
selector may inspect every source before the exact global public solver runs.
"""

from __future__ import annotations

from copy import deepcopy
import hashlib
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np


MAIN_B0_CANDIDATE = "MAIN_B0_CANDIDATE"
TARGET_SESSION_CURRENT_RAW = "TARGET_SESSION_CURRENT_RAW"
TARGET_SESSION_REQUERY = "TARGET_SESSION_REQUERY"
FUTURE_FRAME_REQUERY = "FUTURE_FRAME_REQUERY"
FEATURE_DIM = 512


def _finite_box(value: Any, label: str) -> list[float]:
    box = np.asarray(value, dtype=np.float64).reshape(-1)
    if box.size != 4 or not np.all(np.isfinite(box)):
        raise ValueError(f"{label}: box must be four finite xyxy values")
    return [float(item) for item in box]


def _feature(value: Any, label: str) -> np.ndarray | None:
    if value is None or (isinstance(value, Sequence) and len(value) == 0):
        return None
    array = np.asarray(value, dtype=np.float32).reshape(-1)
    if array.size != FEATURE_DIM or not np.all(np.isfinite(array)):
        raise ValueError(f"{label}: feature must be finite {FEATURE_DIM}-D or absent")
    norm = float(np.linalg.norm(array))
    if norm <= 1.0e-6:
        raise ValueError(f"{label}: feature has zero norm")
    return array / norm


def _feature_hash(array: np.ndarray | None, supplied: Any, raw_value: Any = None) -> str | None:
    if array is None:
        return None if supplied in (None, "", "None") else str(supplied)
    # Frozen artifacts hash the source float32 vector.  Normalizing for the
    # selector can change a nearly-unit vector by a few ulps, so hashing the
    # normalized copy would reject valid provenance.
    source = np.asarray(array if raw_value is None else raw_value, dtype="<f4").reshape(-1)
    actual = hashlib.sha256(source.tobytes()).hexdigest()
    if supplied not in (None, "", "None") and str(supplied) != actual:
        raise ValueError(f"feature hash mismatch: supplied={supplied} actual={actual}")
    return actual


def _raw_id(candidate: Mapping[str, Any]) -> int | None:
    for key in ("official_raw_sam_id", "raw_sam_id", "raw_native_id", "native_tid", "adapter_external_id"):
        value = candidate.get(key)
        if value is not None:
            return int(value)
    return None


def _normalize(
    candidate: Mapping[str, Any],
    *,
    source_kind: str,
    sequence: str,
    frame: int,
    candidate_index: int,
) -> dict[str, Any]:
    raw = deepcopy(dict(candidate))
    uid = raw.get("candidate_uid")
    if uid in (None, "", "None"):
        raise ValueError(f"{source_kind}:{sequence}:{frame} candidate_uid is required")
    uid = str(uid)
    box_value = raw.get("box_xyxy", raw.get("box"))
    box = _finite_box(box_value, f"{source_kind}:{uid}")
    feature = _feature(raw.get("feature", raw.get("embedding")), f"{source_kind}:{uid}")
    raw_feature = raw.get("feature", raw.get("embedding"))
    feature_hash = _feature_hash(feature, raw.get("feature_sha256"), raw_feature)
    source_session = raw.get("source_session_id", raw.get("target_session_scope"))
    incumbent = raw.get("solver_public_id", raw.get("public_id"))
    incumbent_public = None if incumbent in (None, "", "None") else int(incumbent)
    normalized = {
        "candidate_uid": uid,
        "candidate_source": source_kind,
        "source_kind": source_kind,
        "frame": int(frame),
        "sequence": str(sequence),
        "candidate_index": int(candidate_index),
        "box_xyxy": box,
        "geometry_valid": bool(box[2] > box[0] and box[3] > box[1]),
        "mask_sha256": None if raw.get("mask_sha256") is None else str(raw["mask_sha256"]),
        "feature_sha256": feature_hash,
        "feature_available": feature is not None,
        "feature_hash_basis": "source_float32_before_unit_normalization" if feature is not None else "supplied_source_artifact",
        "feature": feature,
        "official_raw_sam_id": _raw_id(raw),
        "adapter_external_id": None if raw.get("adapter_external_id") is None else int(raw["adapter_external_id"]),
        "native_scope": raw.get("native_scope", raw.get("native_tid_scope")),
        "confidence": float(raw.get("confidence", raw.get("presence_score", 0.0))),
        "presence_score": float(raw.get("presence_score", raw.get("confidence", 0.0))),
        "raw_continuity": 0.0,
        "incumbent_public_id_if_any": incumbent_public,
        "public_id": None,
        "public_id_authority": None,
        "source_session_id": None if source_session is None else str(source_session),
        "candidate_kind": raw.get("candidate_kind"),
        "runtime_future_gt_used": False,
        "runtime_gt_read": False,
        "posthoc_gt_used": False,
        "public_id_inference": False,
    }
    if not np.isfinite(normalized["confidence"]) or not np.isfinite(normalized["presence_score"]):
        raise ValueError(f"{source_kind}:{uid}: non-finite confidence/presence")
    if source_kind in {TARGET_SESSION_CURRENT_RAW, TARGET_SESSION_REQUERY, FUTURE_FRAME_REQUERY}:
        if raw.get("public_id") is not None:
            raise ValueError(f"target source carries a public ID: {uid}")
        allowed_kinds = {
            TARGET_SESSION_CURRENT_RAW: {"TARGET_CORRECTION_SESSION_CANDIDATE"},
            TARGET_SESSION_REQUERY: {
                "TARGET_CORRECTION_SESSION_REQUERY_CANDIDATE",
            },
            FUTURE_FRAME_REQUERY: {
                "FUTURE_FRAME_REQUERY_CANDIDATE",
            },
        }[source_kind]
        if raw.get("candidate_kind") not in allowed_kinds:
            raise ValueError(f"target source has wrong candidate kind: {uid}")
    return normalized


def build_candidate_pool(
    main_candidates: Sequence[Mapping[str, Any]],
    target_candidates: Sequence[Mapping[str, Any]] = (),
    *,
    sequence: str,
    frame: int,
    include_target_session: bool,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Normalize the complete current pool without assigning public IDs."""

    main = [
        _normalize(
            item,
            source_kind=MAIN_B0_CANDIDATE,
            sequence=str(sequence),
            frame=int(frame),
            candidate_index=index,
        )
        for index, item in enumerate(main_candidates)
    ]
    target = []
    if include_target_session:
        if len(target_candidates) > 1:
            raise ValueError(f"target-session pool is expected to contain at most one row: {sequence}:{frame}")
        target = [
            _normalize(
                item,
                source_kind=TARGET_SESSION_CURRENT_RAW,
                sequence=str(sequence),
                frame=int(frame),
                candidate_index=len(main) + index,
            )
            for index, item in enumerate(target_candidates)
        ]
    candidates = main + target
    uids = [str(item["candidate_uid"]) for item in candidates]
    if len(uids) != len(set(uids)):
        raise ValueError(f"candidate UID collision at {sequence}:{frame}")
    audit = {
        "schema_version": "N72R7_TARGET_CANDIDATE_POOL_V1",
        "sequence": str(sequence),
        "frame": int(frame),
        "candidate_count": len(candidates),
        "main_b0_candidate_count": len(main),
        "target_session_candidate_count": len(target),
        "candidate_sources": [item["candidate_source"] for item in candidates],
        "candidate_uids": uids,
        "public_id_inference": False,
        "all_candidate_public_ids_null_before_solver": all(item["public_id"] is None for item in candidates),
        "runtime_future_gt_used": False,
        "runtime_gt_read": False,
        "posthoc_gt_used": False,
    }
    return candidates, audit


def build_candidate_pool_with_requery(
    main_candidates: Sequence[Mapping[str, Any]],
    target_candidates: Sequence[Mapping[str, Any]] = (),
    requery_candidates: Sequence[Mapping[str, Any]] = (),
    *,
    sequence: str,
    frame: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Build a pool with the frozen current target row plus re-query rows.

    This is deliberately separate from :func:`build_candidate_pool`: the
    original N72R7 D2 contract allows exactly one current target-session row,
    while the R5 candidate-generator route needs several independently
    prompted, session-local rows.  Neither source carries public-ID authority;
    the exact solver remains the only assignment authority.
    """

    main = [
        _normalize(
            item,
            source_kind=MAIN_B0_CANDIDATE,
            sequence=str(sequence),
            frame=int(frame),
            candidate_index=index,
        )
        for index, item in enumerate(main_candidates)
    ]
    if len(target_candidates) > 1:
        raise ValueError(
            f"current target-session pool is expected to contain at most one row: {sequence}:{frame}"
        )
    current = [
        _normalize(
            item,
            source_kind=TARGET_SESSION_CURRENT_RAW,
            sequence=str(sequence),
            frame=int(frame),
            candidate_index=len(main) + index,
        )
        for index, item in enumerate(target_candidates)
    ]
    requery = [
        _normalize(
            item,
            source_kind=TARGET_SESSION_REQUERY,
            sequence=str(sequence),
            frame=int(frame),
            candidate_index=len(main) + len(current) + index,
        )
        for index, item in enumerate(requery_candidates)
    ]
    candidates = main + current + requery
    uids = [str(item["candidate_uid"]) for item in candidates]
    if len(uids) != len(set(uids)):
        raise ValueError(f"candidate UID collision at {sequence}:{frame}")
    audit = {
        "schema_version": "N72R7_TARGET_CANDIDATE_POOL_WITH_REQUERY_V1",
        "sequence": str(sequence),
        "frame": int(frame),
        "candidate_count": len(candidates),
        "main_b0_candidate_count": len(main),
        "target_session_candidate_count": len(current),
        "target_session_requery_candidate_count": len(requery),
        "candidate_sources": [item["candidate_source"] for item in candidates],
        "candidate_uids": uids,
        "public_id_inference": False,
        "all_candidate_public_ids_null_before_solver": all(
            item["public_id"] is None for item in candidates
        ),
        "runtime_future_gt_used": False,
        "runtime_gt_read": False,
        "posthoc_gt_used": False,
    }
    return candidates, audit


def build_candidate_pool_with_future_requery(
    main_candidates: Sequence[Mapping[str, Any]],
    target_candidates: Sequence[Mapping[str, Any]] = (),
    future_requery_candidates: Sequence[Mapping[str, Any]] = (),
    *,
    sequence: str,
    frame: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Build one frame's pool with a genuinely fresh future-frame query.

    ``FUTURE_FRAME_REQUERY`` is intentionally a separate source from the
    historical ``TARGET_SESSION_REQUERY`` source.  The latter refers to the
    N72R7 frozen/static candidate stream and must never be relabelled as a live
    future query.  All rows remain public-ID-free until the exact global solver
    assigns an identity.
    """

    main = [
        _normalize(
            item,
            source_kind=MAIN_B0_CANDIDATE,
            sequence=str(sequence),
            frame=int(frame),
            candidate_index=index,
        )
        for index, item in enumerate(main_candidates)
    ]
    if len(target_candidates) > 1:
        raise ValueError(
            f"current target-session pool is expected to contain at most one row: {sequence}:{frame}"
        )
    current = [
        _normalize(
            item,
            source_kind=TARGET_SESSION_CURRENT_RAW,
            sequence=str(sequence),
            frame=int(frame),
            candidate_index=len(main) + index,
        )
        for index, item in enumerate(target_candidates)
    ]
    future = [
        _normalize(
            item,
            source_kind=FUTURE_FRAME_REQUERY,
            sequence=str(sequence),
            frame=int(frame),
            candidate_index=len(main) + len(current) + index,
        )
        for index, item in enumerate(future_requery_candidates)
    ]
    candidates = main + current + future
    uids = [str(item["candidate_uid"]) for item in candidates]
    if len(uids) != len(set(uids)):
        raise ValueError(f"candidate UID collision at {sequence}:{frame}")
    audit = {
        "schema_version": "N72R10_CANDIDATE_POOL_WITH_FUTURE_REQUERY_V1",
        "sequence": str(sequence),
        "frame": int(frame),
        "candidate_count": len(candidates),
        "main_b0_candidate_count": len(main),
        "target_session_candidate_count": len(current),
        "future_frame_requery_candidate_count": len(future),
        "candidate_sources": [item["candidate_source"] for item in candidates],
        "candidate_uids": uids,
        "public_id_inference": False,
        "all_candidate_public_ids_null_before_solver": all(
            item["public_id"] is None for item in candidates
        ),
        "runtime_future_gt_used": False,
        "runtime_gt_read": False,
        "posthoc_gt_used": False,
    }
    return candidates, audit


def serializable_candidate(candidate: Mapping[str, Any], *, include_feature: bool = False) -> dict[str, Any]:
    """Return an audit row with no numpy objects and no candidate-owned public ID."""

    result = {
        key: value
        for key, value in candidate.items()
        if key != "feature"
    }
    feature = candidate.get("feature")
    result["feature"] = np.asarray(feature, dtype=np.float32).tolist() if include_feature and feature is not None else None
    result["feature_available"] = feature is not None
    result["public_id"] = None
    result["public_id_authority"] = None
    return result


__all__ = [
    "FEATURE_DIM",
    "FUTURE_FRAME_REQUERY",
    "MAIN_B0_CANDIDATE",
    "TARGET_SESSION_CURRENT_RAW",
    "TARGET_SESSION_REQUERY",
    "build_candidate_pool",
    "build_candidate_pool_with_future_requery",
    "build_candidate_pool_with_requery",
    "serializable_candidate",
]
