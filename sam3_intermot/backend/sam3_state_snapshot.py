"""Lightweight continuation snapshots for the pinned SAM3 multiplex adapter.

The official predictor keeps large image/features and mask tensors inside its
session.  A correction-state experiment needs to roll back the *container
graph* around those tensors, not copy model weights or duplicate the video.
This module therefore copies dictionaries/lists and small numpy metadata while
intentionally retaining tensor storage by reference.  It is a continuation
transaction primitive, not a general serialization/checkpoint format.
"""

from __future__ import annotations

import copy
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np
import torch


def _copy_structure(value: Any, *, max_array_copy_elements: int = 3_000_000) -> Any:
    """Copy control containers without copying large tensor storage."""

    if isinstance(value, torch.Tensor):
        return value
    if isinstance(value, np.ndarray):
        return value.copy() if value.size <= max_array_copy_elements else value
    if isinstance(value, Mapping):
        pairs = [(_copy_structure(key), _copy_structure(child)) for key, child in value.items()]
        if isinstance(value, defaultdict):
            return defaultdict(value.default_factory, pairs)
        try:
            return type(value)(pairs)
        except TypeError:
            return dict(pairs)
    if isinstance(value, list):
        return [_copy_structure(child) for child in value]
    if isinstance(value, tuple):
        return tuple(_copy_structure(child) for child in value)
    if isinstance(value, set):
        return {_copy_structure(child) for child in value}
    if isinstance(value, (str, bytes, int, float, bool, type(None))):
        return value
    try:
        return copy.copy(value)
    except Exception:
        return value


def _copy_output_cache(cache: Mapping[int, Any]) -> dict[int, Any]:
    copied: dict[int, Any] = {}
    for frame, observations in cache.items():
        if isinstance(observations, list):
            copied[int(frame)] = [
                observation.copy() if hasattr(observation, "copy") else _copy_structure(observation)
                for observation in observations
            ]
        else:
            copied[int(frame)] = _copy_structure(observations)
    return copied


@dataclass
class ContinuationSnapshot:
    """In-memory rollback token for one active backend session."""

    session_id: str | None
    state: Any
    objects: Any
    output_cache: dict[int, Any]
    ext_to_sam: dict[int, int]
    sam_to_ext: dict[int, int]
    last_prompt_frame: int | None
    text_prompt: str | None
    prompt_fallback_log: list[dict[str, Any]]
    resume_repair_log: list[dict[str, Any]]


def snapshot_continuation_state(backend: Any) -> ContinuationSnapshot:
    """Capture the active official state and project-side continuation state."""

    predictor = getattr(backend, "_predictor", None)
    session_id = getattr(backend, "_session_id", None)
    entry = None
    if predictor is not None and session_id is not None:
        entry = getattr(predictor, "_all_inference_states", {}).get(session_id)
    state = entry.get("state") if isinstance(entry, Mapping) else None
    return ContinuationSnapshot(
        session_id=session_id,
        state=_copy_structure(state),
        objects=_copy_structure(getattr(backend, "_objects", {})),
        output_cache=_copy_output_cache(getattr(backend, "_output_cache", {})),
        ext_to_sam={int(k): int(v) for k, v in getattr(backend, "_ext_to_sam", {}).items()},
        sam_to_ext={int(k): int(v) for k, v in getattr(backend, "_sam_to_ext", {}).items()},
        last_prompt_frame=getattr(backend, "_last_prompt_frame", None),
        text_prompt=getattr(backend, "_text_prompt", None),
        prompt_fallback_log=_copy_structure(getattr(backend, "_prompt_fallback_log", [])),
        resume_repair_log=_copy_structure(getattr(backend, "_resume_repair_log", [])),
    )


def restore_continuation_state(backend: Any, snapshot: ContinuationSnapshot) -> None:
    """Restore a snapshot into the same predictor session.

    The predictor/model object is deliberately not replaced.  This preserves
    the loaded checkpoint and CUDA allocations while restoring the official
    state dictionaries and the adapter's public-ID bookkeeping.
    """

    if getattr(backend, "_session_id", None) != snapshot.session_id:
        raise RuntimeError("continuation snapshot belongs to a different session")
    predictor = getattr(backend, "_predictor", None)
    if predictor is None or snapshot.session_id is None:
        raise RuntimeError("cannot restore continuation without an active predictor session")
    entry = predictor._all_inference_states.get(snapshot.session_id)
    if not isinstance(entry, Mapping):
        raise RuntimeError("predictor session entry disappeared before rollback")
    if isinstance(snapshot.state, Mapping):
        current = entry.get("state")
        if isinstance(current, dict):
            current.clear()
            current.update(_copy_structure(snapshot.state))
            entry["state"] = current
        else:
            entry["state"] = _copy_structure(snapshot.state)
    else:
        entry["state"] = snapshot.state
    restored_state = entry.get("state")
    if isinstance(restored_state, dict):
        # Official _tracker_add_new_objects passes the outer feature_cache by
        # reference into each inner demo state.  A structural snapshot must
        # restore that alias or later frames are invisible to auto-updates.
        shared_features = restored_state.get("feature_cache")
        if isinstance(shared_features, dict):
            for tracker_state in restored_state.get("sam2_inference_states", []):
                if isinstance(tracker_state, dict) and "cached_features" in tracker_state:
                    tracker_state["cached_features"] = shared_features
    backend._objects = _copy_structure(snapshot.objects)
    backend._output_cache = _copy_output_cache(snapshot.output_cache)
    backend._ext_to_sam = dict(snapshot.ext_to_sam)
    backend._sam_to_ext = dict(snapshot.sam_to_ext)
    backend._last_prompt_frame = snapshot.last_prompt_frame
    backend._text_prompt = snapshot.text_prompt
    backend._prompt_fallback_log = _copy_structure(snapshot.prompt_fallback_log)
    backend._resume_repair_log = _copy_structure(snapshot.resume_repair_log)


def state_container_summary(backend: Any) -> dict[str, Any]:
    """Return a compact, JSON-safe continuation summary for audit logs."""

    predictor = getattr(backend, "_predictor", None)
    session_id = getattr(backend, "_session_id", None)
    entry = getattr(predictor, "_all_inference_states", {}).get(session_id) if predictor is not None else None
    state = entry.get("state") if isinstance(entry, Mapping) else None
    return {
        "session_id_present": session_id is not None,
        "state_keys": sorted(str(key) for key in state.keys()) if isinstance(state, Mapping) else [],
        "tracker_state_count": len(state.get("sam2_inference_states", [])) if isinstance(state, Mapping) else 0,
        "tracker_ids": [
            int(obj_id)
            for tracker_state in (state.get("sam2_inference_states", []) if isinstance(state, Mapping) else [])
            for obj_id in np.asarray(tracker_state.get("obj_ids", [])).reshape(-1)
        ],
        "cached_frame_count": len(state.get("cached_frame_outputs", {})) if isinstance(state, Mapping) else 0,
        "output_cache_frames": sorted(int(frame) for frame in getattr(backend, "_output_cache", {}).keys()),
        "ext_to_sam": {str(k): int(v) for k, v in getattr(backend, "_ext_to_sam", {}).items()},
    }
