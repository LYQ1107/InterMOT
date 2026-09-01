"""Small, target-scoped audits for interactive correction state changes.

The audit intentionally records structure and scalar tensor statistics rather
than serializing SAM3 features or a whole inference session.  It is used by
the N30 write-path ablation to distinguish official tracker state, project
backend bookkeeping, and identity-memory state.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional

import numpy as np
import torch


@dataclass
class StateAuditSnapshot:
    """JSON-safe state payload plus temporary tensors used for delta norms."""

    payload: dict[str, Any]
    captured_tensors: dict[str, torch.Tensor]


def _key(value: Any) -> str:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return str(value)
    return repr(value)


def _finite_stats(value: Any) -> Optional[dict[str, Any]]:
    if isinstance(value, torch.Tensor):
        tensor = value.detach()
        try:
            finite_tensor = tensor.float()
            finite = bool(torch.isfinite(finite_tensor).all().item())
            flat = finite_tensor.reshape(-1)
            if flat.numel():
                return {
                    "shape": list(tensor.shape),
                    "dtype": str(tensor.dtype),
                    "finite": finite,
                    "l2": float(torch.linalg.vector_norm(flat).cpu()),
                    "mean": float(flat.mean().cpu()),
                    "max_abs": float(flat.abs().max().cpu()),
                }
            return {
                "shape": list(tensor.shape),
                "dtype": str(tensor.dtype),
                "finite": finite,
                "l2": 0.0,
                "mean": 0.0,
                "max_abs": 0.0,
            }
        except Exception as exc:  # pragma: no cover - unusual official buffer
            return {"shape": list(tensor.shape), "dtype": str(tensor.dtype), "error": type(exc).__name__}
    if isinstance(value, np.ndarray):
        array = np.asarray(value)
        try:
            numeric = array.astype(np.float32, copy=False)
            finite = bool(np.isfinite(numeric).all())
            return {
                "shape": list(array.shape),
                "dtype": str(array.dtype),
                "finite": finite,
                "l2": float(np.linalg.norm(numeric.reshape(-1))),
                "mean": float(numeric.mean()) if numeric.size else 0.0,
                "max_abs": float(np.abs(numeric).max()) if numeric.size else 0.0,
            }
        except Exception as exc:  # pragma: no cover - unusual object array
            return {"shape": list(array.shape), "dtype": str(array.dtype), "error": type(exc).__name__}
    return None


def _json_scalar(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, torch.Tensor) and value.numel() == 1:
        return value.detach().cpu().item()
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return None


def _summary_value(
    value: Any,
    *,
    path: str,
    captured: dict[str, torch.Tensor],
    capture: bool,
    max_capture_elements: int,
) -> Any:
    stats = _finite_stats(value)
    if stats is not None:
        if capture and isinstance(value, torch.Tensor) and value.numel() <= max_capture_elements:
            captured[path] = value.detach().float().cpu().clone()
        return stats
    scalar = _json_scalar(value)
    if scalar is not None:
        return scalar
    if isinstance(value, Mapping):
        return {
            "keys": sorted(_key(key) for key in value.keys()),
            "items": {
                _key(key): _summary_value(
                    child,
                    path=f"{path}.{_key(key)}",
                    captured=captured,
                    capture=capture,
                    max_capture_elements=max_capture_elements,
                )
                for key, child in value.items()
                if _key(key) in {"cond_frame_outputs", "non_cond_frame_outputs", "output_dict"}
            },
        }
    if isinstance(value, (list, tuple, set)):
        values = list(value)
        return {
            "length": len(values),
            "items": [
                _summary_value(
                    child,
                    path=f"{path}[{index}]",
                    captured=captured,
                    capture=capture,
                    max_capture_elements=max_capture_elements,
                )
                for index, child in enumerate(values[:32])
            ],
        }
    return {"type": type(value).__name__}


def _target_entry(container: Any, target_id: int, obj_index: Optional[int]) -> Any:
    if isinstance(container, Mapping):
        for key in (target_id, str(target_id)):
            if key in container:
                return container[key]
    if isinstance(container, (list, tuple)) and obj_index is not None and 0 <= obj_index < len(container):
        return container[obj_index]
    return None


def _frame_map_summary(
    container: Any,
    *,
    target_id: int,
    obj_index: Optional[int],
    focus_frame: int,
    path: str,
    captured: dict[str, torch.Tensor],
) -> dict[str, Any]:
    entry = _target_entry(container, target_id, obj_index)
    if entry is None:
        return {"present": False}
    if not isinstance(entry, Mapping):
        return {
            "present": True,
            "summary": _summary_value(
                entry,
                path=path,
                captured=captured,
                capture=True,
                max_capture_elements=3_000_000,
            ),
        }
    keys = sorted(_key(key) for key in entry.keys())
    focus_values: dict[str, Any] = {}
    for key, value in entry.items():
        try:
            is_focus = int(key) == int(focus_frame)
        except (TypeError, ValueError):
            is_focus = _key(key) == str(focus_frame)
        if is_focus:
            focus_values[_key(key)] = _summary_value(
                value,
                path=f"{path}.{_key(key)}",
                captured=captured,
                capture=True,
                max_capture_elements=3_000_000,
            )
    return {"present": True, "keys": keys, "focus_frame": focus_values}


def _frame_key_summary(
    container: Any,
    *,
    focus_frame: int,
    path: str,
    captured: dict[str, torch.Tensor],
) -> dict[str, Any]:
    """Summarize a frame-indexed mapping without treating the frame as an ID.

    SAM3 keeps some inputs as ``frame -> value`` mappings (for example
    ``per_frame_raw_box_input``), while tracker-local buffers are commonly
    ``object -> frame -> value``.  The two layouts need separate handling;
    using the object-oriented helper for both silently loses the correction
    frame from the audit.
    """

    if not isinstance(container, Mapping):
        return {
            "present": False,
            "container_type": type(container).__name__,
        }
    keys = sorted(_key(key) for key in container.keys())
    focus_values: dict[str, Any] = {}
    for key, value in container.items():
        try:
            is_focus = int(key) == int(focus_frame)
        except (TypeError, ValueError):
            is_focus = _key(key) == str(focus_frame)
        if is_focus:
            focus_values[_key(key)] = _summary_value(
                value,
                path=f"{path}.{_key(key)}",
                captured=captured,
                capture=True,
                max_capture_elements=3_000_000,
            )
    return {"present": True, "keys": keys, "focus_frame": focus_values}


def _obj_index(tracker_state: Mapping[str, Any], target_id: int) -> Optional[int]:
    values = tracker_state.get("obj_ids")
    if values is None:
        return None
    ids = np.asarray(values).reshape(-1).tolist()
    for index, value in enumerate(ids):
        if int(value) == int(target_id):
            return index
    return None


def _tracker_state_snapshot(
    tracker_state: Mapping[str, Any],
    *,
    target_id: int,
    correction_frame: int,
    path: str,
    captured: dict[str, torch.Tensor],
) -> dict[str, Any]:
    index = _obj_index(tracker_state, target_id)
    result: dict[str, Any] = {
        "keys": sorted(str(key) for key in tracker_state.keys()),
        "obj_ids": np.asarray(tracker_state.get("obj_ids", [])).reshape(-1).tolist(),
        "target_obj_index": index,
    }
    for key in ("obj_id_to_idx", "obj_idx_to_id", "first_ann_frame_idx", "frames_already_tracked"):
        value = tracker_state.get(key)
        if isinstance(value, Mapping):
            result[key] = {
                "keys": sorted(_key(item) for item in value.keys()),
                "target": value.get(target_id, value.get(str(target_id))) if key != "frames_already_tracked" else None,
            }
        elif isinstance(value, (list, tuple, set)):
            result[key] = {"keys": sorted(_key(item) for item in value)}
        else:
            scalar = _json_scalar(value)
            if scalar is not None:
                result[key] = scalar
    for key in ("mask_inputs_per_obj", "point_inputs_per_obj", "output_dict_per_obj", "temp_output_dict_per_obj"):
        if key in tracker_state:
            result[key] = _frame_map_summary(
                tracker_state[key],
                target_id=target_id,
                obj_index=index,
                focus_frame=correction_frame,
                path=f"{path}.{key}",
                captured=captured,
            )
    return result


def _metadata_snapshot(
    metadata: Mapping[str, Any],
    *,
    target_id: int,
    correction_frame: int,
    captured: dict[str, torch.Tensor],
) -> dict[str, Any]:
    result: dict[str, Any] = {"keys": sorted(str(key) for key in metadata.keys())}
    for key in ("obj_ids_all_gpu", "obj_ids_per_gpu", "num_obj_per_gpu", "max_obj_id"):
        if key in metadata:
            result[key] = _summary_value(
                metadata[key],
                path=f"official.tracker_metadata.{key}",
                captured=captured,
                capture=False,
                max_capture_elements=0,
            )
    rank0 = metadata.get("rank0_metadata")
    if isinstance(rank0, Mapping):
        result["rank0_metadata"] = {
            "keys": sorted(str(key) for key in rank0.keys()),
            "target": {
                key: _json_scalar(value.get(target_id, value.get(str(target_id))))
                if isinstance(value, Mapping)
                else _summary_value(
                    value,
                    path=f"official.tracker_metadata.rank0.{key}",
                    captured=captured,
                    capture=False,
                    max_capture_elements=0,
                )
                for key, value in rank0.items()
                if key in {"obj_first_frame_idx", "trk_keep_alive", "obj_id_to_score"}
            },
        }
    gpu = metadata.get("gpu_metadata")
    if isinstance(gpu, Mapping):
        result["gpu_metadata"] = {
            "keys": sorted(str(key) for key in gpu.keys()),
            "values": {
                str(key): _summary_value(
                    value,
                    path=f"official.tracker_metadata.gpu.{key}",
                    captured=captured,
                    capture=False,
                    max_capture_elements=0,
                )
                for key, value in gpu.items()
            },
        }
    return result


def snapshot_backend_state(
    backend: Any,
    *,
    target_id: int,
    correction_frame: int,
    b10_state: Optional[Mapping[str, Any]] = None,
    lora_state: Optional[Mapping[str, Any]] = None,
) -> StateAuditSnapshot:
    """Capture only state groups relevant to one correction target."""

    objects = backend._objects.get(int(target_id))
    object_payload = None
    if objects is not None:
        object_payload = {
            key: _summary_value(
                value,
                path=f"backend._objects[{target_id}].{key}",
                captured={},
                capture=False,
                max_capture_elements=0,
            )
            for key, value in objects.items()
        }
    cache_frames: dict[str, Any] = {}
    for frame, observations in backend._output_cache.items():
        target_obs = [obs for obs in observations if int(getattr(obs, "sam_object_id", -1)) == int(target_id)]
        if target_obs:
            cache_frames[str(frame)] = [
                {
                    "source": str(getattr(obs, "source", "")),
                    "is_human_verified": bool(getattr(obs, "is_human_verified", False)),
                    "box": np.asarray(getattr(obs, "box_xyxy", []), dtype=float).tolist(),
                    "mask_shape": list(np.asarray(getattr(obs, "mask", [])).shape),
                }
                for obs in target_obs
            ]

    captured: dict[str, torch.Tensor] = {}
    payload: dict[str, Any] = {
        "backend": {
            "target_object": object_payload,
            "last_prompt_frame": backend._last_prompt_frame,
            "ext_to_sam": {str(k): int(v) for k, v in backend._ext_to_sam.items()},
            "sam_to_ext": {str(k): int(v) for k, v in backend._sam_to_ext.items()},
            "output_cache_target_frames": cache_frames,
            "prompt_fallback_log_tail": list(backend._prompt_fallback_log[-8:]),
        },
        "official": {"session_present": False},
        "b10": dict(b10_state or {"update_called": False, "state_delta": "NOT_APPLICABLE_SINGLE_ID"}),
        "lora": dict(lora_state or {"update_called": False}),
    }
    predictor = getattr(backend, "_predictor", None)
    session_id = getattr(backend, "_session_id", None)
    states = getattr(predictor, "_all_inference_states", {}) if predictor is not None else {}
    entry = states.get(session_id) if session_id is not None else None
    state = entry.get("state") if isinstance(entry, Mapping) else None
    if isinstance(state, Mapping):
        tracker_states = state.get("sam2_inference_states", [])
        payload["official"] = {
            "session_present": True,
            "state_keys": sorted(str(key) for key in state.keys()),
            "sam2_inference_states": [
                _tracker_state_snapshot(
                    tracker_state,
                    target_id=target_id,
                    correction_frame=correction_frame,
                    path=f"official.sam2_inference_states[{index}]",
                    captured=captured,
                )
                for index, tracker_state in enumerate(tracker_states)
                if isinstance(tracker_state, Mapping)
            ],
            "per_frame_raw_box_input": _frame_key_summary(
                state.get("per_frame_raw_box_input", {}),
                focus_frame=correction_frame,
                path="official.per_frame_raw_box_input",
                captured=captured,
            ),
            "per_frame_visual_prompt": _frame_key_summary(
                state.get("per_frame_visual_prompt", {}),
                focus_frame=correction_frame,
                path="official.per_frame_visual_prompt",
                captured=captured,
            ),
            "action_history": [
                dict(item)
                for item in state.get("action_history", [])
                if isinstance(item, Mapping)
                and int(item.get("frame_idx", item.get("frame", -1))) == int(correction_frame)
            ],
        }
        metadata = state.get("tracker_metadata")
        if isinstance(metadata, Mapping):
            payload["official"]["tracker_metadata"] = _metadata_snapshot(
                metadata,
                target_id=target_id,
                correction_frame=correction_frame,
                captured=captured,
            )
    return StateAuditSnapshot(payload=payload, captured_tensors=captured)


def _diff_payload(left: Any, right: Any, path: str = "") -> list[str]:
    if type(left) is not type(right):
        return [path or "root"]
    if isinstance(left, Mapping):
        changed: list[str] = []
        for key in sorted(set(left) | set(right), key=str):
            child = f"{path}.{key}" if path else str(key)
            if key not in left or key not in right:
                changed.append(child)
            else:
                changed.extend(_diff_payload(left[key], right[key], child))
        return changed
    if isinstance(left, list):
        if left == right:
            return []
        return [path or "root"]
    if isinstance(left, float) and isinstance(right, float):
        return [] if left == right else [path or "root"]
    return [] if left == right else [path or "root"]


def diff_snapshots(before: StateAuditSnapshot, after: StateAuditSnapshot) -> dict[str, Any]:
    """Return changed paths and scalar norms for captured target tensors."""

    changed_paths = _diff_payload(before.payload, after.payload)
    tensor_deltas: dict[str, Any] = {}
    for path in sorted(set(before.captured_tensors) & set(after.captured_tensors)):
        left = before.captured_tensors[path]
        right = after.captured_tensors[path]
        if left.shape != right.shape:
            tensor_deltas[path] = {"shape_before": list(left.shape), "shape_after": list(right.shape)}
            continue
        delta = right - left
        if torch.equal(left, right):
            continue
        tensor_deltas[path] = {
            "shape": list(left.shape),
            "dtype": str(left.dtype),
            "delta_l2": float(torch.linalg.vector_norm(delta.reshape(-1))),
            "delta_linf": float(delta.abs().max()) if delta.numel() else 0.0,
        }
    groups = sorted({path.split(".", 1)[0] for path in changed_paths})
    return {
        "changed": bool(changed_paths or tensor_deltas),
        "changed_paths": changed_paths,
        "changed_state_groups": groups,
        "tensor_deltas": tensor_deltas,
    }
