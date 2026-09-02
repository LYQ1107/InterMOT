#!/usr/bin/env python3
"""Read-only probe of the official SAM3 state immediately after Y_pre.

This probe intentionally stops before the event action and before any future
frame.  It records shape/key invariants needed to diagnose a split official
propagation failure; it is not a candidate tape or scientific result.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import traceback
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.n72r4_official_future_pair import (  # noqa: E402
    EVENT_MANIFEST,
    load_event,
    load_window,
    make_backend,
    collect_pre_event,
)


def atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        directory_fd = os.open(path.parent, os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def shape_of(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return {"type": "ndarray", "shape": [int(v) for v in value.shape], "dtype": str(value.dtype)}
    if hasattr(value, "shape") and hasattr(value, "dtype"):
        try:
            return {"type": "tensor", "shape": [int(v) for v in value.shape], "dtype": str(value.dtype), "device": str(value.device)}
        except Exception:
            return {"type": type(value).__name__}
    if isinstance(value, dict):
        return {str(key): shape_of(child) for key, child in value.items()}
    if isinstance(value, (list, tuple)):
        return {"type": type(value).__name__, "length": len(value), "items": [shape_of(child) for child in value[:8]]}
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return {"type": type(value).__name__}


def object_key_summary(value: Any) -> dict[str, Any]:
    """Describe object/dict structure without retaining tensor payloads.

    This probe is intentionally diagnostic-only.  The official multiplex
    state stores several nested dataclass-like objects, so a plain JSON dump
    of the top-level state hides the fields needed to explain a mask/id shape
    mismatch.  Record names and small scalar/list metadata, never the actual
    tensors or masks.
    """
    if isinstance(value, dict):
        return {"type": "dict", "keys": sorted(str(key) for key in value.keys())}
    attrs = getattr(value, "__dict__", None)
    if isinstance(attrs, dict):
        return {
            "type": type(value).__name__,
            "attrs": sorted(str(key) for key in attrs.keys()),
        }
    return {"type": type(value).__name__}


def id_list(value: Any) -> list[int]:
    if value is None:
        return []
    try:
        return [int(item) for item in value]
    except (TypeError, ValueError):
        return []


def state_summary(backend: Any, event_frame: int) -> dict[str, Any]:
    predictor = backend._predictor
    entry = predictor._all_inference_states.get(backend._session_id, {})
    state = entry.get("state", {}) if isinstance(entry, dict) else {}
    metadata = state.get("tracker_metadata", {})
    sam_states = state.get("sam2_inference_states", [])
    tracker_states = []
    for item in sam_states:
        tracker_states.append(
            {
                "obj_ids": [int(v) for v in item.get("obj_ids", [])],
                "obj_id_count": len(item.get("obj_ids", [])),
                "multiplex_state": shape_of(item.get("multiplex_state")),
            }
        )
    cache = state.get("cached_frame_outputs", {})
    cache_event = cache.get(int(event_frame), cache.get(str(event_frame), {}))
    rank0 = metadata.get("rank0_metadata", {}) if isinstance(metadata, dict) else {}
    suppressed = rank0.get("suppressed_obj_ids", {}) if isinstance(rank0, dict) else {}
    suppressed_at_event = suppressed.get(int(event_frame), suppressed.get(str(event_frame), []))
    gpu_metadata = metadata.get("gpu_metadata", {}) if isinstance(metadata, dict) else {}
    gpu_shapes = {}
    if isinstance(gpu_metadata, dict):
        for key, value in gpu_metadata.items():
            if isinstance(value, (dict, list, tuple)):
                gpu_shapes[str(key)] = shape_of(value)
            else:
                gpu_shapes[str(key)] = shape_of(value)
    return {
        "session_id": str(backend._session_id),
        "backend_objects": sorted(int(key) for key in getattr(backend, "_objects", {}).keys()),
        "backend_ext_to_sam": {str(key): int(value) for key, value in getattr(backend, "_ext_to_sam", {}).items()},
        "backend_sam_to_ext": {str(key): int(value) for key, value in getattr(backend, "_sam_to_ext", {}).items()},
        "tracker_metadata": {
            "obj_ids_all_gpu": [int(v) for v in metadata.get("obj_ids_all_gpu", [])],
            "obj_ids_per_gpu": [[int(v) for v in values] for values in metadata.get("obj_ids_per_gpu", [])],
            "num_obj_per_gpu": [int(v) for v in metadata.get("num_obj_per_gpu", [])],
            "max_obj_id": int(metadata.get("max_obj_id", -1)),
            "removed_obj_ids": id_list(rank0.get("removed_obj_ids", []))
            if isinstance(rank0, dict)
            else [],
            "suppressed_obj_ids_at_event": id_list(suppressed_at_event),
            "gpu_metadata": gpu_shapes,
        },
        "sam2_inference_states": tracker_states,
        "sam2_inference_state_structure": [object_key_summary(item) for item in sam_states],
        "cached_frame_outputs": {
            "key_count": len(cache),
            "min_key": None if not cache else int(min(int(key) for key in cache)),
            "max_key": None if not cache else int(max(int(key) for key in cache)),
            "event_frame_object_ids": sorted(int(key) for key in cache_event.keys()) if isinstance(cache_event, dict) else [],
            "event_frame_shape": shape_of(cache_event),
        },
        "action_history": shape_of(state.get("action_history", [])),
        "previous_stage_event": state.get("previous_stages_out", [None] * (event_frame + 1))[event_frame],
        "feature_cache_keys": sorted(str(key) for key in state.get("feature_cache", {}).keys()),
        "tracking_bounds": shape_of(state.get("feature_cache", {}).get("tracking_bounds", {})),
        "runtime_memory_policy": backend.runtime_memory_policy(),
        "runtime_future_gt_used": False,
        "scientific_result": "STATE_SHAPE_PROBE_ONLY_NO_FUTURE_RESULT",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--event-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--reconcile",
        action="store_true",
        help="also run the adapter-level visible-output reconciliation and snapshot the repaired state",
    )
    args = parser.parse_args()
    backend = None
    event = load_event(args.event_id)
    frame = int(event["event_frame"])
    window = load_window(event)
    try:
        backend = make_backend()
        backend.start_video(str(Path("/data1/LWR/vranlee/DATASETS/JDE/dancetrack") / "train" / str(event["sequence"]) / "img1"))
        pre = collect_pre_event(backend, window, frame)
        reconciliation = None
        repaired_state = None
        if args.reconcile:
            # Freeze the same Y_pre view used by the official future worker
            # before touching the adapter repair.  This probe never runs a
            # future frame and never reads GT.
            reconciliation = backend.reconcile_official_tracker_to_visible_outputs(frame)
            repaired_state = state_summary(backend, frame)
        payload = {
            "schema_version": "N72R4_OFFICIAL_STATE_PROBE_V1",
            "status": "PASS_STATE_PROBE_BEFORE_EVENT_ACTION",
            "event_id": str(args.event_id),
            "event_frame": frame,
            "y_pre_count": len(pre),
            "y_pre_ids": [int(obs.sam_object_id) for obs in pre],
            "state": state_summary(backend, frame),
            "reconciliation": reconciliation,
            "repaired_state": repaired_state,
            "runtime_future_gt_used": False,
        }
        atomic_json(args.output.resolve(), payload)
        print(json.dumps({"status": payload["status"], "output": str(args.output.resolve())}, sort_keys=True))
        return 0
    except Exception as exc:
        failure = args.output.resolve().with_suffix(".failure.json")
        atomic_json(
            failure,
            {
                "schema_version": "N72R4_OFFICIAL_STATE_PROBE_FAILURE_V1",
                "status": "FAIL_PRESERVED",
                "event_id": str(args.event_id),
                "event_frame": frame,
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(),
                "runtime_future_gt_used": False,
                "scientific_result": "NO_SCIENTIFIC_RESULT",
            },
        )
        print(json.dumps({"status": "FAIL_STATE_PROBE", "failure": str(failure)}, sort_keys=True))
        return 1
    finally:
        if backend is not None:
            try:
                backend.close()
            except Exception:
                pass


if __name__ == "__main__":
    raise SystemExit(main())
