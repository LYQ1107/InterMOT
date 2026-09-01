"""Stage-A audit of causal SAM3 features and tracker state.

This probe deliberately inspects the pinned official SAM3.1 multiplex adapter
after a real text-prompt + propagation run.  It records tensor provenance,
shape, dtype/device, finite values and per-object binding without copying large
feature maps into the report.  It does not use GT and does not modify the
third-party SAM3 source.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = Path("/path/to/dancetrack")
sys.path.insert(0, str(ROOT))

from sam3_intermot.backend.sam3_backend import Sam3Backend


def _tensor_summary(value: Any) -> dict[str, Any]:
    out: dict[str, Any] = {"python_type": type(value).__name__}
    if value is None:
        return out
    if hasattr(value, "shape"):
        try:
            out["shape"] = [int(x) for x in value.shape]
        except Exception:
            out["shape"] = str(value.shape)
    if hasattr(value, "dtype"):
        out["dtype"] = str(value.dtype)
    if hasattr(value, "device"):
        out["device"] = str(value.device)
    if hasattr(value, "detach"):
        try:
            t = value.detach()
            flat = t.float().reshape(-1)
            out["numel"] = int(flat.numel())
            if flat.numel():
                finite = bool(torch_isfinite(flat).all().item())
                out["finite"] = finite
                out["mean"] = float(flat.mean().item())
                out["std"] = float(flat.std(unbiased=False).item())
                out["norm"] = float(flat.norm().item())
                out["min"] = float(flat.min().item())
                out["max"] = float(flat.max().item())
        except Exception as exc:  # pragma: no cover - backend-specific tensors
            out["summary_error"] = repr(exc)
    elif isinstance(value, np.ndarray):
        out["numel"] = int(value.size)
        if value.size:
            finite = np.isfinite(value.astype(np.float32, copy=False))
            out["finite"] = bool(finite.all())
            vals = value.astype(np.float32, copy=False)
            out["mean"] = float(vals.mean())
            out["std"] = float(vals.std())
            out["norm"] = float(np.linalg.norm(vals.reshape(-1)))
    return out


def torch_isfinite(value):
    # Import lazily so the script can still be byte-compiled outside CUDA.
    import torch

    return torch.isfinite(value)


def _value_shape(value: Any) -> Any:
    if value is None:
        return None
    if hasattr(value, "shape"):
        try:
            return [int(x) for x in value.shape]
        except Exception:
            return str(value.shape)
    if isinstance(value, (list, tuple)):
        return {"length": len(value), "items": [_value_shape(x) for x in value[:4]]}
    if isinstance(value, dict):
        return {"keys": list(value)[:30], "length": len(value)}
    return type(value).__name__


def _state_records(state: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    records: list[dict[str, Any]] = []
    state_meta: dict[str, Any] = {
        "top_level_keys": sorted(str(k) for k in state.keys()),
        "sam2_state_count": len(state.get("sam2_inference_states", [])),
        "tracker_state_keys": [],
    }
    for si, tracker_state in enumerate(state.get("sam2_inference_states", [])):
        multiplex_state = tracker_state.get("multiplex_state")
        state_meta["tracker_state_keys"].append(
            sorted(str(k) for k in tracker_state.keys())
        )
        state_meta.setdefault("object_bindings", []).append(
            {
                "state_index": si,
                "obj_ids": [int(x) for x in tracker_state.get("obj_ids", [])],
                "obj_id_to_idx": {
                    str(k): int(v)
                    for k, v in tracker_state.get("obj_id_to_idx", {}).items()
                },
                "num_frames": tracker_state.get("num_frames"),
            }
        )
        if multiplex_state is not None:
            state_meta.setdefault("multiplex", []).append(
                {
                    "state_index": si,
                    "assignments": [
                        [int(x) for x in bucket]
                        for bucket in multiplex_state.assignments
                    ],
                    "object_ids": (
                        [int(x) for x in multiplex_state.object_ids]
                        if multiplex_state.object_ids is not None
                        else None
                    ),
                    "num_buckets": int(multiplex_state.num_buckets),
                    "multiplex_count": int(multiplex_state.multiplex_count),
                    "total_valid_entries": int(multiplex_state.total_valid_entries),
                }
            )
        output_dict = tracker_state.get("output_dict", {})
        for storage_key in ("cond_frame_outputs", "non_cond_frame_outputs"):
            for frame_idx, frame_out in output_dict.get(storage_key, {}).items():
                key_summaries = {}
                for key in (
                    "pred_masks",
                    "object_score_logits",
                    "obj_ptr",
                    "maskmem_features",
                    "maskmem_pos_enc",
                    "image_features",
                    "image_pos_enc",
                    "eff_iou_score",
                ):
                    if key in frame_out:
                        key_summaries[key] = _tensor_summary(frame_out[key])
                pointer_binding = None
                if "obj_ptr" in frame_out and multiplex_state is not None:
                    try:
                        data_ptr = multiplex_state.demux(frame_out["obj_ptr"])
                        pointer_binding = {
                            "space": "demuxed_data",
                            "shape": [int(x) for x in data_ptr.shape],
                            "object_indices": [
                                int(x) for x in range(int(data_ptr.shape[0]))
                            ],
                            "per_object_norm": [
                                float(x)
                                for x in data_ptr.float()
                                .reshape(data_ptr.shape[0], -1)
                                .norm(dim=1)
                                .detach()
                                .cpu()
                                .tolist()
                            ],
                        }
                    except Exception as exc:
                        pointer_binding = {"error": repr(exc)}
                local_mapping = frame_out.get("local_obj_id_to_idx")
                if isinstance(local_mapping, dict):
                    local_mapping = {
                        str(k): int(v) for k, v in local_mapping.items()
                    }
                conditioning = frame_out.get("conditioning_objects")
                if isinstance(conditioning, set):
                    conditioning = sorted(int(x) for x in conditioning)
                records.append(
                    {
                        "state_index": si,
                        "storage": storage_key,
                        "frame": int(frame_idx),
                        "frame_keys": sorted(str(k) for k in frame_out.keys()),
                        "local_obj_id_to_idx": local_mapping,
                        "conditioning_objects": conditioning,
                        "features": key_summaries,
                        "pointer_binding": pointer_binding,
                    }
                )

    # Some pinned paths retain per-object dictionaries in addition to the
    # consolidated state.  Record their schema separately without duplicating
    # every tensor in the main report.
    per_obj = state.get("sam2_inference_states", [])
    for si, tracker_state in enumerate(per_obj):
        for dict_name in ("output_dict_per_obj", "temp_output_dict_per_obj"):
            d = tracker_state.get(dict_name, {})
            state_meta.setdefault("per_object_output_schema", []).append(
                {
                    "state_index": si,
                    "name": dict_name,
                    "object_keys": [str(k) for k in d.keys()],
                    "nested_keys": {
                        str(k): sorted(str(x) for x in v.keys())
                        for k, v in list(d.items())[:8]
                        if isinstance(v, dict)
                    },
                }
            )
    return records, state_meta


def _model_meta(backend: Sam3Backend) -> dict[str, Any]:
    model = backend._predictor.model
    tracker = getattr(model, "tracker", None)
    meta = {
        "model_class": type(model).__name__,
        "predictor_class": type(backend._predictor).__name__,
        "tracker_class": type(tracker).__name__ if tracker is not None else None,
    }
    for name in (
        "use_obj_ptrs_in_encoder",
        "max_obj_ptrs_in_encoder",
        "num_maskmem",
        "mem_dim",
        "hidden_dim",
        "save_image_features",
        "forward_backbone_per_frame_for_eval",
        "offload_output_to_cpu_for_eval",
        "trim_past_non_cond_mem_for_eval",
        "use_maskmem_tpos_v2",
    ):
        if tracker is not None and hasattr(tracker, name):
            value = getattr(tracker, name)
        elif hasattr(model, name):
            value = getattr(model, name)
        else:
            continue
        if isinstance(value, (str, int, float, bool)) or value is None:
            meta[name] = value
        else:
            meta[name] = repr(value)
    return meta


def audit_sequence(backend: Sam3Backend, sequence: str, frames: int) -> dict[str, Any]:
    video = DATA_ROOT / "train" / sequence / "img1"
    if not video.is_dir():
        raise FileNotFoundError(video)
    backend.start_video(str(video))
    try:
        detected = backend.detect_concept(0, "person")
        propagated = backend.propagate(
            0, max(0, frames - 1), start_frame_index=0, keep_masks=False
        )
        session = backend._predictor._all_inference_states[backend._session_id]
        state = session["state"]
        records, state_meta = _state_records(state)
        feature_cache = state.get("feature_cache")
        cache_meta = {
            "type": type(feature_cache).__name__,
            "shape": _value_shape(feature_cache),
        }
        return {
            "sequence": sequence,
            "video": str(video),
            "requested_frames": frames,
            "detected_objects": len(detected),
            "propagated_frames": len(propagated),
            "propagated_object_rows": sum(len(v) for v in propagated.values()),
            "prompt_fallbacks": list(backend._prompt_fallback_log),
            "state": state_meta,
            "feature_cache": cache_meta,
            "frame_records": records,
        }
    finally:
        backend.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seqs", nargs="+", default=["dancetrack0074", "dancetrack0096"])
    parser.add_argument("--frames", type=int, default=60)
    parser.add_argument("--out-dir", default="outputs/n25/feature_audit")
    args = parser.parse_args()

    cfg = yaml.safe_load((ROOT / "configs" / "default.yaml").read_text())
    out_dir = ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    backend = Sam3Backend(
        checkpoint_path=cfg["backend"]["checkpoint_path"],
        max_num_objects=cfg["backend"]["max_num_objects"],
        multiplex_count=cfg["backend"]["multiplex_count"],
        use_fa3=False,
        use_rope_real=True,
        compile=False,
        warm_up=False,
        async_loading_frames=False,
    )
    report: dict[str, Any] = {
        "protocol": {
            "gt_used": False,
            "future_frames_used_for_decision": False,
            "prompt": "person",
            "frames": args.frames,
            "checkpoint": cfg["backend"]["checkpoint_path"],
            "source_commit": "4cbac146c1b5a1e3a7f5c6a894901090b4dfd65b",
        },
        "model": {},
        "sequences": [],
    }
    try:
        backend._ensure_model()
        report["model"] = _model_meta(backend)
        for sequence in args.seqs:
            report["sequences"].append(audit_sequence(backend, sequence, args.frames))
    finally:
        backend.close()
    path = out_dir / "n25_sam3_feature_audit.json"
    path.write_text(json.dumps(report, indent=2, default=str) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, default=str)[:16000], flush=True)
    print(f"N25_FEATURE_AUDIT_DONE {path}", flush=True)


if __name__ == "__main__":
    main()
