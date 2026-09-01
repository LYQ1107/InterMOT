#!/usr/bin/env python
"""Probe the real SAM 3.1 API on GPU 8 (raw response structure)."""

import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import torch

from sam3_intermot.backend import sam3_compat  # noqa: F401  (torch 2.5 shim)


def main() -> int:
    project = Path(__file__).resolve().parents[1]
    checkpoint = (
        project
        / "checkpoints"
        / "sam3.1_mirror"
        / "sam3.1_multiplex.pt"
    )
    video_dir = project / "outputs" / "n0" / "backend_examples" / "video_frames"
    if not checkpoint.exists():
        print("CHECKPOINT_MISSING")
        return 1
    if not video_dir.exists():
        print("VIDEO_DIR_MISSING", video_dir)
        return 1

    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    t0 = time.time()
    from sam3.model_builder import build_sam3_multiplex_video_predictor

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        predictor = build_sam3_multiplex_video_predictor(
            checkpoint_path=str(checkpoint),
            max_num_objects=16,
            multiplex_count=16,
            use_fa3=False,
            use_rope_real=True,
            compile=False,
            warm_up=False,
            session_expiration_sec=3600,
            default_output_prob_thresh=0.5,
            async_loading_frames=False,
        )
    print(f"BUILD_OK seconds={time.time()-t0:.1f}")
    print("WARNINGS", len(caught))
    for w in caught[:10]:
        print("WARNING", str(w.message)[:300])

    model = predictor.model
    print("PREDICTOR_TYPE", type(predictor).__name__)
    print("MODEL_TYPE", type(model).__name__)
    for attr in ("is_multiplex", "is_multiplex_dynamic", "per_obj_inference", "multiplex_count"):
        if hasattr(model, attr):
            print("ATTR", attr, getattr(model, attr))

    # Direct init_state workaround (see Sam3Backend.start_video docstring).
    inference_state = predictor.model.init_state(
        resource_path=str(video_dir),
        offload_video_to_cpu=True,
        async_loading_frames=False,
    )
    import time as _time
    import uuid

    session_id = str(uuid.uuid4())
    predictor._all_inference_states[session_id] = {
        "state": inference_state,
        "session_id": session_id,
        "start_time": _time.time(),
        "last_use_time": _time.time(),
    }
    print("SESSION", session_id)

    # GT-derived person box on frame 0 (dancetrack0004).
    gt_path = (
        Path("/path/to/dancetrack/val/dancetrack0004/gt/gt.txt")
    )
    box_xyxy = [100, 100, 220, 360]
    if gt_path.exists():
        for line in gt_path.read_text().splitlines()[:50]:
            parts = line.split(",")
            if parts and parts[0] == "1" and float(parts[6]) > 0:
                x, y, w, h = map(float, parts[2:6])
                box_xyxy = [x, y, x + w, y + h]
                break
    img = np.asarray(
        __import__("PIL").Image.open(sorted(video_dir.glob("*.jpg"))[0])
    )
    H, W = img.shape[:2]
    x, y, w, h = box_xyxy[0], box_xyxy[1], box_xyxy[2] - box_xyxy[0], box_xyxy[3] - box_xyxy[1]
    rel = [x / W, y / H, w / W, h / H]
    print("BOX_XYXY", box_xyxy, "IMAGE", W, H, "REL", rel)

    resp = predictor.handle_request(
        request=dict(
            type="add_prompt",
            session_id=session_id,
            frame_index=0,
            bounding_boxes=[rel],
            bounding_box_labels=[1],
            clear_old_boxes=True,
            obj_id=1,
        )
    )
    print("ADD_RESPONSE_KEYS", list(resp.keys()))
    print("ADD_FRAME", resp.get("frame_index"))
    out = resp.get("outputs")
    print("ADD_OUTPUT_TYPE", type(out).__name__)
    if isinstance(out, list):
        print("ADD_OUTPUT_LEN", len(out))
        if out:
            print("ADD_ITEM_TYPE", type(out[0]).__name__)
            if isinstance(out[0], dict):
                print("ADD_ITEM_KEYS", list(out[0].keys()))
                for k, v in out[0].items():
                    if hasattr(v, "shape"):
                        print("  ", k, "shape", tuple(v.shape), "dtype", v.dtype)
                    else:
                        print("  ", k, type(v).__name__, str(v)[:200])
            else:
                print("ADD_ITEM", str(out[0])[:500])
    elif isinstance(out, dict):
        print("ADD_OUTPUT_DICT_KEYS", list(out.keys()))
        for k, v in out.items():
            if hasattr(v, "shape"):
                print("  ", k, "shape", tuple(v.shape), "dtype", v.dtype)
            else:
                print("  ", k, type(v).__name__, str(v)[:300])

    istate = predictor._all_inference_states[session_id]["state"]
    print("STATE_KEYS", sorted(istate.keys()))
    print("SAM2_STATES", len(istate.get("sam2_inference_states", [])))
    for si, st in enumerate(istate.get("sam2_inference_states", [])):
        print("SAM2_STATE", si, "keys", sorted(st.keys()))
        print("  obj_ids", st.get("obj_ids"))
        ms = st.get("multiplex_state")
        if ms is not None:
            print(
                "  multiplex_state num_buckets",
                ms.num_buckets,
                "total_valid",
                ms.total_valid_entries,
                "assignments",
                ms.assignments,
            )
        else:
            print("  multiplex_state None")
    tm = istate.get("tracker_metadata", {})
    print("TRACKER_META_KEYS", sorted(tm.keys()))
    print("TRACKER_OBJ_IDS", tm.get("obj_ids_all_gpu"))
    print("TRACKER_NUM_OBJ", tm.get("num_obj_per_gpu"))
    fc = istate.get("feature_cache", {})
    print("FEATURE_CACHE_KEYS", sorted(fc.keys(), key=str)[:30])
    for k in sorted(fc.keys(), key=str):
        if isinstance(k, int):
            _img, _bc = fc[k]
            for neck in ("interactive", "sam2_backbone_out"):
                if neck in _bc:
                    fpn = _bc[neck].get("backbone_fpn", [])
                    print(
                        "  FEATURE",
                        k,
                        neck,
                        "levels",
                        len(fpn),
                        "last_shape",
                        tuple(fpn[-1].tensors.shape) if fpn and hasattr(fpn[-1], "tensors") else None,
                    )
    for si, st in enumerate(istate.get("sam2_inference_states", [])):
        cf = st.get("cached_features", {})
        print("SAM2_CACHED_KEYS", sorted(cf.keys(), key=str)[:30])
        for k in sorted(cf.keys(), key=str):
            if isinstance(k, int):
                _img, _bc = cf[k]
                for neck in ("interactive", "sam2_backbone_out"):
                    if neck in _bc:
                        fpn = _bc[neck].get("backbone_fpn", [])
                        print(
                            "  SAM2_FEATURE",
                            k,
                            neck,
                            "levels",
                            len(fpn),
                            "last_shape",
                            tuple(fpn[-1].tensors.shape) if fpn and hasattr(fpn[-1], "tensors") else None,
                        )

    prop = []
    try:
        for response in predictor.handle_stream_request(
            request=dict(
                type="propagate_in_video",
                session_id=session_id,
                propagation_direction="forward",
                start_frame_index=0,
            )
        ):
            prop.append(response)
    except Exception as exc:
        print("PROPAGATE_ERROR", type(exc).__name__, str(exc)[:300])
        fc = istate.get("feature_cache", {})
        print("AFTER_ERROR_FEATURE_KEYS", sorted(fc.keys(), key=str))
        for k in sorted(fc.keys(), key=str):
            if isinstance(k, int):
                _img, _bc = fc[k]
                for neck in ("interactive", "sam2_backbone_out"):
                    if neck in _bc:
                        fpn = _bc[neck].get("backbone_fpn", [])
                        print(
                            "  AFTER_FEATURE",
                            k,
                            neck,
                            [tuple(x.tensors.shape) for x in fpn],
                        )
        for si, st in enumerate(istate.get("sam2_inference_states", [])):
            cf = st.get("cached_features", {})
            for k in sorted(cf.keys(), key=str):
                if isinstance(k, int):
                    _img, _bc = cf[k]
                    for neck in ("interactive", "sam2_backbone_out"):
                        if neck in _bc:
                            fpn = _bc[neck].get("backbone_fpn", [])
                            print(
                                "  AFTER_SAM2_FEATURE",
                                si,
                                k,
                                neck,
                                [tuple(x.tensors.shape) for x in fpn],
                            )
        raise
    print("PROPAGATE_RESPONSES", len(prop))
    for response in prop[:2]:
        print("PROP_FRAME", response.get("frame_index"))
        po = response.get("outputs")
        print("PROP_OUTPUT_TYPE", type(po).__name__)
        if isinstance(po, list) and po:
            item = po[0]
            if isinstance(item, dict):
                print("PROP_ITEM_KEYS", list(item.keys()))
                for k, v in item.items():
                    if hasattr(v, "shape"):
                        print("  ", k, "shape", tuple(v.shape), "dtype", v.dtype)
                    else:
                        print("  ", k, type(v).__name__, str(v)[:200])
            else:
                print("PROP_ITEM", str(item)[:500])

    predictor.handle_request(
        request=dict(type="close_session", session_id=session_id)
    )
    print("PROBE_DONE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
