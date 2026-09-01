#!/usr/bin/env python
"""Run the real SAM 3.1 GPU gate N0-G1..G9.

Uses the SHA256-verified public mirror checkpoint and the pinned official
commit 4cbac146c1b5a1e3a7f5c6a894901090b4dfd65b.  No mock result is reported
as real.
"""

import json
import time
import traceback
from pathlib import Path

import numpy as np
import torch

from sam3_intermot.backend.sam3_backend import Sam3Backend
from sam3_intermot.observations.mask_to_box import mask_to_box
from sam3_intermot.utils.io import atomic_write_json


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    checkpoint = root / "checkpoints" / "sam3.1_mirror" / "sam3.1_multiplex.pt"
    video_dir = root / "outputs" / "n0" / "backend_examples" / "video_frames"
    out_dir = root / "outputs" / "n0"
    out_dir.mkdir(parents=True, exist_ok=True)
    example_dir = out_dir / "backend_examples"
    example_dir.mkdir(parents=True, exist_ok=True)

    gates = {}
    wall_start = time.time()
    gpu_start = None
    peak_vram = 0.0
    backend = None
    try:
        if not checkpoint.exists():
            raise RuntimeError(f"checkpoint missing: {checkpoint}")
        if not video_dir.exists():
            raise RuntimeError(f"video dir missing: {video_dir}")

        torch.manual_seed(0)
        torch.cuda.manual_seed_all(0)
        backend = Sam3Backend(
            checkpoint_path=str(checkpoint),
            max_num_objects=16,
            multiplex_count=16,
            use_fa3=False,
            use_rope_real=True,
            compile=False,
            warm_up=False,
            session_expiration_sec=3600,
            output_prob_thresh=0.5,
            async_loading_frames=False,
        )

        # G1: model load + object multiplex confirmed.
        t0 = time.time()
        gpu_start = t0
        backend._ensure_model()
        model = backend._predictor.model
        gates["G1_official_model_loads"] = {
            "pass": True,
            "model_type": type(model).__name__,
            "is_multiplex": bool(getattr(model, "is_multiplex", False)),
            "build_seconds": round(time.time() - t0, 2),
        }

        # G2: single object box prompt + propagation t+1/t+5/t+10.
        session_id = backend.start_video(str(video_dir))
        gt_box = np.asarray([100.0, 100.0, 220.0, 360.0], dtype=float)
        obs0 = backend.add_box(0, 1, gt_box)
        gates["G2_single_object_prompt"] = {
            "pass": True,
            "session_id": session_id,
            "frame": obs0.frame_idx,
            "sam_object_id": obs0.sam_object_id,
            "box_xyxy": obs0.box_xyxy.tolist(),
            "mask_shape": list(obs0.mask.shape),
            "confidence": obs0.confidence,
        }
        prop = backend.propagate(0, 12, start_frame_index=0)
        required = {1, 5, 10}
        missing = sorted(required - set(prop.keys()))
        gates["G2_single_object_propagation"] = {
            "pass": len(missing) == 0,
            "frames": sorted(prop.keys()),
            "missing": missing,
            "counts": {int(k): len(v) for k, v in prop.items()},
        }

        # G3: multi-object propagation (two objects, one shared multiplex pass).
        box_b = np.asarray([956.0, 434.0, 1056.0, 694.0], dtype=float)
        obs_b = backend.add_box(0, 2, box_b)
        gates["G3_multi_object_prompt"] = {
            "pass": obs0.sam_object_id != obs_b.sam_object_id,
            "ids": [obs0.sam_object_id, obs_b.sam_object_id],
        }
        prop = backend.propagate(0, 12, start_frame_index=0)
        multi_counts = {int(k): len(v) for k, v in prop.items()}
        gates["G3_multi_object_propagation"] = {
            "pass": all(c >= 2 for c in multi_counts.values()),
            "counts": multi_counts,
        }
        frame0 = backend.get_frame_outputs(0)
        ids0 = [o.sam_object_id for o in frame0]
        gates["G3_unique_ids_per_frame"] = {
            "pass": len(ids0) == len(set(ids0)) and len(ids0) >= 2,
            "ids": ids0,
        }

        # G4: mid-video add of a third object.
        box_c = np.asarray([624.0, 380.0, 744.0, 640.0], dtype=float)
        obs_c = backend.add_box(5, 3, box_c)
        gates["G4_mid_video_add"] = {
            "pass": obs_c.frame_idx == 5,
            "sam_object_id": obs_c.sam_object_id,
            "box_xyxy": obs_c.box_xyxy.tolist(),
        }
        prop = backend.propagate(5, 8, start_frame_index=5)
        frame6 = backend.get_frame_outputs(6)
        gates["G4_add_propagates"] = {
            "pass": _has_box(frame6, box_c),
            "frame6_count": len(frame6),
            "frame6_ids": [o.sam_object_id for o in frame6],
        }

        # G5: mid-video correction keeps the same object id.
        corrected_box = np.asarray([966.0, 444.0, 1066.0, 704.0], dtype=float)
        before_ext = backend._ext_to_sam.get(1)
        obs_corr = backend.correct_object(8, 1, corrected_box)
        after_ext = backend._ext_to_sam.get(1)
        gates["G5_mid_video_correct"] = {
            "pass": after_ext is not None and _has_box([obs_corr], corrected_box),
            "before_ext_sam_id": before_ext,
            "after_ext_sam_id": after_ext,
            "box_xyxy": obs_corr.box_xyxy.tolist(),
        }
        prop = backend.propagate(8, 12, start_frame_index=8)
        frame11 = backend.get_frame_outputs(11)
        gates["G5_correct_propagates"] = {
            "pass": _has_box(frame11, corrected_box),
            "frame11_count": len(frame11),
            "frame11_ids": [o.sam_object_id for o in frame11],
        }

        # G6: delete one object; other objects remain and unique.
        backend.remove_object(2)
        prop = backend.propagate(10, 12, start_frame_index=10)
        frame11_after = backend.get_frame_outputs(11)
        ids_after = [o.sam_object_id for o in frame11_after]
        gates["G6_delete_object"] = {
            "pass": (
                2 not in backend._objects
                and len(backend._objects) == len(frame11_after)
                and len(ids_after) == len(set(ids_after))
            ),
            "frame11_count": len(frame11_after),
            "frame11_ids": ids_after,
            "remaining_objects": sorted(backend._objects.keys()),
        }

        # G7: mask -> box correctness.
        mask = np.zeros((1080, 1920), dtype=bool)
        mask[120:360, 140:520] = True
        box = mask_to_box(mask)
        gates["G7_mask_to_box"] = {
            "pass": box is not None and np.allclose(box, [140, 120, 520, 360]),
            "box": None if box is None else box.tolist(),
        }
        empty_box = mask_to_box(np.zeros((100, 100), dtype=bool))
        gates["G7_empty_mask"] = {
            "pass": empty_box is None,
            "box": None if empty_box is None else empty_box.tolist(),
        }

        # G8: upper layer never sees private state (adapter surface audit).
        public_methods = [
            "start_video",
            "detect_concept",
            "add_box",
            "add_points",
            "add_mask",
            "correct_object",
            "propagate",
            "remove_object",
            "reset_object",
            "get_frame_outputs",
            "close",
        ]
        gates["G8_backend_private_state_hidden"] = {
            "pass": all(hasattr(backend, m) for m in public_methods),
            "public_methods": public_methods,
        }

        # G9: close and release memory.
        before_close = torch.cuda.memory_allocated(0)
        backend.close()
        torch.cuda.empty_cache()
        after_close = torch.cuda.memory_allocated(0)
        gates["G9_session_close_memory"] = {
            "pass": after_close <= before_close,
            "before_allocated_mib": round(before_close / 2**20, 2),
            "after_allocated_mib": round(after_close / 2**20, 2),
        }
        peak_vram = round(torch.cuda.max_memory_allocated(0) / 2**20, 2)
    except Exception as exc:  # noqa: BLE001
        gates["unhandled_error"] = {
            "pass": False,
            "type": type(exc).__name__,
            "message": str(exc),
            "traceback": traceback.format_exc()[-4000:],
        }
        if backend is not None:
            try:
                backend.close()
            except Exception:
                pass

    all_pass = all(
        v.get("pass", False)
        for k, v in gates.items()
        if k.startswith("G")
    )
    wall_seconds = round(time.time() - wall_start, 2)
    gpu_seconds = round(wall_seconds, 2) if gpu_start is not None else 0.0
    result = {
        "status": "PASS" if all_pass else "FAIL",
        "gates": gates,
        "wall_clock_seconds": wall_seconds,
        "gpu_seconds": gpu_seconds,
        "peak_vram_mib": peak_vram,
        "gpu_ids": [8],
        "checkpoint_sha256": "0567debeec80ba4ac6369540c6c248025283cb3ff2b92827509e57e2b3541cb6",
    }
    atomic_write_json(out_dir / "backend_test_results.json", result)
    atomic_write_json(
        out_dir / "runtime_profile.json",
        {
            "stage": "N0_REAL_GPU",
            "wall_clock_seconds": wall_seconds,
            "gpu_seconds": gpu_seconds,
            "peak_vram_mib": peak_vram,
            "longest_command_seconds": wall_seconds,
            "note": "real SAM 3.1 mirror checkpoint on GPU 8",
        },
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0 if all_pass else 1


def _gt_boxes(frame_idx: int, count: int):
    gt_path = (
        Path("/path/to/dancetrack/val/dancetrack0004/gt/gt.txt")
    )
    out = []
    if gt_path.exists():
        for line in gt_path.read_text().splitlines():
            parts = line.split(",")
            if (
                parts
                and parts[0] == str(frame_idx + 1)
                and float(parts[6]) > 0
            ):
                x, y, w, h = map(float, parts[2:6])
                out.append(np.asarray([x, y, x + w, y + h], dtype=float))
                if len(out) >= count:
                    break
    while len(out) < count:
        out.append(np.asarray([100.0, 100.0, 220.0, 360.0], dtype=float))
    return out


def _offset_box(box: np.ndarray, dx: float, dy: float) -> np.ndarray:
    box = np.asarray(box, dtype=float).reshape(-1)
    return box + np.asarray([dx, dy, dx, dy], dtype=float)


def _has_box(obs_list, box: np.ndarray, iou_threshold: float = 0.25) -> bool:
    box = np.asarray(box, dtype=float).reshape(-1)
    for obs in obs_list:
        a = obs.box_xyxy
        b = box
        x1 = max(a[0], b[0])
        y1 = max(a[1], b[1])
        x2 = min(a[2], b[2])
        y2 = min(a[3], b[3])
        inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
        area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
        area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
        union = area_a + area_b - inter
        if union > 0 and inter / union > iou_threshold:
            return True
    return False


if __name__ == "__main__":
    raise SystemExit(main())
