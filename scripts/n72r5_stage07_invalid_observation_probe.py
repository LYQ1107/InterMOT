"""Read-only probe for invalid official SAM3 observations in N72R5 Stage07.

This probe does not change the Stage07 runner or scientific protocol.  It runs
one fresh official baseline stream, records every empty-mask or non-positive
box observation, and closes the session before writing one atomic diagnostic
artifact.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np


RUNNER = Path(__file__).resolve().with_name("n72r5_stage07_official_full_loop.py")


def load_runner():
    spec = importlib.util.spec_from_file_location("n72r5_stage07_runner_probe", RUNNER)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import runner: {RUNNER}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--event-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    runner = load_runner()
    event = runner.load_event(args.event_id)
    sequence_dir = runner.DATA_ROOT / "train" / str(event["sequence"])
    backend = None
    try:
        backend = runner.make_backend_n72r5()
        backend.start_video(str(sequence_dir / "img1"))
        runner.install_official_shape_audit(backend)
        runner.enable_official_batched_grounding_batch1(backend)
        runner.enable_b0_official_memory_trim(backend)
        window = runner.event_window(event)
        _pre, outputs = runner.collect_continuous_baseline_n72r5(
            backend,
            window,
            int(event["event_frame"]),
            int(event["event_frame"]) + runner.HORIZON,
        )
        invalid = []
        for frame_idx, observations in sorted(outputs.items()):
            for index, observation in enumerate(observations):
                box = np.asarray(observation.box_xyxy, dtype=np.float64).reshape(-1)
                mask = np.asarray(observation.mask)
                nonzero = np.argwhere(mask.astype(bool))
                positive = bool(
                    box.size == 4
                    and np.all(np.isfinite(box))
                    and box[2] > box[0]
                    and box[3] > box[1]
                )
                if positive and nonzero.size:
                    continue
                invalid.append(
                    {
                        "frame_idx": int(frame_idx),
                        "observation_index": int(index),
                        "sam_object_id": int(observation.sam_object_id),
                        "raw_sam_object_id": (
                            None
                            if observation.raw_sam_object_id is None
                            else int(observation.raw_sam_object_id)
                        ),
                        "box_xyxy": box.tolist(),
                        "box_positive_finite": positive,
                        "confidence": float(observation.confidence),
                        "mask_shape": list(mask.shape),
                        "mask_nonzero_count": int(nonzero.shape[0]),
                        "mask_y_min": int(nonzero[:, 0].min()) if nonzero.size else None,
                        "mask_y_max": int(nonzero[:, 0].max()) if nonzero.size else None,
                        "mask_x_min": int(nonzero[:, 1].min()) if nonzero.size else None,
                        "mask_x_max": int(nonzero[:, 1].max()) if nonzero.size else None,
                    }
                )
        payload = {
            "schema_version": "N72R5_STAGE07_INVALID_OBSERVATION_PROBE_V1",
            "status": "PASS_INVALID_OBSERVATION_PROFILED",
            "stage": "07_OFFICIAL_FULL_LOOP",
            "event_id": str(event["event_id"]),
            "sequence": str(event["sequence"]),
            "event_frame": int(event["event_frame"]),
            "future_end": int(event["event_frame"]) + runner.HORIZON,
            "frames_observed": len(outputs),
            "observation_count": sum(len(values) for values in outputs.values()),
            "invalid_observation_count": len(invalid),
            "invalid_observations": invalid,
            "official_shape_audit_count": len(
                getattr(backend._predictor.model, "_n72r4_shape_audit", [])
            ),
            "runtime_future_gt_used": False,
            "scientific_result": "NO_SCIENTIFIC_RESULT",
        }
        runner.atomic_json(args.output, payload)
        print(json.dumps(payload, sort_keys=True))
        return 0
    finally:
        if backend is not None:
            try:
                backend.close()
            except Exception:
                pass


if __name__ == "__main__":
    raise SystemExit(main())
