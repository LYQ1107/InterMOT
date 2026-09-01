#!/usr/bin/env python
"""N1.5 checkpoint load completeness audit."""

import json
import time
from pathlib import Path

import torch

from sam3_intermot.backend.sam3_backend import Sam3Backend
from sam3_intermot.utils.io import atomic_write_json


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    checkpoint = root / "checkpoints" / "sam3.1_mirror" / "sam3.1_multiplex.pt"
    out_dir = root / "outputs" / "n1_5"
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    backend = Sam3Backend(
        checkpoint_path=str(checkpoint),
        max_num_objects=16,
        multiplex_count=16,
        use_fa3=False,
        use_rope_real=True,
        compile=False,
        warm_up=False,
        async_loading_frames=False,
    )
    backend._ensure_model()
    model = backend._predictor.model
    model_state = model.state_dict()

    raw = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if isinstance(raw, dict) and "model" in raw and isinstance(raw["model"], dict):
        raw = raw["model"]

    model_keys = set(model_state.keys())
    ckpt_keys = set(raw.keys())
    missing = sorted(model_keys - ckpt_keys)
    unexpected = sorted(ckpt_keys - model_keys)
    shape_mismatch = []
    for k in sorted(model_keys & ckpt_keys):
        a = tuple(model_state[k].shape)
        b = tuple(raw[k].shape)
        if a != b:
            shape_mismatch.append({"key": k, "model_shape": a, "ckpt_shape": b})

    missing_param = []
    missing_buffer = []
    other = []
    for k in missing:
        try:
            model.get_parameter(k)
            missing_param.append(k)
            continue
        except Exception:
            pass
        try:
            model.get_buffer(k)
            missing_buffer.append(k)
            continue
        except Exception:
            pass
        other.append(k)

    result = {
        "status": "PASS" if (not missing_param and not unexpected and not shape_mismatch) else "FAIL",
        "missing_total": len(missing),
        "missing_trainable_parameters": len(missing_param),
        "missing_non_trainable_buffers": len(missing_buffer),
        "missing_other": len(other),
        "missing_parameter_keys": missing_param,
        "missing_buffer_keys": missing_buffer,
        "missing_other_keys": other,
        "unexpected_keys": len(unexpected),
        "unexpected_key_samples": unexpected[:20],
        "shape_mismatches": shape_mismatch,
        "model_type": type(model).__name__,
        "is_multiplex": bool(getattr(model, "is_multiplex", False)),
        "checkpoint_sha256": "0567debeec80ba4ac6369540c6c248025283cb3ff2b92827509e57e2b3541cb6",
        "code_commit": "4cbac146c1b5a1e3a7f5c6a894901090b4dfd65b",
        "wall_clock_seconds": round(time.time() - t0, 2),
    }
    atomic_write_json(out_dir / "checkpoint_key_audit.json", result)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    backend.close()
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
