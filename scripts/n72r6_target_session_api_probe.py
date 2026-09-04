#!/usr/bin/env python3
"""Targeted N72R6 API probe after a target-session candidate-absence failure.

This is an engineering diagnostic, not a target-stream experiment.  It uses
the same frozen event/frame/box/checkpoint and compares the adapter's
box-only request with the official text+box request.  No result from this
probe is promoted to a candidate stream or scientific PASS.
"""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import sys
import tempfile
import traceback
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sam3_intermot.backend.sam3_backend import Sam3Backend  # noqa: E402
from scripts.n72r6_target_correction_stream import (  # noqa: E402
    CHECKPOINT,
    DATA_ROOT,
    _main_y_pre,
    eligible_event,
    image_files,
)


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    try:
        with open(fd, "w", encoding="utf-8", closefd=True) as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
        # The probe artifact is diagnostic only; the parent directory is
        # created before the atomic replacement and the source stream remains
        # untouched.
        import os

        os.replace(temporary, path)
    finally:
        import os

        if os.path.exists(temporary):
            os.unlink(temporary)


def _observations(observations: list[Any]) -> list[dict[str, Any]]:
    return [
        {
            "frame_idx": int(item.frame_idx),
            "sam_object_id": int(item.sam_object_id),
            "raw_sam_object_id": (
                None
                if item.raw_sam_object_id is None
                else int(item.raw_sam_object_id)
            ),
            "box_xyxy": np.asarray(item.box_xyxy, dtype=float).tolist(),
            "mask_shape": list(np.asarray(item.mask).shape),
            "mask_nonzero": int(np.asarray(item.mask, dtype=bool).sum()),
            "confidence": float(item.confidence),
            "source": str(item.source),
        }
        for item in observations
    ]


def run(event_id: str, device: str, output: Path, attempt: int) -> dict[str, Any]:
    event, _stage_event, _branch, main_branch = eligible_event(event_id)
    event_frame = int(event["event_frame"])
    sequence = str(event["sequence"])
    sequence_dir = DATA_ROOT / "train" / sequence
    paths = image_files(sequence_dir)
    y_pre_hash, y_pre_candidate_hash, _y_pre_row = _main_y_pre(main_branch, event_frame)
    if not paths or event_frame >= len(paths):
        raise RuntimeError(f"event frame is outside frozen image coverage: {sequence}:{event_frame}")
    box = np.asarray(event.get("current_gt_box"), dtype=float).reshape(-1)
    if box.size != 4 or not np.all(np.isfinite(box)):
        raise RuntimeError("frozen current event box is invalid")

    backend: Sam3Backend | None = None
    started = False
    result: dict[str, Any] = {
        "schema_version": "N72R6_TARGET_SESSION_API_PROBE_V1",
        "status": "FAIL_API_PROBE",
        "probe_only": True,
        "event_id": str(event_id),
        "sequence": sequence,
        "event_frame": event_frame,
        "public_id": int(event["n72r6_target_public_id"]),
        "human_box": box.tolist(),
        "main_y_pre_semantic_hash": y_pre_hash,
        "main_y_pre_candidate_content_sha256": y_pre_candidate_hash,
        "checkpoint": str(CHECKPOINT),
        "runtime_future_gt_used": False,
        "runtime_gt_read": False,
        "posthoc_gt_used": False,
        "interaction_source": "simulated_from_gt",
        "not_real_human_evidence": True,
        "attempt": int(attempt),
    }
    try:
        backend = Sam3Backend(
            checkpoint_path=str(CHECKPOINT),
            max_num_objects=16,
            multiplex_count=16,
            use_fa3=False,
            use_rope_real=True,
            compile=False,
            warm_up=False,
            session_expiration_sec=1200,
            output_prob_thresh=0.30,
            async_loading_frames=False,
            device=device,
        )
        session_id = backend.start_video(str(sequence_dir / "img1"))
        started = True

        # First request is exactly the failed target-session adapter path.
        backend.add_box(event_frame, 1, box)
        box_only = backend.get_last_official_prompt_outputs(event_frame)
        result["box_only"] = {
            "official_count": len(box_only),
            "observations": _observations(box_only),
            "prompt_fallback_log": list(getattr(backend, "_prompt_fallback_log", [])),
            "session_id": session_id,
        }

        # These are capability comparisons only.  They are not used by the
        # N72R6 target runner unless the protocol explicitly authorizes them.
        # In particular, a text prompt may reintroduce scene-wide detections.
        text_variants: dict[str, dict[str, Any]] = {}
        for label, text_prompt in (("visual", "visual"), ("person", "person"), ("empty", "")):
            observations = backend._send_prompt(
                event_frame,
                boxes=[box.copy()],
                text=text_prompt,
                source=f"target_api_probe_{label}_plus_box",
            )
            text_variants[label] = {
                "text": text_prompt,
                "official_count": len(observations),
                "observations": _observations(observations),
                "prompt_fallback_log": list(
                    getattr(backend, "_prompt_fallback_log", [])
                ),
            }
        result["text_variants_plus_box"] = text_variants
        result["status"] = "PASS_API_PROBE_COMPLETE"
        return result
    except Exception as exc:
        result.update(
            {
                "failure_type": type(exc).__name__,
                "message": str(exc),
                "traceback": traceback.format_exc(),
            }
        )
        raise
    finally:
        if backend is not None and started:
            try:
                backend.close()
            except Exception:
                pass
        del backend
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--event-id", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--attempt", type=int, default=1)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs/N72R6/attempts/target_session_api_probe_attempt1.json",
    )
    args = parser.parse_args()
    try:
        result = run(
            str(args.event_id),
            str(args.device),
            args.output,
            int(args.attempt),
        )
    except Exception as exc:
        result = {
            "schema_version": "N72R6_TARGET_SESSION_API_PROBE_V1",
            "status": "FAIL_API_PROBE",
            "event_id": str(args.event_id),
            "failure_type": type(exc).__name__,
            "message": str(exc),
            "traceback": traceback.format_exc(),
            "runtime_future_gt_used": False,
            "probe_only": True,
        }
    atomic_json(args.output, result)
    print(json.dumps({"status": result["status"], "output": str(args.output)}))
    return 0 if result["status"] == "PASS_API_PROBE_COMPLETE" else 1


if __name__ == "__main__":
    raise SystemExit(main())
