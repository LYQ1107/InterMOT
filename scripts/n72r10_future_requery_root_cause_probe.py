#!/usr/bin/env python3
"""N72R10 Stage 03 root-cause probe for the invalid smoke PASS.

This probe is engineering evidence only.  It uses the same frozen causal
runtime row and official checkpoint as the smoke, never reads GT, and keeps
each official backend in a fresh session.  It compares the raw official
propagation response before and after the existing target-isolation adapter
action, without changing that adapter or the official SAM3 code.
"""

from __future__ import annotations

from datetime import datetime, timezone
import gc
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sam3_intermot.backend.sam3_backend import Sam3Backend  # noqa: E402
from scripts.n72r10_true_future_requery_smoke import (  # noqa: E402
    CHECKPOINT,
    DATA_ROOT,
    END_FRAME,
    EVENT_FRAME,
    EVENT_ID,
    MACHINE_CHECKPOINT,
    RUNTIME_INPUT,
    SEQUENCE,
    TRIGGER_FRAME,
    causal_target_from_runtime,
    image_files,
    read_runtime_row,
)


PROBE_END_FRAME = TRIGGER_FRAME + 3


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n"
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def make_backend(device: str, *, max_num_objects: int = 16) -> Sam3Backend:
    return Sam3Backend(
        checkpoint_path=str(CHECKPOINT),
        max_num_objects=int(max_num_objects),
        multiplex_count=int(max_num_objects),
        use_fa3=False,
        use_rope_real=True,
        compile=False,
        warm_up=False,
        session_expiration_sec=1200,
        output_prob_thresh=0.30,
        async_loading_frames=False,
        device=device,
    )


def observation_view(observation: Any) -> dict[str, Any]:
    raw = getattr(observation, "raw_sam_object_id", None)
    return {
        "frame_idx": int(observation.frame_idx),
        "sam_object_id": int(observation.sam_object_id),
        "raw_sam_object_id": None if raw is None else int(raw),
        "box_xyxy": np.asarray(observation.box_xyxy, dtype=float).tolist(),
        "confidence": float(observation.confidence),
        "mask_sha256": None
        if observation.mask is None
        else hashlib.sha256(np.asarray(observation.mask, dtype=bool).tobytes()).hexdigest(),
    }


def raw_frame_view(outputs: dict[int, list[Any]]) -> dict[str, Any]:
    return {
        str(int(frame)): [observation_view(item) for item in rows]
        for frame, rows in sorted(outputs.items())
    }


def materialize(paths: list[Path]) -> tempfile.TemporaryDirectory[str]:
    handle = tempfile.TemporaryDirectory(prefix="n72r10_root_cause_probe_")
    directory = Path(handle.name) / "frames"
    directory.mkdir(parents=True, exist_ok=True)
    for local, global_frame in enumerate(range(TRIGGER_FRAME, PROBE_END_FRAME + 1)):
        source = paths[global_frame]
        os.symlink(str(source), str(directory / f"{local:06d}{source.suffix.lower() or '.jpg'}"))
    return handle


def _state_for_backend(backend: Sam3Backend) -> dict[str, Any]:
    predictor = getattr(backend, "_predictor", None)
    session_id = getattr(backend, "_session_id", None)
    entry = None if predictor is None or session_id is None else predictor._all_inference_states.get(session_id)
    state = entry.get("state") if isinstance(entry, dict) else None
    if not isinstance(state, dict):
        raise RuntimeError("official inference state is unavailable")
    return state


def _history_view(state: dict[str, Any]) -> list[dict[str, Any]]:
    history = state.get("action_history", [])
    if not isinstance(history, list):
        return [{"type": type(history).__name__}]
    result = []
    for item in history:
        if not isinstance(item, dict):
            result.append({"type": type(item).__name__})
            continue
        result.append(
            {
                "type": str(item.get("type")),
                "frame_idx": item.get("frame_idx"),
                "obj_ids": item.get("obj_ids"),
            }
        )
    return result


def _remove_competitors_preserve_state(
    backend: Sam3Backend,
    target_raw: int,
    *,
    clear_action_history: bool,
) -> dict[str, Any]:
    state = _state_for_backend(backend)
    metadata = state.get("tracker_metadata")
    if not isinstance(metadata, dict):
        raise RuntimeError("official tracker metadata is unavailable")
    before_ids = [int(value) for value in metadata.get("obj_ids_all_gpu", [])]
    model = getattr(getattr(backend, "_predictor", None), "model", None)
    remove_object = getattr(model, "remove_object", None)
    if not callable(remove_object):
        raise RuntimeError("official model.remove_object is unavailable")
    for raw_id in before_ids:
        if raw_id != int(target_raw):
            remove_object(state, int(raw_id), frame_idx=None, is_user_action=False)
    after_ids = [int(value) for value in metadata.get("obj_ids_all_gpu", [])]
    backend._compact_official_position_metadata(
        state,
        metadata_ids_before=before_ids,
        metadata_ids_after=after_ids,
    )
    history_before = _history_view(state)
    if clear_action_history:
        state["action_history"] = []
    return {
        "metadata_ids_before": before_ids,
        "metadata_ids_after": after_ids,
        "history_before_or_after_removal": history_before,
        "history_cleared": bool(clear_action_history),
    }


def run_case(case: str, *, predicted_box: list[float], paths: list[Path], device: str) -> dict[str, Any]:
    handle = materialize(paths)
    backend = make_backend(device, max_num_objects=1 if case == "max_num_objects_1" else 16)
    try:
        session_id = backend.start_video(str(Path(handle.name) / "frames"))
        human_observation = backend.add_box(0, 1, np.asarray(predicted_box, dtype=float))
        official_prompt = backend.get_last_official_prompt_outputs(0)
        add_box_official_count = len(official_prompt)
        prompt_route = "box_only_add_box"
        # The normal target session uses the supported text+box recovery when
        # box-only add_box has no singleton target.  For this diagnostic only,
        # issue that same existing adapter request but deliberately do not
        # remove the other official objects in the unisolated case.  This
        # separates the official propagation behavior from retain_object's
        # target-only state repair.
        finder = getattr(backend, "_find_obs_for_ext", None)
        target_match = finder(official_prompt, 1) if callable(finder) else None
        if target_match is None:
            sender = getattr(backend, "_send_prompt", None)
            if not callable(sender):
                raise RuntimeError("backend lacks the existing text+box prompt adapter")
            official_prompt = sender(
                0,
                boxes=[np.asarray(predicted_box, dtype=float)],
                text="person",
                source="n72r10_root_cause_probe_text_box",
            )
            prompt_route = "existing_text_box_prompt_without_or_with_isolation"
        external_raw = getattr(backend, "_ext_to_sam", {}).get(1)
        if external_raw is None:
            # ``add_box`` can expose a multi-object official response before
            # its adapter map is installed.  Reuse the backend's existing
            # one-to-one IoU lookup on that same response; do not infer a
            # public identity or select from any future frame.
            if callable(finder):
                mapped = finder(official_prompt, 1)
                if mapped is not None:
                    external_raw = getattr(mapped, "raw_sam_object_id", None)
                    if external_raw is None:
                        external_raw = getattr(mapped, "sam_object_id", None)
        target_raw = int(external_raw) if external_raw is not None else None
        isolation = None
        if case in {"remove_preserve_history", "remove_no_history"}:
            if target_raw is None:
                raise RuntimeError("box/text prompt did not bind a target raw ID")
            isolation = _remove_competitors_preserve_state(
                backend,
                target_raw,
                clear_action_history=case == "remove_no_history",
            )
        elif case == "isolated":
            if target_raw is None:
                raise RuntimeError("box prompt did not bind external target to a raw ID")
            isolation = backend.retain_official_raw_object(0, target_raw)
        outputs = backend.propagate(
            0,
            PROBE_END_FRAME - TRIGGER_FRAME,
            start_frame_index=0,
            keep_masks=True,
            cache_outputs=True,
        )
        return {
            "case": case,
            "session_id": str(session_id),
            "human_observation": observation_view(human_observation),
            "add_box_official_count": add_box_official_count,
            "prompt_route": prompt_route,
            "official_prompt_count": len(official_prompt),
            "official_prompt": [observation_view(item) for item in official_prompt],
            "external_to_raw": {str(k): int(v) for k, v in getattr(backend, "_ext_to_sam", {}).items()},
            "target_raw_before_isolation": target_raw,
            "isolation": isolation,
            "raw_propagation": raw_frame_view(outputs),
            "raw_nonempty_frames": [int(frame + TRIGGER_FRAME) for frame, rows in outputs.items() if rows],
            "raw_frame_count": len(outputs),
            "runtime_future_gt_used": False,
            "runtime_gt_read": False,
            "posthoc_gt_used": False,
        }
    finally:
        backend.close()
        del backend
        handle.cleanup()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--attempt", type=int, default=1)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs/N72R10/attempts/stage_03_root_cause_probe_attempt_01.json",
    )
    args = parser.parse_args()
    row = read_runtime_row(RUNTIME_INPUT, TRIGGER_FRAME)
    target_public_id, predicted_box, causal_state = causal_target_from_runtime(row)
    del target_public_id, causal_state
    paths = image_files(DATA_ROOT / "train" / SEQUENCE)
    if END_FRAME >= len(paths) or PROBE_END_FRAME >= len(paths):
        raise RuntimeError("frozen image coverage is incomplete")
    cases: list[dict[str, Any]] = []
    case_failures: list[dict[str, Any]] = []
    for case in ("remove_preserve_history", "remove_no_history", "max_num_objects_1"):
        try:
            cases.append(run_case(case, predicted_box=predicted_box, paths=paths, device=str(args.device)))
        except Exception as exc:
            case_failures.append(
                {
                    "case": case,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "traceback": __import__("traceback").format_exc(),
                }
            )
    result = {
        "schema_version": "N72R10_FUTURE_REQUERY_ROOT_CAUSE_PROBE_V1",
        "status": "PASS_PROBE_EVIDENCE_NOT_SCIENTIFIC_RESULT" if not case_failures else "FAIL_PROBE_CASE",
        "attempt": int(args.attempt),
        "event_id": EVENT_ID,
        "sequence": SEQUENCE,
        "event_frame": EVENT_FRAME,
        "trigger_frame": TRIGGER_FRAME,
        "probe_end_frame": PROBE_END_FRAME,
        "frozen_causal_input": {
            "runtime_input": str(RUNTIME_INPUT),
            "runtime_input_sha256": sha256_file(RUNTIME_INPUT),
            "checkpoint": str(CHECKPOINT),
            "checkpoint_sha256": sha256_file(CHECKPOINT),
            "machine_checkpoint": str(MACHINE_CHECKPOINT),
            "machine_checkpoint_sha256": sha256_file(MACHINE_CHECKPOINT),
            "causal_predicted_box_xyxy": predicted_box,
            "source_runtime_frame": TRIGGER_FRAME,
        },
        "cases": cases,
        "case_failures": case_failures,
        "runtime_future_gt_used": False,
        "runtime_gt_read": False,
        "posthoc_gt_used": False,
        "created_at_utc": now_utc(),
    }
    atomic_json(args.output, result)
    print(json.dumps({"status": result["status"], "output": str(args.output)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
