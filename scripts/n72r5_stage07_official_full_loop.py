#!/usr/bin/env python3
"""N72R5 Stage 07: official SAM3 full-loop branches.

This runner is intentionally isolated from the frozen N72R4 official-pair
runner.  It consumes only the N72R5 Stage 06 event manifest and the already
audited image/tape inputs.  One child process owns one event/branch and one
SAM3 session.  The runner records official candidates and causal boundaries;
public-ID effect scoring is a later CPU-only stage.

Branches:

``B0_NO_INTERVENTION``
    One continuous official baseline stream.
``B1_SPATIAL_CORRECTION_ONLY``
    Current-frame official box correction, followed by propagation.
``B2_SPATIAL_CORRECTION_PLUS_IMAGE_RECOVERY``
    B1 plus a causal official add-box recovery only if the target raw object
    is absent at event+1.  The recovery box is a frozen current-event
    corrected box, never a future GT box.
``B3_SPATIAL_CORRECTION_PLUS_TVC`` and ``B4_...RECOVERY_PLUS_TVC``
    The same official candidate streams as B1/B2, with an explicit marker for
    the later target-vs-competitor association stage.  TVC never runs inside
    the official SAM3 worker.

The current event box is read only after the branch freezes and hashes Y_pre.
Future GT is never loaded.  All artifacts are atomic and a failure artifact is
written before the child exits non-zero.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import traceback
from datetime import datetime, timezone
from typing import Any, Iterable

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.n72r4_official_future_pair import (  # noqa: E402
    CHECKPOINT,
    DATA_ROOT,
    atomic_json,
    atomic_jsonl,
    current_candidate_rows,
    image_files,
    install_official_shape_audit,
    make_backend,
    observation_view,
    official_cached_observations,
    semantic_pre_view,
    sha256_file,
)
from sam3_intermot.backend.sam3_backend import Sam3Backend  # noqa: E402


EVENT_MANIFEST = ROOT / "outputs/N72R5/mechanism_rounds/round_06_event_policy/real_event_manifest.json"
DEFAULT_ROUND_ROOT = ROOT / "outputs/N72R5/mechanism_rounds/round_07_official_full_loop"
DEFAULT_STATUS = ROOT / "outputs/N72R5/stage_status/stage_07_status.json"
HORIZON = 100
# Do not send a finite max_frame_num_to_track to the pinned official
# multiplex predictor.  Its propagation iterator is inclusive while the
# detector's buffered grounding bound is half-open; a finite 200-frame
# request therefore produces an empty feature batch at frame 200 on long
# prefixes.  The adapter still stops consuming at the explicit event/H100
# end frame, so this changes no evaluated window or candidate definition.
OFFICIAL_CONTEXT_WINDOW: int | None = None
MIN_TARGET_IOU_FOR_EXISTING_OBJECT = 0.50
MACHINE_CHECKPOINT = Path(
    "/data1/LWR/vranlee/SERVER_ONLY/avis/Interactive/SAM3_InterMOT/"
    "outputs/n9/checkpoints/osnet_x1_0_market1501.pth"
)
# The frozen Stage-06 pool contains one event with 16 visible current-frame
# candidates.  ADD/RECOVER must retain those candidates and add the corrected
# object (and recovery may add one more causal object), while the pinned
# official default max_num_objects=16 returns an empty axis instead of a
# valid new-object response.  This is an adapter-local capacity setting: it
# does not change the checkpoint, candidate tape, branch definitions, or
# evaluation windows.
N72R5_MAX_NUM_OBJECTS = 24

BRANCHES = (
    "B0_NO_INTERVENTION",
    "B1_SPATIAL_CORRECTION_ONLY",
    "B2_SPATIAL_CORRECTION_PLUS_IMAGE_RECOVERY",
    "B3_SPATIAL_CORRECTION_PLUS_TVC",
    "B4_SPATIAL_CORRECTION_PLUS_RECOVERY_PLUS_TVC",
)
RECOVERY_BRANCHES = {
    "B2_SPATIAL_CORRECTION_PLUS_IMAGE_RECOVERY",
    "B4_SPATIAL_CORRECTION_PLUS_RECOVERY_PLUS_TVC",
}
TVC_BRANCHES = {
    "B3_SPATIAL_CORRECTION_PLUS_TVC",
    "B4_SPATIAL_CORRECTION_PLUS_RECOVERY_PLUS_TVC",
}


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object: {path}")
    return value


def jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): jsonable(child) for key, child in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(child) for child in value]
    return value


def json_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            jsonable(value),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def finite_box(value: Any, *, label: str) -> np.ndarray:
    box = np.asarray(value, dtype=np.float64).reshape(-1)
    if box.size != 4 or not np.all(np.isfinite(box)) or box[2] <= box[0] or box[3] <= box[1]:
        raise ValueError(f"{label} is not a finite positive XYXY box")
    return box


def finite_observation_box(value: Any, *, label: str) -> np.ndarray:
    """Validate an official observation without discarding zero-area rows.

    The pinned official tracker can retain a finite, empty-mask disappearance
    row whose box has zero width or height.  N71's frozen candidate contract
    preserves those rows in the native candidate order.  They are not valid
    geometry for IoU or correction prompts, so those callers continue to use
    ``finite_box``; this helper is only for lossless official stream auditing.
    Inverted boxes remain an actionable corruption and are rejected.
    """
    box = np.asarray(value, dtype=np.float64).reshape(-1)
    if box.size != 4 or not np.all(np.isfinite(box)) or box[2] < box[0] or box[3] < box[1]:
        raise ValueError(f"{label} is not a finite non-inverted XYXY box")
    return box


def box_iou(left: Any, right: Any) -> float:
    a = finite_box(left, label="left box")
    b = finite_box(right, label="right box")
    x1, y1 = max(float(a[0]), float(b[0])), max(float(a[1]), float(b[1]))
    x2, y2 = min(float(a[2]), float(b[2])), min(float(a[3]), float(b[3]))
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, float(a[2] - a[0])) * max(0.0, float(a[3] - a[1]))
    area_b = max(0.0, float(b[2] - b[0])) * max(0.0, float(b[3] - b[1]))
    union = area_a + area_b - inter
    return float(inter / union) if union > 0.0 else 0.0


def load_events() -> list[dict[str, Any]]:
    payload = read_json(EVENT_MANIFEST)
    if payload.get("status") != "PASS_N72R5_EVENT_POLICY_FROZEN":
        raise RuntimeError(f"Stage06 manifest is not frozen PASS: {payload.get('status')}")
    events = [dict(item) for item in payload.get("events", [])]
    if len(events) != 40 or len({str(item.get("event_id")) for item in events}) != 40:
        raise RuntimeError(f"Stage06 event count/uniqueness invalid: {len(events)}")
    for event in events:
        if event.get("runtime_future_gt_used") is not False:
            raise RuntimeError(f"event permits runtime future GT: {event.get('event_id')}")
        if event.get("interaction_source") != "simulated_from_gt":
            raise RuntimeError(f"unexpected interaction source: {event.get('event_id')}")
        event_frame = int(event["event_frame"])
        frame_count = int(event["sequence_frame_count"])
        if event_frame < 1 or event_frame + HORIZON >= frame_count:
            raise RuntimeError(f"event does not have t-1/H100 coverage: {event.get('event_id')}")
        tape = Path(str(event["candidate_tape_ref"]))
        if not tape.is_file():
            raise FileNotFoundError(f"frozen N36 tape is missing: {tape}")
    return sorted(events, key=lambda item: str(item["event_id"]))


class FrozenMachineOSNetN72R5:
    """Machine-only ROI encoder with an explicit frozen checkpoint path."""

    feature_dim = 512

    def __init__(self, device: str) -> None:
        if not MACHINE_CHECKPOINT.is_file():
            raise FileNotFoundError(f"frozen machine encoder checkpoint is missing: {MACHINE_CHECKPOINT}")
        from torchreid.reid.utils.feature_extractor import FeatureExtractor

        self.device = device
        self.extractor = FeatureExtractor(
            model_name="osnet_x1_0",
            model_path=str(MACHINE_CHECKPOINT),
            image_size=(256, 128),
            device=device,
            verbose=False,
        )

    @staticmethod
    def _crop(image: Image.Image, box: Any) -> np.ndarray:
        x1, y1, x2, y2 = [int(round(float(value))) for value in box]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(image.width, x2), min(image.height, y2)
        if x2 <= x1 or y2 <= y1:
            return np.zeros((8, 8, 3), dtype=np.uint8)
        return np.asarray(image.crop((x1, y1, x2, y2)), dtype=np.uint8)

    def encode(self, image_path: Path, boxes: Iterable[Any]) -> np.ndarray:
        values = list(boxes)
        if not values:
            return np.zeros((0, self.feature_dim), dtype=np.float32)
        with Image.open(image_path) as handle:
            image = handle.convert("RGB")
            crops = [self._crop(image, box) for box in values]
        with torch.no_grad():
            features = self.extractor(crops).detach().float().cpu().numpy()
        features = np.asarray(features, dtype=np.float32).reshape(len(values), -1)
        if features.shape != (len(values), self.feature_dim) or not np.all(np.isfinite(features)):
            raise RuntimeError(f"machine ROI feature has invalid shape/values: {features.shape}")
        norms = np.linalg.norm(features, axis=1, keepdims=True)
        if np.any(norms <= 1.0e-6):
            raise RuntimeError("machine ROI feature has zero norm")
        return features / norms


def load_event(event_id: str) -> dict[str, Any]:
    matches = [item for item in load_events() if str(item["event_id"]) == str(event_id)]
    if len(matches) != 1:
        raise KeyError(f"N72R5 event not found: {event_id}")
    return matches[0]


def collect_pre_event_n72r5(
    backend: Any,
    window: dict[str, Any],
    event_frame: int,
) -> list[Any]:
    """Collect the frozen pre-event state without the legacy finite bound.

    The N72R4 helper imported by earlier runners owns a module-level 200-frame
    request.  That request is unsafe for long prefixes with the pinned
    official multiplex predictor: its inclusive propagation endpoint can
    meet the detector's half-open grounding buffer and create an empty feature
    batch.  N72R5 keeps the same frame range and official calls, but omits the
    optional bound and stops only by the explicit requested endpoint.
    """
    frame_start = int(window["frame_start"])
    frame_end = int(window["frame_end"])
    if not frame_start <= int(event_frame) <= frame_end:
        raise ValueError(
            f"event frame is outside frozen window: {event_frame} not in {frame_start}:{frame_end}"
        )
    initial = backend.detect_concept(frame_start, "person")
    if int(event_frame) == frame_start:
        return [observation.copy() for observation in initial]
    outputs = backend.propagate(
        frame_start,
        int(event_frame),
        start_frame_index=frame_start,
        max_frame_num_to_track=OFFICIAL_CONTEXT_WINDOW,
        keep_masks=True,
        cache_outputs=True,
    )
    expected = set(range(frame_start, int(event_frame) + 1))
    observed = {int(frame) for frame in outputs}
    if observed != expected:
        raise RuntimeError(
            f"N72R5 pre-event official coverage mismatch: missing={sorted(expected - observed)[:8]} "
            f"extra={sorted(observed - expected)[:8]}"
        )
    observations = outputs.get(int(event_frame), [])
    if not observations:
        raise RuntimeError(f"official backend returned empty Y_pre: frame={event_frame}")
    return [observation.copy() for observation in observations]


def collect_continuous_baseline_n72r5(
    backend: Any,
    window: dict[str, Any],
    event_frame: int,
    frame_end: int,
) -> tuple[list[Any], dict[int, list[Any]]]:
    """Run one untouched official stream through the explicit H100 endpoint.

    This is deliberately local to N72R5 so the frozen N72R4 runner remains
    read-only.  No observations are synthesized and no future GT is loaded.
    """
    frame_start = int(window["frame_start"])
    backend.detect_concept(frame_start, "person")
    outputs = backend.propagate(
        frame_start,
        int(frame_end),
        start_frame_index=frame_start,
        max_frame_num_to_track=OFFICIAL_CONTEXT_WINDOW,
        keep_masks=True,
        cache_outputs=True,
    )
    expected = set(range(frame_start, int(frame_end) + 1))
    observed = {int(frame) for frame in outputs}
    if observed != expected:
        raise RuntimeError(
            f"N72R5 continuous official coverage mismatch: missing={sorted(expected - observed)[:8]} "
            f"extra={sorted(observed - expected)[:8]}"
        )
    pre = outputs.get(int(event_frame), [])
    if not pre:
        raise RuntimeError(f"N72R5 continuous official baseline has empty Y_pre: frame={event_frame}")
    return [observation.copy() for observation in pre], {
        int(frame): [observation.copy() for observation in observations]
        for frame, observations in outputs.items()
    }


def event_window(event: dict[str, Any]) -> dict[str, Any]:
    """Return the N36 tape's legal full-sequence frame window.

    Stage 06 records the frozen N36 frame count and tape digest.  This stage
    does not use the tape rows for runtime decisions; the official SAM3 worker
    uses the same sequence images and only the event's current simulated box
    after Y_pre is frozen.
    """

    tape = Path(str(event["candidate_tape_ref"]))
    if not tape.is_file():
        raise FileNotFoundError(tape)
    return {
        "window_id": f"n72r5-n36-full-sequence-{event['sequence']}",
        "sequence": str(event["sequence"]),
        "frame_start": 0,
        "frame_end": int(event["sequence_frame_count"]) - 1,
        "candidate_tape_ref": str(tape),
        "candidate_tape_sha256": str(event["candidate_tape_sha256"]),
        "runtime_future_gt_used": False,
    }


def obs_key(observation: Any) -> tuple[str, int]:
    raw = observation.raw_sam_object_id
    if raw is not None:
        return ("raw", int(raw))
    return ("adapter", int(observation.sam_object_id))


def validate_observation_axis(observations: Iterable[Any], *, label: str) -> None:
    values = list(observations)
    keys = [obs_key(item) for item in values]
    if len(keys) != len(set(keys)):
        raise RuntimeError(f"duplicate official observation axis at {label}: {keys}")
    for index, observation in enumerate(values):
        try:
            box = finite_observation_box(observation.box_xyxy, label=f"{label}[{index}].box")
        except ValueError as exc:
            raw_box = np.asarray(observation.box_xyxy, dtype=np.float64).reshape(-1)
            mask = np.asarray(observation.mask)
            mask_nonzero = np.argwhere(mask.astype(bool))
            mask_summary = {
                "shape": list(mask.shape),
                "nonzero_count": int(mask_nonzero.shape[0]),
                "y_min": int(mask_nonzero[:, 0].min()) if mask_nonzero.size else None,
                "y_max": int(mask_nonzero[:, 0].max()) if mask_nonzero.size else None,
                "x_min": int(mask_nonzero[:, 1].min()) if mask_nonzero.size else None,
                "x_max": int(mask_nonzero[:, 1].max()) if mask_nonzero.size else None,
            }
            raise ValueError(
                f"{label}[{index}].box invalid; raw_box={raw_box.tolist()} "
                f"frame_idx={int(observation.frame_idx)} "
                f"sam_object_id={int(observation.sam_object_id)} "
                f"raw_sam_object_id={observation.raw_sam_object_id!r} "
                f"confidence={float(observation.confidence)!r} "
                f"mask_summary={mask_summary}"
            ) from exc
        if int(observation.frame_idx) < 0 or not np.all(np.isfinite(box)):
            raise RuntimeError(f"invalid official observation at {label}[{index}]")


def merge_official_observations(
    baseline: Iterable[Any],
    official_response: Iterable[Any],
    *,
    label: str,
) -> list[Any]:
    """Keep a complete baseline set while preferring official prompt outputs."""

    baseline_values = [item.copy() for item in baseline]
    official_values = [item.copy() for item in official_response]
    validate_observation_axis(baseline_values, label=f"{label}:baseline")
    validate_observation_axis(official_values, label=f"{label}:official")
    merged: list[Any] = []
    seen: set[tuple[str, int]] = set()
    for observation in official_values + baseline_values:
        key = obs_key(observation)
        if key in seen:
            continue
        seen.add(key)
        merged.append(observation.copy())
    validate_observation_axis(merged, label=f"{label}:merged")
    return merged


def closest_target(observations: list[Any], human_box: np.ndarray) -> tuple[Any | None, float]:
    if not observations:
        return None, 0.0
    ranked = sorted(
        ((box_iou(item.box_xyxy, human_box), int(item.sam_object_id), item) for item in observations),
        key=lambda value: (-value[0], value[1]),
    )
    return ranked[0][2], float(ranked[0][0])


def official_target_raw_keys(observations: list[Any], human_box: np.ndarray) -> set[tuple[str, int]]:
    target, target_iou = closest_target(observations, human_box)
    if target is None or target_iou < MIN_TARGET_IOU_FOR_EXISTING_OBJECT:
        return set()
    return {obs_key(target)}


def current_correction(
    backend: Any,
    event: dict[str, Any],
    pre: list[Any],
    human_box: np.ndarray,
) -> tuple[list[Any], dict[str, Any]]:
    """Apply one official current-frame correction after Y_pre is frozen."""

    action = str(event["action_type"])
    target, target_iou = closest_target(pre, human_box)
    # ADD and RECOVER are explicit new/recovered-object transactions.  For
    # reassignment/swap, use the current official object only when it is an
    # unambiguous visible match to the frozen current command box.
    use_new_object = action in {"ADD_NEW_IDENTITY", "RECOVER_IDENTITY"} or target is None or target_iou < MIN_TARGET_IOU_FOR_EXISTING_OBJECT
    for observation in pre:
        backend.register_detected_observation(observation)
    if use_new_object:
        existing = [int(value) for value in getattr(backend, "_objects", {}).keys()]
        prompt_id = max(existing, default=0) + 1000
        returned = backend.add_box(int(event["event_frame"]), prompt_id, human_box)
        route = "official_backend.add_box"
        target_sam_id = None
    else:
        prompt_id = int(target.sam_object_id)
        returned = backend.correct_object(int(event["event_frame"]), prompt_id, box_xyxy=human_box)
        route = "official_backend.correct_object"
        target_sam_id = prompt_id
    if returned is None or not np.all(np.isfinite(finite_box(returned.box_xyxy, label="official correction output"))):
        raise RuntimeError(f"official correction returned an invalid observation: {event['event_id']}")
    fallback_entries = list(getattr(backend, "_prompt_fallback_log", []))
    no_output = any(
        int(item.get("object_id", -1)) == int(prompt_id) and bool(item.get("no_sam_output"))
        for item in fallback_entries
    )
    if no_output:
        raise RuntimeError(f"official correction produced no SAM output: {event['event_id']}")
    official_response = official_cached_observations(backend, int(event["event_frame"]))
    merged = merge_official_observations(pre, official_response, label=f"{event['event_id']}:event_correction")
    return merged, {
        "status": "PASS_OFFICIAL_CURRENT_FRAME_CORRECTION",
        "route": route,
        "action_type": action,
        "prompt_object_id": int(prompt_id),
        "target_sam_id_before_correction": target_sam_id,
        "target_match_iou_before_correction": float(target_iou),
        "new_object_transaction": bool(use_new_object),
        "human_box": [float(value) for value in human_box],
        "returned_human_observation": observation_view(returned),
        "official_response_count": len(official_response),
        "merged_candidate_count": len(merged),
        "event_frame_memory_read": False,
        "first_future_frame": int(event["event_frame"]) + 1,
        "runtime_future_gt_used": False,
        "runtime_gt_read": False,
        "interaction_source": "simulated_from_gt",
        "not_real_human_evidence": True,
        "prompt_fallback_log": fallback_entries,
    }


def future_observations(
    backend: Any,
    event: dict[str, Any],
    branch: str,
    frame_end: int,
    corrected_event_observations: list[Any],
    human_box: np.ndarray,
) -> tuple[dict[int, list[Any]], dict[str, Any]]:
    """Propagate H100 and optionally execute one causal image recovery."""

    event_frame = int(event["event_frame"])
    if branch not in RECOVERY_BRANCHES:
        outputs = backend.propagate(
            event_frame,
            frame_end,
            start_frame_index=event_frame,
            max_frame_num_to_track=OFFICIAL_CONTEXT_WINDOW,
            keep_masks=True,
            cache_outputs=True,
        )
        expected = set(range(event_frame, frame_end + 1))
        observed = {int(value) for value in outputs}
        if observed != expected:
            raise RuntimeError(
                f"official future coverage mismatch: missing={sorted(expected - observed)[:8]} "
                f"extra={sorted(observed - expected)[:8]}"
            )
        return (
            {int(frame): [item.copy() for item in values] for frame, values in outputs.items()},
            {
                "enabled": False,
                "status": "NOT_TRIGGERED_BRANCH_DISABLED",
                "trigger_frame": None,
                "runtime_future_gt_used": False,
            },
        )

    first_outputs = backend.propagate(
        event_frame,
        event_frame + 1,
        start_frame_index=event_frame,
        max_frame_num_to_track=OFFICIAL_CONTEXT_WINDOW,
        keep_masks=True,
        cache_outputs=True,
    )
    expected_first = {event_frame, event_frame + 1}
    if {int(value) for value in first_outputs} != expected_first:
        raise RuntimeError(
            f"official recovery trigger coverage mismatch: missing={sorted(expected_first - set(first_outputs))}"
        )
    future = {int(frame): [item.copy() for item in values] for frame, values in first_outputs.items()}
    event1 = future[event_frame + 1]
    target_keys = official_target_raw_keys(corrected_event_observations, human_box)
    event1_keys = {obs_key(item) for item in event1}
    if target_keys and target_keys.intersection(event1_keys):
        trigger = {
            "enabled": True,
            "status": "NOT_TRIGGERED_TARGET_RAW_OBJECT_VISIBLE_AT_EVENT_PLUS_ONE",
            "trigger_frame": None,
            "target_keys": [list(value) for value in sorted(target_keys)],
            "event_plus_one_keys": [list(value) for value in sorted(event1_keys)],
            "runtime_future_gt_used": False,
        }
        continuation = backend.propagate(
            event_frame + 1,
            frame_end,
            start_frame_index=event_frame + 1,
            max_frame_num_to_track=OFFICIAL_CONTEXT_WINDOW,
            keep_masks=True,
            cache_outputs=True,
        )
        expected = set(range(event_frame + 1, frame_end + 1))
        if {int(value) for value in continuation} != expected:
            raise RuntimeError("official continuation coverage failed after non-triggered recovery probe")
        for frame, values in continuation.items():
            if int(frame) > event_frame + 1:
                future[int(frame)] = [item.copy() for item in values]
        return future, trigger

    # The seed is a causal zero-step prediction from the current correction
    # box.  It is deliberately not a future GT lookup.  ``add_box`` is used
    # because the official adapter re-prompts the complete active set and can
    # continue propagation after the new image-grounded candidate is added.
    recovery_frame = event_frame + 1
    existing = [int(value) for value in getattr(backend, "_objects", {}).keys()]
    recovery_id = max(existing, default=0) + 100000
    recovery_box = human_box.copy()
    recovered = backend.add_box(recovery_frame, recovery_id, recovery_box)
    if recovered is None or not np.all(np.isfinite(finite_box(recovered.box_xyxy, label="recovery output"))):
        raise RuntimeError(f"official image recovery returned an invalid observation: {event['event_id']}")
    recovery_response = official_cached_observations(backend, recovery_frame)
    treatment_event1 = merge_official_observations(
        event1,
        recovery_response,
        label=f"{event['event_id']}:event_plus_one_recovery",
    )
    future[recovery_frame] = treatment_event1
    continuation = backend.propagate(
        recovery_frame,
        frame_end,
        start_frame_index=recovery_frame,
        max_frame_num_to_track=OFFICIAL_CONTEXT_WINDOW,
        keep_masks=True,
        cache_outputs=True,
    )
    expected = set(range(recovery_frame, frame_end + 1))
    if {int(value) for value in continuation} != expected:
        raise RuntimeError(
            f"official recovery continuation coverage mismatch: missing={sorted(expected - set(continuation))[:8]} "
            f"extra={sorted(set(continuation) - expected)[:8]}"
        )
    for frame, values in continuation.items():
        if int(frame) > recovery_frame:
            future[int(frame)] = [item.copy() for item in values]
    return future, {
        "enabled": True,
        "status": "PASS_OFFICIAL_IMAGE_GROUNDED_RECOVERY_AT_EVENT_PLUS_ONE",
        "trigger_frame": recovery_frame,
        "target_keys_before_trigger": [list(value) for value in sorted(target_keys)],
        "event_plus_one_keys_before_trigger": [list(value) for value in sorted(event1_keys)],
        "recovery_object_id": int(recovery_id),
        "recovery_box": [float(value) for value in recovery_box],
        "recovery_box_source": "current_event_corrected_box_zero_step_causal_prediction",
        "official_response_count": len(recovery_response),
        "merged_event_plus_one_candidate_count": len(treatment_event1),
        "runtime_future_gt_used": False,
        "runtime_gt_read": False,
        "interaction_source": "simulated_from_gt",
        "not_real_human_evidence": True,
    }


def branch_frame_row(
    event: dict[str, Any],
    branch: str,
    frame: int,
    phase: str,
    frame_path: Path,
    candidates: list[dict[str, Any]],
    y_pre_hash: str,
    correction: dict[str, Any] | None,
    recovery: dict[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": "N72R5_OFFICIAL_SAM3_FULL_LOOP_FRAME_V1",
        "record_type": "official_candidate_frame",
        "event_id": str(event["event_id"]),
        "sequence": str(event["sequence"]),
        "action_type": str(event["action_type"]),
        "branch": branch,
        "tvc_enabled_for_later_association": branch in TVC_BRANCHES,
        "phase": phase,
        "event_frame": int(event["event_frame"]),
        "frame": int(frame),
        "frame_horizon": int(frame) - int(event["event_frame"]),
        "frame_hash_sha256": sha256_file(frame_path),
        "candidates": candidates,
        "candidate_count": len(candidates),
        "candidate_set_complete": True,
        "candidate_order": [int(item["candidate_index"]) for item in candidates],
        "y_pre_semantic_hash": y_pre_hash,
        "correction": correction,
        "recovery": recovery,
        "memory_read": False if int(frame) == int(event["event_frame"]) else None,
        "event_frame_memory_read": False,
        "first_memory_visible_frame": int(event["event_frame"]) + 1,
        "runtime_future_gt_used": False,
        "runtime_gt_read": False,
        "posthoc_gt_used": False,
        "interaction_source": "simulated_from_gt",
        "not_real_human_evidence": True,
        "public_id_inference": False,
        "candidate_stream_kind": "OFFICIAL_SAM3_FULL_LOOP_BRANCH",
    }


def checkpoint_sha256() -> str:
    cached = os.environ.get("N72R5_CHECKPOINT_SHA256")
    return cached if cached else sha256_file(CHECKPOINT)


def enable_b0_official_memory_trim(backend: Any) -> dict[str, Any]:
    """Enable the pinned official long-eval trim for the no-prompt branch.

    The official tracker documents this option for evaluation where only the
    first frame receives prompts.  B0 has exactly that contract; interactive
    branches deliberately do not enable it because they receive the current
    correction prompt later in the sequence.  This changes only the official
    memory/output retention policy and is recorded in the branch provenance.
    """
    predictor = getattr(backend, "_predictor", None)
    model = getattr(predictor, "model", None)
    targets = [model, getattr(model, "tracker", None), getattr(model, "model", None)]
    changed: list[str] = []
    seen: set[int] = set()
    for target in targets:
        if target is None or id(target) in seen or not hasattr(target, "trim_past_non_cond_mem_for_eval"):
            continue
        seen.add(id(target))
        setattr(target, "trim_past_non_cond_mem_for_eval", True)
        changed.append(type(target).__name__)
    if not changed:
        raise RuntimeError(
            "official B0 memory trim is unavailable on the pinned tracker; "
            "refusing an unverified memory workaround"
        )
    policy = getattr(backend, "_runtime_memory_policy", None)
    if isinstance(policy, dict):
        policy["trim_past_non_cond_mem_for_eval"] = True
        policy["source"] = (
            str(policy.get("source", "official_runtime"))
            + "+official_trim_past_non_cond_mem_for_eval_b0"
        )
    return {
        "enabled": True,
        "mechanism": "official_trim_past_non_cond_mem_for_eval",
        "scope": "B0_ONLY_FIRST_FRAME_PROMPT",
        "targets": changed,
        "runtime_future_gt_used": False,
    }


def enable_official_batched_grounding_batch1(backend: Any) -> dict[str, Any]:
    """Use the pinned official batch-1 grounding path for every branch.

    The pinned builder's official default is batched grounding with a larger
    batch.  The same official implementation accepts batch size one; this
    bounds the detector/segmentation peak to one frame and cleans older
    chunks.  This is an adapter-local, official runtime memory policy and is
    recorded so a later candidate audit can verify the mechanism actually
    used.  The setting applies to all branches because every branch must use
    the same official candidate-generation resource policy.
    """
    predictor = getattr(backend, "_predictor", None)
    model = getattr(predictor, "model", None)
    if model is None or not hasattr(model, "use_batched_grounding"):
        raise RuntimeError(
            "official batched grounding is unavailable on the pinned multiplex model; "
            "refusing an unverified prefetch workaround"
        )
    if not hasattr(model, "batched_grounding_batch_size"):
        raise RuntimeError(
            "official batched grounding batch-size control is unavailable; "
            "refusing an unverified prefetch workaround"
        )
    model.use_batched_grounding = True
    model.batched_grounding_batch_size = 1
    policy = getattr(backend, "_runtime_memory_policy", None)
    if isinstance(policy, dict):
        policy["official_batched_grounding"] = True
        policy["official_batched_grounding_batch_size"] = 1
        policy["official_multigpu_next_chunk_prefetch"] = False
        policy["source"] = (
            str(policy.get("source", "official_runtime"))
            + "+official_batched_grounding_batch1_no_prefetch"
        )
    return {
        "enabled": True,
        "mechanism": "official_batched_grounding_batch1_no_next_chunk_prefetch",
        "scope": "ALL_BRANCHES_SINGLE_FRAME_GROUNDING",
        "targets": [type(model).__name__],
        "batch_size": 1,
        "runtime_future_gt_used": False,
    }


def make_backend_n72r5() -> Sam3Backend:
    """Build the pinned official backend with enough object capacity.

    Stage-06 intentionally keeps the complete visible candidate set.  The
    official default capacity of 16 cannot represent a 16-candidate frame
    plus a causal ADD/RECOVER object.  Use the documented constructor option
    locally rather than modifying the pinned third-party source or the older
    N72R4 runner.
    """

    backend = Sam3Backend(
        checkpoint_path=str(CHECKPOINT),
        max_num_objects=N72R5_MAX_NUM_OBJECTS,
        multiplex_count=16,
        use_fa3=False,
        use_rope_real=True,
        compile=False,
        warm_up=False,
        session_expiration_sec=1200,
        output_prob_thresh=0.30,
        async_loading_frames=False,
        device="cuda",
    )
    policy = getattr(backend, "_runtime_memory_policy", None)
    if isinstance(policy, dict):
        policy["official_max_num_objects"] = N72R5_MAX_NUM_OBJECTS
        policy["official_object_capacity_repair"] = (
            "retain_complete_current_candidates_plus_causal_add_or_recovery"
        )
        policy["source"] = (
            str(policy.get("source", "official_runtime"))
            + "+official_object_capacity_24"
        )
    return backend


def run_worker(event: dict[str, Any], branch: str, output: Path) -> dict[str, Any]:
    if branch not in BRANCHES:
        raise ValueError(f"unknown branch: {branch}")
    if output.exists() or output.with_suffix(".done.json").exists() or output.with_suffix(".failure.json").exists():
        raise RuntimeError(f"refusing to overwrite existing branch artifact: {output}")
    event_frame = int(event["event_frame"])
    frame_end = event_frame + HORIZON
    sequence = str(event["sequence"])
    sequence_dir = DATA_ROOT / "train" / sequence
    paths = image_files(sequence_dir)
    if not paths or frame_end >= len(paths):
        raise RuntimeError(f"official image range is incomplete: {sequence}:{event_frame}:{frame_end}")
    window = event_window(event)
    backend: Any | None = None
    encoder: FrozenMachineOSNetN72R5 | None = None
    started = time.time()
    try:
        backend = make_backend_n72r5()
        backend.start_video(str(sequence_dir / "img1"))
        install_official_shape_audit(backend)
        memory_engineering: dict[str, Any] | None = None
        prefetch_repair = enable_official_batched_grounding_batch1(backend)
        if branch == "B0_NO_INTERVENTION":
            memory_engineering = enable_b0_official_memory_trim(backend)
            memory_engineering["prefetch_repair"] = prefetch_repair
        else:
            memory_engineering = {"prefetch_repair": prefetch_repair}
        continuous: dict[int, list[Any]] | None = None
        if branch == "B0_NO_INTERVENTION":
            pre, continuous = collect_continuous_baseline_n72r5(backend, window, event_frame, frame_end)
        else:
            pre = collect_pre_event_n72r5(backend, window, event_frame)
        if not pre:
            raise RuntimeError(f"Y_pre is empty: {event['event_id']}/{branch}")
        validate_observation_axis(pre, label=f"{event['event_id']}/{branch}:Y_pre")
        y_pre_view = semantic_pre_view(pre)
        y_pre_hash = json_hash(y_pre_view)

        # This is the causal boundary: only after the semantic pre-state is
        # frozen may the simulated current human command box be opened.
        human_box = finite_box(event["current_gt_box"], label=f"{event['event_id']}.current_gt_box")
        correction: dict[str, Any] | None = None
        event_observations = [item.copy() for item in pre]
        if branch != "B0_NO_INTERVENTION":
            event_observations, correction = current_correction(backend, event, pre, human_box)
        encoder = FrozenMachineOSNetN72R5("cuda:0")
        runtime_policy = backend.runtime_memory_policy()
        event_candidates = current_candidate_rows(
            backend, encoder, paths[event_frame], event_frame, event_observations
        )
        recovery_audit = {
            "enabled": branch in RECOVERY_BRANCHES,
            "status": "NOT_APPLICABLE_BASELINE" if branch == "B0_NO_INTERVENTION" else "NOT_RUN",
            "runtime_future_gt_used": False,
        }
        if branch == "B0_NO_INTERVENTION":
            future = {
                int(frame): [item.copy() for item in values]
                for frame, values in (continuous or {}).items()
                if event_frame <= int(frame) <= frame_end
            }
        else:
            future, recovery_audit = future_observations(
                backend, event, branch, frame_end, event_observations, human_box
            )
        expected_future = set(range(event_frame, frame_end + 1))
        if set(future) != expected_future:
            raise RuntimeError(
                f"branch future observation coverage mismatch: missing={sorted(expected_future - set(future))[:8]} "
                f"extra={sorted(set(future) - expected_future)[:8]}"
            )
        frame_rows: list[dict[str, Any]] = []
        frame_rows.append(
            branch_frame_row(
                event,
                branch,
                event_frame,
                "Y_PRE_FROZEN",
                paths[event_frame],
                event_candidates,
                y_pre_hash,
                correction,
                recovery_audit,
            )
        )
        for frame in range(event_frame + 1, frame_end + 1):
            observations = [item.copy() for item in future[frame]]
            validate_observation_axis(observations, label=f"{event['event_id']}/{branch}:{frame}")
            candidates = current_candidate_rows(backend, encoder, paths[frame], frame, observations)
            frame_rows.append(
                branch_frame_row(
                    event,
                    branch,
                    frame,
                    "FUTURE_PROPAGATION",
                    paths[frame],
                    candidates,
                    y_pre_hash,
                    None,
                    recovery_audit if frame == event_frame + 1 else {"enabled": recovery_audit.get("enabled", False), "status": "SEE_EVENT_PLUS_ONE_AUDIT", "runtime_future_gt_used": False},
                )
            )
        atomic_jsonl(output, frame_rows)
        result = {
            "schema_version": "N72R5_OFFICIAL_SAM3_FULL_LOOP_BRANCH_DONE_V1",
            "status": "PASS_N72R5_OFFICIAL_FULL_LOOP_BRANCH",
            "event_id": str(event["event_id"]),
            "sequence": sequence,
            "action_type": str(event["action_type"]),
            "branch": branch,
            "session_id": str(getattr(backend, "_session_id", "")),
            "event_frame": event_frame,
            "frame_start": event_frame,
            "frame_end": frame_end,
            "future_frame_count": HORIZON,
            "artifact_first_future_frame": event_frame + 1,
            "y_pre_candidate_count": len(pre),
            "event_candidate_count": len(event_candidates),
            "y_pre_semantic_hash": y_pre_hash,
            "y_pre_frozen": y_pre_view,
            "correction": correction,
            "recovery": recovery_audit,
            "candidate_artifact": str(output),
            "candidate_artifact_sha256": sha256_file(output),
            "candidate_frame_count": len(frame_rows),
            "candidate_complete": True,
            "official_backend": True,
            "official_future_propagation": True,
            "checkpoint_sha256": checkpoint_sha256(),
            "runtime_memory_policy": runtime_policy,
            "memory_engineering": memory_engineering,
            "runtime_future_gt_used": False,
            "runtime_gt_read": False,
            "posthoc_gt_used": False,
            "interaction_source": "simulated_from_gt",
            "not_real_human_evidence": True,
            "public_id_inference": False,
            "tvc_deferred_to_exact_cpu_association": branch in TVC_BRANCHES,
            "scientific_result": "OFFICIAL_FULL_LOOP_CANDIDATE_STREAM_ONLY_NO_POSTHOC_EFFECT",
            "elapsed_sec": time.time() - started,
        }
        atomic_json(output.with_suffix(".done.json"), result)
        return result
    except Exception as exc:
        if backend is not None and getattr(backend, "_predictor", None) is not None:
            setattr(
                exc,
                "_official_shape_audit",
                list(getattr(backend._predictor.model, "_n72r4_shape_audit", [])),
            )
        raise
    finally:
        if backend is not None:
            try:
                backend.close()
            except Exception:
                pass
        encoder = None
        backend = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def write_failure(output: Path, event: dict[str, Any], branch: str, exc: BaseException) -> Path:
    failure = output.with_suffix(".failure.json")
    atomic_json(
        failure,
        {
            "schema_version": "N72R5_STAGE07_FAILURE_RECORD_V1",
            "stage": "07_OFFICIAL_FULL_LOOP",
            "status": "FAIL_PRESERVED",
            "event_id": str(event["event_id"]),
            "sequence": str(event["sequence"]),
            "branch": branch,
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
            "official_shape_audit": getattr(exc, "_official_shape_audit", []),
            "runtime_future_gt_used": False,
            "scientific_result": "NO_SCIENTIFIC_RESULT",
        },
    )
    return failure


def worker_main(args: argparse.Namespace) -> int:
    event = load_event(str(args.event_id))
    output = args.output.resolve()
    try:
        result = run_worker(event, str(args.branch), output)
        print(json.dumps({"status": result["status"], "event_id": args.event_id, "branch": args.branch, "output": str(output)}, sort_keys=True), flush=True)
        return 0
    except Exception as exc:
        failure = write_failure(output, event, str(args.branch), exc)
        print(json.dumps({"status": "FAIL_N72R5_OFFICIAL_FULL_LOOP_BRANCH", "event_id": args.event_id, "branch": args.branch, "failure": str(failure)}, sort_keys=True), flush=True)
        return 1


def launch_worker(event: dict[str, Any], branch: str, root: Path, gpu: int, logs: Path, checkpoint_digest: str) -> dict[str, Any]:
    event_id = str(event["event_id"])
    output = root / branch / f"{event_id}.jsonl"
    log_path = logs / f"{event_id}.{branch}.log"
    output.parent.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_handle = log_path.open("w", encoding="utf-8")
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = str(gpu)
    environment["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    environment["N72R5_CHECKPOINT_SHA256"] = checkpoint_digest
    command = [
        str(Path(sys.executable)),
        str(Path(__file__).resolve()),
        "--worker",
        "--event-id",
        event_id,
        "--branch",
        branch,
        "--output",
        str(output),
    ]
    started = time.time()
    try:
        completed = subprocess.run(
            command,
            cwd=str(ROOT),
            env=environment,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    finally:
        log_handle.close()
    done = output.with_suffix(".done.json")
    failure = output.with_suffix(".failure.json")
    payload = read_json(done) if done.is_file() else (read_json(failure) if failure.is_file() else {})
    return {
        "event_id": event_id,
        "sequence": str(event["sequence"]),
        "branch": branch,
        "gpu": int(gpu),
        "return_code": int(completed.returncode),
        "output": str(output),
        "done": str(done) if done.is_file() else None,
        "failure": str(failure) if failure.is_file() else None,
        "log": str(log_path),
        "status": payload.get("status", "MISSING_ARTIFACT"),
        "elapsed_sec": time.time() - started,
        "payload": payload,
    }


def compare_event(event: dict[str, Any], records: list[dict[str, Any]]) -> dict[str, Any]:
    failures = [
        {
            "branch": str(item["branch"]),
            "return_code": int(item["return_code"]),
            "status": item.get("status"),
            "failure": item.get("failure"),
        }
        for item in records
        if int(item["return_code"]) != 0 or item.get("status") != "PASS_N72R5_OFFICIAL_FULL_LOOP_BRANCH"
    ]
    hashes: dict[str, Any] = {}
    for item in records:
        payload = item.get("payload") if isinstance(item.get("payload"), dict) else None
        if payload is None and item.get("done"):
            done_path = Path(str(item["done"]))
            if done_path.is_file():
                try:
                    payload = read_json(done_path)
                except Exception:
                    payload = None
        hashes[str(item["branch"])] = payload.get("y_pre_semantic_hash") if isinstance(payload, dict) else None
    hash_values = [value for value in hashes.values() if value is not None]
    prefix_equal = len(hash_values) == len(BRANCHES) and len(set(hash_values)) == 1
    statuses = [item.get("status") for item in records]
    passed = not failures and prefix_equal and statuses == ["PASS_N72R5_OFFICIAL_FULL_LOOP_BRANCH"] * len(BRANCHES)
    return {
        "event_id": str(event["event_id"]),
        "sequence": str(event["sequence"]),
        "status": "PASS_N72R5_OFFICIAL_FULL_LOOP_EVENT" if passed else "BLOCKED_N72R5_OFFICIAL_FULL_LOOP_EVENT",
        "branch_count_expected": len(BRANCHES),
        "branch_count_completed": len(records),
        "branch_statuses": {str(item["branch"]): item.get("status") for item in records},
        "return_codes": {str(item["branch"]): int(item["return_code"]) for item in records},
        "y_pre_semantic_hash_by_branch": hashes,
        "y_pre_hash_equal": prefix_equal,
        "prefix_gate": "PASS_SHARED_Y_PRE" if prefix_equal else "FAIL_NONDETERMINISTIC_PREFIX",
        "failures": failures,
        "runtime_future_gt_used": False,
        "interaction_source": "simulated_from_gt",
        "not_real_human_evidence": True,
    }


def orchestrator_main(args: argparse.Namespace) -> int:
    events = load_events()
    scope = "full_frozen_event_set"
    if args.event_id:
        events = [event for event in events if str(event["event_id"]) == str(args.event_id)]
        if not events:
            raise RuntimeError(f"event is not in frozen Stage06 manifest: {args.event_id}")
        scope = "targeted_smoke"
    round_root = args.root.resolve()
    if round_root.exists() and any(round_root.iterdir()):
        raise RuntimeError(f"refusing to overwrite non-empty Stage07 root: {round_root}")
    round_root.mkdir(parents=True, exist_ok=True)
    prior_records: dict[tuple[str, str], dict[str, Any]] = {}
    prior_failures: list[dict[str, Any]] = []
    prior_manifest_path: Path | None = None
    if args.resume_from is not None:
        prior_root = args.resume_from.resolve()
        prior_manifest_path = prior_root / "official_full_loop_manifest.json"
        if not prior_manifest_path.is_file():
            raise FileNotFoundError(f"resume manifest is missing: {prior_manifest_path}")
        prior_manifest = read_json(prior_manifest_path)
        prior_worker_records = prior_manifest.get("worker_records", [])
        if not isinstance(prior_worker_records, list):
            raise TypeError(f"resume worker_records is not a list: {prior_manifest_path}")
        for value in prior_worker_records:
            if not isinstance(value, dict):
                continue
            key = (str(value.get("event_id")), str(value.get("branch")))
            if int(value.get("return_code", 1)) == 0 and value.get("status") == "PASS_N72R5_OFFICIAL_FULL_LOOP_BRANCH":
                prior_records[key] = dict(value)
            else:
                prior_failures.append({
                    "event_id": key[0],
                    "branch": key[1],
                    "return_code": value.get("return_code"),
                    "status": value.get("status"),
                    "failure": value.get("failure"),
                    "source_manifest": str(prior_manifest_path),
                })
        scope = "full_frozen_event_set_resume"
    logs = round_root / "worker_logs"
    checkpoint_digest = sha256_file(CHECKPOINT)
    records: list[dict[str, Any]] = []
    comparisons: list[dict[str, Any]] = []
    stopped_on_failure = False
    for event in events:
        event_records: list[dict[str, Any]] = []
        for branch in BRANCHES:
            prior = prior_records.get((str(event["event_id"]), branch))
            if prior is not None:
                record = prior
                print(json.dumps({
                    "event_id": event["event_id"],
                    "branch": branch,
                    "status": "REUSED_PRIOR_PASS",
                    "source": prior.get("output"),
                }, sort_keys=True), flush=True)
            else:
                record = launch_worker(event, branch, round_root / "runtime", int(args.gpu), logs, checkpoint_digest)
            event_records.append(record)
            records.append({key: value for key, value in record.items() if key != "payload"})
            if int(record["return_code"]) != 0:
                stopped_on_failure = True
                break
        comparison = compare_event(event, event_records)
        comparisons.append(comparison)
        print(json.dumps({"event_id": event["event_id"], "status": comparison["status"], "branches": len(event_records)}, sort_keys=True), flush=True)
        if comparison["status"] != "PASS_N72R5_OFFICIAL_FULL_LOOP_EVENT":
            stopped_on_failure = True
            break
    all_pass = len(comparisons) == len(events) and not stopped_on_failure and all(
        item["status"] == "PASS_N72R5_OFFICIAL_FULL_LOOP_EVENT" for item in comparisons
    )
    manifest = {
        "schema_version": "N72R5_OFFICIAL_SAM3_FULL_LOOP_MANIFEST_V1",
        "status": "PASS_N72R5_OFFICIAL_FULL_LOOP_SET" if all_pass else "BLOCKED_N72R5_OFFICIAL_FULL_LOOP_SET",
        "execution_scope": scope,
        "event_count_expected": len(events),
        "event_count_completed": len(comparisons),
        "events": comparisons,
        "worker_records": records,
        "branches": list(BRANCHES),
        "checkpoint_sha256": checkpoint_digest,
        "horizon": HORIZON,
        "same_checkpoint": True,
        "same_prefix_protocol": True,
        "shared_y_pre_required": True,
        "official_future_propagation": True,
        "runtime_future_gt_used": False,
        "interaction_source": "simulated_from_gt",
        "not_real_human_evidence": True,
        "scientific_result": "OFFICIAL_FULL_LOOP_CANDIDATE_STREAM_ONLY_NO_POSTHOC_EFFECT_RESULT",
        "resume_from": str(prior_manifest_path) if prior_manifest_path is not None else None,
        "prior_failure_records": prior_failures,
    }
    manifest_path = round_root / "official_full_loop_manifest.json"
    atomic_json(manifest_path, manifest)
    status = {
        "schema_version": "N72R5_STAGE_STATUS_V1",
        "stage": "07_OFFICIAL_FULL_LOOP",
        "status": manifest["status"],
        "execution_scope": scope,
        "event_count_expected": len(events),
        "event_count_completed": len(comparisons),
        "branch_count_per_event": len(BRANCHES),
        "branches": list(BRANCHES),
        "manifest": str(manifest_path),
        "checkpoint_sha256": checkpoint_digest,
        "runtime_future_gt_used": False,
        "interaction_source": "simulated_from_gt",
        "not_real_human_evidence": True,
        "full_loop_authorized_for_exact_association": bool(all_pass),
        "posthoc_effect_evaluated": False,
        "stop_reason": "first_actionable_branch_failure_preserved" if stopped_on_failure else None,
        "resume_from": str(prior_manifest_path) if prior_manifest_path is not None else None,
        "reused_prior_pass_count": len([key for key in prior_records if key[0] in {str(item["event_id"]) for item in events}]),
        "prior_failure_count": len(prior_failures),
        "created_at_utc": now_utc(),
    }
    atomic_json(round_root / "stage_07_status.json", status)
    if args.stage_status is not None:
        atomic_json(args.stage_status.resolve(), status)
    print(json.dumps({"status": manifest["status"], "manifest": str(manifest_path), "stage_status": str(round_root / 'stage_07_status.json')}, sort_keys=True))
    return 0 if all_pass else 1


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--event-id")
    parser.add_argument("--branch", choices=BRANCHES)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--gpu", type=int, default=1)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROUND_ROOT)
    parser.add_argument("--resume-from", type=Path, default=None,
                        help="resume from a prior Stage07 manifest; PASS branches are reused read-only")
    parser.add_argument("--stage-status", type=Path, default=None)
    args = parser.parse_args()
    if args.worker:
        if not args.event_id or not args.branch or args.output is None:
            raise SystemExit("--worker requires --event-id, --branch and --output")
        return worker_main(args)
    return orchestrator_main(args)


if __name__ == "__main__":
    raise SystemExit(main())
