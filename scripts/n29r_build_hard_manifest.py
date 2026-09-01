#!/usr/bin/env python3
"""Freeze hard, current-time-triggered N29-R correction episodes.

This pass deliberately runs only the frozen SAM3 anchor on DanceTrack train
sequences.  An episode is selected at the first sufficiently separated frame
where the currently visible target is missing or has box IoU < 0.5.  Future
frames are used only to check that the requested query horizon exists; future
GT is never inspected while deciding whether a frame is a correction trigger.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Optional

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.n29_lit_online_replay import (  # noqa: E402
    _default_sequences,
    _image_files,
    _install_official_box_singleton,
    _iou,
    _make_backend,
    _read_gt,
    _select_observation,
    _session,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _visible_identities(gt: Mapping[int, Mapping[int, np.ndarray]]) -> list[tuple[int, int]]:
    first_seen: dict[int, int] = {}
    for frame in sorted(gt):
        for identity in sorted(gt[frame]):
            first_seen.setdefault(int(identity), int(frame))
    return sorted(first_seen.items(), key=lambda item: (item[1], item[0]))


def _prepare_singleton(
    backend: Any,
    *,
    sequence_dir: Path,
    initialized: bool,
    frame_idx: int,
    public_id: int,
    box_xyxy: np.ndarray,
) -> bool:
    if initialized:
        backend.reset_session()
    else:
        _session(backend, sequence_dir)
    backend.add_box(frame_idx, public_id, box_xyxy)
    _install_official_box_singleton(
        backend,
        frame_idx=frame_idx,
        public_id=public_id,
        box_xyxy=box_xyxy,
    )
    return True


def _current_trigger(
    *,
    frame: int,
    identity: int,
    gt: Mapping[int, Mapping[int, np.ndarray]],
    outputs: Mapping[int, list[Any]],
    public_id: int,
) -> Optional[dict[str, Any]]:
    target = gt.get(frame, {}).get(identity)
    if target is None:
        return None
    observation = _select_observation(outputs.get(frame, []), public_id, target)
    target_present = True
    prediction_present = observation is not None
    predicted_box = None if observation is None else np.asarray(observation.box_xyxy, dtype=float)
    current_iou = None if predicted_box is None else float(_iou(predicted_box, target))
    if not prediction_present:
        reason = "current_visible_target_missing_prediction"
    elif current_iou is not None and current_iou < 0.5:
        previous_target = gt.get(frame - 1, {}).get(identity)
        reason = (
            "occlusion_recovery_first_error"
            if previous_target is None and frame > 0
            else "current_box_iou_lt_0_5"
        )
    else:
        return None
    return {
        "reason": reason,
        "frame": int(frame),
        "target_present": target_present,
        "prediction_present": prediction_present,
        "current_box_iou": current_iou,
        "target_box": np.asarray(target, dtype=float).tolist(),
        "predicted_box": None if predicted_box is None else predicted_box.tolist(),
        "visible_neighbor_ids": [
            int(other)
            for other in sorted(gt.get(frame, {}))
            if int(other) != int(identity)
        ],
    }


def build_manifest(
    *,
    checkpoint: Path,
    dataset_root: Path,
    manifest_path: Path,
    sequence_limit: int,
    identities_per_sequence: int,
    episodes_per_identity: int,
    scan_frames: int,
    separation: int,
) -> dict[str, Any]:
    sequences = _default_sequences(manifest_path, dataset_root, sequence_limit)
    if not sequences:
        raise RuntimeError("no train-fold DanceTrack sequences found")
    backend = _make_backend(checkpoint)
    initialized = False
    episodes: list[dict[str, Any]] = []
    sequence_audits: list[dict[str, Any]] = []
    started = time.perf_counter()
    try:
        for sequence_index, sequence_dir in enumerate(sequences):
            if "val" in sequence_dir.parts or "test" in sequence_dir.parts:
                raise ValueError(f"train-only manifest refused {sequence_dir}")
            gt = _read_gt(sequence_dir)
            images = _image_files(sequence_dir)
            if not images:
                continue
            candidates = _visible_identities(gt)[:identities_per_sequence]
            sequence_episode_count = 0
            identity_audits: list[dict[str, Any]] = []
            for identity, initialization_frame in candidates:
                if initialization_frame + 5 >= len(images):
                    continue
                initialization_box = gt[initialization_frame][identity]
                public_id = 100000 + sequence_index * 1000 + int(identity)
                _prepare_singleton(
                    backend,
                    sequence_dir=sequence_dir,
                    initialized=initialized,
                    frame_idx=initialization_frame,
                    public_id=public_id,
                    box_xyxy=initialization_box,
                )
                initialized = True
                scan_end = min(len(images) - 1, initialization_frame + scan_frames - 1)
                anchor_outputs = backend.propagate(
                    initialization_frame,
                    scan_end,
                    start_frame_index=initialization_frame,
                )
                selected: list[dict[str, Any]] = []
                last_frame = initialization_frame - separation
                for frame in range(initialization_frame + 1, scan_end + 1):
                    trigger = _current_trigger(
                        frame=frame,
                        identity=identity,
                        gt=gt,
                        outputs=anchor_outputs,
                        public_id=public_id,
                    )
                    if trigger is None or frame - last_frame < separation:
                        continue
                    query_end = min(len(images) - 1, frame + 20)
                    if query_end < frame + 5:
                        continue
                    selected.append(trigger)
                    last_frame = frame
                    episode_id = f"{sequence_dir.name}:{identity}:{frame}"
                    episodes.append(
                        {
                            "episode_id": episode_id,
                            "sequence": sequence_dir.name,
                            "sequence_path": str(sequence_dir),
                            "split": "train/train_fold",
                            "dataset": "DanceTrack",
                            "dataset_identity": int(identity),
                            "public_id": int(public_id),
                            "sam_object_id": int(public_id),
                            "identity_binding": {
                                "dataset_identity": int(identity),
                                "public_id": int(public_id),
                                "sam_object_id": int(public_id),
                            },
                            "initialization_frame": int(initialization_frame),
                            "correction_frame": int(frame),
                            "correction_type": "box",
                            "correction_box": np.asarray(gt[frame][identity], dtype=float).tolist(),
                            "supervision_provenance": "BOX_DERIVED_PSEUDO_MASK",
                            "query_start": int(frame + 1),
                            "query_end": int(query_end),
                            "query_horizons": {
                                "5": [int(frame + 1), int(min(query_end, frame + 5))],
                                "10": [int(frame + 1), int(min(query_end, frame + 10))],
                                "20": [int(frame + 1), int(min(query_end, frame + 20))],
                            },
                            "current_trigger_reason": trigger["reason"],
                            "anchor_current_error": trigger,
                            "identity_neighbors": trigger["visible_neighbor_ids"],
                            "anchor_scan": {
                                "scan_start": int(initialization_frame),
                                "scan_end": int(scan_end),
                                "frames_observed": int(len(anchor_outputs)),
                                "selection_used_future_gt": False,
                            },
                        }
                    )
                    sequence_episode_count += 1
                    if len(selected) >= episodes_per_identity:
                        break
                identity_audits.append(
                    {
                        "dataset_identity": int(identity),
                        "initialization_frame": int(initialization_frame),
                        "anchor_frames_observed": int(len(anchor_outputs)),
                        "trigger_candidates_selected": len(selected),
                    }
                )
            sequence_audits.append(
                {
                    "sequence": sequence_dir.name,
                    "split": "train/train_fold",
                    "identities_scanned": identity_audits,
                    "episode_count": sequence_episode_count,
                }
            )
    finally:
        backend.close()
    return {
        "protocol": "N29-R3-HARD-MANIFEST",
        "status": "PASS" if episodes else "NOT_RUN",
        "val25_read": False,
        "test_labels_used": False,
        "selection_frozen_before_paired": True,
        "future_gt_used_for_selection": False,
        "selection_policy": {
            "allowed_current_triggers": [
                "current_visible_target_missing_prediction",
                "current_box_iou_lt_0_5",
                "occlusion_recovery_first_error",
            ],
            "minimum_frame_separation": int(separation),
            "selection_order": "first qualifying current-time triggers in causal frame order",
            "future_use": "query horizon availability only; query GT is withheld from selection",
        },
        "frozen_inputs": {
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": _sha256(checkpoint),
            "dataset_root": str(dataset_root),
            "manifest": str(manifest_path),
            "sequence_limit": int(sequence_limit),
            "identities_per_sequence": int(identities_per_sequence),
            "episodes_per_identity": int(episodes_per_identity),
            "scan_frames": int(scan_frames),
            "separation": int(separation),
        },
        "sequence_count": len(sequence_audits),
        "episode_count": len(episodes),
        "target_episode_count": "30-50; actual is frozen below",
        "sequence_audits": sequence_audits,
        "episodes": episodes,
        "elapsed_seconds": float(time.perf_counter() - started),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "checkpoints/sam3.1_mirror/sam3.1_multiplex.pt")
    parser.add_argument("--dataset-root", type=Path, default=Path("/path/to/dancetrack"))
    parser.add_argument("--manifest", type=Path, default=ROOT / "outputs/n27/dataset_split_manifest.json")
    parser.add_argument("--sequence-limit", type=int, default=10)
    parser.add_argument("--identities-per-sequence", type=int, default=3)
    parser.add_argument("--episodes-per-identity", type=int, default=2)
    parser.add_argument("--scan-frames", type=int, default=60)
    parser.add_argument("--separation", type=int, default=5)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/n29r/hard_episode_manifest.json")
    args = parser.parse_args()
    result = build_manifest(
        checkpoint=args.checkpoint,
        dataset_root=args.dataset_root,
        manifest_path=args.manifest,
        sequence_limit=args.sequence_limit,
        identities_per_sequence=args.identities_per_sequence,
        episodes_per_identity=args.episodes_per_identity,
        scan_frames=args.scan_frames,
        separation=args.separation,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({key: result[key] for key in ("status", "sequence_count", "episode_count", "elapsed_seconds", "val25_read")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
