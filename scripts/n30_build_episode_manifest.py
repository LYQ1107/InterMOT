#!/usr/bin/env python3
"""Freeze sequence-disjoint N30-C/D train-fold episodic data coordinates."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TRAIN_ROOT = Path("/path/to/dancetrack/train")


def _read_gt(sequence_dir: Path) -> dict[int, dict[int, np.ndarray]]:
    result: dict[int, dict[int, np.ndarray]] = {}
    for line in (sequence_dir / "gt" / "gt.txt").read_text(encoding="utf-8").splitlines():
        fields = line.strip().split(",")
        if len(fields) < 6:
            continue
        frame = int(fields[0]) - 1
        identity = int(fields[1])
        x, y, w, h = (float(fields[index]) for index in range(2, 6))
        result.setdefault(frame, {})[identity] = np.asarray([x, y, x + w, y + h], dtype=float)
    return result


def _images(sequence_dir: Path) -> list[Path]:
    return sorted((sequence_dir / "img1").glob("*.jpg"), key=lambda path: int(path.stem))


def _iou(left: np.ndarray, right: np.ndarray) -> float:
    x1 = max(float(left[0]), float(right[0]))
    y1 = max(float(left[1]), float(right[1]))
    x2 = min(float(left[2]), float(right[2]))
    y2 = min(float(left[3]), float(right[3]))
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_left = max(0.0, float(left[2] - left[0])) * max(0.0, float(left[3] - left[1]))
    area_right = max(0.0, float(right[2] - right[0])) * max(0.0, float(right[3] - right[1]))
    union = area_left + area_right - intersection
    return intersection / union if union > 0 else 0.0


def _distort(box: np.ndarray, width: int, height: int) -> tuple[np.ndarray, float]:
    box = np.asarray(box, dtype=float)
    bw = max(1.0, float(box[2] - box[0]))
    bh = max(1.0, float(box[3] - box[1]))
    candidates = (
        np.asarray([box[0] + 0.60 * bw, box[1] + 0.10 * bh, box[2] + 0.60 * bw, box[3] + 0.10 * bh]),
        np.asarray([box[0] - 0.60 * bw, box[1] - 0.10 * bh, box[2] - 0.60 * bw, box[3] - 0.10 * bh]),
        np.asarray([box[0] + 0.35 * bw, box[1] + 0.45 * bh, box[2] + 0.35 * bw, box[3] + 0.45 * bh]),
    )
    for candidate in candidates:
        candidate = candidate.copy()
        candidate[[0, 2]] = np.clip(candidate[[0, 2]], 0, width - 1)
        candidate[[1, 3]] = np.clip(candidate[[1, 3]], 0, height - 1)
        if candidate[2] <= candidate[0]:
            candidate[2] = min(float(width), candidate[0] + max(2.0, bw * 0.25))
        if candidate[3] <= candidate[1]:
            candidate[3] = min(float(height), candidate[1] + max(2.0, bh * 0.25))
        value = _iou(candidate, box)
        if value < 0.5:
            return candidate, value
    return candidates[0], _iou(candidates[0], box)


def _find_episode(gt: dict[int, dict[int, np.ndarray]], image_count: int, sequence: str, index: int, split: str) -> dict[str, Any] | None:
    # N30-D freezes the same H20 future window required by the final benefit gate.
    # The interaction is at start+3, so the last query frame is start+23.
    for start in range(0, image_count - 23):
        interaction = start + 3
        query_frames = list(range(interaction + 1, interaction + 21))
        if start not in gt or interaction not in gt or any(frame not in gt for frame in query_frames):
            continue
        common = set(gt[start]) & set(gt[interaction])
        common = {identity for identity in common if all(identity in gt[frame] for frame in query_frames)}
        if not common:
            continue
        identity = sorted(common)[index % len(common)]
        protected = sorted(
            identity_value
            for identity_value in gt[start]
            if identity_value != identity
            and identity_value in gt[interaction]
            and all(identity_value in gt[frame] for frame in query_frames)
        )[:2]
        image = _images(DEFAULT_TRAIN_ROOT / sequence)[0]
        from PIL import Image

        with Image.open(image) as handle:
            width, height = handle.size
        start_box = np.asarray(gt[start][identity], dtype=float)
        anchor_box, error_iou = _distort(start_box, width, height)
        query_targets = {
            str(frame): np.asarray(gt[frame][identity], dtype=float).tolist()
            for frame in query_frames
        }
        return {
            "episode_id": f"n30_{split}_{sequence}_{start:04d}_{identity}",
            "role": split,
            "dataset": "DanceTrack",
            "split": "train/train_fold",
            "parent_sequence": sequence,
            "video_id": sequence,
            "sequence_path": str(DEFAULT_TRAIN_ROOT / sequence),
            "dataset_identity": int(identity),
            "public_id": int(100000 + identity),
            "initialization_frame": int(start),
            "correction_frame": int(interaction),
            "query_frames": query_frames,
            "query_end": int(query_frames[-1]),
            "anchor_box": anchor_box.tolist(),
            "anchor_current_error_iou": float(error_iou),
            "correction_box": np.asarray(gt[interaction][identity], dtype=float).tolist(),
            "protected_identity_ids": [int(value) for value in protected],
            "query_target_boxes": query_targets,
            "action": "BOX_CORRECTION",
            "correction_provenance": "BOX_DERIVED_PSEUDO_MASK",
            "selection_rule": "first legal frame window in lexicographic train-fold sequence order; identity is deterministic sorted first/indexed identity",
            "future_gt_used_for_selection": False,
            "gt_role": "visibility validation and offline query supervision only",
            "image_size": [int(height), int(width)],
        }
    return None


def build(train_root: Path, output: Path, protocol_output: Path, meta_train_count: int, selection_count: int, calibration_count: int) -> dict[str, Any]:
    if train_root != DEFAULT_TRAIN_ROOT:
        raise ValueError("N30-D is pinned to the audited DanceTrack train/train_fold root")
    requested = meta_train_count + selection_count + calibration_count
    sequences = sorted(path.name for path in train_root.iterdir() if path.is_dir() and path.name.startswith("dancetrack"))
    episodes: list[dict[str, Any]] = []
    used_sequences: list[str] = []
    for sequence in sequences:
        if len(used_sequences) >= requested:
            break
        sequence_dir = train_root / sequence
        images = _images(sequence_dir)
        gt = _read_gt(sequence_dir)
        split = (
            "meta_train" if len(used_sequences) < meta_train_count else
            "selection" if len(used_sequences) < meta_train_count + selection_count else
            "calibration"
        )
        episode = _find_episode(gt, len(images), sequence, len(used_sequences), split)
        if episode is None:
            continue
        used_sequences.append(sequence)
        episodes.append(episode)
    if len(episodes) != requested:
        raise RuntimeError(f"only found {len(episodes)} legal sequence-disjoint episodes, expected {requested}")
    payload: dict[str, Any] = {
        "protocol": "N30-C/D-SEQUENCE-DISJOINT-FUTURE-EPISODES",
        "status": "PASS",
        "dataset": "DanceTrack",
        "train_root": str(train_root),
        "split": "train/train_fold",
        "val25_read": False,
        "test_labels_used": False,
        "future_gt_used_for_selection": False,
        "sequence_disjoint": True,
        "split_counts": {
            "meta_train": meta_train_count,
            "selection": selection_count,
            "calibration": calibration_count,
        },
        "split_sequences": {
            "meta_train": used_sequences[:meta_train_count],
            "selection": used_sequences[meta_train_count:meta_train_count + selection_count],
            "calibration": used_sequences[meta_train_count + selection_count:],
        },
        "selection_rule": "lexicographic train-fold parent sequence order, first legal 20-frame future window after a fixed 3-frame support prefix, deterministic identity choice; no future metric used to choose cases",
        "episodes": episodes,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(output)
    digest = hashlib.sha256(output.read_bytes()).hexdigest()
    payload["manifest_sha256"] = digest
    protocol = {
        "protocol": "N30-C/D-FROZEN-SPLIT",
        "status": "PASS",
        "manifest": str(output),
        "manifest_sha256": digest,
        "parent_sequence_disjoint": True,
        "split_sequences": payload["split_sequences"],
        "val25_read": False,
        "test_labels_used": False,
        "future_gt_used_for_selection": False,
        "selection_metric": "sequence-level future-delivered-box-IoU gain over the same write-only correction",
        "selection_metric_frozen_before_training": True,
    }
    protocol_output.parent.mkdir(parents=True, exist_ok=True)
    protocol_output.write_text(json.dumps(protocol, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-root", type=Path, default=DEFAULT_TRAIN_ROOT)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/n30/episode_manifest.json")
    parser.add_argument("--protocol-output", type=Path, default=ROOT / "outputs/n30/frozen_protocol.json")
    parser.add_argument("--meta-train-count", type=int, default=20)
    parser.add_argument("--selection-count", type=int, default=4)
    parser.add_argument("--calibration-count", type=int, default=4)
    args = parser.parse_args()
    payload = build(args.train_root, args.output, args.protocol_output, args.meta_train_count, args.selection_count, args.calibration_count)
    print(json.dumps({"status": payload["status"], "episode_count": len(payload["episodes"]), "manifest_sha256": payload["manifest_sha256"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
