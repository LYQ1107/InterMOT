#!/usr/bin/env python3
"""Build the sequence-disjoint N72R7 target-decoder training corpus.

The official N72R5 B0 candidate streams are sealed inputs.  This script only
adds offline supervision: a simulated current-frame human ROI feature is
extracted from the event image, while target labels are generated from the
train GT after the runtime candidate stream has been read.  The resulting
candidate/context tensors contain no GT or public-ID values and are the only
arrays consumed by the learned decoder.
"""

from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import traceback
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sam3_intermot.reacquisition.target_id_features import (  # noqa: E402
    CANDIDATE_FEATURE_DIM,
    CONTEXT_FEATURE_DIM,
    candidate_feature_vector,
    context_feature_vector,
)


EVENT_MANIFEST = ROOT / "outputs/N72R5/mechanism_rounds/round_06_event_policy/real_event_manifest.json"
SOURCE_ROOT = ROOT / "outputs/N72R5/mechanism_rounds/round_07_official_full_loop_attempt5/runtime/B0_NO_INTERVENTION"
DATA_ROOT = Path("/data1/LWR/vranlee/DATASETS/JDE/dancetrack")
MACHINE_CHECKPOINT = Path(
    "/data1/LWR/vranlee/SERVER_ONLY/avis/Interactive/SAM3_InterMOT/"
    "outputs/n9/checkpoints/osnet_x1_0_market1501.pth"
)
def resolve_root_path(value: str | None, default: Path) -> Path:
    if value is None:
        return default
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


TRAINING_ROOT = resolve_root_path(
    os.environ.get("N72R7_TRAINING_ROOT"),
    ROOT / "outputs/N72R7/training",
)
PROTOCOL_PATH = TRAINING_ROOT / "training_protocol.json"
MANIFEST_PATH = TRAINING_ROOT / "corpus_manifest.json"
STAGE_PATH = resolve_root_path(
    os.environ.get("N72R7_CORPUS_STATUS"),
    ROOT / "outputs/N72R7/stage_06_corpus_status.json",
)
HORIZON = 100
IOU_THRESHOLD = 0.50
SEED = 7202
CONTEXT_POLICY = os.environ.get("N72R7_CONTEXT_POLICY", "fixed_event_box")

# This split is frozen before reading any post-treatment result.  The two
# confirmation sequences are deliberately untouched by corpus construction;
# N72R7 development events overlap the train/validation pools by design and
# therefore cannot be called independent confirmation.
TRAIN_SEQUENCES = (
    "dancetrack0001", "dancetrack0002", "dancetrack0006", "dancetrack0008",
    "dancetrack0012", "dancetrack0015", "dancetrack0016", "dancetrack0023",
    "dancetrack0024", "dancetrack0027", "dancetrack0029", "dancetrack0032",
    "dancetrack0033", "dancetrack0037", "dancetrack0055", "dancetrack0062",
)
# De-duplicate the explicit tuple while preserving its visible protocol form.
TRAIN_SEQUENCES = tuple(dict.fromkeys(TRAIN_SEQUENCES))
VALIDATION_SEQUENCES = ("dancetrack0051", "dancetrack0052")
CONFIRMATION_SEQUENCES = ("dancetrack0020", "dancetrack0049")


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        directory_fd = os.open(path.parent, os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    atomic_write(path, json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n")


def atomic_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    atomic_write(
        path,
        "".join(json.dumps(dict(row), ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n" for row in rows),
    )


def atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".npz", dir=str(path.parent))
    os.close(fd)
    try:
        np.savez_compressed(temporary, **arrays)
        with open(temporary, "rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        directory_fd = os.open(path.parent, os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object: {path}")
    return value


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def box_iou(left: Sequence[float], right: Sequence[float]) -> float:
    a = np.asarray(left, dtype=np.float64).reshape(-1)
    b = np.asarray(right, dtype=np.float64).reshape(-1)
    if a.size != 4 or b.size != 4 or not np.all(np.isfinite(a)) or not np.all(np.isfinite(b)):
        return 0.0
    if a[2] <= a[0] or a[3] <= a[1] or b[2] <= b[0] or b[3] <= b[1]:
        return 0.0
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    intersection = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    union = area_a + area_b - intersection
    return float(intersection / union) if union > 0.0 else 0.0


def load_gt(sequence: str) -> dict[int, dict[int, list[float]]]:
    path = DATA_ROOT / "train" / sequence / "gt" / "gt.txt"
    if not path.is_file():
        raise FileNotFoundError(path)
    result: dict[int, dict[int, list[float]]] = {}
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            parts = [item.strip() for item in line.split(",")]
            if len(parts) < 6:
                raise ValueError(f"malformed GT row {path}:{line_number}")
            frame = int(parts[0]) - 1
            identity = int(parts[1])
            x, y, width, height = [float(item) for item in parts[2:6]]
            box = [x, y, x + width, y + height]
            if not np.all(np.isfinite(np.asarray(box, dtype=np.float64))):
                raise ValueError(f"non-finite GT box {path}:{line_number}")
            result.setdefault(frame, {})[identity] = box
    return result


class FrozenROIEncoder:
    """The already frozen 512-D OSNet ROI path used by N72R6 anchors."""

    feature_dim = 512

    def __init__(self, device: str) -> None:
        if not MACHINE_CHECKPOINT.is_file():
            raise FileNotFoundError(MACHINE_CHECKPOINT)
        import torch
        from torchreid.reid.utils.feature_extractor import FeatureExtractor

        self.torch = torch
        self.extractor = FeatureExtractor(
            model_name="osnet_x1_0",
            model_path=str(MACHINE_CHECKPOINT),
            image_size=(256, 128),
            device=device,
            verbose=False,
        )

    @staticmethod
    def _crop(image: Image.Image, box: Sequence[float]) -> np.ndarray:
        x1, y1, x2, y2 = [int(round(float(value))) for value in box]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(image.width, x2), min(image.height, y2)
        if x2 <= x1 or y2 <= y1:
            return np.zeros((8, 8, 3), dtype=np.uint8)
        return np.asarray(image.crop((x1, y1, x2, y2)), dtype=np.uint8)

    def encode(self, image_path: Path, box: Sequence[float]) -> np.ndarray:
        with Image.open(image_path) as handle:
            image = handle.convert("RGB")
            crop = self._crop(image, box)
        with self.torch.no_grad():
            value = self.extractor([crop]).detach().float().cpu().numpy()
        feature = np.asarray(value, dtype=np.float32).reshape(-1)
        if feature.size != self.feature_dim or not np.all(np.isfinite(feature)):
            raise RuntimeError(f"invalid ROI feature from {image_path}: {feature.shape}")
        norm = float(np.linalg.norm(feature))
        if norm <= 1.0e-6:
            raise RuntimeError(f"zero ROI feature from {image_path}")
        return feature / norm


def frozen_events() -> list[dict[str, Any]]:
    policy = read_json(EVENT_MANIFEST)
    if policy.get("status") != "PASS_N72R5_EVENT_POLICY_FROZEN":
        raise RuntimeError(f"N72R5 event policy is not frozen PASS: {policy.get('status')}")
    events = [dict(item) for item in policy.get("events", [])]
    if len(events) != 40 or len({str(item.get("event_id")) for item in events}) != 40:
        raise RuntimeError("N72R5 training source must contain exactly 40 unique events")
    if policy.get("selection_uses_future_metrics") is not False:
        raise RuntimeError("event policy does not prove future-metric-free selection")
    return sorted(events, key=lambda item: str(item["event_id"]))


def source_done_index() -> dict[str, dict[str, Any]]:
    index: dict[str, dict[str, Any]] = {}
    for path in sorted(SOURCE_ROOT.glob("*.done.json")):
        done = read_json(path)
        event_id = str(done.get("event_id"))
        if event_id in index:
            raise RuntimeError(f"duplicate source done event: {event_id}")
        if done.get("status") != "PASS_N72R5_OFFICIAL_FULL_LOOP_BRANCH":
            raise RuntimeError(f"source B0 is not PASS: {event_id}")
        artifact = Path(str(done["candidate_artifact"]))
        if not artifact.is_file() or sha256_file(artifact) != str(done["candidate_artifact_sha256"]):
            raise RuntimeError(f"source B0 artifact hash mismatch: {event_id}")
        index[event_id] = {"done": done, "done_path": path, "rows": read_jsonl(artifact)}
    if len(index) != 40:
        raise RuntimeError(f"expected 40 B0 source streams, found {len(index)}")
    return index


def write_protocol(events: Sequence[Mapping[str, Any]], source_index: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    sequences = sorted({str(item["sequence"]) for item in events})
    split_map = {}
    for sequence in sequences:
        if sequence in TRAIN_SEQUENCES:
            split_map[sequence] = "train"
        elif sequence in VALIDATION_SEQUENCES:
            split_map[sequence] = "validation"
        elif sequence in CONFIRMATION_SEQUENCES:
            split_map[sequence] = "confirmation_deferred"
        else:
            raise RuntimeError(f"sequence has no frozen split: {sequence}")
    body: dict[str, Any] = {
        "schema_version": "N72R7_HUMAN_CONDITIONED_DECODER_TRAINING_PROTOCOL_V1",
        "created_at_utc": now_utc(),
        "seed": SEED,
        "source_event_manifest": str(EVENT_MANIFEST),
        "source_event_manifest_sha256": sha256_file(EVENT_MANIFEST),
        "source_b0_root": str(SOURCE_ROOT),
        "source_event_count": len(events),
        "source_sequence_count": len(sequences),
        "sequence_split": {
            "train_sequences": list(TRAIN_SEQUENCES),
            "validation_sequences": list(VALIDATION_SEQUENCES),
            "confirmation_sequences": list(CONFIRMATION_SEQUENCES),
            "development_effect_sequences": sorted({str(item["sequence"]) for item in events if str(item["sequence"]) not in CONFIRMATION_SEQUENCES}),
            "map": split_map,
        },
        "label_protocol": {
            "visible_iou_threshold": IOU_THRESHOLD,
            "visible_label": "highest_iou_candidate_if_iou_at_least_0.50_else_NONE",
            "absent_label": "NONE",
            "gt_use": "offline_posthoc_training_label_generation_only",
            "candidate_features_exclude_gt": True,
            "candidate_features_exclude_public_id": True,
            "runtime_future_gt_used": False,
        },
        "human_anchor": {
            "source": "simulated_from_gt_current_frame_raw_image_roi",
            "feature_encoder": "frozen_osnet_x1_0_market1501",
            "checkpoint": str(MACHINE_CHECKPOINT),
            "checkpoint_sha256": sha256_file(MACHINE_CHECKPOINT),
            "not_real_human_evidence": True,
        },
        "decoder_input": {
            "candidate_feature_dim": CANDIDATE_FEATURE_DIM,
            "context_feature_dim": CONTEXT_FEATURE_DIM,
            "candidate_order": "frozen_source_order",
            "context_policy": CONTEXT_POLICY,
        },
        "forbidden_selection_fields": ["future_identity_error", "H20", "H50", "H100", "IDSW", "post_treatment_replay_result"],
        "artifact_source_hashes": {
            event_id: str(item["done"]["candidate_artifact_sha256"])
            for event_id, item in sorted(source_index.items())
        },
        "interaction_source": "simulated_from_gt",
        "real_human_evidence": False,
        "runtime_future_gt_used": False,
    }
    body["protocol_sha256"] = hashlib.sha256(
        json.dumps(body, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    ).hexdigest()
    atomic_json(PROTOCOL_PATH, body)
    return body


def best_target_label(
    candidates: Sequence[Mapping[str, Any]],
    target_box: Sequence[float] | None,
) -> tuple[int, str, float, bool]:
    if target_box is None:
        return len(candidates), "TARGET_NOT_VISIBLE", 0.0, False
    ious = [box_iou(item["box_xyxy"], target_box) for item in candidates]
    best = max(ious, default=0.0)
    if not ious or best < IOU_THRESHOLD:
        return len(candidates), "VISIBLE_NO_CANDIDATE_IOU_0.50", float(best), True
    index = max(range(len(ious)), key=lambda item: (ious[item], -item))
    return int(index), "HIGHEST_IOU_TARGET_CANDIDATE", float(best), True


def build_split(
    split: str,
    events: Sequence[Mapping[str, Any]],
    source_index: Mapping[str, Mapping[str, Any]],
    gt_by_sequence: Mapping[str, Mapping[int, Mapping[int, Sequence[float]]]],
    anchors: Mapping[str, np.ndarray],
    dimensions: Mapping[str, tuple[int, int]],
) -> dict[str, Any]:
    selected = [item for item in events if item["sequence_split"] == split]
    if not selected:
        raise RuntimeError(f"empty {split} event split")
    examples: list[np.ndarray] = []
    contexts: list[np.ndarray] = []
    labels: list[int] = []
    candidate_counts: list[int] = []
    candidate_masks: list[np.ndarray] = []
    metadata: list[dict[str, Any]] = []
    max_candidates = 0
    label_counts: Counter[str] = Counter()
    for event in selected:
        event_id = str(event["event_id"])
        sequence = str(event["sequence"])
        source = source_index[event_id]
        rows = source["rows"]
        event_frame = int(event["event_frame"])
        expected = list(range(event_frame, event_frame + HORIZON + 1))
        if [int(row["frame"]) for row in rows] != expected:
            raise RuntimeError(f"source frame axis mismatch: {event_id}")
        anchor_box = [float(value) for value in event["current_gt_box"]]
        anchor = anchors[event_id]
        width, height = dimensions[event_id]
        target_id = int(event["dataset_gt_id"])
        event_candidates = list(rows[0].get("candidates", []))
        if CONTEXT_POLICY not in {"fixed_event_box", "causal_motion_anchor_update_v1"}:
            raise ValueError(f"unknown decoder corpus context policy: {CONTEXT_POLICY}")
        raw_candidates = [item.get("official_raw_sam_id") for item in event_candidates if item.get("official_raw_sam_id") is not None]
        event_raw = int(raw_candidates[0]) if raw_candidates and CONTEXT_POLICY == "fixed_event_box" else None
        event_scope = None
        for item in event_candidates:
            if item.get("official_raw_sam_id") == event_raw:
                event_scope = item.get("native_scope", item.get("native_tid_scope"))
                break
        predicted_box = list(anchor_box)
        velocity = np.zeros(2, dtype=np.float64)
        previous_raw = event_raw
        previous_scope = event_scope
        trusted_count = 0
        for row in rows[1:]:
            frame = int(row["frame"])
            candidates = list(row.get("candidates", []))
            if len(candidates) > 64:
                raise RuntimeError(f"unexpected candidate set size: {event_id}:{frame}")
            target_box = gt_by_sequence[sequence].get(frame, {}).get(target_id)
            label, reason, best_iou, visible = best_target_label(candidates, target_box)
            label_counts[reason] += 1
            candidate_vectors = np.stack(
                [
                    candidate_feature_vector(
                        candidate,
                        anchor_feature=anchor,
                        anchor_box=anchor_box,
                        predicted_box=predicted_box,
                        previous_raw_sam_id=previous_raw,
                        previous_native_scope=previous_scope,
                        image_width=width,
                        image_height=height,
                        candidate_count=len(candidates),
                        base_target_score=None,
                    )
                    for candidate in candidates
                ],
                axis=0,
            ) if candidates else np.zeros((0, CANDIDATE_FEATURE_DIM), dtype=np.float32)
            context = context_feature_vector(
                anchor_feature=anchor,
                predicted_box=predicted_box,
                anchor_box=anchor_box,
                velocity=velocity,
                previous_raw_sam_id=previous_raw,
                frame=frame,
                event_frame=event_frame,
                trusted_count=trusted_count,
                image_width=width,
                image_height=height,
            )
            examples.append(candidate_vectors)
            contexts.append(context)
            labels.append(int(label))
            candidate_counts.append(len(candidates))
            candidate_masks.append(np.ones(len(candidates), dtype=np.bool_))
            max_candidates = max(max_candidates, len(candidates))
            metadata.append({
                "event_id": event_id,
                "sequence": sequence,
                "action_type": str(event["action_type"]),
                "event_frame": event_frame,
                "frame": frame,
                "frame_horizon": frame - event_frame,
                "candidate_uids": [str(item.get("candidate_uid")) for item in candidates],
                "candidate_feature_sha256": [str(item.get("feature_sha256")) for item in candidates],
                "label_index": int(label),
                "label_kind": "NONE" if label == len(candidates) else "TARGET_CANDIDATE",
                "label_reason": reason,
                "posthoc_target_visible": bool(visible),
                "posthoc_best_iou": float(best_iou),
                "gt_used_offline": True,
                "runtime_future_gt_used": False,
                "public_id_inference": False,
                "not_real_human_evidence": True,
                "context_policy": CONTEXT_POLICY,
            })
            if CONTEXT_POLICY == "causal_motion_anchor_update_v1" and candidates:
                # A teacher-free, causal context update.  It uses only the
                # current sealed candidate set, the fixed event anchor and
                # the previous causal state; it never consults the offline
                # target label that was just written to metadata.
                anchor_vector = np.asarray(anchor, dtype=np.float32)
                baseline_scores: list[float] = []
                for candidate in candidates:
                    value = np.asarray(candidate.get("feature", []), dtype=np.float32).reshape(-1)
                    if value.size == 512 and np.all(np.isfinite(value)):
                        norm = float(np.linalg.norm(value))
                        similarity = float(np.dot(value / max(norm, 1.0e-6), anchor_vector)) if norm > 1.0e-6 else -1.0
                    else:
                        similarity = -1.0
                    geometry = box_iou(candidate["box_xyxy"], predicted_box)
                    presence = float(np.clip(float(candidate.get("presence_score", candidate.get("confidence", 0.0))), 0.0, 1.0))
                    raw = candidate.get("official_raw_sam_id")
                    scope = candidate.get("native_scope", candidate.get("native_tid_scope"))
                    continuity = float(previous_raw is not None and raw is not None and int(raw) == int(previous_raw) and previous_scope is not None and scope is not None and str(scope) == str(previous_scope))
                    baseline_scores.append(similarity + 0.75 * geometry + 0.25 * presence + 0.25 * continuity)
                best_index = max(range(len(candidates)), key=lambda index: (baseline_scores[index], -index))
                chosen = candidates[best_index]
                old_center = np.asarray([(predicted_box[0] + predicted_box[2]) / 2.0, (predicted_box[1] + predicted_box[3]) / 2.0], dtype=np.float64)
                chosen_box = [float(value) for value in chosen["box_xyxy"]]
                new_center = np.asarray([(chosen_box[0] + chosen_box[2]) / 2.0, (chosen_box[1] + chosen_box[3]) / 2.0], dtype=np.float64)
                velocity = 0.5 * velocity + 0.5 * (new_center - old_center)
                predicted_box = chosen_box
                previous_raw = None if chosen.get("official_raw_sam_id") is None else int(chosen["official_raw_sam_id"])
                previous_scope = chosen.get("native_scope", chosen.get("native_tid_scope"))
                trusted_count = min(trusted_count + 1, 3)
    padded = np.zeros((len(examples), max_candidates, CANDIDATE_FEATURE_DIM), dtype=np.float32)
    mask = np.zeros((len(examples), max_candidates), dtype=np.bool_)
    for index, value in enumerate(examples):
        if len(value):
            padded[index, : len(value)] = value
            mask[index, : len(value)] = candidate_masks[index]
    output_npz = TRAINING_ROOT / f"{split}.npz"
    output_meta = TRAINING_ROOT / f"{split}_metadata.jsonl"
    atomic_npz(
        output_npz,
        candidate_features=padded,
        candidate_mask=mask,
        context_features=np.stack(contexts, axis=0).astype(np.float32),
        # The decoder has one padded candidate axis per split.  Every NONE
        # label must point to the single shared NONE class, not to the first
        # padded slot after that example's shorter candidate set.
        labels=np.asarray(
            [max_candidates if label == count else label for label, count in zip(labels, candidate_counts)],
            dtype=np.int64,
        ),
        candidate_counts=np.asarray(candidate_counts, dtype=np.int64),
    )
    atomic_jsonl(output_meta, metadata)
    return {
        "split": split,
        "event_count": len(selected),
        "sequence_count": len({str(item["sequence"]) for item in selected}),
        "example_count": len(examples),
        "max_candidates": max_candidates,
        "candidate_feature_dim": CANDIDATE_FEATURE_DIM,
        "context_feature_dim": CONTEXT_FEATURE_DIM,
        "label_counts": dict(sorted(label_counts.items())),
        "npz": str(output_npz),
        "npz_sha256": sha256_file(output_npz),
        "metadata": str(output_meta),
        "metadata_sha256": sha256_file(output_meta),
        "gt_labels_only_offline": True,
        "runtime_future_gt_used": False,
    }


def main() -> None:
    result: dict[str, Any] = {
        "schema_version": "N72R7_TRAINING_CORPUS_STATUS_V1",
        "status": "FAIL",
        "started_at_utc": now_utc(),
    }
    try:
        events = frozen_events()
        source_index = source_done_index()
        if set(source_index) != {str(item["event_id"]) for item in events}:
            raise RuntimeError("frozen policy/source event IDs do not match")
        protocol = write_protocol(events, source_index)
        for event in events:
            event["sequence_split"] = protocol["sequence_split"]["map"][str(event["sequence"])]
        if set(item["sequence_split"] for item in events) != {"train", "validation", "confirmation_deferred"}:
            raise RuntimeError("all three frozen sequence splits are not represented")
        gt_by_sequence = {sequence: load_gt(sequence) for sequence in sorted({str(item["sequence"]) for item in events})}
        device = os.environ.get("N72R7_ANCHOR_DEVICE", "cpu")
        encoder = FrozenROIEncoder(device)
        anchors: dict[str, np.ndarray] = {}
        dimensions: dict[str, tuple[int, int]] = {}
        anchor_sources: Counter[str] = Counter()
        for event in events:
            event_id = str(event["event_id"])
            image_path = DATA_ROOT / "train" / str(event["sequence"]) / "img1" / f"{int(event['event_frame']) + 1:08d}.jpg"
            if not image_path.is_file():
                raise FileNotFoundError(image_path)
            with Image.open(image_path) as image:
                dimensions[event_id] = (int(image.width), int(image.height))
            anchors[event_id] = encoder.encode(image_path, event["current_gt_box"])
            anchor_sources["simulated_from_gt_current_frame_raw_image_roi"] += 1
        split_manifests = {
            split: build_split(split, events, source_index, gt_by_sequence, anchors, dimensions)
            for split in ("train", "validation", "confirmation_deferred")
        }
        corpus = {
            "schema_version": "N72R7_HUMAN_CONDITIONED_DECODER_CORPUS_V1",
            "status": "PASS_TRAINING_CORPUS_SEALED",
            "created_at_utc": now_utc(),
            "protocol": str(PROTOCOL_PATH),
            "protocol_sha256": sha256_file(PROTOCOL_PATH),
            "events": len(events),
            "sequences": sorted({str(item["sequence"]) for item in events}),
            "anchor_sources": dict(anchor_sources),
            "splits": split_manifests,
            "runtime_future_gt_used": False,
            "gt_used_only_offline_label_generation": True,
            "public_id_inference": False,
            "interaction_source": "simulated_from_gt",
            "not_real_human_evidence": True,
        }
        atomic_json(MANIFEST_PATH, corpus)
        result.update({
            "status": "PASS_TRAINING_CORPUS_SEALED",
            "protocol": str(PROTOCOL_PATH),
            "protocol_sha256": sha256_file(PROTOCOL_PATH),
            "corpus_manifest": str(MANIFEST_PATH),
            "corpus_manifest_sha256": sha256_file(MANIFEST_PATH),
            "device": device,
            "split_manifests": split_manifests,
            "anchor_source": dict(anchor_sources),
            "runtime_future_gt_used": False,
            "finished_at_utc": now_utc(),
        })
        atomic_json(STAGE_PATH, result)
        print(json.dumps({"status": result["status"], "train_examples": split_manifests["train"]["example_count"], "validation_examples": split_manifests["validation"]["example_count"], "confirmation_examples": split_manifests["confirmation_deferred"]["example_count"]}, sort_keys=True))
    except Exception as exc:
        result.update({"failure_root_cause": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc(), "finished_at_utc": now_utc()})
        failure = ROOT / "outputs/N72R7/attempts" / f"n72r7_training_corpus_failure_{datetime.now().strftime('%Y%m%dT%H%M%S')}.json"
        atomic_json(failure, result)
        raise


if __name__ == "__main__":
    main()
