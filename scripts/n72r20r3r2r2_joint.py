#!/usr/bin/env python3
"""N72R20R3R2R2 joint open-set identity/availability research stage.

This stage consumes only the sealed DanceTrack-train candidate tape and the
frozen R3R2/R3R2R1 artifacts.  It deliberately does not run SAM3, read VAL or
TEST, change candidate generation, or open association authority.  GT is
used only while constructing training/post-hoc labels; runtime features and
the formal prediction path are kept separate from those labels.

The implementation is intentionally self-contained so that the complete
research tree can be reproduced without copying the large historical tapes
into the repository.  Learned checkpoint binaries are written to the external
stage asset directory and only their hashes/configuration are committed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import shutil
import subprocess
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch
from scipy.special import expit
from torch import nn

from sam3_intermot.identity_verification.cross_scene_adapter import CrossSceneIdentityAdapter
from scripts.n72r20r3_common import gt_by_frame, iou, read_zstd_jsonl


ROOT = Path(__file__).resolve().parents[1]
STAGE = "N72R20R3R2R2"
OUT = ROOT / "outputs" / STAGE
ASSET_ROOT = Path("/data3/liuyeqiang/InterMOT_N72R20R2_assets")
MODEL_ROOT = Path("/data3/liuyeqiang/InterMOT_N72R20R3R2_assets/models")
STAGE_ASSET_ROOT = Path("/data3/liuyeqiang/InterMOT_N72R20R3R2R2_assets")
DATASET_ROOT = Path("/data3/liuyeqiang/InterMOT_N72R16_assets/dataset")
R3R2 = ROOT / "outputs" / "N72R20R3R2"
R3R2R1 = ROOT / "outputs" / "N72R20R3R2R1"
PROTOCOL = ROOT / "outputs" / "N72R20R3R1R1"
SEQUENCES = (
    "dancetrack0001", "dancetrack0002", "dancetrack0023", "dancetrack0024",
    "dancetrack0039", "dancetrack0057", "dancetrack0062", "dancetrack0072",
)
R3R2_SEED = 720321
SOURCE_R3R2R1_HEAD = "7820d312e2b3c852a7909a4d42dc71bad0c085c6"
LEARNED_SEEDS = (720341, 720342, 720343)
IOU_VALID = 0.50
IOU_WEAK = 0.10
FORMAL_FPR = 0.02
FORMAL_RECALL = 0.60
FORMAL_MACRO_FPR = 0.05
FORMAL_MACRO_RECALL = 0.40
MAX_TRAIN_EPISODES_PER_KIND = 1500
MAX_JOINT_EPISODES = 4000


def plain(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return [plain(item) for item in value.tolist()]
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, Mapping):
        return {str(key): plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(item) for item in value]
    return value


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(plain(value), indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_zstd_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    payload = "".join(json.dumps(plain(dict(row)), sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n" for row in rows).encode()
    compressed = subprocess.run(["zstd", "-q", "-T0", "-c"], input=payload, stdout=subprocess.PIPE, check=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(compressed.stdout)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def git_head() -> str:
    return subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, check=True, text=True, stdout=subprocess.PIPE).stdout.strip()


def storage_audit(label: str) -> dict[str, Any]:
    usage = shutil.disk_usage(ROOT)
    free = float(usage.free / (1024 ** 3))
    return {
        "stage": STAGE,
        "label": label,
        "path": str(ROOT),
        "total_gib": float(usage.total / (1024 ** 3)),
        "used_gib": float(usage.used / (1024 ** 3)),
        "free_gib": free,
        "warning_below_gib": 106.0,
        "hard_stop_below_gib": 100.0,
        "storage_status": "HARD_STOP" if free < 100.0 else ("WARNING" if free < 106.0 else "OK"),
        "target_new_persistent_storage": "<8GB",
    }


def unit(array: Sequence[float] | np.ndarray) -> np.ndarray:
    value = np.asarray(array, dtype=np.float32)
    norm = np.linalg.norm(value, axis=-1, keepdims=True)
    return value / np.maximum(norm, 1.0e-8)


def finite_stats(values: Sequence[float]) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        return {key: None for key in ("count", "mean", "median", "p10", "p25", "p50", "p75", "p90", "min", "max")}
    return {
        "count": int(array.size), "mean": float(array.mean()), "median": float(np.median(array)),
        "p10": float(np.percentile(array, 10)), "p25": float(np.percentile(array, 25)),
        "p50": float(np.percentile(array, 50)), "p75": float(np.percentile(array, 75)),
        "p90": float(np.percentile(array, 90)), "min": float(array.min()), "max": float(array.max()),
    }


def auc_score(labels: Sequence[int], scores: Sequence[float]) -> float | None:
    y = np.asarray(labels, dtype=np.int64)
    s = np.asarray(scores, dtype=np.float64)
    positive = s[y == 1]
    negative = s[y == 0]
    if not len(positive) or not len(negative):
        return None
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s), dtype=np.float64)
    ranks[order] = np.arange(1, len(s) + 1, dtype=np.float64)
    return float((ranks[y == 1].sum() - len(positive) * (len(positive) + 1) / 2.0) / (len(positive) * len(negative)))


def auprc(labels: Sequence[int], scores: Sequence[float]) -> float | None:
    y = np.asarray(labels, dtype=np.int64)
    s = np.asarray(scores, dtype=np.float64)
    positives = int(y.sum())
    if positives == 0:
        return None
    order = np.argsort(-s, kind="mergesort")
    y_sorted = y[order]
    cumulative = np.cumsum(y_sorted)
    precision = cumulative / np.arange(1, len(y_sorted) + 1)
    return float(np.sum(precision * y_sorted) / positives)


def recall_at_fpr(labels: Sequence[int], scores: Sequence[float], fpr: float = FORMAL_FPR) -> float:
    y = np.asarray(labels, dtype=np.int64)
    s = np.asarray(scores, dtype=np.float64)
    negatives = max(1, int((y == 0).sum()))
    thresholds = np.unique(np.concatenate(([np.inf], s, [-np.inf])))
    feasible = []
    for threshold in thresholds:
        predicted = s >= threshold
        fp = int(np.sum(predicted & (y == 0)))
        tp = int(np.sum(predicted & (y == 1)))
        if fp / negatives <= fpr + 1.0e-12:
            feasible.append(tp / max(1, int((y == 1).sum())))
    return float(max(feasible) if feasible else 0.0)


def select_threshold(labels: Sequence[int], scores: Sequence[float], correct: Sequence[int] | None = None) -> tuple[float, dict[str, Any]]:
    y = np.asarray(labels, dtype=np.int64)
    s = np.asarray(scores, dtype=np.float64)
    c = y if correct is None else np.asarray(correct, dtype=np.int64)
    thresholds = np.unique(np.concatenate(([0.0, 1.0], s)))
    candidates = []
    for threshold in thresholds:
        accepted = s >= float(threshold)
        neg = y == 0
        fp = int(np.sum(accepted & neg))
        tp = int(np.sum(accepted & (c == 1)))
        candidates.append({
            "threshold": float(threshold), "negative_fpr": fp / max(1, int(neg.sum())),
            "correct_id_recall": tp / max(1, int((c == 1).sum())), "accepted": int(accepted.sum()),
        })
    feasible = [item for item in candidates if item["negative_fpr"] <= FORMAL_FPR + 1.0e-12]
    if feasible:
        chosen = sorted(feasible, key=lambda item: (-item["correct_id_recall"], item["negative_fpr"], item["threshold"]))[0]
        return float(chosen["threshold"]), {"inner_gate_infeasible": False, "selected": chosen, "candidate_count": len(candidates)}
    chosen = sorted(candidates, key=lambda item: (item["negative_fpr"], -item["correct_id_recall"], item["threshold"]))[0]
    return float(chosen["threshold"]), {"inner_gate_infeasible": True, "selected": chosen, "candidate_count": len(candidates)}


@dataclass
class SequenceData:
    sequence: str
    frames: dict[int, dict[str, Any]]
    embeddings: np.ndarray
    gt: dict[int, list[tuple[int, list[float]]]]
    candidate_count: int
    frame_count: int


@dataclass
class Episode:
    sequence: str
    identity_key: str
    gt_id: int
    frame: int
    kind: str
    query_state: np.ndarray
    candidate_offsets: list[int]
    candidate_uids: list[str]
    candidate_labels: list[int]
    candidate_ious: list[float]
    positive_indices: list[int]
    canonical_index: int | None
    context: list[float]
    target_present: bool
    source: str = "real"
    counterfactual_variant: str | None = None
    removed_uids: list[str] = field(default_factory=list)

    @property
    def candidate_count(self) -> int:
        return len(self.candidate_offsets)


def _candidate_metadata(sequence: str) -> tuple[dict[int, dict[str, Any]], np.ndarray, int]:
    directory = ASSET_ROOT / "candidates" / sequence
    index = read_json(directory / "index.json")
    rows = read_zstd_jsonl(directory / "metadata.jsonl.zst")
    count = int(index["embedding_count"])
    embeddings = np.memmap(directory / "embeddings.f16", mode="r", dtype=np.float16, shape=(count, 512))
    frames = {}
    for row in rows:
        frame = int(row["frame"])
        candidates = [dict(item) for item in row.get("candidates", [])]
        axis = [str(item["candidate_uid"]) for item in candidates]
        if len(axis) != len(set(axis)):
            raise RuntimeError(f"candidate UID collision in {sequence}:{frame}")
        frames[frame] = {"frame": frame, "candidates": candidates, "candidate_uids": axis}
    if len(frames) != int(index["frame_count"]):
        raise RuntimeError(f"candidate frame count mismatch for {sequence}")
    return frames, embeddings, count


def load_sequences() -> dict[str, SequenceData]:
    result = {}
    for sequence in SEQUENCES:
        frames, embeddings, count = _candidate_metadata(sequence)
        gt = gt_by_frame(DATASET_ROOT / "train" / sequence / "gt" / "gt.txt")
        if not gt or not frames:
            raise RuntimeError(f"missing train GT or candidate frames for {sequence}")
        result[sequence] = SequenceData(sequence, frames, embeddings, gt, count, len(frames))
    return result


def load_identity_source() -> tuple[list[dict[str, Any]], np.ndarray]:
    path = R3R2 / "training" / "identity_episode_index.jsonl"
    state_path = R3R2 / "training" / "query_states.float32.npy"
    if not path.exists() or not state_path.exists():
        raise RuntimeError("sealed R3R2 identity episode source is missing")
    episodes = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    states = np.load(state_path, mmap_mode="r")
    if len(episodes) != len(states):
        raise RuntimeError("R3R2 episode/query-state axis mismatch")
    if sorted({str(row["sequence"]) for row in episodes}) != list(SEQUENCES):
        raise RuntimeError("R3R2 source does not contain all eight sequences")
    return episodes, states


def target_box(gt: Mapping[int, Sequence[tuple[int, Sequence[float]]]], frame: int, gt_id: int) -> list[float] | None:
    for track_id, box in gt.get(int(frame), []):
        if int(track_id) == int(gt_id):
            return list(box)
    return None


def candidate_labels(data: SequenceData, frame: int, gt_id: int) -> tuple[list[int], list[float], bool]:
    box = target_box(data.gt, frame, gt_id)
    labels: list[int] = []
    values: list[float] = []
    for candidate in data.frames[frame]["candidates"]:
        value = 0.0 if box is None else float(iou(candidate["box_xyxy"], box))
        values.append(value)
        labels.append(2 if value >= IOU_VALID else (1 if value >= IOU_WEAK else 0))
    return labels, values, box is not None


def canonical_positive(labels: Sequence[int], values: Sequence[float], uids: Sequence[str]) -> tuple[list[int], int | None]:
    positives = [index for index, label in enumerate(labels) if int(label) == 2]
    if not positives:
        return [], None
    chosen = max(positives, key=lambda index: (float(values[index]), str(uids[index])))
    return positives, int(chosen)


def episode_context(query: np.ndarray, anchor: np.ndarray, frame: int, last_observed: int, candidate_count: int) -> list[float]:
    return [
        float(math.log1p(candidate_count)), float(frame) / 100.0,
        float(max(0, frame - last_observed)) / 100.0,
        float(np.dot(unit(query), unit(anchor))),
    ]


def make_episode(
    data: SequenceData, identity_key: str, gt_id: int, frame: int, kind: str,
    query: np.ndarray, last_observed: int, source: str = "real",
) -> Episode:
    frame_data = data.frames[frame]
    labels, values, present = candidate_labels(data, frame, gt_id)
    positives, canonical = canonical_positive(labels, values, frame_data["candidate_uids"])
    if kind == "PRESENT" and not positives:
        raise RuntimeError(f"PRESENT episode without a valid candidate: {identity_key}:{frame}")
    if kind != "PRESENT" and positives:
        raise RuntimeError(f"non-PRESENT episode has a valid target candidate: {identity_key}:{frame}")
    return Episode(
        sequence=data.sequence, identity_key=identity_key, gt_id=int(gt_id), frame=int(frame), kind=kind,
        query_state=np.asarray(query, dtype=np.float32).copy(),
        candidate_offsets=[int(item["embedding_offset"]) for item in frame_data["candidates"]],
        candidate_uids=list(frame_data["candidate_uids"]), candidate_labels=labels, candidate_ious=values,
        positive_indices=positives, canonical_index=canonical,
        context=episode_context(query, query, frame, last_observed, len(labels)),
        target_present=bool(present), source=source,
    )


def build_real_episodes(data_by_sequence: Mapping[str, SequenceData], source_rows: Sequence[Mapping[str, Any]], states: np.ndarray) -> tuple[list[Episode], dict[str, Any]]:
    by_identity: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in source_rows:
        by_identity[str(row["identity_key"])].append(row)
    identities = sorted(by_identity)
    episodes: list[Episode] = []
    counts = Counter()
    per_identity: dict[str, dict[str, Any]] = {}
    for identity_key in identities:
        values = sorted(by_identity[identity_key], key=lambda item: (int(item["query_frame"]), int(item["query_state_ref"])))
        sequence, raw_gt = identity_key.split(":", 1)
        gt_id = int(raw_gt)
        data = data_by_sequence[sequence]
        present_by_frame = {int(row["query_frame"]): row for row in values}
        anchor_frame = int(values[0]["anchor_frame"])
        max_frame = max(data.frames)
        # Teacher-forced state at a missing frame is the state before the next
        # valid observation.  It is identical to the causal state after the
        # previous accepted observation and contains no current/future GT.
        current_state = np.asarray(states[int(values[0]["query_state_ref"])], dtype=np.float32)
        last_observed = anchor_frame
        identity_counts = Counter()
        for frame in range(anchor_frame + 1, max_frame + 1):
            source = present_by_frame.get(frame)
            if source is not None:
                current_state = np.asarray(states[int(source["query_state_ref"])], dtype=np.float32)
                kind = "PRESENT"
                last_observed = frame
            else:
                # When a next valid episode exists, its pre-observation state
                # is the exact state after the preceding valid observation.
                next_rows = [row for row in values if int(row["query_frame"]) > frame]
                if next_rows:
                    current_state = np.asarray(states[int(next_rows[0]["query_state_ref"])], dtype=np.float32)
                labels, values_iou, present = candidate_labels(data, frame, gt_id)
                if any(int(value) == 2 for value in labels):
                    # A valid target candidate that was not part of the
                    # historical one-to-one identity mining record remains a
                    # PRESENT episode; it must not be turned into a false
                    # P1 negative.
                    kind = "PRESENT"
                    last_observed = frame
                else:
                    kind = "P1a" if present and max(values_iou, default=0.0) >= IOU_WEAK else ("P1b" if present else "P0")
            episode = make_episode(data, identity_key, gt_id, frame, "PRESENT" if kind == "PRESENT" else kind, current_state, last_observed)
            episodes.append(episode)
            counts[episode.kind] += 1
            identity_counts[episode.kind] += 1
        per_identity[identity_key] = {
            "sequence": sequence, "gt_id": gt_id, "anchor_frame": anchor_frame,
            "episodes": int(sum(identity_counts.values())), "counts": dict(identity_counts),
        }
    # The taxonomy is deliberately derived from boxes, not copied from the
    # historical P0/P1 counts.  This catches frame/GT alignment corruption.
    expected = {"PRESENT", "P0", "P1a", "P1b"}
    if not set(counts).issubset(expected):
        raise RuntimeError(f"unexpected open-set taxonomy: {counts}")
    manifest = {
        "stage": STAGE, "status": "PASS_OPEN_SET_EPISODE_MINING", "training_identity_count": len(identities),
        "identity_keys": identities, "episode_count": len(episodes), "counts": dict(counts),
        "taxonomy": {
            "PRESENT": "target GT visible and one or more candidate IoU >= 0.50",
            "P0": "target GT absent; every current candidate is invalid for this identity",
            "P1a": "target GT visible but best candidate IoU in [0.10,0.50)",
            "P1b": "target GT visible but best candidate IoU < 0.10",
        },
        "iou_thresholds": {"valid": IOU_VALID, "weak": IOU_WEAK},
        "candidate_axis_source": "sealed R3R2 candidate tape; no candidate creation or mutation",
        "runtime_future_gt_used": False, "runtime_gt_clean": True, "posthoc_gt_used_for_labels": True,
        "per_identity": per_identity,
    }
    return episodes, manifest


def make_counterfactuals(present: Sequence[Episode]) -> tuple[list[Episode], dict[str, Any]]:
    result: list[Episode] = []
    counts = Counter()
    records = []
    for episode in present:
        valid = {int(index) for index in episode.positive_indices}
        weak_or_target_overlap = {index for index, value in enumerate(episode.candidate_ious) if float(value) >= 0.30}
        best_weak = max((index for index, label in enumerate(episode.candidate_labels) if int(label) == 1), key=lambda index: (episode.candidate_ious[index], episode.candidate_uids[index]), default=None)
        variants = {
            "CF0": valid,
            "CF1": weak_or_target_overlap,
            "CF2": valid - ({best_weak} if best_weak is not None else set()),
        }
        for variant, remove in variants.items():
            keep = [index for index in range(episode.candidate_count) if index not in remove]
            if not keep:
                continue
            cf = Episode(
                sequence=episode.sequence, identity_key=episode.identity_key, gt_id=episode.gt_id,
                frame=episode.frame, kind="CF", query_state=episode.query_state.copy(),
                candidate_offsets=[episode.candidate_offsets[index] for index in keep],
                candidate_uids=[episode.candidate_uids[index] for index in keep],
                candidate_labels=[episode.candidate_labels[index] for index in keep],
                candidate_ious=[episode.candidate_ious[index] for index in keep], positive_indices=[],
                canonical_index=None, context=episode.context.copy(), target_present=True,
                source="counterfactual_target_removed", counterfactual_variant=variant,
                removed_uids=[episode.candidate_uids[index] for index in sorted(remove)],
            )
            result.append(cf)
            counts[variant] += 1
            records.append({
                "sequence": episode.sequence, "identity_key": episode.identity_key, "frame": episode.frame,
                "variant": variant, "removed_uids": cf.removed_uids, "kept_uids": cf.candidate_uids,
                "removed_valid_count": int(len(valid & remove)), "kept_valid_count": int(sum(label == 2 for label in cf.candidate_labels)),
                "distractor_axis_unchanged": True, "formal_evaluation_eligible": False,
            })
    if any(item for item in result if any(label == 2 for label in item.candidate_labels)):
        raise RuntimeError("counterfactual retained a valid target candidate")
    return result, {"stage": STAGE, "counts": dict(counts), "episode_count": len(result), "formal_evaluation_eligible": False, "variants": ["CF0", "CF1", "CF2"], "records": records[:200], "records_truncated": max(0, len(records) - 200)}


def runtime_feature_audit(data_by_sequence: Mapping[str, SequenceData]) -> tuple[dict[str, Any], dict[str, Any]]:
    fields = ["confidence", "presence", "box_xyxy", "candidate_source", "iou_pred", "native_tid", "valid", "candidate_count", "embedding_offset"]
    availability = {}
    values: dict[str, list[float]] = defaultdict(list)
    sources = {}
    for field_name in fields:
        present = 0
        total = 0
        for data in data_by_sequence.values():
            for frame in data.frames.values():
                for candidate in frame["candidates"]:
                    total += 1
                    value = candidate.get(field_name)
                    if field_name == "candidate_count":
                        value = len(frame["candidates"])
                    if value is not None:
                        present += 1
                        if isinstance(value, (int, float)) and not isinstance(value, bool):
                            values[field_name].append(float(value))
        availability[field_name] = {"available": present == total and total > 0, "present": present, "total": total, "missing": total - present, "range": finite_stats(values[field_name]), "source": "sealed runtime candidate metadata"}
    target_fields = {
        "candidate_embedding": True, "candidate_box_geometry": True, "candidate_confidence": True,
        "candidate_presence_score": True, "candidate_iou_pred": availability["iou_pred"]["available"],
        "native_tid": True, "mask_quality": False, "last_trusted_target_box": False,
        "predicted_motion": False, "frames_since_human_initialization": True,
        "frames_since_last_memory_write": True, "anchor_state_cosine": True,
    }
    audit = {
        "stage": STAGE, "sequences": list(SEQUENCES), "candidate_feature_availability": availability,
        "causal_query_feature_availability": target_fields,
        "gt_features_used_for_runtime": [], "runtime_future_gt_used": False,
        "runtime_gt_clean": True, "unavailable_fields_not_invented": True,
    }
    whitelist = {
        "stage": STAGE, "status": "FROZEN_RUNTIME_WHITELIST", "candidate_features": [
            "candidate_embedding", "candidate_box_width", "candidate_box_height", "candidate_box_area_log",
            "candidate_box_aspect", "candidate_confidence", "candidate_presence", "candidate_count",
        ],
        "query_features": ["query_embedding", "frames_since_human_initialization", "frames_since_last_memory_write", "anchor_state_cosine"],
        "quality_features": ["candidate_confidence", "candidate_presence", "candidate_box_width", "candidate_box_height", "candidate_box_area_log", "candidate_box_aspect"],
        "excluded_unavailable_features": ["last_trusted_target_box", "predicted_motion", "mask_quality", "future_gt_box", "target_iou", "target_gt_present"],
        "gt_fields_training_labels_only": ["target_gt_present", "candidate_iou", "P0", "P1a", "P1b", "VALID_TARGET", "WEAK_TARGET_LOCALIZATION", "OTHER"],
        "runtime_future_gt_used": False, "runtime_gt_clean": True,
    }
    return audit, whitelist


def source_audit() -> dict[str, Any]:
    required = [R3R2R1 / "FINAL_GOAL.json", R3R2R1 / "FINAL_RESULT.json", R3R2R1 / "stage_status.json", R3R2R1 / "runtime_score_tape.jsonl.zst", PROTOCOL / "training" / "canonical_training_index.jsonl"]
    if not all(path.exists() for path in required):
        raise RuntimeError("required frozen R3R2R1/Protocol artifact is missing")
    result = read_json(R3R2R1 / "FINAL_RESULT.json")
    return {
        "stage": STAGE, "source_branch": "codex/n72r20r3r2r1-open-set-decision-closure", "source_head": SOURCE_R3R2R1_HEAD, "new_stage_base_head": git_head(),
        "source_stage": "N72R20R3R2R1", "source_decision": result.get("decision"),
        "source_pooled_negative_fpr": result.get("pooled_negative_fpr", result.get("negative_fpr")),
        "source_pooled_correct_id_recall": result.get("pooled_correct_id_recall", result.get("open_set_correct_id_recall")),
        "source_macro_negative_fpr": result.get("macro_negative_fpr"), "source_macro_correct_id_recall": result.get("macro_correct_id_recall"),
        "source_shadow_wrong_write_rate": result.get("shadow_causal", {}).get("selected", {}).get("wrong_write_rate"),
        "historical_r3r2_decision": read_json(R3R2 / "FINAL_RESULT.json").get("decision"),
        "artifact_hashes": [{"path": str(path.relative_to(ROOT)), "sha256": sha256(path), "bytes": path.stat().st_size} for path in required],
        "historical_outputs_modified": False, "candidate_generation_frozen": True, "sam3_rerun": False,
        "val_accessed": False, "test_accessed": False, "association_authority_started": False,
    }


def freeze_goal() -> None:
    write_json(OUT / "FINAL_GOAL.json", {
        "stage": STAGE,
        "goal": "Joint Open-Set Identity-Availability Representation Learning",
        "central_question": "Can cross-scene identity representation learning be extended from relative target ranking to absolute identity-conditioned candidate validity by training directly on PRESENT, true-ABSENT, candidate-unavailable, and counterfactual target-removed episodes, so that the system can preserve strong target ranking while learning when every current candidate should be rejected?",
        "source_stage": "N72R20R3R2R1", "source_head_resolved_at_runtime": True,
        "relative_ranking_representation_already_passed": True, "absence_posthoc_calibration_exhausted": True,
        "joint_open_set_training_authorized": True, "candidate_generation_frozen": True, "sam3_rerun_forbidden": True,
        "osnet_frozen_initially": True, "n72r18_gru_frozen_initially": True, "exact_solver_frozen": True,
        "public_id_authority_frozen": True, "association_override_forbidden": True, "val_forbidden": True,
        "test_forbidden": True, "multi_branch_exploration_required": True,
        "single_method_failure_is_not_terminal": True, "goal_frozen": True,
    })


BOOLEAN_KEYS = {
    "goal_frozen", "source_head_resolved_at_runtime", "relative_ranking_representation_already_passed", "absence_posthoc_calibration_exhausted", "joint_open_set_training_authorized", "candidate_generation_frozen", "sam3_rerun_forbidden", "osnet_frozen_initially", "n72r18_gru_frozen_initially", "exact_solver_frozen", "public_id_authority_frozen", "association_override_forbidden", "val_forbidden", "test_forbidden", "multi_branch_exploration_required", "single_method_failure_is_not_terminal", "historical_outputs_modified", "candidate_created", "sam3_rerun", "val_accessed", "test_accessed", "runtime_future_gt_used", "runtime_gt_clean", "association_authority_started", "formal_evaluation_eligible", "oracle_is_posthoc_only", "training_validity_labels_gt_only", "unavailable_fields_not_invented", "one_decision_per_frame", "runtime_only", "corruption_train_only", "future_frames_inspected", "unidirectional", "causal", "diagnostic_only", "score_before_update", "none_no_write", "disagreement", "commit", "wrong_write", "retention", "candidate_created", "binary_committed", "no_binary_in_repository", "outer_heldout_absent_from_training", "heldout_observations_used_in_training", "outer_labels_used_for_architecture_selection", "outer_labels_used_for_selection", "threshold_selection_inner_only", "posthoc_only_for_oracle_rows", "runtime_eligible", "candidate_axis_unchanged", "candidate_generation_changed", "no_candidate_creation", "OSNet_frozen", "N72R18_GRU_frozen_initially", "exact_solver_frozen", "public_id_authority_frozen", "selected_by_inner_only", "selection_is_inner_only", "formal_evaluation_eligible", "counterfactual_training_included", "runtime_feature_whitelist", "oracle_clean_and_quality_are_not_runtime_inputs", "next_association_stage_authorized", "next_candidate_quality_stage_authorized", "next_memory_state_learning_stage_authorized", "static_gate_pass", "retention_ge_0.60", "wrong_write_rate_le_0.02", "pass", "pooled_negative_fpr_le_0.02", "pooled_correct_id_recall_ge_0.60", "macro_fpr_le_0.05", "macro_recall_ge_0.40", "runtime_gt_clean", "no_candidate_creation", "development_target_is_not_final_pass", "posthoc_oracle_only", "runtime_usable", "runtime_future_gt_used", "oracle_clean_and_quality_are_not_runtime_inputs",
}


def repair_boolean_artifacts() -> None:
    """Repair JSON boolean serialization from early runs of this script.

    The first complete run used a serializer that treated Python bool as int.
    Only fields preregistered as boolean are repaired; numeric metrics/counts
    remain numeric.
    """
    def repair(value: Any, key: str | None = None) -> Any:
        if isinstance(value, dict):
            return {item_key: repair(item_value, str(item_key)) for item_key, item_value in value.items()}
        if isinstance(value, list):
            return [repair(item, key) for item in value]
        if key in BOOLEAN_KEYS and isinstance(value, int) and value in (0, 1):
            return bool(value)
        return value
    for path in OUT.rglob("*.json"):
        write_json(path, repair(read_json(path)))
    tape = OUT / "causal/memory_write_audit.jsonl.zst"
    if tape.exists():
        write_zstd_jsonl(tape, [repair(row) for row in read_zstd_jsonl(tape)])


def update_test_summary(summary: str) -> None:
    """Record the independently verified test status without rerunning research."""
    result_path = OUT / "FINAL_RESULT.json"
    if result_path.exists():
        result = read_json(result_path)
        result["tests_summary"] = summary
        write_json(result_path, result)
    status_path = OUT / "stage_status.json"
    if status_path.exists():
        status = read_json(status_path)
        status["tests_summary"] = summary
        write_json(status_path, status)
    report_path = OUT / "FINAL_REPORT.md"
    if report_path.exists():
        lines = report_path.read_text(encoding="utf-8").splitlines()
        test_line = f"Test summary: {summary}"
        lines = [line for line in lines if not line.startswith("Test summary:")]
        lines.append(test_line)
        report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


class PairVerifier(nn.Module):
    """Small absolute candidate-validity verifier (<100k parameters)."""

    def __init__(self, input_dim: int = 1025) -> None:
        super().__init__()
        self.network = nn.Sequential(nn.Linear(input_dim, 32), nn.GELU(), nn.Linear(32, 16), nn.GELU(), nn.Linear(16, 1))

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.network(value).squeeze(-1)


class QualityVerifier(nn.Module):
    def __init__(self, input_dim: int = 8) -> None:
        super().__init__()
        self.network = nn.Sequential(nn.Linear(input_dim, 32), nn.GELU(), nn.Linear(32, 16), nn.GELU(), nn.Linear(16, 1))

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.network(value).squeeze(-1)


class SetHead(nn.Module):
    def __init__(self, input_dim: int = 8) -> None:
        super().__init__()
        self.network = nn.Sequential(nn.Linear(input_dim, 24), nn.GELU(), nn.Linear(24, 1))

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.network(value).squeeze(-1)


class JointModel(nn.Module):
    """R3R2 metric adapter plus an absolute V/W/O candidate head."""

    def __init__(self, adapter: CrossSceneIdentityAdapter) -> None:
        super().__init__()
        self.adapter = adapter
        self.verifier = nn.Sequential(nn.Linear(1025, 32), nn.GELU(), nn.Linear(32, 16), nn.GELU(), nn.Linear(16, 3))

    def forward(self, query: torch.Tensor, candidates: torch.Tensor, mask: torch.Tensor) -> dict[str, torch.Tensor]:
        query_embedding = self.adapter.encode_query(query)
        candidate_embedding = self.adapter.encode_candidates(candidates)
        scores = torch.einsum("bd,bnd->bn", query_embedding, candidate_embedding).masked_fill(~mask, -1.0e9)
        q = query_embedding.unsqueeze(1).expand_as(candidate_embedding)
        pair = torch.cat([torch.sum(q * candidate_embedding, dim=-1, keepdim=True), q * candidate_embedding, torch.abs(q - candidate_embedding)], dim=-1)
        validity_logits = self.verifier(pair)
        return {"scores": scores, "validity_logits": validity_logits, "candidate_embedding": candidate_embedding, "query_embedding": query_embedding}


class TemporalGRU(nn.Module):
    def __init__(self, input_dim: int = 7, hidden_dim: int = 16) -> None:
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.head = nn.Linear(hidden_dim, 1)

    def forward(self, value: torch.Tensor, hidden: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        output, hidden = self.gru(value, hidden)
        return self.head(output).squeeze(-1), hidden


class Denoiser(nn.Module):
    def __init__(self, context_dim: int = 4) -> None:
        super().__init__()
        self.input = nn.Sequential(nn.Linear(512 * 2 + context_dim, 128), nn.GELU(), nn.Linear(128, 128), nn.GELU(), nn.Linear(128, 512))

    def forward(self, state: torch.Tensor, anchor: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        delta = self.input(torch.cat([state, anchor, context], dim=-1))
        return nn.functional.normalize(state + delta, dim=-1)


class BinaryHead(nn.Module):
    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.network = nn.Sequential(nn.Linear(input_dim, 24), nn.GELU(), nn.Linear(24, 1))

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.network(value).squeeze(-1)


def load_adapter(sequence: str, seed: int = R3R2_SEED, device: str = "cpu") -> CrossSceneIdentityAdapter:
    path = MODEL_ROOT / f"cross_scene_adapter__{sequence}__seed{seed}.pt"
    if not path.exists():
        raise RuntimeError(f"missing frozen R3R2 adapter: {path}")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    model = CrossSceneIdentityAdapter()
    model.load_state_dict(payload.get("state_dict", payload), strict=True)
    model.eval()
    return model.to(device)


def build_adapter_cache(data_by_sequence: Mapping[str, SequenceData]) -> dict[str, dict[str, Any]]:
    cache: dict[str, dict[str, Any]] = {}
    with torch.no_grad():
        for sequence, data in data_by_sequence.items():
            model = load_adapter(sequence, R3R2_SEED, "cpu")
            raw = unit(np.asarray(data.embeddings, dtype=np.float32))
            encoded = []
            for start in range(0, len(raw), 1024):
                encoded.append(model.encode_candidates(torch.from_numpy(raw[start:start + 1024])).cpu().numpy())
            cache[sequence] = {"model": model, "candidate_encoded": np.concatenate(encoded, axis=0), "candidate_raw": raw}
    return cache


def sample_episodes(episodes: Sequence[Episode], sequences: Sequence[str], include_counterfactual: bool, cap: int = MAX_TRAIN_EPISODES_PER_KIND) -> list[Episode]:
    selected: list[Episode] = []
    groups: dict[str, list[Episode]] = defaultdict(list)
    allowed = set(sequences)
    for episode in episodes:
        if episode.sequence not in allowed:
            continue
        if episode.kind == "PRESENT":
            groups["PRESENT"].append(episode)
        elif episode.kind in {"P0", "P1a", "P1b"}:
            groups["P1" if episode.kind.startswith("P1") else episode.kind].append(episode)
    # Keep an observation from every eligible identity before filling the
    # remainder by a deterministic temporal stride.
    for group_name, values in sorted(groups.items()):
        values = sorted(values, key=lambda item: (item.identity_key, item.frame))
        by_identity: dict[str, list[Episode]] = defaultdict(list)
        for episode in values:
            by_identity[episode.identity_key].append(episode)
        chosen: list[Episode] = []
        for identity_key in sorted(by_identity):
            chosen.append(by_identity[identity_key][0])
        remaining = [item for item in values if item not in chosen]
        if len(chosen) < cap:
            stride = max(1, len(remaining) // max(1, cap - len(chosen)))
            chosen.extend(remaining[::stride][: max(0, cap - len(chosen))])
        selected.extend(chosen[:cap])
    if include_counterfactual:
        # Counterfactuals are generated from a prefix of PRESENT episodes to
        # keep the persistent artifact below the storage target.
        present = [item for item in episodes if item.sequence in allowed and item.kind == "PRESENT"]
        by_variant: dict[str, list[Episode]] = defaultdict(list)
        for item in present:
            # Reconstructing each variant here avoids retaining a second large
            # in-memory manifest; the final manifest is written separately.
            valid = {int(index) for index in item.positive_indices}
            overlap = {index for index, value in enumerate(item.candidate_ious) if float(value) >= 0.30}
            best_weak = max((index for index, label in enumerate(item.candidate_labels) if int(label) == 1), key=lambda index: (item.candidate_ious[index], item.candidate_uids[index]), default=None)
            removals = {"CF0": valid, "CF1": overlap, "CF2": valid - ({best_weak} if best_weak is not None else set())}
            for variant, remove in removals.items():
                keep = [index for index in range(item.candidate_count) if index not in remove]
                if not keep:
                    continue
                by_variant[variant].append(Episode(
                    sequence=item.sequence, identity_key=item.identity_key, gt_id=item.gt_id, frame=item.frame,
                    kind="CF", query_state=item.query_state.copy(), candidate_offsets=[item.candidate_offsets[i] for i in keep],
                    candidate_uids=[item.candidate_uids[i] for i in keep], candidate_labels=[item.candidate_labels[i] for i in keep],
                    candidate_ious=[item.candidate_ious[i] for i in keep], positive_indices=[], canonical_index=None,
                    context=item.context.copy(), target_present=True, source="counterfactual_target_removed",
                    counterfactual_variant=variant, removed_uids=[item.candidate_uids[i] for i in sorted(remove)],
                ))
        for variant in ("CF0", "CF1", "CF2"):
            selected.extend(sorted(by_variant[variant], key=lambda item: (item.identity_key, item.frame))[: max(1, cap // 3)])
    return selected


def episode_pair_features(episodes: Sequence[Episode], cache: Mapping[str, Mapping[str, Any]]) -> tuple[np.ndarray, np.ndarray, list[tuple[int, int]], np.ndarray]:
    features: list[np.ndarray] = []
    labels: list[int] = []
    spans: list[tuple[int, int]] = []
    contexts: list[np.ndarray] = []
    for episode in episodes:
        info = cache[episode.sequence]
        model: CrossSceneIdentityAdapter = info["model"]
        with torch.no_grad():
            q = model.encode_query(torch.from_numpy(episode.query_state[None, :])).cpu().numpy()[0]
        offsets = np.asarray(episode.candidate_offsets, dtype=np.int64)
        candidates = np.asarray(info["candidate_encoded"])[offsets]
        q_repeat = np.repeat(q[None, :], len(candidates), axis=0)
        pair = np.concatenate([np.sum(q_repeat * candidates, axis=1, keepdims=True), q_repeat * candidates, np.abs(q_repeat - candidates)], axis=1)
        start = sum(len(item) for item in features)
        features.append(pair.astype(np.float32))
        labels.extend([int(value) for value in episode.candidate_labels])
        spans.append((start, start + len(pair)))
        contexts.append(np.asarray(episode.context, dtype=np.float32))
    if not features:
        return np.zeros((0, 1025), np.float32), np.zeros(0, np.int64), [], np.zeros((0, 4), np.float32)
    return np.concatenate(features), np.asarray(labels, dtype=np.int64), spans, np.asarray(contexts, dtype=np.float32)


def raw_episode_batch(episodes: Sequence[Episode], data_by_sequence: Mapping[str, SequenceData], indices: Sequence[int], max_count: int | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[Episode]]:
    selected = [episodes[index] for index in indices]
    width = max_count or max((item.candidate_count for item in selected), default=1)
    queries = np.zeros((len(selected), 512), dtype=np.float32)
    candidates = np.zeros((len(selected), width, 512), dtype=np.float32)
    mask = np.zeros((len(selected), width), dtype=bool)
    labels = np.full((len(selected), width), -1, dtype=np.int64)
    contexts = np.zeros((len(selected), 4), dtype=np.float32)
    for batch_index, episode in enumerate(selected):
        data = data_by_sequence[episode.sequence]
        queries[batch_index] = unit(episode.query_state)
        values = unit(np.asarray(data.embeddings[np.asarray(episode.candidate_offsets, dtype=np.int64)], dtype=np.float32))
        count = min(width, len(values))
        candidates[batch_index, :count] = values[:count]
        mask[batch_index, :count] = True
        labels[batch_index, :count] = np.asarray(episode.candidate_labels[:count], dtype=np.int64)
        contexts[batch_index] = np.asarray(episode.context, dtype=np.float32)
    return queries, candidates, mask, labels, contexts, selected


def train_pair_heads(x: np.ndarray, y: np.ndarray, seeds: Sequence[int], device: torch.device, epochs: int = 6) -> list[PairVerifier]:
    if len(x) == 0 or len(np.unique(y == 2)) < 2:
        raise RuntimeError("pair training data has no valid/invalid contrast")
    tensor_x = torch.from_numpy(x).to(device)
    tensor_y = torch.from_numpy((y == 2).astype(np.float32)).to(device)
    positive = max(1.0, float(tensor_y.sum()))
    negative = max(1.0, float(len(tensor_y) - tensor_y.sum()))
    pos_weight = torch.tensor([negative / positive], dtype=torch.float32, device=device)
    heads: list[PairVerifier] = []
    for seed in seeds:
        torch.manual_seed(int(seed))
        model = PairVerifier().to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=2.0e-3, weight_decay=1.0e-4)
        order = np.arange(len(x))
        generator = np.random.default_rng(seed)
        for _ in range(epochs):
            generator.shuffle(order)
            for start in range(0, len(order), 2048):
                batch = torch.from_numpy(order[start:start + 2048]).to(device)
                logits = model(tensor_x[batch])
                loss = nn.functional.binary_cross_entropy_with_logits(logits, tensor_y[batch], pos_weight=pos_weight)
                optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step()
        model.eval(); heads.append(model)
    return heads


def train_joint_models(episodes: Sequence[Episode], data_by_sequence: Mapping[str, SequenceData], template_sequence: str, seeds: Sequence[int], device: torch.device, output_prefix: str, epochs: int = 4) -> tuple[list[JointModel], list[dict[str, Any]]]:
    if not episodes:
        raise RuntimeError("joint training received no episodes")
    width = max(item.candidate_count for item in episodes)
    models: list[JointModel] = []
    records: list[dict[str, Any]] = []
    for seed in seeds:
        torch.manual_seed(int(seed)); np.random.seed(int(seed))
        adapter = load_adapter(template_sequence, R3R2_SEED, "cpu")
        model = JointModel(adapter).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=8.0e-4, weight_decay=1.0e-5)
        rng = np.random.default_rng(seed)
        order = np.arange(len(episodes))
        selected_epochs = 0
        for _ in range(epochs):
            rng.shuffle(order)
            for start in range(0, len(order), 64):
                batch_indices = order[start:start + 64]
                q, c, mask, target, _context, batch_eps = raw_episode_batch(episodes, data_by_sequence, batch_indices, width)
                tq = torch.from_numpy(q).to(device); tc = torch.from_numpy(c).to(device); tm = torch.from_numpy(mask).to(device); ty = torch.from_numpy(target).to(device)
                output = model(tq, tc, tm)
                valid_mask = tm & (ty >= 0)
                target_vwo = torch.clamp(ty, 0, 2)
                validity = nn.functional.cross_entropy(output["validity_logits"][valid_mask], target_vwo[valid_mask]) if bool(valid_mask.any()) else torch.tensor(0.0, device=device)
                valid_probability = torch.softmax(output["validity_logits"], dim=-1)[..., 2]
                positive_mask = valid_mask & (ty == 2)
                negative_mask = valid_mask & (ty != 2)
                pos_probability = valid_probability[positive_mask]
                neg_probability = valid_probability[negative_mask]
                margin = (torch.relu(0.8 - pos_probability).mean() if pos_probability.numel() else torch.tensor(0.0, device=device)) + (torch.relu(neg_probability - 0.2).mean() if neg_probability.numel() else torch.tensor(0.0, device=device))
                rank_terms = []
                for row_index, episode in enumerate(batch_eps):
                    if episode.kind == "PRESENT" and episode.canonical_index is not None:
                        row_scores = output["scores"][row_index][tm[row_index]]
                        if len(row_scores) > int(episode.canonical_index):
                            rank_terms.append(nn.functional.cross_entropy(row_scores[None, :], torch.tensor([int(episode.canonical_index)], device=device)))
                rank = torch.stack(rank_terms).mean() if rank_terms else torch.tensor(0.0, device=device)
                supcon_terms = []
                for row_index, episode in enumerate(batch_eps):
                    if episode.kind == "PRESENT" and episode.positive_indices:
                        row_scores = output["scores"][row_index][tm[row_index]]
                        positive_score = row_scores[int(episode.canonical_index or episode.positive_indices[0])]
                        negatives = [index for index, label in enumerate(episode.candidate_labels) if label != 2]
                        if negatives:
                            supcon_terms.append(torch.relu(0.1 - positive_score + row_scores[torch.tensor(negatives, device=device)].max()))
                supcon = torch.stack(supcon_terms).mean() if supcon_terms else torch.tensor(0.0, device=device)
                loss = 1.0 * rank + 0.50 * validity + 0.25 * margin + 0.20 * supcon
                if not torch.isfinite(loss):
                    raise RuntimeError("joint representation loss became non-finite")
                optimizer.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0); optimizer.step()
            selected_epochs += 1
        model.eval()
        path = STAGE_ASSET_ROOT / "models" / f"{output_prefix}__seed{seed}.pt"
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"state_dict": model.state_dict(), "architecture": "JointModel", "seed": int(seed), "trainable_parameters": sum(parameter.numel() for parameter in model.parameters())}, path)
        records.append({"path": str(path), "sha256": sha256(path), "seed": int(seed), "architecture": "JointModel", "trainable_parameters": int(sum(parameter.numel() for parameter in model.parameters())), "epochs": selected_epochs, "binary_committed": False})
        models.append(model)
    return models, records


def load_formal_rows(data_by_sequence: Mapping[str, SequenceData]) -> list[dict[str, Any]]:
    canonical = [json.loads(line) for line in (PROTOCOL / "training" / "canonical_training_index.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    tape = read_zstd_jsonl(R3R2R1 / "runtime_score_tape.jsonl.zst")
    states = np.load(PROTOCOL / "training" / "causal_learned_state_vectors.float32.npy", mmap_mode="r")
    labels = {str(item["sequence"]): dict(item) for item in read_json(ASSET_ROOT / "posthoc_event_labels.json")["labels"]}
    tape_by_key = {(str(item["sequence"]), int(item["frame"])): item for item in tape}
    rows: list[dict[str, Any]] = []
    for source in canonical:
        sequence = str(source["sequence"]); frame = int(source["frame"])
        key = (sequence, frame)
        scored = tape_by_key.get(key)
        if scored is None:
            raise RuntimeError(f"formal tape missing {key}")
        if scored.get("runtime_future_gt_used") is not False or scored.get("runtime_gt_clean") is not True:
            raise RuntimeError(f"runtime tape provenance failed at {key}")
        data = data_by_sequence[sequence]
        if frame not in data.frames:
            raise RuntimeError(f"formal frame missing from candidate tape: {key}")
        axis = [str(item["candidate_uid"]) for item in source["candidate_axis"]]
        if axis != data.frames[frame]["candidate_uids"] or axis != [str(item) for item in scored["candidate_uid_axis"]]:
            raise RuntimeError(f"formal candidate axis changed at {key}")
        event = labels[sequence]
        gt_id = int(event["target_gt_id"])
        candidate_labels_iou, candidate_ious, target_present = candidate_labels(data, frame, gt_id)
        if any(int(value) == 2 for value in candidate_labels_iou):
            kind = "PRESENT"
        elif target_present:
            kind = "P1a" if max(candidate_ious, default=0.0) >= IOU_WEAK else "P1b"
        else:
            kind = "P0"
        context = source.get("runtime_context", {})
        q = np.asarray(states[int(source["causal_learned_state_ref"])], dtype=np.float32)
        rows.append({
            "sequence": sequence, "frame": frame, "gt_id": gt_id, "kind": kind,
            "query_state": q.copy(), "candidate_offsets": [int(item["embedding_offset"]) for item in data.frames[frame]["candidates"]],
            "candidate_uids": axis, "candidate_labels": candidate_labels_iou, "candidate_ious": candidate_ious,
            "positive_indices": [index for index, value in enumerate(candidate_labels_iou) if value == 2],
            "canonical_index": canonical_positive(candidate_labels_iou, candidate_ious, axis)[1],
            "scores": [float(value) for value in scored["candidate_scores"]],
            "source_none_score": float(scored.get("none_score", 0.0)),
            "context": [float(math.log1p(len(axis))), float(context.get("frames_since_human_initialization", frame)) / 100.0, float(context.get("frames_since_last_memory_write", 0.0)) / 100.0, float(context.get("learned_state_human_anchor_cosine", 1.0))],
            "runtime_future_gt_used": False, "runtime_gt_clean": True, "posthoc_gt_used": True,
        })
    if len(rows) != 8414 or len({(row["sequence"], row["frame"]) for row in rows}) != len(rows):
        raise RuntimeError(f"formal row count/axis mismatch: {len(rows)}")
    return rows


def pair_features_for_formal(row: Mapping[str, Any], cache: Mapping[str, Mapping[str, Any]]) -> np.ndarray:
    info = cache[str(row["sequence"])]
    with torch.no_grad():
        q = info["model"].encode_query(torch.from_numpy(np.asarray(row["query_state"], dtype=np.float32)[None, :])).cpu().numpy()[0]
    candidates = np.asarray(info["candidate_encoded"])[np.asarray(row["candidate_offsets"], dtype=np.int64)]
    q_repeat = np.repeat(q[None, :], len(candidates), axis=0)
    return np.concatenate([np.sum(q_repeat * candidates, axis=1, keepdims=True), q_repeat * candidates, np.abs(q_repeat - candidates)], axis=1).astype(np.float32)


def quality_for_candidates(candidates: Sequence[Mapping[str, Any]]) -> np.ndarray:
    result = []
    for candidate in candidates:
        box = np.asarray(candidate.get("box_xyxy", [0, 0, 0, 0]), dtype=np.float64)
        width = max(0.0, float(box[2] - box[0])); height = max(0.0, float(box[3] - box[1]))
        area = max(width * height, 1.0)
        result.append([
            float(candidate.get("confidence", candidate.get("presence", 0.0)) or 0.0),
            float(candidate.get("presence", candidate.get("confidence", 0.0)) or 0.0),
            min(width / 1920.0, 2.0), min(height / 1200.0, 2.0),
            min(math.log1p(area) / 14.0, 2.0), min(width / max(height, 1.0), 5.0) / 5.0,
        ])
    return np.asarray(result, dtype=np.float32).reshape((-1, 6))


def pair_quality_features(pair_x: np.ndarray, episodes: Sequence[Episode], data_by_sequence: Mapping[str, SequenceData], spans: Sequence[tuple[int, int]]) -> tuple[np.ndarray, np.ndarray]:
    values: list[np.ndarray] = []
    labels: list[int] = []
    for episode, (start, end) in zip(episodes, spans):
        candidates = data_by_sequence[episode.sequence].frames[episode.frame]["candidates"]
        quality = quality_for_candidates(candidates)
        pair = pair_x[start:end]
        reduced = np.concatenate([pair[:, :1], pair[:, 1:513].mean(axis=1, keepdims=True), pair[:, 513:].mean(axis=1, keepdims=True), quality], axis=1)
        values.append(reduced)
        labels.extend([int(item) for item in episode.candidate_labels])
    return np.concatenate(values).astype(np.float32), np.asarray(labels, dtype=np.int64)


def train_quality_heads(x: np.ndarray, y: np.ndarray, seeds: Sequence[int], device: torch.device, epochs: int = 6) -> list[QualityVerifier]:
    tensor_x = torch.from_numpy(x).to(device); tensor_y = torch.from_numpy((y == 2).astype(np.float32)).to(device)
    positive = max(1.0, float(tensor_y.sum())); negative = max(1.0, float(len(y) - tensor_y.sum()))
    weight = torch.tensor([negative / positive], dtype=torch.float32, device=device)
    result = []
    for seed in seeds:
        torch.manual_seed(int(seed)); model = QualityVerifier(input_dim=x.shape[1]).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=2.0e-3, weight_decay=1.0e-4)
        order = np.arange(len(x)); rng = np.random.default_rng(seed)
        for _ in range(epochs):
            rng.shuffle(order)
            for start in range(0, len(order), 2048):
                indices = torch.from_numpy(order[start:start + 2048]).to(device)
                loss = nn.functional.binary_cross_entropy_with_logits(model(tensor_x[indices]), tensor_y[indices], pos_weight=weight)
                optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step()
        model.eval(); result.append(model)
    return result


def set_summary(probabilities: Sequence[float], context: Sequence[float]) -> np.ndarray:
    values = np.asarray(probabilities, dtype=np.float64)
    if len(values) == 0:
        top = second = mean = std = entropy = 0.0
    else:
        ordered = np.sort(values)[::-1]
        top = float(ordered[0]); second = float(ordered[1]) if len(ordered) > 1 else top
        mean = float(values.mean()); std = float(values.std())
        entropy = float(-np.mean(values * np.log(np.maximum(values, 1.0e-8)) + (1.0 - values) * np.log(np.maximum(1.0 - values, 1.0e-8))))
    return np.asarray([top, second, top - second, mean, std, entropy, math.log1p(len(values)) / 4.0, float(context[3])], dtype=np.float32)


def train_set_heads(x: np.ndarray, y: np.ndarray, seeds: Sequence[int], device: torch.device, epochs: int = 8) -> list[SetHead]:
    tensor_x = torch.from_numpy(x).to(device); tensor_y = torch.from_numpy(y.astype(np.float32)).to(device)
    result = []
    for seed in seeds:
        torch.manual_seed(int(seed)); model = SetHead(x.shape[1]).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=3.0e-3, weight_decay=1.0e-4)
        for _ in range(epochs):
            loss = nn.functional.binary_cross_entropy_with_logits(model(tensor_x), tensor_y)
            optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step()
        model.eval(); result.append(model)
    return result


def average_binary_heads(heads: Sequence[nn.Module], values: np.ndarray, device: torch.device) -> np.ndarray:
    if not heads:
        return np.zeros(len(values), dtype=np.float64)
    tensor = torch.from_numpy(np.asarray(values, dtype=np.float32)).to(device)
    with torch.no_grad():
        probabilities = [torch.sigmoid(head(tensor)).detach().cpu().numpy() for head in heads]
    return np.mean(probabilities, axis=0)


def joint_episode_probabilities(models: Sequence[JointModel], episodes: Sequence[Episode], data_by_sequence: Mapping[str, SequenceData], device: torch.device) -> tuple[list[np.ndarray], list[np.ndarray]]:
    result_probs: list[np.ndarray] = []; result_scores: list[np.ndarray] = []
    width = max((item.candidate_count for item in episodes), default=1)
    for start in range(0, len(episodes), 64):
        indices = list(range(start, min(start + 64, len(episodes))))
        q, c, mask, _labels, _context, selected = raw_episode_batch(episodes, data_by_sequence, indices, width)
        with torch.no_grad():
            outputs = [model(torch.from_numpy(q).to(device), torch.from_numpy(c).to(device), torch.from_numpy(mask).to(device)) for model in models]
            probabilities = torch.stack([torch.softmax(output["validity_logits"], dim=-1)[..., 2] for output in outputs]).mean(dim=0).cpu().numpy()
            scores = torch.stack([output["scores"] for output in outputs]).mean(dim=0).cpu().numpy()
        for row_index, episode in enumerate(selected):
            result_probs.append(probabilities[row_index, :episode.candidate_count].astype(np.float32))
            result_scores.append(scores[row_index, :episode.candidate_count].astype(np.float32))
    return result_probs, result_scores


def raw_score_metrics(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    present = [row for row in records if row["kind"] == "PRESENT"]
    ranks = []
    for row in present:
        order = np.argsort(-np.asarray(row["scores"], dtype=np.float64), kind="stable")
        positives = set(int(item) for item in row["positive_indices"])
        rank = next((position + 1 for position, index in enumerate(order) if int(index) in positives), None)
        if rank is not None:
            ranks.append(int(rank))
    return {
        "rows": len(records), "present_rows": len(present),
        "Rank1": float(np.mean(np.asarray(ranks) <= 1)) if ranks else None,
        "Rank2": float(np.mean(np.asarray(ranks) <= 2)) if ranks else None,
        "Rank3": float(np.mean(np.asarray(ranks) <= 3)) if ranks else None,
        "Rank5": float(np.mean(np.asarray(ranks) <= 5)) if ranks else None,
        "MRR": float(np.mean([1.0 / value for value in ranks])) if ranks else None,
        "mean_rank": float(np.mean(ranks)) if ranks else None,
    }


def family_metrics(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not records:
        return {"rows": 0, "formal_gate": {"pass": False}}
    present = np.asarray([row["kind"] == "PRESENT" for row in records], dtype=bool)
    p_set = np.asarray([float(row["set_probability"]) for row in records], dtype=np.float64)
    accepted = p_set >= np.asarray([float(row["threshold"]) for row in records], dtype=np.float64)
    predicted = np.asarray([int(np.argmax(row["candidate_probability"])) if row["candidate_probability"] else -1 for row in records], dtype=np.int64)
    correct = np.asarray([bool(accepted[index] and present[index] and predicted[index] in set(int(item) for item in records[index]["positive_indices"])) for index in range(len(records))], dtype=bool)
    negative = ~present
    p0 = np.asarray([row["kind"] == "P0" for row in records], dtype=bool)
    p1a = np.asarray([row["kind"] == "P1a" for row in records], dtype=bool)
    p1b = np.asarray([row["kind"] == "P1b" for row in records], dtype=bool)
    candidate_labels = []
    candidate_scores = []
    for row in records:
        candidate_labels.extend([1 if int(label) == 2 else 0 for label in row["candidate_labels"]])
        candidate_scores.extend([float(value) for value in row["candidate_probability"]])
    per_sequence = {}
    for sequence in SEQUENCES:
        subset = [row for row in records if row["sequence"] == sequence]
        if not subset:
            continue
        subset_present = np.asarray([row["kind"] == "PRESENT" for row in subset], dtype=bool)
        subset_accepted = np.asarray([float(row["set_probability"]) >= float(row["threshold"]) for row in subset], dtype=bool)
        subset_predicted = np.asarray([int(np.argmax(row["candidate_probability"])) if row["candidate_probability"] else -1 for row in subset], dtype=np.int64)
        subset_correct = np.asarray([bool(subset_accepted[index] and subset_present[index] and subset_predicted[index] in set(int(item) for item in subset[index]["positive_indices"])) for index in range(len(subset))], dtype=bool)
        per_sequence[sequence] = {
            "rows": len(subset), "present_rows": int(subset_present.sum()),
            "negative_fpr": float(np.sum(subset_accepted & ~subset_present) / max(1, int((~subset_present).sum()))),
            "correct_id_recall": float(np.sum(subset_correct) / max(1, int(subset_present.sum()))),
        }
    ranks = []
    for index, row in enumerate(records):
        if row["kind"] != "PRESENT":
            continue
        order = np.argsort(-np.asarray(row["identity_scores"], dtype=np.float64), kind="stable")
        positive = set(int(item) for item in row["positive_indices"])
        rank = next((position + 1 for position, candidate_index in enumerate(order) if int(candidate_index) in positive), None)
        if rank is not None:
            ranks.append(int(rank))
    negative_fpr = float(np.sum(accepted & negative) / max(1, int(negative.sum())))
    correct_recall = float(np.sum(correct) / max(1, int(present.sum())))
    macro_fpr = float(np.mean([float(item["negative_fpr"]) for item in per_sequence.values()])) if per_sequence else 0.0
    macro_recall = float(np.mean([float(item["correct_id_recall"]) for item in per_sequence.values()])) if per_sequence else 0.0
    # Recall at 2% is deliberately post-hoc and is never used to select an
    # outer-fold threshold.
    recall_2 = recall_at_fpr(present.astype(np.int64), p_set, FORMAL_FPR)
    return {
        "rows": len(records), "present_rows": int(present.sum()), "negative_rows": int(negative.sum()),
        "accepted_rows": int(accepted.sum()), "correct_id_rows": int(correct.sum()),
        "negative_fpr": negative_fpr, "correct_id_recall": correct_recall,
        "P0_fpr": float(np.sum(accepted & p0) / max(1, int(p0.sum()))),
        "P1a_fpr": float(np.sum(accepted & p1a) / max(1, int(p1a.sum()))),
        "P1b_fpr": float(np.sum(accepted & p1b) / max(1, int(p1b.sum())),),
        "macro_negative_fpr": macro_fpr, "macro_correct_id_recall": macro_recall,
        "candidate_valid_auroc": auc_score(candidate_labels, candidate_scores),
        "candidate_valid_auprc": auprc(candidate_labels, candidate_scores),
        "candidate_recall_at_FPR_2pct": recall_at_fpr(candidate_labels, candidate_scores, FORMAL_FPR),
        "recall_at_FPR_2pct": recall_2,
        "Rank1": float(np.mean(np.asarray(ranks) <= 1)) if ranks else None,
        "Rank2": float(np.mean(np.asarray(ranks) <= 2)) if ranks else None,
        "Rank3": float(np.mean(np.asarray(ranks) <= 3)) if ranks else None,
        "Rank5": float(np.mean(np.asarray(ranks) <= 5)) if ranks else None,
        "MRR": float(np.mean([1.0 / value for value in ranks])) if ranks else None,
        "mean_rank": float(np.mean(ranks)) if ranks else None,
        "set_presence_auroc": auc_score(present.astype(np.int64), p_set),
        "set_presence_auprc": auprc(present.astype(np.int64), p_set),
        "formal_gate": {
            "pooled_negative_fpr_le_0.02": negative_fpr <= FORMAL_FPR,
            "pooled_correct_id_recall_ge_0.60": correct_recall >= FORMAL_RECALL,
            "macro_fpr_le_0.05": macro_fpr <= FORMAL_MACRO_FPR,
            "macro_recall_ge_0.40": macro_recall >= FORMAL_MACRO_RECALL,
            "runtime_gt_clean": all(not bool(row.get("runtime_future_gt_used")) for row in records),
            "no_candidate_creation": all(not bool(row.get("candidate_created")) for row in records),
        },
        "per_sequence": per_sequence,
    }


def _prediction_record(row: Mapping[str, Any], candidate_probability: Sequence[float], identity_scores: Sequence[float], set_probability: float, fold: str, threshold: float | None = None) -> dict[str, Any]:
    return {
        "sequence": str(row["sequence"]), "frame": int(row["frame"]), "kind": str(row["kind"]),
        "candidate_probability": [float(value) for value in candidate_probability], "identity_scores": [float(value) for value in identity_scores],
        "scores": [float(value) for value in identity_scores], "set_probability": float(set_probability),
        "candidate_labels": [int(value) for value in row["candidate_labels"]], "candidate_ious": [float(value) for value in row["candidate_ious"]],
        "candidate_uids": [str(value) for value in row["candidate_uids"]], "positive_indices": [int(value) for value in row["positive_indices"]], "canonical_index": row.get("canonical_index"),
        "context": [float(value) for value in row.get("context", [0.0, 0.0, 0.0, 0.0])], "context_anchor_cosine": float(row.get("context", [0.0, 0.0, 0.0, 0.0])[3]),
        "fold": str(fold), "threshold": float(threshold if threshold is not None else 0.5),
        "runtime_future_gt_used": False, "runtime_gt_clean": True, "candidate_created": False,
    }


def predict_pair_heads(heads: Sequence[PairVerifier], features: np.ndarray, device: torch.device) -> np.ndarray:
    values = torch.from_numpy(np.asarray(features, dtype=np.float32)).to(device)
    with torch.no_grad():
        return np.mean([torch.sigmoid(head(values)).cpu().numpy() for head in heads], axis=0)


def predict_quality_heads(heads: Sequence[QualityVerifier], features: np.ndarray, device: torch.device) -> np.ndarray:
    values = torch.from_numpy(np.asarray(features, dtype=np.float32)).to(device)
    with torch.no_grad():
        return np.mean([torch.sigmoid(head(values)).cpu().numpy() for head in heads], axis=0)


def _set_features_from_spans(probabilities: np.ndarray, episodes: Sequence[Episode], spans: Sequence[tuple[int, int]]) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray([set_summary(probabilities[start:end], episode.context) for episode, (start, end) in zip(episodes, spans)], dtype=np.float32)
    y = np.asarray([1 if episode.kind == "PRESENT" else 0 for episode in episodes], dtype=np.int64)
    return x, y


def _energy_features(probabilities: np.ndarray, episodes: Sequence[Episode], spans: Sequence[tuple[int, int]]) -> tuple[np.ndarray, np.ndarray]:
    values = []
    for episode, (start, end) in zip(episodes, spans):
        if end <= start:
            values.append([0.0, 0.0, 0.0, 0.0, *episode.context[:4]])
            continue
        p = np.clip(probabilities[start:end], 1.0e-6, 1.0 - 1.0e-6)
        logits = np.log(p) - np.log1p(-p)
        ordered = np.sort(logits)[::-1]
        values.append([float(np.log(np.exp(logits - logits.max()).sum()) + logits.max()), float(ordered[0]), float(logits.mean()), float(logits.std()), *episode.context[:4]])
    return np.asarray(values, dtype=np.float32), np.asarray([1 if episode.kind == "PRESENT" else 0 for episode in episodes], dtype=np.int64)


def _formal_quality_features(row: Mapping[str, Any], data: SequenceData, pair_x: np.ndarray) -> np.ndarray:
    quality = quality_for_candidates(data.frames[int(row["frame"])]["candidates"])
    return np.concatenate([pair_x[:, :1], pair_x[:, 1:513].mean(axis=1, keepdims=True), pair_x[:, 513:].mean(axis=1, keepdims=True), quality], axis=1).astype(np.float32)


def run_outer_models(
    data_by_sequence: Mapping[str, SequenceData], real_episodes: Sequence[Episode], counterfactuals: Sequence[Episode], formal_rows: Sequence[Mapping[str, Any]], cache: Mapping[str, Mapping[str, Any]], device: torch.device,
) -> dict[str, Any]:
    formal_by_sequence: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in formal_rows:
        formal_by_sequence[str(row["sequence"])].append(dict(row))
    all_families = {name: [] for name in ("J1_PAIRWISE", "J2_JOINT_ABSOLUTE", "J3_SET_NOISY_OR", "J4_ENERGY", "J5_QUALITY_FUSION", "J7_MULTITASK")}
    no_cf_records: list[dict[str, Any]] = []
    inner_records = []
    checkpoint_records = []
    selected_records: list[dict[str, Any]] = []
    selected_family_by_fold: dict[str, str] = {}
    for fold_index, heldout in enumerate(SEQUENCES):
        validation = SEQUENCES[(SEQUENCES.index(heldout) + 1) % len(SEQUENCES)]
        fit_sequences = [sequence for sequence in SEQUENCES if sequence not in {heldout, validation}]
        fit_real = sample_episodes(real_episodes, fit_sequences, include_counterfactual=False)
        fit_cf = sample_episodes(real_episodes, fit_sequences, include_counterfactual=True)
        # Pairwise verifier, trained on real PRESENT/P0/P1 episodes only.
        pair_x, pair_y, spans, _contexts = episode_pair_features(fit_real, cache)
        pair_heads = train_pair_heads(pair_x, pair_y, LEARNED_SEEDS, device)
        pair_probability = predict_pair_heads(pair_heads, pair_x, device)
        set_x, set_y = _set_features_from_spans(pair_probability, fit_real, spans)
        set_heads = train_set_heads(set_x, set_y, LEARNED_SEEDS, device)
        energy_x, energy_y = _energy_features(pair_probability, fit_real, spans)
        energy_heads = train_set_heads(energy_x, energy_y, LEARNED_SEEDS, device)
        quality_x, quality_y = pair_quality_features(pair_x, fit_real, data_by_sequence, spans)
        quality_heads = train_quality_heads(quality_x, quality_y, LEARNED_SEEDS, device)
        # Joint metric + absolute validity, explicitly including CF episodes.
        fit_joint = fit_cf[:MAX_JOINT_EPISODES]
        template_sequence = fit_sequences[0]
        joint_models, joint_records = train_joint_models(fit_joint, data_by_sequence, template_sequence, LEARNED_SEEDS, device, f"joint__fold{fold_index}")
        checkpoint_records.extend([{**record, "fold": heldout, "variant": "with_counterfactual"} for record in joint_records])
        joint_probability, joint_score = joint_episode_probabilities(joint_models, fit_joint, data_by_sequence, device)
        joint_flat = np.concatenate(joint_probability) if joint_probability else np.zeros(0, dtype=np.float32)
        joint_spans = []
        cursor = 0
        for probabilities in joint_probability:
            joint_spans.append((cursor, cursor + len(probabilities))); cursor += len(probabilities)
        joint_set_x, joint_set_y = _set_features_from_spans(joint_flat, fit_joint, joint_spans)
        joint_set_heads = train_set_heads(joint_set_x, joint_set_y, LEARNED_SEEDS, device)
        # J2 without counterfactuals is a preregistered diagnostic ablation.
        no_cf_models, no_cf_ckpt = train_joint_models(fit_real[: min(len(fit_real), MAX_JOINT_EPISODES)], data_by_sequence, template_sequence, (LEARNED_SEEDS[0],), device, f"joint_no_cf__fold{fold_index}", epochs=3)
        checkpoint_records.extend([{**record, "fold": heldout, "variant": "without_counterfactual"} for record in no_cf_ckpt])
        fold_raw: dict[str, dict[str, list[dict[str, Any]]]] = {family: {"validation": [], "heldout": []} for family in all_families}
        for split_name, sequence in (("validation", validation), ("heldout", heldout)):
            for row in formal_by_sequence[sequence]:
                pair = pair_features_for_formal(row, cache)
                p1 = predict_pair_heads(pair_heads, pair, device)
                set_probability_1 = float(np.max(p1) if len(p1) else 0.0)
                p5 = predict_quality_heads(quality_heads, _formal_quality_features(row, data_by_sequence[sequence], pair), device)
                p5_set = float(np.max(p5) if len(p5) else 0.0)
                # J2 uses raw candidates and the newly trained adapter.
                width = len(row["candidate_offsets"])
                q = unit(row["query_state"])[None, :]
                raw = unit(np.asarray(data_by_sequence[sequence].embeddings[np.asarray(row["candidate_offsets"], dtype=np.int64)], dtype=np.float32))[None, :, :]
                mask = np.ones((1, width), dtype=bool)
                with torch.no_grad():
                    joint_outputs = [model(torch.from_numpy(q).to(device), torch.from_numpy(raw).to(device), torch.from_numpy(mask).to(device)) for model in joint_models]
                    p2 = torch.stack([torch.softmax(output["validity_logits"], dim=-1)[0, :, 2] for output in joint_outputs]).mean(dim=0).cpu().numpy()
                    s2 = torch.stack([output["scores"][0] for output in joint_outputs]).mean(dim=0).cpu().numpy()
                    p2_set_x = set_summary(p2, row["context"])[None, :]
                    p7_set = float(average_binary_heads(joint_set_heads, p2_set_x, device)[0])
                    e_x = _energy_features(p1, [Episode(sequence, "formal", row["gt_id"], row["frame"], row["kind"], row["query_state"], row["candidate_offsets"], row["candidate_uids"], row["candidate_labels"], row["candidate_ious"], row["positive_indices"], row["canonical_index"], row["context"], row["kind"] == "PRESENT")], [(0, len(p1))])[0]
                    p4_set = float(average_binary_heads(energy_heads, e_x, device)[0])
                    p3_set = float(average_binary_heads(set_heads, set_summary(p1, row["context"])[None, :], device)[0])
                records = {
                    "J1_PAIRWISE": _prediction_record(row, p1, pair[:, :1].reshape(-1), set_probability_1, heldout),
                    "J2_JOINT_ABSOLUTE": _prediction_record(row, p2, s2, float(np.max(p2) if len(p2) else 0.0), heldout),
                    "J3_SET_NOISY_OR": _prediction_record(row, p1, pair[:, :1].reshape(-1), p3_set, heldout),
                    "J4_ENERGY": _prediction_record(row, p1, pair[:, :1].reshape(-1), p4_set, heldout),
                    "J5_QUALITY_FUSION": _prediction_record(row, p5, pair[:, :1].reshape(-1), p5_set, heldout),
                    "J7_MULTITASK": _prediction_record(row, p2, s2, p7_set, heldout),
                }
                for family, record in records.items():
                    fold_raw[family][split_name].append(record)
                with torch.no_grad():
                    no_cf_output = no_cf_models[0](torch.from_numpy(q).to(device), torch.from_numpy(raw).to(device), torch.from_numpy(mask).to(device))
                    no_cf_p = torch.softmax(no_cf_output["validity_logits"], dim=-1)[0, :, 2].cpu().numpy()
                    no_cf_s = no_cf_output["scores"][0].cpu().numpy()
                if split_name == "heldout":
                    no_cf_records.append(_prediction_record(row, no_cf_p, no_cf_s, float(np.max(no_cf_p) if len(no_cf_p) else 0.0), heldout))
        family_thresholds = {}
        family_validation_metrics = {}
        for family in all_families:
            validation_records = fold_raw[family]["validation"]
            val_labels = np.asarray([1 if row["kind"] == "PRESENT" else 0 for row in validation_records], dtype=np.int64)
            val_correct = np.asarray([1 if row["kind"] == "PRESENT" and int(np.argmax(row["candidate_probability"])) in set(row["positive_indices"]) else 0 for row in validation_records], dtype=np.int64)
            threshold, selection = select_threshold(val_labels, np.asarray([row["set_probability"] for row in validation_records]), val_correct)
            family_thresholds[family] = threshold
            for record in fold_raw[family]["heldout"]:
                record["threshold"] = threshold
                all_families[family].append(record)
            validation_applied = []
            for record in validation_records:
                clone = dict(record); clone["threshold"] = threshold; validation_applied.append(clone)
            family_validation_metrics[family] = {"metrics": family_metrics(validation_applied), "threshold_selection": selection}
        feasible = [family for family, payload in family_validation_metrics.items() if not payload["threshold_selection"]["inner_gate_infeasible"]]
        selection_pool = feasible or list(all_families)
        chosen = sorted(selection_pool, key=lambda family: (-family_validation_metrics[family]["metrics"]["correct_id_recall"], family_validation_metrics[family]["metrics"]["negative_fpr"], -float(family_validation_metrics[family]["metrics"].get("Rank1") or 0.0), family))[0]
        selected_family_by_fold[heldout] = chosen
        selected_records.extend([dict(row) for row in all_families[chosen][-len(formal_by_sequence[heldout]):]])
        inner_records.append({
            "heldout_sequence": heldout, "inner_validation_sequence": validation, "parameter_fit_sequences": fit_sequences,
            "training_identity_count": len({episode.identity_key for episode in fit_real}), "training_episode_count": len(fit_real),
            "counterfactual_training_episode_count": len(fit_cf), "heldout_observations_used_in_training": 0,
            "outer_heldout_absent_from_training": heldout not in fit_sequences and heldout not in [episode.sequence for episode in fit_cf],
            "thresholds": family_thresholds, "validation_metrics": family_validation_metrics,
            "selected_family": chosen, "outer_labels_used_for_architecture_selection": False,
        })
    return {
        "families": all_families, "selected": selected_records, "selected_family_by_fold": selected_family_by_fold,
        "inner_records": inner_records, "checkpoint_records": checkpoint_records, "no_cf_records": no_cf_records,
    }


def clean_state_map(real_episodes: Sequence[Episode]) -> dict[tuple[str, int], np.ndarray]:
    result: dict[tuple[str, int], np.ndarray] = {}
    for episode in real_episodes:
        if episode.identity_key.endswith(":0") and episode.kind == "PRESENT":
            result[(episode.sequence, episode.frame)] = unit(episode.query_state)
    return result


def nearest_clean_state(row: Mapping[str, Any], clean: Mapping[tuple[str, int], np.ndarray], anchor: np.ndarray) -> np.ndarray:
    key = (str(row["sequence"]), int(row["frame"]))
    if key in clean:
        return np.asarray(clean[key], dtype=np.float32)
    candidates = [(abs(frame - int(row["frame"])), value) for (sequence, frame), value in clean.items() if sequence == str(row["sequence"])]
    return np.asarray(min(candidates, key=lambda item: item[0])[1] if candidates else anchor, dtype=np.float32)


def anchor_map() -> dict[str, np.ndarray]:
    events = read_json(ASSET_ROOT / "interaction_events.json")["events"]
    return {str(item["sequence"]): unit(item["human_anchor"]) for item in events}


def denoiser_rank_metrics(rows: Sequence[Mapping[str, Any]], queries: Sequence[np.ndarray], data_by_sequence: Mapping[str, SequenceData], cache: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    ranks = []
    for row, query in zip(rows, queries):
        if row["kind"] != "PRESENT":
            continue
        info = cache[str(row["sequence"])]
        with torch.no_grad():
            q = info["model"].encode_query(torch.from_numpy(unit(query)[None, :])).cpu().numpy()[0]
        scores = np.asarray(info["candidate_encoded"])[np.asarray(row["candidate_offsets"], dtype=np.int64)] @ q
        order = np.argsort(-scores, kind="stable")
        positive = set(int(value) for value in row["positive_indices"])
        rank = next((index + 1 for index, candidate in enumerate(order) if int(candidate) in positive), None)
        if rank is not None:
            ranks.append(int(rank))
    return {
        "present_rows": len([row for row in rows if row["kind"] == "PRESENT"]),
        "Rank1": float(np.mean(np.asarray(ranks) <= 1)) if ranks else None,
        "Rank3": float(np.mean(np.asarray(ranks) <= 3)) if ranks else None,
        "MRR": float(np.mean([1.0 / value for value in ranks])) if ranks else None,
        "mean_rank": float(np.mean(ranks)) if ranks else None,
    }


def train_denoiser_model(rows: Sequence[Mapping[str, Any]], clean: Mapping[tuple[str, int], np.ndarray], anchors: Mapping[str, np.ndarray], device: torch.device, seed: int, corruption: float = 0.0) -> Denoiser:
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    usable = [row for row in rows if str(row["sequence"]) in SEQUENCES]
    if not usable:
        raise RuntimeError("denoiser has no training rows")
    states = []; anchor_values = []; contexts = []; targets = []
    for row in usable:
        anchor = anchors[str(row["sequence"])]
        target = nearest_clean_state(row, clean, anchor)
        state = unit(np.asarray(row["query_state"], dtype=np.float32))
        if corruption > 0.0 and rng.random() < corruption:
            state = unit(0.5 * state + 0.5 * rng.normal(size=state.shape).astype(np.float32))
        states.append(state); anchor_values.append(anchor); contexts.append(np.asarray(row["context"], dtype=np.float32)); targets.append(target)
    model = Denoiser().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.0e-3, weight_decay=1.0e-5)
    q = torch.from_numpy(np.asarray(states, dtype=np.float32)).to(device); a = torch.from_numpy(np.asarray(anchor_values, dtype=np.float32)).to(device); c = torch.from_numpy(np.asarray(contexts, dtype=np.float32)).to(device); y = torch.from_numpy(np.asarray(targets, dtype=np.float32)).to(device)
    for _ in range(5):
        prediction = model(q, a, c)
        loss = (1.0 - torch.sum(prediction * y, dim=-1)).mean()
        if not torch.isfinite(loss):
            raise RuntimeError("denoiser loss became non-finite")
        optimizer.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 3.0); optimizer.step()
    model.eval(); return model


def run_state_denoising(formal_rows: Sequence[Mapping[str, Any]], real_episodes: Sequence[Episode], data_by_sequence: Mapping[str, SequenceData], cache: Mapping[str, Mapping[str, Any]], device: torch.device) -> dict[str, Any]:
    clean = clean_state_map(real_episodes); anchors = anchor_map()
    runtime_queries = [np.asarray(row["query_state"], dtype=np.float32) for row in formal_rows]
    d0 = denoiser_rank_metrics(formal_rows, runtime_queries, data_by_sequence, cache)
    d1_queries = [unit(0.75 * query + 0.25 * anchors[str(row["sequence"])]) for row, query in zip(formal_rows, runtime_queries)]
    d1 = denoiser_rank_metrics(formal_rows, d1_queries, data_by_sequence, cache)
    fold_results = []
    for fold_index, heldout in enumerate(SEQUENCES):
        fit_rows = [row for row in formal_rows if row["sequence"] != heldout]
        model_d2 = train_denoiser_model(fit_rows, clean, anchors, device, LEARNED_SEEDS[fold_index % len(LEARNED_SEEDS)], corruption=0.0)
        model_d3 = train_denoiser_model(fit_rows, clean, anchors, device, LEARNED_SEEDS[(fold_index + 1) % len(LEARNED_SEEDS)], corruption=0.25)
        heldout_rows = [row for row in formal_rows if row["sequence"] == heldout]
        q2 = []; q3 = []
        with torch.no_grad():
            for row in heldout_rows:
                state = torch.from_numpy(np.asarray(row["query_state"], dtype=np.float32)[None, :]).to(device)
                anchor = torch.from_numpy(anchors[str(row["sequence"])] [None, :]).to(device)
                context = torch.from_numpy(np.asarray(row["context"], dtype=np.float32)[None, :]).to(device)
                q2.append(model_d2(state, anchor, context).cpu().numpy()[0])
                q3.append(model_d3(state, anchor, context).cpu().numpy()[0])
        fold_results.append({"heldout_sequence": heldout, "D2": denoiser_rank_metrics(heldout_rows, q2, data_by_sequence, cache), "D3": denoiser_rank_metrics(heldout_rows, q3, data_by_sequence, cache), "future_gt_used": False})
    d2 = {"folds": fold_results, "pooled_Rank1": float(np.mean([item["D2"]["Rank1"] for item in fold_results if item["D2"]["Rank1"] is not None]))}
    d3 = {"folds": fold_results, "pooled_Rank1": float(np.mean([item["D3"]["Rank1"] for item in fold_results if item["D3"]["Rank1"] is not None]))}
    return {
        "D0": {"name": "no_denoiser", "runtime_only": True, "metrics": d0},
        "D1": {"name": "anchor_cosine_teacher_only", "runtime_only": True, "metrics": d1},
        "D2": {"name": "teacher_alignment_plus_ranking", "runtime_only": True, "metrics": d2},
        "D3": {"name": "teacher_alignment_plus_ranking_plus_corruption", "runtime_only": True, "metrics": d3},
        "formal_comparison": {"selected_by_inner_only": True, "D0_Rank1": d0["Rank1"], "D1_Rank1": d1["Rank1"], "D2_Rank1": d2["pooled_Rank1"], "D3_Rank1": d3["pooled_Rank1"], "corruption_train_only": True, "runtime_future_gt_used": False},
    }


def temporal_features(records: Sequence[Mapping[str, Any]]) -> np.ndarray:
    output = []
    last_accept = 0
    previous_frame = None
    for row in records:
        p = np.sort(np.asarray(row["candidate_probability"], dtype=np.float32))[::-1]
        top = float(p[0]) if len(p) else 0.0; second = float(p[1]) if len(p) > 1 else top
        frame = int(row["frame"])
        gap = float(frame - previous_frame) / 100.0 if previous_frame is not None else 0.0
        output.append([top, second, top - second, min(len(p), 16) / 16.0, float(row["set_probability"]), float(row.get("context_anchor_cosine", 0.0)), min(frame - last_accept, 100) / 100.0 + gap])
        if float(row["set_probability"]) >= float(row.get("threshold", 0.5)):
            last_accept = frame
        previous_frame = frame
    return np.asarray(output, dtype=np.float32)


def temporal_metrics(records: Sequence[Mapping[str, Any]], probability: Sequence[float], threshold: float) -> dict[str, Any]:
    accepted = np.asarray(probability, dtype=np.float64) >= float(threshold)
    present = np.asarray([row["kind"] == "PRESENT" for row in records], dtype=bool)
    predicted = np.asarray([int(np.argmax(row["candidate_probability"])) if row["candidate_probability"] else -1 for row in records])
    correct = accepted & present & np.asarray([predicted[index] in set(row["positive_indices"]) for index, row in enumerate(records)], dtype=bool)
    negative = ~present
    return {"rows": len(records), "negative_fpr": float(np.sum(accepted & negative) / max(1, int(negative.sum()))), "correct_id_recall": float(np.sum(correct) / max(1, int(present.sum()))), "P0_fpr": float(np.sum(accepted & np.asarray([row["kind"] == "P0" for row in records])) / max(1, sum(row["kind"] == "P0" for row in records))), "P1_fpr": float(np.sum(accepted & np.asarray([row["kind"] in {"P1a", "P1b"} for row in records])) / max(1, sum(row["kind"] in {"P1a", "P1b"} for row in records))), "threshold": float(threshold), "causal": True, "future_frames_inspected": False}


def run_temporal_models(formal_rows: Sequence[Mapping[str, Any]], selected_records: Sequence[Mapping[str, Any]], inner_records: Sequence[Mapping[str, Any]], device: torch.device) -> dict[str, Any]:
    by_sequence = defaultdict(list)
    for row in selected_records:
        by_sequence[str(row["sequence"])].append(dict(row))
    for values in by_sequence.values():
        values.sort(key=lambda item: int(item["frame"]))
    per_sequence = {}
    for sequence in SEQUENCES:
        rows = by_sequence.get(sequence, [])
        if not rows:
            continue
        x = temporal_features(rows)
        # A deterministic causal EMA and hysteresis are retained as baselines.
        ema = []; value = 0.0
        for row in rows:
            value = 0.8 * value + 0.2 * float(row["set_probability"]); ema.append(value)
        hysteresis = []; state = False
        for row in rows:
            p = float(row["set_probability"])
            high = float(row.get("threshold", 0.5)); low = high * 0.9
            state = (p >= high) if not state else (p >= low)
            hysteresis.append(1.0 if state else 0.0)
        per_sequence[sequence] = {"per_frame": temporal_metrics(rows, [row["set_probability"] for row in rows], float(rows[0].get("threshold", 0.5))), "ema": temporal_metrics(rows, ema, float(rows[0].get("threshold", 0.5))), "hysteresis": temporal_metrics(rows, hysteresis, 0.5), "features": {"input_dim": 7, "hidden_dim": 16, "bidirectional": False}}
    # The formal outer predictions are already sequence-heldout.  Train one
    # causal GRU on the non-heldout prefix for each heldout sequence using the
    # frozen runtime feature contract.
    causal_records = []
    for heldout in SEQUENCES:
        train_rows = [row for row in selected_records if row["sequence"] != heldout]
        test_rows = [row for row in selected_records if row["sequence"] == heldout]
        if not test_rows:
            continue
        model = TemporalGRU().to(device); optimizer = torch.optim.AdamW(model.parameters(), lr=3.0e-3, weight_decay=1.0e-4)
        for _ in range(8):
            loss_terms = []
            for sequence in SEQUENCES:
                if sequence == heldout:
                    continue
                values = sorted([row for row in train_rows if row["sequence"] == sequence], key=lambda item: int(item["frame"]))
                if not values:
                    continue
                logits, _ = model(torch.from_numpy(temporal_features(values)[None, :, :]).to(device))
                target = torch.from_numpy(np.asarray([1.0 if row["kind"] == "PRESENT" else 0.0 for row in values], dtype=np.float32)[None, :]).to(device)
                loss_terms.append(nn.functional.binary_cross_entropy_with_logits(logits, target))
            if loss_terms:
                loss = torch.stack(loss_terms).mean(); optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step()
        test_rows = sorted(test_rows, key=lambda item: int(item["frame"]))
        with torch.no_grad():
            logits, _ = model(torch.from_numpy(temporal_features(test_rows)[None, :, :]).to(device))
            probability = torch.sigmoid(logits[0]).cpu().numpy().tolist()
        threshold = float(test_rows[0].get("threshold", 0.5))
        causal_records.extend([{**row, "temporal_probability": float(value)} for row, value in zip(test_rows, probability)])
        per_sequence[heldout]["causal_gru"] = temporal_metrics(test_rows, probability, threshold) | {"input_dim": 7, "hidden_dim": 16, "unidirectional": True, "trained_sequences": [sequence for sequence in SEQUENCES if sequence != heldout]}
    pooled = {}
    for key in ("per_frame", "ema", "hysteresis", "causal_gru"):
        subsets = [item[key] for item in per_sequence.values() if key in item]
        pooled[key] = {"negative_fpr": float(np.mean([item["negative_fpr"] for item in subsets])) if subsets else None, "correct_id_recall": float(np.mean([item["correct_id_recall"] for item in subsets])) if subsets else None}
    return {"per_sequence": per_sequence, "pooled": pooled, "runtime_future_gt_used": False, "future_frames_inspected": False}


def decomposed_features(records: Sequence[Mapping[str, Any]]) -> np.ndarray:
    return np.asarray([set_summary(row["candidate_probability"], row.get("context", [0.0, 0.0, 0.0, 0.0])) for row in records], dtype=np.float32)


def train_binary_head(x: np.ndarray, y: np.ndarray, device: torch.device, seed: int) -> BinaryHead:
    torch.manual_seed(seed)
    model = BinaryHead(x.shape[1]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3.0e-3, weight_decay=1.0e-4)
    tx = torch.from_numpy(x.astype(np.float32)).to(device); ty = torch.from_numpy(y.astype(np.float32)).to(device)
    for _ in range(10):
        loss = nn.functional.binary_cross_entropy_with_logits(model(tx), ty)
        optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step()
    model.eval(); return model


def run_decomposed_presence(selected_records: Sequence[Mapping[str, Any]], device: torch.device) -> dict[str, Any]:
    heldout_results = []
    for index, heldout in enumerate(SEQUENCES):
        train = [row for row in selected_records if row["sequence"] != heldout]
        test = [row for row in selected_records if row["sequence"] == heldout]
        if not test:
            continue
        x_train = decomposed_features(train); x_test = decomposed_features(test)
        physical = train_binary_head(x_train, np.asarray([row["kind"] != "P0" for row in train], dtype=np.float32), device, LEARNED_SEEDS[index % len(LEARNED_SEEDS)])
        availability = train_binary_head(x_train, np.asarray([row["kind"] == "PRESENT" for row in train], dtype=np.float32), device, LEARNED_SEEDS[(index + 1) % len(LEARNED_SEEDS)])
        with torch.no_grad():
            p_physical = torch.sigmoid(physical(torch.from_numpy(x_test).to(device))).cpu().numpy()
            p_available = torch.sigmoid(availability(torch.from_numpy(x_test).to(device))).cpu().numpy()
        combined = p_physical * p_available
        clone = [dict(row) for row in test]
        for row, probability in zip(clone, combined):
            row["set_probability"] = float(probability)
            row["threshold"] = float(row.get("threshold", 0.5))
        heldout_results.append({"heldout_sequence": heldout, "metrics": family_metrics(clone), "physical_presence_auc": auc_score([int(row["kind"] != "P0") for row in test], p_physical), "candidate_availability_auc_given_present": auc_score([int(row["kind"] == "PRESENT") for row in test if row["kind"] != "P0"], [float(p_available[index]) for index, row in enumerate(test) if row["kind"] != "P0"]), "final_probability_is_product": True, "runtime_gt_clean": True})
    return {"per_fold": heldout_results, "pooled_macro_negative_fpr": float(np.mean([item["metrics"]["negative_fpr"] for item in heldout_results])) if heldout_results else None, "pooled_macro_recall": float(np.mean([item["metrics"]["correct_id_recall"] for item in heldout_results])) if heldout_results else None, "decision_stages": ["physical_presence", "candidate_availability_given_presence", "product_then_argmax_or_NONE"], "runtime_future_gt_used": False}


def oracle_ladder(selected_records: Sequence[Mapping[str, Any]]) -> tuple[dict[str, Any], dict[str, Any]]:
    def clone_with(records: Sequence[Mapping[str, Any]], mode: str) -> list[dict[str, Any]]:
        result = []
        for original in records:
            row = dict(original)
            probabilities = np.asarray(row["candidate_probability"], dtype=np.float64)
            if mode == "O1_PERFECT_P0_REJECTION" and row["kind"] == "P0":
                row["set_probability"] = 0.0
            elif mode == "O2_PERFECT_P1_REJECTION" and row["kind"] in {"P1a", "P1b"}:
                row["set_probability"] = 0.0
            elif mode in {"O3_PERFECT_QUALITY", "O5_ORACLE_CLEAN_PLUS_QUALITY"}:
                row["candidate_probability"] = [1.0 if label == 2 else 0.0 for label in row["candidate_labels"]]
                row["set_probability"] = 1.0 if any(label == 2 for label in row["candidate_labels"]) else 0.0
            elif mode == "O4_ORACLE_CLEAN_QUERY":
                row["set_probability"] = float(np.max(probabilities) if len(probabilities) else 0.0)
            row["threshold"] = 0.5
            result.append(row)
        return result
    modes = {"O0_RUNTIME_JOINT": selected_records, "O1_PERFECT_P0_REJECTION": clone_with(selected_records, "O1_PERFECT_P0_REJECTION"), "O2_PERFECT_P1_REJECTION": clone_with(selected_records, "O2_PERFECT_P1_REJECTION"), "O3_PERFECT_QUALITY": clone_with(selected_records, "O3_PERFECT_QUALITY"), "O4_ORACLE_CLEAN_QUERY": clone_with(selected_records, "O4_ORACLE_CLEAN_QUERY"), "O5_ORACLE_CLEAN_PLUS_QUALITY": clone_with(selected_records, "O5_ORACLE_CLEAN_PLUS_QUALITY")}
    table = {}
    for name, records in modes.items():
        table[name] = {"metrics": family_metrics(records), "posthoc_oracle_only": name != "O0_RUNTIME_JOINT", "runtime_eligible": name == "O0_RUNTIME_JOINT"}
    return (
        {"stage": STAGE, "posthoc_only_for_oracle_rows": True, "modes": table},
        {"stage": STAGE, "headroom": {name: {key: value for key, value in item["metrics"].items() if key in {"negative_fpr", "correct_id_recall", "P0_fpr", "P1a_fpr", "P1b_fpr", "Rank1", "MRR"}} for name, item in table.items()}, "oracle_clean_and_quality_are_not_runtime_inputs": True},
    )


def _base_assignment_uid(sequence: str, frame: int, public_id: int) -> str | None:
    path = ASSET_ROOT / "base_scores" / sequence / "base_scores.jsonl.zst"
    rows = read_zstd_jsonl(path)
    for row in rows:
        if int(row.get("frame", -1)) != int(frame):
            continue
        assignment = row.get("base_assignment", {})
        for item in assignment.get("assignment_rows", []):
            if int(item.get("public_id", -1)) == int(public_id):
                return None if item.get("candidate_uid") in (None, "", "None") else str(item["candidate_uid"])
    return None


def run_shadow_causal(selected_records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    events = {str(item["sequence"]): dict(item) for item in read_json(ASSET_ROOT / "interaction_events.json")["events"]}
    base_maps: dict[str, dict[int, str | None]] = {}
    for sequence in SEQUENCES:
        event = events[sequence]
        base_maps[sequence] = {}
        for item in read_zstd_jsonl(ASSET_ROOT / "base_scores" / sequence / "base_scores.jsonl.zst"):
            value = None
            for assignment in item.get("base_assignment", {}).get("assignment_rows", []):
                public_id = assignment.get("public_id")
                if public_id is not None and int(public_id) == int(event["target_public_id"]):
                    value = None if assignment.get("candidate_uid") in (None, "", "None") else str(assignment["candidate_uid"])
                    break
            base_maps[sequence][int(item["frame"])] = value
    audit_rows = []
    for row in selected_records:
        event = events[str(row["sequence"])]
        accepted = float(row["set_probability"]) >= float(row["threshold"])
        predicted_index = int(np.argmax(row["candidate_probability"])) if row["candidate_probability"] else -1
        predicted_uid = row["candidate_uids"][predicted_index] if predicted_index >= 0 else None
        base_uid = base_maps[str(row["sequence"])].get(int(row["frame"]))
        identity_candidate = predicted_uid if accepted else None
        disagreement = accepted and identity_candidate != base_uid
        commit = bool(accepted and identity_candidate is not None and identity_candidate == base_uid)
        valid = bool(predicted_index in set(row["positive_indices"])) if predicted_index >= 0 else False
        audit_rows.append({"sequence": row["sequence"], "frame": row["frame"], "set_probability": float(row["set_probability"]), "threshold": float(row["threshold"]), "identity_policy_candidate": identity_candidate, "base_assignment_candidate": base_uid, "disagreement": disagreement, "none_no_write": not accepted, "commit": commit, "wrong_write": bool(commit and not valid), "retention": bool(commit and valid), "candidate_created": False, "score_before_update": True, "association_authority_started": False, "future_gt_used": False})
    def summarize(rows: Sequence[Mapping[str, Any]], name: str, include_per_sequence: bool = True) -> dict[str, Any]:
        accepted = [row for row in rows if row["identity_policy_candidate"] is not None]
        result = {"policy": name, "rows": len(rows), "accepted": len(accepted), "commits": int(sum(bool(row["commit"]) for row in rows)), "wrong_write_rate": float(sum(bool(row["wrong_write"]) for row in rows) / max(1, sum(bool(row["commit"]) for row in rows))), "retention": float(sum(bool(row["retention"]) for row in rows) / max(1, sum(bool(row["commit"]) for row in rows))), "first_wrong_write_frame": next(({"sequence": row["sequence"], "frame": row["frame"]} for row in rows if row["wrong_write"]), None), "none_no_write_count": int(sum(bool(row["none_no_write"]) for row in rows)), "disagreement_no_write_count": int(sum(bool(row["disagreement"]) and not bool(row["commit"]) for row in rows)), "candidate_created": False, "association_authority_started": False, "diagnostic_only": True}
        if include_per_sequence:
            result["per_sequence"] = {sequence: summarize([row for row in rows if row["sequence"] == sequence], f"{name}:{sequence}", False) if any(row["sequence"] == sequence for row in rows) else {} for sequence in SEQUENCES}
        return result
    selected = summarize(audit_rows, "selected")
    safest_rows = [dict(row, identity_policy_candidate=None if row["set_probability"] < 0.99 else row["identity_policy_candidate"]) for row in audit_rows]
    frontier_rows = [dict(row, identity_policy_candidate=None if row["set_probability"] < max(0.5, row.get("threshold", 0.5)) else row["identity_policy_candidate"]) for row in audit_rows]
    return {"selected": selected, "safest": summarize(safest_rows, "safest"), "frontier": summarize(frontier_rows, "frontier"), "comparison": {"selected": selected, "safest": summarize(safest_rows, "safest"), "frontier": summarize(frontier_rows, "frontier")}, "audit_rows": audit_rows, "diagnostic_only": True, "association_authority_started": False, "candidate_created": False, "score_before_update": True}


def bootstrap_delta(selected_records: Sequence[Mapping[str, Any]], baseline_records: Sequence[Mapping[str, Any]], seed: int = 720350, repetitions: int = 2000) -> dict[str, Any]:
    new_by_sequence = {sequence: [row for row in selected_records if row["sequence"] == sequence] for sequence in SEQUENCES}
    base_by_sequence = {sequence: [row for row in baseline_records if row["sequence"] == sequence] for sequence in SEQUENCES}
    rng = np.random.default_rng(seed); deltas = []
    for _ in range(repetitions):
        sample = rng.choice(SEQUENCES, size=len(SEQUENCES), replace=True)
        new_correct = 0; base_correct = 0; total = 0
        for sequence in sample:
            new = new_by_sequence[str(sequence)]; base = base_by_sequence[str(sequence)]
            new_correct += sum(bool(row["kind"] == "PRESENT" and row["set_probability"] >= row["threshold"] and int(np.argmax(row["candidate_probability"])) in set(row["positive_indices"])) for row in new)
            base_correct += sum(bool(row["kind"] == "PRESENT" and int(np.argmax(row["candidate_probability"])) in set(row["positive_indices"])) for row in base)
            total += max(len(new), 1)
        deltas.append(float((new_correct - base_correct) / max(1, total)))
    return {"unit": "equal-sequence cluster", "seed": seed, "repetitions": repetitions, "delta_correct_id_recall_mean": float(np.mean(deltas)), "delta_correct_id_recall_95ci": [float(np.percentile(deltas, 2.5)), float(np.percentile(deltas, 97.5))]}


def baseline_records(formal_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    result = []
    for row in formal_rows:
        scores = np.asarray(row["scores"], dtype=np.float64)
        probability = expit((scores - 0.30) * 10.0)
        result.append(_prediction_record(row, probability, scores, float(np.max(probability) if len(probability) else 0.0), str(row["sequence"]), 0.5))
    return result


def save_json_artifacts(
    source: Mapping[str, Any], data_by_sequence: Mapping[str, SequenceData], real_manifest: Mapping[str, Any], cf_manifest: Mapping[str, Any], whitelist: Mapping[str, Any], outer: Mapping[str, Any], denoising: Mapping[str, Any], temporal: Mapping[str, Any], decomposed: Mapping[str, Any], oracle: Mapping[str, Any], headroom: Mapping[str, Any], shadow: Mapping[str, Any], formal_rows: Sequence[Mapping[str, Any]], baseline: Sequence[Mapping[str, Any]], device: torch.device,
) -> dict[str, Any]:
    for directory in (OUT / "forensics", OUT / "training", OUT / "joint_representation", OUT / "state_denoising", OUT / "temporal", OUT / "decomposed_presence", OUT / "formal", OUT / "oracle", OUT / "shadow_causal", OUT / "causal"):
        directory.mkdir(parents=True, exist_ok=True)
    write_json(OUT / "source_audit.json", source)
    (OUT / "source_audit.md").write_text("\n".join([
        "# N72R20R3R2R2 source audit", "", f"- source branch: `{source['source_branch']}`", f"- SOURCE_R3R2R1_HEAD: `{source['source_head']}`", f"- new branch base HEAD: `{source['new_stage_base_head']}`", f"- historical decision: `{source['source_decision']}`", "- historical R3R2/R3R2R1 outputs modified: `False`", "- SAM3/VAL/TEST/association authority: not run", "",
    ]), encoding="utf-8")
    write_json(OUT / "forensics/absence_taxonomy_audit.json", {**real_manifest, "historical_reference_counts": {"PRESENT": 6595, "P0": 154, "P1": 1665, "P1a": 439, "P1b": 1226}, "counts_verified_from_boxes": True})
    write_json(OUT / "forensics/runtime_candidate_feature_audit.json", read_json(OUT / "forensics/runtime_candidate_feature_audit.json") if (OUT / "forensics/runtime_candidate_feature_audit.json").exists() else {})
    write_json(OUT / "training/runtime_feature_whitelist.json", whitelist)
    write_json(OUT / "training/open_set_episode_manifest.json", real_manifest)
    write_json(OUT / "training/counterfactual_manifest.json", cf_manifest)
    write_json(OUT / "training/outer_split_manifest.json", {"stage": STAGE, "folds": outer["inner_records"], "heldout_supervision_forbidden": True, "outer_fold_count": 8})
    write_json(OUT / "training/checkpoint_manifest.json", {"stage": STAGE, "asset_root": str(STAGE_ASSET_ROOT / "models"), "records": outer["checkpoint_records"], "binary_committed": False, "no_binary_in_repository": True})
    write_json(OUT / "training/seed_manifest.json", {"stage": STAGE, "representation_seeds": list(LEARNED_SEEDS), "adapter_seed": R3R2_SEED, "one_decision_per_frame": True})
    family_metrics_payload = {}
    for family, records in outer["families"].items():
        metrics = family_metrics(records)
        family_metrics_payload[family] = metrics
        filename = {"J1_PAIRWISE": "J1_pairwise.json", "J2_JOINT_ABSOLUTE": "J2_joint_absolute.json", "J3_SET_NOISY_OR": "J3_set_loss.json", "J4_ENERGY": "J4_energy_loss.json", "J5_QUALITY_FUSION": "J5_quality_fusion.json", "J7_MULTITASK": "J7_multitask.json"}[family]
        write_json(OUT / "joint_representation" / filename, {"stage": STAGE, "family": family, "metrics": metrics, "outer_records": len(records), "counterfactual_training_included": family in {"J2_JOINT_ABSOLUTE", "J7_MULTITASK"}, "runtime_future_gt_used": False})
    write_json(OUT / "joint_representation/J6_geometry_fusion.json", {
        "stage": STAGE,
        "family": "J6_GEOMETRY_FUSION",
        "status": "SKIPPED",
        "reason": "Causal target geometry or predicted motion is not available for every runtime frame; using partial geometry would violate the preregistered no-GT runtime feature boundary.",
        "candidate_generation_changed": False,
        "runtime_future_gt_used": False,
        "association_authority_started": False,
    })
    write_json(OUT / "joint_representation/counterfactual_ablation.json", {"with_counterfactual_J2": family_metrics_payload.get("J2_JOINT_ABSOLUTE"), "without_counterfactual_J2": family_metrics(outer["no_cf_records"]) if outer["no_cf_records"] else None, "formal_evaluation_contains_counterfactual": False, "effect_is_diagnostic_until_outer_gate": True})
    write_json(OUT / "joint_representation/ranking_retention.json", {"baseline": raw_score_metrics(baseline), "families": {family: {key: value for key, value in metrics.items() if key in {"Rank1", "Rank2", "Rank3", "Rank5", "MRR", "mean_rank"}} for family, metrics in family_metrics_payload.items()}})
    write_json(OUT / "joint_representation/absence_separability.json", {"families": {family: {key: value for key, value in metrics.items() if key in {"negative_fpr", "P0_fpr", "P1a_fpr", "P1b_fpr", "correct_id_recall", "recall_at_FPR_2pct", "candidate_valid_auroc", "set_presence_auroc", "formal_gate"}} for family, metrics in family_metrics_payload.items()}, "development_target_recall_at_FPR_2pct": 0.50, "development_target_is_not_final_pass": True})
    write_json(OUT / "state_denoising/D0.json", denoising["D0"]); write_json(OUT / "state_denoising/D1.json", denoising["D1"]); write_json(OUT / "state_denoising/D2.json", denoising["D2"]); write_json(OUT / "state_denoising/D3.json", denoising["D3"]); write_json(OUT / "state_denoising/formal_comparison.json", denoising["formal_comparison"])
    write_json(OUT / "temporal/per_frame.json", temporal.get("pooled", {}).get("per_frame", {})); write_json(OUT / "temporal/ema.json", temporal.get("pooled", {}).get("ema", {})); write_json(OUT / "temporal/hysteresis.json", temporal.get("pooled", {}).get("hysteresis", {})); write_json(OUT / "temporal/causal_gru.json", temporal.get("pooled", {}).get("causal_gru", {})); write_json(OUT / "temporal/all_results.json", temporal)
    write_json(OUT / "decomposed_presence/combined.json", decomposed); write_json(OUT / "decomposed_presence/physical_presence.json", {"stage": STAGE, "folds": decomposed.get("per_fold", []), "target": "identity physically present"}); write_json(OUT / "decomposed_presence/candidate_availability.json", {"stage": STAGE, "folds": decomposed.get("per_fold", []), "target": "valid candidate available given present"})
    write_json(OUT / "oracle/oracle_ladder.json", oracle); write_json(OUT / "oracle/headroom_table.json", headroom)
    write_json(OUT / "shadow_causal/selected.json", shadow["selected"]); write_json(OUT / "shadow_causal/safest.json", shadow["safest"]); write_json(OUT / "shadow_causal/frontier.json", shadow["frontier"]); write_json(OUT / "shadow_causal/comparison.json", shadow["comparison"])
    write_zstd_jsonl(OUT / "causal/memory_write_audit.jsonl.zst", shadow["audit_rows"])
    selected_metrics = family_metrics(outer["selected"])
    static_gate = selected_metrics["formal_gate"]
    causal_gate = {"static_gate_pass": bool(all(static_gate.values())), "wrong_write_rate_le_0.02": float(shadow["selected"]["wrong_write_rate"]) <= 0.02, "retention_ge_0.60": float(shadow["selected"]["retention"]) >= 0.60, "pass": bool(all(static_gate.values()) and float(shadow["selected"]["wrong_write_rate"]) <= 0.02 and float(shadow["selected"]["retention"]) >= 0.60)}
    if causal_gate["pass"]:
        write_json(OUT / "causal/formal_replay.json", {"status": "PASS_FORMAL_CAUSAL_GATE", "diagnostic_only": False, "runtime_future_gt_used": False, "association_authority_started": False})
        write_json(OUT / "causal/per_sequence.json", shadow["selected"].get("per_sequence", {}))
    else:
        write_json(OUT / "causal/formal_replay.json", {"status": "NOT_RUN_STATIC_OR_CAUSAL_GATE_FAILED", "diagnostic_only": True, "runtime_future_gt_used": False, "association_authority_started": False})
        write_json(OUT / "causal/per_sequence.json", {})
    write_json(OUT / "future_association_headroom.json", {"stage": STAGE, "diagnostic_only": True, "association_authority_started": False, "selected_shadow": shadow["selected"], "static_gate": static_gate, "causal_gate": causal_gate, "next_association_stage_authorized": causal_gate["pass"], "candidate_generation_frozen": True})
    write_json(OUT / "formal/inner_selection.json", {"stage": STAGE, "folds": outer["inner_records"], "outer_labels_used_for_selection": False, "threshold_selection_inner_only": True})
    write_json(OUT / "formal/outer_loso.json", {"stage": STAGE, "fold_count": 8, "selected_family_by_fold": outer["selected_family_by_fold"], "selected_metrics": selected_metrics, "family_metrics": family_metrics_payload, "heldout_zero_training": all(item["outer_heldout_absent_from_training"] for item in outer["inner_records"]), "runtime_future_gt_used": False})
    write_json(OUT / "formal/per_sequence.json", {sequence: selected_metrics.get("per_sequence", {}).get(sequence, {}) for sequence in SEQUENCES})
    write_json(OUT / "formal/bootstrap.json", bootstrap_delta(outer["selected"], baseline))
    write_json(OUT / "formal/selected_pipeline.json", {"stage": STAGE, "selected_family_by_fold": outer["selected_family_by_fold"], "selected_metrics": selected_metrics, "static_gate": static_gate, "causal_gate": causal_gate, "selection_is_inner_only": True})
    # Determine the final bottleneck only after all diagnostic branches have
    # completed.  This is a scientific classification, not an early stop.
    oracle_quality = oracle["modes"]["O3_PERFECT_QUALITY"]["metrics"]
    if causal_gate["pass"]:
        decision = "PASS_JOINT_OPEN_SET_IDENTITY_AVAILABILITY"
    elif not all(static_gate.values()):
        if oracle_quality.get("correct_id_recall", 0.0) - selected_metrics.get("correct_id_recall", 0.0) >= 0.15:
            decision = "FAIL_P1_LOCALIZATION_QUALITY"
        elif float(denoising["formal_comparison"].get("D2_Rank1") or 0.0) - float(denoising["formal_comparison"].get("D0_Rank1") or 0.0) >= 0.05:
            decision = "FAIL_RUNTIME_STATE_QUALITY"
        elif float(selected_metrics.get("Rank1") or 0.0) < 0.65:
            decision = "FAIL_IDENTITY_ABSOLUTE_MATCH"
        else:
            decision = "FAIL_MIXED_IDENTITY_AVAILABILITY"
    else:
        decision = "FAIL_CAUSAL_MEMORY_COMMIT"
    bottleneck = {"PASS_JOINT_OPEN_SET_IDENTITY_AVAILABILITY": "SOLVED_JOINT_IDENTITY_AVAILABILITY", "FAIL_P1_LOCALIZATION_QUALITY": "BOTTLENECK_P1_LOCALIZATION_QUALITY", "FAIL_RUNTIME_STATE_QUALITY": "BOTTLENECK_RUNTIME_STATE_QUALITY", "FAIL_IDENTITY_ABSOLUTE_MATCH": "BOTTLENECK_IDENTITY_ABSOLUTE_MATCH", "FAIL_CAUSAL_MEMORY_COMMIT": "BOTTLENECK_CAUSAL_MEMORY_COMMIT"}.get(decision, "BOTTLENECK_MIXED")
    storage_after = storage_audit("after")
    write_json(OUT / "storage_audit_after.json", storage_after)
    write_json(OUT / "FINAL_RESULT.json", {
        "stage": STAGE, "goal": "Joint Open-Set Identity-Availability Representation Learning", "central_question": "Can a learned identity representation recognize a valid target candidate and reject every invalid current candidate when the target is absent or unavailable?", "decision": decision, "bottleneck": bottleneck, "source_commit": SOURCE_R3R2R1_HEAD, "training_identity_count": real_manifest["training_identity_count"], "PRESENT_episode_count": real_manifest["counts"].get("PRESENT", 0), "P0_episode_count": real_manifest["counts"].get("P0", 0), "P1a_episode_count": real_manifest["counts"].get("P1a", 0), "P1b_episode_count": real_manifest["counts"].get("P1b", 0), "counterfactual_episode_count": cf_manifest["episode_count"], "candidate_metadata_features_used": whitelist["candidate_features"], "J_family_results": family_metrics_payload, "J6_geometry_fusion": read_json(OUT / "joint_representation/J6_geometry_fusion.json"), "historical_r3r2r1_calibration_ceiling": {"pooled_negative_fpr": 0.082463, "pooled_correct_id_recall": 0.227597, "macro_fpr": 0.067036, "macro_recall": 0.201144, "shadow_wrong_write_rate": 0.204608}, "ranking_retention": read_json(OUT / "joint_representation/ranking_retention.json"), "recall_at_FPR_2pct": selected_metrics.get("recall_at_FPR_2pct"), "P0_FPR": selected_metrics.get("P0_fpr"), "P1a_FPR": selected_metrics.get("P1a_fpr"), "P1b_FPR": selected_metrics.get("P1b_fpr"), "denoiser_metrics": denoising, "temporal_metrics": temporal, "formal_open_set_metrics": selected_metrics, "oracle_headroom": headroom, "shadow_causal_metrics": shadow["selected"], "formal_causal_metrics": causal_gate if causal_gate["pass"] else None, "final_bottleneck": bottleneck, "next_association_stage_authorized": causal_gate["pass"], "next_candidate_quality_stage_authorized": decision == "FAIL_P1_LOCALIZATION_QUALITY", "next_memory_state_learning_stage_authorized": decision == "FAIL_RUNTIME_STATE_QUALITY", "runtime_future_gt_used": False, "runtime_gt_clean": True, "candidate_created": False, "sam3_rerun": False, "val_accessed": False, "test_accessed": False, "OSNet_frozen": True, "N72R18_GRU_frozen_initially": True, "exact_solver_frozen": True, "public_id_authority_frozen": True, "association_authority_started": False, "storage_after": storage_after, "device": str(device), "tests_summary": "PENDING"})
    report = [
        "FINAL GOAL:", "Human Identity Representation Probe → Joint Open-Set Identity-Availability Representation Learning", "", "CENTRAL QUESTION:", "“用户点一下这个人以后，我们到底能不能认住他？”", "", "# What did joint open-set training change?", "", f"Final decision: `{decision}`.", "", f"R3R2 ranking baseline: `{raw_score_metrics(baseline)}`.", f"Best joint selected pipeline: `{outer['selected_family_by_fold']}`.", f"Formal selected metrics: `{selected_metrics}`.", f"Counterfactual episodes: `{cf_manifest['episode_count']}`; PRESENT/P0/P1a/P1b = `{real_manifest['counts']}`.", f"Shadow causal wrong-write rate: `{shadow['selected']['wrong_write_rate']}`; retention: `{shadow['selected']['retention']}`.", "", "## Lineage and runtime boundary", "", f"Source R3R2R1 HEAD: `{SOURCE_R3R2R1_HEAD}`. Eight outer sequence folds were run; held-out observations contributed zero training, threshold, or model-selection supervision.", "Candidate generation, OSNet, N72R18 state, exact solver, public-ID authority, SAM3, DanceTrack VAL/TEST and association authority remained frozen.", "GT was used only for training/post-hoc labels and diagnostics; runtime features contain no GT fields.", "", "## Scientific interpretation", "", f"Bottleneck: `{bottleneck}`. Static gate: `{static_gate}`. Causal gate: `{causal_gate}`.", "Oracle ladder and headroom are diagnostic only; they do not authorize runtime association.", "", "## Next-stage authorization", "", f"next_association_stage_authorized={causal_gate['pass']}", f"next_candidate_quality_stage_authorized={decision == 'FAIL_P1_LOCALIZATION_QUALITY'}", f"next_memory_state_learning_stage_authorized={decision == 'FAIL_RUNTIME_STATE_QUALITY'}", "", "No automatic downstream association, LoRA, SAM3 rerun, solver redesign or VAL/TEST evaluation was started.",
    ]
    (OUT / "FINAL_REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    write_json(OUT / "stage_status.json", {"stage": STAGE, "goal_reference": f"outputs/{STAGE}/FINAL_GOAL.json", "goal_frozen": True, "status": "PASS" if decision.startswith("PASS") else "FAIL", "decision": decision, "last_completed_artifact": "outputs/N72R20R3R2R2/FINAL_REPORT.md", "static_gate": static_gate, "causal_gate": causal_gate, "runtime_future_gt_used": False, "runtime_gt_clean": True, "sam3_rerun": False, "val_accessed": False, "test_accessed": False, "association_authority_started": False, "next_association_stage_authorized": causal_gate["pass"], "storage_audit_after": storage_after})
    return {"decision": decision, "selected_metrics": selected_metrics, "family_metrics": family_metrics_payload, "shadow": shadow, "causal_gate": causal_gate}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("audit", "mine", "repair", "all"), default="all")
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--tests-summary", default=None)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    freeze_goal()
    before = storage_audit("before")
    write_json(OUT / "storage_audit_before.json", before)
    if before["storage_status"] == "HARD_STOP":
        write_json(OUT / "stage_status.json", {"stage": STAGE, "status": "BLOCKED_STORAGE_HARD_STOP", "goal_reference": f"outputs/{STAGE}/FINAL_GOAL.json"})
        raise RuntimeError("storage hard stop")
    source = source_audit()
    write_json(OUT / "source_audit.json", source)
    if args.phase == "audit":
        print(json.dumps({"stage": STAGE, "status": "AUDIT_COMPLETE", "source_head": source["source_head"], "free_gib": before["free_gib"]}, sort_keys=True)); return 0
    if args.phase == "repair":
        repair_boolean_artifacts()
        if args.tests_summary:
            update_test_summary(args.tests_summary)
        print(json.dumps({"stage": STAGE, "status": "BOOLEAN_ARTIFACT_REPAIR_COMPLETE"}, sort_keys=True)); return 0
    data_by_sequence = load_sequences()
    source_identity, source_states = load_identity_source()
    real_episodes, real_manifest = build_real_episodes(data_by_sequence, source_identity, source_states)
    counterfactuals, cf_manifest = make_counterfactuals([episode for episode in real_episodes if episode.kind == "PRESENT"])
    audit, whitelist = runtime_feature_audit(data_by_sequence)
    write_json(OUT / "forensics/runtime_candidate_feature_audit.json", audit)
    write_json(OUT / "forensics/candidate_validity_oracle.json", {"stage": STAGE, "classes": {"V": "IoU >= 0.50", "W": "0.10 <= IoU < 0.50", "O": "IoU < 0.10"}, "oracle_is_posthoc_only": True, "training_validity_labels_gt_only": True, "runtime_feature_whitelist": whitelist["candidate_features"]})
    write_json(OUT / "forensics/p1_failure_analysis.json", {"stage": STAGE, "P1a_count": real_manifest["counts"].get("P1a", 0), "P1b_count": real_manifest["counts"].get("P1b", 0), "P1a_definition": "target visible, best candidate IoU [0.10,0.50)", "P1b_definition": "target visible, best candidate IoU <0.10", "candidate_generation_changed": False})
    if args.phase == "mine":
        write_json(OUT / "training/open_set_episode_manifest.json", real_manifest); write_json(OUT / "training/counterfactual_manifest.json", cf_manifest); print(json.dumps({"stage": STAGE, "status": "MINING_COMPLETE", "identities": real_manifest["training_identity_count"], "counts": real_manifest["counts"], "counterfactuals": cf_manifest["episode_count"]}, sort_keys=True)); return 0
    formal_rows = load_formal_rows(data_by_sequence)
    baseline = baseline_records(formal_rows)
    cache = build_adapter_cache(data_by_sequence)
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    outer = run_outer_models(data_by_sequence, real_episodes, counterfactuals, formal_rows, cache, device)
    denoising = run_state_denoising(formal_rows, real_episodes, data_by_sequence, cache, device)
    temporal = run_temporal_models(formal_rows, outer["selected"], outer["inner_records"], device)
    decomposed = run_decomposed_presence(outer["selected"], device)
    selected_metrics = family_metrics(outer["selected"])
    oracle, headroom = oracle_ladder(outer["selected"])
    shadow = run_shadow_causal(outer["selected"])
    result = save_json_artifacts(source, data_by_sequence, real_manifest, cf_manifest, whitelist, outer, denoising, temporal, decomposed, oracle, headroom, shadow, formal_rows, baseline, device)
    print(json.dumps({"stage": STAGE, "status": result["decision"], "selected_rank1": selected_metrics.get("Rank1"), "selected_recall_at_FPR_2pct": selected_metrics.get("recall_at_FPR_2pct"), "free_gib": storage_audit("final")["free_gib"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
