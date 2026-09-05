#!/usr/bin/env python3
"""Build the isolated N72R10 source-aware corpus with true future rows.

All tensors are built from frozen causal rows and the sealed N72R10 future
re-query artifacts.  Dataset GT is opened only after the candidate/context
tensor for a frame has been constructed, solely to attach an offline label.
No output from this script is used by runtime association.
"""

from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import tempfile
import traceback
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in __import__("sys").path:
    __import__("sys").path.insert(0, str(ROOT))

from sam3_intermot.reacquisition.target_candidate_pool import (  # noqa: E402
    FUTURE_FRAME_REQUERY,
    MAIN_B0_CANDIDATE,
    TARGET_SESSION_CURRENT_RAW,
    build_candidate_pool_with_future_requery,
)
from sam3_intermot.reacquisition.target_id_features import (  # noqa: E402
    CANDIDATE_FEATURE_DIM,
    candidate_feature_vector,
)
from scripts.n72r9_build_temporal_corpus import (  # noqa: E402
    HORIZON,
    MEMORY_SLOTS,
    TEMPORAL_FEATURE_DIM,
    base_scores_for_public,
    best_label,
    causal_selection_score,
    dimensions,
    load_gt,
    normalized_mean,
    padded_memory,
    read_json,
    read_jsonl,
    unit,
)


PROTOCOL_PATH = ROOT / "outputs/N72R9/protocol.json"
N72R10_AUDIT_PATH = ROOT / "outputs/N72R10/stage_03_true_future_requery/batch_integrity_audit.json"
OUTPUT_ROOT = ROOT / "outputs/N72R10/training"
STAGE_PATH = ROOT / "outputs/N72R10/stage_04_status.json"
SOURCE_NAMES = (
    MAIN_B0_CANDIDATE,
    TARGET_SESSION_CURRENT_RAW,
    "STATIC_EVENT_REQUERY",
    FUTURE_FRAME_REQUERY,
    "UNKNOWN",
)
SOURCE_FEATURE_DIM = len(SOURCE_NAMES)
IOU_THRESHOLD = 0.50
SEED = 7210


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
    atomic_write(path, json.dumps(dict(value), ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n")


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


def source_vector(source: str) -> np.ndarray:
    value = np.zeros(SOURCE_FEATURE_DIM, dtype=np.float32)
    try:
        value[SOURCE_NAMES.index(str(source))] = 1.0
    except ValueError:
        value[-1] = 1.0
    return value


def validate_frozen_rows(event: Mapping[str, Any], rows: Sequence[Mapping[str, Any]], label: str) -> None:
    event_frame = int(event["event_frame"])
    expected = list(range(event_frame, event_frame + HORIZON + 1))
    if [int(row.get("frame", -1)) for row in rows] != expected or len(rows) != HORIZON + 1:
        raise RuntimeError(f"{event['event_id']}:{label} frame axis is incomplete")
    for row in rows:
        for key in ("runtime_future_gt_used", "runtime_gt_read", "posthoc_gt_used"):
            if row.get(key) is not False:
                raise RuntimeError(f"{event['event_id']}:{label}:{row.get('frame')} {key} is not false")


def load_future_rows(audit: Mapping[str, Any]) -> dict[str, dict[int, list[dict[str, Any]]]]:
    result: dict[str, dict[int, list[dict[str, Any]]]] = {}
    for item in audit.get("event_rows", []):
        event_id = str(item["event_id"])
        artifact_dir = ROOT / str(item["artifact_dir"])
        if not artifact_dir.is_dir():
            raise FileNotFoundError(artifact_dir)
        candidates = read_json(artifact_dir / "candidates.json")
        rows = list(candidates.get("future_candidates", []))
        if candidates.get("runtime_future_gt_used") is not False or candidates.get("posthoc_gt_used") is not False:
            raise RuntimeError(f"{event_id}: future artifact has an invalid GT flag")
        by_frame: dict[int, list[dict[str, Any]]] = {}
        for row in rows:
            if row.get("candidate_source") != FUTURE_FRAME_REQUERY or row.get("public_id") is not None:
                raise RuntimeError(f"{event_id}: invalid future candidate source/public authority")
            if row.get("runtime_future_gt_used") is not False:
                raise RuntimeError(f"{event_id}: future candidate runtime GT flag is not false")
            by_frame.setdefault(int(row["frame"]), []).append(dict(row))
        for frame, frame_rows in by_frame.items():
            if len({str(row["candidate_uid"]) for row in frame_rows}) != len(frame_rows):
                raise RuntimeError(f"{event_id}:{frame}: duplicate future candidate UID")
        result[event_id] = by_frame
    if len(result) != 32:
        raise RuntimeError(f"future artifact index expected 32 events, found {len(result)}")
    return result


def _event_target_public_id(event: Mapping[str, Any]) -> int:
    manifest = read_json(Path(str(event["source_event_manifest"])))
    value = manifest.get("target_public_id")
    if value is None or int(value) <= 0:
        raise RuntimeError(f"{event['event_id']}: missing frozen target public authority")
    return int(value)


def build_event(
    event: Mapping[str, Any],
    gt: Mapping[int, Mapping[int, Sequence[float]]],
    future_by_frame: Mapping[int, Sequence[Mapping[str, Any]]],
) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray], list[np.ndarray], list[np.ndarray], list[np.ndarray], list[np.ndarray], list[np.ndarray], list[int], list[dict[str, Any]], dict[str, Any]]:
    event_id = str(event["event_id"])
    sequence = str(event["sequence"])
    event_frame = int(event["event_frame"])
    target_public_id = _event_target_public_id(event)
    c0_rows = read_jsonl(Path(str(event["c0_source"])))
    c1_rows = read_jsonl(Path(str(event["c1_source"])))
    target_rows = read_jsonl(Path(str(event["target_stream_source"])))
    validate_frozen_rows(event, c0_rows, "c0")
    validate_frozen_rows(event, c1_rows, "c1")
    validate_frozen_rows(event, target_rows, "target")
    anchor_payload = read_json(Path(str(event["target_stream_source"])).parent / "human_anchor.json")
    anchor = unit(anchor_payload.get("feature"), f"{event_id}:human_anchor")
    anchor_box = [float(value) for value in event["current_gt_box"]]
    width, height = dimensions(sequence, event_frame)
    c0_by_frame = {int(row["frame"]): row for row in c0_rows}
    c1_by_frame = {int(row["frame"]): row for row in c1_rows}
    target_by_frame = {int(row["frame"]): row for row in target_rows}
    event_target_rows = list(target_by_frame[event_frame].get("candidate_rows", []))
    previous_raw = event_target_rows[0].get("official_raw_sam_id") if event_target_rows else None
    previous_scope = event_target_rows[0].get("native_tid_scope") if event_target_rows else None
    previous_raw = None if previous_raw is None else int(previous_raw)
    previous_scope = None if previous_scope is None else str(previous_scope)
    predicted_box = list(anchor_box)
    velocity = np.zeros(2, dtype=np.float64)
    trusted: list[np.ndarray] = [anchor]
    distractors: list[np.ndarray] = []
    previous_score = 0.0
    previous_uncertainty = 1.0
    trusted_age = 0
    examples: list[np.ndarray] = []
    source_arrays: list[np.ndarray] = []
    trusted_arrays: list[np.ndarray] = []
    trusted_masks: list[np.ndarray] = []
    distractor_arrays: list[np.ndarray] = []
    distractor_masks: list[np.ndarray] = []
    neighbor_arrays: list[np.ndarray] = []
    temporal_arrays: list[np.ndarray] = []
    labels: list[int] = []
    metadata: list[dict[str, Any]] = []
    label_counts: Counter[str] = Counter()
    source_counts: Counter[str] = Counter()
    future_rows_total = 0
    future_rows_selected_as_label = 0
    future_rows_not_label = 0
    for frame in range(event_frame + 1, event_frame + HORIZON + 1):
        c0 = c0_by_frame[frame]
        c1 = c1_by_frame[frame]
        current = target_by_frame[frame]
        main_raw = list(c0.get("candidate_rows", []))
        current_raw = list(current.get("candidate_rows", []))
        live_raw = [dict(row) for row in future_by_frame.get(frame, ())]
        pool, pool_audit = build_candidate_pool_with_future_requery(
            main_raw,
            current_raw,
            live_raw,
            sequence=sequence,
            frame=frame,
        )
        if not pool:
            raise RuntimeError(f"{event_id}:{frame} has empty candidate pool")
        c0_scores = base_scores_for_public(c0, target_public_id, f"{event_id}:c0:{frame}")
        c1_scores = base_scores_for_public(c1, target_public_id, f"{event_id}:c1:{frame}")
        base_scores = {
            str(candidate["candidate_uid"]): float(
                c0_scores.get(
                    str(candidate["candidate_uid"]),
                    c1_scores.get(str(candidate["candidate_uid"]), 0.0),
                )
            )
            for candidate in pool
        }
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
                    candidate_count=len(pool),
                    base_target_score=base_scores[str(candidate["candidate_uid"])],
                )
                for candidate in pool
            ],
            axis=0,
        ).astype(np.float32)
        if candidate_vectors.shape != (len(pool), CANDIDATE_FEATURE_DIM) or not np.all(np.isfinite(candidate_vectors)):
            raise RuntimeError(f"{event_id}:{frame} candidate tensor is invalid")
        causal_scores = np.asarray(
            [
                causal_selection_score(
                    candidate,
                    base_scores[str(candidate["candidate_uid"])],
                    trusted,
                    predicted_box,
                )
                for candidate in pool
            ],
            dtype=np.float64,
        )
        order = sorted(range(len(pool)), key=lambda index: (-causal_scores[index], str(pool[index]["candidate_uid"])))
        selected_index = int(order[0])
        second_score = float(causal_scores[order[1]]) if len(order) > 1 else 0.0
        top_score = float(causal_scores[selected_index])
        margin = float(top_score - second_score)
        neighbor_values = [candidate_vectors[index, :512] for index in order[1:] if np.linalg.norm(candidate_vectors[index, :512]) > 1.0e-6]
        neighbor = normalized_mean(neighbor_values, anchor)
        trusted_array, trusted_mask = padded_memory(trusted, MEMORY_SLOTS)
        distractor_array, distractor_mask = padded_memory(distractors, MEMORY_SLOTS)
        temporal = np.asarray(
            [
                float(frame - event_frame) / float(HORIZON),
                float(np.tanh(top_score)),
                float(np.tanh(second_score)),
                float(np.tanh(margin)),
                float(np.clip(previous_score, -1.0, 1.0)),
                float(np.clip(previous_uncertainty, 0.0, 1.0)),
                float(min(trusted_age, HORIZON)) / float(HORIZON),
                float(any(str(candidate["candidate_source"]) == TARGET_SESSION_CURRENT_RAW for candidate in pool)),
            ],
            dtype=np.float32,
        )
        # GT is intentionally opened only after all causal tensors are ready.
        target_box = gt.get(frame, {}).get(int(event["dataset_gt_id"]))
        label, reason, best_iou_value, visible = best_label(pool, target_box)
        label_counts[reason] += 1
        examples.append(candidate_vectors)
        source_arrays.append(np.stack([source_vector(str(candidate["candidate_source"])) for candidate in pool], axis=0))
        trusted_arrays.append(trusted_array)
        trusted_masks.append(trusted_mask)
        distractor_arrays.append(distractor_array)
        distractor_masks.append(distractor_mask)
        neighbor_arrays.append(neighbor)
        temporal_arrays.append(temporal)
        labels.append(int(label))
        future_indices = [index for index, candidate in enumerate(pool) if str(candidate["candidate_source"]) == FUTURE_FRAME_REQUERY]
        future_rows_total += len(future_indices)
        if int(label) in future_indices:
            future_rows_selected_as_label += 1
        future_rows_not_label += sum(int(index != label) for index in future_indices)
        for candidate in pool:
            source_counts[str(candidate["candidate_source"])] += 1
        metadata.append(
            {
                "event_id": event_id,
                "sequence": sequence,
                "split": str(event["split"]),
                "action_type": str(event["action_type"]),
                "event_frame": event_frame,
                "frame": frame,
                "frame_horizon": frame - event_frame,
                "candidate_count": len(pool),
                "candidate_uids": [str(candidate["candidate_uid"]) for candidate in pool],
                "candidate_sources": [str(candidate["candidate_source"]) for candidate in pool],
                "candidate_feature_sha256": [str(candidate.get("feature_sha256")) for candidate in pool],
                "future_candidate_uids": [str(pool[index]["candidate_uid"]) for index in future_indices],
                "future_candidate_count": len(future_indices),
                "causal_selected_index": selected_index,
                "causal_selected_source": str(pool[selected_index]["candidate_source"]),
                "label_index_unpadded": int(label),
                "label_kind": "NONE" if int(label) == len(pool) else "TARGET_CANDIDATE",
                "label_reason": reason,
                "posthoc_target_visible": bool(visible),
                "posthoc_best_iou": float(best_iou_value),
                "gt_used_offline": True,
                "runtime_future_gt_used": False,
                "public_id_inference": False,
                "not_real_human_evidence": True,
                "candidate_pool_audit": pool_audit,
                "causal_state_update": "base_score_plus_previous_trusted_similarity_geometry_presence",
            }
        )
        selected = pool[selected_index]
        selected_feature = selected.get("feature")
        if selected_feature is not None:
            selected_feature = unit(selected_feature, f"{event_id}:{frame}:selected")
            old_center = np.asarray([(predicted_box[0] + predicted_box[2]) / 2.0, (predicted_box[1] + predicted_box[3]) / 2.0])
            new_box = [float(value) for value in selected["box_xyxy"]]
            new_center = np.asarray([(new_box[0] + new_box[2]) / 2.0, (new_box[1] + new_box[3]) / 2.0])
            velocity = 0.5 * velocity + 0.5 * (new_center - old_center)
            predicted_box = new_box
            if top_score > 0.0 and margin >= 0.20:
                trusted.append(selected_feature)
                trusted_age = 0
            else:
                trusted_age += 1
            if len(order) > 1:
                second = pool[order[1]].get("feature")
                if second is not None:
                    distractors.append(unit(second, f"{event_id}:{frame}:distractor"))
        else:
            trusted_age += 1
        previous_raw = selected.get("official_raw_sam_id")
        previous_raw = None if previous_raw is None else int(previous_raw)
        previous_scope = selected.get("native_scope")
        previous_scope = None if previous_scope is None else str(previous_scope)
        previous_score = top_score
        previous_uncertainty = float(1.0 / (1.0 + max(margin, 0.0)))
        trusted = trusted[-MEMORY_SLOTS:]
        distractors = distractors[-MEMORY_SLOTS:]
    return (
        examples,
        source_arrays,
        trusted_arrays,
        trusted_masks,
        distractor_arrays,
        distractor_masks,
        neighbor_arrays,
        temporal_arrays,
        labels,
        metadata,
        {
            "event_count": 1,
            "sequence_count": 1,
            "label_counts": dict(sorted(label_counts.items())),
            "source_counts": dict(sorted(source_counts.items())),
            "future_rows_total": future_rows_total,
            "future_rows_selected_as_label": future_rows_selected_as_label,
            "future_rows_not_label": future_rows_not_label,
        },
    )


def build_split(
    events: Sequence[Mapping[str, Any]],
    gt_by_sequence: Mapping[str, Mapping[int, Mapping[int, Sequence[float]]]],
    future_by_event: Mapping[str, Mapping[int, Sequence[Mapping[str, Any]]]],
) -> dict[str, Any]:
    event_examples: list[np.ndarray] = []
    source_features: list[np.ndarray] = []
    trusted_memory: list[np.ndarray] = []
    trusted_mask: list[np.ndarray] = []
    distractor_memory: list[np.ndarray] = []
    distractor_mask: list[np.ndarray] = []
    neighbor_feature: list[np.ndarray] = []
    temporal_features: list[np.ndarray] = []
    labels: list[int] = []
    metadata: list[dict[str, Any]] = []
    label_counts: Counter[str] = Counter()
    source_counts: Counter[str] = Counter()
    max_candidates = 0
    future_rows_total = 0
    future_rows_selected_as_label = 0
    future_rows_not_label = 0
    for event in sorted(events, key=lambda item: str(item["event_id"])):
        values = build_event(event, gt_by_sequence[str(event["sequence"])], future_by_event[str(event["event_id"])])
        examples, sources, trusted, trusted_masks, distractors, distractor_masks, neighbors, temporal, event_labels, event_metadata, details = values
        event_examples.extend(examples)
        source_features.extend(sources)
        trusted_memory.extend(trusted)
        trusted_mask.extend(trusted_masks)
        distractor_memory.extend(distractors)
        distractor_mask.extend(distractor_masks)
        neighbor_feature.extend(neighbors)
        temporal_features.extend(temporal)
        labels.extend(event_labels)
        metadata.extend(event_metadata)
        label_counts.update(details["label_counts"])
        source_counts.update(details["source_counts"])
        future_rows_total += int(details["future_rows_total"])
        future_rows_selected_as_label += int(details["future_rows_selected_as_label"])
        future_rows_not_label += int(details["future_rows_not_label"])
        max_candidates = max(max_candidates, max(len(value) for value in examples))
    count = len(event_examples)
    candidate_features = np.zeros((count, max_candidates, CANDIDATE_FEATURE_DIM), dtype=np.float32)
    padded_sources = np.zeros((count, max_candidates, SOURCE_FEATURE_DIM), dtype=np.float32)
    candidate_mask = np.zeros((count, max_candidates), dtype=np.bool_)
    for index, value in enumerate(event_examples):
        width = len(value)
        candidate_features[index, :width] = value
        padded_sources[index, :width] = source_features[index]
        candidate_mask[index, :width] = True
    none_index = max_candidates
    encoded_labels = np.asarray(
        [none_index if int(label) == int(metadata[index]["candidate_count"]) else int(label) for index, label in enumerate(labels)],
        dtype=np.int64,
    )
    arrays = {
        "candidate_features": candidate_features,
        "candidate_mask": candidate_mask,
        "source_features": padded_sources,
        "trusted_memory": np.stack(trusted_memory, axis=0).astype(np.float32),
        "trusted_mask": np.stack(trusted_mask, axis=0).astype(np.bool_),
        "distractor_memory": np.stack(distractor_memory, axis=0).astype(np.float32),
        "distractor_mask": np.stack(distractor_mask, axis=0).astype(np.bool_),
        "neighbor_feature": np.stack(neighbor_feature, axis=0).astype(np.float32),
        "temporal_features": np.stack(temporal_features, axis=0).astype(np.float32),
        "labels": encoded_labels,
        "candidate_counts": np.asarray([int(item["candidate_count"]) for item in metadata], dtype=np.int64),
    }
    for key, value in arrays.items():
        if not np.all(np.isfinite(value)):
            raise RuntimeError(f"non-finite N72R10 corpus array: {key}")
    return {
        "arrays": arrays,
        "metadata": metadata,
        "summary": {
            "event_count": len(events),
            "sequence_count": len({str(event["sequence"]) for event in events}),
            "example_count": count,
            "max_candidates": max_candidates,
            "candidate_feature_dim": CANDIDATE_FEATURE_DIM,
            "source_feature_dim": SOURCE_FEATURE_DIM,
            "source_feature_names": list(SOURCE_NAMES),
            "trusted_memory_slots": MEMORY_SLOTS,
            "distractor_memory_slots": MEMORY_SLOTS,
            "temporal_feature_dim": TEMPORAL_FEATURE_DIM,
            "label_counts": dict(sorted(label_counts.items())),
            "source_counts": dict(sorted(source_counts.items())),
            "future_rows_total": future_rows_total,
            "future_rows_selected_as_label": future_rows_selected_as_label,
            "future_rows_not_label": future_rows_not_label,
            "runtime_future_gt_used": False,
            "gt_used_only_offline_label_generation": True,
        },
    }


def main() -> int:
    started = now_utc()
    base = {
        "schema_version": "N72R10_TEMPORAL_CORPUS_STATUS_V1",
        "stage": "N72R10_BUILD_FUTURE_REQUERY_TRAINING_DISTRIBUTION",
        "started_at_utc": started,
        "runtime_future_gt_used": False,
        "interaction_source": "simulated_from_gt",
        "not_real_human_evidence": True,
    }
    try:
        protocol = read_json(PROTOCOL_PATH)
        events = [dict(item) for item in protocol.get("source_event_selection", {}).get("events", [])]
        if len(events) != 32 or len({str(item["event_id"]) for item in events}) != 32:
            raise RuntimeError(f"expected 32 frozen events, found {len(events)}")
        audit = read_json(N72R10_AUDIT_PATH)
        if audit.get("status") != "PASS_N72R10_TRUE_FUTURE_REQUERY_BATCH_AUDIT":
            raise RuntimeError("N72R10 future artifact audit must pass before corpus construction")
        future_by_event = load_future_rows(audit)
        gt_by_sequence = {sequence: load_gt(sequence) for sequence in sorted({str(item["sequence"]) for item in events})}
        split_outputs: dict[str, dict[str, Any]] = {}
        for split in ("train", "validation"):
            selected = [item for item in events if str(item["split"]) == split]
            if not selected:
                raise RuntimeError(f"empty {split} split")
            split_outputs[split] = build_split(selected, gt_by_sequence, future_by_event)
            atomic_npz(OUTPUT_ROOT / f"{split}.npz", **split_outputs[split]["arrays"])
            atomic_jsonl(OUTPUT_ROOT / f"{split}_metadata.jsonl", split_outputs[split]["metadata"])
        manifest: dict[str, Any] = {
            "schema_version": "N72R10_SOURCE_AWARE_TEMPORAL_CORPUS_V2",
            "status": "PASS_N72R10_SOURCE_AWARE_CORPUS_SEALED",
            "created_at_utc": now_utc(),
            "protocol": str(PROTOCOL_PATH),
            "protocol_sha256": sha256_file(PROTOCOL_PATH),
            "future_audit": str(N72R10_AUDIT_PATH),
            "future_audit_sha256": sha256_file(N72R10_AUDIT_PATH),
            "events": len(events),
            "interaction_source": "simulated_from_gt",
            "not_real_human_evidence": True,
            "gt_used_only_offline_label_generation": True,
            "runtime_future_gt_used": False,
            "source_feature_names": list(SOURCE_NAMES),
            "future_source_definition": "sealed official future-frame re-query candidate rows only",
            "splits": {},
        }
        for split, value in split_outputs.items():
            npz_path = OUTPUT_ROOT / f"{split}.npz"
            metadata_path = OUTPUT_ROOT / f"{split}_metadata.jsonl"
            manifest["splits"][split] = {
                **value["summary"],
                "npz": str(npz_path),
                "npz_sha256": sha256_file(npz_path),
                "metadata": str(metadata_path),
                "metadata_sha256": sha256_file(metadata_path),
            }
        atomic_json(OUTPUT_ROOT / "corpus_manifest.json", manifest)
        result = {
            **base,
            "status": "PASS_N72R10_SOURCE_AWARE_CORPUS_SEALED",
            "finished_at_utc": now_utc(),
            "protocol": str(PROTOCOL_PATH),
            "protocol_sha256": manifest["protocol_sha256"],
            "future_audit": str(N72R10_AUDIT_PATH),
            "future_audit_sha256": manifest["future_audit_sha256"],
            "corpus_manifest": str(OUTPUT_ROOT / "corpus_manifest.json"),
            "corpus_manifest_sha256": sha256_file(OUTPUT_ROOT / "corpus_manifest.json"),
            "source_feature_names": list(SOURCE_NAMES),
            "train_examples": int(split_outputs["train"]["summary"]["example_count"]),
            "validation_examples": int(split_outputs["validation"]["summary"]["example_count"]),
            "train_future_rows": int(split_outputs["train"]["summary"]["future_rows_total"]),
            "validation_future_rows": int(split_outputs["validation"]["summary"]["future_rows_total"]),
            "train_future_rows_not_label": int(split_outputs["train"]["summary"]["future_rows_not_label"]),
            "validation_future_rows_not_label": int(split_outputs["validation"]["summary"]["future_rows_not_label"]),
            "fresh_confirmation_authorized": False,
            "production_authorized": False,
        }
        atomic_json(STAGE_PATH, result)
        print(json.dumps(result, ensure_ascii=False, sort_keys=True))
        return 0
    except Exception as exc:
        failure = OUTPUT_ROOT / "attempts" / f"corpus_failure_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.json"
        payload = {
            **base,
            "status": "FAIL_N72R10_SOURCE_AWARE_CORPUS",
            "finished_at_utc": now_utc(),
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }
        atomic_json(failure, payload)
        atomic_json(STAGE_PATH, {**payload, "failure_artifact": str(failure)})
        print(json.dumps({"status": payload["status"], "failure_artifact": str(failure), "error": str(exc)}, sort_keys=True))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
