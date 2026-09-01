#!/usr/bin/env python3
"""Materialize the N71 candidate-by-identity matrix from the frozen N70 cache.

The cache already contains the complete N54/N70 candidate-by-public-ID score
matrix.  This script expands each matrix cell into an isolated training row;
public IDs and native IDs are retained only as offline labels/audit metadata,
never as model features.  All writes use a temporary file followed by an
atomic rename, and an incomplete attempt is retained for diagnosis.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
import traceback
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import n70_association_common as n70  # noqa: E402


N71_PROTOCOL = ROOT / "outputs/N71/protocol.json"
EVENT_MANIFEST = ROOT / "outputs/n37/real_event_manifest.json"
OUTPUT_ROOT = Path("/path/to/cache/SAM3_InterMOT_N71/training/N71_GLOBAL_MATRIX_DATASET_ATTEMPT1")
MANIFEST = ROOT / "outputs/N71/training/global_matrix_dataset_manifest.json"
FEAT_DIM = 512
SCALAR_DIM = 8
CONTEXT_DIM = 15
SPLIT_CODE = {"train": 0, "validation": 1, "holdout": 2}


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
        dfd = os.open(str(path.parent), os.O_DIRECTORY)
        try:
            os.fsync(dfd)
        finally:
            os.close(dfd)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def load_n71() -> tuple[dict[str, Any], dict[str, dict[str, Any]], dict[str, str]]:
    protocol = json.loads(N71_PROTOCOL.read_text(encoding="utf-8"))
    if protocol.get("status") != "FROZEN_BEFORE_N71_EXECUTION":
        raise RuntimeError("N71 protocol is not frozen")
    if protocol.get("runtime_contract", {}).get("runtime_future_gt_used") is not False:
        raise RuntimeError("N71 protocol runtime GT boundary is not false")
    split: dict[str, str] = {}
    for name in ("train", "validation", "holdout"):
        sequences = protocol["sequence_split"].get(name, [])
        for sequence in sequences:
            sequence = str(sequence)
            if sequence in split:
                raise RuntimeError(f"sequence appears in multiple N71 splits: {sequence}")
            split[sequence] = str(name)

    raw = json.loads(EVENT_MANIFEST.read_text(encoding="utf-8"))
    events: dict[str, dict[str, Any]] = {}
    for item in raw.get("events", []):
        event = item.get("event", item)
        event_id = str(item.get("protocol_candidate_id") or event.get("event_id"))
        if event_id in events:
            raise RuntimeError(f"duplicate event id: {event_id}")
        if event.get("interaction_source") != "simulated_from_gt" or item.get("interaction_source") != "simulated_from_gt":
            raise RuntimeError(f"event provenance is not explicit simulated_from_gt: {event_id}")
        if event.get("runtime_future_gt_used") is not False or item.get("runtime_future_gt_used") is not False:
            raise RuntimeError(f"event runtime GT boundary failed: {event_id}")
        anchor = np.asarray(event.get("human_embedding"), dtype=np.float32).reshape(-1)
        if anchor.shape != (FEAT_DIM,) or not np.all(np.isfinite(anchor)) or float(np.linalg.norm(anchor)) <= 1e-6:
            raise RuntimeError(f"invalid event human anchor: {event_id}")
        competitor_vectors = []
        for competitor in event.get("competing_embeddings", []):
            if not isinstance(competitor, dict):
                continue
            vector = np.asarray(competitor.get("embedding"), dtype=np.float32).reshape(-1)
            if vector.shape == (FEAT_DIM,) and np.all(np.isfinite(vector)) and float(np.linalg.norm(vector)) > 1e-6:
                competitor_vectors.append(vector / np.linalg.norm(vector))
        if competitor_vectors:
            negative = np.mean(np.stack(competitor_vectors), axis=0).astype(np.float32)
            norm = float(np.linalg.norm(negative))
            negative = negative / norm if norm > 1e-6 else np.zeros(FEAT_DIM, dtype=np.float32)
        else:
            negative = np.zeros(FEAT_DIM, dtype=np.float32)
        events[event_id] = {
            "event_id": event_id,
            "sequence": str(event["sequence"]),
            "event_frame": int(event["frame"]),
            "target_public_id": int(event["public_id"]),
            "target_native_id": int(event["target_native_tid"]),
            "human_embedding": anchor,
            "negative_embedding": negative,
            "action_type": str(event.get("action_type") or item.get("action_type")),
            "interaction_source": "simulated_from_gt",
        }
    if len(events) != 24:
        raise RuntimeError(f"expected 24 N37 events, found {len(events)}")
    return protocol, events, split


def validate_frame(frame: dict[str, Any], event: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[int], list[dict[str, Any]]]:
    if frame.get("runtime_future_gt_used") is not False or frame.get("runtime_gt_read") is not False:
        raise RuntimeError(f"runtime future GT boundary failed: {frame.get('event_id')}/{frame.get('variant')}/{frame.get('frame')}")
    if frame.get("candidate_set_complete") is not True:
        raise RuntimeError(f"incomplete candidate set: {frame.get('event_id')}/{frame.get('frame')}")
    candidate_rows = frame.get("candidate_rows")
    rows = frame.get("rows")
    public_ids = [int(value) for value in frame.get("public_id_order", [])]
    candidate_features = np.asarray(frame.get("candidate_features_512"), dtype=np.float32)
    memory = np.asarray(frame.get("memory_vectors_512"), dtype=np.float32)
    memory_valid = np.asarray(frame.get("memory_valid"), dtype=bool).reshape(-1)
    score = np.asarray(frame.get("score_matrix"), dtype=np.float32)
    scalar = np.asarray(frame.get("scalar_features_8"), dtype=np.float32)
    if not isinstance(candidate_rows, list) or not isinstance(rows, list):
        raise RuntimeError("candidate rows/rows are not lists")
    n, p = len(candidate_rows), len(public_ids)
    if n <= 0 or p <= 0:
        raise RuntimeError(f"empty candidate or identity set: n={n}, p={p}")
    if candidate_features.shape != (n, FEAT_DIM) or memory.shape != (p, FEAT_DIM) or score.shape != (n, p):
        raise RuntimeError(f"matrix shape mismatch n={n} p={p} candidate={candidate_features.shape} memory={memory.shape} score={score.shape}")
    if memory_valid.shape != (p,) or scalar.shape != (n * p, SCALAR_DIM):
        raise RuntimeError(f"context shape mismatch memory_valid={memory_valid.shape} scalar={scalar.shape} n={n} p={p}")
    for name, value in (("candidate", candidate_features), ("memory", memory), ("score", score), ("scalar", scalar)):
        if not np.all(np.isfinite(value)):
            raise RuntimeError(f"nonfinite {name} values")
    if len(set(public_ids)) != p:
        raise RuntimeError("duplicate public IDs in identity matrix columns")
    if [int(row.get("index", -1)) for row in candidate_rows] != list(range(n)):
        raise RuntimeError("candidate order/index changed")
    if [int(row.get("candidate_index", -1)) for row in rows] != list(range(n)):
        raise RuntimeError("rows order/index changed")
    assignment_columns = np.asarray(frame.get("assignment_columns"), dtype=np.int64).reshape(-1)
    if assignment_columns.shape != (n,):
        raise RuntimeError("assignment_columns shape mismatch")
    if np.any((assignment_columns < -1) | (assignment_columns >= p)):
        raise RuntimeError("assignment column outside identity matrix")
    return candidate_features, memory, score, scalar, public_ids, rows


def count_cells(events: dict[str, dict[str, Any]]) -> tuple[int, int, Counter[str]]:
    cells = 0
    groups = 0
    split_groups: Counter[str] = Counter()
    for event_id, frame in n70.iter_cache_frames(events):
        event = events[event_id]
        candidate_features, memory, score, scalar, public_ids, rows = validate_frame(frame, event)
        cells += int(candidate_features.shape[0] * len(public_ids))
        groups += 1
        split_groups[event["sequence"]] += 1
    if groups != 12000:
        raise RuntimeError(f"expected 12000 N70 cache groups, found {groups}")
    return cells, groups, split_groups


def open_temp_arrays(root: Path, total_cells: int, groups: int) -> tuple[dict[str, np.memmap], dict[str, Path]]:
    specs = {
        "candidate": (np.float16, (total_cells, FEAT_DIM)),
        "identity_memory": (np.float16, (total_cells, FEAT_DIM)),
        "human_anchor": (np.float16, (total_cells, FEAT_DIM)),
        "hard_negative": (np.float16, (total_cells, FEAT_DIM)),
        "context": (np.float32, (total_cells, CONTEXT_DIM)),
        "label": (np.uint8, (total_cells,)),
        "none_label": (np.uint8, (total_cells,)),
        "group": (np.int32, (total_cells,)),
        "split": (np.uint8, (total_cells,)),
        "frame": (np.int32, (total_cells,)),
        "variant": (np.uint8, (total_cells,)),
        "candidate_slot": (np.uint16, (total_cells,)),
        "identity_slot": (np.uint16, (total_cells,)),
        "target_slot": (np.int16, (total_cells,)),
        "target_row": (np.int16, (total_cells,)),
        "target_present": (np.uint8, (total_cells,)),
        "base_score": (np.float32, (total_cells,)),
    }
    arrays: dict[str, np.memmap] = {}
    temp_paths: dict[str, Path] = {}
    root.mkdir(parents=True, exist_ok=True)
    for name, (dtype, shape) in specs.items():
        path = root / f"{name}.npy"
        temp = root / f".{name}.npy.tmp"
        if temp.exists():
            raise RuntimeError(f"stale temp array exists; use a new attempt root: {temp}")
        arrays[name] = np.lib.format.open_memmap(temp, mode="w+", dtype=dtype, shape=shape)
        temp_paths[name] = temp
    offsets = np.lib.format.open_memmap(root / ".group_offsets.npy.tmp", mode="w+", dtype=np.int64, shape=(groups + 1,))
    arrays["group_offsets"] = offsets
    temp_paths["group_offsets"] = root / ".group_offsets.npy.tmp"
    return arrays, temp_paths


def write_matrix_dataset(output_root: Path, manifest_path: Path) -> dict[str, Any]:
    protocol, events, split = load_n71()
    total_cells, group_count, split_groups = count_cells(events)
    if total_cells <= 0:
        raise RuntimeError("no global matrix cells")
    arrays, temp_paths = open_temp_arrays(output_root, total_cells, group_count)
    group_meta_temp = output_root / ".group_metadata.jsonl.tmp"
    group_meta_final = output_root / "group_metadata.jsonl"
    if group_meta_temp.exists():
        raise RuntimeError(f"stale group metadata temp exists: {group_meta_temp}")
    offset = 0
    group_id = 0
    label_counts = Counter()
    with group_meta_temp.open("w", encoding="utf-8") as meta_handle:
        arrays["group_offsets"][0] = 0
        for event_id, frame in n70.iter_cache_frames(events):
            event = events[event_id]
            candidate_features, memory, score, scalar, public_ids, rows = validate_frame(frame, event)
            memory_valid = np.asarray(frame["memory_valid"], dtype=bool).reshape(-1)
            n, p = candidate_features.shape[0], len(public_ids)
            cells = n * p
            end = offset + cells
            target_slot = public_ids.index(event["target_public_id"]) if event["target_public_id"] in public_ids else -1
            target_row = next((index for index, row in enumerate(rows) if int(row.get("native_tid", -1)) == event["target_native_id"]), -1)
            mapped_public = [row.get("public_id") for row in rows]
            labels_grid = np.zeros((n, p), dtype=np.uint8)
            for i, mapped in enumerate(mapped_public):
                if mapped is None:
                    continue
                for j, public_id in enumerate(public_ids):
                    labels_grid[i, j] = int(int(mapped) == int(public_id))
            none_grid = (~labels_grid.any(axis=1)).astype(np.uint8)
            assignment = np.asarray(frame["assignment_columns"], dtype=np.int64).reshape(n)
            assigned_grid = (assignment[:, None] == np.arange(p, dtype=np.int64)[None, :]).astype(np.float32)
            occupancy = np.bincount(assignment[assignment >= 0], minlength=p).astype(np.float32) / max(1, n)
            candidate_rank = np.arange(n, dtype=np.float32) / max(1, n - 1)
            candidate_count_norm = np.full(n, min(1.0, n / 32.0), dtype=np.float32)
            target_role = np.zeros(p, dtype=np.float32)
            if target_slot >= 0:
                target_role[target_slot] = 1.0
            memory_valid_f = memory_valid.astype(np.float32)
            # All arrays are row-major candidate x identity, matching the
            # frozen score matrix and its public_id_order.
            arrays["candidate"][offset:end] = np.repeat(candidate_features, p, axis=0).astype(np.float16)
            arrays["identity_memory"][offset:end] = np.tile(memory, (n, 1)).astype(np.float16)
            anchors = np.zeros((p, FEAT_DIM), dtype=np.float32)
            negatives = np.zeros((p, FEAT_DIM), dtype=np.float32)
            if target_slot >= 0:
                anchors[target_slot] = event["human_embedding"]
                negatives[target_slot] = event["negative_embedding"]
            arrays["human_anchor"][offset:end] = np.tile(anchors, (n, 1)).astype(np.float16)
            arrays["hard_negative"][offset:end] = np.tile(negatives, (n, 1)).astype(np.float16)
            scalar_grid = scalar.reshape(n, p, SCALAR_DIM)
            context = np.concatenate(
                [
                    scalar_grid,
                    score[:, :, None],
                    np.broadcast_to(target_role[None, :, None], (n, p, 1)),
                    memory_valid_f[None, :, None].repeat(n, axis=0),
                    assigned_grid[:, :, None],
                    np.broadcast_to(candidate_rank[:, None, None], (n, p, 1)),
                    np.broadcast_to(candidate_count_norm[:, None, None], (n, p, 1)),
                    np.broadcast_to(occupancy[None, :, None], (n, p, 1)),
                ],
                axis=2,
            )
            if context.shape != (n, p, CONTEXT_DIM) or not np.all(np.isfinite(context)):
                raise RuntimeError(f"global context construction failed: {event_id}/{frame['frame']} {context.shape}")
            arrays["context"][offset:end] = context.reshape(cells, CONTEXT_DIM).astype(np.float32)
            arrays["label"][offset:end] = labels_grid.reshape(cells)
            arrays["none_label"][offset:end] = np.repeat(none_grid, p)
            arrays["group"][offset:end] = group_id
            arrays["split"][offset:end] = SPLIT_CODE[split[event["sequence"]]]
            arrays["frame"][offset:end] = int(frame["frame"])
            arrays["variant"][offset:end] = n70.VARIANTS.index(str(frame["variant"]))
            arrays["candidate_slot"][offset:end] = np.repeat(np.arange(n, dtype=np.uint16), p)
            arrays["identity_slot"][offset:end] = np.tile(np.arange(p, dtype=np.uint16), n)
            arrays["target_slot"][offset:end] = target_slot
            arrays["target_row"][offset:end] = target_row
            arrays["target_present"][offset:end] = int(target_row >= 0)
            arrays["base_score"][offset:end] = score.reshape(cells)
            arrays["group_offsets"][group_id + 1] = end
            label_counts["positive_cells"] += int(labels_grid.sum())
            label_counts["none_candidates"] += int(none_grid.sum())
            meta_handle.write(json.dumps({
                "group": group_id,
                "event_id": event_id,
                "sequence": event["sequence"],
                "split": split[event["sequence"]],
                "action_type": event["action_type"],
                "variant": str(frame["variant"]),
                "frame": int(frame["frame"]),
                "event_frame": event["event_frame"],
                "candidate_count": n,
                "identity_count": p,
                "cell_offset_start": offset,
                "cell_offset_end": end,
                "target_slot_offline": target_slot,
                "target_row_offline": target_row,
                "target_candidate_present_offline": bool(target_row >= 0),
                "public_id_order_offline_only": public_ids,
                "runtime_future_gt_used": False,
                "interaction_source": "simulated_from_gt",
            }, sort_keys=True, allow_nan=False) + "\n")
            offset = end
            group_id += 1
            if group_id % 500 == 0:
                meta_handle.flush()
                print(json.dumps({"groups": group_id, "cells": offset, "total_cells": total_cells}, sort_keys=True), flush=True)
        meta_handle.flush()
        os.fsync(meta_handle.fileno())
    if offset != total_cells or group_id != group_count:
        raise RuntimeError(f"second pass count mismatch offset={offset}/{total_cells}, groups={group_id}/{group_count}")
    for array in arrays.values():
        array.flush()
    # Close memmaps before renaming their backing files.
    arrays.clear()
    for name, temp in temp_paths.items():
        final = output_root / ("group_offsets.npy" if name == "group_offsets" else f"{name}.npy")
        os.replace(temp, final)
    os.replace(group_meta_temp, group_meta_final)
    file_info = {}
    for path in sorted(output_root.glob("*.npy")) + [group_meta_final]:
        file_info[path.name] = {"sha256": sha256(path), "bytes": path.stat().st_size}
    manifest = {
        "schema": "N71_GLOBAL_MATRIX_DATASET_MANIFEST_V1",
        "status": "PASS_GLOBAL_MATRIX_MATERIALIZED",
        "created_at_utc": now(),
        "protocol": str(N71_PROTOCOL),
        "protocol_sha256": sha256(N71_PROTOCOL),
        "event_manifest": str(EVENT_MANIFEST),
        "event_manifest_sha256": sha256(EVENT_MANIFEST),
        "n70_cache_manifest": str(n70.CACHE_MANIFEST),
        "n70_cache_manifest_sha256": sha256(n70.CACHE_MANIFEST),
        "n70_cache_directory": str(n70.CACHE_DIR),
        "output_root": str(output_root),
        "group_count": group_count,
        "cell_count": total_cells,
        "feature_dim": FEAT_DIM,
        "scalar_input_dim": SCALAR_DIM,
        "context_dim": CONTEXT_DIM,
        "split_group_counts": dict(split_groups),
        "label_counts": dict(label_counts),
        "array_files": file_info,
        "cell_order": "row_major_candidate_then_identity_column; identity_slot is a non-public positional index",
        "identity_public_ids_are_offline_metadata_only": True,
        "numeric_public_id_feature": False,
        "numeric_target_native_id_feature": False,
        "runtime_future_gt_used": False,
        "interaction_source": "simulated_from_gt",
        "production_authorized": False,
        "candidate_generator": "N70 frozen rehydrated N36/N54 stream",
        "future_labels": "offline mapping labels only; no runtime replay outcome used",
    }
    atomic_json(manifest_path, manifest)
    return manifest


def record_failure(exc: BaseException, output_root: Path) -> Path:
    attempts = ROOT / "outputs/N71/attempts"
    attempts.mkdir(parents=True, exist_ok=True)
    existing = sorted(attempts.glob("n71_global_matrix_materialize_failure_attempt*.json"))
    path = attempts / f"n71_global_matrix_materialize_failure_attempt{len(existing) + 1}.json"
    atomic_json(path, {
        "schema": "N71_GLOBAL_MATRIX_MATERIALIZE_FAILURE_V1",
        "status": "FAIL_PRESERVED",
        "created_at_utc": now(),
        "failure_type": type(exc).__name__,
        "failure_message": str(exc),
        "traceback": traceback.format_exc(),
        "output_root": str(output_root),
        "protocol": str(N71_PROTOCOL),
        "protocol_sha256": sha256(N71_PROTOCOL),
        "event_manifest": str(EVENT_MANIFEST),
        "event_manifest_sha256": sha256(EVENT_MANIFEST),
        "runtime_future_gt_used": False,
        "interaction_source": "simulated_from_gt",
        "production_authorized": False,
    })
    return path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    args = parser.parse_args()
    try:
        result = write_matrix_dataset(args.output_root.resolve(), args.manifest.resolve())
        print(json.dumps(result, indent=2, sort_keys=True))
    except Exception as exc:
        failure = record_failure(exc, args.output_root.resolve())
        print(json.dumps({"status": "FAIL_PRESERVED", "failure_artifact": str(failure), "error": f"{type(exc).__name__}: {exc}"}, sort_keys=True))
        raise


if __name__ == "__main__":
    main()
