#!/usr/bin/env python3
"""Audit the N71 global-matrix memmap dataset without changing it."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from collections import Counter
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest_path = args.manifest.resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    root = Path(manifest["output_root"])
    errors: list[str] = []
    names = [
        "candidate", "identity_memory", "human_anchor", "hard_negative", "context",
        "label", "none_label", "group", "split", "frame", "variant", "candidate_slot",
        "identity_slot", "target_slot", "target_row", "target_present", "base_score",
    ]
    arrays: dict[str, np.ndarray] = {}
    for name in names + ["group_offsets"]:
        path = root / f"{name}.npy"
        if not path.is_file():
            errors.append(f"missing_array={name}")
            continue
        arrays[name] = np.load(path, mmap_mode="r", allow_pickle=False)
    cells = int(manifest.get("cell_count", -1))
    groups = int(manifest.get("group_count", -1))
    for name in names:
        if name not in arrays:
            continue
        if arrays[name].shape[0] != cells:
            errors.append(f"length_mismatch={name}:{arrays[name].shape[0]}!={cells}")
        if name in {"candidate", "identity_memory", "human_anchor", "hard_negative"} and arrays[name].shape[1:] != (512,):
            errors.append(f"feature_shape={name}:{arrays[name].shape}")
        if name == "context" and arrays[name].shape[1:] != (15,):
            errors.append(f"context_shape={arrays[name].shape}")
        if name in {"candidate", "identity_memory", "human_anchor", "hard_negative", "context", "base_score"}:
            if not np.all(np.isfinite(arrays[name])):
                errors.append(f"nonfinite={name}")
    offsets = arrays.get("group_offsets")
    if offsets is None or offsets.shape != (groups + 1,):
        errors.append(f"group_offsets_shape={None if offsets is None else offsets.shape}")
    elif int(offsets[0]) != 0 or int(offsets[-1]) != cells or np.any(np.diff(offsets) <= 0):
        errors.append("group_offsets_not_strict_or_terminal")

    metadata_path = root / "group_metadata.jsonl"
    metadata: list[dict] = []
    if not metadata_path.is_file():
        errors.append("missing_group_metadata")
    else:
        with metadata_path.open("r", encoding="utf-8") as handle:
            for line_no, line in enumerate(handle, 1):
                try:
                    row = json.loads(line)
                    metadata.append(row)
                except json.JSONDecodeError as exc:
                    errors.append(f"metadata_json_line_{line_no}:{exc}")
    if len(metadata) != groups:
        errors.append(f"metadata_count={len(metadata)}!={groups}")
    split_counts: Counter[str] = Counter()
    action_counts: Counter[str] = Counter()
    variant_counts: Counter[str] = Counter()
    none_candidates = 0
    positive_cells = 0
    max_labels_per_candidate = 0
    runtime_gt_true = 0
    for gid, row in enumerate(metadata):
        if int(row.get("group", -1)) != gid:
            errors.append(f"metadata_group_mismatch={gid}")
            continue
        start, end = int(row.get("cell_offset_start", -1)), int(row.get("cell_offset_end", -1))
        if offsets is not None and (start != int(offsets[gid]) or end != int(offsets[gid + 1])):
            errors.append(f"metadata_offset_mismatch={gid}")
        n, p = int(row.get("candidate_count", 0)), int(row.get("identity_count", 0))
        if end - start != n * p or n <= 0 or p <= 0:
            errors.append(f"metadata_cell_count={gid}")
        split_counts[str(row.get("split"))] += 1
        action_counts[str(row.get("action_type"))] += 1
        variant_counts[str(row.get("variant"))] += 1
        if row.get("runtime_future_gt_used") is not False:
            runtime_gt_true += 1
        if all(name in arrays for name in ("label", "none_label", "candidate_slot", "identity_slot")) and end > start:
            labels = np.asarray(arrays["label"][start:end]).reshape(n, p)
            none = np.asarray(arrays["none_label"][start:end]).reshape(n, p)
            if np.any(labels.sum(axis=1) > 1):
                errors.append(f"multiple_positive_identity_cells={gid}")
            expected_none = (labels.sum(axis=1) == 0)[:, None]
            if not np.array_equal(none.astype(bool), np.broadcast_to(expected_none, (n, p))):
                errors.append(f"none_label_mismatch={gid}")
            if np.asarray(arrays["candidate_slot"][start:end]).reshape(n, p).tolist() != np.repeat(np.arange(n)[:, None], p, axis=1).tolist():
                errors.append(f"candidate_slot_order={gid}")
            if np.asarray(arrays["identity_slot"][start:end]).reshape(n, p).tolist() != np.repeat(np.arange(p)[None, :], n, axis=0).reshape(n, p).tolist():
                errors.append(f"identity_slot_order={gid}")
            positive_cells += int(labels.sum())
            none_candidates += int((labels.sum(axis=1) == 0).sum())
            max_labels_per_candidate = max(max_labels_per_candidate, int(labels.sum(axis=1).max()))
    if runtime_gt_true:
        errors.append(f"runtime_future_gt_used_count={runtime_gt_true}")
    result = {
        "schema": "N71_GLOBAL_MATRIX_DATASET_AUDIT_V1",
        "status": "PASS" if not errors else "FAIL",
        "manifest": str(manifest_path),
        "manifest_sha256": sha256(manifest_path),
        "output_root": str(root),
        "cell_count": cells,
        "group_count": groups,
        "positive_cells": positive_cells,
        "none_candidates": none_candidates,
        "max_labels_per_candidate": max_labels_per_candidate,
        "split_group_counts": dict(split_counts),
        "action_group_counts": dict(action_counts),
        "variant_group_counts": dict(variant_counts),
        "runtime_future_gt_used_count": runtime_gt_true,
        "numeric_public_id_feature": False,
        "numeric_target_native_id_feature": False,
        "errors": errors,
    }
    atomic_json(args.output.resolve(), result)
    print(json.dumps(result, indent=2, sort_keys=True))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
