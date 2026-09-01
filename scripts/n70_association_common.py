"""Shared, isolated N70 association dataset and model contracts.

The module consumes only the N70 rehydrated cache.  It deliberately keeps the
offline target-native ID out of model inputs: that ID is used to make labels
and to score replay after runtime inference.  The event's public ID is the
authoritative intervention column and is already present in the frozen event
protocol; no numeric public-ID feature is constructed.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import sys
import tempfile
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

OUT = ROOT / "outputs/N70"
PROTOCOL = OUT / "protocol.json"
N37_EVENTS = ROOT / "outputs/n37/real_event_manifest.json"
CACHE_MANIFEST = OUT / "cache/candidate_cache_manifest.json"
MAPPING_SUMMARY = OUT / "diagnosis/mapping_summary.json"
CACHE_DIR = Path("/path/to/cache/SAM3_InterMOT_N70/cache/N70_REHYDRATED_N36_MAPPING_ENRICHED_N54_STREAM/event_frames")
TRAIN_ROOT = Path("/path/to/cache/SAM3_InterMOT_N70/training")
DATASET = TRAIN_ROOT / "n70_association_dataset.npz"
DATASET_MANIFEST = OUT / "training/n70_association_dataset_manifest.json"
SMOKE_A = OUT / "training/n70_branch_a_smoke.json"
SMOKE_B = OUT / "training/n70_branch_b_smoke.json"
CHECKPOINT_A = TRAIN_ROOT / "N70_BRANCH_A.pt"
CHECKPOINT_B = TRAIN_ROOT / "N70_BRANCH_B.pt"
TRAIN_MANIFEST_A = OUT / "training/n70_branch_a_training_manifest.json"
TRAIN_MANIFEST_B = OUT / "training/n70_branch_b_training_manifest.json"
VARIANTS = ("M0", "M1", "M2", "M3", "M4")
FEAT_DIM = 512
CONTEXT_DIM = 34
PROJECTION_DIM = 64
SPLIT_CODE = {"train": 0, "validation": 1, "holdout": 2}
SPLIT_NAMES = {value: key for key, value in SPLIT_CODE.items()}
SEED = 7070


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_name, path)
        dfd = os.open(str(path.parent), os.O_DIRECTORY)
        try:
            os.fsync(dfd)
        finally:
            os.close(dfd)
    finally:
        if os.path.exists(temp_name):
            os.unlink(temp_name)


def atomic_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".npz", dir=str(path.parent))
    os.close(fd)
    try:
        np.savez_compressed(temp_name, **arrays)
        with open(temp_name, "rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temp_name, path)
        dfd = os.open(str(path.parent), os.O_DIRECTORY)
        try:
            os.fsync(dfd)
        finally:
            os.close(dfd)
    finally:
        if os.path.exists(temp_name):
            os.unlink(temp_name)


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return value


def load_protocol() -> dict[str, Any]:
    protocol = load_json(PROTOCOL)
    if protocol.get("status") != "FROZEN_BEFORE_N70_EXECUTION":
        raise RuntimeError("N70 protocol is not frozen")
    if protocol.get("runtime_contract", {}).get("runtime_future_gt_used") is not False:
        raise RuntimeError("N70 protocol runtime GT boundary is not false")
    return protocol


def load_event_map() -> dict[str, dict[str, Any]]:
    payload = load_json(N37_EVENTS)
    events: dict[str, dict[str, Any]] = {}
    for item in payload.get("events", []):
        event = item.get("event", {})
        event_id = str(item.get("protocol_candidate_id") or event.get("event_id"))
        if not isinstance(event, dict) or not event_id or event_id == "None":
            raise RuntimeError("N37 event is not addressable")
        if event.get("interaction_source") != "simulated_from_gt" or item.get("interaction_source") != "simulated_from_gt":
            raise RuntimeError(f"N70 training requires explicit simulated provenance: {event_id}")
        if event.get("runtime_future_gt_used") is not False or item.get("runtime_future_gt_used") is not False:
            raise RuntimeError(f"N70 event runtime GT boundary failed: {event_id}")
        anchor = np.asarray(event.get("human_embedding"), dtype=np.float32).reshape(-1)
        if anchor.size != FEAT_DIM or not np.all(np.isfinite(anchor)) or float(np.linalg.norm(anchor)) <= 1e-6:
            raise RuntimeError(f"N70 human anchor invalid: {event_id}")
        events[event_id] = {
            "event_id": event_id,
            "sequence": str(event["sequence"]),
            "event_frame": int(event["frame"]),
            "future_start": int(item["future_frame_start"]),
            "future_end": int(item["future_frame_end"]),
            "target_public_id": int(event["public_id"]),
            "target_native_id": int(event["target_native_tid"]),
            "human_embedding": anchor.tolist(),
            "action_type": str(item.get("action_type") or event.get("action_type")),
            "interaction_source": "simulated_from_gt",
        }
    if len(events) != 24:
        raise RuntimeError(f"expected 24 N37 events, found {len(events)}")
    return events


def sequence_split() -> dict[str, str]:
    protocol = load_protocol()
    result: dict[str, str] = {}
    for split_name, sequences in protocol["sequence_split"].items():
        for sequence in sequences:
            sequence = str(sequence)
            if sequence in result:
                raise RuntimeError(f"sequence appears in multiple splits: {sequence}")
            result[sequence] = str(split_name)
    return result


def adapt_cache_frame(frame: dict[str, Any]) -> dict[str, Any]:
    """Adapt the N70 cache record to the frozen N54/N69 feature API.

    This is a schema adapter only.  It copies the frozen score/memory arrays;
    it does not add target labels or alter candidate ordering.
    """
    required = ("event_id", "variant", "frame", "event_frame", "candidate_rows", "rows", "public_id_order", "score_matrix", "candidate_features_512", "memory_vectors_512", "memory_valid", "scalar_features_8", "assignment_columns")
    missing = [key for key in required if key not in frame]
    if missing:
        raise RuntimeError(f"N70 cache frame missing keys {missing}: {frame.get('event_id')}/{frame.get('variant')}/{frame.get('frame')}")
    branch = {
        "candidate_rows": frame["candidate_rows"],
        "rows": frame["rows"],
        "public_id_order": frame["public_id_order"],
        "score_matrix": frame["score_matrix"],
        "assignment_columns": frame["assignment_columns"],
        "assignment_public_ids": frame.get("assignment_public_ids"),
        "runtime_future_gt_used": frame.get("runtime_future_gt_used"),
    }
    return {
        "event_id": frame["event_id"],
        "variant": frame["variant"],
        "frame": int(frame["frame"]),
        "event_frame": int(frame["event_frame"]),
        "write_baseline": branch,
        "candidate_features_512": frame["candidate_features_512"],
        "memory_vectors_512": frame["memory_vectors_512"],
        "memory_valid": frame["memory_valid"],
        "scalar_features_8": frame["scalar_features_8"],
        "runtime_future_gt_used": frame.get("runtime_future_gt_used"),
        "runtime_gt_read": frame.get("runtime_gt_read"),
        "_n70_frame": frame,
    }


def iter_cache_frames(events: dict[str, dict[str, Any]]) -> Iterator[tuple[str, dict[str, Any]]]:
    if not CACHE_DIR.is_dir():
        raise FileNotFoundError(CACHE_DIR)
    seen: set[tuple[str, str, int]] = set()
    file_count = 0
    for path in sorted(CACHE_DIR.glob("*.jsonl")):
        file_count += 1
        with path.open("r", encoding="utf-8") as handle:
            for line_no, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    frame = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise RuntimeError(f"invalid N70 cache JSON {path}:{line_no}: {exc}") from exc
                if not isinstance(frame, dict):
                    raise RuntimeError(f"N70 cache frame is not object {path}:{line_no}")
                event_id = str(frame.get("event_id"))
                variant = str(frame.get("variant"))
                key = (event_id, variant, int(frame.get("frame", -1)))
                if event_id not in events or variant not in VARIANTS:
                    raise RuntimeError(f"N70 cache key not in frozen protocol: {key}")
                if key in seen:
                    raise RuntimeError(f"duplicate N70 cache key: {key}")
                seen.add(key)
                if frame.get("runtime_future_gt_used") is not False or frame.get("runtime_gt_read") is not False:
                    raise RuntimeError(f"N70 cache runtime GT boundary failed: {key}")
                if frame.get("candidate_set_complete") is not True:
                    raise RuntimeError(f"N70 cache candidate set incomplete: {key}")
                yield event_id, frame
    if file_count != 24:
        raise RuntimeError(f"N70 cache expected 24 event files, found {file_count}")
    if len(seen) != 12000:
        raise RuntimeError(f"N70 cache expected 12000 unique frames, found {len(seen)}")


def build_feature_pack(frame: dict[str, Any], event: dict[str, Any], *, include_offline_label: bool) -> dict[str, Any]:
    """Use the frozen N69 causal feature construction on the N70 adapter."""
    from scripts import n69_stage03_target_conditioned as n69

    adapted = adapt_cache_frame(frame)
    result = n69.build_feature_arrays(adapted, event, include_offline_label=include_offline_label)
    if result["runtime_future_gt_used"] is not False:
        raise RuntimeError("N70 feature pack runtime GT boundary failed")
    if result["candidate"].shape[1:] != (FEAT_DIM,) or result["context"].shape[1:] != (CONTEXT_DIM,):
        raise RuntimeError(f"N70 feature shape mismatch: {result['candidate'].shape}, {result['context'].shape}")
    return result


def materialize_dataset() -> dict[str, Any]:
    protocol = load_protocol()
    events = load_event_map()
    split = sequence_split()
    candidates: list[np.ndarray] = []
    anchors: list[np.ndarray] = []
    memories: list[np.ndarray] = []
    hard_negatives: list[np.ndarray] = []
    contexts: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    groups: list[np.ndarray] = []
    group_split: list[int] = []
    group_has_positive: list[int] = []
    frames: list[np.ndarray] = []
    variants: list[np.ndarray] = []
    native_ids: list[np.ndarray] = []
    sequences: list[np.ndarray] = []
    event_ids: list[np.ndarray] = []
    actions: list[np.ndarray] = []
    target_columns: list[np.ndarray] = []
    target_rows: list[np.ndarray] = []
    target_candidate_present: list[np.ndarray] = []
    target_public_assignment_absent: list[np.ndarray] = []
    mapping_complete: list[np.ndarray] = []
    temporal_left: list[int] = []
    temporal_right: list[int] = []
    previous_positive: dict[tuple[str, str], tuple[int, int] | None] = {}
    group_id = 0
    example_offset = 0
    counts: defaultdict[str, int] = defaultdict(int)
    for event_id, frame in iter_cache_frames(events):
        event = events[event_id]
        pack = build_feature_pack(frame, event, include_offline_label=True)
        n = int(pack["candidate"].shape[0])
        if n <= 0:
            raise RuntimeError(f"empty N70 candidate frame: {event_id}/{frame['variant']}/{frame['frame']}")
        y = np.asarray(pack["labels"], dtype=np.int8)
        if y.shape != (n,) or not np.all(np.isin(y, [0, 1])):
            raise RuntimeError(f"N70 offline labels malformed: {event_id}/{frame['variant']}/{frame['frame']}")
        split_name = split[event["sequence"]]
        candidate_rows = frame["candidate_rows"]
        native = np.asarray([int(item["native_tid"]) for item in candidate_rows], dtype=np.int64)
        if native.size != n:
            raise RuntimeError("N70 native/candidate count mismatch")
        if np.any(native == int(event["target_native_id"])) != bool(pack["target_present_offline"]):
            raise RuntimeError("N70 label/target presence mismatch")
        candidates.append(pack["candidate"].astype(np.float32))
        anchors.append(pack["anchor"].astype(np.float32))
        memories.append(pack["memory"].astype(np.float32))
        hard_negatives.append(pack["hard_negative"].astype(np.float32))
        contexts.append(pack["context"].astype(np.float32))
        labels.append(y)
        groups.append(np.full(n, group_id, dtype=np.int64))
        group_split.append(SPLIT_CODE[split_name])
        group_has_positive.append(int(np.any(y == 1)))
        frames.append(np.full(n, int(frame["frame"]), dtype=np.int32))
        variants.append(np.full(n, VARIANTS.index(str(frame["variant"])), dtype=np.int8))
        sequences.append(np.full(n, event["sequence"], dtype="U64"))
        event_ids.append(np.full(n, event_id, dtype="U128"))
        actions.append(np.full(n, event["action_type"], dtype="U64"))
        native_ids.append(native)
        target_columns.append(np.full(n, -1 if pack["target_column"] is None else int(pack["target_column"]), dtype=np.int32))
        target_rows.append(np.full(n, -1 if pack["target_row_offline"] is None else int(pack["target_row_offline"]), dtype=np.int32))
        target_candidate_present.append(np.full(n, bool(any(int(row.get("native_tid", -1)) == int(event["target_native_id"]) for row in candidate_rows)), dtype=np.int8))
        target_public_assignment_absent.append(np.full(n, bool(frame.get("candidate_rows") and pack["target_column"] is None and pack["target_present_offline"]), dtype=np.int8))
        mapping_complete.append(np.full(n, all(
            isinstance(row.get("mapping"), dict)
            and row["mapping"].get("local_id") is not None
            and row["mapping"].get("global_id") is not None
            and (row["mapping"].get("public_id") is not None or row["mapping"].get("public_id_status") == "EXPLICIT_N54_PUBLIC_ASSIGNMENT_ABSENT")
            for row in candidate_rows if isinstance(row, dict)
        ), dtype=np.int8))
        if pack["target_present_offline"]:
            positive_row = int(np.where(y == 1)[0][0])
            previous_key = (event_id, str(frame["variant"]))
            previous = previous_positive.get(previous_key)
            if previous is not None and int(frame["frame"]) == previous[0] + 1:
                temporal_left.append(previous[1])
                temporal_right.append(example_offset + positive_row)
            previous_positive[previous_key] = (int(frame["frame"]), example_offset + positive_row)
        else:
            previous_positive[(event_id, str(frame["variant"]))] = None
        counts[f"{split_name}_examples"] += n
        counts[f"{split_name}_positive"] += int(np.sum(y))
        counts[f"{event['action_type']}_frames"] += 1
        group_id += 1
        example_offset += n
        if group_id % 500 == 0:
            print(json.dumps({"groups": group_id, "examples": example_offset}, sort_keys=True), flush=True)
    if group_id != 12000:
        raise RuntimeError(f"N70 dataset expected 12000 frame groups, found {group_id}")
    arrays = {
        "candidate": np.concatenate(candidates),
        "anchor": np.concatenate(anchors),
        "memory": np.concatenate(memories),
        "hard_negative": np.concatenate(hard_negatives),
        "context": np.concatenate(contexts),
        "label": np.concatenate(labels),
        "group": np.concatenate(groups),
        "group_split": np.asarray(group_split, dtype=np.int8),
        "group_has_positive": np.asarray(group_has_positive, dtype=np.int8),
        "frame": np.concatenate(frames),
        "variant": np.concatenate(variants),
        "native_id": np.concatenate(native_ids),
        "sequence": np.concatenate(sequences),
        "event_id": np.concatenate(event_ids),
        "action": np.concatenate(actions),
        "target_column": np.concatenate(target_columns),
        "target_row_offline": np.concatenate(target_rows),
        "target_candidate_present": np.concatenate(target_candidate_present),
        "target_public_assignment_absent": np.concatenate(target_public_assignment_absent),
        "mapping_complete": np.concatenate(mapping_complete),
        "temporal_left": np.asarray(temporal_left, dtype=np.int64),
        "temporal_right": np.asarray(temporal_right, dtype=np.int64),
    }
    numeric = [value for value in arrays.values() if np.issubdtype(value.dtype, np.number)]
    if not all(np.all(np.isfinite(value)) for value in numeric):
        raise RuntimeError("N70 dataset has nonfinite numeric values")
    atomic_npz(DATASET, arrays)
    manifest = {
        "schema": "N70_ASSOCIATION_DATASET_MANIFEST_V1",
        "status": "PASS_DATASET_MATERIALIZED",
        "created_at_utc": now(),
        "protocol": str(PROTOCOL),
        "protocol_sha256": sha256_file(PROTOCOL),
        "cache_manifest": str(CACHE_MANIFEST),
        "cache_manifest_sha256": sha256_file(CACHE_MANIFEST),
        "mapping_summary": str(MAPPING_SUMMARY),
        "mapping_summary_sha256": sha256_file(MAPPING_SUMMARY),
        "dataset": str(DATASET),
        "dataset_sha256": sha256_file(DATASET),
        "shape": {key: list(value.shape) for key, value in arrays.items()},
        "groups": group_id,
        "examples": int(arrays["label"].size),
        "positive_examples": int(np.sum(arrays["label"] == 1)),
        "negative_examples": int(np.sum(arrays["label"] == 0)),
        "temporal_pairs": len(temporal_left),
        "counts": dict(sorted(counts.items())),
        "feature_contract": {
            "candidate_raw_dim": FEAT_DIM,
            "human_anchor_raw_dim": FEAT_DIM,
            "target_memory_raw_dim": FEAT_DIM,
            "historical_positive_raw_dim": FEAT_DIM,
            "hard_negative_raw_dim": FEAT_DIM,
            "context_dim": CONTEXT_DIM,
            "context_source": "N54 frozen scalar features plus causal geometry/memory summaries",
            "target_native_id_used_only_for_offline_label": True,
            "numeric_public_id_feature": False,
            "runtime_future_gt_used": False,
        },
        "split": protocol["sequence_split"],
        "interaction_source": "simulated_from_gt",
        "real_human_tape": False,
        "real_sam3_full_loop": False,
        "not_real_human_evidence": True,
        "production_authorized": False,
    }
    atomic_json(DATASET_MANIFEST, manifest)
    return manifest


def load_dataset() -> dict[str, np.ndarray]:
    if not DATASET.is_file() or not DATASET_MANIFEST.is_file():
        raise RuntimeError("N70 dataset missing; run --mode materialize")
    payload = np.load(DATASET, allow_pickle=False)
    arrays = {key: payload[key] for key in payload.files}
    required = {"candidate", "anchor", "memory", "hard_negative", "context", "label", "group", "group_split", "group_has_positive", "temporal_left", "temporal_right"}
    missing = sorted(required - set(arrays))
    if missing:
        raise RuntimeError(f"N70 dataset missing arrays: {missing}")
    if arrays["candidate"].shape[1:] != (FEAT_DIM,) or any(arrays[key].shape != arrays["candidate"].shape for key in ("anchor", "memory", "hard_negative")):
        raise RuntimeError("N70 raw feature shape mismatch")
    if arrays["context"].shape[1:] != (CONTEXT_DIM,) or arrays["context"].shape[0] != arrays["label"].shape[0]:
        raise RuntimeError("N70 context/label shape mismatch")
    if not all(np.all(np.isfinite(arrays[key])) for key in ("candidate", "anchor", "memory", "hard_negative", "context")):
        raise RuntimeError("N70 model features are nonfinite")
    if not np.array_equal(np.unique(arrays["group"]), np.arange(arrays["group_split"].size, dtype=arrays["group"].dtype)):
        raise RuntimeError("N70 group IDs are not contiguous")
    return arrays


def group_index(arrays: dict[str, np.ndarray]) -> dict[int, np.ndarray]:
    result: dict[int, np.ndarray] = {}
    for gid in range(int(arrays["group_split"].size)):
        indices = np.flatnonzero(arrays["group"] == gid)
        if indices.size == 0:
            raise RuntimeError(f"empty N70 group {gid}")
        result[gid] = indices
    return result


def group_batches(group_ids: list[int], groups: dict[int, np.ndarray], max_examples: int, rng: np.random.Generator | None) -> Iterable[np.ndarray]:
    ordered = list(group_ids)
    if rng is not None:
        rng.shuffle(ordered)
    current: list[int] = []
    size = 0
    for gid in ordered:
        next_size = int(groups[gid].size)
        if current and size + next_size > max_examples:
            yield np.concatenate([groups[value] for value in current])
            current, size = [], 0
        current.append(gid)
        size += next_size
    if current:
        yield np.concatenate([groups[value] for value in current])


def context_normalization(arrays: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    train_groups = np.flatnonzero(arrays["group_split"] == SPLIT_CODE["train"])
    mask = np.isin(arrays["group"], train_groups)
    if not np.any(mask):
        raise RuntimeError("N70 training split is empty")
    mean = arrays["context"][mask].mean(axis=0, dtype=np.float64).astype(np.float32)
    std = arrays["context"][mask].std(axis=0, dtype=np.float64).astype(np.float32)
    std[std < 1e-6] = 1.0
    return mean, std


def set_all_seeds(seed: int = SEED) -> None:
    import random
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def torch_device(name: str):
    import torch
    if name.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("N70 requested CUDA but CUDA is unavailable")
    return torch.device(name)


def build_model(branch: str):
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    if branch == "A":
        class BranchA(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.shared_projection = nn.Linear(FEAT_DIM, PROJECTION_DIM, bias=False)
                self.norm = nn.LayerNorm(PROJECTION_DIM * 10 + CONTEXT_DIM)
                self.scorer = nn.Sequential(
                    nn.Linear(PROJECTION_DIM * 10 + CONTEXT_DIM, 128),
                    nn.ReLU(),
                    nn.Linear(128, 64),
                    nn.ReLU(),
                    nn.Linear(64, 2),
                )

            def forward(self, candidate, anchor, memory, hard_negative, context):
                candidate = F.normalize(candidate, dim=-1, eps=1e-6)
                anchor = F.normalize(anchor, dim=-1, eps=1e-6)
                memory = F.normalize(memory, dim=-1, eps=1e-6)
                hard_negative = F.normalize(hard_negative, dim=-1, eps=1e-6)
                c = self.shared_projection(candidate)
                a = self.shared_projection(anchor)
                m = self.shared_projection(memory)
                h = self.shared_projection(hard_negative)
                features = torch.cat([c, a, m, h, c * a, torch.abs(c - a), c * m, torch.abs(c - m), c * h, torch.abs(c - h), context], dim=-1)
                return self.scorer(self.norm(features))

        return BranchA()

    if branch == "B":
        class BranchB(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.query = nn.Linear(FEAT_DIM, PROJECTION_DIM, bias=False)
                self.prototype = nn.Linear(FEAT_DIM, PROJECTION_DIM, bias=False)
                self.context = nn.Sequential(nn.Linear(CONTEXT_DIM, 32), nn.ReLU())
                self.norm = nn.LayerNorm(PROJECTION_DIM * 4 + 32)
                self.scorer = nn.Sequential(
                    nn.Linear(PROJECTION_DIM * 4 + 32, 96),
                    nn.ReLU(),
                    nn.Linear(96, 48),
                    nn.ReLU(),
                    nn.Linear(48, 2),
                )

            def forward(self, candidate, anchor, memory, hard_negative, context):
                candidate = F.normalize(candidate, dim=-1, eps=1e-6)
                anchor = F.normalize(anchor, dim=-1, eps=1e-6)
                memory = F.normalize(memory, dim=-1, eps=1e-6)
                hard_negative = F.normalize(hard_negative, dim=-1, eps=1e-6)
                query = self.query(candidate)
                prototypes = torch.stack([self.prototype(anchor), self.prototype(memory), self.prototype(hard_negative)], dim=1)
                weights = torch.softmax(torch.sum(query.unsqueeze(1) * prototypes, dim=-1) / math.sqrt(PROJECTION_DIM), dim=1)
                pooled = torch.sum(weights.unsqueeze(-1) * prototypes, dim=1)
                set_mean = torch.mean(prototypes, dim=1)
                compact = torch.cat([query, pooled, query * pooled, torch.abs(query - pooled), self.context(context)], dim=-1)
                return self.scorer(self.norm(compact))

        return BranchB()
    raise ValueError(f"unknown N70 branch {branch}")


def model_metadata(branch: str, model: Any) -> dict[str, Any]:
    return {
        "branch": branch,
        "name": "N70_BRANCH_A_LOW_RANK_BILINEAR_NONE_LISTWISE" if branch == "A" else "N70_BRANCH_B_HISTORY_SET_SCORER",
        "parameter_count": int(sum(parameter.numel() for parameter in model.parameters())),
        "raw_embedding_dim": FEAT_DIM,
        "context_dim": CONTEXT_DIM,
        "projection_dim": PROJECTION_DIM,
        "outputs": ["candidate_score", "none_score"],
        "none_explicit": True,
        "numeric_public_id_feature": False,
        "target_native_id_feature": False,
        "runtime_future_gt_used": False,
    }


def tensors_for_indices(arrays: dict[str, np.ndarray], indices: np.ndarray, mean: np.ndarray, std: np.ndarray, device: Any):
    import torch
    index = np.asarray(indices, dtype=np.int64)
    return (
        torch.as_tensor(arrays["candidate"][index], dtype=torch.float32, device=device),
        torch.as_tensor(arrays["anchor"][index], dtype=torch.float32, device=device),
        torch.as_tensor(arrays["memory"][index], dtype=torch.float32, device=device),
        torch.as_tensor(arrays["hard_negative"][index], dtype=torch.float32, device=device),
        torch.as_tensor((arrays["context"][index] - mean) / std, dtype=torch.float32, device=device),
        torch.as_tensor(arrays["label"][index], dtype=torch.long, device=device),
        torch.as_tensor(arrays["group"][index], dtype=torch.long, device=device),
    )


def batch_loss(model: Any, arrays: dict[str, np.ndarray], indices: np.ndarray, mean: np.ndarray, std: np.ndarray, device: Any):
    import torch
    import torch.nn.functional as F
    candidate, anchor, memory, hard_negative, context, labels, group_ids = tensors_for_indices(arrays, indices, mean, std, device)
    logits = model(candidate, anchor, memory, hard_negative, context)
    model_labels = 1 - labels
    ce = F.cross_entropy(logits, model_labels, weight=torch.tensor([2.0, 1.0], device=device))
    target_logits = logits[:, 0]
    target_prob = torch.softmax(logits, dim=-1)[:, 0]
    ranking: list[Any] = []
    none: list[Any] = []
    noop: list[Any] = []
    for gid in torch.unique(group_ids).tolist():
        mask = group_ids == int(gid)
        local_labels = labels[mask]
        local_logits = target_logits[mask]
        local_probs = target_prob[mask]
        has_positive = bool(torch.any(local_labels == 1).detach().cpu())
        if has_positive and torch.any(local_labels == 0):
            ranking.append(F.softplus(torch.max(local_logits[local_labels == 0]) - torch.max(local_logits[local_labels == 1]) + 0.2))
        none.append(F.binary_cross_entropy(torch.max(local_probs), torch.tensor(float(has_positive), device=device)))
        if not has_positive:
            noop.append(F.relu(torch.max(local_probs) - 0.5).square())
    ranking_loss = torch.stack(ranking).mean() if ranking else torch.zeros((), device=device)
    none_loss = torch.stack(none).mean() if none else torch.zeros((), device=device)
    noop_loss = torch.stack(noop).mean() if noop else torch.zeros((), device=device)
    total = ce + 0.5 * ranking_loss + 0.5 * none_loss + 0.05 * noop_loss
    return total, {
        "total": float(total.detach().cpu()),
        "cross_entropy": float(ce.detach().cpu()),
        "ranking": float(ranking_loss.detach().cpu()),
        "none": float(none_loss.detach().cpu()),
        "no_op": float(noop_loss.detach().cpu()),
    }


def evaluate_model(model: Any, arrays: dict[str, np.ndarray], split_name: str, groups: dict[int, np.ndarray], mean: np.ndarray, std: np.ndarray, device: Any) -> dict[str, Any]:
    import torch
    split_value = SPLIT_CODE[split_name]
    group_ids = [gid for gid, value in enumerate(arrays["group_split"].tolist()) if int(value) == split_value]
    probs: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    group_probs: list[float] = []
    group_targets: list[float] = []
    ranking: list[float] = []
    model.eval()
    with torch.no_grad():
        for batch in group_batches(group_ids, groups, 4096, None):
            tensors = tensors_for_indices(arrays, batch, mean, std, device)
            p = torch.softmax(model(*tensors[:5]), dim=-1)[:, 0].cpu().numpy()
            probs.append(p)
            labels.append(arrays["label"][batch].astype(np.int64))
            for gid in np.unique(arrays["group"][batch]):
                local = arrays["group"][batch] == gid
                y = arrays["label"][batch][local]
                q = p[local]
                group_probs.append(float(np.max(q)))
                group_targets.append(float(np.any(y == 1)))
                if np.any(y == 1) and np.any(y == 0):
                    ranking.append(max(0.0, 0.2 - float(np.max(q[y == 1])) + float(np.max(q[y == 0]))))
    y = np.concatenate(labels) if labels else np.zeros(0, dtype=np.int64)
    p = np.concatenate(probs) if probs else np.zeros(0, dtype=np.float32)
    eps = 1e-7
    bce = float(np.mean(-(y * np.log(np.clip(p, eps, 1 - eps)) + (1 - y) * np.log(np.clip(1 - p, eps, 1 - eps))))) if y.size else None
    try:
        from sklearn.metrics import roc_auc_score
        auc = float(roc_auc_score(y, p)) if np.unique(y).size == 2 else None
    except Exception:
        auc = None
    gbce = float(np.mean(-(np.asarray(group_targets) * np.log(np.clip(group_probs, eps, 1 - eps)) + (1 - np.asarray(group_targets)) * np.log(np.clip(1 - np.asarray(group_probs), eps, 1 - eps))))) if group_probs else None
    composite = None if bce is None else bce + 0.5 * (float(np.mean(ranking)) if ranking else 0.0) + 0.5 * (gbce or 0.0)
    return {
        "split": split_name,
        "examples": int(y.size),
        "positive_examples": int(np.sum(y == 1)),
        "groups": len(group_ids),
        "finite_predictions": bool(np.all(np.isfinite(p))),
        "bce": bce,
        "auc": auc,
        "accuracy_at_0_5": float(np.mean((p >= 0.5) == (y == 1))) if y.size else None,
        "group_none_bce": gbce,
        "group_none_accuracy": float(np.mean((np.asarray(group_probs) >= 0.5) == (np.asarray(group_targets) >= 0.5))) if group_probs else None,
        "ranking_proxy": float(np.mean(ranking)) if ranking else None,
        "composite": composite,
        "probability_range": [float(np.min(p)), float(np.max(p))] if p.size else [None, None],
    }
