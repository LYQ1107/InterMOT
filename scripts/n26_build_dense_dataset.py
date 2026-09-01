#!/usr/bin/env python3
"""Build causal dense N26 Round-0 episodes from the frozen N25-R stream.

Every real shadow observation prefix is a model state, while every upstream
attempt remains one parent-normalized human event.  The current simulated
correction is written only after all states for that parent have been emitted.
The script never opens val25.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(".")
DATA_ROOT = Path("/path/to/dancetrack/train")
OUT = ROOT / "outputs/n26/dense_dataset"
GFN = ROOT / "outputs/n18/route_c/gfn_cache"
R0 = ROOT / "outputs/n20/gfn_cache_r0"
MAX_CANDIDATES = 5
MAX_MEMORY = 17
NONE_INDEX = 5
POS_CAP = 4
NEG_CAP = 8
HARD_CAP = 4
B10_LAMBDA = 0.8
B10_DELTA = 0.02

sys.path.insert(0, str(ROOT / "scripts"))
from n25r_r5_gate import load_split, normalize  # noqa: E402


SCALAR_NAMES = [
    "gfn_similarity", "r0_similarity", "b2_similarity", "clip_root_similarity",
    "sam3_f1_similarity", "detector_score", "rank_fraction", "candidate_count_fraction",
    "prefix_fraction", "current_valid", "valid_ratio", "valid_count_fraction", "lost_ratio",
    "box_cx", "box_cy", "box_w", "box_h", "box_area", "motion_speed_mean",
    "motion_acceleration_mean", "motion_dx_last", "motion_dy_last", "neighbor_crowd",
    "neighbor_distance", "neighbor_overlap", "neighbor_density", "temporal_clip_mean",
    "temporal_clip_std", "temporal_clip_first", "temporal_clip_last", "temporal_clip_max",
    "temporal_clip_slope", "query_age_log", "extra_attempt", "missing_gfn", "missing_r0",
    "missing_clip", "missing_sam3", "missing_motion", "missing_neighbor", "group_gfn_z",
    "group_r0_z", "group_clip_z", "group_sam3_z", "positive_memory_count_fraction",
    "negative_memory_count_fraction", "hard_negative_count_fraction", "positive_memory_age_log",
    "negative_memory_age_log", "hard_negative_age_log",
]

STATE_NAMES = [
    "VISIBLE_AND_CANDIDATE_PRESENT",
    "VISIBLE_BUT_CANDIDATE_MISSING",
    "TARGET_NOT_VISIBLE_OR_ABSENT",
    "UNKNOWN",
]
STATE_TO_INDEX = {name: index for index, name in enumerate(STATE_NAMES)}
MEMORY_KIND = {"ROOT_QUERY": 0, "HUMAN_EXPLICIT_POSITIVE": 1, "HUMAN_EXPLICIT_NEGATIVE": 2, "MODEL_INDUCED_HARD_NEGATIVE": 3}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def unit(x: np.ndarray) -> np.ndarray:
    value = np.asarray(x, dtype=np.float32)
    return value / (np.linalg.norm(value, axis=-1, keepdims=True) + 1e-8)


def finite(value: float | None, default: float = 0.0) -> float:
    return float(value) if value is not None and math.isfinite(float(value)) else default


def iou(a: np.ndarray, b: np.ndarray) -> float:
    ax1, ay1, ax2, ay2 = map(float, a)
    bx1, by1, bx2, by2 = map(float, b)
    iw = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    ih = max(0.0, min(ay2, by2) - max(ay1, by1))
    inter = iw * ih
    union = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1) + max(0.0, bx2 - bx1) * max(0.0, by2 - by1) - inter
    return inter / union if union > 0 else 0.0


class GroundTruth:
    def __init__(self, sequence: str):
        self.by_frame: dict[int, set[int]] = defaultdict(set)
        path = DATA_ROOT / sequence / "gt/gt.txt"
        for line in path.read_text(encoding="utf-8").splitlines():
            fields = line.split(",")
            if len(fields) >= 6 and float(fields[4]) > 0 and float(fields[5]) > 0:
                self.by_frame[int(float(fields[0])) - 1].add(int(float(fields[1])))
        self.length = 0
        self.width, self.height = 1280.0, 720.0
        for line in (DATA_ROOT / sequence / "seqinfo.ini").read_text(encoding="utf-8").splitlines():
            if line.startswith("seqLength="):
                self.length = int(line.split("=", 1)[1])
            elif line.startswith("imWidth="):
                self.width = float(line.split("=", 1)[1])
            elif line.startswith("imHeight="):
                self.height = float(line.split("=", 1)[1])

    def visible(self, frame: int, gid: int) -> bool | None:
        return gid in self.by_frame.get(frame, set()) if 0 <= frame < self.length else None


class DetectionStore:
    def __init__(self, sequence: str):
        with np.load(GFN / f"{sequence}.npz") as z, np.load(GFN / f"{sequence}_queries.npz") as q, np.load(R0 / f"{sequence}.npz") as r:
            self.frames = z["frames"].astype(np.int64)
            self.offsets = z["offsets"].astype(np.int64)
            self.boxes = z["boxes"].astype(np.float32)
            self.scores = z["scores"].astype(np.float32)
            self.gfn = unit(z["emb"].astype(np.float32))
            self.r0 = unit(r["r0g"].astype(np.float32))
            self.query = {
                int(gid): (unit(ge[None])[0], unit(re[None])[0])
                for gid, ge, re in zip(q["gids"], q["qemb"], r["r0q"])
            }

    def detection(self, frame: int, box: np.ndarray) -> tuple[np.ndarray, np.ndarray, float] | None:
        position = int(np.searchsorted(self.frames, frame))
        if position >= len(self.frames) or int(self.frames[position]) != frame:
            return None
        lo = int(self.offsets[position - 1]) if position else 0
        hi = int(self.offsets[position])
        if hi <= lo:
            return None
        overlaps = np.asarray([iou(candidate, box) for candidate in self.boxes[lo:hi]], dtype=np.float32)
        best = int(np.argmax(overlaps))
        if float(overlaps[best]) < 0.5:
            return None
        index = lo + best
        return self.gfn[index], self.r0[index], float(self.scores[index])


@dataclass
class MemoryToken:
    embedding: np.ndarray
    kind: int
    frame: int
    correction_id: int


def state_for_extra(sequence: str, frame: int, gid: int, gt: GroundTruth) -> str:
    visible = gt.visible(frame, gid)
    if visible is None:
        return "UNKNOWN"
    return "VISIBLE_BUT_CANDIDATE_MISSING" if visible else "TARGET_NOT_VISIBLE_OR_ABSENT"


def zscores(values: np.ndarray, mask: np.ndarray) -> np.ndarray:
    result = np.zeros_like(values, dtype=np.float32)
    valid = mask & np.isfinite(values)
    if valid.any():
        mean, std = float(values[valid].mean()), float(values[valid].std())
        result[valid] = (values[valid] - mean) / max(std, 1e-6)
    return result


def memory_snapshot(root: np.ndarray, memory: dict[str, list[MemoryToken]], frame: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int | None]:
    tokens = [MemoryToken(root, MEMORY_KIND["ROOT_QUERY"], frame, -1)]
    tokens.extend(memory["positive"][-POS_CAP:])
    tokens.extend(memory["negative"][-NEG_CAP:])
    tokens.extend(memory["hard"][-HARD_CAP:])
    tokens = tokens[:MAX_MEMORY]
    clip = np.zeros((MAX_MEMORY, 1280), dtype=np.float16)
    meta = np.zeros((MAX_MEMORY, 10), dtype=np.float16)
    kind = np.full(MAX_MEMORY, -1, dtype=np.int8)
    mask = np.zeros(MAX_MEMORY, dtype=bool)
    correction_ids = np.full(MAX_MEMORY, -2, dtype=np.int32)
    for index, token in enumerate(tokens):
        clip[index] = token.embedding.astype(np.float16)
        kind[index] = token.kind
        mask[index] = True
        correction_ids[index] = token.correction_id
        meta[index, token.kind] = 1.0
        age = max(0, frame - token.frame)
        meta[index, 4] = min(1.0, math.log1p(age) / 8.0)
        meta[index, 5] = min(1.0, age / 1000.0)
        if token.kind == MEMORY_KIND["HUMAN_EXPLICIT_POSITIVE"]:
            meta[index, 6] = 1.0
        elif token.kind == MEMORY_KIND["HUMAN_EXPLICIT_NEGATIVE"]:
            meta[index, 7] = 1.0
        elif token.kind == MEMORY_KIND["MODEL_INDUCED_HARD_NEGATIVE"]:
            meta[index, 8] = 1.0
        meta[index, 9] = 1.0
    human_ids = [token.correction_id for token in tokens if token.kind in (1, 2) and token.correction_id >= 0]
    latest = max(human_ids) if human_ids else None
    pre_mask = mask.copy()
    if latest is not None:
        pre_mask &= correction_ids != latest
        pre_mask[0] = True
    return clip, meta, kind, pre_mask, latest


def memory_counts(memory: dict[str, list[MemoryToken]], frame: int) -> list[float]:
    values: list[float] = []
    for name, cap in (("positive", POS_CAP), ("negative", NEG_CAP), ("hard", HARD_CAP)):
        values.append(len(memory[name]) / cap)
    for name in ("positive", "negative", "hard"):
        age = frame - memory[name][-1].frame if memory[name] else 10000
        values.append(min(1.0, math.log1p(max(0, age)) / 8.0))
    return values


def candidate_scalar(
    *, gfn: float, r0: float, clip_similarity: float, sam3: float, detector: float,
    rank: int, count: int, prefix: int, valid_steps: np.ndarray, box: np.ndarray | None,
    width: float, height: float, motion: np.ndarray, neighbor: np.ndarray,
    temporal: np.ndarray, query_age: int, extra: bool, memory_values: list[float],
) -> list[float]:
    valid_count = int(valid_steps.sum())
    current_valid = bool(len(valid_steps) and valid_steps[-1])
    b2 = (gfn + r0) / 2.0 if math.isfinite(gfn) and math.isfinite(r0) else math.nan
    if box is None:
        geometry = [0.0] * 5
    else:
        x1, y1, x2, y2 = map(float, box)
        ww, hh = max(0.0, x2 - x1) / width, max(0.0, y2 - y1) / height
        geometry = [(x1 + x2) / (2 * width), (y1 + y2) / (2 * height), ww, hh, ww * hh]
    if valid_count:
        speeds = np.linalg.norm(motion[valid_steps, 4:6], axis=1)
        accels = np.linalg.norm(motion[valid_steps, 6:8], axis=1)
        last_motion = motion[np.flatnonzero(valid_steps)[-1]]
        neighbor_mean = neighbor[valid_steps].mean(axis=0)
        motion_values = [float(speeds.mean()), float(accels.mean()), float(last_motion[4]), float(last_motion[5])]
        neighbor_values = [float(value) for value in neighbor_mean]
    else:
        motion_values, neighbor_values = [0.0] * 4, [0.0] * 4
    temporal = temporal[np.isfinite(temporal)]
    if len(temporal):
        temporal_values = [float(temporal.mean()), float(temporal.std()), float(temporal[0]), float(temporal[-1]), float(temporal.max()), float(temporal[-1] - temporal[0])]
    else:
        temporal_values = [0.0] * 6
    values = [
        finite(gfn), finite(r0), finite(b2), finite(clip_similarity), finite(sam3), finite(detector),
        rank / MAX_CANDIDATES, count / MAX_CANDIDATES, prefix / 10.0, float(current_valid),
        valid_count / max(1, prefix), valid_count / 10.0, 1.0 - valid_count / max(1, prefix),
        *geometry, *motion_values, *neighbor_values, *temporal_values,
        min(1.0, math.log1p(max(0, query_age)) / 8.0), float(extra),
        float(not math.isfinite(gfn)), float(not math.isfinite(r0)), float(not math.isfinite(clip_similarity)),
        float(not math.isfinite(sam3)), float(valid_count == 0), float(valid_count == 0),
        0.0, 0.0, 0.0, 0.0, *memory_values,
    ]
    if len(values) != len(SCALAR_NAMES):
        raise RuntimeError(f"scalar schema mismatch: {len(values)} != {len(SCALAR_NAMES)}")
    return values


def existing_observations(data: dict[str, Any], indices: list[int], store: DetectionStore) -> dict[str, Any]:
    rows, features = data["rows"], data["features"]
    first = rows[indices[0]]
    count = min(MAX_CANDIDATES, len(indices))
    max_prefix = max(len(rows[index].get("candidate_shadow_tracklet", [])) for index in indices[:count])
    max_prefix = max(1, min(9, max_prefix))
    query_clip = unit(features["clip_query"][indices[0]][None])[0]
    candidates = []
    for index in indices[:count]:
        row = rows[index]
        valid = features["valid"][index, :max_prefix].copy()
        clip_steps = features["clip_candidate"][index, :max_prefix].copy()
        f1_mean = features["f1_candidate_mean"][index, :max_prefix].copy()
        f1_max = features["f1_candidate_max"][index, :max_prefix].copy()
        gfn_steps = np.full((max_prefix, 2048), np.nan, dtype=np.float32)
        r0_steps = np.full((max_prefix, 2048), np.nan, dtype=np.float32)
        detector = np.full(max_prefix, np.nan, dtype=np.float32)
        boxes: list[np.ndarray | None] = [None] * max_prefix
        for step, item in enumerate(row.get("candidate_shadow_tracklet", [])[:max_prefix]):
            if item.get("box") is None or not bool(item.get("valid", True)):
                valid[step] = False
                continue
            box = np.asarray(item["box"], dtype=np.float32)
            boxes[step] = box
            match = store.detection(int(item["frame"]), box)
            if match is not None:
                gfn_steps[step], r0_steps[step], detector[step] = match
        candidates.append({
            "row": row, "row_index": index, "valid": valid, "clip": clip_steps,
            "f1_mean": f1_mean, "f1_max": f1_max, "gfn": gfn_steps, "r0": r0_steps,
            "detector": detector, "boxes": boxes, "motion": features["motion"][index, :max_prefix],
            "neighbor": features["neighbor"][index, :max_prefix],
        })
    return {
        "extra": False, "count": count, "max_prefix": max_prefix, "query_clip": query_clip,
        "query_frame": int(first["correction_frame"]), "query_box": first["legal_human_positive"]["box"],
        "candidates": candidates,
    }


def extra_observations(cache: dict[str, np.ndarray], index: int) -> dict[str, Any]:
    mask = cache["candidate_mask"][index].astype(bool)
    candidates = []
    for candidate_index in range(MAX_CANDIDATES):
        valid = np.asarray([bool(mask[candidate_index])])
        candidates.append({
            "row": {"candidate_rank": candidate_index + 1, "positive": False}, "row_index": -1,
            "valid": valid, "clip": cache["candidate_clip"][index, candidate_index][None].astype(np.float32),
            "f1_mean": np.full((1, 256), np.nan, dtype=np.float32), "f1_max": np.full((1, 256), np.nan, dtype=np.float32),
            "gfn_score": float(cache["gfn_similarity"][index, candidate_index]),
            "r0_score": float(cache["r0_similarity"][index, candidate_index]),
            "detector": np.asarray([cache["detection_score"][index, candidate_index]], dtype=np.float32),
            "boxes": [cache["candidate_box"][index, candidate_index] if mask[candidate_index] else None],
            "motion": np.zeros((1, 9), dtype=np.float32), "neighbor": np.zeros((1, 4), dtype=np.float32),
        })
    return {
        "extra": True, "count": int(mask.sum()), "max_prefix": 1,
        "query_clip": unit(cache["query_clip"][index].astype(np.float32)[None])[0],
        "query_frame": int(cache["query_frame"][index]), "query_box": cache["query_box"][index].tolist(),
        "candidates": candidates,
    }


def aggregate_candidate(candidate: dict[str, Any], query_clip: np.ndarray, query_f1_mean: np.ndarray | None, query_f1_max: np.ndarray | None, store: DetectionStore | None, gid: int, prefix: int) -> dict[str, Any]:
    valid = candidate["valid"][:prefix].astype(bool)
    available = bool(valid.any())
    clip = np.zeros(1280, dtype=np.float32)
    clip_similarity = math.nan
    temporal = np.asarray([], dtype=np.float32)
    if available:
        clip = unit(candidate["clip"][:prefix][valid].mean(axis=0, keepdims=True))[0]
        temporal = candidate["clip"][:prefix][valid] @ query_clip
        clip_similarity = float(clip @ query_clip)
    if "gfn_score" in candidate:
        gfn_score, r0_score = candidate["gfn_score"], candidate["r0_score"]
    else:
        gvalid = valid & np.isfinite(candidate["gfn"][:prefix]).all(axis=1)
        rvalid = valid & np.isfinite(candidate["r0"][:prefix]).all(axis=1)
        qg, qr = store.query[gid] if store is not None else (None, None)
        gfn_score = float(unit(candidate["gfn"][:prefix][gvalid].mean(axis=0, keepdims=True))[0] @ qg) if gvalid.any() else math.nan
        r0_score = float(unit(candidate["r0"][:prefix][rvalid].mean(axis=0, keepdims=True))[0] @ qr) if rvalid.any() else math.nan
    fvalid = valid & np.isfinite(candidate["f1_mean"][:prefix]).all(axis=1) & np.isfinite(candidate["f1_max"][:prefix]).all(axis=1)
    if fvalid.any() and query_f1_mean is not None and query_f1_max is not None:
        fm = unit(candidate["f1_mean"][:prefix][fvalid].mean(axis=0, keepdims=True))[0]
        fx = unit(candidate["f1_max"][:prefix][fvalid].mean(axis=0, keepdims=True))[0]
        sam3 = float((fm @ query_f1_mean + fx @ query_f1_max) / 2.0)
    else:
        sam3 = math.nan
    detector = float(np.nanmean(candidate["detector"][:prefix][valid])) if valid.any() and np.isfinite(candidate["detector"][:prefix][valid]).any() else math.nan
    latest_box = candidate["boxes"][int(np.flatnonzero(valid)[-1])] if valid.any() else None
    return {
        "mask": available, "clip": clip, "gfn": gfn_score, "r0": r0_score, "sam3": sam3,
        "detector": detector, "valid": valid, "box": latest_box, "temporal": temporal,
    }


def atomic_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    os.replace(temporary, path)


def build_split(split: str) -> dict[str, Any]:
    data = load_split(split)
    groups_by_key = {str(key): members for key, members in data["groups"].items()}
    trajectory = {json.loads(line)["event_key"]: json.loads(line) for line in (ROOT / f"outputs/n26/on_policy_trajectory_{split}.jsonl").open(encoding="utf-8") if line.strip()}
    attempts = list(csv.DictReader((ROOT / f"outputs/n20/dataset_attempts_{split}.csv").open(newline="", encoding="utf-8")))
    attempts.sort(key=lambda row: (row["sequence"], int(row["frame"]), int(row["gid"])))
    group_lookup: dict[tuple[str, int, int], list[int]] = {}
    for members in groups_by_key.values():
        row = data["rows"][members[0]]
        group_lookup[(str(row["sequence"]), int(row["decision_frame"]), int(row["gid"]))] = members

    extra_cache: dict[str, dict[str, np.ndarray]] = {}
    extra_index: dict[str, dict[tuple[int, int], int]] = {}
    for path in sorted((ROOT / f"outputs/n26/extra_clip/{split}").glob("*.npz")):
        cache = {name: value.copy() for name, value in np.load(path, allow_pickle=False).items()}
        extra_cache[path.stem] = cache
        extra_index[path.stem] = {(int(frame), int(gid)): index for index, (frame, gid) in enumerate(zip(cache["frame"], cache["gid"]))}

    sequence_names = sorted({row["sequence"] for row in attempts})
    sequence_to_index = {name: index for index, name in enumerate(sequence_names)}
    gt_cache = {sequence: GroundTruth(sequence) for sequence in sequence_names}
    memories: dict[tuple[str, int], dict[str, list[MemoryToken]]] = defaultdict(lambda: {"positive": [], "negative": [], "hard": []})

    arrays: dict[str, list[Any]] = defaultdict(list)
    parent_memory: dict[str, list[Any]] = defaultdict(list)
    metadata_rows: list[dict[str, Any]] = []
    correction_ledger: list[dict[str, Any]] = []
    current_sequence = None
    store: DetectionStore | None = None
    correction_id = 0
    parent_prefix_counts: list[int] = []

    for parent_index, attempt in enumerate(attempts):
        sequence, frame, gid = attempt["sequence"], int(attempt["frame"]), int(attempt["gid"])
        public_id = gid + 1000
        if sequence != current_sequence:
            current_sequence = sequence
            store = DetectionStore(sequence)
        members = group_lookup.get((sequence, frame, gid))
        if members is not None:
            observation = existing_observations(data, members, store)
            event = trajectory[f"{sequence}:{frame}:{gid}"]
            state = event["state_label"]
            first_index = members[0]
            query_f1_mean = unit(data["features"]["f1_query_mean"][first_index][None])[0]
            query_f1_max = unit(data["features"]["f1_query_max"][first_index][None])[0]
        else:
            cache = extra_cache[sequence]
            index = extra_index[sequence][(frame, gid)]
            observation = extra_observations(cache, index)
            state = state_for_extra(sequence, frame, gid, gt_cache[sequence])
            query_f1_mean = query_f1_max = None
        root = observation["query_clip"]
        scope = (sequence, public_id)
        memory = memories[scope]
        memory_clip, memory_meta, memory_kind, memory_pre_mask, latest_correction = memory_snapshot(root, memory, frame)
        memory_mask = memory_kind >= 0
        parent_memory["memory_clip"].append(memory_clip)
        parent_memory["memory_meta"].append(memory_meta)
        parent_memory["memory_kind"].append(memory_kind)
        parent_memory["memory_mask"].append(memory_mask)
        parent_memory["memory_pre_mask"].append(memory_pre_mask)
        memory_values = memory_counts(memory, frame)

        canonical: dict[str, Any] | None = None
        parent_state_indices: list[int] = []
        max_prefix = int(observation["max_prefix"])
        for prefix in range(1, max_prefix + 1):
            candidate_clip = np.zeros((MAX_CANDIDATES, 1280), dtype=np.float16)
            candidate_scalar_values = np.zeros((MAX_CANDIDATES, len(SCALAR_NAMES)), dtype=np.float16)
            candidate_mask = np.zeros(MAX_CANDIDATES, dtype=bool)
            candidate_label = np.zeros(MAX_CANDIDATES, dtype=bool)
            blocks = []
            for candidate_index, candidate in enumerate(observation["candidates"][:MAX_CANDIDATES]):
                block = aggregate_candidate(candidate, root, query_f1_mean, query_f1_max, store, gid, prefix)
                blocks.append(block)
                candidate_mask[candidate_index] = block["mask"] and candidate_index < observation["count"]
                candidate_label[candidate_index] = candidate_mask[candidate_index] and bool(candidate["row"].get("positive", False))
                candidate_clip[candidate_index] = block["clip"].astype(np.float16)
                candidate_scalar_values[candidate_index] = np.asarray(candidate_scalar(
                    gfn=block["gfn"], r0=block["r0"], clip_similarity=block["clip"] @ root if block["mask"] else math.nan,
                    sam3=block["sam3"], detector=block["detector"], rank=candidate_index + 1,
                    count=int(observation["count"]), prefix=prefix, valid_steps=block["valid"], box=block["box"],
                    width=gt_cache[sequence].width, height=gt_cache[sequence].height,
                    motion=candidate["motion"][:prefix], neighbor=candidate["neighbor"][:prefix], temporal=block["temporal"],
                    query_age=frame - int(observation["query_frame"]), extra=bool(observation["extra"]), memory_values=memory_values,
                ), dtype=np.float16)
            for feature_index, key in ((40, "gfn"), (41, "r0"), (42, "clip"), (43, "sam3")):
                vals = np.asarray([finite(block[key], math.nan) if key != "clip" else (float(block["clip"] @ root) if block["mask"] else math.nan) for block in blocks] + [math.nan] * (MAX_CANDIDATES - len(blocks)), dtype=np.float32)
                candidate_scalar_values[:, feature_index] = zscores(vals, candidate_mask).astype(np.float16)
            positives = np.flatnonzero(candidate_label)
            target = int(positives[0]) if len(positives) else NONE_INDEX
            existence = float(target != NONE_INDEX)
            existence_mask = state != "UNKNOWN"
            rejected_index = -1
            pair_valid = False
            if latest_correction is not None:
                latest_tokens = [token for token in memory["negative"] if token.correction_id == latest_correction]
                valid_indices = np.flatnonzero(candidate_mask)
                if latest_tokens and len(valid_indices):
                    rejected_index = int(max(valid_indices, key=lambda index: float(candidate_clip[index].astype(np.float32) @ latest_tokens[-1].embedding)))
                    pair_valid = True
            state_index = len(arrays["target"])
            parent_state_indices.append(state_index)
            arrays["candidate_clip"].append(candidate_clip)
            arrays["candidate_scalar"].append(candidate_scalar_values)
            arrays["candidate_mask"].append(candidate_mask)
            arrays["candidate_label"].append(candidate_label)
            arrays["target"].append(target)
            arrays["existence"].append(existence)
            arrays["existence_mask"].append(existence_mask)
            arrays["sequence"].append(sequence_to_index[sequence])
            arrays["frame"].append(frame)
            arrays["gid"].append(gid)
            arrays["parent"].append(parent_index)
            arrays["prefix"].append(prefix)
            arrays["primary_h5"].append(prefix == min(5, max_prefix))
            arrays["pair_valid"].append(pair_valid)
            arrays["rejected_index"].append(rejected_index)
            arrays["state_index"].append(STATE_TO_INDEX[state])
            if prefix == min(5, max_prefix):
                canonical = {"candidate_clip": candidate_clip.astype(np.float32), "candidate_mask": candidate_mask, "target": target, "state_index": state_index}
        parent_prefix_counts.append(len(parent_state_indices))
        for state_index in parent_state_indices:
            arrays["sample_weight"].append(1.0 / max(1, len(parent_state_indices)))
        if canonical is None:
            raise RuntimeError(f"no canonical state {sequence}:{frame}:{gid}")

        valid_indices = np.flatnonzero(canonical["candidate_mask"])
        selected = -1
        b10_scores = np.full(MAX_CANDIDATES, -np.inf, dtype=np.float32)
        if len(valid_indices):
            for candidate_index in valid_indices:
                embedding = canonical["candidate_clip"][candidate_index]
                positive = [float(root @ embedding)] + [float(token.embedding @ embedding) for token in memory["positive"]]
                positive_similarity = max(positive)
                negative = [float(token.embedding @ embedding) for token in memory["negative"]]
                penalty = max(0.0, max(negative) - positive_similarity + B10_DELTA) if negative else 0.0
                b10_scores[candidate_index] = positive_similarity - B10_LAMBDA * penalty
            selected = int(np.argmax(b10_scores))
        wrong = selected != int(canonical["target"])
        correction_written = False
        if wrong:
            correction_id += 1
            if selected >= 0:
                token = MemoryToken(canonical["candidate_clip"][selected].copy(), MEMORY_KIND["HUMAN_EXPLICIT_NEGATIVE"], frame, correction_id)
                memory["negative"].append(token)
                memory["negative"] = memory["negative"][-NEG_CAP:]
                correction_ledger.append({"split": split, "parent_event_id": parent_index, "event_key": f"{sequence}:{frame}:{gid}", "sequence": sequence, "frame": frame, "public_identity_id": public_id, "candidate_rank": selected + 1, "memory_kind": "HUMAN_EXPLICIT_NEGATIVE", "source": "ROUND0_FROZEN_B10_SELECTED_THEN_SIMULATED_HUMAN_REJECTED", "applies_from_next_parent_only": True})
                correction_written = True
            if canonical["target"] != NONE_INDEX:
                positive_index = int(canonical["target"])
                token = MemoryToken(canonical["candidate_clip"][positive_index].copy(), MEMORY_KIND["HUMAN_EXPLICIT_POSITIVE"], frame, correction_id)
                memory["positive"].append(token)
                memory["positive"] = memory["positive"][-POS_CAP:]
                correction_ledger.append({"split": split, "parent_event_id": parent_index, "event_key": f"{sequence}:{frame}:{gid}", "sequence": sequence, "frame": frame, "public_identity_id": public_id, "candidate_rank": positive_index + 1, "memory_kind": "HUMAN_EXPLICIT_POSITIVE", "source": "SIMULATED_HUMAN_CORRECTED_TARGET_AFTER_ROUND0_ERROR", "applies_from_next_parent_only": True})
        wrong_candidates = [index for index in valid_indices if index != canonical["target"] and index != selected]
        if wrong_candidates:
            hard_index = max(wrong_candidates, key=lambda index: float(b10_scores[index]))
            memory["hard"].append(MemoryToken(canonical["candidate_clip"][hard_index].copy(), MEMORY_KIND["MODEL_INDUCED_HARD_NEGATIVE"], frame, -1))
            memory["hard"] = memory["hard"][-HARD_CAP:]
        metadata_rows.append({
            "split": split, "parent_event_id": parent_index, "event_key": f"{sequence}:{frame}:{gid}",
            "sequence": sequence, "frame": frame, "gid": gid, "public_identity_id": public_id,
            "state_label": state, "extra_target_present_zero_attempt": bool(observation["extra"]),
            "real_temporal_states": len(parent_state_indices), "canonical_state_index": int(canonical["state_index"]),
            "canonical_target": int(canonical["target"]), "round0_selected": selected,
            "round0_selected_correct": not wrong, "correction_event": wrong,
            "explicit_negative_written": correction_written, "current_feedback_used_by_current_states": False,
            "policy_version": "N26_ROUND0_FROZEN_B10_H5_LAMBDA0.8_V1",
        })

    output_arrays = {
        "candidate_clip": np.asarray(arrays["candidate_clip"], dtype=np.float16),
        "candidate_scalar": np.asarray(arrays["candidate_scalar"], dtype=np.float16),
        "candidate_mask": np.asarray(arrays["candidate_mask"], dtype=bool),
        "candidate_label": np.asarray(arrays["candidate_label"], dtype=bool),
        "target": np.asarray(arrays["target"], dtype=np.int64),
        "existence": np.asarray(arrays["existence"], dtype=np.float32),
        "existence_mask": np.asarray(arrays["existence_mask"], dtype=bool),
        "sequence": np.asarray(arrays["sequence"], dtype=np.int16),
        "frame": np.asarray(arrays["frame"], dtype=np.int32),
        "gid": np.asarray(arrays["gid"], dtype=np.int32),
        "parent": np.asarray(arrays["parent"], dtype=np.int32),
        "prefix": np.asarray(arrays["prefix"], dtype=np.int8),
        "primary_h5": np.asarray(arrays["primary_h5"], dtype=bool),
        "pair_valid": np.asarray(arrays["pair_valid"], dtype=bool),
        "rejected_index": np.asarray(arrays["rejected_index"], dtype=np.int8),
        "state_index": np.asarray(arrays["state_index"], dtype=np.int8),
        "sample_weight": np.asarray(arrays["sample_weight"], dtype=np.float32),
        "memory_clip": np.asarray(parent_memory["memory_clip"], dtype=np.float16),
        "memory_meta": np.asarray(parent_memory["memory_meta"], dtype=np.float16),
        "memory_kind": np.asarray(parent_memory["memory_kind"], dtype=np.int8),
        "memory_mask": np.asarray(parent_memory["memory_mask"], dtype=bool),
        "memory_pre_mask": np.asarray(parent_memory["memory_pre_mask"], dtype=bool),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"round0_{split}.npz"
    atomic_npz(path, output_arrays)
    metadata_path = OUT / f"round0_{split}_parents.jsonl"
    with metadata_path.open("w", encoding="utf-8") as handle:
        for row in metadata_rows:
            handle.write(json.dumps(row, separators=(",", ":")) + "\n")
    ledger_path = OUT / f"round0_{split}_memory_ledger.jsonl"
    with ledger_path.open("w", encoding="utf-8") as handle:
        for row in correction_ledger:
            handle.write(json.dumps(row, separators=(",", ":")) + "\n")
    summary = {
        "split": split, "policy": "frozen B10 H5 lambda=0.8", "parents": len(attempts),
        "states": len(output_arrays["target"]), "sequences": len(sequence_names), "sequence_names": sequence_names,
        "materialized_n25r_parents": sum(not row["extra_target_present_zero_attempt"] for row in metadata_rows),
        "additional_target_present_zero_parents": sum(row["extra_target_present_zero_attempt"] for row in metadata_rows),
        "state_counts": dict(Counter(row["state_label"] for row in metadata_rows)),
        "target_none_states": int((output_arrays["target"] == NONE_INDEX).sum()),
        "correction_events": sum(row["correction_event"] for row in metadata_rows),
        "explicit_negative_writes": sum(row["explicit_negative_written"] for row in metadata_rows),
        "positive_writes": sum(row["memory_kind"] == "HUMAN_EXPLICIT_POSITIVE" for row in correction_ledger),
        "pair_states": int(output_arrays["pair_valid"].sum()),
        "minimum_prefixes_per_parent": min(parent_prefix_counts), "maximum_prefixes_per_parent": max(parent_prefix_counts),
        "sample_weight_sum": float(output_arrays["sample_weight"].sum()),
        "current_feedback_used_by_current_state": False, "duplicate_parent_counted_as_independent_human": False,
        "candidate_protocol": "frozen N25-R static GFN top-5; no union", "val25_read": False,
        "npz": str(path.relative_to(ROOT)), "npz_sha256": sha256(path),
        "parents_jsonl": str(metadata_path.relative_to(ROOT)), "parents_sha256": sha256(metadata_path),
        "memory_ledger": str(ledger_path.relative_to(ROOT)), "memory_ledger_sha256": sha256(ledger_path),
    }
    (OUT / f"round0_{split}_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=("train30", "cal10", "all"), default="all")
    args = parser.parse_args()
    summaries = []
    for split in (("train30", "cal10") if args.split == "all" else (args.split,)):
        print(f"BUILD {split}", flush=True)
        summaries.append(build_split(split))
        print(json.dumps(summaries[-1], sort_keys=True), flush=True)
    if args.split == "all":
        manifest = {
            "name": "N26 dense Round-0 causal episodes", "policy_version": "N26_ROUND0_FROZEN_B10_H5_LAMBDA0.8_V1",
            "candidate_protocol": "N25-R repaired frozen GFN top-5", "real_temporal_prefixes": True,
            "parent_normalized": True, "same_event_prefixes_are_not_independent_human_corrections": True,
            "memory_kinds": MEMORY_KIND, "scalar_features": SCALAR_NAMES, "state_names": STATE_NAMES,
            "splits": {item["split"]: item for item in summaries}, "val25_read": False,
        }
        (OUT / "round0_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print("N26_DENSE_ROUND0_DONE", flush=True)


if __name__ == "__main__":
    main()
