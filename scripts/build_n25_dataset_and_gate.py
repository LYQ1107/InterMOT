#!/usr/bin/env python3
"""Build the N25 raw episode audit and run the pre-training information gate.

The input is the raw N20 all-candidate shadow stream, not the N24 compact NPZ.
This script preserves the causal boxes and validity masks, recomputes small raw
RGB crop descriptors, motion and neighbour summaries, and evaluates transparent
GFN/R0/raw baselines on the same candidate groups.  SAM3 internal tensors are
never replaced by zero vectors: their status is recorded as unavailable for
candidate-aligned batch data, so B5/B10/B11 remain explicitly uncomputable.

GT-derived ``correct`` fields are copied into a separate post-hoc label block;
they are never used by feature construction or candidate scoring.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import OrderedDict, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = Path("/path/to/dancetrack")
GFN_ROOT = ROOT / "outputs/n18/route_c/gfn_cache"
R0_ROOT = ROOT / "outputs/n20/gfn_cache_r0"
N20_ROOT = ROOT / "outputs/n20"
MAX_H = 20
RAW_SIZE = 8


def normalize(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-8)


def cosine(a: np.ndarray | None, b: np.ndarray | None) -> float | None:
    if a is None or b is None:
        return None
    a = np.asarray(a, dtype=np.float32).reshape(-1)
    b = np.asarray(b, dtype=np.float32).reshape(-1)
    if a.size == 0 or b.size == 0 or a.size != b.size:
        return None
    return float(np.dot(normalize(a[None])[0], normalize(b[None])[0]))


def iou(a: np.ndarray, b: np.ndarray) -> float:
    ax1, ay1, ax2, ay2 = [float(v) for v in a]
    bx1, by1, bx2, by2 = [float(v) for v in b]
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    aa = max(1e-8, (ax2 - ax1) * (ay2 - ay1))
    bb = max(1e-8, (bx2 - bx1) * (by2 - by1))
    return inter / max(1e-8, aa + bb - inter)


def frame_path(seq: str, frame: int) -> Path:
    # The project uses zero-based video indices and one-based JPEG names.
    return DATA_ROOT / "train" / seq / "img1" / f"{int(frame) + 1:08d}.jpg"


def crop_descriptor(image_or_path: Path | np.ndarray | None, box: list[float] | np.ndarray) -> np.ndarray | None:
    """Compute the fixed raw crop descriptor, reusing a decoded frame when possible."""
    if image_or_path is None or box is None:
        return None
    try:
        if isinstance(image_or_path, Path):
            if not image_or_path.is_file():
                return None
            with Image.open(image_or_path) as im:
                image = np.asarray(im.convert("RGB"), dtype=np.uint8).copy()
        else:
            image = np.asarray(image_or_path, dtype=np.uint8)
        h, w = image.shape[:2]
        x1, y1, x2, y2 = [float(v) for v in box]
        x1 = int(max(0, min(w - 1, math.floor(x1))))
        y1 = int(max(0, min(h - 1, math.floor(y1))))
        x2 = int(max(x1 + 1, min(w, math.ceil(x2))))
        y2 = int(max(y1 + 1, min(h, math.ceil(y2))))
        if x2 <= x1 or y2 <= y1:
            return None
        crop = Image.fromarray(image, mode="RGB").crop((x1, y1, x2, y2)).resize(
            (RAW_SIZE, RAW_SIZE), Image.Resampling.BILINEAR
        )
        arr = np.asarray(crop, dtype=np.float32) / 255.0
    except Exception:
        return None
    # Raw resized RGB plus first/second-order colour statistics.  This is a
    # fixed diagnostic feature, not a learned identity model.
    vec = np.concatenate(
        [arr.reshape(-1), arr.mean(axis=(0, 1)), arr.std(axis=(0, 1))]
    ).astype(np.float32)
    return normalize(vec[None])[0]


class SequenceStore:
    def __init__(self, seq: str):
        self.seq = seq
        gz = np.load(GFN_ROOT / f"{seq}.npz")
        qz = np.load(GFN_ROOT / f"{seq}_queries.npz")
        rz = np.load(R0_ROOT / f"{seq}.npz")
        self.frames = gz["frames"].astype(np.int64)
        self.offsets = gz["offsets"].astype(np.int64)
        self.boxes = gz["boxes"].astype(np.float32)
        self.gfn = normalize(gz["emb"].astype(np.float32))
        self.r0 = normalize(rz["r0g"].astype(np.float32))
        self.query = {
            int(g): {
                "frame": int(f),
                "box": np.asarray(b, dtype=np.float32),
                "gfn": normalize(e[None])[0],
                "r0": normalize(r[None])[0],
            }
            for g, f, b, e, r in zip(
                qz["gids"], qz["qframe"], qz["qbox"], qz["qemb"], rz["r0q"]
            )
        }
        self.width = 1280.0
        self.height = 720.0
        first = next(iter((DATA_ROOT / "train" / seq / "img1").glob("*.jpg")), None)
        if first is not None:
            with Image.open(first) as im:
                self.width, self.height = map(float, im.size)
        gz.close()
        qz.close()
        rz.close()
        # Rows are frame-sorted within a sequence.  Decoding each JPEG once per
        # frame removes the dominant I/O cost while keeping the feature path
        # exactly the same.  Both caches are bounded so a split cannot retain
        # all decoded images or crop descriptors in memory.
        self.raw_cache: OrderedDict[tuple[int, tuple[float, ...]], np.ndarray | None] = OrderedDict()
        self.image_cache: OrderedDict[int, np.ndarray | None] = OrderedDict()
        self.image_cache_limit = 32
        self.raw_cache_limit = 4096

    def image(self, frame: int) -> np.ndarray | None:
        frame = int(frame)
        if frame in self.image_cache:
            image = self.image_cache.pop(frame)
            self.image_cache[frame] = image
            return image
        path = frame_path(self.seq, frame)
        image = None
        if path.is_file():
            try:
                with Image.open(path) as im:
                    image = np.asarray(im.convert("RGB"), dtype=np.uint8).copy()
            except Exception:
                image = None
        self.image_cache[frame] = image
        while len(self.image_cache) > self.image_cache_limit:
            self.image_cache.popitem(last=False)
        return image

    def detection(self, frame: int, box: list[float] | np.ndarray):
        pos = int(np.searchsorted(self.frames, int(frame)))
        if pos >= len(self.frames) or int(self.frames[pos]) != int(frame):
            return None
        lo = int(self.offsets[pos - 1]) if pos else 0
        hi = int(self.offsets[pos])
        if hi <= lo:
            return None
        vals = np.asarray([iou(x, box) for x in self.boxes[lo:hi]], dtype=np.float32)
        best = int(np.argmax(vals))
        if float(vals[best]) < 0.5:
            return None
        idx = lo + best
        return self.gfn[idx], self.r0[idx], float(vals[best])

    def raw(self, frame: int, box):
        key = (int(frame), tuple(round(float(v), 2) for v in box))
        if key in self.raw_cache:
            value = self.raw_cache.pop(key)
            self.raw_cache[key] = value
            return value
        value = crop_descriptor(self.image(frame), box)
        self.raw_cache[key] = value
        while len(self.raw_cache) > self.raw_cache_limit:
            self.raw_cache.popitem(last=False)
        return value

    def neighbours(self, frame: int, box):
        pos = int(np.searchsorted(self.frames, int(frame)))
        if pos >= len(self.frames) or int(self.frames[pos]) != int(frame):
            return np.zeros(4, dtype=np.float32)
        lo = int(self.offsets[pos - 1]) if pos else 0
        hi = int(self.offsets[pos])
        dets = self.boxes[lo:hi]
        if len(dets) == 0:
            return np.zeros(4, dtype=np.float32)
        vals = np.asarray([iou(x, box) for x in dets], dtype=np.float32)
        own = int(np.argmax(vals))
        other = np.delete(dets, own, axis=0) if len(dets) > 1 else np.empty((0, 4))
        if len(other) == 0:
            return np.asarray([0.0, 1.0, 0.0, 0.0], dtype=np.float32)
        cx = (float(box[0]) + float(box[2])) / 2.0
        cy = (float(box[1]) + float(box[3])) / 2.0
        ocx = (other[:, 0] + other[:, 2]) / 2.0
        ocy = (other[:, 1] + other[:, 3]) / 2.0
        dist = np.sqrt((ocx - cx) ** 2 + (ocy - cy) ** 2) / math.hypot(
            self.width, self.height
        )
        other_iou = np.asarray([iou(x, box) for x in other], dtype=np.float32)
        return np.asarray(
            [
                min(1.0, float(np.sum((other_iou > 0.05) | (dist < 0.12))) / 8.0),
                min(1.0, float(dist.min()) / 0.8),
                float(other_iou.max(initial=0.0)),
                min(1.0, float(len(dets)) / 32.0),
            ],
            dtype=np.float32,
        )


def load_rows(path: Path) -> list[dict[str, Any]]:
    rows = []
    for fp in sorted(path.glob("*.jsonl")):
        with fp.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    rows.append(json.loads(line))
    rows.sort(key=lambda r: (r["sequence"], int(r["frame"]), int(r["gid"]), int(r["candidate_rank"])))
    return rows


def load_episode_rows(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def event_public_ids(split: str) -> dict[tuple[str, int], int]:
    if split == "train30":
        paths = sorted((ROOT / "outputs/n21/train30_true_onpolicy").glob("events_*.jsonl"))
    else:
        paths = sorted((ROOT / "outputs/n21/live_final_gate/L0").glob("events_*.jsonl"))
    out: dict[tuple[str, int], int] = {}
    for path in paths:
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                if "public_id" in row:
                    out.setdefault((str(row["sequence"]), int(row["gid"])), int(row["public_id"]))
    return out


def motion_series(store: SequenceStore, frames: list[dict[str, Any]]) -> np.ndarray:
    out = np.zeros((MAX_H, 9), dtype=np.float32)
    centers: list[np.ndarray | None] = []
    for i, fr in enumerate(frames[:MAX_H]):
        box = fr.get("box")
        if box is None:
            centers.append(None)
            continue
        b = np.asarray(box, dtype=np.float32)
        cx = (b[0] + b[2]) / 2.0 / store.width
        cy = (b[1] + b[3]) / 2.0 / store.height
        ww = max(0.0, b[2] - b[0]) / store.width
        hh = max(0.0, b[3] - b[1]) / store.height
        centers.append(np.asarray([cx, cy], dtype=np.float32))
        out[i, :4] = [cx, cy, ww, hh]
        out[i, 8] = 1.0
    for i in range(1, min(MAX_H, len(centers))):
        if centers[i] is not None and centers[i - 1] is not None:
            out[i, 4:6] = centers[i] - centers[i - 1]
    for i in range(2, min(MAX_H, len(centers))):
        if out[i, 8] and out[i - 1, 8] and out[i - 2, 8]:
            out[i, 6:8] = out[i, 4:6] - out[i - 1, 4:6]
    return out


def safe_mean(vectors: list[np.ndarray]) -> np.ndarray | None:
    if not vectors:
        return None
    return normalize(np.mean(np.asarray(vectors, dtype=np.float32), axis=0, keepdims=True))[0]


def candidate_scores(store: SequenceStore, row: dict[str, Any], raw, motion, neigh):
    q = store.query.get(int(row["gid"]))
    if q is None:
        return {m: {h: None for h in (1, 5, 10, 20)} for m in (
            "B0_GFN", "B1_R0", "B2_GFN_R0", "B3_RAW_STATIC", "B4_RAW_TEMPORAL",
            "B6_MOTION", "B7_RAW_MOTION", "B8_RAW_NEIGHBOR", "B9_RAW_MOTION_NEIGHBOR")}
    results = defaultdict(dict)
    frames = row.get("frames", [])
    gfn_vecs, r0_vecs, raw_vecs = [], [], []
    motion_vals, neigh_vals = [], []
    for i, fr in enumerate(frames[:MAX_H]):
        box = fr.get("box")
        if box is None:
            continue
        de = store.detection(int(fr["frame"]), box)
        if de is not None:
            gfn_vecs.append(de[0])
            r0_vecs.append(de[1])
        if raw[i] is not None:
            raw_vecs.append(raw[i])
        motion_vals.append(float(np.linalg.norm(motion[i, 4:6]) + 0.5 * np.linalg.norm(motion[i, 6:8])))
        neigh_vals.append(float(neigh[i, 1] + neigh[i, 2] + neigh[i, 0]))
        for h in (1, 5, 10, 20):
            if int(fr["frame"]) >= int(row["frame"]) + h:
                continue
            # Values are recomputed below from the prefix, so this loop only
            # preserves the causal ordering for the raw arrays.
    for h in (1, 5, 10, 20):
        prefix = [i for i, fr in enumerate(frames[:MAX_H]) if int(fr["frame"]) < int(row["frame"]) + h and fr.get("box") is not None]
        gvec, rvec, rawvec = [], [], []
        mvals, nvals = [], []
        for i in prefix:
            de = store.detection(int(frames[i]["frame"]), frames[i]["box"])
            if de is not None:
                gvec.append(de[0])
                rvec.append(de[1])
            if raw[i] is not None:
                rawvec.append(raw[i])
            mvals.append(float(np.linalg.norm(motion[i, 4:6]) + 0.5 * np.linalg.norm(motion[i, 6:8])))
            nvals.append(float(neigh[i, 1] + neigh[i, 2] + neigh[i, 0]))
        gmean, rmean, rawmean = safe_mean(gvec), safe_mean(rvec), safe_mean(rawvec)
        gscore = cosine(q["gfn"], gmean)
        rscore = cosine(q["r0"], rmean)
        results["B0_GFN"][h] = gscore
        results["B1_R0"][h] = rscore
        if gmean is not None and rmean is not None:
            results["B2_GFN_R0"][h] = cosine(
                np.concatenate([q["gfn"], q["r0"]]), np.concatenate([gmean, rmean])
            )
        else:
            results["B2_GFN_R0"][h] = None
        results["B3_RAW_STATIC"][h] = cosine(q.get("raw"), raw[0] if raw and raw[0] is not None else None)
        results["B4_RAW_TEMPORAL"][h] = cosine(q.get("raw"), rawmean)
        # Lower motion jitter and lower neighbour conflict are higher scores.
        results["B6_MOTION"][h] = -float(np.mean(mvals)) if mvals else None
        results["B7_RAW_MOTION"][h] = results["B4_RAW_TEMPORAL"][h]
        results["B8_RAW_NEIGHBOR"][h] = results["B4_RAW_TEMPORAL"][h]
        results["B9_RAW_MOTION_NEIGHBOR"][h] = results["B4_RAW_TEMPORAL"][h]
    return dict(results)


def zscore(vals: list[float | None]) -> list[float | None]:
    x = np.asarray([v for v in vals if v is not None], dtype=np.float32)
    if len(x) == 0:
        return [None] * len(vals)
    mu, sd = float(x.mean()), float(x.std())
    sd = sd if sd > 1e-6 else 1.0
    return [None if v is None else float((v - mu) / sd) for v in vals]


def add_group_combinations(score_rows: list[dict[str, Any]]) -> None:
    groups = defaultdict(list)
    for row in score_rows:
        groups[(row["sequence"], int(row["decision_frame"]), int(row["gid"]))].append(row)
    for members in groups.values():
        for h in (1, 5, 10, 20):
            raw = zscore([r["scores"]["B4_RAW_TEMPORAL"][str(h)] for r in members])
            mot = zscore([r["scores"]["B6_MOTION"][str(h)] for r in members])
            nei = zscore([
                (r["neighbor_raw"][str(h)] if r["neighbor_raw"].get(str(h)) is not None else None)
                for r in members
            ])
            for i, r in enumerate(members):
                components = [(raw[i], 1.0), (mot[i], 0.5), (nei[i], 0.5)]
                base = raw[i]
                if base is None:
                    r["scores"]["B7_RAW_MOTION"][str(h)] = None
                    r["scores"]["B8_RAW_NEIGHBOR"][str(h)] = None
                    r["scores"]["B9_RAW_MOTION_NEIGHBOR"][str(h)] = None
                    continue
                r["scores"]["B7_RAW_MOTION"][str(h)] = float(base + (mot[i] or 0.0) * 0.5)
                r["scores"]["B8_RAW_NEIGHBOR"][str(h)] = float(base + (nei[i] or 0.0) * 0.5)
                r["scores"]["B9_RAW_MOTION_NEIGHBOR"][str(h)] = float(
                    base + (mot[i] or 0.0) * 0.5 + (nei[i] or 0.0) * 0.5
                )


def auc_pairwise(groups: list[list[dict[str, Any]]], method: str, h: str) -> float | None:
    wins = total = 0.0
    for members in groups:
        pos = [r["scores"][method].get(h) for r in members if r["positive"] and r["scores"][method].get(h) is not None]
        neg = [r["scores"][method].get(h) for r in members if not r["positive"] and r["scores"][method].get(h) is not None]
        for p in pos:
            for n in neg:
                total += 1
                wins += 1.0 if p > n else 0.5 if p == n else 0.0
    return float(wins / total) if total else None


def metrics(groups: list[list[dict[str, Any]]], method: str, h: str, threshold: float | None):
    ranks, margins, present_total, present_available, absent_total, absent_fa = [], [], 0, 0, 0, 0
    accepted, correct_accepts = 0, 0
    for members in groups:
        valid = [r for r in members if r["scores"][method].get(h) is not None]
        if not valid:
            continue
        positive = [r for r in valid if r["positive"]]
        present = any(r["positive"] for r in members)
        if present:
            present_total += 1
            if positive:
                present_available += 1
                ordered = sorted(valid, key=lambda r: (-float(r["scores"][method][h]), int(r["candidate_rank"])))
                rank = next((i + 1 for i, r in enumerate(ordered) if r["positive"]), None)
                if rank is not None:
                    ranks.append(rank)
                    neg = [float(r["scores"][method][h]) for r in valid if not r["positive"]]
                    if neg:
                        margins.append(float(max(r["scores"][method][h] for r in positive) - max(neg)))
        else:
            absent_total += 1
        ordered = sorted(valid, key=lambda r: (-float(r["scores"][method][h]), int(r["candidate_rank"])))
        top = ordered[0]
        max_score = float(top["scores"][method][h])
        accept = threshold is not None and max_score >= threshold
        if accept:
            accepted += 1
            correct_accepts += int(top["positive"])
            if not present:
                absent_fa += 1
    n = len(ranks)
    return {
        "status": "COMPUTED",
        "groups": len(groups),
        "present_groups": present_total,
        "present_candidate_recall": present_available / max(1, present_total),
        "absent_groups": absent_total,
        "target_absent_false_acceptance": absent_fa / max(1, absent_total),
        "top1": sum(x == 1 for x in ranks) / max(1, n),
        "top3": sum(x <= 3 for x in ranks) / max(1, n),
        "top5": sum(x <= 5 for x in ranks) / max(1, n),
        "mrr": float(np.mean([1.0 / x for x in ranks])) if ranks else None,
        "hardest_negative_margin": float(np.mean(margins)) if margins else None,
        "pair_auc": auc_pairwise(groups, method, h),
        "threshold_from_train": threshold,
        "commit_precision": correct_accepts / max(1, accepted),
        "commit_coverage": accepted / max(1, len(groups)),
        "ece": None,
        "brier": None,
        "calibration_status": "NOT_RUN_BEFORE_INFORMATION_GATE",
    }


def threshold_from_train(groups, method, h):
    vals = []
    for members in groups:
        if any(r["positive"] for r in members):
            valid = [r["scores"][method].get(h) for r in members if r["scores"][method].get(h) is not None]
            if valid:
                vals.append(max(valid))
    return float(np.quantile(vals, 0.10)) if vals else None


def bootstrap_top1(groups, method, h, seed=25, n_boot=500):
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        sample = [groups[int(i)] for i in rng.integers(0, len(groups), len(groups))] if groups else []
        m = metrics(sample, method, h, None)
        vals.append(float(m["top1"]))
    if not vals:
        return {"n_boot": 0, "low": None, "high": None}
    return {"n_boot": n_boot, "low": float(np.quantile(vals, 0.025)), "high": float(np.quantile(vals, 0.975))}


def process_split(split: str, input_dir: Path, out_dir: Path):
    rows = load_rows(input_dir)
    public_ids = event_public_ids(split)
    current_seq = None
    store = None
    score_rows = []
    stats = {
        "split": split,
        "input_dir": str(input_dir),
        "raw_rows": len(rows),
        "sequences": sorted({r["sequence"] for r in rows}),
        "groups": len({(r["sequence"], int(r["frame"]), int(r["gid"])) for r in rows}),
        "candidate_rank_counts": {},
        "raw_crop_valid_steps": 0,
        "raw_crop_requested_steps": 0,
        "sam3_candidate_feature_rows": 0,
        "explicit_human_negative_rows": 0,
    }
    rank_counts = defaultdict(int)
    raw_dim = RAW_SIZE * RAW_SIZE * 3 + 6
    raw_arr = np.zeros((len(rows), MAX_H, raw_dim), dtype=np.float32)
    raw_valid = np.zeros((len(rows), MAX_H), dtype=np.float32)
    motion_arr = np.zeros((len(rows), MAX_H, 9), dtype=np.float32)
    neighbor_arr = np.zeros((len(rows), MAX_H, 4), dtype=np.float32)
    row_out = out_dir / f"episodes_{split}.jsonl"
    for ri, row in enumerate(rows):
        seq = str(row["sequence"])
        if seq != current_seq:
            store = SequenceStore(seq)
            current_seq = seq
        if store is None:
            continue
        q = store.query.get(int(row["gid"]))
        if q is None:
            continue
        qraw = store.raw(q["frame"], q["box"])
        q["raw"] = qraw
        raw = [None] * MAX_H
        neigh = np.zeros((MAX_H, 4), dtype=np.float32)
        for i, fr in enumerate(row.get("frames", [])[:MAX_H]):
            box = fr.get("box")
            if box is None:
                continue
            raw[i] = store.raw(int(fr["frame"]), box)
            neigh[i] = store.neighbours(int(fr["frame"]), box)
            stats["raw_crop_requested_steps"] += 1
            if raw[i] is not None:
                raw_arr[ri, i] = raw[i]
                raw_valid[ri, i] = 1.0
                stats["raw_crop_valid_steps"] += 1
        motion = motion_series(store, row.get("frames", []))
        motion_arr[ri] = motion
        neighbor_arr[ri] = neigh
        scores = candidate_scores(store, row, raw, motion, neigh)
        # Keep JSON compact and make the missing internal path explicit.
        scores_json = {m: {str(h): v for h, v in hs.items()} for m, hs in scores.items()}
        group_key = (seq, int(row["frame"]), int(row["gid"]))
        compact = {
                "sequence": seq,
                "public_identity_id": public_ids.get((seq, int(row["gid"]))),
                "gid": int(row["gid"]),
                "correction_frame": int(q["frame"]),
                "legal_human_positive": {
                    "frame": int(q["frame"]),
                    "box": [float(x) for x in q["box"]],
                    "source": "human_root_query_cache",
                },
                "decision_frame": int(row["frame"]),
                "candidate_rank": int(row["candidate_rank"]),
                "candidate_source": "GFN_top5_shadow",
                "candidate_start_box": [float(x) for x in row["start_box"]],
                "candidate_shadow_tracklet": [
                    {
                        "frame": int(fr["frame"]),
                        "box": None if fr.get("box") is None else [float(x) for x in fr["box"]],
                        "valid": bool(fr.get("box") is not None),
                        "propagation_status": "lost" if fr.get("box") is None else "propagated",
                    }
                    for fr in row.get("frames", [])[:MAX_H]
                ],
                "causal_validity": {
                    "observed_steps": int(sum(bool(fr.get("box") is not None) for fr in row.get("frames", [])[:MAX_H])),
                    "requested_horizons": {str(h): int(sum(int(fr["frame"]) < int(row["frame"]) + h and fr.get("box") is not None for fr in row.get("frames", [])[:MAX_H])) for h in (1, 5, 10, 20)},
                    "future_frames_used_for_decision": False,
                },
                "raw_visual_feature": {
                    "artifact": f"raw_features_{split}.npz",
                    "row_index": ri,
                    "descriptor": "fixed_resized_rgb_crop_plus_color_stats",
                    "valid_steps": [int(x) for x in np.flatnonzero(raw_valid[ri])],
                },
                "sam3_visual_feature": {
                    "status": "NOT_MATERIALIZED_CANDIDATE_ALIGNED",
                    "reason": "Stage-A pointer is multiplex-state/internal and consolidated mask memory did not align with public candidate rows; no zero fill",
                    "audit_artifact": "outputs/n25/feature_audit/n25_sam3_feature_audit.json",
                },
                "gfn_r0_feature": {"status": "available_in_pinned_cache", "source": "outputs/n18/route_c and outputs/n20/gfn_cache_r0"},
                "motion_feature": {"artifact": f"raw_features_{split}.npz", "row_index": ri, "dimensions": 9},
                "neighbor_feature": {"artifact": f"raw_features_{split}.npz", "row_index": ri, "dimensions": 4},
                "negative_provenance": {
                    "human_explicit_negative_available": False,
                    "posthoc_wrong_candidate": bool(not int(row.get("is_correct", 0))),
                    "hard_negative_role": "posthoc_only_not_human_denial",
                },
                "posthoc_labels_only": {
                    "candidate_is_correct": int(row.get("is_correct", 0)),
                    "frame_labels": [
                        {"frame": int(fr["frame"]), "iou": float(fr.get("iou") or 0.0), "correct": int(fr.get("correct") or 0)}
                        for fr in row.get("frames", [])[:MAX_H]
                    ],
                },
                "scores": scores_json,
                "neighbor_raw": {str(h): (None if scores["B4_RAW_TEMPORAL"].get(h) is None else float(np.mean(neigh[: max(1, min(h, MAX_H)), 1:4]))) for h in (1, 5, 10, 20)},
                "positive": bool(int(row.get("is_correct", 0))),
                "group_key": f"{group_key[0]}:{group_key[1]}:{group_key[2]}",
        }
        score_rows.append(compact)
        rank_counts[str(int(row["candidate_rank"]))] += 1
    # Relational raw/motion/neighbor combinations are normalized within each
    # candidate group only after all members are present.
    add_group_combinations(score_rows)
    with row_out.open("w", encoding="utf-8") as handle:
        for row in score_rows:
            handle.write(json.dumps(row, separators=(",", ":")) + "\n")
    np.savez_compressed(
        out_dir / f"raw_features_{split}.npz",
        raw=raw_arr.astype(np.float16),
        raw_valid=raw_valid,
        motion=motion_arr.astype(np.float32),
        neighbor=neighbor_arr.astype(np.float32),
    )
    stats["candidate_rank_counts"] = dict(rank_counts)
    stats["raw_crop_coverage"] = stats["raw_crop_valid_steps"] / max(1, stats["raw_crop_requested_steps"])
    stats["target_present_groups"] = sum(
        any(r["positive"] for r in members)
        for members in group_members(score_rows).values()
    )
    stats["target_absent_groups"] = stats["groups"] - stats["target_present_groups"]
    stats["sam3_candidate_feature_rows"] = 0
    stats["explicit_human_negative_rows"] = 0
    (out_dir / f"stats_{split}.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    return score_rows, stats


def group_members(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[row["group_key"]].append(row)
    return groups


def gate_rows(score_rows, split, thresholds):
    groups = list(group_members(score_rows).values())
    methods = ["B0_GFN", "B1_R0", "B2_GFN_R0", "B3_RAW_STATIC", "B4_RAW_TEMPORAL", "B5_SAM3_INTERNAL", "B6_MOTION", "B7_RAW_MOTION", "B8_RAW_NEIGHBOR", "B9_RAW_MOTION_NEIGHBOR", "B10_EXPLICIT_NEGATIVE", "B11_ALL_LEGAL"]
    out = []
    for h in (1, 5, 10, 20):
        for method in methods:
            if method in {"B5_SAM3_INTERNAL", "B10_EXPLICIT_NEGATIVE", "B11_ALL_LEGAL"}:
                out.append({"split": split, "history": h, "method": method, "status": "NOT_COMPUTABLE", "reason": "required candidate-aligned causal feature or explicit human negative is absent"})
                continue
            # Use string horizon keys because the JSON episode is machine-readable.
            key = str(h)
            m = metrics(groups, method, key, thresholds.get((method, key)))
            out.append({"split": split, "history": h, "method": method, **m})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="outputs/n25/dataset")
    ap.add_argument(
        "--train-cache-dir",
        default=str(N20_ROOT / "full_shadow_cache_train30"),
        help="Raw train30 all-candidate shadow directory.",
    )
    ap.add_argument(
        "--cal-cache-dir",
        default=str(N20_ROOT / "full_shadow_cache_cal10"),
        help="Raw cal10 all-candidate shadow directory.",
    )
    ap.add_argument("--reuse-episodes", action="store_true", help="reuse completed episode JSONL/NPZ artifacts")
    args = ap.parse_args()
    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = ROOT / out_dir
    train_cache_dir = Path(args.train_cache_dir)
    cal_cache_dir = Path(args.cal_cache_dir)
    if not train_cache_dir.is_absolute():
        train_cache_dir = ROOT / train_cache_dir
    if not cal_cache_dir.is_absolute():
        cal_cache_dir = ROOT / cal_cache_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    episode_files = (out_dir / "episodes_train30.jsonl", out_dir / "episodes_cal10.jsonl")
    if args.reuse_episodes and all(path.is_file() for path in episode_files):
        train_rows = load_episode_rows(episode_files[0])
        cal_rows = load_episode_rows(episode_files[1])
        train_stats = json.loads((out_dir / "stats_train30.json").read_text(encoding="utf-8"))
        cal_stats = json.loads((out_dir / "stats_cal10.json").read_text(encoding="utf-8"))
    else:
        train_rows, train_stats = process_split("train30", train_cache_dir, out_dir)
        cal_rows, cal_stats = process_split("cal10", cal_cache_dir, out_dir)
    train_groups = list(group_members(train_rows).values())
    methods = ["B0_GFN", "B1_R0", "B2_GFN_R0", "B3_RAW_STATIC", "B4_RAW_TEMPORAL", "B6_MOTION", "B7_RAW_MOTION", "B8_RAW_NEIGHBOR", "B9_RAW_MOTION_NEIGHBOR"]
    thresholds = {(method, str(h)): threshold_from_train(train_groups, method, str(h)) for method in methods for h in (1, 5, 10, 20)}
    results = gate_rows(train_rows, "train30", thresholds) + gate_rows(cal_rows, "cal10", thresholds)
    with (out_dir / "n25_information_sufficiency.csv").open("w", newline="", encoding="utf-8") as handle:
        keys = sorted({k for r in results for k in r})
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(results)
    boot = {}
    for method in ("B0_GFN", "B1_R0", "B2_GFN_R0", "B3_RAW_STATIC", "B4_RAW_TEMPORAL"):
        for h in ("1", "5", "10", "20"):
            boot[f"cal10/{method}/H{h}"] = bootstrap_top1(list(group_members(cal_rows).values()), method, h)
    (out_dir / "n25_bootstrap_ci.json").write_text(json.dumps(boot, indent=2), encoding="utf-8")
    manifest = {
        "name": "N25 raw on-policy episode audit",
        "protocol": {
            "gt_used_for_features": False,
            "gt_used_for_labels_only": True,
            "future_frames_used_for_decision": False,
            "canonical_val25_read": False,
            "max_history": MAX_H,
            "candidate_count": "variable_raw_top5_with_incomplete_rows",
            "source": "raw all-candidate shadow train30/cal10; not N24 compact NPZ",
            "train_cache_dir": str(train_cache_dir),
            "cal_cache_dir": str(cal_cache_dir),
        },
        "splits": {"train30": train_stats, "cal10": cal_stats},
        "candidate_sources": {
            "GFN_top5": {"status": "MATERIALIZED", "source": "N20 all-candidate shadow stream"},
            "N23_whole_frame": {"status": "LEGACY_REFERENCE_ONLY", "artifact": "outputs/n23/n23_query_discovery_calibration.json", "same_candidate_groups": False},
            "controlled_union": {"status": "NOT_MATERIALIZED", "reason": "N23 bank has a different event key and was not merged after Stage-A gate"},
            "SAM3_official_proposals": {"status": "NOT_MATERIALIZED", "reason": "Stage-A audit exposed internal state but no candidate-aligned public proposal cache"},
        },
        "episode_fields": ["sequence", "public_identity_id", "correction_frame", "legal_human_positive", "causal_validity", "candidate_source", "candidate_shadow_tracklet", "raw_visual_feature", "sam3_visual_feature", "gfn_r0_feature", "motion_feature", "neighbor_feature", "negative_provenance", "posthoc_labels_only"],
        "data_limits": [
            "Prior raw shadow stream has 9 cal10 sequences; dancetrack0087 is present in the ten-sequence FULL_LOOP ledger but has no N20 candidate-shadow rows.",
            "No per-candidate human denial field exists in the inherited ledger; posthoc wrong is not relabeled as explicit human negative.",
            "No first CCRIM on-policy rollout was run because Stage C information gate did not pass; no large training is authorized.",
        ],
        "sam3_internal_feature_policy": "not zero-filled; B5/B10/B11 remain NOT_COMPUTABLE",
        "thresholds_from_train30": {f"{m}/H{h}": v for (m, h), v in thresholds.items()},
        "artifacts": [str(out_dir / "episodes_train30.jsonl"), str(out_dir / "episodes_cal10.jsonl"), str(out_dir / "raw_features_train30.npz"), str(out_dir / "raw_features_cal10.npz"), str(out_dir / "n25_information_sufficiency.csv"), str(out_dir / "n25_bootstrap_ci.json")],
    }
    (out_dir / "n25_dataset_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({"train": train_stats, "cal": cal_stats, "gate": str(out_dir / "n25_information_sufficiency.csv")}, indent=2), flush=True)
    print("N25_DATASET_AND_GATE_DONE", flush=True)


if __name__ == "__main__":
    main()
