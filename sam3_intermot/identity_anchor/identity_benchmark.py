"""Human Seed Identity Benchmark construction.

Queries mimic a human correction: the GT box of one identity at frame t.
Galleries are built at future deltas with the same identity (positive) and
other identities visible at the same future frame (negatives).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


@dataclass
class Crop:
    crop_id: int
    seq: str
    frame: int
    gid: int
    box: Tuple[float, float, float, float]  # x1,y1,x2,y2


@dataclass
class Query:
    query_id: str
    seq: str
    split: str
    query_frame: int
    query_gid: int
    delta: int
    query_crop_id: int = -1
    gallery_crop_ids: List[int] = field(default_factory=list)
    positive_crop_ids: List[int] = field(default_factory=list)
    negative_crop_ids: List[int] = field(default_factory=list)
    crowd: int = 0
    query_area: float = 0.0


def load_gt(seq_dir: Path) -> Dict[int, List[Tuple[int, np.ndarray]]]:
    gt_path = seq_dir / "gt" / "gt.txt"
    out: Dict[int, List[Tuple[int, np.ndarray]]] = {}
    if not gt_path.exists():
        return out
    for line in gt_path.read_text(encoding="utf-8").splitlines():
        parts = line.strip().split(",")
        if len(parts) < 6:
            continue
        frame_1based = int(float(parts[0]))
        gid = int(float(parts[1]))
        x, y, w, h = (float(parts[2]), float(parts[3]), float(parts[4]), float(parts[5]))
        if w <= 0 or h <= 0:
            continue
        frame = frame_1based - 1
        out.setdefault(frame, []).append((gid, np.asarray([x, y, x + w, y + h], float)))
    return out


def center(box: np.ndarray) -> np.ndarray:
    return (box[:2] + box[2:]) / 2.0


def build_benchmark(
    dancetrack_root: Path,
    seqs: List[str],
    split: str,
    out_json: Path,
    label: Optional[str] = None,
    max_queries_per_seq: int = 50,
    max_ids_per_frame: int = 2,
    deltas: Tuple[int, ...] = (1, 3, 5, 10, 30),
    max_negs: int = 6,
    min_area: float = 1200.0,
    seed: int = 42,
) -> Tuple[List[Crop], List[Query]]:
    rng = np.random.default_rng(seed)
    split_label = label or split
    crops: List[Crop] = []
    crop_key: Dict[Tuple[str, int, int], int] = {}
    queries: List[Query] = []
    crop_id = 0
    query_id = 0

    for seq in seqs:
        seq_dir = dancetrack_root / split / seq
        gt = load_gt(seq_dir)
        frames = sorted(gt.keys())
        if not frames:
            continue
        n = len(frames)
        # evenly sample query frames, skipping the last 30 frames (need future)
        usable = [f for f in frames if f + max(deltas) < n]
        if not usable:
            continue
        step = max(1, len(usable) // max_queries_per_seq)
        sampled_frames = usable[::step][:max_queries_per_seq]
        for tf in sampled_frames:
            rows = sorted(gt[tf], key=lambda r: -((r[1][2] - r[1][0]) * (r[1][3] - r[1][1])))
            picked = 0
            for gid, box in rows:
                if picked >= max_ids_per_frame:
                    break
                area = (box[2] - box[0]) * (box[3] - box[1])
                if area < min_area:
                    continue
                # ensure at least one positive future frame exists
                has_future = any(
                    any(g == gid for g, _ in gt.get(tf + d, [])) for d in deltas
                )
                if not has_future:
                    continue
                picked += 1
                for d in deltas:
                    target_frame = tf + d
                    if target_frame not in gt:
                        continue
                    pos = [r for r in gt[target_frame] if r[0] == gid]
                    if not pos:
                        continue
                    pos_box = pos[0][1]
                    # query crop
                    qk = (seq, tf, gid)
                    if qk not in crop_key:
                        crops.append(Crop(crop_id, seq, tf, gid, tuple(box)))
                        crop_key[qk] = crop_id
                        crop_id += 1
                    q_crop = crop_key[qk]
                    # positive crop
                    pk = (seq, target_frame, gid)
                    if pk not in crop_key:
                        crops.append(Crop(crop_id, seq, target_frame, gid, tuple(pos_box)))
                        crop_key[pk] = crop_id
                        crop_id += 1
                    p_crop = crop_key[pk]
                    # negatives: other identities at target frame, nearest center first
                    others = [r for r in gt[target_frame] if r[0] != gid]
                    qc = center(box)
                    negs = []
                    for ogid, obox in others:
                        dist = float(np.linalg.norm(qc - center(obox)))
                        negs.append((dist, ogid, obox))
                    negs.sort(key=lambda t: t[0])
                    negs = negs[:max_negs]
                    neg_crops = []
                    for _, ngid, nbox in negs:
                        nk = (seq, target_frame, ngid)
                        if nk not in crop_key:
                            crops.append(Crop(crop_id, seq, target_frame, ngid, tuple(nbox)))
                            crop_key[nk] = crop_id
                            crop_id += 1
                        neg_crops.append(crop_key[nk])
                    queries.append(
                        Query(
                            query_id=f"{seq}_{tf}_{gid}_d{d}",
                            seq=seq,
                            split=split_label,
                            query_frame=tf,
                            query_gid=gid,
                            query_crop_id=q_crop,
                            delta=d,
                            gallery_crop_ids=[p_crop] + neg_crops,
                            positive_crop_ids=[p_crop],
                            negative_crop_ids=neg_crops,
                            crowd=len(rows),
                            query_area=area,
                        )
                    )
                    query_id += 1

    out_json.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "crops": [
            {"crop_id": c.crop_id, "seq": c.seq, "frame": c.frame, "gid": c.gid, "box": list(c.box)}
            for c in crops
        ],
        "queries": [
            {
                "query_id": q.query_id,
                "seq": q.seq,
                "split": q.split,
                "query_frame": q.query_frame,
                "query_gid": q.query_gid,
                "query_crop_id": q.query_crop_id,
                "delta": q.delta,
                "gallery_crop_ids": q.gallery_crop_ids,
                "positive_crop_ids": q.positive_crop_ids,
                "negative_crop_ids": q.negative_crop_ids,
                "crowd": q.crowd,
                "query_area": q.query_area,
            }
            for q in queries
        ],
    }
    out_json.write_text(json.dumps(payload), encoding="utf-8")
    return crops, queries
