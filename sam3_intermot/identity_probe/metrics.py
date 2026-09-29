"""Hard-negative identity metrics for the frozen N72R16 question."""

from __future__ import annotations

from collections import defaultdict
import json
from pathlib import Path
from typing import Iterable

import numpy as np

from .protocol import Anchor, FutureObservation, iter_anchors
from .storage import EmbeddingStore


HORIZONS = (20, 50, 100)
TIME_BINS = ((1, 5, "1_5"), (6, 20, "6_20"), (21, 50, "21_50"), (51, 100, "51_100"))


def _cosine_scores(anchor: np.ndarray, candidates: Iterable[np.ndarray]) -> np.ndarray:
    matrix = np.asarray(list(candidates), dtype=np.float32)
    if matrix.ndim != 2 or matrix.shape[0] == 0:
        raise ValueError("at least one candidate embedding is required")
    anchor = np.asarray(anchor, dtype=np.float32).reshape(-1)
    anchor = anchor / max(float(np.linalg.norm(anchor)), 1e-12)
    matrix = matrix / np.maximum(np.linalg.norm(matrix, axis=1, keepdims=True), 1e-12)
    return matrix @ anchor


def _iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    left = max(ax, bx)
    top = max(ay, by)
    right = min(ax + aw, bx + bw)
    bottom = min(ay + ah, by + bh)
    intersection = max(0.0, right - left) * max(0.0, bottom - top)
    union = aw * ah + bw * bh - intersection
    return intersection / union if union > 0 else 0.0


def _spatial_index(target, competitors, mode: str) -> int:
    if mode == "nearest_center":
        tx, ty = target.center
        return min(
            range(len(competitors)),
            key=lambda i: ((competitors[i].center[0] - tx) ** 2 + (competitors[i].center[1] - ty) ** 2, competitors[i].track_id),
        )
    if mode == "highest_iou":
        return max(range(len(competitors)), key=lambda i: (_iou(target.tlwh, competitors[i].tlwh), -competitors[i].track_id))
    raise ValueError(f"unknown spatial hard mode: {mode}")


def sample_records(document: dict[str, object], store: EmbeddingStore, spatial_mode: str = "nearest_center") -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for anchor in iter_anchors(document):
        anchor_embedding = store.get(anchor.sequence, anchor.anchor_frame, anchor.track_id)
        for future in anchor.future:
            if not future.competitors:
                continue
            target_embedding = store.get(anchor.sequence, future.frame, anchor.track_id)
            competitor_embeddings = [store.get(anchor.sequence, future.frame, box.track_id) for box in future.competitors]
            scores = _cosine_scores(anchor_embedding, [target_embedding, *competitor_embeddings])
            positive_score = float(scores[0])
            hard_index = int(np.argmax(scores[1:]))
            hard_negative_score = float(scores[1 + hard_index])
            margin = positive_score - hard_negative_score
            rank = 1 + int(np.sum(scores[1:] > positive_score + 1e-12))
            spatial_index = _spatial_index(future.target, future.competitors, spatial_mode)
            spatial_score = float(scores[1 + spatial_index])
            iou_index = _spatial_index(future.target, future.competitors, "highest_iou")
            iou_score = float(scores[1 + iou_index])
            records.append(
                {
                    "sequence": anchor.sequence,
                    "track_id": anchor.track_id,
                    "anchor_frame": anchor.anchor_frame,
                    "frame": future.frame,
                    "gap": future.frame - anchor.anchor_frame,
                    "positive_score": positive_score,
                    "hard_negative_score": hard_negative_score,
                    "margin": margin,
                    "win": bool(positive_score > hard_negative_score),
                    "rank": rank,
                    "rank_1": bool(rank == 1),
                    "rank_2": bool(rank <= 2),
                    "rank_3": bool(rank <= 3),
                    "competitor_count": len(future.competitors),
                    "appearance_hard_track_id": future.competitors[hard_index].track_id,
                    "spatial_hard_score": spatial_score,
                    "spatial_hard_margin": positive_score - spatial_score,
                    "iou_hard_score": iou_score,
                    "iou_hard_margin": positive_score - iou_score,
                }
            )
    return records


def _percentile(values: np.ndarray, percentile: float) -> float | None:
    return float(np.percentile(values, percentile)) if values.size else None


def sequence_cluster_ci(records: list[dict[str, object]], field: str, reps: int = 2000, seed: int = 7216) -> dict[str, object]:
    if not records:
        return {"reps": reps, "seed": seed, "lower": None, "upper": None, "clusters": 0}
    by_sequence: dict[str, list[dict[str, object]]] = defaultdict(list)
    for record in records:
        by_sequence[str(record["sequence"])].append(record)
    names = sorted(by_sequence)
    rng = np.random.default_rng(seed)
    values: list[float] = []
    for _ in range(reps):
        selected = rng.integers(0, len(names), size=len(names))
        numer = 0.0
        denom = 0
        for index in selected:
            cluster = by_sequence[names[int(index)]]
            if field == "win":
                numer += sum(bool(row[field]) for row in cluster)
                denom += len(cluster)
            else:
                numer += sum(float(row[field]) for row in cluster)
                denom += len(cluster)
        values.append(numer / denom if denom else float("nan"))
    array = np.asarray(values, dtype=np.float64)
    array = array[np.isfinite(array)]
    return {
        "reps": reps,
        "seed": seed,
        "lower": float(np.percentile(array, 2.5)) if array.size else None,
        "upper": float(np.percentile(array, 97.5)) if array.size else None,
        "clusters": len(names),
    }


def summarize_records(records: list[dict[str, object]], bootstrap_reps: int = 2000, seed: int = 7216) -> dict[str, object]:
    def summarize(subset: list[dict[str, object]], label: str) -> dict[str, object]:
        wins = np.asarray([bool(row["win"]) for row in subset], dtype=np.float64)
        margins = np.asarray([float(row["margin"]) for row in subset], dtype=np.float64)
        ranks = np.asarray([int(row["rank"]) for row in subset], dtype=np.float64)
        return {
            "label": label,
            "samples": int(len(subset)),
            "sequences": int(len({str(row["sequence"]) for row in subset})),
            "hard_negative_win_rate": float(wins.mean()) if wins.size else None,
            "hard_negative_win_rate_sequence_cluster_ci95": sequence_cluster_ci(subset, "win", bootstrap_reps, seed),
            "rank_1_accuracy": float(np.mean(ranks == 1)) if ranks.size else None,
            "rank_2_accuracy": float(np.mean(ranks <= 2)) if ranks.size else None,
            "rank_3_accuracy": float(np.mean(ranks <= 3)) if ranks.size else None,
            "mrr": float(np.mean(1.0 / ranks)) if ranks.size else None,
            "mean_margin": float(margins.mean()) if margins.size else None,
            "median_margin": float(np.median(margins)) if margins.size else None,
            "p10_margin": _percentile(margins, 10),
            "p25_margin": _percentile(margins, 25),
            "p75_margin": _percentile(margins, 75),
            "mean_spatial_hard_margin": float(np.mean([float(row["spatial_hard_margin"]) for row in subset])) if subset else None,
            "mean_iou_hard_margin": float(np.mean([float(row["iou_hard_margin"]) for row in subset])) if subset else None,
        }

    horizons = {
        f"H{horizon}": summarize([row for row in records if int(row["gap"]) <= horizon], f"cumulative_H{horizon}")
        for horizon in HORIZONS
    }
    bins = {
        label: summarize([row for row in records if low <= int(row["gap"]) <= high], label)
        for low, high, label in TIME_BINS
    }
    return {
        "metric_version": "N72R16-hard-negative-metrics-v1",
        "primary_endpoint": "VAL_H100_HARD_NEGATIVE_IDENTITY_WIN_RATE",
        "sample_count": len(records),
        "horizons": horizons,
        "time_gap_bins": bins,
        "tie_policy": "strict positive_score > hard_negative_score; ties are losses for the primary win rate",
        "rank_tie_policy": "rank is one plus the number of competitors with strictly greater score",
        "bootstrap": {"unit": "sequence", "reps": bootstrap_reps, "seed": seed},
    }


def write_records(path: str | Path, records: list[dict[str, object]]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
