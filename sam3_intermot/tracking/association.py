"""Explainable, frozen rule-based association for the first version."""

from typing import Dict, List, Tuple

import numpy as np


def box_iou(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def center_distance(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    ca = np.asarray([(a[0] + a[2]) / 2, (a[1] + a[3]) / 2])
    cb = np.asarray([(b[0] + b[2]) / 2, (b[1] + b[3]) / 2])
    return float(np.linalg.norm(ca - cb))


def greedy_associate(
    detections: List[Tuple[int, np.ndarray]],
    tracks: List[Tuple[int, np.ndarray]],
    iou_threshold: float = 0.2,
    center_threshold: float = 120.0,
) -> List[Tuple[int, int]]:
    """Greedy detection->track matching by IoU then center distance."""
    matches: List[Tuple[int, int]] = []
    used_tracks = set()
    remaining = list(tracks)
    for det_id, det_box in detections:
        best_track: int | None = None
        best_score = -1.0
        for trk_id, trk_box in remaining:
            if trk_id in used_tracks:
                continue
            iou = box_iou(det_box, trk_box)
            dist = center_distance(det_box, trk_box)
            if iou >= iou_threshold and dist <= center_threshold:
                score = iou - 1e-4 * dist
                if score > best_score:
                    best_score = score
                    best_track = trk_id
        if best_track is not None:
            matches.append((det_id, best_track))
            used_tracks.add(best_track)
    return matches
