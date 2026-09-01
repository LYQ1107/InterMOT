"""N23 correction-conditioned proposal primitives.

N17 attempted to decode boxes directly from a frozen SAM3 encoder grid.  N23
uses a different boundary: a human correction is a query, and a multi-scale
window generator searches the whole future frame.  This module contains only
the small trainable pair scorer and deterministic proposal geometry; the
expensive CLIP-ReID feature extraction is kept in the reproducible scripts.
"""

from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np
import torch
from torch import nn


IMAGE_W = 1920.0
IMAGE_H = 1080.0
ANCHOR_DIM = 1280
GEOM_DIM = 10


def _cxcywh(box: torch.Tensor) -> torch.Tensor:
    x1, y1, x2, y2 = box.unbind(-1)
    return torch.stack(
        [
            (x1 + x2) / 2.0 / IMAGE_W,
            (y1 + y2) / 2.0 / IMAGE_H,
            (x2 - x1).clamp_min(1.0) / IMAGE_W,
            (y2 - y1).clamp_min(1.0) / IMAGE_H,
        ],
        dim=-1,
    )


def pair_geometry(
    query_box: torch.Tensor,
    candidate_boxes: torch.Tensor,
    delta: torch.Tensor | float,
) -> torch.Tensor:
    """Return bounded geometry for a correction query and candidate boxes.

    The absolute candidate location is retained so the learned scorer can use
    the correction's spatial prior, while relative center/scale terms allow
    it to distinguish a target-consistent window from a same-class distractor.
    """

    q = _cxcywh(query_box.reshape(1, 4)).expand(candidate_boxes.shape[0], -1)
    c = _cxcywh(candidate_boxes)
    d = torch.as_tensor(delta, dtype=c.dtype, device=c.device).reshape(1, 1)
    d = (d / 120.0).expand(candidate_boxes.shape[0], 1).clamp(0.0, 1.0)
    rel = c[:, :2] - q[:, :2]
    ratio = torch.log((c[:, 2:4] + 1e-4) / (q[:, 2:4] + 1e-4)).clamp(-3.0, 3.0)
    dist = torch.linalg.vector_norm(rel, dim=-1, keepdim=True).clamp(0.0, 2.0)
    return torch.cat([c, rel, ratio, dist, d], dim=-1)


class PairRanker(nn.Module):
    """Small query/candidate compatibility adapter.

    CLIP-ReID remains frozen.  The adapter learns a sequence-local
    correction-to-future compatibility relation from train30 human-query
    episodes and is evaluated on calibration sequences without updating.
    """

    def __init__(
        self,
        anchor_dim: int = ANCHOR_DIM,
        projection_dim: int = 128,
        hidden: int = 256,
        geom_dim: int = GEOM_DIM,
    ) -> None:
        super().__init__()
        self.anchor_dim = anchor_dim
        self.projection_dim = projection_dim
        self.geom_dim = geom_dim
        self.query_proj = nn.Sequential(
            nn.LayerNorm(anchor_dim),
            nn.Linear(anchor_dim, projection_dim),
            nn.GELU(),
        )
        self.candidate_proj = nn.Sequential(
            nn.LayerNorm(anchor_dim),
            nn.Linear(anchor_dim, projection_dim),
            nn.GELU(),
        )
        self.head = nn.Sequential(
            nn.LayerNorm(4 * projection_dim + geom_dim),
            nn.Linear(4 * projection_dim + geom_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden // 2),
            nn.GELU(),
            nn.Linear(hidden // 2, 1),
        )

    def forward(
        self,
        query: torch.Tensor,
        candidate: torch.Tensor,
        geometry: torch.Tensor,
    ) -> torch.Tensor:
        q = self.query_proj(query)
        c = self.candidate_proj(candidate)
        z = torch.cat([q, c, torch.abs(q - c), q * c, geometry], dim=-1)
        return self.head(z).squeeze(-1)


class NoneGate(nn.Module):
    """Episode-level gate with an explicit NONE outcome."""

    def __init__(self, in_dim: int = 7, hidden: int = 32) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features).squeeze(-1)


def _clip_box(box: np.ndarray, image_w: float, image_h: float) -> np.ndarray:
    x1, y1, x2, y2 = [float(v) for v in box]
    x1 = min(max(x1, 0.0), image_w - 1.0)
    y1 = min(max(y1, 0.0), image_h - 1.0)
    x2 = min(max(x2, x1 + 2.0), image_w)
    y2 = min(max(y2, y1 + 2.0), image_h)
    return np.asarray([x1, y1, x2, y2], dtype=np.float32)


def generate_windows(
    anchor_box: Sequence[float],
    image_w: float = IMAGE_W,
    image_h: float = IMAGE_H,
    scales: Iterable[float] = (0.70, 0.90, 1.10, 1.35, 1.65),
    stride_fraction: float = 0.85,
    max_windows: int = 1200,
) -> np.ndarray:
    """Generate a deterministic whole-frame, person-shaped proposal bank.

    It is intentionally independent of GFN/SAM3 detections.  The anchor only
    supplies expected aspect/scale; centers cover the whole image.  A later
    scorer can return NONE, so this broad bank is not itself an acceptance.
    """

    x1, y1, x2, y2 = [float(v) for v in anchor_box]
    aw = max(4.0, x2 - x1)
    ah = max(4.0, y2 - y1)
    out: list[np.ndarray] = []
    seen: set[tuple[int, int, int, int]] = set()
    for scale in scales:
        w = min(image_w, aw * float(scale))
        h = min(image_h, ah * float(scale))
        sx = max(32.0, w * stride_fraction)
        sy = max(32.0, h * stride_fraction)
        xs = np.arange(w / 2.0, image_w - w / 2.0 + 0.01, sx)
        ys = np.arange(h / 2.0, image_h - h / 2.0 + 0.01, sy)
        # Include the correction center even when the regular grid phase is
        # unlucky, then sweep all centers across the frame.
        cx0 = (x1 + x2) / 2.0
        cy0 = (y1 + y2) / 2.0
        xs = np.unique(np.concatenate([xs, np.asarray([cx0])]))
        ys = np.unique(np.concatenate([ys, np.asarray([cy0])]))
        for cx in xs:
            for cy in ys:
                box = _clip_box(
                    [cx - w / 2.0, cy - h / 2.0, cx + w / 2.0, cy + h / 2.0],
                    image_w,
                    image_h,
                )
                key = tuple(np.rint(box).astype(int).tolist())
                if key not in seen:
                    seen.add(key)
                    out.append(box)
    # Keep the output bounded for tiny targets.  The center-first additions
    # make the local correction neighborhood survive the deterministic cap.
    if len(out) > max_windows:
        qcx, qcy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        out.sort(
            key=lambda b: ((b[0] + b[2]) / 2 - qcx) ** 2
            + ((b[1] + b[3]) / 2 - qcy) ** 2
        )
        out = out[:max_windows]
    return np.stack(out, axis=0).astype(np.float32)


def gate_features(
    logits: torch.Tensor,
    raw_scores: torch.Tensor,
    boxes: torch.Tensor,
    query_box: torch.Tensor,
    delta: float,
) -> torch.Tensor:
    """Build the seven-dimensional episode feature used by ``NoneGate``."""

    order = torch.argsort(logits, descending=True)
    top = logits[order[0]]
    second = logits[order[1]] if logits.numel() > 1 else logits[order[0]]
    raw = raw_scores[order[0]]
    margin = top - second
    q = _cxcywh(query_box.reshape(1, 4))[0]
    c = _cxcywh(boxes[order[0]].reshape(1, 4))[0]
    dist = torch.linalg.vector_norm(c[:2] - q[:2]).clamp(0.0, 2.0)
    return torch.stack(
        [
            top,
            second,
            margin,
            raw,
            dist,
            torch.as_tensor(float(delta) / 120.0, device=logits.device).clamp(0, 1),
            torch.as_tensor(float(logits.numel()) / 1200.0, device=logits.device),
        ]
    )

