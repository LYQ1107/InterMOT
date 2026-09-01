"""Online association scorers (pairwise / set) and one-to-one matching."""

from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from sam3_intermot.n9.models import MLP
from sam3_intermot.association.appearance_memory import AppearanceMemory

MOTION_DIM = 12


def box_iou(a: np.ndarray, b: np.ndarray) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def center_dist(a: np.ndarray, b: np.ndarray) -> float:
    ca = np.asarray([(a[0] + a[2]) / 2, (a[1] + a[3]) / 2])
    cb = np.asarray([(b[0] + b[2]) / 2, (b[1] + b[3]) / 2])
    return float(np.linalg.norm(ca - cb))


def predicted_iou(state, box: np.ndarray, frame: int) -> float:
    if state.last_box is None:
        return 0.0
    gap = max(0, frame - state.last_seen_frame)
    pred = np.asarray(state.last_box, dtype=float).copy()
    if gap > 0:
        pred[0] += state.velocity[0] * gap
        pred[2] += state.velocity[0] * gap
        pred[1] += state.velocity[1] * gap
        pred[3] += state.velocity[1] * gap
    return box_iou(pred, np.asarray(box, dtype=float))


def motion_vec(state, obs, frame: int, n_obs: int) -> np.ndarray:
    """12-d online motion/geometry vector shared by training and inference."""
    b = np.asarray(obs["box"], dtype=float)
    gap = max(0, frame - state.last_seen_frame)
    dist = center_dist(state.last_box, b) if state.last_box is not None else 1000.0
    native_same = 1.0 if obs["native_tid"] == state.last_native_tid else 0.0
    native_pos = 1.0 if state.has_positive(obs["native_tid"], frame) else 0.0
    return np.asarray(
        [
            min(1.0, gap / 200.0),
            min(1.0, (frame - state.birth_frame) / 2000.0),
            min(1.0, n_obs / 20.0),
            predicted_iou(state, b, frame),
            min(1.0, dist / 1000.0),
            min(1.0, (b[2] - b[0]) / 2000.0),
            min(1.0, (b[3] - b[1]) / 1000.0),
            min(1.0, state.last_seen_frame / 2000.0),
            native_same,
            min(1.0, obs["native_age"] / 2000.0),
            float(obs["has_feat"]),
            native_pos,
        ],
        dtype=np.float32,
    )


def has_valid_feature(observation: dict) -> bool:
    """Return whether an observation carries a usable causal embedding.

    Candidate-complete replay must distinguish a missing/zero embedding from a
    real feature.  The check is intentionally local and does not consult GT,
    future frames, or a native-track fallback.
    """
    if not bool(observation.get("has_feat", 1.0)):
        return False
    try:
        value = np.asarray(observation["feat"], dtype=np.float32).reshape(-1)
    except (KeyError, TypeError, ValueError):
        return False
    return bool(value.size > 0 and np.all(np.isfinite(value)) and np.linalg.norm(value) > 1e-6)


class PairwiseMLP(nn.Module):
    """Same-identity logit from (memory feature, observation feature, motion)."""

    def __init__(self, feat_dim: int = 512, motion_dim: int = MOTION_DIM, hidden: int = 256):
        super().__init__()
        self.net = MLP(feat_dim * 2 + motion_dim, hidden, 1, depth=3)

    def forward(
        self,
        mem_feat: torch.Tensor,
        row_feat: torch.Tensor,
        motion: torch.Tensor,
    ) -> torch.Tensor:
        return self.net(torch.cat([mem_feat, row_feat, motion], dim=-1)).squeeze(-1)


class SetAssociator(nn.Module):
    """Lightweight set-level cross-attention scorer (rows x memories)."""

    def __init__(
        self,
        feat_dim: int = 512,
        motion_dim: int = MOTION_DIM,
        d: int = 128,
        layers: int = 2,
        heads: int = 2,
    ):
        super().__init__()
        self.d = d
        self.mem_proj = MLP(feat_dim + motion_dim, d, d, depth=2)
        self.row_proj = MLP(feat_dim + motion_dim, d, d, depth=2)
        self.layers = nn.ModuleList(
            [nn.MultiheadAttention(d, heads, batch_first=True) for _ in range(layers)]
        )
        self.out_proj = nn.Linear(d, d)
        self.logit_scale = nn.Parameter(torch.ones(1) * 8.0)
        self.motion_net = nn.Sequential(nn.Linear(motion_dim * 2, 32), nn.ReLU(), nn.Linear(32, 1))

    def forward(
        self,
        mem_feat: torch.Tensor,
        row_feat: torch.Tensor,
        mem_motion: torch.Tensor,
        row_motion: torch.Tensor,
        mem_mask: Optional[torch.Tensor] = None,
        row_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        m = self.mem_proj(torch.cat([mem_feat, mem_motion], dim=-1))
        r = self.row_proj(torch.cat([row_feat, row_motion], dim=-1))
        for layer in self.layers:
            r2, _ = layer(r, m, m)
            r = r + r2
        r = self.out_proj(r)
        logits = torch.bmm(r, m.transpose(1, 2)) * self.logit_scale
        B, R, M = logits.shape
        rmv = row_motion[:, :, None, :].expand(B, R, M, -1)
        mmv = mem_motion[:, None, :, :].expand(B, R, M, -1)
        logits = logits + self.motion_net(torch.cat([rmv, mmv], dim=-1)).squeeze(-1)
        if row_mask is not None:
            logits = logits.masked_fill(~row_mask.unsqueeze(2), -1e9)
        if mem_mask is not None:
            logits = logits.masked_fill(~mem_mask.unsqueeze(1), -1e9)
        return logits


def hungarian_max(scores: np.ndarray) -> np.ndarray:
    """Max-weight one-to-one assignment; returns obs->state index or -1."""
    n, m = scores.shape
    if n == 0 or m == 0:
        return np.full(n, -1, dtype=int)
    from scipy.optimize import linear_sum_assignment

    rows, cols = linear_sum_assignment(-scores)
    assign = np.full(n, -1, dtype=int)
    for r, c in zip(rows, cols):
        assign[r] = c
    return assign


def score_matrix_pairwise(
    states: List,
    obs_list: List[dict],
    frame: int,
    model: Optional[nn.Module],
    reid_weights: Optional[dict] = None,
    positive_bonus: float = 5.0,
    native_bonus: float = 3.0,
    authority_mode: str = "permanent",
    hard_frames: int = 1,
    decay_frames: int = 8,
    refresh_threshold: float = 0.5,
    appearance_memory: Optional[AppearanceMemory] = None,
    appearance_score_weight: float = 1.0,
    appearance_positive_weight: float = 1.0,
    appearance_negative_weight: float = 1.0,
    score_audit: Optional[dict] = None,
) -> np.ndarray:
    n = len(obs_list)
    m = len(states)
    if n == 0 or m == 0:
        if score_audit is not None:
            score_audit.clear()
            score_audit.update(
                {
                    "base_scores_before_appearance": np.zeros((n, m), dtype=np.float32).tolist(),
                    "appearance_memory_scores": np.zeros((n, m), dtype=np.float32).tolist(),
                    "appearance_score_deltas": np.zeros((n, m), dtype=np.float32).tolist(),
                    "fused_scores": np.zeros((n, m), dtype=np.float32).tolist(),
                }
            )
        return np.zeros((n, m), dtype=np.float32)
    scores = np.zeros((n, m), dtype=np.float32)
    if model is not None:
        mem = np.stack(
            [
                s.effective_feat(
                    frame,
                    authority_mode=authority_mode,
                    hard_frames=hard_frames,
                    decay_frames=decay_frames,
                    refresh_threshold=refresh_threshold,
                )
                for s in states
            ]
        )[None].repeat(n, axis=0).reshape(n * m, -1)
        rows = np.stack([o["feat"] for o in obs_list])[:, None].repeat(m, axis=1).reshape(n * m, -1)
        mot = np.stack(
            [
                motion_vec(s, o, frame, len(obs_list))
                for o in obs_list
                for s in states
            ]
        )
        with torch.no_grad():
            logits = (
                model(
                    torch.as_tensor(mem),
                    torch.as_tensor(rows),
                    torch.as_tensor(mot),
                )
                .cpu()
                .numpy()
                .reshape(n, m)
            )
        scores = logits
    else:
        for i, o in enumerate(obs_list):
            for j, s in enumerate(states):
                sim = float(np.dot(o["feat"], s.effective_feat()))
                piou = predicted_iou(s, o["box"], frame)
                gap = min(1.0, max(0, frame - s.last_seen_frame) / 200.0)
                nsame = 1.0 if o["native_tid"] == s.last_native_tid else 0.0
                w = reid_weights or {"sim": 1.5, "iou": 1.0, "native": 0.5, "gap": 0.1}
                scores[i, j] = (
                    w["sim"] * sim
                    + w["iou"] * piou
                    + w["native"] * nsame
                    - w["gap"] * gap
                )
    # native continuity: same P0 tracklet as the identity's last match is a
    # strong short-term cue (a cue only; public identity is never native tid)
    for i, o in enumerate(obs_list):
        for j, s in enumerate(states):
            if o["native_tid"] == s.last_native_tid:
                scores[i, j] += native_bonus
    # Preserve the complete legacy score as the audit baseline.  The final
    # hard-negative pass below is repeated after the additive appearance term
    # so the new evidence can never override a native constraint.
    for i, o in enumerate(obs_list):
        for j, s in enumerate(states):
            if s.has_negative(o["native_tid"], frame):
                scores[i, j] = -1e9
            elif s.has_positive(o["native_tid"], frame):
                scores[i, j] += positive_bonus
    base_scores = scores.copy()
    appearance_memory_scores = np.zeros_like(scores, dtype=np.float32)
    appearance_deltas = np.zeros_like(scores, dtype=np.float32)
    for i, o in enumerate(obs_list):
        for j, s in enumerate(states):
            if appearance_memory is not None and has_valid_feature(o):
                memory_score = float(
                    appearance_memory.score(
                        s.pid,
                        o["feat"],
                        frame,
                        positive_weight=appearance_positive_weight,
                        negative_weight=appearance_negative_weight,
                    )
                )
                appearance_memory_scores[i, j] = memory_score
                appearance_deltas[i, j] = float(appearance_score_weight) * memory_score
                scores[i, j] += appearance_deltas[i, j]
    # Re-apply hard negatives after the additive term.  Positive/native terms
    # were already included in ``base_scores`` and remain additive.
    for i, o in enumerate(obs_list):
        for j, s in enumerate(states):
            if s.has_negative(o["native_tid"], frame):
                scores[i, j] = -1e9
    if score_audit is not None:
        score_audit.clear()
        score_audit.update(
            {
                "base_scores_before_appearance": base_scores.astype(float).tolist(),
                "appearance_memory_scores": appearance_memory_scores.astype(float).tolist(),
                "appearance_score_deltas": appearance_deltas.astype(float).tolist(),
                "fused_scores": scores.astype(float).tolist(),
            }
        )
    return scores


def score_matrix_set(
    states: List,
    obs_list: List[dict],
    frame: int,
    model: nn.Module,
    positive_bonus: float = 5.0,
    native_bonus: float = 3.0,
    authority_mode: str = "permanent",
    hard_frames: int = 1,
    decay_frames: int = 8,
    refresh_threshold: float = 0.5,
    appearance_memory: Optional[AppearanceMemory] = None,
    appearance_score_weight: float = 1.0,
    appearance_positive_weight: float = 1.0,
    appearance_negative_weight: float = 1.0,
    score_audit: Optional[dict] = None,
) -> np.ndarray:
    n = len(obs_list)
    m = len(states)
    if n == 0 or m == 0:
        if score_audit is not None:
            score_audit.clear()
            score_audit.update(
                {
                    "base_scores_before_appearance": np.zeros((n, m), dtype=np.float32).tolist(),
                    "appearance_memory_scores": np.zeros((n, m), dtype=np.float32).tolist(),
                    "appearance_score_deltas": np.zeros((n, m), dtype=np.float32).tolist(),
                    "fused_scores": np.zeros((n, m), dtype=np.float32).tolist(),
                }
            )
        return np.zeros((n, m), dtype=np.float32)
    mem = np.stack(
        [
            s.effective_feat(
                frame,
                authority_mode=authority_mode,
                hard_frames=hard_frames,
                decay_frames=decay_frames,
                refresh_threshold=refresh_threshold,
            )
            for s in states
        ]
    )
    rows = np.stack([o["feat"] for o in obs_list])
    mem_mot = np.stack(
        [
            np.asarray(
                [
                    min(1.0, max(0, frame - s.last_seen_frame) / 200.0),
                    min(1.0, (frame - s.birth_frame) / 2000.0),
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    min(1.0, s.last_seen_frame / 2000.0),
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                ],
                dtype=np.float32,
            )
            for s in states
        ]
    )
    row_mot = np.stack(
        [
            np.asarray(
                [
                    0.0,
                    0.0,
                    min(1.0, n / 20.0),
                    0.0,
                    0.0,
                    min(1.0, (o["box"][2] - o["box"][0]) / 2000.0),
                    min(1.0, (o["box"][3] - o["box"][1]) / 1000.0),
                    min(1.0, frame / 2000.0),
                    0.0,
                    0.0,
                    float(o["has_feat"]),
                    0.0,
                ],
                dtype=np.float32,
            )
            for o in obs_list
        ]
    )
    with torch.no_grad():
        logits = model(
            torch.as_tensor(mem[None]),
            torch.as_tensor(rows[None]),
            torch.as_tensor(mem_mot[None]),
            torch.as_tensor(row_mot[None]),
        )[0].cpu().numpy()
    for i, o in enumerate(obs_list):
        for j, s in enumerate(states):
            if s.has_negative(o["native_tid"], frame):
                logits[i, j] = -1e9
            elif s.has_positive(o["native_tid"], frame):
                logits[i, j] += positive_bonus
            elif o["native_tid"] == s.last_native_tid:
                logits[i, j] += native_bonus
    base_scores = logits.copy()
    appearance_memory_scores = np.zeros_like(logits, dtype=np.float32)
    appearance_deltas = np.zeros_like(logits, dtype=np.float32)
    for i, o in enumerate(obs_list):
        for j, s in enumerate(states):
            if appearance_memory is not None and has_valid_feature(o):
                memory_score = float(
                    appearance_memory.score(
                        s.pid,
                        o["feat"],
                        frame,
                        positive_weight=appearance_positive_weight,
                        negative_weight=appearance_negative_weight,
                    )
                )
                appearance_memory_scores[i, j] = memory_score
                appearance_deltas[i, j] = float(appearance_score_weight) * memory_score
                logits[i, j] += appearance_deltas[i, j]
    # Re-apply hard constraints after the additive memory term.  This keeps
    # appearance evidence from overriding the existing native identity rules.
    for i, o in enumerate(obs_list):
        for j, s in enumerate(states):
            if s.has_negative(o["native_tid"], frame):
                logits[i, j] = -1e9
    if score_audit is not None:
        score_audit.clear()
        score_audit.update(
            {
                "base_scores_before_appearance": base_scores.astype(float).tolist(),
                "appearance_memory_scores": appearance_memory_scores.astype(float).tolist(),
                "appearance_score_deltas": appearance_deltas.astype(float).tolist(),
                "fused_scores": logits.astype(float).tolist(),
            }
        )
    return logits
