"""Offline correction-conditioned residual for official SAM3 decoder inputs.

The module is intentionally independent of the frozen SAM3 implementation.  It
only produces a per-correction residual for the official
``extra_per_object_embeddings`` input, so the base decoder and all other
identity slices remain untouched.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, Optional

import torch
from torch import Tensor, nn
from torch.nn import functional as F


ACTION_NAMES: tuple[str, ...] = (
    "ADD",
    "RECOVER",
    "REASSIGN",
    "ID_SWAP",
    "BOX_CORRECTION",
)
ACTION_TO_ID = {name: index for index, name in enumerate(ACTION_NAMES)}


@dataclass(frozen=True)
class CorrectionMemoryInputs:
    """Current-time, identity-scoped evidence available at a correction."""

    e_obj: Tensor
    e_roi: Tensor
    e_pred: Tensor
    f_clip: Tensor
    g_box: Tensor
    g_residual: Tensor
    missing_flag: Tensor
    action: Tensor


def _as_batch(value: Tensor, *, ndim_without_batch: int) -> Tensor:
    if value.ndim == ndim_without_batch:
        return value.unsqueeze(0)
    if value.ndim != ndim_without_batch + 1:
        raise ValueError(f"expected {ndim_without_batch}D or batched tensor, got {tuple(value.shape)}")
    return value


class CorrectionMemoryWriter(nn.Module):
    """Fixed N30-C writer architecture.

    ``public_id`` is deliberately absent from the forward signature.  Identity
    scope is represented by the transaction key and target decoder slice, not
    by a learnable lookup over training IDs.
    """

    def __init__(
        self,
        *,
        clip_dim: int = 1280,
        object_embedding_dim: int = 256,
        num_object_tokens: int = 16,
        rank: int = 4,
        num_actions: int = len(ACTION_NAMES),
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if object_embedding_dim != 256:
            raise ValueError("N30-C fixes the official object embedding width at 256")
        if rank != 4:
            raise ValueError("N30-C fixes residual rank at 4")
        if num_object_tokens <= 0:
            raise ValueError("num_object_tokens must be positive")
        self.clip_dim = int(clip_dim)
        self.object_embedding_dim = int(object_embedding_dim)
        self.num_object_tokens = int(num_object_tokens)
        self.rank = int(rank)
        self.roi_proj = nn.Sequential(nn.Linear(256, 128), nn.GELU(), nn.LayerNorm(128))
        self.pred_roi_proj = nn.Sequential(nn.Linear(256, 128), nn.GELU(), nn.LayerNorm(128))
        self.obj_proj = nn.Sequential(nn.Linear(256, 128), nn.GELU(), nn.LayerNorm(128))
        self.clip_proj = nn.Sequential(nn.Linear(clip_dim, 128), nn.GELU(), nn.LayerNorm(128))
        self.geom_proj = nn.Sequential(
            nn.Linear(8 + 4 + 1, 64),
            nn.GELU(),
            nn.Linear(64, 64),
        )
        self.action_embed = nn.Embedding(num_actions, 32)
        self.z_mlp = nn.Sequential(
            nn.Linear(608, 512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
        )
        self.u_head = nn.Linear(256, num_object_tokens * rank)
        self.v_head = nn.Linear(256, rank * object_embedding_dim)
        self.gate_head = nn.Linear(256, 1)
        self._initialize_heads()

    def _initialize_heads(self) -> None:
        # U must be live at step zero.  V is exactly zero so the complete
        # writer is initially equivalent to the frozen write-only baseline,
        # while V receives a non-zero first gradient.
        nn.init.kaiming_uniform_(self.u_head.weight, a=math.sqrt(5))
        nn.init.zeros_(self.u_head.bias)
        nn.init.zeros_(self.v_head.weight)
        nn.init.zeros_(self.v_head.bias)
        nn.init.zeros_(self.gate_head.weight)
        nn.init.constant_(self.gate_head.bias, -2.0)

    @property
    def parameter_count(self) -> int:
        return sum(int(parameter.numel()) for parameter in self.parameters())

    def forward(self, inputs: CorrectionMemoryInputs | Mapping[str, Tensor]) -> dict[str, Tensor]:
        if isinstance(inputs, Mapping):
            inputs = CorrectionMemoryInputs(**inputs)
        # Official SAM3 tape exposes ``extra_per_object_embeddings`` as
        # [B,N,256].  The unbatched form accepted by this adapter is
        # [N,256], hence the feature rank without batch is 2 (not 3).
        e_obj = _as_batch(inputs.e_obj, ndim_without_batch=2)
        e_roi = _as_batch(inputs.e_roi, ndim_without_batch=1)
        e_pred = _as_batch(inputs.e_pred, ndim_without_batch=1)
        f_clip = _as_batch(inputs.f_clip, ndim_without_batch=1)
        g_box = _as_batch(inputs.g_box, ndim_without_batch=1)
        g_residual = _as_batch(inputs.g_residual, ndim_without_batch=1)
        missing_flag = _as_batch(inputs.missing_flag, ndim_without_batch=1)
        action = _as_batch(inputs.action, ndim_without_batch=0).long().reshape(-1)
        if e_obj.shape[-1] != 256 or e_obj.shape[1] != self.num_object_tokens:
            raise ValueError(
                "extra_per_object_embeddings shape does not match writer: "
                f"got {tuple(e_obj.shape)}, expected [B,{self.num_object_tokens},256]"
            )
        if e_roi.shape[-1] != 256 or e_pred.shape[-1] != 256:
            raise ValueError("ROI evidence must be 256-D")
        if f_clip.shape[-1] != self.clip_dim:
            raise ValueError(f"CLIP evidence must be {self.clip_dim}-D")
        if g_box.shape[-1] != 8 or g_residual.shape[-1] != 4 or missing_flag.shape[-1] != 1:
            raise ValueError("geometry evidence must have widths 8, 4 and 1")
        if action.shape[0] != e_obj.shape[0]:
            raise ValueError("action batch does not match evidence batch")
        z = torch.cat(
            (
                self.roi_proj(e_roi),
                self.pred_roi_proj(e_pred),
                self.obj_proj(e_obj.mean(dim=1)),
                self.clip_proj(f_clip),
                self.geom_proj(torch.cat((g_box, g_residual, missing_flag), dim=-1)),
                self.action_embed(action),
            ),
            dim=-1,
        )
        if z.shape[-1] != 608:
            raise RuntimeError(f"N30-C writer concatenation must be 608-D, got {z.shape[-1]}")
        z = self.z_mlp(z)
        u = self.u_head(z).reshape(-1, self.num_object_tokens, self.rank)
        v = self.v_head(z).reshape(-1, self.rank, self.object_embedding_dim)
        delta_e = torch.tanh(torch.matmul(u, v) / math.sqrt(self.rank))
        gate = torch.sigmoid(self.gate_head(z))
        residual = gate.unsqueeze(-1) * 0.1 * delta_e
        return {
            "z": z,
            "u": u,
            "v": v,
            "delta_e": delta_e,
            "gate": gate,
            "residual": residual,
        }

    def architecture_summary(self) -> dict[str, Any]:
        return {
            "class": type(self).__name__,
            "clip_dim": self.clip_dim,
            "object_embedding_dim": self.object_embedding_dim,
            "num_object_tokens": self.num_object_tokens,
            "rank": self.rank,
            "z_widths": [608, 512, 256],
            "residual_scale": 0.1,
            "initial_gate_bias": -2.0,
            "initial_v_weight_and_bias": "strict_zero",
            "public_id_input": False,
            "parameter_count": self.parameter_count,
        }


def roi_average_pool(image_embeddings: Tensor, boxes_xyxy: Tensor) -> Tensor:
    """Differentiable average ROI pooling on a [B,256,H,W] feature map."""

    image_embeddings = _as_batch(image_embeddings, ndim_without_batch=3)
    boxes_xyxy = _as_batch(boxes_xyxy, ndim_without_batch=1)
    if image_embeddings.shape[0] != boxes_xyxy.shape[0] or boxes_xyxy.shape[-1] != 4:
        raise ValueError("ROI map and boxes have incompatible batch shapes")
    _, _, height, width = image_embeddings.shape
    outputs = []
    for feature, box in zip(image_embeddings, boxes_xyxy):
        x1, y1, x2, y2 = box
        left = int(torch.floor(x1.clamp(0, 1) * width).item())
        top = int(torch.floor(y1.clamp(0, 1) * height).item())
        right = int(torch.ceil(x2.clamp(0, 1) * width).item())
        bottom = int(torch.ceil(y2.clamp(0, 1) * height).item())
        left = min(max(left, 0), max(width - 1, 0))
        top = min(max(top, 0), max(height - 1, 0))
        right = min(max(right, left + 1), width)
        bottom = min(max(bottom, top + 1), height)
        outputs.append(feature[:, top:bottom, left:right].mean(dim=(-1, -2)))
    return torch.stack(outputs, dim=0)


def action_id(action: str) -> int:
    try:
        return ACTION_TO_ID[str(action).upper()]
    except KeyError as exc:
        raise ValueError(f"unknown N30-C action: {action}") from exc
