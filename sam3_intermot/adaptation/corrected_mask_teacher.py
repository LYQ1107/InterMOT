"""Legal spatial supervision for N29 decoder updates.

This module keeps pixel supervision and identity supervision separate.  It
never turns a missing candidate into a human negative and never labels a
synthetic rectangle as a confirmed click/mask.  Differentiable training uses
logits; thresholded masks are reserved for evaluation/candidate extraction.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Sequence

import numpy as np
import torch
from torch import Tensor
from torch.nn import functional as F


HUMAN_CONFIRMED_MASK = "HUMAN_CONFIRMED_MASK"
POINT_REFINED_CONFIRMED_MASK = "POINT_REFINED_CONFIRMED_MASK"
BOX_PROMPTED_CONFIRMED_MASK = "BOX_PROMPTED_CONFIRMED_MASK"
BOX_DERIVED_PSEUDO_MASK = "BOX_DERIVED_PSEUDO_MASK"
GT_MASK_ORACLE = "GT_MASK_ORACLE"

LEGAL_MASK_PROVENANCE = frozenset(
    {
        HUMAN_CONFIRMED_MASK,
        POINT_REFINED_CONFIRMED_MASK,
        BOX_PROMPTED_CONFIRMED_MASK,
        BOX_DERIVED_PSEUDO_MASK,
        GT_MASK_ORACLE,
    }
)
ONLINE_MASK_PROVENANCE = frozenset(
    {
        HUMAN_CONFIRMED_MASK,
        POINT_REFINED_CONFIRMED_MASK,
        BOX_PROMPTED_CONFIRMED_MASK,
        BOX_DERIVED_PSEUDO_MASK,
    }
)
CONFIRMED_MASK_PROVENANCE = frozenset(
    {
        HUMAN_CONFIRMED_MASK,
        POINT_REFINED_CONFIRMED_MASK,
        BOX_PROMPTED_CONFIRMED_MASK,
    }
)


class MaskSupervisionError(ValueError):
    """Raised when a mask event cannot be used under N29 provenance rules."""


@dataclass(frozen=True)
class MaskTeacherConfig:
    focal_alpha: float = 0.25
    focal_gamma: float = 2.0
    lambda_focal: float = 20.0
    lambda_dice: float = 1.0
    lambda_box: float = 0.25
    lambda_spatial_preserve: float = 0.1
    lambda_weight: float = 1.0e-4
    mask_threshold: float = 0.0


@dataclass(frozen=True)
class CompiledMaskSupervision:
    provenance: str
    mask_target: Optional[Tensor]
    box_xyxy: Optional[tuple[float, float, float, float]]
    source: str
    online_legal: bool
    is_confirmed_mask: bool
    is_pseudo: bool
    is_oracle: bool
    image_size: Optional[tuple[int, int]] = None

    @property
    def has_pixel_target(self) -> bool:
        return self.mask_target is not None


def _as_box(box_xyxy: Optional[Sequence[float]]) -> Optional[tuple[float, float, float, float]]:
    if box_xyxy is None:
        return None
    values = tuple(float(x) for x in box_xyxy)
    if len(values) != 4 or not all(np.isfinite(values)):
        raise MaskSupervisionError("box_xyxy must contain four finite values")
    if values[2] <= values[0] or values[3] <= values[1]:
        raise MaskSupervisionError("box_xyxy must have positive area")
    return values


def _as_mask(mask: object, *, dtype: torch.dtype = torch.float32) -> Tensor:
    tensor = torch.as_tensor(mask, dtype=dtype)
    if tensor.ndim == 3 and tensor.shape[0] == 1:
        tensor = tensor[0]
    if tensor.ndim != 2:
        raise MaskSupervisionError("corrected mask must be [H, W]")
    if tensor.numel() == 0 or not torch.isfinite(tensor).all():
        raise MaskSupervisionError("corrected mask must be finite and non-empty")
    return (tensor > 0).to(dtype)


def box_to_pseudo_mask(
    box_xyxy: Sequence[float],
    image_size: tuple[int, int],
    *,
    device: Optional[torch.device | str] = None,
    dtype: torch.dtype = torch.float32,
) -> Tensor:
    """Create an explicitly named weak rectangle target.

    This is allowed only for ``BOX_DERIVED_PSEUDO_MASK`` and is never labeled
    as a human-confirmed mask.  The rectangle is inclusive at the lower edge
    and exclusive at the upper edge after clipping.
    """

    height, width = (int(image_size[0]), int(image_size[1]))
    if height <= 0 or width <= 0:
        raise MaskSupervisionError("image_size must be positive")
    box = _as_box(box_xyxy)
    assert box is not None
    x1, y1, x2, y2 = box
    left = max(0, min(width - 1, int(np.floor(x1))))
    top = max(0, min(height - 1, int(np.floor(y1))))
    right = max(left + 1, min(width, int(np.ceil(x2))))
    bottom = max(top + 1, min(height, int(np.ceil(y2))))
    target = torch.zeros((height, width), device=device, dtype=dtype)
    target[top:bottom, left:right] = 1.0
    return target


def compile_mask_supervision(
    *,
    provenance: str,
    corrected_mask: Optional[object] = None,
    box_xyxy: Optional[Sequence[float]] = None,
    image_size: Optional[tuple[int, int]] = None,
    allow_oracle: bool = False,
) -> CompiledMaskSupervision:
    """Validate a user event and compile its legal spatial target."""

    if provenance not in LEGAL_MASK_PROVENANCE:
        raise MaskSupervisionError(f"illegal mask provenance: {provenance}")
    box = _as_box(box_xyxy)
    if provenance == GT_MASK_ORACLE and not allow_oracle:
        raise MaskSupervisionError("GT_MASK_ORACLE is evaluation-only")
    if provenance in CONFIRMED_MASK_PROVENANCE and corrected_mask is None:
        raise MaskSupervisionError(
            f"{provenance} requires a supplied confirmed mask; no mask may be fabricated"
        )
    if provenance == BOX_DERIVED_PSEUDO_MASK:
        if corrected_mask is not None:
            target = _as_mask(corrected_mask)
            source = "box_derived_event_with_supplied_mask"
        else:
            if box is None or image_size is None:
                raise MaskSupervisionError(
                    "BOX_DERIVED_PSEUDO_MASK needs box_xyxy and image_size"
                )
            target = box_to_pseudo_mask(box, image_size)
            source = "explicit_box_rectangle_pseudo_target"
        return CompiledMaskSupervision(
            provenance=provenance,
            mask_target=target,
            box_xyxy=box,
            source=source,
            online_legal=True,
            is_confirmed_mask=False,
            is_pseudo=True,
            is_oracle=False,
            image_size=image_size,
        )
    target = None if corrected_mask is None else _as_mask(corrected_mask)
    return CompiledMaskSupervision(
        provenance=provenance,
        mask_target=target,
        box_xyxy=box,
        source=("gt_mask_oracle" if provenance == GT_MASK_ORACLE else "user_confirmed_mask"),
        online_legal=provenance in ONLINE_MASK_PROVENANCE,
        is_confirmed_mask=provenance in CONFIRMED_MASK_PROVENANCE,
        is_pseudo=False,
        is_oracle=provenance == GT_MASK_ORACLE,
        image_size=image_size,
    )


def focal_binary_loss(
    logits: Tensor,
    target: Tensor,
    *,
    alpha: float = 0.25,
    gamma: float = 2.0,
) -> Tensor:
    logits, target = _align_logits_target(logits, target)
    probability = torch.sigmoid(logits)
    ce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    p_t = probability * target + (1.0 - probability) * (1.0 - target)
    alpha_t = alpha * target + (1.0 - alpha) * (1.0 - target)
    return (alpha_t * (1.0 - p_t).pow(gamma) * ce).mean()


def dice_loss(logits: Tensor, target: Tensor, eps: float = 1.0e-6) -> Tensor:
    logits, target = _align_logits_target(logits, target)
    probability = torch.sigmoid(logits).reshape(-1)
    target = target.reshape(-1)
    intersection = (probability * target).sum()
    return 1.0 - (2.0 * intersection + eps) / (
        probability.sum() + target.sum() + eps
    )


def weak_box_loss(
    logits: Tensor,
    box_xyxy: Sequence[float],
    *,
    source_image_size: Optional[tuple[int, int]] = None,
) -> Tensor:
    """Differentiable box-only weak loss; no thresholded mask is used."""

    logits = _squeeze_mask(logits)
    height, width = logits.shape[-2:]
    box = _as_box(box_xyxy)
    assert box is not None
    x1, y1, x2, y2 = box
    if source_image_size is not None:
        source_height, source_width = (int(value) for value in source_image_size)
        if source_height <= 0 or source_width <= 0:
            raise MaskSupervisionError("source image size must be positive")
        x1, x2 = x1 * width / source_width, x2 * width / source_width
        y1, y2 = y1 * height / source_height, y2 * height / source_height
    yy = torch.arange(height, device=logits.device, dtype=logits.dtype) + 0.5
    xx = torch.arange(width, device=logits.device, dtype=logits.dtype) + 0.5
    inside_x = (xx >= x1) & (xx < x2)
    inside_y = (yy >= y1) & (yy < y2)
    inside = inside_y[:, None] & inside_x[None, :]
    inside = inside.to(dtype=logits.dtype)
    if inside.sum() == 0:
        return logits.new_zeros(())
    probability = torch.sigmoid(logits)
    inside_loss = ((1.0 - probability) * inside).sum() / inside.sum().clamp_min(1.0)
    outside = 1.0 - inside
    outside_loss = (probability * outside).sum() / outside.sum().clamp_min(1.0)
    return inside_loss + outside_loss


def spatial_preserve_loss(logits: Tensor, reference_logits: Tensor) -> Tensor:
    """Keep the update local when a frozen/reference logit map is available."""

    logits, reference = _align_logits_target(logits, reference_logits)
    return F.smooth_l1_loss(torch.sigmoid(logits), torch.sigmoid(reference))


def decoder_loss(
    logits: Tensor,
    supervision: CompiledMaskSupervision,
    *,
    config: MaskTeacherConfig = MaskTeacherConfig(),
    reference_logits: Optional[Tensor] = None,
    weight_delta: Optional[Iterable[Tensor]] = None,
) -> dict[str, Tensor]:
    """Compute spatial losses only; identity labels are compiled elsewhere."""

    terms: dict[str, Tensor] = {}
    total = logits.new_zeros(())
    if supervision.mask_target is not None:
        target = supervision.mask_target.to(device=logits.device, dtype=logits.dtype)
        terms["focal"] = focal_binary_loss(
            logits,
            target,
            alpha=config.focal_alpha,
            gamma=config.focal_gamma,
        )
        terms["dice"] = dice_loss(logits, target)
        total = total + config.lambda_focal * terms["focal"]
        total = total + config.lambda_dice * terms["dice"]
    if supervision.box_xyxy is not None:
        terms["box_weak"] = weak_box_loss(
            logits,
            supervision.box_xyxy,
            source_image_size=supervision.image_size,
        )
        total = total + config.lambda_box * terms["box_weak"]
    if reference_logits is not None:
        terms["spatial_preserve"] = spatial_preserve_loss(logits, reference_logits)
        total = total + config.lambda_spatial_preserve * terms["spatial_preserve"]
    if weight_delta is not None:
        terms["weight_l2"] = sum(
            (tensor.float().pow(2).mean() for tensor in weight_delta),
            logits.new_zeros(()),
        )
        total = total + config.lambda_weight * terms["weight_l2"]
    terms["total"] = total
    return terms


def mask_iou(
    prediction_logits: Tensor,
    target: Tensor,
    *,
    threshold: float = 0.0,
) -> float:
    prediction = (_squeeze_mask(prediction_logits).detach().cpu().numpy() > threshold)
    truth = (_squeeze_mask(target).detach().cpu().numpy() > 0.5)
    intersection = np.logical_and(prediction, truth).sum()
    union = np.logical_or(prediction, truth).sum()
    return float(intersection / union) if union else 1.0


def _squeeze_mask(value: Tensor) -> Tensor:
    result = value
    while result.ndim > 2 and result.shape[0] == 1:
        result = result[0]
    while result.ndim > 2 and result.shape[1] == 1:
        result = result[:, 0]
    if result.ndim != 2:
        raise MaskSupervisionError(f"expected one mask, got shape={tuple(value.shape)}")
    return result


def _align_logits_target(logits: Tensor, target: Tensor) -> tuple[Tensor, Tensor]:
    logits = _squeeze_mask(logits)
    target = _squeeze_mask(target.to(device=logits.device, dtype=logits.dtype))
    if logits.shape != target.shape:
        target = F.interpolate(
            target[None, None],
            size=logits.shape,
            mode="nearest",
        )[0, 0]
    return logits, target
