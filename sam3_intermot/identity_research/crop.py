"""On-the-fly square CLIP crops with no image files written to disk."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from torchvision.io import ImageReadMode, read_image
from torchvision.ops import roi_align
from torchvision.transforms import InterpolationMode
from torchvision.transforms.functional import normalize, resize

from .dataset import GTBox

CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


@dataclass(frozen=True)
class ClipCropConfig:
    height: int = 224
    width: int = 224
    context: float = 1.0


def _expanded_box(box: GTBox, image_width: int, image_height: int, context: float) -> tuple[int, int, int, int]:
    if context <= 0:
        raise ValueError("context must be positive")
    cx, cy = box.center
    width = box.width * context
    height = box.height * context
    left = max(0, int(round(cx - width / 2.0)))
    top = max(0, int(round(cy - height / 2.0)))
    right = min(image_width, int(round(cx + width / 2.0)))
    bottom = min(image_height, int(round(cy + height / 2.0)))
    if right <= left or bottom <= top:
        raise ValueError(f"empty crop for {box.sequence}:{box.frame}:{box.track_id}")
    return left, top, right, bottom


def _prepare(image: torch.Tensor, box: GTBox, config: ClipCropConfig) -> torch.Tensor:
    _, image_height, image_width = image.shape
    left, top, right, bottom = _expanded_box(box, image_width, image_height, config.context)
    crop = image[:, top:bottom, left:right].float().div(255.0)
    crop = resize(
        crop,
        [config.height, config.width],
        interpolation=InterpolationMode.BICUBIC,
        antialias=True,
    )
    normalize(crop, CLIP_MEAN, CLIP_STD, inplace=True)
    return crop


def load_clip_person_crops(
    image_path: str | Path,
    boxes: list[GTBox],
    config: ClipCropConfig = ClipCropConfig(),
) -> list[torch.Tensor]:
    """Decode a frame once and return CLIP-ready tensors without persistence."""

    image = read_image(str(image_path), mode=ImageReadMode.RGB)
    return [_prepare(image, box, config) for box in boxes]


def load_clip_person_crops_gpu(
    image_path: str | Path,
    boxes: list[GTBox],
    device: torch.device,
    config: ClipCropConfig = ClipCropConfig(),
) -> torch.Tensor:
    """Return a batched crop tensor using GPU ROI resize, without persistence.

    The crop rectangle is the same integer rectangle as the CPU backend.  GPU
    ROI resize avoids launching a separate CPU interpolation operation for
    every person while keeping the frame decode and crop provenance explicit.
    """

    image = read_image(str(image_path), mode=ImageReadMode.RGB).float().div(255.0)
    _, image_height, image_width = image.shape
    rectangles = [
        _expanded_box(box, image_width, image_height, config.context)
        for box in boxes
    ]
    if not rectangles:
        return torch.empty((0, 3, config.height, config.width), device=device)
    roi_boxes = torch.tensor(
        [[0.0, float(left), float(top), float(right), float(bottom)] for left, top, right, bottom in rectangles],
        dtype=torch.float32,
        device=device,
    )
    image = image.unsqueeze(0).to(device, non_blocking=True)
    crops = roi_align(
        image,
        roi_boxes,
        output_size=(config.height, config.width),
        spatial_scale=1.0,
        aligned=False,
    )
    normalize(crops, CLIP_MEAN, CLIP_STD, inplace=True)
    return crops


__all__ = [
    "CLIP_MEAN",
    "CLIP_STD",
    "ClipCropConfig",
    "load_clip_person_crops",
    "load_clip_person_crops_gpu",
]
