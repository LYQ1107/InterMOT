"""On-the-fly person crop preparation for the N72R16 encoder."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from torchvision.io import ImageReadMode, read_image
from torchvision.transforms import InterpolationMode
from torchvision.transforms.functional import normalize, resize

from .dataset import GTBox


@dataclass(frozen=True)
class CropConfig:
    height: int = 256
    width: int = 128
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


def load_person_crop(image_path: str | Path, box: GTBox, config: CropConfig = CropConfig()) -> torch.Tensor:
    """Return an ImageNet-normalized CHW float tensor; no crop is saved."""
    image = read_image(str(image_path), mode=ImageReadMode.RGB)
    return _crop_tensor(image, box, config)


def _crop_tensor(image: torch.Tensor, box: GTBox, config: CropConfig) -> torch.Tensor:
    _, image_height, image_width = image.shape
    left, top, right, bottom = _expanded_box(box, image_width, image_height, config.context)
    tensor = image[:, top:bottom, left:right].float().div(255.0)
    tensor = resize(tensor, [config.height, config.width], interpolation=InterpolationMode.BILINEAR, antialias=True)
    normalize(tensor, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225], inplace=True)
    return tensor


def load_person_crops(image_path: str | Path, boxes: list[GTBox], config: CropConfig = CropConfig()) -> list[torch.Tensor]:
    """Decode one frame once and return its on-the-fly crops."""
    image = read_image(str(image_path), mode=ImageReadMode.RGB)
    return [_crop_tensor(image, box, config) for box in boxes]
