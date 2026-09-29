"""Frozen public CLIP visual encoder used by the N72R17 benchmark."""

from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path

import timm
import torch
from torch import Tensor
from torch.nn import functional as F

from .crop import CLIP_MEAN, CLIP_STD


class OpenAIClipEncoder:
    """OpenAI CLIP ViT-B/32 loaded through timm, with no identity training."""

    name = "openai_clip_vit_b32_zero_shot"
    model_name = "vit_base_patch32_clip_224.openai"
    input_size = (224, 224)
    embedding_dim = 512

    def __init__(self, device: str = "cpu"):
        self.device = torch.device(device)
        model = timm.create_model(self.model_name, pretrained=True, num_classes=self.embedding_dim)
        model.eval().to(self.device)
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        self.model = model

    @torch.inference_mode()
    def encode(self, batch: Tensor) -> Tensor:
        if batch.ndim != 4 or tuple(batch.shape[1:]) != (3, *self.input_size):
            raise ValueError(f"expected NCHW with C,H,W=3,224,224; got {tuple(batch.shape)}")
        autocast = (
            torch.autocast(device_type="cuda", dtype=torch.float16)
            if self.device.type == "cuda"
            else nullcontext()
        )
        with autocast:
            features = self.model(batch.to(self.device, non_blocking=True))
        if not isinstance(features, Tensor) or features.ndim != 2 or features.shape[1] != self.embedding_dim:
            raise ValueError(f"CLIP model returned unexpected features: {type(features)!r}, {getattr(features, 'shape', None)}")
        return F.normalize(features.float(), p=2, dim=1)

    def metadata(self) -> dict[str, object]:
        return {
            "encoder": self.name,
            "model_name": self.model_name,
            "embedding_dim": self.embedding_dim,
            "input_height": self.input_size[0],
            "input_width": self.input_size[1],
            "device": str(self.device),
            "weights_frozen": True,
            "trained_on_dancetrack": False,
            "identity_reid_finetuning": False,
            "normalization": "L2",
            "pixel_mean": list(CLIP_MEAN),
            "pixel_std": list(CLIP_STD),
            "public_model_id": "timm/vit_base_patch32_clip_224.openai",
            "public_model_url": "https://huggingface.co/timm/vit_base_patch32_clip_224.openai",
            "original_clip_url": "https://github.com/openai/CLIP",
        }


__all__ = ["OpenAIClipEncoder"]
