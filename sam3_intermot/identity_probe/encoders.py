"""Frozen OSNet x1.0 encoder used by the N72R16 probe.

The implementation follows the public Torchreid OSNet x1.0 module so the
official Market-1501 checkpoint can be loaded without installing Torchreid.
This checkpoint is a fresh public dependency for N72R16; it is not claimed to
be equivalent to any historical InterMOT checkpoint.
"""

from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
from typing import Mapping

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class ConvLayer(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: Tensor) -> Tensor:
        return self.relu(self.bn(self.conv(x)))


class Conv1x1(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: Tensor) -> Tensor:
        return self.relu(self.bn(self.conv(x)))


class Conv1x1Linear(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x: Tensor) -> Tensor:
        return self.bn(self.conv(x))


class LightConv3x3(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, groups=out_channels, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: Tensor) -> Tensor:
        return self.relu(self.bn(self.conv2(self.conv1(x))))


class ChannelGate(nn.Module):
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.global_avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(channels, channels // reduction, 1)
        self.norm1 = None
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(channels // reduction, channels, 1)
        self.gate_activation = nn.Sigmoid()

    def forward(self, x: Tensor) -> Tensor:
        gate = self.fc1(self.global_avgpool(x))
        gate = self.relu(gate)
        gate = self.gate_activation(self.fc2(gate))
        return x * gate


class OSBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        mid_channels = out_channels // 4
        self.conv1 = Conv1x1(in_channels, mid_channels)
        self.conv2a = LightConv3x3(mid_channels, mid_channels)
        self.conv2b = nn.Sequential(LightConv3x3(mid_channels, mid_channels), LightConv3x3(mid_channels, mid_channels))
        self.conv2c = nn.Sequential(
            LightConv3x3(mid_channels, mid_channels),
            LightConv3x3(mid_channels, mid_channels),
            LightConv3x3(mid_channels, mid_channels),
        )
        self.conv2d = nn.Sequential(
            LightConv3x3(mid_channels, mid_channels),
            LightConv3x3(mid_channels, mid_channels),
            LightConv3x3(mid_channels, mid_channels),
            LightConv3x3(mid_channels, mid_channels),
        )
        self.gate = ChannelGate(mid_channels)
        self.conv3 = Conv1x1Linear(mid_channels, out_channels)
        self.downsample = Conv1x1Linear(in_channels, out_channels) if in_channels != out_channels else None

    def forward(self, x: Tensor) -> Tensor:
        identity = x if self.downsample is None else self.downsample(x)
        x1 = self.conv1(x)
        x2 = self.gate(self.conv2a(x1))
        x2 = x2 + self.gate(self.conv2b(x1))
        x2 = x2 + self.gate(self.conv2c(x1))
        x2 = x2 + self.gate(self.conv2d(x1))
        return F.relu(self.conv3(x2) + identity)


class OSNet(nn.Module):
    def __init__(self, num_classes: int = 751, feature_dim: int = 512):
        super().__init__()
        self.conv1 = ConvLayer(3, 64, 7, stride=2, padding=3)
        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)
        self.conv2 = self._make_layer(64, 256, reduce=True)
        self.conv3 = self._make_layer(256, 384, reduce=True)
        self.conv4 = self._make_layer(384, 512, reduce=False)
        self.conv5 = Conv1x1(512, 512)
        self.global_avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(nn.Linear(512, feature_dim), nn.BatchNorm1d(feature_dim), nn.ReLU(inplace=True))
        self.classifier = nn.Linear(feature_dim, num_classes)

    @staticmethod
    def _make_layer(in_channels: int, out_channels: int, reduce: bool) -> nn.Sequential:
        blocks: list[nn.Module] = [OSBlock(in_channels, out_channels), OSBlock(out_channels, out_channels)]
        if reduce:
            blocks.append(nn.Sequential(Conv1x1(out_channels, out_channels), nn.AvgPool2d(2, stride=2)))
        return nn.Sequential(*blocks)

    def featuremaps(self, x: Tensor) -> Tensor:
        x = self.maxpool(self.conv1(x))
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        return self.conv5(x)

    def forward(self, x: Tensor) -> Tensor:
        x = self.global_avgpool(self.featuremaps(x)).flatten(1)
        return self.fc(x)


def _state_dict(checkpoint: object) -> Mapping[str, Tensor]:
    if not isinstance(checkpoint, Mapping):
        raise TypeError(f"unsupported checkpoint type: {type(checkpoint)!r}")
    state = checkpoint.get("state_dict", checkpoint)
    if not isinstance(state, Mapping):
        raise TypeError("checkpoint state_dict is not a mapping")
    normalized: OrderedDict[str, Tensor] = OrderedDict()
    for key, value in state.items():
        key = key[7:] if key.startswith("module.") else key
        if isinstance(value, Tensor):
            normalized[key] = value
    return normalized


class OSNetEncoder:
    """Frozen 512-D L2-normalized OSNet x1.0 feature extractor."""

    name = "osnet_x1_0_market1501"
    input_size = (256, 128)
    embedding_dim = 512

    def __init__(self, checkpoint_path: str | Path, device: str = "cpu"):
        self.checkpoint_path = Path(checkpoint_path).expanduser().resolve()
        if not self.checkpoint_path.is_file():
            raise FileNotFoundError(f"missing OSNet checkpoint: {self.checkpoint_path}")
        self.device = torch.device(device)
        model = OSNet(num_classes=751, feature_dim=self.embedding_dim)
        checkpoint = torch.load(self.checkpoint_path, map_location="cpu", weights_only=False)
        state = _state_dict(checkpoint)
        model.load_state_dict(state, strict=True)
        model.eval().to(self.device)
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        self.model = model

    @torch.inference_mode()
    def encode(self, batch: Tensor) -> Tensor:
        if batch.ndim != 4 or tuple(batch.shape[1:]) != (3, *self.input_size):
            raise ValueError(f"expected NCHW with C,H,W=3,256,128; got {tuple(batch.shape)}")
        features = self.model(batch.to(self.device, non_blocking=True))
        return F.normalize(features.float(), p=2, dim=1)

    def metadata(self) -> dict[str, object]:
        return {
            "encoder": self.name,
            "embedding_dim": self.embedding_dim,
            "input_height": self.input_size[0],
            "input_width": self.input_size[1],
            "checkpoint_path": str(self.checkpoint_path),
            "device": str(self.device),
            "weights_frozen": True,
            "normalization": "L2",
        }
