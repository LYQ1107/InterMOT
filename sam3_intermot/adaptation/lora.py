"""Minimal LoRA injection for the frozen SAM3 tracker (adapter level only)."""

import math
from typing import Iterable, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class LoRALinear(nn.Module):
    def __init__(self, linear: nn.Linear, r: int, alpha: float = 1.0):
        super().__init__()
        self.linear = linear
        self.r = r
        self.scale = alpha / max(1, r)
        in_f, out_f = linear.in_features, linear.out_features
        self.lora_a = nn.Parameter(torch.zeros(in_f, r))
        self.lora_b = nn.Parameter(torch.zeros(r, out_f))
        nn.init.kaiming_uniform_(self.lora_a, a=math.sqrt(5))

    @property
    def weight(self):
        # nn.MultiheadAttention reads `self.out_proj.weight` directly.
        return self.linear.weight

    @property
    def bias(self):
        return self.linear.bias

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        a, b = self.lora_a.to(dtype), self.lora_b.to(dtype)
        return self.linear(x) + (x @ a @ b) * self.scale


class LoRAConv2d(nn.Module):
    def __init__(self, conv: nn.Conv2d, r: int, alpha: float = 1.0):
        super().__init__()
        self.conv = conv
        self.r = r
        self.scale = alpha / max(1, r)
        k = int(conv.kernel_size[0])
        self.lora_a = nn.Parameter(
            torch.zeros(r, conv.in_channels * k * k)
        )
        self.lora_b = nn.Parameter(torch.zeros(conv.out_channels, r))
        nn.init.kaiming_uniform_(self.lora_a, a=math.sqrt(5))
        nn.init.zeros_(self.lora_b)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        k = int(self.conv.kernel_size[0])
        delta = (self.lora_b.to(dtype) @ self.lora_a.to(dtype)).view(
            self.conv.out_channels, self.conv.in_channels, k, k
        )
        return self.conv(x) + F.conv2d(
            x, delta, stride=self.conv.stride, padding=self.conv.padding,
            dilation=self.conv.dilation,
        ) * self.scale


def _wrap(module: nn.Module, r: int, alpha: float) -> nn.Module:
    if isinstance(module, nn.Linear):
        return LoRALinear(module, r, alpha)
    if isinstance(module, nn.Conv2d):
        if module.groups != 1 or module.kernel_size[0] != module.kernel_size[1]:
            raise TypeError("unsupported conv shape for LoRA")
        return LoRAConv2d(module, r, alpha)
    raise TypeError(f"unsupported LoRA target: {type(module)}")


def inject_lora(
    model: nn.Module,
    target_prefixes: Iterable[str],
    r: int = 8,
    alpha: float = 1.0,
    skip_biases: bool = True,
) -> Tuple[List[str], List[torch.nn.Parameter]]:
    """Replace linear/conv modules under target_prefixes with LoRA wrappers.

    Base weights are frozen; LoRA A/B params are trainable.  Returns the list of
    modified parameter paths and the trainable LoRA parameters.
    """
    prefixes = tuple(target_prefixes)
    modified: List[str] = []
    params: List[torch.nn.Parameter] = []
    for name, module in list(model.named_modules()):
        if not name.startswith(prefixes):
            continue
        if not isinstance(module, (nn.Linear, nn.Conv2d)):
            continue
        parent_name, _, child_name = name.rpartition(".")
        parent = model if not parent_name else _get_module(model, parent_name)
        try:
            wrapped = _wrap(module, r, alpha)
        except TypeError:
            continue
        wrapped = wrapped.to(device=module.weight.device)
        setattr(parent, child_name, wrapped)
        for p in module.parameters():
            p.requires_grad = False
        if not skip_biases and hasattr(module, "bias") and module.bias is not None:
            module.bias.requires_grad = False
        modified.append(name)
        params.extend(p for p in wrapped.parameters() if p.requires_grad)
    return modified, params


def _get_module(model: nn.Module, path: str) -> nn.Module:
    node = model
    for part in path.split("."):
        node = getattr(node, part)
    return node


def lora_parameter_count(params: Iterable[torch.nn.Parameter]) -> int:
    return sum(int(p.numel()) for p in params)
