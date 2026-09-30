"""Frozen feature contract for N72R18.

N72R18 deliberately does not contain an image encoder.  The only accepted
input is the inherited, normalized, 512-D OSNet cache from N72R17.
"""

from __future__ import annotations

import numpy as np
import torch

FEATURE_DIMENSION = 512
FROZEN_ENCODER_NAME = "osnet_x1_0_market1501"


def normalize(value: torch.Tensor | np.ndarray) -> torch.Tensor | np.ndarray:
    """L2-normalize the last dimension without changing the value type."""

    if isinstance(value, torch.Tensor):
        return torch.nn.functional.normalize(value, dim=-1, eps=1e-12)
    array = np.asarray(value, dtype=np.float32)
    norm = np.linalg.norm(array, axis=-1, keepdims=True)
    return array / np.maximum(norm, 1e-12)


def validate_feature_tensor(value: torch.Tensor, name: str = "feature") -> None:
    """Reject accidental use of a non-512-D or invalid feature tensor."""

    if value.ndim < 1 or value.shape[-1] != FEATURE_DIMENSION:
        raise ValueError(f"{name} must end in {FEATURE_DIMENSION} dimensions, got {tuple(value.shape)}")
    if not torch.isfinite(value).all():
        raise ValueError(f"{name} contains non-finite values")


__all__ = ["FEATURE_DIMENSION", "FROZEN_ENCODER_NAME", "normalize", "validate_feature_tensor"]
