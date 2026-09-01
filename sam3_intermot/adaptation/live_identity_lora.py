"""Small identity-scoped Q/K/V LoRA challenger for N28.

The base relation network and every ``A`` factor are frozen.  A corrected
identity owns only its zero-initialised ``B`` factors.  The public score is
always a zero-reference residual, ``h(B) - h(0)``, so a missing or reset live
state is exactly the frozen anchor rather than an approximately equivalent
copy of it.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Iterable, Optional

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F


@dataclass(frozen=True)
class LiveLoRAConfig:
    input_dim: int = 10
    d_model: int = 256
    rank: int = 8
    blocks: int = 2
    alpha: float = 8.0
    seed: int = 28


class _IdentityFastState(nn.Module):
    def __init__(self, blocks: int, d_model: int, rank: int) -> None:
        super().__init__()
        self.factors = nn.ParameterList(
            [
                nn.Parameter(torch.zeros(d_model, rank), requires_grad=True)
                for _ in range(blocks * 3)
            ]
        )

    def is_zero(self) -> bool:
        return all(not bool(torch.count_nonzero(p.detach())) for p in self.factors)

    def snapshot(self) -> tuple[Tensor, ...]:
        return tuple(p.detach().clone() for p in self.factors)

    def restore(self, snapshot: Iterable[Tensor]) -> None:
        with torch.no_grad():
            for parameter, value in zip(self.factors, snapshot):
                parameter.copy_(value)


class LiveIdentityLoRA(nn.Module):
    """Frozen relation backbone plus dynamic identity-specific B factors."""

    def __init__(self, config: Optional[LiveLoRAConfig] = None) -> None:
        super().__init__()
        self.config = config or LiveLoRAConfig()
        if self.config.rank <= 0 or self.config.rank > self.config.d_model:
            raise ValueError("rank must be in [1, d_model]")
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(self.config.seed)
            self.input_projection = nn.Linear(self.config.input_dim, self.config.d_model)
            self.output_projection = nn.Linear(self.config.d_model, 1)
            self.qkv = nn.ModuleList(
                [
                    nn.ModuleList(
                        [
                            nn.Linear(self.config.d_model, self.config.d_model, bias=False)
                            for _ in range(3)
                        ]
                    )
                    for _ in range(self.config.blocks)
                ]
            )
            self.norms = nn.ModuleList(
                [nn.LayerNorm(self.config.d_model) for _ in range(self.config.blocks)]
            )
            self.a_factors = nn.ParameterList(
                [
                    nn.Parameter(
                        torch.empty(self.config.rank, self.config.d_model),
                        requires_grad=False,
                    )
                    for _ in range(self.config.blocks * 3)
                ]
            )
            for factor in self.a_factors:
                nn.init.kaiming_uniform_(factor, a=np.sqrt(5.0))
        self._freeze_backbone()
        self._identity_keys: dict[Any, str] = {}
        self.fast_states = nn.ModuleDict()

    def _freeze_backbone(self) -> None:
        for name, parameter in self.named_parameters():
            if not name.startswith("fast_states."):
                parameter.requires_grad_(False)

    def _key(self, identity_id: Any) -> str:
        if identity_id not in self._identity_keys:
            digest = hashlib.sha1(repr(identity_id).encode("utf-8")).hexdigest()[:16]
            self._identity_keys[identity_id] = f"identity_{digest}"
        return self._identity_keys[identity_id]

    def ensure_identity(self, identity_id: Any) -> _IdentityFastState:
        key = self._key(identity_id)
        if key not in self.fast_states:
            self.fast_states[key] = _IdentityFastState(
                self.config.blocks, self.config.d_model, self.config.rank
            )
        return self.fast_states[key]

    def has_identity(self, identity_id: Any) -> bool:
        return self._key(identity_id) in self.fast_states

    def _project(
        self,
        x: Tensor,
        base: nn.Linear,
        a: Tensor,
        b: Optional[Tensor],
    ) -> Tensor:
        output = F.linear(x, base.weight, base.bias)
        if b is not None:
            low_rank = F.linear(F.linear(x, a), b)
            output = output + (self.config.alpha / self.config.rank) * low_rank
        return output

    def _forward(self, relation: Tensor, state: Optional[_IdentityFastState]) -> Tensor:
        if relation.ndim != 3 or relation.shape[-1] != self.config.input_dim:
            raise ValueError(
                f"expected [batch, candidates, {self.config.input_dim}], got {tuple(relation.shape)}"
            )
        hidden = self.input_projection(relation)
        factor_index = 0
        scale = float(self.config.d_model) ** -0.5
        for block_index in range(self.config.blocks):
            b_factors = None if state is None else state.factors[factor_index : factor_index + 3]
            qkv = []
            for projection_index, projection in enumerate(self.qkv[block_index]):
                b = None if b_factors is None else b_factors[projection_index]
                qkv.append(
                    self._project(
                        hidden,
                        projection,
                        self.a_factors[factor_index + projection_index],
                        b,
                    )
                )
            factor_index += 3
            q, k, value = qkv
            attention = torch.softmax(torch.matmul(q, k.transpose(-1, -2)) * scale, dim=-1)
            hidden = self.norms[block_index](hidden + torch.matmul(attention, value))
        return self.output_projection(hidden).squeeze(-1)

    def delta_tensor(self, relation: Tensor, identity_id: Any) -> Tensor:
        """Return the exact zero-reference residual for one identity batch."""
        state = self.ensure_identity(identity_id)
        # During inference a literal zero is preferable: it makes the
        # no-update invariant byte-exact.  During an online optimization
        # step we must still execute the graph so B receives a gradient at
        # its zero initialization.
        if state.is_zero() and not torch.is_grad_enabled():
            return torch.zeros(
                relation.shape[:-1], dtype=relation.dtype, device=relation.device
            )
        return self._forward(relation, state) - self._forward(relation, None)

    def delta_batch(self, relation: Tensor, identity_ids: list[Any]) -> Tensor:
        """Evaluate a ``[identity, candidate, feature]`` relation batch."""
        if relation.ndim != 3 or len(identity_ids) != relation.shape[0]:
            raise ValueError("relation batch and identity_ids must have matching first dimensions")
        rows = [self.delta_tensor(relation[index : index + 1], identity_id)[0] for index, identity_id in enumerate(identity_ids)]
        return torch.stack(rows) if rows else relation.new_zeros((0, relation.shape[1]))

    def delta_numpy(self, relation: np.ndarray, identity_id: Any) -> np.ndarray:
        tensor = torch.as_tensor(relation, dtype=torch.float32)
        squeeze_batch = tensor.ndim == 2
        if squeeze_batch:
            tensor = tensor.unsqueeze(0)
        with torch.no_grad():
            output = self.delta_tensor(tensor, identity_id).cpu().numpy()
        return output[0] if squeeze_batch else output

    def snapshot(self, identity_ids: Iterable[Any]) -> dict[Any, tuple[Tensor, ...]]:
        return {identity_id: self.ensure_identity(identity_id).snapshot() for identity_id in identity_ids}

    def restore(self, snapshot: dict[Any, tuple[Tensor, ...]]) -> None:
        for identity_id, values in snapshot.items():
            self.ensure_identity(identity_id).restore(values)

    def reset(self, identity_ids: Optional[Iterable[Any]] = None) -> None:
        ids = list(self._identity_keys) if identity_ids is None else list(identity_ids)
        with torch.no_grad():
            for identity_id in ids:
                state = self.ensure_identity(identity_id)
                for parameter in state.factors:
                    parameter.zero_()

    def live_parameters(self, identity_ids: Iterable[Any]) -> list[nn.Parameter]:
        parameters: list[nn.Parameter] = []
        for identity_id in identity_ids:
            parameters.extend(list(self.ensure_identity(identity_id).factors))
        return parameters

    def live_parameter_count(self, identity_id: Any) -> int:
        return sum(parameter.numel() for parameter in self.ensure_identity(identity_id).factors)

    def live_state_norm(self, identity_ids: Iterable[Any]) -> float:
        values = [parameter.detach().float().pow(2).sum() for parameter in self.live_parameters(identity_ids)]
        return float(torch.sqrt(torch.stack(values).sum()).item()) if values else 0.0
